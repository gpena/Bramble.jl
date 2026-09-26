# bilinear.jl: BilinearForm's struct, construction, mesh-compatibility checks shared by
# every downstream assembly path, and the public assemble/assemble!/assemble_parallel!
# dispatch. The shared traversal and its sinks live in `bilinear_traversal.jl`; sparsity
# pattern discovery in `bilinear_pattern.jl`; the serial (record/replay) and threaded
# (band-coloured) execution strategies in `bilinear_execution.jl`.

# --- Struct definitions ----------------------------------------------------------- #

# Every scattered contribution's nzval position, for one term routed into one block: the
# same shape `add_to_sparse!` used to re-derive by searching on every call (gpena/Bramble.jl#26).
# `point_ptr[lin_idx]:point_ptr[lin_idx + 1] - 1` is the slice of `positions` holding the
# positions for grid point `lin_idx`'s own (guard-passing, non-deduplicated) stencil entries,
# in the same order a scatter walk visits them -- addressed per point rather than by a shared
# running counter, so the threaded refill reads it from every band at once without a race
# (gpena/Bramble.jl#338).
"""
    NzvalSegment = Tuple{Vector{Int},Vector{Int}}

One term's recorded nzval positions for one block, as `(point_ptr, positions)`.

`point_ptr[lin_idx]:point_ptr[lin_idx + 1] - 1` is the slice of `positions` holding grid
point `lin_idx`'s own entries, in the order a scatter walk visits them. Addressed per point
rather than by a shared running counter, so a replay stays correct whatever order the grid
is visited in.

See also: [`ReplaySink`](@ref).
"""
const NzvalSegment = Tuple{Vector{Int}, Vector{Int}}

"""
    Segment{D}

One term's recorded nzval positions for one block, on a `D`-dimensional structured grid.
Replaces a former `Union{NzvalSegment,DiagonalSegment{D}}` (`AnySegment{D}`) with a single
concrete struct (gpena/Bramble.jl#240): `Enzyme`'s strict-aliasing type analysis rejects a
`Union` return outright (`IllegalTypeAnalysisException`), regardless of whether the segment
itself is ever the target of differentiation, so `_try_diagonal_segment` needed a return type
Enzyme can reason about -- a discriminated struct, not a two-branch `Union`.

`is_diagonal` selects which shape is populated:
- `false` (flat): `point_ptr`/`positions` cover every entry, exactly as [`NzvalSegment`](@ref)
  always has; `base`/`stride`/`interior` are unused placeholders (empty, zero).
- `true` (diagonal, gpena/Bramble.jl#160): `point_ptr`/`positions` cover only the boundary
  shell; `base`/`stride`/`P`/`interior` carry the interior's per-tap stride arithmetic --
  interior entries are `base[k] + stride[k] * n` for the `n`-th point `interior`'s own
  iteration order visits (`n` zero-based), rather than one stored `Int` per entry, so
  `positions` never carries the interior's `O(N * P)` share at all. 1D forms only
  (`_diagonal_replay`): from 2D up no difference term's interior has a constant stride.

Built by `_try_diagonal_segment` only when every interior point produces the same number of
entries `P` and the same per-tap stride holds across the whole interior -- checked once, not
assumed, because a form summing terms of different margins can make a column's true `nzval`
footprint vary inside what this one term calls its own interior (see [`_stencil_margin`](@ref)).
Any point where the check fails falls back to the flat shape instead, the general case
[`ReplaySink`](@ref) already handles.

Parametrized by `D` alone -- `interior`'s ranges-tuple type is pinned to
`NTuple{D,UnitRange{Int}}` (what `_interior_range` always produces), never left as an
independent free parameter -- so that for one `BilinearForm`'s fixed dimension, `Segment{D}`
is concrete. A `CartesianIndices{D}` alone, or a `Segment` with `D` left to vary, is not
concrete (its ranges type is still a `UnionAll`) and stores boxed: exactly what made an early
version of this allocate 80-400 B on every replay, `@test_allocs`-checked paths included.

`positions_t` is empty except on a segment recorded for a transposed pair
⟨Au, Bv⟩ + ⟨Bu, Av⟩ (`_segments_from_positions`): there it holds, entry for entry, the `nzval`
position of the transposed entry the second term writes, so one walk of ⟨Au, Bv⟩ fills both.
The half a pair's unit does not write (`_PairReplaySink`) leaves `positions` or
`positions_t` empty. A pair segment is always flat.

See also: [`DiagonalReplaySink`](@ref).
"""
struct Segment{D}
    is_diagonal::Bool
    point_ptr::Vector{Int}
    positions::Vector{Int}
    base::Vector{Int}
    stride::Vector{Int}
    P::Int
    interior::CartesianIndices{D, NTuple{D, UnitRange{Int}}}
    positions_t::Vector{Int}
end

# A concrete, zero-length placeholder for `interior` on the flat path, where nothing reads
# it -- still has to be a real `CartesianIndices{D,NTuple{D,UnitRange{Int}}}` value, since
# `Segment{D}` has no field left optional (that is the entire point: one concrete shape,
# not a Union of two).
_empty_interior(::Val{D}) where {D} = CartesianIndices(ntuple(_ -> 1:0, D))

# The flat shape: built at every early return in `_try_diagonal_segment` (bilinear_execution.jl)
# where the interior/boundary split does not hold or is not worth it.
function _flat_segment(::Val{D}, point_ptr::Vector{Int}, positions::Vector{Int}) where {D}
    Segment{D}(false, point_ptr, positions, Int[], Int[], 0, _empty_interior(Val(D)), Int[])
end

# One `BilinearForm`'s nzval-position cache: valid only for the exact matrix object last
# assembled into, one `Segment{D}` per (term, block) the serial assembly walk visits, in
# visitation order. A companion *value*, not a type parameter of `BilinearForm` -- so it can
# be filled in lazily, on the first `assemble!` call, without the form itself needing to be
# mutable or its type to depend on whether a cache exists yet. `D` itself, though, is
# threaded in at construction (matching `BilinearForm`'s own `D`): without it, `segments`'s
# eltype would be the unparametrized (non-concrete) `Segment`, and every push/read would box
# (see [`Segment`](@ref)).
#
# `A_id::UInt` (`objectid(A)`), not `A::SparseMatrixCSC` compared with `===`: the field
# needs to be concrete for the same reason `Segment` does (gpena/Bramble.jl#240 -- one level
# up from the `Segment`/`AnySegment{D}` union that issue originally reported, surviving that
# fix because it sits here rather than in `_try_diagonal_segment`, and only showing up once
# a differentiated closure builds the `BilinearForm` itself, putting the cache on Enzyme's
# active graph). A bare `SparseMatrixCSC` field is already a `UnionAll` (unparametrized in
# `Tv`/`Ti`), and different backends genuinely use different matrix types (`matrix_type`,
# `src/utils/backend.jl`), so pinning the field to one concrete `SparseMatrixCSC{Tv,Ti}`
# would either be wrong for another backend or force threading a matrix-type parameter
# through `BilinearForm` itself. `objectid` needs none of that: identity comparison is all
# the cache ever wanted, `UInt` is concrete regardless of backend, and `UInt(0)` never
# collides with a real object's id under `valid = false`.
#
# `AST` is threaded in for the same reason `D` is, one paragraph up: the field has to be
# concrete or every cache hit loads it as `Any`. Leaving it untyped was measurable --
# `Base.getproperty(cache, :ast)::ANY` in `_assemble_bilinear_core_cached!`'s IR -- but only
# for an AST that *carries data* (a grid-function coefficient, a source array), since a
# singleton AST makes the whole `===` comparison a compile-time constant and the load folds
# away. Data-carrying is exactly the case a differentiated closure builds, so the untyped
# field cost nothing on the forms that did not need it and put boxed, untyped heap data on
# Enzyme's tape for the ones that did.
#
# The parameter costs nothing to supply: `BilinearForm` already carries `AST`, and `form`
# builds the cache where `typeof(ast)` is known. It also sharpens the guard -- a foreign AST
# of a different type now folds `cache.ast === ast` to `false` at compile time instead of
# comparing at run time.
mutable struct _AssemblyCache{D, AST}
    valid::Bool
    A_id::UInt
    ast::AST
    segments::Vector{Segment{D}}
end

# One shared, never-written empty `segments` vector per dimension, so a fresh form's cache
# costs one allocation -- the mutable struct itself -- rather than two. Sharing is safe
# because nothing ever writes *through* this reference: a cache miss in
# `_assemble_bilinear_core_cached!` (form/bilinear_execution.jl) builds its own vector and
# assigns it, deliberately never `push!`ing into or `empty!`ing whatever `segments` points at.
#
# One constant per `D` rather than one shared `Segment[]`, because the eltype is
# dimension-parametric (gpena/Bramble.jl#161): an unparametrized `Segment` is not concrete,
# and storing into such a vector boxes every element (see [`Segment`](@ref)). Only the
# dimensions this package meshes get a constant; any other `D` falls back to allocating, which
# is what every `D` did before this.
const _NO_SEGMENTS_1 = Segment{1}[]
const _NO_SEGMENTS_2 = Segment{2}[]
const _NO_SEGMENTS_3 = Segment{3}[]

_no_segments(::Val{D}) where {D} = Segment{D}[]
_no_segments(::Val{1}) = _NO_SEGMENTS_1
_no_segments(::Val{2}) = _NO_SEGMENTS_2
_no_segments(::Val{3}) = _NO_SEGMENTS_3

"""
    _zero_stored!(A::AbstractMatrix) -> AbstractMatrix

Zero every value a refill will touch, before `_assemble_bilinear_core_cached!`/
`_assemble_bilinear_parallel_core!` add each term's contribution back in.

Part of the matrix-type seam (S1.1, gpena/Bramble.jl#12): `SparseMatrixCSC` only has to zero
its stored `nzval` entries (the sparsity pattern itself never changes between refills), while
a dense `Matrix` -- the seam's positive control -- has no such distinction and zeros the
whole backing array. A future backend (tridiagonal, banded, ...) implements whichever of the
two shapes its own storage has.
"""
@inline function _zero_stored!(A::SparseMatrixCSC)
    fill!(nonzeros(A), zero(eltype(A)))
    return A
end
@inline function _zero_stored!(A::AbstractMatrix)
    fill!(A, zero(eltype(A)))
    return A
end

# Deliberately policy-blind, and measured rather than assumed. Zeroing is the one serial
# stretch of an otherwise threaded refill, so it looks like the obvious candidate for a
# policy-dispatched sweep: it is 4.3% of `assemble_parallel!` at 1024^2 (0.40 ms of 9.48 ms,
# 5,238,784 stored entries), which by Amdahl's law alone would cap a four-thread speedup at
# 3.55x, close to the 3.47x actually measured. That reasoning is wrong, because the 4.3% is
# not compute. Writing 40 MB of zeros saturates memory bandwidth on one core, so splitting
# the write gains nothing: measured on the reference M2 at 1024^2, `fill!` takes 0.492 ms
# serially, 0.495 ms under `Threads.@threads :static` (0.99x) and 0.633 ms under
# `Polyester.@batch` (0.78x -- per-call overhead with no compute to hide behind). Even a free
# zeroing would take the sweep from 9.84 ms to about 8.5 ms and leave it behind `assemble!`'s
# cached replay at 6.9 ms. Dispatching this on the execution policy would also change a
# `public` seam method (gpena/Bramble.jl#12) that out-of-tree backends implement, for a gain
# that measures as zero. `benchmark/polyester_crossover.jl` carries the surrounding numbers.

# Takes the AST the form was built from, so the cache's own `AST` parameter comes from the
# same expression tree `BilinearForm` stores -- never `nothing`, which would have made the
# parameter a lie on the first cache miss.
function _AssemblyCache{D}(ast::AST) where {D, AST}
    return _AssemblyCache{D, AST}(false, UInt(0), ast, _no_segments(Val(D)))
end

"""
    BilinearForm{D, TrialSpace, TestSpace, AST}

Represents a bilinear form defined over a trial space and test space.

# Arguments
- `trial_space::TrialSpace`: Space for the trial function.
- `test_space::TestSpace`: Space for the test function.
- `ast::AST`: Resolved expression tree.

The form resolves its expression tree `ast` once at construction, referencing the underlying
storage of any coefficient grid functions (`VectorElement`). In-place updates via `Rₕ!(cₕ, ...)`
or `parent(cₕ) .= ...` are automatically seen by subsequent assemblies with zero heap allocations.
The expression itself is not kept: downstream routines evaluate the resolved AST directly.

Constant scalar coefficients can be written directly as numbers (e.g. `2.0 * innerₕ(D₋ₓ(u), D₋ₓ(v))`).
`Ref` is only needed if a dynamic scalar coefficient changes across loop iterations:
```julia
using Bramble: D₋ₓ
β = Ref(1.0)
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(β * D₋ₓ(u), D₋ₓ(v)))
# Inside time loop:
β[] = 3.0
assemble!(A, a) # zero allocations, evaluates with β = 3.0
```
"""
struct BilinearForm{D, TrialSpace, TestSpace, AST}
    trial_space::TrialSpace
    test_space::TestSpace
    ast::AST
    cache::_AssemblyCache{D, AST}
end

"""
    trial_space(form::BilinearForm)

Return the trial space of the bilinear form.
"""
trial_space(form::BilinearForm) = form.trial_space

"""
    test_space(form::BilinearForm)

Return the test space of the bilinear form.
"""
test_space(form::BilinearForm) = form.test_space

# `a(uₕ, vₕ) = vᵀ A u`. Assembles a whole matrix per call: intended for testing/convenience.
# Multiplies by `parent(u)`, the raw vector: on Julia 1.14 SparseArrays reads `u[k, 1]`, which a
# 2D `VectorElement` answers as grid point `(k, 1)` rather than entry `k` (BoundsError in the
# nightly precompile run of 1e0352c3). `parent` of a plain vector is the vector itself.
@inline (form::BilinearForm)(u, v) = dot(v, assemble(form) * parent(u))

"""
    resolve_form_ast(form::BilinearForm)

Return the resolved AST stored inside the bilinear form.
"""
@inline resolve_form_ast(form::BilinearForm) = form.ast

"""
    form(Wₕ, Vₕ, f) -> BilinearForm

Construct a `BilinearForm` over the trial space `Wₕ` and the test space `Vₕ` from the
bilinear expression `f` (a function of trial and test arguments `(u, v)`). The AST is
resolved once and run through [`simplify_ast`](@ref) -- factoring common scalings and
shared inner-product arguments, combining like terms, and eliding zero-scaled ones -- before
it is stored.

# Examples
```julia
using Bramble: D₋ₓ, inner₊ₓ
# a(u, v) = (∇ₕu, ∇ₕv)₊
a = form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))

# a coupled system, one term per block
a = form(Vₕ, Vₕ, (u, v) -> inner₊ₓ(D₋ₓ(u(1)), D₋ₓ(v(1))) + innerₕ(u(2), v(1)))
```
"""
function form(Wₕ, Vₕ, f)
    D = dim(Wₕ)
    raw_ast = f(trial_function(Wₕ), test_function(Vₕ))
    _validate_form_expression(raw_ast, Val(D))
    ast = simplify_ast(resolve_ast(raw_ast))
    return BilinearForm{D, typeof(Wₕ), typeof(Vₕ), typeof(ast)}(
        Wₕ, Vₕ, ast, _AssemblyCache{D}(ast)
    )
end

# --- Mesh compatibility checks ------------------------------------------------------ #
#
# Shared by pattern discovery (`bilinear_pattern.jl`) and both execution strategies
# (`bilinear_execution.jl`): every path that walks a term over a block needs the same
# guard, so it is stated once, here, rather than once per caller.

# Refuse cross-mesh coupling unless an explicit mapping (such as interpolation) is provided.
@noinline function _throw_cross_mesh_block(@nospecialize(term), Ωu, Ωv)
    throw(
        ArgumentError(
        "a bilinear term coupling two leaves over different meshes has no assembly: the " *
        "trial leaf has $(npoints(Ωu, Tuple)) points and the test leaf $(npoints(Ωv, Tuple)), " *
        "so an index on one names no point on the other. Got $(typeof(term)). Couple leaves " *
        "that share a mesh, or wrap the trial or the test function in an interpolation " *
        "operator: `πₕ(u)`, `πₕ(v)`.",
    ),
    )
end

@inline function _check_block_meshes(term, trial_leaf, test_leaf)
    # Either side interpolating is a mapping between the two index spaces, so the leaves need
    # not share one: the trial side names columns on its own mesh, the test side rows on its
    # own (gpena/Bramble.jl#263).
    _all_trial_interpolated(term) && return nothing
    _all_test_interpolated(term) && return nothing

    Ωu = mesh(trial_leaf)
    Ωv = mesh(test_leaf)
    npoints(Ωu, Tuple) == npoints(Ωv, Tuple) || _throw_cross_mesh_block(term, Ωu, Ωv)
    return nothing
end

@inline _check_block_meshes(op::OperatorAdd, trial_leaf, test_leaf) = _visit_operator_add1(_check_block_meshes, op, trial_leaf, test_leaf)

# --- Assembly implementations ----------------------------------------------------- #

function apply_dirichlet_labels!(
        A::AbstractMatrix, form::BilinearForm, dirichlet_labels, dirichlet_components = nothing
)
    if dirichlet_labels !== nothing
        if dirichlet_labels isa Symbol
            dirichlet_bc!(
                A, test_space(form), dirichlet_labels; components = dirichlet_components
            )
        elseif dirichlet_labels isa Tuple
            if !isempty(dirichlet_labels)
                dirichlet_bc!(
                    A,
                    test_space(form),
                    dirichlet_labels...;
                    components = dirichlet_components
                )
            end
        end
    end
end

"""
    assemble(form::BilinearForm; dirichlet = nothing, dirichlet_components = nothing) -> AbstractMatrix

Allocate a matrix with the form's sparsity pattern and assemble into it. The matrix type is
`matrix_type(backend(test_space(form)))` -- `SparseMatrixCSC{Float64,Int}` by default, or
whatever [`backend`](@ref) the space's mesh was built with.

**Call this once, then assemble into what it returns.** Building the sparsity pattern is the
larger part of the work (at 250,000 degrees of freedom it is 9,700 us and 52 MB against 1,500 us
and zero allocations to refill the matrix), and the pattern does not change between assemblies.
A time loop or Newton iteration benefits from preallocating the pattern once:

```julia
A = assemble(a)                        # once: pattern, allocation and initial fill
for step in 1:nsteps
    Rₕ!(cₕ, coefficient_at(step))      # modified in-place
    assemble!(A, a)                    # or assemble_parallel!(A, a)
end
```

Runs serially or across threads following `form.trial_space`'s backend
[`execution_policy`](@ref): [`Serial`](@ref) by default. Optional `dirichlet` applies
boundary conditions to the matrix -- a label `Symbol`, a `Tuple` of labels, a `label => f`
`Pair` (values ignored on the matrix side), or constraints from
[`dirichlet_constraints`](@ref); see [`_normalize_dirichlet`](@ref) for every accepted
form. `dirichlet_components` restricts which leaf components of a composite trial space
they bind to (see [`dirichlet_bc!`](@ref)).
"""
function assemble(form::BilinearForm; dirichlet = nothing, dirichlet_components = nothing)
    A = allocate_system_matrix(form, form.ast)
    _assemble_bilinear!(A, form, form.ast, dirichlet, dirichlet_components)
    return A
end

"""
    assemble!(A::AbstractMatrix, form::BilinearForm; dirichlet = nothing, dirichlet_components = nothing) -> AbstractMatrix

Assemble the `BilinearForm` into the preallocated matrix `A`, allocating nothing (**0 bytes**)
when `A` is the default `SparseMatrixCSC`.

Runs serially or across threads following `form.trial_space`'s backend
[`execution_policy`](@ref): [`Serial`](@ref) (the default) or [`Parallel`](@ref).
[`assemble_parallel!`](@ref) is a separate, lower-level entry point that always threads,
ignoring the backend's policy.

`assemble!` uses the pre-resolved `form.ast` stored directly inside the form.

## Live coefficients
- Grid functions: the stored AST retains references to source `VectorElement` storage. Mutating values in-place (`Rₕ!(cₕ, ...)` or `parent(cₕ) .= ...`) between steps automatically updates the matrix entries with 0 allocations.
- Dynamic scalars: plain numbers work directly for constant scalars. To update a scalar dynamically across loop iterations, wrap it in a `Ref(val)` (e.g. `β = Ref(1.0); a = form(Wₕ, Wₕ, (u, v) -> innerₕ(β * D₋ₓ(u), D₋ₓ(v)))`). Mutating `β[] = new_val` evaluates live during assembly with 0 allocations.
"""
function assemble!(
        A::AbstractMatrix,
        form::BilinearForm;
        dirichlet = nothing,
        dirichlet_components = nothing,
        ast = nothing
)
    resolved_ast = ast === nothing ? form.ast : (_warn_ast_keyword(:assemble!); ast)
    return _assemble_bilinear!(A, form, resolved_ast, dirichlet, dirichlet_components)
end

# The shared core behind `assemble!` and `assemble`: takes its `ast` positionally, already
# resolved and already past the deprecation check, so neither public entry point warns twice
# calling into the other.
function _assemble_bilinear!(
        A::AbstractMatrix, form::BilinearForm, ast, dirichlet, dirichlet_components
)
    dirichlet_labels, _ = _normalize_dirichlet(dirichlet)
    _zero_stored!(A)

    if execution_policy(form.trial_space) isa CpuSerial
        _assemble_bilinear_core_cached!(
            A, form.trial_space, form.test_space, ast, form.cache
        )
    else
        _assemble_bilinear_parallel_cached!(
            A, form.trial_space, form.test_space, ast, form.cache
        )
    end

    apply_dirichlet_labels!(A, form, dirichlet_labels, dirichlet_components)
    return A
end

"""
    assemble_parallel!(A::AbstractMatrix, form::BilinearForm) -> AbstractMatrix

Refill `A` with the assembled `form` across threads and return it, regardless of
`form.trial_space`'s backend policy. `A` must already carry the correct sparsity pattern from
[`allocate_system_matrix`](@ref) or a previous [`assemble`](@ref). Unlike `assemble!`, does
not apply `dirichlet_labels`.

Colouring on the test side ensures thread safety when updating stored matrix values concurrently.
The band-coloured sweep runs for every matrix type: it reaches storage only through the
matrix-type seam (`_scatter_position`/`_scatter_add!`), so a dense, banded or CSR backend
colours exactly as `SparseMatrixCSC` does (gpena/Bramble.jl#12, #190).

Shares [`assemble!`](@ref)'s record/replay cache (gpena/Bramble.jl#338): once the form has
recorded where each entry of `A` lives -- `A` came from [`assemble`](@ref) or
[`allocate_system_matrix`](@ref) on a serial form, or an earlier fill into this same `A`
recorded it -- every sweep writes through those positions instead of searching for them.
The first fill into any other matrix object records once, serially, then replays across
threads. A device-resident matrix still searches.
"""
function assemble_parallel!(A::AbstractMatrix, form::BilinearForm, ast = nothing)
    resolved_ast = ast === nothing ? form.ast : (_warn_ast_keyword(:assemble_parallel!); ast)
    _zero_stored!(A)

    _assemble_bilinear_parallel_cached!(
        A, form.trial_space, form.test_space, resolved_ast, form.cache
    )

    return A
end

"""
    assemble(a::BilinearForm, l::LinearForm; dirichlet = nothing, dirichlet_components = nothing, symmetrize::Bool = false) -> (A, F)

Assemble both the matrix and the vector in one call, applying the same `dirichlet`
constraints to each -- the common case of building a system to solve `A \\ F` against.
`a` and `l` should share the same test space; boundary values are read from `dirichlet`
once rather than once per form.

`symmetrize = true` additionally restores symmetry in `A` after `dirichlet_bc!` (which
only clears rows, not the columns) and updates `F` to match, via [`symmetrize!`](@ref).
Requires `dirichlet` to name at least one label -- there is nothing to symmetrize against
otherwise.

```julia
A, F = assemble(a, l; dirichlet = :boundary => sol, symmetrize = true)
u = A \\ F
```

is the one-call equivalent of assembling `a` and `l` separately and calling
[`symmetrize!`](@ref) by hand.
"""
function assemble(
        a::BilinearForm,
        l::LinearForm;
        dirichlet = nothing,
        dirichlet_components = nothing,
        symmetrize::Bool = false
)
    A = assemble(a; dirichlet = dirichlet, dirichlet_components = dirichlet_components)
    F = assemble(l; dirichlet = dirichlet, dirichlet_components = dirichlet_components)

    if symmetrize
        dirichlet_labels, _ = _normalize_dirichlet(dirichlet)
        dirichlet_labels === nothing && _throw_symmetrize_without_dirichlet()
        symmetrize!(
            A, F, test_space(a), dirichlet_labels...; components = dirichlet_components
        )
    end

    return A, F
end

@noinline function _throw_symmetrize_without_dirichlet()
    throw(
        ArgumentError(
        "symmetrize = true has nothing to symmetrize against without dirichlet naming " *
        "at least one label.",
    ),
    )
end
