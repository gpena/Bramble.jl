# bilinear.jl: BilinearForm's struct, construction, mesh-compatibility checks shared by
# every downstream assembly path, and the public assemble/assemble!/assemble_parallel!
# dispatch. The shared traversal and its sinks live in `bilinear_traversal.jl`; sparsity
# pattern discovery in `bilinear_pattern.jl`; the serial (record/replay) and threaded
# (band-coloured) execution strategies in `bilinear_execution.jl`.

# --- Struct definitions ----------------------------------------------------------- #

"""
    Segment{D}

One term's recorded nzval positions for one block, on a `D`-dimensional structured grid.
A single concrete struct, with `is_diagonal` discriminating the two shapes (gpena/Bramble.jl#240):
`Enzyme`'s strict-aliasing type analysis rejects a `Union` return outright
(`IllegalTypeAnalysisException`), regardless of whether the segment itself is ever the target
of differentiation, so `_try_diagonal_segment` returns a type Enzyme can reason about, never
a two-branch `Union`.

`is_diagonal` selects which shape is populated:
- `false` (flat). `point_ptr`/`positions` cover every entry; `base`/`stride`/`interior` are
  unused placeholders (empty, zero).
- `true` (diagonal, gpena/Bramble.jl#160). `point_ptr`/`positions` cover only the boundary
  shell; `base`/`stride`/`P`/`interior` carry the interior's per-tap stride arithmetic --
  interior entries are `base[k] + stride[k] * n` for the `n`-th point `interior`'s own
  iteration order visits (`n` zero-based), rather than one stored `Int` per entry, so
  `positions` never carries the interior's `O(N * P)` share at all. 1D forms only
  (`_diagonal_replay`), since from 2D up no difference term's interior has a constant stride.

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
#
# `stamp` is the recording's `_pattern_stamp`: `markers!` redraws a mesh's marker stamp, so
# a recording made before it no longer matches and the next fill re-records
# (gpena/Bramble.jl#465). A restricted term's walk reads the markers, so the recorded
# positions depend on them as much as on `A`.
#
# `stored` is `A`'s stored-entry count at the recording (`_pattern_length`): a
# `dirichlet_bc!` that inserts a missing diagonal shifts every later `nzval` position while
# `A` keeps its identity, so a grown count re-records (gpena/Bramble.jl#534).
mutable struct _AssemblyCache{D, AST}
    valid::Bool
    A_id::UInt
    ast::AST
    segments::Vector{Segment{D}}
    stamp::UInt64
    stored::Int
end

# The stored-entry count a recording's positions index into. A `SparseMatrixCSC` can grow
# (`dirichlet_bc!` inserts a missing diagonal); any other matrix keeps `length(A)`, which
# never changes. No `AbstractSparseMatrix` method: the CSR extension never inserts, it
# refuses a missing diagonal instead.
@inline _pattern_length(A::SparseMatrixCSC) = length(nonzeros(A))
@inline _pattern_length(A::AbstractMatrix) = length(A)

# The newest marker stamp among `Wₕ`'s leaf meshes. The stamps come from one global counter,
# so a `markers!` on any leaf mesh makes it strictly larger, never equal to an older value.
@inline _marker_stamp(Wₕ::AbstractSpaceType) = maximum(
    map(((s, _),) -> _marker_stamp(mesh(s)), leaf_spaces_offsets(Wₕ)))

# The stamp a recording over `trial` and `test` is valid for: any `markers!` on either
# side's meshes changes it.
@inline _pattern_stamp(trial, test) = max(_marker_stamp(trial), _marker_stamp(test))

# One shared, never-written empty `segments` vector per dimension, so a fresh form's cache
# costs one allocation -- the mutable struct itself -- rather than two. Sharing is safe
# because nothing ever writes *through* this reference: a cache miss in
# `_assemble_bilinear_core_cached!` (`bilinear_execution.jl`) builds its own vector and
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
`_assemble_bilinear_parallel_cached!` add each term's contribution back in.

Part of the matrix-type seam (gpena/Bramble.jl#12): `SparseMatrixCSC` only has to zero
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
    return _AssemblyCache{D, AST}(false, UInt(0), ast, _no_segments(Val(D)), UInt64(0), 0)
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

"""
    (a::BilinearForm)(u, v) -> Number

The form evaluated at a trial function `u` and a test function `v`: `vᴴ A u` for
`A = assemble(a)`, without `A`. One walk of the form's stencil, the one
[`matrix_free_operator`](@ref)'s serial `mul!` makes, adds `conj(v[row]) * weight * u[col]`
for every entry into a single scalar. No Dirichlet rows are applied, as `assemble(a)`
applies none.

`u` and `v` are [`VectorElement`](@ref)s or plain vectors, of lengths `ndofs(trial_space(a))`
and `ndofs(test_space(a))`. The result's type is promoted from the element types of `A`, `u`
and `v`, so a `ForwardDiff.Dual` in either argument comes through.

On a CPU backend the call allocates only the one cell its sum runs in (a
`ContractionSink`'s), whatever the grid size. A product with three or more
region-restricted summands is split by `form` into products of at most two
(gpena/Bramble.jl#727); one it leaves whole, as it would split into more than 16 or its
constant scales keep several types, allocates per point, as in `assemble!`.
Each call has its own cell, so tasks may call one form at once. On a GPU backend the matrix
is assembled and multiplied, as the walk reads its vectors on the host.

# Throws
- `DimensionMismatch`: `u` or `v` has the wrong length.
- `ArgumentError`: `u` or `v` is not indexed from 1 (an offset-axes view, say).

# Examples
```jldoctest
using Bramble, LinearAlgebra
Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 11, false))
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
uₕ = Rₕ(Wₕ, x -> sin(x[1]))
vₕ = Rₕ(Wₕ, x -> x[1] + 1)
a(uₕ, vₕ) ≈ dot(parent(vₕ), assemble(a) * parent(uₕ))

# output
true
```
"""
function (form::BilinearForm)(u, v)
    execution_policy(trial_space(form)) isa GpuPolicy &&
        return dot(v, assemble(form) * _mf_data(u))
    return _contract(form, _mf_data(u), _mf_data(v))
end

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
    _validate_form_expression(raw_ast, Val(D), Val(:bilinear))
    ast = _split_restricted_products(
        simplify_ast(_resolve_surfaces_for_space(resolve_ast(raw_ast), Vₕ, Wₕ)),
        promote_type(eltype(Wₕ), eltype(Vₕ)))
    _check_space_components(ast, Wₕ, trial_component_or_nothing, "trial")
    _check_space_components(ast, Vₕ, test_component_or_nothing, "test")
    _check_centered_meshes(ast, Wₕ, Vₕ)  # linear.jl (gpena/Bramble.jl#629)
    _check_source_meshes(ast, Wₕ, Vₕ)    # linear.jl (gpena/Bramble.jl#685)
    return BilinearForm{D, typeof(Wₕ), typeof(Vₕ), typeof(ast)}(
        Wₕ, Vₕ, ast, _AssemblyCache{D}(ast)
    )
end

# A non-composite side has one component, so a term re-indexing its leaf past it
# (`u(1)(2)`) is refused here, at `form` (gpena/Bramble.jl#638). A composite side is
# checked at assembly, by `block_of`, as before.
_check_space_components(ast, space, side_component, side) = _check_scalar_components(
    ast, side_component, side)
_check_space_components(ast, ::CompositeGridSpace, side_component, side) = nothing

# --- At most two restricted summands per product (gpena/Bramble.jl#727) ------------- #
#
# An unprobed `RegionRestriction` answers `()` outside its region and its operand's stencil
# inside, so a summand reaching one has two stencil shapes, and a product whose operands
# hold `k` such summands has `2^k`. The walk splits that `Union` for `k <= 2`. Past that
# every grid point boxes (30528 B on a 7x9 grid for `innerₕ(u, v + Rₕ(:l, v) + Rₕ(:r, v) +
# Rₕ(:b, v))`, what `simplify_ast` folds four terms into). So such a product is split back
# here into products of at most two each, one walk unit apiece (`_foreach_unit`,
# bilinear_execution.jl). Their entries land where the one product's did and the sparsity
# pattern is the same; each extra product costs one walk. Zero entries padded into the
# restriction would also keep one shape, but the coordinate walk puts them in the pattern.
#
# Every method first asks whether `op`'s type holds a `RegionRestriction` at all
# (`_holds_restriction`, read off the type once), so a form without one is returned as it
# was built and compiles nothing past that question.
_split_restricted_products(op, ::Type) = op

function _split_restricted_products(op::OperatorAdd, ::Type{E}) where {E}
    _holds_restriction(op) || return op
    l = _split_restricted_products(op.left_op, E)
    r = _split_restricted_products(op.right_op, E)
    return l === op.left_op && r === op.right_op ? op : OperatorAdd(l, r)
end

# A scale over a split product is distributed over the products, so each stays its own
# unit. A scale inside a product's operand is never distributed.
function _split_restricted_products(
        op::Union{OperatorScale, GridFunctionScale}, ::Type{E}) where {E}
    _holds_restriction(op) || return op
    inner = _split_restricted_products(op.inner_op, E)
    return inner === op.inner_op ? op : _distribute_scale(op, inner)
end
_distribute_scale(op, inner) = _rewrap_inner(op, inner)
function _distribute_scale(op, inner::OperatorAdd)
    OperatorAdd(
        _distribute_scale(op, inner.left_op), _distribute_scale(op, inner.right_op))
end

# Both sides' shapes multiply, so with restricted summands on both sides each piece takes
# at most one from each; with them on one side only, at most two from it. The counts are
# read off the operand types, so a product with at most two folds to its early return.
function _split_restricted_products(op::BilinearProduct{D, W}, ::Type{E}) where {D, W, E}
    _holds_restriction(op) || return op
    l = _uniform_scales(op.left_op, E)
    r = _uniform_scales(op.right_op, E)
    nl = _vanishing_count(l)
    nr = _vanishing_count(r)
    capl = Val(nr == 0 ? 2 : 1)
    capr = Val(nl == 0 ? 2 : 1)
    ls = _operand_groups(_expanded_summands(l), capl, Val(true))
    rs = _operand_groups(_expanded_summands(r), capr, Val(true))
    # A plain summand's own group (`_joins_restriction`) is one more per side; where that
    # takes the product past the cap below, the plain summands join the first group as
    # they did before it, rather than the product being left whole (1205744 B against
    # 82848 B for a plain and four restricted differences against three).
    length(ls) * length(rs) > 16 || return _split_product(op, l, r, ls, rs)
    return _split_product(op, l, r, _operand_groups(_expanded_summands(l), capl, Val(false)),
        _operand_groups(_expanded_summands(r), capr, Val(false)))
end

function _split_product(op::BilinearProduct{D, W}, l, r, ls, rs) where {D, W}
    # Each product is a walk of its own and a type to compile, and restrictions on both
    # sides multiply them: past 16 (six distinct types a side made 36, a first assemble
    # up to 15 times slower and more bytes than the unsplit product), it stays as built.
    length(ls) * length(rs) > 16 && return op
    if (length(ls) == 1 && length(rs) == 1) || _mixed_scales(l) || _mixed_scales(r)
        return l === op.left_op && r === op.right_op ? op :
               BilinearProduct{D, W, typeof(l), typeof(r)}(l, r)
    end
    return _product_sum(op, ls, rs)
end

# Whether a `RegionRestriction` appears anywhere in `T`, its parameters searched in turn.
# Defined before the generator that calls it, which runs in the world it was defined in.
function _type_holds_restriction(T)
    T isa DataType || return false
    T <: RegionRestriction && return true
    for p in T.parameters
        p isa Type && _type_holds_restriction(p) && return true
    end
    return false
end
precompile(_type_holds_restriction, (Any,))
@generated _holds_restriction(::T) where {T} = _type_holds_restriction(T)

# A summed operand whose summands' constant scales differ in type (`2.0 * Rₕ(:l, v) +
# 3 * Rₕ(:r, v)`) has a stencil whose weights differ in type from summand to summand, and
# the walk boxes it at every point even with two restricted summands. Those scales, the
# outermost of each summand only, are converted to the spaces' element type `E` the walk
# computes in, when their common `promote_type` promotes to `E` (`Int` with `Rational` on a
# `Float64` mesh): the entries are promoted to `E` anyway, and `Int * Float64` and
# `Rational * Float64` convert the same way. A `Ref` scale is read live, so it is left as it
# is and only its value type joins the promotion. Only a product holding a restriction gets
# here, and an operand whose scales agree is returned untouched, as is one whose common type
# does not promote to `E`: on a Float32 mesh, a `Float64` scale would move the values and
# add a third weight type, not remove one. `simplify_ast` keeps
# the literals (its folding reads them).
_uniform_scales(op, ::Type) = op
# Scales still of several types after that keep the walk boxing whatever the split does, and
# a split then only adds walks (45648 B against 39680 B on a Float32 mesh), so it is skipped.
@inline _mixed_scales(op) = _scales_differ(op, _scale_type(op))
function _uniform_scales(op::OperatorAdd, ::Type{E}) where {E}
    T = _scale_type(op)
    return promote_type(T, E) === E && _scales_differ(op, T) ? _promote_scales(op, E) : op
end

@inline _scale_type(op::OperatorScale) = _scalar_type(op.scalar)
@inline _scale_type(op::OperatorAdd) = promote_type(
    _scale_type(op.left_op), _scale_type(op.right_op))
@inline _scale_type(::Any) = Union{}
@inline _scalar_type(x) = typeof(x)
@inline _scalar_type(::Base.RefValue{S}) where {S} = S

@inline _scales_differ(op::OperatorScale, T) = _scalar_differs(op.scalar, T)
@inline _scales_differ(op::OperatorAdd, T) = _scales_differ(op.left_op, T) ||
                                             _scales_differ(op.right_op, T)
@inline _scales_differ(::Any, T) = false
@inline _scalar_differs(x, T) = typeof(x) !== T
@inline _scalar_differs(::Base.RefValue, T) = false

_promote_scales(op::OperatorScale, T) = OperatorScale(
    _promote_scalar(op.scalar, T), op.inner_op)
_promote_scales(op::OperatorAdd, T) = OperatorAdd(
    _promote_scales(op.left_op, T), _promote_scales(op.right_op, T))
_promote_scales(op, T) = op
_promote_scalar(x, T) = convert(T, x)
_promote_scalar(x::Base.RefValue, T) = x

# How many restricted summands `op` holds, under any wrapper, as a bound on the stencil
# shapes it brings (at most `2^bits`). A restriction nested in another adds no shape: both
# answer `()` or the same full tuple. `g * (Rₕ(:l, v) + Rₕ(:r, v))` counts two, and stays
# one summand, since a scale is never distributed.
@inline _shape_bits(op::RegionRestriction) = op.zero_stencil === nothing ?
                                             max(1, _shape_bits(op.inner_op)) :
                                             _shape_bits(op.inner_op)
@inline _shape_bits(op::UnaryWrapper) = _shape_bits(op.inner_op)
@inline _shape_bits(op::OperatorAdd) = _shape_bits(op.left_op) + _shape_bits(op.right_op)
@inline _shape_bits(::Any) = 0

# An operand's summands with every unprobed restriction over a sum distributed over it:
# `Rₕ(:c, v + Rₕ(:l, v))` is `Rₕ(:c, v) + Rₕ(:c, Rₕ(:l, v))`, entry for entry. Left whole,
# the restriction would be one summand of too many shapes to group.
@inline _expanded_summands(op::OperatorAdd) = (
    _expanded_summands(op.left_op)..., _expanded_summands(op.right_op)...)
@inline function _expanded_summands(op::RegionRestriction)
    op.zero_stencil === nothing || return (op,)
    return map(x -> _rewrap_inner(op, x), _expanded_summands(op.inner_op))
end
# A difference is linear in its operand entry for entry, so one over a sum holding a
# restriction distributes the same way: `D₋ₓ(Rₕ(:c, v + Rₕ(:l, v)))` is
# `D₋ₓ(Rₕ(:c, v)) + D₋ₓ(Rₕ(:c, Rₕ(:l, v)))` (gpena/Bramble.jl#759). An operand with no
# restriction is left whole, so an unrestricted difference compiles nothing new.
const _DifferenceNode = Union{BackwardDifference, ForwardDifference, CenteredDifference,
    StarDifference, CrossWeightedDifference}
@inline function _expanded_summands(op::_DifferenceNode)
    _holds_restriction(op.inner_op) || return (op,)
    return map(x -> _rewrap_inner(op, x), _expanded_summands(op.inner_op))
end
@inline _expanded_summands(op) = (op,)

# The bits of `_expanded_summands(op)` summed, read off the types alone so the count folds
# to a constant.
@inline _vanishing_count(op::OperatorAdd) = _vanishing_count(op.left_op) +
                                            _vanishing_count(op.right_op)
@inline _vanishing_count(op::RegionRestriction) = op.zero_stencil === nothing ?
                                                  _restricted_bits(op.inner_op) :
                                                  _shape_bits(op)
@inline _vanishing_count(op) = _shape_bits(op)

# Each expanded summand of a restricted operand brings at least the restriction's bit.
@inline _restricted_bits(op::OperatorAdd) = _restricted_bits(op.left_op) +
                                            _restricted_bits(op.right_op)
@inline _restricted_bits(op::RegionRestriction) = op.zero_stencil === nothing ?
                                                  _restricted_bits(op.inner_op) :
                                                  max(1, _shape_bits(op))
@inline _restricted_bits(op) = max(1, _shape_bits(op))

# The grouping is decided by the summand types alone, at compile time, so the split infers
# its concrete result. Two restricted summands share a sum only when their types are the
# same: a scaled and an unscaled restriction have different stencil types, and a sum of the
# two boxes as three restricted summands do (`0.3 * Rₕ(:b, v) + Rₕ(:t, v)`). Summands of one
# type are paired in the order they come, up to `cap` bits a sum, and a summand of more bits
# takes a sum of its own; the summands with no bits join the first sum. Which sums come out
# depends on the types, not on the order the summands were written in.
#
# A difference reads its operand at two points, so one over a restricted difference has
# more stencil shapes than its restriction's bit says: `D₋ₓ(D₊ᵧ(Rₕ(:l, v)))` and its `:r`
# twin, paired in one sum, boxed (62496 B, gpena/Bramble.jl#759). Each difference level over
# a restricted difference adds a bit, so such a summand takes a sum of its own at any depth.
function _type_shape_bits(T, nested::Bool = true)
    if T <: RegionRestriction
        inner = _type_shape_bits(fieldtype(T, :inner_op), nested)
        return fieldtype(T, :zero_stencil) === Nothing ? max(1, inner) : inner
    elseif nested && T <: _DifferenceNode
        I = fieldtype(T, :inner_op)
        inner = _type_shape_bits(I)
        return inner > 0 && _type_restricted_difference(I) ? inner + 1 : inner
    elseif T <: UnaryWrapper
        return _type_shape_bits(fieldtype(T, :inner_op), nested)
    elseif T <: OperatorAdd
        return _type_shape_bits(fieldtype(T, :left_op), nested) +
               _type_shape_bits(fieldtype(T, :right_op), nested)
    end
    return 0
end

# Whether `T` holds, under any wrapper or sum, a difference over a restriction.
function _type_restricted_difference(T)
    if T <: _DifferenceNode
        return _type_shape_bits(fieldtype(T, :inner_op)) > 0
    elseif T <: UnaryWrapper
        return _type_restricted_difference(fieldtype(T, :inner_op))
    elseif T <: OperatorAdd
        return _type_restricted_difference(fieldtype(T, :left_op)) ||
               _type_restricted_difference(fieldtype(T, :right_op))
    end
    return false
end
# The same count at run time, read off the type, so a linear form walks two such summands
# apart (`_unit_bits`, linear.jl): walked as one, `D₋ₓ(D₊ᵧ(Rₕ(:l, v) + Rₕ(:r, v)))` boxed.
@generated _shape_bits(::T) where {T <: _DifferenceNode} = _type_shape_bits(T)

# `T` with every scale and restriction around it peeled off: what a summand's full stencil
# is built from, entry for entry, so two summands with the same core have stencils of the
# same length.
function _type_core(T)
    while T <: Union{OperatorScale, GridFunctionScale, RegionRestriction}
        T = fieldtype(T, :inner_op)
    end
    return T
end

# With restricted summands on both sides (`cap == 1`), every product's stencil length is
# one of a left length times a right one, and past three distinct lengths the walk's
# `Union` widens to `Tuple{Vararg}` and boxes at every point. A restricted summand alone
# brings `0` or `q` entries; a plain summand of `p` entries beside it brings `p` or `p + q`,
# and only `q == p` keeps the lengths on both sides at `n` and `2n`, three products in all.
# So a lone plain summand joins a restriction of its own core (`u + Rₕ(:l, u)`, a scale on
# either allowed), and otherwise the plain summands take a sum of their own:
# `v + Rₕ(:t, D₋ₓ(v))`, `D₋ₓ(u) + D₋ₓ(Rₕ(:l, u))` and `u + D₋ₓ(u) + Rₕ(:l, u)` boxed
# 11152 B to 16704 B (gpena/Bramble.jl#759).
function _joins_restriction(P, K)
    while K <: Union{OperatorScale, GridFunctionScale}
        K = fieldtype(K, :inner_op)
    end
    return K <: RegionRestriction && _type_core(K) === _type_core(P)
end

# The summands of `Ts` (indices) as groups: one per sum, the plain summands (no bits)
# joining the first group, or, both sides restricted and `separate` set, as
# `_joins_restriction` allows.
function _type_groups(Ts::Vector{Any}, cap::Int, separate::Bool, nested::Bool = true)
    # Plain loops over plain vectors: compiled once into the package image (`precompile`
    # below), with no closure or generator of its own left to compile per session.
    plain = Int[]
    kinds = Any[]
    members = Vector{Int}[]
    for i in 1:length(Ts)
        if _type_shape_bits(Ts[i], nested) == 0
            push!(plain, i)
            continue
        end
        k = findfirst(isequal(Ts[i]), kinds)
        if k === nothing
            push!(kinds, Ts[i])
            push!(members, [i])
        else
            push!(members[k], i)
        end
    end
    groups = Vector{Int}[]
    firsts = Int[]
    for k in 1:length(kinds)
        per = max(1, cap ÷ _type_shape_bits(kinds[k], nested))
        same = members[k]
        push!(firsts, length(groups) + 1)
        for j in 1:per:length(same)
            push!(groups, same[j:min(j + per - 1, length(same))])
        end
    end
    isempty(groups) && return [plain]
    if isempty(plain) || cap > 1 || !separate
        groups[1] = vcat(plain, groups[1])
        return groups
    end
    if length(plain) == 1
        for k in 1:length(kinds)
            if _joins_restriction(Ts[plain[1]], kinds[k])
                groups[firsts[k]] = vcat(plain, groups[firsts[k]])
                return groups
            end
        end
    end
    return pushfirst!(groups, plain)
end

function _sum_expr(name::Symbol, idx::Vector{Int})
    ex = :($name[$(idx[1])])
    for i in idx[2:end]
        ex = :(OperatorAdd($ex, $name[$i]))
    end
    return ex
end

# A one-sided operand's nested-difference bits (`_type_shape_bits`) can give it more than
# the 16 sums a product splits into (seventeen `D₋ₓ(D₊ᵧ(Rₕ(s, v)))`), and the product would
# then stay whole (3413584 B against 673104 B paired); it is then paired as before them.
function _groups_expr(Ts::Vector{Any}, cap::Int, separate::Bool)
    groups = _type_groups(Ts, cap, separate)
    length(groups) > 16 && (groups = _type_groups(Ts, cap, separate, false))
    ex = Expr(:tuple)
    for g in groups
        push!(ex.args, _sum_expr(:ts, g))
    end
    return ex
end

function _product_sum_expr(D::Int, W, Ls::Vector{Any}, Rs::Vector{Any})
    ex = nothing
    for i in 1:length(Ls), j in 1:length(Rs)

        t = :(BilinearProduct{$D, $W, $(Ls[i]), $(Rs[j])}(ls[$i], rs[$j]))
        ex = ex === nothing ? t : :(OperatorAdd($ex, $t))
    end
    return ex
end

# The generators below run this Julia code at compile time; compiled into the package
# image, it does not cost the first restricted `form` of a session its own compilation.
precompile(_type_groups, (Vector{Any}, Int, Bool, Bool))
precompile(_type_shape_bits, (Any, Bool))
precompile(_type_restricted_difference, (Any,))
precompile(_type_core, (Any,))
precompile(_joins_restriction, (Any, Any))
precompile(_sum_expr, (Symbol, Vector{Int}))
precompile(_groups_expr, (Vector{Any}, Int, Bool))
precompile(_product_sum_expr, (Int, Any, Vector{Any}, Vector{Any}))

# `ts`'s summands regrouped into the sums `_type_groups` gives, as a tuple of operands.
@generated _operand_groups(ts::Tuple, ::Val{cap}, ::Val{separate}) where {cap, separate} = _groups_expr(
    collect(ts.parameters), cap, separate)
# A linear form's one side (linear.jl) has a cap of 2, where `separate` changes nothing.
@inline _operand_groups(ts::Tuple, cap::Val) = _operand_groups(ts, cap, Val(false))

# The sum of one product per pair of a left and a right group.
@generated _product_sum(op::BilinearProduct{D, W}, ls::Tuple, rs::Tuple) where {D, W} = _product_sum_expr(
    D, W, collect(ls.parameters), collect(rs.parameters))

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

    # The walk reads the trial leaf's mesh only here, so a trial leaf on another mesh that
    # moved in place would pass unnoticed: check its weights, as `_bind_walk` checks the
    # walked leaf's (gpena/Bramble.jl#466). A few integer loads per walk entry.
    weights(trial_leaf)
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
larger part of the work (a refill allocates nothing), and the pattern does not change between
assemblies.
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
form. `dirichlet_components` restricts which leaf components of a composite test space
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
when `A` is the default `SparseMatrixCSC`. A product with three or more region-restricted
summands is split by `form` into products of at most two (gpena/Bramble.jl#727); one it
leaves whole, as it would split into more than 16 or its constant scales keep several types,
allocates per grid point.

Runs serially or across threads following `form.trial_space`'s backend
[`execution_policy`](@ref): [`Serial`](@ref) (the default) or [`Parallel`](@ref).
[`assemble_parallel!`](@ref) is a separate, lower-level entry point that always threads,
ignoring the backend's policy.

`assemble!` uses the pre-resolved `form.ast` stored directly inside the form.

## Live coefficients
- Grid functions. The stored AST retains references to source `VectorElement` storage. Mutating values in-place (`Rₕ!(cₕ, ...)` or `parent(cₕ) .= ...`) between steps automatically updates the matrix entries. Nested scales such as `uₕ * (wₕ * v)` stay live too: each grid function is read at assembly time, never fused into a copy when the form is built. On the host this costs 0 allocations; on a device-backed space each fill copies the coefficient to the host anew, so it stays live but is not allocation-free there.
- Dynamic scalars. Plain numbers work directly for constant scalars. To update a scalar dynamically across loop iterations, wrap it in a `Ref(val)` (e.g. `β = Ref(1.0); a = form(Wₕ, Wₕ, (u, v) -> innerₕ(β * D₋ₓ(u), D₋ₓ(v)))`). Mutating `β[] = new_val` evaluates live during assembly with 0 allocations on the host; the `Ref` itself adds nothing, but a device-backed space's call still allocates regardless of the scalar.
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
threads. A device-resident matrix replays too, through positions recorded against its
host mirror, so only its recording fill searches (gpena/Bramble.jl#318).
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
        locality(typeof(A)) isa DeviceLocality &&
            throw(ArgumentError("symmetrize = true is not supported for a matrix in device memory."))
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
