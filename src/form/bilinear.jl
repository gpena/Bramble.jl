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
# running counter so a future caller could read it without a race even if the walk over grid
# points were threaded (today's cached path is serial-only; see below).
"""
    NzvalSegment = Tuple{Vector{Int},Vector{Int}}

One term's recorded nzval positions for one block, as `(point_ptr, positions)`.

`point_ptr[lin_idx]:point_ptr[lin_idx + 1] - 1` is the slice of `positions` holding grid
point `lin_idx`'s own entries, in the order a scatter walk visits them. Addressed per point
rather than by a shared running counter, so a replay stays correct whatever order the grid
is visited in.

See also: [`RecordSink`](@ref), [`ReplaySink`](@ref).
"""
const NzvalSegment = Tuple{Vector{Int},Vector{Int}}

# One `BilinearForm`'s nzval-position cache: valid only for the exact matrix object last
# assembled into (`A === cache.A`), one `NzvalSegment` per (term, block) the serial assembly
# walk visits, in visitation order. A companion *value*, not a type parameter of
# `BilinearForm` -- so it can be filled in lazily, on the first `assemble!` call, without the
# form itself needing to be mutable or its type to depend on whether a cache exists yet.
mutable struct _AssemblyCache
    A::Union{Nothing,SparseMatrixCSC}
    ast::Any
    segments::Vector{NzvalSegment}
end

# Every fresh `BilinearForm` starts pointing at this one, shared, empty vector rather than
# allocating its own: it is never mutated in place (a cache miss *replaces* `cache.segments`
# wholesale, see `_assemble_bilinear_core_cached!`, rather than `empty!`ing whatever it
# currently references), so sharing it across every not-yet-assembled form is safe. Keeps
# `form(Wₕ, Vₕ, f)` itself allocation-free: only the `_AssemblyCache` wrapper is a genuine
# per-form cost (one allocation, since it is a `mutable struct` and therefore always
# heap-boxed), not a second one for an empty vector nothing has scattered into yet.
const _NO_SEGMENTS = NzvalSegment[]

_AssemblyCache() = _AssemblyCache(nothing, nothing, _NO_SEGMENTS)

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
β = Ref(1.0)
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(β * D₋ₓ(u), D₋ₓ(v)))
# Inside time loop:
β[] = 3.0
assemble!(A, a) # zero allocations, evaluates with β = 3.0
```
"""
struct BilinearForm{D,TrialSpace,TestSpace,AST}
    trial_space::TrialSpace
    test_space::TestSpace
    ast::AST
    cache::_AssemblyCache
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
@inline (form::BilinearForm)(u, v) = dot(v, assemble(form) * u)

"""
    resolve_form_ast(form::BilinearForm)

Return the resolved AST stored inside the bilinear form.
"""
@inline resolve_form_ast(form::BilinearForm) = form.ast

"""
    form(Wₕ, Vₕ, f) -> BilinearForm

Construct a `BilinearForm` over the trial space `Wₕ` and the test space `Vₕ` from the
bilinear expression `f` (a function of trial and test arguments `(u, v)`).

# Examples
```julia
# a(u, v) = (∇₋ₕu, ∇₋ₕv)₊
a = form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))

# a coupled system, one term per block
a = form(Vₕ, Vₕ, (u, v) -> inner₊ₓ(D₋ₓ(u(1)), D₋ₓ(v(1))) + innerₕ(u(2), v(1)))
```
"""
function form(Wₕ, Vₕ, f)
    D = dim(Wₕ)
    raw_ast = f(TrialFunction{D}(), TestFunction{D}())
    _validate_form_expression(raw_ast, Val(D))
    ast = resolve_ast(raw_ast)
    return BilinearForm{D,typeof(Wₕ),typeof(Vₕ),typeof(ast)}(Wₕ, Vₕ, ast, _AssemblyCache())
end

# --- Mesh compatibility checks ------------------------------------------------------ #
#
# Shared by pattern discovery (`bilinear_pattern.jl`) and both execution strategies
# (`bilinear_execution.jl`): every path that walks a term over a block needs the same
# guard, so it is stated once, here, rather than once per caller.

# Refuse cross-mesh coupling unless an explicit mapping (such as interpolation) is provided.
@noinline function _throw_cross_mesh_block(term, Ωu, Ωv)
    throw(
        ArgumentError(
            "a bilinear term coupling two leaves over different meshes has no assembly: the " *
            "trial leaf has $(npoints(Ωu, Tuple)) points and the test leaf $(npoints(Ωv, Tuple)), " *
            "so an index on one names no point on the other. Got $(typeof(term)). Couple leaves " *
            "that share a mesh, or wrap the trial function in an interpolation operator: `πₕ(Wtrial, u)`.",
        ),
    )
end

@inline function _check_block_meshes(term, trial_leaf, test_leaf)
    _check_interp_spaces(term, trial_leaf)
    _all_trial_interpolated(term) && return nothing

    Ωu = mesh(trial_leaf)
    Ωv = mesh(test_leaf)
    npoints(Ωu, Tuple) == npoints(Ωv, Tuple) || _throw_cross_mesh_block(term, Ωu, Ωv)
    return nothing
end

@inline function _check_one_interp_space(term, Wsrc, trial_leaf)
    Ωsrc = mesh(Wsrc)
    Ωu = mesh(trial_leaf)
    npoints(Ωsrc, Tuple) == npoints(Ωu, Tuple) ||
        _throw_interp_space_mismatch(term, Ωsrc, Ωu)
    return nothing
end

@noinline function _throw_interp_space_mismatch(term, Ωsrc, Ωu)
    throw(
        ArgumentError(
            "the interpolation operator in a bilinear term names a space that is not the trial " *
            "function's: `πₕ` was given a space over a mesh of $(npoints(Ωsrc, Tuple)) points, " *
            "while the trial leaf this term assembles into has $(npoints(Ωu, Tuple)). Got " *
            "$(typeof(term)). `πₕ(Wsrc, u)` interpolates from the space the trial function " *
            "lives on, so `Wsrc` must be that space.",
        ),
    )
end

@inline _check_block_meshes(op::OperatorAdd, trial_leaf, test_leaf) =
    _visit_operator_add1(_check_block_meshes, op, trial_leaf, test_leaf)

# --- Assembly implementations ----------------------------------------------------- #

function apply_dirichlet_labels!(
    A::AbstractMatrix, form::BilinearForm, dirichlet_labels, dirichlet_components=nothing
)
    if dirichlet_labels !== nothing
        if dirichlet_labels isa Symbol
            dirichlet_bc!(
                A, test_space(form), dirichlet_labels; components=dirichlet_components
            )
        elseif dirichlet_labels isa Tuple
            if !isempty(dirichlet_labels)
                dirichlet_bc!(
                    A,
                    test_space(form),
                    dirichlet_labels...;
                    components=dirichlet_components,
                )
            end
        end
    end
end

"""
    assemble(form::BilinearForm; dirichlet = nothing, dirichlet_components = nothing) -> SparseMatrixCSC

Allocate a matrix with the form's sparsity pattern and assemble into it.

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
function assemble(form::BilinearForm; dirichlet=nothing, dirichlet_components=nothing)
    ast_resolved = form.ast
    A = allocate_system_matrix(form, ast_resolved)
    assemble!(
        A,
        form;
        dirichlet=dirichlet,
        dirichlet_components=dirichlet_components,
        ast=ast_resolved,
    )
    return A
end

"""
    assemble!(A::SparseMatrixCSC, form::BilinearForm; dirichlet = nothing, dirichlet_components = nothing, ast = form.ast) -> SparseMatrixCSC

Assemble the `BilinearForm` into the preallocated sparse matrix `A`, allocating nothing (**0 bytes**).

Runs serially or across threads following `form.trial_space`'s backend
[`execution_policy`](@ref): [`Serial`](@ref) (the default) or [`Parallel`](@ref).
[`assemble_parallel!`](@ref) is a separate, lower-level entry point that always threads,
ignoring the backend's policy.

By default `assemble!` uses the pre-resolved `form.ast` stored directly inside the form.

## Live coefficients
- Grid functions: the stored AST retains references to source `VectorElement` storage. Mutating values in-place (`Rₕ!(cₕ, ...)` or `parent(cₕ) .= ...`) between steps automatically updates the matrix entries with 0 allocations.
- Dynamic scalars: plain numbers work directly for constant scalars. To update a scalar dynamically across loop iterations, wrap it in a `Ref(val)` (e.g. `β = Ref(1.0); a = form(Wₕ, Wₕ, (u, v) -> innerₕ(β * D₋ₓ(u), D₋ₓ(v)))`). Mutating `β[] = new_val` evaluates live during assembly with 0 allocations.
"""
function assemble!(
    A::SparseMatrixCSC,
    form::BilinearForm{D,TrialSpace,TestSpace,AST};
    dirichlet=nothing,
    dirichlet_components=nothing,
    ast=form.ast,
) where {D,TrialSpace,TestSpace,AST}
    dirichlet_labels, _ = _normalize_dirichlet(dirichlet)
    fill!(nonzeros(A), zero(eltype(nonzeros(A))))

    if execution_policy(form.trial_space) isa Serial
        _assemble_bilinear_core_cached!(
            A, form.trial_space, form.test_space, ast, form.cache
        )
    else
        _assemble_bilinear_parallel_core!(A, form.trial_space, form.test_space, ast)
    end

    apply_dirichlet_labels!(A, form, dirichlet_labels, dirichlet_components)
    return A
end

"""
    assemble_parallel!(A::SparseMatrixCSC, form::BilinearForm, ast = form.ast) -> SparseMatrixCSC

Refill `A` with the assembled `form` across threads and return it, regardless of
`form.trial_space`'s backend policy. `A` must already carry the correct sparsity pattern from
[`allocate_system_matrix`](@ref) or a previous [`assemble`](@ref). Unlike `assemble!`, does
not apply `dirichlet_labels`.

Colouring on the test side ensures thread safety when updating stored matrix values concurrently.
"""
function assemble_parallel!(
    A::SparseMatrixCSC, form::BilinearForm{D,TrialSpace,TestSpace,AST}, ast=form.ast
) where {D,TrialSpace,TestSpace,AST}
    fill!(nonzeros(A), zero(eltype(nonzeros(A))))

    _assemble_bilinear_parallel_core!(A, form.trial_space, form.test_space, ast)

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
    dirichlet=nothing,
    dirichlet_components=nothing,
    symmetrize::Bool=false,
)
    A = assemble(a; dirichlet=dirichlet, dirichlet_components=dirichlet_components)
    F = assemble(l; dirichlet=dirichlet, dirichlet_components=dirichlet_components)

    if symmetrize
        dirichlet_labels, _ = _normalize_dirichlet(dirichlet)
        dirichlet_labels === nothing && _throw_symmetrize_without_dirichlet()
        symmetrize!(
            A, F, test_space(a), dirichlet_labels...; components=dirichlet_components
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
