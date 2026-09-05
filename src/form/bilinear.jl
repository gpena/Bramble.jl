# --- Struct definitions ----------------------------------------------------------- #

"""
    BilinearForm{D, TrialSpace, TestSpace, AST}

Represents a bilinear form defined over a trial space and test space.

# Arguments
- `trial_space::TrialSpace`: Space for the trial function.
- `test_space::TestSpace`: Space for the test function.
- `ast::AST`: Resolved expression tree.

The form resolves its expression tree `ast` once at construction, referencing the underlying
storage of any coefficient grid functions (`VectorElement`). In-place updates via `Rₕ!(cₕ, ...)`
or `values(cₕ) .= ...` are automatically seen by subsequent assemblies with zero heap allocations.
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
struct BilinearForm{D, TrialSpace, TestSpace, AST}
    trial_space::TrialSpace
    test_space::TestSpace
    ast::AST
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
    return BilinearForm{D, typeof(Wₕ), typeof(Vₕ), typeof(ast)}(Wₕ, Vₕ, ast)
end

# --- Utility helpers -------------------------------------------------------------- #

@inline function add_to_sparse!(A::SparseMatrixCSC, row::Int, col::Int, val::Number)
    p1 = A.colptr[col]
    p2 = A.colptr[col + 1] - 1

    if (p2 - p1) < 32
        idx = p1
        @inbounds while idx <= p2
            if A.rowval[idx] == row
                A.nzval[idx] += val
                return
            end
            idx += 1
        end
    else
        lo = p1
        hi = p2
        @inbounds while lo <= hi
            mid = (lo + hi) >>> 1
            mid_row = A.rowval[mid]
            if mid_row < row
                lo = mid + 1
            elseif mid_row > row
                hi = mid - 1
            else
                A.nzval[mid] += val
                return
            end
        end
    end
end

# Dispatches on `::Type{T}` to ensure concrete vector return type.
@inline _zeros_of(::Type{T}, n::Int) where {T} = zeros(T, n)

# Whether an earlier entry of this stencil already named the same pair of offsets.
@inline function _offsets_seen_before(stencil, k::Int, off_u, off_v)
    @inbounds for l in 1:(k - 1)
        stencil[l][1] == off_u && stencil[l][2] == off_v && return true
    end
    return false
end

# Which column of the trial block a stencil entry's trial slot names, or `0` when it names
# none of them.
#
# Ordinary offsets are bounds-checked and dropped on boundaries. An interpolation entry
# (`AbsoluteColumn`) names a source column directly.
@inline function _trial_column(lin_indices, I::CartesianIndex, off_u)
    Iu = I + CartesianIndex(off_u)
    return checkbounds(Bool, lin_indices, Iu) ? lin_indices[Iu] : 0
end

@inline _trial_column(lin_indices, I::CartesianIndex, off_u::AbsoluteColumn) = off_u.col

# Refuse cross-mesh coupling unless an explicit mapping (such as interpolation) is provided.
@noinline function _throw_cross_mesh_block(term, Ωu, Ωv)
    throw(ArgumentError(
        "a bilinear term coupling two leaves over different meshes has no assembly: the " *
        "trial leaf has $(npoints(Ωu, Tuple)) points and the test leaf $(npoints(Ωv, Tuple)), " *
        "so an index on one names no point on the other. Got $(typeof(term)). Couple leaves " *
        "that share a mesh, or wrap the trial function in an interpolation operator: `πₕ(Wtrial, u)`."))
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
    throw(ArgumentError(
        "the interpolation operator in a bilinear term names a space that is not the trial " *
        "function's: `πₕ` was given a space over a mesh of $(npoints(Ωsrc, Tuple)) points, " *
        "while the trial leaf this term assembles into has $(npoints(Ωu, Tuple)). Got " *
        "$(typeof(term)). `πₕ(Wsrc, u)` interpolates from the space the trial function " *
        "lives on, so `Wsrc` must be that space."))
end

@inline _check_block_meshes(op::OperatorAdd, trial_leaf, test_leaf) = _visit_operator_add1(
    _check_block_meshes, op, trial_leaf, test_leaf)

function _pattern_upper_bound(ast::AST_TYPE, sp, mesh_markers, lin_indices) where {AST_TYPE}
    grid_inds = indices(mesh(sp))
    npts = length(grid_inds)
    I = grid_inds[length(grid_inds) ÷ 2 + 1]
    return npts * length(local_stencil(ast, sp, I, mesh_markers, lin_indices[I]))
end

"""
    allocate_system_matrix(form::BilinearForm, ast = resolve_form_ast(form)) -> SparseMatrixCSC

Build the sparse matrix a `BilinearForm` assembles into: the appropriate size, correct sparsity
pattern, and stored zeros throughout.

The pattern follows from the stencil rather than coefficient values, remaining invariant while the mesh
and expression structure are unchanged. Preallocating the matrix once outside loops allows zero-allocation
in-place assembly:

```julia
A = allocate_system_matrix(a)
for step in 1:nsteps
    assemble!(A, a)          # refills values in-place with zero allocations
end
```

Only the structure is preallocated here; all stored entries are zero until `assemble!` fills them.

See also [`assemble`](@ref) and [`assemble!`](@ref).
"""
function allocate_system_matrix(
        form::BilinearForm{D, TrialSpace, TestSpace, AST},
        ast = form.ast) where {D, TrialSpace, TestSpace, AST}
    # The test space: matrix rows are indexed by the test function and the quadrature weight
    # belongs to the integral over the test space mesh.
    space = form.test_space
    _check_block_meshes(ast, form.trial_space, form.test_space)
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    _validate_term_markers(ast, mesh_markers, "the form's space")
    lin_indices = LinearIndices(indices(Ωₕ))
    n = length(lin_indices)

    I_vec = Int[]
    J_vec = Int[]
    hint = _pattern_upper_bound(ast, space, mesh_markers, lin_indices)
    sizehint!(I_vec, hint)
    sizehint!(J_vec, hint)

    @inbounds for I in indices(Ωₕ)
        lin_idx = lin_indices[I]
        stencil = local_stencil(ast, space, I, mesh_markers, lin_idx)

        for k in eachindex(stencil)
            off_u, off_v, _ = stencil[k]
            _offsets_seen_before(stencil, k, off_u, off_v) && continue

            Iv = I + CartesianIndex(off_v)
            col = _trial_column(lin_indices, I, off_u)
            if checkbounds(Bool, lin_indices, Iv) && col != 0
                push!(I_vec, lin_indices[Iv])
                push!(J_vec, col)
            end
        end
    end

    # The element type is the one the form's own weights have, promoted against the space's
    # (supporting automatic differentiation dual numbers).
    V_vec = _zeros_of(
        promote_type(_assembled_eltype(ast, space), eltype(form.trial_space)), length(I_vec))
    return sparse!(I_vec, J_vec, V_vec, ndofs(form.test_space), ndofs(form.trial_space), +)
end

# Which entries a term can reach, block by block.
function _pattern_term!(I_vec::Vector{Int}, J_vec::Vector{Int}, term::TERM, trial_leaf,
        test_leaf, row_offset::Int, col_offset::Int) where {TERM}
    Ωₕ = mesh(test_leaf)
    mesh_markers = markers(Ωₕ)
    _validate_term_markers(term, mesh_markers, "one of the composite space's leaves")
    lin_indices = LinearIndices(indices(Ωₕ))

    @inbounds for I in indices(Ωₕ)
        stencil = local_stencil(term, test_leaf, I, mesh_markers, lin_indices[I])

        for k in eachindex(stencil)
            off_u, off_v, _ = stencil[k]
            _offsets_seen_before(stencil, k, off_u, off_v) && continue

            Iv = I + CartesianIndex(off_v)
            col = _trial_column(lin_indices, I, off_u)

            if checkbounds(Bool, lin_indices, Iv) && col != 0
                push!(I_vec, lin_indices[Iv] + row_offset)
                push!(J_vec, col + col_offset)
            end
        end
    end
    return nothing
end

# Recursion shape shared via `_visit_operator_add3` (form/common.jl).
function _pattern_blocks!(I_vec::Vector{Int}, J_vec::Vector{Int}, op::OperatorAdd,
        trial_leaves, test_leaves)
    _visit_operator_add3(
        _pattern_blocks!, I_vec, J_vec, op, trial_leaves, test_leaves)
end

function _pattern_blocks!(I_vec::Vector{Int}, J_vec::Vector{Int}, term::TERM,
        trial_leaves, test_leaves) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        _check_block_meshes(term, blk.trial_leaf, blk.test_leaf)
        _pattern_term!(I_vec, J_vec, term, blk.trial_leaf, blk.test_leaf,
            blk.row_offset, blk.col_offset)
    end
    return nothing
end

function allocate_system_matrix(
        form::BilinearForm{D, TrialSpace, TestSpace, AST},
        ast = form.ast) where {D, TrialSpace <: CompositeGridSpace,
        TestSpace <: CompositeGridSpace, AST}
    trial_leaves = leaf_spaces_offsets(form.trial_space)
    test_leaves = leaf_spaces_offsets(form.test_space)

    I_vec = Int[]
    J_vec = Int[]

    sp = first(first(test_leaves))
    Ωₛ = mesh(sp)
    hint = length(test_leaves) *
           _pattern_upper_bound(ast, sp, markers(Ωₛ), LinearIndices(indices(Ωₛ)))
    sizehint!(I_vec, hint)
    sizehint!(J_vec, hint)

    _pattern_blocks!(I_vec, J_vec, ast, trial_leaves, test_leaves)

    ncols = ndofs(form.trial_space)
    nrows = ndofs(form.test_space)
    V_vec = _zeros_of(
        promote_type(_assembled_eltype(ast, form.test_space), eltype(form.trial_space)),
        length(I_vec))
    return sparse!(I_vec, J_vec, V_vec, nrows, ncols, +)
end

# --- Assembly implementations ----------------------------------------------------- #

function apply_dirichlet_labels!(
        A::AbstractMatrix, form::BilinearForm, dirichlet_labels, dirichlet_components = nothing)
    if dirichlet_labels !== nothing
        if dirichlet_labels isa Symbol
            dirichlet_bc!(A, test_space(form), dirichlet_labels; components = dirichlet_components)
        elseif dirichlet_labels isa Tuple
            if !isempty(dirichlet_labels)
                dirichlet_bc!(A, test_space(form), dirichlet_labels...;
                    components = dirichlet_components)
            end
        end
    end
end

"""
    assemble(form::BilinearForm; dirichlet_labels = nothing, dirichlet_components = nothing) -> SparseMatrixCSC

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
[`execution_policy`](@ref): [`Serial`](@ref) by default. Optional `dirichlet_labels` applies
boundary conditions to the matrix; `dirichlet_components` restricts which leaf components of a
composite trial space they bind to (see [`dirichlet_bc!`](@ref)).
"""
function assemble(form::BilinearForm; dirichlet_labels = nothing, dirichlet_components = nothing)
    _validate_dirichlet_labels(dirichlet_labels)
    ast_resolved = form.ast
    A = allocate_system_matrix(form, ast_resolved)
    assemble!(A, form; dirichlet_labels = dirichlet_labels,
        dirichlet_components = dirichlet_components, ast = ast_resolved)
    return A
end

# --- Helper cores for function barrier optimization ------------------------------- #

# The scalar case: one block, no offsets.
function _assemble_bilinear_core!(A::SparseMatrixCSC, trial_space, test_space,
        ast::AST_TYPE) where {AST_TYPE}
    _check_block_meshes(ast, trial_space, test_space)
    _scatter_block!(A, ast, test_space, 0, 0)
    return A
end

# One grid point's stencil, scattered into the matrix. The whole of what differs between
# a serial sweep and a threaded one: both call this for each `I`, only the surrounding
# `for` differs (serial `indices(Ωₕ)` vs. `Threads.@threads` over one colour's subgrid).
# Any future change to how a stencil entry lands in the matrix (gpena/Bramble.jl#26) is
# written once here rather than fitted into both drivers.
@inline function _scatter_point!(A::SparseMatrixCSC, term::TERM, sp, I::CartesianIndex,
        lin_indices, mesh_markers, row_offset::Int, col_offset::Int) where {TERM}
    stencil = local_stencil(term, sp, I, mesh_markers, lin_indices[I])

    for (off_u, off_v, weight) in stencil
        Iv = I + CartesianIndex(off_v)
        col = _trial_column(lin_indices, I, off_u)

        if checkbounds(Bool, lin_indices, Iv) && col != 0
            add_to_sparse!(A, lin_indices[Iv] + row_offset, col + col_offset, weight)
        end
    end
    return nothing
end

# One term into one block, serially. `row_offset` comes from the test leaf and `col_offset`
# from the trial leaf: a matrix row is indexed by the test function.
function _scatter_block!(A::SparseMatrixCSC, term::TERM, sp, row_offset::Int,
        col_offset::Int) where {TERM}
    Ωₕ = mesh(sp)
    mesh_markers = markers(Ωₕ)
    lin_indices = LinearIndices(indices(Ωₕ))

    @inbounds for I in indices(Ωₕ)
        _scatter_point!(A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset)
    end
    return A
end

function _assemble_blocks!(A::SparseMatrixCSC, op::OperatorAdd, trial_leaves, test_leaves)
    _visit_operator_add2(
        _assemble_blocks!, A, op, trial_leaves, test_leaves)
end

function _assemble_blocks!(A::SparseMatrixCSC, term::TERM, trial_leaves,
        test_leaves) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        _check_block_meshes(term, blk.trial_leaf, blk.test_leaf)
        _scatter_block!(A, term, blk.test_leaf, blk.row_offset, blk.col_offset)
    end
    return A
end

function _assemble_bilinear_core!(A::SparseMatrixCSC, trial_space::CompositeGridSpace,
        test_space::CompositeGridSpace, ast::AST_TYPE) where {AST_TYPE}
    _assemble_blocks!(A, ast, leaf_spaces_offsets(trial_space),
        leaf_spaces_offsets(test_space))
    return A
end

# Safe stride for matrix assembly, determined from a sample stencil evaluation.
function _bilinear_colour_strides(ast::AST_TYPE, sp, ::Val{D}) where {AST_TYPE, D}
    Ωₕ = mesh(sp)
    grid_inds = indices(Ωₕ)
    lin_indices = LinearIndices(grid_inds)
    I = grid_inds[length(grid_inds) ÷ 2 + 1]
    stencil = local_stencil(ast, sp, I, markers(Ωₕ), lin_indices[I])
    isempty(stencil) && return ntuple(_ -> 1, D)

    lo = stencil[1][2]
    hi = stencil[1][2]
    for (_, off_v, _) in stencil
        lo = min.(lo, off_v)
        hi = max.(hi, off_v)
    end
    return hi .- lo .+ 1
end

# One colour, threaded, writing directly into the matrix.
@noinline function _sweep_bilinear_colour!(A::SparseMatrixCSC, sp, term::TERM, idxs,
        lin_indices, mesh_markers, row_offset::Int, col_offset::Int) where {TERM}
    Threads.@threads for I in idxs
        _scatter_point!(A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset)
    end
    return nothing
end

# Every colour in turn, using strided subgrids.
function _sweep_bilinear!(A::SparseMatrixCSC, sp, term::TERM, strides, row_offset::Int,
        col_offset::Int) where {TERM}
    Ωₕ = mesh(sp)
    grid_inds = indices(Ωₕ)
    lin_indices = LinearIndices(grid_inds)
    mesh_markers = markers(Ωₕ)

    if prod(strides) == 1
        _sweep_bilinear_colour!(A, sp, term, grid_inds, lin_indices, mesh_markers,
            row_offset, col_offset)
        return A
    end

    for c in CartesianIndices(strides)
        _sweep_bilinear_colour!(A, sp, term, _colour_subgrid(grid_inds, c, strides),
            lin_indices, mesh_markers, row_offset, col_offset)
    end
    return A
end

function _assemble_blocks_parallel!(A::SparseMatrixCSC, op::OperatorAdd, trial_leaves,
        test_leaves, dim_val::Val)
    _visit_operator_add2(
        _assemble_blocks_parallel!, A, op, trial_leaves, test_leaves, dim_val)
end

function _assemble_blocks_parallel!(A::SparseMatrixCSC, term::TERM, trial_leaves,
        test_leaves, dim_val::Val) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        _check_block_meshes(term, blk.trial_leaf, blk.test_leaf)
        _sweep_bilinear!(A, blk.test_leaf, term,
            _bilinear_colour_strides(term, blk.test_leaf, dim_val),
            blk.row_offset, blk.col_offset)
    end
    return A
end

function _assemble_bilinear_parallel_core!(A::SparseMatrixCSC, trial_space, test_space,
        ast::AST_TYPE, dim_val::Val) where {AST_TYPE}
    _check_block_meshes(ast, trial_space, test_space)
    _sweep_bilinear!(A, test_space, ast,
        _bilinear_colour_strides(ast, test_space, dim_val), 0, 0)
    return A
end

function _assemble_bilinear_parallel_core!(A::SparseMatrixCSC,
        trial_space::CompositeGridSpace, test_space::CompositeGridSpace,
        ast::AST_TYPE, dim_val::Val) where {AST_TYPE}
    _assemble_blocks_parallel!(A, ast, leaf_spaces_offsets(trial_space),
        leaf_spaces_offsets(test_space), dim_val)
    return A
end

"""
    assemble!(A::SparseMatrixCSC, form::BilinearForm; dirichlet_labels = nothing, dirichlet_components = nothing, ast = form.ast) -> SparseMatrixCSC

Assemble the `BilinearForm` into the preallocated sparse matrix `A`, allocating nothing (**0 bytes**).

Runs serially or across threads following `form.trial_space`'s backend
[`execution_policy`](@ref): [`Serial`](@ref) (the default) or [`Parallel`](@ref).
[`assemble_parallel!`](@ref) is a separate, lower-level entry point that always threads,
ignoring the backend's policy.

By default `assemble!` uses the pre-resolved `form.ast` stored directly inside the form.

## Live coefficients
- Grid functions: the stored AST retains references to source `VectorElement` storage. Mutating values in-place (`Rₕ!(cₕ, ...)` or `values(cₕ) .= ...`) between steps automatically updates the matrix entries with 0 allocations.
- Dynamic scalars: plain numbers work directly for constant scalars. To update a scalar dynamically across loop iterations, wrap it in a `Ref(val)` (e.g. `β = Ref(1.0); a = form(Wₕ, Wₕ, (u, v) -> innerₕ(β * D₋ₓ(u), D₋ₓ(v)))`). Mutating `β[] = new_val` evaluates live during assembly with 0 allocations.
"""
function assemble!(
        A::SparseMatrixCSC, form::BilinearForm{D, TrialSpace, TestSpace, AST};
        dirichlet_labels = nothing,
        dirichlet_components = nothing,
        ast = form.ast) where {D, TrialSpace, TestSpace, AST}
    _validate_dirichlet_labels(dirichlet_labels)
    fill!(nonzeros(A), zero(eltype(nonzeros(A))))

    if execution_policy(form.trial_space) isa Serial
        _assemble_bilinear_core!(A, form.trial_space, form.test_space, ast)
    else
        _assemble_bilinear_parallel_core!(A, form.trial_space, form.test_space, ast, Val(D))
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
        A::SparseMatrixCSC, form::BilinearForm{D, TrialSpace, TestSpace, AST},
        ast = form.ast) where {D, TrialSpace, TestSpace, AST}
    fill!(nonzeros(A), zero(eltype(nonzeros(A))))

    _assemble_bilinear_parallel_core!(
        A, form.trial_space, form.test_space, ast, Val(D))

    return A
end
