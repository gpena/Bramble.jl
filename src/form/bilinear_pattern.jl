# bilinear_pattern.jl: matrix allocation and sparsity pattern discovery for a
# `BilinearForm`. Walks each term once with `PatternSink` (`bilinear_traversal.jl`) to
# collect every `(row, col)` it can reach, then hands the coordinates to `sparse!`.

# Dispatches on `::Type{T}` to ensure concrete vector return type.
@inline _zeros_of(::Type{T}, n::Int) where {T} = zeros(T, n)

# The element type is the one the form's own weights have, promoted against the trial
# space's (supporting automatic differentiation dual numbers). One place for this rule:
# reading it from the space alone instead of promoting against the data broke ForwardDiff in
# four separate places, each with the same symptom (`MethodError: no method matching
# Float64(::Dual)`), each time only on the AD path (bramble-verification §4).
@inline _matrix_eltype(ast, form::BilinearForm) =
    promote_type(_assembled_eltype(ast, form.test_space), eltype(form.trial_space))

# A hint for `sizehint!`, not a real bound: `local_stencil` can return a longer stencil at a
# boundary point than at this representative interior one, so this can undercount. Cheap to
# get wrong, since the only cost is a reallocation of `I_vec`/`J_vec` -- computing the true
# maximum (over the boundary stencils too) would cost more than the reallocation it saves.
# Named for what it is after gpena/Bramble.jl#41 pointed out that "upper bound" was a
# guarantee this never gave.
function _pattern_size_hint(ast::AST_TYPE, sp, mesh_markers, lin_indices) where {AST_TYPE}
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
    form::BilinearForm{D,TrialSpace,TestSpace,AST}, ast=form.ast
) where {D,TrialSpace,TestSpace,AST}
    # The test space: matrix rows are indexed by the test function and the quadrature weight
    # belongs to the integral over the test space mesh.
    space = form.test_space
    _check_block_meshes(ast, form.trial_space, form.test_space)
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    _validate_term_markers(ast, mesh_markers, "the form's space")
    lin_indices = LinearIndices(indices(Ωₕ))

    I_vec = Int[]
    J_vec = Int[]
    hint = _pattern_size_hint(ast, space, mesh_markers, lin_indices)
    sizehint!(I_vec, hint)
    sizehint!(J_vec, hint)

    visit_bilinear_stencil(PatternSink(I_vec, J_vec), ast, space, 0, 0)

    V_vec = _zeros_of(_matrix_eltype(ast, form), length(I_vec))
    return sparse!(I_vec, J_vec, V_vec, ndofs(form.test_space), ndofs(form.trial_space), +)
end

# Which entries a term can reach, block by block.
function _pattern_term!(
    I_vec::Vector{Int},
    J_vec::Vector{Int},
    term::TERM,
    trial_leaf,
    test_leaf,
    row_offset::Int,
    col_offset::Int,
) where {TERM}
    Ωₕ = mesh(test_leaf)
    mesh_markers = markers(Ωₕ)
    _validate_term_markers(term, mesh_markers, "one of the composite space's leaves")
    visit_bilinear_stencil(
        PatternSink(I_vec, J_vec), term, test_leaf, row_offset, col_offset
    )
    return nothing
end

# Recursion shape shared via `_visit_operator_add3` (form/common.jl).
function _pattern_blocks!(
    I_vec::Vector{Int}, J_vec::Vector{Int}, op::OperatorAdd, trial_leaves, test_leaves
)
    return _visit_operator_add3(
        _pattern_blocks!, I_vec, J_vec, op, trial_leaves, test_leaves
    )
end

function _pattern_blocks!(
    I_vec::Vector{Int}, J_vec::Vector{Int}, term::TERM, trial_leaves, test_leaves
) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        _check_block_meshes(term, blk.trial_leaf, blk.test_leaf)
        _pattern_term!(
            I_vec,
            J_vec,
            term,
            blk.trial_leaf,
            blk.test_leaf,
            blk.row_offset,
            blk.col_offset,
        )
    end
    return nothing
end

function allocate_system_matrix(
    form::BilinearForm{D,TrialSpace,TestSpace,AST}, ast=form.ast
) where {D,TrialSpace<:CompositeGridSpace,TestSpace<:CompositeGridSpace,AST}
    trial_leaves = leaf_spaces_offsets(form.trial_space)
    test_leaves = leaf_spaces_offsets(form.test_space)

    I_vec = Int[]
    J_vec = Int[]

    sp = first(first(test_leaves))
    Ωₛ = mesh(sp)
    hint =
        length(test_leaves) *
        _pattern_size_hint(ast, sp, markers(Ωₛ), LinearIndices(indices(Ωₛ)))
    sizehint!(I_vec, hint)
    sizehint!(J_vec, hint)

    _pattern_blocks!(I_vec, J_vec, ast, trial_leaves, test_leaves)

    ncols = ndofs(form.trial_space)
    nrows = ndofs(form.test_space)
    V_vec = _zeros_of(_matrix_eltype(ast, form), length(I_vec))
    return sparse!(I_vec, J_vec, V_vec, nrows, ncols, +)
end
