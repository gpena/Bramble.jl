# bilinear_pattern.jl: matrix allocation and sparsity pattern discovery for a
# `BilinearForm`. Walks each term once with `PatternSink` (`bilinear_traversal.jl`) to
# collect every `(row, col)` it can reach, then hands the coordinates to `sparse!`.

# Dispatches on `::Type{T}` to ensure concrete vector return type.
@inline _zeros_of(::Type{T}, n::Int) where {T} = zeros(T, n)

# `_assembled_eltype`/`_probed_eltype` (`form/linear.jl`, not owned by this subplan) probe
# the same representative point `_pattern_size_hint` does, through the same `local_stencil`,
# so they hit the identical device scalar-indexing wall (gpena/Bramble.jl#94 S4.0) -- a third
# call site reaching it, discovered running S4.1's own CHECK rather than assumed. `host_weights`
# is only defined for a `ScalarGridSpace`; a `CompositeGridSpace` test space is passed through
# unchanged; `_routed_eltype` probes its own leaves internally and is out of reach from here.
@inline _pattern_probe_space(sp::ScalarGridSpace) = host_weights(sp)
@inline _pattern_probe_space(sp) = sp

# The element type is the one the form's own weights have, promoted against the trial
# space's (supporting automatic differentiation dual numbers). One place for this rule:
# reading it from the space alone instead of promoting against the data broke ForwardDiff in
# four separate places, each with the same symptom (`MethodError: no method matching
# Float64(::Dual)`), each time only on the AD path (bramble-verification §4).
@inline _matrix_eltype(ast, form::BilinearForm) = promote_type(
    _assembled_eltype(ast, _pattern_probe_space(form.test_space)), eltype(form.trial_space)
)

"""
    _allocate_from_pattern(::Type{MT}, nrows::Int, ncols::Int, I::Vector{Int}, J::Vector{Int}, V::AbstractVector) -> MT

Build the `nrows × ncols` matrix a form's sparsity pattern describes, from the coordinate
triplet `(I, J, V)` `visit_bilinear_stencil`/`PatternSink` collected -- summing `V[k]` into
any `(row, col)` that `I`/`J` name more than once, matching `sparse!`'s own combiner.

The one place a fresh system matrix is born (S1.1, gpena/Bramble.jl#12): every backend's
matrix type implements exactly this to be usable with [`allocate_system_matrix`](@ref).
`SparseMatrixCSC`'s method is `sparse!` itself, consuming `I`/`J` in place. The generic
`AbstractMatrix` fallback -- the dense `Matrix{Float64}` positive control among them --
allocates zeros and scatters into it, since a dense matrix has no sparsity pattern to build.
"""
@inline function _allocate_from_pattern(
        ::Type{MT}, nrows::Int, ncols::Int, I_vec::Vector{Int}, J_vec::Vector{Int},
        V_vec::AbstractVector
) where {MT <: SparseMatrixCSC}
    return sparse!(I_vec, J_vec, V_vec, nrows, ncols, +)
end

function _allocate_from_pattern(
        ::Type{MT}, nrows::Int, ncols::Int, I_vec::Vector{Int}, J_vec::Vector{Int},
        V_vec::AbstractVector{T}
) where {MT <: AbstractMatrix, T}
    # Built via `Array{T}(undef, ...)` + `fill!`, not `zeros(T, nrows, ncols)`: `zeros`
    # dispatches on its first argument as an ordinary value, and analysed abstractly (a
    # `V_vec` too generic to pin `T` down at inference time, as `report_package` does) that
    # argument's own inferred type is `Any`, which inference can't rule out being another
    # `Integer` dimension rather than a type -- so it also considers `zeros(dims::Integer...)`,
    # producing a phantom `Array{Float64, 3}` no backend ever actually returns
    # (gpena/Bramble.jl#12, JET gate). `Array{T}` is `Core.apply_type`, not a value-dispatched
    # call, so it carries no such ambiguity.
    A = Array{T}(undef, nrows, ncols)
    fill!(A, zero(T))
    @inbounds for k in eachindex(I_vec, J_vec, V_vec)
        A[I_vec[k], J_vec[k]] += V_vec[k]
    end
    return A
end

# A hint for `sizehint!`, not a real bound: `local_stencil` can return a longer stencil at a
# boundary point than at this representative interior one, so this can undercount. Cheap to
# get wrong, since the only cost is a reallocation of `I_vec`/`J_vec` -- computing the true
# maximum (over the boundary stencils too) would cost more than the reallocation it saves.
# Named for what it is after gpena/Bramble.jl#41 pointed out that "upper bound" was a
# guarantee this never gave.
#
# `host_weights` (gpena/Bramble.jl#94 S4.0): `local_stencil` reads `sp`'s weights and its
# mesh's spacings one grid point at a time, which a device-backed `sp` refuses outright --
# a no-op on a host-backed `sp`, so the CPU path pays one locality check and nothing else.
#
# `sp` is typed `::ScalarGridSpace` (every call site passes a walked leaf, which bottoms
# out at one) rather than left generic: `host_weights` also has a method for
# `SeparableWeights`, and an untyped `sp` makes JET consider that branch reachable here too,
# reporting `mesh(::SeparableWeights)` as unresolved below even though nothing ever calls
# this with one -- a static inference artefact, not a live path.
function _pattern_size_hint(
        ast::AST_TYPE, sp::ScalarGridSpace, mesh_markers, lin_indices
) where {AST_TYPE}
    hp = host_weights(sp)
    grid_inds = indices(mesh(hp))
    npts = length(grid_inds)
    I = grid_inds[length(grid_inds) ÷ 2 + 1]
    return npts * length(local_stencil(ast, hp, I, mesh_markers, lin_indices[I]))
end

"""
    allocate_system_matrix(form::BilinearForm, ast = resolve_form_ast(form)) -> AbstractMatrix

Build the matrix a `BilinearForm` assembles into: the appropriate size, correct sparsity
pattern, and stored zeros throughout, in `matrix_type(backend(test_space(form)))` --
`SparseMatrixCSC{Float64,Int}` by default, or whatever [`backend`](@ref) the space's mesh was
built with (see [`_allocate_from_pattern`](@ref)). On a [`metal_backend`](@ref) space this
returns a device-resident `BrambleMetalExt.MetalSparseMatrixCSR{Float32,Int64}`, not a host
`SparseMatrixCSC` -- verified against real triplets (gpena/Bramble.jl#94 S4.1).

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
        form::BilinearForm{D, TrialSpace, TestSpace, AST}, ast = form.ast
) where {D, TrialSpace, TestSpace, AST}
    # The test space: matrix rows are indexed by the test function and the quadrature weight
    # belongs to the integral over the test space mesh -- unless the test side is what
    # interpolates, in which case the trial leaf is the one that stays native and supplies
    # both (gpena/Bramble.jl#263, `_walked_leaf`).
    ast = _bind_interp_spaces(ast, form.trial_space, form.test_space)
    _check_block_meshes(ast, form.trial_space, form.test_space)
    space = _walked_leaf(ast, form.trial_space, form.test_space)
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    _validate_term_markers(ast, mesh_markers, "the form's space")
    lin_indices = LinearIndices(indices(Ωₕ))

    I_vec = Int[]
    J_vec = Int[]
    hint = _pattern_size_hint(ast, space, mesh_markers, lin_indices)
    sizehint!(I_vec, hint)
    sizehint!(J_vec, hint)

    # `host_weights` (gpena/Bramble.jl#94 S4.0): the walk below reads `space`'s weights and
    # mesh spacings one grid point at a time, same reasoning as `_pattern_size_hint` above.
    visit_bilinear_stencil(PatternSink(I_vec, J_vec), ast, host_weights(space), 0, 0)

    MT = matrix_type(backend(form.test_space))
    V_vec = _zeros_of(_matrix_eltype(ast, form), length(I_vec))
    return _allocate_from_pattern(
        MT, ndofs(form.test_space), ndofs(form.trial_space), I_vec, J_vec, V_vec
    )
end

# Which entries a term can reach, block by block.
function _pattern_term!(
        I_vec::Vector{Int},
        J_vec::Vector{Int},
        term::TERM,
        trial_leaf,
        test_leaf,
        row_offset::Int,
        col_offset::Int
) where {TERM}
    sp = _walked_leaf(term, trial_leaf, test_leaf)
    Ωₕ = mesh(sp)
    mesh_markers = markers(Ωₕ)
    _validate_term_markers(term, mesh_markers, "one of the composite space's leaves")
    # `host_weights` (gpena/Bramble.jl#94 S4.0), same reasoning as the scalar overload above.
    visit_bilinear_stencil(
        PatternSink(I_vec, J_vec), term, host_weights(sp), row_offset, col_offset
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
        bound = _bind_interp_spaces(term, blk.trial_leaf, blk.test_leaf)
        _check_block_meshes(bound, blk.trial_leaf, blk.test_leaf)
        _pattern_term!(
            I_vec,
            J_vec,
            bound,
            blk.trial_leaf,
            blk.test_leaf,
            blk.row_offset,
            blk.col_offset
        )
    end
    return nothing
end

# A composite space on either side is walked block by block, the scalar side (if any) as a
# one-leaf composite. `_walked_leaf` on a mixed pair would pick one whole space and drop the
# component the term names on the composite side, so a term on leaf `d > 1` would land in
# block 1 (composite trial) or not resolve at all (composite test).
function allocate_system_matrix(
        form::BilinearForm{D, <:CompositeGridSpace, <:CompositeGridSpace}, ast = form.ast
) where {D}
    return _allocate_block_system_matrix(form, ast)
end

function allocate_system_matrix(
        form::BilinearForm{D, <:ScalarGridSpace, <:CompositeGridSpace}, ast = form.ast
) where {D}
    return _allocate_block_system_matrix(form, ast)
end

function allocate_system_matrix(
        form::BilinearForm{D, <:CompositeGridSpace, <:ScalarGridSpace}, ast = form.ast
) where {D}
    return _allocate_block_system_matrix(form, ast)
end

function _allocate_block_system_matrix(form::BilinearForm, ast)
    trial_leaves = leaf_spaces_offsets(form.trial_space)
    test_leaves = leaf_spaces_offsets(form.test_space)

    I_vec = Int[]
    J_vec = Int[]

    sp = first(first(test_leaves))
    Ωₛ = mesh(sp)
    # Both scaffolding walks below -- the size hint and the element-type probe -- evaluate a
    # stencil once, so an interpolation in the term needs a source space for them too. The
    # first leaf serves: neither walk reads which columns come back, only how many and what
    # their weights' type is, and every leaf answers those the same way. The entries
    # themselves are produced by `_pattern_blocks!`, which binds the term block by block.
    probe_ast = _bind_interp_spaces(
        ast, first(first(trial_leaves)), first(first(test_leaves))
    )
    hint = length(test_leaves) *
           _pattern_size_hint(probe_ast, sp, markers(Ωₛ), LinearIndices(indices(Ωₛ)))
    sizehint!(I_vec, hint)
    sizehint!(J_vec, hint)

    _pattern_blocks!(I_vec, J_vec, ast, trial_leaves, test_leaves)

    ncols = ndofs(form.trial_space)
    nrows = ndofs(form.test_space)
    MT = matrix_type(backend(form.test_space))
    V_vec = _zeros_of(_matrix_eltype(probe_ast, form), length(I_vec))
    return _allocate_from_pattern(MT, nrows, ncols, I_vec, J_vec, V_vec)
end
