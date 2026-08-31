# ==============================================================================
# Struct Definitions
# ==============================================================================

# `ParallelWorkspace` moved to form/parallel_workspace.jl. `linear.jl` names it as the type
# of a `LinearForm` field, which has to resolve when the struct is defined rather than when
# it is called, so keeping it here made unlocking that file alone impossible.

"""
    BilinearForm{D,TrialSpace,TestSpace,FType}

Represents a bilinear form defined over a trial space and test space.

# Fields
- `trial_space::TrialSpace`: The space for the trial function.
- `test_space::TestSpace`: The space for the test function.
- `f::FType`: The expression, as a function of a trial and a test argument.

The expression is kept as `f` rather than as a resolved tree, which is what makes a
coefficient live: [`resolve_form_ast`](@ref) calls it afresh on every assembly. A stored
tree was there and read by nothing, exactly as in `LinearForm`; the partition a parallel
assembly walks was built here too, and is now derived where it is used.
"""
struct BilinearForm{D, TrialSpace, TestSpace, FType}
    trial_space::TrialSpace
    test_space::TestSpace
    f::FType
end

"""
    trial_space(form::BilinearForm)

Returns the trial space of the bilinear form.
"""
trial_space(form::BilinearForm) = form.trial_space

"""
    test_space(form::BilinearForm)

Returns the test space of the bilinear form.
"""
test_space(form::BilinearForm) = form.test_space

# ==============================================================================
# CoupledBilinearForm (off-diagonal block coupling)
# ==============================================================================

"""
    CoupledBilinearForm{D}

A bilinear form on a **hierarchical** composite trial/test space, supporting
full off-diagonal coupling between different field components. The user-provided
lambda receives named tuple arguments matching the block structure, and the
assembled matrix contains all coupling blocks.

# Fields
- `trial_space`: hierarchical `CompositeGridSpace` for the trial field.
- `test_space`: hierarchical `CompositeGridSpace` for the test field.
- `block_asts`: `(n_test_leaves × n_trial_leaves)` matrix of per-block ASTs (or `nothing`).
- `trial_leaf_info`: flat list of `(ScalarGridSpace, dof_offset)` pairs for trial.
- `test_leaf_info`: flat list of `(ScalarGridSpace, dof_offset)` pairs for test.
"""
struct CoupledBilinearForm{D}
    trial_space::Any
    test_space::Any
    block_asts::Matrix{Any}           # (n_test_leaves, n_trial_leaves)
    trial_leaf_info::Vector{Any}      # Vector of (ScalarGridSpace, Int)
    test_leaf_info::Vector{Any}       # Vector of (ScalarGridSpace, Int)
end

trial_space(form::CoupledBilinearForm) = form.trial_space
test_space(form::CoupledBilinearForm) = form.test_space

@inline (form::BilinearForm)(u, v) = dot(v, assemble(form) * u)

"""
    resolve_form_ast(form::BilinearForm)

Fully resolves grid coefficient functions and scales inside the bilinear form's AST.
"""
@inline resolve_form_ast(form::BilinearForm{D, TrialSpace, TestSpace,
    FType}) where {
    D, TrialSpace, TestSpace, FType} = resolve_ast(form.f(
    TrialFunction{D}(), TestFunction{D}()))

"""
    form(Wₕ, Vₕ, f)

Constructs a `BilinearForm` over the trial space `Wₕ` and test space `Vₕ` using the bilinear expression `f`.

The constructor evaluates the stencil of the operator at a representative node to compute the safe multi-coloring stride needed for race-free parallel assembly.

If both `Wₕ` and `Vₕ` are **hierarchical** `CompositeGridSpace`s (i.e. their top-level components are themselves `CompositeGridSpace`s), the lambda `f` is expected to accept **tuple arguments** matching the block structure (e.g. `((u, p), (v, q)) -> ...`), and a `CoupledBilinearForm` is returned instead.

# Examples
```julia
# 1D Poisson bilinear form: a(u, v) = (∇u, ∇v)
a = form(Wh, Wh, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v)))

# Coupled Stokes form on X = (Vh × Wh):
a = form(X, X, ((u, p), (v, q)) ->
    inner₊(∇₋ₕ(u), ∇₋ₕ(v)) +
    innerₕ(D₋ₓ(u[1]), q) + innerₕ(D₋ᵧ(u[2]), q) +
    inner₊(p, D₋ₓ(v[1])) + inner₊(p, D₋ᵧ(v[2])))
```
"""
function form(Wₕ, Vₕ, f)
    D = dim(Wₕ)

    # Built and discarded, for the error rather than for the tree: an expression that does
    # not describe an operator fails here rather than at the first `assemble`. Nothing is
    # kept, because assembly resolves from `f` every time — see `LinearForm`.
    _validate_form_expression(f(TrialFunction{D}(), TestFunction{D}()), Val(D))

    return BilinearForm{D, typeof(Wₕ), typeof(Vₕ), typeof(f)}(Wₕ, Vₕ, f)
end

"""
    form(trial_space::CompositeGridSpace, test_space::CompositeGridSpace, f)

Specialized constructor for coupled bilinear forms on hierarchical composite spaces.
Detected when both spaces have at least one component that is itself a `CompositeGridSpace`.
The lambda `f` receives a tuple of symbolic trial args and a tuple of test args matching
the block structure of `trial_space` and `test_space`.
"""
function form(trial_space::CompositeGridSpace, test_space::CompositeGridSpace, f)
    if is_hierarchical(trial_space) || is_hierarchical(test_space)
        return _build_coupled_bilinear_form(trial_space, test_space, f)
    else
        # Fall through to the general constructor. Which blocks a term reaches is decided
        # at assembly, not here.
        D = dim(trial_space)
        _validate_form_expression(f(TrialFunction{D}(), TestFunction{D}()), Val(D))

        return BilinearForm{
            D, typeof(trial_space), typeof(test_space), typeof(f)}(
            trial_space, test_space, f)
    end
end

"""
    _build_coupled_bilinear_form(trial_space, test_space, f)

Internal constructor for `CoupledBilinearForm`. Generates indexed symbolic arguments
matching the block hierarchy, calls the lambda to build the full AST, then decomposes
it into per-block sub-ASTs using `extract_block_asts`.
"""
function _build_coupled_bilinear_form(trial_space::CompositeGridSpace, test_space::CompositeGridSpace, f)
    D = dim(trial_space)

    # Generate symbolic trial/test args matching the hierarchical block structure
    trial_args = make_trial_args(trial_space, D)
    test_args = make_test_args(test_space, D)

    # Build full coupled AST
    ast = f(trial_args, test_args)

    # Flatten to leaf spaces for assembly
    trial_leaf_info = collect_leaf_spaces_offsets(trial_space)
    test_leaf_info = collect_leaf_spaces_offsets(test_space)
    NT = length(trial_leaf_info)
    NS = length(test_leaf_info)

    # Decompose AST into (NS × NT) block matrix
    block_asts = extract_block_asts(ast, NT, NS)

    return CoupledBilinearForm{D}(
        trial_space, test_space, block_asts, trial_leaf_info, test_leaf_info)
end

# ==============================================================================
# Utility Helpers
# ==============================================================================

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

# `zeros(T, n)` with `T` a value rather than a literal type matches two methods —
# `zeros(::Type, ::Integer)` giving a vector and `zeros(::Integer, ::Integer)` giving a
# matrix — so the result infers as a union and `sparse` has no method for half of it.
# Dispatching on `::Type{T}` settles which one, and the vector comes back concrete.
@inline _zeros_of(::Type{T}, n::Int) where {T} = zeros(T, n)

"""
    allocate_system_matrix(form::BilinearForm, ast = resolve_form_ast(form))

Allocates a sparse matrix with the correct sparsity pattern corresponding to the bilinear form.
"""
function allocate_system_matrix(
        form::BilinearForm{D, TrialSpace, TestSpace, FType},
        ast = resolve_form_ast(form)) where {D, TrialSpace, TestSpace, FType}
    space = form.trial_space
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    lin_indices = LinearIndices(indices(Ωₕ))
    n = length(lin_indices)

    I_vec = Int[]
    J_vec = Int[]

    @inbounds for I in indices(Ωₕ)
        lin_idx = lin_indices[I]
        stencil = local_stencil(ast, space, I, mesh_markers, lin_idx)
        for (off_u, off_v, _) in stencil
            Iv = I + CartesianIndex(off_v)
            Iu = I + CartesianIndex(off_u)
            if checkbounds(Bool, lin_indices, Iv) && checkbounds(Bool, lin_indices, Iu)
                push!(I_vec, lin_indices[Iv])
                push!(J_vec, lin_indices[Iu])
            end
        end
    end

    # The element type is the one the form's own weights have, promoted against the space's
    # — the same rule `assemble` uses for a right-hand side, and for the same two reasons.
    # It is what lets a `Dual` through, and taking it from the space instead left it
    # inferred as a union: `zeros(eltype(space), n)` can be `zeros(n₁, n₂)` when `eltype`
    # of an unconstrained type parameter is not known to be a type, so `V_vec` came out as
    # `Union{Vector{Float64}, Matrix{Float64}}` and `sparse` had no method for half of it.
    # JET found that; nothing else would have, since the bad half is unreachable in
    # practice.
    V_vec = _zeros_of(_assembled_eltype(ast, space), length(I_vec))
    return sparse(I_vec, J_vec, V_vec, n, n)
end

# Which entries a term can reach, block by block.
#
# The version this replaces walked `space.spaces` with one offset used for both the row and
# the column, so the pattern it built had diagonal blocks and nothing else — and
# `add_to_sparse!` searches for an entry and returns quietly when it is missing, so every
# off-diagonal contribution was dropped without a word. `innerₕ(u(1), v(2))` assembled to
# zeros.
function _pattern_term!(I_vec::Vector{Int}, J_vec::Vector{Int}, term::TERM, trial_leaf,
        test_leaf, row_offset::Int, col_offset::Int) where {TERM}
    Ωₕ = mesh(test_leaf)
    mesh_markers = markers(Ωₕ)
    lin_indices = LinearIndices(indices(Ωₕ))

    @inbounds for I in indices(Ωₕ)
        stencil = local_stencil(term, test_leaf, I, mesh_markers, lin_indices[I])

        for (off_u, off_v, _) in stencil
            Iv = I + CartesianIndex(off_v)
            Iu = I + CartesianIndex(off_u)

            if checkbounds(Bool, lin_indices, Iv) && checkbounds(Bool, lin_indices, Iu)
                push!(I_vec, lin_indices[Iv] + row_offset)
                push!(J_vec, lin_indices[Iu] + col_offset)
            end
        end
    end
    return nothing
end

function _pattern_blocks!(I_vec::Vector{Int}, J_vec::Vector{Int}, op::OperatorAdd,
        trial_leaves, test_leaves)
    _pattern_blocks!(I_vec, J_vec, op.left_op, trial_leaves, test_leaves)
    _pattern_blocks!(I_vec, J_vec, op.right_op, trial_leaves, test_leaves)
    return nothing
end

function _pattern_blocks!(I_vec::Vector{Int}, J_vec::Vector{Int}, term::TERM,
        trial_leaves, test_leaves) where {TERM}
    blk = block_of(term, length(trial_leaves), length(test_leaves))

    if blk === nothing
        for c in 1:min(length(trial_leaves), length(test_leaves))
            _pattern_term!(I_vec, J_vec, term, first(trial_leaves[c]),
                first(test_leaves[c]), last(test_leaves[c]), last(trial_leaves[c]))
        end
        return nothing
    end

    tc, sc = blk
    _pattern_term!(I_vec, J_vec, term, first(trial_leaves[tc]), first(test_leaves[sc]),
        last(test_leaves[sc]), last(trial_leaves[tc]))
    return nothing
end

function allocate_system_matrix(
        form::BilinearForm{D, TrialSpace, TestSpace, FType},
        ast = resolve_form_ast(form)) where {D, TrialSpace <: CompositeGridSpace,
        TestSpace <: CompositeGridSpace, FType}
    trial_leaves = leaf_spaces_offsets(form.trial_space)
    test_leaves = leaf_spaces_offsets(form.test_space)

    I_vec = Int[]
    J_vec = Int[]
    _pattern_blocks!(I_vec, J_vec, ast, trial_leaves, test_leaves)

    ncols = ndofs(form.trial_space)
    nrows = ndofs(form.test_space)
    V_vec = _zeros_of(_assembled_eltype(ast, first_space(form.trial_space)), length(I_vec))
    return sparse(I_vec, J_vec, V_vec, nrows, ncols)
end

# ==============================================================================
# Assembly Implementations
# ==============================================================================

function apply_dirichlet_labels!(A::AbstractMatrix, form::BilinearForm, dirichlet_labels)
    if dirichlet_labels !== nothing
        if dirichlet_labels isa Symbol
            dirichlet_bc!(A, trial_space(form), dirichlet_labels)
        elseif dirichlet_labels isa Tuple
            if !isempty(dirichlet_labels)
                dirichlet_bc!(A, trial_space(form), dirichlet_labels...)
            end
        end
    end
end

"""
    assemble(form::BilinearForm; dirichlet_labels = nothing)

Assembles the system matrix of the `BilinearForm` using parallel lock-free assembly. Optional `dirichlet_labels` applies boundary conditions to the matrix.
"""
function assemble(form::BilinearForm; dirichlet_labels = nothing)
    _validate_dirichlet_labels(dirichlet_labels)
    ast_resolved = resolve_form_ast(form)
    A = allocate_system_matrix(form, ast_resolved)
    assemble_parallel!(A, form, ast_resolved)
    apply_dirichlet_labels!(A, form, dirichlet_labels)
    return A
end

# ==============================================================================
# Helper Cores for Function Barrier Optimization
# ==============================================================================

# The scalar case: one block, no offsets.
function _assemble_bilinear_core!(A::SparseMatrixCSC, trial_space, test_space,
        ast::AST_TYPE) where {AST_TYPE}
    _scatter_block!(A, ast, trial_space, 0, 0)
    return A
end

# One term into one block, serially. `row_offset` comes from the test leaf and `col_offset`
# from the trial leaf: a matrix row is indexed by the test function.
function _scatter_block!(A::SparseMatrixCSC, term::TERM, sp, row_offset::Int,
        col_offset::Int) where {TERM}
    Ωₕ = mesh(sp)
    mesh_markers = markers(Ωₕ)
    lin_indices = LinearIndices(indices(Ωₕ))

    @inbounds for I in indices(Ωₕ)
        stencil = local_stencil(term, sp, I, mesh_markers, lin_indices[I])

        for (off_u, off_v, weight) in stencil
            Iv = I + CartesianIndex(off_v)
            Iu = I + CartesianIndex(off_u)

            if checkbounds(Bool, lin_indices, Iv) && checkbounds(Bool, lin_indices, Iu)
                add_to_sparse!(A, lin_indices[Iv] + row_offset,
                    lin_indices[Iu] + col_offset, weight)
            end
        end
    end
    return A
end

# Walk the sum and send each term to the blocks it belongs to. The same shape as
# `_route_terms!` on the vector side, and for the same reason: recursing keeps each term
# concretely typed at its own call, where flattening into a vector makes every one a dynamic
# read.
#
# A term naming neither side goes to the diagonal blocks, since `Σᵢ innerₕ(uᵢ, vᵢ)` is block
# diagonal and not full. A term naming both goes to one block, off-diagonal included — which
# is what the version this replaces could not do: it walked the top-level components with a
# single offset for row and column, so off-diagonal terms had nowhere to land and
# `add_to_sparse!` dropped them in silence.
function _assemble_blocks!(A::SparseMatrixCSC, op::OperatorAdd, trial_leaves, test_leaves)
    _assemble_blocks!(A, op.left_op, trial_leaves, test_leaves)
    _assemble_blocks!(A, op.right_op, trial_leaves, test_leaves)
    return A
end

function _assemble_blocks!(A::SparseMatrixCSC, term::TERM, trial_leaves,
        test_leaves) where {TERM}
    blk = block_of(term, length(trial_leaves), length(test_leaves))

    if blk === nothing
        for c in 1:min(length(trial_leaves), length(test_leaves))
            _scatter_block!(A, term, first(test_leaves[c]), last(test_leaves[c]),
                last(trial_leaves[c]))
        end
        return A
    end

    tc, sc = blk
    _scatter_block!(A, term, first(test_leaves[sc]), last(test_leaves[sc]),
        last(trial_leaves[tc]))
    return A
end

function _assemble_bilinear_core!(A::SparseMatrixCSC, trial_space::CompositeGridSpace,
        test_space::CompositeGridSpace, ast::AST_TYPE) where {AST_TYPE}
    _assemble_blocks!(A, ast, leaf_spaces_offsets(trial_space),
        leaf_spaces_offsets(test_space))
    return A
end

# The safe stride for a matrix assembly, read from a sample stencil.
#
# A bilinear stencil writes to `(I + off_v, I + off_u)`, so two points collide on an entry
# only if their *row* footprints overlap: rows disjoint implies entries disjoint whatever the
# columns do. So the span of the test-side offsets is enough, which is the same quantity
# `_colour_strides` computes for a vector assembly.
#
# Taken from an evaluated stencil rather than from `stencil_offsets`, which refuses a
# `BilinearProduct` on purpose: its offsets are pairs, and what is wanted here is one side.
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

# One colour, threaded, writing straight into the matrix. The colouring is what makes that
# safe: `add_to_sparse!` searches a column and updates in place, so two threads landing on
# one entry would race on the value.
@noinline function _sweep_bilinear_colour!(A::SparseMatrixCSC, sp, term::TERM, idxs,
        lin_indices, mesh_markers, row_offset::Int, col_offset::Int) where {TERM}
    Threads.@threads for I in idxs
        stencil = local_stencil(term, sp, I, mesh_markers, lin_indices[I])

        for (off_u, off_v, weight) in stencil
            Iv = I + CartesianIndex(off_v)
            Iu = I + CartesianIndex(off_u)

            if checkbounds(Bool, lin_indices, Iv) && checkbounds(Bool, lin_indices, Iu)
                add_to_sparse!(A, lin_indices[Iv] + row_offset,
                    lin_indices[Iu] + col_offset, weight)
            end
        end
    end
    return nothing
end

# Every colour in turn, as strided sub-grids rather than a materialised list of indices. The
# version this replaces binned the whole grid into a `Vector{Vector{CartesianIndex}}` at
# construction — 9.3 MB at 90,000 degrees of freedom, before a single entry was assembled.
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

# The threaded counterpart of `_assemble_blocks!`, term outside and grid inside for the same
# reason the vector side is: routing per point redoes the component walk at every one.
function _assemble_blocks_parallel!(A::SparseMatrixCSC, op::OperatorAdd, trial_leaves,
        test_leaves, dim_val::Val)
    _assemble_blocks_parallel!(A, op.left_op, trial_leaves, test_leaves, dim_val)
    _assemble_blocks_parallel!(A, op.right_op, trial_leaves, test_leaves, dim_val)
    return A
end

function _assemble_blocks_parallel!(A::SparseMatrixCSC, term::TERM, trial_leaves,
        test_leaves, dim_val::Val) where {TERM}
    blk = block_of(term, length(trial_leaves), length(test_leaves))

    if blk === nothing
        for c in 1:min(length(trial_leaves), length(test_leaves))
            sp = first(test_leaves[c])
            _sweep_bilinear!(A, sp, term, _bilinear_colour_strides(term, sp, dim_val),
                last(test_leaves[c]), last(trial_leaves[c]))
        end
        return A
    end

    tc, sc = blk
    sp = first(test_leaves[sc])
    _sweep_bilinear!(A, sp, term, _bilinear_colour_strides(term, sp, dim_val),
        last(test_leaves[sc]), last(trial_leaves[tc]))
    return A
end

function _assemble_bilinear_parallel_core!(A::SparseMatrixCSC, trial_space, test_space,
        ast::AST_TYPE, dim_val::Val) where {AST_TYPE}
    _sweep_bilinear!(A, trial_space, ast,
        _bilinear_colour_strides(ast, trial_space, dim_val), 0, 0)
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
    assemble!(A::SparseMatrixCSC, form::BilinearForm; dirichlet_labels = nothing, ast = resolve_form_ast(form))

Assembles the `BilinearForm` serially into the preallocated sparse matrix `A`.
"""
function assemble!(
        A::SparseMatrixCSC, form::BilinearForm{D, TrialSpace, TestSpace, FType};
        dirichlet_labels = nothing,
        ast = resolve_form_ast(form)) where {D, TrialSpace, TestSpace, FType}
    _validate_dirichlet_labels(dirichlet_labels)
    fill!(nonzeros(A), zero(eltype(nonzeros(A))))

    _assemble_bilinear_core!(A, form.trial_space, form.test_space, ast)

    apply_dirichlet_labels!(A, form, dirichlet_labels)
    return A
end

"""
    assemble_parallel!(A::SparseMatrixCSC, form::BilinearForm, ast = resolve_form_ast(form))

Assembles the `BilinearForm` into `A` across threads, one colour of the grid at a time.
"""
function assemble_parallel!(
        A::SparseMatrixCSC, form::BilinearForm{D, TrialSpace, TestSpace, FType},
        ast = resolve_form_ast(form)) where {D, TrialSpace, TestSpace, FType}
    fill!(nonzeros(A), zero(eltype(nonzeros(A))))

    _assemble_bilinear_parallel_core!(
        A, form.trial_space, form.test_space, ast, Val(D))

    return A
end

# ==============================================================================
# CoupledBilinearForm Assembly
# ==============================================================================

"""
    allocate_system_matrix(form::CoupledBilinearForm)

Allocates a sparse matrix for the coupled block system. The sparsity pattern is
determined by scanning all non-zero block ASTs and collecting the (row, col) pairs
they can contribute to (including their global DOF offsets).
"""
function allocate_system_matrix(form::CoupledBilinearForm{D}) where {D}
    NS = size(form.block_asts, 1)
    NT = size(form.block_asts, 2)

    total_test_dofs = sum(info -> ndofs(info[1]), form.test_leaf_info)
    total_trial_dofs = sum(info -> ndofs(info[1]), form.trial_leaf_info)

    I_vec = Int[]
    J_vec = Int[]

    for i in 1:NS, j in 1:NT

        ast_ij = form.block_asts[i, j]
        ast_ij === nothing && continue

        sp_test, offset_row = form.test_leaf_info[i]
        sp_trial, offset_col = form.trial_leaf_info[j]

        Ωₕ = mesh(sp_test)
        mesh_markers = markers(Ωₕ)
        lin_indices = LinearIndices(indices(Ωₕ))

        for I in indices(Ωₕ)
            lin_idx = lin_indices[I]
            stencil = local_stencil(ast_ij, sp_test, I, mesh_markers, lin_idx)

            for (off_u, off_v, _) in stencil
                Iv = I + CartesianIndex(off_v)
                Iu = I + CartesianIndex(off_u)
                if checkbounds(Bool, lin_indices, Iv) && checkbounds(Bool, lin_indices, Iu)
                    push!(I_vec, lin_indices[Iv] + offset_row)
                    push!(J_vec, lin_indices[Iu] + offset_col)
                end
            end
        end
    end

    V_vec = zeros(Float64, length(I_vec))
    return sparse(I_vec, J_vec, V_vec, total_test_dofs, total_trial_dofs)
end

"""
    apply_dirichlet_labels!(A::AbstractMatrix, form::CoupledBilinearForm, dirichlet_labels)

Applies Dirichlet boundary conditions to the assembled coupled system matrix.
"""
function apply_dirichlet_labels!(A::AbstractMatrix, form::CoupledBilinearForm, dirichlet_labels)
    if dirichlet_labels !== nothing
        if dirichlet_labels isa Symbol
            dirichlet_bc!(A, form.trial_space, dirichlet_labels)
        elseif dirichlet_labels isa Tuple && !isempty(dirichlet_labels)
            dirichlet_bc!(A, form.trial_space, dirichlet_labels...)
        end
    end
end

"""
    assemble(form::CoupledBilinearForm; dirichlet_labels = nothing)

Assembles the full coupled block system matrix for the `CoupledBilinearForm`.
Each non-null block `(i,j)` is assembled separately using the appropriate leaf
scalar spaces and global DOF offsets, then contributions are added to the
preallocated sparse matrix with the correct (row, col) block positions.
"""
function assemble(form::CoupledBilinearForm{D}; dirichlet_labels = nothing) where {D}
    _validate_dirichlet_labels(dirichlet_labels)

    A = allocate_system_matrix(form)
    NS = size(form.block_asts, 1)
    NT = size(form.block_asts, 2)

    for i in 1:NS, j in 1:NT

        ast_ij = form.block_asts[i, j]
        ast_ij === nothing && continue

        # Resolve any lazy grid coefficients in this block's AST
        ast_resolved = resolve_ast(ast_ij)

        sp_test, offset_row = form.test_leaf_info[i]
        sp_trial, offset_col = form.trial_leaf_info[j]

        Ωₕ = mesh(sp_test)
        mesh_markers = markers(Ωₕ)
        lin_indices = LinearIndices(indices(Ωₕ))

        _assemble_coupled_block!(
            A, ast_resolved, sp_test, lin_indices, mesh_markers, offset_row, offset_col)
    end

    apply_dirichlet_labels!(A, form, dirichlet_labels)
    return A
end

"""
    _assemble_coupled_block!(A, ast, sp, lin_indices, mesh_markers, offset_row, offset_col)

Core loop for assembling a single `(i,j)` block of a `CoupledBilinearForm` into
the full sparse matrix `A`. Contributions are offset by `offset_row` (test DOFs)
and `offset_col` (trial DOFs).
"""
function _assemble_coupled_block!(A::SparseMatrixCSC, ast, sp, lin_indices, mesh_markers,
        offset_row::Int, offset_col::Int)
    for I in indices(mesh(sp))
        lin_idx = lin_indices[I]
        stencil = local_stencil(ast, sp, I, mesh_markers, lin_idx)

        for (off_u, off_v, weight) in stencil
            Iv = I + CartesianIndex(off_v)
            Iu = I + CartesianIndex(off_u)

            if checkbounds(Bool, lin_indices, Iv) && checkbounds(Bool, lin_indices, Iu)
                row = lin_indices[Iv] + offset_row
                col = lin_indices[Iu] + offset_col
                add_to_sparse!(A, row, col, weight)
            end
        end
    end
    return A
end
