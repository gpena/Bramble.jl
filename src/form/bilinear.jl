# ==============================================================================
# Struct Definitions
# ==============================================================================

# `ParallelWorkspace` moved to form/parallel_workspace.jl. `linear.jl` names it as the type
# of a `LinearForm` field, which has to resolve when the struct is defined rather than when
# it is called, so keeping it here made unlocking that file alone impossible.

"""
    BilinearForm{D,TrialSpace,TestSpace,ExprType,FType}

Represents a bilinear form defined over a trial space and test space.

# Fields
- `trial_space::TrialSpace`: The space for the trial function.
- `test_space::TestSpace`: The space for the test function.
- `ast::ExprType`: The symbolic expression AST representation of the form.
- `f::FType`: The user-defined lambda function representing the form.
- `workspace::ParallelWorkspace{D}`: Preallocated coordinate partitions for lock-free parallel assembly.
"""
struct BilinearForm{D, TrialSpace, TestSpace, ExprType <: LazyOp{D}, FType}
    trial_space::TrialSpace
    test_space::TestSpace
    ast::ExprType
    f::FType
    workspace::ParallelWorkspace{D}
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
@inline resolve_form_ast(form::BilinearForm{D, TrialSpace, TestSpace, ExprType,
    FType}) where {D, TrialSpace, TestSpace, ExprType, FType} = resolve_ast(form.f(
    TrialFunction{D}(), TestFunction{D}()))

"""
    form(Wₕ, Vₕ, f; stride_multiplier::Int = 1)

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
function form(Wₕ, Vₕ, f; stride_multiplier::Int = 1)
    D = dim(Wₕ)
    ast = f(TrialFunction{D}(), TestFunction{D}())

    # Extract mesh characteristics to discover stencil bounds upfront
    sp = first_space(Wₕ)
    Ωₕ = mesh(sp)
    mesh_markers = markers(Ωₕ)
    grid_inds = indices(Ωₕ)
    lin_indices = LinearIndices(grid_inds)

    # Evaluate the stencil at a representative interior node to discover off_v bounds
    center_I = grid_inds[length(grid_inds) ÷ 2 + 1]
    center_lin_idx = lin_indices[center_I]
    sample_stencil = local_stencil(ast, sp, center_I, mesh_markers, center_lin_idx)

    first_off_v = sample_stencil[1][2]
    min_v = first_off_v
    max_v = first_off_v

    for (_, off_v, _) in sample_stencil
        min_v = min.(min_v, off_v)
        max_v = max.(max_v, off_v)
    end

    # Compute mathematical safe strides and apply optional inflation
    base_strides = max_v .- min_v .+ 1
    strides = base_strides .* stride_multiplier
    stride_tuple = Tuple(strides)
    num_colors = prod(stride_tuple)

    # Group grid coordinates by color identifier
    color_groups = [CartesianIndex{D}[] for _ in 1:num_colors]
    linear_mapper = LinearIndices(stride_tuple)

    for I in grid_inds
        color_coord = ntuple(d -> mod(I[d] - 1, stride_tuple[d]) + 1, D)
        color_id = linear_mapper[color_coord...]
        push!(color_groups[color_id], I)
    end

    workspace = ParallelWorkspace{D}(color_groups)

    return BilinearForm{D, typeof(Wₕ), typeof(Vₕ), typeof(ast), typeof(f)}(
        Wₕ, Vₕ, ast, f, workspace)
end

"""
    form(trial_space::CompositeGridSpace, test_space::CompositeGridSpace, f; stride_multiplier=1)

Specialized constructor for coupled bilinear forms on hierarchical composite spaces.
Detected when both spaces have at least one component that is itself a `CompositeGridSpace`.
The lambda `f` receives a tuple of symbolic trial args and a tuple of test args matching
the block structure of `trial_space` and `test_space`.
"""
function form(trial_space::CompositeGridSpace, test_space::CompositeGridSpace, f; stride_multiplier::Int = 1)
    if is_hierarchical(trial_space) || is_hierarchical(test_space)
        return _build_coupled_bilinear_form(trial_space, test_space, f)
    else
        # Fall through to the general form constructor (block-diagonal behaviour)
        D = dim(trial_space)
        ast = f(TrialFunction{D}(), TestFunction{D}())

        sp = first_space(trial_space)
        Ωₕ = mesh(sp)
        mesh_markers = markers(Ωₕ)
        grid_inds = indices(Ωₕ)
        lin_indices = LinearIndices(grid_inds)

        center_I = grid_inds[length(grid_inds) ÷ 2 + 1]
        center_lin_idx = lin_indices[center_I]
        sample_stencil = local_stencil(ast, sp, center_I, mesh_markers, center_lin_idx)

        first_off_v = sample_stencil[1][2]
        min_v = first_off_v
        max_v = first_off_v
        for (_, off_v, _) in sample_stencil
            min_v = min.(min_v, off_v)
            max_v = max.(max_v, off_v)
        end

        base_strides = max_v .- min_v .+ 1
        strides = base_strides .* stride_multiplier
        stride_tuple = Tuple(strides)
        num_colors = prod(stride_tuple)

        color_groups = [CartesianIndex{D}[] for _ in 1:num_colors]
        linear_mapper = LinearIndices(stride_tuple)
        for I in grid_inds
            color_coord = ntuple(d -> mod(I[d] - 1, stride_tuple[d]) + 1, D)
            color_id = linear_mapper[color_coord...]
            push!(color_groups[color_id], I)
        end

        workspace = ParallelWorkspace{D}(color_groups)
        return BilinearForm{
            D, typeof(trial_space), typeof(test_space), typeof(ast), typeof(f)}(
            trial_space, test_space, ast, f, workspace)
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

"""
    allocate_system_matrix(form::BilinearForm, ast = resolve_form_ast(form))

Allocates a sparse matrix with the correct sparsity pattern corresponding to the bilinear form.
"""
function allocate_system_matrix(
        form::BilinearForm{D, TrialSpace, TestSpace, ExprType, FType},
        ast = resolve_form_ast(form)) where {D, TrialSpace, TestSpace, ExprType, FType}
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

    V_vec = zeros(eltype(form.trial_space), length(I_vec))
    return sparse(I_vec, J_vec, V_vec, n, n)
end

function allocate_system_matrix(
        form::BilinearForm{D, TrialSpace, TestSpace, ExprType, FType},
        ast = resolve_form_ast(form)) where {D, TrialSpace <: CompositeGridSpace,
        TestSpace <: CompositeGridSpace, ExprType, FType}
    space = form.trial_space
    N = ncomponents(TrialSpace)

    # Calculate DOF offsets for each subspace
    offsets = Int[0]
    for sp in space.spaces
        push!(offsets, offsets[end] + ndofs(sp))
    end
    total_dofs = offsets[end]

    I_vec = Int[]
    J_vec = Int[]

    # Iterate over each component space
    for c in 1:N
        sp = space.spaces[c]
        offset = offsets[c]
        Ωₕ = mesh(sp)
        mesh_markers = markers(Ωₕ)
        lin_indices = LinearIndices(indices(Ωₕ))

        @inbounds for I in indices(Ωₕ)
            lin_idx = lin_indices[I]
            stencil = local_stencil(ast, sp, I, mesh_markers, lin_idx)
            for (off_u, off_v, _) in stencil
                Iv = I + CartesianIndex(off_v)
                Iu = I + CartesianIndex(off_u)
                if checkbounds(Bool, lin_indices, Iv) && checkbounds(Bool, lin_indices, Iu)
                    push!(I_vec, lin_indices[Iv] + offset)
                    push!(J_vec, lin_indices[Iu] + offset)
                end
            end
        end
    end

    V_vec = zeros(eltype(space), length(I_vec))
    return sparse(I_vec, J_vec, V_vec, total_dofs, total_dofs)
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

function _assemble_bilinear_core!(A::SparseMatrixCSC, space, ast::AST_TYPE,
        lin_indices, mesh_markers) where {AST_TYPE}
    for I in indices(mesh(space))
        lin_idx = lin_indices[I]
        stencil = local_stencil(ast, space, I, mesh_markers, lin_idx)

        for (off_u, off_v, weight) in stencil
            Iv = I + CartesianIndex(off_v)
            Iu = I + CartesianIndex(off_u)

            if checkbounds(Bool, lin_indices, Iv) && checkbounds(Bool, lin_indices, Iu)
                row = lin_indices[Iv]
                col = lin_indices[Iu]
                add_to_sparse!(A, row, col, weight)
            end
        end
    end
    return A
end

function _assemble_bilinear_parallel_core!(
        A::SparseMatrixCSC, space, ast::AST_TYPE, lin_indices,
        mesh_markers, color_groups) where {AST_TYPE}
    num_colors = length(color_groups)
    num_threads = Threads.nthreads()
    for color_id in 1:num_colors
        color_group = color_groups[color_id]
        len = length(color_group)
        chunk_size = ceil(Int, len / num_threads)

        # Coarse-grained parallel chunking to minimize thread task creation overhead and false sharing
        Threads.@threads for tid in 1:num_threads
            start_idx = (tid - 1) * chunk_size + 1
            end_idx = min(tid * chunk_size, len)

            for idx in start_idx:end_idx
                I = color_group[idx]
                lin_idx = lin_indices[I]
                stencil = local_stencil(ast, space, I, mesh_markers, lin_idx)

                for (off_u, off_v, weight) in stencil
                    Iv = I + CartesianIndex(off_v)
                    Iu = I + CartesianIndex(off_u)

                    if checkbounds(Bool, lin_indices, Iv) &&
                       checkbounds(Bool, lin_indices, Iu)
                        row = lin_indices[Iv]
                        col = lin_indices[Iu]

                        add_to_sparse!(A, row, col, weight)
                    end
                end
            end
        end
    end
    return A
end

function _assemble_bilinear_core!(A::SparseMatrixCSC, space::CompositeGridSpace{N},
        ast::AST_TYPE, lin_indices, mesh_markers) where {N, AST_TYPE}
    offsets = Int[0]
    for sp in space.spaces
        push!(offsets, offsets[end] + ndofs(sp))
    end

    for c in 1:N
        sp = space.spaces[c]
        offset = offsets[c]

        for I in indices(mesh(sp))
            lin_idx = lin_indices[I]
            stencil = local_stencil(ast, sp, I, mesh_markers, lin_idx)

            for (off_u, off_v, weight) in stencil
                Iv = I + CartesianIndex(off_v)
                Iu = I + CartesianIndex(off_u)

                if checkbounds(Bool, lin_indices, Iv) && checkbounds(Bool, lin_indices, Iu)
                    row_local = lin_indices[Iv]
                    col_local = lin_indices[Iu]

                    row_global = row_local + offset
                    col_global = col_local + offset

                    add_to_sparse!(A, row_global, col_global, weight)
                end
            end
        end
    end
    return A
end

function _assemble_bilinear_parallel_core!(
        A::SparseMatrixCSC, space::CompositeGridSpace{N}, ast::AST_TYPE,
        lin_indices, mesh_markers, color_groups) where {N, AST_TYPE}
    num_colors = length(color_groups)
    num_threads = Threads.nthreads()

    offsets = Int[0]
    for sp in space.spaces
        push!(offsets, offsets[end] + ndofs(sp))
    end

    for color_id in 1:num_colors
        color_group = color_groups[color_id]
        len = length(color_group)
        chunk_size = ceil(Int, len / num_threads)

        # Coarse-grained parallel chunking
        Threads.@threads for tid in 1:num_threads
            start_idx = (tid - 1) * chunk_size + 1
            end_idx = min(tid * chunk_size, len)

            for idx in start_idx:end_idx
                I = color_group[idx]
                lin_idx = lin_indices[I]

                for c in 1:N
                    sp = space.spaces[c]
                    offset = offsets[c]

                    stencil = local_stencil(ast, sp, I, mesh_markers, lin_idx)

                    for (off_u, off_v, weight) in stencil
                        Iv = I + CartesianIndex(off_v)
                        Iu = I + CartesianIndex(off_u)

                        if checkbounds(Bool, lin_indices, Iv) &&
                           checkbounds(Bool, lin_indices, Iu)
                            row_local = lin_indices[Iv]
                            col_local = lin_indices[Iu]

                            row_global = row_local + offset
                            col_global = col_local + offset

                            add_to_sparse!(A, row_global, col_global, weight)
                        end
                    end
                end
            end
        end
    end
    return A
end

"""
    assemble!(A::SparseMatrixCSC, form::BilinearForm; dirichlet_labels = nothing, ast = resolve_form_ast(form))

Performs sequential assembly of the `BilinearForm` directly into the preallocated sparse matrix `A`.
"""
function assemble!(
        A::SparseMatrixCSC, form::BilinearForm{D, TrialSpace, TestSpace, ExprType, FType};
        dirichlet_labels = nothing,
        ast = resolve_form_ast(form)) where {D, TrialSpace, TestSpace, ExprType, FType}
    _validate_dirichlet_labels(dirichlet_labels)
    fill!(nonzeros(A), 0.0)
    space = form.trial_space
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    lin_indices = LinearIndices(indices(Ωₕ))

    _assemble_bilinear_core!(A, space, ast, lin_indices, mesh_markers)

    apply_dirichlet_labels!(A, form, dirichlet_labels)
    return A
end

"""
    assemble_parallel!(A::SparseMatrixCSC, form::BilinearForm, ast = resolve_form_ast(form))

Performs multi-threaded parallel assembly using lock-free multi-coloring partition, strictly allocation-free at runtime.
"""
function assemble_parallel!(
        A::SparseMatrixCSC, form::BilinearForm{D, TrialSpace, TestSpace, ExprType, FType},
        ast = resolve_form_ast(form)) where {D, TrialSpace, TestSpace, ExprType, FType}

    # Reset tracking arrays in place without reallocating
    fill!(nonzeros(A), 0.0)

    space = form.trial_space
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    lin_indices = LinearIndices(indices(Ωₕ))

    # Retrieve preallocated coloring groups directly from the workspace field
    color_groups = form.workspace.color_groups

    _assemble_bilinear_parallel_core!(
        A, space, ast, lin_indices, mesh_markers, color_groups)

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
