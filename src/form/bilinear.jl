# ==============================================================================
# Struct Definitions
# ==============================================================================

"""
    ParallelWorkspace{D}

Preallocated structure containing coordinate indices partitioned into lock-free/independent color groups.
"""
struct ParallelWorkspace{D}
    color_groups::Vector{Vector{CartesianIndex{D}}}
    thread_buffers::Vector{Vector{Float64}}

    function ParallelWorkspace{D}(color_groups::Vector{Vector{CartesianIndex{D}}}) where D
        new{D}(color_groups, Vector{Float64}[])
    end

    function ParallelWorkspace{D}(color_groups::Vector{Vector{CartesianIndex{D}}}, thread_buffers::Vector{Vector{Float64}}) where D
        new{D}(color_groups, thread_buffers)
    end
end

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
struct BilinearForm{D,TrialSpace,TestSpace,ExprType<:LazyOp{D},FType}
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

@inline (form::BilinearForm)(u, v) = dot(v, assemble(form) * u)

"""
    resolve_form_ast(form::BilinearForm)

Fully resolves grid coefficient functions and scales inside the bilinear form's AST.
"""
@inline resolve_form_ast(form::BilinearForm{D,TrialSpace,TestSpace,ExprType,FType}) where {D,TrialSpace,TestSpace,ExprType,FType} = resolve_ast(form.f(TrialFunction{D}(), TestFunction{D}()))

"""
    form(Wₕ, Vₕ, f; stride_multiplier::Int = 1)

Constructs a `BilinearForm` over the trial space `Wₕ` and test space `Vₕ` using the bilinear expression `f`.

The constructor evaluates the stencil of the operator at a representative node to compute the safe multi-coloring stride needed for race-free parallel assembly.

# Examples
```julia
# 1D Poisson bilinear form: a(u, v) = (∇u, ∇v)
a = form(Wh, Wh, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v)))
```
"""
function form(Wₕ, Vₕ, f; stride_multiplier::Int=1)
    D = dim(Wₕ)
    ast = f(TrialFunction{D}(), TestFunction{D}())

    # Extract mesh characteristics to discover stencil bounds upfront
    sp = first_space(Wₕ)
    Ωₕ = mesh(sp)
    mesh_markers = markers(Ωₕ)
    grid_inds = indices(Ωₕ)
    lin_indices = LinearIndices(grid_inds)

    # Evaluate the stencil at a representative interior node to discover off_v bounds
    center_I = grid_inds[length(grid_inds)÷2+1]
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

    return BilinearForm{D,typeof(Wₕ),typeof(Vₕ),typeof(ast),typeof(f)}(Wₕ, Vₕ, ast, f, workspace)
end

# ==============================================================================
# Utility Helpers
# ==============================================================================

@inline function add_to_sparse!(A::SparseMatrixCSC, row::Int, col::Int, val::Number)
    p1 = A.colptr[col]
    p2 = A.colptr[col+1] - 1

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
function allocate_system_matrix(form::BilinearForm{D,TrialSpace,TestSpace,ExprType,FType}, ast=resolve_form_ast(form)) where {D,TrialSpace,TestSpace,ExprType,FType}
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

function allocate_system_matrix(form::BilinearForm{D,TrialSpace,TestSpace,ExprType,FType}, ast=resolve_form_ast(form)) where {D,TrialSpace<:CompositeGridSpace,TestSpace<:CompositeGridSpace,ExprType,FType}
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
function assemble(form::BilinearForm; dirichlet_labels=nothing)
    _validate_dirichlet_labels(dirichlet_labels)
    ast_resolved = resolve_form_ast(form)
    A = allocate_system_matrix(form, ast_resolved)
    assemble_parallel!(A, form, ast_resolved)
    apply_dirichlet_labels!(A, form, dirichlet_labels)
    return A
end

"""
    assemble!(A::SparseMatrixCSC, form::BilinearForm; dirichlet_labels = nothing, ast = resolve_form_ast(form))

Performs sequential assembly of the `BilinearForm` directly into the preallocated sparse matrix `A`.
"""
# ==============================================================================
# Helper Cores for Function Barrier Optimization
# ==============================================================================

function _assemble_bilinear_core!(A::SparseMatrixCSC, space, ast::AST_TYPE, lin_indices, mesh_markers) where {AST_TYPE}
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

function _assemble_bilinear_parallel_core!(A::SparseMatrixCSC, space, ast::AST_TYPE, lin_indices, mesh_markers, color_groups) where {AST_TYPE}
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

                    if checkbounds(Bool, lin_indices, Iv) && checkbounds(Bool, lin_indices, Iu)
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

function _assemble_bilinear_core!(A::SparseMatrixCSC, space::CompositeGridSpace{N}, ast::AST_TYPE, lin_indices, mesh_markers) where {N, AST_TYPE}
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

function _assemble_bilinear_parallel_core!(A::SparseMatrixCSC, space::CompositeGridSpace{N}, ast::AST_TYPE, lin_indices, mesh_markers, color_groups) where {N, AST_TYPE}
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
        end
    end
    return A
end

function assemble!(A::SparseMatrixCSC, form::BilinearForm{D,TrialSpace,TestSpace,ExprType,FType}; dirichlet_labels=nothing, ast=resolve_form_ast(form)) where {D,TrialSpace,TestSpace,ExprType,FType}

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
function assemble_parallel!(A::SparseMatrixCSC, form::BilinearForm{D,TrialSpace,TestSpace,ExprType,FType}, ast=resolve_form_ast(form)) where {D,TrialSpace,TestSpace,ExprType,FType}

    # Reset tracking arrays in place without reallocating
    fill!(nonzeros(A), 0.0)

    space = form.trial_space
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    lin_indices = LinearIndices(indices(Ωₕ))

    # Retrieve preallocated coloring groups directly from the workspace field
    color_groups = form.workspace.color_groups

    _assemble_bilinear_parallel_core!(A, space, ast, lin_indices, mesh_markers, color_groups)

    return A
end