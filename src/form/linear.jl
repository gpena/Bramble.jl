
# ==============================================================================
# Struct Definitions
# ==============================================================================

# `ParallelWorkspace` lives in form/parallel_workspace.jl, included before both
# assembly files, so this file no longer depends on bilinear.jl for it.
"""
    LinearForm{D,TestSpace,ExprType,FType}

Represents a linear form defined over a test space.

# Fields
- `test_space::TestSpace`: The space for the test function.
- `ast::ExprType`: The symbolic expression AST representation of the form.
- `f::FType`: The user-defined lambda function representing the form.
- `workspace::ParallelWorkspace{D}`: Preallocated coordinate partitions for lock-free parallel assembly.
"""
struct LinearForm{D, TestSpace, ExprType <: LazyOp{D}, FType}
    test_space::TestSpace
    ast::ExprType
    f::FType
    workspace::ParallelWorkspace{D}
end

"""
    test_space(form::LinearForm)

Returns the test space of the linear form.
"""
test_space(form::LinearForm) = form.test_space

@inline (form::LinearForm)(v) = dot(assemble(form), v)

"""
    resolve_form_ast(form::LinearForm)

Fully resolves grid coefficient functions and scales inside the linear form's AST.
"""
@inline resolve_form_ast(form::LinearForm{D, TestSpace, ExprType,
    FType}) where {D, TestSpace, ExprType, FType} = resolve_ast(form.f(TestFunction{D}()))

"""
    form(Wₕ, f; stride_multiplier::Int = 1)

Constructs a `LinearForm` over the test space `Wₕ` using the linear expression `f`.

The constructor evaluates the stencil of the operator at a representative node to compute the safe multi-coloring stride needed for race-free parallel assembly.

# Examples
```julia
# 1D linear form: l(v) = (f, v)
l = form(Wh, v -> innerₕ(fh, v))
```
"""
function form(Wₕ, f; stride_multiplier::Int = 1)
    D = dim(Wₕ)
    ast = f(TestFunction{D}())

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

    # For a linear form, the stencil yields (off_v, weight) tuples
    first_off_v = sample_stencil[1][1]
    min_v = first_off_v
    max_v = first_off_v

    for (off_v, _) in sample_stencil
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

    # Preallocate thread-local buffers for parallel reduction
    num_threads = Threads.nthreads()
    thread_buffers = [zeros(Float64, ndofs(Wₕ)) for _ in 1:num_threads]
    workspace = ParallelWorkspace{D}(color_groups, thread_buffers)

    return LinearForm{D, typeof(Wₕ), typeof(ast), typeof(f)}(Wₕ, ast, f, workspace)
end

# ==============================================================================
# Assembly Implementations
# ==============================================================================

# Nothing to do unless labels were named. `dirichlet_conditions` defaults to `nothing`
# rather than to an empty constraint set, because the empty set was built on every call and
# then discarded by exactly this test — 2,080 B per assembly, for an argument that went
# unread. Naming labels without conditions is a usage error and says so.
function apply_dirichlet_conditions!(
        b::AbstractVector, form::LinearForm, dirichlet_conditions, dirichlet_labels)
    dirichlet_labels === nothing && return b

    dirichlet_conditions === nothing && _throw_labels_without_conditions(dirichlet_labels)

    if dirichlet_labels isa Symbol
        dirichlet_bc!(b, test_space(form), dirichlet_conditions, dirichlet_labels)
    elseif dirichlet_labels isa Tuple && !isempty(dirichlet_labels)
        dirichlet_bc!(b, test_space(form), dirichlet_conditions, dirichlet_labels...)
    end
    return b
end

@noinline function _throw_labels_without_conditions(labels)
    throw(ArgumentError(
        "dirichlet_labels = $labels was given without dirichlet_conditions. Pass the " *
        "constraints as well: assemble(form; dirichlet_conditions = bcs, " *
        "dirichlet_labels = $labels)."))
end

"""
    assemble(form::LinearForm; dirichlet_conditions = nothing, dirichlet_labels = nothing)

Assembles the system vector of the `LinearForm`, applying `dirichlet_conditions` on the
regions `dirichlet_labels` names when both are given.

Assembles serially. It used to call [`assemble_parallel!`](@ref), and that was slower at
every size measured — 76x at 2,500 degrees of freedom, 3.0x at 250,000, 2.7x at 810,000,
on four threads. The ratio narrows with size but does not cross over, and the reason is
structural rather than a tuning problem: that routine gives each thread a full-length
buffer and reduces them afterwards, so it pays O(n · threads) of memory traffic against
the O(n · stencil) the assembly itself costs. A linear form's stencil is one to three
entries, so the reduction dominates however large the problem gets.
"""
function assemble(form::LinearForm; dirichlet_conditions = nothing,
        dirichlet_labels = nothing)
    b = zeros(eltype(test_space(form)), ndofs(test_space(form)))
    return assemble!(b, form; dirichlet_conditions = dirichlet_conditions,
        dirichlet_labels = dirichlet_labels)
end

# ==============================================================================
# Helper Cores for Function Barrier Optimization
# ==============================================================================

function _assemble_linear_core!(
        b::Vector, space, ast::AST_TYPE, lin_indices, mesh_markers) where {AST_TYPE}
    for I in indices(mesh(space))
        lin_idx = lin_indices[I]
        stencil = local_stencil(ast, space, I, mesh_markers, lin_idx)

        for (off_v, weight) in stencil
            Iv = I + CartesianIndex(off_v)

            if checkbounds(Bool, lin_indices, Iv)
                row = lin_indices[Iv]
                @inbounds b[row] += weight
            end
        end
    end
    return b
end

function _assemble_linear_parallel_core!(b::Vector, space, ast::AST_TYPE, lin_indices,
        mesh_markers, thread_buffers) where {AST_TYPE}
    num_threads = Threads.nthreads()

    # Reset existing thread-local buffers in parallel
    Threads.@threads for t in 1:num_threads
        fill!(thread_buffers[t], 0.0)
    end

    # Parallel loop over the grid's contiguous indices with coarse chunking
    grid_inds = indices(mesh(space))
    len = length(grid_inds)
    chunk_size = ceil(Int, len / num_threads)

    Threads.@threads for tid in 1:num_threads
        local_b = thread_buffers[tid]
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = min(tid * chunk_size, len)

        for idx in start_idx:end_idx
            I = grid_inds[idx]
            lin_idx = lin_indices[I]
            stencil = local_stencil(ast, space, I, mesh_markers, lin_idx)

            for (off_v, weight) in stencil
                Iv = I + CartesianIndex(off_v)

                if checkbounds(Bool, lin_indices, Iv)
                    row = lin_indices[Iv]
                    @inbounds local_b[row] += weight
                end
            end
        end
    end

    # Reduce thread-local buffers into the destination vector b in parallel
    Threads.@threads for row in 1:length(b)
        val = 0.0
        for t in 1:num_threads
            @inbounds val += thread_buffers[t][row]
        end
        @inbounds b[row] = val
    end
    return b
end

function _assemble_linear_core!(b::Vector, space::CompositeGridSpace{N}, ast::AST_TYPE,
        lin_indices, mesh_markers) where {N, AST_TYPE}
    # `leaf_spaces_offsets` rather than a Vector of offsets accumulated with `push!`, for
    # two reasons. It allocates nothing, where the Vector cost 128 B on every call — small,
    # but this is what a time loop calls each step. And it walks the *leaves*, where
    # `space.spaces` walks the top-level components: for a composite whose component is
    # itself composite, `indices(mesh(sp))` covers one leaf's grid while `ndofs(sp)` counts
    # the whole nested block, so the old loop wrote into a fraction of the range it had
    # reserved.
    for (sp, offset) in leaf_spaces_offsets(space)
        for I in indices(mesh(sp))
            lin_idx = lin_indices[I]
            stencil = local_stencil(ast, sp, I, mesh_markers, lin_idx)

            for (off_v, weight) in stencil
                Iv = I + CartesianIndex(off_v)

                if checkbounds(Bool, lin_indices, Iv)
                    row_local = lin_indices[Iv]
                    row_global = row_local + offset
                    @inbounds b[row_global] += weight
                end
            end
        end
    end
    return b
end

function _assemble_linear_parallel_core!(
        b::Vector, space::CompositeGridSpace{N}, ast::AST_TYPE,
        lin_indices, mesh_markers, thread_buffers) where {N, AST_TYPE}
    num_threads = Threads.nthreads()

    # Reset existing thread-local buffers in parallel
    Threads.@threads for t in 1:num_threads
        fill!(thread_buffers[t], 0.0)
    end

    # See the serial core for why this is `leaf_spaces_offsets` and not an accumulated
    # Vector of offsets.
    leaves = leaf_spaces_offsets(space)

    # Parallel loop over the grid's contiguous indices with coarse chunking
    grid_inds = indices(mesh(first(first(leaves))))
    len = length(grid_inds)
    chunk_size = ceil(Int, len / num_threads)

    Threads.@threads for tid in 1:num_threads
        local_b = thread_buffers[tid]
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = min(tid * chunk_size, len)

        for idx in start_idx:end_idx
            I = grid_inds[idx]
            lin_idx = lin_indices[I]

            for (sp, offset) in leaves
                stencil = local_stencil(ast, sp, I, mesh_markers, lin_idx)

                for (off_v, weight) in stencil
                    Iv = I + CartesianIndex(off_v)

                    if checkbounds(Bool, lin_indices, Iv)
                        row_local = lin_indices[Iv]
                        row_global = row_local + offset
                        @inbounds local_b[row_global] += weight
                    end
                end
            end
        end
    end

    # Reduce thread-local buffers into the destination vector b in parallel
    Threads.@threads for row in 1:length(b)
        val = 0.0
        for t in 1:num_threads
            @inbounds val += thread_buffers[t][row]
        end
        @inbounds b[row] = val
    end
    return b
end

# `ast` is a keyword so a caller assembling repeatedly can resolve once and hand it in;
# resolving is 160 B per call otherwise. It is not cached on the form on purpose: a
# `GridFunctionScale` over a thunk has its values read when the AST resolves, so resolving
# eagerly at construction would freeze coefficients a caller may still be changing.
function assemble!(b::Vector, form::LinearForm{D, TestSpace, ExprType, FType};
        dirichlet_conditions = nothing,
        dirichlet_labels = nothing,
        ast = resolve_form_ast(form)) where {D, TestSpace, ExprType, FType}
    _validate_dirichlet_labels(dirichlet_labels)
    fill!(b, 0.0)
    space = form.test_space
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    lin_indices = LinearIndices(indices(Ωₕ))

    _assemble_linear_core!(b, space, ast, lin_indices, mesh_markers)

    apply_dirichlet_conditions!(b, form, dirichlet_conditions, dirichlet_labels)
    return b
end

"""
    assemble_parallel!(b::Vector, form::LinearForm, ast = resolve_form_ast(form))

Performs multi-threaded parallel assembly of the `LinearForm` into `b` using lock-free multi-coloring partitions.
"""
function assemble_parallel!(b::Vector, form::LinearForm{D, TestSpace, ExprType, FType},
        ast = resolve_form_ast(form)) where {D, TestSpace, ExprType, FType}
    space = form.test_space
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    lin_indices = LinearIndices(indices(Ωₕ))

    thread_buffers = form.workspace.thread_buffers

    # Safeguard against dynamic thread count changes (ensure enough buffers exist)
    while length(thread_buffers) < Threads.nthreads()
        push!(thread_buffers, zeros(Float64, length(b)))
    end

    _assemble_linear_parallel_core!(
        b, space, ast, lin_indices, mesh_markers, thread_buffers)

    return b
end

# update_ast_grid_coefficients! has been deprecated and deleted.
