# ==============================================================================
# Struct Definitions
# ==============================================================================

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

# `a(uₕ, vₕ) = vᵀ A u`. Assembles a whole matrix per call, which is what makes it a
# convenience rather than something to put in a loop.
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

Constructs a `BilinearForm` over the trial space `Wₕ` and the test space `Vₕ` from the
bilinear expression `f`, a function of a trial and a test argument.

A composite space needs no separate constructor and no separate type. Its blocks are
addressed by leaf index, `u(i)` and `v(j)`, whatever the nesting: `leaf_spaces_offsets`
flattens a space of spaces into leaves, so a two-by-two nesting is four blocks numbered one
to four. A term naming neither side is the same integrand on every diagonal block, and one
naming both is the single block it belongs to, off-diagonal included.

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

    # Built and discarded, for the error rather than for the tree: an expression that does
    # not describe an operator fails here rather than at the first `assemble`. Nothing is
    # kept, because assembly resolves from `f` every time — see `LinearForm`.
    _validate_form_expression(f(TrialFunction{D}(), TestFunction{D}()), Val(D))

    return BilinearForm{D, typeof(Wₕ), typeof(Vₕ), typeof(f)}(Wₕ, Vₕ, f)
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

# Whether an earlier entry of this stencil already named the same pair of offsets.
#
# The pattern wants each position once, however many terms write a value there. A form like
# `innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v))` has both products contributing at `(0, 0)`, so
# its stencil is five entries over four distinct pairs — a fifth of the coordinates it
# generated were the same entry twice, and `sparse!` then paid to sort and merge them.
#
# A linear scan over the entries already seen, which is the whole stencil: a handful, not a
# collection worth a set.
@inline function _offsets_seen_before(stencil, k::Int, off_u, off_v)
    @inbounds for l in 1:(k - 1)
        stencil[l][1] == off_u && stencil[l][2] == off_v && return true
    end
    return false
end

# How many entries the pattern can hold at most: one interior stencil's worth per grid
# point. An upper bound, because truncation at a boundary drops entries and never adds any.
#
# Worth computing because `push!` grows by doubling: filling two 1,250,000-element vectors
# cost 28.4 MB each where the data is 9.5 MB, so a third of the pattern's memory was
# reallocation. `sizehint!` is the whole fix for that part.
function _pattern_upper_bound(ast::AST_TYPE, sp, mesh_markers, lin_indices) where {AST_TYPE}
    grid_inds = indices(mesh(sp))
    npts = length(grid_inds)
    I = grid_inds[length(grid_inds) ÷ 2 + 1]
    return npts * length(local_stencil(ast, sp, I, mesh_markers, lin_indices[I]))
end

"""
    allocate_system_matrix(form::BilinearForm, ast = resolve_form_ast(form)) -> SparseMatrixCSC

Builds the sparse matrix a `BilinearForm` assembles into: the right size, the right sparsity
pattern, and every stored value zero.

The pattern follows from the stencil rather than from the numbers, so it is known before any
value is computed, and it does not change while the mesh and the expression do not. That is
what makes the two-step idiom worth using — build the pattern once, outside a time loop, and
refill it inside:

```julia
A = allocate_system_matrix(a)
for step in 1:nsteps
    assemble!(A, a)          # refills the values, allocates nothing
    # …
end
```

against calling [`assemble`](@ref) each step, which allocates a new matrix every time.

Only the structure is computed here. The entries are all zero on return, so a matrix from
this is not usable until `assemble!` has filled it.

See also [`assemble`](@ref) and [`assemble!`](@ref).
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
    # `sparse!` rather than `sparse`: it uses the coordinate vectors as its own scratch
    # instead of copying them, and they are discarded here either way. Worth 13.4 MB of the
    # 64.9 the pattern cost at 250,000 degrees of freedom.
    V_vec = _zeros_of(_assembled_eltype(ast, space), length(I_vec))
    return sparse!(I_vec, J_vec, V_vec, n, n, +)
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

        for k in eachindex(stencil)
            off_u, off_v, _ = stencil[k]
            _offsets_seen_before(stencil, k, off_u, off_v) && continue

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

    # One block's worth per block is the bound: a term reaching every diagonal block reaches
    # as many as there are leaves, and one naming a block reaches exactly one.
    sp = first(first(test_leaves))
    Ωₛ = mesh(sp)
    hint = length(test_leaves) *
           _pattern_upper_bound(ast, sp, markers(Ωₛ), LinearIndices(indices(Ωₛ)))
    sizehint!(I_vec, hint)
    sizehint!(J_vec, hint)

    _pattern_blocks!(I_vec, J_vec, ast, trial_leaves, test_leaves)

    ncols = ndofs(form.trial_space)
    nrows = ndofs(form.test_space)
    V_vec = _zeros_of(_assembled_eltype(ast, first_space(form.trial_space)), length(I_vec))
    return sparse!(I_vec, J_vec, V_vec, nrows, ncols, +)
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

Allocates a matrix with the form's sparsity pattern and assembles into it.

**Call this once, then assemble into what it returns.** Building the sparsity pattern is by
far the larger half of the work — at 250,000 degrees of freedom it is 9,700 us and 52 MB
against 1,500 us and nothing to fill the matrix — and the pattern does not change between
assemblies. So a time loop or a Newton iteration written with `assemble` pays for the same
pattern on every step:

```julia
A = assemble(a)                        # once: pattern, allocation and the first fill
for step in 1:nsteps
    Rₕ!(cₕ, coefficient_at(step))      # written through, so the form still sees it
    assemble!(A, a)                    # or assemble_parallel!(A, a)
end
```

which is seven times cheaper per step. It assumes the pattern stays the same, which it does
as long as the form's operators do: a coefficient changes values, not which entries exist.

`assemble!` also takes an `ast`, and for a form like the one above there is no reason to use
it — resolving is 11 us against 550 to assemble, and 288 bytes. It earns its place in one
case, where a *coefficient* carries an operator: `innerₕ(D₋ₓ(cₕ) * u, v)` recomputes
`D₋ₓ(cₕ)` on every resolve, 2 MB of it, where the same operator on the test argument is
symbolic and free. Hoisting is the better answer there, and keeps the form live —

```julia
dcₕ = D₋ₓ(cₕ)                          # computed once, and still written through
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(dcₕ * u, v))
```

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
    assemble_parallel!(A::SparseMatrixCSC, form::BilinearForm, ast = resolve_form_ast(form)) -> A

Refills `A` with the assembled `form` across threads, and returns it. `A` must already carry
the right sparsity pattern, from [`allocate_system_matrix`](@ref) or a previous
[`assemble`](@ref).

Colouring is what makes this correct rather than merely fast. Filling an entry goes through a
column search and an in-place update, so two threads landing on the same entry would race on
the *value*, not just on the structure.

A matrix colours on the **test side alone**. A bilinear stencil writes to
`(I + off_v, I + off_u)`, so two points collide on an entry only if their row footprints
overlap — rows disjoint implies entries disjoint, whatever the columns do — which is the same
span a vector assembly uses.

Takes `ast` positionally, matching the vector form and unlike the keyword on
[`assemble!`](@ref).
"""
function assemble_parallel!(
        A::SparseMatrixCSC, form::BilinearForm{D, TrialSpace, TestSpace, FType},
        ast = resolve_form_ast(form)) where {D, TrialSpace, TestSpace, FType}
    fill!(nonzeros(A), zero(eltype(nonzeros(A))))

    _assemble_bilinear_parallel_core!(
        A, form.trial_space, form.test_space, ast, Val(D))

    return A
end
