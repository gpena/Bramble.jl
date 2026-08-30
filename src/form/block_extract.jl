# block_extract.jl
# Utilities for decomposing a coupled bilinear form AST into per-block sub-ASTs.

# ==============================================================================
# 1. Leaf Space Collection
# ==============================================================================

"""
    collect_leaf_spaces_offsets(space::CompositeGridSpace) -> Vector{Tuple{ScalarGridSpace,Int}}

Returns a flat list of `(scalar_space, global_dof_offset)` pairs by walking the
`CompositeGridSpace` hierarchy in depth-first (left-to-right) order. The offset
is the cumulative count of DOFs from all preceding leaf spaces.
"""
# The traversal itself is a grid space concern, not a form one — it reads nothing but
# `ScalarGridSpace`, `CompositeGridSpace` and `ndofs` — and lives in
# space/vector_gridspace.jl as `leaf_spaces_offsets`, which answers with a tuple and so
# stays type stable. This is the vector-shaped view of it, for code that wants to index or
# iterate the leaves dynamically.
collect_leaf_spaces_offsets(space::CompositeGridSpace) = collect(leaf_spaces_offsets(space))

"""
    is_hierarchical(space) -> Bool

Returns `true` if `space` is a `CompositeGridSpace` with at least one component
that is itself a `CompositeGridSpace` (i.e. a two-level or deeper hierarchy).
"""
is_hierarchical(sp::ScalarGridSpace) = false
is_hierarchical(sp::CompositeGridSpace) = any(s -> s isa CompositeGridSpace, sp.spaces)

# ==============================================================================
# 2. Symbolic Argument Generation
# ==============================================================================

"""
    make_trial_args(space::CompositeGridSpace, D::Int) -> Tuple

Generates a tuple of symbolic trial arguments matching the top-level component
structure of `space`. Each top-level scalar component becomes an
`IndexedTrialFunction{D}(idx)`, and each composite component becomes a nested
tuple of indexed functions. Component indices are assigned depth-first.

# Example (Stokes)
```julia
# Vh = Wh × Wh,  X = Vh × Wh  (hierarchical: 2 top-level components)
U, P = make_trial_args(X, 2)
# U = (IndexedTrialFunction{2}(1), IndexedTrialFunction{2}(2))
# P =  IndexedTrialFunction{2}(3)
```
"""
function make_trial_args(space::CompositeGridSpace{NT}, D::Int) where {NT}
    counter = Ref(1)
    return ntuple(k -> _make_trial_arg(space.spaces[k], counter, D), Val(NT))
end

function _make_trial_arg(sp::ScalarGridSpace, counter::Ref{Int}, D::Int)
    idx = counter[]
    counter[] += 1
    return IndexedTrialFunction{D}(idx)
end

function _make_trial_arg(sp::CompositeGridSpace{N}, counter::Ref{Int}, D::Int) where {N}
    return ntuple(k -> _make_trial_arg(sp.spaces[k], counter, D), Val(N))
end

"""
    make_test_args(space::CompositeGridSpace, D::Int) -> Tuple

Same as `make_trial_args` but produces `IndexedTestFunction{D}` nodes.
"""
function make_test_args(space::CompositeGridSpace{NS}, D::Int) where {NS}
    counter = Ref(1)
    return ntuple(k -> _make_test_arg(space.spaces[k], counter, D), Val(NS))
end

function _make_test_arg(sp::ScalarGridSpace, counter::Ref{Int}, D::Int)
    idx = counter[]
    counter[] += 1
    return IndexedTestFunction{D}(idx)
end

function _make_test_arg(sp::CompositeGridSpace{N}, counter::Ref{Int}, D::Int) where {N}
    return ntuple(k -> _make_test_arg(sp.spaces[k], counter, D), Val(N))
end

# ==============================================================================
# 3. AST Block Extraction
# ==============================================================================

"""
    flatten_sum(op) -> Vector{LazyOp}

Recursively decomposes an `OperatorAdd` tree into a flat `Vector` of leaf terms
(the addends of the outermost sum). Non-`OperatorAdd` nodes are returned as a
single-element vector.
"""
function flatten_sum(op::OperatorAdd)
    return vcat(flatten_sum(op.left_op), flatten_sum(op.right_op))
end
function flatten_sum(op::LazyOp)
    return Any[op]
end

"""
    find_trial_component(op) -> Int

Walks a bilinear term AST to find the `IndexedTrialFunction` leaf and returns
its `component_idx`. Raises an error if no indexed trial function is found.
"""
find_trial_component(op::IndexedTrialFunction) = op.component_idx
find_trial_component(op::BackwardDifference) = find_trial_component(op.inner_op)
find_trial_component(op::ForwardDifference) = find_trial_component(op.inner_op)
find_trial_component(op::OperatorScale) = find_trial_component(op.inner_op)
find_trial_component(op::GridFunctionScale) = find_trial_component(op.inner_op)
find_trial_component(op::BackwardAverage) = find_trial_component(op.inner_op)
find_trial_component(op::ForwardAverage) = find_trial_component(op.inner_op)
find_trial_component(op::RegionRestriction) = find_trial_component(op.inner_op)
find_trial_component(op::BilinearProduct) = find_trial_component(op.left_op)

"""
    find_test_component(op) -> Int

Walks a bilinear term AST to find the `IndexedTestFunction` leaf and returns
its `component_idx`.
"""
find_test_component(op::IndexedTestFunction) = op.component_idx
find_test_component(op::BackwardDifference) = find_test_component(op.inner_op)
find_test_component(op::ForwardDifference) = find_test_component(op.inner_op)
find_test_component(op::OperatorScale) = find_test_component(op.inner_op)
find_test_component(op::GridFunctionScale) = find_test_component(op.inner_op)
find_test_component(op::BackwardAverage) = find_test_component(op.inner_op)
find_test_component(op::ForwardAverage) = find_test_component(op.inner_op)
find_test_component(op::RegionRestriction) = find_test_component(op.inner_op)
find_test_component(op::BilinearProduct) = find_test_component(op.right_op)

"""
    extract_block_asts(ast::LazyOp{D}, NT::Int, NS::Int) -> Matrix{Any}

Decomposes the full coupled bilinear form AST into a `(NS × NT)` matrix of
per-block sub-ASTs. Each entry `[i, j]` is the part of the form involving
test component `i` (row block) and trial component `j` (column block), or
`nothing` if there is no coupling between those components.

The decomposition relies on the `IndexedTrialFunction` and `IndexedTestFunction`
leaves carried by each `BilinearProduct` term.
"""
function extract_block_asts(ast::LazyOp{D}, NT::Int, NS::Int) where {D}
    terms = flatten_sum(ast)
    blocks = Matrix{Any}(nothing, NS, NT)

    for term in terms
        j = find_trial_component(term)   # column block (trial)
        i = find_test_component(term)    # row    block (test)

        if i < 1 || i > NS || j < 1 || j > NT
            error("Block indices out of range: got ($i, $j) for a ($NS × $NT) system.")
        end

        if blocks[i, j] === nothing
            blocks[i, j] = term
        else
            # Accumulate multiple terms in the same block as an OperatorAdd
            blocks[i, j] = OperatorAdd(blocks[i, j], term)
        end
    end

    return blocks
end
