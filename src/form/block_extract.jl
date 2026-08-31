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

A `BilinearProduct` is searched on its **left**, and `find_test_component` searches the
right: a coupled form is written trial first, test second, as `innerₕ(u, v)`. Writing the
two the other way round is not detected and cannot be — both sides are operators, and which
one is the trial is exactly what the indexed leaf records.
"""
find_trial_component(op::IndexedTrialFunction) = op.component_idx
find_trial_component(op::BackwardDifference) = find_trial_component(op.inner_op)
find_trial_component(op::ForwardDifference) = find_trial_component(op.inner_op)
find_trial_component(op::OperatorScale) = find_trial_component(op.inner_op)
find_trial_component(op::GridFunctionScale) = find_trial_component(op.inner_op)
find_trial_component(op::BackwardAverage) = find_trial_component(op.inner_op)
find_trial_component(op::ForwardAverage) = find_trial_component(op.inner_op)
find_trial_component(op::RegionRestriction) = find_trial_component(op.inner_op)
find_trial_component(op::CenteredDifference) = find_trial_component(op.inner_op)
find_trial_component(op::StarDifference) = find_trial_component(op.inner_op)
find_trial_component(op::CrossWeightedDifference) = find_trial_component(op.inner_op)
find_trial_component(op::JumpNode) = find_trial_component(op.inner_op)
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
find_test_component(op::CenteredDifference) = find_test_component(op.inner_op)
find_test_component(op::StarDifference) = find_test_component(op.inner_op)
find_test_component(op::CrossWeightedDifference) = find_test_component(op.inner_op)
find_test_component(op::JumpNode) = find_test_component(op.inner_op)
find_test_component(op::BilinearProduct) = find_test_component(op.right_op)
# A linear form's term is a LinearProduct, and its test function is on the right just as a
# bilinear product's is. Without this a coupled *right-hand side* could not be routed to its
# block at all, though the machinery for the matrix was already here.
find_test_component(op::LinearProduct) = find_test_component(op.right_op)

"""
    test_component_or_nothing(op) -> Union{Int, Nothing}

The component `op` is written against, or `nothing` when it names none.

The counterpart of [`find_test_component`](@ref) for code that has to *ask* rather than
assume — assembling a composite right-hand side, where a term built from an indexed test
function belongs to one block while a term built from a plain one belongs to all of them.
Written out rather than wrapping the throwing form in a `try`, because it is asked once per
term per assembly and an assembly happens every step of a time loop.
"""
test_component_or_nothing(op::IndexedTestFunction) = op.component_idx
test_component_or_nothing(op::LinearProduct) = test_component_or_nothing(op.right_op)
test_component_or_nothing(op::BilinearProduct) = test_component_or_nothing(op.right_op)
test_component_or_nothing(op::BackwardDifference) = test_component_or_nothing(op.inner_op)
test_component_or_nothing(op::ForwardDifference) = test_component_or_nothing(op.inner_op)
test_component_or_nothing(op::CenteredDifference) = test_component_or_nothing(op.inner_op)
test_component_or_nothing(op::StarDifference) = test_component_or_nothing(op.inner_op)
function test_component_or_nothing(op::CrossWeightedDifference)
    test_component_or_nothing(op.inner_op)
end
test_component_or_nothing(op::JumpNode) = test_component_or_nothing(op.inner_op)
test_component_or_nothing(op::BackwardAverage) = test_component_or_nothing(op.inner_op)
test_component_or_nothing(op::ForwardAverage) = test_component_or_nothing(op.inner_op)
test_component_or_nothing(op::ShiftNode) = test_component_or_nothing(op.inner_op)
test_component_or_nothing(op::OperatorScale) = test_component_or_nothing(op.inner_op)
test_component_or_nothing(op::GridFunctionScale) = test_component_or_nothing(op.inner_op)
test_component_or_nothing(op::RegionRestriction) = test_component_or_nothing(op.inner_op)
# A sum *inside* one inner product — `innerₕ(uₕ, v + 2 * D₋ₓ(v) - M₋ₓ(v))` — is still one
# term of the form, and every test leaf in it names the same component or none. So the
# component of a sum is the component its sides agree on.
#
# Without this the `::Any` fallback answered `nothing` and the term broadcast to every block,
# so a coupled form with an operator sum in its test slot put every component's source into
# every block. It summed to something plausible and was wrong.
function test_component_or_nothing(op::OperatorAdd)
    l = test_component_or_nothing(op.left_op)
    r = test_component_or_nothing(op.right_op)
    l === r && return l
    return _throw_mixed_components(l, r)
end

# Sides naming different components cannot be one term of one block. It is ill-formed rather
# than ambiguous — `innerₕ(uₕ, v(1) + v(2))` is not a component of anything — and silence
# here is what produced the bug above, so it is an error.
@noinline function _throw_mixed_components(l, r)
    throw(ArgumentError(
        "the two sides of a sum inside one inner product name different components " *
        "($l and $r). Each inner product belongs to one component: write the sum of " *
        "products instead, innerₕ(u, v(1)) + innerₕ(u, v(2))."))
end

test_component_or_nothing(::Any) = nothing

"""
    routes_by_component(op) -> Bool

Whether `op` names components anywhere inside it, and so has to be split across the blocks
of a composite space rather than assembled into all of them.

Asked once per assembly, before any splitting happens, so that a form written without
component indices keeps the path that allocates nothing: splitting means `flatten_sum`,
which builds a vector.
"""
function routes_by_component(op::OperatorAdd)
    routes_by_component(op.left_op) ||
        routes_by_component(op.right_op)
end
routes_by_component(op) = test_component_or_nothing(op) !== nothing

# The docstrings above promise an error when a term carries no indexed leaf, and without
# these that error is a `MethodError` naming an internal function — which happens for two
# quite different mistakes. Either the form was built from plain `TrialFunction`s rather
# than the indexed ones `make_trial_args` hands out, so it is not a coupled form at all; or
# it was written test-first, and the search reached the wrong kind of leaf on the side it
# looked at. Both are usage errors, and the message says which.
@noinline function find_trial_component(op::LazyOp)
    throw(ArgumentError(
        "no IndexedTrialFunction in this term: got $(typeof(op)). A coupled form is " *
        "built from the arguments `make_trial_args` and `make_test_args` return, and is " *
        "written trial first — innerₕ(u, v), not innerₕ(v, u)."))
end

@noinline function find_test_component(op::LazyOp)
    throw(ArgumentError(
        "no IndexedTestFunction in this term: got $(typeof(op)). A coupled form is " *
        "built from the arguments `make_trial_args` and `make_test_args` return, and is " *
        "written test second — innerₕ(u, v), not innerₕ(v, u)."))
end

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
