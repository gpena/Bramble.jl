# block_extract.jl
# Reading which block of a coupled form a term belongs to.
#
# Determines component routing for coupled assembly: trial and test component indices
# inspected via `trial_component_or_nothing` and `test_component_or_nothing`, which
# `block_of` converts to a block coordinate `(trial, test)` or validates. Routing by
# leaf index supports arbitrary composite function spaces.

"""
    test_component_or_nothing(op) -> Union{Int, Nothing}

The component `op` is written against, or `nothing` when it names none.

Used when assembling composite right-hand sides, where a term built from an indexed test
function belongs to one block while a term built from an unindexed one belongs to all of them.
Written as a query rather than catching an exception because it is evaluated once per
term per assembly in time-stepping loops.
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
# A sum inside one inner product, `innerₕ(uₕ, v + 2 * D₋ₓ(v) - M₋ₓ(v))`, is still one
# term of the form, and every test leaf in it names the same component or none. So the
# component of a sum is the component its sides agree on.
#
# Without this check the fallback returned `nothing` and the term broadcast to every block.
function test_component_or_nothing(op::OperatorAdd)
    l = test_component_or_nothing(op.left_op)
    r = test_component_or_nothing(op.right_op)
    l === r && return l
    return _throw_mixed_components(l, r)
end

# Sides naming different components cannot be one term of one block:
# `innerₕ(uₕ, v(1) + v(2))` is ill-formed, so an error is raised.
@noinline function _throw_mixed_components(l, r)
    throw(ArgumentError(
        "the two sides of a sum inside one inner product name different components " *
        "($l and $r). Each inner product belongs to one component: write the sum of " *
        "products instead, innerₕ(u, v(1)) + innerₕ(u, v(2))."))
end

test_component_or_nothing(::Any) = nothing

"""
    trial_component_or_nothing(op) -> Union{Int, Nothing}

The trial component `op` is written against, or `nothing` when it names none.

The trial-side mirror of [`test_component_or_nothing`](@ref), and needed for the same
reason one step further on: a bilinear form's term belongs to a *block*, which takes a
component from each side. A term naming neither is the same integrand in every diagonal
block; a term naming both is one block; and a term naming one but not the other is not
something the mathematics can express, so it is an error rather than a guess.
"""
trial_component_or_nothing(op::IndexedTrialFunction) = op.component_idx
trial_component_or_nothing(op::BilinearProduct) = trial_component_or_nothing(op.left_op)
trial_component_or_nothing(op::BackwardDifference) = trial_component_or_nothing(op.inner_op)
trial_component_or_nothing(op::ForwardDifference) = trial_component_or_nothing(op.inner_op)
trial_component_or_nothing(op::CenteredDifference) = trial_component_or_nothing(op.inner_op)
trial_component_or_nothing(op::StarDifference) = trial_component_or_nothing(op.inner_op)
function trial_component_or_nothing(op::CrossWeightedDifference)
    trial_component_or_nothing(op.inner_op)
end
trial_component_or_nothing(op::JumpNode) = trial_component_or_nothing(op.inner_op)
trial_component_or_nothing(op::BackwardAverage) = trial_component_or_nothing(op.inner_op)
trial_component_or_nothing(op::ForwardAverage) = trial_component_or_nothing(op.inner_op)
trial_component_or_nothing(op::ShiftNode) = trial_component_or_nothing(op.inner_op)
trial_component_or_nothing(op::OperatorScale) = trial_component_or_nothing(op.inner_op)
trial_component_or_nothing(op::GridFunctionScale) = trial_component_or_nothing(op.inner_op)
trial_component_or_nothing(op::RegionRestriction) = trial_component_or_nothing(op.inner_op)

function trial_component_or_nothing(op::OperatorAdd)
    l = trial_component_or_nothing(op.left_op)
    r = trial_component_or_nothing(op.right_op)
    l === r && return l
    return _throw_mixed_components(l, r)
end

trial_component_or_nothing(::Any) = nothing

"""
    block_of(term, nblocks_trial, nblocks_test) -> Union{Nothing, Tuple{Int, Int}}

The `(trial, test)` block `term` belongs to, or `nothing` when it belongs to every diagonal
block.

A term naming neither side is the same integrand on each block, which for a matrix means the
diagonal: `Σᵢ innerₕ(uᵢ, vᵢ)` is block diagonal, not full. A term naming both is one block.
A term naming one and not the other is refused: `innerₕ(u(1), v)` is not something written
in a variational formulation, and reading it as a whole row or column of blocks would be a
guess about what was meant.
"""
function block_of(term, nblocks_trial::Int, nblocks_test::Int)
    tc = trial_component_or_nothing(term)
    sc = test_component_or_nothing(term)

    tc === nothing && sc === nothing && return nothing
    (tc === nothing || sc === nothing) && _throw_half_named_block(tc, sc)

    1 <= tc <= nblocks_trial || _throw_block_out_of_range("trial", tc, nblocks_trial)
    1 <= sc <= nblocks_test || _throw_block_out_of_range("test", sc, nblocks_test)
    return (tc, sc)
end

@noinline function _throw_half_named_block(tc, sc)
    named, missing_side = tc === nothing ? ("test", "trial") : ("trial", "test")
    throw(ArgumentError(
        "a term of this form names its $named component but not its $missing_side one. " *
        "A block of a bilinear form takes a component from each side: write " *
        "innerₕ(u(i), v(j)) for one block, or innerₕ(u, v) for the same integrand on " *
        "every diagonal block."))
end

@noinline function _throw_block_out_of_range(side::String, c::Int, n::Int)
    throw(ArgumentError(
        "a term of this form names $side component $c, and that side has $n blocks. " *
        "Components are numbered 1 to $n; a term written for a wider space contributes " *
        "nothing here, which is why this is an error rather than an empty block."))
end

"""
    Block{TrialLeaf, TestLeaf}

One leaf-space pair's rectangle within a composite system matrix: the concrete trial and
test leaf spaces a term couples, and the row/column offset each contributes to that
rectangle's position in the assembled matrix.

Matrix rows are indexed by the test function (see `bilinear.jl`'s file header), so
`row_offset` always comes from `test_leaf`'s offset in `leaf_spaces_offsets`, and
`col_offset` from `trial_leaf`'s. That asymmetry used to be carried by convention across
six call sites, each unpacking a bare `(tc, sc)` tuple into `first`/`last` calls in the
right order -- one of which got it backwards (gpena/Bramble.jl#48). Naming it here means a
caller reads `blk.row_offset`/`blk.col_offset` off the type instead of re-deriving which
positional element means which.
"""
struct Block{TrialLeaf, TestLeaf}
    trial_leaf::TrialLeaf
    test_leaf::TestLeaf
    row_offset::Int
    col_offset::Int
end

@inline _block_from_indices(trial_leaves, test_leaves, tc::Int, sc::Int) = Block(
    first(trial_leaves[tc]), first(test_leaves[sc]),
    last(test_leaves[sc]), last(trial_leaves[tc]))

@inline function _diagonal_blocks(trial_leaves::Tuple, test_leaves::Tuple)
    n = min(length(trial_leaves), length(test_leaves))
    return ntuple(c -> _block_from_indices(trial_leaves, test_leaves, c, c), n)
end

"""
    blocks(term, trial_leaves, test_leaves) -> Tuple{Vararg{Block}}

Every [`Block`](@ref) `term` must be assembled into.

A term naming both sides (via [`block_of`](@ref)) resolves to the one `Block` it names. A
term naming neither is the same integrand on every diagonal block, so it resolves to one
`Block` per diagonal leaf pair. `trial_leaves`/`test_leaves` are `leaf_spaces_offsets`
results.
"""
@inline function blocks(term, trial_leaves::Tuple, test_leaves::Tuple)
    blk = block_of(term, length(trial_leaves), length(test_leaves))
    blk === nothing && return _diagonal_blocks(trial_leaves, test_leaves)
    tc, sc = blk
    return (_block_from_indices(trial_leaves, test_leaves, tc, sc),)
end

"""
    routes_by_component(op) -> Bool

Whether `op` names components anywhere inside it, and so has to be split across the blocks
of a composite space rather than assembled into all of them.

Asked once per assembly, before any splitting happens, so that a form written without
component indices keeps the path that allocates nothing.
"""
function routes_by_component(op::OperatorAdd)
    routes_by_component(op.left_op) ||
        routes_by_component(op.right_op)
end
routes_by_component(op) = test_component_or_nothing(op) !== nothing

"""
    _collect_region_labels(op) -> NTuple{N, Symbol}

Every marker label a `RegionRestriction` anywhere in `op` names: from `restrict_to` calls
written directly, or from the `markers = (...)` keyword on `innerₕ`/`inner₊` and friends.
Flattened into one tuple; a term naming several restrictions (nested, or one on each side of
a product) reports all of them, since every one has to exist on every leaf the term reaches
for assembly to mean what it says.

Recurses the same way [`trial_component_or_nothing`](@ref)/[`test_component_or_nothing`](@ref)
do, so a marker nested behind any operator those already see through is found here too.
"""
function _collect_region_labels(op::RegionRestriction)
    (
        _region_labels(op.region)..., _collect_region_labels(op.inner_op)...)
end

_region_labels(region::Symbol) = (region,)
_region_labels(region::NTuple{N, Symbol}) where {N} = region

for W in (:BackwardDifference, :ForwardDifference, :CenteredDifference,
    :StarDifference, :CrossWeightedDifference, :BackwardAverage,
    :ForwardAverage, :ShiftNode, :JumpNode, :OperatorScale, :GridFunctionScale)
    @eval _collect_region_labels(op::$W) = _collect_region_labels(op.inner_op)
end

function _collect_region_labels(op::Union{BilinearProduct, LinearProduct, OperatorAdd})
    (_collect_region_labels(op.left_op)..., _collect_region_labels(op.right_op)...)
end

_collect_region_labels(op) = ()

"""
    _validate_term_markers(term, mesh_markers, context::String)

Throws if `term` names, via `restrict_to` or `markers = (...)`, a label that does not exist
in `mesh_markers` (the mesh a term is about to be scattered against). Checked once, while the
sparsity pattern is built (`allocate_system_matrix`/`_pattern_term!`), rather than left to
`RegionRestriction`'s own `local_stencil`: that answers `false` for a missing key the same way
it does for "not marked", so a typo'd or leaf-missing label would otherwise assemble to a
silent all-zero contribution instead of failing loudly.
"""
function _validate_term_markers(term, mesh_markers, context::String)
    for label in _collect_region_labels(term)
        haskey(mesh_markers, label) || _throw_marker_not_on_space(label, context)
    end
    return nothing
end

@noinline function _throw_marker_not_on_space(label::Symbol, context::String)
    throw(ArgumentError(
        "the marker :$label is not defined on $context. A marker named in restrict_to or " *
        "markers = (...) must exist on every space a term reaches; if it is only defined " *
        "on some of a composite space's leaves, write the term per component instead, one " *
        "innerₕ(u(i), v(i)) per leaf with that leaf's own markers, rather than one term " *
        "naming a marker not every leaf it reaches has."))
end
