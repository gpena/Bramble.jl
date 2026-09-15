# stencil_pattern.jl
#
# Grid offsets an operator reaches, read off its AST before assembling it.
#
# Every node here reaches a fixed set of neighbours: a backward difference reads the point
# and the one before it, a cross-weighted difference reads three in a row, an average reads
# two. Those offsets are a property of the tree, not of the grid point: truncation at a
# boundary zeroes the coefficients and keeps the offsets, which is what makes the set well
# defined. So the sparsity pattern of the assembled matrix is known before a single entry
# is computed.
#
# The pattern is used for preallocation. The sparsity pattern of an assembled operator
# matches the stencil's reach, and building the backend matrix with this pattern ensures
# assembly only updates stored values without structural inserts. The offsets inform how
# the backend matrix is preallocated; they do not dictate the concrete matrix type, which
# is configured by `Backend{VT, MT}`.

"""
    stencil_offsets(op) -> Vector{NTuple{D, Int}}

The grid offsets the operator `op` reaches, sorted and without repeats.

Read from the AST rather than from an evaluated stencil, and the two agree: a truncated
point keeps its offsets and zeroes its coefficients, so the set does not vary over the grid.
The one node whose reach is not fixed by its type is [`ShiftNode`](@ref), which carries its
step as a field; the value is available here because this walks the built tree.

For a `BilinearProduct` this is the row (test-side) reach only, not the full row/column
pattern -- see the note on that method.
"""
function stencil_offsets end

# --- the leaves: a single entry on the diagonal ------------------------------------ #
_origin(::LazyOp{D}) where {D} = [ntuple(_ -> 0, D)]

stencil_offsets(op::TrialFunction) = _origin(op)
stencil_offsets(op::TestFunction) = _origin(op)
stencil_offsets(op::IndexedTrialFunction) = _origin(op)
stencil_offsets(op::IndexedTestFunction) = _origin(op)
stencil_offsets(op::SourceFunction) = _origin(op)
stencil_offsets(op::SourceVector) = _origin(op)
stencil_offsets(op::SourceConstant) = _origin(op)
stencil_offsets(op::DiracSource) = _origin(op)
stencil_offsets(op::IdentityOperator) = _origin(op)
stencil_offsets(op::ZeroOperator) = _origin(op)

# --- combining what a node's child reaches ----------------------------------------- #

# `op` reaches everything its child does, each moved by every step in `steps`.
@inline function _reach(inner::Vector, ::Val{Dim}, steps::Tuple) where {Dim}
    out = eltype(inner)[]
    for s in steps, o in inner

        p = shift_offset(o, Dim, s)
        p in out || push!(out, p)
    end
    return sort!(out)
end

# --- the operators ------------------------------------------------------------------ #

# A difference, an average or a jump reaches the point and one or two neighbours, and which
# ones is precisely the tap set its stencil is built from. That used to be written out a
# second time here, once per node, so the reach and the stencil agreed only by both being
# maintained: it now reads the node's own `_stencil_taps` (gpena/Bramble.jl#70). `_reach`
# sorts and de-duplicates, so the taps' evaluation order does not matter to it -- only the
# set does, which is why the same declaration serves both.
function stencil_offsets(op::TappedNode{D, Dim}) where {D, Dim}
    return _reach(
        stencil_offsets(op.inner_op), Val(Dim), map(_shift_delta, _stencil_taps(op))
    )
end

# a shift moves the whole reach and widens nothing
function stencil_offsets(op::ShiftNode{D, Dim}) where {D, Dim}
    return _reach(stencil_offsets(op.inner_op), Val(Dim), (op.shift_amount,))
end

# scaling changes coefficients, not reach
stencil_offsets(op::OperatorScale) = stencil_offsets(op.inner_op)
stencil_offsets(op::GridFunctionScale) = stencil_offsets(op.inner_op)

# a restriction empties the stencil outside its region and leaves it alone inside, so the
# reach is its child's wherever the operator contributes at all
stencil_offsets(op::RegionRestriction) = stencil_offsets(op.inner_op)

# A linear product contracts the left factor away (`multiply_stencils_linear` keeps only
# the right offsets), so its reach is the test side's.
stencil_offsets(op::LinearProduct) = stencil_offsets(op.right_op)

# A bilinear product's stencil entries are (row offset, column offset, value) triples, so
# the general "what does this reach" question is genuinely ambiguous between the two sets.
# The one caller that matters (`_colour_strides`, form/linear.jl, used for both the linear
# and bilinear parallel-assembly paths) only ever needs the row side, though: `add_to_sparse!`
# (form/bilinear.jl) finds a fixed nzval slot for each stored (row, col) pair, so two writes
# can only collide if both their row AND column coincide. If colouring already keeps rows
# from coinciding, the pair can't either, whatever the columns are -- so this reduces to the
# test factor's reach, the same way `stencil_offsets(::LinearProduct)` above already does.
# Building the full sparsity pattern (both sets) is a different question, answered
# separately by walking `local_stencil` directly (`_pattern_term!`, form/bilinear.jl).
stencil_offsets(op::BilinearProduct) = stencil_offsets(op.right_op)

function stencil_offsets(op::OperatorAdd)
    return sort!(union(stencil_offsets(op.left_op), stencil_offsets(op.right_op)))
end

"""
    _stencil_margin(op) -> Int

The largest `|offset|` component anywhere in `op`'s stencil, trial and test sides combined.

Unlike [`stencil_offsets`](@ref), which keeps only the test-side (row) reach at a
`BilinearProduct` because that is all its one caller (`_colour_strides`) needs, this keeps
both. It exists for `visit_bilinear_stencil`'s interior/boundary split
(`form/bilinear_traversal.jl`, gpena/Bramble.jl#160), which has to guard every offset an
entry can carry -- trial and test alike, not just the row's -- before it can skip the guard
anywhere.

Answered as a scalar rather than the offset set `stencil_offsets` returns: the traversal
only needs a margin to build a rectangular interior box from, and a scalar costs no
allocation to compute, where reusing `stencil_offsets` (built on `union`-ing `Vector`s)
would mean paying that cost inside `visit_bilinear_stencil` itself on the `ReplaySink` path,
once per `assemble!` call rather than once per form.

A composed reach past magnitude 1 is not a hypothetical this ignores: `D₋ₓ(D₋ₓ(u))` and
`Shift(u, dim, 2)` both carry offsets wider than the one step a single difference or a unit
shift would suggest, which is why this walks the same tree `stencil_offsets` does rather
than assuming every node caps out at 1.
"""
_stencil_margin(op::TrialFunction) = 0
_stencil_margin(op::TestFunction) = 0
_stencil_margin(op::IndexedTrialFunction) = 0
_stencil_margin(op::IndexedTestFunction) = 0
_stencil_margin(op::SourceFunction) = 0
_stencil_margin(op::SourceVector) = 0
_stencil_margin(op::SourceConstant) = 0
_stencil_margin(op::DiracSource) = 0
_stencil_margin(op::IdentityOperator) = 0
_stencil_margin(op::ZeroOperator) = 0

# A tapped node's own step is always exactly 1: every `_stencil_taps` entry across every
# difference, average and jump is drawn from `{-1, 0, 1}` (`stencil_eval.jl`), so composing
# margin here is `+1` regardless of which node it is -- the same fact `_reach` relies on to
# shift the offset set by one tap.
_stencil_margin(op::TappedNode) = _stencil_margin(op.inner_op) + 1

# A shift's step is its own field, unlike a tap's, and unbounded: `simplify_ast` folds nested
# shifts along the same dimension into one (`Shift_a(Shift_b(u)) -> Shift_{a+b}(u)`), so this
# has to read the value rather than assume magnitude 1.
_stencil_margin(op::ShiftNode) = _stencil_margin(op.inner_op) + abs(op.shift_amount)

_stencil_margin(op::OperatorScale) = _stencil_margin(op.inner_op)
_stencil_margin(op::GridFunctionScale) = _stencil_margin(op.inner_op)
_stencil_margin(op::RegionRestriction) = _stencil_margin(op.inner_op)

# Names an absolute column on another mesh via `locate_cell`, never a grid-relative offset on
# the mesh being walked -- see `stencil_offsets(::InterpolationNode)` just above.
_stencil_margin(op::InterpolationNode) = 0

# The one place trial and test reach actually differ: `left_op` is what `off_u` (trial/column)
# ranges over, `right_op` what `off_v` (test/row) does (`multiply_stencils_bilinear`,
# `form/common.jl`), so the margin has to cover whichever side reaches further.
function _stencil_margin(op::BilinearProduct)
    return max(_stencil_margin(op.left_op), _stencil_margin(op.right_op))
end

function _stencil_margin(op::OperatorAdd)
    return max(_stencil_margin(op.left_op), _stencil_margin(op.right_op))
end
