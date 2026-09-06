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

# a difference or an average reads the point and one neighbour; which neighbour is the
# only thing that separates the backward and forward members of each pair
function stencil_offsets(op::BackwardDifference{D, Dim}) where {D, Dim}
    _reach(
        stencil_offsets(op.inner_op), Val(Dim), (0, -1))
end
function stencil_offsets(op::ForwardDifference{D, Dim}) where {D, Dim}
    _reach(
        stencil_offsets(op.inner_op), Val(Dim), (0, 1))
end
function stencil_offsets(op::BackwardAverage{D, Dim}) where {D, Dim}
    _reach(
        stencil_offsets(op.inner_op), Val(Dim), (0, -1))
end
function stencil_offsets(op::ForwardAverage{D, Dim}) where {D, Dim}
    _reach(
        stencil_offsets(op.inner_op), Val(Dim), (0, 1))
end
function stencil_offsets(op::JumpNode{D, Dim}) where {D, Dim}
    _reach(
        stencil_offsets(op.inner_op), Val(Dim), (0, 1))
end
function stencil_offsets(op::StarDifference{D, Dim}) where {D, Dim}
    _reach(
        stencil_offsets(op.inner_op), Val(Dim), (0, 1))
end

# the centered difference skips its own centre, and the cross-weighted one does not
function stencil_offsets(op::CenteredDifference{D, Dim}) where {D, Dim}
    _reach(
        stencil_offsets(op.inner_op), Val(Dim), (-1, 1))
end
function stencil_offsets(op::CrossWeightedDifference{D, Dim}) where {D, Dim}
    _reach(
        stencil_offsets(op.inner_op), Val(Dim), (-1, 0, 1))
end

# a shift moves the whole reach and widens nothing
function stencil_offsets(op::ShiftNode{D, Dim}) where {D, Dim}
    _reach(
        stencil_offsets(op.inner_op), Val(Dim), (op.shift_amount,))
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
    sort!(union(
        stencil_offsets(op.left_op), stencil_offsets(op.right_op)))
end
