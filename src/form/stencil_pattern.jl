# stencil_pattern.jl
#
# Which grid offsets an operator reaches, read off its AST before assembling it.
#
# Every node here reaches a fixed set of neighbours: a backward difference reads the point
# and the one before it, a cross-weighted difference reads three in a row, an average reads
# two. Those offsets are a property of the tree, not of the grid point — truncation at a
# boundary zeroes the coefficients and keeps the offsets, which is what makes the set well
# defined. So the sparsity pattern of the assembled matrix is known before a single entry
# is computed.
#
# What that is for is preallocation. The pattern of an assembled operator is the stencil's,
# and building the backend's matrix with exactly that pattern means assembly only ever
# updates stored values and never performs a structural insert — which rebuilds a column.
# It is the same fact the removal of the `dropzeros` option rested on: the pattern is known
# ahead of time and is worth keeping fixed.
#
# What this deliberately does NOT do is choose a matrix type. An earlier version classified
# operators into `Diagonal`, `Tridiagonal` or sparse, and that was wrong twice over. It is
# circumstantial — a diagonal operator is a special case that rarely survives being part of
# a real form — and, more importantly, the matrix type is not the form layer's to decide.
# `Backend{VT, MT}` carries it as a type parameter and `matrix(backend, n, m)` builds it,
# which is the whole point of the backend: a Metal backend must get a GPU matrix, and
# handing back a `Tridiagonal` because the stencil happened to be narrow would break that
# contract. The offsets inform how the backend's matrix is preallocated; they do not
# override what it is.

"""
	stencil_offsets(op) -> Vector{NTuple{D, Int}}

The grid offsets the operator `op` reaches, sorted and without repeats.

Read from the AST rather than from an evaluated stencil, and the two agree: a truncated
point keeps its offsets and zeroes its coefficients, so the set does not vary over the grid.
The one node whose reach is not fixed by its type is [`ShiftNode`](@ref), which carries its
step as a field; the value is available here because this walks the built tree.
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

# A linear product contracts the left factor away — `multiply_stencils_linear` keeps only
# the right offsets — so its reach is the test side's. Found missing when the parallel
# assembly wanted to ask what an assembled form reaches: the products are the only nodes
# assembly ever evaluates, and neither had a method.
stencil_offsets(op::LinearProduct) = stencil_offsets(op.right_op)

# A bilinear product deliberately has none. Its stencil entries are (row offset, column
# offset, value) triples, so what it reaches is a set of *pairs* and does not fit the shape
# this function answers with. Matrix assembly wants both sets; ask the two factors.
@noinline function stencil_offsets(op::BilinearProduct)
    throw(ArgumentError(
        "stencil_offsets is not defined for a BilinearProduct: its stencil pairs a row " *
        "offset with a column offset, so it reaches a set of pairs rather than a set of " *
        "offsets. Ask its factors instead."))
end

function stencil_offsets(op::OperatorAdd)
    sort!(union(
        stencil_offsets(op.left_op), stencil_offsets(op.right_op)))
end
