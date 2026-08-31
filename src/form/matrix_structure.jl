# matrix_structure.jl
#
# What shape of matrix an operator assembles into, read off its AST before assembling it.
#
# Every node here reaches a fixed set of neighbours: a backward difference reads the point
# and the one before it, a cross-weighted difference reads three in a row, an average reads
# two. Those offsets are a property of the tree, not of the grid point — truncation at a
# boundary zeroes the coefficients and keeps the offsets, which is what makes the set well
# defined. So the sparsity pattern of the assembled matrix is known before a single entry
# is computed, and with it the narrowest LinearAlgebra type that can hold the operator.
#
# The payoff is uneven, and worth stating plainly because the obvious generalisation is a
# trap. `Diagonal` is a large win wherever it applies: O(n) storage against O(nnz), an O(n)
# matrix-vector product, and a solve that is elementwise division. `Tridiagonal` in one
# dimension is a large win too — the Thomas algorithm is O(n) against sparse LU with
# fill-in. Above one dimension it inverts. A 2D five-point stencil has offsets ±1 and ±nₓ,
# so the *band* is 2nₓ + 1 wide while each row holds five entries: a banded format would
# store the whole band, including the zeros between the diagonals, and lose badly to
# sparse. Hence the rule below is not "map bandwidth to a banded type"; it is `Diagonal`
# whenever nothing but the origin is touched, `Tridiagonal` only in one dimension, and
# `SparseMatrixCSC` for everything else.

"""
	MatrixStructure

The shape of matrix an operator assembles into: [`DiagonalStructure`](@ref),
[`TridiagonalStructure`](@ref) or [`SparseStructure`](@ref).
"""
abstract type MatrixStructure end

"""
	DiagonalStructure <: MatrixStructure

The operator touches no neighbour, so it assembles into a `Diagonal`.
"""
struct DiagonalStructure <: MatrixStructure end

"""
	TridiagonalStructure <: MatrixStructure

A one-dimensional operator reaching at most one point either side, so it assembles into a
`Tridiagonal`.
"""
struct TridiagonalStructure <: MatrixStructure end

"""
	SparseStructure <: MatrixStructure

Anything else. Above one dimension this is the right answer even for a narrow stencil: the
band between the diagonals of a five-point stencil is `2nₓ + 1` wide and almost entirely
zero, so a banded format stores far more than a sparse one.
"""
struct SparseStructure <: MatrixStructure end

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

function stencil_offsets(op::OperatorAdd)
    sort!(union(
        stencil_offsets(op.left_op), stencil_offsets(op.right_op)))
end

# --- classification ------------------------------------------------------------------ #

"""
	matrix_structure(op) -> MatrixStructure

The narrowest matrix type the operator `op` assembles into, from the offsets it reaches.

`Diagonal` when it touches nothing but the point itself. `Tridiagonal` when the space is
one-dimensional and it reaches at most one point either side. `SparseMatrixCSC` otherwise —
including for a narrow stencil in two or three dimensions, where the band between the
diagonals is mostly zero and a banded format would store more than a sparse one.
"""
function matrix_structure(op::LazyOp{D}) where {D}
    offsets = stencil_offsets(op)
    origin = ntuple(_ -> 0, D)

    all(==(origin), offsets) && return DiagonalStructure()
    D == 1 && all(o -> -1 <= o[1] <= 1, offsets) && return TridiagonalStructure()
    return SparseStructure()
end
