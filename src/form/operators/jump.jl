# jump.jl
#
# The symbolic jump across an interface,
#
#     ⟦u⟧ᵢ = u_{i+1} - u_i
#
# One of these, not a forward and a backward pair, for the reason the space layer's
# operators/jump.jl gives: the jump belongs to the interface between two cells rather than
# to a direction of travel across it, so naming a backward jump would name the same
# interface twice.
#
# Arithmetically it is the *unscaled* forward difference, and the form layer has no node
# for that — `ForwardDifference` carries the 1/h. So unlike the space layer, where every
# jump name forwards to its difference counterpart, this is a node of its own.
#
# It also does not truncate, and that is deliberate rather than an oversight. The space
# layer treats the missing u_{n+1} as zero, giving -uₙ at the last point rather than 0,
# which is what makes the operator agree with its matrix: the last row of `jumpₓ(Ωₕ)` is
# [0 … 0 -1]. The scaled differences do truncate, because the truncation lives in their
# weights. So the stencil here drops the forward term and keeps the local one, where
# `ForwardDifference` zeroes both.

"""
    JumpNode{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node for the jump across interfaces along `Dim`, ``u_{i+1} - u_i``.

Not truncated at the far end: the absent `u_{i+1}` is taken as zero, giving `-uᵢ` there.
That matches the space layer's matrix, whose last row keeps the `-1`.
"""
struct JumpNode{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    inner_op::OpType
end

"""
    jumpₓ(op::LazyOp{D}) where D
    jumpᵧ(op::LazyOp{D}) where D
    jump₂(op::LazyOp{D}) where D

Symbolic jumps across the interfaces along each coordinate direction.
"""
jumpₓ(op::LazyOp{D}) where {D} = JumpNode{D, 1, typeof(op)}(op)
jumpᵧ(op::LazyOp{D}) where {D} = JumpNode{D, 2, typeof(op)}(op)
jump₂(op::LazyOp{D}) where {D} = JumpNode{D, 3, typeof(op)}(op)

"""
    jumpₕ(op::LazyOp{D}) where D

Every direction at once, as a `D`-tuple of `JumpNode`s. In one dimension the node
itself, not a one-element tuple.
"""
jumpₕ(op::LazyOp{1}) = jumpₓ(op)
jumpₕ(op::LazyOp{D}) where {D} = ntuple(dim -> JumpNode{D, dim, typeof(op)}(op), Val(D))

@inline function local_stencil(op::JumpNode{D, Dim}, space, I::CartesianIndex{D},
        markers, lin_idx::Int) where {D, Dim}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    dims = npoints(mesh(space), Tuple)

    # only the forward term goes; the local one stays, which is the -uₙ convention
    reach = I[Dim] == dims[Dim] ? 0.0 : 1.0

    forward = scale_stencil(shift_stencil(inner, Val(Dim), Val(1)), reach)
    here = scale_stencil(inner, -1.0)
    return concatenate_stencils(forward, here)
end

function resolve_ast(op::JumpNode{D, Dim}) where {D, Dim}
    inner = resolve_ast(op.inner_op)
    return JumpNode{D, Dim, typeof(inner)}(inner)
end

Bramble.get_innermost_dim(op::JumpNode{D, Dim}) where {D, Dim} = Dim
