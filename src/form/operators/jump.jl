# jump.jl
#
# The symbolic jump across an interface,
#
#     ⟦u⟧ᵢ = u_{i+1} - u_i
#
# A single definition rather than forward/backward pairs: the jump belongs to the interface
# between two cells rather than to a direction of travel across it.
#
# Arithmetically it is the unscaled forward difference (as `ForwardDifference` carries the 1/h).
# Unlike the space layer, where each jump forwards to its difference counterpart, here it is
# an independent AST node.
#
# Boundary points do not truncate: the absent u_{n+1} is taken as zero, yielding -uₙ at the
# boundary point. This matches the space-layer matrix representation where the final row of
# `jumpₓ(Ωₕ)` is [0, ..., 0, -1].

"""
    JumpNode{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}

AST node representing the jump across interfaces along dimension `Dim`, ``u_{i+1} - u_i``.

Not truncated at the far end: the absent `u_{i+1}` is taken as zero, yielding `-uᵢ` there.
This matches the space-layer matrix convention, whose boundary row preserves `-1`.
"""
struct JumpNode{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    inner_op::OpType
end

"""
    jumpₓ(op::LazyOp{D}) -> JumpNode
    jumpᵧ(op::LazyOp{D}) -> JumpNode
    jump₂(op::LazyOp{D}) -> JumpNode

Symbolic jumps across the interfaces along coordinate directions ``x``, ``y``, and ``z``.
"""
jumpₓ(op::LazyOp{D}) where {D} = JumpNode{D, 1, typeof(op)}(op)
jumpᵧ(op::LazyOp{D}) where {D} = JumpNode{D, 2, typeof(op)}(op)
jump₂(op::LazyOp{D}) where {D} = JumpNode{D, 3, typeof(op)}(op)

"""
    jumpₕ(op::LazyOp{D})

Symbolic jumps across every coordinate direction simultaneously. Returns a `JumpNode`
in 1D, or a `NTuple{D, JumpNode}` in higher dimensions.
"""
jumpₕ(op::LazyOp{1}) = jumpₓ(op)
jumpₕ(op::LazyOp{D}) where {D} = ntuple(dim -> JumpNode{D, dim, typeof(op)}(op), Val(D))

@inline function local_stencil(op::JumpNode{D, Dim}, space, I::CartesianIndex{D},
        markers, lin_idx::Int) where {D, Dim}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    dims = npoints(mesh(space), Tuple)

    # only the forward term goes; the local one stays, which is the -uₙ convention
    reach = I[Dim] == dims[Dim] ? 0 : 1

    forward = scale_stencil(
        shifted_inner_stencil(op.inner_op, inner, space, I, markers, Val(Dim), Val(1)),
        reach)
    here = scale_stencil(inner, -1)
    return concatenate_stencils(forward, here)
end

function resolve_ast(op::JumpNode{D, Dim}) where {D, Dim}
    inner = resolve_ast(op.inner_op)
    return JumpNode{D, Dim, typeof(inner)}(inner)
end

Bramble.get_innermost_dim(op::JumpNode{D, Dim}) where {D, Dim} = Dim
