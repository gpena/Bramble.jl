# difference.jl
# Discrete finite difference operators for Bramble lazy AST

# ==============================================================================
# Struct Definitions
# ==============================================================================

"""
    BackwardDifference{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node representing a backward finite difference operator acting in dimension `Dim`.
"""
struct BackwardDifference{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}
	inner_op::OpType
end

"""
    ForwardDifference{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node representing a forward finite difference operator acting in dimension `Dim`.
"""
struct ForwardDifference{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}
	inner_op::OpType
end

# ==============================================================================
# User-Facing API & Overloads
# ==============================================================================

"""
    grad_backward(op::LazyOp{D}) where D

Constructs a backward gradient operator tuple, yielding `D`-tuple of `BackwardDifference` operators.
"""
grad_backward(op::LazyOp{1}) = BackwardDifference{1,1,typeof(op)}(op)
grad_backward(op::LazyOp{D}) where {D} = ntuple(dim -> BackwardDifference{D,dim,typeof(op)}(op), Val(D))

"""
    grad_forward(op::LazyOp{D}) where D

Constructs a forward gradient operator tuple, yielding `D`-tuple of `ForwardDifference` operators.
"""
grad_forward(op::LazyOp{1}) = ForwardDifference{1,1,typeof(op)}(op)
grad_forward(op::LazyOp{D}) where {D} = ntuple(dim -> ForwardDifference{D,dim,typeof(op)}(op), Val(D))

# Add standard Bramble operator overloads mapped to the fast lazy AST:

"""
    ∇₋ₕ(op::LazyOp{D}) where D

Symbolic backward gradient operator.
"""
∇₋ₕ(op::LazyOp{D}) where D = grad_backward(op)

"""
    ∇₊ₕ(op::LazyOp{D}) where D

Symbolic forward gradient operator.
"""
∇₊ₕ(op::LazyOp{D}) where D = grad_forward(op)

# AST-based difference operators (distinct names)

"""
    D₋ₓ(op::LazyOp{D}) where D
    D₊ₓ(op::LazyOp{D}) where D
    D₋ᵧ(op::LazyOp{D}) where D
    D₊ᵧ(op::LazyOp{D}) where D
    D₋₂(op::LazyOp{D}) where D
    D₊₂(op::LazyOp{D}) where D

Symbolic finite difference operators in specified coordinate directions (x, y, z).
"""
D₋ₓ(op::LazyOp{D}) where D = BackwardDifference{D,1,typeof(op)}(op)
D₊ₓ(op::LazyOp{D}) where D = ForwardDifference{D,1,typeof(op)}(op)
D₋ᵧ(op::LazyOp{D}) where D = BackwardDifference{D,2,typeof(op)}(op)
D₊ᵧ(op::LazyOp{D}) where D = ForwardDifference{D,2,typeof(op)}(op)
D₋₂(op::LazyOp{D}) where D = BackwardDifference{D,3,typeof(op)}(op)
D₊₂(op::LazyOp{D}) where D = ForwardDifference{D,3,typeof(op)}(op)


# ==============================================================================
# Zero-Allocation Stencil Evaluators
# ==============================================================================

@inline function local_stencil(op::BackwardDifference{D,Dim}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D,Dim}
	inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
	m = mesh(space)
	h = get_spacing(m, I, Dim)

	mask = I[Dim] == 1 ? 0.0 : 1.0
	t1 = scale_stencil(inner, mask / h)

	inner_shifted = shift_stencil(inner, Val(Dim), Val(-1))
	t2 = scale_stencil(inner_shifted, -mask / h)

	return concatenate_stencils(t1, t2)
end

@inline function local_stencil(op::ForwardDifference{D,Dim}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D,Dim}
	inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
	m = mesh(space)
	dims = npoints(m, Tuple)
	h = get_forward_spacing(m, I, Dim)

	mask = I[Dim] == dims[Dim] ? 0.0 : 1.0
	inner_shifted = shift_stencil(inner, Val(Dim), Val(1))
	t1 = scale_stencil(inner_shifted, mask / h)
	t2 = scale_stencil(inner, -mask / h)

	return concatenate_stencils(t1, t2)
end

# ==============================================================================
# AST Resolution
# ==============================================================================

resolve_ast(op::BackwardDifference{D,Dim}) where {D,Dim} = BackwardDifference{D,Dim,typeof(resolve_ast(op.inner_op))}(resolve_ast(op.inner_op))
resolve_ast(op::ForwardDifference{D,Dim}) where {D,Dim} = ForwardDifference{D,Dim,typeof(resolve_ast(op.inner_op))}(resolve_ast(op.inner_op))

# ==============================================================================
# Direct integration helpers for linear_operators.jl
# ==============================================================================

function Bramble.get_derivative_matrix_and_scale(op::BackwardDifference{D,Dim}, W) where {D,Dim}
	return backward_difference_matrix(W, Dim), 1.0
end

function Bramble.get_innermost_dim(op::BackwardDifference{D,Dim}) where {D,Dim}
	return Dim
end
