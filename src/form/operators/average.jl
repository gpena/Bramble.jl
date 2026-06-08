# average.jl
# Discrete averaging operators for Bramble lazy AST

# ==============================================================================
# Struct Definitions
# ==============================================================================

"""
    BackwardAverage{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node representing a backward spatial averaging operator acting in dimension `Dim`.
"""
struct BackwardAverage{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}
	inner_op::OpType
end

"""
    ForwardAverage{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node representing a forward spatial averaging operator acting in dimension `Dim`.
"""
struct ForwardAverage{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}
	inner_op::OpType
end

"""
    ShiftNode{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node representing a stencil shift operation by `shift_amount` grid points in dimension `Dim`.
"""
struct ShiftNode{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}
	shift_amount::Int
	inner_op::OpType
end

# ==============================================================================
# User-Facing API & Overloads
# ==============================================================================

"""
    avg_backward(op::LazyOp{D}, dim::Int) where D

Applies a backward average operator to `op` in dimension `dim`.
"""
avg_backward(op::LazyOp{D}, dim::Int) where D = BackwardAverage{D,dim,typeof(op)}(op)

"""
    avg_forward(op::LazyOp{D}, dim::Int) where D

Applies a forward average operator to `op` in dimension `dim`.
"""
avg_forward(op::LazyOp{D}, dim::Int) where D = ForwardAverage{D,dim,typeof(op)}(op)

"""
    shift_op(op::LazyOp{D}, dim::Int, amount::Int) where D

Shifts the stencil of `op` by `amount` grid points in dimension `dim`.
"""
shift_op(op::LazyOp{D}, dim::Int, amount::Int) where D = ShiftNode{D,dim,typeof(op)}(amount, op)

# AST-based average operators (distinct names)

"""
    M₋ₓ(op::LazyOp{D}) where D
    M₊ₓ(op::LazyOp{D}) where D
    M₋ᵧ(op::LazyOp{D}) where D
    M₊ᵧ(op::LazyOp{D}) where D
    M₋₂(op::LazyOp{D}) where D
    M₊₂(op::LazyOp{D}) where D

Symbolic averaging operators in specified coordinate directions (x, y, z).
"""
M₋ₓ(op::LazyOp{D}) where D = BackwardAverage{D,1,typeof(op)}(op)
M₊ₓ(op::LazyOp{D}) where D = ForwardAverage{D,1,typeof(op)}(op)
M₋ᵧ(op::LazyOp{D}) where D = BackwardAverage{D,2,typeof(op)}(op)
M₊ᵧ(op::LazyOp{D}) where D = ForwardAverage{D,2,typeof(op)}(op)
M₋₂(op::LazyOp{D}) where D = BackwardAverage{D,3,typeof(op)}(op)
M₊₂(op::LazyOp{D}) where D = ForwardAverage{D,3,typeof(op)}(op)

"""
    vectorial_avg_backward(op::LazyOp{D}) where D

Applies backward spatial averaging component-wise across all dimensions.
"""
vectorial_avg_backward(op::LazyOp{D}) where D = ntuple(dim -> BackwardAverage{D,dim,typeof(op)}(op), Val(D))
vectorial_avg_backward(op::LazyOp{1}) = BackwardAverage{1,1,typeof(op)}(op)

"""
    vectorial_avg_forward(op::LazyOp{D}) where D

Applies forward spatial averaging component-wise across all dimensions.
"""
vectorial_avg_forward(op::LazyOp{D}) where D = ntuple(dim -> ForwardAverage{D,dim,typeof(op)}(op), Val(D))
vectorial_avg_forward(op::LazyOp{1}) = ForwardAverage{1,1,typeof(op)}(op)

"""
    M₋ₕ(op::LazyOp{D}) where D

Symbolic backward spatial averaging operator tuple.
"""
M₋ₕ(op::LazyOp{D}) where D = vectorial_avg_backward(op)

"""
    M₊ₕ(op::LazyOp{D}) where D

Symbolic forward spatial averaging operator tuple.
"""
M₊ₕ(op::LazyOp{D}) where D = vectorial_avg_forward(op)


# ==============================================================================
# Zero-Allocation Stencil Evaluators
# ==============================================================================

@inline function local_stencil(op::BackwardAverage{D,Dim}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D,Dim}
	inner = local_stencil(op.inner_op, space, I, markers, lin_idx)

	mask = I[Dim] == 1 ? 0.0 : 0.5
	t1 = scale_stencil(inner, mask)

	inner_shifted = shift_stencil(inner, Val(Dim), Val(-1))
	t2 = scale_stencil(inner_shifted, mask)

	return concatenate_stencils(t1, t2)
end

@inline function local_stencil(op::ForwardAverage{D,Dim}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D,Dim}
	inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
	m = mesh(space)
	dims = npoints(m, Tuple)

	mask = I[Dim] == dims[Dim] ? 0.0 : 0.5
	inner_shifted = shift_stencil(inner, Val(Dim), Val(1))
	t1 = scale_stencil(inner_shifted, mask)
	t2 = scale_stencil(inner, mask)

	return concatenate_stencils(t1, t2)
end

@inline function local_stencil(op::ShiftNode{D,Dim}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D,Dim}
	inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
	return shift_stencil(inner, Val(Dim), op.shift_amount)
end

# ==============================================================================
# AST Resolution
# ==============================================================================

resolve_ast(op::BackwardAverage{D,Dim}) where {D,Dim} = BackwardAverage{D,Dim,typeof(resolve_ast(op.inner_op))}(resolve_ast(op.inner_op))
resolve_ast(op::ForwardAverage{D,Dim}) where {D,Dim} = ForwardAverage{D,Dim,typeof(resolve_ast(op.inner_op))}(resolve_ast(op.inner_op))
resolve_ast(op::ShiftNode{D,Dim}) where {D,Dim} = ShiftNode{D,Dim,typeof(resolve_ast(op.inner_op))}(op.shift_amount, resolve_ast(op.inner_op))
