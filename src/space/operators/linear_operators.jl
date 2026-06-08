# linear_operators.jl

abstract type OperatorType end
abstract type LazyOp{D} <: OperatorType end

# Base interface implementations
@inline space(op::OperatorType) = op.space
@inline eltype(op::OperatorType) = eltype(space(op))

# ==============================================================================
# Unified Struct Definitions
# ==============================================================================

struct IdentityOperator{D,S} <: LazyOp{D}
	space::S
end

struct ZeroOperator{D,S} <: LazyOp{D}
	space::S
end

IdentityOperator(space::AbstractSpaceType) = IdentityOperator{dim(space), typeof(space)}(space)
ZeroOperator(space::AbstractSpaceType) = ZeroOperator{dim(space), typeof(space)}(space)

# Base AST Node types required for scaling and composition
struct OperatorScale{D,ScalarType,OpType<:LazyOp{D}} <: LazyOp{D}
	scalar::ScalarType
	inner_op::OpType

	function OperatorScale{D,ScalarType,OpType}(scalar::ScalarType, inner_op::OpType) where {D,ScalarType,OpType}
		new{D,ScalarType,OpType}(scalar, inner_op)
	end
end

struct GridFunctionScale{D,VType,OpType<:LazyOp{D}} <: LazyOp{D}
	grid_function::VType
	inner_op::OpType

	function GridFunctionScale{D,VType,OpType}(grid_function::VType, inner_op::OpType) where {D,VType,OpType}
		new{D,VType,OpType}(grid_function, inner_op)
	end
end

struct OperatorAdd{D,LeftType<:LazyOp{D},RightType<:LazyOp{D}} <: LazyOp{D}
	left_op::LeftType
	right_op::RightType

	function OperatorAdd{D,LeftType,RightType}(left::LeftType, right::RightType) where {D,LeftType,RightType}
		new{D,LeftType,RightType}(left, right)
	end
end

OperatorScale(scalar::S, op::LazyOp{D}) where {D,S} = OperatorScale{D,S,typeof(op)}(scalar, op)
GridFunctionScale(grid_function::V, op::LazyOp{D}) where {D,V} = GridFunctionScale{D,V,typeof(op)}(grid_function, op)
OperatorAdd(left::LazyOp{D}, right::LazyOp{D}) where D = OperatorAdd{D,typeof(left),typeof(right)}(left, right)

# ==============================================================================
# Symbolic AST vs Discrete Operator Traits
# ==============================================================================

function is_symbolic end

is_symbolic(::LazyOp) = false
is_symbolic(ops::Tuple) = any(is_symbolic, ops)

is_symbolic(op::OperatorScale) = is_symbolic(op.inner_op)
is_symbolic(op::GridFunctionScale) = is_symbolic(op.inner_op)
is_symbolic(op::OperatorAdd) = is_symbolic(op.left_op) || is_symbolic(op.right_op)

# ==============================================================================
# Display Methods
# ==============================================================================

import Base: show

show(io::IO, _::IdentityOperator) = print(io, "I")
show(io::IO, _::ZeroOperator) = print(io, "0")

# ==============================================================================
# Algebraic Rules & Overloads
# ==============================================================================

# Zero operator absorption
@inline *(α, op::ZeroOperator) = op
@inline *(op::ZeroOperator, α) = op
@inline ⋅(α, op::ZeroOperator) = op
@inline ⋅(op::ZeroOperator, α) = op

# AST algebra overloads
@inline +(op1::LazyOp{D}, op2::LazyOp{D}) where D = OperatorAdd(op1, op2)
@inline -(op1::LazyOp{D}, op2::LazyOp{D}) where D = op1 + OperatorScale(-1.0, op2)

# Multiplications
@inline *(c::Number, op::LazyOp) = OperatorScale(c, op)
@inline *(op::LazyOp, c::Number) = OperatorScale(c, op)
@inline /(op::LazyOp, c::Number) = OperatorScale(1.0 / c, op)

@inline *(vh::AbstractVector, op::LazyOp) = GridFunctionScale(vh, op)
@inline *(op::LazyOp, vh::AbstractVector) = GridFunctionScale(vh, op)

@inline *(vh::Function, op::LazyOp) = GridFunctionScale(vh, op)
@inline *(op::LazyOp, vh::Function) = GridFunctionScale(vh, op)

@inline *(vh::AbstractVector, ops::NTuple{D,LazyOp{D}}) where {D} = ntuple(dim -> vh * ops[dim], Val(D))
@inline *(ops::NTuple{D,LazyOp{D}}, vh::AbstractVector) where {D} = ntuple(dim -> ops[dim] * vh, Val(D))

@inline *(vh::Function, ops::NTuple{D,LazyOp{D}}) where {D} = ntuple(dim -> vh * ops[dim], Val(D))
@inline *(ops::NTuple{D,LazyOp{D}}, vh::Function) where {D} = ntuple(dim -> ops[dim] * vh, Val(D))

@inline *(c::Number, ops::NTuple{D,LazyOp{D}}) where {D} = ntuple(dim -> c * ops[dim], Val(D))
@inline *(ops::NTuple{D,LazyOp{D}}, c::Number) where {D} = ntuple(dim -> ops[dim] * c, Val(D))

@inline ⋅(c::Number, op::LazyOp) = OperatorScale(c, op)
@inline ⋅(op::LazyOp, c::Number) = OperatorScale(c, op)
@inline ⋅(vh::AbstractVector, op::LazyOp) = GridFunctionScale(vh, op)
@inline ⋅(op::LazyOp, vh::AbstractVector) = GridFunctionScale(vh, op)

# Gradient Operator Constructor
GradientOperator(space::AbstractSpaceType) = ∇₋ₕ(IdentityOperator(space))

# ==============================================================================
# Helper functions for scalar/scaling information
# ==============================================================================

function scalar end
function codomaintype end

scalar(op::IdentityOperator) = one(eltype(space(op)))
scalar(op::ZeroOperator) = zero(eltype(space(op)))
scalar(op::OperatorScale) = op.scalar * scalar(op.inner_op)
scalar(op::GridFunctionScale) = op.grid_function .* scalar(op.inner_op)
scalar(op::OperatorAdd) = scalar(op.left_op) + scalar(op.right_op)

# Fallback for derivative/averaging/restriction operators
scalar(op::LazyOp) = scalar(op.inner_op)

# Tuple fallback
scalar(ops::Tuple) = all(y -> y == first(ops), ops) ? scalar(first(ops)) : map(scalar, ops)

codomaintype(op::IdentityOperator) = eltype(space(op))
codomaintype(op::ZeroOperator) = eltype(space(op))
codomaintype(op::OperatorScale{D,S,V}) where {D,S,V} = typeof(op.scalar)
codomaintype(op::GridFunctionScale{D,V,OP}) where {D,V,OP} = V
codomaintype(op::LazyOp) = eltype(space(op))

# ==============================================================================
# Direct evaluations of discrete L² inner products on VectorElements
# ==============================================================================

function get_derivative_matrix_and_scale end
function get_innermost_dim end

function get_derivative_matrix_and_scale(op::OperatorScale, W)
	mat, s = get_derivative_matrix_and_scale(op.inner_op, W)
	return mat, s * op.scalar
end

function get_derivative_matrix_and_scale(op::GridFunctionScale, W)
	mat, s = get_derivative_matrix_and_scale(op.inner_op, W)
	return mat, s .* op.grid_function
end

function get_innermost_dim(op::OperatorScale)
	return get_innermost_dim(op.inner_op)
end
function get_innermost_dim(op::GridFunctionScale)
	return get_innermost_dim(op.inner_op)
end

# Trapezoidal rule innerₕ
@inline innerₕ(uₕ::VectorElement, ::IdentityOperator) = innerh_weights(space(uₕ)) .* uₕ.values
@inline innerₕ(uₕ::VectorElement, ::ZeroOperator) = zeros(eltype(uₕ), length(uₕ.values))
@inline innerₕ(uₕ::VectorElement, l::OperatorScale{D,S,OP}) where {D,S,OP<:Union{IdentityOperator,OperatorScale}} = innerh_weights(space(uₕ)) .* l.scalar .* uₕ.values
@inline innerₕ(uₕ::VectorElement, l::GridFunctionScale{D,V,OP}) where {D,V,OP<:Union{IdentityOperator,OperatorScale}} = innerh_weights(space(uₕ)) .* l.grid_function .* uₕ.values

# Modified inner plus
@inline inner₊(uₕ::VectorElement, ::IdentityOperator) = innerplus_weights(space(uₕ), Val(1)) .* uₕ.values
@inline inner₊(uₕ::VectorElement, ::ZeroOperator) = zeros(eltype(uₕ), length(uₕ.values))
@inline inner₊(uₕ::VectorElement, l::OperatorScale{D,S,OP}) where {D,S,OP<:Union{IdentityOperator,OperatorScale}} = innerplus_weights(space(uₕ), Val(1)) .* l.scalar .* uₕ.values
@inline inner₊(uₕ::VectorElement, l::GridFunctionScale{D,V,OP}) where {D,V,OP<:Union{IdentityOperator,OperatorScale}} = innerplus_weights(space(uₕ), Val(1)) .* l.grid_function .* uₕ.values

# Multi-dimensional evaluations (l is a Tuple of LazyOps, e.g. GradientOperator(W))
@inline function inner₊(uₕ::VectorElement, l::NTuple{D,LazyOp{D}}) where D
	res = similar(uₕ.values)
	res .= 0
	inner₊!(res, uₕ, l)
	return res
end

function inner₊!(vₕ::AbstractVector, uₕ::VectorElement, l::NTuple{D,LazyOp{D}}) where D
	W = space(uₕ)
	for i in 1:D
		x = innerplus_weights(W, i)
		mat, s = get_derivative_matrix_and_scale(l[i], W)
		@. W.vec_cache = x * uₕ.values * s
		mul!(vₕ, transpose(mat), W.vec_cache, 1, 1)
	end
	nothing
end

function inner₊!(vₕ::AbstractVector, uₕ::NTuple{D,VectorElement}, l::NTuple{D,LazyOp{D}}) where D
	W = space(first(uₕ))
	for i in 1:D
		x = innerplus_weights(W, i)
		mat, s = get_derivative_matrix_and_scale(l[i], W)
		@. W.vec_cache = x * uₕ[i].values * s
		mul!(vₕ, transpose(mat), W.vec_cache, 1, 1)
	end
	nothing
end

# Commutative overloads
@inline inner₊(l::NTuple{D,LazyOp{D}}, uₕ::Union{VectorElement,NTuple{D,VectorElement}}) where D = inner₊(uₕ, l)
