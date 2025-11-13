"""
# operator_types.jl

This file defines the base type hierarchy and helper utilities for operators
in the Bramble finite element framework.

## Type Hierarchy

```
OperatorType (abstract)
├── ScalarOperatorType (abstract)
└── VectorOperatorType (abstract)
```

## Helper Functions

- `_scalar2wrapper`: Convert scalars to FunctionWrappers for type stability
- `_process_scalar`: Format scalar coefficients for display

## Usage

These types and utilities are used throughout the operator system to ensure
type stability and efficient dispatch.
"""

"""
	OperatorType

Abstract base type for all operators acting on finite element spaces.

Operators represent linear transformations on grid space elements, supporting:

  - Scalar multiplication
  - Addition and subtraction
  - Composition
  - Inner product operations

# Subtypes

  - `ScalarOperatorType`: Operators with scalar codomain
  - `VectorOperatorType`: Operators with vector codomain

# Interface

All operator types must implement:

  - `space(op)`: Return the underlying grid space
  - `show(io, op)`: Display representation

See also: [`ScalarOperatorType`](@ref), [`VectorOperatorType`](@ref)
"""
abstract type OperatorType end

"""
	ScalarOperatorType <: OperatorType

Abstract type for operators with scalar-valued codomain.

Used for operators that map grid elements to scalar fields.
"""
abstract type ScalarOperatorType <: OperatorType end

"""
	VectorOperatorType <: OperatorType

Abstract type for operators with vector-valued codomain.

Used for operators that map grid elements to vector fields (e.g., gradients).
"""
abstract type VectorOperatorType <: OperatorType end

#------------------------------------------------------------------------------------------#
# Basic Operator Interface
#------------------------------------------------------------------------------------------#

"""
	space(op::OperatorType) -> AbstractSpaceType

Return the grid space on which the operator `op` acts.
"""
@inline space(op::OperatorType) = op.space

"""
	eltype(op::OperatorType)

Return the element type of the underlying grid space.
"""
@inline eltype(op::OperatorType) = eltype(space(op))

# Default scalar multiplication (commutative)
@inline *(op::OperatorType, α) = α * op

#------------------------------------------------------------------------------------------#
# Helper Functions for Scalar Wrapping
#------------------------------------------------------------------------------------------#

"""
	_scalar2wrapper(::Type{T}, α) -> FunctionWrapper

Convert a scalar value `α` to a `FunctionWrapper` with return type `T`.

This enables efficient storage and evaluation of scalar coefficients in operators.
Handles numbers, tuples, and `VectorElement` types.

# Arguments

  - `T`: Target return type
  - `α`: Scalar value to wrap (Number, Tuple, or VectorElement)

# Returns

  - `FunctionWrapper{T,Tuple{}}`: Zero-argument function wrapper returning the scalar

# Examples

```julia
f = _scalar2wrapper(Float64, 2.5)  # Wraps the number 2.5
f()  # Returns 2.5 as Float64
```
"""
@inline _scalar2wrapper(::Type{T}, α::ElType) where {T,ElType<:Number} = FunctionWrapper{T,Tuple{}}(() -> convert(T, α)::T)
@inline _scalar2wrapper(::Type{T}, α) where T = FunctionWrapper{typeof(α),Tuple{}}(() -> α)

# Binary operation wrappers for broadcast
@inline _scalar2wrapper(::Type{T}, op, α::Tuple, β) where T = FunctionWrapper{typeof(α),Tuple{}}(() -> broadcast(op, α, β))
@inline _scalar2wrapper(::Type{T}, op, α, β::Tuple) where T = FunctionWrapper{typeof(β),Tuple{}}(() -> broadcast(op, α, β))

@inline _scalar2wrapper(::Type{T}, op, α::Number, β::Number) where T = FunctionWrapper{T,Tuple{}}(() -> broadcast(op, convert(T, α)::T, β))
@inline _scalar2wrapper(::Type{T}, op, α::VectorElement, β::Number) where T = FunctionWrapper{typeof(α),Tuple{}}(() -> broadcast(op, α, β))
@inline _scalar2wrapper(::Type{T}, op, α::Number, β::VectorElement) where T = FunctionWrapper{typeof(β),Tuple{}}(() -> broadcast(op, α, β))
@inline _scalar2wrapper(::Type{T}, op, α::VectorElement, β::VectorElement) where T = FunctionWrapper{typeof(α),Tuple{}}(() -> broadcast(op, α, β))

"""
	_process_scalar(f::FunctionWrapper) -> String

Convert a scalar function wrapper to its string representation for display purposes.

Used by operator `show` methods to format scalar coefficients in expressions.

# Arguments

  - `f::FunctionWrapper`: Wrapped scalar value

# Returns

String representation: the numeric value for numbers, or "uₕ" for element types
"""
_process_scalar(f::FunctionWrapper{T,Tuple{}}) where T<:Number = "$(f())"
_process_scalar(f::FunctionWrapper{CoType,Tuple{}}) where CoType = "uₕ"

#------------------------------------------------------------------------------------------#
# Generic Scalar Multiplication
#------------------------------------------------------------------------------------------#

"""
	*(α, op::OperatorType) -> OperatorType

Scale an operator by a scalar or vector coefficient `α`.

# Arguments

  - `α`: Scalar (Number) or vector (VectorElement) coefficient
  - `op`: Operator to scale

# Returns

  - `ZeroOperator` if `α == 0`
  - `op` unchanged if `α == 1`
  - `ScaledOperator{...}` containing the scaled operator

# Examples

```julia
I = IdentityOperator(Wh)
op = 2.5 * I      # Scaled identity
op = uₕ * I       # Element-wise scaling
```

See also: [`ScaledOperator`](@ref), [`ZeroOperator`](@ref)
"""
@inline function *(α::_T, op::OP) where {_T,OP<:OperatorType}
	T = eltype(op)

	if α isa Number && α == 0
		return ZeroOperator(space(op))
	end

	if α isa Number && α == 1
		return op
	end

	S = typeof(space(op))
	operator_cotype = codomaintype(op)

	if _T <: Number && operator_cotype <: Number
		f = _scalar2wrapper(T, α)
		return ScaledOperator{typeof(op),S,typeof(f)}(op.space, f, op)
	end

	if _T <: VectorElement || operator_cotype <: VectorElement
		f = _scalar2wrapper(T, *, α, 1)
		return ScaledOperator{typeof(op),S,typeof(f)}(op.space, f, op)
	end

	@error "Don't know how to handle this expression"
end
