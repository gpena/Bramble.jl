"""
# scalar_operators.jl

This file implements basic scalar operators in the finite element framework.

## Operators

- `ZeroOperator`: The zero transformation
- `IdentityOperator`: The identity transformation  
- `ScaledOperator`: A scaled version of another operator (α * Op)

## Algebraic Properties

These operators support standard algebraic simplifications:

- `0 * anything = ZeroOperator`
- `1 * op = op`
- `α * (β * op) = (α*β) * op`
- `op + ZeroOperator = op`

See also: [`OperatorType`](@ref), [`GradientOperator`](@ref)
"""

#########################################
#                                       #
#             Zero Operator             #
#                                       #
#########################################

"""
	ZeroOperator{S} <: OperatorType

Represents the zero operator `0` that maps all elements to zero.

This operator absorbs all scalar multiplication and addition operations,
providing algebraic simplification in operator expressions.

# Fields

  - `space::S`: The grid space on which the operator acts

# Constructor

```julia
ZeroOperator(space)
```

# Algebraic Properties

  - `scalar(op)` returns zero
  - `α * ZeroOperator` returns `ZeroOperator` for any `α`
  - `op + ZeroOperator` returns `op` for any operator `op`
  - `ZeroOperator + op` returns `op`
  - `ZeroOperator - op` returns `-1 * op`

# Examples

```julia
Wh = gridspace(mesh)
zero_op = ZeroOperator(Wh)
op = 100 * zero_op  # Still ZeroOperator
```
"""
struct ZeroOperator{S} <: OperatorType
	space::S
end

show(io::IO, _::ZeroOperator) = print(io, "0")

"""
	scalar(op::ZeroOperator)

Return the scalar representation of the zero operator (zero of the space element type).
"""
scalar(op::ZeroOperator) = zero(eltype(space(op)))

# Algebraic rules for zero operator
@inline *(α, op::ZeroOperator) = op
@inline ⋅(α, op::ZeroOperator) = op

#########################################
#                                       #
#           Identity Operator           #
#                                       #
#########################################

"""
	IdentityOperator{S} <: OperatorType

Represents the identity operator `I` that maps each element to itself.

The identity operator is the neutral element for operator composition and
serves as the base case for many operator algebraic simplifications.

# Fields

  - `space::S`: The grid space on which the operator acts

# Constructor

```julia
IdentityOperator(space)
```

# Algebraic Properties

  - `scalar(I)` returns one (multiplicative identity)
  - `α * I` creates a `ScaledOperator` with coefficient `α`
  - `I * α` is equivalent to `α * I`
  - `I + ZeroOperator` returns `I`

# Examples

```julia
Wh = gridspace(mesh)
I = IdentityOperator(Wh)
op = 3.5 * I         # Scaled identity operator
result = innerₕ(uₕ, I)  # Identity in inner product
```

See also: [`ScaledOperator`](@ref), [`ZeroOperator`](@ref)
"""
struct IdentityOperator{S} <: OperatorType
	space::S
end

show(io::IO, _::IdentityOperator) = print(io, "I")

"""
	scalar(op::IdentityOperator)

Return the scalar representation of the identity operator (one).
"""
scalar(op::IdentityOperator) = one(eltype(space(op)))

"""
	⋅(α, op::IdentityOperator)

Dot product notation for scaling the identity operator.

Creates a `ScaledOperator` with scalar coefficient `α`.
"""
@inline function ⋅(α, op::IdentityOperator)
	if α isa Number
		return ScaledOperator(op.space, α, op)
	end

	if α isa VectorElement
		return ScaledOperator(op.space, α, op)
	end

	if α isa Tuple
		# Tuple case - could be extended for vector-valued operators
		@show "Tuple"
	end

	@error "Don't know how to handle this expression"
end

# Multiplication rules for identity operator
@inline *(α, op::IdentityOperator) = ScaledOperator(space(op), α)
@inline *(op::IdentityOperator, α) = α * op

#########################################
#                                       #
#           Scaling Operator            #
#                                       #
#########################################

"""
	ScaledOperator{OP,S,V} <: OperatorType

Represents a scaled operator `α * Op`, where `α` is a scalar coefficient.

The scalar is stored as a `FunctionWrapper` for efficient evaluation and type stability.
Scaled operators support nested composition and algebraic simplification.

# Fields

  - `space::S`: The grid space on which the operator acts
  - `scalar::V`: Function wrapper containing the scalar coefficient
  - `operator::OP`: The underlying operator being scaled

# Type Parameters

  - `OP`: Type of the underlying operator
  - `S`: Type of the grid space
  - `V`: Type of the scalar function wrapper

# Constructor

```julia
ScaledOperator(space, α, operator)
```

# Algebraic Simplification

  - `ScaledOperator(S, 0, op)` returns `ZeroOperator(S)`
  - `ScaledOperator(S, 1, op)` returns `op` unchanged
  - `α * (β * op)` combines to `(α*β) * op`

# Examples

```julia
I = IdentityOperator(Wh)
op1 = ScaledOperator(Wh, 2.5, I)    # 2.5 * I
op2 = 3 * op1                         # (3 * 2.5) * I = 7.5 * I
```

See also: [`IdentityOperator`](@ref), [`scalar`](@ref), [`parent_operator`](@ref)
"""
struct ScaledOperator{OP,S,V} <: OperatorType
	space::S
	scalar::V
	operator::OP
end

"""
	codomaintype(op::ScaledOperator)

Return the codomain type of the scaled operator (derived from the scalar wrapper type).
"""
@inline codomaintype(_::ScaledOperator{OP,S,V}) where {OP,S,V} = codomaintype(V)

"""
	scalar(op::ScaledOperator)

Evaluate and return the scalar coefficient of the scaled operator.

# Examples

```julia
op = ScaledOperator(Wh, 2.5, IdentityOperator(Wh))
scalar(op)  # Returns 2.5
```
"""
@inline scalar(op::ScaledOperator) = op.scalar()

"""
	parent_operator(op::ScaledOperator)

Return the underlying operator being scaled.

# Examples

```julia
I = IdentityOperator(Wh)
op = 3 * I
parent_operator(op)  # Returns I
```
"""
@inline parent_operator(op::ScaledOperator) = op.operator

# Display methods for scaled operators
show(io::IO, op::ScaledOperator{OP}) where OP<:IdentityOperator = print(io, "$(_process_scalar(op.scalar)) * I")
show(io::IO, op::ScaledOperator{OP1}) where {OP2<:IdentityOperator,OP1<:ScaledOperator{OP2}} = print(io, "$(_process_scalar(op.scalar)) * I")
show(io::IO, op::ScaledOperator{OP}) where OP<:OperatorType = print(io, "$(_process_scalar(op.scalar)) * ($(op.operator))")

#------------------------------------------------------------------------------------------#
# ScaledOperator Constructors with Algebraic Simplification
#------------------------------------------------------------------------------------------#

"""
	ScaledOperator(S::AbstractSpaceType, α, op::ZeroOperator)

Scaling a zero operator always returns the zero operator (absorbing property).
"""
ScaledOperator(S::AbstractSpaceType, α, op::ZeroOperator) = op

"""
	ScaledOperator(S::AbstractSpaceType, α, op = IdentityOperator(S))

Construct a scaled operator `α * op` with algebraic simplification.

# Arguments

  - `S::AbstractSpaceType`: The grid space
  - `α`: Scalar coefficient (Number or VectorElement)
  - `op`: Operator to scale (defaults to `IdentityOperator(S)`)

# Returns

  - `op` if `α == 1` (identity property)
  - `ZeroOperator(S)` if `α == 0` (annihilation property)
  - `ScaledOperator{...}` otherwise

# Examples

```julia
op1 = ScaledOperator(Wh, 2.5)           # 2.5 * I
op2 = ScaledOperator(Wh, 1, D₋ₓ)        # D₋ₓ (unchanged)
op3 = ScaledOperator(Wh, 0, any_op)     # ZeroOperator
```
"""
function ScaledOperator(S::AbstractSpaceType, α, op = IdentityOperator(S))
	T = eltype(S)

	if α isa Number && α == 1
		return op
	end

	if α isa Number && α == 0
		return ZeroOperator(S)
	end

	f = _scalar2wrapper(T, α)
	return ScaledOperator{typeof(op),typeof(S),typeof(f)}(S, f, op)
end

#------------------------------------------------------------------------------------------#
# Multiplication Rules for ScaledOperator
#------------------------------------------------------------------------------------------#

"""
	*(α, op::ScaledOperator)

Multiply a scaled operator by another scalar, combining the coefficients.

Implements the algebraic simplification: `α * (β * op) = (α*β) * op`

# Examples

```julia
op1 = 2.5 * I
op2 = 3 * op1  # Results in 7.5 * I
```
"""
@inline function *(α::_T, op::OP) where {_T,OP<:ScaledOperator}
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
		f = _scalar2wrapper(T, α * scalar(op))
		return ScaledOperator{typeof(op.operator),S,typeof(f)}(op.space, f, op.operator)
	end

	if _T <: VectorElement || operator_cotype <: VectorElement
		f = _scalar2wrapper(T, *, α, scalar(op))
		return ScaledOperator{typeof(op.operator),S,typeof(f)}(op.space, f, op.operator)
	end

	@error "Don't know how to handle this expression"
end
