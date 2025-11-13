"""
# differential_operators.jl

This file implements differential operators for finite element spaces.

## Operators

- `GradientOperator`: Discrete gradient operator ∇ₕ
- `ScaledGradientOperator`: Type alias for scaled gradients

## Usage

The gradient operator computes spatial derivatives using backward difference matrices.
For multi-dimensional problems, it produces a vector of partial derivatives.

See also: [`OperatorType`](@ref), [`ScaledOperator`](@ref)
"""

#########################################
#                                       #
#           Gradient Operators          #
#                                       #
#########################################

"""
	GradientOperator{S} <: OperatorType

Represents the discrete gradient operator ∇ₕ acting on grid space elements.

The gradient operator computes spatial derivatives using backward difference matrices.
For multi-dimensional problems, it produces a vector of partial derivatives.

# Fields

  - `space::S`: The grid space on which the operator acts

# Constructor

```julia
GradientOperator(space)
```

# Properties

  - `scalar(∇)` returns one
  - `codomaintype(∇)` returns the element type of the space
  - Works with `inner₊` for proper integration

# Examples

```julia
Wh = gridspace(mesh)
∇ = GradientOperator(Wh)
uₕ = element(Wh)

# Use in bilinear form for Laplacian
a(u, v) = inner₊(∇₋ₕ(u), ∇₋ₕ(v))
```

See also: [`∇₋ₕ`](@ref), [`inner₊`](@ref)
"""
struct GradientOperator{S} <: OperatorType
	space::S
end

"""
	ScaledGradientOperator{S,V}

Type alias for a scaled gradient operator `α * ∇ₕ`.

Created automatically when multiplying a gradient operator by a scalar.
"""
ScaledGradientOperator{S,V} = ScaledOperator{GradientOperator{S},S,V}

"""
	scalar(op::GradientOperator)

Return the scalar representation of the gradient operator (one).
"""
scalar(op::GradientOperator) = one(eltype(space(op)))

"""
	codomaintype(op::GradientOperator)

Return the codomain type of the gradient operator (element type of the space).
"""
codomaintype(op::GradientOperator) = eltype(space(op))

# Display methods for gradient operators
show(io::IO, _::GradientOperator) = print(io, "∇vₕ")
show(io::IO, op::ScaledOperator{OP}) where OP<:GradientOperator = print(io, "$(_process_scalar(op.scalar)) * $(op.operator)")
show(io::IO, op::ScaledOperator{OP1}) where {OP<:GradientOperator,OP1<:ScaledOperator{OP}} = print(io, "$(op.operator)")

"""
	codomaintype(op::ScaledGradientOperator)

Return the codomain type of a scaled gradient operator.
"""
@inline codomaintype(_::ScaledGradientOperator{S,V}) where {S,V} = codomaintype(V)

"""
	scalar(op::ScaledGradientOperator)

Evaluate and return the scalar coefficient of a scaled gradient operator.
"""
@inline scalar(op::ScaledGradientOperator) = op.scalar()

"""
	parent_operator(op::ScaledGradientOperator)

Return the underlying gradient operator.
"""
@inline parent_operator(op::ScaledGradientOperator) = op.operator

#------------------------------------------------------------------------------------------#
# Algebraic Rules for Gradient Operators
#------------------------------------------------------------------------------------------#

"""
	ScaledGradientOperator(S::AbstractSpaceType, α, op::ZeroOperator)

Scaling a gradient by zero returns the zero operator.
"""
ScaledGradientOperator(_::AbstractSpaceType, _, op::ZeroOperator) = op

"""
	ScaledGradientOperator(S::AbstractSpaceType, α)

Construct a scaled gradient operator `α * ∇ₕ`.

# Examples

```julia
op = ScaledGradientOperator(Wh, 0.5)  # 0.5 * ∇ₕ
```
"""
ScaledGradientOperator(S::AbstractSpaceType, α) = ScaledGradientOperator(S, α, GradientOperator(S))

"""
	∇₋ₕ(op::IdentityOperator) -> GradientOperator

Apply the gradient operator to an identity operator, returning a `GradientOperator`.

This is typically used in weak form expressions like `∇₋ₕ(u)` where `u` is represented
by an identity operator.

# Examples

```julia
I = IdentityOperator(Wh)
∇_u = ∇₋ₕ(I)  # Gradient operator
```
"""
@inline ∇₋ₕ(op::IdentityOperator) = GradientOperator(op.space)
