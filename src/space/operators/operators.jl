"""
# Operators Module

This module defines operator types and their algebra for finite element computations.
Operators act on grid space elements and support arithmetic operations like scaling,
addition, and composition.

## Operator Types

- `IdentityOperator`: Identity transformation I
- `ZeroOperator`: Zero transformation  
- `ScaledOperator`: Scaled operator α*Op
- `GradientOperator`: Gradient operator ∇
- `AddOperator`: Sum/difference of operators Op₁ ± Op₂
- `VectorOperator`: Tuple of component operators

## Helper Functions

Internal utilities for wrapping scalars and processing operator expressions.
"""

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
"""
@inline _scalar2wrapper(::Type{T}, α::ElType) where {T,ElType<:Number} = FunctionWrapper{T,Tuple{}}(() -> convert(T, α)::T)
@inline _scalar2wrapper(::Type{T}, α) where T = FunctionWrapper{typeof(α),Tuple{}}(() -> α)

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
"""
_process_scalar(f::FunctionWrapper{T,Tuple{}}) where T<:Number = "$(f())"
_process_scalar(f::FunctionWrapper{CoType,Tuple{}}) where CoType = "uₕ"

"""
	OperatorType <: BrambleType

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
"""
abstract type OperatorType <: BrambleType end

"""
	ScalarOperatorType <: OperatorType

Abstract type for operators with scalar-valued codomain.
"""
abstract type ScalarOperatorType <: OperatorType end

"""
	VectorOperatorType <: OperatorType

Abstract type for operators with vector-valued codomain.
"""
abstract type VectorOperatorType <: OperatorType end

# Basic operator interface
@inline *(op::OperatorType, α) = α * op
@inline eltype(op::OperatorType) = eltype(space(op))

"""
	space(op::OperatorType) -> AbstractSpaceType

Return the grid space on which the operator `op` acts.
"""
@inline space(op::OperatorType) = op.space

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

# Properties

  - `scalar(op)` returns zero
  - `α * ZeroOperator` returns `ZeroOperator` for any `α`
  - `op + ZeroOperator` returns `op` for any operator `op`
"""
struct ZeroOperator{S} <: OperatorType
	space::S
end

show(io::IO, _::ZeroOperator) = print(io, "0")

# Algebraic rules for zero operator
@inline *(α, op::ZeroOperator) = op
@inline ⋅(α, op::ZeroOperator) = op

"""
	scalar(op::ZeroOperator)

Return the scalar representation of the zero operator (zero of the space element type).
"""
scalar(op::ZeroOperator) = zero(eltype(space(op)))

#########################################
#                                       #
#            Vector Operator            #
#                                       #
#########################################

"""
	VectorOperator{S,CompType} <: OperatorType

Represents a vector-valued operator as a tuple of component operators.

This allows construction of operators acting on product spaces S × S × ... × S,
where each component space has its own operator.

# Fields

  - `space::S`: The underlying grid space
  - `component_operators::CompType`: Tuple of operators for each component

```
```
"""
struct VectorOperator{S,CompType} <: OperatorType
	space::S
	component_operators::CompType
end

show(io::IO, op::VectorOperator) = print(io, "0")

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

# Properties

  - `scalar(I)` returns one (multiplicative identity)
  - `α * I` creates a `ScaledOperator` with coefficient `α`
  - `I * α` is equivalent to `α * I`

# Examples

```julia
Wh = gridspace(mesh)
I = IdentityOperator(Wh)
op = 3.5 * I         # Scaled identity
uₕ = element(Wh)
result = innerₕ(uₕ, I)  # Identity in inner product
```
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

@inline function ⋅(α, op::IdentityOperator)
	if α isa Number
		return ScaledOperator(op.space, α, op)
	end

	if α isa VectorElement
		return ScaledOperator(op.space, α, op)
	end

	if α isa Tuple
		@show "Tuple"

		#elem = ntuple(i -> op, length(α))
		#return VectorOperator(space(op), elem)
	end

	@error "Don't know how to handle this expression"
end

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

# Constructor

```julia
ScaledOperator(space, α, operator)
```

# Properties

  - `scalar(op)` evaluates and returns the scalar coefficient
  - `parent_operator(op)` returns the underlying operator
  - `codomaintype(op)` returns the codomain type of the scalar

# Algebraic simplification

  - `ScaledOperator(S, 0, op)` returns `ZeroOperator(S)`
  - `ScaledOperator(S, 1, op)` returns `op` unchanged
  - `α * (β * op)` combines to `(α*β) * op`

# Examples

```julia
I = IdentityOperator(Wh)
op1 = ScaledOperator(Wh, 2.5, I)    # 2.5 * I
op2 = 3 * op1                         # (3 * 2.5) * I = 7.5 * I
```
"""
struct ScaledOperator{OP,S,V} <: OperatorType
	space::S
	scalar::V
	operator::OP
end

@inline codomaintype(_::ScaledOperator{OP,S,V}) where {OP,S,V} = codomaintype(V)

"""
	scalar(op::ScaledOperator)

Evaluate and return the scalar coefficient of the scaled operator.
"""
@inline scalar(op::ScaledOperator) = op.scalar()

"""
	parent_operator(op::ScaledOperator)

Return the underlying operator being scaled.
"""
@inline parent_operator(op::ScaledOperator) = op.operator

# Display methods for scaled operators
show(io::IO, op::ScaledOperator{OP}) where OP<:IdentityOperator = print(io, "$(_process_scalar(op.scalar)) * I")
show(io::IO, op::ScaledOperator{OP1}) where {OP2<:IdentityOperator,OP1<:ScaledOperator{OP2}} = print(io, "$(_process_scalar(op.scalar)) * I")
show(io::IO, op::ScaledOperator{OP}) where OP<:OperatorType = print(io, "$(_process_scalar(op.scalar)) * ($(op.operator))")

# Algebraic rules for scaled operators

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

# Multiplication rules for identity operator
@inline *(α, op::IdentityOperator) = ScaledOperator(space(op), α)
@inline *(op::IdentityOperator, α) = α * op

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

# Type Alias

```julia
ScaledGradientOperator{S,V} = ScaledOperator{GradientOperator{S},S,V}
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
"""
struct GradientOperator{S} <: OperatorType
	space::S
end

"""
	ScaledGradientOperator{S,V}

Type alias for a scaled gradient operator `α * ∇ₕ`.
"""
ScaledGradientOperator{S,V} = ScaledOperator{GradientOperator{S},S,V}

scalar(op::GradientOperator) = one(eltype(space(op)))
codomaintype(op::GradientOperator) = eltype(space(op))

# Display methods
show(io::IO, _::GradientOperator) = print(io, "∇vₕ")
show(io::IO, op::ScaledOperator{OP}) where OP<:GradientOperator = print(io, "$(_process_scalar(op.scalar)) * $(op.operator)")
show(io::IO, op::ScaledOperator{OP1}) where {OP<:GradientOperator,OP1<:ScaledOperator{OP}} = print(io, "$(op.operator)")

@inline codomaintype(_::ScaledGradientOperator{S,V}) where {S,V} = codomaintype(V)
@inline scalar(op::ScaledGradientOperator) = op.scalar()
@inline parent_operator(op::ScaledGradientOperator) = op.operator

# Algebraic rules for (scaled) gradient operators
ScaledGradientOperator(_::AbstractSpaceType, _, op::ZeroOperator) = op
ScaledGradientOperator(S::AbstractSpaceType, α) = ScaledGradientOperator(S, α, GradientOperator(S))

"""
	∇₋ₕ(op::IdentityOperator) -> GradientOperator

Apply the gradient operator to an identity operator, returning a `GradientOperator`.

This is typically used in weak form expressions like `∇₋ₕ(u)` where `u` is represented
by an identity operator.
"""
@inline ∇₋ₕ(op::IdentityOperator) = GradientOperator(op.space)

#########################################
#                                       #
#         Add/Subtract Operators        #
#                                       #
#########################################

"""
	AddOperator{OP1,OP2,S} <: OperatorType

Represents the sum or difference of two operators: `op₁ ± op₂`.

This operator type enables algebraic manipulation of operator expressions,
storing the operation as `op₁ + scalar*op₂` where `scalar ∈ {-1, 1}`.

# Fields

  - `space::S`: The grid space on which the operators act
  - `operator1::OP1`: The first operator
  - `operator2::OP2`: The second operator
  - `scalar::Int`: Sign indicator (+1 for addition, -1 for subtraction)

# Constructor

Created automatically through `+` and `-` operations:

```julia
op_sum = op₁ + op₂
op_diff = op₁ - op₂
```

# Properties

  - Both operators must have the same codomain type
  - `first(op)` returns the first operator
  - `second(op)` returns the second operator
  - `scalar(op)` returns +1 or -1

# Algebraic simplification

  - `op + ZeroOperator` returns `op`
  - `ZeroOperator + op` returns `op`
  - `op - ZeroOperator` returns `op`
  - `ZeroOperator - op` returns `-1 * op`
"""
struct AddOperator{OP1,OP2,S} <: OperatorType
	space::S
	operator1::OP1
	operator2::OP2
	scalar::Int        # op1 + scalar*op2, scalar = 1 or -1
end

@inline function codomaintype(op::AddOperator)
	@assert codomaintype(op.operator1) == codomaintype(op.operator2)
	return codomaintype(op.operator1)
end

"""
	scalar(op::AddOperator)

Return the sign of the second operator (+1 for addition, -1 for subtraction).
"""
@inline scalar(op::AddOperator) = op.scalar

"""
	first(op::AddOperator)

Return the first operator in the sum/difference.
"""
@inline first(op::AddOperator) = op.operator1

"""
	second(op::AddOperator)

Return the second operator in the sum/difference.
"""
@inline second(op::AddOperator) = op.operator2

show(io::IO, op::AddOperator) = print(io, "$(op.operator1) $(scalar(op) == 1 ? "+" : "-") $(op.operator2)")

# Addition rules
+(op1::OP1, op2::OP2) where {OP1<:OperatorType,OP2<:OperatorType} = AddOperator{typeof(op1),typeof(op2),typeof(space(op1))}(space(op1), op1, op2, 1)
+(op1::OP1, _::ZeroOperator) where OP1<:OperatorType = op1
+(_::ZeroOperator, op2::OP2) where OP2<:OperatorType = op2

# Subtraction rules
-(op1::OP1, op2::OP2) where {OP1<:OperatorType,OP2<:OperatorType} = AddOperator(space(op1), op1, op2, -1)
-(op1::OP1, _::ZeroOperator) where {OP1<:OperatorType} = op1
-(_::ZeroOperator, op2::OP2) where {OP2<:OperatorType} = ScaledOperator(space(op2), -1, op2)

###################################
#                                 #
#   Specialized Inner Products    #
#                                 #
###################################

"""
# Inner Product Methods

This section defines specialized inner product operations between operators and
vector elements, enabling efficient assembly of finite element forms.

Two types of inner products are provided:
- `innerₕ`: L² inner product using trapezoidal weights
- `inner₊`: L² inner product using forward difference weights

Each has both allocating and in-place (`!`) variants for performance.
"""

############### innerₕ ############

"""
	innerₕ(l::OperatorType, uₕ::VectorElement) -> Vector

Compute the L² inner product ⟨l(uₕ), 1⟩ using trapezoidal integration weights.

The arguments can be in either order; they are automatically commuted.
"""
innerₕ(l::OperatorType, uₕ::VectorElement) = innerₕ(uₕ, l)

"""
	innerₕ(uₕ::VectorElement, ::IdentityOperator) -> Vector

Compute ⟨uₕ, 1⟩ using trapezoidal weights: `wₕ .* uₕ.values`
"""
@inline innerₕ(uₕ::VectorElement, _::IdentityOperator) = innerh_weights(space(uₕ)) .* uₕ.values

"""
	innerₕ!(vₕ::AbstractVector, uₕ::VectorElement, ::IdentityOperator)

In-place version: `vₕ .+= wₕ .* uₕ.values`
"""
@inline function innerₕ!(vₕ::AbstractVector, uₕ::VectorElement, _::IdentityOperator)
	@assert length(vₕ) == length(uₕ.values)
	vₕ .+= innerh_weights(space(uₕ)) .* uₕ.values
end

"""
	innerₕ(uₕ::VectorElement, l::ScaledOperator) -> Vector

Compute ⟨α * uₕ, 1⟩ = α * ⟨uₕ, 1⟩ using trapezoidal weights.
"""
@inline function innerₕ(uₕ::VectorElement, l::ScaledOperator)
	return innerh_weights(space(uₕ)) .* l.scalar() .* uₕ.values
end

"""
	innerₕ!(vₕ::AbstractVector, uₕ::VectorElement, l::ScaledOperator)

In-place version: `vₕ .+= wₕ .* α .* uₕ.values`
"""
@inline function innerₕ!(vₕ::AbstractVector, uₕ::VectorElement, l::ScaledOperator)
	vₕ .+= innerh_weights(space(uₕ)) .* l.scalar() .* uₕ.values
end

############### inner₊ ############

"""
	inner₊(l::OperatorType, uₕ::VectorElement) -> Vector

Compute the L² inner product ⟨l(uₕ), 1⟩ using forward difference integration weights.

The arguments can be in either order; they are automatically commuted.
This inner product is used with difference operators for proper integration.
"""
inner₊(l::OperatorType, uₕ::VectorElement) = inner₊(uₕ, l)

"""
	inner₊(uₕ::VectorElement, ::IdentityOperator) -> Vector

Compute ⟨uₕ, 1⟩ using forward difference weights: `w₊ .* uₕ.values`
"""
@inline inner₊(uₕ::VectorElement, _::IdentityOperator) = innerplus_weights(space(uₕ), Val(1)) .* uₕ.values

"""
	inner₊!(vₕ::AbstractVector, uₕ::VectorElement, ::IdentityOperator)

In-place version: `vₕ .+= w₊ .* uₕ.values`
"""
@inline function inner₊!(vₕ::AbstractVector, uₕ::VectorElement, _::IdentityOperator)
	@assert length(vₕ) == length(uₕ.values)
	vₕ .+= innerplus_weights(space(uₕ), Val(1)) .* uₕ.values
end

"""
	inner₊(uₕ::VectorElement, l::ScaledOperator) -> Vector

Compute ⟨α * uₕ, 1⟩ = α * ⟨uₕ, 1⟩ using forward difference weights.
"""
@inline inner₊(uₕ::VectorElement, l::ScaledOperator) = innerplus_weights(space(uₕ), Val(1)) .* l.scalar() .* uₕ.values

"""
	inner₊!(vₕ::AbstractVector, uₕ::VectorElement, l::ScaledOperator)

In-place version: `vₕ .+= w₊ .* α .* uₕ.values`

Uses fused multiply-add for efficiency.
"""
@inline function inner₊!(vₕ::AbstractVector, uₕ::VectorElement, l::ScaledOperator)
	x = innerplus_weights(space(uₕ), Val(1))
	α = l.scalar()
	u = uₕ.values
	@. vₕ += x * α * u
	nothing
end

"""
	inner₊(uₕ::VectorElement, l::GradientOperator) -> Vector

Compute ⟨∇ₕuₕ, 1⟩ using forward difference weights and backward difference matrices.

For a D-dimensional space, computes:

```
∑ᵢ₌₁ᴰ Dᵢᵀ * diag(w₊,ᵢ) * uₕ
```

where Dᵢ is the backward difference matrix in dimension i.

# Returns

Vector of length equal to the number of degrees of freedom.
"""
@inline function inner₊(uₕ::VectorElement, l::GradientOperator)
	res = similar(uₕ.values)
	res .= 0
	inner₊!(res, uₕ, l)

	return res
end

"""
	inner₊!(vₕ::AbstractVector, uₕ::VectorElement, l::GradientOperator)

In-place version of gradient inner product.

Accumulates the result into `vₕ`, summing contributions from all spatial dimensions.
"""
function inner₊!(vₕ::AbstractVector, uₕ::VectorElement, l::GradientOperator)
	W = l.space
	D = dim(mesh(W))

	for i in 1:D
		x = innerplus_weights(W, i)
		@. W.vec_cache = x * uₕ.values
		mul!(vₕ, transpose(W.diff_matrix_cache[i].values), W.vec_cache, 1, 1)
	end

	nothing
end

"""
	inner₊(uₕ::NTuple{D,VectorElement}, l::GradientOperator) -> Vector

Compute inner product of a gradient operator with a tuple of vector elements.

Used when the gradient components are provided separately, typically from
a vector-valued function space.
"""
@inline function inner₊(uₕ::NTuple{D,VectorElement}, l::GradientOperator) where D
	res = similar(uₕ.values)
	res .= 0
	inner₊!(res, uₕ, l)

	return res
end

"""
	inner₊!(vₕ::AbstractVector, uₕ::NTuple{D,VectorElement}, l::GradientOperator)

In-place version for tuple of vector elements.

Each component `uₕ[i]` contributes to the result through its corresponding
dimension's difference matrix and weights.
"""
function inner₊!(vₕ::AbstractVector, uₕ::NTuple{D,VectorElement}, l::GradientOperator) where D
	W = l.space

	for i in 1:D
		x = innerplus_weights(W, i)
		@. W.vec_cache = x * uₕ[i].values
		mul!(vₕ, transpose(W.diff_matrix_cache[i].values), W.vec_cache, 1, 1)
	end

	return nothing
end