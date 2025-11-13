"""
# composite_operators.jl

This file implements composite operators formed by combining other operators.

## Operators

- `AddOperator`: Sum or difference of two operators (op₁ ± op₂)
- `VectorOperator`: Tuple of component operators

## Algebraic Simplifications

Addition and subtraction rules include zero-operator simplifications:
- `op + ZeroOperator = op`
- `ZeroOperator + op = op`
- `op - ZeroOperator = op`
- `ZeroOperator - op = -1 * op`

See also: [`OperatorType`](@ref), [`ZeroOperator`](@ref)
"""

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

# Algebraic Simplification

  - `op + ZeroOperator` returns `op`
  - `ZeroOperator + op` returns `op`
  - `op - ZeroOperator` returns `op`
  - `ZeroOperator - op` returns `-1 * op`

# Examples

```julia
I = IdentityOperator(Wh)
∇ = GradientOperator(Wh)
op = I + 2*∇  # AddOperator
```

See also: [`first`](@ref), [`second`](@ref), [`scalar`](@ref)
"""
struct AddOperator{OP1,OP2,S} <: OperatorType
	space::S
	operator1::OP1
	operator2::OP2
	scalar::Int        # op1 + scalar*op2, scalar = 1 or -1
end

"""
	codomaintype(op::AddOperator)

Return the codomain type of the add operator.

Both operators must have matching codomain types.
"""
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

# Display method
show(io::IO, op::AddOperator) = print(io, "$(op.operator1) $(scalar(op) == 1 ? "+" : "-") $(op.operator2)")

#------------------------------------------------------------------------------------------#
# Algebraic Rules for Addition and Subtraction
#------------------------------------------------------------------------------------------#

"""
	+(op1::OperatorType, op2::OperatorType)

Add two operators element-wise.

# Examples

```julia
I = IdentityOperator(Wh)
∇ = GradientOperator(Wh)
combined = I + ∇  # AddOperator
```
"""
+(op1::OP1, op2::OP2) where {OP1<:OperatorType,OP2<:OperatorType} = AddOperator{typeof(op1),typeof(op2),typeof(space(op1))}(space(op1), op1, op2, 1)

"""
	+(op1::OperatorType, ::ZeroOperator)
	+(::ZeroOperator, op2::OperatorType)

Zero-operator simplifications for addition.
"""
+(op1::OP1, _::ZeroOperator) where OP1<:OperatorType = op1
+(_::ZeroOperator, op2::OP2) where OP2<:OperatorType = op2

"""
	-(op1::OperatorType, op2::OperatorType)

Subtract one operator from another element-wise.

# Examples

```julia
I = IdentityOperator(Wh)
∇ = GradientOperator(Wh)
diff = I - ∇  # AddOperator with scalar = -1
```
"""
-(op1::OP1, op2::OP2) where {OP1<:OperatorType,OP2<:OperatorType} = AddOperator(space(op1), op1, op2, -1)

"""
	-(op1::OperatorType, ::ZeroOperator)
	-(::ZeroOperator, op2::OperatorType)

Zero-operator simplifications for subtraction.
"""
-(op1::OP1, _::ZeroOperator) where {OP1<:OperatorType} = op1
-(_::ZeroOperator, op2::OP2) where {OP2<:OperatorType} = ScaledOperator(space(op2), -1, op2)

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

# Examples

```julia
# Create a vector operator with gradient components in each direction
ops = (∇ₓ, ∇ᵧ, ∇ᵧ)
vec_op = VectorOperator(Wh, ops)
```

See also: [`OperatorType`](@ref)
"""
struct VectorOperator{S,CompType} <: OperatorType
	space::S
	component_operators::CompType
end

# Display method
show(io::IO, op::VectorOperator) = print(io, "VectorOp($(length(op.component_operators)) components)")
