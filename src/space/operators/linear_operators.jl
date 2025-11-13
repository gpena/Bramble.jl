"""
# Linear Operators Module

This module provides a complete operator algebra for finite element computations.
Operators act on grid space elements and support arithmetic operations like scaling,
addition, and composition.

## Module Organization

This file serves as an orchestrator that includes specialized submodules:

  - `operator_types.jl`: Base type hierarchy and helper functions
  - `scalar_operators.jl`: Identity, Zero, and Scaled operators
  - `differential_operators.jl`: Gradient operator
  - `composite_operators.jl`: Addition and Vector operators
  - `operator_inner_products.jl`: Specialized inner product methods

## Operator Types

  - `IdentityOperator`: Identity transformation I
  - `ZeroOperator`: Zero transformation  
  - `ScaledOperator`: Scaled operator α*Op
  - `GradientOperator`: Gradient operator ∇
  - `AddOperator`: Sum/difference of operators Op₁ ± Op₂
  - `VectorOperator`: Tuple of component operators

## Inner Products

  - `innerₕ`: L² inner product using trapezoidal weights
  - `inner₊`: L² inner product using forward difference weights

## Usage Examples

```julia
# Create a grid space
mesh = Mesh1D(0.0, 1.0, 100)
Wh = gridspace(mesh)

# Define operators
I = IdentityOperator(Wh)
∇ = GradientOperator(Wh)
α = 2.5

# Operator algebra
scaled = α * I          # ScaledOperator
sum_op = I + ∇          # AddOperator
diff_op = I - ∇         # AddOperator with negative sign

# Apply to element
uₕ = element(Wh)
∫u = innerₕ(uₕ, I)      # Mass integral
∫∇u = inner₊(uₕ, ∇)     # Gradient integral
```

## Refactoring Note

This file was refactored from a single 830-line file into 5 focused modules
to improve maintainability and code organization (see PACKAGE_REVIEW.md).
The original functionality is preserved.

See also: [`OperatorType`](@ref), [`VectorElement`](@ref), [`gridspace`](@ref)
"""#========================================##========================================#

#   Include Submodules                   #

# Base type hierarchy and helper functions
include("operator_types.jl")

# Basic scalar operators (Identity, Zero, Scaled)
include("scalar_operators.jl")

# Differential operators (Gradient)
include("differential_operators.jl")

# Composite operators (Add, Vector)
include("composite_operators.jl")

# Specialized inner product methods
include("operator_inner_products.jl")

