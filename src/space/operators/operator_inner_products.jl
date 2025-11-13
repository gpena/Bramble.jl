"""
# operator_inner_products.jl

This file implements specialized inner product methods for operators.

## Inner Products

Two types of inner products are provided:

- `innerₕ`: L² inner product using trapezoidal (central) integration weights
- `inner₊`: L² inner product using forward difference weights

Each has both allocating and in-place (`!`) variants for performance.

## Usage

Inner products compute ⟨op(uₕ), 1⟩ using appropriate quadrature weights:
- `innerₕ` uses trapezoidal weights for standard L² norms
- `inner₊` uses forward difference weights for proper integration with difference operators

## Examples

```julia
uₕ = element(Wh)
I = IdentityOperator(Wh)
∇ = GradientOperator(Wh)

# Compute mass integral
∫u = innerₕ(uₕ, I)

# Compute gradient integral
∫∇u = inner₊(uₕ, ∇)
```

See also: [`innerₕ`](@ref), [`inner₊`](@ref), [`IdentityOperator`](@ref), [`GradientOperator`](@ref)
"""

#########################################
#                                       #
#            innerₕ Methods             #
#                                       #
#########################################

"""
	innerₕ(l::OperatorType, uₕ::VectorElement) -> Vector

Compute the L² inner product ⟨l(uₕ), 1⟩ using trapezoidal integration weights.

The arguments can be in either order; they are automatically commuted.

# Examples

```julia
uₕ = element(Wh)
I = IdentityOperator(Wh)
∫u = innerₕ(I, uₕ)  # or innerₕ(uₕ, I)
```
"""
innerₕ(l::OperatorType, uₕ::VectorElement) = innerₕ(uₕ, l)

"""
	innerₕ(uₕ::VectorElement, ::IdentityOperator) -> Vector

Compute ⟨uₕ, 1⟩ using trapezoidal weights: `wₕ .* uₕ.values`

This implements the standard L² inner product for identity operators,
using trapezoidal quadrature weights appropriate for centered differences.
"""
@inline innerₕ(uₕ::VectorElement, _::IdentityOperator) = innerh_weights(space(uₕ)) .* uₕ.values

"""
	innerₕ!(vₕ::AbstractVector, uₕ::VectorElement, ::IdentityOperator)

In-place version: `vₕ .+= wₕ .* uₕ.values`

Accumulates the inner product into the existing vector `vₕ`, useful for
assembling contributions from multiple terms.

# Examples

```julia
vₕ = zeros(length(uₕ.values))
innerₕ!(vₕ, uₕ, I)  # vₕ now contains ∫uₕ
```
"""
@inline function innerₕ!(vₕ::AbstractVector, uₕ::VectorElement, _::IdentityOperator)
	@assert length(vₕ) == length(uₕ.values)
	vₕ .+= innerh_weights(space(uₕ)) .* uₕ.values
end

"""
	innerₕ(uₕ::VectorElement, l::ScaledOperator) -> Vector

Compute ⟨α * uₕ, 1⟩ = α * ⟨uₕ, 1⟩ using trapezoidal weights.

For scaled identity operators, this scales the inner product by the coefficient.

# Examples

```julia
α = 2.5
I = IdentityOperator(Wh)
op = α * I
∫αu = innerₕ(uₕ, op)  # Same as α * innerₕ(uₕ, I)
```
"""
@inline function innerₕ(uₕ::VectorElement, l::ScaledOperator)
	return innerh_weights(space(uₕ)) .* l.scalar() .* uₕ.values
end

"""
	innerₕ!(vₕ::AbstractVector, uₕ::VectorElement, l::ScaledOperator)

In-place version: `vₕ .+= wₕ .* α .* uₕ.values`

Efficiently computes the scaled inner product in-place using fused operations.
"""
@inline function innerₕ!(vₕ::AbstractVector, uₕ::VectorElement, l::ScaledOperator)
	vₕ .+= innerh_weights(space(uₕ)) .* l.scalar() .* uₕ.values
end

#########################################
#                                       #
#            inner₊ Methods             #
#                                       #
#########################################

"""
	inner₊(l::OperatorType, uₕ::VectorElement) -> Vector

Compute the L² inner product ⟨l(uₕ), 1⟩ using forward difference integration weights.

The arguments can be in either order; they are automatically commuted.
This inner product is used with difference operators for proper integration.

# Use Cases

  - Integration of gradient operators (⟨∇uₕ, 1⟩)
  - Weak form assembly with finite difference operators
  - Integration with forward difference stencils

# Examples

```julia
∇ = GradientOperator(Wh)
uₕ = element(Wh)
∫∇u = inner₊(∇, uₕ)  # or inner₊(uₕ, ∇)
```
"""
inner₊(l::OperatorType, uₕ::VectorElement) = inner₊(uₕ, l)

"""
	inner₊(uₕ::VectorElement, ::IdentityOperator) -> Vector

Compute ⟨uₕ, 1⟩ using forward difference weights: `w₊ .* uₕ.values`

Uses forward difference quadrature weights, consistent with forward
difference operators.
"""
@inline inner₊(uₕ::VectorElement, _::IdentityOperator) = innerplus_weights(space(uₕ), Val(1)) .* uₕ.values

"""
	inner₊!(vₕ::AbstractVector, uₕ::VectorElement, ::IdentityOperator)

In-place version: `vₕ .+= w₊ .* uₕ.values`

Accumulates the forward-weighted inner product into `vₕ`.
"""
@inline function inner₊!(vₕ::AbstractVector, uₕ::VectorElement, _::IdentityOperator)
	@assert length(vₕ) == length(uₕ.values)
	vₕ .+= innerplus_weights(space(uₕ), Val(1)) .* uₕ.values
end

"""
	inner₊(uₕ::VectorElement, l::ScaledOperator) -> Vector

Compute ⟨α * uₕ, 1⟩ = α * ⟨uₕ, 1⟩ using forward difference weights.

For scaled identity operators with forward difference weights.
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

```math
\\sum_{i=1}^D D_i^T \\cdot \\text{diag}(w_{+,i}) \\cdot u_h
```

where ``D_i`` is the backward difference matrix in dimension ``i``.

# Implementation

The gradient operator in each coordinate direction is represented by the
transpose of a backward difference matrix, weighted by forward difference
quadrature weights. The contributions from all dimensions are summed.

# Returns

Vector of length equal to the number of degrees of freedom.

# Examples

```julia
∇ = GradientOperator(Wh)
uₕ = element(Wh)
∫∇u = inner₊(uₕ, ∇)  # Integral of gradient
```
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
This is the workhorse method for gradient integration in weak forms.

# Algorithm

For each spatial dimension ``i``:

 1. Weight the element values by forward difference weights: `w₊,ᵢ .* uₕ`
 2. Apply transpose of backward difference matrix: `Dᵢᵀ * (w₊,ᵢ .* uₕ)`
 3. Accumulate into result vector

# Performance

Uses pre-allocated cache vectors (`vec_cache` and `diff_matrix_cache`)
from the grid space to avoid allocations.
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
a vector-valued function space where each component is represented by
a separate `VectorElement`.

# Arguments

  - `uₕ::NTuple{D,VectorElement}`: Tuple of D vector elements, one per dimension
  - `l::GradientOperator`: Gradient operator

# Returns

Vector accumulating contributions from all gradient components.

# Examples

```julia
# For a 2D vector field (u, v)
uₕ = (element(Wh), element(Wh))  # u and v components
∇ = GradientOperator(Wh)
∫∇u = inner₊(uₕ, ∇)
```
"""
@inline function inner₊(uₕ::NTuple{D,VectorElement}, l::GradientOperator) where D
	res = similar(first(uₕ).values)
	res .= 0
	inner₊!(res, uₕ, l)

	return res
end

"""
	inner₊!(vₕ::AbstractVector, uₕ::NTuple{D,VectorElement}, l::GradientOperator)

In-place version for tuple of vector elements.

Each component `uₕ[i]` contributes to the result through its corresponding
dimension's difference matrix and weights.

# Algorithm

For each dimension ``i``:

 1. Weight the i-th component: `w₊,ᵢ .* uₕ[i]`
 2. Apply transpose of i-th difference matrix
 3. Accumulate into result

This is used in weak forms for vector-valued problems.
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
