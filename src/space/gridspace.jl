#=
# gridspace.jl

Core abstractions for function spaces on structured grids.

## Overview

- Abstract types: `AbstractSpaceType`, `InnerProductType`.
- Element types: `VectorElement`, a wrapper for grid functions.
- Interface functions: the methods any concrete space implementation must provide.

## Design

The space framework uses Julia's type system and multiple dispatch to:
1. Maintain type stability through compile-time information (`Val`, type parameters)
2. Enable specialized implementations for scalar vs vector fields via `AbstractSpaceType{N}`
3. Support different discrete inner products (`InnerProductType`)
4. Keep data (vectors/matrices) and context (spaces) separate

## Usage

```julia
# Create a space from a mesh
Wₕ = gridspace(Ωₕ)

# Create elements (grid functions)
uₕ = element(Wₕ)
uₕ = Rₕ(Wₕ, x -> sin(x[1]))

# Apply operators
vₕ = D₊ₓ(uₕ)  # Differentiate in x
wₕ = M₋ᵧ(vₕ)  # Average in y

# Compute inner products
norm = normₕ(uₕ)
ip = innerₕ(uₕ, vₕ)
```

See also: [`ScalarGridSpace`](@ref), [`VectorElement`](@ref)
=#

"""
Subscript Unicode symbols for x, y, z coordinates used in operator notation.

These symbols are used to generate directional operator aliases via metaprogramming.

# Examples
- `D₊ₓ` - forward difference in x-direction
- `M₋ᵧ` - backward average in y-direction
- `jump₂` - jump in the z-direction

See also: [`_BRAMBLE_var2label`](@ref)
"""
const _BRAMBLE_var2symbol = ("ₓ", "ᵧ", "₂")

"""
Coordinate axis labels used in documentation and error messages.

Corresponds to the x, y, and z spatial dimensions (1st, 2nd, and 3rd dimensions).

See also: [`_BRAMBLE_var2symbol`](@ref)
"""
const _BRAMBLE_var2label = ("x", "y", "z")
"""
	AbstractSpaceType{N}

Abstract supertype for all function spaces defined on a mesh.

This is the top-level abstraction for a grid-based function space. The parameter `N` represents the number of components of the field (e.g., `N=1` for a scalar field, `N=3` for a 3D vector field).
"""
abstract type AbstractSpaceType{N} end

"""
	$(TYPEDEF)

Represents a **grid function** (a vector) that belongs to a specific function space.

This is a wrapper that bundles the raw numerical data (the vector `data`) with its parent `space`. The `space` provides the essential context, such as the underlying mesh and associated operators. By subtyping `AbstractVector`, a [`VectorElement`](@ref) can be used just like a regular Julia vector in most operations.

# Fields

$(FIELDS)
"""
struct VectorElement{S, T, VT <: AbstractVector{T}} <: AbstractVector{T}
    "the raw vector data containing the degrees of freedom."
    data::VT
    "the parent function space to which this vector belongs."
    space::S
end

"""
	InnerProductType

Abstract type for selecting which discrete inner product formula to use.

Different inner product types correspond to different weight distributions on the grid,
used in various finite difference schemes and stability analyses. The choice of inner
product affects energy estimates and numerical stability properties.

# Subtypes
- [`Innerh`](@ref): Standard ``L^2`` inner product using cell measures (volumes)
- [`Innerplus`](@ref): Modified inner product using staggered grid spacings

# Background
In finite difference methods, different inner products arise naturally from:
- Summation-by-parts (SBP) operators
- Energy method stability analysis
- Discrete integration formulas

The standard inner product (`Innerh`) uses cell volumes as weights, while the 
modified inner products (`Innerplus`) use combinations of forward/backward spacings,
appearing in discrete energy estimates for difference operators.

# Usage
```julia
# Compute standard L² inner product
result = innerₕ(uₕ, vₕ)  # Uses Innerh() internally

# Compute modified inner product in x-direction
result = inner₊ₓ(uₕ, vₕ)  # Uses Innerplus() internally
```

See also: [`Innerh`](@ref), [`Innerplus`](@ref), [`innerₕ`](@ref), [`inner₊ₓ`](@ref)
"""
abstract type InnerProductType end

"""
	Innerplus <: InnerProductType

Selector for modified discrete ``L^2`` inner products using staggered grid spacings.

These inner products use a combination of forward spacings ``h_i`` and centered cell 
widths ``h_{i+1/2}``, appearing naturally in energy estimates for finite difference 
operators. Different spatial directions may have different weight formulas.

The modified inner products are used for:
- Proving discrete energy stability
- Analyzing discrete conservation properties
- Constructing stable finite difference schemes

# Mathematical form
For a 2D grid in the x-direction:
```math
(u_h, v_h)_{+x} = \\sum_{i,j} h_{x,i} h_{y,j+1/2} u_h(x_i, y_j) v_h(x_i, y_j)
```

# Example
```julia
# These functions use Innerplus internally
result_x = inner₊ₓ(uₕ, vₕ)  # Modified inner product, x-direction
result_y = inner₊ᵧ(uₕ, vₕ)  # Modified inner product, y-direction
```

See also: [`InnerProductType`](@ref), [`Innerh`](@ref), [`inner₊ₓ`](@ref), [`weights`](@ref)
"""
struct Innerplus <: InnerProductType end

"""
	Innerh <: InnerProductType

Selector for the standard discrete ``L^2`` inner product weighted by cell measures.

The weights are the volumes (1D: lengths, 2D: areas, 3D: volumes) of grid cells,
denoted ``|\\square_k|``. This is the most common inner product for finite difference
methods and corresponds to the trapezoid rule for integration on non-uniform grids.

# Mathematical form
For a 2D grid:
```math
(u_h, v_h)_h = \\sum_{i,j} |\\square_{i,j}| u_h(x_i, y_j) v_h(x_i, y_j)
```

where ``|\\square_{i,j}|`` is the area of the cell centered at ``(x_i, y_j)``.

# Example
```julia
# Compute L² inner product
result = innerₕ(uₕ, vₕ)  # Uses Innerh() internally

# Compute L² norm
norm_value = normₕ(uₕ)  # Equivalent to sqrt(innerₕ(uₕ, uₕ))
```

See also: [`InnerProductType`](@ref), [`Innerplus`](@ref), [`innerₕ`](@ref), [`normₕ`](@ref)
"""
struct Innerh <: InnerProductType end

#=
The following functions define the **mandatory interface** for any concrete subtype
of `AbstractSpaceType`. Any struct that subtypes `AbstractSpaceType` must implement
these methods to be considered a valid function space in this framework. They are
declared without methods, so a space that omits one fails with a `MethodError` naming
the exact missing signature.
=#

"""
	space(Wₕ::AbstractSpaceType)

Returns the function space `Wₕ` itself.
"""
@inline space(Wₕ::AbstractSpaceType) = return Wₕ

"""
	mesh(Wₕ::AbstractSpaceType)

Returns the underlying mesh object associated with the function space `Wₕ`.
"""
function mesh end

"""
	mesh_type(Wₕ::AbstractSpaceType)

Returns the type of the mesh associated with the function space `Wₕ`. Also works if the argument is the type of the space.
"""
function mesh_type end

"""
	$(TYPEDSIGNATURES)

Returns the computational backend associated with the space `Wₕ`.
"""
function backend end

"""
	dim(Wₕ::AbstractSpaceType)

Returns the spatial dimension of the mesh associated with the functionpace `Wₕ`.
"""
function dim end

"""
	ndofs(Wₕ::AbstractSpaceType, [::Type{Tuple}])

Returns the total number of degrees of freedom (DOFs) in the function space `Wₕ`.
If `Tuple` is passed, it returns a tuple with the number of DOFs in each dimension.
"""
function ndofs end

"""
	ncomponents(Wₕ::AbstractSpaceType)
	ncomponents(::Type{<:AbstractSpaceType})

Returns the number of field components of the function space (e.g. 1 for scalar, D for vector).
"""
@inline ncomponents(::AbstractSpaceType{N}) where {N} = N
@inline ncomponents(::Type{<:AbstractSpaceType{N}}) where {N} = N

"""
	spaces(Wₕ::AbstractSpaceType)

Returns the constituent subspace(s) of `Wₕ` as a tuple.
"""
@inline spaces(Wₕ::AbstractSpaceType) = (Wₕ,)
