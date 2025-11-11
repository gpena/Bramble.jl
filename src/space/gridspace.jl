#=
# gridspace.jl

This file defines the core abstractions for function spaces on structured grids.

## Key Components

- **Abstract Types**: `AbstractSpaceType`, `ComponentStyle`, `InnerProductType`
- **Element Types**: `VectorElement`, `MatrixElement` - wrappers for grid functions and operators
- **Interface Functions**: Required methods for any concrete space implementation

## Design Philosophy

The space framework uses Julia's type system and multiple dispatch to:
1. Maintain type stability through compile-time information (`Val`, type parameters)
2. Enable specialized implementations for scalar vs vector fields (`ComponentStyle`)
3. Support different discrete inner products (`InnerProductType`)
4. Provide a clean separation between data (vectors/matrices) and context (spaces)

## Usage Pattern

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

See also: [`ScalarGridSpace`](@ref), [`VectorElement`](@ref), [`MatrixElement`](@ref)
=#

"""
Subscript Unicode symbols for x, y, z coordinates used in operator notation.

These symbols are used to generate directional operator aliases via metaprogramming.

# Examples
- `D₊ₓ` - forward difference in x-direction
- `M₋ᵧ` - backward average in y-direction
- `jump₊₂` - forward jump in z-direction

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

This is a wrapper that bundles the raw numerical data (the vector `data`) with its parent `space`. The `space` provides the essential context, such as the underlying mesh and associated operators. By subtyping `AbstractVector`, a [VectorElement](@ref) can be used just like a regular Julia vector in most operations.

# Fields

$(FIELDS)
"""
struct VectorElement{S,T,VT<:AbstractVector{T}} <: AbstractVector{T}
	"the raw vector data containing the degrees of freedom."
	data::VT
	"the parent function space to which this vector belongs."
	space::S
end

"""
	$(TYPEDEF)

Represents a **discrete linear operator** (a matrix) that acts on a function space.

Similar to , this container bundles a raw matrix (`data`) with its parent `space`. This is used to represent discretization matrices from methods like finite differences (e.g., a differentiation or averaging matrix). Subtyping `AbstractMatrix` allows it to be used like a standard Julia matrix.

# Fields

$(FIELDS)
"""
struct MatrixElement{S,T,MT<:AbstractMatrix{T}} <: AbstractMatrix{T}
	"the matrix data representing the linear operator."
	data::MT
	"the parent function space to which this vector belongs."
	space::S
end

"""
	ComponentStyle

Abstract type for compile-time dispatch on the number of field components in a function space.

The `ComponentStyle` hierarchy is used to specialize algorithms for scalar fields (single component) 
versus vector fields (multiple components). This pattern enables efficient code generation through 
Julia's multiple dispatch, avoiding runtime `if/else` checks.

# Subtypes
- [`SingleComponent`](@ref): For scalar fields (e.g., temperature, pressure)
- [`MultiComponent{D}`](@ref): For vector fields with `D` components (e.g., velocity, displacement)

# Usage
```julia
ComponentStyle(typeof(Wₕ))  # Returns SingleComponent() or MultiComponent{D}()
```

# Example
```julia
Wₕ = gridspace(Ωₕ)  # Scalar space
ComponentStyle(typeof(Wₕ))  # Returns SingleComponent()

# Dispatch example
to_matrix(uₕ, ::SingleComponent) = # ... scalar implementation
to_matrix(uₕ, ::MultiComponent{D}) where D = # ... vector implementation
```

See also: [`SingleComponent`](@ref), [`MultiComponent`](@ref), [`to_matrix`](@ref)
"""
abstract type ComponentStyle end

"""
	SingleComponent <: ComponentStyle

Indicates a function space for scalar fields (single component per grid point).

This is the component style for spaces representing scalar quantities such as 
temperature, pressure, or density, where each grid point has a single value.

# Example
```julia
Wₕ = gridspace(Ωₕ)  # Creates a scalar grid space
ComponentStyle(typeof(Wₕ))  # Returns SingleComponent()

uₕ = element(Wₕ)  # Vector has npoints(Ωₕ) entries
```

See also: [`ComponentStyle`](@ref), [`MultiComponent`](@ref), [`ScalarGridSpace`](@ref)
"""
struct SingleComponent <: ComponentStyle end

"""
	MultiComponent{D} <: ComponentStyle

Indicates a function space for vector fields with `D` components per grid point.

This component style is used for vector-valued quantities such as velocity, 
displacement, or force fields, where each grid point stores `D` scalar values.

# Type Parameter
- `D::Int`: Number of vector components (typically equal to spatial dimension)

# Example
```julia
# Create a 2D vector space (e.g., for velocity field)
Wₕ = gridspace(Ωₕ)
Vₕ = CompositeGridSpace((Wₕ, Wₕ))  # 2-component vector space
ComponentStyle(typeof(Vₕ))  # Returns MultiComponent{2}()

uₕ = element(Vₕ)  # Vector has 2 * npoints(Ωₕ) entries
```

See also: [`ComponentStyle`](@ref), [`SingleComponent`](@ref), [`CompositeGridSpace`](@ref)
"""
struct MultiComponent{D} <: ComponentStyle end

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

The modified inner products are essential for:
- Proving discrete energy stability
- Analyzing discrete conservation properties
- Constructing stable finite difference schemes

# Mathematical Form
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

# Mathematical Form
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
these methods to be considered a valid function space in this framework. The default
implementations throw an error, guiding developers to provide the correct implementation.
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
@inline function mesh(Wₕ::AbstractSpaceType)
	error("Interface function 'mesh' not implemented for $(typeof(Wₕ))")
end

"""
	mesh_type(Wₕ::AbstractSpaceType)

Returns the type of the mesh associated with the function space `Wₕ`. Also works if the argument is the type of the space.
"""
@inline function mesh_type(Wₕ::AbstractSpaceType)
	error("Interface function 'mesh_type' not implemented for $(typeof(Wₕ))")
end
@inline function mesh_type(::Type{<:AbstractSpaceType})
	error("Interface function 'mesh_type' not implemented for this space type.")
end

"""
	backward_difference_matrix(Wₕ::AbstractSpaceType, i)

Returns the backward difference matrix for the `i`-th dimension of the space `Wₕ`.
"""
@inline function backward_difference_matrix(Wₕ::AbstractSpaceType, i)
	error("Interface function 'backward_difference_matrix' not implemented for $(typeof(Wₕ))")
end

"""
	average_matrix(Wₕ::AbstractSpaceType, i)

Returns the averaging matrix for the `i`-th dimension of the space `Wₕ`.
"""
@inline function average_matrix(Wₕ::AbstractSpaceType, i)
	error("Interface function 'average_matrix' not implemented for $(typeof(Wₕ))")
end

"""
	vector_buffer(Wₕ::AbstractSpaceType)

Returns the [GridSpaceBuffer](@ref) used for efficient memory management in the space `Wₕ`.
"""
@inline function vector_buffer(Wₕ::AbstractSpaceType)
	error("Interface function 'vector_buffer' not implemented for $(typeof(Wₕ))")
end

"""
	has_backward_difference_matrix(Wₕ::AbstractSpaceType)

Checks if the backward difference matrices have been computed and stored for `Wₕ`.
"""
@inline function has_backward_difference_matrix(Wₕ::AbstractSpaceType)
	error("Interface function 'has_backward_difference_matrix' not implemented for $(typeof(Wₕ))")
end

"""
	has_average_matrix(Wₕ::AbstractSpaceType)

Checks if the averaging matrices have been computed and stored for `Wₕ`.
"""
@inline function has_average_matrix(Wₕ::AbstractSpaceType)
	error("Interface function 'has_average_matrix' not implemented for $(typeof(Wₕ))")
end

"""
	$(TYPEDSIGNATURES)

Returns the computational backend associated with the space `Wₕ`.
"""
@inline function backend(Wₕ::AbstractSpaceType)
	error("Interface function 'backend' not implemented for $(typeof(Wₕ))")
end

"""
	dim(Wₕ::AbstractSpaceType)

Returns the spatial dimension of the mesh associated with the functionpace `Wₕ`.
"""
@inline function dim(Wₕ::AbstractSpaceType)
	error("Interface function 'dim' not implemented for $(typeof(Wₕ))")
end

"""
	ndofs(Wₕ::AbstractSpaceType, [::Type{Tuple}])

Returns the total number of degrees of freedom (DOFs) in the function space `Wₕ`.
If `Tuple` is passed, it returns a tuple with the number of DOFs in each dimension.
"""
@inline function ndofs(Wₕ::AbstractSpaceType)
	error("Interface function 'ndofs' not implemented for $(typeof(Wₕ))")
end
@inline function ndofs(Wₕ::AbstractSpaceType, ::Type{Tuple})
	error("Interface function 'ndofs' not implemented for $(typeof(Wₕ))")
end

"""
	eltype(Wₕ::AbstractSpaceType)

Returns the element type (e.g., `Float64`) of the data in the function space `Wₕ`. It also works if the argument is the type of the space.
"""
@inline function eltype(Wₕ::AbstractSpaceType)
	error("Interface function 'eltype' not implemented for $(typeof(Wₕ))")
end
@inline function eltype(::Type{W}) where W<:AbstractSpaceType
	error("Interface function 'eltype' not implemented for this space type.")
end
