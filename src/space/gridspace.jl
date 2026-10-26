#=
# gridspace.jl

Core abstractions for function spaces on structured grids.

## Overview

- Abstract types: `AbstractSpaceType`.
- Element types: `VectorElement`, a wrapper for grid functions.
- Interface functions: the methods any concrete space implementation must provide.

## Design

The space framework uses Julia's type system and multiple dispatch to:
1. Maintain type stability through compile-time information (`Val`, type parameters)
2. Enable specialized implementations for scalar vs vector fields via `AbstractSpaceType{N}`
3. Support different discrete inner products (`Innerh`, `Innerplus`)
4. Keep data (vectors/matrices) and context (spaces) separate

## Usage

```julia
using Bramble: Mᵧ
# Create a space from a mesh
Wₕ = gridspace(Ωₕ)

# Create elements (grid functions)
uₕ = element(Wₕ)
uₕ = Rₕ(Wₕ, x -> sin(x[1]))

# Apply operators
vₕ = D₊ₓ(uₕ)  # Differentiate in x
wₕ = Mᵧ(vₕ)  # Average in y

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
- `Mᵧ` - backward average in y-direction
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
    VectorElement(data::VT, space::S)
    VectorElement{S, T, VT}(data::VT, space::S)

Represents a **grid function** (a vector) that belongs to a specific function space.

This is a wrapper that bundles the raw numerical data (the vector `data`) with its parent `space`. The `space` provides the essential context, such as the underlying mesh and associated operators. By subtyping `AbstractVector`, a [`VectorElement`](@ref) can be used just like a regular Julia vector in most operations.

Data with offset axes (for example a view over `Base.IdentityUnitRange(2:11)`) is refused with an `ArgumentError`: the element indexes its data from 1.

# Fields

  - `data::VT`: the raw vector data containing the degrees of freedom.
  - `space::S`: the parent function space to which this vector belongs.
"""
struct VectorElement{S, T, VT <: AbstractVector{T}} <: AbstractVector{T}
    "the raw vector data containing the degrees of freedom."
    data::VT
    "the parent function space to which this vector belongs."
    space::S

    function VectorElement{S, T, VT}(data, space) where {S, T, VT <: AbstractVector{T}}
        d = convert(VT, data)
        Base.require_one_based_indexing(d)
        return new{S, T, VT}(d, space)
    end
end

function VectorElement(data::VT, space::S) where {S, T, VT <: AbstractVector{T}}
    return VectorElement{S, T, VT}(data, space)
end

"""
    Innerplus

Selector for modified discrete ``L^2`` inner products using staggered grid spacings.

These inner products use a combination of backward spacings ``h_i = x_i - x_{i-1}`` (zero
at the first node) and cell widths ``h_{i+1/2}``, appearing naturally in energy
estimates for finite difference operators. Different spatial directions may have different weight formulas.

The modified inner products are used for:
- Proving discrete energy stability
- Analyzing discrete conservation properties
- Constructing stable finite difference schemes

# Mathematical form

For a 2D grid in the x-direction:
```math
(u_h, v_h)_{+x} = \\sum_{i,j} h_{x,i} h_{y,j+1/2} u_h(x_i, y_j) v_h(x_i, y_j)
```

# Examples

```julia
using Bramble: inner₊ₓ, inner₊ᵧ
# These functions use Innerplus internally
result_x = inner₊ₓ(uₕ, vₕ)  # Modified inner product, x-direction
result_y = inner₊ᵧ(uₕ, vₕ)  # Modified inner product, y-direction
```

See also: [`Innerh`](@ref), [`inner₊ₓ`](@ref), [`weights`](@ref)
"""
struct Innerplus end

"""
    Innerh

Selector for the standard discrete ``L^2`` inner product weighted by cell measures.

The weights are the volumes (lengths in 1D, areas in 2D, volumes in 3D) of grid cells,
denoted ``|\\square_k|``. This is the most common inner product for finite difference
methods and corresponds to the trapezoid rule for integration on non-uniform grids.

# Mathematical form

For a 2D grid:
```math
(u_h, v_h)_h = \\sum_{i,j} |\\square_{i,j}| u_h(x_i, y_j) v_h(x_i, y_j)
```

where ``|\\square_{i,j}|`` is the area of the cell centered at ``(x_i, y_j)``.

# Examples

```julia
# Compute L² inner product
result = innerₕ(uₕ, vₕ)  # Uses Innerh() internally

# Compute L² norm
norm_value = normₕ(uₕ)  # Equivalent to sqrt(innerₕ(uₕ, uₕ))
```

See also: [`Innerplus`](@ref), [`innerₕ`](@ref), [`normₕ`](@ref)
"""
struct Innerh end

#=
The following functions define the **mandatory interface** for any concrete subtype
of `AbstractSpaceType`. Any struct that subtypes `AbstractSpaceType` must implement
these methods to be considered a valid function space in this framework. The stubs below
without a method (`mesh`, `mesh_type`, `backend`, `dim`, `ndofs`) are the mandatory
interface: a space that omits one fails with a `MethodError` naming it. The rest have
generic definitions.
=#

"""
    space(Wₕ::AbstractSpaceType) -> AbstractSpaceType

Returns the function space `Wₕ` itself.
"""
@inline space(Wₕ::AbstractSpaceType) = return Wₕ

"""
    mesh(Wₕ::AbstractSpaceType) -> AbstractMeshType

Returns the underlying mesh object associated with the function space `Wₕ`.
"""
function mesh end

"""
    points(Wₕ::AbstractSpaceType) -> Union{Vector, NTuple}

Forwards to [`points`](@ref)`(mesh(Wₕ))`: the coordinates of the mesh points underlying `Wₕ`.

```julia
x, y = points(Wₕ)
```

See also: [`half_points`](@ref).
"""
@inline points(Wₕ::AbstractSpaceType) = points(mesh(Wₕ))

"""
    half_points(Wₕ::AbstractSpaceType)

Forwards to [`half_points`](@ref)`(mesh(Wₕ))`: the precomputed cell boundaries (half-points) of
the mesh underlying `Wₕ`.
"""
@inline half_points(Wₕ::AbstractSpaceType) = half_points(mesh(Wₕ))

"""
    mesh_type(Wₕ::AbstractSpaceType) -> Type{<:AbstractMeshType}
    mesh_type(::Type{<:AbstractSpaceType}) -> Type{<:AbstractMeshType}

Returns the type of the mesh associated with the function space `Wₕ`. Also works if the argument is the type of the space.
"""
function mesh_type end

"""
    backend(Wₕ::AbstractSpaceType) -> AbstractBackend

Returns the computational backend associated with the space `Wₕ`.
"""
function backend end

"""
    dim(Wₕ::AbstractSpaceType) -> Int
    dim(::Type{<:AbstractSpaceType}) -> Int

Returns the spatial dimension of the mesh associated with the function space `Wₕ`.
"""
function dim end

"""
    ndofs(Wₕ::AbstractSpaceType) -> Int
    ndofs(Wₕ::AbstractSpaceType, ::Type{Tuple}) -> NTuple{N, Int}

Returns the total number of degrees of freedom (DOFs) in the function space `Wₕ`.

!!! warning "The `Tuple` form means something different for a composite space"
    On a [`ScalarGridSpace`](@ref), `ndofs(Wₕ, Tuple)` is the grid's shape, with one entry
    per **spatial dimension** (`Nₓ`, `Nᵧ`, ...). On a [`CompositeGridSpace`](@ref), it is
    instead one entry per **top-level sub-space**, each that sub-space's total DOF count,
    unrelated to spatial dimension, and not a shape a `prod` should be taken over. A nested
    sub-space's entry counts all its leaves; `leaf_count` counts the scalar leaves. Code
    that does not know in advance which kind of space it was given should reach for one
    of the two unambiguous forms instead: `npoints(mesh(Wₕ), Tuple)` for the grid shape,
    or `map(ndofs, spaces(Wₕ))` for the per-component counts. Mixing them up is not
    hypothetical, since `_grid_dims` (`src/operators/stencil.jl`) reads the
    grid shape from the mesh, not from `ndofs(Wₕ, Tuple)`, for exactly this reason.
"""
function ndofs end

"""
    ncomponents(Wₕ::AbstractSpaceType) -> Int
    ncomponents(::Type{<:AbstractSpaceType}) -> Int

Returns the number of field components of the function space (e.g. 1 for scalar, D for vector).

On a [`CompositeGridSpace`](@ref) it counts the top-level sub-spaces: a nested composite
counts its direct children. `leaf_count` counts the scalar leaves, and `components(uₕ)`
lists them.
"""
@inline ncomponents(::AbstractSpaceType{N}) where {N} = N
@inline ncomponents(::Type{<:AbstractSpaceType{N}}) where {N} = N

"""
    spaces(Wₕ::AbstractSpaceType) -> Tuple

Returns the constituent subspace(s) of `Wₕ` as a tuple.
"""
@inline spaces(Wₕ::AbstractSpaceType) = (Wₕ,)
