"""
# mesh_common_methods.jl

This file provides common method implementations shared between Mesh1D and MeshnD
to reduce code duplication and improve maintainability (DRY principle).

## Included Methods

### Common Type Methods
- `eltype` for both value and type dispatch

### Common Validation
- `_check_point_bounds` - Unified bounds checking for point access
- `_check_half_point_bounds` - Bounds checking for half-points

### Common Collapsed Mesh Handling
- `_handle_collapsed_spacing` - Returns zero for degenerate meshes

## Design Pattern

Instead of duplicating the same logic in mesh1d.jl and meshnd.jl, these methods
provide shared implementations that both concrete types can use, either directly
or through composition.

See also: [`AbstractMeshType`](@ref), [`Mesh1D`](@ref), [`MeshnD`](@ref)
"""

#------------------------------------------------------------------------------------------#
# Common Type Methods
#------------------------------------------------------------------------------------------#

"""
	eltype(::AbstractMeshType{D, BT}) where {D, BT}
	eltype(::Type{<:AbstractMeshType{D, BT}}) where {D, BT}

Returns the element type of a mesh based on its backend type.

This is a common implementation that works for both Mesh1D and MeshnD since
they both parameterize on a backend type.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0)), 100, true)
T = eltype(Ωₕ)  # Float64
```
"""
# Note: These are defined in the concrete mesh files since the type parameter 
# structure differs (Mesh1D{BT,...} vs MeshnD{D,BT,...})
# But the implementation logic is identical:
# @inline eltype(::MeshType{...,BT,...}) where BT = eltype(BT)

#------------------------------------------------------------------------------------------#
# Common Validation Helpers
#------------------------------------------------------------------------------------------#

"""
	_check_point_bounds(Ωₕ::AbstractMeshType, idx, location="point")

Validates that index `idx` is within bounds for accessing points in mesh `Ωₕ`.

Throws an `AssertionError` if the index is out of bounds. Handles both integer
indices and CartesianIndex.

# Arguments

  - `Ωₕ`: The mesh
  - `idx`: Index to validate (Int or CartesianIndex)
  - `location`: Description of what's being accessed (for error message)

# Examples

```julia
_check_point_bounds(Ωₕ, 5, "grid point")
_check_point_bounds(Ωₕ, CartesianIndex(10, 15), "point")
```
"""
@inline function _check_point_bounds(Ωₕ::AbstractMeshType, idx::Int, location::String = "point")
	n = npoints(Ωₕ)
	@assert 1 <= idx <= n "Index $idx out of bounds for $location access in mesh with $n points."
	return nothing
end

@inline function _check_point_bounds(Ωₕ::AbstractMeshType, idx::CartesianIndex{1}, location::String = "point")
	_check_point_bounds(Ωₕ, idx[1], location)
end

@inline function _check_point_bounds(Ωₕ::AbstractMeshType{D}, idx::CartesianIndex{D}, location::String = "point") where D
	npts = npoints(Ωₕ, Tuple)
	for i in 1:D
		@assert 1 <= idx[i] <= npts[i] "Index $idx[$i] out of bounds for $location access in dimension $i (max $(npts[i]))."
	end
	return nothing
end

"""
	_check_half_point_bounds(Ωₕ::AbstractMeshType, idx)

Validates that index `idx` is within bounds for accessing half-points (cell centers).

Half-points exist at indices 1 through npoints+1 (for cell edges/centers).
"""
@inline function _check_half_point_bounds(Ωₕ::AbstractMeshType, idx::Int)
	n = npoints(Ωₕ)
	@assert 1 <= idx <= n + 1 "Index $idx out of bounds for half-point access (valid range: 1 to $(n+1))."
	return nothing
end

#------------------------------------------------------------------------------------------#
# Common Collapsed Mesh Handling
#------------------------------------------------------------------------------------------#

"""
	_handle_collapsed_spacing(Ωₕ::AbstractMeshType, default_value)

Returns zero if the mesh is collapsed (degenerate), otherwise returns `default_value`.

This is a common pattern in spacing calculations where collapsed dimensions
should contribute zero spacing.

# Arguments

  - `Ωₕ`: The mesh to check
  - `default_value`: The value to return if mesh is not collapsed

# Examples

```julia
# In spacing calculation:
if is_collapsed(Ωₕ)
	return zero(eltype(Ωₕ))
else
	return pts[i] - pts[i-1]
end

# Can be replaced with:
return _handle_collapsed_spacing(Ωₕ, pts[i] - pts[i-1])
```
"""
@inline function _handle_collapsed_spacing(Ωₕ::AbstractMeshType, default_value)
	return is_collapsed(Ωₕ) ? zero(eltype(Ωₕ)) : default_value
end

#------------------------------------------------------------------------------------------#
# Common Helper for CartesianIndex Conversion
#------------------------------------------------------------------------------------------#

"""
	_extract_linear_index(idx::Union{Int, CartesianIndex{1}})

Extracts the linear index from either an Int or a 1D CartesianIndex.

This provides a unified interface for functions that accept both Int and
CartesianIndex{1} for 1D meshes.

# Examples

```julia
_extract_linear_index(5)                  # Returns 5
_extract_linear_index(CartesianIndex(5))  # Returns 5
```
"""
@inline _extract_linear_index(idx::Int) = idx
@inline _extract_linear_index(idx::CartesianIndex{1}) = idx[1]

#------------------------------------------------------------------------------------------#
# Common Iterator Construction Patterns
#------------------------------------------------------------------------------------------#

"""
	_spacing_generator(Ωₕ::AbstractMeshType, spacing_func)

Creates a generator for spacing values using the provided spacing function.

This encapsulates the common pattern of creating iterators over spacing values.

# Arguments

  - `Ωₕ`: The mesh
  - `spacing_func`: Function that computes spacing at index `i`

# Examples

```julia
# Instead of:
spacings_iterator(Ωₕ) = (spacing(Ωₕ, i) for i in eachindex(points(Ωₕ)))

# Can use:
spacings_iterator(Ωₕ) = _spacing_generator(Ωₕ, spacing)
```
"""
@inline _spacing_generator(Ωₕ::AbstractMeshType, spacing_func) = (spacing_func(Ωₕ, i) for i in 1:npoints(Ωₕ))

#------------------------------------------------------------------------------------------#
# Documentation Utilities
#------------------------------------------------------------------------------------------#

"""
	_bounds_check_error_message(idx, n, mesh_type="mesh")

Generates a standardized error message for bounds checking failures.

This ensures consistent error messages across the codebase.

# Returns

A formatted string describing the bounds violation.
"""
function _bounds_check_error_message(idx, n, mesh_type::String = "mesh")
	return "Index $idx out of bounds for $mesh_type with $n points."
end

#------------------------------------------------------------------------------------------#
# Common Logic for Spacing Calculations (Reference Implementation)
#------------------------------------------------------------------------------------------#

"""
	_compute_backward_spacing_1d(pts::AbstractVector, i::Int, collapsed::Bool, T::Type)

Reference implementation for computing backward spacing in 1D meshes.

This encapsulates the common logic for spacing calculations that handle:

  - Collapsed (degenerate) meshes
  - Boundary conditions (first point uses forward spacing)
  - Standard backward difference

# Arguments

  - `pts`: Vector of point coordinates
  - `i`: Index of the point
  - `collapsed`: Whether the mesh is degenerate
  - `T`: Element type for zero value

# Returns

The spacing value (distance to previous point, or zero if collapsed)

# Algorithm

```
if collapsed:
	return 0
else if i == 1:
	return pts[2] - pts[1]  # Forward spacing at boundary
else:
	return pts[i] - pts[i-1]  # Standard backward spacing
```
"""
@inline function _compute_backward_spacing_1d(pts::AbstractVector, i::Int, collapsed::Bool, T::Type)
	if collapsed
		return zero(T)
	elseif i == 1
		# Boundary case: use forward spacing
		return pts[2] - pts[1]
	else
		# Standard backward spacing
		return pts[i] - pts[i-1]
	end
end

"""
	_compute_forward_spacing_1d(pts::AbstractVector, i::Int, N::Int, collapsed::Bool, T::Type)

Reference implementation for computing forward spacing in 1D meshes.

Similar to backward spacing but looks ahead instead of behind.

# Algorithm

```
if collapsed:
	return 0
else if i == N:
	return pts[N] - pts[N-1]  # Backward spacing at boundary
else:
	return pts[i+1] - pts[i]  # Standard forward spacing
```
"""
@inline function _compute_forward_spacing_1d(pts::AbstractVector, i::Int, N::Int, collapsed::Bool, T::Type)
	if collapsed
		return zero(T)
	elseif i == N
		# Boundary case: use backward spacing
		return pts[N] - pts[N-1]
	else
		# Standard forward spacing
		return pts[i+1] - pts[i]
	end
end

#------------------------------------------------------------------------------------------#
# Common Pattern: Apply Logic to Tuple Elements
#------------------------------------------------------------------------------------------#

"""
	_apply_hs_logic(value::T) where T

Helper for half-spacing logic: replace zero with one.

Used in MeshnD to handle collapsed dimensions in cell measure calculations.
This is extracted here as a common utility.

# Examples

```julia
_apply_hs_logic(0.0)  # Returns 1.0
_apply_hs_logic(0.5)  # Returns 0.5
```
"""
@inline _apply_hs_logic(value::T) where T = ifelse(iszero(value), one(T), value)

#------------------------------------------------------------------------------------------#
# Common Mesh Property Queries
#------------------------------------------------------------------------------------------#

"""
	is_uniform(Ωₕ::AbstractMeshType; tol=1e-10)

Checks if the mesh has uniform spacing (within tolerance).

This is a common query that can be useful for optimizations or validation.

# Arguments

  - `Ωₕ`: The mesh to check
  - `tol`: Tolerance for floating point comparison

# Returns

`true` if all spacings are approximately equal, `false` otherwise
"""
function is_uniform(Ωₕ::AbstractMeshType{1}; tol = 1e-10)
	if npoints(Ωₕ) <= 1
		return true
	end

	spacings = collect(spacings_iterator(Ωₕ))
	h_ref = first(spacings)

	return all(h -> abs(h - h_ref) < tol, spacings)
end

function is_uniform(Ωₕ::AbstractMeshType{D}; tol = 1e-10) where D
	# For nD meshes, check each dimension independently
	return all(i -> is_uniform(Ωₕ(i); tol = tol), 1:D)
end
