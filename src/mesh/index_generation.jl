"""
# index_generation.jl

This file provides utilities for generating CartesianIndices from point counts
and working with index ranges in meshes.

## Key Functions

- `generate_indices`: Create CartesianIndices from point counts
- `is_boundary_index`: Check if an index is on the boundary
- `boundary_indices`: Get all boundary facet indices
- `interior_indices`: Get indices of interior points

## Performance

Uses compile-time dispatch via dimension traits for optimal performance
in both 1D and multi-dimensional cases.
"""

"""
	generate_indices([::Dimension], pts)

Returns the `CartesianIndices` of a mesh with `pts[i]` points in each direction.

For scalar input (Int), returns a 1D `CartesianIndices`. For tuple/vector input,
returns multi-dimensional `CartesianIndices`.

# Arguments

  - `pts`: Either an `Int` (1D) or `NTuple{D,Int}`/`SVector{D,Int}` (nD)
  - Optional first argument: Dimension trait for explicit dispatch

# Returns

`CartesianIndices` appropriate for the dimensionality

# Examples

```julia
generate_indices(10)           # CartesianIndices((1:10,))
generate_indices((5, 7))       # CartesianIndices((1:5, 1:7))
generate_indices((3, 4, 5))    # CartesianIndices((1:3, 1:4, 1:5))
```

# Implementation Notes

Dispatches through `dimension_one_or_all` trait to select specialized
implementations for 1D vs. nD cases at compile time.
"""
@inline generate_indices(pts::PointsType) where PointsType = generate_indices(dimension_one_or_all(PointsType), pts)

"""
	generate_indices(::OneDimensional, pts::Int)

Generate 1D CartesianIndices from a point count.
"""
@inline generate_indices(::OneDimensional, pts) = CartesianIndices((pts,))

"""
	generate_indices(::MultiDimensional, pts::NTuple{D})
	generate_indices(::MultiDimensional, pts::SVector{D})

Generate multi-dimensional CartesianIndices from a tuple/vector of point counts.
"""
@inline generate_indices(::MultiDimensional, pts::NTuple{D}) where D = CartesianIndices(ntuple(i -> 1:pts[i], Val(D)))
@inline generate_indices(::MultiDimensional, pts::SVector{D}) where D = CartesianIndices(ntuple(i -> 1:pts[i], Val(D)))

"""
	is_boundary_index(idxs::CartesianIndices, idx)

Checks if a given index `idx` lies on the boundary of a `CartesianIndices` domain.

An index is on the boundary if any of its coordinates matches the first or last
element of the corresponding axis range (for axes with length > 1).

# Arguments

  - `idxs::CartesianIndices{D}`: The index domain
  - `idx`: The index to check (CartesianIndex or tuple)

# Returns

`true` if the index is on the boundary, `false` otherwise

# Examples

```jldoctest
julia> domain = CartesianIndices((3, 4));
	   is_boundary_index(domain, (1, 2))
true

julia> is_boundary_index(domain, (2, 2))
false
```
"""
function is_boundary_index(idxs::CartesianIndices{D}, idx) where D
	_idx = CartesianIndex(idx)
	@inbounds for i in 1:D
		axis = idxs.indices[i]
		# A point is on the boundary if its coordinate along any axis of length > 1 
		# matches the first or last element of that axis range.
		if length(axis) > 1 && (_idx[i] == first(axis) || _idx[i] == last(axis))
			return true
		end
	end
	return false
end

"""
	boundary_indices(idxs::CartesianIndices) 

Returns all boundary facets of a `CartesianIndices` domain as a tuple of `CartesianIndices`.

Each element of the returned tuple represents a distinct boundary section, such as
a face or edge of the domain.

# Arguments

  - `idxs::CartesianIndices`: The index domain

# Returns

Tuple of `CartesianIndices`, one for each boundary facet

# Examples

```jldoctest
julia> domain = CartesianIndices((2, 2));
	   boundary_indices(domain)
(CartesianIndices((1:1, 1:2)), CartesianIndices((2:2, 1:2)), 
 CartesianIndices((1:2, 1:1)), CartesianIndices((1:2, 2:2)))
```
"""
@inline function boundary_indices(idxs::CartesianIndices)
	tup = boundary_symbol_to_cartesian(idxs)
	return ntuple(i -> tup[i], length(tup))
end

"""
	interior_indices(indices::CartesianIndices)

Computes the `CartesianIndices` representing the interior of a given domain, excluding
all boundary points.

This is achieved by shrinking the index range in each dimension by one from both ends.
Dimensions with a length of one or less are returned unchanged (collapsed dimensions).

# Arguments

  - `indices::CartesianIndices{D}`: The full index domain

# Returns

`CartesianIndices{D}` representing only the interior points

# Examples

```jldoctest
julia> domain = CartesianIndices((3, 3));
	   interior_indices(domain)
CartesianIndices((2:2, 2:2))
```

```jldoctest
julia> domain_2d_line = CartesianIndices((1, 5));
	   interior_indices(domain_2d_line)
CartesianIndices((1:1, 2:4))
```
"""
@inline function interior_indices(indices::CartesianIndices{D}) where D
	original_ranges = indices.indices

	interior_ranges_tuple = ntuple(Val(D)) do i
		@inbounds r = original_ranges[i]

		if length(r) <= 1
			return r
		else
			(first(r) + 1):(last(r) - 1)
		end
	end

	return CartesianIndices(interior_ranges_tuple)
end
