"""
# interface.jl

This file defines the abstract interface and shared foundational methods for all mesh types in Bramble.

## Key Components

- `AbstractMeshType{D}`: Abstract supertype parameterized by spatial dimension ``D``.
- **Cartesian Index Operations**: `generate_indices`, `boundary_indices`, `interior_indices`, `is_boundary_index`.
- **Interface Declarations & Fallbacks**: Field getters, bounds checking, iterators, and spacing calculations.
- **Mesh Construction**: Top-level `mesh(Ω, npts, ...)` dispatch.

See also: [`Mesh1D`](@ref), [`MeshnD`](@ref), [`Domain`](@ref)
"""

#------------------------------------------------------------------------------------------#
# Abstract Supertype
#------------------------------------------------------------------------------------------#

"""
	AbstractMeshType{D}

Abstract supertype for all mesh types in Bramble. The type parameter `D` represents
the spatial dimension of the mesh (1, 2, or 3).

All concrete mesh types must implement the AbstractMeshType interface, including:

  - `eltype`, `dim`, `topo_dim`, `indices`, `backend`, `markers`
  - `points`, `point`, `half_points`, `half_point`
  - `spacing`, `half_spacing`, `forward_spacing`

# Type Parameter

  - `D`: Spatial dimension (1, 2, or 3)

# Related Types

  - Meshes are created from a [Domain](@ref) using the [`mesh`](@ref) function.
  - See [MeshMarkers](@ref) for marker management on meshes.

See also: [`Mesh1D`](@ref), [`MeshnD`](@ref), [`Domain`](@ref)
"""
abstract type AbstractMeshType{D} end

#------------------------------------------------------------------------------------------#
# Dimension Traits (Backwards-Compatibility & Explicit Dispatch)
#------------------------------------------------------------------------------------------#

"""
	Dimension

Abstract base type for dimension traits used in compile-time dispatch.
"""
abstract type Dimension end

struct OneDimensional <: Dimension end
struct TwoDimensional <: Dimension end
struct ThreeDimensional <: Dimension end
struct MultiDimensional <: Dimension end

@inline dimension_one_or_all(::Type{<:Int}) = OneDimensional()
@inline dimension_one_or_all(::Type{<:Union{NTuple, SVector}}) = MultiDimensional()

#------------------------------------------------------------------------------------------#
# Cartesian Index Generation & Boundary Queries
#------------------------------------------------------------------------------------------#

"""
	generate_indices([::Dimension], pts)

Returns the `CartesianIndices` of a mesh with `pts[i]` points in each direction.

For scalar input (`Int`), returns 1D `CartesianIndices`. For tuple/vector input,
returns multi-dimensional `CartesianIndices`.
"""
@inline generate_indices(pts::Int) = CartesianIndices((pts,))
@inline generate_indices(pts::NTuple{D, Int}) where D = CartesianIndices(pts)
@inline generate_indices(pts::SVector{D, Int}) where D = CartesianIndices(Tuple(pts))
@inline generate_indices(::OneDimensional, pts::Int) = CartesianIndices((pts,))
@inline generate_indices(::MultiDimensional, pts::NTuple{D}) where D = CartesianIndices(ntuple(i -> 1:pts[i], Val(D)))
@inline generate_indices(::MultiDimensional, pts::SVector{D}) where D = CartesianIndices(ntuple(i -> 1:pts[i], Val(D)))
@inline generate_indices(pts::PointsType) where PointsType = generate_indices(dimension_one_or_all(PointsType), pts)

"""
	is_boundary_index(idxs::CartesianIndices, idx)

Checks if a given index `idx` lies on the boundary of a `CartesianIndices` domain.
"""
function is_boundary_index(idxs::CartesianIndices{D}, idx) where D
	_idx = CartesianIndex(idx)
	@inbounds for i in 1:D
		axis = idxs.indices[i]
		if length(axis) > 1 && (_idx[i] == first(axis) || _idx[i] == last(axis))
			return true
		end
	end
	return false
end

"""
	boundary_indices(idxs::CartesianIndices)

Returns all boundary facets of a `CartesianIndices` domain as a tuple of `CartesianIndices`.
"""
@inline function boundary_indices(idxs::CartesianIndices)
	tup = boundary_symbol_to_cartesian(idxs)
	return ntuple(i -> tup[i], length(tup))
end

"""
	interior_indices(indices::CartesianIndices)

Computes the `CartesianIndices` representing the interior of a given domain, excluding
all boundary points. Dimensions with a length of one or less remain unchanged.
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

#------------------------------------------------------------------------------------------#
# Field Getters & Index Delegation
#------------------------------------------------------------------------------------------#

"""
	set(Ωₕ::AbstractMeshType)

Returns the geometric set of the domain over which the mesh `Ωₕ` is defined.
"""
@inline set(Ωₕ::AbstractMeshType) = Ωₕ.set

"""
	indices(Ωₕ::AbstractMeshType)

Returns the `CartesianIndices` associated with the points of mesh `Ωₕ`.
"""
@inline indices(Ωₕ::AbstractMeshType) = Ωₕ.indices

"""
	backend(Ωₕ::AbstractMeshType)

Returns the linear algebra [Backend](@ref) associated with the mesh `Ωₕ`.
"""
@inline backend(Ωₕ::AbstractMeshType) = Ωₕ.backend

"""
	markers(Ωₕ::AbstractMeshType)

Returns the `MeshMarkers` dictionary associated with the mesh `Ωₕ`.
"""
@inline markers(Ωₕ::AbstractMeshType) = Ωₕ.markers

"""
	index_in_marker(Ωₕ::AbstractMeshType, label::Symbol)

Returns the `BitVector` associated with the marker `label` in mesh `Ωₕ`.
"""
@inline index_in_marker(Ωₕ::AbstractMeshType, label::Symbol) = markers(Ωₕ)[label]

"""
	set_indices!(Ωₕ::AbstractMeshType, indices)

Overrides the indices in `Ωₕ`. Used internally during mesh refinement.
"""
@inline set_indices!(Ωₕ::AbstractMeshType, indices) = (Ωₕ.indices = indices; return)

@inline is_boundary_index(Ωₕ::AbstractMeshType, idx) = is_boundary_index(indices(Ωₕ), idx)
@inline boundary_indices(Ωₕ::AbstractMeshType) = boundary_indices(indices(Ωₕ))
@inline interior_indices(Ωₕ::AbstractMeshType) = interior_indices(indices(Ωₕ))

#------------------------------------------------------------------------------------------#
# Bounds Checking & Internal Helpers
#------------------------------------------------------------------------------------------#

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

@inline function _check_half_point_bounds(Ωₕ::AbstractMeshType, idx::Int)
	n = npoints(Ωₕ)
	@assert 1 <= idx <= n + 1 "Index $idx out of bounds for half-point access (valid range: 1 to $(n+1))."
	return nothing
end

@inline _handle_collapsed_spacing(Ωₕ::AbstractMeshType, default_value) = is_collapsed(Ωₕ) ? zero(eltype(Ωₕ)) : default_value
@inline _extract_linear_index(idx::Int) = idx
@inline _extract_linear_index(idx::CartesianIndex{1}) = idx[1]
@inline _spacing_generator(Ωₕ::AbstractMeshType, spacing_func) = (spacing_func(Ωₕ, i) for i in 1:npoints(Ωₕ))
@inline _apply_hs_logic(value::T) where T = ifelse(iszero(value), one(T), value)

function _bounds_check_error_message(idx, n, mesh_type::String = "mesh")
	return "Index $idx out of bounds for $mesh_type with $n points."
end

# 1D spacing reference routines
@inline function _compute_backward_spacing_1d(pts::AbstractVector, i::Int, collapsed::Bool, T::Type)
	if collapsed
		return zero(T)
	elseif i == 1
		return pts[2] - pts[1]
	else
		return pts[i] - pts[i-1]
	end
end

@inline function _compute_forward_spacing_1d(pts::AbstractVector, i::Int, N::Int, collapsed::Bool, T::Type)
	if collapsed
		return zero(T)
	elseif i == N
		return pts[N] - pts[N-1]
	else
		return pts[i+1] - pts[i]
	end
end

#------------------------------------------------------------------------------------------#
# High-Level Mesh Constructor Dispatch
#------------------------------------------------------------------------------------------#

"""
	$(SIGNATURES)

Returns a [Mesh1D](@ref) or a [MeshnD](@ref) (``D=2,3``) defined on the [Domain](@ref) `Ω`.

The number of points for each coordinate direction is given in `npts`.
The distribution of points on the submeshes is given by `unif` (or keyword `uniform`, default `true`).

# Examples

```julia
I = interval(0.0, 1.0)
Ωₕ = mesh(domain(I), 10)                      # uniform by default
Ωₕ_nonunif = mesh(domain(I), 10, false)       # explicit non-uniform

X = domain(interval(0, 1) × interval(4, 5))
Ωₕ_2d = mesh(X, (10, 15))                     # uniform by default
Ωₕ_mixed = mesh(X, (10, 15), (true, false))
```
"""
@inline mesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}; backend = backend()) where D = _mesh(Ω, npts, unif, backend)
@inline mesh(Ω::Domain{CartesianProduct{1,T}}, npts::Int, unif::Bool; backend = backend()) where T = _mesh(Ω, (npts,), (unif,), backend)
@inline mesh(Ω::Domain{CartesianProduct{1,T}}, npts::Int; uniform::Bool = true, backend = backend()) where T = _mesh(Ω, (npts,), (uniform,), backend)
@inline mesh(Ω::Domain, npts::NTuple{D,Int}; uniform::NTuple{D,Bool} = ntuple(_ -> true, Val(D)), backend = backend()) where D = _mesh(Ω, npts, uniform, backend)

#------------------------------------------------------------------------------------------#
# Required Interface Methods
#------------------------------------------------------------------------------------------#

@inline dim(::AbstractMeshType{D}) where D = D
@inline dim(::Type{<:AbstractMeshType{D}}) where D = D

@inline function topo_dim(Ωₕ::AbstractMeshType{D}) where D
	count = 0
	@inbounds for i in 1:D
		npoints(Ωₕ(i)) > 1 && (count += 1)
	end
	return count
end

function eltype(Ωₕ::AbstractMeshType)
	error("Interface function 'eltype' not implemented for mesh of type $(typeof(Ωₕ)).")
end

function eltype(::Type{<:AbstractMeshType})
	error("Interface function 'eltype(::Type{...})' not implemented for mesh type.")
end

@inline points(Ωₕ::AbstractMeshType) = error("Interface function 'points' not implemented for mesh of type $(typeof(Ωₕ)).")
@inline point(Ωₕ::AbstractMeshType, idx) = error("Interface function 'point' not implemented for mesh of type $(typeof(Ωₕ)).")
@inline points_iterator(Ωₕ::AbstractMeshType) = error("Interface function 'points_iterator' not implemented for mesh of type $(typeof(Ωₕ)).")

@inline half_points(Ωₕ::AbstractMeshType) = error("Interface function 'half_points' not implemented for mesh of type $(typeof(Ωₕ)).")
@inline half_point(Ωₕ::AbstractMeshType, idx) = error("Interface function 'half_point' not implemented for mesh of type $(typeof(Ωₕ)).")
@inline half_points_iterator(Ωₕ::AbstractMeshType) = error("Interface function 'half_points_iterator' not implemented for mesh of type $(typeof(Ωₕ)).")

@inline spacing(Ωₕ::AbstractMeshType, idx) = error("Interface function 'spacing' not implemented for mesh of type $(typeof(Ωₕ)).")
@inline spacings_iterator(Ωₕ::AbstractMeshType) = error("Interface function 'spacings_iterator' not implemented for mesh of type $(typeof(Ωₕ)).")

@inline forward_spacing(Ωₕ::AbstractMeshType, idx) = error("Interface function 'forward_spacing' not implemented for mesh of type $(typeof(Ωₕ)).")
@inline forward_spacings_iterator(Ωₕ::AbstractMeshType) = error("Interface function 'forward_spacings_iterator' not implemented for mesh of type $(typeof(Ωₕ)).")

@inline half_spacings(Ωₕ::AbstractMeshType) = error("Interface function 'half_spacings' not implemented for mesh of type $(typeof(Ωₕ)).")
@inline half_spacing(Ωₕ::AbstractMeshType, idx) = error("Interface function 'half_spacing' not implemented for mesh of type $(typeof(Ωₕ)).")
@inline half_spacings_iterator(Ωₕ::AbstractMeshType) = error("Interface function 'half_spacings_iterator' not implemented for mesh of type $(typeof(Ωₕ)).")

@inline npoints(Ωₕ::AbstractMeshType) = error("Interface function 'npoints' not implemented for mesh of type $(typeof(Ωₕ)).")
@inline npoints(Ωₕ::AbstractMeshType, ::Type{Tuple}) = error("Interface function 'npoints' not implemented for mesh of type $(typeof(Ωₕ)).")

@inline hₘₐₓ(Ωₕ::AbstractMeshType) = error("Interface function 'hₘₐₓ' not implemented for mesh of type $(typeof(Ωₕ)).")
@inline cell_measure(Ωₕ::AbstractMeshType, idx) = error("Interface function 'cell_measure' not implemented for mesh of type $(typeof(Ωₕ)).")
@inline cell_measures_iterator(Ωₕ::AbstractMeshType) = error("Interface function 'cell_measures_iterator' not implemented for mesh of type $(typeof(Ωₕ)).")

@inline iterative_refinement!(Ωₕ::AbstractMeshType) = error("Interface function 'iterative_refinement!' not implemented for mesh of type $(typeof(Ωₕ)).")
@inline iterative_refinement!(Ωₕ::AbstractMeshType, domain_markers::DomainMarkers) = error("Interface function 'iterative_refinement!' not implemented for mesh of type $(typeof(Ωₕ)).")

@inline change_points!(Ωₕ::AbstractMeshType, pts) = error("Interface function 'change_points!' not implemented for mesh of type $(typeof(Ωₕ)).")
@inline change_points!(Ωₕ::AbstractMeshType, ::DomainMarkers, pts) = error("Interface function 'change_points!' not implemented for mesh of type $(typeof(Ωₕ)).")

#------------------------------------------------------------------------------------------#
# Uniformity Query
#------------------------------------------------------------------------------------------#

"""
	is_uniform(Ωₕ::AbstractMeshType; tol=1e-10)

Checks if the mesh has uniform spacing (within numerical tolerance).
"""
function is_uniform(Ωₕ::AbstractMeshType{1}; tol = 1e-10)
	n = npoints(Ωₕ)
	if n <= 1
		return true
	end

	h_ref = spacing(Ωₕ, 1)
	@inbounds for i in 2:n
		if abs(spacing(Ωₕ, i) - h_ref) >= tol
			return false
		end
	end
	return true
end

function is_uniform(Ωₕ::AbstractMeshType{D}; tol = 1e-10) where D
	return all(i -> is_uniform(Ωₕ(i); tol = tol), 1:D)
end
