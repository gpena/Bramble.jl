"""
	 AbstractMeshType{D}

Abstract type for meshes. Meshes are only parametrized by their dimension `D`.
"""
abstract type AbstractMeshType{D} <: BrambleType end

"""
	Dimension

Abstract type representing spatial dimensions. Serves as a base type for implementing specific dimension types.
"""
abstract type Dimension end

"""
	struct OneDimensional end

A struct representing a one-dimensional mesh. It is a subtype of `Dimension`.
"""
struct OneDimensional <: Dimension end

"""
	struct NDimensional end

A struct representing a multi-dimensional mesh. It is a subtype of `Dimension`.
"""
struct NDimensional <: Dimension end

dimension_one_or_all(::Type{<:Int}) = OneDimensional()
dimension_one_or_all(::Type{<:NTuple{D}}) where D = NDimensional()

dimension(::Type{<:NTuple{1}}) = OneDimensional()
dimension(::Type{<:NTuple{2}}) = TwoDimensional()
dimension(::Type{<:NTuple{3}}) = ThreeDimensional()

"""
	generate_indices([::Dimension], nPoints)

Returns the `CartesianIndices` of a mesh with `nPoints[i]` in each direction or just `nPoints`, if the argument is an `Int`.
"""
@inline generate_indices(npts::PointsType) where PointsType = generate_indices(dimension_one_or_all(PointsType), npts)
@inline generate_indices(::OneDimensional, npts) = CartesianIndices((npts,))
@inline generate_indices(::NDimensional, npts) = CartesianIndices(ntuple(i -> 1:npts[i], length(npts)))

"""
	set(Ωₕ::AbstractMeshType)

Returns the set of the domain over which the mesh `Ωₕ` is defined.
"""
@inline set(Ωₕ::AbstractMeshType) = Ωₕ.set

"""
	is_boundary_index(idx, Ωₕ::AbstractMeshType)

Returns true if the index `idx` is a boundary index of the mesh.
"""
@inline is_boundary_index(idx, Ωₕ::AbstractMeshType) = is_boundary_index(idx, indices(Ωₕ))

"""
	boundary_indices(Ωₕ::AbstractMeshType)

Returns the boundary indices of mesh `Ωₕ`.
"""
@inline boundary_indices(Ωₕ::AbstractMeshType) = boundary_indices(indices(Ωₕ))

"""
	interior_indices(Ωₕ::AbstractMeshType)

Returns the interior indices of mesh `Ωₕ`.
"""
@inline interior_indices(Ωₕ::AbstractMeshType) = interior_indices(indices(Ωₕ))

"""
	is_boundary_index(idx, idxs::CartesianIndices) -> Bool

Checks if a given index `idx` lies on the boundary of a `CartesianIndices` domain. This function determines if a point is part of any boundary facet (e.g., face, edge, or corner) of the specified domain.

# Example

```julia
is_boundary_index((1, 2), domain)
domain = CartesianIndices((3, 4))
is_boundary_index((2, 2), domain)
true
```
"""
function is_boundary_index(idx, idxs::CartesianIndices{D}) where D
	_idx = CartesianIndex(idx)
	for i in 1:D
		axis = idxs.indices[i]
		if length(axis) > 1 && (_idx[i] == first(axis) || _idx[i] == last(axis))
			return true
		end
	end
	return false
end

"""
	boundary_indices(idxs::CartesianIndices) 

Returns all boundary facets of a `CartesianIndices` domain as a tuple of `CartesianIndices`. Each element of the returned tuple represents a distinct boundary section, such as a face or edge of the domain.

# Example

```julia
domain = CartesianIndices((2, 2));
boundary_indices(domain)
(CartesianIndices((1:1, 1:2)), CartesianIndices((2:2, 1:2)), CartesianIndices((1:2, 1:1)), CartesianIndices((1:2, 2:2)))
```
"""
@inline function boundary_indices(idxs::CartesianIndices)
	tup = boundary_symbol_to_cartesian(idxs)

	return ntuple(i -> tup[i], length(tup))
end

"""
	interior_indices(indices::CartesianIndices)

Computes the `CartesianIndices` representing the interior of a given domain, excluding all boundary points. This is achieved by shrinking the index range in each dimension by one from both ends. Dimensions with a length of one or less are returned unchanged.

# Examples

```@jldoctest
domain = CartesianIndices((3, 3)); # A 3x3 grid
interior_indices(domain)
CartesianIndices((2:2, 2:2))
```

```julia
domain_2d_line = CartesianIndices((1, 5)); # A line in 2D space
interior_indices(domain_2d_line)
CartesianIndices((1:1, 2:4))
```
"""
@inline function interior_indices(indices::CartesianIndices{D}) where D
	original_ranges = indices.indices

	interior_ranges_tuple = ntuple(Val(D)) do i
		r = original_ranges[i]

		if length(r) <= 1
			return r
		else
			(first(r) + 1):(last(r) - 1)
		end
	end

	return CartesianIndices(interior_ranges_tuple)
end

"""
	eltype(Ωₕ::AbstractMeshType)
	eltype(::Type{<:AbstractMeshType})

Returns the element type of the points of `Ωₕ`.

This function is a required part of the `AbstractMeshType` interface. Any concrete subtype of `AbstractMeshType` must implement this method.
"""
function eltype(Ωₕ::AbstractMeshType)
	error("Interface function 'eltype' not implemented for mesh of type $(typeof(Ωₕ)).")
end

function eltype(::Type{<:AbstractMeshType})
	error("Interface function 'eltype(::Type{...})' not implemented for mesh type $(MeshType).")
end

"""
	dim(Ωₕ::AbstractMeshType)
	dim(::Type{<:AbstractMeshType})

Returns the dimension of the space where `Ωₕ` is embedded.

This function is a required part of the `AbstractMeshType` interface. Any concrete subtype of `AbstractMeshType` must implement this method.
"""
@inline dim(_::AbstractMeshType{D}) where D = D
@inline dim(::Type{<:AbstractMeshType{D}}) where D = D

"""
	topo_dim(Ωₕ::AbstractMeshType)

Returns the topological dimension `Ωₕ`.
"""
@inline function topo_dim(Ωₕ::AbstractMeshType{D}) where D
	terms = ntuple(i -> ifelse(npoints(Ωₕ(i)) == 1, false, true), Val(D))
	return sum(terms)
end

"""
	indices(Ωₕ::AbstractMeshType)

Returns the `CartesianIndices` associated with the points of mesh `Ωₕ`. This function is a required part of the `AbstractMeshType` interface. Any concrete subtype of `AbstractMeshType` must implement this method and have a field called `indices` of type `CartesianIndices`.
"""
@inline indices(Ωₕ::AbstractMeshType) = Ωₕ.indices

"""
	backend(Ωₕ::AbstractMeshType)

Returns the linear algebra [Backend](@ref) associated with the mesh `Ωₕ`. This function is a required part of the `AbstractMeshType` interface. Any concrete subtype of `AbstractMeshType` must implement this method and have a field called `backend` of type [Backend](@ref).
"""
@inline backend(Ωₕ::AbstractMeshType) = Ωₕ.backend

"""
	index_in_marker(Ωₕ::AbstractMeshType, label::Symbol)

Returns the `BitVector` associated with the marker with `label` of mesh `Ωₕ`.
"""
@inline index_in_marker(Ωₕ::AbstractMeshType, label::Symbol) = markers(Ωₕ)[label]

"""
	markers(Ωₕ::AbstractMeshType)

Returns the [DomainMarkers](@ref)) associated with the mesh `Ωₕ`. This function is a required part of the `AbstractMeshType` interface. Any concrete subtype of `AbstractMeshType` must implement this method and have a field called `markers` of type [DomainMarkers](@ref).
"""
@inline markers(Ωₕ::AbstractMeshType) = Ωₕ.markers

"""
	set_indices!(Ωₕ::AbstractMeshType, indices)

Overrides the indices in Ωₕ.
"""
@inline set_indices!(Ωₕ::AbstractMeshType, indices) = (Ωₕ.indices = indices; return)

"""
	mesh(Ω::Domain, npts::Int, unif::Bool)
	mesh(Ω::Domain, npts::NTuple{D}, unif::NTuple{D})

Returns a [Mesh1D](@ref) or a [MeshnD](@ref) (``D=2,3``) defined on the [Domain](@ref) `Ω`. The number of points for each coordinate projection mesh are given in the tuple `npts`. The distribution of points on the submeshes are encoded in the tuple `unif`.

For future reference, the mesh points are denoted as

  - 1D mesh, with `npts` = ``N_x``

```math
x_i, \\, i=1,\\dots,N.
```

  - 2D mesh, with `npts` = (``N_x``, ``N_y``)

```math
(x_i,y_j), \\, i=1,\\dots,N_x, \\, j=1,\\dots,N_y
```

  - 3D mesh, with `npts` = (``N_x``, ``N_y``, ``N_z``)

```math
(x_i,y_j,z_l), \\, i=1,\\dots,N_x, \\, j=1,\\dots,N_y, \\, l=1,\\dots,N_z.
```

# Examples

```julia
I = interval(0, 1);
Ωₕ = mesh(domain(I), 10, true);
```

```julia
X = domain(interval(0, 1) × interval(4, 5));
Ωₕ = mesh(X, (10, 15), (true, false))
```
"""
@inline mesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}; backend = backend()) where D = _mesh(Ω, npts, unif, backend)
@inline mesh(Ω::Domain{CartesianProduct{1,T}}, npts::Int, unif::Bool; backend = backend()) where T = _mesh(Ω, (npts,), (unif,), backend)

"""
	points(Ωₕ::AbstractMeshType)

Returns the points of `Ωₕ` either as a vector (1D case) or a tuple of vectors (nD case).

  - 1D mesh, with `npts` = ``N_x``

```math
x_i, \\, i=1,\\dots,N_x
```

  - 2D mesh, with `npts` = (``N_x``, ``N_y``)

```math
([x_i]_{i=1}^{N_x}, [y_j]_{j=1}^{N_y})
```

  - 3D mesh, with `npts` = (``N_x``, ``N_y``, ``N_z``)

```math
([x_i]_{i=1}^{N_x}, [y_j]_{j=1}^{N_y}, [z_l]_{l=1}^{N_z}).
```
"""
@inline function points(Ωₕ::AbstractMeshType)
	error("Interface function 'points' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	point(Ωₕ::AbstractMeshType, idx)

Returns the tuple with the point from the mesh corresponding to index `idx`. See [points](@ref).
"""
@inline function point(Ωₕ::AbstractMeshType, idx)
	error("Interface function 'point' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	points_iterator(Ωₕ::AbstractMeshType)

Returns an iterator over the  points of `Ωₕ`.
"""
@inline function points_iterator(Ωₕ::AbstractMeshType)
	error("Interface function 'points_iterator' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	half_points(Ωₕ::AbstractMeshType)

Returns the half points, for each submesh.

  - 1D mesh (with `idx`=``(i,)`` or ``i``)

```math
x_{i+1/2} \\vcentcolon = x_i + \\frac{h_{i+1}}{2}, \\, i=1,\\dots,N-1,
```

```math
x_{N+1/2} \\vcentcolon = x_{N}``and``x_{1/2} \\vcentcolon = x_1``.
```

  - 2D mesh, with `idx` = ``(i,j)``

```math
(x_{i+1/2}, y_{j+1/2})
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(x_{i+1/2}, y_{j+1/2}, z_{l+1/2}).
```
"""
@inline function half_points(Ωₕ::AbstractMeshType)
	error("Interface function 'half_points' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	half_point(Ωₕ::AbstractMeshType, idx)

Returns a tuple with the [half_points](@ref), for each submesh, of the points at index `idx`.
"""
@inline function half_point(Ωₕ::AbstractMeshType, idx)
	error("Interface function 'half_point' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	half_points_iterator(Ωₕ::AbstractMeshType)

Returns an iterator for the [half_points](@ref), for each submesh.
"""
@inline function half_points_iterator(Ωₕ::AbstractMeshType)
	error("Interface function 'half_points_iterator' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	spacing(Ωₕ::AbstractMeshType, idx)

Returns a tuple with the [spacing](@ref), for each submesh, at index `idx`.

  - 1D mesh, with `idx` = ``(i,)`` or ``i``

```math
h_{x,i} \\vcentcolon = x_i - x_{i-1}, \\, i=2,\\dots,N
```

and ``h_{x,1} \\vcentcolon = x_2 - x_1``

  - 2D mesh, with `idx` = ``(i,j)``

```math
(h_{x,i}, h_{y,j}) \\vcentcolon = (x_i - x_{i-1}, y_j - y_{j-1})
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(h_{x,i}, h_{y,j}, h_{z,l}) \\vcentcolon = (x_i - x_{i-1}, y_j - y_{j-1}, z_l - z_{l-1})
```
"""
@inline function spacing(Ωₕ::AbstractMeshType, idx)
	error("Interface function 'spacing' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	spacings_iterator(Ωₕ::AbstractMeshType)

Returns an iterator for the [spacing](@ref), for each submesh.
"""
@inline function spacings_iterator(Ωₕ::AbstractMeshType)
	error("Interface function 'spacings_iterator' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	forward_spacing(Ωₕ::AbstractMeshType, idx)

Returns a tuple with the [forward_spacing](@ref), for each submesh, at index `idx`.

  - 1D mesh, with `idx` = ``(i,)`` or ``i``

```math
h_{x,i} \\vcentcolon = x_{i+1} - x_i, \\, i=1,\\dots,N-1
```

and ``h_{x,N} \\vcentcolon = x_N - x_{N-1}``

  - 2D mesh, with `idx` = ``(i,j)``

```math
(h_{x,i}, h_{y,j}) \\vcentcolon = (x_{i+1} - x_i, y_{j+1} - y_j)
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(h_{x,i}, h_{y,j}, h_{z,l}) \\vcentcolon = (x_{i+1} - x_i, y_{j+1} - y_j, z_{l+1} - z_l)
```
"""
@inline function forward_spacing(Ωₕ::AbstractMeshType, idx)
	error("Interface function 'forward_spacing' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	spacings_iterator(Ωₕ::AbstractMeshType)

Returns an iterator for the [spacing](@ref), for each submesh.
"""
@inline function forward_spacings_iterator(Ωₕ::AbstractMeshType)
	error("Interface function 'forward_spacings_iterator' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	half_spacings(Ωₕ::AbstractMeshType)

Returns the indexwise average of the space stepsize, for each submesh.

  - 1D mesh, with `idx` = ``(i,)`` or ``i``

```math
h_{x,i+1/2} \\vcentcolon = \\frac{h_{x,i} + h_{x,i+1}}{2}, \\, i=1,\\dots,N-1,
```

```math
h_{x,N+1/2} \\vcentcolon = \\frac{h_{N}}{2}`` and ``h_{x,1/2} \\vcentcolon = \\frac{h_1}{2}``.
```

  - 2D mesh, with `idx` = ``(i,j)``

```math
(h_{x,i+1/2}, h_{y,j+1/2})
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(h_{x,i+1/2}, h_{y,j+1/2}, h_{z,l+1/2})
```
"""
@inline function half_spacings(Ωₕ::AbstractMeshType)
	error("Interface function 'half_spacings' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	half_spacing(Ωₕ::AbstractMeshType, idx)

Returns a tuple with the [half_spacings](@ref), for each submesh, at index `idx`.
"""
@inline function half_spacing(Ωₕ::AbstractMeshType, idx)
	error("Interface function 'half_spacing' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	half_spacings_iterator(Ωₕ::AbstractMeshType)

Returns an iterator for the [half_spacing](@ref), for each submesh.
"""
@inline function half_spacings_iterator(Ωₕ::AbstractMeshType)
	error("Interface function 'half_spacings_iterator' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	npoints(Ωₕ::AbstractMeshType)
	npoints(Ωₕ::AbstractMeshType, [::Type{Tuple}])

Returns the number of points of mesh `Ωₕ`. If `Tuple` is passed as the second argument, it returns a tuple with the number of points of each submesh composing `Ωₕ`.
"""
@inline function npoints(Ωₕ::AbstractMeshType)
	error("Interface function 'npoints' not implemented for mesh of type $(typeof(Ωₕ)).")
end

@inline function npoints(Ωₕ::AbstractMeshType, ::Type{Tuple})
	error("Interface function 'npoints' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	hₘₐₓ(Ωₕ::AbstractMeshType)

Returns the maximum diagonal of mesh `Ωₕ`.

  - 1D mesh

```math
h_{max} \\vcentcolon = \\max_{i=1,\\dots,N} x_i - x_{i-1}.
```

  - 2D mesh

```math
\\max_{i,j} \\Vert (h_{x,i}, h_{y,j}) \\Vert_2
```

  - 3D mesh

```math
\\max_{i,j,l} \\Vert (h_{x,i}, h_{y,j},  h_{z,l}) \\Vert_2
```
"""
@inline function hₘₐₓ(Ωₕ::AbstractMeshType)
	error("Interface function 'hₘₐₓ' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	cell_measure(Ωₕ::AbstractMeshType, idx)

Returns the measure of the cell ``\\square_{idx}`` centered at the index `idx` (can be a `CartesianIndex` or a `Tuple`):

  - 1D mesh, with `idx` = ``(i,)``

```math
\\square_{i} \\vcentcolon = \\left[x_i - \\frac{h_{i}}{2}, x_i + \\frac{h_{i+1}}{2} \\right]
```

at `CartesianIndex` `i` in mesh `Ωₕ`, which is given by ``h_{i+1/2}``

  - 2D mesh

```math
  \\square_{i,j} \\vcentcolon = \\left[x_i - \\frac{h_{x,i}}{2}, x_i + \\frac{h_{x,i+1}}{2} \\right] \\times \\left[y_j - \\frac{h_{y,j}}{2}, y_j + \\frac{h_{y,j+1}}{2} \\right]
```

with area ``h_{x,i+1/2} h_{y,j+1/2}``, where `idx` = ``(i,j)``,

  - 3D mesh

```math
\\square_{i,j,l} \\vcentcolon = \\left[x_i - \\frac{h_{x,i}}{2}, x_i + \\frac{h_{x,i+1}}{2}\\right] \\times \\left[y_j - \\frac{h_{y,j}}{2}, y_j + \\frac{h_{y,j+1}}{2}\\right] \\times \\left[z_l - \\frac{h_{z,l}}{2}, z_l + \\frac{h_{z,l+1}}{2}\\right]
```

with volume ``h_{x,i+1/2} h_{y,j+1/2} h_{z,l+1/2}``, where `idx` = ``(i,j,l)``.
"""
@inline function cell_measure(Ωₕ::AbstractMeshType, idx)
	error("Interface function 'cell_measure' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	cell_measure_iterator(Ωₕ::AbstractMeshType)

Returns an iterator for the measure of the cells.
"""
@inline function cell_measure_iterator(Ωₕ::AbstractMeshType)
	error("Interface function 'cell_measure_iterator' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	iterative_refinement!(Ωₕ::AbstractMeshType)
	iterative_refinement!(Ωₕ::AbstractMeshType, domain_markers::DomainMarkers)

Refines the given mesh `Ωₕ` by halving each existing cell (in every direction). If an object of type [DomainMarkers](@ref) is passed as an argument, it also updates the markers according to accordingly after the refinement.
"""
@inline function iterative_refinement!(Ωₕ::AbstractMeshType)
	error("Interface function 'iterative_refinement!' not implemented for mesh of type $(typeof(Ωₕ)).")
end
@inline function iterative_refinement!(Ωₕ::AbstractMeshType, domain_markers::DomainMarkers)
	error("Interface function 'iterative_refinement!' not implemented for mesh of type $(typeof(Ωₕ)).")
end

"""
	change_points!(Ωₕ::AbstractMeshType, pts)
	change_points!(Ωₕ::AbstractMeshType, Ω::Domain, pts)

Changes the coordinates of the internal points of the mesh `Ωₕ` to the new coordinates specified in `pts`. This function assumes the points in `pts` are ordered and that the first and last of them coincide with the bounds of the `Ω`. if the domain `Ω` is passed as an argument, the markers of the mesh are also recalculated after this change.
"""
@inline function change_points!(Ωₕ::AbstractMeshType, pts)
	error("Interface function 'change_points!' not implemented for mesh of type $(typeof(Ωₕ)).")
end
@inline function change_points!(Ωₕ::AbstractMeshType, ::DomainMarkers, pts)
	error("Interface function 'change_points!' not implemented for mesh of type $(typeof(Ωₕ)).")
end