"""
	 AbstractMeshType{D}

Abstract type for meshes. Meshes are only parametrized by their dimension `D``.
"""
abstract type AbstractMeshType{D} <: BrambleType end

abstract type PointType end
struct ZeroDimensional <: PointType end
struct OneDimensional <: PointType end
struct NDimensional <: PointType end

PointTypeTrait(::Type{<:Int}) = OneDimensional()
PointTypeTrait(::Type{<:NTuple{D}}) where D = NDimensional()

"""
	generate_indices(nPoints)

Returns the `CartesianIndices` of a mesh with `nPoints[i]` in each direction or just `nPoints`, if the argument is an `Int`.
"""
@inline generate_indices(npts::T) where T = generate_indices(PointTypeTrait(T), npts)
@inline generate_indices(::OneDimensional, npts) = CartesianIndices((npts,))
@inline generate_indices(::NDimensional, npts) = CartesianIndices(ntuple(i -> 1:npts[i], length(npts)))

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

function is_boundary_index(idx, idxs::CartesianIndices)
	boundary_sections = boundary_indices(idxs)

	_idx = CartesianIndex(idx)
	return any(section -> _idx in section, boundary_sections)
end

@inline function boundary_indices(idxs::CartesianIndices)
	tup = boundary_symbol_to_cartesian(idxs)

	return ntuple(i -> tup[i], length(tup))
end

@inline function interior_indices(indices::CartesianIndices{D}) where D
	original_ranges = indices.indices

	interior_ranges_tuple = ntuple(D) do i
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

Returns the element type of the points of `Ωₕ`.
"""
@inline Base.eltype(Ωₕ::AbstractMeshType) = _eltype(Ωₕ)

"""
	dim(Ωₕ::AbstractMeshType)
	dim(::Type{<:AbstractMeshType})

Returns the dimension of the space where `Ωₕ` is embedded.
"""
@inline dim(_::AbstractMeshType{D}) where D = D
@inline dim(::Type{<:AbstractMeshType{D}}) where D = D

"""
	topo_dim(Ωₕ::AbstractMeshType)

Returns the topological dimension `Ωₕ`.
"""
@inline function topo_dim(Ωₕ::AbstractMeshType{D}) where D
	terms = ntuple(i -> ifelse(npoints(Ωₕ(i)) == 1, 0, 1), Val(D))
	return sum(terms)
end

"""
	indices(Ωₕ::AbstractMeshType)

Returns the `CartesianIndices` associated with the points of mesh `Ωₕ`.
"""
@inline indices(Ωₕ::AbstractMeshType) = Ωₕ.indices

"""
	backend(Ωₕ::AbstractMeshType)

Returns the linear algebra backend`associated with the mesh`Ωₕ`.
"""
@inline backend(Ωₕ::AbstractMeshType) = Ωₕ.backend

"""
	markers(Ωₕ::AbstractMeshType)

Returns the [DomainMarkers](@ref)) associated with the mesh `Ωₕ`.
"""
@inline markers(Ωₕ::AbstractMeshType) = Ωₕ.markers

"""
	set_indices!(Ωₕ::AbstractMeshType, indices)

	Overrides the indices in Ωₕ.
"""
@inline set_indices!(Ωₕ::AbstractMeshType, indices) = (Ωₕ.indices = indices)

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

```@example
julia> I = interval(0, 1);
	   Ωₕ = mesh(domain(I), 10, true);
1D mesh
nPoints: 10
Markers: dirichlet
```

```@example
julia> X = domain(interval(0, 1) × interval(4, 5));
	   Ωₕ = mesh(X, (10, 15), (true, false));
   2D mesh
Resolution: 150 (10 × 15)
   Markers: dirichlet
```
"""
@inline mesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}; backend = Backend()) where D = _mesh(Ω, npts, unif, backend)
@inline mesh(Ω::Domain{CartesianProduct{1,T}}, npts::Int, unif::Bool; backend = Backend()) where T = _mesh(Ω, (npts,), (unif,), backend)

"""
	points(Ωₕ::AbstractMeshType)
	points(Ωₕ::AbstractMeshType, idx)

Returns the points of `Ωₕ` either as a vector (1D case) or a tuple of vectors (nD case). If the `Tuple` `idx` is passed as the second argument, it returns the tuple with the point corresponding to that index.

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
@inline points(Ωₕ::AbstractMeshType) = _points(Ωₕ)
@inline points(Ωₕ::AbstractMeshType, idx) = _points(Ωₕ, idx)

"""
	points_iterator(Ωₕ::AbstractMeshType)

Returns an iterator over the  points of `Ωₕ`.
"""
@inline points_iterator(Ωₕ::AbstractMeshType) = _points_iterator(Ωₕ)

"""
	half_points(Ωₕ::AbstractMeshType, idx)

Returns a tuple with the half points (defined as follows), for each submesh, of the points at index `idx`.

  - 1D mesh

```
math, with `idx`=``(i,)`` or ``i``
x_{i+1/2} \\vcentcolon = x_i + \\frac{h_{i+1}}{2}, \\, i=1,\\dots,N-1,
```

``x_{N+1/2} \\vcentcolon = x_{N}`` and ``x_{1/2} \\vcentcolon = x_1``.

  - 2D mesh, with `idx` = ``(i,j)``

```math
(x_{i+1/2}, y_{j+1/2})
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(x_{i+1/2}, y_{j+1/2}, z_{l+1/2}).
```
"""

@inline half_points(Ωₕ::AbstractMeshType, idx) = _half_points(Ωₕ, idx)

"""
	half_points_iterator(Ωₕ::AbstractMeshType)

Returns an iterator for the [half_points](@ref), for each submesh.
"""
@inline half_points_iterator(Ωₕ::AbstractMeshType) = _half_points_iterator(Ωₕ)

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
@inline spacing(Ωₕ::AbstractMeshType, idx) = _spacing(Ωₕ, idx)

"""
	spacing_iterator(Ωₕ::AbstractMeshType)

Returns an iterator for the [spacing](@ref), for each submesh.
"""
@inline spacing_iterator(Ωₕ::AbstractMeshType) = _spacing_iterator(Ωₕ)

"""
	half_spacing(Ωₕ::AbstractMeshType, idx)

Returns a tuple with the indexwise average of the space stepsize, for each submesh, at index `idx`.

	- 1D mesh, with `idx` = ``(i,)`` or ``i``

```math
h_{x,i+1/2} \\vcentcolon = \\frac{h_{x,i} + h_{x,i+1}}{2}, \\, i=1,\\dots,N-1,
```

``h_{x,N+1/2} \\vcentcolon = \\frac{h_{N}}{2}`` and ``h_{x,1/2} \\vcentcolon = \\frac{h_1}{2}``.

  - 2D mesh, with `idx` = ``(i,j)``

```math
(h_{x,i+1/2}, h_{y,j+1/2})
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(h_{x,i+1/2}, h_{y,j+1/2}, h_{z,l+1/2})
```
"""
@inline half_spacing(Ωₕ::AbstractMeshType, idx) = _half_spacing(Ωₕ, idx)

"""
	half_spacing_iterator(Ωₕ::AbstractMeshType)

Returns an iterator for the [half_spacing](@ref), for each submesh.
"""
@inline half_spacing_iterator(Ωₕ::AbstractMeshType) = _half_spacing_iterator(Ωₕ)

"""
	npoints(Ωₕ::AbstractMeshType)
	npoints(Ωₕ::AbstractMeshType, Tuple)

Returns the number of points of mesh `Ωₕ`. If `Tuple` is passed as the second argument, it returns a tuple with the number of points of each submesh composing `Ωₕ`.
"""
@inline npoints(Ωₕ::AbstractMeshType) = _npoints(Ωₕ)
@inline npoints(Ωₕ::AbstractMeshType, ::Type{Tuple}) = _npoints(Ωₕ, Tuple)

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
@inline hₘₐₓ(Ωₕ::AbstractMeshType) = _hₘₐₓ(Ωₕ)

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
@inline cell_measure(Ωₕ::AbstractMeshType, idx) = _cell_measure(Ωₕ, idx)

"""
	cell_measure_iterator(Ωₕ::AbstractMeshType)

Returns an iterator for the measure of the cells.
"""
@inline cell_measure_iterator(Ωₕ::AbstractMeshType) = _cell_measure_iterator(Ωₕ)

"""
	iterative_refinement!(Ωₕ::AbstractMeshType)
	iterative_refinement!(Ωₕ::AbstractMeshType, domain_markers::DomainMarkers)

Refines the given mesh `Ωₕ` by halving each existing cell (in every direction). If [DomainMarkers](@ref) is passed as an argument, it also updates the markers according to `domain_markers` after the refinement.
"""
@inline iterative_refinement!(Ωₕ::AbstractMeshType) = _iterative_refinement!(Ωₕ)
@inline iterative_refinement!(Ωₕ::AbstractMeshType, domain_markers::DomainMarkers) = _iterative_refinement!(Ωₕ, domain_markers)

"""
	change_points!(Ωₕ::AbstractMeshType, pts)
	change_points!(Ωₕ::AbstractMeshType, Ω::Domain, pts)

Changes the coordinates of the internal points of the mesh `Ωₕ` to the new coordinates specified in `pts`. This function assumes the points in `pts` are ordered and that the first and last of them coincide with the bounds of the `Ω`. if the domain `Ω` is passed as an argument, the markers of the mesh are also recalculated after this change.
"""
@inline change_points!(Ωₕ::AbstractMeshType, pts) = _change_points!(Ωₕ, pts)
@inline change_points!(Ωₕ::AbstractMeshType, domain_markers::DomainMarkers, pts) = _change_points!(Ωₕ, domain_markers, pts)