"""
	 MeshType{D}

Abstract type for meshes. Meshes are only parametrized by their dimension `D``.
"""
abstract type MeshType{D} <: BrambleType end

"""
	generate_indices(nPoints::Int)
	generate_indices(nPoints::NTuple)

Returns the `CartesianIndices` of a mesh with `nPoints[i]` in each direction or just `nPoints`, if the argument is an `Int`.
"""
@inline generate_indices(npts::Int) = CartesianIndices((npts,))
@inline generate_indices(nPoints::NTuple{D,Int}) where D = CartesianIndices(ntuple(i -> 1:nPoints[i], D))

"""
	is_boundary_index(idx, Ωₕ::MeshType)

Returns true if the index `idx` is a boundary index of the mesh.
"""
@inline is_boundary_index(idx, Ωₕ::MeshType) = is_boundary_index(idx, indices(Ωₕ))

"""
	boundary_indices(Ωₕ::MeshType)

Returns the boundary indices of mesh `Ωₕ`.
"""
@inline boundary_indices(Ωₕ::MeshType) = boundary_indices(indices(Ωₕ))

"""
	interior_indices(Ωₕ::MeshType)

Returns the interior indices of mesh `Ωₕ`.
"""
@inline interior_indices(Ωₕ::MeshType) = interior_indices(indices(Ωₕ))

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
	eltype(Ωₕ::MeshType)

Returns the element type of the points of `Ωₕ`.
"""
@inline Base.eltype(Ωₕ::MeshType) = _eltype(Ωₕ)

"""
	dim(Ωₕ::MeshType)
	dim(::Type{<:MeshType})

Returns the dimension of the space where `Ωₕ` is embedded.
"""
@inline dim(_::MeshType{D}) where D = D
@inline dim(::Type{<:MeshType{D}}) where D = D

"""
	topo_dim(Ωₕ::MeshType)

Returns the topological dimension `Ωₕ`.
"""
@inline @generated function topo_dim(Ωₕ::MeshType{D}) where D
	if D <= 0
		return :(0) # Handle edge case
	end

	term_expression = :((npoints(Ωₕ(i)) == 1) ? 0 : dim(Ωₕ(i)))
	generated_code = :(sum(Base.Cartesian.@ntuple $D i->$term_expression))
	return generated_code
end

"""
	indices(Ωₕ::MeshType)

Returns the `CartesianIndices` associated with the points of mesh `Ωₕ`.
"""
@inline indices(Ωₕ::MeshType) = Ωₕ.indices

"""
	backend(Ωₕ::MeshType)

Returns the linear algebra backend`associated with the mesh`Ωₕ`.
"""
@inline backend(Ωₕ::MeshType) = Ωₕ.backend

"""
	markers(Ωₕ::MeshType)

Returns the [DomainMarkers](@ref)) associated with the mesh `Ωₕ`.
"""
@inline markers(Ωₕ::MeshType) = Ωₕ.markers

"""
	set_indices!(Ωₕ::MeshType, indices)

	Overrides the indices in Ωₕ.
"""
@inline set_indices!(Ωₕ::MeshType, indices) = (Ωₕ.indices = indices)

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
	points(Ωₕ::MeshType)
	points(Ωₕ::MeshType, idx)

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
@inline points(Ωₕ::MeshType) = _points(Ωₕ)
@inline points(Ωₕ::MeshType, idx) = _points(Ωₕ, idx)

"""
	points_iterator(Ωₕ::MeshType)

Returns an iterator over the  points of `Ωₕ`.
"""
@inline points_iterator(Ωₕ::MeshType) = _points_iterator(Ωₕ)

"""
	half_points(Ωₕ::MeshType, idx)

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

@inline half_points(Ωₕ::MeshType, idx) = _half_points(Ωₕ, idx)

"""
	half_points_iterator(Ωₕ::MeshType)

Returns an iterator for the [half_points](@ref), for each submesh.
"""
@inline half_points_iterator(Ωₕ::MeshType) = _half_points_iterator(Ωₕ)

"""
	spacing(Ωₕ::MeshType, idx)

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
@inline spacing(Ωₕ::MeshType, idx) = _spacing(Ωₕ, idx)

"""
	spacing_iterator(Ωₕ::MeshType)

Returns an iterator for the [spacing](@ref), for each submesh.
"""
@inline spacing_iterator(Ωₕ::MeshType) = _spacing_iterator(Ωₕ)

"""
	half_spacing(Ωₕ::MeshType, idx)

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
@inline half_spacing(Ωₕ::MeshType, idx) = _half_spacing(Ωₕ, idx)

"""
	half_spacing_iterator(Ωₕ::MeshType)

Returns an iterator for the [half_spacing](@ref), for each submesh.
"""
@inline half_spacing_iterator(Ωₕ::MeshType) = _half_spacing_iterator(Ωₕ)

"""
	npoints(Ωₕ::MeshType)
	npoints(Ωₕ::MeshType, Tuple)

Returns the number of points of mesh `Ωₕ`. If `Tuple` is passed as the second argument, it returns a tuple with the number of points of each submesh composing `Ωₕ`.
"""
@inline npoints(Ωₕ::MeshType) = _npoints(Ωₕ)
@inline npoints(Ωₕ::MeshType, ::Type{Tuple}) = _npoints(Ωₕ, Tuple)

"""
	hₘₐₓ(Ωₕ::MeshType)

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
@inline hₘₐₓ(Ωₕ::MeshType) = _hₘₐₓ(Ωₕ)

"""
	cell_measure(Ωₕ::MeshType, idx)

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
@inline cell_measure(Ωₕ::MeshType, idx) = _cell_measure(Ωₕ, idx)

"""
	cell_measure_iterator(Ωₕ::MeshType)

Returns an iterator for the measure of the cells.
"""
@inline cell_measure_iterator(Ωₕ::MeshType) = _cell_measure_iterator(Ωₕ)

"""
	iterative_refinement!(Ωₕ::MeshType)
	iterative_refinement!(Ωₕ::MeshType, domain_markers::DomainMarkers)

Refines the given mesh `Ωₕ` by halving each existing cell (in every direction). If [DomainMarkers](@ref) is passed as an argument, it also updates the markers according to `domain_markers` after the refinement.
"""
@inline iterative_refinement!(Ωₕ::MeshType) = _iterative_refinement!(Ωₕ)
@inline iterative_refinement!(Ωₕ::MeshType, domain_markers::DomainMarkers) = _iterative_refinement!(Ωₕ, domain_markers)

"""
	change_points!(Ωₕ::MeshType, pts)
	change_points!(Ωₕ::MeshType, Ω::Domain, pts)

Changes the coordinates of the internal points of the mesh `Ωₕ` to the new coordinates specified in `pts`. This function assumes the points in `pts` are ordered and that the first and last of them coincide with the bounds of the `Ω`. if the domain `Ω` is passed as an argument, the markers of the mesh are also recalculated after this change.
"""
@inline change_points!(Ωₕ::MeshType, pts) = _change_points!(Ωₕ, pts)
@inline change_points!(Ωₕ::MeshType, domain_markers::DomainMarkers, pts) = _change_points!(Ωₕ, domain_markers, pts)