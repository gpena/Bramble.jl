"""
	struct MeshnD{n,T} <: MeshType{n}
		markers::MeshMarkers{n}
		indices::CartesianIndices{n,NTuple{n,UnitRange{Int}}}
		npts::Int
		submeshes::NTuple{n,Mesh1D{T}}
	end

Type to store a cartesian nD-mesh (``2 \\leq n \\leq 3``) with `prod(npts)` points of type `T`. For efficiency, the mesh points are not stored. Instead, we store the points of the 1D meshes that make up the nD mesh. To connect both nD and 1D meshes, we use the indices in `indices`. The [Domain](@ref) markers are translated to `markers` as for [Mesh1D](@ref).
"""
struct MeshnD{n,T} <: MeshType{n}
	markers::MeshMarkers{n}
	indices::CartesianIndices{n,NTuple{n,UnitRange{Int}}}
	npts::Int
	submeshes::NTuple{n,Mesh1D{T}}
end

"""
	mesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool})

Returns a [MeshnD](@ref) (``1 \\leq n \\leq 3``) from the [Domain](@ref) `Ω`. The number of points for each coordinate projection mesh are given in the tuple `npts`. The distribution of points on the submeshes are encoded in the tuple `unif`. The [Domain](@ref) markers are translated to `markers` as for [Mesh1D](@ref).

For future reference, the mesh points are denoted as

  - 2D mesh, with `npts` = (``N_x``, ``N_y``)

```math
(x_i,y_j), \\, i=1,\\dots,N_x, \\, j=1,\\dots,N_y
```

  - 3D mesh, with `npts` = (``N_x``, ``N_y``, ``N_z``)

```math
(x_i,y_j,z_l), \\, i=1,\\dots,N_x, \\, j=1,\\dots,N_y, \\, l=1,\\dots,N_z.
```

# Example

```
julia> X = Domain(Interval(0,1) × Interval(4,5)); Ωₕ = Mesh(X, (10, 15), (true, false))
2D Mesh
nPoints: 150
Markers: ["Dirichlet"]

Submeshes:
  x direction | nPoints: 10
  y direction | nPoints: 15
```
"""
function mesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}) where D
	@assert dim(Ω) == D
	T = eltype(Ω)
	R = generate_indices(npts)
	meshes = ntuple(i -> mesh(domain(projection(Ω, i)), npts[i], unif[i])::Mesh1D{T}, D)

	markersForMesh::MeshMarkers{D} = MeshMarkers{D}()

	for label in labels(Ω)
		merge!(markersForMesh, Dict(label => VecCartIndex{D}()))
	end

	addmarkers!(markersForMesh, Ω, meshes)

	return MeshnD{D,T}(markersForMesh, R, prod(npts), meshes)
end

@inline dim(Ωₕ::MeshnD{D}) where D = D
@inline dim(Ωₕ::Type{<:MeshnD{D,T}}) where {T,D} = D

@inline eltype(Ωₕ::MeshnD{D,T}) where {D,T} = T
@inline eltype(Ωₕ::Type{<:MeshnD{D,T}}) where {D,T} = T

function show(io::IO, Ωₕ::MeshnD)
	D = dim(Ωₕ)
	properties = ["$(D)D Mesh",
		"nPoints: $(Ωₕ.npts)",
		"Markers: $(keys(Ωₕ.markers))"]

	println(io, join(properties, "\n"))

	print(io, "\nSubmeshes:\n")

	direction = ["x", "y", "z"]
	properties = ["  $(direction[i]) direction | nPoints: $(npoints(Ωₕ)[i])" for i in 1:D]

	print(io, join(properties, "\n"))
end

"""
	(Ωₕ::MeshnD)(i)

Returns the `i`-th submesh of `Ωₕ`.
"""
@inline (Ωₕ::MeshnD)(i) = Ωₕ.submeshes[i]

"""
	points(Ωₕ::MeshnD)

Returns a tuple with the points of `Ωₕ`

  - 2D mesh, with `npts` = (``N_x``, ``N_y``)

```math
([x_i]_{i=1}^{N_x}, [y_j]_{j=1}^{N_y})
```

  - 3D mesh, with `npts` = (``N_x``, ``N_y``, ``N_z``)

```math
([x_i]_{i=1}^{N_x}, [y_j]_{j=1}^{N_y}, [z_l]_{l=1}^{N_z}).
```
"""
@inline @generated points(Ωₕ::MeshnD{D}) where D = :(Base.Cartesian.@ntuple $D i->points(Ωₕ(i)))

"""
	point(Ωₕ::MeshnD, idx::NTuple)

Returns the [point](@ref) at index `idx` of `Ωₕ` as a tuple

  - 2D mesh, with `idx` = ``(i,j)``

```math
(x_i, y_j)
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(x_i, y_j, z_l).
```
"""
@inline @generated point(Ωₕ::MeshnD{D}, idx::NTuple{D,Int}) where D = :(Base.Cartesian.@ntuple $D i->point(Ωₕ(i), idx[i]))

"""
	half_points(Ωₕ::MeshnD, idx::NTuple)

Returns a tuple with the [half_points](@ref), for each submesh, of the points at index `idx`

  - 2D mesh, with `idx` = ``(i,j)``

```math
(x_{i+1/2}, y_{j+1/2})
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(x_{i+1/2}, y_{j+1/2}, z_{l+1/2}).
```
"""
@inline @generated half_points(Ωₕ::MeshnD{D}, idx::NTuple{D,Int}) where D = :(Base.Cartesian.@ntuple $D i->half_points(Ωₕ(i), idx[i]))

"""
	spacing(Ωₕ::MeshnD, idx::NTuple)

Returns a tuple with the [spacing](@ref), for each submesh, at index `idx`

  - 2D mesh, with `idx` = ``(i,j)``

```math
(h_{x,i}, h_{y,j}) := (x_i - x_{i-1}, y_j - y_{j-1})
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(h_{x,i}, h_{y,j}, h_{z,l}) := (x_i - x_{i-1}, y_j - y_{j-1}, z_l - z_{l-1})
```
"""
@inline @generated spacing(Ωₕ::MeshnD{D}, idx::NTuple{D,Int}) where D = :(Base.Cartesian.@ntuple $D i->spacing(Ωₕ(i), idx[i]))

"""
	half_spacing(Ωₕ::MeshnD, idx::NTuple)

Returns a tuple with the [half_spacing](@ref), for each submesh, at index `idx`

  - 2D mesh, with `idx` = ``(i,j)``

```math
(h_{x,i+1/2}, h_{y,j+1/2})
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(h_{x,i+1/2}, h_{y,j+1/2}, h_{z,l+1/2})
```
"""
@inline @generated half_spacing(Ωₕ::MeshnD{D}, idx::NTuple{D,Int}) where D = :(Base.Cartesian.@ntuple $D i->half_spacing(Ωₕ(i), idx[i]))

"""
	points_iterator(Ωₕ::MeshnD)

Returns an iterator over all [points](@ref) in mesh `Ωₕ`

  - 2D mesh

```math
(x_i, y_j), i = 1,...,N_{x}, j = 1,...,N_{y}
```

  - 3D mesh

```math
(x_i, y_j, z_l), i = 1,...,N_{x}, j = 1,...,N_{y}, l = 1,...,N_{z}.
```
"""
@inline points_iterator(Ωₕ::MeshnD{D}) where D = Base.Iterators.product(ntuple(i -> points_iterator(Ωₕ(i)), D)...)

"""
	half_points_iterator(Ωₕ::MeshnD)

Returns an iterator over all [half_points](@ref) points in mesh `Ωₕ`

  - 2D mesh

```math
(x_{i+1/2}, y_{j+1/2}), i = 1,...,N_{x}, j = 1,...,N_{y}
```

  - 3D mesh

```math
(x_{i+1/2}, y_{j+1/2}, z_{l+1/2}), i = 1,...,N_{x}, j = 1,...,N_{y}, l = 1,...,N_{z}.
```
"""
@inline half_points_iterator(Ωₕ::MeshnD{D}) where D = Base.Iterators.product(ntuple(i -> half_points_iterator(Ωₕ(i)), D)...)

"""
	spacing_iterator(Ωₕ::MeshnD{D})

Returns an iterator over all space step sizes [spacing](@ref) in mesh `Ωₕ`

  - 2D mesh

```math
(h_{x,i}, h_{y,j}), i = 1,...,N_{x}, j = 1,...,N_{y}
```

  - 3D mesh

```math
(h_{x,i}, h_{y,j}, h_{z,l}), i = 1,...,N_{x}, j = 1,...,N_{y}, l = 1,...,N_{z}.
```
"""
@inline spacing_iterator(Ωₕ::MeshnD{D}) where D = Base.Iterators.product(ntuple(i -> spacing_iterator(Ωₕ(i)), D)...)

"""
	half_spacing_iterator(Ωₕ::MeshnD)

Returns an iterator over all mean space step sizes [half_spacing](@ref) in mesh `Ωₕ`

  - 2D mesh

```math
(h_{x,i+1/2}, h_{y,j+1/2}), i = 1,...,N_{x}, j = 1,...,N_{y}
```

  - 3D mesh

```math
(h_{x,i+1/2}, h_{y,j+1/2}, h_{z,l+1/2}), i = 1,...,N_{x}, j = 1,...,N_{y}, l = 1,...,N_{z}.
```
"""
@inline half_spacing_iterator(Ωₕ::MeshnD{D}) where D = Base.Iterators.product(ntuple(i -> half_spacing_iterator(Ωₕ(i)), D)...)

"""
	npoints(Ωₕ::MeshnD)

Returns a tuple with the number of [points](@ref) of mesh `Ωₕ`, in each coordinate direction.
"""
@inline @generated npoints(Ωₕ::MeshnD{D,T}) where {D,T} = :(Base.Cartesian.@ntuple $D i->ndofs(Ωₕ(i)))

"""
	hₘₐₓ(Ωₕ::MeshnD)

Returns the maximum diagonal of mesh `Ωₕ`

  - 2D mesh

```math
\\max_{i,j} \\Vert (h_{x,i}, h_{y,j}) \\Vert_2
```

  - 3D mesh

```math
\\max_{i,j,l} \\Vert (h_{x,i}, h_{y,j},  h_{z,l}) \\Vert_2
```
"""
@inline function hₘₐₓ(Ωₕ::MeshnD{D,T}) where {D,T}
	diagonals = Iterators.map(h -> hypot(h...), spacing_iterator(Ωₕ))
	return maximum(diagonals)
end

"""
	cell_measure(Ωₕ::MeshnD, idx::CartesianIndex)

Returns the measure of the cell ``\\square_{idx}`` centered at the index `idx`

  - 2D mesh, ``\\square_{i,j} = [x_i - \\frac{h_{x,i}}{2}, x_i + \\frac{h_{x,i+1}}{2}] \\times [y_j - \\frac{h_{y,j}}{2}, y_j + \\frac{h_{y,j+1}}{2}]`` is

```math
h_{x,i+1/2} h_{y,j+1/2}
```

where `idx` = ``(i,j)``,

  - 3D mesh, ``\\square_{i,j,l} = [x_i - \\frac{h_{x,i}}{2}, x_i + \\frac{h_{x,i+1}}{2}] \\times [y_j - \\frac{h_{y,j}}{2}, y_j + \\frac{h_{y,j+1}}{2}] \\times [z_l - \\frac{h_{z,l}}{2}, z_l + \\frac{h_{z,l+1}}{2}]`` is

```math
h_{x,i+1/2} h_{y,j+1/2} h_{z,l+1/2}
```

where `idx` = ``(i,j,l)``.
"""
@inline cell_measure(Ωₕ::MeshnD{D,T}, idx::CartesianIndex{D}) where {D,T} = prod(half_spacing(Ωₕ, Tuple(idx)))
@inline cell_measure_iterator(Ωₕ::MeshnD{D,T}) where {D,T} = (cell_measure(Ωₕ, idx) for idx in indices(Ωₕ))

"""
	generate_indices(nPoints::NTuple)

Returns the `CartesianIndices` indices of a mesh with `nPoints[i]` in each direction.
"""
@inline generate_indices(nPoints::NTuple{D,Int}) where D = CartesianIndices(ntuple(i -> 1:nPoints[i], D))

"""
	is_boundary_index(idx::CartesianIndex, R::CartesianIndices)

Returns true if the index `idx` is a boundary index of the mesh with indices stored in `R`.
"""
@inline function is_boundary_index(idx::CartesianIndex{D}, R::CartesianIndices{D}) where D
	b = false
	dims = size(R)
	for i in 1:D
		if idx[i] == 1 || idx[i] == dims[i]
			b = true
			break
		end
	end

	return b
end

"""
	boundary_indices(Ωₕ::MeshnD)

Returns the boundary indices of mesh ``Ωₕ``.
"""
@inline boundary_indices(Ωₕ::MeshnD) = boundary_indices(indices(Ωₕ))
@inline boundary_indices(R::CartesianIndices{D}) where D = (i for i in R if is_boundary_index(i, R))

"""
	interior_indices(R::CartesianIndices)

Returns the interior indices of mesh ``Ωₕ``.
"""
@inline interior_indices(R::CartesianIndices{D}) where D = CartesianIndices(ntuple(i -> 2:size(R)[i], D))
@inline interior_indices(Ωₕ::MeshnD) = interior_indices(indices(Ωₕ))

"""
	addmarkers!(mrks::MeshMarkers, Ω::Domain, submeshes::NTuple{D,Mesh1D})

Adds the markers of [Domain](@ref) to the `markers` of the mesh, using the `submeshes`` in each coordinate direction.
"""
function addmarkers!(mrks::MeshMarkers{D}, Ω::Domain, submeshes::NTuple{D,Mesh1D}) where D
	R = generate_indices(ntuple(i -> ndofs(submeshes[i]), D))
	boundary = boundary_indices(R)

	for idx in boundary, m in markers(Ω)
		point = ntuple(i -> points(submeshes[i])[idx[i]], D)
		if m.f(point) ≈ 0
			push!(mrks[m.label], idx)
		end
	end
end
