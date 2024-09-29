"""
	struct MeshnD{n,T} <: MeshType{n}
		markers::MeshMarkers{n}
		indices::CartesianIndices{n,NTuple{n,UnitRange{Int}}}
		submeshes::NTuple{n,Mesh1D{T}}
	end

Structure to store a cartesian nD-mesh (``2 \\leq n \\leq 3``) with points of type `T`. For efficiency, the mesh points are not stored. Instead, we store the points of the 1D meshes that make up the nD mesh. To connect both nD and 1D meshes, we use the indices in `indices`. The [Domain](@ref) markers are translated to `markers` as for [Mesh1D](@ref).
"""
struct MeshnD{n,T} <: MeshType{n}
	markers::MeshMarkers{n}
	indices::CartesianIndices{n,NTuple{n,UnitRange{Int}}}
	submeshes::NTuple{n,Mesh1D{T}}
end

"""
	mesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool})

Returns a [MeshnD](@ref) (``n=2,3``) defined on the [Domain](@ref) `Ω`. The number of points for each coordinate projection mesh are given in the tuple `npts`. The distribution of points on the submeshes are encoded in the tuple `unif`. The [Domain](@ref) markers are translated to `markers` as for [Mesh1D](@ref).

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
julia> X = domain(interval(0,1) × interval(4,5)); Ωₕ = mesh(X, (10, 15), (true, false))
2D mesh
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
	submeshes = ntuple(i -> mesh(domain(projection(Ω, i)), npts[i], unif[i])::Mesh1D{T}, D)

	markersForMesh::MeshMarkers{D} = MeshMarkers{D}()

	for label in labels(Ω)
		merge!(markersForMesh, Dict(label => VecCartIndex{D}()))
	end

	R = generate_indices(ntuple(i -> npoints(submeshes[i]), D))
	boundary = boundary_indices(R)

	for idx in boundary, m in markers(Ω)
		point = ntuple(i -> points(submeshes[i])[idx[i]], D)
		if isapprox(m.f(point), 0)
			push!(markersForMesh[m.label], idx)
		end
	end

	return MeshnD{D,T}(markersForMesh, R, submeshes)
end

@inline dim(_::MeshnD{D}) where D = D
@inline dim(::Type{<:MeshnD{D,T}}) where {T,D} = D

@inline eltype(_::MeshnD{D,T}) where {D,T} = T
@inline eltype(_::Type{<:MeshnD{D,T}}) where {D,T} = T

function show(io::IO, Ωₕ::MeshnD)
	D = dim(Ωₕ)
	properties = ["$(D)D mesh",
		"nPoints: $(npoints(Ωₕ))",
		"Markers: $(keys(Ωₕ.markers))"]

	println(io, join(properties, "\n"))

	print(io, "\nSubmeshes:\n")

	direction = ["x", "y", "z"]
	properties = ["  $(direction[i]) direction | nPoints: $(npoints(Ωₕ, Tuple)[i])" for i in 1:D]

	print(io, join(properties, "\n"))
end

"""
	(Ωₕ::MeshnD)(i)

Returns the `i`-th submesh of `Ωₕ`.
"""
@inline function (Ωₕ::MeshnD)(i)
	@assert 1 <= i <= dim(Ωₕ)
	return Ωₕ.submeshes[i]
end

"""
	points(Ωₕ::MeshnD)
	points(Ωₕ::MeshnD{D}, idx)
	points(Ωₕ::MeshnD{D}, Iterator)

Returns a tuple with the points of `Ωₕ`. If the `Tuple` `idx` is passed as the second argument is passed, it returns the tuple with the point corresponding to that index. Alternatively, if `Iterator` is passed as the second argument, a generator iterating over all points of the mesh is returned.

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
@inline @generated points(Ωₕ::MeshnD{D}, idx) where D = :(Base.Cartesian.@ntuple $D i->points(Ωₕ(i), idx[i]))
@inline points(Ωₕ::MeshnD{D}, ::Type{Iterator}) where D = Base.Iterators.product(ntuple(i -> points(Ωₕ(i), Iterator), D)...)

"""
	half_points(Ωₕ::MeshnD{D}, idx)
	half_points(Ωₕ::MeshnD{D}, Iterator)

Returns a tuple with the [half_points](@ref), for each submesh, of the points at index `idx`. If `Iterator` is passed as the second argument, a generator iterating over all `half_points` of the mesh is returned.

  - 2D mesh, with `idx` = ``(i,j)``

```math
(x_{i+1/2}, y_{j+1/2})
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(x_{i+1/2}, y_{j+1/2}, z_{l+1/2}).
```
"""
@inline @generated half_points(Ωₕ::MeshnD{D}, idx) where D = :(Base.Cartesian.@ntuple $D i->half_points(Ωₕ(i), idx[i]))
@inline half_points(Ωₕ::MeshnD{D}, ::Type{Iterator}) where D = Base.Iterators.product(ntuple(i -> half_points(Ωₕ(i), Iterator), D)...)

"""
	spacing(Ωₕ::MeshnD, idx::NTuple)
	spacing(Ωₕ::MeshnD{D}, Iterator)

Returns a tuple with the [spacing](@ref), for each submesh, at index `idx`. If `Iterator` is passed as the second argument, a generator iterating over all `spacing`s of the mesh is returned.

  - 2D mesh, with `idx` = ``(i,j)``

```math
(h_{x,i}, h_{y,j}) \\vcentcolon = (x_i - x_{i-1}, y_j - y_{j-1})
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(h_{x,i}, h_{y,j}, h_{z,l}) \\vcentcolon = (x_i - x_{i-1}, y_j - y_{j-1}, z_l - z_{l-1})
```
"""
@inline @generated spacing(Ωₕ::MeshnD{D}, idx) where D = :(Base.Cartesian.@ntuple $D i->spacing(Ωₕ(i), idx[i]))
@inline spacing(Ωₕ::MeshnD{D}, ::Type{Iterator}) where D = Base.Iterators.product(ntuple(i -> spacing(Ωₕ(i), Iterator), D)...)

"""
	half_spacing(Ωₕ::MeshnD, idx)
	half_spacing(Ωₕ::MeshnD{D}, Iterator)

Returns a tuple with the [half_spacing](@ref), for each submesh, at index `idx`. If `Iterator` is passed as the second argument, a generator iterating over all `half_spacing`s of the mesh is returned.

  - 2D mesh, with `idx` = ``(i,j)``

```math
(h_{x,i+1/2}, h_{y,j+1/2})
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(h_{x,i+1/2}, h_{y,j+1/2}, h_{z,l+1/2})
```
"""
@inline @generated half_spacing(Ωₕ::MeshnD{D}, idx) where D = :(Base.Cartesian.@ntuple $D i->half_spacing(Ωₕ(i), idx[i]))
@inline half_spacing(Ωₕ::MeshnD{D}, ::Type{Iterator}) where D = Base.Iterators.product(ntuple(i -> half_spacing(Ωₕ(i), Iterator), D)...)

"""
	npoints(Ωₕ::MeshnD)
	npoints(Ωₕ::MeshnD, Tuple)

Returns the number of points of mesh `Ωₕ`. If `Tuple` is passed as the second argument, it returns a tuple with the number of points of each submesh composing `Ωₕ`.
"""
@inline npoints(Ωₕ::MeshnD) = prod(npoints(Ωₕ, Tuple))
@inline @generated npoints(Ωₕ::MeshnD{D}, ::Type{Tuple}) where D = :(Base.Cartesian.@ntuple $D i->npoints(Ωₕ(i)))

"""
	hₘₐₓ(Ωₕ::MeshnD)

Returns the maximum diagonal of mesh `Ωₕ`.

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
	diagonals = Iterators.map(h -> hypot(h...), spacing(Ωₕ, Iterator))
	return maximum(diagonals)
end

"""
	cell_measure(Ωₕ::MeshnD, idx)

Returns the measure of the cell ``\\square_{idx}`` centered at the index `idx` (can be a `CartesianIndex` or a `Tuple`):

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
@inline cell_measure(Ωₕ::MeshnD{D,T}, idx) where {D,T} = prod(half_spacing(Ωₕ, idx))
@inline cell_measure(Ωₕ::MeshnD{D,T}, ::Type{Iterator}) where {D,T} = (cell_measure(Ωₕ, idx) for idx in indices(Ωₕ))

"""
	generate_indices(nPoints::NTuple)

Returns the `CartesianIndices` of a mesh with `nPoints[i]` in each direction.
"""
@inline generate_indices(nPoints::NTuple{D,Int}) where D = CartesianIndices(ntuple(i -> 1:nPoints[i], D))

"""
	is_boundary_index(idx, R::CartesianIndices)

Returns true if the index `idx` is a boundary index of the mesh with indices stored in `R`.
"""
@inline function is_boundary_index(idx, R::CartesianIndices{D}) where D
	dims = size(R)
	_idx = CartesianIndex(idx)

	for i in 1:D
		if _idx[i] == 1 || _idx[i] == dims[i]
			return true
		end
	end

	return false
end

"""
	boundary_indices(Ωₕ::MeshnD)

Returns the boundary indices of mesh `Ωₕ`.
"""
@inline boundary_indices(Ωₕ::MeshnD) = boundary_indices(indices(Ωₕ))
@inline boundary_indices(R::CartesianIndices{D}) where D = (i for i in R if is_boundary_index(i, R))

"""
	interior_indices(Ωₕ::MeshnD)

Returns the interior indices of mesh `Ωₕ`.
"""
@inline interior_indices(Ωₕ::MeshnD) = interior_indices(indices(Ωₕ))
@inline interior_indices(R::CartesianIndices{D}) where D = CartesianIndices(ntuple(i -> 2:size(R)[i], D))