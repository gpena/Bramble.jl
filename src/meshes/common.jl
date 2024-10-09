"""
	 MeshType{D}

Abstract type for meshes. Meshes are only parametrized by their topological dimension `D``.
"""
abstract type MeshType{D} <: BrambleType end

VecCartIndex{D} = Set{CartesianIndex{D}} where D

"""
	MeshMarkers{D}

Type of dictionary to store the `CartesianIndices` associated with a [Marker](@ref).
"""
MeshMarkers{D} = Dict{Symbol,VecCartIndex{D}} where D

struct Iterator <: BrambleType end

"""
	dim(Ωₕ::MeshType)
	dim(::Type{<:MeshType})

Returns the tolopogical dimension of `Ωₕ`.
"""
@inline dim(Ωₕ::MeshType{D}) where D = D
@inline dim(::Type{<:MeshType{D}}) where D = D

"""
	eltype(Ωₕ::MeshType)
	eltype(::Type{<:MeshType})

Returns the type of element of the points of the mesh.
"""
@inline eltype(Ωₕ::MeshType) = eltype(typeof(Ωₕ))
@inline eltype(Ωₕ::Type{<:MeshType}) = eltype(typeof(Ωₕ))

"""
	indices(Ωₕ::MeshType)

Returns the `CartesianIndices` associated with the points of mesh `Ωₕ`.
"""
@inline indices(Ωₕ::MeshType{D}) where D = (Ωₕ.indices)::CartesianIndices{D}

"""
	marker(Ωₕ::MeshType, str::Symbol)

Returns the [Marker](@ref) function with label `str`.
"""
@inline marker(Ωₕ::MeshType, str::Symbol) = Ωₕ.markers[str]

# investigate if this function is necessary
@inline _i2p(pts, idx) = pts[idx]

"""
	_i2p(pts::NTuple{D, Vector{T}}, index::CartesianIndex{D})

Returns a `D` tuple with the coordinates of the point in `pts` associated with the `CartesianIndex` given by ìndex`.
"""
@inline @generated _i2p(pts::NTuple{D,Vector{T}}, index::CartesianIndex{D}) where {D,T} = :(Base.Cartesian.@ntuple $D i->pts[i][index[i]])

# necessary?!
@inline @generated _i2ppo(pts::NTuple{D,Vector{T}}, index::CartesianIndex{D}) where {D,T} = :(Base.Cartesian.@ntuple $D i->pts[i][index[i] + 1])

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
julia> I = interval(0,1); Ωₕ = mesh(domain(I), 10, true)
1D mesh
nPoints: 10
Markers: Dirichlet
```

```@example
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
	return _mesh(Ω, npts, unif)
end

function mesh(Ω::Domain{CartesianProduct{1,T},Markers}, npts::Int, unif::Bool) where {T,Markers}
	return _mesh(Ω, (npts,), (unif,))
end