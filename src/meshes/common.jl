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
	embed(Ωₕ::MeshType, f)

See [embed](@ref embed(f, Ω::Domain)) for [Domain](@ref)s on general use. This function creates an extra wrapper allowing the embedded function be be directly calculated on a `CartesianIndex` associated with the points of mesh `Ωₕ`.
"""
function embed(Ωₕ::MeshType{D}, f) where D
	T = eltype(Ωₕ)
	wrapped_f_tuple = FunctionWrapper{T,Tuple{NTuple{D,T}}}(f)
	pts = points(Ωₕ)
	g = Base.Fix1(_i2p, pts)
	fog(idx) = f(g(idx))
	wrapped_f_cartesian = FunctionWrapper{T,Tuple{CartesianIndex{D}}}(fog)

	return BrambleFunction{D,T,true}(wrapped_f_tuple, wrapped_f_cartesian)
end

"""
	↪(Ωₕ::MeshType, f)

Alias for [embed](@ref embed(Ωₕ::MeshType, f)).
"""
@inline ↪(Ωₕ::MeshType, f) = embed(Ωₕ, f)

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
