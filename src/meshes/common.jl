"""
     MeshType{D}
     
Abstract type for meshes of topological dimension `D`.
"""
abstract type MeshType{D} <: BrambleType end

VecCartIndex{D} = Set{CartesianIndex{D}} where D
MeshMarkers{D} = Dict{String,VecCartIndex{D}} where D

"""
    ndofs(Ωₕ::MeshType)

Returns the number of points in mesh `Ωₕ`.
"""
ndofs(Ωₕ::MeshType) = Ωₕ.npts

"""
    dim(Ωₕ::MeshType)

Returns the topological dimension of mesh `Ωₕ`.
"""
dim(Ωₕ::MeshType{D}) where D = D

"""
    eltype(Ωₕ::MeshType) 

Returns the element type of mesh `Ωₕ` points.
"""
eltype(Ωₕ::MeshType) = eltype(typeof(Ωₕ))

"""
    indices(Ωₕ::MeshType)
    
Returns the `CartesianIndices` associated with points of mesh `Ωₕ`.
"""
indices(Ωₕ::MeshType{D}) where D = (Ωₕ.indices)::CartesianIndices{D}

marker(Ωₕ::MeshType, m) = Ωₕ.markers[m]

@inline _index2point(pts, idx) = pts[idx]
@inline @generated _index2point(pts::NTuple{D}, indices::CartesianIndex{D}) where D = :(Base.Cartesian.@ntuple $D i->pts[i][indices[i]])
@inline @generated _index2nextpoint(pts::NTuple{D}, indices::CartesianIndex{D}) where D = :(Base.Cartesian.@ntuple $D i->pts[i][indices[i] + 1])