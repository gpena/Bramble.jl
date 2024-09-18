"""
Abstract type for meshes of topological dimension `D`.
"""
abstract type MeshType{D} <: BrambleType end

VecCartIndex{D} = Set{CartesianIndex{D}} where D
MeshMarkers{D} = Dict{String,VecCartIndex{D}} where D

"""
$(SIGNATURES)
Returns the number of points in the `mesh`.
"""
npoints(mesh::MeshType) = mesh.npts

"""
$(SIGNATURES)
Returns the topological dimension of the `mesh`.
"""
dim(mesh::MeshType{D}) where D = D

"""
$(SIGNATURES)
Returns the element type of the `mesh` points.
"""
eltype(mesh::MeshType) = eltype(typeof(mesh))

"""
$(SIGNATURES)
Returns the `CartesianIndices` associated with the `mesh` points.
"""
indices(mesh::MeshType{D}) where D = (mesh.indices)::CartesianIndices{D}

marker(mesh::MeshType, m) = mesh.markers[m]

@inline _index2point(pts, idx) = pts[idx]
@inline @generated _index2point(pts::NTuple{D}, indices::CartesianIndex{D}) where D = :(Base.Cartesian.@ntuple $D i->pts[i][indices[i]])
@inline @generated _index2nextpoint(pts::NTuple{D}, indices::CartesianIndex{D}) where D = :(Base.Cartesian.@ntuple $D i->pts[i][indices[i] + 1])