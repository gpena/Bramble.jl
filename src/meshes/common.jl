abstract type MeshType{D} <: BrambleType end

VecCartIndex{D} = Set{CartesianIndex{D}} where D
MeshMarkers{D} = Dict{String,VecCartIndex{D}} where D

"""
	ndofs(mesh::MeshType)

Return the number of points in the mesh
"""
ndofs(mesh::MeshType) = mesh.npts

"""
	dim(mesh::MeshType)

Return the dimension of the mesh
"""
dim(mesh::MeshType{D}) where D = D

"""
	eltype(mesh::MeshType)

Return the type of the points in the mesh
"""
eltype(mesh::MeshType) = eltype(typeof(mesh))

"""
	indices(mesh::MeshType)

Return the indices of the points in the mesh
"""
indices(mesh::MeshType{D}) where D = (mesh.indices)::CartesianIndices{D}

"""
	marker(mesh::MeshType, m)

Return the indices of the points in the mesh that have the marker `m`
"""
marker(mesh::MeshType, m) = mesh.markers[m]

@inline _index2point(pts, idx) = pts[idx]
@inline @generated _index2point(pts::NTuple{D}, indices::CartesianIndex{D}) where D = :(Base.Cartesian.@ntuple $D i->pts[i][indices[i]])
@inline @generated _index2nextpoint(pts::NTuple{D}, indices::CartesianIndex{D}) where D = :(Base.Cartesian.@ntuple $D i->pts[i][indices[i] + 1])
