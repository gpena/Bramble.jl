module BrambleMeshesExt
using Bramble: CartesianProduct

if !isdefined(Base, :get_extension)
	using ..Meshes, ..GLMakie
else
	using Meshes
	using GLMakie
end

"""
	Meshes.viz(X::CartesianProduct{D,T}) where {D,T}

Visualize a [CartesianProduct](@ref).
"""
function Meshes.viz(X::CartesianProduct{D,T}) where {D,T}
	@assert D >= 2
	println("Meshes")
	println(X.box)
	mins = ntuple(i -> X.box[i][1], D)
	maxs = ntuple(i -> X.box[i][2], D)
	mesh_set = Meshes.Box(mins, maxs)
	println(mesh_set)
	Meshes.viz(mesh_set)
end

end
