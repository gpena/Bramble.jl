module BrambleMeshesExt
using Bramble: CartesianProduct, dim, Mesh1D, MeshnD, points

using Meshes: Meshes, RectilinearGrid

@inline _error_install_makie() = println("Please install `GLMakie` and add `using GLMakie` to your preamble.")
"""
	Meshes.viz(X::CartesianProduct{D,T}) where {D,T}

Visualize a [CartesianProduct](@ref).
"""
function Meshes.viz(X::CartesianProduct)
	D = dim(X)
	@assert D >= 2

	mins = ntuple(i -> X.box[i][1], D)
	maxs = ntuple(i -> X.box[i][2], D)
	mesh_set = Meshes.Box(mins, maxs)

	try
		Meshes.viz(mesh_set)
	catch
		_error_install_makie()
	end
end

"""
	Meshes.viz(Ωₕ::MeshType)

Visualize a [MeshnD](@ref).
"""
function Meshes.viz(M::MeshnD{D}) where D
	@assert D >= 2

	pts = points(M)
	grid = Meshes.RectilinearGrid(pts...)

	try
		Meshes.viz(grid, showsegments = true)
	catch
		_error_install_makie()
	end
end

Meshes.viz(M::Mesh1D) = @error "Visualization of 1D meshes is not supported"

end
