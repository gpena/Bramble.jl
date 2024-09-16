"""
	MeshnD{n,T}(markers, indices, pts, npts)

	Create a cartesian nD-mesh with `npts` points. For efficiency,
	the mesh points are not stored. Instead, we store the points
	of the 1D meshes that make up the nD mesh. To connect both nD and 
	1D meshes, we use the indices in `indices`. The mesh is defined 
	by the markers in `markers`, which stores which indices 
	correspond to which marker.
"""
struct MeshnD{n,T} <: MeshType{n}
	markers::MeshMarkers{n}
	indices::CartesianIndices{n,NTuple{n,UnitRange{Int}}}
	npts::Int
	submeshes::NTuple{n,Mesh1D{T}}
end

"""
	Mesh(domain, npts, unif)

Create a cartesian nD-mesh from the domain `domain`. The number of points for each projection
mesh are given in the tuple `npts`. The distribution of points on
the submeshes are encoded in the tuple `unif`. The mesh is defined
by the markers in `markers`, which stores which indices
correspond to which marker.

# Example

```jldoctest
julia> mesh = Mesh(domain, (10, 15), (true, true))
domain = Domain(Interval(0,1) × Interval(4,5))
```
"""
function Mesh(domain::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}) where D
	@assert dim(domain) == D
	T = eltype(domain)
	R = generate_indices(npts)
	meshes = ntuple(i -> Mesh(Domain(projection(domain, i)), npts[i], unif[i])::Mesh1D{T}, D)

	markersForMesh::MeshMarkers{D} = MeshMarkers{D}()

	for label in labels(domain)
		merge!(markersForMesh, Dict(label => VecCartIndex{D}()))
	end

	addmarkers!(markersForMesh, domain, meshes)

	return MeshnD{D,T}(markersForMesh, R, prod(npts), meshes)
end

"""
	dim(mesh::MeshnD)

Return the topological dimension of the mesh
"""
dim(_::Type{<:MeshnD{D,T}}) where {T,D} = D

"""
	eltype(mesh::MeshnD)

Return the type of the points in the mesh
"""
eltype(_::Type{<:MeshnD{D,T}}) where {D,T} = T

"""
	show(io, mesh)

Print a description of the mesh to the IO stream `io`.
"""
function show(io::IO, mesh::MeshnD)
	D = dim(mesh)
	properties = ["$(D)D Mesh",
		"#Points: $(mesh.npts)",
		"Markers: $(keys(mesh.markers))"]

	println(io, join(properties, "\n"))

	print(io, "\nSubmeshes:\n")

	direction = ["x", "y", "z"]
	properties = ["  $(direction[i]) direction | #Points: $(npoints(mesh)[i])" for i in 1:D]

	print(io, join(properties, "\n"))
end

"""
	eltype(mesh)

Return the eltype of the mesh.
"""
@inline eltype(_::MeshnD{D,T}) where {D,T} = T

"""
	dim(mesh)

Return the dimension of the mesh.
"""
@inline dim(_::MeshnD{D}) where D = D

"""
	mesh(i)

Return the i-th submesh.
"""
@inline (mesh::MeshnD)(i) = mesh.submeshes[i]

"""
	points(mesh)

Return the points of the mesh.
"""
@inline @generated points(mesh::MeshnD{D}) where D = :(Base.Cartesian.@ntuple $D i->points(mesh(i)))
#ntuple(i -> points(mesh(i)), D)

"""
	points(mesh, idx)

Return the point at index `idx` in the mesh.
"""
@inline @generated points(mesh::MeshnD{D}, idx::NTuple{D,Int}) where D = :(Base.Cartesian.@ntuple $D i->points(mesh(i), idx[i]))
#ntuple(i -> points(mesh(i), idx[i]), D)

"""
	xmean(mesh, idx)

Compute the mean of the points at index `idx` in the mesh.
"""
@inline @generated xmean(mesh::MeshnD{D}, idx::NTuple{D,Int}) where D = :(Base.Cartesian.@ntuple $D i->xmean(mesh(i), idx[i]))
#ntuple(i -> xmean(mesh(i), idx[i]), D)

"""
	hspace(mesh, idx)

Compute the space step size at index `idx` in the mesh.
"""
@inline @generated hspace(mesh::MeshnD{D}, idx::NTuple{D,Int}) where D = :(Base.Cartesian.@ntuple $D i->hspace(mesh(i), idx[i]))
#ntuple(i -> hspace(mesh(i), idx[i]), D)

"""
	hmean(mesh, idx)

Compute the mean of the space step sizes at index `idx` in the mesh.
"""
@inline @generated hmean(mesh::MeshnD{D}, idx::NTuple{D,Int}) where D = :(Base.Cartesian.@ntuple $D i->hmean(mesh(i), idx[i]))
#ntuple(i -> hmean(mesh(i), idx[i]), D)

"""
	pointsit(mesh)

Return an iterator over all points in the mesh.
"""
pointsit(mesh::MeshnD{D}) where D = Base.Iterators.product(ntuple(i -> pointsit(mesh(i)), D)...)

"""
	xmeanit(mesh)

Return an iterator over all mean points in the mesh.
"""
xmeanit(mesh::MeshnD{D}) where D = Base.Iterators.product(ntuple(i -> xmeanit(mesh(i)), D)...)

"""
	hspaceit(mesh)

Return an iterator over all space step sizes in the mesh.
"""
hspaceit(mesh::MeshnD{D}) where D = Base.Iterators.product(ntuple(i -> hspaceit(mesh(i)), D)...)

"""
	hmeanit(mesh)

Return an iterator over all mean space step sizes in the mesh.
"""
hmeanit(mesh::MeshnD{D}) where D = Base.Iterators.product(ntuple(i -> hmeanit(mesh(i)), D)...)

"""
	npoints(mesh)

Return the number of points of the mesh.
"""
@inline @generated npoints(mesh::MeshnD{D,T}) where {D,T} = :(Base.Cartesian.@ntuple $D i->ndofs(mesh(i)))
#ntuple(i -> ndofs(mesh(i)), D)

"""
	hₘₐₓ(mesh)

Return the maximum diagonal of the mesh.
"""
@inline function hₘₐₓ(mesh::MeshnD{D,T}) where {D,T}
	diagonals = Iterators.map(h -> hypot(h...), hspaceit(mesh)) #(hypot(h...) for h in hspaceit(mesh))
	return maximum(diagonals)
end

"""
	meas_cell(mesh, idx)

Return the measure of the cell at the index `idx`.
"""
@inline meas_cell(mesh::MeshnD{D,T}, idx::CartesianIndex{D}) where {D,T} = prod(hmean(mesh, Tuple(idx)))
@inline meas_cellit(mesh::MeshnD{D,T}) where {D,T} = (meas_cell(mesh, idx) for idx in indices(mesh))

"""
	generate_indices(nPoints)

Return the indices of a mesh with `nPoints[i]` in each direction.
"""
@inline generate_indices(nPoints::NTuple{D,Int}) where D = CartesianIndices(ntuple(i -> 1:nPoints[i], D))

"""
	is_boundary_index(idx, R)

Return true if the index `idx` is a boundary index of the mesh with indices stored in `R`.
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
	bcindices(mesh)

Return the boundary indices of the mesh.
"""
@inline bcindices(mesh::MeshnD) = bcindices(indices(mesh))
@inline bcindices(R::CartesianIndices{D}) where D = (i for i in R if is_boundary_index(i, R))

"""
	intindices(mesh)

Return the interior indices of the mesh.
"""
@inline intindices(R::CartesianIndices{D}) where D = CartesianIndices(ntuple(i -> 2:size(R)[i], D))
@inline intindices(mesh::MeshnD) = intindices(indices(mesh))

"""
	addmarkers!(markers, domain, meshes)

Add the markers of the `domain` to the `markers` of the mesh, using the `submeshes`.
"""
function addmarkers!(mrks::MeshMarkers{D}, domain::Domain, submeshes::NTuple{D,Mesh1D}) where D
	R = generate_indices(ntuple(i -> ndofs(submeshes[i]), D))
	boundary = bcindices(R)

	for idx in boundary, m in markers(domain)
		point = ntuple(i -> points(submeshes[i])[idx[i]], D)
		if m.f(point) ≈ 0
			push!(mrks[m.label], idx)
		end
	end
end
