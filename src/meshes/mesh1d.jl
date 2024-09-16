"""
	Mesh1D{T}(markers, indices, pts, npts)

	Create a 1D mesh with `npts` points of type `T`. The points that define the mesh are 
	stored in `pts` and are identified, following the same order, with the
	indices in `indices`. The mesh is defined by the markers in `markers`, 
	which stores which indices correspond to which marker.
"""
struct Mesh1D{T} <: MeshType{1}
	markers::MeshMarkers{1}
	indices::CartesianIndices{1,Tuple{Base.OneTo{Int}}}
	pts::Vector{T}
	npts::Int
end

"""
	Mesh(domain::Domain{CartesianProduct{1,T}, MarkersType}, npts::Int, unif::Bool) where {T, MarkersType}

Create a 1D mesh based on `domain` and `npts` with uniform spacing
if `unif` is `true` (otherwise, the points are randomly generated).

#Example

```jldoctest
julia> mesh = Mesh(Domain(I), 10, true);
I = Interval(0,1);
```
"""
function Mesh(domain::Domain{CartesianProduct{1,T},MarkersType}, npts::Int, unif::Bool) where {T,MarkersType}
	R, pts, markersForMesh = create_mesh1d_basics(domain, npts, unif)

	return Mesh1D{T}(markersForMesh, R, pts, npts)
end

"""
	Mesh(domain::Domain{CartesianProduct{1,T}, MarkersType}, npts::NTuple{1,Int}, uniform::NTuple{1,Bool}) where {T, MarkersType}

Create a 1D mesh based on `domain` and `npts` with uniform spacing
if `unif` is `true` (otherwise, the points are randomly generated).
"""
Mesh(domain::Domain{CartesianProduct{1,T},Markers}, npts::NTuple{1,Int}, unif::NTuple{1,Bool}) where {T,Markers} = Mesh(domain, npts[1], unif[1])

"""
	dim(mesh::Mesh1D)

Return the topological dimension of the mesh
"""
dim(_::Type{<:Mesh1D{T}}) where T = 1

"""
	eltype(mesh::Mesh1D)

Return the type of the points in the mesh
"""
eltype(_::Type{<:Mesh1D{T}}) where T = T

"""
	create_mesh1d_basics(domain::Domain, npts::Int, unif::Bool)

Create the basic components of a 1D mesh.
"""
function create_mesh1d_basics(domain::Domain, npts::Int, unif::Bool)
	pts = Vector{eltype(domain)}(undef, npts)
	createpoints!(pts, set(domain), npts, unif)
	R = generate_indices(npts)

	markersForMesh = MeshMarkers{1}()

	for label in labels(domain)
		merge!(markersForMesh, Dict(label => VecCartIndex{1}()))
	end

	addmarkers!(markersForMesh, domain, R, pts)

	return R, pts, markersForMesh
end

"""
	eltype(mesh::Mesh1D{T})

Return the type of the points in `mesh`.
"""
@inline eltype(_::Mesh1D{T}) where T = T

"""
	dim(mesh::Mesh1D{T})

Return the topologicaldimension of the domain
associated with `mesh`.
"""
@inline dim(_::Mesh1D{T}) where T = 1

"""
	show(io::IO, mesh::Mesh1D)

Print a summary of the mesh to `io`.

# Example

```jldoctest
julia> mesh
I = Interval(0,1);
```
"""
function show(io::IO, mesh::Mesh1D)
	l = join(keys(mesh.markers), ", ")
	properties = ["1D Mesh",
		"#Points: $(mesh.npts)",
		"Markers: $l"]

	print(io, join(properties, "\n"))
end

"""
	(mesh::Mesh1D)()

Return the mesh itself.
"""
@inline (mesh::Mesh1D)(_) = mesh

"""
	npoints(mesh::Mesh1D)

Return a tuple of the number of points in the mesh.

#Example

```jldoctest
julia> npoints(mesh)
I = Interval(0,1);
```
"""
@inline npoints(mesh::Mesh1D) = (mesh.npts,)

"""
	points(mesh::Mesh1D)

Return a vector of all points in the mesh.
"""
@inline points(mesh::Mesh1D) = mesh.pts

"""
	points(mesh::Mesh1D, i)

Return the point at index `i`.
"""
@inline points(mesh::Mesh1D, i) = mesh.pts[i]

"""
	pointsit(mesh::Mesh1D)

Return an iterator over all points in the mesh.
"""
@inline pointsit(mesh::Mesh1D) = points(mesh)

"""
	hₘₐₓ(mesh::Mesh1D)

Compute the maximum of the space step sizes in the mesh.
"""
@inline hₘₐₓ(mesh::Mesh1D) = maximum(hspaceit(mesh))

"""
	hspace(mesh::Mesh1D, i)

Compute the space step size at index `i` in the mesh.
"""
@inline function hspace(mesh::Mesh1D, i)
	@assert 1 <= i <= ndofs(mesh)

	if i === 1
		return mesh.pts[2] - mesh.pts[1]
	else
		return mesh.pts[i] - mesh.pts[i - 1]
	end
end

"""
	hspaceit(mesh::Mesh1D)

Return an iterator over all space step sizes in the mesh.
"""
@inline hspaceit(mesh::Mesh1D) = (hspace(mesh, i) for i in 1:ndofs(mesh))

"""
	hmean(mesh::Mesh1D{T}, i)

Compute the mean of the space step size at index `i` in the mesh.
"""
@inline function hmean(mesh::Mesh1D{T}, i) where T
	@assert 1 <= i <= ndofs(mesh)

	x = zero(eltype(mesh))

	if i == 1 || i == ndofs(mesh)
		x = hspace(mesh, i)
	else
		x = hspace(mesh, i + 1) + hspace(mesh, i)
	end

	return x * convert(T, 0.5)
end

"""
	hmeanit(mesh::Mesh1D)

Return an iterator over all mean space step sizes in the mesh.
"""
@inline hmeanit(mesh::Mesh1D) = (hmean(mesh, i) for i in 1:ndofs(mesh))

"""
	xmean(mesh::Mesh1D{T}, i)

Compute the mean of the points at index `i` in the mesh.
"""
@inline function xmean(mesh::Mesh1D{T}, i) where T
	@assert 1 <= i <= ndofs(mesh) + 1

	if i == 1
		return points(mesh, 1)
	elseif i == ndofs(mesh) + 1
		return points(mesh, ndofs(mesh))
	else
		return (points(mesh, i) + points(mesh, i - 1)) * convert(T, 0.5)
	end
end

"""
	xmeanit(mesh::Mesh1D)

Return an iterator over all mean points in the mesh.
"""
@inline xmeanit(mesh::Mesh1D) = (xmean(mesh, i) for i in 1:(ndofs(mesh) + 1))

"""
	meas_cell(mesh::Mesh1D, idx::CartesianIndex{1})

Compute the measure of the cell at index `idx` in the mesh.
"""
@inline meas_cell(mesh::Mesh1D, idx::CartesianIndex{1}) = hmean(mesh, idx[1])

"""
	meas_cellit(mesh::Mesh1D)

Return an iterator over all cell measures in the mesh.
"""
@inline meas_cellit(mesh::Mesh1D) = Iterators.map(Base.Fix1(meas_cell, mesh), indices(mesh))

"""
	createpoints!(x::Vector{T}, I::CartesianProduct{1,T}, npts::Int, unif::Bool) where T

Create a vector of points `x` uniformly or randomly distributed in the interval `I`.
"""
@inline function createpoints!(x::Vector{T}, I::CartesianProduct{1,T}, npts::Int, unif::Bool) where T
	x .= range(zero(T), one(T), length = npts)
	v = view(x, 2:(npts - 1))

	if !unif
		rand!(v)
		sort!(v)
	end

	a, b = tails(I)
	@.. x = a + x * (b - a)
end

"""
	generate_indices(npts::Int)

Generate a CartesianIndices object for a 1D array of length `npts`.
"""
@inline generate_indices(npts::Int) = CartesianIndices((npts,))

"""
	bcindices(R::CartesianIndices{1})

Compute the boundary indices of the mesh.
"""
@inline bcindices(R::CartesianIndices{1}) = (first(R), last(R))

"""
	bcindices(mesh::Mesh1D)

Compute the boundary indices of the mesh.
"""
@inline bcindices(mesh::Mesh1D) = bcindices(indices(mesh))

"""
	intindices(R::CartesianIndices{1})

Compute the interior indices of the mesh.
"""
@inline intindices(R::CartesianIndices{1}) = CartesianIndices((2:(length(R) - 1),))

"""
	intindices(mesh::Mesh1D)

Compute the interior indices of the mesh.
"""
@inline intindices(mesh::Mesh1D) = CartesianIndices((2:(ndofs(mesh) - 1),))

"""
	addmarkers!(mrks::MeshMarkers{1}, domain::Domain, R::CartesianIndices{1}, pts)

Add markers to the mesh based on the domain.
"""
function addmarkers!(markerList::MeshMarkers{1}, domain::Domain, R::CartesianIndices{1}, pts)
	boundary = bcindices(R)

	for idx in boundary, marker in markers(domain)
		if marker.f(_index2point(pts, idx)) ≈ 0
			push!(markerList[marker.label], idx)
		end
	end
end
