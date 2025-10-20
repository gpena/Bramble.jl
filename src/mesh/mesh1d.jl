"""
	mutable struct Mesh1D{BackendType,CartIndicesType,VectorType,T} <: AbstractMeshType{1}
		set::CartesianProduct{1,T}
		markers::MeshMarkers
		indices::CartIndicesType
		const backend::BackendType
		pts::VectorType
		half_pts::VectorType
		half_spacing::VectorType
		collapsed::Bool
	end	

Structure to create a 1D mesh. The points that define the grid are stored in `pts` and are identified, following the same order, with the indices in field `indices`. See also [CartesianProduct](@ref), [MeshMarkers](@ref) and [Backend](@ref).

For future reference, the entries of vector `pts` are

```math
x_i, \\, i=1,\\dots,N.
```
"""
mutable struct Mesh1D{BackendType<:Backend,CartIndicesType,VectorType<:AbstractVector,T} <: AbstractMeshType{1}
	const set::CartesianProduct{1,T}
	markers::MeshMarkers
	indices::CartIndicesType
	const backend::BackendType
	pts::VectorType
	half_pts::VectorType
	half_spacings::VectorType
	collapsed::Bool
end

@inline is_collapsed(Ωₕ::Mesh1D) = Ωₕ.collapsed

@inline points(Ωₕ::Mesh1D) = Ωₕ.pts
@inline point(Ωₕ::Mesh1D, i) = points(Ωₕ)[i]
@inline points_iterator(Ωₕ::Mesh1D) = Ωₕ.pts

@inline half_points(Ωₕ::Mesh1D) = Ωₕ.half_pts
@inline half_spacings(Ωₕ::Mesh1D) = Ωₕ.half_spacings
@inline cell_measures(Ωₕ::Mesh1D) = half_spacings(Ωₕ)

"""
	set_points!(Ωₕ::Mesh1D, pts)

Overrides the points in Ωₕ. This function recalculates the cached [half_points](@ref) and [half_spacings](@ref).
"""
@inline function set_points!(Ωₕ::Mesh1D, pts)
	Ωₕ.pts = pts

	half_points!(Ωₕ, vector(backend(Ωₕ), length(pts)+1))
	half_spacings!(Ωₕ, vector(backend(Ωₕ), length(pts)))

	half_points!(half_points(Ωₕ), Ωₕ)
	half_spacing!(half_spacings(Ωₕ), Ωₕ)
	return
end

"""
	half_points!(Ωₕ::Mesh1D, pts)

Overrides the half_points in Ωₕ.
"""
@inline half_points!(Ωₕ::Mesh1D, pts) = (Ωₕ.half_pts = pts; return)

"""
	half_spacings!(Ωₕ::Mesh1D, pts)

Overrides the spacings in Ωₕ.
"""
@inline half_spacings!(Ωₕ::Mesh1D, pts) = (Ωₕ.half_spacings = pts; return)

@inline eltype(::Mesh1D{BackendType}) where BackendType = eltype(BackendType)
@inline eltype(::Type{<:Mesh1D{BackendType}}) where BackendType = eltype(BackendType)

@inline (Ωₕ::Mesh1D)(_) = Ωₕ

@inline npoints(Ωₕ::Mesh1D) = length(points(Ωₕ))
@inline npoints(Ωₕ::Mesh1D, ::Type{Tuple}) = (npoints(Ωₕ),)

@inline hₘₐₓ(Ωₕ::Mesh1D) = maximum(spacings_iterator(Ωₕ))

@inline @inbounds function spacing(Ωₕ::Mesh1D, i::Int)
	if is_collapsed(Ωₕ)
		return eltype(Ωₕ)(0.0)
	end

	pts = points(Ωₕ)
	if i == 1
		return pts[2] - pts[1]
	end

	return pts[i] - pts[i-1]
end

@inline spacing(Ωₕ::Mesh1D, i::CartesianIndex{1}) = spacing(Ωₕ, i[1])

@inline @inbounds function forward_spacing(Ωₕ::Mesh1D, i::Int)
	if is_collapsed(Ωₕ)
		return eltype(Ωₕ)(0.0)
	end

	pts = points(Ωₕ)
	N = npoints(Ωₕ)

	if i == N
		return pts[N] - pts[N-1]
	end

	return pts[i+1] - pts[i]
end

@inline forward_spacing(Ωₕ::Mesh1D, i::CartesianIndex{1}) = forward_spacing(Ωₕ, i[1])

@inline half_point(Ωₕ::Mesh1D, i::Int) = Ωₕ.half_pts[i]

@inline spacings_iterator(Ωₕ::Mesh1D) = (spacing(Ωₕ, i) for i in eachindex(points(Ωₕ)))
@inline forward_spacings_iterator(Ωₕ::Mesh1D) = (forward_spacing(Ωₕ, i) for i in eachindex(points(Ωₕ)))

@inline half_spacing(Ωₕ::Mesh1D, i::Int) = Ωₕ.half_spacings[i]

@inline half_spacing(Ωₕ::Mesh1D, idx::CartesianIndex{1}) = half_spacing(Ωₕ, idx[1])
@inline cell_measure(Ωₕ::Mesh1D, i) = half_spacing(Ωₕ, i)

@inline half_spacings_iterator(Ωₕ::Mesh1D) = Ωₕ.half_spacings
@inline half_points_iterator(Ωₕ::Mesh1D) = Ωₕ.half_pts
@inline cell_measures_iterator(Ωₕ::Mesh1D) = half_spacings_iterator(Ωₕ)

_generate_random_points!(v) = (rand!(v); sort!(v); return)

@inline @inbounds @muladd function _points!(x, I::CartesianProduct{1}, unif::Bool)
	npts = length(x)
	T = eltype(I)
	a, b = tails(I)

	if npts == 1
		x .= zero(T)
		return
	end

	if unif
		h = (b - a) / (npts - 1)
		@simd for i in eachindex(x)
			x[i] = a + (i - 1) * h
		end
	else
		x[1] = zero(T)
		x[npts] = one(T)

		v = view(x, 2:(npts - 1))
		_generate_random_points!(v)

		@. x = a + x * (b - a)
	end
	return
end

@inline @muladd function half_points!(x, Ωₕ)
	n = npoints(Ωₕ)
	pts = points(Ωₕ)

	x[1] = pts[1]
	x[n+1] = pts[n]

	@inbounds @simd for i in 2:n
		x[i] = (pts[i] + pts[i-1]) * 0.5
	end

	return
end

@inline @inbounds @muladd function half_spacing!(x, Ωₕ)
	n = npoints(Ωₕ)

	x[1] = spacing(Ωₕ, 1) * 0.5
	x[n] = spacing(Ωₕ, n) * 0.5

	@inbounds @simd for i in 2:(n - 1)
		x[i] = (spacing(Ωₕ, i) + spacing(Ωₕ, i+1)) * 0.5
	end

	return
end

function _mesh(Ω::Domain{CartesianProduct{1,T}}, npts::Tuple{Int}, unif::Tuple{Bool}, backend) where T
	@unpack set, markers = Ω
	n_points, = npts

	is_collapsed = topo_dim(set) == 0

	if is_collapsed
		n_points = 1
	end

	is_uniform, = unif

	pts = vector(backend, n_points)
	_points!(pts, set, is_uniform)

	_half_pts = vector(backend, n_points+1)
	_half_spacings = vector(backend, n_points)

	idxs = generate_indices(n_points)

	mesh_markers = MeshMarkers()
	mesh = Mesh1D(set, mesh_markers, idxs, backend, pts, _half_pts, _half_spacings, is_collapsed)

	half_points!(half_points(mesh), mesh)
	half_spacing!(half_spacings(mesh), mesh)

	set_markers!(mesh, markers)
	return mesh
end

function iterative_refinement!(Ωₕ::Mesh1D)
	if is_collapsed(Ωₕ)
		return
	end

	N_old = npoints(Ωₕ)

	if N_old <= 1
		return
	end

	N_new = 2 * N_old - 1

	new_points = vector(backend(Ωₕ), N_new)
	old_points = points(Ωₕ)

	@views @inbounds new_points[1:2:end] .= old_points
	@views @inbounds new_points[2:2:end] .= (old_points[1:(end - 1)] .+ old_points[2:end]) .* 0.5

	new_indices = generate_indices(N_new)

	set_indices!(Ωₕ, new_indices)
	set_points!(Ωₕ, new_points)
	return
end

function iterative_refinement!(Ωₕ::Mesh1D, domain_markers::DomainMarkers)
	if is_collapsed(Ωₕ)
		return
	end

	iterative_refinement!(Ωₕ)
	set_markers!(Ωₕ, domain_markers)
	return
end

function change_points!(Ωₕ::Mesh1D, domain_markers::DomainMarkers, pts)
	change_points!(Ωₕ, pts)
	set_markers!(Ωₕ, domain_markers)
	return
end

function change_points!(Ωₕ::Mesh1D, pts)
	npts = npoints(Ωₕ)
	@assert npts == length(pts)

	set_points!(Ωₕ, pts)
	return
end
