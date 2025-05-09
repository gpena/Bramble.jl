"""
	mutable struct Mesh1D{BackendType,CartIndicesType,VectorType} <: AbstractMeshType{1}
		markers::MeshMarkers{1}
		indices::CartIndicesType
		const backend::BackendType
		pts::VectorType
		collapsed::Bool
	end	

Structure to create a 1D mesh. The points that define the grid are stored in `pts` and are identified, following the same order, with the indices in field `indices`. The variable `markers` is a dictionary that stores the indices associated with the [DomainMarkers](@ref) using [MarkerIndices](@ref).

For future reference, the entries of vector `pts` are

```math
x_i, \\, i=1,\\dots,N.
```
"""
mutable struct Mesh1D{BackendType<:Backend,CartIndicesType,VectorType<:AbstractVector} <: AbstractMeshType{1}
	markers::MeshMarkers{1}
	indices::CartIndicesType
	const backend::BackendType
	pts::VectorType
	collapsed::Bool
end

@inline is_collapsed(Ωₕ::Mesh1D) = Ωₕ.collapsed

@inline _points(Ωₕ::Mesh1D) = Ωₕ.pts
@inline function _points(Ωₕ::Mesh1D, i)
	idx = CartesianIndex(i)
	@assert idx in indices(Ωₕ)

	return getindex(_points(Ωₕ), idx[1])
end

@inline _points_iterator(Ωₕ::Mesh1D) = (point for point in _points(Ωₕ))

"""
	set_points!(Ωₕ::Mesh1D, pts)

	Overrides the points in Ωₕ.
"""
@inline set_points!(Ωₕ::Mesh1D, pts) = (Ωₕ.pts = pts;
										return nothing)

@inline _eltype(_::Mesh1D{BackendType}) where BackendType = eltype(BackendType)
@inline Base.eltype(::Type{<:Mesh1D{BackendType}}) where BackendType = eltype(BackendType)

function Base.show(io::IO, Ωₕ::Mesh1D)
	labels = keys(markers(Ωₕ))
	labels_styled_combined = color_markers(labels)

	fields = ("Markers",)
	mlength = max_length_fields(fields)

	labels_output = style_field("Markers", labels_styled_combined, max_length = mlength)
	type_info = style_title("1D mesh")
	npoints_info = style_field("nPoints", format_with_underscores(npoints(Ωₕ)), max_length = mlength)

	final_output = style_join(type_info, npoints_info, labels_output)
	print(io, final_output)
	return nothing
end

@inline (Ωₕ::Mesh1D)(_) = Ωₕ

@inline _npoints(Ωₕ::Mesh1D) = length(_points(Ωₕ))
@inline _npoints(Ωₕ::Mesh1D, ::Type{Tuple}) = (_npoints(Ωₕ),)

@inline _hₘₐₓ(Ωₕ::Mesh1D) = maximum(_spacing_iterator(Ωₕ))

@inline function _spacing(Ωₕ::Mesh1D, i)
	if topo_dim(Ωₕ) == 0
		return _eltype(Ωₕ)(0.0)
	end

	pts = _points(Ωₕ)
	idx = CartesianIndex(i)
	@assert idx in indices(Ωₕ)

	if idx === first(indices(Ωₕ))
		return pts[2] - pts[1]
	end

	_i = idx[1]
	_i_1 = idx[1] - 1
	return pts[_i] - pts[_i_1]
end

@inline function _half_points(Ωₕ::Mesh1D, i)
	T = eltype(Ωₕ)

	@assert i in 1:(npoints(Ωₕ) + 1)

	if i === 1
		return _points(Ωₕ, 1)
	end

	if i === _npoints(Ωₕ) + 1
		return _points(Ωₕ, _npoints(Ωₕ))
	end

	return (_points(Ωₕ, i) + _points(Ωₕ, i - 1)) * convert(T, 0.5)
end

@inline _spacing_iterator(Ωₕ::Mesh1D) = (_spacing(Ωₕ, i) for i in eachindex(_points(Ωₕ)))

@inline function _half_spacing(Ωₕ::Mesh1D, i)
	idx = CartesianIndex(i)
	idxs = indices(Ωₕ)

	@assert idx in idxs
	T = _eltype(Ωₕ)

	if idx === first(idxs) || idx === last(idxs)
		return _spacing(Ωₕ, idx) * convert(T, 0.5)::T
	end

	next = idx[1] + 1
	return (_spacing(Ωₕ, next) + _spacing(Ωₕ, idx)) * T(0.5)
end

@inline _half_spacing_iterator(Ωₕ::Mesh1D) = (_half_spacing(Ωₕ, i) for i in eachindex(_points(Ωₕ)))

@inline _half_points_iterator(Ωₕ::Mesh1D) = (_half_points(Ωₕ, i) for i in 1:(_npoints(Ωₕ) + 1))

@inline _cell_measure(Ωₕ::Mesh1D, i) = _half_spacing(Ωₕ, i)

@inline _cell_measure_iterator(Ωₕ::Mesh1D) = map(Base.Fix1(_cell_measure, Ωₕ), indices(Ωₕ))

_generate_random_points!(v) = (rand!(v); sort!(v); nothing)

@inline function _set_points!(x, I::CartesianProduct{1}, unif::Bool)
	npts = length(x)
	T = eltype(I)

	if npts == 1
		x .= zero(T)
		return nothing
	end

	x .= range(zero(T), one(T), length = npts)
	v = view(x, 2:(npts - 1))

	if !unif
		_generate_random_points!(v)
	end

	a, b = tails(I)
	@. x = a + x * (b - a)
	return nothing
end

function _mesh(Ω::Domain{CartesianProduct{1,T}}, npts::Tuple{Int}, unif::Tuple{Bool}, backend) where T
	@unpack set, markers = Ω
	n_points, = npts

	if topo_dim(Ω) == 0
		n_points = 1
	end

	is_collapsed = topo_dim(set) == 0
	is_uniform, = unif

	pts = vector(backend, n_points)

	_set_points!(pts, set, is_uniform)
	idxs = generate_indices(n_points)

	mesh_markers = MeshMarkers{1}()
	mesh = Mesh1D(mesh_markers, idxs, backend, pts, is_collapsed)

	set_markers!(mesh, markers)
	return mesh
end

function _iterative_refinement!(Ωₕ::Mesh1D)
	if is_collapsed(Ωₕ)
		return nothing
	end

	N_old = npoints(Ωₕ)

	if N_old <= 1
		return
	end

	N_new = 2 * N_old - 1

	new_points = vector(backend(Ωₕ), N_new)
	old_points = _points(Ωₕ)

	@views new_points[1:2:end] .= old_points
	@views new_points[2:2:end] .= (old_points[1:(end - 1)] .+ old_points[2:end]) .* 0.5

	new_indices = generate_indices(N_new)

	set_indices!(Ωₕ, new_indices)
	set_points!(Ωₕ, new_points)
	return nothing
end

function _iterative_refinement!(Ωₕ::Mesh1D, domain_markers)
	if is_collapsed(Ωₕ)
		return nothing
	end

	_iterative_refinement!(Ωₕ)
	set_markers!(Ωₕ, domain_markers)
	return nothing
end

function _change_points!(Ωₕ::Mesh1D, domain_markers, pts)
	_change_points!(Ωₕ, pts)
	set_markers!(Ωₕ, domain_markers)
	return nothing
end

function _change_points!(Ωₕ::Mesh1D, pts)
	npts = _npoints(Ωₕ)
	@assert npts == length(pts)

	set_points!(Ωₕ, pts)
	return nothing
end