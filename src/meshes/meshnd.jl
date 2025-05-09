"""
	struct MeshnD{D, BackendType, CartIndicesType, Mesh1DType} <: AbstractMeshType{D}
		markers::MeshMarkers{D}
		indices::CartIndicesType
		backend::BackendType
		submeshes::NTuple{D, Mesh1DType}
	end

Structure to store a cartesian nD-mesh (``2 \\leq n \\leq 3``). For efficiency, the mesh points are not stored. Instead, we store the points of the 1D meshes that make up the nD mesh. To connect both nD and 1D meshes, we use the indices in `indices`. The [DomainMarkers](@ref) are translated to `markers` as for [Mesh1D](@ref).
"""
mutable struct MeshnD{D,BackendType<:Backend,CartIndicesType,Mesh1DType<:AbstractMeshType{1}} <: AbstractMeshType{D}
	markers::MeshMarkers{D}
	indices::CartIndicesType
	const backend::BackendType
	submeshes::NTuple{D,Mesh1DType}
end

@inline function generate_submeshes(Ω::Domain, npts, unif, backend)
	return ntuple(i -> mesh(domain(projection(Ω, i)), npts[i], unif[i], backend = backend), Val(dim(Ω)))
end

function _mesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}, backend) where D
	@assert dim(Ω) == D
	_set = set(Ω)
	npts_with_collapsed = ntuple(i -> is_collapsed(_set(i)...) ? 1 : npts[i], D)

	idxs = generate_indices(npts_with_collapsed)
	submeshes = generate_submeshes(Ω, npts_with_collapsed, unif, backend)

	mesh_markers = MeshMarkers{D}()
	output_mesh = MeshnD(mesh_markers, idxs, backend, submeshes)
	set_markers!(output_mesh, markers(Ω))

	return output_mesh
end

@inline _eltype(_::MeshnD{D,BackendType}) where {D,BackendType} = eltype(BackendType)
@inline Base.eltype(_::Type{<:MeshnD{D,BackendType}}) where {D,BackendType} = eltype(BackendType)

function Base.show(io::IO, Ωₕ::MeshnD{D}) where D
	fields = ("Markers", "Resolution")
	mlength = max_length_fields(fields)

	type_info = style_title("$(D)D mesh", max_length = mlength)
	println(io, type_info)

	npts_per_dim = npoints(Ωₕ, Tuple)
	npts_per_dim_str = format_with_underscores.(npts_per_dim)
	total_pts = format_with_underscores(prod(npts_per_dim))

	colors = style_color_sets()
	num_colors = length(colors)
	styled_dims = (styled"{$(colors[mod1(i, num_colors)]):$(npts_per_dim_str[i])}" for i in 1:D)

	dims_str = join(styled_dims, styled" {light_black:× }")

	points_info_str = "$total_pts (" * dims_str * ")"

	resolution_line = style_field("Resolution", points_info_str, max_length = mlength)
	println(io, resolution_line)

	show(io, markers(Ωₕ))
	return nothing
end

"""
	(Ωₕ::MeshnD)(i)

Returns the `i`-th submesh of `Ωₕ`.
"""
@inline function (Ωₕ::MeshnD)(i)
	@assert 1 <= i <= dim(Ωₕ)
	return Ωₕ.submeshes[i]
end

@inline _points(Ωₕ::MeshnD{D}) where D = ntuple(i -> _points(Ωₕ(i)), Val(D))

@inline _points(Ωₕ::MeshnD{D}, idx) where D = ntuple(i -> _points(Ωₕ(i), idx[i]), Val(D))
@inline _half_points(Ωₕ::MeshnD{D}, idx) where D = ntuple(i -> _half_points(Ωₕ(i), idx[i]), Val(D))
@inline _spacing(Ωₕ::MeshnD{D}, idx) where D = ntuple(i -> _spacing(Ωₕ(i), idx[i]), Val(D))
@inline _half_spacing(Ωₕ::MeshnD{D}, idx) where D = ntuple(i -> _apply_hs_logic(_half_spacing(Ωₕ(i), idx[i])), Val(D))

@inline _points_iterator(Ωₕ::MeshnD{D}) where D = Iterators.product(ntuple(i -> _points_iterator(Ωₕ(i)), Val(D))...)
@inline _half_points_iterator(Ωₕ::MeshnD{D}) where D = Iterators.product(ntuple(i -> _half_points_iterator(Ωₕ(i)), Val(D))...)
@inline _spacing_iterator(Ωₕ::MeshnD{D}) where D = Iterators.product(ntuple(i -> _spacing_iterator(Ωₕ(i)), Val(D))...)
@inline _half_spacing_iterator(Ωₕ::MeshnD{D}) where D = Iterators.product(ntuple(i -> _half_spacing_iterator(Ωₕ(i)), Val(D))...)

@inline _apply_hs_logic(value::T) where T = ifelse(value == zero(T), one(T), value)

@inline _npoints(Ωₕ::MeshnD) = prod(_npoints(Ωₕ, Tuple))
@inline _npoints(Ωₕ::MeshnD{D}, ::Type{Tuple}) where D = ntuple(i -> _npoints(Ωₕ(i)), Val(D))

@inline function _hₘₐₓ(Ωₕ::MeshnD)
	diagonals = Iterators.map(h -> hypot(h...), spacing_iterator(Ωₕ))
	return maximum(diagonals)
end

@inline _cell_measure(Ωₕ::MeshnD, idx) = prod(_half_spacing(Ωₕ, idx))

@inline _cell_measure_iterator(Ωₕ::MeshnD) = (_cell_measure(Ωₕ, idx) for idx in indices(Ωₕ))

function _iterative_refinement!(Ωₕ::MeshnD{D}) where D
	for i in 1:D
		_iterative_refinement!(Ωₕ(i))
	end
	return nothing
end

function _iterative_refinement!(Ωₕ::MeshnD{D}, domain_markers) where D
	_iterative_refinement!(Ωₕ)

	npts = _npoints(Ωₕ, Tuple)
	idxs = generate_indices(npts)

	set_indices!(Ωₕ, idxs)
	set_markers!(Ωₕ, domain_markers)
	return nothing
end

function _change_points!(Ωₕ::MeshnD{D}, pts) where D
	for i in 1:D
		_change_points!(Ωₕ(i), pts[i])
	end
	return nothing
end

function _change_points!(Ωₕ::MeshnD{D}, domain_markers, pts) where D
	_change_points!(Ωₕ, pts)
	set_markers!(Ωₕ, domain_markers)
	return nothing
end