"""
	struct MeshnD{D, BackendType, CartIndicesType, Mesh1DType} <: MeshType{D}
		markers::MeshMarkers{D}
		indices::CartIndicesType
		backend::BackendType
		submeshes::NTuple{D, Mesh1DType}
	end

Structure to store a cartesian nD-mesh (``2 \\leq n \\leq 3``). For efficiency, the mesh points are not stored. Instead, we store the points of the 1D meshes that make up the nD mesh. To connect both nD and 1D meshes, we use the indices in `indices`. The [DomainMarkers](@ref) are translated to `markers` as for [Mesh1D](@ref).
"""
mutable struct MeshnD{D,BackendType<:Backend,CartIndicesType,Mesh1DType<:MeshType{1}} <: MeshType{D}
	markers::MeshMarkers{D}
	indices::CartIndicesType
	const backend::BackendType
	submeshes::NTuple{D,Mesh1DType}
end

@generated function generate_submeshes(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}, backend) where D
	return :(Base.Cartesian.@ntuple $D i->_mesh(domain(projection(Ω, i)), (npts[i],), (unif[i],), backend))
end

function _mesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}, backend) where D
	@assert dim(Ω) == D
	_set = set(Ω)
	npts_with_collapsed = ntuple(i -> _set.collapsed[i] ? 1 : npts[i], D)

	idxs = generate_indices(npts_with_collapsed)
	submeshes = generate_submeshes(Ω, npts_with_collapsed, unif, backend)

	mesh_markers = MeshMarkers{D}()
	__mesh = MeshnD(mesh_markers, idxs, backend, submeshes)
	set_markers!(__mesh, markers(Ω))

	return __mesh
end

@inline _eltype(_::MeshnD{D,BackendType}) where {D,BackendType} = eltype(BackendType)
@inline Base.eltype(_::Type{<:MeshnD{D,BackendType}}) where {D,BackendType} = eltype(BackendType)

function Base.show(io::IO, Ωₕ::MeshnD{D}) where D
	fields = ("Markers", "Resolution")
	mlength = max_length_fields(fields)

	type_info = style_title("$(D)D mesh", max_length = mlength)
	println(io, type_info)

	npts_per_dim = npoints(Ωₕ, Tuple)
	total_pts = prod(npts_per_dim)

	colors = style_color_sets()
	num_colors = length(colors)
	styled_dims = (styled"{$(colors[mod1(i, num_colors)]):$(npts_per_dim[i])}" for i in 1:D)

	dims_str = join(styled_dims, styled" {light_black:× }")

	points_info_str = "$(total_pts) (" * dims_str * ")"

	resolution_line = style_field("Resolution", points_info_str, max_length = mlength)
	println(io, resolution_line)

	show(io, markers(Ωₕ))
end

"""
	(Ωₕ::MeshnD)(i)

Returns the `i`-th submesh of `Ωₕ`.
"""
@inline function (Ωₕ::MeshnD)(i)
	@assert 1 <= i <= dim(Ωₕ)
	return Ωₕ.submeshes[i]
end

@inline @generated _points(Ωₕ::MeshnD{D}) where D = :(Base.Cartesian.@ntuple $D i->_points(Ωₕ(i)))
@inline @generated _points(Ωₕ::MeshnD{D}, idx) where D = :(Base.Cartesian.@ntuple $D i->_points(Ωₕ(i), idx[i]))

@inline @generated _points_iterator(Ωₕ::MeshnD{D}) where D = :(Base.Iterators.product((Base.Cartesian.@ntuple $D i->_points_iterator(Ωₕ(i)))...))

@inline @generated _half_points(Ωₕ::MeshnD{D}, idx) where D = :(Base.Cartesian.@ntuple $D i->_half_points(Ωₕ(i), idx[i]))

@inline @generated _half_points_iterator(Ωₕ::MeshnD{D}) where D = :(Base.Iterators.product((Base.Cartesian.@ntuple $D i->_half_points_iterator(Ωₕ(i)))...))

@inline @generated _spacing(Ωₕ::MeshnD{D}, idx) where D = :(Base.Cartesian.@ntuple $D i->_spacing(Ωₕ(i), idx[i]))

@inline @generated _spacing_iterator(Ωₕ::MeshnD{D}) where D = :(Base.Iterators.product((Base.Cartesian.@ntuple $D i->_spacing_iterator(Ωₕ(i)))...))

@inline @generated function _half_spacing(Ωₕ::MeshnD{D}, idx) where D
	return :(Base.Cartesian.@ntuple $D i->_apply_hs_logic(_half_spacing(Ωₕ(i), idx[i])))
end

@inline _apply_hs_logic(value::T) where T = ifelse(value == zero(T), one(T), value)

@inline @generated _half_spacing_iterator(Ωₕ::MeshnD{D}) where D = :(Base.Iterators.product((Base.Cartesian.@ntuple $D i->_half_spacing_iterator(Ωₕ(i)))...))

@inline @generated _npoints(Ωₕ::MeshnD) = :(prod(_npoints(Ωₕ, Tuple)))
@inline @generated _npoints(Ωₕ::MeshnD{D}, ::Type{Tuple}) where D = :(Base.Cartesian.@ntuple $D i->_npoints(Ωₕ(i)))

@inline function _hₘₐₓ(Ωₕ::MeshnD)
	diagonals = Base.Iterators.map(h -> hypot(h...), spacing_iterator(Ωₕ))
	return maximum(diagonals)
end

@inline _cell_measure(Ωₕ::MeshnD, idx) = prod(_half_spacing(Ωₕ, idx))

@inline _cell_measure_iterator(Ωₕ::MeshnD) = (_cell_measure(Ωₕ, idx) for idx in indices(Ωₕ))

function _iterative_refinement!(Ωₕ::MeshnD{D}) where D
	for i in 1:D
		_iterative_refinement!(Ωₕ(i))
	end
end

function _iterative_refinement!(Ωₕ::MeshnD{D}, domain_markers) where D
	_iterative_refinement!(Ωₕ)

	npts = _npoints(Ωₕ, Tuple)
	idxs = generate_indices(npts)

	set_indices!(Ωₕ, idxs)
	set_markers!(Ωₕ, domain_markers)
end

function _change_points!(Ωₕ::MeshnD{D}, pts::NTuple{D,VecType}) where {D,VecType}
	for i in 1:D
		_change_points!(Ωₕ(i), pts[i])
	end
end

function _change_points!(Ωₕ::MeshnD{D}, domain_markers, pts::NTuple{D,VecType}) where {D,VecType}
	_change_points!(Ωₕ, pts)
	set_markers!(Ωₕ, domain_markers)
end