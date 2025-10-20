"""
	struct MeshnD{D, BackendType, CartIndicesType, Mesh1DType,T} <: AbstractMeshType{D}
		set::CartesianProduct{D,T}
		markers::MeshMarkers
		indices::CartIndicesType
		backend::BackendType
		submeshes::NTuple{D,Mesh1DType}
	end

Structure to store a cartesian nD-mesh (``2 \\leq n \\leq 3``). For efficiency, the mesh points are not stored. Instead, we store the points of the 1D meshes that make up the nD mesh. To connect both nD and 1D meshes, we use the indices in `indices`. The [DomainMarkers](@ref) are translated to [MeshMarkers](@ref) as for [Mesh1D](@ref).
"""
mutable struct MeshnD{D,BackendType<:Backend,CartIndicesType,Mesh1DType<:AbstractMeshType{1},T} <: AbstractMeshType{D}
	set::CartesianProduct{D,T}
	markers::MeshMarkers
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
	npts_with_collapsed = ntuple(i -> is_collapsed(_set(i)...) ? 1 : npts[i], Val(D))

	idxs = generate_indices(npts_with_collapsed)
	submeshes = generate_submeshes(Ω, npts_with_collapsed, unif, backend)

	mesh_markers = MeshMarkers()
	output_mesh = MeshnD(_set, mesh_markers, idxs, backend, submeshes)
	set_markers!(output_mesh, markers(Ω))

	return output_mesh
end

@inline eltype(::MeshnD{D,BackendType}) where {D,BackendType} = eltype(BackendType)
@inline eltype(::Type{<:MeshnD{D,BackendType}}) where {D,BackendType} = eltype(BackendType)

"""
	(Ωₕ::MeshnD)(i)

Returns the `i`-th submesh of `Ωₕ`.
"""
@inline @inline (Ωₕ::MeshnD)(i) = Ωₕ.submeshes[i]

# A macro for functions of the form: func(Ωₕ) -> ntuple(...)
macro generate_mesh_ntuple_func(fname)
	return esc(quote
				   @inline $fname(Ωₕ::MeshnD{D}) where {D} = ntuple(i -> $fname(Ωₕ(i)), Val(D))
			   end)
end

# A macro for functions of the form: func(Ωₕ, idx) -> ntuple(...)
macro generate_mesh_ntuple_func_with_idx(fname)
	return esc(quote
				   @inline $fname(Ωₕ::MeshnD{D}, idx) where {D} = ntuple(i -> $fname(Ωₕ(i), idx[i]), Val(D))
			   end)
end

# A macro for functions of the form: func(Ωₕ) -> Iterators.product(...)
macro generate_mesh_iterator_func(fname)
	return esc(quote
				   @inline $fname(Ωₕ::MeshnD{D}) where {D} = Iterators.product(ntuple(i -> $fname(Ωₕ(i)), Val(D))...)
			   end)
end

# ntuple wrappers
@generate_mesh_ntuple_func points
@generate_mesh_ntuple_func half_points
@generate_mesh_ntuple_func half_spacings

# ntuple wrappers with an index
@generate_mesh_ntuple_func_with_idx point
@generate_mesh_ntuple_func_with_idx half_point
@generate_mesh_ntuple_func_with_idx spacing
@generate_mesh_ntuple_func_with_idx forward_spacing

@inline half_spacing(Ωₕ::MeshnD{D}, idx) where D = ntuple(i -> _apply_hs_logic(half_spacing(Ωₕ(i), idx[i])), Val(D))

# Iterator wrappers
@generate_mesh_iterator_func points_iterator
@generate_mesh_iterator_func half_points_iterator
@generate_mesh_iterator_func spacings_iterator
@generate_mesh_iterator_func forward_spacings_iterator
@generate_mesh_iterator_func half_spacings_iterator

@inline _apply_hs_logic(value::T) where T = ifelse(iszero(value), one(T), value)

@inline npoints(Ωₕ::MeshnD) = prod(npoints(Ωₕ, Tuple))
@inline npoints(Ωₕ::MeshnD{D}, ::Type{Tuple}) where D = ntuple(i -> npoints(Ωₕ(i)), Val(D))

@inline function hₘₐₓ(Ωₕ::MeshnD)
	diagonals = Iterators.map(h -> hypot(h...), spacings_iterator(Ωₕ))
	return maximum(diagonals)
end

@inline @generated function cell_measure(Ωₕ::MeshnD{D}, idx) where D
	if D == 0
		return :(1)
	end

	mesh_component_expr_1 = Expr(:call, :Ωₕ, 1)
	index_component_expr_1 = Expr(:ref, :idx, 1)
	current_product_expr = Expr(:call, :half_spacing, mesh_component_expr_1, index_component_expr_1)

	for i in 2:D
		mesh_component_expr_i = Expr(:call, :Ωₕ, i)
		index_component_expr_i = Expr(:ref, :idx, i)
		term_i_expr = Expr(:call, :half_spacing, mesh_component_expr_i, index_component_expr_i)

		current_product_expr = Expr(:call, :*, current_product_expr, term_i_expr)
	end

	return current_product_expr
end

@inline cell_measures_iterator(Ωₕ::MeshnD) = (cell_measure(Ωₕ, idx) for idx in indices(Ωₕ))

function iterative_refinement!(Ωₕ::MeshnD{D}) where D
	for i in 1:D
		iterative_refinement!(Ωₕ(i))
	end
	return
end

function iterative_refinement!(Ωₕ::MeshnD{D}, domain_markers::DomainMarkers) where D
	iterative_refinement!(Ωₕ)

	npts = npoints(Ωₕ, Tuple)
	idxs = generate_indices(npts)

	set_indices!(Ωₕ, idxs)
	set_markers!(Ωₕ, domain_markers)
	return
end

function change_points!(Ωₕ::MeshnD{D}, pts) where D
	for i in 1:D
		change_points!(Ωₕ(i), pts[i])
	end
	return
end

function change_points!(Ωₕ::MeshnD{D}, domain_markers::DomainMarkers, pts) where D
	change_points!(Ωₕ, pts)
	set_markers!(Ωₕ, domain_markers)
	return
end