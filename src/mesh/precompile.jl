# 1D meshes are indexed by a scalar, nD meshes by the index tuple.
function _precompile_indexed_ops(Ωₕ::AbstractMeshType{1}, idx_tup)
	i = idx_tup[1]
	point(Ωₕ, i)
	spacing(Ωₕ, i)
	half_spacing(Ωₕ, i)
	half_point(Ωₕ, i)
	cell_measure(Ωₕ, i)
	forward_spacing(Ωₕ, i)
	return nothing
end

function _precompile_indexed_ops(Ωₕ::AbstractMeshType, idx_tup)
	point(Ωₕ, idx_tup)
	spacing(Ωₕ, idx_tup)
	half_spacing(Ωₕ, idx_tup)
	half_point(Ωₕ, idx_tup)
	cell_measure(Ωₕ, idx_tup)
	return nothing
end

function _precompile_common_interface(Ωₕ)
	# Ensure types are concrete for precompilation
	idx_cart = first(indices(Ωₕ))
	idx_tup = Tuple(idx_cart)
	D = dim(Ωₕ)

	# Basic Accessors
	dim(Ωₕ)
	dim(typeof(Ωₕ))
	eltype(Ωₕ)
	eltype(typeof(Ωₕ))
	indices(Ωₕ)
	markers(Ωₕ)
	backend(Ωₕ)
	npoints(Ωₕ)
	npoints(Ωₕ, Tuple)

	# Points. Dispatch on the dimension rather than branching on `dim(Ωₕ) == 1`, so that
	# inference never has to consider the nD signatures for a 1D mesh (and vice versa).
	points(Ωₕ)
	_precompile_indexed_ops(Ωₕ, idx_tup)

	point(Ωₕ, idx_cart)
	spacing(Ωₕ, idx_cart)

	hₘₐₓ(Ωₕ)

	# Indexing
	is_boundary_index(indices(Ωₕ), idx_cart)
	boundary_indices(Ωₕ)
	interior_indices(Ωₕ)

	# Iterators
	for iterator_func in (points_iterator, half_points_iterator, spacings_iterator, half_spacings_iterator, cell_measures_iterator, forward_spacings_iterator)
		iter = iterator_func(Ωₕ)
		!isempty(iter) && first(iter)
	end

	# nD-specific calls
	if D > 1
		Ωₕ(1) # Test accessor for sub-mesh
	end

	# Additional accessors
	set(Ωₕ)
	topo_dim(Ωₕ)
	is_collapsed(Ωₕ(1))

	forward_spacing(Ωₕ, idx_cart)
end

function _precompile_mutating_interface!(Ωₕ, dm)
	iterative_refinement!(deepcopy(Ωₕ))
	iterative_refinement!(deepcopy(Ωₕ), dm)

	# Use existing points for change_points! call
	pts = points(Ωₕ)
	change_points!(deepcopy(Ωₕ), pts)
	change_points!(deepcopy(Ωₕ), dm, pts)
end

@setup_workload begin
	# --- Common Setup ---
	_PrecompileBackendType = Bramble.backend
	_PrecompilePointType = Float64
	_backend_inst = _PrecompileBackendType()

	# --- ESSENTIAL: 1D Setup ---
	_I = interval(zero(_PrecompilePointType), one(_PrecompilePointType))
	_dm1D = markers(_I, :left => :left, :right => :right)
	_Ω1D = domain(_I, _dm1D)
	_Ωₕ1D = mesh(_Ω1D, 5, true; backend = _backend_inst)

	# --- ESSENTIAL: 2D Setup ---
	_domain2D = domain(box((0.0, 0.0), (1.0, 1.0)))
	_markers2D = markers(_domain2D)
	_mesh2D = mesh(_domain2D, (5, 5), (true, true); backend = _backend_inst)
	_mesh2D_nonuniform = mesh(_domain2D, (5, 5), (false, false); backend = _backend_inst)

	@compile_workload begin
		# --- ESSENTIAL: 1D Workload ---
		_precompile_common_interface(_Ωₕ1D)
		_precompile_mutating_interface!(_Ωₕ1D, _dm1D)

		# 1D-specific calls
		set_points!(deepcopy(_Ωₕ1D), points(_Ωₕ1D))
		index_in_marker(_Ωₕ1D, :left)

		# Test boundary_indices on both mesh and indices
		boundary_indices(_Ωₕ1D)
		boundary_indices(indices(_Ωₕ1D))
		interior_indices(_Ωₕ1D)
		interior_indices(indices(_Ωₕ1D))
		boundary_symbol_to_dict(indices(_Ωₕ1D))

		# --- ESSENTIAL: 2D Workload ---
		_precompile_common_interface(_mesh2D)
		_precompile_mutating_interface!(_mesh2D, _markers2D)
		_precompile_common_interface(_mesh2D_nonuniform)
		boundary_indices(_mesh2D)
		interior_indices(_mesh2D)

		# Direct indexing & show
		_mesh2D[1, 1]
		_ = sprint(show, _mesh2D)

		# --- EXTENDED: 3D Workload ---
		if BRAMBLE_EXTENDED_PRECOMPILE
			domain3D = domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
			markers3D = markers(domain3D)
			mesh3D = mesh(domain3D, (4, 4, 4), (true, true, true); backend = _backend_inst)

			_precompile_common_interface(mesh3D)
			_precompile_mutating_interface!(mesh3D, markers3D)
		end
	end
end
