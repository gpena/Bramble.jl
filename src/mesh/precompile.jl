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

	# Points
	points(Ωₕ)
	if dim(Ωₕ) == 1
		_idx = idx_tup[1]
		point(Ωₕ, _idx)
		spacing(Ωₕ, _idx)
		half_spacing(Ωₕ, _idx)
		half_point(Ωₕ, _idx)
		cell_measure(Ωₕ, _idx)
	else
		point(Ωₕ, idx_tup)
		spacing(Ωₕ, idx_tup)
		half_spacing(Ωₕ, idx_tup)
		half_point(Ωₕ, idx_tup)
		cell_measure(Ωₕ, idx_tup)
	end

	point(Ωₕ, idx_cart)
	spacing(Ωₕ, idx_cart)

	hₘₐₓ(Ωₕ)

	# Indexing
	is_boundary_index(indices(Ωₕ), idx_cart)
	boundary_indices(Ωₕ)
	interior_indices(Ωₕ)

	# Iterators
	for iterator_func in (points_iterator, half_points_iterator, spacings_iterator, half_spacings_iterator, cell_measures_iterator)
		iter = iterator_func(Ωₕ)
		!isempty(iter) && first(iter)
	end

	# nD-specific calls
	if D > 1
		Ωₕ(1) # Test accessor for sub-mesh
	end
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

	# --- 1D Setup ---
	_I = interval(zero(_PrecompilePointType), one(_PrecompilePointType))
	_dm1D = markers(_I, :left => :left, :right => :right)
	_Ω1D = domain(_I, _dm1D)
	_Ωₕ1D = mesh(_Ω1D, 5, true; backend = _backend_inst)

	# --- nD Setup ---
	domain2D = domain(box((0, 0), (1, 1)))
	domain3D = domain(box((0, 0, 0), (1, 1, 1)))
	markers2D = markers(domain2D)
	markers3D = markers(domain3D)

	mesh2D = mesh(domain2D, (10, 10), (true, true))
	mesh3D = mesh(domain3D, (5, 5, 5), (true, true, true))
	mesh2D_nonuniform = mesh(domain2D, (10, 10), (false, false))

	@compile_workload begin
		# --- 1D Workload ---
		_precompile_common_interface(_Ωₕ1D)
		_precompile_mutating_interface!(_Ωₕ1D, _dm1D)

		# 1D-specific calls from original file
		set_points!(deepcopy(_Ωₕ1D), points(_Ωₕ1D))
		index_in_marker(_Ωₕ1D, :left)

		# --- nD Workload ---
		_precompile_common_interface(mesh2D)
		_precompile_mutating_interface!(mesh2D, markers2D)

		_precompile_common_interface(mesh3D)
		_precompile_mutating_interface!(mesh3D, markers3D)

		_precompile_common_interface(mesh2D_nonuniform)

		boundary_symbol_to_dict(indices(_Ωₕ1D))

		@info "Mesh precompilation complete."
	end
end
