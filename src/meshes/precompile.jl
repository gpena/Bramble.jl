@setup_workload begin
	# --- Setup Phase ---

	# Define BackendType and PointType based on typical usage
	# Use your actual Backend type here!
	const _PrecompileBackendType = Backend
	const _PrecompilePointType = Float64

	# --- Create dummy instances ---
	_I = interval(zero(_PrecompilePointType), one(_PrecompilePointType))

	# Markers
	_dm = create_markers(_I, :left => :left, :right => (:right,), :center => x -> isapprox(x[1], 0.5, atol = 1e-2))

	_Ω = domain(_I, _dm)

	# Mesh parameters
	_npts = 5
	_unif = true
	_backend_inst = _PrecompileBackendType()

	# Create points vector and indices
	_pts_vec = vector(_backend_inst, _npts)
	# Use the actual _set_points! function from your module
	_set_points!(_pts_vec, _I, _unif) # Pts: 0.0, 0.25, 0.5, 0.75, 1.0
	_idxs = generate_indices(_npts)

	# Create Mesh1D instance (manually or via mesh function)
	# Calling mesh() might be easier if it reliably produces the target type
	_Ωₕ = mesh(_Ω, _npts, _unif; backend = _backend_inst)

	# Ensure markers are set for subsequent calls
	# set_markers!(_Ωₕ, _dm) # mesh() should already call this

	# Prepare data for modification functions
	_new_pts_valid = [0.0, 0.15, 0.55, 0.85, 1.0] # Ensure sorted and endpoints match

	# Get a MarkerIndices instance to test merge_consecutive_indices! directly
	# This assumes the :ends marker exists and uses c_index before merging
	_marker_indices_instance = marker(_Ωₕ, :center)
	# Force calculation if needed, though mesh() should have done it
	# May need manual setup if mesh() doesn't create the exact structure needed here
	if !isa(_marker_indices_instance, MarkerIndices{1})
		# Fallback: create one manually for precompile test
		_marker_indices_instance = MarkerIndices{1}(Set([CartesianIndex(1), CartesianIndex(2), CartesianIndex(5)]), Set{CartesianIndices{1}}())
	end

	# --- End Setup Phase ---
end # @setup

# This block also runs only during precompilation.
# Use the instances created in @setup to make direct function calls.
# These calls will trigger the compilation of the specific method instances.
@compile_workload begin
	# --- Workload Phase ---
	set_markers!(_Ωₕ, _dm)
	# Call internal marker functions on representative data
	_mesh_markers_dict = markers(_Ωₕ) # Get the dict after set_markers!
	if !isempty(_mesh_markers_dict)
		process_label_for_mesh!(_mesh_markers_dict, Set([:test1, :test2]))
		# Call merge directly on a marker that likely has consecutive indices
		merge_consecutive_indices!(_marker_indices_instance)
	end
	boundary_symbol_to_cartesian(_idxs)

	# Basic Accessors
	dim(_Ωₕ)
	dim(typeof(_Ωₕ))
	idx_val = indices(_Ωₕ)
	set_indices!(_Ωₕ, idx_val) # Call setter
	markers(_Ωₕ)
	marker(_Ωₕ, :left)
	backend(_Ωₕ)
	pts_val = points(_Ωₕ)
	points(_Ωₕ, 1)
	points(_Ωₕ, CartesianIndex(2))
	set_points!(_Ωₕ, pts_val) # Call setter
	npoints(_Ωₕ)
	npoints(_Ωₕ, Tuple)
	eltype(_Ωₕ)
	eltype(typeof(_Ωₕ))

	# Geometric Properties (call with different indices)
	spacing(_Ωₕ, 1)
	spacing(_Ωₕ, 2)
	spacing(_Ωₕ, _npts)
	hₘₐₓ(_Ωₕ)
	half_spacing(_Ωₕ, 1)
	half_spacing(_Ωₕ, 2)
	half_spacing(_Ωₕ, _npts)
	half_points(_Ωₕ, 1)
	half_points(_Ωₕ, 2)
	half_points(_Ωₕ, _npts + 1) # Access last half point index
	cell_measure(_Ωₕ, 1)
	cell_measure(_Ωₕ, 2)
	cell_measure(_Ωₕ, _npts)

	# Indexing Helpers
	generate_indices(_npts)
	boundary_indices(_Ωₕ)
	interior_indices(_Ωₕ)
	boundary_indices(_idxs) # Call on CartesianIndices directly
	interior_indices(_idxs) # Call on CartesianIndices directly

	# Mesh Modification (use copies to avoid side effects between calls)
	# Must wrap modification calls in try/catch if they might error on specific inputs
	# Or ensure the setup provides valid inputs for these calls.
	try
		_Ωₕ_copy1 = deepcopy(_Ωₕ)
		iterative_refinement!(_Ωₕ_copy1)
	catch e
		; println("Precompile Warning: iterative_refinement! failed: $e");
	end

	try
		_Ωₕ_copy2 = deepcopy(_Ωₕ)
		iterative_refinement!(_Ωₕ_copy2, _dm)
	catch e
		; println("Precompile Warning: iterative_refinement!(..., dm) failed: $e");
	end

	try
		_Ωₕ_copy3 = deepcopy(_Ωₕ)
		change_points!(_Ωₕ_copy3, _new_pts_valid)
	catch e
		; println("Precompile Warning: change_points! failed: $e");
	end

	try
		_Ωₕ_copy4 = deepcopy(_Ωₕ)
		change_points!(_Ωₕ_copy4, _dm, _new_pts_valid)
	catch e
		; println("Precompile Warning: change_points!(..., dm, ...) failed: $e");
	end


	# Output / Display
	io_buffer = IOBuffer()
	show(io_buffer, _Ωₕ)

	# Internal helpers 
	try
		_pts_copy = deepcopy(_pts_vec)
		_set_points!(_pts_copy, _I, false) # Call with non-uniform flag
	catch e
		; println("Precompile Warning: _set_points! failed: $e");
	end

	# --- End Workload Phase ---
	@info "Mesh1D: complete"
end # @compile_workload




@setup_workload begin
	npts2D = (10, 11) # Example number of points
	npts3D = (5, 6, 7)
	unif2D = (true, true)
	unif3D = (true, true, true)
	domain2D = domain(box((0, 0), (1, 1)))
	domain3D = domain(box((0, 0, 0), (1, 1, 1)))

    markers2D_obj = markers(domain2D)
    markers3D_obj = markers(domain3D)
	# Example indices (ensure they are within bounds)
	idx2D_example = CartesianIndex(min(2, npts2D[1]), min(2, npts2D[2]))
	idx3D_example = CartesianIndex(min(2, npts3D[1]), min(2, npts3D[2]), min(2, npts3D[3]))
	idx2D_tuple_example = (min(2, npts2D[1]), min(2, npts2D[2]))
	idx3D_tuple_example = (min(2, npts3D[1]), min(2, npts3D[2]), min(2, npts3D[3]))

	io_buffer = IOBuffer()
end


@compile_workload begin
	# Create 2D and 3D mesh instances using the constructor
	mesh2D = mesh(domain2D, npts2D, unif2D)
	mesh3D = mesh(domain3D, npts3D, unif3D)

    mesh2D_nonuniform = mesh(domain2D, npts2D, (false, false))
	mesh3D_nonuniform = mesh(domain3D, npts3D, (false, false, false))

    # Example points for change_points!
    pts2D_example = points(mesh2D_nonuniform)
	pts3D_example = points(mesh3D_nonuniform)

	# Get concrete index types
	Idx2DType = typeof(indices(mesh2D))
	Idx3DType = typeof(indices(mesh3D))

	# Buffer for show
	# --- Execute functions for 2D Mesh to precompile them ---
	dim(mesh2D)
	eltype(mesh2D)
	show(io_buffer, mesh2D);
	take!(io_buffer) # Execute show, clear buffer
	mesh2D(1) # Calling the mesh instance (ensure D >= 1)
	points(mesh2D)
	points(mesh2D, idx2D_tuple_example)
	points_iterator(mesh2D)
	half_points(mesh2D, idx2D_tuple_example)
	half_points_iterator(mesh2D)
	spacing(mesh2D, idx2D_tuple_example)
	spacing_iterator(mesh2D)
	half_spacing(mesh2D, idx2D_tuple_example)
	half_spacing_iterator(mesh2D)
	npoints(mesh2D)
	npoints(mesh2D, Tuple)
	hₘₐₓ(mesh2D)
	cell_measure(mesh2D, idx2D_tuple_example)
	cell_measure_iterator(mesh2D)
	indices(mesh2D)
	is_boundary_index(idx2D_example, indices(mesh2D))
	is_boundary_index(idx2D_tuple_example, indices(mesh2D))
	boundary_indices(mesh2D)
	boundary_indices(indices(mesh2D))
	interior_indices(mesh2D)
	interior_indices(indices(mesh2D))
	
	change_points!(mesh2D, markers2D_obj, pts2D_example)
    iterative_refinement!(mesh2D, markers2D_obj)

	# --- Execute functions for 3D Mesh to precompile them ---
	dim(mesh3D)
	eltype(mesh3D)
	show(io_buffer, mesh3D);
	take!(io_buffer) # Execute show, clear buffer
	mesh3D(1) # Calling the mesh instance (ensure D >= 1)
	points(mesh3D)
	points(mesh3D, idx3D_tuple_example)
	points_iterator(mesh3D)
	half_points(mesh3D, idx3D_tuple_example)
	half_points_iterator(mesh3D)
	spacing(mesh3D, idx3D_tuple_example)
	spacing_iterator(mesh3D)
	half_spacing(mesh3D, idx3D_tuple_example)
	half_spacing_iterator(mesh3D)
	npoints(mesh3D)
	npoints(mesh3D, Tuple)
	hₘₐₓ(mesh3D)
	cell_measure(mesh3D, idx3D_tuple_example)
	cell_measure_iterator(mesh3D)
	indices(mesh3D)
	is_boundary_index(idx3D_example, indices(mesh3D))
	is_boundary_index(idx3D_tuple_example, indices(mesh3D))
	boundary_indices(mesh3D)
	boundary_indices(indices(mesh3D))
	interior_indices(mesh3D)
	interior_indices(indices(mesh3D))

    change_points!(mesh3D, markers3D_obj, pts3D_example)
	iterative_refinement!(mesh3D, markers3D_obj)

	# --- Precompile iterator consumption (still useful) ---
	# Getting the first element forces compilation of the iteration logic
	# Add checks for empty iterators if necessary
	!isempty(points_iterator(mesh2D)) && first(points_iterator(mesh2D))
	!isempty(half_points_iterator(mesh2D)) && first(half_points_iterator(mesh2D))
	!isempty(spacing_iterator(mesh2D)) && first(spacing_iterator(mesh2D))
	!isempty(half_spacing_iterator(mesh2D)) && first(half_spacing_iterator(mesh2D))
	!isempty(cell_measure_iterator(mesh2D)) && first(cell_measure_iterator(mesh2D))
	!isempty(boundary_indices(mesh2D)) && first(boundary_indices(mesh2D))
	!isempty(interior_indices(mesh2D)) && first(interior_indices(mesh2D))

	!isempty(points_iterator(mesh3D)) && first(points_iterator(mesh3D))
	!isempty(half_points_iterator(mesh3D)) && first(half_points_iterator(mesh3D))
	!isempty(spacing_iterator(mesh3D)) && first(spacing_iterator(mesh3D))
	!isempty(half_spacing_iterator(mesh3D)) && first(half_spacing_iterator(mesh3D))
	!isempty(cell_measure_iterator(mesh3D)) && first(cell_measure_iterator(mesh3D))
	!isempty(boundary_indices(mesh3D)) && first(boundary_indices(mesh3D))
	!isempty(interior_indices(mesh3D)) && first(interior_indices(mesh3D))

    @info "MeshnD: complete"
end # @compile_workload
