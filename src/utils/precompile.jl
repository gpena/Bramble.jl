using SparseArrays: SparseVector
using Bramble: matrix, vector
using Bramble: interval, embed_function

@setup_workload begin
	# ESSENTIAL: Float64 only
	backend_default = backend() # Vector{Float64}, SparseMatrixCSC{Float64,Int}
	backend_dense64 = backend(vector_type = Vector{Float64}, matrix_type = Matrix{Float64})
	backend_sparse64 = backend(vector_type = SparseVector{Float64,Int}, matrix_type = SparseMatrixCSC{Float64,Int})

	# EXTENDED: Float32 variants
	if BRAMBLE_EXTENDED_PRECOMPILE
		backend_dense32 = backend(vector_type = Vector{Float32}, matrix_type = Matrix{Float32})
		backend_sparse32 = backend(vector_type = SparseVector{Float32,Int32}, matrix_type = SparseMatrixCSC{Float32,Int32})
	end

	@compile_workload begin
		# ESSENTIAL: Linear algebra utilities with Float64
		u = Vector{Float64}(undef, 3)
		v = Vector{Float64}(undef, 3)
		w = Vector{Float64}(undef, 3)
		fill!(u, 1.0)
		fill!(v, 2.0)
		fill!(w, 0.5)
		_dot(u, v, w)
		_inner_product(u, w, v)

		# ESSENTIAL: vector/matrix for Float64 backends
		vector(backend_default, 10)
		vector(backend_dense64, 10)
		vector(backend_sparse64, 10)

		matrix(backend_default, 5, 5)
		matrix(backend_dense64, 5, 5)
		matrix(backend_sparse64, 5, 5)

		# EXTENDED: Float32 and other types
		if BRAMBLE_EXTENDED_PRECOMPILE
			# Test with Float32
			u_f32 = Vector{Float32}(undef, 3)
			v_f32 = Vector{Float32}(undef, 3)
			w_f32 = Vector{Float32}(undef, 3)
			fill!(u_f32, 1.0f0)
			fill!(v_f32, 2.0f0)
			fill!(w_f32, 0.5f0)
			_dot(u_f32, v_f32, w_f32)
			_inner_product(u_f32, w_f32, v_f32)

			# Backend constructor itself (the keyword method implicitly calls the inner one)
			backend(vector_type = Vector{ComplexF64}, matrix_type = Matrix{ComplexF64})

			# vector for Float32 backends
			vector(backend_dense32, 10)
			vector(backend_sparse32, 10)

			# matrix for Float32 backends
			matrix(backend_dense32, 5, 5)
			matrix(backend_sparse32, 5, 5)
		end

		# Test zero dimensions too
		vector(backend_default, 0)
		matrix(backend_default, 0, 5)
		matrix(backend_default, 5, 0)
		matrix(backend_default, 0, 0)

		# Test backend utility functions
		using Bramble: backend_types, vector_type, matrix_type, backend_eye, backend_zeros

		backend_types(backend_default)
		backend_types(typeof(backend_default))
		vector_type(backend_default)
		matrix_type(backend_default)
		eltype(backend_default)
		eltype(typeof(backend_default))

		# backend_eye and backend_zeros
		backend_eye(backend_default, 5)
		backend_zeros(backend_default, 5)

		# Test _serial_for! and _parallel_for!
		test_vec = Vector{Float64}(undef, 10)
		_serial_for!(test_vec, 1:10, i -> Float64(i^2))
		_parallel_for!(test_vec, 1:10, i -> Float64(i^2))
	end
end

# precompile bramblefunctions
@setup_workload begin
	# --- Define representative domains/sets ---
	# Use Float64 as the most common case for precompilation
	I_f64 = interval(-1.0, 1.0)
	X1 = I_f64 # Alias for clarity
	X2 = X1 × interval(0.0, 2.0)
	X3 = X2 × interval(-0.5, 0.5)

	# ESSENTIAL: 1D only
	test_sets = BRAMBLE_EXTENDED_PRECOMPILE ? (X1, X2, X3) : (X1,) # Sets for testing (1D, 2D, 3D)

	# Representative time interval
	I_time = interval(0.0, 1.0) # Float64 time interval

	# --- Define representative functions ---
	# Space-only functions
	f_space1d = x -> x + 1.0           # Expects T, returns T
	f_space2d = x -> x[1] * x[2]         # Expects NTuple{2,T}, returns T
	f_space3d = x -> x[1] + x[2] + x[3]    # Expects NTuple{3,T}, returns T
	test_space_funcs = (f_space1d, f_space2d, f_space3d)

	# Space-time functions (f(x,t))
	ft_spacetime1d = (x, t) -> (x + 1.0) * t
	ft_spacetime2d = (x, t) -> (x[1] * x[2]) * t
	ft_spacetime3d = (x, t) -> (x[1] + x[2] + x[3]) * t
	test_spacetime_funcs = (ft_spacetime1d, ft_spacetime2d, ft_spacetime3d)

	# Example points for calling embedded functions (using Float64)
	pt1d = 0.5
	pt2d = (0.5, 1.5)
	pt3d = (0.5, 1.5, 0.0)
	test_points = (pt1d, pt2d, pt3d)
	test_time = 0.5

	# Example points with a different type (e.g., Float32) for convert tests
	pt1d_f32 = Float32(pt1d)
	pt2d_f32 = Float32.(pt2d)
	pt3d_f32 = Float32.(pt3d)
	test_points_f32 = (pt1d_f32, pt2d_f32, pt3d_f32)

	@compile_workload begin
		# Loop through dimensions (1 only in essential mode, 1-3 in extended)
		max_dim = BRAMBLE_EXTENDED_PRECOMPILE ? 3 : 1
		for D in 1:max_dim
			X_set = test_sets[D]          # CartesianProduct for space
			f_space = test_space_funcs[D] # Space-only function
			ft_spacetime = test_spacetime_funcs[D] # Space-time function
			test_pt = test_points[D]      # Point for calling (Float64)
			test_pt_f32 = test_points_f32[D] # Point for calling (Float32)

			# === Test embed_function (function interface) ===
			bf_func_space = embed_function(X_set, f_space)
			bf_func_spacetime = embed_function(X_set, I_time, ft_spacetime)

			# === Test BrambleFunction Calls ===
			# Test space-only functions (hastime=false)
			if D == 1
				bf_func_space(test_pt)      # Call with Number (correct type)
				bf_func_space(pt1d_f32)     # Call with Number (needs convert)
			else
				bf_func_space(test_pt...)   # Call with splatted tuple NTuple{D, Float64}
				bf_func_space(test_pt)      # Call with tuple NTuple{D, Float64}
				bf_func_space(test_pt_f32)  # Call with tuple NTuple{D, Float32} (needs convert)
			end

			# Test space-time functions (hastime=true)
			# Calling with time 't' returns the inner space-only BrambleFunction
			inner_bf = bf_func_spacetime(test_time)
			# Now call the inner function with spatial points
			if D == 1
				inner_bf(test_pt)
				inner_bf(pt1d_f32)
			else
				inner_bf(test_pt...)
				inner_bf(test_pt)
				inner_bf(test_pt_f32)
			end
		end # loop D

		# Test has_time, argstype, codomaintype
		using Bramble: has_time, argstype, codomaintype, _get_args_type

		bf_notime = embed_function(X1, f_space1d)
		bf_withtime = embed_function(X1, I_time, ft_spacetime1d)

		has_time(bf_notime)
		has_time(typeof(bf_notime))
		has_time(bf_withtime)
		has_time(typeof(bf_withtime))

		argstype(bf_notime.wrapped)
		codomaintype(bf_notime.wrapped)

		_get_args_type(X1)
		_get_args_type(X2)
		_get_args_type(X3)

		# Test with BrambleFunction input (identity operation)
		embed_function(X1, bf_notime)
	end
end

