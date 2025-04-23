using SparseArrays: SparseVector
using Bramble: Backend, matrix, vector
using Bramble: interval, @embed, embed_function

@setup_workload begin
	# Define common backend configurations to precompile
	backend_default = Backend() # Vector{Float64}, SparseMatrixCSC{Float64,Int}
	backend_dense64 = Backend(vector_type = Vector{Float64}, matrix_type = Matrix{Float64})
	backend_sparse64 = Backend(vector_type = SparseVector{Float64,Int}, matrix_type = SparseMatrixCSC{Float64,Int})
	backend_dense32 = Backend(vector_type = Vector{Float32}, matrix_type = Matrix{Float32})
	backend_sparse32 = Backend(vector_type = SparseVector{Float32,Int32}, matrix_type = SparseMatrixCSC{Float32,Int32}) # Note Int32 for Sparse*
end

@compile_workload begin
	# Backend constructor itself (the keyword method implicitly calls the inner one)
	Backend(vector_type = Vector{ComplexF64}, matrix_type = Matrix{ComplexF64})

	# vector for various backends and a typical size
	vector(backend_default, 10)
	vector(backend_dense64, 10)
	vector(backend_sparse64, 10)
	vector(backend_dense32, 10)
	vector(backend_sparse32, 10)

	# matrix for various backends and typical sizes
	matrix(backend_default, 5, 5)
	matrix(backend_dense64, 5, 5)
	matrix(backend_sparse64, 5, 5)
	matrix(backend_dense32, 5, 5)
	matrix(backend_sparse32, 5, 5)

	# Test zero dimensions too
	vector(backend_default, 0)
	matrix(backend_default, 0, 5)
	matrix(backend_default, 5, 0)
	matrix(backend_default, 0, 0)

	@info "Backend: complete" # Specific message
end

# precompile bramblefunctions
@setup_workload begin
	# --- Define representative domains/sets ---
	# Use Float64 as the most common case for precompilation
	I_f64 = interval(-1.0, 1.0)
	X1 = I_f64 # Alias for clarity
	X2 = X1 × interval(0.0, 2.0)
	X3 = X2 × interval(-0.5, 0.5)
	test_sets = (X1, X2, X3) # Sets for testing (1D, 2D, 3D)

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
end # @setup_workload

@compile_workload begin
	# Loop through dimensions 1, 2, 3
	for D in 1:3
		X_set = test_sets[D]          # CartesianProduct for space
		f_space = test_space_funcs[D] # Space-only function
		ft_spacetime = test_spacetime_funcs[D] # Space-time function
		test_pt = test_points[D]      # Point for calling (Float64)
		test_pt_f32 = test_points_f32[D] # Point for calling (Float32)

		# === Test @embed macro ===
		# Use function symbols as arguments to the macro
		bf_macro_space = @embed X_set f_space
		bf_macro_spacetime = @embed X_set×I_time ft_spacetime
		# Using lambdas directly is also possible but less critical for precompilation
		# if the function call path (`_embed_notime`/`_embed_withtime`) is covered.

		# === Test embed_function (function interface) ===
		bf_func_space = embed_function(X_set, f_space)
		bf_func_spacetime = embed_function(X_set, I_time, ft_spacetime)

		# === Test BrambleFunction Calls ===
		# Test space-only functions (hastime=false)
		for bf_s in (bf_macro_space, bf_func_space)
			if D == 1
				bf_s(test_pt)      # Call with Number (correct type)
				bf_s(pt1d_f32)     # Call with Number (needs convert)
			else
				bf_s(test_pt...)   # Call with splatted tuple NTuple{D, Float64}
				bf_s(test_pt)      # Call with tuple NTuple{D, Float64}
				bf_s(test_pt_f32)  # Call with tuple NTuple{D, Float32} (needs convert)
			end
		end

		# Test space-time functions (hastime=true)
		for bf_st in (bf_macro_spacetime, bf_func_spacetime)
			# Calling with time 't' returns the inner space-only BrambleFunction
			inner_bf = bf_st(test_time)
			# Now call the inner function with spatial points
			if D == 1
				inner_bf(test_pt)
				inner_bf(pt1d_f32)
			else
				inner_bf(test_pt...)
				inner_bf(test_pt)
				inner_bf(test_pt_f32)
			end
		end
	end # loop D

	@info "BrambleFunction and embeddings: complete"
end # @compile_workload
