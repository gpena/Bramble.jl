using Bramble: interval, embed_function

@setup_workload begin
	# --- Basic Types and Intervals ---
	# Representative points including different types
	pts = [-1, -1.0, 0, 0.0, 1 // 2, Float32(π)]
	# Combinations for interval creation (ensuring start <= end)
	combinations = ((pts[i], pts[j]) for i in eachindex(pts) for j in eachindex(pts) if isless(pts[i], pts[j]) || pts[i] == pts[j])

	# Basic Float64 interval (most common case via interval())
	I_f64 = interval(-3.0, 10.0)

	# Basic Int intervals/products (using cartesian_product(NTuple))
	I_int = cartesian_product(((0, 1),))         # 1D Int
	R2_int = cartesian_product(((0, 1), (2, 3))) # 2D Int

	# Basic Float32 product
	R2_f32 = cartesian_product(((0.0f0, 1.0f0), (2.0f0, 3.0f0))) # 2D Float32

	# --- Higher Dimensional Products (Float64) ---
	R1_f64 = cartesian_product(I_f64) # Essentially I_f64 but via cartesian_product
	R2_f64 = R1_f64 × interval(0.0, 1.0)
	R3_f64 = R2_f64 × interval(2.0, 3.0)
	R4_f64 = R1_f64 × R3_f64 # Test product of non-1D products

	# Collection of representative products
	products = (I_f64, R1_f64, I_int, R2_int, R2_f32, R2_f64, R3_f64, R4_f64)

	@compile_workload begin
		# --- Constructors ---
		# interval(x, y) with various types (results in Float64 Product)
		for p in combinations
			interval(p...)
		end

		# interval(::CartesianProduct{1})
		interval(I_f64)
		interval(I_int) # Note: Creates Float64 product

		# cartesian_product(x, y) alias
		cartesian_product(-1.0, 1.0)

		# cartesian_product(::NTuple) (creates specific type)
		cartesian_product(((0, 1),))         # Int
		cartesian_product(((0.0, 1.0),))     # Float64
		cartesian_product(((0.0f0, 1.0f0),)) # Float32
		cartesian_product(((0, 1), (2, 3)))  # 2D Int
		cartesian_product(((0.0, 1.0), (2.0, 3.0))) # 2D Float64

		#box
		box(0, 1)         # Int
		box(0.0, 1.0)     # Float64
		box((0, 1), (2, 3))  # 2D Int
		box((0.0, 1.0), (2.0, 3.0)) # 2D Float64
		box((0, 1, 4), (2, 3, 5))  # 3D Int
		box((0.0, 1.0, 3.0), (2.0, 3.0, 4.0)) # 3D Float64

		# cartesian_product(::CartesianProduct) (identity)
		cartesian_product(I_f64)
		cartesian_product(R2_int)

		# --- Accessors & Properties ---
		for P in products
			T = eltype(P)
			D = dim(P)
			eltype(typeof(P))
			dim(typeof(P))

			# Call syntax P(i) for valid indices
			for i in 1:D
				P(i)
				tails(P, i)
			end

			# tails(P)
			tails(P)

			# projection(P, i)
			for i in 1:D
				proj = projection(P, i) # Creates CartesianProduct{1}

				first(proj)
				last(proj)
				tails(proj)
				proj(1)
			end

			# Specific 1D methods
			if D == 1
				first(P)
				last(P)
			end
		end

		# --- Operations ---
		# × operator (already created R2_f64, R3_f64, R4_f64 above)
		I_int × I_int

		# Add Float32 x Float32
		cartesian_product(((0.0f0, 1.0f0),)) × cartesian_product(((2.0f0, 3.0f0),))

		for i in 1:3
			Ii = ntuple(j -> I_f64, i) # Tuple of intervals
			Ω = reduce(×, Ii)
			projection(Ω, 1)
			tails(Ω)
			tails(Ω, 1)
		end
	end
end

@setup_workload begin
	I_f64 = interval(-1.0, 1.0)
	X1 = I_f64
	X2 = X1 × interval(0.0, 2.0)
	X3 = X2 × interval(-0.5, 0.5)
	test_sets = (X1, X2, X3)

	# --- Define helper functions needed ---
	f_1d = x -> x[1] - 0.0 <= 0
	f_2d = x -> x[1]^2 + x[2]^2 - 0.5 >= 0
	f_3d = x -> x[1] + x[2] + x[3] - 1.0 == 0
	test_funcs = (f_1d, f_2d, f_3d)
end

@compile_workload begin
	# === Boundary Symbols ===
	get_boundary_symbols(X1)
	get_boundary_symbols(X2)
	get_boundary_symbols(X3)

	# Iterate through representative sets (1D, 2D, 3D)
	for X_set in test_sets
		D = dim(X_set)
		T = eltype(X_set)
		f_for_dim = D <= length(test_funcs) ? test_funcs[D] : test_funcs[1] # Select function
		boundary_syms = get_boundary_symbols(X_set)
		sym1 = boundary_syms[1]
		sym_tuple = D > 1 ? (boundary_syms[1], boundary_syms[2]) : (boundary_syms[1],)

		process_identifier(X_set, f_for_dim)
		process_identifier(X_set, sym1)
		process_identifier(X_set, sym_tuple)

		try
			process_identifier(X_set, 123)
		catch
		end

		# === create_markers ===
		m_func = markers(X_set, :m_func => f_for_dim)
		m_sym = markers(X_set, :m_sym => sym1)
		m_tup = markers(X_set, :m_tup => sym_tuple)
		m_mixed = markers(X_set, :m_func => f_for_dim, :m_sym => sym1, :m_tup => sym_tuple)
		m_empty = markers(X_set)

		# === Domain Constructors ===
		# Default constructor
		d_def = domain(X_set)

		# Constructor with marker tuple
		d_tup = domain(X_set, m_mixed)

		# Constructor with varargs pairs
		d_vp = domain(X_set, :m_func => f_for_dim, :m_sym => sym1)

		# === Domain Accessors ===
		domains_to_test = (d_def, d_tup, d_vp)
		for dom in domains_to_test
			set(dom)

			dim(dom)
			eltype(dom)
			dim(typeof(dom))
			eltype(typeof(dom))
			if dim(dom) > 0
				projection(dom, 1)
			end

			# Test marker accessors
			mks = marker_identifiers(dom)
			sbs = marker_symbols(dom)
			tds = marker_tuples(dom)
			fds = marker_conditions(dom)

			# Iterate through generators to force compilation
			collect(mks)
			collect(sbs)
			collect(tds)
			collect(fds)
		end

		# === Show Methods ===
		# Create specific Marker instances
		marker_func = Marker(:a, embed_function(X_set, f_for_dim))
		marker_sym = Marker(:b, sym1)
		marker_tup = Marker(:c, sym_tuple)
	end
	@info "Domains: complete"
end
