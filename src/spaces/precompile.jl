# --- Precompilation Workload ---
@compile_workload begin
	precompile_backend = Backend()
	precompile_vector_len = 10
	precompile_num_buffers = 2

	# --- VectorBuffer Operations ---
	vb = create_vector_buffer(precompile_backend, precompile_vector_len)
	is_in_use(vb)
	v = vector(vb)
	lock!(vb)
	unlock!(vb)

	# --- GridSpaceBuffer Operations ---
	gsb = create_simple_space_buffer(precompile_backend, precompile_vector_len; nbuffers = precompile_num_buffers)
	nbuffers(gsb)

	# Ensure at least one buffer exists for lock/unlock/get
	if nbuffers(gsb) == 0
		add_buffer!(gsb)
	end

	local key1::Int # Ensure type stability
	local vec1 # Type will be inferred from get_vector_buffer
	vec1, key1 = get_vector_buffer(gsb) # This locks the buffer

	# Need another buffer to test lock/unlock on a specific key if get_vector_buffer added one
	if nbuffers(gsb) < 2
		add_buffer!(gsb)
	end
	key_to_test = 1 == key1 ? 2 : 1 # Pick a key different from the one locked by get_vector_buffer

	# Lock/Unlock a specific buffer (that wasn't just locked by get_vector_buffer)
	lock!(gsb, key_to_test)
	unlock!(gsb, key_to_test)

	# Unlock the one obtained from get_vector_buffer
	unlock!(gsb, key1)

	# Add another buffer
	add_buffer!(gsb)

	@info "GridSpace buffers: complete"
end

# precompile gridspace basics
@setup_workload begin
	DIMS = (1, 2, 3)
	T = Float64
	I = interval(0, 1)

	@compile_workload begin
		for D in DIMS
			S = reduce(×, ntuple(i -> I, D))
			X = domain(S)
			dims_tuple = ntuple(i -> 3, D)
			unif_tuple = ntuple(i -> true, D)
			test_mesh = mesh(X, dims_tuple, unif_tuple)
			test_space = gridspace(test_mesh, cache_backward_diff_matrices = true, cache_average_matrices = true)
			w = weights(test_space)
			mesh(test_space)
			eltype(test_space)
			dim(test_space)
			ndofs(test_space)
			weights_innerh(w)
			weights_innerplus(w, 1)
			has_backward_diff_matrix(test_space)
			backward_diff_matrix(test_space, 1)
			has_average_matrix(test_space)
			average_matrix(test_space, 1)
		end
	end
	@info "GridSpace basics: complete"

	for D in DIMS
		S = reduce(×, ntuple(i -> I, D))
		X = domain(S)
		dims_tuple = ntuple(i -> 3, D)
		unif_tuple = ntuple(i -> true, D)
		test_mesh = mesh(X, dims_tuple, unif_tuple)
		W = gridspace(test_mesh)
		v_data = collect(Float64, 1:3)
		v_data2 = fill(2.0, 3)
		α = 3.0
		β = 2

		# Constructors
		u1 = element(W)
		u2 = element(W, α)

		v_data = deepcopy(values(u2))
		u3 = element(W, v_data)
		u4 = element(W, β) # Int constructor

		# Getters/Setters
		s = space(u3)
		vals = values(u3)
		values!(u1, vals)

		# Forwarded methods (exercise them)
		len = length(u3)
		et = eltype(u3)
		sz = size(u3)
		fi = firstindex(u3)
		li = lastindex(u3)
		iter_sum = sum(u3) # Test iteration
		msh = mesh(u3)

		# ndims
		nd = ndims(VectorElement)

		# Indexing
		val_at_1 = u3[1]
		u3[2] = 99.0
		u3[3] = 99 # Set Int

		# Similar
		s1 = similar(u3)

		# Copyto!
		z = element(W)
		copyto!(z, u3)
		copyto!(z, v_data)
		copyto!(z, α)
		copyto!(z, β)

		# Arithmetic Operators
		r1 = α + u3
		r2 = u3 + α
		r3 = u3 + u2
		r4 = α * u3
		r5 = u3 * α
		r6 = u3 * u2
		r7 = u3 - u2
		r8 = u3 - α
		r9 = α - u3
		r10 = u3 / β
		r11 = β / u3
		r12 = u3 / u2
		r13 = u3^β
		# r14 = β ^ u3 # Might cause issues depending on interpretation
		r15 = u3^u2

		# Broadcasting
		w = element(W)
		# copyto! broadcast
		copyto!(w, Base.broadcasted(identity, u3))
		# materialize! / fused broadcast
		w .= u3 .+ u2 .* α
		w .= β # Scalar assignment
		w .= u3 ./ β

		# Other common operations (example)
		nrm = norm(u3)
	end
	@info "VectorElement constructor and operations: complete"
end
