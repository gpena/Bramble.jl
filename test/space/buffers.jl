using Bramble: vector_buffer, is_in_use, vector, lock!, unlock!, VectorBuffer, GridSpaceBuffer, create_simple_space_buffer, add_buffer!, nbuffers, get_vector_buffer, Backend

# --- Test Suite ---
@testset "Buffer Management Tests" begin

	# --- Test Setup ---
	test_backend = backend()
	BackendType = typeof(test_backend)
	test_vector_len = 15
	TestVecType = Vector{Float64} # Expected vector type

	@testset "VectorBuffer Tests" begin
		vb = vector_buffer(test_backend, test_vector_len)

		@test vb isa VectorBuffer{Float64,TestVecType}
		@test !is_in_use(vb)
		@test vector(vb) isa TestVecType
		@test length(vector(vb)) == test_vector_len

		lock!(vb)
		@test is_in_use(vb)

		unlock!(vb)
		@test !is_in_use(vb)
	end

	@testset "GridSpaceBuffer Tests" begin
		@testset "Creation" begin
			# Test creation with 0 initial buffers
			gsb0 = create_simple_space_buffer(test_backend, test_vector_len; nbuffers = 0)
			@test gsb0 isa GridSpaceBuffer{BackendType,TestVecType,Float64}
			@test gsb0.backend === test_backend
			@test gsb0.npts == test_vector_len
			@test nbuffers(gsb0) == 0
			@test isempty(gsb0.buffer)

			# Test creation with > 0 initial buffers
			num_initial = 3
			gsb3 = create_simple_space_buffer(test_backend, test_vector_len; nbuffers = num_initial)
			@test gsb3 isa GridSpaceBuffer{BackendType,TestVecType,Float64}
			@test nbuffers(gsb3) == num_initial
			@test length(gsb3.buffer) == num_initial
			for i in 1:num_initial
				@test haskey(gsb3.buffer, i)
				@test gsb3.buffer[i] isa VectorBuffer{Float64,TestVecType}
				@test !is_in_use(gsb3.buffer[i])
				@test length(vector(gsb3.buffer[i])) == test_vector_len
			end
		end

		@testset "add_buffer!" begin
			gsb = create_simple_space_buffer(test_backend, test_vector_len; nbuffers = 1)
			initial_count = nbuffers(gsb)

			returned_vec, returned_key = add_buffer!(gsb)

			@test nbuffers(gsb) == initial_count + 1
			@test returned_key == initial_count + 1
			@test returned_vec isa TestVecType
			@test length(returned_vec) == test_vector_len
			@test haskey(gsb.buffer, returned_key)
			@test gsb.buffer[returned_key].vector === returned_vec # Check it's the same object
			@test !is_in_use(gsb.buffer[returned_key]) # add_buffer! doesn't lock it
		end

		@testset "lock! and unlock!" begin
			gsb = create_simple_space_buffer(test_backend, test_vector_len; nbuffers = 2)
			key_to_test = 1
			internal_buffer = gsb.buffer[key_to_test]

			@test !is_in_use(internal_buffer) # Pre-condition

			returned_vec = lock!(gsb, key_to_test)

			@test returned_vec === vector(internal_buffer)
			@test is_in_use(internal_buffer)
			@test is_in_use(gsb.buffer[key_to_test]) # Verify via gsb access too

			unlock!(gsb, key_to_test)
			@test !is_in_use(internal_buffer)
			@test !is_in_use(gsb.buffer[key_to_test])
		end

		@testset "get_vector_buffer" begin
			# Case 1: Free buffer exists
			gsb_free = create_simple_space_buffer(test_backend, test_vector_len; nbuffers = 2)
			lock!(gsb_free, 1) # Lock the first one
			@test is_in_use(gsb_free.buffer[1])
			@test !is_in_use(gsb_free.buffer[2]) # Second one is free

			ret_vec1, ret_key1 = get_vector_buffer(gsb_free)
			@test ret_key1 == 2 # Should get the free one (key 2)
			@test ret_vec1 === vector(gsb_free.buffer[2])
			@test is_in_use(gsb_free.buffer[ret_key1]) # Should now be locked

			# Case 2: No free buffer exists (needs adding)
			gsb_full = create_simple_space_buffer(test_backend, test_vector_len; nbuffers = 1)
			lock!(gsb_full, 1) # Lock the only existing buffer
			initial_count = nbuffers(gsb_full)
			@test is_in_use(gsb_full.buffer[1])

			ret_vec2, ret_key2 = get_vector_buffer(gsb_full)
			@test nbuffers(gsb_full) == initial_count + 1 # New buffer added
			@test ret_key2 == initial_count + 1 # Key should be the new one
			@test haskey(gsb_full.buffer, ret_key2)
			@test ret_vec2 === vector(gsb_full.buffer[ret_key2])
			@test is_in_use(gsb_full.buffer[ret_key2]) # The new buffer should be locked

			# Case 3: Starts empty
			gsb_empty = create_simple_space_buffer(test_backend, test_vector_len; nbuffers = 0)
			@test nbuffers(gsb_empty) == 0

			ret_vec3, ret_key3 = get_vector_buffer(gsb_empty)
			@test nbuffers(gsb_empty) == 1
			@test ret_key3 == 1
			@test haskey(gsb_empty.buffer, 1)
			@test ret_vec3 === vector(gsb_empty.buffer[1])
			@test is_in_use(gsb_empty.buffer[1])
		end
	end
end