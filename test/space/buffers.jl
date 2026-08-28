using Bramble: vector_buffer, in_use, vector, lock!, unlock!, VectorBuffer, GridSpaceBuffer,
			   simple_space_buffer, add_buffer!, nbuffers, with_buffer, Backend

@testset "Buffer Management Tests" begin
	test_backend = backend()
	BackendType = typeof(test_backend)
	test_vector_len = 15
	TestVecType = Vector{Float64}

	@testset "VectorBuffer Tests" begin
		vb = vector_buffer(test_backend, test_vector_len)

		@test vb isa VectorBuffer{Float64,TestVecType}
		@test !in_use(vb)
		@test vector(vb) isa TestVecType
		@test length(vector(vb)) == test_vector_len

		lock!(vb)
		@test in_use(vb)

		unlock!(vb)
		@test !in_use(vb)
	end

	@testset "GridSpaceBuffer Tests" begin
		@testset "Creation" begin
			gsb0 = simple_space_buffer(test_backend, test_vector_len; nbuffers = 0)
			@test gsb0 isa GridSpaceBuffer{BackendType,TestVecType,Float64}
			@test gsb0.backend === test_backend
			@test gsb0.npts == test_vector_len
			@test nbuffers(gsb0) == 0
			@test isempty(gsb0.buffer)

			num_initial = 3
			gsb3 = simple_space_buffer(test_backend, test_vector_len; nbuffers = num_initial)
			@test gsb3 isa GridSpaceBuffer{BackendType,TestVecType,Float64}
			@test nbuffers(gsb3) == num_initial
			@test length(gsb3.buffer) == num_initial
			for i in 1:num_initial
				@test checkbounds(Bool, gsb3.buffer, i)
				@test gsb3.buffer[i] isa VectorBuffer{Float64,TestVecType}
				@test !in_use(gsb3.buffer[i])
				@test length(vector(gsb3.buffer[i])) == test_vector_len
			end

			@test_throws ArgumentError simple_space_buffer(test_backend, test_vector_len; nbuffers = -1)
			@test_throws ArgumentError simple_space_buffer(test_backend, -1)
		end

		@testset "add_buffer!" begin
			gsb = simple_space_buffer(test_backend, test_vector_len; nbuffers = 1)
			initial_count = nbuffers(gsb)

			returned_vec, returned_key = add_buffer!(gsb)

			@test nbuffers(gsb) == initial_count + 1
			@test returned_key == initial_count + 1
			@test returned_vec isa TestVecType
			@test length(returned_vec) == test_vector_len
			@test checkbounds(Bool, gsb.buffer, returned_key)
			@test gsb.buffer[returned_key].vector === returned_vec
			@test !in_use(gsb.buffer[returned_key])
		end

		@testset "lock! and unlock!" begin
			gsb = simple_space_buffer(test_backend, test_vector_len; nbuffers = 2)
			key_to_test = 1
			internal_buffer = gsb.buffer[key_to_test]

			@test !in_use(internal_buffer)

			returned_vec = lock!(gsb, key_to_test)

			@test returned_vec === vector(internal_buffer)
			@test in_use(internal_buffer)
			@test in_use(gsb.buffer[key_to_test])

			unlock!(gsb, key_to_test)
			@test !in_use(internal_buffer)
			@test !in_use(gsb.buffer[key_to_test])
		end

		@testset "vector_buffer" begin
			# Case 1: a free buffer exists
			gsb_free = simple_space_buffer(test_backend, test_vector_len; nbuffers = 2)
			lock!(gsb_free, 1)
			@test in_use(gsb_free.buffer[1])
			@test !in_use(gsb_free.buffer[2])

			ret_vec1, ret_key1 = vector_buffer(gsb_free)
			@test ret_key1 == 2
			@test ret_vec1 === vector(gsb_free.buffer[2])
			@test in_use(gsb_free.buffer[ret_key1])

			# Case 2: everything is in use, so the pool grows
			gsb_full = simple_space_buffer(test_backend, test_vector_len; nbuffers = 1)
			lock!(gsb_full, 1)
			initial_count = nbuffers(gsb_full)

			ret_vec2, ret_key2 = vector_buffer(gsb_full)
			@test nbuffers(gsb_full) == initial_count + 1
			@test ret_key2 == initial_count + 1
			@test ret_vec2 === vector(gsb_full.buffer[ret_key2])
			@test in_use(gsb_full.buffer[ret_key2])

			# Case 3: starts empty
			gsb_empty = simple_space_buffer(test_backend, test_vector_len; nbuffers = 0)
			@test nbuffers(gsb_empty) == 0

			ret_vec3, ret_key3 = vector_buffer(gsb_empty)
			@test nbuffers(gsb_empty) == 1
			@test ret_key3 == 1
			@test ret_vec3 === vector(gsb_empty.buffer[1])
			@test in_use(gsb_empty.buffer[1])
		end

		@testset "Buffers are reused rather than reallocated" begin
			gsb = simple_space_buffer(test_backend, test_vector_len; nbuffers = 1)

			seen = Set{UInt}()
			for _ in 1:200
				v, key = vector_buffer(gsb)
				push!(seen, objectid(v))
				unlock!(gsb, key)
			end

			# One buffer, acquired and released serially, must be the same vector
			# every time and the pool must never have grown.
			@test length(seen) == 1
			@test nbuffers(gsb) == 1
		end

		@testset "Double release is a no-op" begin
			gsb = simple_space_buffer(test_backend, test_vector_len; nbuffers = 1)
			v, key = vector_buffer(gsb)

			unlock!(gsb, key)
			unlock!(gsb, key)   # must not push the key onto the free stack twice
			@test !in_use(gsb.buffer[key])

			# Two consecutive acquisitions must hand out *different* buffers.
			v1, k1 = vector_buffer(gsb)
			v2, k2 = vector_buffer(gsb)
			@test k1 != k2
			@test v1 !== v2
		end

		@testset "Nested acquisitions do not alias" begin
			gsb = simple_space_buffer(test_backend, test_vector_len; nbuffers = 0)
			a, ka = vector_buffer(gsb)
			b, kb = vector_buffer(gsb)
			c, kc = vector_buffer(gsb)

			@test length(unique(objectid.((a, b, c)))) == 3
			@test length(unique((ka, kb, kc))) == 3
			@test nbuffers(gsb) == 3

			unlock!(gsb, ka)
			unlock!(gsb, kb)
			unlock!(gsb, kc)
			@test all(!in_use, gsb.buffer)
		end

		@testset "with_buffer" begin
			gsb = simple_space_buffer(test_backend, test_vector_len; nbuffers = 1)

			result = with_buffer(gsb) do v
				@test v isa TestVecType
				@test length(v) == test_vector_len
				@test any(in_use, gsb.buffer)
				fill!(v, 2.0)
				sum(v)
			end
			@test result == 2.0 * test_vector_len
			@test all(!in_use, gsb.buffer)   # released on normal exit

			# The buffer must also be released when the body throws.
			@test_throws ErrorException with_buffer(gsb) do v
				error("boom")
			end
			@test all(!in_use, gsb.buffer)
			@test nbuffers(gsb) == 1         # and no buffer was leaked

			# Nested blocks get distinct vectors.
			with_buffer(gsb) do outer
				with_buffer(gsb) do inner
					@test outer !== inner
				end
			end
			@test all(!in_use, gsb.buffer)
		end

		@testset "Acquisition is O(1) in pool size" begin
			# With a linear scan this grows linearly; with the free stack it is flat.
			function per_acquire(n)
				pool = simple_space_buffer(test_backend, 8; nbuffers = 0)
				for _ in 1:n
					vector_buffer(pool)
				end
				v, k = vector_buffer(pool)
				unlock!(pool, k)                       # warm
				t = @elapsed for _ in 1:2000
					v, k = vector_buffer(pool)
					unlock!(pool, k)
				end
				return t / 2000
			end

			small = per_acquire(25)
			large = per_acquire(800)
			# 32x the pool must not cost anywhere near 32x per acquisition.
			@test large < 8 * small
		end
	end
end
