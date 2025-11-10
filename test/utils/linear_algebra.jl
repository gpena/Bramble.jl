using Bramble: _dot, _inner_product, _parallel_for!, _serial_for!
using LinearAlgebra: Diagonal

@testset "Linear Algebra Utilities" begin
	@testset "_dot function" begin
		# Test basic dot product
		u = [1.0, 2.0, 3.0]
		v = [4.0, 5.0, 6.0]
		w = [2.0, 2.0, 2.0]

		result = _dot(u, v, w)
		expected = (1.0 * 4.0 * 2.0) + (2.0 * 5.0 * 2.0) + (3.0 * 6.0 * 2.0)
		@test result ≈ expected
		@test result ≈ 56.0

		# Test with zeros
		u_zero = [0.0, 0.0, 0.0]
		@test _dot(u_zero, v, w) ≈ 0.0
		@test _dot(u, u_zero, w) ≈ 0.0
		@test _dot(u, v, u_zero) ≈ 0.0

		# Test with ones
		ones_vec = [1.0, 1.0, 1.0, 1.0]
		@test _dot(ones_vec, ones_vec, ones_vec) ≈ 4.0

		# Test Float32
		u_f32 = Float32[1.0, 2.0, 3.0]
		v_f32 = Float32[4.0, 5.0, 6.0]
		w_f32 = Float32[2.0, 2.0, 2.0]
		result_f32 = _dot(u_f32, v_f32, w_f32)
		@test result_f32 isa Float32
		@test result_f32 ≈ 56.0f0

		# Test single element
		@test _dot([2.0], [3.0], [4.0]) ≈ 24.0

		# Test larger vectors
		n = 100
		u_large = collect(1.0:n)
		v_large = ones(n)
		w_large = fill(2.0, n)
		result_large = _dot(u_large, v_large, w_large)
		expected_large = 2.0 * sum(1:n)
		@test result_large ≈ expected_large
	end

	@testset "_inner_product vector version" begin
		# Test weighted inner product for vectors
		u = [1.0, 2.0, 3.0]
		h = [0.5, 1.0, 1.5]
		v = [4.0, 5.0, 6.0]

		# Using the specialized vector version
		result = _inner_product(u, h, v)
		# Should compute ∑ᵢ uᵢ * hᵢ * vᵢ
		expected = (1.0 * 0.5 * 4.0) + (2.0 * 1.0 * 5.0) + (3.0 * 1.5 * 6.0)
		@test result ≈ expected
		@test result ≈ 2.0 + 10.0 + 27.0
		@test result ≈ 39.0

		# Test symmetry (not exactly symmetric due to h in middle)
		result2 = _inner_product(v, h, u)
		@test result2 ≈ expected

		# Test with uniform weights
		h_ones = ones(3)
		result_uniform = _inner_product(u, h_ones, v)
		@test result_uniform ≈ dot(u, v)

		# Test Float32
		u_f32 = Float32[1.0, 2.0, 3.0]
		h_f32 = Float32[0.5, 1.0, 1.5]
		v_f32 = Float32[4.0, 5.0, 6.0]
		result_f32 = _inner_product(u_f32, h_f32, v_f32)
		@test result_f32 isa Float32
		@test result_f32 ≈ 39.0f0
	end

	@testset "_inner_product generic version" begin
		# Test with matrices (falls back to generic implementation)
		U = [1.0 2.0; 3.0 4.0]
		H = [0.5, 1.0]
		V = [5.0 6.0; 7.0 8.0]

		result = _inner_product(U, H, V)
		expected = transpose(V) * (Diagonal(H) * U)
		@test result ≈ expected

		# Test with single column matrices (behaves like vectors but uses generic path)
		u_mat = reshape([1.0, 2.0, 3.0], 3, 1)
		h_vec = [0.5, 1.0, 1.5]
		v_mat = reshape([4.0, 5.0, 6.0], 3, 1)

		result_mat = _inner_product(u_mat, h_vec, v_mat)
		expected_mat = transpose(v_mat) * (Diagonal(h_vec) * u_mat)
		@test result_mat ≈ expected_mat
	end

	@testset "_serial_for!" begin
		n = 10
		v = zeros(n)
		idxs = 1:n

		# Test simple assignment
		f = i -> Float64(i^2)
		_serial_for!(v, idxs, f)
		@test v == [Float64(i^2) for i in 1:n]

		# Test with partial indices
		v2 = ones(n)
		idxs_partial = 3:7
		f2 = i -> Float64(i * 10)
		_serial_for!(v2, idxs_partial, f2)
		@test v2[1:2] == [1.0, 1.0]
		@test v2[3:7] == [30.0, 40.0, 50.0, 60.0, 70.0]
		@test v2[8:10] == [1.0, 1.0, 1.0]

		# Test with CartesianIndices (2D array)
		A = zeros(3, 4)
		cart_idxs = CartesianIndices(A)
		f3 = idx -> Float64(idx[1] + idx[2])
		_serial_for!(A, cart_idxs, f3)
		for i in 1:3, j in 1:4
			@test A[i, j] ≈ Float64(i + j)
		end
	end

	@testset "_parallel_for!" begin
		n = 100
		v = zeros(n)
		idxs = 1:n

		# Test parallel assignment
		f = i -> Float64(i^2)
		_parallel_for!(v, idxs, f)
		@test v == [Float64(i^2) for i in 1:n]

		# Test with partial indices
		v2 = ones(n)
		idxs_partial = 10:50
		f2 = i -> Float64(i * 2)
		_parallel_for!(v2, idxs_partial, f2)
		@test v2[1:9] == ones(9)
		@test v2[10:50] == [Float64(i * 2) for i in 10:50]
		@test v2[51:100] == ones(50)

		# Test with CartesianIndices
		B = zeros(10, 10)
		cart_idxs = CartesianIndices(B)
		f3 = idx -> Float64(idx[1] * idx[2])
		_parallel_for!(B, cart_idxs, f3)
		for i in 1:10, j in 1:10
			@test B[i, j] ≈ Float64(i * j)
		end

		# Verify parallel gives same result as serial
		v_serial = zeros(n)
		v_parallel = zeros(n)
		f_test = i -> sin(Float64(i)) + cos(Float64(i))
		_serial_for!(v_serial, idxs, f_test)
		_parallel_for!(v_parallel, idxs, f_test)
		@test v_serial ≈ v_parallel
	end

	@testset "Performance: serial vs parallel" begin
		# Note: This is more of a smoke test than a performance test
		# Just verify both methods work and produce same results
		n = 1000
		v_serial = zeros(n)
		v_parallel = zeros(n)
		idxs = 1:n
		f = i -> sqrt(Float64(i)) + log(Float64(i) + 1)

		_serial_for!(v_serial, idxs, f)
		_parallel_for!(v_parallel, idxs, f)

		@test v_serial ≈ v_parallel
	end
end
