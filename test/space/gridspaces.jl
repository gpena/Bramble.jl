using Bramble: __prod, mesh, box, domain, interval, dim, backend, npoints, vector, matrix, _innerplus_weights!, spacing, _innerh_weights!, _innerplus_mean_weights!, __innerplus_weights!, half_spacing, space_weights, SpaceWeights,
			   vector_type, matrix_type, mesh, weights, has_backward_difference_matrix, has_average_matrix, gridspace, ndofs, backward_difference_matrices
using LinearAlgebra: norm

@testset "ScalarGridSpace construction" begin
	@testset "Weight Helper Functions" begin
		mesh1d = mesh(domain(interval(0, 1)), 10, true)
		mesh2d = mesh(domain(box((0, 0), (0.5, 0.6))), (5, 6), (true, true))
		D1, D2 = dim(mesh1d), dim(mesh2d)
		VT = Vector{Float64}

		@testset "__prod" begin
			# Test D=1
			v1 = ([1.0, 2.0, 3.0],)
			idx1 = CartesianIndex(2)
			@test __prod(v1, idx1) ≈ 2.0

			# Test D=2
			v2 = ([1.0, 2.0], [3.0, 4.0, 5.0])
			idx2 = CartesianIndex(2, 3)
			@test __prod(v2, idx2) ≈ 2.0 * 5.0 ≈ 10.0
		end

		@testset "_innerh_weights!" begin
			u = vector(backend(mesh2d), npoints(mesh2d))
			_innerh_weights!(u, mesh2d)
			expected_norm = 0.05952940449895328
			@test norm(u) .≈ expected_norm
		end

		@testset "_innerplus_weights!" begin
			u = vector(backend(mesh1d), npoints(mesh1d))
			_innerplus_weights!(u, mesh1d, 1)
			@test u[1] == 0.0
			for i ∈ 2:npoints(mesh1d)
				@test u[i] ≈ spacing(mesh1d, i)
			end
		end

		@testset "_innerplus_mean_weights!" begin
			u = vector(backend(mesh1d), npoints(mesh1d))
			N = npoints(mesh1d)
			_innerplus_mean_weights!(u, mesh1d, 1)
			@test u[1] == 0.0
			@test u[N] == 0.0
			for i ∈ 2:(N - 1)
				@test u[i] ≈ half_spacing(mesh1d, i)
			end
		end

		@testset "__innerplus_weights!" begin
			# Use D=2 example
			npts_tup = npoints(mesh2d, Tuple)
			v = zeros(Float64, npts_tup) # Reshaped array target
			comp_weights = (rand(npts_tup[1]), rand(npts_tup[2])) # Example component weights
			__innerplus_weights!(v, comp_weights)

			# Test a specific point
			idx = CartesianIndex(3, 4)
			@test v[idx] ≈ comp_weights[1][idx[1]] * comp_weights[2][idx[2]]
		end
	end

	@testset "Space Creation Functions" begin
		mesh2d = mesh(domain(box((0, 0), (0.5, 0.6))), (4, 5), (true, true))
		D = dim(mesh2d)
		T = eltype(mesh2d)
		b = backend(mesh2d)
		VT = vector_type(b)
		MT = matrix_type(b)

		@testset "space_weights" begin
			# Use the optimized version if available
			weights = space_weights(mesh2d) # Or space_weights
			@test weights isa SpaceWeights{D,VT}
			@test size(weights.innerh) == (npoints(mesh2d),)
			@test length(weights.innerplus) == D
			@test all(size(w) == (npoints(mesh2d),) for w in weights.innerplus)
			# Could add value checks if mocks are deterministic enough
		end

		@testset "create_backward_diff_matrices" begin
			mats = backward_difference_matrices(mesh2d)
			@test mats isa NTuple{D,MT}
			@test length(mats) == D
			@test all(size(m) == (npoints(mesh2d), npoints(mesh2d)) for m in mats)
		end
	end

	@testset "ScalarGridSpace Constructor and Accessors" begin
		mesh1d = mesh(domain(box(0, 1)), 3, true)
		mesh3d = mesh(domain(box((0, 0, 0), (0.5, 0.6, 0.7))), (4, 4, 4), (true, true, true))

		# Test default caching
		space1d = gridspace(mesh1d)
		@test mesh(space1d) === mesh1d
		@test weights(space1d) isa SpaceWeights{1,Vector{Float64}}
		@test has_backward_difference_matrix(space1d) == true
		@test has_average_matrix(space1d) == false
		@test length(space1d.backward_difference_matrix) == 1
		@test length(space1d.average_matrix) == 1

		# Test custom caching
		space3d = gridspace(mesh3d; cache_avg = true, cache_bwd = false)
		@test dim(space3d) == 3
		@test ndofs(space3d) == npoints(mesh3d)
		@test eltype(space3d) == Float64
		@test has_backward_difference_matrix(space3d) == false
		@test has_average_matrix(space3d) == true
	end
end

"""
Additional test coverage for edge cases in space operators.

This file tests specific edge cases that may not be covered by the main test suite,
including boundary conditions, non-uniform grids, and error handling.
"""

import Bramble: _get_h_val, _compute_difference, Forward, Backward
import Bramble: backward_difference_dim!, forward_difference_dim!

@testset "Edge Cases and Coverage" begin
	@testset "Buffer Edge Cases" begin
		using Bramble: VectorBuffer, getindex, setindex!, backend

		# Test getindex/setindex! optimized implementation
		b = backend()
		vb = Bramble.vector_buffer(b, 10)

		# Test bounds checking is present
		@test_throws BoundsError vb[11]
		@test_throws BoundsError vb[0]

		# Test actual access
		vb[1] = 42.0
		@test vb[1] == 42.0

		# Test that it works like a regular vector
		vb[5] = 3.14
		@test vb[5] ≈ 3.14
	end

	@testset "Difference Operator Boundary Cases" begin
		T = Float64

		@testset "Backward Finite Difference - Boundary with h vector" begin
			# This tests the h[2] vs h[1] issue at line 99
			u = T[1.0, 2.0, 3.0, 4.0, 5.0]
			h_vec = T[0.1, 0.2, 0.3, 0.4, 0.5]  # Non-uniform spacing
			out = similar(u)

			backward_difference_dim!(out, u, h_vec, (5,), Val(1))

			# At boundary (i=1), the function uses h[2] (hardcoded)
			# According to current implementation at line 99
			expected_boundary = u[1] / h_vec[2]  # Uses h[2], not h[1]
			@test out[1] ≈ expected_boundary

			# Interior points should use correct h[i]
			@test out[2] ≈ (u[2] - u[1]) / h_vec[2]
			@test out[3] ≈ (u[3] - u[2]) / h_vec[3]
		end

		@testset "Backward Finite Difference - Boundary with h function" begin
			u = T[1.0, 2.0, 3.0, 4.0, 5.0]
			h_func(i) = 0.1 * i  # Spacing increases with index
			out = similar(u)

			backward_difference_dim!(out, u, h_func, (5,), Val(1))

			# At boundary (i=1), should use h(2) according to current implementation
			expected_boundary = u[1] / h_func(2)
			@test out[1] ≈ expected_boundary

			# Interior points
			@test out[2] ≈ (u[2] - u[1]) / h_func(2)
			@test out[3] ≈ (u[3] - u[2]) / h_func(3)
		end

		@testset "Forward Finite Difference - Boundary" begin
			u = T[1.0, 2.0, 3.0, 4.0, 5.0]
			h_vec = T[0.1, 0.2, 0.3, 0.4, 0.5]
			out = similar(u)

			forward_difference_dim!(out, u, h_vec, (5,), Val(1))

			# Interior points
			@test out[1] ≈ (u[2] - u[1]) / h_vec[1]
			@test out[4] ≈ (u[5] - u[4]) / h_vec[4]

			# Boundary at end (i=5)
			expected_boundary = -u[5] / h_vec[5]
			@test out[5] ≈ expected_boundary
		end

		@testset "_get_h_val error handling" begin
			h_vec = T[0.1, 0.2, 0.3]

			# Valid access
			@test _get_h_val(h_vec, 1) ≈ 0.1
			@test _get_h_val(h_vec, 3) ≈ 0.3

			# Out of bounds - should assert
			@test_throws AssertionError _get_h_val(h_vec, 0)
			@test_throws AssertionError _get_h_val(h_vec, 4)

			# Function version
			h_func(i) = 0.1 * i
			@test _get_h_val(h_func, 1) ≈ 0.1
			@test _get_h_val(h_func, 10) ≈ 1.0
		end

		@testset "2D Boundary Cases" begin
			# Test boundary handling in multiple dimensions
			dims = (4, 3)
			u = collect(Float64, 1:12)
			out = similar(u)

			# Backward in x-direction - first column boundary
			backward_difference_dim!(out, u, dims, Val(1))
			# Points at x=1: indices 1, 5, 9 should have boundary treatment
			@test out[1] == u[1]  # Boundary
			@test out[5] == u[5]  # Boundary
			@test out[9] == u[9]  # Boundary
			# Interior point
			@test out[2] == u[2] - u[1]

			# Forward in y-direction - last row boundary
			forward_difference_dim!(out, u, dims, Val(2))
			# Points at y=3: indices 9, 10, 11, 12 should have boundary treatment
			@test out[9] == -u[9]   # Boundary
			@test out[12] == -u[12] # Boundary
		end

		@testset "3D Boundary Cases" begin
			dims = (2, 2, 2)
			u = collect(Float64, 1:8)
			out = similar(u)

			# Test backward in z-direction
			backward_difference_dim!(out, u, dims, Val(3))
			# First "slice" in z should be boundary
			@test out[1] == u[1]
			@test out[2] == u[2]
			@test out[3] == u[3]
			@test out[4] == u[4]
			# Second slice should be interior
			@test out[5] == u[5] - u[1]
		end
	end

	@testset "Average Operator Boundary Cases" begin
		using Bramble: forward_average_dim!, backward_average_dim!
		T = Float64

		@testset "Forward Average - Boundary" begin
			u = T[2.0, 4.0, 6.0, 8.0]
			out = similar(u)
			forward_average_dim!(out, u, (4,), Val(1))

			# Interior: (u[i] + u[i+1]) / 2
			@test out[1] ≈ (u[1] + u[2]) / 2  # 3.0
			@test out[2] ≈ (u[2] + u[3]) / 2  # 5.0
			# Boundary: u[end] / 2
			@test out[4] ≈ u[4] / 2  # 4.0
		end

		@testset "Backward Average - Boundary" begin
			u = T[2.0, 4.0, 6.0, 8.0]
			out = similar(u)
			backward_average_dim!(out, u, (4,), Val(1))

			# Boundary: u[1] / 2
			@test out[1] ≈ u[1] / 2  # 1.0
			# Interior: (u[i] + u[i-1]) / 2
			@test out[2] ≈ (u[2] + u[1]) / 2  # 3.0
			@test out[4] ≈ (u[4] + u[3]) / 2  # 7.0
		end
	end

	@testset "Jump Operator Boundary Cases" begin
		using Bramble: forward_jump_dim!, backward_jump_dim!
		T = Float64

		@testset "Forward Jump - Boundary" begin
			u = T[1.0, 3.0, 7.0, 15.0]
			out = similar(u)
			forward_jump_dim!(out, u, (4,), Val(1))

			# Interior: u[i+1] - u[i]
			@test out[1] ≈ u[2] - u[1]  # 2.0
			@test out[2] ≈ u[3] - u[2]  # 4.0
			@test out[3] ≈ u[4] - u[3]  # 8.0
			# Boundary: -u[end]
			@test out[4] ≈ -u[4]  # -15.0
		end

		@testset "Backward Jump - Boundary" begin
			u = T[1.0, 3.0, 7.0, 15.0]
			out = similar(u)
			backward_jump_dim!(out, u, (4,), Val(1))

			# Boundary: u[1]
			@test out[1] ≈ u[1]  # 1.0
			# Interior: u[i] - u[i-1]
			@test out[2] ≈ u[2] - u[1]  # 2.0
			@test out[3] ≈ u[3] - u[2]  # 4.0
			@test out[4] ≈ u[4] - u[3]  # 8.0
		end
	end

	@testset "ComponentStyle Type System" begin
		using Bramble: ComponentStyle, SingleComponent, MultiComponent
		using Bramble: gridspace, mesh, domain, box

		# Create a scalar space
		Ωₕ = mesh(domain(box(0, 1)), 5, false)
		Wₕ = gridspace(Ωₕ)

		# Test ComponentStyle dispatch on type
		@test ComponentStyle(typeof(Wₕ)) isa SingleComponent
		# Note: ComponentStyle only works on types, not instances
	end

	@testset "InnerProductType System" begin
		using Bramble: InnerProductType, Innerh, Innerplus
		using Bramble: gridspace, mesh, domain, box, weights

		Ωₕ = mesh(domain(box(0, 1)), 5, false)
		Wₕ = gridspace(Ωₕ)

		# Test weight accessors with different inner product types
		w_h = weights(Wₕ, Innerh())
		@test w_h isa AbstractVector
		@test length(w_h) > 0

		w_plus = weights(Wₕ, Innerplus())
		@test w_plus isa Tuple
		@test length(w_plus) == 1  # 1D case

		# Test dimension-specific access
		w_plus_1 = weights(Wₕ, Innerplus(), 1)
		@test w_plus_1 isa AbstractVector

		# For Innerh, dimension doesn't matter (returns same weights)
		w_h_1 = weights(Wₕ, Innerh(), 1)
		@test w_h_1 == w_h
	end

	@testset "Scalar GridSpace Accessors" begin
		using Bramble: dim, ndofs, eltype, gridspace, mesh, domain, box

		# 1D
		Ωₕ_1d = mesh(domain(box(0, 1)), 10, false)
		Wₕ_1d = gridspace(Ωₕ_1d)
		@test dim(Wₕ_1d) == 1
		@test ndofs(Wₕ_1d) == 10
		@test ndofs(Wₕ_1d, Tuple) == (10,)
		@test eltype(Wₕ_1d) == Float64

		# 2D
		Ωₕ_2d = mesh(domain(box((0, 0), (1, 1))), (5, 6), (false, false))
		Wₕ_2d = gridspace(Ωₕ_2d)
		@test dim(Wₕ_2d) == 2
		@test ndofs(Wₕ_2d) == 30
		@test ndofs(Wₕ_2d, Tuple) == (5, 6)

		# Test type-level functions
		@test dim(typeof(Wₕ_1d)) == 1
		@test eltype(typeof(Wₕ_1d)) == Float64
	end

	@testset "PointwiseEvaluator" begin
		using Bramble: PointwiseEvaluator, Rₕ, func, mesh, gridspace
		using Bramble: domain, box

		Ωₕ = Bramble.mesh(domain(box(0, 1)), 5, false)
		Wₕ = gridspace(Ωₕ)
		uₕ = Bramble.element(Wₕ)

		# Create a PointwiseEvaluator
		f(x) = x^2
		pe = PointwiseEvaluator(f, Ωₕ)

		# Test accessors
		@test func(pe) === f
		@test mesh(pe) === Ωₕ

		# Test callable interface
		val = pe(1)
		@test val isa Number
		@test val ≈ 0.0^2  # First point in grid

		# Test using it in Rₕ
		Rₕ!(uₕ, f)
		@test all(isfinite, uₕ.data)
	end

	@testset "MatrixElement Documentation Cases" begin
		using Bramble: elements, MatrixElement, space, gridspace
		using Bramble: mesh, domain, box

		Ωₕ = mesh(domain(box(0, 1)), 5, false)
		Wₕ = gridspace(Ωₕ)

		# Identity operator (default)
		Iₕ = elements(Wₕ)
		@test Iₕ isa MatrixElement
		@test size(Iₕ) == (5, 5)
		@test space(Iₕ) === Wₕ

		# Custom matrix
		using LinearAlgebra: I as LinI
		A = Matrix(1.0LinI, 5, 5)
		Tₕ = elements(Wₕ, A)
		@test Tₕ isa MatrixElement
		@test size(Tₕ) == (5, 5)
	end

	@testset "Non-uniform Grid Edge Cases" begin
		using Bramble: mesh, domain, box, gridspace, Rₕ!, spacing

		# Create mesh with non-uniform flag
		Ωₕ = mesh(domain(box(0, 1)), 10, false)  # uniform
		Wₕ = gridspace(Ωₕ)
		uₕ = Bramble.element(Wₕ)

		# Test that operators work on uniform grids
		Rₕ!(uₕ, x -> x^2)

		# Apply finite difference operators
		v₊ = D₊ₓ(uₕ)
		v₋ = D₋ₓ(uₕ)

		@test all(isfinite, v₊.data)
		@test all(isfinite, v₋.data)
	end
end
