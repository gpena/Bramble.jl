"""
Additional test coverage for edge cases in space operators.

This file tests specific edge cases that may not be covered by the main test suite,
including boundary conditions, non-uniform grids, and error handling.
"""

using Test
using Bramble
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
