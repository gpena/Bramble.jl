import Bramble: MatrixElement, elements, space, eltype, ⊗, _Eye, shift, npoints, spacing
using Bramble: backward_difference_dim!, forward_difference_dim!
import SparseArrays: issparse, sprand, spdiagm, spzeros

# Backward difference operators
backward_ops(::Val{1}) = (diff₋ₓ, D₋ₓ)
backward_ops(::Val{2}) = (D₋ᵧ, diff₋ₓ, diff₋ᵧ, backward_ops(Val(1))...)
backward_ops(::Val{3}) = (D₋₂, diff₋ₓ, diff₋₂, backward_ops(Val(2))...)

# Forward difference operators
forward_ops(::Val{1}) = (diff₊ₓ, D₊ₓ)
forward_ops(::Val{2}) = (D₊ᵧ, diff₊ₓ, diff₊ᵧ, forward_ops(Val(1))...)
forward_ops(::Val{3}) = (D₊₂, diff₊ₓ, diff₊₂, forward_ops(Val(2))...)

# Compares operator application to explicit matrix-vector multiplication
function test_operator_matrix_equivalence(op_generator)
	for D in 1:3
		@testset "$(D)D" begin
			_, W, U = setup_test_grid(Val(D))
			Rₕ!(U, x -> exp(-sum(x)))

			u₁ₕ = similar(U.data)
			u₂ₕ = similar(u₁ₕ)

			for op in unique(op_generator(Val(D)))
				u₁ₕ .= op(U).data
				u₂ₕ .= op(W) * U.data
				@test u₁ₕ ≈ u₂ₕ
			end
		end
	end
end

@testset "Finite Difference Operators" begin
	import LinearAlgebra: Diagonal, UniformScaling
	import LinearAlgebra: I as identity_matrix

	# --- Common Setup for All Tests ---
	mesh1D = mesh(domain(box(0, 1)), 5, false)
	mesh2D = mesh(domain(box((0, 1), (2, 3))), (5, 4), (true, true))
	mesh3D = mesh(domain(box((0, 1, 2), (4, 5, 6))), (4, 5, 4), (true, true, true))
	T = Float64

	@testset "Helper Operators" begin
		A = [1 2; 3 4]
		B = [5 6; 7 8]
		@test (A ⊗ B) == kron(A, B)

		MatrixType = typeof(SparseArrays.spzeros(T, 5, 5))
		@test _Eye(MatrixType, 5, Val(0)) * ones(5) == ones(5)

		S_super = _Eye(MatrixType, 5, Val(1))
		S_sub   = _Eye(MatrixType, 5, Val(-2))

		@test S_super == spdiagm(1 => ones(4))
		@test S_sub == spdiagm(-2 => ones(3))
		@test S_super * [1, 2, 3, 4, 5] == [2, 3, 4, 5, 0]
		@test S_sub * [1, 2, 3, 4, 5] == [0, 0, 1, 2, 3]
	end

	@testset "Shift Operators" begin
		for val in [-1, 1]
			name = val == 1 ? "Forward" : "Backward"
			@testset "$name Shifts" begin
				# 1D
				n = npoints(mesh1D)
				@test shift(mesh1D, Val(1), Val(val)) == spdiagm(val => ones(n - abs(val)))

				# 2D
				nx, ny = npoints(mesh2D, Tuple)
				Sₓ_expected = identity_matrix(ny) ⊗ spdiagm(val => ones(nx - abs(val)))
				Sᵧ_expected = spdiagm(val => ones(ny - abs(val))) ⊗ identity_matrix(nx)
				@test shift(mesh2D, Val(1), Val(val)) == Sₓ_expected
				@test shift(mesh2D, Val(2), Val(val)) == Sᵧ_expected

				# 3D
				nx, ny, nz = npoints(mesh3D, Tuple)
				Sₓ_3D_expected = identity_matrix(ny*nz) ⊗ spdiagm(val => ones(nx - abs(val)))
				Sᵧ_3D_expected = identity_matrix(nz) ⊗ spdiagm(val => ones(ny - abs(val))) ⊗ identity_matrix(nx)
				S₂_3D_expected = spdiagm(val => ones(nz - abs(val))) ⊗ identity_matrix(nx*ny)
				@test shift(mesh3D, Val(1), Val(val)) == Sₓ_3D_expected
				@test shift(mesh3D, Val(2), Val(val)) == Sᵧ_3D_expected
				@test shift(mesh3D, Val(3), Val(val)) == S₂_3D_expected
			end
		end
	end

	@testset "Backward Difference" begin
		@testset "In-place Calculation" begin
			# 1D
			u_1d = T[1, 2, 4, 8, 16]
			out_1d = similar(u_1d)
			backward_difference_dim!(out_1d, u_1d, (5,), Val(1))
			@test out_1d == [1, 1, 2, 4, 8]

			# 2D
			u_2d = T[1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35]
			out_2d = similar(u_2d)
			backward_difference_dim!(out_2d, u_2d, (5, 4), Val(1))
			@test out_2d == T[1, 1, 1, 1, 1, 11, 1, 1, 1, 1, 21, 1, 1, 1, 1, 31, 1, 1, 1, 1]
			backward_difference_dim!(out_2d, u_2d, (5, 4), Val(2))
			@test out_2d == T[1, 2, 3, 4, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

			# 3D
			u_3d = collect(Iterators.flatten(T[i+j+k for i ∈ 1:4, j ∈ 1:5, k ∈ 1:4]))
			out_3d = similar(u_3d)
			backward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(1))
			@test out_3d ==
				  T[3, 1, 1, 1, 4, 1, 1, 1, 5, 1, 1, 1, 6, 1, 1, 1, 7, 1, 1, 1, 4, 1, 1, 1, 5, 1, 1, 1, 6, 1, 1, 1, 7, 1, 1, 1, 8, 1, 1, 1, 5, 1, 1, 1, 6, 1, 1, 1, 7, 1, 1, 1, 8, 1, 1, 1, 9, 1, 1, 1, 6, 1, 1, 1, 7, 1, 1, 1, 8, 1, 1, 1, 9, 1, 1, 1, 10, 1,
					1, 1]
			backward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(2))
			@test out_3d ==
				  T[3, 4, 5, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 5, 6, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 6, 7, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 7, 8, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
					1]
			backward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(3))
			@test out_3d ==
				  T[3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
					1, 1]
		end

		@testset "In-place Finite Difference" begin
			u = T[2, 3, 5, 9, 8]
			h = Base.Fix1(spacing, mesh1D)
			out = similar(u)
			backward_difference_dim!(out, u, h, (5,), Val(1))
			expected = [u[1]/h(2), (u[2]-u[1])/h(2), (u[3]-u[2])/h(3), (u[4]-u[3])/h(4), (u[5]-u[4])/h(5)]
			@test out ≈ expected
		end
	end

	@testset "Forward Difference" begin
		@testset "In-place Calculation" begin
			# 1D
			u_1d = T[1, 2, 4, 8, 16]
			out_1d = similar(u_1d)
			forward_difference_dim!(out_1d, u_1d, (5,), Val(1))
			@test out_1d == [1, 2, 4, 8, -16]

			# 2D
			u_2d = T[1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35]
			out_2d = similar(u_2d)
			forward_difference_dim!(out_2d, u_2d, (5, 4), Val(1))
			@test out_2d == T[1, 1, 1, 1, -5, 1, 1, 1, 1, -15, 1, 1, 1, 1, -25, 1, 1, 1, 1, -35]
			forward_difference_dim!(out_2d, u_2d, (5, 4), Val(2))
			@test out_2d == T[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, -31, -32, -33, -34, -35]

			# 3D
			u_3d = collect(Iterators.flatten(T[i+j+k for i ∈ 1:4, j ∈ 1:5, k ∈ 1:4]))
			out_3d = similar(u_3d)
			forward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(1))
			@test out_3d ==
				  T[1, 1, 1, -6, 1, 1, 1, -7, 1, 1, 1, -8, 1, 1, 1, -9, 1, 1, 1, -10, 1, 1, 1, -7, 1, 1, 1, -8, 1, 1, 1, -9, 1, 1, 1, -10, 1, 1, 1, -11, 1, 1, 1, -8, 1, 1, 1, -9, 1, 1, 1, -10, 1, 1, 1, -11, 1, 1, 1, -12, 1, 1, 1, -9, 1, 1, 1, -10, 1, 1, 1,
					-11, 1, 1, 1, -12, 1, 1, 1, -13]
			forward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(2))
			@test out_3d ==
				  T[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -7, -8, -9, -10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -8, -9, -10, -11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -9, -10, -11, -12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
					1, 1, 1, -10, -11, -12, -13]
			forward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(3))
			@test out_3d ==
				  T[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -6, -7, -8, -9, -7, -8, -9, -10, -8, -9, -10, -11, -9,
					-10, -11, -12, -10, -11, -12, -13]
		end

		@testset "In-place Finite Difference" begin
			u = T[2, 3, 5, 9, 8]
			h = Base.Fix1(spacing, mesh1D)
			out = similar(u)
			N = length(u)
			forward_difference_dim!(out, u, h, (N,), Val(1))
			expected = [(u[2]-u[1])/h(1), (u[3]-u[2])/h(2), (u[4]-u[3])/h(3), (u[5]-u[4])/h(4), -u[5]/h(5)]
			@test out ≈ expected
		end
	end

	@testset "Operator vs. Matrix Application" begin
		@testset "Backward" test_operator_matrix_equivalence(backward_ops)
		@testset "Forward" test_operator_matrix_equivalence(forward_ops)
	end
end