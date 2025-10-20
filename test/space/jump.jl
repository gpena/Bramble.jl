using Test
import Bramble: forward_difference, forward_jump, forward_jump_dim!
import Bramble: backward_difference, backward_jump, backward_jump_dim!
using LinearAlgebra: norm

# Backward difference operators
backward_jump_ops(::Val{1}) = (jump₋ₓ,)
backward_jump_ops(::Val{2}) = (jump₋ₓ, jump₋ᵧ, backward_jump_ops(Val(1))...)
backward_jump_ops(::Val{3}) = (jump₋ₓ, jump₋₂, backward_jump_ops(Val(2))...)

# Forward difference operators
forward_jump_ops(::Val{1}) = (jump₊ₓ,)
forward_jump_ops(::Val{2}) = (jump₊ₓ, jump₊ᵧ, forward_jump_ops(Val(1))...)
forward_jump_ops(::Val{3}) = (jump₊ₓ, jump₊₂, forward_jump_ops(Val(2))...)

@testset "Jump Operators" begin
	for D in 1:3
		@testset "$D-Dimensional Tests" begin
			dims, Wₕ, uₕ = setup_test_grid(Val(D))
			Ωₕ = mesh(Wₕ)

			vₕ = similar(uₕ)

			# Define a linear test function and project it onto the grid
			coeffs = (2.0, 3.0, 5.0)
			linear_func(x) = sum(coeffs[i] * x[i] for i in 1:D)
			Rₕ!(uₕ, linear_func)

			# Test forward jump operators (jump₊)
			@testset "Forward Jump (jump₊)" begin
				for i in 1:D
					# Test the primary applicator: forward_jump(u, Val(i))
					res_oop = forward_jump(uₕ, Val(i)) # Out-of-place

					# Test the in-place version
					forward_jump_dim!(vₕ.data, uₕ.data, dims, Val(i))
					@test norm(values(res_oop) - values(vₕ)) < 1e-14

					expected_val = forward_difference(uₕ, Val(i))
					res_reshaped = reshape(values(res_oop), dims)
					@test @views norm(res_oop .- expected_val) < 1e-14
				end

				# Test aliases (jump₊ₓ, jump₊ₕ)
				if D >= 1
					@test norm(jump₊ₓ(uₕ) - forward_jump(uₕ, Val(1))) < 1e-14
				end
				if D >= 2
					@test norm(jump₊ᵧ(uₕ) - forward_jump(uₕ, Val(2))) < 1e-14
				end
				if D >= 3
					@test norm(jump₊₂(uₕ) - forward_jump(uₕ, Val(3))) < 1e-14
				end

				# Test vectorial alias
				jumps = jump₊ₕ(uₕ)
				if D == 1
					@test jumps isa VectorElement
					@test norm(jumps - forward_jump(uₕ, Val(1))) < 1e-14
				else
					@test jumps isa NTuple{D,VectorElement}
					for i in 1:D
						@test norm(jumps[i] - forward_jump(uₕ, Val(i))) < 1e-14
					end
				end

				# Test backward jump operators (jump₋)
				@testset "Backward Jump (jump₋)" begin
					for i in 1:D
						# Test primary applicator: backward_jump(u, Val(i))
						res_oop = backward_jump(uₕ, Val(i)) # Out-of-place

						# Test in-place version
						backward_jump_dim!(vₕ.data, uₕ.data, dims, Val(i))
						@test norm(values(res_oop) - values(vₕ)) < 1e-14

						expected_val = backward_difference(uₕ, Val(i))
						@test @views norm(res_oop .- expected_val) < 1e-14
					end

					# Test aliases (jump₋ₓ, jump₋ₕ)
					if D >= 1
						@test norm(jump₋ₓ(uₕ) - backward_jump(uₕ, Val(1))) < 1e-14
					end
					if D >= 2
						@test norm(jump₋ᵧ(uₕ) - backward_jump(uₕ, Val(2))) < 1e-14
					end
					if D >= 3
						@test norm(jump₋₂(uₕ) - backward_jump(uₕ, Val(3))) < 1e-14
					end

					# Test vectorial alias
					jumps = jump₋ₕ(uₕ)
					if D == 1
						@test jumps isa VectorElement
						@test norm(jumps - backward_jump(uₕ, Val(1))) < 1e-14
					else
						@test jumps isa NTuple{D,VectorElement}
						for i in 1:D
							@test norm(jumps[i] - backward_jump(uₕ, Val(i))) < 1e-14
						end
					end
				end
			end
		end
	end

	@testset "Operator vs. Matrix Application" begin
		@testset "Backward" test_operator_matrix_equivalence(backward_jump_ops)
		@testset "Forward" test_operator_matrix_equivalence(forward_jump_ops)
	end
end
