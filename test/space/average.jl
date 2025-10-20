using Test
import Bramble: forward_average, backward_average
import Bramble: forward_average_dim!, backward_average_dim!
using LinearAlgebra: norm

@testset "Averaging Operators" begin
	# Backward average operators
	backward_average_ops(::Val{1}) = (M₋ₓ,)
	backward_average_ops(::Val{2}) = (M₋ₓ, M₋ᵧ)
	backward_average_ops(::Val{3}) = (M₋ₓ, M₋ᵧ, M₋₂)

	# Forward average operators
	forward_average_ops(::Val{1}) = (M₊ₓ,)
	forward_average_ops(::Val{2}) = (M₊ₓ, M₊ᵧ)
	forward_average_ops(::Val{3}) = (M₊ₓ, M₊ᵧ, M₊₂)

	get_coord(pts, I::CartesianIndex{D}) where D = ntuple(d -> pts[d][I[d]], length(I))
	get_coord(pts, I::CartesianIndex{1}) = pts[I[1]]
	coeffs = (2.0, 3.0, 5.0)

	for D in 1:3
		@testset "$D-Dimensional Tests" begin
			dims, Wₕ, uₕ = setup_test_grid(Val(D))
			Ωₕ = mesh(Wₕ)
			pts = points(Ωₕ)
			vₕ = similar(uₕ)
			coords = Base.Fix1(get_coord, pts)

			# Define a linear test function and project it onto the grid
			linear_func(x) = sum(coeffs[i] * x[i] for i in 1:D)
			Rₕ!(uₕ, linear_func)

			@testset "Forward Average (M₊)" begin
				for i in 1:D
					# --- Calculate the analytical expected result ---
					expected_vals = similar(uₕ.data)
					li = LinearIndices(dims)
					step_cartesian = CartesianIndex(ntuple(d -> d == i ? 1 : 0, D))

					for I in CartesianIndices(dims)
						# Interior points: f((x_i + x_{i+1})/2)
						if I[i] < dims[i]
							midpoint = (coords(I) .+ coords(I + step_cartesian)) ./ 2
							expected_vals[li[I]] = linear_func(midpoint)
							# Boundary point: f(x_N)/2
						else
							expected_vals[li[I]] = linear_func(coords(I)) / 2
						end
					end

					# Test the primary out-of-place applicator
					res_oop = forward_average(uₕ, Val(i))
					@test norm(res_oop.data - expected_vals) < 1e-12

					# Test the in-place version against the out-of-place one
					forward_average_dim!(vₕ.data, uₕ.data, dims, Val(i))
					@test norm(res_oop.data - vₕ.data) < 1e-12
				end

				# --- Test aliases ---
				if D >= 1
					@test norm(M₊ₓ(uₕ) - forward_average(uₕ, Val(1))) < 1e-12;
				end
				if D >= 2
					@test norm(M₊ᵧ(uₕ) - forward_average(uₕ, Val(2))) < 1e-12;
				end
				if D >= 3
					@test norm(M₊₂(uₕ) - forward_average(uₕ, Val(3))) < 1e-12;
				end

				# --- Test vectorial alias ---
				averages = M₊ₕ(uₕ)
				if D == 1
					@test averages isa VectorElement
					@test norm(averages - forward_average(uₕ, Val(1))) < 1e-12
				else
					@test averages isa NTuple{D,VectorElement}
					for i in 1:D
						@test norm(averages[i] - forward_average(uₕ, Val(i))) < 1e-12
					end
				end
			end

			@testset "Backward Average (M₋)" begin
				for i in 1:D
					# --- Calculate the analytical expected result ---
					expected_vals = similar(uₕ.data)
					li = LinearIndices(dims)
					step_cartesian = CartesianIndex(ntuple(d -> d == i ? 1 : 0, D))

					for I in CartesianIndices(dims)
						# Interior points: f((x_i + x_{i-1})/2)
						if I[i] > 1
							midpoint = (coords(I) .+ coords(I - step_cartesian)) ./ 2
							expected_vals[li[I]] = linear_func(midpoint)
							# Boundary point: f(x_1)/2
						else
							expected_vals[li[I]] = linear_func(coords(I)) / 2
						end
					end

					# (The rest of the tests in this block remain the same)
					# Test the primary out-of-place applicator
					res_oop = backward_average(uₕ, Val(i))
					@test norm(res_oop.data - expected_vals) < 1e-12

					# Test the in-place version against the out-of-place one
					backward_average_dim!(vₕ.data, uₕ.data, dims, Val(i))
					@test norm(res_oop.data - vₕ.data) < 1e-12
				end

				# --- Test aliases ---
				if D >= 1
					@test norm(M₋ₓ(uₕ) - backward_average(uₕ, Val(1))) < 1e-12;
				end
				if D >= 2
					@test norm(M₋ᵧ(uₕ) - backward_average(uₕ, Val(2))) < 1e-12;
				end
				if D >= 3
					@test norm(M₋₂(uₕ) - backward_average(uₕ, Val(3))) < 1e-12;
				end

				# --- Test vectorial alias ---
				averages = M₋ₕ(uₕ)
				if D == 1
					@test averages isa VectorElement
					@test norm(averages - backward_average(uₕ, Val(1))) < 1e-12
				else
					@test averages isa NTuple{D,VectorElement}
					for i in 1:D
						@test norm(averages[i] - backward_average(uₕ, Val(i))) < 1e-12
					end
				end
			end
		end
	end

	@testset "Operator vs. Matrix Application" begin
		@testset "Forward" test_operator_matrix_equivalence(forward_average_ops)
		@testset "Backward" test_operator_matrix_equivalence(backward_average_ops)
	end
end
