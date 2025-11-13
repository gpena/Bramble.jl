"""
Test coverage for mesh module

Focus on edge cases, marker combinations, and complex interactions
"""

import Bramble: set, markers, CartesianProduct, Mesh1D, MeshnD

@testset "Geometry Modules Coverage" begin
	@testset "Domain Edge Cases" begin
		@testset "1D Domain variations" begin
			I = interval(-1.0, 2.0)

			# Domain without markers
			X1 = domain(I)
			@test set(X1) isa CartesianProduct

			# Domain with single marker
			X2 = domain(I, markers(I, :left => x -> x[1] < -0.5))
			@test !isnothing(markers(X2))

			# Domain with multiple markers
			X3 = domain(I, markers(I,
								   :left => x -> x[1] < -0.5,
								   :right => x -> x[1] > 1.5,
								   :center => x -> -0.5 ≤ x[1] ≤ 1.5))
			@test !isnothing(markers(X3))
		end

		@testset "2D Domain variations" begin
			I = interval(0.0, 1.0)
			Ω = I × I

			# Domain without markers
			X1 = domain(Ω)
			@test !isnothing(X1)

			# Domain with boundary markers
			X2 = domain(Ω, markers(Ω,
								   :bottom => x -> x[2] < 0.01,
								   :top => x -> x[2] > 0.99,
								   :left => x -> x[1] < 0.01,
								   :right => x -> x[1] > 0.99))
			@test !isnothing(X2)

			# Domain with interior markers
			X3 = domain(Ω, markers(Ω,
								   :interior => x -> 0.25 < x[1] < 0.75 && 0.25 < x[2] < 0.75))
			@test !isnothing(X3)
		end

		@testset "3D Domain" begin
			I = interval(0.0, 1.0)
			Ω = I × I × I

			X = domain(Ω, markers(Ω,
								  :boundary => x -> any(x .< 0.01) || any(x .> 0.99)))
			@test !isnothing(X)
			@test set(X) isa CartesianProduct
		end
	end

	@testset "Marker Combinations" begin
		I = interval(0.0, 1.0)

		@testset "Disjoint markers" begin
			X = domain(I, markers(I,
								  :region1 => x -> x[1] < 0.33,
								  :region2 => x -> 0.33 ≤ x[1] < 0.67,
								  :region3 => x -> x[1] ≥ 0.67))

			Mh = mesh(X, 10, false)
			@test Mh isa Mesh1D
		end

		@testset "Overlapping markers" begin
			X = domain(I, markers(I,
								  :left_half => x -> x[1] ≤ 0.6,
								  :right_half => x -> x[1] ≥ 0.4,
								  :center => x -> 0.3 ≤ x[1] ≤ 0.7))

			Mh = mesh(X, 10, false)
			@test Mh isa Mesh1D
		end

		@testset "Nested markers" begin
			Ω = I × I
			X = domain(Ω, markers(Ω,
								  :outer => x -> all(0.1 .≤ x .≤ 0.9),
								  :middle => x -> all(0.3 .≤ x .≤ 0.7),
								  :inner => x -> all(0.4 .≤ x .≤ 0.6)))

			Mh = mesh(X, (5, 5), (false, false))
			@test Mh isa MeshnD
		end
	end

	@testset "Marker Evaluation" begin
		I = interval(0.0, 1.0)

		@testset "Boolean marker functions" begin
			# Simple threshold
			m1 = markers(I, :boundary => x -> x[1] < 0.1 || x[1] > 0.9)
			@test !isnothing(m1)

			# Complex logical expression
			m2 = markers(I, :region => x -> (x[1] > 0.2 && x[1] < 0.4) || (x[1] > 0.6 && x[1] < 0.8))
			@test !isnothing(m2)
		end

		@testset "Marker with different predicates" begin
			Ω = I × I

			# Distance-based marker
			center = (0.5, 0.5)
			m1 = markers(Ω, :circle => x -> sqrt((x[1] - center[1])^2 + (x[2] - center[2])^2) < 0.3)
			@test !isnothing(m1)

			# Box marker
			m2 = markers(Ω, :box => x -> all(0.2 .≤ x .≤ 0.8))
			@test !isnothing(m2)

			# Annulus marker
			m3 = markers(Ω, :annulus => x -> begin
							 r = sqrt((x[1] - center[1])^2 + (x[2] - center[2])^2)
							 0.2 < r < 0.4
						 end)
			@test !isnothing(m3)
		end
	end

	@testset "Set Operations" begin
		@testset "1D Sets" begin
			I1 = interval(0.0, 1.0)
			I2 = interval(-1.0, 0.5)
			I3 = interval(0.5, 2.0)

			@test I1 isa CartesianProduct
			@test I2 isa CartesianProduct
			@test I3 isa CartesianProduct
		end

		@testset "2D Sets (Cartesian Products)" begin
			I = interval(0.0, 1.0)
			J = interval(-1.0, 1.0)

			Ω1 = I × I  # Square
			@test Ω1 isa CartesianProduct

			Ω2 = I × J  # Rectangle
			@test Ω2 isa CartesianProduct

			Ω3 = J × I  # Different rectangle
			@test Ω3 isa CartesianProduct
		end

		@testset "3D Sets" begin
			I = interval(0.0, 1.0)
			J = interval(-0.5, 0.5)
			K = interval(0.0, 2.0)

			Ω1 = I × I × I  # Cube
			@test Ω1 isa CartesianProduct

			Ω2 = I × J × K  # Box
			@test Ω2 isa CartesianProduct
		end
	end

	@testset "Domain with Mesh Integration" begin
		@testset "1D Domain to Mesh" begin
			I = interval(0.0, π)
			X = domain(I, markers(I,
								  :left => x -> x[1] < 0.1,
								  :right => x -> x[1] > π - 0.1))

			Mh = mesh(X, 20, false)
			@test Mh isa Mesh1D

			Wh = gridspace(Mh)
			@test !isnothing(Wh)
		end

		@testset "2D Domain to Mesh" begin
			I = interval(0.0, 1.0)
			Ω = I × I
			X = domain(Ω, markers(Ω,
								  :boundary => x -> any(x .< 0.01) || any(x .> 0.99)))

			Mh = mesh(X, (6, 6), (false, false))
			@test Mh isa MeshnD

			Wh = gridspace(Mh)
			@test !isnothing(Wh)
		end

		@testset "3D Domain to Mesh (small)" begin
			I = interval(0.0, 1.0)
			Ω = I × I × I
			X = domain(Ω)

			Mh = mesh(X, (3, 3, 3), (false, false, false))
			@test Mh isa MeshnD

			Wh = gridspace(Mh)
			@test !isnothing(Wh)
		end
	end

	@testset "Marker Access and Queries" begin
		I = interval(0.0, 1.0)
		Ω = I × I

		X = domain(Ω, markers(Ω,
							  :left => x -> x[1] < 0.01,
							  :right => x -> x[1] > 0.99,
							  :bottom => x -> x[2] < 0.01,
							  :top => x -> x[2] > 0.99))

		m = markers(X)
		@test !isnothing(m)

		# Test that markers can be used with dirichlet_constraints
		bcs = dirichlet_constraints(set(X),
									:left => x -> 0.0,
									:right => x -> 1.0)
		@test !isnothing(bcs)
	end

	@testset "Empty and Trivial Cases" begin
		@testset "Domain without markers" begin
			I = interval(0.0, 1.0)
			X = domain(I)

			@test !isnothing(X)
			# Should work with mesh even without markers
			Mh = mesh(X, 5, false)
			@test Mh isa Mesh1D
		end

		@testset "Marker that includes everything" begin
			I = interval(0.0, 1.0)
			X = domain(I, markers(I, :all => x -> true))

			@test !isnothing(X)
			Mh = mesh(X, 5, false)
			@test Mh isa Mesh1D
		end

		@testset "Marker that includes nothing" begin
			I = interval(0.0, 1.0)
			X = domain(I, markers(I, :none => x -> false))

			@test !isnothing(X)
			Mh = mesh(X, 5, false)
			@test Mh isa Mesh1D
		end
	end
end
