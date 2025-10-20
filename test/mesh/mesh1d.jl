import Bramble: indices, change_points!, npoints, dim, spacing, half_spacings, generate_indices, boundary_symbol_to_dict, markers, backend, set_indices!, points, set_points!, set_markers!, point, half_point,
				half_spacing, iterative_refinement!
import Bramble: cell_measure, hₘₐₓ, half_points, boundary_indices, interior_indices
import Bramble: DomainMarkers, Mesh1D, Backend, points_iterator, half_points_iterator, spacings_iterator, cell_measures_iterator, half_spacings_iterator
import Base: diff

@testset "Mesh1D Tests" begin
	function create_test_domain(a = 0.0, b = 1.0; markers = nothing)
		I = interval(a, b)

		if markers isa Nothing
			return domain(I)
		else
			return domain(I, markers)
		end
	end

	@testset "Helper Functions" begin
		@testset "generate_indices" begin
			@test generate_indices(5) == CartesianIndices((5,))
			@test generate_indices(1) == CartesianIndices((1,))
		end

		@testset "boundary_symbol_to_dict (D=1)" begin
			indices = CartesianIndices((10,))
			dict = boundary_symbol_to_dict(indices)
			@test dict[:left] == CartesianIndex(1)
			@test dict[:right] == CartesianIndex(10)

			indices_single = CartesianIndices((1,))
			dict_single = boundary_symbol_to_dict(indices_single)
			@test dict_single[:left] == CartesianIndex(1)
			@test dict_single[:right] == CartesianIndex(1)
		end
	end

	@testset "Mesh Construction and Basic Properties" begin
		Ω = create_test_domain(0.0, 2.0)
		npts = 5
		Ωₕ_unif = mesh(Ω, npts, true; backend = backend())
		Ωₕ_nonunif = mesh(Ω, npts, false; backend = backend())

		@testset "Uniform Mesh" begin
			@test Ωₕ_unif isa Mesh1D
			@test backend(Ωₕ_unif) isa Backend
			@test eltype(Ωₕ_unif) == Float64
			@test dim(Ωₕ_unif) == 1
			@test dim(typeof(Ωₕ_unif)) == 1
			@test npoints(Ωₕ_unif) == npts
			@test npoints(Ωₕ_unif, Tuple) == (npts,)
			@test indices(Ωₕ_unif) == CartesianIndices((npts,))
			@test length(points(Ωₕ_unif)) == npts
			@test points(Ωₕ_unif) ≈ [0.0, 0.5, 1.0, 1.5, 2.0]
			@test point(Ωₕ_unif, 3) ≈ 1.0
			@test point(Ωₕ_unif, CartesianIndex(3)) ≈ 1.0
			@test collect(points_iterator(Ωₕ_unif)) ≈ [0.0, 0.5, 1.0, 1.5, 2.0]
		end

		@testset "Non-Uniform Mesh" begin
			@test Ωₕ_nonunif isa Mesh1D
			@test backend(Ωₕ_nonunif) isa Backend
			@test eltype(Ωₕ_nonunif) == Float64
			@test dim(Ωₕ_nonunif) == 1
			@test npoints(Ωₕ_nonunif) == npts
			@test npoints(Ωₕ_nonunif, Tuple) == (npts,)
			@test indices(Ωₕ_nonunif) == CartesianIndices((npts,))
			pts_nonunif = points(Ωₕ_nonunif)
			@test length(pts_nonunif) == npts
			@test pts_nonunif[1] ≈ 0.0
			@test pts_nonunif[end] ≈ 2.0
			@test all(diff(pts_nonunif) .> 0) # Check sorted
			# Cannot test exact values, but check bounds and sorting
			@test all(pts_nonunif .>= 0.0) && all(pts_nonunif .<= 2.0)
			@test point(Ωₕ_nonunif, 1) ≈ 0.0
			@test point(Ωₕ_nonunif, npts) ≈ 2.0
			@test collect(points_iterator(Ωₕ_nonunif)) ≈ pts_nonunif
		end

		@testset "set_points! and set_indices!" begin
			Ω = create_test_domain(0.0, 1.0)
			Ωₕ = mesh(Ω, 3, true; backend = backend()) # [0.0, 0.5, 1.0]

			new_pts = [0.0, 0.3, 0.7, 1.0]
			new_indices = CartesianIndices((4,))

			set_points!(Ωₕ, new_pts)
			set_indices!(Ωₕ, new_indices) # Need to update indices if npts changes via set_points!

			@test points(Ωₕ) === new_pts # Check identity for mutable struct
			@test npoints(Ωₕ) == 4
			@test indices(Ωₕ) == new_indices
		end
	end

	@testset "Geometric Properties" begin
		npts = 5
		Ω_unif = create_test_domain(0.0, 4.0) # Step = 1.0
		Ωₕ_unif = mesh(Ω_unif, npts, true; backend = backend()) # Pts: 0, 1, 2, 3, 4

		# Create a non-uniform mesh manually for predictable spacing
		Ω_nonunif = create_test_domain(0.0, 5.0)
		Ωₕ_nonunif = mesh(Ω_nonunif, 4, true; backend = backend()) # Start uniform
		nonunif_pts = [0.0, 1.0, 3.0, 5.0] # Spacing: 1.0, 2.0, 2.0
		set_points!(Ωₕ_nonunif, nonunif_pts)
		set_indices!(Ωₕ_nonunif, CartesianIndices((4,)))

		@testset "spacing" begin
			# Uniform
			@test spacing(Ωₕ_unif, 1) ≈ 1.0 # Defined as pts[2]-pts[1]
			@test spacing(Ωₕ_unif, 2) ≈ 1.0
			@test spacing(Ωₕ_unif, 5) ≈ 1.0
			@test collect(spacings_iterator(Ωₕ_unif)) ≈ [1.0, 1.0, 1.0, 1.0, 1.0]

			# Non-uniform
			@test spacing(Ωₕ_nonunif, 1) ≈ 1.0 # pts[2]-pts[1]
			@test spacing(Ωₕ_nonunif, 2) ≈ 1.0 # pts[2]-pts[1]
			@test spacing(Ωₕ_nonunif, 3) ≈ 2.0 # pts[3]-pts[2]
			@test spacing(Ωₕ_nonunif, 4) ≈ 2.0 # pts[4]-pts[3]
			# Iterator starts from index 1, using the definition for spacing(mesh, i)
			@test collect(spacings_iterator(Ωₕ_nonunif)) ≈ [1.0, 1.0, 2.0, 2.0]
		end

		@testset "hₘₐₓ" begin
			@test hₘₐₓ(Ωₕ_unif) ≈ 1.0
			@test hₘₐₓ(Ωₕ_nonunif) ≈ 2.0
		end

		@testset "half_spacing" begin
			# Uniform: h=1.0 => h_half should be 0.5, 1.0, 1.0, 1.0, 0.5
			@test half_spacing(Ωₕ_unif, 1) ≈ 0.5 * spacing(Ωₕ_unif, 1) ≈ 0.5
			@test half_spacing(Ωₕ_unif, 2) ≈ 0.5 * (spacing(Ωₕ_unif, 2) + spacing(Ωₕ_unif, 3)) ≈ 1.0
			@test half_spacing(Ωₕ_unif, 4) ≈ 0.5 * (spacing(Ωₕ_unif, 4) + spacing(Ωₕ_unif, 5)) ≈ 1.0
			@test half_spacing(Ωₕ_unif, 5) ≈ 0.5 * spacing(Ωₕ_unif, 5) ≈ 0.5
			@test collect(half_spacings_iterator(Ωₕ_unif)) ≈ [0.5, 1.0, 1.0, 1.0, 0.5]

			# Non-uniform: h = [1.0, 1.0, 2.0, 2.0] (spacings at indices 1, 2, 3, 4)
			# h_half should be: h1/2, (h1+h2)/2, (h2+h3)/2, h3/2  <- NO! Definition uses i and i+1
			# h_half(1) = spacing(1)/2 = 1.0/2 = 0.5
			# h_half(2) = (spacing(2) + spacing(3))/2 = (1.0 + 2.0)/2 = 1.5
			# h_half(3) = (spacing(3) + spacing(4))/2 = (2.0 + 2.0)/2 = 2.0
			# h_half(4) = spacing(4)/2 = 2.0/2 = 1.0
			@test half_spacing(Ωₕ_nonunif, 1) ≈ 0.5 * spacing(Ωₕ_nonunif, 1) ≈ 0.5
			@test half_spacing(Ωₕ_nonunif, 2) ≈ 0.5 * (spacing(Ωₕ_nonunif, 2) + spacing(Ωₕ_nonunif, 3)) ≈ 1.5
			@test half_spacing(Ωₕ_nonunif, 3) ≈ 0.5 * (spacing(Ωₕ_nonunif, 3) + spacing(Ωₕ_nonunif, 4)) ≈ 2.0
			@test half_spacing(Ωₕ_nonunif, 4) ≈ 0.5 * spacing(Ωₕ_nonunif, 4) ≈ 1.0
			@test collect(half_spacings_iterator(Ωₕ_nonunif)) ≈ [0.5, 1.5, 2.0, 1.0]
		end

		@testset "cell_measure" begin
			# Should be identical to half_spacing
			@test cell_measure(Ωₕ_unif, 1) ≈ half_spacing(Ωₕ_unif, 1)
			@test cell_measure(Ωₕ_unif, 3) ≈ half_spacing(Ωₕ_unif, 3)
			@test cell_measure(Ωₕ_unif, 5) ≈ half_spacing(Ωₕ_unif, 5)
			@test collect(cell_measures_iterator(Ωₕ_unif)) ≈ collect(half_spacings_iterator(Ωₕ_unif))

			@test cell_measure(Ωₕ_nonunif, 1) ≈ half_spacing(Ωₕ_nonunif, 1)
			@test cell_measure(Ωₕ_nonunif, 2) ≈ half_spacing(Ωₕ_nonunif, 2)
			@test cell_measure(Ωₕ_nonunif, 4) ≈ half_spacing(Ωₕ_nonunif, 4)
			@test collect(cell_measures_iterator(Ωₕ_nonunif)) ≈ collect(half_spacings_iterator(Ωₕ_nonunif))
		end

		@testset "half_points" begin
			# Uniform: pts = 0, 1, 2, 3, 4; npts=5
			# Indices for half_points go from 1 to npts+1 = 6
			# hp(1) = pts(1) = 0
			# hp(2) = (pts(1)+pts(2))/2 = 0.5
			# hp(3) = (pts(2)+pts(3))/2 = 1.5
			# hp(4) = (pts(3)+pts(4))/2 = 2.5
			# hp(5) = (pts(4)+pts(5))/2 = 3.5
			# hp(6) = pts(5) = 4
			@test half_point(Ωₕ_unif, 1) ≈ 0.0
			@test half_point(Ωₕ_unif, 2) ≈ 0.5
			@test half_point(Ωₕ_unif, 3) ≈ 1.5
			@test half_point(Ωₕ_unif, 5) ≈ 3.5
			@test half_point(Ωₕ_unif, 6) ≈ 4.0
			@test collect(half_points_iterator(Ωₕ_unif)) ≈ [0.0, 0.5, 1.5, 2.5, 3.5, 4.0]

			# Non-uniform: pts = 0, 1, 3, 5; npts=4
			# Indices for half_points go from 1 to npts+1 = 5
			# hp(1) = pts(1) = 0
			# hp(2) = (pts(1)+pts(2))/2 = 0.5
			# hp(3) = (pts(2)+pts(3))/2 = 2.0
			# hp(4) = (pts(3)+pts(4))/2 = 4.0
			# hp(5) = pts(4) = 5.0
			@test half_point(Ωₕ_nonunif, 1) ≈ 0.0
			@test half_point(Ωₕ_nonunif, 2) ≈ 0.5
			@test half_point(Ωₕ_nonunif, 3) ≈ 2.0
			@test half_point(Ωₕ_nonunif, 4) ≈ 4.0
			@test half_point(Ωₕ_nonunif, 5) ≈ 5.0
			@test collect(half_points_iterator(Ωₕ_nonunif)) ≈ [0.0, 0.5, 2.0, 4.0, 5.0]
		end
	end

	@testset "Index Subsets" begin
		npts = 5
		Ω = create_test_domain(0.0, 1.0)
		Ωₕ = mesh(Ω, npts, true; backend = backend())

		@test boundary_indices(Ωₕ) == (CartesianIndex(1), CartesianIndex(npts))
		@test interior_indices(Ωₕ) == CartesianIndices((2:(npts - 1),))

		# Edge cases
		Ωₕ_2 = mesh(Ω, 2, true; backend = backend())
		@test boundary_indices(Ωₕ_2) == (CartesianIndex(1), CartesianIndex(2))
		@test isempty(interior_indices(Ωₕ_2)) # Interior is empty range 2:1

		# Test on CartesianIndices directly
		inds = CartesianIndices((10,))
		@test boundary_indices(inds) == (CartesianIndex(1), CartesianIndex(10))
		@test interior_indices(inds) == CartesianIndices((2:9,))
	end

	@testset "Marker Setting" begin
		I = interval(0, 1)

		# Define markers
		dm = markers(I,
					 :Dirichlet => :left,
					 :Neumann => :right,
					 :Mixed => (:left, :right),
					 :LowerHalf => x -> x[1] < 0.5,
					 :PointMarker => x -> isapprox(x[1], 0.75))

		Ω = create_test_domain(0.0, 1.0; markers = dm)

		npts = 5 # Points: 0.0, 0.25, 0.5, 0.75, 1.0
		Ωₕ = mesh(Ω, npts, true; backend = backend())

		# Test marker retrieval before explicit setting (should be done by constructor)
		@test Set(keys(markers(Ωₕ))) == Set([:Dirichlet, :Neumann, :Mixed, :LowerHalf, :PointMarker])

		# Test explicit call to set_markers! (should ideally yield the same)
		set_markers!(Ωₕ, dm) # Recalculate
		@test Set(keys(markers(Ωₕ))) == Set([:Dirichlet, :Neumann, :Mixed, :LowerHalf, :PointMarker])
	end

	@testset "Mesh Modification" begin
		@testset "iterative_refinement!" begin
			dm = markers(interval(0, 1),
						 :BC => :left,
						 :Center => x -> 0.4 < x[1] < 0.6)
			Ω = create_test_domain(0.0, 1.0; markers = dm)

			npts_initial = 3 # Pts: 0.0, 0.5, 1.0
			Ωₕ = mesh(Ω, npts_initial, true; backend = backend())

			# Refine without marker update
			iterative_refinement!(Ωₕ)
			npts_refined1 = 2 * npts_initial - 1 # 2*3 - 1 = 5
			@test npoints(Ωₕ) == npts_refined1
			@test indices(Ωₕ) == CartesianIndices((npts_refined1,))
			@test points(Ωₕ) ≈ [0.0, 0.25, 0.5, 0.75, 1.0]

			# Refine *with* marker update
			Ωₕ2 = mesh(Ω, npts_initial, true; backend = backend()) # Start fresh: 0.0, 0.5, 1.0
			iterative_refinement!(Ωₕ2, dm)
			npts_refined2 = 2 * npts_initial - 1 # 5
			@test npoints(Ωₕ2) == npts_refined2
			@test indices(Ωₕ2) == CartesianIndices((npts_refined2,))
			@test points(Ωₕ2) ≈ [0.0, 0.25, 0.5, 0.75, 1.0]
		end

		@testset "change_points!" begin
			dm = markers(interval(0, 1),
						 :Endpoint => :right,
						 :NearStart => x -> x[1] < 0.3)
			Ω = create_test_domain(0.0, 2.0; markers = dm)
			npts = 5 # Pts: 0.0, 0.5, 1.0, 1.5, 2.0
			Ωₕ = mesh(Ω, npts, true; backend = backend())

			# Original markers
			new_pts_valid = [0.0, 0.1, 0.5, 1.5, 2.0] # Keep endpoints, change interior
			new_pts_invalid_len = [0.0, 1.0, 2.0]
			new_pts_invalid_ends = [0.1, 0.5, 1.0, 1.5, 2.1]

			# Test valid change without marker update
			Ωₕ_copy1 = deepcopy(Ωₕ)
			change_points!(Ωₕ_copy1, new_pts_valid)
			@test points(Ωₕ_copy1) ≈ new_pts_valid

			# Test valid change *with* marker update
			Ωₕ_copy2 = deepcopy(Ωₕ)
			change_points!(Ωₕ_copy2, dm, new_pts_valid)
			@test points(Ωₕ_copy2) ≈ new_pts_valid
		end
	end
end