using Bramble: is_boundary_index, CartesianProduct, MeshnD, Backend, backend
using LinearAlgebra: hypot

# --- Test Suite ---
@testset "MeshnD Tests" begin
	# Helper function to create a simple nD domain
	function create_test_nd_domain(intervals::NTuple{D,Tuple{Float64,Float64}}; markers = nothing) where D
		nd_intervals = map(t -> interval(t[1], t[2]), intervals)
		prod = reduce(×, nd_intervals)

		if markers isa Nothing
			return domain(prod)
		else
			return domain(prod, markers)
		end
	end

	function create_test_nd_set(intervals::NTuple{D,Tuple{Float64,Float64}}) where D
		nd_intervals = map(t -> interval(t[1], t[2]), intervals)
		prod = reduce(×, nd_intervals)
		return prod
	end

	@testset "Helper Functions (D>1)" begin
		@testset "boundary_symbol_to_dict (D=2)" begin
			indices = CartesianIndices((4, 5)) # N=4, M=5
			dict = boundary_symbol_to_dict(indices)
			@test dict[:left] == CartesianIndices((1:1, 1:5))
			@test dict[:right] == CartesianIndices((4:4, 1:5))
			@test dict[:bottom] == CartesianIndices((1:4, 1:1))
			@test dict[:top] == CartesianIndices((1:4, 5:5))
		end

		@testset "boundary_symbol_to_dict (D=3)" begin
			indices = CartesianIndices((3, 4, 5)) # N=3, M=4, K=5
			dict = boundary_symbol_to_dict(indices)
			@test dict[:left] == CartesianIndices((1:3, 1:1, 1:5))
			@test dict[:right] == CartesianIndices((1:3, 4:4, 1:5))
			@test dict[:bottom] == CartesianIndices((1:3, 1:4, 1:1))
			@test dict[:top] == CartesianIndices((1:3, 1:4, 5:5))
			@test dict[:back] == CartesianIndices((1:1, 1:4, 1:5))
			@test dict[:front] == CartesianIndices((3:3, 1:4, 1:5))
		end

		@testset "is_boundary_index (D=2)" begin
			idxs = CartesianIndices((3, 4))
			@test is_boundary_index(CartesianIndex(1, 1), idxs) == true
			@test is_boundary_index(CartesianIndex(1, 2), idxs) == true
			@test is_boundary_index(CartesianIndex(1, 4), idxs) == true
			@test is_boundary_index(CartesianIndex(2, 1), idxs) == true
			@test is_boundary_index(CartesianIndex(3, 1), idxs) == true
			@test is_boundary_index(CartesianIndex(2, 4), idxs) == true
			@test is_boundary_index(CartesianIndex(3, 4), idxs) == true
			@test is_boundary_index(CartesianIndex(3, 2), idxs) == true
			@test is_boundary_index(CartesianIndex(2, 2), idxs) == false
			@test is_boundary_index(CartesianIndex(2, 3), idxs) == false
			# Test tuple input
			@test is_boundary_index((1, 3), idxs) == true
			@test is_boundary_index((2, 2), idxs) == false
		end

		@testset "is_boundary_index (D=3)" begin
			idxs = CartesianIndices((3, 4, 2))
			@test is_boundary_index(CartesianIndex(1, 1, 1), idxs) == true # Corner
			@test is_boundary_index(CartesianIndex(2, 2, 1), idxs) == true # On face z=1
			@test is_boundary_index(CartesianIndex(2, 1, 2), idxs) == true # On face y=1
			@test is_boundary_index(CartesianIndex(1, 3, 2), idxs) == true # On face x=1
			@test is_boundary_index(CartesianIndex(3, 2, 1), idxs) == true # On face x=3
			@test is_boundary_index(CartesianIndex(2, 4, 2), idxs) == true # On face y=4
			@test is_boundary_index(CartesianIndex(3, 4, 2), idxs) == true # Corner
			@test is_boundary_index(CartesianIndex(2, 2, 2), idxs) == true # On face z=2
			@test is_boundary_index(CartesianIndex(2, 3, 1), idxs) == true # On face z=1

			@test is_boundary_index(CartesianIndex(2, 2, 1), idxs) == true # Interior in x,y but boundary in z
			@test is_boundary_index(CartesianIndex(2, 3, 1), idxs) == true # Interior in x,y but boundary in z

			# Need an interior point test if dims allow
			idxs_larger = CartesianIndices((4, 4, 4))
			@test is_boundary_index(CartesianIndex(2, 2, 2), idxs_larger) == false
			@test is_boundary_index(CartesianIndex(2, 3, 2), idxs_larger) == false
			@test is_boundary_index(CartesianIndex(3, 2, 3), idxs_larger) == false
		end
	end # Helper Functions Testset

	@testset "Dimension D=2" begin
		D = 2
		npts_2d = (4, 5) # Nx=4, Ny=5
		intervals_2d = ((0.0, 3.0), (0.0, 4.0)) # dx=1.0, dy=1.0
		Ω_2d = create_test_nd_domain(intervals_2d)
		Ωₕ_2d_unif = mesh(Ω_2d, npts_2d, (true, true); backend = backend())
		Ωₕ_2d_nonunif = mesh(Ω_2d, npts_2d, (false, true); backend = backend())

		@testset "Construction and Basic Properties (D=2)" begin
			@test Ωₕ_2d_unif isa MeshnD{2}
			@test backend(Ωₕ_2d_unif) isa Backend
			@test eltype(Ωₕ_2d_unif) == Float64
			@test dim(Ωₕ_2d_unif) == D
			@test dim(typeof(Ωₕ_2d_unif)) == D
			@test npoints(Ωₕ_2d_unif) == prod(npts_2d) == 20
			@test npoints(Ωₕ_2d_unif, Tuple) == npts_2d
			@test indices(Ωₕ_2d_unif) == CartesianIndices(npts_2d)
			@test length(Ωₕ_2d_unif.submeshes) == D
			@test Ωₕ_2d_unif(1) isa Mesh1D # Access submesh
			@test Ωₕ_2d_unif(2) isa Mesh1D
			@test npoints(Ωₕ_2d_unif(1)) == npts_2d[1]
			@test npoints(Ωₕ_2d_unif(2)) == npts_2d[2]

			# Check non-uniform construction basics
			@test Ωₕ_2d_nonunif isa MeshnD{2}
			@test npoints(Ωₕ_2d_nonunif) == 20
			@test npoints(Ωₕ_2d_nonunif, Tuple) == npts_2d
			@test indices(Ωₕ_2d_nonunif) == CartesianIndices(npts_2d)
			# Check if first submesh points are non-uniform (sorted, endpoints match)
			pts_x = points(Ωₕ_2d_nonunif(1))
			@test pts_x[1] ≈ intervals_2d[1][1]
			@test pts_x[end] ≈ intervals_2d[1][2]
			@test all(diff(pts_x) .> 0)
			# Check if second submesh points are uniform
			pts_y = points(Ωₕ_2d_nonunif(2))
			@test pts_y ≈ range(intervals_2d[2][1], intervals_2d[2][2], length = npts_2d[2])
		end

		@testset "Geometric Properties (D=2, Uniform)" begin
			# Points: x = [0,1,2,3], y = [0,1,2,3,4]
			@test points(Ωₕ_2d_unif(1)) ≈ [0.0, 1.0, 2.0, 3.0]
			@test points(Ωₕ_2d_unif(2)) ≈ [0.0, 1.0, 2.0, 3.0, 4.0]

			# points(mesh) returns tuple of point vectors
			pts_tuple = points(Ωₕ_2d_unif)
			@test pts_tuple[1] ≈ [0.0, 1.0, 2.0, 3.0]
			@test pts_tuple[2] ≈ [0.0, 1.0, 2.0, 3.0, 4.0]

			# points(mesh, index)
			@test point(Ωₕ_2d_unif, CartesianIndex(2, 3)) == (1.0, 2.0) # x[2], y[3]
			@test point(Ωₕ_2d_unif, (4, 5)) == (3.0, 4.0) # x[4], y[5]

			# points_iterator(mesh)
			pts_iter = points_iterator(Ωₕ_2d_unif)
			pts = collect(pts_iter)

			@test length(pts_iter) == 20
			@test first(pts_iter) == (0.0, 0.0)
			@test last(pts_iter) == (3.0, 4.0)
			@test collect(pts_iter)[1] == (0.0, 0.0)
			@test collect(pts_iter)[4] == (3.0, 0.0) # End of first column
			@test collect(pts_iter)[5] == (0.0, 1.0) # Start of second column

			# Spacing (uniform dx=1, dy=1)
			@test spacing(Ωₕ_2d_unif, (2, 3)) == (1.0, 1.0)
			@test spacing(Ωₕ_2d_unif, CartesianIndex(1, 1)) == (1.0, 1.0) # Uses h1 definition

			# Half spacing (uniform hx=1, hy=1 => half_hx = 0.5,1,1,0.5; half_hy = 0.5,1,1,1,0.5)
			@test half_spacing(Ωₕ_2d_unif, (1, 1)) == (0.5, 0.5) # Corner
			@test half_spacing(Ωₕ_2d_unif, (2, 3)) == (1.0, 1.0) # Interior
			@test half_spacing(Ωₕ_2d_unif, (4, 5)) == (0.5, 0.5) # Corner
			@test half_spacing(Ωₕ_2d_unif, (1, 5)) == (0.5, 0.5) # Corner
			@test half_spacing(Ωₕ_2d_unif, (4, 1)) == (0.5, 0.5) # Corner
			@test half_spacing(Ωₕ_2d_unif, (2, 1)) == (1.0, 0.5) # Edge
			@test half_spacing(Ωₕ_2d_unif, (1, 3)) == (0.5, 1.0) # Edge

			# hₘₐₓ (uniform spacing = (1,1))
			@test hₘₐₓ(Ωₕ_2d_unif) ≈ hypot(1.0, 1.0) ≈ sqrt(2.0)

			# Cell Measure (product of half spacings)
			@test cell_measure(Ωₕ_2d_unif, (1, 1)) ≈ 0.5 * 0.5 == 0.25
			@test cell_measure(Ωₕ_2d_unif, (2, 3)) ≈ 1.0 * 1.0 == 1.0
			@test cell_measure(Ωₕ_2d_unif, (4, 5)) ≈ 0.5 * 0.5 == 0.25

			# Half points
			# hp_x = [0.0, 0.5, 1.5, 2.5, 3.0] (len 5)
			# hp_y = [0.0, 0.5, 1.5, 2.5, 3.5, 4.0] (len 6)
			@test half_point(Ωₕ_2d_unif, (1, 1)) == (0.0, 0.0) # hp_x[1], hp_y[1]
			@test half_point(Ωₕ_2d_unif, (2, 3)) == (0.5, 1.5) # hp_x[2], hp_y[3]
			@test half_point(Ωₕ_2d_unif, (4 + 1, 5 + 1)) == (3.0, 4.0) # hp_x[5], hp_y[6] - Accessing end half points
			@test half_point(Ωₕ_2d_unif, CartesianIndex(3, 4)) == (1.5, 2.5) # hp_x[3], hp_y[4]
		end

		@testset "Index Subsets (D=2)" begin
			idxs = indices(Ωₕ_2d_unif) # (4, 5)
			int_indices = collect(interior_indices(Ωₕ_2d_unif))

			@test is_boundary_index(CartesianIndex(1, 1), Ωₕ_2d_unif)
			@test is_boundary_index(CartesianIndex(4, 5), Ωₕ_2d_unif)
			@test is_boundary_index(CartesianIndex(3, 1), Ωₕ_2d_unif)
			@test is_boundary_index(CartesianIndex(1, 3), Ωₕ_2d_unif)
			@test is_boundary_index(CartesianIndex(4, 2), Ωₕ_2d_unif)
			@test is_boundary_index(CartesianIndex(2, 5), Ωₕ_2d_unif)
			@test !is_boundary_index(CartesianIndex(2, 2), Ωₕ_2d_unif)
			@test !is_boundary_index(CartesianIndex(3, 4), Ωₕ_2d_unif)

			# Interior indices are (2:N-1, 2:M-1) -> (2:3, 2:4)
			@test interior_indices(Ωₕ_2d_unif) == CartesianIndices((2:3, 2:4))
			@test length(int_indices) == (4 - 2) * (5 - 2) == 2 * 3 == 6
			@test CartesianIndex(2, 2) in int_indices
			@test CartesianIndex(3, 4) in int_indices
			@test !(CartesianIndex(1, 2) in int_indices)
			@test !(CartesianIndex(3, 5) in int_indices)
		end

		@testset "Marker Setting (D=2)" begin
			Ω_2d_dummy = create_test_nd_set(intervals_2d)

			dm_2d = markers(Ω_2d_dummy,
							:LeftWall => :left,
							:RightWall => :right,
							:TopBottom => (:top, :bottom),
							:CenterRegion => p -> 0.8 < p[1] < 2.2 && 1.5 < p[2] < 2.5)
			Ω_2d_marked = create_test_nd_domain(intervals_2d, markers = dm_2d)
			Ωₕ_2d_marked = mesh(Ω_2d_marked, npts_2d, (true, true); backend = backend()) # Pts: x=[0,1,2,3], y=[0,1,2,3,4]
		end

		@testset "Mesh Modification (D=2)" begin
			Ω_2d_dummy = create_test_nd_set(intervals_2d)

			# Setup for modification tests
			dm_2d = markers(Ω_2d_dummy, :L => :left)
			Ω_2d_mod = create_test_nd_domain(intervals_2d, markers = dm_2d)
			npts_initial = (3, 3) # Pts: x=[0, 1.5, 3], y=[0, 2, 4]
			Ωₕ_2d_orig = mesh(Ω_2d_mod, npts_initial, (true, true); backend = backend())

			# iterative_refinement!
			Ωₕ_2d_refined = deepcopy(Ωₕ_2d_orig)
			iterative_refinement!(Ωₕ_2d_refined, dm_2d)
			npts_refined = (2 * npts_initial[1] - 1, 2 * npts_initial[2] - 1) # (5, 5)
			@test npoints(Ωₕ_2d_refined, Tuple) == npts_refined
			@test indices(Ωₕ_2d_refined) == CartesianIndices(npts_refined)
			# Check a point coordinate after refinement
			@test points(Ωₕ_2d_refined(1)) ≈ [0.0, 0.75, 1.5, 2.25, 3.0]
			@test points(Ωₕ_2d_refined(2)) ≈ [0.0, 1.0, 2.0, 3.0, 4.0]

			# change_points!
			Ωₕ_2d_changed = deepcopy(Ωₕ_2d_orig)
			new_pts_x = [0.0, 1.0, 3.0] # Keep endpoints
			new_pts_y = [0.0, 3.0, 4.0] # Keep endpoints
			change_points!(Ωₕ_2d_changed, dm_2d, (new_pts_x, new_pts_y))
			@test npoints(Ωₕ_2d_changed, Tuple) == npts_initial # Size doesn't change
			@test points(Ωₕ_2d_changed(1)) ≈ new_pts_x
			@test points(Ωₕ_2d_changed(2)) ≈ new_pts_y
		end
	end # Dimension D=2 Testset

	@testset "Dimension D=3" begin
		D = 3
		npts_3d = (3, 4, 2) # Nx=3, Ny=4, Nz=2
		intervals_3d = ((0.0, 2.0), (0.0, 3.0), (0.0, 1.0)) # dx=1.0, dy=1.0, dz=1.0
		Ω_3d = create_test_nd_domain(intervals_3d)
		Ωₕ_3d_unif = mesh(Ω_3d, npts_3d, (true, true, true); backend = backend())

		@testset "Construction and Basic Properties (D=3)" begin
			@test Ωₕ_3d_unif isa MeshnD{3}
			@test backend(Ωₕ_3d_unif) isa Backend
			@test eltype(Ωₕ_3d_unif) == Float64
			@test dim(Ωₕ_3d_unif) == D
			@test npoints(Ωₕ_3d_unif) == prod(npts_3d) == 24
			@test npoints(Ωₕ_3d_unif, Tuple) == npts_3d
			@test indices(Ωₕ_3d_unif) == CartesianIndices(npts_3d)
			@test length(Ωₕ_3d_unif.submeshes) == D
			@test Ωₕ_3d_unif(1) isa Mesh1D
			@test Ωₕ_3d_unif(2) isa Mesh1D
			@test Ωₕ_3d_unif(3) isa Mesh1D
			@test npoints(Ωₕ_3d_unif(1)) == npts_3d[1]
			@test npoints(Ωₕ_3d_unif(2)) == npts_3d[2]
			@test npoints(Ωₕ_3d_unif(3)) == npts_3d[3]
		end

		@testset "Geometric Properties (D=3, Uniform)" begin
			# Points: x = [0,1,2], y = [0,1,2,3], z = [0,1]
			@test points(Ωₕ_3d_unif(1)) ≈ [0.0, 1.0, 2.0]
			@test points(Ωₕ_3d_unif(2)) ≈ [0.0, 1.0, 2.0, 3.0]
			@test points(Ωₕ_3d_unif(3)) ≈ [0.0, 1.0]

			# point(mesh, index)
			@test point(Ωₕ_3d_unif, CartesianIndex(2, 3, 1)) == (1.0, 2.0, 0.0) # x[2], y[3], z[1]
			@test point(Ωₕ_3d_unif, (3, 4, 2)) == (2.0, 3.0, 1.0) # x[3], y[4], z[2]

			# Spacing (uniform dx=1, dy=1, dz=1)
			@test spacing(Ωₕ_3d_unif, (2, 3, 1)) == (1.0, 1.0, 1.0)

			# Half spacing (uniform h=1 => half_h = 0.5,1,0.5 for x; 0.5,1,1,0.5 for y; 0.5,0.5 for z)
			@test half_spacing(Ωₕ_3d_unif, (1, 1, 1)) == (0.5, 0.5, 0.5) # Corner
			@test half_spacing(Ωₕ_3d_unif, (2, 3, 1)) == (1.0, 1.0, 0.5) # Interior x,y; boundary z
			@test half_spacing(Ωₕ_3d_unif, (3, 4, 2)) == (0.5, 0.5, 0.5) # Corner

			# hₘₐₓ (uniform spacing = (1,1,1))
			@test hₘₐₓ(Ωₕ_3d_unif) ≈ hypot(1.0, 1.0, 1.0) ≈ sqrt(3.0)

			# Cell Measure (product of half spacings)
			@test cell_measure(Ωₕ_3d_unif, (1, 1, 1)) ≈ 0.5 * 0.5 * 0.5 == 0.125
			@test cell_measure(Ωₕ_3d_unif, (2, 3, 1)) ≈ 1.0 * 1.0 * 0.5 == 0.5
			@test cell_measure(Ωₕ_3d_unif, (3, 4, 2)) ≈ 0.5 * 0.5 * 0.5 == 0.125
		end

		@testset "Index Subsets (D=3)" begin
			idxs = indices(Ωₕ_3d_unif) # (3, 4, 2)
			bnd_indices = collect(boundary_indices(Ωₕ_3d_unif))
			int_indices = collect(interior_indices(Ωₕ_3d_unif))

			# Interior indices are (2:N-1, 2:M-1, 2:K-1) -> (2:2, 2:3, 2:1 -> Empty!)
			# Let's redefine for a mesh where interior exists
			npts_3d_larger = (4, 5, 4)
			Ω_3d_larger = create_test_nd_domain(((0.0, 3.0), (0.0, 4.0), (0.0, 3.0)))
			Ωₕ_3d_larger = mesh(Ω_3d_larger, npts_3d_larger, (true, true, true); backend = backend())
			idxs_larger = indices(Ωₕ_3d_larger) # (4,5,4)
			int_indices_lg = interior_indices(Ωₕ_3d_larger)

			@test is_boundary_index((1, 1, 1), Ωₕ_3d_larger)
			@test is_boundary_index((4, 5, 4), Ωₕ_3d_larger)
			@test is_boundary_index((2, 2, 1), Ωₕ_3d_larger)
			@test is_boundary_index((2, 1, 2), Ωₕ_3d_larger)
			@test is_boundary_index((1, 3, 3), Ωₕ_3d_larger)
			@test !is_boundary_index((2, 2, 2), Ωₕ_3d_larger)
			@test !is_boundary_index((3, 4, 3), Ωₕ_3d_larger)
			@test interior_indices(Ωₕ_3d_larger) == CartesianIndices((2:3, 2:4, 2:3))

			@test CartesianIndex(2, 2, 2) in int_indices_lg
			@test CartesianIndex(3, 4, 3) in int_indices_lg
			@test !(CartesianIndex(1, 2, 2) in int_indices_lg)
			@test !(CartesianIndex(3, 5, 3) in int_indices_lg)
		end

		@testset "Marker Setting (D=3)" begin
			_set = create_test_nd_set(intervals_3d)
			dm_3d = markers(_set,
							:BottomFace => :bottom,
							:FrontFace => :front,
							:SmallCorner => p -> p[1] < 0.5 && p[2] < 0.5 && p[3] < 0.5)
			Ω_3d_marked = create_test_nd_domain(intervals_3d, markers = dm_3d) # (3,4,2) pts, intervals ((0,2),(0,3),(0,1))
			Ωₕ_3d_marked = mesh(Ω_3d_marked, npts_3d, (true, true, true); backend = backend()) # Pts: x=[0,1,2], y=[0,1,2,3], z=[0,1]
		end
	end # Dimension D=3 Testset
end # Main Testset