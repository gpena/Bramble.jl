# grid_coloring.jl
# Unit tests for the grid coloring partitioning and safety stride computations.

using Test
using Bramble

@testset "Grid Coloring Tests" begin
	@testset "Safety Stride Computes" begin
		# 1D: {-1, 0, 1} stencil
		offsets_1d = [CartesianIndex(-1), CartesianIndex(0), CartesianIndex(1)]
		stride_1d = Bramble.compute_safe_strides(offsets_1d)
		@test stride_1d == CartesianIndex(3)

		# 2D: 5-point stencil (Laplacian)
		offsets_2d = [
			CartesianIndex(-1, 0),
			CartesianIndex(0, -1),
			CartesianIndex(0, 0),
			CartesianIndex(1, 0),
			CartesianIndex(0, 1)
		]
		stride_2d = Bramble.compute_safe_strides(offsets_2d)
		@test stride_2d == CartesianIndex(3, 3)

		# 3D: Asymmetric stencil offsets
		offsets_3d = [
			CartesianIndex(-2, 0, 1),
			CartesianIndex(1, -1, 2)
		]
		# Expected stride in x: 1 - (-2) + 1 = 4
		# Expected stride in y: 0 - (-1) + 1 = 2
		# Expected stride in z: 2 - 1 + 1 = 2
		stride_3d = Bramble.compute_safe_strides(offsets_3d)
		@test stride_3d == CartesianIndex(4, 2, 2)
	end

	@testset "Grid Partitioning by Colors" begin
		grid_indices = CartesianIndices((10, 12))
		strides = CartesianIndex(3, 4)
		color_groups = Bramble.partition_grid_by_colors(grid_indices, strides)

		# Verify number of color groups matches the stride volume
		@test length(color_groups) == 3 * 4

		# Collect all indices and check coverage
		all_partitioned = CartesianIndex{2}[]
		for group in color_groups
			append!(all_partitioned, group)
		end

		# 1. Coverage check: all elements should be present exactly once
		@test length(all_partitioned) == length(grid_indices)
		@test sort(all_partitioned, by=x->(x[1], x[2])) == sort(vec(collect(grid_indices)), by=x->(x[1], x[2]))

		# 2. Independence check: elements within the same group must be at least one stride apart
		# i.e., they shouldn't share color conflicts.
		for group in color_groups
			for i in 1:length(group)
				for j in (i + 1):length(group)
					idx1 = group[i]
					idx2 = group[j]
					# Compute grid coordinate diffs
					diffs = abs.(Tuple(idx1 - idx2))
					# For any coordinates of the same color, they must have mod differences of 0 with strides
					# meaning the difference in coordinate d is a multiple of strides[d].
					# If diffs is less than strides in any coordinate, it must be zero in that coordinate.
					# But since idx1 != idx2, they cannot be identical.
					# More precisely: mod(I[d] - 1, strides[d]) is identical for all elements in the same color group.
					@test mod(idx1[1] - 1, 3) == mod(idx2[1] - 1, 3)
					@test mod(idx1[2] - 1, 4) == mod(idx2[2] - 1, 4)
				end
			end
		end
	end

	@testset "Serial vs Parallel Assembly Comparison" begin
		# Setup a 2D mesh & grid space
		N = 6
		I = interval(0.0, 1.0)
		Ω = I × I
		Mh = mesh(domain(Ω), (N, N), (false, false))
		Wh = gridspace(Mh)

		# 2D Laplacian stiffness bilinear form using backward gradients
		a_stiff = form(Wh, Wh, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))

		# Assemble in parallel using the in-place parallel assembly function
		A_parallel = Bramble.allocate_system_matrix(a_stiff)
		Bramble.assemble_parallel!(A_parallel, a_stiff)

		# Assemble sequentially
		A_serial = Bramble.allocate_system_matrix(a_stiff)
		Bramble.assemble!(A_serial, a_stiff)

		# Verify that they yield the exact same matrices
		@test A_serial ≈ A_parallel
	end
end

