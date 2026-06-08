"""
    compute_safe_strides(stencil_offsets::Vector{CartesianIndex{D}}) where D

Computes the safe multi-coloring strides in each coordinate direction given the stencil offsets.
"""
function compute_safe_strides(stencil_offsets::Vector{CartesianIndex{D}}) where {D}
	# Initialize bounds with the first offset
	lower_bounds = first(stencil_offsets)
	upper_bounds = first(stencil_offsets)

	# Find the maximum span of the stencil in every dimension
	for offset in stencil_offsets
		lower_bounds = min(lower_bounds, offset)
		upper_bounds = max(upper_bounds, offset)
	end

	# The safe stride in dimension d is (max - min) + 1
	# For a standard 3-point stencil {-1, 0, 1}, this yields 1 - (-1) + 1 = 3
	return (upper_bounds - lower_bounds) + CartesianIndex(ntuple(x -> 1, D))
end

"""
    partition_grid_by_colors(grid_indices::CartesianIndices{D}, strides::CartesianIndex{D}) where D

Partitions the grid coordinates into independent color groups using the specified stride vector.
"""
function partition_grid_by_colors(grid_indices::CartesianIndices{D}, strides::CartesianIndex{D}) where {D}
	stride_tuple = Tuple(strides)
	num_colors = prod(stride_tuple)

	# Pre-allocate arrays for each independent color
	color_groups = [CartesianIndex{D}[] for _ in 1:num_colors]

	# Map multidimensional mod results to a linear 1D color ID
	linear_mapper = LinearIndices(stride_tuple)

	for I in grid_indices
		# Compute the periodic color coordinate (1-based for Julia)
		color_coord = ntuple(d -> mod(I[d] - 1, stride_tuple[d]) + 1, D)
		color_id = linear_mapper[color_coord...]

		push!(color_groups[color_id], I)
	end

	return color_groups
end

"""
    assemble_matrix_parallel!(A, grid_indices, ast_operator)

Helper function showing the pattern of lock-free parallel assembly of matrix `A` using grid coloring.
"""
function assemble_matrix_parallel!(A, grid_indices, ast_operator)

	# 1. Ask the AST for the test function offsets
	# e.g., for D_{-x}, this might return [CartesianIndex(0), CartesianIndex(-1)]
	offsets = get_test_offsets(ast_operator)

	# 2. Compute periodic bounds and bin the grid
	strides = compute_safe_strides(offsets)
	color_groups = partition_grid_by_colors(grid_indices, strides)

	# 3. Assemble!
	for color_group in color_groups
		# IMPLICIT THREAD BARRIER: Wait for the previous color to finish

		# All evaluations in this loop are strictly independent.
		# No two threads will ever target the same matrix row.
		Threads.@threads for I in color_group
			# Your zero-allocation AST tuple generator
			evaluate_and_add!(A, ast_operator, I)
		end
	end

	return A
end
