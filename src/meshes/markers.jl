"""
	struct MarkerIndices{D}
		cartesian_index::Set{CartesianIndex{D}}
		cartesian_indices::Set{CartesianIndices{D}}
	end

	Structure to hold sets of individual `CartesianIndex` or `CartesianIndices`.
"""
struct MarkerIndices{D,CartIndicesType} <: BrambleType
	cartesian_index::Set{CartesianIndex{D}}
	cartesian_indices::Set{CartIndicesType}
end

"""
	boundary_symbol_to_cartesian(indices::CartesianIndices)

	Returns a named tuple connecting the facet labels of a set to the corresponding `CartesianIndices`.

```@example
julia> boundary_symbol_to_cartesian(CartesianIndices((1:3, 1:4)))
(left = CartesianIndices((1:1, 1:4)), 
 right = CartesianIndices((3:3, 1:4)), 
 top = CartesianIndices((1:3, 4:4)), 
 bottom = CartesianIndices((1:3, 1:1))
)
```

```@example
julia> boundary_symbol_to_cartesian(CartesianIndices((1:10, 1:20, 1:15)))
(left = CartesianIndices((1:10, 1:1, 1:15)), 
 right = CartesianIndices((1:10, 20:20, 1:15)), 
 top = CartesianIndices((1:10, 1:20, 15:15)), 
 bottom = CartesianIndices((1:10, 1:20, 1:1)), 
 front = CartesianIndices((10:10, 1:20, 1:15)), 
 back = CartesianIndices((1:1, 1:20, 1:15))
)
```
"""
@inline function boundary_symbol_to_cartesian(indices::CartesianIndices{1})
	named_tuple_1d = (; :left => first(indices), :right => last(indices))
	return named_tuple_1d
end

function boundary_symbol_to_cartesian(indices::CartesianIndices{2})
	N, M = size(indices)

	named_tuple_2d = (;
					  :left => indices[1:1, 1:M],
					  :right => indices[N:N, 1:M],
					  :top => indices[1:N, M:M],
					  :bottom => indices[1:N, 1:1])
	return named_tuple_2d
end

function boundary_symbol_to_cartesian(indices::CartesianIndices{3})
	N, M, K = size(indices)

	named_tuple_3d = (;
					  :left => indices[1:N, 1:1, 1:K],
					  :right => indices[1:N, M:M, 1:K],
					  :top => indices[1:N, 1:M, K:K],
					  :bottom => indices[1:N, 1:M, 1:1],
					  :front => indices[N:N, 1:M, 1:K],
					  :back => indices[1:1, 1:M, 1:K])
	return named_tuple_3d
end

"""
	boundary_symbol_to_dict(indices::CartesianIndices)

	Returns a dictionary connecting the facet labels of a set to the corresponding `CartesianIndices`. See [boundary_symbol_to_cartesian](@ref)
"""
boundary_symbol_to_dict(indices::CartesianIndices) = Dict(pairs(boundary_symbol_to_cartesian(indices)))

"""
	MeshMarkers{D}

Type of dictionary to store the `CartesianIndices` associated with a [MarkerIndices](@ref).
"""
CIndices_Type{D} = CartesianIndices{D,NTuple{D,UnitRange{Int}}}
MeshMarkers{D} = Dict{Symbol,MarkerIndices{D,CIndices_Type{D}}}

function Base.show(io::IO, markers::MeshMarkers{D}) where D
	labels = collect(keys(markers))
	labels_styled_combined = color_markers(labels)

	fields = D == 1 ? ("Markers",) : ("Resolution",)
	mlength = max_length_fields(fields)

	final_output = style_field("Markers", labels_styled_combined, max_length = mlength)
	print(io, final_output)
end

"""
	marker(Ωₕ::MeshType, symbol::Symbol)

Returns the [Marker](@ref) identifier with label `symbol`.
"""
@inline marker(Ωₕ::MeshType, symbol::Symbol) = Ωₕ.markers[symbol]

function process_label_for_mesh!(markers_mesh::MeshMarkers{D}, set_labels) where D
	cart_idxs_type = CartesianIndices{D,NTuple{D,UnitRange{Int}}}
	cart_idx_type = CartesianIndex{D}

	for label in set_labels
		idxs = MarkerIndices{D,cart_idxs_type}(Set{cart_idx_type}(), Set{cart_idxs_type}())
		markers_mesh[label] = idxs
	end
end

function _init_mesh_markers(_::MeshType{D}, domain_markers::DomainMarkers) where D
	markers_mesh = MeshMarkers{D}()

	process_label_for_mesh!(markers_mesh, label_symbols(domain_markers))
	process_label_for_mesh!(markers_mesh, label_tuples(domain_markers))
	process_label_for_mesh!(markers_mesh, label_conditions(domain_markers))

	return markers_mesh
end

"""
	set_markers!(Ωₕ::MeshType, domain_markers::DomainMarkers)

Populates the marker index collections of mesh Ωₕ based on boundary symbols or geometric conditions defined in the [DomainMarkers](@ref).
"""
function set_markers!(Ωₕ::MeshType, domain_markers)
	mesh_markers = _init_mesh_markers(Ωₕ, domain_markers)

	_set_markers_symbols!(mesh_markers, symbols(domain_markers), Ωₕ)
	_set_markers_symbols!(mesh_markers, tuples(domain_markers), Ωₕ)
	_set_markers_conditions!(mesh_markers, conditions(domain_markers), Ωₕ)

	Ωₕ.markers = mesh_markers
end

function __process_symbols(identifier)
	source_iterable = identifier isa Symbol ? (identifier,) : identifier
	return Set{Symbol}(source_iterable)
end

function _set_markers_symbols!(mesh_markers::MeshMarkers{D}, symbols, Ωₕ) where D
	symbol_to_index_map = boundary_symbol_to_dict(indices(Ωₕ))

	for marker in symbols
		@unpack label, identifier = marker
		marker_label = mesh_markers[label]
		target_indices = D == 1 ? marker_label.cartesian_index : marker_label.cartesian_indices

		processed_symbols = __process_symbols(identifier)
		indices_to_add = (symbol_to_index_map[sym] for sym in processed_symbols)

		union!(target_indices, indices_to_add)
	end
end

function __process_condition!(mesh_marker, identifier, Ωₕ)
	c_index = mesh_marker.cartesian_index
	indices_to_add = (idx for idx in indices(Ωₕ) if identifier(points(Ωₕ, idx)))

	union!(c_index, indices_to_add)
end

function _set_markers_conditions!(mesh_markers::MeshMarkers, conditions, Ωₕ)
	for marker in conditions
		@unpack label, identifier = marker

		__process_condition!(mesh_markers[label], identifier, Ωₕ)
		merge_consecutive_indices!(mesh_markers[label])
	end
end

"""
	merge_consecutive_indices!(marker_data::MarkerIndices{1})

Finds sequences of consecutive `CartesianIndex{1}` elements within `marker_data.cartesian_index`. Removes these sequences (if longer than one element) and adds the corresponding `CartesianIndices{1}` range object to `marker_data.cartesian_indices`.
"""
function merge_consecutive_indices!(marker_data::MarkerIndices{1})
	cartesian_index_set = marker_data.cartesian_index
	cartesian_indices_set = marker_data.cartesian_indices

	n = length(cartesian_index_set)

	# Need at least 2 elements to potentially form a mergeable range
	if n < 2
		return nothing
	end

	# --- Optimization using BitSet ---
	# 1. Convert CartesianIndex values to integers in a BitSet
	#    This allocates the BitSet but avoids collect+sort.
	#    Iteration over the BitSet is fast and yields sorted integers.
	int_values_bs = BitSet(ci.I[1] for ci in cartesian_index_set)
	# --------------------------------

	# Store results temporarily as primitive types to minimize allocations until the end
	ranges_found = Vector{Base.UnitRange{Int}}()     # Stores integer ranges like 1:3, 7:8
	vals_to_remove = Vector{Int}()             # Stores integers like 1,2,3, 7,8

	# --- Iterate efficiently through the BitSet ---
	# We need to manually handle the iterator to detect runs
	iter_state = iterate(int_values_bs)
	while iter_state !== nothing
		start_val, state = iter_state # Current value is the start of a potential run
		end_val = start_val           # Track the end of the run

		# Look ahead for consecutive values
		prev_val = start_val
		iter_state = iterate(int_values_bs, state) # Advance iterator once

		while iter_state !== nothing
			current_val, state = iter_state
			if current_val == prev_val + 1
				# Extend the run
				end_val = current_val
				prev_val = current_val
				iter_state = iterate(int_values_bs, state) # Consume this element
			else
				# Run ended (or next element is not consecutive)
				break
			end
		end
		# --- Run identified: start_val to end_val ---

		# Check if the run had more than one element
		if end_val > start_val
			push!(ranges_found, start_val:end_val)
			# Add all values in the found range to removal list
			# This avoids creating intermediate CartesianIndex objects here
			for v in start_val:end_val
				push!(vals_to_remove, v)
			end
		end
		# The outer loop continues with the state from where the inner loop broke or finished
	end
	# --- Finished iterating through BitSet ---

	# --- Apply changes if any ranges were found ---
	if !isempty(ranges_found)
		# Convert collected integer ranges to CartesianIndices objects
		# Using a generator avoids allocating an intermediate collection
		ranges_to_add = Set(CartesianIndices((r,)) for r in ranges_found)

		# Convert collected integer values to CartesianIndex objects for removal
		# Using a generator avoids allocating an intermediate collection
		indices_to_remove = Set(CartesianIndex(v) for v in vals_to_remove)

		# Modify the original marker_data sets
		union!(cartesian_indices_set, ranges_to_add)
		setdiff!(cartesian_index_set, indices_to_remove)
	end
	# --- End apply changes ---

	return nothing
end

"""
	merge_consecutive_indices!(marker_data::MarkerIndices{D}) where D

Finds sequences of `CartesianIndex{D}` elements consecutive along any single
axis within `marker_data.cartesian_index`. Removes these sequences (if longer than one element)
and adds the corresponding `CartesianIndices{D}` range object to
`marker_data.cartesian_indices`.
"""
function merge_consecutive_indices!(marker_data::MarkerIndices{D}; check_consistency = true) where D
	initial_indices_copy = check_consistency ? copy(marker_data.cartesian_index) : Set{CartesianIndex{D}}()

	remaining_indices = copy(marker_data.cartesian_index)
	if isempty(remaining_indices)
		if check_consistency && !isempty(initial_indices_copy)
			error("Internal logic error: Input cartesian_index was empty but initial copy wasn't.")
		end

		return nothing
	end

	# These store the results of *this merging run*
	output_ranges = Set{CartesianIndices{D}}()
	# Keep track of indices successfully merged into multi-index blocks *in this run*
	merged_indices_in_run = Set{CartesianIndex{D}}()

	# Use a copy for safe iteration while modifying remaining_indices
	seeds_to_process = copy(remaining_indices)

	while !isempty(seeds_to_process)
		seed_index = first(seeds_to_process)

		# Skip if seed was already incorporated into a block previously found in this run
		if !(seed_index in remaining_indices)
			pop!(seeds_to_process, seed_index)
			continue
		end

		current_ranges = ntuple(k -> seed_index.I[k]:seed_index.I[k], D)

		# --- Expansion Phase (Identical to previous version) ---
		while true
			expanded_in_pass = false
			for dim in 1:D
				# Positive expansion
				while true
					next_coord = current_ranges[dim].stop + 1
					slice_ranges = Base.setindex(current_ranges, next_coord:next_coord, dim)
					slice_indices = CartesianIndices(slice_ranges)
					can_expand = true
					for idx_in_slice in slice_indices
						if !(idx_in_slice in remaining_indices)
							can_expand = false
							break
						end
					end
					if can_expand
						expanded_ranges = Base.setindex(current_ranges, (current_ranges[dim].start):next_coord, dim)
						current_ranges = expanded_ranges
						expanded_in_pass = true
					else
						break
					end
				end
				# Negative expansion
				while true
					prev_coord = current_ranges[dim].start - 1
					slice_ranges = Base.setindex(current_ranges, prev_coord:prev_coord, dim)
					slice_indices = CartesianIndices(slice_ranges)
					can_expand = true
					for idx_in_slice in slice_indices
						if !(idx_in_slice in remaining_indices)
							can_expand = false
							break
						end
					end
					if can_expand
						expanded_ranges = Base.setindex(current_ranges, prev_coord:(current_ranges[dim].stop), dim)
						current_ranges = expanded_ranges
						expanded_in_pass = true
					else
						break
					end
				end
			end # end loop dimensions
			!expanded_in_pass && break
		end # end expansion phase
		# --- End Expansion Phase ---

		final_block = CartesianIndices(current_ranges)
		block_size = length(final_block)

		# --- Decision & Update Sets ---
		indices_in_block_set = Set(final_block) # Needed in both cases for removal

		if block_size > 1
			# Successful merge
			push!(output_ranges, final_block)
			union!(merged_indices_in_run, indices_in_block_set)
			setdiff!(remaining_indices, indices_in_block_set) # Remove from available pool
		else
			# Isolated index (block_size == 1)
			# Don't add to output_ranges or merged_indices_in_run
			delete!(remaining_indices, seed_index) # Just remove the single seed from available pool
		end

		# Remove processed indices (seed or full block) from the list of seeds to process
		setdiff!(seeds_to_process, indices_in_block_set)
	end # end while !isempty(seeds_to_process)

	# --- Apply changes to marker_data ---
	# Update cartesian_index: Remove indices that were successfully merged in this run
	setdiff!(marker_data.cartesian_index, merged_indices_in_run)
	# Add the newly found multi-element blocks to the cartesian_indices set
	union!(marker_data.cartesian_indices, output_ranges)

	# --- Consistency Check ---
	if check_consistency
		# Reconstruct the total set of points represented *after* the merge operation.
		# Start with the indices remaining in cartesian_index
		reconstructed_indices = copy(marker_data.cartesian_index)

		# Add all indices contained within the *newly added* ranges
		# Note: We are checking the conservation of points *originating* from the
		# initial cartesian_index set. If marker_data.cartesian_indices had pre-existing ranges,
		# this check doesn't include them, which is correct for verifying this function's action.
		for block in output_ranges
			union!(reconstructed_indices, Set(block))
		end

		# Compare the reconstructed set with the initial set
		if reconstructed_indices != initial_indices_copy
			# --- Debug Output ---
			println("Consistency Check FAILED!")
			println("Initial index count:      ", length(initial_indices_copy))
			println("Reconstructed index count:", length(reconstructed_indices))

			lost_indices = setdiff(initial_indices_copy, reconstructed_indices)
			gained_indices = setdiff(reconstructed_indices, initial_indices_copy)

			if !isempty(lost_indices)
				println("Lost indices (present initially, missing finally):")
				display(lost_indices)
			end
			if !isempty(gained_indices)
				println("Gained indices (present finally, missing initially):")
				display(gained_indices)
			end

			println("\nState before error:")
			println("Initial cartesian_index copy: ")
			display(initial_indices_copy)
			println("Final cartesian_index: ")
			display(marker_data.cartesian_index)
			println("Final added cartesian_indices (output_ranges): ")
			display(output_ranges)
			println("Reconstructed indices: ")
			display(reconstructed_indices)

			error("Consistency check failed: The set of indices after merging does not match the initial set.")
		end
	end

	return nothing
end