"""
	struct MeshnD{D, BackendType, CartIndicesType, Mesh1DType} <: MeshType{D}
		markers::MeshMarkers{D}
		indices::CartIndicesType
		backend::BackendType
		submeshes::NTuple{D, Mesh1DType}
	end

Structure to store a cartesian nD-mesh (``2 \\leq n \\leq 3``). For efficiency, the mesh points are not stored. Instead, we store the points of the 1D meshes that make up the nD mesh. To connect both nD and 1D meshes, we use the indices in `indices`. The [DomainMarkers](@ref) are translated to `markers` as for [Mesh1D](@ref).
"""
mutable struct MeshnD{D,BackendType<:Backend,CartIndicesType,Mesh1DType<:MeshType{1}} <: MeshType{D}
	markers::MeshMarkers{D}
	indices::CartIndicesType
	const backend::BackendType
	submeshes::NTuple{D,Mesh1DType}
end

@generated function generate_submeshes(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}, backend) where D
	return :(Base.Cartesian.@ntuple $D i->_mesh(domain(projection(Ω, i)), (npts[i],), (unif[i],), backend))
end

function _mesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}, backend) where D
	@assert dim(Ω) == D
	_set = set(Ω)
	npts_with_collapsed = ntuple(i -> _set.collapsed[i] ? 1 : npts[i], D)

	idxs = generate_indices(npts_with_collapsed)
	submeshes = generate_submeshes(Ω, npts_with_collapsed, unif, backend)

	mesh_markers = MeshMarkers{D}()
	__mesh = MeshnD(mesh_markers, idxs, backend, submeshes)
	set_markers!(__mesh, markers(Ω))

	return __mesh
end

@inline _eltype(_::MeshnD{D,BackendType}) where {D,BackendType} = eltype(BackendType)
@inline Base.eltype(_::Type{<:MeshnD{D,BackendType}}) where {D,BackendType} = eltype(BackendType)

function Base.show(io::IO, Ωₕ::MeshnD)
	D = dim(Ωₕ)
	fields = ("Markers", "Resolution")
	mlength = max_length_fields(fields)

	colors = style_color_sets()
	num_colors = length(colors)

	type_info = style_title("$(D)D mesh", max_length = mlength)
	print(io, type_info * "\n")

	npts = npoints(Ωₕ, Tuple)
	styled_points = [let
						 set_str = "$(npts[i])"
						 color_sym = colors[mod1(i, num_colors)]
						 styled"{$color_sym:$(set_str)}"
					 end
					 for i in 1:D]

	sets_styled_combined = join(styled_points, " × ")

	points_info = "$(npoints(Ωₕ)) (" * sets_styled_combined * ")"
	submeshes = style_field("Resolution", points_info, max_length = mlength)
	print(io, submeshes * "\n")
	show(io, markers(Ωₕ))
end

"""
	(Ωₕ::MeshnD)(i)

Returns the `i`-th submesh of `Ωₕ`.
"""
@inline function (Ωₕ::MeshnD)(i)
	@assert 1 <= i <= dim(Ωₕ)
	return Ωₕ.submeshes[i]
end

@inline @generated _points(Ωₕ::MeshnD{D}) where D = :(Base.Cartesian.@ntuple $D i->_points(Ωₕ(i)))
@inline @generated _points(Ωₕ::MeshnD{D}, idx) where D = :(Base.Cartesian.@ntuple $D i->_points(Ωₕ(i), idx[i]))

@inline @generated _points_iterator(Ωₕ::MeshnD{D}) where D = :(Base.Iterators.product((Base.Cartesian.@ntuple $D i->_points_iterator(Ωₕ(i)))...))

@inline @generated _half_points(Ωₕ::MeshnD{D}, idx) where D = :(Base.Cartesian.@ntuple $D i->_half_points(Ωₕ(i), idx[i]))

@inline @generated _half_points_iterator(Ωₕ::MeshnD{D}) where D = :(Base.Iterators.product((Base.Cartesian.@ntuple $D i->_half_points_iterator(Ωₕ(i)))...))

@inline @generated _spacing(Ωₕ::MeshnD{D}, idx) where D = :(Base.Cartesian.@ntuple $D i->_spacing(Ωₕ(i), idx[i]))

@inline @generated _spacing_iterator(Ωₕ::MeshnD{D}) where D = :(Base.Iterators.product((Base.Cartesian.@ntuple $D i->_spacing_iterator(Ωₕ(i)))...))

@inline @generated function _half_spacing(Ωₕ::MeshnD{D}, idx) where D
	return :(Base.Cartesian.@ntuple $D i->_apply_hs_logic(_half_spacing(Ωₕ(i), idx[i])))
end

@inline _apply_hs_logic(value::T) where T = ifelse(value == zero(T), one(T), value)

@inline @generated _half_spacing_iterator(Ωₕ::MeshnD{D}) where D = :(Base.Iterators.product((Base.Cartesian.@ntuple $D i->_half_spacing_iterator(Ωₕ(i)))...))

@inline @generated _npoints(Ωₕ::MeshnD) = :(prod(_npoints(Ωₕ, Tuple)))
@inline @generated _npoints(Ωₕ::MeshnD{D}, ::Type{Tuple}) where D = :(Base.Cartesian.@ntuple $D i->_npoints(Ωₕ(i)))

@inline function _hₘₐₓ(Ωₕ::MeshnD)
	diagonals = Base.Iterators.map(h -> hypot(h...), spacing_iterator(Ωₕ))
	return maximum(diagonals)
end

@inline _cell_measure(Ωₕ::MeshnD, idx) = prod(_half_spacing(Ωₕ, idx))

@inline _cell_measure_iterator(Ωₕ::MeshnD) = (_cell_measure(Ωₕ, idx) for idx in indices(Ωₕ))

function _iterative_refinement!(Ωₕ::MeshnD{D}) where D
	for i in 1:D
		_iterative_refinement!(Ωₕ(i))
	end
end

function _iterative_refinement!(Ωₕ::MeshnD{D}, domain_markers) where D
	_iterative_refinement!(Ωₕ)

	npts = _npoints(Ωₕ, Tuple)

	idxs = generate_indices(npts)

	set_indices!(Ωₕ, idxs)
	set_markers!(Ωₕ, domain_markers)
end

function _change_points!(Ωₕ::MeshnD{D}, pts::NTuple{D,VecType}) where {D,VecType}
	for i in 1:D
		_change_points!(Ωₕ(i), pts[i])
	end
end

function _change_points!(Ωₕ::MeshnD{D}, domain_markers, pts::NTuple{D,VecType}) where {D,VecType}
	_change_points!(Ωₕ, pts)
	set_markers!(Ωₕ, domain_markers)
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