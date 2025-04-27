"""
	 MeshType{D}

Abstract type for meshes. Meshes are only parametrized by their topological dimension `D``.
"""
abstract type MeshType{D} <: BrambleType end

"""
	struct MarkerIndices{D}
		c_index::Set{CartesianIndex{D}}
		c_indices::Set{CartesianIndices{D}}
	end

	Structure to hold sets of individual `CartesianIndex` or `CartesianIndices`.
"""
struct MarkerIndices{D,CartIndexType<:CartesianIndex{D},CartIndicesType} <: BrambleType
	c_index::Set{CartIndexType}
	c_indices::Set{CartIndicesType}
end

"""
	boundary_symbol_to_cartesian(indices::CartesianIndices{D}) where D

	Returns a dictionary connecting the facet labels of a set to the corresponding `CartesianIndices`.

```@example
julia> boundary_symbol_to_cartesian(CartesianIndices((1:10,)))
Dict{Symbol, CartesianIndex{1}} with 2 entries:
  :left  => CartesianIndex(1,)
  :right => CartesianIndex(10,)
```

```@example
julia> boundary_symbol_to_cartesian(CartesianIndices((1:10,)))
Dict{Symbol, CartesianIndex{1}} with 2 entries:
  :left  => CartesianIndex(1,)
  :right => CartesianIndex(10,)
```

```@example
julia> boundary_symbol_to_cartesian(CartesianIndices((1:10, 1:20)))
Dict{Symbol, CartesianIndices{2, R} where R<:Tuple{OrdinalRange{Int64, Int64}, OrdinalRange{Int64, Int64}}} with 4 entries:
  :left   => CartesianIndices((1:1, 1:20))
  :right  => CartesianIndices((10:10, 1:20))
  :bottom => CartesianIndices((1:10, 1:1))
  :top    => CartesianIndices((1:10, 20:20))
```

```@example
julia> boundary_symbol_to_cartesian(CartesianIndices((1:10, 1:20, 1:15)))
Dict{Symbol, CartesianIndices{3, R} where R<:Tuple{OrdinalRange{Int64, Int64}, OrdinalRange{Int64, Int64}, OrdinalRange{Int64, Int64}}} with 6 entries:
  :left   => CartesianIndices((1:10, 1:1, 1:15))
  :right  => CartesianIndices((1:10, 20:20, 1:15))
  :bottom => CartesianIndices((1:10, 1:20, 1:1))
  :top    => CartesianIndices((1:10, 1:20, 15:15))
  :front  => CartesianIndices((10:10, 1:20, 1:15))
  :back   => CartesianIndices((1:1, 1:20, 1:15))
```
"""
@inline boundary_symbol_to_cartesian(indices::CartesianIndices{1}) = Dict(:left => first(indices), :right => last(indices))

function boundary_symbol_to_cartesian(indices::CartesianIndices{2})
	N, M = size(indices)

	dict_2d = Dict{Symbol,CartesianIndices{2}}()
	dict_2d[:left] = CartesianIndices((1:1, 1:M))
	dict_2d[:right] = CartesianIndices((N:N, 1:M))

	dict_2d[:top] = CartesianIndices((1:N, M:M))
	dict_2d[:bottom] = CartesianIndices((1:N, 1:1))

	return dict_2d
end

function boundary_symbol_to_cartesian(indices::CartesianIndices{3})
	N, M, K = size(indices)
	dict_3d = Dict{Symbol,CartesianIndices{3}}()
	dict_3d[:left] = CartesianIndices((1:N, 1:1, 1:K))
	dict_3d[:right] = CartesianIndices((1:N, M:M, 1:K))

	dict_3d[:top] = CartesianIndices((1:N, 1:M, K:K))
	dict_3d[:bottom] = CartesianIndices((1:N, 1:M, 1:1))

	dict_3d[:front] = CartesianIndices((N:N, 1:M, 1:K))
	dict_3d[:back] = CartesianIndices((1:1, 1:M, 1:K))

	return dict_3d
end

"""
	merge_consecutive_indices!(marker_data::MarkerIndices{D}) where D

Finds sequences of `CartesianIndex{D}` elements consecutive along any single
axis within `marker_data.c_index`. Removes these sequences (if longer than one element)
and adds the corresponding `CartesianIndices{D}` range object to
`marker_data.c_indices`.
"""
function merge_consecutive_indices!(marker_data::MarkerIndices{D}; check_consistency = true) where D
	# --- Save Initial State (only if checking consistency) ---
	initial_indices_copy = check_consistency ? copy(marker_data.c_index) : Set{CartesianIndex{D}}()
	# We assume initial marker_data.c_indices should be preserved and added to.
	# The check focuses on conserving the points from the initial c_index set.

	# --- Main Merging Logic ---
	remaining_indices = copy(marker_data.c_index)
	if isempty(remaining_indices)
		# Consistency check is trivial if input was empty
		if check_consistency && !isempty(initial_indices_copy)
			# This case should ideally not happen if logic is correct
			error("Internal logic error: Input c_index was empty but initial copy wasn't.")
		end
		# Optional: Add a print statement if consistency check passes trivially
		# if check_consistency; println("Consistency check passed (empty input)."); end
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
	# Update c_index: Remove indices that were successfully merged in this run
	setdiff!(marker_data.c_index, merged_indices_in_run)
	# Add the newly found multi-element blocks to the c_indices set
	union!(marker_data.c_indices, output_ranges)

	# --- Consistency Check ---
	if check_consistency
		# Reconstruct the total set of points represented *after* the merge operation.
		# Start with the indices remaining in c_index
		reconstructed_indices = copy(marker_data.c_index)

		# Add all indices contained within the *newly added* ranges
		# Note: We are checking the conservation of points *originating* from the
		# initial c_index set. If marker_data.c_indices had pre-existing ranges,
		# this check doesn't include them, which is correct for verifying this function's action.
		for block in output_ranges
			# Efficiently add all elements from the CartesianIndices iterator
			union!(reconstructed_indices, Set(block))
			# Alternative (potentially less memory for huge blocks, maybe slower):
			# for idx in block
			#     push!(reconstructed_indices, idx)
			# end
		end

		# Compare the reconstructed set with the initial set
		if reconstructed_indices != initial_indices_copy
			# --- Detailed Debug Output ---
			println("Consistency Check FAILED!")
			println("Initial index count:      ", length(initial_indices_copy))
			println("Reconstructed index count:", length(reconstructed_indices))

			lost_indices = setdiff(initial_indices_copy, reconstructed_indices)
			gained_indices = setdiff(reconstructed_indices, initial_indices_copy)

			if !isempty(lost_indices)
				println("Lost indices (present initially, missing finally):")
				display(lost_indices) # Use display for potentially large sets
				# Consider limiting output: display(collect(Iterators.take(lost_indices, 10)))
			end
			if !isempty(gained_indices)
				println("Gained indices (present finally, missing initially):")
				display(gained_indices)
				# Consider limiting output: display(collect(Iterators.take(gained_indices, 10)))
			end

			# You might want to inspect the state just before the error
			println("\nState before error:")
			println("Initial c_index copy: ")
			display(initial_indices_copy)
			println("Final c_index: ")
			display(marker_data.c_index)
			println("Final added c_indices (output_ranges): ")
			display(output_ranges)
			println("Reconstructed indices: ")
			display(reconstructed_indices)

			error("Consistency check failed: The set of indices after merging does not match the initial set.")
		end
	end # end if check_consistency

	return nothing
end

"""
	MeshMarkers{D}

Type of dictionary to store the `CartesianIndices` associated with a [MarkerIndices](@ref).
"""
MeshMarkers{D} = Dict{Symbol,MarkerIndices{D,CartesianIndex{D},CartesianIndices{D,NTuple{D,UnitRange{Int}}}}}

function Base.show(io::IO, markers::MeshMarkers{D}) where D
	labels = collect(keys(markers))
	labels_styled_combined = color_markers(labels)

	fields = D == 1 ? ("Markers",) : ("Resolution",)
	mlength = max_length_fields(fields)

	final_output = style_field("Markers", labels_styled_combined, max_length = mlength)
	print(io, final_output)
end

"""
	dim(Ωₕ::MeshType)
	dim(::Type{<:MeshType})

Returns the dimension of the space where `Ωₕ` is embedded.
"""
@inline dim(_::MeshType{D}) where D = D
@inline dim(::Type{<:MeshType{D}}) where D = D

"""
	topo_dim(Ωₕ::MeshType)

Returns the topological dimension `Ωₕ`.
"""
@inline @generated function topo_dim(Ωₕ::MeshType{D}) where D
	if D <= 0
		return :(0) # Handle edge case
	end

	term_expression = :((npoints(Ωₕ(i)) == 1) ? 0 : dim(Ωₕ(i)))
	generated_code = :(sum(Base.Cartesian.@ntuple $D i->$term_expression))
	return generated_code
end

"""
	indices(Ωₕ::MeshType)

Returns the `CartesianIndices` associated with the points of mesh `Ωₕ`.
"""
@inline indices(Ωₕ::MeshType) = Ωₕ.indices

@inline backend(Ωₕ::MeshType) = Ωₕ.backend
@inline markers(Ωₕ::MeshType) = Ωₕ.markers

"""
	set_indices!(Ωₕ::MeshType, indices)

	Overrides the indices in Ωₕ.
"""
@inline set_indices!(Ωₕ::MeshType, indices) = (Ωₕ.indices = indices)

"""
	marker(Ωₕ::MeshType, str::Symbol)

Returns the [Marker](@ref) function with label `str`.
"""
@inline marker(Ωₕ::MeshType, symbol::Symbol) = Ωₕ.markers[symbol]

function process_label_for_mesh!(markers_mesh::MeshMarkers{D}, set_labels) where D
	c_indices_type = CartesianIndices{D,NTuple{D,UnitRange{Int}}}
	c_index_type = CartesianIndex{D}

	for label in set_labels
		markers_mesh[label] = MarkerIndices{D,c_index_type,c_indices_type}(Set{c_index_type}(), Set{c_indices_type}())
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
	set_markers!(Ωₕ::Mesh1D, domain_markers::DomainMarkers)

Populates the marker index collections of `Mesh1D` Ωₕ based on boundary symbols or geometric conditions defined in the `Domain` Ω, applied to the `Mesh1D` Ωₕ.
"""
function set_markers!(Ωₕ::MeshType, domain_markers)
	D = dim(Ωₕ)
	mesh_indices = indices(Ωₕ)

	mesh_markers = _init_mesh_markers(Ωₕ, domain_markers)
	symbol_to_index_map = boundary_symbol_to_cartesian(mesh_indices)

	for marker in symbols(domain_markers)
		@unpack label, identifier = marker
		target_indices = D == 1 ? mesh_markers[label].c_index : mesh_markers[label].c_indices

		push!(target_indices, symbol_to_index_map[identifier])
	end

	for marker in tuples(domain_markers)
		@unpack label, identifier = marker
		target_indices = D == 1 ? mesh_markers[label].c_index : mesh_markers[label].c_indices

		for sym in identifier
			push!(target_indices, symbol_to_index_map[sym])
		end
	end

	for marker in conditions(domain_markers)
		@unpack label, identifier = marker
		for idx in mesh_indices
			if identifier(points(Ωₕ, idx))
				push!(mesh_markers[label].c_index, idx)
			end
		end

		merge_consecutive_indices!(mesh_markers[label])
	end

	Ωₕ.markers = mesh_markers
end

#@inline _i2p(pts, idx) = pts[idx]
#=
"""
	_i2p(pts::NTuple{D, Vector{T}}, index::CartesianIndex{D})

Returns a `D` tuple with the coordinates of the point in `pts` associated with the `CartesianIndex` given by ìndex`.
"""
@inline @generated _i2p(pts::NTuple{D,Vector{T}}, index::CartesianIndex{D}) where {D,T} = :(Base.Cartesian.@ntuple $D i->pts[i][index[i]])

# necessary?!
@inline @generated _i2ppo(pts::NTuple{D,Vector{T}}, index::CartesianIndex{D}) where {D,T} = :(Base.Cartesian.@ntuple $D i->pts[i][index[i] + 1])
=#
"""
	mesh(Ω::Domain, npts::Int, unif::Bool)
	mesh(Ω::Domain, npts::NTuple{D}, unif::NTuple{D})

Returns a [Mesh1D](@ref) or a [MeshnD](@ref) (``D=2,3``) defined on the [Domain](@ref) `Ω`. The number of points for each coordinate projection mesh are given in the tuple `npts`. The distribution of points on the submeshes are encoded in the tuple `unif`.

For future reference, the mesh points are denoted as

	- 1D mesh, with `npts` = ``N_x``

```math
x_i, \\, i=1,\\dots,N.
```

  - 2D mesh, with `npts` = (``N_x``, ``N_y``)

```math
(x_i,y_j), \\, i=1,\\dots,N_x, \\, j=1,\\dots,N_y
```

  - 3D mesh, with `npts` = (``N_x``, ``N_y``, ``N_z``)

```math
(x_i,y_j,z_l), \\, i=1,\\dots,N_x, \\, j=1,\\dots,N_y, \\, l=1,\\dots,N_z.
```

# Examples

```@example
julia> I = interval(0, 1);
	   Ωₕ = mesh(domain(I), 10, true);
1D mesh
nPoints: 10
Markers: Dirichlet
```

```@example
julia> X = domain(interval(0, 1) × interval(4, 5));
	   Ωₕ = mesh(X, (10, 15), (true, false));
2D mesh
nPoints: 150
Markers: ["Dirichlet"]

Submeshes:
  x direction | nPoints: 10
  y direction | nPoints: 15
```
"""
@inline mesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}; backend = Backend()) where D = _mesh(Ω, npts, unif, backend)
@inline mesh(Ω::Domain{CartesianProduct{1,T}}, npts::Int, unif::Bool; backend = Backend()) where T = _mesh(Ω, (npts,), (unif,), backend)