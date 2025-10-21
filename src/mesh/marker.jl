"""
	boundary_symbol_to_cartesian(indices::CartesianIndices)

Returns a named tuple connecting the facet labels of a set to the corresponding `CartesianIndices`.

# Examples

```jldoctest
julia> boundary_symbol_to_cartesian(CartesianIndices((1:3, 1:4)))
(left = CartesianIndices((1:1, 1:4)),
 right = CartesianIndices((3:3, 1:4)),
 top = CartesianIndices((1:3, 4:4)),
 bottom = CartesianIndices((1:3, 1:1)))
```

```jldoctest
julia> boundary_symbol_to_cartesian(CartesianIndices((1:10, 1:20, 1:15)))
(left = CartesianIndices((1:10, 1:1, 1:15)),
 right = CartesianIndices((1:10, 20:20, 1:15)),
 top = CartesianIndices((1:10, 1:20, 15:15)),
 bottom = CartesianIndices((1:10, 1:20, 1:1)),
 front = CartesianIndices((10:10, 1:20, 1:15)),
 back = CartesianIndices((1:1, 1:20, 1:15)))
```
"""
@inline boundary_symbol_to_cartesian(indices::CartesianIndices{1}) = (; :left => first(indices), :right => last(indices))

function boundary_symbol_to_cartesian(indices::CartesianIndices{2})
	N, M = size(indices)

	return (;
			:left => indices[1:1, 1:M],
			:right => indices[N:N, 1:M],
			:top => indices[1:N, M:M],
			:bottom => indices[1:N, 1:1])
end

function boundary_symbol_to_cartesian(indices::CartesianIndices{3})
	N, M, K = size(indices)

	return (;
			:left => indices[1:N, 1:1, 1:K],
			:right => indices[1:N, M:M, 1:K],
			:top => indices[1:N, 1:M, K:K],
			:bottom => indices[1:N, 1:M, 1:1],
			:front => indices[N:N, 1:M, 1:K],
			:back => indices[1:1, 1:M, 1:K])
end

"""
	boundary_symbol_to_dict(indices::CartesianIndices)

	Returns a dictionary connecting the facet labels of a set to the corresponding `CartesianIndices` (see [`boundary_symbol_to_cartesian`](@ref)).
"""
boundary_symbol_to_dict(indices::CartesianIndices) = Dict(pairs(boundary_symbol_to_cartesian(indices)))

"""
	$(TYPEDEF)

Efficient storage type for mesh markers as a `Dict` of `Symbols`. For each label, a BitVector is assigned that determines, for a given index, if the corresponding geometric point is identified by the marker.
"""
const MeshMarkers = Dict{Symbol,BitVector}

"""
	process_label_for_mesh!(npts, markers_mesh, set_labels)

Initializes boolean vectors for a given set of labels within the mesh markers dictionary.

For each label in `set_labels`, this function creates a `BitVector` of length `npts`
(the total number of points in the mesh), initializes it with all `false` values,
and assigns it to the corresponding key in the `markers_mesh` dictionary. This
prepares the storage for later marking which points belong to which labeled region.

# Arguments

  - `npts`: The total number of points in the mesh.
  - `markers_mesh`: The [MeshMarkers](@ref) dictionary to be modified in-place.
  - `set_labels`: An iterator or collection of `Symbol` labels to initialize.
"""
function process_label_for_mesh!(npts, markers_mesh::MeshMarkers, set_labels)
	# Iterate through each label provided (e.g., :inlet, :boundary, etc.).
	for label in set_labels
		# For each label, create a new boolean vector of the correct size,
		# initialized to all `false`, and add it to the dictionary.
		markers_mesh[label] = falses(npts)
	end
end

"""
	_init_mesh_markers(Ωₕ, domain_markers)

Internal helper function to create and initialize the [MeshMarkers](@ref) dictionary.

This function extracts all unique labels from the provided [DomainMarkers](@ref) object,
which can come from symbol-, tuple-, or condition-based markers. It then prepares
a [MeshMarkers](@ref) dictionary where each label is a key associated with a `BitVector`
of `false`s, ready to be populated.
"""
function _init_mesh_markers(Ωₕ::AbstractMeshType, domain_markers::DomainMarkers)
	# Create an empty dictionary to store the mesh markers.
	markers_mesh = MeshMarkers()
	# Get the total number of points in the mesh.
	npts = npoints(Ωₕ)

	# Initialize boolean vectors for each category of marker labels.
	process_label_for_mesh!(npts, markers_mesh, label_symbols(domain_markers))
	process_label_for_mesh!(npts, markers_mesh, label_tuples(domain_markers))
	process_label_for_mesh!(npts, markers_mesh, label_conditions(domain_markers))

	# Return the fully initialized (but empty) markers dictionary.
	return markers_mesh
end

"""
	set_markers!(Ωₕ::AbstractMeshType, domain_markers)

Populates the markers of a mesh `Ωₕ` based on a [DomainMarkers](@ref) object.

This is the main entry point for applying user-defined labels to the mesh points. It initializes the marker storage and then delegates to specialized functions for each type of marker identifier (symbols, tuples of symbols, and functions).
"""
function set_markers!(Ωₕ::AbstractMeshType, domain_markers)
	# Initialize a dictionary to hold the boolean vectors for each marker label.
	mesh_markers = _init_mesh_markers(Ωₕ, domain_markers)

	# Process markers identified by single symbols (e.g., :left) and tuples/sets (e.g., (:top, :right)).
	_set_markers_symbols!(mesh_markers, symbols(domain_markers), Ωₕ)
	_set_markers_symbols!(mesh_markers, tuples(domain_markers), Ωₕ)

	# Process markers identified by user-defined functions (level-set conditions).
	_set_markers_conditions!(mesh_markers, conditions(domain_markers), Ωₕ)

	# Assign the populated markers dictionary to the mesh object.
	Ωₕ.markers = mesh_markers
end

"""
	_mark_indices!(marker_set, linear_indices, indices_to_mark)

A utility function to efficiently update a boolean marker vector.

It sets the value to `true` at the linear positions corresponding to the `CartesianIndex` or collection of `CartesianIndices` provided in `indices_to_mark`.
"""
function _mark_indices!(marker_set::AbstractVector{Bool}, linear_indices, indices_to_mark)
	# Handle the case of a single CartesianIndex.
	if indices_to_mark isa CartesianIndex
		# Convert the CartesianIndex to a linear index and set the marker to true.
		marker_set[linear_indices[indices_to_mark]] = true
	else
		# Handle a collection of CartesianIndices (e.g., for a boundary face).
		for idx in indices_to_mark
			marker_set[linear_indices[idx]] = true
		end
	end
end

"""
	_set_markers_symbols!(mesh_markers, symbols, Ωₕ)

Processes markers that are identified by predefined symbols (e.g., `:left`, `:top`) or sets of those symbols.
"""
function _set_markers_symbols!(mesh_markers::MeshMarkers, symbols, Ωₕ)
	# Create a map from predefined boundary symbols to their corresponding CartesianIndices.
	symbol_to_index_map = boundary_symbol_to_dict(indices(Ωₕ))
	# Create a converter from Cartesian to linear indices for efficient array access.
	linear_indices = LinearIndices(npoints(Ωₕ, Tuple))

	# Iterate over each marker defined by a symbol or set of symbols.
	for marker in symbols
		@unpack label, identifier = marker

		# Get the boolean vector for the current marker label.
		target_marker_set = mesh_markers[label]

		# Case 1: The identifier is a single symbol (e.g., :left).
		if identifier isa Symbol
			# Look up the CartesianIndices for this boundary symbol.
			idxs = symbol_to_index_map[identifier]
			# Mark these indices in the boolean vector.
			_mark_indices!(target_marker_set, linear_indices, idxs)

			# Case 2: The identifier is a Set of symbols (e.g., Set([:top, :right])).
		elseif identifier isa Set
			# Iterate through each symbol in the set.
			for id in identifier
				# Look up the CartesianIndices for each symbol.
				idxs = symbol_to_index_map[id]
				# Mark the corresponding indices.
				_mark_indices!(target_marker_set, linear_indices, idxs)
			end
		end
	end
end

"""
	__process_condition!(mesh_marker, identifier, Ωₕ)

Core logic for evaluating a function-based (level-set) marker.

It iterates through every point in the mesh, evaluates the `identifier` function
at that point's coordinates, and sets the marker to `true` if the function returns `true`.
"""
function __process_condition!(mesh_marker, identifier, Ωₕ)
	linear_indices = LinearIndices(npoints(Ωₕ, Tuple))
	# Loop over every CartesianIndex in the mesh.
	for idx in indices(Ωₕ)
		# Check if the function `identifier(point)` returns true.
		if identifier(point(Ωₕ, idx))
			# If it does, mark this point's linear index as true.
			mesh_marker[linear_indices[idx]] = true
		end
	end
end

"""
	_set_markers_conditions!(mesh_markers, conditions, Ωₕ)

Iterates through all function-based markers and applies them to the mesh.
"""
function _set_markers_conditions!(mesh_markers::MeshMarkers, conditions, Ωₕ)
	# Loop through each marker that is defined by a condition (function).
	for marker in conditions
		@unpack label, identifier = marker
		# Call the helper function to evaluate the condition for the entire mesh.
		__process_condition!(mesh_markers[label], identifier, Ωₕ)
	end

	return nothing
end