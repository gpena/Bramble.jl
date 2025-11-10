"""
	boundary_symbol_to_cartesian(indices::CartesianIndices)

Maps standard boundary symbols (:left, :right, :top, :bottom, :front, :back) to their
corresponding CartesianIndices on the mesh boundary.

# Returns

A NamedTuple with boundary symbols as keys and CartesianIndices as values:

  - 1D: `:left`, `:right` (single points)
  - 2D: `:left`, `:right`, `:top`, `:bottom` (faces)
  - 3D: All six faces of a rectangular prism

# Examples

```jldoctest
julia> boundary_symbol_to_cartesian(CartesianIndices((1:3, 1:4)))
(left = CartesianIndices((1:1, 1:4)),
 right = CartesianIndices((3:3, 1:4)),
 top = CartesianIndices((1:3, 4:4)),
 bottom = CartesianIndices((1:3, 1:1)))
```

See also: [`boundary_symbol_to_dict`](@ref), [`set_markers!`](@ref)
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
@inline function process_label_for_mesh!(npts, markers_mesh::MeshMarkers, set_labels)
	@inbounds for label in set_labels
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
	set_markers!(Ωₕ::AbstractMeshType, domain_markers::DomainMarkers)

Applies domain markers to mesh points, creating BitVector indicators for each label.

This function handles three types of markers:

 1. **Symbol markers**: Predefined boundary labels (:left, :right, etc.)
 2. **Tuple markers**: Collections of boundary symbols
 3. **Function markers**: Level-set conditions (including time-dependent functions)

# Notes

  - Time-dependent markers are evaluated at the current time
  - Markers are stored as BitVectors in the mesh's `markers` field
  - Existing markers are completely replaced

# Example

```julia
Ω = domain(interval(0, 1) × interval(0, 1),
		   :inlet => :left,
		   :outlet => :right,
		   :walls => (:top, :bottom),
		   :obstacle => x -> norm(x .- 0.5) < 0.2)
Ωₕ = mesh(Ω, (20, 20), (true, true))
# Ωₕ.markers now contains BitVectors for :inlet, :outlet, :walls, :obstacle
```

See also: [`DomainMarkers`](@ref), [`MeshMarkers`](@ref)    # Initialize a dictionary to hold the boolean vectors for each marker label.
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
@inline function _mark_indices!(marker_set::AbstractVector{Bool}, linear_indices, idx::CartesianIndex)
	@inbounds marker_set[linear_indices[idx]] = true
end

@inline function _mark_indices!(marker_set::AbstractVector{Bool}, linear_indices, indices_to_mark)
	@inbounds for idx in indices_to_mark
		marker_set[linear_indices[idx]] = true
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
	@inbounds for idx in indices(Ωₕ)
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