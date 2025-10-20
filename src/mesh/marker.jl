"""
	boundary_symbol_to_cartesian(indices::CartesianIndices)

	Returns a named tuple connecting the facet labels of a set to the corresponding `CartesianIndices`.

# Examples

```julia
boundary_symbol_to_cartesian(CartesianIndices((1:3, 1:4)))
(left = CartesianIndices((1:1, 1:4)),
 right = CartesianIndices((3:3, 1:4)),
 top = CartesianIndices((1:3, 4:4)),
 bottom = CartesianIndices((1:3, 1:1)))
```

```julia
boundary_symbol_to_cartesian(CartesianIndices((1:10, 1:20, 1:15)))
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

	Returns a dictionary connecting the facet labels of a set to the corresponding `CartesianIndices`. See [boundary_symbol_to_cartesian](@ref)
"""
boundary_symbol_to_dict(indices::CartesianIndices) = Dict(pairs(boundary_symbol_to_cartesian(indices)))

"""
	MeshMarkers

Efficient storage type for mesh markers as a `Dict` of `Symbols`. For each label, a BitVector is assigned that determines, for a given index, if the corresponding geometric point is identified by the marker.
"""
const MeshMarkers = Dict{Symbol,BitVector}

function process_label_for_mesh!(npts, markers_mesh::MeshMarkers, set_labels)
	for label in set_labels
		markers_mesh[label] = falses(npts)
	end
end

function _init_mesh_markers(Ωₕ::AbstractMeshType{D}, domain_markers::DomainMarkers) where D
	markers_mesh = MeshMarkers()
	npts = npoints(Ωₕ)

	process_label_for_mesh!(npts, markers_mesh, label_symbols(domain_markers))
	process_label_for_mesh!(npts, markers_mesh, label_tuples(domain_markers))
	process_label_for_mesh!(npts, markers_mesh, label_conditions(domain_markers))

	return markers_mesh
end

"""
	set_markers!(Ωₕ::AbstractMeshType, domain_markers::DomainMarkers)

Populates the marker index collections of mesh Ωₕ based on boundary symbols or geometric conditions defined in the [DomainMarkers](@ref).
"""
function set_markers!(Ωₕ::AbstractMeshType, domain_markers)
	mesh_markers = _init_mesh_markers(Ωₕ, domain_markers)

	_set_markers_symbols!(mesh_markers, symbols(domain_markers), Ωₕ)
	_set_markers_symbols!(mesh_markers, tuples(domain_markers), Ωₕ)
	_set_markers_conditions!(mesh_markers, conditions(domain_markers), Ωₕ)

	Ωₕ.markers = mesh_markers
end

function _mark_indices!(marker_set::AbstractVector{Bool}, linear_indices, indices_to_mark)
	if indices_to_mark isa CartesianIndex
		marker_set[linear_indices[indices_to_mark]] = true
	else
		for idx in indices_to_mark
			marker_set[linear_indices[idx]] = true
		end
	end
end

function _set_markers_symbols!(mesh_markers::MeshMarkers, symbols, Ωₕ)
	symbol_to_index_map = boundary_symbol_to_dict(indices(Ωₕ))
	linear_indices = LinearIndices(npoints(Ωₕ, Tuple))

	for marker in symbols
		@unpack label, identifier = marker

		target_marker_set = mesh_markers[label]

		if identifier isa Symbol
			idxs = symbol_to_index_map[identifier]
			_mark_indices!(target_marker_set, linear_indices, idxs)

		elseif identifier isa Set
			for id in identifier
				idxs = symbol_to_index_map[id]
				_mark_indices!(target_marker_set, linear_indices, idxs)
			end
		end
	end
end

function __process_condition!(mesh_marker, identifier, Ωₕ)
	linear_indices = LinearIndices(npoints(Ωₕ, Tuple))
	for idx in indices(Ωₕ)
		if identifier(point(Ωₕ, idx))
			mesh_marker[linear_indices[idx]] = true
		end
	end
end

function _set_markers_conditions!(mesh_markers::MeshMarkers, conditions, Ωₕ)
	for marker in conditions
		@unpack label, identifier = marker
		__process_condition!(mesh_markers[label], identifier, Ωₕ)
	end

	return nothing
end