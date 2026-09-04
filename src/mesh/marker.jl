"""
    boundary_symbol_to_cartesian(indices::CartesianIndices{1}) -> NamedTuple
    boundary_symbol_to_cartesian(indices::CartesianIndices{2}) -> NamedTuple
    boundary_symbol_to_cartesian(indices::CartesianIndices{3}) -> NamedTuple

Map standard boundary symbols (`:left`, `:right`, `:top`, `:bottom`, `:front`, `:back`) to their
corresponding `CartesianIndices` on the mesh boundary.

# Returns

A `NamedTuple` with boundary symbols as keys and `CartesianIndices` as values:
  - 1D: `:left`, `:right` (single points)
  - 2D: `:left`, `:right`, `:top`, `:bottom` (faces)
  - 3D: All six faces of a rectangular prism (`:left`, `:right`, `:top`, `:bottom`, `:front`, `:back`)

# Examples

```jldoctest
julia> boundary_symbol_to_cartesian(CartesianIndices((1:3, 1:4)))
(left = CartesianIndices((1:1, 1:4)), right = CartesianIndices((3:3, 1:4)), top = CartesianIndices((1:3, 4:4)), bottom = CartesianIndices((1:3, 1:1)))
```

See also: [`boundary_symbol_to_dict`](@ref), [`set_markers!`](@ref).
"""
@inline boundary_symbol_to_cartesian(indices::CartesianIndices{1}) = (;
    :left => first(indices), :right => last(indices))

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
    boundary_symbol_to_dict(indices::CartesianIndices) -> Dict{Symbol, CartesianIndices}

Return a dictionary connecting the facet labels of a set to the corresponding `CartesianIndices`.

See also: [`boundary_symbol_to_cartesian`](@ref).
"""
function boundary_symbol_to_dict(indices::CartesianIndices)
    Dict(pairs(boundary_symbol_to_cartesian(indices)))
end

"""
    const MeshMarkers = Dict{Symbol, BitVector}

Dictionary mapping semantic marker symbols to boolean indicator vectors across mesh points.

For each label, a `BitVector` indicates whether the corresponding mesh point satisfies the marker.
"""
const MeshMarkers = Dict{Symbol, BitVector}

"""
    process_label_for_mesh!(npts::Integer, markers_mesh::MeshMarkers, set_labels) -> Nothing

Initialize boolean indicator vectors for a collection of marker labels within `markers_mesh`.

For each label in `set_labels`, assigns a `BitVector` of length `npts` initialized to `false`.

# Arguments

  - `npts`: Total number of grid points in the mesh.
  - `markers_mesh`: [`MeshMarkers`](@ref) dictionary modified in-place.
  - `set_labels`: Collection of `Symbol` labels to initialize.
"""
@inline function process_label_for_mesh!(npts, markers_mesh::MeshMarkers, set_labels)
    @inbounds for label in set_labels
        markers_mesh[label] = falses(npts)
    end
    return nothing
end

"""
    _init_mesh_markers(Ωₕ::AbstractMeshType, domain_markers::DomainMarkers) -> MeshMarkers

Internal helper function to construct and initialize the [`MeshMarkers`](@ref) dictionary.

Allocates `BitVector` storage initialized to `false` for every symbol, tuple, and condition
label defined in `domain_markers`.
"""
function _init_mesh_markers(Ωₕ::AbstractMeshType, domain_markers::DomainMarkers)
    markers_mesh = MeshMarkers()
    npts = npoints(Ωₕ)

    process_label_for_mesh!(npts, markers_mesh, label_symbols(domain_markers))
    process_label_for_mesh!(npts, markers_mesh, label_tuples(domain_markers))
    process_label_for_mesh!(npts, markers_mesh, label_conditions(domain_markers))

    return markers_mesh
end

"""
    set_markers!(Ωₕ::AbstractMeshType, domain_markers::DomainMarkers) -> Nothing

Evaluate domain markers onto mesh points, creating `BitVector` indicators for each label.

Supports three classes of domain markers:
  1. Symbol markers: predefined boundary labels (`:left`, `:right`, etc.).
  2. Tuple markers: unions of boundary symbols.
  3. Function markers: level-set boolean predicates `x -> Bool`.

Also seeds the default geometric markers `:boundary` and `:interior` if not already defined.

# Arguments

  - `Ωₕ`: Target mesh whose `markers` field is populated.
  - `domain_markers`: [`DomainMarkers`](@ref) containing semantic boundary or regional labels.

# Keywords

  - `warn_marker_mismatch::Bool = true`: whether to warn when a custom `:boundary`/`:interior`
    marker disagrees with the mesh's own geometric definition. The custom marker is kept
    either way; set to `false` to silence the warning for an intentional redefinition (see
    [`mesh`](@ref)).

# Examples

```julia
Ω = domain(interval(0, 1) × interval(0, 1),
           :inlet => :left,
           :outlet => :right,
           :walls => (:top, :bottom),
           :obstacle => x -> norm(x .- 0.5) < 0.2)
Ωₕ = mesh(Ω, (20, 20), (true, true))
# Ωₕ.markers contains BitVectors for :inlet, :outlet, :walls, :obstacle, :boundary, :interior
```

See also: [`DomainMarkers`](@ref), [`MeshMarkers`](@ref).
"""
function set_markers!(Ωₕ::AbstractMeshType, domain_markers; warn_marker_mismatch::Bool = true)
    mesh_markers = _init_mesh_markers(Ωₕ, domain_markers)

    _set_markers_symbols!(mesh_markers, symbols(domain_markers), Ωₕ)
    _set_markers_symbols!(mesh_markers, tuples(domain_markers), Ωₕ)
    _set_markers_conditions!(mesh_markers, conditions(domain_markers), Ωₕ)

    # `:boundary`/`:interior` are reserved, always-available markers; see note above _ensure_geometric_markers!.
    _ensure_geometric_markers!(mesh_markers, Ωₕ; warn_marker_mismatch)

    Ωₕ.markers = mesh_markers
    return nothing
end

#=
Every mesh carries :boundary and :interior, computed from the mesh's own geometry rather
than from user registrations: every other label depends on a domain(...) call naming it.
RegionRestriction's local_stencil (form/operators/restriction.jl) reads :interior as
"not :boundary"; ensuring :boundary exists guarantees well-defined complementary indexing.

:boundary is computed via boundary_symbol_to_dict (the same face ranges marked by
get_boundary_symbols) rather than is_boundary_index, which excludes degenerate (length-1)
axes. The face-based definition marks :left and :right consistently even for degenerate sets.
:interior is defined as the logical complement .!boundary_set.
=#

"""
    _ensure_geometric_markers!(mesh_markers::MeshMarkers, Ωₕ::AbstractMeshType) -> Nothing

Seed `:boundary` and `:interior` from the mesh's own geometry, preserving any existing
custom definitions registered under those names.

If a pre-existing custom marker with the same name disagrees with the geometric boundary,
a warning is issued because downstream operators (`restrict_to`) assume geometric semantics —
unless `warn_marker_mismatch` is `false`, for a caller that has deliberately redefined the
label and does not want to be told so on every mesh built from it.
"""
function _ensure_geometric_markers!(mesh_markers::MeshMarkers, Ωₕ::AbstractMeshType;
        warn_marker_mismatch::Bool = true)
    linear_indices = LinearIndices(npoints(Ωₕ, Tuple))
    boundary_set = falses(npoints(Ωₕ))
    for idxs in values(boundary_symbol_to_dict(indices(Ωₕ)))
        _mark_indices!(boundary_set, linear_indices, idxs)
    end

    _default_geometric_marker!(mesh_markers, :boundary, boundary_set, warn_marker_mismatch)
    _default_geometric_marker!(mesh_markers, :interior, .!boundary_set, warn_marker_mismatch)
    return nothing
end

function _default_geometric_marker!(mesh_markers::MeshMarkers, label::Symbol,
        geometric::BitVector, warn_marker_mismatch::Bool)
    if haskey(mesh_markers, label)
        mesh_markers[label] == geometric ||
            (warn_marker_mismatch && _warn_geometric_marker_mismatch(label))
    else
        mesh_markers[label] = geometric
    end
    return nothing
end

@noinline function _warn_geometric_marker_mismatch(label::Symbol)
    @warn ":$label is defined here to mean something other than the mesh's own geometric " *
          "$(label === :boundary ? "boundary" : "interior") (every boundary face for " *
          ":boundary, its complement for :interior). restrict_to(:$label, ...) and " *
          "innerₕ(...; markers = (:$label,)) will use this mesh's own definition, not the " *
          "geometric one; give the custom label a different name to avoid the ambiguity."
end

"""
    _mark_indices!(marker_set::AbstractVector{Bool}, linear_indices, indices_to_mark) -> Nothing

Utility function to update a boolean marker vector.

Sets entries to `true` at the linear positions corresponding to `indices_to_mark`.
"""
@inline function _mark_indices!(marker_set::AbstractVector{Bool}, linear_indices, idx::CartesianIndex)
    @inbounds marker_set[linear_indices[idx]] = true
    return nothing
end

@inline function _mark_indices!(marker_set::AbstractVector{Bool}, linear_indices, indices_to_mark)
    @inbounds for idx in indices_to_mark
        marker_set[linear_indices[idx]] = true
    end
    return nothing
end

"""
    _set_markers_symbols!(mesh_markers::MeshMarkers, symbols, Ωₕ::AbstractMeshType) -> Nothing

Process markers identified by predefined symbols (`:left`, `:top`, etc.) or collections of symbols.
"""
function _set_markers_symbols!(mesh_markers::MeshMarkers, symbols, Ωₕ)
    symbol_to_index_map = boundary_symbol_to_dict(indices(Ωₕ))
    linear_indices = LinearIndices(npoints(Ωₕ, Tuple))

    for marker in symbols
        (; label, identifier) = marker
        target_marker_set = mesh_markers[label]

        if identifier isa Symbol
            idxs = symbol_to_index_map[identifier]
            _mark_indices!(target_marker_set, linear_indices, idxs)
        elseif identifier isa Union{Set, Tuple}
            for id in identifier
                idxs = symbol_to_index_map[id]
                _mark_indices!(target_marker_set, linear_indices, idxs)
            end
        end
    end
    return nothing
end

"""
    __process_condition!(mesh_marker::BitVector, identifier, Ωₕ::AbstractMeshType) -> Nothing

Core logic for evaluating a function-based (level-set) marker predicate across all mesh points.
"""
function __process_condition!(mesh_marker, identifier, Ωₕ)
    linear_indices = LinearIndices(npoints(Ωₕ, Tuple))
    @inbounds for idx in indices(Ωₕ)
        if identifier(point(Ωₕ, idx))
            mesh_marker[linear_indices[idx]] = true
        end
    end
    return nothing
end

"""
    _set_markers_conditions!(mesh_markers::MeshMarkers, conditions, Ωₕ::AbstractMeshType) -> Nothing

Iterate through all function-based markers and evaluate them across the mesh.
"""
function _set_markers_conditions!(mesh_markers::MeshMarkers, conditions, Ωₕ)
    for marker in conditions
        (; label, identifier) = marker
        __process_condition!(mesh_markers[label], identifier, Ωₕ)
    end
    return nothing
end
