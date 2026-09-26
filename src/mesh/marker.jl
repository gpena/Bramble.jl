"""
    boundary_symbol_to_cartesian(indices::CartesianIndices{1}) -> NamedTuple
    boundary_symbol_to_cartesian(indices::CartesianIndices{2}) -> NamedTuple
    boundary_symbol_to_cartesian(indices::CartesianIndices{3}) -> NamedTuple

Map canonical boundary symbols (`:xmin`, `:xmax`, `:ymin`, `:ymax`, `:zmin`, `:zmax`) and legacy
viewpoint aliases (`:left`, `:right`, `:top`, `:bottom`, `:front`, `:back`) to their
corresponding `CartesianIndices` on the mesh boundary.

# Returns

A `NamedTuple` with boundary symbols as keys and `CartesianIndices` as values:
  - 1D: `:xmin`, `:xmax`, `:left`, `:right`
  - 2D: `:xmin`, `:xmax`, `:ymin`, `:ymax`, `:left`, `:right`, `:bottom`, `:top`
  - 3D: All six faces: `:xmin`, `:xmax`, `:ymin`, `:ymax`, `:zmin`, `:zmax`, `:back`, `:front`, `:left`, `:right`, `:bottom`, `:top`

# Examples

```jldoctest
faces = boundary_symbol_to_cartesian(CartesianIndices((1:3, 1:4)))
faces.xmin == CartesianIndices((1:1, 1:4)) && faces.ymax == CartesianIndices((1:3, 4:4)) &&
    faces.left === faces.xmin && faces.top === faces.ymax

# output

true
```

See also: [`boundary_symbol_to_dict`](@ref), [`set_markers!`](@ref).
"""
@inline function boundary_symbol_to_cartesian(indices::CartesianIndices{1})
    N = length(indices)
    xmin = indices[1:1]
    xmax = indices[N:N]
    return (; :xmin => xmin, :xmax => xmax, :left => xmin, :right => xmax)
end

function boundary_symbol_to_cartesian(indices::CartesianIndices{2})
    N, M = size(indices)

    return (;
        :xmin => indices[1:1, 1:M],
        :xmax => indices[N:N, 1:M],
        :ymin => indices[1:N, 1:1],
        :ymax => indices[1:N, M:M],
        :left => indices[1:1, 1:M],
        :right => indices[N:N, 1:M],
        :bottom => indices[1:N, 1:1],
        :top => indices[1:N, M:M]
    )
end

function boundary_symbol_to_cartesian(indices::CartesianIndices{3})
    N, M, K = size(indices)

    return (;
        :xmin => indices[1:1, 1:M, 1:K],
        :xmax => indices[N:N, 1:M, 1:K],
        :ymin => indices[1:N, 1:1, 1:K],
        :ymax => indices[1:N, M:M, 1:K],
        :zmin => indices[1:N, 1:M, 1:1],
        :zmax => indices[1:N, 1:M, K:K],
        :back => indices[1:1, 1:M, 1:K],
        :front => indices[N:N, 1:M, 1:K],
        :left => indices[1:N, 1:1, 1:K],
        :right => indices[1:N, M:M, 1:K],
        :bottom => indices[1:N, 1:M, 1:1],
        :top => indices[1:N, 1:M, K:K]
    )
end

"""
    boundary_symbol_to_dict(indices::CartesianIndices) -> Dict{Symbol, CartesianIndices}

Return a dictionary connecting the facet labels of a set to the corresponding `CartesianIndices`.

See also: [`boundary_symbol_to_cartesian`](@ref).
"""
function boundary_symbol_to_dict(indices::CartesianIndices)
    return Dict(pairs(boundary_symbol_to_cartesian(indices)))
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
RegionRestriction's local_stencil (ast/operators/restriction.jl) reads :interior as
"not :boundary"; ensuring :boundary exists guarantees well-defined complementary indexing.

:boundary is computed via boundary_symbol_to_cartesian (the same face ranges marked by
boundary_symbols) rather than is_boundary_index, which excludes degenerate (length-1)
axes. The face-based definition marks :left and :right consistently even for degenerate sets.
:interior is defined as the logical complement .!boundary_set.
=#

"""
    _ensure_geometric_markers!(mesh_markers::MeshMarkers, Ωₕ::AbstractMeshType) -> Nothing

Seed `:boundary` and `:interior` from the mesh's own geometry, preserving any existing
custom definitions registered under those names.

If a pre-existing custom marker with the same name disagrees with the geometric boundary,
a warning is issued because downstream operators (`restrict_to`) assume geometric semantics,
unless `warn_marker_mismatch` is `false`, for a caller that has deliberately redefined the
label and does not want to be told so on every mesh built from it.
"""
function _ensure_geometric_markers!(
        mesh_markers::MeshMarkers, Ωₕ::AbstractMeshType; warn_marker_mismatch::Bool = true
)
    linear_indices = LinearIndices(npoints(Ωₕ, Tuple))
    boundary_set = falses(npoints(Ωₕ))
    for idxs in boundary_indices(Ωₕ)
        _mark_indices!(boundary_set, linear_indices, idxs)
    end

    _default_geometric_marker!(mesh_markers, :boundary, boundary_set, warn_marker_mismatch)
    _default_geometric_marker!(
        mesh_markers, :interior, .!boundary_set, warn_marker_mismatch
    )
    return nothing
end

function _default_geometric_marker!(
        mesh_markers::MeshMarkers,
        label::Symbol,
        geometric::BitVector,
        warn_marker_mismatch::Bool
)
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
@inline function _mark_indices!(
        marker_set::AbstractVector{Bool}, linear_indices, indices_to_mark
)
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
    boundary_lookup = boundary_symbol_to_cartesian(indices(Ωₕ))
    linear_indices = LinearIndices(npoints(Ωₕ, Tuple))

    for marker in symbols
        (; label, identifier) = marker
        target_marker_set = mesh_markers[label]

        if identifier isa Symbol
            idxs = boundary_lookup[identifier]
            _mark_indices!(target_marker_set, linear_indices, idxs)
        elseif identifier isa Union{Set, Tuple}
            for id in identifier
                idxs = boundary_lookup[id]
                _mark_indices!(target_marker_set, linear_indices, idxs)
            end
        end
    end
    return nothing
end

"""
    __process_condition!(mesh_marker::BitVector, identifier, Ωₕ::AbstractMeshType) -> Nothing

Core logic for evaluating a function-based (level-set) marker predicate across all mesh points.

Reads coordinates once through [`host_points`](@ref) rather than [`point`](@ref)`(Ωₕ, idx)`
per point, so `identifier` -- an arbitrary user predicate that must keep running on the
host -- is evaluated against a host-resident array whether `Ωₕ` itself is device-backed or
not (gpena/Bramble.jl#309).
"""
function __process_condition!(mesh_marker, identifier, Ωₕ)
    pts = host_points(Ωₕ)
    linear_indices = LinearIndices(npoints(Ωₕ, Tuple))
    @inbounds for idx in indices(Ωₕ)
        if identifier(_condition_point(pts, idx))
            mesh_marker[linear_indices[idx]] = true
        end
    end
    return nothing
end

"""
    _condition_point(pts, idx) -> Union{Real, NTuple}

Read the coordinate at `idx` from `pts`, [`host_points`](@ref)`(Ωₕ)`'s return value.

A `Mesh1D` gives a `Vector` and `idx` is a `CartesianIndex{1}`; a `MeshnD` gives the
per-axis `NTuple{D, Vector}` and `idx` is a `CartesianIndex{D}`, assembled here into the
coordinate tuple the same way [`point`](@ref)`(Ωₕ::MeshnD, idx)` does.
"""
@inline _condition_point(pts::AbstractVector, idx::CartesianIndex{1}) = @inbounds pts[idx[1]]

@inline function _condition_point(
        pts::NTuple{D, AbstractVector}, idx::CartesianIndex{D}
) where {D}
    return ntuple(d -> (@inbounds pts[d][idx[d]]), Val(D))
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
