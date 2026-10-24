"""
    boundary_symbol_to_cartesian(indices::CartesianIndices{1}) -> NamedTuple
    boundary_symbol_to_cartesian(indices::CartesianIndices{2}) -> NamedTuple
    boundary_symbol_to_cartesian(indices::CartesianIndices{3}) -> NamedTuple

Map canonical boundary symbols (`:xmin`, `:xmax`, `:ymin`, `:ymax`, `:zmin`, `:zmax`) and legacy
viewpoint aliases (`:left`, `:right`, `:top`, `:bottom`, `:front`, `:back`) to their
corresponding `CartesianIndices` on the mesh boundary.

# Returns

A `NamedTuple` with boundary symbols as keys and `CartesianIndices` as values.
  - 1D has `:xmin`, `:xmax`, `:left`, `:right`
  - 2D has `:xmin`, `:xmax`, `:ymin`, `:ymax`, `:left`, `:right`, `:bottom`, `:top`
  - 3D has all six faces, `:xmin`, `:xmax`, `:ymin`, `:ymax`, `:zmin`, `:zmax`, `:back`, `:front`, `:left`, `:right`, `:bottom`, `:top`

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
    xmin, xmax = boundary_indices(indices)
    return (; :xmin => xmin, :xmax => xmax, :left => xmin, :right => xmax)
end

function boundary_symbol_to_cartesian(indices::CartesianIndices{2})
    xmin, xmax, ymin, ymax = boundary_indices(indices)
    return (;
        :xmin => xmin,
        :xmax => xmax,
        :ymin => ymin,
        :ymax => ymax,
        :left => xmin,
        :right => xmax,
        :bottom => ymin,
        :top => ymax
    )
end

function boundary_symbol_to_cartesian(indices::CartesianIndices{3})
    xmin, xmax, ymin, ymax, zmin, zmax = boundary_indices(indices)
    return (;
        :xmin => xmin,
        :xmax => xmax,
        :ymin => ymin,
        :ymax => ymax,
        :zmin => zmin,
        :zmax => zmax,
        :back => xmin,
        :front => xmax,
        :left => ymin,
        :right => ymax,
        :bottom => zmin,
        :top => zmax
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

The dictionary [`markers`](@ref)`(Ωₕ)` returns is a read-only view; see there for how to
change labels.
"""
const MeshMarkers = Dict{Symbol, BitVector}

"""
    process_label_for_mesh!(npts, markers_mesh::MeshMarkers, set_labels) -> Nothing

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
    _init_mesh_markers(Ωₕ::AbstractMeshType, domain_markers) -> MeshMarkers

Internal helper function to construct and initialize the [`MeshMarkers`](@ref) dictionary.

Allocates `BitVector` storage initialized to `false` for every symbol, tuple, and condition
label defined in `domain_markers`, a [`DomainMarkers`](@ref) or the `EvaluatedDomainMarkers`
that `Ω(t)` of a time-dependent domain holds, and for the three labels
of each polygon and level-set marker.
"""
function _init_mesh_markers(
        Ωₕ::AbstractMeshType, domain_markers::Union{DomainMarkers, EvaluatedDomainMarkers}
)
    markers_mesh = MeshMarkers()
    npts = npoints(Ωₕ)

    process_label_for_mesh!(npts, markers_mesh, label_symbols(domain_markers))
    process_label_for_mesh!(npts, markers_mesh, label_tuples(domain_markers))
    process_label_for_mesh!(npts, markers_mesh, label_conditions(domain_markers))
    for m in regions(domain_markers)
        process_label_for_mesh!(npts, markers_mesh, _polygon_labels(m))
    end

    return markers_mesh
end

"""
    set_markers!(Ωₕ::AbstractMeshType, domain_markers; warn_marker_mismatch = true) -> Nothing

Evaluate domain markers onto mesh points, creating `BitVector` indicators for each label.

Supports five classes of domain markers:
  1. Symbol markers, the predefined boundary labels (`:left`, `:right`, etc.).
  2. Tuple markers, the unions of boundary symbols.
  3. Function markers, the level-set boolean predicates `x -> Bool`.
  4. Polygon markers `:L => polygon(V)`, which split the nodes of a 2D mesh into `L`
     (inside), `L_boundary` and `L_exterior` (outside); see [`markers`](@ref) for the
     boundary tolerance.
  5. Level-set markers `:L => levelset(φ)`, which give the same three labels. Their
     labels come from the polygon that [`mesh`](@ref) inscribes in the curve ``φ = 0`` when
     it fits the grid, so `set_markers!` alone refuses a level-set marker (below).

Also seeds the default geometric markers `:boundary` and `:interior` if not already defined.

# Arguments

  - `Ωₕ`: Target mesh whose `markers` field is populated.
  - `domain_markers`: a [`DomainMarkers`](@ref) or the evaluated markers of `Ω(t)`, holding
    the semantic boundary or regional labels.

# Keywords

  - `warn_marker_mismatch::Bool = true`: whether to warn when a custom `:boundary`/`:interior`
    marker disagrees with the mesh's own geometric definition. The custom marker is kept
    either way; set to `false` to silence the warning for an intentional redefinition (see
    [`mesh`](@ref)).

# Throws

  - `ArgumentError`: if `domain_markers` holds a level-set marker, whose labels only
    [`mesh`](@ref) can set: the grid lines and the inscribed polygon are fitted together,
    and neither is kept. This covers `change_points!` and `iterative_refinement!` of such a
    mesh too.

# Examples

```julia
Ω = domain(interval(0, 1) × interval(0, 1),
           :inlet => :left,
           :outlet => :right,
           :walls => (:top, :bottom),
           :obstacle => x -> norm(x .- 0.5) < 0.2)
Ωₕ = mesh(Ω, (20, 20), (true, true))
# markers(Ωₕ) contains BitVectors for :inlet, :outlet, :walls, :obstacle, :boundary, :interior
```

See also: [`DomainMarkers`](@ref), [`MeshMarkers`](@ref).
"""
function set_markers!(Ωₕ::AbstractMeshType, domain_markers; warn_marker_mismatch::Bool = true)
    fitted = map(_ -> nothing, regions(domain_markers))
    return _set_markers!(Ωₕ, domain_markers, fitted; warn_marker_mismatch)
end

# `set_markers!` with the polygon `mesh` fitted to each region marker of `domain_markers`
# (`fitted`, one entry per region, `nothing` for a polygon marker, whose own polygon labels).
function _set_markers!(Ωₕ::AbstractMeshType, domain_markers, fitted::Tuple;
        warn_marker_mismatch::Bool = true)
    mesh_markers = _init_mesh_markers(Ωₕ, domain_markers)

    _set_markers_symbols!(mesh_markers, symbols(domain_markers), Ωₕ)
    _set_markers_symbols!(mesh_markers, tuples(domain_markers), Ωₕ)
    _set_markers_conditions!(mesh_markers, conditions(domain_markers), Ωₕ)
    _set_markers_regions!(mesh_markers, regions(domain_markers), fitted, Ωₕ)

    # `:boundary`/`:interior` are reserved, always-available markers; see note above _ensure_geometric_markers!.
    _ensure_geometric_markers!(mesh_markers, Ωₕ; warn_marker_mismatch)

    # The labels a predicate of position or a polygon defines are remembered: moving the
    # points leaves them stale, which `change_points!(Ωₕ, pts)` refuses rather than lets pass.
    position_labels = Symbol[m.label for m in conditions(domain_markers)]
    for m in regions(domain_markers)
        append!(position_labels, _polygon_labels(m))
    end
    _store_markers!(Ωₕ, mesh_markers, position_labels)
    return nothing
end

#=
The marker word matrix (gpena/Bramble.jl#437). A mesh keeps its public `MeshMarkers`
dictionary, whose `BitVector`s `markers(Ωₕ)` and `index_in_marker` return as they are, and
beside it a label-to-column table and one `Matrix{UInt64}` holding the same bits: column
`_marker_id(Ωₕ, label)` is that label's `BitVector` chunks. Only the matrix and the integer
ids cross a task boundary. All three are built together by `_marker_table`, whenever the
label set is stored (`set_markers!`, `markers!`); the matrix is never resized in place.
=#

"""
    _marker_table(mesh_markers::MeshMarkers, npts::Int) -> Tuple{Dict{Symbol, Int}, Matrix{UInt64}}

The label-to-column table of `mesh_markers`, labels sorted, and the word matrix holding
each label's `BitVector` chunks in its column, `cld(npts, 64)` words per column.
"""
function _marker_table(mesh_markers::MeshMarkers, npts::Int)
    labels = sort!(collect(keys(mesh_markers)))
    nwords = cld(npts, 64)
    ids = Dict{Symbol, Int}()
    words = zeros(UInt64, nwords, length(labels))
    for (j, label) in enumerate(labels)
        ids[label] = j
        chunks = mesh_markers[label].chunks
        copyto!(words, (j - 1) * nwords + 1, chunks, 1, min(nwords, length(chunks)))
    end
    return ids, words
end

# The word matrix of a mesh with no labels yet.
@inline _no_marker_words(npts::Int) = zeros(UInt64, cld(npts, 64), 0)

"""
    _marker_words(Ωₕ::AbstractMeshType) -> Matrix{UInt64}

The marker word matrix of `Ωₕ` (or of its state): word × label, column
[`_marker_id`](@ref)`(Ωₕ, label)` holding the chunks of `markers(Ωₕ)[label]`.
"""
function _marker_words end

"""
    _marker_id(Ωₕ::AbstractMeshType, label::Symbol) -> Int

The column of `label` in [`_marker_words`](@ref)`(Ωₕ)`. Reads the label exactly as stored:
no boundary alias is resolved, unlike [`index_in_marker`](@ref).

# Throws
- `KeyError`: `Ωₕ` has no marker `label`.
"""
@inline function _marker_id(Ωₕ::AbstractMeshType, label::Symbol)
    id = get(_marker_ids(Ωₕ), label, 0)
    id == 0 && _throw_unknown_marker_label(Ωₕ, label)
    return id
end

#=
Every mesh carries `:boundary` and `:interior`, computed from the mesh's own geometry rather
than from user registrations, while every other label depends on a domain(...) call naming it.
RegionRestriction's local_stencil (operators/region_restriction.jl) reads `:interior` as
"not `:boundary`", and ensuring `:boundary` exists guarantees well-defined complementary indexing.

`:boundary` is the union of `boundary_indices`, the face ranges, rather than
`is_boundary_index`, which skips length-1 axes, so a degenerate set still marks both ends.
`:interior` is defined as the logical complement `.!boundary_set`.
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

#=
A polygon marker classifies each node by its signed distance s to the polygon's boundary:
on the boundary when |s| <= τ, inside when s < -τ, outside when s > τ. The tolerance
τ = max(√ε hₘᵢₙ, 10³ ε L) + 4 ε M follows the mesh spacing hₘᵢₙ, so a node a rounding error
off an edge (a moved point, a refined grid) is still on it, and scales with the polygon's
extent L. The 10³ ε L term keeps τ above the rounding of `signed_distance` itself when a
sliver cell makes hₘᵢₙ tiny, where √ε hₘᵢₙ alone falls below it and boundary nodes are lost.

ε is the larger machine epsilon of the points' and the vertices' types. A `Float32` node
meant to sit on a `Float64` edge, or a `Float64` node on a `Float32` edge, is a `Float32`
rounding off it, which a `Float64` ε would not cover. That rounding is up to ε |x| / 2 and
grows with the coordinates, not with hₘᵢₙ or L: on [1000, 1001]² it reaches 3e-5 against
10³ ε L ≈ 2.4e-5 for an extent of 0.2. The 4 ε M term, M the largest coordinate magnitude of
the polygon's vertices, covers it; on that grid it is 5e-4, far below the spacing 0.01. M
leaves out the grid: a node within τ of an edge has coordinates about the vertices' size,
and a far grid corner would widen τ around a small polygon and swallow its nodes.
=#

"""
    _set_markers_regions!(mesh_markers::MeshMarkers, regions, fitted::Tuple, Ωₕ::AbstractMeshType) -> Nothing

Mark the interior, boundary and exterior nodes of every polygon and level-set marker in
`regions`. `fitted` holds, per region, the polygon `mesh` inscribed in a level-set curve
(`nothing` for a polygon marker, which labels by its own polygon).
"""
function _set_markers_regions!(mesh_markers::MeshMarkers, regions, fitted::Tuple, Ωₕ)
    isempty(regions) && return nothing
    length(fitted) == length(regions) || throw(DimensionMismatch("one fitted polygon per region"))
    pts = host_points(Ωₕ)
    for (marker, fit) in zip(regions, fitted)
        Li, Lb, Le = _polygon_labels(marker)
        inside, on, outside = mesh_markers[Li], mesh_markers[Lb], mesh_markers[Le]
        _label_region!(inside, on, outside, identifier(marker), fit, Ωₕ, pts, label(marker))
    end
    return nothing
end

# A polygon labels the nodes by the signed distance to it.
function _label_region!(inside, on, outside, P::Polygon, ::Nothing, Ωₕ, pts, _)
    linear_indices = LinearIndices(npoints(Ωₕ, Tuple))
    τ = _polygon_tolerance(P, pts)
    @inbounds for idx in indices(Ωₕ)
        k = linear_indices[idx]
        s = signed_distance(P, _condition_point(pts, idx))
        if abs(s) <= τ
            on[k] = true
        elseif s < 0
            inside[k] = true
        else
            outside[k] = true
        end
    end
    return nothing
end

# A level set labels the nodes by the polygon fitted to it.
_label_region!(inside, on, outside, ::LevelSet, P::Polygon, _, pts, _) = _label_fitted!(
    inside, on, outside, P, pts)

@noinline function _label_region!(_, _, _, ::LevelSet, ::Nothing, _, _, label)
    return _throw_levelset_refusal(label)
end

@noinline function _throw_levelset_refusal(label)
    throw(ArgumentError("the level-set marker :$label labels the nodes by the polygon that " *
                        "mesh(Ω, npts, unif) fits to its curve together with the grid lines, " *
                        "and no mesh keeps that polygon: build the mesh with mesh(Ω, npts, " *
                        "unif) instead of set_markers!, change_points! or iterative_refinement!."))
end

# Refuses a level-set marker before a caller mutates its mesh: `iterative_refinement!` and
# `change_points!` refine or move the points first and mark them after, so the refusal of
# `_label_region!` would come too late and leave the mesh changed (#866, #891).
function _check_no_levelset_regions(domain_markers)
    for m in regions(domain_markers)
        identifier(m) isa LevelSet && _throw_levelset_refusal(label(m))
    end
    return nothing
end

# Labels the nodes of the lines `pts` by the polygon `P` whose vertices are nodes: a vertex
# is a boundary node, any other node is inside or outside by the parity of the polygon edges
# above it. Consecutive vertices are corners of one cell, so a vertical ray up from a node of
# column `i` crosses (half-open: one end at a larger x) only edges with an end on column `i`.
# Linear in the nodes and vertices, where the signed distance is their product.
function _label_fitted!(inside, on, outside, P::Polygon, pts::NTuple{2, AbstractVector})
    X, Y = pts
    nx, ny = length(X), length(Y)
    V = P.vertices
    ij = Vector{NTuple{2, Int}}(undef, length(V))
    for (k, v) in enumerate(V)
        i, j = searchsortedfirst(X, v[1]), searchsortedfirst(Y, v[2])
        (i <= nx && j <= ny && X[i] == v[1] && Y[j] == v[2]) || _throw_fitted_off_node(v)
        ij[k] = (i, j)
    end
    toggle = falses(nx, ny)
    for k in eachindex(ij)
        (i, j), (i₂, j₂) = ij[k], ij[mod1(k + 1, length(ij))]
        i == i₂ && continue
        iₑ, jₑ = i < i₂ ? (i, j) : (i₂, j₂)
        toggle[iₑ, jₑ] = !toggle[iₑ, jₑ]
    end
    onpoly = falses(nx, ny)
    for (i, j) in ij
        onpoly[i, j] = true
    end
    @inbounds for i in 1:nx
        above = false
        for j in ny:-1:1
            k = i + (j - 1) * nx
            if onpoly[i, j]
                on[k] = true
            elseif above
                inside[k] = true
            else
                outside[k] = true
            end
            above ⊻= toggle[i, j]
        end
    end
    return nothing
end

@noinline function _throw_fitted_off_node(v)
    throw(ArgumentError("the polygon fitted to the level set has the vertex $v, which is not " *
                        "a node of the mesh's lines."))
end

# The boundary tolerance of polygon `P` on a grid with axis points `pts`: see the note above.
# An axis with a single point has no spacing; with none at all, hₘᵢₙ is the extent.
function _polygon_tolerance(P::Polygon, pts::NTuple{2, AbstractVector})
    T = float(promote_type(eltype(pts[1]), eltype(pts[2]), eltype(eltype(P.vertices))))
    ε = T(max(eps(float(eltype(pts[1]))), eps(float(eltype(pts[2]))),
        eps(float(eltype(eltype(P.vertices))))))
    V = P.vertices
    L = T(max(maximum(first, V) - minimum(first, V), maximum(last, V) - minimum(last, V)))
    h = typemax(T)
    for x in pts
        for i in 2:length(x)
            h = min(h, T(x[i] - x[i - 1]))
        end
    end
    isinf(h) && (h = L)
    M = T(maximum(v -> max(abs(v[1]), abs(v[2])), V))
    return max(sqrt(ε) * h, 1000 * ε * L) + 4 * ε * M
end
