# ghost_mesh.jl: `GhostMesh`, a mesh together with the ghost points that extend it past
# its boundary facets, for the finite-difference relations of the ghost-point plan
# (v4.0.0).
#
# A `GhostMesh` wraps the physical `Mesh1D`/`MeshnD` and a padded one of the same kind
# with a layer of ghost points beyond each named facet, placed at `x₁ - h₂` and `xₙ + hₙ`
# (the cached `spacing(Ωₕ, 2)` and `forward_spacing(Ωₕ, n)`). The padded mesh is a real
# mesh with submeshes of its own, so every operator and the assembly walk read ghosts with
# no change; the wrapper forwards the mesh accessors to it, and answers the face queries
# (`boundary_indices`, the surface weight and mask, the geometric markers) about the
# physical facets, as index-derived stored markers, never as predicates of position.
#
# Ghost DOF order: the named facets in the order `xmin, xmax, ymin, ymax, zmin, zmax`, each
# facet's ghosts in the column-major order of its nodes. A padded point outside along two
# or more axes (a 2D corner, a 3D edge or corner) is absent and is never a ghost.
#
# The padded points follow `change_points!` of the physical mesh through the version stamp
# of its submeshes (the pattern of `SpaceWeights.built_version`): the padded mesh is
# re-padded on the access that finds it stale. `iterative_refinement!` of the physical
# mesh changes the point count, which a ghost mesh cannot follow, and throws.
#
# See also: `Mesh1D`, `MeshnD`

"""
    GhostMesh{D, M, P, CI} <: AbstractMeshType{D}

A physical mesh with one ghost point beyond each node of the facets it names.

[`GhostMesh`](@ref)`(Ωₕ; ghosts = :boundary)` pads `Ωₕ` on every boundary facet,
`ghosts = (:xmin, :ymax)` only on the named ones (canonical names or their aliases),
`ghosts = ()` on none. The ghost beyond node `x₁` sits at `x₁ - h₂` and the one beyond `xₙ`
at `xₙ + hₙ`, read from the physical mesh's points whenever they are asked for, so the mesh
follows `change_points!` of `Ωₕ`.

Its points, spacings and indices are those of the padded mesh, `npoints(G, Tuple)` the
padded sizes and `G(i)` the padded axis submesh; the labels and facet queries
(`boundary_indices`, `index_in_marker`, the surface weights) name the physical nodes. The
ghosts are numbered `1:nghosts(G)`: facets in the order `xmin, xmax, ymin, ymax, zmin,
zmax` (named ones only), each facet's ghosts in the column-major order of its nodes. A
padded point outside along two or more axes is absent ([`is_absent`](@ref)) and has no
ghost number. [`ghost_index`](@ref) and [`boundary_node`](@ref) map in `O(1)` between a
facet node and its ghost.

After `iterative_refinement!` of `Ωₕ` the ghost mesh is stale and every access throws an
`ArgumentError`: build a new one.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, true))
G = Bramble.GhostMesh(Ωₕ; ghosts = :boundary)
npoints(G, Tuple)            # (7, 9)
Bramble.nghosts(G)           # 24
```
"""
mutable struct GhostMesh{D, M <: AbstractMeshType{D}, P <: AbstractMeshType{D},
    CI <: CartesianIndices{D}} <: AbstractMeshType{D}
    "the physical mesh."
    physical::M
    "the padded mesh, a real mesh of its own with the physical labels on its facet nodes."
    padded::P
    "the `CartesianIndices` of the padded grid."
    indices::CI
    "the physical point counts."
    n::NTuple{D, Int}
    "the ghost layers below each axis, 0 or 1: the padded index of physical node `I` is `I + lo`."
    lo::NTuple{D, Int}
    "`active[d][s]`: facet `(d, s)` carries ghosts (`s` is 1 for the minimum, 2 for the maximum)."
    active::NTuple{D, NTuple{2, Bool}}
    "the ghosts of facet `(d, s)` are numbered `offset[d][s] + 1` onwards."
    offset::NTuple{D, NTuple{2, Int}}
    "the number of ghosts."
    nghosts::Int
    "the physical mesh version the padded points were set at."
    version::Int
    "the physical mesh's marker stamp the padded markers were built at."
    marker_stamp::UInt64
end

#------------------------------------------------------------------------------------------#
# Construction
#------------------------------------------------------------------------------------------#

const _GHOST_SIDES = (:min, :max)

@inline function _ghost_side(side::Symbol)
    side === :min && return 1
    side === :max && return 2
    throw(ArgumentError("side must be :min or :max, got :$side"))
end

# The facets `ghosts` names as a per-axis `(min, max)` mask: `:boundary` is all of them, a
# `Symbol` or a tuple of them names canonical faces or their aliases.
function _ghost_facets(::Val{D}, ghosts) where {D}
    names = ghosts isa Symbol ? (ghosts,) : Tuple(ghosts)
    mask = falses(D, 2)
    for s in names
        s isa Symbol || _throw_bad_ghosts(Val(D), s)
        if s === :boundary
            mask .= true
        else
            face = _find_face(Val(D), s)
            face === nothing && _throw_bad_ghosts(Val(D), s)
            mask[face[1], face[2]] = true
        end
    end
    return ntuple(d -> (mask[d, 1], mask[d, 2]), Val(D))
end

@noinline function _throw_bad_ghosts(::Val{D}, s) where {D}
    faces = join((":$c" for (c, _, _, _) in _boundary_faces(Val(D))), ", ")
    throw(
        ArgumentError(
        "ghosts must be :boundary or a tuple of facet names, got $(repr(s)). Name facets " *
        "among $faces (or their aliases), or pass `ghosts = ()` for none.",
    ),
    )
end

@noinline function _throw_ghost_axis_too_short(d::Int, n::Int)
    throw(
        ArgumentError(
        "axis $d has $n point(s) and a ghost beyond a facet of it is placed from the first " *
        "two (or last two) points. Refine the axis to at least two points, or leave its " *
        "facets out of `ghosts`.",
    ),
    )
end

# The coordinates of the padded axis, on the host: a ghost beyond each named end.
function _padded_axis_points(x::AbstractVector, padlo::Bool, padhi::Bool)
    n = length(x)
    p = Vector{eltype(x)}(undef, n + padlo + padhi)
    copyto!(p, 1 + padlo, x, 1, n)
    padlo && (p[1] = x[1] - (x[2] - x[1]))
    padhi && (p[end] = x[n] + (x[n] - x[n - 1]))
    return p
end

# The padded axis submesh of `Ωd`, a mesh of its own with the points of `p`.
function _padded_axis(Ωd::Mesh1D, p::AbstractVector, padded::Bool)
    padded || return copy(Ωd)
    ax = mesh(domain(interval(first(p), last(p))), length(p), false; backend = backend(Ωd))
    _change_points!(ax, p)
    return ax
end

# The mesh of the padded axes: the axis itself in 1D, a `MeshnD` over them otherwise.
_padded_mesh(::Mesh1D, axes::Tuple{<:Mesh1D}, idxs) = only(axes)

function _padded_mesh(::MeshnD, axes::Tuple, idxs)
    return MeshnD(foldl(×, map(set, axes)), MeshMarkers(), idxs, backend(first(axes)), axes)
end

function GhostMesh(Ωₕ::Union{Mesh1D, MeshnD}; ghosts = :boundary)
    D = dim(Ωₕ)
    active = _ghost_facets(Val(D), ghosts)
    n = npoints(Ωₕ, Tuple)
    for d in 1:D
        (active[d][1] || active[d][2]) && n[d] < 2 && _throw_ghost_axis_too_short(d, n[d])
    end
    lo = ntuple(d -> Int(active[d][1]), Val(D))
    psize = ntuple(d -> n[d] + Int(active[d][1]) + Int(active[d][2]), Val(D))
    counts = [active[d][s] ? prod(_ghost_transverse(n, d)) : 0 for s in 1:2, d in 1:D]
    starts = cumsum([0; vec(counts)])
    offset = ntuple(d -> (starts[2d - 1], starts[2d]), Val(D))
    idxs = generate_indices(psize)
    pts = ntuple(d -> _padded_axis_points(host_points(Ωₕ(d)), active[d]...), Val(D))
    subs = ntuple(d -> _padded_axis(Ωₕ(d), pts[d], active[d][1] || active[d][2]), Val(D))
    padded = _padded_mesh(Ωₕ, subs, idxs)
    G = GhostMesh{D, typeof(Ωₕ), typeof(padded), typeof(idxs)}(
        Ωₕ, padded, idxs, n, lo, active, offset, starts[end],
        _mesh_version(Ωₕ), _marker_stamp(Ωₕ))
    _store_markers!(padded, _padded_markers(G))
    return G
end

#------------------------------------------------------------------------------------------#
# Staleness
#------------------------------------------------------------------------------------------#

# The padded mesh, brought up to date with the physical one: its points after a
# `change_points!`, its labels after a marker change.
@inline function _padded(G::GhostMesh)
    p = G.physical
    (_mesh_version(p) == G.version && _marker_stamp(p) == G.marker_stamp) || _refresh!(G)
    return G.padded
end

@noinline function _refresh!(G::GhostMesh{D}) where {D}
    p = G.physical
    npoints(p, Tuple) == G.n || _throw_ghost_mesh_stale(G, npoints(p, Tuple))
    if _mesh_version(p) != G.version
        for d in 1:D
            x = _padded_axis_points(host_points(p(d)), G.active[d]...)
            _change_points!(G.padded(d), x)
        end
        G.version = _mesh_version(p)
    end
    if _marker_stamp(p) != G.marker_stamp
        _store_markers!(G.padded, _padded_markers(G))
        G.marker_stamp = _marker_stamp(p)
    end
    return nothing
end

@noinline function _throw_ghost_mesh_stale(G::GhostMesh, now)
    throw(
        ArgumentError(
        "the ghost mesh was built on a mesh of $(G.n) points, which now has $now: the " *
        "physical mesh was refined or resized in place (iterative_refinement!), and a " *
        "ghost mesh keeps its point count. Build a new ghost mesh, or a new space with " *
        "`gridspace(Ωₕ; ghosts = ...)`, on the refined mesh.",
    ),
    )
end

# The labels of the physical mesh on the padded index space, ghost points unmarked, and
# `:boundary` and `:interior` where it holds none.
function _padded_markers(G::GhostMesh)
    mm = MeshMarkers()
    plin = LinearIndices(size(G.indices))
    phys = CartesianIndices(G.n)
    shift = CartesianIndex(G.lo)
    for (label, bits) in markers(G.physical)
        out = falses(length(plin))
        k = findfirst(bits)
        while k !== nothing
            out[plin[phys[k] + shift]] = true
            k = findnext(bits, k + 1)
        end
        mm[label] = out
    end
    _ensure_geometric_markers!(mm, G; warn_marker_mismatch = false)
    return mm
end

#------------------------------------------------------------------------------------------#
# Ghosts: counts and the maps between a facet node and its ghost
#------------------------------------------------------------------------------------------#

"""
    nghosts(G::GhostMesh) -> Int

The number of ghosts of `G`: the nodes of its named facets, one ghost each.
"""
@inline nghosts(G::GhostMesh) = G.nghosts

"""
    has_ghosts(G::GhostMesh, d::Int, side::Symbol) -> Bool

Whether facet `(d, side)` (`side` is `:min` or `:max`) carries ghosts.
"""
@inline has_ghosts(G::GhostMesh, d::Int, side::Symbol) = G.active[d][_ghost_side(side)]

"""
    physical_mesh(G::GhostMesh) -> Union{Mesh1D, MeshnD}

The physical mesh `G` pads.
"""
@inline physical_mesh(G::GhostMesh) = G.physical

"""
    physical_indices(G::GhostMesh) -> CartesianIndices

The padded indices of the physical nodes, `CartesianIndices(npoints(physical_mesh(G), Tuple))`
shifted past the ghost layers below each axis.
"""
@inline physical_indices(G::GhostMesh{D}) where {D} = CartesianIndices(
    ntuple(d -> (1 + G.lo[d]):(G.n[d] + G.lo[d]), Val(D)))

"""
    padded_index(G::GhostMesh, I::CartesianIndex) -> CartesianIndex

The padded index of the physical node `I`.
"""
@inline padded_index(G::GhostMesh{D}, I::CartesianIndex{D}) where {D} = I + CartesianIndex(G.lo)

@inline _ghost_transverse(n::NTuple{D, Int}, d::Int) where {D} = ntuple(
    e -> e == d ? 1 : n[e], Val(D))

# The number of ghost of node `I` on facet `(d, s)`: the position of `I` in the facet's
# column-major order, after the ghosts of the facets before it.
@inline function _ghost_number(G::GhostMesh{D}, I, d::Int, s::Int) where {D}
    K = CartesianIndex(ntuple(e -> e == d ? 1 : I[e], Val(D)))
    return G.offset[d][s] + LinearIndices(_ghost_transverse(G.n, d))[K]
end

"""
    ghost_index(G::GhostMesh, I, d::Int, side::Symbol) -> Int

The number `g ∈ 1:nghosts(G)` of the ghost beyond the node `I` (a tuple or a
`CartesianIndex` of the physical mesh) of facet `(d, side)`, with `side` `:min` or `:max`.
`O(1)`; the inverse of [`boundary_node`](@ref).

Throws an `ArgumentError` when the facet carries no ghosts, or `I` is not on it.
"""
@inline function ghost_index(G::GhostMesh, I, d::Int, side::Symbol)
    s = _ghost_side(side)
    G.active[d][s] || _throw_facet_without_ghosts(d, side)
    I[d] == (s == 1 ? 1 : G.n[d]) || _throw_node_off_facet(I, d, side)
    return _ghost_number(G, I, d, s)
end

@noinline function _throw_facet_without_ghosts(d::Int, side::Symbol)
    throw(
        ArgumentError(
        "facet $(d), $(repr(side)) carries no ghosts on this mesh. Name it in " *
        "`GhostMesh(Ωₕ; ghosts = ...)`, or use `ghosts = :boundary`.",
    ),
    )
end

@noinline function _throw_node_off_facet(I, d::Int, side::Symbol)
    throw(
        ArgumentError(
        "node $(Tuple(I)) is not on facet $(d), $(repr(side)): its index along axis $d " *
        "must be $(side === :min ? "1" : "the last point").",
    ),
    )
end

"""
    boundary_node(G::GhostMesh, g::Int) -> (I::CartesianIndex, d::Int, side::Symbol)

The facet node, the axis of its facet and the side (`:min` or `:max`) whose ghost is number
`g`; the inverse of [`ghost_index`](@ref). `I` is an index of the physical mesh.
"""
@inline function boundary_node(G::GhostMesh{D}, g::Int) where {D}
    1 <= g <= G.nghosts || _throw_ghost_out_of_range(g, G.nghosts)
    for d in 1:D, s in 1:2

        G.active[d][s] || continue
        r = g - G.offset[d][s]
        ts = _ghost_transverse(G.n, d)
        if 1 <= r <= prod(ts)
            K = CartesianIndices(ts)[r]
            I = CartesianIndex(ntuple(e -> e == d ? (s == 1 ? 1 : G.n[d]) : K[e], Val(D)))
            return I, d, @inbounds _GHOST_SIDES[s]
        end
    end
    return _throw_ghost_out_of_range(g, G.nghosts)
end

@noinline function _throw_ghost_out_of_range(g::Int, n::Int)
    throw(ArgumentError("ghost $g is out of range: the mesh has $n ghosts (1:$n)"))
end

"""
    is_absent(G::GhostMesh, J::CartesianIndex) -> Bool

Whether the padded index `J` is an absent point: outside the physical mesh along two or
more axes (a 2D corner, a 3D edge or corner of the padded grid), which is no node and no
ghost.
"""
@inline function is_absent(G::GhostMesh{D}, J::CartesianIndex{D}) where {D}
    out = 0
    for d in 1:D
        out += (J[d] <= G.lo[d]) | (J[d] > G.lo[d] + G.n[d])
    end
    return out >= 2
end

# The step into the mesh along axis `d` from the physical node `J` (a padded index) of a
# facet normal to `d`: `+1` from the first physical node, `-1` from the last.
@inline _inward_step(G::GhostMesh, J::CartesianIndex, d::Int) = J[d] == G.lo[d] + 1 ? 1 : -1

#------------------------------------------------------------------------------------------#
# Mesh accessors, forwarded to the padded mesh
#------------------------------------------------------------------------------------------#

@inline (G::GhostMesh)(i) = _padded(G)(i)

# Every accessor of the mesh interface reads the padded mesh.
for f in (:points, :half_points, :spacings, :half_spacings, :forward_spacings, :host_points,
    :host_spacings, :host_half_spacings, :cell_measures, :hₘₐₓ, :hₘᵢₙ, :is_collapsed,
    :_marker_words, :_marker_ids, :_marker_stamp, :_mesh_version, :_walk_mesh, :markers,
    :set, :backend, :_predicate_labels, :_parent_predicate_labels, :eltype)
    @eval @inline $f(G::GhostMesh) = $f(_padded(G))
end

for f in (:point, :half_point, :spacing, :forward_spacing, :half_spacing, :cell_measure,
    :spacing_for_derivative, :forward_spacing_for_derivative)
    @eval @inline $f(G::GhostMesh, i) = $f(_padded(G), i)
end

@inline spacing(G::GhostMesh, i, dim::Int) = spacing(_padded(G), i, dim)
@inline forward_spacing(G::GhostMesh, i, dim::Int) = forward_spacing(_padded(G), i, dim)

@inline npoints(G::GhostMesh) = prod(size(G.indices))
@inline npoints(G::GhostMesh, ::Type{Tuple}) = size(G.indices)
@inline indices(G::GhostMesh) = G.indices
@inline eltype(::Type{<:GhostMesh{D, M}}) where {D, M} = eltype(M)

@inline Base.getindex(G::GhostMesh, idx...) = getindex(_padded(G), idx...)

# A stored marker is a write into the padded mesh, which the next access reads back.
@inline _store_markers!(G::GhostMesh, mm, labels = Symbol[]) = _store_markers!(
    _padded(G), mm, labels)

#------------------------------------------------------------------------------------------#
# Facet queries: the physical facets, in the padded index space
#------------------------------------------------------------------------------------------#

@inline boundary_indices(G::GhostMesh) = boundary_indices(physical_indices(G))

@inline function is_boundary_index(G::GhostMesh{D}, idx) where {D}
    J = CartesianIndex(idx)
    return J in physical_indices(G) && is_boundary_index(physical_indices(G), J)
end

@inline interior_indices(G::GhostMesh) = interior_indices(physical_indices(G))

# `:boundary` is the physical boundary and `:interior` the physical nodes off it: a ghost
# is neither.
function _ensure_geometric_markers!(
        mesh_markers::MeshMarkers, G::GhostMesh; warn_marker_mismatch::Bool = true
)
    linear = LinearIndices(size(G.indices))
    boundary_set = falses(length(linear))
    for idxs in boundary_indices(G)
        _mark_indices!(boundary_set, linear, idxs)
    end
    interior_set = falses(length(linear))
    _mark_indices!(interior_set, linear, physical_indices(G))
    interior_set .&= .!boundary_set
    _default_geometric_marker!(mesh_markers, :boundary, boundary_set, warn_marker_mismatch)
    _default_geometric_marker!(mesh_markers, :interior, interior_set, warn_marker_mismatch)
    return nothing
end

# `_marked_face_mask` of the physical faces, the points numbered in the padded index space.
@inline function _marked_face_mask(G::GhostMesh{D}, marked_at) where {D}
    idxs = indices(G)
    faces = boundary_indices(G)
    return ntuple(Val(D)) do d
        (_face_is_marked(marked_at, idxs, faces[2d - 1], d),
            _face_is_marked(marked_at, idxs, faces[2d], d))
    end
end

@inline function _check_marked_on_faces(
        G::GhostMesh{D}, m, labels, mask, caller) where {D}
    cis = indices(G)
    shift = CartesianIndex(G.lo)
    for s in labels
        b = get(m, s, nothing)
        b === nothing && continue
        k = findfirst(b)
        while k !== nothing
            _on_mask_face(mask, cis[k] - shift, G.n) || _throw_markers_not_faces(m, labels, caller)
            k = findnext(b, k + 1)
        end
    end
    return nothing
end

@inline function _check_surface_is_thin(G::GhostMesh{D}, mask) where {D}
    for d in 1:D
        mask[d][1] && mask[d][2] && G.n[d] <= 2 && _throw_surface_not_thin(d, G.n[d])
    end
    return nothing
end

# The weight of the physical mesh at the physical node `I` sits at its padded index `J`;
# a ghost and an absent point weigh nothing.
@inline function _surface_weight(
        G::GhostMesh{D}, mask, J::CartesianIndex{D}) where {D}
    J in physical_indices(G) || return zero(eltype(G))
    return _surface_weight(G.physical, mask, J - CartesianIndex(G.lo))
end

#------------------------------------------------------------------------------------------#
# Display
#------------------------------------------------------------------------------------------#

function Base.show(io::IO, G::GhostMesh{D}) where {D}
    print(io, "GhostMesh{$(D)D, ", G.n, " pts + ", G.nghosts, " ghosts}")
    return nothing
end
