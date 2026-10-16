# ghost_mesh.jl: `GhostMesh`, a mesh together with the ghost points that extend it past
# its boundary facets.
#
# The padded mesh is a real mesh with submeshes of its own, so every operator and the
# assembly walk read ghosts with no change; the wrapper forwards the mesh accessors to it
# and answers the face queries (`boundary_indices`, the surface weight and mask, the
# geometric markers) about the physical facets, as index-derived stored markers, never as
# predicates of position.
#
# The padded points follow `change_points!` of the physical mesh through the version stamp
# of its submeshes (the pattern of `SpaceWeights.built_version`): the padded mesh is
# re-padded on the access that finds it stale. `iterative_refinement!` of the physical
# mesh changes the point count, which a ghost mesh cannot follow, and throws.
#
# The walk reads a ghost mesh through its immutable `GhostMeshState` (`_walk_mesh`), as it
# reads a `MeshnD` through a `MeshnDState`: the two answer every accessor and ghost query
# alike (`_GhostMeshLike`), so a space rebuilt around the state in a `Polyester.@batch`
# task still takes the ghost methods.
#
# See also: `Mesh1D`, `MeshnD`, `MeshnDState`

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
padded point outside along two or more axes is absent (`Bramble.is_absent`) and has no
ghost number. `Bramble.ghost_index` and `Bramble.boundary_node` map in `O(1)` between a
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

"""
    GhostMeshState{D, M, P, CI} <: AbstractMeshType{D}

The immutable walk state of a [`GhostMesh`](@ref), built on demand by [`_walk_mesh`](@ref)
once the padded mesh is brought up to date with the physical one, so it is never stale: the
walk states of the physical and padded meshes, the padded indices and the facet tuples. It
answers every accessor and ghost query a `GhostMesh` does; its version, identity (`uid`) and
marker words are the padded state's. The state types `M` and `P` are parameters so a rebuilt
state can hold other array types ([`_batch_rebuild`](@ref)).
"""
struct GhostMeshState{D, M <: AbstractMeshType{D}, P <: AbstractMeshType{D},
    CI <: CartesianIndices{D}} <: AbstractMeshType{D}
    "the walk state of the physical mesh."
    physical::M
    "the walk state of the padded mesh."
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
    "the identity of the padded mesh."
    uid::UInt64
end

# Both a `GhostMesh` and its state answer the accessors and ghost queries below.
const _GhostMeshLike{D} = Union{GhostMesh{D}, GhostMeshState{D}}

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

# `n` entries of type `T` in the storage the backend `be` chooses for its vectors (`vector`
# of length 0, then `similar`: the element type is free, the vector type is the backend's).
@inline _backend_vector(be, ::Type{T}, n::Integer) where {T} = similar(vector(be, 0), T, n)

# The coordinates of the padded axis, on the host: a ghost beyond each named end, in the
# vector type of the physical mesh's host points.
function _padded_axis_points(x::AbstractVector, padlo::Bool, padhi::Bool)
    n = length(x)
    p = similar(x, n + padlo + padhi)
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
@inline _padded(s::GhostMeshState) = s.padded

# The state of `G`, taken after `_padded` brings the padded mesh up to date.
@inline function _walk_mesh(G::GhostMesh)
    p = _walk_mesh(_padded(G))
    return GhostMeshState(_walk_mesh(G.physical), p, G.indices, G.n, G.lo, G.active,
        G.offset, G.nghosts, p.uid)
end
@inline _walk_mesh(s::GhostMeshState) = s

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
@inline nghosts(G::_GhostMeshLike) = G.nghosts

"""
    has_ghosts(G::GhostMesh, d::Int, side::Symbol) -> Bool

Whether facet `(d, side)` (`side` is `:min` or `:max`) carries ghosts.
"""
@inline has_ghosts(G::_GhostMeshLike, d::Int, side::Symbol) = G.active[d][_ghost_side(side)]

"""
    physical_mesh(G::GhostMesh) -> Union{Mesh1D, MeshnD}

The physical mesh `G` pads.
"""
@inline physical_mesh(G::_GhostMeshLike) = G.physical

"""
    physical_indices(G::GhostMesh) -> CartesianIndices

The padded indices of the physical nodes, `CartesianIndices(npoints(physical_mesh(G), Tuple))`
shifted past the ghost layers below each axis.
"""
@inline physical_indices(G::_GhostMeshLike{D}) where {D} = CartesianIndices(
    ntuple(d -> (1 + G.lo[d]):(G.n[d] + G.lo[d]), Val(D)))

"""
    padded_index(G::GhostMesh, I::CartesianIndex) -> CartesianIndex

The padded index of the physical node `I`.
"""
@inline padded_index(G::_GhostMeshLike{D}, I::CartesianIndex{D}) where {D} = I + CartesianIndex(G.lo)

@inline _ghost_transverse(n::NTuple{D, Int}, d::Int) where {D} = ntuple(
    e -> e == d ? 1 : n[e], Val(D))

# The number of ghost of node `I` on facet `(d, s)`: the position of `I` in the facet's
# column-major order, after the ghosts of the facets before it.
@inline function _ghost_number(G::_GhostMeshLike{D}, I, d::Int, s::Int) where {D}
    K = CartesianIndex(ntuple(e -> e == d ? 1 : I[e], Val(D)))
    return G.offset[d][s] + LinearIndices(_ghost_transverse(G.n, d))[K]
end

"""
    ghost_index(G::GhostMesh, I, d::Int, side::Symbol) -> Int

The number `g ∈ 1:nghosts(G)` of the ghost beyond the node `I` (a tuple or a
`CartesianIndex` of the physical mesh) of facet `(d, side)`, with `side` `:min` or `:max`.
`O(1)`; the inverse of `Bramble.boundary_node`.

Throws an `ArgumentError` when the facet carries no ghosts, or `I` is not on it.
"""
@inline function ghost_index(G::_GhostMeshLike, I, d::Int, side::Symbol)
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
`g`; the inverse of `Bramble.ghost_index`. `I` is an index of the physical mesh.
"""
@inline function boundary_node(G::_GhostMeshLike{D}, g::Int) where {D}
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
@inline function is_absent(G::_GhostMeshLike{D}, J::CartesianIndex{D}) where {D}
    out = 0
    for d in 1:D
        out += (J[d] <= G.lo[d]) | (J[d] > G.lo[d] + G.n[d])
    end
    return out >= 2
end

# The step into the mesh along axis `d` from the physical node `J` (a padded index) of a
# facet normal to `d`: `+1` from the first physical node, `-1` from the last.
@inline _inward_step(G::_GhostMeshLike, J::CartesianIndex, d::Int) = J[d] == G.lo[d] + 1 ? 1 : -1

#------------------------------------------------------------------------------------------#
# Mesh accessors, forwarded to the padded mesh
#------------------------------------------------------------------------------------------#

@inline (G::_GhostMeshLike)(i) = _padded(G)(i)

# Every accessor of the mesh interface reads the padded mesh, or the padded state of a
# `GhostMeshState`; the marker table and stamps a walk state does not keep, the mesh only.
for f in (:points, :half_points, :spacings, :half_spacings, :forward_spacings, :host_points,
    :host_spacings, :host_half_spacings, :cell_measures, :hₘₐₓ, :hₘᵢₙ, :is_collapsed,
    :_marker_words, :_mesh_version, :set, :backend, :eltype)
    @eval @inline $f(G::_GhostMeshLike) = $f(_padded(G))
end

for f in (:_marker_ids, :_marker_stamp, :markers, :_predicate_labels,
    :_parent_predicate_labels)
    @eval @inline $f(G::GhostMesh) = $f(_padded(G))
end

for f in (:point, :half_point, :spacing, :forward_spacing, :half_spacing, :cell_measure,
    :spacing_for_derivative, :forward_spacing_for_derivative)
    @eval @inline $f(G::_GhostMeshLike, i) = $f(_padded(G), i)
end

@inline spacing(G::_GhostMeshLike, i, dim::Int) = spacing(_padded(G), i, dim)
@inline forward_spacing(G::_GhostMeshLike, i, dim::Int) = forward_spacing(_padded(G), i, dim)

@inline npoints(G::_GhostMeshLike) = prod(size(G.indices))
@inline npoints(G::_GhostMeshLike, ::Type{Tuple}) = size(G.indices)
@inline indices(G::_GhostMeshLike) = G.indices
@inline eltype(::Type{<:GhostMesh{D, M}}) where {D, M} = eltype(M)
@inline eltype(::Type{<:GhostMeshState{D, M}}) where {D, M} = eltype(M)

@inline Base.getindex(G::_GhostMeshLike, idx...) = getindex(_padded(G), idx...)

# A stored marker is a write into the padded mesh, which the next access reads back.
@inline _store_markers!(G::GhostMesh, mm, labels = Symbol[]) = _store_markers!(
    _padded(G), mm, labels)

#------------------------------------------------------------------------------------------#
# Facet queries: the physical facets, in the padded index space
#------------------------------------------------------------------------------------------#

@inline boundary_indices(G::_GhostMeshLike) = boundary_indices(physical_indices(G))

@inline function is_boundary_index(G::_GhostMeshLike{D}, idx) where {D}
    J = CartesianIndex(idx)
    return J in physical_indices(G) && is_boundary_index(physical_indices(G), J)
end

@inline interior_indices(G::_GhostMeshLike) = interior_indices(physical_indices(G))

# `:boundary` is the physical boundary and `:interior` the physical nodes off it: a ghost
# is neither.
function _ensure_geometric_markers!(
        mesh_markers::MeshMarkers, G::_GhostMeshLike; warn_marker_mismatch::Bool = true
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
@inline function _marked_face_mask(G::_GhostMeshLike{D}, marked_at) where {D}
    idxs = indices(G)
    faces = boundary_indices(G)
    return ntuple(Val(D)) do d
        (_face_is_marked(marked_at, idxs, faces[2d - 1], d),
            _face_is_marked(marked_at, idxs, faces[2d], d))
    end
end

@inline function _check_marked_on_faces(
        G::_GhostMeshLike{D}, m, labels, mask, caller) where {D}
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

@inline function _check_surface_is_thin(G::_GhostMeshLike{D}, mask) where {D}
    for d in 1:D
        mask[d][1] && mask[d][2] && G.n[d] <= 2 && _throw_surface_not_thin(d, G.n[d])
    end
    return nothing
end

# The weight of the physical mesh at the physical node `I` sits at its padded index `J`;
# a ghost and an absent point weigh nothing.
@inline function _surface_weight(
        G::_GhostMeshLike{D}, mask, J::CartesianIndex{D}) where {D}
    J in physical_indices(G) || return zero(eltype(G))
    return _surface_weight(G.physical, mask, J - CartesianIndex(G.lo))
end

#------------------------------------------------------------------------------------------#
# Display
#------------------------------------------------------------------------------------------#

function Base.show(io::IO, G::_GhostMeshLike{D}) where {D}
    print(io, nameof(typeof(G)), "{$(D)D, ", G.n, " pts + ", G.nghosts, " ghosts}")
    return nothing
end
