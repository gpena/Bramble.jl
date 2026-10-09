"""
    Mesh1DState{BT, CI, VT, T, WT} <: AbstractMeshType{1}

The immutable state of a [`Mesh1D`](@ref): its geometry, version, uniformity flag and
marker words, held as plain arrays and isbits values only (gpena/Bramble.jl#437), so it can
cross a task boundary (`Polyester.@batch`) where a mutable mesh holding a `Dict` cannot.

[`_walk_mesh`](@ref)`(Ωₕ)` returns it. It implements every geometric accessor a `Mesh1D`
does (`points`, `point`, `spacing`, `half_spacing`, `cell_measure`, `npoints`,
`_mesh_version`, `is_uniform`, ...), reading the same arrays: a state taken before
[`set_points!`](@ref) with an unchanged point count still aliases the mutated arrays, and
only its `version` tells it is stale. The marker `Dict` stays on the `Mesh1D`
([`markers`](@ref)); the state carries the same bits as one word matrix
([`_marker_words`](@ref)), column [`_marker_id`](@ref)`(Ωₕ, label)` per label.

# Fields

  - `set`: the 1D [`CartesianProduct`](@ref) interval the mesh discretises.
  - `indices`: `CartesianIndices{1}` of the grid points.
  - `backend`: the linear algebra [`Backend`](@ref).
  - `pts`: grid points ``x_i``, ``i = 1, \\dots, N``.
  - `half_pts`: cell centers ``x_{i+1/2}``, ``i = 1, \\dots, N+1``.
  - `half_spacings`: cell widths ``h_{i+1/2}``, ``i = 1, \\dots, N``.
  - `spacings`: backward spacings ``h_i = x_i - x_{i-1}``, with ``h_1 = x_2 - x_1``.
  - `collapsed`: whether the interval is degenerate (a single point).
  - `version`: the mesh version when this state was built (gpena/Bramble.jl#221); see
    [`_mesh_version`](@ref).
  - `uniform`: the default-tolerance [`is_uniform`](@ref) answer for these points,
    computed whenever the points change (gpena/Bramble.jl#332).
  - `words`: one column of `BitVector` chunks per marker label, a `Matrix{UInt64}` on a
    mesh; its type `WT` is a parameter so a rebuilt state can hold another array type
    ([`_batch_rebuild`](@ref)).
  - `uid`: an identity unique to the owning mesh, kept across its mutations.
"""
struct Mesh1DState{BT <: Backend, CI <: CartesianIndices{1}, VT <: AbstractVector, T,
    WT <: AbstractMatrix{UInt64}} <: AbstractMeshType{1}
    "the geometric domain, a 1D CartesianProduct (interval), over which the mesh is defined."
    set::CartesianProduct{1, T}
    "the `CartesianIndices` of the grid, for array-like iteration and indexing over the points."
    indices::CI
    "the computational backend used for linear algebra operations."
    backend::BT
    "a vector holding the coordinates of the grid points, ``x_i``."
    pts::VT
    "a vector of pre-computed cell centers (midpoints), ``x_{i+1/2}``."
    half_pts::VT
    "a vector of pre-computed cell widths, ``h_{i+1/2}``."
    half_spacings::VT
    "a vector of pre-computed backward spacings, ``h_i = x_i - x_{i-1}``, with `h_1 = x_2 - x_1`."
    spacings::VT
    "a boolean flag indicating if the domain is degenerate (a single point)."
    collapsed::Bool
    "the mesh version this state was built at; see `_mesh_version`."
    version::Int
    "the default-tolerance `is_uniform` answer for these points."
    uniform::Bool
    "the marker bits, one column of `BitVector` chunks per label (`_marker_id`)."
    words::WT
    "an identity unique to the owning mesh, kept across its mutations."
    uid::UInt64
end

"""
    Mesh1D{BT, CI, VT, T} <: AbstractMeshType{1}

One-dimensional grid discretizing a 1D [`CartesianProduct`](@ref) interval.

The geometry lives in an immutable [`Mesh1DState`](@ref) (points, cell centers
`half_pts`, cell measures `half_spacings`, backward `spacings`, indices, set, backend,
version, uniformity flag and marker words); every mutation (`set_points!`,
`set_markers!`, refinement) replaces it whole. The mesh itself keeps the public marker
dictionary and the label-to-column table of the state's marker words.

# Fields

  - `markers`: [`MeshMarkers`](@ref) dictionary mapping symbols to `BitVector` indicators.
  - `marker_ids`: `Dict{Symbol, Int}`, the column of each label in the state's word matrix.
  - `state`: the [`Mesh1DState`](@ref) holding everything else, read through the
    accessors ([`points`](@ref), [`spacings`](@ref), ...).
  - `marker_stamp`: a counter value drawn afresh at construction and at every marker
    replacement, so a cached marker read can tell the markers changed.

See also: [`MeshnD`](@ref), [`mesh`](@ref), [`AbstractMeshType`](@ref).
"""
mutable struct Mesh1D{BT <: Backend, CI <: CartesianIndices{1}, VT <: AbstractVector, T} <:
               AbstractMeshType{1}
    "a dictionary mapping `Symbol` labels to `BitVector`s, marking specific points on the mesh."
    markers::MeshMarkers
    "the column of each marker label in the state's word matrix."
    marker_ids::Dict{Symbol, Int}
    "the immutable geometry, version, uniformity flag and marker words."
    state::Mesh1DState{BT, CI, VT, T, Matrix{UInt64}}
    "a fresh counter value at construction and at every marker replacement."
    marker_stamp::UInt64
end

# The mesh over `state` with a fresh marker stamp.
function Mesh1D(markers::MeshMarkers, marker_ids::Dict{Symbol, Int}, state::Mesh1DState)
    return Mesh1D(markers, marker_ids, state, _next_mesh_uid())
end

# The mesh over `s` with `markers`, the label table and the state's words built together.
function _mesh1d_with_markers(markers::MeshMarkers, s::Mesh1DState)
    ids, words = _marker_table(markers, length(s.pts))
    return Mesh1D(markers, ids, _restate(s; words))
end

"""
    _rebackend(Ωₕ::Mesh1D, be::Backend, f) -> Mesh1D

A mesh over the same markers, label table, version and uniformity flag as `Ωₕ`, on backend
`be`, with `f` applied to each of its four arrays (`Array` for a host mirror, `identity` to
share them), and an identity of its own.
"""
function _rebackend(Ωₕ::Mesh1D, be::Backend, f::F) where {F}
    s = _st(Ωₕ)
    r = Mesh1DState(s.set, s.indices, be, f(s.pts), f(s.half_pts), f(s.half_spacings),
        f(s.spacings), s.collapsed, s.version, s.uniform, s.words, _next_mesh_uid())
    return Mesh1D(markers(Ωₕ), _marker_ids(Ωₕ), r)
end

# Both a `Mesh1D` and its state answer the geometric accessors below, reading the state.
const _Mesh1DLike{BT, CI, VT, T} = Union{Mesh1D{BT, CI, VT, T}, Mesh1DState{BT, CI, VT, T}}

@inline _st(Ωₕ::Mesh1D) = getfield(Ωₕ, :state)
@inline _st(s::Mesh1DState) = s

"""
    _walk_mesh(Ωₕ::AbstractMeshType) -> AbstractMeshType

The immutable state the assembly walk reads `Ωₕ` through: a [`Mesh1D`](@ref)'s stored
[`Mesh1DState`](@ref), or for a [`MeshnD`](@ref) a [`MeshnDState`](@ref) built on demand
from its submeshes' states. A state is its own walk state.
"""
@inline _walk_mesh(Ωₕ::Mesh1D) = _st(Ωₕ)
@inline _walk_mesh(s::Mesh1DState) = s

@inline _set_state!(Ωₕ::Mesh1D, s::Mesh1DState) = (setfield!(Ωₕ, :state, s); return nothing)

# `s` with the named fields replaced; `set`, `backend`, `collapsed` and `uid` never change.
@inline function _restate(
        s::Mesh1DState;
        indices = s.indices,
        pts = s.pts,
        half_pts = s.half_pts,
        half_spacings = s.half_spacings,
        spacings = s.spacings,
        version = s.version,
        uniform = s.uniform,
        words = s.words
)
    return typeof(s)(s.set, indices, s.backend, pts, half_pts, half_spacings, spacings,
        s.collapsed, version, uniform, words, s.uid)
end

# The default-tolerance uniformity of `s`'s current spacings: what `uniform` stores. `mag`
# is read from the points, which `change_points!` keeps current, not from the stale set.
@inline function _computed_uniform(s::Mesh1DState)
    p = host_points(s)
    mag = max(abs(first(p)), abs(last(p)))
    return _uniform_default_tol(host_spacings(s), eltype(s), mag)
end

@inline _marker_words(Ωₕ::_Mesh1DLike) = _st(Ωₕ).words
@inline _marker_ids(Ωₕ::Mesh1D) = getfield(Ωₕ, :marker_ids)

# The stamp of `Ωₕ`'s marker words: redrawn by every `_store_markers!`, never by a geometry
# change, and never equal across meshes or across two stores.
@inline _marker_stamp(Ωₕ::Mesh1D) = getfield(Ωₕ, :marker_stamp)

# Replaces the marker dictionary, its label table and the state's words together (O6: a
# label set may change after construction; the word matrix is rebuilt, never resized).
function _store_markers!(Ωₕ::Mesh1D, mesh_markers)
    mm = convert(MeshMarkers, mesh_markers)
    ids, words = _marker_table(mm, npoints(Ωₕ))
    setfield!(Ωₕ, :markers, mm)
    setfield!(Ωₕ, :marker_ids, ids)
    _set_state!(Ωₕ, _restate(_st(Ωₕ); words))
    setfield!(Ωₕ, :marker_stamp, _next_mesh_uid())
    return nothing
end

@inline set(Ωₕ::Mesh1D) = _st(Ωₕ).set
@inline indices(Ωₕ::Mesh1D) = _st(Ωₕ).indices
@inline backend(Ωₕ::Mesh1D) = _st(Ωₕ).backend
@inline set_indices!(Ωₕ::Mesh1D, idxs) = _set_state!(Ωₕ, _restate(_st(Ωₕ); indices = idxs))

@noinline _throw_point_count_mismatch(expected::Int, got::Int) = throw(
    DimensionMismatch(
    "change_points! keeps the point count: the mesh has $expected points and $got were given",
),
)

@inline is_collapsed(Ωₕ::_Mesh1DLike) = _st(Ωₕ).collapsed

"""
    _mesh_version(Ωₕ::AbstractMeshType) -> Int

The monotone counter `set_points!` bumps every in-place point mutation
(gpena/Bramble.jl#221) -- what a [`ScalarGridSpace`](@ref)'s [`SpaceWeights`](@ref) records
at construction and every discrete inner product/norm re-checks against, to catch weights
computed from a mesh that has since been mutated underneath them.

A `Mesh1D` reads the counter stored in its [`Mesh1DState`](@ref). A `MeshnD` has no
coordinates of its own -- only [`submeshes`](@ref), each independently mutated by
`change_points!(Ωₕ::MeshnD, ...)` -- so its version is the sum of theirs: strictly
increasing whenever any one axis changes, regardless of which, with no second counter of
its own to keep in sync. A state answers the version it was built at.
"""
@inline _mesh_version(Ωₕ::_Mesh1DLike) = _st(Ωₕ).version

@inline points(Ωₕ::_Mesh1DLike) = _st(Ωₕ).pts

# `point` is the choke point every mesh iteration (`Base.iterate`, `Ωₕ[i]`, and `MeshnD`'s
# per-submesh `point`) goes through one index at a time. A host-backed mesh reads straight
# off `points(Ωₕ)`; a device-backed one throws a named error instead of attempting `N`
# sequential scalar reads that its own scalar-indexing guard would refuse one at a time
# anyway (gpena/Bramble.jl#308) -- call [`host_points`](@ref) once and index or iterate that
# `Array` instead.
@inline function point(Ωₕ::_Mesh1DLike, i)
    idx = _extract_linear_index(i)
    _check_point_bounds(Ωₕ, idx, "point")
    return _point(locality(typeof(points(Ωₕ))), Ωₕ, idx)
end

@inline _point(::HostLocality, Ωₕ::_Mesh1DLike, idx) = @inbounds points(Ωₕ)[idx]
@noinline _point(::DeviceLocality, Ωₕ::_Mesh1DLike, idx) = _throw_no_scalar_point()

# Kept out of `point` itself so its success path -- one type-level `locality` check the
# compiler folds away -- is all that is ever compiled inline there, matching
# `_throw_device_scalar_weights` (`space/scalar_gridspace.jl`).
@noinline _throw_no_scalar_point() = error(
    "point(Ωₕ, i) scalar-indexes a device-backed mesh's coordinates one point at a time, " *
    "which its own scalar-indexing guard refuses -- and iterating the mesh (`for p in Ωₕ`, " *
    "`Ωₕ[i]`, `collect(Ωₕ)`) goes through this same call once per point. Call " *
    "host_points(Ωₕ) once to bring every coordinate to the host in a single transfer, then " *
    "index or iterate that Array as many times as needed.",
)

"""
    host_points(Ωₕ::Mesh1D) -> Vector

Return [`points`](@ref)`(Ωₕ)` as a host-resident `Array`, in one bulk transfer regardless
of where `Ωₕ`'s storage lives, the same way [`host_spacings`](@ref) does for
[`spacings`](@ref) (gpena/Bramble.jl#308).

[`point`](@ref)`(Ωₕ, i)` throws outright on a device-backed mesh rather than scalar-reading
it one point at a time. Call this once instead to bring every coordinate to the host in a
single transfer, then index the `Array` it returns as many times as needed --
[`locate_cell`](@ref)'s non-uniform search does exactly that.

On a host-backed mesh, `points(Ωₕ)` already is an `Array`, so this returns it directly with
no copy -- `host_points(Ωₕ) === points(Ωₕ)`; only a device-backed mesh pays the one bulk
`Array(...)` transfer.

See also: [`host_spacings`](@ref), [`points`](@ref), [`point`](@ref).
"""
@inline host_points(Ωₕ::_Mesh1DLike) = _host_points(locality(typeof(points(Ωₕ))), Ωₕ)

@inline _host_points(::HostLocality, Ωₕ::_Mesh1DLike) = points(Ωₕ)
@inline _host_points(::DeviceLocality, Ωₕ::_Mesh1DLike) = Array(points(Ωₕ))

"""
    locate_cell(Ωₕ::Mesh1D, x::Real) -> Int

See [`locate_cell`](@ref)'s generic docstring (`mesh/queries.jl`) for the contract: the
largest node index `i` with `pts[i] <= x`, clamped to `1:n-1` -- exactly what
`searchsortedlast` on the point array answers, and what this method must keep answering,
uniform or not.

A uniform host-backed mesh reads its own end points `pts[1]` and `pts[n]`, an O(1) read,
and takes a first estimate `clamp(floor(Int, (x - pts[1]) / h) + 1, 1, n - 1)` with
`h = (pts[n] - pts[1]) / (n - 1)`; it then walks against `pts` until
`pts[idx] <= x < pts[idx + 1]`, one step at most in practice. That is exactly
`searchsortedlast`, with no allocation, whatever the points are: [`change_points!`](@ref)
accepts points that are uniform only within tolerance and need not lie on the domain's set,
so neither the set nor a closed form `a + (i - 1) * h` may stand in for them
(gpena/Bramble.jl#495).

A uniform device-backed mesh refuses scalar reads (gpena/Bramble.jl#308). While its version
is still 0 its points are the ones `_mesh` built from the set, so it answers with no array
read: the same estimate on the set's endpoints `a`, `b`, then one correction step against
the closed-form coordinates of that estimate's neighbouring nodes (`a + idx * h`,
`a + (idx - 1) * h`), since `(x - a) / h` rounds to either side of an integer at a node
coordinate -- worse in `Float32` (Metal's only type) than in `Float64`. Once
[`set_points!`](@ref) has replaced its points (version above 0), it searches
[`host_points`](@ref)`(Ωₕ)` as a non-uniform device mesh does: one bulk transfer, exact.

A point at or past either endpoint, `±Inf` included, returns the boundary cell before any
division, and `NaN` returns `n - 1` as the search does, so no path throws.

A non-uniform mesh has no formula to fall back on and searches [`host_points`](@ref)`(Ωₕ)`
instead of the raw, possibly device-resident `points(Ωₕ)`.
"""
function locate_cell(Ωₕ::_Mesh1DLike, x::Real)
    n = npoints(Ωₕ)
    n <= 1 && return 1
    is_uniform(Ωₕ) && return _locate_uniform(locality(typeof(points(Ωₕ))), Ωₕ, x, n)
    return _locate_search(Ωₕ, x, n)
end

# Host: estimate from the mesh's own end points, then walk against `pts` to the exact
# `searchsortedlast` answer; the set is never read, so moved points cannot mislead it.
function _locate_uniform(::HostLocality, Ωₕ::_Mesh1DLike, x::Real, n::Int)
    pts = points(Ωₕ)
    a = pts[1]
    b = pts[n]
    # answer the ends before `floor(Int, ...)`, which throws on a quotient past
    # `typemax(Int)`, on ±Inf and on NaN; NaN takes the last cell, as the search
    # path does (`searchsortedlast` sorts NaN last)
    x <= a && return 1
    (x >= b || isnan(x)) && return n - 1
    h = (b - a) / (n - 1)
    idx = clamp(floor(Int, (x - a) / h) + 1, 1, n - 1)

    # here a < x < b, so the walk stops inside 1:n-1 with pts[idx] <= x < pts[idx + 1]
    while idx < n - 1 && pts[idx + 1] <= x
        idx += 1
    end
    while idx > 1 && pts[idx] > x
        idx -= 1
    end
    return idx
end

# Device: no scalar reads. A version-0 mesh still holds the points `_mesh` built from the
# set, so the closed form on `extrema(set)` is valid; after `set_points!` (the only writer
# that bumps the version) the points may have left the set, so search them exactly.
function _locate_uniform(::DeviceLocality, Ωₕ::_Mesh1DLike, x::Real, n::Int)
    _mesh_version(Ωₕ) == 0 || return _locate_search(Ωₕ, x, n)

    a, b = extrema(_st(Ωₕ).set)
    x <= a && return 1
    (x >= b || isnan(x)) && return n - 1
    h = (b - a) / (n - 1)
    idx = floor(Int, (x - a) / h) + 1

    # `idx` is a candidate for the largest node index with node <= x, built from
    # `(x - a) / h` alone; that division can round either side of an integer at an
    # exact node coordinate, so re-derive both of `idx`'s neighbouring nodes the same
    # closed-form way (never by reading `pts`) and shift by one if `x` actually sits
    # past the upper one or short of the lower one. At most one of the two branches
    # below can fire, since they move `idx` in opposite directions.
    upper = a + idx * h
    if x >= upper
        idx += 1
    else
        lower = a + (idx - 1) * h
        if x < lower
            idx -= 1
        end
    end

    return clamp(idx, 1, n - 1)
end

function _locate_search(Ωₕ::_Mesh1DLike, x::Real, n::Int)
    pts = host_points(Ωₕ)
    if x <= pts[1]
        return 1
    elseif x >= pts[n]
        return n - 1
    end
    idx = searchsortedlast(pts, x)
    return clamp(idx, 1, n - 1)
end

@inline half_points(Ωₕ::_Mesh1DLike) = _st(Ωₕ).half_pts
@inline half_spacings(Ωₕ::_Mesh1DLike) = _st(Ωₕ).half_spacings

"""
    spacings(Ωₕ::Mesh1D) -> AbstractVector

Return the cached vector of backward spacings, where `spacings(Ωₕ)[i]` is
[`spacing`](@ref)`(Ωₕ, i)`. Recomputed by [`set_points!`](@ref) whenever the
grid points change.
"""
@inline spacings(Ωₕ::_Mesh1DLike) = _st(Ωₕ).spacings
@inline spacings!(Ωₕ::Mesh1D, v) = _set_state!(Ωₕ, _restate(_st(Ωₕ); spacings = v))

"""
    host_spacings(Ωₕ::Mesh1D) -> Vector

Return [`spacings`](@ref)`(Ωₕ)` as a host-resident `Array`, in one bulk transfer regardless
of where `Ωₕ`'s storage lives (gpena/Bramble.jl#94).

[`spacing`](@ref)`(Ωₕ, i)` and [`forward_spacing`](@ref)`(Ωₕ, i)` read `spacings(Ωₕ)` one
element at a time, which a device-backed mesh (Metal.jl, ...) refuses outright: its scalar-
indexing guard throws before a per-point loop over either accessor gets anywhere. Call this
once instead to bring every spacing to the host in a single transfer, then index the `Array`
it returns as many times as needed -- `stencil.jl`'s dense stencil-matrix builder does exactly
that.

On a host-backed mesh, `spacings(Ωₕ)` already is an `Array`, so this returns it directly with
no copy; only a device-backed mesh pays the one bulk `Array(...)` transfer.

See also: [`spacings`](@ref), [`spacing`](@ref), [`forward_spacing`](@ref).
"""
@inline host_spacings(Ωₕ::_Mesh1DLike) = _host_spacings(locality(typeof(spacings(Ωₕ))), Ωₕ)

@inline _host_spacings(::HostLocality, Ωₕ::_Mesh1DLike) = spacings(Ωₕ)
@inline _host_spacings(::DeviceLocality, Ωₕ::_Mesh1DLike) = Array(spacings(Ωₕ))

"""
    host_half_spacings(Ωₕ::Mesh1D) -> Vector

Return [`half_spacings`](@ref)`(Ωₕ)` as a host-resident `Array`, in one bulk transfer
regardless of where `Ωₕ`'s storage lives, the same way [`host_spacings`](@ref) does for
[`spacings`](@ref) (gpena/Bramble.jl#307).

On a host-backed mesh, `half_spacings(Ωₕ)` already is an `Array`, so this returns it
directly with no copy; only a device-backed mesh pays the one bulk `Array(...)` transfer.

See also: [`host_spacings`](@ref), [`half_spacings`](@ref), [`half_spacing`](@ref).
"""
@inline host_half_spacings(Ωₕ::_Mesh1DLike) = _host_half_spacings(locality(typeof(half_spacings(Ωₕ))), Ωₕ)

@inline _host_half_spacings(::HostLocality, Ωₕ::_Mesh1DLike) = half_spacings(Ωₕ)
@inline _host_half_spacings(::DeviceLocality, Ωₕ::_Mesh1DLike) = Array(half_spacings(Ωₕ))

"""
    forward_spacings(Ωₕ::Mesh1D) -> AbstractVector

Return the forward spacings of `Ωₕ`, where `forward_spacings(Ωₕ)[i]` is
[`forward_spacing`](@ref)`(Ωₕ, i)`. Unlike [`spacings`](@ref), this is not cached: it is
[`spacing`](@ref)'s vector read one index ahead, a new vector each call, of the same array
type as `spacings(Ωₕ)`.

See also: [`forward_spacing_for_derivative`](@ref).
"""
@inline function forward_spacings(Ωₕ::_Mesh1DLike)
    h = spacings(Ωₕ)
    n = length(h)
    n == 1 && return copy(h)
    f = similar(h)
    copyto!(f, 1, h, 2, n - 1)
    copyto!(f, n, h, n, 1)
    return f
end

# A single-point mesh (n == 1, whether from a topologically collapsed domain or simply a
# one-point request) has no adjacent interval, so `half_spacings` is the honest raw zero
# there -- that raw value stays untouched, since other code (collapse detection, among it)
# reads it as exactly that. `cell_measures` is a *measure*, though, and the same `_apply_hs_logic`
# coercion `half_spacing(::MeshnD, idx)` already applies is needed here too, or a mesh with
# a collapsed axis silently gets a zero weight everywhere (gpena/Bramble.jl#89): the zero
# case is only ever the single-element one, so this stays the same zero-copy array in
# every other case and only allocates on that one rare, one-element path. That copy is
# `similar(hs)` filled by a broadcast, so it keeps the mesh's own vector type (a device
# vector stays one) and never reads `hs[1]` as a scalar (gpena/Bramble.jl#497).
@inline function cell_measures(Ωₕ::_Mesh1DLike)
    hs = half_spacings(Ωₕ)
    return length(hs) == 1 ? (similar(hs) .= _apply_hs_logic.(hs)) : hs
end

"""
    set_points!(Ωₕ::Mesh1D, pts::AbstractVector) -> Nothing

Override the grid coordinates in `Ωₕ`. Recalculates cached [`spacings`](@ref),
[`half_points`](@ref), and [`half_spacings`](@ref), in place when the point count is
unchanged, then replaces the mesh's [`Mesh1DState`](@ref) with one carrying the new version
and the uniformity of the new points.

A new point count also rebuilds the `:boundary` and `:interior` markers (and the marker
words assembly reads) for the new grid. A mesh carrying any other label has no domain here
to re-evaluate it onto the new points, so such a resize throws an `ArgumentError` before
anything changes. Build `mesh(Ω, length(pts))` and call
[`change_points!`](@ref)`(Ωₕ, markers(Ω), pts)` instead. A call that keeps the point count
leaves every marker alone, custom labels included.

Bumps `Ωₕ`'s mesh version (gpena/Bramble.jl#221): every [`ScalarGridSpace`](@ref) already
built on `Ωₕ` -- via [`gridspace`](@ref), directly or as a leaf of a
[`CompositeGridSpace`](@ref) -- keeps its own weights, precomputed from the mesh *before*
this call. Its `innerₕ`/`inner₊*`/norms now throw naming the mismatch instead of silently
computing against stale weights; call `gridspace(Ωₕ)` again for a space that reads the
mutated mesh. This is also what [`change_points!`](@ref) goes through, so the same applies
to it.
"""
function set_points!(Ωₕ::Mesh1D, pts)
    n = length(pts)
    if length(_st(Ωₕ).pts) == n
        _set_points_geometry!(Ωₕ, pts)
        return nothing
    end

    # A resize leaves every marker sized for the old grid. `:boundary`/`:interior` are
    # rebuilt below; any other label has no domain to be re-evaluated from, so it is
    # refused before the geometry changes, as `iterative_refinement!(Ωₕ)` refuses it
    # (gpena/Bramble.jl#19).
    extra_labels = setdiff(keys(markers(Ωₕ)), (:boundary, :interior))
    isempty(extra_labels) || _throw_resize_drops_markers(extra_labels)

    _set_points_geometry!(Ωₕ, pts)
    fresh_markers = MeshMarkers()
    _ensure_geometric_markers!(fresh_markers, Ωₕ)
    markers!(Ωₕ, fresh_markers)
    return nothing
end

@noinline function _throw_resize_drops_markers(extra_labels)
    throw(
        ArgumentError(
        "set_points!(Ωₕ, pts) was asked to change the point count of a mesh carrying " *
        "custom markers $(Tuple(extra_labels)), and there is no domain here to " *
        "re-evaluate them onto the new points. Build mesh(Ω, length(pts)) and call " *
        "change_points!(Ωₕ, markers(Ω), pts) instead.",
    ),
    )
end

# The geometry of `set_points!` alone, with markers left untouched: the public setter adds
# the marker rebuild for a new point count, and `_refine_indices!` calls this directly
# because both refinement forms rebuild markers themselves.
@inline function _set_points_geometry!(Ωₕ::Mesh1D, pts)
    s = _st(Ωₕ)
    n = length(pts)

    if length(s.pts) == n
        # Same length: the arrays are updated in place, so a state taken earlier still
        # aliases them and only its `version` tells it is stale.
        s.pts .= pts
    else
        be = s.backend
        _set_state!(Ωₕ,
            _restate(s; indices = generate_indices(n), pts, half_pts = vector(be, n + 1),
                half_spacings = vector(be, n), spacings = vector(be, n)))
    end

    # A device-backed mesh with at least two points fills its three derived arrays in the
    # one fused, non-uniform-formula kernel (gpena/Bramble.jl#305) -- the same kernel
    # `_mesh` dispatches to for a non-uniform device mesh at construction, and valid here
    # for any points (uniform or not), since `set_points!` carries no uniformity flag to
    # branch on. Everything else (host-backed, or a single/no-interval mesh) keeps the
    # three separate passes; the spacings come first there, since half_spacing! below
    # reads them back through `spacing`.
    if n >= 2 && !(points(Ωₕ) isa Array)
        _nonuniform_mesh1d_metrics!(Ωₕ)
    else
        spacing!(spacings(Ωₕ), Ωₕ)

        # Re-compute the cell centers (half_pts) using the new grid points.
        half_points!(half_points(Ωₕ), Ωₕ)

        # Re-compute the cell widths (half_spacings) using the new grid points.
        half_spacing!(half_spacings(Ωₕ), Ωₕ)
    end

    # The new state: one version on, with the uniformity of the new spacings.
    s = _st(Ωₕ)
    _set_state!(Ωₕ, _restate(s; version = s.version + 1, uniform = _computed_uniform(s)))
    return nothing
end

"""
    half_points!(Ωₕ::Mesh1D, pts::AbstractVector) -> Nothing

Override the precomputed cell center cache in `Ωₕ`.
"""
@inline half_points!(Ωₕ::Mesh1D, pts) = _set_state!(Ωₕ, _restate(_st(Ωₕ); half_pts = pts))

"""
    half_spacings!(Ωₕ::Mesh1D, pts::AbstractVector) -> Nothing

Override the precomputed cell width cache in `Ωₕ`.
"""
@inline half_spacings!(Ωₕ::Mesh1D, pts) = _set_state!(Ωₕ, _restate(_st(Ωₕ); half_spacings = pts))

@inline eltype(::_Mesh1DLike{BT}) where {BT} = eltype(BT)
@inline eltype(::Type{<:Mesh1D{BT}}) where {BT} = eltype(BT)
@inline eltype(::Type{<:Mesh1DState{BT}}) where {BT} = eltype(BT)

"""
    (Ωₕ::Mesh1D)(i::Integer) -> Mesh1D

Return the `i`-th submesh of `Ωₕ`. A 1D mesh is its own only submesh, returning
`Ωₕ` itself; provided for uniform indexing in multi-dimensional generic algorithms.
"""
@inline (Ωₕ::Mesh1D)(::Integer) = Ωₕ
@inline (s::Mesh1DState)(::Integer) = s

@inline npoints(Ωₕ::_Mesh1DLike) = length(points(Ωₕ))
@inline npoints(Ωₕ::_Mesh1DLike, ::Type{Tuple}) = (npoints(Ωₕ),)

@inline hₘₐₓ(Ωₕ::_Mesh1DLike) = maximum(spacings(Ωₕ))
@inline hₘᵢₙ(Ωₕ::_Mesh1DLike) = minimum(spacings(Ωₕ))

# On a device-backed mesh, indexing `spacings(Ωₕ)` here scalar-indexes a device array and
# is refused by that array type's own scalar-indexing guard (gpena/Bramble.jl#94) -- left as
# the raw upstream error deliberately, the same one every other GPU array in the ecosystem
# raises for the same mistake. Catching it here to redirect to `host_spacings` would put a
# locality check in the single most-called accessor in this file for a message this array
# type already gives; a caller that hits it once should stop calling this per point and call
# `host_spacings(Ωₕ)` once instead, as `stencil.jl` now does.
@inline function spacing(Ωₕ::_Mesh1DLike, i::Int)
    _check_point_bounds(Ωₕ, i, "spacing")
    return @inbounds spacings(Ωₕ)[i]
end

@inline spacing(Ωₕ::_Mesh1DLike, i::CartesianIndex{1}) = spacing(Ωₕ, _extract_linear_index(i))

# D == 1, so dim is always 1: a plain passthrough matching the MeshnD 3-arg accessor
# (gpena/Bramble.jl#111), so mesh-generic callers can use one signature regardless of D.
@inline spacing(Ωₕ::_Mesh1DLike, i, dim::Int) = spacing(Ωₕ, i)
"""
    spacing_for_derivative(Ωₕ::Mesh1D, idx) -> eltype(Ωₕ)

Return the spacing that a backward finite difference divides by at `idx`, which is
[`spacing`](@ref)`(Ωₕ, idx)` everywhere except the first point, where the difference has
no stencil and this is zero.

See also: [`forward_spacing_for_derivative`](@ref), [`spacings`](@ref).
"""
@inline function spacing_for_derivative(Ωₕ::_Mesh1DLike, idx)
    i = idx isa CartesianIndex{1} ? _extract_linear_index(idx) : idx
    if i == 1
        zero(eltype(Ωₕ))
    else
        spacing(Ωₕ, i)
    end
end

"""
    backward_spacings_for_derivative(Ωₕ::Mesh1D) -> AbstractVector

Return a vector `h` with `h[i] == `[`spacing_for_derivative`](@ref)`(Ωₕ, i)` for every
`i > 1`. Entry 1 holds [`spacing`](@ref)`(Ωₕ, 1)`, which no backward stencil reads; the
scalar [`spacing_for_derivative`](@ref)`(Ωₕ, 1)` is zero.
"""
@inline backward_spacings_for_derivative(Ωₕ::_Mesh1DLike) = spacings(Ωₕ)

"""
    forward_spacings_for_derivative(Ωₕ::Mesh1D) -> AbstractVector

Return a vector `h` with `h[i] == `[`forward_spacing_for_derivative`](@ref)`(Ωₕ, i)` for
every `i < npoints(Ωₕ)`, as a view onto cached spacings. The last entry is omitted because
the forward difference has no stencil at the right boundary.
"""
@inline function forward_spacings_for_derivative(Ωₕ::_Mesh1DLike)
    h = spacings(Ωₕ)
    return @inbounds @view h[min(2, length(h)):end]
end

# Same device-mesh tradeoff as `spacing` above, and the same choice: this scalar-indexes
# `spacings(Ωₕ)` and is left to throw the array type's own raw scalar-indexing error rather
# than a redirect added here. Use `host_spacings(Ωₕ)` for a per-point loop instead.
@inline function forward_spacing(Ωₕ::_Mesh1DLike, i::Int)
    _check_point_bounds(Ωₕ, i, "forward_spacing")
    # forward_spacing(i) is spacing(i + 1) away from the last point, and repeats the
    # final interval at it, which is exactly what the cached vector already holds.
    n = npoints(Ωₕ)
    return @inbounds spacings(Ωₕ)[i == n ? n : i + 1]
end

@inline forward_spacing(Ωₕ::_Mesh1DLike, i::CartesianIndex{1}) = forward_spacing(Ωₕ, _extract_linear_index(i))

@inline forward_spacing(Ωₕ::_Mesh1DLike, i, dim::Int) = forward_spacing(Ωₕ, i)
"""
    forward_spacing_for_derivative(Ωₕ::Mesh1D, idx) -> eltype(Ωₕ)

Return the spacing that a forward finite difference divides by at `idx`, which is
[`forward_spacing`](@ref)`(Ωₕ, idx)` everywhere except the last point, where the
difference has no stencil and this is zero.

See also: [`spacing_for_derivative`](@ref), [`spacings`](@ref).
"""
@inline function forward_spacing_for_derivative(Ωₕ::_Mesh1DLike, idx)
    i = idx isa CartesianIndex{1} ? _extract_linear_index(idx) : idx

    if i == npoints(Ωₕ)
        zero(eltype(Ωₕ))
    else
        forward_spacing(Ωₕ, i)
    end
end

@inline function half_point(Ωₕ::_Mesh1DLike, i::Int)
    _check_half_point_bounds(Ωₕ, i)
    return _st(Ωₕ).half_pts[i]
end

@inline function half_spacing(Ωₕ::_Mesh1DLike, i::Int)
    _check_point_bounds(Ωₕ, i, "half_spacing")
    return _st(Ωₕ).half_spacings[i]
end

@inline half_spacing(Ωₕ::_Mesh1DLike, idx::CartesianIndex{1}) = half_spacing(Ωₕ, _extract_linear_index(idx))

@inline function cell_measure(Ωₕ::_Mesh1DLike, i)
    idx = _extract_linear_index(i)
    _check_point_bounds(Ωₕ, idx, "cell_measure")
    return _apply_hs_logic(half_spacing(Ωₕ, idx))
end

# `_generate_random_points!` used to draw from `Random.default_rng()` unconditionally, but a
# device backend's kernel launches (metrics, fused init, ...) draw from that same global
# stream as a side effect of launching -- even a uniform device mesh, which calls no RNG
# code of its own, perturbs it. That meant seeding the global RNG and building a non-uniform
# mesh on the host, then seeding again and building the "same" mesh on a device backend,
# produced different interior coordinates: the device path's other kernel launches had
# already burned draws from the global stream before this function ever ran
# (gpena/Bramble.jl#320).
#
# The fix is opt-in, not a blanket switch to a package-local RNG: dozens of existing tests
# (test/form/jacobian_pattern.jl:132, test/form/kronecker.jl, test/ext/kronecker_ext.jl,
# test/space/centered_difference.jl, and ~23 others) call `Random.seed!(N)` immediately
# before building a non-uniform mesh and rely on that call alone controlling the mesh's
# interior points -- some for a single deterministic mesh a numerical assertion depends on
# (jacobian_pattern.jl's Newton-iteration-count bound is one), not a "build twice, compare"
# pattern. Drawing from a separate RNG unconditionally would make every one of those
# `Random.seed!(N)` calls silently stop controlling its mesh, and would instead have
# `_generate_random_points!` read whatever state a permanently-live, never-reseeded package
# RNG happens to have accumulated from every other non-uniform mesh built earlier in the
# same process -- order-dependent and effectively random in practice. Arming
# `_generate_random_points!` onto the isolated RNG only for a caller that explicitly asks
# for it, via `_seed_mesh1d_rng!`, keeps every existing `Random.seed!(N)` call working
# exactly as before (`_MESH1D_RNG_ARMED` stays `false`, so the `else` branch below runs,
# unchanged from the original code) while still giving device-mesh-reproducibility code
# (gpena/Bramble.jl#320) a way to opt out of the global stream entirely.
const _MESH1D_RNG = Random.Xoshiro()
const _MESH1D_RNG_ARMED = Ref(false)

"""
    _seed_mesh1d_rng!(seed) -> Nothing

Seed the package-local RNG `_generate_random_points!` can draw from, and switch it
onto that RNG instead of `Random.default_rng()` (gpena/Bramble.jl#320). A device backend's
kernel launches (for metrics, fused init, and other work) consume draws from the global RNG
as a side effect of launching, so seeding `Random.default_rng()` alone does not make a host
and device build of the "same" non-uniform mesh agree.

Call this immediately before building one or more non-uniform [`Mesh1D`](@ref)s that must
reproduce the same interior coordinates for a given seed regardless of what any kernel
launch does to the global stream, then call [`_unseed_mesh1d_rng!`](@ref) once done. Code
that never calls this is unaffected: `_generate_random_points!` keeps drawing from
`Random.default_rng()` exactly as it always did, so an existing `Random.seed!(N)` call
immediately before a non-uniform mesh build keeps controlling it.
"""
function _seed_mesh1d_rng!(seed)
    Random.seed!(_MESH1D_RNG, seed)
    _MESH1D_RNG_ARMED[] = true
    return nothing
end

"""
    _unseed_mesh1d_rng!() -> Nothing

Switch `_generate_random_points!` back onto `Random.default_rng()`, undoing
[`_seed_mesh1d_rng!`](@ref). Call this once code that opted into the package-local RNG is
done building its non-uniform mesh(es), so no later, unrelated code silently keeps reading
from it.
"""
function _unseed_mesh1d_rng!()
    _MESH1D_RNG_ARMED[] = false
    return nothing
end

@inline function _generate_random_points!(v)
    _draw_random_points!(v)
    sort!(v)  # In-place sort
    return nothing
end

# The unsorted draw behind `_generate_random_points!`, shared with `_redraw_tied_points!`
# so a redraw consumes the same stream, in the same way, as the first draw.
@inline function _draw_random_points!(v)
    if _MESH1D_RNG_ARMED[]
        # Canonical `Float64` draws converted to `eltype(v)`, rather than
        # `rand!(_MESH1D_RNG, v)` directly on `v` itself: Julia's `Float32` and `Float64`
        # samplers consume a different number of bits per draw from the same `Xoshiro`
        # stream, so a `Float32` destination (Metal's only type, gpena/Bramble.jl#308) and a
        # `Float64` one would disagree on the same seed even with the RNG itself perfectly
        # isolated from kernel-launch side effects.
        draws = Vector{Float64}(undef, length(v))
        rand!(_MESH1D_RNG, draws)
        v .= draws
    else
        rand!(v)
    end
    return nothing
end

# Redraw rounds `_redraw_tied_points!` tries before giving up.
const _MAX_REDRAW_ROUNDS = 100

# Whether the mapped points `x` fail to increase strictly anywhere.
@inline function _has_tied_points(x)
    @inbounds for k in 2:length(x)
        x[k] <= x[k - 1] && return true
    end
    return false
end

# Interior indices of `x` to redraw: `k` for each tie `x[k] <= x[k - 1]` with `k`
# interior, and `n - 1` when the last interior point rounded onto `x[n]`.
function _tied_interior_indices(x)
    n = length(x)
    tied = Int[]
    @inbounds for k in 2:(n - 1)
        x[k] <= x[k - 1] && push!(tied, k)
    end
    if n >= 3 && x[n] <= x[n - 1] && (isempty(tied) || last(tied) != n - 1)
        push!(tied, n - 1)
    end
    return tied
end

# Random draws on a small eltype collide, and so does the map onto [a, b], which rounds
# distinct canonical draws onto one stored value (gpena/Bramble.jl#494). Redraw only the
# tied interior entries, map them, re-sort the interior and repeat, so a collision-free mesh
# draws nothing extra and a seeded mesh keeps its points. The rounds are bounded: past the
# values representable in (a, b) no draw can succeed.
function _redraw_tied_points!(x, a, b)
    n = length(x)
    for _ in 1:_MAX_REDRAW_ROUNDS
        _has_tied_points(x) || return nothing
        tied = _tied_interior_indices(x)
        u = Vector{eltype(x)}(undef, length(tied))
        _draw_random_points!(u)
        @inbounds for (j, k) in enumerate(tied)
            x[k] = a + u[j] * (b - a)
        end
        sort!(view(x, 2:(n - 1)))
    end
    _has_tied_points(x) || return nothing
    msg = "could not draw $n distinct points in [$a, $b] with eltype $(eltype(x)) " *
          "after $_MAX_REDRAW_ROUNDS redraw rounds; use fewer points, a wider interval " *
          "or a wider eltype"
    throw(ArgumentError(msg))
end

#------------------------------------------------------------------------------------------#
# Device kernel launch stubs (gpena/Bramble.jl#94, #174)
#
# Every filler below (`_points!`, `half_points!`, `spacing!`, `half_spacing!`,
# `_refine_indices!`) keeps its `x isa Array` method exactly as it always was -- a CPU loop
# -- and gains a sibling method for a non-`Array` destination (a device-backed vector, e.g.
# `MtlVector`). That sibling computes only the host-side scalars the kernel needs, then goes
# through `ka_device` (`src/utils/device_kernels.jl`) and one of the launchers below, which
# `ext/BrambleKernelAbstractionsExt.jl` implements as a `KernelAbstractions.@kernel`. Without
# that extension loaded, the launcher throws a named diagnostic instead of failing several
# frames later on a scalar index.
#------------------------------------------------------------------------------------------#

@noinline function _throw_no_ka_mesh_kernel(fname::String)
    return error(
        "$fname requires KernelAbstractions.jl to fill a device-backed mesh vector. Add " *
        "`using KernelAbstractions` (and the package providing this backend's device, e.g. " *
        "`using Metal`) before constructing or refining a mesh on this backend.",
    )
end

# Deliberately untyped (matching the `ka_device(be)`/`metal_backend`/`_metal_backend`
# fallback idiom, `src/utils/device_kernels.jl`): the extension's method for each of these
# is typed on `AbstractVector`, and a fallback with the same signature would overwrite it
# instead of adding a genuinely more specific dispatch (method overwriting is an error during
# precompilation).
"""
    _launch_uniform_mesh1d_init!(pts, half_pts, spacings, half_spacings, a, h, n::Int, dev) -> Nothing

Fills all four of a uniform mesh's arrays -- `pts`, `half_pts`, `spacings` and
`half_spacings` -- in a single `KernelAbstractions.@kernel` launch on `dev`, over
`1:(n + 1)` work items: `pts[i] = a + (i - 1) * h`, `spacings[i] = h`, `half_spacings[i] =
h / 2` at `i = 1` or `i = n` and `h` elsewhere, and `half_pts[i] = a` at `i = 1`,
`a + (n - 1) * h` at `i = n + 1`, and the midpoint formula in between. Every entry comes
from `a`, `h` and `n` alone, with no read of `pts` itself -- the fusion of the uniform
point fill, `_launch_spacing!`, `_launch_half_points!`
and `_launch_half_spacing!` into one kernel for a uniform device mesh
(gpena/Bramble.jl#303).

# Throws
- `ErrorException`: no `KernelAbstractions` extension is loaded, so there is no device
  kernel to reach (`_throw_no_ka_mesh_kernel`).
"""
function _launch_uniform_mesh1d_init!(pts, half_pts, spacings, half_spacings, a, h, n, dev)
    _throw_no_ka_mesh_kernel("_launch_uniform_mesh1d_init!")
end

"""
    _launch_half_points!(x::AbstractVector, pts, n::Int, dev) -> Nothing

Fills `x` (length `n + 1`) with the mesh's half points: the boundary entries `x[1] = pts[1]`
and `x[n + 1] = pts[n]`, and the interior midpoints `x[i] = (pts[i] + pts[i - 1]) / 2`, via a
`KernelAbstractions.@kernel` launch on `dev`. The device counterpart of `half_points!`'s CPU
loop.

# Throws
- `ErrorException`: no `KernelAbstractions` extension is loaded (`_throw_no_ka_mesh_kernel`).
"""
_launch_half_points!(x, pts, n, dev) = _throw_no_ka_mesh_kernel("_launch_half_points!")

"""
    _launch_spacing!(x::AbstractVector, pts, n::Int, dev) -> Nothing

Fills `x` (length `n`) with the backward spacings of `pts`: the boundary entry
`x[1] = pts[2] - pts[1]`, and `x[i] = pts[i] - pts[i - 1]` elsewhere, via a
`KernelAbstractions.@kernel` launch on `dev`. The device counterpart of `spacing!`'s CPU
loop.

# Throws
- `ErrorException`: no `KernelAbstractions` extension is loaded (`_throw_no_ka_mesh_kernel`).
"""
_launch_spacing!(x, pts, n, dev) = _throw_no_ka_mesh_kernel("_launch_spacing!")

"""
    _launch_half_spacing!(x::AbstractVector, h, n::Int, dev) -> Nothing

Fills `x` (length `n`) with the mesh's half spacings: the boundary entries
`x[1] = h[1] / 2` and `x[n] = h[n] / 2`, and the interior `x[i] = (h[i] + h[i + 1]) / 2`, via
a `KernelAbstractions.@kernel` launch on `dev`. The device counterpart of `half_spacing!`'s
CPU loop.

# Throws
- `ErrorException`: no `KernelAbstractions` extension is loaded (`_throw_no_ka_mesh_kernel`).
"""
_launch_half_spacing!(x, h, n, dev) = _throw_no_ka_mesh_kernel("_launch_half_spacing!")

"""
    _launch_nonuniform_mesh1d_metrics!(half_pts, spacings, half_spacings, pts, n::Int, dev) -> Nothing

Fills a non-uniform mesh's three derived arrays -- `spacings`, `half_pts` and
`half_spacings` -- in a single `KernelAbstractions.@kernel` launch on `dev`, over
`1:(n + 1)` work items, from `pts` alone: `spacings[i] = pts[i] - pts[i - 1]` (`pts[2] -
pts[1]` at `i = 1`); `half_pts[i] = (pts[i] + pts[i - 1]) / 2` for `2 <= i <= n`, with
boundary entries `pts[1]` at `i = 1` and `pts[n]` at `i = n + 1`; and `half_spacings[i] =
spacing(i) / 2` at `i = 1` or `i = n`, and, with `back = pts[i] - pts[i - 1]` and
`fwd = pts[i + 1] - pts[i]`, `half_spacings[i] = (back + fwd) / 2` elsewhere -- the same
association `half_spacing!` uses on its own already-rounded spacings, not the cheaper
two-point difference `(pts[i + 1] - pts[i - 1]) / 2`, so a device mesh matches the CPU one
bit for bit -- the fusion of `_launch_spacing!`,
`_launch_half_points!` and `_launch_half_spacing!` into one kernel for a
non-uniform device mesh (gpena/Bramble.jl#305). Each thread reads only its own 3-point
local stencil of `pts`, so `spacings` never has to be written to device memory before
`half_spacings` can be computed from it.

# Throws
- `ErrorException`: no `KernelAbstractions` extension is loaded, so there is no device
  kernel to reach (`_throw_no_ka_mesh_kernel`).
"""
function _launch_nonuniform_mesh1d_metrics!(half_pts, spacings, half_spacings, pts, n, dev)
    _throw_no_ka_mesh_kernel("_launch_nonuniform_mesh1d_metrics!")
end

"""
    _launch_refine_indices!(new_points::AbstractVector, old_points, N_old::Int, dev) -> Nothing

Fills `new_points` (length `2 * N_old - 1`) with the refined mesh: `new_points[2i - 1] =
old_points[i]` copies each old point to its odd slot, and `new_points[2i] = (old_points[i] +
old_points[i + 1]) / 2` inserts the midpoint at the even slot for `i < N_old`, via a
`KernelAbstractions.@kernel` launch on `dev`. The device counterpart of
`_refine_indices_fill!`'s CPU loop.

# Throws
- `ErrorException`: no `KernelAbstractions` extension is loaded (`_throw_no_ka_mesh_kernel`).
"""
_launch_refine_indices!(new_points, old_points, N_old, dev) = _throw_no_ka_mesh_kernel("_launch_refine_indices!")

# Internal function to populate a vector `x` with grid point coordinates over a 1D interval `I`.
@inline function _points!(x, I::CartesianProduct{1}, unif::Bool)
    # Get the number of points and the interval's element type and bounds.
    npts = length(x)
    T = eltype(I)
    a, b = extrema(I)

    # Handle the trivial case of a single point mesh. The point sits at the lower
    # bound of the interval (for a collapsed interval a == b, so this is the point itself).
    if npts == 1
        x .= a
        return nothing
    end

    # Check if the point distribution should be uniform.
    if unif
        # For a uniform grid, calculate the constant step size `h`.
        h = (b - a) / (npts - 1)
        # Populate the grid points using an arithmetic progression.
        @simd ivdep for i in eachindex(x)
            x[i] = a + (i - 1) * h
        end
    else
        # For a non-uniform grid, first generate points in the canonical interval [0, 1].
        x[1] = zero(T)
        x[npts] = one(T)

        # Generate random points for the interior of the [0, 1] interval.
        v = view(x, 2:(npts - 1))
        _generate_random_points!(v)

        # Scale and shift the points from [0, 1] to the target interval [a, b].
        @. x = a + x * (b - a)

        # Redraw any interior point the draw or the map left equal to a neighbour.
        _redraw_tied_points!(x, a, b)
    end
    return nothing
end

# Device counterpart of the method above, reached only for the two cases `_mesh` does not
# route through the fused `_uniform_mesh1d_init!` kernel: a single-point mesh and a
# non-uniform one. `unif` and `backend` are carried to keep this method's signature distinct
# from the `Array` one above, since with `unif` dropped a three-argument device method would
# be ambiguous with it. Past the single-point guard `unif` is always `false`. There is no
# uniform branch to take. The non-uniform fill runs the `Array` method above into a scratch
# `Vector{eltype(x)}`, so its tie redraw sees exactly the values that reach `x`, and
# transfers them to `x` in one `copyto!`. Device `rand!`/`sort!` exist, but host generation
# keeps a seeded mesh identical across backends, at a one-time O(n) construction cost.
function _points!(x::AbstractVector, I::CartesianProduct{1}, unif::Bool, backend)
    cpu_pts = Vector{eltype(x)}(undef, length(x))
    _points!(cpu_pts, I, false)
    copyto!(x, cpu_pts)
    return nothing
end

# Calculates the "half points" (cell centers) for a 1D mesh.
@inline function half_points!(x::Array, Ωₕ)
    n = npoints(Ωₕ)
    pts = points(Ωₕ)

    # The first and last half-points are set to the boundary points. This is a common
    # convention in finite volume methods for defining boundary control volumes.
    x[1] = pts[1]
    x[n + 1] = pts[n]

    # For the interior, each half-point is the midpoint between two adjacent grid points.
    @simd ivdep for i in 2:n
        x[i] = (pts[i] + pts[i - 1]) * 0.5
    end

    return nothing
end

# Device counterpart: the whole computation (boundary entries and interior midpoints) runs
# in a single kernel over 1:(n+1), since scalar-reading `pts[1]`/`pts[n]` from the host is
# exactly the "Scalar indexing is disallowed" failure this exists to avoid.
function half_points!(x::AbstractVector, Ωₕ)
    n = npoints(Ωₕ)
    pts = points(Ωₕ)
    dev = ka_device(backend(Ωₕ))
    _launch_half_points!(x, pts, n, dev)
    return nothing
end

# Calculates the "half spacings" (cell widths/measures) for a 1D mesh.
# Fills `x` with the backward spacings of `Ωₕ`. Must run before half_spacing!, which
# reads them back through `spacing`.
@inline function spacing!(x::Array, Ωₕ::_Mesh1DLike)
    pts = points(Ωₕ)
    n = length(pts)
    T = eltype(Ωₕ)

    if is_collapsed(Ωₕ) || n < 2
        fill!(x, zero(T))
        return nothing
    end

    # The boundary convention: the first point has no interval behind it, so it repeats
    # the first interval instead. Hoisted out of the loop, as the sibling kernels in this
    # file (half_points!/half_spacing! above) already do for their own boundary entries.
    @inbounds begin
        x[1] = pts[2] - pts[1]
        @simd for i in 2:n
            x[i] = pts[i] - pts[i - 1]
        end
    end

    return nothing
end

# Device counterpart: the collapsed/short-mesh case is still a plain `fill!`, which is not
# scalar indexing and needs no kernel; only the general case goes through one.
function spacing!(x::AbstractVector, Ωₕ::_Mesh1DLike)
    pts = points(Ωₕ)
    n = length(pts)
    T = eltype(Ωₕ)

    if is_collapsed(Ωₕ) || n < 2
        fill!(x, zero(T))
        return nothing
    end

    dev = ka_device(backend(Ωₕ))
    _launch_spacing!(x, pts, n, dev)
    return nothing
end

@inline function half_spacing!(x::Array, Ωₕ)
    n = npoints(Ωₕ)

    # The boundary cell widths are defined as half of the spacing of the first/last interval.
    x[1] = spacing(Ωₕ, 1) * 0.5
    x[n] = spacing(Ωₕ, n) * 0.5

    # For interior points, the cell width is the average of the spacings of the two adjacent intervals.
    # This corresponds to the distance between the cell's half-points.
    @simd ivdep for i in 2:(n - 1)
        x[i] = (spacing(Ωₕ, i) + spacing(Ωₕ, i+1)) * 0.5
    end

    return nothing
end

# Device counterpart: reads the backing `spacings(Ωₕ)` vector directly (rather than calling
# the bounds-checked `spacing(Ωₕ, i)` accessor per element), so the kernel closes only over
# plain arrays and integers.
function half_spacing!(x::AbstractVector, Ωₕ)
    n = npoints(Ωₕ)
    h = spacings(Ωₕ)
    dev = ka_device(backend(Ωₕ))
    _launch_half_spacing!(x, h, n, dev)
    return nothing
end

# Fused device counterpart of `_points!` + `spacing!` + `half_points!` + `half_spacing!`
# together (gpena/Bramble.jl#303): one kernel launch fills `pts`, `half_pts`, `spacings`
# and `half_spacings` from `a`, `h` and `n` alone. `_mesh` below is the only caller, before
# a `Mesh1D` exists to pass in, which is why this takes the raw arrays rather than `Ωₕ`.
@inline function _uniform_mesh1d_init!(
        pts::AbstractVector,
        half_pts,
        spacings,
        half_spacings,
        a,
        h,
        n::Int,
        backend
)
    dev = ka_device(backend)
    _launch_uniform_mesh1d_init!(pts, half_pts, spacings, half_spacings, a, h, n, dev)
    return nothing
end

# Fused device counterpart of `spacing!` + `half_points!` + `half_spacing!` together
# (gpena/Bramble.jl#305): one kernel launch fills `spacings`, `half_pts` and
# `half_spacings` from `pts` alone, over `1:(n + 1)` work items, reading only each thread's
# own 3-point local stencil of `pts`. Dispatched from both `_mesh` (construction) and
# `set_points!`, for any device-backed mesh with at least two points -- the formula is
# correct whether or not the points happen to be evenly spaced.
@inline function _nonuniform_mesh1d_metrics!(Ωₕ::_Mesh1DLike)
    n = npoints(Ωₕ)
    pts = points(Ωₕ)
    dev = ka_device(backend(Ωₕ))
    _launch_nonuniform_mesh1d_metrics!(half_points(Ωₕ), spacings(Ωₕ), half_spacings(Ωₕ), pts, n, dev)
    return nothing
end

# Internal constructor function for creating a 1D mesh.
function _mesh(
        Ω::Domain{CartesianProduct{1, T}},
        npts::Tuple{Int},
        unif::Tuple{Bool},
        backend;
        warn_marker_mismatch::Bool = true
) where {T}
    # Unpack the domain's set and markers, and the number of points.
    (; set, markers) = Ω
    n_points, = npts

    # Check if the domain is a single point (topological dimension is 0).
    is_collapsed = topo_dim(set) == 0

    # If the domain is collapsed, force the number of points to be 1.
    if is_collapsed
        n_points = 1
    end

    # Unpack the uniformity flag.
    is_uniform, = unif

    # Allocate a vector for the grid points using the specified backend.
    pts = vector(backend, n_points)

    # Allocate vectors for derived quantities (cell centers and widths).
    _half_pts = vector(backend, n_points + 1)
    _half_spacings = vector(backend, n_points)
    _spacings = vector(backend, n_points)

    # A uniform, non-collapsed, device-backed mesh fills all four arrays above in the one
    # fused kernel (gpena/Bramble.jl#303); a non-uniform, non-collapsed, device-backed mesh
    # fills its three derived arrays in a different fused kernel instead
    # (gpena/Bramble.jl#305), once `pts` itself is generated below; every other case
    # (host-backed, or a single-point mesh) keeps going through the three separate
    # derived-quantity kernels/loops.
    fused_uniform_device = is_uniform && n_points >= 2 && !(pts isa Array)
    fused_nonuniform_device = !is_uniform && n_points >= 2 && !(pts isa Array)

    if fused_uniform_device
        a, b = extrema(set)
        h = (b - a) / (n_points - 1)
        _uniform_mesh1d_init!(
            pts,
            _half_pts,
            _spacings,
            _half_spacings,
            a,
            h,
            n_points,
            backend
        )
    elseif pts isa Array
        # Populate the vector with coordinates, either uniformly or non-uniformly.
        _points!(pts, set, is_uniform)
    else
        # The device method needs `backend` itself (to reach `ka_device`), which `pts`
        # alone does not carry.
        _points!(pts, set, is_uniform, backend)
    end

    # Generate the CartesianIndices for the grid.
    idxs = generate_indices(n_points)

    # Instantiate the Mesh1D struct with initial (empty) markers. The uniformity flag is set
    # below, once the spacings it is read from are filled.
    state = Mesh1DState(set, idxs, backend, pts, _half_pts, _half_spacings, _spacings,
        is_collapsed, 0, false, _no_marker_words(n_points), _next_mesh_uid())
    mesh = Mesh1D(MeshMarkers(), Dict{Symbol, Int}(), state)

    # The fused uniform path above already filled spacings/half_pts/half_spacings. A fused
    # non-uniform device mesh has `pts` filled (by `_points!` above) but still needs its
    # three derived arrays, from the one kernel `_nonuniform_mesh1d_metrics!` dispatches to
    # (gpena/Bramble.jl#305). Everything else still needs the three separate
    # derived-quantity passes; spacings come first there, since half_spacing! reads them
    # back through `spacing`.
    if fused_nonuniform_device
        _nonuniform_mesh1d_metrics!(mesh)
    elseif !fused_uniform_device
        spacing!(spacings(mesh), mesh)
        half_points!(half_points(mesh), mesh)
        half_spacing!(half_spacings(mesh), mesh)
    end
    _set_state!(mesh, _restate(_st(mesh); uniform = _computed_uniform(_st(mesh))))

    # Finally, apply the domain markers to the mesh points.
    set_markers!(mesh, markers; warn_marker_mismatch)
    return mesh
end

# The refinement fill itself, split out so it can dispatch on the destination's array type:
# the `Array` method is the original CPU loop, unchanged; the other goes through a kernel.
@inline function _refine_indices_fill!(new_points::Array, old_points, N_old, backend)
    @inbounds @simd for i in 1:N_old
        new_points[2i - 1] = old_points[i]
        if i < N_old
            new_points[2i] = (old_points[i] + old_points[i + 1]) * 0.5
        end
    end
    return nothing
end

function _refine_indices_fill!(new_points::AbstractVector, old_points, N_old, backend)
    dev = ka_device(backend)
    _launch_refine_indices!(new_points, old_points, N_old, dev)
    return nothing
end

# The geometric refinement alone, with markers left untouched: shared by both public
# methods below, neither of which wants the *other*'s marker handling as an intermediate
# step of its own.
function _refine_indices!(Ωₕ::Mesh1D)
    # Do nothing if the mesh is just a single point.
    if is_collapsed(Ωₕ)
        return nothing
    end

    N_old = npoints(Ωₕ)

    # No intervals to refine if there's only one point.
    if N_old <= 1
        return nothing
    end

    # Calculate the number of points in the new, refined mesh.
    N_new = 2 * N_old - 1

    # Allocate a new vector for the refined grid points.
    new_points = vector(backend(Ωₕ), N_new)
    old_points = points(Ωₕ)

    _refine_indices_fill!(new_points, old_points, N_old, backend(Ωₕ))

    # Generate new indices for the refined mesh.
    new_indices = generate_indices(N_new)

    # Update the mesh struct with the new indices and points.
    set_indices!(Ωₕ, new_indices)
    _set_points_geometry!(Ωₕ, new_points)
    return nothing
end

# A `Mesh1D` has nothing to refine when it is collapsed, or is a genuine single-point mesh
# over a non-degenerate domain (both leave `_refine_indices!` a no-op). The one real
# difference from `MeshnD`, which always has something to refine; the rest of the
# refinement/`change_points!` plumbing is shared, in `mesh/interface.jl`
# (gpena/Bramble.jl#68).
@inline _nothing_to_refine(Ωₕ::Mesh1D) = is_collapsed(Ωₕ) || npoints(Ωₕ) <= 1

function change_points!(Ωₕ::Mesh1D, pts)
    npts = npoints(Ωₕ)
    npts == length(pts) || _throw_point_count_mismatch(npts, length(pts))

    set_points!(Ωₕ, pts)
    return nothing
end

"""
    Base.copy(Ωₕ::Mesh1D) -> Mesh1D

Create a copy of mesh `Ωₕ`. The copy is shallow with respect to immutable fields
(`set`, `indices`, `backend`, `collapsed`), but deep with respect to mutable data fields
(`pts`, `half_pts`, `half_spacings`, `spacings`, `markers`). The copy keeps the version and
gets an identity of its own.
"""
function Base.copy(Ωₕ::Mesh1D)
    s = _st(Ωₕ)
    c = Mesh1DState(s.set, s.indices, s.backend, copy(s.pts), copy(s.half_pts),
        copy(s.half_spacings), copy(s.spacings), s.collapsed, s.version, s.uniform, s.words,
        _next_mesh_uid())
    return _mesh1d_with_markers(deepcopy(markers(Ωₕ)), c)
end

# A deepcopy is an independent mutable mesh, so it gets a fresh identity: with the uid
# copied, the original and the copy would share (uid, version) after one point change each
# while holding different points.
function Base.deepcopy_internal(Ωₕ::Mesh1D, dict::IdDict)
    haskey(dict, Ωₕ) && return dict[Ωₕ]::typeof(Ωₕ)
    c = invoke(Base.deepcopy_internal, Tuple{Any, IdDict}, Ωₕ, dict)::typeof(Ωₕ)
    s = _st(c)
    _set_state!(c,
        typeof(s)(s.set, s.indices, s.backend, s.pts, s.half_pts,
            s.half_spacings, s.spacings, s.collapsed, s.version, s.uniform, s.words,
            _next_mesh_uid()))
    return c
end

@inline Base.getindex(Ωₕ::_Mesh1DLike, i::Int) = point(Ωₕ, i)
@inline Base.getindex(Ωₕ::_Mesh1DLike, i::CartesianIndex{1}) = point(Ωₕ, i)

"""
    Base.show(io::IO, Ωₕ::Mesh1D) -> Nothing

Custom display for `Mesh1D` objects with detailed mesh summary, domain information, and markers.
"""
function Base.show(io::IO, Ωₕ::Mesh1D)
    print(io, "Mesh1D{", npoints(Ωₕ), " pts}")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", Ωₕ::Mesh1D{BT, CI, VT, T}) where {BT, CI, VT, T}
    return show_block(io) do io
        return _show_mesh1d_detailed(io, Ωₕ)
    end
end

function _show_mesh1d_detailed(io::IO, Ωₕ::Mesh1D{BT, CI, VT, T}) where {BT, CI, VT, T}
    pp = PrettyPrinter(io)

    n_pts = npoints(Ωₕ)
    topodim = topo_dim(Ωₕ)
    collapsed = is_collapsed(Ωₕ)

    # Header
    print_mesh_header(pp, "Mesh1D", 1, T, n_pts)
    println(io)

    # Summary line
    print_mesh_summary(pp, n_pts, topodim, collapsed)

    # Domain information
    print_mesh_domain_info(pp, set(Ωₕ))

    # Spacing information
    if !collapsed
        print_mesh_spacing_info(pp, is_uniform(Ωₕ), hₘₐₓ(Ωₕ))
    end

    # Markers information
    print_mesh_markers(pp, markers(Ωₕ))
    return nothing
end
