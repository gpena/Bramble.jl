@noinline _throw_domain_dim_mismatch(d::Int, D::Int) = throw(
    DimensionMismatch("the domain is $(d)-dimensional but npts and unif have length $D")
)

"""
    MeshnD{D, BT, CI, SM, T} <: AbstractMeshType{D}

Structured multi-dimensional tensor-product mesh for spatial dimensions ``D \\in \\{2, 3\\}``.

Constructed as a Cartesian product of 1D submeshes ([`Mesh1D`](@ref)). Coordinate points
are evaluated on demand from the tensor-product submeshes.

# Type parameters

  - `D`: Spatial dimension (2 or 3).
  - `BT <: Backend`: Computational linear algebra backend.
  - `CI <: CartesianIndices{D}`: Cartesian index space.
  - `SM <: Tuple`: Tuple of 1D submeshes (`Mesh1D`).
  - `T`: Coordinate element type (`Float64`, `Float32`, etc.).

# Fields

  - `set`: Multi-dimensional geometric [`CartesianProduct`](@ref) domain.
  - `markers`: [`MeshMarkers`](@ref) dictionary mapping symbols to `BitVector` indicators.
  - `indices`: Multi-dimensional `CartesianIndices{D}` for the grid.
  - `backend`: Linear algebra [`Backend`](@ref).
  - `submeshes`: Tuple of `D` [`Mesh1D`](@ref) objects along each coordinate axis.
  - `marker_ids`: `Dict{Symbol, Int}`, the column of each label in `words`.
  - `words`: `Matrix{UInt64}`, the marker bits, one column of `BitVector` chunks per label.
  - `uid`: an identity unique to this mesh, kept across its mutations.
  - `marker_stamp`: a counter value drawn afresh at construction and at every marker
    replacement.
  - `predicate_labels`: the labels `set_markers!` defined by a predicate of position, whose
    sets a point change leaves stale (see [`change_points!`](@ref)).

A `MeshnD` stores no state of its own: its submeshes can be mutated directly, so the
[`MeshnDState`](@ref) the walk reads is built on demand by [`_walk_mesh`](@ref).

# Examples

```julia
using Bramble: point
# Create a 2D mesh with 20×30 grid points
X = domain(interval(0, 1) × interval(0, 2))
Ωₕ = mesh(X, (20, 30), (true, false))

# Access submeshes
x_mesh = Ωₕ(1)  # 1D mesh along x-axis
y_mesh = Ωₕ(2)  # 1D mesh along y-axis

# Query a specific point coordinate
point(Ωₕ, (10, 15))  # returns (x₁₀, y₁₅)
```

See also: [`Mesh1D`](@ref), [`submeshes`](@ref), [`mesh`](@ref).
"""
mutable struct MeshnD{D, BT <: Backend, CI <: CartesianIndices{D}, SM <: Tuple, T} <:
               AbstractMeshType{D}
    "the D-dimensional CartesianProduct (hyperrectangle) defining the geometric domain."
    set::CartesianProduct{D, T}
    "a dictionary mapping `Symbol` labels to `BitVector`s, marking grid points."
    markers::MeshMarkers
    "the `CartesianIndices` for the full D-dimensional grid, for multi-dimensional indexing."
    indices::CI
    "the computational backend used for linear algebra operations."
    backend::BT
    "a tuple of `D` 1D mesh objects, representing the grid along each spatial dimension."
    submeshes::SM
    "the column of each marker label in `words`."
    marker_ids::Dict{Symbol, Int}
    "the marker bits, one column of `BitVector` chunks per label (`_marker_id`)."
    words::Matrix{UInt64}
    "an identity unique to this mesh, kept across its mutations."
    uid::UInt64
    "a fresh counter value at construction and at every marker replacement."
    marker_stamp::UInt64
    "the labels defined by a predicate of position, which moving the points leaves stale."
    predicate_labels::Vector{Symbol}
end

# The mesh over the given marker table and words, with a fresh identity and marker stamp,
# and the predicate labels given or none.
function MeshnD(
        set::CartesianProduct{D}, markers::MeshMarkers, indices::CartesianIndices{D},
        backend::Backend, submeshes::Tuple, marker_ids::Dict{Symbol, Int},
        words::Matrix{UInt64}, uid::UInt64, predicate_labels::Vector{Symbol} = Symbol[]
) where {D}
    return MeshnD(set, markers, indices, backend, submeshes, marker_ids, words, uid,
        _next_mesh_uid(), predicate_labels)
end

"""
    MeshnD(set, markers, indices, backend, submeshes) -> MeshnD

A mesh over the given submeshes, with the marker table and words built from `markers` and a
fresh identity.
"""
function MeshnD(
        set::CartesianProduct{D}, markers::MeshMarkers, indices::CartesianIndices{D},
        backend::Backend, submeshes::Tuple, predicate_labels::Vector{Symbol} = Symbol[]
) where {D}
    ids, words = _marker_table(markers, length(indices))
    return MeshnD(set, markers, indices, backend, submeshes, ids, words, _next_mesh_uid(),
        predicate_labels)
end

"""
    MeshnDState{D, BT, CI, SM, T, WT} <: AbstractMeshType{D}

The immutable walk state of a [`MeshnD`](@ref) (gpena/Bramble.jl#437), built on demand by
[`_walk_mesh`](@ref): the set, parent indices and backend, the tuple of its submeshes'
[`Mesh1DState`](@ref)s, its marker word matrix, its version (the sum of the submeshes') and
its identity. Plain arrays and isbits values only. It answers every geometric accessor a
`MeshnD` does, through its submesh states. The word matrix's type `WT` is a parameter so a
rebuilt state can hold another array type ([`_batch_rebuild`](@ref)).
"""
struct MeshnDState{D, BT <: Backend, CI <: CartesianIndices{D}, SM <: Tuple, T,
    WT <: AbstractMatrix{UInt64}} <: AbstractMeshType{D}
    "the D-dimensional CartesianProduct (hyperrectangle) defining the geometric domain."
    set::CartesianProduct{D, T}
    "the `CartesianIndices` for the full D-dimensional grid."
    indices::CI
    "the computational backend used for linear algebra operations."
    backend::BT
    "the tuple of the submeshes' `Mesh1DState`s."
    submeshes::SM
    "the marker bits, one column of `BitVector` chunks per label (`_marker_id`)."
    words::WT
    "the mesh version when this state was built: the sum of the submeshes'."
    version::Int
    "the identity of the `MeshnD` this state was built from."
    uid::UInt64
end

@inline function _walk_mesh(Ωₕ::MeshnD)
    return MeshnDState(Ωₕ.set, Ωₕ.indices, Ωₕ.backend, map(_walk_mesh, Ωₕ.submeshes),
        Ωₕ.words, _mesh_version(Ωₕ), Ωₕ.uid)
end
@inline _walk_mesh(s::MeshnDState) = s

# Both a `MeshnD` and its state answer the geometric accessors below, through `Ωₕ(i)`.
const _MeshnDLike{D} = Union{MeshnD{D}, MeshnDState{D}}

@inline _marker_words(Ωₕ::_MeshnDLike) = Ωₕ.words
@inline _marker_ids(Ωₕ::MeshnD) = Ωₕ.marker_ids

# The stamp of `Ωₕ`'s marker words: redrawn by every `_store_markers!`, never by a geometry
# change, and never equal across meshes or across two stores.
@inline _marker_stamp(Ωₕ::MeshnD) = Ωₕ.marker_stamp

# The labels `set_markers!` defined by a predicate of position; every other store clears them.
@inline _predicate_labels(Ωₕ::MeshnD) = Ωₕ.predicate_labels

# Tells each submesh the predicate labels of its parent.
function _stamp_submeshes!(submeshes::Tuple, labels::Vector{Symbol})
    for s in submeshes
        setfield!(s, :parent_predicate_labels, labels)
    end
    return nothing
end

# A `MeshnD` is nobody's submesh: no labels, and no vector built to say so.
@inline _parent_predicate_labels(::MeshnD) = ()

# Replaces the marker dictionary, its label table and the word matrix together. A label set
# may change after construction; the word matrix is rebuilt, never resized.
function _store_markers!(Ωₕ::MeshnD, mesh_markers, predicate_labels = Symbol[])
    mm = convert(MeshMarkers, mesh_markers)
    ids, words = _marker_table(mm, npoints(Ωₕ))
    Ωₕ.predicate_labels = Symbol[predicate_labels...]
    # The submeshes are shared with whoever holds `Ωₕ(i)`, and moving one alone would leave
    # these labels stale, so each learns them (every other store clears them).
    _stamp_submeshes!(Ωₕ.submeshes, Ωₕ.predicate_labels)
    Ωₕ.markers = mm
    Ωₕ.marker_ids = ids
    Ωₕ.words = words
    Ωₕ.marker_stamp = _next_mesh_uid()
    return nothing
end

# Re-marks in place when the label set is unchanged (`_remark_markers!`, marker.jl).
_remark_in_place!(Ωₕ::MeshnD, domain_markers, fitted::Tuple) = _remark_markers!(
    Ωₕ, domain_markers, fitted)

"""
    submeshes(Ω::Domain, npts, unif, backend) -> NTuple{D, Mesh1D}

Create the component 1D submeshes for a tensor-product grid.

Generates a tuple of `D` independent [`Mesh1D`](@ref) objects corresponding to each coordinate axis of `Ω`.

# Arguments

  - `Ω`: Multi-dimensional continuous [`Domain`](@ref).
  - `npts`: Number of points along each dimension.
  - `unif`: Flags indicating whether each axis is uniformly partitioned.
  - `backend`: Computational linear algebra [`Backend`](@ref).
"""
@inline function submeshes(Ω::Domain, npts, unif, backend)
    # Use ntuple for a type-stable way to generate the tuple of 1D meshes.
    # For each dimension `i` from 1 to D:
    # 1. `projection(Ω, i)` gets the i-th 1D interval from the domain's set.
    # 2. `domain(...)` wraps it in a Domain object.
    # 3. `mesh(...)` creates the corresponding Mesh1D for that dimension.
    return ntuple(
        i -> mesh(domain(projection(Ω, i)), npts[i], unif[i], backend = backend), Val(dim(Ω))
    )
end

"""
    _mesh(Ω::Domain, npts::NTuple{D, Int}, unif::NTuple{D, Bool}, backend) -> MeshnD{D}

Internal constructor for multi-dimensional tensor-product mesh [`MeshnD`](@ref).

Builds the 1D submeshes along each axis and combines them into a [`MeshnD`](@ref). Collapsed
dimensions (degenerate single-point intervals) are forced to a point count of 1.

# Arguments

  - `Ω`: Multi-dimensional continuous [`Domain`](@ref) to discretize.
  - `npts`: Number of points in each spatial dimension.
  - `unif`: Uniformity flags for each spatial dimension.
  - `backend`: Linear algebra [`Backend`](@ref).
"""
function _mesh(
        Ω::Domain,
        npts::NTuple{D, Int},
        unif::NTuple{D, Bool},
        backend;
        warn_marker_mismatch::Bool = true
) where {D}
    dim(Ω) == D || _throw_domain_dim_mismatch(dim(Ω), D)
    _set = set(Ω)

    # Adjust the number of points for any collapsed dimensions. For example, if a domain
    # is a line in 3D space, the two collapsed dimensions will have npts = 1. Collapse is
    # judged in the storage eltype, exactly as the 1D `_mesh` building each submesh does.
    npts_with_collapsed = ntuple(i -> _storage_collapsed(_set(i)..., backend) ? 1 : npts[i],
        Val(D))

    idxs = generate_indices(npts_with_collapsed)

    _submeshes = submeshes(Ω, npts_with_collapsed, unif, backend)
    idxs, fitted = _close_grid!(_submeshes, regions(markers(Ω)), idxs)

    mesh_markers = MeshMarkers()
    output_mesh = MeshnD(_set, mesh_markers, idxs, backend, _submeshes)

    _set_markers!(output_mesh, markers(Ω), fitted; warn_marker_mismatch)

    return output_mesh
end

# A domain without polygon or level-set markers keeps the submeshes it was given, so a plain
# mesh has the points (and draws the random numbers) it always had. The second value is the
# polygon fitted to each region marker, `nothing` for a polygon marker.
@inline _close_grid!(_, ::Tuple{}, idxs) = (idxs, ())

# A domain with polygon markers closes its grid lines (`_grid_closure`): each submesh takes
# the closed lines, and the indices follow the new point counts. A level-set marker fits the
# lines to its curve instead (`_levelset_grid`, on the submeshes' points as seeds), and the
# polygon it returns labels the nodes. One level-set marker per domain, with no polygon
# marker beside it.
function _close_grid!(subs::NTuple{2, Mesh1D}, regs::Tuple{Any, Vararg}, idxs)
    ids = map(identifier, regs)
    nls = count(P -> P isa LevelSet, ids)
    nls == 0 && return _close_polygon_grid!(subs, ids, regs, idxs)
    (nls == 1 && length(ids) == 1) || _throw_levelset_markers(nls, length(ids) - nls)
    xs, ys = host_points(subs[1]), host_points(subs[2])
    X, Y, P = _levelset_grid(only(ids).φ, xs, ys)
    _set_points!(subs[1], X)
    _set_points!(subs[2], Y)
    return generate_indices((length(X), length(Y))), (P,)
end

function _close_polygon_grid!(subs, ids, regs, idxs)
    xs, ys = _grid_closure(ids, host_points(subs[1]), host_points(subs[2]))
    _set_points!(subs[1], xs)
    _set_points!(subs[2], ys)
    return generate_indices((length(xs), length(ys))), map(_ -> nothing, regs)
end

@noinline function _throw_levelset_markers(nls, npoly)
    throw(ArgumentError("a domain takes one level-set marker and no polygon marker beside it, " *
                        "got $nls level-set markers and $npoly polygon markers: the grid " *
                        "lines are fitted to one curve."))
end

"""
    _grid_closure(P, xs, ys) -> (Vector{T}, Vector{T})

Close the grid lines `xs` and `ys` (sorted, end points included) over the polygon `P`, or a
tuple of polygons and level sets, so that every crossing of a grid line with ``\\partial P``
is a grid node. `T` is the floating-point type of `xs` and `ys`; the vertices are converted
to `T` first.

The end points and the vertex coordinates are lines first, then the lines the closure
computes: a grid line crossing an oblique edge, or the curve ``\\varphi = 0`` of a level set
(`_crossings`, between the end-point lines), adds the crossing's other coordinate as a
line, until no line is new. The remaining entries of `xs` and `ys` are seeds. Lines closer
than the merge tolerance ``\\max(10^{-10}, 64\\varepsilon) L`` (``L`` the polygons' extent,
``\\varepsilon`` the larger epsilon of `T` and of the vertices' type, whose rounding a
vertex carries; with no vertices, ``L`` is the larger span of the end-point lines) are one
line. A seed that meets a vertex or end-point line, or whose lines meet those of a vertex
or an earlier seed, within the tolerance but not exactly, is dropped and the closure
restarts without it: the seed gives way, never the vertex or the line
computed from it. Two lines that both come from the end points and vertices and lie within
the tolerance are one line. Polygon detail finer than the tolerance is merged, as is
rounding, since floating point cannot tell the two apart. Every vertex coordinate is a line,
and every crossing is on a node, up to the merge tolerance.

The oblique edges must share one slope magnitude, a necessary condition only: with equal
slopes the closure may still grow without bound, or beyond the cap. A level set closes only
when the orbits of its crossing map are finite; otherwise the cap is reached.

# Throws

  - `ArgumentError`: if the oblique edges do not share one slope magnitude, or if the
    closure exceeds a number of lines proportional to the input (the grid
    closure of a polygon or level set is unbounded or too fine for this grid).
"""
_grid_closure(P::Union{Polygon, LevelSet}, xs, ys) = _grid_closure((P,), xs, ys)

function _grid_closure(Ps::Tuple, xs::AbstractVector, ys::AbstractVector)
    T = float(promote_type(eltype(xs), eltype(ys)))
    polys = filter(P -> P isa Polygon, Ps)
    sets = filter(P -> P isa LevelSet, Ps)
    V = NTuple{2, T}[(T(v[1]), T(v[2])) for P in polys for v in P.vertices]
    E = _oblique_edges(T, polys)
    _check_closure_slopes(E)

    sx, sy = sort!(T.(collect(xs))), sort!(T.(collect(ys)))
    L = if isempty(V)
        max(sx[end] - sx[1], sy[end] - sy[1])
    else
        max(maximum(first, V) - minimum(first, V), maximum(last, V) - minimum(last, V))
    end
    ε = T(max(eps(T), maximum(P -> eps(eltype(eltype(P.vertices))), polys; init = zero(T))))
    tol = max(T(1e-10), 64 * ε) * L
    fixed = (T[sx[1]; sx[end]; first.(V)], T[sy[1]; sy[end]; last.(V)])
    seeds = (sx[2:(end - 1)], sy[2:(end - 1)])
    keep = trues(length(seeds[1]) + length(seeds[2]))
    cap = 64 * (length(sx) + length(sy) + length(V)) + 1000
    cross! = _Crossings(E, sets, (sx[1], sy[1]), (sx[end], sy[end]))

    while true
        lines, loser = _closure_pass(fixed, seeds, keep, cross!, tol, cap)
        iszero(loser) && return lines
        keep[loser] = false
    end
end

# The crossing rule of a closure: `cross!(out, a, v)` empties `out` and fills it with the
# other coordinate of every crossing of the line `x_a = v` with the oblique edges `E` of the
# polygons (in edge order) and then with each level set `sets`, whose curve is searched
# between the end-point lines `lo` and `hi` of the other axis.
struct _Crossings{T, S <: Tuple}
    E::Vector{NTuple{4, T}}
    sets::S
    lo::NTuple{2, T}
    hi::NTuple{2, T}
end

function (R::_Crossings{T})(out::Vector{T}, a::Int, v::T) where {T}
    empty!(out)
    for (ax, ay, bx, by) in R.E
        # The edge as a map from coordinate `a` to coordinate `b`.
        p₀, q₀, p₁, q₁ = a == 1 ? (ax, ay, bx, by) : (ay, ax, by, bx)
        min(p₀, p₁) < v < max(p₀, p₁) || continue
        push!(out, q₀ + (q₁ - q₀) * (v - p₀) / (p₁ - p₀))
    end
    b = 3 - a
    for S in R.sets
        append!(out, _crossings(S.φ, a, v, R.lo[b], R.hi[b]))
    end
    return out
end

# The oblique edges (neither horizontal nor vertical) of the polygons `Ps`, in type `T`.
function _oblique_edges(::Type{T}, Ps::Tuple) where {T}
    E = NTuple{4, T}[]
    for P in Ps
        W = P.vertices
        n = length(W)
        for k in 1:n
            a, b = W[k], W[mod1(k + 1, n)]
            ax, ay, bx, by = T(a[1]), T(a[2]), T(b[1]), T(b[2])
            (ax == bx || ay == by) || push!(E, (ax, ay, bx, by))
        end
    end
    return E
end

# Equal slope magnitudes are necessary for a fixpoint: the maps between crossings are then
# translations and reflections, though their orbits may still be unbounded or very long
# (the cap in `_closure_pass` catches those). Mixed magnitudes contract an orbit towards a
# point instead, adding lines until the merge tolerance absorbs them.
function _check_closure_slopes(E::Vector{NTuple{4, T}}) where {T}
    isempty(E) && return nothing
    s = [abs((e[4] - e[2]) / (e[3] - e[1])) for e in E]
    all(m -> isapprox(m, s[1]; rtol = sqrt(eps(T))), s) && return nothing
    throw(ArgumentError("Polygon grid closure: the oblique edges have slopes of magnitudes " *
                        "$(unique(s)); every crossing of a grid line with the polygon " *
                        "becomes a grid node only when all oblique edges share one slope " *
                        "magnitude."))
end

# One closure from the fixed lines and the kept seeds. Returns the lines and 0, or, as soon
# as a seed must give way, the lines so far and the index of that seed.
function _closure_pass(fixed::NTuple{2, Vector{T}}, seeds::NTuple{2, Vector{T}}, keep::BitVector,
        cross!, tol::T, cap::Int) where {T}
    pos = (T[], T[])
    root = (Int[], Int[])
    queue = Tuple{Int, T, Int}[]
    for a in 1:2, c in fixed[a]

        _closure_insert!(pos[a], root[a], queue, a, c, 0, tol)
    end
    k = 0
    for a in 1:2, c in seeds[a]

        k += 1
        keep[k] || continue
        loser = _closure_insert!(pos[a], root[a], queue, a, c, k, tol)
        loser > 0 && return pos, loser
    end

    added = 0
    q = 1
    cs = T[]
    while q <= length(queue)
        a, v, r = queue[q]
        q += 1
        b = 3 - a
        for c in cross!(cs, a, v)
            loser = _closure_insert!(pos[b], root[b], queue, b, c, r, tol)
            loser > 0 && return pos, loser
            if loser < 0
                added += 1
                added > cap && _throw_closure_cap(cap)
            end
        end
    end
    return pos, 0
end

@noinline _throw_closure_cap(cap) = throw(ArgumentError("Grid closure: the " *
                                                        "closure exceeded $cap added grid lines (a cap proportional to the input); the " *
                                                        "region's grid closure is unbounded or too fine for this grid."))

# Insert the line `c`, grown from root `r` (0 for an end point or a vertex, k for the k-th
# seed), into the sorted lines `pos` of axis `a`. Returns -1 when it is new, 0 when it is
# one already there (exactly equal, or from the same root, so two root-0 lines within `tol`
# are one), and otherwise the seed that gives way: the seed root against root 0, the later
# of two seeds.
function _closure_insert!(
        pos::Vector{T}, root::Vector{Int}, queue::Vector{Tuple{Int, T, Int}}, a::Int, c::T,
        r::Int, tol::T) where {T}
    i = searchsortedfirst(pos, c)
    j = (i <= length(pos) && pos[i] - c <= tol) ? i : 0
    if i > 1 && c - pos[i - 1] <= tol && (j == 0 || c - pos[i - 1] < pos[i] - c)
        j = i - 1
    end
    if iszero(j)
        insert!(pos, i, c)
        insert!(root, i, r)
        push!(queue, (a, c, r))
        return -1
    end
    rj = root[j]
    (pos[j] == c || rj == r) && return 0
    return iszero(rj) ? r : iszero(r) ? rj : max(r, rj)
end

@inline eltype(::MeshnD{D, BT}) where {D, BT} = eltype(BT)
@inline eltype(::MeshnDState{D, BT}) where {D, BT} = eltype(BT)
@inline eltype(::Type{<:MeshnD{D, BT}}) where {D, BT} = eltype(BT)
@inline eltype(::Type{<:MeshnDState{D, BT}}) where {D, BT} = eltype(BT)

"""
    (Ωₕ::MeshnD)(i::Integer) -> Mesh1D

Return the `i`-th 1D submesh of `Ωₕ` along coordinate axis `i`.
"""
@inline function (Ωₕ::MeshnD{D})(i) where {D}
    @boundscheck 1 <= i <= D || throw(BoundsError(Ωₕ.submeshes, i))
    return @inbounds Ωₕ.submeshes[i]
end

@inline function (s::MeshnDState{D})(i) where {D}
    @boundscheck 1 <= i <= D || throw(BoundsError(s.submeshes, i))
    return @inbounds s.submeshes[i]
end

# See `_mesh_version`'s own docstring (mesh1d.jl): a `MeshnD` has no point storage of its
# own, so its version is the sum of its submeshes' -- strictly increasing whenever any one
# axis is mutated (`change_points!(Ωₕ::MeshnD, ...)` delegates per-axis, so this stays
# correct however many axes actually change), with no second counter to keep in sync.
# `Ωₕ.submeshes` is a `Tuple`, so this unrolls at compile time and allocates nothing.
@inline _mesh_version(Ωₕ::MeshnD) = sum(_mesh_version, Ωₕ.submeshes)
@inline _mesh_version(s::MeshnDState) = s.version

# Refining or resizing one submesh in place (`iterative_refinement!(Ωₕ(1))`) changes that
# axis's point count, but the parent's `indices` and `markers` are sized for the whole grid
# and only `_refine_indices!(::MeshnD)` rebuilds them. A submesh holds no reference to its
# parent, so the mutation itself cannot be refused; instead `gridspace` checks the parent
# here, before a space, a form and `assemble` build on a grid that no longer matches its
# own index set and return a wrong matrix rather than an error.
@inline function _check_submesh_sizes(Ωₕ::MeshnD)
    size(indices(Ωₕ)) == npoints(Ωₕ, Tuple) || _throw_submesh_resized(Ωₕ)
    return nothing
end

@noinline function _throw_submesh_resized(Ωₕ::MeshnD)
    throw(
        ArgumentError(
        "the $(dim(Ωₕ))D mesh was built with $(size(indices(Ωₕ))) points, but its " *
        "submeshes now have $(npoints(Ωₕ, Tuple)): one submesh was refined or resized " *
        "in place (e.g. iterative_refinement!(Ωₕ(1))), which leaves the mesh's indices " *
        "and markers sized for the old grid. Refine the whole mesh with " *
        "iterative_refinement!(Ωₕ), or build a new mesh with the points you want.",
    ),
    )
end

#------------------------------------------------------------------------------------------#
# Macros for Boilerplate Reduction
#
# These macros generate functions that apply 1D mesh operations to all submeshes of a
# multidimensional mesh. They eliminate repetitive code and ensure type-stability.
#
# Usage patterns:
# - `@generate_mesh_ntuple_func`: For functions returning tuples of values (one per dimension)
#   Example: points(Ωₕ) returns (x_points, y_points, z_points)
#
# - `@generate_mesh_ntuple_func_with_idx`: For indexed operations returning tuples
#   Example: point(Ωₕ, idx) returns (x[idx[1]], y[idx[2]], z[idx[3]])
#------------------------------------------------------------------------------------------#

# A macro for functions of the form: func(Ωₕ) -> ntuple(...)
macro generate_mesh_ntuple_func(fname)
    return esc(
        quote
        @inline $fname(Ωₕ::_MeshnDLike{D}) where {D} = ntuple(i -> $fname(Ωₕ(i)), Val(D))
    end
    )
end

# A macro for functions of the form: func(Ωₕ, idx) -> ntuple(...)
macro generate_mesh_ntuple_func_with_idx(fname)
    return esc(
        quote
        @inline $fname(Ωₕ::_MeshnDLike{D}, idx) where {D} = ntuple(i -> $fname(Ωₕ(i), idx[i]), Val(D))
    end,
    )
end

@generate_mesh_ntuple_func points
@generate_mesh_ntuple_func half_points
@generate_mesh_ntuple_func half_spacings

"""
    host_points(Ωₕ::MeshnD{D}) -> NTuple{D, Vector}

Return the per-axis [`host_points`](@ref)`(Ωₕ(i))` of each submesh, one bulk transfer per
axis regardless of where that axis's storage lives (gpena/Bramble.jl#308).
"""
@generate_mesh_ntuple_func host_points

"""
    spacings(Ωₕ::MeshnD{D}) -> NTuple{D, AbstractVector}

Return the per-axis backward spacings as an `NTuple{D}` of vectors, where
`spacings(Ωₕ)[d][i]` is [`spacing`](@ref)`(Ωₕ(d), i)`.

See also: [`half_spacings`](@ref), [`cell_measures`](@ref).
"""
@generate_mesh_ntuple_func spacings
@generate_mesh_ntuple_func forward_spacings

@generate_mesh_ntuple_func_with_idx point
@generate_mesh_ntuple_func_with_idx half_point
@generate_mesh_ntuple_func_with_idx spacing
@generate_mesh_ntuple_func_with_idx forward_spacing

# Single-axis spacings, queried straight from submesh `dim` instead of building the full
# `D`-tuple above and discarding the other `D - 1` entries (gpena/Bramble.jl#111).
@inline spacing(Ωₕ::_MeshnDLike, idx, dim::Int) = spacing(Ωₕ(dim), idx[dim])
@inline forward_spacing(Ωₕ::_MeshnDLike, idx, dim::Int) = forward_spacing(Ωₕ(dim), idx[dim])

@inline half_spacing(Ωₕ::_MeshnDLike{D}, idx) where {D} = ntuple(i -> _apply_hs_logic(half_spacing(Ωₕ(i), idx[i])), Val(D))

"""
    cell_measures(Ωₕ::MeshnD{D}) -> NTuple{D, AbstractVector}

Return the per-axis cell widths as an `NTuple{D}` of vectors. The measure of an
individual cell is the product of its per-axis widths; see [`cell_measure`](@ref).
"""
@inline cell_measures(Ωₕ::_MeshnDLike{D}) where {D} = ntuple(i -> cell_measures(Ωₕ(i)), Val(D))

@inline npoints(Ωₕ::_MeshnDLike) = prod(npoints(Ωₕ, Tuple))
@inline npoints(Ωₕ::_MeshnDLike{D}, ::Type{Tuple}) where {D} = ntuple(i -> npoints(Ωₕ(i)), Val(D))

# The diagonal of the largest cell. On a tensor-product mesh the spacing along axis d does
# not depend on the other coordinates, and `hypot` is increasing in each argument, so the
# maximum over the whole index set is attained at the per-axis maxima and there is no need
# to visit every cell:
#
#     max_idx ‖(h₁,ᵢ₁, …, h_D,i_D)‖₂ = hypot(hₘₐₓ(Ωₕ(1)), …, hₘₐₓ(Ωₕ(D)))
#
# Each submesh reads its own maximum off its cached spacings, which turns a pass over
# prod(Nd) cells into D lookups.
@inline hₘₐₓ(Ωₕ::_MeshnDLike{D}) where {D} = hypot(ntuple(i -> hₘₐₓ(Ωₕ(i)), Val(D))...)

# The diagonal of the smallest cell, the counterpart of `hₘₐₓ` above and computed the same
# way: `hypot` is increasing in each argument and the per-axis index sets are independent,
# so the minimum over the whole index set sits at the per-axis minima.
#
# This is a diagonal, not an edge length. For a per-axis extent, ask a submesh:
# `hₘᵢₙ(Ωₕ(i))`.
@inline hₘᵢₙ(Ωₕ::_MeshnDLike{D}) where {D} = hypot(ntuple(i -> hₘᵢₙ(Ωₕ(i)), Val(D))...)

@inline function cell_measure(Ωₕ::_MeshnDLike{D}, idx) where {D}
    # Routed through the coerced, tuple-returning `half_spacing(::MeshnD, idx)` above
    # (not the raw per-submesh `half_spacing(Ωₕ(i), idx[i])`), so a collapsed axis's
    # zero half-spacing is replaced by `_apply_hs_logic` before the product, not left
    # to make the whole cell measure zero.
    return prod(half_spacing(Ωₕ, idx))
end

@inline Base.getindex(Ωₕ::_MeshnDLike, idx::CartesianIndex) = point(Ωₕ, idx)
@inline Base.getindex(Ωₕ::_MeshnDLike, idx...) = point(Ωₕ, idx)

function locate_cell(Ωₕ::_MeshnDLike{D}, x::NTuple{D, Real}) where {D}
    indices_tuple = ntuple(i -> locate_cell(Ωₕ(i), x[i]), Val(D))
    return CartesianIndex(indices_tuple)
end

# Disambiguates against `locate_cell(::AbstractMeshType{1}, ::Tuple{Real})`
# (mesh/queries.jl): both match `(MeshnD{1}, Tuple{Real})` and neither is more specific
# than the other, so `Test.detect_ambiguities` flags the pair. `MeshnD{1}` is never
# actually produced by the public `mesh()` constructor (`_mesh` routes `D == 1` to
# `Mesh1D` instead), so this exists only to make the method table unambiguous, with the
# same body the generic method above would have run for `D == 1`.
@inline function locate_cell(Ωₕ::_MeshnDLike{1}, x::NTuple{1, Real})
    return CartesianIndex(locate_cell(Ωₕ(1), x[1]))
end

# The geometric refinement, with the parent's markers left untouched. Both public
# methods below share it, and neither wants the *other*'s marker handling as an
# intermediate step of its own. Each axis submesh gets its geometric markers reseeded
# for its refined size, so its marker vectors and words match its points under either
# form. Submeshes are built from `domain(projection(Ω, i))` and carry only the boundary
# and interior labels, so a label a user `markers!`'d onto `Ωₕ(i)` is dropped here.
# Every axis is filled and checked for ties before any is committed, so a tie on a later
# axis leaves the earlier ones unrefined.
function _refine_indices!(Ωₕ::MeshnD{D}) where {D}
    new_points = ntuple(i -> _refined_points(Ωₕ(i)), Val(D))
    @inbounds for i in 1:D
        _commit_refined_points!(Ωₕ(i), new_points[i])
        submesh_markers = MeshMarkers()
        _ensure_geometric_markers!(submesh_markers, Ωₕ(i))
        markers!(Ωₕ(i), submesh_markers)
    end

    # Each submesh regenerated its own indices, but the parent holds a CartesianIndices
    # spanning the whole grid and it has to be rebuilt from the new sizes. Without this
    # the mesh is left inconsistent: npoints reports the refined count while indices
    # still spans the old one, so everything that iterates indices(Ωₕ), which is every
    # restriction and every operator, writes only the old index set and leaves the rest
    # of a grid function holding whatever was in the fresh allocation.
    set_indices!(Ωₕ, generate_indices(npoints(Ωₕ, Tuple)))
    return nothing
end

# A `MeshnD` always has something to refine (each axis handles its own collapse
# independently, inside `_refine_indices!` above); the rest of the refinement/
# `change_points!` plumbing is shared with `Mesh1D`, in `mesh/interface.jl`
# (gpena/Bramble.jl#68). `_nothing_to_refine`'s default (`interface.jl`) already answers
# `false` for any mesh type that does not override it, so no override is needed here.

# Every axis is checked, for its count and for its values, before any is mutated, so a
# refusal on a later axis leaves the earlier ones and the version as they were, as
# `_refine_indices!` does for a tie.
function change_points!(Ωₕ::MeshnD, pts)
    _check_no_predicate_labels(Ωₕ,
        "change_points!(Ωₕ, pts) was asked to move",
        "change_points!(Ωₕ, markers(Ω), pts)")
    return _change_points!(Ωₕ, pts)
end

# The geometry alone, with markers left as they are (see the `Mesh1D` method).
function _change_points!(Ωₕ::MeshnD{D}, pts) where {D}
    @inbounds for i in 1:D
        n = npoints(Ωₕ(i))
        n == length(pts[i]) || _throw_point_count_mismatch(n, length(pts[i]))
    end
    @inbounds for i in 1:D
        _check_set_points(Ωₕ(i), pts[i])
    end
    @inbounds for i in 1:D
        _change_points!(Ωₕ(i), pts[i])
    end
    return nothing
end

"""
    Base.copy(Ωₕ::MeshnD{D}) -> MeshnD{D}

Create a copy of mesh `Ωₕ`. The copy is shallow with respect to immutable fields
(`set`, `indices`, `backend`), but deep with respect to mutable data fields
(`submeshes`, `markers`). The copy and each of its submeshes get an identity of their own.
"""
function Base.copy(Ωₕ::MeshnD{D}) where {D}
    c = MeshnD(
        Ωₕ.set, deepcopy(Ωₕ.markers), Ωₕ.indices, Ωₕ.backend, map(copy, Ωₕ.submeshes),
        copy(Ωₕ.predicate_labels)
    )
    _stamp_submeshes!(c.submeshes, c.predicate_labels)
    return c
end

# A deepcopy is an independent mutable mesh with a fresh identity and marker stamp; its
# submeshes get theirs through the `Mesh1D` method, and its version stays the sum of theirs
# (each keeps its own).
function Base.deepcopy_internal(Ωₕ::MeshnD, dict::IdDict)
    haskey(dict, Ωₕ) && return dict[Ωₕ]::typeof(Ωₕ)
    c = invoke(Base.deepcopy_internal, Tuple{Any, IdDict}, Ωₕ, dict)::typeof(Ωₕ)
    c.uid = _next_mesh_uid()
    c.marker_stamp = _next_mesh_uid()
    return c
end

"""
    Base.show(io::IO, Ωₕ::MeshnD) -> Nothing

Custom display for `MeshnD` objects with detailed mesh summary, domain information, and markers.
"""
function Base.show(io::IO, Ωₕ::MeshnD{D}) where {D}
    print(io, "MeshnD{$(D)D, ", npoints(Ωₕ), " pts}")
    return nothing
end

function Base.show(
        io::IO, ::MIME"text/plain", Ωₕ::MeshnD{D, BT, CI, SM, T}
) where {D, BT, CI, SM, T}
    return show_block(io) do io
        return _show_meshnd_detailed(io, Ωₕ)
    end
end

function _show_meshnd_detailed(io::IO, Ωₕ::MeshnD{D, BT, CI, SM, T}) where {D, BT, CI, SM, T}
    pp = PrettyPrinter(io)

    npts_tuple = npoints(Ωₕ, Tuple)
    n_total = npoints(Ωₕ)
    topodim = topo_dim(Ωₕ)

    # Check if all dimensions are collapsed (topological dimension is 0)
    collapsed = (topodim == 0)

    print_mesh_header(pp, "MeshnD", D, T, npts_tuple)
    println(io)

    print_mesh_summary(pp, npts_tuple, topodim, collapsed)

    print_mesh_domain_info(pp, set(Ωₕ))

    if !collapsed
        uniform_tuple = ntuple(i -> is_uniform(Ωₕ(i)), Val(D))
        print_mesh_spacing_info(pp, uniform_tuple, hₘₐₓ(Ωₕ))
    end

    print_mesh_markers(pp, markers(Ωₕ))
    return nothing
end
