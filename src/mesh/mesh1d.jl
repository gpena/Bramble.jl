"""
    Mesh1D{BT, CI, VT, T} <: AbstractMeshType{1}

One-dimensional grid discretizing a 1D [`CartesianProduct`](@ref) interval.

Stores grid point coordinates `pts`, underlying geometric interval `set`, semantic markers `markers`,
Cartesian indices `indices`, and computational backend `backend`. Also precomputes and caches cell centers
(`half_pts`), cell measures (`half_spacings`), and backward spacings (`spacings`).

# Fields

  - `set`: 1D geometric [`CartesianProduct`](@ref) interval over which the mesh is defined.
  - `markers`: [`MeshMarkers`](@ref) dictionary mapping symbols to `BitVector` indicators.
  - `indices`: `CartesianIndices{1}` of the grid points.
  - `backend`: Linear algebra [`Backend`](@ref) for memory management and operations.
  - `pts`: Coordinate vector storing grid points ``x_i`` for ``i = 1, \\dots, N``.
  - `half_pts`: Precomputed cell centers (midpoints) ``x_{i+1/2}`` for ``i = 1, \\dots, N+1``.
  - `half_spacings`: Precomputed cell widths (control volume measures) ``h_{i+1/2}`` for ``i = 1, \\dots, N``.
  - `spacings`: Precomputed backward grid spacings ``h_i = x_i - x_{i-1}``, with ``h_1 = x_2 - x_1``.
  - `collapsed`: Boolean flag indicating whether the interval is degenerate (a single point).
  - `version`: Monotone counter bumped by every in-place point mutation (gpena/Bramble.jl#221);
    see [`set_points!`](@ref) and [`_mesh_version`](@ref).

See also: [`MeshnD`](@ref), [`mesh`](@ref), [`AbstractMeshType`](@ref).
"""
mutable struct Mesh1D{BT <: Backend, CI <: CartesianIndices{1}, VT <: AbstractVector, T} <:
               AbstractMeshType{1}
    "the geometric domain, a 1D CartesianProduct (interval), over which the mesh is defined."
    set::CartesianProduct{1, T}
    "a dictionary mapping `Symbol` labels to `BitVector`s, marking specific points on the mesh."
    markers::MeshMarkers
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
    "monotone counter bumped by every in-place point mutation; see `_mesh_version`."
    version::Int
end

@noinline _throw_point_count_mismatch(expected::Int, got::Int) = throw(
    DimensionMismatch(
    "change_points! keeps the point count: the mesh has $expected points and $got were given",
),
)

@inline is_collapsed(Ωₕ::Mesh1D) = Ωₕ.collapsed

"""
    _mesh_version(Ωₕ::AbstractMeshType) -> Int

The monotone counter `set_points!` bumps every in-place point mutation
(gpena/Bramble.jl#221) -- what a [`ScalarGridSpace`](@ref)'s [`SpaceWeights`](@ref) records
at construction and every discrete inner product/norm re-checks against, to catch weights
computed from a mesh that has since been mutated underneath them.

A `Mesh1D` reads its own stored counter directly. A `MeshnD` has no coordinates of its own
-- only [`submeshes`](@ref), each independently mutated by `change_points!(Ωₕ::MeshnD, ...)`
-- so its version is the sum of theirs: strictly increasing whenever any one axis changes,
regardless of which, with no second counter of its own to keep in sync.
"""
@inline _mesh_version(Ωₕ::Mesh1D) = Ωₕ.version

@inline points(Ωₕ::Mesh1D) = Ωₕ.pts

# `point` is the choke point every mesh iteration (`Base.iterate`, `Ωₕ[i]`, and `MeshnD`'s
# per-submesh `point`) goes through one index at a time. A host-backed mesh reads straight
# off `points(Ωₕ)`; a device-backed one throws a named error instead of attempting `N`
# sequential scalar reads that its own scalar-indexing guard would refuse one at a time
# anyway (gpena/Bramble.jl#308) -- call [`host_points`](@ref) once and index or iterate that
# `Array` instead.
@inline function point(Ωₕ::Mesh1D, i)
    idx = _extract_linear_index(i)
    _check_point_bounds(Ωₕ, idx, "point")
    return _point(locality(typeof(points(Ωₕ))), Ωₕ, idx)
end

@inline _point(::HostLocality, Ωₕ::Mesh1D, idx) = @inbounds points(Ωₕ)[idx]
@noinline _point(::DeviceLocality, Ωₕ::Mesh1D, idx) = _throw_no_scalar_point()

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
@inline host_points(Ωₕ::Mesh1D) = _host_points(locality(typeof(points(Ωₕ))), Ωₕ)

@inline _host_points(::HostLocality, Ωₕ::Mesh1D) = points(Ωₕ)
@inline _host_points(::DeviceLocality, Ωₕ::Mesh1D) = Array(points(Ωₕ))

"""
    locate_cell(Ωₕ::Mesh1D, x::Real) -> Int

See [`locate_cell`](@ref)'s generic docstring (`mesh/queries.jl`) for the contract: the
largest node index `i` with `pts[i] <= x`, clamped to `1:n-1` -- exactly what
`searchsortedlast` on the point array answers, and what this method must keep answering,
uniform or not.

A uniform mesh answers with no array read at all, device-backed or not
(gpena/Bramble.jl#308): a first estimate `clamp(floor(Int, (x - a) / h) + 1, 1, n - 1)` on
the interval's own endpoint `a` and stepsize `h = (b - a) / (n - 1)`, matching how `_points!`
built the grid, then one correction step against the closed-form coordinates of that
estimate's own neighbouring nodes (`a + idx * h`, `a + (idx - 1) * h`) rather than the
estimate's raw division. `(x - a) / h` rounds to either side of an integer at a node
coordinate -- worse in `Float32` (Metal's only type, relative error ~1e-7) than in
`Float64` (~1e-16) -- so the first estimate can land one cell short or one cell over at an
exact grid point; the correction is what keeps this path agreeing with `searchsortedlast`
there; see gpena/Bramble.jl#308 (round 2) for the measured Float64 disagreement this fixes.

A non-uniform mesh has no formula to fall back on and searches [`host_points`](@ref)`(Ωₕ)`
instead of the raw, possibly device-resident `points(Ωₕ)`.
"""
function locate_cell(Ωₕ::Mesh1D, x::Real)
    n = npoints(Ωₕ)
    n <= 1 && return 1

    if is_uniform(Ωₕ)
        a, b = extrema(Ωₕ.set)
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

    pts = host_points(Ωₕ)
    if x <= pts[1]
        return 1
    elseif x >= pts[n]
        return n - 1
    end
    idx = searchsortedlast(pts, x)
    return clamp(idx, 1, n - 1)
end

@inline half_points(Ωₕ::Mesh1D) = Ωₕ.half_pts
@inline half_spacings(Ωₕ::Mesh1D) = Ωₕ.half_spacings

"""
    spacings(Ωₕ::Mesh1D) -> AbstractVector

Return the cached vector of backward spacings, where `spacings(Ωₕ)[i]` is
[`spacing`](@ref)`(Ωₕ, i)`. Recomputed by [`set_points!`](@ref) whenever the
grid points change.
"""
@inline spacings(Ωₕ::Mesh1D) = Ωₕ.spacings
@inline spacings!(Ωₕ::Mesh1D, v) = (Ωₕ.spacings = v; return nothing)

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
@inline host_spacings(Ωₕ::Mesh1D) = _host_spacings(locality(typeof(spacings(Ωₕ))), Ωₕ)

@inline _host_spacings(::HostLocality, Ωₕ::Mesh1D) = spacings(Ωₕ)
@inline _host_spacings(::DeviceLocality, Ωₕ::Mesh1D) = Array(spacings(Ωₕ))

"""
    host_half_spacings(Ωₕ::Mesh1D) -> Vector

Return [`half_spacings`](@ref)`(Ωₕ)` as a host-resident `Array`, in one bulk transfer
regardless of where `Ωₕ`'s storage lives, the same way [`host_spacings`](@ref) does for
[`spacings`](@ref) (gpena/Bramble.jl#307).

On a host-backed mesh, `half_spacings(Ωₕ)` already is an `Array`, so this returns it
directly with no copy; only a device-backed mesh pays the one bulk `Array(...)` transfer.

See also: [`host_spacings`](@ref), [`half_spacings`](@ref), [`half_spacing`](@ref).
"""
@inline host_half_spacings(Ωₕ::Mesh1D) = _host_half_spacings(locality(typeof(half_spacings(Ωₕ))), Ωₕ)

@inline _host_half_spacings(::HostLocality, Ωₕ::Mesh1D) = half_spacings(Ωₕ)
@inline _host_half_spacings(::DeviceLocality, Ωₕ::Mesh1D) = Array(half_spacings(Ωₕ))

"""
    forward_spacings(Ωₕ::Mesh1D) -> AbstractVector

Return the forward spacings of `Ωₕ`, where `forward_spacings(Ωₕ)[i]` is
[`forward_spacing`](@ref)`(Ωₕ, i)`. Unlike [`spacings`](@ref), this is not cached: it is
[`spacing`](@ref)'s vector read one index ahead, computed lazily on iteration.

See also: [`forward_spacing_for_derivative`](@ref).
"""
@inline forward_spacings(Ωₕ::Mesh1D) = _spacing_generator(Ωₕ, forward_spacing)

# A single-point mesh (n == 1, whether from a topologically collapsed domain or simply a
# one-point request) has no adjacent interval, so `half_spacings` is the honest raw zero
# there -- that raw value stays untouched, since other code (collapse detection, among it)
# reads it as exactly that. `cell_measures` is a *measure*, though, and the same `_apply_hs_logic`
# coercion `half_spacing(::MeshnD, idx)` already applies is needed here too, or a mesh with
# a collapsed axis silently gets a zero weight everywhere (gpena/Bramble.jl#89): the zero
# case is only ever the single-element one, so this stays the same zero-copy array in
# every other case and only allocates on that one rare, one-element path.
@inline function cell_measures(Ωₕ::Mesh1D)
    hs = half_spacings(Ωₕ)
    return length(hs) == 1 ? [_apply_hs_logic(hs[1])] : hs
end

"""
    set_points!(Ωₕ::Mesh1D, pts::AbstractVector) -> Nothing

Override the grid coordinates in `Ωₕ`. Recalculates cached [`spacings`](@ref),
[`half_points`](@ref), and [`half_spacings`](@ref).

Bumps `Ωₕ`'s mesh version (gpena/Bramble.jl#221): every [`ScalarGridSpace`](@ref) already
built on `Ωₕ` -- via [`gridspace`](@ref), directly or as a leaf of a
[`CompositeGridSpace`](@ref) -- keeps its own weights, precomputed from the mesh *before*
this call. Its `innerₕ`/`inner₊*`/norms now throw naming the mismatch instead of silently
computing against stale weights; call `gridspace(Ωₕ)` again for a space that reads the
mutated mesh. This is also what [`change_points!`](@ref) and [`iterative_refinement!`](@ref)
go through, so the same applies to both.
"""
@inline function set_points!(Ωₕ::Mesh1D, pts)
    Ωₕ.version += 1
    n = length(pts)

    if length(Ωₕ.pts) == n
        Ωₕ.pts .= pts
    else
        Ωₕ.pts = pts
        set_indices!(Ωₕ, generate_indices(n))
        half_points!(Ωₕ, vector(backend(Ωₕ), n + 1))
        half_spacings!(Ωₕ, vector(backend(Ωₕ), n))
        spacings!(Ωₕ, vector(backend(Ωₕ), n))
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

    return nothing
end

"""
    half_points!(Ωₕ::Mesh1D, pts::AbstractVector) -> Nothing

Override the precomputed cell center cache in `Ωₕ`.
"""
@inline half_points!(Ωₕ::Mesh1D, pts) = (Ωₕ.half_pts = pts; return nothing)

"""
    half_spacings!(Ωₕ::Mesh1D, pts::AbstractVector) -> Nothing

Override the precomputed cell width cache in `Ωₕ`.
"""
@inline half_spacings!(Ωₕ::Mesh1D, pts) = (Ωₕ.half_spacings = pts; return nothing)

@inline eltype(::Mesh1D{BT}) where {BT} = eltype(BT)
@inline eltype(::Type{<:Mesh1D{BT}}) where {BT} = eltype(BT)

"""
    (Ωₕ::Mesh1D)(i::Integer) -> Mesh1D

Return the `i`-th submesh of `Ωₕ`. A 1D mesh is its own only submesh, returning
`Ωₕ` itself; provided for uniform indexing in multi-dimensional generic algorithms.
"""
@inline (Ωₕ::Mesh1D)(::Integer) = Ωₕ

@inline npoints(Ωₕ::Mesh1D) = length(points(Ωₕ))
@inline npoints(Ωₕ::Mesh1D, ::Type{Tuple}) = (npoints(Ωₕ),)

@inline hₘₐₓ(Ωₕ::Mesh1D) = maximum(spacings(Ωₕ))
@inline hₘᵢₙ(Ωₕ::Mesh1D) = minimum(spacings(Ωₕ))

# On a device-backed mesh, indexing `spacings(Ωₕ)` here scalar-indexes a device array and
# is refused by that array type's own scalar-indexing guard (gpena/Bramble.jl#94) -- left as
# the raw upstream error deliberately, the same one every other GPU array in the ecosystem
# raises for the same mistake. Catching it here to redirect to `host_spacings` would put a
# locality check in the single most-called accessor in this file for a message this array
# type already gives; a caller that hits it once should stop calling this per point and call
# `host_spacings(Ωₕ)` once instead, as `stencil.jl` now does.
@inline function spacing(Ωₕ::Mesh1D, i::Int)
    _check_point_bounds(Ωₕ, i, "spacing")
    return @inbounds spacings(Ωₕ)[i]
end

@inline spacing(Ωₕ::Mesh1D, i::CartesianIndex{1}) = spacing(Ωₕ, _extract_linear_index(i))

# D == 1, so dim is always 1: a plain passthrough matching the MeshnD 3-arg accessor
# (gpena/Bramble.jl#111), so mesh-generic callers can use one signature regardless of D.
@inline spacing(Ωₕ::Mesh1D, i, dim::Int) = spacing(Ωₕ, i)
"""
    spacing_for_derivative(Ωₕ::Mesh1D, idx) -> eltype(Ωₕ)

Return the spacing that a backward finite difference divides by at `idx`, which is
[`spacing`](@ref)`(Ωₕ, idx)` everywhere except the first point, where the difference has
no stencil and this is zero.

See also: [`forward_spacing_for_derivative`](@ref), [`spacings`](@ref).
"""
@inline function spacing_for_derivative(Ωₕ::Mesh1D, idx)
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
`i > 1`. Entry 1 is zero: the backward difference has no stencil at the first point.
"""
@inline backward_spacings_for_derivative(Ωₕ::Mesh1D) = spacings(Ωₕ)

"""
    forward_spacings_for_derivative(Ωₕ::Mesh1D) -> AbstractVector

Return a vector `h` with `h[i] == `[`forward_spacing_for_derivative`](@ref)`(Ωₕ, i)` for
every `i < npoints(Ωₕ)`, as a view onto cached spacings. The last entry is omitted because
the forward difference has no stencil at the right boundary.
"""
@inline function forward_spacings_for_derivative(Ωₕ::Mesh1D)
    h = spacings(Ωₕ)
    return @inbounds @view h[min(2, length(h)):end]
end

# Same device-mesh tradeoff as `spacing` above, and the same choice: this scalar-indexes
# `spacings(Ωₕ)` and is left to throw the array type's own raw scalar-indexing error rather
# than a redirect added here. Use `host_spacings(Ωₕ)` for a per-point loop instead.
@inline function forward_spacing(Ωₕ::Mesh1D, i::Int)
    _check_point_bounds(Ωₕ, i, "forward_spacing")
    # forward_spacing(i) is spacing(i + 1) away from the last point, and repeats the
    # final interval at it, which is exactly what the cached vector already holds.
    n = npoints(Ωₕ)
    return @inbounds spacings(Ωₕ)[i == n ? n : i + 1]
end

@inline forward_spacing(Ωₕ::Mesh1D, i::CartesianIndex{1}) = forward_spacing(Ωₕ, _extract_linear_index(i))

@inline forward_spacing(Ωₕ::Mesh1D, i, dim::Int) = forward_spacing(Ωₕ, i)
"""
    forward_spacing_for_derivative(Ωₕ::Mesh1D, idx) -> eltype(Ωₕ)

Return the spacing that a forward finite difference divides by at `idx`, which is
[`forward_spacing`](@ref)`(Ωₕ, idx)` everywhere except the last point, where the
difference has no stencil and this is zero.

See also: [`spacing_for_derivative`](@ref), [`spacings`](@ref).
"""
@inline function forward_spacing_for_derivative(Ωₕ::Mesh1D, idx)
    i = idx isa CartesianIndex{1} ? _extract_linear_index(idx) : idx

    if i == npoints(Ωₕ)
        zero(eltype(Ωₕ))
    else
        forward_spacing(Ωₕ, i)
    end
end

@inline function half_point(Ωₕ::Mesh1D, i::Int)
    _check_half_point_bounds(Ωₕ, i)
    return Ωₕ.half_pts[i]
end

@inline function half_spacing(Ωₕ::Mesh1D, i::Int)
    _check_point_bounds(Ωₕ, i, "half_spacing")
    return Ωₕ.half_spacings[i]
end

@inline half_spacing(Ωₕ::Mesh1D, idx::CartesianIndex{1}) = half_spacing(Ωₕ, _extract_linear_index(idx))

@inline function cell_measure(Ωₕ::Mesh1D, i)
    idx = _extract_linear_index(i)
    _check_point_bounds(Ωₕ, idx, "cell_measure")
    return _apply_hs_logic(half_spacing(Ωₕ, idx))
end

@inline function _generate_random_points!(v)
    rand!(v)
    sort!(v)  # In-place sort
    return nothing
end

#------------------------------------------------------------------------------------------#
# Device kernel launch stubs (gpena/Bramble.jl#94, #174, S2.1 of
# .agents/plans/metal-and-apple-silicon-acceleration.md)
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
point fill, [`_launch_spacing!`](@ref), [`_launch_half_points!`](@ref)
and [`_launch_half_spacing!`](@ref) into one kernel for a uniform device mesh
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
spacing(i) / 2` at `i = 1` or `i = n`, and the telescoped interior form `(pts[i + 1] -
pts[i - 1]) / 2` elsewhere -- the fusion of [`_launch_spacing!`](@ref),
[`_launch_half_points!`](@ref) and [`_launch_half_spacing!`](@ref) into one kernel for a
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
    end
    return nothing
end

# Device counterpart of the method above, reached only for the two cases `_mesh` does not
# route through the fused `_uniform_mesh1d_init!` kernel (gpena/Bramble.jl#303): a
# single-point mesh, and a non-uniform one. `unif` and `backend` are carried to keep this
# method's signature distinct from the `Array` one above -- with `unif` dropped, a
# three-argument device method would be ambiguous with it. Past the single-point guard
# `unif` is always `false`, so there is no uniform branch here to take. The non-uniform fill
# has no device RNG/sort to launch, so it generates the coordinates on the host with the
# same routine the `Array` method above uses, into a scratch `Vector`, and transfers them to
# `x` in one `copyto!` (gpena/Bramble.jl#304) -- a one-time O(n) construction cost, not a
# per-iteration one.
function _points!(x::AbstractVector, I::CartesianProduct{1}, unif::Bool, backend)
    npts = length(x)
    T = eltype(I)
    a, b = extrema(I)

    if npts == 1
        x .= a
        return nothing
    end

    cpu_pts = Vector{T}(undef, npts)
    cpu_pts[1] = zero(T)
    cpu_pts[npts] = one(T)

    v = view(cpu_pts, 2:(npts - 1))
    _generate_random_points!(v)

    @. cpu_pts = a + cpu_pts * (b - a)

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
@inline function spacing!(x::Array, Ωₕ::Mesh1D)
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
function spacing!(x::AbstractVector, Ωₕ::Mesh1D)
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
@inline function _nonuniform_mesh1d_metrics!(Ωₕ::Mesh1D)
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

    # Instantiate the Mesh1D struct with initial (empty) markers.
    mesh_markers = MeshMarkers()
    mesh = Mesh1D(
        set,
        mesh_markers,
        idxs,
        backend,
        pts,
        _half_pts,
        _half_spacings,
        _spacings,
        is_collapsed,
        0
    )

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
    set_points!(Ωₕ, new_points)
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
(`pts`, `half_pts`, `half_spacings`, `markers`).
"""
function Base.copy(Ωₕ::Mesh1D)
    return Mesh1D(
        Ωₕ.set,
        deepcopy(Ωₕ.markers),
        Ωₕ.indices,
        Ωₕ.backend,
        copy(Ωₕ.pts),
        copy(Ωₕ.half_pts),
        copy(Ωₕ.half_spacings),
        copy(Ωₕ.spacings),
        Ωₕ.collapsed,
        Ωₕ.version
    )
end

@inline Base.getindex(Ωₕ::Mesh1D, i::Int) = point(Ωₕ, i)
@inline Base.getindex(Ωₕ::Mesh1D, i::CartesianIndex{1}) = point(Ωₕ, i)

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
