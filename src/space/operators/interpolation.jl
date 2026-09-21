#===========================================================================#
# Interpolation between two grid functions built over different meshes.
#
# Enables transfers of grid functions between distinct leaf meshes within
# composite spaces. The numeric operators (`πₕ` / `πₕ!`) live here, alongside
# the other operators over `VectorElement`, following the `Xₕ` / `Xₕ!` in-place
# convention used by `Rₕ` / `Rₕ!` and `avgₕ` / `avgₕ!`. The symbolic AST wrapper
# `πₕ` (taking one argument) lives in `form/operators/interpolation.jl`,
# distinguished by dispatch arity.
#===========================================================================#

#===========================================================================#
# Out-of-domain policy (gpena/Bramble.jl#223)
#
# `locate_cell` clamps which *cell* an outside point is read against to the boundary
# cell, but never clamped the fraction `x` is weighted by inside it -- so a point beyond
# the boundary used to interpolate as a silent linear extrapolation along that cell's own
# slope, with no error, no warning, and blend weights that still summed to one (nothing
# downstream could tell). `outside` names that choice explicitly instead of making it by
# accident:
#
#   :error        -- (the default) throw, naming the point and the domain extent
#   :clamp        -- clamp the point to the domain, then interpolate there
#   :extrapolate  -- today's original behaviour, now opt-in
#   a Number      -- return this value outright (e.g. 0.0, NaN), bypassing the blend
#
# A point outside by no more than a handful of `eps`, relative to the domain extent, is
# treated as exactly on the boundary under *every* policy, `:error` included: this is what
# keeps `πₕ` between two meshes over the same nominal domain from tripping over floating-
# point endpoint noise (gpena/Bramble.jl#223's own acceptance criterion).
#
# The fill-value policy is pointwise-only, by mathematical necessity, not by omission:
# `interpolation_matrix`/`InterpolationNode` (form/operators/interpolation.jl) represent
# interpolation as a *linear* map, `P * parent(src)`, and a row that returns a constant
# regardless of `src` cannot be written as a weighted combination of `src`'s own entries
# unless that constant is exactly zero. Those two paths accept only the three symbols and
# reject a Number outright, with a message saying why; `interpolate_at`/`πₕ`/`πₕ!` and the
# one-argument symbolic `πₕ(uₕ)` (a source, not a matrix) accept all four.
#===========================================================================#

# 8, not 1: measured against the actual boundary mismatch two independently constructed
# meshes over the same nominal domain can carry (a handful of ULPs from unrelated rounding
# in each mesh's own point generation), with margin -- a tolerance that just barely covers
# today's worst case is one rounding-mode change away from failing again.
@inline function _interp_tol(lo, hi)
    T = promote_type(typeof(lo), typeof(hi))
    return 8 * eps(T) * max(one(T), abs(hi - lo))
end

@inline function _interp_near_boundary(lo, hi, x)
    tol = _interp_tol(lo, hi)
    return (lo - tol) <= x <= (hi + tol)
end

@noinline function _throw_outside_domain(x, lo, hi)
    throw(
        ArgumentError(
        "interpolation point $x lies outside the domain [$lo, $hi] (outside = :error, " *
        "the default). Pass outside = :clamp, outside = :extrapolate, or an explicit " *
        "fill value (e.g. outside = 0.0 or outside = NaN) to choose what happens there " *
        "instead.",
    ),
    )
end

@noinline function _throw_invalid_outside(outside)
    throw(
        ArgumentError(
        "outside = $(repr(outside)) is not recognised: expected :error, :clamp, " *
        ":extrapolate, or a Number (the fill value returned for an out-of-domain point).",
    ),
    )
end

@noinline function _throw_outside_domain_linear(outside)
    throw(
        ArgumentError(
        "outside = $(repr(outside)) cannot be represented here: interpolation_matrix and " *
        "the symbolic bilinear πₕ(u) are linear maps, P * parent(src), and a row " *
        "that returns a constant regardless of src cannot be written as a weighted " *
        "combination of src's own entries unless that constant is zero. Use :error, " *
        ":clamp, or :extrapolate here; a fill value is only meaningful for the pointwise " *
        "interpolate_at/πₕ/πₕ! and the symbolic source πₕ(uₕ).",
    ),
    )
end

# Validated once, at each public entry point, rather than resolved lazily on first use:
# a bad `outside` should fail before any work is done, not after `_interp_cell_frac` has
# already walked partway through a multi-dimensional point.
@inline _validate_outside(outside::Symbol) = outside in (:error, :clamp, :extrapolate) ||
                                             _throw_invalid_outside(outside)
@inline _validate_outside(::Number) = nothing
@inline _validate_outside(outside) = _throw_invalid_outside(outside)

@inline function _validate_outside_linear(outside::Symbol)
    outside in (:error, :clamp, :extrapolate) || _throw_invalid_outside(outside)
    return nothing
end
@inline _validate_outside_linear(outside) = _throw_outside_domain_linear(outside)

# One coordinate against one axis's domain `[lo, hi]`, already known not to need the
# fill-value short-circuit (that is decided one layer up, in `_interp_cell_frac`, before
# any axis is visited, so this only ever sees the three symbols). A near-boundary point is
# snapped exactly onto the boundary under every policy, `:error` included.
@inline function _interp_resolve_coord(xd, lo, hi, outside::Symbol)
    _interp_near_boundary(lo, hi, xd) && return clamp(xd, lo, hi)
    outside === :clamp && return clamp(xd, lo, hi)
    outside === :extrapolate && return xd
    return _throw_outside_domain(xd, lo, hi)
end

@inline function _interp_in_domain(Ωₕ::AbstractMeshType{1}, x)
    pts = points(Ωₕ)
    return _interp_near_boundary(pts[1], pts[end], x)
end

@inline function _interp_in_domain(Ωₕ::AbstractMeshType{D}, x) where {D}
    return all(ntuple(Val(D)) do d
        pts = points(Ωₕ(d))
        _interp_near_boundary(pts[1], pts[end], x[d])
    end)
end

"""
    interpolate_at(uₕ::VectorElement, x; outside = :error)

The piecewise (multi)linear interpolant of `uₕ` at the physical point `x`, using `uₕ`'s
own mesh.

Locates the cell of `mesh(space(uₕ))` containing `x` ([`locate_cell`](@ref)) and blends the
``2^D`` grid values at that cell's corners, weighted by `x`'s relative position within it
(the standard bilinear/trilinear construction), exact for any affine function of the
coordinates and correct on a non-uniform mesh, since it reads the mesh's own point
coordinates rather than assuming a fixed step.

`outside` (gpena/Bramble.jl#223) names what happens when `x` falls outside the domain:

  - `:error` (the default): throw an `ArgumentError` naming `x` and the domain extent.
  - `:clamp`: clamp `x` to the domain first, then interpolate there.
  - `:extrapolate`: the original behaviour -- blend using the boundary cell's own corner
    weights and slope, unclamped, which is a linear extrapolation rather than a hold.
  - a `Number` (e.g. `0.0`, `NaN`): return this value outright, bypassing the blend.

A point outside by no more than a few `eps`, relative to the domain extent, is always
treated as exactly on the boundary, regardless of `outside` -- this is what keeps `πₕ`
between two meshes over the same nominal domain from tripping over floating-point endpoint
noise at the rim.

This is the building block both `πₕ!`/`πₕ` (below, the numeric operator) and the
one-argument, symbolic `πₕ` use: `x -> interpolate_at(uₕ, x)` is itself a valid source
function, usable anywhere one is accepted, including directly as [`Rₕ`](@ref)'s own argument:
`πₕ(Wₕ, src)` is `Rₕ` applied to this one function, not a separate mechanism. `Rₕ(Wₕ, f)`
restricts an arbitrary continuous `f`; when `f` happens to be another grid function's own
interpolant, restricting it is interpolating it, which is why `πₕ` generalises `Rₕ` for the
case the source is discrete rather than a closed-form function.

See also [`interpolation_matrix`](@ref), whose `outside` accepts only `:error`, `:clamp`
and `:extrapolate` -- a fill value cannot be represented as a linear map (see the note atop
this file).
"""
function interpolate_at(uₕ::VectorElement{<:ScalarGridSpace{1}}, x; outside = :error)
    _validate_outside(outside)
    Ωₕ = mesh(space(uₕ))
    frac = _interp_cell_frac(Ωₕ, x, outside)
    frac === nothing && return outside
    i, t = frac
    return (1 - t) * uₕ[i] + t * uₕ[i + 1]
end

function interpolate_at(uₕ::VectorElement{<:ScalarGridSpace{D}}, x; outside = :error) where {D}
    _validate_outside(outside)
    Ωₕ = mesh(space(uₕ))
    frac = _interp_cell_frac(Ωₕ, x, outside)
    frac === nothing && return outside
    idx, ts = frac
    li = LinearIndices(indices(Ωₕ))

    acc = zero(promote_type(eltype(uₕ), typeof(first(ts))))
    for corner in CartesianIndices(ntuple(_ -> 0:1, Val(D)))
        acc += _interp_corner_weight(ts, corner, Val(D)) * uₕ[li[idx + corner]]
    end
    return acc
end

# --- The corner blend, in one place ------------------------------------------------- #
#
# Which cell of `Ωₕ` holds `x`, and where inside it, per direction. Three callers want
# exactly this and nothing more: `interpolate_at` above blends grid *values* with the
# weights, `interpolation_matrix` emits them as matrix entries, and the symbolic
# `InterpolationNode` (form/operators/interpolation.jl) emits them as stencil entries against
# absolute trial columns. Factored so the three cannot drift.
#
# Returns `(i, t)`/`(idx, ts)` as before, or the sentinel `nothing` when `outside isa
# Number` *and* `x` is genuinely outside the domain -- the one shape change from #223,
# read by `interpolate_at` alone (the only caller a fill value is meaningful for; the
# linear callers validate `outside` down to the three symbols before ever reaching here,
# so `nothing` never arises on those paths).
@inline function _interp_cell_frac(Ωₕ::AbstractMeshType{1}, x, outside)
    pts = points(Ωₕ)
    lo, hi = pts[1], pts[end]
    if outside isa Number && !_interp_near_boundary(lo, hi, x)
        return nothing
    end
    xc = _interp_resolve_coord(x, lo, hi, outside isa Number ? :extrapolate : outside)
    i = locate_cell(Ωₕ, xc)
    plo, phi = pts[i], pts[i + 1]
    t = phi > plo ? (xc - plo) / (phi - plo) : zero(xc - plo)
    return i, t
end

@inline function _interp_cell_frac(Ωₕ::AbstractMeshType{D}, x, outside) where {D}
    if outside isa Number && !_interp_in_domain(Ωₕ, x)
        return nothing
    end
    # Every axis is now known to be resolvable without the fill-value short-circuit --
    # either genuinely in bounds, or `outside` is one of the three symbols -- so each
    # per-axis resolution reads a concrete `Symbol`, keeping `ntuple` below uniformly
    # typed (a `Number`/`nothing` mix would not be).
    outside_sym = outside isa Number ? :extrapolate : outside
    xc = ntuple(Val(D)) do d
        pts = points(Ωₕ(d))
        _interp_resolve_coord(x[d], pts[1], pts[end], outside_sym)
    end
    idx = locate_cell(Ωₕ, xc)
    ts = ntuple(Val(D)) do d
        pts = points(Ωₕ(d))
        i = idx[d]
        lo, hi = pts[i], pts[i + 1]
        hi > lo ? (xc[d] - lo) / (hi - lo) : zero(xc[d] - lo)
    end
    return idx, ts
end

# The multilinear weight of one corner: `tᵈ` where the corner is on the far side of
# direction `d`, `1 - tᵈ` where it is on the near side. Over all `2ᴰ` corners these sum to
# one, which is what keeps the interpolant from overshooting.
@inline function _interp_corner_weight(ts::NTuple{D}, corner, ::Val{D}) where {D}
    w = one(eltype(ts))
    for d in 1:D
        w *= corner[d] == 1 ? ts[d] : (1 - ts[d])
    end
    return w
end

"""
    πₕ!(dest::VectorElement, src::VectorElement; outside = :error) -> VectorElement

Fills `dest` with the piecewise (multi)linear interpolant of `src`, sampled at `dest`'s own
mesh points, providing the in-place numeric interpolation operator named after the [`Rₕ!`](@ref)/
[`avgₕ!`](@ref) convention.

`interpolate_at(src, ·)` is a genuine function of a physical point (evaluable anywhere,
not only at `src`'s own grid points), so this is equivalent to `Rₕ!(dest, x -> interpolate_at(src, x; outside))`.
`dest` and `src` may be built over entirely different meshes. `Rₕ!` handles evaluating the interpolant
at each of `dest`'s grid points, following `dest`'s backend [`execution_policy`](@ref).
`outside` is forwarded to [`interpolate_at`](@ref) unchanged; see its docstring for the
policy this decides among (gpena/Bramble.jl#223).

Every call here re-locates, via [`locate_cell`](@ref), which cell of `src`'s mesh each of
`dest`'s points falls in. Interpolating repeatedly between the same two meshes (a time loop
transferring a coefficient between two composite leaves, say) should build that once instead:
see the [`interpolation_matrix`](@ref)-based method below, following the same "build the
pattern once" shape [`allocate_system_matrix`](@ref)/[`assemble!`](@ref) already use.

When `dest` and `src` are both host-backed, this evaluates `interpolate_at` pointwise as
described above. When either is device-backed (gpena/Bramble.jl#312), the pointwise path
would compile a device kernel around the host closure `x -> interpolate_at(src, x)` --
which cannot compile, since `interpolate_at` dispatches, calls `locate_cell` and reads
`src`'s own coefficients on the host. Instead, this assembles
`interpolation_matrix(space(dest), space(src); outside)` once and runs it through the
`mul!`-based method below, which uploads `P` and `src` next to `dest` as needed -- so a
fill value (`outside::Number`, meaningless as a linear map) throws the same
`ArgumentError` [`interpolation_matrix`](@ref) itself throws, naming that function.
"""
@inline function πₕ!(dest::VectorElement, src::VectorElement; outside = :error)
    return _πₕ!(locality(typeof(parent(dest))), locality(typeof(parent(src))), dest, src, outside)
end

@inline _πₕ!(::HostLocality, ::HostLocality, dest, src, outside) = Rₕ!(
    dest, x -> interpolate_at(src, x; outside)
)

# At least one of dest/src is device-backed: assemble the matrix once and reuse the
# mul!-based method below rather than launching a host closure inside a device kernel.
function _πₕ!(dest_loc, src_loc, dest, src, outside)
    _validate_outside_linear(outside)
    P = interpolation_matrix(space(dest), space(src); outside)
    return πₕ!(dest, P, src)
end

"""
    πₕ!(dest::VectorElement, P::AbstractMatrix, src::VectorElement) -> VectorElement

Fills `dest` via a precomputed [`interpolation_matrix`](@ref) `P` instead of re-locating
each destination point's cell: `parent(dest) .= P * parent(src)`, computed in place via
`mul!`.

`P` must be `interpolation_matrix(space(dest), space(src))` (or an equal-shape matrix
built the same way) -- a mismatched size throws the usual `DimensionMismatch` from `mul!`.
`interpolation_matrix` always returns a host `SparseMatrixCSC`, regardless of `dest`'s own
backend (gpena/Bramble.jl#312): passing it straight to a device-backed `dest`/`src` would
otherwise fall through `SparseArrays`' own generic sparse-times-vector method, which
scalar-indexes the device vectors one entry at a time. So `P` and `src` are each brought
to `dest`'s own locality first -- `P` uploaded with [`metal_sparse_csr`](@ref) if `dest` is
device-backed and `P` is not already, `src` copied wholesale (never scalar-read) if its
locality disagrees with `dest`'s -- and only then does `mul!` run, zero-allocation once
`P` already lives next to `dest` and `src` (the loop in [`πₕ!`](@ref)'s own docstring
above).

Repeated interpolation between the same two meshes should build `P` once and reuse it here
every subsequent call, exactly as `allocate_system_matrix`/`assemble!` split the sparsity
pattern (expensive, built once) from refilling values (cheap, every step):

```julia
P = interpolation_matrix(space(dest), space(src))
for step in 1:nsteps
    Rₕ!(src, coefficient_at(step))
    πₕ!(dest, P, src)   # no locate_cell search, zero allocations
end
```
"""
@inline function πₕ!(dest::VectorElement, P::AbstractMatrix, src::VectorElement)
    dest_loc = locality(typeof(parent(dest)))
    Pm = _πₕ_matrix(dest_loc, P)
    xs = _πₕ_vector(dest_loc, parent(dest), parent(src))
    mul!(parent(dest), Pm, xs)
    return dest
end

# `P` already lives at `dest`'s locality: nothing to move.
@inline _πₕ_matrix(dest_loc, P) = _πₕ_matrix_at(dest_loc, locality(typeof(P)), P)
@inline _πₕ_matrix_at(loc, ::T, P) where {T} = P

# `dest` is device-backed and `P` is the host SparseMatrixCSC interpolation_matrix
# returns: upload it once (gpena/Bramble.jl#250, #313), a bulk sparse-to-sparse transfer,
# never a per-entry scalar write into device memory.
@inline _πₕ_matrix_at(::DeviceLocality, ::HostLocality, P) = metal_sparse_csr(P)

@inline _πₕ_vector(dest_loc, dest_parent, v) = _πₕ_vector_at(dest_loc, locality(typeof(v)), dest_parent, v)
@inline _πₕ_vector_at(loc, ::T, dest_parent, v) where {T} = v

# `src`'s locality disagrees with `dest`'s: move it in one bulk transfer (never a scalar
# read/write) rather than have `mul!` fail trying to read across localities itself.
@inline _πₕ_vector_at(::HostLocality, ::DeviceLocality, dest_parent, v) = Array(v)
@inline function _πₕ_vector_at(::DeviceLocality, ::HostLocality, dest_parent, v)
    xs = similar(dest_parent, length(v))
    copyto!(xs, v)
    return xs
end

"""
    πₕ(Wₕ::ScalarGridSpace, src::VectorElement; outside = :error) -> VectorElement

Evaluates `Rₕ(Wₕ, x -> interpolate_at(src, x; outside))`; see [`πₕ!`](@ref). Distinguished
by multiple dispatch from the one-argument symbolic wrapper `πₕ(uₕ)` in
`form/operators/interpolation.jl`. The element type is promoted from `Wₕ`'s and `src`'s
own, so interpolating a `Dual`-valued `src` yields a `Dual`-valued result on an
undifferentiated `Wₕ`. `outside` is forwarded to [`interpolate_at`](@ref) unchanged
(gpena/Bramble.jl#223).
"""
@inline πₕ(Wₕ::ScalarGridSpace, src::VectorElement; outside = :error) = Rₕ(
    Wₕ, x -> interpolate_at(src, x; outside)
)

# --- Triplet assembly shared by both dimensionalities of interpolation_matrix ---
#
# The same corner-weight arithmetic interpolate_at uses, emitting (row, col, weight)
# triplets instead of accumulating a value against one src's data, so the two are kept in
# step by construction. `outside` is already validated down to :error/:clamp/:extrapolate
# by `interpolation_matrix` before this runs (see this file's header note on why a fill
# value cannot be represented here), so the fraction helpers below never see the fill-value
# short-circuit `_interp_cell_frac` carries for `interpolate_at` alone.
#
# `point(Ωdest, ·)` and `_interp_cell_frac` both read straight off `points`/`points(Ωₕ(d))`,
# which throws outright on a device-backed mesh (gpena/Bramble.jl#308) rather than
# scalar-indexing it one point at a time. `host_points` (#308) is called exactly once per
# mesh below -- not once per destination point -- so the whole assembly is a host loop over
# host arrays regardless of where `Ωdest`/`Ωsrc` actually live; `locate_cell` (also #308) is
# the one per-point mesh query left, already safe on a device mesh by construction.

@inline function _interp_triplet_frac(Ωsrc::AbstractMeshType{1}, pts_src, x, outside::Symbol)
    lo, hi = pts_src[1], pts_src[end]
    xc = _interp_resolve_coord(x, lo, hi, outside)
    i = locate_cell(Ωsrc, xc)
    plo, phi = pts_src[i], pts_src[i + 1]
    t = phi > plo ? (xc - plo) / (phi - plo) : zero(xc - plo)
    return i, t
end

# The method above assumes `pts_src` is the flat vector `host_points(::Mesh1D)` actually
# returns; the D-dimensional method below assumes `pts_src::NTuple{D}`, which at D=1 is
# also a 1-tuple. Both match an `AbstractMeshType{1}` called with a 1-tuple `pts_src`
# (Aqua's ambiguity report), which is otherwise only a static possibility -- `mesh()` always
# builds `Mesh1D`, never a degenerate `MeshnD{1}`, for a 1D domain, so `host_points` never
# actually returns a 1-tuple here. Disambiguating with this method, rather than by
# restricting either of the two above, keeps both behaviours and simply unwraps the 1-tuple
# down to the same scalar arithmetic the flat-vector method uses, so the case behaves
# correctly if it is ever reached after all.
@inline function _interp_triplet_frac(Ωsrc::AbstractMeshType{1}, pts_src::Tuple{Any}, x, outside::Symbol)
    pts = pts_src[1]
    lo, hi = pts[1], pts[end]
    xc = _interp_resolve_coord(x, lo, hi, outside)
    i = locate_cell(Ωsrc, xc)
    plo, phi = pts[i], pts[i + 1]
    t = phi > plo ? (xc - plo) / (phi - plo) : zero(xc - plo)
    return i, t
end

@inline function _interp_triplet_frac(
        Ωsrc::AbstractMeshType{D}, pts_src::NTuple{D}, x, outside::Symbol
) where {D}
    xc = ntuple(Val(D)) do d
        pts = pts_src[d]
        _interp_resolve_coord(x[d], pts[1], pts[end], outside)
    end
    idx = locate_cell(Ωsrc, xc)
    ts = ntuple(Val(D)) do d
        pts = pts_src[d]
        i = idx[d]
        lo, hi = pts[i], pts[i + 1]
        hi > lo ? (xc[d] - lo) / (hi - lo) : zero(xc[d] - lo)
    end
    return idx, ts
end

function _interpolation_triplets!(
        rows, cols, vals, Ωdest::AbstractMeshType, Ωsrc::AbstractMeshType{1}, outside::Symbol
)
    li_dest = LinearIndices(indices(Ωdest))
    pts_dest = host_points(Ωdest)
    pts_src = host_points(Ωsrc)
    for i in indices(Ωdest)
        row = li_dest[i]
        x = pts_dest[row]
        j, t = _interp_triplet_frac(Ωsrc, pts_src, x, outside)
        push!(rows, row, row)
        push!(cols, j, j + 1)
        push!(vals, 1 - t, t)
    end
end

function _interpolation_triplets!(
        rows, cols, vals, Ωdest::AbstractMeshType, Ωsrc::AbstractMeshType{D}, outside::Symbol
) where {D}
    li_dest = LinearIndices(indices(Ωdest))
    li_src = LinearIndices(indices(Ωsrc))
    pts_dest = host_points(Ωdest)
    pts_src = host_points(Ωsrc)
    for I in indices(Ωdest)
        row = li_dest[I]
        x = ntuple(d -> pts_dest[d][I[d]], Val(D))
        idx, ts = _interp_triplet_frac(Ωsrc, pts_src, x, outside)

        for corner in CartesianIndices(ntuple(_ -> 0:1, Val(D)))
            push!(rows, row)
            push!(cols, li_src[idx + corner])
            push!(vals, _interp_corner_weight(ts, corner, Val(D)))
        end
    end
end

"""
    interpolation_matrix(Wdest::ScalarGridSpace, Wsrc::ScalarGridSpace; outside = :error) -> SparseMatrixCSC

The piecewise (multi)linear interpolant of [`πₕ`](@ref)/[`interpolate_at`](@ref) as
a sparse matrix `P` rather than applied pointwise:
`P * parent(src) ≈ parent(πₕ(Wdest, src))` for any `src::VectorElement` over
`Wsrc`. `P` is `ndofs(Wdest) × ndofs(Wsrc)`, generally rectangular (since `Wdest` and
`Wsrc` are built over different meshes), with at most ``2^D`` nonzero entries per row:
the corner weights of the source cell [`locate_cell`](@ref) places that destination point in.

`outside` (gpena/Bramble.jl#223) accepts only `:error` (the default), `:clamp`, and
`:extrapolate` -- not a fill value, unlike [`interpolate_at`](@ref): `P` is a linear map,
and a row that returns a constant regardless of `src` cannot be written as a weighted
combination of `src`'s own entries unless that constant is exactly zero. Passing a `Number`
throws, naming this. Agrees entry-for-entry with pointwise `interpolate_at` under each of
the three policies it does accept.

Unlike [`D₋ₓ`](@ref)`(Wₕ)` and the other operator matrices, this is always a
`SparseMatrixCSC`, regardless of either space's own backend `matrix_type`. Those matrices
are built from [`shift`](@ref) (a fixed diagonal offset generalized via Kronecker products),
but which source cell a destination point falls in has no such regular structure across two
independent meshes: it is genuinely sparse and irregular, assembled directly from `locate_cell`
rather than composed from a handful of shifts. Converting the result to another matrix type,
where that is meaningful, is left to the caller.
"""
function interpolation_matrix(
        Wdest::ScalarGridSpace{D}, Wsrc::ScalarGridSpace{D}; outside = :error
) where {D}
    _validate_outside_linear(outside)
    Ωdest, Ωsrc = mesh(Wdest), mesh(Wsrc)
    ndest, nsrc = ndofs(Wdest), ndofs(Wsrc)
    T = promote_type(eltype(Wdest), eltype(Wsrc))
    nnz_hint = ndest * 2^D

    rows = sizehint!(Int[], nnz_hint)
    cols = sizehint!(Int[], nnz_hint)
    vals = sizehint!(T[], nnz_hint)

    _interpolation_triplets!(rows, cols, vals, Ωdest, Ωsrc, outside)

    return sparse(rows, cols, vals, ndest, nsrc)
end
