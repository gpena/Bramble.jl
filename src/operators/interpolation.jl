# interpolation.jl
#
# The file holds both halves of the interpolation family: the numerical operators, which act on grid
# functions and build matrices, and then the AST nodes that the same names build when handed
# a `LazyOp`. It is included after the AST core (`ast/ast.jl`, `common.jl`,
# `expression.jl`, `operators/node_family.jl`), which the node half needs.

#===========================================================================#
# Interpolation between two grid functions built over different meshes.
#
# Enables transfers of grid functions between distinct leaf meshes within
# composite spaces. The numeric operators (`πₕ` / `πₕ!`) live here, alongside
# the other operators over `VectorElement`, following the `Xₕ` / `Xₕ!` in-place
# convention used by `Rₕ` / `Rₕ!` and `avgₕ` / `avgₕ!`. The symbolic AST wrapper
# `πₕ` (taking one argument) lives in `operators/interpolation.jl`,
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
# `interpolation_matrix`/`InterpolationNode` (operators/interpolation.jl) represent
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

# `interpolate_at` fetches the 2^D corner values around a single query point one scalar
# read at a time -- fine on the host, but a device transfer hidden inside a scalar call on
# a device-backed element, and one this package refuses rather than perform silently
# (gpena/Bramble.jl#336). Named after `_throw_no_scalar_point` (mesh/mesh1d.jl), guarding
# before any corner is read rather than mid-blend.
@noinline function _throw_no_scalar_interpolate_at()
    throw(
        ArgumentError(
        "interpolate_at scalar-indexes a device-backed element's corner values one at a " *
        "time, which its own scalar-indexing guard refuses. Use πₕ!/πₕ to interpolate " *
        "onto another grid space's points in bulk, or bring the element to the host first " *
        "with Array(parent(u)) and call interpolate_at on that.",
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
    locality(typeof(parent(uₕ))) isa DeviceLocality && _throw_no_scalar_interpolate_at()
    _validate_outside(outside)
    Ωₕ = mesh(space(uₕ))
    frac = _interp_cell_frac(Ωₕ, x, outside)
    frac === nothing && return outside
    i, t = frac
    return (1 - t) * uₕ[i] + t * uₕ[i + 1]
end

function interpolate_at(uₕ::VectorElement{<:ScalarGridSpace{D}}, x; outside = :error) where {D}
    locality(typeof(parent(uₕ))) isa DeviceLocality && _throw_no_scalar_interpolate_at()
    _validate_outside(outside)
    Ωₕ = mesh(space(uₕ))
    frac = _interp_cell_frac(Ωₕ, x, outside)
    frac === nothing && return outside
    idx, ts = frac
    li = LinearIndices(indices(Ωₕ))

    acc = zero(promote_type(eltype(uₕ), typeof(first(ts))))
    # No far corner along a collapsed axis: it has a single point.
    for corner in CartesianIndices(ntuple(d -> 0:min(1, size(li, d) - 1), Val(D)))
        acc += _interp_corner_weight(ts, corner, Val(D)) * uₕ[li[idx + corner]]
    end
    return acc
end

# --- The corner blend, in one place ------------------------------------------------- #
#
# Which cell of `Ωₕ` holds `x`, and where inside it, per direction. Three callers want
# exactly this and nothing more: `interpolate_at` above blends grid *values* with the
# weights, `interpolation_matrix` emits them as matrix entries, and the symbolic
# `InterpolationNode` (operators/interpolation.jl) emits them as stencil entries against
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
    # A collapsed axis (one point) has no cell: `t = 0` there, and every caller drops or
    # zero-weights its far corner, so the axis is a 1×1 identity factor.
    ts = ntuple(Val(D)) do d
        pts = points(Ωₕ(d))
        length(pts) == 1 && return zero(xc[d] - pts[1])
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
using Bramble: interpolation_matrix
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
`operators/interpolation.jl`. The element type is promoted from `Wₕ`'s and `src`'s
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
    # A collapsed axis (one point) has no cell to read `pts[i + 1]` from: `t = 0` there,
    # and `_interpolation_triplets!` emits only its near corner, so the axis contributes
    # a 1×1 identity factor to the per-axis Kronecker product.
    ts = ntuple(Val(D)) do d
        pts = pts_src[d]
        length(pts) == 1 && return zero(xc[d] - pts[1])
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
    # No far corner along a collapsed source axis: it has a single point.
    corners = CartesianIndices(ntuple(d -> 0:min(1, length(pts_src[d]) - 1), Val(D)))
    for I in indices(Ωdest)
        row = li_dest[I]
        x = ntuple(d -> pts_dest[d][I[d]], Val(D))
        idx, ts = _interp_triplet_frac(Ωsrc, pts_src, x, outside)

        for corner in corners
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

# ==============================================================================
# ==============================================================================
# The AST nodes: the symbolic interpolant
# ==============================================================================
# ==============================================================================

# interpolation.jl
# Symbolic counterpart of interpolate_at / πₕ! / πₕ (operators/interpolation.jl):
# wraps a grid function's interpolant as a SourceFunction, so it composes with the rest of
# the AST layer exactly the way any other source does. This is a second method of the same
# `πₕ` the numeric layer defines (dispatch distinguishes them by arity), not a separate name.

"""
    πₕ(uₕ::VectorElement; outside = :error) -> LazyOp

The interpolant of `uₕ`, as a symbolic source term; usable anywhere a source is, including
inside another operator: `innerₕ(D₋ₓ(πₕ(uₕ)), D₋ₓ(v))` differentiates the interpolated field
the same way `D₋ₓ` differentiates any other source, `innerₕ(Mₓ(πₕ(uₕ)), v)` averages it,
and so on. This enables a coupled form to evaluate a leaf's grid function on a different
leaf's mesh.

Built as `source_function(x -> interpolate_at(uₕ, x; outside), Val(D))`: a
`SourceFunction`'s own `local_stencil` evaluates its function at the current point of
whichever mesh is being walked, so `uₕ` can originate from another leaf without special
handling; the interpolation occurs once per point inside `interpolate_at`, where ordinary
source function calls occur. `outside` (gpena/Bramble.jl#223) is forwarded to
[`interpolate_at`](@ref) unchanged, fill values included -- this is a source (a function of
`uₕ`'s values), not a linear map, so unlike the operator [`πₕ`](@ref)`(op)` below it
carries no such restriction.
"""
function πₕ(uₕ::VectorElement{<:ScalarGridSpace{D}}; outside = :error) where {D}
    _validate_outside(outside)
    return source_function(x -> interpolate_at(uₕ, x; outside), Val(D))
end

#===========================================================================#
# The interpolation operator: πₕ over a trial function.
#
# The source wrapper above requires concrete nodal values, so it cannot take a trial function:
# there are no values to blend. What a bilinear form requires instead is the operator itself: for
# each point of the test mesh, the `2ᴰ` trial degrees of freedom of the cell containing it,
# along with the corner weights. That matches one row of `interpolation_matrix`, produced a row
# at a time during the assembly sweep rather than as a separate matrix, which keeps `assemble!`
# refilling at zero allocations.
#
# The entries name absolute trial columns, not relative offsets. Every other node's stencil says
# "this many points from here, on the mesh being walked"; an interpolation says "these dofs
# of the other mesh", determined by where the point falls via `locate_cell`.
# `AbsoluteColumn` marks this distinction so bilinear assembly routines resolve each kind
# by dispatch rather than by a runtime flag.
#===========================================================================#

"""
    InterpolationNode{D, S, OpType} <: LazyOp{D}

The symbolic interpolation operator: `inner_op` lives on `src_space`, and this node evaluates
it at points of whatever mesh the assembly is walking.

`Side` is [`TrialSide`](@ref) or [`TestSide`](@ref), taken from the leaf `πₕ` wrapped. It
decides which leaf `_bind_interp_spaces` stamps into `src_space`, whether the stencil names
[`AbsoluteColumn`](@ref)s or [`AbsoluteRow`](@ref)s, and which leaf's grid the sweep walks:
always the other one, the side that stays native (gpena/Bramble.jl#263).

`src_space` is `nothing` until assembly binds it. `πₕ(op)` cannot name the space itself: on a
composite space the leaf a term interpolates from is only resolved block by block, so
`_bind_interp_spaces` stamps that leaf in at that point, once per block, and the
`S === Nothing` node exists only between construction and that binding.

Distinct from the source wrapper `πₕ(uₕ)`, which carries a grid function's values. This node
carries no values; it carries the map, and its stencil names degrees of freedom of the space
it interpolates from.

`outside` (gpena/Bramble.jl#223) is one of `:error`, `:clamp` or `:extrapolate` -- never a
fill value, since this node's stencil is a *linear* map (weighted trial columns), and a
constant independent of the trial unknowns cannot be written that way; see
[`interpolation_matrix`](@ref)'s docstring for the same restriction.
"""
struct InterpolationNode{D, S, OpType <: LazyOp{D}, Side} <: LazyOp{D}
    src_space::S
    inner_op::OpType
    outside::Symbol
end

"""
    TrialSide
    TestSide

Which side of a bilinear term an [`InterpolationNode`](@ref) sits on.

A trial-side interpolation names absolute *columns* and leaves the rows to the walked mesh; a
test-side one names absolute *rows* and leaves the columns to it (gpena/Bramble.jl#263). The
side is a type parameter rather than a field so the node stays a singleton the like-term
simplifier can fold, and so every walk that reads it folds away at compile time.

The walked mesh is the native side's in both cases: it is the one carrying the quadrature
weight the product integrates against.
"""
struct TrialSide end

@doc (@doc TrialSide)
struct TestSide end

# Which leaf a node of each side binds to, and which slot type its stencil entries name.
@inline _interp_slot(::Type{TrialSide}) = AbsoluteColumn
@inline _interp_slot(::Type{TestSide}) = AbsoluteRow

# The side as a value, read off the type. Every side-dependent query goes through this rather
# than dispatching on the parameter directly: `UnaryWrapper` (form/stencil_eval.jl) names
# `InterpolationNode{D}`, and a method fixing a *later* parameter than `D` is neither more
# nor less specific than that union, which makes the pair ambiguous. One method on the
# unparameterized type is not, and the branch still folds away, since `Side` is in the type.
@inline _interp_side(::InterpolationNode{D, S, O, Side}) where {D, S, O, Side} = Side

@noinline function _throw_interp_inner(op)
    throw(
        ArgumentError(
        "πₕ as a bilinear operator wraps a trial or test function directly (`πₕ(u)`, " *
        "`πₕ(u(2))`, `πₕ(v)`), but received $(typeof(op)). An operator applied before the " *
        "interpolation (`πₕ(D₋ₓ(u))`, differencing on the source mesh and then " *
        "interpolating) is a different operator and is not implemented; write the operator " *
        "outside instead, `D₋ₓ(πₕ(u))`, which differences on the mesh being integrated over.",
    ),
    )
end

# The side a leaf puts the interpolation on. Both leaf kinds are accepted, plain or indexed
# (gpena/Bramble.jl#263); anything else is the operator-inside-interpolation refusal above.
@inline _interp_side_of(::Union{TrialFunction, IndexedTrialFunction}) = TrialSide
@inline _interp_side_of(::Union{TestFunction, IndexedTestFunction}) = TestSide

"""
    πₕ(op::LazyOp{D}; outside = :error) -> InterpolationNode

The interpolation operator onto whichever mesh the form integrates over, applied to the
trial or the test function `op`. The space interpolated *from* is that function's own, and is
not written here: it is bound during assembly, once the concrete leaf is known
(`_bind_interp_spaces`), which is also what makes this work on a composite space, where the
leaf a term names is only resolved block by block.

This is the bilinear counterpart of `πₕ(uₕ)`: that one interpolates a grid function whose
values are already known, and belongs on the source side of a linear form; this one
interpolates the unknown, and so contributes matrix columns. `innerₕ(πₕ(u), v)` assembles
`Hᵥ · P`, with `P` the same matrix `interpolation_matrix` builds: computed a row at a time
during the sweep rather than as a matrix product, which allows `assemble!` to refill it with
zero allocations.

Operators wrap it from the outside, acting on the mesh being integrated over:
`inner₊(D₋ₓ(πₕ(u)), D₋ₓ(v))` is `D_x^⊤ H_+ D_x P`. Writing an operator inside is a different
operation and is refused, since it would difference on the source mesh instead.

`innerₕ(u, πₕ(w))` is the mirror (gpena/Bramble.jl#263): the test side interpolates, so the
*trial* mesh is the one integrated over and the entries name absolute rows instead of absolute
columns. The assembled matrix is `Pᵀ · H · (trial factor)`. A single term interpolating both
sides is refused -- one side has to stay native, since it is the side whose mesh carries the
quadrature weight.

`op` must be a trial- or test-function leaf, plain or indexed. `outside`
(gpena/Bramble.jl#223) accepts only `:error` (the default), `:clamp` and `:extrapolate` -- see
[`InterpolationNode`](@ref)'s own docstring for why a fill value is refused here.
"""
function πₕ(op::LazyOp{D}; outside = :error) where {D}
    _is_interp_leaf(op) || _throw_interp_inner(op)
    _validate_outside_linear(outside)
    return InterpolationNode{D, Nothing, typeof(op), _interp_side_of(op)}(
        nothing, op, outside
    )
end

@inline _is_interp_leaf(op) = op isa TrialFunction || op isa IndexedTrialFunction ||
                              op isa TestFunction || op isa IndexedTestFunction

# --- The stencil: absolute trial columns, with the corner weights ------------------- #

@inline function local_stencil(
        op::InterpolationNode{D, S, OpType, Side}, space, I::CartesianIndex{D}, markers,
        lin_idx::Int
) where {D, S, OpType, Side}
    return _interp_stencil(
        mesh(op.src_space), point(mesh(space), I), Val(D), op.outside, _interp_slot(Side)
    )
end

# An unbound node reaching the sweep means an assembly path walked a term without calling
# `_bind_interp_spaces` first. Caught by dispatch rather than by a `nothing` check inside
# the bound method, so the common path carries no test, and loudly, since the alternative
# is a `MethodError` from `mesh(nothing)` several frames deeper.
@noinline function local_stencil(
        ::InterpolationNode{D, Nothing}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    throw(
        ArgumentError(
        "an interpolation node reached the assembly sweep without a source space. Every " *
        "path that walks a bilinear term over a block binds one first with " *
        "`_bind_interp_spaces(term, trial_leaf)`; this one did not.",
    ),
    )
end

# `Slot` is `AbsoluteColumn` on the trial side and `AbsoluteRow` on the test side: the
# blend itself is the same map either way, and only which half of the matrix position it
# names differs (gpena/Bramble.jl#263).
@inline function _interp_stencil(
        Ωsrc::AbstractMeshType{1}, x, ::Val{1}, outside::Symbol, Slot
)
    j, t = _interp_cell_frac(Ωsrc, x, outside)
    return ((Slot(j), 1 - t), (Slot(j + 1), t))
end

@inline function _interp_stencil(
        Ωsrc::AbstractMeshType{D}, x, ::Val{D}, outside::Symbol, Slot
) where {D}
    idx, ts = _interp_cell_frac(Ωsrc, x, outside)
    li = LinearIndices(indices(Ωsrc))
    # the `2ᴰ` corners, decoded from the bits of `k - 1` so the tuple length is static. A
    # collapsed axis (one point) has no far corner: its slot is clamped onto the near one,
    # where `t = 0` makes the far corner's weight exactly zero.
    return ntuple(Val(1 << D)) do k
        corner = CartesianIndex(ntuple(d -> ((k - 1) >> (d - 1)) & 1, Val(D)))
        slot = CartesianIndex(ntuple(d -> min(corner[d], size(li, d) - 1), Val(D)))
        (Slot(li[idx + slot]), _interp_corner_weight(ts, corner, Val(D)))
    end
end

# --- Traits: every walker that sees through a wrapper has to see through this one ---- #

# It carries a trial function, so it is never a source however it is wrapped:
# this ensures `innerₕ` constructs a `BilinearProduct` for it.
_is_source_only(::InterpolationNode) = false

function resolve_ast(op::InterpolationNode{D, S, OpType, Side}) where {D, S, OpType, Side}
    inner = resolve_ast(op.inner_op)
    return InterpolationNode{D, S, typeof(inner), Side}(op.src_space, inner, op.outside)
end

@inline function component(
        op::InterpolationNode{D, S, OpType, Side}, i::Int
) where {D, S, OpType, Side}
    inner = component(op.inner_op, i)
    return InterpolationNode{D, S, typeof(inner), Side}(op.src_space, inner, op.outside)
end

# `_collect_region_labels` for `InterpolationNode` comes from its `UnaryWrapper` membership
# (form/block_extract.jl); it recursed the same way and needed no override.

# The reach on the mesh being walked is the inner leaf's (a single point). The columns this
# node names are on the other mesh and are not offsets, so they have no place in an
# offset set; a bilinear term's colouring only ever reads its test factor's reach anyway
# (`stencil_offsets(::BilinearProduct)`, form/stencil_pattern.jl), and an interpolation
# names a trial-side space, not a test one.
stencil_offsets(op::InterpolationNode) = stencil_offsets(op.inner_op)

# Two interpolations are the same shape only when they interpolate from the same space
# under the same out-of-domain policy (gpena/Bramble.jl#223) -- :clamp and :extrapolate
# disagree exactly at the points that matter, so treating them as interchangeable here
# would let symmetry detection paper over a real difference. The symmetry fast path
# compares the two sides of a product for structural equality, and an interpolation on one
# side only must not read as symmetric.
# Two interpolations on opposite sides are never the same shape either: one names columns and
# the other rows, so the symmetry fast path must not read such a product as symmetric.
function _same_operator_shape(a::InterpolationNode{D}, b::InterpolationNode{D}) where {D}
    return _interp_side(a) === _interp_side(b) && a.src_space === b.src_space &&
           a.outside === b.outside && _same_operator_shape(a.inner_op, b.inner_op)
end

# --- The shift trait: which nodes carry something a relabelled offset cannot express -- #
#
# `stencil_shift_trait`'s base method (form/common.jl) indicates translation invariance, which
# holds for a trial or test function regardless of wrapper depth. An interpolation is not: its
# entries name absolute columns determined by `locate_cell` from the point's own coordinates, and
# adding one to an offset indicates nothing about which columns the neighbour reaches. This ladder
# discovers non-translation-invariant nodes under arbitrary wrappers. Each method is determined by
# the operator type alone, allowing the trait to fold away at compile time.
#
# A source is also point-dependent. Marking it here allows `_contracted_left_stencil`
# (operators/inner.jl) to avoid re-deriving masks and spacings manually: a source-only
# subtree's own `local_stencil`, read through this trait, re-evaluates at each neighbour as
# required by value contraction.
stencil_shift_trait(::InterpolationNode) = PointDependentStencil()
stencil_shift_trait(::SourceFunction) = PointDependentStencil()
stencil_shift_trait(::SourceVector) = PointDependentStencil()
stencil_shift_trait(::SourceConstant) = PointDependentStencil()
stencil_shift_trait(::DiracSource) = PointDependentStencil()

# A `GridFunctionScale` is point-dependent in its own right, whatever it wraps: the
# coefficient it reads varies from point to point exactly like a source's value does, so a
# neighbour's contribution needs the coefficient re-read there, not relabelled here. Without
# this, `UnaryWrapper`'s fallback (`stencil_shift_trait(op.inner_op)`, form/stencil_eval.jl)
# forwards to whatever the wrapped trial/test function reports -- translation-invariant --
# and `D₋ₓ(cₕ * u)` reads the coefficient at the point being visited instead of the point
# the difference's tap reaches (gpena/Bramble.jl#271).
#
# This line alone is not enough, and briefly worse than the bug it targets: the generic
# `PointDependentStencil` branch (form/common.jl) discards the operand's own stencil and
# re-evaluates the whole node at the shifted point, which for a `GridFunctionScale` loses
# the trial or test column the operand contributed -- `local_stencil(GridFunctionScale(c, u),
# ..., Ishift)` returns a single entry at offset zero, not the trial column shifted by
# `delta`. The two `shifted_inner_stencil` overrides in form/common.jl are what make this
# line correct: they shift the operand by its own rule and read the coefficient at the
# shifted point separately, instead of asking the trait's two stock branches to do both at
# once.
stencil_shift_trait(::GridFunctionScale) = PointDependentStencil()

function stencil_shift_trait(op::OperatorAdd)
    return _combine_shift_traits(
        stencil_shift_trait(op.left_op), stencil_shift_trait(op.right_op)
    )
end

# --- Which trial contributions interpolate, and from where --------------------------- #
#
# Two separate questions arise:
#
# `_all_trial_interpolated` verifies whether every trial column contributed by the term originates
# from an interpolation. Only in that case is the term exempt from the cross-mesh refusal
# (`_check_block_meshes`). A sum like `πₕ(u) + u` contributes absolute columns from one
# summand and ordinary offsets from the other; the offsets still require both leaves to share an
# index space.
#
# `_bind_interp_spaces` then supplies each interpolation with the leaf whose degrees of freedom
# it names -- the trial leaf for a trial-side node, the test leaf for a test-side one. Nothing
# validates the two against each other any more: the node is given that leaf and has no other
# space to disagree with, which is what dropping `πₕ`'s space argument bought
# (gpena/Bramble.jl#10).
#
# Both are decided by the operator's type alone, allowing each rung to fold to a constant.

# A node that contributes no trial column at all (such as a source or test function) answers `true`
# vacuously, as no mesh correspondence is required.
_all_trial_interpolated(::LazyOp) = false
_all_trial_interpolated(op::InterpolationNode) = _interp_side(op) === TrialSide
_all_trial_interpolated(::SourceFunction) = true
_all_trial_interpolated(::SourceVector) = true
_all_trial_interpolated(::SourceConstant) = true
_all_trial_interpolated(::DiracSource) = true
_all_trial_interpolated(::TestFunction) = true
_all_trial_interpolated(::IndexedTestFunction) = true

# The mirror question, for a test-side interpolation (gpena/Bramble.jl#263): whether every
# row the term scatters into is named by an interpolation. That is the other way a block may
# straddle two meshes without the two leaves having to share an index space.
_all_test_interpolated(::LazyOp) = false
_all_test_interpolated(op::InterpolationNode) = _interp_side(op) === TestSide
_all_test_interpolated(::SourceFunction) = true
_all_test_interpolated(::SourceVector) = true
_all_test_interpolated(::SourceConstant) = true
_all_test_interpolated(::DiracSource) = true
_all_test_interpolated(::TrialFunction) = true
_all_test_interpolated(::IndexedTrialFunction) = true

function _all_test_interpolated(op::OperatorAdd)
    return _all_test_interpolated(op.left_op) && _all_test_interpolated(op.right_op)
end

_all_test_interpolated(op::BilinearProduct) = _all_test_interpolated(op.right_op)
_all_test_interpolated(op::LinearProduct) = true

# Whether the term carries a test-side interpolation anywhere. This is what decides which
# leaf's grid the sweep walks: the native side's, since that is the side whose mesh supplies
# the quadrature weight the product integrates against. Decided by type alone, so the choice
# folds away at compile time.
_has_test_interp(::LazyOp) = false
_has_test_interp(op::InterpolationNode) = _interp_side(op) === TestSide

# The same question for the trial side, which only `_check_one_interpolated_side`
# (operators/inner.jl) asks: the walked leaf does not depend on it, since a trial-side
# interpolation leaves the test side native and that is where the sweep already walks.
_has_trial_interp(::LazyOp) = false
_has_trial_interp(op::InterpolationNode) = _interp_side(op) === TrialSide

function _has_trial_interp(op::OperatorAdd)
    return _has_trial_interp(op.left_op) || _has_trial_interp(op.right_op)
end

function _has_trial_interp(op::BilinearProduct)
    return _has_trial_interp(op.left_op) || _has_trial_interp(op.right_op)
end

_has_trial_interp(op::LinearProduct) = false

function _has_test_interp(op::OperatorAdd)
    return _has_test_interp(op.left_op) || _has_test_interp(op.right_op)
end

function _has_test_interp(op::BilinearProduct)
    return _has_test_interp(op.left_op) || _has_test_interp(op.right_op)
end

_has_test_interp(op::LinearProduct) = false

"""
    _walked_leaf(term, trial_leaf, test_leaf)

The leaf whose grid the assembly sweep walks for `term`, and whose weights and markers its
stencil sees.

The test leaf, as it has always been, unless the term interpolates on the test side: then the
rows are named absolutely and the trial leaf is the one that stays native, so it supplies the
grid, the quadrature weight and the columns (gpena/Bramble.jl#263).
"""
@inline function _walked_leaf(term, trial_leaf, test_leaf)
    return _has_test_interp(term) ? trial_leaf : test_leaf
end

# A sum requires both summands to interpolate.
function _all_trial_interpolated(op::OperatorAdd)
    return _all_trial_interpolated(op.left_op) && _all_trial_interpolated(op.right_op)
end

# Only the trial side of a product contributes columns, so only the trial side is inspected. A
# linear product contributes none: its left factor is contracted away
# (`multiply_stencils_linear`), which is why a source interpolation belongs there and an
# operator one does not.
_all_trial_interpolated(op::BilinearProduct) = _all_trial_interpolated(op.left_op)
_all_trial_interpolated(op::LinearProduct) = true

# Bind every interpolation the term carries to the leaf whose columns it writes into.
#
# `πₕ(u)` names no space: the columns it produces are numbered in the trial function's own
# space, which is exactly `blk.trial_leaf`, and on a composite space that leaf is only known
# once `blocks` has resolved the term's `component_idx`. This pass stamps it in, with the
# recursion shape of `resolve_ast` -- the fallback returns the term untouched, so a term
# carrying no interpolation rebuilds nothing.
#
# Every method is decided by the operator's type alone, so the walk folds away at compile
# time and a bound term is as concrete as the one it came from. Pattern discovery and
# execution must bind identically, or the pattern reserves entries the sweep never fills.
_bind_interp_spaces(op::Any, trial_leaf, test_leaf) = op

# Each node binds to the leaf whose degrees of freedom it names: the trial leaf for a
# trial-side interpolation, the test leaf for a test-side one (gpena/Bramble.jl#263). Both
# leaves are threaded through the whole walk, so one form may interpolate on either side in
# different terms.
#
# The leaf is stored as its `host_weights` mirror (gpena/Bramble.jl#363): the stencil locates
# the cell by reading the source mesh's points one at a time, which a device mesh cannot serve,
# and the mirror numbers the same degrees of freedom. On a host leaf `host_weights` returns the
# leaf itself, so the host path binds exactly what it bound before. Binding happens once per
# fill, never at `form` construction, so a mirror never outlives a `change_points!`.
function _bind_interp_spaces(
        op::InterpolationNode{D, S, OpType, TrialSide}, trial_leaf, test_leaf
) where {D, S, OpType}
    inner = _bind_interp_spaces(op.inner_op, trial_leaf, test_leaf)
    src = host_weights(trial_leaf)
    return InterpolationNode{D, typeof(src), typeof(inner), TrialSide}(
        src, inner, op.outside
    )
end

function _bind_interp_spaces(
        op::InterpolationNode{D, S, OpType, TestSide}, trial_leaf, test_leaf
) where {D, S, OpType}
    inner = _bind_interp_spaces(op.inner_op, trial_leaf, test_leaf)
    src = host_weights(test_leaf)
    return InterpolationNode{D, typeof(src), typeof(inner), TestSide}(
        src, inner, op.outside
    )
end

function _bind_interp_spaces(op::OperatorAdd{D}, trial_leaf, test_leaf) where {D}
    left = _bind_interp_spaces(op.left_op, trial_leaf, test_leaf)
    right = _bind_interp_spaces(op.right_op, trial_leaf, test_leaf)
    return OperatorAdd{D, typeof(left), typeof(right)}(left, right)
end

# Both sides of a product are bound, since either may carry an interpolation: the trial side
# names columns and the test side rows. A `LinearProduct` contracts its left factor away and
# can hold no interpolation node at all (`_is_source_only` answers false for one, which is
# what makes `innerₕ` build a `BilinearProduct` instead), so it binds nothing.
function _bind_interp_spaces(
        op::BilinearProduct{D, InnerType}, trial_leaf, test_leaf
) where {D, InnerType}
    left = _bind_interp_spaces(op.left_op, trial_leaf, test_leaf)
    right = _bind_interp_spaces(op.right_op, trial_leaf, test_leaf)
    return BilinearProduct{D, InnerType, typeof(left), typeof(right)}(left, right)
end

function _bind_interp_spaces(ops::NTuple{N, Any}, trial_leaf, test_leaf) where {N}
    map(
        op -> _bind_interp_spaces(op, trial_leaf, test_leaf), ops
    )
end

# --- Expression rendering (gpena/Bramble.jl#274) ----------------------------------- #

# Operand only, per the plan's departure from the issue text: `src_space` is `nothing` until
# assembly binds it (`_bind_interp_spaces`) and carries no name a caller wrote, so rendering
# it would show either `nothing` or an internal leaf object instead of anything the caller
# recognizes.
expression(op::InterpolationNode) = "πₕ($(expression(op.inner_op)))"
