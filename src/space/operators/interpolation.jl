#===========================================================================#
# Interpolation between two grid functions built over different meshes.
#
# Point 25: what makes a heterogeneous composite space (point 24) useful rather than
# merely indexable — moving a grid function from one leaf's mesh to another's. The
# numeric operators (πₕ/πₕ!) live here, alongside the other operators over
# `VectorElement`, following the same `Xₕ`/`Xₕ!` pairing `Rₕ`/`Rₕ!` and `avgₕ`/`avgₕ!`
# already use. `πₕ` also names the *symbolic* AST wrapper, one argument fewer — dispatch
# tells the two apart by arity, not by a different name — which lives in
# `form/operators/interpolation.jl` since it needs `SourceFunction`, not defined until
# `form/common.jl`; that file adds a method to the same `πₕ`, it does not shadow it.
#===========================================================================#

"""
	interpolate_at(uₕ::VectorElement, x)

The piecewise (multi)linear interpolant of `uₕ` at the physical point `x`, using `uₕ`'s
own mesh.

Locates the cell of `mesh(space(uₕ))` containing `x` ([`locate_cell`](@ref)) and blends the
``2^D`` grid values at that cell's corners, weighted by `x`'s relative position within it —
the standard bilinear/trilinear construction, exact for any affine function of the
coordinates and correct on a non-uniform mesh, since it reads the mesh's own point
coordinates rather than assuming a fixed step. `locate_cell` clamps which *cell* a point
outside the mesh is read against to the boundary cell, but the relative position `x` is
weighted by is not itself clamped — so a point outside the mesh is a linear extrapolation
along that boundary cell's own slope, not a constant hold of the boundary value.

This is the building block both `πₕ!`/`πₕ` (below, the numeric operator) and the
one-argument, symbolic `πₕ` use — `x -> interpolate_at(uₕ, x)` is itself a valid source
function, usable anywhere one is accepted, including directly as [`Rₕ`](@ref)'s own argument:
`πₕ(Wₕ, src)` *is* `Rₕ` applied to this one function, not a separate mechanism. `Rₕ(Wₕ, f)`
restricts an arbitrary continuous `f`; when `f` happens to be another grid function's own
interpolant, restricting it is interpolating it, which is why `πₕ` generalises `Rₕ` for the
case the source is discrete rather than a closed-form function.
"""
function interpolate_at(uₕ::VectorElement{<:ScalarGridSpace{1}}, x)
    Ωₕ = mesh(space(uₕ))
    i, t = _interp_cell_frac(Ωₕ, x)
    return (1 - t) * uₕ[i] + t * uₕ[i + 1]
end

function interpolate_at(uₕ::VectorElement{<:ScalarGridSpace{D}}, x) where {D}
    Ωₕ = mesh(space(uₕ))
    idx, ts = _interp_cell_frac(Ωₕ, x)
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
# absolute trial columns. Factored so the three cannot drift: an interpolation that disagreed
# with its own matrix depending on which spelling you reached for would be a bad bug to have.
#
# `locate_cell` clamps which cell an outside point is read against, and the fraction is *not*
# clamped, so a point beyond the boundary extrapolates along that cell's slope — see
# `interpolate_at`'s docstring.
@inline function _interp_cell_frac(Ωₕ::AbstractMeshType{1}, x)
    i = locate_cell(Ωₕ, x)
    pts = points(Ωₕ)
    lo, hi = pts[i], pts[i + 1]
    t = hi > lo ? (x - lo) / (hi - lo) : zero(x - lo)
    return i, t
end

@inline function _interp_cell_frac(Ωₕ::AbstractMeshType{D}, x) where {D}
    idx = locate_cell(Ωₕ, x)
    ts = ntuple(Val(D)) do d
        pts = points(Ωₕ(d))
        i = idx[d]
        lo, hi = pts[i], pts[i + 1]
        hi > lo ? (x[d] - lo) / (hi - lo) : zero(x[d] - lo)
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
	πₕ!(dest::VectorElement, src::VectorElement) -> dest

Fills `dest` with the piecewise (multi)linear interpolant of `src`, sampled at `dest`'s own
mesh points — the in-place numeric interpolation operator, named after [`Rₕ!`](@ref)/
[`avgₕ!`](@ref)'s own `Xₕ!` convention.

Not a separate implementation: `interpolate_at(src, ·)` is itself a genuine function of a
physical point — the interpolant, evaluable anywhere, not only at `src`'s own grid points —
so this is exactly `Rₕ!(dest, x -> interpolate_at(src, x))`. `dest` and `src` may be built
over entirely different meshes, which is the whole point; `Rₕ!` already handles "evaluate a
function of a physical point at each of `dest`'s own grid points" for any function,
interpolants included, threading or not following `dest`'s own backend
[`execution_policy`](@ref) the same way it always does.
"""
@inline πₕ!(dest::VectorElement, src::VectorElement) = Rₕ!(
    dest, x -> interpolate_at(src, x))

"""
	πₕ(Wₕ::ScalarGridSpace, src::VectorElement) -> VectorElement

`Rₕ(Wₕ, x -> interpolate_at(src, x))` — see [`πₕ!`](@ref). The same name as the one-argument,
*symbolic* `πₕ(uₕ)` ([`source_function`](@ref) wrapper, `form/operators/interpolation.jl`):
dispatch tells the two apart by argument count, not by a different name, the numeric pair
following `Rₕ`/`Rₕ!`'s own naming exactly. The element type is promoted from `Wₕ`'s and
`src`'s own, the same rule `Rₕ` already applies to any source function, so interpolating a
`Dual`-valued `src` gives a `Dual`-valued result on an undifferentiated `Wₕ`.
"""
@inline πₕ(Wₕ::ScalarGridSpace, src::VectorElement) = Rₕ(
    Wₕ, x -> interpolate_at(src, x))

# --- Triplet assembly shared by both dimensionalities of interpolation_matrix ---
#
# The same corner-weight arithmetic interpolate_at uses, emitting (row, col, weight)
# triplets instead of accumulating a value against one src's data — so the two are kept in
# step by construction, not by remembering to update both when the formula changes.

function _interpolation_triplets!(rows, cols, vals, Ωdest::AbstractMeshType,
        Ωsrc::AbstractMeshType{1})
    li_dest = LinearIndices(indices(Ωdest))
    for i in indices(Ωdest)
        x = point(Ωdest, i)
        j, t = _interp_cell_frac(Ωsrc, x)
        row = li_dest[i]
        push!(rows, row, row)
        push!(cols, j, j + 1)
        push!(vals, 1 - t, t)
    end
end

function _interpolation_triplets!(rows, cols, vals, Ωdest::AbstractMeshType,
        Ωsrc::AbstractMeshType{D}) where {D}
    li_dest = LinearIndices(indices(Ωdest))
    li_src = LinearIndices(indices(Ωsrc))
    for I in indices(Ωdest)
        x = point(Ωdest, I)
        idx, ts = _interp_cell_frac(Ωsrc, x)

        row = li_dest[I]
        for corner in CartesianIndices(ntuple(_ -> 0:1, Val(D)))
            push!(rows, row)
            push!(cols, li_src[idx + corner])
            push!(vals, _interp_corner_weight(ts, corner, Val(D)))
        end
    end
end

"""
    interpolation_matrix(Wdest::ScalarGridSpace, Wsrc::ScalarGridSpace) -> SparseMatrixCSC

The piecewise (multi)linear interpolant of [`πₕ`](@ref)/[`interpolate_at`](@ref) as
a sparse matrix `P` rather than applied pointwise:
`P * values(src) ≈ values(πₕ(Wdest, src))` for any `src::VectorElement` over
`Wsrc`. `P` is `ndofs(Wdest) × ndofs(Wsrc)` — generally rectangular, since `Wdest` and
`Wsrc` are (the whole point) built over different meshes — with at most ``2^D`` nonzero
entries per row: the corner weights of the source cell [`locate_cell`](@ref) places that
destination point in.

Unlike [`D₋ₓ`](@ref)`(Wₕ)` and the other operator matrices, this is always a
`SparseMatrixCSC`, regardless of either space's own backend `matrix_type`. Those matrices
are built from [`shift`](@ref) — a fixed diagonal offset that a Kronecker product or a
dense fast path both generalise cleanly — but which source cell a destination point falls
in has no such regular structure across two independent meshes: it is genuinely sparse and
irregular, assembled directly from `locate_cell` rather than composed from a handful of
shifts. Converting the result to another matrix type, where that is meaningful, is left to
the caller.
"""
function interpolation_matrix(Wdest::ScalarGridSpace{D}, Wsrc::ScalarGridSpace{D}) where {D}
    Ωdest, Ωsrc = mesh(Wdest), mesh(Wsrc)
    ndest, nsrc = ndofs(Wdest), ndofs(Wsrc)
    T = promote_type(eltype(Wdest), eltype(Wsrc))
    nnz_hint = ndest * 2^D

    rows = sizehint!(Int[], nnz_hint)
    cols = sizehint!(Int[], nnz_hint)
    vals = sizehint!(T[], nnz_hint)

    _interpolation_triplets!(rows, cols, vals, Ωdest, Ωsrc)

    return sparse(rows, cols, vals, ndest, nsrc)
end
