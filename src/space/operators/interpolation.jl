#===========================================================================#
# Interpolation between two grid functions built over different meshes.
#
# Point 25: what makes a heterogeneous composite space (point 24) useful rather than
# merely indexable — moving a grid function from one leaf's mesh to another's. The
# numeric operators live here, alongside the other operators over `VectorElement`; the
# symbolic AST wrapper `πₕ` lives in `form/operators/interpolation.jl`, since it needs
# `SourceFunction`, which is not defined until `form/common.jl`.
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

This is the building block both `interpolate!`/`interpolate` (below, the numeric operator)
and [`πₕ`](@ref) (the symbolic one) use — `x -> interpolate_at(uₕ, x)` is itself a valid
source function, usable anywhere one is accepted, including directly as [`Rₕ`](@ref)'s own
argument: `interpolate` *is* `Rₕ` applied to this one function, not a separate mechanism.
`Rₕ(Wₕ, f)` restricts an arbitrary continuous `f`; when `f` happens to be another grid
function's own interpolant, restricting it is interpolating it, which is why `interpolate`
generalises `Rₕ` for the case the source is discrete rather than a closed-form function.
"""
function interpolate_at(uₕ::VectorElement{<:ScalarGridSpace{1}}, x)
    Ωₕ = mesh(space(uₕ))
    i = locate_cell(Ωₕ, x)
    pts = points(Ωₕ)
    lo, hi = pts[i], pts[i + 1]
    t = hi > lo ? (x - lo) / (hi - lo) : zero(x - lo)
    return (1 - t) * uₕ[i] + t * uₕ[i + 1]
end

function interpolate_at(uₕ::VectorElement{<:ScalarGridSpace{D}}, x) where {D}
    Ωₕ = mesh(space(uₕ))
    idx = locate_cell(Ωₕ, x)
    li = LinearIndices(indices(Ωₕ))

    ts = ntuple(Val(D)) do d
        pts = points(Ωₕ(d))
        i = idx[d]
        lo, hi = pts[i], pts[i + 1]
        hi > lo ? (x[d] - lo) / (hi - lo) : zero(x[d] - lo)
    end

    acc = zero(promote_type(eltype(uₕ), typeof(first(ts))))
    for corner in CartesianIndices(ntuple(_ -> 0:1, Val(D)))
        w = one(eltype(ts))
        for d in 1:D
            w *= corner[d] == 1 ? ts[d] : (1 - ts[d])
        end
        acc += w * uₕ[li[idx + corner]]
    end
    return acc
end

"""
	interpolate!(dest::VectorElement, src::VectorElement) -> dest

Fills `dest` with the piecewise (multi)linear interpolant of `src`, sampled at `dest`'s own
mesh points.

Not a separate implementation: `interpolate_at(src, ·)` is itself a genuine function of a
physical point — the interpolant, evaluable anywhere, not only at `src`'s own grid points —
so this is exactly [`Rₕ!`](@ref)`(dest, x -> interpolate_at(src, x))`. `dest` and `src` may
be built over entirely different meshes, which is the whole point; `Rₕ!` already handles
"evaluate a function of a physical point at each of `dest`'s own grid points" for any
function, interpolants included, threading or not following `dest`'s own backend
[`execution_policy`](@ref) the same way it always does.
"""
@inline interpolate!(dest::VectorElement, src::VectorElement) = Rₕ!(
    dest, x -> interpolate_at(src, x))

"""
	interpolate(Wₕ::ScalarGridSpace, src::VectorElement) -> VectorElement

`Rₕ(Wₕ, x -> interpolate_at(src, x))` — see [`interpolate!`](@ref). The element type is
promoted from `Wₕ`'s and `src`'s own, the same rule `Rₕ` already applies to any source
function, so interpolating a `Dual`-valued `src` gives a `Dual`-valued result on an
undifferentiated `Wₕ`.
"""
@inline interpolate(Wₕ::ScalarGridSpace, src::VectorElement) = Rₕ(
    Wₕ, x -> interpolate_at(src, x))

# --- Triplet assembly shared by both dimensionalities of interpolation_matrix ---
#
# The same corner-weight arithmetic interpolate_at uses, emitting (row, col, weight)
# triplets instead of accumulating a value against one src's data — so the two are kept in
# step by construction, not by remembering to update both when the formula changes.

function _interpolation_triplets!(rows, cols, vals, Ωdest::AbstractMeshType,
        Ωsrc::AbstractMeshType{1})
    li_dest = LinearIndices(indices(Ωdest))
    pts = points(Ωsrc)
    for i in indices(Ωdest)
        x = point(Ωdest, i)
        j = locate_cell(Ωsrc, x)
        lo, hi = pts[j], pts[j + 1]
        t = hi > lo ? (x - lo) / (hi - lo) : zero(x - lo)
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
        idx = locate_cell(Ωsrc, x)

        ts = ntuple(Val(D)) do d
            pts = points(Ωsrc(d))
            i = idx[d]
            lo, hi = pts[i], pts[i + 1]
            hi > lo ? (x[d] - lo) / (hi - lo) : zero(x[d] - lo)
        end

        row = li_dest[I]
        for corner in CartesianIndices(ntuple(_ -> 0:1, Val(D)))
            w = one(eltype(ts))
            for d in 1:D
                w *= corner[d] == 1 ? ts[d] : (1 - ts[d])
            end
            push!(rows, row)
            push!(cols, li_src[idx + corner])
            push!(vals, w)
        end
    end
end

"""
    interpolation_matrix(Wdest::ScalarGridSpace, Wsrc::ScalarGridSpace) -> SparseMatrixCSC

The piecewise (multi)linear interpolant of [`interpolate`](@ref)/[`interpolate_at`](@ref) as
a sparse matrix `P` rather than applied pointwise:
`P * values(src) ≈ values(interpolate(Wdest, src))` for any `src::VectorElement` over
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
