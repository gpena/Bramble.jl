# difference.jl
#
# The difference families over grid functions: the undivided difference (`diff₋`/`diff₊`),
# the divided one that approximates a derivative (`D₋`/`D₊`), the summation-by-parts pairing
# `D̃`, the centered `Dc`, and the second-order non-uniform `D̽ₕ`. Each is generated per
# coordinate from the templates below.
#
# A stencil that runs off the grid is truncated rather than extrapolated: a backward operator
# at the first point and a forward one at the last read the missing neighbour as zero, which
# keeps the operator square and matches a homogeneous Dirichlet condition. The docs tutorial
# on operators shows what that does to each family's boundary slice.
#
# The file holds both halves of the family: the numerical stencils and their
# `@operator_family` aliases, which act on grid functions and build matrices, and then the
# AST nodes (`@node_family`) that the same aliases build when handed a `LazyOp`. It is
# included after the AST core (`ast/ast.jl`, `common.jl`, `expression.jl`,
# `operators/node_family.jl`), which the node half needs, and before every numerical file
# that consumes a difference.

# --- Type System for Dispatch ---
# GridDirection, Forward and Backward moved to stencil.jl (gpena/Bramble.jl#42): the
# generic one-sided traversal framework average.jl, jump.jl and inner_product.jl also
# build on, not just the difference operators here.

# A centered stencil reads both neighbours rather than one, so it is truncated on two
# boundary slices rather than one. `_average_engine!` and the shared `_stencil_ranges`
# take a one-sided direction only; the centered traversal is separate, below, and both
# centered operators share it.
abstract type CenteredStencil <: GridDirection end

# Divides by the whole span the stencil covers, xᵢ₊₁ - xᵢ₋₁.
struct Centered <: CenteredStencil end

# Weights the two one-sided differences by the *opposite* spacings, which is what makes
# it second order on a non-uniform grid where `Centered` is first.
struct CrossWeighted <: CenteredStencil end

# --- Core Difference Computation ---
# @boundscheck rather than @assert: this runs once per grid point and the engine's
# loops are marked @inbounds, which elides the former and cannot elide the latter.
@inline function _get_h_val(h::AbstractVector, i::Int)
    @boundscheck 1 <= i <= length(h) || throw(BoundsError(h, i))
    return @inbounds h[i]
end
@inline _get_h_val(h::F, i::Int) where {F <: Function} = h(i)

# The device kernels in `ext/BrambleKernelAbstractionsExt.jl` need, per grid point, exactly
# the boundary test `_stencil_ranges` (operators/stencil.jl) already encodes as index
# ranges: a forward stencil has no neighbour at the last point along the direction, a
# backward one at the first. Read here as a single index comparison instead of re-deriving a
# `CartesianIndices` split inside the kernel body. Shared by the difference and average
# engines, since both stencils are one-sided in exactly the same sense.
@inline _stencil_boundary_dim(::Forward, n::Int) = n
@inline _stencil_boundary_dim(::Backward, n::Int) = 1

# The kernels take the point and its neighbour in that order, whichever direction the
# stencil runs, so that the engine can hand them `(cur, other)` without knowing which
# is which. `Val{false}` is an interior point, which has a neighbour; `Val{true}` is
# the one boundary slice that does not, where the stencil is truncated.
#
# Unscaled, h === nothing: the plain difference.
@inline @propagate_inbounds _compute_difference(
    ::Forward, ::Val{false}, cur, other, ::Nothing, i
) = other - cur
@inline @propagate_inbounds _compute_difference(
    ::Backward, ::Val{false}, cur, other, ::Nothing, i
) = cur - other
@inline @propagate_inbounds _compute_difference(::Forward, ::Val{true}, cur, ::Nothing, i) = -cur
@inline @propagate_inbounds _compute_difference(
    ::Backward, ::Val{true}, cur, ::Nothing, i
) = cur

# Scaled by the grid spacing: the finite difference.
@inline @propagate_inbounds _compute_difference(::Forward, ::Val{false}, cur, other, h, i) = (other - cur) /
                                                                                             _get_h_val(h, i)
@inline @propagate_inbounds _compute_difference(
    ::Backward, ::Val{false}, cur, other, h, i
) = (cur - other) / _get_h_val(h, i)

# The scaled (finite difference) and centered families' boundary case -- zero, for any
# direction -- is covered by the one `_compute_difference(::GridDirection, ::Val{true}, ...)`
# method below, after the interior kernels.

# The centered kernels take the three points of their stencil in grid order, rather than
# a point and its neighbour. `Centered` does not read the middle one; it is passed anyway
# so that both centered operators can share one traversal.
#
# `h` is the averaged spacing, the same view `D̃` divides by, because
#
#     x_{i+1} - x_{i-1} = h_i + h_{i+1} = 2 h*_i
#
# so the centered denominator is twice it and one lazy view serves both operators.
@inline @propagate_inbounds _compute_difference(
    ::Centered, ::Val{false}, back, _, fwd, h, i
) = (fwd - back) / (2 * _get_h_val(h, i))

# The cross-weighted kernel needs the two spacings separately rather than their sum, so
# its `h` is the mesh's cached spacings themselves:
#
#     D̽ₕ(u)(i) = [h_i (u_{i+1} - u_i) / h_{i+1} + h_{i+1} (u_i - u_{i-1}) / h_i]
#                / (h_i + h_{i+1})
#
# which is the backward differences at x_{i+1} and at x_i weighted by h_i and h_{i+1}
# respectively. Reading h[i+1] is in range because the interior stops at the last point
# that has a forward neighbour.
@inline @propagate_inbounds function _compute_difference(
        ::CrossWeighted, ::Val{false}, back, cur, fwd, h, i
)
    hᵢ = _get_h_val(h, i)
    hᵢ₊₁ = _get_h_val(h, i + 1)
    return (hᵢ * (fwd - cur) / hᵢ₊₁ + hᵢ₊₁ * (cur - back) / hᵢ) / (hᵢ + hᵢ₊₁)
end

# The one boundary method for every direction: Forward, Backward or CenteredStencil alike
# have no stencil on a truncated slice, so all read zero there. `zero(cur)` rather than a
# literal keeps the element type of the grid. No ambiguity against the `::Nothing` methods
# above (the unscaled boundary case): those are more specific in both the direction and the
# `h` slot.
@inline @propagate_inbounds _compute_difference(::GridDirection, ::Val{true}, cur, h, i) = zero(cur)

# The two-sided (centred) engine's boundary call carries the one neighbour still on the
# grid, in addition to `cur` -- `Centered`/`D̃` (the latter routed through the one-sided
# engine and the fallback above; only `Centered` reaches this one) still have no stencil at
# a truncated end and read zero regardless, ignoring it. `CrossWeighted` overrides this
# below to use it (gpena/Bramble.jl#183).
@inline @propagate_inbounds _compute_difference(
    ::CenteredStencil, ::Val{true}, cur, neighbour, h, i
) = zero(cur)

# `D̽ₕ` has no missing-neighbour convention of its own to truncate to: with only one side
# of the stencil still on the grid, it collapses to the one-sided difference that side
# still defines -- the forward difference at the first point (`neighbour` is `u_2`) and
# the backward difference at the last (`neighbour` is `u_{n-1}`). The engine only ever
# calls this at `i == 1` or `i == n` (`_check_centered_points` guarantees they differ), so
# that comparison alone tells the two apart (gpena/Bramble.jl#183).
@inline @propagate_inbounds function _compute_difference(
        ::CrossWeighted, ::Val{true}, cur, neighbour, h, i
)
    hᵢ = _get_h_val(h, i)
    return i == 1 ? (neighbour - cur) / hᵢ : (cur - neighbour) / hᵢ
end

# --- The forward difference over the averaged spacing (D̃) ------------------------- #
#
#   D̃(uₕ)(i) = (u(xᵢ₊₁) - u(xᵢ)) / ((hᵢ + hᵢ₊₁) / 2)
#
# The forward difference divided by the averaged spacing rather than by the forward
# spacing. Away from the boundary that denominator is the width of the cell around xᵢ,
# so this and `D₊` differ only in what they scale by, but the averaged form is what makes
# the discrete integration-by-parts identity close.
#
# The denominator is read lazily off the mesh's cached spacings rather than stored: entry
# i needs `spacings[i]` and `spacings[i+1]`, so a vector of its own would be a third copy
# of the axis to keep in step with refinement, for two loads it already has.

"""
    StarSpacings(h)

Lazy view of the averaged spacings ``(h_i + h_{i+1})/2`` over a mesh's cached backward
spacings `h`, which is what [`D̃ₓ`](@ref) divides by.

Entry `i` reads `h[i]` and `h[i+1]`, so it is defined for `i < length(h)`. That is exactly
the range the forward stencil's interior covers; the last point has no forward neighbour
and the engine truncates it to zero without consulting this.
"""
struct StarSpacings{T, V <: AbstractVector{T}} <: AbstractVector{T}
    h::V
end

@inline Base.size(s::StarSpacings) = (length(s.h) - 1,)
@inline Base.@propagate_inbounds function Base.getindex(s::StarSpacings, i::Int)
    @boundscheck 1 <= i < length(s.h) || throw(BoundsError(s, i))
    return @inbounds (s.h[i] + s.h[i + 1]) / 2
end

"""
    star_spacings(Ωₕ::Mesh1D)

Returns the averaged spacings ``(h_i + h_{i+1})/2`` of `Ωₕ` as a [`StarSpacings`](@ref)
view over its cached backward spacings. Allocates nothing.

Away from the first point this equals [`half_spacing`](@ref)`(Ωₕ, i)`. At `i = 1` it does
not: the cached `h₁` repeats the first interval, so this gives ``x_2 - x_1`` where the cell
width gives half of it, the boundary cell being a half cell.
"""
@inline star_spacings(Ωₕ::Mesh1D) = StarSpacings(spacings(Ωₕ))

# The device kernels below take `h` as a plain top-level array or `nothing`, never as a
# wrapper struct: a struct nesting a device array fails kernel compilation even as a
# top-level kernel argument (`ext/BrambleKernelAbstractionsExt.jl`'s module comment explains
# why, gpena/Bramble.jl#94, #174). `StarSpacings` is exactly such a wrapper, so its lazy
# averaging is materialized into a plain vector once, with the same two-array bulk
# arithmetic the device kernels for the mesh's own O(n) setup use, before any difference
# kernel launches. Every other shape `_apply_spaced!` ever derives -- `nothing`, or a plain
# vector/view of cached spacings -- is already kernel-safe and passes through unchanged.
@inline _resolve_device_spacing(::Nothing) = nothing
@inline _resolve_device_spacing(h::AbstractVector) = h
@inline function _resolve_device_spacing(h::StarSpacings)
    hv = h.h
    return (@view(hv[1:(end - 1)]) .+ @view(hv[2:end])) ./ 2
end

# A centered stencil reads both neighbours, so it needs a point on each side and is
# undefined on a mesh with fewer than three points along the direction it differences.
# Without this the operator returns all zeros (every point being truncated), which is a
# plausible-looking answer to a question that has none, and the kind of silent result that
# halves a measured convergence order without failing anything. Thrown rather than
# asserted: this checks a caller argument, and an @assert reports a size mismatch as an
# AssertionError, which is not what a caller should have to catch.
@noinline function _throw_centered_too_few_points(dim::Int, n::Int)
    throw(
        ArgumentError(
        "a centered difference along direction $dim needs at least 3 points there, got $n",
    ),
    )
end

@inline function _apply_stencil!(
        vₕ::VectorElement{<:ScalarGridSpace},
        uₕ::VectorElement{<:ScalarGridSpace},
        h,
        dir::GridDirection,
        dim_val::Val
)
    _check_no_alias(vₕ, uₕ)
    _check_same_grid(vₕ, uₕ)
    sp = space(uₕ)
    if execution_policy(sp) isa GpuPolicy
        dev = ka_device(backend(sp))
        _launch_stencil_engine!(
            vₕ.data, uₕ.data, _resolve_device_spacing(h), _grid_dims(uₕ), dir, dim_val, dev
        )
    else
        _difference_engine!(
            execution_policy(sp), vₕ.data, uₕ.data, h, _grid_dims(uₕ), dir, dim_val
        )
    end
    return nothing
end

# --- Deriving h and checking preconditions from a leaf's own submesh -------------- #
#
# Every family below (unscaled and finite differences, D̃, Dc, D̽ₕ) shares one shape:
# derive `h` (or nothing) from the direction's submesh, optionally check a precondition on
# it, then apply the stencil. Only what `h` is and whether there is a precondition differ.
# `spacing_func`/`precheck` are ordinary named functions, never closures over local state,
# so each of the family's call sites still specializes to its own zero-allocation method --
# the same guarantee the duplicated versions this replaces already had.

@inline _no_spacing(sub) = nothing
@inline _no_precheck(sub, dim::Int) = nothing

# A centered stencil needs a point on each side; shared by Dc and D̽ₕ, the two families that
# check it.
@inline function _check_centered_points(sub, dim::Int)
    npoints(sub) >= 3 || _throw_centered_too_few_points(dim, npoints(sub))
    return nothing
end

@inline function _apply_spaced!(
        vₕ::VectorElement{<:ScalarGridSpace},
        uₕ::VectorElement{<:ScalarGridSpace},
        spacing_func::F,
        precheck::P,
        dir::GridDirection,
        dim_val::Val{DIM}
) where {F, P, DIM}
    # The direction check matters because a `Val` past the mesh dimension would otherwise
    # reach the stencil and fault or return all zeros. It comes before `_op_mesh(uₕ)(DIM)`:
    # on a 1D mesh `Ωₕ(2) === Ωₕ` passes, and the index `I[DIM]` faults only later.
    D = length(_grid_dims(uₕ))
    1 <= DIM <= D || _throw_stencil_dim_error(DIM, D)
    sub = _op_mesh(uₕ)(DIM)
    precheck(sub, DIM)
    _apply_stencil!(vₕ, uₕ, spacing_func(sub), dir, dim_val)
    return vₕ
end

# A composite grid function is differenced one component at a time. A leaf's mesh is not
# necessarily the whole composite's, so `sub` (and whatever it derives) has to be
# re-evaluated per leaf rather than once from `uₕ` -- checking or fetching it once only
# validated leaf 1, and reused its value on every other leaf (gpena/Bramble.jl#79).
# Recursing into the scalar method above does that for free, since each leaf is itself a
# scalar space.
@inline function _apply_spaced!(
        vₕ::VectorElement{<:CompositeGridSpace},
        uₕ::VectorElement{<:CompositeGridSpace},
        spacing_func,
        precheck,
        dir::GridDirection,
        dim_val::Val
)
    _apply_componentwise!(
        (v, u) -> _apply_spaced!(v, u, spacing_func, precheck, dir, dim_val), vₕ, uₕ
    )
    return vₕ
end

# --- Unified Difference Engine ---
# `h` carries a type parameter on purpose. Julia does not specialise on an argument of
# function type unless the body calls it directly, and this body only forwards it to
# _get_h_val. Without `H` the spacing callable stays boxed and each element pays a
# dynamic dispatch: measured 13768 us and 6.4 MB against 29 us and no allocation on a
# 100000-point 1D grid.
function _difference_engine!(
        out, in_ref, h::H, dims::NTuple{D, Int}, dir::GridDirection, ::Val{DIM}
) where {H, D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    interior, boundary = _stencil_ranges(axes(li), Val(DIM), dir)

    @inbounds @simd for I in CartesianIndices(interior)
        idx, other = li[I], li[_neighbour(dir, I, step)]
        out[idx] = _compute_difference(
            dir, Val(false), in_ref[idx], in_ref[other], h, I[DIM]
        )
    end

    @inbounds @simd for I in CartesianIndices(boundary)
        idx = li[I]
        out[idx] = _compute_difference(dir, Val(true), in_ref[idx], h, I[DIM])
    end

    return nothing
end

# --- Centered traversal ---------------------------------------------------------- #
# A centered stencil reaches both ways, so its interior is the slice with a neighbour on
# each side and it truncates on two boundary slices rather than one. That is a different
# shape from `_stencil_ranges`, whose two-value result the one-sided engines and the
# average engine destructure, so it is written separately rather than folded in.
@inline function _centered_stencil_ranges(
        full_axes::NTuple{D, Any}, ::Val{DIM}
) where {D, DIM}
    interior = ntuple(
        d -> d == DIM ? ((first(full_axes[d]) + 1):(last(full_axes[d]) - 1)) : full_axes[d],
        Val(D)
    )
    lo = ntuple(
        d -> d == DIM ? (first(full_axes[d]):first(full_axes[d])) : full_axes[d], Val(D)
    )
    hi = ntuple(
        d -> d == DIM ? (last(full_axes[d]):last(full_axes[d])) : full_axes[d], Val(D)
    )
    return interior, lo, hi
end

# `h` carries a type parameter for the same reason it does in the one-sided engine: an
# argument of function type that the body only forwards is not specialised on, and the
# spacing would be boxed for every grid point.
function _difference_engine!(
        out, in_ref, h::H, dims::NTuple{D, Int}, dir::CenteredStencil, ::Val{DIM}
) where {H, D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    interior, lo, hi = _centered_stencil_ranges(axes(li), Val(DIM))

    @inbounds @simd for I in CartesianIndices(interior)
        idx = li[I]
        back, fwd = li[I - step], li[I + step]
        out[idx] = _compute_difference(
            dir, Val(false), in_ref[back], in_ref[idx], in_ref[fwd], h, I[DIM]
        )
    end

    # Each end slice reads its own single neighbour still on the grid -- the one at `lo`
    # forward (there is nothing behind it), the one at `hi` backward (nothing past it) --
    # so unlike the interior loop above, the two are not the same shape and stay two
    # explicit loops rather than one over `(lo, hi)` (gpena/Bramble.jl#183).
    @inbounds @simd for I in CartesianIndices(lo)
        idx, fwd = li[I], li[I + step]
        out[idx] = _compute_difference(dir, Val(true), in_ref[idx], in_ref[fwd], h, I[DIM])
    end

    @inbounds @simd for I in CartesianIndices(hi)
        idx, back = li[I], li[I - step]
        out[idx] = _compute_difference(dir, Val(true), in_ref[idx], in_ref[back], h, I[DIM])
    end

    return nothing
end

#------------------------------------------------------------------------------------------#
# Policy-dispatched CPU stencil engines (gpena/Bramble.jl#356)
#
# `CpuSerial` runs the two engines above unchanged. `CpuThreaded` cuts the grid into
# `Threads.nthreads()` static bands along its last axis (the single axis in 1D) and runs,
# per band, the same interior and boundary loops restricted to that slab. The
# interior/boundary split is along the differencing axis `DIM`, which is the banded axis
# only when `DIM == D`: a band's slab then still reads its neighbour across the band edge,
# which is safe because only the writes need to be disjoint, and each boundary slice falls
# in exactly one band. `CpuPolyester` reaches `_batch_difference_engine!`, which
# `BramblePolyesterExt` fills by running the same `_difference_band!` per band.
#------------------------------------------------------------------------------------------#

# `ranges` with its last axis cut down to `band`. Built from `first`/`last` rather than
# `intersect` so every slab has one concrete `UnitRange` type; an empty slab (a band
# missing the boundary slice, or a band of a grid shorter than the thread count) is an
# empty range and its loop does nothing.
@inline function _band_slab(ranges::NTuple{D, Any}, band::AbstractUnitRange) where {D}
    r = ranges[D]
    return (Base.front(ranges)..., max(first(r), first(band)):min(last(r), last(band)))
end

"""
    _difference_band!(out, in_ref, h, dims::NTuple{D, Int}, dir::GridDirection, dim_val::Val, nbands::Int, b::Int) -> Nothing

Runs `_difference_engine!`'s loops for `dir` on the `b`-th of `nbands` slabs of the grid
cut along its last axis ([`_band_range`](@ref)). The slabs partition the grid, so running
every band, in any order or concurrently, writes every point of `out` exactly once with the
value the serial engine gives it.
"""
@inline function _difference_band!(
        out, in_ref, h::H, dims::NTuple{D, Int}, dir::GridDirection, ::Val{DIM},
        nbands::Int, b::Int
) where {H, D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    band = _band_range(axes(li, D), nbands, b)
    interior, boundary = _stencil_ranges(axes(li), Val(DIM), dir)
    interior, boundary = _band_slab(interior, band), _band_slab(boundary, band)

    @inbounds @simd for I in CartesianIndices(interior)
        idx, other = li[I], li[_neighbour(dir, I, step)]
        out[idx] = _compute_difference(
            dir, Val(false), in_ref[idx], in_ref[other], h, I[DIM]
        )
    end

    @inbounds @simd for I in CartesianIndices(boundary)
        idx = li[I]
        out[idx] = _compute_difference(dir, Val(true), in_ref[idx], h, I[DIM])
    end

    return nothing
end

@inline function _difference_band!(
        out, in_ref, h::H, dims::NTuple{D, Int}, dir::CenteredStencil, ::Val{DIM},
        nbands::Int, b::Int
) where {H, D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    band = _band_range(axes(li, D), nbands, b)
    interior, lo, hi = _centered_stencil_ranges(axes(li), Val(DIM))
    interior, lo, hi = _band_slab(interior, band), _band_slab(lo, band), _band_slab(hi, band)

    @inbounds @simd for I in CartesianIndices(interior)
        idx = li[I]
        back, fwd = li[I - step], li[I + step]
        out[idx] = _compute_difference(
            dir, Val(false), in_ref[back], in_ref[idx], in_ref[fwd], h, I[DIM]
        )
    end

    @inbounds @simd for I in CartesianIndices(lo)
        idx, fwd = li[I], li[I + step]
        out[idx] = _compute_difference(dir, Val(true), in_ref[idx], in_ref[fwd], h, I[DIM])
    end

    @inbounds @simd for I in CartesianIndices(hi)
        idx, back = li[I], li[I - step]
        out[idx] = _compute_difference(dir, Val(true), in_ref[idx], in_ref[back], h, I[DIM])
    end

    return nothing
end

"""
    _threaded_difference_engine!(out, in_ref, h, dims::Tuple, dir::GridDirection, dim_val::Val) -> Nothing

[`CpuThreaded`](@ref)'s `_difference_engine!`: one [`_difference_band!`](@ref) per thread
under `Threads.@threads :static`, or every band in turn where a `:static` loop cannot start
([`_static_or_serial`](@ref)). Kept in an isolated function, as [`_threaded_for!`](@ref)
is, so the `Threads.@threads` closure is never built on a serial path.
"""
@noinline function _threaded_difference_engine!(
        out, in_ref, h::H, dims::NTuple{D, Int}, dir::GridDirection, dim_val::Val
) where {H, D}
    return _static_or_serial(_static_bands!, _serial_bands!, _difference_band!,
        Threads.nthreads(), out, in_ref, h, dims, dir, dim_val)
end

"""
    _batch_difference_engine!(out, in_ref, h, dims::Tuple, dir::GridDirection, dim_val::Val) -> Nothing

[`CpuPolyester`](@ref)'s `_difference_engine!`, filled by `BramblePolyesterExt` (one
[`_difference_band!`](@ref) per band). The only `src/` method errors naming Polyester.
"""
@noinline function _batch_difference_engine!(out, in_ref, h, dims, dir, dim_val)
    return _throw_cpubatch_without_polyester(:_batch_difference_engine!)
end

# `h::H` for the reason the engines above carry it: a spacing callable that is only
# forwarded is otherwise not specialised on.
@inline _difference_engine!(::CpuSerial, out, in_ref, h::H, dims, dir, dim_val) where {H} = _difference_engine!(
    out, in_ref, h, dims, dir, dim_val)
@inline _difference_engine!(
    ::CpuThreaded, out, in_ref, h::H, dims, dir, dim_val) where {H} = _threaded_difference_engine!(
    out, in_ref, h, dims, dir, dim_val)
@noinline _difference_engine!(
    ::CpuPolyester, out, in_ref, h::H, dims, dir, dim_val) where {H} = _late(_batch_difference_engine!,
    out, in_ref, h, dims, dir, dim_val)

#------------------------------------------------------------------------------------------#
# Device kernel launch stubs (gpena/Bramble.jl#94, #174)
#
# `_apply_stencil!` reaches these once `execution_policy` names a `GpuPolicy`, instead of
# `_difference_engine!`'s scalar-indexing CPU sweep above, which a device array refuses.
# `ext/BrambleKernelAbstractionsExt.jl` fills in the two launchers with a
# `KernelAbstractions.@kernel` that calls the very `_compute_difference` methods above, once
# per grid point, so the device and host answers stay identical by construction rather than
# by two implementations agreeing -- the same reasoning `avgₕ!`'s device path
# (`operators/cell_average.jl`) documents. Without that extension loaded, the launcher
# throws a named diagnostic instead of failing several frames later on a scalar index.
#------------------------------------------------------------------------------------------#

@noinline function _throw_no_ka_stencil_kernel(fname::String)
    return error(
        "$fname requires KernelAbstractions.jl to apply a difference operator to a " *
        "device-backed VectorElement. Add `using KernelAbstractions` (and the package " *
        "providing this backend's device, e.g. `using Metal`) before calling it on this " *
        "backend.",
    )
end

# Deliberately untyped, matching the `_launch_restriction!`/`_launch_half_points!`
# fallback idiom (`operators/restriction.jl`, `src/mesh/mesh1d.jl`): the extension's
# methods are typed on `AbstractVector`/`Tuple`, and a fallback with the same signature
# would overwrite them instead of adding a genuinely more specific dispatch.
"""
    _launch_difference_onesided!(out::AbstractVector, in_ref, h, dims::Tuple, dir::GridDirection, dim_val::Val, dev) -> Nothing

Fills `out` with the one-sided (`Forward`/`Backward`) difference of `in_ref` along the axis
`dim_val`, scaled by `h` when given (`nothing` for the unscaled difference), truncating the
one boundary slice with no neighbour to zero, matching `_compute_difference`'s
`GridDirection` methods. Runs via a `KernelAbstractions.@kernel` launch on `dev`, filled by
`ext/BrambleKernelAbstractionsExt.jl`. The device counterpart of the one-sided branch of
`_difference_engine!`'s CPU sweep above.

# Throws
- `ErrorException`: no `KernelAbstractions` extension is loaded, so there is no device
  kernel to reach (`_throw_no_ka_stencil_kernel`).
"""
function _launch_difference_onesided!(out, in_ref, h, dims, dir, dim_val, dev)
    _throw_no_ka_stencil_kernel(
        "_launch_difference_onesided!"
    )
end

"""
    _launch_difference_centered!(out::AbstractVector, in_ref, h, dims::Tuple, dir::CenteredStencil, dim_val::Val, dev) -> Nothing

Fills `out` with the centered (`Centered`/`CrossWeighted`) difference of `in_ref` along the
axis `dim_val`, scaled by `h`. At the two boundary slices it defers to `dir`'s own
`CenteredStencil` boundary rule in `_compute_difference` -- zero for `Centered`, the
one-sided difference the near side still defines for `CrossWeighted`. Runs via a
`KernelAbstractions.@kernel` launch on `dev`. The device counterpart of the centered branch
of `_difference_engine!`'s CPU sweep above.

# Throws
- `ErrorException`: no `KernelAbstractions` extension is loaded (`_throw_no_ka_stencil_kernel`).
"""
function _launch_difference_centered!(out, in_ref, h, dims, dir, dim_val, dev)
    _throw_no_ka_stencil_kernel(
        "_launch_difference_centered!"
    )
end

# One-sided (`Forward`/`Backward`) and centered (`Centered`/`CrossWeighted`) stencils reach
# different launchers, the same split `_difference_engine!` itself makes above.
@inline function _launch_stencil_engine!(out, in_ref, h, dims, dir::GridDirection, dim_val, dev)
    return _launch_difference_onesided!(out, in_ref, h, dims, dir, dim_val, dev)
end
@inline function _launch_stencil_engine!(out, in_ref, h, dims, dir::CenteredStencil, dim_val, dev)
    return _launch_difference_centered!(out, in_ref, h, dims, dir, dim_val, dev)
end

function difference_shift(
        Ωₕ::AbstractMeshType, ::Val{DIFF_DIM}, ::Val{first}, ::Val{second}
) where {DIFF_DIM, first, second}
    return shift(Ωₕ, Val(DIFF_DIM), Val(first)) - shift(Ωₕ, Val(DIFF_DIM), Val(second))
end

function _difference_operator(
        Ωₕ::AbstractMeshType, ::Forward, ::Val{DIFF_DIM}
) where {DIFF_DIM}
    return difference_shift(Ωₕ, Val(DIFF_DIM), Val(1), Val(0))
end

function _difference_operator(
        Ωₕ::AbstractMeshType, ::Backward, ::Val{DIFF_DIM}
) where {DIFF_DIM}
    return difference_shift(Ωₕ, Val(DIFF_DIM), Val(0), Val(-1))
end

# `spacing_func` carries a type parameter so Julia specialises on it. An argument of
# function type is not specialised on unless the body calls it directly, and this one used
# to be wrapped in a `Base.Fix1` instead, so the closure stayed boxed and every grid point
# paid a dynamic dispatch: 12047 us and 6.5 MB on a 450x450 grid against 322 us and no
# allocation once the parameter is named and the submesh is hoisted.
#
# It is the per-index function rather than the mesh's cached spacing vector on purpose.
# The vector's truncated entry is not meaningful, whereas `spacing_for_derivative` returns
# zero there, and this loop covers every index and turns that zero into a zero weight. The
# engines can use the vector because they visit the truncated slice separately.
function _derivative_weights!(
        v::AbstractVector, Ωₕ::AbstractMeshType, spacing_func::F, ::Val{DIFF_DIM}
) where {F, DIFF_DIM}
    dims = npoints(Ωₕ, Tuple)

    1 <= DIFF_DIM <= dim(Ωₕ) || _throw_stencil_dim_error(DIFF_DIM, dim(Ωₕ))

    sub = Ωₕ(DIFF_DIM)
    li = LinearIndices(dims)

    @inbounds @simd for I in CartesianIndices(dims)
        x = spacing_func(sub, I[DIFF_DIM])
        v[li[I]] = iszero(x) ? zero(eltype(v)) : inv(x)
    end
    return nothing
end

# Configuration array to define forward and backward difference operators.
const _DIFFERENCE_OP_CONFIGS = [
    (
        direction = Forward(),
        diff_name = :forward_difference,
        finite_diff_name = :forward_finite_difference,
        (weights_func!) = :forward_derivative_weights!,
        spacing_func = :forward_spacing_for_derivative,
        spacings_func = :forward_spacings_for_derivative,
        diff_alias = :diff₊,
        finite_diff_alias = :D₊,
        grad_alias = :diff₊ₕ,
        finite_grad_alias = :∇₊ₕ,
        dir_string = "Forward",
        dir_string_lowercase = "forward",
        math_op = "u_{i+1} - u_i",
        math_finite_op = "\\frac{u_{i+1} - u_i}{h_i}",
        unscaled_stencil_op = :UnscaledForwardDiffOp,
        finite_stencil_op = :ForwardFiniteDiffOp
    ),
    (
        direction = Backward(),
        diff_name = :backward_difference,
        finite_diff_name = :backward_finite_difference,
        (weights_func!) = :backward_derivative_weights!,
        spacing_func = :spacing_for_derivative,
        spacings_func = :backward_spacings_for_derivative,
        diff_alias = :diff₋,
        finite_diff_alias = :D₋,
        grad_alias = :diff₋ₕ,
        finite_grad_alias = :∇ₕ,
        dir_string = "Backward",
        dir_string_lowercase = "backward",
        math_op = "u_{i} - u_{i-1}",
        math_finite_op = "\\frac{u_{i} - u_{i-1}}{h_i}",
        unscaled_stencil_op = :UnscaledBackwardDiffOp,
        finite_stencil_op = :BackwardFiniteDiffOp
    )
]

# Metaprogramming loop to generate all specified difference operators.
for config in _DIFFERENCE_OP_CONFIGS
    # Extract ALL values from `config` into local variables here.
    dir_instance = config.direction
    diff_name = config.diff_name
    finite_diff_name = config.finite_diff_name
    weights_func! = config.weights_func!
    spacing_func = config.spacing_func
    spacings_func = config.spacings_func
    diff_alias = config.diff_alias
    finite_diff_alias = config.finite_diff_alias
    grad_alias = config.grad_alias
    finite_grad_alias = config.finite_grad_alias
    dir_string = config.dir_string
    dir_string_lowercase = config.dir_string_lowercase
    math_op = config.math_op
    math_finite_op = config.math_finite_op
    unscaled_stencil_op = config.unscaled_stencil_op
    finite_stencil_op = config.finite_stencil_op

    # This first @eval block is fine because it doesn't depend on any inner loops.
    @eval begin
        # --- In-place applicators ---
        @doc """
            $($(QuoteNode(Symbol(diff_name, :_dim!))))(out, in, [h], dims, diff_dim)

        Low-level, in-place function to compute the **unscaled** $($dir_string_lowercase) difference of vector `in` along dimension `diff_dim`, storing the result in `out`. This function computes ``$($math_op)``.
        """
        function $(Symbol(diff_name, :_dim!))(
                out, in, h, dims::NTuple{D, Int}, diff_dim::Val{DIFF_DIM}
        ) where {D, DIFF_DIM}
            1 <= DIFF_DIM <= D || _throw_stencil_dim_error(DIFF_DIM, D)
            length(out) == length(in) == prod(dims) ||
                _throw_stencil_size_error(length(out), length(in), dims)
            in_ref = (out === in) ? copy(in) : in
            _difference_engine!(out, in_ref, h, dims, $dir_instance, diff_dim)
            return nothing
        end

        function $(Symbol(diff_name, :_dim!))(
                out, in, dims::NTuple{D, Int}, diff_dim::Val{DIFF_DIM}
        ) where {D, DIFF_DIM}
            return $(Symbol(diff_name, :_dim!))(out, in, nothing, dims, diff_dim)
        end

        # --- Weight calculation function ---
        @doc """
            $($(QuoteNode(weights_func!)))(v::AbstractVector, Ωₕ::AbstractMeshType, diff_dim::Val)

        Computes the geometric weights for the $($dir_string_lowercase) finite difference operator and stores them in-place in vector `v`.
        """
        @inline function $weights_func!(
                v::AbstractVector, Ωₕ::AbstractMeshType, diff_dim::Val
        )
            _derivative_weights!(v, Ωₕ, $spacing_func, diff_dim)
        end

        # --- Retained Kronecker oracle (gpena/Bramble.jl#185) ---
        #
        # The bodies `$diff_name`/`$finite_diff_name` used to have, kept under `_kron_`
        # names so `kronecker_operator_matrix` has an independent construction to check
        # `stencil_matrix` against.
        @inline $(Symbol(:_kron_, diff_name))(Ωₕ::AbstractMeshType, dim_val::Val) = _difference_operator(Ωₕ, $dir_instance, dim_val)

        function $(Symbol(:_kron_, finite_diff_name))(
                Ωₕ::AbstractMeshType, dim_val::Val; vector_cache = __vector(Ωₕ)
        )
            diff_matrix = $(Symbol(:_kron_, diff_name))(Ωₕ, dim_val)
            $weights_func!(vector_cache, Ωₕ, dim_val)
            return _scale_rows!(diff_matrix, vector_cache)
        end

        # --- Matrix operator functions (single-pass, gpena/Bramble.jl#185) ---
        @doc """
            $($(QuoteNode(diff_name)))(arg, dim_val::Val)

        Constructs the **unscaled** $($dir_string_lowercase) difference operator, representing the operation ``$($math_op)``.
        """
        @inline $diff_name(Ωₕ::AbstractMeshType, dim_val::Val{DIM}) where {DIM} = stencil_matrix(Ωₕ, $(unscaled_stencil_op){DIM}())

        @doc """
            $($(QuoteNode(finite_diff_name)))(arg, dim_val::Val)

        Constructs the $($dir_string_lowercase) **finite difference** operator, which approximates the first derivative using the formula ``$($math_finite_op)``.
        """
        @inline $finite_diff_name(Ωₕ::AbstractMeshType, dim_val::Val{DIM}) where {DIM} = stencil_matrix(Ωₕ, $(finite_stencil_op){DIM}())

        # --- Generic applicators ---
        #
        # Only the mesh-forwarding overloads are generated here; the grid-function trio
        # (scalar `!`, composite `!`, allocating) comes from
        # `@operator_family` below, which `average.jl` and the centred families
        # share (gpena/Bramble.jl#101).
        @inline $diff_name(Wₕ::AbstractSpaceType, dim_val::Val) = $diff_name(mesh(Wₕ), dim_val)

        @inline $finite_diff_name(Wₕ::AbstractSpaceType, dim_val::Val) = $finite_diff_name(mesh(Wₕ), dim_val)
    end
end

# The grid-function forms and the alias surface of the four one-sided families, one
# `@operator_family` call each. They sit outside the loop above because a macro is expanded
# where it is written: the family's configuration has to be literal at that point, not a
# `config.field` read at load time (gpena/Bramble.jl#258).
#
# The unscaled difference divides by nothing, so it passes `_no_spacing`. The finite one
# hands the engine the mesh's *cached* spacings vector rather than a callable: indexing it
# is 3.6x faster than one call per grid point, and it allocates nothing.
@operator_family(base=forward_difference,
    stem=diff₊,
    apply_fn=_apply_spaced!,
    extra_args=(_no_spacing, _no_precheck),
    direction=Forward(),
    dir_string="forward",
    what="unscaled difference",
    formula="u_{i+1} - u_i",
    formula_note="The unscaled difference is not divided by the grid spacing; the finite difference is.",
    vectorial_alias=diff₊ₕ)

@operator_family(base=forward_finite_difference,
    stem=D₊,
    apply_fn=_apply_spaced!,
    extra_args=(forward_spacings_for_derivative, _no_precheck),
    direction=Forward(),
    dir_string="forward",
    what="finite difference",
    formula="\\frac{u_{i+1} - u_i}{h_i}",
    formula_note="The unscaled difference is not divided by the grid spacing; the finite difference is.",
    vectorial_alias=∇₊ₕ)

@operator_family(base=backward_difference,
    stem=diff₋,
    apply_fn=_apply_spaced!,
    extra_args=(_no_spacing, _no_precheck),
    direction=Backward(),
    dir_string="backward",
    what="unscaled difference",
    formula="u_{i} - u_{i-1}",
    formula_note="The unscaled difference is not divided by the grid spacing; the finite difference is.",
    vectorial_alias=diff₋ₕ)

@operator_family(base=backward_finite_difference,
    stem=D₋,
    apply_fn=_apply_spaced!,
    extra_args=(backward_spacings_for_derivative, _no_precheck),
    direction=Backward(),
    dir_string="backward",
    what="finite difference",
    formula="\\frac{u_{i} - u_{i-1}}{h_i}",
    formula_note="The unscaled difference is not divided by the grid spacing; the finite difference is.",
    vectorial_alias=∇ₕ)

# --- The three centred families: D̃, Dc, D̽ₕ ------------------------------------ #
#
# Each family's grid-function form is the same three-method shape `_DIFFERENCE_OP_CONFIGS`
# already generates above for the two one-sided families: a scalar `!`, a composite `!`
# recursing into it, and a non-mutating wrapper built on `_apply_spaced!` -- differing only
# in the spacing function, the precondition, and the direction tag. The composite `!`
# method's own rationale (a leaf's mesh is not necessarily the whole composite's, so
# whatever `spacing_func` derives is rebuilt per leaf, gpena/Bramble.jl#79) lives once, on
# `_apply_spaced!`'s composite method above, rather than repeated per family here.
#
# Each is its own `@operator_family` call rather than an entry in a shared config array:
# these three have no unscaled/finite split and no weight function, and each alias's own
# docstring text ("over the averaged spacing", "truncated to zero", "second order... where
# Dc is first") is bespoke prose rather than something `math_op`/`dir_string` could
# template. That prose travels as a string literal carrying `{direction}`/`{suffix}`
# placeholders, which is what a macro can take; it was a closure per family until
# gpena/Bramble.jl#258 removed the `Core.eval` these were generated through.

@operator_family(base=forward_star_difference,
    stem=D̃,
    apply_fn=_apply_spaced!,
    extra_args=(star_spacings, _no_precheck),
    direction=Forward(),
    docstring="""
          forward_star_difference(uₕ::VectorElement, dim_val::Val)

      The forward difference of `uₕ` along `dim_val`, divided by the averaged spacing:

      ```math
      \\tilde{\\textrm{D}}_{+}(\\textrm{u}_h)(i) =
          \\frac{\\textrm{u}_h(x_{i+1}) - \\textrm{u}_h(x_i)}{(h_i + h_{i+1})/2}
      ```

      Reached through [`D̃ₓ`](@ref) and its siblings, and takes a mesh, a grid space or a
      grid function as the other difference families do.

      The last point has no forward neighbour, so it is truncated to zero, as in
      [`D₊ₓ`](@ref).

      See also: [`star_spacings`](@ref), [`D₊ₓ`](@ref).
      """,
    opening_sentence="The forward difference of `uₕ` along the `{direction}` "*
                     "direction over the averaged spacing, "*
                     "``\\frac{u_{i+1} - u_i}{(h_i + h_{i+1})/2}``.",
    trailing_note="The last point along `{direction}` is truncated to zero.",
    bang_opening_sentence="The forward difference of `uₕ` along the `{direction}` "*
                          "direction over the averaged spacing, "*
                          "``\\frac{u_{i+1} - u_i}{(h_i + h_{i+1})/2}``, written "*
                          "into `vₕ`.",
    vectorial_alias=D̃ₕ,
    vectorial_dir_string="averaged-spacing forward",
    vectorial_what="difference")

@operator_family(base=centered_difference,
    stem=Dc,
    apply_fn=_apply_spaced!,
    extra_args=(star_spacings, _check_centered_points),
    direction=Centered(),
    docstring="""
          centered_difference(uₕ::VectorElement, dim_val::Val)

      The centered difference of `uₕ` along `dim_val`:

      ```math
      \\textrm{Dc}(\\textrm{u}_h)(i) =
          \\frac{\\textrm{u}_h(x_{i+1}) - \\textrm{u}_h(x_{i-1})}{h_i + h_{i+1}}
      ```

      Reached through [`Dcₓ`](@ref) and its siblings, and takes a mesh, a grid space or a grid
      function as the other difference families do.

      The denominator is ``x_{i+1} - x_{i-1}``, so the operator reproduces the derivative of an
      affine function exactly on any grid, uniform or not. Both the first and the last point
      lack a neighbour on one side, so both are truncated to zero.

      See also: [`star_spacings`](@ref), [`D₋ₓ`](@ref), [`D₊ₓ`](@ref).
      """,
    opening_sentence="The centered difference of `uₕ` along the `{direction}` "*
    "direction, ``\\frac{u_{i+1} - u_{i-1}}{h_i + h_{i+1}}``.",
    trailing_note="The first and last points along `{direction}` are truncated "*
                  "to zero, so the mesh needs at least three points along "*
                  "`{direction}` and an `ArgumentError` is thrown when it has "*
                  "fewer.",
    bang_opening_sentence="The centered difference of `uₕ` along the `{direction}` "*
                          "direction, ``\\frac{u_{i+1} - u_{i-1}}{h_i + h_{i+1}}``, "*
                          "written into `vₕ`.",
    vectorial_alias=Dcₕ,
    vectorial_dir_string="centered",
    vectorial_what="difference")

@operator_family(base=cross_weighted_difference,
    stem=D̽,
    apply_fn=_apply_spaced!,
    extra_args=(spacings, _check_centered_points),
    direction=CrossWeighted(),
    docstring="""
          cross_weighted_difference(uₕ::VectorElement, dim_val::Val)

      The cross-weighted centered difference of `uₕ` along `dim_val`:

      ```math
      \\overset{\\times}{\\textrm{D}}_{h}(\\textrm{u}_h)(i) =
          \\frac{h_i}{h_i + h_{i+1}}\\, \\textrm{D}_{-}\\textrm{u}_h(x_{i+1}) +
          \\frac{h_{i+1}}{h_i + h_{i+1}}\\, \\textrm{D}_{-}\\textrm{u}_h(x_i)
      ```

      Reached through [`D̽ₓ`](@ref) and its siblings, and takes a mesh, a grid space or a grid
      function as the other difference families do.

      It is the same two one-sided differences [`Dcₓ`](@ref) combines, weighted by the opposite
      spacings. That is the combination which cancels the leading truncation term on a
      non-uniform grid, so this is second order where `Dcₓ` is first, and the two coincide when
      the spacing is constant.

      The first and the last point each lack a neighbour on one side, but unlike `Dcₓ` neither
      is truncated: each collapses to the one-sided difference its near side still defines,
      [`D₊ₓ`](@ref)`(uₕ)` at the first point and [`D₋ₓ`](@ref)`(uₕ)` at the last.

      See also: [`Dcₓ`](@ref), [`D₋ₓ`](@ref), [`D₊ₓ`](@ref).
      """,
    opening_sentence="The cross-weighted centered difference of `uₕ` along "*
                     "the `{direction}` direction, the backward differences "*
                     "at ``x_{i+1}`` and ``x_i`` weighted by ``h_i`` and "*
                     "``h_{i+1}``.",
    alias_note="Second order on a non-uniform grid, where [`Dc{suffix}`](@ref) "*
    "is first.",
    trailing_note="Unlike [`Dc{suffix}`](@ref), the first and last points along "*
                  "`{direction}` are not truncated: with no neighbour on the "*
                  "far side, each collapses to the one-sided difference the "*
                  "near side still gives, [`D₊{suffix}`](@ref) at the first "*
                  "point and [`D₋{suffix}`](@ref) at the last. The mesh still "*
                  "needs at least three points along `{direction}`, and an "*
                  "`ArgumentError` is thrown when it has fewer.",
    bang_opening_sentence="The cross-weighted centered difference of `uₕ` along "*
                          "the `{direction}` direction, the backward differences "*
                          "at ``x_{i+1}`` and ``x_i`` weighted by ``h_i`` and "*
                          "``h_{i+1}``, written into `vₕ`.",
    dispatch_alias=D̽ₕ,
    vectorial_alias=D̽ₕ,
    vectorial_dir_string="cross-weighted centered",
    vectorial_what="difference",
    vectorial_note="The second-order, non-uniform-grid counterpart of [`∇ₕ`](@ref) and "*
                   "[`∇₊ₕ`](@ref), built from [`D̽ₓ`](@ref) rather than from the one-sided "*
                   "differences.")

# ==============================================================================
# Matrix forms for the three centred families
# ==============================================================================
#
# `D̃`, `Dc` and `D̽ₕ` had grid-function forms only, so of the eight operator families
# five could be had as a matrix and three could not. That asymmetry had to be explained in
# every one of their docstrings, and it left the form layer's nodes for them with nothing
# to be checked against.
#
# Each is a diagonal scaling of unscaled difference matrices this file already builds, so
# none needs a new traversal:
#
#     D̃ = diag(2/(hᵢ + hᵢ₊₁))                  · (shift₊₁ - shift₀)
#     Dc     = diag(1/(hᵢ + hᵢ₊₁))                  · (shift₊₁ - shift₋₁)
#     D̽ₕ     = diag(hᵢ/((hᵢ+hᵢ₊₁)hᵢ₊₁))             · diff₊
#            + diag(hᵢ₊₁/((hᵢ+hᵢ₊₁)hᵢ))             · diff₋
#
# The cross-weighted one falls out of its own definition: it is D₋ at xᵢ₊₁ weighted by hᵢ
# and D₋ at xᵢ weighted by hᵢ₊₁, over their sum, and D₋(u)ᵢ₊₁ is diff₊(u)ᵢ/hᵢ₊₁ while
# D₋(u)ᵢ is diff₋(u)ᵢ/hᵢ. So the two weights above are what is left after dividing through.
#
# The weights read the mesh's cached `spacings` rather than `spacing_for_derivative`. The
# cached vector repeats the first interval in h₁ instead of zeroing it, and that repeated
# value is the one the grid-function kernels use, so reading it is what makes the matrix
# agree with them at the first point. Truncation is applied by index here instead, which
# is also what the form layer's stencils do.

# Returns `w`, as a mutating function with a single destination does, so that the builders
# below can write `_scale_rows!(matrix, _extended_weights!(cache, …))` rather than filling
# the cache on one line and reaching for it on the next.
@inline function _extended_weights!(
        w::AbstractVector, Ωₕ::AbstractMeshType, ::Val{DIFF_DIM}, weight::F
) where {F, DIFF_DIM}
    1 <= DIFF_DIM <= dim(Ωₕ) || _throw_stencil_dim_error(DIFF_DIM, dim(Ωₕ))

    dims = npoints(Ωₕ, Tuple)
    h = spacings(Ωₕ(DIFF_DIM))
    n = dims[DIFF_DIM]
    li = LinearIndices(dims)

    @inbounds for I in CartesianIndices(dims)
        w[li[I]] = weight(h, I[DIFF_DIM], n)
    end
    return w
end

# `_star_weight`/`_centered_weight` return zero wherever their stencil would need a
# neighbour the grid does not have, truncating that slice of the matrix to an empty row.
@inline _star_weight(h, i, n) = i == n ? zero(eltype(h)) : 2 / (h[i] + h[i + 1])
@inline _centered_weight(h, i, n) = (i == 1 || i == n) ? zero(eltype(h)) : inv(h[i] + h[i + 1])

# The cross-weighted pair instead falls back to the one-sided difference on whichever
# side is still on the grid at each end: `diff₊`'s row 1 is already `u_2 - u_1` and
# `diff₋`'s row `n` is already `u_n - u_{n-1}` (the unscaled matrices' own boundary
# convention, `_difference_operator`), so weighting those rows by `1/h[1]` and `1/h[n]`
# respectively -- with the other family's row zeroed there -- reproduces `D₊`/`D₋`
# exactly, matching the grid-function engine's boundary case above
# (gpena/Bramble.jl#183).
@inline _cross_forward_weight(h, i, n) = i == n ? zero(eltype(h)) :
                                         (i == 1 ? inv(h[1]) : h[i] / ((h[i] + h[i + 1]) * h[i + 1]))
@inline _cross_backward_weight(h, i, n) = i == 1 ? zero(eltype(h)) :
                                          (i == n ? inv(h[n]) : h[i + 1] / ((h[i] + h[i + 1]) * h[i]))

"""
    forward_star_difference(Ωₕ::AbstractMeshType, dim_val::Val)

The forward difference over the averaged spacing along `dim_val`, as a sparse matrix.

The forward difference scaled by the averaged spacing instead of the forward one. The last
point along the direction has no forward neighbour, so its row is empty.
"""
function forward_star_difference(Ωₕ::AbstractMeshType, dim_val::Val{DIM}) where {DIM}
    return stencil_matrix(Ωₕ, StarDiffOp{DIM}())
end

"""
    centered_difference(Ωₕ::AbstractMeshType, dim_val::Val)

The centered difference along `dim_val`, as a sparse matrix.

Reaches one point either side, so both end rows are empty and the mesh needs at least three
points along the direction.
"""
function centered_difference(Ωₕ::AbstractMeshType, dim_val::Val{DIM}) where {DIM}
    1 <= DIM <= dim(Ωₕ) || _throw_stencil_dim_error(DIM, dim(Ωₕ))
    n = npoints(Ωₕ(DIM))
    n >= 3 || _throw_centered_too_few_points(DIM, n)

    return stencil_matrix(Ωₕ, CenteredDiffOp{DIM}())
end

"""
    cross_weighted_difference(Ωₕ::AbstractMeshType, dim_val::Val)

The cross-weighted centered difference along `dim_val`, as a sparse matrix.

A three-point stencil in the interior, but neither end row is empty: with no neighbour on the
far side, row 1 agrees with [`D₊ₓ`](@ref)`(Ωₕ, dim_val)` and row `n` with
[`D₋ₓ`](@ref)`(Ωₕ, dim_val)`, each under its own diagonal weight alongside the interior
cross-weighting. The mesh still needs at least three points along the direction.
"""
function cross_weighted_difference(Ωₕ::AbstractMeshType, dim_val::Val{DIM}) where {DIM}
    1 <= DIM <= dim(Ωₕ) || _throw_stencil_dim_error(DIM, dim(Ωₕ))
    n = npoints(Ωₕ(DIM))
    n >= 3 || _throw_centered_too_few_points(DIM, n)

    return stencil_matrix(Ωₕ, CrossWeightedDiffOp{DIM}())
end

# --- Retained Kronecker oracle (gpena/Bramble.jl#185) --------------------------------- #
#
# The bodies the three functions above used to have, kept so `kronecker_operator_matrix`
# has an independent construction to check `stencil_matrix` against.

function _kron_forward_star_difference(
        Ωₕ::AbstractMeshType, dim_val::Val; vector_cache = __vector(Ωₕ)
)
    w = _extended_weights!(vector_cache, Ωₕ, dim_val, _star_weight)
    return _scale_rows!(_difference_operator(Ωₕ, Forward(), dim_val), w)
end

function _kron_centered_difference(
        Ωₕ::AbstractMeshType, dim_val::Val{DIM}; vector_cache = __vector(Ωₕ)
) where {DIM}
    n = npoints(Ωₕ(DIM))
    n >= 3 || _throw_centered_too_few_points(DIM, n)

    w = _extended_weights!(vector_cache, Ωₕ, dim_val, _centered_weight)
    return _scale_rows!(difference_shift(Ωₕ, dim_val, Val(1), Val(-1)), w)
end

function _kron_cross_weighted_difference(
        Ωₕ::AbstractMeshType, dim_val::Val{DIM}; vector_cache = __vector(Ωₕ)
) where {DIM}
    n = npoints(Ωₕ(DIM))
    n >= 3 || _throw_centered_too_few_points(DIM, n)

    forward = _scale_rows!(_difference_operator(Ωₕ, Forward(), dim_val),
        _extended_weights!(vector_cache, Ωₕ, dim_val, _cross_forward_weight))

    # the scaling above is already applied, so the cache is free to be rewritten
    backward = _scale_rows!(_difference_operator(Ωₕ, Backward(), dim_val),
        _extended_weights!(vector_cache, Ωₕ, dim_val, _cross_backward_weight))
    return forward + backward
end

# --- Kronecker oracle dispatch (gpena/Bramble.jl#185) --------------------------------- #
#
# `kronecker_operator_matrix` is declared in shift.jl; each family's dispatch method maps
# its public per-axis alias to the `_kron_*` construction kept above.
for (i, suffix) in enumerate(_BRAMBLE_var2symbol)
    for (stem, kron_fn) in (
        (:D₋, :_kron_backward_finite_difference),
        (:D₊, :_kron_forward_finite_difference),
        (:D̃, :_kron_forward_star_difference),
        (:Dc, :_kron_centered_difference),
        (:D̽, :_kron_cross_weighted_difference)
    )
        alias = Symbol(stem, suffix)
        @eval kronecker_operator_matrix(Ωₕ::AbstractMeshType, ::typeof($alias)) = $kron_fn(Ωₕ, Val($i))
    end
end

# A grid space carries its mesh, as for every other family here.
@inline forward_star_difference(Wₕ::AbstractSpaceType, dim_val::Val) = forward_star_difference(mesh(Wₕ), dim_val)
@inline centered_difference(Wₕ::AbstractSpaceType, dim_val::Val) = centered_difference(mesh(Wₕ), dim_val)
@inline cross_weighted_difference(Wₕ::AbstractSpaceType, dim_val::Val) = cross_weighted_difference(mesh(Wₕ), dim_val)

# ==============================================================================
# ==============================================================================
# The AST nodes: discrete finite difference operators for the Bramble lazy AST
# ==============================================================================
# ==============================================================================

# ==============================================================================
# Struct Definitions
# ==============================================================================

"""
    BackwardDifference{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node representing a backward finite difference operator acting in dimension `Dim`.
"""
struct BackwardDifference{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    inner_op::OpType
end

"""
    ForwardDifference{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node representing a forward finite difference operator acting in dimension `Dim`.
"""
struct ForwardDifference{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    inner_op::OpType
end

# ==============================================================================
# User-Facing API & Overloads
# ==============================================================================
#
# The two one-sided families, generated by `@node_family` (`node_family.jl`). What was
# written out here -- `grad_backward`/`grad_forward`, six subscript one-liners, the two
# gradients and their tuple methods -- is the same text for both families but for the node
# name, so it is now said once (gpena/Bramble.jl#74).
#
# `D₋(op, Val(1))` is the entry point the subscripts forward through, and it is a method of
# the same `D₋` the space layer applies to a grid function: the argument decides whether a
# difference is computed now or a node is built to compute it during assembly.

@node_family(node=BackwardDifference,
    stem=D₋,
    what="backward finite difference",
    vectorial_alias=∇ₕ,
    componentwise=true)

@node_family(node=ForwardDifference,
    stem=D₊,
    what="forward finite difference",
    vectorial_alias=∇₊ₕ,
    componentwise=true)

# ==============================================================================
# Zero-Allocation Stencil Evaluators
# ==============================================================================

# Taps and weights, consumed by the one `local_stencil` in `form/stencil_eval.jl` and --
# the point of declaring them -- by `stencil_offsets` in `form/stencil_pattern.jl`, which
# used to spell the same reach out a second time (gpena/Bramble.jl#70).
@inline _stencil_taps(::BackwardDifference) = (Val(0), Val(-1))
@inline _stencil_taps(::ForwardDifference) = (Val(1), Val(0))

@inline function _stencil_weights(
        op::BackwardDifference{D, Dim}, space, I::CartesianIndex{D}
) where {D, Dim}
    h = spacing(mesh(space), I, Dim)
    # select, not `mask / h`: a collapsed axis has h == 0 and only edge points (#622)
    c = I[Dim] == 1 ? zero(inv(h)) : inv(h)
    return (c, -c)
end

@inline function _stencil_weights(
        op::ForwardDifference{D, Dim}, space, I::CartesianIndex{D}
) where {D, Dim}
    m = mesh(space)
    h = forward_spacing(m, I, Dim)
    c = I[Dim] == npoints(m, Tuple)[Dim] ? zero(inv(h)) : inv(h)
    return (c, -c)
end

# ==============================================================================
# AST Resolution
# ==============================================================================

# `resolve_ast` for both is generated with the families above.

# ==============================================================================
# Direct integration helpers for the form AST (ast.jl)
# ==============================================================================

"""
    DifferenceNode{D, Dim}

Either one-sided difference node over a `D`-dimensional space, differencing along `Dim`.

The two carry the same parameters, so anything that reads only the *direction* off the
node is written against this alias and stays symmetric between them by construction.

Not everything can be: `inner₊` takes backward differences alone, because the staggered
weights it carries are the ones the summation-by-parts identity pairs with a backward
difference. Use this alias where the distinction genuinely does not arise.
"""
const DifferenceNode{D, Dim} = Union{BackwardDifference{D, Dim}, ForwardDifference{D, Dim}}

# ==============================================================================
# The remaining difference families
# ==============================================================================
#
# Three more differences, each a symbolic counterpart of an operator the space layer
# already provides. These nodes assemble through their stencils; the space layer's matrix
# forms of the same three operators are what those stencils are tested against.
#
# The boundary convention is the one the one-sided nodes already use: the offsets stay and
# the coefficients go to zero. A truncated point contributes nothing while the stencil
# keeps the same shape, which is what lets the assembly loop stay branch-free.
#
# Writing `h` for `spacing` (xᵢ - xᵢ₋₁) and `hf` for `forward_spacing` (xᵢ₊₁ - xᵢ), as the
# space layer's own docstrings do.

"""
    CenteredDifference{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node for the centered difference along `Dim`,

```math
Dc(u)_i = \\frac{u_{i+1} - u_{i-1}}{h_i + h_{i+1}}
```

Truncated at both ends of `Dim`, having no neighbour on one side.
"""
struct CenteredDifference{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    inner_op::OpType
end

"""
    StarDifference{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node for the forward difference over the averaged spacing along `Dim`,

```math
\\tilde{D}_{+}(u)_i = \\frac{u_{i+1} - u_i}{(h_i + h_{i+1})/2}
```

The forward difference over the *averaged* spacing rather than the forward one, which is
what makes the discrete integration by parts close. Truncated at the far end of `Dim`.
"""
struct StarDifference{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    inner_op::OpType
end

"""
    CrossWeightedDifference{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node for the cross-weighted centered difference along `Dim`,

```math
\\overset{\\times}{D}_h(u)_i = \\frac{h_i}{h_i + h_{i+1}} D_{-}(u)_{i+1}
         + \\frac{h_{i+1}}{h_i + h_{i+1}} D_{-}(u)_i
```

The same two one-sided differences the centered difference combines, weighted by the
*opposite* spacings. That swap is what makes it second order on a non-uniform grid where
`Dc` is first, and the two coincide when the spacing is constant. Neither end is truncated:
each collapses to the one-sided difference its near side still defines, `D₊` at the first
point and `D₋` at the last.
"""
struct CrossWeightedDifference{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    inner_op::OpType
end

# The three extended families, as `@node_family` calls like the one-sided pair above. Their
# prose is thinner than the space layer's: the boundary conventions and the order-of-accuracy
# comparisons that `Dcₓ` and `D̽ₓ` carry there describe arithmetic, and the arithmetic of
# these nodes is `_stencil_weights` below, which documents itself.
#
# `D̽ₕ` is the one family whose dispatch alias and tuple-valued alias are the same name
# (gpena/Bramble.jl#349: the family formerly called `Dₕ`), so `D̽ₕ(op)` gives the `D`-tuple
# and `D̽ₕ(op, Val(2))` the `y` node. They coexist by arity, which is the decision
# gpena/Bramble.jl#140 asked to be made on purpose rather than by merge.

@node_family(node=CenteredDifference,
    stem=Dc,
    what="centered difference",
    vectorial_alias=Dcₕ)

@node_family(node=StarDifference,
    stem=D̃,
    what="averaged-spacing forward difference",
    vectorial_alias=D̃ₕ)

@node_family(node=CrossWeightedDifference,
    stem=D̽,
    what="cross-weighted centered difference",
    dispatch_alias=D̽ₕ,
    vectorial_alias=D̽ₕ)

# --- Stencils --------------------------------------------------------------------- #

@inline _stencil_taps(::CenteredDifference) = (Val(1), Val(-1))
@inline _stencil_taps(::StarDifference) = (Val(1), Val(0))

@inline function _stencil_weights(
        op::CenteredDifference{D, Dim}, space, I::CartesianIndex{D}
) where {D, Dim}
    m = mesh(space)
    # no neighbour on one side at either end
    s = spacing(m, I, Dim) + forward_spacing(m, I, Dim)
    edge = I[Dim] == 1 || I[Dim] == npoints(m, Tuple)[Dim]
    c = edge ? zero(inv(s)) : inv(s)
    return (c, -c)
end

@inline function _stencil_weights(
        op::StarDifference{D, Dim}, space, I::CartesianIndex{D}
) where {D, Dim}
    m = mesh(space)
    s = spacing(m, I, Dim) + forward_spacing(m, I, Dim)
    # the averaged spacing, which is what D̃ divides by; `2 / s` as in `_star_weight`
    c = I[Dim] == npoints(m, Tuple)[Dim] ? zero(2 / s) : 2 / s
    return (c, -c)
end

# Expanding the definition over the two one-sided differences gives a three-point stencil.
# With S = h + hf,
#
#   D_h(u)_i = h/(S·hf) · u_{i+1} + (hf/(S·h) - h/(S·hf)) · u_i - hf/(S·h) · u_{i-1}
#
# which is where the two coefficients below come from: `a` is the weight of the forward
# neighbour and `b` the magnitude of the backward one.
@inline _stencil_taps(::CrossWeightedDifference) = (Val(1), Val(0), Val(-1))

@inline function _stencil_weights(
        op::CrossWeightedDifference{D, Dim}, space, I::CartesianIndex{D}
) where {D, Dim}
    m = mesh(space)

    if I[Dim] == 1
        # No point behind the first one: D̽ₕ has no truncated-boundary convention of its
        # own, so it collapses to the one-sided difference the near side still gives,
        # D₊(u)_1 = (u_2 - u_1)/h_1 (gpena/Bramble.jl#183). A one-point axis has h_1 == 0
        # and no neighbour at all, so the weights are zero there (#622).
        h = spacing(m, I, Dim)
        a = npoints(m, Tuple)[Dim] == 1 ? zero(inv(h)) : inv(h)
        return (a, -a, zero(a))
    elseif I[Dim] == npoints(m, Tuple)[Dim]
        # No point past the last one: collapses to D₋(u)_n = (u_n - u_{n-1})/h_n.
        b = inv(spacing(m, I, Dim))
        return (zero(b), b, -b)
    else
        h = spacing(m, I, Dim)
        hf = forward_spacing(m, I, Dim)
        total = h + hf

        a = h / (total * hf)
        b = hf / (total * h)
        return (a, b - a, -b)
    end
end

# --- Traits ----------------------------------------------------------------------- #

"""
    ExtendedDifferenceNode{D, Dim}

The three difference nodes that are neither one-sided nor a jump, differencing along `Dim`.
Grouped so that everything reading only the direction off a node covers all of them at
once, as `DifferenceNode` does for the one-sided pair.
"""
const ExtendedDifferenceNode{D, Dim} = Union{
    CenteredDifference{D, Dim}, StarDifference{D, Dim}, CrossWeightedDifference{D, Dim}
}

# ==============================================================================
# Composite trial/test functions: ∇ₕ, ∇₊ₕ, εₕ, divₕ over several leaves at once
# ==============================================================================
#
# `form(Wₕ, Vₕ, f)` (bilinear.jl) already hands `f` the space's own `TrialFunction{D,N}`/
# `TestFunction{D,N}`, and that object is already tuple-like: `u(i)` and `components(u)`
# (form/component.jl) address its `N` immediate subspaces, and
# `TrialFunction`/`TestFunction` already answer `iterate`/`getindex`/`length`. What is
# missing is passing `u` itself, unindexed, straight to a vectorial operator. `∇ₕ(u)` on a
# composite `u` would hit the generic `∇ₕ(op::LazyOp{D})` method above, differencing the
# whole composite as if it were one scalar function, rather than each of its `N` components
# in turn. The two methods below intercept `TrialFunction{D,N}`/
# `TestFunction{D,N}` ahead of that generic method -- a concrete struct is always more
# specific than the abstract `LazyOp{D}` it is a subtype of, whatever `N` is -- and, only
# when `N` is a genuine composite leaf count, forward to `∇ₕ`'s/`∇₊ₕ`'s own `componentwise`
# tuple method (`@node_family`'s `vectorial_alias`, `node_family.jl`) over `components(u)`,
# an `N`-tuple of `D`-tuples, the gradient tensor. A scalar space (`N === nothing` or
# `N == 1`) has to keep exactly what the generic method already gave it -- reproduced in
# `_∇ₕ_noncomposite`/`_∇₊ₕ_noncomposite` below, rather than falling through to it, since our
# own method is the more specific one for every `N` and would otherwise shadow the scalar
# case too.

@inline _∇ₕ_noncomposite(op::LazyOp{1}) = D₋ₓ(op)
@inline _∇ₕ_noncomposite(op::LazyOp{D}) where {D} = ntuple(dim -> D₋(op, Val(dim)), Val(D))

"""
    ∇ₕ(u::Union{TrialFunction, TestFunction})

The backward gradient tensor of a composite trial or test function `u`: an `N`-tuple (one
per component of `u`) of `D`-tuples (one per spatial direction), `∇ₕ(u)[c][d]` the backward
difference of component `c` along direction `d`.

Equivalent to `∇ₕ(components(u))`, spelled without the explicit `components` call so `u`
alone can be passed to a vectorial operator. On a scalar space (one component) this is the
same `D`-tuple `∇ₕ` already returns for any operator.
"""
@inline function ∇ₕ(op::Union{TrialFunction{D, N}, TestFunction{D, N}}) where {D, N}
    (N isa Integer && N > 1) && return map(∇ₕ, components(op))
    return _∇ₕ_noncomposite(op)
end

@inline _∇₊ₕ_noncomposite(op::LazyOp{1}) = D₊ₓ(op)
@inline _∇₊ₕ_noncomposite(op::LazyOp{D}) where {D} = ntuple(dim -> D₊(op, Val(dim)), Val(D))

"""
    ∇₊ₕ(u::Union{TrialFunction, TestFunction})

The forward twin of [`∇ₕ`](@ref)`(u::Union{TrialFunction,TestFunction})`.
"""
@inline function ∇₊ₕ(op::Union{TrialFunction{D, N}, TestFunction{D, N}}) where {D, N}
    (N isa Integer && N > 1) && return map(∇₊ₕ, components(op))
    return _∇₊ₕ_noncomposite(op)
end

# --- εₕ, divₕ: symmetric strain and staggered divergence over composite functions ---- #
#
# Placement settled by gpena/Bramble.jl#234 (issue comment, 2026-09-18): ε_ii on the face
# centre normal to axis `i` (a bare backward difference, no averaging), ε_ij for `i != j` on
# the edge centre the pair `{i,j}` shares (each side's cross difference averaged once, onto
# that edge), and div u on the cell centre every axis shares. No collocation choice is
# exposed: these are the only placements `εₕ`/`divₕ` build.
#
# Both are builder-only: `εₕ(u)`/`divₕ(u)` return a small struct holding the `D`-many (or
# `D×D`-many) `LazyOp`s the formula above names, and the *only* thing that consumes it is
# the `inner₊` method beside each struct, which expands immediately -- inside the closure
# `form` resolves, before `simplify_ast` ever runs -- into the sum of ordinary single-block
# `BilinearProduct`s a user would write by hand (`test/form/vector_calculus.jl`'s hand-
# expanded reference is exactly that sum). Nothing here is a `LazyOp`, so nothing here ever
# reaches assembly, `local_stencil` or `block_of` directly: the architecture stays "every
# `LazyOp` is scalar-valued, expansion happens at the builder" (gpena/Bramble.jl#234).
#
# `εₕ`/`divₕ` differ from `operators/vector_calculus.jl`'s *runtime* `divₕ`
# (gpena/Bramble.jl#158) the way every symbolic/numeric pair in this package can (CONTEXT.md):
# the runtime `divₕ` sums raw backward differences with no cross-axis averaging, read at the
# mesh's own nodes; placing every one of the `D` terms here at the same shared quadrature
# point first is what makes `inner₊(divₕ(u), divₕ(v))` well posed. The name is shared on
# purpose -- both are "the backward-difference divergence" -- and the two are never mixed in
# one expression, so nothing has to choose between them.

@inline function _vc_bwd(op, d::Int)
    d == 1 && return D₋ₓ(op)
    d == 2 && return D₋ᵧ(op)
    return D₋₂(op)
end

@inline function _vc_avg(op, d::Int)
    d == 1 && return Mₓ(op)
    d == 2 && return Mᵧ(op)
    return M₂(op)
end

@inline _stagger_set(i::Int, j::Int) = i == j ? (i,) : (min(i, j), max(i, j))

# The additive pieces of one entry εᵢⱼ(u) of the strain tensor, kept apart rather than
# summed into one node. `ε_ii` is a single piece; `ε_ij` (`i != j`) is the two half-averaged
# cross differences the definition adds together. Keeping them apart is what lets
# `inner₊(::_StrainTensor, ::_StrainTensor)` below expand their cross product into clean
# single-component terms at the builder: a *summed* `ε_ij(u)` mixes trial (or test)
# components `i` and `j` in one subtree, and `simplify_ast`'s component-distribution rule
# only compares one node's class against its sibling's -- once a scale wraps the whole
# tensor's sum and two already-mixing siblings meet, both report the same "mixed" marker and
# the rule cannot tell them apart, leaving the scale over a term `block_of` cannot route.
# Building the four cross terms directly, the way a user would expand `(a+b)*(c+d)` by hand,
# means no node here ever mixes components in the first place.
@inline function _strain_pieces(u, i::Int, j::Int)
    i == j && return (_vc_bwd(u(i), i),)
    return (0.5 * _vc_avg(_vc_bwd(u(i), j), i), 0.5 * _vc_avg(_vc_bwd(u(j), i), j))
end

# One term of the divergence: D_{-i}(u_i), averaged onto the shared cell centre by every
# other axis. `foldl` over a runtime-filtered range, not `ntuple`: this runs once per
# `εₕ`/`divₕ` call (builder time, not per grid point), the same footing
# `docs/src/examples/elasticity_3d.jl`'s `divₜ` already stands on.
@inline _div_term(u, i::Int, ::Val{D}) where {D} = foldl(
    (op, d) -> _vc_avg(op, d), Iterators.filter(!=(i), 1:D); init = _vc_bwd(u(i), i)
)

"""
    Bramble._StrainTensor{D}

Builder-only container for [`εₕ`](@ref)'s `D × D` entries, each held as its additive pieces
(`Bramble._strain_pieces`) rather than their sum. Never reaches assembly or an AST walker:
[`inner₊`](@ref) on two of these expands immediately into the sum of single-block products a
user would write out by hand.
"""
struct _StrainTensor{D, T}
    entries::T
end

"""
    εₕ(u) -> Bramble._StrainTensor

The symbolic small-strain tensor \$\\varepsilon(u) = \\tfrac12(\\nabla u + \\nabla u^{T})\$ of
a composite trial or test function `u` with one component per spatial dimension.

`u` is whatever `form`'s trial or test argument already is: `u(i)` addresses its `i`-th
component exactly as it does everywhere else in the form layer (gpena/Bramble.jl#74). Every
entry is placed the way the discrete energy form needs it, with no collocation choice
exposed -- see this section's header comment for the placement.

The only supported use is `inner₊(εₕ(u), εₕ(v))`, which expands to
``\\sum_{i,j} (\\varepsilon_{ij}(u), \\varepsilon_{ij}(v))_{S_{ij}}`` with
``S_{ii} = \\{i\\}`` and ``S_{ij} = \\{i,j\\}`` for `i != j`, through
[`inner₊`](@ref)`(left, right, Val(S))`.

See also: [`divₕ`](@ref), [`∇ₕ`](@ref).
"""
function εₕ(u::LazyOp{D}) where {D}
    entries = ntuple(Val(D)) do i
        ntuple(Val(D)) do j
            _strain_pieces(u, i, j)
        end
    end
    return _StrainTensor{D, typeof(entries)}(entries)
end

# The cross product of two entries' additive pieces, each pair expanded into its own
# `inner₊(..., Val(S))` term -- the builder-time equivalent of multiplying out `(a+b)*(c+d)`
# by hand, so that every term reaching `inner₊` already carries one component per side and
# `simplify_ast` is never asked to untangle a mixing sum (see `_strain_pieces` above).
@inline _cross_terms(lu::Tuple, lv::Tuple, ::Val{S}) where {S} = _flatten_tuples(
    map(a -> map(b -> inner₊(a, b, Val(S)), lv), lu)
)

@inline function inner₊(left::_StrainTensor{D}, right::_StrainTensor{D}) where {D}
    terms = _flatten_tuples(
        ntuple(Val(D)) do i
        _flatten_tuples(
            ntuple(Val(D)) do j
            _cross_terms(
                left.entries[i][j], right.entries[i][j], Val(_stagger_set(i, j))
            )
        end
        )
    end
    )
    return foldl(+, terms)
end

"""
    Bramble._DivergenceTerms{D}

Builder-only container for [`divₕ`](@ref)'s `D` per-component terms: the symbolic twin of
[`Bramble._StrainTensor`](@ref) for the divergence, and with the same lifetime -- consumed
immediately by [`inner₊`](@ref), never part of an assembled AST.
"""
struct _DivergenceTerms{D, T}
    terms::T
end

"""
    divₕ(u) -> Bramble._DivergenceTerms

The symbolic staggered divergence of a composite trial or test function `u` with one
component per spatial dimension, placed at the cell centre every axis shares: term `i` is
``D_{-,i}(u_i)`` averaged onto that centre by every axis other than `i`.

Shares its name with the *runtime* [`divₕ`](@ref) over grid functions
(`operators/vector_calculus.jl`, gpena/Bramble.jl#158); see this section's header
comment for how and why the two differ.

The only supported use is `inner₊(divₕ(u), divₕ(v))`, which expands to
``\\sum_{i,j} (\\mathrm{div}_h^{(i)}(u), \\mathrm{div}_h^{(j)}(v))_{\\{1,\\dots,D\\}}``
through [`inner₊`](@ref)`(left, right, Val(S))`.

See also: [`εₕ`](@ref).
"""
function divₕ(u::LazyOp{D}) where {D}
    terms = ntuple(i -> _div_term(u, i, Val(D)), Val(D))
    return _DivergenceTerms{D, typeof(terms)}(terms)
end

@inline function inner₊(left::_DivergenceTerms{D}, right::_DivergenceTerms{D}) where {D}
    S = Val(ntuple(identity, Val(D)))
    return foldl(
        +, ntuple(Val(D)) do i
            foldl(+, ntuple(Val(D)) do j
                inner₊(left.terms[i], right.terms[j], S)
            end)
        end
    )
end

# --- divcₕ, εcₕ: centered divergence and strain over composite functions ---------------- #
#
# The centered difference is collocated: every `Dc` term sits on the grid point itself, so
# neither builder averages or staggers anything. `divcₕ(u)` is therefore an ordinary scalar
# `LazyOp` sum usable anywhere a test or trial expression is. `εcₕ(u)` keeps its entries'
# additive pieces apart, for the reason `_strain_pieces` above gives, and is consumed only by
# the `innerₕ` method beside it. Both methods take `LazyOp{D}`, which is more specific than
# the runtime `divcₕ(uₕ)`/`εcₕ(uₕ)` over grid functions (untyped) and disjoint from any grid
# function, so the symbolic and numeric families never meet.

@inline function _vc_centered(op, d::Int)
    d == 1 && return Dcₓ(op)
    d == 2 && return Dcᵧ(op)
    return Dc₂(op)
end

"""
    divcₕ(u::LazyOp{D}) -> LazyOp

The symbolic centered divergence of a trial or test function `u` with one component per
spatial dimension,

```math
\\textrm{div}_{c,h}(u) = \\sum_{i=1}^{D} \\textrm{Dc}_{x_i}(u_i).
```

Every term is collocated at the grid point, so the result is a plain operator sum usable
wherever an operator is, e.g. `innerₕ(p, divcₕ(v))`. Shares its name with the runtime
[`divcₕ`](@ref) over grid functions (`operators/vector_calculus.jl`).

See also: [`εcₕ`](@ref), [`divₕ`](@ref).
"""
function divcₕ(u::LazyOp{D}) where {D}
    return foldl(+, ntuple(i -> _vc_centered(u(i), i), Val(D)))
end

"""
    Bramble._CenteredStrainTensor{D}

Builder-only container for [`εcₕ`](@ref)'s and [`ε̽ₕ`](@ref)'s `D × D` entries, each held
as its additive pieces. Consumed only by [`innerₕ`](@ref) on two of these, never part of an
assembled AST.
"""
struct _CenteredStrainTensor{D, T}
    entries::T
end

# `diff(op, d)` is the family's difference along `d`: `_vc_centered` for `εcₕ`,
# `_vc_cross_weighted` for `ε̽ₕ`.
@inline function _centered_strain_pieces(diff::F, u, i::Int, j::Int) where {F}
    i == j && return (diff(u(i), i),)
    return ((1 // 2) * diff(u(i), j), (1 // 2) * diff(u(j), i))
end

"""
    εcₕ(u::LazyOp{D}) -> Bramble._CenteredStrainTensor

The symbolic centered small-strain tensor of a composite trial or test function `u`,

```math
\\varepsilon^{ii}_{c,h}(u) = \\textrm{Dc}_{x_i}(u_i), \\qquad
\\varepsilon^{ij}_{c,h}(u) = \\tfrac{1}{2}\\left(\\textrm{Dc}_{x_j}(u_i)
    + \\textrm{Dc}_{x_i}(u_j)\\right), \\quad i \\neq j.
```

The only supported use is `innerₕ(εcₕ(u), εcₕ(v))`, which expands to
``\\sum_{i,j} (\\varepsilon^{ij}_{c,h}(u), \\varepsilon^{ij}_{c,h}(v))_h``. Shares its name
with the runtime [`εcₕ`](@ref) over grid functions.

See also: [`divcₕ`](@ref), [`εₕ`](@ref).
"""
function εcₕ(u::LazyOp{D}) where {D}
    entries = ntuple(Val(D)) do i
        ntuple(j -> _centered_strain_pieces(_vc_centered, u, i, j), Val(D))
    end
    return _CenteredStrainTensor{D, typeof(entries)}(entries)
end

# The upper triangle only: `ε^{ij} = ε^{ji}`, so each off-diagonal pair enters once, doubled
# (27 terms in 3D become 15). The diagonal's `1 // 1` is not a no-op for the compiler. A
# non-`Integer` scale is kept (`_wrap_scale`), so a diagonal `Dcₓ(u(1)) Dcₓ(v(1))` term shares
# its type with the off-diagonal `Dcₓ(u(2)) Dcₓ(v(2))` one, and 12 distinct term types become
# 9. The scales here and in `_centered_strain_pieces` are `Rational`, not `Float64`, since a rational
# times a `Float32` weight stays `Float32`, so the form keeps the mesh's element type.
@inline function _centered_strain_products(left, right, i::Int, j::Int)
    products = _flatten_tuples(
        map(a -> map(b -> innerₕ(a, b), right.entries[i][j]), left.entries[i][j])
    )
    return map(p -> (i == j ? 1 // 1 : 2 // 1) * p, products)
end

@inline function innerₕ(left::_CenteredStrainTensor{D}, right::_CenteredStrainTensor{D}) where {D}
    terms = _flatten_tuples(
        ntuple(Val(D)) do i
        _flatten_tuples(
            ntuple(j -> j < i ? () : _centered_strain_products(left, right, i, j), Val(D))
        )
    end
    )
    return foldl(+, terms)
end

# --- div̽ₕ, ε̽ₕ: cross-weighted divergence and strain over composite functions (#349) ------ #
#
# The cross-weighted difference is co-located too, so these are `divcₕ`/`εcₕ` with `D̽` in
# place of `Dc`: an ordinary operator sum, and a `_CenteredStrainTensor` consumed by the same
# `innerₕ` method. No form curl, matching the centered family. The same `LazyOp{D}` versus
# untyped split keeps them apart from the runtime `div̽ₕ(uₕ)`/`ε̽ₕ(uₕ)`.

@inline function _vc_cross_weighted(op, d::Int)
    d == 1 && return D̽ₕ(op, Val(1))
    d == 2 && return D̽ₕ(op, Val(2))
    return D̽ₕ(op, Val(3))
end

"""
    div̽ₕ(u::LazyOp{D}) -> LazyOp

The symbolic cross-weighted divergence of a trial or test function `u` with one component
per spatial dimension,

```math
\\overset{\\times}{\\textrm{div}}_h(u) = \\sum_{i=1}^{D} \\overset{\\times}{\\textrm{D}}_{x_i}(u_i).
```

Every term is co-located at the grid point, so the result is a plain operator sum usable
wherever an operator is, e.g. `innerₕ(p, div̽ₕ(v))`. Shares its name with the runtime
[`div̽ₕ`](@ref) over grid functions (`operators/vector_calculus.jl`).

See also: [`ε̽ₕ`](@ref), [`divcₕ`](@ref).
"""
function div̽ₕ(u::LazyOp{D}) where {D}
    return foldl(+, ntuple(i -> D̽ₕ(u(i), Val(i)), Val(D)))
end

"""
    ε̽ₕ(u::LazyOp{D}) -> Bramble._CenteredStrainTensor

The symbolic cross-weighted small-strain tensor of a composite trial or test function `u`,

```math
\\overset{\\times}{\\varepsilon}^{ii}_h(u) = \\overset{\\times}{\\textrm{D}}_{x_i}(u_i), \\qquad
\\overset{\\times}{\\varepsilon}^{ij}_h(u) = \\tfrac{1}{2}\\left(
    \\overset{\\times}{\\textrm{D}}_{x_j}(u_i) + \\overset{\\times}{\\textrm{D}}_{x_i}(u_j)\\right),
    \\quad i \\neq j.
```

The only supported use is `innerₕ(ε̽ₕ(u), ε̽ₕ(v))`, which expands to
``\\sum_{i,j} (\\overset{\\times}{\\varepsilon}^{ij}_h(u),
\\overset{\\times}{\\varepsilon}^{ij}_h(v))_h``. Shares its name with the runtime
[`ε̽ₕ`](@ref) over grid functions.

See also: [`div̽ₕ`](@ref), [`εcₕ`](@ref).
"""
function ε̽ₕ(u::LazyOp{D}) where {D}
    entries = ntuple(Val(D)) do i
        ntuple(j -> _centered_strain_pieces(_vc_cross_weighted, u, i, j), Val(D))
    end
    return _CenteredStrainTensor{D, typeof(entries)}(entries)
end

# ==============================================================================
# Expression rendering (gpena/Bramble.jl#274)
# ==============================================================================

expression(op::BackwardDifference{D, Dim}) where {D, Dim} = "D₋$(_BRAMBLE_var2symbol[Dim])($(expression(op.inner_op)))"
expression(op::ForwardDifference{D, Dim}) where {D, Dim} = "D₊$(_BRAMBLE_var2symbol[Dim])($(expression(op.inner_op)))"
expression(op::CenteredDifference{D, Dim}) where {D, Dim} = "Dc$(_BRAMBLE_var2symbol[Dim])($(expression(op.inner_op)))"
expression(op::StarDifference{D, Dim}) where {D, Dim} = "D̃$(_BRAMBLE_var2symbol[Dim])($(expression(op.inner_op)))"
function expression(op::CrossWeightedDifference{D, Dim}) where {D, Dim}
    "D̽$(_BRAMBLE_var2symbol[Dim])($(expression(op.inner_op)))"
end
