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

# The device kernels in `ext/BrambleKernelAbstractionsExt.jl` (S2.4 of
# .agents/plans/metal-and-apple-silicon-acceleration.md) need, per grid point, exactly the
# boundary test `_stencil_ranges` (operators/stencil.jl) already encodes as index ranges: a
# forward stencil has no neighbour at the last point along the direction, a backward one at
# the first. Read here as a single index comparison instead of re-deriving a
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

# The device kernels below (S2.4 of .agents/plans/metal-and-apple-silicon-acceleration.md)
# take `h` as a plain top-level array or `nothing`, never as a wrapper struct: a struct
# nesting a device array fails kernel compilation even as a top-level kernel argument
# (`ext/BrambleKernelAbstractionsExt.jl`'s module comment explains why, gpena/Bramble.jl#94,
# #174). `StarSpacings` is exactly such a wrapper, so its lazy averaging is materialized
# into a plain vector once, with the same two-array bulk arithmetic S2.2's device kernels
# use for the mesh's own O(n) setup, before any difference kernel launches. Every other
# shape `_apply_spaced!` ever derives -- `nothing`, or a plain vector/view of cached
# spacings -- is already kernel-safe and passes through unchanged.
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
    sp = space(uₕ)
    if execution_policy(sp) isa GpuPolicy
        dev = ka_device(backend(sp))
        _launch_stencil_engine!(
            vₕ.data, uₕ.data, _resolve_device_spacing(h), _grid_dims(uₕ), dir, dim_val, dev
        )
    else
        _difference_engine!(vₕ.data, uₕ.data, h, _grid_dims(uₕ), dir, dim_val)
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
# Device kernel launch stubs (gpena/Bramble.jl#94, #174, S2.4 of
# .agents/plans/metal-and-apple-silicon-acceleration.md)
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
# fallback idiom (`operators/restriction.jl`, `src/mesh/mesh1d.jl`): the extension's methods
# are typed on `AbstractVector`/`Tuple`, and a fallback with the same signature would
# overwrite them instead of adding a genuinely more specific dispatch.
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
