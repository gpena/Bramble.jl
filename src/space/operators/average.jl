###############################################################
#                                                             #
#           Implementation of the average operators           #
#                                                             #
###############################################################

#=
# average.jl

This file implements averaging operators for staggered grid computations.

## Mathematical formulation

For a function uₕ on a grid, the average operator ⟨·⟩ computes the mean value between 
adjacent grid points:

**Forward average** (at point xᵢ):
    ⟨u⟩ᵢᶠ = (uᵢ + uᵢ₊₁) / 2

**Backward average** (at point xᵢ):
    ⟨u⟩ᵢᵇ = (uᵢ + uᵢ₋₁) / 2

At boundary points where no neighbor exists:
    ⟨u⟩ᵢ = uᵢ / 2

## Use cases

Typical uses:
1. Staggered grids: transfer variables between cell centers and faces
2. Discontinuous Galerkin: compute interface values
3. Finite difference: approximate derivatives at intermediate points
4. Conservative schemes: maintain flux conservation

## Example

```julia
# Average velocity from cell centers to faces
u_face = ⟨u⟩ᶠ(Vₕ, dim)  # Forward average in dimension dim

# Average pressure from faces to centers  
p_center = ⟨p⟩ᵇ(Pₕ, dim)  # Backward average in dimension dim
```

## Implementation details

- Uses `@simd` for vectorization
- Separate loops for interior (2-point average) and boundary (1-point)
- Direction (Forward/Backward) controlled via trait dispatch
- Boundary handling: halve the value (maintains consistency with operator algebra)

See also: [`_compute_average`](@ref), [`add_half_shift`](@ref), [`Forward`](@ref), [`Backward`](@ref)
=#

# Both directions average a point with its neighbour, so one method covers them. The
# kernels take `(cur, other)` in that order, matching _compute_difference, and the
# engine below hands them the pair without knowing which way the stencil runs.
#
# Dividing by 2 rather than multiplying by 0.5 keeps the element type: a Float32 grid
# would otherwise be promoted through Float64 on every point.
@inline @propagate_inbounds _compute_average(::GridDirection, ::Val{false}, cur, other) = (cur + other) / 2

# The one boundary slice has no neighbour, so the average is truncated to zero there.
# `zero(cur)` rather than a literal keeps the element type of the grid.
@inline @propagate_inbounds _compute_average(::GridDirection, ::Val{true}, cur) = zero(cur)

# The traversal is shared with the difference engine; see _stencil_ranges in
# operators/stencil.jl.
function _average_engine!(
        out, in_ref, dims::NTuple{D, Int}, dir::GridDirection, ::Val{DIM}
) where {D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    interior, boundary = _stencil_ranges(axes(li), Val(DIM), dir)

    @inbounds @simd for I in CartesianIndices(interior)
        idx = li[I]
        out[idx] = _compute_average(
            dir, Val(false), in_ref[idx], in_ref[li[_neighbour(dir, I, step)]]
        )
    end

    @inbounds @simd for I in CartesianIndices(boundary)
        idx = li[I]
        out[idx] = _compute_average(dir, Val(true), in_ref[idx])
    end

    return nothing
end

#------------------------------------------------------------------------------------------#
# Device kernel launch stub (gpena/Bramble.jl#94, #174, S2.4 of
# .agents/plans/metal-and-apple-silicon-acceleration.md)
#
# `_apply_averaged!` reaches this once `execution_policy` names a `GpuPolicy`, instead of
# `_average_engine!`'s scalar-indexing CPU sweep above. `ext/BrambleKernelAbstractionsExt.jl`
# fills it in with a `KernelAbstractions.@kernel` that calls the very `_compute_average`
# methods above, once per grid point -- see `operators/difference.jl`'s matching stub for
# why that keeps the device and host answers identical by construction. Without the
# extension loaded, the launcher throws a named diagnostic instead of failing on a scalar
# index.
#------------------------------------------------------------------------------------------#

@noinline function _throw_no_ka_average_kernel(fname::String)
    return error(
        "$fname requires KernelAbstractions.jl to apply an average operator to a " *
        "device-backed VectorElement. Add `using KernelAbstractions` (and the package " *
        "providing this backend's device, e.g. `using Metal`) before calling it on this " *
        "backend.",
    )
end

"""
    _launch_average_engine!(out::AbstractVector, in_ref, dims::Tuple, dir::GridDirection, dim_val::Val, dev) -> Nothing

Fills `out` with the two-point average (`_compute_average`) of `in_ref` along the axis
`dim_val`, in direction `dir` (`Forward`/`Backward`), truncating the one boundary slice with
no neighbour to zero, via a `KernelAbstractions.@kernel` launch on `dev`, filled by
`ext/BrambleKernelAbstractionsExt.jl`. The device counterpart of `_average_engine!`'s CPU
sweep above.

# Throws
- `ErrorException`: no `KernelAbstractions` extension is loaded, so there is no device
  kernel to reach (`_throw_no_ka_average_kernel`).
"""
_launch_average_engine!(out, in_ref, dims, dir, dim_val, dev) = _throw_no_ka_average_kernel("_launch_average_engine!")

# Shared by every averaging direction (gpena/Bramble.jl#44): the alias check and the engine
# call, once, rather than once per direction and again inside each one's composite closure.
@inline function _apply_averaged!(
        vₕ::VectorElement{<:ScalarGridSpace},
        uₕ::VectorElement{<:ScalarGridSpace},
        dir::GridDirection,
        dim_val::Val
)
    _check_no_alias(vₕ, uₕ)
    sp = space(uₕ)
    if execution_policy(sp) isa GpuPolicy
        dev = ka_device(backend(sp))
        _launch_average_engine!(vₕ.data, uₕ.data, _grid_dims(uₕ), dir, dim_val, dev)
    else
        _average_engine!(execution_policy(sp), vₕ.data, uₕ.data, _grid_dims(uₕ), dir, dim_val)
    end
    return vₕ
end

# A composite grid function is averaged one component at a time. Recursing into the scalar
# method above re-checks aliasing and re-derives `_grid_dims` per leaf, same as the
# difference operators' `_apply_spaced!` (operators/difference.jl).
@inline function _apply_averaged!(
        vₕ::VectorElement{<:CompositeGridSpace},
        uₕ::VectorElement{<:CompositeGridSpace},
        dir::GridDirection,
        dim_val::Val
)
    _apply_componentwise!((v, u) -> _apply_averaged!(v, u, dir, dim_val), vₕ, uₕ)
    return vₕ
end

#------------------------------------------------------------------------------------------#
# Centered average (gpena/Bramble.jl#287)
#
# `(u(i-1) + 2u(i) + u(i+1))/4` along one direction, zero on both end slices of that
# direction, the truncation `Dc` uses. It reads both neighbours, so it traverses with
# `_centered_stencil_ranges` (operators/difference.jl) rather than `_stencil_ranges`.
#------------------------------------------------------------------------------------------#

# Divided by 4 rather than multiplied by 0.25, for the element-type reason given at
# `_compute_average`.
@inline @propagate_inbounds _compute_average(::Centered, ::Val{false}, back, cur, fwd) = (back + 2 * cur + fwd) / 4

function _centered_average_engine!(out, in_ref, dims::NTuple{D, Int}, ::Val{DIM}) where {D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    interior, lo, hi = _centered_stencil_ranges(axes(li), Val(DIM))

    @inbounds @simd for I in CartesianIndices(interior)
        idx = li[I]
        out[idx] = _compute_average(
            Centered(), Val(false), in_ref[li[I - step]], in_ref[idx], in_ref[li[I + step]]
        )
    end

    @inbounds for bnd in (lo, hi)
        @simd for I in CartesianIndices(bnd)
            idx = li[I]
            out[idx] = _compute_average(Centered(), Val(true), in_ref[idx])
        end
    end

    return nothing
end

#------------------------------------------------------------------------------------------#
# Policy-dispatched CPU average engines (gpena/Bramble.jl#356)
#
# The same split `_difference_engine!`'s policy methods make (operators/difference.jl):
# `CpuSerial` runs the engines above unchanged, `CpuThreaded` runs one band of the grid's
# last axis per thread, and `CpuPolyester` reaches the `_batch_…` hooks
# `BramblePolyesterExt` fills.
#------------------------------------------------------------------------------------------#

"""
    _average_band!(out, in_ref, dims::NTuple{D, Int}, dir::GridDirection, dim_val::Val, nbands::Int, b::Int) -> Nothing

Runs `_average_engine!`'s loops for `dir` on the `b`-th of `nbands` slabs of the grid cut
along its last axis, as [`_difference_band!`](@ref) does for the difference.
"""
@inline function _average_band!(
        out, in_ref, dims::NTuple{D, Int}, dir::GridDirection, ::Val{DIM}, nbands::Int, b::Int
) where {D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    band = _band_range(axes(li, D), nbands, b)
    interior, boundary = _stencil_ranges(axes(li), Val(DIM), dir)
    interior, boundary = _band_slab(interior, band), _band_slab(boundary, band)

    @inbounds @simd for I in CartesianIndices(interior)
        idx = li[I]
        out[idx] = _compute_average(
            dir, Val(false), in_ref[idx], in_ref[li[_neighbour(dir, I, step)]]
        )
    end

    @inbounds @simd for I in CartesianIndices(boundary)
        idx = li[I]
        out[idx] = _compute_average(dir, Val(true), in_ref[idx])
    end

    return nothing
end

"""
    _centered_average_band!(out, in_ref, dims::NTuple{D, Int}, dim_val::Val, nbands::Int, b::Int) -> Nothing

Runs `_centered_average_engine!`'s loops on the `b`-th of `nbands` slabs of the grid cut
along its last axis, as [`_difference_band!`](@ref) does for the difference.
"""
@inline function _centered_average_band!(
        out, in_ref, dims::NTuple{D, Int}, ::Val{DIM}, nbands::Int, b::Int
) where {D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    band = _band_range(axes(li, D), nbands, b)
    interior, lo, hi = _centered_stencil_ranges(axes(li), Val(DIM))
    interior, lo, hi = _band_slab(interior, band), _band_slab(lo, band), _band_slab(hi, band)

    @inbounds @simd for I in CartesianIndices(interior)
        idx = li[I]
        out[idx] = _compute_average(
            Centered(), Val(false), in_ref[li[I - step]], in_ref[idx], in_ref[li[I + step]]
        )
    end

    @inbounds for bnd in (lo, hi)
        @simd for I in CartesianIndices(bnd)
            idx = li[I]
            out[idx] = _compute_average(Centered(), Val(true), in_ref[idx])
        end
    end

    return nothing
end

"""
    _threaded_average_engine!(out, in_ref, dims::Tuple, dir::GridDirection, dim_val::Val) -> Nothing

[`CpuThreaded`](@ref)'s `_average_engine!`: one [`_average_band!`](@ref) per thread under
`Threads.@threads :static`, isolated so the closure is never built on a serial path.
"""
@noinline function _threaded_average_engine!(
        out, in_ref, dims::NTuple{D, Int}, dir::GridDirection, dim_val::Val
) where {D}
    nbands = Threads.nthreads()
    Threads.@threads :static for b in 1:nbands
        _average_band!(out, in_ref, dims, dir, dim_val, nbands, b)
    end
    return nothing
end

"""
    _threaded_centered_average_engine!(out, in_ref, dims::Tuple, dim_val::Val) -> Nothing

[`CpuThreaded`](@ref)'s `_centered_average_engine!`: one [`_centered_average_band!`](@ref)
per thread under `Threads.@threads :static`.
"""
@noinline function _threaded_centered_average_engine!(
        out, in_ref, dims::NTuple{D, Int}, dim_val::Val
) where {D}
    nbands = Threads.nthreads()
    Threads.@threads :static for b in 1:nbands
        _centered_average_band!(out, in_ref, dims, dim_val, nbands, b)
    end
    return nothing
end

"""
    _batch_average_engine!(out, in_ref, dims::Tuple, dir::GridDirection, dim_val::Val) -> Nothing

[`CpuPolyester`](@ref)'s `_average_engine!`, filled by `BramblePolyesterExt` (one
[`_average_band!`](@ref) per band). The only `src/` method errors naming Polyester.
"""
@noinline function _batch_average_engine!(out, in_ref, dims, dir, dim_val)
    return _throw_cpubatch_without_polyester(:_batch_average_engine!)
end

"""
    _batch_centered_average_engine!(out, in_ref, dims::Tuple, dim_val::Val) -> Nothing

[`CpuPolyester`](@ref)'s `_centered_average_engine!`, filled by `BramblePolyesterExt` (one
[`_centered_average_band!`](@ref) per band). The only `src/` method errors naming Polyester.
"""
@noinline function _batch_centered_average_engine!(out, in_ref, dims, dim_val)
    return _throw_cpubatch_without_polyester(:_batch_centered_average_engine!)
end

@inline _average_engine!(::CpuSerial, out, in_ref, dims, dir, dim_val) = _average_engine!(
    out, in_ref, dims, dir, dim_val)
@inline _average_engine!(::CpuThreaded, out, in_ref, dims, dir, dim_val) = _threaded_average_engine!(
    out, in_ref, dims, dir, dim_val)
@noinline _average_engine!(::CpuPolyester, out, in_ref, dims, dir, dim_val) = _batch_average_engine!(
    out, in_ref, dims, dir, dim_val)

@inline _centered_average_engine!(::CpuSerial, out, in_ref, dims, dim_val) = _centered_average_engine!(out, in_ref, dims, dim_val)
@inline _centered_average_engine!(::CpuThreaded, out, in_ref, dims, dim_val) = _threaded_centered_average_engine!(
    out, in_ref, dims, dim_val)
@noinline _centered_average_engine!(::CpuPolyester, out, in_ref, dims, dim_val) = _batch_centered_average_engine!(
    out, in_ref, dims, dim_val)

@noinline function _throw_no_device_centered_average()
    return error(
        "the centered average (Mcₓ, Mcᵧ, Mc₂, Mcₕ) has no device kernel yet; apply it to " *
        "a host-backed VectorElement instead.",
    )
end

@inline function _apply_averaged!(
        vₕ::VectorElement{<:ScalarGridSpace},
        uₕ::VectorElement{<:ScalarGridSpace},
        ::Centered,
        dim_val::Val
)
    _check_no_alias(vₕ, uₕ)
    sp = space(uₕ)
    (execution_policy(sp) isa GpuPolicy || locality(typeof(vₕ.data)) isa DeviceLocality ||
     locality(typeof(uₕ.data)) isa DeviceLocality) && _throw_no_device_centered_average()
    _centered_average_engine!(execution_policy(sp), vₕ.data, uₕ.data, _grid_dims(uₕ), dim_val)
    return vₕ
end

# Divided by 2 rather than multiplied by 0.5, for the reason given at `_compute_average`:
# the literal is a Float64 and promotes the whole matrix. On a Float32 backend everything
# else in the library stayed Float32 and only the averaging matrices came back Float64.
function add_half_shift(
        Ωₕ::AbstractMeshType, ::Val{DIFF_DIM}, ::Val{first}, ::Val{second}
) where {DIFF_DIM, first, second}
    return (shift(Ωₕ, Val(DIFF_DIM), Val(first)) + shift(Ωₕ, Val(DIFF_DIM), Val(second))) /
           2
end

function _average_operator(Ωₕ::AbstractMeshType, ::Forward, ::Val{AVG_DIM}) where {AVG_DIM}
    return add_half_shift(Ωₕ, Val(AVG_DIM), Val(0), Val(1))
end

function _average_operator(Ωₕ::AbstractMeshType, ::Backward, ::Val{AVG_DIM}) where {AVG_DIM}
    return add_half_shift(Ωₕ, Val(AVG_DIM), Val(0), Val(-1))
end

function _average_operator(Ωₕ::AbstractMeshType, ::Centered, ::Val{AVG_DIM}) where {AVG_DIM}
    return (shift(Ωₕ, Val(AVG_DIM), Val(-1)) + 2 * shift(Ωₕ, Val(AVG_DIM), Val(0)) +
            shift(Ωₕ, Val(AVG_DIM), Val(1))) / 4
end

# The centered average's rows are one away from both end slices and zero on them, the
# weight vector of the retained Kronecker oracle below; host-only, as the centered average
# has no device path.
function _average_weights!(
        v::AbstractVector, Ωₕ::AbstractMeshType, ::Centered, ::Val{DIFF_DIM}
) where {DIFF_DIM}
    dims = npoints(Ωₕ, Tuple)
    1 <= DIFF_DIM <= dim(Ωₕ) || _throw_stencil_dim_error(DIFF_DIM, dim(Ωₕ))
    li = LinearIndices(dims)
    n = dims[DIFF_DIM]

    @inbounds for I in CartesianIndices(dims)
        v[li[I]] = (I[DIFF_DIM] == 1 || I[DIFF_DIM] == n) ? zero(eltype(v)) : one(eltype(v))
    end
    return nothing
end

function _average_weights!(
        v::AbstractVector, Ωₕ::AbstractMeshType, dir::GridDirection, dim_val::Val{DIFF_DIM}
) where {DIFF_DIM}
    dims = npoints(Ωₕ, Tuple)

    1 <= DIFF_DIM <= dim(Ωₕ) || _throw_stencil_dim_error(DIFF_DIM, dim(Ωₕ))

    if execution_policy(Ωₕ) isa GpuPolicy
        _device_average_weights!(v, dims, dir, dim_val)
        return nothing
    end

    li = LinearIndices(dims)

    @inbounds @simd for I in CartesianIndices(dims)
        idx = I[DIFF_DIM]
        if dir isa Forward
            v[li[I]] = idx == dims[DIFF_DIM] ? zero(eltype(v)) : one(eltype(v))
        else # Backward
            v[li[I]] = idx == 1 ? zero(eltype(v)) : one(eltype(v))
        end
    end
    return nothing
end

# The `GpuPolicy` branch above (S2.4 of
# .agents/plans/metal-and-apple-silicon-acceleration.md): every entry is one except a
# single boundary slice along `DIFF_DIM`, which is exactly a broadcast fill followed by a
# bulk zero over a `view` of that one slice -- no per-element kernel needed, matching S2.2's
# "reach for a broadcast before a kernel" precedent. `reshape` and `view` both work on a
# device array without scalar indexing, unlike the `li[I]`-indexed CPU loop above.
@inline function _device_average_weights!(
        v::AbstractVector, dims::NTuple{D, Int}, ::Forward, ::Val{DIFF_DIM}
) where {D, DIFF_DIM}
    v .= one(eltype(v))
    vr = reshape(v, dims)
    slice = ntuple(d -> d == DIFF_DIM ? (dims[d]:dims[d]) : Colon(), Val(D))
    view(vr, slice...) .= zero(eltype(v))
    return nothing
end

@inline function _device_average_weights!(
        v::AbstractVector, dims::NTuple{D, Int}, ::Backward, ::Val{DIFF_DIM}
) where {D, DIFF_DIM}
    v .= one(eltype(v))
    vr = reshape(v, dims)
    slice = ntuple(d -> d == DIFF_DIM ? (1:1) : Colon(), Val(D))
    view(vr, slice...) .= zero(eltype(v))
    return nothing
end

# Configuration array for average operators, expanded with descriptive strings.
const _AVERAGE_OP_CONFIGS = [
    (
        direction = Forward(),
        average_name = :forward_average,
        average_alias = :M₊,
        vectorial_average_alias = :M₊ₕ,
        dir_string_lowercase = "forward",
        math_op = "\\frac{u_{i} + u_{i+1}}{2}",
        stencil_op = :ForwardAvgOp
    ),
    (
        direction = Backward(),
        average_name = :backward_average,
        average_alias = :M,
        vectorial_average_alias = :Mₕ,
        dir_string_lowercase = "backward",
        math_op = "\\frac{u_{i-1} + u_{i}}{2}",
        stencil_op = :BackwardAvgOp
    ),
    (
        direction = Centered(),
        average_name = :centered_average,
        average_alias = :Mc,
        vectorial_average_alias = :Mcₕ,
        dir_string_lowercase = "centered",
        math_op = "\\frac{u_{i-1} + 2 u_{i} + u_{i+1}}{4}",
        stencil_op = :CenteredAvgOp
    )
]

# Metaprogramming loop to generate all specified average operators.
for config in _AVERAGE_OP_CONFIGS
    # Extract ALL values from `config` to avoid scope issues with @eval.
    dir_instance = config.direction
    average_name = config.average_name
    average_alias = config.average_alias
    vectorial_average_alias = config.vectorial_average_alias
    dir_string_lowercase = config.dir_string_lowercase
    math_op = config.math_op
    stencil_op = config.stencil_op

    @eval begin
        # --- In-place applicators ---
        @doc """
            $($(QuoteNode(Symbol(average_name, :_dim!))))(out, in, dims, average_dim)

        Low-level, in-place function to compute the $($dir_string_lowercase) average of vector `in` along dimension `average_dim`, storing the result in `out`. This function computes ``$($math_op)``.
        """
        function $(Symbol(average_name, :_dim!))(
                out, in, h, dims::NTuple{D, Int}, average_dim::Val{DIFF_DIM}
        ) where {D, DIFF_DIM}
            1 <= DIFF_DIM <= D || _throw_stencil_dim_error(DIFF_DIM, D)
            length(out) == length(in) == prod(dims) ||
                _throw_stencil_size_error(length(out), length(in), dims)
            in_ref = (out === in) ? copy(in) : in
            if $dir_instance isa Centered
                _centered_average_engine!(out, in_ref, dims, average_dim)
            else
                _average_engine!(out, in_ref, dims, $dir_instance, average_dim)
            end
            return nothing
        end

        function $(Symbol(average_name, :_dim!))(
                out, in, dims::NTuple{D, Int}, average_dim::Val{DIFF_DIM}
        ) where {D, DIFF_DIM}
            return $(Symbol(average_name, :_dim!))(out, in, nothing, dims, average_dim)
        end

        # --- Retained Kronecker oracle (gpena/Bramble.jl#185) ---
        #
        # The body `$average_name` used to have, kept under a `_kron_` name so
        # `kronecker_operator_matrix` has an independent construction to check
        # `stencil_matrix` against.
        function $(Symbol(:_kron_, average_name))(
                Ωₕ::AbstractMeshType, dim_val::Val; vector_cache = __vector(Ωₕ)
        )
            avg_matrix = _average_operator(Ωₕ, $dir_instance, dim_val)
            _average_weights!(vector_cache, Ωₕ, $dir_instance, dim_val)
            return _scale_rows!(avg_matrix, vector_cache)
        end

        # --- Matrix operator functions (single-pass, gpena/Bramble.jl#185) ---
        @doc """
            $($(QuoteNode(average_name)))(arg, dim_val::Val)

        Constructs or applies the $($dir_string_lowercase) averaging operator, representing the operation ``$($math_op)``.
        """
        @inline $average_name(Ωₕ::AbstractMeshType, dim_val::Val{DIM}) where {DIM} = stencil_matrix(Ωₕ, $(stencil_op){DIM}())

        # --- Generic applicators ---
        #
        # Only the mesh-forwarding overload is generated here; the grid-function trio
        # (scalar `!`, composite `!`, allocating) comes from
        # `@operator_family` below, shared with `difference.jl`
        # (gpena/Bramble.jl#101).
        @inline $average_name(Wₕ::AbstractSpaceType, dim_val::Val) = $average_name(mesh(Wₕ), dim_val)
    end
end

# The grid-function forms and the alias surface, one `@operator_family` call per family.
# They sit outside the loop above because a macro is expanded where it is written: the
# family's configuration has to be literal at that point, not a `config.field` read at load
# time (gpena/Bramble.jl#258).
#
# An average divides by nothing the direction does not already say, so unlike the
# differences it needs no spacing function and no precondition: `_apply_averaged!` takes the
# direction alone, and `extra_args` is left out.
#
# `dispatch_alias` is the one place these two differ from every other family
# (gpena/Bramble.jl#74). Elsewhere the dimensional entry point takes the family's stem --
# `D₋(uₕ, d)` next to `D₋ₓ` -- but the average's stem is `M`, and `M` is the most common
# local name in finite-element code for a mass matrix. Minting it would mean `using Bramble`
# reserved it, and a caller writing `M = assemble(a, Wₕ)` at top level would get "cannot
# assign a value to imported variable M" for their trouble. So the averages put the
# direction argument on the tuple-valued alias they already export instead: `Mₕ(uₕ)` is
# still the tuple, `Mₕ(uₕ, 2)` is the `y` average, and no new name enters the surface.
@operator_family(base=forward_average,
    stem=M₊,
    apply_fn=_apply_averaged!,
    direction=Forward(),
    dir_string="forward",
    what="average",
    formula="\\frac{u_{i} + u_{i+1}}{2}",
    dispatch_alias=M₊ₕ,
    vectorial_alias=M₊ₕ)

@operator_family(base=backward_average,
    stem=M,
    apply_fn=_apply_averaged!,
    direction=Backward(),
    dir_string="backward",
    what="average",
    formula="\\frac{u_{i-1} + u_{i}}{2}",
    dispatch_alias=Mₕ,
    vectorial_alias=Mₕ)

# The centered average follows the same rule: there is no bare `Mc`, and `Mcₕ(uₕ, d)` is the
# dimensional entry point (gpena/Bramble.jl#287).
@operator_family(base=centered_average,
    stem=Mc,
    apply_fn=_apply_averaged!,
    direction=Centered(),
    dir_string="centered",
    what="average",
    formula="\\frac{u_{i-1} + 2 u_{i} + u_{i+1}}{4}",
    trailing_note="The first and last points along `{direction}` are truncated "*
                  "to zero. On a device-backed grid function it throws, as there is "*
                  "no device kernel yet.",
    dispatch_alias=Mcₕ,
    vectorial_alias=Mcₕ)

# --- Kronecker oracle dispatch (gpena/Bramble.jl#185) --------------------------------- #
#
# `kronecker_operator_matrix` is declared in shift.jl; each family's dispatch method maps
# its public per-axis alias to the `_kron_*` construction kept above.
for (i, suffix) in enumerate(_BRAMBLE_var2symbol)
    for (stem, kron_fn) in ((:M, :_kron_backward_average), (:M₊, :_kron_forward_average),
        (:Mc, :_kron_centered_average))
        alias = Symbol(stem, suffix)
        @eval kronecker_operator_matrix(Ωₕ::AbstractMeshType, ::typeof($alias)) = $kron_fn(Ωₕ, Val($i))
    end
end
