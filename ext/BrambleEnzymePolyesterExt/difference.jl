# difference.jl: Enzyme rules for `_batch_difference_engine!` (src/operators/difference.jl).
#
# The engine writes `out = A(h) in_ref`, linear in `in_ref`, one `_difference_band!` per
# `@batch` band. Every family reaches it: one-sided with or without a spacing, `D̃`, `Dc` and
# `D̽ₕ`, with `h` as `nothing`, a vector, a view of one or a `StarSpacings`.
#
# Forward: the tangent of `out` is `A(h)` applied to the tangent of `in_ref`, so each band
# runs the engine's own `_difference_band!` on the tangents, then adds the term through the
# spacing tangent when `h` is active (`_difference_band_spacing_tangent!`).
#
# Both rules take `Duplicated` or `BatchDuplicated` arguments, the width from
# `EnzymeRules.width(config)`: DI's sparse Jacobians seed several colours per pass. Every
# band runs every lane, so a launch serves the whole batch; no loop over lanes launches
# `@batch` once per lane.
#
# Reverse: `_difference_band_adjoint!` adds `A(h)ᵀ d_out` to `d_in` in gather form, each
# `d_in[j]` written by the band owning `j`; with `h` active, `_difference_spacing_adjoint!`
# gathers each spacing's cotangent over its own band of the spacing index, so no two tasks
# write one entry whichever axis is differenced. `out` was overwritten, so `d_out` is cleared
# after, in a second `@batch`: when the differenced axis is the banded one, a band reads its
# neighbour's `d_out` across the band edge, and clearing inside the first loop would race.
#
# A `StarSpacings` spacing's shadow is a `StarSpacings` of the shadow spacings. Its tangent
# reads through it unchanged; its cotangent gathers into a scratch vector, folded into the
# spacings' own shadow by `_star_spacings_adjoint!` under a third `@batch`.

const _DifferenceEngine = typeof(Bramble._batch_difference_engine!)

# Where the spacing cotangent gathers: `d_h` itself, or a scratch vector for a
# `StarSpacings`, whose entries are not stored. A callable spacing has no shadow to write.
@inline _spacing_target(::Nothing) = nothing
@inline _spacing_target(d_h::AbstractVector) = d_h
@inline _spacing_target(d_h::StarSpacings) = zeros(eltype(d_h.h), length(d_h))
@noinline function _spacing_target(d_h)
    throw(ArgumentError("the Enzyme rule for `_batch_difference_engine!` differentiates a \
        vector spacing, not a $(typeof(d_h)); pass the spacing as a vector"))
end

# A copy of the spacing values, for a tape that must outlive a later overwrite.
@inline _copy_spacing(h) = h
@inline _copy_spacing(h::AbstractVector) = copy(h)
@inline _copy_spacing(h::StarSpacings) = StarSpacings(copy(h.h))

# The linear indices of band `b`'s slab, the block `_difference_band!` writes: a cut of the
# last axis of a column-major grid is one contiguous block.
@inline function _slab_block(dims::Tuple, n::Int, b::Int)
    stride = prod(Base.front(dims))
    band = _band_range(Base.OneTo(last(dims)), n, b)
    return ((first(band) - 1) * stride + 1):(last(band) * stride)
end

# One lane's tangent on band `b`: `A(h)` applied to the tangent of `in_ref` (zero when
# `in_ref` is inactive in that lane), plus the term through the spacing tangent.
@inline function _forward_lane!(d_out, d_in, d_h, in_ref, h, dims, dir, dim_val, n, b)
    d_out === nothing && return nothing
    if d_in === nothing
        for i in _slab_block(dims, n, b)
            @inbounds d_out[i] = zero(eltype(d_out))
        end
    else
        Bramble._difference_band!(d_out, d_in, h, dims, dir, dim_val, n, b)
    end
    d_h === nothing ||
        _difference_band_spacing_tangent!(d_out, in_ref, h, d_h, dims, dir, dim_val, n, b)
    return nothing
end

@inline function _forward_band!(d_outs, d_ins, d_hs, in_ref, h, dims, dir, dim_val, n, b)
    map((o, i, s) -> _forward_lane!(o, i, s, in_ref, h, dims, dir, dim_val, n, b),
        d_outs, d_ins, d_hs)
    return nothing
end

# Every lane's tangent in one `@batch` launch, each band running every lane.
function _difference_forward!(d_outs, d_ins, d_hs, in_ref, h, dims, dir, dim_val)
    n = Threads.nthreads()
    @batch for b in 1:n
        _forward_band!(d_outs, d_ins, d_hs, in_ref, h, dims, dir, dim_val, n, b)
    end
    return nothing
end

@inline function _reverse_lane!(d_out, d_in, d_hv, in_ref, h, dims, dir, dim_val, n, b)
    d_out === nothing && return nothing
    d_in === nothing || _difference_band_adjoint!(d_in, d_out, h, dims, dir, dim_val, n, b)
    d_hv === nothing ||
        _difference_spacing_adjoint!(d_hv, in_ref, d_out, h, dims, dir, dim_val, n, b)
    return nothing
end

@inline function _reverse_band!(d_outs, d_ins, d_hvs, in_ref, h, dims, dir, dim_val, n, b)
    map((o, i, s) -> _reverse_lane!(o, i, s, in_ref, h, dims, dir, dim_val, n, b),
        d_outs, d_ins, d_hvs)
    return nothing
end

@inline _fold_lane!(d_h, d_hv, n, b) = nothing
@inline _fold_lane!(d_h::StarSpacings, d_hv, n, b) = _star_spacings_adjoint!(
    d_h.h, d_hv, n, b)
@inline function _fold_band!(d_hs, d_hvs, n, b)
    map((s, v) -> _fold_lane!(s, v, n, b), d_hs, d_hvs)
    return nothing
end

@inline function _zero_lane!(d_out, n, b)
    d_out === nothing && return nothing
    for i in _band_range(Base.OneTo(length(d_out)), n, b)
        @inbounds d_out[i] = zero(eltype(d_out))
    end
    return nothing
end
@inline function _zero_band!(d_outs, n, b)
    map(o -> _zero_lane!(o, n, b), d_outs)
    return nothing
end

# Every lane's adjoint, in three `@batch` launches that each run every lane: the gathers,
# the `StarSpacings` fold, and clearing `d_out`.
function _difference_reverse!(d_outs, d_ins, d_hs, in_ref, h, dims, dir, dim_val)
    n = Threads.nthreads()
    d_hvs = map(_spacing_target, d_hs)
    @batch for b in 1:n
        _reverse_band!(d_outs, d_ins, d_hvs, in_ref, h, dims, dir, dim_val, n, b)
    end
    if any(d -> d isa StarSpacings, d_hs)
        @batch for b in 1:n
            _fold_band!(d_hs, d_hvs, n, b)
        end
    end
    @batch for b in 1:n
        _zero_band!(d_outs, n, b)
    end
    return nothing
end

function EnzymeRules.forward(config::FwdConfig, func::Const{_DifferenceEngine}, ::Type{RT},
        out::Annotation{<:AbstractVector}, in_ref::Annotation, h::Annotation,
        dims::Annotation, dir::Annotation, dim_val::Annotation) where {RT}
    _entered!(:_batch_difference_engine!)
    func.val(out.val, in_ref.val, h.val, dims.val, dir.val, dim_val.val)
    W = Val(EnzymeRules.width(config))
    d_outs = _lanes(out, W)
    all(isnothing, d_outs) && return nothing
    _difference_forward!(d_outs, _lanes(in_ref, W), _lanes(h, W), in_ref.val, h.val,
        dims.val, dir.val, dim_val.val)
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_DifferenceEngine},
        ::Type{RT}, out::Annotation{<:AbstractVector}, in_ref::Annotation, h::Annotation,
        dims::Annotation, dir::Annotation, dim_val::Annotation) where {RT}
    _entered!(:_batch_difference_engine!)
    func.val(out.val, in_ref.val, h.val, dims.val, dir.val, dim_val.val)
    # `overwritten` counts `func` first: `in_ref` is entry 3, `h` entry 4. `in_ref`'s values
    # are read back only for the spacing cotangent.
    ow = EnzymeRules.overwritten(config)
    in_tape = ow[3] && !(h isa Const) ? copy(in_ref.val) : nothing
    h_tape = ow[4] ? _copy_spacing(h.val) : nothing
    return AugmentedReturn(nothing, nothing, (in_tape, h_tape))
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_DifferenceEngine}, ::Type{RT},
        tape, out::Annotation{<:AbstractVector}, in_ref::Annotation, h::Annotation,
        dims::Annotation, dir::Annotation, dim_val::Annotation) where {RT}
    W = Val(EnzymeRules.width(config))
    d_outs = _lanes(out, W)
    if !all(isnothing, d_outs)
        in_tape, h_tape = tape
        _difference_reverse!(d_outs, _lanes(in_ref, W), _lanes(h, W),
            in_tape === nothing ? in_ref.val : in_tape,
            h_tape === nothing ? h.val : h_tape, dims.val, dir.val, dim_val.val)
    end
    return (nothing, nothing, nothing, nothing, nothing, nothing)
end
