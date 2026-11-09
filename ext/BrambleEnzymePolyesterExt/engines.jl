# engines.jl: Enzyme rules for the average, centred average and band engines under
# `CpuPolyester` (`_batch_average_engine!`, `_batch_centered_average_engine!`,
# `_batch_run_bands!`).
#
# The two average engines take `difference.jl`'s fixed-kernel route. Each writes
# `out = A in_ref` with `A` constant, one `_average_band!`/`_centered_average_band!` per
# `@batch` band. Forward runs the same band kernel on the tangents, every lane in each band.
# Reverse runs the gather-form `_average_band_adjoint!`/`_centered_average_band_adjoint!`
# (src/operators/average.jl), each `d_in[j]` written by the band owning `j`, then clears
# `d_out` in a second `@batch`, since a band reads its neighbour's `d_out` across the band
# edge when the averaged axis is the banded one. `A` reads no data, so the tape is empty.
#
# `_batch_run_bands!` takes `sweeps.jl`'s closure route: its kernel `f` is whichever banded
# engine the caller passes (`_shift_band!`, the `_accumulate_*!` engines, the in-place
# averages, the ghost, multigrid and matrix-free bands), most of them outside this file's
# reach, so no fixed adjoint covers it. Each `@batch` task runs an Enzyme derivative of
# its own band, `f(out, rest..., nbands, b)`, through a thunk built on the host and handed
# to the task through `BramblePolyesterExt`'s slot (`_handed`/`_take`), as `sweeps.jl` does.
#
# - Forward: each task runs Enzyme's forward derivative of its band, writing `out` and its
#   tangent at the band's points only; a batched thunk runs every lane in one call.
# - Reverse: each task runs Enzyme's split reverse of its band, every lane through its own
#   width-1 thunk. `d_out` is shared: a band reads and clears it only at the points it
#   writes, the disjointness the primal's bands already rely on. Every other shadow is read
#   across band edges, so each task gathers into its own zeroed copy, added into the shadow
#   after the join, under `@batch` for an array, by `_accumulate!` otherwise. An `Active`
#   argument (a scalar weight) has its per-task derivatives summed on the host.
#
# The reverse pass replays each band's primal on `out` as it was before the call (the
# `+=` engines would otherwise add twice, and the in-place averages read `out` itself), so
# `out`'s values are kept from before the primal and put back for the reverse. An argument
# that later code overwrites (`overwritten`) is copied whole before the primal.
#
# `_run_bands!` hands the kernel behind a `Ref` (src/operators/vector_calculus.jl), for the
# calling-convention reason `_sweep_for!` does; the task unwraps it with `_kernel`.

const _AverageEngine = typeof(Bramble._batch_average_engine!)
const _CenteredEngine = typeof(Bramble._batch_centered_average_engine!)
const _RunBands = typeof(Bramble._batch_run_bands!)

# --- The average engines ----------------------------------------------------------- #

# One lane's tangent on band `b`: the band kernel on the tangent of `in_ref`, or zero over
# the slab where `in_ref` is inactive in that lane. `extra` is `(dir, dim_val)` or
# `(dim_val,)`.
@inline function _avg_tangent_lane!(band!::K, d_out, d_in, dims, extra, n, b) where {K}
    d_out === nothing && return nothing
    if d_in === nothing
        for i in _slab_block(dims, n, b)
            @inbounds d_out[i] = zero(eltype(d_out))
        end
    else
        band!(d_out, d_in, dims, extra..., n, b)
    end
    return nothing
end

@inline function _avg_tangent_band!(band!::K, d_outs, d_ins, dims, extra, n, b) where {K}
    map((o, i) -> _avg_tangent_lane!(band!, o, i, dims, extra, n, b), d_outs, d_ins)
    return nothing
end

function _avg_forward!(band!::K, d_outs, d_ins, dims, extra) where {K}
    n = Threads.nthreads()
    @batch for b in 1:n
        _avg_tangent_band!(band!, d_outs, d_ins, dims, extra, n, b)
    end
    return nothing
end

@inline function _avg_adjoint_band!(adj!::K, d_outs, d_ins, dims, extra, n, b) where {K}
    map(d_outs, d_ins) do d_out, d_in
        (d_out === nothing || d_in === nothing) && return nothing
        return adj!(d_in, d_out, dims, extra..., n, b)
    end
    return nothing
end

# Every lane's adjoint in one `@batch` launch, then `d_out` cleared in a second.
function _avg_reverse!(adj!::K, d_outs, d_ins, dims, extra) where {K}
    n = Threads.nthreads()
    @batch for b in 1:n
        _avg_adjoint_band!(adj!, d_outs, d_ins, dims, extra, n, b)
    end
    @batch for b in 1:n
        _zero_band!(d_outs, n, b)
    end
    return nothing
end

function EnzymeRules.forward(config::FwdConfig, func::Const{_AverageEngine}, ::Type{RT},
        out::Annotation{<:AbstractVector}, in_ref::Annotation, dims::Annotation,
        dir::Annotation, dim_val::Annotation) where {RT}
    _entered!(:_batch_average_engine!)
    func.val(out.val, in_ref.val, dims.val, dir.val, dim_val.val)
    W = Val(EnzymeRules.width(config))
    d_outs = _lanes(out, W)
    all(isnothing, d_outs) && return nothing
    _avg_forward!(Bramble._average_band!, d_outs, _lanes(in_ref, W), dims.val,
        (dir.val, dim_val.val))
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_AverageEngine},
        ::Type{RT}, out::Annotation{<:AbstractVector}, in_ref::Annotation,
        dims::Annotation, dir::Annotation, dim_val::Annotation) where {RT}
    _entered!(:_batch_average_engine!)
    func.val(out.val, in_ref.val, dims.val, dir.val, dim_val.val)
    return AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_AverageEngine}, ::Type{RT},
        tape, out::Annotation{<:AbstractVector}, in_ref::Annotation, dims::Annotation,
        dir::Annotation, dim_val::Annotation) where {RT}
    W = Val(EnzymeRules.width(config))
    d_outs = _lanes(out, W)
    all(isnothing, d_outs) || _avg_reverse!(Bramble._average_band_adjoint!, d_outs,
        _lanes(in_ref, W), dims.val, (dir.val, dim_val.val))
    return (nothing, nothing, nothing, nothing, nothing)
end

function EnzymeRules.forward(config::FwdConfig, func::Const{_CenteredEngine}, ::Type{RT},
        out::Annotation{<:AbstractVector}, in_ref::Annotation, dims::Annotation,
        dim_val::Annotation) where {RT}
    _entered!(:_batch_centered_average_engine!)
    func.val(out.val, in_ref.val, dims.val, dim_val.val)
    W = Val(EnzymeRules.width(config))
    d_outs = _lanes(out, W)
    all(isnothing, d_outs) && return nothing
    _avg_forward!(Bramble._centered_average_band!, d_outs, _lanes(in_ref, W), dims.val,
        (dim_val.val,))
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_CenteredEngine},
        ::Type{RT}, out::Annotation{<:AbstractVector}, in_ref::Annotation,
        dims::Annotation, dim_val::Annotation) where {RT}
    _entered!(:_batch_centered_average_engine!)
    func.val(out.val, in_ref.val, dims.val, dim_val.val)
    return AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_CenteredEngine}, ::Type{RT},
        tape, out::Annotation{<:AbstractVector}, in_ref::Annotation, dims::Annotation,
        dim_val::Annotation) where {RT}
    W = Val(EnzymeRules.width(config))
    d_outs = _lanes(out, W)
    all(isnothing, d_outs) || _avg_reverse!(Bramble._centered_average_band_adjoint!,
        d_outs, _lanes(in_ref, W), dims.val, (dim_val.val,))
    return (nothing, nothing, nothing, nothing)
end

# --- _batch_run_bands! --------------------------------------------------------------- #

# What each task differentiates: band `b` of `n` of the kernel `f`.
@inline function _bands_task!(f::F, out, n::Int, b::Int, rest::Vararg{Any, N}) where {F, N}
    _kernel(f)(out, rest..., n, b)
    return nothing
end

@inline _bands_values(args::Tuple) = map(a -> a.val, args)

# The first exception a task left behind, rethrown on the host after the join.
@noinline function _bands_rethrow(errs::Vector{Any})
    for e in errs
        e === nothing || throw(e)
    end
    error("a CpuPolyester derivative task failed without an exception")
end

# Forward: one batched thunk, every lane in each task's one call.

struct _BandsForward{TH, FA, OA, RA}
    thunk::TH
    f::FA
    out::OA
    rest::RA
    errs::Vector{Any}
end

@noinline function _bands_forward_threw(p::_BandsForward, n::Int, b::Int)
    try
        p.thunk(Const(_bands_task!), p.f, p.out, Const(n), Const(b), p.rest...)
    catch e
        p.errs[b] = e
        return true
    end
    return false
end

@noinline function _bands_forward_run!(take::T, n::Int, h::H) where {T, H}
    failed = false
    @batch reduction=((|, failed),) for b in 1:n
        failed |= _bands_forward_threw(take(h), n, b)
    end
    return failed
end

function EnzymeRules.forward(config::FwdConfig, func::Const{_RunBands}, ::Type{RT},
        f::Annotation, nbands::Annotation, out::Annotation{<:AbstractVector},
        rest::Vararg{Annotation, N}) where {RT, N}
    _entered!(:_batch_run_bands!)
    n = nbands.val
    if all(isnothing, _lanes(out, Val(EnzymeRules.width(config))))
        func.val(f.val, n, out.val, _bands_values(rest)...)
        return nothing
    end
    mode = Enzyme.set_runtime_activity(Enzyme.Forward, EnzymeRules.runtime_activity(config))
    thunk = Enzyme.autodiff_thunk(mode, Const{typeof(_bands_task!)}, Const, typeof(f),
        typeof(out), Const{Int}, Const{Int}, map(typeof, rest)...)
    p = _BandsForward(thunk, f, out, rest, Vector{Any}(nothing, n))
    pe = _polyester_ext()
    pe._handed(_bands_forward_run!, p, pe._take, n) && _bands_rethrow(p.errs)
    return nothing
end

# Reverse.
#
# The shadow of an input (the kernel or a `rest` argument) in lane `w`: `nothing` where it
# has none to write (`Const`, `Active`, runtime activity's primal-as-shadow), the shadow
# `Ref` for an immutable value with active fields (`MixedDuplicated`).
@inline _bands_shadow(a, w::Int) = a isa Active ? nothing : _shadow(a, w)
@inline _bands_shadow(a::MixedDuplicated, w::Int) = a.dval
@inline _bands_shadow(a::BatchMixedDuplicated, w::Int) = a.dval[w]

# A task's annotation of an input, given its per-task shadow copy `c`.
@inline _bands_annotation(a, v, ::Nothing) = a isa Active ? Active(v) : Const(v)
@inline _bands_annotation(a, v, c) = Duplicated(v, c)
@inline function _bands_annotation(
        a::Union{MixedDuplicated, BatchMixedDuplicated}, v, c::Base.RefValue)
    return MixedDuplicated(v, c)
end

# One lane: its thunks, its `d_out`, and `copies[b]`, task `b`'s zeroed copies of the lane's
# input shadows (`nothing` for an input with none). `act[b]` holds what task `b`'s reverse
# returned, read for the `Active` inputs.
struct _BandsLane{FW, RV, DO, C}
    fwd::FW
    rev::RV
    d_out::DO
    copies::Vector{C}
    act::Vector{Any}
end

# `ins` the inputs `(f, rest...)` as annotated, `vals` their values for the replay.
struct _BandsReverse{A, V, O, L}
    ins::A
    vals::V
    out::O
    lanes::L
    errs::Vector{Any}
end

@inline function _bands_lane!(p::_BandsReverse, lane::_BandsLane, n::Int, b::Int)
    anns = map(_bands_annotation, p.ins, p.vals, lane.copies[b])
    fa, ra, oa = first(anns), Base.tail(anns), Duplicated(p.out, lane.d_out)
    tape, _, _ = lane.fwd(Const(_bands_task!), fa, oa, Const(n), Const(b), ra...)
    r = lane.rev(Const(_bands_task!), fa, oa, Const(n), Const(b), ra..., tape)
    lane.act[b] = first(r)
    return nothing
end

@noinline function _bands_reverse_threw(p::_BandsReverse, n::Int, b::Int)
    try
        map(lane -> lane === nothing ? nothing : _bands_lane!(p, lane, n, b), p.lanes)
    catch e
        p.errs[b] = e
        return true
    end
    return false
end

@noinline function _bands_reverse_run!(take::T, n::Int, h::H) where {T, H}
    failed = false
    @batch reduction=((|, failed),) for b in 1:n
        failed |= _bands_reverse_threw(take(h), n, b)
    end
    return failed
end

# The per-task copies `cs` of a shadow `d` added into it: an array of numbers under
# `@batch`, each band summing its own indices over the copies; anything else by
# `_accumulate!`, which skips data that is its own primal (`primal` the input as the caller
# passed it).
struct _BandsSum{D, C}
    d::D
    cs::C
end

@inline function _bands_sum_band!(p::_BandsSum, n::Int, b::Int)
    d = p.d
    for i in _band_range(Base.OneTo(length(d)), n, b)
        acc = @inbounds d[i]
        for c in p.cs
            acc += @inbounds c[i]
        end
        @inbounds d[i] = acc
    end
    return nothing
end

@noinline function _bands_sum_run!(take::T, n::Int, h::H) where {T, H}
    @batch for b in 1:n
        _bands_sum_band!(take(h), n, b)
    end
    return false
end

function _bands_reduce!(handed::HD, take::TK, d::Array{<:Union{AbstractFloat, Complex}},
        cs, primal) where {HD, TK}
    handed(_bands_sum_run!, _BandsSum(d, cs), take, Threads.nthreads())
    return nothing
end
function _bands_reduce!(handed::HD, take::TK, d, cs, primal) where {HD, TK}
    foreach(c -> _accumulate!(d, c, primal), cs)
    return nothing
end

# `overwritten` counts `func` first: input `k` of `(f, rest...)` is entry `k + 3`, `f`
# entry 2 (`nbands` and `out` sit between).
@inline _bands_overwritten(ow, k::Int) = ow[k == 1 ? 2 : k + 3]

# Two derivatives of one `Active` value added field by field: a number, or a tuple or an
# immutable struct of them (the ghost operators' `_FacetFix` pairs), every field that is not
# a float kept from `x`.
_bands_add(x::T, y::T) where {T <: Union{AbstractFloat, Complex}} = x + y
_bands_add(x::Tuple, y::Tuple) = map(_bands_add, x, y)
function _bands_add(x::T, y::T) where {T}
    (ismutabletype(T) || fieldcount(T) == 0) && return x
    vals = Any[_bands_add(getfield(x, i), getfield(y, i)) for i in 1:fieldcount(T)]
    return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, vals, fieldcount(T))::T
end
_bands_add(x, y) = x

# What `Active` input `k`'s derivative is: the sum of its per-task derivatives, a lane each;
# `nothing` for any other input. Dispatch on the annotation, unrolled over the inputs by
# `_bands_actives`, keeps each slot's type its own: `ins[k]` under `ntuple(k -> ...)` is not
# const-propagated, and every slot came back `Union{Nothing, T}`, which Enzyme rejects.
@inline _bands_slot(k::Int) = k == 1 ? 1 : k + 3
@inline _bands_active(lanes, a, k::Int, ::Val{W}) where {W} = nothing
function _bands_active(lanes, a::Active{T}, k::Int, ::Val{W}) where {T, W}
    d = ntuple(Val(W)) do w
        lane = lanes[w]
        lane === nothing && return Enzyme.make_zero(a.val)::T
        return reduce(_bands_add, (r[_bands_slot(k)]::T for r in lane.act))::T
    end
    return W == 1 ? only(d) : d
end

@inline _bands_actives(lanes, ::Tuple{}, k::Int, V::Val) = ()
@inline function _bands_actives(lanes, ins::Tuple, k::Int, V::Val)
    return (_bands_active(lanes, first(ins), k, V),
        _bands_actives(lanes, Base.tail(ins), k + 1, V)...)
end

@inline _bands_zero_active(a, ::Val{W}) where {W} = nothing
@inline function _bands_zero_active(a::Active{T}, ::Val{W}) where {T, W}
    z = Enzyme.make_zero(a.val)::T
    return W == 1 ? z : ntuple(_ -> z, Val(W))
end

function _bands_reverse!(handed::HD, take::TK, mode::M, n::Int, ins, vals, out, d_outs,
        V::Val{W}) where {HD, TK, M, W}
    lanes = ntuple(V) do w
        d_outs[w] === nothing && return nothing
        shadows = map(a -> _bands_shadow(a, w), ins)
        copies = [map(s -> s === nothing ? nothing : _graph_copy(s, true), shadows)
                  for _ in 1:n]
        anns = map(_bands_annotation, ins, vals, first(copies))
        fwd, rev = Enzyme.autodiff_thunk(mode, Const{typeof(_bands_task!)}, Const,
            typeof(first(anns)), Duplicated{typeof(out)}, Const{Int}, Const{Int},
            map(typeof, Base.tail(anns))...)
        return _BandsLane(fwd, rev, d_outs[w], copies, Vector{Any}(nothing, n))
    end
    p = _BandsReverse(ins, vals, out, lanes, Vector{Any}(nothing, n))
    handed(_bands_reverse_run!, p, take, n) && _bands_rethrow(p.errs)
    for w in 1:W
        lane = lanes[w]
        lane === nothing && continue
        for k in eachindex(ins)
            s = _bands_shadow(ins[k], w)
            s === nothing ||
                _bands_reduce!(handed, take, s, [c[k] for c in lane.copies], ins[k].val)
        end
    end
    return _bands_actives(lanes, ins, 1, V)
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_RunBands},
        ::Type{RT}, f::Annotation, nbands::Annotation, out::Annotation{<:AbstractVector},
        rest::Vararg{Annotation, N}) where {RT, N}
    _entered!(:_batch_run_bands!)
    tape = nothing
    if !all(isnothing, _lanes(out, Val(EnzymeRules.width(config))))
        ow, ins = EnzymeRules.overwritten(config), (f, rest...)
        kept = ntuple(
            k -> _bands_overwritten(ow, k) ? _graph_copy(ins[k].val, false) : nothing,
            Val(N + 1))
        tape = (kept, copy(out.val))
    end
    func.val(f.val, nbands.val, out.val, _bands_values(rest)...)
    return AugmentedReturn(nothing, nothing, tape)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_RunBands}, ::Type{RT}, tape,
        f::Annotation, nbands::Annotation, out::Annotation{<:AbstractVector},
        rest::Vararg{Annotation, N}) where {RT, N}
    W = Val(EnzymeRules.width(config))
    ins = (f, rest...)
    d_outs = _lanes(out, W)
    d = if all(isnothing, d_outs)
        map(a -> _bands_zero_active(a, W), ins)
    else
        kept, before = tape
        vals = map((a, t) -> t === nothing ? a.val : t, ins, kept)
        mode = Enzyme.set_runtime_activity(Enzyme.ReverseSplitNoPrimal,
            EnzymeRules.runtime_activity(config))
        pe = _polyester_ext()
        after = copy(out.val)
        copyto!(out.val, before)
        try
            _bands_reverse!(pe._handed, pe._take, mode, nbands.val, ins, vals, out.val,
                d_outs, W)
        finally
            copyto!(out.val, after)
        end
    end
    return (first(d), nothing, nothing, Base.tail(d)...)
end
