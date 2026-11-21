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
#   across band edges, so each task works on its own copy of it. The copy starts as the
#   shadow itself, because a kernel may write an argument besides `out` (`jacobian!`'s
#   coefficient values), and its band then reads that cotangent at the points it writes
#   and clears it there. After the join each entry gains every copy's change from a
#   snapshot of the shadow taken before the tasks ran (`_bands_merge`), under `@batch` for
#   an array, by `_bands_merge!` otherwise. An entry no task changed keeps its value. A
#   shadow that is its own primal (runtime activity's inactive data) gets no copy and
#   reaches the tasks as `Const`. An `Active` argument (a scalar weight) has its per-task
#   derivatives summed on the host.
#
# The reverse pass replays each band's primal on `out` as it was before the call (the
# `+=` engines would otherwise add twice, and the in-place averages read `out` itself), so
# `out`'s values are kept from before the primal and put back for the reverse. Lanes run in
# turn. From width 2 on, every lane after the first replays on a copy of `out` taken before
# the call and made once on the host, since `out` itself holds the previous lane's writes by
# then and a kernel whose derivative reads `out` would differentiate those. For the same
# reason each such lane replays on its own copy of the arguments too. The kernel
# and every other argument, `Const` ones too since the reverse reads them, are copied whole
# before the primal, in one walk so that arrays they share stay shared, whenever `out`
# carries a shadow. Their `overwritten` entries do not say whether later code writes the
# data they hold: the kernel arrives behind a fresh `Ref`, and a closure or a `StarSpacings`
# passed as an argument reaches its arrays by reference.
#
# `_run_bands!` hands the kernel behind a `Ref`, and every argument that may mix arrays with
# plain numbers behind a `Ref` of a `_BandArg` (src/operators/vector_calculus.jl), for the
# calling-convention reason `_sweep_for!` does; the task unwraps them (`_kernel`,
# `_band_unbox`). Reverse hands the band thunks, by value as a `MixedDuplicated` whose
# shadow is that `Ref`, the kernel and each `_BandArg` whose value mixes arrays with active
# numbers (`_slab_primal`, sweeps.jl, `_bands_primal`); every other input, a caller's own
# `Ref` included, reaches them as the rule got it. Behind the `Ref`, the thunks gave a
# captured `Float32` a zero derivative; passed by value at the hook, a closure argument
# lost it too, and a batched call stopped Enzyme.

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
    _kernel(f)(out, map(Bramble._band_unbox, rest)..., n, b)
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
@inline function _bands_shadow(a, w::Int)
    a isa Active && return nothing
    s = _shadow(a, w)
    return s === a.val ? nothing : s
end
@inline _bands_shadow(a::MixedDuplicated, w::Int) = a.dval
@inline _bands_shadow(a::BatchMixedDuplicated, w::Int) = a.dval[w]

# A task's annotation of an input, given its per-task shadow copy `c`.
@inline _bands_annotation(a, v, ::Nothing) = a isa Active ? Active(v) : Const(v)
@inline _bands_annotation(a, v, c) = Duplicated(v, c)
@inline function _bands_annotation(
        a::Union{MixedDuplicated, BatchMixedDuplicated}, v, c::Base.RefValue)
    return MixedDuplicated(v, c)
end
# The kernel's `Ref` unwrapped by `_slab_primal`: the kernel by value, its shadow the `Ref`.
@inline function _bands_annotation(
        a::Union{Duplicated{<:Base.RefValue{K}}, BatchDuplicated{<:Base.RefValue{K}}},
        v::K, c::Base.RefValue{K}) where {K}
    return MixedDuplicated(v, c)
end

# One lane: its thunks, its `d_out`, and `copies[b]`, task `b`'s copies of the lane's
# input shadows (`nothing` for an input with none). `act[b]` holds what task `b`'s reverse
# returned, read for the `Active` inputs.
struct _BandsLane{FW, RV, DO, C}
    fwd::FW
    rev::RV
    d_out::DO
    copies::Vector{C}
    act::Vector{Any}
end

# `ins` the inputs `(f, rest...)` as annotated, `vals` their values for each lane's replay,
# `outs` the `out` each lane's replay writes, or the one `out` at width 1.
struct _BandsReverse{A, V, O, L}
    ins::A
    vals::V
    outs::O
    lanes::L
    errs::Vector{Any}
end

@inline function _bands_lane!(
        p::_BandsReverse, lane::_BandsLane, vals, out, n::Int, b::Int)
    anns = map(_bands_annotation, p.ins, vals, lane.copies[b])
    fa, ra, oa = first(anns), Base.tail(anns), Duplicated(out, lane.d_out)
    tape, _, _ = lane.fwd(Const(_bands_task!), fa, oa, Const(n), Const(b), ra...)
    r = lane.rev(Const(_bands_task!), fa, oa, Const(n), Const(b), ra..., tape)
    lane.act[b] = first(r)
    return nothing
end

# Every lane of task `b`, in turn: on `outs` itself at width 1, else each on its own.
@inline function _bands_lanes!(p::_BandsReverse, outs::AbstractArray, n::Int, b::Int)
    map((lane, v) -> lane === nothing ? nothing : _bands_lane!(p, lane, v, outs, n, b),
        p.lanes, p.vals)
end
@inline function _bands_lanes!(p::_BandsReverse, outs::Tuple, n::Int, b::Int)
    map((lane, v, o) -> lane === nothing ? nothing : _bands_lane!(p, lane, v, o, n, b),
        p.lanes, p.vals, outs)
end

@noinline function _bands_reverse_threw(p::_BandsReverse, n::Int, b::Int)
    try
        _bands_lanes!(p, p.outs, n, b)
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

# `x` plus every copy's change from `x0`. `x` is one entry of a shadow, `x0` that entry
# before the tasks ran, `ys` the entry in each per-task copy (each began as `x0`). A copy
# equal to `x0` adds nothing. While the sum still equals `x0`, a changed copy is taken
# whole, so an entry one task changed gets exactly that task's value. Changes count from
# `x0`, not `x`: a shadow two inputs share is merged once per input, and `d_out`, reached
# through an argument, has been cleared by the tasks. Tuples and immutable structs go field
# by field; fields that are not floats stay as in `x`.
@inline function _bands_merge(x::T, x0::T, ys) where {T <: Union{AbstractFloat, Complex}}
    acc = x
    for y in ys
        y::T == x0 && continue
        acc = acc == x0 ? y : acc + (y - x0)
    end
    return acc
end
function _bands_merge(x::Tuple, x0::Tuple, ys)
    return ntuple(i -> _bands_merge(x[i], x0[i], (y[i] for y in ys)), length(x))
end
function _bands_merge(x::T, x0::T, ys) where {T}
    (isbitstype(T) || !ismutabletype(T)) && fieldcount(T) > 0 || return x
    vals = Any[_bands_merge(getfield(x, i), getfield(x0, i), (getfield(y, i) for y in ys))
               for i in 1:fieldcount(T)]
    return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, vals, fieldcount(T))::T
end
_bands_merge(x, x0, ys) = x

# The per-task copies `cs` of a shadow `d` added into it: an array of numbers under
# `@batch`, each band summing its own indices over the copies; anything else by
# `_accumulate!`, which skips data that is its own primal (`primal` the input as the caller
# passed it). For copies that started at zero (operators.jl's `_act_add_input!`).
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

# The per-task copies `cs` of a shadow `d` merged into it, `d0` a copy of `d` taken before
# the tasks ran: an array of numbers under `@batch`, each band merging its own indices;
# anything else by `_bands_merge!`, which skips data that is its own primal (`primal` the
# input as the caller passed it).
struct _BandsMerge{D, C}
    d::D
    d0::D
    cs::C
end

@inline function _bands_merge_band!(p::_BandsMerge, n::Int, b::Int)
    d, d0 = p.d, p.d0
    for i in _band_range(Base.OneTo(length(d)), n, b)
        @inbounds d[i] = _bands_merge(d[i], d0[i], (c[i] for c in p.cs))
    end
    return nothing
end

@noinline function _bands_merge_run!(take::T, n::Int, h::H) where {T, H}
    @batch for b in 1:n
        _bands_merge_band!(take(h), n, b)
    end
    return false
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

function _bands_merge_into!(handed::HD, take::TK,
        d::Array{<:Union{AbstractFloat, Complex}}, d0, cs, primal) where {HD, TK}
    handed(_bands_merge_run!, _BandsMerge(d, d0, cs), take, Threads.nthreads())
    return nothing
end
function _bands_merge_into!(handed::HD, take::TK, d, d0, cs, primal) where {HD, TK}
    _bands_merge!(d, d0, cs, primal)
    return nothing
end

# `_bands_merge` over a shadow's object graph, walked as `_accumulate!` (sweeps.jl) walks
# it: with a work list, `d`, its snapshot `d0` and its copies `cs` in step, an object shared
# in `d` merged once, a shadow that is its own primal skipped. Floats merge where they are
# stored mutably: in a buffer, or in a mutable object's field, an immutable value there
# rebuilt.
function _bands_merge!(d, d0, cs, primal)
    q = d isa Base.RefValue && !(primal isa Base.RefValue) ? Ref(primal) : primal
    seen, work = Base.IdSet{Any}(), Any[(d, d0, cs, q)]
    while !isempty(work)
        _bands_merge_one!(pop!(work)..., seen, work)
    end
    return nothing
end

function _bands_merge_one!(d::T, d0, cs, q, seen, work) where {T}
    (d0 isa T && all(c -> c isa T, cs)) || return nothing
    (isbitstype(T) || d isa _Opaque || d === q) && return nothing
    if ismutabletype(T)
        d in seen && return nothing
        push!(seen, d)
    end
    if d isa _Buffer
        E = eltype(d)
        if isbitstype(E)
            _has_float(E) || return nothing
            for i in eachindex(d)
                @inbounds d[i] = _bands_merge(d[i], d0[i], (c[i] for c in cs))
            end
        else
            for i in eachindex(d)
                (isassigned(d, i) && isassigned(d0, i) && all(c -> isassigned(c, i), cs)) ||
                    continue
                qi = q isa T && isassigned(q, i) ? q[i] : nothing
                push!(work, (d[i], d0[i], [c[i] for c in cs], qi))
            end
        end
        return nothing
    end
    for i in 1:nfields(d)
        (isdefined(d, i) && isdefined(d0, i) && all(c -> isdefined(c, i), cs)) || continue
        di, d0i, ci = getfield(d, i), getfield(d0, i), [getfield(c, i) for c in cs]
        qi = q isa T && isdefined(q, i) ? getfield(q, i) : nothing
        if ismutabletype(T) && !ismutable(di)
            new = _bands_merge(di, d0i, ci)
            new === di ||
                ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), d, i - 1, new)
            di = new
        end
        isbits(di) || push!(work, (di, d0i, ci, qi))
    end
    return nothing
end

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

# The `out` each lane's replay writes: `out` itself for the first lane with a cotangent,
# a copy of `before` for each later one.
function _bands_outs(out, before, d_outs::Tuple)
    live = findfirst(!isnothing, d_outs)
    return map(d_outs, ntuple(identity, Val(length(d_outs)))) do d, w
        return (d === nothing || w == live) ? out : copyto!(_asm_scratch(out), before)
    end
end

# A rest argument as the thunks take it: `_slab_primal` of a `_band_box` box, else as is.
@inline _bands_primal(v) = v
@inline _bands_primal(r::Base.RefValue{<:Bramble._BandArg}) = _slab_primal(r)

# `out` holds its values from before the call, kept in `before` too for the later lanes.
# The kernel and the boxed arguments reach the thunks as the header says.
function _bands_reverse!(handed::HD, take::TK, mode::M, n::Int, ins, vals0, out, before,
        d_outs, V::Val{W}) where {HD, TK, M, W}
    vals = ntuple(V) do w
        v = w == 1 ? vals0 : _graph_copy(vals0, false)::typeof(vals0)
        return (_slab_primal(first(v)), map(_bands_primal, Base.tail(v))...)
    end
    lanes = ntuple(V) do w
        d_outs[w] === nothing && return nothing
        shadows = map(a -> _bands_shadow(a, w), ins)
        copies = [map(s -> s === nothing ? nothing : _graph_copy(s, false), shadows)
                  for _ in 1:n]
        anns = map(_bands_annotation, ins, vals[w], first(copies))
        fwd, rev = Enzyme.autodiff_thunk(mode, Const{typeof(_bands_task!)}, Const,
            typeof(first(anns)), Duplicated{typeof(out)}, Const{Int}, Const{Int},
            map(typeof, Base.tail(anns))...)
        return _BandsLane(fwd, rev, d_outs[w], copies, Vector{Any}(nothing, n))
    end
    outs = W == 1 ? out : _bands_outs(out, before, d_outs)
    pres = ntuple(V) do w
        lanes[w] === nothing && return nothing
        shadows = map(a -> _bands_shadow(a, w), ins)
        return map(s -> s === nothing ? nothing : _graph_copy(s, false), shadows)
    end
    p = _BandsReverse(ins, vals, outs, lanes, Vector{Any}(nothing, n))
    handed(_bands_reverse_run!, p, take, n) && _bands_rethrow(p.errs)
    for w in 1:W
        lane = lanes[w]
        lane === nothing && continue
        for k in eachindex(ins)
            s = _bands_shadow(ins[k], w)
            s === nothing || _bands_merge_into!(handed, take, s, pres[w][k],
                [c[k] for c in lane.copies], ins[k].val)
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
        vals = (f.val, _bands_values(rest)...)
        tape = (_graph_copy(vals, false)::typeof(vals), copy(out.val))
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
        vals, before = tape
        mode = Enzyme.set_runtime_activity(Enzyme.ReverseSplitNoPrimal,
            EnzymeRules.runtime_activity(config))
        pe = _polyester_ext()
        after = copy(out.val)
        copyto!(out.val, before)
        try
            _bands_reverse!(pe._handed, pe._take, mode, nbands.val, ins, vals, out.val,
                before, d_outs, W)
        finally
            copyto!(out.val, after)
        end
    end
    return (first(d), nothing, nothing, Base.tail(d)...)
end
