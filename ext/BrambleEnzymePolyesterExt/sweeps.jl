# sweeps.jl: Enzyme rules for `_batch_for!` (src/utils/linear_algebra.jl), which writes
# `v[i] = f(i)` for `i` in `idxs`, `f` a closure the caller passes.
#
# Each rule cuts `idxs` into `Threads.nthreads()` slabs (`_rule_slab`) and runs one Enzyme
# derivative per slab, one slab per `@batch` task. Enzyme compiles a thunk on first use, so
# the thunk is built on the host before the launch. The task's arguments (the thunk, `v`,
# the closure and its shadows) are not isbits, and `@batch` would turn their arrays into
# `PtrArray`s that the thunk's signature rejects; they cross through `BramblePolyesterExt`'s
# typed slot (`_handed`, read back in the task with `_take`), as its whole kernels do. That
# extension is looked up when a rule runs (`_polyester_ext`): an extension cannot import
# another, and both load with Polyester.
#
# Forward: each task runs Enzyme's forward derivative of `_fill_slab!` on its slab, which
# writes the slab's `v` and its tangent. No two tasks write one entry.
#
# Reverse: each task runs Enzyme's reverse derivative of `_slab_dot(f, d_v, r)`, the sum of
# `f(i) * d_v[i]` over its slab `r`, with `d_v` constant: the gradient of that sum in the
# closure is the slab's pullback of `d_v`, and the pass never writes the user's `v`. Each
# slab then clears its own `d_v` entries, since `v` was overwritten. The closure's shadow
# (the shadow of every array it captures) is written by every slab, so each task
# accumulates into its own zeroed copy (`_graph_copy`, built on the host), and the host
# adds the copies into the closure's shadow after the join (`_accumulate!`). The copies
# cost one zeroed shadow per task and call, the reduction one pass over them; atomics would
# need Enzyme to emit them inside the user's closure, and a lock would serialise the slabs.
#
# `_sweep_for!` passes the kernel behind a `Ref`, since a by-value closure that mixes an array
# with a plain number fails Enzyme's calling convention at the hook. The rules carry the
# `Ref` through as the closure (its shadow a `Ref` of the closure's shadow), and the slab
# functions unwrap it (`_kernel`), so a captured `Float64` gets its derivative too.
#
# Under runtime activity, a captured array that is not differentiated arrives with the
# primal itself as its shadow. Its per-task copy is a fresh zero array, so the reduction
# skips any shadow array that is its own primal (`_accumulate!`), and the user's data is
# never written.
#
# Batch width: `v` and `f` arrive as `Duplicated` or `BatchDuplicated`. Forward builds one
# batched thunk, so each task runs every lane in one call. Reverse runs, in each task of one
# launch, a width-1 thunk per lane, since each lane seeds the slab with its own `d_v`; each
# task keeps a zeroed shadow copy per lane, and the host reduces each lane after the join.

const _BatchFor = typeof(Bramble._batch_for!)

@inline function _polyester_ext()
    pe = Base.get_extension(Bramble, :BramblePolyesterExt)
    pe === nothing && error("BramblePolyesterExt is not loaded, though Polyester is")
    return pe
end

# Slab `b` of `n` of `idxs`, cut by position: a sub-range of a range, a slab of a
# `CartesianIndices`' last axis, or a view of any other index vector.
@inline _rule_slab(idxs::AbstractRange, n::Int, b::Int) = _band_range(idxs, n, b)
@inline function _rule_slab(idxs::CartesianIndices, n::Int, b::Int)
    ax = idxs.indices
    return CartesianIndices((Base.front(ax)..., _band_range(last(ax), n, b)))
end
@inline function _rule_slab(idxs::AbstractVector, n::Int, b::Int)
    return view(idxs, (firstindex(idxs) - 1) .+ _band_range(Base.OneTo(length(idxs)), n, b))
end
@noinline function _rule_slab(idxs, n::Int, b::Int)
    throw(ArgumentError("the Enzyme rules for `_batch_for!` take a range, a \
        `CartesianIndices` or an index vector, not a $(typeof(idxs))"))
end

# The kernel itself, or the one `_sweep_for!` hands over behind a `Ref`.
@inline _kernel(f) = f
@inline _kernel(r::Base.RefValue) = r[]

@inline function _fill_slab!(v, f::F, r) where {F}
    k = _kernel(f)
    for i in r
        @inbounds v[i] = k(i)
    end
    return nothing
end

@inline function _slab_dot(f::F, d_v, r) where {F}
    k = _kernel(f)
    s = zero(eltype(d_v))
    for i in r
        s += k(i) * @inbounds(d_v[i])
    end
    return s
end

# Clears every lane's `d_v` over `idxs`, one `@batch` launch for all lanes.
@inline function _clear_lane!(d_v, r)
    d_v === nothing && return nothing
    for i in r
        @inbounds d_v[i] = zero(eltype(d_v))
    end
    return nothing
end

function _zero_slabs!(d_vs::Tuple, idxs)
    n = Threads.nthreads()
    @batch for b in 1:n
        _clear_band!(d_vs, _rule_slab(idxs, n, b))
    end
    return nothing
end
@inline function _clear_band!(d_vs::Tuple, r)
    map(d -> _clear_lane!(d, r), d_vs)
    return nothing
end

# What a task's exception leaves behind: the first one is rethrown on the host after the
# join, whatever task raised it.
@noinline function _rethrow_first(errs::Vector{Any})
    for e in errs
        e === nothing || throw(e)
    end
    error("a CpuPolyester derivative task failed without an exception")
end

# --- Forward ----------------------------------------------------------------------- #

struct _ForwardSlabs{TH, V, F, I}
    thunk::TH
    v::V
    f::F
    idxs::I
    errs::Vector{Any}
end

@noinline function _forward_slab_threw(p::_ForwardSlabs, n::Int, b::Int)
    try
        p.thunk(Const(_fill_slab!), p.v, p.f, Const(_rule_slab(p.idxs, n, b)))
    catch e
        p.errs[b] = e
        return true
    end
    return false
end

@noinline function _forward_run!(take::T, n::Int, h::H) where {T, H}
    failed = false
    @batch reduction=((|, failed),) for b in 1:n
        failed |= _forward_slab_threw(take(h), n, b)
    end
    return failed
end

function _batch_for_forward!(handed::HD, take::TK, mode::M, v::VA, idxs,
        f::FA) where {HD, TK, M, VA, FA}
    n = Threads.nthreads()
    R = typeof(_rule_slab(idxs, n, 1))
    thunk = Enzyme.autodiff_thunk(mode, Const{typeof(_fill_slab!)}, Const, VA, FA, Const{R})
    p = _ForwardSlabs(thunk, v, f, idxs, Vector{Any}(nothing, n))
    handed(_forward_run!, p, take, n) && _rethrow_first(p.errs)
    return nothing
end

function EnzymeRules.forward(config::FwdConfig, func::Const{_BatchFor}, ::Type{RT},
        v::Annotation{<:AbstractArray}, idxs::Annotation, f::Annotation) where {RT}
    _entered!(:_batch_for!)
    if v isa Const || f isa Const
        func.val(v.val, idxs.val, f.val)
        d_vs = _lanes(v, Val(EnzymeRules.width(config)))
        all(isnothing, d_vs) || _zero_slabs!(d_vs, idxs.val)
        return nothing
    end
    # A `BatchDuplicated` `v` and `f` make a batched thunk: each task runs every lane.
    mode = Enzyme.set_runtime_activity(Enzyme.Forward, EnzymeRules.runtime_activity(config))
    pe = _polyester_ext()
    _batch_for_forward!(pe._handed, pe._take, mode, v, idxs.val, f)
    return nothing
end

# --- Reverse ----------------------------------------------------------------------- #

# `dfs[b]` holds task `b`'s zeroed copy of each lane's closure shadow (`nothing` in a lane
# where the closure is inactive), `d_vs` each lane's `d_v`.
struct _ReverseSlabs{FW, RV, F, DFS, DVS, I}
    fwd::FW
    rev::RV
    f::F
    dfs::Vector{DFS}
    d_vs::DVS
    idxs::I
    errs::Vector{Any}
end

@inline _task_annotation(f, df) = Duplicated(f, df)
@inline _task_annotation(f::F, df::Base.RefValue{F}) where {F} = MixedDuplicated(f, df)

@inline function _reverse_lane!(p::_ReverseSlabs, d_v, df, r)
    d_v === nothing && return nothing
    if df !== nothing
        fa = _task_annotation(p.f, df)
        tape, _, _ = p.fwd(Const(_slab_dot), fa, Const(d_v), Const(r))
        p.rev(Const(_slab_dot), fa, Const(d_v), Const(r), one(eltype(d_v)), tape)
    end
    return _clear_lane!(d_v, r)
end

@noinline function _reverse_slab_threw(p::_ReverseSlabs, n::Int, b::Int)
    try
        r = _rule_slab(p.idxs, n, b)
        map((d_v, df) -> _reverse_lane!(p, d_v, df, r), p.d_vs, p.dfs[b])
    catch e
        p.errs[b] = e
        return true
    end
    return false
end

@noinline function _reverse_run!(take::T, n::Int, h::H) where {T, H}
    failed = false
    @batch reduction=((|, failed),) for b in 1:n
        failed |= _reverse_slab_threw(take(h), n, b)
    end
    return failed
end

# Every lane runs in each task of one `@batch` launch, through a width-1 thunk per lane:
# a lane's seed is its own `d_v`, so the lanes share no batched call.
function _batch_for_reverse!(handed::HD, take::TK, mode::M, d_vs, idxs, fp::F, f0::F,
        dfl, ::Type{FA}) where {HD, TK, M, F, FA}
    n = Threads.nthreads()
    R = typeof(_rule_slab(idxs, n, 1))
    DV = typeof(something(d_vs...))
    fwd, rev = Enzyme.autodiff_thunk(mode, Const{typeof(_slab_dot)}, Active, FA, Const{DV},
        Const{R})
    dfs = [map(df -> df === nothing ? nothing : _graph_copy(df, true), dfl) for _ in 1:n]
    p = _ReverseSlabs(fwd, rev, fp, dfs, d_vs, idxs, Vector{Any}(nothing, n))
    handed(_reverse_run!, p, take, n) && _rethrow_first(p.errs)
    for task in dfs
        map((df, d) -> df === nothing ? nothing : _accumulate!(df, d, f0), dfl, task)
    end
    return nothing
end

# --- Shadow copies and their reduction ---------------------------------------------- #
#
# `Enzyme.make_zero`, `Base.deepcopy` and a recursive reduction all recurse once per object,
# so a closure holding a long linked structure overflows the stack (a 200_000-node list
# does). These walk with an explicit work list instead. Only a chain of immutable values
# nested by reference still recurses, one level per value.

# Values a copy keeps as they are: they hold no floating-point data of the closure's.
const _Opaque = Union{Symbol, String, Type, Module, Method, Core.MethodInstance, Task, IO,
    Base.AbstractLock}
const _Buffer = Union{Array, GenericMemory}

# A copy of `x` with its sharing and cycles kept: every float zeroed when `zero` (a shadow's
# zeroed copy, as `Enzyme.make_zero` builds), every value copied otherwise (as `deepcopy`).
# A mutable object is allocated before its fields are filled (`work`), so a cycle through it
# closes on the copy (`seen`).
function _graph_copy(x, zero::Bool)
    seen, work = IdDict{Any, Any}(), Any[]
    y = _copy_value(x, zero, seen, work)
    while !isempty(work)
        _fill_copy!(pop!(work)..., zero, seen, work)
    end
    return y
end

function _copy_value(x, zero::Bool, seen, work)
    T = typeof(x)
    isbitstype(T) && return zero ? _zero_bits(x) : x
    x isa _Opaque && return x
    if ismutabletype(T)
        haskey(seen, x) && return seen[x]
        y = _shell(x, zero)
        seen[x] = y
        y === x || push!(work, (x, y))
        return y
    end
    n = nfields(x)
    all(i -> isdefined(x, i), 1:n) || return x
    vals = Any[_copy_value(getfield(x, i), zero, seen, work) for i in 1:n]
    return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, vals, n)::T
end

# The copy of a mutable `x` before its contents are filled: a buffer of plain elements
# comes back whole (zeroed or copied, or `x` itself when it holds no floats to zero).
function _shell(x::_Buffer, zero::Bool)
    E = eltype(x)
    isbitstype(E) || return similar(x)
    zero || return copy(x)
    _has_float(E) || return x
    y = similar(x)
    for i in eachindex(x)
        @inbounds y[i] = _zero_bits(x[i])
    end
    return y
end
_shell(x::T, zero::Bool) where {T} = ccall(:jl_new_struct_uninit, Any, (Any,), T)::T

function _fill_copy!(x::_Buffer, y, zero::Bool, seen, work)
    isbitstype(eltype(x)) && return nothing
    for i in eachindex(x)
        isassigned(x, i) && (y[i] = _copy_value(x[i], zero, seen, work))
    end
    return nothing
end
function _fill_copy!(x, y, zero::Bool, seen, work)
    for i in 1:nfields(x)
        isdefined(x, i) || continue
        v = _copy_value(getfield(x, i), zero, seen, work)
        ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i - 1, v)
    end
    return nothing
end

# A plain value with its floats zeroed; its other fields kept.
_zero_bits(x::Union{AbstractFloat, Complex}) = zero(x)
function _zero_bits(x::T) where {T}
    (fieldcount(T) == 0 || !_has_float(T)) && return x
    return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T,
        Any[_zero_bits(getfield(x, i)) for i in 1:fieldcount(T)], fieldcount(T))::T
end
_has_float(::Type{<:Union{AbstractFloat, Complex}}) = true
_has_float(::Type{T}) where {T} = any(_has_float, fieldtypes(T))

# Adds the per-task shadow `src` into the closure's shadow `dst`, `primal` the closure itself,
# with a work list as `_graph_copy` walks. Floats add where they are stored mutably: in a
# buffer, in a mutable object's field, or, for an immutable value held in such a field, by
# rebuilding it (`_add`). A shadow that is its own primal is skipped (runtime activity's
# inactive data), and an object shared in the shadow is added once (`seen`).
function _accumulate!(dst, src, primal)
    # A `MixedDuplicated` shadow is a `Ref` to the closure's shadow; match its primal to it.
    q = dst isa Base.RefValue && !(primal isa Base.RefValue) ? Ref(primal) : primal
    seen, work = Base.IdSet{Any}(), Any[(dst, src, q)]
    while !isempty(work)
        _accumulate_one!(pop!(work)..., seen, work)
    end
    return nothing
end

function _accumulate_one!(d::T, s, q, seen, work) where {T}
    s isa T || return nothing
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
                @inbounds d[i] = _add(d[i], s[i])
            end
        else
            for i in eachindex(d)
                (isassigned(d, i) && isassigned(s, i)) || continue
                qi = q isa T && isassigned(q, i) ? q[i] : nothing
                push!(work, (d[i], s[i], qi))
            end
        end
        return nothing
    end
    for i in 1:nfields(d)
        (isdefined(d, i) && isdefined(s, i)) || continue
        di, si = getfield(d, i), getfield(s, i)
        qi = q isa T && isdefined(q, i) ? getfield(q, i) : nothing
        if ismutabletype(T) && !ismutable(di)
            # An immutable value stored in a mutable object: its floats add by rebuilding it.
            new = _add(di, si)
            new === di || ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), d, i - 1, new)
            di = new
        end
        isbits(di) || push!(work, (di, si, qi))
    end
    return nothing
end

# `x + y` over the float fields of an immutable value, every other field kept from `x`.
_add(x, y) = x
_add(x::T, y::T) where {T <: Union{AbstractFloat, Complex}} = x + y
function _add(x::T, y::T) where {T}
    (isbitstype(T) || !ismutabletype(T)) && fieldcount(T) > 0 || return x
    return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T,
        Any[_add(getfield(x, i), getfield(y, i)) for i in 1:fieldcount(T)], fieldcount(T))::T
end

# The closure route assumes two things of its callers, both true of every `_sweep_for!` call
# (`project!`, `backend_profile.jl`):
#
# - `idxs` holds each index once. A repeated index writes `v` twice in the primal, but its
#   cotangent is pulled back once per occurrence in reverse.
# - Differentiation sets `Enzyme.set_runtime_activity`, as `Rₕ` under `CpuSerial` needs: a
#   closure capturing a constant array otherwise throws `EnzymeRuntimeActivityError`.
#
# The reverse pass evaluates the closure again, so it must see the closure's captured data as
# it was before this call. Two things can change it: later code (`overwritten`, which counts
# `func` first, so `f` is entry 4), and this very call, when the closure reads `v`
# (`i -> v[i]^2`) by any route: a capture, a view, a `Dict`, a `Memory`, a global. No walk of
# the closure finds every route, so whenever `v` carries a shadow its values are kept from
# before the primal (one copy of `v` per call), and put back in `v` for the reverse. A closure
# whose data may change later is copied whole, before the primal too.
function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_BatchFor},
        ::Type{RT}, v::Annotation{<:AbstractArray}, idxs::Annotation,
        f::Annotation) where {RT}
    _entered!(:_batch_for!)
    shadowed = !all(isnothing, _lanes(v, Val(EnzymeRules.width(config))))
    f_tape = !(f isa Const) && EnzymeRules.overwritten(config)[4] ?
             _graph_copy(f.val, false) : nothing
    v_tape = shadowed ? copy(v.val) : nothing
    func.val(v.val, idxs.val, f.val)
    return AugmentedReturn(nothing, nothing, (f_tape, v_tape))
end

# Runs `run!()` with `v` holding `before`, its values when the primal began, and puts the
# primal's values back after, whatever `run!` does.
function _with_values(run!::R, v, before) where {R}
    after = copy(v)
    copyto!(v, before)
    try
        run!()
    finally
        copyto!(v, after)
    end
    return nothing
end

# The closure's shadow in lane `w`: `nothing` where it is inactive, the shadow `Ref` for an
# immutable closure with active fields (`MixedDuplicated`).
@inline _closure_shadow(f::Const, w::Int) = nothing
@inline _closure_shadow(f, w::Int) = _shadow(f, w)
@inline _closure_shadow(f::MixedDuplicated, w::Int) = f.dval
@inline _closure_shadow(f::BatchMixedDuplicated, w::Int) = f.dval[w]
@noinline function _closure_shadow(f::Active, w::Int)
    throw(ArgumentError("the Enzyme reverse rule for `_batch_for!` under CpuPolyester takes \
        a closure that captures an array, not an active $(typeof(f.val)) of plain numbers"))
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_BatchFor}, ::Type{RT}, tape,
        v::Annotation{<:AbstractArray}, idxs::Annotation, f::Annotation) where {RT}
    W = Val(EnzymeRules.width(config))
    d_vs = _lanes(v, W)
    all(isnothing, d_vs) && return (nothing, nothing, nothing)
    dfl = ntuple(w -> _closure_shadow(f, w), W)
    if all(isnothing, dfl)
        _zero_slabs!(d_vs, idxs.val)
    else
        f_tape, v_tape = tape
        fp = f_tape === nothing ? f.val : f_tape
        FA = f isa Union{MixedDuplicated, BatchMixedDuplicated} ?
             MixedDuplicated{typeof(fp)} : Duplicated{typeof(fp)}
        mode = Enzyme.set_runtime_activity(Enzyme.ReverseSplitNoPrimal,
            EnzymeRules.runtime_activity(config))
        pe = _polyester_ext()
        run!() = _batch_for_reverse!(pe._handed, pe._take, mode, d_vs, idxs.val, fp, f.val,
            dfl, FA)
        v_tape === nothing ? run!() : _with_values(run!, v.val, v_tape)
    end
    return (nothing, nothing, nothing)
end
