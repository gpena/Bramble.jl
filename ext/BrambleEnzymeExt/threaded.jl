# threaded.jl: Enzyme rules for the `CpuThreaded` engine entries (gpena/Bramble.jl#369).
#
# Enzyme cannot differentiate a `Threads.@threads` loop on Julia 1.13.3: every Enzyme
# release tried throws `AssertionError: mi2 !== nothing` there
# (docs/notes/internals/enzyme_threaded.md). Each `CpuThreaded` engine entry, the function
# whose only work is that loop (`_static_bands!`, `_static_for!`, `_static_scatter_for!`,
# `_static_partials!`, the three bilinear and linear colour runners, `_kron_fused!` and the
# `Threads.@spawn` `_mf_run_bands!`), gets rules here whose own derivative loops run under
# `Threads.@threads :static`. No derivative loop calls a `_serial_*` engine or
# `_static_or_serial`; a rule's primal is the entry's own call.
#
# Two routes, as `BrambleEnzymePolyesterExt` takes them for its `@batch` hooks:
#
# - Fixed kernels: `_static_bands!` running `_difference_band!`, `_average_band!` or
#   `_centered_average_band!`. Each is linear in `in_ref`, so forward runs the same band
#   kernel on the tangents, and reverse runs its gather-form adjoint
#   (`_difference_band_adjoint!`, `_average_band_adjoint!`,
#   `_centered_average_band_adjoint!`, src/operators), each `d_in[j]` written by the one
#   band that owns `j`. `d_out` is cleared in a second loop, since a band reads its
#   neighbour's `d_out` across the band edge.
# - Closure kernels: every other entry. Its loop body is a task `task(n, b, args...)`, `b`
#   one of `n` tasks, `args` the entry's own arguments. Forward runs Enzyme's forward
#   derivative of each task, one task per thread, on the caller's annotations: a task writes
#   its tangents where its primal writes, so the tasks race no more than the primal does.
#   Reverse runs Enzyme's split reverse of each task, every lane through its own width-1
#   thunk. The written array (`out`) keeps one shadow, shared by the tasks, since its
#   entries are split between them as the primal's are; every other shadow is read across
#   task edges, so each task gathers into its own zeroed copy, added into the caller's
#   after the join. An `Active` argument's per-task derivatives are summed on the host.
#
# A task replays its primal in reverse, on a copy of every argument's values taken before
# the primal, in one walk so that arguments sharing an array still share it. The replay
# never writes the caller's arrays (an accumulating sweep would otherwise add twice), and
# reads what the primal read even when later code overwrites data an argument holds by
# reference.
#
# The ghost, shift and multigrid band engines get their rule on `_run_bands!` itself, above
# `_static_or_serial`, as `_kron_fused!` does; inside a user's threaded region those rules'
# derivative takes the primal's own branch, the tasks in order (`_each_task`).
#
# Every rule takes `Duplicated` or `BatchDuplicated` arguments, the width from
# `EnzymeRules.width(config)`: DI's sparse Jacobians seed several colours per pass. A
# forward thunk is batched, so each task runs every lane in one call; reverse runs every
# lane in each task of one `Threads.@threads` loop.
#
# The shadow-copy helpers (`_graph_copy`, `_accumulate!`, `_add`, `_active_sum`) repeat
# those of ext/BrambleEnzymePolyesterExt/sweeps.jl and engines.jl, since an extension
# cannot load another; these add `keep`, the shadows a copy shares rather than zeroes.
#
# Accepted limits, as for the `CpuPolyester` rules: the closure route needs
# `Enzyme.set_runtime_activity`, and a closure that reaches `out` only through a global gets
# no gradient through that path.

const _StaticBands = typeof(Bramble._static_bands!)
const _StaticFor = typeof(Bramble._static_for!)
const _StaticScatter = typeof(Bramble._static_scatter_for!)
const _StaticPartials = typeof(Bramble._static_partials!)
const _StaticBandColour = typeof(Bramble._static_band_colour!)
const _StaticColour = typeof(Bramble._static_colour!)
const _StaticCutBandColour = typeof(Bramble._static_cut_band_colour!)
const _KronFused = typeof(Bramble._kron_fused!)
const _ThreadedKron = Annotation{<:KroneckerLinearOperator{<:Any, <:Any, <:Tuple, CpuThreaded}}

# How many times each entry's rule has been entered (forward, or the augmented primal of
# reverse). test/ext/enzyme_policy_cases.jl reads it to tell that the rule ran.
const _ENTRIES = Dict{Symbol, Int}()
const _ENTRIES_LOCK = ReentrantLock()
function _entered!(hook::Symbol)
    @lock _ENTRIES_LOCK _ENTRIES[hook] = get(_ENTRIES, hook, 0) + 1
    return nothing
end
_rule_entries(hook::Symbol) = @lock _ENTRIES_LOCK get(_ENTRIES, hook, 0)

# The shadow of `x` in lane `w`, or `nothing` where `x` has none to write: a `Const`, an
# `Active` value, or under runtime activity a shadow that is the primal itself. An immutable
# value with active fields (`MixedDuplicated`) has its shadow behind a `Ref`.
@inline _shadow(x::Union{Const, Active}, w::Int) = nothing
@inline function _shadow(x::Union{Duplicated, DuplicatedNoNeed}, w::Int)
    return x.dval === x.val ? nothing : x.dval
end
@inline function _shadow(x::Union{BatchDuplicated, BatchDuplicatedNoNeed}, w::Int)
    d = x.dval[w]
    return d === x.val ? nothing : d
end
@inline _shadow(x::MixedDuplicated, w::Int) = x.dval
@inline _shadow(x::BatchMixedDuplicated, w::Int) = x.dval[w]

@inline _lanes(x, ::Val{W}) where {W} = ntuple(w -> _shadow(x, w), Val(W))

# --- Fixed kernels: the difference and average bands --------------------------------- #

# The linear indices of band `b`'s slab, the block a band kernel writes: a cut of the last
# axis of a column-major grid is one contiguous block.
@inline function _slab_block(dims::Tuple, n::Int, b::Int)
    stride = prod(Base.front(dims))
    band = _band_range(Base.OneTo(last(dims)), n, b)
    return ((first(band) - 1) * stride + 1):(last(band) * stride)
end

@inline function _zero_slab!(d_out, dims, n, b)
    for i in _slab_block(dims, n, b)
        @inbounds d_out[i] = zero(eltype(d_out))
    end
    return nothing
end

@inline function _zero_lane!(d_out, n, b)
    d_out === nothing && return nothing
    for i in _band_range(Base.OneTo(length(d_out)), n, b)
        @inbounds d_out[i] = zero(eltype(d_out))
    end
    return nothing
end

# Clears every lane's `d_out`, one band per thread.
function _zero_outs!(d_outs::Tuple, n::Int)
    _each_task(n) do b
        foreach(d -> _zero_lane!(d, n, b), d_outs)
    end
    return nothing
end

# Where a spacing cotangent gathers: `d_h` itself, or a scratch vector for a `StarSpacings`,
# whose entries are not stored.
@inline _spacing_target(::Nothing) = nothing
@inline _spacing_target(d_h::AbstractVector) = d_h
@inline _spacing_target(d_h::StarSpacings) = zeros(eltype(d_h.h), length(d_h))
@noinline function _spacing_target(d_h)
    throw(ArgumentError("the Enzyme rule for `_static_bands!` differentiates a vector \
        spacing, not a $(typeof(d_h)); pass the spacing as a vector"))
end

@inline _copy_spacing(h) = h
@inline _copy_spacing(h::AbstractVector) = copy(h)
@inline _copy_spacing(h::StarSpacings) = StarSpacings(copy(h.h))

# One lane's difference tangent on band `b`: `A(h)` applied to the tangent of `in_ref` (zero
# where `in_ref` is inactive), plus the term through the spacing tangent.
@inline function _difference_tangent!(d_out, d_in, d_h, in_ref, h, dims, dir, dim_val, n, b)
    d_out === nothing && return nothing
    if d_in === nothing
        _zero_slab!(d_out, dims, n, b)
    else
        Bramble._difference_band!(d_out, d_in, h, dims, dir, dim_val, n, b)
    end
    d_h === nothing ||
        _difference_band_spacing_tangent!(d_out, in_ref, h, d_h, dims, dir, dim_val, n, b)
    return nothing
end

@inline function _difference_adjoint!(d_out, d_in, d_hv, in_ref, h, dims, dir, dim_val, n, b)
    d_out === nothing && return nothing
    d_in === nothing || _difference_band_adjoint!(d_in, d_out, h, dims, dir, dim_val, n, b)
    d_hv === nothing ||
        _difference_spacing_adjoint!(d_hv, in_ref, d_out, h, dims, dir, dim_val, n, b)
    return nothing
end

@inline _fold_spacing!(d_h, d_hv, n, b) = nothing
@inline _fold_spacing!(d_h::StarSpacings, d_hv, n, b) = _star_spacings_adjoint!(
    d_h.h, d_hv, n, b)

function _difference_forward!(d_outs, d_ins, d_hs, in_ref, h, dims, dir, dim_val, n::Int)
    _each_task(n) do b
        foreach((o, i, s) -> _difference_tangent!(o, i, s, in_ref, h, dims, dir, dim_val, n, b),
            d_outs, d_ins, d_hs)
    end
    return nothing
end

# The gathers, the `StarSpacings` fold, then `d_out` cleared: three loops, every lane in each.
function _difference_reverse!(d_outs, d_ins, d_hs, in_ref, h, dims, dir, dim_val, n::Int)
    d_hvs = map(_spacing_target, d_hs)
    _each_task(n) do b
        foreach((o, i, s) -> _difference_adjoint!(o, i, s, in_ref, h, dims, dir, dim_val, n, b),
            d_outs, d_ins, d_hvs)
    end
    if any(d -> d isa StarSpacings, d_hs)
        _each_task(n) do b
            foreach((s, v) -> _fold_spacing!(s, v, n, b), d_hs, d_hvs)
        end
    end
    _zero_outs!(d_outs, n)
    return nothing
end

# An average band's tangent: the band kernel on the tangent of `in_ref`, or zero over the
# slab where `in_ref` is inactive. `extra` is `(dir, dim_val)` or `(dim_val,)`.
@inline function _average_tangent!(band!::K, d_out, d_in, dims, extra, n, b) where {K}
    d_out === nothing && return nothing
    d_in === nothing && return _zero_slab!(d_out, dims, n, b)
    band!(d_out, d_in, dims, extra..., n, b)
    return nothing
end

function _average_forward!(band!::K, d_outs, d_ins, dims, extra, n::Int) where {K}
    _each_task(n) do b
        foreach((o, i) -> _average_tangent!(band!, o, i, dims, extra, n, b), d_outs, d_ins)
    end
    return nothing
end

function _average_reverse!(adj!::K, d_outs, d_ins, dims, extra, n::Int) where {K}
    _each_task(n) do b
        foreach(d_outs, d_ins) do d_out, d_in
            (d_out === nothing || d_in === nothing) && return nothing
            adj!(d_in, d_out, dims, extra..., n, b)
            return nothing
        end
    end
    _zero_outs!(d_outs, n)
    return nothing
end

const _DifferenceBand = Const{typeof(Bramble._difference_band!)}
const _AverageBand = Const{typeof(Bramble._average_band!)}
const _CenteredBand = Const{typeof(Bramble._centered_average_band!)}
const _FixedBand = Union{_DifferenceBand, _AverageBand, _CenteredBand}

_vals(args::Tuple) = map(a -> a.val, args)

function _fixed_forward(config, ::_DifferenceBand, n::Int, args)
    out, in_ref, h, dims, dir, dim_val = args
    W = Val(EnzymeRules.width(config))
    d_outs = _lanes(out, W)
    all(isnothing, d_outs) && return nothing
    _difference_forward!(d_outs, _lanes(in_ref, W), _lanes(h, W), in_ref.val, h.val,
        dims.val, dir.val, dim_val.val, n)
    return nothing
end

function _fixed_forward(config, f::Union{_AverageBand, _CenteredBand}, n::Int, args)
    out, in_ref, dims = args[1], args[2], args[3]
    W = Val(EnzymeRules.width(config))
    d_outs = _lanes(out, W)
    all(isnothing, d_outs) && return nothing
    _average_forward!(f.val, d_outs, _lanes(in_ref, W), dims.val, _vals(Base.tail(Base.tail(Base.tail(args)))), n)
    return nothing
end

# The difference tape: `in_ref`'s values when the spacing cotangent will read them, and the
# spacing's, when later code overwrites them. `overwritten` counts `func` first, then `f`
# and `nbands`, so `in_ref` is entry 5 and `h` entry 6.
function _fixed_tape(config, ::_DifferenceBand, args)
    _, in_ref, h = args
    ow = EnzymeRules.overwritten(config)
    in_tape = ow[5] && !(h isa Const) ? copy(in_ref.val) : nothing
    h_tape = ow[6] ? _copy_spacing(h.val) : nothing
    return (in_tape, h_tape)
end
_fixed_tape(config, ::Union{_AverageBand, _CenteredBand}, args) = nothing

function _fixed_reverse(config, ::_DifferenceBand, n::Int, args, tape)
    out, in_ref, h, dims, dir, dim_val = args
    W = Val(EnzymeRules.width(config))
    d_outs = _lanes(out, W)
    all(isnothing, d_outs) && return nothing
    in_tape, h_tape = tape
    _difference_reverse!(d_outs, _lanes(in_ref, W), _lanes(h, W),
        in_tape === nothing ? in_ref.val : in_tape, h_tape === nothing ? h.val : h_tape,
        dims.val, dir.val, dim_val.val, n)
    return nothing
end

_adjoint_of(::typeof(Bramble._average_band!)) = Bramble._average_band_adjoint!
_adjoint_of(::typeof(Bramble._centered_average_band!)) = Bramble._centered_average_band_adjoint!

function _fixed_reverse(config, f::Union{_AverageBand, _CenteredBand}, n::Int, args, tape)
    out, in_ref, dims = args[1], args[2], args[3]
    W = Val(EnzymeRules.width(config))
    d_outs = _lanes(out, W)
    all(isnothing, d_outs) && return nothing
    _average_reverse!(_adjoint_of(f.val), d_outs, _lanes(in_ref, W), dims.val,
        _vals(Base.tail(Base.tail(Base.tail(args)))), n)
    return nothing
end

# The value behind a `Ref`, or the value itself.
@inline _unref(x) = x
@inline _unref(r::Base.RefValue) = r[]

# --- Closure kernels: the tasks ---------------------------------------------------------- #
#
# Each entry's loop body as a task `task(n, b, args...)`, `args` the entry's arguments in
# its own order.

@inline function _bands_task(n::Int, b::Int, f::F, nbands::Int, args::Vararg{Any, N}) where {F, N}
    f(args..., nbands, b)
    return nothing
end

# `_run_bands!(::CpuThreaded, f, args...)`'s band `b` of `n`.
@inline function _run_bands_task(n::Int, b::Int, policy, f::F, args::Vararg{Any, N}) where {F, N}
    f(args..., n, b)
    return nothing
end

# Slab `b` of `n` of `idxs`, cut by position: a sub-range of a range, a slab of a
# `CartesianIndices`' last axis, or a view of any other index vector.
@inline _slab(idxs::AbstractRange, n::Int, b::Int) = _band_range(idxs, n, b)
@inline function _slab(idxs::CartesianIndices, n::Int, b::Int)
    ax = idxs.indices
    return CartesianIndices((Base.front(ax)..., _band_range(last(ax), n, b)))
end
@inline function _slab(idxs::AbstractVector, n::Int, b::Int)
    return view(idxs, (firstindex(idxs) - 1) .+ _band_range(Base.OneTo(length(idxs)), n, b))
end
@noinline function _slab(idxs, n::Int, b::Int)
    throw(ArgumentError("the Enzyme rules for `_static_for!` take a range, a \
        `CartesianIndices` or an index vector, not a $(typeof(idxs))"))
end

# A sweep's kernel arrives as the closure itself, or behind a `Ref` (`_unref`): a closure
# that captures an array and a plain value fails Enzyme's calling convention by value.
@inline function _for_task(n::Int, b::Int, v, idxs, f::F) where {F}
    k = _unref(f)
    for i in _slab(idxs, n, b)
        @inbounds v[i] = k(i)
    end
    return nothing
end

@inline function _scatter_task(n::Int, b::Int, mats::Tuple, idxs, g::G) where {G}
    k = _unref(g)
    for i in _slab(idxs, n, b)
        @inbounds Bramble._write_components!(mats, k(i), i)
    end
    return nothing
end

@inline function _partials_task(n::Int, k::Int, partials, nn::Int, band::F,
        args::Vararg{Any, N}) where {F, N}
    @inbounds partials[k] = band(args..., k)
    return nothing
end

@inline function _band_colour_task(n::Int, k::Int, point::P, ax, bidx, nbands, rest) where {P}
    pt = _unref(point)
    for I in CartesianIndices((rest..., _band_range(ax, nbands, @inbounds bidx[k])))
        pt(I)
    end
    return nothing
end

# A colour's points cut by position into `n` balanced slabs.
@inline function _colour_task(n::Int, b::Int, point::P, idxs) where {P}
    pt = _unref(point)
    for j in _band_range(Base.OneTo(length(idxs)), n, b)
        pt(@inbounds idxs[j])
    end
    return nothing
end

@inline function _cut_task(n::Int, k::Int, point::P, span, bidx, nbands) where {P}
    Bramble._cut_band!(_unref(point), nbands, span, @inbounds bidx[k])
    return nothing
end

@inline function _kron_task(n::Int, b::Int, loc, y, K, x, cs::Tuple, β)
    dims = K.dims
    ss = Base.front(cumprod(dims))::Tuple{Vararg{Int}}
    lines = CartesianIndices(Base.tail(dims))
    Bramble._kron_line_band!(y, K.terms, cs, x, β, ss, lines, dims[1], n, b)
    return nothing
end

# The same band of a fused product whose sink, plan and form parts each cross behind their
# own `Ref`, `CpuPolyester`'s layout: the band reads the form through its own parts
# `(Wu, Wv, ast)`, whose shadow the operator's form carries (a scalar in the AST included),
# and the plan only through `info`, constant.
@inline function _mf_parts_task(n::Int, b::Int, policy, s::Base.RefValue, info::Base.RefValue,
        form::Base.RefValue, nbands::Int)
    Wu, Wv, ast = form[]
    walk, geom, omin, omax, len = info[]
    own = _band_range(1:len, nbands, b)
    Bramble._mf_apply_parts!(
        Bramble._MFPass(Bramble._MF_BAND, own, omin, omax, Bramble._MF_NO_COLLECT),
        Bramble._mf_with_geom(s[], geom), Wu, Wv, ast, walk)
    return nothing
end

# What a band reads of the plan and the walk, constant: the walk, the plan's geometry and row
# reach, and the banded axis's length.
@inline function _mf_info(parts::Base.RefValue)
    plan, walk = parts[]
    return Ref((walk, plan.geom, plan.omin, plan.omax, last(plan.dims)))
end

# --- Closure kernels: the written arrays -------------------------------------------------- #
#
# The arrays a task writes, found in its primal arguments or, the same way, in a lane's
# shadows of them: a band kernel's first argument, a `_FillBlock`'s destination, a sweep's
# `v` or `mats`, `partials`, and a point's matrix storage or right-hand side.

@inline _point_out(p::Bramble._LinearPoint) = (p.b,)
@inline _point_out(p::Bramble._BilinearPoint) = (_target_out(p.target),)
@inline _target_out(t::Bramble._ReplayTarget) = t.nzval
@inline _target_out(t::Bramble._ActionTarget) = t.y
@noinline function _target_out(t)
    throw(ArgumentError("the Enzyme rules for the `CpuThreaded` sweeps write a matrix's \
        storage or an action sink, not a $(typeof(t))"))
end

# Which argument holds the written arrays (`_holder`, from the primal values) and the arrays
# in it (`_written`, of the primal or of a shadow of it).
@inline _holder(::Val{:bands}, vals) = vals[1] isa Bramble._FillBlock ? 1 : 3
@inline _holder(::Val{:kron}, vals) = 2
@inline _holder(::Val{:run}, vals) = 3
@inline _holder(::Val{:mfparts}, vals) = 2
@inline _holder(::Val, vals) = 1

@inline _written(::Val{:bands}, h::Bramble._FillBlock) = (h.v,)
@inline _written(::Val{:scatter}, h::Tuple) = h
@inline _written(::Val{:point}, h) = _point_out(h)
@inline _written(::Val{:mfparts}, h::Bramble.ActionSink) = (h.y,)
@inline _written(::Val, h) = (h,)

# The mutable storage under an array: a view's or a reshape's parent, down to the buffer a
# shadow copy must share.
@inline _root(x) = x
@inline _root(x::AbstractArray) = (p = parent(x); p === x ? x : _root(p))

# The written arrays of the primal call, or of lane `w`'s shadows: `nothing` where their
# shadow is missing or is the primal itself, inactive under runtime activity.
@inline function _outs(hook::Val, ins::Tuple)
    vals = map(a -> a.val, ins)
    return _written(hook, _unref(vals[_holder(hook, vals)]))
end
function _outs(hook::Val, ins::Tuple, w::Int)
    k = _holder(hook, map(a -> a.val, ins))
    s = _shadow(ins[k], w)
    s === nothing && return nothing
    outs = map(_root, _written(hook, _unref(s)))
    all(map(===, outs, map(_root, _outs(hook, ins)))) && return nothing
    return outs
end

# --- Closure kernels: shadow copies and their reduction --------------------------------- #
#
# `Enzyme.make_zero`, `Base.deepcopy` and a recursive reduction all recurse once per object,
# so a closure holding a long linked structure overflows the stack. These walk with an
# explicit work list instead.

const _Opaque = Union{Symbol, String, Type, Module, Method, Core.MethodInstance, Task, IO,
    Base.AbstractLock}
const _Buffer = Union{Array, GenericMemory}

# A copy of `x` with its sharing and cycles kept: every float zeroed when `zero` (a shadow's
# per-task copy), every value copied otherwise (a tape's). An object in `keep` is shared,
# not copied: the written arrays, whose one shadow every task reads its cotangent from.
function _graph_copy(x, zero::Bool, keep::Tuple = ())
    seen, work = IdDict{Any, Any}(), Any[]
    foreach(o -> o === nothing || (seen[o] = o), keep)
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

_zero_bits(x::Union{AbstractFloat, Complex}) = zero(x)
function _zero_bits(x::T) where {T}
    (fieldcount(T) == 0 || !_has_float(T)) && return x
    return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T,
        Any[_zero_bits(getfield(x, i)) for i in 1:fieldcount(T)], fieldcount(T))::T
end
_has_float(::Type{<:Union{AbstractFloat, Complex}}) = true
_has_float(::Type{T}) where {T} = any(_has_float, fieldtypes(T))

# Adds the per-task shadow `src` into the caller's shadow `dst`, `primal` its primal: floats
# add where they are stored mutably. A shadow that is its own primal (runtime activity's
# inactive data), an object shared in the shadow, and an object in `keep` are skipped.
function _accumulate!(dst, src, primal, keep::Tuple = ())
    q = dst isa Base.RefValue && !(primal isa Base.RefValue) ? Ref(primal) : primal
    seen, work = Base.IdSet{Any}(), Any[(dst, src, q)]
    foreach(o -> o === nothing || push!(seen, o), keep)
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
_add(x::Tuple, y::Tuple) = map(_add, x, y)
function _add(x::T, y::T) where {T}
    (isbitstype(T) || !ismutabletype(T)) && fieldcount(T) > 0 || return x
    return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T,
        Any[_add(getfield(x, i), getfield(y, i)) for i in 1:fieldcount(T)], fieldcount(T))::T
end

# The per-task copies `cs` of an array of numbers added into its shadow `d`, a band of its
# indices per thread.
function _sum_copies!(d::Array{<:Union{AbstractFloat, Complex}}, cs, primal, keep)
    any(o -> o === d, keep) && return nothing
    n = Threads.nthreads()
    _each_task(n) do b
        for i in _band_range(Base.OneTo(length(d)), n, b)
            acc = @inbounds d[i]
            for c in cs
                acc += @inbounds c[i]
            end
            @inbounds d[i] = acc
        end
    end
    return nothing
end
function _sum_copies!(d, cs, primal, keep)
    foreach(c -> _accumulate!(d, c, primal, keep), cs)
    return nothing
end

# --- Closure kernels: forward and reverse ----------------------------------------------- #

# A task's annotation of an input, given its per-task shadow copy `c`.
@inline _task_annotation(a, v, ::Nothing) = a isa Active ? Active(v) : Const(v)
@inline _task_annotation(a, v, c) = Duplicated(v, c)
@inline function _task_annotation(
        a::Union{MixedDuplicated, BatchMixedDuplicated}, v, c::Base.RefValue)
    return MixedDuplicated(v, c)
end

# Every task's forward derivative, one task per thread, on the caller's annotations. A
# batched annotation makes a batched thunk, so each task runs every lane in one call.
function _tasks_forward!(config, task::T, n::Int, ins::Tuple) where {T}
    mode = Enzyme.set_runtime_activity(Enzyme.Forward, EnzymeRules.runtime_activity(config))
    thunk = Enzyme.autodiff_thunk(mode, Const{T}, Const, Const{Int}, Const{Int},
        map(typeof, ins)...)
    _each_task(b -> thunk(Const(task), Const(n), Const(b), ins...), n)
    return nothing
end

# One lane of a reverse: its thunks, `vals`, the inputs' values its replay reads and writes,
# and `copies[b]`, task `b`'s shadow copies of the inputs (`nothing` for an input with none),
# the written arrays' shadow shared among them. `act[b]` holds what task `b`'s reverse
# returned, read for the `Active` inputs. Each lane replays on its own copy of the values:
# a replay writes the written arrays, and an entry such as `β y` reads them first.
struct _Lane{FW, RV, V, C}
    fwd::FW
    rev::RV
    vals::V
    copies::Vector{C}
    act::Vector{Any}
end

@inline function _lane_reverse!(lane::_Lane, task::T, ins, n::Int, b::Int) where {T}
    anns = map(_task_annotation, ins, lane.vals, lane.copies[b])
    tape, _, _ = lane.fwd(Const(task), Const(n), Const(b), anns...)
    lane.act[b] = first(lane.rev(Const(task), Const(n), Const(b), anns..., tape))
    return nothing
end

# What input `k`'s derivative is: for an `Active` input the sum of its per-task
# derivatives, a lane each (the task's derivative tuple starts with `n` and `b`), `nothing`
# for any other.
# Dispatch on the annotation keeps the rule's return type inferable, as Enzyme requires.
@inline _active_sum(lanes, a, k::Int, ::Val{W}) where {W} = nothing
function _active_sum(lanes, a::Active{T}, k::Int, ::Val{W}) where {T, W}
    d = ntuple(Val(W)) do w
        lane = lanes[w]
        lane === nothing && return Enzyme.make_zero(a.val)::T
        return reduce(_add, (r[k + 2]::T for r in lane.act))::T
    end
    return W == 1 ? only(d) : d
end

# Every input's `_active_sum`, in order, unrolled over the tuple so its type is inferred.
@inline _active_sums(lanes, ::Tuple{}, k::Int, V::Val) = ()
@inline function _active_sums(lanes, ins::Tuple, k::Int, V::Val)
    return (_active_sum(lanes, first(ins), k, V), _active_sums(lanes, Base.tail(ins), k + 1, V)...)
end

# Runs `body(b)` for every task `b in 1:n`, one per thread under `Threads.@threads :static`.
# Inside a threaded region a `:static` loop cannot start, and the entry's primal ran its
# bands in order on the calling task (`_static_or_serial`); the derivative follows that
# same branch. Every derivative loop of this file goes through it: the rules above
# `_static_or_serial` (`_run_bands!`, `_kron_fused!`) are entered inside a user's region.
@inline function _each_task(body::B, n::Int) where {B}
    if Bramble._in_threaded_region()
        for b in 1:n
            body(b)
        end
    else
        Threads.@threads :static for b in 1:n
            body(b)
        end
    end
    return nothing
end

# Every task's split reverse, every lane in each task of one loop, then each lane's per-task
# shadow copies added into the caller's shadows. `vals` are the inputs' values for the
# replay, `keeps[w]` lane `w`'s written arrays' shadows (`nothing` in an inactive lane).
function _tasks_reverse!(config, task::T, n::Int, ins::Tuple, vals::Tuple, keeps,
        V::Val{W}) where {T, W}
    mode = Enzyme.set_runtime_activity(Enzyme.ReverseSplitNoPrimal,
        EnzymeRules.runtime_activity(config))
    lanes = ntuple(V) do w
        keep = keeps[w]
        keep === nothing && return nothing
        shadows = map(a -> _shadow(a, w), ins)
        copies = [map(s -> s === nothing ? nothing : _graph_copy(s, true, keep), shadows)
                  for _ in 1:n]
        lane_vals = w == 1 ? vals : _graph_copy(vals, false)
        anns = map(_task_annotation, ins, lane_vals, first(copies))
        fwd, rev = Enzyme.autodiff_thunk(mode, Const{T}, Const, Const{Int}, Const{Int},
            map(typeof, anns)...)
        return _Lane(fwd, rev, lane_vals, copies, Vector{Any}(nothing, n))
    end
    _each_task(n) do b
        foreach(lane -> lane === nothing || _lane_reverse!(lane, task, ins, n, b), lanes)
    end
    for w in 1:W
        lane = lanes[w]
        lane === nothing && continue
        for k in eachindex(ins)
            s = _shadow(ins[k], w)
            s === nothing ||
                _sum_copies!(s, [c[k] for c in lane.copies], ins[k].val, keeps[w])
        end
    end
    return _active_sums(lanes, ins, 1, V)
end

# The forward rule of a closure entry: `prim()` its primal, `hook` the shape of its written
# arrays (`_written`), `ins` its arguments as annotated.
function _closure_forward(config, prim::P, task::T, n::Int, hook::Val, ins::Tuple) where {P, T}
    W = EnzymeRules.width(config)
    if all(w -> _outs(hook, ins, w) === nothing, 1:W)
        prim()
        return nothing
    end
    _tasks_forward!(config, task, n, ins)
    return nothing
end

# The augmented primal of a closure entry. The tape is a copy of every argument's values
# from before the primal, taken whole in one walk so that arguments sharing an array (an
# in-place average's `out` and `in_ref`) still share it in the copy. `overwritten` is not
# consulted: an argument can hold data by reference (a `Broadcasted`'s arrays, an action
# target's `x`) that later code overwrites while its own entry says it does not. The
# replay reads the copy, and its writes land there, so the caller's arrays never move.
function _closure_tape(config, prim::P, hook::Val, ins::Tuple) where {P}
    W = EnzymeRules.width(config)
    live = any(w -> _outs(hook, ins, w) !== nothing, 1:W)
    tape = live ? _graph_copy(map(a -> a.val, ins), false) : nothing
    prim()
    return tape
end
function _closure_augmented(config, prim::P, hook::Val, ins::Tuple) where {P}
    return AugmentedReturn(nothing, nothing, _closure_tape(config, prim, hook, ins))
end

# The reverse of a closure entry: its derivative for each `Active` argument, `nothing` for
# every other, in argument order.
function _closure_reverse(config, task::T, n::Int, hook::Val, ins::Tuple, tape) where {T}
    V = Val(EnzymeRules.width(config))
    keeps = ntuple(w -> _outs(hook, ins, w), V)
    all(isnothing, keeps) && return map(a -> _zero_active(a, V), ins)
    return _tasks_reverse!(config, task, n, ins, tape, keeps, V)
end

@inline _zero_active(a, ::Val{W}) where {W} = nothing
@inline function _zero_active(a::Active{T}, ::Val{W}) where {T, W}
    z = Enzyme.make_zero(a.val)::T
    return W == 1 ? z : ntuple(_ -> z, Val(W))
end

# --- The rules ----------------------------------------------------------------------------- #

const _BANDS = Val(:bands)

function EnzymeRules.forward(config::FwdConfig, func::Const{_StaticBands}, ::Type{RT},
        f::Annotation, nbands::Annotation, args::Vararg{Annotation, N}) where {RT, N}
    _entered!(:_static_bands!)
    if f isa _FixedBand
        func.val(f.val, nbands.val, _vals(args)...)
        _fixed_forward(config, f, nbands.val, args)
    else
        _closure_forward(config, () -> func.val(f.val, nbands.val, _vals(args)...),
            _bands_task, nbands.val, _BANDS, (f, nbands, args...))
    end
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_StaticBands},
        ::Type{RT}, f::Annotation, nbands::Annotation, args::Vararg{Annotation, N}) where {
        RT, N}
    _entered!(:_static_bands!)
    prim() = func.val(f.val, nbands.val, _vals(args)...)
    if f isa _FixedBand
        tape = _fixed_tape(config, f, args)
        prim()
        return AugmentedReturn(nothing, nothing, tape)
    end
    return _closure_augmented(config, prim, _BANDS, (f, nbands, args...))
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_StaticBands}, ::Type{RT}, tape,
        f::Annotation, nbands::Annotation, args::Vararg{Annotation, N}) where {RT, N}
    if f isa _FixedBand
        _fixed_reverse(config, f, nbands.val, args, tape)
        return ntuple(_ -> nothing, Val(N + 2))
    end
    return _closure_reverse(config, _bands_task, nbands.val, _BANDS, (f, nbands, args...), tape)
end

# `_run_bands!(::CpuThreaded, f, out, args...)`, the accumulating, shift, ghost and
# multigrid band engines: a rule above `_static_or_serial`, so Enzyme never compiles that
# function's `try`/`catch` around these arguments (the ghost operators' `Active`
# `_FacetFix` tuples there fail LLVM's verifier, "PHI nodes not grouped at top of basic
# block"). One band per thread, `out` the written array, as `_static_bands!`'s closure
# route.
const _RunBands = typeof(Bramble._run_bands!)
const _RUN = Val(:run)

function EnzymeRules.forward(config::FwdConfig, func::Const{_RunBands}, ::Type{RT},
        policy::Annotation{CpuThreaded}, f::Annotation, args::Vararg{Annotation, N}) where {
        RT, N}
    _entered!(:_run_bands!)
    ins = (policy, f, args...)
    _closure_forward(config, () -> func.val(_vals(ins)...), _run_bands_task,
        Threads.nthreads(), _RUN, ins)
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_RunBands},
        ::Type{RT}, policy::Annotation{CpuThreaded}, f::Annotation,
        args::Vararg{Annotation, N}) where {RT, N}
    _entered!(:_run_bands!)
    ins = (policy, f, args...)
    return _closure_augmented(config, () -> func.val(_vals(ins)...), _RUN, ins)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_RunBands}, ::Type{RT}, tape,
        policy::Annotation{CpuThreaded}, f::Annotation, args::Vararg{Annotation, N}) where {
        RT, N}
    return _closure_reverse(config, _run_bands_task, Threads.nthreads(), _RUN,
        (policy, f, args...), tape)
end

# `_static_for!(v, idxs, f)` and `_static_scatter_for!(mats, idxs, g)`: one slab of `idxs`
# per thread.
for (F, hook, task, shape) in ((_StaticFor, :_static_for!, _for_task, Val(:for)),
    (_StaticScatter, :_static_scatter_for!, _scatter_task, Val(:scatter)))
    @eval begin
        function EnzymeRules.forward(config::FwdConfig, func::Const{$F}, ::Type{RT},
                v::Annotation, idxs::Annotation, f::Annotation) where {RT}
            _entered!($(QuoteNode(hook)))
            _closure_forward(config, () -> func.val(v.val, idxs.val, f.val), $task,
                Threads.nthreads(), $shape, (v, idxs, f))
            return nothing
        end

        function EnzymeRules.augmented_primal(config::RevConfig, func::Const{$F},
                ::Type{RT}, v::Annotation, idxs::Annotation, f::Annotation) where {RT}
            _entered!($(QuoteNode(hook)))
            return _closure_augmented(config, () -> func.val(v.val, idxs.val, f.val),
                $shape, (v, idxs, f))
        end

        function EnzymeRules.reverse(config::RevConfig, func::Const{$F}, ::Type{RT}, tape,
                v::Annotation, idxs::Annotation, f::Annotation) where {RT}
            return _closure_reverse(config, $task, Threads.nthreads(), $shape,
                (v, idxs, f), tape)
        end
    end
end

# `_static_partials!(partials, n, band, args...)`: `partials[k] = band(args..., k)`, one `k`
# per task.
const _PARTIALS = Val(:partials)

function EnzymeRules.forward(config::FwdConfig, func::Const{_StaticPartials}, ::Type{RT},
        partials::Annotation, n::Annotation, band::Annotation,
        args::Vararg{Annotation, N}) where {RT, N}
    _entered!(:_static_partials!)
    _closure_forward(config, () -> func.val(partials.val, n.val, band.val, _vals(args)...),
        _partials_task, n.val, _PARTIALS, (partials, n, band, args...))
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_StaticPartials},
        ::Type{RT}, partials::Annotation, n::Annotation, band::Annotation,
        args::Vararg{Annotation, N}) where {RT, N}
    _entered!(:_static_partials!)
    return _closure_augmented(config,
        () -> func.val(partials.val, n.val, band.val, _vals(args)...), _PARTIALS,
        (partials, n, band, args...))
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_StaticPartials}, ::Type{RT},
        tape, partials::Annotation, n::Annotation, band::Annotation,
        args::Vararg{Annotation, N}) where {RT, N}
    return _closure_reverse(config, _partials_task, n.val, _PARTIALS,
        (partials, n, band, args...), tape)
end

# The bilinear and linear sweeps' runners: `point` holds the written storage and the term,
# and arrives behind a `Ref` (`_threaded_bands!`, src/assembly/bilinear_execution.jl).
const _POINT = Val(:point)

for (F, hook, task, ntasks) in (
    (_StaticBandColour, :_static_band_colour!, _band_colour_task, :(length(ins[3].val))),
    (_StaticColour, :_static_colour!, _colour_task, :(Threads.nthreads())),
    (_StaticCutBandColour, :_static_cut_band_colour!, _cut_task, :(length(ins[3].val))))
    @eval begin
        function EnzymeRules.forward(config::FwdConfig, func::Const{$F}, ::Type{RT},
                point::Annotation, args::Vararg{Annotation, N}) where {RT, N}
            _entered!($(QuoteNode(hook)))
            ins = (point, args...)
            _closure_forward(config, () -> func.val(_vals(ins)...), $task, $ntasks, _POINT,
                ins)
            return nothing
        end

        function EnzymeRules.augmented_primal(config::RevConfig, func::Const{$F},
                ::Type{RT}, point::Annotation, args::Vararg{Annotation, N}) where {RT, N}
            _entered!($(QuoteNode(hook)))
            ins = (point, args...)
            return _closure_augmented(config, () -> func.val(_vals(ins)...), _POINT, ins)
        end

        function EnzymeRules.reverse(config::RevConfig, func::Const{$F}, ::Type{RT}, tape,
                point::Annotation, args::Vararg{Annotation, N}) where {RT, N}
            ins = (point, args...)
            return _closure_reverse(config, $task, $ntasks, _POINT, ins, tape)
        end
    end
end

# `_kron_fused!(loc, y, K, x, cs, β)` under `CpuThreaded`, which returns `y`: its lines cut
# into one band per thread, each band `_kron_line_band!`.
const _KRON = Val(:kron)

# The return as `RT` asks for it: `y`'s value and shadow, either, or neither.
function _kron_return(config, y)
    p, s = EnzymeRules.needs_primal(config), EnzymeRules.needs_shadow(config)
    W = EnzymeRules.width(config)
    p && s && return W == 1 ? Duplicated(y.val, y.dval) : BatchDuplicated(y.val, y.dval)
    s && return y.dval
    p && return y.val
    return nothing
end

function EnzymeRules.forward(config::FwdConfig, func::Const{_KronFused}, ::Type{RT},
        loc::Annotation, y::Annotation, K::_ThreadedKron, x::Annotation, cs::Annotation,
        β::Annotation) where {RT}
    _entered!(:_kron_fused!)
    ins = (loc, y, K, x, cs, β)
    _closure_forward(config, () -> func.val(_vals(ins)...), _kron_task, Threads.nthreads(),
        _KRON, ins)
    return _kron_return(config, y)
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_KronFused},
        ::Type{RT}, loc::Annotation, y::Annotation, K::_ThreadedKron, x::Annotation,
        cs::Annotation, β::Annotation) where {RT}
    _entered!(:_kron_fused!)
    ins = (loc, y, K, x, cs, β)
    tape = _closure_tape(config, () -> func.val(_vals(ins)...), _KRON, ins)
    p = EnzymeRules.needs_primal(config) ? y.val : nothing
    s = EnzymeRules.needs_shadow(config) ? y.dval : nothing
    return AugmentedReturn(p, s, tape)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_KronFused}, ::Type{RT}, tape,
        loc::Annotation, y::Annotation, K::_ThreadedKron, x::Annotation, cs::Annotation,
        β::Annotation) where {RT}
    return _closure_reverse(config, _kron_task, Threads.nthreads(), _KRON,
        (loc, y, K, x, cs, β), tape)
end

const _MFRunBands = typeof(Bramble._mf_run_bands!)

# `_mf_run_bands!(::CpuThreaded, s, parts, form, nbands)`: the sink, the plan with its
# walk, and the form's parts `(Wu, Wv, ast)` each behind a `Ref`, as under `CpuPolyester`.
# The plan is structure built once, so the tasks read it as a constant `_mf_info`, and the
# form through `form`, whose shadow carries every coefficient's. The plan's build walks the
# form with untyped collectors that Enzyme cannot compile (a scalar in the AST crashed it,
# a segfault in its dead-code pass), so the build is inactive, as under `CpuPolyester`.
EnzymeRules.inactive(::typeof(Bramble._mf_plan), ::CpuThreaded, args...) = nothing
const _MFPARTS = Val(:mfparts)
const _MFSinkRef = Annotation{<:Base.RefValue{<:Bramble.ActionSink}}

function EnzymeRules.forward(config::FwdConfig, func::Const{_MFRunBands}, ::Type{RT},
        policy::Annotation{CpuThreaded}, s::_MFSinkRef, parts::Annotation{<:Base.RefValue},
        form::Annotation{<:Base.RefValue}, nbands::Annotation{Int}) where {RT}
    _entered!(:_mf_run_bands!)
    ins = (policy, s, Const(_mf_info(parts.val)), form, nbands)
    _closure_forward(config, () -> func.val(policy.val, s.val, parts.val, form.val, nbands.val),
        _mf_parts_task, nbands.val, _MFPARTS, ins)
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_MFRunBands},
        ::Type{RT}, policy::Annotation{CpuThreaded}, s::_MFSinkRef,
        parts::Annotation{<:Base.RefValue}, form::Annotation{<:Base.RefValue},
        nbands::Annotation{Int}) where {RT}
    _entered!(:_mf_run_bands!)
    ins = (policy, s, Const(_mf_info(parts.val)), form, nbands)
    return _closure_augmented(config,
        () -> func.val(policy.val, s.val, parts.val, form.val, nbands.val), _MFPARTS, ins)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_MFRunBands}, ::Type{RT}, tape,
        policy::Annotation{CpuThreaded}, s::_MFSinkRef, parts::Annotation{<:Base.RefValue},
        form::Annotation{<:Base.RefValue}, nbands::Annotation{Int}) where {RT}
    ins = (policy, s, Const(_mf_info(parts.val)), form, nbands)
    return _closure_reverse(config, _mf_parts_task, nbands.val, _MFPARTS, ins, tape)
end
