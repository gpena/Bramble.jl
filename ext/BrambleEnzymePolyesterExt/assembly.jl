# assembly.jl: Enzyme rules for the `CpuPolyester` assembly sweeps. The bilinear replays
# (`_batch_bilinear_band_replay!`, `_batch_bilinear_colour_replay!`,
# src/assembly/bilinear_execution.jl) add each point's stencil into a replay target's `nzval`;
# the linear sweeps (`_batch_linear_band_sweep!`, `_batch_linear_colour_sweep!`,
# src/assembly/linear.jl) add it into `b`. Each hook call is one colour, so no two points of a
# call write one entry.
#
# The rules take the calls `src/` makes through the hooks' `Ref` methods: the target (a
# bilinear sweep's) and the parts `(sp, term, mesh_markers)` (and `α`, a linear sweep's) each
# behind a `Ref`, since a by-value struct that mixes arrays with plain values fails Enzyme's
# calling convention at the hook. The form term's functor is a closure over its coefficient
# elements, so the rules take the closure route of `_batch_for!` (sweeps.jl): one Enzyme
# derivative of a serial slab of the hook's loop per `@batch` task, through a thunk built on
# the host and handed to the tasks through `BramblePolyesterExt`'s typed slot. A band hook
# runs one task per band of its colour, as its primal does; a colour hook cuts its colour's
# points into `Threads.nthreads()` slabs, one `@batch` per colour as its primal.
#
# Forward: each task runs Enzyme's forward derivative of its slab on the caller's arguments,
# which adds the slab's values and their tangents into the output and its shadow.
#
# Reverse: the output only accumulates (`+=`), so its cotangent passes through unchanged and
# the reverse adds into the parts' shadow alone. Each task runs Enzyme's split reverse
# derivative of its slab with the output's shadow seeded by the caller's cotangent, shared by
# the tasks (the colour keeps their entries apart). Its augmented forward adds the slab once
# more, so it writes a scratch output (`similar`, one per call) and the user's never moves.
# The parts' shadow (the coefficient elements a term captures, and the space's data) is
# written by every task, so each task accumulates into its own zeroed copy (`_graph_copy`),
# and the host adds the copies into the caller's shadow after the join (`_accumulate!`), as
# sweeps.jl does for a closure. The same holds for a replay target's own `α`: each task gets a
# zeroed copy of the target's shadow around the shared cotangent, added back on the host.
#
# Batch width: forward passes `BatchDuplicated` arguments through, so each task runs every
# lane in one call. Reverse runs a width-1 thunk per lane in each task of one launch.
#
# A matrix-free product reaches the bilinear hooks with an action target behind a `Ref`
# (src/assembly/matrix_free.jl). Its rules are in operators.jl and reuse these slabs.

const _AsmBandReplay = typeof(Bramble._batch_bilinear_band_replay!)
const _AsmColourReplay = typeof(Bramble._batch_bilinear_colour_replay!)
const _AsmBandSweep = typeof(Bramble._batch_linear_band_sweep!)
const _AsmColourSweep = typeof(Bramble._batch_linear_colour_sweep!)

# The replay hooks' launches into `BramblePolyesterExt`'s sweep, inactive: where a replay is
# inactive (a ghost form's constant `A_pad`), Enzyme clones the caller without frees,
# callees included, a hook's body too since its rule does not stop the clone, and the
# `@batch` scheduler holds runtime calls it cannot clone (gpena/Bramble.jl#1157). Only the
# hooks' `Ref` methods call the launches, so every differentiated call meets a rule first.
EnzymeRules.inactive(::typeof(Bramble._launch_band_replay!), args...) = nothing
EnzymeRules.inactive(::typeof(Bramble._launch_colour_replay!), args...) = nothing

const _AsmTarget = Annotation{<:Base.RefValue{<:Bramble._ReplayTarget}}
const _AsmParts = Annotation{<:Base.RefValue{<:Tuple}}

# --- Slabs: a serial run of each hook's loop over the items `r` ---------------------- #

function _asm_replay_bands!(target::Base.RefValue, parts::Base.RefValue, ax, bidx, nbands,
        rest, lin_indices, row_offset, col_offset, r)
    t = target[]
    sp, term, mesh_markers = parts[]
    for k in r
        for I in CartesianIndices((rest..., _band_range(ax, nbands, @inbounds bidx[k])))
            Bramble._replay_point!(
                t, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset)
        end
    end
    return nothing
end

function _asm_replay_points!(target::Base.RefValue, parts::Base.RefValue, idxs, lin_indices,
        row_offset, col_offset, r)
    t = target[]
    sp, term, mesh_markers = parts[]
    for k in r
        Bramble._replay_point!(t, term, sp, @inbounds(idxs[k]), lin_indices, mesh_markers,
            row_offset, col_offset)
    end
    return nothing
end

function _asm_scatter_bands!(b, parts::Base.RefValue, ax, bidx, nbands, rest, lin_indices,
        offset, r)
    sp, term, mesh_markers, α = parts[]
    for k in r
        for I in CartesianIndices((rest..., _band_range(ax, nbands, @inbounds bidx[k])))
            Bramble._scatter_linear_point!(b, sp, term, I, lin_indices, mesh_markers, offset, α)
        end
    end
    return nothing
end

function _asm_scatter_points!(b, parts::Base.RefValue, idxs, lin_indices, offset, r)
    sp, term, mesh_markers, α = parts[]
    for k in r
        Bramble._scatter_linear_point!(
            b, sp, term, @inbounds(idxs[k]), lin_indices, mesh_markers, offset, α)
    end
    return nothing
end

# The items of task `b` of `n` over `len` items: one band each for a band hook (`n == len`),
# a balanced slab of the colour's points for a colour hook.
@inline _asm_items(len::Int, n::Int, b::Int) = _band_range(1:len, n, b)

# --- Forward ----------------------------------------------------------------------- #

struct _AsmForward{TH, F, A}
    thunk::TH
    slab::F
    args::A
    len::Int
    errs::Vector{Any}
end

@noinline function _asm_forward_threw(p::_AsmForward, n::Int, b::Int)
    try
        p.thunk(Const(p.slab), p.args..., Const(_asm_items(p.len, n, b)))
    catch e
        p.errs[b] = e
        return true
    end
    return false
end

@noinline function _asm_band_forward_run!(take::T, n::Int, h::H) where {T, H}
    failed = false
    @batch reduction=((|, failed),) for b in 1:n
        failed |= _asm_forward_threw(take(h), n, b)
    end
    return failed
end

@noinline function _asm_colour_forward_run!(take::T, n::Int, h::H) where {T, H}
    failed = false
    @batch reduction=((|, failed),) for b in 1:n
        failed |= _asm_forward_threw(take(h), n, b)
    end
    return failed
end

function _asm_forward!(config, run!::R, slab::F, n::Int, len::Int, args...) where {R, F}
    mode = Enzyme.set_runtime_activity(Enzyme.Forward, EnzymeRules.runtime_activity(config))
    thunk = Enzyme.autodiff_thunk(mode, Const{F}, Const, map(typeof, args)...,
        Const{UnitRange{Int}})
    p = _AsmForward(thunk, slab, args, len, Vector{Any}(nothing, n))
    pe = _polyester_ext()
    pe._handed(run!, p, pe._take, n) && _rethrow_first(p.errs)
    return nothing
end

# --- Reverse ----------------------------------------------------------------------- #

# `args[j, b]` holds the slab's arguments for the `j`-th active lane in task `b`.
struct _AsmReverse{FW, RV, F, A}
    fwd::FW
    rev::RV
    slab::F
    args::Matrix{A}
    len::Int
    errs::Vector{Any}
end

@noinline function _asm_reverse_threw(p::_AsmReverse, n::Int, b::Int)
    try
        r = Const(_asm_items(p.len, n, b))
        for j in axes(p.args, 1)
            a = p.args[j, b]
            tape, _, _ = p.fwd(Const(p.slab), a..., r)
            p.rev(Const(p.slab), a..., r, tape)
        end
    catch e
        p.errs[b] = e
        return true
    end
    return false
end

@noinline function _asm_band_reverse_run!(take::T, n::Int, h::H) where {T, H}
    failed = false
    @batch reduction=((|, failed),) for b in 1:n
        failed |= _asm_reverse_threw(take(h), n, b)
    end
    return failed
end

@noinline function _asm_colour_reverse_run!(take::T, n::Int, h::H) where {T, H}
    failed = false
    @batch reduction=((|, failed),) for b in 1:n
        failed |= _asm_reverse_threw(take(h), n, b)
    end
    return failed
end

# The output's storage: a replay target's `nzval`, or `b` itself.
@inline _asm_storage(t::Base.RefValue) = first(_polyester_ext()._replay_parts(t[]))
@inline _asm_storage(b::AbstractArray) = b

# The output's cotangent in lane `w`, `nothing` where it is inactive.
@inline function _asm_cotangent(x, w::Int)
    d = _shadow(x, w)
    d === nothing && return nothing
    s = _asm_storage(d)
    return s === _asm_storage(x.val) ? nothing : s
end

# A task's output argument: the scratch `out` as primal, seeded by the cotangent `d`. A
# target's shadow is rebuilt around `d` with its other fields' shadow zeroed for the task.
@inline _asm_output(out::AbstractArray, d, _) = Duplicated(out, d)
function _asm_output(out::Base.RefValue, d, dt::Base.RefValue)
    pe = _polyester_ext()
    _, _, cold = pe._replay_parts(dt[])
    return Duplicated(out, Ref(pe._replay_join(d, nothing, _graph_copy(cold, true))))
end

# Adds a task's target shadow (its `α`'s, say) into the caller's, around the cotangent.
@inline _asm_add_output!(_, _) = nothing
function _asm_add_output!(dt::Base.RefValue, task::Duplicated{<:Base.RefValue})
    pe = _polyester_ext()
    d, _, cold = pe._replay_parts(dt[])
    _, _, add = pe._replay_parts(task.dval[])
    dt[] = pe._replay_join(d, nothing, _add(cold, add))
    return nothing
end

# Whether the output's other fields hold a float that may carry a derivative: a replay
# target's `α`, unless it is a `Bool`.
@inline _asm_cold_float(::AbstractArray) = false
@inline _asm_cold_float(t::Base.RefValue) = _has_float(
    typeof(last(_polyester_ext()._replay_parts(t[]))))

# A scratch the tasks' augmented forwards write in place of the user's output.
@inline _asm_scratch(b::AbstractArray) = similar(b)
function _asm_scratch(t::Base.RefValue)
    pe = _polyester_ext()
    h1, _, cold = pe._replay_parts(t[])
    return Ref(pe._replay_join(similar(h1), nothing, cold))
end

# The reverse of an assembly hook whose output is `out` and whose parts are `parts`, the
# rest of its arguments `rest` (indices and offsets, constant). `parts_tape` holds the parts
# from before the primal when later code overwrites them.
function _asm_reverse!(config, run!::R, slab::F, n::Int, len::Int, out, parts, parts_tape,
        rest::Tuple) where {R, F}
    W = Val(EnzymeRules.width(config))
    d_outs = ntuple(w -> _asm_cotangent(out, w), W)
    d_parts = _lanes(parts, W)
    cold = _asm_cold_float(out.val)
    live = [w for w in 1:EnzymeRules.width(config)
            if d_outs[w] !== nothing && (cold || d_parts[w] !== nothing)]
    isempty(live) && return nothing
    scratch = _asm_scratch(out.val)
    pp = parts_tape === nothing ? parts.val : parts_tape
    cs = map(Const, rest)
    # Inactive parts in a live lane (only a target's `α` is shadowed) get a zeroed copy of
    # their own, which keeps every lane one type and is dropped after the join.
    task(w) = (_asm_output(scratch, d_outs[w], _shadow(out, w)),
        Duplicated(pp, _graph_copy(something(d_parts[w], pp), true)), cs...)
    args = [task(w) for w in live, _ in 1:n]
    mode = Enzyme.set_runtime_activity(Enzyme.ReverseSplitNoPrimal,
        EnzymeRules.runtime_activity(config))
    fwd, rev = Enzyme.autodiff_thunk(mode, Const{F}, Const, map(typeof, first(args))...,
        Const{UnitRange{Int}})
    p = _AsmReverse(fwd, rev, slab, args, len, Vector{Any}(nothing, n))
    pe = _polyester_ext()
    pe._handed(run!, p, pe._take, n) && _rethrow_first(p.errs)
    for (j, w) in enumerate(live), b in 1:n

        a = args[j, b]
        _asm_add_output!(_shadow(out, w), a[1])
        d_parts[w] === nothing || _accumulate!(d_parts[w], a[2].dval, parts.val)
    end
    return nothing
end

# The parts from before the primal, when later code overwrites them (`overwritten` counts
# `func` first, so the parts are entry 3), else `nothing`.
@inline function _asm_tape(config, parts)
    parts isa Const && return nothing
    return EnzymeRules.overwritten(config)[3] ? _graph_copy(parts.val, false) : nothing
end

# --- The rules ----------------------------------------------------------------------- #
#
# Every hook's reverse returns no derivative for an argument: the output and the parts are
# shadowed, and the rest are indices and offsets.

function EnzymeRules.forward(config::FwdConfig, func::Const{_AsmBandReplay}, ::Type{RT},
        target::_AsmTarget, parts::_AsmParts, ax::Annotation, bidx::Annotation,
        nbands::Annotation, rest::Annotation, lin_indices::Annotation,
        row_offset::Annotation, col_offset::Annotation) where {RT}
    _entered!(:_batch_bilinear_band_replay!)
    len = length(bidx.val)
    _asm_forward!(config, _asm_band_forward_run!, _asm_replay_bands!, len, len, target,
        parts, ax, bidx, nbands, rest, lin_indices, row_offset, col_offset)
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_AsmBandReplay},
        ::Type{RT}, target::_AsmTarget, parts::_AsmParts, ax::Annotation, bidx::Annotation,
        nbands::Annotation, rest::Annotation, lin_indices::Annotation,
        row_offset::Annotation, col_offset::Annotation) where {RT}
    _entered!(:_batch_bilinear_band_replay!)
    tape = _asm_tape(config, parts)
    func.val(target.val, parts.val, ax.val, bidx.val, nbands.val, rest.val, lin_indices.val,
        row_offset.val, col_offset.val)
    return AugmentedReturn(nothing, nothing, tape)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_AsmBandReplay}, ::Type{RT},
        tape, target::_AsmTarget, parts::_AsmParts, ax::Annotation, bidx::Annotation,
        nbands::Annotation, rest::Annotation, lin_indices::Annotation,
        row_offset::Annotation, col_offset::Annotation) where {RT}
    len = length(bidx.val)
    _asm_reverse!(config, _asm_band_reverse_run!, _asm_replay_bands!, len, len, target,
        parts, tape, (ax.val, bidx.val, nbands.val, rest.val, lin_indices.val,
            row_offset.val, col_offset.val))
    return ntuple(_ -> nothing, Val(9))
end

function EnzymeRules.forward(config::FwdConfig, func::Const{_AsmColourReplay}, ::Type{RT},
        target::_AsmTarget, parts::_AsmParts, idxs::Annotation, lin_indices::Annotation,
        row_offset::Annotation, col_offset::Annotation) where {RT}
    _entered!(:_batch_bilinear_colour_replay!)
    _asm_forward!(config, _asm_colour_forward_run!, _asm_replay_points!, Threads.nthreads(),
        length(idxs.val), target, parts, idxs, lin_indices, row_offset, col_offset)
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_AsmColourReplay},
        ::Type{RT}, target::_AsmTarget, parts::_AsmParts, idxs::Annotation,
        lin_indices::Annotation, row_offset::Annotation, col_offset::Annotation) where {RT}
    _entered!(:_batch_bilinear_colour_replay!)
    tape = _asm_tape(config, parts)
    func.val(target.val, parts.val, idxs.val, lin_indices.val, row_offset.val,
        col_offset.val)
    return AugmentedReturn(nothing, nothing, tape)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_AsmColourReplay}, ::Type{RT},
        tape, target::_AsmTarget, parts::_AsmParts, idxs::Annotation,
        lin_indices::Annotation, row_offset::Annotation, col_offset::Annotation) where {RT}
    _asm_reverse!(config, _asm_colour_reverse_run!, _asm_replay_points!, Threads.nthreads(),
        length(idxs.val), target, parts, tape,
        (idxs.val, lin_indices.val, row_offset.val, col_offset.val))
    return ntuple(_ -> nothing, Val(6))
end

function EnzymeRules.forward(config::FwdConfig, func::Const{_AsmBandSweep}, ::Type{RT},
        b::Annotation{<:AbstractVector}, parts::_AsmParts, ax::Annotation,
        bidx::Annotation, nbands::Annotation, rest::Annotation, lin_indices::Annotation,
        offset::Annotation) where {RT}
    _entered!(:_batch_linear_band_sweep!)
    len = length(bidx.val)
    _asm_forward!(config, _asm_band_forward_run!, _asm_scatter_bands!, len, len, b, parts,
        ax, bidx, nbands, rest, lin_indices, offset)
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_AsmBandSweep},
        ::Type{RT}, b::Annotation{<:AbstractVector}, parts::_AsmParts, ax::Annotation,
        bidx::Annotation, nbands::Annotation, rest::Annotation, lin_indices::Annotation,
        offset::Annotation) where {RT}
    _entered!(:_batch_linear_band_sweep!)
    tape = _asm_tape(config, parts)
    func.val(b.val, parts.val, ax.val, bidx.val, nbands.val, rest.val, lin_indices.val,
        offset.val)
    return AugmentedReturn(nothing, nothing, tape)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_AsmBandSweep}, ::Type{RT},
        tape, b::Annotation{<:AbstractVector}, parts::_AsmParts, ax::Annotation,
        bidx::Annotation, nbands::Annotation, rest::Annotation, lin_indices::Annotation,
        offset::Annotation) where {RT}
    len = length(bidx.val)
    _asm_reverse!(config, _asm_band_reverse_run!, _asm_scatter_bands!, len, len, b, parts,
        tape, (ax.val, bidx.val, nbands.val, rest.val, lin_indices.val, offset.val))
    return ntuple(_ -> nothing, Val(8))
end

function EnzymeRules.forward(config::FwdConfig, func::Const{_AsmColourSweep}, ::Type{RT},
        b::Annotation{<:AbstractVector}, parts::_AsmParts, idxs::Annotation,
        lin_indices::Annotation, offset::Annotation) where {RT}
    _entered!(:_batch_linear_colour_sweep!)
    _asm_forward!(config, _asm_colour_forward_run!, _asm_scatter_points!, Threads.nthreads(),
        length(idxs.val), b, parts, idxs, lin_indices, offset)
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{_AsmColourSweep},
        ::Type{RT}, b::Annotation{<:AbstractVector}, parts::_AsmParts, idxs::Annotation,
        lin_indices::Annotation, offset::Annotation) where {RT}
    _entered!(:_batch_linear_colour_sweep!)
    tape = _asm_tape(config, parts)
    func.val(b.val, parts.val, idxs.val, lin_indices.val, offset.val)
    return AugmentedReturn(nothing, nothing, tape)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{_AsmColourSweep}, ::Type{RT},
        tape, b::Annotation{<:AbstractVector}, parts::_AsmParts, idxs::Annotation,
        lin_indices::Annotation, offset::Annotation) where {RT}
    _asm_reverse!(config, _asm_colour_reverse_run!, _asm_scatter_points!, Threads.nthreads(),
        length(idxs.val), b, parts, tape, (idxs.val, lin_indices.val, offset.val))
    return ntuple(_ -> nothing, Val(5))
end
