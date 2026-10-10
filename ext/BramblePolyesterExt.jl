# ext/BramblePolyesterExt.jl: the Polyester-batched sweeps behind `CpuPolyester`.
#
# Nineteen hooks in `src/` are `@noinline` methods that error naming Polyester. This file
# defines each of them, as the `Polyester.@batch` counterpart of an existing
# `Threads.@threads` body, called from the identical call site once `execution_policy`
# resolves to `CpuPolyester()` instead of `CpuThreaded()`. The colouring, the band splitting
# and, critically, the matrix/vector zeroing that happens in the *callers*
# (`_assemble_bilinear!`, `assemble_parallel!`, `_assemble_linear!`) are untouched by this
# file: it only supplies what runs once the caller has already zeroed and dispatched.
#
# `src/utils/linear_algebra.jl`: `_batch_for!`, `_batch_axis_for!`, `_batch_scatter_for!`,
# `_batch_dot` and `_batch_dot_masked`.
#
# `src/assembly/bilinear_execution.jl`: `_batch_bilinear_colour_sweep!` and
# `_batch_bilinear_band_sweep!`, and their warmed-refill counterparts
# `_batch_bilinear_colour_replay!` and `_batch_bilinear_band_replay!`.
#
# `src/assembly/linear.jl`: `_batch_linear_colour_sweep!` and `_batch_linear_band_sweep!`.
#
# `src/operators/difference.jl` and `src/operators/average.jl`: `_batch_difference_engine!`,
# `_batch_average_engine!` and `_batch_centered_average_engine!`, each the `@batch`
# counterpart of `_threaded_difference_engine!`/`_threaded_average_engine!`/
# `_threaded_centered_average_engine!`, running one
# `_difference_band!`/`_average_band!`/`_centered_average_band!` per band.
#
# `src/operators/vector_calculus.jl`: `_batch_run_bands!`, the `@batch` counterpart of
# `_run_bands!`'s `CpuThreaded` arm, reached by the divergence, curl and strain-average
# engines. Unlike the three engine hooks, it stays generic over the band function `f`.
#
# `src/assembly/matrix_free.jl`: `_batch_mf_bands!`, the fused matrix-free product, one band
# per `@batch` task.
#
# `src/space/vectorelement.jl`: `_batch_broadcast!`, one `_broadcast_band!` per band of a
# fused broadcast.
#
# `src/assembly/kronecker.jl`: `_batch_kron_lines!`, which runs a `KroneckerLinearOperator`
# product's grid lines under `@batch` when the operator's own policy is `CpuPolyester`.
#
# `src/problems/semidiscrete_rhs.jl`: `_batch_csr_spmv!`, which runs the explicit
# right-hand side's product `du .-= A u` one compressed row per `@batch` iteration.
#
# A warmed refill replays recorded `nzval` positions instead of searching, gated per unit by
# `_threaded_replay_policy`; only `CpuThreaded` answers `true` in `src/`.
# This file's `_threaded_replay_policy(::CpuPolyester) = true` opts `CpuPolyester` in, and
# `_batch_bilinear_band_replay!`/`_batch_bilinear_colour_replay!` are the `@batch` counterparts
# of `_batch_bilinear_band_sweep!`/`_batch_bilinear_colour_sweep!`, reached instead of
# them once a unit's leaf can replay (`_leaf_replays`, bilinear_execution.jl): the colouring is
# identical, only `_replay_point!` (reads the recording) stands in for `_scatter_point!`
# (searches).
#
# `Polyester.@batch` accepts a `CartesianIndices` directly (`closure.jl`'s own `splitloop`
# already splits it along its last axis, the same trick `_threaded_axis_for!` hand-rolls for
# `Threads.@threads`), so none of the manual axis-chunking `src/utils/linear_algebra.jl` uses
# for the threaded path is reproduced here. Axis-chunking helps `Threads.@threads` (it removes a linear-index conversion `Threads`
# cannot avoid on its own) but hurts `@batch`, which already does the equivalent split
# internally -- chunking on top would split twice. The reductions are the exception: they
# run `CpuThreaded`'s own bands (`_last_axis_chunks` included), so their sums equal
# `CpuThreaded`'s bitwise (gpena/Bramble.jl#473).
#
# Allocation. A warm `CpuPolyester` call allocates 0 B, on any grid, bar the limits
# below. Polyester's argument box is the only source, bar a task's first reduction of an
# eltype, which allocates that task's `Threads.nthreads()` partial sums once (`_partials`).
# `@batch` copies the arguments its loop captures into a `ManualMemory.Reference` on every
# call. The box is on the stack when it holds only plain arrays and isbits values (arrays
# become `PtrArray`s), and on the heap when it holds one GC reference. So no loop captures a
# form, a space, a mesh, a sink or a `SparseMatrixCSC` whole. Bramble's own kernels and walk
# arguments go through `_batch_split` (`src/utils/batch_split.jl`): the arrays cross as
# top-level loop arguments, the rest as an isbits skeleton, and each task calls
# `_batch_rebuild` on them (`_batch_for!`, the replay, linear and bilinear sweeps,
# `_batch_mf_bands!`). The loops that never held a struct (the engines,
# `_batch_run_bands!`, `_batch_csr_spmv!`) capture only arrays and isbits values, and
# `_batch_broadcast!`, `_batch_kron_lines!` and `_batch_dot` rebuild their light structs
# inside each task (`_bc_host_rebuild`, `_kron_host_rebuild`, `_weights`). The test file's
# "allocation under CpuPolyester" testset asserts 0 B for each path.
#
# A kernel the split cannot take apart (a user closure over an array, or data it cannot
# split) crosses `@batch` whole, so that a method typed on `Vector` and an `isa Vector`
# branch in user code still see the `Vector` and not a `PtrArray` (`_splits_kernel` below).
# Whole, it would put a GC reference in the box and heap-allocate it. So the host stores the
# kernel unchanged in a slot kept for its type and the loop captures only an isbits handle
# to the slot (`_SlotRef`, `_handed` below): `avgₕ!` over such a closure allocates 0 B, and
# each task reads the kernel back. A call claims its slot atomically, so concurrent calls
# with one kernel type never share one, and gives it back after the join. A call that the
# join does not finish normally (a Ctrl-C while the host waits) retires its slot: it stays
# claimed for the rest of the session, because workers still running may read it.
#
# Boxing remains where no slot serves, and never gives a wrong result: an isbits kernel, a
# kernel type arriving when 256 types already have slots, a call finding all 64 slots of its
# type taken or retired, a `BigFloat`, a `Dict` or a `String` in a form
# (`_batch_splittable`), and a reduction over storage other than a dense vector or a
# contiguous view of one (`_Opaque`). A task body runs under an exception guard (`@_task`
# below), so a throw inside a task reaches the caller with `CpuSerial`'s exception type and
# text, never as a crash. No `CpuPolyester` call reaches `Threads.@threads` or `Threads.@spawn`
# (gpena/Bramble.jl#400).
module BramblePolyesterExt

using Bramble
using Bramble: MarkedIndicesUnion, SeparableWeights, _throw_dot_dim_error,
               _write_components!, _band_range, _scatter_point!, _scatter_linear_point!,
               CpuPolyester, _ReplayTarget, _ActionTarget, _replay_point!, _difference_band!,
               _average_band!, _centered_average_band!, _broadcast_band!, _kron_line_init!,
               _kron_line_terms!, _kron_host_raw, _kron_host_rebuild, _bc_host_raw,
               _bc_host_rebuild, _MaskedKernel, __prod, _RₕKernel, _AvgKernel, _AvgScatterKernel,
               ReplaySink, _PairReplaySink, _DiagonalReplayTarget, ActionSink,
               _PairActionSink, _batch_split, _batch_rebuild, _MFFusedPlan, _MFPass,
               _MF_BAND, _MF_NO_COLLECT, _mf_apply_parts!, _mf_host_ast, _batch_splittable,
               _ScatterCSC, _dot_band, _last_axis_chunks,
               _separable_line_band, _separable_block_band
using Polyester: Polyester, @batch
using LinearAlgebra: mul!
using SparseArrays: SparseMatrixCSC
using PrecompileTools: @setup_workload, @compile_workload

# --- _batch_dot/_batch_dot_masked (src/utils/linear_algebra.jl) -------------------- #
#
# The Polyester counterparts of `CpuThreaded`'s reductions, with the same result. Each cuts
# the work into `Threads.nthreads()` fixed bands, runs the very band bodies `CpuThreaded`
# runs (`_dot_band`, `_separable_line_band`, `_separable_block_band`), one band per `@batch`
# iteration, stores each band's sum in `partials[b]`, and sums `partials` on the host as
# `CpuThreaded` does. So the result depends on the data and `Threads.nthreads()` only, never
# on how many workers `@batch` finds free: a reduction nested in another `CpuPolyester`
# sweep, which runs every band on one worker, returns the top-level value bitwise
# (gpena/Bramble.jl#473). A `@batch reduction=((+, s),)` would not: it splits the range by
# the free workers and adds their sums, so a nested call associated differently.
#
# `partials` is this task's own `Vector{T}` of `Threads.nthreads()` entries, built on the
# task's first reduction of eltype `T` and reused after (`_partials`), so a warm call
# allocates nothing. It is task-local, never shared: two tasks reducing at once (a nested
# call on every worker of an outer sweep) each write their own, and one task never runs two
# reductions at once, since a band body calls no user code.
struct _PartialsKey{T} end

@inline function _partials(::Type{T}) where {T}
    p = get(task_local_storage(), _PartialsKey{T}(), nothing)
    p === nothing && return _new_partials(T)
    return p::Vector{T}
end

@noinline function _new_partials(::Type{T}) where {T}
    p = Vector{T}(undef, Threads.nthreads())
    task_local_storage(_PartialsKey{T}(), p)
    return p
end

# The arrays a reduction's bands read cross `@batch` as they are only when the `PtrArray`
# Polyester turns them into runs the band body's loop as the array itself does: a dense
# vector or a contiguous view of one. Any other storage (a strided view in a
# `VectorElement`, say) crosses inside `_Opaque`, which `@batch` passes through untouched,
# and each band unwraps it (`_take`, whose other methods read back a `_SlotRef` below).
# Otherwise the top-level call would run the band body on a strided `PtrArray`, and a nested
# one, which `@batch` runs on the host without converting, on the view itself: the two
# `@simd` loops associate differently (gpena/Bramble.jl#473). The opaque form costs a
# heap-boxed argument tuple per call, never a different sum.
const _PtrAlike = Union{DenseVector, Base.FastContiguousSubArray{<:Any, 1, <:DenseVector}}

struct _Opaque{A}
    a::A
end

@inline _hand_arrays(xs::Tuple{Vararg{Union{_PtrAlike, Tuple{Vararg{_PtrAlike}}}}}) = xs
@inline _hand_arrays(xs::Tuple) = map(_Opaque, xs)
@inline _take(o::_Opaque) = o.a  # the other `_take`s are with `_SlotRef` below

function Bramble._batch_dot(u::AbstractVector, v::AbstractVector, w::AbstractVector)
    (length(u) == length(v) == length(w)) ||
        _throw_dot_dim_error(length(u), length(v), length(w))
    T = promote_type(eltype(u), eltype(v), eltype(w))
    nb = Threads.nthreads()
    partials = _partials(T)
    ax = 1:length(u)
    hu, hv, hw = _hand_arrays((u, v, w))
    @batch for b in 1:nb
        @inbounds partials[b] = _dot_band(_take(hu), _take(hv), _take(hw), ax, nb, b)
    end
    return sum(partials)
end

# A mask crosses `@batch` as a tuple of its 64-bit word vectors, never as the `BitVector` or
# `MarkedIndicesUnion` itself: a struct holding a GC reference puts Polyester's argument box
# on the heap, while each word vector becomes a `PtrArray` (`_mask_chunks`). A `BitVector`
# is the one-vector case.
@inline _mask_chunks(mask::BitVector) = (mask.chunks,)
@inline _mask_chunks(mask::MarkedIndicesUnion) = mask.chunks

@inline _mask_length(mask::BitVector) = length(mask)
@inline _mask_length(mask::MarkedIndicesUnion) = mask.len

# The OR of every vector's word `k`, the union `MarkedIndicesUnion` walks
# (`_reduce_or_chunk`, utils/linear_algebra.jl). Unrolled by recursion over the tuple:
# `_reduce_or_chunk`'s `reduce` over an `ntuple` ran the masked `innerₕ` about 1.2x slower
# on a 33² grid once the words were `PtrArray`s, and this ran it no slower than v3.22.0.
@inline _mask_word(chunks::Tuple{Any}, k::Int) = @inbounds chunks[1][k]
@inline _mask_word(chunks::Tuple, k::Int) = @inbounds chunks[1][k] | _mask_word(Base.tail(chunks), k)

# One-based bit test against the word vectors. `MarkedIndicesUnion` has no `getindex` by
# design (see its docstring), so this also spares collecting it into an indexable mask,
# which would allocate on every call.
@inline function _mask_bit(chunks::Tuple, i::Int)
    word = _mask_word(chunks, ((i - 1) >> 6) + 1)
    return ((word >> ((i - 1) & 63)) & 0x1) != 0
end

# The masked band bodies: `_dot_masked_band` and `_separable_masked_band`
# (src/utils/linear_algebra.jl, src/space/inner_product.jl) over the mask's words rather
# than the mask, the same walk and the same `muladd`s in the same order, so the same sums.
@noinline function _masked_words_band(u, v, w, chunks::Tuple, ax, nb::Int, b::Int)
    T = promote_type(eltype(u), eltype(v), eltype(w))
    s = zero(T)
    @inbounds for widx in _band_range(ax, nb, b)
        word = _mask_word(chunks, widx)
        base = (widx - 1) * 64
        while word != zero(UInt64)
            i = base + trailing_zeros(word) + 1
            s = muladd(T(u[i]) * T(v[i]), T(w[i]), s)
            word &= word - 1
        end
    end
    return s
end

@noinline function _separable_masked_words_band(u, w::SeparableWeights, v, cart, chunks::Tuple,
        ax, nb::Int, b::Int)
    T = promote_type(eltype(u), eltype(w), eltype(v))
    s = zero(T)
    @inbounds for widx in _band_range(ax, nb, b)
        word = _mask_word(chunks, widx)
        base = (widx - 1) * 64
        while word != zero(UInt64)
            i = base + trailing_zeros(word) + 1
            s = muladd(T(u[i]) * T(v[i]), T(w[cart[i]]), s)
            word &= word - 1
        end
    end
    return s
end

function Bramble._batch_dot_masked(
        u::AbstractVector, v::AbstractVector, w::AbstractVector,
        mask::Union{BitVector, MarkedIndicesUnion}
)
    (length(u) == length(v) == length(w) == _mask_length(mask)) ||
        _throw_dot_dim_error(length(u), length(v), length(w), _mask_length(mask))
    T = promote_type(eltype(u), eltype(v), eltype(w))
    nb = Threads.nthreads()
    partials = _partials(T)
    chunks = _mask_chunks(mask)
    ax = 1:length(first(chunks))
    hu, hv, hw = _hand_arrays((u, v, w))
    @batch for b in 1:nb
        @inbounds partials[b] = _masked_words_band(
            _take(hu), _take(hv), _take(hw), chunks, ax, nb, b)
    end
    return sum(partials)
end

# `SeparableWeights` specializations. `inner₊(uₕ, vₕ, Val(S))` passes the weight as the
# *second* positional argument (space/inner_product.jl), so under `CpuPolyester` it is
# `_batch_dot`'s own second parameter, not third -- these dispatch on that position,
# mirroring the `CpuThreaded` specializations of `_threaded_dot`/`_threaded_dot_masked` in
# space/inner_product.jl, whose band bodies they run. The weight crosses `@batch` as its
# factor vectors and `dims`, not as the struct, for the reason the masks do above, and each
# band rebuilds it from them (`_weights`).
@inline _weights(factors::NTuple{D, VT}, dims) where {D, VT} = SeparableWeights{
    D, eltype(VT), VT}(factors, dims)

function Bramble._batch_dot(u::AbstractVector, w::SeparableWeights{D}, v::AbstractVector) where {D}
    n = length(w)
    (length(u) == n == length(v)) || _throw_dot_dim_error(length(u), n, length(v))
    T = promote_type(eltype(u), eltype(w), eltype(v))
    nb = Threads.nthreads()
    partials = _partials(T)
    dims = w.dims
    hu, hf, hv = _hand_arrays((u, w.factors, v))
    if D == 1
        ax = 1:first(dims)
        @batch for b in 1:nb
            @inbounds partials[b] = _separable_line_band(
                _take(hu), _weights(_take(hf), dims), _take(hv), ax, nb, b)
        end
    else
        # `_last_axis_chunks` clamps the block count to the last axis, as in `CpuThreaded`,
        # whose surplus partial sums stay zero.
        tail = Base.tail(dims)
        lin = LinearIndices(tail)
        blocks = _last_axis_chunks(CartesianIndices(tail), nb)
        fill!(partials, zero(T))
        @batch for b in 1:length(blocks)
            @inbounds partials[b] = _separable_block_band(
                _take(hu), _weights(_take(hf), dims), _take(hv), lin, blocks, b)
        end
    end
    return sum(partials)
end

function Bramble._batch_dot_masked(
        u::AbstractVector, w::SeparableWeights{D}, v::AbstractVector,
        mask::Union{BitVector, MarkedIndicesUnion}
) where {D}
    n = length(w)
    (length(u) == n == length(v) == _mask_length(mask)) ||
        _throw_dot_dim_error(length(u), n, length(v), _mask_length(mask))
    T = promote_type(eltype(u), eltype(w), eltype(v))
    nb = Threads.nthreads()
    partials = _partials(T)
    dims = w.dims
    hu, hf, hv = _hand_arrays((u, w.factors, v))
    cart = CartesianIndices(dims)
    chunks = _mask_chunks(mask)
    ax = 1:length(first(chunks))
    @batch for b in 1:nb
        @inbounds partials[b] = _separable_masked_words_band(
            _take(hu), _weights(_take(hf), dims), _take(hv), cart, chunks, ax, nb, b)
    end
    return sum(partials)
end

# --- Exceptions inside the split hooks' tasks ---------------------------------------- #
#
# An exception thrown inside a `@batch` task must not leave it. When the host's own chunk
# throws, Polyester's `batch.jl` skips waiting on the worker tasks and ends the
# `GC.@preserve` around their `PtrArray`s while they still run (a segfault in 8 of 12 runs of
# a missing-pattern sweep), and never frees the threads it reserved. So each split hook's
# task body catches, records the failure in an isbits `|` reduction (a scalar Polyester
# keeps in its task buffers, so success allocates nothing, where a flag array would be one
# allocation per call), and once `@batch` has joined every task the host reruns the same
# iterations in order on the original arguments. That raises the exception of the first
# iteration that fails, as a serial sweep of the same colour or band would. Work that threw
# in a task but not on the host is still refused rather than returned.
#
# `@_task function f!(args...) ... end` defines the task's work `f!` and its guard
# `f_threw(args...)`, both `@noinline`; the guard runs `f!` inside the `try` and returns
# whether it threw. LLVM compiles a function that calls `setjmp` conservatively, so the walk
# stays out of it: a `try` around the walk inlined into the `@batch` body ran a 65² fused
# product 1.2x slower (3.0 against 2.5 µs, 4 threads).
macro _task(def)
    call = def.args[1]
    name, args = call.args[1], call.args[2:end]
    guard = Symbol(chopsuffix(string(name), "!"), "_threw")
    return esc(quote
        @noinline $def
        @noinline function $guard($(args...))
            try
                $name($(args...))
            catch
                return true
            end
            return false
        end
    end)
end

@noinline function _rerun_on_host(f::F, iter) where {F}
    foreach(f, iter)
    throw(ErrorException("a CpuPolyester task threw, but its work did not throw again \
        when rerun on the host"))
end

# --- A whole kernel handed to the tasks through a typed slot ------------------------- #
#
# A kernel that crosses `@batch` whole (a user closure over an array, `_splits_kernel`
# below) would put Polyester's argument box on the heap, since the box holds a GC
# reference. Instead the host stores the kernel, unchanged, in a slot kept for its type
# `K`, and the loop captures only an isbits `_SlotRef{K}`, the slot table's index and the
# slot's. Each task reads the kernel back (`_take`). User code sees its own objects.
#
# A slot is a `Vector{K}` that holds the kernel inline or is empty. `push!` fills it within
# the capacity it keeps and `empty!` clears it, neither allocating, and `empty!` drops the
# reference so an idle slot keeps no user data alive. A `Vector{Any}` would box the
# kernel, and so would a `Vector{Union{Nothing, K}}`, whose union-typed store boxes the
# kernel before copying it in (16 B per call, measured).
#
# Each call claims its own slot with an atomic bit swap, so two calls with the same kernel
# type in flight at once (`Threads.@spawn`, a nested call from inside the user function,
# recursion) never share one. The slot is cleared and given back only when `@batch`
# returns, so after its join: a throw inside a task is caught by the `@_task` guard, and
# the join completes first. Any exception that unwinds out of the join instead (a Ctrl-C
# while the host waits) leaves workers still running, which may still read the slot, so
# the slot is retired: it stays claimed and holds its kernel for the rest of the session,
# and no later call can take it and hand those workers another call's kernel.
#
# Limits fall back to boxing, never to a wrong result: an isbits kernel (whose box stays
# on the stack anyway), a kernel type arriving when 256 types have slots already, and a
# call finding all 64 slots of its type taken or retired all cross whole as before
# gpena/Bramble.jl#476.
#
# Ordering: the kernel is stored before `@batch` launches its tasks, and Polyester's launch
# is a release that each task's acquire pairs with; the host clears the slot only after the
# join. The type table maps each kernel type to its index in `_SLOTS`. It is never mutated
# once published: registering a type takes `_SLOT_LOCK`, writes the new `_Slots` into
# `_SLOTS`, and publishes a copy of the table with the type added. So a registered type is
# found without the lock, and tasks read only `_SLOTS`, which is never resized.
mutable struct _Slots{K}
    const v::Vector{Vector{K}}
    @atomic free::UInt64     # bit `key - 1` set: slot `key` is free
    @atomic retired::UInt64  # bit `key - 1` set: slot `key` is retired
end
_Slots{K}() where {K} = _Slots{K}([sizehint!(K[], 1) for _ in 1:64], typemax(UInt64),
    UInt64(0))

struct _SlotRef{K}
    id::Int
    key::Int
end

mutable struct _SlotTable
    @atomic ids::IdDict{Type, Int}
end

const _SLOT_TYPES = 256
const _SLOT_LOCK = ReentrantLock()
const _SLOT_TABLE = _SlotTable(IdDict{Type, Int}())
const _SLOTS = Vector{Any}(nothing, _SLOT_TYPES)

# The index of `K`'s slots in `_SLOTS`, registered on first use; 0 once the table is full.
@inline function _slots_id(::Type{K}) where {K}
    ids = @atomic :acquire _SLOT_TABLE.ids
    id = get(ids, K, 0)::Int
    (id != 0 || length(ids) >= _SLOT_TYPES) && return id
    return _register_slots(K)
end

@noinline function _register_slots(::Type{K}) where {K}
    return @lock _SLOT_LOCK begin
        ids = @atomic :acquire _SLOT_TABLE.ids
        id = get(ids, K, 0)::Int
        if id == 0 && length(ids) < _SLOT_TYPES
            id = length(ids) + 1
            _SLOTS[id] = _Slots{K}()
            grown = copy(ids)
            grown[K] = id
            @atomic :release _SLOT_TABLE.ids = grown
        end
        id
    end
end

# A free slot of `k`'s type holding `k`, or `_SlotRef(0, 0)` to cross whole.
@inline function _claim(k::K) where {K}
    isbitstype(K) && return _SlotRef{K}(0, 0)
    id = _slots_id(K)
    id == 0 && return _SlotRef{K}(0, 0)
    s = _SLOTS[id]::_Slots{K}
    free = @atomic :acquire s.free
    while free != 0
        key = trailing_zeros(free) + 1
        bit = UInt64(1) << (key - 1)
        free, ok = @atomicreplace :acquire_release :acquire s.free free=>(free & ~bit)
        if ok
            push!(@inbounds(s.v[key]), k)
            return _SlotRef{K}(id, key)
        end
    end
    return _SlotRef{K}(0, 0)
end

function _release(r::_SlotRef{K}) where {K}
    s = _SLOTS[r.id]::_Slots{K}
    empty!(@inbounds(s.v[r.key]))
    @atomic :acquire_release s.free |= UInt64(1) << (r.key - 1)
    return nothing
end

function _retire(r::_SlotRef{K}) where {K}
    s = _SLOTS[r.id]::_Slots{K}
    @atomic :acquire_release s.retired |= UInt64(1) << (r.key - 1)
    return nothing
end

# The kernel a task runs: itself, or read back from its slot.
@inline _take(k) = k
@inline _take(r::_SlotRef{K}) where {K} = @inbounds (_SLOTS[r.id]::_Slots{K}).v[r.key][1]

# `run!(args..., h)` with `h` the slot holding `k`, else `k` itself; returns what `run!`
# returns (whether a task threw). `run!` holds the `@batch`, so the `try` stays out of it.
@inline function _handed(run!::R, k::K, args...) where {R, K}
    r = _claim(k)
    r.id == 0 && return run!(args..., k)
    failed = try
        run!(args..., r)
    catch
        _retire(r)
        rethrow()
    end
    _release(r)
    return failed
end

# The number of kernel types with slots, and whether every slot not retired is free and
# empty: the checks of the slot tests.
_slot_types() = length(@atomic :acquire _SLOT_TABLE.ids)

function _slots_quiescent()
    ids = @atomic :acquire _SLOT_TABLE.ids
    return all(id -> _quiescent(_SLOTS[id]), values(ids))
end

function _quiescent(s::_Slots)
    free, retired = (@atomic :acquire s.free), (@atomic :acquire s.retired)
    return all(1:length(s.v)) do key
        bit = UInt64(1) << (key - 1)
        return free & bit != 0 ? isempty(s.v[key]) : retired & bit != 0
    end
end

function __init__()
    # Start from an empty type table in every session, whatever precompilation left in it.
    @atomic :release _SLOT_TABLE.ids = IdDict{Type, Int}()
    fill!(_SLOTS, nothing)
    return nothing
end

# --- _batch_for!/_batch_axis_for! (src/utils/linear_algebra.jl) -------------------- #
#
# The counterparts of `_threaded_for!`/`_threaded_axis_for!`: `v[i] = f(i)` over `idxs`, a
# linear range for `_batch_for!` and a `CartesianIndices` for `_batch_axis_for!`, one band of
# `Threads.nthreads()` per `@batch` task (`_for_fill!`: a slab of the range, or of the last
# axis of the `CartesianIndices`, as `Polyester` splits it itself).
#
# `v::AbstractArray` (rather than an unconstrained `v`) so each of these is a genuine
# specialisation of its `src/` stub, not a redefinition of the identical, fully unconstrained
# signature -- precompilation refuses that ("Method overwriting is not permitted"), the same
# reason `ext/BrambleSparseMatricesCSRExt.jl` bounds its own `_csr_backend` with `T <: Number`.
#
# A kernel Bramble defines does not cross `@batch` whole: `_RₕKernel` holds the mesh and a
# `_MaskedKernel` its `BitVector`s, and any GC reference puts Polyester's argument box on the
# heap. `v` crosses as a top-level loop argument and the kernel as one `_batch_split`
# (`_split_or_whole` below), each task rebuilding it around the `PtrArray`s: the mesh as its
# walk state, the `_AvgKernel`/`_AvgScatterKernel` quadrature arrays and the weight build's
# `Fix1(__prod, factors)` vectors in place. A `_MaskedKernel`'s masks first become their
# 64-bit word vectors (`_for_host_raw`), each rebuilt as `_ChunkBits`, a bit test over the
# words. A single-iteration `@batch` runs its body inline on the plain `Vector`s, which the
# rebuild accepts as well.
#
# Only those kernel types split (`_splits_kernel`), and only around a user function with
# nothing to split (isbits: no captured arrays). A user closure over an array crosses whole
# with its kernel, through a typed slot (`_handed` above): rebuilt, it would capture a
# `PtrArray` in place of its `Vector`, which a method typed on `Vector` rejects and an
# `isa Vector` branch reads differently. So does any other kernel, and a kernel the split
# cannot take apart (a `BigFloat` target). A throwing kernel (a user function evaluated per
# point) is caught in the task and rerun on the host (`@_task` below), so its exception
# reaches the caller unchanged. A kernel crossing whole takes `_batch_for_whole!`, an index
# collection other than a unit range or a `CartesianIndices` (a `Vector{Int}`, a stepped
# range) a per-index `@batch`.
#
# The kernel argument is `::F where {F}` because `Base.Fix1 <: Function`: Julia does not
# specialise on an unannotated `Function` argument it only passes on, so the split was
# built from an abstractly typed kernel and boxed on the heap on every weight build.

# One mask's 64-bit words, indexed as the `BitVector` they came from (`_mask_bit` below).
struct _ChunkBits{C}
    chunks::C
end
@inline Base.getindex(b::_ChunkBits, i::Int) = _mask_bit((b.chunks,), i)

_for_host_raw(f) = f
function _for_host_raw(k::_MaskedKernel{K, <:Tuple{Vararg{BitVector}}}) where {K}
    return _MaskedKernel(k.kernel, map(m -> _ChunkBits(m.chunks), k.masks), k.zeroval)
end

# Whether a kernel of type `K` crosses `@batch` split: one Bramble defines, around a user
# function `F` that is isbits. A type test, so it folds.
_splits_kernel(::Type) = false
_splits_kernel(::Type{<:_RₕKernel{F}}) where {F} = isbitstype(F)
_splits_kernel(::Type{<:_AvgKernel{F}}) where {F} = isbitstype(F)
_splits_kernel(::Type{<:_AvgScatterKernel{F}}) where {F} = isbitstype(F)
_splits_kernel(::Type{<:Base.Fix1{typeof(__prod), <:Tuple{Vararg{AbstractVector}}}}) = true
_splits_kernel(::Type{<:_MaskedKernel{K}}) where {K} = _splits_kernel(K)

# The kernel as the tasks receive it: split when `_splits_kernel` allows it, whole otherwise.
@inline function _kernel_parts(f::F, hot...) where {F}
    raw = _for_host_raw(f)
    _splits_kernel(F) && return _split_or_whole(raw, hot...)
    return _Whole(), raw
end

# Band `b` of `n` of `idxs`: a slab of a range, or of a `CartesianIndices`' last axis.
@inline _slab(idxs::AbstractRange, n::Int, b::Int) = _band_range(idxs, n, b)
@inline function _slab(idxs::CartesianIndices, n::Int, b::Int)
    ax = idxs.indices
    return CartesianIndices((Base.front(ax)..., _band_range(last(ax), n, b)))
end

# `v[i] = k(i)` over a slab, a `CartesianIndices` one walked as `Polyester` walks it, a plain
# loop over its last axis around one over the others. A loop over the slab as one
# `CartesianIndices` ran the weight build 1.3x slower at 1025² (210 against 160 µs, 4
# threads).
@inline function _fill_slab!(v, k::K, r::AbstractUnitRange) where {K}
    for i in r
        @inbounds v[i] = k(i)
    end
    return nothing
end
@inline function _fill_slab!(v, k::K, c::CartesianIndices) where {K}
    ax = c.indices
    inner = CartesianIndices(Base.front(ax))
    for j in last(ax), J in inner

        I = CartesianIndex(J, j)
        @inbounds v[I] = k(I)
    end
    return nothing
end
@inline _for_fill!(v, k::K, idxs, n::Int, b::Int) where {K} = _fill_slab!(v, k, _slab(idxs, n, b))

# Each task copies its destination and rebuilt kernel through a local `Ref` before the loop.
# A task body reads its arguments through pointers, and LLVM reloads a field read that way
# after every store through `v`, which might alias it: the weight build's fill reloaded both
# vectors' pointers per point and took 163 µs at 1025² (4 threads, -O1), about what the
# boxed parent took. Copied into the `Ref`, the fields load once and the same build ran in
# 90 µs. Written out in each task, not behind a helper: inlined through
# one, the `Ref` was elided and the reloads came back.
@_task function _for_slab!(skel, arrays, v, idxs, n, b)
    w, k = Ref((v, _rejoin(skel, arrays)))[]
    _for_fill!(w, k, idxs, n, b)
    return nothing
end

function _batch_for_bands!(v::AbstractArray, idxs::Union{AbstractUnitRange, CartesianIndices},
        f::F) where {F}
    _offset(v, idxs) && return _batch_for_each!(v, idxs, f)
    skel, arrays = _kernel_parts(f, v)
    skel isa _Whole && return _batch_for_whole!(v, idxs, arrays)
    n = Threads.nthreads()
    failed = false
    @batch reduction=((|, failed),) for b in 1:n
        failed |= _for_slab_threw(skel, arrays, v, idxs, n, b)
    end
    failed && _rerun_on_host(b -> _for_fill!(v, f, idxs, n, b), 1:n)
    return nothing
end

# A whole kernel's loop runs one guarded slab per iteration, not one guarded index: a
# guarded call per index ran an `avgₕ!` closure over an array 5% slower than the parent
# of gpena/Bramble.jl#433. Its box should stay no bigger than the parent's, which captured
# `v`, the kernel and `idxs`. When `idxs` covers `v` (every `Rₕ!`/`avgₕ!` sweep), the loop
# runs over `Base.OneTo(Threads.nthreads())` and each task rebuilds its slab from `v`'s own
# indices (`_full`), so it captures `v`, the kernel and that range, and no `idxs`. Any other
# `idxs` crosses as `_Slabs`, the slabs as one loop range: that captures `idxs` plus the
# index range `@batch` takes of any array it loops over, which makes the box larger than the
# parent's. Looping over `1:n` with `idxs` captured would make it larger still.
struct _Lin end
struct _Cart end
@inline _cover(v, idxs::AbstractUnitRange) = !Base.has_offset_axes(v) &&
                                             idxs == Base.OneTo(length(v)) ? _Lin() : nothing
@inline _cover(v, idxs::CartesianIndices) = idxs == CartesianIndices(v) ? _Cart() : nothing
@inline _cover(v, idxs) = nothing
@inline _full(v, ::_Lin) = Base.OneTo(length(v))
@inline _full(v, ::_Cart) = CartesianIndices(size(v))

struct _Slabs{T, R} <: AbstractVector{T}
    idxs::R
end
_Slabs(idxs) = _Slabs{typeof(_slab(idxs, 1, 1)), typeof(idxs)}(idxs)
Base.size(::_Slabs) = (Threads.nthreads(),)
Base.@propagate_inbounds Base.getindex(s::_Slabs, b::Int) = _slab(
    s.idxs, Threads.nthreads(), b)

# The whole kernel `h` reaches each task as itself or through its slot (`_handed` above).
@_task function _whole_slab!(v, h, r)
    _fill_slab!(v, _take(h), r)
    return nothing
end

@_task function _whole_part!(v, h, kind, b)
    _fill_slab!(v, _take(h), _slab(_full(v, kind), Threads.nthreads(), b))
    return nothing
end

@noinline function _cover_run!(v, kind, h::H) where {H}
    failed = false
    @batch reduction=((|, failed),) for b in Base.OneTo(Threads.nthreads())
        failed |= _whole_part_threw(v, h, kind, b)
    end
    return failed
end

# `idxs` covering `v`, as `kind` says: the slabs rebuilt from `v` in each task.
function _batch_for_cover!(v::AbstractArray, kind, k::K) where {K}
    failed = _handed(_cover_run!, k, v, kind)
    failed && _rerun_on_host(b -> _fill_slab!(v, k, _slab(_full(v, kind),
            Threads.nthreads(), b)), 1:Threads.nthreads())
    return nothing
end

@noinline function _slabs_run!(v, idxs, h::H) where {H}
    failed = false
    @batch reduction=((|, failed),) for r in _Slabs(idxs)
        failed |= _whole_slab_threw(v, h, r)
    end
    return failed
end

# A kernel crossing whole, unsplit, one slab per iteration behind the same guard.
function _batch_for_whole!(v::AbstractArray, idxs, k::K) where {K}
    kind = _cover(v, idxs)
    kind === nothing || return _batch_for_cover!(v, kind, k)
    failed = _handed(_slabs_run!, k, v, idxs)
    failed && _rerun_on_host(r -> _fill_slab!(v, k, r), _Slabs(idxs))
    return nothing
end

# Any other index collection (a `Vector{Int}`, a stepped range), and any destination or
# index range with offset axes, which the slabs (`_band_range` counts positions from 1) and
# the covering loop (`_full` rebuilds 1-based indices) would misplace: one index per
# iteration, the kernel crossing whole, behind the same guard.
@inline _offset(arrays...) = any(Base.has_offset_axes, arrays)

# `@batch` splits its range assuming a positive step: on a descending range (`9:-1:1`) it
# skips every index, and on a `CartesianIndices` with a descending axis it hangs. Every
# `@batch ... for x in idxs|bidx` below walks `_ascending(idxs)` instead, the same indices
# in ascending order. Order does not matter, since each sweep writes distinct entries.
# `_batch_scatter_for!`'s slabs are cut from `_ascending(idxs)` too: `_band_range` of a
# descending `StepRange{UInt, Int}` comes back with a corrupt step, which `@batch` reads as
# empty. `_rerun_on_host` keeps the original collection, so a throwing kernel raises the
# same first exception as `CpuSerial`.
@inline _ascending(r::AbstractUnitRange) = r
@inline _ascending(r::AbstractRange) = step(r) < 0 ? reverse(r) : r
@inline _ascending(c::CartesianIndices) = CartesianIndices(map(_ascending, c.indices))
@inline _ascending(idxs) = idxs

@_task function _for_one!(v, h, i)
    @inbounds v[i] = _take(h)(i)
    return nothing
end

@noinline function _each_run!(v, idxs, h::H) where {H}
    failed = false
    @batch reduction=((|, failed),) for i in _ascending(idxs)
        failed |= _for_one_threw(v, h, i)
    end
    return failed
end

_batch_for_bands!(v::AbstractArray, idxs, f::F) where {F} = _batch_for_each!(v, idxs, f)

function _batch_for_each!(v::AbstractArray, idxs, f::F) where {F}
    k = _for_host_raw(f)
    failed = _handed(_each_run!, k, v, idxs)
    failed && _rerun_on_host(i -> (@inbounds v[i] = k(i)), idxs)
    return nothing
end

Bramble._batch_for!(v::AbstractArray, idxs, f::F) where {F} = _batch_for_bands!(v, idxs, f)

function Bramble._batch_axis_for!(v::AbstractArray, idxs::CartesianIndices, f::F) where {F}
    return _batch_for_bands!(v, idxs, f)
end

# --- _batch_scatter_for! (src/utils/linear_algebra.jl) ------------------------------ #
#
# Mirrors `_threaded_scatter_for!`: `g` is evaluated once per index and its tuple of results
# is unpacked into every destination array in `mats` by `_write_components!` (unrolled at
# compile time, so this stays a single scalar write per component, not a tuple allocation).
# `mats` crosses as top-level arrays and `g` as `_batch_for!`'s kernels do above.

# `idxs::AbstractRange` and `idxs::AbstractArray` both specialise the stub's fully
# unconstrained signature (`_batch_scatter_for!(mats::Tuple, idxs, g)`). The range method
# serves `project!` (operators/projection.jl), which passes `1:n`; any other index set, a
# `Vector{Int}` or a `CartesianIndices`, takes the per-index loop of the array method below.
@_task function _scatter_slab!(skel, arrays, mats, idxs, n, b)
    # A local `Ref` copy, for the reason `_for_slab!` gives.
    ms, k = Ref((mats, _rejoin(skel, arrays)))[]
    for i in _band_range(idxs, n, b)
        @inbounds _write_components!(ms, k(i), i)
    end
    return nothing
end

# A kernel crossing whole: one slab per iteration, as `_batch_for_whole!` above.
@inline function _scatter_range!(mats, k::K, r) where {K}
    for i in r
        @inbounds _write_components!(mats, k(i), i)
    end
    return nothing
end

@_task function _scatter_whole_slab!(mats, h, r)
    _scatter_range!(mats, _take(h), r)
    return nothing
end

@_task function _scatter_whole_part!(mats, h, b)
    _scatter_range!(mats, _take(h),
        _band_range(Base.OneTo(length(mats[1])), Threads.nthreads(), b))
    return nothing
end

@noinline function _scatter_cover_run!(mats, h::H) where {H}
    hit = false
    @batch reduction=((|, hit),) for b in Base.OneTo(Threads.nthreads())
        hit |= _scatter_whole_part_threw(mats, h, b)
    end
    return hit
end

@noinline function _scatter_slabs_run!(mats, idxs, h::H) where {H}
    failed = false
    @batch reduction=((|, failed),) for r in _Slabs(idxs)
        failed |= _scatter_whole_slab_threw(mats, h, r)
    end
    return failed
end

function _batch_scatter_whole!(mats::Tuple, idxs, k::K) where {K}
    if _cover(mats[1], idxs) isa _Lin
        n = Threads.nthreads()
        hit = _handed(_scatter_cover_run!, k, mats)
        hit && _rerun_on_host(b -> _scatter_range!(mats, k,
                _band_range(Base.OneTo(length(mats[1])), n, b)), 1:n)
        return nothing
    end
    failed = _handed(_scatter_slabs_run!, k, mats, _ascending(idxs))
    failed && _rerun_on_host(i -> (@inbounds _write_components!(mats, k(i), i)), idxs)
    return nothing
end

# Destinations or an index range with offset axes: one index per iteration, as
# `_batch_for_each!` above.
@_task function _scatter_one!(mats, h, i)
    @inbounds _write_components!(mats, _take(h)(i), i)
    return nothing
end

@noinline function _scatter_each_run!(mats, idxs, h::H) where {H}
    failed = false
    @batch reduction=((|, failed),) for i in _ascending(idxs)
        failed |= _scatter_one_threw(mats, h, i)
    end
    return failed
end

function _batch_scatter_each!(mats::Tuple, idxs, g::G) where {G}
    k = _for_host_raw(g)
    failed = _handed(_scatter_each_run!, k, mats, idxs)
    failed && _rerun_on_host(i -> (@inbounds _write_components!(mats, k(i), i)), idxs)
    return nothing
end

function Bramble._batch_scatter_for!(mats::Tuple, idxs::AbstractRange, g::G) where {G}
    _offset(idxs, mats...) && return _batch_scatter_each!(mats, idxs, g)
    skel, arrays = _kernel_parts(g, mats...)
    skel isa _Whole && return _batch_scatter_whole!(mats, idxs, arrays)
    a, n = _ascending(idxs), Threads.nthreads()
    failed = false
    @batch reduction=((|, failed),) for b in 1:n
        failed |= _scatter_slab_threw(skel, arrays, mats, a, n, b)
    end
    failed && _rerun_on_host(i -> (@inbounds _write_components!(mats, g(i), i)), idxs)
    return nothing
end

function Bramble._batch_scatter_for!(mats::Tuple, idxs::AbstractArray, g::G) where {G}
    return _batch_scatter_each!(mats, idxs, g)
end

# --- _batch_bilinear_colour_sweep!/_batch_bilinear_band_sweep! (src/assembly/bilinear_execution.jl) --- #
#
# Direct translations of `_sweep_bilinear_colour!`/`_sweep_band_colour!`'s `CpuThreaded`
# bodies: each colour/band is independent by construction (the caller's colouring already
# guarantees no two concurrently-swept points target the same matrix entry), so swapping the
# scheduler changes nothing about correctness. `_scatter_point!` is the single shared entry
# rule both this and the `CpuThreaded` sweep call, so the two can never drift apart on what a
# stencil tap writes.
#
# `A::AbstractMatrix` (matching the `CpuThreaded` reference signature in
# bilinear_execution.jl), rather than the stub's fully unconstrained `A`, so this is a genuine
# specialisation of the `src/` stub and not a redefinition of the identical signature.
# Redefining it aborts the module body partway, leaving everything defined after it
# uninstalled, so a bilinear `assemble!` under `CpuPolyester` reaches the `src/` error stub
# with Polyester loaded.
#
# Each signature here must match the `CpuThreaded` reference in `bilinear_execution.jl`.
# A mismatched arity makes dispatch fall through silently to the `src/` error stub, whose
# message ("you forgot to load Polyester") is then wrong. Diff against
# `bilinear_execution.jl` whenever it changes.
#
# A `SparseMatrixCSC` does not cross `@batch` whole, since it holds GC references, which put
# the argument box on the heap. Its `colptr`, `rowval` and `nzval` cross as top-level loop
# arguments, and each task rebuilds them as a `_ScatterCSC` (bilinear_traversal.jl), whose
# position search and write are the `SparseMatrixCSC` ones. `sp`, `term` and `mesh_markers`
# cross as one `_batch_split` (`_split_or_whole` below), as in the replay hooks. The caller
# (`_sweep_bilinear!`) has already bound the term on the host (`_bind_walk`), so no `Symbol`
# region reaches the split. A single-iteration `@batch` runs its body inline on the plain
# `Vector`s, which `_ScatterCSC` and the rebuild accept as well. Any other matrix type (a
# dense `Matrix`, a device matrix with its mirror) crosses whole, as before. No local is
# named `cp`, because `@batch` takes a name bound in `Base` for a global and does not pass
# it, so the loop would close over the local and fail to compile to a `cfunction`.

function Bramble._batch_bilinear_colour_sweep!(
        A::AbstractMatrix, sp, term, idxs, lin_indices, mesh_markers, row_offset, col_offset, α
)
    @batch for I in _ascending(idxs)
        _scatter_point!(A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset, α)
    end
    return nothing
end

function Bramble._batch_bilinear_band_sweep!(
        A::AbstractMatrix, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, row_offset, col_offset, α
)
    @batch for b in _ascending(bidx)
        for I in CartesianIndices((rest..., _band_range(ax, nbands, b)))
            _scatter_point!(
                A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset, α
            )
        end
    end
    return nothing
end

@_task function _csc_point!(skel, arrays, cptr, rval, nzv, I, lin_indices, row_offset,
        col_offset, α)
    s, tm, mm = _rejoin(skel, arrays)
    _scatter_point!(_ScatterCSC(cptr, rval, nzv), tm, s, I, lin_indices, mm, row_offset,
        col_offset, α)
    return nothing
end

function Bramble._batch_bilinear_colour_sweep!(
        A::SparseMatrixCSC, sp, term, idxs, lin_indices, mesh_markers, row_offset, col_offset, α
)
    cptr, rval, nzv = A.colptr, A.rowval, A.nzval
    skel, arrays = _split_or_whole((sp, term, mesh_markers), cptr, rval, nzv)
    failed = false
    @batch reduction=((|, failed),) for I in _ascending(idxs)
        failed |= _csc_point_threw(skel, arrays, cptr, rval, nzv, I, lin_indices,
            row_offset, col_offset, α)
    end
    failed && _rerun_on_host(idxs) do I
        _scatter_point!(A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset, α)
    end
    return nothing
end

@_task function _csc_band!(skel, arrays, cptr, rval, nzv, rest, ax, nbands, b,
        lin_indices, row_offset, col_offset, α)
    s, tm, mm = _rejoin(skel, arrays)
    Ab = _ScatterCSC(cptr, rval, nzv)
    for I in CartesianIndices((rest..., _band_range(ax, nbands, b)))
        _scatter_point!(Ab, tm, s, I, lin_indices, mm, row_offset, col_offset, α)
    end
    return nothing
end

function Bramble._batch_bilinear_band_sweep!(
        A::SparseMatrixCSC, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers,
        row_offset, col_offset, α
)
    cptr, rval, nzv = A.colptr, A.rowval, A.nzval
    skel, arrays = _split_or_whole((sp, term, mesh_markers), cptr, rval, nzv)
    failed = false
    @batch reduction=((|, failed),) for b in _ascending(bidx)
        failed |= _csc_band_threw(skel, arrays, cptr, rval, nzv, rest, ax, nbands, b,
            lin_indices, row_offset, col_offset, α)
    end
    failed && _rerun_on_host(bidx) do b
        for I in CartesianIndices((rest..., _band_range(ax, nbands, b)))
            _scatter_point!(
                A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset, α)
        end
    end
    return nothing
end

# --- _threaded_replay_policy/_batch_bilinear_band_replay!/_batch_bilinear_colour_replay! ---- #
# (src/assembly/bilinear_execution.jl)
#
# Opts `CpuPolyester` into the warmed-refill replay `CpuThreaded` already gets: without this,
# `_leaf_replays` never answers `true` for a `CpuPolyester` leaf, so its units keep searching
# even once a recording exists. `target::_ReplayTarget` -- rather than the stub's unconstrained
# `target` -- is what makes each of these a genuine specialisation of its `src/` stub, the same
# `A::AbstractMatrix` reasoning the sweep hooks above give. A matrix-free product
# (`MatrixFreeOperator`, src/assembly/matrix_free.jl) sweeps through
# the same two hooks with an `_ActionTarget`, the sink adding `α * w * x[col]` into
# `y[row]`: the colouring and the per-point step (`_replay_point!`) are the same.
#
# Neither hook lets `@batch` capture the target, the space or the term whole: each holds GC
# references (a mesh, a sink's vectors), which put the argument box on the heap. The target's
# storage, `nzval` for a replay target and `y`, `x` for an action target, crosses as
# top-level loop arguments (`_replay_parts`): wrapped in a struct, a `PtrArray` ran 1.11×
# slower in the prototype (commit f0f2d538; gpena/Bramble.jl#437 item 3).
# The target's other fields cross with `sp`, `term` and `mesh_markers` as one
# `_batch_split`, and each task rebuilds the four (`_batch_rebuild`) and joins the target
# back around its storage (`_replay_join`). A single-iteration `@batch` runs its body
# inline on the plain `Vector`s, which the rebuild and the join accept as well. The caller
# (`_sweep_bilinear!`) has already bound the term and checked the weights on the host.
#
# A value `_batch_split` cannot take apart (a `BigFloat` coefficient, an array of
# `BigFloat`s) does not throw: `_split_or_whole` decides from the types, at compile time,
# that the split does not apply, and the loop captures the parts whole, which boxes as every
# hook did before gpena/Bramble.jl#437. The storage vectors join the test (`hot`), so a
# `BigFloat` matrix or vector falls back too.
Bramble._threaded_replay_policy(::CpuPolyester) = true

# The skeleton of parts that cross `@batch` whole (`_split_or_whole`).
struct _Whole end

# `parts` as `_batch_split` gives them when every leaf of `parts` and `hot` splits, otherwise
# `parts` itself behind `_Whole`; `_rejoin` is the inverse in each task.
@inline function _split_or_whole(parts, hot...)
    _batch_splittable(typeof((parts, hot))) && return _batch_split(parts)
    return _Whole(), parts
end
@inline _rejoin(skel, arrays) = _batch_rebuild(skel, arrays)
@inline _rejoin(::_Whole, parts) = parts

# Which target type `_replay_join` rebuilds, an isbits tag the split keeps as it is.
struct _Hot{K} end

# A target as its storage vectors (the second `nothing` for a replay target) and the tuple
# of its other fields, led by its tag.
@inline function _replay_parts(t::ReplaySink)
    return t.nzval, nothing, (_Hot{:replay}(), t.point_ptr, t.positions, t.α)
end
@inline function _replay_parts(t::_PairReplaySink)
    cold = (_Hot{:pair}(), t.point_ptr, t.positions, t.positions_t, t.α1, t.α2, t.half)
    return t.nzval, nothing, cold
end
@inline function _replay_parts(t::_DiagonalReplayTarget)
    cold = (_Hot{:diagonal}(), t.point_ptr, t.positions, t.base, t.stride, t.interior, t.α)
    return t.nzval, nothing, cold
end
@inline _replay_parts(s::ActionSink) = s.y, s.x, (_Hot{:action}(), s.α, s.mask, s.geom)
@inline function _replay_parts(s::_PairActionSink)
    cold = (_Hot{:pair_action}(), s.α1, s.α2, s.mask, s.dr, s.dc, s.half, s.geom)
    return s.y, s.x, cold
end

# The inverse of `_replay_parts`, around the arrays a task received.
@inline function _replay_join(h1, _, c::Tuple{_Hot{:replay}, Vararg})
    _, pp, pos, α = c
    return ReplaySink{typeof(h1), typeof(pp), typeof(α)}(h1, pp, pos, α)
end
@inline function _replay_join(h1, _, c::Tuple{_Hot{:pair}, Vararg})
    _, pp, pos, pos_t, α1, α2, half = c
    return _PairReplaySink{typeof(h1), typeof(pp), typeof(α1), typeof(α2)}(
        h1, pp, pos, pos_t, α1, α2, half)
end
@inline _replay_join(h1, _, c::Tuple{_Hot{:diagonal}, Vararg}) = _DiagonalReplayTarget(
    h1, Base.tail(c)...)
@inline _replay_join(y, x, c::Tuple{_Hot{:action}, Vararg}) = ActionSink(
    y, x, Base.tail(c)...)
@inline _replay_join(y, x, c::Tuple{_Hot{:pair_action}, Vararg}) = _PairActionSink(
    y, x, Base.tail(c)...)

@_task function _replay_band!(skel, arrays, h1, h2, rest, ax, nbands, b, lin_indices,
        row_offset, col_offset)
    c, s, tm, mm = _rejoin(skel, arrays)
    t = _replay_join(h1, h2, c)
    for I in CartesianIndices((rest..., _band_range(ax, nbands, b)))
        _replay_point!(t, tm, s, I, lin_indices, mm, row_offset, col_offset)
    end
    return nothing
end

function Bramble._batch_bilinear_band_replay!(
        target::Union{_ReplayTarget, _ActionTarget}, sp, term, ax, bidx, nbands, rest,
        lin_indices, mesh_markers, row_offset, col_offset
)
    h1, h2, cold = _replay_parts(target)
    skel, arrays = _split_or_whole((cold, sp, term, mesh_markers), h1, h2)
    failed = false
    @batch reduction=((|, failed),) for b in _ascending(bidx)
        failed |= _replay_band_threw(skel, arrays, h1, h2, rest, ax, nbands, b,
            lin_indices, row_offset, col_offset)
    end
    failed && _rerun_on_host(bidx) do b
        for I in CartesianIndices((rest..., _band_range(ax, nbands, b)))
            _replay_point!(target, term, sp, I, lin_indices, mesh_markers, row_offset,
                col_offset)
        end
    end
    return nothing
end

@_task function _replay_one!(skel, arrays, h1, h2, I, lin_indices, row_offset, col_offset)
    c, s, tm, mm = _rejoin(skel, arrays)
    _replay_point!(
        _replay_join(h1, h2, c), tm, s, I, lin_indices, mm, row_offset, col_offset)
    return nothing
end

function Bramble._batch_bilinear_colour_replay!(
        target::Union{_ReplayTarget, _ActionTarget}, sp, term, idxs, lin_indices, mesh_markers,
        row_offset, col_offset
)
    h1, h2, cold = _replay_parts(target)
    skel, arrays = _split_or_whole((cold, sp, term, mesh_markers), h1, h2)
    failed = false
    @batch reduction=((|, failed),) for I in _ascending(idxs)
        failed |= _replay_one_threw(skel, arrays, h1, h2, I, lin_indices, row_offset,
            col_offset)
    end
    failed && _rerun_on_host(idxs) do I
        _replay_point!(target, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset)
    end
    return nothing
end

# --- _batch_linear_colour_sweep!/_batch_linear_band_sweep! (src/assembly/linear.jl) ---- #
#
# As above, for the right-hand-side sweep: `_scatter_linear_point!` is the shared entry rule
# with `_sweep_colour!`/`_sweep_linear_band_colour!`'s `CpuThreaded` bodies.
#
# `b::AbstractVector` (matching the `CpuThreaded` reference signature in linear.jl), for the
# same reason the bilinear pair above is now constrained on `A`.
#
# As the replay hooks do, neither captures the space or the term whole: `b` crosses as a
# top-level loop argument, and `sp`, `term` and `mesh_markers` as one `_batch_split` that
# each task rebuilds (`_batch_rebuild`), so the box stays on the stack. The caller
# (`_sweep_parallel!`) has already bound the term to its leaf's marker ids and checked the
# weights on the host (`_bind_walk`), so no `Symbol` region reaches the split. A
# single-iteration `@batch` runs its body inline on the plain `Vector`s, which the rebuild
# accepts as well. Parts that do not split cross whole (`_split_or_whole` above).

@_task function _linear_point!(skel, arrays, b, I, lin_indices, offset, α)
    s, tm, mm = _rejoin(skel, arrays)
    _scatter_linear_point!(b, s, tm, I, lin_indices, mm, offset, α)
    return nothing
end

function Bramble._batch_linear_colour_sweep!(b::AbstractVector, sp, term, idxs, lin_indices, mesh_markers, offset, α)
    skel, arrays = _split_or_whole((sp, term, mesh_markers), b)
    failed = false
    @batch reduction=((|, failed),) for I in _ascending(idxs)
        failed |= _linear_point_threw(skel, arrays, b, I, lin_indices, offset, α)
    end
    failed && _rerun_on_host(idxs) do I
        _scatter_linear_point!(b, sp, term, I, lin_indices, mesh_markers, offset, α)
    end
    return nothing
end

@_task function _linear_band!(skel, arrays, b, rest, ax, nbands, k, lin_indices, offset, α)
    s, tm, mm = _rejoin(skel, arrays)
    for I in CartesianIndices((rest..., _band_range(ax, nbands, k)))
        _scatter_linear_point!(b, s, tm, I, lin_indices, mm, offset, α)
    end
    return nothing
end

function Bramble._batch_linear_band_sweep!(
        b::AbstractVector, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, offset, α
)
    skel, arrays = _split_or_whole((sp, term, mesh_markers), b)
    failed = false
    @batch reduction=((|, failed),) for k in _ascending(bidx)
        failed |= _linear_band_threw(skel, arrays, b, rest, ax, nbands, k, lin_indices,
            offset, α)
    end
    failed && _rerun_on_host(bidx) do k
        for I in CartesianIndices((rest..., _band_range(ax, nbands, k)))
            _scatter_linear_point!(b, sp, term, I, lin_indices, mesh_markers, offset, α)
        end
    end
    return nothing
end

# --- _batch_difference_engine!/_batch_average_engine!/_batch_centered_average_engine! ---- #
# (src/operators/difference.jl, src/operators/average.jl)
#
# `CpuPolyester`'s counterpart of `_threaded_difference_engine!`/`_threaded_average_engine!`/
# `_threaded_centered_average_engine!`: one `_difference_band!`/`_average_band!`/
# `_centered_average_band!` per band under `@batch`, exactly the shape of the sweeps above.
# `out::AbstractVector` (matching the `CpuThreaded` reference bodies), the same reasoning as
# the sweep hooks above, so this is a genuine specialisation of the `src/` stub rather than a
# redefinition of its fully unconstrained signature.

function Bramble._batch_difference_engine!(out::AbstractVector, in_ref, h, dims, dir, dim_val)
    n = Threads.nthreads()
    @batch for b in 1:n
        _difference_band!(out, in_ref, h, dims, dir, dim_val, n, b)
    end
    return nothing
end

function Bramble._batch_average_engine!(out::AbstractVector, in_ref, dims, dir, dim_val)
    n = Threads.nthreads()
    @batch for b in 1:n
        _average_band!(out, in_ref, dims, dir, dim_val, n, b)
    end
    return nothing
end

function Bramble._batch_centered_average_engine!(out::AbstractVector, in_ref, dims, dim_val)
    n = Threads.nthreads()
    @batch for b in 1:n
        _centered_average_band!(out, in_ref, dims, dim_val, n, b)
    end
    return nothing
end

# --- _batch_run_bands! (src/operators/vector_calculus.jl) -------------------- #
#
# `CpuPolyester`'s `_run_bands!`: unlike the three engine-specific hooks above, `f` here is
# whichever accumulating engine (`_accumulate_backward!`, `_accumulate_centered!`,
# `_avg_backward_inplace!`, ...) the caller passed to `_run_bands!` itself, so this stays
# generic over `f` rather than naming one. `out::AbstractVector` is always the first of
# `args...` at every `_run_bands!` call site (vector_calculus.jl), the same constraint that
# makes this a genuine specialisation of the `src/` stub rather than a redefinition of its
# fully unconstrained signature.
function Bramble._batch_run_bands!(f::F, nbands::Int, out::AbstractVector, rest::Vararg{Any, N}) where {F, N}
    @batch for b in 1:nbands
        f(out, rest..., nbands, b)
    end
    return nothing
end

# --- _batch_mf_bands! (src/assembly/matrix_free.jl) -------------------- #
#
# `CpuPolyester`'s fused matrix-free sweep runs one band per `@batch` task, each walking
# every unit over its band, as `_mf_band_task!` does. No plan, form or sink crosses whole,
# since each holds GC references (the spaces' meshes, the form's mutable cache), which put
# the argument box on the heap. `y` and `x` cross as top-level loop arguments, as the replay
# hooks' do (`_replay_parts`), the plan's band cut as its isbits fields, and the sink's `α`
# and mask with the form's walked parts and the plan's geometry as one `_batch_split`. Each
# task rebuilds them and walks `_mf_apply_parts!`. The form's regions are bound on the host
# (`_mf_host_ast`), because a `Symbol` region would not split and the task's rebuilt walk
# state has no label table to bind it against. The rebuilt geometry still names its mesh by
# the walk state's `uid` and version, so it answers the rebuilt space (`_mf_evaluator`). A
# single-iteration `@batch` runs its body inline on the plain `Vector`s, which the rebuild
# accepts as well. The caller (`_mf_product!`) has checked on the host that the plan is the
# product's form's. Parts that do not split cross whole, the host-bound AST among them
# (`_split_or_whole` above).
@_task function _mf_band!(skel, arrays, y, x, len, nbands, b, omin, omax)
    (α, mask), Wu, Wv, ast, geom = _rejoin(skel, arrays)
    own = _band_range(1:len, nbands, b)
    _mf_apply_parts!(_MFPass(_MF_BAND, own, omin, omax, _MF_NO_COLLECT),
        ActionSink(y, x, α, mask, geom), Wu, Wv, ast)
    return nothing
end

function Bramble._batch_mf_bands!(s::ActionSink, plan::_MFFusedPlan{D}, nbands::Int) where {D}
    a = plan.form
    y, x = s.y, s.x
    skel, arrays = _split_or_whole(
        ((s.α, s.mask), a.trial_space, a.test_space, _mf_host_ast(a), plan.geom), y, x)
    len, omin, omax = plan.dims[D], plan.omin, plan.omax
    failed = false
    @batch reduction=((|, failed),) for b in 1:nbands
        failed |= _mf_band_threw(skel, arrays, y, x, len, nbands, b, omin, omax)
    end
    failed && _rerun_on_host(1:nbands) do b
        _mf_apply_parts!(
            _MFPass(_MF_BAND, _band_range(1:len, nbands, b), omin, omax, _MF_NO_COLLECT),
            ActionSink(y, x, s.α, s.mask, plan.geom), a.trial_space, a.test_space,
            _mf_host_ast(a))
    end
    return nothing
end

# --- _batch_broadcast! (src/space/vectorelement.jl) --------------------------------- #
#
# `CpuPolyester`'s counterpart of `_threaded_broadcast!`: one `_broadcast_band!` per band
# under `@batch`, exactly the shape of the engine hooks above. `v::AbstractVector` (matching
# the `CpuThreaded` reference body), the same reasoning as those hooks, so this is a genuine
# specialisation of the `src/` stub rather than a redefinition of its fully unconstrained
# signature.
#
# Only plain arrays and isbits values cross `@batch`, the tree's leaves as `_bc_host_raw`
# gives them, which each task rebuilds into a `Broadcasted` with `_bc_host_rebuild`
# (vectorelement.jl), so a warm broadcast allocates 0 B. Capturing `bc` whole fails, since
# `@batch` gc-preserves every free variable through `StrideArraysCore.object_and_preserve`,
# whose `Broadcast.Broadcasted` method rebuilds the tree through the 3-argument
# `Broadcasted(f, args, axes)` constructor, and that recomputes the style over the
# already-`preprocess`ed args, where an `Extruded` leaf has no `BroadcastStyle`
# (`MethodError: no method matching ndims(::Type{Extruded{...}})`). A `Ref` box around `bc`
# works but puts Polyester's box on the heap (128 B per call), so it was rejected.
function Bramble._batch_broadcast!(v::AbstractVector, bc, ax)
    n = Threads.nthreads()
    raw = _bc_host_raw(bc)
    @batch for b in 1:n
        _broadcast_band!(v, _bc_host_rebuild(raw), ax, n, b)
    end
    return nothing
end

# --- _batch_kron_lines! (src/assembly/kronecker.jl) -------------------------------- #
#
# `CpuPolyester`'s host `KroneckerLinearOperator` product: one grid line along axis 1 per
# `@batch` iteration, each line `_kron_line_init!` then `_kron_line_terms!`, exactly the
# serial `_kron_fused!` loop.
#
# Only plain arrays and isbits values cross `@batch`. Those are `y`, `x`, the coefficients,
# `β`, the strides, the line indices, `m`, and each term's factors as `_kron_host_raw`
# gives them (nested tuples of diagonals, CSC `colptr`/`rowval`/`nzval` and `_KronTridiag`
# bands). `@batch` turns every one of those arrays into a `PtrArray` under its own
# `GC.@preserve`, so its argument box holds no GC reference and stays on the stack, and a
# warm product allocates 0 B. Each task rebuilds the light structs (`Diagonal`, `_KronCSC`,
# `_KronTridiag`) around the `PtrArray`s with `_kron_host_rebuild`. Rejected: a `Ref` box
# around the terms (it put Polyester's box on the heap, 272 B per call), and capturing the
# terms whole (the kernels ran at half speed on a struct-wrapped `PtrArray`,
# benchmark/batch_survey.jl candidate 1). `y::AbstractVector` and
# `lines::CartesianIndices` make this a genuine specialisation of the `src/` stub, the same
# reasoning as the hooks above.
function Bramble._batch_kron_lines!(
        y::AbstractVector, terms::Tuple, cs::Tuple, x::AbstractVector, β, ss::Tuple,
        lines::CartesianIndices, m::Int
)
    raw = map(_kron_host_raw, terms)
    @batch for o in 1:length(lines)
        off = (o - 1) * m
        _kron_line_init!(y, β, off, m)
        _kron_line_terms!(y, map(_kron_host_rebuild, raw), cs, x, Tuple(lines[o]), ss, off, m)
    end
    return nothing
end

# --- _batch_csr_spmv! (src/problems/semidiscrete_rhs.jl) ---------------------------- #
#
# `SemidiscretizeRHS`'s product under `CpuPolyester`: row `i` sums
# `nzval[perm[k]] * u[colval[k]]` over its compressed-row slice and subtracts it from `du[i]`,
# so each iteration writes one entry of `du` and none races. `nzval` is `nonzeros(A)` itself,
# gathered through `perm`, so the values are always `A`'s current ones. The arrays are plain arguments, captured directly: `@batch`
# turns each into a `PtrArray`, and with no struct to rebuild around them the row loop runs
# at full speed (benchmark/batch_survey.jl, candidate 3). The accumulator starts from
# `z`, the zero of the promoted element type, computed outside the loop so the sum is
# type-stable for `Dual` and other element types.
function Bramble._batch_csr_spmv!(
        du::AbstractVector, rowptr::AbstractVector{<:Integer},
        colval::AbstractVector{<:Integer}, perm::AbstractVector{<:Integer},
        nzval::AbstractVector, u::AbstractVector
)
    z = zero(promote_type(eltype(du), eltype(nzval), eltype(u)))
    @batch for i in 1:(length(rowptr) - 1)
        acc = z
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
            acc += nzval[perm[k]] * u[colval[k]]
        end
        @inbounds du[i] -= acc
    end
    return nothing
end

# The threaded matrix-free apply: `matrix_free_operator` under
# `CpuPolyester` in 1D-3D on non-uniform meshes, in the forms the tests and users reach --
# the policy passed to the operator or carried by the mesh backend, with and without
# `dirichlet = :boundary`, and the 3- and 5-argument `mul!`; then the other first-call
# paths of benchmark/polyester_first_call.jl that cost 100 ms or more.
#
# The calls of benchmark/polyester_first_call.jl, except its `_newf` rows, which time a user
# function no workload can name, made from inside functions that take their arguments as
# ordinary values. Code in a user's function is inferred statically, so its
# call sites carry the partly abstract types inference sees there (`<:Tuple{...}`,
# unbound `_MFFusedPlan` parameters). Calls made at top level are dispatched at run time on
# the concrete values and cache other instances, which leave the first call to infer these.
function _pw_first_calls(u, w, g, a, l, x, y)
    avgₕ!(u, g)
    innerₕ(u, w)
    w .= 2.0 .* u .+ w
    A = Bramble.allocate_system_matrix(a)
    assemble!(A, a)
    assemble!(A, a)
    Bramble.semidiscretize_rhs(semidiscretize(a, l))(y, x, nothing, 0.0)
    return nothing
end
_pw_kron_call(a, x, y) = (mul!(y, kronecker_operator(a), x); nothing)
_pw_mf_call(a, x, y) = (mul!(y, matrix_free_operator(a), x); nothing)

if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        _pw_form(W) = form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        @compile_workload begin
            for (X, n) in (
                (interval(0.0, 1.0), 6),
                (interval(0.0, 1.0) × interval(0.0, 1.0), (5, 4)),
                (box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (4, 3, 3))
            )
                nu = map(_ -> false, n)
                W = gridspace(mesh(domain(X), n, nu))
                Wp = gridspace(mesh(domain(X), n, nu; backend = backend(policy = CpuPolyester())))
                x = ones(ndofs(W))
                y = similar(x)
                mul!(y, matrix_free_operator(_pw_form(W); policy = CpuPolyester()), x)
                mul!(y, matrix_free_operator(_pw_form(Wp)), x)
                op = matrix_free_operator(_pw_form(Wp); dirichlet = :boundary)
                mul!(y, op, x)
                mul!(y, op, x, 0.5, 2.0)
                # The other paths whose first call paid 100 ms or more (#434). Cheaper paths
                # stay out: each precompiled `@batch` signature adds about four binding-edge
                # invalidations.
                g(x) = sum(x)
                u = Rₕ(Wp, g)
                w = Rₕ(Wp, x -> x[1])
                a = _pw_form(Wp)
                l = form(Wp, v -> innerₕ(u, v))
                xp = ones(ndofs(Wp))
                yp = similar(xp)
                _pw_first_calls(u, w, g, a, l, xp, yp)
                _pw_mf_call(a, xp, yp)
                length(n) > 1 && _pw_kron_call(a, xp, yp)
            end
        end
    end
end

end # module BramblePolyesterExt
