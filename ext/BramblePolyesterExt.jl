# ext/BramblePolyesterExt.jl: the Polyester-batched sweeps behind `CpuPolyester`.
#
# Nine hooks in `src/` are `@noinline` methods that error naming Polyester
# (`src/utils/linear_algebra.jl`, `src/assembly/bilinear_execution.jl`, `src/assembly/linear.jl`):
# `_batch_for!`, `_batch_axis_for!`, `_batch_scatter_for!`, `_batch_dot`, `_batch_dot_masked`,
# `_batch_bilinear_colour_sweep!`, `_batch_bilinear_band_sweep!`, `_batch_linear_colour_sweep!`
# and `_batch_linear_band_sweep!`. Every one of them is the `Polyester.@batch` counterpart of
# an existing `Threads.@threads` body, called from the identical call site once `execution_policy`
# resolves to `CpuPolyester()` instead of `CpuThreaded()` -- so the colouring, the band splitting and,
# critically, the matrix/vector zeroing that happens in the *callers* (`_assemble_bilinear!`,
# `assemble_parallel!`, `_assemble_linear!`) are untouched by this file: this only supplies what
# runs once the caller has already zeroed and dispatched.
#
# Three more hooks have the same shape in
# `src/operators/difference.jl`/`src/operators/average.jl`: `_batch_difference_engine!`,
# `_batch_average_engine!` and `_batch_centered_average_engine!`, each the `@batch` counterpart of
# `_threaded_difference_engine!`/`_threaded_average_engine!`/`_threaded_centered_average_engine!`,
# running one `_difference_band!`/`_average_band!`/`_centered_average_band!` per band.
#
# A warmed refill replays recorded `nzval` positions instead of searching, gated per unit by
# `_threaded_replay_policy`; only `CpuThreaded` answers `true` in `src/`.
# This file's `_threaded_replay_policy(::CpuPolyester) = true` opts `CpuPolyester` in, and
# `_batch_bilinear_band_replay!`/`_batch_bilinear_colour_replay!` are the `@batch` counterparts
# of `_batch_bilinear_band_sweep!`/`_batch_bilinear_colour_sweep!` above, reached instead of
# them once a unit's leaf can replay (`_leaf_replays`, bilinear_execution.jl): the colouring is
# identical, only `_replay_point!` (reads the recording) stands in for `_scatter_point!`
# (searches).
#
# One more hook lives in `src/operators/vector_calculus.jl`:
# `_batch_run_bands!`, the `@batch` counterpart of `_run_bands!`'s `CpuThreaded` arm, reached
# by the divergence, curl and strain-average engines. Unlike the three hooks above, it stays
# generic over the band function `f` instead of naming one.
#
# `src/assembly/kronecker.jl` has `_batch_kron_lines!`, which runs a `KroneckerLinearOperator`
# product's grid lines under `@batch` when the operator's own policy is `CpuPolyester`.
#
# `src/problems/semidiscrete_rhs.jl` has `_batch_csr_spmv!`, which runs the explicit
# right-hand side's product `du .-= A u` one compressed row per `@batch` iteration.
#
# `Polyester.@batch` accepts a `CartesianIndices` directly (`closure.jl`'s own `splitloop`
# already splits it along its last axis, the same trick `_threaded_axis_for!` hand-rolls for
# `Threads.@threads`), so none of the manual axis-chunking `src/utils/linear_algebra.jl` uses
# for the threaded path is reproduced here. Axis-chunking helps `Threads.@threads` (it removes a linear-index conversion `Threads`
# cannot avoid on its own) but hurts `@batch`, which already does the equivalent split
# internally -- chunking on top would split twice.
#
# Allocation bound: a warm `CpuPolyester` call allocates a small constant amount per call,
# independent of the grid (the same on a 33² and a 513² grid). The cause is
# Polyester's argument box: `@batch` copies the arguments its loop captures into a
# heap-allocated `ManualMemory.Reference` on every call, sized by what the loop captures.
# Plain arrays become `PtrArray`s, which allocate nothing. Measured on a 2D non-uniform grid
# with 4 threads, every box is at most 512 B.
# Differences, shifts, averages, divergence, curl, `εₕ!`, `avgₕ!`, broadcasts, `innerₕ` and
# `inner₊` allocate 0 B, and so does a `KroneckerLinearOperator` product. Their loops capture
# only plain arrays and isbits values, rebuilding any struct around them inside each task
# (`_batch_kron_lines!`, `_batch_broadcast!`, `_batch_for!` below), so the box stays on the
# stack. Linear assembly allocates 256 B; a fused matrix-free product 112 B; a bilinear
# refill 1520 B (five colour sweeps of 304 B each); a per-unit matrix-free product 1088 B, a
# GMG V-cycle 1008 B per level, an explicit RHS 256 B. Those loops capture a form or another
# struct holding a GC reference, which puts the box on the heap; this release keeps the
# 512 B bound for them and gpena/Bramble.jl#437 follows up. An `avgₕ!` whose source closure
# captures an array, and a broadcast with a 0-dimensional array leaf, box likewise. The test
# file's `_PA_ZERO_PATHS` asserts 0 B for the first group. No `CpuPolyester` call reaches
# `Threads.@threads` or `Threads.@spawn` (gpena/Bramble.jl#400).
module BramblePolyesterExt

using Bramble
using Bramble: MarkedIndicesUnion, SeparableWeights, _throw_dot_dim_error,
               _write_components!, _band_range, _scatter_point!, _scatter_linear_point!,
               CpuPolyester, _ReplayTarget, _ActionTarget, _replay_point!, _difference_band!,
               _average_band!, _centered_average_band!, _broadcast_band!, _kron_line_init!,
               _kron_line_terms!, _kron_host_raw, _kron_host_rebuild, _bc_host_raw,
               _bc_host_rebuild, _AvgKernel, __prod
using Polyester: Polyester, @batch
using LinearAlgebra: mul!
using PrecompileTools: @setup_workload, @compile_workload

# --- _batch_for!/_batch_axis_for! (src/utils/linear_algebra.jl) -------------------- #
#
# Direct translations of `_threaded_for!`/`_threaded_axis_for!`'s bodies: `idxs` is whatever
# `_sweep_for!` was handed (a linear range for `_batch_for!`, a `CartesianIndices` for
# `_batch_axis_for!`), and `@batch` partitions it without any conversion of its own. 

#
# `v::AbstractArray` (rather than an unconstrained `v`) so each of these is a genuine
# specialisation of its `src/` stub, not a redefinition of the identical, fully unconstrained
# signature -- precompilation refuses that ("Method overwriting is not permitted"), the same
# reason `ext/BrambleSparseMatricesCSRExt.jl` bounds its own `_csr_backend` with `T <: Number`.
#
# `_batch_for!` hands `@batch` the kernel through `_for_host_raw`, so a kernel struct that
# holds arrays crosses as a tuple of them and each task rebuilds it (`_for_host_rebuild`),
# as `_batch_kron_lines!` below does for its factors. A struct captured whole holds GC
# references, which put Polyester's argument box on the heap (160 B per `avgₕ!` call).
# Only `avgₕ!`'s `_AvgKernel` has a raw form; any other kernel crosses as it is. So does an
# `_AvgKernel` whose source function holds a GC reference (a closure over an array): the box
# goes on the heap either way, and the raw tuple is the bigger one (16 B more per call).
struct _AvgRaw end

_for_host_raw(f) = f
function _for_host_raw(k::_AvgKernel{F}) where {F}
    isbitstype(F) || return k
    return (_AvgRaw(), k.f, k.x, k.idxs, k.nodes, k.wts)
end

@inline _for_host_rebuild(f) = f
@inline _for_host_rebuild(r::Tuple{_AvgRaw, Vararg}) = _AvgKernel(Base.tail(r)...)

function Bramble._batch_for!(v::AbstractArray, idxs, f)
    raw = _for_host_raw(f)
    @batch for idx in idxs
        k = _for_host_rebuild(raw)
        @inbounds v[idx] = k(idx)
    end
    return nothing
end

function Bramble._batch_axis_for!(v::AbstractArray, idxs::CartesianIndices, f)
    @batch for I in idxs
        @inbounds v[I] = f(I)
    end
    return nothing
end

# --- _batch_scatter_for! (src/utils/linear_algebra.jl) ------------------------------ #
#
# Mirrors `_threaded_scatter_for!`: `g` is evaluated once per index and its tuple of results
# is unpacked into every destination array in `mats` by `_write_components!` (unrolled at
# compile time, so this stays a single scalar write per component, not a tuple allocation).

# `idxs::AbstractRange` specialises this past the stub's fully unconstrained signature
# (`_batch_scatter_for!(mats::Tuple, idxs, g)`); the only caller (`project!`,
# operators/projection.jl) always passes `1:n`.
function Bramble._batch_scatter_for!(mats::Tuple, idxs::AbstractRange, g)
    @batch for idx in idxs
        @inbounds _write_components!(mats, g(idx), idx)
    end
    return nothing
end

# --- _batch_dot/_batch_dot_masked (src/utils/linear_algebra.jl) -------------------- #
#
# `CpuThreaded` now has its own threaded `_dot`/`_dot_masked` (linear_algebra.jl); this is the
# Polyester counterpart to that reduction.
# `@batch reduction=((+, s),)` keeps the running sum as a scalar the macro reduces itself
# (Polyester's own README: "does not incur any additional allocations"), rather than a
# per-task buffer this file would have to allocate and reduce by hand.

function Bramble._batch_dot(u::AbstractVector, v::AbstractVector, w::AbstractVector)
    (length(u) == length(v) == length(w)) ||
        _throw_dot_dim_error(length(u), length(v), length(w))
    T = promote_type(eltype(u), eltype(v), eltype(w))
    s = zero(T)
    n = length(u)
    @batch reduction=((+, s),) for i in 1:n
        @inbounds s += T(u[i]) * T(v[i]) * T(w[i])
    end
    return s
end

# A mask crosses `@batch` as a tuple of its 64-bit word vectors, never as the `BitVector` or
# `MarkedIndicesUnion` itself: a struct holding a GC reference puts Polyester's argument box
# on the heap, while each word vector becomes a `PtrArray` (`_mask_chunks`). A `BitVector`
# is the one-vector case.
#
# The `BitVector` mask (a single named region, or `dirichlet_bc!`'s own use) is then a plain
# `O(1)` bit test, so the batched sweep walks every index and skips the unset ones, rather
# than reproducing `MarkedIndices`' whole-word skip -- the whole-word skip only pays off
# walking sequentially, and `@batch` already divides the range across tasks itself.
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

function Bramble._batch_dot_masked(
        u::AbstractVector, v::AbstractVector, w::AbstractVector,
        mask::Union{BitVector, MarkedIndicesUnion}
)
    (length(u) == length(v) == length(w) == _mask_length(mask)) ||
        _throw_dot_dim_error(length(u), length(v), length(w), _mask_length(mask))
    T = promote_type(eltype(u), eltype(v), eltype(w))
    s = zero(T)
    n = length(u)
    chunks = _mask_chunks(mask)
    @batch reduction=((+, s),) for i in 1:n
        @inbounds if _mask_bit(chunks, i)
            s += T(u[i]) * T(v[i]) * T(w[i])
        end
    end
    return s
end

# `SeparableWeights` specializations. `inner₊(uₕ, vₕ, Val(S))`
# passes the weight as the *second* positional argument (space/inner_product.jl), so
# under `CpuPolyester` it is `_batch_dot`'s own second parameter, not third -- these dispatch on
# that position, mirroring the `CpuSerial`/`CpuThreaded` specializations in
# space/inner_product.jl. Same reasoning as those: the weight at `I` multiplies per-axis
# factors directly (`__prod`, what `w[I]` computes). `w[i]` (linear) divrems `i` back
# into a `CartesianIndex` first, an `O(n^D)` cost paid on every point, every batch task.
# The unmasked walk goes straight over `CartesianIndices(w.dims)` (which `@batch` also
# accepts, see this file's header comment); the masked walks stay over the flat `1:n` mask
# index space (masks are linear-indexed) and convert only the one index needed for `w`.
# The weight crosses `@batch` as its factor vectors and `dims`, not as the struct, for the
# reason the masks do above.
function Bramble._batch_dot(u::AbstractVector, w::SeparableWeights{D}, v::AbstractVector) where {D}
    n = length(w)
    (length(u) == n == length(v)) || _throw_dot_dim_error(length(u), n, length(v))
    T = promote_type(eltype(u), eltype(w), eltype(v))
    s = zero(T)
    factors = w.factors
    li = LinearIndices(w.dims)
    @batch reduction=((+, s),) for I in CartesianIndices(w.dims)
        @inbounds i = li[I]
        @inbounds s += T(u[i]) * T(v[i]) * T(__prod(factors, I))
    end
    return s
end

function Bramble._batch_dot_masked(
        u::AbstractVector, w::SeparableWeights{D}, v::AbstractVector,
        mask::Union{BitVector, MarkedIndicesUnion}
) where {D}
    n = length(w)
    (length(u) == n == length(v) == _mask_length(mask)) ||
        _throw_dot_dim_error(length(u), n, length(v), _mask_length(mask))
    T = promote_type(eltype(u), eltype(w), eltype(v))
    s = zero(T)
    factors = w.factors
    cart = CartesianIndices(w.dims)
    chunks = _mask_chunks(mask)
    @batch reduction=((+, s),) for i in 1:n
        @inbounds if _mask_bit(chunks, i)
            s += T(u[i]) * T(v[i]) * T(__prod(factors, cart[i]))
        end
    end
    return s
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

function Bramble._batch_bilinear_colour_sweep!(
        A::AbstractMatrix, sp, term, idxs, lin_indices, mesh_markers, row_offset, col_offset, α
)
    @batch for I in idxs
        _scatter_point!(A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset, α)
    end
    return nothing
end

function Bramble._batch_bilinear_band_sweep!(
        A::AbstractMatrix, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, row_offset, col_offset, α
)
    @batch for b in bidx
        for I in CartesianIndices((rest..., _band_range(ax, nbands, b)))
            _scatter_point!(
                A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset, α
            )
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
Bramble._threaded_replay_policy(::CpuPolyester) = true

function Bramble._batch_bilinear_band_replay!(
        target::Union{_ReplayTarget, _ActionTarget}, sp, term, ax, bidx, nbands, rest,
        lin_indices, mesh_markers, row_offset, col_offset
)
    @batch for b in bidx
        for I in CartesianIndices((rest..., _band_range(ax, nbands, b)))
            _replay_point!(target, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset)
        end
    end
    return nothing
end

function Bramble._batch_bilinear_colour_replay!(
        target::Union{_ReplayTarget, _ActionTarget}, sp, term, idxs, lin_indices, mesh_markers,
        row_offset, col_offset
)
    @batch for I in idxs
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

function Bramble._batch_linear_colour_sweep!(b::AbstractVector, sp, term, idxs, lin_indices, mesh_markers, offset, α)
    @batch for I in idxs
        _scatter_linear_point!(b, sp, term, I, lin_indices, mesh_markers, offset, α)
    end
    return nothing
end

function Bramble._batch_linear_band_sweep!(
        b::AbstractVector, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, offset, α
)
    @batch for k in bidx
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
# The calls of benchmark/polyester_first_call.jl, made from inside functions that take their
# arguments as ordinary values. Code in a user's function is inferred statically, so its
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
