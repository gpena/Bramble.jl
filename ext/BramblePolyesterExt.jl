# ext/BramblePolyesterExt.jl: the Polyester-batched sweeps behind `CpuBatch` (S7.2,
# gpena/Bramble.jl#190, .agents/plans/v3-3-0-memory-scaling.md).
#
# S7.1 left nine hooks in `src/` as `@noinline` methods that error naming Polyester
# (`src/utils/linear_algebra.jl`, `src/form/bilinear_execution.jl`, `src/form/linear.jl`):
# `_batch_for!`, `_batch_axis_for!`, `_batch_scatter_for!`, `_batch_dot`, `_batch_dot_masked`,
# `_batch_bilinear_colour_sweep!`, `_batch_bilinear_band_sweep!`, `_batch_linear_colour_sweep!`
# and `_batch_linear_band_sweep!`. Every one of them is the `Polyester.@batch` counterpart of
# an existing `Threads.@threads` body, called from the identical call site once `execution_policy`
# resolves to `CpuBatch()` instead of `CpuThreaded()` -- so the colouring, the band splitting and,
# critically, the matrix/vector zeroing that happens in the *callers* (`_assemble_bilinear!`,
# `assemble_parallel!`, `_assemble_linear!`) are untouched by this file: this only supplies what
# runs once the caller has already zeroed and dispatched.
#
# `Polyester.@batch` accepts a `CartesianIndices` directly (`closure.jl`'s own `splitloop`
# already splits it along its last axis, the same trick `_threaded_axis_for!` hand-rolls for
# `Threads.@threads`), so none of the manual axis-chunking `src/utils/linear_algebra.jl` uses
# for the threaded path is reproduced here. gpena/Bramble.jl#190's own measurement is why:
# axis-chunking helps `Threads.@threads` (it removes a linear-index conversion `Threads`
# cannot avoid on its own) and hurts `@batch`, which already does the equivalent split
# internally -- chunking on top would split twice.
module BramblePolyesterExt

using Bramble
using Bramble: MarkedIndicesUnion, _reduce_or_chunk, _throw_dot_dim_error, _write_components!,
               _band_range, _scatter_point!, _scatter_linear_point!
using Polyester: Polyester, @batch

# --- _batch_for!/_batch_axis_for! (src/utils/linear_algebra.jl) -------------------- #
#
# Direct translations of `_threaded_for!`/`_threaded_axis_for!`'s bodies: `idxs` is whatever
# `_cpu_threaded_for!` was handed (a linear range for `_batch_for!`, a `CartesianIndices` for
# `_batch_axis_for!`), and `@batch` partitions it without any conversion of its own.

#
# `v::AbstractArray` (rather than an unconstrained `v`) so each of these is a genuine
# specialisation of its `src/` stub, not a redefinition of the identical, fully unconstrained
# signature -- precompilation refuses that ("Method overwriting is not permitted"), the same
# reason `ext/BrambleSparseMatricesCSRExt.jl` bounds its own `_csr_backend` with `T <: Number`.

function Bramble._batch_for!(v::AbstractArray, idxs, f)
    @batch for idx in idxs
        @inbounds v[idx] = f(idx)
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
# space/operators/projection.jl) always passes `1:n`.
function Bramble._batch_scatter_for!(mats::Tuple, idxs::AbstractRange, g)
    @batch for idx in idxs
        @inbounds _write_components!(mats, g(idx), idx)
    end
    return nothing
end

# --- _batch_dot/_batch_dot_masked (src/utils/linear_algebra.jl) -------------------- #
#
# `CpuThreaded`'s own `_dot`/`_dot_masked` (linear_algebra.jl) fall through to the plain
# serial reduction -- there never was a threaded reduction to match, only this Polyester one.
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

# The `BitVector` mask (a single named region, or `dirichlet_bc!`'s own use): `mask[i]` is a
# plain `O(1)` bit test, so the batched sweep walks every index and skips the unset ones,
# rather than reproducing `MarkedIndices`' whole-word skip -- the whole-word skip only pays
# off walking sequentially, and `@batch` already divides the range across tasks itself.
function Bramble._batch_dot_masked(u::AbstractVector, v::AbstractVector, w::AbstractVector, mask::BitVector)
    (length(u) == length(v) == length(w) == length(mask)) ||
        _throw_dot_dim_error(length(u), length(v), length(w), length(mask))
    T = promote_type(eltype(u), eltype(v), eltype(w))
    s = zero(T)
    n = length(u)
    @batch reduction=((+, s),) for i in 1:n
        @inbounds if mask[i]
            s += T(u[i]) * T(v[i]) * T(w[i])
        end
    end
    return s
end

# One-based bit test against a `MarkedIndicesUnion`'s own chunk storage (utils/linear_algebra.jl),
# reusing its `_reduce_or_chunk` (the same OR-of-chunks it walks with) rather than duplicating
# the chunk arithmetic a third time.
@inline function _mask_bit(mask::MarkedIndicesUnion, i::Int)
    chunk_idx = ((i - 1) >> 6) + 1
    bitpos = (i - 1) & 63
    word = _reduce_or_chunk(mask.chunks, chunk_idx)
    return ((word >> bitpos) & 0x1) != 0
end

# The multi-marker union mask (`innerₕ`/`inner₊*` restricted to two or more regions at once):
# no `getindex` exists on `MarkedIndicesUnion` (by design -- see its own docstring), so this
# reaches its chunk storage directly through `_mask_bit` instead of collecting it into an
# indexable mask first, which would allocate on every call.
function Bramble._batch_dot_masked(
        u::AbstractVector, v::AbstractVector, w::AbstractVector, mask::MarkedIndicesUnion
)
    (length(u) == length(v) == length(w) == mask.len) ||
        _throw_dot_dim_error(length(u), length(v), length(w), mask.len)
    T = promote_type(eltype(u), eltype(v), eltype(w))
    s = zero(T)
    n = length(u)
    @batch reduction=((+, s),) for i in 1:n
        @inbounds if _mask_bit(mask, i)
            s += T(u[i]) * T(v[i]) * T(w[i])
        end
    end
    return s
end

# --- _batch_bilinear_colour_sweep!/_batch_bilinear_band_sweep! (src/form/bilinear_execution.jl) --- #
#
# Direct translations of `_sweep_bilinear_colour!`/`_sweep_band_colour!`'s `CpuThreaded`
# bodies: each colour/band is independent by construction (the caller's colouring already
# guarantees no two concurrently-swept points target the same matrix entry), so swapping the
# scheduler changes nothing about correctness. `_scatter_point!` is the single shared entry
# rule both this and the `CpuThreaded` sweep call, so the two can never drift apart on what a
# stencil tap writes.

function Bramble._batch_bilinear_colour_sweep!(
        A, sp, term, idxs, lin_indices, mesh_markers, row_offset, col_offset, α
)
    @batch for I in idxs
        _scatter_point!(A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset, α)
    end
    return nothing
end

function Bramble._batch_bilinear_band_sweep!(
        A, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, row_offset, col_offset, α
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

# --- _batch_linear_colour_sweep!/_batch_linear_band_sweep! (src/form/linear.jl) ---- #
#
# As above, for the right-hand-side sweep: `_scatter_linear_point!` is the shared entry rule
# with `_sweep_colour!`/`_sweep_linear_band_colour!`'s `CpuThreaded` bodies.

function Bramble._batch_linear_colour_sweep!(b, sp, term, idxs, lin_indices, mesh_markers, offset, α)
    @batch for I in idxs
        _scatter_linear_point!(b, sp, term, I, lin_indices, mesh_markers, offset, α)
    end
    return nothing
end

function Bramble._batch_linear_band_sweep!(
        b, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, offset, α
)
    @batch for k in bidx
        for I in CartesianIndices((rest..., _band_range(ax, nbands, k)))
            _scatter_linear_point!(b, sp, term, I, lin_indices, mesh_markers, offset, α)
        end
    end
    return nothing
end

end # module BramblePolyesterExt
