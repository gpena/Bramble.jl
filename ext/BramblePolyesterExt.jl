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
# `Polyester.@batch` accepts a `CartesianIndices` directly (`closure.jl`'s own `splitloop`
# already splits it along its last axis, the same trick `_threaded_axis_for!` hand-rolls for
# `Threads.@threads`), so none of the manual axis-chunking `src/utils/linear_algebra.jl` uses
# for the threaded path is reproduced here. Axis-chunking helps `Threads.@threads` (it removes a linear-index conversion `Threads`
# cannot avoid on its own) but hurts `@batch`, which already does the equivalent split
# internally -- chunking on top would split twice.
module BramblePolyesterExt

using Bramble
using Bramble: MarkedIndicesUnion, SeparableWeights, _reduce_or_chunk, _throw_dot_dim_error,
               _write_components!, _band_range, _scatter_point!, _scatter_linear_point!,
               CpuPolyester, _ReplayTarget, _ActionTarget, _replay_point!, _difference_band!,
               _average_band!, _centered_average_band!, _broadcast_band!, _kron_line_init!,
               _kron_line_terms!
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

# `SeparableWeights` specializations. `inner₊(uₕ, vₕ, Val(S))`
# passes the weight as the *second* positional argument (space/inner_product.jl), so
# under `CpuPolyester` it is `_batch_dot`'s own second parameter, not third -- these dispatch on
# that position, mirroring the `CpuSerial`/`CpuThreaded` specializations in
# space/inner_product.jl. Same reasoning as those: `w[I]` (the `CartesianIndex`
# `getindex`) multiplies per-axis factors directly. `w[i]` (linear) divrems `i` back
# into a `CartesianIndex` first, an `O(n^D)` cost paid on every point, every batch task.
# The unmasked walk goes straight over `CartesianIndices(w.dims)` (which `@batch` also
# accepts, see this file's header comment); the masked walks stay over the flat `1:n` mask
# index space (masks are linear-indexed) and convert only the one index needed for `w`.
function Bramble._batch_dot(u::AbstractVector, w::SeparableWeights{D}, v::AbstractVector) where {D}
    n = length(w)
    (length(u) == n == length(v)) || _throw_dot_dim_error(length(u), n, length(v))
    T = promote_type(eltype(u), eltype(w), eltype(v))
    s = zero(T)
    li = LinearIndices(w.dims)
    @batch reduction=((+, s),) for I in CartesianIndices(w.dims)
        @inbounds i = li[I]
        @inbounds s += T(u[i]) * T(v[i]) * T(w[I])
    end
    return s
end

function Bramble._batch_dot_masked(
        u::AbstractVector, w::SeparableWeights{D}, v::AbstractVector, mask::BitVector
) where {D}
    n = length(w)
    (length(u) == n == length(v) == length(mask)) ||
        _throw_dot_dim_error(length(u), n, length(v), length(mask))
    T = promote_type(eltype(u), eltype(w), eltype(v))
    s = zero(T)
    cart = CartesianIndices(w.dims)
    @batch reduction=((+, s),) for i in 1:n
        @inbounds if mask[i]
            s += T(u[i]) * T(v[i]) * T(w[cart[i]])
        end
    end
    return s
end

function Bramble._batch_dot_masked(
        u::AbstractVector, w::SeparableWeights{D}, v::AbstractVector, mask::MarkedIndicesUnion
) where {D}
    n = length(w)
    (length(u) == n == length(v) == mask.len) ||
        _throw_dot_dim_error(length(u), n, length(v), mask.len)
    T = promote_type(eltype(u), eltype(w), eltype(v))
    s = zero(T)
    cart = CartesianIndices(w.dims)
    @batch reduction=((+, s),) for i in 1:n
        @inbounds if _mask_bit(mask, i)
            s += T(u[i]) * T(v[i]) * T(w[cart[i]])
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
# `bc` is boxed in a `Ref` before the loop rather than closed over directly: `@batch`
# gc-preserves every free variable through `StrideArraysCore.object_and_preserve`, which has
# a `Broadcast.Broadcasted`-specific method that rebuilds the tree through the 3-argument
# `Broadcasted(f, args, axes)` constructor whenever `bc.f` and `bc.axes` are both `isbits` --
# true of every broadcast this reaches (`bc.f` a plain function, `bc.axes` a tuple of
# `OneTo`s). That rebuild recomputes the style via `combine_styles` over the *unpacked*,
# already-`preprocess`ed args, including each `Broadcast.Extruded` leaf -- and `Extruded` has
# no `BroadcastStyle` of its own, so combining throws (`MethodError: no method matching
# ndims(::Type{Extruded{...}})`) before a single band ever runs, for any `bc` this reaches,
# not only a `VectorElement`-specific shape. A `Base.RefValue` wrapping `bc` has no such
# specialised `object_and_preserve` method, so it takes the plain, non-reconstructing
# fallback instead; `bcref[]` inside the loop hands `_broadcast_band!` the same `bc` either
# way.
function Bramble._batch_broadcast!(v::AbstractVector, bc, ax)
    n = Threads.nthreads()
    bcref = Ref(bc)
    @batch for b in 1:n
        _broadcast_band!(v, bcref[], ax, n, b)
    end
    return nothing
end

# --- _batch_kron_lines! (src/assembly/kronecker.jl) -------------------------------- #
#
# `CpuPolyester`'s host `KroneckerLinearOperator` product: one grid line along axis 1 per
# `@batch` iteration, each line `_kron_line_init!` then `_kron_line_terms!`, exactly the
# serial `_kron_fused!` loop.
#
# `y`, `x` and the terms cross `@batch` in one `Ref` box rather than being captured:
# `@batch` turns a captured array into a `PtrArray` and rebuilds every struct around one (a
# sparse factor, a diagonal) inside each task, and the line kernels then ran about half as
# fast (benchmark/batch_survey.jl, candidate 1). The box is fresh per call, so concurrent
# products on one operator share nothing. It costs a small heap allocation per call that does
# not grow with the grid: Polyester boxes its own argument tuple in a `ManualMemory.Reference`,
# which goes on the heap once the tuple holds a reference to a GC object, and the `Ref` box
# is one such object. The coefficients and `β` ride in the box too; the strides, the line
# indices and `m` are captured directly. `y::AbstractVector` and
# `lines::CartesianIndices` make this a genuine specialisation of the `src/` stub, the same
# reasoning as the hooks above.
function Bramble._batch_kron_lines!(
        y::AbstractVector, terms::Tuple, cs::Tuple, x::AbstractVector, β, ss::Tuple,
        lines::CartesianIndices, m::Int
)
    args = Ref((y, terms, cs, x, β))
    @batch for o in 1:length(lines)
        a = args[]
        off = (o - 1) * m
        _kron_line_init!(a[1], a[5], off, m)
        _kron_line_terms!(a[1], a[2], a[3], a[4], Tuple(lines[o]), ss, off, m)
    end
    return nothing
end

# The threaded matrix-free apply: `matrix_free_operator` under
# `CpuPolyester` in 1D-3D on non-uniform meshes, in the forms the tests and users reach --
# the policy passed to the operator or carried by the mesh backend, with and without
# `dirichlet = :boundary`, and the 3- and 5-argument `mul!`.
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
            end
        end
    end
end

end # module BramblePolyesterExt
