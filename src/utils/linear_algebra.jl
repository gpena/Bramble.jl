@noinline function _throw_dot_dim_error(lu::Integer, lv::Integer, lw::Integer)
    throw(
        DimensionMismatch(
        "Vectors must have matching lengths, but got lengths ($lu, $lv, $lw)."
    ),
    )
end

@noinline function _throw_dot_dim_error(lu::Integer, lv::Integer, lw::Integer, lm::Integer)
    throw(
        DimensionMismatch(
        "Vectors and mask must have matching lengths, but got lengths ($lu, $lv, $lw, $lm).",
    ),
    )
end

"""
    _sweep_for!(loc::Locality, policy::ExecutionPolicy, v::AbstractArray, idxs, f::Function) -> Nothing
    _sweep_for!(policy::ExecutionPolicy, v::AbstractArray, idxs, f::Function) -> Nothing

Apply `f` across indices `idxs` and write the result into `v` in place.

Legality first, strategy second (gpena/Bramble.jl#298): a [`HostLocality`](@ref) destination
paired with a [`CpuPolicy`](@ref) dispatches to sequential iteration for [`CpuSerial`](@ref)
or static work partitioning across threads for [`CpuThreaded`](@ref); a
[`DeviceLocality`](@ref) destination paired with a [`GpuPolicy`](@ref) reaches
[`_gpu_for!`](@ref). Any other pairing -- a host `v` under a `GpuPolicy`, or a device `v`
under a `CpuPolicy` -- throws via [`_throw_locality_mismatch`](@ref) rather than indexing `v`
element by element or reaching for a device sweep that does not exist.

The four-argument method is the seam; the three-argument method is the thin wrapper every
caller actually uses, deriving `loc` from the destination array itself
(`locality(typeof(v))`) so callers pass a policy alone and never compute a locality.

# Arguments
- `loc`: [`Locality`](@ref) of the destination array (derived, not passed by callers).
- `policy`: Execution policy ([`CpuSerial`](@ref), [`CpuThreaded`](@ref), [`CpuBatch`](@ref)
  or a [`GpuPolicy`](@ref)).
- `v`: Destination array mutated in place.
- `idxs`: Iterable collection of linear or Cartesian indices.
- `f`: Kernel mapping each index `idx` to the scalar value stored in `v[idx]`.
"""
@inline _sweep_for!(::HostLocality, ::CpuSerial, v, idxs, f) = _serial_for!(v, idxs, f)
@inline _sweep_for!(::HostLocality, ::CpuThreaded, v, idxs, f) = _threaded_for!(v, idxs, f)
@noinline _sweep_for!(::HostLocality, ::CpuBatch, v, idxs, f) = _batch_for!(v, idxs, f)
@noinline _sweep_for!(::DeviceLocality, policy::GpuPolicy, v, idxs, f) = _gpu_for!(policy, v, idxs, f)
@noinline _sweep_for!(loc::Locality, policy, v, idxs, f) = _throw_locality_mismatch(loc, policy)

@inline _sweep_for!(policy::ExecutionPolicy, v, idxs, f) = _sweep_for!(locality(typeof(v)), policy, v, idxs, f)

# `Threads.@threads` needs an indexable collection, so handed a `CartesianIndices` it
# linearly indexes it and pays an index conversion per point, where the serial loop
# iterates the block natively. That made the threaded weight build *slower* than the
# serial one it replaces: `__innerplus_weights!` at 1e6 points measured 0.117 ms serial
# against 0.365 ms on four threads in 3D (0.200 ms in 2D). Splitting the last axis hands
# every thread a `CartesianIndices` block of its own, iterated exactly as the serial loop
# iterates the whole, which brings it back to 0.125 ms.
#
# How much the threading then buys depends on how close one core gets to the machine's
# store ceiling, which is ~68-73 GB/s here: on AC power a single core already reaches it
# (four threads matching rather than beating it, from 8 MB to 488 MB), while on battery
# one core manages 46-55 GB/s and four recover 1.24-1.39x. The point of this method is the
# removed index conversion, which is a penalty in every power state; the parallel gain on
# top of it is the machine's to give.
@inline _sweep_for!(::HostLocality, ::CpuThreaded, v, idxs::CartesianIndices, f) = _threaded_axis_for!(v, idxs, f)
@noinline _sweep_for!(::HostLocality, ::CpuBatch, v, idxs::CartesianIndices, f) = _batch_axis_for!(v, idxs, f)

"""
    _throw_locality_mismatch(loc::Locality, policy)

Throw the `ArgumentError` [`_sweep_for!`](@ref)/[`_sweep_scatter_for!`](@ref) give when the
destination's locality and the execution policy's locality disagree, in either direction
(gpena/Bramble.jl#191, #298).

Stated here rather than left to a `MethodError`, which would name `_sweep_for!` and not say
which half disagreed. Dispatched on the destination's own locality so each direction names
what actually disagreed -- the destination's locality and the policy's -- rather than
assuming a [`GpuPolicy`](@ref) is always the one out of place. Neither method advises
building the backend with a [`CpuPolicy`](@ref) as a fix for device storage: that
combination is itself rejected at construction (gpena/Bramble.jl#296), and is not a route
this message may point to.

# Arguments
- `loc`: [`Locality`](@ref) of the destination array.
- `policy`: The execution policy whose locality disagrees with `loc`.

# Throws
- `ArgumentError`: always.
"""
@noinline function _throw_locality_mismatch(::HostLocality, policy)
    throw(
        ArgumentError(
        "the destination array has host locality, but execution policy $(typeof(policy)) " *
        "claims device locality: a sweep cannot mix the two. Reach for a kernel that runs " *
        "on the device, or pass a policy whose locality matches the destination -- " *
        "CpuSerial(), CpuThreaded() or CpuBatch().",
    ),
    )
end
@noinline function _throw_locality_mismatch(::DeviceLocality, policy)
    throw(
        ArgumentError(
        "the destination array has device locality, but execution policy $(typeof(policy)) " *
        "claims host locality: a sweep cannot mix the two. A host loop cannot scalar-index " *
        "device memory; pass a GpuPolicy (GpuAsync()) instead, with the device extension " *
        "that recognises the destination's array type loaded.",
    ),
    )
end

"""
    _throw_gpu_in_cpu_loop(policy)

Throw the `ArgumentError` [`_gpu_for!`](@ref)/[`_gpu_scatter_for!`](@ref) give when no device
sweep implementation is available at all -- a case [`_throw_locality_mismatch`](@ref) never
reaches, since it is only called once locality has already been checked to agree.

# Arguments
- `policy`: The [`GpuPolicy`](@ref) with no device sweep loaded for it.

# Throws
- `ArgumentError`: always.
"""
@noinline function _throw_gpu_in_cpu_loop(policy)
    throw(
        ArgumentError(
        "execution policy $(typeof(policy)) is a GpuPolicy, but no device sweep is loaded " *
        "for it. Add `using KernelAbstractions` and the package providing the destination " *
        "array's device (e.g. `using Metal`) before calling this.",
    ),
    )
end

#===========================================================================#
# The GpuPolicy device sweep seam (gpena/Bramble.jl#94, #174, #298, S2.3 of
# .agents/plans/metal-and-apple-silicon-acceleration.md).
#
# `_sweep_for!`/`_sweep_scatter_for!` hand a `(DeviceLocality, GpuPolicy)` pair to
# `_gpu_for!`/`_gpu_scatter_for!` rather than refusing it outright. Both are declared here
# with their array/index/kernel arguments deliberately untyped -- matching the `ka_device`/
# `_launch_uniform_points!` fallback idiom (`src/utils/device_kernels.jl`,
# `src/mesh/mesh1d.jl`) -- so `ext/BrambleKernelAbstractionsExt.jl` can add a strictly more
# specific method (typed on `AbstractArray`) instead of overwriting this one. Without that
# extension loaded, both fall straight through to the same `_throw_gpu_in_cpu_loop` message
# CPU callers have always seen, so the diagnostic never regresses into a bare `MethodError`.
#===========================================================================#

"""
    _gpu_for!(policy::GpuPolicy, v::AbstractArray, idxs, f::Function) -> Nothing

The [`GpuPolicy`](@ref) counterpart of [`_serial_for!`](@ref)/[`_threaded_for!`](@ref):
fills `v[idx]` with `f(idx)` across `idxs` with a `KernelAbstractions.@kernel` launch on
`v`'s own device, filled by `ext/BrambleKernelAbstractionsExt.jl`.

`f` runs on the device, so it must be GPU-compilable: no heap allocation, no boxed
captures (a `Ref`, a `mutable struct` field, a global), and every function it calls must
itself compile for the device. A closure over a plain number or over another device array
works; one that allocates or calls a non-inlineable function fails at kernel-compile time
with a `GPUCompiler.InvalidIRError`, not a Bramble-specific message -- this is inherent to
GPU execution.

# Throws
- `ArgumentError`: no `KernelAbstractions` extension is loaded, so there is no device sweep
  to reach ([`_throw_gpu_in_cpu_loop`](@ref)).
"""
@noinline _gpu_for!(policy, v, idxs, f) = _throw_gpu_in_cpu_loop(policy)

"""
    _gpu_scatter_for!(policy::GpuPolicy, mats::Tuple, idxs, g::Function) -> Nothing

The [`GpuPolicy`](@ref) counterpart of [`_threaded_scatter_for!`](@ref): evaluates
tuple-valued `g` across `idxs` and scatters the results into `mats` with a
`KernelAbstractions.@kernel` launch, filled by `ext/BrambleKernelAbstractionsExt.jl`. Same
GPU-compilability requirement on `g` as [`_gpu_for!`](@ref).

# Throws
- `ArgumentError`: no `KernelAbstractions` extension is loaded ([`_throw_gpu_in_cpu_loop`](@ref)).
"""
@noinline _gpu_scatter_for!(policy, mats, idxs, g) = _throw_gpu_in_cpu_loop(policy)

# The message every `CpuBatch` hook gives without `Polyester` loaded (gpena/Bramble.jl#190).
# `CpuBatch`'s own sweeps have no `src/` implementation -- `BramblePolyesterExt` (S7.2) adds
# it -- so a `CpuBatch` backend used without that extension loaded stops here, named, rather
# than silently falling through to `CpuThreaded`'s `Threads.@threads` code (which would
# defeat the whole point of choosing `CpuBatch`) or a bare `MethodError`. Mirrors
# `_throw_gpu_in_cpu_loop` above and `_metal_backend` (`src/utils/backend.jl`).
@noinline function _throw_cpubatch_without_polyester(fname::Symbol)
    throw(
        ArgumentError(
        "CpuBatch requires Polyester.jl. Add `using Polyester` before calling $(fname) " *
        "under a CpuBatch backend.",
    ),
    )
end

"""
    _batch_for!(v::AbstractArray, idxs, f::Function) -> Nothing

[`CpuBatch`](@ref)'s counterpart of [`_threaded_for!`](@ref), filled by
`BramblePolyesterExt` (gpena/Bramble.jl#190). The only `src/` method errors naming
Polyester, the way `_metal_backend` errors naming Metal.
"""
@noinline function _batch_for!(v, idxs, f)
    return _throw_cpubatch_without_polyester(:_batch_for!)
end

"""
    _batch_axis_for!(v::AbstractArray, idxs::CartesianIndices, f::Function) -> Nothing

[`CpuBatch`](@ref)'s counterpart of [`_threaded_axis_for!`](@ref), filled by
`BramblePolyesterExt`. The only `src/` method errors naming Polyester.
"""
@noinline function _batch_axis_for!(v, idxs::CartesianIndices, f)
    return _throw_cpubatch_without_polyester(:_batch_axis_for!)
end

"""
    _effective_parallel_policy(sp) -> CpuPolicy

The policy a forced-threaded sweep (`assemble_parallel!`, or the non-serial branch of
`assemble!`/`assemble_add!`) actually runs under: [`CpuSerial`](@ref) is coerced to
[`CpuThreaded`](@ref) -- a threaded sweep threads even from a serially configured backend,
which is the entire point of forcing it -- while [`CpuThreaded`](@ref) and [`CpuBatch`](@ref)
pass through unchanged, so a `CpuBatch` backend still runs its own (Polyester) sweep, or
errors clearly without it, rather than silently substituting `Threads.@threads`.
"""
@inline _effective_parallel_policy(sp) = _coerce_serial_to_threaded(execution_policy(sp))
@inline _coerce_serial_to_threaded(::CpuSerial) = CpuThreaded()
@inline _coerce_serial_to_threaded(policy::CpuPolicy) = policy

"""
    _threaded_for!(v::AbstractArray, idxs, f::Function) -> Nothing

Fill `v[idx]` with `f(idx)` across threads, statically partitioning `idxs`.

Kept in an isolated function to prevent `Threads.@threads` closure boxing allocations on
paths that execute serially.
"""
@noinline function _threaded_for!(v, idxs, f)
    # Static partitioning distributes work evenly across available threads
    Threads.@threads :static for idx in idxs
        @inbounds v[idx] = f(idx)
    end
    return nothing
end

"""
    _serial_for!(v::AbstractArray, idxs, f::Function) -> Nothing

Iterate sequentially over `idxs`, writing `v[idx] = f(idx)` in place.

# Arguments
- `v`: Destination array mutated in place.
- `idxs`: Iterable collection of indices.
- `f`: Kernel evaluating values at each index.
"""
@inline function _serial_for!(v, idxs, f)
    @inbounds for idx in idxs
        v[idx] = f(idx)
    end
    return nothing
end

"""
    _band_range(ax::AbstractRange, nbands::Int, b::Int) -> AbstractRange

The `b`-th of `nbands` contiguous slabs of `ax`.

Slabs differ in length by at most one, the remainder spread over the first of them rather
than left on the last. `b` indexes positions within `ax`, not values, so an axis carrying a
stride keeps it.
"""
@inline function _band_range(ax::AbstractRange, nbands::Int, b::Int)
    q, r = divrem(length(ax), nbands)
    lo = (b - 1) * q + min(b - 1, r) + 1
    hi = lo + q - 1 + (b <= r ? 1 : 0)
    return @inbounds ax[lo:hi]
end

"""
    _band_count(len::Int, span::Int, nthreads::Int) -> Int

How many slabs to cut an axis of `len` points into for a stencil reaching `span` along it.

Enough for every thread to hold one slab per colour, never so many that a slab falls below
`span` (which is what keeps alternate slabs free of each other's stencil footprints), and
always even so the two colours are balanced. Returns `0` when the axis is too short to band
at all, leaving the caller on its point-coloured path.
"""
@inline function _band_count(len::Int, span::Int, nthreads::Int)
    span < 1 && return 0
    nbands = min(2 * nthreads, div(len, span))
    isodd(nbands) && (nbands -= 1)
    return nbands < 2 ? 0 : nbands
end

"""
    _LastAxisChunks(rest::Tuple, ax::AbstractRange, n::Int)

Splits a `CartesianIndices` into `n` blocks along its last axis, each block itself a
`CartesianIndices` over the leading axes `rest`. Indexable and lazy, so the split allocates
nothing and `Threads.@threads` can partition it directly.

Blocks differ in length by at most one slice: the remainder is spread over the first of
them rather than left on the last, so no thread receives a double-sized tail.
"""
struct _LastAxisChunks{D, R <: Tuple, A <: AbstractRange}
    rest::R
    ax::A
    n::Int
end

"""
    _last_axis_chunks(idxs::CartesianIndices{D}, n::Integer) -> _LastAxisChunks{D}

Return `idxs` split into at most `n` blocks along its last axis, clamped to the length of
that axis so no block is empty.
"""
@inline function _last_axis_chunks(idxs::CartesianIndices{D}, n::Integer) where {D}
    inds = idxs.indices
    ax = inds[D]
    nblocks = max(1, min(Int(n), length(ax)))
    rest = Base.front(inds)
    return _LastAxisChunks{D, typeof(rest), typeof(ax)}(rest, ax, nblocks)
end

@inline Base.length(c::_LastAxisChunks) = c.n
@inline Base.firstindex(::_LastAxisChunks) = 1
@inline Base.lastindex(c::_LastAxisChunks) = c.n

# `lo:hi` are positions *within* the axis rather than values, so an axis that carries a
# stride keeps it.
@inline function Base.getindex(c::_LastAxisChunks{D}, k::Int) where {D}
    q, r = divrem(length(c.ax), c.n)
    lo = (k - 1) * q + min(k - 1, r) + 1
    hi = lo + q - 1 + (k <= r ? 1 : 0)
    return CartesianIndices((c.rest..., @inbounds c.ax[lo:hi]))
end

"""
    _threaded_axis_for!(v::AbstractArray, idxs::CartesianIndices, f::Function) -> Nothing

As [`_threaded_for!`](@ref), for a `CartesianIndices`: each thread takes one block of
whole last-axis slices and walks it natively, never converting a linear index.
"""
@noinline function _threaded_axis_for!(v, idxs::CartesianIndices, f)
    blocks = _last_axis_chunks(idxs, Threads.nthreads())
    Threads.@threads :static for k in 1:length(blocks)
        block = blocks[k]
        @inbounds for I in block
            v[I] = f(I)
        end
    end
    return nothing
end

# Scatters tuple-valued kernel outputs into separate component arrays in a single pass,
# evaluating multi-component evaluations once per index instead of per component.

"""
    _write_components!(mats::Tuple, vals::Tuple, idx) -> Nothing

Recursively unpack and write elements of `vals` into destination arrays `mats` at index `idx`.

Recursion on tuples unrolls at compile time with zero heap allocations.
"""
@inline _write_components!(::Tuple{}, ::Tuple, idx) = nothing
@inline function _write_components!(mats::Tuple, vals::Tuple, idx)
    @inbounds mats[1][idx] = vals[1]
    return _write_components!(Base.tail(mats), Base.tail(vals), idx)
end

"""
    _sweep_scatter_for!(loc::Locality, policy::ExecutionPolicy, mats::Tuple, idxs, g::Function) -> Nothing
    _sweep_scatter_for!(policy::ExecutionPolicy, mats::Tuple, idxs, g::Function) -> Nothing

Evaluate tuple-valued kernel `g` across `idxs` and scatter results into destination arrays `mats`.

Legality first, strategy second (gpena/Bramble.jl#298), the same shape as [`_sweep_for!`](@ref):
a [`HostLocality`](@ref) destination paired with a [`CpuPolicy`](@ref) dispatches to sequential
execution for [`CpuSerial`](@ref) or static multithreaded execution for [`CpuThreaded`](@ref);
a [`DeviceLocality`](@ref) destination paired with a [`GpuPolicy`](@ref) reaches
[`_gpu_scatter_for!`](@ref). Any other pairing throws via [`_throw_locality_mismatch`](@ref).

The five-argument method is the seam; the four-argument method is the thin wrapper every
caller actually uses, deriving `loc` from the first destination array (`locality(typeof(mats[1]))`)
so callers pass a policy alone and never compute a locality.

# Arguments
- `loc`: [`Locality`](@ref) of the destination arrays (derived, not passed by callers).
- `policy`: Execution policy ([`CpuSerial`](@ref), [`CpuThreaded`](@ref), [`CpuBatch`](@ref)
  or a [`GpuPolicy`](@ref)).
- `mats`: Tuple of destination arrays mutated in place.
- `idxs`: Iterable collection of indices.
- `g`: Kernel mapping each index to a tuple of values matching `length(mats)`.
"""
@inline function _sweep_scatter_for!(::HostLocality, ::CpuSerial, mats::Tuple, idxs, g)
    @inbounds for idx in idxs
        _write_components!(mats, g(idx), idx)
    end
    return nothing
end
@inline _sweep_scatter_for!(::HostLocality, ::CpuThreaded, mats::Tuple, idxs, g) = _threaded_scatter_for!(mats, idxs, g)
@noinline _sweep_scatter_for!(::HostLocality, ::CpuBatch, mats::Tuple, idxs, g) = _batch_scatter_for!(mats, idxs, g)
@noinline _sweep_scatter_for!(::DeviceLocality, policy::GpuPolicy, mats::Tuple, idxs, g) = _gpu_scatter_for!(policy, mats, idxs, g)
@noinline _sweep_scatter_for!(loc::Locality, policy, mats::Tuple, idxs, g) = _throw_locality_mismatch(loc, policy)

@inline _sweep_scatter_for!(policy::ExecutionPolicy, mats::Tuple, idxs, g) = _sweep_scatter_for!(
    locality(typeof(mats[1])), policy, mats, idxs, g)

"""
    _threaded_scatter_for!(mats::Tuple, idxs, g::Function) -> Nothing

Evaluate tuple-valued `g` across `idxs` and scatter the results into `mats`, statically
partitioning `idxs` across threads.

Kept in an isolated function to prevent `Threads.@threads` closure boxing allocations on
paths that execute serially.
"""
@noinline function _threaded_scatter_for!(mats::Tuple, idxs, g)
    Threads.@threads :static for idx in idxs
        @inbounds _write_components!(mats, g(idx), idx)
    end
    return nothing
end

"""
    _batch_scatter_for!(mats::Tuple, idxs, g::Function) -> Nothing

[`CpuBatch`](@ref)'s counterpart of `_threaded_scatter_for!`, filled by
`BramblePolyesterExt`. The only `src/` method errors naming Polyester.
"""
@noinline function _batch_scatter_for!(mats::Tuple, idxs, g)
    return _throw_cpubatch_without_polyester(:_batch_scatter_for!)
end

#===========================================================================#
# Walking a boundary mask
#
# The general-purpose chunk-skipping walk over a `BitVector`'s set bits. Lives here, not
# in `form/dirichlet_constraints.jl` where it used to be written out by hand a second time
# (as `_each_marked`) and a third (inside `_dot_masked` below, twice): a bit-walk is a
# linear-algebra utility, with Dirichlet boundary conditions one caller among several
# (gpena/Bramble.jl#71).
#===========================================================================#

"""
    MarkedIndices(mask::BitVector, offset::Int = 0)

Lazily iterates the 1-based positions where `mask` is set, each shifted by `offset` (so a
leaf's mask, consulted at its offset into a global vector, yields global indices without
copying). Walks whole 64-bit words at a time, skipping zero chunks entirely and extracting
set bits via `trailing_zeros`, so the work is proportional to the number of set bits, not
to `length(mask)`.

No bounds guard against `length(mask)` is needed: `BitVector` guarantees the padding bits
of its final chunk are zero, so the walk never yields an index past the mask's own length.
"""
struct MarkedIndices
    chunks::Vector{UInt64}
    offset::Int
end

@inline MarkedIndices(mask::BitVector, offset::Int = 0) = MarkedIndices(mask.chunks, offset)

@inline function Base.iterate(m::MarkedIndices, (chunk_idx, rest) = (0, zero(UInt64)))
    chunks = m.chunks
    @inbounds while rest == zero(UInt64)
        chunk_idx += 1
        chunk_idx > length(chunks) && return nothing
        rest = chunks[chunk_idx]
    end
    i = m.offset + (chunk_idx - 1) * 64 + trailing_zeros(rest) + 1
    return i, (chunk_idx, rest & (rest - 1))
end

Base.IteratorSize(::Type{MarkedIndices}) = Base.SizeUnknown()
Base.eltype(::Type{MarkedIndices}) = Int

"""
    MarkedIndicesUnion(masks::NTuple{N,BitVector}) where N

Lazily iterates the 1-based positions where the union of `masks` is set.

The multi-marker counterpart of [`MarkedIndices`](@ref): `_combined_mask` (space/inner_product.jl)
used to materialize the union into a fresh `BitVector` via `copy` + `.|=` before walking it,
one heap allocation per call (gpena/Bramble.jl#149). This ORs each mask's 64-bit chunk on the
fly instead, so the union is never materialized -- still proportional to the number of set
bits, still a whole-word skip wherever every mask's chunk is zero, and allocates nothing.
"""
struct MarkedIndicesUnion{N}
    chunks::NTuple{N, Vector{UInt64}}
    len::Int
end

@inline function MarkedIndicesUnion(masks::NTuple{N, BitVector}) where {N}
    return MarkedIndicesUnion{N}(map(m -> m.chunks, masks), length(masks[1]))
end

@inline _reduce_or_chunk(chunks::NTuple{N, Vector{UInt64}}, i::Int) where {N} = reduce(|, ntuple(k -> chunks[k][i], Val(N)))

@inline function Base.iterate(
        m::MarkedIndicesUnion{N}, (chunk_idx, rest) = (0, zero(UInt64))
) where {N}
    nchunks = length(m.chunks[1])
    @inbounds while rest == zero(UInt64)
        chunk_idx += 1
        chunk_idx > nchunks && return nothing
        rest = _reduce_or_chunk(m.chunks, chunk_idx)
    end
    i = (chunk_idx - 1) * 64 + trailing_zeros(rest) + 1
    return i, (chunk_idx, rest & (rest - 1))
end

Base.IteratorSize(::Type{<:MarkedIndicesUnion}) = Base.SizeUnknown()
Base.eltype(::Type{<:MarkedIndicesUnion}) = Int

# Discrete space inner product kernels

"""
    _dot(u::AbstractVector, v::AbstractVector, w::AbstractVector) -> Real

Compute the weighted trilinear dot product
```math
\\sum_{i=1}^n u_i v_i w_i
```

Accumulates via fused multiply-add operations (`muladd`) with `@simd` vectorization.

A same-eltype specialization used to sit alongside this one, skipping the `promote_type`
call and the `T(...)` conversions on the (dispatch-favoured) assumption that they cost
something. Compared by `@code_llvm`/`@code_native` with matching element types
(gpena/Bramble.jl#71): identical generated code, since `promote_type(T, T, T) === T` and
`T(x::T)` is an identity conversion the compiler elides. One method now covers both cases.

# Arguments
- `u`: First vector.
- `v`: Second vector.
- `w`: Weight vector.

# Throws
- `DimensionMismatch`: If `length(u)`, `length(v)`, and `length(w)` do not match.
"""
@inline function _dot(u::AbstractVector, v::AbstractVector, w::AbstractVector)
    (length(u) == length(v) == length(w)) ||
        _throw_dot_dim_error(length(u), length(v), length(w))
    T = promote_type(eltype(u), eltype(v), eltype(w))
    s = zero(T)

    @inbounds @simd for i in 1:length(u)
        s = muladd(T(u[i]) * T(v[i]), T(w[i]), s)
    end

    return s
end

"""
    _dot_masked(u::AbstractVector, v::AbstractVector, w::AbstractVector, mask::BitVector) -> Real

Compute the weighted dot product restricted to indices where `mask` is true:
```math
\\sum_{i \\in \\mathrm{supp}(\\mathrm{mask})} u_i v_i w_i
```

Walks `MarkedIndices(mask)`, so the work is proportional to the number of set bits
rather than to `length(mask)`.

As with `_dot`, a same-eltype specialization used to sit alongside this one;
`@code_llvm`/`@code_native` with matching element types (gpena/Bramble.jl#71) showed
identical generated code, so one method now covers both cases.

# Arguments
- `u`: First vector.
- `v`: Second vector.
- `w`: Weight vector.
- `mask`: Boolean selection mask.

# Throws
- `DimensionMismatch`: If vector or mask lengths do not match.
"""
@inline function _dot_masked(
        u::AbstractVector, v::AbstractVector, w::AbstractVector, mask::BitVector
)
    (length(u) == length(v) == length(w) == length(mask)) ||
        _throw_dot_dim_error(length(u), length(v), length(w), length(mask))
    T = promote_type(eltype(u), eltype(v), eltype(w))
    s = zero(T)

    @inbounds for i in MarkedIndices(mask)
        s = muladd(T(u[i]) * T(v[i]), T(w[i]), s)
    end

    return s
end

# The multi-marker counterpart, walking a `MarkedIndicesUnion` (gpena/Bramble.jl#149)
# instead of a `BitVector`: `_combined_mask` hands one of these straight in, with no
# intermediate combined mask to check the length of, so the guard reads `mask.len`.
@inline function _dot_masked(
        u::AbstractVector, v::AbstractVector, w::AbstractVector, mask::MarkedIndicesUnion
)
    (length(u) == length(v) == length(w) == mask.len) ||
        _throw_dot_dim_error(length(u), length(v), length(w), mask.len)
    T = promote_type(eltype(u), eltype(v), eltype(w))
    s = zero(T)

    @inbounds for i in mask
        s = muladd(T(u[i]) * T(v[i]), T(w[i]), s)
    end

    return s
end

# --- Execution-policy-dispatched reduction entries (gpena/Bramble.jl#190, S7.1) ---------- #
#
# `inner₊` (`src/space/inner_product.jl`) does pass a policy into `_dot`/`_dot_masked`.
# `CpuSerial`/`CpuThreaded` both fall through to today's single (already vectorised)
# implementation -- there is no separate threaded reduction to pick between, only a
# `CpuBatch` one, which the `_batch_dot`/`_batch_dot_masked` hooks below supply once
# `BramblePolyesterExt` is loaded, and a `GpuPolicy` one, below the CpuBatch hooks.

"""
    _dot(policy::ExecutionPolicy, u, v, w) -> Real

Policy-dispatched [`_dot`](@ref): [`CpuSerial`](@ref) and [`CpuThreaded`](@ref) fall
through to the plain three-vector method; [`CpuBatch`](@ref) reaches [`_batch_dot`](@ref).
"""
@inline _dot(::CpuSerial, u, v, w) = _dot(u, v, w)
@inline _dot(::CpuThreaded, u, v, w) = _dot(u, v, w)
@noinline _dot(::CpuBatch, u, v, w) = _batch_dot(u, v, w)

"""
    _dot_masked(policy::ExecutionPolicy, u, v, w, mask) -> Real

Policy-dispatched [`_dot_masked`](@ref): [`CpuSerial`](@ref) and [`CpuThreaded`](@ref) fall
through to the plain masked method; [`CpuBatch`](@ref) reaches [`_batch_dot_masked`](@ref).
"""
@inline _dot_masked(::CpuSerial, u, v, w, mask) = _dot_masked(u, v, w, mask)
@inline _dot_masked(::CpuThreaded, u, v, w, mask) = _dot_masked(u, v, w, mask)
@noinline _dot_masked(::CpuBatch, u, v, w, mask) = _batch_dot_masked(u, v, w, mask)

"""
    _batch_dot(u, v, w) -> Real

[`CpuBatch`](@ref)'s counterpart of [`_dot`](@ref), filled by `BramblePolyesterExt`. The
only `src/` method errors naming Polyester.
"""
@noinline function _batch_dot(u, v, w)
    return _throw_cpubatch_without_polyester(:_batch_dot)
end

"""
    _batch_dot_masked(u, v, w, mask) -> Real

[`CpuBatch`](@ref)'s counterpart of [`_dot_masked`](@ref), filled by
`BramblePolyesterExt`. The only `src/` method errors naming Polyester.
"""
@noinline function _batch_dot_masked(u, v, w, mask)
    return _throw_cpubatch_without_polyester(:_batch_dot_masked)
end

#===========================================================================#
# Device reductions (gpena/Bramble.jl#94, #174, S2.5 of
# .agents/plans/metal-and-apple-silicon-acceleration.md).
#
# `CpuSerial`/`CpuThreaded`/`CpuBatch` above are each a concrete `CpuPolicy`, so a
# `GpuPolicy` (`GpuAsync`) never matches one of them and falls through to the generic
# `ExecutionPolicy` method below instead. That method derives locality from the policy
# itself -- `locality(::GpuPolicy) = DeviceLocality()` (`src/utils/backend.jl`) -- the same
# "legality first, strategy second" shape `_sweep_for!` uses, so the actual device
# implementation keys on `DeviceLocality`, never on `GpuPolicy` alone: a call site that
# later starts passing a `Locality` directly (as `_device_project!` now does, gpena/Bramble.jl#298)
# would still reach it.
#
# No `@kernel` is needed for either reduction: `GPUArrays` already provides `sum` and
# broadcasting over device arrays, so the weighted sum is one broadcasted reduction rather
# than a scalar loop. `mask` always comes from a mesh marker (`index_in_marker`), which is
# host memory regardless of backend, so it is copied onto `u`'s own device once before the
# masked reduction, rather than walked index by index (which would scalar-index the device
# array once per set bit).
#===========================================================================#

@inline _dot(policy::ExecutionPolicy, u, v, w) = _dot(locality(policy), policy, u, v, w)
@inline _dot_masked(policy::ExecutionPolicy, u, v, w, mask) = _dot_masked(
    locality(policy), policy, u, v, w, mask)

# Completes the `(Locality, ExecutionPolicy)` dispatch the two lines above open, the same
# shape `_sweep_for!` already uses: a `HostLocality` destination reaches straight back to the
# concrete-policy methods above (no behaviour change on CPU), the `DeviceLocality`/`GpuPolicy`
# methods below are the device side, and the catch-all at the end of this file throws via
# `_throw_locality_mismatch` for the two mismatched pairings. Without these, `report_package`
# only sees the `(DeviceLocality, GpuPolicy)` case and flags the rest as unreachable dispatch.
@inline _dot(::HostLocality, ::CpuSerial, u, v, w) = _dot(u, v, w)
@inline _dot(::HostLocality, ::CpuThreaded, u, v, w) = _dot(u, v, w)
@noinline _dot(::HostLocality, ::CpuBatch, u, v, w) = _batch_dot(u, v, w)

@inline _dot_masked(::HostLocality, ::CpuSerial, u, v, w, mask) = _dot_masked(u, v, w, mask)
@inline _dot_masked(::HostLocality, ::CpuThreaded, u, v, w, mask) = _dot_masked(u, v, w, mask)
@noinline _dot_masked(::HostLocality, ::CpuBatch, u, v, w, mask) = _batch_dot_masked(u, v, w, mask)

"""
    _dot(::DeviceLocality, ::GpuPolicy, u::AbstractVector, v::AbstractVector, w::AbstractVector) -> Real

The device counterpart of [`_dot`](@ref)`(u, v, w)`: `sum(u .* v .* w)`, one broadcasted
`GPUArrays` reduction, reached once a [`GpuPolicy`](@ref) derives [`DeviceLocality`](@ref)
from itself.

# Throws
- `DimensionMismatch`: If `length(u)`, `length(v)`, and `length(w)` do not match.
"""
@noinline function _dot(
        ::DeviceLocality, ::GpuPolicy, u::AbstractVector, v::AbstractVector, w::AbstractVector
)
    (length(u) == length(v) == length(w)) ||
        _throw_dot_dim_error(length(u), length(v), length(w))
    return sum(u .* v .* w)
end

"""
    _dot_masked(::DeviceLocality, ::GpuPolicy, u, v, w, mask::BitVector) -> Real

The device counterpart of [`_dot_masked`](@ref)`(u, v, w, mask)`: `mask` is copied onto
`u`'s own device once, then the masked sum is `sum(u .* v .* w .* md)`, one broadcasted
reduction instead of a walk over [`MarkedIndices`](@ref).

# Throws
- `DimensionMismatch`: If vector or mask lengths do not match.
"""
@noinline function _dot_masked(
        ::DeviceLocality, ::GpuPolicy, u::AbstractVector, v::AbstractVector,
        w::AbstractVector, mask::BitVector
)
    (length(u) == length(v) == length(w) == length(mask)) ||
        _throw_dot_dim_error(length(u), length(v), length(w), length(mask))
    md = similar(u, Bool)
    copyto!(md, Vector{Bool}(mask))
    return sum(u .* v .* w .* md)
end

"""
    _dot_masked(::DeviceLocality, ::GpuPolicy, u, v, w, mask::MarkedIndicesUnion) -> Real

The multi-marker counterpart of the `BitVector` method above: `mask` is walked once, on the
host, to build a plain `BitVector` (the same union `_combined_mask` would have
materialized), which is then copied onto `u`'s own device exactly as above.

# Throws
- `DimensionMismatch`: If vector or mask lengths do not match.
"""
@noinline function _dot_masked(
        ::DeviceLocality, ::GpuPolicy, u::AbstractVector, v::AbstractVector,
        w::AbstractVector, mask::MarkedIndicesUnion
)
    (length(u) == length(v) == length(w) == mask.len) ||
        _throw_dot_dim_error(length(u), length(v), length(w), mask.len)
    hostmask = falses(mask.len)
    @inbounds for i in mask
        hostmask[i] = true
    end
    md = similar(u, Bool)
    copyto!(md, Vector{Bool}(hostmask))
    return sum(u .* v .* w .* md)
end

# The `_sweep_for!` catch-all's counterpart: a locality/policy pairing neither the host nor
# the device methods above claim (a device destination under a `CpuPolicy`, or a host one
# under a `GpuPolicy`) names which half disagreed instead of falling through to a `MethodError`.
@noinline _dot(loc::Locality, policy, u, v, w) = _throw_locality_mismatch(loc, policy)
@noinline _dot_masked(loc::Locality, policy, u, v, w, mask) = _throw_locality_mismatch(loc, policy)
