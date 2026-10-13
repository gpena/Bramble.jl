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
- `policy`: Execution policy ([`CpuSerial`](@ref), [`CpuThreaded`](@ref), [`CpuPolyester`](@ref)
  or a [`GpuPolicy`](@ref)).
- `v`: Destination array mutated in place.
- `idxs`: Iterable collection of linear or Cartesian indices.
- `f`: Kernel mapping each index `idx` to the scalar value stored in `v[idx]`.
"""
@inline _sweep_for!(::HostLocality, ::CpuSerial, v, idxs, f::F) where {F} = _serial_for!(v, idxs, f)
@inline _sweep_for!(::HostLocality, ::CpuThreaded, v, idxs, f::F) where {F} = _threaded_for!(v, idxs, f)
@noinline _sweep_for!(::HostLocality, ::CpuPolyester, v, idxs, f::F) where {F} = _late(_batch_for!, v, idxs, f)
@noinline _sweep_for!(::DeviceLocality, policy::GpuPolicy, v, idxs, f::F) where {F} = _gpu_for!(policy, v, idxs, f)
@noinline _sweep_for!(loc::Locality, policy, v, idxs, f::F) where {F} = _throw_locality_mismatch(loc, policy)

@inline _sweep_for!(policy::ExecutionPolicy, v, idxs, f::F) where {F} = _sweep_for!(
    locality(typeof(v)), policy, v, idxs, f)

# `Threads.@threads` needs an indexable collection, so a `CartesianIndices` would be indexed
# linearly and pay an index conversion per point. Splitting the last axis gives each thread a
# `CartesianIndices` block of its own, iterated as the serial loop iterates the whole.
@inline _sweep_for!(::HostLocality, ::CpuThreaded, v, idxs::CartesianIndices, f::F) where {F} = _threaded_axis_for!(
    v, idxs, f)
@noinline _sweep_for!(::HostLocality, ::CpuPolyester, v, idxs::CartesianIndices, f::F) where {F} = _late(
    _batch_axis_for!, v, idxs, f)

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
        "CpuSerial(), CpuThreaded() or CpuPolyester().",
    ),
    )
end
@noinline function _throw_locality_mismatch(::DeviceLocality, policy)
    throw(
        ArgumentError(
        "the destination array has device locality, but execution policy $(typeof(policy)) " *
        "claims host locality: a sweep cannot mix the two. A host loop cannot scalar-index " *
        "device memory; pass a GpuPolicy (GpuKernel()) instead, with the device extension " *
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
# The GpuPolicy device sweep seam (gpena/Bramble.jl#94, #174, #298).
#
# `_sweep_for!`/`_sweep_scatter_for!` hand a `(DeviceLocality, GpuPolicy)` pair to
# `_gpu_for!`/`_gpu_scatter_for!` rather than refusing it outright. Both are declared here
# with their array/index/kernel arguments deliberately untyped -- matching the `ka_device`/
# `_launch_half_points!` fallback idiom (`src/utils/device_kernels.jl`,
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

# The message every `CpuPolyester` hook gives without `Polyester` loaded (gpena/Bramble.jl#190).
# `CpuPolyester`'s own sweeps have no `src/` implementation -- `BramblePolyesterExt` adds
# it -- so a `CpuPolyester` backend used without that extension loaded stops here, named, rather
# than silently falling through to `CpuThreaded`'s `Threads.@threads` code (which would
# defeat the whole point of choosing `CpuPolyester`) or a bare `MethodError`. Mirrors
# `_throw_gpu_in_cpu_loop` above and `_metal_backend` (`src/utils/backend.jl`).
@noinline function _throw_cpubatch_without_polyester(fname::Symbol)
    throw(
        ArgumentError(
        "CpuPolyester requires Polyester.jl. Add `using Polyester` before calling $(fname) " *
        "under a CpuPolyester backend.",
    ),
    )
end

# Every `src/` call into a `_batch_*` hook goes through `_late` (gpena/Bramble.jl#414). The
# precompile workload infers some `CpuPolyester` branches with abstract argument types, and
# an abstract call into a hook records a backedge that `BramblePolyesterExt`'s narrower
# method later fires, invalidating the caller. For abstract, non-Union argument types `_late`
# hides `f` behind `Base.inferencebarrier`, so no edge is recorded; for concrete ones it emits
# the plain call, keeping static dispatch and zero allocation. A small Union is split before it
# gets here, so its concrete arms record edges unprotected; the Polyester invalidation snoop
# (test/quality/invalidations_polyester_snoop.jl) is what catches such a caller.
@generated function _late(f, args...)
    Base.isdispatchtuple(Tuple{f, args...}) && return :(f(args...))
    return :(Base.inferencebarrier(f)(args...))
end

"""
    _batch_for!(v::AbstractArray, idxs, f::Function) -> Nothing

[`CpuPolyester`](@ref)'s counterpart of [`_threaded_for!`](@ref), filled by
`BramblePolyesterExt` (gpena/Bramble.jl#190). The only `src/` method errors naming
Polyester, the way `_metal_backend` errors naming Metal.
"""
@noinline function _batch_for!(v, idxs, f)
    return _throw_cpubatch_without_polyester(:_batch_for!)
end

"""
    _batch_axis_for!(v::AbstractArray, idxs::CartesianIndices, f::Function) -> Nothing

[`CpuPolyester`](@ref)'s counterpart of [`_threaded_axis_for!`](@ref), filled by
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
which is the entire point of forcing it -- while [`CpuThreaded`](@ref) and [`CpuPolyester`](@ref)
keep their own policy, so a `CpuPolyester` backend still runs its own (Polyester) sweep, or
errors clearly without it, rather than silently substituting `Threads.@threads`. A
[`GpuPolicy`](@ref) is coerced to [`CpuThreaded`](@ref) as well (`bilinear_traversal.jl`): the
colour-banded sweep is storage-generic, so a device backend runs the same `Threads.@threads`
sweep.
"""
@inline _effective_parallel_policy(sp) = _coerce_serial_to_threaded(execution_policy(sp))
@inline _coerce_serial_to_threaded(::CpuSerial) = CpuThreaded()
@inline _coerce_serial_to_threaded(policy::CpuPolicy) = policy

"""
    _in_threaded_region() -> Bool

Whether a threaded region is running, so that a `Threads.@threads :static` loop started now
would throw "`@threads :static` cannot be used concurrently or nested". This is exactly the
condition `Base.Threads` itself tests before a `:static` loop
(`ccall(:jl_in_threaded_region, Cint, ()) != 0`, `base/threadingconstructs.jl`): `true`
inside a user's `Threads.@threads` loop, `false` at top level. The counter behind it is
process-wide, so it is also `true` in a `Threads.@spawn`ed task while some other task's
threaded region runs -- and can turn `true` between this check and Base's own, which is why
[`_static_or_serial`](@ref) also catches Base's error.
"""
@inline _in_threaded_region() = ccall(:jl_in_threaded_region, Cint, ()) != 0

"""
    _is_static_nesting_error(err) -> Bool

Whether `err` is the error `Threads.@threads :static` raises when a threaded region is
already running: an `ErrorException` carrying exactly Base's message. Base raises it before
it creates or schedules a single task, so no iteration of the loop has run.
"""
@inline function _is_static_nesting_error(err)
    err isa ErrorException || return false
    msg = err.msg
    return msg isa String && msg == "`@threads :static` cannot be used concurrently or nested"
end

"""
    _static_or_serial(threaded!, serial!, args...) -> Nothing

Run `threaded!(args...)`, a function whose only work is one `Threads.@threads :static`
loop, or `serial!(args...)`, the same loop body run over the same chunks or bands in order
on the calling task, when a `:static` loop cannot start here.

Every [`CpuThreaded`](@ref) sweep goes through it. [`_in_threaded_region`](@ref) is checked
first, so a call from inside a user's `Threads.@threads` loop goes straight to `serial!`.
Because that counter is process-wide, a concurrent task can still enter its own threaded
region between this check and Base's; Base then throws before running any iteration, and
exactly that error ([`_is_static_nesting_error`](@ref)) is caught and answered by
`serial!`. Anything else is rethrown. Either way the caller gets what a top-level call
returns: reductions fill the same partial sums in the same order, so match it bitwise.

`threaded!` and `serial!` are isolated functions rather than closures, so no closure is
boxed on the path that runs serially.
"""
@inline function _static_or_serial(
        threaded!::F, serial!::G, args::Vararg{Any, N}
) where {F, G, N}
    _in_threaded_region() && return serial!(args...)
    try
        return threaded!(args...)
    catch err
        _is_static_nesting_error(err) || rethrow()
    end
    return serial!(args...)
end

"""
    _static_partials!(partials, n::Int, band, args...) -> Nothing
    _serial_partials!(partials, n::Int, band, args...) -> Nothing

Fill `partials[k] = band(args..., k)` for `k in 1:n`: one `k` per thread under
`Threads.@threads :static`, or in order on the calling task. The pair every threaded
reduction hands to [`_static_or_serial`](@ref); both call the same `@noinline` `band`, so
their partial sums agree bitwise.
"""
@noinline function _static_partials!(partials, n::Int, band::F, args::Vararg{Any, N}) where {F, N}
    Threads.@threads :static for k in 1:n
        @inbounds partials[k] = band(args..., k)
    end
    return nothing
end

@noinline function _serial_partials!(partials, n::Int, band::F, args::Vararg{Any, N}) where {F, N}
    for k in 1:n
        @inbounds partials[k] = band(args..., k)
    end
    return nothing
end

"""
    _static_bands!(f, nbands::Int, args...) -> Nothing
    _serial_bands!(f, nbands::Int, args...) -> Nothing

Run `f(args..., nbands, b)` for every band `b in 1:nbands`: one band per thread under
`Threads.@threads :static`, or in order on the calling task. The pair the banded stencil
engines hand to [`_static_or_serial`](@ref).
"""
@noinline function _static_bands!(f::F, nbands::Int, args::Vararg{Any, N}) where {F, N}
    Threads.@threads :static for b in 1:nbands
        f(args..., nbands, b)
    end
    return nothing
end

@noinline function _serial_bands!(f::F, nbands::Int, args::Vararg{Any, N}) where {F, N}
    for b in 1:nbands
        f(args..., nbands, b)
    end
    return nothing
end

"""
    _threaded_for!(v::AbstractArray, idxs, f::Function) -> Nothing

Fill `v[idx]` with `f(idx)` across threads, statically partitioning `idxs`, or serially
where a `:static` loop cannot start ([`_static_or_serial`](@ref)).

The `Threads.@threads` loop is kept in an isolated function to prevent closure boxing
allocations on paths that execute serially.
"""
@noinline _threaded_for!(v, idxs, f::F) where {F} = _static_or_serial(
    _static_for!, _serial_for!, v, idxs, f)

@noinline function _static_for!(v, idxs, f::F) where {F}
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
stride keeps it, and so does a descending one.

The positions are `Int` whatever `length(ax)` returns: a `UInt64` range has a `UInt64`
length, and Base indexes a descending `StepRange` by a `UnitRange{UInt64}` with a wrapped,
unsigned step (`(UInt(9):-1:UInt(1))[UInt(1):UInt(5)]` is empty).
"""
@inline function _band_range(ax::AbstractRange, nbands::Int, b::Int)
    q, r = divrem(Int(length(ax)), nbands)
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

@inline function Base.getindex(c::_LastAxisChunks{D}, k::Int) where {D}
    return CartesianIndices((c.rest..., _band_range(c.ax, c.n, k)))
end

"""
    _threaded_axis_for!(v::AbstractArray, idxs::CartesianIndices, f::Function) -> Nothing

As [`_threaded_for!`](@ref), for a `CartesianIndices`: each thread takes one block of
whole last-axis slices and walks it natively, never converting a linear index.
"""
@noinline function _threaded_axis_for!(v, idxs::CartesianIndices, f::F) where {F}
    blocks = _last_axis_chunks(idxs, Threads.nthreads())
    return _static_or_serial(_static_bands!, _serial_bands!, _FillBlock(v, blocks, f), length(blocks))
end

# The `b`-th block of `_threaded_axis_for!`, in the `f(args..., nbands, b)` shape
# `_static_bands!` calls with no `args`. A callable struct carries the kernel in its type,
# where a function passed with `v, blocks, f` through the `args...` Vararg would compile
# the band helpers with a `Function` kernel (gpena/Bramble.jl#459).
struct _FillBlock{V, B, F}
    v::V
    blocks::B
    f::F
end

@inline function (k::_FillBlock)(_, b::Int)
    v, f = k.v, k.f
    @inbounds for I in k.blocks[b]
        v[I] = f(I)
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
- `policy`: Execution policy ([`CpuSerial`](@ref), [`CpuThreaded`](@ref), [`CpuPolyester`](@ref)
  or a [`GpuPolicy`](@ref)).
- `mats`: Tuple of destination arrays mutated in place.
- `idxs`: Iterable collection of indices.
- `g`: Kernel mapping each index to a tuple of values matching `length(mats)`.
"""
@inline function _sweep_scatter_for!(::HostLocality, ::CpuSerial, mats::Tuple, idxs, g::G) where {G}
    @inbounds for idx in idxs
        _write_components!(mats, g(idx), idx)
    end
    return nothing
end
@inline _sweep_scatter_for!(::HostLocality, ::CpuThreaded, mats::Tuple, idxs, g::G) where {G} = _threaded_scatter_for!(
    mats, idxs, g)
@noinline _sweep_scatter_for!(::HostLocality, ::CpuPolyester, mats::Tuple, idxs, g::G) where {G} = _late(
    _batch_scatter_for!, mats, idxs, g)
@noinline _sweep_scatter_for!(
    ::DeviceLocality, policy::GpuPolicy, mats::Tuple, idxs, g::G) where {G} = _gpu_scatter_for!(
    policy, mats, idxs, g)
@noinline _sweep_scatter_for!(loc::Locality, policy, mats::Tuple, idxs, g::G) where {G} = _throw_locality_mismatch(
    loc, policy)

@inline _sweep_scatter_for!(policy::ExecutionPolicy, mats::Tuple, idxs, g::G) where {G} = _sweep_scatter_for!(
    locality(typeof(mats[1])), policy, mats, idxs, g)

"""
    _threaded_scatter_for!(mats::Tuple, idxs, g::Function) -> Nothing

Evaluate tuple-valued `g` across `idxs` and scatter the results into `mats`, statically
partitioning `idxs` across threads.

Kept in an isolated function to prevent `Threads.@threads` closure boxing allocations on
paths that execute serially.
"""
@noinline _threaded_scatter_for!(mats::Tuple, idxs, g::G) where {G} = _static_or_serial(
    _static_scatter_for!, _serial_scatter_for!, mats, idxs, g)

@noinline function _static_scatter_for!(mats::Tuple, idxs, g::G) where {G}
    Threads.@threads :static for idx in idxs
        @inbounds _write_components!(mats, g(idx), idx)
    end
    return nothing
end

@noinline function _serial_scatter_for!(mats::Tuple, idxs, g)
    @inbounds for idx in idxs
        _write_components!(mats, g(idx), idx)
    end
    return nothing
end

"""
    _batch_scatter_for!(mats::Tuple, idxs, g::Function) -> Nothing

[`CpuPolyester`](@ref)'s counterpart of `_threaded_scatter_for!`, filled by
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

All `masks` must have the same length: `len` is taken from the first, and no constructor
check compares the others, so a longer or shorter mask reads indices outside `1:len`.
"""
struct MarkedIndicesUnion{N}
    chunks::NTuple{N, Vector{UInt64}}
    len::Int
end

@inline function MarkedIndicesUnion(masks::NTuple{N, BitVector}) where {N}
    return MarkedIndicesUnion{N}(map(m -> m.chunks, masks), length(masks[1]))
end

@inline _reduce_or_chunk(chunks::NTuple{N, Vector{UInt64}}, i::Int) where {N} = reduce(|, ntuple(k -> chunks[k][i], Val(N)))

# Word accessors for the masked dot bands (`_dot_masked_band`), the same three for both mask
# types. `_mask_word(mask, w)` returns the `w`-th 64-bit word of the mask: for a
# `MarkedIndicesUnion`, the OR of every marker's. `_mask_nwords(mask)` counts the words, and
# `_mask_length(mask)` gives the length of the index range the mask selects from.
#
# A tuple of word vectors counts as a mask too, whatever its array type: `BramblePolyesterExt`
# hands the words across `@batch` as `PtrArray`s. The OR is unrolled by recursion over the
# tuple, since `reduce` over an `ntuple` ran the masked `innerₕ` about 1.2x slower once the
# words were `PtrArray`s (gpena/Bramble.jl#473).
@inline _mask_word(chunks::Tuple{Any}, w::Int) = @inbounds chunks[1][w]
@inline function _mask_word(chunks::Tuple, w::Int)
    return @inbounds chunks[1][w] | _mask_word(Base.tail(chunks), w)
end
@inline _mask_word(mask::BitVector, w::Int) = @inbounds mask.chunks[w]
@inline _mask_word(mask::MarkedIndicesUnion, w::Int) = _mask_word(mask.chunks, w)

@inline _mask_nwords(mask::BitVector) = length(mask.chunks)
@inline _mask_nwords(mask::MarkedIndicesUnion) = length(mask.chunks[1])

@inline _mask_length(mask::BitVector) = length(mask)
@inline _mask_length(mask::MarkedIndicesUnion) = mask.len

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

# One term of the weighted trilinear sum, `s + conj(u[i]) * w[i] * v[i]` in the common
# element type `T`: the line every `_dot`, `_dot_masked` and band below accumulates with.
@inline function _dot_term(s, u, w, v, i, ::Type{T}) where {T}
    return muladd(conj(T(u[i])) * T(w[i]), T(v[i]), s)
end

# The masked bands read the weight through `_masked_cart(w)`: `nothing` for a vector, which
# `_masked_term` reads as `_dot_term` does, and the `CartesianIndices` of a
# `SeparableWeights` (`src/space/inner_product.jl`), which `w` is read through instead of its
# linear `getindex`. One band and one driver then serve both.
@inline _masked_cart(::AbstractVector) = nothing
@inline _masked_term(s, u, w, v, i, ::Type{T}, ::Nothing) where {T} = _dot_term(s, u, w, v, i, T)

"""
    _dot(u::AbstractVector, w::AbstractVector, v::AbstractVector) -> Number

Compute the weighted trilinear dot product
```math
\\sum_{i=1}^n \\overline{u_i} w_i v_i
```

Accumulates via fused multiply-add operations (`muladd`) with `@simd` vectorization.

The first argument is conjugated (gpena/Bramble.jl#644), and so is `u` in every policy and
device counterpart below; the weight is the second argument, as in every caller. `conj` of a `Real` is the identity, so real data, `Dual`s
included, gives the same bits as the unconjugated product.

A same-eltype specialization used to sit alongside this one, skipping the `promote_type`
call and the `T(...)` conversions on the (dispatch-favoured) assumption that they cost
something. Compared by `@code_llvm`/`@code_native` with matching element types
(gpena/Bramble.jl#71): identical generated code, since `promote_type(T, T, T) === T` and
`T(x::T)` is an identity conversion the compiler elides. One method now covers both cases.

# Arguments
- `u`: First vector, the one that is conjugated.
- `w`: The weights, second in the call as at every caller.
- `v`: Third vector.

# Throws
- `DimensionMismatch`: If `length(u)`, `length(w)`, and `length(v)` do not match.
"""
@inline function _dot(u::AbstractVector, w::AbstractVector, v::AbstractVector)
    (length(u) == length(w) == length(v)) ||
        _throw_dot_dim_error(length(u), length(w), length(v))
    T = promote_type(eltype(u), eltype(w), eltype(v))
    s = zero(T)

    @inbounds @simd for i in 1:length(u)
        s = _dot_term(s, u, w, v, i, T)
    end

    return s
end

"""
    _dot_masked(u::AbstractVector, w::AbstractVector, v::AbstractVector, mask::BitVector) -> Number

Compute the weighted dot product restricted to indices where `mask` is true:
```math
\\sum_{i \\in \\mathrm{supp}(\\mathrm{mask})} \\overline{u_i} w_i v_i
```

Walks `MarkedIndices(mask)`, so the work is proportional to the number of set bits
rather than to `length(mask)`.

As with `_dot`, a same-eltype specialization used to sit alongside this one;
`@code_llvm`/`@code_native` with matching element types (gpena/Bramble.jl#71) showed
identical generated code, so one method now covers both cases.

# Arguments
- `u`: First vector, the one that is conjugated.
- `w`: The weights, second in the call as at every caller.
- `v`: Third vector.
- `mask`: Boolean selection mask.

# Throws
- `DimensionMismatch`: If vector or mask lengths do not match.
"""
@inline function _dot_masked(
        u::AbstractVector, w::AbstractVector, v::AbstractVector, mask::BitVector
)
    (length(u) == length(w) == length(v) == length(mask)) ||
        _throw_dot_dim_error(length(u), length(w), length(v), length(mask))
    T = promote_type(eltype(u), eltype(w), eltype(v))
    s = zero(T)

    @inbounds for i in MarkedIndices(mask)
        s = _dot_term(s, u, w, v, i, T)
    end

    return s
end

# The multi-marker counterpart, walking a `MarkedIndicesUnion` (gpena/Bramble.jl#149)
# instead of a `BitVector`: `_combined_mask` hands one of these straight in, with no
# intermediate combined mask to check the length of, so the guard reads `mask.len`.
@inline function _dot_masked(
        u::AbstractVector, w::AbstractVector, v::AbstractVector, mask::MarkedIndicesUnion
)
    (length(u) == length(w) == length(v) == mask.len) ||
        _throw_dot_dim_error(length(u), length(w), length(v), mask.len)
    T = promote_type(eltype(u), eltype(w), eltype(v))
    s = zero(T)

    @inbounds for i in mask
        s = _dot_term(s, u, w, v, i, T)
    end

    return s
end

# --- Execution-policy-dispatched reduction entries (gpena/Bramble.jl#190, #301) #
#
# `inner₊` (`src/space/inner_product.jl`) passes a policy into `_dot`/`_dot_masked`.
# `CpuSerial` runs the single vectorised serial kernel. `CpuThreaded` reduces over
# `Threads.nthreads()` static chunks (`_threaded_dot`, `_threaded_dot_masked`) into a buffer
# sized by the thread count and sums it serially. `CpuPolyester` reaches the
# `_batch_dot`/`_batch_dot_masked` hooks below, which `BramblePolyesterExt` supplies, and
# `GpuPolicy` the device methods further down.

"""
    _threaded_dot(u::AbstractVector, w::AbstractVector, v::AbstractVector) -> Number

[`CpuThreaded`](@ref)'s own counterpart of [`_dot`](@ref): partitions `1:length(u)` into
`Threads.nthreads()` static bands via [`_band_range`](@ref), each `Threads.@threads :static`
task accumulating its own `muladd`/`@simd` partial sum, landed in a fixed
`Threads.nthreads()`-length buffer and summed serially once every task has returned. Where
a `:static` loop cannot start ([`_static_or_serial`](@ref)) the same bands run in turn on
the calling task, through the same [`_dot_band`](@ref), so the value is unchanged bitwise.

The buffer's size depends only on the thread count, never on `length(u)`, so allocation is
independent of vector length rather than zero. No size threshold falls back to the serial
kernel below a cutoff, and no new dependency is pulled in to avoid the buffer.

Kept in an isolated function, the same reason [`_threaded_for!`](@ref) is, to keep
`Threads.@threads` from boxing a closure over `u`/`w`/`v` on paths that execute serially.

# Throws
- `DimensionMismatch`: If `length(u)`, `length(w)`, and `length(v)` do not match.
"""
@noinline function _threaded_dot(u::AbstractVector, w::AbstractVector, v::AbstractVector)
    (length(u) == length(w) == length(v)) ||
        _throw_dot_dim_error(length(u), length(w), length(v))
    T = promote_type(eltype(u), eltype(w), eltype(v))
    n = length(u)
    nchunks = Threads.nthreads()
    partials = zeros(T, nchunks)
    ax = 1:n
    _static_or_serial(
        _static_partials!, _serial_partials!, partials, nchunks, _dot_band, u, w, v, ax, nchunks)
    return sum(partials)
end

"""
    _dot_band(u, w, v, ax, nchunks::Int, k::Int) -> Number

The `k`-th of `nchunks` partial sums of [`_threaded_dot`](@ref), over the band
[`_band_range`](@ref)`(ax, nchunks, k)`. One compiled body serves both the threaded loop and
the serial one [`_static_or_serial`](@ref) falls back to, so the two return the same partial
sums bitwise. `BramblePolyesterExt` runs `_dot_band` and keeps a copy of
[`_dot_masked_band`](@ref) over the mask's words (`_masked_words_band`), so a change here
must be made there too (gpena/Bramble.jl#473).
"""
@noinline function _dot_band(u, w, v, ax, nchunks::Int, k::Int)
    T = promote_type(eltype(u), eltype(w), eltype(v))
    rng = _band_range(ax, nchunks, k)
    s = zero(T)
    @inbounds @simd for i in rng
        s = _dot_term(s, u, w, v, i, T)
    end
    return s
end

"""
    _threaded_dot_masked(u::AbstractVector, w::AbstractVector, v::AbstractVector, mask::Union{BitVector, MarkedIndicesUnion}) -> Number

[`CpuThreaded`](@ref)'s own counterpart of [`_dot_masked`](@ref): partitions the mask's
64-bit *words* (not `1:length(u)`) into `Threads.nthreads()` static bands, so the work stays
proportional to the number of set bits rather than to `length(u)` -- masks here are typically
a mesh marker such as `:boundary`, `O(perimeter)` set bits inside an `O(n^D)` index range, and
walking every index (as an earlier version of this function did) turned that sparsity into a
slowdown under `CpuThreaded` relative to plain `CpuSerial`.

Each task loads its band's words once (`_mask_word`: `mask.chunks[w]` for a `BitVector`, joined
across markers for a [`MarkedIndicesUnion`](@ref)), skips whole zero
words, and walks a nonzero word's set bits with `trailing_zeros`/`word & (word - 1)`, the same
walk [`MarkedIndices`](@ref)/[`MarkedIndicesUnion`](@ref) use. No bounds guard against
`length(u)` is needed for the padding bits of the final word, for the same reason
[`MarkedIndices`](@ref) needs none: a `BitVector`'s own invariant guarantees they are zero,
and a [`MarkedIndicesUnion`](@ref) ORs only `BitVector`s of that same length together.

# Throws
- `DimensionMismatch`: If vector or mask lengths do not match.
"""
@noinline function _threaded_dot_masked(
        u::AbstractVector, w::AbstractVector, v::AbstractVector,
        mask::Union{BitVector, MarkedIndicesUnion}
)
    (length(u) == length(w) == length(v) == _mask_length(mask)) ||
        _throw_dot_dim_error(length(u), length(w), length(v), _mask_length(mask))
    T = promote_type(eltype(u), eltype(w), eltype(v))
    nchunks = Threads.nthreads()
    partials = zeros(T, nchunks)
    ax = 1:_mask_nwords(mask)
    _static_or_serial(_static_partials!, _serial_partials!, partials, nchunks,
        _dot_masked_band, u, w, v, mask, ax, nchunks)
    return sum(partials)
end

"""
    _dot_masked_band(u, w, v, mask::Union{BitVector, MarkedIndicesUnion}, ax, nchunks::Int, k::Int) -> Number

The `k`-th of `nchunks` partial sums of [`_threaded_dot_masked`](@ref), over the mask words
[`_band_range`](@ref)`(ax, nchunks, k)`. As [`_dot_band`](@ref), one compiled body serves the
threaded loop and the serial one. A `SeparableWeights` weight goes through it too. The band
reads such a weight through its `CartesianIndices` (`_masked_cart`, `_masked_term`) and never
through the linear `getindex`. `BramblePolyesterExt` runs [`_dot_band`](@ref) and keeps a copy
of `_dot_masked_band` over the mask's words (`_masked_words_band` and
`_separable_masked_words_band`), so a change here must be made there too
(gpena/Bramble.jl#473).
"""
@noinline function _dot_masked_band(
        u, w, v, mask::Union{BitVector, MarkedIndicesUnion}, ax, nchunks::Int, k::Int)
    T = promote_type(eltype(u), eltype(w), eltype(v))
    rng = _band_range(ax, nchunks, k)
    cart = _masked_cart(w)
    s = zero(T)
    @inbounds for widx in rng
        word = _mask_word(mask, widx)
        base = (widx - 1) * 64
        while word != zero(UInt64)
            i = base + trailing_zeros(word) + 1
            s = _masked_term(s, u, w, v, i, T, cart)
            word &= word - 1
        end
    end
    return s
end

"""
    _batch_dot(u, w, v) -> Number

[`CpuPolyester`](@ref)'s counterpart of [`_dot`](@ref), filled by `BramblePolyesterExt`. The
only `src/` method errors naming Polyester.
"""
@noinline function _batch_dot(u, w, v)
    return _throw_cpubatch_without_polyester(:_batch_dot)
end

"""
    _batch_dot_masked(u, w, v, mask) -> Number

[`CpuPolyester`](@ref)'s counterpart of [`_dot_masked`](@ref), filled by
`BramblePolyesterExt`. The only `src/` method errors naming Polyester.
"""
@noinline function _batch_dot_masked(u, w, v, mask)
    return _throw_cpubatch_without_polyester(:_batch_dot_masked)
end

#===========================================================================#
# Device reductions (gpena/Bramble.jl#94, #174).
#
# Every policy, CPU or GPU, reaches the generic `ExecutionPolicy` method below. It derives
# locality from the policy itself -- `locality(::CpuPolicy) = HostLocality()`,
# `locality(::GpuPolicy) = DeviceLocality()` (`src/utils/backend.jl`) -- the same
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

"""
    _dot(policy::ExecutionPolicy, u, w, v) -> Number

Policy-dispatched [`_dot`](@ref): derives the locality from `policy` and dispatches on the
pair. On the host, [`CpuSerial`](@ref) reaches the plain three-vector method,
[`CpuThreaded`](@ref) reaches [`_threaded_dot`](@ref) and [`CpuPolyester`](@ref) reaches
[`_batch_dot`](@ref); a [`GpuPolicy`](@ref) reaches the device methods below.
"""
@inline _dot(policy::ExecutionPolicy, u, w, v) = _dot(locality(policy), policy, u, w, v)

"""
    _dot_masked(policy::ExecutionPolicy, u, w, v, mask) -> Number

Policy-dispatched [`_dot_masked`](@ref): as [`_dot`](@ref)`(policy, u, w, v)`, with the plain
masked method, [`_threaded_dot_masked`](@ref) and [`_batch_dot_masked`](@ref) on the host.
"""
@inline _dot_masked(policy::ExecutionPolicy, u, w, v, mask) = _dot_masked(
    locality(policy), policy, u, w, v, mask)

# Completes the `(Locality, ExecutionPolicy)` dispatch the methods above open, the same
# shape `_sweep_for!` already uses: the `HostLocality` methods below are the whole CPU side
# (reached through the policy-level methods, or by a caller passing a locality), the
# `DeviceLocality`/`GpuPolicy` methods after them are the device side, and the catch-all at
# the end of this file throws via `_throw_locality_mismatch` for the two mismatched
# pairings. Without these, `report_package` only sees the `(DeviceLocality, GpuPolicy)`
# case and flags the rest as unreachable dispatch.
@inline _dot(::HostLocality, ::CpuSerial, u, w, v) = _dot(u, w, v)
@inline _dot(::HostLocality, ::CpuThreaded, u, w, v) = _threaded_dot(u, w, v)
@noinline _dot(::HostLocality, ::CpuPolyester, u, w, v) = _late(_batch_dot, u, w, v)

@inline _dot_masked(::HostLocality, ::CpuSerial, u, w, v, mask) = _dot_masked(u, w, v, mask)
@inline _dot_masked(::HostLocality, ::CpuThreaded, u, w, v, mask) = _threaded_dot_masked(u, w, v, mask)
@noinline _dot_masked(::HostLocality, ::CpuPolyester, u, w, v, mask) = _late(_batch_dot_masked, u, w, v, mask)

"""
    _dot(::DeviceLocality, ::GpuPolicy, u::AbstractVector, w::AbstractVector, v::AbstractVector) -> Number

The device counterpart of [`_dot`](@ref)`(u, w, v)`: `sum(conj.(u) .* w .* v)`,
one broadcasted `GPUArrays` reduction, reached once a [`GpuPolicy`](@ref) derives
[`DeviceLocality`](@ref) from itself.

# Throws
- `DimensionMismatch`: If `length(u)`, `length(w)`, and `length(v)` do not match.
"""
@noinline function _dot(
        ::DeviceLocality, ::GpuPolicy, u::AbstractVector, w::AbstractVector, v::AbstractVector
)
    (length(u) == length(w) == length(v)) ||
        _throw_dot_dim_error(length(u), length(w), length(v))
    return sum(conj.(u) .* w .* v)
end

"""
    _dot_masked(::DeviceLocality, ::GpuPolicy, u, w, v, mask::BitVector) -> Number

The device counterpart of [`_dot_masked`](@ref)`(u, w, v, mask)`: `mask` is copied onto
`u`'s own device once, then the masked sum is `sum(conj.(u) .* w .* v .* md)`, one
broadcasted reduction instead of a walk over [`MarkedIndices`](@ref).

# Throws
- `DimensionMismatch`: If vector or mask lengths do not match.
"""
@noinline function _dot_masked(
        ::DeviceLocality, ::GpuPolicy, u::AbstractVector, w::AbstractVector,
        v::AbstractVector, mask::BitVector
)
    (length(u) == length(w) == length(v) == length(mask)) ||
        _throw_dot_dim_error(length(u), length(w), length(v), length(mask))
    md = similar(u, Bool)
    copyto!(md, Vector{Bool}(mask))
    return sum(conj.(u) .* w .* v .* md)
end

"""
    _dot_masked(::DeviceLocality, ::GpuPolicy, u, w, v, mask::MarkedIndicesUnion) -> Number

The multi-marker counterpart of the `BitVector` method above: `mask` is walked once, on the
host, to build a plain `BitVector` (the same union `_combined_mask` would have
materialized), which is then copied onto `u`'s own device exactly as above.

# Throws
- `DimensionMismatch`: If vector or mask lengths do not match.
"""
@noinline function _dot_masked(
        ::DeviceLocality, ::GpuPolicy, u::AbstractVector, w::AbstractVector,
        v::AbstractVector, mask::MarkedIndicesUnion
)
    (length(u) == length(w) == length(v) == mask.len) ||
        _throw_dot_dim_error(length(u), length(w), length(v), mask.len)
    hostmask = falses(mask.len)
    @inbounds for i in mask
        hostmask[i] = true
    end
    md = similar(u, Bool)
    copyto!(md, Vector{Bool}(hostmask))
    return sum(conj.(u) .* w .* v .* md)
end

# The `_sweep_for!` catch-all's counterpart: a locality/policy pairing neither the host nor
# the device methods above claim (a device destination under a `CpuPolicy`, or a host one
# under a `GpuPolicy`) names which half disagreed instead of falling through to a `MethodError`.
@noinline _dot(loc::Locality, policy, u, v, w) = _throw_locality_mismatch(loc, policy)
@noinline _dot_masked(loc::Locality, policy, u, v, w, mask) = _throw_locality_mismatch(loc, policy)
