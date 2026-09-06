@noinline function _throw_dot_dim_error(lu::Integer, lv::Integer, lw::Integer)
    throw(DimensionMismatch("Vectors must have matching lengths, but got lengths ($lu, $lv, $lw)."))
end

@noinline function _throw_dot_dim_error(lu::Integer, lv::Integer, lw::Integer, lm::Integer)
    throw(DimensionMismatch("Vectors and mask must have matching lengths, but got lengths ($lu, $lv, $lw, $lm)."))
end

"""
    _cpu_threaded_for!(policy::ExecutionPolicy, v::AbstractArray, idxs, f::Function) -> Nothing

Apply `f` across indices `idxs` and write the result into `v` in place.

Dispatches to sequential iteration for [`Serial`](@ref) or static work partitioning across
threads for [`Parallel`](@ref).

# Arguments
- `policy`: Execution policy ([`Serial`](@ref) or [`Parallel`](@ref)).
- `v`: Destination array mutated in place.
- `idxs`: Iterable collection of linear or Cartesian indices.
- `f`: Kernel mapping each index `idx` to the scalar value stored in `v[idx]`.
"""
@inline _cpu_threaded_for!(::Serial, v, idxs, f) = _serial_for!(v, idxs, f)
@inline _cpu_threaded_for!(::Parallel, v, idxs, f) = _threaded_for!(v, idxs, f)

# Kept in an isolated function to prevent Threads.@threads closure boxing allocations
# on paths that execute serially.
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
    _cpu_threaded_scatter_for!(policy::ExecutionPolicy, mats::Tuple, idxs, g::Function) -> Nothing

Evaluate tuple-valued kernel `g` across `idxs` and scatter results into destination arrays `mats`.

Dispatches to sequential execution for [`Serial`](@ref) or static multithreaded execution for [`Parallel`](@ref).

# Arguments
- `policy`: Execution policy ([`Serial`](@ref) or [`Parallel`](@ref)).
- `mats`: Tuple of destination arrays mutated in place.
- `idxs`: Iterable collection of indices.
- `g`: Kernel mapping each index to a tuple of values matching `length(mats)`.
"""
@inline function _cpu_threaded_scatter_for!(::Serial, mats::Tuple, idxs, g)
    @inbounds for idx in idxs
        _write_components!(mats, g(idx), idx)
    end
    return nothing
end
@inline _cpu_threaded_scatter_for!(::Parallel, mats::Tuple, idxs, g) = _threaded_scatter_for!(
    mats, idxs, g)

# Kept in an isolated function to prevent closure boxing allocations on serial execution paths.
@noinline function _threaded_scatter_for!(mats::Tuple, idxs, g)
    Threads.@threads :static for idx in idxs
        @inbounds _write_components!(mats, g(idx), idx)
    end
    return nothing
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
        u::AbstractVector, v::AbstractVector, w::AbstractVector, mask::BitVector)
    (length(u) == length(v) == length(w) == length(mask)) ||
        _throw_dot_dim_error(length(u), length(v), length(w), length(mask))
    T = promote_type(eltype(u), eltype(v), eltype(w))
    s = zero(T)

    @inbounds for i in MarkedIndices(mask)
        s = muladd(T(u[i]) * T(v[i]), T(w[i]), s)
    end

    return s
end
