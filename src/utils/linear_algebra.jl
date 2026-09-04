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

# Discrete space inner product kernels

"""
    _dot(u::AbstractVector, v::AbstractVector, w::AbstractVector) -> Real

Compute the weighted trilinear dot product
```math
\\sum_{i=1}^n u_i v_i w_i
```

Accumulates via fused multiply-add operations (`muladd`) with `@simd` vectorization.

# Arguments
- `u`: First vector.
- `v`: Second vector.
- `w`: Weight vector.

# Throws
- `DimensionMismatch`: If `length(u)`, `length(v)`, and `length(w)` do not match.
"""
@inline function _dot(u::AbstractVector{T}, v::AbstractVector{T},
        w::AbstractVector{T}) where {T}
    (length(u) == length(v) == length(w)) ||
        _throw_dot_dim_error(length(u), length(v), length(w))
    s = zero(T)

    @inbounds @simd for i in 1:length(u)
        s = muladd(u[i] * v[i], w[i], s)
    end

    return s
end

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

Traverses 64-bit integer words in `mask.chunks`, skipping zero chunks and extracting active
bit positions via `trailing_zeros` to avoid branch mispredictions and allocations.

# Arguments
- `u`: First vector.
- `v`: Second vector.
- `w`: Weight vector.
- `mask`: Boolean selection mask.

# Throws
- `DimensionMismatch`: If vector or mask lengths do not match.
"""
@inline function _dot_masked(
        u::AbstractVector{T}, v::AbstractVector{T}, w::AbstractVector{T},
        mask::BitVector) where {T}
    n = length(u)
    (n == length(v) == length(w) == length(mask)) ||
        _throw_dot_dim_error(n, length(v), length(w), length(mask))
    s = zero(T)

    chunks = mask.chunks
    @inbounds for c in eachindex(chunks)
        chunk = chunks[c]
        chunk == 0 && continue
        base_i = (c - 1) * 64
        while chunk != 0
            tz = trailing_zeros(chunk)
            i = base_i + tz + 1
            i > n && break
            s = muladd(u[i] * v[i], w[i], s)
            chunk &= chunk - 1
        end
    end

    return s
end

@inline function _dot_masked(
        u::AbstractVector, v::AbstractVector, w::AbstractVector, mask::BitVector)
    n = length(u)
    (n == length(v) == length(w) == length(mask)) ||
        _throw_dot_dim_error(n, length(v), length(w), length(mask))
    T = promote_type(eltype(u), eltype(v), eltype(w))
    s = zero(T)

    chunks = mask.chunks
    @inbounds for c in eachindex(chunks)
        chunk = chunks[c]
        chunk == 0 && continue
        base_i = (c - 1) * 64
        while chunk != 0
            tz = trailing_zeros(chunk)
            i = base_i + tz + 1
            i > n && break
            s = muladd(T(u[i]) * T(v[i]), T(w[i]), s)
            chunk &= chunk - 1
        end
    end

    return s
end
