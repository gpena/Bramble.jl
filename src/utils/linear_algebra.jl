@noinline function _throw_dot_dim_error(lu::Integer, lv::Integer, lw::Integer)
    throw(DimensionMismatch("Vectors must have matching lengths, but got lengths ($lu, $lv, $lw)."))
end

@noinline function _throw_dot_dim_error(lu::Integer, lv::Integer, lw::Integer, lm::Integer)
    throw(DimensionMismatch("Vectors and mask must have matching lengths, but got lengths ($lu, $lv, $lw, $lm)."))
end

"""
	$(SIGNATURES)

Applies `f` across `idxs`, writing into `v` in place, either as a plain loop or across
threads depending on `policy` -- [`Serial`](@ref) or [`Parallel`](@ref), read off a
[`Backend`](@ref) via [`execution_policy`](@ref).

# Arguments
- `policy`: [`Serial`](@ref) or [`Parallel`](@ref)
- `v`: Array to be modified in-place
- `idxs`: Iterable of indices to process
- `f`: Function that takes an index and returns the value to be stored at that index
"""
@inline _cpu_threaded_for!(::Serial, v, idxs, f) = _serial_for!(v, idxs, f)
@inline _cpu_threaded_for!(::Parallel, v, idxs, f) = _threaded_for!(v, idxs, f)

# Kept in its own function on purpose. `Threads.@threads` builds a closure over
# the loop body, and having it in the same body as the serial branch makes that
# closure allocate even on calls that never reach it.
@noinline function _threaded_for!(v, idxs, f)
    # :static partitions work evenly across threads — lower overhead for uniform workloads
    Threads.@threads :static for idx in idxs
        @inbounds v[idx] = f(idx)
    end
    return nothing
end

"""
	$(SIGNATURES)

Performs a serial (single-threaded) iteration over the specified indices, applying a function `f` to modify array `v` in-place.

# Arguments
- `v`: Array to be modified in-place
- `idxs`: Indices to iterate over
- `f`: Function to be applied at each index
"""
@inline function _serial_for!(v, idxs, f)
    @inbounds for idx in idxs
        v[idx] = f(idx)
    end
    return nothing
end

#=========================================================================
Scattering a tuple-valued kernel across several arrays.

`_cpu_threaded_for!` writes one value per index into one array. When the kernel
returns a tuple instead -- one value per component of a multi-component field --
the alternative is to call it once per component and keep only the i-th result,
which evaluates it `NC` times per index and discards `NC - 1` of every `NC`
values. Measured on an expensive kernel that is 1.9x the work at `NC = 2`, 2.8x
at `NC = 3` and 3.6x at `NC = 4`.

These evaluate the kernel once per index and scatter its tuple across the
target arrays.
=========================================================================#

"""
	$(SIGNATURES)

Writes each element of `vals` into the corresponding array of `mats` at `idx`.
Unrolled by recursion on the tuple, so there is no loop and no allocation.
"""
@inline _write_components!(::Tuple{}, ::Tuple, idx) = nothing
@inline function _write_components!(mats::Tuple, vals::Tuple, idx)
    @inbounds mats[1][idx] = vals[1]
    return _write_components!(Base.tail(mats), Base.tail(vals), idx)
end

"""
	$(SIGNATURES)

Applies `g` across `idxs` and scatters each returned tuple over the arrays in
`mats`, dispatching on `policy` -- [`Serial`](@ref) or [`Parallel`](@ref) -- on the
same terms as [`_cpu_threaded_for!`](@ref).

# Arguments
- `policy`: [`Serial`](@ref) or [`Parallel`](@ref)
- `mats`: Tuple of arrays to be modified in-place, one per component
- `idxs`: Iterable of indices to process
- `g`: Function taking an index and returning a tuple of values, one per array
"""
@inline function _cpu_threaded_scatter_for!(::Serial, mats::Tuple, idxs, g)
    @inbounds for idx in idxs
        _write_components!(mats, g(idx), idx)
    end
    return nothing
end
@inline _cpu_threaded_scatter_for!(::Parallel, mats::Tuple, idxs, g) = _threaded_scatter_for!(
    mats, idxs, g)

# Separate function for the same reason as `_threaded_for!`: sharing a body with
# the serial branch makes the `@threads` closure allocate even when unused.
@noinline function _threaded_scatter_for!(mats::Tuple, idxs, g)
    Threads.@threads :static for idx in idxs
        @inbounds _write_components!(mats, g(idx), idx)
    end
    return nothing
end

##################################################################################
#                                                                                #
#   Helper functions to calculate inner products in discrete spaces               #
#                                                                                #
##################################################################################

"""
	$(SIGNATURES)

Computes the weighted element-wise dot product of three vectors:
``\\sum_{i} u_i \\cdot v_i \\cdot w_i``

Uses SIMD and `muladd` for outer accumulation.
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
	$(SIGNATURES)

As [`_dot`](@ref), restricted to the indices `mask` marks:
``\\sum_{i \\,:\\, mask_i} u_i \\cdot v_i \\cdot w_i``.

Uses `BitVector` word-level skipping (`mask.chunks`) and bit scanning (`trailing_zeros`)
to traverse marked entries efficiently with zero allocations.
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
