@noinline function _throw_dot_dim_error(lu::Integer, lv::Integer, lw::Integer)
    throw(DimensionMismatch("Vectors must have matching lengths, but got lengths ($lu, $lv, $lw)."))
end

"""
	$(SIGNATURES)

Applies `f` across `idxs`, writing into `v` in place, either as a plain loop or across
threads depending on `policy` -- [`Serial`](@ref) or [`Parallel`](@ref), read off a
[`Backend`](@ref) via [`execution_policy`](@ref).

Resolved by ordinary dispatch on `policy`'s type, not a runtime size check: there is no
threshold below which a `Parallel` backend falls back to serial. That used to be
`CPU_THREADED_MIN`, a threshold the caller could not see or override and that would have
needed its own value per operation (`Rₕ!`'s crossover is not `avgₕ!`'s) -- removed together
with the automatic switching it existed to drive. Choosing `Serial()` on the backend is now
how a caller with many small, frequently repeated calls avoids `Threads.@threads`'s spawn
cost, deterministically, rather than relying on a heuristic to guess it for them.

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

"""
	$(SIGNATURES)

Applies `g` only at the indices selected by `masks`, writing into `v` in place
and leaving every unselected entry at zero.

Each mask is a `BitVector` over the linear indices of `v`, as returned by
`index_in_marker`; several masks act as a union. Marked sets are usually small
boundary strips, so this runs serially: the selection test costs more than the
kernel for most markers, and threading a short scattered write does not pay.

# Arguments
- `v`: Array to be modified in-place
- `masks`: Tuple of `BitVector`s over `LinearIndices(v)`
- `g`: Function that takes an index and returns the value to be stored there
"""
function _masked_for!(v, masks::Tuple, g)
    fill!(v, zero(eltype(v)))
    lin = LinearIndices(v)
    for mask in masks
        @inbounds for idx in CartesianIndices(v)
            mask[lin[idx]] && (v[idx] = g(idx))
        end
    end
    return nothing
end

# The masked counterpart of `_cpu_threaded_scatter_for!`, standing to it as `_masked_for!` stands to
# `_cpu_threaded_for!`: writes only where a mask selects, leaving every other entry zero.
#
# Serial by construction. The marked region is a boundary slice in every use so far, which
# is O(n^(D-1)) against the O(n^D) of the whole grid, and threading it would cost more than
# it saves.
function _masked_scatter_for!(mats::Tuple, masks::Tuple, g)
    for m in mats
        fill!(m, zero(eltype(m)))
    end
    first_mat = first(mats)
    lin = LinearIndices(first_mat)
    for mask in masks
        @inbounds for idx in CartesianIndices(first_mat)
            mask[lin[idx]] && _write_components!(mats, g(idx), idx)
        end
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

Uses SIMD and `muladd` for the outer accumulation (`u_i * v_i * w_i + s`); inner products are not FMA-fused.
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
	$(SIGNATURES)

Computes the weighted inner product ``\\langle u, v \\rangle_h``.

- For vectors `u` and `v` with weights `h`, computes ``\\sum_i u_i h_i v_i`` via [`_dot`](@ref).
- For matrices `u` and `v`, computes ``v^T \\operatorname{diag}(h) u``.
"""
@inline _inner_product(u::AbstractVector, h::AbstractVector, v::AbstractVector) = _dot(u, h, v)

"""
	$(SIGNATURES)

As [`_dot`](@ref), restricted to the indices `mask` marks:
``\\sum_{i \\,:\\, mask_i} u_i \\cdot v_i \\cdot w_i``.

No `@simd`: the branch on `mask[i]` rules it out, the same tradeoff
`symmetrize!` (form/dirichlet_constraints.jl) makes for the same reason.
"""
@inline function _dot_masked(
        u::AbstractVector, v::AbstractVector, w::AbstractVector, mask::BitVector)
    (length(u) == length(v) == length(w)) ||
        _throw_dot_dim_error(length(u), length(v), length(w))
    T = promote_type(eltype(u), eltype(v), eltype(w))
    s = zero(T)

    @inbounds for i in 1:length(u)
        mask[i] && (s = muladd(T(u[i]) * T(v[i]), T(w[i]), s))
    end

    return s
end

"""
	$(SIGNATURES)

As [`_inner_product`](@ref), restricted to the indices `mask` marks.
"""
@inline _inner_product_masked(u::AbstractVector, h::AbstractVector, v::AbstractVector,
    mask::BitVector) = _dot_masked(u, h, v, mask)

# Production code never reaches this: `_directional_inner_plus` passes `uₕ.data`, a vector,
# and so does the composite path through its components. The callers are `precompile.jl` and
# the testset that pins its semantics. It also allocates twice — `similar(u)` and then the
# product, 15,008 B where the result is 512 — so if it is ever given a real caller, a 5-arg
# `mul!` is the fix. Kept because deleting a tested utility on a reachability argument is a
# decision to take deliberately, and that argument was already wrong once: this method's
# test was missed by a sweep that only grepped `src/`.
function _inner_product(u::AbstractMatrix, h::AbstractVector, v::AbstractMatrix)
    tmp = similar(u)
    mul!(tmp, Diagonal(h), u)
    return v' * tmp
end
