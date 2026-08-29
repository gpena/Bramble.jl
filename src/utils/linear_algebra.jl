@noinline function _throw_dot_dim_error(lu::Integer, lv::Integer, lw::Integer)
	throw(DimensionMismatch("Vectors must have matching lengths, but got lengths ($lu, $lv, $lw)."))
end

"""
	PARALLEL_FOR_MIN

Below this many indices, [`_parallel_for!`](@ref) runs serially.

`Threads.@threads` allocates its tasks on every call, and on a short loop that
setup costs more than the work it distributes. Measured on 4 threads with the
cell-average kernel, serial against threaded, in microseconds:

    indices     16      64     256    1024    4096   16384
    serial     3.8    15.4    61.5   246.4   985.0  3941.2
    threaded  16.0    25.3    44.3   133.6   436.3  1597.3

so threading starts paying somewhere between 64 and 256. Running short loops
serially also makes them allocation free, which matters when the caller is a
time-stepping loop invoking this once per step.
"""
const PARALLEL_FOR_MIN = 256

"""
	$(SIGNATURES)

Applies `f` across `idxs`, writing into `v` in place, using static thread
scheduling once there is enough work to be worth it and running serially
otherwise. See [`PARALLEL_FOR_MIN`](@ref).

# Arguments
- `v`: Array to be modified in-place
- `idxs`: Iterable of indices to process
- `f`: Function that takes an index and returns the value to be stored at that index
"""
function _parallel_for!(v, idxs, f)
	# A single thread, or too little work to amortise spawning tasks: the serial
	# path is both faster and allocation free.
	if Threads.nthreads() == 1 || length(idxs) < PARALLEL_FOR_MIN
		return _serial_for!(v, idxs, f)
	end
	return _threaded_for!(v, idxs, f)
end

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

`_parallel_for!` writes one value per index into one array. When the kernel
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
`mats`, threading on the same terms as [`_parallel_for!`](@ref).

# Arguments
- `mats`: Tuple of arrays to be modified in-place, one per component
- `idxs`: Iterable of indices to process
- `g`: Function taking an index and returning a tuple of values, one per array
"""
function _scatter_for!(mats::Tuple, idxs, g)
	if Threads.nthreads() == 1 || length(idxs) < PARALLEL_FOR_MIN
		@inbounds for idx in idxs
			_write_components!(mats, g(idx), idx)
		end
		return nothing
	end
	return _threaded_scatter_for!(mats, idxs, g)
end

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
	(length(u) == length(v) == length(w)) || _throw_dot_dim_error(length(u), length(v), length(w))
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
function _inner_product(u::AbstractMatrix, h::AbstractVector, v::AbstractMatrix)
	tmp = similar(u)
	mul!(tmp, Diagonal(h), u)
	return v' * tmp
end
