@noinline function _throw_dot_dim_error(lu::Integer, lv::Integer, lw::Integer)
	throw(DimensionMismatch("Vectors must have matching lengths, but got lengths ($lu, $lv, $lw)."))
end

"""
	$(SIGNATURES)

Parallel implementation of a for-loop that modifies array `v` in-place using static thread scheduling.

# Arguments
- `v`: Array to be modified in-place
- `idxs`: Iterable of indices to process
- `f`: Function that takes an index and returns the value to be stored at that index
"""
function _parallel_for!(v, idxs, f)
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

##################################################################################
#                                                                                #
#   Helper functions to calculate inner products in discrete spaces               #
#                                                                                #
##################################################################################

"""
	$(SIGNATURES)

Computes the weighted element-wise dot product of three vectors:
``\\sum_{i} u_i \\cdot v_i \\cdot w_i``

Uses SIMD and fused multiply-add operations for maximal performance without allocations.
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
	return transpose(v) * tmp
end
