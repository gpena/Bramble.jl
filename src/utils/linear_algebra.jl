"""
	_parallel_for!(v, idxs, f)

Parallel implementation of a for loop that modifies array `v` in-place.

# Arguments

  - `v`: Array to be modified in-place
  - `idxs`: Iterable of indices to process
  - `f`: Function that takes an index and returns the value to be stored at that index
"""
function _parallel_for!(v, idxs, f)
	Threads.@threads for idx in idxs
		v[idx] = f(idx)
	end
	return nothing
end

"""
	_serial_for!(v, idxs, f)

Performs a serial (non-parallel) iteration over the specified indices, applying a function `f` to modify vector `v` in-place.

# Arguments

  - `v`: Vector to be modified in-place
  - `idxs`: Indices to iterate over
  - `f`: Function to be applied at each index
"""
function _serial_for!(v, idxs, f)
	for idx in idxs
		v[idx] = f(idx)
	end
	return nothing
end

##################################################################################
# 																				 #
#   some helper functions to calculate the inner products in discrete spaces     #
#																				 #
##################################################################################

"""
	_dot(u::Vector{T}, v::Vector{T}, w::Vector{T})

Compute the triple dot product of three vectors of the same type and length.
"""
@inline function _dot(u::Vector{T}, v::Vector{T}, w::Vector{T}) where T
	s = zero(T)

	@fastmath @simd for i in eachindex(u, v, w)
		@inbounds s += u[i] * v[i] * w[i]
	end

	return s
end

"""
	_inner_product(u::Vector{T}, h::Vector{T}, v::Vector{T})

Compute the inner product of three vectors of the same type `T` using weights `h`.
"""
@inline _inner_product(u::Vector{T}, h::Vector{T}, v::Vector{T}) where T = _dot(u, h, v)

"""
	_inner_product(u, h, v)

Computes the inner product between `u` and `v` with respect to the weight `h`.
This is an internal helper function optimized for performance using the `@inline` directive.

Variables `u` and `v` can be either vectors or matrices. This applies to the general case when `u` or `v` are a matrix.
"""
@inline _inner_product(u, h, v) = transpose(v) * (Diagonal(h) * u)