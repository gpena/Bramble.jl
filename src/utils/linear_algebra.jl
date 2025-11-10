"""
	$(SIGNATURES)

Parallel implementation of a for loop that modifies array `v` in-place.

# Arguments

  - `v`: Array to be modified in-place
  - `idxs`: Iterable of indices to process
  - `f`: Function that takes an index and returns the value to be stored at that index
"""
function _parallel_for!(v, idxs, f)
	# This loop iterates over the indices in `idxs` across multiple threads.
	# For each index `idx`, it computes `f(idx)` and assigns the result to `v[idx]`.
	# The `@threads` macro from Julia's standard library `Threads` is used to enable multi-threading.
	Threads.@threads for idx in idxs
		v[idx] = f(idx)
	end
	# The function returns `nothing` as it modifies `v` in-place.
	return
end

"""
	$(SIGNATURES)

Performs a serial (non-parallel) iteration over the specified indices, applying a function `f` to modify vector `v` in-place.

# Arguments

  - `v`: Vector to be modified in-place
  - `idxs`: Indices to iterate over
  - `f`: Function to be applied at each index
"""
function _serial_for!(v, idxs, f)
	# This is a standard, single-threaded for loop that serves as the non-parallel
	# counterpart to `_parallel_for!`.
	for idx in idxs
		v[idx] = f(idx)
	end
	# The function returns `nothing`.
	return
end

##################################################################################
#                                                                                #
#   some helper functions to calculate the inner products in discrete spaces     #
#                                                                                #
##################################################################################
"""
	$(SIGNATURES)

Computes the element-wise product of three vectors `u`, `v`, and `w` and sums the results.
This is equivalent to the mathematical operation `∑ᵢ uᵢ * vᵢ * wᵢ`. It is used as an
optimized implementation for the weighted inner product of vectors.

The `@fastmath` macro allows for aggressive floating-point optimizations, and `@simd`
instructs the compiler to vectorize the loop if possible.
"""
@inline function _dot(u::VT, v::VT, w::VT) where {T,VT<:AbstractVector{T}}
	s = zero(T)

	@fastmath @simd for i in eachindex(u, v, w)
		@inbounds s += u[i] * v[i] * w[i]
	end

	return s
end

"""
	$(SIGNATURES)

Computes the inner product between `u` and `v` with respect to the vector weights `h`. Variables `u` and `v` can be either vectors or matrices.

There is a specialized version when `u` and `v` are vectors.
"""
# This is the generic implementation for the weighted inner product, often written as `⟨u, v⟩_h`.
# It computes `transpose(v) * (Diagonal(h) * u)`, which works for both vectors and matrices `u` and `v`.
@inline _inner_product(u, h, v) = transpose(v) * (Diagonal(h) * u)

# This is a specialized and highly optimized version for when `u`, `h`, and `v` are all vectors.
# It avoids the overhead of matrix operations by calling the `_dot` function.
@inline _inner_product(u::VT, h::VT, v::VT) where {T,VT<:AbstractVector{T}} = _dot(u, h, v)
