
# some helper functions to calculate the inner products in discrete spaces
@inline function _dot(u::Vector{T}, v::Vector{T}, w::Vector{T}) where T
	s = zero(T)

	@simd for i in eachindex(u, v, w)
		s += u[i] * v[i] * w[i]
	end

	return s
end

@inline _inner_product(u::Vector{T}, h::Vector{T}, v::Vector{T}) where T = _dot(u, h, v)

@inline function _inner_product_add!(z::Vector{T}, u::Vector{T}, v::Vector{T}, w::Vector{T}) where T
	@simd for i in eachindex(z, u, v, w)
		z[i] += u[i] * v[i] * w[i]
	end
end

@inline function _inner_product(u::Vector{T}, h::Vector{T}, A::SparseMatrixCSC{T,Int}) where T
	z = zeros(T, size(u))
	#@show "1"
	_inner_product_add!(z, u, h, A)

	return z
end

@inline function _inner_product(A::SparseMatrixCSC{T,Int}, h::Vector{T}, v::Vector{T}) where T
	z = transpose(zeros(T, size(v)))
	_inner_product_add!(z, A, h, v)

	return z
end

@inline _inner_product(u, h, v) = transpose(v) * Diagonal(h) * u

@inline function _inner_product_add!(z, u::Vector{T}, h::Vector{T}, A::SparseMatrixCSC{T,Int}) where T
	#z = transpose(A) * Diagonal(h) * u
	#@show "2"

	@simd for j in eachindex(h, u)
		for idx in A.colptr[j]:(A.colptr[j + 1] - 1)
			i = A.rowval[idx]
			val = A.nzval[idx]

			z[j] += val * h[i] * u[i]
		end
	end
end


@inline function _inner_product_add!(z, A::SparseMatrixCSC{T,Int}, h::Vector{T}, v::Vector{T}) where T
	#z = transpose(v) * Diagonal(h) * A
	#@show "3"

	@simd for j in eachindex(h, v)
		for idx in A.colptr[j]:(A.colptr[j + 1] - 1)
			i = A.rowval[idx]
			val = A.nzval[idx]

			z[j] += h[i] * v[i] * val
		end
	end
end

@inline function _inner_product_add!(Z::SparseMatrixCSC{T,Int}, U::SparseMatrixCSC{T,Int}, h::Vector{T}, V::SparseMatrixCSC{T,Int}) where T
	#@show "4"
	#Z .+= transpose(V) * Diagonal(h) * U
	mul!(Z, transpose(V), Diagonal(h) * U)
end


#=
function __sp_add_matrix_transpose_times_vector(y, A, x)
	@finch begin
		#y .= 0
		for j ∈ _, i ∈ _
			y[i] += A[i, j] * x[j]
		end
	end
	return nothing
end

function __sp_matrix_transpose_times_vector(y, A, x)
	@finch begin
		y .= 0
		for i ∈ _, j ∈ _
			y[i] += A[j, i] * x[j]
		end
	end
	return nothing
end

function __sp_tuple_matrix_transpose_times_vector(y, A1, A2, x1, x2, d1, d2)
	@finch begin
		#y .= 0
		for i ∈ _, j ∈ _
			y[i] += A1[j, i] * x1[j] * d1[j] + A2[j, i] * x2[j] * d2[j]
		end
	end
	return nothing
end
=#