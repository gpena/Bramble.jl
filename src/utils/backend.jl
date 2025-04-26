"""
	Backend

A structure containing types and configuration for backend linear algebra objects.

This allows specifying the desired concrete types for vectors and matrices
(e.g., dense `Vector`, sparse `SparseVector`, different element types like
`Float32` or `Float64`).

# Fields

  - `vector_type`: The concrete vector type (e.g., `Vector{Float64}`).
  - `matrix_type`: The concrete matrix type (e.g., `Matrix{Float64}`, `SparseMatrixCSC{Float64, Int}`).
"""
struct Backend{VType<:AbstractVector,MType<:AbstractMatrix} <: BrambleType
end

@inline vector_type(b::Backend{VType,MType}) where {VType,MType} = VType
@inline matrix_type(b::Backend{VType,MType}) where {VType,MType} = MType

"""
	Backend(;
		vector_type::Type{<:AbstractVector} = Vector{Float64},
		matrix_type::Type{<:AbstractMatrix} = SparseMatrixCSC{Float64,Int},
	)

Create a linear algebra `Backend` using keyword arguments.

Defaults to standard dense `Float64` vectors and `SparseMatrixCSC` matrices , ensuring the provided
`vector_type` and `matrix_type` are constructible with the intended dimensions
via standard patterns like `T(undef, dims...)` or `T(dims...)`.

# Examples

```julia
# Default backend (Dense-Sparse Float64)
backend_dense_sparse = Backend()

# Sparse Float64 backend
using SparseArrays
backend_sparse_sparse = Backend(vector_type = SparseVector{Float64,Int},
								matrix_type = SparseMatrixCSC{Float64,Int})

# Dense-Sparse Float32 backend
backend_dense32 = Backend(vector_type = Vector{Float32},
						  matrix_type = SparseMatrixCSC{Float32,Int})
```
"""
function Backend(; vector_type = Vector{Float64}, matrix_type = SparseMatrixCSC{Float64,Int})
	return Backend{vector_type,matrix_type}()
end

"""
	create_vector(b::Backend, n::Integer)

Create a vector of the type specified in `b.vector_type` with length `n`.
"""
function vector(b::Backend{Vector{T},MType}, n::Integer) where {MType,T}
	return Vector{T}(undef, n)
end

function vector(b::Backend{VecType,MType}, n::Integer) where {VecType,MType}
	try
		# Attempt standard constructor for dense-like arrays (e.g., Vector)
		# Requires VecType to have a method accepting undef and size.
		return VecType(undef, n)
	catch e_undef
		# Fallback for types like SparseVector which take only size `n`
		# and initialize to empty/zero.
		try
			return VecType(n)
		catch e_size
			# Combine error messages for clarity if both fail
			error("Cannot create vector of type $VecType with size $n. Tried T(undef, n) (failed: $e_undef) and T(n) (failed: $e_size).")
		end
	end
end

"""
	create_matrix(b::Backend, m::Integer, n::Integer) 

Create a matrix of the type specified in `b.matrix_type` with dimensions `m` x `n`.
"""
function matrix(b::Backend{VType,MatType}, m::Integer, n::Integer) where {VType,MatType}
	try
		# Attempt standard constructor for dense-like arrays (e.g., Matrix)
		# Requires MatType to have a method accepting undef and dims.
		return MatType(undef, m, n)
	catch e_undef
		# Fallback for types like SparseMatrixCSC which take only dims `m, n`
		# and initialize to empty/zero.
		try
			return MatType(m, n)
		catch e_size
			# Combine error messages for clarity if both fail
			error("Cannot create matrix of type $MatType with size ($m, $n). Tried T(undef, m, n) (failed: $e_undef) and T(m, n) (failed: $e_size).")
		end
	end
end

@inline eltype(b::Backend{VecType,MatType}) where {VecType,MatType} = eltype(Backend{VecType,MatType})

function eltype(::Type{Backend{VecType,MatType}}) where {VecType,MatType}
	@assert eltype(VecType) == eltype(MatType)
	return eltype(VecType)
end
