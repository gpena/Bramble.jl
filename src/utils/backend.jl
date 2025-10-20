"""
	Backend{VType<:AbstractVector,MType<:AbstractMatrix}

A structure containing types and configuration for backend linear algebra objects.

This allows specifying the desired concrete types for vectors and matrices
(e.g., dense `Vector`, sparse `SparseVector`, different element types like
`Float32` or `Float64`).
"""
struct Backend{VType<:AbstractVector,MType<:AbstractMatrix} <: BrambleType end

"""
	vector_type(b::Backend{VType, MType})

Returns the vector type (`VType`) associated with the given `Backend` instance.
This function is useful for extracting the underlying vector type used by a specific backend.
"""
@inline vector_type(::Backend{VType,MType}) where {VType,MType} = VType

"""
	matrix_type(b::Backend{VType, MType})

Returns the matrix type (`MType`) associated with the given `Backend` instance.
This function is useful for extracting the underlying matrix type used by a specific backend.
"""
@inline matrix_type(::Backend{VType,MType}) where {VType,MType} = MType

"""
	Backend(; vector_type::Type{<:AbstractVector}, matrix_type::Type{<:AbstractMatrix})

Create a linear algebra `Backend` using keyword arguments.

Defaults to standard dense `Float64` vectors and `SparseMatrixCSC` matrices, ensuring the provided
`vector_type` and `matrix_type` are constructible with the intended dimensions
via standard patterns like `T(undef, dims...)` or `T(dims...)`.

# Examples

```julia
# Default backend (Dense-Sparse Float64)
dense_sparse = backend()

# Sparse Float64 backend
using SparseArrays
SVec{T} = SparseVector{T,Int}
SMat{T} = SparseMatrixCSC{T,Int}

T64 = Float64
sparse_sparse = backend(vector_type = SVec{T64}, matrix_type = SMat{T64})

# Dense-Sparse Float32 backend
T32 = Float32
dense32 = backend(vector_type = Vector{T32}, matrix_type = SMat{T32})
```
"""
@inline backend(; vector_type = Vector{Float64}, matrix_type = SparseMatrixCSC{Float64,Int}) = Backend{vector_type,matrix_type}()

"""
	backend_types(::Type{Backend{VT,MT}})
	backend_types(::Backend{VT,MT})

Returns a tuple containing:

  - the element type of `VT`,
  - the type `VT`,
  - the type `MT`,
  - and the concrete backend type `Backend{VT,MT}`.

This is useful for extracting type information from either a `Backend` type or instance.
"""
@inline backend_types(::Type{Backend{VT,MT}}) where {VT,MT} = eltype(VT), VT, MT, Backend{VT,MT}
@inline backend_types(::Backend{VT,MT}) where {VT,MT} = eltype(VT), VT, MT, Backend{VT,MT}

"""
	vector(::Backend{VecType,MType}, n::Integer)

Create a vector of the type specified in `VecType` with length `n`.
"""

function vector(::Backend{VecType,MType}, n::Integer) where {VecType,MType}
	try
		return VecType(undef, n)
	catch e_undef
		try
			return VecType(n)
		catch e_size
			# Combine error messages for clarity if both fail
			error("Cannot create vector of type $VecType with size $n. Tried T(undef, n) (failed: $e_undef) and T(n) (failed: $e_size).")
		end
	end
end

@inline vector(::Backend{Vector{T},MType}, n::Integer) where {MType,T} = Vector{T}(undef, n)

"""
	matrix(::Backend{VType,MatType}, n::Integer, m::Integer) 

Create a matrix of the type specified in `MatType` with dimensions `n` x `m`.
"""
function matrix(::Backend{VType,MatType}, n::Integer, m::Integer) where {VType,MatType}
	try
		return MatType(undef, n, m)
	catch e_undef
		try
			return MatType(n, m)
		catch e_size
			# Combine error messages for clarity if both fail
			error("Cannot create matrix of type $MatType with size ($n, $m). Tried T(undef, n, m) (failed: $e_undef) and T(n, m) (failed: $e_size).")
		end
	end
end

"""
	eye(::Type{<:SparseMatrixCSC{T,Int}}, npts)

Constructs an `npts`-by-`npts` sparse identity matrix of element type `T`.
"""
@inline eye(::Type{<:SparseMatrixCSC{T,Int}}, npts) where T = spdiagm(0 => Ones(T, npts))

"""
	zeros(::Type{<:SparseMatrixCSC{T,Int}}, npts)

Constructs an `npts`-by-`npts` sparse matrix of zeros with element type `T`.
"""
@inline zeros(::Type{<:SparseMatrixCSC{T,Int}}, npts) where T = spzeros(T, npts, npts)

"""
	eltype(::Type{<:Backend{VecType,MatType}}) -> DataType

Returns the element type of the vector type (`VecType`) used in the given `Backend` type.
This function allows querying the underlying element type stored in the backend's vector representation.
"""
@inline eltype(::Type{<:Backend{VecType,MatType}}) where {VecType,MatType} = eltype(VecType)