"""
	struct Backend{VT<:AbstractVector,MT<:AbstractMatrix}

A structure containing types and configuration for backend linear algebra objects. This structure has no fields, only types.

This allows specifying the desired concrete types for vectors and matrices
(e.g., dense `Vector`, sparse `SparseVector`, different element types like
`Float32` or `Float64`).
"""
struct Backend{VT<:AbstractVector,MT<:AbstractMatrix} end

"""
	vector_type(backend::Backend)

Returns the vector type (`VT`) associated with the given [Backend](@ref) instance or type.
This function is useful for extracting the underlying vector type used by a specific backend.
"""
@inline vector_type(::Backend{VT,MT}) where {VT,MT} = VT
@inline vector_type(::Type{<:Backend{VT,MT}}) where {VT,MT} = VT

"""
	matrix_type(backend::Backend)

Returns the matrix type (`MT`) associated with the given [Backend](@ref) instance or type.
This function is useful for extracting the underlying matrix type used by a specific backend.
"""
@inline matrix_type(::Backend{VT,MT}) where {VT,MT} = MT
@inline matrix_type(::Type{<:Backend{VT,MT}}) where {VT,MT} = MT

"""
	$(SIGNATURES)

Create a linear algebra [Backend](@ref) using keyword arguments.

Defaults to standard dense `Float64` vectors and `SparseMatrixCSC` matrices, ensuring the provided
`vector_type` and `matrix_type` are constructible with the intended dimensions
via standard patterns like `T(undef, dims...)` or `T(dims...)`.

# Examples

```jldoctest
julia> dense_sparse = backend() # Default backend (Dense-Sparse Float64)

julia> using SparseArrays;
	   SVec{T} = SparseVector{T,Int};
	   SMat{T} = SparseMatrixCSC{T,Int};
	   T64 = Float64

julia> sparse_sparse = backend(vector_type = SVec{T64}, matrix_type = SMat{T64}) # Sparse-Sparse Float64 backend

julia> T32 = Float32;
	   dense32 = backend(vector_type = Vector{T32}, matrix_type = SMat{T32}) # Dense-Sparse Float32 backend

```
"""
@inline backend(; vector_type = Vector{Float64}, matrix_type = SparseMatrixCSC{Float64,Int}) = Backend{vector_type,matrix_type}()

"""
	backend_types(backend::Backend)

Returns a tuple with the backend associated types:

  - the element type of `VT`,
  - the type `VT`,
  - the type `MT`,
  - the concrete backend type `Backend{VT,MT}`.

This is useful for extracting type information from either a [Backend](@ref) type or instance.
"""
@inline backend_types(backend::Backend{VT,MT}) where {VT,MT} = eltype(VT), VT, MT, typeof(backend)
@inline backend_types(::Type{<:Backend{VT,MT}}) where {VT,MT} = eltype(VT), VT, MT, Backend{VT,MT}

"""
	vector(backend::Backend, n::Integer)

Create a vector of the type `VT` associated with the given [Backend](@ref) instance with length `n`.
"""

function vector(::Backend{VT,MT}, n::Integer) where {VT,MT}
	try
		# Attempt to construct the vector using `VT(undef, n)`.
		return VT(undef, n)
	catch e_undef
		try
			# If the first method fails, try `VT(n)`.
			# Some vector types might not support `undef` initialization
			# and may use a different constructor signature.
			return VT(n)
		catch e_size
			# If both construction attempts fail, throw an error.
			error("Cannot create vector of type $VT with size $n. Tried T(undef, n) (failed: $e_undef) and T(n) (failed: $e_size).")
		end
	end
end

# Specialized method for standard `Vector` types for performance.
# This avoids the overhead of `try-catch` for the most common case.
@inline vector(::Backend{VT,MT}, n::Integer) where {MT,T,VT<:Vector{T}} = Vector{T}(undef, n)

"""
	matrix(backend::Backend, n::Integer, m::Integer)

Create a matrix of the type `MT` associated with the given [Backend](@ref) instance with dimensions `n` x `m`.
"""
function matrix(backend::Backend, n::Integer, m::Integer)
	_, _, MT = backend_types(backend)
	try
		# Attempt to construct the matrix using `MT(undef, n, m)`.
		return MT(undef, n, m)
	catch e_undef
		try
			# Fallback to `MT(n, m)` if the `undef` constructor is not supported.
			return MT(n, m)
		catch e_size
			# If both attempts fail, provide a detailed error message
			# to assist in debugging backend configurations.
			error("Cannot create matrix of type $MT with size ($n, $m). Tried T(undef, n, m) (failed: $e_undef) and T(n, m) (failed: $e_size).")
		end
	end
end

"""
	backend_eye(backend::Backend, n)

Constructs a square `n` x `n` sparse identity matrix associated with the given [Backend](@ref) instance.
"""
@inline backend_eye(backend::Backend, n) = _backend_eye(matrix_type(backend), n)
@inline _backend_eye(::Type{<:SparseMatrixCSC{T,Int}}, n) where T = spdiagm(0 => Ones(T, n))

"""
	backend_zeros(backend::Backend, n)

Constructs a square `n` x `n` sparse matrix of zeros associated with the given [Backend](@ref) instance.
"""
@inline backend_zeros(backend::Backend, n) = _backend_zeros(matrix_type{backend}, n)
@inline _backend_zeros(::Type{<:SparseMatrixCSC{T,Int}}, n) where T = spzeros(T, n, n)

"""
	eltype(backend::Backend)

Returns the element type of the vector type (`VT`) used in the given [Backend](@ref) type or instance.
This function allows querying the underlying element type stored in the backend's vector representation.
"""
@inline eltype(backend::Backend{VT,MT}) where {VT,MT} = eltype(typeof(backend))
@inline eltype(::Type{<:Backend{VT,MT}}) where {VT,MT} = eltype(VT)