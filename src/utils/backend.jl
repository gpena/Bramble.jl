"""
	$(TYPEDEF)

A singleton structure containing type-level configuration for backend linear algebra objects.
This structure has no fields, only type parameters `VT` and `MT`.

This allows specifying the desired concrete types for vectors and matrices
(e.g., dense `Vector`, sparse `SparseVector`, different element types like
`Float32`, `Float64`, or custom GPU arrays).
"""
struct Backend{VT<:AbstractVector,MT<:AbstractMatrix} end

"""
	$(SIGNATURES)

Returns the vector type (`VT`) associated with the given [`Backend`](@ref) instance or type.
"""
@inline vector_type(::Backend{VT,MT}) where {VT,MT} = VT
@inline vector_type(::Type{<:Backend{VT,MT}}) where {VT,MT} = VT

"""
	$(SIGNATURES)

Returns the matrix type (`MT`) associated with the given [`Backend`](@ref) instance or type.
"""
@inline matrix_type(::Backend{VT,MT}) where {VT,MT} = MT
@inline matrix_type(::Type{<:Backend{VT,MT}}) where {VT,MT} = MT

"""
	$(SIGNATURES)

Creates a linear algebra [`Backend`](@ref) using keyword arguments.

Defaults to standard dense `Float64` vectors and `SparseMatrixCSC` matrices:
- `vector_type = Vector{Float64}`
- `matrix_type = SparseMatrixCSC{Float64,Int}`

# Examples

```jldoctest
julia> dense_sparse = backend() # Default backend (Dense-Sparse Float64)

julia> using SparseArrays;
       SVec{T} = SparseVector{T,Int};
       SMat{T} = SparseMatrixCSC{T,Int};
       T64 = Float64;

julia> sparse_sparse = backend(vector_type = SVec{T64}, matrix_type = SMat{T64})

julia> T32 = Float32;
       dense32 = backend(vector_type = Vector{T32}, matrix_type = SMat{T32})
```
"""
@inline backend(; vector_type = Vector{Float64}, matrix_type = SparseMatrixCSC{Float64,Int}) = Backend{vector_type,matrix_type}()

"""
	$(SIGNATURES)

Returns a tuple with the backend associated types:
1. Element type of `VT`
2. Vector type `VT`
3. Matrix type `MT`
4. Concrete backend type `Backend{VT,MT}`
"""
@inline backend_types(backend::Backend{VT,MT}) where {VT,MT} = eltype(VT), VT, MT, typeof(backend)
@inline backend_types(::Type{<:Backend{VT,MT}}) where {VT,MT} = eltype(VT), VT, MT, Backend{VT,MT}

@noinline function _throw_vector_error(VT, n, e_undef, e_size)
	error("Cannot create vector of type $VT with size $n. Tried T(undef, n) (failed: $e_undef) and T(n) (failed: $e_size).")
end

@noinline function _throw_matrix_error(MT, n, m, e_undef, e_size)
	error("Cannot create matrix of type $MT with size ($n, $m). Tried T(undef, n, m) (failed: $e_undef) and T(n, m) (failed: $e_size).")
end

"""
	$(SIGNATURES)

Creates a vector of type `VT` associated with the given [`Backend`](@ref) instance with length `n`.
"""
function vector(::Backend{VT,MT}, n::Integer) where {VT,MT}
	try
		return VT(undef, n)
	catch e_undef
		try
			return VT(n)
		catch e_size
			_throw_vector_error(VT, n, e_undef, e_size)
		end
	end
end

# Specialized zero-overhead methods for dense (CPU/GPU) and sparse types
@inline vector(::Backend{VT,MT}, n::Integer) where {MT,T,VT<:DenseVector{T}} = VT(undef, n)
@inline vector(::Backend{VT,MT}, n::Integer) where {MT,T,Ti,VT<:SparseVector{T,Ti}} = spzeros(T, Ti, n)

"""
	$(SIGNATURES)

Creates a matrix of type `MT` associated with the given [`Backend`](@ref) instance with dimensions `n` × `m`.
"""
function matrix(backend::Backend{VT,MT}, n::Integer, m::Integer) where {VT,MT}
	try
		return MT(undef, n, m)
	catch e_undef
		try
			return MT(n, m)
		catch e_size
			_throw_matrix_error(MT, n, m, e_undef, e_size)
		end
	end
end

# Specialized zero-overhead methods for dense (CPU/GPU) and sparse types
@inline matrix(::Backend{VT,MT}, n::Integer, m::Integer) where {VT,T,MT<:DenseMatrix{T}} = MT(undef, n, m)
@inline matrix(::Backend{VT,MT}, n::Integer, m::Integer) where {VT,T,Ti,MT<:SparseMatrixCSC{T,Ti}} = spzeros(T, Ti, n, m)

"""
	$(SIGNATURES)

Constructs an `n` × `n` identity matrix associated with the given [`Backend`](@ref) instance.
"""
@inline backend_eye(backend::Backend, n::Integer) = _backend_eye(matrix_type(backend), n)
@inline _backend_eye(::Type{<:SparseMatrixCSC{T,Ti}}, n::Integer) where {T,Ti} = spdiagm(n, n, 0 => fill(one(T), n))
@inline _backend_eye(::Type{<:Matrix{T}}, n::Integer) where T = Matrix{T}(I, n, n)

"""
	$(SIGNATURES)

Constructs an `n` × `n` zero matrix associated with the given [`Backend`](@ref) instance.
"""
@inline backend_zeros(backend::Backend, n::Integer) = _backend_zeros(matrix_type(backend), n)
@inline _backend_zeros(::Type{<:SparseMatrixCSC{T,Ti}}, n::Integer) where {T,Ti} = spzeros(T, Ti, n, n)
@inline _backend_zeros(::Type{MT}, n::Integer) where {T,MT<:AbstractMatrix{T}} = fill!(MT(undef, n, n), zero(T))

"""
	$(SIGNATURES)

Returns the coordinate/element type of the vector type (`VT`) used in the given [`Backend`](@ref).
"""
@inline Base.eltype(backend::Backend{VT,MT}) where {VT,MT} = eltype(typeof(backend))
@inline Base.eltype(::Type{<:Backend{VT,MT}}) where {VT,MT} = eltype(VT)

function Base.show(io::IO, be::Backend{VT,MT}) where {VT,MT}
	if get(io, :compact, false)
		print(io, "Backend{$(eltype(be))}")
	else
		print(io, "Backend(vector = $VT, matrix = $MT)")
	end
end