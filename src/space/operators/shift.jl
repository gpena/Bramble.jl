"""
	⊗(A, B)

Kronecker product operator (alias for `kron`).

Computes the Kronecker product (tensor product) of matrices `A` and `B`.
This operator is used extensively in constructing multidimensional shift operators.

# Example

```julia
I₂ = Eye(2)
I₃ = Eye(3)
result = I₂ ⊗ I₃  # 6×6 identity matrix
```

See also: [`shift`](@ref)
"""
@inline ⊗(A, B) = kron(A, B)

"""
	_Eye(::Type{MType}, npts, ::Val{i})

Internal helper to create identity or shifted diagonal matrices.

Creates either an identity matrix (`i=0`) or a matrix with ones on the `i`-th diagonal.
The `Val{i}` allows compile-time specialization for the diagonal offset.

# Arguments

  - `MType`: Matrix type to construct
  - `npts::Int`: Size of the square matrix
  - `::Val{i}`: Diagonal offset (0 = main diagonal, 1 = superdiagonal, -1 = subdiagonal)

# Returns

  - For `i=0`: Identity matrix of size `npts × npts`
  - For `i≠0`: Matrix with ones on the `i`-th diagonal, zeros elsewhere

# Example

```julia
_Eye(Matrix{Float64}, 5, Val(0))   # 5×5 identity
_Eye(Matrix{Float64}, 5, Val(1))   # 5×5 with ones on superdiagonal
_Eye(Matrix{Float64}, 5, Val(-1))  # 5×5 with ones on subdiagonal
```

See also: [`shift`](@ref)
"""
@inline _Eye(::Type{MType}, npts::Int, ::Val{0}) where {MType<:AbstractMatrix} = Eye{eltype(MType)}(npts)
@inline _Eye(::Type{MType}, npts::Int, ::Val{i}) where {i,MType<:AbstractMatrix} = spdiagm(i => Ones(eltype(MType), npts - abs(i)))

@inline _Eye(::Type{MType}, npts::Int, ::Val{0}) where MType<:SparseMatrixCSC = Eye{eltype(MType)}(npts)
@inline _Eye(::Type{MType}, npts::Int, ::Val{i}) where {i,MType<:SparseMatrixCSC} = spdiagm(i => Ones(eltype(MType), npts - abs(i)))

@inline @inbounds function _recursive_shift(Ωₕ::AbstractMeshType, ::Val{1}, ::Val{DIFF_DIM}, ::Val{i}) where {DIFF_DIM,i}
	dims = npoints(Ωₕ, Tuple)
	MType = matrix_type(backend(Ωₕ))

	if DIFF_DIM == 1
		return _Eye(MType, dims[1], Val(i))
	else
		return Eye{eltype(MType)}(dims[1])
	end
end

@inline @inbounds function _recursive_shift(Ωₕ::AbstractMeshType, ::Val{D}, ::Val{DIFF_DIM}, ::Val{i}) where {D,DIFF_DIM,i}
	dims = npoints(Ωₕ, Tuple)
	MType = matrix_type(backend(Ωₕ))

	# Determine the operator for the current (outermost) dimension D.
	if DIFF_DIM == D
		op_current = _Eye(MType, dims[D], Val(i))
	else
		op_current = Eye{eltype(MType)}(dims[D])
	end

	# Recurse on the inner dimensions (from D-1 down to 1).
	op_lower_dims = _recursive_shift(Ωₕ, Val(D - 1), Val(DIFF_DIM), Val(i))

	# Combine them: M_D ⊗ (M_{D-1} ⊗ ...)
	return op_current ⊗ op_lower_dims
end

"""
	shift(Ωₕ, ::Val{SHIFT_DIM}, ::Val{i})

Creates a shift operator on a D-dimensional mesh.

  - `SHIFT_DIM`: The dimension along which to apply the shift (e.g., 1 for x, 2 for y).
  - `i`: The shift amount (e.g., -1 for a backward shift).
"""
function shift(Ωₕ::AbstractMeshType, ::Val{SHIFT_DIM}, ::Val{i}) where {SHIFT_DIM,i}
	if i == 0
		return Eye{eltype(eltype(Ωₕ))}(npoints(Ωₕ))
	end

	return _recursive_shift(Ωₕ, Val(dim(Ωₕ)), Val(SHIFT_DIM), Val(i))
end

#=
This code shall remain to clarify the definition of each shift function
@inline shiftₓ(Ωₕ::AbstractMeshType, ::Val{1}, ::Val{0}) = _Eye(eltype(Ωₕ), npoints(Ωₕ, Tuple)[1], Val(0))
@inline shiftₓ(Ωₕ::AbstractMeshType, ::Val{D}, ::Val{0}) where D = Eye(npoints(Ωₕ))
@inline shiftₓ(Ωₕ::AbstractMeshType, ::Val{1}, ::Val{i}) where i = _Eye(eltype(Ωₕ), npoints(Ωₕ, Tuple)[1], Val(i))
@inline shiftₓ(Ωₕ::AbstractMeshType, ::Val{D}, ::Val{i}) where {D,i} = Eye(npoints(Ωₕ(D))) ⊗ shiftₓ(Ωₕ, Val(D - 1), Val(i))

@inline shiftᵧ(Ωₕ::AbstractMeshType, ::Val{2}, ::Val{0}) = Eye(npoints(Ωₕ))
@inline shiftᵧ(Ωₕ::AbstractMeshType, ::Val{3}, ::Val{0}) = Eye(npoints(Ωₕ))
@inline shiftᵧ(Ωₕ::AbstractMeshType, ::Val{1}, ::Val{i}) where i = _Eye(eltype(Ωₕ), npoints(Ωₕ(2)), Val(i))
@inline shiftᵧ(Ωₕ::AbstractMeshType, ::Val{2}, ::Val{i}) where i = shiftᵧ(Ωₕ, Val(1), Val(i)) ⊗ Eye(npoints(Ωₕ(1)))
@inline shiftᵧ(Ωₕ::AbstractMeshType, ::Val{3}, ::Val{i}) where i = Eye(npoints(Ωₕ(3))) ⊗ shiftᵧ(Ωₕ, Val(2), Val(i))

@inline shift₂(Ωₕ::AbstractMeshType, ::Val{3}, ::Val{0}) = Eye(npoints(Ωₕ))
@inline shift₂(Ωₕ::AbstractMeshType, ::Val{3}, ::Val{i}) where i = _Eye(eltype(Ωₕ), npoints(Ωₕ(3)), Val(i)) ⊗ Eye(prod(npoints(Ωₕ, Tuple)[1:2]))
=#