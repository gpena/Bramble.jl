@inline ⊗(A, B) = kron(A, B)

@inline _Eye(::Type{MType}, npts::Int, ::Val{0}) where {MType<:AbstractMatrix} = Eye{eltype(MType)}(npts)
@inline _Eye(::Type{MType}, npts::Int, ::Val{i}) where {i,MType<:AbstractMatrix} = spdiagm(i => Ones(eltype(MType), npts - abs(i)))

@inline _Eye(::Type{MType}, npts::Int, ::Val{0}) where MType<:SparseMatrixCSC = Eye{eltype(MType)}(npts)
@inline _Eye(::Type{MType}, npts::Int, ::Val{i}) where {i,MType<:SparseMatrixCSC} = spdiagm(i => Ones(eltype(MType), npts - abs(i)))

@inline function _recursive_shift(Ωₕ::AbstractMeshType, ::Val{1}, ::Val{DIFF_DIM}, ::Val{i}) where {DIFF_DIM,i}
	dims = npoints(Ωₕ, Tuple)
	MType = matrix_type(backend(Ωₕ))

	if DIFF_DIM == 1
		return _Eye(MType, dims[1], Val(i))
	else
		return Eye{eltype(MType)}(dims[1])
	end
end

@inline function _recursive_shift(Ωₕ::AbstractMeshType, ::Val{D}, ::Val{DIFF_DIM}, ::Val{i}) where {D,DIFF_DIM,i}
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