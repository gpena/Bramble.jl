##############################################################################
#                                                                            #
# Implementation of the backward difference operators for vector elements    #
#                                                                            #
##############################################################################

#############
#           #
#   diff₋ₓ  #
#           #
#############

"""
	diff₋ₓ(U::VectorElement)

Compute the difference in the `x` direction of element of `U`.
"""
function diff₋ₓ(U::VectorElement)
	V = similar(U)
	dims = npoints(U)

	if dim(U) === 1
		_diff1!(V.values, U.values, dims)
	else
		_differencex!(V.values, U.values, dims)
	end

	return V
end

"""
	_differencex!(out, in, dims::NTuple{D,Int}, ::Val{D})

Compute the difference in the `x` direction of vector `in` and stores the result in `out`
when `in` and `out` are `D`-dimensional vectors.
"""
function _differencex!(out, in, dims::NTuple{D,Int}) where D
	first_dim = prod(dims[1:(D - 1)])
	last_dim = dims[D]

	first_dims = ntuple(i -> dims[i], D - 1)

	@inbounds for m in 1:last_dim
		idx = ((m - 1) * (first_dim) + 1):(m * (first_dim))

		@views _differencex!(out[idx], in[idx], first_dims)
	end

	return out
end

"""
	_differencex!(out, in, dims::NTuple{1,Int})

Compute the difference in the `x` direction of vector `in` and stores the result in `out`.
"""
@inline function _differencex!(out, in, dims::NTuple{1,Int})
	_diff1!(out, in, dims)
	return out
end

#############
# backward difference operators on vectors #
#############
@inline D₋ᵢ(u::VectorElement, ::Val{1}) = D₋ₓ(u)

#############
#           #
#    D₋ₓ    #
#           #
#############
"""
	D_ₓ(U::VectorElement)

Compute the backward difference, in the `x` direction, of `U`:

	``D_ₓ(U)(x_i,\\dots) = \\frac{U(x_i,\\dots) - U(x_{i-1},\\dots)}{h_i}``
"""
function D₋ₓ(U::VectorElement)
	h = Base.Fix1(hspace, mesh(U)(1))
	dims = npoints(U)
	V = similar(U)

	_backward_differencex!(V.values, U.values, h, dims)

	return V
end

@inline function _backward_differencex!(out, in, h, dims::NTuple{1,Int})
	_diff1!(out, in, h, dims)
	return out
end

"""
	_backward_differencex!(out, in, dims::NTuple{D,Int})

Compute the difference in the `x` direction of vector `in` and stores the result in `out`
when `in` and `out` are `D`-dimensional vectors.
"""
@inline function _backward_differencex!(out, in, h, dims::NTuple{D,Int}) where D
	first_dim = prod(dims[1:(D - 1)])
	last_dim = dims[D]

	first_dims = ntuple(i -> dims[i], D - 1)

	@inbounds for m in 1:last_dim
		idx = ((m - 1) * (first_dim) + 1):(m * (first_dim))
		@views _backward_differencex!(out[idx], in[idx], h, first_dims)
	end

	return out
end

#############
#           #
#   diff₋ᵧ  #
#           #
#############
"""
	diff₋ᵧ(U::VectorElement)

Compute the difference in the `y` direction of element of `U`.
"""
function diff₋ᵧ(U::VectorElement)
	D = dim(U)
	V = similar(U)
	dims = npoints(U)

	if D === 1
		@error "no backward difference on y variable in 1D"
	end

	if D === 2
		_diff1!(V.values, U.values, dims)
	else
		_differencey!(V.values, U.values, dims)
	end

	return V
end

@inline function _differencey!(out, in, dims::NTuple{2,Int})
	_diff1!(out, in, dims)
	return out
end

@inline function _differencey!(out, in, dims::NTuple{3,Int})
	N, M = dims[1:2]
	O = dims[3]

	@inbounds for lev ∈ 1:O
		idx = ((lev - 1) * N * M + 1):(lev * N * M)
		@views _differencey!(out[idx], in[idx], (N, M))
	end

	return out
end

#############
#           #
#    D₋ᵧ    #֪
#           #
#############
@inline D₋ᵢ(u::VectorElement, ::Val{2}) = D₋ᵧ(u)

"""
	D_ᵧ(U::VectorElement)

Compute the backward difference in the `y` direction of element of `U`:

	``D_ᵧ(U)(\\dots,y_j,\\dots) = \\frac{U(\\dots,y_j,\\dots) - U(\\dots,y_{j-1},\\dots)}{k_j}``
"""
function D₋ᵧ(U::VectorElement)
	h = Base.Fix1(hspace, mesh(U)(2))
	dims = npoints(U)
	V = similar(U)

	_backward_differencey!(V.values, U.values, h, dims)

	return V
end

@inline function _backward_differencey!(out, in, k, dims::NTuple{2,Int})
	_diff1!(out, in, k, dims)
	return out
end

@inline function _backward_differencey!(out, in, k, dims::NTuple{3,Int})
	N, M = dims[1:2]
	O = dims[3]

	@inbounds for lev in 1:O
		idx = ((lev - 1) * N * M + 1):(lev * N * M)
		@views _backward_differencey!(out[idx], in[idx], k, (N, M))
	end

	return out
end

#############
#           #
#   diff₋₂  #
#           #
#############
"""
	diff₋₂(U::VectorElement)

Compute the difference in the `z` direction of element of `U`.
"""
function diff₋₂(U::VectorElement)
	D = dim(U)
	V = similar(U)
	dims = npoints(U)

	if D === 1 || D === 2
		@error "No backward difference in variable z in 1D ou 2D"
	end

	_diff1!(V.values, U.values, dims)

	return V
end

#############
#           #
#    D₋₂    #
#           #
#############

@inline D₋ᵢ(u::VectorElement, ::Val{3}) = D₋₂(u)

"""
	D_₂(U::VectorElement)

Compute the backward difference in the `z` direction of element of `U`:

	``D_₂(U)(\\dots,z_m) = \\frac{U(\\dots,z_m) - U(\\dots,z_{m-1}}{l_m}``
"""
function D₋₂(U::VectorElement)
	D = dim(U)
	h = Base.Fix1(hspace, mesh(U)(3))
	dims = npoints(U)
	V = similar(U)

	if D === 1 || D === 2
		@error "No backward difference in variable z in 1D ou 2D"
	end

	_diff1!(V.values, U.values, h, dims)

	return V
end

## backward difference helper functions
"""
	_diff1!(out, in, dims::NTuple{1,Int})

Calculate the difference of a 1D vector `in` with coefficients stored in `out`.
"""
@inline function _diff1!(out, in, dims::NTuple{1,Int})
	@assert length(out) == length(in) == dims[1]
	out[1] = in[1]

	@simd for i in 2:dims[1]
		out[i] = in[i] - in[i - 1]
	end
	return out
end

"""
	_diff1!(out, in, h::AbstractVector, dims::NTuple{1,Int})

Calculate the backward difference, wrt `x`, of a 1D vector `in` with coefficients stored in `out`
using the given step vector `h`.
"""
@inline function _diff1!(out, in, h::AbstractVector, dims::NTuple{1,Int})
	@assert length(out) == length(in) == dims[1]
	out[1] = in[1] / h[2]

	@simd for i in 2:dims[1]
		out[i] = (in[i] - in[i - 1]) / h[i]
	end
	return out
end

@inline function _diff1!(out, in, h::F, dims::NTuple{1,Int}) where {F<:Function}
	@assert length(out) == length(in) == dims[1]
	out[1] = in[1] / h(2)

	Threads.@threads for i in 2:dims[1]
		out[i] = (in[i] - in[i - 1]) / h(i)
	end
	return out
end

@inline function _diff1!(out, in, dims::NTuple{D,Int}) where D
	@assert length(out) == length(in)
	_diff2!(out, in, dims)
	_diff3!(out, in, dims)
	return out
end

@inline function _diff1!(out, in, h::F, dims::NTuple{D,Int}) where {D,F}
	@assert length(out) == length(in)
	_diff2!(out, in, h, dims)
	_diff3!(out, in, h, dims)
	return out
end

@inline function _diff2!(out, in, last::Int, h::T) where {T<:AbstractFloat}
	@simd for i in 1:last
		out[i] = in[i] / h
	end
	return out
end

@inline function _diff2!(out, in, dims::NTuple{D,Int}) where D
	_diff2!(out, in, prod(dims[2:D]), one(eltype(in)))
	return out
end

@inline function _diff2!(out, in, h::AbstractVector, dims::NTuple{D,Int}) where D
	_diff2!(out, in, prod(dims[2:D]), h[2])
	return out
end

@inline function _diff2!(out, in, h::F, dims::NTuple{D,Int}) where {D,F<:Function}
	_diff2!(out, in, prod(dims[2:D]), h(2))
	return out
end

@inline function _diff3!(out, in, c::Int, first::Int, h::T) where {T<:AbstractFloat}
	@simd for l in ((c - 1) * first + 1):(c * first)
		out[l] = (in[l] - in[l - first]) / h
	end
	return out
end

@inline function _diff3!(out, in, dims::NTuple{D,Int}) where D
	first = prod(dims[1:(D - 1)])

	for c ∈ 2:dims[D]
		_diff3!(out, in, c, first, one(eltype(in)))
	end
	return out
end

@inline function _diff3!(out, in, h::AbstractVector, dims::NTuple{D,Int}) where D
	first = prod(dims[1:(D - 1)])

	for c ∈ 2:dims[D]
		_diff3!(out, in, c, first, h[c])
	end
	return out
end

@inline function _diff3!(out, in, h::F, dims::NTuple{D,Int}) where {D,F<:Function}
	first = prod(dims[1:(D - 1)])

	for c in 2:dims[D]
		_diff3!(out, in, c, first, h(c))
	end
	return out
end

##############################################################################
#                                                                            #
# Implementation of the backward difference operators for matrix elements    #
#                                                                            #
##############################################################################

###########################
#    helper functions     #
###########################

"""
	⊗(A, B)

Return the Kronecker product of `A` and `B`.

**Inputs:**

  - `A`: a matrix
  - `B`: a matrix

**Output:**

  - a matrix
"""
@inline ⊗(A, B) = kron(A, B)

#@inline __ones(::Type{T}, npts) where T = FillArrays.Ones(T, npts)

#@inline eye(npts) = Eye(npts)
#@inline eye(::Type{T}, npts) where T = eye(T, npts, Val(0))
"""
	_Eye(::Type{T}, npts::Int, i::Int) where T

Return a sparse matrix with ones on the `i`th diagonal.

**Inputs:**

  - `::Type{T}`: the type of the elements in the matrix
  - `npts`: the size of the matrix
  - `i`: the index of the diagonal

**Output:**

  - a sparse matrix
"""
@inline _Eye(::Type{T}, npts::Int, i::Int) where {T} = spdiagm(i => FillArrays.Ones(T, npts - abs(i)))

# shift functions
# a apagar estas duas funcoes
@inline shift(M::MeshType, ::Val{i}) where i = _Eye(eltype(typeof(M)), npoints(M)[1], i)
@inline shift(M::MeshType, ::Val{0}) = Eye(npoints(M)[1])

"""
	shiftₓ(M::MeshType, ::Val{1}, ::Val{0})

Returns the identity matrix for the first dimension of the mesh.

**Inputs:**

  - `M`: a mesh
  - `::Val{1}`: value of type `Val{1}` indicating the topological dimension of the mesh
  - `::Val{0}`: value of type `Val{0}` indicating that no shift is to be applied

**Output:**

  - a sparse matrix
"""
@inline shiftₓ(M::MeshType, ::Val{1}, ::Val{0}) = _Eye(eltype(M), npoints(M)[1], 0)

"""
	shiftₓ(M::MeshType, ::Val{D}, ::Val{0})

Returns the identity matrix for the first dimension of the mesh.

**Inputs:**

  - `M`: a mesh
  - `::Val{D}`: value of type `Val{D}` indicating the topological dimension of the mesh
  - `::Val{0}`: value of type `Val{0}` indicating that no shift is to be applied

**Output:**

  - a sparse matrix
"""
@inline shiftₓ(M::MeshType, ::Val{D}, ::Val{0}) where D = Eye(ndofs(M))

"""
	shiftₓ(M::MeshType, ::Val{1}, ::Val{i})

Returns the sparse matrix that, when applied to a vector, shifts the components by `i`.

**Inputs:**

  - `M`: a mesh
  - `::Val{1}`: value of type `Val{1}` indicating the topological dimension of the mesh
  - `::Val{i}`: value of type `Val{i}` indicating the shift

**Output:**

  - a sparse matrix
"""
@inline shiftₓ(M::MeshType, ::Val{1}, ::Val{i}) where i = shift(M, Val(i))
@inline shiftₓ(M::MeshType, ::Val{D}, ::Val{i}) where {D,i} = Eye(ndofs(M(D))) ⊗ shiftₓ(M, Val(D - 1), Val(i))

@inline shiftᵧ(M::MeshType, ::Val{2}, ::Val{0}) = Eye(ndofs(M))
@inline shiftᵧ(M::MeshType, ::Val{3}, ::Val{0}) = Eye(ndofs(M))
@inline shiftᵧ(M::MeshType, ::Val{1}, ::Val{i}) where i = shift(M(2), Val(i))
@inline shiftᵧ(M::MeshType, ::Val{2}, ::Val{i}) where i = shiftᵧ(M, Val(1), Val(i)) ⊗ Eye(ndofs(M(1)))
@inline shiftᵧ(M::MeshType, ::Val{3}, ::Val{i}) where i = Eye(ndofs(M(3))) ⊗ shiftᵧ(M, Val(2), Val(i))

@inline shift₂(M::MeshType, ::Val{3}, ::Val{0}) = Eye(ndofs(M))
@inline shift₂(M::MeshType, ::Val{3}, ::Val{i}) where i = shift(M(3), Val(i)) ⊗ Eye(prod(npoints(M)[1:2]))

"""
	diff₋ₓ(S::MeshType)

Returns the difference matrix for the mesh `M` in the x-direction.

**Inputs:**

  - `M`: a mesh
"""
@inline diff₋ₓ(M::MeshType) = shiftₓ(M, Val(dim(M)), Val(0)) - shiftₓ(M, Val(dim(M)), Val(-1))

"""
	diff₋ᵧ(M::MeshType)

Returns the difference matrix for the mesh `M` in the y-direction.

**Inputs:**

  - `M`: a mesh
"""
@inline diff₋ᵧ(M::MeshType) = shiftᵧ(M, Val(dim(M)), Val(0)) - shiftᵧ(M, Val(dim(M)), Val(-1))

"""
	diff₋₂(M::MeshType)

Returns the difference matrix for the mesh `M` in the z-direction.

**Inputs:**

  - `M`: a mesh
"""
@inline diff₋₂(M::MeshType) = shift₂(M, Val(dim(M)), Val(0)) - shift₂(M, Val(dim(M)), Val(-1))

@inline diff₋ₓ(S::SpaceType) = diff₋ₓ(mesh(S))
@inline diff₋ᵧ(S::SpaceType) = diff₋ᵧ(mesh(S))
@inline diff₋₂(S::SpaceType) = diff₋₂(mesh(S))

@inline diff₋ₓ(u::MatrixElement) = Elements(space(u), diff₋ₓ(mesh(u))*u.values)
@inline diff₋ᵧ(u::MatrixElement) = Elements(space(u), diff₋ᵧ(mesh(u))*u.values)
@inline diff₋₂(u::MatrixElement) = Elements(space(u), diff₋₂(mesh(u))*u.values)


##########################################################################
#    helper functions for defining the backward difference operators     #
##########################################################################

# implementation of differentiation matrices
function invert_hspace!(v, M::MeshType, component::Int = 1)
	for (i, h) in zip(eachindex(v), hspaceit(M(component)))
		v[i] = inv(h)
	end
end

@inline function repeat_across_dims1!(v, ncopies, dims)
	v_reshaped = reshape(v, dims, ncopies) # Reshape for efficient broadcasting
	@views v_reshaped[:, 2:end] .= v_reshaped[:, 1] # Broadcast the first column
end

@inline repeat_across_dims2!(v, ncopies) = @views v[:, 2:ncopies] .= v[:, 1]

"""
	weights_D₋ₓ!(v, M::MeshType, ::Val{1})

Sets `v` to the inverse of the `x`-component of the grid spacings of `M`.

**Inputs:**

  - `v`: a vector to be set
  - `M`: a mesh
"""
@inline weights_D₋ₓ!(v, M::MeshType, ::Val{1}) = invert_hspace!(v, M)

"""
	weights_D₋ₓ!(v, M::MeshType, ::Val{D})

Sets `v` to the inverse of the `x`-component of the grid spacings of `M`.

**Inputs:**

  - `v`: a vector to be set
  - `M`: a mesh
  - `D`: the dimension of the mesh
"""
@inline function weights_D₋ₓ!(v, M::MeshType, ::Val{D}) where D
	first_dims = prod(npoints(M)[1:(D - 1)])
	weights_D₋ₓ!(view(v, 1:first_dims), M, Val(D - 1))

	repeat_across_dims1!(v, ndofs(M(D)), first_dims)
end

"""
	weights_D₋ᵧ!(v, M::MeshType, ::Val{2})

Sets `v` to the inverse of the `y`-component of the grid spacings of `M`.

**Inputs:**

  - `v`: a vector to be set
  - `M`: a mesh
"""
@inline function weights_D₋ᵧ!(v, M::MeshType, ::Val{2})
	dims = npoints(M)
	_dims = (dims[1], dims[2])

	aux2 = transpose(Base.ReshapedArray(v, _dims, ()))
	t = view(aux2, 1:dims[2], 1)

	invert_hspace!(t, M, 2)

	repeat_across_dims2!(aux2, dims[1])
end

"""
	weights_D₋ᵧ!(v, M::MeshType, ::Val{3})

Sets `v` to the inverse of the `y`-component of the grid spacings of `M`.

**Inputs:**

  - `v`: a vector to be set
  - `M`: a mesh
"""
@inline function weights_D₋ᵧ!(v, M::MeshType, ::Val{3})
	dims = npoints(M)
	first_dims = dims[1] * dims[2]

	weights_D₋ᵧ!(view(v, 1:first_dims), M, Val(2))

	repeat_across_dims1!(v, ndofs(M(3)), first_dims)
end

"""
	weights_D₋₂!(v, M::MeshType, ::Val{3})

Sets `v` to the inverse of the `z`-component of the grid spacings of `M`.

**Inputs:**

  - `v`: a vector to be set
  - `M`: a mesh
"""
@inline function weights_D₋₂!(v, M::MeshType, ::Val{3})
	dims = npoints(M)
	first_dims = dims[1] * dims[2]

	xx = Base.ReshapedArray(v, (first_dims, dims[3]), ())

	t = view(xx, 1, 1:dims[3])
	invert_hspace!(t, M, 3)

	yy = transpose(xx)
	repeat_across_dims2!(yy, first_dims)
end

function _create_D₋ₓ(M::MeshType; diagonal = _create_diagonal(M))
	weights_D₋ₓ!(diagonal.diag, M, Val(dim(M)))
	return diagonal * diff₋ₓ(M)
end

function _create_D₋ᵧ(M::MeshType; diagonal = _create_diagonal(M))
	weights_D₋ᵧ!(diagonal.diag, M, Val(dim(M)))
	return diagonal * diff₋ᵧ(M)
end

function _create_D₋₂(M::MeshType; diagonal = _create_diagonal(M))
	weights_D₋₂!(diagonal.diag, M, Val(dim(M)))
	return diagonal * diff₋₂(M)
end

@inline function _create_backward_diff_matrix(S::SpaceType, ::Val{1}; diagonal = _create_diagonal(S))
	A = _create_D₋ₓ(mesh(S), diagonal = diagonal)
	return Elements(S, A)
end

@inline function _create_backward_diff_matrix(S::SpaceType, ::Val{2}; diagonal = _create_diagonal(S))
	A = _create_D₋ᵧ(mesh(S), diagonal = diagonal)
	return Elements(S, A)
end

@inline function _create_backward_diff_matrix(S::SpaceType, ::Val{3}; diagonal = _create_diagonal(S))
	A = _create_D₋₂(mesh(S), diagonal = diagonal)
	return Elements(S, A)
end

@inline function create_backward_diff_matrices(S::SpaceType; diagonal = _create_diagonal(S))
	return ntuple(i -> _create_backward_diff_matrix(S, Val(i), diagonal = diagonal), dim(S))
end

@inline get_symbol_diff_matrix(::Val{1}) = Symbol("diff_mat_D₋ₓ")
@inline get_symbol_diff_matrix(::Val{2}) = Symbol("diff_mat_D₋ᵧ")
@inline get_symbol_diff_matrix(::Val{3}) = Symbol("diff_mat_D₋₂")

@inline D₋ₓ(S::SpaceType) = getcache(S, get_symbol_diff_matrix(Val(1)))
@inline D₋ᵧ(S::SpaceType) = getcache(S, get_symbol_diff_matrix(Val(2)))
@inline D₋₂(S::SpaceType) = getcache(S, get_symbol_diff_matrix(Val(3)))
@inline D₋ᵢ(S::SpaceType, ::Val{i}) where i = getcache(S, get_symbol_diff_matrix(Val(i)))

@inline D₋ₓ(u::MatrixElement) = getcache(space(u), get_symbol_diff_matrix(Val(1))) * u
@inline D₋ᵧ(u::MatrixElement) = getcache(space(u), get_symbol_diff_matrix(Val(2))) * u
@inline D₋₂(u::MatrixElement) = getcache(space(u), get_symbol_diff_matrix(Val(3))) * u

#∇ₕ(u::VecOrMatElem) = ntuple(i -> D₋2dim(u, Val(i))::VecOrMatElem, Val(dim(u)))::NTuple{dim(u),VecOrMatElem}
∇ₕ(u::VectorElement) = ntuple(i -> D₋ᵢ(u, Val(i)), dim(u))
∇ₕ(u::MatrixElement) = ntuple(i -> D₋ᵢ(space(u), Val(i))*u, dim(u))