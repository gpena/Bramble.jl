
"""
	MatrixElement{S, T}

A `MatrixElement` is a container with a matrix where each entry is a `T` and associated with space `S`.
The matrix is represented as a `SparseMatrixCSC` for efficiency.
"""
struct MatrixElement{S,T} <: AbstractMatrix{T}
	space::S
	values::SparseMatrixCSC{T,Int}
end

"""
	Elements(S::SpaceType)

Create a `MatrixElement` from a given `SpaceType`.

# Inputs

  - `S`: a `SpaceType`
"""
@inline Elements(S::SpaceType) = MatrixElement(S, spdiagm(0 => FillArrays.Ones(eltype(S), ndofs(S))))

"""
	Elements(S::SpaceType, A::SparseMatrixCSC)

Create a `MatrixElement` from a given `SpaceType` and `SparseMatrixCSC`.

# Inputs

  - `S`: a `SpaceType`
  - `A`: a `SparseMatrixCSC`
"""
@inline Elements(S::SpaceType, A::SparseMatrixCSC) = MatrixElement(S, A)

"""
	ndims(::Type{<:MatrixElement})

Get the number of dimensions of a `MatrixElement`.

# Inputs

  - `::Type{<:MatrixElement}`: the type of the `MatrixElement`
"""
@inline ndims(::Type{<:MatrixElement}) = 2

"""
	show(io::IO, u::MatrixElement)

Show a `MatrixElement`.

# Inputs

  - `io::IO`: the output stream
  - `u::MatrixElement`: the `MatrixElement` to show
"""
show(io::IO, u::MatrixElement) = show(io, "text/plain", u.values)

"""
	eltype(M::MatrixElement{S,T})

Get the type of each element in a `MatrixElement`.

# Inputs

  - `M::MatrixElement{S,T}`: a `MatrixElement`
"""
@inline eltype(M::MatrixElement{S,T}) where {S,T} = T
@inline eltype(::Type{<:MatrixElement{S,T}}) where {S,T} = T

"""
	dim(u::MatrixElement)

Get the dimensionality of a `MatrixElement`.

# Inputs

  - `u::MatrixElement`: the `MatrixElement`
"""
@inline dim(u::MatrixElement) = dim(typeof(u))
@inline dim(::Type{<:MatrixElement{S}}) where S = dim(S)

"""
	length(u::MatrixElement)

Get the number of elements of a `MatrixElement`.

# Inputs

  - `u::MatrixElement`: the `MatrixElement`
"""
@inline length(u::MatrixElement) = length(u.values)

"""
	space(u::MatrixElement)

Get the space associated with a `MatrixElement`.

# Inputs

  - `u::MatrixElement`: the `MatrixElement`
"""
@inline space(u::MatrixElement) = u.space

"""
	mesh(u::MatrixElement)

Get the mesh associated with a `MatrixElement`.

# Inputs

  - `u::MatrixElement`: the `MatrixElement`
"""
@inline mesh(u::MatrixElement) = mesh(space(u))

"""
	similar(uh::MatrixElement)

Create a `MatrixElement` similar to another one.

# Inputs

  - `uh::MatrixElement`: the `MatrixElement` to copy
"""
@inline similar(uh::MatrixElement) = MatrixElement(space(uh), similar(uh.values))

"""
	size(u::MatrixElement)

Get the size of a `MatrixElement`.

# Inputs

  - `u::MatrixElement`: the `MatrixElement`
"""
@inline size(u::MatrixElement) = size(u.values)

"""
	Copy the values of `v` into `u`.

# Inputs

  - `u::MatrixElement`: the destination
  - `v::MatrixElement`: the source
"""
@inline copyto!(u::MatrixElement, v::MatrixElement) = (@.. u.values.nzval = v.values.nzval)

"""
	Get the `i`-th element of `u`.

# Inputs

  - `u::MatrixElement`: the `MatrixElement`
  - `i`: the index
"""
getindex(u::MatrixElement, i::Int) = getindex(u.values, i)
getindex(u::MatrixElement, I::Vararg{Int,N}) where N = getindex(u.values, I...)

"""
	Set the `i`-th element of `u` to `v`.

# Inputs

  - `u::MatrixElement`: the `MatrixElement`
  - `i`: the index
  - `v`: the new value
"""
setindex!(u::MatrixElement, v, i::Int) = (setindex!(u.values, v, i))
setindex!(u::MatrixElement, v, I::Vararg{Int,N}) where N = (setindex!(u.values, v, I))

"""
	Get the first index of `u`.

# Inputs

  - `u::MatrixElement`: the `MatrixElement`
"""
@inline firstindex(u::MatrixElement) = firstindex(u.values)

"""
	Get the last index of `u`.

# Inputs

  - `u::MatrixElement`: the `MatrixElement`
"""
@inline lastindex(u::MatrixElement) = lastindex(u.values)

@inline axes(u::MatrixElement) = axes(u.values)

"""
	Iterate over the `MatrixElement`.

# Inputs

  - `u::MatrixElement`: the `MatrixElement`
"""
@inline iterate(u::MatrixElement) = iterate(u.values)

"""
	Iterate over the `MatrixElement` with a state.

# Inputs

  - `u::MatrixElement`: the `MatrixElement`
  - `state`: the state
"""
@inline iterate(u::MatrixElement, state) = iterate(u.values, state)

const VecOrMatElem{S,T} = Union{VectorElement{S,T},MatrixElement{S,T}}

"""
	*(u::VectorElement, V::MatrixElement)

Return a new `MatrixElement` `R` with coefficients obtained by multiplying the coefficients of `u` with the coefficients of `V`.

**Inputs:**

  - `u`: a vector element
  - `V`: a matrix element
"""
function *(u::VectorElement, V::MatrixElement)
	R = similar(V)
	mul!(R.values, Diagonal(u.values), V.values)

	return R
end

"""
	*(u::MatrixElement, V::VectorElement)

Return a new `VectorElement` `R` with coefficients obtained by multiplying the coefficients of `u` with the coefficients of `V`.

**Inputs:**

  - `u`: a matrix element
  - `V`: a vector element
"""
function *(u::MatrixElement, V::VectorElement)
	R = similar(V)
	mul!(R.values, V.values, Diagonal(u.values))

	return R
end

for op in (:+, :-, :*)
	@eval begin
		($op)(u::MatrixElement, v::MatrixElement) = MatrixElement(space(u), ($op)(u.values, v.values))
	end
end

for op in (:+, :-, :*, :/, :^, :\)
	@eval begin
		"""
			(op)(α::AbstractFloat, u::MatrixElement)

		Return a new `MatrixElement` `R` with coefficients obtained by applying `(op)` to the coefficients of `u` and `α`.

		**Inputs:**

		  - `α`: a scalar
		  - `u`: a matrix element
		"""
		function ($op)(α::AbstractFloat, u::MatrixElement)
			r = similar(u)
			map!(x -> $op(α, x), r.values.nzval, u.values.nzval)
			return r
		end

		"""
			(op)(u::MatrixElement, α::AbstractFloat)

		Return a new `MatrixElement` `R` with coefficients obtained by applying `(op)` to the coefficients of `u` and `α`.

		**Inputs:**

		  - `u`: a matrix element
		  - `α`: a scalar
		"""
		function ($op)(u::MatrixElement, α::AbstractFloat)
			r = similar(u)
			map!(x -> $op(x, α), r.values.nzval, u.values.nzval)
			return r
		end
	end
end
