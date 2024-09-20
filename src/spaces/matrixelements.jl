"""
	MatrixElement{S, T}

A `MatrixElement` is a container with a sparse matrix where each entry is a `T` and a space `S`. Its purpose is to represent discretization matrices from finite difference methods.
"""
struct MatrixElement{S,T} <: AbstractMatrix{T}
	space::S
	values::SparseMatrixCSC{T,Int}
end

"""
	elements(Wₕ::SpaceType)

Returns a [MatrixElement](@ref) from a given [SpaceType](@ref), initialized with as the identity matrix.
"""
@inline elements(Wₕ::SpaceType) = MatrixElement(Wₕ, spdiagm(0 => FillArrays.Ones(eltype(Wₕ), ndofs(Wₕ))))

"""
	elements(Wₕ::SpaceType, A::SparseMatrixCSC)

Returns a [MatrixElement](@ref) from a given [SpaceType](@ref), initialized with the sparse matrix `A`.
"""
@inline elements(Wₕ::SpaceType, A::SparseMatrixCSC) = MatrixElement(Wₕ, A)

@inline ndims(::Type{MatrixElement{S,T}}) where {S,T} = 2

show(io::IO, Uₕ::MatrixElement) = show(io, "text/plain", Uₕ.values)

"""
	eltype(Uₕ::MatrixElement{S,T})

Returns the element type of a [MatrixElement](@ref), `T``.
"""
@inline eltype(Uₕ::MatrixElement{S,T}) where {S,T} = T
@inline eltype(::Type{MatrixElement{S,T}}) where {S,T} = T

"""
	length(Uₕ::MatrixElement)

Returns the length of a [MatrixElement](@ref).
"""
@inline length(Uₕ::MatrixElement) = length(Uₕ.values)

"""
	space(Uₕ::MatrixElement)

Returns the space associated with the [MatrixElement](@ref) `Uₕ`.
"""
@inline space(Uₕ::MatrixElement) = Uₕ.space

"""
	similar(Uₕ::MatrixElement)

Returns a similar [MatrixElement](@ref) to `Uₕ`.
"""
@inline similar(Uₕ::MatrixElement) = MatrixElement(space(Uₕ), similar(Uₕ.values))

"""
	size(Uₕ::MatrixElement)

Returns the size of a [MatrixElement](@ref) `Uₕ`.
"""
@inline size(Uₕ::MatrixElement) = size(Uₕ.values)

"""
	copyto!(Uₕ::MatrixElement, Vₕ::MatrixElement)

Copies the coefficients of [MatrixElement](@ref) `Vₕ` into [MatrixElement](@ref) `Uₕ`.
"""
@inline copyto!(Uₕ::MatrixElement, Vₕ::MatrixElement) = (@.. Uₕ.values.nzval = Vₕ.values.nzval)

Base.@propagate_inbounds getindex(Uₕ::MatrixElement, i::Int) = getindex(Uₕ.values, i)
Base.@propagate_inbounds getindex(Uₕ::MatrixElement, I::Vararg{Int,N}) where N = getindex(Uₕ.values, I...)
Base.@propagate_inbounds getindex(Uₕ::MatrixElement, I::NTuple{N,Int}) where N = getindex(Uₕ.values, I...)

Base.@propagate_inbounds setindex!(Uₕ::MatrixElement, v, i::Int) = (setindex!(Uₕ.values, v, i))
Base.@propagate_inbounds setindex!(Uₕ::MatrixElement, v, I::Vararg{Int,N}) where N = (setindex!(Uₕ.values, v, I...))
Base.@propagate_inbounds setindex!(Uₕ::MatrixElement, v, I::NTuple{N,Int}) where N = (setindex!(Uₕ.values, v, I...))

@inline firstindex(Uₕ::MatrixElement) = firstindex(Uₕ.values)
@inline lastindex(Uₕ::MatrixElement) = lastindex(Uₕ.values)
@inline axes(Uₕ::MatrixElement) = axes(Uₕ.values)

@inline iterate(Uₕ::MatrixElement) = iterate(Uₕ.values)
@inline iterate(Uₕ::MatrixElement, state) = iterate(Uₕ.values, state)

const VecOrMatElem{S,T} = Union{VectorElement{S,T},MatrixElement{S,T}}

"""
	*(uₕ::VectorElement, Vₕ::MatrixElement)

Returns a new [MatrixElement](@ref) calculated by multiplying each coefficient of `uₕ` with the corresponding row of `Vₕ`.
"""
function *(uₕ::VectorElement, Vₕ::MatrixElement)
	Zₕ = similar(Vₕ)
	mul!(Zₕ.values, Diagonal(uₕ.values), Vₕ.values)

	return Zₕ
end

"""
	*(Uₕ::MatrixElement, vₕ::VectorElement)

Returns a new [MatrixElement](@ref) calculated by multiplying each coefficient of `Uₕ` with the corresponding column of `vₕ`.
"""
function *(Uₕ::MatrixElement, vₕ::VectorElement)
	Zₕ = similar(Uₕ)
	mul!(Zₕ.values, Uₕ.values, Diagonal(vₕ.values))

	return Zₕ
end

for op in (:+, :-, :*) #### parei aqui
	@eval begin
		(Base.$op)(Uₕ::MatrixElement, Vₕ::MatrixElement) = MatrixElement(space(Uₕ), ($op)(Uₕ.values, Vₕ.values))
	end
end

for op in (:+, :-, :*, :/, :^, :\)
	@eval begin
		"""
			(op)(α::AbstractFloat, Uₕ::MatrixElement)

		Returns a new `MatrixElement` `R` with coefficients obtained by applying `(op)` to the coefficients of `u` and `α`.
		"""
		function (Base.$op)(α::Number, Uₕ::MatrixElement)
			r = similar(Uₕ)
			map!(Base.Fix1(Base.$op, α), r.values.nzval, Uₕ.values.nzval)
			return r
		end

		"""
			(op)(Uₕ::MatrixElement, α::AbstractFloat)

		Returns a new `MatrixElement` `R` with coefficients obtained by applying `(op)` to the coefficients of `u` and `α`.
		"""
		function (Base.$op)(Uₕ::MatrixElement, α::Number)
			r = similar(Uₕ)
			map!(Base.Fix2(Base.$op, α), r.values.nzval, Uₕ.values.nzval)
			return r
		end
	end
end