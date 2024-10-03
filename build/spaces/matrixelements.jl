"""
	elements(Wₕ::SpaceType)

Returns a [MatrixElement](@ref) from a given [SpaceType](@ref), initialized with the identity matrix.
"""
@inline elements(Wₕ::SpaceType) = MatrixElement(Wₕ, spdiagm(0 => Ones(eltype(Wₕ), ndofs(Wₕ))))

"""
	elements(Wₕ::SpaceType, A::SparseMatrixCSC)

Returns a [MatrixElement](@ref) from a given [SpaceType](@ref), initialized with the sparse matrix `A`.
"""
@inline elements(Wₕ::SpaceType, A::SparseMatrixCSC) = MatrixElement(Wₕ, A)

@inline ndims(::Type{MatrixElement{S,T}}) where {S,T} = 2

show(io::IO, Uₕ::MatrixElement) = show(io, "text/plain", Uₕ.values)

"""
	eltype(Uₕ::MatrixElement{S,T})
	eltype(::Type{MatrixElement{S,T}})

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

@inline size(Uₕ::MatrixElement) = size(Uₕ.values)

"""
	copyto!(Uₕ::MatrixElement, Vₕ::MatrixElement)

Copies the coefficients of [MatrixElement](@ref) `Vₕ` into [MatrixElement](@ref) `Uₕ`.
"""
@inline copyto!(Uₕ::MatrixElement, Vₕ::MatrixElement) = (copyto!(Uₕ.values, Vₕ.values))

@inline Base.@propagate_inbounds function getindex(Uₕ::MatrixElement, i::Int)
	@boundscheck checkbounds(Uₕ.values, i)
	getindex(Uₕ.values, i)
end

@inline Base.@propagate_inbounds function getindex(Uₕ::MatrixElement, I::Vararg{Int,N}) where N
	@boundscheck checkbounds(Uₕ.values, I...)
	getindex(Uₕ.values, I...)
end

@inline Base.@propagate_inbounds function getindex(Uₕ::MatrixElement, I::NTuple{N,Int}) where N
	@boundscheck checkbounds(Uₕ.values, I...)
	getindex(Uₕ.values, I...)
end

@inline Base.@propagate_inbounds function setindex!(Uₕ::MatrixElement, v, i::Int)
	@boundscheck checkbounds(Uₕ.values, i)
	setindex!(Uₕ.values, v, i)
end

@inline Base.@propagate_inbounds function setindex!(Uₕ::MatrixElement, v, I::Vararg{Int,N}) where N
	@boundscheck checkbounds(Uₕ.values, I...)
	setindex!(Uₕ.values, v, I...)
end

@inline Base.@propagate_inbounds function setindex!(Uₕ::MatrixElement, v, I::NTuple{N,Int}) where N
	@boundscheck checkbounds(Uₕ.values, I...)
	setindex!(Uₕ.values, v, I...)
end

@inline firstindex(Uₕ::MatrixElement) = firstindex(Uₕ.values)
@inline lastindex(Uₕ::MatrixElement) = lastindex(Uₕ.values)
@inline axes(Uₕ::MatrixElement) = axes(Uₕ.values)

@inline iterate(Uₕ::MatrixElement) = iterate(Uₕ.values)
@inline iterate(Uₕ::MatrixElement, state) = iterate(Uₕ.values, state)

const VecOrMatElem{S,T} = Union{VectorElement{S,T},MatrixElement{S,T}}

"""
	*(uₕ::VectorElement, Vₕ::MatrixElement)

Returns a new [MatrixElement](@ref) calculated by multiplying each coefficient of [VectorElement](@ref) `uₕ` with the corresponding row of `Vₕ`.
"""
function *(uₕ::VectorElement, Vₕ::MatrixElement)
	Zₕ = similar(Vₕ)
	mul!(Zₕ.values, Diagonal(uₕ.values), Vₕ.values)
	return Zₕ
end

function *(uₕ::VectorElement, Vₕ::NTuple{D,MatrixElement}) where D
	Zₕ = ntuple(i-> similar(Vₕ[i]), D)
	for i in 1:D
		mul!(Zₕ[i].values, Diagonal(uₕ.values), Vₕ[i].values)
	end
	return Zₕ
end

@inline *(uₕ::Tuple{VectorElement}, Vₕ::MatrixElement) = uₕ[1] * Vₕ


"""
	*(Uₕ::MatrixElement, vₕ::VectorElement)

Returns a new [MatrixElement](@ref) calculated by multiplying each coefficient of [VectorElement](@ref) `vₕ` with the corresponding column of `Uₕ`.
"""
@inline function *(Uₕ::MatrixElement, vₕ::VectorElement)
	Zₕ = similar(Uₕ)
	mul!(Zₕ.values, Uₕ.values, Diagonal(vₕ.values))

	return Zₕ
end

@inline *(Uₕ::MatrixElement, vₕ::Tuple{VectorElement}) = Uₕ * vₕ[1]

@inline *(uₕ::NTuple{D, VectorElement}, Vₕ::NTuple{D,MatrixElement}) where D = ntuple(i-> uₕ[i] * Vₕ[i], D)

for op in (:+, :-, :*)
	dict = Dict(:+ => ("adding", "to", true), :- => ("subtracting", "to", false), :* => ("multiplying", "by", true))
	w1, w2, w3 = dict[op]
	el1, el2 = w3 ? ("`Uₕ`", "`Vₕ`") : ("`Vₕ`", "`Uₕ`")
	text = "\n\nReturns a new [MatrixElement](@ref) given by $w1 the matrix of [MatrixElement](@ref) $el1 $w2 the matrix of [MatrixElement](@ref) $el2."

	docstr = "	$(string(op))(Uₕ::MatrixElement, Vₕ::MatrixElement) $text"
	@eval begin
		@doc $docstr @inline (Base.$op)(Uₕ::MatrixElement, Vₕ::MatrixElement) = MatrixElement(space(Uₕ), (Base.$op)(Uₕ.values, Vₕ.values))
	end
end

@inline function Base.:^(Uₕ::MatrixElement, i::Integer)
	Vₕ = similar(Uₕ)

	map!(Base.Fix1(Base.:^, i), Vₕ.values.nzval, Uₕ.values.nzval)
	return Vₕ
end

@inline function Base.:^(Uₕ::MatrixElement, i::Real)
	Vₕ = similar(Uₕ)

	map!(Base.Fix1(Base.:^, i), Vₕ.values.nzval, Uₕ.values.nzval)
	return Vₕ
end

for op in (:+, :-, :*, :/, :^)
	same_text = "\n\nReturns a new [MatrixElement](@ref) with coefficients given by the elementwise evaluation of"
	docstr1 = "	" * string(op) * "(α::Number, Uₕ::MatrixElement)" * same_text * "`α`" * string(op) * "`Uₕ`."
	docstr2 = "	" * string(op) * "(Uₕ::MatrixElement, α::Number)" * same_text * "`Uₕ`" * string(op) * "`α`."

	@eval begin
		@doc $docstr1 @inline function (Base.$op)(α::Number, Uₕ::MatrixElement)
			Vₕ = similar(Uₕ)
			map!(Base.Fix1(Base.$op, α), Vₕ.values.nzval, Uₕ.values.nzval)
			return Vₕ
		end

		@doc $docstr2 @inline function (Base.$op)(Uₕ::MatrixElement, α::Number)
			Vₕ = similar(Uₕ)
			map!(Base.Fix2(Base.$op, α), Vₕ.values.nzval, Uₕ.values.nzval)
			return Vₕ
		end
	end
end