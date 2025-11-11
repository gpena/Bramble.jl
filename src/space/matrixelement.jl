#=
# matrixelement.jl

This file implements the MatrixElement type - a matrix-valued function on a finite element space.

## Mathematical Interpretation

A MatrixElement represents a linear operator T : Vₕ → Wₕ as a matrix. If {φᵢ} is a basis 
for Vₕ and {ψⱼ} is a basis for Wₕ, then:

    T = [Tᵢⱼ] where Tᵢⱼ = ⟨T(φⱼ), ψᵢ⟩

Common use cases:
- **Identity operator**: I : Vₕ → Vₕ (default)
- **Projection operators**: Πₕ : V → Vₕ
- **Interpolation operators**: Iₕ : C(Ω) → Vₕ
- **Restriction/prolongation**: R : Vₕ → Vₕ/₂ or P : Vₕ/₂ → Vₕ

## Design Rationale

MatrixElement differs from VectorElement in that:
1. It represents operators, not functions
2. It stores a matrix, not point values
3. Composition is matrix multiplication, not pointwise operations

This enables:
- Efficient operator composition: (A ∘ B)ₕ = Aₕ * Bₕ
- Precomputed transformations for performance
- Reusable operators across multiple solves

## Example

```julia
# Create identity operator
Iₕ = elements(Vₕ)  # Returns MatrixElement with identity matrix

# Create interpolation operator
A = compute_interpolation_matrix(...)
Tₕ = elements(Vₕ, A)

# Apply operator to vector element
uₕ = elements(Vₕ, u_values)
v_values = Tₕ.data * uₕ.data
```

See also: [`MatrixElement`](@ref), [`VectorElement`](@ref), [`elements`](@ref)
=#

"""
	elements(Wₕ::AbstractSpaceType, [A::AbstractMatrix])

Returns a [MatrixElement](@ref) from a given [AbstractSpaceType](@ref), initialized with the identity matrix. If matrix `A` is provided, it is used instead.
"""
@inline function elements(Wₕ::AbstractSpaceType)
	b = backend(Wₕ)

	ST = typeof(Wₕ)
	MT = matrix_type(b)
	T = eltype(b)
	matrix = backend_eye(b, ndofs(Wₕ))
	return MatrixElement{ST,T,MT}(matrix, Wₕ)
end

@inline function elements(Wₕ::AbstractSpaceType, A::AbstractMatrix)
	b = backend(Wₕ)

	ST = typeof(Wₕ)
	MT = matrix_type(b)
	T = eltype(b)
	matrix = MT(A)
	return MatrixElement{ST,T,MT}(matrix, Wₕ)
end

"""
	eltype(Uₕ::MatrixElement{S,T})

Returns the element type of a [MatrixElement](@ref), `T`. CAn be applied to the type of the [MatrixElement](@ref).
"""
@inline eltype(::MatrixElement{S,T}) where {S,T} = T
@inline eltype(::Type{<:MatrixElement{S,T}}) where {S,T} = T

@inline space_type(::MatrixElement{S}) where S = S
@inline space_type(::Type{<:MatrixElement{S}}) where S = S

"""
	space(Uₕ::MatrixElement)

Returns the space associated with the [MatrixElement](@ref) `Uₕ`.
"""
@inline space(Uₕ::MatrixElement) = Uₕ.space

"""
	Base.similar(Uₕ::MatrixElement)

Returns a similar [MatrixElement](@ref) to `Uₕ`.
"""
@inline Base.similar(Uₕ::MatrixElement{S,T,MT}) where {S,T,MT} = MatrixElement{S,T,MT}(similar(Uₕ.data), space(Uₕ))

@forward MatrixElement.data (Base.size, Base.length, Base.ndims, Base.firstindex, Base.lastindex, Base.iterate, Base.axes, Bramble.show)
@forward MatrixElement.space (Bramble.mesh,)

"""
	Base.copyto!(Uₕ::MatrixElement, Vₕ::MatrixElement)

Copies the coefficients of [MatrixElement](@ref) `Vₕ` into [MatrixElement](@ref) `Uₕ`.
"""
@inline Base.copyto!(Uₕ::MatrixElement, Vₕ::MatrixElement) = (copyto!(Uₕ.data, Vₕ.data))

@inline Base.@propagate_inbounds getindex(Uₕ::MatrixElement, i::Int) = getindex(Uₕ.data, i)
@inline Base.@propagate_inbounds getindex(Uₕ::MatrixElement, I::Vararg{Int,N}) where N = getindex(Uₕ.data, I...)
@inline Base.@propagate_inbounds getindex(Uₕ::MatrixElement, I::NTuple{N,Int}) where N = getindex(Uₕ.data, I...)

@inline Base.@propagate_inbounds setindex!(Uₕ::MatrixElement, v, i::Int) = (setindex!(Uₕ.data, v, i); return)
@inline Base.@propagate_inbounds setindex!(Uₕ::MatrixElement, v, I::Vararg{Int,N}) where N = (setindex!(Uₕ.data, v, I...); return)
@inline Base.@propagate_inbounds setindex!(Uₕ::MatrixElement, v, I::NTuple{N,Int}) where N = (setindex!(Uₕ.data, v, I...); return)

"""
	VecOrMatElem{S,T}

Type alias for a union of [VectorElement](@ref) and [MatrixElement](@ref).

This is used in function signatures that accept either vectors or matrices from
the same function space, particularly in inner product calculations.

# Type Parameters
- `S`: The space type (e.g., `ScalarGridSpace{...}`)
- `T`: The element type (e.g., `Float64`)

# Example
```julia
# This function accepts either VectorElement or MatrixElement
function my_inner_product(uₕ::VecOrMatElem, vₕ::VecOrMatElem)
	# ... implementation ...
end
```

See also: [`VectorElement`](@ref), [`MatrixElement`](@ref), [`innerₕ`](@ref)
"""
const VecOrMatElem{S,T} = Union{VectorElement{S,T},MatrixElement{S,T}}

"""
	Base.:*(uₕ::VectorElement, Vₕ::MatrixElement)

Returns a new [MatrixElement](@ref) calculated by multiplying each coefficient of [VectorElement](@ref) `uₕ` with the corresponding row of `Vₕ`.
"""
function Base.:*(uₕ::VectorElement, Vₕ::MatrixElement)
	Zₕ = similar(Vₕ)
	mul!(Zₕ.data, Diagonal(uₕ.data), Vₕ.data)
	return Zₕ
end

@inline Base.:*(uₕ::VectorElement, Vₕ::NTuple{D,MatrixElement}) where D = ntuple(i -> uₕ * Vₕ[i], Val(D))

@inline Base.:*(uₕ::Tuple{VectorElement}, Vₕ::MatrixElement) = uₕ[1] * Vₕ

"""
	Base.:*(Uₕ::MatrixElement, vₕ::VectorElement)

Returns a new [MatrixElement](@ref) calculated by multiplying each coefficient of [VectorElement](@ref) `vₕ` with the corresponding column of `Uₕ`.
"""
@inline function Base.:*(Uₕ::MatrixElement, vₕ::VectorElement)
	Zₕ = similar(Uₕ)
	mul!(Zₕ.data, Uₕ.data, Diagonal(vₕ.data))

	return Zₕ
end

@inline *(Uₕ::MatrixElement, vₕ::Tuple{VectorElement}) = Uₕ * vₕ[1]

@inline ⋅(Uₕ::NTuple{D,VectorElement}, Vₕ::NTuple{D,MatrixElement}) where D = ntuple(i -> Uₕ[i] * Vₕ[i], D)

for op in (:+, :-, :*)
	dict = Dict(:+ => ("adding", "to", true), :- => ("subtracting", "to", false), :* => ("multiplying", "by", true))
	w1, w2, w3 = dict[op]
	el1, el2 = w3 ? ("`Uₕ`", "`Vₕ`") : ("`Vₕ`", "`Uₕ`")
	text = "\n\nReturns a new [MatrixElement](@ref) given by $w1 the matrix of [MatrixElement](@ref) $el1 $w2 the matrix of [MatrixElement](@ref) $el2."

	docstr = "	$(string(op))(Uₕ::MatrixElement, Vₕ::MatrixElement) $text"
	@eval begin
		@doc $docstr @inline (Base.$op)(Uₕ::MatrixElement, Vₕ::MatrixElement) = elements(space(Uₕ), (Base.$op)(Uₕ.data, Vₕ.data))
	end
end

@inline function Base.:^(Uₕ::MatrixElement, i::Integer)
	Vₕ = similar(Uₕ)

	map!(Base.Fix2(Base.:^, i), Vₕ.data, Uₕ.data)
	return Vₕ
end

@inline function Base.:^(Uₕ::MatrixElement, i::Real)
	Vₕ = similar(Uₕ)

	map!(Base.Fix2(Base.:^, i), Vₕ.data, Uₕ.data)
	return Vₕ
end

for op in (:+, :-, :*, :/)
	same_text = "\n\nReturns a new [MatrixElement](@ref) with coefficients given by the elementwise evaluation of"
	docstr1 = "	" * string(op) * "(α::Number, Uₕ::MatrixElement)" * same_text * " `α` " * string(op) * " `Uₕ`."
	docstr2 = "	" * string(op) * "(Uₕ::MatrixElement, α::Number)" * same_text * " `Uₕ` " * string(op) * " `α`."

	@eval begin
		@doc $docstr1 @inline function (Base.$op)(α::Number, Uₕ::MatrixElement)
			Vₕ = similar(Uₕ)
			map!(Base.Fix1(Base.$op, α), Vₕ.data, Uₕ.data)
			return Vₕ
		end

		@doc $docstr2 @inline function (Base.$op)(Uₕ::MatrixElement, α::Number)
			Vₕ = similar(Uₕ)
			map!(Base.Fix2(Base.$op, α), Vₕ.data, Uₕ.data)
			return Vₕ
		end
	end
end