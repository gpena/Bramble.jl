# Functions to help create ScalarGridSpace instante
"""
	$(TYPEDEF)

A `CompositeGridSpace` represents a grid space that is formed by composing `N` individual sub-spaces.

# Fields

$(FIELDS)
"""
mutable struct CompositeGridSpace{N,Spaces<:NTuple{N,AbstractSpaceType{<:Any}}} <: AbstractSpaceType{N}
	spaces::Spaces
end

ncomponents(::Type{<:CompositeGridSpace{N}}) where N = N
ComponentStyle(::Type{<:CompositeGridSpace{N}}) where N = MultiComponent{ncomponents(CompositeGridSpace{N})}()

# Interface implementations for CompositeGridSpace
@inline mesh(Wₕ::CompositeGridSpace) = mesh(Wₕ.spaces[1])
@inline mesh_type(Wₕ::CompositeGridSpace) = typeof(mesh(Wₕ))
@inline dim(Wₕ::CompositeGridSpace) = dim(Wₕ.spaces[1])
@inline eltype(Wₕ::CompositeGridSpace) = eltype(Wₕ.spaces[1])
@inline eltype(::Type{<:CompositeGridSpace{N,Spaces}}) where {N,Spaces} = eltype(Spaces.parameters[1])
@inline backend(Wₕ::CompositeGridSpace) = backend(Wₕ.spaces[1])
@inline ndofs(Wₕ::CompositeGridSpace) = sum(ndofs, Wₕ.spaces)
@inline ndofs(Wₕ::CompositeGridSpace, ::Type{Tuple}) = map(ndofs, Wₕ.spaces)

@inline first_space(Wₕ::ScalarGridSpace) = Wₕ
@inline first_space(Wₕ::CompositeGridSpace) = Wₕ.spaces[1]

# Overload product operator for space construction
@inline ×(X::AbstractSpaceType, Y::AbstractSpaceType) = CompositeGridSpace((X, Y))
@inline ×(X::CompositeGridSpace, Y::AbstractSpaceType) = CompositeGridSpace((X.spaces..., Y))
@inline ×(X::AbstractSpaceType, Y::CompositeGridSpace) = CompositeGridSpace((X, Y.spaces...))
@inline ×(X::CompositeGridSpace, Y::CompositeGridSpace) = CompositeGridSpace((X.spaces..., Y.spaces...))

