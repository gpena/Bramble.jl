# Functions to help create ScalarGridSpace instante
"""
	struct CompositeGridSpace{Spaces<:NTuple{N,AbstractSpaceType{<:Any}}} <: AbstractSpaceType{N}
	end
"""
mutable struct CompositeGridSpace{N,Spaces<:NTuple{N,AbstractSpaceType{<:Any}}} <: AbstractSpaceType{N}
	spaces::Spaces
end

ncomponents(::Type{<:CompositeGridSpace{N}}) where N = N
ComponentStyle(::Type{<:CompositeGridSpace{N}}) where N = MultiComponent{ncomponents(CompositeGridSpace{N})}()
