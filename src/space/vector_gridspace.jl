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
