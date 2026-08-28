#=
# vector_gridspace.jl

This file defines composite and vector grid spaces (`CompositeGridSpace`, alias `VectorGridSpace`),
allowing multi-component fields (such as vector-valued velocities or coupled solution blocks)
to be constructed and queried with zero overhead.
=#

"""
	$(TYPEDEF)

A `CompositeGridSpace` represents a grid space formed by composing `N` individual sub-spaces.
It is immutable and stack-allocatable, wrapping a tuple of spaces.

# Fields

$(FIELDS)
"""
struct CompositeGridSpace{N,Spaces<:Tuple} <: AbstractSpaceType{N}
	"the tuple of constituent sub-spaces."
	spaces::Spaces
end

CompositeGridSpace(spaces::Tuple) = CompositeGridSpace{length(spaces),typeof(spaces)}(spaces)
CompositeGridSpace{N}(spaces::Spaces) where {N,Spaces<:Tuple} = CompositeGridSpace{N,Spaces}(spaces)
CompositeGridSpace(spaces::AbstractSpaceType...) = CompositeGridSpace(spaces)

"""
	VectorGridSpace{N}

Type alias for a `CompositeGridSpace{N}`, representing a vector-valued function space.
"""
const VectorGridSpace{N} = CompositeGridSpace{N}

# ==============================================================================
# Constructors
# ==============================================================================

"""
	gridspace(Ωₕ::AbstractMeshType, ::Val{N}; nbuffers::Int = 1) where N
	gridspace(Ωₕ::AbstractMeshType, N::Int; nbuffers::Int = 1)

Constructs a vector function space with `N` components on mesh `Ωₕ`.
The underlying scalar space and its weights are computed once and shared across components.

`N == 1` yields the [`ScalarGridSpace`](@ref) itself rather than a one-component
composite, for both spellings. The element interface is uniform either way:
`uₕ(1)` and `components(uₕ)` work on a scalar element.

The `Val` form is always type stable. The `Int` form is stable wherever `N` is a
literal or otherwise constant-foldable, and returns a small `Union` when `N` is
only known at run time; prefer `Val` on hot paths.
"""
function gridspace(Ωₕ::AbstractMeshType, ::Val{N}; nbuffers::Int = 1) where N
	W = gridspace(Ωₕ; nbuffers = nbuffers)
	return CompositeGridSpace(ntuple(_ -> W, Val(N)))
end

@inline gridspace(Ωₕ::AbstractMeshType, ::Val{1}; nbuffers::Int = 1) = gridspace(Ωₕ; nbuffers = nbuffers)

# Forwarding to the Val method keeps a single implementation and makes the two
# spellings agree by construction; :aggressive recovers type stability whenever
# the caller's N is a constant.
Base.@constprop :aggressive function gridspace(Ωₕ::AbstractMeshType, N::Int; nbuffers::Int = 1)
	N >= 1 || throw(ArgumentError("Number of components N must be >= 1, got $N"))
	return gridspace(Ωₕ, Val(N); nbuffers = nbuffers)
end

"""
	vector_gridspace(Ωₕ::AbstractMeshType, [N = dim(Ωₕ)]; nbuffers::Int = 1)

Convenience constructor for a vector grid space on mesh `Ωₕ`. If `N` is omitted,
it defaults to the spatial dimension of the mesh (`dim(Ωₕ)`).
"""
@inline vector_gridspace(Ωₕ::AbstractMeshType; nbuffers::Int = 1) = gridspace(Ωₕ, Val(dim(Ωₕ)); nbuffers = nbuffers)
@inline vector_gridspace(Ωₕ::AbstractMeshType, ::Val{N}; nbuffers::Int = 1) where N = gridspace(Ωₕ, Val(N); nbuffers = nbuffers)
@inline vector_gridspace(Ωₕ::AbstractMeshType, N::Int; nbuffers::Int = 1) = gridspace(Ωₕ, N; nbuffers = nbuffers)

"""
	^(Wₕ::ScalarGridSpace, ::Val{N}) where N
	^(Wₕ::ScalarGridSpace, N::Int)

Constructs an `N`-component vector grid space from a scalar grid space `Wₕ` using mathematical exponentiation syntax:
`Vₕ = Wₕ^2` or `Vₕ = Wₕ^dim(mesh)`.

`Wₕ^1` is `Wₕ`, for both the `Int` and `Val` spellings.
"""
@inline Base.:^(Wₕ::ScalarGridSpace, ::Val{N}) where N = CompositeGridSpace(ntuple(_ -> Wₕ, Val(N)))
@inline Base.:^(Wₕ::ScalarGridSpace, ::Val{1}) = Wₕ

Base.@constprop :aggressive function Base.:^(Wₕ::ScalarGridSpace, N::Int)
	N >= 1 || throw(ArgumentError("Power N must be >= 1, got $N"))
	return Wₕ^Val(N)
end

# ==============================================================================
# Interface Implementations
# ==============================================================================

@inline first_space(Wₕ::ScalarGridSpace) = Wₕ
# Recursive: always return the first leaf ScalarGridSpace
@inline first_space(Wₕ::CompositeGridSpace) = first_space(Wₕ.spaces[1])

@inline mesh(Wₕ::CompositeGridSpace) = mesh(first_space(Wₕ))
@inline mesh_type(Wₕ::CompositeGridSpace) = typeof(mesh(Wₕ))
@inline dim(Wₕ::CompositeGridSpace) = dim(first_space(Wₕ))
@inline eltype(Wₕ::CompositeGridSpace) = eltype(first_space(Wₕ))
@inline eltype(::Type{<:CompositeGridSpace{<:Any,Spaces}}) where Spaces = eltype(fieldtype(Spaces, 1))
@inline backend(Wₕ::CompositeGridSpace) = backend(first_space(Wₕ))
@inline vector_buffer(Wₕ::CompositeGridSpace) = vector_buffer(first_space(Wₕ))
@inline ndofs(Wₕ::CompositeGridSpace) = sum(ndofs, Wₕ.spaces)
@inline ndofs(Wₕ::CompositeGridSpace, ::Type{Tuple}) = map(ndofs, Wₕ.spaces)

@inline weights(Wₕ::CompositeGridSpace) = weights(first_space(Wₕ))
@inline weights(Wₕ::CompositeGridSpace, ip::InnerProductType) = weights(first_space(Wₕ), ip)
@inline weights(Wₕ::CompositeGridSpace, ip::InnerProductType, i::Int) = weights(first_space(Wₕ), ip, i)

@inline spaces(Wₕ::CompositeGridSpace) = Wₕ.spaces

# ==============================================================================
# Collection Interface
# ==============================================================================

@inline Base.getindex(Wₕ::CompositeGridSpace, i::Int) = Wₕ.spaces[i]
@inline Base.length(::CompositeGridSpace{N}) where N = N
@inline Base.firstindex(::CompositeGridSpace) = 1
@inline Base.lastindex(::CompositeGridSpace{N}) where N = N
@inline Base.iterate(Wₕ::CompositeGridSpace, state...) = iterate(Wₕ.spaces, state...)
@inline Base.eachindex(::CompositeGridSpace{N}) where N = 1:N
@inline Base.keys(::CompositeGridSpace{N}) where N = 1:N

# ==============================================================================
# Cartesian Product (×)
# ==============================================================================

# Overload product operator for space construction.
# Scalar × Scalar → 2-element flat composite.
# CompositeGridSpace × anything → hierarchical composite (no flattening),
# enabling forms like form(Vh × Wh, Vh × Wh, ((u,p),(v,q)) -> ...).
@inline ×(X::AbstractSpaceType, Y::AbstractSpaceType) = CompositeGridSpace((X, Y))
