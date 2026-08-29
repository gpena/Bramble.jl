#===========================================================================#
# The VectorElement interface.
#
# Accessors, the component views of a composite element, the constructors, and the
# array interface: indexing, `similar`, broadcasting and the tuple arithmetic. The two
# operators that produce a VectorElement from a function live beside this file, in
# restriction.jl and cell_average.jl.
#===========================================================================#

"""
	values(uₕ::VectorElement)

Returns the coefficients of the [VectorElement](@ref) `uₕ`.
"""
@inline values(uₕ::VectorElement) = uₕ.data

"""
	to_matrix(uₕ::VectorElement)

Reshapes the flat coefficient vector of `uₕ` into a multidimensional array that matches the logical layout of the grid points.

  - For a scalar space, this returns a D-dimensional array.
  - For an N-component vector space, it returns an N-tuple of arrays, one for each component.
"""
@inline to_matrix(uₕ::VectorElement{<:ScalarGridSpace}) = Base.ReshapedArray(
    values(uₕ), npoints(mesh(space(uₕ)), Tuple), ())
@inline to_matrix(uₕ::VectorElement{<:CompositeGridSpace{N}}) where {N} = ntuple(i -> to_matrix(uₕ(i)), Val(N))

"""
	values!(uₕ::VectorElement, s)

Copies the values of `s` into the coefficients of [VectorElement](@ref) `uₕ`. Returns `nothing`.
"""
@inline function values!(uₕ::VectorElement, s)
    copyto!(values(uₕ), s)
    return nothing
end

"""
	space(uₕ::VectorElement)

Returns the grid space associated with [VectorElement](@ref) `uₕ`.
"""
@inline space(uₕ::VectorElement) = uₕ.space
@inline space_type(::Type{<:VectorElement{S}}) where {S} = S

# Forward array-like methods to the `data` field. This allows a VectorElement
# to behave like a standard Julia vector (e.g., support `size`, `length`, `eltype`).
@forward VectorElement.data (Base.size, Bramble.show)

# A VectorElement wraps a vector, so indexing is linear; without this the
# AbstractArray default of IndexCartesian() is used.
@inline Base.IndexStyle(::Type{<:VectorElement}) = IndexLinear()
@forward VectorElement.space (Bramble.mesh,)

# ==============================================================================
# Component Indexing
# ==============================================================================

"""
	component_range(Wₕ::CompositeGridSpace{N}, i::Int) where N

Returns the degree-of-freedom index range for the `i`-th constituent space of composite space `Wₕ`.
"""
@inline function component_range(Wₕ::CompositeGridSpace{N}, i::Int) where {N}
    @boundscheck (1 <= i <= N) || throw(BoundsError(Wₕ, i))
    return @inbounds component_ranges(Wₕ)[i]
end

"""
	component_ranges(Wₕ::CompositeGridSpace{N}) where N

Returns an `NTuple{N, UnitRange{Int}}` containing the degree-of-freedom ranges for all `N` components.
"""
@inline function component_ranges(Wₕ::CompositeGridSpace{N}) where {N}
    subs = spaces(Wₕ)
    # `ntuple` over `Val(N)` and `cumsum` on a tuple both unroll, so this is a
    # handful of adds with no loop and no allocation. Sizes are summed rather
    # than assumed equal: subspaces of the same *type* can still hold different
    # numbers of degrees of freedom, so nothing here may be inferred from types.
    ns = ntuple(i -> ndofs(subs[i]), Val(N))
    stops = cumsum(ns)
    return ntuple(i -> (stops[i] - ns[i] + 1):stops[i], Val(N))
end

"""
	(uₕ::VectorElement)(i::Int)
	component(uₕ::VectorElement, i::Int)

Extracts a [`VectorElement`](@ref) view of the `i`-th field component of `uₕ`.

For a [`CompositeGridSpace`](@ref), this creates a lightweight, zero-copy view of the
`i`-th component's degrees of freedom. Mutating the returned component modifies `uₕ` in-place.

For a scalar [`ScalarGridSpace`](@ref), `uₕ(1)` returns `uₕ`.

# Examples
```julia
Vₕ = Wₕ^2
uₕ = element(Vₕ)
u_x = uₕ(1)
u_y = uₕ(2)

# In-place component assignment
u_x .= 1.0
```
"""
@inline function (uₕ::VectorElement{<:CompositeGridSpace})(i::Int)
    rng = component_range(space(uₕ), i)
    subspace = spaces(space(uₕ))[i]
    v_data = @views values(uₕ)[rng]
    return VectorElement(v_data, subspace)
end

@inline function (uₕ::VectorElement{<:ScalarGridSpace})(i::Int)
    @boundscheck i == 1 || throw(BoundsError(uₕ, i))
    return uₕ
end

"""
	component(uₕ::VectorElement, i::Int)

Extracts a [`VectorElement`](@ref) view of the `i`-th field component of `uₕ`. Alias for `uₕ(i)`.
"""
@inline component(uₕ::VectorElement, i::Int) = uₕ(i)

"""
	components(uₕ::VectorElement)

Returns an `NTuple` of [`VectorElement`](@ref) views for all components of `uₕ`.
"""
@inline function components(uₕ::VectorElement{<:CompositeGridSpace{N}}) where {N}
    ranges = component_ranges(space(uₕ))
    subs = spaces(space(uₕ))
    raw = values(uₕ)
    return ntuple(i -> VectorElement(@views(raw[ranges[i]]), subs[i]), Val(N))
end

@inline components(uₕ::VectorElement{<:ScalarGridSpace}) = (uₕ,)

# Constructor for VectorElement
"""
	element(Wₕ::AbstractSpaceType, [α::Number])

Returns a [VectorElement](@ref) for grid space `Wₕ` with uninitialized components. if ``\\alpha`` is provided, the components are initialized to ``\\alpha``.
"""
@inline function element(Wₕ::AbstractSpaceType)
    # Get the backend (e.g., CPU, GPU) from the space.
    b = backend(Wₕ)

    # Determine the types for the space, vector, and elements.
    ST = typeof(Wₕ)
    VT = vector_type(b)
    T = eltype(b)

    # Allocate a vector with the correct number of degrees of freedom (DoFs) and return the element.
    return VectorElement{ST, T, VT}(vector(b, ndofs(Wₕ)), Wₕ)
end

# Constructor with a fill value `α`.
function element(Wₕ::AbstractSpaceType, α::Number)
    uₕ = element(Wₕ)
    fill!(uₕ, α)
    return uₕ
end

"""
	element(Wₕ::AbstractSpaceType, v::AbstractVector)

Returns a [VectorElement](@ref) for a grid space `Wₕ` with the same coefficients of `v`.
"""
@inline function element(Wₕ::AbstractSpaceType, v::AbstractVector)
    # Ensure the provided vector has the correct number of DoFs.
    length(v) == ndofs(Wₕ) || throw(DimensionMismatch(
        "input vector has length $(length(v)), but the space has $(ndofs(Wₕ)) degrees of freedom."))
    elem = element(Wₕ)
    copyto!(elem, v)
    return elem
end

# Enable array-like indexing `uₕ[i]` for VectorElement.
@inline Base.@propagate_inbounds getindex(uₕ::VectorElement, i) = getindex(uₕ.data, i)
@inline Base.@propagate_inbounds setindex!(uₕ::VectorElement, val, i) = setindex!(uₕ.data, val, i)

# Create a new, uninitialized VectorElement with the same space as the input.
@inline Base.similar(uₕ::VectorElement) = element(space(uₕ))

# Broadcasting

# Enable broadcasting capabilities (e.g., uₕ .= vₕ .+ 1) for VectorElement.
Base.BroadcastStyle(::Type{<:VectorElement}) = Broadcast.ArrayStyle{VectorElement}()

# Define how to create a `similar` container for the broadcast result, preserving the space.
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}},
        ::Type{ElType}) where {ElType}
    vec_elem = _find_vec_in_broadcast(bc)
    vec_elem === nothing &&
        throw(ArgumentError("No VectorElement found in broadcast expression"))
    return VectorElement(similar(values(vec_elem), ElType), space(vec_elem))
end

# Version without a specified ElType.
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}})
    vec_elem = _find_vec_in_broadcast(bc)
    vec_elem === nothing &&
        throw(ArgumentError("No VectorElement found in broadcast expression"))
    return VectorElement(similar(values(vec_elem)), space(vec_elem))
end

"""
	_find_vec_in_broadcast(bc)

Internal helper to extract a [VectorElement](@ref) from a broadcast expression.

Recursively searches through the arguments of a broadcast expression tree to find
a `VectorElement` instance. This is used by the broadcasting machinery to determine
which function space should be used for the result.

# Arguments

  - `bc`: A broadcast expression, tuple of arguments, or individual value

# Returns

  - The first `VectorElement` found in the expression tree
  - `nothing` if no `VectorElement` is found

# Implementation Notes

Uses multiple dispatch to handle:

  - `Broadcasted` objects: Extract and search arguments
  - `Tuple`s: Recursively search each element
  - `VectorElement`: Return immediately (found!)
  - Other types: Return `nothing` and continue searching

This enables broadcasts like `uₕ .+ vₕ .* 2` to automatically preserve the space information.
"""
_find_vec_in_broadcast(bc::Broadcast.Broadcasted) = _find_vec_in_broadcast(bc.args)
function _find_vec_in_broadcast(args::Tuple)
    _find_vec_in_broadcast(_find_vec_in_broadcast(args[1]), Base.tail(args))
end
_find_vec_in_broadcast(x::VectorElement) = x
_find_vec_in_broadcast(::Any) = nothing # Not a VectorElement
_find_vec_in_broadcast(::Tuple{}) = nothing # End of recursion
_find_vec_in_broadcast(a::VectorElement, rest) = a # Found one
_find_vec_in_broadcast(::Any, rest) = _find_vec_in_broadcast(rest) # Keep searching

function Base.:*(uₕ::VectorElement, vₕ::NTuple{D, VectorElement}) where {D}
    zₕ = ntuple(i -> similar(vₕ[i]), D)
    for i in 1:D
        zₕ[i].data .= uₕ.data .* vₕ[i].data
    end
    return zₕ
end

function Base.:*(a::Number, vₕ::NTuple{D, VectorElement}) where {D}
    zₕ = ntuple(i -> similar(vₕ[i]), D)
    for i in 1:D
        zₕ[i].data .= a .* vₕ[i].data
    end
    return zₕ
end

@inline Base.:*(Vₕ::NTuple{D, VectorElement}, a::Number) where {D} = a * Vₕ
@inline Base.:*(Vₕ::NTuple{D, VectorElement}, uₕ::VectorElement) where {D} = uₕ * Vₕ
