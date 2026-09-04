#===========================================================================#
# The VectorElement interface.
#
# Accessors, the component views of a composite element, the constructors, and the
# array interface: indexing, `similar`, broadcasting and the tuple arithmetic. The two
# operators that produce a VectorElement from a function live beside this file, in
# restriction.jl and cell_average.jl.
#===========================================================================#

# Extends `Base.values` rather than defining a `Bramble.values` of its own. As a separate
# function it was a second binding exported under the same name, so `using Bramble` made
# `values` ambiguous with `Base.values` for the whole session: a user who then called
# `values(some_dict)` got an UndefVarError about two modules exporting different bindings.
# Extending Base is not piracy here, `VectorElement` being ours.
"""
    values(uₕ::VectorElement) -> AbstractVector

Returns the coefficient vector containing the degrees of freedom of [`VectorElement`](@ref) `uₕ`.
"""
@inline Base.values(uₕ::VectorElement) = uₕ.data

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
    values!(uₕ::VectorElement, s) -> uₕ

Copies the values of `s` into the coefficients of [`VectorElement`](@ref) `uₕ`. Returns `uₕ`.
"""
@inline function values!(uₕ::VectorElement, s)
    copyto!(values(uₕ), s)
    return uₕ
end

"""
    space(uₕ::VectorElement) -> AbstractSpaceType

Returns the grid space associated with [`VectorElement`](@ref) `uₕ`.
"""
@inline space(uₕ::VectorElement) = uₕ.space

"""
    space_type(::Type{<:VectorElement}) -> Type{<:AbstractSpaceType}
    space_type(uₕ::VectorElement) -> Type{<:AbstractSpaceType}

Returns the concrete [`AbstractSpaceType`](@ref) associated with a [`VectorElement`](@ref) type or instance.

# Examples

```julia
Wₕ = gridspace(Ωₕ)
uₕ = element(Wₕ)
space_type(typeof(uₕ)) === typeof(Wₕ)  # true
space_type(uₕ) === typeof(Wₕ)          # true
```

See also: [`space`](@ref), [`VectorElement`](@ref)
"""
@inline space_type(::Type{<:VectorElement{S}}) where {S} = S
@inline space_type(uₕ::VectorElement) = space_type(typeof(uₕ))

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
    component_range(Wₕ::CompositeGridSpace{N}, i::Int) where N -> UnitRange{Int}

Returns the degree-of-freedom index range for the `i`-th constituent space of composite space `Wₕ`.
"""
@inline function component_range(Wₕ::CompositeGridSpace{N}, i::Int) where {N}
    @boundscheck (1 <= i <= N) || throw(BoundsError(Wₕ, i))
    return @inbounds component_ranges(Wₕ)[i]
end

"""
    component_ranges(Wₕ::CompositeGridSpace{N}) where N -> NTuple{N, UnitRange{Int}}

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
    (uₕ::VectorElement)(i::Int) -> VectorElement
    component(uₕ::VectorElement, i::Int) -> VectorElement

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
    component(uₕ::VectorElement, i::Int) -> VectorElement

Extracts a [`VectorElement`](@ref) view of the `i`-th field component of `uₕ`. Alias for `uₕ(i)`.
"""
@inline component(uₕ::VectorElement, i::Int) = uₕ(i)

"""
    components(uₕ::VectorElement) -> Tuple

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
    element(Wₕ::AbstractSpaceType) -> VectorElement
    element(Wₕ::AbstractSpaceType, α::Number) -> VectorElement

Returns a [`VectorElement`](@ref) for grid space `Wₕ` with uninitialized components.
If `α` is provided, the components are initialized to `α`.
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

"""
    element(Wₕ::AbstractSpaceType, ::Type{T}) -> VectorElement

Returns a [`VectorElement`](@ref) for grid space `Wₕ` holding coefficients of type `T`,
with uninitialized components.

The coefficients of a grid function and the coordinates of the mesh under it are two
different things, and this is where they part company. `element(Wₕ)` takes its type from
the backend, which is the mesh's own; this takes whatever is asked for, and the container
follows through `similar`, so a `Vector` backend gives a `Vector{T}` and a device array
gives a device array of `T`.

The case this exists for is automatic differentiation: a `ForwardDiff.Dual` grid function
over an ordinary `Float64` mesh, so that the geometry is not differentiated along with the
field. See [`Rₕ`](@ref), which uses it to give back an element of whatever type the
restricted function returns.
"""
@inline function element(Wₕ::AbstractSpaceType, ::Type{T}) where {T}
    b = backend(Wₕ)
    # `similar` on an empty prototype rather than a `Vector{T}` literal, so the container
    # type stays the backend's and only the element type changes.
    v = similar(vector(b, 0), T, ndofs(Wₕ))
    return VectorElement{typeof(Wₕ), T, typeof(v)}(v, Wₕ)
end

# Constructor with a fill value `α`. The element type is promoted rather than taken from
# `α` outright, so `element(Wₕ, 2)` still gives a Float64 element on a Float64 backend
# while `element(Wₕ, dual)` gives a Dual one.
function element(Wₕ::AbstractSpaceType, α::Number)
    uₕ = element(Wₕ, promote_type(eltype(backend(Wₕ)), typeof(α)))
    fill!(uₕ, α)
    return uₕ
end

"""
    element(Wₕ::AbstractSpaceType, v::AbstractVector) -> VectorElement

Returns a [`VectorElement`](@ref) for a grid space `Wₕ` with the same coefficients as `v`.
"""
@inline function element(Wₕ::AbstractSpaceType, v::AbstractVector)
    # Ensure the provided vector has the correct number of DoFs.
    length(v) == ndofs(Wₕ) || throw(DimensionMismatch(
        "input vector has length $(length(v)), but the space has $(ndofs(Wₕ)) degrees of freedom."))
    elem = element(Wₕ, promote_type(eltype(backend(Wₕ)), eltype(v)))
    copyto!(elem, v)
    return elem
end

# Enable array-like indexing `uₕ[i]` for VectorElement.
@inline Base.@propagate_inbounds getindex(uₕ::VectorElement, i) = getindex(uₕ.data, i)
@inline Base.@propagate_inbounds setindex!(uₕ::VectorElement, val, i) = setindex!(uₕ.data, val, i)

# Create a new, uninitialized VectorElement with the same space as the input.
#
# The result keeps `uₕ`'s own coefficient type, not the space's. Every operator allocates
# its output with this, so going back to the backend's type here would mean a Dual-valued
# grid function differenced into a Float64 one, which fails on the first store. For an
# element built the ordinary way the two coincide, so nothing changes for a Float64 run.
@inline function Base.similar(uₕ::VectorElement)
    v = similar(values(uₕ))
    return VectorElement{typeof(space(uₕ)), eltype(v), typeof(v)}(v, space(uₕ))
end

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

Internal helper to extract a [`VectorElement`](@ref) from a broadcast expression.

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

# Both of these delegate to broadcasting rather than filling a tuple allocated up front.
#
# The type of `a * vₕ` is the type of the product, and `similar(vₕ[i])` gives the type of
# `vₕ[i]` alone; it copies the operand and drops the scalar. So a `Dual` scalar against a
# `Float64` element allocated `Vector{Float64}` and then threw on the first store:
#
#     julia> ForwardDiff.Dual{Nothing}(2.0, 1.0) * (uₕ, vₕ)
#     ERROR: MethodError: no method matching Float64(::ForwardDiff.Dual{...})
#
# `similar(::Broadcasted, ElType)` already computes the promoted type, and the scalar case
# needs no method here at all: a `VectorElement` is an `AbstractVector`, so Base's own
# `a * A == a .* A` covers it and has always promoted correctly. These two reimplemented
# that by hand for tuples and lost the promotion, deriving the element type from the space
# rather than the values.
#
# `vₕ[i] .* uₕ` rather than `uₕ .* vₕ[i]`: the broadcast picks the space off the first
# `VectorElement` it finds, and the code this replaces took it from `vₕ[i]`. The product is
# the same either way.
#
# `Val(D)` so the tuple is built unrolled, with `D` a static parameter rather than a count
# passed at run time.
@inline Base.:*(uₕ::VectorElement, vₕ::NTuple{D, VectorElement}) where {D} = ntuple(
    i -> vₕ[i] .* uₕ, Val(D))

@inline Base.:*(a::Number, vₕ::NTuple{D, VectorElement}) where {D} = ntuple(
    i -> a .* vₕ[i], Val(D))

@inline Base.:*(Vₕ::NTuple{D, VectorElement}, a::Number) where {D} = a * Vₕ
@inline Base.:*(Vₕ::NTuple{D, VectorElement}, uₕ::VectorElement) where {D} = uₕ * Vₕ
