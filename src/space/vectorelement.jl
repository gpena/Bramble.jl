#===========================================================================#
# The VectorElement interface.
#
# Accessors, the component views of a composite element, the constructors, and the
# array interface: indexing, `similar`, broadcasting and the tuple arithmetic. The two
# operators that produce a VectorElement from a function live beside this file, in
# restriction.jl and cell_average.jl.
#===========================================================================#

# `Base.parent` is Julia's own name for "the storage a wrapper array delegates to" (see
# `parent(::SubArray)`, `parent(::ReshapedArray)`); `VectorElement` never defined it, so
# `parent(uₕ)` fell through to the generic `AbstractArray` fallback and returned `uₕ`
# itself rather than unwrapping anything -- the wrong answer for a wrapper type, silently,
# since nothing in this package called `parent` to notice. Replaces the former `values`
# accessor outright (gpena/Bramble.jl#73).
"""
    parent(uₕ::VectorElement) -> AbstractVector

Returns the coefficient vector containing the degrees of freedom of [`VectorElement`](@ref) `uₕ`.
"""
@inline Base.parent(uₕ::VectorElement) = uₕ.data

"""
    reshape(uₕ::VectorElement)

Reshapes the flat coefficient vector of `uₕ` into a multidimensional array that matches the logical layout of the grid points.

  - For a scalar space, this returns a D-dimensional array.
  - For an N-component vector space, it returns an N-tuple of arrays, one for each component.

This zero-argument form is specific to `VectorElement` and does not conflict with
`reshape(A, dims)`; it does shadow `Base.reshape(A) = reshape(A, ())`'s 0-dimensional
result for this type specifically, in favor of the shape a grid function actually has.
Replaces the former `to_matrix` outright (gpena/Bramble.jl#73).
"""
@inline Base.reshape(uₕ::VectorElement{<:ScalarGridSpace}) = Base.ReshapedArray(parent(uₕ), npoints(mesh(space(uₕ)), Tuple), ())
@inline Base.reshape(uₕ::VectorElement{<:CompositeGridSpace}) = map(reshape, components(uₕ))

# `values!` is gone outright (gpena/Bramble.jl#73): `copyto!(uₕ, s)` already does the same
# in-place copy through `VectorElement`'s `AbstractVector` interface, so it needed no
# replacement definition of its own.

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
#
# `size` is defined directly, not through `@forward`: the macro's generated
# `size(x::VectorElement, args...; kwargs...)` is as broad as `Base`'s own
# `size(t::AbstractArray, dim)` fallback (abstractarray.jl), so inserting it invalidated
# that fallback's cached `MethodInstance`s package-wide the moment Bramble loaded (#198).
# The zero-argument method below is all `size` ever needs from `VectorElement`: `Base`'s
# fallback already answers `size(uₕ, dim)` as `size(uₕ)[dim]` without a second definition.
@inline Base.size(uₕ::VectorElement) = size(uₕ.data)
@forward VectorElement.data (Bramble.show,)

# A VectorElement wraps a vector, so indexing is linear; without this the
# AbstractArray default of IndexCartesian() is used.
@inline Base.IndexStyle(::Type{<:VectorElement}) = IndexLinear()
@forward VectorElement.space (Bramble.mesh,)

# ==============================================================================
# Component Indexing
# ==============================================================================

"""
    component_range(Wₕ::CompositeGridSpace, i::Int) -> UnitRange{Int}

Returns the degree-of-freedom index range for the `i`-th **leaf** of composite space `Wₕ`,
numbered depth-first (see [`leaf_spaces_offsets`](@ref)) — the same numbering `uₕ(i)` and
[`dirichlet_components`](@ref dirichlet_bc!) use, so it agrees with them regardless of how
deeply `Wₕ` nests.
"""
@inline function component_range(Wₕ::CompositeGridSpace, i::Int)
    leaves = leaf_spaces_offsets(Wₕ)
    @boundscheck (1 <= i <= length(leaves)) || throw(BoundsError(Wₕ, i))
    sp, offset = @inbounds leaves[i]
    return (offset + 1):(offset + ndofs(sp))
end

"""
    component_ranges(Wₕ::CompositeGridSpace) -> NTuple{N, UnitRange{Int}}

`N` is the number of scalar leaves underneath `Wₕ`, counting through any nesting.

Returns the degree-of-freedom ranges for every **leaf** of `Wₕ`, depth-first — see
[`component_range`](@ref).
"""
@inline function component_ranges(Wₕ::CompositeGridSpace)
    # `map` over `leaf_spaces_offsets(Wₕ)` unrolls exactly as the old `ntuple(…, Val(N))`
    # did: the tuple it walks is built by compile-time recursion (leaf_spaces_offsets),
    # so its length and element types are known to the compiler, not just at runtime.
    return map(
        leaves_entry -> (leaves_entry[2] + 1):(leaves_entry[2] + ndofs(leaves_entry[1])),
        leaf_spaces_offsets(Wₕ)
    )
end

"""
    (uₕ::VectorElement)(i::Int) -> VectorElement

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
    leaves = leaf_spaces_offsets(space(uₕ))
    @boundscheck (1 <= i <= length(leaves)) || throw(BoundsError(uₕ, i))
    sp, offset = @inbounds leaves[i]
    v_data = @views parent(uₕ)[(offset + 1):(offset + ndofs(sp))]
    return VectorElement(v_data, sp)
end

@inline function (uₕ::VectorElement{<:ScalarGridSpace})(i::Int)
    @boundscheck i == 1 || throw(BoundsError(uₕ, i))
    return uₕ
end

"""
    components(uₕ::VectorElement) -> Tuple

Returns an `NTuple` of [`VectorElement`](@ref) views, one per **leaf** of `uₕ`'s space,
depth-first — the same leaf a matching-index `uₕ(i)` returns, regardless of nesting.
"""
@inline function components(uₕ::VectorElement{<:CompositeGridSpace})
    raw = parent(uₕ)
    return map(leaf_spaces_offsets(space(uₕ))) do (sp, offset)
        VectorElement(@views(raw[(offset + 1):(offset + ndofs(sp))]), sp)
    end
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
    length(v) == ndofs(Wₕ) || throw(
        DimensionMismatch(
        "input vector has length $(length(v)), but the space has $(ndofs(Wₕ)) degrees of freedom.",
    ),
    )
    elem = element(Wₕ, promote_type(eltype(backend(Wₕ)), eltype(v)))
    copyto!(elem, v)
    return elem
end

# ==============================================================================
# Indexing Interface
# ==============================================================================

# Enable array-like linear indexing `uₕ[i]` for VectorElement.
@inline Base.@propagate_inbounds Base.getindex(uₕ::VectorElement, i) = getindex(uₕ.data, i)
@inline Base.@propagate_inbounds function Base.setindex!(uₕ::VectorElement, val, i)
    setindex!(uₕ.data, val, i)
    return uₕ
end

# Multidimensional and Cartesian bounds checks
@inline function Base.checkbounds(
    ::Type{Bool}, uₕ::VectorElement{<:ScalarGridSpace{2}}, i, j
)
    return checkbounds(Bool, LinearIndices(indices(mesh(uₕ))), i, j)
end

@inline function Base.checkbounds(
    ::Type{Bool}, uₕ::VectorElement{<:ScalarGridSpace{3}}, i, j, k
)
    return checkbounds(Bool, LinearIndices(indices(mesh(uₕ))), i, j, k)
end

@inline function Base.checkbounds(
    ::Type{Bool}, uₕ::VectorElement{<:ScalarGridSpace{D}}, I::CartesianIndex{D}
) where {D}
    return checkbounds(Bool, LinearIndices(indices(mesh(uₕ))), I)
end

@inline function Base.checkbounds(
    ::Type{Bool}, uₕ::VectorElement{<:ScalarGridSpace}, I::CartesianIndex
)
    return checkbounds(Bool, LinearIndices(indices(mesh(uₕ))), I)
end

"""
    getindex(uₕ::VectorElement{<:ScalarGridSpace{2}}, i::Integer, j::Integer)
    getindex(uₕ::VectorElement{<:ScalarGridSpace{3}}, i::Integer, j::Integer, k::Integer)
    getindex(uₕ::VectorElement{<:ScalarGridSpace{D}}, I::CartesianIndex{D}) where {D}
    getindex(uₕ::VectorElement{<:ScalarGridSpace}, I::CartesianIndex)

Access field degrees of freedom by spatial grid coordinates or `CartesianIndex`.

Translates spatial grid coordinates directly into flat linear coefficient offsets using the
mesh's `LinearIndices` with zero heap allocations and full `@inbounds` transparency.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (10, 10))
Wₕ = gridspace(Ωₕ)
uₕ = element(Wₕ, 0.0)

# Set and get via 2D coordinates
uₕ[2, 3] = 42.0
uₕ[2, 3] == 42.0

# Access via CartesianIndex
I = CartesianIndex(2, 3)
uₕ[I] == 42.0
```

See also: [`VectorElement`](@ref), [`ScalarGridSpace`](@ref), [`reshape`](@ref)
"""
@inline Base.@propagate_inbounds function Base.getindex(
    uₕ::VectorElement{<:ScalarGridSpace{2}}, i::Integer, j::Integer
)
    @boundscheck checkbounds(uₕ, i, j)
    li = LinearIndices(indices(mesh(uₕ)))
    return @inbounds uₕ.data[li[i, j]]
end

@inline Base.@propagate_inbounds function Base.getindex(
    uₕ::VectorElement{<:ScalarGridSpace{3}}, i::Integer, j::Integer, k::Integer
)
    @boundscheck checkbounds(uₕ, i, j, k)
    li = LinearIndices(indices(mesh(uₕ)))
    return @inbounds uₕ.data[li[i, j, k]]
end

@inline Base.@propagate_inbounds function Base.getindex(
    uₕ::VectorElement{<:ScalarGridSpace{D}}, I::CartesianIndex{D}
) where {D}
    @boundscheck checkbounds(uₕ, I)
    li = LinearIndices(indices(mesh(uₕ)))
    return @inbounds uₕ.data[li[I]]
end

@inline Base.@propagate_inbounds function Base.getindex(
    uₕ::VectorElement{<:ScalarGridSpace}, I::CartesianIndex
)
    @boundscheck checkbounds(uₕ, I)
    li = LinearIndices(indices(mesh(uₕ)))
    return @inbounds uₕ.data[li[I]]
end

"""
    setindex!(uₕ::VectorElement{<:ScalarGridSpace{2}}, val, i::Integer, j::Integer) -> VectorElement
    setindex!(uₕ::VectorElement{<:ScalarGridSpace{3}}, val, i::Integer, j::Integer, k::Integer) -> VectorElement
    setindex!(uₕ::VectorElement{<:ScalarGridSpace{D}}, val, I::CartesianIndex{D}) where {D} -> VectorElement
    setindex!(uₕ::VectorElement{<:ScalarGridSpace}, val, I::CartesianIndex) -> VectorElement

Mutate field degrees of freedom by spatial grid coordinates or `CartesianIndex` in-place.

Translates spatial grid coordinates directly into flat linear coefficient offsets using the
mesh's `LinearIndices` with zero heap allocations and full `@inbounds` transparency.

Returns `uₕ` matching Base collection conventions.
"""
@inline Base.@propagate_inbounds function Base.setindex!(
    uₕ::VectorElement{<:ScalarGridSpace{2}}, val, i::Integer, j::Integer
)
    @boundscheck checkbounds(uₕ, i, j)
    li = LinearIndices(indices(mesh(uₕ)))
    @inbounds uₕ.data[li[i, j]] = val
    return uₕ
end

@inline Base.@propagate_inbounds function Base.setindex!(
    uₕ::VectorElement{<:ScalarGridSpace{3}}, val, i::Integer, j::Integer, k::Integer
)
    @boundscheck checkbounds(uₕ, i, j, k)
    li = LinearIndices(indices(mesh(uₕ)))
    @inbounds uₕ.data[li[i, j, k]] = val
    return uₕ
end

@inline Base.@propagate_inbounds function Base.setindex!(
    uₕ::VectorElement{<:ScalarGridSpace{D}}, val, I::CartesianIndex{D}
) where {D}
    @boundscheck checkbounds(uₕ, I)
    li = LinearIndices(indices(mesh(uₕ)))
    @inbounds uₕ.data[li[I]] = val
    return uₕ
end

@inline Base.@propagate_inbounds function Base.setindex!(
    uₕ::VectorElement{<:ScalarGridSpace}, val, I::CartesianIndex
)
    @boundscheck checkbounds(uₕ, I)
    li = LinearIndices(indices(mesh(uₕ)))
    @inbounds uₕ.data[li[I]] = val
    return uₕ
end

# Create a new, uninitialized VectorElement with the same space as the input.
#
# The result keeps `uₕ`'s own coefficient type, not the space's. Every operator allocates
# its output with this, so going back to the backend's type here would mean a Dual-valued
# grid function differenced into a Float64 one, which fails on the first store. For an
# element built the ordinary way the two coincide, so nothing changes for a Float64 run.
@inline function Base.similar(uₕ::VectorElement)
    v = similar(parent(uₕ))
    return VectorElement{typeof(space(uₕ)), eltype(v), typeof(v)}(v, space(uₕ))
end

# Broadcasting

# Enable broadcasting capabilities (e.g., uₕ .= vₕ .+ 1) for VectorElement.
Base.BroadcastStyle(::Type{<:VectorElement}) = Broadcast.ArrayStyle{VectorElement}()

# Define how to create a `similar` container for the broadcast result, preserving the space.
function Base.similar(
        bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}}, ::Type{ElType}
) where {ElType}
    vec_elem = _find_vec_in_broadcast(bc)
    vec_elem === nothing &&
        throw(ArgumentError("No VectorElement found in broadcast expression"))
    return VectorElement(similar(parent(vec_elem), ElType), space(vec_elem))
end

# Version without a specified ElType.
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}})
    vec_elem = _find_vec_in_broadcast(bc)
    vec_elem === nothing &&
        throw(ArgumentError("No VectorElement found in broadcast expression"))
    return VectorElement(similar(parent(vec_elem)), space(vec_elem))
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
    return _find_vec_in_broadcast(_find_vec_in_broadcast(args[1]), Base.tail(args))
end
_find_vec_in_broadcast(x::VectorElement) = x
_find_vec_in_broadcast(::Any) = nothing # Not a VectorElement
_find_vec_in_broadcast(::Tuple{}) = nothing # End of recursion
_find_vec_in_broadcast(a::VectorElement, rest) = a # Found one
_find_vec_in_broadcast(::Any, rest) = _find_vec_in_broadcast(rest) # Keep searching

# `dest .= expr` lowers to `copyto!(dest, bc)` and, absent a specialised method, runs
# Base's generic `AbstractArray` fallback: it indexes `bc[i]` on every iteration, and each
# `VectorElement` leaf re-enters `getindex`/`checkbounds` there, hiding the contiguous loop
# over `dest.data` from the compiler. Unwrapping every `VectorElement` leaf down to its own
# `parent` before delegating lets `copyto!` run directly against the backend's own storage
# instead -- the plain `Vector` broadcast loop for the default backend, whatever loop a
# GPU backend's own array type provides otherwise (gpena/Bramble.jl#181).
@inline function Base.copyto!(
        dest::VectorElement, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{VectorElement}}
)
    _check_broadcast_space(space(dest), bc)
    copyto!(parent(dest), _unwrap_broadcast(bc))
    return dest
end

# Rebuild the same expression tree over each `VectorElement` leaf's own storage. The
# reconstructed `Broadcasted` carries no `VectorElement` among its args, so its own style
# is computed fresh from plain arrays/scalars -- ordinary array broadcasting, not this
# package's.
@inline _unwrap_broadcast(bc::Broadcast.Broadcasted) = Broadcast.Broadcasted(bc.f, map(_unwrap_broadcast, bc.args), bc.axes)
@inline _unwrap_broadcast(x::VectorElement) = parent(x)
@inline _unwrap_broadcast(x) = x

# Every `VectorElement` operand must share `dest`'s grid space: mixing spaces would
# silently index one operand's coefficients against another mesh's layout. One check
# before the loop, not per element -- mirrors `_find_vec_in_broadcast`'s tuple recursion.
@inline _check_broadcast_space(Wₕ, bc::Broadcast.Broadcasted) = _check_broadcast_space(Wₕ, bc.args)
@inline function _check_broadcast_space(Wₕ, args::Tuple)
    _check_broadcast_space(Wₕ, args[1])
    return _check_broadcast_space(Wₕ, Base.tail(args))
end
@inline _check_broadcast_space(Wₕ, ::Tuple{}) = nothing
@inline function _check_broadcast_space(Wₕ, x::VectorElement)
    space(x) === Wₕ || _throw_broadcast_space_mismatch()
    return nothing
end
@inline _check_broadcast_space(Wₕ, ::Any) = nothing

@noinline _throw_broadcast_space_mismatch() = throw(
    ArgumentError("VectorElement operands in a broadcast must share the same grid space"),
)

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
@inline Base.:*(uₕ::VectorElement, vₕ::NTuple{D, VectorElement}) where {D} = ntuple(i -> vₕ[i] .* uₕ, Val(D))

@inline Base.:*(a::Number, vₕ::NTuple{D, VectorElement}) where {D} = ntuple(i -> a .* vₕ[i], Val(D))

@inline Base.:*(Vₕ::NTuple{D, VectorElement}, a::Number) where {D} = a * Vₕ
@inline Base.:*(Vₕ::NTuple{D, VectorElement}, uₕ::VectorElement) where {D} = uₕ * Vₕ

"""
    f::Function * uₕ::VectorElement -> VectorElement
    uₕ::VectorElement * f::Function -> VectorElement

Project the continuous function `f` onto `uₕ`'s own space and scale `uₕ` pointwise by it
(gpena/Bramble.jl#197): `Rₕ(space(uₕ), f) .* uₕ`.

A plain `Function` has no meaning as a grid function on its own -- a form built from
`innerₕ(f, v)` restricts it first through [`source_function`](@ref)/[`form`](@ref)'s own
lowering, but `f * uₕ` outside a form (or a continuous spatial *condition* multiplying a
grid function, `(x -> x[1] < 1) * uₕ`) had no operator to reach that with:
```julia
julia> (x -> x[1] < 1) * uₕ
ERROR: MethodError: no method matching *(::Function, ::VectorElement)
```
now restricts `f` to `space(uₕ)` and multiplies elementwise, so the result is an ordinary
`VectorElement`, usable anywhere one is -- including as a `SourceFunction`-lowered term
inside another form.
"""
@inline Base.:*(f::Function, uₕ::VectorElement) = Rₕ(space(uₕ), f) .* uₕ
@inline Base.:*(uₕ::VectorElement, f::Function) = f * uₕ

# --- Display ---------------------------------------------------------------------- #
#
# `VectorElement` is an `AbstractVector`, so without a `summary` of its own it inherited
# the default one: 615 characters of nested type parameters in the header above its values
# (gpena/Bramble.jl#17).

function Base.show(io::IO, uₕ::VectorElement)
    print(
        io,
        "VectorElement{$(dim(space(uₕ)))D, $(eltype(uₕ)), ",
        length(parent(uₕ)),
        " dofs}"
    )
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", uₕ::VectorElement)
    return show_block(io) do io
        pp = PrettyPrinter(io)
        printstyled(io, "VectorElement"; bold = true, color = :cyan)
        print(io, " {")
        printstyled(io, "$(dim(space(uₕ)))D"; color = :yellow)
        print(io, ", ")
        printstyled(io, "$(eltype(uₕ))"; color = :yellow)
        println(io, "}:")

        pp_indented = with_indent(pp, 1)
        print_key_value(pp_indented, "Space", sprint(show, space(uₕ)); separator = ": ")
        # The values themselves, through the backing array's own display, which already
        # abbreviates a long vector rather than printing every entry.
        print_indent(pp_indented)
        printstyled(io, "Values"; color = :green)
        print(io, ": ")
        return print(IOContext(io, :compact => true, :limit => true), parent(uₕ))
    end
end

Base.summary(uₕ::VectorElement) = sprint(show, uₕ)
