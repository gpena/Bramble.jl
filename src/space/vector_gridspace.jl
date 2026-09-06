#=
# vector_gridspace.jl

This file defines composite and vector grid spaces (`CompositeGridSpace`, alias `VectorGridSpace`),
allowing multi-component fields (such as vector-valued velocities or coupled solution blocks)
to be constructed and queried with zero overhead.
=#

"""
    CompositeGridSpace(spaces::Tuple)
    CompositeGridSpace{N}(spaces::Spaces) where {N, Spaces <: Tuple}
    CompositeGridSpace{N, Spaces}(spaces::Spaces) where {N, Spaces <: Tuple}

A `CompositeGridSpace` represents a grid space formed by composing `N` individual sub-spaces.
It is immutable and stack-allocatable, wrapping a tuple of spaces.

# Fields

  - `spaces::Spaces`: the tuple of constituent sub-spaces.
"""
struct CompositeGridSpace{N, Spaces <: Tuple} <: AbstractSpaceType{N}
    "the tuple of constituent sub-spaces."
    spaces::Spaces
end

function CompositeGridSpace(spaces::Tuple)
    CompositeGridSpace{length(spaces), typeof(spaces)}(spaces)
end
function CompositeGridSpace{N}(spaces::Spaces) where {N, Spaces <: Tuple}
    CompositeGridSpace{N, Spaces}(spaces)
end
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
    gridspace(Ωₕ::AbstractMeshType, ::Val{N}) where N -> CompositeGridSpace{N}
    gridspace(Ωₕ::AbstractMeshType, N::Int) -> CompositeGridSpace{N}

Constructs a vector function space with `N` components on mesh `Ωₕ`.
The underlying scalar space and its weights are computed once and shared across components.

`N == 1` yields the [`ScalarGridSpace`](@ref) itself rather than a one-component
composite, for both spellings. The element interface is uniform either way:
`uₕ(1)` and `components(uₕ)` work on a scalar element.

The `Val` form is always type stable. The `Int` form is stable wherever `N` is a
literal or otherwise constant-foldable, and returns a small `Union` when `N` is
only known at run time; prefer `Val` on hot paths.
"""
function gridspace(Ωₕ::AbstractMeshType, ::Val{N}) where {N}
    W = gridspace(Ωₕ)
    return CompositeGridSpace(ntuple(_ -> W, Val(N)))
end

@inline gridspace(Ωₕ::AbstractMeshType, ::Val{1}) = gridspace(Ωₕ)

# Forwarding to the Val method keeps a single implementation and makes the two
# spellings agree by construction; :aggressive recovers type stability whenever
# the caller's N is a constant.
Base.@constprop :aggressive function gridspace(Ωₕ::AbstractMeshType, N::Int)
    N >= 1 || throw(ArgumentError("Number of components N must be >= 1, got $N"))
    return gridspace(Ωₕ, Val(N))
end

"""
    vector_gridspace(Ωₕ::AbstractMeshType, [N = dim(Ωₕ)]) -> CompositeGridSpace

Convenience constructor for a vector grid space on mesh `Ωₕ`. If `N` is omitted,
it defaults to the spatial dimension of the mesh (`dim(Ωₕ)`).
"""
@inline vector_gridspace(Ωₕ::AbstractMeshType) = gridspace(
    Ωₕ, Val(dim(Ωₕ)))
@inline vector_gridspace(Ωₕ::AbstractMeshType, ::Val{N}) where {N} = gridspace(
    Ωₕ, Val(N))
@inline vector_gridspace(Ωₕ::AbstractMeshType, N::Int) = gridspace(
    Ωₕ, N)

"""
    ^(Wₕ::ScalarGridSpace, ::Val{N}) where N -> CompositeGridSpace{N}
    ^(Wₕ::ScalarGridSpace, N::Int) -> CompositeGridSpace{N}

Constructs an `N`-component vector grid space from a scalar grid space `Wₕ` using mathematical exponentiation syntax:
`Vₕ = Wₕ^2` or `Vₕ = Wₕ^dim(mesh)`.

`Wₕ^1` is `Wₕ`, for both the `Int` and `Val` spellings.
"""
@inline Base.:^(Wₕ::ScalarGridSpace, ::Val{N}) where {N} = CompositeGridSpace(ntuple(_ -> Wₕ, Val(N)))
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
@inline eltype(::Type{<:CompositeGridSpace{
    <:Any, Spaces}}) where {Spaces} = eltype(fieldtype(Spaces, 1))
@inline backend(Wₕ::CompositeGridSpace) = backend(first_space(Wₕ))
@inline execution_policy(Wₕ::CompositeGridSpace) = execution_policy(backend(Wₕ))
@inline ndofs(Wₕ::CompositeGridSpace) = sum(ndofs, Wₕ.spaces)
# One entry per *component*, not per spatial dimension: see the warning on the
# `ndofs` stub (space/gridspace.jl) for why this is not interchangeable with the
# scalar method's `Tuple` form of the same name.
@inline ndofs(Wₕ::CompositeGridSpace, ::Type{Tuple}) = map(ndofs, Wₕ.spaces)

# No `weights(::CompositeGridSpace, ...)` method: leaves can have different meshes and
# therefore different weights, so there is no single vector that could correctly answer
# for the whole composite (gpena/Bramble.jl#67) — the same reason `normₕ`/`norm₊`
# (space/inner_product.jl) are typed for a `ScalarGridSpace` only. A composite raises the
# `MethodError` those already document as the contract; call `weights` on a `components`
# leaf instead.

@inline spaces(Wₕ::CompositeGridSpace) = Wₕ.spaces

"""
    _shares_one_mesh(comps::Tuple) -> Bool

Whether every leaf element in `comps` (a composite's `components(uₕ)`) sits on the exact
same mesh object.

A composite's leaves need not share a mesh — heterogeneous composites, whose leaves are
built over differently-sized meshes, are a supported pattern (see the interpolation
tutorial). When they do share one, a single evaluation of a vector-valued function at one
leaf's grid points is valid for every leaf, which is what the scatter paths of
[`Rₕ!`](@ref)/[`avgₕ!`](@ref) use this to decide; when they do not, each leaf needs its own
evaluation, since there is no shared "grid point i" across differently-sized meshes.
"""
@inline function _shares_one_mesh(comps::Tuple)
    Ωₕ = mesh(space(comps[1]))
    return all(c -> mesh(space(c)) === Ωₕ, comps)
end

# ==============================================================================
# Collection Interface
# ==============================================================================

@inline Base.getindex(Wₕ::CompositeGridSpace, i::Int) = Wₕ.spaces[i]
@inline Base.length(::CompositeGridSpace{N}) where {N} = N
@inline Base.firstindex(::CompositeGridSpace) = 1
@inline Base.lastindex(::CompositeGridSpace{N}) where {N} = N
@inline Base.iterate(Wₕ::CompositeGridSpace, state...) = iterate(Wₕ.spaces, state...)
@inline Base.eachindex(::CompositeGridSpace{N}) where {N} = 1:N
@inline Base.keys(::CompositeGridSpace{N}) where {N} = 1:N

# ==============================================================================
# Cartesian Product (×)
# ==============================================================================

# Overload product operator for space construction.
# Scalar × Scalar → 2-element flat composite.
# CompositeGridSpace × anything → hierarchical composite (no flattening),
# enabling forms like form(Vh × Wh, Vh × Wh, ((u,p),(v,q)) -> ...).
@inline ×(X::AbstractSpaceType, Y::AbstractSpaceType) = CompositeGridSpace((X, Y))

#===========================================================================#
# Walking a composite space's leaves
#
# A `CompositeGridSpace` may nest, so the scalar spaces underneath it form a tree. Several
# operations need that tree flattened together with each leaf's offset into the global vector
# of degrees of freedom: imposing Dirichlet conditions, and extracting the blocks of a
# coupled form.
#
# The result is a tuple, not a vector. Leaves may have different concrete types, so a
# `Vector` of them would become `Vector{Tuple{Any, Int}}`, forcing dynamic typing on every
# read through a leaf (mesh, dof count, marker mask). A tuple preserves each leaf's concrete
# type, unrolling the iteration and compiling to zero allocations.
#===========================================================================#

"""
    leaf_spaces_offsets(Wₕ) -> Tuple

The scalar spaces underneath `Wₕ` paired with their offsets into the global degree of
freedom vector, depth first and left to right, as a tuple of `(space, offset)`.

A scalar space is its own only leaf, at offset zero.
"""
@inline leaf_spaces_offsets(Wₕ) = first(_leaf_spaces_offsets(Wₕ, 0))

@inline _leaf_spaces_offsets(Wₕ::ScalarGridSpace, offset::Int) = (
    ((Wₕ, offset),), offset + ndofs(Wₕ))
@inline _leaf_spaces_offsets(Wₕ::CompositeGridSpace, offset::Int) = _leaves_of(
    Wₕ.spaces, offset)

@inline _leaves_of(::Tuple{}, offset::Int) = ((), offset)
@inline function _leaves_of(spaces::Tuple, offset::Int)
    head, next = _leaf_spaces_offsets(first(spaces), offset)
    tail, final = _leaves_of(Base.tail(spaces), next)
    return ((head..., tail...), final)
end

"""
    n_leaf_spaces(Wₕ) -> Int

The number of scalar spaces underneath `Wₕ`, counting through any nesting.
"""
@inline n_leaf_spaces(::ScalarGridSpace) = 1
@inline n_leaf_spaces(Wₕ::CompositeGridSpace) = sum(n_leaf_spaces, Wₕ.spaces)
