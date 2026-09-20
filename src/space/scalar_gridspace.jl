"""
    SeparableWeights{D, T, VT}(factors::NTuple{D, VT}, dims::NTuple{D, Int})

A lazy, separable weight vector over a `D`-dimensional grid: entry `I` (linear or
`CartesianIndex{D}`) is ``\\prod_{d=1}^D`` `factors[d][I[d]]`, computed on every access
rather than stored once for the whole grid.

Every family [`SpaceWeights`](@ref) offers is one of these (gpena/Bramble.jl#115, #234,
#115 again for the 100³-under-1MB target S6.8 set): `innerh` and each entry of
`innerplus` are built once, when the space is constructed, from the per-axis `aligned`
and `cellfactor` factors, and stored on `SpaceWeights` -- so reading them costs no more
than the field access plus whatever the caller's own indexing does. [`weights`](@ref)`(Wₕ,
Val(S))` for `|S| ≥ 2` builds a fresh one on every call instead, from those same two
tuples, since nothing yet asks for the same larger set twice in a hot loop.

A linear `getindex` converts to a `CartesianIndex` first (one division per axis); a caller
that already holds the `CartesianIndex` -- an assembly loop over `local_stencil`, for
instance -- should use it directly and skip that cost. `src/space/inner_product.jl`'s
`_dot`/`_dot_masked` do this for every weight family alike. `src/form/operators/inner.jl`'s
`compute_weight` does it only for `InnerPlusSet` (`|S| ≥ 2`); `InnerH` and `InnerPlus{Dim}`
still index by linear position, so a symbolic `innerₕ`/`inner₊ₓ`/etc. term inside a form now
pays that division per point during assembly, not only the two hot paths already routed
through the `CartesianIndex` -- measured in `docs/src/internals/space.md`.

# Fields

  - `factors::NTuple{D, VT}`: the per-axis vectors multiplied together at each index.
  - `dims::NTuple{D, Int}`: the grid shape, `npoints(Ωₕ, Tuple)`.

See also: [`weights`](@ref), [`SpaceWeights`](@ref).
"""
struct SeparableWeights{D, T, VT <: AbstractVector{T}} <: AbstractVector{T}
    "the per-axis vectors multiplied together at each index."
    factors::NTuple{D, VT}
    "the grid shape, `npoints(Ωₕ, Tuple)`."
    dims::NTuple{D, Int}
end

@inline Base.size(w::SeparableWeights) = (prod(w.dims),)
@inline Base.IndexStyle(::Type{<:SeparableWeights}) = IndexLinear()

# No `@boundscheck` here: a `CartesianIndex{D}` doesn't fit the generic `checkbounds`
# machinery for a 1-dimensional `AbstractVector` (its `size` is `(prod(dims),)`, not
# `dims`), and every caller with a `CartesianIndex` in hand already has it from iterating
# `CartesianIndices(dims)` -- the same trust `__prod`'s own callers already extend it.
@inline function Base.getindex(w::SeparableWeights{D}, I::CartesianIndex{D}) where {D}
    return __prod(w.factors, I)
end

@inline function Base.getindex(w::SeparableWeights{D}, li::Int) where {D}
    @boundscheck checkbounds(w, li)
    return @inbounds w[CartesianIndices(w.dims)[li]]
end

"""
    SpaceWeights(innerh::SeparableWeights, innerplus::NTuple{D}, aligned::NTuple{D, VT}, cellfactor::NTuple{D, VT}, built_version::Int)
    SpaceWeights{D, T, VT}(innerh, innerplus, aligned::NTuple{D, VT}, cellfactor::NTuple{D, VT}, built_version::Int)

Holds the diagonal weight vectors for a grid space's discrete inner products, both the
standard ``L^2`` weights and the staggered ones, precomputed once rather than recomputed
on every call.

# Fields

  - `innerh::SeparableWeights{D, T, VT}`: weight for the standard discrete ``L^2`` inner
    product (`:innerₕ`), based on cell measures (``|\\square_k|``). Built once, straight
    from `cellfactor` (no axis is in `S = ()`), and stored here rather than recomputed on
    every access -- it is a lazy per-axis product rather than a full-grid vector
    (gpena/Bramble.jl#115, #234), so `weights(Wₕ, Innerh())` costs the field read plus
    whatever the caller's own indexing does; see [`SeparableWeights`](@ref).
  - `innerplus::NTuple{D, SeparableWeights{D, T, VT}}`: one such lazy weight per spatial
    direction, for the modified, staggered inner products (`:inner₊ₓ`, `:inner₊ᵧ`, etc.).
  - `aligned::NTuple{D, VT}`: per-axis factor ``h_d(i)`` (length `npoints(Ωₕ, d)`, not the
    full grid) -- the same values `innerplus` uses on the axis aligned with its own
    difference direction. Kept so [`weights`](@ref)`(Wₕ, Val(S))` can build any staggered
    set from `O(D)` numbers instead of a fresh `O(n^D)` vector (gpena/Bramble.jl#115, #234).
  - `cellfactor::NTuple{D, VT}`: per-axis factor ``h_d(i+1/2)`` (also length
    `npoints(Ωₕ, d)`). This is `innerh`'s own per-axis cell measure -- `innerh`'s
    `factors` tuple *is* `cellfactor`, not a copy of it -- and doubles as every staggered
    direction's transverse factor: the two coincide for any axis with more than one point
    (both read the submesh's own cached half-spacings), so one vector serves both roles
    rather than two.
  - `built_version::Int`: the mesh's [`_mesh_version`](@ref) at the moment these weights
    were computed (gpena/Bramble.jl#221) -- [`weights`](@ref) re-checks it against the
    mesh's current version on every access, so a space built before an in-place mutation
    (`set_points!`, `change_points!`, `iterative_refinement!`) throws naming the mismatch
    rather than silently returning weights for a mesh that no longer exists.

For a detailed explanation of the mathematical formulas corresponding to these weights, please refer to the documentation for [`ScalarGridSpace`](@ref).
"""
struct SpaceWeights{D, T, VT <: AbstractVector{T}}
    "weight for the standard discrete ``L^2`` inner product (`:innerₕ`), based on cell measures (``|\\square_k|``)."
    innerh::SeparableWeights{D, T, VT}
    "one lazy weight per spatial direction, for the modified, staggered inner products (`:inner₊ₓ`, `:inner₊ᵧ`, etc.)."
    innerplus::NTuple{D, SeparableWeights{D, T, VT}}
    "per-axis aligned factor ``h_d(i)``, one vector of length `npoints(Ωₕ, d)` per axis."
    aligned::NTuple{D, VT}
    "per-axis cell-measure factor ``h_d(i+1/2)``, one vector of length `npoints(Ωₕ, d)` per axis; shared by `innerh` and every staggered direction's transverse factor."
    cellfactor::NTuple{D, VT}
    "the mesh's `_mesh_version` when these weights were built; see the staleness note above."
    built_version::Int
end

"""
    ScalarGridSpace(mesh::MType, weights::SpaceWeights{D, T, VT})
    ScalarGridSpace{D, T, VT, MType}(mesh::MType, weights::SpaceWeights{D, T, VT})

Represents a function space for **scalar fields** defined on a mesh.

A `ScalarGridSpace` pairs the mesh with the precomputed weight vectors
([`SpaceWeights`](@ref)) its discrete inner products need.

# Fields

  - `mesh::MType`: the underlying mesh of the grid space.
  - `weights::SpaceWeights{D, T, VT}`: precomputed inner product weight vectors.

## Discrete inner products

The `weights` object stores vectors for different discrete ``L^2`` inner products on the space of grid functions. They are defined as follows:

### - **`:innerₕ`**: The standard discrete ``L^2`` inner product, weighted by the cell measure ``|\\square_k|``.

  - **1D case:**

```math
(u_h, v_h)_h = \\sum_{i=1}^{N_x} |\\square_{i}| u_h(x_i) v_h(x_i)
```

  - **2D case:**

```math
(u_h, v_h)_h = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} |\\square_{i,j}| u_h(x_i,y_j) v_h(x_i,y_j)
```

  - **3D case:**

```math
(u_h, v_h)_h = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} |\\square_{i,j,l}| u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l)
```

Here, ``|\\cdot|`` denotes the measure of the set (length, area, or volume). See [`cell_measure`](@ref) for details.

### - **`:inner₊`, `:inner₊ₓ`, `:inner₊ᵧ`, `:inner₊₂`**: Modified discrete ``L^2`` inner products, weighted by a mix of forward/backward spacings (``h_k``) and cell widths (``h_{k+1/2}``).

  - **1D case (`:inner₊`):**

```math
(u_h, v_h)_+ = \\sum_{i=1}^{N_x} h_{i} u_h(x_i) v_h(x_i)
```

  - **2D case (`:inner₊ₓ`, `:inner₊ᵧ`):**

```math
(u_h, v_h)_{+x} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} h_{x,i} h_{y,j+1/2} u_h(x_i,y_j) v_h(x_i,y_j)
```

```math
(u_h, v_h)_{+y} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} h_{x,i+1/2} h_{y,j} u_h(x_i,y_j) v_h(x_i,y_j)
```

  - **3D case (`:inner₊ₓ`, `:inner₊ᵧ`, `:inner₊₂`):**

```math
(u_h, v_h)_{+x} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} h_{x,i} h_{y,j+1/2} h_{z,l+1/2} u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l)
```

```math
(u_h, v_h)_{+y} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} h_{x,i+1/2} h_{y,j} h_{z,l+1/2} u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l)
```

```math
(u_h, v_h)_{+z} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} h_{x,i+1/2} h_{y,j+1/2} h_{z,l} u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l)
```
"""
struct ScalarGridSpace{
    D,
    T,                               # Dimension and Element Type
    VT <: AbstractVector{T},             # Vector Type
    MType <: AbstractMeshType{D}
} <: AbstractSpaceType{1}
    "the underlying mesh of the grid space."
    mesh::MType
    "a [`SpaceWeights`](@ref) object holding vectors for various discrete inner products."
    weights::SpaceWeights{D, T, VT}
end

"""
    gridspace(Ωₕ::AbstractMeshType{D}) -> ScalarGridSpace{D}

Constructs a [`ScalarGridSpace`](@ref) defined on the mesh `Ωₕ`, precomputing the inner product
weights listed in [`ScalarGridSpace`](@ref).

Scratch memory is supplied explicitly by callers through in-place mutating operators
(such as `D₋ₓ!(vₕ, uₕ)`), avoiding hidden internal vector buffers.

Weights are a snapshot: they read `Ωₕ`'s grid at the moment of this call and are never
refreshed afterward. If `Ωₕ` is later mutated in place (`set_points!`, `change_points!`,
`iterative_refinement!`), the space this call returns keeps the *old* weights, and its
`innerₕ`/`inner₊*`/every norm then throw naming the mismatch rather than silently computing
against a mesh that no longer exists (gpena/Bramble.jl#221) -- call `gridspace(Ωₕ)` again to
get a space that reads the mutated mesh.

# Examples

```jldoctest
using Bramble
Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 11))
ndofs(Wₕ) == 11 && sum(weights(Wₕ, Bramble.Innerh())) ≈ 1.0

# output
true
```
"""
function gridspace(Ωₕ::AbstractMeshType{D}) where {D}
    return _gridspace(Ωₕ, space_weights(Ωₕ))
end

# Split out so `T`/`VT` are read off `weights`' own concrete type parameters (gpena/Bramble.jl#94,
# #174, S2.2 of .agents/plans/metal-and-apple-silicon-acceleration.md) rather than from
# `backend_types(backend(Ωₕ))`: on a Metal backend, `Backend{VT, MT, EP}`'s own `VT` (e.g.
# `MtlVector{Float32}`) leaves the storage-mode parameter free, but `vector`/`space_weights`
# allocate the concretely-typed `MtlVector{Float32, <storage>}` -- a different, invariant type
# parameter -- so building `ScalarGridSpace{D, T, VT, MType}` from the backend's `VT` failed to
# convert `weights` into its own field. Reading the parameters back off `weights` keeps the two
# in sync by construction, on every backend.
@inline function _gridspace(Ωₕ::AbstractMeshType{D}, weights::SpaceWeights{D, T, VT}) where {D, T, VT}
    return ScalarGridSpace{D, T, VT, typeof(Ωₕ)}(Ωₕ, weights)
end

# Allocates a work vector sized to a mesh. Typed rather than generic: with an
# untyped signature this also admits spaces, for which npoints has no method.
@inline __vector(Ωₕ::AbstractMeshType) = vector(backend(Ωₕ), npoints(Ωₕ))

# `innerh`/`innerplus` are lazy `SeparableWeights` for every `D`, including `D = 1`
# (gpena/Bramble.jl#115, #234, S6.8), rather than one dense `O(n)` vector for one dimension
# and a lazy `O(D)`-factor product from `D = 2` on: `SpaceWeights{D, T, VT}` fixes the
# field type at `SeparableWeights{D, T, VT}` for every `D`, and a second, dense-vector
# variant of that field would need a further type parameter on `SpaceWeights` (and, since
# `ScalarGridSpace` stores `SpaceWeights{D, T, VT}` in its own field type, on
# `ScalarGridSpace` too) purely to special-case one dimension. Measured instead of assumed
# (`docs/src/internals/space.md`): wrapping the 1D vector costs nothing detectable, since
# `SeparableWeights` with one factor degenerates to that same vector's own indexing after
# inlining. One dimension still skips building a *second* full-length vector for the
# transverse factor -- there is none, since `aligned` and `cellfactor` already *are*
# `innerplus₁` and `inner_h_vec` below, zero-copy.
function space_weights(Ωₕ::AbstractMeshType{1})
    innerplus₁ = __vector(Ωₕ)
    _innerplus_weights!(innerplus₁, Ωₕ, 1)

    inner_h_vec = __vector(Ωₕ)
    _innerh_weights!(inner_h_vec, Ωₕ)

    VT = typeof(inner_h_vec)
    T = eltype(VT)
    dims = npoints(Ωₕ, Tuple)

    return SpaceWeights{1, T, VT}(
        SeparableWeights{1, T, VT}((inner_h_vec,), dims),
        (SeparableWeights{1, T, VT}((innerplus₁,), dims),),
        (innerplus₁,), (inner_h_vec,), _mesh_version(Ωₕ)
    )
end

function space_weights(Ωₕ::AbstractMeshType{D}) where {D}
    # Per-axis factors, kept on `SpaceWeights` (not just used and discarded here) so
    # `weights(Wₕ, Val(S))` can answer any staggered set from `O(D)` numbers rather than a
    # fresh `O(n^D)` vector (gpena/Bramble.jl#115, #234):
    #   `aligned[k]`     applies to the axis aligned with the difference direction.
    #   `cellfactor[k]`  applies to every transverse axis, and is exactly the submesh's own
    #                    cell measures -- the two coincide for any axis with more than one
    #                    point, since both read the same cached half-spacings vector, so
    #                    this is a zero-copy reference rather than a second fill.
    aligned = ntuple(k -> __vector(Ωₕ(k)), Val(D))
    cellfactor = ntuple(k -> cell_measures(Ωₕ(k)), Val(D))
    for k in 1:D
        _innerplus_weights!(aligned[k], Ωₕ, k)
    end

    npts_tuple = npoints(Ωₕ, Tuple)
    VT = typeof(first(aligned))
    T = eltype(VT)

    # `innerh` and each `innerplus[i]` used to be filled, full grid, by
    # `_innerh_weights!`/`__innerplus_weights!` into a fresh `O(n^D)` vector apiece --
    # exactly the cost this milestone removes (gpena/Bramble.jl#115): `SeparableWeights`
    # answers the same values from `aligned`/`cellfactor` alone, computed at access time,
    # so `space_weights` itself never allocates more than the `2D` per-axis vectors above.
    innerh = SeparableWeights{D, T, VT}(cellfactor, npts_tuple)
    innerplus = ntuple(Val(D)) do i
        factors = ntuple(k -> k == i ? aligned[k] : cellfactor[k], Val(D))
        SeparableWeights{D, T, VT}(factors, npts_tuple)
    end

    return SpaceWeights{D, T, VT}(innerh, innerplus, aligned, cellfactor, _mesh_version(Ωₕ))
end

# Implementation of the interface functions for AbstractSpaceType
@inline mesh(Wₕ::ScalarGridSpace) = Wₕ.mesh
@inline backend(Wₕ::ScalarGridSpace) = backend(mesh(Wₕ))
@inline execution_policy(Wₕ::ScalarGridSpace) = execution_policy(backend(Wₕ))
@inline mesh_type(Wₕ::ScalarGridSpace) = typeof(mesh(Wₕ))
@inline mesh_type(::Type{<:ScalarGridSpace{<:Any, <:Any, <:Any, MType}}) where {MType} = MType

"""
    weights(Wₕ::ScalarGridSpace) -> SpaceWeights
    weights(Wₕ::ScalarGridSpace, ::Innerh) -> AbstractVector
    weights(Wₕ::ScalarGridSpace, ::Innerplus) -> NTuple{D, AbstractVector}
    weights(Wₕ::ScalarGridSpace, ::InnerProductType, i::Int) -> AbstractVector
    weights(Wₕ::ScalarGridSpace{D}, ::Val{S}) where {D, S} -> AbstractVector

Returns the precomputed weight vectors for discrete inner products.

The weights are diagonal matrices (stored as vectors) used in computing discrete
``L^2`` inner products. They represent cell measures or staggered grid spacings.

Every one of them is a [`SeparableWeights`](@ref) (gpena/Bramble.jl#115, #234): a lazy
per-axis product, not a full-grid vector, whatever `S` is asked for. `innerh` and each
`innerplus[d]` are built once, at `gridspace` construction time, and returned unchanged
here; the rest are built fresh on each call from the same per-axis factors.

# Methods

1. `weights(Wₕ)` - Returns the full [`SpaceWeights`](@ref) struct
2. `weights(Wₕ, Innerh())` - Returns weights for standard ``L^2`` inner product (cell volumes)
3. `weights(Wₕ, Innerplus())` - Returns tuple of weights for modified inner products (all directions)
4. `weights(Wₕ, Innerplus(), i)` - Returns weights for modified inner product in direction `i`
5. `weights(Wₕ, Innerh(), i)` - Same as `weights(Wₕ, Innerh())`; the cell measures do not
   depend on a direction, so `i` is accepted and ignored for interface symmetry
6. `weights(Wₕ, Val(S))` - Returns the weights for the staggered set `S ⊆ 1:D`, whose entry
   at grid index `I` is ``\\prod_{d \\in S} h_d(I_d) \\cdot \\prod_{d \\notin S} h_d(I_d +
   1/2)`` (gpena/Bramble.jl#115, #234). `S` is an `NTuple{K, Int}` with `K ≤ D` and any
   axis order; `weights(Wₕ, Val(()))` is `weights(Wₕ, Innerh())` and `weights(Wₕ,
   Val((d,)))` is `weights(Wₕ, Innerplus(), d)`, returning the very same object rather than
   a recomputed copy. Every other `S` returns a freshly-built [`SeparableWeights`](@ref).

# Examples

```julia
Wₕ = gridspace(Ωₕ)

# Get all weights
w = weights(Wₕ)  # Returns SpaceWeights{D, T, VT}

# Get standard L² weights
w_h = weights(Wₕ, Innerh())  # SeparableWeights, entry I = cell volume at I

# Get modified inner product weights for x-direction
w_plus_x = weights(Wₕ, Innerplus(), 1)  # SeparableWeights for x-direction

# Use in inner product
result = dot(uₕ.data, w_h, vₕ.data)  # Weighted inner product
```

Defined for a [`ScalarGridSpace`](@ref) only, the same rule [`normₕ`](@ref)/[`norm₊`](@ref)
follow: a composite grid space's leaves can have different meshes and therefore different
weights, so there is no single vector that could correctly answer for the whole composite.
A [`CompositeGridSpace`](@ref) raises a `MethodError`; take a scalar component of it with
[`components`](@ref) first.

# Staleness (gpena/Bramble.jl#221)

Every method here funnels through the one-argument form, which checks `Wₕ`'s stored
[`SpaceWeights`](@ref) against `mesh(Wₕ)`'s *current* [`_mesh_version`](@ref) and throws
naming the mismatch if an in-place mutator (`set_points!`, `change_points!`,
`iterative_refinement!`) has run on the mesh since these weights were built -- rather than
`innerₕ`/`inner₊*`/every norm silently computing against weights for a mesh that no longer
exists. Call [`gridspace`](@ref) again to get a space that reads the mutated mesh. On a
`CompositeGridSpace` this is checked per leaf, the moment `innerₕ`/etc. recurse into it --
there is no separate composite-level check to keep in step.

See also: [`SpaceWeights`](@ref), [`SeparableWeights`](@ref), [`Innerh`](@ref), [`Innerplus`](@ref), `innerₕ`
"""
@inline function weights(Wₕ::ScalarGridSpace)
    w = Wₕ.weights
    w.built_version == _mesh_version(mesh(Wₕ)) || _throw_stale_weights(Wₕ)
    return w
end
@inline weights(Wₕ::ScalarGridSpace, ::Innerh) = weights(Wₕ).innerh
@inline weights(Wₕ::ScalarGridSpace, ::Innerplus) = weights(Wₕ).innerplus
@inline weights(Wₕ::ScalarGridSpace, ::Innerh, i) = weights(Wₕ, Innerh())
@inline weights(Wₕ::ScalarGridSpace, ::Innerplus, i) = weights(Wₕ, Innerplus())[i]

@inline weights(Wₕ::ScalarGridSpace, ::Val{()}) = weights(Wₕ, Innerh())

@inline function weights(Wₕ::ScalarGridSpace, ::Val{S}) where {S}
    return _weights_val(Wₕ, Val(S), Val(length(S)))
end

@inline _weights_val(Wₕ::ScalarGridSpace{D}, ::Val{S}, ::Val{1}) where {D, S} = weights(Wₕ, Innerplus(), only(S))

@inline function _weights_val(Wₕ::ScalarGridSpace{D}, ::Val{S}, ::Val{K}) where {D, S, K}
    w = weights(Wₕ)
    factors = ntuple(d -> (d in S ? w.aligned[d] : w.cellfactor[d]), Val(D))
    VT = typeof(w.aligned[1])
    return SeparableWeights{D, eltype(VT), VT}(
        factors, npoints(mesh(Wₕ), Tuple)
    )
end

# Kept out of `weights` itself so the success path -- one integer comparison -- is all
# that is ever compiled inline there; the message (and `summary`, which walks the space's
# type parameters) is built only once a mismatch is already known to have happened.
@noinline function _throw_stale_weights(Wₕ::ScalarGridSpace)
    throw(
        ArgumentError(
        "$(summary(Wₕ))'s weights were computed from its mesh before an in-place " *
        "mutation (set_points!, change_points!, or iterative_refinement!) changed it -- " *
        "innerₕ/inner₊*/every norm through this space would silently use weights for a " *
        "mesh that no longer exists. Call gridspace(mesh(Wₕ)) again to get a space that " *
        "reads the mutated mesh.",
    ),
    )
end

"""
    dim(Wₕ::ScalarGridSpace) -> Int
    dim(::Type{<:ScalarGridSpace}) -> Int

Returns the spatial dimension of the function space (1, 2, or 3).

See also: [`ndofs`](@ref), [`mesh`](@ref)
"""
@inline dim(::ScalarGridSpace{D}) where {D} = D
@inline dim(::Type{<:ScalarGridSpace{D}}) where {D} = D

"""
    ndofs(Wₕ::ScalarGridSpace) -> Int
    ndofs(Wₕ::ScalarGridSpace, ::Type{Tuple}) -> NTuple{D, Int}

Returns the number of degrees of freedom (grid points) in the space.

# Methods

- `ndofs(Wₕ)` - Returns total number of DOFs as an integer
- `ndofs(Wₕ, Tuple)` - Returns DOFs per dimension as a tuple (Nₓ, Nᵧ, Nᵤ)

# Examples

```julia
Wₕ = gridspace(Ωₕ)
n = ndofs(Wₕ)        # Total DOFs (e.g., 10000 for 100×100 grid)
dims = ndofs(Wₕ, Tuple)  # Per dimension (e.g., (100, 100))
```

See also: [`npoints`](@ref), [`dim`](@ref). On a [`CompositeGridSpace`](@ref), the
`Tuple` form means something different: see the warning on [`ndofs`](@ref).
"""
@inline ndofs(Wₕ::ScalarGridSpace) = npoints(mesh(Wₕ))
@inline ndofs(Wₕ::ScalarGridSpace, ::Type{Tuple}) = npoints(mesh(Wₕ), Tuple)

"""
    eltype(Wₕ::ScalarGridSpace) -> Type
    eltype(::Type{<:ScalarGridSpace}) -> Type

Returns the element type of vectors in this space (e.g., `Float64`).

See also: [`backend`](@ref)
"""
@inline eltype(::ScalarGridSpace{D, T}) where {D, T} = T
@inline eltype(::Type{<:ScalarGridSpace{D, T}}) where {D, T} = T

"""
    _innerh_weights!(u, Ωₕ::AbstractMeshType)

Builds the weights for the standard discrete ``L^2`` inner product, ``inner_h(\\cdot, \\cdot)``, on the space of grid functions, following the order of the points provided by `indices(Ωₕ)`. The values are stored in vector `u`.
"""
function _innerh_weights!(u::Array, Ωₕ::AbstractMeshType{1})
    idxs = indices(Ωₕ)
    @inbounds @simd for idx in idxs
        i = idx[1]
        u[i] = cell_measure(Ωₕ, i)
    end
    return nothing
end

# Device counterpart (gpena/Bramble.jl#94, #174, S2.2 of
# .agents/plans/metal-and-apple-silicon-acceleration.md): `cell_measure(Ωₕ, i)` is
# `_apply_hs_logic(half_spacing(Ωₕ, i))` (src/mesh/mesh1d.jl:292-296), the same formula at
# every index with no boundary special case, so it needs no `@kernel` of its own -- it is
# `_apply_hs_logic` broadcast over the mesh's own `half_spacings` vector, which dispatches
# to a GPUArrays broadcast kernel instead of scalar `getindex`/`setindex!`.
function _innerh_weights!(u::AbstractVector, Ωₕ::AbstractMeshType{1})
    u .= _apply_hs_logic.(half_spacings(Ωₕ))
    return nothing
end

# No longer called by `space_weights` for `D ≥ 2` (gpena/Bramble.jl#115, S6.8): the
# `SeparableWeights` `innerh` there is `cellfactor` itself, needing no full-grid fill.
# Kept as the tested, directly-callable building block it always was.
function _innerh_weights!(u, Ωₕ::AbstractMeshType{D}) where {D}
    # The submeshes already hold these, so they are read rather than rebuilt: the
    # comprehension this replaces allocated one vector per axis on every call.
    cell_measures_per_component = ntuple(k -> cell_measures(Ωₕ(k)), Val(D))
    dims = npoints(Ωₕ, Tuple)
    v = Base.ReshapedArray(u, dims, ())
    __innerplus_weights!(execution_policy(Ωₕ), v, cell_measures_per_component)
    return nothing
end

"""
    _innerplus_weights!(u::VT, Ωₕ, component = 1) where VT

Builds a set of weights based on the spacings, associated with the `component`-th direction, for the modified discrete ``L^2`` inner product on the space of grid functions, following the order of the points provided by `indices(Ωₕ)`. The values are stored in vector `u`.
"""
function _innerplus_weights!(u::VT, Ωₕ, component = 1) where {VT <: Array}
    T = eltype(VT)
    mesh_component = Ωₕ(component)

    # These weights are the mesh's backward spacings with the first entry zeroed, which
    # the mesh now caches, so this is a copy rather than a call per point.
    copyto!(u, spacings(mesh_component))

    @inbounds u[1] = zero(T)
    return nothing
end

# Device counterpart: `copyto!` above is already bulk, so the only scalar operation is
# `u[1] = zero(T)`. Assigning into a one-element `view` keeps that write inside GPU
# broadcasting (`materialize!`) instead of a host-side scalar `setindex!`.
function _innerplus_weights!(u::AbstractVector, Ωₕ, component = 1)
    T = eltype(u)
    mesh_component = Ωₕ(component)

    copyto!(u, spacings(mesh_component))

    view(u, 1:1) .= zero(T)
    return nothing
end

"""
    _innerplus_mean_weights!(u::VT, Ωₕ, component::Int = 1) where VT

Builds a set of weights based on the half spacings, associated with the `component`-th direction, for the modified discrete ``L^2`` inner product on the space of grid functions, following the order of the [`points`](@ref). The values are stored in vector `u`.

# The two boundary entries (gpena/Bramble.jl#236)

`u[1]`/`u[N]` are `half_spacing(mesh_component, ·)` there -- the boundary half-cell width,
the same nonzero value [`cell_measure`](@ref) already uses for `innerₕ`'s own weight --
**not** zero. They used to be hand-zeroed, which matches neither the docstrings of
[`inner₊ₓ`](@ref)/[`inner₊ᵧ`](@ref)/[`inner₊₂`](@ref) (a sum over *every* transverse index,
boundary included) nor [`weights`](@ref)'s own, and deleted real quadrature weight: a node
on two or more boundary hyperplanes got zero weight from every staggered direction and was
absent from the assembled operator entirely -- the staggered Neumann Laplacian's kernel was
larger than the physical one, with identically-zero rows at those nodes, and a free-surface
elasticity stiffness matrix built the same way came out indefinite (negative eigenvalues),
not merely singular.

**Whether the zero was load-bearing for the discrete summation-by-parts identities was
checked, not assumed**, before this was changed: `test/space/sbp_identities.jl`'s own
property tests (1D/2D/3D, uniform and non-uniform meshes, Supposition-generated random
grids) all still hold, unmodified, against this weight. The reason is structural, not
coincidental -- every one of those identities is stated for a test function vanishing on
the boundary, and the boundary remainder a discrete integration by parts leaves behind is a
function of that test function's own boundary values, not of this weight. With the test
function already zero there, the remainder was already zero on its own account; the
boundary zeroing this weight used to carry was redundant with that, never the thing making
the identity hold. No boundary term needed writing out explicitly, because none was ever
implicit in the weight to begin with.

Fixing this also makes the natural (unconstrained, traction-free) Neumann boundary
condition `inner₊` encodes correct: a scalar Neumann Poisson MMS on the old weight
converged at order ≈1.04 (an inconsistent discretisation dressed as a working one); on this
weight it is a clean order 2, matching every other Neumann/traction-free path in this
package.
"""
function _innerplus_mean_weights!(u::VT, Ωₕ, component::Int = 1) where {VT <: Array}
    mesh_component = Ωₕ(component)
    N = npoints(mesh_component)

    @inbounds @simd for i in 1:N
        u[i] = half_spacing(mesh_component, i)
    end

    return nothing
end

# Device counterpart: `half_spacing(Ωₕ, i)` is a bounds-checked read of `Ωₕ.half_spacings[i]`
# with no boundary branch (src/mesh/mesh1d.jl:285-288), so this fill is exactly a copy of
# that backing vector -- a bulk `copyto!`, not a per-index scalar read.
function _innerplus_mean_weights!(u::AbstractVector, Ωₕ, component::Int = 1)
    mesh_component = Ωₕ(component)
    copyto!(u, half_spacings(mesh_component))
    return nothing
end

@inline function __prod(diags::NTuple{D, Any}, I) where {D}
    return prod(ntuple(i -> @inbounds(diags[i][I[i]]), Val(D)))
end

"""
    __innerplus_weights!(policy, v, innerplus_per_component)

Builds the weights for the modified discrete ``L^2`` inner product on the space of grid functions [`ScalarGridSpace`](@ref). The result is stored in vector `v`.
"""
function __innerplus_weights!(policy, v, innerplus_per_component)
    idxs = CartesianIndices(v)
    f = Base.Fix1(__prod, innerplus_per_component)
    return _sweep_for!(policy, v, idxs, f)
end

# --- Display ---------------------------------------------------------------------- #
#
# Neither grid space nor grid function had a `show` or `summary` method at all, so both
# fell through to Julia's default: `summary(gridspace(...))` was 563 characters of nested
# type parameters, and displaying one dumped every weight vector alongside it
# (gpena/Bramble.jl#17). Two-argument `show` is the embeddable one-liner;
# `MIME"text/plain"` is the detailed block (gpena/Bramble.jl#45).

function Base.show(io::IO, Wₕ::ScalarGridSpace{D, T}) where {D, T}
    print(io, "ScalarGridSpace{$(D)D, $T, ", ndofs(Wₕ), " dofs}")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", Wₕ::ScalarGridSpace{D, T}) where {D, T}
    return show_block(io) do io
        pp = PrettyPrinter(io)
        printstyled(io, "ScalarGridSpace"; bold = true, color = :cyan)
        print(io, " {")
        printstyled(io, "$(D)D"; color = :yellow)
        print(io, ", ")
        printstyled(io, "$T"; color = :yellow)
        println(io, "}:")

        pp_indented = with_indent(pp, 1)
        print_key_value(pp_indented, "Mesh", sprint(show, mesh(Wₕ)); separator = ": ")
        return print_key_value(pp_indented, "Dofs", string(ndofs(Wₕ)); separator = ": ")
    end
end

# `summary` is what an array of grid functions prints in its header and what `Base.show`
# for an `AbstractArray` reaches for; the default spelled out every type parameter.
Base.summary(Wₕ::ScalarGridSpace) = sprint(show, Wₕ)
