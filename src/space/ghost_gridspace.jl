# ghost_gridspace.jl: `GhostGridSpace`, the grid space over a `GhostMesh`: the physical
# degrees of freedom together with the ghost values beyond the named facets.
#
# The space holds the padded `ScalarGridSpace` (its mesh the `GhostMesh`, its weights
# physical-only) and a closure `(E, d)`: the padded values of a vector `u` of the space's
# degrees of freedom are `E u + d`. `gridspace(Ωₕ; ghosts)` builds the selection closure
# (`E` picks each degree of freedom's padded point, `d = 0`); a constrained space keeps the
# same fields with the relations' `E` and `d`.
#
# DOF order: the physical nodes in the mesh's linear order, then the ghosts in the order of
# `GhostMesh` (facets `xmin, xmax, ymin, ymax, zmin, zmax`, named ones only, each facet's
# ghosts column-major). An absent padded point is never a degree of freedom.
#
# See also: `GhostMesh`, `ScalarGridSpace`, `SpaceWeights`

"""
    GhostGridSpace{D, T, S, ET, DT} <: AbstractSpaceType{1}

A scalar grid space with ghost points beyond the named facets of its mesh, built by
[`gridspace`](@ref)`(Ωₕ; ghosts = :boundary)` or `gridspace(Ωₕ; ghosts = (:xmin, ...))`.

Its degrees of freedom are the physical nodes of `Ωₕ`, in the mesh's linear order, then the
ghosts in the order of [`GhostMesh`](@ref). `padded_space(Wₕ)` is the `ScalarGridSpace` on
the padded mesh, the one forms assemble on: its `innerₕ`, `inner₊` and `inner_Γ` weigh the
physical nodes, cells and facets only, and are 0 at every ghost and absent point, so they
equal those of `gridspace(Ωₕ)` on the physical nodes. `closure(Wₕ)` is the pair `(E, d)`
that maps the degrees of freedom `u` to the padded values `E u + d`.

The weights follow the version of the mesh they were built from: after `change_points!` of
`Ωₕ` the ghost coordinates follow, and the padded space's weights throw as those of any
stale space; call `gridspace(Ωₕ; ghosts = ...)` again.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, false))
Wₕ = gridspace(Ωₕ; ghosts = :boundary)
ndofs(Wₕ)                     # 35 physical nodes + 24 ghosts
E, d = Bramble.closure(Wₕ)    # E is 63 × 59, d = 0
```
"""
struct GhostGridSpace{D, T, S <: ScalarGridSpace{D, T}, ET <: SparseMatrixCSC{T, Int},
    DT <: AbstractVector{T}} <: AbstractSpaceType{1}
    "the space on the padded mesh, with physical-only weights."
    padded::S
    "the `N_pad × ndofs` closure matrix: the padded values are `E * u + d`."
    E::ET
    "the affine part of the closure, one entry per padded point."
    d::DT
end

function _gridspace_ghosts(Ωₕ::Union{Mesh1D, MeshnD},
        ghosts::Union{Symbol, Tuple{Symbol, Vararg{Symbol}}})
    _check_submesh_sizes(Ωₕ)
    G = GhostMesh(Ωₕ; ghosts = ghosts)
    P = _gridspace(G, _ghost_space_weights(G))
    T = eltype(P)
    return GhostGridSpace(P, _ghost_selection(G, T), zeros(T, npoints(G)))
end

@noinline function _gridspace_ghosts(Ωₕ::AbstractMeshType, ghosts)
    throw(
        ArgumentError(
        "gridspace(Ωₕ; ghosts) takes `ghosts = ()`, `:boundary` or a tuple of facet " *
        "names such as `(:xmin, :ymax)` on a Mesh1D or MeshnD; got $(repr(ghosts)) on a " *
        "$(nameof(typeof(Ωₕ))). Pass facet names as a tuple of Symbols.",
    ),
    )
end

# A space on a `GhostMesh` itself would weigh its ghosts as nodes: the ghost space is built
# from the physical mesh, which holds its weights and closure.
@inline _gridspace_ghosts(G::GhostMesh, ::Tuple{}) = _throw_gridspace_on_ghost_mesh()
@inline _gridspace_ghosts(G::GhostMesh, ghosts) = _throw_gridspace_on_ghost_mesh()

@noinline function _throw_gridspace_on_ghost_mesh()
    throw(
        ArgumentError(
        "gridspace on a GhostMesh would weigh its ghost and absent points as nodes. Build " *
        "the space from the physical mesh instead, `gridspace(Ωₕ; ghosts = ...)` with " *
        "`Ωₕ = Bramble.physical_mesh(G)`, and assemble on its `Bramble.padded_space`.",
    ),
    )
end

# The remedy of the stale-weights error of a padded space: rebuild the ghost space from the
# physical mesh, not a space on the ghost mesh.
_stale_weights_remedy(::GhostMesh) = "Call gridspace(Ωₕ; ghosts = ...) again on the " *
                                     "physical mesh `Ωₕ` and take its padded_space."

# A per-axis factor of the physical mesh on the padded axis: 0 at the ghost end points.
function _pad_factor(f::AbstractVector, G::GhostMesh, d::Int)
    v = similar(f, npoints(G, Tuple)[d])
    fill!(v, zero(eltype(f)))
    copyto!(v, first(physical_indices(G).indices[d]), f, 1, length(f))
    return v
end

# The weights of the physical mesh on the padded grid: each per-axis factor in a vector of
# its own, the physical one at the physical points and 0 at the ghosts, so every tensor
# product is the physical weight at a physical node and 0 at a ghost or absent point.
function _ghost_space_weights(G::GhostMesh{D}) where {D}
    w = space_weights(physical_mesh(G))
    aligned = ntuple(d -> _pad_factor(w.aligned[d], G, d), Val(D))
    cellfactor = ntuple(d -> _pad_factor(w.cellfactor[d], G, d), Val(D))
    VT = typeof(first(aligned))
    T = eltype(VT)
    dims = npoints(G, Tuple)
    innerh = SeparableWeights{D, T, VT}(cellfactor, dims)
    innerplus = ntuple(Val(D)) do i
        SeparableWeights{D, T, VT}(ntuple(k -> k == i ? aligned[k] : cellfactor[k], Val(D)), dims)
    end
    return SpaceWeights{D, T, VT}(innerh, innerplus, aligned, cellfactor, _mesh_version(G))
end

# The selection closure: column `k` holds a single 1 at the padded point of DOF `k`.
function _ghost_selection(G::GhostMesh{D}, ::Type{T}) where {D, T}
    plin = LinearIndices(npoints(G, Tuple))
    phys = physical_indices(G)
    np = length(phys)
    n = np + nghosts(G)
    rows = Vector{Int}(undef, n)
    for (k, J) in enumerate(phys)
        rows[k] = plin[J]
    end
    for g in 1:nghosts(G)
        I, d, side = boundary_node(G, g)
        s = side === :min ? -1 : 1
        rows[np + g] = plin[padded_index(G, I) + CartesianIndex(ntuple(k -> k == d ? s : 0, Val(D)))]
    end
    return sparse(rows, collect(1:n), ones(T, n), length(plin), n)
end

"""
    padded_space(Wₕ::GhostGridSpace) -> ScalarGridSpace

The space on the padded mesh of `Wₕ`, with physical-only weights: the space forms assemble on.
"""
@inline padded_space(Wₕ::GhostGridSpace) = Wₕ.padded

"""
    closure(Wₕ::GhostGridSpace) -> (E::SparseMatrixCSC, d::Vector)

The closure of `Wₕ`: the padded values of the degrees of freedom `u` are `E * u + d`, `E`
of size `npoints(mesh(Wₕ)) × ndofs(Wₕ)` with rows in the padded linear order.
"""
@inline closure(Wₕ::GhostGridSpace) = (Wₕ.E, Wₕ.d)

@inline mesh(Wₕ::GhostGridSpace) = mesh(Wₕ.padded)
@inline mesh_type(::Type{<:GhostGridSpace{D, T, S}}) where {D, T, S} = mesh_type(S)
@inline mesh_type(Wₕ::GhostGridSpace) = mesh_type(typeof(Wₕ))
@inline backend(Wₕ::GhostGridSpace) = backend(Wₕ.padded)
@inline dim(::GhostGridSpace{D}) where {D} = D
@inline dim(::Type{<:GhostGridSpace{D}}) where {D} = D
@inline eltype(::GhostGridSpace{D, T}) where {D, T} = T
@inline eltype(::Type{<:GhostGridSpace{D, T}}) where {D, T} = T
@inline ndofs(Wₕ::GhostGridSpace) = size(Wₕ.E, 2)

function Base.show(io::IO, Wₕ::GhostGridSpace{D, T}) where {D, T}
    print(io, "GhostGridSpace{$(D)D, $T, ", ndofs(Wₕ), " dofs}")
    return nothing
end

Base.summary(Wₕ::GhostGridSpace) = sprint(show, Wₕ)

"""
    lift(Wₕ::GhostGridSpace) -> VectorElement
    lift(Wₕ::GhostGridSpace, u::AbstractVector) -> VectorElement

Grid functions on the padded space of `Wₕ` (`Bramble.padded_space(Wₕ)`) from its closure
`(E, d)` (`Bramble.closure(Wₕ)`).

`lift(Wₕ)` is the data `d`: on a space built by [`constrained`](@ref), the ghost values the
relations' data give when every degree of freedom is 0, and 0 everywhere else.
`lift(Wₕ, u)` is `E u + d`, the grid function of the degrees of freedom `u` (a solution of a
system assembled on `Wₕ`) with its eliminated ghosts filled in by the relations. The runtime
operators read the ghosts of the result, and both are data in forms on `Wₕ`, so a linear
form can name, say, the full `Δₕ` of the lifting at the facet nodes. Absent padded points
(corners beyond two ghosted facets) hold 0.

`lift` is public, not exported: write `Bramble.lift` or `using Bramble: lift`.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)
Cₕ = constrained(gridspace(Ωₕ; ghosts = :boundary),
                 u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => 2.0))
a = form(Cₕ, Cₕ, (u, v) -> innerₕ(-1.0 * Bramble.D̃ₓ(Bramble.D₋ₓ(u)), v))
l = form(Cₕ, v -> innerₕ(x -> 1.0, v))
A, F = assemble(a, l)
uₕ = Bramble.lift(Cₕ, A \\ F)          # 11 padded values, ghosts included
Δₕ(uₕ)                                 # reads the ghosts at the facet nodes
```
"""
@inline lift(Wₕ::GhostGridSpace) = VectorElement(copy(Wₕ.d), Wₕ.padded)

function lift(Wₕ::GhostGridSpace, u::AbstractVector)
    length(u) == ndofs(Wₕ) || _throw_lift_length(Wₕ, u)
    E, d = closure(Wₕ)
    U = similar(d, promote_type(eltype(u), eltype(d)))
    copyto!(U, d)
    mul!(U, E, u, true, true)
    return VectorElement(U, Wₕ.padded)
end

@noinline function lift(Wₕ, u...)
    throw(
        ArgumentError(
        "lift takes a space with ghosts, built by gridspace(Ωₕ; ghosts = ...) or " *
        "constrained, and optionally a vector of its degrees of freedom; got " *
        "$(nameof(typeof(Wₕ))). Build the space with ghosts first.",
    ),
    )
end

@noinline function lift(Wₕ::GhostGridSpace, u)
    throw(
        ArgumentError(
        "lift(Wₕ, u) takes a vector of the degrees of freedom of $(Wₕ); got a " *
        "$(nameof(typeof(u))). Pass the solution vector of a system assembled on Wₕ.",
    ),
    )
end

@noinline function _throw_lift_length(Wₕ::GhostGridSpace, u)
    throw(
        ArgumentError(
        "lift(Wₕ, u) takes one value per degree of freedom of $(Wₕ), $(ndofs(Wₕ)) of " *
        "them; got $(length(u)). Pass the solution of a system assembled on Wₕ, not a " *
        "padded or physical-mesh vector.",
    ),
    )
end
