# ghost_gridspace.jl: `GhostGridSpace`, the grid space over a `GhostMesh`: the physical
# degrees of freedom together with the ghost values beyond the named facets.
#
# The space holds the padded `ScalarGridSpace` (its mesh the `GhostMesh`, its weights
# physical-only) and a closure `(E, d)`: the padded values of a vector `u` of the space's
# degrees of freedom are `E u + d`. `gridspace(Ωₕ; ghosts)` builds the selection closure
# (`E` picks each degree of freedom's padded point, `d = 0`); a constrained space keeps the
# same fields with the relations' `E` and `d`.
#
# See also: `GhostMesh`, `ScalarGridSpace`, `SpaceWeights`

# Row replacement (`dirichlet`) on a ghost space, worked out once with the space: whether
# its reduced rows are its physical nodes in mesh order, and the physical nodes of each
# facet (facet `(d, s)` at `2(d - 1) + s`), so that a warm `assemble!` with `dirichlet`, and
# `dirichlet_bc!` and `symmetrize!` on the space, allocate nothing.
struct _RowsDirichlet
    physical::Bool
    facets::Vector{BitVector}
end

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
struct GhostGridSpace{D, T, S <: ScalarGridSpace{D, T}, ET <: AbstractMatrix{T},
    DT <: AbstractArray{T, 1}} <: AbstractSpaceType{1}
    "the space on the padded mesh, with physical-only weights."
    padded::S
    "the `N_pad × ndofs` closure matrix: the padded values are `E * u + d`."
    E::ET
    "the affine part of the closure, one entry per padded point."
    d::DT
    "the transpose of `E`, whose columns give `lift` its padded rows."
    Et::ET
    "row replacement on the space, for `dirichlet`."
    rows::_RowsDirichlet
end

function GhostGridSpace(P::ScalarGridSpace, E::AbstractMatrix, d::AbstractVector)
    G = mesh(P)
    Ωₕ = physical_mesh(G)
    rows = _RowsDirichlet(_rows_are_physical(G, E), _facet_masks(Ωₕ, Val(dim(Ωₕ))))
    return GhostGridSpace(P, E, d, _closure_transpose(E), rows)
end

@inline _closure_transpose(E::SparseMatrixCSC) = copy(transpose(E))

function _gridspace_ghosts(Ωₕ::Union{Mesh1D, MeshnD},
        ghosts::Union{Symbol, Tuple{Symbol, Vararg{Symbol}}})
    _check_submesh_sizes(Ωₕ)
    G = GhostMesh(Ωₕ; ghosts = ghosts)
    P = _gridspace(G, _ghost_space_weights(G))
    T = eltype(P)
    be = backend(Ωₕ)
    d = fill!(_backend_vector(be, T, npoints(G)), zero(T))
    return GhostGridSpace(P, _closure_matrix(be, T, _ghost_selection(G, T)), d)
end

# The closure matrix in the backend's matrix type. A type that is not a `SparseMatrixCSC`
# is left as built: forms on it are refused by `_check_ghost_host`.
function _closure_matrix(be, ::Type{T}, E::SparseMatrixCSC) where {T}
    MT = matrix_type(be)
    E = SparseMatrixCSC{T, Int}(E)
    return MT <: SparseMatrixCSC ? convert(MT, E) : E
end

# The affine part of a closure, in the backend's vector type.
@inline _closure_data(be, ::Type{T}, d::AbstractVector) where {T} = copyto!(
    _backend_vector(be, T, length(d)), d)

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
function _stale_weights_remedy(::GhostMesh)
    "Call gridspace(Ωₕ; ghosts = ...) again on the " *
    "physical mesh `Ωₕ` and take its padded_space."
end

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
(corners beyond two ghosted facets) hold 0. After `change_points!` of the mesh, `lift`
throws as the space's weights do: build the space again.

`lift(Wₕ, u)` runs in the bands of the mesh's execution policy, each padded point summed by
one band, so it gives the same values on every CPU policy. `Bramble.lift!(uₕ, Wₕ, u)`
refills a grid function `lift` returned, allocating nothing.

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
function lift(Wₕ::GhostGridSpace)
    _check_lift_fresh(Wₕ)
    return VectorElement(copy(Wₕ.d), Wₕ.padded)
end

function lift(Wₕ::GhostGridSpace, u::AbstractVector)
    _check_lift_fresh(Wₕ)
    length(u) == ndofs(Wₕ) || _throw_lift_length(:lift, Wₕ, u)
    U = similar(Wₕ.d, promote_type(eltype(u), eltype(Wₕ.d)))
    return VectorElement(_lift_values!(U, Wₕ, u), Wₕ.padded)
end

"""
    lift!(uₕ::VectorElement, Wₕ::GhostGridSpace, u::AbstractVector) -> uₕ

Write `E u + d` into the grid function `uₕ`, the in-place form of
[`lift`](@ref)`(Wₕ, u)`: `uₕ` lives on the padded space of `Wₕ`, as the result of `lift`
does, so a solve loop lifts each new solution without allocating. The values are those of
`lift(Wₕ, u)`, bit for bit, on every CPU policy.

`lift!` refuses a grid function on another space, or of an element type that cannot hold
the values (a real `uₕ` for a complex `u`), with an `ArgumentError`. It is public, not
exported: write `Bramble.lift!`.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)
Cₕ = constrained(gridspace(Ωₕ; ghosts = :boundary),
                 u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => 2.0))
a = form(Cₕ, Cₕ, (u, v) -> innerₕ(-1.0 * Bramble.D̃ₓ(Bramble.D₋ₓ(u)), v))
l = form(Cₕ, v -> innerₕ(x -> 1.0, v))
A, F = assemble(a, l)
uₕ = Bramble.lift(Cₕ, A \\ F)
assemble!(A, F, a, l)                  # the same system, refilled in place
Bramble.lift!(uₕ, Cₕ, A \\ F)
```
"""
function lift!(uₕ::VectorElement, Wₕ::GhostGridSpace, u::AbstractVector)
    _check_lift_fresh(Wₕ)
    length(u) == ndofs(Wₕ) || _throw_lift_length(:lift!, Wₕ, u)
    uₕ.space === Wₕ.padded || _throw_lift_target(Wₕ)
    T = promote_type(eltype(u), eltype(Wₕ.d))
    promote_type(eltype(uₕ), T) === eltype(uₕ) || _throw_lift_eltype(uₕ, T)
    _lift_values!(uₕ.data, Wₕ, u)
    return uₕ
end

# `U = E u + d` in the bands of the mesh's policy.
function _lift_values!(U::AbstractVector, Wₕ::GhostGridSpace, u::AbstractVector)
    d, Et = Wₕ.d, Wₕ.Et
    _run_bands!(execution_policy(Wₕ.padded), _lift_band!, U, d, SparseArrays.getcolptr(Et),
        rowvals(Et), nonzeros(Et), u)
    return U
end

@noinline function _throw_lift_target(Wₕ::GhostGridSpace)
    throw(
        ArgumentError(
        "lift!(uₕ, Wₕ, u) writes into a grid function on the padded space of $(Wₕ), and " *
        "uₕ lives on another space. Build it once with uₕ = Bramble.lift(Wₕ, u).",
    ),
    )
end

@noinline function _throw_lift_eltype(uₕ::VectorElement, T::Type)
    throw(
        ArgumentError(
        "lift!(uₕ, Wₕ, u) gives values of type $T, and uₕ holds $(eltype(uₕ)). Build uₕ " *
        "once with uₕ = Bramble.lift(Wₕ, u).",
    ),
    )
end

# Padded points `i` of band `b` of `nbands`: `U[i] = d[i] + Σ E[i, j] u[j]`, from column `i`
# of `Eᵀ`, in the order of `j`.
@noinline function _lift_band!(U, d, colptr, rowval, nz, u, nbands::Int, b::Int)
    T = eltype(U)
    @inbounds for i in _band_range(1:length(U), nbands, b)
        s = convert(T, d[i])
        for k in colptr[i]:(colptr[i + 1] - 1)
            s = convert(T, s + nz[k] * u[rowval[k]])
        end
        U[i] = s
    end
    return nothing
end

# Entries `r` of band `b` of `nbands` of `y = Mᵀ x`, `M` given by its CSC arrays: column `r`
# of `M` holds the terms of `y[r]`.
@noinline function _csc_columns_band!(y, colptr, rowval, nz, x, nbands::Int, b::Int)
    T = eltype(y)
    @inbounds for r in _band_range(1:length(y), nbands, b)
        s = zero(T)
        for k in colptr[r]:(colptr[r + 1] - 1)
            s = convert(T, s + nz[k] * x[rowval[k]])
        end
        y[r] = s
    end
    return nothing
end

# The closure was built from the mesh `Wₕ`'s weights were: after `change_points!` its ghost
# values solve the relations on a mesh that no longer exists. The padded space's own stale
# message would send the user to `padded_space`, which `lift` refuses.
@inline function _check_lift_fresh(Wₕ::GhostGridSpace)
    Wₕ.padded.weights.built_version == _mesh_version(mesh(Wₕ)) || _throw_lift_stale(Wₕ)
    return nothing
end

@noinline function _throw_lift_stale(Wₕ::GhostGridSpace)
    throw(
        ArgumentError(
        "lift refuses $(Wₕ): its mesh was changed in place (set_points!, change_points!, " *
        "or iterative_refinement!) after the space was built, so its ghost values would " *
        "solve the relations on a mesh that no longer exists. Call " *
        "gridspace(Ωₕ; ghosts = ...) again, and " *
        "constrained(gridspace(Ωₕ; ghosts = ...), relations) again for a constrained space.",
    ),
    )
end

@noinline function lift(Wₕ::GhostGridSpace, u, rest...)
    throw(
        ArgumentError(
        "lift takes the space and at most one vector, lift(Wₕ) or lift(Wₕ, u); got " *
        "$(1 + length(rest)) arguments after $(Wₕ). Drop the extra arguments.",
    ),
    )
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

@noinline function _throw_lift_length(name::Symbol, Wₕ::GhostGridSpace, u)
    call = name === :lift ? "lift(Wₕ, u)" : "lift!(uₕ, Wₕ, u)"
    throw(
        ArgumentError(
        "$call takes one value per degree of freedom of $(Wₕ), $(ndofs(Wₕ)) of " *
        "them; got $(length(u)). Pass the solution of a system assembled on Wₕ, not a " *
        "padded or physical-mesh vector.",
    ),
    )
end

# The restriction and cell-average operators refuse on a ghost space, in space and element
# form: whether a restriction holds values at the physical nodes only or at the ghost
# coordinates too is not decided. Without these methods `Rₕ!` falls to its general keyword
# method, which calls itself until the stack overflows, and `avgₕ!` reaches a `project!`
# with no method for the space.
Rₕ(Wₕ::GhostGridSpace, f; kwargs...) = _throw_projection_on_ghost_space(:Rₕ, Wₕ)
function Rₕ!(uₕ::VectorElement{<:GhostGridSpace}, f; kwargs...)
    return _throw_projection_on_ghost_space(:Rₕ!, uₕ.space)
end
avgₕ(Wₕ::GhostGridSpace, f; kwargs...) = _throw_projection_on_ghost_space(:avgₕ, Wₕ)
function avgₕ!(uₕ::VectorElement{<:GhostGridSpace}, f, rest...; kwargs...)
    return _throw_projection_on_ghost_space(:avgₕ!, uₕ.space)
end

@noinline function _throw_projection_on_ghost_space(op::Symbol, Wₕ::GhostGridSpace)
    throw(
        ArgumentError(
        "$op is not defined on a ghost space, $(Wₕ): what a restriction holds at the " *
        "ghost points is not decided yet. For the values of a function at the physical " *
        "nodes, apply $op on the plain space `gridspace(Ωₕ)`; for the values of a " *
        "constrained solution at the ghosts, use `Bramble.lift(Wₕ, u)`.",
    ),
    )
end
