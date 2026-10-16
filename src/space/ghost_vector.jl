# ghost_vector.jl: `GhostVector`, the facet-only storage of a ghost space's values in a
# time loop: the physical values together with the ghost values beyond the named facets,
# `update_ghosts!` refreshing the latter from the former, and `physical`/`ghost_values`
# reading the two parts (gpena/Bramble.jl#776).
#
# The physical part is the caller's vector itself, so a solver's state needs no copy; the
# ghosts are a buffer in the order of `GhostMesh` (`boundary_node`). `update_ghosts!` reads
# the space's `_GhostUpdate` (ghost_gridspace.jl), the eliminated ghosts' rows of `E` in
# this order, and sums them in the bands of the mesh's policy, one band per run of
# eliminated ghosts: facet-sized work, whatever the size of the mesh.
#
# See also: `GhostGridSpace`, `GhostMesh`, `lift`

"""
    GhostVector{T, VP, VG, G} <: AbstractVector{T}

The values of a grid function on a ghost space: the physical values, the very vector the
solver holds, and the values at the ghosts beyond the named facets, kept in a buffer of
their own.

`GhostVector(Wₕ, u)` takes the vector `u` of the values at the physical nodes of the mesh of
`Wₕ`, in the mesh's linear order, and holds `u` itself as its physical part, not a copy: a
time loop that writes its state into `u` sees it in the `GhostVector`. The ghost buffer has
one entry per ghost of `mesh(Wₕ)`, in the ghost order of [`GhostMesh`](@ref) (the facets
`xmin`, `xmax`, `ymin`, …, each facet's nodes in the mesh's column-major order), and starts
at 0; [`update_ghosts!`](@ref) fills it from the relations of a constrained space.
`GhostVector(Wₕ)` starts from zeros, its physical part a vector of the backend of `Wₕ`.
`Wₕ` is any space with ghosts, as built by `gridspace(Ωₕ; ghosts = ...)` or
[`constrained`](@ref).

A `GhostVector` is an `AbstractVector` of length `np + ng`: the `np` physical values, then the
`ng` ghosts, the order of the degrees of freedom of the unconstrained ghost space
`gridspace(Ωₕ; ghosts = ...)`. `Bramble.physical(v)` and `Bramble.ghost_values(v)` return
the two parts. It carries the mesh with ghosts, `mesh(Wₕ)`, and `update_ghosts!` refuses a
space on another mesh.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, false))
Cₕ = constrained(gridspace(Ωₕ; ghosts = :boundary),
                 u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => 0.0,
                       :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0))
u = rand(35)                    # the values at the 5 × 7 physical nodes
v = GhostVector(Cₕ, u)          # Bramble.physical(v) === u
update_ghosts!(v, Cₕ)           # the 24 ghosts the relations give
length(v)                       # 59
```
"""
struct GhostVector{T, VP <: AbstractArray{T, 1}, VG <: AbstractArray{T, 1}, G <: GhostMesh} <:
       AbstractVector{T}
    "the values at the physical nodes, in the mesh's linear order: the caller's vector."
    physical::VP
    "the values at the ghosts, in the ghost order of the mesh."
    ghosts::VG
    "the mesh with ghosts the values live on."
    mesh::G
end

function GhostVector(Wₕ::GhostGridSpace, u::AbstractVector)
    G = mesh(Wₕ)
    np, ng = length(physical_indices(G)), nghosts(G)
    length(u) == np || _throw_ghost_vector_length(Wₕ, np, u)
    T = eltype(u)
    g = fill!(_backend_vector(backend(Wₕ), T, ng), zero(T))
    return GhostVector(u, g, G)
end

function GhostVector(Wₕ::GhostGridSpace)
    T = eltype(Wₕ)
    u = fill!(_backend_vector(backend(Wₕ), T, length(physical_indices(mesh(Wₕ)))), zero(T))
    return GhostVector(Wₕ, u)
end

@noinline function GhostVector(Wₕ, u...)
    throw(
        ArgumentError(
        "GhostVector takes a space with ghosts, built by gridspace(Ωₕ; ghosts = ...) or " *
        "constrained, and optionally the vector of its physical values; got " *
        "$(nameof(typeof(Wₕ))) and $(length(u)) more argument(s). Build the space with " *
        "ghosts first.",
    ),
    )
end

@noinline function _throw_ghost_vector_length(Wₕ::GhostGridSpace, np::Int, u)
    throw(
        ArgumentError(
        "GhostVector(Wₕ, u) takes the values at the $np physical nodes of the mesh of " *
        "$(Wₕ); got a vector of length $(length(u)). Pass the physical values only, in " *
        "the mesh's linear order (not a vector of the space's degrees of freedom).",
    ),
    )
end

Base.size(v::GhostVector) = (length(v.physical) + length(v.ghosts),)
Base.IndexStyle(::Type{<:GhostVector}) = IndexLinear()

@inline function Base.getindex(v::GhostVector, i::Int)
    @boundscheck checkbounds(v, i)
    np = length(v.physical)
    return @inbounds i <= np ? v.physical[i] : v.ghosts[i - np]
end

@inline function Base.setindex!(v::GhostVector, x, i::Int)
    @boundscheck checkbounds(v, i)
    np = length(v.physical)
    @inbounds i <= np ? (v.physical[i] = x) : (v.ghosts[i - np] = x)
    return v
end

"""
    physical(v::GhostVector) -> AbstractVector

The values of `v` at the physical nodes, in the mesh's linear order: the vector `v` was built
from, not a copy. `physical` is public, not exported: write `Bramble.physical`.
"""
@inline physical(v::GhostVector) = v.physical

"""
    ghost_values(v::GhostVector) -> AbstractVector

The values of `v` at the ghosts, one per ghost of its mesh in the ghost order of
[`GhostMesh`](@ref): the buffer [`update_ghosts!`](@ref) writes, not a copy.
`ghost_values` is public, not exported: write `Bramble.ghost_values`.
"""
@inline ghost_values(v::GhostVector) = v.ghosts

"""
    update_ghosts!(v::GhostVector, Cₕ::GhostGridSpace) -> v
    update_ghosts!(v::GhostVector, Cₕ::GhostGridSpace, t::Number[, g]) -> v

Write into the ghost buffer of `v` the ghosts that the relations of `Cₕ` eliminate, from the
values in `v`: each is its row of the closure `(E, d)` of `Cₕ` (see
[`lift`](@ref)), applied to the physical values of `v` and to the ghosts `Cₕ` keeps as
unknowns, which are read from the buffer. These are the ghost values of
`lift(Cₕ, u)` for the degrees of freedom `u` that `v` holds, the conormal corner blocks
included; a point a Dirichlet relation fixes contributes its datum, whatever `v` holds there.
The ghosts of a facet without a relation, and the physical part, are left as they are. On
`gridspace(Ωₕ; ghosts = ...)` itself, with no relations, nothing is written.

The update reads the eliminated ghosts' rows only, stored with the space, so it costs a
sweep over facet-sized data, allocates nothing, and runs in the bands of the mesh's
execution policy, each ghost summed by one band: every CPU policy gives the same values.

With `t`, the relation data are taken at time `t` (data `(x, t) -> …`, see
[`constrained`](@ref)): the ghosts of `lift(Cₕ, u, t)`. On a static space the time is
ignored; on a space whose data depend on time, the form without `t` refuses with an
`ArgumentError`. The data at `t` go into a buffer the space stores, so
`update_ghosts!(v, Cₕ, t)` is not safe to call concurrently on one space; each concurrent
caller passes its own buffer `g = Bramble._affine_buffer(Cₕ)`.

`update_ghosts!` refuses with an `ArgumentError` a `GhostVector` on the mesh of another
space, one whose parts were resized, one whose element type cannot hold the ghost values (a
real `v` on a complex space), and a space on a mesh changed in place since it was built.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)
Cₕ = constrained(gridspace(Ωₕ; ghosts = :boundary),
                 u -> (:xmin => ∂ₙ(u) => (x, t) -> sin(t), :xmax => u + ∂ₙ(u) => 0.0))
v = GhostVector(Cₕ, collect(points(Ωₕ)))
update_ghosts!(v, Cₕ, 0.5)
Bramble.ghost_values(v)         # the ghosts at xmin and xmax at t = 0.5
```
"""
function update_ghosts!(v::GhostVector, Cₕ::GhostGridSpace)
    _is_time_dependent(Cₕ) && _throw_update_needs_time(Cₕ)
    gu = _check_update(v, Cₕ, eltype(Cₕ))
    isempty(gu.slot) && return v
    _run_bands!(execution_policy(Cₕ.padded), _ghost_update_band!, v.ghosts, v.physical,
        gu.slot, gu.colptr, gu.rowval, gu.nz)
    return v
end

function update_ghosts!(v::GhostVector, Cₕ::GhostGridSpace, t::Number,
        g = _stored_buffer(Cₕ.td, t))
    _is_time_dependent(Cₕ) || return update_ghosts!(v, Cₕ)
    gu = _check_update(v, Cₕ, _affine_eltype(Cₕ, t))
    td = Cₕ.td
    length(g) == length(td.g0) || _throw_affine_buffer(td, g)
    pol = execution_policy(Cₕ.padded)
    copyto!(g, td.g0)
    _time_data!(pol, g, td.groups, t)
    isempty(gu.slot) && return v
    _run_bands!(pol, _ghost_update_time_band!, v.ghosts, v.physical, gu.slot, gu.colptr,
        gu.rowval, gu.nz, gu.prow, td.colptr, td.rowval, td.nz, g)
    return v
end

@noinline function update_ghosts!(v, Cₕ, rest...)
    throw(
        ArgumentError(
        "update_ghosts! takes a GhostVector, a space with ghosts and optionally a time, " *
        "update_ghosts!(v, Cₕ) or update_ghosts!(v, Cₕ, t); got $(nameof(typeof(v))), " *
        "$(nameof(typeof(Cₕ))) and $(length(rest)) more argument(s). Build v with " *
        "GhostVector(Cₕ, u).",
    ),
    )
end

# The update map of `Cₕ`, once `v` is checked to live on its mesh with the sizes it had and
# an element type that holds values of type `S`.
@inline function _check_update(v::GhostVector, Cₕ::GhostGridSpace, ::Type{S}) where {S}
    gu = Cₕ.gu
    v.mesh === mesh(Cₕ) || _throw_update_mesh(Cₕ)
    (length(v.physical) == gu.np && length(v.ghosts) == gu.ng) || _throw_update_sizes(v, Cₕ)
    Cₕ.padded.weights.built_version == _mesh_version(mesh(Cₕ)) || _throw_update_stale(Cₕ)
    gu.complete || _throw_update_incomplete(Cₕ)
    T = promote_type(eltype(v), eltype(Cₕ.E), S)
    T === eltype(v) || _throw_update_eltype(v, T)
    return gu
end

@noinline function _throw_update_needs_time(Cₕ::GhostGridSpace)
    throw(
        ArgumentError(
        "update_ghosts!(v, Cₕ) has no time, and the relation data of $(Cₕ) depend on " *
        "time, so its ghosts do too. Give the time, update_ghosts!(v, Cₕ, t).",
    ),
    )
end

@noinline function _throw_update_mesh(Cₕ::GhostGridSpace)
    throw(
        ArgumentError(
        "update_ghosts!(v, Cₕ) takes a GhostVector on the mesh of $(Cₕ), and v lives on " *
        "another mesh. Build it on this space, v = GhostVector(Cₕ, u).",
    ),
    )
end

@noinline function _throw_update_sizes(v::GhostVector, Cₕ::GhostGridSpace)
    throw(
        ArgumentError(
        "update_ghosts!(v, Cₕ) takes $(Cₕ.gu.np) physical values and $(Cₕ.gu.ng) ghosts; " *
        "v holds $(length(v.physical)) and $(length(v.ghosts)), so one of its parts was " *
        "resized. Build it again, v = GhostVector(Cₕ, u).",
    ),
    )
end

@noinline function _throw_update_stale(Cₕ::GhostGridSpace)
    throw(
        ArgumentError(
        "update_ghosts! refuses $(Cₕ): its mesh was changed in place (set_points!, " *
        "change_points!, or iterative_refinement!) after the space was built, so its " *
        "relations hold on a mesh that no longer exists. Build the space again, and " *
        "constrained(gridspace(Ωₕ; ghosts = ...), relations) for a constrained space.",
    ),
    )
end

@noinline function _throw_update_incomplete(Cₕ::GhostGridSpace)
    throw(
        ArgumentError(
        "update_ghosts! refuses $(Cₕ): some degree of freedom of its closure is neither a " *
        "physical node nor a ghost (a closure built by hand), so a GhostVector cannot " *
        "hold it. Build the space with gridspace(Ωₕ; ghosts = ...) and constrained.",
    ),
    )
end

@noinline function _throw_update_eltype(v::GhostVector, T::Type)
    throw(
        ArgumentError(
        "update_ghosts! gives ghost values of type $T, and v holds $(eltype(v)). Build v " *
        "from a vector of element type $T.",
    ),
    )
end

# Eliminated ghosts `r` of band `b` of `nbands`: ghost `slot[r]` is its affine part `nz[r]`
# plus the terms of column `r` (coefficients from `nz[m + 1]`), each a physical value (index
# up to `np`) or a kept ghost of the buffer itself, which no band writes.
@noinline function _ghost_update_band!(gb, x, slot, colptr, rowval, nz, nbands::Int, b::Int)
    T = eltype(gb)
    np, m = length(x), length(slot)
    @inbounds for r in _band_range(1:m, nbands, b)
        s = convert(T, nz[r])
        for k in colptr[r]:(colptr[r + 1] - 1)
            j = rowval[k]
            s = convert(T, s + nz[m + k] * (j <= np ? x[j] : gb[j - np]))
        end
        gb[slot[r]] = s
    end
    return nothing
end

# As `_ghost_update_band!`, with the affine part `d(t)` of each eliminated ghost summed from
# the data `g` at `t`, column `prow[r]` of `Lᵀ` (`lcolptr`, `lrowval`, `lnz`), in place of
# the static one.
@noinline function _ghost_update_time_band!(gb, x, slot, colptr, rowval, nz, prow, lcolptr,
        lrowval, lnz, g, nbands::Int, b::Int)
    T = eltype(gb)
    np, m = length(x), length(slot)
    @inbounds for r in _band_range(1:m, nbands, b)
        s = zero(T)
        i = prow[r]
        for k in lcolptr[i]:(lcolptr[i + 1] - 1)
            s = convert(T, s + lnz[k] * g[lrowval[k]])
        end
        for k in colptr[r]:(colptr[r + 1] - 1)
            j = rowval[k]
            s = convert(T, s + nz[m + k] * (j <= np ? x[j] : gb[j - np]))
        end
        gb[slot[r]] = s
    end
    return nothing
end
