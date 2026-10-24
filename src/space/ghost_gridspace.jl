# ghost_gridspace.jl: `GhostGridSpace`, the grid space over a `GhostMesh`: the physical
# degrees of freedom together with the ghost values beyond the named facets.
#
# The space holds the padded `ScalarGridSpace` (its mesh the `GhostMesh`, its weights
# physical-only) and a closure `(E, d)`: the padded values of a vector `u` of the space's
# degrees of freedom are `E u + d`. `gridspace(Ωₕ; ghosts)` builds the selection closure
# (`E` picks each degree of freedom's padded point, `d = 0`); a constrained space keeps the
# same fields with the relations' `E` and `d`. When some relation data depend on time, `d`
# does too, `d(t) = L g(t)`: the space then also stores `L` and the points where `g` is read
# (`_TimeData`), and `_lift_affine!` evaluates `d(t)` from them; `E` never depends on time.
#
# See also: `GhostMesh`, `ScalarGridSpace`, `SpaceWeights`

# Row replacement (`dirichlet`) on a ghost space, worked out once with the space: whether
# its reduced rows are its physical nodes in mesh order, and the physical nodes of each
# facet (facet `(d, s)` at `2(d - 1) + s`), so that a warm `assemble!` with `dirichlet`, and
# `dirichlet_bc!` and `symmetrize!` on the space, allocate nothing.
#
# A polygon space (polygon_space.jl) refuses `dirichlet`, and stores what the reduction
# reads on it as a trial side (`_ghost_maps`): whether it is no test space, and three masks
# over the padded points: the exterior nodes left without a value (`unclosed`, every one on
# the test space), those the (3.2) extension gives (`extended`), and among these the
# conflict nodes, substituted per reading equation (`conflict`). Every other space has
# empty masks.
struct _RowsDirichlet
    physical::Bool
    facets::Vector{BitVector}
    polygon::Bool
    trial_only::Bool
    unclosed::BitVector
    extended::BitVector
    conflict::BitVector
end

function _RowsDirichlet(physical::Bool, facets::Vector{BitVector})
    return _RowsDirichlet(physical, facets, false, false, BitVector(), BitVector(),
        BitVector())
end

"""
    GhostGridSpace{D, T, S, ET, DT, TD, GU} <: AbstractSpaceType{1}

A scalar grid space with ghost points beyond the named facets of its mesh, built by
[`gridspace`](@ref)`(Ωₕ; ghosts = :boundary)` or `gridspace(Ωₕ; ghosts = (:xmin, ...))`.

Its degrees of freedom are the physical nodes of `Ωₕ`, in the mesh's linear order, then the
ghosts in the order of [`GhostMesh`](@ref). `padded_space(Wₕ)` is the `ScalarGridSpace` on
the padded mesh, the one forms assemble on: its `innerₕ`, `inner₊` and `inner_Γ` weigh the
physical nodes, cells and facets only, and are 0 at every ghost and absent point, so they
equal those of `gridspace(Ωₕ)` on the physical nodes. `closure(Wₕ)` is the pair `(E, d)`
that maps the degrees of freedom `u` to the padded values `E u + d`; on a space whose
relations have data depending on time (see [`constrained`](@ref)), `d` depends on time too.

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
    DT <: AbstractArray{T, 1}, TD, GU} <: AbstractSpaceType{1}
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
    "the map `d(t) = L g(t)` of data depending on time (a `_TimeData`), or `nothing`."
    td::TD
    "the eliminated ghosts' rows of `E` in the order of a `GhostVector` (a `_GhostUpdate`)."
    gu::GU
    "whether the first `npoints` degrees of freedom are the physical nodes' values (`_dofs_are_grid_function`)."
    grid_function::Bool
end

function GhostGridSpace(P::ScalarGridSpace, E::AbstractMatrix, d::AbstractVector,
        td = nothing)
    G = mesh(P)
    Ωₕ = physical_mesh(G)
    rows = _RowsDirichlet(_rows_are_physical(G, E), _facet_masks(Ωₕ, Val(dim(Ωₕ))))
    Et = _closure_transpose(E)
    return GhostGridSpace(P, E, d, Et, rows, td, _ghost_update(backend(P), G, Et, d, td),
        _dofs_are_grid_function(G, Et, d, td))
end

# Whether the first `npoints` degrees of freedom of a closure are the physical nodes' values,
# in the mesh's order, so that a vector of them is a grid function there, with the ghosts
# beyond: the padded row of each physical node `k` is the unit row of degree of freedom `k`
# alone (column `plin[J]` of `Eᵀ` is the single entry `k`, 1: a fixed node has no entry, and
# every node after it is shifted), and the affine part is 0 and constant in time. A
# constrained space with data, or with a node fixed by a Dirichlet relation, is not.
function _dofs_are_grid_function(G::GhostMesh, Et::SparseMatrixCSC, d::AbstractVector,
        td)
    td === nothing && all(iszero, Array(d)) || return false
    plin = LinearIndices(npoints(G, Tuple))
    cp, rv, nzv = SparseArrays.getcolptr(Et), rowvals(Et), nonzeros(Et)
    for (k, J) in enumerate(physical_indices(G))
        i = plin[J]
        (cp[i + 1] - cp[i] == 1 && rv[cp[i]] == k && isone(nzv[cp[i]])) || return false
    end
    return true
end

# What `update_ghosts!` reads: the rows of `E` of the ghosts the relations eliminate,
# re-indexed to a `GhostVector`'s order (physical node `k` at `k`, ghost `g` at `np + g`) and
# stored transposed. Column `r` holds the terms of eliminated ghost `slot[r]`, at padded
# point `prow[r]`.
#
# With `m` eliminated ghosts, `nz` starts with the static affine part `d` of ghost `slot[r]`
# at `r` (0 on a space with data in time, whose `d` is filled after the constructor and
# whose band sums `d(t)` instead); entry `k` of `rowval` has its coefficient at `nz[m + k]`.
# That saves the band pass one array, and `CpuThreaded` pays per argument.
#
# A degree of freedom is the point whose row of `E` is that degree of freedom alone, with
# coefficient 1: a physical node first, then the ghosts in order. Every other ghost is
# eliminated; `E` alone decides. `complete` is `false` when some degree of freedom is no
# such point, or when the affine part can be nonzero at a kept ghost (a closure built by
# hand), and `update_ghosts!` refuses the space.
struct _GhostUpdate{IV <: AbstractArray{Int, 1}, NV <: AbstractArray{<:Any, 1}}
    np::Int
    ng::Int
    complete::Bool
    slot::IV
    prow::IV
    colptr::IV
    rowval::IV
    nz::NV
end

function _ghost_update(be, G::GhostMesh{D}, Et::SparseMatrixCSC, d::AbstractVector,
        td) where {D}
    plin = LinearIndices(npoints(G, Tuple))
    phys = physical_indices(G)
    np, ng = length(phys), nghosts(G)
    cp, rv, nzv = SparseArrays.getcolptr(Et), rowvals(Et), nonzeros(Et)
    dh = td === nothing ? Array(d) : zeros(eltype(d), length(d))
    affine = _affine_support(d, td)
    own = zeros(Int, size(Et, 1))           # the vector index of each degree of freedom
    alone(i) = cp[i + 1] - cp[i] == 1 && isone(nzv[cp[i]]) ? rv[cp[i]] : 0
    for (k, J) in enumerate(phys)
        j = alone(plin[J])
        (j != 0 && own[j] == 0) && (own[j] = k)
    end
    gpts = [plin[_ghost_point(G, g)] for g in 1:ng]
    eliminated = Int[]
    kept_affine = false
    for g in 1:ng
        j = alone(gpts[g])
        if j != 0 && own[j] == 0
            own[j] = np + g
            kept_affine |= affine[gpts[g]]
        else
            push!(eliminated, g)
        end
    end
    complete = !kept_affine && all(!iszero, own)
    prow = gpts[eliminated]
    m = length(eliminated)
    colptr = ones(Int, m + 1)
    rowval = Int[]
    nz = Vector{promote_type(eltype(nzv), eltype(dh))}(dh[prow])
    for (r, g) in enumerate(eliminated)
        for q in cp[gpts[g]]:(cp[gpts[g] + 1] - 1)
            push!(rowval, complete ? own[rv[q]] : 0)
            push!(nz, nzv[q])
        end
        colptr[r + 1] = length(rowval) + 1
    end
    stored(x) = copyto!(_backend_vector(be, eltype(x), length(x)), x)
    return _GhostUpdate(np, ng, complete, stored(eliminated), stored(prow), stored(colptr),
        stored(rowval), stored(nz))
end

# The affine part of a closure whose relation data depend on time: `d(t) = L g(t)`, `g` one
# value per eliminated ghost (the facet-weighted datum at its node, or the weight of a datum
# depending on time) and per point a Dirichlet relation fixes, `L` the closure's block solve
# applied to each. `L` is stored transposed, so column `i` holds the terms of `d[i]`. `g0`
# holds the static values (0 at the sources in time) and is never written; `g` is the
# stored buffer, and a call copies `g0` into its buffer and refills each group's range with
# a datum `f(x, t)` at its points, `f` a type parameter so that the evaluation infers.
struct _TimeGroup{F, PV <: AbstractArray{<:Any, 1}}
    f::F
    points::PV
    first::Int
end

struct _TimeData{G <: Tuple, IV <: AbstractArray{Int, 1}, LV <: AbstractArray{<:Any, 1},
    GV <: AbstractArray{<:Any, 1}}
    groups::G
    colptr::IV
    rowval::IV
    nz::LV
    g0::GV
    g::GV
end

@inline _is_time_dependent(Wₕ::GhostGridSpace) = Wₕ.td !== nothing

# The padded points the affine part can be nonzero at, at any time: those of `d ≠ 0` on a
# static space, the rows of `L` otherwise.
_affine_support(Wₕ::GhostGridSpace) = _affine_support(Wₕ.d, Wₕ.td)
_affine_support(d, ::Nothing) = map(!iszero, Array(d))
_affine_support(_, td::_TimeData) = map(>(0), diff(Array(td.colptr)))

"""
    _lift_affine!(d, Wₕ::GhostGridSpace, t[, g]) -> d

Write the affine part of the closure of `Wₕ` at time `t` into the padded vector `d`: on a
space whose relation data depend on time, the data at `t` at their points and `d = L g(t)`,
in the bands of the mesh's policy, with no block solve; on any other space, its `d`.

The data go into the buffer `g`, by default the one the space stores, which makes the call
allocate nothing but also makes it unsafe to call concurrently on one space. A time whose
type the stored buffer cannot hold (a `ForwardDiff.Dual`) gets a buffer of its own. Pass
`g = _affine_buffer(Wₕ, S)`, of an element type `S` that holds the data at `t`, for a call
that touches no stored state.
"""
function _lift_affine!(d::AbstractVector, Wₕ::GhostGridSpace, t::Number)
    return _lift_affine!(d, Wₕ, t, _stored_buffer(Wₕ.td, t))
end

function _lift_affine!(d::AbstractVector, Wₕ::GhostGridSpace, t::Number, g)
    length(d) == length(Wₕ.d) || _throw_affine_length(Wₕ, d)
    return _lift_affine!(d, Wₕ, Wₕ.td, t, g)
end

@inline function _lift_affine!(d::AbstractVector, Wₕ::GhostGridSpace, ::Nothing, t, g)
    return copyto!(d, Wₕ.d)
end

function _lift_affine!(d::AbstractVector, Wₕ::GhostGridSpace, td::_TimeData, t,
        g::AbstractVector)
    length(g) == length(td.g0) || _throw_affine_buffer(td, g)
    pol = execution_policy(Wₕ.padded)
    copyto!(g, td.g0)
    _time_data!(pol, g, td.groups, t)
    _run_bands!(pol, _csc_columns_band!, d, td.colptr, td.rowval, td.nz, g)
    return d
end

# The element type that holds the affine part at `t`: the space's, for a time of a plain
# number type, and promoted with the time's type otherwise (a `ForwardDiff.Dual`), whose
# data carry it. A static space's `d` does not depend on `t`.
@inline _affine_eltype(Wₕ::GhostGridSpace, t) = _affine_eltype(eltype(Wₕ.d), Wₕ.td, t)
@inline _affine_eltype(::Type{T}, ::Nothing, t) where {T} = T
@inline _affine_eltype(::Type{T}, ::_TimeData,
    ::Union{AbstractFloat, Integer, Rational}) where {T} = T
@inline _affine_eltype(::Type{T}, ::_TimeData, t) where {T} = promote_type(T, typeof(t))

"""
    _affine_buffer(Wₕ::GhostGridSpace, S = eltype(Wₕ)) -> g

A buffer for the data of `Wₕ` at a time, of element type `S`, to pass to `_lift_affine!`
and `lift!` so that they touch no stored state; `nothing` on a static space.
"""
_affine_buffer(Wₕ::GhostGridSpace, ::Type{S} = eltype(Wₕ.d)) where {S} = _affine_buffer(
    Wₕ.td, S)
_affine_buffer(::Nothing, ::Type) = nothing
_affine_buffer(td::_TimeData, ::Type{S}) where {S} = similar(td.g0, S)

# The stored buffer when it holds the data at `t`, else one of their type.
@inline _stored_buffer(::Nothing, t) = nothing
@inline function _stored_buffer(td::_TimeData, t)
    S = _affine_eltype(eltype(td.g), td, t)
    return S === eltype(td.g) ? td.g : _affine_buffer(td, S)
end

@noinline function _throw_affine_buffer(td::_TimeData, g)
    throw(
        DimensionMismatch(
        "the data buffer of this space has $(length(td.g0)) entries; got a vector of " *
        "length $(length(g)). Build it with Bramble._affine_buffer(Wₕ, S).",
    ),
    )
end

# Each group's data at `t`, unrolled over the groups so that each datum's type is known.
@inline _time_data!(pol, g, ::Tuple{}, t) = nothing
@inline function _time_data!(pol, g, groups::Tuple, t)
    grp = first(groups)
    _run_bands!(pol, _TimeDatumBand(grp.f), g, grp.points, grp.first, t)
    return _time_data!(pol, g, Base.tail(groups), t)
end

# The band function of a datum `f`: points `j` of band `b` of `nbands` give
# `g[first + j - 1] = f(points[j], t)`. The datum rides in the band function, not among its
# arguments, where a `CpuPolyester` task would box it.
struct _TimeDatumBand{F}
    f::F
end

@noinline function (k::_TimeDatumBand)(g, pts, first::Int, t, nbands::Int, b::Int)
    @inbounds for j in _band_range(1:length(pts), nbands, b)
        g[first + j - 1] = k.f(pts[j], t)
    end
    return nothing
end

@noinline function _throw_affine_length(Wₕ::GhostGridSpace, d)
    throw(
        DimensionMismatch(
        "the affine part of $(Wₕ) has $(length(Wₕ.d)) padded values; got a vector of " *
        "length $(length(d))",
    ),
    )
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
# is left as built: forms on it are refused by `_check_ghost_host`. The result's type
# follows from `be` and `T` alone, whatever the type of `E` inference sees.
function _closure_matrix(be, ::Type{T}, E::SparseMatrixCSC) where {T}
    MT = matrix_type(be)
    Ec = SparseMatrixCSC{T, Int}(E)::SparseMatrixCSC{T, Int}
    return MT <: SparseMatrixCSC ? convert(MT, Ec)::MT : Ec
end

# The affine part of a closure, in the backend's vector type, which `be` and `T` decide.
@inline function _closure_data(be, ::Type{T}, d::AbstractVector) where {T}
    v = _backend_vector(be, T, length(d))
    copyto!(v, d)
    return v
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
of size `npoints(mesh(Wₕ)) × ndofs(Wₕ)` with rows in the padded linear order. On a space
whose relation data depend on time, `d` is its value at `t = 0`, and
[`lift`](@ref)`(Wₕ, u, t)` gives `E u + d(t)`.
"""
@inline closure(Wₕ::GhostGridSpace) = (Wₕ.E, Wₕ.d)

@inline mesh(Wₕ::GhostGridSpace) = mesh(Wₕ.padded)
@inline mesh_type(::Type{<:GhostGridSpace{D, T, S}}) where {D, T, S} = mesh_type(S)
@inline mesh_type(Wₕ::GhostGridSpace) = mesh_type(typeof(Wₕ))
@inline backend(Wₕ::GhostGridSpace) = backend(Wₕ.padded)
# The policy element broadcasts run under, as on a plain space: `uₕ - vₕ` on one space.
@inline execution_policy(Wₕ::GhostGridSpace) = execution_policy(backend(Wₕ))
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
    lift(Wₕ::GhostGridSpace, u::AbstractVector, t::Number) -> VectorElement

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

`lift(Wₕ, u, t)` is `E u + d(t)`, for a space whose relation data depend on time,
`(x, t) -> …` (see [`constrained`](@ref)): `d(t)` comes from the data at `t`, and `E` is the
same at every `t`. On a static space it is `lift(Wₕ, u)`. It evaluates the data into a
buffer of its own, so it may be called concurrently on one space, and a time of a
non-float type, such as a `ForwardDiff.Dual`, gives values of that type. On a space with
time-dependent data, `lift(Wₕ)` and `lift(Wₕ, u)` refuse with an `ArgumentError`, as there
is no single `d`: give the time, `lift(Wₕ, u, t)`, or pose the problem in time through
`semidiscretize`.

`lift(Wₕ, u)` runs in the bands of the mesh's execution policy, each padded point summed by
one band, so it gives the same values on every CPU policy. `Bramble.lift!(uₕ, Wₕ, u)` and
`Bramble.lift!(uₕ, Wₕ, u, t)` refill a grid function `lift` returned, allocating nothing.

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

Dₜ = constrained(gridspace(Ωₕ; ghosts = :boundary),
                 u -> (:xmin => ∂ₙ(u) => (x, t) -> sin(t), :xmax => ∂ₙ(u) => 0.0))
vₕ = Bramble.lift(Dₜ, zeros(Bramble.ndofs(Dₜ)), 0.5)   # the ghosts of the data at t = 0.5
```
"""
function lift(Wₕ::GhostGridSpace)
    _check_lift_fresh(Wₕ)
    _check_lift_static(Wₕ, "lift(Wₕ)")
    return VectorElement(copy(Wₕ.d), Wₕ.padded)
end

function lift(Wₕ::GhostGridSpace, u::AbstractVector)
    _check_lift_fresh(Wₕ)
    _check_lift_static(Wₕ, "lift(Wₕ, u)")
    length(u) == ndofs(Wₕ) || _throw_lift_length(:lift, Wₕ, u)
    U = similar(Wₕ.d, promote_type(eltype(u), eltype(Wₕ.d)))
    return VectorElement(_lift_values!(U, Wₕ, u), Wₕ.padded)
end

function lift(Wₕ::GhostGridSpace, u::AbstractVector, t::Number)
    _check_lift_fresh(Wₕ)
    length(u) == ndofs(Wₕ) || _throw_lift_length(:lift, Wₕ, u)
    S = _affine_eltype(Wₕ, t)
    U = similar(Wₕ.d, promote_type(eltype(u), S))
    return VectorElement(_lift_values!(U, Wₕ, u, t, _affine_buffer(Wₕ, S)), Wₕ.padded)
end

"""
    lift!(uₕ::VectorElement, Wₕ::GhostGridSpace, u::AbstractVector) -> uₕ
    lift!(uₕ::VectorElement, Wₕ::GhostGridSpace, u::AbstractVector, t::Number[, g]) -> uₕ

Write `E u + d` into the grid function `uₕ`, the in-place form of
[`lift`](@ref)`(Wₕ, u)`: `uₕ` lives on the padded space of `Wₕ`, as the result of `lift`
does, so a solve loop lifts each new solution without allocating. The values are those of
`lift(Wₕ, u)`, bit for bit, on every CPU policy. With `t`, it writes `E u + d(t)`, the
values of `lift(Wₕ, u, t)`; on a space whose relation data depend on time, the form
without `t` refuses, as `lift(Wₕ, u)` does. The data at `t` go into a buffer the space
stores, so `lift!(uₕ, Wₕ, u, t)` is not safe to call concurrently on one space; each
concurrent caller passes its own buffer `g = Bramble._affine_buffer(Wₕ)` instead.

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
    _check_lift_static(Wₕ, "lift!(uₕ, Wₕ, u)")
    _check_lift_into(uₕ, Wₕ, u)
    _lift_values!(uₕ.data, Wₕ, u)
    return uₕ
end

function lift!(uₕ::VectorElement, Wₕ::GhostGridSpace, u::AbstractVector, t::Number,
        g = _stored_buffer(Wₕ.td, t))
    _check_lift_into(uₕ, Wₕ, u, _affine_eltype(Wₕ, t))
    _lift_values!(uₕ.data, Wₕ, u, t, g)
    return uₕ
end

@inline function _check_lift_into(uₕ::VectorElement, Wₕ::GhostGridSpace, u::AbstractVector,
        ::Type{S} = eltype(Wₕ.d)) where {S}
    _check_lift_fresh(Wₕ)
    length(u) == ndofs(Wₕ) || _throw_lift_length(:lift!, Wₕ, u)
    uₕ.space === Wₕ.padded || _throw_lift_target(uₕ, Wₕ)
    T = promote_type(eltype(u), S)
    promote_type(eltype(uₕ), T) === eltype(uₕ) || _throw_lift_eltype(uₕ, T)
    return nothing
end

# `U = E u + d` in the bands of the mesh's policy; `d` may be `U` itself, as each padded
# point reads its own `d[i]` before it writes `U[i]`.
function _lift_values!(U::AbstractVector, Wₕ::GhostGridSpace, u::AbstractVector,
        d::AbstractVector = Wₕ.d)
    Et = Wₕ.Et
    _run_bands!(execution_policy(Wₕ.padded), _lift_band!, U, d, SparseArrays.getcolptr(Et),
        rowvals(Et), nonzeros(Et), u)
    return U
end

# `U = E u + d(t)`: `d(t)` written into `U` first, through the data buffer `g`, then read
# back in place.
function _lift_values!(U::AbstractVector, Wₕ::GhostGridSpace, u::AbstractVector, t::Number,
        g)
    _lift_affine!(U, Wₕ, t, g)
    return _lift_values!(U, Wₕ, u, U)
end

# Without a time, a space whose data depend on time has no affine part to lift with.
@inline function _check_lift_static(Wₕ::GhostGridSpace, call::String)
    _is_time_dependent(Wₕ) && _throw_time_dependent(Wₕ, call)
    return nothing
end

@noinline function _throw_time_dependent(Wₕ::GhostGridSpace, call::AbstractString)
    throw(
        ArgumentError(
        "$call has no time, and the relation data of $(Wₕ) depend on time, so its " *
        "closure's affine part d(t) does too. Give the time, lift(Cₕ, u, t) or " *
        "lift!(uₕ, Cₕ, u, t), or pose the problem in time with semidiscretize.",
    ),
    )
end

@noinline function _throw_lift_target(uₕ, Wₕ::GhostGridSpace)
    throw(
        ArgumentError(
        "lift!(uₕ, Wₕ, u) writes into a grid function on the padded space of $(Wₕ), and " *
        "uₕ lives on $(summary(uₕ.space)). Build it once with uₕ = Bramble.lift(Wₕ, u).",
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

# The closure holds on the mesh version the weights record: after `change_points!` its
# ghost values solve the relations on a mesh that no longer exists. The padded space's own
# stale message would send the user to `padded_space`, which `lift` refuses.
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
        "lift takes the space, at most one vector and a time, lift(Wₕ), lift(Wₕ, u) or " *
        "lift(Wₕ, u, t); got $(1 + length(rest)) arguments after $(Wₕ), of types " *
        "$(join(map(x -> nameof(typeof(x)), (u, rest...)), ", ")). Pass a vector of " *
        "degrees of freedom and a number for the time, and drop any other argument.",
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
# with no method for the space. A polygon trial space is the exception for `Rₕ`: its
# degrees of freedom are the interior nodes, so `Rₕ(U, f)` holds `f` there
# (`_polygon_restrict!`, polygon_space.jl); its test space keeps the refusal.
function Rₕ(Wₕ::GhostGridSpace, f; kwargs...)
    _is_polygon_trial(Wₕ) || _throw_projection_on_ghost_space(:Rₕ, Wₕ)
    isempty(kwargs) || _throw_polygon_restriction_markers(:Rₕ)
    uₕ = element(Wₕ, _polygon_restriction_eltype(Wₕ, f))
    return _polygon_restrict!(uₕ, f)
end
function Rₕ!(uₕ::VectorElement{<:GhostGridSpace}, f; kwargs...)
    _is_polygon_trial(uₕ.space) || _throw_projection_on_ghost_space(:Rₕ!, uₕ.space)
    isempty(kwargs) || _throw_polygon_restriction_markers(:Rₕ!)
    return _polygon_restrict!(uₕ, f)
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

# The first `npoints` degrees of freedom of a ghost element, the values at the physical nodes
# in the mesh's order, for `normₕ` and `interpolate_at`. A space whose degrees of freedom
# omit fixed nodes or carry data has no such vector: refused, as a vector of its degrees of
# freedom is not a grid function there.
@inline function _physical_values(op::Symbol, uₕ::VectorElement{<:GhostGridSpace})
    Wₕ = uₕ.space
    Wₕ.grid_function || _throw_not_grid_function(op, Wₕ)
    return view(parent(uₕ), 1:length(physical_indices(mesh(Wₕ))))
end

@noinline function _throw_not_grid_function(op::Symbol, Wₕ::GhostGridSpace)
    throw(
        ArgumentError(
        "$op reads the physical values of a ghost space's grid function, and a vector of " *
        "the degrees of freedom of $(Wₕ) is not one: its relations fix nodes or carry data, " *
        "so its degrees of freedom are not the values at the physical nodes. Build the " *
        "grid function with `uₕ = Bramble.lift(Cₕ, u)` and call $op on that.",
    ),
    )
end
