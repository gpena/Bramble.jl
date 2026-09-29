# gmg_hierarchy.jl: the nested meshes of geometric multigrid (gpena/Bramble.jl#329).
#
# Each coarser level keeps every other point of the finer one along every axis, so the levels
# nest exactly on a non-uniform mesh, which is what makes linear interpolation between them a
# prolongation. A level is built as a mesh over the same set with the halved point count and
# then moved onto those points by `change_points!`; rebuilding each level from the domain
# instead would place a non-uniform level's points afresh, and the levels would not nest.
#
# Markers are carried over by taking every other entry of the finer level's `BitVector`s,
# along the same axes as the points. A marker is a pointwise property of the coordinates
# (a boundary face or a predicate on `x`), and the coarse points are a subset of the fine
# ones, so this is exactly what re-evaluating the domain's markers would give. A mesh does
# not keep its `DomainMarkers`, so re-evaluation could not recover a custom label.

"""
    GeometricMeshHierarchy{D, M <: AbstractMeshType{D}}

The nested meshes of geometric multigrid, ordered coarse to fine: `H[1]` is the coarsest
level and `H[end]` the mesh the hierarchy was built from. Each level holds every other point
of the next finer one along every axis, and its markers are those of the finer level at the
points it keeps. Every level has the type, and so the backend and element type, of the
finest. Build one with `GeometricMeshHierarchy(Ωₕ, levels)`.

A hierarchy indexes and iterates like a vector of meshes: `length`, `getindex`, `keys`,
`firstindex`, `lastindex` and iteration are defined. It holds the finest mesh itself, not a
copy, so it must be rebuilt after that mesh's points change (`change_points!`,
`iterative_refinement!`): the coarser levels would no longer nest in it.

# Type parameters
- `D`: The dimension of the meshes.
- `M`: The mesh type of every level.

See also: [`mesh`](@ref), [`change_points!`](@ref).
"""
struct GeometricMeshHierarchy{D, M <: AbstractMeshType{D}}
    meshes::Vector{M}
end

"""
    GeometricMeshHierarchy(Ωₕ::AbstractMeshType, levels::Integer) -> GeometricMeshHierarchy

The hierarchy of `levels` meshes coarsened by 2 from `Ωₕ`: level `l` has
``(n_d - 1) / 2^{L-l} + 1`` points along axis `d`, with ``n_d`` the points of `Ωₕ` along it
and ``L`` the number of levels, and its coordinates are those of `Ωₕ` at the indices
``1, 1 + 2^{L-l}, 1 + 2 \\cdot 2^{L-l}, \\dots``. Levels nest exactly on non-uniform meshes,
and a marker of `Ωₕ`, custom ones included, marks the same points on every level it reaches.

`Ωₕ` itself is the finest level, not a copy: `H[end] === Ωₕ`. Coarsening needs
``(n_d - 1)`` divisible by ``2^{L-1}`` on every axis; a collapsed axis (one point) stays one
point. Construction allocates one mesh per coarser level.

# Arguments
- `Ωₕ`: The finest mesh, a [`Mesh1D`](@ref) or [`MeshnD`](@ref).
- `levels`: The number of levels, `Ωₕ` included.

# Returns
- [`GeometricMeshHierarchy`](@ref): The levels, coarsest first.

# Throws
- `ArgumentError`: `levels < 1`.
- `ArgumentError`: ``n_d - 1`` is not divisible by ``2^{L-1}`` on some axis `d`.

# Examples
Every other point of a non-uniform mesh.
```jldoctest
using Bramble
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (9, 17), false)
H = GeometricMeshHierarchy(Ωₕ, 3)
(length(H), npoints(H[1], Tuple), points(H[2])[1] == points(Ωₕ)[1][1:2:end], H[end] === Ωₕ)

# output
(3, (3, 5), true, true)
```

See also: [`mesh`](@ref), [`change_points!`](@ref), [`markers`](@ref).
"""
function GeometricMeshHierarchy(Ωₕ::AbstractMeshType{D}, levels::Integer) where {D}
    levels >= 1 ||
        throw(ArgumentError("GeometricMeshHierarchy needs levels >= 1, got $levels"))
    n = npoints(Ωₕ, Tuple)
    for d in 1:D
        n[d] == 1 || trailing_zeros(n[d] - 1) >= levels - 1 ||
            _throw_hierarchy_not_divisible(n, d, levels)
    end
    meshes = Vector{typeof(Ωₕ)}(undef, levels)
    meshes[levels] = Ωₕ
    for l in (levels - 1):-1:1
        meshes[l] = _coarsen_mesh(meshes[l + 1])
    end
    return GeometricMeshHierarchy{D, typeof(Ωₕ)}(meshes)
end

@noinline function _throw_hierarchy_not_divisible(n, d, levels)
    throw(
        ArgumentError(
        "GeometricMeshHierarchy with $levels levels needs (n - 1) divisible by 2^$(levels - 1) on " *
        "every axis, but axis $d has n = $(n[d]) points (mesh points $n)",
    ),
    )
end

# The next coarser level of `Ωf`: a mesh over the same set and backend with every other point
# of `Ωf`, then `Ωf`'s markers at those points. The mesh is built uniform, since its points are
# replaced at once.
function _coarsen_mesh(Ωf::AbstractMeshType{D}) where {D}
    nf = npoints(Ωf, Tuple)
    nc = map(k -> (k - 1) ÷ 2 + 1, nf)
    Ωc = mesh(set(Ωf), D == 1 ? nc[1] : nc, true; backend = backend(Ωf))
    change_points!(Ωc, _every_other_point(Ωf))
    coarse_markers = MeshMarkers()
    for (label, bv) in markers(Ωf)
        coarse_markers[label] = _every_other_entry(bv, nf)
    end
    markers!(Ωc, coarse_markers)
    return Ωc
end

@inline _every_other_point(Ωf::Mesh1D) = points(Ωf)[1:2:end]
@inline _every_other_point(Ωf::MeshnD) = map(p -> p[1:2:end], points(Ωf))

# A marker is a `BitVector` over the points in linear (column-major) order.
@inline _every_other_entry(bv::BitVector, ::NTuple{1, Int}) = bv[1:2:end]
@inline function _every_other_entry(bv::BitVector, nf::NTuple{D, Int}) where {D}
    return BitVector(vec(reshape(bv, nf)[ntuple(d -> 1:2:nf[d], Val(D))...]))
end

Base.length(H::GeometricMeshHierarchy) = length(H.meshes)
Base.@propagate_inbounds Base.getindex(H::GeometricMeshHierarchy, l::Integer) = H.meshes[l]
Base.firstindex(::GeometricMeshHierarchy) = 1
Base.keys(H::GeometricMeshHierarchy) = Base.OneTo(length(H))
Base.lastindex(H::GeometricMeshHierarchy) = length(H.meshes)
Base.iterate(H::GeometricMeshHierarchy, state...) = iterate(H.meshes, state...)
Base.eltype(::Type{<:GeometricMeshHierarchy{D, M}}) where {D, M} = M

function Base.show(io::IO, H::GeometricMeshHierarchy{D}) where {D}
    L = length(H)
    print(
        io,
        "GeometricMeshHierarchy{",
        D,
        "D, ",
        L,
        " level",
        L == 1 ? "" : "s",
        ", ",
        npoints(H[1], Tuple),
        " to ",
        npoints(H[end], Tuple),
        " pts}"
    )
    return nothing
end
