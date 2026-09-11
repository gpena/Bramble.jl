"""
# interface.jl

Abstract supertype, interface contracts, and shared fallbacks for every mesh type in
Bramble: `AbstractMeshType{D}`, field getters, bounds checking, refinement
(`iterative_refinement!`, `change_points!`), and interface stubs (`eltype`, `dim`,
`topo_dim`, `points`, `point`, `half_points`, `half_point`, `spacing`, `forward_spacing`,
`half_spacings`, `half_spacing`, `npoints`, `hₘₐₓ`, `hₘᵢₙ`, `cell_measure`).

Cartesian index generation and boundary/interior predicates live in `mesh/indices.jl`; the
public `mesh(...)` constructor dispatch lives in `mesh/constructors.jl`; `is_uniform`,
`stepsize`, `locate_cell`, `normal_vector`, and the `Base` collection interface live in
`mesh/queries.jl`.

See also: [`Mesh1D`](@ref), [`MeshnD`](@ref), [`Domain`](@ref)
"""

#------------------------------------------------------------------------------------------#
# Abstract Supertype
#------------------------------------------------------------------------------------------#

"""
    AbstractMeshType{D}

Abstract supertype for all mesh types in Bramble. The type parameter `D` represents
the spatial dimension of the mesh (1, 2, or 3).

All concrete mesh types must implement the AbstractMeshType interface, including:

  - `eltype`, `dim`, `topo_dim`, `indices`, `backend`, `markers`
  - `points`, `point`, `half_points`, `half_point`
  - `spacing`, `half_spacing`, `forward_spacing`

# Type parameters

  - `D`: Spatial dimension (1, 2, or 3)

# Related types

  - Meshes are created from a [`Domain`](@ref) using the [`mesh`](@ref) function.
  - See [`MeshMarkers`](@ref) for marker management on meshes.

See also: [`Mesh1D`](@ref), [`MeshnD`](@ref), [`Domain`](@ref)
"""
abstract type AbstractMeshType{D} end

#------------------------------------------------------------------------------------------#
# Field Getters & Index Delegation
#------------------------------------------------------------------------------------------#

"""
    set(Ωₕ::AbstractMeshType) -> AbstractSetType

Return the underlying geometric set of the domain over which mesh `Ωₕ` is defined.
"""
@inline set(Ωₕ::AbstractMeshType) = Ωₕ.set

"""
    indices(Ωₕ::AbstractMeshType) -> CartesianIndices

Return the `CartesianIndices` associated with the points of mesh `Ωₕ`.
"""
@inline indices(Ωₕ::AbstractMeshType) = Ωₕ.indices

"""
    backend(Ωₕ::AbstractMeshType) -> Backend

Return the linear algebra [`Backend`](@ref) associated with mesh `Ωₕ`.
"""
@inline backend(Ωₕ::AbstractMeshType) = Ωₕ.backend

"""
    execution_policy(Ωₕ::AbstractMeshType) -> ExecutionPolicy

Return the [`ExecutionPolicy`](@ref) ([`Serial`](@ref) or [`Parallel`](@ref)) of the
[`Backend`](@ref) associated with mesh `Ωₕ`.
"""
@inline execution_policy(Ωₕ::AbstractMeshType) = execution_policy(backend(Ωₕ))

"""
    markers(Ωₕ::AbstractMeshType) -> MeshMarkers

Return the [`MeshMarkers`](@ref) dictionary associated with mesh `Ωₕ`.
"""
@inline markers(Ωₕ::AbstractMeshType) = Ωₕ.markers

"""
    index_in_marker(Ωₕ::AbstractMeshType, label::Symbol) -> BitVector

Return the `BitVector` indicator associated with marker `label` in mesh `Ωₕ`.
"""
@inline index_in_marker(Ωₕ::AbstractMeshType, label::Symbol) = markers(Ωₕ)[label]

"""
    set_indices!(Ωₕ::AbstractMeshType, indices::CartesianIndices) -> Nothing

Override the grid indices in `Ωₕ`. Used internally during mesh refinement.
"""
@inline set_indices!(Ωₕ::AbstractMeshType, indices) = (Ωₕ.indices=indices; return nothing)

"""
    markers!(Ωₕ::AbstractMeshType, mesh_markers::MeshMarkers) -> Nothing

Override the markers dictionary in `Ωₕ`. Used internally during mesh refinement, where
the old dictionary is sized for the old grid and is replaced outright rather than merged
into.
"""
@inline markers!(Ωₕ::AbstractMeshType, mesh_markers) =
    (Ωₕ.markers=mesh_markers; return nothing)

"""
    is_collapsed(Ωₕ::AbstractMeshType) -> Bool

Whether `Ωₕ` has no interval to refine or measure a spacing over — a [`Mesh1D`](@ref)
built over a single point. A [`MeshnD`](@ref) is never collapsed as a whole: each axis is
its own `Mesh1D` and may be collapsed individually, which is handled per axis rather than
at this level, so the default here is `false`.
"""
@inline is_collapsed(::AbstractMeshType) = false

# The only real difference between `Mesh1D`'s and `MeshnD`'s refinement: a `MeshnD` always
# has something to refine (each axis handles its own collapse independently, inside
# `_refine_indices!`), while a `Mesh1D` with fewer than two points — collapsed, or a
# genuine single-point mesh over a non-degenerate domain — has no interval at all.
@inline _nothing_to_refine(::AbstractMeshType) = false

#===========================================================================#
# Refinement and point replacement
#
# The geometric part (`_refine_indices!`, `change_points!(Ωₕ, pts)`) is type-specific and
# defined alongside each mesh type. Everything downstream of it — deciding whether there
# is anything to do, and rebuilding markers afterward — reads only the fields this file
# already assumes exist, so it is written once here rather than once per mesh type
# (gpena/Bramble.jl#68).
#===========================================================================#

# The one-argument form: no domain to re-derive custom markers from, so a mesh carrying
# any is refused outright rather than silently losing them. The public contract for both
# this and the two-argument form below is documented once, on the
# `function iterative_refinement! end` stub further down this file, rather than repeated
# on each method.
function iterative_refinement!(Ωₕ::AbstractMeshType)
    # Mirrored from `_refine_indices!`'s own no-op cases (a collapsed or single-point
    # `Mesh1D`), rather than just inherited from calling it: `_refine_indices!` returning
    # "did nothing" is not visible to its caller, and rebuilding markers anyway would drop
    # a mesh's custom labels for no reason at all.
    _nothing_to_refine(Ωₕ) && return nothing

    # The old markers dict is sized for the old grid and would otherwise be left silently
    # wrong rather than merely absent: `haskey(markers(Ωₕ), :boundary)` still answers
    # `true`, its `BitVector` still indexes without erroring (it is shorter than the new
    # point count, not longer), and every point beyond its old length reads as "not
    # boundary" (found by refining a mesh and reassembling a Poisson problem on it, where
    # the boundary rows past the old length never got constrained and the system went
    # singular with no error naming why). `:boundary`/`:interior` need no domain and are
    # rebuilt unconditionally; anything else has no domain to re-derive it from, so this
    # refuses rather than drops it with a warning the caller could miss
    # (gpena/Bramble.jl#19). The two-argument form below does not route through this
    # method at all, precisely so it never triggers this check on its own account.
    old_markers = markers(Ωₕ)
    extra_labels = setdiff(keys(old_markers), (:boundary, :interior))
    isempty(extra_labels) || _throw_refinement_drops_markers(extra_labels)

    _refine_indices!(Ωₕ)
    fresh_markers = MeshMarkers()
    _ensure_geometric_markers!(fresh_markers, Ωₕ)
    markers!(Ωₕ, fresh_markers)
    return nothing
end

@noinline function _throw_refinement_drops_markers(extra_labels)
    throw(
        ArgumentError(
            "iterative_refinement!(Ωₕ) was asked to refine a mesh carrying custom markers " *
            "$(Tuple(extra_labels)), and there is no domain here to re-evaluate them onto " *
            "the refined points. Call iterative_refinement!(Ωₕ, domain_markers) instead to " *
            "keep them.",
        ),
    )
end

# The two-argument form: a real domain to re-derive markers from, so refinement always
# proceeds except when there is no interval at all (`is_collapsed`) — unlike the
# one-argument form above, a single-point, non-collapsed mesh still has its (unchanged)
# point's markers correctly re-evaluated, since `set_markers!` needs no interval to do that.
function iterative_refinement!(
    Ωₕ::AbstractMeshType, domain_markers::DomainMarkers; warn_marker_mismatch::Bool=true
)
    is_collapsed(Ωₕ) && return nothing

    _refine_indices!(Ωₕ)
    set_markers!(Ωₕ, domain_markers; warn_marker_mismatch)
    return nothing
end

function change_points!(
    Ωₕ::AbstractMeshType,
    domain_markers::DomainMarkers,
    pts;
    warn_marker_mismatch::Bool=true,
)
    change_points!(Ωₕ, pts)
    set_markers!(Ωₕ, domain_markers; warn_marker_mismatch)
    return nothing
end

#------------------------------------------------------------------------------------------#
# Bounds Checking & Internal Helpers
#------------------------------------------------------------------------------------------#

@noinline _throw_mesh_bounds_error(Ωₕ, idx) = throw(BoundsError(Ωₕ, idx))

@noinline _throw_not_uniform() = throw(
    ArgumentError(
        "stepsize is only defined for a uniform mesh; use spacing(Ωₕ, idx) on a non-uniform one",
    ),
)

@inline function _check_point_bounds(
    Ωₕ::AbstractMeshType, idx::Int, location::String="point"
)
    @boundscheck 1 <= idx <= npoints(Ωₕ) || _throw_mesh_bounds_error(Ωₕ, idx)
    return nothing
end

@inline function _check_half_point_bounds(Ωₕ::AbstractMeshType, idx::Int)
    @boundscheck 1 <= idx <= npoints(Ωₕ) + 1 || _throw_mesh_bounds_error(Ωₕ, idx)
    return nothing
end

@inline _extract_linear_index(idx::Int) = idx
@inline _extract_linear_index(idx::CartesianIndex{1}) = idx[1]
@inline _spacing_generator(Ωₕ::AbstractMeshType, spacing_func) =
    (spacing_func(Ωₕ, i) for i in 1:npoints(Ωₕ))
@inline _apply_hs_logic(value::T) where {T} = ifelse(iszero(value), one(T), value)

#------------------------------------------------------------------------------------------#
# Required Interface Methods
#------------------------------------------------------------------------------------------#

"""
    dim(Ωₕ::AbstractMeshType{D}) -> Int
    dim(::Type{<:AbstractMeshType{D}}) -> Int

Return the spatial dimension ``D`` of the domain where `Ωₕ` is embedded.
"""
@inline dim(::AbstractMeshType{D}) where {D} = D
@inline dim(::Type{<:AbstractMeshType{D}}) where {D} = D

"""
    topo_dim(Ωₕ::AbstractMeshType{D}) -> Int

Return the topological dimension of `Ωₕ`.

The topological dimension counts the number of coordinate axes with more than one point,
identifying degenerate or collapsed dimensions (such as manifolds or boundaries embedded
in higher-dimensional ambient space).
"""
@inline function topo_dim(Ωₕ::AbstractMeshType{D}) where {D}
    count = 0
    @inbounds for i in 1:D
        npoints(Ωₕ(i)) > 1 && (count += 1)
    end
    return count
end

"""
    eltype(Ωₕ::AbstractMeshType) -> Type
    eltype(::Type{<:AbstractMeshType}) -> Type

Return the floating-point coordinate element type of the points in `Ωₕ`.
"""
function eltype(Ωₕ::AbstractMeshType)
    return error(
        "Interface function 'eltype' not implemented for mesh of type $(typeof(Ωₕ))."
    )
end

function eltype(::Type{<:AbstractMeshType})
    return error("Interface function 'eltype(::Type{...})' not implemented for mesh type.")
end

"""
    points(Ωₕ::AbstractMeshType) -> Union{Vector, NTuple}

Return the coordinates of the mesh points:
  - For 1D meshes ([`Mesh1D`](@ref)): returns a coordinate vector `Vector{T}` of length ``N_x``.
  - For nD meshes ([`MeshnD`](@ref)): returns an `NTuple{D, Vector{T}}` containing the 1D coordinate vectors along each axis.

See also: [`point`](@ref).
"""
function points end

"""
    point(Ωₕ::AbstractMeshType, idx)

Return the coordinate point at index `idx` (linear integer, tuple `(i, j)`, or `CartesianIndex`):
  - For 1D meshes: scalar coordinate ``x_i``.
  - For nD meshes: coordinate tuple ``(x_{i_1}, \\dots, x_{i_D})``.

Direct indexing `Ωₕ[idx]` delegates to `point(Ωₕ, idx)`.
"""
function point end

"""
    half_points(Ωₕ::AbstractMeshType)

Return the precomputed cell centers (half-points) for each coordinate axis:
```math
x_{i+1/2} = \\frac{x_i + x_{i+1}}{2}, \\quad i = 1, \\dots, N-1.
```
"""
function half_points end

"""
    half_point(Ωₕ::AbstractMeshType, idx)

Return the cell center (half-point) coordinate corresponding to index `idx`.
"""
function half_point end

"""
    spacing(Ωₕ::AbstractMeshType, idx)

Return the backward spacing ``h_i = x_i - x_{i-1}`` at index `idx` (for ``i=1``, returns ``x_2 - x_1``).
For nD meshes, returns a tuple of backward spacings along each axis.
"""
function spacing end

"""
    forward_spacing(Ωₕ::AbstractMeshType, idx)

Return the forward spacing ``h_{i+1} = x_{i+1} - x_i`` at index `idx` (for ``i=N``, returns ``x_N - x_{N-1}``).
For nD meshes, returns a tuple of forward spacings along each axis.
"""
function forward_spacing end

"""
    half_spacings(Ωₕ::AbstractMeshType)

Return the cell widths (half-spacings) along each axis:
```math
h_{i+1/2} = \\frac{h_i + h_{i+1}}{2}.
```
"""
function half_spacings end

"""
    half_spacing(Ωₕ::AbstractMeshType, idx)

Return the cell width (half-spacing) at index `idx`.
"""
function half_spacing end

"""
    npoints(Ωₕ::AbstractMeshType) -> Int
    npoints(Ωₕ::AbstractMeshType, ::Type{Tuple}) -> NTuple{D, Int}

Return the total number of points in `Ωₕ`.
When passing `Tuple` as the second argument, returns a tuple with the number of points along each dimension.
"""
function npoints end

"""
    hₘₐₓ(Ωₕ::AbstractMeshType) -> Real

Return the maximum diagonal stepsize across all cells in the mesh:
```math
h_{\\max} = \\max_{\\mathbf{i}} \\| (h_{1, i_1}, \\dots, h_{D, i_D}) \\|_2.
```
"""
function hₘₐₓ end

"""
    hₘᵢₙ(Ωₕ::AbstractMeshType) -> Real

Return the diagonal of the smallest cell in the mesh, the counterpart of [`hₘₐₓ`](@ref):
  - In 1D: ``\\min_i (x_i - x_{i-1})``.
  - In nD:

```math
h_{\\min} = \\min_{\\mathbf{i}} \\| (h_{1, i_1}, \\dots, h_{D, i_D}) \\|_2.
```

This is a diagonal rather than an edge length, so that `hₘₐₓ` and `hₘᵢₙ` measure the same
kind of quantity. For the smallest extent along one coordinate, query that submesh directly:
`hₘᵢₙ(Ωₕ(i))`.
"""
function hₘᵢₙ end

"""
    cell_measure(Ωₕ::AbstractMeshType, idx) -> Real

Return the control volume (length, area, or volume) of the cell centered at index `idx`:
```math
\\operatorname{meas}(\\square_{\\mathbf{i}}) = \\prod_{d=1}^D h_{d, i_d+1/2}.
```
"""
function cell_measure end

"""
    iterative_refinement!(Ωₕ::AbstractMeshType, [domain_markers::DomainMarkers]) -> AbstractMeshType

Refine the mesh `Ωₕ` in-place by halving each existing cell (inserting new points at midpoints).
If domain markers are supplied, they are re-evaluated onto the refined grid points.

Without `domain_markers`, any custom marker `Ωₕ` carries beyond `:boundary`/`:interior` has
no domain here to re-derive it from, so this throws an `ArgumentError` rather than silently
dropping it. Pass `domain_markers` (the same ones the mesh was built with, or equivalent) to
keep them.
"""
function iterative_refinement! end

"""
    change_points!(Ωₕ::AbstractMeshType, [domain_markers::DomainMarkers], pts) -> AbstractMeshType

Update the coordinates of mesh `Ωₕ` in-place using new point coordinates in `pts`,
recalculating all cached half-points and cell spacings.
"""
function change_points! end
