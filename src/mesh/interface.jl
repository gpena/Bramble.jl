"""
# interface.jl

This file defines the abstract interface and shared foundational methods for all mesh types in Bramble.

## Key Components

- `AbstractMeshType{D}`: Abstract supertype parameterized by spatial dimension ``D``.
- **Cartesian Index Operations**: `generate_indices`, `boundary_indices`, `interior_indices`, `is_boundary_index`.
- **Interface Declarations & Fallbacks**: Field getters, bounds checking, iterators, and spacing calculations.
- **Mesh Construction**: Top-level `mesh(Ω, npts, ...)` dispatch.

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

# Type Parameter

  - `D`: Spatial dimension (1, 2, or 3)

# Related Types

  - Meshes are created from a [Domain](@ref) using the [`mesh`](@ref) function.
  - See [MeshMarkers](@ref) for marker management on meshes.

See also: [`Mesh1D`](@ref), [`MeshnD`](@ref), [`Domain`](@ref)
"""
abstract type AbstractMeshType{D} end

#------------------------------------------------------------------------------------------#
# Cartesian Index Generation & Boundary Queries
#------------------------------------------------------------------------------------------#

"""
	generate_indices(pts)

Returns the `CartesianIndices` of a mesh with `pts[i]` points in each direction.

For scalar input (`Int`), returns 1D `CartesianIndices`. For tuple/vector input,
returns multi-dimensional `CartesianIndices`.
"""
@inline generate_indices(pts::Int) = CartesianIndices((pts,))
@inline generate_indices(pts::NTuple{D, Int}) where {D} = CartesianIndices(pts)
@inline generate_indices(pts::SVector{D, Int}) where {D} = CartesianIndices(Tuple(pts))

"""
	is_boundary_index(idxs::CartesianIndices, idx)

Checks if a given index `idx` lies on the boundary of a `CartesianIndices` domain.
"""
function is_boundary_index(idxs::CartesianIndices{D}, idx) where {D}
    _idx = CartesianIndex(idx)
    @inbounds for i in 1:D
        axis = idxs.indices[i]
        if length(axis) > 1 && (_idx[i] == first(axis) || _idx[i] == last(axis))
            return true
        end
    end
    return false
end

"""
	boundary_indices(idxs::CartesianIndices)

Returns all boundary facets of a `CartesianIndices` domain as a tuple of `CartesianIndices`.
"""
@inline function boundary_indices(idxs::CartesianIndices)
    tup = boundary_symbol_to_cartesian(idxs)
    return ntuple(i -> tup[i], length(tup))
end

"""
	interior_indices(indices::CartesianIndices)

Computes the `CartesianIndices` representing the interior of a given domain, excluding
all boundary points. Dimensions with a length of one or less remain unchanged.
"""
@inline function interior_indices(indices::CartesianIndices{D}) where {D}
    original_ranges = indices.indices

    interior_ranges_tuple = ntuple(Val(D)) do i
        @inbounds r = original_ranges[i]
        if length(r) <= 1
            return r
        else
            (first(r) + 1):(last(r) - 1)
        end
    end

    return CartesianIndices(interior_ranges_tuple)
end

#------------------------------------------------------------------------------------------#
# Field Getters & Index Delegation
#------------------------------------------------------------------------------------------#

"""
	set(Ωₕ::AbstractMeshType)

Returns the geometric set of the domain over which the mesh `Ωₕ` is defined.
"""
@inline set(Ωₕ::AbstractMeshType) = Ωₕ.set

"""
	indices(Ωₕ::AbstractMeshType)

Returns the `CartesianIndices` associated with the points of mesh `Ωₕ`.
"""
@inline indices(Ωₕ::AbstractMeshType) = Ωₕ.indices

"""
	backend(Ωₕ::AbstractMeshType)

Returns the linear algebra [Backend](@ref) associated with the mesh `Ωₕ`.
"""
@inline backend(Ωₕ::AbstractMeshType) = Ωₕ.backend

"""
	markers(Ωₕ::AbstractMeshType)

Returns the `MeshMarkers` dictionary associated with the mesh `Ωₕ`.
"""
@inline markers(Ωₕ::AbstractMeshType) = Ωₕ.markers

"""
	index_in_marker(Ωₕ::AbstractMeshType, label::Symbol)

Returns the `BitVector` associated with the marker `label` in mesh `Ωₕ`.
"""
@inline index_in_marker(Ωₕ::AbstractMeshType, label::Symbol) = markers(Ωₕ)[label]

"""
	set_indices!(Ωₕ::AbstractMeshType, indices)

Overrides the indices in `Ωₕ`. Used internally during mesh refinement.
"""
@inline set_indices!(Ωₕ::AbstractMeshType, indices) = (Ωₕ.indices = indices; return)

@inline is_boundary_index(Ωₕ::AbstractMeshType, idx) = is_boundary_index(indices(Ωₕ), idx)
@inline boundary_indices(Ωₕ::AbstractMeshType) = boundary_indices(indices(Ωₕ))
@inline interior_indices(Ωₕ::AbstractMeshType) = interior_indices(indices(Ωₕ))

#------------------------------------------------------------------------------------------#
# Bounds Checking & Internal Helpers
#------------------------------------------------------------------------------------------#

@noinline _throw_mesh_bounds_error(Ωₕ, idx) = throw(BoundsError(Ωₕ, idx))

@noinline _throw_not_uniform() = throw(ArgumentError("stepsize is only defined for a uniform mesh; use spacing(Ωₕ, idx) on a non-uniform one"))

@inline function _check_point_bounds(Ωₕ::AbstractMeshType, idx::Int, location::String = "point")
    @boundscheck 1 <= idx <= npoints(Ωₕ) || _throw_mesh_bounds_error(Ωₕ, idx)
    return nothing
end

@inline function _check_half_point_bounds(Ωₕ::AbstractMeshType, idx::Int)
    @boundscheck 1 <= idx <= npoints(Ωₕ) + 1 || _throw_mesh_bounds_error(Ωₕ, idx)
    return nothing
end

@inline _extract_linear_index(idx::Int) = idx
@inline _extract_linear_index(idx::CartesianIndex{1}) = idx[1]
@inline _spacing_generator(Ωₕ::AbstractMeshType, spacing_func) = (spacing_func(Ωₕ, i)
for i in 1:npoints(Ωₕ))
@inline _apply_hs_logic(value::T) where {T} = ifelse(iszero(value), one(T), value)

# 1D spacing reference routines
# A mesh with fewer than two points has no interval to measure, so every spacing is zero.
# The single definition of the backward spacing convention, including the boundary case
# where the first point has no interval behind it and repeats the first one. Mesh1D fills
# its cache with this; forward_spacing then reads that cache one entry along.
@inline function _compute_backward_spacing_1d(pts::AbstractVector, i::Int, collapsed::Bool, T::Type)
    if collapsed || length(pts) < 2
        return zero(T)
    elseif i == 1
        return pts[2] - pts[1]
    else
        return pts[i] - pts[i - 1]
    end
end

#------------------------------------------------------------------------------------------#
# High-Level Mesh Constructor Dispatch
#------------------------------------------------------------------------------------------#

# The backend defaults to one over the domain's own element type rather than always to
# Float64, so a Float32 domain gives a Float32 mesh. The element type is a property of the
# storage, and the storage should follow the geometry it is built on; passing `backend`
# explicitly still overrides it, which is how a mesh gets a type the domain does not have.
"""
	$(SIGNATURES)

Returns a [Mesh1D](@ref) or a [MeshnD](@ref) (``D=2,3``) defined on the [Domain](@ref) `Ω`.

The number of points for each coordinate direction is given in `npts`.
The distribution of points on the submeshes is given by `unif` (or keyword `uniform`, default `true`).

# Examples

```julia
I = interval(0.0, 1.0)
Ωₕ = mesh(domain(I), 10)                      # uniform by default
Ωₕ_nonunif = mesh(domain(I), 10, false)       # explicit non-uniform

X = domain(interval(0, 1) × interval(4, 5))
Ωₕ_2d = mesh(X, (10, 15))                     # uniform by default
Ωₕ_mixed = mesh(X, (10, 15), (true, false))
```
"""
@inline mesh(Ω::Domain, npts::NTuple{D, Int}, unif::NTuple{D, Bool};
    backend = backend(eltype(Ω))) where {D} = _mesh(Ω, npts, unif, backend)
@inline mesh(Ω::Domain{CartesianProduct{1, T}}, npts::Int, unif::Bool;
    backend = backend(eltype(Ω))) where {T} = _mesh(Ω, (npts,), (unif,), backend)
@inline mesh(Ω::Domain{CartesianProduct{1, T}}, npts::Int; uniform::Bool = true,
    backend = backend(eltype(Ω))) where {T} = _mesh(Ω, (npts,), (uniform,), backend)
@inline mesh(
    Ω::Domain, npts::NTuple{D, Int}; uniform::NTuple{D, Bool} = ntuple(_ -> true, Val(D)),
    backend = backend(eltype(Ω))) where {D} = _mesh(Ω, npts, uniform, backend)

#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
# Required Interface Methods
#------------------------------------------------------------------------------------------#

"""
	dim(Ωₕ::AbstractMeshType)
	dim(::Type{<:AbstractMeshType})

Returns the spatial dimension ``D`` of the domain where `Ωₕ` is embedded.
"""
@inline dim(::AbstractMeshType{D}) where {D} = D
@inline dim(::Type{<:AbstractMeshType{D}}) where {D} = D

"""
	topo_dim(Ωₕ::AbstractMeshType)

Returns the topological dimension of `Ωₕ`.

The topological dimension counts the number of coordinate axes with more than one point,
identifying degenerate or collapsed dimensions (e.g. lines or points embedded in 2D/3D).
"""
@inline function topo_dim(Ωₕ::AbstractMeshType{D}) where {D}
    count = 0
    @inbounds for i in 1:D
        npoints(Ωₕ(i)) > 1 && (count += 1)
    end
    return count
end

"""
	eltype(Ωₕ::AbstractMeshType)
	eltype(::Type{<:AbstractMeshType})

Returns the floating-point coordinate element type of the points in `Ωₕ`.
"""
function eltype(Ωₕ::AbstractMeshType)
    error("Interface function 'eltype' not implemented for mesh of type $(typeof(Ωₕ)).")
end

function eltype(::Type{<:AbstractMeshType})
    error("Interface function 'eltype(::Type{...})' not implemented for mesh type.")
end

"""
	points(Ωₕ::AbstractMeshType)

Returns the coordinates of the mesh points:
- For 1D meshes (`Mesh1D`): returns a coordinate vector `Vector{T}` of length ``N_x``.
- For nD meshes (`MeshnD`): returns an `NTuple{D, Vector{T}}` containing the 1D coordinate vectors along each axis.

See also: [`point`](@ref), [`points_iterator`](@ref).
"""
function points end

"""
	point(Ωₕ::AbstractMeshType, idx)

Returns the coordinate point at index `idx` (linear integer, tuple `(i, j)`, or `CartesianIndex`):
- For 1D meshes: scalar coordinate ``x_i``.
- For nD meshes: coordinate tuple ``(x_{i_1}, \\dots, x_{i_D})``.

Direct indexing `Ωₕ[idx]` is also supported.
"""
function point end

"""
	points_iterator(Ωₕ::AbstractMeshType)

Returns an iterator yielding coordinate points across the entire mesh.
"""
function points_iterator end

"""
	half_points(Ωₕ::AbstractMeshType)

Returns the precomputed cell centers (half-points) for each coordinate axis:
```math
x_{i+1/2} = \\frac{x_i + x_{i+1}}{2}, \\quad i = 1, \\dots, N-1.
```
"""
function half_points end

"""
	half_point(Ωₕ::AbstractMeshType, idx)

Returns the cell center (half-point) coordinate corresponding to index `idx`.
"""
function half_point end

"""
	half_points_iterator(Ωₕ::AbstractMeshType)

Returns an iterator over cell center (half-point) coordinates.
"""
function half_points_iterator end

"""
	spacing(Ωₕ::AbstractMeshType, idx)

Returns the backward spacing ``h_i = x_i - x_{i-1}`` at index `idx` (for ``i=1``, returns ``x_2 - x_1``).
For nD meshes, returns a tuple of backward spacings along each axis.
"""
function spacing end

"""
	spacings_iterator(Ωₕ::AbstractMeshType)

Returns an iterator over backward spacings across mesh points.
"""
function spacings_iterator end

"""
	forward_spacing(Ωₕ::AbstractMeshType, idx)

Returns the forward spacing ``h_{i+1} = x_{i+1} - x_i`` at index `idx` (for ``i=N``, returns ``x_N - x_{N-1}``).
For nD meshes, returns a tuple of forward spacings along each axis.
"""
function forward_spacing end

"""
	forward_spacings_iterator(Ωₕ::AbstractMeshType)

Returns an iterator over forward spacings across mesh points.
"""
function forward_spacings_iterator end

"""
	half_spacings(Ωₕ::AbstractMeshType)

Returns the cell widths (half-spacings) ``h_{i+1/2} = \\frac{h_i + h_{i+1}}{2}`` along each axis.
"""
function half_spacings end

"""
	half_spacing(Ωₕ::AbstractMeshType, idx)

Returns the cell width (half-spacing) at index `idx`.
"""
function half_spacing end

"""
	half_spacings_iterator(Ωₕ::AbstractMeshType)

Returns an iterator over cell widths (half-spacings).
"""
function half_spacings_iterator end

"""
	npoints(Ωₕ::AbstractMeshType, [::Type{Tuple}])

Returns the total number of points in `Ωₕ`.
When passing `Tuple` as the second argument, returns a tuple with the number of points along each dimension.
"""
function npoints end

"""
	hₘₐₓ(Ωₕ::AbstractMeshType)

Returns the maximum diagonal stepsize across all cells in the mesh:
```math
h_{\\max} = \\max_{idx} \\| (h_{1, idx_1}, \\dots, h_{D, idx_D}) \\|_2.
```
"""
function hₘₐₓ end

"""
	hₘᵢₙ(Ωₕ::AbstractMeshType)

Returns the diagonal of the smallest cell in the mesh, the counterpart of [`hₘₐₓ`](@ref):
- In 1D: ``\\min_i (x_i - x_{i-1})``.
- In nD:

```math
h_{\\min} = \\min_{idx} \\| (h_{1, idx_1}, \\dots, h_{D, idx_D}) \\|_2.
```

This is a diagonal rather than an edge length, so that `hₘₐₓ` and `hₘᵢₙ` measure the same
kind of quantity. For the smallest extent along one coordinate, ask that submesh:
`hₘᵢₙ(Ωₕ(i))`.
"""
function hₘᵢₙ end

"""
	cell_measure(Ωₕ::AbstractMeshType, idx)

Returns the control volume (length, area, or volume) of the cell centered at index `idx`:
```math
\\text{meas}(\\square_{idx}) = \\prod_{d=1}^D h_{d, idx_d+1/2}.
```
"""
function cell_measure end

"""
	cell_measures_iterator(Ωₕ::AbstractMeshType)

Returns an iterator yielding the volume/measure of each cell in the mesh.
"""
function cell_measures_iterator end

"""
	iterative_refinement!(Ωₕ::AbstractMeshType, [domain_markers::DomainMarkers])

Refines the mesh `Ωₕ` in-place by halving each existing cell (inserting new points at midpoints).
If domain markers are supplied, they are re-evaluated onto the refined grid points.
"""
function iterative_refinement! end

"""
	change_points!(Ωₕ::AbstractMeshType, [domain_markers::DomainMarkers], pts)

Updates the coordinates of mesh `Ωₕ` in-place using new point coordinates in `pts`,
recalculating all cached half-points and cell spacings.
"""
function change_points! end

#------------------------------------------------------------------------------------------#
# Uniformity Query
#------------------------------------------------------------------------------------------#

"""
	is_uniform(Ωₕ::AbstractMeshType; tol=1e-10)

Checks if the mesh has uniform spacing (within numerical tolerance).
"""
function is_uniform(Ωₕ::AbstractMeshType{1}; tol = 1e-10)
    n = npoints(Ωₕ)
    if n <= 1
        return true
    end

    h_ref = spacing(Ωₕ, 1)
    @inbounds for i in 2:n
        if abs(spacing(Ωₕ, i) - h_ref) >= tol
            return false
        end
    end
    return true
end

function is_uniform(Ωₕ::AbstractMeshType{D}; tol = 1e-10) where {D}
    return all(i -> is_uniform(Ωₕ(i); tol = tol), 1:D)
end

#------------------------------------------------------------------------------------------#
# Base Collection Interface
#------------------------------------------------------------------------------------------#

"""
	Base.size(Ωₕ::AbstractMeshType)
	Base.size(Ωₕ::AbstractMeshType, d::Integer)

Returns the tuple of point counts along each spatial dimension, matching `npoints(Ωₕ, Tuple)`.
"""
@inline Base.size(Ωₕ::AbstractMeshType) = npoints(Ωₕ, Tuple)
@inline Base.size(Ωₕ::AbstractMeshType, d::Integer) = npoints(Ωₕ, Tuple)[d]

"""
	Base.length(Ωₕ::AbstractMeshType)

Returns the total number of points in `Ωₕ`, matching `npoints(Ωₕ)`.
"""
@inline Base.length(Ωₕ::AbstractMeshType) = npoints(Ωₕ)

"""
	Base.axes(Ωₕ::AbstractMeshType)
	Base.axes(Ωₕ::AbstractMeshType, d::Integer)

Returns the axes of the mesh's `CartesianIndices`.
"""
@inline Base.axes(Ωₕ::AbstractMeshType) = axes(indices(Ωₕ))
@inline Base.axes(Ωₕ::AbstractMeshType, d::Integer) = axes(indices(Ωₕ), d)

"""
	Base.firstindex(Ωₕ::AbstractMeshType)
	Base.firstindex(Ωₕ::AbstractMeshType, d::Integer)

Returns the first valid index of `Ωₕ`.
"""
@inline Base.firstindex(::AbstractMeshType{1}) = 1
@inline Base.firstindex(Ωₕ::AbstractMeshType{D}) where {D} = first(indices(Ωₕ))
@inline Base.firstindex(Ωₕ::AbstractMeshType, d::Integer) = 1

"""
	Base.lastindex(Ωₕ::AbstractMeshType)
	Base.lastindex(Ωₕ::AbstractMeshType, d::Integer)

Returns the last valid index of `Ωₕ`.
"""
@inline Base.lastindex(Ωₕ::AbstractMeshType{1}) = npoints(Ωₕ)
@inline Base.lastindex(Ωₕ::AbstractMeshType{D}) where {D} = last(indices(Ωₕ))
@inline Base.lastindex(Ωₕ::AbstractMeshType, d::Integer) = size(Ωₕ, d)

"""
	Base.iterate(Ωₕ::AbstractMeshType, [state])

Iterates over all grid points of `Ωₕ`, returning coordinates `point(Ωₕ, idx)` for each index.
"""
@inline function Base.iterate(Ωₕ::AbstractMeshType{1}, state = 1)
    state > npoints(Ωₕ) && return nothing
    return (point(Ωₕ, state), state + 1)
end

@inline function Base.iterate(Ωₕ::AbstractMeshType{D}, state = iterate(indices(Ωₕ))) where {D}
    state === nothing && return nothing
    idx, next_state = state
    return (point(Ωₕ, idx), iterate(indices(Ωₕ), next_state))
end

#------------------------------------------------------------------------------------------#
# Advanced Mesh Queries
#------------------------------------------------------------------------------------------#

"""
	stepsize(Ωₕ::AbstractMeshType)
	stepsize(Ωₕ::AbstractMeshType, d::Integer)

Returns the constant stepsize for a uniform mesh:
- In 1D: returns scalar ``h = x_2 - x_1``.
- In nD: returns a tuple ``(h_1, \\dots, h_D)`` of stepsizes along each coordinate axis.
- When `d` is specified: returns the stepsize along dimension `d`.

Throws an error if the mesh is not uniform.

See also: [`is_uniform`](@ref), [`spacing`](@ref).
"""
@inline function stepsize(Ωₕ::AbstractMeshType{1})
    is_uniform(Ωₕ) || _throw_not_uniform()
    npoints(Ωₕ) <= 1 && return zero(eltype(Ωₕ))
    return spacing(Ωₕ, 2)
end

@inline function stepsize(Ωₕ::AbstractMeshType{D}) where {D}
    is_uniform(Ωₕ) || _throw_not_uniform()
    return ntuple(i -> stepsize(Ωₕ(i)), Val(D))
end

@inline stepsize(Ωₕ::AbstractMeshType, d::Integer) = stepsize(Ωₕ(d))

"""
	locate_cell(Ωₕ::AbstractMeshType, x)

Locates the cell containing continuous coordinate `x`:
- For 1D meshes: returns integer index `i \\in 1:N-1` such that ``x_i \\le x \\le x_{i+1}``
  (clamped to the domain boundaries).
- For nD meshes: returns a `CartesianIndex{D}` locating the bounding cell along each dimension.

# Examples
```julia
Ωₕ = mesh(domain(interval(0.0, 1.0)), 11)  # h = 0.1
locate_cell(Ωₕ, 0.35)  # returns 4 (interval [0.3, 0.4])
```
"""
function locate_cell end
@inline locate_cell(Ωₕ::AbstractMeshType{D}, x::AbstractVector) where {D} = locate_cell(Ωₕ, Tuple(x))

"""
	normal_vector(::AbstractMeshType{D}, symbol::Symbol)
	normal_vector(Val(D), symbol::Symbol)
	normal_vector(symbol::Symbol)

Returns the outward unit normal vector (as an `SVector{D, Float64}`) associated with a standard
boundary facet label (`:left`, `:right`, `:bottom`, `:top`, `:front`, `:back`).

# Conventions
- 1D:
  - `:left`  ``\\to (-1.0)``
  - `:right` ``\\to (+1.0)``
- 2D:
  - `:left`   ``\\to (-1.0, 0.0)``
  - `:right`  ``\\to (+1.0, 0.0)``
  - `:bottom` ``\\to (0.0, -1.0)``
  - `:top`    ``\\to (0.0, +1.0)``
- 3D:
  - `:back`   ``\\to (-1.0, 0.0, 0.0)``
  - `:front`  ``\\to (+1.0, 0.0, 0.0)``
  - `:left`   ``\\to (0.0, -1.0, 0.0)``
  - `:right`  ``\\to (0.0, +1.0, 0.0)``
  - `:bottom` ``\\to (0.0, 0.0, -1.0)``
  - `:top`    ``\\to (0.0, 0.0, +1.0)``

See also: [`get_boundary_symbols`](@ref).
"""
@inline normal_vector(::AbstractMeshType{D}, symbol::Symbol) where {D} = normal_vector(Val(D), symbol)

@inline function normal_vector(::Val{1}, symbol::Symbol)
    symbol === :left && return SVector{1, Float64}(-1.0)
    symbol === :right && return SVector{1, Float64}(1.0)
    throw(ArgumentError("Unknown 1D boundary symbol: :$symbol. Expected :left or :right."))
end

@inline function normal_vector(::Val{2}, symbol::Symbol)
    symbol === :left && return SVector{2, Float64}(-1.0, 0.0)
    symbol === :right && return SVector{2, Float64}(1.0, 0.0)
    symbol === :bottom && return SVector{2, Float64}(0.0, -1.0)
    symbol === :top && return SVector{2, Float64}(0.0, 1.0)
    throw(ArgumentError("Unknown 2D boundary symbol: :$symbol. Expected :left, :right, :bottom, or :top."))
end

@inline function normal_vector(::Val{3}, symbol::Symbol)
    symbol === :back && return SVector{3, Float64}(-1.0, 0.0, 0.0)
    symbol === :front && return SVector{3, Float64}(1.0, 0.0, 0.0)
    symbol === :left && return SVector{3, Float64}(0.0, -1.0, 0.0)
    symbol === :right && return SVector{3, Float64}(0.0, 1.0, 0.0)
    symbol === :bottom && return SVector{3, Float64}(0.0, 0.0, -1.0)
    symbol === :top && return SVector{3, Float64}(0.0, 0.0, 1.0)
    throw(ArgumentError("Unknown 3D boundary symbol: :$symbol. Expected :left, :right, :bottom, :top, :front, or :back."))
end
