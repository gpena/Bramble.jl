"""
# queries.jl

Uniformity checks, the `Base` collection interface, and higher-level spatial queries
shared by every mesh type in Bramble.

- `is_uniform`: uniform-spacing check.
- `Base.size`, `Base.length`, `Base.axes`, `Base.firstindex`, `Base.lastindex`,
  `Base.iterate`: the collection interface over mesh points.
- `stepsize`, `locate_cell`, `normal_vector`: constant-spacing, cell lookup, and boundary
  normal queries.

See also: [`Mesh1D`](@ref), [`MeshnD`](@ref)
"""

#------------------------------------------------------------------------------------------#
# Uniformity Query
#------------------------------------------------------------------------------------------#

"""
    is_uniform(Ωₕ::AbstractMeshType; tol = 1e-10) -> Bool

Check whether the mesh has uniform spacing (within numerical tolerance `tol`).
"""
function is_uniform(Ωₕ::AbstractMeshType{1}; tol=1e-10)
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

function is_uniform(Ωₕ::AbstractMeshType{D}; tol=1e-10) where {D}
    return all(i -> is_uniform(Ωₕ(i); tol=tol), 1:D)
end

#------------------------------------------------------------------------------------------#
# Base Collection Interface
#------------------------------------------------------------------------------------------#

"""
    Base.size(Ωₕ::AbstractMeshType) -> NTuple{D, Int}
    Base.size(Ωₕ::AbstractMeshType, d::Integer) -> Int

Return the tuple of point counts along each spatial dimension, matching `npoints(Ωₕ, Tuple)`.
"""
@inline Base.size(Ωₕ::AbstractMeshType) = npoints(Ωₕ, Tuple)
@inline Base.size(Ωₕ::AbstractMeshType, d::Integer) = npoints(Ωₕ, Tuple)[d]

"""
    Base.length(Ωₕ::AbstractMeshType) -> Int

Return the total number of points in `Ωₕ`, matching `npoints(Ωₕ)`.
"""
@inline Base.length(Ωₕ::AbstractMeshType) = npoints(Ωₕ)

"""
    Base.axes(Ωₕ::AbstractMeshType)
    Base.axes(Ωₕ::AbstractMeshType, d::Integer)

Return the axes of the mesh's `CartesianIndices`.
"""
@inline Base.axes(Ωₕ::AbstractMeshType) = axes(indices(Ωₕ))
@inline Base.axes(Ωₕ::AbstractMeshType, d::Integer) = axes(indices(Ωₕ), d)

"""
    Base.firstindex(Ωₕ::AbstractMeshType)
    Base.firstindex(Ωₕ::AbstractMeshType, d::Integer)

Return the first valid index of `Ωₕ`.
"""
@inline Base.firstindex(::AbstractMeshType{1}) = 1
@inline Base.firstindex(Ωₕ::AbstractMeshType{D}) where {D} = first(indices(Ωₕ))
@inline Base.firstindex(Ωₕ::AbstractMeshType, d::Integer) = 1

"""
    Base.lastindex(Ωₕ::AbstractMeshType)
    Base.lastindex(Ωₕ::AbstractMeshType, d::Integer)

Return the last valid index of `Ωₕ`.
"""
@inline Base.lastindex(Ωₕ::AbstractMeshType{1}) = npoints(Ωₕ)
@inline Base.lastindex(Ωₕ::AbstractMeshType{D}) where {D} = last(indices(Ωₕ))
@inline Base.lastindex(Ωₕ::AbstractMeshType, d::Integer) = size(Ωₕ, d)

"""
    Base.iterate(Ωₕ::AbstractMeshType, [state])

Iterate over all grid points of `Ωₕ`, returning coordinates `point(Ωₕ, idx)` for each index.
"""
@inline function Base.iterate(Ωₕ::AbstractMeshType{1}, state=1)
    state > npoints(Ωₕ) && return nothing
    return (point(Ωₕ, state), state + 1)
end

@inline function Base.iterate(Ωₕ::AbstractMeshType{D}, state=iterate(indices(Ωₕ))) where {D}
    state === nothing && return nothing
    idx, next_state = state
    return (point(Ωₕ, idx), iterate(indices(Ωₕ), next_state))
end

#------------------------------------------------------------------------------------------#
# Advanced Mesh Queries
#------------------------------------------------------------------------------------------#

"""
    stepsize(Ωₕ::AbstractMeshType) -> Union{Real, NTuple{D, Real}}
    stepsize(Ωₕ::AbstractMeshType, d::Integer) -> Real

Return the constant stepsize for a uniform mesh:
  - In 1D: returns scalar ``h = x_2 - x_1``.
  - In nD: returns a tuple ``(h_1, \\dots, h_D)`` of stepsizes along each coordinate axis.
  - When `d` is specified: returns the stepsize along dimension `d`.

Throws an `ArgumentError` if the mesh is not uniform.

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
    locate_cell(Ωₕ::AbstractMeshType{1}, x::Real) -> Int
    locate_cell(Ωₕ::AbstractMeshType{D}, x) -> CartesianIndex{D}

Locate the cell containing continuous coordinate `x`:
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
@inline locate_cell(Ωₕ::AbstractMeshType{D}, x::AbstractVector) where {D} =
    locate_cell(Ωₕ, Tuple(x))

"""
    normal_vector(Ωₕ::AbstractMeshType{D}, symbol::Symbol) -> NTuple{D, Float64}
    normal_vector(::Val{D}, symbol::Symbol) -> NTuple{D, Float64}

Return the outward unit normal vector (as an `NTuple{D, Float64}`) associated with a standard
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

See also: [`boundary_symbols`](@ref).
"""
@inline normal_vector(::AbstractMeshType{D}, symbol::Symbol) where {D} =
    normal_vector(Val(D), symbol)

@inline function normal_vector(::Val{1}, symbol::Symbol)
    symbol === :left && return (-1.0,)
    symbol === :right && return (1.0,)
    throw(ArgumentError("Unknown 1D boundary symbol: :$symbol. Expected :left or :right."))
end

@inline function normal_vector(::Val{2}, symbol::Symbol)
    symbol === :left && return (-1.0, 0.0)
    symbol === :right && return (1.0, 0.0)
    symbol === :bottom && return (0.0, -1.0)
    symbol === :top && return (0.0, 1.0)
    throw(
        ArgumentError(
            "Unknown 2D boundary symbol: :$symbol. Expected :left, :right, :bottom, or :top.",
        ),
    )
end

@inline function normal_vector(::Val{3}, symbol::Symbol)
    symbol === :back && return (-1.0, 0.0, 0.0)
    symbol === :front && return (1.0, 0.0, 0.0)
    symbol === :left && return (0.0, -1.0, 0.0)
    symbol === :right && return (0.0, 1.0, 0.0)
    symbol === :bottom && return (0.0, 0.0, -1.0)
    symbol === :top && return (0.0, 0.0, 1.0)
    throw(
        ArgumentError(
            "Unknown 3D boundary symbol: :$symbol. Expected :left, :right, :bottom, :top, :front, or :back.",
        ),
    )
end
