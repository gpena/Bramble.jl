# queries.jl
#
# Uniformity checks, the `Base` collection interface, and higher-level spatial queries
# shared by every mesh type in Bramble.
#
# - `is_uniform`: uniform-spacing check.
# - `Base.size`, `Base.length`, `Base.axes`, `Base.firstindex`, `Base.lastindex`,
#   `Base.iterate`: the collection interface over mesh points.
# - `stepsize`, `locate_cell`, `normal_vector`: constant-spacing, cell lookup, and boundary
#   normal queries.
#
# See also: `Mesh1D`, `MeshnD`

#------------------------------------------------------------------------------------------#
# Uniformity Query
#------------------------------------------------------------------------------------------#

"""
    is_uniform(Ωₕ::AbstractMeshType; tol = 1e-10) -> Bool

Check whether the mesh has uniform spacing (within numerical tolerance `tol`).
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

**`x` outside `[x_1, x_N]` is silently clamped to the boundary cell** -- `locate_cell` never
throws and never signals that `x` was out of range, so a caller that assumes the returned
cell means `x` was inside the mesh inherits that assumption unknowingly. This is exactly
what [`interpolate_at`](@ref)'s `outside` keyword (gpena/Bramble.jl#223) exists to make an
explicit, checked choice about one layer up, rather than leaving to a bare cell lookup with
no way to say "no" -- a new caller of `locate_cell` directly should decide its own
out-of-range policy the same way, not assume this one already did.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0)), 11)  # h = 0.1
locate_cell(Ωₕ, 0.35)  # returns 4 (interval [0.3, 0.4])
locate_cell(Ωₕ, 5.0)   # returns 10 (the last cell) -- no error, x = 5.0 is well outside [0, 1]
```
"""
function locate_cell end
@inline locate_cell(Ωₕ::AbstractMeshType{D}, x::AbstractVector) where {D} = locate_cell(Ωₕ, Tuple(x))

"""
    normal_vector(Ωₕ::AbstractMeshType{D}, symbol::Symbol) -> NTuple{D, Float64}
    normal_vector(::Val{D}, symbol::Symbol) -> NTuple{D, Float64}

Return the outward unit normal vector (as an `NTuple{D, Float64}`) associated with a standard
boundary facet label (`:xmin`, `:xmax`, `:ymin`, `:ymax`, `:zmin`, `:zmax`) or legacy viewpoint alias
(`:left`, `:right`, `:bottom`, `:top`, `:front`, `:back`).

# Conventions

  - 1D:
      - `:xmin`, `:left`  ``\\to (-1.0)``
      - `:xmax`, `:right` ``\\to (+1.0)``
  - 2D:
      - `:xmin`, `:left`   ``\\to (-1.0, 0.0)``
      - `:xmax`, `:right`  ``\\to (+1.0, 0.0)``
      - `:ymin`, `:bottom` ``\\to (0.0, -1.0)``
      - `:ymax`, `:top`    ``\\to (0.0, +1.0)``
  - 3D:
      - `:xmin`, `:back`   ``\\to (-1.0, 0.0, 0.0)``
      - `:xmax`, `:front`  ``\\to (+1.0, 0.0, 0.0)``
      - `:ymin`, `:left`   ``\\to (0.0, -1.0, 0.0)``
      - `:ymax`, `:right`  ``\\to (0.0, +1.0, 0.0)``
      - `:zmin`, `:bottom` ``\\to (0.0, 0.0, -1.0)``
      - `:zmax`, `:top`    ``\\to (0.0, 0.0, +1.0)``

See also: [`boundary_symbols`](@ref).
"""
@inline normal_vector(::AbstractMeshType{D}, symbol::Symbol) where {D} = normal_vector(Val(D), symbol)

@inline function normal_vector(::Val{1}, symbol::Symbol)
    (symbol === :xmin || symbol === :left) && return (-1.0,)
    (symbol === :xmax || symbol === :right) && return (1.0,)
    throw(ArgumentError("Unknown 1D boundary symbol: :$symbol. Expected :xmin/:left or :xmax/:right."))
end

@inline function normal_vector(::Val{2}, symbol::Symbol)
    (symbol === :xmin || symbol === :left) && return (-1.0, 0.0)
    (symbol === :xmax || symbol === :right) && return (1.0, 0.0)
    (symbol === :ymin || symbol === :bottom) && return (0.0, -1.0)
    (symbol === :ymax || symbol === :top) && return (0.0, 1.0)
    throw(
        ArgumentError(
        "Unknown 2D boundary symbol: :$symbol. Expected :xmin/:left, :xmax/:right, :ymin/:bottom, or :ymax/:top.",
    ),
    )
end

@inline function normal_vector(::Val{3}, symbol::Symbol)
    (symbol === :xmin || symbol === :back) && return (-1.0, 0.0, 0.0)
    (symbol === :xmax || symbol === :front) && return (1.0, 0.0, 0.0)
    (symbol === :ymin || symbol === :left) && return (0.0, -1.0, 0.0)
    (symbol === :ymax || symbol === :right) && return (0.0, 1.0, 0.0)
    (symbol === :zmin || symbol === :bottom) && return (0.0, 0.0, -1.0)
    (symbol === :zmax || symbol === :top) && return (0.0, 0.0, 1.0)
    throw(
        ArgumentError(
        "Unknown 3D boundary symbol: :$symbol. Expected :xmin/:back, :xmax/:front, :ymin/:left, :ymax/:right, :zmin/:bottom, or :zmax/:top.",
    ),
    )
end

################################################################################
#              (D-1)-dimensional surface weights on grid faces                 #
################################################################################

#=
The lumped quadrature weight of a union of axis-aligned grid faces (gpena/Bramble.jl#157).

At a grid point `p`, with `N(p)` the set of normal directions of the surface pieces through
`p`,

    ω(p) = Σ_{d ∈ N(p)} ∏_{e ≠ d} half_spacing(Ωₕ(e), I[e])

which is what the general definition

    ω(p) = 2^{-(D-1)} Σ_{F ⊂ Γ, p ∈ F} |F|

collapses to once every face with a given normal is present: the `2^(D-1)` faces with normal
`d` differ by taking the forward or the backward cell on each transverse axis, so their sum
factorises into a product of averaged spacings, and `half_spacing` is exactly that average.

Corners need no special case, because `half_spacing!` (mesh/mesh1d.jl) already truncates at
the two ends: `half_spacings[1] = h₁/2` and `half_spacings[n] = h_N/2` are the one-sided
halves a point at the end of a surface needs. A corner of the unit square therefore receives
`(h₁ + k₁)/2`, the sum of its two incident edges' halves, and a marker union needs no
special handling either: `ω(:xmin) + ω(:ymin) == ω(:xmin, :ymin)` pointwise.

**1D is counting measure, not a limit of the 2D formula.** A `(D-1)`-face is then a *point*,
of measure 1, the product over `e ≠ d` is empty, and `ω ≡ 1`. That is the correct pairing for
a 1D Neumann term `g·v|∂Ω`: dimensionless, with no spacing anywhere. It falls out of the
empty product rather than needing a branch, but a reader carrying 2D intuition will look for
an `h/2` that must not be there.

The face set is carried as a mask, `NTuple{D, NTuple{2, Bool}}` -- per axis, whether the
`min` and the `max` face belong to the surface. Its *type* is fixed by `D` alone, so the
weight is type-stable and allocation-free whether the mask is a runtime value (the numeric
`inner_Γ`) or a type parameter (the symbolic `InnerGamma`, form/operators/inner.jl).
=#

# Canonical symbols and the legacy viewpoint aliases, per dimension. The 3D aliases are not
# the 2D ones extended: there `:left`/`:right` name the y faces and `:front`/`:back` the x
# ones (mesh/marker.jl, `boundary_symbol_to_cartesian`). Read from there rather than guessed.
@inline function _face_of_symbol(::Val{1}, s::Symbol)
    (s === :xmin || s === :left) && return (1, 1)
    (s === :xmax || s === :right) && return (1, 2)
    return _throw_not_a_face(s, 1)
end

@inline function _face_of_symbol(::Val{2}, s::Symbol)
    (s === :xmin || s === :left) && return (1, 1)
    (s === :xmax || s === :right) && return (1, 2)
    (s === :ymin || s === :bottom) && return (2, 1)
    (s === :ymax || s === :top) && return (2, 2)
    return _throw_not_a_face(s, 2)
end

@inline function _face_of_symbol(::Val{3}, s::Symbol)
    (s === :xmin || s === :back) && return (1, 1)
    (s === :xmax || s === :front) && return (1, 2)
    (s === :ymin || s === :left) && return (2, 1)
    (s === :ymax || s === :right) && return (2, 2)
    (s === :zmin || s === :bottom) && return (3, 1)
    (s === :zmax || s === :top) && return (3, 2)
    return _throw_not_a_face(s, 3)
end

@noinline function _throw_not_a_face(s::Symbol, D::Int)
    throw(
        ArgumentError(
        "inner_Γ integrates over whole coordinate faces, and :$s does not name one in $(D)D. " *
        "Expected `:boundary` or a canonical face symbol (:xmin, :xmax" *
        (D > 1 ? ", :ymin, :ymax" : "") * (D > 2 ? ", :zmin, :zmax" : "") *
        ") or one of its aliases. A user-defined marker naming part of a face, an interior " *
        "interface or a staircase is a genuinely more general surface: its weight does not " *
        "factorise where the surface is cut at an interior index, so it needs the one-sided " *
        "face sum, which is not implemented yet.",
    ),
    )
end

"""
    _face_mask(::Val{D}, labels::NTuple{N, Symbol}) -> NTuple{D, NTuple{2, Bool}}

The face set `labels` names, as a per-axis `(min, max)` mask.

`:boundary` is every face. Everything else has to name a whole coordinate face; see
[`inner_Γ`](@ref) for why a general marked set is a different computation.
"""
@inline function _face_mask(::Val{D}, labels::NTuple{N, Symbol}) where {D, N}
    mask = ntuple(_ -> (false, false), Val(D))
    for s in labels
        mask = s === :boundary ? ntuple(_ -> (true, true), Val(D)) :
               _add_face(mask, _face_of_symbol(Val(D), s), Val(D))
    end
    return mask
end

@inline function _add_face(mask::NTuple{D, NTuple{2, Bool}}, face, ::Val{D}) where {D}
    axis, side = face
    return ntuple(Val(D)) do d
        d == axis ? (mask[d][1] || side == 1, mask[d][2] || side == 2) : mask[d]
    end
end

@inline _no_faces(mask) = all(m -> !m[1] && !m[2], mask)

"""
    _check_surface_is_thin(Ωₕ::AbstractMeshType{D}, mask) -> Nothing

Refuse a face set that is not a surface on this mesh.

An axis carrying both its faces and at most two points puts *every* grid point on the set, so
it has dimension `D` rather than `D - 1` and no `(D-1)`-dimensional weight is defined on it.
It fires on a 2x2 mesh of the square asked for `:boundary`, and on any mesh too coarse to
separate the two ends of an axis; it never fires with three or more points on every axis.
"""
@inline function _check_surface_is_thin(Ωₕ::AbstractMeshType{D}, mask) where {D}
    np = npoints(Ωₕ, Tuple)
    for d in 1:D
        mask[d][1] && mask[d][2] && np[d] <= 2 && _throw_surface_not_thin(d, np[d])
    end
    return nothing
end

@noinline function _throw_surface_not_thin(d::Int, n::Int)
    throw(
        ArgumentError(
        "the requested surface is not (D-1)-dimensional on this mesh: axis $d carries both " *
        "of its faces and only $n point(s), so every grid point lies on the surface and no " *
        "surface weight is defined. Refine that axis to at least three points.",
    ),
    )
end

"""
    _surface_weight(Ωₕ::AbstractMeshType{D}, mask, I::CartesianIndex{D}) -> Real

The lumped surface weight `ω(I)` of the face set `mask`, zero off the surface.

Computed from the mesh's live `half_spacings` on every call: there is no vector to store, key
by label set, or stamp with a mesh version, so `SpaceWeights` (space/scalar_gridspace.jl)
needs no fourth family and the staleness question does not arise here.
"""
@inline function _surface_weight(
        Ωₕ::AbstractMeshType{D}, mask, I::CartesianIndex{D}
) where {D}
    return _surface_weight_dir(Ωₕ, mask, I, npoints(Ωₕ, Tuple), Val(D), Val(D))
end

# One direction per rung, recursing on `Val(d)` rather than summing a comprehension: a
# closure over `d` boxes it (gpena/Bramble.jl#146), and this is called per grid point.
@inline function _surface_weight_dir(
        Ωₕ, mask, I, np, ::Val{d}, ::Val{D}
) where {d, D}
    on_face = (mask[d][1] && I[d] == 1) || (mask[d][2] && I[d] == np[d])
    m = _transverse_measure(Ωₕ, I, Val(d), Val(D))
    return ifelse(on_face, m, zero(m)) +
           _surface_weight_dir(Ωₕ, mask, I, np, Val(d - 1), Val(D))
end

@inline _surface_weight_dir(Ωₕ, mask, I, np, ::Val{0}, ::Val{D}) where {D} = zero(
    eltype(Ωₕ)
)

# `∏_{e ≠ d}`, empty in 1D, so a 1D face weighs 1 rather than half a cell.
@inline function _transverse_measure(Ωₕ, I, ::Val{d}, ::Val{D}) where {d, D}
    return prod(
        ntuple(Val(D)) do e
        e == d ? one(eltype(Ωₕ)) :
        _apply_hs_logic(half_spacing(Ωₕ(e), I[e]))
    end
    )
end

@inline _transverse_measure(Ωₕ, I, ::Val{1}, ::Val{1}) = one(eltype(Ωₕ))
