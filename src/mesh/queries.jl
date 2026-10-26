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
    is_uniform(Ωₕ::AbstractMeshType; tol = nothing) -> Bool

Check whether the mesh has uniform spacing, within a numerical tolerance.

When `tol` is not given, the tolerance is `_default_is_uniform_tol(eltype(Ωₕ), h₁, n, mag)`,
with `mag` the larger end-point magnitude, which keeps today's `1e-10` for a `Float64` mesh
near the origin and widens it for `Float32` or far from the origin (gpena/Bramble.jl#307,
#492); see that function for the measurement behind the constant. Passing `tol` explicitly uses it
verbatim, with no further scaling.

On a [`Mesh1D`](@ref), the default-tolerance (`tol === nothing`) answer is the `uniform` flag
its [`Mesh1DState`](@ref) stores, computed whenever the points change
(gpena/Bramble.jl#332, #437): O(1), and the mesh is never written. Passing `tol` explicitly
always rescans the spacings.
"""
function is_uniform(Ωₕ::AbstractMeshType{1}; tol = nothing)
    tol === nothing && return _stored_uniform(Ωₕ)
    h = host_spacings(Ωₕ)
    n = length(h)
    n <= 1 && return true
    return _uniform_scan(h, h[1], tol, n)
end

@inline _stored_uniform(Ωₕ::AbstractMeshType{1}) = _walk_mesh(Ωₕ(1)).uniform

# The default-tolerance answer for the host spacings `h` of a mesh of element type `T` whose
# end points have magnitude at most `mag`: what a `Mesh1DState` stores as `uniform`.
@inline function _uniform_at_default_tol(h, ::Type{T}, mag) where {T}
    n = length(h)
    n <= 1 && return true
    h_ref = h[1]
    return _uniform_scan(h, h_ref, _default_is_uniform_tol(T, h_ref, n, mag), n)
end

@inline function _uniform_scan(h, h_ref, atol, n)
    @inbounds for i in 2:n
        if abs(h[i] - h_ref) >= atol
            return false
        end
    end
    return true
end

function is_uniform(Ωₕ::AbstractMeshType{D}; tol = nothing) where {D}
    return all(ntuple(i -> is_uniform(Ωₕ(i); tol = tol), Val(D)))
end

# Float64 keeps the historical bare `1e-10`: eps(Float64) times a mesh span of order 1 is
# ~2e-16, so the floor below always wins and nothing changes for an existing Float64 mesh
# (gpena/Bramble.jl#307's constraint 1).
#
# Float32 is Metal's only floating type, and its uniform-mesh construction
# (`x[i] = a + (i - 1) * h` in `_points!`, then `pts[i] - pts[i - 1]` in `spacing!`,
# `src/mesh/mesh1d.jl`) leaves a few ULPs of drift between spacings that are mathematically
# equal. That drift tracks the mesh's span, not `h_ref` alone -- `n * abs(h_ref)` estimates
# the span of a uniform mesh without any extra array read -- so the eps term below scales
# with it instead of with `h_ref` directly, which would need a forever-growing constant as
# the mesh is refined (`h_ref` shrinks while the span, and the rounding it carries, does
# not). The span is used as-is, with no lower clamp: clamping it to `one(T)` would pin the
# tolerance at `4 * eps(T)` for a sub-unit domain while the mesh's own spacings keep
# shrinking with it, so a genuinely non-uniform mesh on, say, a micrometre-scale domain
# would come out uniform. Below span 1, the `1e-10` floor takes over instead, which is
# already far under any spacing spread worth calling non-uniform (see the b = 1e-6 row in
# the measurement below).
#
# `c = 4` was picked from measuring this repo's own drift on a uniform Float32 mesh, host
# backend, `n * abs(h_ref)` in parentheses: 2.24e-8 (n = 10, span 1.0), 6.71e-8 (n = 16, span
# 1.0), 6.15e-8 (n = 64, span 1.0). `4 * eps(Float32) * 1.0 ≈ 4.77e-7` clears the worst of
# those (6.71e-8) with over 7x to spare. A genuinely non-uniform Float32 mesh at span 1 sits
# many orders of magnitude past this floor; at span 1e-6 (n = 16) the eps term itself falls
# to ~4.8e-13, so the `1e-10` floor takes over, and a non-uniform mesh's spread there
# (~3.2e-7) clears it by three orders of magnitude.
#
# Far from the origin the drift tracks the coordinates' ulp, not the span: a uniform mesh
# on [a, a + 1] rounds each point to eps(a), so the eps term scales with the larger of the
# span estimate and `mag`, the end points' largest magnitude (gpena/Bramble.jl#492).
# The measured drift sits well under the resulting tolerance (drift vs tolerance): Float32
# [10, 11] 9.5e-7 vs 5.2e-6, Float32 [1000, 1001] 6.1e-5 vs 4.8e-4, Float64 [1e6, 1e6 + 1]
# 1.16e-10 vs 8.9e-10, Float64 [3e7, 3e7 + 1] 3.7e-9 vs 2.7e-8. No constant floor clears
# the last drift and still calls a non-uniform mesh on [0, 1e-8] (spread 2.8e-9)
# non-uniform. `mag` comes from the points, not the domain's set, which `change_points!`
# leaves stale. That term is capped at `abs(h_ref)`. The scan rejects `|h - h_ref| >= tol`,
# so every accepted spacing lies strictly between 0 and `2h_ref` and keeps `h_ref`'s sign.
# Uncapped, Float32 near 1e6 gives ~0.48, and a non-increasing set would pass as uniform.
# A cap of `abs(h_ref) / 2` is too tight. At `h_ref` = 2 ulp it leaves a 1-ulp tolerance,
# which the 1-ulp drift of a uniform Float32 mesh does not clear.
#
# Returns `T`, not `Float64`: `1e-10` is a `Float64` literal, and without the `T(...)`
# conversion every call would promote the per-point comparison in `is_uniform` to `Float64`
# regardless of `T`. `T(1e-10)` underflows to `0` for `Float16` (below its smallest
# subnormal), which is harmless here since the `eps(T)` term already dominates at any `T`
# fine enough to underflow the floor.
@inline function _default_is_uniform_tol(::Type{T}, h_ref, n, mag) where {T}
    span = n * abs(h_ref)
    return max(T(1e-10), min(4 * eps(T) * max(span, mag), abs(h_ref)))
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

Return the constant stepsize for a uniform mesh.
  - In 1D, returns scalar ``h = x_2 - x_1``.
  - In nD, returns a tuple ``(h_1, \\dots, h_D)`` of stepsizes along each coordinate axis.
  - When `d` is specified, returns the stepsize along dimension `d`.

Throws an `ArgumentError` if the mesh is not uniform.

See also: [`is_uniform`](@ref), [`spacing`](@ref).
"""
@inline function stepsize(Ωₕ::AbstractMeshType{1})
    is_uniform(Ωₕ) || _throw_not_uniform()
    h = host_spacings(Ωₕ)
    length(h) <= 1 && return zero(eltype(Ωₕ))
    return h[2]
end

@inline function stepsize(Ωₕ::AbstractMeshType{D}) where {D}
    return ntuple(i -> stepsize(Ωₕ(i)), Val(D))
end

@inline stepsize(Ωₕ::AbstractMeshType, d::Integer) = stepsize(Ωₕ(d))

"""
    locate_cell(Ωₕ::AbstractMeshType{1}, x::Real) -> Int
    locate_cell(Ωₕ::AbstractMeshType{D}, x) -> CartesianIndex{D}

Locate the cell containing continuous coordinate `x`.
  - For 1D meshes, returns integer index `i \\in 1:N-1` such that ``x_i \\le x \\le x_{i+1}``
    (clamped to the domain boundaries).
  - For nD meshes, returns a `CartesianIndex{D}` locating the bounding cell along each dimension.

**`x` outside `[x_1, x_N]` is silently clamped to the boundary cell** -- `locate_cell` never
throws and never signals that `x` was out of range, so a caller that assumes the returned
cell means `x` was inside the mesh inherits that assumption unknowingly. This is exactly
what [`interpolate_at`](@ref)'s `outside` keyword (gpena/Bramble.jl#223) exists to make an
explicit, checked choice about one layer up, rather than leaving to a bare cell lookup with
no way to say "no" -- a new caller of `locate_cell` directly should decide its own
out-of-range policy the same way, not assume this one already did.

In a hot loop, pass `x` as a `Tuple` or an `SVector`: both are stack-allocated and go
through the zero-cost `ntuple`-based methods below, while a `Vector` allocates wherever
it is built.

# Examples

```julia
using Bramble: locate_cell
Ωₕ = mesh(domain(interval(0.0, 1.0)), 11)  # h = 0.1
locate_cell(Ωₕ, 0.35)  # returns 4 (interval [0.3, 0.4])
locate_cell(Ωₕ, 5.0)   # returns 10 (the last cell) -- no error, x = 5.0 is well outside [0, 1]
```
"""
function locate_cell end

@inline function locate_cell(Ωₕ::AbstractMeshType{1}, x::Tuple{Real})
    x1 = x[1]
    return locate_cell(Ωₕ, x1)
end

@inline function locate_cell(Ωₕ::AbstractMeshType{1}, x::AbstractVector)
    @boundscheck length(x) == 1 ||
                 throw(DimensionMismatch("locate_cell expected a length-1 vector, got length $(length(x))"))
    x1 = @inbounds x[1]
    return locate_cell(Ωₕ, x1)
end

@inline function locate_cell(Ωₕ::AbstractMeshType{D}, x::AbstractVector) where {D}
    @boundscheck length(x) == D ||
                 throw(DimensionMismatch("locate_cell expected a length-$D vector, got length $(length(x))"))
    xt = ntuple(i -> (@inbounds x[i]), Val(D))
    return locate_cell(Ωₕ, xt)
end

"""
    normal_vector(Ωₕ::AbstractMeshType{D}, symbol::Symbol) -> NTuple{D, eltype(Ωₕ)}
    normal_vector(::Val{D}, symbol::Symbol) -> NTuple{D, Float64}

Return the outward unit normal vector associated with a standard boundary facet label
(`:xmin`, `:xmax`, `:ymin`, `:ymax`, `:zmin`, `:zmax`) or legacy viewpoint alias
(`:left`, `:right`, `:bottom`, `:top`, `:front`, `:back`). The mesh form returns the mesh's
element type, `eltype(Ωₕ)`, and the `Val` form, which has no mesh to take a type from, returns
`Float64`.

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
@inline function normal_vector(Ωₕ::AbstractMeshType{D}, symbol::Symbol) where {D}
    return map(eltype(Ωₕ), normal_vector(Val(D), symbol))
end

# One method per dimension, its body one `return` of a literal tuple per row of the table,
# written when the module loads. A loop over the rows (what `_face_of_symbol` does) gives the
# same values, but it boxes a computed tuple, and so allocates, when the call is dispatched at
# run time (a `Val(D)` built from a variable); a literal is a preallocated constant. Nor does a
# `@generated` method do: its specialisations box their result as well.
for D in 1:3
    checks = map(_boundary_faces(Val(D))) do (canonical, alias, axis, side)
        direction = ntuple(d -> d == axis ? (side == 1 ? -1.0 : 1.0) : 0.0, D)
        test = :(symbol === $(QuoteNode(canonical)) || symbol === $(QuoteNode(alias)))
        :($test && return $direction)
    end
    @eval @inline function normal_vector(::Val{$D}, symbol::Symbol)
        $(checks...)
        return _throw_unknown_boundary_symbol(Val($D), symbol)
    end
end

# One line per face, from the table: `:xmin/:left, :xmax/:right, ...`.
@noinline function _throw_unknown_boundary_symbol(::Val{D}, symbol::Symbol) where {D}
    expected = join((":$c/:$a" for (c, a, _, _) in _boundary_faces(Val(D))), ", ")
    throw(ArgumentError("Unknown $(D)D boundary symbol: :$symbol. Expected $expected."))
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
`inner_Γ`) or the field of the symbolic `InnerGamma` (operators/inner.jl).
=#

# The `(axis, side)` of the face `s` names, canonical symbol or alias, or `nothing`. Read from
# the table of `_BOUNDARY_FACES` (geometry/set.jl), where the 3D aliases are not the 2D ones.
@inline function _find_face(::Val{D}, s::Symbol) where {D}
    for (canonical, alias, axis, side) in _boundary_faces(Val(D))
        (s === canonical || s === alias) && return (axis, side)
    end
    return nothing
end

@inline function _face_of_symbol(::Val{D}, s::Symbol) where {D}
    face = _find_face(Val(D), s)
    face === nothing && _throw_not_a_face(s, D)
    return face
end

@noinline function _throw_not_a_face(s::Symbol, D::Int)
    throw(
        ArgumentError(
        "inner_Γ integrates over whole coordinate faces, and :$s does not name one in $(D)D. " *
        "Expected `:boundary` or a canonical face symbol (:xmin, :xmax" *
        (D > 1 ? ", :ymin, :ymax" : "") * (D > 2 ? ", :zmin, :zmax" : "") *
        ") or one of its aliases. A label is read as a face name only when the mesh holds " *
        "no marker by that name; a marker is read from its marked points instead.",
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
    _marked_face_mask(Ωₕ::AbstractMeshType{D}, marked_at) -> NTuple{D, NTuple{2, Bool}}

The coordinate faces a marked point set covers, as a per-axis `(min, max)` mask, read off
the marked points rather than a label's name. `marked_at(k)` says whether the point with
linear index `k` is marked.

A face is selected when every point of its relative interior is marked: the points off the
edges and corners it shares with its neighbours. A marker that leaves the corners out
(`x -> x[1] == 1 && 0 < x[2] < 1`) therefore still selects its face. A face with no relative
interior (two points or fewer on a transverse axis) needs all its points marked; in 1D a
face is a single point and its own interior.
"""
@inline function _marked_face_mask(Ωₕ::AbstractMeshType{D}, marked_at) where {D}
    idxs = indices(Ωₕ)
    faces = boundary_indices(idxs)
    return ntuple(Val(D)) do d
        (_face_is_marked(marked_at, idxs, faces[2d - 1], d),
            _face_is_marked(marked_at, idxs, faces[2d], d))
    end
end

@inline function _face_is_marked(
        marked_at, idxs::CartesianIndices{D}, face::CartesianIndices{D}, d::Int
) where {D}
    lin = LinearIndices(idxs)
    open = CartesianIndices(
        ntuple(e -> e == d ? face.indices[e] : _open_range(face.indices[e]), Val(D))
    )
    for I in ifelse(isempty(open), face, open)
        marked_at(lin[I]) || return false
    end
    return true
end

@inline _open_range(r::AbstractUnitRange) = (first(r) + 1):(last(r) - 1)

"""
    _surface_mask(Ωₕ::AbstractMeshType{D}, labels::NTuple{N, Symbol}, caller = "inner_Γ") -> NTuple{D, NTuple{2, Bool}}

The coordinate faces `labels` cover on the mesh `Ωₕ`, as a per-axis `(min, max)` mask. A
label is read from its definition: one that `markers(Ωₕ)` holds is read off its marked
points, the union over every such label going through [`_marked_face_mask`](@ref); any other
label is a face name, resolved by `_face_of_symbol`. The alias fallback of
[`index_in_marker`](@ref) is not used: it answers a face name with the aliased marker's own
points whenever they all lie on that face, and they can be only part of it, while a face
name here means the whole coordinate face.

Throws an `ArgumentError`, prefixed with `caller`, when the markers select no face, or when
one of their marked points lies on no selected face. "On a face" is index arithmetic, never
a weight, so the check reads no spacing and does not scalar-index a device mesh.
"""
@inline function _surface_mask(Ωₕ::AbstractMeshType{D}, labels::NTuple{N, Symbol},
        caller::String = "inner_Γ") where {D, N}
    m = markers(Ωₕ)
    mask = ntuple(_ -> (false, false), Val(D))
    has_marker = false
    for s in labels
        if haskey(m, s)
            has_marker = true
        else
            mask = _or_mask(mask, _face_mask(Val(D), (s,)))
        end
    end
    has_marker || return mask
    marked_at = k -> _any_marked(m, labels, k)
    from_markers = _marked_face_mask(Ωₕ, marked_at)
    _no_faces(from_markers) && _throw_markers_not_faces(m, labels, caller)
    mask = _or_mask(mask, from_markers)
    _check_marked_on_faces(Ωₕ, m, labels, mask, caller)
    return mask
end

@inline _or_mask(a::NTuple{D, NTuple{2, Bool}}, b::NTuple{D, NTuple{2, Bool}}) where {D} = ntuple(
    d -> (a[d][1] || b[d][1], a[d][2] || b[d][2]), Val(D))

# Whether point `k` is in any of the markers `labels` names; a label `m` does not hold marks
# nothing.
@inline function _any_marked(m, labels, k)
    for s in labels
        b = get(m, s, nothing)
        b !== nothing && @inbounds(b[k]) && return true
    end
    return false
end

# Walks the set bits of each marker alone (`findnext` skips empty chunks), so a marker on a
# large mesh costs its marked points, not a dictionary lookup per grid point.
@inline function _check_marked_on_faces(
        Ωₕ::AbstractMeshType{D}, m, labels, mask, caller) where {D}
    cis = indices(Ωₕ)
    np = npoints(Ωₕ, Tuple)
    for s in labels
        b = get(m, s, nothing)
        b === nothing && continue
        k = findfirst(b)
        while k !== nothing
            _on_mask_face(mask, cis[k], np) || _throw_markers_not_faces(m, labels, caller)
            k = findnext(b, k + 1)
        end
    end
    return nothing
end

@inline _on_mask_face(mask::NTuple{D}, I::CartesianIndex{D}, np) where {D} = any(
    d -> (mask[d][1] && I[d] == 1) || (mask[d][2] && I[d] == np[d]), ntuple(identity, Val(D)))

@noinline function _throw_markers_not_faces(m, labels, caller)
    held = join((":$s" for s in labels if haskey(m, s)), ", ")
    throw(
        ArgumentError(
        "$caller: the markers $held do not consist of whole boundary faces. A marker " *
        "selects a coordinate face when every point of the face off its edges and corners " *
        "is marked, and every marked point has to lie on a selected face. Part of a face, " *
        "an interior interface or a staircase is a more general surface: its weight does " *
        "not factorise where the surface is cut at an interior index, so it needs the " *
        "one-sided face sum, which is not implemented yet.",
    ),
    )
end

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
