# ghost_vector_operators.jl: the runtime difference operators (`Δₕ!` and the others) on a
# `GhostVector`, on every CPU policy. It follows `vector_calculus.jl` because it needs every
# runtime engine (gpena/Bramble.jl#776).
#
# Each operator runs today's engine on the physical vector, which gives every point its
# value except along a ghosted facet, where the engine truncates. It then recomputes that
# boundary slab alone, reading the ghost buffer and the padded mesh's spacings at the
# facet: the values the same operator gives on `lift(Cₕ, u)`, the padded grid function, at
# the physical nodes. A facet without ghosts keeps the engine's truncation. The result is
# the physical values only, a plain vector (no ghosts), so composing two operators
# truncates as on a plain space.
#
# A one-axis operator runs the engine's band function and its facet slabs in one band pass
# of `_run_bands!`, banded along the last axis as the engines are under `CpuThreaded` and
# `CpuPolyester`; the Laplacian runs the plain serial `Δₕ!` engine, then its facet slabs in
# one band pass. Each pass takes arrays and isbits values only, so a `Polyester.@batch`
# task boxes nothing.
#
# Rejected: copying into the padded `VectorElement` and calling today's operators (the copy
# alone costs two stencils), and testing for a ghost at every read (twice as slow).
#
# See also: `GhostVector`, `update_ghosts!`, `lift`

# The slab formulas of the averages and of the Laplacian; a difference's is its direction.
struct _GhostAverage end
struct _GhostLaplacian end

# The fix-up of one facet normal to axis `d`: its physical index `i` along `d` (1 or the
# last), the ghost number before its first ghost, whether it is the minimum facet, whether
# the operator reads its ghosts at all, and the padded spacings the slab formula divides by.
struct _FacetFix{T}
    d::Int
    i::Int
    off::Int
    min::Bool
    on::Bool
    h::NTuple{3, T}
end

# Which facets a direction's stencil reads beyond: a backward one the minimum, a forward
# one the maximum, a centered one both.
@inline _reads_ghost(::Backward, min::Bool) = min
@inline _reads_ghost(::Forward, min::Bool) = !min
@inline _reads_ghost(::CenteredStencil, ::Bool) = true

# The padded spacings at the padded index `p` of the facet node, read from the family's own
# spacing function on the padded axis `Gd`: what the operator on the lift divides by there.
@inline function _ghost_h(::GridDirection, sf::S, Gd, p::Int, ::Bool, ::Type{T}) where {S, T}
    return (convert(T, sf(Gd)[p]), zero(T), zero(T))
end
@inline function _ghost_h(::CrossWeighted, sf::S, Gd, p::Int, ::Bool, ::Type{T}) where {S, T}
    h = sf(Gd)
    return (convert(T, h[p]), convert(T, h[p + 1]), zero(T))
end
@inline _ghost_h(::_GhostAverage, sf, Gd, p::Int, ::Bool, ::Type{T}) where {T} = (
    zero(T), zero(T), zero(T))
@inline function _ghost_h(::_GhostLaplacian, sf, Gd, p::Int, min::Bool, ::Type{T}) where {T}
    hb, hs = backward_spacings_for_derivative(Gd), star_spacings(Gd)
    min && return (convert(T, hb[p]), convert(T, hs[p]), zero(T))
    return (convert(T, hb[p + 1]), convert(T, hb[p]), convert(T, hs[p]))
end

@inline function _facet_fix(G::GhostMesh, Gd, d::Int, min::Bool, reads::Bool, kind, sf,
        ::Type{T}) where {T}
    s = min ? 1 : 2
    on = reads && G.active[d][s]
    i = min ? 1 : G.n[d]
    h = on ? _ghost_h(kind, sf, Gd, i + G.lo[d], min, T) : (zero(T), zero(T), zero(T))
    return _FacetFix{T}(d, i, G.offset[d][s], min, on, h)
end

# The value at a facet node `cur` with ghost `g` beyond it, `nb` its inward neighbour and
# `o` what the engine wrote there: the operator's interior formula on the padded grid,
# term for term as `_compute_difference`, `_compute_average` and `_accumulate_laplacian!`
# write it.
@inline _ghost_slab(::Backward, min, o, cur, g, nb, h) = (cur - g) / h[1]
@inline _ghost_slab(::Forward, min, o, cur, g, nb, h) = (g - cur) / h[1]
@inline _ghost_slab(::Centered, min::Bool, o, cur, g, nb, h) = (min ? nb - g : g - nb) /
                                                               (2 * h[1])
@inline function _ghost_slab(::CrossWeighted, min::Bool, o, cur, g, nb, h)
    back, fwd = min ? (g, nb) : (nb, g)
    hᵢ, hᵢ₊₁ = h[1], h[2]
    return (hᵢ * (fwd - cur) / hᵢ₊₁ + hᵢ₊₁ * (cur - back) / hᵢ) / (hᵢ + hᵢ₊₁)
end
@inline _ghost_slab(::_GhostAverage, min, o, cur, g, nb, h) = (cur + g) / 2
@inline function _ghost_slab(::_GhostLaplacian, min::Bool, o, cur, g, nb, h)
    min && return o - (cur - g) / h[1] / h[2]
    return o + ((g - cur) / h[1] - (cur - nb) / h[2]) / h[3]
end

# The slabs of `fixes` in band `b` of `nbands` along the last axis: each facet node is
# written by the one band holding it, and a node on two facets (a corner) by that band in
# the order of `fixes`, so every policy gives the serial values.
@noinline function _ghost_facet_band!(out, x, gb, kind::K, n::NTuple{D, Int},
        fixes::NTuple{F, _FacetFix{T}}, nbands::Int, b::Int) where {K, D, F, T}
    li = LinearIndices(n)
    band = _band_range(1:n[D], nbands, b)
    for f in fixes
        f.on || continue
        d = f.d
        slab = _band_slab(ntuple(e -> e == d ? (f.i:f.i) : (1:n[e]), Val(D)), band)
        lt = LinearIndices(ntuple(e -> e == d ? 1 : n[e], Val(D)))
        inward = (f.min ? 1 : -1) * prod(ntuple(e -> e < d ? n[e] : 1, Val(D)))
        @inbounds for I in CartesianIndices(slab)
            idx = li[I]
            g = gb[f.off + lt[CartesianIndex(ntuple(e -> e == d ? 1 : I[e], Val(D)))]]
            out[idx] = _ghost_slab(kind, f.min, out[idx], x[idx], g, x[idx + inward], f.h)
        end
    end
    return nothing
end

# --- Destination and checks ------------------------------------------------------------ #

# The vector the operator writes: `out` itself, the physical part of a `GhostVector` on
# the same mesh, or the data of a grid function on the physical mesh.
@inline _ghost_dest(out::AbstractVector, ::GhostVector) = out
@inline function _ghost_dest(out::GhostVector, v::GhostVector)
    out.mesh === v.mesh || _throw_ghost_dest_mesh("a GhostVector on another mesh")
    return out.physical
end
@inline function _ghost_dest(out::VectorElement{<:ScalarGridSpace}, v::GhostVector)
    mesh(space(out)) === physical_mesh(v.mesh) ||
        _throw_ghost_dest_mesh("a grid function on another mesh")
    return parent(out)
end

@inline function _check_ghost_operand(dest::AbstractVector, v::GhostVector)
    G = v.mesh
    np = prod(G.n)
    (length(v.physical) == np && length(v.ghosts) == G.nghosts) ||
        _throw_ghost_operand_sizes(v, np, G.nghosts)
    length(dest) == np || _throw_ghost_dest_length(length(dest), np)
    (Base.mightalias(dest, v.physical) || Base.mightalias(dest, v.ghosts)) &&
        _throw_alias_error()
    return nothing
end

@noinline function _throw_ghost_dest_mesh(what::String)
    throw(
        ArgumentError(
        "an operator on a GhostVector v writes the values at the physical nodes of the " *
        "mesh of v, and the destination is $what. Pass a Vector, a GhostVector on the " *
        "mesh of v, or a grid function on its physical mesh.",
    ),
    )
end

@noinline function _throw_ghost_operand_sizes(v::GhostVector, np::Int, ng::Int)
    throw(
        ArgumentError(
        "the mesh of the GhostVector has $np physical nodes and $ng ghosts, and v holds " *
        "$(length(v.physical)) and $(length(v.ghosts)), so one of its parts was resized. " *
        "Build it again, v = GhostVector(Cₕ, u).",
    ),
    )
end

@noinline function _throw_ghost_dest_length(l::Int, np::Int)
    throw(
        ArgumentError(
        "an operator on a GhostVector writes its $np physical values, and the " *
        "destination has $l entries. Pass a destination of length $np, or call the " *
        "allocating form.",
    ),
    )
end

# --- The one-axis families ------------------------------------------------------------- #

# One band of a one-axis operator: the engine's own band (`_difference_band!`,
# `_average_band!`) on the physical values, then the facet slabs of that band, which the
# engine has just written truncated. Both cut the grid along its last axis into the same
# bands, so a band rewrites only points it wrote itself. The spacing crosses the band pass as
# an array: `StarSpacings`, a struct around one, is rebuilt inside (`_in_band`), since a
# `Polyester.@batch` task boxes a struct holding an array (the plain `Dcₓ!` and `D̃ₓ!` pay
# 80 B for it under `CpuPolyester`).
@inline function _ghost_stencil_band!(out, x, gb, h, wrap::W, n::NTuple{D, Int}, dir,
        kind::K, dv::Val, fixes, nbands::Int, b::Int) where {W, D, K}
    _ghost_interior_band!(kind, out, x, _in_band(wrap, h), n, dir, dv, nbands, b)
    _ghost_facet_band!(out, x, gb, kind, n, fixes, nbands, b)
    return nothing
end

@inline _ghost_interior_band!(::GridDirection, out, x, h::H, n, dir, dv, nbands, b) where {H} = _difference_band!(
    out, x, h, n, dir, dv, nbands, b)
@inline _ghost_interior_band!(::_GhostAverage, out, x, h, n, dir, dv, nbands, b) = _average_band!(
    out, x, n, dir, dv, nbands, b)

# The spacing the engine divides by, as the array that crosses the band pass, and the tag
# `_in_band` rebuilds it from: a tag rather than the spacing function itself, which a band
# pass forwarding it in a `Vararg` would not specialise on (it then allocates).
struct _StarTag end
@inline _band_spacing(sf::S, sub) where {S} = sf(sub)
@inline _band_spacing(::typeof(star_spacings), sub) = spacings(sub)
@inline _band_spacing(::Nothing, sub) = nothing
@inline _band_tag(sf) = nothing
@inline _band_tag(::typeof(star_spacings)) = _StarTag()
@inline _in_band(::Nothing, h) = h
@inline _in_band(::_StarTag, h) = StarSpacings(h)

function _ghost_stencil!(out::AbstractVector, v::GhostVector{<:Any, <:Any, <:Any, <:GhostMesh{D}},
        sf::S, precheck::P, dir::GridDirection, kind::K,
        ::Val{DIM}) where {D, S, P, K, DIM}
    _check_stencil_dim(DIM, D)
    G = v.mesh
    Gd = _padded(G)(DIM)
    dest = _ghost_dest(out, v)
    _check_ghost_operand(dest, v)
    precheck(Gd, DIM)
    Ω = G.physical
    T = eltype(G)
    fixes = (_facet_fix(G, Gd, DIM, true, _reads_ghost(dir, true), kind, sf, T),
        _facet_fix(G, Gd, DIM, false, _reads_ghost(dir, false), kind, sf, T))
    _run_bands!(execution_policy(Ω), _ghost_stencil_band!, dest, v.physical, v.ghosts,
        _band_spacing(sf, Ω(DIM)), _band_tag(sf), G.n, dir, kind, Val(DIM), fixes)
    return out
end

# A `GhostVector` reaches the vectorial aliases (`∇ₕ(v)`, one `D₋` per axis) through its mesh.
@inline _op_mesh(v::GhostVector) = v.mesh

const _GHOST_ONE_AXIS_DOC = """
    backward_finite_difference!(out::AbstractVector, v::GhostVector, dim_val::Val) -> out
    backward_finite_difference(v::GhostVector, dim_val::Val) -> Vector

The one-axis operators on a [`GhostVector`](@ref): `D₋ₓ`, `D₊ₓ`, `Mₓ` (the backward
average), `M₊ₓ`, `Dcₓ`, `D̃ₓ`, `D̽ₓ` and their `ᵧ`, `₂` and `!` forms, and `∇ₕ(v)`, the
tuple of `D₋` along every axis. Each gives the values at the physical nodes, in the mesh's
linear order, as a plain vector with no ghosts: those of the same operator on
`lift(Cₕ, u)` at the physical nodes, when `v` holds the degrees of freedom `u` and
[`update_ghosts!`](@ref) has filled its ghosts. A stencil reaching beyond a facet with
ghosts reads the ghost there, at `x₁ - h₂` or `xₙ + hₙ`; beyond a facet without ghosts it
is truncated, as on a plain space.

`out` is a vector of the physical length, a `GhostVector` on the mesh of `v` (its physical
part is written) or a grid function on the physical mesh; any other mesh, or an `out`
sharing memory with `v`, refuses with an `ArgumentError`. The in-place forms allocate
nothing on `CpuSerial` and `CpuPolyester`. A result composed with another operator is
truncated at the facets, as on a plain space.

See also: [`Δₕ`](@ref), [`update_ghosts!`](@ref)
"""

@doc _GHOST_ONE_AXIS_DOC
@inline backward_finite_difference!(out::AbstractVector, v::GhostVector, dv::Val) = _ghost_stencil!(
    out, v, backward_spacings_for_derivative, _no_precheck, Backward(), Backward(), dv)
@inline forward_finite_difference!(out::AbstractVector, v::GhostVector, dv::Val) = _ghost_stencil!(
    out, v, forward_spacings_for_derivative, _no_precheck, Forward(), Forward(), dv)
@inline backward_average!(out::AbstractVector, v::GhostVector, dv::Val) = _ghost_stencil!(
    out, v, nothing, _no_precheck, Backward(), _GhostAverage(), dv)
@inline forward_average!(out::AbstractVector, v::GhostVector, dv::Val) = _ghost_stencil!(
    out, v, nothing, _no_precheck, Forward(), _GhostAverage(), dv)
@inline centered_difference!(out::AbstractVector, v::GhostVector, dv::Val) = _ghost_stencil!(
    out, v, star_spacings, _check_centered_points, Centered(), Centered(), dv)
@inline forward_star_difference!(out::AbstractVector, v::GhostVector, dv::Val) = _ghost_stencil!(
    out, v, star_spacings, _no_precheck, Forward(), Forward(), dv)
@inline cross_weighted_difference!(out::AbstractVector, v::GhostVector, dv::Val) = _ghost_stencil!(
    out, v, spacings, _check_centered_points, CrossWeighted(), CrossWeighted(), dv)

@doc _GHOST_ONE_AXIS_DOC
@inline backward_finite_difference(v::GhostVector, dv::Val) = backward_finite_difference!(
    similar(v.physical), v, dv)
@inline forward_finite_difference(v::GhostVector, dv::Val) = forward_finite_difference!(
    similar(v.physical), v, dv)
@inline backward_average(v::GhostVector, dv::Val) = backward_average!(similar(v.physical), v, dv)
@inline forward_average(v::GhostVector, dv::Val) = forward_average!(similar(v.physical), v, dv)
@inline centered_difference(v::GhostVector, dv::Val) = centered_difference!(
    similar(v.physical), v, dv)
@inline forward_star_difference(v::GhostVector, dv::Val) = forward_star_difference!(
    similar(v.physical), v, dv)
@inline cross_weighted_difference(v::GhostVector, dv::Val) = cross_weighted_difference!(
    similar(v.physical), v, dv)

# --- Laplacian ------------------------------------------------------------------------- #

const _GHOST_LAPLACIAN_DOC = """
    Δₕ(v::GhostVector) -> Vector
    Δₕ!(out::AbstractVector, v::GhostVector) -> out

The conservative discrete Laplacian of a [`GhostVector`](@ref) at the physical nodes, as a
plain vector with no ghosts: the values of `Δₕ(lift(Cₕ, u))` there, when `v` holds the
degrees of freedom `u` and [`update_ghosts!`](@ref) has filled its ghosts. Along an axis
whose facet carries ghosts, the flux across the facet reads the ghost; along one without,
the stencil is truncated as on a plain space.

`out` follows the rule of the one-axis operators (`D₋ₓ!` on a `GhostVector`). `Δₕ!`
allocates nothing on `CpuSerial` and `CpuPolyester`.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, false))
Cₕ = constrained(gridspace(Ωₕ; ghosts = :boundary),
                 u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => 0.0,
                       :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0))
v = update_ghosts!(GhostVector(Cₕ, rand(35)), Cₕ)
Δₕ(v)                           # 35 values, the Neumann and Robin data in the boundary rows
```
"""

@doc _GHOST_LAPLACIAN_DOC
@inline Δₕ(v::GhostVector) = Δₕ!(similar(v.physical), v)

@doc _GHOST_LAPLACIAN_DOC
function Δₕ!(out::AbstractVector, v::GhostVector{<:Any, <:Any, <:Any, <:GhostMesh{D}}) where {D}
    G = v.mesh
    P = _padded(G)
    dest = _ghost_dest(out, v)
    _check_ghost_operand(dest, v)
    x, Ω = v.physical, G.physical
    fill!(dest, zero(eltype(dest)))
    _laplacian_direction!(dest, x, Ω, G.n, Val(D), Val(D))
    T = eltype(G)
    fixes = ntuple(
        k -> _facet_fix(G, P((k + 1) ÷ 2), (k + 1) ÷ 2, isodd(k), true, _GhostLaplacian(),
            nothing, T),
        Val(2D))
    any(f -> f.on, fixes) && _run_bands!(execution_policy(Ω), _ghost_facet_band!, dest, x,
        v.ghosts, _GhostLaplacian(), G.n, fixes)
    return out
end
