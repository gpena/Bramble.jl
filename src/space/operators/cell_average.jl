#===========================================================================#
# The cell average operator avgₕ, and its quadrature.
#
# Averages a function over the cell around each grid point with a tensor-product
# Gauss-Legendre rule.
#
# Distinct from operators/average.jl, which is the two-point average of a grid point with
# its neighbour. That one is a stencil; this one integrates over a cell.
#===========================================================================#

######################
#                    #
# Averaging operator #
#                    #
######################

# The coefficient type of the result comes from `f`, as it does in `Rₕ` and for the same
# reason: the average of a Dual-valued function is Dual-valued, over a mesh that stays
# Float64. The quadrature weights are the mesh's type and promote against it.
#
# Placed above the docstring, not between it and the definition. A comment there detaches
# the docstring from `avgₕ` and binds it to nothing, which the exported-names-are-
# documented check in test/quality catches.
"""
    avgₕ(Wₕ::AbstractSpaceType, f; quad_points = AVG_QUAD_POINTS, markers = ()) -> VectorElement

Returns a [`VectorElement`](@ref) with the average of function `f` with respect to the [`cell_measure`](@ref) of `mesh(Wₕ)` around each grid point.

Each cell average is a tensor-product Gauss-Legendre rule with `quad_points`
points per direction, exact for polynomials of degree `2 * quad_points - 1`.

# Arguments

  - `Wₕ::AbstractSpaceType`: grid space on which to average `f`.
  - `f`: function of one grid point. Receives coordinates as a scalar on 1D
    meshes or an `NTuple{D}` on `D`-dimensional meshes, never an `SVector`.

# Keywords

  - `quad_points::Union{Integer, Val}`: points per direction, per cell.
    Defaults to `Val(AVG_QUAD_POINTS)`.
  - `markers::NTuple{N, Symbol}`: restrict evaluation to the named marked
    regions, leaving every other entry zero.

# Examples

```julia
avgₕ(Wₕ, x -> sin(x))
avgₕ(Wₕ, x -> sin(x[1]) * x[2]; quad_points = Val(4))
```

See also: [`avgₕ!`](@ref), [`Rₕ`](@ref).
"""
Base.@constprop :aggressive function avgₕ(
    Wₕ::AbstractSpaceType,
    f;
    quad_points::Union{Integer,Val}=Val(AVG_QUAD_POINTS),
    markers::NTuple{N,Symbol}=NTuple{0,Symbol}(),
) where {N}
    uₕ = element(Wₕ, _restriction_eltype(Wₕ, f, markers))
    return avgₕ!(uₕ, f; quad_points=quad_points, markers=markers)
end

"""
    avgₕ!(uₕ::VectorElement, f; quad_points = AVG_QUAD_POINTS, markers = ()) -> VectorElement

In-place version of the averaging operator [`avgₕ`](@ref). Returns `uₕ`.

Evaluates the tensor-product Gauss-Legendre cell average of `f` and writes
the result into `uₕ`.

# Arguments

  - `uₕ::VectorElement`: pre-allocated element to write into.
  - `f`: function of one grid point. Receives coordinates as a scalar on 1D
    meshes or an `NTuple{D}` on `D`-dimensional meshes.

# Keywords

  - `quad_points::Union{Integer, Val}`: points per direction, per cell.
    Defaults to `Val(AVG_QUAD_POINTS)`. Using a `Val` allows compile-time
    specialization of the quadrature nodes and weights without boxing.
  - `markers::NTuple{N, Symbol}`: restrict evaluation to the named marked
    regions, leaving every other entry zero.

See also: [`avgₕ`](@ref), [`Rₕ!`](@ref).
"""
@inline avgₕ!(uₕ::VectorElement{<:ScalarGridSpace{D}}, f::Tuple{Any}) where {D} =
    avgₕ!(uₕ, f[1])

@inline avgₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f::F) where {F} =
    project!(uₕ, _average_rule(f, Val(AVG_QUAD_POINTS)))

@inline avgₕ!(uₕ::VectorElement{<:CompositeGridSpace}, f::Tuple) =
    project!(uₕ, _average_rule(f, Val(AVG_QUAD_POINTS)))

@inline avgₕ!(uₕ::VectorElement{<:CompositeGridSpace}, f::F) where {F} =
    project!(uₕ, _average_rule(f, Val(AVG_QUAD_POINTS)))

@inline avgₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f::F, nq::Val{NQ}) where {F,NQ} =
    project!(uₕ, _average_rule(f, nq))

@inline avgₕ!(uₕ::VectorElement{<:CompositeGridSpace}, f::Tuple, nq::Val{NQ}) where {NQ} =
    project!(uₕ, _average_rule(f, nq))

@inline avgₕ!(uₕ::VectorElement{<:CompositeGridSpace}, f::F, nq::Val{NQ}) where {F,NQ} =
    project!(uₕ, _average_rule(f, nq))

# A one-component space is a scalar space, so an NC-tuple of functions with
# NC == 1 must still work.
#
# This method and the positional `Tuple{Any}` one at the top of the file are a pair, and the
# `ScalarGridSpace{D}` on that one is load-bearing rather than leftover: written as a bare
# `ScalarGridSpace` its positional signature becomes identical to this method's, and a
# keyword method with the same positional signature *overwrites* the positional one instead
# of coexisting with it -- precompilation then fails with "Method overwriting is not
# permitted". Its `D` is otherwise unused.
@inline avgₕ!(
    uₕ::VectorElement{<:ScalarGridSpace},
    f::Tuple{Any};
    quad_points::Union{Integer,Val}=Val(AVG_QUAD_POINTS),
    markers::NTuple{N,Symbol}=NTuple{0,Symbol}(),
) where {N} = avgₕ!(uₕ, f[1]; quad_points=quad_points, markers=markers)

# `NQ` is a compile-time constant here (a type parameter), so the check folds away,
# costing nothing on the hot path. Without it, `quad_points = Val(0)` reached QuadGK's
# internal error message rather than validating the argument early.
@inline function _to_quad_val(nq::Val{NQ}) where {NQ}
    NQ >= 1 || throw(ArgumentError("quad_points must be >= 1, got $NQ"))
    return nq
end
@inline function _to_quad_val(nq::Integer)
    nq >= 1 || throw(ArgumentError("quad_points must be >= 1, got $nq"))
    return Val(Int(nq))
end

Base.@constprop :aggressive function avgₕ!(
    uₕ::VectorElement,
    f::F;
    quad_points::Union{Integer,Val}=Val(AVG_QUAD_POINTS),
    markers::NTuple{N,Symbol}=NTuple{0,Symbol}(),
) where {F,N}
    nq = _to_quad_val(quad_points)

    if N > 0
        return project!(uₕ, _average_rule(f, nq), markers)
    end

    return project!(uₕ, _average_rule(f, nq))
end

# A tuple of functions is a rule per leaf; anything else is one rule.
@inline _average_rule(f::Tuple, nq::Val) = map(g -> CellAverage(g, nq), f)
@inline _average_rule(f, nq::Val) = CellAverage(f, nq)

# A concretely typed kernel (`_AvgKernel`) for the quadrature loop, avoiding anonymous
# closures over captures (`f`, `x`, `idxs`, `nodes`, `wts`). Explicit struct types ensure
# predictable inlining and eliminate allocation flakes inside parallel loop dispatch.
struct _AvgKernel{F,X,IX,NQ,T}
    f::F
    x::X
    idxs::IX
    nodes::NTuple{NQ,T}
    wts::NTuple{NQ,T}
end
@inline (k::_AvgKernel)(i) = _cell_average(k.f, k.x, k.idxs[i], k.nodes, k.wts)

# Same reasoning, for the tuple-valued (composite) quadrature call. `NC` is the space's
# *leaf* count (`length(components(uₕ))`, which flattens any nesting), not the space's own
# structural type parameter.
struct _AvgScatterKernel{F,X,IX,NQ,T,NC}
    f::F
    x::X
    idxs::IX
    nodes::NTuple{NQ,T}
    wts::NTuple{NQ,T}
end
@inline (k::_AvgScatterKernel{F,X,IX,NQ,T,NC})(i) where {F,X,IX,NQ,T,NC} =
    _cell_average(k.f, k.x, k.idxs[i], k.nodes, k.wts, Val(NC))

# `CellAverage`'s side of the `project!` contract (`operators/projection.jl`). The rule
# carries the quadrature order; the driver decides the space's shape, the masking and the
# execution policy.
@inline function _rule_kernel(rule::CellAverage{F,NQ}, sp) where {F,NQ}
    Ωₕ = mesh(sp)
    nodes, wts = _gauss_rule(rule.nq, eltype(Ωₕ))
    return _AvgKernel(rule.f, half_points(Ωₕ), indices(Ωₕ), nodes, wts)
end

@inline function _rule_scatter_kernel(
    rule::CellAverage{F,NQ}, sp, ::Val{NC}
) where {F,NQ,NC}
    Ωₕ = mesh(sp)
    T = eltype(Ωₕ)
    nodes, wts = _gauss_rule(rule.nq, T)
    x = half_points(Ωₕ)
    idxs = indices(Ωₕ)
    return _AvgScatterKernel{typeof(rule.f),typeof(x),typeof(idxs),NQ,T,NC}(
        rule.f, x, idxs, nodes, wts
    )
end

@inline _rule_component(rule::CellAverage, k) = CellAverage(pt -> rule.f(pt)[k], rule.nq)

#=
Cell averages are computed with a fixed tensor-product Gauss-Legendre rule per
cell. Every cell integral is independent, low dimensional and over a smooth
integrand, so a small fixed rule is both cheaper and allocation free.

Writing the cell integral on the reference cube,

    1/|C| ∫_C f = ∫_{[0,1]^D} f(a + t ⊙ (b - a)) dt,

because |C| = ∏ₖ (bₖ - aₖ) exactly: the cell around a grid point spans
consecutive half points, whose spacing is the half spacing that
`cell_measure` returns. The quadrature weights below sum to one, so the
weighted sum is the average and no measure division is needed.
=#

"""
    AVG_QUAD_POINTS

Default number of Gauss-Legendre points per direction, per cell, used by
[`avgₕ`](@ref). Six points are exact for polynomials up to degree eleven.

Unlike an adaptive rule, a fixed one does not tighten itself on coarse cells, so
the default is chosen to be accurate on cells far coarser than any practical
grid. Measured on 4 points spanning [-1, 4] with a function varying by a factor
of e^5 across the domain (deliberately harsher than a real mesh), the worst
error over 30 random grids was

    points   1D        2D        3D        evaluations per cell (3D)
    3        6.1e-5    1.6e-4    3.1e-4     27
    4        2.8e-7    8.1e-7    3.4e-6     64
    5        1.9e-9    4.1e-9    7.8e-9    125
    6        5.5e-12   1.4e-11   1.6e-11   216

Cost is `quad_points^D` evaluations per cell. On a fine grid three points are
usually ample; lower it with the `quad_points` keyword when the integrand is
cheap to resolve and the cells are small.
"""
const AVG_QUAD_POINTS = 6

"""
    _gauss_rule(::Val{N}, ::Type{T})

Returns `(nodes, weights)` for the `N`-point Gauss-Legendre rule on `[0, 1]` as
`NTuple{N,T}`, so the per-cell loop that consumes them does not allocate.

The rule is built by `QuadGK.gauss` in the requested element type, so `Float32`
and `BigFloat` grids get a rule at their own precision rather than a rounded
`Float64` one. Weights sum to one, which makes the weighted sum over a cell the
cell average directly.
"""
@generated function _gauss_rule(::Val{N}, ::Type{T}) where {N,T}
    # When `T` is an isbits float its precision is fixed by the type, so the rule
    # depends only on (N, T) and is folded into a compile-time constant: obtaining it
    # then costs nothing at all. This covers Float16/32/64 and equally the stack
    # allocated extended precision types such as Double64 or Float64x2.
    #
    # Non-isbits types (notably BigFloat, whose precision is a runtime setting)
    # fall back to building the rule dynamically per call at current precision.
    if isbitstype(T)
        try
            x, w = gauss(T, N, zero(T), one(T))
            nodes = Expr(:tuple, x...)
            wts = Expr(:tuple, w...)
            return :(($nodes, $wts))
        catch
            # fall through to the run-time rule
        end
    end
    return :(_gauss_rule_runtime(Val($N), $T))
end

@inline function _gauss_rule_runtime(::Val{N}, ::Type{T}) where {N,T}
    x, w = gauss(T, N, zero(T), one(T))
    return NTuple{N,T}(x), NTuple{N,T}(w)
end

# A one-dimensional mesh answers `half_points` with a plain vector but indexes with
# `CartesianIndex{1}`, and unwrapping that to its `Int` is the *only* thing the whole
# `Val{1}`/`Val{D}` split in this file ever did (gpena/Bramble.jl#69). Doing it here, once,
# lets the D-dimensional kernels and sweeps cover one dimension too: dispatch already
# separates 1D (`x::AbstractVector`) from nD (`x::NTuple{D}`), so nothing else had to know.
# `CartesianIndex{1}[1]` is free -- confirmed against the allocation gates, not assumed.
@inline _cell_average(
    f, x::AbstractVector, idx::CartesianIndex{1}, nodes::NTuple{NQ,T}, wts::NTuple{NQ,T}
) where {NQ,T} = _cell_average(f, x, idx[1], nodes, wts)

@inline _cell_average(
    f,
    x::AbstractVector,
    idx::CartesianIndex{1},
    nodes::NTuple{NQ,T},
    wts::NTuple{NQ,T},
    ::Val{NC},
) where {NQ,T,NC} = _cell_average(f, x, idx[1], nodes, wts, Val(NC))

# Average of `f` over the 1D cell spanned by `x[i] .. x[i+1]`.
@inline function _cell_average(
    f, x::AbstractVector, i::Int, nodes::NTuple{NQ,T}, wts::NTuple{NQ,T}
) where {NQ,T}
    @inbounds a = T(x[i])
    @inbounds d = T(x[i + 1]) - a

    s = zero(T)
    @inbounds for q in 1:NQ
        s += wts[q] * f(a + nodes[q] * d)
    end
    return s
end

# The one-dimensional composite case. A 1D mesh answers `half_points` with a plain vector
# rather than a one-tuple of vectors, so the D-dimensional method below does not match it
# and this one is needed: without it, `avgₕ!` on a composite space over a 1D mesh, given a
# single function returning all components, raised a MethodError. The per-component tuple
# form with a tuple of functions was unaffected, since it dispatches to the scalar path
# once per component.
@inline function _cell_average(
    f, x::AbstractVector, i::Int, nodes::NTuple{NQ,T}, wts::NTuple{NQ,T}, ::Val{NC}
) where {NQ,T,NC}
    @inbounds a = T(x[i])
    @inbounds b = T(x[i + 1])

    s = ntuple(_ -> zero(T), Val(NC))
    @inbounds for q in 1:NQ
        s = s .+ wts[q] .* f(a + nodes[q] * (b - a))
    end
    return s
end

# 2D specialized scalar cell average
@inline function _cell_average(
    f, x::NTuple{2}, idx::CartesianIndex{2}, nodes::NTuple{NQ,T}, wts::NTuple{NQ,T}
) where {NQ,T}
    @inbounds i, j = idx[1], idx[2]
    @inbounds a1 = T(x[1][i])
    @inbounds d1 = T(x[1][i + 1]) - a1
    @inbounds a2 = T(x[2][j])
    @inbounds d2 = T(x[2][j + 1]) - a2

    s = zero(T)
    @inbounds for q2 in 1:NQ
        w2 = wts[q2]
        p2 = a2 + nodes[q2] * d2
        for q1 in 1:NQ
            w1 = wts[q1] * w2
            p1 = a1 + nodes[q1] * d1
            s += w1 * f((p1, p2))
        end
    end
    return s
end

# 2D specialized composite cell average
@inline function _cell_average(
    f,
    x::NTuple{2},
    idx::CartesianIndex{2},
    nodes::NTuple{NQ,T},
    wts::NTuple{NQ,T},
    ::Val{NC},
) where {NQ,T,NC}
    @inbounds i, j = idx[1], idx[2]
    @inbounds a1 = T(x[1][i])
    @inbounds d1 = T(x[1][i + 1]) - a1
    @inbounds a2 = T(x[2][j])
    @inbounds d2 = T(x[2][j + 1]) - a2

    s = ntuple(_ -> zero(T), Val(NC))
    @inbounds for q2 in 1:NQ
        w2 = wts[q2]
        p2 = a2 + nodes[q2] * d2
        for q1 in 1:NQ
            w1 = wts[q1] * w2
            p1 = a1 + nodes[q1] * d1
            s = s .+ w1 .* f((p1, p2))
        end
    end
    return s
end

# 3D specialized scalar cell average
@inline function _cell_average(
    f, x::NTuple{3}, idx::CartesianIndex{3}, nodes::NTuple{NQ,T}, wts::NTuple{NQ,T}
) where {NQ,T}
    @inbounds i, j, k = idx[1], idx[2], idx[3]
    @inbounds a1 = T(x[1][i])
    @inbounds d1 = T(x[1][i + 1]) - a1
    @inbounds a2 = T(x[2][j])
    @inbounds d2 = T(x[2][j + 1]) - a2
    @inbounds a3 = T(x[3][k])
    @inbounds d3 = T(x[3][k + 1]) - a3

    s = zero(T)
    @inbounds for q3 in 1:NQ
        w3 = wts[q3]
        p3 = a3 + nodes[q3] * d3
        for q2 in 1:NQ
            w23 = wts[q2] * w3
            p2 = a2 + nodes[q2] * d2
            for q1 in 1:NQ
                w1 = wts[q1] * w23
                p1 = a1 + nodes[q1] * d1
                s += w1 * f((p1, p2, p3))
            end
        end
    end
    return s
end

# 3D specialized composite cell average
@inline function _cell_average(
    f,
    x::NTuple{3},
    idx::CartesianIndex{3},
    nodes::NTuple{NQ,T},
    wts::NTuple{NQ,T},
    ::Val{NC},
) where {NQ,T,NC}
    @inbounds i, j, k = idx[1], idx[2], idx[3]
    @inbounds a1 = T(x[1][i])
    @inbounds d1 = T(x[1][i + 1]) - a1
    @inbounds a2 = T(x[2][j])
    @inbounds d2 = T(x[2][j + 1]) - a2
    @inbounds a3 = T(x[3][k])
    @inbounds d3 = T(x[3][k + 1]) - a3

    s = ntuple(_ -> zero(T), Val(NC))
    @inbounds for q3 in 1:NQ
        w3 = wts[q3]
        p3 = a3 + nodes[q3] * d3
        for q2 in 1:NQ
            w23 = wts[q2] * w3
            p2 = a2 + nodes[q2] * d2
            for q1 in 1:NQ
                w1 = wts[q1] * w23
                p1 = a1 + nodes[q1] * d1
                s = s .+ w1 .* f((p1, p2, p3))
            end
        end
    end
    return s
end

# Average of a vector-valued `f` over the D-dimensional cell around `idx`, one value
# per component. No mesh this package builds is more than 3D, so the 1D/2D/3D
# specialized methods above always take priority in practice; this generic one exists
# for dispatch correctness at any `D`, tested directly rather than through a mesh.
@inline function _cell_average(
    f,
    x::NTuple{D},
    idx::CartesianIndex{D},
    nodes::NTuple{NQ,T},
    wts::NTuple{NQ,T},
    ::Val{NC},
) where {D,NQ,T,NC}
    a = ntuple(k -> @inbounds(T(x[k][idx[k]])), Val(D))
    b = ntuple(k -> @inbounds(T(x[k][idx[k] + 1])), Val(D))

    s = ntuple(_ -> zero(T), Val(NC))
    @inbounds for q in CartesianIndices(ntuple(_ -> NQ, Val(D)))
        w = one(T)
        for k in 1:D
            w *= wts[q[k]]
        end
        pt = ntuple(k -> a[k] + nodes[q[k]] * (b[k] - a[k]), Val(D))
        s = s .+ w .* f(pt)
    end
    return s
end

# Average of `f` over the D-dimensional cell around `idx`, whose corners are the
# half points `x[k][idx[k]]` and `x[k][idx[k] + 1]` along each axis.
@inline function _cell_average(
    f, x::NTuple{D}, idx::CartesianIndex{D}, nodes::NTuple{NQ,T}, wts::NTuple{NQ,T}
) where {D,NQ,T}
    a = ntuple(k -> @inbounds(T(x[k][idx[k]])), Val(D))
    b = ntuple(k -> @inbounds(T(x[k][idx[k] + 1])), Val(D))

    s = zero(T)
    @inbounds for q in CartesianIndices(ntuple(_ -> NQ, Val(D)))
        w = one(T)
        for k in 1:D
            w *= wts[q[k]]
        end
        pt = ntuple(k -> a[k] + nodes[q[k]] * (b[k] - a[k]), Val(D))
        s += w * f(pt)
    end
    return s
end
