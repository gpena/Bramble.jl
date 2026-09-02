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
	avgₕ(Wₕ::AbstractSpaceType, f; quad_points = AVG_QUAD_POINTS, markers = ())

Returns a [`VectorElement`](@ref) with the average of function `f` with respect to the [`cell_measure`](@ref) of `mesh(Wₕ)` around each grid point.

Each cell average is a tensor-product Gauss-Legendre rule with `quad_points`
points per direction, exact for polynomials of degree `2 * quad_points - 1`.

# The shape of `f`

As for [`Rₕ`](@ref), `f` receives the coordinates of a point: a scalar on a 1D
mesh and an `NTuple{D}` on a `D`-dimensional one, never an `SVector`. For an
`N`-component space either a tuple of functions or a single function returning
all components works, and both give the same result.

# Keywords

  - `quad_points`: points per direction, per cell. Has no counterpart in
    [`Rₕ`](@ref), which involves no quadrature.
  - `markers`: restrict evaluation to the named marked regions, as in
    [`Rₕ`](@ref), leaving every other entry zero.
"""
Base.@constprop :aggressive function avgₕ(
        Wₕ::AbstractSpaceType, f; quad_points::Int = AVG_QUAD_POINTS,
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N}
    uₕ = element(Wₕ, _restriction_eltype(Wₕ, f, markers))
    return avgₕ!(uₕ, f; quad_points = quad_points, markers = markers)
end

"""
	avgₕ!(uₕ::VectorElement, f; quad_points = AVG_QUAD_POINTS, markers = ())

In-place version of the averaging operator [`avgₕ`](@ref). Returns `uₕ`.

Allocates only the task overhead of the parallel loop, independently of the
number of grid points, and `f` is called directly rather than wrapped so that it
specialises and inlines into the quadrature loop. This is the form to use inside
a time-stepping loop.

`f` and the keywords are as described for [`avgₕ`](@ref).

See also: [`avgₕ`](@ref), [`Rₕ!`](@ref)
"""
@inline avgₕ!(uₕ::VectorElement{<:ScalarGridSpace{D}}, f::Tuple{Any}) where {D} = avgₕ!(uₕ, f[1])

@inline avgₕ!(uₕ::VectorElement{<:ScalarGridSpace{D}}, f::F) where {D, F} = _avgₕ!(
    uₕ, f, Val(D), Val(AVG_QUAD_POINTS))

@inline avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::Tuple) where {NC} = _avgₕ!(
    uₕ, f, Val(dim(mesh(space(uₕ)))), Val(AVG_QUAD_POINTS))

@inline avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::F) where {NC, F} = _avgₕ!(
    uₕ, f, Val(dim(mesh(space(uₕ)))), Val(AVG_QUAD_POINTS))

# A one-component space is a scalar space, so an NC-tuple of functions with
# NC == 1 must still work.
@inline avgₕ!(
    uₕ::VectorElement{<:ScalarGridSpace}, f::Tuple{Any}; quad_points::Int = AVG_QUAD_POINTS,
    markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N} = avgₕ!(
    uₕ, f[1]; quad_points = quad_points, markers = markers)

Base.@constprop :aggressive function avgₕ!(
        uₕ::VectorElement, f; quad_points::Int = AVG_QUAD_POINTS,
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N}
    quad_points >= 1 || throw(ArgumentError("quad_points must be >= 1, got $quad_points"))
    Ωₕ = mesh(space(uₕ))
    D = dim(Ωₕ)

    if N > 0
        return _avg_masked!(uₕ, f, markers, Val(D), Val(quad_points))
    end

    return _avgₕ!(uₕ, f, Val(D), Val(quad_points))
end

# Evaluations per grid point for a tensor-product rule of `NQ` points per
# direction on a `D`-dimensional mesh. The threading threshold counts
# evaluations, so a kernel this expensive per index reaches the crossover
# at proportionally fewer indices.
@inline _avg_min_work(::Val{D}, ::Val{NQ}) where {D, NQ} = max(
    1, cld(CPU_THREADED_MIN, NQ^D))

@inline function _avgₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f::F, ::Val{1}, nq::Val{NQ}) where {
        F, NQ}
    (; space) = uₕ
    Ωₕ = mesh(space)
    x = half_points(Ωₕ)
    T = eltype(Ωₕ)
    raw = values(uₕ)
    idxs = indices(Ωₕ)
    n = length(idxs)
    nodes, wts = _gauss_rule(nq, T)

    _cpu_threaded_for!(raw, 1:n, i -> _cell_average(f, x, idxs[i][1], nodes, wts);
        min_work = _avg_min_work(Val(1), nq))
    return uₕ
end

@inline function _avgₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f::F, ::Val{D}, nq::Val{NQ}) where {
        F, D, NQ}
    (; space) = uₕ
    Ωₕ = mesh(space)
    x = half_points(Ωₕ)
    T = eltype(Ωₕ)
    raw = values(uₕ)
    idxs = indices(Ωₕ)
    n = length(idxs)
    nodes, wts = _gauss_rule(nq, T)

    _cpu_threaded_for!(raw, 1:n, i -> _cell_average(f, x, idxs[i], nodes, wts);
        min_work = _avg_min_work(Val(D), nq))
    return uₕ
end

# Composite space: one function per component
@inline function _avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}},
        f::Tuple, ::Val{D}, nq::Val{NQ}) where {NC, D, NQ}
    comps = components(uₕ)
    ntuple(i -> _avgₕ!(comps[i], f[i], Val(D), nq), Val(NC))
    return uₕ
end

@inline function _avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}},
        f::Tuple, ::Val{1}, nq::Val{NQ}) where {NC, NQ}
    comps = components(uₕ)
    ntuple(i -> _avgₕ!(comps[i], f[i], Val(1), nq), Val(NC))
    return uₕ
end

# Composite space: single vector-valued function returning all components
@inline function _avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f, ::Val{1}, nq::Val{NQ}) where {
        NC, NQ}
    Ωₕ = mesh(space(uₕ))
    x = half_points(Ωₕ)
    T = eltype(Ωₕ)
    comps = components(uₕ)
    raws = ntuple(i -> values(comps[i]), Val(NC))
    idxs = indices(Ωₕ)
    n = length(idxs)
    nodes, wts = _gauss_rule(nq, T)

    _cpu_threaded_scatter_for!(
        raws, 1:n, i -> _cell_average(f, x, idxs[i][1], nodes, wts, Val(NC));
        min_work = _avg_min_work(Val(1), nq))
    return uₕ
end

@inline function _avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f, ::Val{D}, nq::Val{NQ}) where {
        NC, D, NQ}
    Ωₕ = mesh(space(uₕ))
    x = half_points(Ωₕ)
    T = eltype(Ωₕ)
    comps = components(uₕ)
    raws = ntuple(i -> values(comps[i]), Val(NC))
    idxs = indices(Ωₕ)
    n = length(idxs)
    nodes, wts = _gauss_rule(nq, T)

    _cpu_threaded_scatter_for!(
        raws, 1:n, i -> _cell_average(f, x, idxs[i], nodes, wts, Val(NC));
        min_work = _avg_min_work(Val(D), nq))
    return uₕ
end

@inline function _avg_masked!(uₕ::VectorElement{<:ScalarGridSpace}, f,
        markers::NTuple{N, Symbol}, ::Val{1}, nq::Val{NQ}) where {N, NQ}
    (; space) = uₕ
    Ωₕ = mesh(space)
    x = half_points(Ωₕ)
    T = eltype(Ωₕ)
    raw = values(uₕ)
    idxs = indices(Ωₕ)
    n = length(idxs)
    nodes, wts = _gauss_rule(nq, T)

    fill!(raw, zero(eltype(raw)))
    for m in markers
        mask = index_in_marker(Ωₕ, m)
        @inbounds for i in 1:n
            if mask[i]
                raw[i] = _cell_average(f, x, idxs[i][1], nodes, wts)
            end
        end
    end
    return uₕ
end

@inline function _avg_masked!(uₕ::VectorElement{<:ScalarGridSpace}, f,
        markers::NTuple{N, Symbol}, ::Val{D}, nq::Val{NQ}) where {N, D, NQ}
    (; space) = uₕ
    Ωₕ = mesh(space)
    x = half_points(Ωₕ)
    T = eltype(Ωₕ)
    raw = values(uₕ)
    idxs = indices(Ωₕ)
    n = length(idxs)
    nodes, wts = _gauss_rule(nq, T)

    fill!(raw, zero(eltype(raw)))
    for m in markers
        mask = index_in_marker(Ωₕ, m)
        @inbounds for i in 1:n
            if mask[i]
                raw[i] = _cell_average(f, x, idxs[i], nodes, wts)
            end
        end
    end
    return uₕ
end

@inline function _avg_masked!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::Tuple,
        markers::NTuple{N, Symbol}, ::Val{D}, nq::Val{NQ}) where {NC, N, D, NQ}
    comps = components(uₕ)
    ntuple(i -> _avg_masked!(comps[i], f[i], markers, Val(D), nq), Val(NC))
    return uₕ
end

@inline function _avg_masked!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::Tuple,
        markers::NTuple{N, Symbol}, ::Val{1}, nq::Val{NQ}) where {NC, N, NQ}
    comps = components(uₕ)
    ntuple(i -> _avg_masked!(comps[i], f[i], markers, Val(1), nq), Val(NC))
    return uₕ
end

@inline function _avg_masked!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f,
        markers::NTuple{N, Symbol}, ::Val{1}, nq::Val{NQ}) where {NC, N, NQ}
    Ωₕ = mesh(space(uₕ))
    x = half_points(Ωₕ)
    T = eltype(Ωₕ)
    comps = components(uₕ)
    raws = ntuple(i -> values(comps[i]), Val(NC))
    idxs = indices(Ωₕ)
    n = length(idxs)
    nodes, wts = _gauss_rule(nq, T)

    for raw in raws
        fill!(raw, zero(eltype(raw)))
    end
    for m in markers
        mask = index_in_marker(Ωₕ, m)
        @inbounds for i in 1:n
            if mask[i]
                vals = _cell_average(f, x, idxs[i][1], nodes, wts, Val(NC))
                _scatter_comp!(raws, vals, i)
            end
        end
    end
    return uₕ
end

@inline function _avg_masked!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f,
        markers::NTuple{N, Symbol}, ::Val{D}, nq::Val{NQ}) where {NC, N, D, NQ}
    Ωₕ = mesh(space(uₕ))
    x = half_points(Ωₕ)
    T = eltype(Ωₕ)
    comps = components(uₕ)
    raws = ntuple(i -> values(comps[i]), Val(NC))
    idxs = indices(Ωₕ)
    n = length(idxs)
    nodes, wts = _gauss_rule(nq, T)

    for raw in raws
        fill!(raw, zero(eltype(raw)))
    end
    for m in markers
        mask = index_in_marker(Ωₕ, m)
        @inbounds for i in 1:n
            if mask[i]
                vals = _cell_average(f, x, idxs[i], nodes, wts, Val(NC))
                _scatter_comp!(raws, vals, i)
            end
        end
    end
    return uₕ
end

#=
Cell averages are computed with a fixed tensor-product Gauss-Legendre rule per
cell. Every cell integral is independent, low dimensional and over a smooth
integrand, so a small fixed rule is both cheaper and allocation free.

Writing the cell integral on the reference cube,

	1/|C| ∫_C f = ∫_{[0,1]^D} f(a + t ⊙ (b - a)) dt,

because |C| = ∏ₖ (bₖ - aₖ) exactly: the cell around a grid point spans
consecutive half points, whose spacing is the half spacing that
`cell_measure` returns. The quadrature weights below sum to one, so the
weighted sum *is* the average and no measure division is needed.
=#

"""
	AVG_QUAD_POINTS

Default number of Gauss-Legendre points per direction, per cell, used by
[`avgₕ`](@ref). Six points are exact for polynomials up to degree eleven.

Unlike an adaptive rule, a fixed one does not tighten itself on coarse cells, so
the default is chosen to be accurate on cells far coarser than any practical
grid. Measured on 4 points spanning [-1, 4] with a function varying by a factor
of e^5 across the domain -- deliberately harsher than a real mesh -- the worst
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
`SVector{N,T}`, so the per-cell loop that consumes them does not allocate.

The rule is built by `QuadGK.gauss` in the requested element type, so `Float32`
and `BigFloat` grids get a rule at their own precision rather than a rounded
`Float64` one. Weights sum to one, which makes the weighted sum over a cell the
cell *average* directly.
"""
@generated function _gauss_rule(::Val{N}, ::Type{T}) where {N, T}
    # When `T` is an isbits float its precision is fixed by the type, so the rule
    # depends only on (N, T) and is folded into a compile-time constant: obtaining it
    # then costs nothing at all. This covers Float16/32/64 and equally the stack
    # allocated extended precision types such as Double64 or Float64x2.
    #
    # Anything else -- notably BigFloat, whose precision is a run-time setting that
    # must not be baked in at compile time -- falls back to building the rule per
    # call, at the precision in force at that moment. The same fallback catches an
    # isbits type that QuadGK cannot construct a rule for.
    if isbitstype(T)
        try
            x, w = gauss(T, N, zero(T), one(T))
            nodes = Expr(:tuple, x...)
            wts = Expr(:tuple, w...)
            return :((SVector{$N, $T}($nodes), SVector{$N, $T}($wts)))
        catch
            # fall through to the run-time rule
        end
    end
    return :(_gauss_rule_runtime(Val($N), $T))
end

@inline function _gauss_rule_runtime(::Val{N}, ::Type{T}) where {N, T}
    x, w = gauss(T, N, zero(T), one(T))
    return SVector{N, T}(x), SVector{N, T}(w)
end

# Average of `f` over the 1D cell spanned by `x[i] .. x[i+1]`.
@inline function _cell_average(f, x::AbstractVector, i::Int, nodes::SVector{NQ, T},
        wts::SVector{NQ, T}) where {NQ, T}
    @inbounds a = T(x[i])
    @inbounds d = T(x[i + 1]) - a

    s = zero(T)
    @inbounds for q in 1:NQ
        s += wts[q] * f(a + nodes[q] * d)
    end
    return s
end

# Kernel form of the cell average: closes over the geometry so it can be handed to a
# generic index loop.
#
# The quadrature rule is built once, outside the kernel, and captured by it. That costs
# `SVector{6,T}` twice — 96 bytes for `Float64`, stored inline because it is isbits — and
# `Threads.@threads` gives each task its own copy of the closure, so it is paid once per
# thread per call. Measured as the per-thread gap between `avgₕ!` at 472 B and `Rₕ!` at
# 376 B.
#
# It used to be fetched *inside* the kernel for `Float16`/`Float32`/`Float64`, to avoid
# exactly that: `_gauss_rule` is `@generated` and folds to an `SVector` literal, so the
# closure shrank to 16 bytes from 104 and the loop ran within noise of the old one.
#
# That fold is not guaranteed, and on x86_64 Linux it does not happen. CI measured `avgₕ!`
# at exactly 80 bytes per grid point — 83,887,904 B on a 1024x1024 mesh, against a 100,000
# bound — while the same commit measured 0 B per point on aarch64 macOS. Both the serial
# and the threaded path showed it, so it is codegen, not threading.
#
# What made it expensive to diagnose: a probe that built the kernel and called it directly
# reported the fold surviving *on the failing machine*. Calling the closure on its own
# folds; inlining it into `_cpu_threaded_for!`'s loop does not. So the probe has to measure the
# real path, and the test now does.
#
# The trade was therefore 96 bytes per thread against 80 MiB per million points on one of
# the two architectures we test, decided by an inlining budget nothing can query. Capturing
# is what the code did for every type before the split, and it is what the extended types
# always needed: `BigFloat`'s precision is a run-time setting so `_gauss_rule` cannot fold
# at all, and `Double64` is isbits and takes the folding branch without folding, costing
# 3184 bytes per call — 2977 B per grid point on the fetch-inside path.
@inline function _cell_average_kernel(f, x, nq::Val, ::Type{T}, ::Val{1}) where {T}
    nodes, wts = _gauss_rule(nq, T)
    return idx -> _cell_average(f, x, idx[1], nodes, wts)
end
@inline function _cell_average_kernel(f, x, nq::Val, ::Type{T}, ::Val{D}) where {D, T}
    nodes, wts = _gauss_rule(nq, T)
    return idx -> _cell_average(f, x, idx, nodes, wts)
end

# The composite form, one value per component.
@inline function _cell_average_kernel(
        f, x, nq::Val, ::Type{T}, ::Val{1}, ::Val{NC}) where {T, NC}
    nodes, wts = _gauss_rule(nq, T)
    return idx -> _cell_average(f, x, idx[1], nodes, wts, Val(NC))
end
@inline function _cell_average_kernel(
        f, x, nq::Val, ::Type{T}, ::Val{D}, ::Val{NC}) where {D, T, NC}
    nodes, wts = _gauss_rule(nq, T)
    return idx -> _cell_average(f, x, idx, nodes, wts, Val(NC))
end

# The one-dimensional composite case. A 1D mesh answers `half_points` with a plain vector
# rather than a one-tuple of vectors, so the D-dimensional method below does not match it
# and this one is needed: without it, `avgₕ!` on a composite space over a 1D mesh, given a
# single function returning all components, raised a MethodError. The per-component tuple
# form with a tuple of functions was unaffected, since it dispatches to the scalar path
# once per component.
@inline function _cell_average(f, x::AbstractVector, i::Int, nodes::SVector{NQ, T},
        wts::SVector{NQ, T}, ::Val{NC}) where {NQ, T, NC}
    @inbounds a = T(x[i])
    @inbounds b = T(x[i + 1])

    s = ntuple(_ -> zero(T), Val(NC))
    @inbounds for q in 1:NQ
        s = s .+ wts[q] .* f(a + nodes[q] * (b - a))
    end
    return s
end

# 2D specialized scalar cell average
@inline function _cell_average(f, x::NTuple{2}, idx::CartesianIndex{2},
        nodes::SVector{NQ, T}, wts::SVector{NQ, T}) where {NQ, T}
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
        f, x::NTuple{2}, idx::CartesianIndex{2}, nodes::SVector{NQ, T},
        wts::SVector{NQ, T}, ::Val{NC}) where {NQ, T, NC}
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
@inline function _cell_average(f, x::NTuple{3}, idx::CartesianIndex{3},
        nodes::SVector{NQ, T}, wts::SVector{NQ, T}) where {NQ, T}
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
        f, x::NTuple{3}, idx::CartesianIndex{3}, nodes::SVector{NQ, T},
        wts::SVector{NQ, T}, ::Val{NC}) where {NQ, T, NC}
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

# Average of a vector valued `f` over the cell around `idx`, one value per
# component. `f` is evaluated once per quadrature node instead of once per node
# per component; the accumulator is a tuple and every operation on it broadcasts
# over `NC` isbits values, so nothing allocates.
@inline function _cell_average(
        f, x::NTuple{D}, idx::CartesianIndex{D}, nodes::SVector{NQ, T},
        wts::SVector{NQ, T}, ::Val{NC}) where {D, NQ, T, NC}
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
@inline function _cell_average(f, x::NTuple{D}, idx::CartesianIndex{D},
        nodes::SVector{NQ, T}, wts::SVector{NQ, T}) where {D, NQ, T}
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
