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
    [`Rₕ`](@ref), which involves no quadrature. An `Int` or a `Val`; see
    [`avgₕ!`](@ref) for what the difference costs.
  - `markers`: restrict evaluation to the named marked regions, as in
    [`Rₕ`](@ref), leaving every other entry zero.
"""
Base.@constprop :aggressive function avgₕ(
        Wₕ::AbstractSpaceType, f; quad_points::Union{Integer, Val} = Val(AVG_QUAD_POINTS),
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

`quad_points` accepts a plain `Int` or a `Val`. A `Val` (the keyword's own default,
`Val(AVG_QUAD_POINTS)`) is a compile-time constant the quadrature rule specialises
on directly, so it costs nothing beyond the parallel loop's own task overhead — the
same is true of the positional form `avgₕ!(uₕ, f, Val(n))`, which skips the keyword
sorter entirely. A runtime `Int` is genuinely dynamic and cannot specialise the same
way, so it costs a small, constant allocation (tens of bytes, not the per-point cost
this docstring's first paragraph is about) to box the resulting `Val` — acceptable
for an occasional call, worth avoiding with `Val` inside a hot loop.

See also: [`avgₕ`](@ref), [`Rₕ!`](@ref)
"""
@inline avgₕ!(uₕ::VectorElement{<:ScalarGridSpace{D}}, f::Tuple{Any}) where {D} = avgₕ!(uₕ, f[1])

@inline avgₕ!(uₕ::VectorElement{<:ScalarGridSpace{D}}, f::F) where {D, F} = _avgₕ!(
    uₕ, f, Val(D), Val(AVG_QUAD_POINTS))

@inline avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::Tuple) where {NC} = _avgₕ!(
    uₕ, f, Val(dim(mesh(space(uₕ)))), Val(AVG_QUAD_POINTS))

@inline avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::F) where {NC, F} = _avgₕ!(
    uₕ, f, Val(dim(mesh(space(uₕ)))), Val(AVG_QUAD_POINTS))

@inline avgₕ!(uₕ::VectorElement{<:ScalarGridSpace{D}}, f::F, nq::Val{NQ}) where {
    D, F, NQ} = _avgₕ!(
    uₕ, f, Val(D), nq)

@inline avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::Tuple, nq::Val{NQ}) where {
    NC, NQ} = _avgₕ!(
    uₕ, f, Val(dim(mesh(space(uₕ)))), nq)

@inline avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::F, nq::Val{NQ}) where {
    NC, F, NQ} = _avgₕ!(
    uₕ, f, Val(dim(mesh(space(uₕ)))), nq)

# A one-component space is a scalar space, so an NC-tuple of functions with
# NC == 1 must still work.
@inline avgₕ!(
    uₕ::VectorElement{<:ScalarGridSpace}, f::Tuple{Any}; quad_points::Union{Integer, Val} = Val(AVG_QUAD_POINTS),
    markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N} = avgₕ!(
    uₕ, f[1]; quad_points = quad_points, markers = markers)

# `NQ` is a compile-time constant here (a type parameter), so the check folds away same as
# `_is_source_only`'s own branches do — it costs nothing on the path this exists to keep
# allocation-free. Without it, `quad_points = Val(0)` reached QuadGK's own "Gauss rules
# require positive order" instead of this function's own message, for the keyword path only
# (the direct positional `avgₕ!(uₕ, f, Val(n))` escape hatch stays unchecked, matching every
# other internal call that already hard-codes a known-good `Val` without validating it).
@inline function _to_quad_val(nq::Val{NQ}) where {NQ}
    NQ >= 1 || throw(ArgumentError("quad_points must be >= 1, got $NQ"))
    return nq
end
@inline function _to_quad_val(nq::Integer)
    nq >= 1 || throw(ArgumentError("quad_points must be >= 1, got $nq"))
    return Val(Int(nq))
end

Base.@constprop :aggressive function avgₕ!(
        uₕ::VectorElement, f::F; quad_points::Union{Integer, Val} = Val(AVG_QUAD_POINTS),
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {F, N}
    nq = _to_quad_val(quad_points)
    Ωₕ = mesh(space(uₕ))
    D = dim(Ωₕ)

    if N > 0
        return _avg_masked!(uₕ, f, markers, Val(D), nq)
    end

    return _avgₕ!(uₕ, f, Val(D), nq)
end

# A named, concretely-typed kernel for the quadrature loop's inner call, rather than an
# anonymous closure over the same five captures (`f`, `x`, `idxs`, `nodes`, `wts`).
#
# The two should be equivalent -- a Julia closure is already a concrete, compiler-generated
# struct with the same fields -- and mostly are: `Rₕ!`'s equivalent closure, half the
# captures, never showed *this* problem (the allocation flake below). It did turn out to cost
# real time for an unrelated reason, closed the same way -- see `_RₕKernel`, point 67. This one,
# five captures deep inside `_cpu_threaded_for!`,
# is a *closer call* for the compiler's inlining cost model, and closer calls are where an
# "inlining budget nothing can query" (this file's own words, on the related `_gauss_rule`
# fold) occasionally goes the wrong way -- point 51 in docs/form-unlock-plan.md measured it:
# 1 to 3 in 20 independent compiles landed on a path that cost 176 B *per grid point*, 80 MiB
# on a million-point mesh, on otherwise identical code. A hand-written struct doesn't remove
# that risk outright, but it is one fewer compiler decision between the data and the call --
# no closure-capture-list inference, just field access on a type this file wrote down. Pay a
# few named types now, deterministically, rather than gamble on an unobservable heuristic.
struct _AvgKernel1{F, X, IX, NQ, T}
    f::F
    x::X
    idxs::IX
    nodes::NTuple{NQ, T}
    wts::NTuple{NQ, T}
end
@inline (k::_AvgKernel1)(i) = _cell_average(k.f, k.x, k.idxs[i][1], k.nodes, k.wts)

struct _AvgKernelD{F, X, IX, NQ, T}
    f::F
    x::X
    idxs::IX
    nodes::NTuple{NQ, T}
    wts::NTuple{NQ, T}
end
@inline (k::_AvgKernelD)(i) = _cell_average(k.f, k.x, k.idxs[i], k.nodes, k.wts)

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

    _cpu_threaded_for!(execution_policy(space), raw, 1:n, _AvgKernel1(
        f, x, idxs, nodes, wts))
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

    _cpu_threaded_for!(execution_policy(space), raw, 1:n, _AvgKernelD(
        f, x, idxs, nodes, wts))
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

# Same reasoning as `_AvgKernel1`/`_AvgKernelD` above, for the tuple-valued (composite)
# quadrature call.
struct _AvgScatterKernel1{F, X, IX, NQ, T, NC}
    f::F
    x::X
    idxs::IX
    nodes::NTuple{NQ, T}
    wts::NTuple{NQ, T}
end
@inline (k::_AvgScatterKernel1{
    F, X, IX, NQ, T, NC})(i) where {F, X, IX, NQ, T, NC} = _cell_average(
    k.f, k.x, k.idxs[i][1], k.nodes, k.wts, Val(NC))

struct _AvgScatterKernelD{F, X, IX, NQ, T, NC}
    f::F
    x::X
    idxs::IX
    nodes::NTuple{NQ, T}
    wts::NTuple{NQ, T}
end
@inline (k::_AvgScatterKernelD{
    F, X, IX, NQ, T, NC})(i) where {F, X, IX, NQ, T, NC} = _cell_average(
    k.f, k.x, k.idxs[i], k.nodes, k.wts, Val(NC))

# Composite space: single vector-valued function returning all components
@inline function _avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f, ::Val{1}, nq::Val{NQ}) where {
        NC, NQ}
    sp = space(uₕ)
    Ωₕ = mesh(sp)
    x = half_points(Ωₕ)
    T = eltype(Ωₕ)
    comps = components(uₕ)
    raws = ntuple(i -> values(comps[i]), Val(NC))
    idxs = indices(Ωₕ)
    n = length(idxs)
    nodes, wts = _gauss_rule(nq, T)

    _cpu_threaded_scatter_for!(
        execution_policy(sp), raws, 1:n,
        _AvgScatterKernel1{typeof(f), typeof(x), typeof(idxs), NQ, T, NC}(
            f, x, idxs, nodes, wts))
    return uₕ
end

@inline function _avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f, ::Val{D}, nq::Val{NQ}) where {
        NC, D, NQ}
    sp = space(uₕ)
    Ωₕ = mesh(sp)
    x = half_points(Ωₕ)
    T = eltype(Ωₕ)
    comps = components(uₕ)
    raws = ntuple(i -> values(comps[i]), Val(NC))
    idxs = indices(Ωₕ)
    n = length(idxs)
    nodes, wts = _gauss_rule(nq, T)

    _cpu_threaded_scatter_for!(
        execution_policy(sp), raws, 1:n,
        _AvgScatterKernelD{typeof(f), typeof(x), typeof(idxs), NQ, T, NC}(
            f, x, idxs, nodes, wts))
    return uₕ
end

@inline function _avg_masked!(uₕ::VectorElement{<:ScalarGridSpace}, f::F,
        markers::NTuple{N, Symbol}, ::Val{1}, nq::Val{NQ}) where {F, N, NQ}
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

@inline function _avg_masked!(uₕ::VectorElement{<:ScalarGridSpace}, f::F,
        markers::NTuple{N, Symbol}, ::Val{D}, nq::Val{NQ}) where {F, N, D, NQ}
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

@inline function _avg_masked!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::F,
        markers::NTuple{N, Symbol}, ::Val{1}, nq::Val{NQ}) where {NC, F, N, NQ}
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
                _write_components!(raws, vals, i)
            end
        end
    end
    return uₕ
end

@inline function _avg_masked!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::F,
        markers::NTuple{N, Symbol}, ::Val{D}, nq::Val{NQ}) where {NC, F, N, D, NQ}
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
                _write_components!(raws, vals, i)
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
`NTuple{N,T}`, so the per-cell loop that consumes them does not allocate.

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
            return :(($nodes, $wts))
        catch
            # fall through to the run-time rule
        end
    end
    return :(_gauss_rule_runtime(Val($N), $T))
end

@inline function _gauss_rule_runtime(::Val{N}, ::Type{T}) where {N, T}
    x, w = gauss(T, N, zero(T), one(T))
    return NTuple{N, T}(x), NTuple{N, T}(w)
end

# Average of `f` over the 1D cell spanned by `x[i] .. x[i+1]`.
@inline function _cell_average(f, x::AbstractVector, i::Int, nodes::NTuple{NQ, T},
        wts::NTuple{NQ, T}) where {NQ, T}
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
# `NTuple{6,T}` twice — 96 bytes for `Float64`, stored inline because it is isbits — and
# `Threads.@threads` gives each task its own copy of the closure, so it is paid once per
# thread per call. Measured as the per-thread gap between `avgₕ!` at 472 B and `Rₕ!` at
# 376 B.
#
# It used to be fetched *inside* the kernel for `Float16`/`Float32`/`Float64`, to avoid
# exactly that: `_gauss_rule` is `@generated` and folds to a tuple literal, so the
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
@inline function _cell_average(f, x::AbstractVector, i::Int, nodes::NTuple{NQ, T},
        wts::NTuple{NQ, T}, ::Val{NC}) where {NQ, T, NC}
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
        nodes::NTuple{NQ, T}, wts::NTuple{NQ, T}) where {NQ, T}
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
        f, x::NTuple{2}, idx::CartesianIndex{2}, nodes::NTuple{NQ, T},
        wts::NTuple{NQ, T}, ::Val{NC}) where {NQ, T, NC}
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
        nodes::NTuple{NQ, T}, wts::NTuple{NQ, T}) where {NQ, T}
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
        f, x::NTuple{3}, idx::CartesianIndex{3}, nodes::NTuple{NQ, T},
        wts::NTuple{NQ, T}, ::Val{NC}) where {NQ, T, NC}
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
        f, x::NTuple{D}, idx::CartesianIndex{D}, nodes::NTuple{NQ, T},
        wts::NTuple{NQ, T}, ::Val{NC}) where {D, NQ, T, NC}
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
        nodes::NTuple{NQ, T}, wts::NTuple{NQ, T}) where {D, NQ, T}
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
