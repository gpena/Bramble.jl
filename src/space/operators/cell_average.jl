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
Base.@constprop :aggressive function avgₕ!(
        uₕ::VectorElement, f; quad_points::Int = AVG_QUAD_POINTS,
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N}
    quad_points >= 1 || throw(ArgumentError("quad_points must be >= 1, got $quad_points"))
    Ωₕ = mesh(space(uₕ))

    if N > 0
        masks = ntuple(i -> index_in_marker(Ωₕ, markers[i]), Val(N))
        # The rule is built in the mesh's element type, not the grid function's: the
        # nodes and weights are geometry, and the Gauss rule is tabulated for real
        # types only. A Dual-valued grid function integrates against a Float64 rule
        # and the product promotes, which is what makes avgₕ differentiable.
        _avg_masked!(uₕ, f, masks, half_points(Ωₕ), eltype(Ωₕ), Val(quad_points),
            Val(dim(Ωₕ)))
        return uₕ
    end

    # `f` is passed through unwrapped on purpose. Embedding it in a
    # BrambleFunction gives a fixed compiled signature at the cost of an
    # indirect call, which stops `f` inlining into the quadrature loop and
    # measured 2.3x slower on the grids in the test suite. It also matches
    # `avgₕ`, which has always passed the raw function.
    _avgₕ!(uₕ, f, Val(dim(Ωₕ)), Val(quad_points))
    return uₕ
end

# The marked branch, split the same way the unmarked `_avgₕ!` family is. It used to be a
# single `_masked_for!(to_matrix(uₕ), …)` call, which is wrong for a composite grid
# function twice over: `to_matrix` answers with an NTuple of matrices there, not one array,
# and the scalar kernel was built where the composite one is needed. `avgₕ!` on a composite
# space with markers raised a MethodError as a result.
@inline function _avg_masked!(uₕ::VectorElement{<:ScalarGridSpace}, f, masks, x,
        ::Type{T}, nq::Val, dv::Val) where {T}
    _masked_for!(to_matrix(uₕ), masks, _cell_average_kernel(
        f, x, nq, T, dv))
    return nothing
end

# One function per component: each is a scalar restriction over the same mask.
@inline function _avg_masked!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::Tuple, masks,
        x, ::Type{T}, nq::Val, dv::Val) where {NC, T}
    comps = components(uₕ)
    ntuple(i -> (_avg_masked!(comps[i], f[i], masks, x, T, nq, dv); nothing), Val(NC))
    return nothing
end

# A single function returning every component: one masked pass, scattering the tuple.
@inline function _avg_masked!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f, masks, x,
        ::Type{T}, nq::Val, dv::Val) where {NC, T}
    mats = ntuple(i -> to_matrix(components(uₕ)[i]), Val(NC))
    _masked_scatter_for!(mats, masks,
        _cell_average_kernel(f, x, nq, T, dv, Val(NC)))
    return nothing
end

# A one-component space is a scalar space, so an NC-tuple of functions with
# NC == 1 must still work.
@inline avgₕ!(
    uₕ::VectorElement{<:ScalarGridSpace}, f::Tuple{Any}; quad_points::Int = AVG_QUAD_POINTS,
    markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N} = avgₕ!(
    uₕ, f[1]; quad_points = quad_points, markers = markers)

# Evaluations per grid point for a tensor-product rule of `NQ` points per
# direction on a `D`-dimensional mesh. The threading threshold in
# `_parallel_for!` counts evaluations, so a kernel this expensive per index
# reaches the crossover at proportionally fewer indices.
@inline _avg_min_work(::Val{D}, ::Val{NQ}) where {D, NQ} = max(
    1, cld(PARALLEL_FOR_MIN, NQ^D))

function _avgₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f, ::Val{1}, nq::Val)
    Ωₕ = mesh(space(uₕ))
    x = half_points(Ωₕ)
    T = eltype(Ωₕ)

    _parallel_for!(values(uₕ), indices(Ωₕ),
        _cell_average_kernel(f, x, nq, T, Val(1));
        min_work = _avg_min_work(Val(1), nq))
    return nothing
end

function _avgₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f, ::Val{D}, nq::Val) where {D}
    Ωₕ = mesh(space(uₕ))
    x = half_points(Ωₕ)
    T = eltype(Ωₕ)

    _parallel_for!(to_matrix(uₕ), indices(Ωₕ),
        _cell_average_kernel(f, x, nq, T, Val(D));
        min_work = _avg_min_work(Val(D), nq))
    return nothing
end

function _avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::Tuple, val_dim::Val, nq::Val) where {NC}
    comps = components(uₕ)
    ntuple(i -> _avgₕ!(comps[i], f[i], val_dim, nq), Val(NC))
    return nothing
end

# A single function returning all components: average it in one pass over the
# grid rather than once per component.
function _avgₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f, ::Val{D}, nq::Val) where {
        NC, D}
    Ωₕ = mesh(space(uₕ))
    x = half_points(Ωₕ)
    T = eltype(Ωₕ)
    mats = ntuple(i -> to_matrix(components(uₕ)[i]), Val(NC))

    _scatter_for!(mats, indices(Ωₕ),
        _cell_average_kernel(f, x, nq, T, Val(D), Val(NC));
        min_work = _avg_min_work(Val(D), nq))
    return nothing
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
# folds; inlining it into `_parallel_for!`'s loop does not. So the probe has to measure the
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
