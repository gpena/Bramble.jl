# Convergence of the Ferreira and Grigorieff scheme on polygons (#204).
#
# Each level builds a fresh grid: seed lines drawn by the test, then the polygon's grid
# closure (`_grid_closure`) and a mesh moved onto it, so the levels are not nested. The
# seeds are jittered, seed i at (i + 0.7(U − ½))L/(n + 1): i.i.d. seeds put 1.8 inside the
# spread of a single run's order (1.65–2.27), and a single jittered run still dips to 1.72.
# Errors are pooled over three seeds (the geometric mean per level), and the order is the
# least-squares slope of log e against log hₘₐₓ over six levels.
#
# The H¹ₕ error sums, over the grid edges with both ends in Ω̄, the dual length times the
# squared difference over the edge length (the paper's P1 seminorm of the error); the max
# error is over the interior nodes. The load is `avgₕ`, the dual-cell mean of f.
#
# Self-contained: the file needs only `Test` in the including module, and defines
# `polygon_errors` there.

using Bramble
using Bramble: D₋ₓ, D₋ᵧ, D̃ₓ, D̃ᵧ, Dcₓ, Dcᵧ
using Random

const _PP_SLOW = isdefined(Main, :TestUtils) && Main.TestUtils.WITH_SLOW_TESTS

const _PP_FIG1 = [(0.0, 2.0), (2.0, 0.0), (4.0, 0.0), (6.0, 2.0), (6.0, 3.0), (4.0, 3.0),
    (4.0, 5.0), (2.0, 5.0), (0.0, 3.0)]
const _PP_LSHAPE = [(0.0, 0.0), (6.0, 0.0), (6.0, 3.0), (4.0, 3.0), (4.0, 5.0), (0.0, 5.0)]

# Exact solution and its derivatives.
_pp_u(p) = sin(p[1]) * cos(0.7p[2]) + 0.1 * p[1] * p[2]
_pp_ux(p) = cos(p[1]) * cos(0.7p[2]) + 0.1p[2]
_pp_uy(p) = -0.7sin(p[1]) * sin(0.7p[2]) + 0.1p[1]
_pp_uxx(p) = -sin(p[1]) * cos(0.7p[2])
_pp_uyy(p) = -0.49sin(p[1]) * cos(0.7p[2])
_pp_uxy(p) = -0.7cos(p[1]) * sin(0.7p[2]) + 0.1

# −∂ₓ(a ∂ₓu + b ∂ᵧu) − ∂ᵧ(b ∂ₓu + c ∂ᵧu) + βₓ ∂ₓu + βᵧ ∂ᵧu + r u, with the partial
# derivatives of the coefficients that the expansion needs.
const _PP_COEF = (
    poisson = (a = p -> 1.0, ax = p -> 0.0, b = nothing, c = p -> 1.0, cy = p -> 0.0,
        β = (0.0, 0.0), r = 0.0),
    mixed = (a = p -> 2 + 0.5sin(p[1] + p[2]), ax = p -> 0.5cos(p[1] + p[2]),
        b = p -> 0.5 + 0.2sin(p[1] * p[2]), bx = p -> 0.2p[2] * cos(p[1] * p[2]),
        by = p -> 0.2p[1] * cos(p[1] * p[2]), c = p -> 2 + 0.3cos(p[1]), cy = p -> 0.0,
        β = (0.0, 0.0), r = 0.0),
    variable = (a = p -> 2 + 0.5sin(p[1] + p[2]), ax = p -> 0.5cos(p[1] + p[2]), b = nothing,
        c = p -> 2 + 0.3cos(p[1]), cy = p -> 0.0, β = (0.4, -0.3), r = 1.5))

function _pp_rhs(k)
    return function (p)
        f = -(k.ax(p) * _pp_ux(p) + k.a(p) * _pp_uxx(p)) -
            (k.cy(p) * _pp_uy(p) + k.c(p) * _pp_uyy(p))
        if k.b !== nothing
            f -= k.bx(p) * _pp_uy(p) + k.by(p) * _pp_ux(p) + 2k.b(p) * _pp_uxy(p)
        end
        return f + k.β[1] * _pp_ux(p) + k.β[2] * _pp_uy(p) + k.r * _pp_u(p)
    end
end

const _PP_CASES = (fig1_poisson = (_PP_FIG1, :poisson), fig1_mixed = (_PP_FIG1, :mixed),
    lshape_mixed = (_PP_LSHAPE, :mixed), fig1_variable = (_PP_FIG1, :variable))

# n seed lines in (lo, hi): jittered about the uniform positions, or the uniform ones.
function _pp_seeds(rng, lo, hi, n, kind)
    kind === :uniform && return [lo + i * (hi - lo) / (n + 1) for i in 1:n]
    return [lo + (i + 0.7 * (rand(rng) - 0.5)) * (hi - lo) / (n + 1) for i in 1:n]
end

# The H¹ₕ error over the edges with both ends in Ω̄ and the max error over the interior
# nodes, of the grid function `uh` against the exact solution.
function _pp_errors(xs, ys, inn, Γ, uh)
    nx, ny = length(xs), length(ys)
    E = reshape(uh, nx, ny) .- [_pp_u((x, y)) for x in xs, y in ys]
    In = reshape(inn, nx, ny)
    Ω̄ = In .| reshape(Γ, nx, ny)
    half(z, j) = ((j < length(z) ? z[j + 1] : z[j]) - (j > 1 ? z[j - 1] : z[j])) / 2
    s = 0.0
    for l in 1:ny, j in 1:(nx - 1)

        Ω̄[j, l] && Ω̄[j + 1, l] || continue
        s += half(ys, l) * (E[j + 1, l] - E[j, l])^2 / (xs[j + 1] - xs[j])
    end
    for l in 1:(ny - 1), j in 1:nx

        Ω̄[j, l] && Ω̄[j, l + 1] || continue
        s += half(xs, j) * (E[j, l + 1] - E[j, l])^2 / (ys[l + 1] - ys[l])
    end
    return sqrt(s), maximum(abs, E[In])
end

# One level: closure of the seed lines, solve, (hₘₐₓ, H¹ₕ error, max error).
function _pp_level(V, k, n, seed, kind)
    rng = Xoshiro(1000seed + n)
    X = interval(0.0, 6.0) × interval(0.0, 5.0)
    P = Bramble.polygon(V)
    Ω = domain(X, :P => P)
    sx = [0.0; _pp_seeds(rng, 0.0, 6.0, n, kind); 6.0]
    sy = [0.0; _pp_seeds(rng, 0.0, 5.0, n, kind); 5.0]
    xs, ys = Bramble._grid_closure(P, sx, sy)
    Ωₕ = mesh(domain(X), (length(xs), length(ys)), false)
    Bramble.change_points!(Ωₕ, markers(Ω), (xs, ys))

    U, W = Bramble.polygon_spaces(Ωₕ, :P, _pp_u)
    S = gridspace(Ωₕ)
    nx, ny = length(xs), length(ys)
    aₓ, cᵧ, fₕ = element(S), element(S), element(S)
    A3, C3 = reshape(aₓ.data, nx, ny), reshape(cᵧ.data, nx, ny)
    for l in 1:ny, j in 1:nx

        A3[j, l] = j > 1 ? k.a(((xs[j - 1] + xs[j]) / 2, ys[l])) : 0.0
        C3[j, l] = l > 1 ? k.c((xs[j], (ys[l - 1] + ys[l]) / 2)) : 0.0
    end
    avgₕ!(fₕ, _pp_rhs(k))
    diffusion(u) = -D̃ₓ(aₓ * D₋ₓ(u)) - D̃ᵧ(cᵧ * D₋ᵧ(u))
    op = if k.b !== nothing
        bₕ = Rₕ(S, k.b)
        u -> diffusion(u) - Dcₓ(bₕ * Dcᵧ(u)) - Dcᵧ(bₕ * Dcₓ(u))
    elseif k.r != 0
        u -> diffusion(u) + k.β[1] * Dcₓ(u) + k.β[2] * Dcᵧ(u) + k.r * u
    else
        diffusion
    end
    A, F = assemble(form(U, W, (u, v) -> innerₕ(op(u), v)), form(W, v -> innerₕ(fₕ, v)))
    uh = Bramble.lift(U, A \ F).data
    m = markers(Ωₕ)
    e1, emax = _pp_errors(xs, ys, BitVector(m[:P]), BitVector(m[:P_boundary]), uh)
    return max(maximum(diff(xs)), maximum(diff(ys))), e1, emax
end

_pp_gmean(v) = exp(sum(log, v) / length(v))

"""
    polygon_errors(case; seeds = 1:3, levels = (6, 12, 24, 48, 96, 192), kind = :jitter)
        -> (h, e1, emax)

Vectors over `levels` of the geometric mean over `seeds` of hₘₐₓ, the H¹ₕ error and the max
error, for `case` in `keys(_PP_CASES)`. `kind = :uniform` places the seeds uniformly.
"""
function polygon_errors(case::Symbol; seeds = 1:3, levels = (6, 12, 24, 48, 96, 192),
        kind::Symbol = :jitter)
    V, coef = _PP_CASES[case]
    k = _PP_COEF[coef]
    R = [_pp_level(V, k, n, s, kind) for n in levels, s in seeds]
    pool(i) = [_pp_gmean([R[l, s][i] for s in axes(R, 2)]) for l in axes(R, 1)]
    return (h = pool(1), e1 = pool(2), emax = pool(3))
end

# Least-squares slope of log e against log h.
function _pp_order(h, e)
    x, y = log.(h), log.(e)
    x̄, ȳ = sum(x) / length(x), sum(y) / length(y)
    return sum((x .- x̄) .* (y .- ȳ)) / sum(abs2, x .- x̄)
end

# (case, kind, H¹ₕ floor, max floor); `nothing`: the max order is bounded above only.
const _PP_GATES = ((:fig1_poisson, :jitter, 1.8, 1.8), (:fig1_mixed, :jitter, 1.4, nothing),
    (:lshape_mixed, :jitter, 1.8, 1.8), (:fig1_variable, :jitter, 1.8, 1.8),
    (:fig1_poisson, :uniform, 1.8, 1.8))

@testset "Polygon convergence" begin
    # The slow group adds two more disjoint seed triples per case.
    triples = _PP_SLOW ? (1:3, 4:6, 7:9) : (1:3,)
    @testset "$case, $kind seeds" for (case, kind, lo1, lomax) in _PP_GATES
        for seeds in (kind === :uniform ? (1:1,) : triples)
            o = polygon_errors(case; seeds, kind)
            p1, pmax = _pp_order(o.h, o.e1), _pp_order(o.h, o.emax)
            @test lo1 <= p1 < 3
            lomax === nothing || @test lomax <= pmax
            @test pmax < 3
        end
    end
end
