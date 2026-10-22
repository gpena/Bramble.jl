# Convergence of the Shortley-Weller scheme on level-set domains (#834): the disk and the
# five-petal star in the box [−1.25, 1.25]².
#
# Each level builds a fresh mesh `mesh(Ω, (n, n), (false, false))` and moves it by
# `change_points!` onto jittered lines, point i at (i + 0.7(U − ½))h with both ends pinned,
# so the levels are not nested. Errors are pooled over three seeds (the geometric mean per
# level) and the order is the least-squares slope of log e against log hₘₐₓ over the levels
# n = 20, 40, 80, 160, 320.
#
# The max error is over the `:D` nodes. The H¹ₕ error sums, over the grid edges with both
# ends interior or boundary nodes, the dual length times the squared difference over the
# edge length. The load is `avgₕ`, the dual-cell mean of f.
#
# Self-contained: the file needs only `Test` in the including module, and defines
# `levelset_orders` there.

using Test
using Random
using Bramble
using Bramble: levelset, levelset_spaces, D₋ₓ, D₋ᵧ, D̃ₓ, D̃ᵧ, Dcₓ, Dcᵧ

const _PL_BOX = interval(-1.25, 1.25) × interval(-1.25, 1.25)
const _PL_SHAPES = (disk = p -> hypot(p[1], p[2]) - 1,
    star = p -> hypot(p[1], p[2]) - (0.6 + 0.2 * sin(5 * atan(p[2], p[1]))))

# Exact solution and its derivatives.
_pl_u(p) = sin(2p[1]) * cos(2p[2]) + p[1] * p[2]
_pl_ux(p) = 2cos(2p[1]) * cos(2p[2]) + p[2]
_pl_uy(p) = -2sin(2p[1]) * sin(2p[2]) + p[1]
_pl_uxx(p) = -4sin(2p[1]) * cos(2p[2])
_pl_uyy(p) = -4sin(2p[1]) * cos(2p[2])

# −∂ₓ(a ∂ₓu) − ∂ᵧ(c ∂ᵧu) + β·∇u + r u, as `_PP_COEF` in poisson_polygon.jl.
const _PL_COEF = (
    poisson = (a = p -> 1.0, ax = p -> 0.0, c = p -> 1.0, cy = p -> 0.0, β = (0.0, 0.0),
        r = 0.0),
    variable = (a = p -> 2 + 0.5sin(p[1] + p[2]), ax = p -> 0.5cos(p[1] + p[2]),
        c = p -> 2 + 0.3cos(p[1]), cy = p -> 0.0, β = (0.4, -0.3), r = 1.5))

function _pl_rhs(k)
    return p -> -(k.ax(p) * _pl_ux(p) + k.a(p) * _pl_uxx(p)) -
                (k.cy(p) * _pl_uy(p) + k.c(p) * _pl_uyy(p)) +
                k.β[1] * _pl_ux(p) + k.β[2] * _pl_uy(p) + k.r * _pl_u(p)
end

# n points in [lo, hi], jittered about the uniform ones, ends pinned.
function _pl_lines(rng, n)
    lo, hi = -1.25, 1.25
    h = (hi - lo) / (n - 1)
    return [lo; [lo + (i + 0.7 * (rand(rng) - 0.5)) * h for i in 1:(n - 2)]; hi]
end

# The H¹ₕ error over the edges with both ends interior or boundary nodes and the max error
# over the interior nodes, of the grid function `uh` against the exact solution.
function _pl_errors(xs, ys, inn, Γ, uh)
    nx, ny = length(xs), length(ys)
    E = reshape(uh, nx, ny) .- [_pl_u((x, y)) for x in xs, y in ys]
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

# One level: (hₘₐₓ, H¹ₕ error, max error).
function _pl_level(φ, k, n, seed)
    rng = Xoshiro(1000seed + n)
    xs, ys = _pl_lines(rng, n), _pl_lines(rng, n)
    Ω = domain(_PL_BOX, :D => levelset(φ))
    Ωₕ = mesh(Ω, (n, n), (false, false))
    Bramble.change_points!(Ωₕ, markers(Ω), (xs, ys))
    U, W = levelset_spaces(Ωₕ, Ω, :D, _pl_u)
    S = Bramble.padded_space(U)
    aₓ, cᵧ = element(S), element(S)
    A3, C3 = reshape(aₓ.data, n, n), reshape(cᵧ.data, n, n)
    for l in 1:n, j in 1:n

        A3[j, l] = j > 1 ? k.a(((xs[j - 1] + xs[j]) / 2, ys[l])) : 0.0
        C3[j, l] = l > 1 ? k.c((xs[j], (ys[l - 1] + ys[l]) / 2)) : 0.0
    end
    fₕ = avgₕ(S, _pl_rhs(k))
    op(u) = -D̃ₓ(aₓ * D₋ₓ(u)) - D̃ᵧ(cᵧ * D₋ᵧ(u)) + k.β[1] * Dcₓ(u) + k.β[2] * Dcᵧ(u) +
            k.r * u
    A, F = assemble(form(U, W, (u, v) -> innerₕ(op(u), v)), form(W, v -> innerₕ(fₕ, v)))
    uh = Bramble.lift(U, A \ F).data
    m = markers(Ωₕ)
    e1, emax = _pl_errors(xs, ys, BitVector(m[:D]), BitVector(m[:D_boundary]), uh)
    return max(maximum(diff(xs)), maximum(diff(ys))), e1, emax
end

_pl_gmean(v) = exp(sum(log, v) / length(v))

# Least-squares slope of log e against log h.
function _pl_order(h, e)
    x, y = log.(h), log.(e)
    x̄, ȳ = sum(x) / length(x), sum(y) / length(y)
    return sum((x .- x̄) .* (y .- ȳ)) / sum(abs2, x .- x̄)
end

"""
    levelset_orders(shape, case; seeds = 1:3, levels = (20, 40, 80, 160, 320))
        -> (max_order, h1_order)

The orders of the max and H¹ₕ errors on `shape` in `(:disk, :star)` with the coefficients
`case` in `(:poisson, :variable)`.
"""
function levelset_orders(shape::Symbol, case::Symbol; seeds = 1:3,
        levels = (20, 40, 80, 160, 320))
    φ, k = _PL_SHAPES[shape], _PL_COEF[case]
    R = [_pl_level(φ, k, n, s) for n in levels, s in seeds]
    pool(i) = [_pl_gmean([R[l, s][i] for s in axes(R, 2)]) for l in axes(R, 1)]
    h = pool(1)
    return _pl_order(h, pool(3)), _pl_order(h, pool(2))
end

@testset "Level-set convergence" begin
    @testset "$shape, $case" for (shape, case) in
                                 ((:disk, :poisson), (:star, :poisson), (:disk, :variable))
        om, oh = levelset_orders(shape, case)
        @test 1.8 <= om < 3.5
        @test 1.8 <= oh < 3
    end
end
