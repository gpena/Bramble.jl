# Convergence of the Ferreira and Grigorieff scheme on level-set domains (#834).
#
# `mesh` fits a grid to the curve φ = 0 and `polygon_spaces` solves on the inscribed polygon.
# The disk takes the closure path: the lines grow to close the grid, the seeds are kept, and
# the error falls at second order. The rotated ellipse takes the orbit path: the lines are
# fitted anew and the vertices lie within h²/4 of the curve, so the error is bounded by a
# multiple of h² at every level; the slope is only floored, since the cost rule that picks
# the number of lines makes consecutive orders jump (the architect's report §5.2), so
# consecutive orders are not asserted.
#
# The boundary datum is g = u + φ/2: it equals u on the curve and not at a vertex, which
# lies off the curve, so the offset of the polygon from the curve is part of the error. The
# load is `avgₕ` of f = -Δu, the dual-cell mean as in poisson_polygon.jl, and so are the
# errors: the max error over the interior nodes, and the H¹ₕ error over the grid edges with
# both ends in Ω̄ (dual length times the squared difference over the edge length). The
# step hₘₐₓ is the largest gap between the lines that hold a node of Ω̄ (the lines outside
# the curve's extent are seeds that no node of the domain uses). Errors are pooled over three
# seeds (the geometric mean per level), the order is the least-squares slope of log e
# against log hₘₐₓ.
#
# Self-contained: the file needs only `Test` in the including module, and defines
# `levelset_errors` there.

using Bramble
using Bramble: lift, polygon_spaces
using Random

# Exact solution and f = −Δu: the sine gives 4 + 2.25 and the exponential −(0.36 + 0.16).
_ls_u(p) = sin(2p[1] + 0.5) * cos(1.5p[2] - 0.2) + exp(0.6p[1] - 0.4p[2])
_ls_f(p) = 6.25 * sin(2p[1] + 0.5) * cos(1.5p[2] - 0.2) - 0.52 * exp(0.6p[1] - 0.4p[2])

# The ellipse with semi-axes 1 and 0.6, turned by 30°.
function _ls_ellipse(p)
    c, s = cos(π / 6), sin(π / 6)
    return (c * p[1] + s * p[2])^2 + ((-s * p[1] + c * p[2]) / 0.6)^2 - 1
end

# (φ, box of the seeds) of each curve.
const _LS_CURVES = (
    disk = (p -> p[1]^2 + p[2]^2 - 1, interval(-1.25, 1.25) × interval(-1.25, 1.25)),
    ellipse30 = (_ls_ellipse, interval(-1.2, 1.2) × interval(-1.2, 1.2)))

# The H¹ₕ error over the edges with both ends in Ω̄ and the max error over the interior
# nodes, of the grid function `uh` against the exact solution.
function _ls_errors(xs, ys, inn, Γ, uh)
    nx, ny = length(xs), length(ys)
    E = reshape(uh, nx, ny) .- [_ls_u((x, y)) for x in xs, y in ys]
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
    # Largest gap between the lines that hold a node of Ω̄.
    jx, jy = findall(vec(any(Ω̄; dims = 2))), findall(vec(any(Ω̄; dims = 1)))
    h = max(maximum(diff(xs[first(jx):last(jx)])), maximum(diff(ys[first(jy):last(jy)])))
    return h, sqrt(s), maximum(abs, E[In])
end

# One level: the mesh of n seeds per axis, solve, (hₘₐₓ, H¹ₕ error, max error).
function _ls_level(φ, box, n, seed)
    Random.seed!(1000seed + n)
    Ωₕ = mesh(domain(box, :D => levelset(φ)), (n, n), (false, false))
    xs, ys = collect(points(Ωₕ(1))), collect(points(Ωₕ(2)))
    U, W = polygon_spaces(Ωₕ, :D, p -> _ls_u(p) + φ(p) / 2)
    fₕ = element(gridspace(Ωₕ))
    avgₕ!(fₕ, _ls_f)
    A, F = assemble(form(U, W, (u, v) -> innerₕ(-1.0 * Δₕ(u), v)),
        form(W, v -> innerₕ(fₕ, v)))
    uh = lift(U, A \ F).data
    m = markers(Ωₕ)
    return _ls_errors(xs, ys, BitVector(m[:D]), BitVector(m[:D_boundary]), uh)
end

_ls_gmean(v) = exp(sum(log, v) / length(v))

"""
    levelset_errors(curve; seeds = 1:3, levels) -> (h, e1, emax)

Vectors over `levels` of the geometric mean over `seeds` of hₘₐₓ, the H¹ₕ error and the max
error, for `curve` in `keys(_LS_CURVES)`.
"""
function levelset_errors(curve::Symbol; seeds = 1:3, levels)
    φ, box = _LS_CURVES[curve]
    R = [_ls_level(φ, box, n, s) for n in levels, s in seeds]
    pool(i) = [_ls_gmean([R[l, s][i] for s in axes(R, 2)]) for l in axes(R, 1)]
    return (h = pool(1), e1 = pool(2), emax = pool(3))
end

# Least-squares slope of log e against log h.
function _ls_order(h, e)
    x, y = log.(h), log.(e)
    x̄, ȳ = sum(x) / length(x), sum(y) / length(y)
    return sum((x .- x̄) .* (y .- ȳ)) / sum(abs2, x .- x̄)
end

@testset "Level-set convergence (#834)" begin
    @testset "disk, closure path: second order" begin
        o = levelset_errors(:disk; levels = (20, 40, 80, 160))
        @test _ls_order(o.h, o.emax) >= 1.8
        @test _ls_order(o.h, o.e1) >= 1.8
    end
    @testset "ellipse 30°, orbit path: error ≤ c h²" begin
        o = levelset_errors(:ellipse30; levels = (40, 80, 160, 320))
        @test all(o.emax ./ o.h .^ 2 .<= 0.3)
        @test all(o.e1 ./ o.h .^ 2 .<= 0.6)
        @test _ls_order(o.h, o.emax) >= 1.7
        @test _ls_order(o.h, o.e1) >= 1.7
    end
end
