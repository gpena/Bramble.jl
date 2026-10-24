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
# errors, `norminf` and `snorm₁ₕ` of `uₕ - Rₕ(U, u)` (#879), the max error over the interior
# nodes and the H¹ₕ error over the grid edges with both ends in Ω̄ (dual length times the
# squared difference over the edge length); one level checks them against those sums
# written out by hand (`_ls_errors`). The norms read the error as 0 on Γ, where the error
# is that of g, the offset of the vertex from the curve. That is exact on the disk, whose
# vertices lie on the curve, but not on the ellipse, so there the H¹ₕ gates take the hand
# sum with the lifted solution, g, on Γ (`e1Γ`). The offset is part of the error measured;
# without it the interior error, offset by φ/2 near Γ, jumps across the edges to Γ, and its
# order falls to 1.4 over the levels below. The
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

# The oracle of the errors: the H¹ₕ error over the edges with both ends in Ω̄ and the max
# error over the interior nodes, of the grid function `uh` against the exact solution.
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
    return sqrt(s), maximum(abs, E[In])
end

# Largest gap between the lines that hold a node of Ω̄.
function _ls_hmax(Ωₕ, xs, ys)
    m = markers(Ωₕ)
    Ω̄ = reshape(BitVector(m[:D]) .| BitVector(m[:D_boundary]), length(xs), length(ys))
    jx, jy = findall(vec(any(Ω̄; dims = 2))), findall(vec(any(Ω̄; dims = 1)))
    return max(maximum(diff(xs[first(jx):last(jx)])), maximum(diff(ys[first(jy):last(jy)])))
end

# One level: the mesh of n seeds per axis and solve; the mesh, its lines, the trial space
# and the solution's degrees of freedom.
function _ls_solve(φ, box, n, seed)
    Random.seed!(1000seed + n)
    Ωₕ = mesh(domain(box, :D => levelset(φ)), (n, n), (false, false))
    xs, ys = collect(points(Ωₕ(1))), collect(points(Ωₕ(2)))
    U, W = polygon_spaces(Ωₕ, :D, p -> _ls_u(p) + φ(p) / 2)
    fₕ = element(gridspace(Ωₕ))
    avgₕ!(fₕ, _ls_f)
    A, F = assemble(form(U, W, (u, v) -> innerₕ(-1.0 * Δₕ(u), v)),
        form(W, v -> innerₕ(fₕ, v)))
    return Ωₕ, xs, ys, U, A \ F
end

# One level: (hₘₐₓ, H¹ₕ error, max error, H¹ₕ error with g on Γ).
function _ls_level(φ, box, n, seed)
    Ωₕ, xs, ys, U, u = _ls_solve(φ, box, n, seed)
    e = element(U, u) - Rₕ(U, _ls_u)
    m = markers(Ωₕ)
    e1Γ = first(_ls_errors(xs, ys, BitVector(m[:D]), BitVector(m[:D_boundary]),
        lift(U, u).data))
    return _ls_hmax(Ωₕ, xs, ys), snorm₁ₕ(e), norminf(e), e1Γ
end

_ls_gmean(v) = exp(sum(log, v) / length(v))

"""
    levelset_errors(curve; seeds = 1:3, levels) -> (h, e1, emax, e1Γ)

Vectors over `levels` of the geometric mean over `seeds` of hₘₐₓ, the H¹ₕ error, the max
error and the H¹ₕ error with g on Γ, for `curve` in `keys(_LS_CURVES)`.
"""
function levelset_errors(curve::Symbol; seeds = 1:3, levels)
    φ, box = _LS_CURVES[curve]
    R = [_ls_level(φ, box, n, s) for n in levels, s in seeds]
    pool(i) = [_ls_gmean([R[l, s][i] for s in axes(R, 2)]) for l in axes(R, 1)]
    return (h = pool(1), e1 = pool(2), emax = pool(3), e1Γ = pool(4))
end

# Least-squares slope of log e against log h.
function _ls_order(h, e)
    x, y = log.(h), log.(e)
    x̄, ȳ = sum(x) / length(x), sum(y) / length(y)
    return sum((x .- x̄) .* (y .- ȳ)) / sum(abs2, x .- x̄)
end

@testset "Level-set convergence (#834)" begin
    # On the disk the vertices lie on the curve to the root finder's precision, so g = u
    # on Γ and the hand sums, which read Γ, measure the same error as the norms.
    @testset "Norm errors match the hand sums" begin
        φ, box = _LS_CURVES[:disk]
        Ωₕ, xs, ys, U, u = _ls_solve(φ, box, 20, 1)
        e = element(U, u) - Rₕ(U, _ls_u)
        m = markers(Ωₕ)
        e1, emax = _ls_errors(xs, ys, BitVector(m[:D]), BitVector(m[:D_boundary]),
            lift(U, u).data)
        @test emax > 1e-6   # the solution is not exact: the sums are not both 0
        @test snorm₁ₕ(e) ≈ e1 rtol = 1e-10
        @test norminf(e) == emax
    end

    @testset "disk, closure path: second order" begin
        o = levelset_errors(:disk; levels = (20, 40, 80, 160))
        @test _ls_order(o.h, o.emax) >= 1.8
        @test _ls_order(o.h, o.e1) >= 1.8
    end
    @testset "ellipse 30°, orbit path: error ≤ c h²" begin
        o = levelset_errors(:ellipse30; levels = (40, 80, 160, 320))
        @test all(o.emax ./ o.h .^ 2 .<= 0.3)
        @test all(o.e1Γ ./ o.h .^ 2 .<= 0.6)
        @test _ls_order(o.h, o.emax) >= 1.7
        @test _ls_order(o.h, o.e1Γ) >= 1.7
    end
end
