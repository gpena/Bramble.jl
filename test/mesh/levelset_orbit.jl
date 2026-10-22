module MeshLevelSetOrbitTests

using Test
using LinearAlgebra
using Random
using Bramble

# `Bramble._levelset_grid(φ, xs, ys)` fits grid lines to φ = 0 and returns them with the
# inscribed polygon (#834): periodic orbits of the crossing map closed by least squares on the
# orbit path, the seeds' closure on the closure path. The checks here are independent of the
# package's own post-check: φ signs at every node, a finite-difference distance at every
# vertex, two vertices on every line inside the curve's extent.

# n jittered seeds strictly between lo and hi, plus the end points.
function seeds(rng, lo, hi, n)
    return sort([lo; [lo + (i + 0.6 * (rand(rng) - 0.5)) * (hi - lo) / (n + 1) for i in 1:n]; hi])
end

function ellipse(θ)
    c, s = cos(θ), sin(θ)
    return p -> (c * p[1] + s * p[2])^2 + ((-s * p[1] + c * p[2]) / 0.6)^2 - 1
end

function egg(θ)
    c, s = cos(θ), sin(θ)
    return function (p)
        x, y = p[1] - 0.1, p[2] + 0.05
        u, v = c * x + s * y, -s * x + c * y
        return u^2 + v^2 * (1 + 0.4u) / 0.36 - 1
    end
end

disk(p) = hypot(p[1], p[2]) - 1

# |φ| / |∇φ| at p, the gradient by central differences in Float64.
function distance(φ, p)
    x, y, δ = Float64(p[1]), Float64(p[2]), 1e-6
    gx = (φ((x + δ, y)) - φ((x - δ, y))) / 2δ
    gy = (φ((x, y + δ)) - φ((x, y - δ))) / 2δ
    return abs(φ((x, y))) / hypot(gx, gy)
end

# The grid checks: vertices on nodes; consecutive vertices corners of one cell; the polygon
# counter-clockwise; no grid edge joins nodes where φ has opposite signs (vertices excepted);
# every line inside the vertices' extent carries exactly two vertices. Returns h (the largest
# spacing inside that extent) and the largest vertex distance to φ = 0 over h².
function check(φ, X, Y, P)
    W = P.vertices
    V = Set(W)
    @test all(v -> v[1] in X && v[2] in Y, W)
    ij = [(searchsortedfirst(X, v[1]), searchsortedfirst(Y, v[2])) for v in W]
    nv = length(ij)
    @test all(k -> maximum(abs.(ij[k] .- ij[mod1(k + 1, nv)])) == 1, 1:nv)
    @test sum(k -> W[k][1] * W[mod1(k + 1, nv)][2] - W[mod1(k + 1, nv)][1] * W[k][2], 1:nv) > 0
    sg(i, j) = (p = (X[i], Y[j]); p in V ? 0 : sign(φ(p)))
    S = [sg(i, j) for i in eachindex(X), j in eachindex(Y)]
    bad = count(S[1:(end - 1), :] .* S[2:end, :] .< 0) + count(S[:, 1:(end - 1)] .* S[:, 2:end] .< 0)
    @test bad == 0
    xl, xh = extrema(first, W)
    yl, yh = extrema(last, W)
    @test all(x -> count(v -> v[1] == x, W) == 2, filter(x -> xl < x < xh, X))
    @test all(y -> count(v -> v[2] == y, W) == 2, filter(y -> yl < y < yh, Y))
    h = max(maximum(diff(filter(x -> xl <= x <= xh, X))), maximum(diff(filter(y -> yl <= y <= yh, Y))))
    return h, maximum(v -> distance(φ, v), W) / h^2
end

function message(f)
    try
        f()
    catch e
        e isa ArgumentError && return sprint(showerror, e)
        rethrow()
    end
    return ""
end

@testset "Orbit path: rotated ellipse" begin
    rng = MersenneTwister(1)
    φ = ellipse(π / 6)
    for n in (40, 80)
        X, Y, P = Bramble._levelset_grid(φ, seeds(rng, -1.25, 1.25, n), seeds(rng, -1.25, 1.25, n))
        @test length(X) <= 1.5n && length(Y) <= 1.5n
        h, d = check(φ, X, Y, P)
        @test d <= 0.25
        @test h <= 4 * 2.5 / n
    end
end

@testset "Orbit path: tilted egg" begin
    rng = MersenneTwister(2)
    φ = egg(deg2rad(35))
    X, Y, P = Bramble._levelset_grid(φ, seeds(rng, -1.2, 1.3, 80), seeds(rng, -1.25, 1.15, 80))
    @test length(X) <= 4.5 * 80 && length(Y) <= 4.5 * 80
    _, d = check(φ, X, Y, P)
    @test d <= 0.25
end

@testset "Seeds outside the extent are kept" begin
    rng = MersenneTwister(3)
    φ = ellipse(π / 6)
    xs, ys = seeds(rng, -1.25, 1.25, 60), seeds(rng, -1.25, 1.25, 60)
    X, Y, P = Bramble._levelset_grid(φ, xs, ys)
    # The ellipse's half-widths: √(cos²θ + 0.36 sin²θ) in x, √(sin²θ + 0.36 cos²θ) in y.
    for (s, L, e) in ((xs, X, sqrt(0.75 + 0.09)), (ys, Y, sqrt(0.25 + 0.27)))
        d = 2.5 / (length(s) - 1)
        @test all(v -> v in L, filter(v -> abs(v) > e + d / 4, s))
        @test !any(v -> v in L, filter(v -> abs(v) < e, s))
        @test L[1] == -1.25 && L[end] == 1.25
    end
end

@testset "Closure path keeps every seed" begin
    rng = MersenneTwister(4)
    xs, ys = seeds(rng, -1.25, 1.25, 20), seeds(rng, -1.25, 1.25, 20)
    X, Y, P = Bramble._levelset_grid(disk, xs, ys)
    @test all(in(X), xs) && all(in(Y), ys)
    @test maximum(v -> abs(disk(v)), P.vertices) < 1e-12
    _, d = check(disk, X, Y, P)
    @test d < 1e-10
end

@testset "Float32 seeds give Float32 lines" begin
    rng = MersenneTwister(5)
    φ = ellipse(π / 6)
    xs, ys = Float32.(seeds(rng, -1.25, 1.25, 40)), Float32.(seeds(rng, -1.25, 1.25, 40))
    X, Y, P = Bramble._levelset_grid(φ, xs, ys)
    @test eltype(X) === Float32 && eltype(Y) === Float32
    @test P isa Bramble.Polygon{Float32}
    _, d = check(φ, X, Y, P)
    @test d <= 0.25
end

@testset "Named errors" begin
    rng = MersenneTwister(6)
    φ = ellipse(π / 6)
    m = message(() -> Bramble._levelset_grid(φ, seeds(rng, -1.25, 1.25, 4), seeds(rng, -1.25, 1.25, 4)))
    @test occursin("too coarse", m) && occursin("6 × 6 points", m)
    xs, ys = seeds(rng, -1.25, 1.25, 40), seeds(rng, -1.25, 1.25, 40)
    ρ = 0.36227066168253
    m = message(() -> Bramble._levelset_orbit_grid(φ, xs, ys, ρ; cmax = 1e-3))
    @test occursin("too many lines", m)
    # A rotation number off the curve's: the orbits cannot close.
    m = message(() -> Bramble._levelset_orbit_grid(φ, xs, ys, 0.3))
    @test occursin("check failed", m)
    star = p -> hypot(p[1], p[2]) - (0.6 + 0.2 * sin(5 * atan(p[2], p[1])))
    m = message(() -> Bramble._levelset_grid(star, seeds(rng, -1.0, 1.0, 20), seeds(rng, -1.0, 1.0, 20)))
    @test occursin("meets φ = 0 in", m)
end

@testset "A scaled curve gets the scaled grid" begin
    φ = ellipse(π / 6)
    for n in (40, 160)
        xs, ys = seeds(MersenneTwister(8), -1.25, 1.25, n), seeds(MersenneTwister(9), -1.25, 1.25, n)
        X, Y, _ = Bramble._levelset_grid(φ, xs, ys)
        for L in (0.01, 100.0)
            φL = p -> φ((p[1] / L, p[2] / L))
            XL, YL, PL = Bramble._levelset_grid(φL, L .* xs, L .* ys)
            @test length(XL) == length(X) && length(YL) == length(Y)
            @test XL ≈ L .* X rtol = 1e-6
            @test YL ≈ L .* Y rtol = 1e-6
            _, d = check(φL, XL, YL, PL)
            @test d * L <= 0.25
        end
    end
end

# The phases are anchored at the leftmost point in both, so the grids differ inside the
# extent; the line counts, which nin decides, agree.
@testset "Mirror inputs give equal line counts" begin
    rng = MersenneTwister(10)
    a, b = seeds(rng, -1.25, 1.25, 5), seeds(rng, -1.25, 1.25, 80)
    φ₁, φ₂ = ellipse(π / 6), ellipse(π / 3)   # mirror images in y = x
    X₁, Y₁, P₁ = Bramble._levelset_grid(φ₁, a, b)
    X₂, Y₂, P₂ = Bramble._levelset_grid(φ₂, b, a)
    @test length(X₁) == length(Y₂) && length(Y₁) == length(X₂)
    @test length(P₁.vertices) == length(P₂.vertices)
    check(φ₁, X₁, Y₁, P₁)
end

@testset "Duplicated seeds are one seed" begin
    rng = MersenneTwister(11)
    φ = ellipse(π / 6)
    xs, ys = seeds(rng, -1.25, 1.25, 40), seeds(rng, -1.25, 1.25, 40)
    X, Y, P = Bramble._levelset_grid(φ, xs, ys)
    X₂, Y₂, P₂ = Bramble._levelset_grid(φ, sort([xs; xs[2]; xs[end - 1]]), [ys; ys[3]])
    @test X₂ == X && Y₂ == Y && P₂.vertices == P.vertices
end

@testset "Cyclic tridiagonal solve" begin
    rng = MersenneTwister(7)
    for q in (2, 3, 8)
        d, e, r = 4 .+ rand(rng, q), rand(rng, q) .- 0.5, rand(rng, q)
        A = zeros(q, q)
        for j in 1:q
            A[j, j] += d[j]
            j₊ = mod1(j + 1, q)
            A[j, j₊] += e[j]
            A[j₊, j] += e[j]
        end
        x = Bramble._levelset_cyclic_solve!(zeros(q), d, e, r, zeros(4q))
        @test x ≈ A \ r rtol = 1e-12
    end
end

end
