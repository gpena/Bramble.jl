module GhostLevelsetOracleTests

# `levelset_spaces` (#834) against SWOracle.jl, a hand-built Shortley-Weller system that
# shares no code with the package: its own crossings, labels and rows. The library row of an
# interior node is the oracle's pointwise row times the node's dual-cell area (the `innerₕ`
# weight), so the matrix and the load are compared after dividing each library row by that
# area; the load of a constant `f` is `area f`, and the rest is the oracle's `b`, the data
# `g` at the crossings and boundary nodes moved to the right-hand side. `g` is no polynomial,
# so each crossing's datum and weight enter.
#
# Positive control: every grid here has rows whose arms the curve cuts, and the plain
# space's matrix, which reads the exterior nodes as 0 at full arm length, differs from the
# oracle there.

using Test
using Random
using LinearAlgebra: opnorm
using Bramble
import Bramble
using Bramble: levelset, levelset_spaces

isdefined(Main, :SWOracle) || Base.include(Main, joinpath(@__DIR__, "SWOracle.jl"))
const O = Main.SWOracle

const BOX = interval(-1.25, 1.25) × interval(-1.25, 1.25)
disk(p) = hypot(p[1], p[2]) - 1
star(p) = hypot(p[1], p[2]) - (0.6 + 0.2 * sin(5 * atan(p[2], p[1])))
g_var(x) = exp(x[1]) * sin(2 * x[2]) + x[1]^3 * x[2] + 1

lines(Ωₕ) = (collect(points(Ωₕ(1))), collect(points(Ωₕ(2))))

# The dual-cell area of each interior node (none lies on the grid's boundary).
function dual_areas(xs, ys, inside)
    half(z, i) = (z[i + 1] - z[i - 1]) / 2
    return [half(xs, k[1]) * half(ys, k[2]) for k in findall(inside)]
end

relerr(x, y) = maximum(abs, x .- y) / maximum(abs, y)

function compare(φ, Ωₕ; f = -4.0)
    xs, ys = lines(Ωₕ)
    Aoracle, b, inside = O.sw_system(φ, xs, ys, g_var)
    U, W = levelset_spaces(Ωₕ, domain(BOX, :D => levelset(φ)), :D, g_var)
    A, F = assemble(form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w), v)),
        form(W, v -> innerₕ(x -> f, v)))
    ar = dual_areas(xs, ys, inside)
    # The plain space's rows: exterior nodes read as 0 at full arm length.
    Wp = gridspace(Ωₕ)
    I = Bramble.index_in_marker(Ωₕ, :D)
    Az = Matrix(assemble(form(Wp, Wp, (w, v) -> innerₕ(-1.0 * Δₕ(w), v))))[I, I]
    return (; inside, I, A = Matrix(A) ./ ar, b = F ./ ar .- f,
        Aoracle = Matrix(Aoracle), boracle = b, Az = Az ./ ar)
end

@testset "Level-set spaces against the SW oracle" begin
    @testset "$name, $(n) points, seed $seed" for (name, φ) in (("disk", disk),
            ("star", star)),
        (n, seed) in ((23, 3), (41, 8))

        Random.seed!(seed)
        Ωₕ = mesh(domain(BOX, :D => levelset(φ)), (n, n + 3), (false, false))
        @test !Bramble.is_uniform(Ωₕ(1))
        r = compare(φ, Ωₕ)
        @test vec(r.inside) == r.I
        @test size(r.A) == size(r.Aoracle)
        @test relerr(r.A, r.Aoracle) < 1e-10
        @test relerr(r.b, r.boracle) < 1e-10
        @test opnorm(r.A - r.Az, Inf) > 1e-3 * opnorm(r.Aoracle, Inf)
    end

    @testset "jittered lines, nodes within √ε h" begin
        # Lines jittered off a uniform grid; the nodes (1 − 1e-10, 0) and (−1 − 1e-10, 0)
        # are boundary nodes (D9), one with φ < 0 and one with φ > 0, which their interior
        # neighbours read at full arm length with g at the node.
        Random.seed!(11)
        Ω = domain(BOX, :D => levelset(disk))
        Ωₕ = mesh(Ω, (37, 33), (false, false))
        h = 2.5 / 36
        xs = [-1.25; [-1.25 + (i + 0.7 * (rand() - 0.5)) * h for i in 1:35]; 1.25]
        ia, ib = findmin(abs.(xs .- 1))[2], findmin(abs.(xs .+ 1))[2]
        xs[ia], xs[ib] = 1 - 1e-10, -1 - 1e-10
        ys = collect(range(-1.25, 1.25, 33))
        Bramble.change_points!(Ωₕ, markers(Ω), (xs, ys))
        Γ = reshape(Bramble.index_in_marker(Ωₕ, :D_boundary), 37, 33)
        @test Γ[ia, 17] && Γ[ib, 17]
        r = compare(disk, Ωₕ; f = 2.5)
        @test vec(r.inside) == r.I
        @test relerr(r.A, r.Aoracle) < 1e-10
        @test relerr(r.b, r.boracle) < 1e-10
        @test opnorm(r.A - r.Az, Inf) > 1e-3 * opnorm(r.Aoracle, Inf)
    end
end

end # module
