module GhostCornersTests

# Pins the corner-ghost oracle (GhostCorners.jl, notes §4) on a random 2D mesh: padded
# counts in 2D and 3D, the facet relations' 2×2 coupling at each corner, the cross-weighted
# factor, the three corner closures on bilinear data and which rows read a corner point. The
# orders of every closure × mixed form × mesh kind are in test/convergence/ghost_oracles.jl.

using Test
using Bramble
import Bramble
using LinearAlgebra: det, norm, rank
using SparseArrays: SparseMatrixCSC, findnz

isdefined(Main, :GhostCorners) || Base.include(Main, joinpath(@__DIR__, "GhostCorners.jl"))
const C = Main.GhostCorners
const O1 = Main.GhostOracle1D

const Ωₕ = C.make_mesh(:random, 0; n0 = 9, seed = 7)
const nx, ny = C.sizes(Ωₕ)
const x = collect(Float64, points(Ωₕ(1)))
const y = collect(Float64, points(Ωₕ(2)))
const xp = O1.extended_points(Ωₕ(1))
const yp = O1.extended_points(Ωₕ(2))
const tol = 1e-12

# u = a + b x + c y + e x y and the data GhostCorners derives from it.
const bilinear = (u = (s, t) -> 0.3 + 1.2s - 0.7t + 2.5s * t,
    ux = (s, t) -> 1.2 + 2.5t, uy = (s, t) -> -0.7 + 2.5s,
    uxx = (s, t) -> 0.0, uxy = (s, t) -> 2.5, uyy = (s, t) -> 0.0)

# The values of `ex.u` on the whole padded grid, x fastest.
padded_values(ex) = vec([ex.u(a, b) for a in xp, b in yp])
physical_values(ex) = vec([ex.u(a, b) for a in x, b in y])
corner_indices() = [C.padded(i, j, nx) for (i, j) in C.corner_points(nx, ny)]

@testset "Corner ghosts" begin
    @testset "Padded counts" begin
        @test C.padded_ghost_count((5, 7)) == 2 * 5 + 2 * 7 + 4
        @test C.padded_ghost_count((4, 5, 6)) == 6 * 7 * 8 - 4 * 5 * 6
        @test C.facet_ghost_count((5, 7)) == 2 * 5 + 2 * 7
        @test C.facet_ghost_count((4, 5, 6)) == 2 * (5 * 6 + 4 * 6 + 4 * 5)
        # 3D: what padding adds beyond facet ghosts is 12 edge lines and 8 corners.
        n = (4, 5, 6)
        @test C.padded_ghost_count(n) - C.facet_ghost_count(n) == 4 * sum(n) + 8
        @test length(C.facet_ghosts(nx, ny)) == C.facet_ghost_count((nx, ny))
    end

    @testset "Mesh is a random tensor mesh" begin
        @test !Bramble.is_uniform(Ωₕ(1)) && !Bramble.is_uniform(Ωₕ(2))
        @test x != y
    end

    @testset "Corner relations are 2×2" begin
        R, _ = C.ghost_relations(Ωₕ)
        gcols = [t[4] for t in C.facet_ghosts(nx, ny)]
        Rg = R[:, gcols]
        # Off the diagonal, the ghost columns are coupled only in corner pairs.
        rows, cols, _ = findnz(Rg)
        off = sort([(r, c) for (r, c) in zip(rows, cols) if r != c])
        gs = C.facet_ghosts(nx, ny)
        pairs = Tuple{Int, Int}[]
        for (i, j) in C.corner_nodes(nx, ny)
            rx = findfirst(t -> C.facet_axis(t[1]) == 1 && t[2] == i && t[3] == j, gs)
            ry = findfirst(t -> C.facet_axis(t[1]) == 2 && t[2] == i && t[3] == j, gs)
            append!(pairs, [(rx, ry), (ry, rx)])
        end
        @test off == sort(pairs)
        @test all(!iszero, [Rg[r, r] for r in axes(Rg, 1)])
        # No relation reads a padded corner point.
        @test iszero(R[:, corner_indices()])
        # det = β_x β_y det(K) / (4 h_x h_y), h the spacings next to the corner node.
        K, β = C.DEFAULT_K, C.DEFAULT_β
        hx = (x[2] - x[1], x[nx] - x[nx - 1])
        hy = (y[2] - y[1], y[ny] - y[ny - 1])
        expect = [β[1] * β[3] * det(K) / (4hx[1] * hy[1]),
            β[2] * β[3] * det(K) / (4hx[2] * hy[1]),
            β[1] * β[4] * det(K) / (4hx[1] * hy[2]),
            β[2] * β[4] * det(K) / (4hx[2] * hy[2])]
        blocks = C.corner_blocks(Ωₕ)
        @test det.(blocks) ≈ expect rtol=tol
        # Solvable exactly when K is non-singular: a rank-one K leaves each block rank one.
        Ksing = [1.0 2.0; 2.0 4.0]
        @test all(B -> rank(B) == 1, C.corner_blocks(Ωₕ; K = Ksing))
        # The closure satisfies every relation, for any physical values.
        u = sin.(3 .* (1:(nx * ny)))
        E, d = C.ghost_closure(Ωₕ)
        Rr, g = C.ghost_relations(Ωₕ)
        @test norm(Rr * (E * u + d) - g) <= tol * norm(g)
    end

    @testset "Cross-weighted factor" begin
        D̽ = C.cross_weighted(Ωₕ(1))
        n = nx
        for i in 1:n
            r = Vector(D̽[O1.ext(i), :])
            @test sum(r) ≈ 0 atol=tol * norm(r)
            @test r' * xp ≈ 1 rtol=tol
            @test r' * xp .^ 2 ≈ 2x[i] atol=tol
        end
        Dc = O1.operators(Ωₕ(1)).Dc
        @test Vector(D̽[O1.ext(1), :]) ≈ Vector(Dc[O1.ext(1), :]) rtol=tol
        @test Vector(D̽[O1.ext(n), :]) ≈ Vector(Dc[O1.ext(n), :]) rtol=tol
        # Interior rows agree with Bramble's D̽ₓ (which, without ghosts, is one-sided at
        # the ends).
        W1 = gridspace(Ωₕ(1))
        f = t -> exp(t) * sin(4t)
        got = (D̽ * f.(xp))[O1.ext.(2:(n - 1))]
        @test got ≈ parent(Bramble.D̽ₓ(Rₕ(W1, f)))[2:(n - 1)] rtol=tol
    end

    @testset "Bilinear data" begin
        up = padded_values(bilinear)
        uh = physical_values(bilinear)
        gcols = [t[4] for t in C.facet_ghosts(nx, ny)]
        cidx = corner_indices()
        for closure in C.CLOSURES
            E, d = C.ghost_closure(Ωₕ; closure, exact = bilinear)
            ue = E * uh + d
            @test ue[gcols] ≈ up[gcols] rtol=tol
            if closure === :padded_mixed
                @test ue[cidx] ≈ up[cidx] rtol=tol
            elseif closure === :padded_extrapolate
                # Off by the mixed difference of the corner cell, e (x_c − x_n)(y_c − y_n).
                miss = [2.5 * (xp[i + 1] - xp[O1.ext(i == 0 ? 1 : nx)]) *
                        (yp[j + 1] - yp[O1.ext(j == 0 ? 1 : ny)])
                        for (i, j) in C.corner_points(nx, ny)]
                @test up[cidx] - ue[cidx] ≈ miss rtol=1e-10
            else
                @test iszero(E[cidx, :]) && iszero(d[cidx])
            end
        end
        # The scheme is exact for bilinear u, except the extrapolated corner at its four
        # corner rows.
        crow = [C.physical(i, j, nx) for (i, j) in C.corner_nodes(nx, ny)]
        for mixed in C.MIXED, closure in C.CLOSURES

            A, F = C.system(:aniso_robin, Ωₕ; closure, mixed, exact = bilinear)
            res = A * uh - F
            if closure === :padded_extrapolate
                @test all(!iszero, res[crow])
                rest = setdiff(1:(nx * ny), crow)
                @test norm(res[rest]) <= tol * norm(F)
            else
                @test norm(res) <= tol * norm(F)
            end
        end
    end

    @testset "Which rows read a corner" begin
        crow = [C.physical(i, j, nx) for (i, j) in C.corner_nodes(nx, ny)]
        cidx = corner_indices()
        P = C.pde_operator(Ωₕ; closure = :facet_onesided)
        @test iszero(P[:, cidx])
        for mixed in C.MIXED
            P = C.pde_operator(Ωₕ; closure = :padded_mixed, mixed)
            Pc = P[:, cidx]
            @test all(k -> !iszero(Pc[crow[k], k]), 1:4)
            @test iszero(Pc[setdiff(1:(nx * ny), crow), :])
        end
    end

    @testset "Mixed forms" begin
        Ωu = C.make_mesh(:uniform, 0; n0 = 9)
        for closure in C.CLOSURES
            Ac = C.system_matrix(:aniso_robin, Ωu; closure, mixed = :centered)
            Ax = C.system_matrix(:aniso_robin, Ωu; closure, mixed = :cross_weighted)
            @test Ac ≈ Ax rtol=tol
            @test size(Ac) == (81, 81)
            Ac = C.system_matrix(:aniso_robin, Ωₕ; closure, mixed = :centered)
            Ax = C.system_matrix(:aniso_robin, Ωₕ; closure, mixed = :cross_weighted)
            @test norm(Ac - Ax) > 1e-3 * norm(Ac)
        end
        @test_throws ArgumentError C.system_matrix(:aniso_robin, Ωₕ; closure = :none)
        @test_throws ArgumentError C.system_matrix(:aniso_robin, Ωₕ; mixed = :forward)
        @test_throws ArgumentError C.system_matrix(:robin_a, Ωₕ)
        @test_throws ArgumentError C.observed_order(:aniso_robin, :uniform; dim = 3)
    end
end

end # module GhostCornersTests
