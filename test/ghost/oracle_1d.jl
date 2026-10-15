module GhostOracle1dTests

# Pins the 1D ghost-point oracle (GhostOracle1D.jl) against notes §1 on a random mesh:
# operator boundary rows entry by entry, summation by parts, the ghost closure, the
# variational form of the eliminated rows, the augmented route, then the orders.

using Test
using Bramble
import Bramble
using LinearAlgebra: Diagonal, dot, norm

isdefined(Main, :GhostOracle1D) ||
    Base.include(Main, joinpath(@__DIR__, "GhostOracle1D.jl"))
const O = Main.GhostOracle1D

# Paper index i ∈ 0:n+1 to the 1-based extended index.
k(i) = i + 1

const Ωₕ = O.make_mesh(:random, 0; n0 = 9, seed = 7)
const x = collect(Float64, points(Ωₕ))
const n = length(x)
const ops = O.operators(Ωₕ)
const h2 = x[2] - x[1]
const hn = x[n] - x[n - 1]
const tol = 1e-12

rowvec(A, i) = Vector(A[k(i), :])

# The expected row with `entries` (paper index => value) and zeros elsewhere.
function expected(entries...)
    r = zeros(n + 2)
    for (i, v) in entries
        r[k(i)] = v
    end
    return r
end

@testset "1D ghost oracle" begin
    @testset "Mesh facts (notes §0)" begin
        @test !Bramble.is_uniform(Ωₕ)
        @test O.extended_spacings(Ωₕ)[2:end] ≈ vcat(collect(Bramble.spacings(Ωₕ)), hn)
        @test O.cell_measures(Ωₕ) ≈ collect(Bramble.half_spacings(Ωₕ))
        @test O.extended_points(Ωₕ)[[1, n + 2]] ≈ [x[1] - h2, x[n] + hn]
    end

    @testset "Boundary rows (notes §1.1)" begin
        # D₋ at 1 reads the ghost; at n+1 it is evaluated at the ghost; empty at 0.
        @test rowvec(ops.Dminus, 1) ≈ expected(0 => -1 / h2, 1 => 1 / h2) rtol=tol
        @test rowvec(ops.Dminus, n + 1) ≈ expected(n => -1 / hn, n + 1 => 1 / hn) rtol=tol
        @test iszero(rowvec(ops.Dminus, 0))
        @test rowvec(ops.Dstar, 1) ≈ expected(1 => -1 / h2, 2 => 1 / h2) rtol=tol
        @test rowvec(ops.Dstar, n) ≈ expected(n => -1 / hn, n + 1 => 1 / hn) rtol=tol
        @test iszero(rowvec(ops.Dstar, 0)) && iszero(rowvec(ops.Dstar, n + 1))
        @test rowvec(ops.Dc, 1) ≈ expected(0 => -1 / 2h2, 2 => 1 / 2h2) rtol=tol
        @test rowvec(ops.Dc, n) ≈ expected(n - 1 => -1 / 2hn, n + 1 => 1 / 2hn) rtol=tol
        @test rowvec(ops.Lap, 1) ≈ expected(0 => 1 / h2^2, 1 => -2 / h2^2, 2 => 1 / h2^2) rtol=tol
        @test rowvec(ops.Lap, n) ≈
              expected(n - 1 => 1 / hn^2, n => -2 / hn^2, n + 1 => 1 / hn^2) rtol=tol
        @test iszero(rowvec(ops.Lap, 0)) && iszero(rowvec(ops.Lap, n + 1))
        @test rowvec(ops.Mminus, 1) ≈ expected(0 => 0.5, 1 => 0.5)
        @test rowvec(ops.Mminus, n + 1) ≈ expected(n => 0.5, n + 1 => 0.5)
        @test rowvec(ops.Jump, 0) ≈ expected(0 => -1.0, 1 => 1.0)
        @test rowvec(ops.Jump, n) ≈ expected(n => -1.0, n + 1 => 1.0)
        @test iszero(rowvec(ops.Jump, n + 1))
        # One interior non-uniform Δ_H row: 2/(h_i + h_{i+1}) (D₋u_{i+1} − D₋u_i).
        hi, hf = x[3] - x[2], x[4] - x[3]
        s = hi + hf
        @test rowvec(ops.Lap, 3) ≈
              expected(2 => 2 / (s * hi), 3 => -2 / (s * hi) - 2 / (s * hf),
            4 => 2 / (s * hf)) rtol=tol
    end

    @testset "Summation by parts (§1.1)" begin
        # (D̃u, v)_{H,*} = −(u, D₋v)_+ + Σ η_i (M₋u)_{i+1} v_i, as bilinear forms on V_H*.
        ĥ = O.cell_measures(Ωₕ)
        wstar = vcat(0.0, ĥ, 0.0)
        wplus = [2 <= i <= n ? x[i] - x[i - 1] : 0.0 for i in 0:(n + 1)]
        # As matrices M with (form) = vᵀ M u.
        lhs = Matrix(Diagonal(wstar) * ops.Dstar)
        bnd = zeros(n + 2, n + 2)
        bnd[k(1), :] .= -Vector(ops.Mminus[k(2), :])
        bnd[k(n), :] .= Vector(ops.Mminus[k(n + 1), :])
        rhs = -Matrix(ops.Dminus' * Diagonal(wplus)) + bnd
        @test norm(lhs - rhs) <= tol * norm(lhs)
    end

    α, β, g = (2.0, 0.5), (1.5, 3.0), (0.7, -1.3)
    robin = (xmin = O.Robin(α[1], β[1], g[1]), xmax = O.Robin(α[2], β[2], g[2]))
    neumann = (xmin = O.Neumann(β[1], g[1]), xmax = O.Neumann(β[2], g[2]))
    mixed = (xmin = O.Dirichlet(0.4), xmax = O.Robin(α[2], β[2], g[2]))
    f = t -> sin(3t) + 1

    @testset "Ghost closure" begin
        u = sin.(4 .* x) .+ x .^ 2
        E, d = O.ghost_closure(neumann, Ωₕ)
        ue = E * u + d
        @test ue[k(0)] ≈ u[2] + 2h2 * g[1] / β[1]
        @test ue[k(n + 1)] ≈ u[n - 1] + 2hn * g[2] / β[2]
        E, d = O.ghost_closure(robin, Ωₕ)
        ue = E * u + d
        @test ue[k(0)] ≈ u[2] + (2h2 / β[1]) * (g[1] - α[1] * u[1])
        @test ue[k(n + 1)] ≈ u[n - 1] + (2hn / β[2]) * (g[2] - α[2] * u[n])
        @test ue[k.(1:n)] == u
        for side in (:xmin, :xmax)
            r, gs = O.relation_row(getfield(robin, side), Ωₕ, side)
            @test dot(r, ue) ≈ gs
        end
        E, d = O.ghost_closure(mixed, Ωₕ)
        @test size(E) == (n + 2, n - 1)
        ue = E * u[2:n] + d
        @test ue[k(1)] == 0.4 && iszero(E[k(1), :])
        @test iszero(E[k(0), :]) && d[k(0)] == 0
        @test ue[k(n + 1)] ≈ u[n - 1] + (2hn / β[2]) * (g[2] - α[2] * u[n])
    end

    @testset "Eliminated rows are variational" begin
        for (name, rel) in (("Neumann", neumann), ("Robin", robin), ("mixed", mixed))
            A, F = O.eliminated_system(rel, Ωₕ, f)
            Av, Fv = O.variational_system(rel, Ωₕ, f)
            @test norm(A - Av) <= tol * norm(Av)
            @test norm(F - Fv) <= tol * norm(Fv)
            @test norm(A - A') <= tol * norm(A)
        end
        # The Neumann rows annihilate constants; the boundary rows carry g/β.
        A, F = O.eliminated_system(neumann, Ωₕ, f)
        ĥ = O.cell_measures(Ωₕ)
        @test norm(A * ones(n)) <= tol * norm(A)
        @test F[[1, n]] ≈ ĥ[[1, n]] .* f.(x[[1, n]]) .+ [g[1] / β[1], g[2] / β[2]]
    end

    @testset "Augmented equals eliminated" begin
        A, F = O.augmented_system(robin, Ωₕ, f)
        @test size(A) == (n + 2, n + 2)
        for (row, side) in ((n + 1, :xmin), (n + 2, :xmax))
            r, gs = O.relation_row(getfield(robin, side), Ωₕ, side)
            @test Vector(A[row, :]) ≈ r[vcat(k.(1:n), k(0), k(n + 1))]
            @test F[row] == gs
        end
        for p in (:robin, :neumann)
            ua = O.solve(Symbol(p, :_a), Ωₕ)
            ub = O.solve(Symbol(p, :_b), Ωₕ)
            @test length(ub) == n + 2
            @test norm(ua - ub[1:n]) <= tol * norm(ua)
            rel = O.problem_relations(Symbol(p, :_a), Ωₕ)
            E, d = O.ghost_closure(rel, Ωₕ)
            @test ub[[n + 1, n + 2]] ≈ (E * ua + d)[[k(0), k(n + 1)]]
        end
        @test O.system_matrix(:neumann_a, Ωₕ) * ones(n) ≈ zeros(n) atol=tol
        @test dot(O.mean_weights(Ωₕ), O.solve(:neumann_a, Ωₕ)) ≈ 0 atol=tol
        @test_throws ArgumentError O.system_matrix(:dirichlet, Ωₕ)
        @test_throws ArgumentError O.observed_order(:robin_a, :uniform; dim = 2)
    end

    @testset "Second order" begin
        for p in O.PROBLEMS, kind in O.MESH_KINDS

            r = O.observed_order(p, kind)
            @test r > 1.8
        end
    end
end

end # module GhostOracle1dTests
