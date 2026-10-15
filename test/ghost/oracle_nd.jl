module GhostOracleNdTests

# Pins the 2D/3D facet-only ghost oracle (GhostOracleND.jl) against notes §2 on random
# meshes: ghost count, order and coordinates; that Δ_H at physical nodes never reads a
# padded corner or edge point; the per-(facet, node) closure, a corner node taking one
# datum per incident facet; the eliminated rows against the tensor product of GhostOracle1D's
# variational 1D matrices; the augmented route; then the orders, printed.

using Test
using Bramble
import Bramble
using LinearAlgebra: Diagonal, dot, norm, ×
using SparseArrays: nnz, sparse

using ..TestUtils: WITH_SLOW_TESTS, WITH_3D_TESTS

isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(@__DIR__, "GhostOracleND.jl"))
const O = Main.GhostOracleND
const O1 = Main.GhostOracle1D

const tol = 1e-12

function random_mesh(ns; seed = 11)
    Ω = domain(foldl(×, ntuple(_ -> interval(0.0, 1.0), length(ns))))
    Bramble._seed_mesh1d_rng!(seed)
    try
        return mesh(Ω, ns, ntuple(_ -> false, length(ns)))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end

const Ω2 = random_mesh((6, 7))
# The 3D mesh is built only where the group allows it (`unit` builds none, #650).
const Ω3 = WITH_3D_TESTS ? random_mesh((4, 5, 6)) : nothing
const meshes = WITH_3D_TESTS ? (Ω2, Ω3) : (Ω2,)

# Different α, β and data on every facet; the data depend on the point.
function test_relations(D; neumann = false)
    α = (2.0, 0.5, 1.0, 3.0, 0.7, 1.5)
    β = (1.5, 3.0, 2.0, 1.0, 0.8, 2.5)
    rels = map(1:(2D)) do j
        g = x -> 0.3j + sum(x) - j * x[1]^2
        neumann ? O.FacetNeumann(β[j], g) : O.FacetRobin(α[j], β[j], g)
    end
    return NamedTuple{O.FACETS[1:(2D)]}(Tuple(rels))
end

# The tensor-product variational matrix: Σ_d (⊗_{e≠d} diag ĥ_e) ⊗ K_d, with K_d the 1D
# variational matrix (its α/β terms) on axis d.
function variational_matrix(relations, Ωₕ)
    ms = O.axis_meshes(Ωₕ)
    D = length(ms)
    total = 0.0 * sparse(Diagonal(ones(prod(length.(O.axis_points(Ωₕ))))))
    for d in 1:D
        cmin = getfield(relations, O.facet_symbol(d, :min))
        cmax = getfield(relations, O.facet_symbol(d, :max))
        rel1 = (xmin = O1.Robin(cmin.α, cmin.β, 0.0), xmax = O1.Robin(cmax.α, cmax.β, 0.0))
        K = first(O1.variational_system(rel1, ms[d], x -> 0.0))
        factors = [e == d ? K : sparse(Diagonal(O1.cell_measures(ms[e]))) for e in 1:D]
        total += foldl((a, b) -> kron(b, a), factors)
    end
    return total
end

@testset "ND ghost oracle" begin
    @testset "Ghost count and coordinates" begin
        @test O.ghost_count((5, 7)) == 24
        @test O.ghost_count((4, 5, 6)) == 2 * (30 + 24 + 20)
        for Ωₕ in meshes
            ns = O.sizes(Ωₕ)
            D = length(ns)
            @test !Bramble.is_uniform(Ωₕ)
            L = O.dof_layout(Ωₕ)
            @test length(L.ghost) == O.ghost_count(ns)
            # The padded points left out: 4 corners in 2D; 8 corners and 12 edge lines in
            # 3D (4 along each axis).
            nabsent = D == 2 ? 4 : 8 + 4 * sum(ns)
            @test length(L.absent) == nabsent
            @test length(L.kept) + nabsent == prod(ns .+ 2)
            xs = O.axis_points(Ωₕ)
            gp = O.ghost_points(Ωₕ)
            ok = true
            for ((d, side, I), p) in zip(O.ghosts(Ωₕ), gp)
                n = ns[d]
                xg = side === :min ? xs[d][1] - (xs[d][2] - xs[d][1]) :
                     xs[d][n] + (xs[d][n] - xs[d][n - 1])
                ok &= p[d] ≈ xg
                ok &= all(p[e] == xs[e][I[e]] for e in 1:D if e != d)
                ok &= I[d] == (side === :min ? 1 : n)
            end
            @test ok
        end
        # DOF order: xmin ghosts first, in the order of their nodes, then xmax, ymin, …
        g = O.ghosts(Ω2)
        @test g[1:7] == [(1, :min, CartesianIndex(1, j)) for j in 1:7]
        @test g[8:14] == [(1, :max, CartesianIndex(6, j)) for j in 1:7]
        @test g[15:20] == [(2, :min, CartesianIndex(i, 1)) for i in 1:6]
        @test g[21:26] == [(2, :max, CartesianIndex(i, 7)) for i in 1:6]
    end

    @testset "Δ_H never reads absent points" begin
        for Ωₕ in meshes
            L = O.dof_layout(Ωₕ)
            ops = O.operators(Ωₕ)
            @test nnz(ops.Lap[L.phys, L.absent]) == 0
            @test nnz(ops.Lap[L.phys, L.kept]) > 0
            # Contrast: a mixed difference does read them, so the check can fail.
            @test nnz((ops.Dc[1] * ops.Dc[2])[L.phys, L.absent]) > 0
        end
        # The 2D corner row: one ghost per incident facet.
        L = O.dof_layout(Ω2)
        xs = O.axis_points(Ω2)
        hx, hy = xs[1][2] - xs[1][1], xs[2][2] - xs[2][1]
        LP = LinearIndices(L.ps)
        row = Vector(O.operators(Ω2).Lap[LP[2, 2], :])
        expected = zeros(prod(L.ps))
        expected[LP[1, 2]] = 1 / hx^2
        expected[LP[3, 2]] = 1 / hx^2
        expected[LP[2, 1]] = 1 / hy^2
        expected[LP[2, 3]] = 1 / hy^2
        expected[LP[2, 2]] = -2 / hx^2 - 2 / hy^2
        @test row ≈ expected rtol=tol
        @test LP[1, 2] in L.ghost && LP[2, 1] in L.ghost
    end

    @testset "Closure per facet and node" begin
        for Ωₕ in meshes
            ns = O.sizes(Ωₕ)
            np = prod(ns)
            xs = O.axis_points(Ωₕ)
            rels = test_relations(length(ns))
            E, d = O.ghost_closure(rels, Ωₕ)
            u = sin.(1:np) .+ 0.1 .* (1:np)
            ue = E * u + d
            @test size(E) == (np + O.ghost_count(ns), np)
            @test ue[1:np] == u
            LI = LinearIndices(ns)
            maxdiff = 0.0
            for (k, (ax, side, I)) in enumerate(O.ghosts(Ωₕ))
                c = getfield(rels, O.facet_symbol(ax, side))
                n = ns[ax]
                inward = side === :min ? I + CartesianIndex(ntuple(==(ax), length(ns))) :
                         I - CartesianIndex(ntuple(==(ax), length(ns)))
                h = side === :min ? xs[ax][2] - xs[ax][1] : xs[ax][n] - xs[ax][n - 1]
                g = c.g(O.node_point(xs, I))
                want = u[LI[inward]] + (2h / c.β) * (g - c.α * u[LI[I]])
                maxdiff = max(maxdiff, abs(ue[np + k] - want) / max(1, abs(want)))
            end
            @test maxdiff <= tol
        end
        # A 2D corner node: one ghost and one datum per incident facet.
        rels = test_relations(2)
        g = O.ghosts(Ω2)
        corner = CartesianIndex(1, 1)
        @test count(t -> t[3] == corner, g) == 2
        @test Set(t[1] for t in g if t[3] == corner) == Set([1, 2])
    end

    @testset "Eliminated rows are variational" begin
        for Ωₕ in meshes, neumann in (false, true)

            D = Bramble.dim(Ωₕ)
            rels = test_relations(D; neumann)
            f = x -> sin(3x[1]) + x[end]^2
            A, F = O.eliminated_system(rels, Ωₕ, f)
            Av = variational_matrix(rels, Ωₕ)
            @test norm(A - Av) <= tol * norm(Av)
            @test norm(A - A') <= tol * norm(A)
            # Right-hand side: ĥ f plus ω_F g_F/β_F once per incident facet.
            Fv = O.cell_measures(Ωₕ) .* f.(O.node_points(Ωₕ))
            LI = LinearIndices(O.sizes(Ωₕ))
            xs = O.axis_points(Ωₕ)
            for (ax, side, I) in O.ghosts(Ωₕ)
                c = getfield(rels, O.facet_symbol(ax, side))
                Fv[LI[I]] += O.facet_weight(Ωₕ, ax, I) * c.g(O.node_point(xs, I)) / c.β
            end
            @test norm(F - Fv) <= tol * norm(Fv)
            neumann && @test norm(A * ones(size(A, 2))) <= tol * norm(A)
        end
    end

    @testset "Augmented equals eliminated" begin
        for Ωₕ in meshes
            ns = O.sizes(Ωₕ)
            np = prod(ns)
            ng = O.ghost_count(ns)
            @test size(O.system_matrix(:robin_b, Ωₕ)) == (np + ng, np + ng)
            # A ghost row is ω_F times its facet relation.
            rels = O.problem_relations(:robin_b, Ωₕ)
            A, F = O.system(:robin_b, Ωₕ)
            L = O.dof_layout(Ωₕ)
            ax, side, I = O.ghosts(Ωₕ)[1]
            pc, pv, gr = O.relation_row(getfield(rels, :xmin), Ωₕ, ax, side, I)
            ω = O.facet_weight(Ωₕ, ax, I)
            want = zeros(np + ng)
            for (p, v) in zip(pc, pv)
                want[findfirst(==(p), L.kept)] = ω * v
            end
            @test Vector(A[np + 1, :]) ≈ want rtol=tol
            @test F[np + 1] ≈ ω * gr rtol=tol
            for p in (:robin, :neumann)
                ua = O.solve(Symbol(p, :_a), Ωₕ)
                ub = O.solve(Symbol(p, :_b), Ωₕ)
                @test length(ub) == np + ng
                @test norm(ua - ub[1:np]) <= 1e-10 * norm(ua)
                E, d = O.ghost_closure(O.problem_relations(Symbol(p, :_a), Ωₕ), Ωₕ)
                @test norm(ub[(np + 1):end] - (E * ua + d)[(np + 1):end]) <=
                      1e-10 * norm(ub)
            end
            @test dot(O.mean_weights(Ωₕ), O.solve(:neumann_a, Ωₕ)) ≈ 0 atol=tol
        end
        @test_throws ArgumentError O.system_matrix(:dirichlet, Ω2)
        @test_throws ArgumentError O.observed_order(:robin_a, :uniform; dim = 1)
        @test_throws ArgumentError O.make_mesh(:smooth, 0)
    end

    @testset "Orders" begin
        # Gated on uniform and graded meshes; refined random meshes printed. The errors
        # must also fall at every level and end small (the slope alone hides a blow-up).
        # The unit group runs the graded 2D meshes on three levels; `slow` runs every mesh
        # kind in 2D and 3D on four.
        levels = WITH_SLOW_TESTS ? 4 : 3
        dims = WITH_SLOW_TESTS ? (2, 3) : (2,)
        kinds = WITH_SLOW_TESTS ? O.MESH_KINDS : (:graded,)
        for dim in dims, kind in kinds, p in O.PROBLEMS
            hs, errs = O.observed_errors(p, kind; dim, levels)
            X, Y = log.(hs), log.(errs)
            X̄, Ȳ = sum(X) / levels, sum(Y) / levels
            r = sum((X .- X̄) .* (Y .- Ȳ)) / sum(abs2, X .- X̄)
            println("ND oracle $(dim)D $kind $p: order ", round(r; digits = 3),
                ", errors ", round.(errs; sigdigits = 3))
            @test all(errs[2:end] .< 0.5 .* errs[1:(end - 1)])
            @test errs[end] < 1e-2
            kind === :random || @test r > 1.8
        end
    end
end

end # module GhostOracleNdTests
