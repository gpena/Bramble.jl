module FormGhostConstrainedTests

# `constrained(Wₕ, u -> (:F => operator => data, …))` (gpena/Bramble.jl#205): the closure
# `u_pad = E u + d` its relations give, and the systems forms assemble on it.
#
# Every reference is a plain-matrix oracle that shares no code with the library:
# GhostOracle1D's and GhostOracleND's ghost closures and eliminated systems (Robin and
# Neumann relations solved ghost by ghost), GhostCorners' conormal relations (a 2×2 block per
# corner node) and its `:facet_onesided` closure, and, for `∂ₙₙ`, the oracle's `Δ_H` row at
# the facet node solved for the ghost here. Library closures have rows in padded linear
# order; the ND oracle's are in DOF order, so they are compared through the ghost space's
# selection. Random and graded meshes only; 3D where the group allows it (#650).

using Test
using Bramble
import Bramble
using LinearAlgebra: det, dot, norm, ×
using SparseArrays: SparseMatrixCSC, rowvals, sparse

using ..TestUtils: WITH_3D_TESTS

isdefined(Main, :GhostOracle1D) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracle1D.jl"))
isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
isdefined(Main, :GhostCorners) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostCorners.jl"))
const O1 = Main.GhostOracle1D
const ON = Main.GhostOracleND
const OC = Main.GhostCorners

const tol = 1e-12

relerr(A, B) = norm(A - B) / max(norm(B), eps())

# The padded index of each degree of freedom of the ghost space: its selection closure has
# one entry per column.
dof_rows(W) = rowvals(first(Bramble.closure(W)))

# The relations of an oracle's `FacetRobin`s (or 1D `Robin`s) as `operator => data` pairs.
robin_pairs(u, rels) = map(keys(rels)) do F
    r = rels[F]
    F => r.α * u + r.β * ∂ₙ(u) => r.g
end

# `-Δₕ u = f`, row-scaled by the cell measure, as the oracles' eliminated systems are.
poisson(C, f) = (form(C, C, (u, v) -> innerₕ(-1.0 * Δₕ(u), v)), form(C, v -> innerₕ(f, v)))

refusal(f) = try
    f()
    nothing
catch err
    err
end

# Whether `f()` throws an `ArgumentError` whose message holds `words`.
function refuses(f, words)
    e = refusal(f)
    return e isa ArgumentError && occursin(words, sprint(showerror, e))
end

@testset "1D Robin and Neumann vs oracle" begin
    for kind in (:random, :graded), problem in (:robin_a, :neumann_a)
        Ω = O1.make_mesh(kind, 0)
        rels = O1.problem_relations(problem, Ω)
        C = constrained(gridspace(Ω; ghosts = :boundary), u -> Tuple(robin_pairs(u, rels)))
        @test Bramble.ndofs(C) == length(points(Ω))
        E, d = Bramble.closure(C)
        E0, d0 = O1.ghost_closure(rels, Ω)
        @test relerr(E, E0) <= tol
        @test relerr(d, d0) <= tol
        a, l = poisson(C, x -> -O1.DEFAULT_EXACT.d2u(x[1]))
        A, F = assemble(a, l)
        A0, F0 = O1.system(problem, Ω)
        @test relerr(A, A0) <= tol
        @test relerr(F, F0) <= tol
    end
end

@testset "1D ∂ₙₙ relations vs the oracle's Δ_H" begin
    Ω = O1.make_mesh(:random, 0)
    n = length(points(Ω))
    Lap = O1.operators(Ω).Lap
    g = (xmin = 0.7, xmax = x -> 2.0 + x[1])
    gx = (xmin = 0.7, xmax = 2.0 + points(Ω)[n])
    for c in (0.0, 1.5)       # ∂ₙₙ(u) = g, and c u + ∂ₙₙ(u) = g
        C = constrained(gridspace(Ω; ghosts = :boundary),
            u -> (:xmin => c * u + ∂ₙₙ(u) => g.xmin, :xmax => c * u + ∂ₙₙ(u) => g.xmax))
        E, d = Bramble.closure(C)
        # The reference: the ghost of each end solves `c u_i + (Δ_H u)_i = g` (extended
        # rows ext(i) = i + 1), on the identity of the physical nodes.
        E0 = zeros(n + 2, n)
        d0 = zeros(n + 2)
        for i in 1:n
            E0[i + 1, i] = 1.0
        end
        for (i, k, gk) in ((1, 1, gx.xmin), (n, n + 2, gx.xmax))
            r = Vector(Lap[i + 1, :])
            r[i + 1] += c
            E0[k, :] = -r[2:(n + 1)] / r[k]
            d0[k] = gk / r[k]
        end
        @test relerr(E, E0) <= tol
        @test relerr(d, d0) <= tol
    end
end

@testset "2D Robin and Neumann vs ND oracle" begin
    for kind in (:random, :graded), problem in (:robin_a, :neumann_a)
        Ω = ON.make_mesh(kind, 0; n0 = 6)
        W = gridspace(Ω; ghosts = :boundary)
        rels = ON.problem_relations(problem, Ω)
        C = constrained(W, u -> Tuple(robin_pairs(u, rels)))
        E, d = Bramble.closure(C)
        E0, d0 = ON.ghost_closure(rels, Ω)
        rows = dof_rows(W)
        @test relerr(E[rows, :], E0) <= tol
        @test relerr(d[rows], d0) <= tol
        @test norm(E[setdiff(axes(E, 1), rows), :]) == 0     # absent points stay empty
        ex = ON.default_exact(2)
        A, F = assemble(poisson(C, x -> -ON.laplacian(ex, x))...)
        A0, F0 = ON.system(problem, Ω)
        @test relerr(A, A0) <= tol
        @test relerr(F, F0) <= tol
    end
end

@testset "Unrelated ghosted facets stay unknowns" begin
    Ω = ON.make_mesh(:random, 0; n0 = 6)
    nx, ny = ON.sizes(Ω)
    np = nx * ny
    W = gridspace(Ω; ghosts = (:xmin, :ymax))
    rels = ON.problem_relations(:robin_a, Ω)
    C = constrained(W, u -> :xmin => rels.xmin.α * u + rels.xmin.β * ∂ₙ(u) => rels.xmin.g)
    @test Bramble.ndofs(C) == np + nx               # the ymax ghosts remain
    E, d = Bramble.closure(C)
    Ew, _ = Bramble.closure(W)
    @test E[:, (np + 1):end] == Ew[:, (np + ny + 1):end]
    # The xmin ghosts as in the oracle's all-facet closure (DOF rows np+1 … np+ny), reading
    # physical nodes only.
    E0, d0 = ON.ghost_closure(rels, Ω)
    xg = dof_rows(W)[(np + 1):(np + ny)]
    @test relerr(E[xg, 1:np], E0[(np + 1):(np + ny), :]) <= tol
    @test norm(E[xg, (np + 1):end]) == 0
    @test relerr(d[xg], d0[(np + 1):(np + ny)]) <= tol
    @test count(!iszero, d) == ny
end

# The conormal relation of GhostCorners, `α_F u + β_F η_F (K ∇u)_d = g_F`, as pairs.
function conormal_pairs(u, K)
    ex, α, β = OC.DEFAULT_EXACT, OC.DEFAULT_α, OC.DEFAULT_β
    return ntuple(4) do F
        d, η = OC.facet_axis(F), OC.facet_normal(F)
        g = p -> begin
            ∇u = (ex.ux(p[1], p[2]), ex.uy(p[1], p[2]))
            α[F] * ex.u(p[1], p[2]) + β[F] * η * (K[d, 1] * ∇u[1] + K[d, 2] * ∇u[2])
        end
        OC.FACETS[F] => α[F] * u + β[F] * ∂ₙ(u; K) => g
    end
end

@testset "2D conormal corner blocks vs oracle" begin
    K = OC.DEFAULT_K
    Ω = OC.make_mesh(:random, 0; n0 = 7)
    nx, ny = OC.sizes(Ω)
    W = gridspace(Ω; ghosts = :boundary)
    C = constrained(W, u -> conormal_pairs(u, K))
    E, d = Bramble.closure(C)
    E0, d0 = OC.ghost_closure(Ω; closure = :facet_onesided)
    @test relerr(E, E0) <= tol
    @test relerr(d, d0) <= tol
    # The 2×2 blocks themselves, unweighted by the facet weight the forms put on each row.
    P = Bramble.padded_space(W)
    G = Bramble.mesh(P)
    rels = Bramble._facet_relations(G, conormal_pairs(Bramble.trial_function(P), K))
    R, _, ghosts = Bramble._relation_rows(P, rels)
    gpos = Dict(g => k for (k, g) in enumerate(ghosts))
    pts = LinearIndices(Bramble.npoints(G, Tuple))
    x, y = OC.axis_points(Ω, 1), OC.axis_points(Ω, 2)
    for ((i, j), B0) in zip(OC.corner_nodes(nx, ny), OC.corner_blocks(Ω))
        I = CartesianIndex(i, j)
        gx = Bramble.ghost_index(G, I, 1, i == 1 ? :min : :max)
        gy = Bramble.ghost_index(G, I, 2, j == 1 ? :min : :max)
        rws = [gpos[gx], gpos[gy]]
        cls = [pts[Bramble._ghost_point(G, gx)], pts[Bramble._ghost_point(G, gy)]]
        ω = [ON.facet_weight(Ω, 1, I), ON.facet_weight(Ω, 2, I)]
        B = Matrix(R[rws, cls]) ./ ω
        @test relerr(B, B0) <= tol
        hx = abs(x[i == 1 ? 2 : nx] - x[i == 1 ? 1 : nx - 1])
        hy = abs(y[j == 1 ? 2 : ny] - y[j == 1 ? 1 : ny - 1])
        βx, βy = OC.DEFAULT_β[i == 1 ? 1 : 2], OC.DEFAULT_β[j == 1 ? 3 : 4]
        @test abs(det(B)) ≈ βx * βy * det(K) / (4hx * hy) rtol = 1e-12
    end
end

@testset "Float32 spaces keep their element type" begin
    Ω = mesh(domain(interval(0.0f0, 1.0f0)), 9, false)
    C = constrained(gridspace(Ω; ghosts = :boundary),
        u -> (:xmin => ∂ₙ(u) => 1, :xmax => u + 2 * ∂ₙ(u) => x -> x[1]))
    E, d = Bramble.closure(C)
    @test eltype(E) === Float32 && eltype(d) === Float32
    @test eltype(C) === Float32
end

@testset "Refusals name the fix" begin
    Ω = ON.make_mesh(:random, 0; n0 = 5)
    W = gridspace(Ω; ghosts = :boundary)
    Wx = gridspace(Ω; ghosts = (:xmin,))
    @test_throws ArgumentError constrained(gridspace(Ω), u -> :xmin => ∂ₙ(u) => 0.0)
    @test refuses(() -> constrained(gridspace(Ω), u -> :xmin => ∂ₙ(u) => 0.0),
        "space with ghosts")
    C = constrained(Wx, u -> :xmin => ∂ₙ(u) => 0.0)
    @test refuses(() -> constrained(C, u -> :xmin => ∂ₙ(u) => 0.0), "already")
    @test refuses(() -> constrained(Wx, u -> :xmax => ∂ₙ(u) => 0.0), "carries no ghosts")
    @test refuses(() -> constrained(W, u -> (:xmin => ∂ₙ(u) => 0.0, :left => u + ∂ₙ(u) => 1.0)),
        "two relations")
    @test refuses(() -> constrained(W, u -> :foo => ∂ₙ(u) => 0.0), "does not name a facet")
    @test refuses(() -> constrained(W, u -> :xmin => 2.0 * u + 0.0 * ∂ₙ(u) => 1.0),
        "does not read the ghost")
    @test refuses(() -> constrained(W, u -> :ymax => jumpₙ(u) => 1.0), "does not read the ghost")
    @test refuses(() -> constrained(W, u -> :xmin => 2.0 * u => 1.0), "reads no ghost")
    @test refuses(() -> constrained(W, u -> :xmin => ∂ₙ(u) => [1.0, 2.0]), "number or a function")
    @test refuses(() -> constrained(W, u -> 1.0), "pairs")
    @test refuses(() -> constrained(W, u -> :xmin => ∂ₙ(u)), "pairs")
    Ks = [1.0 1.0; 1.0 1.0]
    @test refuses(() -> constrained(W, u -> map(F -> F => ∂ₙ(u; K = Ks) => 0.0,
            (:xmin, :xmax, :ymin, :ymax))), "singular")
    @test_throws ArgumentError constrained(Wx, u -> :xmin => ∂ₙ(u; K = Ks) => 0.0)
end

# Nightly only: 3D meshes are not built under `unit` (#650).
WITH_3D_TESTS && @testset "3D Robin vs ND oracle" begin
    Ω = ON.make_mesh(:random, 0; dim = 3, n0 = 4)
    W = gridspace(Ω; ghosts = :boundary)
    rels = ON.problem_relations(:robin_a, Ω)
    C = constrained(W, u -> Tuple(robin_pairs(u, rels)))
    E, d = Bramble.closure(C)
    E0, d0 = ON.ghost_closure(rels, Ω)
    rows = dof_rows(W)
    @test relerr(E[rows, :], E0) <= tol
    @test relerr(d[rows], d0) <= tol
    ex = ON.default_exact(3)
    A, F = assemble(poisson(C, x -> -ON.laplacian(ex, x))...)
    A0, F0 = ON.system(:robin_a, Ω)
    @test relerr(A, A0) <= tol
    @test relerr(F, F0) <= tol
end

end # module FormGhostConstrainedTests
