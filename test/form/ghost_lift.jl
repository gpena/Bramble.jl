module FormGhostLiftTests

# `lift(Cₕ)` and `lift(Cₕ, u)` (gpena/Bramble.jl#205): the data `d` and the solution
# `E u + d` of a constrained space as grid functions on its padded space.
#
# Every reference is a plain-matrix oracle that shares no code with the library:
# GhostOracle1D's and GhostOracleND's ghost closures, their `Dc` relation rows and their
# `Δ_H` on the padded grid. The 1D oracle's extended order is the padded linear order; the
# ND oracle's closure has rows in DOF order, compared through `dof_layout`'s padded indices.
# Random meshes only.

using Test
using Bramble
import Bramble
using LinearAlgebra: dot, norm
using SparseArrays: rowvals

isdefined(Main, :GhostOracle1D) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracle1D.jl"))
isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
const O1 = Main.GhostOracle1D
const ON = Main.GhostOracleND

const tol = 1e-12

relerr(a, b) = norm(a - b) / max(norm(b), eps())

robin_pairs(u, rels) = map(keys(rels)) do F
    r = rels[F]
    F => r.α * u + r.β * ∂ₙ(u) => r.g
end

laplacian_form(C) = form(C, C, (u, v) -> innerₕ(-1.0 * Δₕ(u), v))
zero_load(C) = form(C, v -> innerₕ(x -> 0.0, v))

# The form-side `Δₕ(E u + d)` at the physical node of DOF `k`, weighted by `innerₕ`:
# with a zero load, `assemble(a, l)` returns `F = -Eᵀ A_pad d`, so `A u - F` is
# `Eᵀ A_pad (E u + d)`, and `A_pad` has no rows at ghosts.
function weighted_form_laplacian(C, u)
    A, F = assemble(form(C, C, (w, v) -> innerₕ(Δₕ(w), v)), zero_load(C))
    return A * u - F
end

function refusal(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end

refuses(f, words) = (e = refusal(f); e isa ArgumentError && occursin(words, sprint(showerror, e)))

@testset "1D Robin lift vs oracle" begin
    Ω = O1.make_mesh(:random, 1)
    n = length(points(Ω))
    ops = O1.operators(Ω)
    rels = O1.problem_relations(:robin_a, Ω)
    f = x -> -O1.DEFAULT_EXACT.d2u(x[1])
    C = constrained(gridspace(Ω; ghosts = :boundary), u -> robin_pairs(u, rels))
    Eo, do_ = O1.ghost_closure(rels, Ω)

    L0 = Bramble.lift(C)
    @test L0 isa Bramble.VectorElement
    @test Bramble.space(L0) === Bramble.padded_space(C)
    @test relerr(parent(L0), do_) <= tol
    @test count(!iszero, parent(L0)) == 2

    A, F = assemble(laplacian_form(C), form(C, v -> innerₕ(f, v)))
    u = A \ F
    U = parent(Bramble.lift(C, u))
    @test length(U) == n + 2
    @test relerr(U, Eo * u + do_) <= tol
    @test U[2:(n + 1)] == u

    for (i, rel, η) in ((2, rels.xmin, -1.0), (n + 1, rels.xmax, 1.0))
        @test abs(rel.α * U[i] + rel.β * η * dot(ops.Dc[i, :], U) - rel.g) <= 1e-10
    end
end

@testset "1D lift copies d and refuses bad input" begin
    Ω = O1.make_mesh(:random, 1)
    rels = O1.problem_relations(:robin_a, Ω)
    C = constrained(gridspace(Ω; ghosts = :boundary), u -> robin_pairs(u, rels))
    E, d = Bramble.closure(C)
    d0 = copy(d)
    L0 = Bramble.lift(C)
    parent(L0) .= 7.0
    @test Bramble.closure(C)[2] == d0

    @test refuses(() -> Bramble.lift(C, ones(Bramble.ndofs(C) + 1)), "degree of freedom")
    @test refuses(() -> Bramble.lift(C, ones(length(d))), "padded")
    @test refuses(() -> Bramble.lift(C, 1.0), "solution vector")
    @test refuses(() -> Bramble.lift(gridspace(Ω)), "gridspace(Ωₕ; ghosts")
    @test refuses(() -> Bramble.lift(gridspace(Ω), ones(length(points(Ω)))), "with ghosts")
end

@testset "1D lift refuses a stale space" begin
    Ω = O1.make_mesh(:random, 1)
    rels = O1.problem_relations(:robin_a, Ω)
    C = constrained(gridspace(Ω; ghosts = :boundary), u -> robin_pairs(u, rels))
    u = ones(Bramble.ndofs(C))
    Bramble.change_points!(Ω, collect(points(Ω)) .^ 1.5)
    @test refuses(() -> Bramble.lift(C, u), "gridspace(Ωₕ; ghosts = ...) again")
    @test refuses(() -> Bramble.lift(C), "gridspace(Ωₕ; ghosts = ...) again")
end

@testset "1D lift refuses extra arguments" begin
    W = gridspace(O1.make_mesh(:random, 1); ghosts = :boundary)
    u = zeros(Bramble.ndofs(W))
    @test refuses(() -> Bramble.lift(W, u, 1), "at most one vector")
    @test !refuses(() -> Bramble.lift(W, u, 1), "with ghosts")
end

@testset "1D lift of an unconstrained space" begin
    Ω = O1.make_mesh(:random, 1)
    W = gridspace(Ω; ghosts = :boundary)
    m = Bramble.ndofs(W)
    u = collect(1.0:m)
    U = parent(Bramble.lift(W, u))
    rows = rowvals(first(Bramble.closure(W)))
    @test U[rows] == u
    @test iszero(parent(Bramble.lift(W)))
    @test U[1] == u[m - 1] && U[end] == u[m]
end

@testset "1D runtime Δₕ(lift) at facet nodes" begin
    Ω = O1.make_mesh(:random, 1)
    n = length(points(Ω))
    Lap = O1.operators(Ω).Lap
    ĥ = O1.cell_measures(Ω)
    rels = O1.problem_relations(:robin_a, Ω)
    f = x -> -O1.DEFAULT_EXACT.d2u(x[1])
    C = constrained(gridspace(Ω; ghosts = :boundary), u -> robin_pairs(u, rels))
    A, F = assemble(laplacian_form(C), form(C, v -> innerₕ(f, v)))
    u = A \ F
    Uₕ = Bramble.lift(C, u)
    U = parent(Uₕ)
    ΔU = parent(Δₕ(Uₕ))
    wform = weighted_form_laplacian(C, u)
    for (i, k) in ((2, 1), (n + 1, n))
        @test abs(ΔU[i] - dot(Lap[i, :], U)) <= tol * (1 + abs(ΔU[i]))
        @test abs(ĥ[k] * ΔU[i] - wform[k]) <= tol * (1 + abs(wform[k]))
    end
    # The ghosts move the facet values: the same runtime operator on the padded vector
    # with its ghosts zeroed gives a different Δₕ there.
    Z = copy(U)
    Z[1] = Z[end] = 0.0
    ΔZ = parent(Δₕ(Bramble.VectorElement(Z, Bramble.padded_space(C))))
    @test abs(ΔZ[2] - ΔU[2]) > 1e-3 * abs(ΔU[2])
end

@testset "1D lift as data in a linear form" begin
    Ω = O1.make_mesh(:random, 1)
    n = length(points(Ω))
    ops = O1.operators(Ω)
    rels = O1.problem_relations(:robin_a, Ω)
    C = constrained(gridspace(Ω; ghosts = :boundary), u -> robin_pairs(u, rels))
    Eo, do_ = O1.ghost_closure(rels, Ω)
    a = laplacian_form(C)
    l = form(C, v -> inner_Γ(Δₕ(Bramble.lift(C)), ∂ₙ(v)))
    _, Fl = assemble(a, l)
    _, F0 = assemble(a, zero_load(C))
    g = ops.Lap * do_
    expected = zeros(n + 2)
    for (i, η) in ((2, -1.0), (n + 1, 1.0))
        expected .+= g[i] * η * Vector(ops.Dc[i, :])
    end
    @test relerr(Fl - F0, Eo' * expected) <= tol
    @test norm(Fl - F0) > 0
end

@testset "2D Robin lift vs ND oracle" begin
    Ω = ON.make_mesh(:random, 1; dim = 2)
    rels = ON.problem_relations(:robin_a, Ω)
    exact = ON.default_exact(2)
    f = x -> -ON.laplacian(exact, x)
    Lo = ON.dof_layout(Ω)
    Eo, do_ = ON.ghost_closure(rels, Ω)
    C = constrained(gridspace(Ω; ghosts = :boundary), u -> robin_pairs(u, rels))
    @test Bramble.ndofs(C) == length(Lo.phys)

    L0 = parent(Bramble.lift(C))
    @test length(L0) == prod(Lo.ps)
    @test relerr(L0[Lo.kept], do_) <= tol
    @test all(iszero, L0[Lo.absent])

    A, F = assemble(laplacian_form(C), form(C, v -> innerₕ(f, v)))
    u = A \ F
    Uₕ = Bramble.lift(C, u)
    U = parent(Uₕ)
    @test relerr(U[Lo.kept], Eo * u + do_) <= tol
    @test all(iszero, U[Lo.absent])
    for (ax, side, I) in Lo.ghosts
        cols, vals, g = ON.relation_row(rels[ON.facet_symbol(ax, side)], Ω, ax, side, I)
        @test abs(dot(vals, U[cols]) - g) <= 1e-10 * (1 + abs(g))
    end

    # Runtime Δₕ at every facet node, corner nodes included (each reads its two ghosts).
    Lap = ON.operators(Ω).Lap
    ĥ = ON.cell_measures(Ω)
    ΔU = parent(Δₕ(Uₕ))
    wform = weighted_form_laplacian(C, u)
    facet = [k for (k, I) in enumerate(CartesianIndices(Lo.ns))
             if any(d -> I[d] == 1 || I[d] == Lo.ns[d], 1:2)]
    @test length(facet) == 2 * sum(Lo.ns) - 4
    p = Lo.phys[facet]
    @test relerr(ΔU[p], Lap[p, :] * U) <= tol
    @test relerr(ĥ[facet] .* ΔU[p], wform[facet]) <= tol
end

end # module FormGhostLiftTests
