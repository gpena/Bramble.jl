module FormGhostDirichletTests

# Dirichlet conditions on ghost spaces (gpena/Bramble.jl#205), both ways: as a relation
# `:F => u => g` in `constrained`, which removes the facet's nodes from the unknowns and puts
# `g` in the closure's `d`, and as `assemble(a, l; dirichlet, symmetrize)`, which replaces
# rows of the reduced system when its unknowns are the physical nodes.
#
# The references share no code with the library: GhostOracle1D's closure and eliminated
# system with a `Dirichlet` relation, and in 2D a closure built here from GhostOracleND's
# relation rows and padded `Δ_H`, the fixed nodes lifted into the load. Random and graded
# meshes only.

using Test
using Bramble
import Bramble
using LinearAlgebra: Diagonal, dot, norm
using SparseArrays: rowvals, sparse

isdefined(Main, :GhostOracle1D) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracle1D.jl"))
isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
const O1 = Main.GhostOracle1D
const ON = Main.GhostOracleND

const tol = 1e-12

relerr(A, B) = norm(A - B) / max(norm(B), eps())

lap(u, v) = innerₕ(-1.0 * Δₕ(u), v)

refusal(f) =
    try
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

@testset "1D u => g vs oracle" begin
    for kind in (:random, :graded)
        Ω = O1.make_mesh(kind, 0)
        n = length(points(Ω))
        x = collect(points(Ω))
        f = x -> -O1.DEFAULT_EXACT.d2u(x[1])
        rels = O1.problem_relations(:robin_a, Ω)
        gD = O1.DEFAULT_EXACT.u(x[1])
        r = rels.xmax
        W = gridspace(Ω; ghosts = (:xmax,))
        for data in (gD, p -> O1.DEFAULT_EXACT.u(p[1]))       # a number and a function
            C = constrained(W, u -> (:xmin => u => data, :xmax => r.α * u + r.β * ∂ₙ(u) => r.g))
            @test Bramble.ndofs(C) == n - 1
            E, d = Bramble.closure(C)
            E0, d0 = O1.ghost_closure((xmin = O1.Dirichlet(gD), xmax = r), Ω)
            @test relerr(E, E0[2:end, :]) <= tol     # padded rows: no xmin ghost
            @test relerr(d, d0[2:end]) <= tol
            @test isempty(rowvals(E[1:1, :]))        # the fixed node is no unknown
            A, F = assemble(form(C, C, lap), form(C, v -> innerₕ(f, v)))
            A0, F0 = O1.eliminated_system((xmin = O1.Dirichlet(gD), xmax = r), Ω, f)
            @test relerr(A, A0) <= tol
            @test relerr(F, F0) <= tol
            @test norm(A - A') <= 1e-14 * norm(A)
        end
    end
end

@testset "dirichlet keyword: matrix and refill" begin
    # The two routes' systems and solutions agree in test/form/two_spaces.jl; the matrix
    # alone, its refill and `dirichlet_bc!` by hand give the same rows as the pair.
    Ω = O1.make_mesh(:random, 0)
    x = collect(points(Ω))
    f = x -> -O1.DEFAULT_EXACT.d2u(x[1])
    gD = O1.DEFAULT_EXACT.u(x[end])
    r = O1.problem_relations(:robin_a, Ω).xmin
    W = gridspace(Ω; ghosts = (:xmin,))
    Ca = constrained(W, u -> :xmin => r.α * u + r.β * ∂ₙ(u) => r.g)
    a, l = form(Ca, Ca, lap), form(Ca, v -> innerₕ(f, v))
    A1, _ = assemble(a, l; dirichlet = :xmax => (p -> gD))
    A2 = assemble(a; dirichlet = :xmax)
    @test A2 == A1
    assemble!(A2, a; dirichlet = :xmax)
    @test A2 == A1
    A3 = dirichlet_bc!(assemble(a), Ca, :xmax)
    @test A3 == A1
end

@testset "u => g beside a ghost relation" begin
    # `u => g` fixes the node at xmax and `∂ₙ(u) => h` still solves its ghost, the fixed
    # node's term moved into the data.
    Ω = O1.make_mesh(:graded, 0)
    n = length(points(Ω))
    Wx = gridspace(Ω; ghosts = (:xmax,))
    C = constrained(Wx, u -> (:xmax => u => 0.7, :xmax => ∂ₙ(u) => 1.3))
    @test Bramble.ndofs(C) == n - 1
    E, d = Bramble.closure(C)
    @test d[n] == 0.7
    rel, h = O1.relation_row(O1.Neumann(1.0, 1.3), Ω, :xmax)
    for z in (zeros(n - 1), collect(range(-1.0, 2.0; length = n - 1)))
        uext = vcat(0.0, E * z + d)                   # no xmin ghost: 0, never read
        @test dot(rel, uext) ≈ h rtol = 1e-13
    end
    # A ghosted facet with only `u => g` would keep a ghost no equation reads: refused.
    Wb = gridspace(Ω; ghosts = :boundary)
    rels = u -> (:xmin => u => 1.0, :xmax => ∂ₙ(u) => 0.0)
    @test_throws ArgumentError constrained(Wb, rels)
    @test refuses(() -> constrained(Wb, rels), "Drop :xmin")
end

# −Δu = f with u fixed to `g` on the facets `fixed` and the relations `rels` on the others:
# each ghost of a related facet solved from GhostOracleND's row at its node, the fixed
# nodes in the affine part; rows and unknowns are the free nodes in mesh order.
function reference_system(Ω, rels, fixed, g, f)
    L = ON.dof_layout(Ω)
    pts = ON.node_points(Ω)
    nodes = vec(collect(CartesianIndices(L.ns)))
    on(I, F) = (d = (findfirst(==(F), ON.FACETS) + 1) ÷ 2;
        I[d] == (iseven(findfirst(==(F), ON.FACETS)) ? L.ns[d] : 1))
    isfixed = [any(F -> on(I, F), fixed) for I in nodes]
    free = findall(!, isfixed)
    E = zeros(prod(L.ps), length(free))
    dv = zeros(prod(L.ps))
    for (c, k) in enumerate(free)
        E[L.phys[k], c] = 1.0
    end
    for k in findall(isfixed)
        dv[L.phys[k]] = g(pts[k])
    end
    for (k, (ax, side, I)) in enumerate(L.ghosts)
        F = ON.facet_symbol(ax, side)
        haskey(rels, F) || continue
        cols, vals, gr = ON.relation_row(rels[F], Ω, ax, side, I)
        p0 = L.ghost[k]
        cg = vals[findfirst(==(p0), cols)]
        for (p, v) in zip(cols, vals)
            p == p0 && continue
            E[p0, :] .-= (v / cg) .* E[p, :]
            dv[p0] -= v / cg * dv[p]
        end
        dv[p0] += gr / cg
    end
    Lap = ON.operators(Ω).Lap[L.phys[free], :]
    Wf = Diagonal(ON.cell_measures(Ω)[free])
    return sparse(Wf * (-Lap * E)), Wf * (f.(pts[free]) .+ Lap * dv), free
end

@testset "2D Dirichlet and Robin, both routes" begin
    ex = ON.default_exact(2)
    g = x -> ON.value(ex, x)
    f = x -> -ON.laplacian(ex, x)
    for (kind, problem) in ((:random, :robin_a), (:graded, :neumann_a))
        Ω = ON.make_mesh(kind, 0; n0 = 6)
        rels = ON.problem_relations(problem, Ω)
        W = gridspace(Ω; ghosts = (:xmax, :ymin))
        pairs(u) = map(F -> F => rels[F].α * u + rels[F].β * ∂ₙ(u) => rels[F].g, (:xmax, :ymin))
        # route (b): the corner node (1, ny) lies on both Dirichlet facets
        Cb = constrained(W, u -> (:xmin => u => g, :ymax => u => g, pairs(u)...))
        Ab, Fb = assemble(form(Cb, Cb, lap), form(Cb, v -> innerₕ(f, v)))
        A0, F0, free = reference_system(Ω, (xmax = rels.xmax, ymin = rels.ymin),
            (:xmin, :ymax), g, f)
        @test Bramble.ndofs(Cb) == length(free)
        @test relerr(Ab, A0) <= tol
        @test relerr(Fb, F0) <= tol
        @test norm(Ab - Ab') <= 1e-14 * norm(Ab)
        ub = Ab \ Fb
        # route (a): the physical nodes are the unknowns, the Dirichlet rows replaced
        Ca = constrained(W, u -> pairs(u))
        Aa, Fa = assemble(form(Ca, Ca, lap), form(Ca, v -> innerₕ(f, v));
            dirichlet = (:xmin => g, :ymax => g), symmetrize = true)
        @test norm(Aa - Aa') <= 1e-14 * norm(Aa)
        ua = Aa \ Fa
        pts = ON.node_points(Ω)
        fixed = setdiff(eachindex(pts), free)
        @test relerr(ua[free], ub) <= 1e-10
        @test relerr(ua[fixed], g.(pts[fixed])) <= 1e-14
    end
    # Data as numbers, equal at the corner the two facets share.
    Ω = ON.make_mesh(:random, 0; n0 = 5)
    C = constrained(gridspace(Ω; ghosts = (:xmax,)),
        u -> (:xmin => u => 2, :ymin => u => x -> 2.0, :ymax => u => 2.0, :xmax => ∂ₙ(u) => 0.0))
    nx, ny = ON.sizes(Ω)
    @test Bramble.ndofs(C) == (nx - 1) * (ny - 2)
end

@testset "Lifted solution holds g" begin
    ex = ON.default_exact(2)
    g = x -> ON.value(ex, x)
    Ω = ON.make_mesh(:graded, 0; n0 = 6)
    rels = ON.problem_relations(:robin_a, Ω)
    W = gridspace(Ω; ghosts = (:xmax,))
    Cb = constrained(W,
        u -> (:xmin => u => g, :ymin => u => g, :ymax => u => g,
            :xmax => rels.xmax.α * u + rels.xmax.β * ∂ₙ(u) => rels.xmax.g))
    a, l = form(Cb, Cb, lap), form(Cb, v -> innerₕ(x -> -ON.laplacian(ex, x), v))
    ub = \(assemble(a, l)...)
    uh = parent(Bramble.lift(Cb, ub))
    G = Bramble.mesh(Cb)
    plin = LinearIndices(Bramble.npoints(G, Tuple))
    nx, ny = ON.sizes(Ω)
    pts = ON.axis_points(Ω)
    for I in CartesianIndices((nx, ny))
        I[1] == 1 || I[2] == 1 || I[2] == ny || continue
        @test uh[plin[Bramble.padded_index(G, I)]] ≈ g((pts[1][I[1]], pts[2][I[2]])) rtol = 1e-14
    end
end

@testset "Refusals name the fix" begin
    Ω = ON.make_mesh(:random, 0; n0 = 5)
    W = gridspace(Ω; ghosts = (:xmax,))
    rob = u -> u + ∂ₙ(u)
    # given both ways
    Cb = constrained(W, u -> (:xmin => u => 1.0, :xmax => rob(u) => 0.0))
    ab, lb = form(Cb, Cb, lap), form(Cb, v -> innerₕ(x -> 1.0, v))
    g1 = x -> 1.0
    @test_throws ArgumentError assemble(ab, lb; dirichlet = :xmin => g1)
    @test refuses(() -> assemble(ab, lb; dirichlet = :xmin => g1), "given both ways")
    @test refuses(() -> assemble(ab; dirichlet = :left), "given both ways")   # an alias
    # the two routes on different facets
    @test refuses(() -> assemble(ab, lb; dirichlet = :ymax => g1), "given both ways") # corner
    @test refuses(() -> assemble(ab, lb; dirichlet = :xmax => g1), "or every one")
    # ghosts left unknown
    aw = form(W, W, lap)
    @test refuses(() -> assemble(aw; dirichlet = :xmin), "keeps ghosts")
    @test refuses(() -> dirichlet_bc!(assemble(aw), W, :xmin), "keeps ghosts")
    # disagreeing at a corner node
    @test refuses(() -> constrained(W, u -> (:xmin => u => 0.0, :ymax => u => 1.0)),
        "values")
    # agreement is judged on the two values at the node, not on the largest datum
    @test refuses(() -> constrained(W, u -> (:xmin => u => (x -> 1.0 + 1e6 * x[2]),
            :ymin => u => 1.001)), "values")
    @test refuses(() -> constrained(W, u -> (:xmin => u => 1e-20, :ymin => u => 0.0)),
        "values")
    @test Bramble.ndofs(constrained(W, u -> (:xmin => u => 1e-20, :ymin => u => 1e-20))) ==
          Bramble.ndofs(W) - (sum(ON.sizes(Ω)) - 1)
    # a rounding zero meets an exact one: sin(π x) at x = 1 is 1.2e-16
    @test Bramble.ndofs(constrained(W, u -> (:ymin => u => (x -> sin(π * x[1])),
        :xmax => u => 0.0, :xmax => ∂ₙ(u) => 0.0))) ==
          Bramble.ndofs(W) - (sum(ON.sizes(Ω)) - 1) - ON.sizes(Ω)[2]
    @test refuses(() -> constrained(W, u -> (:xmin => u => 0.0, :left => u => 0.0)),
        "two Dirichlet relations")
    @test refuses(() -> constrained(W, u -> (:xmax => rob(u) => 0.0, :xmax => ∂ₙ(u) => 0.0)),
        "two relations reading")
    # a ghost relation still needs the facet's ghosts; `u => g` does not
    @test refuses(() -> constrained(W, u -> :ymin => ∂ₙ(u) => 0.0), "carries no ghosts")
    Ca = constrained(W, u -> :xmax => rob(u) => 0.0)
    @test refuses(
        () -> assemble(form(Ca, Ca, lap), form(Ca, v -> innerₕ(x -> 1.0, v));
            dirichlet = :xmin, symmetrize = true),
        "carries no values")
    @test refuses(
        () -> assemble(form(Ca, Ca, lap), form(Ca, v -> innerₕ(x -> 1.0, v));
            symmetrize = true), "nothing to symmetrize")
end

end # module FormGhostDirichletTests
