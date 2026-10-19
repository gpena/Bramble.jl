module GhostCrossComponentTests

# Ghost relations on a composite of ghost spaces constrained as a whole,
# `constrained(S × S, u -> (… u(1) …, … u(2) …))`. With relations of one component each,
# the result must equal the product of the separately constrained components,
# `constrained(S, r₁) × constrained(S, r₂)`, whose closure is block diagonal (#781): `E`,
# `d`, the assembled matrix and load, and `lift`. Every mesh is non-uniform, and the
# components' relations and data differ, so a swapped or shared component shows. The
# refusals (data in time, the `dirichlet` rows, `GhostVector`, relations naming no
# component or several) are pinned by their messages.

using Test
using Bramble
import Bramble
using Bramble: lift, lift!
using LinearAlgebra: dot, norm
using Random: Xoshiro

isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
const O = Main.GhostOracleND

# Refused with an `ArgumentError` whose message holds `needle`: the message pins which
# refusal fired.
function refuses(f, needle)
    try
        f()
    catch e
        return e isa ArgumentError && occursin(needle, sprint(showerror, e))
    end
    return false
end

close(a, b; rtol = 1.0e-12) = norm(a - b) <= rtol * max(norm(b), 1)

# Two relation sets per ghosting: `:sides` ghosts the x facets only and fixes the y facets
# with `u => g`; `:boundary` ghosts every facet and closes the corners with conormal
# relations, whose 2×2 corner blocks the closure solves together.
const K1 = [2.0 0.5; 0.5 1.0]
const K2 = [1.0 -0.25; -0.25 3.0]
const CASES = (
    sides = (ghosts = (:xmin, :xmax),
        r1 = u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => 0.0,
            :ymin => u => 0.0, :ymax => u => x -> x[1]),
        r2 = u -> (:xmin => 2.0 * u + ∂ₙ(u) => 50.0, :xmax => ∂ₙ(u) => x -> 100.0 * x[2],
            :ymin => u => 100.0, :ymax => u => x -> 100.0 + x[1]^2)),
    boundary = (ghosts = :boundary,
        r1 = u -> (:xmin => ∂ₙ(u; K = K1) => 1.0, :xmax => ∂ₙ(u; K = K1) => x -> x[2],
            :ymin => u + ∂ₙ(u; K = K1) => 0.0, :ymax => ∂ₙ(u; K = K1) => 2.0),
        r2 = u -> (:xmin => ∂ₙ(u; K = K2) => 100.0, :xmax => ∂ₙ(u; K = K2) => 0.0,
            :ymin => ∂ₙ(u; K = K2) => x -> 100.0 * x[1],
            :ymax => 3.0 * u + ∂ₙ(u; K = K2) => -50.0)))

const KINDS = (:random, :graded)

whole(r1, r2) = u -> (r1(u(1))..., r2(u(2))...)

# The product of the separately constrained components, the reference side of a form.
X2(S, c) = constrained(S, c.r1) × constrained(S, c.r2)

@testset "Cross-component ghost relations" begin
    fa = (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + 100.0 * inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) +
                   0.5 * innerₕ(u(1), v(2)) + innerₕ(u(2), v(1))
    fl = v -> innerₕ(x -> 1.0, v(1)) + innerₕ(x -> 100.0 * x[2], v(2))

    @testset "Whole equals product: $kind, $name" for kind in KINDS,
        name in keys(CASES)

        c = CASES[name]
        Ωₕ = O.make_mesh(kind, 0; dim = 2, n0 = 7)
        S = gridspace(Ωₕ; ghosts = c.ghosts)
        C = constrained(S × S, whole(c.r1, c.r2))
        X = X2(S, c)
        @test C isa Bramble.ConstrainedCompositeSpace{2}
        @test Bramble.ndofs(C) == Bramble.ndofs(X)

        E, d = Bramble.closure(C)
        Ex, dx = Bramble.closure(X)
        @test size(E) == size(Ex)
        @test close(Matrix(E), Matrix(Ex); rtol = 1.0e-13)
        @test close(d, dx; rtol = 1.0e-13)
        # the data of the two components differ, so a swapped component shows in `d`
        m = Bramble.ndofs(Bramble.padded_space(S))
        @test norm(d[(m + 1):end]) > 10 * norm(d[1:m])

        a, l = form(C, C, fa), form(C, fl)
        A, F = assemble(a, l)
        Ax, Fx = assemble(form(X, X, fa), form(X, fl))
        @test close(Matrix(A), Matrix(Ax))
        @test close(F, Fx)
        @test close(Matrix(assemble(a)), Matrix(Ax))
        # either kind of side pairs with the other
        @test close(Matrix(assemble(form(C, X, fa))), Matrix(Ax))
        assemble!(A, F, a, l)
        @test close(Matrix(A), Matrix(Ax))
        @test close(F, Fx)

        rng = Xoshiro(20261019)
        u, v = rand(rng, Bramble.ndofs(C)), rand(rng, Bramble.ndofs(C))
        @test a(element(C, u), element(C, v)) ≈ dot(v, Ax * u) rtol = 1.0e-12

        uₕ = lift(C, u)
        @test uₕ.space === Bramble.padded_space(C)
        @test close(parent(uₕ), E * u + d; rtol = 1.0e-13)
        @test close(parent(uₕ), parent(lift(X, u)); rtol = 1.0e-13)
        @test parent(lift(C)) == d
        @test parent(lift(C, u, 0.5)) == parent(uₕ)
        w = 2.0 .* u
        lift!(uₕ, C, w)
        @test close(parent(uₕ), E * w + d; rtol = 1.0e-13)
        lift!(uₕ, C, u, 1.0)
        @test close(parent(uₕ), E * u + d; rtol = 1.0e-13)

        # the solution of the whole equals the product's
        @test close(A \ F, Ax \ Fx; rtol = 1.0e-10)
    end

    @testset "Refusals on a whole composite" begin
        Ωₕ = O.make_mesh(:random, 0; dim = 2, n0 = 7)
        S = gridspace(Ωₕ; ghosts = (:xmin, :xmax))
        c = CASES.sides
        V = S × S
        C = constrained(V, whole(c.r1, c.r2))
        a, l = form(C, C, fa), form(C, fl)

        # data in time: none on a composite as a whole
        @test refuses(
            () -> constrained(V, u -> (:xmin => ∂ₙ(u(1)) => (x, t) -> t,
                :xmax => ∂ₙ(u(1)) => 0.0)), "depend on time")
        @test refuses(
            () -> constrained(V, u -> (:xmin => ∂ₙ(u(1)) => 0.0,
                :ymin => u(2) => (x, t) -> t)), "depend on time")
        # a relation names exactly one component
        @test refuses(() -> constrained(V, u -> (:xmin => ∂ₙ(u) => 0.0,)),
            "names no component")
        @test refuses(() -> constrained(V, u -> (:ymin => u => 0.0,)), "names no component")
        @test refuses(() -> constrained(V, u -> (:xmin => ∂ₙ(u(1)) + ∂ₙ(u(2)) => 0.0,)),
            "reads several components")
        # one relation of each kind per component and facet; both components may share one
        @test refuses(() -> constrained(V, u -> (:xmin => ∂ₙ(u(1)) => 0.0,
                :xmin => ∂ₙₙ(u(1)) => 0.0)),
            ":xmin of component 1 has two relations")
        @test refuses(
            () -> constrained(V, u -> (:ymin => u(2) => 0.0,
                :ymin => u(2) => 1.0)), ":ymin of component 2 has two Dirichlet")
        @test Bramble.ndofs(constrained(V, u -> (:xmin => ∂ₙ(u(1)) => 0.0,
            :xmin => ∂ₙ(u(2)) => 0.0))) == Bramble.ndofs(V) - 2 * npoints(Ωₕ, Tuple)[2]
        # a ghost relation on an unghosted facet, and a ghosted Dirichlet facet left open
        @test refuses(() -> constrained(V, u -> (:ymin => ∂ₙ(u(2)) => 0.0,)),
            "carries no ghosts")
        @test refuses(() -> constrained(V, u -> (:xmin => u(1) => 0.0,)),
            "carries ghosts and only the Dirichlet relation")
        # constrained once, on unconstrained components
        @test refuses(() -> constrained(C, whole(c.r1, c.r2)), "already constrained")
        @test refuses(
            () -> constrained(constrained(S, c.r1) × S, u -> (:xmin => ∂ₙ(u(2)) => 0.0,)),
            "already constrained")

        # the rows route: `dirichlet`, `dirichlet_bc!` and `symmetrize!`
        rows = "does not take them"
        @test refuses(() -> assemble(a; dirichlet = :ymin), rows)
        gD = :ymin => (x -> 0.0)
        @test refuses(() -> assemble(a, l; dirichlet = gD), rows)
        @test refuses(() -> assemble(l; dirichlet = gD), rows)
        A, F = assemble(a, l)
        @test refuses(() -> assemble!(A, F, a, l; dirichlet = gD), rows)
        @test refuses(() -> dirichlet_bc!(copy(A), C, :ymin), rows)
        @test refuses(
            () -> dirichlet_bc!(copy(F), C,
                Bramble.dirichlet_constraints(Ωₕ, :boundary => (x -> 0.0)), :boundary),
            rows)
        @test refuses(() -> symmetrize!(copy(A), copy(F), C, :ymin), rows)
        @test refuses(() -> assemble(form(C, X2(S, c), fa); dirichlet = :ymin), rows)
        @test refuses(() -> assemble(form(X2(S, c), C, fa); dirichlet = :ymin), rows)

        # no ghost vector of its own
        @test refuses(() -> Bramble.GhostVector(C), "no ghost vector of its own")
        @test refuses(() -> Bramble.GhostVector(C, zeros(3)), "no ghost vector")
        v = Bramble.GhostVector(S)
        @test refuses(() -> Bramble.update_ghosts!(v, C), "no ghost vector of its own")

        # what the product refuses
        @test refuses(() -> lift(C, zeros(Bramble.ndofs(C) + 1)), "one value per degree")
        @test refuses(() -> lift!(lift(C), C, zeros(3)), "one value per degree")
        @test refuses(
            () -> lift!(lift(S, zeros(Bramble.ndofs(S))), C,
                zeros(Bramble.ndofs(C))), "lives on another space")
        @test refuses(() -> Rₕ(C, x -> x[1]), "Rₕ or avgₕ")
        @test refuses(() -> Rₕ!(element(C), x -> x[1]), "Rₕ or avgₕ")
        @test refuses(() -> avgₕ(C, x -> x[1]), "Rₕ or avgₕ")
        @test refuses(() -> avgₕ!(element(C), x -> x[1]), "Rₕ or avgₕ")
        @test refuses(() -> form(C, S, (u, v) -> innerₕ(u, v)), "takes such a composite")
        @test refuses(() -> semidiscretize(a, l), "semidiscretize")
    end
end

end # module
