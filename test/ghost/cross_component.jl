module GhostCrossComponentTests

# Ghost relations on a composite of ghost spaces constrained as a whole,
# `constrained(S × S, u -> (… u(1) …, … u(2) …))`. With relations of one component each,
# the result must equal the product of the separately constrained components,
# `constrained(S, r₁) × constrained(S, r₂)`, whose closure is block diagonal (#781): `E`,
# `d`, the assembled matrix and load, and `lift`. A relation reading several components (a
# traction, #793) must hold on the lifted values, its rows assembled by one `inner_Γ` per
# component as the oracle, and must change the closure against one-component relations.
# At a corner of two tractions each relation's tangential difference is one-sided, which a
# residual written out by hand checks there. On a facet whose relations read several
# components, a mixed difference of the form reads the ghosts by extrapolation from the
# nodes inward (the facet rule), checked against that extrapolation built by hand, and
# keeps the reduced matrix free of eigenvalues of negative real part on cells stretched
# across a traction facet; relations of one component each leave the form as it was. Every
# mesh is non-uniform but the stretched uniform ones, and the components' relations and
# data differ, so a swapped or shared component shows. The refusals (data in time, the
# `dirichlet` rows, `GhostVector`, relations naming no component, a facet with too few or
# too many ghost relations, a singular block, a corner too short for the one-sided
# difference, a facet too short for the facet rule) are pinned by their messages.

using Test
using Bramble
import Bramble
using Bramble: lift, lift!, Dcₓ, Dcᵧ, D̃ₓ, D̃ᵧ, D₋ₓ, D₋ᵧ
using LinearAlgebra: Diagonal, dot, eigvals, norm
using SparseArrays: nnz
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
        # a facet with a relation reading several components takes one per component
        @test refuses(() -> constrained(V, u -> (:xmin => ∂ₙ(u(1)) + ∂ₙ(u(2)) => 0.0,)),
            "the facet :xmin has 1 relation reading its ghosts")
        # one ghost relation of each component per facet; both components may share one
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

        # no further component: the closure is solved once for the whole product
        whole_product = "takes no further component"
        W = gridspace(Ωₕ)
        for (left, right) in ((C, W), (W, C), (C, S), (S, C), (C, C), (C, V), (V, C))
            @test refuses(() -> left × right, whole_product)
        end
        @test refuses(() -> C^2, whole_product)
        @test refuses(() -> C^Val(2), whole_product)

        # a component of an element has no space of its own; its lift's component does
        uₕ = element(C, collect(1.0:Bramble.ndofs(C)))
        @test refuses(() -> uₕ(1), "has no component of its own")
        @test refuses(() -> Bramble.components(uₕ), "has no component of its own")
        Uₕ = lift(C, parent(uₕ))
        Ux = lift(X2(S, c), parent(uₕ))
        @test close(parent(Uₕ(2)), parent(Ux(2)); rtol = 1.0e-13)
        @test close(parent(Uₕ(1)), parent(Ux(1)); rtol = 1.0e-13)
    end

    # Isotropic traction, component i: Σⱼ ∂ₙ(u(j); K = Kᵢⱼ), the conormal of departure 1.
    λ, μ = 2.0, 0.5
    Kt(i, j) = [λ * (i == d) * (j == e) + μ * ((i == j) * (d == e) + (i == e) * (j == d))
                for d in 1:2, e in 1:2]
    t(u, i) = ∂ₙ(u(1); K = Kt(i, 1)) + ∂ₙ(u(2); K = Kt(i, 2))
    gd = (x -> x[1] * x[2], x -> 100.0 + x[1])
    disp(F) = u -> (F => u(1) => gd[1], F => u(2) => gd[2], F => ∂ₙₙ(u(1)) => 0.0,
        F => ∂ₙₙ(u(2)) => 0.0)
    # The traction data `tg[F][i]`, distinct per facet and component.
    tg = (xmin = (x -> 1.0 + x[2], 0.5), xmax = (-2.0, x -> 50.0 * x[2]^2),
        ymin = (x -> x[1], -1.0), ymax = (3.0, x -> 10.0 * x[1]))
    TRACTIONS = (
        one = (facets = (:xmin,), others = (:ymin, :xmax, :ymax)),
        apart = (facets = (:xmin, :xmax), others = (:ymin, :ymax)))
    traction(F) = u -> (F => t(u, 1) => tg[F][1], F => t(u, 2) => tg[F][2])
    rels_of(c) = u -> (Iterators.flatten(traction(F)(u) for F in c.facets)...,
        Iterators.flatten(disp(F)(u) for F in c.others)...)

    # The relation of component `i` on `F`: its rows by one `inner_Γ` per component, against
    # the facet weights times the data at the padded points. The rows of the padded points
    # `skip` (the corners of two tractions, where the difference along `F` is one-sided) are
    # left out of the residual, which comes with the whole residual vector.
    function traction_residual(PV, PS, U, i, F, g; K = Kt, skip = ())
        R = assemble(form(PV, PV,
            (w, v) -> inner_Γ(∂ₙ(w(1); K = K(i, 1)), v(i); markers = (F,)) +
                      inner_Γ(∂ₙ(w(2); K = K(i, 2)), v(i); markers = (F,))))
        W = assemble(form(PV, PV, (w, v) -> inner_Γ(w(i), v(i); markers = (F,))))
        m = Bramble.ndofs(PS)
        G = zeros(length(U))
        G[((i - 1) * m + 1):(i * m)] .= collect(Rₕ(PS, g isa Number ? (x -> g) : g))
        RU = R * U
        r = RU - W * G
        kept = copy(r)
        kept[[(i - 1) * m + p for p in skip]] .= 0.0
        return norm(kept, Inf) / max(1.0, norm(RU, Inf)), norm(W * G, Inf), r
    end

    # The padded point of the physical node `I`, every facet ghosted.
    padded_point(np, I) = LinearIndices(np)[I[1] + 1, I[2] + 1]

    # The relation of component `i` on the facet of axis `d`, side `s`, at the corner node
    # `I` (physical), on the padded values `U`, written out from the mesh points: the
    # outward sign times `Σⱼ Kᵢⱼ[d, d] ∂_d uⱼ + Kᵢⱼ[d, e] ∂_e uⱼ` minus the datum, `∂_d` the
    # centred difference over the ghost (the first spacing mirrored), `∂_e` the derivative
    # at the node of the quadratic through the three nodes along the facet into the domain.
    # The second value is the size of the terms, for a relative tolerance.
    function corner_relation(Ωₕ, np, U, i, d, s, I, K, g)
        e = 3 - d
        xs = (collect(points(Ωₕ(1))), collect(points(Ωₕ(2))))
        m = prod(np)
        val(j, J) = U[(j - 1) * m + padded_point(np, J)]
        ud = CartesianIndex(ntuple(a -> a == d ? 1 : 0, 2))
        ue = CartesianIndex(ntuple(a -> a == e ? 1 : 0, 2))
        h = abs(xs[d][I[d] == 1 ? 2 : end] - xs[d][I[d] == 1 ? 1 : end - 1])
        σ = I[e] == 1 ? 1 : -1                 # into the domain along `e`
        y = (xs[e][I[e]], xs[e][I[e] + σ], xs[e][I[e] + 2σ])
        w = (1 / (y[1] - y[2]) + 1 / (y[1] - y[3]),
            (y[1] - y[3]) / ((y[2] - y[1]) * (y[2] - y[3])),
            (y[1] - y[2]) / ((y[3] - y[1]) * (y[3] - y[2])))
        η = s == 1 ? -1.0 : 1.0
        terms = Float64[]
        for j in 1:2
            ∂d = (val(j, I + ud) - val(j, I - ud)) / (2h)
            ∂e = sum(w[k] * val(j, I + (k - 1) * σ * ue) for k in 1:3)
            push!(terms, η * K(i, j)[d, d] * ∂d, η * K(i, j)[d, e] * ∂e)
        end
        x = (xs[1][I[1]], xs[2][I[2]])
        gx = g isa Number ? g : g(x)
        return sum(terms) - gx, max(maximum(abs, terms), abs(gx))
    end

    @testset "Traction on lifted values: $kind, $name" for kind in KINDS,
        name in keys(TRACTIONS)

        c = TRACTIONS[name]
        Ωₕ = O.make_mesh(kind, 0; dim = 2, n0 = 7)
        S = gridspace(Ωₕ; ghosts = :boundary)
        V = S × S
        PV, PS = Bramble.padded_space(V), Bramble.padded_space(S)
        C = constrained(V, rels_of(c))
        E, d = Bramble.closure(C)
        @test size(E, 1) == Bramble.ndofs(PV)
        U = E * rand(Xoshiro(20261019), Bramble.ndofs(C)) + d
        for F in c.facets, i in 1:2

            res, scale = traction_residual(PV, PS, U, i, F, tg[F][i])
            @test res <= 1.0e-10
            @test scale > 0     # the data reach the oracle: a zero weight would hide them
        end
        # the displacement facets hold their data on the lifted values
        m = Bramble.ndofs(PS)
        Gs = (collect(Rₕ(PS, gd[1])), collect(Rₕ(PS, gd[2])))
        np = Bramble.npoints(Bramble.mesh(PS), Tuple)
        for j in 2:(np[1] - 1)
            p = LinearIndices(np)[j, 2]   # a node of :ymin, beyond its ghost row
            @test U[p] ≈ Gs[1][p] atol = 1.0e-13
            @test U[m + p] ≈ Gs[2][p] atol = 1.0e-13
        end

        # the cross terms: component 1's padded values read component 2's degrees of
        # freedom, which a closure of one-component relations never does
        flux(F) = u -> (F => ∂ₙ(u(1)) => 0.0, F => ∂ₙ(u(2)) => 0.0)
        one = u -> (Iterators.flatten(flux(F)(u) for F in c.facets)...,
            Iterators.flatten(disp(F)(u) for F in c.others)...)
        Ex, _ = Bramble.closure(constrained(V, one))
        # the degrees of freedom of component 1 alone, the first `n1` of the whole
        r1(u) = (Iterators.flatten((F => ∂ₙ(u) => 0.0,) for F in c.facets)...,
            Iterators.flatten((F => u => gd[1], F => ∂ₙₙ(u) => 0.0) for F in c.others)...)
        n1 = Bramble.ndofs(constrained(S, r1))
        @test size(Ex) == size(E)
        @test nnz(Ex[1:m, (n1 + 1):end]) == 0
        @test count(!iszero, E[1:m, (n1 + 1):end]) > 0
        @test count(!iszero, E[(m + 1):end, 1:n1]) > 0

        # the order of the relations changes nothing: each is attached by what it reads
        rev = u -> reverse(rels_of(c)(u))
        Er, dr = Bramble.closure(constrained(V, rev))
        @test close(Matrix(Er), Matrix(E); rtol = 1.0e-13)
        @test close(dr, d; rtol = 1.0e-13)
    end

    @testset "Refusals of cross-component relations" begin
        Ωₕ = O.make_mesh(:graded, 0; dim = 2, n0 = 7)
        S = gridspace(Ωₕ; ghosts = :boundary)
        V = S × S
        rest = u -> (disp(:ymin)(u)..., disp(:xmax)(u)..., disp(:ymax)(u)...)
        # one ghost relation per component on a facet that has any
        @test refuses(() -> constrained(V, u -> (:xmin => t(u, 1) => 0.0, rest(u)...)),
            "the facet :xmin has 1 relation reading its ghosts")
        @test refuses(
            () -> constrained(V, u -> (:xmin => t(u, 1) => 0.0, :xmin => t(u, 2) => 0.0,
                :xmin => ∂ₙ(u(1)) => 0.0, rest(u)...)),
            "the facet :xmin has 3 relations reading its ghosts")
        # two relations of component 1 alone on one facet
        @test refuses(
            () -> constrained(V, u -> (:xmin => ∂ₙ(u(1)) => 0.0,
                :xmin => 2.0 * ∂ₙ(u(1)) => 0.0, rest(u)...)),
            ":xmin of component 1 has two relations")
        # two relations reading both components that agree: a singular block at each node
        @test refuses(
            () -> constrained(V, u -> (:xmin => ∂ₙ(u(1)) + ∂ₙ(u(2)) => 0.0,
                :xmin => 2.0 * (∂ₙ(u(1)) + ∂ₙ(u(2))) => 0.0, rest(u)...)),
            "cannot be solved for the ghosts of components 1 and 2")
        # a corner of two tractions with two nodes along a facet: no one-sided difference
        corner = u -> (:xmin => t(u, 1) => 0.0, :xmin => t(u, 2) => 0.0,
            :ymin => t(u, 1) => 0.0, :ymin => t(u, 2) => 0.0, disp(:xmax)(u)...,
            disp(:ymax)(u)...)
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 2), (false, true))
        V2 = gridspace(Ω2; ghosts = :boundary)^2
        @test refuses(() -> constrained(V2, corner), "along axis 2, which has 2")
        # still one component's Dirichlet relation per facet
        @test refuses(
            () -> constrained(V,
                u -> (:xmin => t(u, 1) => 0.0, :xmin => t(u, 2) => 0.0,
                    :ymin => u(1) => 0.0, :ymin => u(1) => 1.0, rest(u)...)),
            ":ymin of component 1 has two Dirichlet")
    end

    # Isotropic tractions on every facet: at each corner the two shear relations are one
    # equation, so a joint corner block is singular. Each facet's relations read the other
    # facet's ghosts through the one-sided difference, and hold on the lifted values: the
    # centred rows away from the corners, the one-sided relation written out at them.
    @testset "Traction corners: $kind" for kind in KINDS
        Ωₕ = O.make_mesh(kind, 0; dim = 2, n0 = 7)
        S = gridspace(Ωₕ; ghosts = :boundary)
        V = S × S
        PV, PS = Bramble.padded_space(V), Bramble.padded_space(S)
        np = Bramble.npoints(Bramble.mesh(PS), Tuple)
        n = npoints(Ωₕ, Tuple)
        facets = (:xmin, :xmax, :ymin, :ymax)
        C = constrained(V, u -> (Iterators.flatten(traction(F)(u) for F in facets)...,))
        @test Bramble.ndofs(C) == 2 * prod(n)
        E, d = Bramble.closure(C)
        U = E * rand(Xoshiro(20261019), Bramble.ndofs(C)) + d
        corners = [CartesianIndex(a, b) for a in (1, n[1]), b in (1, n[2])]
        skip = [padded_point(np, I) for I in corners]
        for (k, F) in enumerate(facets), i in 1:2

            res, scale, r = traction_residual(PV, PS, U, i, F, tg[F][i]; skip)
            @test res <= 1.0e-10
            @test scale > 0
            dd, s = (k + 1) ÷ 2, 2 - k % 2
            for I in corners
                I[dd] == (s == 1 ? 1 : n[dd]) || continue
                c, sz = corner_relation(Ωₕ, np, U, i, dd, s, I, Kt, tg[F][i])
                @test abs(c) <= 1.0e-10 * max(1.0, sz)
                # the centred row is not what holds there
                @test abs(r[(i - 1) * prod(np) + padded_point(np, I)]) > 1.0e-6
            end
        end
    end

    # A corner of tractions with a K without the stress's symmetry takes the same one-sided
    # difference, and every relation holds on the lifted values, the corner nodes included.
    @testset "Anisotropic traction corner: $kind" for kind in KINDS
        Ωₕ = O.make_mesh(kind, 0; dim = 2, n0 = 7)
        S = gridspace(Ωₕ; ghosts = :boundary)
        V = S × S
        PV, PS = Bramble.padded_space(V), Bramble.padded_space(S)
        rng = Xoshiro(3)
        Ka = Dict((i, j) => randn(rng, 2, 2) .+ (i == j ? 4.0 : 0.0) .* [1 0; 0 1]
        for i in 1:2, j in 1:2)
        Kf(i, j) = Ka[(i, j)]
        ta(u, i) = ∂ₙ(u(1); K = Kf(i, 1)) + ∂ₙ(u(2); K = Kf(i, 2))
        g = (xmin = (1.0, 0.0), ymin = (0.0, x -> x[1]))
        C = constrained(V,
            u -> (:xmin => ta(u, 1) => g.xmin[1],
                :xmin => ta(u, 2) => g.xmin[2], :ymin => ta(u, 1) => g.ymin[1],
                :ymin => ta(u, 2) => g.ymin[2], disp(:xmax)(u)..., disp(:ymax)(u)...))
        E, d = Bramble.closure(C)
        U = E * rand(Xoshiro(20261019), Bramble.ndofs(C)) + d
        np = Bramble.npoints(Bramble.mesh(PS), Tuple)
        I = CartesianIndex(1, 1)
        for (dd, F) in ((1, :xmin), (2, :ymin)), i in 1:2

            res, _, r = traction_residual(PV, PS, U, i, F, g[F][i]; K = Kf,
                skip = (padded_point(np, I),))
            @test res <= 1.0e-10
            c, sz = corner_relation(Ωₕ, np, U, i, dd, 1, I, Kf, g[F][i])
            @test abs(c) <= 1.0e-10 * max(1.0, sz)
            @test abs(r[(i - 1) * prod(np) + padded_point(np, I)]) > 1.0e-6
        end
    end

    # Relations of one component each, component 2 leaving the ghosts of :ymin and :ymax as
    # unknowns: the whole equals the product, as without any relation reading several.
    @testset "Ghosts of one component kept: $kind" for kind in KINDS
        Ωₕ = O.make_mesh(kind, 0; dim = 2, n0 = 7)
        S = gridspace(Ωₕ; ghosts = :boundary)
        r1 = u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => ∂ₙ(u) => 0.0, :ymin => ∂ₙ(u) => 0.0,
            :ymax => ∂ₙ(u) => x -> x[1])
        r2 = u -> (:xmin => ∂ₙ(u) => 2.0, :xmax => ∂ₙ(u) => x -> 100.0 * x[2])
        X = constrained(S, r1) × constrained(S, r2)
        C = constrained(S × S, whole(r1, r2))
        @test Bramble.ndofs(C) == Bramble.ndofs(X)
        E, d = Bramble.closure(C)
        Ex, dx = Bramble.closure(X)
        @test size(E) == size(Ex)
        @test close(Matrix(E), Matrix(Ex); rtol = 1.0e-13)
        @test close(d, dx; rtol = 1.0e-13)
        # component 2 keeps the ghosts of :ymin and :ymax, component 1 none
        np = npoints(Ωₕ, Tuple)
        @test Bramble.ndofs(C) == 2 * prod(np) + 2 * np[1]
    end

    # The facet rule: on a facet whose relations read several components, a mixed
    # difference of the form reads the facet's ghosts as the quadratic through the three
    # nodes inward along the normal, extrapolated to the ghost; second differences read the
    # closure's ghosts. The oracle builds that extrapolation by hand from the mesh points, a
    # matrix `X` on the padded values, and reduces `Eᵀ (A_second + A_mixed X) E` from forms
    # on the padded composite.
    second(u, a) = a == 1 ? D̃ₓ(D₋ₓ(u)) : D̃ᵧ(D₋ᵧ(u))
    fs = (u, v) -> sum(innerₕ(-μ * Δₕ(u(i)), v(i)) +
                       innerₕ(-(λ + μ) * second(u(i), i), v(i)) for i in 1:2)
    fx = (u, v) -> innerₕ(-(λ + μ) * Dcₓ(Dcᵧ(u(2))), v(1)) +
                   innerₕ(-(λ + μ) * Dcₓ(Dcᵧ(u(1))), v(2))
    fe = (u, v) -> fs(u, v) + fx(u, v)
    SIDE = (xmin = (1, 1), xmax = (1, 2), ymin = (2, 1), ymax = (2, 2))
    function extrapolation(Ωₕ, facets)
        xs = (collect(points(Ωₕ(1))), collect(points(Ωₕ(2))))
        np = map(length, xs) .+ 2
        L = LinearIndices(np)
        m = prod(np)
        px = map(x -> [2x[1] - x[2]; x; 2x[end] - x[end - 1]], xs)
        X = Matrix{Float64}(Diagonal(ones(2m)))
        for c in 0:1, F in facets

            a, s = SIDE[F]
            g, σ = s == 1 ? (1, 1) : (np[a], -1)
            for r in 2:(np[3 - a] - 1)
                at(k) = a == 1 ? L[g + k * σ, r] : L[r, g + k * σ]
                z, y = px[a][g], ntuple(k -> px[a][g + k * σ], 3)
                X[c * m + at(0), c * m + at(0)] = 0.0
                for k in 1:3
                    X[c * m + at(0), c * m + at(k)] = prod((z - y[o]) / (y[k] - y[o])
                    for o in 1:3 if o != k)
                end
            end
        end
        return X
    end
    corner_rels = u -> (traction(:xmin)(u)..., traction(:ymin)(u)..., disp(:xmax)(u)...,
        disp(:ymax)(u)...)

    FACET_CASES = (((:xmin,), rels_of(TRACTIONS.one)), ((:xmin, :ymin), corner_rels))
    @testset "Facet rule vs extrapolation: $kind, $(facets)" for kind in KINDS,
        (facets, rels) in FACET_CASES

        Ωₕ = O.make_mesh(kind, 0; dim = 2, n0 = 7)
        V = gridspace(Ωₕ; ghosts = :boundary)^2
        PV = Bramble.padded_space(V)
        C = constrained(V, rels)
        E, d = Bramble.closure(C)
        E = Matrix(E)
        X = extrapolation(Ωₕ, facets)
        As, Ax = Matrix(assemble(form(PV, PV, fs))), Matrix(assemble(form(PV, PV, fx)))
        A = Matrix(assemble(form(C, C, fe)))
        @test close(A, transpose(E) * (As + Ax * X) * E)
        @test close(Matrix(assemble(form(C, C, fx))), transpose(E) * Ax * X * E)
        # the centred mixed difference over the closure's ghosts is not what assembles
        @test norm(A - transpose(E) * (As + Ax) * E) > 1.0e-3 * norm(A)
        # the load reduces through the same padded matrix
        fl = v -> innerₕ(x -> 1.0 + x[1], v(1)) + innerₕ(x -> 100.0 * x[2], v(2))
        _, F = assemble(form(C, C, fe), form(C, fl))
        Fp = assemble(form(PV, fl))
        @test close(F, transpose(E) * (Fp - (As + Ax * X) * d))
    end

    # Mixed differences spelled with sums and scales on either side of the inner difference,
    # each `(rest, mixed)` against `Eᵀ (A_rest + A_mixed X) E`, the form assembled being
    # the one spelled; equal operators assemble equal matrices. A facet of one-component
    # conormal relations beside the traction keeps its ghosts in the mixed differences.
    @testset "Facet rule, other spellings: $kind" for kind in KINDS
        Ωₕ = O.make_mesh(kind, 0; dim = 2, n0 = 7)
        S = gridspace(Ωₕ; ghosts = :boundary)
        V = S^2
        PV = Bramble.padded_space(V)
        gv = 1.0 .+ rand(Xoshiro(5), Bramble.ndofs(Bramble.padded_space(S)))
        c = -(λ + μ)
        none = (u, v) -> innerₕ(0.0 * u(1), v(1))
        spellings = (
            (none, (u, v) -> innerₕ(Dcₓ(2.0 * Dcᵧ(u(2))), v(1)),
                (u, v) -> innerₕ(Dcₓ(2.0 * Dcᵧ(u(2))), v(1))),
            (none, (u, v) -> innerₕ(Dcₓ(Dcᵧ(u(1)) + Dcᵧ(u(2))), v(1)),
                (u, v) -> innerₕ(Dcₓ(Dcᵧ(u(1)) + Dcᵧ(u(2))), v(1))),
            ((u, v) -> innerₕ(c * Dcₓ(Dcₓ(u(1))), v(1)),
                (u, v) -> innerₕ(c * Dcₓ(Dcᵧ(u(2))), v(1)),
                (u, v) -> innerₕ(c * Dcₓ(Dcₓ(u(1)) + Dcᵧ(u(2))), v(1))),
            ((u, v) -> innerₕ(Dcₓ(D₋ₓ(u(1))), v(2)),
                (u, v) -> innerₕ(Dcₓ(Dcᵧ(u(2))), v(2)),
                (u, v) -> innerₕ(Dcₓ(Dcᵧ(u(2)) + D₋ₓ(u(1))), v(2))),
            (none, (u, v) -> innerₕ(Dcₓ(gv * Dcᵧ(u(2))), v(1)),
                (u, v) -> innerₕ(Dcₓ(gv * Dcᵧ(u(2))), v(1))),
            (none, (u, v) -> innerₕ(Dcₓ(Dcᵧ(gv * u(2))), v(1)),
                (u, v) -> innerₕ(Dcₓ(Dcᵧ(gv * u(2))), v(1))),
            (none, (u, v) -> innerₕ(Dcₓ(2.0 * (Dcᵧ(u(2)) + Dcᵧ(u(1)))), v(2)),
                (u, v) -> innerₕ(Dcₓ(2.0 * (Dcᵧ(u(2)) + Dcᵧ(u(1)))), v(2))),
            (none, (u, v) -> innerₕ(Dcᵧ(Dcₓ(u(1) + 3.0 * u(2))), v(1)),
                (u, v) -> innerₕ(Dcᵧ(Dcₓ(u(1) + 3.0 * u(2))), v(1))))
        for (facets, rels) in FACET_CASES
            C = constrained(V, rels)
            E = Matrix(first(Bramble.closure(C)))
            X = extrapolation(Ωₕ, facets)
            for (rest, mixed, spelled) in spellings
                Ar = Matrix(assemble(form(PV, PV, rest)))
                Am = Matrix(assemble(form(PV, PV, mixed)))
                A = Matrix(assemble(form(C, C, spelled)))
                @test close(A, transpose(E) * (Ar + Am * X) * E)
                @test norm(A - transpose(E) * (Ar + Am) * E) > 1.0e-3 * norm(A)
            end
            Asum = assemble(form(C, C, (u, v) -> innerₕ(Dcₓ(Dcᵧ(u(1) + u(2))), v(1))))
            Asplit = assemble(form(C, C,
                (u, v) -> innerₕ(Dcₓ(Dcᵧ(u(1))), v(1)) + innerₕ(Dcₓ(Dcᵧ(u(2))), v(1))))
            @test close(Matrix(Asum), Matrix(Asplit))
            # either argument order: the left argument's stencil gives the columns, the
            # padded values the closure fills, so its mixed difference reads them by
            # extrapolation whether it is written of `u` or of `v`; the right argument's
            # gives the rows and is left as it is (Bramble places the sides by position)
            for g in ((u, v) -> innerₕ(Dcₓ(Dcᵧ(v(2))), u(1)),
                (u, v) -> innerₕ(Dcₓ(Dcᵧ(u(2))), v(1)))
                Ag, Pg = Matrix(assemble(form(C, C, g))), Matrix(assemble(form(PV, PV, g)))
                @test close(Ag, transpose(E) * Pg * X * E)
                @test norm(Ag - transpose(E) * Pg * E) > 1.0e-3 * norm(Ag)
            end
            for g in ((u, v) -> innerₕ(v(1), Dcₓ(Dcᵧ(u(2)))),
                (u, v) -> innerₕ(u(1), Dcₓ(Dcᵧ(v(2)))))
                Ag, Pg = Matrix(assemble(form(C, C, g))), Matrix(assemble(form(PV, PV, g)))
                @test close(Ag, transpose(E) * Pg * E; rtol = 1.0e-13)
            end
        end
        Ka, Kb = [2.0 0.6; 0.6 1.0], [1.0 0.3; 0.3 1.0]
        C = constrained(V,
            u -> (traction(:xmin)(u)..., :ymin => ∂ₙ(u(1); K = Ka) => 0.0,
                :ymin => ∂ₙ(u(2); K = Kb) => x -> x[1], disp(:xmax)(u)...,
                disp(:ymax)(u)...))
        @test C.facets == ((true, false), (false, false))
        E = Matrix(first(Bramble.closure(C)))
        X = extrapolation(Ωₕ, (:xmin,))
        As, Ax = Matrix(assemble(form(PV, PV, fs))), Matrix(assemble(form(PV, PV, fx)))
        @test close(Matrix(assemble(form(C, C, fe))), transpose(E) * (As + Ax * X) * E)
        # a relation reading several components marks its facet by what it is written of,
        # though its term in `u(2)` vanishes and its ghosts read component 1 alone
        C = constrained(V,
            u -> (:xmin => ∂ₙ(u(1)) + 0.0 * u(2) => 1.0, :xmin => ∂ₙ(u(2)) => 0.0,
                disp(:ymin)(u)..., disp(:xmax)(u)..., disp(:ymax)(u)...))
        @test C.facets == ((true, false), (false, false))
        E = Matrix(first(Bramble.closure(C)))
        @test close(Matrix(assemble(form(C, C, fe))), transpose(E) * (As + Ax * X) * E)
        Afx = Matrix(assemble(form(C, C, fx)))
        @test norm(Afx - transpose(E) * Ax * E) > 1.0e-3 * norm(Afx)
    end

    # Traction on one facet, displacement on the others, on cells twice as long across the
    # facet as along it (uniform) and on a random mesh whose ratio reaches 3.7 on :ymin: the
    # centred mixed difference over the traction's ghosts gave eigenvalues of negative real
    # part there; with the facet rule the reduced matrix has none.
    @testset "Facet rule, stretched cells: $(n), $(F)" for (n, F) in (
        ((17, 9), :ymin), ((9, 17), :xmin), (:random, :ymin))
        Ωₕ = n === :random ? O.make_mesh(:random, 0; dim = 2, n0 = 9, seed = 7) :
             mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), n, (true, true))
        a, s = SIDE[F]
        xs = (collect(points(Ωₕ(1))), collect(points(Ωₕ(2))))
        across = s == 1 ? xs[a][2] - xs[a][1] : xs[a][end] - xs[a][end - 1]
        @test across / minimum(diff(xs[3 - a])) >= 2.0 - 1.0e-12
        others = filter(!=(F), (:xmin, :xmax, :ymin, :ymax))
        C = constrained(gridspace(Ωₕ; ghosts = :boundary)^2,
            u -> (traction(F)(u)..., Iterators.flatten(disp(G)(u) for G in others)...))
        @test minimum(real, eigvals(Matrix(assemble(form(C, C, fe))))) > 0
    end

    # Relations of one component each, scalar or per component, leave the mixed difference
    # centred over the closure's ghosts.
    @testset "Facet rule off one component: $kind" for kind in KINDS
        Ωₕ = O.make_mesh(kind, 0; dim = 2, n0 = 7)
        S = gridspace(Ωₕ; ghosts = :boundary)
        PV = Bramble.padded_space(S^2)
        c = CASES.boundary
        for C in (constrained(S^2, whole(c.r1, c.r2)), X2(S, c))
            E = Matrix(first(Bramble.closure(C)))
            Ap = Matrix(assemble(form(PV, PV, fe)))
            @test close(Matrix(assemble(form(C, C, fe))), transpose(E) * Ap * E;
                rtol = 1.0e-13)
        end
        W = constrained(S, c.r1)
        P = Bramble.padded_space(S)
        f1 = (u, v) -> innerₕ(-Δₕ(u), v) + innerₕ(Dcₓ(Dcᵧ(u)), v)
        E = Matrix(first(Bramble.closure(W)))
        @test close(Matrix(assemble(form(W, W, f1))),
            transpose(E) * Matrix(assemble(form(P, P, f1))) * E; rtol = 1.0e-13)
    end

    @testset "Facet rule refuses a short axis" begin
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (2, 7), (true, false))
        C = constrained(gridspace(Ω2; ghosts = :boundary)^2,
            u -> (traction(:xmin)(u)..., disp(:xmax)(u)..., disp(:ymin)(u)...,
                disp(:ymax)(u)...))
        @test refuses(() -> form(C, C, fe), "three nodes along axis 1, which has 2")
    end

    @testset "Arithmetic on a whole element" begin
        Ωₕ = O.make_mesh(:random, 0; dim = 2, n0 = 7)
        S = gridspace(Ωₕ; ghosts = :boundary)
        V = S × S
        C = constrained(V, u -> (:xmin => t(u, 1) => 1.0, :xmin => t(u, 2) => 0.0,
            disp(:ymin)(u)..., disp(:xmax)(u)..., disp(:ymax)(u)...))
        u = collect(1.0:Bramble.ndofs(C)) ./ 7
        w = element(C, u)
        z = copy(w)
        @test z isa Bramble.VectorElement
        @test parent(z) == u && parent(z) !== parent(w)
        @test parent(w + 2 * w) == 3 .* u
        @test parent(w - w) == zero(u)
        y = similar(w)
        y .= 2 .* w .+ 1.0
        @test parent(y) == 2 .* u .+ 1.0
        @test w ≈ w
        @test !(w ≈ 2 * w)
        x = element(C)
        x .= 1.0
        @test parent(copy(x) + 2 * x) == fill(3.0, Bramble.ndofs(C))
    end
end

end # module
