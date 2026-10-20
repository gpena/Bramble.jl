module GhostKroneckerTests

# `kronecker_operator` on forms over ghost spaces (#781): with one relation per facet along
# its normal, the operator is the reduced matrix `Eᵀ A_pad E` of `assemble(a)`, built from
# 1D factors. The reference is that product from the padded space and the closure, in
# dense matrices; the meshes are seeded random ones, with a different size per axis.

using Random
using Test
using Bramble
import Bramble
using Bramble: padded_space, closure, restrict_to
using LinearAlgebra: ×
using SparseArrays: SparseMatrixCSC
using ..TestUtils: WITH_3D_TESTS

isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
const O = Main.GhostOracleND

function random_mesh(ns::Tuple)
    Ω = domain(foldl(×, map(_ -> interval(0.0, 1.0), ns)))
    Bramble._seed_mesh1d_rng!(20261017)
    try
        return mesh(Ω, ns, map(_ -> false, ns))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end

nonuniform(Ωₕ) = all(d -> !allequal(round.(diff(Bramble.points(Ωₕ(d))); digits = 12)),
    1:Bramble.dim(Ωₕ))

# The operator as a dense matrix, column by column through `mul!`.
function dense(K)
    n = size(K, 2)
    M = zeros(n, n)
    for j in 1:n
        e = zeros(n)
        e[j] = 1.0
        M[:, j] = K * e
    end
    return M
end

# `Eᵀ A_pad E` from the padded space and the closure, sharing no code with the operator.
function reference(C, f)
    P = padded_space(C)
    E = Matrix(closure(C)[1])
    return E' * Matrix(assemble(form(P, P, f))) * E
end

FORMS = (
    "stiffness" => (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)),
    "mass + stiffness" => (u, v) -> innerₕ(u, v) + 0.5 * inner₊(∇ₕ(u), ∇ₕ(v)),
    "advection" => (u, v) -> innerₕ(Bramble.D₋ₓ(u), v) + inner₊(∇ₕ(u), ∇ₕ(v)),
    "facet term" => (u, v) -> inner_Γ(u, v; markers = (:xmin, :ymax)) +
                              inner₊(∇ₕ(u), ∇ₕ(v))
)

function check(C, f)
    a = form(C, C, f)
    K = kronecker_operator(a)
    R = reference(C, f)
    @test K isa Bramble.KroneckerLinearOperator
    @test size(K) == size(R)
    @test isapprox(dense(K), R; rtol = 1e-12)
    @test isapprox(Matrix(SparseMatrixCSC(K)), Matrix(assemble(a)); rtol = 1e-12)
    return K
end

RELATIONS_2D = (
    "Neumann" =>
        u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => ∂ₙ(u) => 0.0, :ymin => ∂ₙ(u) => 0.0,
            :ymax => ∂ₙ(u) => x -> x[1]),
    "Robin" =>
        u -> (:xmin => 2.0 * u + ∂ₙ(u) => 0.0, :xmax => 0.5 * u + 3.0 * ∂ₙ(u) => 1.0,
            :ymin => u + ∂ₙ(u) => 0.0, :ymax => 4.0 * u + ∂ₙ(u) => x -> x[1]),
    "Dirichlet, Neumann and Robin" =>
        u -> (:xmin => u => 0.0,
            :xmax => 2.0 * u + ∂ₙ(u) => x -> x[2], :ymin => ∂ₙ(u) => 1.0, :ymax => u => 0.0),
    "diagonal conormal" =>
        u -> (:xmin => ∂ₙ(u; K = [2.0 0.0; 0.0 0.5]) => 0.0,
            :xmax => ∂ₙ(u; K = [2.0 0.0; 0.0 0.5]) => 0.0, :ymin => ∂ₙ(u) => 0.0,
            :ymax => u + ∂ₙ(u) => 0.0)
)

@testset "Ghost kronecker_operator 2D" begin
    for Ωₕ in (random_mesh((9, 7)), O.make_mesh(:graded, 0))
        @test nonuniform(Ωₕ)
        for (name, rel) in RELATIONS_2D
            ghosts = name == "Dirichlet, Neumann and Robin" ? (:xmax, :ymin) : :boundary
            C = constrained(gridspace(Ωₕ; ghosts = ghosts), rel)
            @testset "$name, $fname" for (fname, f) in FORMS
                check(C, f)
            end
        end
    end
end

@testset "Ghost kronecker_operator 1D factors" begin
    Ωₕ = random_mesh((9, 7))
    C = constrained(gridspace(Ωₕ; ghosts = (:xmin, :ymax)),
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => u => 0.0, :ymin => u => 0.0,
            :ymax => u + ∂ₙ(u) => 1.0))
    K = check(C, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    # 9 points along x less the fixed xmax, 7 along y less the fixed ymin.
    @test K.dims == (8, 6)
    @test all(t -> map(F -> size(F), t.factors) == ((8, 8), (6, 6)), K.terms)
end

@testset "Ghost kronecker_operator two spaces" begin
    Ωₕ = random_mesh((7, 6))
    Cu = constrained(gridspace(Ωₕ; ghosts = :boundary),
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.0, :ymin => u + ∂ₙ(u) => 0.0,
            :ymax => ∂ₙ(u) => 0.0))
    Cv = constrained(gridspace(Ωₕ; ghosts = :boundary),
        u -> (:xmin => 2.0 * u + ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.0,
            :ymin => ∂ₙ(u) => 0.0, :ymax => 0.5 * u + ∂ₙ(u) => 1.0))
    @test Bramble.mesh(Cu) !== Bramble.mesh(Cv)
    for (fname, f) in FORMS
        a = form(Cu, Cv, f)
        K = kronecker_operator(a)
        Eu, Ev = Matrix(closure(Cu)[1]), Matrix(closure(Cv)[1])
        P, Q = padded_space(Cu), padded_space(Cv)
        R = Ev' * Matrix(assemble(form(P, Q, f))) * Eu
        @test K isa Bramble.KroneckerLinearOperator
        @test isapprox(dense(K), R; rtol = 1e-12)
        @test isapprox(dense(K), Matrix(assemble(a)); rtol = 1e-12)
    end
end

@testset "Ghost kronecker_operator live Ref" begin
    Ωₕ = random_mesh((7, 8))
    C = constrained(gridspace(Ωₕ; ghosts = :boundary),
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.0, :ymin => u + ∂ₙ(u) => 0.0,
            :ymax => ∂ₙ(u) => 0.0))
    c = Ref(2.0)
    a = form(C, C, (u, v) -> c * innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
    K = kronecker_operator(a)
    c[] = -3.0
    @test isapprox(dense(K), Matrix(assemble(a)); rtol = 1e-12)
    @test isapprox(dense(K), reference(C, (u, v) -> -3.0 * innerₕ(u, v) +
                                                    inner₊(∇ₕ(u), ∇ₕ(v))); rtol = 1e-12)
end

WITH_3D_TESTS && @testset "Ghost kronecker_operator 3D" begin
    Ωₕ = random_mesh((5, 6, 4))
    @test nonuniform(Ωₕ)
    rels = (
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => 2.0 * u + ∂ₙ(u) => 1.0, :ymin => ∂ₙ(u) => 0.0,
            :ymax => u + ∂ₙ(u) => 0.0, :zmin => ∂ₙ(u) => x -> x[3], :zmax => ∂ₙ(u) => 0.0),
        u -> (:xmin => u => 0.0, :xmax => ∂ₙ(u) => 0.0, :ymin => u + ∂ₙ(u) => 0.0,
            :ymax => ∂ₙ(u) => 0.0, :zmin => ∂ₙ(u) => 0.0, :zmax => u => 0.0)
    )
    ghosts = (:boundary, (:xmax, :ymin, :ymax, :zmin))
    for (rel, gh) in zip(rels, ghosts), (fname, f) in FORMS
        C = constrained(gridspace(Ωₕ; ghosts = gh), rel)
        @testset "$gh, $fname" begin
            check(C, f)
        end
    end
end

@testset "Ghost kronecker_operator refusals" begin
    Ωₕ = random_mesh((9, 7))
    stiffness = (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))
    message(f) =
        try
            f()
            ""
        catch e
            e isa ArgumentError ? sprint(showerror, e) : rethrow()
        end
    names_fix(msg) = occursin("separable", msg) && occursin("assemble(a)", msg)

    # A conormal with off-diagonal entries couples the axes at the corners.
    K2 = [2.0 0.5; 0.5 1.0]
    Cn = constrained(gridspace(Ωₕ; ghosts = :boundary),
        u -> (:xmin => ∂ₙ(u; K = K2) => 0.0, :xmax => ∂ₙ(u; K = K2) => 0.0,
            :ymin => ∂ₙ(u; K = K2) => 0.0, :ymax => ∂ₙ(u; K = K2) => 0.0))
    @test_throws ArgumentError kronecker_operator(form(Cn, Cn, stiffness))
    @test names_fix(message(() -> kronecker_operator(form(Cn, Cn, stiffness))))

    # Off-diagonal entries on one facet only still couple its ghosts to tangential nodes.
    C1 = constrained(gridspace(Ωₕ; ghosts = :boundary),
        u -> (:xmin => ∂ₙ(u; K = K2) => 0.0, :xmax => ∂ₙ(u) => 0.0, :ymin => ∂ₙ(u) => 0.0,
            :ymax => ∂ₙ(u) => 0.0))
    @test names_fix(message(() -> kronecker_operator(form(C1, C1, stiffness))))

    # Ghosts left as unknowns: the degrees of freedom are not a product of axis points.
    Wg = gridspace(Ωₕ; ghosts = (:xmin, :ymax))
    @test names_fix(message(() -> kronecker_operator(form(Wg, Wg, stiffness))))

    # A mixed derivative reads the absent padded points, where the closure is 0 but the
    # Kronecker product of the 1D closures is not.
    Cr = constrained(gridspace(Ωₕ; ghosts = :boundary),
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.0, :ymin => ∂ₙ(u) => 0.0,
            :ymax => u + ∂ₙ(u) => 0.0))
    mixed = (u, v) -> innerₕ(Bramble.D₋ₓ(Bramble.D₋ᵧ(u)), v)
    msg = message(() -> kronecker_operator(form(Cr, Cr, mixed)))
    @test names_fix(msg) && occursin("absent", msg) && !occursin("conormal", msg)

    # Spaces on two physical meshes of one size, or ghosting different facets.
    Ω2 = random_mesh((9, 7))
    rel = u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.0, :ymin => ∂ₙ(u) => 0.0,
        :ymax => ∂ₙ(u) => 0.0)
    Cm = constrained(gridspace(Ω2; ghosts = :boundary), rel)
    msg = message(() -> kronecker_operator(form(Cr, Cm, stiffness)))
    @test occursin("different mesh objects", msg) && occursin("assemble(a)", msg)
    # Equal point for point, but another object: the operator checks one mesh for changes.
    Ω3 = random_mesh((9, 7))
    @test Bramble.points(Ω3(1)) == Bramble.points(Ωₕ(1))
    Ce = constrained(gridspace(Ω3; ghosts = :boundary), rel)
    msg = message(() -> kronecker_operator(form(Cr, Ce, stiffness)))
    @test occursin("different mesh objects", msg) && !occursin("separable", msg)

    # A Dirichlet relation on the test side only: the reduced system is rectangular.
    Cd = constrained(gridspace(Ωₕ; ghosts = :boundary),
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.0, :ymin => ∂ₙ(u) => 0.0,
            :ymin => u => 0.0, :ymax => u + ∂ₙ(u) => 0.0))
    ad = form(Cr, Cd, stiffness)
    @test size(assemble(ad)) == (ndofs(Cd), ndofs(Cr)) != (ndofs(Cr), ndofs(Cr))
    msg = message(() -> kronecker_operator(ad))
    @test occursin("rectangular", msg) && occursin("assemble(a)", msg) &&
          occursin(string(ndofs(Cr)), msg) && occursin(string(ndofs(Cd)), msg)
    Cx = constrained(gridspace(Ωₕ; ghosts = (:xmin, :xmax)),
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.0))
    @test_throws ArgumentError kronecker_operator(form(Cr, Cx, stiffness))

    # A region restriction other than `:interior` is refused, naming `assemble(a)`.
    Random.seed!(20261017)
    Ωb = mesh(
        domain(interval(0.0, 1.0) × interval(0.0, 1.0),
            :blob => x -> (x[1] - 0.4)^2 + (x[2] - 0.5)^2 < 0.1), (9, 7), (false, false))
    Cb = constrained(gridspace(Ωb; ghosts = :boundary),
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.0, :ymin => ∂ₙ(u) => 0.0,
            :ymax => ∂ₙ(u) => 0.0))
    restricted = (u, v) -> innerₕ(Bramble.D₋ₓ(restrict_to(:blob, u)), v)
    @test_throws ArgumentError kronecker_operator(form(Cb, Cb, restricted))
    @test occursin("assemble(a)",
        message(() -> kronecker_operator(form(Cb, Cb, restricted))))

    # A 1D form has nothing to factor.
    Ω1 = Main.GhostOracle1D.make_mesh(:random, 0)
    C1d = constrained(gridspace(Ω1; ghosts = :boundary),
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => u + ∂ₙ(u) => 0.0))
    @test occursin("two dimensions",
        message(() -> kronecker_operator(form(C1d, C1d, stiffness))))
end

end
