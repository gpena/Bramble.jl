module GhostTractionTests

# Linear elasticity in collocated form with traction and displacement facets, closed
# through ghost points (#793). Component i of the operator is
# `−μ Δₕ uᵢ − (λ + μ) Σⱼ ∂ᵢ∂ⱼ uⱼ`, the second difference of one axis for `j = i` and the
# centred mixed difference `Dcᵢ(Dcⱼ(·))` otherwise, on a composite of ghost spaces
# constrained as a whole. The traction of component i on a facet is
# `Σⱼ ∂ₙ(u(j); K = Kᵢⱼ)`, `Kᵢⱼ[d, e] = λ δᵢd δⱼe + μ (δᵢⱼ δde + δᵢe δⱼd)`; a displacement
# facet carries `u(i) => g` and `∂ₙₙ(u(i)) => h`, both from the manufactured solution.
#
# On a facet whose relations read several components (a traction), the mixed differences
# of the form read the facet's ghosts as the quadratic through the three nodes inward,
# extrapolated along the normal (the facet rule, T2.4): the centred mixed difference over
# the traction's ghosts is unstable once cells are longer across the facet than along it.
# So the reduced matrix is `Eᵀ (A_rest + A_mixed X) E`, `X` that extrapolation.
#
# Covered: the reduced matrix and load against `Eᵀ (A_rest + A_mixed X) E` and
# `Eᵀ (F_pad − (A_rest + A_mixed X) d)` from the padded composite, `X` built by hand; the
# relation at the corner of the two tractions written out from the mesh points; symmetry
# of the reduced matrix of a symmetric padded form, and the padded elasticity matrix's lack
# of it; and, nightly, the same in 3D. The problem and its errors are in GhostTraction.jl;
# its convergence on graded and random meshes is in test/convergence/ghost_traction.jl.

using Test
using Bramble
import Bramble
using Bramble: Dcₓ, Dcᵧ, εₕ, divₘₕ
using LinearAlgebra: Diagonal, issymmetric, norm
using Random: Xoshiro

using ..TestUtils: WITH_3D_TESTS

isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
const O = Main.GhostOracleND
isdefined(Main, :GhostTraction) || Base.include(Main, joinpath(@__DIR__, "GhostTraction.jl"))
using Main.GhostTraction: AXIS, CASE2, Kt, SIDE, UEX2, UEX3, elasticity, elasticity_space,
                          load_form, mixed_part, rest_part, traction_data, λ, μ

close(a, b; rtol = 1.0e-12) = norm(a - b) <= rtol * max(norm(b), 1)

# The facet rule by hand: the identity on the padded values of every component, but each
# ghost of `facets` replaced by the quadratic through the three nodes inward along the
# normal, extrapolated to the ghost (at the first spacing mirrored).
function extrapolation(Ωₕ, facets)
    xs = O.axis_points(Ωₕ)
    D = length(xs)
    np = map(length, xs) .+ 2
    L = LinearIndices(np)
    m = prod(np)
    px = map(x -> [2x[1] - x[2]; x; 2x[end] - x[end - 1]], xs)
    X = Matrix{Float64}(Diagonal(ones(D * m)))
    for c in 0:(D - 1), F in facets

        a = AXIS[F]
        g, σ = SIDE[F] < 0 ? (1, 1) : (np[a], -1)
        z, y = px[a][g], ntuple(k -> px[a][g + k * σ], 3)
        w = ntuple(k -> prod((z - y[o]) / (y[k] - y[o]) for o in 1:3 if o != k), 3)
        for J in CartesianIndices(ntuple(b -> b == a ? (1:1) : (2:(np[b] - 1)), D))
            at(k) = c * m + L[CartesianIndex(ntuple(b -> b == a ? g + k * σ : J[b], D))]
            X[at(0), at(0)] = 0.0
            for k in 1:3
                X[at(0), at(k)] = w[k]
            end
        end
    end
    return X
end

# The reduced matrix and load against the dense `Eᵀ (A_rest + A_mixed X) E` and
# `Eᵀ (F_pad − (A_rest + A_mixed X) d)`, the facet rule on the traction facets `facets`,
# and the plain `Eᵀ A_pad E`, which the rule moves.
function reference(V, C, Ωₕ, facets, fl)
    D = length(V.spaces)
    PV = Bramble.padded_space(V)
    Ar = Matrix(assemble(form(PV, PV, rest_part(D))))
    Am = Matrix(assemble(form(PV, PV, mixed_part(D))))
    Ap = Ar + Am * extrapolation(Ωₕ, facets)
    Fp = assemble(form(PV, fl))
    E, d = Bramble.closure(C)
    E = Matrix(E)
    return transpose(E) * Ap * E, transpose(E) * (Fp - Ap * d), transpose(E) * (Ar + Am) * E
end

# --- The corner of two tractions -------------------------------------------------------- #

# The padded point of the physical node `I`, every facet ghosted.
padded_point(np, I) = LinearIndices(np)[I + oneunit(I)]

# The relation of component `i` on the facet of axis `d` (outward side `η`) at the corner
# node `I`, on the padded values `U`, written out from the mesh points: `η` times
# `Σⱼ Kᵢⱼ[d, d] ∂_d uⱼ + Kᵢⱼ[d, e] ∂_e uⱼ` minus the datum `g`, `∂_d` the centred difference
# over the ghost (the first spacing mirrored), `∂_e` the derivative at the node of the
# quadratic through the three nodes along the facet into the domain. The second value is
# the size of the terms, for a relative tolerance.
function corner_relation(xs, np, U, i, d, η, I, g)
    e = 3 - d
    m = prod(np)
    val(j, J) = U[(j - 1) * m + padded_point(np, J)]
    ud = CartesianIndex(ntuple(a -> a == d ? 1 : 0, 2))
    uex = CartesianIndex(ntuple(a -> a == e ? 1 : 0, 2))
    h = abs(xs[d][I[d] == 1 ? 2 : end] - xs[d][I[d] == 1 ? 1 : end - 1])
    σ = I[e] == 1 ? 1 : -1
    y = (xs[e][I[e]], xs[e][I[e] + σ], xs[e][I[e] + 2σ])
    w = (1 / (y[1] - y[2]) + 1 / (y[1] - y[3]),
        (y[1] - y[3]) / ((y[2] - y[1]) * (y[2] - y[3])),
        (y[1] - y[2]) / ((y[3] - y[1]) * (y[3] - y[2])))
    terms = Float64[]
    for j in 1:2
        ∂d = (val(j, I + ud) - val(j, I - ud)) / (2h)
        ∂e = sum(w[k] * val(j, I + (k - 1) * σ * uex) for k in 1:3)
        push!(terms, η * Kt(i, j, 2)[d, d] * ∂d, η * Kt(i, j, 2)[d, e] * ∂e)
    end
    return sum(terms) - g, max(maximum(abs, terms), abs(g))
end

const KINDS = (:random, :graded)

@testset "Traction conditions through ghost points" begin
    @testset "Reduced matrix and load: $kind" for kind in KINDS
        Ωₕ = O.make_mesh(kind, 0; dim = 2, n0 = 9)
        V, C = elasticity_space(Ωₕ, UEX2, CASE2)
        fa, fl = elasticity(2), load_form(UEX2)
        A, F = assemble(form(C, C, fa), form(C, fl))
        Aref, Fref, Aplain = reference(V, C, Ωₕ, CASE2.tfacets, fl)
        @test size(A) == size(Aref)
        @test close(Matrix(A), Aref)
        @test close(F, Fref)
        # the facet rule is on: the plain `Eᵀ A_pad E` is not what assembles
        @test norm(Matrix(A) - Aplain) > 1.0e-3 * norm(Aref)
        # the data reach the load: the traction and displacement data move `F`
        _, F0 = assemble(form(C, C, fa), form(C, v -> innerₕ(0.0, v(1))))
        @test norm(F - F0) > 1.0e-3 * norm(F)
    end

    @testset "Traction-free data" begin
        # the manufactured solution leaves `:ymin` traction-free and loads `:xmin`
        for x in (0.1, 0.5, 0.9), i in 1:2

            @test abs(traction_data(UEX2, i, (0.0, -1.0))((x, 0.0))) <= 1.0e-14
        end
        @test abs(traction_data(UEX2, 1, (-1.0, 0.0))((0.0, 0.5))) > 0.1
        @test abs(traction_data(UEX2, 2, (-1.0, 0.0))((0.0, 0.5))) > 0.1
    end

    @testset "Corner of two tractions: $kind" for kind in KINDS
        Ωₕ = O.make_mesh(kind, 0; dim = 2, n0 = 9)
        V, C = elasticity_space(Ωₕ, UEX2, CASE2)
        xs = O.axis_points(Ωₕ)
        n = map(length, xs)
        np = n .+ 2
        # every ghost closed, and the displacement nodes of `:xmax` and `:ymax` fixed
        @test Bramble.ndofs(C) == 2 * prod(n .- 1)
        E, d = Bramble.closure(C)
        I = CartesianIndex(1, 1)
        m = prod(np)
        corner_ghosts = (padded_point(np, I) - 1, padded_point(np, I) - np[1])
        for i in 1:2, p in corner_ghosts

            @test norm(E[(i - 1) * m + p, :]) > 0
        end
        A, F = assemble(form(C, C, elasticity(2)), form(C, load_form(UEX2)))
        rng = Xoshiro(20261019)
        for U in (E * rand(rng, Bramble.ndofs(C)) + d, parent(Bramble.lift(C, A \ F)))
            for (dd, Fc) in ((1, :xmin), (2, :ymin)), i in 1:2

                g = Fc === :ymin ? 0.0 : traction_data(UEX2, i, (-1.0, 0.0))((0.0, 0.0))
                r, sz = corner_relation(xs, np, U, i, dd, -1.0, I, g)
                @test abs(r) <= 1.0e-10 * max(1.0, sz)
            end
        end
    end

    @testset "Symmetry of the reduced matrix" begin
        # #793 AC 3: the reduced matrix is symmetric whenever the padded one is. A symmetric
        # padded form here: the energy form, which reads no ghost, plus a term that does.
        Ωₕ = O.make_mesh(:random, 0; dim = 2, n0 = 9)
        V, C = elasticity_space(Ωₕ, UEX2, CASE2)
        energy = (u, v) -> 2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₘₕ(u), divₘₕ(v))
        fs = (u, v) -> energy(u, v) + innerₕ(Dcₓ(u(1)), Dcₓ(v(1))) +
                       innerₕ(Dcᵧ(u(2)), Dcᵧ(v(2)))
        PV = Bramble.padded_space(V)
        Ap = Matrix(assemble(form(PV, PV, fs)))
        @test issymmetric(Ap)
        E = Matrix(Bramble.closure(C)[1])
        As = Matrix(assemble(form(C, C, fs)))
        @test close(As, transpose(E) * Ap * E)
        @test issymmetric(As)
        # the term through the ghosts reaches the reduced matrix
        Aen = Matrix(assemble(form(PV, PV, energy)))
        @test !close(As, transpose(E) * Aen * E; rtol = 1.0e-3)
        # the collocated elasticity form is a strong form (a row per node, not a Gram
        # matrix), not symmetric on the padded composite: AC 3 asks nothing of it
        Ae = Matrix(assemble(form(PV, PV, elasticity(2))))
        @test !issymmetric(Ae)
        @test norm(Ae - transpose(Ae)) > 1.0e-3 * norm(Ae)
    end

    # 3D: nightly (`WITH_3D_TESTS`). Traction on `:xmin`, displacement on the other five.
    WITH_3D_TESTS && @testset "3D: $kind" for kind in KINDS
        case = (tfacets = (:xmin,), dfacets = (:xmax, :ymin, :ymax, :zmin, :zmax),
            free = ())
        Ωₕ = O.make_mesh(kind, 0; dim = 3, n0 = 5)
        V, C = elasticity_space(Ωₕ, UEX3, case)
        fa, fl = elasticity(3), load_form(UEX3)
        A, F = assemble(form(C, C, fa), form(C, fl))
        Aref, Fref, Aplain = reference(V, C, Ωₕ, case.tfacets, fl)
        @test close(Matrix(A), Aref)
        @test close(F, Fref)
        @test norm(Matrix(A) - Aplain) > 1.0e-3 * norm(Aref)
    end
end

end # module
