module GhostTractionTests

# Linear elasticity in collocated form with traction and displacement facets, closed
# through ghost points (#793). Component i of the operator is
# `−μ Δₕ uᵢ − (λ + μ) Σⱼ ∂ᵢ∂ⱼ uⱼ`, the second difference of one axis for `j = i` and the
# centred mixed difference `Dcᵢ(Dcⱼ(·))` otherwise, on a composite of ghost spaces
# constrained as a whole. The traction of component i on a facet is
# `Σⱼ ∂ₙ(u(j); K = Kᵢⱼ)`, `Kᵢⱼ[d, e] = λ δᵢd δⱼe + μ (δᵢⱼ δde + δᵢe δⱼd)`; a displacement
# facet carries `u(i) => g` and `∂ₙₙ(u(i)) => h`, both from the manufactured solution.
#
# Covered: the reduced matrix and load against the dense `Eᵀ A_pad E` and
# `Eᵀ (F_pad − A_pad d)` from the padded composite; second-order H¹ convergence of each
# component on random and graded meshes with a loaded traction facet, a traction-free one
# and two displacement facets, the two tractions meeting at a corner; the corner relation
# written out by hand from the mesh points; symmetry of the reduced matrix of a symmetric
# padded form, and the padded elasticity matrix's lack of it; and, nightly, the same in 3D.
#
# A random mesh is refined, never redrawn, across levels (`O.make_mesh(:random, ℓ)`): the
# centred tangential difference of the traction is first order on a redrawn one.

using Test
using Bramble
import Bramble
using Bramble: Dcₓ, Dcᵧ, Dc₂, D̃ₓ, D̃ᵧ, D̃₂, D₋ₓ, D₋ᵧ, D₋₂, εₕ, divₘₕ
using ForwardDiff: gradient, hessian
using LinearAlgebra: issymmetric, norm
using Random: Xoshiro

using ..TestUtils: WITH_SLOW_TESTS, WITH_3D_TESTS

isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
const O = Main.GhostOracleND

const λ, μ = 2.0, 0.5

Kt(i, j, D) = [λ * (i == d) * (j == e) + μ * ((i == j) * (d == e) + (i == e) * (j == d))
               for d in 1:D, e in 1:D]

# --- The manufactured displacement and its data ---------------------------------------- #

# In 2D the traction on `:ymin` (outward normal −e₂) vanishes: `∂ᵧu₁ = ∂ₓu₂ = 0` there and
# `λ ∂ₓu₁ + (λ + 2μ) ∂ᵧu₂ = 0` through the factor `c = λ / (λ + 2μ)`.
const c = λ / (λ + 2μ)
const UE2 = (x -> sin(x[1] + 0.3) * cos(x[2]) + exp(x[1]) * x[2]^2 / 2,
    x -> -c * cos(x[1] + 0.3) * sin(x[2]) + cos(2x[1]) * x[2]^2)
const UE3 = (x -> sin(x[1] + 0.3) * cos(x[2]) * exp(x[3] / 2),
    x -> cos(x[1]) * x[2] * x[3] + exp(x[1] + x[2] / 2),
    x -> sin(x[1] + x[2]) * cos(x[3] + 0.2))

∇(f, x) = gradient(f, collect(Float64, x))
H(f, x) = hessian(f, collect(Float64, x))

# The traction of component i of `ue` on the facet with outward normal `n`.
function traction_data(ue, i, n)
    D = length(ue)
    return x -> sum(n[d] * Kt(i, j, D)[d, e] * ∇(ue[j], x)[e]
    for j in 1:D, d in 1:D, e in 1:D)
end

# The load `−μ Δuᵢ − (λ + μ) ∂ᵢ div u`.
function load(ue, i)
    D = length(ue)
    return x -> -μ * sum(H(ue[i], x)[a, a] for a in 1:D) -
                (λ + μ) * sum(H(ue[j], x)[i, j] for j in 1:D)
end

const AXIS = (xmin = 1, xmax = 1, ymin = 2, ymax = 2, zmin = 3, zmax = 3)
const SIDE = (xmin = -1.0, xmax = 1.0, ymin = -1.0, ymax = 1.0, zmin = -1.0, zmax = 1.0)
normal(F, D) = ntuple(d -> d == AXIS[F] ? SIDE[F] : 0.0, D)

# --- The form and the relations --------------------------------------------------------- #

const DC = (Dcₓ, Dcᵧ, Dc₂)
const DT = (D̃ₓ, D̃ᵧ, D̃₂)
const DM = (D₋ₓ, D₋ᵧ, D₋₂)

# `∂ₐ∂_b u`: the second difference of axis `a` when `a = b`, else the centred mixed one.
second(u, a, b) = a == b ? DT[a](DM[a](u)) : DC[min(a, b)](DC[max(a, b)](u))

function elasticity(D)
    row(u, v, i) = innerₕ(-μ * Δₕ(u(i)), v(i)) +
                   sum(innerₕ(-(λ + μ) * second(u(j), i, j), v(i)) for j in 1:D)
    return (u, v) -> sum(row(u, v, i) for i in 1:D)
end

load_form(ue) = v -> sum(innerₕ(load(ue, i), v(i)) for i in eachindex(ue))

tr(u, i, D) = sum(∂ₙ(u(j); K = Kt(i, j, D)) for j in 1:D)

# Traction on `tfacets`, the data of `ue` (or zero on `free`), displacement on `dfacets`.
function relations(ue, tfacets, dfacets; free = ())
    D = length(ue)
    tdata(F, i) = F in free ? 0.0 : traction_data(ue, i, normal(F, D))
    ∂ₙₙdata(F, i) = x -> H(ue[i], x)[AXIS[F], AXIS[F]]
    traction(u, F) = ntuple(i -> F => tr(u, i, D) => tdata(F, i), D)
    displacement(u, F) = (ntuple(i -> F => u(i) => ue[i], D)...,
        ntuple(i -> F => ∂ₙₙ(u(i)) => ∂ₙₙdata(F, i), D)...)
    return u -> (Iterators.flatten(traction(u, F) for F in tfacets)...,
        Iterators.flatten(displacement(u, F) for F in dfacets)...)
end

const CASE2 = (tfacets = (:xmin, :ymin), dfacets = (:xmax, :ymax), free = (:ymin,))

function elasticity_space(Ωₕ, ue, case)
    S = gridspace(Ωₕ; ghosts = :boundary)
    V = S^Val(length(ue))
    return V, constrained(V, relations(ue, case.tfacets, case.dfacets; free = case.free))
end

# --- Errors ----------------------------------------------------------------------------- #

# The physical values of component `i` of the padded values `U`, every facet ghosted.
function physical(U, n, i)
    np = n .+ 2
    m = prod(np)
    P = reshape(U[((i - 1) * m + 1):(i * m)], np)
    return P[ntuple(d -> 2:(np[d] - 1), length(n))...]
end

# The discrete H¹ error `(‖e‖_h² + Σₐ ‖D₋ₐ e‖₊²)^{1/2}` at the physical nodes.
function h1_error(xs, e)
    D = length(xs)
    ĥ(x) = [(i > 1 ? x[i] - x[i - 1] : 0.0) / 2 +
            (i < length(x) ? x[i + 1] - x[i] : 0.0) / 2 for i in eachindex(x)]
    along(v, a) = reshape(v, ntuple(d -> d == a ? length(v) : 1, D))
    w = reduce((p, q) -> p .* q, (along(ĥ(xs[a]), a) for a in 1:D))
    total = sum(w .* abs2.(e))
    for a in 1:D
        wa = reduce((p, q) -> p .* q,
            (b == a ? along(1 ./ diff(xs[a]), a) : along(ĥ(xs[b]), b) for b in 1:D))
        total += sum(wa .* abs2.(diff(e; dims = a)))
    end
    return sqrt(total)
end

# `hₘₐₓ` and the H¹ error of each component on `make_mesh(kind, ℓ)`, ℓ in `levels`.
function errors(ue, case, kind, levels; n0)
    D = length(ue)
    hs, errs = Float64[], NTuple{D, Float64}[]
    for ℓ in levels
        Ωₕ = O.make_mesh(kind, ℓ; dim = D, n0)
        _, C = elasticity_space(Ωₕ, ue, case)
        A, F = assemble(form(C, C, elasticity(D)), form(C, load_form(ue)))
        U = parent(Bramble.lift(C, A \ F))
        xs = O.axis_points(Ωₕ)
        n = map(length, xs)
        nodes = collect(Iterators.product(xs...))
        push!(hs, maximum(x -> maximum(diff(x)), xs))
        push!(errs, ntuple(i -> h1_error(xs, physical(U, n, i) - ue[i].(nodes)), D))
    end
    return hs, errs
end

function lsq_order(hs, errs)
    X, Y = log.(hs), log.(errs)
    return sum((X .- sum(X) / length(X)) .* (Y .- sum(Y) / length(Y))) /
           sum(abs2, X .- sum(X) / length(X))
end

close(a, b; rtol = 1.0e-12) = norm(a - b) <= rtol * max(norm(b), 1)

# The reduced matrix and load against the dense `Eᵀ A_pad E`, `Eᵀ (F_pad − A_pad d)`.
function reference(V, C, fa, fl)
    PV = Bramble.padded_space(V)
    Ap = Matrix(assemble(form(PV, PV, fa)))
    Fp = assemble(form(PV, fl))
    E, d = Bramble.closure(C)
    E = Matrix(E)
    return transpose(E) * Ap * E, transpose(E) * (Fp - Ap * d), Ap
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
    ue = CartesianIndex(ntuple(a -> a == e ? 1 : 0, 2))
    h = abs(xs[d][I[d] == 1 ? 2 : end] - xs[d][I[d] == 1 ? 1 : end - 1])
    σ = I[e] == 1 ? 1 : -1
    y = (xs[e][I[e]], xs[e][I[e] + σ], xs[e][I[e] + 2σ])
    w = (1 / (y[1] - y[2]) + 1 / (y[1] - y[3]),
        (y[1] - y[3]) / ((y[2] - y[1]) * (y[2] - y[3])),
        (y[1] - y[2]) / ((y[3] - y[1]) * (y[3] - y[2])))
    terms = Float64[]
    for j in 1:2
        ∂d = (val(j, I + ud) - val(j, I - ud)) / (2h)
        ∂e = sum(w[k] * val(j, I + (k - 1) * σ * ue) for k in 1:3)
        push!(terms, η * Kt(i, j, 2)[d, d] * ∂d, η * Kt(i, j, 2)[d, e] * ∂e)
    end
    return sum(terms) - g, max(maximum(abs, terms), abs(g))
end

const KINDS = (:random, :graded)

@testset "Traction conditions through ghost points" begin
    @testset "Eᵀ A_pad E: $kind" for kind in KINDS
        Ωₕ = O.make_mesh(kind, 0; dim = 2, n0 = 9)
        V, C = elasticity_space(Ωₕ, UE2, CASE2)
        fa, fl = elasticity(2), load_form(UE2)
        A, F = assemble(form(C, C, fa), form(C, fl))
        Aref, Fref, _ = reference(V, C, fa, fl)
        @test size(A) == size(Aref)
        @test close(Matrix(A), Aref)
        @test close(F, Fref)
        # the data reach the load: the traction and displacement data move `F`
        _, F0 = assemble(form(C, C, fa), form(C, v -> innerₕ(0.0, v(1))))
        @test norm(F - F0) > 1.0e-3 * norm(F)
    end

    @testset "Traction-free data" begin
        # the manufactured solution leaves `:ymin` traction-free and loads `:xmin`
        for x in (0.1, 0.5, 0.9), i in 1:2

            @test abs(traction_data(UE2, i, (0.0, -1.0))((x, 0.0))) <= 1.0e-14
        end
        @test abs(traction_data(UE2, 1, (-1.0, 0.0))((0.0, 0.5))) > 0.1
        @test abs(traction_data(UE2, 2, (-1.0, 0.0))((0.0, 0.5))) > 0.1
    end

    @testset "Corner of two tractions: $kind" for kind in KINDS
        Ωₕ = O.make_mesh(kind, 0; dim = 2, n0 = 9)
        V, C = elasticity_space(Ωₕ, UE2, CASE2)
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
        A, F = assemble(form(C, C, elasticity(2)), form(C, load_form(UE2)))
        rng = Xoshiro(20261019)
        for U in (E * rand(rng, Bramble.ndofs(C)) + d, parent(Bramble.lift(C, A \ F)))
            for (dd, Fc) in ((1, :xmin), (2, :ymin)), i in 1:2

                g = Fc === :ymin ? 0.0 : traction_data(UE2, i, (-1.0, 0.0))((0.0, 0.0))
                r, sz = corner_relation(xs, np, U, i, dd, -1.0, I, g)
                @test abs(r) <= 1.0e-10 * max(1.0, sz)
            end
        end
    end

    @testset "Order, $kind meshes" for kind in KINDS
        levels = WITH_SLOW_TESTS ? (0:3) : (0:2)
        hs, errs = errors(UE2, CASE2, kind, levels; n0 = 9)
        @test maximum(hs) / minimum(hs) > 3
        for i in 1:2
            @test errs[end][i] < errs[1][i] / 8
            @test lsq_order(hs, getindex.(errs, i)) > 1.8
        end
    end

    @testset "Symmetry of the reduced matrix" begin
        # #793 AC 3: the reduced matrix is symmetric whenever the padded one is. A symmetric
        # padded form here: the energy form, which reads no ghost, plus a term that does.
        Ωₕ = O.make_mesh(:random, 0; dim = 2, n0 = 9)
        V, C = elasticity_space(Ωₕ, UE2, CASE2)
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
        V, C = elasticity_space(Ωₕ, UE3, case)
        fa, fl = elasticity(3), load_form(UE3)
        A, F = assemble(form(C, C, fa), form(C, fl))
        Aref, Fref, _ = reference(V, C, fa, fl)
        @test close(Matrix(A), Aref)
        @test close(F, Fref)
        # the order on graded meshes of 9, 17 and 33 points per axis: from 5 points per
        # axis the slope of u₃ is still pre-asymptotic
        if kind === :graded
            hs, errs = errors(UE3, case, kind, 0:2; n0 = 9)
            for i in 1:3
                @test lsq_order(hs, getindex.(errs, i)) > 1.8
            end
        end
    end
end

end # module
