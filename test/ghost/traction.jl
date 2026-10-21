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
# relation at the corner of the two tractions written out from the mesh points;
# second-order H¹ convergence of each component on graded meshes with a loaded traction
# facet, a traction-free one and two displacement facets, the two tractions meeting at a
# corner; stability on random meshes; symmetry of the reduced matrix of a symmetric padded
# form, and the padded elasticity matrix's lack of it; and, nightly, the same in 3D.
#
# A random mesh is refined, never redrawn, across levels (`O.make_mesh(:random, ℓ)`), and
# carries no order gate: on these meshes the collocated form reaches only 1.77-1.84 over
# levels 0-3 with displacement on every facet, no traction at all (`n0 = 9`, seeds
# 20261014 and 7). It is held instead to the error of that all-displacement problem: the
# traction's within a factor 5 at every level.

using Test
using Bramble
import Bramble
using Bramble: Dcₓ, Dcᵧ, Dc₂, D̃ₓ, D̃ᵧ, D̃₂, D₋ₓ, D₋ᵧ, D₋₂, εₕ, divₘₕ
using ForwardDiff: gradient, hessian
using LinearAlgebra: Diagonal, issymmetric, norm
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
const UEX2 = (x -> sin(x[1] + 0.3) * cos(x[2]) + exp(x[1]) * x[2]^2 / 2,
    x -> -c * cos(x[1] + 0.3) * sin(x[2]) + cos(2x[1]) * x[2]^2)
const UEX3 = (x -> sin(x[1] + 0.3) * cos(x[2]) * exp(x[3] / 2),
    x -> cos(x[1]) * x[2] * x[3] + exp(x[1] + x[2] / 2),
    x -> sin(x[1] + x[2]) * cos(x[3] + 0.2))

∇(f, x) = gradient(f, collect(Float64, x))
H(f, x) = hessian(f, collect(Float64, x))

# The traction of component i of `uex` on the facet with outward normal `n`.
function traction_data(uex, i, n)
    D = length(uex)
    return x -> sum(n[d] * Kt(i, j, D)[d, e] * ∇(uex[j], x)[e]
    for j in 1:D, d in 1:D, e in 1:D)
end

# The load `−μ Δuᵢ − (λ + μ) ∂ᵢ div u`.
function load(uex, i)
    D = length(uex)
    return x -> -μ * sum(H(uex[i], x)[a, a] for a in 1:D) -
                (λ + μ) * sum(H(uex[j], x)[i, j] for j in 1:D)
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

# The form without its mixed differences, and those alone (the facet rule's operand).
function rest_part(D)
    row(u, v, i) = innerₕ(-μ * Δₕ(u(i)), v(i)) + innerₕ(-(λ + μ) * second(u(i), i, i), v(i))
    return (u, v) -> sum(row(u, v, i) for i in 1:D)
end
function mixed_part(D)
    term(u, v, i, j) = innerₕ(-(λ + μ) * second(u(j), i, j), v(i))
    return (u, v) -> sum(term(u, v, i, j) for i in 1:D, j in 1:D if i != j)
end
function elasticity(D)
    rest, mixed = rest_part(D), mixed_part(D)
    return (u, v) -> rest(u, v) + mixed(u, v)
end

load_form(uex) = v -> sum(innerₕ(load(uex, i), v(i)) for i in eachindex(uex))

tr(u, i, D) = sum(∂ₙ(u(j); K = Kt(i, j, D)) for j in 1:D)

# Traction on `tfacets`, the data of `uex` (or zero on `free`), displacement on `dfacets`.
function relations(uex, tfacets, dfacets; free = ())
    D = length(uex)
    tdata(F, i) = F in free ? 0.0 : traction_data(uex, i, normal(F, D))
    ∂ₙₙdata(F, i) = x -> H(uex[i], x)[AXIS[F], AXIS[F]]
    traction(u, F) = ntuple(i -> F => tr(u, i, D) => tdata(F, i), D)
    displacement(u, F) = (ntuple(i -> F => u(i) => uex[i], D)...,
        ntuple(i -> F => ∂ₙₙ(u(i)) => ∂ₙₙdata(F, i), D)...)
    return u -> (Iterators.flatten(traction(u, F) for F in tfacets)...,
        Iterators.flatten(displacement(u, F) for F in dfacets)...)
end

const CASE2 = (tfacets = (:xmin, :ymin), dfacets = (:xmax, :ymax), free = (:ymin,))
const ALL_DISPLACEMENT = (tfacets = (), dfacets = (:xmin, :xmax, :ymin, :ymax), free = ())

function elasticity_space(Ωₕ, uex, case)
    S = gridspace(Ωₕ; ghosts = :boundary)
    V = S^Val(length(uex))
    return V, constrained(V, relations(uex, case.tfacets, case.dfacets; free = case.free))
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
function errors(uex, case, kind, levels; n0, seed = 20261014)
    D = length(uex)
    hs, errs = Float64[], NTuple{D, Float64}[]
    for ℓ in levels
        Ωₕ = O.make_mesh(kind, ℓ; dim = D, n0, seed)
        _, C = elasticity_space(Ωₕ, uex, case)
        A, F = assemble(form(C, C, elasticity(D)), form(C, load_form(uex)))
        U = parent(Bramble.lift(C, A \ F))
        xs = O.axis_points(Ωₕ)
        n = map(length, xs)
        nodes = collect(Iterators.product(xs...))
        push!(hs, maximum(x -> maximum(diff(x)), xs))
        push!(errs, ntuple(i -> h1_error(xs, physical(U, n, i) - uex[i].(nodes)), D))
    end
    return hs, errs
end

function lsq_order(hs, errs)
    X, Y = log.(hs), log.(errs)
    return sum((X .- sum(X) / length(X)) .* (Y .- sum(Y) / length(Y))) /
           sum(abs2, X .- sum(X) / length(X))
end

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

    @testset "Order, graded meshes" begin
        levels = WITH_SLOW_TESTS ? (0:3) : (0:2)
        hs, errs = errors(UEX2, CASE2, :graded, levels; n0 = 9)
        @test maximum(hs) / minimum(hs) > 3
        for i in 1:2
            @test errs[end][i] < errs[1][i] / 8
            @test lsq_order(hs, getindex.(errs, i)) > 1.8
        end
    end

    # Random meshes: no order gate (see the header); the traction's error stays within 5×
    # that of displacement on every facet, through ghosts, level by level.
    @testset "Stability, random meshes, seed $seed" for seed in (20261014, 7)
        levels = WITH_SLOW_TESTS ? (0:3) : (0:2)
        hs, errs = errors(UEX2, CASE2, :random, levels; n0 = 9, seed)
        _, base = errors(UEX2, ALL_DISPLACEMENT, :random, levels; n0 = 9, seed)
        @test maximum(hs) / minimum(hs) > 3
        for ℓ in eachindex(levels), i in 1:2

            @test errs[ℓ][i] <= 5 * base[ℓ][i]
        end
        # the errors at least halve per level: a scheme stalled as the all-displacement
        # one is would pass the ratio alone
        for ℓ in 2:length(levels), i in 1:2

            @test errs[ℓ][i] < errs[ℓ - 1][i] / 2
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
        # the order on graded meshes of 9 and 17 points per axis. The 33-point level costs
        # ~280 s on one thread and is left out: measured, the slope of u₃ is 1.84 from 9 to
        # 17 and 1.96 from 17 to 33 (u₁, u₂ above 1.98), and 1.68 from 5 to 17 points
        if kind === :graded
            hs, errs = errors(UEX3, case, kind, 0:1; n0 = 9)
            for i in 1:3
                @test lsq_order(hs, getindex.(errs, i)) > 1.8
            end
        end
    end
end

end # module
