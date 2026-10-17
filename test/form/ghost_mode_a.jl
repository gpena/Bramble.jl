module FormGhostModeATests

# Today's forms already assemble the ghost-eliminated Neumann and Robin systems of the
# plain-matrix oracles (GhostOracle1D.jl, GhostOracleND.jl) with no ghost DOFs: matrices and
# right-hand sides agree to 1e-12 on random meshes in 1D, 2D and 3D (3D behind
# `TestUtils.WITH_3D_TESTS`).
#   Robin    inner₊(∇ₕ(u), ∇ₕ(v)) + Σ_F inner_Γ((α_F/β_F) * u, v; markers = (F,))
#            = innerₕ(f̂ₕ, v) + Σ_F inner_Γ(g_F/β_F, v; markers = (F,))
#   Neumann  inner₊(∇ₕ(u), ∇ₕ(v)) = innerₕ(f̂ₕ, v) + inner_Γ(dot(∇u, η), v)
# against the oracles' `:robin_a` and `:neumann_a` problems. Then a Helmholtz mass term,
# symmetry, solves through Bramble on graded 2D meshes (error size against the oracle's, and
# order; behind `TestUtils.WITH_SLOW_TESTS`), and the corner value of a scalar `inner_Γ` on
# the whole boundary against the per-facet flux.

using Test
using Bramble
import Bramble
using LinearAlgebra: Diagonal, dot, issymmetric, norm, ×
using SparseArrays: sparse
using ..TestUtils: WITH_3D_TESTS, WITH_SLOW_TESTS, random_mesh

isdefined(Main, :GhostOracle1D) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracle1D.jl"))
isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
const O1 = Main.GhostOracle1D
const O = Main.GhostOracleND

const tol = 1e-12
const κ = 3.0   # Helmholtz: −Δu + κu = f

relerr(A, B) = norm(A - B) / norm(B)

# The oracles' data, in one shape: α, β per facet (xmin, xmax, ymin, …), the exact
# solution, its gradient as a tuple and its Laplacian; 1D points are scalars in Bramble.
function problem_data(D)
    if D == 1
        e = O1.DEFAULT_EXACT
        return (; α = (2.0, 0.5), β = (1.5, 3.0), oracle = O1, u = e.u,
            grad = x -> (e.du(x),), lap = e.d2u)
    end
    w = O.default_exact(D)
    return (; α = O.DEFAULT_α[1:(2D)], β = O.DEFAULT_β[1:(2D)], oracle = O,
        u = x -> O.value(w, x), grad = x -> Tuple(O.gradient(w, x)),
        lap = x -> O.laplacian(w, x))
end

face(j) = O.FACETS[j]
normal_axis(j) = (j + 1) ÷ 2
outward(j) = isodd(j) ? -1.0 : 1.0

# g_F = α_F u + β_F ∇u·η_F, the Robin datum of facet j.
robin_datum(P, j) = x -> P.α[j] * P.u(x) + P.β[j] * outward(j) * P.grad(x)[normal_axis(j)]

function robin_forms(Wₕ, P, D; mass = 0.0)
    a = form(Wₕ, Wₕ,
        (u, v) -> foldl(
            (acc, j) -> acc + inner_Γ((P.α[j] / P.β[j]) * u, v; markers = (face(j),)),
            1:(2D); init = inner₊(∇ₕ(u), ∇ₕ(v)) + mass * innerₕ(u, v)))
    f̂ₕ = Rₕ(Wₕ, x -> -P.lap(x) + mass * P.u(x))
    l = form(Wₕ,
        v -> foldl(
            (acc, j) -> acc + inner_Γ(x -> robin_datum(P, j)(x) / P.β[j], v;
                markers = (face(j),)),
            1:(2D); init = innerₕ(f̂ₕ, v)))
    return a, l
end

function neumann_forms(Wₕ, P, D)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    f̂ₕ = Rₕ(Wₕ, x -> -P.lap(x))
    F = ntuple(d -> (x -> P.grad(x)[d]), D)
    l = form(Wₕ, v -> innerₕ(f̂ₕ, v) + inner_Γ(dot(F, η), v; markers = (:boundary,)))
    return a, l
end

assembled(a, l) = (sparse(assemble(a)), collect(assemble(l)))

# Oracle Helmholtz: the eliminated Robin system with f = −Δu + κu, plus κ ĥ on the
# diagonal (the mass term reads no ghost).
function oracle_helmholtz(P, Ωₕ)
    Or = P.oracle
    rels = Or.problem_relations(:robin_a, Ωₕ)
    A, F = Or.eliminated_system(rels, Ωₕ, x -> -P.lap(x) + κ * P.u(x))
    return A + κ * Diagonal(Or.mean_weights(Ωₕ)), F
end

# (‖D₋e‖_+² + ‖e‖_{H,*}²)^{1/2} on a 2D mesh, as the oracle's `discrete_error`.
function h1_error(Ωₕ, e)
    xs = O.axis_points(Ωₕ)
    ĥs = map(O1.cell_measures, O.axis_meshes(Ωₕ))
    E = reshape(e, map(length, xs))
    total = dot(O.cell_measures(Ωₕ), abs2.(e))
    total += sum((1 ./ diff(xs[1])) .* reshape(ĥs[2], 1, :) .* abs2.(diff(E; dims = 1)))
    total += sum(ĥs[1] .* reshape(1 ./ diff(xs[2]), 1, :) .* abs2.(diff(E; dims = 2)))
    return sqrt(total)
end

function slope(hs, errs)
    X, Y = log.(hs), log.(errs)
    X̄, Ȳ = sum(X) / length(X), sum(Y) / length(Y)
    return sum((X .- X̄) .* (Y .- Ȳ)) / sum(abs2, X .- X̄)
end

# Bramble's discrete solution and the oracle's error for the same problem on a graded mesh.
function bramble_errors(problem, Ωₕ)
    P = problem_data(2)
    Wₕ = gridspace(Ωₕ)
    ue = P.u.(O.node_points(Ωₕ))
    if problem === :robin_a
        A, F = assembled(robin_forms(Wₕ, P, 2)...)
        uh = pde_solve(A, F)
    else
        A, F = assembled(neumann_forms(Wₕ, P, 2)...)
        w = collect(assemble(form(Wₕ, v -> innerₕ(x -> 1.0, v))))
        uh = pde_solve(A, F; mean_zero = w)
        ue .-= dot(w, ue) / sum(w)
    end
    return h1_error(Ωₕ, uh .- ue), O.discrete_error(problem, Ωₕ)
end

# The 3D mesh is built only where the 3D cases run (`WITH_3D_TESTS`).
const MESHES = (random_mesh((11,); seed = 7), random_mesh((6, 7)),
    (WITH_3D_TESTS ? (random_mesh((4, 5, 6)),) : ())...)

@testset "Mode A from existing forms" begin
    @testset "Random meshes are non-uniform" begin
        for Ωₕ in MESHES
            D = Bramble.dim(Ωₕ)
            @test D == 1 ? !Bramble.is_uniform(Ωₕ) :
                  any(d -> !Bramble.is_uniform(Ωₕ(d)), 1:D)
        end
    end

    for (D, Ωₕ) in enumerate(MESHES)
        P = problem_data(D)
        Wₕ = gridspace(Ωₕ)
        @testset "$(D)D Robin equals :robin_a" begin
            A, F = assembled(robin_forms(Wₕ, P, D)...)
            Ao, Fo = P.oracle.system(:robin_a, Ωₕ)
            ea, ef = relerr(A, Ao), relerr(F, Fo)
            println("Mode A $(D)D robin_a: matrix ", ea, ", rhs ", ef)
            @test size(A) == size(Ao)
            @test ea <= tol
            @test ef <= tol
            @test issymmetric(A)
        end

        @testset "$(D)D Neumann equals :neumann_a" begin
            A, F = assembled(neumann_forms(Wₕ, P, D)...)
            Ao, Fo = P.oracle.system(:neumann_a, Ωₕ)
            ea, ef = relerr(A, Ao), relerr(F, Fo)
            println("Mode A $(D)D neumann_a: matrix ", ea, ", rhs ", ef)
            @test ea <= tol
            @test ef <= tol
            @test issymmetric(A)
            # the singular operator: constants in its kernel
            @test norm(A * ones(size(A, 2))) <= tol * norm(A)
        end

        @testset "$(D)D Helmholtz mass term" begin
            A, F = assembled(robin_forms(Wₕ, P, D; mass = κ)...)
            Ao, Fo = oracle_helmholtz(P, Ωₕ)
            ea, ef = relerr(A, Ao), relerr(F, Fo)
            println("Mode A $(D)D helmholtz: matrix ", ea, ", rhs ", ef)
            @test ea <= tol
            @test ef <= tol
            @test issymmetric(A)
            # the mass term is not a no-op: it moves the matrix by κ ĥ
            A0 = first(assembled(robin_forms(Wₕ, P, D)...))
            @test relerr(A - A0, κ * Diagonal(P.oracle.mean_weights(Ωₕ))) <= tol
        end

        @testset "$(D)D mean weights from innerₕ" begin
            w = collect(assemble(form(Wₕ, v -> innerₕ(x -> 1.0, v))))
            @test relerr(w, P.oracle.mean_weights(Ωₕ)) <= tol
        end
    end

    WITH_SLOW_TESTS && @testset "Graded 2D solves through Bramble" begin
        for problem in (:robin_a, :neumann_a)
            hs, errs, oerrs = Float64[], Float64[], Float64[]
            for ℓ in 0:3
                Ωₕ = O.make_mesh(:graded, ℓ; dim = 2)
                e, eo = bramble_errors(problem, Ωₕ)
                push!(hs, O.hmax(Ωₕ))
                push!(errs, e)
                push!(oerrs, eo)
            end
            r = slope(hs, errs)
            println("Mode A graded 2D $problem: order ", round(r; digits = 3),
                ", errors ", errs, ", oracle ", oerrs)
            # the same error as the oracle's solve, not only the same slope
            @test all(abs.(errs .- oerrs) .<= 1e-8 .* oerrs)
            # halving hₘₐₓ divides the error by about 4 at every step, the last included
            @test all(3.5 .< errs[1:(end - 1)] ./ errs[2:end] .< 4.5)
            @test r > 1.8
        end
    end

    @testset "Scalar g on :boundary at a corner" begin
        Ωₕ = MESHES[2]
        P = problem_data(2)
        Wₕ = gridspace(Ωₕ)
        g = x -> -P.grad(x)[1]   # ∇u·η on xmin
        F = ntuple(d -> (x -> P.grad(x)[d]), 2)
        Fs = collect(assemble(form(Wₕ, v -> inner_Γ(g, v; markers = (:boundary,)))))
        Ff = collect(assemble(form(Wₕ, v -> inner_Γ(dot(F, η), v; markers = (:boundary,)))))
        I = CartesianIndex(1, 1)
        x = O.node_point(O.axis_points(Ωₕ), I)
        ωx, ωy = O.facet_weight(Ωₕ, 1, I), O.facet_weight(Ωₕ, 2, I)
        flux = ωx * (-P.grad(x)[1]) + ωy * (-P.grad(x)[2])
        lumped = (ωx + ωy) * g(x)
        println("Mode A corner (1,1): scalar g ", Fs[1], ", per-facet flux ", Ff[1],
            ", difference ", Fs[1] - Ff[1])
        @test isapprox(Ff[1], flux; rtol = tol)
        @test isapprox(Fs[1], lumped; rtol = tol)
        @test abs(Fs[1] - Ff[1]) > 1e-3 * abs(Ff[1])
        # away from the corners of xmin both give ω_x g
        LI = LinearIndices(map(length, O.axis_points(Ωₕ)))
        rows = [LI[1, j] for j in 2:(size(LI, 2) - 1)]
        @test isapprox(Fs[rows], Ff[rows]; rtol = tol)
    end
end

end # module FormGhostModeATests
