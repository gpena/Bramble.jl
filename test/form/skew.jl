module FormSkewTests

using Test
using Bramble
using LinearAlgebra: Diagonal, dot, norm
using Random
using Bramble: weights, Innerh
using ..TestUtils: @test_allocs

# Skew-symmetric split forms (gpena/Bramble.jl#165).
#
# The property under test is algebraic, not asymptotic: the assembled matrix is exactly
# skew-symmetric, so every assertion below is against zero at machine precision rather than
# against a tolerance that a coarser mesh would blow. The one convergence statement here is
# about the *time integrator*, and it is there to show that the drift a time-stepped run
# shows is the integrator's and not the operator's.

@testset "Skew-symmetric split forms" begin
    Random.seed!(20260918)

    @testset "The assembled matrix is exactly skew-symmetric" begin
        @testset "1D" begin
            for (n, unif) in ((21, true), (21, false))
                Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), n, unif))
                wₕ = Rₕ(Wₕ, x -> 1.0 + x[1]^2)
                M = Matrix(assemble(form(Wₕ, Wₕ, skew_symmetric(wₕ))))
                @test M + M' == zeros(size(M))
                uv = randn(ndofs(Wₕ))
                @test abs(dot(uv, M * uv)) < 1e-12 * norm(uv)^2
            end
        end

        @testset "2D and 3D" begin
            W2 = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (11, 9),
                (false, false)))
            w2 = (Rₕ(W2, x -> sin(x[2])), Rₕ(W2, x -> cos(x[1])))
            M2 = Matrix(assemble(form(W2, W2, skew_symmetric(w2))))
            @test M2 + M2' == zeros(size(M2))

            W3 = gridspace(mesh(
                domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
                (7, 6, 5), (true, true, true)))
            w3 = (Rₕ(W3, x -> x[2]), Rₕ(W3, x -> x[3]), Rₕ(W3, x -> x[1]))
            M3 = Matrix(assemble(form(W3, W3, skew_symmetric(w3))))
            @test M3 + M3' == zeros(size(M3))
        end
    end

    @testset "It is the skew part of the advective form" begin
        # ½(N - Nᵀ) for the naive advective N, stated as an identity rather than described:
        # this is what makes the construction *the* split form and not a different operator
        # that happens to be skew.
        Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 17, false))
        wₕ = Rₕ(Wₕ, x -> 1.0 + x[1]^2)
        M = Matrix(assemble(form(Wₕ, Wₕ, skew_symmetric(wₕ))))
        N = Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(wₕ * Dcₓ(u), v))))
        @test M ≈ 0.5 * (N - N')
        # and the naive one is genuinely not skew, so the test distinguishes the two
        @test maximum(abs, N + N') > 0.1
    end

    @testset "The operator spelling and the coefficient spelling agree" begin
        Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 15, true))
        wₕ = Rₕ(Wₕ, x -> 2.0 - x[1])
        @test assemble(form(Wₕ, Wₕ, skew_symmetric(wₕ))) ≈
              assemble(form(Wₕ, Wₕ, skew_symmetric(x -> wₕ * Dcₓ(x))))

        # a 1-tuple in 1D means the same field as the bare grid function
        @test assemble(form(Wₕ, Wₕ, skew_symmetric((wₕ,)))) ≈
              assemble(form(Wₕ, Wₕ, skew_symmetric(wₕ)))
    end

    @testset "It composes with a symmetric term" begin
        # added to a diffusion term, the symmetric part of the sum is the diffusion alone:
        # the convective part contributes nothing to it, which is the whole point
        W2 = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (11, 9),
            (true, true)))
        w2 = (Rₕ(W2, x -> sin(x[2])), Rₕ(W2, x -> cos(x[1])))
        a = form(W2, W2, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + skew_symmetric(w2)(u, v))
        A = Matrix(assemble(a))
        diffusion = Matrix(assemble(form(W2, W2, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))))
        @test 0.5 * (A + A') ≈ diffusion

        # and it refills in place at zero allocations, like any other form
        Am = assemble(a)
        @test_allocs assemble!(Am, a)
    end

    @testset "Burgers: the semidiscrete energy rate is zero" begin
        # d/dt ‖u‖²_H = -2 uᵀ M u with the mass matrix H, so the statement the scheme rests
        # on is that uᵀMu vanishes for the *current* iterate, which it does exactly. A
        # time-stepped run drifts by the integrator's own error: explicit Euler is first
        # order, and halving the step halves the drift, which is what distinguishes the
        # integrator's contribution from the operator's.
        Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 65, true))
        H = Diagonal(collect(weights(Wₕ, Innerh())))
        u0 = parent(Rₕ(Wₕ, x -> exp(-200 * (x[1] - 0.5)^2)))   # compact support, so the
        # boundary never enters

        burgers_matrix(uh) = begin
            wₕ = element(Wₕ)
            parent(wₕ) .= uh
            assemble(form(Wₕ, Wₕ, skew_symmetric(wₕ)))
        end

        # the semidiscrete rate, at the initial state and after a few steps
        uh = copy(u0)
        for _ in 1:5
            @test abs(dot(uh, burgers_matrix(uh) * uh)) < 1e-12
            uh .-= 1.0e-4 .* (H \ Vector(burgers_matrix(uh) * uh))
        end

        drift(dt, nsteps) = begin
            u = copy(u0)
            e0 = dot(u, H * u)
            for _ in 1:nsteps
                u .-= dt .* (H \ Vector(burgers_matrix(u) * u))
            end
            return abs(dot(u, H * u) - e0) / e0
        end

        d1 = drift(2.0e-4, 100)
        d2 = drift(1.0e-4, 200)
        @test d2 < 0.6 * d1          # first order in the step, so halving nearly halves it
        @test d2 < 1e-3              # and small in absolute terms over the same interval
    end
end

end # module FormSkewTests
