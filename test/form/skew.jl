module FormSkewTests

using Test
using Bramble
using LinearAlgebra: Diagonal, dot, norm
using Random
using Bramble: weights, Innerh, Dcₓ, Dcᵧ, Dc₂, skew_symmetric
using ..TestUtils: @test_allocs, WITH_3D_TESTS

# Skew-symmetric split forms (gpena/Bramble.jl#165).
#
# The property under test is algebraic, not asymptotic: the assembled matrix is exactly
# skew-symmetric, so every assertion below is against zero at machine precision rather than
# against a tolerance that a coarser mesh would blow. The one convergence statement here is
# about the *time integrator*, and it is there to show that the drift a time-stepped run
# shows is the integrator's and not the operator's.

@testset "Skew-symmetric split forms" begin
    Random.seed!(20260918)

    @testset "assembled matrix is exactly skew" begin
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
            # exactly skew is not enough: a zero matrix, or one direction dropped or paired
            # with the wrong coefficient, is skew too; ½(N - Nᵀ) of the naive form pins it
            N2 = Matrix(assemble(form(W2, W2,
                (u, v) -> innerₕ(w2[1] * Dcₓ(u) + w2[2] * Dcᵧ(u), v))))
            @test M2 ≈ 0.5 * (N2 - N2')
            # a composite field is its components, every direction summed
            wc2 = Rₕ(W2^Val(2), x -> (sin(x[2]), cos(x[1])))
            @test Matrix(assemble(form(W2, W2, skew_symmetric(wc2)))) ≈ M2

            # The 3D half runs nightly (`slow`).
            if WITH_3D_TESTS
                W3 = gridspace(mesh(
                    domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
                    (7, 6, 5), (false, false, false)))
                w3 = (Rₕ(W3, x -> x[2]), Rₕ(W3, x -> x[3]), Rₕ(W3, x -> x[1]))
                M3 = Matrix(assemble(form(W3, W3, skew_symmetric(w3))))
                @test M3 + M3' == zeros(size(M3))
                N3 = Matrix(assemble(form(W3, W3,
                    (u, v) -> innerₕ(w3[1] * Dcₓ(u) + w3[2] * Dcᵧ(u) + w3[3] * Dc₂(u), v))))
                @test M3 ≈ 0.5 * (N3 - N3')
                wc3 = Rₕ(W3^Val(3), x -> (x[2], x[3], x[1]))
                @test Matrix(assemble(form(W3, W3, skew_symmetric(wc3)))) ≈ M3
            end
        end

        @testset "Float32 stays Float32 (#522)" begin
            # The Float64 twin reuses the Float32 mesh's points, so the two matrices
            # describe the same grid and differ only by rounding.
            Ω32 = mesh(domain(interval(0.0f0, 1.0f0)), 21, false)
            Ω64 = mesh(domain(interval(0.0, 1.0)), 21, true)
            Bramble.set_points!(Ω64, Float64.(points(Ω32)))
            W32, W64 = gridspace(Ω32), gridspace(Ω64)
            w32 = Rₕ(W32, x -> 1.0f0 + x[1]^2)
            w64 = Rₕ(W64, x -> 1.0 + x[1]^2)
            @test eltype(w32) === Float32
            M32 = Matrix(assemble(form(W32, W32, skew_symmetric(w32))))
            M64 = Matrix(assemble(form(W64, W64, skew_symmetric(w64))))
            @test eltype(M32) === Float32
            @test M32 + M32' == zeros(Float32, size(M32))
            @test isapprox(M32, M64; rtol = 10 * eps(Float32))
        end
    end

    @testset "skew part of the advective form" begin
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

    @testset "operator and coefficient spellings agree" begin
        Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 15, true))
        wₕ = Rₕ(Wₕ, x -> 2.0 - x[1])
        @test assemble(form(Wₕ, Wₕ, skew_symmetric(wₕ))) ≈
              assemble(form(Wₕ, Wₕ, skew_symmetric(x -> wₕ * Dcₓ(x))))

        # a 1-tuple in 1D means the same field as the bare grid function
        @test assemble(form(Wₕ, Wₕ, skew_symmetric((wₕ,)))) ≈
              assemble(form(Wₕ, Wₕ, skew_symmetric(wₕ)))
    end

    @testset "advection operator is concrete" begin
        # each direction must reach `Dc` as a literal `Val`, or every build dispatches
        # dynamically and compiles abstract `simplify_ast` paths
        for D in 1:2
            Ω = D == 1 ? mesh(domain(interval(0.0, 1.0)), 5, false) :
                mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 5),
                (false, false))
            Wₕ = gridspace(Ω)
            A = Bramble._advection_operator(ntuple(_ -> element(Wₕ, 1.0), D))
            rt = Base.return_types(A, (typeof(Bramble.trial_function(Wₕ)),))
            @test isconcretetype(only(rt))
        end
    end

    @testset "a ghost-space field is refused (#1097)" begin
        # its values are one per degree of freedom, the walk reads one per padded point
        Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (false, false))
        S = gridspace(Ω; ghosts = :boundary)
        e = element(S)
        e .= 1.0
        ec = element(S × S)
        ec .= 1.0
        for w in (e, (e, e), ec)
            err = try
                form(S, S, skew_symmetric(w))
                nothing
            catch x
                x
            end
            @test err isa ArgumentError
            @test occursin("padded_space", sprint(showerror, err))
        end
        # the padded space's element is the spelling the refusal names: Eᵀ A_pad E
        P = Bramble.padded_space(S)
        c = Rₕ(P, x -> 1.0 + x[1])
        E, _ = Bramble.closure(S)
        M = Matrix(assemble(form(S, S, skew_symmetric(c))))
        Mp = Matrix(assemble(form(P, P, skew_symmetric(c))))
        @test size(M) == (ndofs(S), ndofs(S))
        @test M ≈ E' * Mp * E
        @test norm(M) > 0
    end

    @testset "It composes with a symmetric term" begin
        # added to a diffusion term, the symmetric part of the sum is the diffusion alone:
        # the convective part contributes nothing to it, which is the whole point
        W2 = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (11, 9),
            (false, false)))
        w2 = (Rₕ(W2, x -> sin(x[2])), Rₕ(W2, x -> cos(x[1])))
        a = form(W2, W2, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + skew_symmetric(w2)(u, v))
        A = Matrix(assemble(a))
        diffusion = Matrix(assemble(form(W2, W2, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))))
        @test 0.5 * (A + A') ≈ diffusion
        # and the skew part survives the sum: dropping it would leave A == diffusion
        @test A - diffusion ≈ Matrix(assemble(form(W2, W2, skew_symmetric(w2))))

        # and it refills in place at zero allocations, like any other form
        Am = assemble(a)
        @test_allocs assemble!(Am, a)
    end

    @testset "Burgers: zero semidiscrete energy rate" begin
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
