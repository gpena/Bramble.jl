using Test
using Bramble
using Random
using Bramble: values, components

# The cross-weighted centered difference.
#
#   Dₕ(uₕ)(i) = (h_i / (h_i + h_{i+1})) D₋(uₕ)(x_{i+1})
#             + (h_{i+1} / (h_i + h_{i+1})) D₋(uₕ)(x_i)
#
# The same two one-sided differences the centered difference combines, weighted by the
# opposite spacings. That swap is what makes it second order on a non-uniform grid where
# Dc is first, and the two coincide when the spacing is constant.
#
# The property underneath the order is that it differences a quadratic exactly on any
# grid: with u = x², D₋(i) = x_i + x_{i-1}, and the weighted sum telescopes to
# 2x_i(h_i + h_{i+1}) over h_i + h_{i+1}. Dc does not do this unless the grid is uniform,
# which is the whole difference between the two operators.

@testset "Cross-weighted centered difference" begin
    @testset "matches the definition" begin
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                Random.seed!(20260830)
                Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, unif)
                Wₕ = gridspace(Ωₕ)
                n = npoints(Ωₕ)
                uₕ = Rₕ(Wₕ, x -> x^2 + sin(x))
                dm = values(D₋ₓ(uₕ))
                h = [spacing(Ωₕ, i) for i in 1:n]

                want = [(i == 1 || i == n) ? 0.0 :
                        (h[i] / (h[i] + h[i + 1])) * dm[i + 1] +
                        (h[i + 1] / (h[i] + h[i + 1])) * dm[i]
                        for i in 1:n]
                @test values(Dₕₓ(uₕ)) ≈ want
            end
        end
    end

    @testset "truncated at both ends" begin
        Random.seed!(20260830)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7), (true, false))
        Wₕ = gridspace(Ωₕ)
        n = npoints(Ωₕ, Tuple)
        uₕ = Rₕ(Wₕ, x -> exp(x[1]) * (x[2] + 1))

        rx = reshape(values(Dₕₓ(uₕ)), n)
        @test all(iszero, rx[1, :])
        @test all(iszero, rx[end, :])
        @test !any(iszero, rx[2:(end - 1), :])

        ry = reshape(values(Dₕᵧ(uₕ)), n)
        @test all(iszero, ry[:, 1])
        @test all(iszero, ry[:, end])
        @test !any(iszero, ry[:, 2:(end - 1)])
    end

    @testset "coincides with Dc on a uniform grid" begin
        # Both are the mean of D₋ and D₊ there; they part company only where the two
        # spacings differ.
        Ωu = mesh(domain(interval(0.0, 1.0)), 21, true)
        uu = Rₕ(gridspace(Ωu), x -> sin(3x))
        @test values(Dₕₓ(uu)) ≈ values(Dcₓ(uu))

        Random.seed!(20260830)
        Ωr = mesh(domain(interval(0.0, 1.0)), 21, false)
        ur = Rₕ(gridspace(Ωr), x -> sin(3x))
        @test !isapprox(values(Dₕₓ(ur)), values(Dcₓ(ur)))
    end

    @testset "exact on quadratics, on any grid" begin
        # The property that separates it from Dc, and the reason for the second order
        # below. Dc reproduces affine functions on any grid; this reproduces quadratics.
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                Random.seed!(20260830)
                Ωₕ = mesh(domain(interval(0.0, 1.0)), 15, unif)
                Wₕ = gridspace(Ωₕ)
                n = npoints(Ωₕ)
                x = points(Ωₕ)

                q = values(Dₕₓ(Rₕ(Wₕ, t -> 5t^2 - 2t + 1)))
                @test all(q[i] ≈ 10x[i] - 2 for i in 2:(n - 1))

                # a cubic is not reproduced, so the test above is not vacuous
                c = values(Dₕₓ(Rₕ(Wₕ, t -> t^3)))
                @test !all(c[i] ≈ 3x[i]^2 for i in 2:(n - 1))

                # and on a non-uniform grid Dc misses the quadratic, which is what the
                # cross weighting fixes
                unif || @test !all(values(Dcₓ(Rₕ(Wₕ, t -> 5t^2 - 2t + 1)))[i] ≈
                           10x[i] - 2 for i in 2:(n - 1))
            end
        end

        # every direction, in three dimensions
        Random.seed!(20260830)
        Ω3 = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 6, 4),
            (false, false, false))
        W3 = gridspace(Ω3)
        n3 = npoints(Ω3, Tuple)
        for (d, op) in ((1, Dₕₓ), (2, Dₕᵧ), (3, Dₕ₂))
            @test all(iszero, values(op(Rₕ(W3, x -> x[mod1(d + 1, 3)]))))
            r = reshape(values(op(Rₕ(W3, x -> x[d]^2))), n3)
            xd = points(Ω3)[d]
            interior = CartesianIndices(ntuple(
                k -> k == d ? (2:(n3[k] - 1)) : (1:n3[k]), 3))
            @test all(r[I] ≈ 2 * xd[I[d]] for I in interior)
        end
    end

    @testset "convergence order" begin
        # Second order on both, which is the point: Dc is first order on a non-uniform
        # grid and this is not.
        function orders(unif; steps = 4)
            Random.seed!(20260830)
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 21, unif)
            errs = Float64[]
            for k in 0:steps
                k > 0 && iterative_refinement!(Ωₕ)
                Wₕ = gridspace(Ωₕ)
                e = values(Dₕₓ(Rₕ(Wₕ, sin))) .- values(Rₕ(Wₕ, cos))
                push!(errs, maximum(abs, e[2:(end - 1)]))
            end
            return [log2(errs[k] / errs[k + 1]) for k in 1:(length(errs) - 1)]
        end

        @test all(o -> abs(o - 2.0) < 0.05, orders(true))

        # The coarsest random pair is not yet asymptotic — measured 1.18 there against
        # 1.97 and better afterwards — so only the refined ones are held to second order.
        orand = orders(false)
        @test all(o -> abs(o - 2.0) < 0.1, orand[2:end])
        @test all(>(1.0), orand)
    end

    @testset "the whole family" begin
        Random.seed!(20260830)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, false))
        Wₕ = gridspace(Ωₕ)
        Vₕ = gridspace(Ωₕ, Val(2))
        uₕ = Rₕ(Wₕ, x -> x[1] * x[2])

        @test ∇ₕ(uₕ) isa NTuple{2, VectorElement}
        @test values(∇ₕ(uₕ)[1]) == values(Dₕₓ(uₕ))
        @test values(∇ₕ(uₕ)[2]) == values(Dₕᵧ(uₕ))

        # in one dimension the tuple and the grid function coincide, as for ∇₋ₕ
        Ω1 = mesh(domain(interval(0.0, 1.0)), 7, true)
        u1 = Rₕ(gridspace(Ω1), sin)
        @test !(∇ₕ(u1) isa Tuple)
        @test values(∇ₕ(u1)) == values(Dₕₓ(u1))

        # composite grid functions apply componentwise, as the other operators do
        fs = (x -> x[1], x -> x[2]^2)
        cₕ = Rₕ(Vₕ, fs)
        scalars = (Rₕ(Wₕ, fs[1]), Rₕ(Wₕ, fs[2]))
        rₕ = Dₕₓ(cₕ)
        @test length(values(rₕ)) == length(values(cₕ))
        for k in 1:2
            @test values(components(rₕ)[k]) == values(Dₕₓ(scalars[k]))
        end
    end

    @testset "type stable and allocates only its output" begin
        Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 33, false)
        Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 8), (true, false))
        u1 = Rₕ(gridspace(Ωₕ1), sin)
        u2 = Rₕ(gridspace(Ωₕ2), x -> x[1] * x[2])

        @test @inferred(Dₕₓ(u1)) isa VectorElement
        @test @inferred(Dₕᵧ(u2)) isa VectorElement
        @test @inferred(∇ₕ(u2)) isa NTuple{2, VectorElement}

        if Base.JLOptions().code_coverage == 0
            @test alloc_test(Dₕₓ, u1) == alloc_test(similar, u1)
            @test alloc_test(Dₕᵧ, u2) == alloc_test(similar, u2)
        else
            @test_skip "Allocation comparison skipped under code coverage"
        end
    end
end
