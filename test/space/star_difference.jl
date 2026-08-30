using Test
using Bramble
using Random
using Bramble: values, components, star_spacings, StarSpacings

# The starred forward difference and the identity it exists for.
#
#   Dstar₊(uₕ)(i) = (u(x_{i+1}) - u(x_i)) / ((h_i + h_{i+1}) / 2)
#
# It is the forward difference over the averaged spacing rather than over the forward
# spacing, and it is the operator that makes the discrete integration by parts close:
#
#   innerₕ(Dstar₊ₓ(uₕ), vₕ) == -inner₊ₓ(uₕ, D₋ₓ(vₕ))
#
# whenever vₕ vanishes on the boundary.

@testset "Starred forward difference" begin
    @testset "the averaged spacing it divides by" begin
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                Random.seed!(20260830)
                Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, unif)
                n = npoints(Ωₕ)
                hs = star_spacings(Ωₕ)

                @test hs isa StarSpacings
                @test length(hs) == n - 1
                @test all(hs[i] ≈ (spacing(Ωₕ, i) + spacing(Ωₕ, i + 1)) / 2
                for i in 1:(n - 1))

                # away from the first point this is the width of the cell around xᵢ
                @test all(hs[i] ≈ half_spacing(Ωₕ, i) for i in 2:(n - 1))
                # at the first point it is not: the cached h₁ repeats the first interval,
                # so this gives x₂ - x₁ where the cell width gives half of it
                @test hs[1] ≈ 2 * half_spacing(Ωₕ, 1)
                @test hs[1] ≈ points(Ωₕ)[2] - points(Ωₕ)[1]
            end
        end
    end

    @testset "matches the definition" begin
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                Random.seed!(20260830)
                Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, unif)
                Wₕ = gridspace(Ωₕ)
                n = npoints(Ωₕ)
                uₕ = Rₕ(Wₕ, x -> x^2 + sin(x))
                u = values(uₕ)

                want = [i == n ? 0.0 :
                        (u[i + 1] - u[i]) / ((spacing(Ωₕ, i) + spacing(Ωₕ, i + 1)) / 2)
                        for i in 1:n]
                @test values(Dstar₊ₓ(uₕ)) ≈ want

                # the last point has no forward neighbour and is truncated, as in D₊ₓ
                @test values(Dstar₊ₓ(uₕ))[n] == 0.0
            end
        end
    end

    @testset "exact on the functions it is exact on" begin
        # a constant differences to zero, and x differences to one, in every direction
        Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 6, 4),
            (true, true, true))
        Wₕ = gridspace(Ωₕ)
        n = npoints(Ωₕ, Tuple)

        @test all(iszero, values(Dstar₊ₓ(Rₕ(Wₕ, x -> 3.0))))

        for (d, op) in ((1, Dstar₊ₓ), (2, Dstar₊ᵧ), (3, Dstar₊₂))
            # a function constant along d differences to zero along d
            @test all(iszero, values(op(Rₕ(Wₕ, x -> x[mod1(d + 1, 3)]))))
            # and one linear along d differences to one, away from the truncated slice
            r = reshape(values(op(Rₕ(Wₕ, x -> x[d]))), n)
            interior = ntuple(k -> k == d ? (1:(n[k] - 1)) : (1:n[k]), 3)
            @test all(≈(1.0), r[interior...])
        end
    end

    @testset "the whole family" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, false))
        Wₕ = gridspace(Ωₕ)
        Vₕ = gridspace(Ωₕ, Val(2))
        uₕ = Rₕ(Wₕ, x -> x[1] * x[2])

        @test Dstar₊ₕ(uₕ) isa NTuple{2, VectorElement}
        @test values(Dstar₊ₕ(uₕ)[1]) == values(Dstar₊ₓ(uₕ))
        @test values(Dstar₊ₕ(uₕ)[2]) == values(Dstar₊ᵧ(uₕ))

        # in one dimension the tuple and the grid function coincide
        Ω1 = mesh(domain(interval(0.0, 1.0)), 7, true)
        u1 = Rₕ(gridspace(Ω1), sin)
        @test !(Dstar₊ₕ(u1) isa Tuple)
        @test values(Dstar₊ₕ(u1)) == values(Dstar₊ₓ(u1))

        # composite grid functions apply componentwise, as the other operators do
        fs = (x -> x[1], x -> x[2]^2)
        cₕ = Rₕ(Vₕ, fs)
        scalars = (Rₕ(Wₕ, fs[1]), Rₕ(Wₕ, fs[2]))
        rₕ = Dstar₊ₓ(cₕ)
        @test length(values(rₕ)) == length(values(cₕ))
        for k in 1:2
            @test values(components(rₕ)[k]) == values(Dstar₊ₓ(scalars[k]))
        end
    end

    @testset "type stable and allocates only its output" begin
        Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 33, false)
        Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 8), (true, false))
        u1 = Rₕ(gridspace(Ωₕ1), sin)
        u2 = Rₕ(gridspace(Ωₕ2), x -> x[1] * x[2])

        @test @inferred(Dstar₊ₓ(u1)) isa VectorElement
        @test @inferred(Dstar₊ᵧ(u2)) isa VectorElement
        @test @inferred(Dstar₊ₕ(u2)) isa NTuple{2, VectorElement}
        @test @inferred(star_spacings(Ωₕ1)) isa StarSpacings

        # the denominator is a lazy view over the cached spacings, so it costs nothing
        @test_allocs star_spacings(Ωₕ1)
        @test alloc_test(Dstar₊ₓ, u1) == alloc_test(similar, u1)
        @test alloc_test(Dstar₊ᵧ, u2) == alloc_test(similar, u2)
    end

    @testset "summation by parts" begin
        # innerₕ(Dstar₊(uₕ), vₕ) == -inner₊(uₕ, D₋(vₕ)) when vₕ vanishes on the boundary.
        #
        # Only vₕ has to vanish: the boundary term of the discrete integration by parts
        # is the product of the two, so vₕ being zero there is enough. uₕ below is
        # deliberately non-zero on the boundary to pin that.
        #
        # Compared with an absolute floor as well as a relative one. Where uₕ happens not
        # to vary along the direction being differenced both sides are zero, and a purely
        # relative comparison reports a large error on two values of order 1e-17.
        agree(a, b) = isapprox(a, b; atol = 1e-12, rtol = 1e-12)

        @testset "1D" begin
            for (lbl, unif) in (("uniform", true), ("random", false)), n in (11, 51, 201)

                Random.seed!(20260830)
                Ωₕ = mesh(domain(interval(0.0, 1.0)), n, unif)
                Wₕ = gridspace(Ωₕ)
                uₕ = Rₕ(Wₕ, x -> cos(x) + 0.7)          # not zero at the boundary
                vₕ = Rₕ(Wₕ, x -> sin(pi * x))           # zero at both ends
                @test agree(innerₕ(Dstar₊ₓ(uₕ), vₕ), -inner₊ₓ(uₕ, D₋ₓ(vₕ)))
            end
        end

        @testset "2D and 3D, every direction" begin
            u2 = x -> cos(x[1]) + 0.7 + 0.3x[2]^2 + 0.2x[1] * x[2]
            v2 = x -> sin(pi * x[1]) * sin(pi * x[2])
            u3 = x -> cos(x[1]) + 0.7 + 0.3x[2]^2 + 0.4x[3] + 0.2x[1] * x[3]
            v3 = x -> sin(pi * x[1]) * sin(pi * x[2]) * sin(pi * x[3])

            for unif in (true, false)
                Random.seed!(20260830)
                Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (21, 19),
                    (unif, unif))
                W2 = gridspace(Ω2)
                a, b = Rₕ(W2, u2), Rₕ(W2, v2)
                @test agree(innerₕ(Dstar₊ₓ(a), b), -inner₊ₓ(a, D₋ₓ(b)))
                @test agree(innerₕ(Dstar₊ᵧ(a), b), -inner₊ᵧ(a, D₋ᵧ(b)))

                Ω3 = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (11, 9, 8),
                    (unif, unif, unif))
                W3 = gridspace(Ω3)
                c, d = Rₕ(W3, u3), Rₕ(W3, v3)
                @test agree(innerₕ(Dstar₊ₓ(c), d), -inner₊ₓ(c, D₋ₓ(d)))
                @test agree(innerₕ(Dstar₊ᵧ(c), d), -inner₊ᵧ(c, D₋ᵧ(d)))
                @test agree(innerₕ(Dstar₊₂(c), d), -inner₊₂(c, D₋₂(d)))
            end
        end

        @testset "it needs vₕ to vanish, and only vₕ" begin
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 101, true)
            Wₕ = gridspace(Ωₕ)
            zero_bdry = Rₕ(Wₕ, x -> sin(pi * x))
            nonzero = Rₕ(Wₕ, x -> cos(x) + 0.7)

            sbp(uₕ, vₕ) = agree(innerₕ(Dstar₊ₓ(uₕ), vₕ), -inner₊ₓ(uₕ, D₋ₓ(vₕ)))

            @test sbp(zero_bdry, zero_bdry)
            @test sbp(nonzero, zero_bdry)      # uₕ need not vanish
            @test !sbp(zero_bdry, nonzero)     # vₕ must
            @test !sbp(nonzero, nonzero)
        end
    end
end
