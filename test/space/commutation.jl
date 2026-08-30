using Test
using Bramble
using Random
using Bramble: values

# Mixed differences commute.
#
#   D₋ₓ(D₋ᵧ(uₕ)) == D₋ᵧ(D₋ₓ(uₕ))
#
# and likewise for every pair of operators acting along different coordinates, whichever
# families they come from. This is a structural property of the stencil traversal rather
# than of any one operator: applying a stencil along `x` touches only the `x` index, so
# the order in which two directions are visited cannot matter.
#
# It is worth pinning because nothing else in the suite would catch a direction being
# swapped inside `_stencil_step`, `_stencil_ranges` or `_grid_dims`. A value test compares
# an operator against its own formula and stays self-consistent under exactly that kind of
# error; this compares two orderings that must agree independently of the formula.
#
# It also holds across the truncated slices, and that is not an accident: the finite
# differences write a zero on their truncated slice, and a zero differences to zero from
# either side, so the identity is exact everywhere rather than only in the interior.

@testset "Directional operators commute" begin
    # One from each family, so the pairs cross families as well as directions.
    PAIRS_2D = (("D₋ₓ", D₋ₓ, "D₋ᵧ", D₋ᵧ),
        ("D₊ₓ", D₊ₓ, "D₋ᵧ", D₋ᵧ),
        ("D₋ₓ", D₋ₓ, "D₊ᵧ", D₊ᵧ),
        ("Dcₓ", Dcₓ, "Dcᵧ", Dcᵧ),
        ("Dₕₓ", Dₕₓ, "Dₕᵧ", Dₕᵧ),
        ("Dstar₊ₓ", Dstar₊ₓ, "D₋ᵧ", D₋ᵧ),
        ("Dcₓ", Dcₓ, "Dₕᵧ", Dₕᵧ),
        ("M₋ₓ", M₋ₓ, "D₋ᵧ", D₋ᵧ),
        ("M₊ₓ", M₊ₓ, "M₋ᵧ", M₋ᵧ),
        ("diff₋ₓ", diff₋ₓ, "diff₊ᵧ", diff₊ᵧ),
        ("jumpₓ", jumpₓ, "jumpᵧ", jumpᵧ))

    @testset "2D" begin
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                Random.seed!(20260830)
                Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (9, 11),
                    (unif, unif))
                Wₕ = gridspace(Ωₕ)
                uₕ = Rₕ(Wₕ, x -> exp(x[1]) * sin(3x[2]) + x[1] * x[2])

                for (n1, op1, n2, op2) in PAIRS_2D
                    @testset "$n1 ∘ $n2" begin
                        @test values(op1(op2(uₕ))) ≈ values(op2(op1(uₕ)))
                    end
                end
            end
        end
    end

    @testset "3D, all three pairs of directions" begin
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                Random.seed!(20260830)
                Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))), (7, 6, 5),
                    (unif, unif, unif))
                Wₕ = gridspace(Ωₕ)
                uₕ = Rₕ(Wₕ, x -> exp(x[1]) * sin(x[2]) * (x[3] + 1) + x[1] * x[3])

                for (n1, op1, n2, op2) in (("D₋ₓ", D₋ₓ, "D₋ᵧ", D₋ᵧ),
                    ("D₋ₓ", D₋ₓ, "D₋₂", D₋₂),
                    ("D₋ᵧ", D₋ᵧ, "D₋₂", D₋₂),
                    ("Dcₓ", Dcₓ, "Dc₂", Dc₂),
                    ("Dₕᵧ", Dₕᵧ, "Dₕ₂", Dₕ₂),
                    ("M₊ₓ", M₊ₓ, "D₋₂", D₋₂))
                    @testset "$n1 ∘ $n2" begin
                        @test values(op1(op2(uₕ))) ≈ values(op2(op1(uₕ)))
                    end
                end
            end
        end
    end

    @testset "on composite grid functions" begin
        # The composite dispatch applies the operator to each component in turn, so it
        # must commute for the same reason. This is the path that once wrote out of
        # bounds, so it is worth exercising here too.
        Random.seed!(20260830)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7), (true, false))
        Vₕ = gridspace(Ωₕ, Val(3))
        cₕ = Rₕ(Vₕ, (x -> x[1] * x[2], x -> sin(x[1]), x -> exp(x[2])))

        for (op1, op2) in ((D₋ₓ, D₋ᵧ), (Dcₓ, Dcᵧ), (Dₕₓ, Dₕᵧ), (M₋ₓ, D₊ᵧ))
            @test values(op1(op2(cₕ))) ≈ values(op2(op1(cₕ)))
        end
    end

    @testset "the test is not vacuous" begin
        # Two operators along the SAME direction do not commute in general, so a version
        # of the check that compared any two operators would pass for the wrong reason.
        Random.seed!(20260830)
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 17, false)
        uₕ = Rₕ(gridspace(Ωₕ), x -> exp(x) + x^3)

        @test !isapprox(values(D₋ₓ(M₋ₓ(uₕ))), values(M₋ₓ(D₊ₓ(uₕ))))
        @test !isapprox(values(Dcₓ(D₋ₓ(uₕ))), values(Dₕₓ(D₋ₓ(uₕ))))
    end
end
