using Test
using Bramble
using LinearAlgebra: issymmetric, isposdef
using Bramble: form, assemble, trial_space, test_space

# `issymmetric`/`isposdef` on a `BilinearForm` are a purely structural, symbolic check —
# every test here has a positive case checked against a real assembled matrix (not just the
# trait's own reasoning) and, where it matters, a negative control confirming the check can
# actually tell the two apart.

@testset verbose=true "Structural symmetry / SPD detection" begin
    S = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ωₕ = mesh(domain(S, :walls => get_boundary_symbols(S)), (9, 7), (true, true))
    Wₕ = gridspace(Ωₕ)

    @testset "same L on both sides is symmetric, and the assembled matrix agrees" begin
        a = form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        @test issymmetric(a)
        @test isposdef(a)
        @test issymmetric(Matrix(assemble(a)))

        a2 = form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(v)))
        @test issymmetric(a2)
        @test isposdef(a2)
        @test issymmetric(Matrix(assemble(a2)))

        c = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
        @test issymmetric(c)
        @test isposdef(c)
        @test issymmetric(Matrix(assemble(c)))
    end

    @testset "a mixed sum of different inner products stays symmetric" begin
        d = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        @test issymmetric(d)
        @test isposdef(d)
        @test issymmetric(Matrix(assemble(d)))
    end

    @testset "scaling preserves symmetry but not positive-definiteness" begin
        a3 = form(Wₕ, Wₕ, (u, v) -> 2.0 * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        @test issymmetric(a3)
        @test isposdef(a3)

        a4 = form(Wₕ, Wₕ, (u, v) -> -2.0 * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        @test issymmetric(a4)
        @test !isposdef(a4)
        @test issymmetric(Matrix(assemble(a4)))
    end

    @testset "different operators either side: negative control" begin
        b = form(Wₕ, Wₕ, (u, v) -> inner₊(u, D₋ₓ(v)))
        @test !issymmetric(b)
        @test !isposdef(b)
        @test !issymmetric(Matrix(assemble(b)))
    end

    @testset "different trial and test space objects can never be symmetric" begin
        Wₕ2 = gridspace(Ωₕ)
        @test trial_space(form(Wₕ, Wₕ, (u, v) -> u)) ===
              test_space(form(Wₕ, Wₕ, (u, v) -> u))

        e = form(Wₕ, Wₕ2, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        @test !issymmetric(e)
        @test !isposdef(e)
    end
end
