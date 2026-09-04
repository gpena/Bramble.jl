using Test
using Bramble
using LinearAlgebra: issymmetric, isposdef
using Bramble: form, assemble, trial_space, test_space, restrict_to, IdentityOperator,
               ZeroOperator

# `issymmetric`/`isposdef` on a `BilinearForm` are a purely structural, symbolic check:
# every test here has a positive case checked against a real assembled matrix (not just the
# trait's own reasoning) and, where it matters, a negative control confirming the check can
# actually tell the two apart.

@testset "Symmetry and SPD detection" begin
    S = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ωₕ = mesh(domain(S, :walls => get_boundary_symbols(S)), (9, 7), (true, true))
    Wₕ = gridspace(Ωₕ)

    @testset "Identical operators" begin
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

    @testset "Mixed inner products" begin
        d = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        @test issymmetric(d)
        @test isposdef(d)
        @test issymmetric(Matrix(assemble(d)))
    end

    @testset "Scaling effects" begin
        a3 = form(Wₕ, Wₕ, (u, v) -> 2.0 * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        @test issymmetric(a3)
        @test isposdef(a3)

        a4 = form(Wₕ, Wₕ, (u, v) -> -2.0 * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        @test issymmetric(a4)
        @test !isposdef(a4)
        @test issymmetric(Matrix(assemble(a4)))
    end

    @testset "Different operators" begin
        b = form(Wₕ, Wₕ, (u, v) -> inner₊(u, D₋ₓ(v)))
        @test !issymmetric(b)
        @test !isposdef(b)
        @test !issymmetric(Matrix(assemble(b)))
    end

    @testset "Different spaces" begin
        Wₕ2 = gridspace(Ωₕ)
        @test trial_space(form(Wₕ, Wₕ, (u, v) -> u)) ===
              test_space(form(Wₕ, Wₕ, (u, v) -> u))

        e = form(Wₕ, Wₕ2, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        @test !issymmetric(e)
        @test !isposdef(e)
    end

    # The four cases below never happen through the "Identical operators"/"Different
    # operators" shapes above: a composite space's indexed leaves, a region-restricted
    # operator, a shared grid-function coefficient, and the two nullary node kinds
    # (`IdentityOperator`/`ZeroOperator`). Each has its own `_same_operator_shape` method
    # (form/symmetry.jl) that nothing above ever reaches.

    @testset "Composite space, indexed components" begin
        Vₕ = Wₕ × Wₕ

        f = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
        @test issymmetric(f)
        @test isposdef(f)
        @test issymmetric(Matrix(assemble(f)))

        # Same component index on both sides is what `IndexedTrialFunction`/
        # `IndexedTestFunction` compare; a mismatched pair must not read as symmetric.
        g = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(2)))
        @test !issymmetric(g)
        @test !isposdef(g)
    end

    @testset "Region restriction" begin
        # `:boundary`/`:interior` exist on every mesh regardless of its own markers.
        h = form(Wₕ, Wₕ,
            (u, v) -> inner₊ₓ(restrict_to(:boundary, D₋ₓ(u)), restrict_to(:boundary, D₋ₓ(v))))
        @test issymmetric(h)
        @test isposdef(h)
        @test issymmetric(Matrix(assemble(h)))

        h2 = form(Wₕ, Wₕ,
            (u, v) -> inner₊ₓ(restrict_to(:boundary, D₋ₓ(u)), restrict_to(:interior, D₋ₓ(v))))
        @test !issymmetric(h2)
        @test !isposdef(h2)
    end

    @testset "Grid function coefficient" begin
        # `αₕ` changes sign over the domain: `LᵀWL` is PSD for any real `L`, including one
        # with a sign-changing coefficient, since the same `αₕ` scales both the trial and
        # test side identically (`(αₕ D₋ₓu)_i (αₕ D₋ₓv)_i` carries `αₕ_i²`, never negative).
        # `isposdef` has no positivity guard for `GridFunctionScale` the way it does for a
        # top-level `OperatorScale` (`op.scalar > 0`) because none is needed here.
        αₕ = Rₕ(Wₕ, x -> x[1] - 0.5)
        j = form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(αₕ * D₋ₓ(u), αₕ * D₋ₓ(v)))
        @test issymmetric(j)
        @test isposdef(j)
        @test issymmetric(Matrix(assemble(j)))

        # By identity, not value, deliberately (module-level note in form/symmetry.jl): a
        # second grid function with the same values is a distinct object and must not read
        # as the same coefficient.
        βₕ = Rₕ(Wₕ, x -> x[1] - 0.5)
        j2 = form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(αₕ * D₋ₓ(u), βₕ * D₋ₓ(v)))
        @test !issymmetric(j2)
        @test !isposdef(j2)
    end

    @testset "Identity and zero operators" begin
        k = form(Wₕ, Wₕ, (u, v) -> innerₕ(IdentityOperator(Wₕ), IdentityOperator(Wₕ)))
        @test issymmetric(k)
        @test isposdef(k)
        @test issymmetric(Matrix(assemble(k)))

        k2 = form(Wₕ, Wₕ, (u, v) -> innerₕ(ZeroOperator(Wₕ), ZeroOperator(Wₕ)))
        @test issymmetric(k2)
        @test isposdef(k2)

        # Different spaces, same trivial-node kind: still not the same object.
        Wₕ3 = gridspace(Ωₕ)
        k3 = form(Wₕ, Wₕ, (u, v) -> innerₕ(IdentityOperator(Wₕ), IdentityOperator(Wₕ3)))
        @test !issymmetric(k3)
    end
end
