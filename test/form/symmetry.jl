using Test
using Bramble
using Random
using LinearAlgebra: issymmetric, isposdef, cholesky, Symmetric, issuccess
using Bramble: form, assemble, trial_space, test_space, restrict_to, shift_op,
               IdentityOperator, ZeroOperator

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

    @testset "Shifted operators (#65)" begin
        # `shift_amount` is a field, not a type parameter, so it has to be compared
        # explicitly (form/symmetry.jl) rather than folded into the same `where`-clause
        # trick used for BackwardDifference et al. Before that field comparison existed,
        # two DIFFERENT shifts read as the same operator, and local_stencil(::BilinearProduct)
        # (form/operators/inner.jl:521-532) takes that as license to evaluate one side only
        # and mirror it — corrupting the assembled matrix itself, not just the `issymmetric`
        # trait.
        m = form(Wₕ, Wₕ, (u, v) -> innerₕ(shift_op(u, 1, 1), shift_op(v, 1, 1)))
        @test issymmetric(m)
        @test isposdef(m)
        @test issymmetric(Matrix(assemble(m)))

        m2 = form(Wₕ, Wₕ, (u, v) -> innerₕ(shift_op(u, 1, 1), shift_op(v, 1, 2)))
        @test !issymmetric(m2)
        @test !isposdef(m2)
        @test !issymmetric(Matrix(assemble(m2)))

        # The trait alone isn't the point: the assembled *values* have to be right too.
        # Wrapping the test side in a no-op OperatorScale forces `_same_operator_shape` to
        # its generic `false` fallback (a ShiftNode and an OperatorScale are never the same
        # shape), which routes assembly through the always-correct general path regardless
        # of what the fast-path trait would have said. The fast path must agree with it.
        m2_general = form(
            Wₕ, Wₕ, (u, v) -> innerₕ(shift_op(u, 1, 1), 1.0 * shift_op(v, 1, 2)))
        @test Matrix(assemble(m2)) ≈ Matrix(assemble(m2_general))

        # And the fast path must NOT agree with mirroring one side, which is what the bug
        # actually did: this is the same wrong answer `assemble(m)` (matching shifts) gives.
        @test !(Matrix(assemble(m2)) ≈ Matrix(assemble(m)))
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

    @testset "Numerically SPD after assembly" begin
        # `isposdef`/`issymmetric` above are a symbolic, structural check on the form
        # itself (module note at the top of this file): they confirm a shape recognized as
        # SPD, never an actual number. This asserts the numeric property that a direct
        # solver actually relies on -- that the assembled, Dirichlet-constrained Poisson
        # matrix is strictly positive-definite -- via a real Cholesky factorization on
        # several random, non-uniform meshes, in 1D/2D/3D.
        Random.seed!(20260906)
        for (D, n) in ((1, 21), (2, (9, 11)), (3, (5, 6, 4)))
            Ωd = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D)))
            Ωr = D == 1 ? mesh(Ωd, n, false) : mesh(Ωd, n, ntuple(_ -> false, D))
            Wr = gridspace(Ωr)
            a = form(Wr, Wr, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
            l = form(Wr, v -> innerₕ(x -> 1.0, v))
            bcs = dirichlet_constraints(Bramble.set(Ωr), :boundary => (x -> 0.0))

            A = assemble(a; dirichlet_labels = :boundary)
            b = assemble(l; dirichlet_conditions = bcs, dirichlet_labels = :boundary)
            # `dirichlet_bc!` (inside `assemble`) zeros the marked rows, which on its own
            # destroys symmetry; `symmetrize!` restores it by eliminating the marked
            # columns into `b`, so the matrix Cholesky actually sees is the real, complete
            # constrained system, not merely one triangle of an asymmetric one.
            symmetrize!(A, b, Ωr, :boundary)
            @test issymmetric(Matrix(A))

            F = cholesky(Symmetric(Matrix(A)); check = false)
            @test issuccess(F)
        end
    end
end
