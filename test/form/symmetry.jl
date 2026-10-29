module FormSymmetryTests

using Test
using Bramble
using Random
using ..TestUtils: WITH_3D_TESTS
using LinearAlgebra: issymmetric, isposdef, cholesky, Symmetric, issuccess, eigvals, opnorm
using SparseArrays: nonzeros
using Bramble:
               form,
               assemble,
               trial_space,
               test_space,
               restrict_to,
               shift_op,
               IdentityOperator,
               ZeroOperator,
               assemble_add!,
               D₋ₓ,
               D₊ᵧ,
               D₋ᵧ,
               Dcₓ,
               Dcᵧ,
               inner₊ᵧ,
               inner₊ₓ,
               πₕ

# `issymmetric`/`isposdef` on a `BilinearForm` are a purely structural, symbolic check:
# every test here has a positive case checked against a real assembled matrix (not just the
# trait's own reasoning) and, where it matters, a negative control confirming the check can
# actually tell the two apart.

@testset "Symmetry and SPD detection" begin
    S = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ωₕ = mesh(domain(S, :walls => boundary_symbols(S)), (9, 7), (true, true))
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

        β = Ref(1.5)
        a5 = form(Wₕ, Wₕ, (u, v) -> β * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        @test isposdef(a5) === true
        β[] = -1.5
        @test isposdef(a5) === false
        β[] = 0.0
        @test isposdef(a5) === false
        β[] = 2.0
        @test isposdef(a5) === true

        @test isposdef(form(Wₕ, Wₕ, (u, v) -> Ref(-1.0) * innerₕ(u, v))) === false
        @test isposdef(form(Wₕ, Wₕ, (u, v) -> Ref(0.0) * innerₕ(u, v))) === false
        @test isposdef(form(Wₕ, Wₕ, (u, v) -> (1.0 + 0.0im) * innerₕ(u, v))) === false

        γ = Ref(1.0)
        a6 = form(Wₕ, Wₕ, (u, v) -> γ * innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        @test isposdef(a6) === true
        γ[] = -1.0
        @test isposdef(a6) === false
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

    @testset "Indexed stencil test folds (#1030)" begin
        # The product shortcut asks whether both sides give the same stencil, and an indexed
        # leaf's stencil is the identity tap whatever its component: `u(1)` against `v(2)`
        # answers `true` by type, so the test folds to a constant. `issymmetric` still asks
        # whether they are the same operator, and they are not.
        Ωn = mesh(domain(S), (9, 11), (true, true))
        pts(n, a) = collect(range(0.0, 1.0; length = n)) .^ (1.2 + 0.2a)
        Bramble.change_points!(Ωn, (pts(9, 1), pts(11, 2)))
        Vₕ = gridspace(Ωn)^Val(2)
        product(a) = only(l[2]
        for l in Bramble._kron_leaves(Bramble.resolve_form_ast(a), ())
        if l[2] isa Bramble.BilinearProduct)

        e = form(Vₕ, Vₕ, (u, v) -> innerₕ(D₋ₓ(u(1)), D₋ₓ(v(2))))
        p = product(e)
        @test p.left_op isa
              Bramble.BackwardDifference{<:Any, <:Any, <:Bramble.IndexedTrialFunction}
        ci = only(code_typed(Bramble._same_operator_shape,
            (typeof(p.left_op), typeof(p.right_op), Bramble._SameStencil)))[1]
        # A coverage run (CI) adds a `code_coverage_effect` per line ahead of the return
        code = filter(s -> !Meta.isexpr(s, :code_coverage_effect), ci.code)
        @test length(code) == 1 && code[1] isa Core.ReturnNode && code[1].val === true
        @test !Bramble._same_operator_shape(p.left_op, p.right_op)
        @test !issymmetric(e)
        @test !isposdef(e)

        # The shortcut's values: a coefficient of ones on the test side forces the general
        # product path, which must assemble the same matrix.
        onesₕ = Rₕ(gridspace(Ωn), x -> 1.0)
        e_general = form(Vₕ, Vₕ, (u, v) -> innerₕ(D₋ₓ(u(1)), onesₕ * D₋ₓ(v(2))))
        q = product(e_general)
        @test !Bramble._same_operator_shape(q.left_op, q.right_op, Bramble._SameStencil())
        @test Matrix(assemble(e)) == Matrix(assemble(e_general))
        @test !iszero(Matrix(assemble(e)))
    end

    @testset "Region restriction" begin
        # `:boundary`/`:interior` exist on every mesh regardless of its own markers.
        h = form(
            Wₕ,
            Wₕ,
            (u, v) -> inner₊ₓ(restrict_to(:boundary, D₋ₓ(u)), restrict_to(:boundary, D₋ₓ(v)))
        )
        @test issymmetric(h)
        @test isposdef(h)
        @test issymmetric(Matrix(assemble(h)))

        h2 = form(
            Wₕ,
            Wₕ,
            (u, v) -> inner₊ₓ(restrict_to(:boundary, D₋ₓ(u)), restrict_to(:interior, D₋ₓ(v)))
        )
        @test !issymmetric(h2)
        @test !isposdef(h2)
    end

    @testset "Shifted operators (#65)" begin
        # `shift_amount` is a field, not a type parameter, so it has to be compared
        # explicitly (form/symmetry.jl) rather than folded into the same `where`-clause
        # trick used for BackwardDifference et al. Two DIFFERENT shifts must not read as the same
        # operator, since `local_stencil` on a BilinearProduct (operators/inner.jl:521-532) would take that
        # as license to evaluate one side only and mirror it, corrupting the assembled
        # matrix itself, not just the `issymmetric` trait.
        m = form(Wₕ, Wₕ, (u, v) -> innerₕ(shift_op(u, 1, 1), shift_op(v, 1, 1)))
        @test issymmetric(m)
        @test isposdef(m)
        @test issymmetric(Matrix(assemble(m)))

        m2 = form(Wₕ, Wₕ, (u, v) -> innerₕ(shift_op(u, 1, 1), shift_op(v, 1, 2)))
        @test !issymmetric(m2)
        @test !isposdef(m2)
        @test !issymmetric(Matrix(assemble(m2)))

        # The trait alone isn't the point: the assembled *values* have to be right too.
        # Wrapping the test side in a no-op scaling forces `_same_operator_shape` to its
        # generic `false` fallback (a ShiftNode and its wrapper are never the same shape),
        # which routes assembly through the always-correct general path regardless of what
        # the fast-path trait would have said. The fast path must agree with it.
        #
        # A grid-function-of-ones rather than a literal `1.0 *`: `form`
        # runs `simplify_ast`, which lifts *any* `OperatorScale`
        # sitting directly inside an inner product's argument back out to scale the whole
        # product -- so a literal `1.0 * shift_op(...)` does not build the wrapper this
        # test needs between the product and its argument. `GridFunctionScale` is not
        # something that pass touches inside a product's argument, so it still forces the
        # mismatch.
        onesₕ = Rₕ(Wₕ, x -> 1.0)
        m2_general = form(
            Wₕ, Wₕ, (u, v) -> innerₕ(shift_op(u, 1, 1), onesₕ * shift_op(v, 1, 2))
        )
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

    # `InnerGammaNormal` weighs a minimum facet by `-ĥ`, so a product weighted by the
    # signed normal is indefinite even with matching sides (#865). Graded mesh.
    @testset "Signed normal weight (#865)" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 5), (false, true))
        Wn = gridspace(Ωₕ)
        for markers in ((:xmin,), (:ymin,), (:xmin, :xmax))
            a = form(Wn, Wn, (u, v) -> inner_Γ(u * η[1] + u * η[2], v; markers))
            @test isposdef(a) === false
        end
        a = form(Wn, Wn, (u, v) -> inner_Γ(u * η[1], v; markers = (:xmin,)))
        @test !isposdef(Matrix(assemble(a)))
        @test isposdef(a) === false
        # Unsigned weights keep answering `true`: the plain surface and the cell products.
        for f in ((u, v) -> inner_Γ(u, v; markers = (:xmin, :ymax)),
            (u, v) -> innerₕ(u, v), (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
            b = form(Wn, Wn, f)
            @test isposdef(b) === true
            M = Matrix(assemble(b))
            @test minimum(eigvals(Symmetric(M))) >= -1e-10 * opnorm(M, 1)
        end
        # `InnerGammaFacet` weighs a facet by `∏ ĥ_e`, its sign on the `FacetOperator`: the
        # padded form of `inner_Γ(∂ₙ(u), ∂ₙ(v))` keeps answering `true`.
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        for markers in ((:xmin,), (:xmin, :ymax))
            p = form(Wg, Wg, (u, v) -> inner_Γ(∂ₙ(u), ∂ₙ(v); markers)).padded
            @test isposdef(p) === true
            M = Matrix(assemble(p))
            @test minimum(eigvals(Symmetric(M))) >= -1e-10 * opnorm(M, 1)
        end
    end

    # Assembly does not conjugate the test side, so a non-real coefficient inside a side
    # gives `LᵀGWGL`, which `g = im` makes negative definite (#687). Graded mesh.
    @testset "Non-real coefficient in a side (#687)" begin
        I = interval(0.0, 1.0)
        Wg = gridspace(mesh(domain(I × I), (7, 6), (false, false)))
        g = element(Wg, ComplexF64)
        parent(g) .= im
        a = form(Wg, Wg, (u, v) -> innerₕ(g * u, g * v))
        @test !isposdef(Matrix(assemble(a)))
        @test isposdef(a) === false
        @test issymmetric(a)

        # Deep inside a side: under a difference, in a sum, as a scalar or a `Ref`.
        c = Ref(1.0 + 0.0im)
        for f in ((u, v) -> innerₕ(D₋ₓ(g * u), D₋ₓ(g * v)),
            (u, v) -> innerₕ(D₋ₓ(u + g * D₋ᵧ(u)), D₋ₓ(v + g * D₋ᵧ(v))),
            (u, v) -> innerₕ(D₋ₓ(im * u), D₋ₓ(im * v)),
            (u, v) -> innerₕ(D₋ₓ(c * u), D₋ₓ(c * v)))
            @test isposdef(form(Wg, Wg, f)) === false
        end
        @test !isposdef(Matrix(assemble(form(
            Wg, Wg, (u, v) -> innerₕ(D₋ₓ(im * u), D₋ₓ(im * v))))))

        # Real coefficients, sign-changing or negative, keep answering `true`.
        r = Rₕ(Wg, x -> x[1] - 0.5)
        d = Ref(-2.0)
        for f in ((u, v) -> innerₕ(r * u, r * v),
            (u, v) -> innerₕ(D₋ₓ(r * u), D₋ₓ(r * v)),
            (u, v) -> innerₕ(D₋ₓ(u + r * D₋ᵧ(u)), D₋ₓ(v + r * D₋ᵧ(v))),
            (u, v) -> innerₕ(D₋ₓ(d * u), D₋ₓ(d * v)),
            (u, v) -> innerₕ(u, v))
            a_r = form(Wg, Wg, f)
            @test isposdef(a_r) === true
            M = Matrix(assemble(a_r))
            @test M ≈ transpose(M)
            @test minimum(eigvals(Symmetric(M))) >= -1e-10 * opnorm(M, 1)
        end
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

    @testset "Transposed pairs" begin
        g1 = (u, v) -> innerₕ(D₋ₓ(u), D₊ᵧ(v))
        g2 = (u, v) -> innerₕ(D₊ᵧ(u), D₋ₓ(v))
        pair = form(Wₕ, Wₕ, (u, v) -> g1(u, v) + g2(u, v))
        @test issymmetric(pair)
        @test !isposdef(pair)
        @test !issymmetric(form(Wₕ, Wₕ, g1))
        A = assemble(pair)
        @test issymmetric(Matrix(A))
        @test A ≈ assemble(form(Wₕ, Wₕ, g1)) + assemble(form(Wₕ, Wₕ, g2))

        # Coefficients: the same object on both terms, or none.
        a, b = Ref(2.0), Ref(3.0)
        @test issymmetric(form(
            Wₕ, Wₕ, (u, v) -> innerₕ(a * D₋ₓ(u), D₊ᵧ(v)) + innerₕ(a * D₊ᵧ(u), D₋ₓ(v))
        ))
        @test !issymmetric(form(
            Wₕ, Wₕ, (u, v) -> innerₕ(a * D₋ₓ(u), D₊ᵧ(v)) + innerₕ(b * D₊ᵧ(u), D₋ₓ(v))
        ))

        # Integer divisors stay nested rational scalings (#633): a pair matches on the
        # exact product of its scalings, not factor by factor.
        @test issymmetric(form(Wₕ, Wₕ, (u, v) -> (g1(u, v) / 2) / 3 + g2(u, v) / 6))
        @test issymmetric(form(Wₕ, Wₕ, (u, v) -> (g1(u, v) / 2) / 3 + (g2(u, v) / 3) / 2))
        @test !issymmetric(form(Wₕ, Wₕ, (u, v) -> g1(u, v) / 2 + g2(u, v) / 3))

        # Mixed with symmetric terms, anywhere in the sum; a lone partner-less term is not.
        @test issymmetric(form(
            Wₕ, Wₕ, (u, v) -> g1(u, v) + innerₕ(D₋ₓ(u), D₋ₓ(v)) + g2(u, v)
        ))
        @test !issymmetric(form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v)) + g1(u, v)))
        @test !issymmetric(form(Wₕ, Wₕ, (u, v) -> g1(u, v) + g2(u, v) + g1(u, v)))

        # A sum written on one side of a product.
        c = form(Wₕ, Wₕ,
            (u, v) -> innerₕ(D₋ₓ(u), D₊ᵧ(v) + D₋ᵧ(v)) + innerₕ(D₊ᵧ(u), D₋ₓ(v)) +
                      innerₕ(D₋ᵧ(u), D₋ₓ(v)))
        @test issymmetric(c)
        @test issymmetric(Matrix(assemble(c)))
    end

    # `simplify_ast` leaves a scaling, a sum or a zero inside a difference where it is, so the
    # trait compares those nodes one by one on both sides. Non-uniform mesh: a positive case
    # is checked against the assembled matrix, symmetric and positive semi-definite.
    @testset "Scalings, sums and zeros inside a side" begin
        Random.seed!(20261002)
        Ωr = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 8), (false, false))
        Wr = gridspace(Ωr)
        mat(g) = Matrix(assemble(form(Wr, Wr, g)))
        psd(M) = issymmetric(M) && minimum(eigvals(Symmetric(M))) >= -1e-10 * opnorm(M, 1)
        c = Ref(2.0)

        base = mat((u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v)))
        for (s, g) in ((2.0, (u, v) -> innerₕ(D₋ₓ(2.0 * u), D₋ₓ(2.0 * v))),
            (c[], (u, v) -> innerₕ(D₋ₓ(c * u), D₋ₓ(c * v))))
            a = form(Wr, Wr, g)
            @test issymmetric(a)
            @test isposdef(a)
            @test mat(g) ≈ s^2 * base
        end

        add = (u, v) -> innerₕ(D₋ₓ(u) + D₋ᵧ(u), D₋ₓ(v) + D₋ᵧ(v))
        @test issymmetric(form(Wr, Wr, add))
        @test isposdef(form(Wr, Wr, add))
        @test mat(add) ≈
              base + mat((u, v) -> innerₕ(D₋ᵧ(u), D₋ₓ(v))) +
              mat((u, v) -> innerₕ(D₋ₓ(u), D₋ᵧ(v))) +
              mat((u, v) -> innerₕ(D₋ᵧ(u), D₋ᵧ(v)))
        @test psd(mat(add))

        nested = (u, v) -> innerₕ(D₋ₓ(u + 2.0 * D₋ᵧ(u)), D₋ₓ(v + 2.0 * D₋ᵧ(v)))
        @test issymmetric(form(Wr, Wr, nested))
        @test isposdef(form(Wr, Wr, nested))
        @test psd(mat(nested))
        # A different scalar inside the sum: neither the trait nor the matrix is symmetric.
        skew = (u, v) -> innerₕ(D₋ₓ(u + 2.0 * D₋ᵧ(u)), D₋ₓ(v + 3.0 * D₋ᵧ(v)))
        @test !issymmetric(form(Wr, Wr, skew))
        @test !isposdef(form(Wr, Wr, skew))
        @test !issymmetric(mat(skew))
        # A different summand on one side: expanded into four products, two without a
        # transposed partner.
        mixed = (u, v) -> innerₕ(D₋ₓ(u) + D₋ᵧ(u), D₋ₓ(v) + D₊ᵧ(v))
        @test !issymmetric(form(Wr, Wr, mixed))
        @test !issymmetric(mat(mixed))

        # A zero inside a difference keeps only its dimension: zeros over two spaces of one
        # dimension are the same operator, unlike two `IdentityOperator`s.
        z = form(Wr, Wr, (u, v) -> innerₕ(D₋ₓ(ZeroOperator(Wr)), D₋ₓ(ZeroOperator(Wr))))
        @test issymmetric(z)
        @test isposdef(z)
        Wr2 = gridspace(Ωr)
        @test issymmetric(form(
            Wr, Wr, (u, v) -> innerₕ(D₋ₓ(ZeroOperator(Wr)), D₋ₓ(ZeroOperator(Wr2)))))
    end

    @testset "Non-product non-sum; distinct products" begin
        Random.seed!(20261002)
        Ωr = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 8), (false, false))
        Wr = gridspace(Ωr)
        # A bare trial function is no inner product at all.
        bare = form(Wr, Wr, (u, v) -> u)
        @test !issymmetric(bare)
        @test !isposdef(bare)
        # Transposed sides but different inner products: no pair, and on a non-uniform mesh
        # the two weights differ, so the matrix is not symmetric either.
        g = (u, v) -> innerₕ(D₋ₓ(u), D₊ᵧ(v)) + inner₊(D₊ᵧ(u), D₋ₓ(v))
        @test !issymmetric(form(Wr, Wr, g))
        @test !issymmetric(Matrix(assemble(form(Wr, Wr, g))))
    end

    @testset "Transposed pair under Ref: in assembly" begin
        Random.seed!(20261002)
        Ωr = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 8), (false, false))
        Wr = gridspace(Ωr)
        c = Ref(2.0)
        g1 = (u, v) -> innerₕ(D₋ₓ(u), D₊ᵧ(v))
        g2 = (u, v) -> innerₕ(D₊ᵧ(u), D₋ₓ(v))
        R = assemble(form(Wr, Wr, g1)) + assemble(form(Wr, Wr, g2))
        f = form(Wr, Wr, (u, v) -> c * g1(u, v) + c * g2(u, v))
        @test issymmetric(f)
        A = assemble(f)
        @test A ≈ 2.0 * R
        c[] = 5.0
        assemble!(A, f)
        @test A ≈ 5.0 * R
    end

    @testset "transposed pairs: assemble as two terms" begin
        Random.seed!(20260924)
        Ωr = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 8), (false, false))
        Wr = gridspace(Ωr)
        Vr = gridspace(Ωr, Val(2))
        # Component-indexed pairs: components matching (a transposed pair), and not (types
        # still transposed, so one kernel, with the second term's own block); then two
        # distinct leaf spaces, which records the terms apart.
        cases = (
            (Wr,
                (
                    (u, v) -> 0.5 * innerₕ(D₋ₓ(u), D₊ᵧ(v)),
                    (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v)),
                    (u, v) -> 3.0 * innerₕ(D₊ᵧ(u), D₋ₓ(v))
                )),
            (Vr,
                (
                    (u, v) -> innerₕ(Dcᵧ(u(1)), Dcₓ(v(2))),
                    (u, v) -> innerₕ(Dcₓ(u(2)), Dcᵧ(v(1))),
                    (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2))
                )),
            (Vr,
                (
                    (u, v) -> innerₕ(Dcᵧ(u(1)), Dcₓ(v(2))),
                    (u, v) -> innerₕ(Dcₓ(u(1)), Dcᵧ(v(2))),
                    (u, v) -> innerₕ(u(2), v(2))
                )),
            (Bramble.CompositeGridSpace((Wr, gridspace(Ωr))),
                (
                    (u, v) -> innerₕ(Dcᵧ(u(1)), Dcₓ(v(2))),
                    (u, v) -> innerₕ(Dcₓ(u(2)), Dcᵧ(v(1))),
                    (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2))
                ))
        )
        for (S, gs) in cases
            f = form(S, S, (u, v) -> foldl(+, map(g -> g(u, v), gs)))
            R = sum(assemble(form(S, S, g)) for g in gs)
            A = assemble(f)
            @test A ≈ R
            assemble!(A, f)
            @test A ≈ R
            B = copy(A)
            fill!(nonzeros(B), 1.0)
            B0 = Matrix(B)
            assemble_add!(B, f, 2.0)
            @test Matrix(B) ≈ B0 + 2 * Matrix(R)
        end
        # The warm refill of a pair allocates nothing.
        f = form(Vr, Vr, (u, v) -> foldl(+, map(g -> g(u, v), cases[2][2])))
        A = assemble(f)
        assemble!(A, f)
        @test (@allocated assemble!(A, f)) == 0
        @testset "distinct-leaf pair: zero allocations" begin
            S, gs = cases[4]
            f = form(S, S, (u, v) -> foldl(+, map(g -> g(u, v), gs)))
            A = assemble(f)
            assemble!(A, f)
            @test (@allocated assemble!(A, f)) == 0
        end
    end

    @testset "Numerically SPD after assembly" begin
        # `isposdef`/`issymmetric` above are a symbolic, structural check on the form
        # itself (module note at the top of this file): they confirm a shape recognized as
        # SPD, never an actual number. This asserts the numeric property that a direct
        # solver actually relies on -- that the assembled, Dirichlet-constrained Poisson
        # matrix is strictly positive-definite -- via a real Cholesky factorization on
        # several random, non-uniform meshes, in 1D/2D/3D.
        Random.seed!(20260906)
        for (D, n) in ((1, 21), (2, (9, 11)), (WITH_3D_TESTS ? ((3, (5, 6, 4)),) : ())...)
            Ωd = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D)))
            Ωr = D == 1 ? mesh(Ωd, n, false) : mesh(Ωd, n, ntuple(_ -> false, D))
            Wr = gridspace(Ωr)
            a = form(Wr, Wr, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
            l = form(Wr, v -> innerₕ(x -> 1.0, v))
            bcs = dirichlet_constraints(Ωr, :boundary => (x -> 0.0))

            A = assemble(a; dirichlet = :boundary)
            b = assemble(l; dirichlet = bcs)
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

    # A surface weight holds its faces as data (#895), so an `:xmin` and an `:xmax` term
    # share a weight type: a transposed pair needs equal weights, not equal types.
    @testset "Surface pairs on different faces (#895)" begin
        Ωs = domain(interval(0.0, 1.0) × interval(0.0, 2.0))
        Ωr = (Random.seed!(7); mesh(Ωs, (7, 6), (false, false)))
        Wr = gridspace(Ωr)
        Vr = Wr × Wr
        apart = (
            (Wr, (u, v) -> inner_Γ(D₋ₓ(u), v; markers = :xmin) +
                           inner_Γ(u, D₋ₓ(v); markers = :xmax)),
            (Wr, (u, v) -> inner_Γ(D₋ᵧ(u), D₋ₓ(v); markers = :ymin) +
                           inner_Γ(D₋ₓ(u), D₋ᵧ(v); markers = :ymax)),
            (Wr,
                (u, v) -> 2.0 * inner_Γ(D₋ₓ(u), v; markers = (:xmin, :ymin)) +
                          2.0 * inner_Γ(u, D₋ₓ(v); markers = :boundary)),
            (Vr, (u, v) -> inner_Γ(u(1), v(2); markers = :xmin) +
                           inner_Γ(u(2), v(1); markers = :xmax)))
        for (S, f) in apart
            a = form(S, S, f)
            @test !issymmetric(Matrix(assemble(a)))
            @test !issymmetric(a)
        end
        # Control: the same pair on one face is still a transposed pair.
        same = form(Wr, Wr, (u, v) -> inner_Γ(D₋ₓ(u), v; markers = :xmin) +
                                      inner_Γ(u, D₋ₓ(v); markers = :xmin))
        @test issymmetric(same)
        @test issymmetric(Matrix(assemble(same)))
    end
end

end # module FormSymmetryTests
