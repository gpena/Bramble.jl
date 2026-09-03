using Test
using Bramble
using ForwardDiff
using Bramble: source_function, SourceVector, Innerh, restrict_to, shift_op,
               _source_value, _is_source_only, values, form, assemble, assemble!

# An operator wrapped around a *source* in a linear form.
#
# `multiply_stencils_linear` keeps only the test side's offsets and multiplies the
# coefficients, so the assembly contracts the left factor by summing its coefficients and
# discarding where each one sat. That is exact only when the left stencil is a single entry
# at offset zero carrying the factor's true value — an invariant nothing stated, and one a
# source under an operator breaks: `local_stencil` composes operators by relabelling offsets
# (`shift_stencil`), which is right for a translation-invariant node and wrong for a source,
# whose coefficient *is* a value read at the current point.
#
# Before `_source_value` existed, `innerₕ(D₋ₓ(f), v)` assembled to exactly zero (the two
# relabelled copies of f(xᵢ) cancelled) and `innerₕ(M₋ₓ(f), v)` reproduced `innerₕ(f, v)`
# (they summed back to f(xᵢ)) — the operator silently dropped either way. The forms tutorial
# shipped an example of the first kind.
#
# Every check below is against the NUMERIC operator layer, which is a third, independent
# implementation of the same arithmetic: `assemble(innerₕ(Op(f), v))` must equal
# `values(Op(Rₕ(Wₕ, f))) .* weights`. Each is paired with a negative control, because a zero
# vector satisfies `isfinite`, `isa` and `≈ 0` alike — that is exactly how this went unnoticed.

@testset "operators over sources in linear forms" begin
    @testset "every operator matches the numeric layer, 1D non-uniform" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)
        Wₕ = gridspace(Ωₕ)
        f = x -> x^2 + sin(3x)
        sf = source_function(f, Val(1))
        fₕ = Rₕ(Wₕ, f)
        w = weights(Wₕ, Innerh())

        for (nm, op) in (("D₋ₓ", D₋ₓ), ("D₊ₓ", D₊ₓ), ("M₋ₓ", M₋ₓ), ("M₊ₓ", M₊ₓ),
            ("jumpₓ", jumpₓ), ("Dcₓ", Dcₓ), ("Dstar₊ₓ", Dstar₊ₓ), ("Dₕₓ", Dₕₓ))
            b = assemble(form(Wₕ, v -> innerₕ(op(sf), v)))
            @test b ≈ values(op(fₕ)) .* w                     # the oracle
            @test !all(iszero, b)                             # the control the old tests lacked
        end
    end

    @testset "2D, both directions, on a mesh non-uniform in one of them" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (false, true))
        Wₕ = gridspace(Ωₕ)
        f = x -> x[1]^2 + sin(3x[2])
        sf = source_function(f, Val(2))
        fₕ = Rₕ(Wₕ, f)
        w = weights(Wₕ, Innerh())

        for (nm, op) in (("D₋ₓ", D₋ₓ), ("D₋ᵧ", D₋ᵧ), ("D₊ₓ", D₊ₓ), ("D₊ᵧ", D₊ᵧ),
            ("M₋ₓ", M₋ₓ), ("M₋ᵧ", M₋ᵧ), ("M₊ₓ", M₊ₓ), ("M₊ᵧ", M₊ᵧ),
            ("jumpₓ", jumpₓ), ("jumpᵧ", jumpᵧ), ("Dcₓ", Dcₓ), ("Dcᵧ", Dcᵧ),
            ("Dstar₊ₓ", Dstar₊ₓ), ("Dₕₓ", Dₕₓ))
            b = assemble(form(Wₕ, v -> innerₕ(op(sf), v)))
            @test b ≈ values(op(fₕ)) .* w
            @test !all(iszero, b)
        end
    end

    @testset "nesting, scaling and sums compose" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (false, true))
        Wₕ = gridspace(Ωₕ)
        f = x -> x[1]^2 + sin(3x[2])
        sf = source_function(f, Val(2))
        fₕ = Rₕ(Wₕ, f)
        w = weights(Wₕ, Innerh())

        # a difference of an average, and an average of a difference: the outer operator has
        # to re-read the inner subtree at the shifted point, which is precisely what
        # relabelling an offset cannot do
        @test assemble(form(Wₕ, v -> innerₕ(D₋ₓ(M₋ᵧ(sf)), v))) ≈ values(D₋ₓ(M₋ᵧ(fₕ))) .* w
        @test assemble(form(Wₕ, v -> innerₕ(M₋ₓ(D₋ₓ(sf)), v))) ≈ values(M₋ₓ(D₋ₓ(fₕ))) .* w
        @test assemble(form(Wₕ, v -> innerₕ(D₋ₓ(D₋ᵧ(sf)), v))) ≈ values(D₋ₓ(D₋ᵧ(fₕ))) .* w

        # scaling by a number, and by a Ref that a caller can rebind between assemblies
        @test assemble(form(Wₕ, v -> innerₕ(3 * D₋ₓ(sf), v))) ≈ 3 .* values(D₋ₓ(fₕ)) .* w

        # a sum of two differently-operated copies of the same source, and of two different
        # sources — the addends are contracted independently
        gf = source_function(x -> x[2], Val(2))
        gₕ = Rₕ(Wₕ, x -> x[2])
        @test assemble(form(Wₕ, v -> innerₕ(D₋ₓ(sf) + M₋ₓ(sf), v))) ≈
              (values(D₋ₓ(fₕ)) .+ values(M₋ₓ(fₕ))) .* w
        @test assemble(form(Wₕ, v -> innerₕ(D₋ₓ(sf) + D₋ᵧ(gf), v))) ≈
              (values(D₋ₓ(fₕ)) .+ values(D₋ᵧ(gₕ))) .* w

        for b in (assemble(form(Wₕ, v -> innerₕ(D₋ₓ(M₋ᵧ(sf)), v))),
            assemble(form(Wₕ, v -> innerₕ(3 * D₋ₓ(sf), v))),
            assemble(form(Wₕ, v -> innerₕ(D₋ₓ(sf) + D₋ᵧ(gf), v))))
            @test !all(iszero, b)
        end
    end

    @testset "a grid function scaling the source is read at the shifted point too" begin
        # `GridFunctionScale` has the same defect as the source it wraps: its coefficient is
        # read at the current point, so a relabelled offset carries the wrong one. Under a
        # difference the two readings differ, which is what makes this a real check.
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)
        Wₕ = gridspace(Ωₕ)
        f = x -> x^2
        c = x -> 1 + 2x                       # varies, so cᵢ ≠ cᵢ₋₁
        sf = source_function(f, Val(1))
        fₕ, cₕ = Rₕ(Wₕ, f), Rₕ(Wₕ, c)
        w = weights(Wₕ, Innerh())

        # the oracle: the pointwise product restricted to the grid, then differenced
        cfₕ = Rₕ(Wₕ, x -> c(x) * f(x))
        b = assemble(form(Wₕ, v -> innerₕ(D₋ₓ(cₕ * sf), v)))
        @test b ≈ values(D₋ₓ(cfₕ)) .* w
        @test !all(iszero, b)
        # and it is genuinely different from scaling *after* the difference, so the test
        # distinguishes "read at the shifted point" from "read here"
        @test !isapprox(b, values(cₕ) .* values(D₋ₓ(fₕ)) .* w)
    end

    @testset "a shift off the grid reads as zero, matching the truncation convention" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 6, true)
        Wₕ = gridspace(Ωₕ)
        f = x -> x + 1
        sf = source_function(f, Val(1))
        fₕ = Rₕ(Wₕ, f)
        w = weights(Wₕ, Innerh())

        b = assemble(form(Wₕ, v -> innerₕ(shift_op(sf, 1, 1), v)))
        expected = [i < length(w) ? values(fₕ)[i + 1] * w[i] : zero(eltype(w))
                    for i in eachindex(w)]
        @test b ≈ expected
        @test !all(iszero, b)
    end

    @testset "a region restriction masks the value" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 6), (true, true))
        Wₕ = gridspace(Ωₕ)
        f = x -> x[1] + x[2] + 1
        sf = source_function(f, Val(2))
        fₕ = Rₕ(Wₕ, f)
        w = weights(Wₕ, Innerh())

        b = assemble(form(Wₕ, v -> innerₕ(restrict_to(:interior, D₋ₓ(sf)), v)))
        full = assemble(form(Wₕ, v -> innerₕ(D₋ₓ(sf), v)))
        @test !all(iszero, b)                       # something survives the mask
        @test b != full                             # and the mask actually removed something
        # every entry is either the unmasked one or zero — the mask selects, it does not scale
        @test all(i -> b[i] ≈ full[i] || iszero(b[i]), eachindex(b))
    end

    @testset "a VectorElement source behaves as a SourceVector under an operator" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 5), (true, false))
        Wₕ = gridspace(Ωₕ)
        f = x -> x[1] * x[2] + x[1]
        fₕ = Rₕ(Wₕ, f)
        w = weights(Wₕ, Innerh())
        sv = SourceVector{2, typeof(values(fₕ))}(values(fₕ))

        b = assemble(form(Wₕ, v -> innerₕ(D₋ₓ(sv), v)))
        @test b ≈ values(D₋ₓ(fₕ)) .* w
        @test !all(iszero, b)
    end

    @testset "the interpolated source, which is what made this reachable (point 25)" begin
        Ωbig = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 8), (true, true))
        Ωsmall = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (true, true))
        Wbig, Wsmall = gridspace(Ωbig), gridspace(Ωsmall)
        f = x -> x[1] + x[2]
        us = Rₕ(Wsmall, f)
        w = weights(Wbig, Innerh())

        # the interpolant landed on Wbig, then differenced there — the numeric spelling of
        # exactly what D₋ₓ(πₕ(us)) means symbolically
        b = assemble(form(Wbig, v -> innerₕ(D₋ₓ(πₕ(us)), v)))
        @test b ≈ values(D₋ₓ(πₕ(Wbig, us))) .* w
        @test !all(iszero, b)

        # and the composite-space spelling the forms tutorial teaches
        Vh = CompositeGridSpace((Wbig, Wsmall))
        uv = Rₕ(Vh, (x -> 0.0, f))
        bc = assemble(form(Vh, v -> innerₕ(D₋ₓ(πₕ(uv(2))), D₋ₓ(v(1)))))
        @test !all(iszero, bc)                      # it contributed nothing before this fix
    end

    @testset "the plain source path is unchanged" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 6), (true, true))
        Wₕ = gridspace(Ωₕ)
        f = x -> x[1] + x[2]
        sf = source_function(f, Val(2))
        fₕ = Rₕ(Wₕ, f)
        w = weights(Wₕ, Innerh())

        @test assemble(form(Wₕ, v -> innerₕ(sf, v))) ≈ values(fₕ) .* w
        @test assemble(form(Wₕ, v -> innerₕ(fₕ, v))) ≈ values(fₕ) .* w
        @test assemble(form(Wₕ, v -> innerₕ(2.0, v))) ≈ 2.0 .* w
    end

    @testset "the test side keeps its offsets — this fix is only about the left" begin
        # `innerₕ(f, D₋ₓ(v))` is the discrete adjoint: the coefficient at grid point J picks
        # up contributions from both I = J and I = J+1, which is exactly what the test-side
        # offsets are for. It must NOT be collapsed the way the source side is.
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 7, true)
        Wₕ = gridspace(Ωₕ)
        fₕ = Rₕ(Wₕ, x -> x + 1)
        l = form(Wₕ, v -> innerₕ(fₕ, D₋ₓ(v)))
        b = assemble(l)
        # contract against a test function: l(uₕ) = Σ wᵢ fᵢ (D₋ₓu)ᵢ, computable numerically
        uₕ = Rₕ(Wₕ, x -> x^2)
        @test l(uₕ) ≈ sum(weights(Wₕ, Innerh()) .* values(fₕ) .* values(D₋ₓ(uₕ)))
        @test !all(iszero, b)
    end

    @testset "allocations, which are part of the contract" begin
        function refill_bytes(n, build)
            Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n), (true, true))
            Wₕ = gridspace(Ωₕ)
            l = form(Wₕ, build(Wₕ))
            b = assemble(l)
            assemble!(b, l)                       # warm up
            return @allocated assemble!(b, l)
        end

        f = x -> x[1]^2 + x[2]
        plain = Wₕ -> (sf = source_function(f, Val(2)); v -> innerₕ(sf, v))
        diffed = Wₕ -> (sf = source_function(f, Val(2)); v -> innerₕ(D₋ₓ(sf), v))
        nested = Wₕ -> (sf = source_function(f, Val(2)); v -> innerₕ(D₋ₓ(M₋ᵧ(sf)), v))

        # the source-value path must not cost an allocation, at any size: the branch on
        # `_is_source_only` is decided by the operand's type and folds away
        for build in (plain, diffed, nested)
            @test refill_bytes(8, build) == 0
            @test refill_bytes(16, build) == 0
        end
    end

    @testset "differentiating through an operated source" begin
        # the element type comes from the data, so a Dual-valued source stays Dual through
        # the value path exactly as it does through the stencil path
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, true)
        Wₕ = gridspace(Ωₕ)
        w = weights(Wₕ, Innerh())

        resid(p) = begin
            sf = source_function(x -> p[1] * x^2, Val(1))
            sum(assemble(form(Wₕ, v -> innerₕ(D₋ₓ(sf), v))))
        end
        g = ForwardDiff.gradient(resid, [2.0])
        # linear in p, so the gradient is the residual at p = 1
        @test g[1] ≈ resid([1.0])
        @test isfinite(g[1])
        @test !iszero(g[1])
    end

    @testset "a node _is_source_only accepts but cannot value is a loud error" begin
        # the fallback exists so that a future node added to `_is_source_only` without a
        # `_source_value` twin fails with a message rather than silently mis-assembling,
        # which is the failure mode this whole testset exists to prevent
        @test_throws ArgumentError _source_value(Bramble.TrialFunction{1}(), nothing,
            CartesianIndex(1), nothing)
    end
end
