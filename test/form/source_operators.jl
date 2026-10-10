module FormSourceOperatorsTests

using Test
using Bramble
using ..TestUtils: WITH_AD_TESTS
# Internal since v3.0 (gpena/Bramble.jl#211): defined and documented, not exported.
import Bramble: D₊ₓ, D₊ᵧ, M₊ₓ, M₊ᵧ
using ForwardDiff
using Bramble:
               source_function,
               SourceVector,
               SourceFunction,
               Innerh,
               restrict_to,
               shift_op,
               form,
               assemble,
               assemble!,
               resolve_form_ast,
               LinearProduct,
               CompositeGridSpace,
               components,
               Dcᵧ,
               Dcₓ,
               D̃ₓ,
               D̽ₓ,
               D₋ᵧ,
               D₋ₓ,
               Mᵧ,
               Mₓ,
               inner₊₂,
               jumpᵧ,
               jumpₓ,
               weights

# An operator wrapped around a *source* in a linear form.
#
# `multiply_stencils_linear` keeps only the test side's offsets and multiplies the
# coefficients, so the assembly contracts the left factor by summing its coefficients and
# discarding where each one sat. That is exact only when the left stencil is a single entry
# at offset zero carrying the factor's true value (an invariant nothing stated, and one a
# source under an operator breaks: `local_stencil` composes operators by relabelling offsets
# (`shift_stencil`), which is right for a translation-invariant node and wrong for a source,
# whose coefficient *is* a value read at the current point).
#
# Previously, `innerₕ(D₋ₓ(f), v)` assembled to exactly zero (the two
# relabelled copies of f(xᵢ) cancelled) and `innerₕ(Mₓ(f), v)` reproduced `innerₕ(f, v)`
# (they summed back to f(xᵢ)): the operator silently dropped either way.
# Now, `_contracted_left_stencil` reads the subtree's own `local_stencil`, correct once a
# source is marked `PointDependentStencil` (`operators/interpolation.jl`).
# This file's checks pin the observable behaviour.
#
# Every check below is against the NUMERIC operator layer, which is a third, independent
# implementation of the same arithmetic: `assemble(innerₕ(Op(f), v))` must equal
# `parent(Op(Rₕ(Wₕ, f))) .* weights`. Each is paired with a negative control, because a zero
# vector satisfies `isfinite`, `isa` and `≈ 0` alike: that is exactly how this went unnoticed.

@testset "Source operators" begin
    @testset "1D numeric equivalence" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)
        Wₕ = gridspace(Ωₕ)
        f = x -> x^2 + sin(3x)
        sf = source_function(f, Val(1))
        fₕ = Rₕ(Wₕ, f)
        w = weights(Wₕ, Innerh())

        for (nm, op) in (
            ("D₋ₓ", D₋ₓ),
            ("D₊ₓ", D₊ₓ),
            ("Mₓ", Mₓ),
            ("M₊ₓ", M₊ₓ),
            ("jumpₓ", jumpₓ),
            ("Dcₓ", Dcₓ),
            ("D̃ₓ", D̃ₓ),
            ("D̽ₓ", D̽ₓ)
        )
            b = assemble(form(Wₕ, v -> innerₕ(op(sf), v)))
            @test b ≈ parent(op(fₕ)) .* w                     # the oracle
            @test !all(iszero, b)                             # the control the old tests lacked
        end
    end

    @testset "2D directional equivalence" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (false, true))
        Wₕ = gridspace(Ωₕ)
        f = x -> x[1]^2 + sin(3x[2])
        sf = source_function(f, Val(2))
        fₕ = Rₕ(Wₕ, f)
        w = weights(Wₕ, Innerh())

        for (nm, op) in (
            ("D₋ₓ", D₋ₓ),
            ("D₋ᵧ", D₋ᵧ),
            ("D₊ₓ", D₊ₓ),
            ("D₊ᵧ", D₊ᵧ),
            ("Mₓ", Mₓ),
            ("Mᵧ", Mᵧ),
            ("M₊ₓ", M₊ₓ),
            ("M₊ᵧ", M₊ᵧ),
            ("jumpₓ", jumpₓ),
            ("jumpᵧ", jumpᵧ),
            ("Dcₓ", Dcₓ),
            ("Dcᵧ", Dcᵧ),
            ("D̃ₓ", D̃ₓ),
            ("D̽ₓ", D̽ₓ)
        )
            b = assemble(form(Wₕ, v -> innerₕ(op(sf), v)))
            @test b ≈ parent(op(fₕ)) .* w
            @test !all(iszero, b)
        end
    end

    @testset "Composition & scaling" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (false, true))
        Wₕ = gridspace(Ωₕ)
        f = x -> x[1]^2 + sin(3x[2])
        sf = source_function(f, Val(2))
        fₕ = Rₕ(Wₕ, f)
        w = weights(Wₕ, Innerh())

        # a difference of an average, and an average of a difference: the outer operator has
        # to re-read the inner subtree at the shifted point, which is precisely what
        # relabelling an offset cannot do
        @test assemble(form(Wₕ, v -> innerₕ(D₋ₓ(Mᵧ(sf)), v))) ≈ parent(D₋ₓ(Mᵧ(fₕ))) .* w
        @test assemble(form(Wₕ, v -> innerₕ(Mₓ(D₋ₓ(sf)), v))) ≈ parent(Mₓ(D₋ₓ(fₕ))) .* w

        # `f` is separable, x²  +  sin(3y), so its mixed difference is mathematically zero at
        # every point (both sides here are machine-epsilon noise (~1e-16), not a value an
        # unqualified `≈`'s relative tolerance can compare meaningfully; an `atol` this loose
        # would swallow a real regression anywhere else in this file, where every other
        # comparison is against a value orders of magnitude larger)
        @test isapprox(
            assemble(form(Wₕ, v -> innerₕ(D₋ₓ(D₋ᵧ(sf)), v))),
            parent(D₋ₓ(D₋ᵧ(fₕ))) .* w;
            atol = 1e-12
        )

        # scaling by a number, and by a Ref that a caller can rebind between assemblies
        @test assemble(form(Wₕ, v -> innerₕ(3 * D₋ₓ(sf), v))) ≈ 3 .* parent(D₋ₓ(fₕ)) .* w

        # a sum of two differently-operated copies of the same source, and of two different
        # sources: the addends are contracted independently
        gf = source_function(x -> x[2], Val(2))
        gₕ = Rₕ(Wₕ, x -> x[2])
        @test assemble(form(Wₕ, v -> innerₕ(D₋ₓ(sf) + Mₓ(sf), v))) ≈
              (parent(D₋ₓ(fₕ)) .+ parent(Mₓ(fₕ))) .* w
        @test assemble(form(Wₕ, v -> innerₕ(D₋ₓ(sf) + D₋ᵧ(gf), v))) ≈
              (parent(D₋ₓ(fₕ)) .+ parent(D₋ᵧ(gₕ))) .* w

        for b in (
            assemble(form(Wₕ, v -> innerₕ(D₋ₓ(Mᵧ(sf)), v))),
            assemble(form(Wₕ, v -> innerₕ(3 * D₋ₓ(sf), v))),
            assemble(form(Wₕ, v -> innerₕ(D₋ₓ(sf) + D₋ᵧ(gf), v)))
        )
            @test !all(iszero, b)
        end
    end

    @testset "Shifted coefficient scaling" begin
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
        @test b ≈ parent(D₋ₓ(cfₕ)) .* w
        @test !all(iszero, b)
        # and it is genuinely different from scaling *after* the difference, so the test
        # distinguishes "read at the shifted point" from "read here"
        @test !isapprox(b, parent(cₕ) .* parent(D₋ₓ(fₕ)) .* w)
    end

    @testset "Boundary truncation" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 6, true)
        Wₕ = gridspace(Ωₕ)
        f = x -> x + 1
        sf = source_function(f, Val(1))
        fₕ = Rₕ(Wₕ, f)
        w = weights(Wₕ, Innerh())

        b = assemble(form(Wₕ, v -> innerₕ(shift_op(sf, 1, 1), v)))
        expected = [i < length(w) ? parent(fₕ)[i + 1] * w[i] : zero(eltype(w)) for i in eachindex(w)]
        @test b ≈ expected
        @test !all(iszero, b)

        # `shift_op` carries no mask of its own (every difference/average/jump does, and
        # that mask is what makes clamping the shifted point safe for them: the clamped,
        # possibly-wrong read gets multiplied by exactly zero). A shift by more than one point
        # makes the distinction sharp: the wrongly-clamped answer would read the *boundary
        # point's own value* rather than contribute zero, which a shift of amount 1 cannot
        # tell apart from the correct answer at every row but the last.
        b2 = assemble(form(Wₕ, v -> innerₕ(shift_op(sf, 1, 2), v)))
        expected2 = [i + 2 <= length(w) ? parent(fₕ)[i + 2] * w[i] : zero(eltype(w)) for
                     i in eachindex(w)]
        wrongly_clamped = [parent(fₕ)[min(i + 2, length(w))] * w[i] for i in eachindex(w)]
        @test b2 ≈ expected2
        @test !isapprox(b2, wrongly_clamped)
    end

    @testset "Shifted source infers (#524)" begin
        # Off the grid a shifted source used to be an empty stencil and on it a one-entry
        # one, so `local_stencil` inferred a `Union`; a Dirac's `Int` weight carried it past
        # the `LinearProduct` into the assembly loop. Both branches now keep one length, and
        # so does a shifted restricted source: outside its region it answers the zero
        # stencil `form` stored (gpena/Bramble.jl#639).
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 6, true)
        Wₕ = gridspace(Ωₕ)
        sf = source_function(x -> x + 1, Val(1))
        n = length(weights(Wₕ, Innerh()))
        shapes = (
            v -> innerₕ(shift_op(sf, 1, 1), v),
            v -> innerₕ(shift_op(dirac(0.4), 1, 1), v),
            v -> innerₕ(D₋ₓ(shift_op(sf, 1, 1)), v),
            v -> innerₕ(shift_op(restrict_to(:interior, sf), 1, 1), v)
        )
        for shape in shapes
            ast = resolve_form_ast(form(Wₕ, shape))
            for op in (ast.left_op, ast), i in (1, 3, n)  # `n`: the shift leaves the grid

                @test @inferred(Bramble.local_stencil(op, Wₕ, CartesianIndex(i), nothing, i)) isa
                      Tuple
            end
        end
    end

    @testset "Shifted Inf source off grid (#524)" begin
        # `false` is a strong zero, so the last row, whose shifted point is off the grid, is
        # exactly 0 even though the clamped read there is `Inf`; scaling by `zero(T)` instead
        # would give `Inf * 0.0 = NaN`. The row before it legitimately reads `f(1) = Inf`.
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 6, true)
        Wₕ = gridspace(Ωₕ)
        f = x -> 1 / (1 - x)
        fₕ = Rₕ(Wₕ, f)
        w = weights(Wₕ, Innerh())
        @test isinf(parent(fₕ)[end])

        b = assemble(form(Wₕ, v -> innerₕ(shift_op(source_function(f, Val(1)), 1, 1), v)))
        expected = [i < length(w) ? parent(fₕ)[i + 1] * w[i] : 0.0 for i in eachindex(w)]
        @test b[end] === 0.0
        @test isequal(b, expected)
    end

    @testset "Shifted restricted Inf source (#524)" begin
        # A restricted source shifted onto a grid point outside its region contributes
        # exactly 0 even where the source is not finite there. The `zero(T)` the restriction's
        # own tap override scales by would give `Inf * 0.0 = NaN` instead.
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 6, true)
        Wₕ = gridspace(Ωₕ)
        w = weights(Wₕ, Innerh())
        n = length(w)

        f = x -> 1 / (1 - x)                        # Inf at x = 1, a boundary point
        fₕ = Rₕ(Wₕ, f)
        b = assemble(form(Wₕ,
            v -> innerₕ(shift_op(restrict_to(:interior, source_function(f, Val(1))), 1, 1), v)))
        expected = [1 < i + 1 < n ? parent(fₕ)[i + 1] * w[i] : 0.0 for i in 1:n]
        @test b[n - 1] === 0.0                      # reads x = 1, outside :interior
        @test isequal(b, expected)

        g = x -> x < 0.5 ? NaN : x                  # NaN at interior points too
        gₕ = Rₕ(Wₕ, g)
        b2 = assemble(form(Wₕ,
            v -> innerₕ(shift_op(restrict_to(:boundary, source_function(g, Val(1))), 1, -1), v)))
        expected2 = [i - 1 in (1, n) ? parent(gₕ)[i - 1] * w[i] : 0.0 for i in 1:n]
        @test isnan(b2[2])                          # reads x = 0, on :boundary, g(0) = NaN
        @test all(i -> b2[i] === 0.0, 3:n)          # reads interior NaNs, all outside
        @test isequal(b2, expected2)
    end

    @testset "Shifted source kept in region (#524)" begin
        # A source may be undefined outside its region; shifting the restriction must not
        # call it there. `g` throws at both boundary points, outside `:interior`.
        Bramble._seed_mesh1d_rng!(7)
        Ωₕ = try
            mesh(domain(interval(0.0, 1.0)), 9, false)
        finally
            Bramble._unseed_mesh1d_rng!()
        end
        Wₕ = gridspace(Ωₕ)
        w = weights(Wₕ, Innerh())
        x = points(Ωₕ)
        n = length(w)
        g = x -> 0 < x < 1 ? x : throw(DomainError(x))
        rg = restrict_to(:interior, source_function(g, Val(1)))

        b = assemble(form(Wₕ, v -> innerₕ(shift_op(rg, 1, 1), v)))
        @test isequal(b, [1 < i + 1 < n ? x[i + 1] * w[i] : 0.0 for i in 1:n])
        b2 = assemble(form(Wₕ, v -> innerₕ(shift_op(rg, 1, -2), v)))
        @test isequal(b2, [1 < i - 2 < n ? x[i - 2] * w[i] : 0.0 for i in 1:n])

        # outside the region the stencil is the zero `form` stored, built without reading
        # the source there (gpena/Bramble.jl#639)
        ast = resolve_form_ast(form(Wₕ, v -> innerₕ(shift_op(rg, 1, 1), v)))
        op, mk = Bramble._bind_walk(ast.left_op, Wₕ)
        @test Bramble.local_stencil(op, Wₕ, CartesianIndex(n - 1), mk, n - 1) ===
              (((0,), 0.0),)
        @test only(Bramble.local_stencil(op, Wₕ, CartesianIndex(2), mk, 2))[end] == x[3]

        # refilling allocates nothing
        l = form(Wₕ, v -> innerₕ(shift_op(rg, 1, 1), v))
        bl = assemble(l)
        assemble!(bl, l)                            # warm up
        @test (@allocated assemble!(bl, l)) == 0
    end

    @testset "Restricted source keeps one type (#639)" begin
        # `form` calls each restricted source once, at a point of its region the walk reads,
        # and stores a zero stencil of that value: outside the region the node answers it
        # instead of `()`, so `local_stencil` infers one type. `n` stays odd and the mesh
        # uniform: the middle point, x = 0.5, is sampled for the element type
        # (`_leaf_eltype`) whatever the region, and every source below is defined there.
        n = 9
        Ωₕ = mesh(domain(interval(0.0, 1.0), :half => x -> x[1] > 0.5,
                :none => x -> x[1] > 2), n, true)
        Wₕ = gridspace(Ωₕ)
        x = points(Ωₕ)
        w = weights(Wₕ, Innerh())
        ri = s -> restrict_to(:interior, s)       # 2:n-1
        rb = s -> restrict_to(:boundary, s)       # 1 and n
        rh = s -> restrict_to(:half, s)           # 6:n
        sf = source_function(x -> x + 1, Val(1))
        q = source_function(x -> sqrt(x - 0.5), Val(1))
        g = source_function(x -> 0 < x < 1 ? x : throw(DomainError(x)), Val(1))
        fi = source_function(x -> 1 / (x * (1 - x)), Val(1))  # Inf at both ends
        bvec(src) = assemble(form(Wₕ, v -> innerₕ(src, v)))
        ue = [1 < i < n ? 2.0 : Inf for i in 1:n]   # infinite outside :interior
        v2 = [1 < i < n ? 3.0 : Inf for i in 1:n]

        @testset "inferred, bound and unbound" begin
            srcs = (ri(sf), 2.0 * ri(sf), ri(sf) + sf, D₊ₓ(ri(sf)), ri(dirac(0.4)),
                ri(ri(sf)), rh(ri(q)), ri(sf) + rb(sf) + rh(sf), collect(1.0:n) * ri(sf),
                shift_op(2.0 * rb(sf), 1, 1), shift_op(ri(sf), 1, -1), ue * (-ri(sf)),
                v2 * (ue * ri(rh(sf))))
            for src in srcs
                ast = resolve_form_ast(form(Wₕ, v -> innerₕ(src, v)))
                p = ast isa Bramble.OperatorScale ? ast.inner_op : ast
                for op in (ast, p, p.left_op)
                    bound, mk = Bramble._bind_walk(op, Wₕ)
                    for (o, m) in ((op, nothing), (bound, mk)), i in (1, 5, n)

                        @test @inferred(Bramble.local_stencil(o, Wₕ, CartesianIndex(i), m, i)) isa
                              Tuple
                    end
                end
            end
        end

        @testset "no allocation" begin
            function refill_bytes(src)
                l = form(Wₕ, v -> innerₕ(src, v))
                b = assemble(l)
                assemble!(b, l)                   # warm up
                return @allocated assemble!(b, l)
            end
            @test refill_bytes(ri(sf) + rb(sf) + rh(sf)) == 0
            @test refill_bytes(source_function(x -> 3, Val(1)) + ri(sf)) == 0
            @test refill_bytes(ri(dirac(0.4)) + sf) == 0
            @test refill_bytes(ue * (-ri(sf))) == 0
            @test refill_bytes(v2 * (ue * ri(rh(sf)))) == 0
        end

        @testset "never read outside the region" begin
            inner_g = [1 < i < n ? x[i] * w[i] : 0.0 for i in 1:n]
            b = bvec(ri(g))
            @test b[1] === 0.0 && b[n] === 0.0
            @test isequal(b, inner_g)
            @test isequal(bvec(ri(fi)), [1 < i < n ? 1 / (x[i] * (1 - x[i])) * w[i] : 0.0
                                         for i in 1:n])
            # nested both ways: only 6:n-1 lies in both regions
            both = [6 <= i < n ? x[i] * w[i] : 0.0 for i in 1:n]
            @test isequal(bvec(rh(ri(g))), both)
            @test isequal(bvec(ri(rh(g))), both)
            # shifted: row `i` reads point `i - 1`
            @test isequal(bvec(shift_op(ri(fi), 1, -1)),
                [1 < i - 1 < n ? 1 / (x[i - 1] * (1 - x[i - 1])) * w[i] : 0.0 for i in 1:n])
            # a restriction over a shift: the boundary rows read points 0 (off the grid) and
            # n - 1, so `qq` is probed at x[n - 1], never at x[n] = 1, where it throws
            qq = source_function(x -> x == 1.0 ? throw(DomainError(x)) : x, Val(1))
            b = bvec(rb(shift_op(rh(qq), 1, -1)))
            @test isequal(b, [i == n ? x[n - 1] * w[n] : 0.0 for i in 1:n])
        end

        @testset "infinite scale outside the region" begin
            # directly over the restriction an infinite coefficient still gives exactly 0
            u = [1 < i < n ? 2.0 : Inf for i in 1:n]
            bu = bvec(u * ri(fi))
            @test bu[1] === 0.0 && bu[n] === 0.0
            @test bu ≈ [1 < i < n ? 2 / (x[i] * (1 - x[i])) * w[i] : 0.0 for i in 1:n]
            expected = [1 < i < n ? Inf : (x[i] + 1) * w[i] for i in 1:n]
            @test isequal(bvec(Inf * ri(sf) + sf), expected)
            @test isequal(bvec(Ref(Inf) * ri(sf) + sf), expected)
            # so it does over restrictions and scales nested in the restriction: these rows
            # lie outside the inner region, where `u` is infinite
            u5 = [i <= 5 || i == n ? Inf : 2.0 for i in 1:n]
            inside = [6 <= i < n ? 2 * (x[i] + 1) * w[i] : 0.0 for i in 1:n]
            @test isequal(bvec(u5 * ri(rh(sf))), inside)
            @test isequal(bvec(u5 * rh(ri(sf))), inside)
            @test isequal(bvec(u5 * ri(2.0 * rh(sf))), 2 .* inside)
            @test isequal(bvec(Inf * ri(rh(sf)) + sf),
                [6 <= i < n ? Inf : (x[i] + 1) * w[i] for i in 1:n])
            # residue (`form` docstring): over a shift, a difference, an average or a sum it
            # gives NaN, not 0 or ±Inf
            @test isnan(bvec(u * shift_op(ri(sf), 1, 1))[n])
            @test isnan(bvec(u * (ri(sf) + sf))[1])
            @test isnan(bvec(u5 * ri(shift_op(rh(sf), 1, -1)))[2])
            @test isnan(bvec(u5 * D₊ₓ(ri(sf)))[1])
            @test isnan(bvec(u5 * D₋ₓ(ri(sf)))[n])
            @test isnan(bvec(u5 * M₊ₓ(ri(sf)))[1])
        end

        @testset "scale chain outside the region" begin
            # only pointwise scales lie between the coefficient and the restriction, so the
            # chain is zeroed once at the top: an outer infinite coefficient still gives 0
            u5 = [i <= 5 || i == n ? Inf : 2.0 for i in 1:n]
            ri_b(c) = [1 < i < n ? (x[i] + 1) * c * 2.0 * w[i] : 0.0 for i in 1:n]
            rh_b(c, d) = [6 <= i < n ? (x[i] + 1) * c * d * w[i] : 0.0 for i in 1:n]
            @test isequal(bvec(ue * (2.0 * ri(sf))), ri_b(2.0))
            @test isequal(bvec(ue * (-ri(sf))), ri_b(-1.0))
            @test isequal(bvec(ue * (ri(sf) / 2)), ri_b(0.5))
            @test isequal(bvec(u5 * (2.0 * ri(rh(sf)))), rh_b(2.0, 2.0))
            @test isequal(bvec(v2 * (u5 * ri(rh(sf)))), rh_b(2.0, 3.0))
            @test isequal(bvec(u5 * ri(v2 * rh(sf))), rh_b(3.0, 2.0))
            @test isequal(bvec(u5 * ri(Ref(2.0) * rh(sf))), rh_b(2.0, 2.0))
            # three and four scales above the restriction: the trait follows every one
            v3 = fill(3.0, n)
            @test isequal(bvec(ue * (v3 * (2.0 * ri(sf)))), ri_b(6.0))
            @test isequal(bvec(u5 * (Ref(2.0) * (2.0 * (v2 * ri(rh(sf)))))),
                rh_b(12.0, 2.0))
            @test isequal(bvec(Inf * (v2 * ri(sf)) + sf),
                [1 < i < n ? Inf : (x[i] + 1) * w[i] for i in 1:n])
        end

        @testset "positive zero outside the region" begin
            F = fill(-0.0, n)
            assemble_add!(F, form(Wₕ, v -> innerₕ(ri(source_function(x -> -1.0, Val(1))), v)))
            @test F[1] === 0.0 && F[n] === 0.0
            @test F[2] == -w[2]
        end

        @testset "call set" begin
            calls = Float64[]
            rec = source_function(x -> (push!(calls, x); x + 1), Val(1))
            l = form(Wₕ, v -> innerₕ(rh(rec), v))
            @test calls == [x[6]]                 # the probe: the region's first point
            assemble(l)
            @test sort!(unique(calls)) == x[5:n]  # x[5]: the element-type sample
            empty!(calls)
            l = form(Wₕ, v -> innerₕ(shift_op(rh(rec), 1, -1), v))
            assemble(l)
            @test sort!(unique(calls)) == x[5:(n - 1)]  # x[n] is read by no row
            # a source throwing at the first point of its region throws at `form`
            x₂ = x[2]
            th = source_function(y -> y == x₂ ? throw(DomainError(y)) : y, Val(1))
            @test_throws DomainError form(Wₕ, v -> innerₕ(ri(th), v))
        end

        @testset "markers! after form" begin
            # the probe point comes from the markers at `form`: once the region moves, the
            # source has been called at x[6], which no assembly reads any more
            Ωm = mesh(domain(interval(0.0, 1.0), :half => x -> x[1] > 0.5), n, true)
            Wm = gridspace(Ωm)
            calls = Float64[]
            rec = source_function(x -> (push!(calls, x); x + 1), Val(1))
            l = form(Wm, v -> innerₕ(rh(rec), v))
            m = copy(Bramble.markers(Ωm))
            m[:half] = [xi > 0.8 for xi in points(Ωm)]
            Bramble.markers!(Ωm, m)
            b = assemble(l)
            @test sort!(unique(calls)) == [x[5], x[6], x[8], x[n]]
            @test isequal(b, [i >= 8 ? (x[i] + 1) * w[i] : 0.0 for i in 1:n])
        end

        @testset "empty region stays empty" begin
            ast = resolve_form_ast(form(Wₕ, v -> innerₕ(restrict_to(:none, sf), v)))
            @test ast.left_op.zero_stencil === nothing
            op, mk = Bramble._bind_walk(ast.left_op, Wₕ)
            @test Bramble.local_stencil(op, Wₕ, CartesianIndex(3), mk, 3) === ()
            @test all(iszero, bvec(restrict_to(:none, sf)))
        end
    end

    @testset "Region restriction" begin
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
        # every entry is either the unmasked one or zero: the mask selects, it does not scale
        @test all(i -> b[i] ≈ full[i] || iszero(b[i]), eachindex(b))
    end

    @testset "VectorElement source" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 5), (true, false))
        Wₕ = gridspace(Ωₕ)
        f = x -> x[1] * x[2] + x[1]
        fₕ = Rₕ(Wₕ, f)
        w = weights(Wₕ, Innerh())
        sv = SourceVector{2, typeof(parent(fₕ))}(parent(fₕ))

        b = assemble(form(Wₕ, v -> innerₕ(D₋ₓ(sv), v)))
        @test b ≈ parent(D₋ₓ(fₕ)) .* w
        @test !all(iszero, b)
    end

    @testset "Interpolated source" begin
        Ωbig = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 8), (true, true))
        Ωsmall = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (true, true))
        Wbig, Wsmall = gridspace(Ωbig), gridspace(Ωsmall)
        f = x -> x[1] + x[2]
        us = Rₕ(Wsmall, f)
        w = weights(Wbig, Innerh())

        # the interpolant landed on Wbig, then differenced there: the numeric spelling of
        # exactly what D₋ₓ(πₕ(us)) means symbolically
        b = assemble(form(Wbig, v -> innerₕ(D₋ₓ(πₕ(us)), v)))
        @test b ≈ parent(D₋ₓ(πₕ(Wbig, us))) .* w
        @test !all(iszero, b)
    end

    @testset "Plain source regression" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 6), (true, true))
        Wₕ = gridspace(Ωₕ)
        f = x -> x[1] + x[2]
        sf = source_function(f, Val(2))
        fₕ = Rₕ(Wₕ, f)
        w = weights(Wₕ, Innerh())

        @test assemble(form(Wₕ, v -> innerₕ(sf, v))) ≈ parent(fₕ) .* w
        @test assemble(form(Wₕ, v -> innerₕ(fₕ, v))) ≈ parent(fₕ) .* w
        @test assemble(form(Wₕ, v -> innerₕ(2.0, v))) ≈ 2.0 .* w
    end

    @testset "Test-side offsets" begin
        # innerₕ(f, D₋ₓ(v)) is the discrete adjoint: the coefficient at grid point J picks
        # up contributions from both I = J and I = J+1. It must NOT be collapsed the way
        # the source side is.
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 7, true)
        Wₕ = gridspace(Ωₕ)
        fₕ = Rₕ(Wₕ, x -> x + 1)
        l = form(Wₕ, v -> innerₕ(fₕ, D₋ₓ(v)))
        b = assemble(l)
        # contract against a test function: l(uₕ) = Σ wᵢ fᵢ (D₋ₓu)ᵢ, computable numerically
        uₕ = Rₕ(Wₕ, x -> x^2)
        @test l(uₕ) ≈ sum(weights(Wₕ, Innerh()) .* parent(fₕ) .* parent(D₋ₓ(uₕ)))
        @test !all(iszero, b)
    end

    @testset "inner₊/inner₊₂ source-only left operand" begin
        # inner₊ with a LazyOp left and a BackwardDifference right, and its mirror, and the plain
        # inner₊ₓ/inner₊ᵧ/inner₊₂(left, right) methods, all branch the same way
        # innerₕ does: a LinearProduct when left is source-only, a BilinearProduct
        # otherwise. Both branches are the same function, so the untested LinearProduct
        # side is checked against the already-verified BilinearProduct side, contracted at
        # a concrete vector equal to the source's own values (not a from-scratch oracle, but
        # a genuinely independent code path, `_contracted_left_stencil` versus `local_stencil`
        # on a BilinearProduct, computing what should be the identical number).
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 8, true)
        Wₕ = gridspace(Ωₕ)
        f = x -> x^2 + sin(2x)
        sf = source_function(f, Val(1))
        fₕ = Rₕ(Wₕ, f)

        # inner₊(source, D₋ₓ(v)): source-only left, direction named by the right side
        b1 = assemble(form(Wₕ, v -> inner₊(sf, D₋ₓ(v))))
        A1 = assemble(form(Wₕ, Wₕ, (u, v) -> inner₊(u, D₋ₓ(v))))
        @test b1 ≈ A1 * parent(fₕ)
        @test !all(iszero, b1)

        # inner₊(D₋ₓ(source), v): source-only left wrapped in a difference, plain right
        b2 = assemble(form(Wₕ, v -> inner₊(D₋ₓ(sf), v)))
        A2 = assemble(form(Wₕ, Wₕ, (u, v) -> inner₊(D₋ₓ(u), v)))
        @test b2 ≈ A2 * parent(fₕ)
        @test !all(iszero, b2)

        # inner₊₂(source, v): the plain directional (z) form, 3D since Dim = 3 needs D ≥ 3
        Ω3 = mesh(
            domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (4, 3, 5), (true, true, true)
        )
        W3 = gridspace(Ω3)
        f3 = x -> x[1] + x[2]^2 - x[3]
        sf3 = source_function(f3, Val(3))
        f3ₕ = Rₕ(W3, f3)

        b3 = assemble(form(W3, v -> inner₊₂(sf3, v)))
        A3 = assemble(form(W3, W3, (u, v) -> inner₊₂(u, v)))
        @test b3 ≈ A3 * parent(f3ₕ)
        @test !all(iszero, b3)
    end

    @testset "Allocation contract" begin
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
        nested = Wₕ -> (sf = source_function(f, Val(2)); v -> innerₕ(D₋ₓ(Mᵧ(sf)), v))

        # the source-value path must not cost an allocation, at any size: the branch on
        # `_is_source_only` is decided by the operand's type and folds away
        for build in (plain, diffed, nested)
            @test refill_bytes(8, build) == 0
            @test refill_bytes(16, build) == 0
        end
    end

    WITH_AD_TESTS && @testset "Source differentiation" begin
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

    @testset "Invalid source node error" begin
        # `_is_source_only` and `stencil_shift_trait` are two independent ladders over the
        # same node types: a source-only subtree is contracted by reading its own
        # `local_stencil`, correct only because a source is marked `PointDependentStencil`.
        # A future node accepted by the first ladder without a matching entry in the second
        # would otherwise relabel offsets instead of re-reading the neighbour:
        # this checks that case and throws instead. Not
        # reachable through today's node types (every one that answers `true` to
        # `_is_source_only` already answers `PointDependentStencil` here), so exercised
        # directly rather than by constructing a form that hits it.
        struct _UnmarkedSourceNode{D} <: Bramble.LazyOp{D} end
        Bramble._is_source_only(::_UnmarkedSourceNode) = true
        @test_throws ArgumentError Bramble._contracted_left_stencil(
            _UnmarkedSourceNode{1}(), nothing, CartesianIndex(1), nothing, 1
        )
    end

    # gpena/Bramble.jl#197: form(Wₕ, f) now samples every reachable SourceFunction once,
    # against its own leaf's space, and stores the result as a SourceVector -- so a form
    # built from a brand-new closure pays its one-time compilation cost at construction,
    # not on every later assemble!/assemble.
    @testset "Eager source lowering (#197)" begin
        @testset "Plain source lowers to SourceVector" begin
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)
            Wₕ = gridspace(Ωₕ)
            f = x -> x^2 + sin(3x)
            l = form(Wₕ, v -> innerₕ(f, v))
            ast = resolve_form_ast(l)
            @test ast isa LinearProduct
            @test ast.left_op isa SourceVector

            b = assemble(l)
            fₕ = Rₕ(Wₕ, f)
            w = weights(Wₕ, Innerh())
            @test b ≈ parent(fₕ) .* w
        end

        @testset "wrapped source: unlowered" begin
            # D₋ₓ(sf) builds a node type `_lower_sources` has no method for, so it falls
            # through to the generic leaf fallback -- unchanged, not incorrectly rewritten.
            # A missed optimisation, not a correctness gap: "1D numeric equivalence" above
            # checks its values.
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)
            Wₕ = gridspace(Ωₕ)
            f = x -> x^2 + sin(3x)
            sf = source_function(f, Val(1))
            l = form(Wₕ, v -> innerₕ(D₋ₓ(sf), v))
            ast = resolve_form_ast(l)
            @test ast.left_op isa Bramble.BackwardDifference
            @test ast.left_op.inner_op isa SourceFunction
        end

        @testset "composite: per-component term lowers" begin
            Wleaf = gridspace(mesh(domain(interval(0.0, 1.0)), 21, true))
            Vₕ = Wleaf^Val(2)
            f = x -> x[1]^2
            l = form(Vₕ, v -> innerₕ(f, v(2)))
            ast = resolve_form_ast(l)
            @test ast.left_op isa SourceVector

            n = ndofs(Wleaf)
            b = assemble(l)
            w = weights(Wleaf, Innerh())
            @test b[1:n] == zeros(n)
            @test b[(n + 1):end] ≈ parent(Rₕ(Wleaf, f)) .* w
        end

        # A term shared across leaves is left unlowered.
        @testset "composite: shared term stays unlowered" begin
            # A term naming no component goes to every leaf (`_routed_target`); those
            # leaves may have different meshes, so there is no single space to eagerly
            # sample against. Skipped, not incorrectly lowered against one arbitrary leaf.
            Wleaf = gridspace(mesh(domain(interval(0.0, 1.0)), 21, true))
            Vₕ = Wleaf^Val(2)
            f = x -> x[1] + 1.0
            l = form(Vₕ, v -> innerₕ(f, v))
            ast = resolve_form_ast(l)
            @test ast.left_op isa SourceFunction

            n = ndofs(Wleaf)
            b = assemble(l)
            w = weights(Wleaf, Innerh())
            expected = parent(Rₕ(Wleaf, f)) .* w
            @test b[1:n] ≈ expected
            @test b[(n + 1):end] ≈ expected
        end

        # Non-negotiable: lowering must not freeze a dynamic coefficient.
        @testset "dynamic coefficients stay live" begin
            # A raw closure loses live re-evaluation once lowered; Ref and VectorElement
            # coefficients must not, since neither is ever wrapped in a SourceFunction.
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, true)
            Wₕ = gridspace(Ωₕ)

            α = Ref(1.0)
            uₕ = Rₕ(Wₕ, x -> 1.0)
            l = form(Wₕ, v -> α * innerₕ(uₕ, v))
            b1 = assemble(l)
            α[] = 3.0
            b2 = assemble(l)
            @test b2 ≈ 3 .* b1

            vₕ = Rₕ(Wₕ, x -> 1.0)
            l2 = form(Wₕ, v -> innerₕ(vₕ, v))
            c1 = assemble(l2)
            vₕ .= 2.0
            c2 = assemble(l2)
            @test c2 ≈ 2 .* c1
        end

        WITH_AD_TESTS && @testset "lowered source: Dual propagates" begin
            # Distinct from "Source differentiation" above, which wraps its source in
            # D₋ₓ and so never reaches the lowering path this checks.
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, true)
            Wₕ = gridspace(Ωₕ)

            resid(p) = sum(assemble(form(Wₕ, v -> innerₕ(x -> p[1] * x[1]^2, v))))
            g = ForwardDiff.gradient(resid, [2.0])
            @test g[1] ≈ resid([1.0])
            @test isfinite(g[1])
            @test !iszero(g[1])
        end
    end

    @testset "Function * VectorElement (#197)" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, true)
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> 1.0)
        cond = x -> x[1] < 0.5

        left = cond * uₕ
        right = uₕ * cond
        @test left == right
        @test parent(left) == parent(Rₕ(Wₕ, cond))

        # The issue's own worked example: a continuous condition scaling a grid function,
        # used directly as a form's source.
        l = form(Wₕ, v -> innerₕ(cond * uₕ, v))
        b = assemble(l)
        w = weights(Wₕ, Innerh())
        @test b ≈ parent(cond * uₕ) .* w
    end
end

end # module FormSourceOperatorsTests
