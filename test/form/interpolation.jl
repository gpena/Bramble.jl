using Test
using Bramble
using Bramble: SourceFunction, TrialFunction, TestFunction, LinearProduct,
               BilinearProduct, resolve_form_ast, _is_source_only, Innerh

# πₕ(uₕ) wraps a grid function's interpolant as a genuine LazyOp source (SourceFunction), so
# it composes with the same operators (D₋ₓ, M₋ₓ, ...) any other source does. The one thing
# that is not automatic is `innerₕ`'s own dispatch: its generic LazyOp×LazyOp constructor used
# to assume "trial × test" unconditionally and build a BilinearProduct, which is the wrong AST
# shape for a source — πₕ(uₕ) (and D₋ₓ(πₕ(uₕ)), etc.) never reaches the Function/Number/
# VectorElement overloads that build a LinearProduct, because it already arrives as a LazyOp.
# `_is_source_only` is the fix: a recursive predicate that lets innerₕ tell "a source, however
# deeply wrapped" from "a trial function" and route to the correct AST node either way.

@testset "_is_source_only" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 5), (true, true))
    Wₕ = gridspace(Ωₕ)
    uₕ = Rₕ(Wₕ, x -> x[1])

    # a bare source, and sources wrapped in every operator πₕ is meant to compose with
    src = πₕ(uₕ)
    @test src isa SourceFunction
    @test _is_source_only(src)
    @test _is_source_only(D₋ₓ(src))
    @test _is_source_only(D₊ₓ(src))
    @test _is_source_only(M₋ₓ(src))
    @test _is_source_only(M₊ₓ(src))
    @test _is_source_only(D₋ₓ(D₋ᵧ(src)))          # nested wrapping, still source-only
    @test _is_source_only(2 * src)
    @test _is_source_only(src + src)

    # a trial function is never source-only, wrapped or not — this is the case the fix must
    # not disturb, since it is what every existing bilinear form is built from
    u = TrialFunction{2}()
    @test !_is_source_only(u)
    @test !_is_source_only(D₋ₓ(u))
    @test !_is_source_only(2 * u)
    @test !_is_source_only(u + u)

    # a mix of a source and a trial function under a sum is not source-only either
    @test !_is_source_only(src + u)

    # the products themselves are never source-only, and a test function is not a source
    v = TestFunction{2}()
    @test !_is_source_only(v)
    @test !_is_source_only(innerₕ(uₕ, v))
end

@testset "innerₕ dispatch: πₕ(u) as a LazyOp source builds a LinearProduct, not Bilinear (point 25)" begin
    Ωbig = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (8, 8), (true, true))
    Ωsmall = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (4, 4), (true, true))
    Wbig, Wsmall = gridspace(Ωbig), gridspace(Ωsmall)
    Vh = Bramble.CompositeGridSpace((Wbig, Wsmall))
    uv = Rₕ(Vh, (x -> x[1] + x[2], x -> 2x[1] - x[2]))
    u_leaf2 = uv(2)   # lives on Wsmall, moved onto Wbig below

    # the dispatch itself: innerₕ(πₕ(u), v) must build a LinearProduct (source × test) — a
    # BilinearProduct here would crash at assembly, since the linear-form walk only knows how
    # to scatter a single-offset stencil
    lf1 = form(Vh, v -> innerₕ(πₕ(u_leaf2), v(1)))
    @test resolve_form_ast(lf1) isa LinearProduct

    # composed with a difference/average, still source-only, still a LinearProduct
    lf2 = form(Vh, v -> innerₕ(D₋ₓ(πₕ(u_leaf2)), D₋ₓ(v(1))))
    @test resolve_form_ast(lf2) isa LinearProduct

    lf3 = form(Vh, v -> innerₕ(M₋ₓ(πₕ(u_leaf2)), v(1)))
    @test resolve_form_ast(lf3) isa LinearProduct

    # and it actually assembles, rather than only type-checking
    b1, b2, b3 = assemble(lf1), assemble(lf2), assemble(lf3)
    @test length(b1) == ndofs(Vh)
    @test all(isfinite, b1)
    @test all(isfinite, b2)
    @test all(isfinite, b3)

    # numeric consistency: innerₕ(πₕ(u), v(1)) scatters |cell_i| * interpolate_at(u, x_i) into
    # block 1, so it must equal the numeric interpolate path times the weights directly
    Ib1 = values(interpolate(Wbig, u_leaf2))
    w1 = weights(Wbig, Innerh())
    @test sum(b1[1:ndofs(Wbig)]) ≈ sum(Ib1 .* w1) atol=1e-8

    # an ordinary bilinear form is unaffected: a trial function, wrapped or not, still builds
    # a BilinearProduct, exactly as it always has
    lf_bilinear = form(Wbig, Wbig, (u, w) -> innerₕ(D₋ₓ(u), D₋ₓ(w)))
    @test resolve_form_ast(lf_bilinear) isa BilinearProduct
end

@testset "the parallel path agrees with the serial one" begin
    @info "interpolation composition tested on $(Threads.nthreads()) thread(s)"

    Ωbig = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (10, 10), (true, true))
    Ωsmall = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (5, 5), (true, true))
    Wbig, Wsmall = gridspace(Ωbig), gridspace(Ωsmall)
    Vh = Bramble.CompositeGridSpace((Wbig, Wsmall))
    uv = Rₕ(Vh, (x -> sin(x[1]) + x[2], x -> cos(x[1]) - x[2]))
    u_leaf2 = uv(2)

    for (nm, g) in (
        ("plain interpolant", v -> innerₕ(πₕ(u_leaf2), v(1))),
        ("difference of the interpolant", v -> innerₕ(D₋ₓ(πₕ(u_leaf2)), D₋ₓ(v(1)))),
        ("average of the interpolant", v -> innerₕ(M₋ₓ(πₕ(u_leaf2)), v(1))))
        lf = form(Vh, g)
        bs = assemble(lf)
        bp = similar(bs)
        @test assemble_parallel!(bp, lf) === bp
        @test bp ≈ bs
    end

    if Threads.nthreads() > 1
        # more work than threads, so every thread gets a chunk of every colour — a race in
        # the scatter, or a colour built against the wrong AST shape, shows here
        Ωb = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (40, 40), (true, true))
        Ωb_small = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (17, 17), (true, true))
        Wb, Wb_small = gridspace(Ωb), gridspace(Ωb_small)
        Vb = Bramble.CompositeGridSpace((Wb, Wb_small))
        uvb = Rₕ(Vb, (x -> x[1] * x[2] + 1, x -> x[1] - 2x[2]))
        ub2 = uvb(2)

        for (nm, g) in (
            ("plain interpolant", v -> innerₕ(πₕ(ub2), v(1))),
            ("difference composed with the interpolant",
            v -> innerₕ(D₋ₓ(πₕ(ub2)), D₋ₓ(v(1)))))
            lfb = form(Vb, g)
            bb = assemble(lfb)
            bbp = similar(bb)
            assemble_parallel!(bbp, lfb)
            @test bbp ≈ bb

            # repeated runs agree with each other, which a race would break
            bbp2 = similar(bb)
            assemble_parallel!(bbp2, lfb)
            @test bbp2 == bbp
        end
    else
        @test_skip "concurrency not exercised: only one thread available"
        @test_skip "repeated-run race check not exercised: only one thread available"
    end
end
