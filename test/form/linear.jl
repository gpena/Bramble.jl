using Test
using Bramble
using ForwardDiff
using LinearAlgebra: Diagonal, diag, dot, I
using Bramble: LinearForm, form, assemble, assemble!, assemble_parallel!, test_space,
               element, resolve_form_ast, apply_dirichlet_conditions!, LinearProduct,
               values, TestFunction, TrialFunction,
               IndexedTestFunction, IndexedTrialFunction, test_component_or_nothing,
               routes_by_component, component, components, _colour_strides,
               stencil_offsets, ndofs, Innerh, Innerplus, evaluate!

# Assembling the right-hand side of a system.
#
# `form(Wₕ, v -> …)` closes over a test function and records the AST; `assemble` walks the
# grid, evaluates the stencil at each point and scatters the weights into a vector. This is
# the file a time loop calls every step, so the allocation counts below are part of its
# contract rather than a nicety.
#
# The values are checked against integrals that can be worked out by hand: `innerₕ(uₕ, v)`
# assembles the vector whose entries are ``|□ᵢ| uᵢ``, so summing it gives ``∫ u`` over the
# domain to the accuracy of the quadrature — exactly, for the cases below.

@testset "Linear forms" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
        (8, 8), (true, true))
    Wₕ = gridspace(Ωₕ)
    Vₕ = gridspace(Ωₕ, Val(2))
    uₕ = Rₕ(Wₕ, x -> x[1] + x[2])
    n = ndofs(Wₕ)

    @testset "what it builds" begin
        lf = form(Wₕ, v -> innerₕ(uₕ, v))
        @test lf isa LinearForm
        @test test_space(lf) === Wₕ
        @test resolve_form_ast(lf) isa LinearProduct
    end

    @testset "what it assembles" begin
        # ∫(x + y) over the unit square is 1, and the assembled vector's entries are the
        # cell measures times the coefficients, so it sums to exactly that.
        b = assemble(form(Wₕ, v -> innerₕ(uₕ, v)))
        @test length(b) == n
        @test sum(b) ≈ 1.0

        # a constant source integrates to the area
        @test sum(assemble(form(Wₕ, v -> innerₕ(Rₕ(Wₕ, x -> 3.0), v)))) ≈ 3.0

        # and a number or a function on the left works the same way
        @test sum(assemble(form(Wₕ, v -> innerₕ(2.0, v)))) ≈ 2.0
        @test sum(assemble(form(Wₕ, v -> innerₕ(x -> x[1], v)))) ≈ 0.5

        # in place, into a vector the caller owns
        b2 = zeros(n)
        returned = assemble!(b2, form(Wₕ, v -> innerₕ(uₕ, v)))
        @test returned === b2
        @test b2 ≈ b

        # assembling twice into the same vector does not accumulate
        assemble!(b2, form(Wₕ, v -> innerₕ(uₕ, v)))
        @test b2 ≈ b
    end

    @testset "over a composite space" begin
        # The composite core walks the components and assembles the *same* AST into each
        # block at its offset. So a form written against one source puts that source in
        # every component: ∫x over the unit square is 0.5, and the assembled vector sums to
        # 1.0 across the two blocks. Giving each component a different integrand needs the
        # indexed trial and test functions, which is the coupled path in bilinear.jl.
        uv = Rₕ(Vₕ, (x -> x[1], x -> x[2]))
        lf = form(Vₕ, v -> innerₕ(uv(1), v))
        b = assemble(lf)
        @test length(b) == ndofs(Vₕ)
        @test sum(b) ≈ 2 * 0.5

        # and each block holds the same thing, which is what "the same AST" means
        m = ndofs(Wₕ)
        @test b[1:m] ≈ b[(m + 1):(2m)]
    end

    @testset "a nested composite space covers every block" begin
        # The cores used to walk `space.spaces`, the top-level components, and reserve
        # `ndofs(sp)` for each. For a component that is itself composite that reserved the
        # whole nested block while `indices(mesh(sp))` covered one leaf's grid, so the
        # assembly wrote into a fraction of the range. Walking the leaves fixes it, and the
        # check is that a nested space and a flat one of the same size now agree.
        inner = gridspace(Ωₕ, Val(2))
        nested = Bramble.CompositeGridSpace((Wₕ, inner, Wₕ))
        flat = gridspace(Ωₕ, Val(4))
        @test ndofs(nested) == ndofs(flat)

        bn = assemble(form(nested, v -> innerₕ(uₕ, v)))
        bf = assemble(form(flat, v -> innerₕ(uₕ, v)))
        @test bn ≈ bf
        @test sum(bn) ≈ 4 * 1.0            # four blocks, each integrating to 1

        # and the parallel core walks the leaves the same way
        bp = similar(bn)
        assemble_parallel!(bp, form(nested, v -> innerₕ(uₕ, v)))
        @test bp ≈ bn
    end

    @testset "a coupled right-hand side, one term per component" begin
        # `v(i)` gives the i-th component of the symbolic test function, so a coupled form
        # reads the way an indexed grid function does:
        #
        #     form(Vₕ, v -> innerₕ(uₕ(1), v(1)) + innerₕ(uₕ(2), v(2)))
        #
        # Three things had to exist for that. `TestFunction` was not callable at all.
        # `block_extract.jl` routed a `BilinearProduct` to its block but not a
        # `LinearProduct`, so a coupled *right-hand side* could not be routed. And the
        # composite core assembled one AST into every block, which is the opposite of what a
        # per-component form means.
        m = ndofs(Wₕ)
        uv = Rₕ(Vₕ, (x -> x[1], x -> 10 * x[1]))
        nblocks = ndofs(Vₕ) ÷ m
        blocks(b) = [sum(b[(k * m + 1):((k + 1) * m)]) for k in 0:(nblocks - 1)]

        @testset "v(i) is the indexed test function" begin
            v = TestFunction{2}()
            @test v(2) === IndexedTestFunction{2}(2)
            @test TrialFunction{2}()(1) === IndexedTrialFunction{2}(1)
        end

        @testset "each term lands in its own block" begin
            # ∫x = 0.5 into the first block, ∫10x = 5.0 into the second
            b = assemble(form(Vₕ, v -> innerₕ(uv(1), v(1)) + innerₕ(uv(2), v(2))))
            @test blocks(b) ≈ [0.5, 5.0]

            # one term alone reaches only its own block
            @test blocks(assemble(form(Vₕ, v -> innerₕ(uv(2), v(2))))) ≈ [0.0, 5.0]
        end

        @testset "a form naming no component still reaches every block" begin
            # The prior behaviour, which has to keep working: the same integrand in each.
            @test blocks(assemble(form(Vₕ, v -> innerₕ(uv(1), v)))) ≈ [0.5, 0.5]

            # and the two spellings mix, because routing is decided per term
            mixed = assemble(form(Vₕ, v -> innerₕ(uv(1), v) + innerₕ(uv(2), v(2))))
            @test blocks(mixed) ≈ [0.5, 0.5 + 5.0]
        end

        @testset "the routing query" begin
            v = TestFunction{2}()
            @test test_component_or_nothing(innerₕ(uv(1), v(2))) == 2
            @test test_component_or_nothing(innerₕ(uv(1), v)) === nothing
            @test test_component_or_nothing(innerₕ(uv(1), D₋ₓ(v(2)))) == 2

            # asked once per assembly, to keep an un-indexed form off the routing branch
            @test routes_by_component(innerₕ(uv(1), v(1)) + innerₕ(uv(2), v(2)))
            @test !routes_by_component(innerₕ(uv(1), v))
            @test routes_by_component(innerₕ(uv(1), v) + innerₕ(uv(2), v(2)))
        end

        @testset "and it costs nothing" begin
            # The routing recurses the AST rather than flattening it first, which answers
            # with a Vector{Any}: that allocated 544 B per assembly and made every term a
            # dynamic read.
            function bytes(mk)
                Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 8),
                    (true, true))
                V = gridspace(Ω, Val(3))
                u = Rₕ(V, ntuple(_ -> (x -> x[1] + x[2]), 3))
                lf = form(V, mk(u))
                ast = resolve_form_ast(lf)
                b = zeros(ndofs(V))
                assemble!(b, lf; ast = ast)
                return @allocated assemble!(b, lf; ast = ast)
            end
            @test bytes(u -> (v -> innerₕ(u(1), v))) == 0
            @test bytes(u -> (v -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))) == 0
        end
    end

    @testset "a linear combination of products, each over a combination of operators" begin
        # The shape a linear form actually has: a linear combination of inner products, and
        # inside each one a linear combination of operators applied to the test function.
        #
        # This testset exists because the simpler cases all passed while this one was
        # silently wrong. `innerₕ(uₕ, v)` on a composite space was right; adding an operator
        # sum inside the argument was not, because `test_component_or_nothing` had no
        # `OperatorAdd` method, so a sum *inside* one product routed to no component and
        # broadcast to every block. It summed to a plausible number.
        #
        # The check is the defining property of an assembled right-hand side rather than a
        # hand-computed integral: b is the vector for which bᵀw is the form evaluated at w,
        # so the same combination applied numerically through the space layer to a grid
        # function must give bᵀw. That validates the whole path — routing, offsets, weights
        # and truncation — against code that has nothing to do with assembly.
        Ωf = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 11), (true, false))
        Wf = gridspace(Ωf)
        Vf = gridspace(Ωf, Val(3))

        @testset "scalar" begin
            g1 = Rₕ(Wf, x -> x[1] + 2x[2])
            g2 = Rₕ(Wf, x -> exp(x[1]))
            g3 = Rₕ(Wf, x -> 1 + x[2]^2)
            w = Rₕ(Wf, x -> sin(3x[1]) * cos(2x[2]) + 1)

            b = assemble(form(Wf,
                v -> innerₕ(g1, v + 2 * D₋ₓ(v) - M₋ₓ(v)) +
                     inner₊ₓ(g2, D₋ᵧ(v) + jumpₓ(v)) +
                     innerₕ(g3, 3 * M₊ᵧ(v) - Dₕₓ(v))))

            reference = innerₕ(g1, w + 2 * D₋ₓ(w) - M₋ₓ(w)) +
                        inner₊ₓ(g2, D₋ᵧ(w) + jumpₓ(w)) +
                        innerₕ(g3, 3 * M₊ᵧ(w) - Dₕₓ(w))

            @test dot(b, values(w)) ≈ reference
            @test !iszero(reference)          # the identity is not being met by both sides
        end                                   # being zero

        @testset "composite, through the shorthand" begin
            gv = Rₕ(Vf, (x -> x[1] + 2x[2], x -> exp(x[1]), x -> 1 + x[2]^2))
            wv = Rₕ(Vf, (x -> sin(3x[1]) + 1, x -> cos(2x[2]) + 2, x -> x[1] * x[2] + 1))

            b = assemble(form(Vf, v -> innerₕ(gv, v + 2 * D₋ₓ(v) - M₋ₓ(v))))
            reference = sum(innerₕ(components(gv)[c],
                                (w = components(wv)[c]; w + 2 * D₋ₓ(w) - M₋ₓ(w)))
            for c in 1:3)

            @test dot(b, values(wv)) ≈ reference
            @test !iszero(reference)
        end

        @testset "the shorthand equals writing the components out" begin
            # Which is the property that makes it a shorthand rather than a second meaning.
            uv = Rₕ(Vf, (x -> x[1], x -> 100 * x[1], x -> x[2]))
            for (short, long) in (
                (v -> innerₕ(uv, v),
                v -> innerₕ(uv(1), v(1)) + innerₕ(uv(2), v(2)) + innerₕ(uv(3), v(3))),
                (v -> innerₕ(uv, v + D₋ₓ(v)),
                v -> innerₕ(uv(1), v(1) + D₋ₓ(v(1))) +
                     innerₕ(uv(2), v(2) + D₋ₓ(v(2))) +
                     innerₕ(uv(3), v(3) + D₋ₓ(v(3)))),
                (v -> innerₕ(uv, v + 2 * D₋ₓ(v) - M₋ₓ(v)),
                v -> innerₕ(uv(1), v(1) + 2 * D₋ₓ(v(1)) - M₋ₓ(v(1))) +
                     innerₕ(uv(2), v(2) + 2 * D₋ₓ(v(2)) - M₋ₓ(v(2))) +
                     innerₕ(uv(3), v(3) + 2 * D₋ₓ(v(3)) - M₋ₓ(v(3)))),
                (v -> inner₊ₓ(uv, v - M₊ᵧ(v)),
                v -> inner₊ₓ(uv(1), v(1) - M₊ᵧ(v(1))) +
                     inner₊ₓ(uv(2), v(2) - M₊ᵧ(v(2))) +
                     inner₊ₓ(uv(3), v(3) - M₊ᵧ(v(3)))))
                @test assemble(form(Vf, short)) ≈ assemble(form(Vf, long))
            end

            # and the components really are distinguishable, so the check has teeth: the
            # bug it guards against put every component into every block, which agrees with
            # the truth whenever the components integrate to the same thing
            b = assemble(form(Vf, v -> innerₕ(uv, v)))
            m = ndofs(Wf)
            @test [sum(b[(k * m + 1):((k + 1) * m)]) for k in 0:2] ≈ [0.5, 50.0, 0.5]
        end

        @testset "the index distributes through the expression" begin
            v = TestFunction{2}()
            @test (v + D₋ₓ(v))(1) == v(1) + D₋ₓ(v(1))
            @test (3 * M₋ᵧ(v))(2) == 3 * M₋ᵧ(v(2))
            @test (v + 2 * D₋ₓ(v) - M₋ₓ(v))(2) ==
                  v(2) + 2 * D₋ₓ(v(2)) - M₋ₓ(v(2))
            @test v(1)(2) === IndexedTestFunction{2}(2)      # re-indexing replaces

            # a sum inside one product takes the component its sides agree on
            @test test_component_or_nothing(innerₕ(Rₕ(Wf, x -> 1.0), v(2) + D₋ₓ(v(2)))) == 2
            @test test_component_or_nothing(innerₕ(Rₕ(Wf, x -> 1.0), v + D₋ₓ(v))) ===
                  nothing

            # and sides naming different components are ill-formed rather than ambiguous
            @test_throws ArgumentError test_component_or_nothing(
                innerₕ(Rₕ(Wf, x -> 1.0), v(1) + v(2)))
        end
    end

    @testset "the parallel path agrees with the serial one" begin
        # Note what this does and does not establish. `Pkg.test()` runs on one thread unless
        # the environment says otherwise, and on one thread the sweep is a degenerate case:
        # the colours never actually run concurrently, so nothing can race. CI sets
        # JULIA_NUM_THREADS=auto, so it is exercised there. The concurrent assertions below
        # are skipped rather than passed when only one thread is available, so a
        # single-threaded run cannot claim to have tested concurrency.
        @info "parallel assembly tested on $(Threads.nthreads()) thread(s)"

        @testset "the colouring the sweep partitions by" begin
            # Two points of one colour must have disjoint write footprints, so the stride is
            # the span of the reach plus one.
            @test _colour_strides([(0,)]) == (1,)
            @test _colour_strides([(0,), (-1,)]) == (2,)
            @test _colour_strides([(-1, 0), (0, 0), (1, 0)]) == (3, 1)
            @test _colour_strides([(0, -1), (0, 0)]) == (1, 2)
            @test _colour_strides(NTuple{2, Int}[]) == (1, 1)

            # and read off a real form, an operator reaching only its own point gives one
            # colour — the whole grid in a single flat pass — while a difference gives two
            plain = _colour_strides(stencil_offsets(resolve_form_ast(
                form(Wₕ, v -> innerₕ(uₕ, v)))))
            wide = _colour_strides(stencil_offsets(resolve_form_ast(
                form(Wₕ, v -> innerₕ(uₕ, D₋ₓ(v))))))
            @test prod(plain) == 1
            @test prod(wide) == 2
        end

        for (nm, sp, u) in (("scalar", Wₕ, uₕ),
            ("composite", Vₕ, Rₕ(Vₕ, (x -> sin(x[1]), x -> cos(x[2])))(1)))
            lf = form(sp, v -> innerₕ(u, v))
            bs = assemble(lf)
            bp = similar(bs)
            @test assemble_parallel!(bp, lf) === bp
            @test bp ≈ bs
        end

        # The sweep accumulates where the version before it overwrote from a reduction, so a
        # vector assembled into twice has to give the same answer and not double it.
        lfr = form(Wₕ, v -> innerₕ(uₕ, v))
        br = zeros(ndofs(Wₕ))
        assemble_parallel!(br, lfr)
        first_pass = copy(br)
        assemble_parallel!(br, lfr)
        @test br ≈ first_pass

        @testset "differentiating through the parallel sweep" begin
            # The per-thread buffers this path used to carry were `Vector{Float64}`
            # outright, so a Dual-valued assembly could not take it at all. Nothing in the
            # sweep names an element type now, so it can.
            Ωa = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (12, 12),
                (true, true))
            Wa = gridspace(Ωa)
            resid(w, into!) = begin
                uu = element(Wa, w)
                bb = zeros(eltype(w), ndofs(Wa))
                into!(bb, form(Wa, v -> innerₕ(uu, v)))
                sum(bb)
            end
            w0 = fill(0.5, ndofs(Wa))
            gp = ForwardDiff.gradient(w -> resid(w, assemble_parallel!), w0)
            gs = ForwardDiff.gradient(w -> resid(w, (bb, f) -> assemble!(bb, f)), w0)

            @test gp ≈ gs
            @test all(isfinite, gp)
            @test !all(iszero, gp)
        end

        if Threads.nthreads() > 1
            # More work than threads, so every thread gets a chunk of every colour. A race
            # in the scatter shows here and nowhere else.
            Ωb = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (40, 40),
                (true, true))
            Wb = gridspace(Ωb)
            ub = Rₕ(Wb, x -> x[1] * x[2] + 1)
            lfb = form(Wb, v -> innerₕ(ub, v))
            bs = assemble(lfb)
            bp = similar(bs)
            assemble_parallel!(bp, lfb)
            @test bp ≈ bs

            # repeated runs agree with each other, which a race would break
            bp2 = similar(bs)
            assemble_parallel!(bp2, lfb)
            @test bp2 == bp

            # A form whose test argument carries differences colours into more than one
            # group, so the sweep takes the strided path rather than the single flat pass.
            # Every one of these is a distinct arrangement of the routing, and the
            # single-colour cases above exercise none of them.
            for (cnm, g) in (
                ("one difference", v -> innerₕ(ub, D₋ₓ(v))),
                ("a linear combination", v -> innerₕ(ub, v + 2 * D₋ₓ(v) - M₋ₓ(v))),
                ("innerₕ and inner₊ mixed", v -> innerₕ(ub, v) + inner₊(ub, D₋ₓ(v))),
                ("differences in both directions",
                v -> innerₕ(ub, D₋ₓ(v)) + innerₕ(ub, D₋ᵧ(v))))
                lfw = form(Wb, g)
                @test prod(_colour_strides(stencil_offsets(resolve_form_ast(lfw)))) > 1
                bw = assemble(lfw)
                bwp = similar(bw)
                assemble_parallel!(bwp, lfw)
                @test bwp ≈ bw
            end

            # and the coupled path under threads, in each of its arrangements: the terms
            # written out per component, the shorthand that sums them, a routed term
            # carrying operators, and a term whose source and test components differ
            Vb = gridspace(Ωb, Val(2))
            uv2 = Rₕ(Vb, (x -> x[1], x -> 10 * x[1]))
            cv = components(uv2)
            for (cnm, g) in (
                ("per component", v -> innerₕ(cv[1], v(1)) + innerₕ(cv[2], v(2))),
                ("the shorthand", v -> innerₕ(uv2, v)),
                ("the shorthand with operators", v -> innerₕ(uv2, v + D₋ₓ(v))),
                ("routed, with operators",
                v -> innerₕ(cv[1], v(1) + 2 * D₋ₓ(v(1))) + innerₕ(cv[2], v(2))),
                ("crossed components", v -> innerₕ(cv[1], v(2))),
                ("a routed term beside an unrouted one",
                v -> innerₕ(cv[1], v(1)) + innerₕ(uv2, v)))
                lfc = form(Vb, g)
                bc = assemble(lfc)
                bcp = similar(bc)
                assemble_parallel!(bcp, lfc)
                @test bcp ≈ bc
            end
        else
            @test_skip "concurrency not exercised: only one thread available"
            @test_skip "repeatability under threads not exercised"
            @test_skip "multi-colour parallel path under threads not exercised"
            @test_skip "coupled parallel path under threads not exercised"
        end
    end

    @testset "what a reassembly notices" begin
        # The AST is resolved on every `assemble` rather than cached on the form, so a form
        # assembled twice around a change to its source gives two different vectors. That is
        # the whole reason it is not cached, and it is worth pinning: the alternative would
        # freeze coefficients a caller is still changing.
        us = Rₕ(Wₕ, x -> 1.0)
        lfs = form(Wₕ, v -> innerₕ(us, v))

        first_sum = sum(assemble(lfs))
        values(us) .= 5.0
        @test sum(assemble(lfs)) ≈ 5 * first_sum

        # and a rebinding is seen too, since the closure reads the binding at resolve time
        global _rebound = Rₕ(Wₕ, x -> 1.0)
        lfr = form(Wₕ, v -> innerₕ(_rebound, v))
        before = sum(assemble(lfr))
        global _rebound = Rₕ(Wₕ, x -> 5.0)
        @test sum(assemble(lfr)) ≈ 5 * before

        # A scalar coefficient is live too, and by a different route from the element: the
        # closure reads it when `resolve_form_ast` calls `f`, so it is the *rebuilding* of
        # the tree that picks it up, not a shared array. Nothing about the element changes
        # here — only the number in front of it.
        global _alpha = 2.0
        ua_scalar = Rₕ(Wₕ, x -> 1.0)
        lfa_scalar = form(Wₕ, v -> innerₕ(_alpha * ua_scalar, v))
        at_two = sum(assemble(lfa_scalar))
        global _alpha = 10.0
        @test sum(assemble(lfa_scalar)) ≈ 5 * at_two

        # both at once: a scalar and the element it scales
        global _alpha = 3.0
        values(ua_scalar) .= 2.0
        @test sum(assemble(lfa_scalar)) ≈ 3 * at_two   # 3 x 2 against 2 x 1

        # `assemble!` overwrites its destination rather than accumulating into it
        d = zeros(ndofs(Wₕ))
        assemble!(d, lfs)
        once = copy(d)
        assemble!(d, lfs)
        @test d ≈ once

        # Handing in a resolved `ast` caches the expression, which is the point of the
        # keyword and also its one sharp edge. A resolved tree references the source's
        # values, so writing through them is still seen...
        ua = Rₕ(Wₕ, x -> 1.0)
        lfa = form(Wₕ, v -> innerₕ(ua, v))
        ast = resolve_form_ast(lfa)
        assemble!(d, lfa; ast = ast)
        @test sum(d) ≈ first_sum
        values(ua) .= 5.0
        assemble!(d, lfa; ast = ast)
        @test sum(d) ≈ 5 * first_sum

        # ...but replacing what the tree points at is not. This is the documented
        # limitation, asserted so it cannot drift into being a silent one.
        global _frozen = Rₕ(Wₕ, x -> 1.0)
        lff = form(Wₕ, v -> innerₕ(_frozen, v))
        stale_ast = resolve_form_ast(lff)
        global _frozen = Rₕ(Wₕ, x -> 5.0)

        assemble!(d, lff; ast = stale_ast)
        with_stale = sum(d)
        assemble!(d, lff)
        with_fresh = sum(d)

        @test with_fresh ≈ 5 * before          # resolving afresh sees the new element
        @test with_stale ≈ before              # the cached one still reads the old
        @test !isapprox(with_stale, with_fresh)

        # A scalar cannot be rescued by indirection, which is the part worth pinning: `r[]`
        # is dereferenced when the closure runs, so a plain number reaches the scale node and
        # a cached tree has nothing left to re-read.
        r = Ref(2.0)
        ur = Rₕ(Wₕ, x -> 1.0)
        lfref = form(Wₕ, v -> innerₕ(r[] * ur, v))
        ref_ast = resolve_form_ast(lfref)
        assemble!(d, lfref; ast = ref_ast)
        ref_before = sum(d)
        r[] = 10.0
        assemble!(d, lfref; ast = ref_ast)
        @test sum(d) ≈ ref_before               # frozen, Ref or not
        assemble!(d, lfref)
        @test sum(d) ≈ 5 * ref_before           # and live again once resolved afresh
    end

    @testset "the functor contracts without building the vector" begin
        # `l(vₕ)` answers with a number, and used to allocate a whole right-hand side to
        # get it: `dot(assemble(form), values(vₕ))`. The walk now multiplies each stencil
        # weight by `v` at the row it would have written to, so the same sum is taken as it
        # goes and no vector is built.
        n = ndofs(Wₕ)
        lfc = form(Wₕ, v -> innerₕ(uₕ, v))
        astc = resolve_form_ast(lfc)
        b = assemble(lfc)

        @test lfc(uₕ) ≈ sum(b .* values(uₕ))

        # The functor takes no `ast` on purpose, so it resolves every call and cannot be
        # allocation free. What it must not do is build a full-length vector, which is what
        # this pins: the bound is a fraction of one, not zero. Behind a barrier, because at
        # top level over globals the call itself reports bytes it does not spend.
        function _contract_allocs(lf, v)
            lf(v)
            return @allocated lf(v)
        end
        @test _contract_allocs(lfc, uₕ) < 8 * n ÷ 100
        @test assemble(lfc) ≈ b                     # and the vector path still works

        # Every arrangement has to agree with assembling and contracting by hand, not just
        # the simple one: the composite cores are separate code from the scalar one.
        Vc = gridspace(Ωₕ, Val(3))
        uc = Rₕ(Vc, (x -> 1.0, x -> 100.0, x -> 10000.0))
        wc = Rₕ(Vc, (x -> 2.0, x -> 3.0, x -> 5.0))
        cc = components(uc)
        for (nm, g) in (
            ("scalar, a difference", v -> innerₕ(uₕ, D₋ₓ(v))),
            ("scalar, a linear combination", v -> innerₕ(uₕ, v + 2 * D₋ₓ(v) - M₋ₓ(v))),
            ("scalar, two kinds summed", v -> innerₕ(uₕ, v) + inner₊ₓ(uₕ, D₋ₓ(v))))
            lfx = form(Wₕ, g)
            @test lfx(uₕ) ≈ sum(assemble(lfx) .* values(uₕ))
        end
        for (nm, g) in (
            ("composite shorthand", v -> innerₕ(uc, v)),
            ("composite per component", v -> innerₕ(cc[1], v(1)) + innerₕ(cc[2], v(2))),
            ("composite routed with operators",
            v -> innerₕ(cc[1], v(1) + D₋ₓ(v(1))) + innerₕ(cc[3], v(3))),
            ("composite crossed", v -> innerₕ(cc[1], v(2))))
            lfx = form(Vc, g)
            @test lfx(wc) ≈ sum(assemble(lfx) .* values(wc))
        end

        # A source carrying an operator is the one case that still allocates, and it is the
        # resolve rather than the contraction: `D₋ₓ(uₕ)` is evaluated eagerly into an
        # element of its own every time the tree is rebuilt. Caching the tree avoids it,
        # which is the same trade as the liveness one above.
        lfd = form(Wₕ, v -> innerₕ(D₋ₓ(uₕ), v))
        @test lfd(uₕ) ≈ sum(assemble(lfd) .* values(uₕ))

        @test _contract_allocs(lfd, uₕ) >= 8 * n    # one full-length element, from D₋ₓ

        # and hoisting the operator out of the form removes it, which is the documented
        # remedy rather than caching the tree
        duₕ = D₋ₓ(uₕ)
        lfh = form(Wₕ, v -> innerₕ(duₕ, v))
        @test lfh(uₕ) ≈ lfd(uₕ)
        @test _contract_allocs(lfh, uₕ) < 8 * n ÷ 100
    end

    @testset "evaluation sees live coefficients too" begin
        # Everything above went through `assemble`. Evaluation is the other way a form gets
        # used, and it has to agree: `l(vₕ)` and `evaluate!` both resolve from `f`, so a
        # coefficient changed between two calls is read again.
        uev = Rₕ(Wₕ, x -> 1.0)
        wₕ = Rₕ(Wₕ, x -> 1.0)
        lfe = form(Wₕ, v -> innerₕ(uev, v))

        first_value = lfe(wₕ)
        values(uev) .= 5.0
        @test lfe(wₕ) ≈ 5 * first_value

        # and a rebound element, which only a fresh resolve can see
        global _ev_rebound = Rₕ(Wₕ, x -> 1.0)
        lfer = form(Wₕ, v -> innerₕ(_ev_rebound, v))
        before_rebind = lfer(wₕ)
        global _ev_rebound = Rₕ(Wₕ, x -> 5.0)
        @test lfer(wₕ) ≈ 5 * before_rebind

        # a live scalar, through evaluation rather than assembly
        global _ev_alpha = 2.0
        us = Rₕ(Wₕ, x -> 1.0)
        lfes = form(Wₕ, v -> innerₕ(_ev_alpha * us, v))
        at_two = lfes(wₕ)
        global _ev_alpha = 10.0
        @test lfes(wₕ) ≈ 5 * at_two

        # `evaluate!` agrees with the functor on every one of those
        scratch = zeros(ndofs(Wₕ))
        @test evaluate!(scratch, lfes, wₕ) ≈ lfes(wₕ)
        @test evaluate!(scratch, lfer, wₕ) ≈ lfer(wₕ)

        # The documented loop: a resolved `ast` reused across calls, with the source written
        # through rather than rebound. This is the combination `evaluate!`'s docstring
        # recommends, so it is the one worth asserting stays live.
        ul = Rₕ(Wₕ, x -> 1.0)
        lfl = form(Wₕ, v -> innerₕ(ul, v))
        ast = resolve_form_ast(lfl)
        loop_first = evaluate!(scratch, lfl, wₕ; ast = ast)
        Rₕ!(ul, x -> 5.0)                        # written through, as the docstring says
        @test evaluate!(scratch, lfl, wₕ; ast = ast) ≈ 5 * loop_first

        # and the other half of that recommendation: a scalar is not live under a reused
        # `ast`, which is why the docstring tells a loop that changes one to resolve afresh
        global _loop_alpha = 2.0
        uk = Rₕ(Wₕ, x -> 1.0)
        lfk = form(Wₕ, v -> innerₕ(_loop_alpha * uk, v))
        kast = resolve_form_ast(lfk)
        frozen_first = evaluate!(scratch, lfk, wₕ; ast = kast)
        global _loop_alpha = 10.0
        @test evaluate!(scratch, lfk, wₕ; ast = kast) ≈ frozen_first   # frozen
        @test evaluate!(scratch, lfk, wₕ) ≈ 5 * frozen_first           # live when resolved
    end

    @testset "constant, function and tuple sources" begin
        # The source of a linear form need not be a grid function. A number and a function
        # both work, and an integer promotes rather than forcing the output's element type —
        # the same rule that lets a Dual through.
        Wc = gridspace(Ωₕ)
        one_over_Ω = 1.0                       # the mesh has measure 1, so ∫c = c

        @test sum(assemble(form(Wc, v -> innerₕ(1, v)))) ≈ one_over_Ω
        @test eltype(assemble(form(Wc, v -> innerₕ(1, v)))) === Float64
        @test sum(assemble(form(Wc, v -> innerₕ(2.5, v)))) ≈ 2.5
        @test sum(assemble(form(Wc, v -> innerₕ(x -> 1.0, v)))) ≈ one_over_Ω

        # and they mix with a grid-function source in one form
        gₕ = Rₕ(Wc, x -> 3.0)
        @test sum(assemble(form(Wc, v -> innerₕ(1.0, v) + innerₕ(gₕ, v)))) ≈ 4.0

        Vt = gridspace(Ωₕ, Val(2))
        nb = ndofs(Wc)
        blk(b) = [sum(b[((k - 1) * nb + 1):(k * nb)]) for k in 1:(length(b) ÷ nb)]

        # On a composite space a source naming no component goes to every block, which is
        # the rule that lets the two spellings mix.
        @test blk(assemble(form(Vt, v -> innerₕ(1.0, v)))) ≈ [1.0, 1.0]
        @test blk(assemble(form(Vt, v -> innerₕ(1.0, v(1))))) ≈ [1.0, 0.0]

        # A tuple reads one entry per component, the way `Rₕ(Vₕ, (f, g))` already does.
        @test blk(assemble(form(Vt, v -> innerₕ((1.0, 2.0), v)))) ≈ [1.0, 2.0]
        @test blk(assemble(form(Vt, v -> innerₕ((x -> 1.0, x -> 2.0), v)))) ≈ [1.0, 2.0]

        # which agrees with writing the components out
        @test assemble(form(Vt, v -> innerₕ((1.0, 2.0), v))) ≈
              assemble(form(Vt, v -> innerₕ(1.0, v(1)) + innerₕ(2.0, v(2))))

        # an empty tuple names nothing, and says so
        @test_throws ArgumentError form(Vt, v -> innerₕ((), v))
    end

    @testset "a component the space does not have is an error" begin
        # It used to contribute nothing, in silence. On a two-block space
        # `innerₕ(1.0, v(3))` assembled to zeros, and summed with a valid term it dropped
        # itself and kept the other, so a form written for a wider space quietly produced a
        # narrower answer instead of complaining.
        Vt = gridspace(Ωₕ, Val(2))
        b = zeros(ndofs(Vt))

        @test_throws ArgumentError assemble(form(Vt, v -> innerₕ(1.0, v(3))))
        @test_throws ArgumentError assemble(form(Vt, v -> innerₕ(1.0, v(0))))
        @test_throws ArgumentError assemble(form(Vt,
            v -> innerₕ(1.0, v(1)) + innerₕ(2.0, v(9))))

        # every route has to agree: the vector, the in-place vector, the contraction, and
        # the threaded sweep are four separate walks over the same terms
        lf3 = form(Vt, v -> innerₕ(1.0, v(3)))
        wt = Rₕ(Vt, (x -> 1.0, x -> 1.0))
        @test_throws ArgumentError assemble!(b, lf3)
        @test_throws ArgumentError lf3(wt)
        @test_throws ArgumentError assemble_parallel!(b, lf3)

        # and a tuple longer than the space is the same mistake, caught the same way
        @test_throws ArgumentError assemble(form(Vt, v -> innerₕ((1.0, 2.0, 3.0), v)))

        # while a valid one still works, so the guard is not simply rejecting everything
        @test sum(assemble(form(Vt, v -> innerₕ(1.0, v(2))))) ≈ 1.0
    end

    @testset "a form's expression is checked when it is built" begin
        # The `ast` field that used to hold a resolved tree was read by nothing, but its type
        # parameter did one useful thing: it rejected an expression that does not describe an
        # operator. Dropping the field kept the check and gave it a message.
        @test fieldnames(typeof(form(Wₕ, v -> innerₕ(uₕ, v)))) ==
              (:test_space, :f)

        @test_throws ArgumentError form(Wₕ, v -> 42)
        @test_throws ArgumentError form(Wₕ, v -> "not an operator")

        # and an expression built over a different dimension than the space it is given
        @test_throws ArgumentError form(Wₕ, v -> innerₕ(uₕ, TestFunction{3}()))
    end

    @testset "the symbolic assembly is the matrix expression" begin
        # A linear form is `v -> (Au, v)` for some operator `A` and inner product, so its
        # assembled vector has to be `Aᵀ H u` exactly: `H` the diagonal weight matrix of the
        # inner product, `A` the operator's own sparse matrix. Both are built by code that
        # shares nothing with `local_stencil`, which is what makes this an independent check
        # rather than a restatement — the two agree or one of them is wrong.
        n = ndofs(Wₕ)
        uu = values(uₕ)
        Hh = Diagonal(collect(weights(Wₕ, Innerh())))
        Hpx = Diagonal(collect(weights(Wₕ, Innerplus(), 1)))
        Dx = Matrix(D₋ₓ(Wₕ))
        Mx = Matrix(M₋ₓ(Wₕ))
        Idm = Matrix(1.0I, n, n)

        @test assemble(form(Wₕ, v -> innerₕ(uₕ, v))) ≈ Hh * uu
        @test assemble(form(Wₕ, v -> innerₕ(uₕ, D₋ₓ(v)))) ≈ transpose(Dx) * (Hh * uu)
        @test assemble(form(Wₕ, v -> innerₕ(uₕ, M₋ₓ(v)))) ≈ transpose(Mx) * (Hh * uu)
        @test assemble(form(Wₕ, v -> inner₊ₓ(uₕ, D₋ₓ(v)))) ≈ transpose(Dx) * (Hpx * uu)

        # a linear combination of operators in the test argument is the same combination of
        # their matrices, and inner products of different kinds add
        @test assemble(form(Wₕ, v -> innerₕ(uₕ, v + 2 * D₋ₓ(v) - M₋ₓ(v)))) ≈
              transpose(Idm + 2 * Dx - Mx) * (Hh * uu)
        @test assemble(form(Wₕ, v -> innerₕ(uₕ, v) + inner₊ₓ(uₕ, D₋ₓ(v)))) ≈
              Hh * uu + transpose(Dx) * (Hpx * uu)

        # and the Jacobian of a nonlinear residual is the same expression with the
        # nonlinearity's own derivative on the diagonal. This is the shape a Newton step
        # needs, so it is worth pinning: differentiating the assembly agrees with assembling
        # the derivative.
        resid(g) = w -> begin
            b = zeros(eltype(w), n)
            assemble!(b, form(Wₕ, v -> g(element(Wₕ, w .^ 2), v)))
            b
        end
        w0 = collect(range(0.3, 1.7; length = n))
        dg = Diagonal(2 .* w0)

        @test ForwardDiff.jacobian(resid((s, v) -> innerₕ(s, v)), w0) ≈ Hh * dg
        @test ForwardDiff.jacobian(resid((s, v) -> innerₕ(s, D₋ₓ(v))), w0) ≈
              transpose(Dx) * Hh * dg
        @test ForwardDiff.jacobian(resid((s, v) -> inner₊ₓ(s, D₋ₓ(v))), w0) ≈
              transpose(Dx) * Hpx * dg

        # the Jacobian's sparsity is the stencil's, which is what makes a sparse-AD colouring
        # unnecessary here: the pattern is known from the AST before anything is evaluated
        Jd = ForwardDiff.jacobian(resid((s, v) -> innerₕ(s, v)), w0)
        Jw = ForwardDiff.jacobian(resid((s, v) -> innerₕ(s, D₋ₓ(v))), w0)
        offs_d = length(stencil_offsets(resolve_form_ast(
            form(Wₕ, v -> innerₕ(uₕ, v)))))
        offs_w = length(stencil_offsets(resolve_form_ast(
            form(Wₕ, v -> innerₕ(uₕ, D₋ₓ(v))))))
        @test maximum(i -> count(!iszero, Jd[i, :]), 1:n) == offs_d
        @test maximum(i -> count(!iszero, Jw[i, :]), 1:n) == offs_w
    end

    @testset "Dirichlet conditions" begin
        bcs = dirichlet_constraints(set(Ωₕ), :bottom => (x -> 5.0))
        marked = index_in_marker(Ωₕ, :bottom)
        @test any(marked)

        b = assemble(form(Wₕ, v -> innerₕ(uₕ, v));
            dirichlet_conditions = bcs, dirichlet_labels = :bottom)
        @test all(b[marked] .≈ 5.0)

        # a tuple of labels, and an empty tuple which asks for nothing
        b2 = assemble(form(Wₕ, v -> innerₕ(uₕ, v));
            dirichlet_conditions = bcs, dirichlet_labels = (:bottom,))
        @test b2 ≈ b
        plain = assemble(form(Wₕ, v -> innerₕ(uₕ, v)))
        @test assemble(form(Wₕ, v -> innerₕ(uₕ, v)); dirichlet_conditions = bcs,
            dirichlet_labels = ()) ≈ plain

        # naming labels without conditions is a usage error rather than a silent no-op.
        # The conditions default to `nothing` now: they used to default to an empty
        # constraint set, built on every call and then discarded whenever no labels were
        # named — 2,080 B per assembly for an argument nothing read.
        @test_throws ArgumentError assemble(form(Wₕ, v -> innerₕ(uₕ, v));
            dirichlet_labels = :bottom)
        msg = try
            assemble(form(Wₕ, v -> innerₕ(uₕ, v)); dirichlet_labels = :bottom)
        catch e
            sprint(showerror, e)
        end
        @test occursin("dirichlet_conditions", msg)

        # and an invalid label type is rejected before anything is assembled
        @test_throws ErrorException assemble(form(Wₕ, v -> innerₕ(uₕ, v));
            dirichlet_conditions = bcs, dirichlet_labels = 3)
    end

    @testset "allocations, which are part of the contract" begin
        # This is what a time loop calls every step. The assembly kernel itself is
        # allocation free; what used to cost was the two default arguments — an empty
        # constraint set at 2,080 B and the AST resolution at 160 B, both recomputed per
        # call and both invariant for a given form.
        #
        # Measured inside a function on concrete locals: read from a non-const global, the
        # arguments box at the call boundary and the reading is of the box.
        function counts(N)
            Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (N, N), (true, true))
            W = gridspace(Ω)
            u = Rₕ(W, x -> x[1] + x[2])
            lf = form(W, v -> innerₕ(u, v))
            ast = resolve_form_ast(lf)
            b = zeros(ndofs(W))

            assemble!(b, lf)                      # warm both paths
            assemble!(b, lf; ast = ast)

            return (with_ast = @allocated(assemble!(b, lf; ast = ast)),
                without_ast = @allocated(assemble!(b, lf)))
        end

        for N in (8, 24)          # 9x the degrees of freedom apart
            c = counts(N)
            # handed a resolved AST, assembly allocates nothing at all
            @test c.with_ast == 0
            # and resolving costs a fixed amount that does not grow with the grid
            @test c.without_ast < 512
        end

        # the cost of resolving does not scale, which is what makes the keyword worth having
        @test counts(8).without_ast == counts(24).without_ast

        # the composite path too. It used to allocate a Vector of dof offsets on every
        # call — 128 B, constant in the grid but paid every step of a time loop.
        function composite_bytes(N)
            Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (N, N), (true, true))
            V = gridspace(Ω, Val(3))
            uv = Rₕ(V, ntuple(_ -> (x -> x[1] + x[2]), 3))
            lf = form(V, v -> innerₕ(uv(1), v))
            ast = resolve_form_ast(lf)
            b = zeros(ndofs(V))
            assemble!(b, lf; ast = ast)
            return @allocated assemble!(b, lf; ast = ast)
        end
        @test composite_bytes(8) == 0
        @test composite_bytes(16) == 0
    end

    @testset "differentiating an assembled residual" begin
        # The shape a nonlinear solve has: build the residual of a form, and let the solver
        # differentiate it with respect to the coefficient vector to get a Jacobian.
        #
        # This did not work. `assemble` allocated its output with `eltype(test_space(form))`
        # — the *space's* element type — so a Float64 space could only assemble a Float64
        # right-hand side, and writing a Dual weight into it met
        # `MethodError: no method matching Float64(::Dual)`. The type now comes from the
        # form's own weights, promoted against the space's, which is the rule `Rₕ` already
        # used and the same defect `dirichlet_constraints` had.
        u0 = values(uₕ)

        @testset "with respect to a parameter in the source" begin
            J(a) = sum(assemble(form(Wₕ, v -> innerₕ(Rₕ(Wₕ, x -> a * (x[1] + x[2])), v))))
            h = 1e-6
            @test isapprox(ForwardDiff.derivative(J, 1.3), (J(1.3 + h) - J(1.3 - h)) / 2h;
                rtol = 1e-5)
        end

        @testset "the Jacobian of a nonlinear residual" begin
            # innerₕ(u², v) assembles the vector with entries |□ᵢ| uᵢ², so its Jacobian is
            # diagonal with 2 |□ᵢ| uᵢ. Checked against that closed form rather than against
            # a finite difference, which is a stronger statement.
            residual(u) = assemble(form(Wₕ, v -> innerₕ(element(Wₕ, u .* u), v)))
            Jm = ForwardDiff.jacobian(residual, u0)

            @test size(Jm) == (n, n)
            @test Jm ≈ Diagonal(diag(Jm))

            measures = assemble(form(Wₕ, v -> innerₕ(element(Wₕ, ones(n)), v)))
            @test diag(Jm) ≈ 2 .* measures .* u0

            # and the undifferentiated path is untouched
            @test eltype(residual(u0)) === Float64
        end

        @testset "and with the constraints applied, as a solve would" begin
            bcs = dirichlet_constraints(set(Ωₕ), :bottom => (x -> 0.0))
            res(u) = assemble(form(Wₕ, v -> innerₕ(element(Wₕ, u .* u), v));
                dirichlet_conditions = bcs, dirichlet_labels = :bottom)
            Jb = ForwardDiff.jacobian(res, u0)
            @test size(Jb) == (n, n)

            # a pinned value does not depend on the coefficients, so its row is zero —
            # which is what a solver needs in order not to move it
            marked = index_in_marker(Ωₕ, :bottom)
            @test any(marked)
            @test all(iszero, Jb[marked, :])
        end
    end

    @testset "the functor contracts against a vector" begin
        lf = form(Wₕ, v -> innerₕ(uₕ, v))
        b = assemble(lf)
        ones_el = Rₕ(Wₕ, x -> 1.0)
        @test lf(ones_el) ≈ sum(b)        # against the all-ones element, the sum
        @test lf(uₕ) ≈ sum(b .* values(uₕ))

        # A bare vector is refused rather than contracted. Its length carries no claim about
        # whether its blocks match the components a form routes to, so accepting one would
        # make a composite mismatch silent.
        @test_throws ArgumentError lf(values(uₕ))
        @test_throws ArgumentError lf(fill(1.0, ndofs(Wₕ)))

        # on a composite space, where the components have to line up with the blocks rather
        # than merely add up to the right length
        Vc = gridspace(Ωₕ, Val(2))
        uc = Rₕ(Vc, (x -> 1.0, x -> 3.0))
        wc = Rₕ(Vc, (x -> 2.0, x -> 5.0))
        lfv = form(Vc, v -> innerₕ(uc, v))
        @test lfv(wc) ≈ sum(assemble(lfv) .* values(wc))
        @test_throws ArgumentError lfv(values(wc))

        # `evaluate!` agrees, and reuses its scratch rather than assembling afresh
        scratch = zeros(ndofs(Wₕ))
        @test evaluate!(scratch, lf, uₕ) ≈ lf(uₕ)
        @test scratch ≈ b                          # the scratch holds the assembled vector

        # repeatable, so the scratch is rewritten rather than accumulated into
        @test evaluate!(scratch, lf, uₕ) ≈ evaluate!(scratch, lf, uₕ)

        # resolving once across a loop gives the same answer as resolving per call
        ast = resolve_form_ast(lf)
        @test evaluate!(scratch, lf, uₕ; ast = ast) ≈ lf(uₕ)

        scratchv = zeros(ndofs(Vc))
        @test evaluate!(scratchv, lfv, wc) ≈ lfv(wc)
        @test_throws ArgumentError evaluate!(scratchv, lfv, values(wc))

        # and it allocates nothing per call, once the AST is resolved outside the loop.
        #
        # Behind a barrier, like every other allocation assertion here: measured at testset
        # top level this reports the bytes of the surrounding closure rather than of the
        # call.
        #
        # One callsite, measured three times, and the steady state is what is asserted.
        #
        # On Julia nightly the first measurement at any given callsite carries a one-time
        # 16 bytes and every later one is 0. The raw triples:
        #
        #     1.14.0-DEV.3077  (16, 0, 0)
        #     1.13.0-rc3       ( 0, 0, 0)
        #     1.12.7           ( 0, 0, 0)
        #
        # The cost is per-callsite, not per-process, which is why neither of the simpler
        # fixes works. A plain warmup call does not absorb it — the warmup's return value is
        # unused, so the `dot` inside it can be optimised away — and neither does a
        # discarded `@allocated` written as a second statement, because that is a different
        # callsite and pays its own one-time 16 bytes. Repeating one callsite is what
        # reaches the steady state.
        #
        # It is `dot`, and it is upstream rather than anything here. `assemble!(; ast)`
        # measures 0 on the same build, a bare `dot(::Vector{Float64}, ::Vector{Float64})`
        # reproduces the identical (16, 0, 0) with no Bramble in the picture, and
        # `@allocated` over a non-allocating expression is 0 on all three, so it is not the
        # macro. An earlier version of this comment blamed the `ast` keyword box; that was
        # wrong, and the keyword has nothing to do with it.
        #
        # What is under test is that a time loop calling this does not allocate, and the
        # steady state is exactly that property.
        function _evaluate_bytes(scratch, lf, v, ast)
            evaluate!(scratch, lf, v; ast = ast)
            return minimum(ntuple(_ -> @allocated(evaluate!(scratch, lf, v; ast = ast)), 3))
        end
        @test _evaluate_bytes(scratch, lf, uₕ, ast) == 0
    end
end
