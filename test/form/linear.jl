module FormLinearTests

using Test
using Bramble
using Bramble: execution_policy
# Internal names: defined and documented, not exported.
import Bramble: M₊ᵧ
using ForwardDiff
using LinearAlgebra: Diagonal, diag, dot, I
using Bramble:
               LinearForm,
               form,
               assemble,
               assemble!,
               assemble_parallel!,
               test_space,
               element,
               resolve_form_ast,
               apply_dirichlet_conditions!,
               LinearProduct,
               TestFunction,
               TrialFunction,
               IndexedTestFunction,
               IndexedTrialFunction,
               test_component_or_nothing,
               components,
               _colour_strides,
               stencil_offsets,
               ndofs,
               restrict_to,
               Innerh,
               Innerplus,
               evaluate!,
               D̽ₓ,
               D₋ᵧ,
               D₋ₓ,
               Mᵧ,
               Mₓ,
               index_in_marker,
               inner₊ₓ,
               jumpₓ,
               weights
using ..TestUtils: alloc_test, @test_allocs, WITH_AD_TESTS

# Assembling the right-hand side of a system.
#
# `form(Wₕ, v -> …)` closes over a test function and records the AST; `assemble` walks the
# grid, evaluates the stencil at each point and scatters the weights into a vector. This is
# the file a time loop calls every step, so the allocation counts below are part of its
# contract rather than a nicety.
#
# The values are checked against integrals that can be worked out by hand: `innerₕ(uₕ, v)`
# assembles the vector whose entries are ``|□ᵢ| uᵢ``, so summing it gives ``∫ u`` over the
# domain to the accuracy of the quadrature (exactly, for the cases below).

@testset "Linear forms" begin
    Ωₕ = mesh(
        domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
        (8, 8),
        (true, true)
    )
    Wₕ = gridspace(Ωₕ)
    Vₕ = gridspace(Ωₕ, Val(2))
    uₕ = Rₕ(Wₕ, x -> x[1] + x[2])
    n = ndofs(Wₕ)

    @testset "Construction" begin
        lf = form(Wₕ, v -> innerₕ(uₕ, v))
        @test lf isa LinearForm
        @test test_space(lf) === Wₕ
        @test resolve_form_ast(lf) isa LinearProduct
    end

    @testset "Assembly" begin
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

    @testset "Composite space" begin
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

    @testset "Nested composite space" begin
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

    @testset "Heterogeneous composite spaces" begin
        # `lin_indices`/`mesh_markers` used to be built once from the composite space's
        # first leaf and handed to every leaf's own assembly walk. For a homogeneous
        # composite every leaf shares one size, so nothing caught it; for a genuinely
        # heterogeneous one (built directly from a tuple of differently-sized leaves,
        # bypassing `gridspace(Ω, Val(N))`, which always broadcasts one shared mesh), any
        # leaf past the first ran its own indices through leaf 1's (smaller or larger)
        # `LinearIndices` and either threw `BoundsError` or, worse, read the wrong leaf's
        # data at a coincidentally in-bounds index.
        Ωbig = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (8, 8), (true, true))
        Ωsmall = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (4, 4), (true, true))
        Wbig, Wsmall = gridspace(Ωbig), gridspace(Ωsmall)
        Vh = Bramble.CompositeGridSpace((Wbig, Wsmall))
        @test ndofs(Vh) == ndofs(Wbig) + ndofs(Wsmall)

        # non-constant, and different per leaf, so an index mix-up would misread a value
        # rather than merely surviving by coincidence
        uv = Rₕ(Vh, (x -> x[1] + x[2], x -> 2x[1] - x[2]))

        # leaf 1 alone: already worked before this fix, kept as the reference case
        b1 = assemble(form(Vh, v -> innerₕ(uv(1), v(1))))
        expected1 = sum(parent(Rₕ(Wbig, x -> x[1] + x[2])) .* weights(Wbig, Innerh()))
        @test sum(b1) ≈ expected1
        @test all(iszero, b1[(ndofs(Wbig) + 1):end])   # leaf 2's block untouched

        # leaf 2 alone: this is exactly what used to throw BoundsError
        b2 = assemble(form(Vh, v -> innerₕ(uv(2), v(2))))
        expected2 = sum(parent(Rₕ(Wsmall, x -> 2x[1] - x[2])) .* weights(Wsmall, Innerh()))
        @test sum(b2[(ndofs(Wbig) + 1):end]) ≈ expected2
        @test all(iszero, b2[1:ndofs(Wbig)])            # leaf 1's block untouched

        # both together, routed: each block gets its own leaf's own answer
        lboth = form(Vh, v -> innerₕ(uv(1), v(1)) + innerₕ(uv(2), v(2)))
        bboth = assemble(lboth)
        @test bboth[1:ndofs(Wbig)] ≈ b1[1:ndofs(Wbig)]
        @test bboth[(ndofs(Wbig) + 1):end] ≈ b2[(ndofs(Wbig) + 1):end]

        # contraction takes the same walk, so it needs the same fix independently:
        # l(uv) = Σᵢ bᵢ uvᵢ, not Σᵢ bᵢ (uv is not constant here on purpose)
        @test lboth(uv) ≈ dot(bboth, parent(uv))

        # assemble! into a pre-allocated vector: the everyday, allocation-free call
        b3 = similar(bboth)
        assemble!(b3, lboth)
        @test b3 ≈ bboth

        # the parallel core walks leaves the same way and needs its own leaf's mesh too
        bp = similar(bboth)
        assemble_parallel!(bp, lboth)
        @test bp ≈ bboth
    end

    @testset "Coupled right-hand side" begin
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

        @testset "Indexed test function" begin
            v = TestFunction{2}()
            @test v(2) === IndexedTestFunction{2}(2)
            @test TrialFunction{2}()(1) === IndexedTrialFunction{2}(1)
        end

        @testset "Block placement" begin
            # ∫x = 0.5 into the first block, ∫10x = 5.0 into the second
            b = assemble(form(Vₕ, v -> innerₕ(uv(1), v(1)) + innerₕ(uv(2), v(2))))
            @test blocks(b) ≈ [0.5, 5.0]

            # one term alone reaches only its own block
            @test blocks(assemble(form(Vₕ, v -> innerₕ(uv(2), v(2))))) ≈ [0.0, 5.0]
        end

        @testset "Unindexed broadcast" begin
            # The two spellings mix, because routing is decided per term
            mixed = assemble(form(Vₕ, v -> innerₕ(uv(1), v) + innerₕ(uv(2), v(2))))
            @test blocks(mixed) ≈ [0.5, 0.5 + 5.0]
        end

        @testset "Routing query" begin
            v = TestFunction{2}()
            @test test_component_or_nothing(innerₕ(uv(1), v(2))) == 2
            @test test_component_or_nothing(innerₕ(uv(1), v)) === nothing
            @test test_component_or_nothing(innerₕ(uv(1), D₋ₓ(v(2)))) == 2
        end

        @testset "Zero-cost routing" begin
            # The routing recurses the AST rather than flattening it first, which answers
            # with a Vector{Any}: that allocated 544 B per assembly and made every term a
            # dynamic read.
            function bytes(mk)
                Ω = mesh(
                    domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 8), (true, true)
                )
                V = gridspace(Ω, Val(3))
                u = Rₕ(V, ntuple(_ -> (x -> x[1] + x[2]), 3))
                lf = form(V, mk(u))
                b = zeros(ndofs(V))
                assemble!(b, lf)
                return @allocated assemble!(b, lf)
            end
            @test bytes(u -> (v -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))) == 0
        end
    end

    @testset "Linear combination of products" begin
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
        # function must give bᵀw. That validates the whole path (routing, offsets, weights
        # and truncation) against code that has nothing to do with assembly.
        Ωf = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 11), (true, false))
        Wf = gridspace(Ωf)
        Vf = gridspace(Ωf, Val(3))

        @testset "Scalar" begin
            g1 = Rₕ(Wf, x -> x[1] + 2x[2])
            g2 = Rₕ(Wf, x -> exp(x[1]))
            g3 = Rₕ(Wf, x -> 1 + x[2]^2)
            w = Rₕ(Wf, x -> sin(3x[1]) * cos(2x[2]) + 1)

            b = assemble(
                form(
                Wf,
                v -> innerₕ(g1, v + 2 * D₋ₓ(v) - Mₓ(v)) +
                     inner₊ₓ(g2, D₋ᵧ(v) + jumpₓ(v)) +
                     innerₕ(g3, 3 * M₊ᵧ(v) - D̽ₓ(v))
            ),
            )

            reference = innerₕ(g1, w + 2 * D₋ₓ(w) - Mₓ(w)) +
                        inner₊ₓ(g2, D₋ᵧ(w) + jumpₓ(w)) +
                        innerₕ(g3, 3 * M₊ᵧ(w) - D̽ₓ(w))

            @test dot(b, parent(w)) ≈ reference
            @test !iszero(reference)          # the identity is not being met by both sides
        end                                   # being zero

        @testset "Composite shorthand" begin
            gv = Rₕ(Vf, (x -> x[1] + 2x[2], x -> exp(x[1]), x -> 1 + x[2]^2))
            wv = Rₕ(Vf, (x -> sin(3x[1]) + 1, x -> cos(2x[2]) + 2, x -> x[1] * x[2] + 1))

            b = assemble(form(Vf, v -> innerₕ(gv, v + 2 * D₋ₓ(v) - Mₓ(v))))
            reference = sum(
                innerₕ(components(gv)[c], (w = components(wv)[c]; w + 2 * D₋ₓ(w) - Mₓ(w)))
            for c in 1:3
            )

            @test dot(b, parent(wv)) ≈ reference
            @test !iszero(reference)
        end

        @testset "Shorthand equivalence" begin
            # Which is the property that makes it a shorthand rather than a second meaning.
            uv = Rₕ(Vf, (x -> x[1], x -> 100 * x[1], x -> x[2]))
            # Any: each closure pair is its own type
            for (short, long) in Any[
                (
                    v -> innerₕ(uv, v),
                    v -> innerₕ(uv(1), v(1)) + innerₕ(uv(2), v(2)) + innerₕ(uv(3), v(3))
                ),
                (
                    v -> innerₕ(uv, v + D₋ₓ(v)),
                    v -> innerₕ(uv(1), v(1) + D₋ₓ(v(1))) +
                         innerₕ(uv(2), v(2) + D₋ₓ(v(2))) +
                         innerₕ(uv(3), v(3) + D₋ₓ(v(3)))
                ),
                (
                    v -> innerₕ(uv, v + 2 * D₋ₓ(v) - Mₓ(v)),
                    v -> innerₕ(uv(1), v(1) + 2 * D₋ₓ(v(1)) - Mₓ(v(1))) +
                         innerₕ(uv(2), v(2) + 2 * D₋ₓ(v(2)) - Mₓ(v(2))) +
                         innerₕ(uv(3), v(3) + 2 * D₋ₓ(v(3)) - Mₓ(v(3)))
                ),
                (
                    v -> inner₊ₓ(uv, v - M₊ᵧ(v)),
                    v -> inner₊ₓ(uv(1), v(1) - M₊ᵧ(v(1))) +
                         inner₊ₓ(uv(2), v(2) - M₊ᵧ(v(2))) +
                         inner₊ₓ(uv(3), v(3) - M₊ᵧ(v(3)))
                )
            ]
                @test assemble(form(Vf, short)) ≈ assemble(form(Vf, long))
            end

            # and the components really are distinguishable, so the check has teeth: the
            # bug it guards against put every component into every block, which agrees with
            # the truth whenever the components integrate to the same thing
            b = assemble(form(Vf, v -> innerₕ(uv, v)))
            m = ndofs(Wf)
            @test [sum(b[(k * m + 1):((k + 1) * m)]) for k in 0:2] ≈ [0.5, 50.0, 0.5]
        end

        @testset "Index distribution" begin
            v = TestFunction{2}()
            @test (v + D₋ₓ(v))(1) == v(1) + D₋ₓ(v(1))
            @test (3 * Mᵧ(v))(2) == 3 * Mᵧ(v(2))
            @test (v + 2 * D₋ₓ(v) - Mₓ(v))(2) == v(2) + 2 * D₋ₓ(v(2)) - Mₓ(v(2))
            @test v(1)(2) === IndexedTestFunction{2}(2)      # re-indexing replaces

            # a sum inside one product takes the component its sides agree on
            @test test_component_or_nothing(innerₕ(Rₕ(Wf, x -> 1.0), v(2) + D₋ₓ(v(2)))) == 2
            @test test_component_or_nothing(innerₕ(Rₕ(Wf, x -> 1.0), v + D₋ₓ(v))) ===
                  nothing

            # and sides naming different components are ill-formed rather than ambiguous
            @test_throws ArgumentError test_component_or_nothing(
                innerₕ(Rₕ(Wf, x -> 1.0), v(1) + v(2))
            )
        end
    end

    @testset "Parallel vs serial agreement" begin
        # Note what this does and does not establish. `Pkg.test()` runs on one thread unless
        # the environment says otherwise, and on one thread the sweep is a degenerate case:
        # the colours never actually run concurrently, so nothing can race. CI sets
        # JULIA_NUM_THREADS=auto, so it is exercised there. The concurrent assertions below
        # are skipped rather than passed when only one thread is available, so a
        # single-threaded run cannot claim to have tested concurrency.
        # @info "parallel assembly tested on $(Threads.nthreads()) thread(s)"

        @testset "Sweep colouring" begin
            # Two points of one colour must have disjoint write footprints, so the stride is
            # the span of the reach plus one.
            @test _colour_strides([(0,)]) == (1,)
            @test _colour_strides([(0,), (-1,)]) == (2,)
            @test _colour_strides([(-1, 0), (0, 0), (1, 0)]) == (3, 1)
            @test _colour_strides([(0, -1), (0, 0)]) == (1, 2)
            @test _colour_strides(NTuple{2, Int}[]) == (1, 1)

            # and read off a real form, an operator reaching only its own point gives one
            # colour (the whole grid in a single flat pass), while a difference gives two
            plain = _colour_strides(
                stencil_offsets(resolve_form_ast(form(Wₕ, v -> innerₕ(uₕ, v))))
            )
            wide = _colour_strides(
                stencil_offsets(resolve_form_ast(form(Wₕ, v -> innerₕ(uₕ, D₋ₓ(v)))))
            )
            @test prod(plain) == 1
            @test prod(wide) == 2
        end

        # Any: each space and source is its own type
        for (nm, sp, u) in Any[
            ("scalar", Wₕ, uₕ),
            ("composite", Vₕ, Rₕ(Vₕ, (x -> sin(x[1]), x -> cos(x[2])))(1))
        ]
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

        WITH_AD_TESTS && @testset "Parallel differentiation" begin
            # The per-thread buffers this path used to carry were `Vector{Float64}`
            # outright, so a Dual-valued assembly could not take it at all. Nothing in the
            # sweep names an element type now, so it can.
            Ωa = mesh(
                domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (12, 12), (true, true)
            )
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
            Ωb = mesh(
                domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (40, 40), (true, true)
            )
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
            # Any: each closure is its own type
            for (cnm, g) in Any[
                ("one difference", v -> innerₕ(ub, D₋ₓ(v))),
                ("a linear combination", v -> innerₕ(ub, v + 2 * D₋ₓ(v) - Mₓ(v))),
                ("innerₕ and inner₊ mixed", v -> innerₕ(ub, v) + inner₊(ub, D₋ₓ(v))),
                (
                    "differences in both directions",
                    v -> innerₕ(ub, D₋ₓ(v)) + innerₕ(ub, D₋ᵧ(v))
                )
            ]
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
            # Any: each closure is its own type
            for (cnm, g) in Any[
                ("per component", v -> innerₕ(cv[1], v(1)) + innerₕ(cv[2], v(2))),
                ("the shorthand", v -> innerₕ(uv2, v)),
                ("the shorthand with operators", v -> innerₕ(uv2, v + D₋ₓ(v))),
                (
                    "routed, with operators",
                    v -> innerₕ(cv[1], v(1) + 2 * D₋ₓ(v(1))) + innerₕ(cv[2], v(2))
                ),
                ("crossed components", v -> innerₕ(cv[1], v(2))),
                (
                    "a routed term beside an unrouted one",
                    v -> innerₕ(cv[1], v(1)) + innerₕ(uv2, v)
                )
            ]
                lfc = form(Vb, g)
                bc = assemble(lfc)
                bcp = similar(bc)
                assemble_parallel!(bcp, lfc)
                @test bcp ≈ bc
            end

            # A heterogeneous leaf under real concurrency, same "more work than
            # threads" sizing as the homogeneous stress case above: a race in the scatter,
            # or a colour built from the wrong leaf's `LinearIndices`, would show here.
            Ωb_small = mesh(
                domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (17, 17), (true, true)
            )
            Vhet = Bramble.CompositeGridSpace((Wb, gridspace(Ωb_small)))
            uhet = Rₕ(Vhet, (x -> x[1] * x[2] + 1, x -> x[1] - 2x[2]))
            # Any: each closure is its own type
            for (cnm, g) in Any[
                ("per component", v -> innerₕ(uhet(1), v(1)) + innerₕ(uhet(2), v(2))),
                (
                    "routed, with operators",
                    v -> innerₕ(uhet(1), v(1) + 2 * D₋ₓ(v(1))) + innerₕ(uhet(2), v(2))
                )
            ]
                lfh = form(Vhet, g)
                bh = assemble(lfh)
                bhp = similar(bh)
                assemble_parallel!(bhp, lfh)
                @test bhp ≈ bh

                # repeated runs agree, which a race would break
                bhp2 = similar(bh)
                assemble_parallel!(bhp2, lfh)
                @test bhp2 == bhp
            end
        else
            @test_skip "concurrency not exercised: only one thread available"
            @test_skip "repeatability under threads not exercised"
            @test_skip "multi-colour parallel path under threads not exercised"
            @test_skip "coupled parallel path under threads not exercised"
            @test_skip "heterogeneous composite parallel path under threads not exercised"
        end
    end

    @testset "Backend policy" begin
        # assemble!/assemble no longer hardcode serial: they read test_space(form)'s
        # execution_policy and dispatch to the same serial/parallel cores assemble_parallel!
        # uses, so a Parallel()-backend form threads through the plain assemble!/assemble
        # call, not only through the separate assemble_parallel! entry point.
        Ω_par = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
            (8, 8),
            (true, true);
            backend = backend(policy = Parallel())
        )
        W_par = gridspace(Ω_par)
        u_par = Rₕ(W_par, x -> x[1] + x[2])
        @test execution_policy(W_par) isa Parallel

        lf_serial = form(Wₕ, v -> innerₕ(uₕ, v))
        lf_parallel = form(W_par, v -> innerₕ(u_par, v))

        b_default = assemble(lf_serial)
        b_via_policy = assemble(lf_parallel)
        @test b_via_policy ≈ b_default

        bang_default = zeros(ndofs(Wₕ))
        bang_via_policy = zeros(ndofs(W_par))
        assemble!(bang_default, lf_serial)
        assemble!(bang_via_policy, lf_parallel)
        @test bang_via_policy ≈ bang_default

        # Directly against assemble_parallel!, the lower-level entry point that always
        # threads regardless of the backend's policy: a Parallel()-backend assemble! must
        # agree with it exactly, not merely with the serial answer up to tolerance.
        b_forced_parallel = similar(bang_via_policy)
        assemble_parallel!(b_forced_parallel, lf_parallel)
        @test bang_via_policy ≈ b_forced_parallel
    end

    @testset "Reassembly" begin
        # The AST is stored on the form and references the underlying VectorElement arrays,
        # so updating an element in-place via `parent(us) .= ...` or `Rₕ!(us, ...)` is
        # automatically seen without allocating a new AST.
        us = Rₕ(Wₕ, x -> 1.0)
        lfs = form(Wₕ, v -> innerₕ(us, v))

        first_sum = sum(assemble(lfs))
        parent(us) .= 5.0
        @test sum(assemble(lfs)) ≈ 5 * first_sum

        # Dynamic scalar coefficients use Julia-native RefValue:
        # `α = Ref(2.0)` with `α * v` or `α * innerₕ(...)`. Mutating `α[] = 10.0`
        # is seen live during assembly with zero allocations.
        α_ref = Ref(2.0)
        ua_scalar = Rₕ(Wₕ, x -> 1.0)
        lfa_scalar = form(Wₕ, v -> α_ref * innerₕ(ua_scalar, v))
        at_two = sum(assemble(lfa_scalar))
        α_ref[] = 10.0
        @test sum(assemble(lfa_scalar)) ≈ 5 * at_two

        # both at once: a dynamic Ref scalar and in-place element update
        α_ref[] = 3.0
        parent(ua_scalar) .= 2.0
        @test sum(assemble(lfa_scalar)) ≈ 3 * at_two   # 3 x 2 against 2 x 1

        # In-place assembly allocates zero bytes
        d = zeros(ndofs(Wₕ))
        @test_allocs assemble!(d, lfa_scalar)
    end

    @testset "Direct contraction" begin
        # `l(vₕ)` answers with a number, and used to allocate a whole right-hand side to
        # get it: `dot(assemble(form), parent(vₕ))`. The walk now multiplies each stencil
        # weight by `v` at the row it would have written to, so the same sum is taken as it
        # goes and no vector is built.
        n = ndofs(Wₕ)
        lfc = form(Wₕ, v -> innerₕ(uₕ, v))
        astc = resolve_form_ast(lfc)
        b = assemble(lfc)

        @test lfc(uₕ) ≈ sum(b .* parent(uₕ))

        # The functor takes no `ast` on purpose, so it resolves every call and cannot be
        # allocation free. What it must not do is build a full-length vector, which is what
        # this pins: the bound is a fraction of one, not zero. Behind a barrier, because at
        # top level over globals the call itself reports bytes it does not spend.
        @test alloc_test(lfc, uₕ) < 8 * n ÷ 100
        @test assemble(lfc) ≈ b                     # and the vector path still works

        # Every arrangement has to agree with assembling and contracting by hand, not just
        # the simple one: the composite cores are separate code from the scalar one.
        Vc = gridspace(Ωₕ, Val(3))
        uc = Rₕ(Vc, (x -> 1.0, x -> 100.0, x -> 10000.0))
        wc = Rₕ(Vc, (x -> 2.0, x -> 3.0, x -> 5.0))
        cc = components(uc)
        # Any: each closure is its own type
        for (nm, g) in Any[
            ("scalar, a difference", v -> innerₕ(uₕ, D₋ₓ(v))),
            ("scalar, a linear combination", v -> innerₕ(uₕ, v + 2 * D₋ₓ(v) - Mₓ(v))),
            ("scalar, two kinds summed", v -> innerₕ(uₕ, v) + inner₊ₓ(uₕ, D₋ₓ(v)))
        ]
            lfx = form(Wₕ, g)
            @test lfx(uₕ) ≈ sum(assemble(lfx) .* parent(uₕ))
        end
        for (nm, g) in Any[
            ("composite shorthand", v -> innerₕ(uc, v)),
            ("composite per component", v -> innerₕ(cc[1], v(1)) + innerₕ(cc[2], v(2))),
            (
                "composite routed with operators",
                v -> innerₕ(cc[1], v(1) + D₋ₓ(v(1))) + innerₕ(cc[3], v(3))
            ),
            ("composite crossed", v -> innerₕ(cc[1], v(2)))
        ]
            lfx = form(Vc, g)
            @test lfx(wc) ≈ sum(assemble(lfx) .* parent(wc))
        end

        # A source carrying an operator is evaluated into an element during form construction
        lfd = form(Wₕ, v -> innerₕ(D₋ₓ(uₕ), v))
        @test lfd(uₕ) ≈ sum(assemble(lfd) .* parent(uₕ))
        @test alloc_test(lfd, uₕ) < 8 * n ÷ 100

        # and hoisting the operator out of the form agrees
        duₕ = D₋ₓ(uₕ)
        lfh = form(Wₕ, v -> innerₕ(duₕ, v))
        @test lfh(uₕ) ≈ lfd(uₕ)
        @test alloc_test(lfh, uₕ) < 8 * n ÷ 100
    end

    @testset "Live coefficient evaluation" begin
        # Everything above went through `assemble`. Evaluation is the other way a form gets
        # used, and it has to agree: `l(vₕ)` and `evaluate!` both resolve from `f`, so a
        # coefficient changed between two calls is read again.
        uev = Rₕ(Wₕ, x -> 1.0)
        wₕ = Rₕ(Wₕ, x -> 1.0)
        lfe = form(Wₕ, v -> innerₕ(uev, v))

        first_value = lfe(wₕ)
        parent(uev) .= 5.0
        @test lfe(wₕ) ≈ 5 * first_value

        # a live scalar via RefValue, through evaluation rather than assembly
        α_eval = Ref(2.0)
        us = Rₕ(Wₕ, x -> 1.0)
        lfes = form(Wₕ, v -> α_eval * innerₕ(us, v))
        at_two = lfes(wₕ)
        α_eval[] = 10.0
        @test lfes(wₕ) ≈ 5 * at_two

        # `evaluate!` agrees with the functor
        scratch = zeros(ndofs(Wₕ))
        @test evaluate!(scratch, lfes, wₕ) ≈ lfes(wₕ)
        @test evaluate!(scratch, lfe, wₕ) ≈ lfe(wₕ)

        # In-place evaluation writes through
        ul = Rₕ(Wₕ, x -> 1.0)
        lfl = form(Wₕ, v -> innerₕ(ul, v))
        loop_first = evaluate!(scratch, lfl, wₕ)
        Rₕ!(ul, x -> 5.0)
        @test evaluate!(scratch, lfl, wₕ) ≈ 5 * loop_first
    end

    @testset "Source variants" begin
        # The source of a linear form need not be a grid function. A number and a function
        # both work, and an integer promotes rather than forcing the output's element type:
        # the same rule that lets a Dual through.
        Wc = gridspace(Ωₕ)
        one_over_Ω = 1.0                       # the mesh has measure 1, so ∫c = c

        @test sum(assemble(form(Wc, v -> innerₕ(1, v)))) ≈ one_over_Ω
        @test eltype(assemble(form(Wc, v -> innerₕ(1, v)))) === Float64

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

    @testset "Invalid component error" begin
        # It used to contribute nothing, in silence. On a two-block space
        # `innerₕ(1.0, v(3))` assembled to zeros, and summed with a valid term it dropped
        # itself and kept the other, so a form written for a wider space quietly produced a
        # narrower answer instead of complaining.
        Vt = gridspace(Ωₕ, Val(2))
        b = zeros(ndofs(Vt))

        @test_throws ArgumentError assemble(form(Vt, v -> innerₕ(1.0, v(3))))
        @test_throws ArgumentError assemble(form(Vt, v -> innerₕ(1.0, v(0))))
        @test_throws ArgumentError assemble(
            form(Vt, v -> innerₕ(1.0, v(1)) + innerₕ(2.0, v(9)))
        )

        # every route has to agree: the vector, the in-place vector, the contraction, and
        # the threaded sweep are four separate walks over the same terms
        @test_throws ArgumentError form(Vt, v -> innerₕ(1.0, v(3)))
        raw_ast = innerₕ(1.0, IndexedTestFunction{2}(3))
        lf3 = Bramble.LinearForm{2, typeof(Vt), typeof(raw_ast)}(Vt, raw_ast)
        wt = Rₕ(Vt, (x -> 1.0, x -> 1.0))
        @test_throws ArgumentError assemble!(b, lf3)
        @test_throws ArgumentError lf3(wt)
        @test_throws ArgumentError assemble_parallel!(b, lf3)

        # and a tuple longer than the space is the same mistake, caught the same way
        @test_throws ArgumentError assemble(form(Vt, v -> innerₕ((1.0, 2.0, 3.0), v)))

        # while a valid one still works, so the guard is not simply rejecting everything
        @test sum(assemble(form(Vt, v -> innerₕ(1.0, v(2))))) ≈ 1.0
    end

    @testset "Wrong-length vector refused" begin
        # The sweep writes with `@inbounds`, so a short vector would be written out of
        # bounds and a long one would keep a stale tail. Every linear entry
        # point checks the length before writing anything, so the vector comes back
        # untouched.
        # Any: each mesh, space and closure is its own type
        for (Ωw, dim) in Any[(mesh(domain(interval(0.0, 1.0)), 13, false), "1D"),
            (mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 11), (false, false)),
                "2D")]
            Ww = gridspace(Ωw)
            for (label, sp, l) in Any[("scalar", Ww, v -> innerₕ(1.0, v)),
                ("composite", Ww × Ww, V -> innerₕ(1.0, V[1]) + innerₕ(2.0, V[2]))]
                lw = form(sp, l)
                @test execution_policy(sp) isa Bramble.CpuSerial
                for m in (ndofs(sp) - 5, ndofs(sp) + 5)
                    @testset "$dim $label, length $m" begin
                        b = fill(7.0, m)
                        @test_throws DimensionMismatch assemble!(b, lw)
                        @test all(==(7.0), b)
                        @test_throws DimensionMismatch Bramble.assemble_add!(b, lw)
                        @test all(==(7.0), b)
                        @test_throws DimensionMismatch Bramble.assemble_add!(b, lw, 2.0)
                        @test all(==(7.0), b)
                        @test_throws DimensionMismatch assemble_parallel!(b, lw)
                        @test all(==(7.0), b)
                    end
                end
            end
        end
    end

    @testset "Expression validation" begin
        # The `ast` field stores the pre-resolved tree; the expression itself is not kept
        # since nothing downstream ever calls it again.
        @test fieldnames(typeof(form(Wₕ, v -> innerₕ(uₕ, v)))) == (:test_space, :ast)

        @test_throws ArgumentError form(Wₕ, v -> 42)
        @test_throws ArgumentError form(Wₕ, v -> "not an operator")

        # and an expression built over a different dimension than the space it is given
        @test_throws ArgumentError form(Wₕ, v -> innerₕ(uₕ, TestFunction{3}()))
    end

    @testset "Matrix expression equivalence" begin
        # A linear form is `v -> (Au, v)` for some operator `A` and inner product, so its
        # assembled vector has to be `Aᵀ H u` exactly: `H` the diagonal weight matrix of the
        # inner product, `A` the operator's own sparse matrix. Both are built by code that
        # shares nothing with `local_stencil`, which is what makes this an independent check
        # rather than a restatement: the two agree or one of them is wrong.
        n = ndofs(Wₕ)
        uu = parent(uₕ)
        Hh = Diagonal(collect(weights(Wₕ, Innerh())))
        Hpx = Diagonal(collect(weights(Wₕ, Innerplus(), 1)))
        Dx = Matrix(D₋ₓ(Wₕ))
        Mx = Matrix(Mₓ(Wₕ))
        Idm = Matrix(1.0I, n, n)

        @test assemble(form(Wₕ, v -> innerₕ(uₕ, v))) ≈ Hh * uu
        @test assemble(form(Wₕ, v -> innerₕ(uₕ, D₋ₓ(v)))) ≈ transpose(Dx) * (Hh * uu)
        @test assemble(form(Wₕ, v -> innerₕ(uₕ, Mₓ(v)))) ≈ transpose(Mx) * (Hh * uu)
        @test assemble(form(Wₕ, v -> inner₊ₓ(uₕ, D₋ₓ(v)))) ≈ transpose(Dx) * (Hpx * uu)

        # a linear combination of operators in the test argument is the same combination of
        # their matrices, and inner products of different kinds add
        @test assemble(form(Wₕ, v -> innerₕ(uₕ, v + 2 * D₋ₓ(v) - Mₓ(v)))) ≈
              transpose(Idm + 2 * Dx - Mx) * (Hh * uu)
        @test assemble(form(Wₕ, v -> innerₕ(uₕ, v) + inner₊ₓ(uₕ, D₋ₓ(v)))) ≈
              Hh * uu + transpose(Dx) * (Hpx * uu)

        if WITH_AD_TESTS
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
            offs_d = length(stencil_offsets(resolve_form_ast(form(Wₕ, v -> innerₕ(uₕ, v)))))
            offs_w = length(
                stencil_offsets(resolve_form_ast(form(Wₕ, v -> innerₕ(uₕ, D₋ₓ(v)))))
            )
            @test maximum(i -> count(!iszero, Jd[i, :]), 1:n) == offs_d
            @test maximum(i -> count(!iszero, Jw[i, :]), 1:n) == offs_w
        end
    end

    @testset "Dirichlet conditions" begin
        bcs = dirichlet_constraints(Ωₕ, :bottom => (x -> 5.0))
        marked = index_in_marker(Ωₕ, :bottom)
        @test any(marked)

        b = assemble(form(Wₕ, v -> innerₕ(uₕ, v)); dirichlet = bcs)
        @test all(b[marked] .≈ 5.0)

        # the same constraint, spelled as a Tuple of pairs rather than pre-built constraints
        b2 = assemble(form(Wₕ, v -> innerₕ(uₕ, v)); dirichlet = (:bottom => (x -> 5.0),))
        @test b2 ≈ b

        # no dirichlet at all is the unconstrained assembly
        plain = assemble(form(Wₕ, v -> innerₕ(uₕ, v)))
        @test assemble(form(Wₕ, v -> innerₕ(uₕ, v)); dirichlet = nothing) ≈ plain

        # naming a label alone, with no values, is a usage error rather than a silent
        # no-op: a linear form has nowhere to read boundary values from.
        @test_throws ArgumentError assemble(form(Wₕ, v -> innerₕ(uₕ, v)); dirichlet = :bottom)
        msg = try
            assemble(form(Wₕ, v -> innerₕ(uₕ, v)); dirichlet = :bottom)
        catch e
            sprint(showerror, e)
        end
        @test occursin("carries no values", msg)

        # and a value dirichlet does not know how to normalize is rejected before
        # anything is assembled
        @test_throws ArgumentError assemble(form(Wₕ, v -> innerₕ(uₕ, v)); dirichlet = 3)

        # `apply_dirichlet_conditions!` takes a lone label `Symbol` as well as a Tuple of
        # them, from callers that normalised the labels themselves
        lsym = form(Wₕ, v -> innerₕ(uₕ, v))
        b_sym = apply_dirichlet_conditions!(copy(plain), lsym, bcs, :bottom)
        @test b_sym ≈ b
        @test b_sym[.!marked] ≈ plain[.!marked]
    end

    @testset "dirichlet_components restriction" begin
        bcs = dirichlet_constraints(Ωₕ, :bottom => (x -> 5.0))
        marked = index_in_marker(Ωₕ, :bottom)
        uv = Rₕ(Vₕ, (x -> sin(x[1]), x -> cos(x[2])))
        l = form(Vₕ, v -> innerₕ(uv(1), v(1)) + innerₕ(uv(2), v(2)))

        b = assemble(l; dirichlet = bcs, dirichlet_components = 1)
        plain = assemble(l)

        leaf1, leaf2 = view(b, 1:n), view(b, (n + 1):(2n))
        plain2 = view(plain, (n + 1):(2n))

        @test all(leaf1[i] ≈ 5.0 for i in 1:n if marked[i])   # leaf 1 constrained
        @test leaf2 == plain2                                  # leaf 2 untouched
        @test any(i -> marked[i] && leaf1[i] != view(plain, 1:n)[i], 1:n) # leaf 1 changed

        # without dirichlet_components, the same labels bind to both leaves
        b_both = assemble(l; dirichlet = bcs)
        @test all(view(b_both, (n + 1):(2n))[i] ≈ 5.0 for i in 1:n if marked[i])
    end

    @testset "Allocation contract" begin
        # This is what a time loop calls every step. The assembly kernel itself is
        # allocation free -- `form.ast` is a field read, not a resolution, so there is
        # nothing left to hoist out of the call the way the deprecated `ast` keyword
        # claimed to: both `assemble!(b, lf)` and the equivalent with the keyword measure
        # 0 bytes.
        #
        # Measured inside a function on concrete locals: read from a non-const global, the
        # arguments box at the call boundary and the reading is of the box.
        function counts(N)
            Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (N, N), (true, true))
            W = gridspace(Ω)
            u = Rₕ(W, x -> x[1] + x[2])
            lf = form(W, v -> innerₕ(u, v))
            b = zeros(ndofs(W))

            assemble!(b, lf)                      # warm up
            return @allocated assemble!(b, lf)
        end

        for N in (8, 24)          # 9x the degrees of freedom apart
            @test counts(N) == 0
        end

        # the composite path too. It used to allocate a Vector of dof offsets on every
        # call (128 B, constant in the grid but paid every step of a time loop).
        function composite_bytes(N)
            Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (N, N), (true, true))
            V = gridspace(Ω, Val(3))
            uv = Rₕ(V, ntuple(_ -> (x -> x[1] + x[2]), 3))
            lf = form(V, v -> innerₕ(uv(1), v))
            b = zeros(ndofs(V))
            assemble!(b, lf)
            return @allocated assemble!(b, lf)
        end
        @test composite_bytes(8) == 0
        @test composite_bytes(16) == 0

        # a heterogeneous leaf costs the same zero bytes: `_scatter_term!` now
        # builds its own `LinearIndices`/`markers` from `mesh(sp)` per leaf instead of
        # receiving them from the caller, once per leaf per term (not once per grid
        # point), so it must not reopen the door to a per-point allocation the homogeneous
        # case above doesn't have.
        function heterogeneous_bytes(N)
            Ωbig = mesh(
                domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (N, N), (true, true)
            )
            Ωsmall = mesh(
                domain(interval(0.0, 1.0) × interval(0.0, 1.0)),
                (max(N ÷ 2, 3), max(N ÷ 2, 3)),
                (true, true)
            )
            V = Bramble.CompositeGridSpace((gridspace(Ωbig), gridspace(Ωsmall)))
            uv = Rₕ(V, (x -> x[1] + x[2], x -> x[1] - x[2]))
            lf = form(V, v -> innerₕ(uv(1), v(1)) + innerₕ(uv(2), v(2)))
            b = zeros(ndofs(V))
            assemble!(b, lf)
            return @allocated assemble!(b, lf)
        end
        het8, het16 = heterogeneous_bytes(8), heterogeneous_bytes(16)
        @test het8 == 0
        @test het16 == 0
        @test het8 == composite_bytes(8)         # literally the same as the homogeneous case
        @test het16 == composite_bytes(16)

        # and the parallel path's per-call cost doesn't grow just because the leaves
        # differ in size: same thread-set allocation either way, not asserted at zero
        # (that's tracked, not guaranteed, the same way the benchmark suite treats it)
        function parallel_bytes(space, u1, u2)
            lf = form(space, v -> innerₕ(u1(1), v(1)) + innerₕ(u2(2), v(2)))
            b = zeros(ndofs(space))
            assemble_parallel!(b, lf)               # warm up
            return @allocated assemble_parallel!(b, lf)
        end
        Ωhc = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 8), (true, true))
        Vhomo = gridspace(Ωhc, Val(2))
        uhomo = Rₕ(Vhomo, (x -> x[1] + x[2], x -> x[1] - x[2]))
        Ωhc_small = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (true, true)
        )
        Vhet_alloc = Bramble.CompositeGridSpace((gridspace(Ωhc), gridspace(Ωhc_small)))
        uhet_alloc = Rₕ(Vhet_alloc, (x -> x[1] + x[2], x -> x[1] - x[2]))

        # On Julia nightly specifically (never on a release build, across many
        # repeated runs), a single measurement drifts by one or two
        # 64-byte quanta, nondeterministically, in either direction: the thread-spawn
        # machinery's own scheduling noise, not anything the composite-space routing does
        # differently per leaf. Neither the minimum over repeats nor an exact equality
        # converged alone (it still flaked after 5 repeats), so the two are
        # combined: the minimum rejects one-off spikes, then a tolerance allows the
        # genuine floor-level noise that remains; the property under test is that the count is constant,
        # not bit-for-bit reproducible, which nightly's scheduler does not promise.
        min_parallel_bytes(space, u1, u2) = minimum(ntuple(_ -> parallel_bytes(space, u1, u2), 5))
        het = min_parallel_bytes(Vhet_alloc, uhet_alloc, uhet_alloc)
        homo = min_parallel_bytes(Vhomo, uhomo, uhomo)
        @test abs(het - homo) <= 256
    end

    @testset "Restricted refill allocation contract" begin
        # A `RegionRestriction`'s stencil is `()` or a full tuple depending on the point's
        # marker, and the scalar path evaluates the whole sum's stencil at every point. The
        # `map` that used to collect the summands' stencils built a tuple with a
        # non-concrete element there, boxed per point: 8736 B on the 2D grid, where the
        # unrestricted form allocated nothing. The refill must also match the two terms
        # assembled apart.
        function restricted_bytes(b, lf)
            assemble!(b, lf)
            assemble!(b, lf)
            return @allocated assemble!(b, lf)
        end
        Ω1d = mesh(domain(interval(0.0, 1.0)), 11, true)
        Ω2d = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 9), (false, true))
        for Ω in (Ω1d, Ω2d)
            W = gridspace(Ω)
            lf = form(W, v -> innerₕ(1.0, v) + Ref(2.0) * innerₕ(1.0, restrict_to(:boundary, v)))
            b = zeros(ndofs(W))
            @test restricted_bytes(b, lf) == 0
            parts = assemble(form(W, v -> innerₕ(1.0, v))) +
                    2 * assemble(form(W, v -> innerₕ(1.0, restrict_to(:boundary, v))))
            @test b ≈ parts
        end
    end

    WITH_AD_TESTS && @testset "Assembled residual differentiation" begin
        # The shape a nonlinear solve has: build the residual of a form, and let the solver
        # differentiate it with respect to the coefficient vector to get a Jacobian.
        #
        # This did not work. `assemble` allocated its output with `eltype(test_space(form))`
        # (the space's element type), so a Float64 space could only assemble a Float64
        # right-hand side, and writing a Dual weight into it met
        # `MethodError: no method matching Float64(::Dual)`. The type now comes from the
        # form's own weights, promoted against the space's, which is the rule `Rₕ` already
        # used and the same defect `dirichlet_constraints` had.
        u0 = parent(uₕ)

        @testset "Source parameter differentiation" begin
            J(a) = sum(assemble(form(Wₕ, v -> innerₕ(Rₕ(Wₕ, x -> a * (x[1] + x[2])), v))))
            h = 1e-6
            @test isapprox(
                ForwardDiff.derivative(J, 1.3), (J(1.3 + h) - J(1.3 - h)) / 2h; rtol = 1e-5
            )
        end

        @testset "Nonlinear residual Jacobian" begin
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

        @testset "Constrained Jacobian" begin
            bcs = dirichlet_constraints(Ωₕ, :bottom => (x -> 0.0))
            res(u) = assemble(form(Wₕ, v -> innerₕ(element(Wₕ, u .* u), v)); dirichlet = bcs)
            Jb = ForwardDiff.jacobian(res, u0)
            @test size(Jb) == (n, n)

            # a pinned value does not depend on the coefficients, so its row is zero:
            # which is what a solver needs in order not to move it
            marked = index_in_marker(Ωₕ, :bottom)
            @test any(marked)
            @test all(iszero, Jb[marked, :])
        end
    end

    @testset "Vector contraction" begin
        lf = form(Wₕ, v -> innerₕ(uₕ, v))
        b = assemble(lf)

        # A bare vector is refused rather than contracted. Its length carries no claim about
        # whether its blocks match the components a form routes to, so accepting one would
        # make a composite mismatch silent.
        @test_throws ArgumentError lf(parent(uₕ))
        @test_throws ArgumentError lf(fill(1.0, ndofs(Wₕ)))

        # on a composite space, where the components have to line up with the blocks rather
        # than merely add up to the right length
        Vc = gridspace(Ωₕ, Val(2))
        uc = Rₕ(Vc, (x -> 1.0, x -> 3.0))
        wc = Rₕ(Vc, (x -> 2.0, x -> 5.0))
        lfv = form(Vc, v -> innerₕ(uc, v))
        @test_throws ArgumentError lfv(parent(wc))

        # `evaluate!` agrees, and reuses its scratch rather than assembling afresh
        scratch = zeros(ndofs(Wₕ))
        @test evaluate!(scratch, lf, uₕ) ≈ lf(uₕ)
        @test scratch ≈ b                          # the scratch holds the assembled vector

        # repeatable, so the scratch is rewritten rather than accumulated into
        @test evaluate!(scratch, lf, uₕ) ≈ evaluate!(scratch, lf, uₕ)

        scratchv = zeros(ndofs(Vc))
        @test evaluate!(scratchv, lfv, wc) ≈ lfv(wc)
        @test_throws ArgumentError evaluate!(scratchv, lfv, parent(wc))

        # and it allocates nothing per call, once warmed.
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
        # fixes works. A plain warmup call does not absorb it (the warmup's return value is
        # unused, so the `dot` inside it can be optimised away), and neither does a
        # discarded `@allocated` written as a second statement, because that is a different
        # callsite and pays its own one-time 16 bytes. Repeating one callsite is what
        # reaches the steady state.
        #
        # It is `dot`, and it is upstream rather than anything here: a bare
        # `dot(::Vector{Float64}, ::Vector{Float64})` reproduces the identical (16, 0, 0)
        # with no Bramble in the picture, and `@allocated` over a non-allocating expression
        # is 0 on all three, so it is not the macro.
        #
        # What is under test is that a time loop calling this does not allocate, and the
        # steady state is exactly that property.
        function _evaluate_bytes(scratch, lf, v)
            evaluate!(scratch, lf, v)
            return minimum(ntuple(_ -> @allocated(evaluate!(scratch, lf, v)), 3))
        end
        @test _evaluate_bytes(scratch, lf, uₕ) == 0
    end

    @testset "Point-coloured sweep on a collapsed axis" begin
        # `_sweep_parallel!` bands the last axis whenever it can hold two slabs, which every
        # mesh above can, so its point-coloured fallback never ran. A collapsed last axis (one
        # point) cannot be banded, and the sweep walks colours instead: one flat colour for a
        # term reaching only its own point, `prod(strides)` strided colours for a difference.
        # The x axis is non-uniform. Two oracles: the Serial sweep, and the defining identity
        # bᵀw = l(w), evaluated through the space layer, which shares nothing with assembly.
        Ωc = mesh(domain(interval(0.0, 1.0) × interval(0.5, 0.5)), (9, 1), (false, true))
        Wc = gridspace(Ωc)
        @test size(Bramble.indices(Ωc)) == (9, 1)
        uc = Rₕ(Wc, x -> 1 + x[1]^2)
        wc = Rₕ(Wc, x -> sin(3x[1]) + 2)

        # Any: each closure is its own type
        for (nm, g, gw, ncolours) in Any[
            ("one colour", v -> innerₕ(uc, v), w -> innerₕ(uc, w), 1),
            (
                "strided colours", v -> innerₕ(uc, v + 2 * D₋ₓ(v) - Mₓ(v)),
                w -> innerₕ(uc, w + 2 * D₋ₓ(w) - Mₓ(w)), 2
            )
        ]
            lc = form(Wc, g)
            @test prod(_colour_strides(stencil_offsets(resolve_form_ast(lc)))) == ncolours
            bs = assemble(lc)
            reference = gw(wc)
            @test dot(bs, parent(wc)) ≈ reference
            @test !iszero(reference)

            bp = fill(7.0, ndofs(Wc))
            @test assemble_parallel!(bp, lc) === bp
            @test bp ≈ bs

            # inside a threaded region a `:static` loop cannot start, so each colour runs
            # in order on the calling task (`_serial_linear_colour!`), to the same answer
            bt = fill(7.0, ndofs(Wc))
            Threads.@threads for _ in 1:1
                assemble_parallel!(bt, lc)
            end
            @test bt ≈ bs
        end

        # the composite sweep hands each leaf to the same fallback at its offset; values
        # 1 and 100 apart so a block landing in the wrong place cannot pass
        Vc2 = gridspace(Ωc, Val(2))
        uv = Rₕ(Vc2, (x -> 1 + x[1]^2, x -> 100 * (2 - x[1])))
        lv = form(Vc2, v -> innerₕ(uv(1), v(1) + 2 * D₋ₓ(v(1))) + innerₕ(uv(2), v(2)))
        bvs = assemble(lv)
        bvp = similar(bvs)
        assemble_parallel!(bvp, lv)
        m = ndofs(Wc)
        @test bvp[1:m] ≈ bvs[1:m]
        @test bvp[(m + 1):(2m)] ≈ bvs[(m + 1):(2m)]
        @test bvs[(m + 1):(2m)] ≈ parent(components(uv)[2]) .* weights(Wc, Innerh())
    end

    @testset "CpuPolyester colour hooks" begin
        # `CpuPolyester` sweeps reach the `_batch_*` hooks, never `Threads.@threads`. Without
        # `BramblePolyesterExt` the hooks raise naming Polyester; with it they fill `b`. A
        # one-colour form, so the whole grid is one race-free colour.
        Ωp = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 9), (false, true))
        Wp = gridspace(Ωp)
        up = Rₕ(Wp, x -> 1 + x[1] * x[2])
        lp = form(Wp, v -> innerₕ(up, v))
        astp = resolve_form_ast(lp)
        idxs = Bramble.indices(Ωp)
        lin = LinearIndices(idxs)
        mk = Bramble.markers(Ωp)
        bpol = zeros(ndofs(Wp))
        if Base.get_extension(Bramble, :BramblePolyesterExt) === nothing
            err = try
                Bramble._sweep_colour!(
                    Bramble.CpuPolyester(), bpol, Wp, astp, idxs, lin, mk, 0, true)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("Polyester", err.msg)
            @test all(iszero, bpol)
        else
            Bramble._sweep_colour!(
                Bramble.CpuPolyester(), bpol, Wp, astp, idxs, lin, mk, 0, true)
            @test bpol ≈ assemble(lp)
        end

        # Untyped arguments reach the `src/` stubs even when the extension is loaded.
        for (hook, nargs) in ((Bramble._batch_linear_colour_sweep!, 8),
            (Bramble._batch_linear_band_sweep!, 11))
            err = try
                hook(ntuple(_ -> nothing, nargs)...)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("Polyester", err.msg)
        end
    end

    @testset "Shifted source lowering" begin
        # `form` samples a `SourceFunction` once, through a `ShiftNode` too: the stored AST
        # holds a `SourceVector` under the shift, and the assembled entry at a point is the
        # source one point up in y times the point's weight (zero past the top row).
        # Non-uniform in both axes, so a shift read off the wrong axis would differ.
        Ωs = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (false, false))
        Ws = gridspace(Ωs)
        f = x -> 1 + x[1] + 3x[2]^2
        sf = Bramble.source_function(f, Val(2))
        ls = form(Ws, v -> innerₕ(Bramble.shift_op(sf, 2, 1), v))
        asts = resolve_form_ast(ls)
        @test asts isa LinearProduct
        @test asts.left_op isa Bramble.ShiftNode
        @test asts.left_op.inner_op isa Bramble.SourceVector

        F = reshape(parent(Rₕ(Ws, f)), 7, 6)
        w = reshape(collect(weights(Ws, Innerh())), 7, 6)
        expected = [j < 6 ? F[i, j + 1] * w[i, j] : 0.0 for i in 1:7, j in 1:6]
        @test assemble(ls) ≈ vec(expected)
    end

    @testset "Unnamed source, heterogeneous composite" begin
        # A term naming no component is not lowered on a composite space (its leaves may
        # have different meshes), so `assemble` sizes its vector by sampling the function
        # on each leaf. Non-uniform leaves of different sizes; the routed second term adds
        # 100 to the second block only.
        Ωa = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (7, 9), (false, false))
        Ωb = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (5, 4), (false, true))
        Wa, Wb = gridspace(Ωa), gridspace(Ωb)
        Vh = Bramble.CompositeGridSpace((Wa, Wb))
        f = x -> x[1] + 2x[2]
        lh = form(Vh, v -> innerₕ(f, v) + innerₕ(100.0, v(2)))
        bh = assemble(lh)
        na = ndofs(Wa)
        @test eltype(bh) === Float64
        @test bh[1:na] ≈ parent(Rₕ(Wa, f)) .* weights(Wa, Innerh())
        @test bh[(na + 1):end] ≈ (parent(Rₕ(Wb, f)) .+ 100.0) .* weights(Wb, Innerh())
        # the trapezoidal sums are exact for a bilinear integrand: ∫(x + 2y) = 1.5
        @test sum(bh[1:na]) ≈ 1.5
        @test sum(bh[(na + 1):end]) ≈ 101.5
    end

    @testset "Point sources with Ref strengths" begin
        # A `Vector` of `Ref` strengths reads its type off the element type, and stays live:
        # multilinear spreading preserves each point's total, so the vector sums to the sum
        # of the strengths, before and after one is changed. Non-uniform mesh, off-grid points.
        # The interior points are random, so the mesh is seeded: an unseeded draw can put both
        # points in one cell, or in cells sharing corners, and the 8-entry count fails.
        Bramble._seed_mesh1d_rng!(1)
        Ωd = try
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 7), (false, false))
        finally
            Bramble._unseed_mesh1d_rng!()
        end
        Wd = gridspace(Ωd)
        strengths = [Ref(2.0), Ref(3.0)]
        ld = form(Wd, v -> innerₕ(dirac([(0.31, 0.42), (0.77, 0.18)], strengths), v))
        bd = assemble(ld)
        @test eltype(bd) === Float64
        @test sum(bd) ≈ 5.0
        @test count(!iszero, bd) == 8         # 4 corners per point, no overlap
        strengths[1][] = 10.0
        assemble!(bd, ld)
        @test sum(bd) ≈ 13.0
    end

    @testset "Element type helpers" begin
        # a thunk reveals its type through its value, and anything that is not a number,
        # `Ref`, array or thunk promotes by its own type
        @test Bramble._value_eltype(Float32, () -> 2.0) === Float64
        @test Bramble._value_eltype(Float64, () -> Float32[1, 2]) === Float64
        @test Bramble._value_eltype(Float32, nothing) === Union{Nothing, Float32}

        # the diagonal blocks of a test side shorter than its trial side are the test
        # side's own leaves
        @test Bramble._paired((1, 2), (:a, :b, :c)) == (1, 2)
        @test Bramble._paired((), (:a,)) == ()
        V2, V3 = gridspace(Ωₕ, Val(2)), gridspace(Ωₕ, Val(3))
        @test Bramble._assembled_eltype(
            resolve_form_ast(form(V2, v -> innerₕ(1.0f0, v))), V2, V3) === Float64
    end

    @testset "Host mirror helpers" begin
        # The device paths assemble on a host mirror of the test space, into a host buffer,
        # and copy it into the storage of `b`. On a host space each helper is the identity
        # or an equal copy, which is what lets the mirror assemble the same vector.
        Ωm = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 8), (false, true))
        Wm = gridspace(Ωm)
        Vm = Bramble.CompositeGridSpace((Wm, gridspace(Ωₕ)))
        um = Rₕ(Wm, x -> 1 + x[1] - x[2]^2)
        # Any: each space and closure is its own type
        for (sp, g) in Any[(Wm, v -> innerₕ(um, v + D₋ₓ(v))),
            (Vm, v -> innerₕ(um, v(1) + D₋ₓ(v(1))) + innerₕ(100.0, v(2)))]
            lm = form(sp, g)
            hs = Bramble._host_mirror_space(sp)
            @test ndofs(hs) == ndofs(sp)
            astm = resolve_form_ast(lm)
            hform = LinearForm{2, typeof(hs), typeof(astm)}(hs, astm)
            hb = zeros(ndofs(sp))
            Bramble._assemble_linear!(
                Bramble.HostLocality(), hb, hform, astm, nothing, nothing)
            @test hb ≈ assemble(lm)
        end

        v_plain = zeros(ndofs(Wm))
        @test Bramble._vector_storage(v_plain) === v_plain
        @test Bramble._vector_storage(um) === parent(um)
        @test Bramble._host_array(2.5) === 2.5

        # every `@node_family` node is rebuilt around its walked operand, here the test
        # function, which the walk returns untouched
        v = TestFunction{2}()
        @test Bramble._host_sources(v) === v
        for N in (:BackwardDifference, :ForwardDifference, :CenteredDifference,
            :StarDifference, :CrossWeightedDifference, :BackwardAverage, :ForwardAverage,
            :CenteredAverage, :JumpNode)
            node = getfield(Bramble, N){2, 1, typeof(v)}(v)
            @test Bramble._host_sources(node) === node
        end
    end
end

end # module FormLinearTests
