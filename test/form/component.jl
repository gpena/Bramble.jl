module FormComponentTests

using Test
using Bramble
using Bramble:
               IndexedTrialFunction,
               IndexedTestFunction,
               TrialFunction,
               TestFunction,
               IdentityOperator,
               ZeroOperator,
               GridFunctionScale,
               shift_op,
               source_function,
               trial_component_or_nothing,
               test_component_or_nothing,
               components,
               restrict_to
using ..TestUtils: alloc_test, @test_allocs

# `component(op, i)` (the mechanism behind `u(1)`, `D₋ₓ(v)(2)`, and the composite
# `innerₕ(uₕ, r)` shorthand alike) rebuilds a symbolic tree with its trial/test leaves
# replaced by their indexed forms. `test/form/linear.jl`'s "Shorthand equivalence" and
# "Source variants" testsets already exercise most of this file's methods incidentally,
# through `innerₕ`/`inner₊ₓ` and a real composite form; this file covers what those do
# not: the individual `component` methods directly (so a routing bug shows up here rather
# than only in a plausible-looking wrong sum), and the composite shorthand for `inner₊`,
# `inner₊ᵧ` and `inner₊₂`, which nothing else in the suite reaches.

@testset "Component indexing" begin
    @testset "Leaves and pass-through nodes" begin
        u, v = TrialFunction{2}(), TestFunction{2}()
        @test u(3) isa IndexedTrialFunction{2}
        @test v(2) isa IndexedTestFunction{2}

        # re-indexing replaces the index rather than erroring or no-oping
        @test test_component_or_nothing(v(2)(7)) == 7
        @test trial_component_or_nothing(u(3)(5)) == 5

        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (3, 3), (true, true))
        Wₕ = gridspace(Ωₕ)

        # a source, the identity and the zero operator have no component to name, so
        # indexing is the identity on the node itself
        idop, zop = IdentityOperator(Wₕ), ZeroOperator(Wₕ)
        @test idop(4) === idop
        @test zop(4) === zop

        sf = source_function(x -> x[1] + x[2], Val(2))
        @test sf(4) === sf
    end

    @testset "Operators rebuild around the indexed leaf" begin
        v = TestFunction{2}()

        # the whole directional family generated in difference.jl/average.jl/jump.jl:
        # every one rebuilds around the indexed leaf rather than being left behind
        for op in (D₋ₓ, D₊ₓ, Dcₓ, D̽ₓ, Dₕₓ, jumpₓ, Mₓ, M₊ₓ)
            @test test_component_or_nothing(op(v)(3)) == 3
        end

        # a shift, a region restriction, a scalar scale, a grid-function scale and a sum
        # all pass the index through to their own inner operand
        @test test_component_or_nothing(shift_op(v, 1, 2)(3)) == 3
        @test test_component_or_nothing(restrict_to(:boundary, v)(3)) == 3
        @test test_component_or_nothing((7 * D₋ₓ(v))(3)) == 3
        @test test_component_or_nothing((v + D₋ₓ(v))(3)) == 3

        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (true, true))
        Wₕ = gridspace(Ωₕ)
        wₕ = Rₕ(Wₕ, x -> x[1] + 1)
        @test test_component_or_nothing((wₕ * D₋ₓ(v))(3)) == 3
        @test (wₕ * D₋ₓ(v))(3) isa GridFunctionScale
    end

    @testset "Composite shorthand: inner₊, inner₊ᵧ" begin
        # Same style as linear.jl's "Shorthand equivalence" (`innerₕ`, `inner₊ₓ`), extended
        # to the two directional-family members nothing else in the suite reaches.
        Ωf = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 5), (true, true))
        Vf = gridspace(Ωf, Val(3))

        # distinguishable per component (1x, 100x, x²) so a routing bug has teeth
        uv = Rₕ(Vf, (x -> x[1], x -> 100 * x[1], x -> x[2]^2))

        for (short, long) in (
            (
            v -> inner₊(D₋ₓ(uv), D₋ₓ(v)),
            v -> inner₊(D₋ₓ(uv(1)), D₋ₓ(v(1))) +
                 inner₊(D₋ₓ(uv(2)), D₋ₓ(v(2))) +
                 inner₊(D₋ₓ(uv(3)), D₋ₓ(v(3)))
        ),
            (
            v -> inner₊ᵧ(uv, v + Mᵧ(v)),
            v -> inner₊ᵧ(uv(1), v(1) + Mᵧ(v(1))) +
                 inner₊ᵧ(uv(2), v(2) + Mᵧ(v(2))) +
                 inner₊ᵧ(uv(3), v(3) + Mᵧ(v(3)))
        )
        )
            b = assemble(form(Vf, short))
            reference = assemble(form(Vf, long))
            @test b ≈ reference
            @test !all(iszero, b)
        end
    end

    @testset "Composite shorthand: inner₊₂" begin
        # inner₊₂ needs a spatial dimension of at least 3, unlike the 2D composite spaces
        # used above; here we use a separate small 3D setup.
        Ω3 = mesh(
            domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (3, 4, 3), (true, true, true)
        )
        V3 = gridspace(Ω3, Val(2))
        uv = Rₕ(V3, (x -> x[1] + x[2], x -> 100 + x[3]))

        b = assemble(form(V3, v -> inner₊₂(uv, v)))
        reference = assemble(form(V3, v -> inner₊₂(uv(1), v(1)) + inner₊₂(uv(2), v(2))))
        @test b ≈ reference
        @test !all(iszero, b)
    end

    @testset "Composite shorthand, tuple source" begin
        # `innerₕ((f, g), v)`'s tuple form is already checked in linear.jl; the sibling
        # `inner₊`/`inner₊ₓ`/`inner₊ᵧ`/`inner₊₂` tuple forms are not, anywhere.
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 6), (true, true))
        Vₕ = gridspace(Ωₕ, Val(2))

        for f in (inner₊, inner₊ₓ, inner₊ᵧ)
            b = assemble(form(Vₕ, v -> f((1.0, 2.0), v)))
            reference = assemble(form(Vₕ, v -> f(1.0, v(1)) + f(2.0, v(2))))
            @test b ≈ reference
            @test !all(iszero, b)

            @test_throws ArgumentError form(Vₕ, v -> f((), v))
        end
    end

    @testset "Bracket indexing and destructuring (gpena/Bramble.jl#153)" begin
        p = TrialFunction{2, 2}()
        q = TestFunction{2, 2}()

        # 1. Bracket indexing equivalence
        @test p[1] isa IndexedTrialFunction{2}
        @test p[2] isa IndexedTrialFunction{2}
        @test p[1] == p(1)
        @test p[2] == p(2)
        @test q[1] isa IndexedTestFunction{2}
        @test q[2] isa IndexedTestFunction{2}
        @test q[1] == q(1)
        @test q[2] == q(2)

        # Indexing distributes through operators
        @test test_component_or_nothing((D₋ₓ(q))[1]) == 1
        @test test_component_or_nothing((D₋ₓ(q))[2]) == 2
        @test test_component_or_nothing((q + D₋ₓ(q))[2]) == 2

        # 2. Out-of-range checking at AST construction time
        @test_throws ArgumentError p[3]
        @test_throws ArgumentError p(3)
        @test_throws ArgumentError q[3]
        @test_throws ArgumentError q(3)
        @test_throws ArgumentError p[0]
        @test_throws ArgumentError p(-1)
        @test_throws ArgumentError (D₋ₓ(p))[3]

        # 3. components helper
        comps_p = components(p)
        @test comps_p == (p[1], p[2])
        comps_q = components(q)
        @test comps_q == (q[1], q[2])

        # Tuple destructuring
        (u, v) = components(p)
        @test u == p[1] && v == p[2]
        (w, z) = components(q)
        @test w == q[1] && z == q[2]

        # Direct destructuring and iteration
        (u_dir, v_dir) = p
        @test u_dir == p[1] && v_dir == p[2]
        (w_dir, z_dir) = q
        @test w_dir == q[1] && z_dir == q[2]
        @test length(p) == 2 && firstindex(p) == 1 && lastindex(p) == 2
        @test collect(p) == [p[1], p[2]]

        # Unparameterized TrialFunction/TestFunction components helper
        p_unparam = TrialFunction{2}()
        @test_throws ArgumentError components(p_unparam)
        @test components(p_unparam, 2) == (p_unparam[1], p_unparam[2])

        # 4. Form assembly equivalence (Bilinear & Linear)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 5), (true, true))
        Vₕ = gridspace(Ωₕ, Val(2))

        # Bilinear form
        a_functor = form(Vₕ, Vₕ, (p, q) -> inner₊ₓ(D₋ₓ(p(1)), D₋ₓ(q(1))) + innerₕ(p(2), q(1)) +
                                           inner₊ᵧ(D₋ᵧ(p(2)), D₋ᵧ(q(2))))
        a_bracket = form(Vₕ, Vₕ, (p, q) -> inner₊ₓ(D₋ₓ(p[1]), D₋ₓ(q[1])) + innerₕ(p[2], q[1]) +
                                           inner₊ᵧ(D₋ᵧ(p[2]), D₋ᵧ(q[2])))
        a_components = form(Vₕ, Vₕ, (p, q) -> begin
            u, v = components(p)
            w, z = components(q)
            inner₊ₓ(D₋ₓ(u), D₋ₓ(w)) + innerₕ(v, w) + inner₊ᵧ(D₋ᵧ(v), D₋ᵧ(z))
        end)
        a_direct = form(Vₕ, Vₕ, (p, q) -> begin
            u, v = p
            w, z = q
            inner₊ₓ(D₋ₓ(u), D₋ₓ(w)) + innerₕ(v, w) + inner₊ᵧ(D₋ᵧ(v), D₋ᵧ(z))
        end)

        A_ref = assemble(a_functor)
        @test assemble(a_bracket) == A_ref
        @test assemble(a_components) == A_ref
        @test assemble(a_direct) == A_ref

        # Linear form
        fₕ = Rₕ(Vₕ, (x -> x[1], x -> x[2]))
        l_functor = form(Vₕ, q -> innerₕ(fₕ(1), q(1)) + innerₕ(fₕ(2), q(2)))
        l_bracket = form(Vₕ, q -> innerₕ(fₕ(1), q[1]) + innerₕ(fₕ(2), q[2]))
        l_components = form(Vₕ, q -> begin
            w, z = components(q)
            innerₕ(fₕ(1), w) + innerₕ(fₕ(2), z)
        end)
        l_direct = form(Vₕ, q -> begin
            w, z = q
            innerₕ(fₕ(1), w) + innerₕ(fₕ(2), z)
        end)

        b_ref = assemble(l_functor)
        @test assemble(l_bracket) == b_ref
        @test assemble(l_components) == b_ref
        @test assemble(l_direct) == b_ref

        # Out-of-range checking during form construction
        @test_throws ArgumentError form(Vₕ, Vₕ, (p, q) -> innerₕ(p[3], q[1]))
        @test_throws ArgumentError form(Vₕ, Vₕ, (p, q) -> innerₕ(p[1], q[3]))
        @test_throws ArgumentError form(Vₕ, q -> innerₕ(fₕ(1), q[3]))

        # Zero allocations
        @test_allocs components(p)
        @test_allocs components(q)
        @test_allocs p[1]
        @test_allocs q[2]
    end
end

end # module FormComponentTests
