using Test
using InteractiveUtils: subtypes
using Bramble
using Bramble:
    IndexedTrialFunction,
    IndexedTestFunction,
    TrialFunction,
    TestFunction,
    trial_component_or_nothing,
    test_component_or_nothing,
    block_of,
    restrict_to

# Reading which block of a coupled form a term belongs to.
#
# A term of a bilinear form belongs to one block, and the block is a component from each
# side: the row comes from the test function and the column from the trial one. These two
# walks are how that is read off the tree, and `block_of` turns the pair into a block or
# into an error.

@testset "Block extraction" begin
    @testset "Component search" begin
        # The walk descends to the leaf, so every node that wraps one has to pass the
        # question through. A node missed here reports `nothing`, and a term reporting
        # `nothing` goes to every diagonal block rather than the one it names, which is
        # silent, and produced a wrong answer that summed to something plausible.
        u, v = IndexedTrialFunction{2}(3), IndexedTestFunction{2}(2)

        for D in (D₋ₓ, D₊ₓ, D₋ᵧ, D₊ᵧ, M₋ₓ, M₊ₓ, jumpₓ, Dcₓ, Dstar₊ₓ, Dₕₓ)
            @test trial_component_or_nothing(D(u)) == 3
            @test test_component_or_nothing(D(v)) == 2
        end

        # scaling by a number and by a grid function, and restriction, all pass through
        @test trial_component_or_nothing(7 * D₋ₓ(u)) == 3
        @test test_component_or_nothing(7 * D₋ₓ(v)) == 2
        @test trial_component_or_nothing(restrict_to(:interior, D₋ₓ(u))) == 3
        @test test_component_or_nothing(restrict_to(:interior, D₋ₓ(v))) == 2

        # and through a product, from the side that owns it
        @test trial_component_or_nothing(innerₕ(u, v)) == 3
        @test test_component_or_nothing(innerₕ(u, v)) == 2
    end

    @testset "Each collapsed ladder answers through its operand (#52)" begin
        # Seven queries used to be registered against all thirteen wrapper types by hand;
        # each is now one method on `UnaryWrapper`. One assertion per query, through a
        # wrapper, so a collapsed method that stopped recursing would fail here rather
        # than inherit a fallback that looks like an answer.
        u, v = TrialFunction{2}(), TestFunction{2}()
        iu, iv = IndexedTrialFunction{2}(1), IndexedTestFunction{2}(2)
        sf = Bramble.SourceFunction{2,typeof(sin)}(sin)

        # every member of the union, wrapped once, for the two component queries
        for wrap in (D₋ₓ, D₊ₓ, Dcₓ, Dstar₊ₓ, Dₕₓ, jumpₓ, M₋ₓ, M₊ₓ)
            @test test_component_or_nothing(wrap(iv)) == 2
            @test trial_component_or_nothing(wrap(iu)) == 1
            @test test_component_or_nothing(wrap(v)) === nothing
        end
        # the non-difference wrappers too: scale, grid-function scale, restriction
        @test test_component_or_nothing(2.0 * iv) == 2
        @test trial_component_or_nothing(2.0 * iu) == 1
        @test test_component_or_nothing(Bramble.restrict_to(:left, iv)) == 2

        # _is_source_only: a wrapped source stays a source, a wrapped trial never is
        @test Bramble._is_source_only(D₋ₓ(sf))
        @test Bramble._is_source_only(M₊ᵧ(D₋ₓ(sf)))
        @test Bramble._is_source_only(2.0 * jumpₓ(sf))
        @test !Bramble._is_source_only(D₋ₓ(u))

        # stencil_shift_trait: this one overrides an *abstract* fallback that answers
        # translation-invariant, so a wrapper that stopped recursing would relabel a
        # source's stencil instead of re-evaluating it -- a wrong matrix, not an error
        @test Bramble.stencil_shift_trait(sf) isa Bramble.PointDependentStencil
        @test Bramble.stencil_shift_trait(D₋ₓ(sf)) isa Bramble.PointDependentStencil
        @test Bramble.stencil_shift_trait(M₊ᵧ(D₋ₓ(sf))) isa Bramble.PointDependentStencil
        @test Bramble.stencil_shift_trait(D₋ₓ(u)) isa Bramble.TranslationInvariantStencil

        # is_symbolic, through several wrappers at once
        @test Bramble.is_symbolic(D₋ₓ(M₊ᵧ(2.0 * iv)))
        @test Bramble.is_symbolic(jumpₓ(sf))

        # _all_trial_interpolated: also an abstract fallback (false), so the same risk
        W = gridspace(
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 5), (true, true))
        )
        πu = πₕ(W, u)
        @test Bramble._all_trial_interpolated(πu)
        @test Bramble._all_trial_interpolated(D₋ₓ(πu))
        @test Bramble._all_trial_interpolated(2.0 * M₊ᵧ(πu))
        @test !Bramble._all_trial_interpolated(D₋ₓ(u))
    end

    @testset "Every single-operand node is in UnaryWrapper (#52)" begin
        # The queries that answer for a wrapper whatever they answer for its operand are
        # now one method each, dispatched on `UnaryWrapper`. That only stays correct while
        # the union lists every node with a single operand: a new wrapper left out of it
        # would not fail to compile, it would inherit a fallback, and two of those
        # fallbacks are wrong answers rather than missing ones
        # (`stencil_shift_trait(::LazyOp)` says translation-invariant,
        # `_all_trial_interpolated(::LazyOp)` says false).
        #
        # So the membership is asserted here rather than maintained by hand. Before the
        # collapse this test could not have been written: there was nothing to compare a
        # node against, only thirteen separate registrations to remember.
        concrete(T) =
            isabstracttype(T) ? reduce(vcat, concrete.(subtypes(T)); init=Type[]) : [T]
        nodes = concrete(Bramble.LazyOp)
        @test length(nodes) >= 25

        single_operand = filter(nodes) do T
            kids = filter(in((:inner_op, :left_op, :right_op)), fieldnames(T))
            kids == (:inner_op,)
        end
        @test !isempty(single_operand)

        missing_from_union = filter(T -> !(T <: Bramble.UnaryWrapper), single_operand)
        @test isempty(missing_from_union)

        # and nothing with two operands sneaked in: a product's sides have different roles,
        # so a query about the test component reads `right_op` alone
        two_operand = filter(nodes) do T
            kids = filter(in((:inner_op, :left_op, :right_op)), fieldnames(T))
            kids == (:left_op, :right_op)
        end
        @test !isempty(two_operand)
        @test all(T -> !(T <: Bramble.UnaryWrapper), two_operand)
    end

    @testset "_collect_region_labels through wrappers and products (#106)" begin
        # The twelve `@eval`-generated methods collapsed onto `UnaryWrapper` (with
        # `RegionRestriction`'s own method still winning on specificity); this checks the
        # collapse changed no answer: a restriction nested behind several wrappers, and one
        # on each side of a product, must still be found.
        u, v = TrialFunction{2}(), TestFunction{2}()

        nested = M₊ᵧ(D₋ₓ(restrict_to(:bottom, u)))
        @test Bramble._collect_region_labels(nested) == (:bottom,)

        prod = innerₕ(restrict_to(:left, u), restrict_to(:right, v))
        @test Set(Bramble._collect_region_labels(prod)) == Set((:left, :right))

        # through an InterpolationNode too: its own override was redundant with the
        # UnaryWrapper method and is gone, so this must still recurse
        W = gridspace(
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 5), (true, true))
        )
        @test Bramble._collect_region_labels(restrict_to(:top, πₕ(W, u))) == (:top,)
    end

    @testset "Unindexed terms" begin
        # Not an error: a form written without component indices is the same integrand on
        # every block, which is how the two spellings mix in one form.
        u, v = TrialFunction{2}(), TestFunction{2}()

        @test trial_component_or_nothing(innerₕ(u, v)) === nothing
        @test test_component_or_nothing(innerₕ(u, v)) === nothing
        @test trial_component_or_nothing(D₋ₓ(u)) === nothing
    end

    @testset "Sum consistency" begin
        # `innerₕ(uₕ, v(2) + D₋ₓ(v(2)))` is one term of one block, so its sides have to
        # name the same component. Sides naming different ones are not a component of
        # anything, and answering `nothing` there is what let a term broadcast to every
        # block instead of the one it belonged to.
        u, v = TrialFunction{2}(), TestFunction{2}()

        @test test_component_or_nothing(v(2) + D₋ₓ(v(2))) == 2
        @test test_component_or_nothing(v + D₋ₓ(v)) === nothing
        @test trial_component_or_nothing(u(1) + M₋ₓ(u(1))) == 1

        @test_throws ArgumentError test_component_or_nothing(v(1) + v(2))
        @test_throws ArgumentError trial_component_or_nothing(u(1) + u(3))
    end

    @testset "Block routing" begin
        u, v = TrialFunction{2}(), TestFunction{2}()

        # both sides named: one block, column from the trial, row from the test
        @test block_of(innerₕ(u(1), v(2)), 3, 3) == (1, 2)
        @test block_of(innerₕ(u(3), v(1)), 3, 3) == (3, 1)

        # neither named: every diagonal block, since Σᵢ innerₕ(uᵢ, vᵢ) is block diagonal
        @test block_of(innerₕ(u, v), 3, 3) === nothing

        # one named and not the other is refused rather than guessed at: it is not
        # something written in a variational formulation
        @test_throws ArgumentError block_of(innerₕ(u(1), v), 3, 3)
        @test_throws ArgumentError block_of(innerₕ(u, v(2)), 3, 3)

        # and a component the system does not have is an error, not an empty block
        @test_throws ArgumentError block_of(innerₕ(u(4), v(1)), 3, 3)
        @test_throws ArgumentError block_of(innerₕ(u(1), v(4)), 3, 3)
        @test_throws ArgumentError block_of(innerₕ(u(0), v(1)), 3, 3)

        # the two sides are counted separately, so a rectangular system works
        @test block_of(innerₕ(u(2), v(1)), 2, 1) == (2, 1)
        @test_throws ArgumentError block_of(innerₕ(u(2), v(1)), 1, 2)
    end
end
