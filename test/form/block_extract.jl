using Test
using Bramble
using Bramble: IndexedTrialFunction, IndexedTestFunction, TrialFunction, TestFunction,
               trial_component_or_nothing, test_component_or_nothing, block_of,
               routes_by_component, restrict_to

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

    @testset "Unindexed terms" begin
        # Not an error: a form written without component indices is the same integrand on
        # every block, which is how the two spellings mix in one form.
        u, v = TrialFunction{2}(), TestFunction{2}()

        @test trial_component_or_nothing(innerₕ(u, v)) === nothing
        @test test_component_or_nothing(innerₕ(u, v)) === nothing
        @test trial_component_or_nothing(D₋ₓ(u)) === nothing
        @test !routes_by_component(innerₕ(u, v))
        @test routes_by_component(innerₕ(u(1), v(1)))
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
