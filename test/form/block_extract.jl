using Test
using Bramble
using Bramble: CompositeGridSpace, IndexedTrialFunction, IndexedTestFunction,
               TrialFunction, TestFunction, LazyOp, OperatorAdd,
               collect_leaf_spaces_offsets, is_hierarchical,
               make_trial_args, make_test_args, flatten_sum,
               find_trial_component, find_test_component, extract_block_asts,
               restrict_to

# Decomposing a coupled form into its blocks.
#
# A coupled system over `X = V × W` is one AST, but it assembles as a matrix of blocks:
# entry `[i, j]` is the part of the form pairing test component `i` with trial component
# `j`. This file is what performs that split, and until now none of it had ever run — nine
# tracked lines out of 168, with `extract_block_asts` never called.
#
# The decomposition works entirely off the indexed leaves. `make_trial_args` hands out
# `IndexedTrialFunction`s numbered depth first across the space's leaves; every term then
# carries the two indices that place it, and `extract_block_asts` reads them back.

@testset "Coupled form block extraction" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
        (5, 5), (true, true))
    Wₕ = gridspace(Ωₕ)
    Vₕ = gridspace(Ωₕ, Val(2))
    Xₕ = CompositeGridSpace((Vₕ, Wₕ))       # Stokes shaped: a velocity pair and a pressure

    @testset "recognising a hierarchy" begin
        @test !is_hierarchical(Wₕ)          # a scalar space is not composite at all
        @test !is_hierarchical(Vₕ)          # flat: every component is scalar
        @test is_hierarchical(Xₕ)           # one component is itself composite
        @test is_hierarchical(CompositeGridSpace((Wₕ, Xₕ)))
    end

    @testset "the symbolic arguments mirror the space" begin
        U, P = make_trial_args(Xₕ, 2)
        # the shape follows the top-level components: a pair, then a scalar
        @test U isa NTuple{2, IndexedTrialFunction{2}}
        @test P isa IndexedTrialFunction{2}

        # numbered depth first, left to right, across all the leaves
        @test U[1].component_idx == 1
        @test U[2].component_idx == 2
        @test P.component_idx == 3

        V, Q = make_test_args(Xₕ, 2)
        @test V isa NTuple{2, IndexedTestFunction{2}}
        @test Q isa IndexedTestFunction{2}
        @test (V[1].component_idx, V[2].component_idx, Q.component_idx) == (1, 2, 3)

        # one index per leaf space, and the count agrees with the traversal
        @test length(collect_leaf_spaces_offsets(Xₕ)) == 3
        @test Bramble.n_leaf_spaces(Xₕ) == 3

        # a flat space gives a flat tuple
        @test make_trial_args(Vₕ, 2) isa NTuple{2, IndexedTrialFunction{2}}

        # and the dimension is carried through
        @test make_trial_args(Xₕ, 3)[2] isa IndexedTrialFunction{3}
    end

    @testset "flattening a sum" begin
        u, v = IndexedTrialFunction{2}(1), IndexedTestFunction{2}(1)
        t1 = innerₕ(u, v)
        t2 = innerₕ(D₋ₓ(u), D₋ₓ(v))
        t3 = innerₕ(M₋ₓ(u), M₋ₓ(v))

        @test length(flatten_sum(t1)) == 1        # a lone term is a one-element list
        @test first(flatten_sum(t1)) === t1

        # nested however the sum was associated, the addends come back in order
        @test length(flatten_sum(t1 + t2 + t3)) == 3
        @test length(flatten_sum((t1 + t2) + t3)) == 3
        @test length(flatten_sum(t1 + (t2 + t3))) == 3
        @test flatten_sum(t1 + (t2 + t3)) == Any[t1, t2, t3]
    end

    @testset "finding the component through every node it can hide behind" begin
        u, v = IndexedTrialFunction{2}(2), IndexedTestFunction{2}(3)
        uₕ = Rₕ(Wₕ, x -> x[1])

        for wrap in (identity, D₋ₓ, D₊ₓ, M₋ₓ, M₊ₓ,
            op -> 7 * op,                       # OperatorScale
            op -> uₕ * op,                      # GridFunctionScale
            op -> restrict_to(:interior, op))
            @test find_trial_component(wrap(u)) == 2
            @test find_test_component(wrap(v)) == 3
        end

        # and through the product itself: trial on the left, test on the right
        @test find_trial_component(innerₕ(D₋ₓ(u), M₋ₓ(v))) == 2
        @test find_test_component(innerₕ(D₋ₓ(u), M₋ₓ(v))) == 3

        # nested wrappers all the way down
        @test find_trial_component(3 * D₋ₓ(restrict_to(:interior, M₊ᵧ(u)))) == 2
    end

    @testset "a term with no indexed leaf says so" begin
        # The docstrings promise an error here, and it used to be a MethodError naming an
        # internal function. Two different mistakes reach it: a form built from plain
        # trial and test functions rather than the indexed ones, and a form written the
        # wrong way round, where the search meets the other kind of leaf.
        @test_throws ArgumentError find_trial_component(
            innerₕ(TrialFunction{2}(), TestFunction{2}()))
        @test_throws ArgumentError find_test_component(
            innerₕ(TrialFunction{2}(), TestFunction{2}()))

        v1, v2 = IndexedTestFunction{2}(1), IndexedTestFunction{2}(2)
        @test_throws ArgumentError find_trial_component(innerₕ(D₋ₓ(v1), D₋ₓ(v2)))

        msg = try
            find_trial_component(innerₕ(TrialFunction{2}(), TestFunction{2}()))
        catch e
            sprint(showerror, e)
        end
        @test occursin("IndexedTrialFunction", msg)
        @test occursin("make_trial_args", msg)
    end

    @testset "splitting a Stokes-shaped form into blocks" begin
        U, P = make_trial_args(Xₕ, 2)
        V, Q = make_test_args(Xₕ, 2)

        # the velocity diagonal, plus a pressure-velocity coupling
        a = innerₕ(D₋ₓ(U[1]), D₋ₓ(V[1])) +
            innerₕ(D₋ᵧ(U[2]), D₋ᵧ(V[2])) +
            inner₊(P, D₋ₓ(V[1]))

        blocks = extract_block_asts(a, 3, 3)
        @test size(blocks) == (3, 3)

        filled = [blocks[i, j] !== nothing for i in 1:3, j in 1:3]
        @test filled == Bool[1 0 1; 0 1 0; 0 0 0]     # (1,1), (2,2) and (1,3)

        # each block holds the term that belongs to it
        @test find_trial_component(blocks[1, 1]) == 1
        @test find_test_component(blocks[1, 1]) == 1
        @test find_trial_component(blocks[1, 3]) == 3   # the pressure column
        @test find_test_component(blocks[1, 3]) == 1
    end

    @testset "several terms in one block are summed" begin
        u, v = IndexedTrialFunction{2}(1), IndexedTestFunction{2}(1)
        t1, t2 = innerₕ(u, v), innerₕ(D₋ₓ(u), D₋ₓ(v))

        one_term = extract_block_asts(t1, 1, 1)
        @test one_term[1, 1] === t1

        both = extract_block_asts(t1 + t2, 1, 1)
        @test both[1, 1] isa OperatorAdd
        @test length(flatten_sum(both[1, 1])) == 2

        # three of them accumulate too, and nothing leaks into another block
        three = extract_block_asts(t1 + t2 + innerₕ(M₋ₓ(u), M₋ₓ(v)), 2, 2)
        @test length(flatten_sum(three[1, 1])) == 3
        @test three[1, 2] === nothing
        @test three[2, 1] === nothing
        @test three[2, 2] === nothing
    end

    @testset "a component index outside the system is refused" begin
        # A form written against a larger space than the one being assembled would
        # otherwise write past the block matrix.
        u3, v1 = IndexedTrialFunction{2}(3), IndexedTestFunction{2}(1)
        @test_throws ErrorException extract_block_asts(innerₕ(u3, v1), 2, 2)

        u1, v3 = IndexedTrialFunction{2}(1), IndexedTestFunction{2}(3)
        @test_throws ErrorException extract_block_asts(innerₕ(u1, v3), 2, 2)

        # the same form is fine in a system large enough to hold it
        @test extract_block_asts(innerₕ(u3, v1), 3, 3)[1, 3] !== nothing
    end
end
