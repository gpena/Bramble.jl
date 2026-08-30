using Test
using Bramble
using Bramble: IdentityOperator, IndexedTrialFunction, IndexedTestFunction,
               BackwardDifference, ForwardDifference, DifferenceNode,
               get_derivative_matrix_and_scale, get_innermost_dim, is_symbolic,
               resolve_ast, find_trial_component, find_test_component,
               backward_finite_difference, forward_finite_difference,
               grad_backward, grad_forward

# The two one-sided difference nodes of the symbolic layer.
#
# They are meant to be interchangeable: whatever a form can do with D₋ it can do with D₊.
# That had drifted — `get_derivative_matrix_and_scale` and `get_innermost_dim` existed for
# the backward node alone, so any form built on D₊ met a MethodError as soon as it was
# assembled. The backward one was also calling `backward_difference_matrix`, a name no
# revision of the package ever defined.
#
# The one deliberate exception is `inner₊`, which takes backward differences only. Its
# weights are the staggered ones of the summation-by-parts identity, and those pair with a
# backward difference; a forward difference sits on the other staggering.

@testset "Symbolic difference nodes" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, false))
    Wₕ = gridspace(Ωₕ)
    id = IdentityOperator(Wₕ)

    BD, FD = typeof(D₋ₓ(id)), typeof(D₊ₓ(id))

    @testset "both resolve to the matrix they stand for" begin
        # each node carries its direction, and resolves to the difference *including* its
        # 1/h weights — the same matrix the space layer builds directly
        for (op, want, dim) in ((D₋ₓ(id), backward_finite_difference(Wₕ, Val(1)), 1),
            (D₊ₓ(id), forward_finite_difference(Wₕ, Val(1)), 1),
            (D₋ᵧ(id), backward_finite_difference(Wₕ, Val(2)), 2),
            (D₊ᵧ(id), forward_finite_difference(Wₕ, Val(2)), 2))
            mat, scale = get_derivative_matrix_and_scale(op, Wₕ)
            @test mat == want
            @test get_innermost_dim(op) == dim

            # `1`, not `1.0`: the scale has to promote against the space's element type
            # rather than dragging a Float32 assembly up to Float64
            @test scale === 1
            @test scale isa Integer
        end
    end

    @testset "the scaling nodes recurse through both alike" begin
        for op in (D₋ₓ(id), D₊ₓ(id))
            _, s = get_derivative_matrix_and_scale(2 * op, Wₕ)
            @test s == 2
            @test get_innermost_dim(2 * op) == 1
            @test get_innermost_dim(op / 4) == 1
        end
    end

    @testset "every tool the backward node has, the forward node has" begin
        # A structural guard rather than a list of cases: whatever generic function has a
        # method mentioning one node must have one mentioning the other. A method written
        # against the `DifferenceNode` alias satisfies it for both at once.
        function mentioning(needle)
            found = Set{Symbol}()
            for nm in names(Bramble; all = true)
                startswith(string(nm), '#') && continue
                isdefined(Bramble, nm) || continue
                f = getfield(Bramble, nm)
                f isa Function || continue
                for m in methods(f)
                    occursin(needle, string(m.sig)) && (push!(found, nm); break)
                end
            end
            return found
        end

        backward_only = setdiff(mentioning("BackwardDifference"),
            mentioning("ForwardDifference"))
        forward_only = setdiff(mentioning("ForwardDifference"),
            mentioning("BackwardDifference"))

        # `inner₊` is the documented exception, and the only one
        @test backward_only == Set([:inner₊])
        @test isempty(forward_only)
    end

    @testset "inner₊ is backward only, by design" begin
        @test hasmethod(inner₊, Tuple{IndexedTrialFunction{2}, BD})
        @test hasmethod(inner₊, Tuple{IndexedTestFunction{2}, BD})
        @test hasmethod(inner₊, Tuple{BD, IndexedTrialFunction{2}})

        # There is no symbolic method for the forward node. Dispatch does not stop there,
        # though: it falls through to the numeric `inner₊(uₕ, vₕ)` over grid functions,
        # which is not a symbolic method at all.
        for T in (FD, typeof(D₊ᵧ(id)))
            m = which(inner₊, Tuple{IndexedTrialFunction{2}, T})
            @test !occursin("ForwardDifference", string(m.sig))
            @test occursin("inner_product.jl", string(m.file))
        end
    end

    @testset "the rest of the AST treats them the same" begin
        for (bwd, fwd) in ((D₋ₓ(id), D₊ₓ(id)), (D₋ᵧ(id), D₊ᵧ(id)))
            @test is_symbolic(bwd) == is_symbolic(fwd) == false
            @test resolve_ast(bwd) isa BackwardDifference
            @test resolve_ast(fwd) isa ForwardDifference
            @test bwd isa DifferenceNode
            @test fwd isa DifferenceNode
        end

        # the block walk reaches its leaf through either node, and through a scaling on
        # top of it. Its leaf is an indexed trial or test function — that is what it is
        # looking for — so it is those the nodes wrap here, not the identity.
        u, v = IndexedTrialFunction{2}(3), IndexedTestFunction{2}(2)
        for D in (D₋ₓ, D₊ₓ, D₋ᵧ, D₊ᵧ)
            @test find_trial_component(D(u)) == 3
            @test find_test_component(D(v)) == 2
            @test find_trial_component(7 * D(u)) == 3
            @test find_test_component(7 * D(v)) == 2
        end

        # a symbolic leaf makes the whole node symbolic, either way round
        for D in (D₋ₓ, D₊ₓ)
            @test is_symbolic(D(u))
            @test is_symbolic(D(v))
            @test !is_symbolic(D(id))
        end
    end

    @testset "gradients, in both directions and both shapes" begin
        @test grad_backward(id) isa NTuple{2, BackwardDifference}
        @test grad_forward(id) isa NTuple{2, ForwardDifference}
        @test ∇₋ₕ(id) === grad_backward(id)
        @test ∇₊ₕ(id) === grad_forward(id)

        # the tuple form, applied component-wise, which only ∇₋ₕ used to have
        @test ∇₋ₕ((id, id)) == map(grad_backward, (id, id))
        @test ∇₊ₕ((id, id)) == map(grad_forward, (id, id))
        @test length(∇₊ₕ((id, id))) == 2
        @test all(g -> g isa NTuple{2, ForwardDifference}, ∇₊ₕ((id, id)))

        # in one dimension the gradient is the node itself, not a 1-tuple
        Ω1 = mesh(domain(interval(0.0, 1.0)), 7, true)
        id1 = IdentityOperator(gridspace(Ω1))
        @test !(∇₋ₕ(id1) isa Tuple)
        @test !(∇₊ₕ(id1) isa Tuple)
        @test get_innermost_dim(∇₊ₕ(id1)) == 1
    end
end
