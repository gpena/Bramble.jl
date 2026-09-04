using Test
using Bramble
using Bramble: IdentityOperator, IndexedTrialFunction, IndexedTestFunction,
               TrialFunction, TestFunction,
               BackwardDifference, ForwardDifference, DifferenceNode,
               get_innermost_dim, is_symbolic,
               resolve_ast, trial_component_or_nothing, test_component_or_nothing,
               grad_backward, grad_forward

# The two one-sided difference nodes of the symbolic layer.
#
# They are meant to be interchangeable: whatever a form can do with D₋ it can do with D₊.
# Previously, `get_derivative_matrix_and_scale` and `get_innermost_dim` existed for
# the backward node alone, so any form built on D₊ met a MethodError as soon as it was
# assembled.
#
# The one deliberate exception is `inner₊`, which takes backward differences only. Its
# weights are the staggered ones of the summation-by-parts identity, and those pair with a
# backward difference; a forward difference sits on the other staggering.

@testset "Difference nodes" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, false))
    Wₕ = gridspace(Ωₕ)
    id = IdentityOperator(Wₕ)

    BD, FD = typeof(D₋ₓ(id)), typeof(D₊ₓ(id))

    @testset "Scaling recursion" begin
        for op in (D₋ₓ(id), D₊ₓ(id))
            @test get_innermost_dim(2 * op) == 1
            @test get_innermost_dim(op / 4) == 1
            @test get_innermost_dim(7 * (op / 4)) == 1
        end
    end

    @testset "Backward vs forward parity" begin
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

    @testset "inner₊ backward-only" begin
        @test hasmethod(inner₊, Tuple{IndexedTrialFunction{2}, BD})
        @test hasmethod(inner₊, Tuple{IndexedTestFunction{2}, BD})
        @test hasmethod(inner₊, Tuple{BD, IndexedTrialFunction{2}})

        # There is no symbolic method for the forward node, and none of the
        # direction-inferring ones accept it. It used to fall through to the *numeric*
        # `inner₊(uₕ, vₕ)` over grid functions and fail there, complaining about types the
        # caller never wrote; the guard in inner.jl now catches it as a usage error.
        for T in (FD, typeof(D₊ᵧ(id)))
            m = which(inner₊, Tuple{IndexedTrialFunction{2}, T})
            @test !occursin("ForwardDifference", string(m.sig))
            @test occursin("form/operators/inner.jl", replace(string(m.file), "\\" => "/"))
        end

        u2, v2 = TrialFunction{2}(), TestFunction{2}()
        @test_throws ArgumentError inner₊(u2, D₊ₓ(v2))
        @test_throws ArgumentError inner₊(D₊ₓ(u2), D₊ₓ(v2))
    end

    @testset "AST equivalence" begin
        for (bwd, fwd) in ((D₋ₓ(id), D₊ₓ(id)), (D₋ᵧ(id), D₊ᵧ(id)))
            @test is_symbolic(bwd) == is_symbolic(fwd) == false
            @test resolve_ast(bwd) isa BackwardDifference
            @test resolve_ast(fwd) isa ForwardDifference
            @test bwd isa DifferenceNode
            @test fwd isa DifferenceNode
        end

        # the block walk reaches its leaf through either node, and through a scaling on
        # top of it. Its leaf is an indexed trial or test function (that is what it is
        # looking for), so it is those the nodes wrap here, not the identity.
        u, v = IndexedTrialFunction{2}(3), IndexedTestFunction{2}(2)
        for D in (D₋ₓ, D₊ₓ, D₋ᵧ, D₊ᵧ)
            @test trial_component_or_nothing(D(u)) == 3
            @test test_component_or_nothing(D(v)) == 2
            @test trial_component_or_nothing(7 * D(u)) == 3
            @test test_component_or_nothing(7 * D(v)) == 2
        end

        # a symbolic leaf makes the whole node symbolic, either way round
        for D in (D₋ₓ, D₊ₓ)
            @test is_symbolic(D(u))
            @test is_symbolic(D(v))
            @test !is_symbolic(D(id))
        end
    end

    @testset "Gradient shapes" begin
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
