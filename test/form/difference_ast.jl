module FormDifferenceAstTests

using Test
using Bramble
# Internal since v3.0 (gpena/Bramble.jl#211): defined and documented, not exported.
import Bramble: D₊ₓ, D₊ᵧ, ∇₊ₕ
using Bramble:
               IdentityOperator,
               IndexedTrialFunction,
               IndexedTestFunction,
               TrialFunction,
               TestFunction,
               BackwardDifference,
               ForwardDifference,
               DifferenceNode,
               is_symbolic,
               resolve_ast,
               trial_component_or_nothing,
               test_component_or_nothing

# The two one-sided difference nodes of the symbolic layer.
#
# They are meant to be interchangeable: whatever a form can do with D₋ it can do with D₊.
# Previously, `get_derivative_matrix_and_scale` existed for the backward node alone, so any
# form built on D₊ met a MethodError as soon as it was assembled.
#
# The one deliberate exception is `inner₊`, which takes backward differences only. Its
# weights are the staggered ones of the summation-by-parts identity, and those pair with a
# backward difference; a forward difference sits on the other staggering.

@testset "Difference nodes" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, false))
    Wₕ = gridspace(Ωₕ)
    id = IdentityOperator(Wₕ)

    BD, FD = typeof(D₋ₓ(id)), typeof(D₊ₓ(id))

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

        backward_only = setdiff(
            mentioning("BackwardDifference"), mentioning("ForwardDifference")
        )
        forward_only = setdiff(
            mentioning("ForwardDifference"), mentioning("BackwardDifference")
        )

        # Two exceptions, and both are the same exception. `inner₊` is the documented one:
        # the modified inner product is defined against backward differences and has no
        # forward counterpart. `_separable_axis` (`src/form/kronecker.jl`, gpena/Bramble.jl#162)
        # inherits it rather than introducing a second asymmetry -- it matches
        # `BilinearProduct{D, InnerPlus{Dim}, ...}`, so it can only ever see the operand
        # `inner₊` itself admits. A form written with forward differences is simply not
        # recognised as separable, which is the conservative direction: `is_separable` may
        # answer no to something separable, never yes to something that is not.
        @test backward_only == Set([:inner₊, :_separable_axis])
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
        # an indexed trial/test pair, for the symbolic-leaf checks below. The component
        # walk over these nodes is block_extract.jl's "Component search".
        u, v = IndexedTrialFunction{2}(3), IndexedTestFunction{2}(2)

        for (bwd, fwd) in ((D₋ₓ(id), D₊ₓ(id)), (D₋ᵧ(id), D₊ᵧ(id)))
            @test is_symbolic(bwd) == is_symbolic(fwd) == false
            @test resolve_ast(bwd) isa BackwardDifference
            @test resolve_ast(fwd) isa ForwardDifference
            @test bwd isa DifferenceNode
            @test fwd isa DifferenceNode
        end

        # a symbolic leaf makes the whole node symbolic, either way round
        for D in (D₋ₓ, D₊ₓ)
            @test is_symbolic(D(u))
            @test is_symbolic(D(v))
            @test !is_symbolic(D(id))
        end
    end

    @testset "Gradient shapes" begin
        # `grad_backward`/`grad_forward` were these two under another name until
        # gpena/Bramble.jl#74 generated the families; the gradients themselves are what
        # they always were.
        @test ∇ₕ(id) isa NTuple{2, BackwardDifference}
        @test ∇₊ₕ(id) isa NTuple{2, ForwardDifference}
        @test ∇ₕ(id) === (D₋(id, Val(1)), D₋(id, Val(2)))
        @test ∇₊ₕ(id) === (Bramble.D₊(id, Val(1)), Bramble.D₊(id, Val(2)))

        # the tuple form, applied component-wise, which only ∇ₕ used to have
        @test ∇ₕ((id, id)) == map(∇ₕ, (id, id))
        @test ∇₊ₕ((id, id)) == map(∇₊ₕ, (id, id))
        @test length(∇₊ₕ((id, id))) == 2
        @test all(g -> g isa NTuple{2, ForwardDifference}, ∇₊ₕ((id, id)))

        # in one dimension the gradient is the node itself, not a 1-tuple
        Ω1 = mesh(domain(interval(0.0, 1.0)), 7, true)
        id1 = IdentityOperator(gridspace(Ω1))
        @test !(∇ₕ(id1) isa Tuple)
        @test !(∇₊ₕ(id1) isa Tuple)
    end

    @testset "Operator tuple scaling" begin
        k_elem = element(Wₕ, 2.5)
        kx_elem = element(Wₕ, 1.2)
        ky_elem = element(Wₕ, 3.4)

        # Scalar and VectorElement scaling
        g_sc = 3.0 * ∇ₕ(id)
        @test g_sc isa NTuple{2, Bramble.OperatorScale}
        @test ∇ₕ(id) * 3.0 isa NTuple{2, Bramble.OperatorScale}

        g_elem = k_elem * ∇ₕ(id)
        @test g_elem isa NTuple{2, Bramble.GridFunctionScale}
        @test ∇ₕ(id) * k_elem isa NTuple{2, Bramble.GridFunctionScale}

        # Component-wise tuple scaling
        g_tuple = (kx_elem, ky_elem) * ∇ₕ(id)
        @test g_tuple isa NTuple{2, Bramble.GridFunctionScale}
        @test g_tuple[1].grid_function === kx_elem
        @test g_tuple[2].grid_function === ky_elem

        # Assembly correctness
        a_scaled = form(Wₕ, Wₕ, (u, v) -> inner₊(k_elem * ∇ₕ(u), ∇ₕ(v)))
        a_manual = form(Wₕ, Wₕ, (u, v) -> inner₊(k_elem * D₋ₓ(u), D₋ₓ(v)) + inner₊(k_elem * D₋ᵧ(u), D₋ᵧ(v)))
        @test assemble(a_scaled) ≈ assemble(a_manual)

        a_aniso = form(Wₕ, Wₕ, (u, v) -> inner₊((kx_elem, ky_elem) * ∇ₕ(u), ∇ₕ(v)))
        a_aniso_man = form(Wₕ, Wₕ, (u, v) -> inner₊(kx_elem * D₋ₓ(u), D₋ₓ(v)) + inner₊(ky_elem * D₋ᵧ(u), D₋ᵧ(v)))
        @test assemble(a_aniso) ≈ assemble(a_aniso_man)
    end
end

end # module FormDifferenceAstTests
