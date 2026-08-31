using Test
using Bramble
using ForwardDiff
using Bramble: values

# The two differentiation backends that are expensive to run, and the two that cannot.
#
# The operators mutate: every allocating form is `similar(uₕ)` followed by a stencil engine
# writing into it with `setindex!`. That single fact decides the whole survey, because it is
# what separates the backends that can be used from the ones that cannot.
#
#   ForwardDiff           works. Duals ride inside the arrays; mutation is irrelevant.
#   PolyesterForwardDiff  works. Chunked ForwardDiff, so the same reasoning.
#   ReverseDiff           works. Its tracked types flow through like Duals.
#   Mooncake              works, with no configuration.
#   Enzyme                works, but only with `set_runtime_activity`. Without it every
#                         call fails with EnzymeRuntimeActivityError.
#   Zygote                cannot. `setindex!` is unsupported, so it fails on any operator,
#                         not merely on the `!` forms. Supporting it would mean hand-written
#                         ChainRules rrules; it is out of scope by decision.
#   Diffractor            cannot even load on Julia 1.12 — it overwrites a Compiler method
#                         during precompilation and then overflows the stack. Nothing to do
#                         with Bramble.
#
# Each backend is loaded only if it is present in the active environment, so this file costs
# nothing in a test run that does not have them. Enzyme and Mooncake in particular are
# expensive to install and compile, and are meant to be enabled deliberately rather than
# carried by every run. `_have(:Name)` is what gates each block.

_have(mod::Symbol) = Base.identify_package(String(mod)) !== nothing

@testset "AD backends, expensive" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0)), 6, true)
    Wₕ = gridspace(Ωₕ)
    J(a) = sum(values(D₋ₓ(Rₕ(Wₕ, x -> a * sin(x)))))
    a0 = 1.3
    d_ref = ForwardDiff.derivative(J, a0)

    @testset "Mooncake" begin
        if _have(:Mooncake)
            @eval using Mooncake
            rule = Mooncake.build_rrule(J, a0)
            _, grad = Mooncake.value_and_gradient!!(rule, J, a0)
            @test grad[2] ≈ d_ref rtol=1e-8
        else
            @test_skip "Mooncake not in this environment"
        end
    end

    @testset "Enzyme needs runtime activity" begin
        if _have(:Enzyme)
            @eval using Enzyme
            # Two things Enzyme needs here, neither of them a Bramble problem.
            #
            # `set_runtime_activity`, or every call raises EnzymeRuntimeActivityError:
            # Enzyme cannot prove activity through the closure and the mesh, and says so.
            #
            # And `Const(J)`, because `J` closes over the grid space. Enzyme requires the
            # function argument itself to be provably readonly, and a closure carrying a
            # mutable capture is not — `EnzymeMutabilityException` otherwise. Marking it
            # `Const` says the closure is not being differentiated with respect to.
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
            d = Enzyme.autodiff(mode, Enzyme.Const(J), Enzyme.Active,
                Enzyme.Active(a0))[1][1]
            @test d ≈ d_ref rtol=1e-8
        else
            @test_skip "Enzyme not in this environment"
        end
    end

    @testset "Zygote is out of scope, and this records why" begin
        if _have(:Zygote)
            @eval using Zygote
            # Not a limitation of the `!` forms: the allocating `D₋ₓ` writes into a
            # `similar`, so Zygote fails on the ordinary path too. If this ever starts
            # passing, Zygote has gained mutation support and the decision can be revisited.
            @test_throws Exception Zygote.gradient(J, a0)
        else
            @test_skip "Zygote not in this environment"
        end
    end
end
