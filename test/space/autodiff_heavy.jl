module SpaceAutodiffHeavyTests

using Test
using Bramble
using DifferentiationInterface
using ..SpaceAutodiffBackendsTests: check_backend, _ad_problems
using ..TestUtils: _have

# Enzyme is the expensive backend to compile, and Zygote is the one that cannot run at all.
#
# Enzyme's first-call compilation is what makes it expensive, so it sits behind the
# `backends` group and the weekly workflow rather than running on every push. What it
# establishes changes when a *backend* changes rather than when Bramble does.
#
# `check_backend` comes from autodiff_backends.jl, which the group includes first; `_have`
# is TestUtils'.

@testset "AD backends (slow)" begin
    @testset "Enzyme annotations" begin
        if _have(:Enzyme)
            @eval import Enzyme

            # Neither annotation is a workaround for a Bramble defect, and both produce
            # errors that read like one.
            #
            # `set_runtime_activity`: Enzyme cannot prove activity through a closure that
            # captures a grid space and a mesh, and raises EnzymeRuntimeActivityError.
            #
            # `function_annotation = Const`: Enzyme requires the *function* argument to be
            # provably readonly, and a closure carrying a grid space is not. Through
            # Enzyme's own API this is `Enzyme.Const(f)`; through DI it is this field, which
            # the failure message names when it is missing.
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
            check_backend(
                "AutoEnzyme", AutoEnzyme(mode = mode, function_annotation = Enzyme.Const)
            )

            # Without the annotation it fails, which is worth pinning: if a later Enzyme
            # infers this on its own, this test tells us the annotation can go.
            scalar, _ = _ad_problems()
            @test_throws Exception DifferentiationInterface.derivative(
                scalar, AutoEnzyme(mode = mode), 1.3
            )
        else
            @test_skip "Enzyme not in this environment"
        end
    end

    @testset "Zygote out of scope" begin
        if _have(:Zygote)
            @eval import Zygote
            # Not a limitation of the `!` forms: the allocating `D₋ₓ` writes into a
            # `similar`, so Zygote fails on the ordinary path too. If this ever starts
            # passing, Zygote has gained mutation support and the decision can be revisited.
            scalar, _ = _ad_problems()
            @test_throws Exception DifferentiationInterface.derivative(
                scalar, AutoZygote(), 1.3
            )
        else
            @test_skip "Zygote not in this environment"
        end
    end
end

end # module SpaceAutodiffHeavyTests
