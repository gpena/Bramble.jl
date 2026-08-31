using Test
using Bramble
using ForwardDiff
using DifferentiationInterface
using Bramble: values

# The two backends that are expensive to run, and the one that cannot run at all.
#
# Mooncake and Enzyme cost 25.1 s and 33.2 s of first-call compilation, against 0.5 s for
# ReverseDiff, which runs per push. What they establish changes when a
# *backend* changes rather than when Bramble does, so they sit behind the `ad` group and the
# weekly workflow rather than running on every push.
#
# `check_backend` and `_have` come from autodiff_backends.jl, which the group includes first.

@testset "AD backends, expensive" begin
    @testset "PolyesterForwardDiff" begin
        # Here rather than in the per-push group for two reasons. It exercises no Bramble
        # path ForwardDiff does not — the same Dual arithmetic, chunked across threads — so
        # its marginal coverage is close to nothing. And it nests its own threading inside
        # kernels that already thread, which is the kind of thing that behaves differently
        # on a CI runner than on a laptop: it passes here at one and four threads and
        # errored on the macOS runner.
        if _have(:PolyesterForwardDiff)
            @eval import PolyesterForwardDiff
            check_backend("AutoPolyesterForwardDiff", AutoPolyesterForwardDiff())
        else
            @test_skip "PolyesterForwardDiff not in this environment"
        end
    end

    @testset "Mooncake" begin
        if _have(:Mooncake)
            @eval import Mooncake
            check_backend("AutoMooncake", AutoMooncake(config = nothing))
        else
            @test_skip "Mooncake not in this environment"
        end
    end

    @testset "Enzyme, and the two annotations it needs" begin
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
            check_backend("AutoEnzyme",
                AutoEnzyme(mode = mode, function_annotation = Enzyme.Const))

            # Without the annotation it fails, which is worth pinning: if a later Enzyme
            # infers this on its own, this test tells us the annotation can go.
            scalar, _ = _ad_problems()
            @test_throws Exception DifferentiationInterface.derivative(
                scalar, AutoEnzyme(mode = mode), 1.3)
        else
            @test_skip "Enzyme not in this environment"
        end
    end

    @testset "Zygote is out of scope, and this records why" begin
        if _have(:Zygote)
            @eval import Zygote
            # Not a limitation of the `!` forms: the allocating `D₋ₓ` writes into a
            # `similar`, so Zygote fails on the ordinary path too. If this ever starts
            # passing, Zygote has gained mutation support and the decision can be revisited.
            scalar, _ = _ad_problems()
            @test_throws Exception DifferentiationInterface.derivative(
                scalar, AutoZygote(), 1.3)
        else
            @test_skip "Zygote not in this environment"
        end
    end
end
