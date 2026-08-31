using Test
using Bramble
using ForwardDiff
using Bramble: values

# Which automatic differentiation backends can differentiate through Bramble.
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
# This file holds the three cheap backends, which run on every push: 3.3 s between them,
# load included. Mooncake and Enzyme cost 58 s of first-call compilation between them and
# live in autodiff_heavy.jl, which only the `ad` and `full` groups reach.
#
# Each backend is still loaded only if it is present in the active environment, so an
# environment missing one reports a skip rather than an error.

_have(mod::Symbol) = Base.identify_package(String(mod)) !== nothing

@testset "AD backends, cheap" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0)), 6, true)
    Wₕ = gridspace(Ωₕ)

    # A scalar functional of one parameter, and a gradient-shaped one of two. Both run the
    # parameter through Rₕ and then through an operator, which is the path that mutates.
    J(a) = sum(values(D₋ₓ(Rₕ(Wₕ, x -> a * sin(x)))))
    function Jv(p)
        uₕ = Rₕ(Wₕ, x -> p[1] * sin(x) + p[2] * x^2)
        return sum(values(D₋ₓ(uₕ))) + normₕ(uₕ)^2
    end

    a0 = 1.3
    p0 = [1.3, 0.7]
    d_ref = ForwardDiff.derivative(J, a0)
    g_ref = ForwardDiff.gradient(Jv, p0)

    @testset "ForwardDiff is the reference, and agrees with finite differences" begin
        h = 1e-6
        @test isapprox(d_ref, (J(a0 + h) - J(a0 - h)) / 2h; rtol = 1e-5)
        for k in 1:2
            e = zeros(2)
            e[k] = h
            @test isapprox(g_ref[k], (Jv(p0 .+ e) - Jv(p0 .- e)) / 2h; rtol = 1e-5)
        end
    end

    @testset "ReverseDiff" begin
        if _have(:ReverseDiff)
            @eval using ReverseDiff
            @test ReverseDiff.gradient(Jv, p0) ≈ g_ref rtol=1e-8
        else
            @test_skip "ReverseDiff not in this environment"
        end
    end

    @testset "PolyesterForwardDiff" begin
        if _have(:PolyesterForwardDiff)
            @eval using PolyesterForwardDiff
            g = similar(p0)
            PolyesterForwardDiff.threaded_gradient!(Jv, g, p0, ForwardDiff.Chunk(1))
            @test g ≈ g_ref rtol=1e-8
        else
            @test_skip "PolyesterForwardDiff not in this environment"
        end
    end
end
