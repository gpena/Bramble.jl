using Test
using Bramble
using ForwardDiff
using DifferentiationInterface
using Bramble: values

# Which differentiation backends can differentiate through Bramble, and through one API.
#
# The operators mutate: every allocating form is `similar(uₕ)` followed by a stencil engine
# writing into it with `setindex!`. That one fact decides the survey, because it is what
# separates the backends that work from the ones that cannot.
#
#   ForwardDiff           works. Duals ride inside the arrays; mutation is irrelevant.
#   PolyesterForwardDiff  works. Chunked ForwardDiff, so the same reasoning.
#   ReverseDiff           works. Its tracked types flow through like Duals.
#   Mooncake              works, with no configuration.
#   Enzyme                works, with two annotations — see autodiff_heavy.jl.
#   Zygote                cannot. `setindex!` is unsupported, so it fails on the ordinary
#                         path too, not merely on the `!` forms. Supporting it would mean
#                         hand-written ChainRules rrules; out of scope by decision.
#   Diffractor            cannot even load on Julia 1.12 — it overwrites a Compiler method
#                         during precompilation and then overflows the stack.
#
# Written against DifferentiationInterface rather than each backend's own API. That is a
# deliberate choice and not only for brevity: DI is the layer the surrounding ecosystem
# reaches for, so testing through it tests the path a caller actually takes, and adding a
# backend becomes one more entry in a list instead of another block speaking another API.
#
# This file holds the two that run on every push: ForwardDiff, and ReverseDiff for the
# reverse direction. autodiff_heavy.jl holds the rest, which only the `ad` and `full` groups
# reach — Mooncake and Enzyme because they spend 58 s compiling between them, and
# PolyesterForwardDiff for a different reason.
#
# PolyesterForwardDiff exercises no Bramble path that ForwardDiff does not: it is the same
# Dual arithmetic chunked across threads, so its marginal coverage here is close to nothing.
# It also nests its threading inside kernels that already thread, and it is the one backend
# that has failed in CI — on the macOS runner, where it passes locally at one and four
# threads. Weekly is the right place for a canary like that.

_have(mod::Symbol) = Base.identify_package(String(mod)) !== nothing

# The two shapes worth checking: a scalar parameter, and a gradient in two. Both run the
# parameter through Rₕ and then an operator, which is the path that mutates.
function _ad_problems()
    Ωₕ = mesh(domain(interval(0.0, 1.0)), 6, true)
    Wₕ = gridspace(Ωₕ)
    scalar = a -> sum(values(D₋ₓ(Rₕ(Wₕ, x -> a * sin(x)))))
    vector = p -> begin
        uₕ = Rₕ(Wₕ, x -> p[1] * sin(x) + p[2] * x^2)
        return sum(values(D₋ₓ(uₕ))) + normₕ(uₕ)^2
    end
    return scalar, vector
end

# Every backend is checked the same way, against ForwardDiff and against a central
# difference, so a wrong answer fails rather than merely a thrown one.
function check_backend(name, backend)
    scalar, vector = _ad_problems()
    a0, p0 = 1.3, [1.3, 0.7]
    h = 1e-6

    @testset "$name" begin
        d = DifferentiationInterface.derivative(scalar, backend, a0)
        @test d ≈ ForwardDiff.derivative(scalar, a0) rtol=1e-8
        @test isapprox(d, (scalar(a0 + h) - scalar(a0 - h)) / 2h; rtol = 1e-5)

        g = DifferentiationInterface.gradient(vector, backend, p0)
        @test g ≈ ForwardDiff.gradient(vector, p0) rtol=1e-8
        for k in 1:2
            e = zeros(2)
            e[k] = h
            @test isapprox(g[k], (vector(p0 .+ e) - vector(p0 .- e)) / 2h; rtol = 1e-5)
        end
    end
end

@testset "AD backends, cheap" begin
    check_backend("AutoForwardDiff", AutoForwardDiff())

    if _have(:ReverseDiff)
        @eval import ReverseDiff
        check_backend("AutoReverseDiff", AutoReverseDiff())
    else
        @test_skip "ReverseDiff not in this environment"
    end

end
