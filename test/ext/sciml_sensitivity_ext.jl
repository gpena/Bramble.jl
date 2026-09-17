module ExtSciMLSensitivityExtTests

using Test
using Bramble
using LinearAlgebra: dot
using SciMLBase: SciMLBase, solve
using OrdinaryDiffEqBDF: FBDF

# `Bramble.adjoint_sensitivities`: `SciMLSensitivity.adjoint_sensitivities` wrapped with the
# two corrections gpena/Bramble.jl#239's own composition experiment found necessary --
# `initializealg = BrownFullBasicInit()` (SciMLSensitivity's own default only checks the
# adjoint DAE's algebraic consistency at `t = T`, and a constrained problem's seed fails
# that check) and an `Mᵀ` correction on the returned initial-condition gradient
# (`adjoint_sensitivities` returns `λ(0)`, dropping the mass matrix the Lagrangian's own
# boundary term carries there). Both checked here against central differences, not just
# against "the call didn't error" -- `bramble-verification`'s whole point.
#
# `SciMLSensitivity` is a genuinely heavy dependency (Zygote, Tracker, ReverseDiff, NNlib
# all load behind it), so this lives in the "ext" group precisely like `sciml_ext.jl`, never
# on the default push path.
using SciMLSensitivity

_ssens_uex(x, θ) = θ * sinpi(x[1])

function _ssens_problem(n; uniform = true)
    Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), n, uniform)
    Wₕ = gridspace(Ωₕ)
    I = Bramble.interval(0.0, 0.5)
    fₕ = Bramble.element(Wₕ, 0.0)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t, p) -> p[1] * t)
    sd = semidiscretize(a, l; dirichlet = bcs)
    return Ωₕ, Wₕ, I, sd
end

@testset "BrambleSciMLSensitivityExt" begin
    @testset "parameter gradient matches central differences" begin
        Ωₕ, Wₕ, I, sd = _ssens_problem(21)
        u₀ = Rₕ(Wₕ, x -> 0.0)
        p₀ = [0.7]
        ts = [0.125, 0.25, 0.375, 0.5]

        prob = ode_problem(sd, u₀, I; p = p₀, specialize = SciMLBase.FullSpecialize)
        sol = solve(prob, FBDF(); abstol = 1e-12, reltol = 1e-12, saveat = ts)

        obs = [copy(u) .+ 0.01 for u in sol.u]
        dgdu!(out, u, p, t, i) = (@. out = 2 * (u - obs[i]); nothing)

        du0, dp = Bramble.adjoint_sensitivities(
            sol, FBDF(); t = ts, dgdu_discrete = dgdu!, abstol = 1e-12, reltol = 1e-12
        )

        loss(p) = begin
            s = solve(
                SciMLBase.remake(prob; p = p), FBDF(); abstol = 1e-12, reltol = 1e-12,
                saveat = ts
            )
            sum(sum(abs2, s.u[k] - obs[k]) for k in eachindex(ts))
        end
        h = 1e-6
        fd = (loss(p₀ .+ [h]) - loss(p₀ .- [h])) / (2h)

        @test isapprox(vec(dp)[1], fd; rtol = 1e-5)
    end

    @testset "initial-condition gradient is Mᵀ-corrected" begin
        # A non-uniform mesh, deliberately: `M`'s diagonal is not a single scalar `h`, so an
        # uncorrected `du0` off by `1/h` and a correctly `Mᵀ`-corrected one are told apart by
        # more than a constant factor -- the same check the composition experiment used to
        # confirm the correction is the full `Mᵀ`, not a mesh-width fudge.
        Ωₕ, Wₕ, I, sd = _ssens_problem(21; uniform = false)
        p₀ = [0.7]
        θ₀ = 0.4
        ts = [0.125, 0.25, 0.375, 0.5]
        pts = points(Ωₕ)
        u0θ(θ) = begin
            v = [_ssens_uex(x, θ) for x in pts]
            v[1] = 0.0
            v[end] = 0.0
            v
        end

        prob = ode_problem(sd, u0θ(θ₀), I; p = p₀, specialize = SciMLBase.FullSpecialize)
        sol = solve(prob, FBDF(); abstol = 1e-12, reltol = 1e-12, saveat = ts)
        obs = [copy(u) .+ 0.01 for u in sol.u]
        dgdu!(out, u, p, t, i) = (@. out = 2 * (u - obs[i]); nothing)

        du0, dp = Bramble.adjoint_sensitivities(
            sol, FBDF(); t = ts, dgdu_discrete = dgdu!, abstol = 1e-12, reltol = 1e-12
        )

        h = 1e-6
        du0dθ = (u0θ(θ₀ + h) - u0θ(θ₀ - h)) / (2h)
        adjoint_dθ = dot(du0, du0dθ)

        loss(θ) = begin
            s = solve(
                SciMLBase.remake(prob; u0 = u0θ(θ)), FBDF(); abstol = 1e-12, reltol = 1e-12,
                saveat = ts
            )
            sum(sum(abs2, s.u[k] - obs[k]) for k in eachindex(ts))
        end
        fd_dθ = (loss(θ₀ + h) - loss(θ₀ - h)) / (2h)

        @test isapprox(adjoint_dθ, fd_dθ; rtol = 1e-4)
    end

    @testset "a non-ODESolution first argument still reaches the core stub" begin
        # This extension's own method restricts its first argument to `ODESolution`, so
        # anything else -- even with `SciMLSensitivity` loaded, as it is in this file --
        # falls through to `Bramble.adjoint_sensitivities`'s core fallback
        # (`form/semidiscrete.jl`), the one a user actually sees if they call this before
        # `using SciMLSensitivity` at all. Pinned here since that fallback is what carries
        # the "add `using SciMLSensitivity`" guidance, and dispatch falling through to it
        # correctly is what this test checks, not the message text alone.
        @test_throws "requires SciMLSensitivity.jl" Bramble.adjoint_sensitivities(
            nothing, nothing
        )
    end
end

end # module ExtSciMLSensitivityExtTests
