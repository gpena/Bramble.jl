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
# `SciMLSensitivity` is not a `test/Project.toml` dependency, the same `_have`/`@test_skip`
# idiom `test/ext/chainrules_enzyme_ext.jl` already uses for Enzyme/Mooncake and for the same
# reason: it pulls in Zygote, Tracker, ReverseDiff and NNlib, none of which anything else
# here needs, so keeping it out of the ordinary dependency graph is what lets an every-push
# run skip that compile cost entirely rather than merely defer loading it. `Weekly.yml`'s
# "add the expensive differentiation backends" step installs it at CI runtime for exactly
# this group.

_have(mod::Symbol) = Base.identify_package(String(mod)) !== nothing

@testset "BrambleSciMLSensitivityExt" begin
    @testset "a non-ODESolution first argument still reaches the core stub" begin
        # This extension's own method restricts its first argument to `ODESolution`, so
        # anything else falls through to `Bramble.adjoint_sensitivities`'s core fallback
        # (`form/semidiscrete.jl`), the one a user actually sees if they call this before
        # `using SciMLSensitivity` at all -- checked outside the `_have` guard below since
        # this is the one behavior that is *more* representative without the extension
        # loaded, not less.
        @test_throws "requires SciMLSensitivity.jl" Bramble.adjoint_sensitivities(
            nothing, nothing
        )
    end

    if _have(:SciMLSensitivity)
        @eval using SciMLSensitivity

        # Only the `using` above needs `@eval` -- everything below is ordinary code in the
        # same branch, the same split `chainrules_enzyme_ext.jl` uses for Enzyme.
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

        @testset "parameter gradient matches central differences" begin
            Ωₕ, Wₕ, I, sd = _ssens_problem(21)
            u₀ = Rₕ(Wₕ, x -> 0.0)
            p₀ = [0.7]
            ts = [0.125, 0.25, 0.375, 0.5]

            prob = ode_problem(sd, u₀, I; p = p₀, specialize = SciMLBase.FullSpecialize)
            sol = solve(prob, FBDF(); abstol = 1e-9, reltol = 1e-9, saveat = ts)

            obs = [copy(u) .+ 0.01 for u in sol.u]
            dgdu!(out, u, p, t, i) = (@. out = 2 * (u - obs[i]); nothing)

            du0, dp = Bramble.adjoint_sensitivities(
                sol, FBDF(); t = ts, dgdu_discrete = dgdu!, abstol = 1e-9, reltol = 1e-9
            )

            loss(p) = begin
                s = solve(
                    SciMLBase.remake(prob; p = p), FBDF(); abstol = 1e-9, reltol = 1e-9,
                    saveat = ts
                )
                sum(sum(abs2, s.u[k] - obs[k]) for k in eachindex(ts))
            end
            # `abstol`/`reltol` at `1e-9`, not `1e-12`: measured directly, `1e-12` made
            # FBDF's corrector intermittently unstable on this stiff, algebraically
            # constrained, non-uniform-mesh problem -- one of the two `θ₀ ± h` forward
            # solves below would occasionally blow up (a boundary DOF read back at ~600
            # instead of its constrained value), silently corrupting the finite-difference
            # reference with a `retcode = Success` result that was actually garbage. Not a
            # gradient bug: `Bramble.adjoint_sensitivities` itself never touched `1e-9` vs
            # `1e-12`, and the *adjoint* value stayed consistent across runs throughout --
            # only the FD reference, run twice per check, was exposed to the instability
            # twice as often. `h = 1e-4` keeps the FD reference's own noise floor
            # (`solver_tol / h ~ 1e-5`) well under this test's `rtol`.
            h = 1e-4
            fd = (loss(p₀ .+ [h]) - loss(p₀ .- [h])) / (2h)

            @test isapprox(vec(dp)[1], fd; rtol = 1e-5)
        end

        @testset "initial-condition gradient is Mᵀ-corrected" begin
            # A non-uniform mesh, deliberately: `M`'s diagonal is not a single scalar `h`, so
            # an uncorrected `du0` off by `1/h` and a correctly `Mᵀ`-corrected one are told
            # apart by more than a constant factor -- the same check the composition
            # experiment used to confirm the correction is the full `Mᵀ`, not a mesh-width
            # fudge.
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
            sol = solve(prob, FBDF(); abstol = 1e-9, reltol = 1e-9, saveat = ts)
            obs = [copy(u) .+ 0.01 for u in sol.u]
            dgdu!(out, u, p, t, i) = (@. out = 2 * (u - obs[i]); nothing)

            du0, dp = Bramble.adjoint_sensitivities(
                sol, FBDF(); t = ts, dgdu_discrete = dgdu!, abstol = 1e-9, reltol = 1e-9
            )

            # Same `abstol`/`reltol`/`h` reasoning as the parameter-gradient testset above:
            # `1e-9`, not `1e-12`, keeps `loss(θ)`'s own forward solves (below) clear of the
            # same intermittent FBDF instability measured there.
            h = 1e-4
            du0dθ = (u0θ(θ₀ + h) - u0θ(θ₀ - h)) / (2h)
            adjoint_dθ = dot(du0, du0dθ)

            loss(θ) = begin
                s = solve(
                    SciMLBase.remake(prob; u0 = u0θ(θ)), FBDF(); abstol = 1e-9,
                    reltol = 1e-9, saveat = ts
                )
                sum(sum(abs2, s.u[k] - obs[k]) for k in eachindex(ts))
            end
            fd_dθ = (loss(θ₀ + h) - loss(θ₀ - h)) / (2h)

            @test isapprox(adjoint_dθ, fd_dθ; rtol = 1e-4)
        end
    else
        @test_skip "SciMLSensitivity not in this environment"
    end
end

end # module ExtSciMLSensitivityExtTests
