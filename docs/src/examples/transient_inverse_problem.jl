# # Recovering parameters from a trajectory (transient adjoint gradients)
#
# [Recovering a diffusion coefficient](inverse_diffusion.md) inverts a *steady* problem: one
# solve, one set of point observations, gpena/Bramble.jl#228's own adjoint rule for
# [`pde_solve`](@ref). This page inverts a *transient* one -- a handful of noisy observations
# taken at several different times, of a solution that is itself evolving -- which needs a
# different adjoint: [`Bramble.adjoint_sensitivities`](@ref), wrapping
# `SciMLSensitivity.adjoint_sensitivities` for a [`Semidiscretization`](@ref)'s solved
# trajectory (gpena/Bramble.jl#239).
#
# Two parameters are recovered here, not one, entering the problem two different ways: `κ`
# scales the initial condition, `β` is the rate of a boundary value ramping up over time. One
# backward solve gives the gradient with respect to *both* at once, the same O(1)-in-parameter-
# count trade the steady page's adjoint rule makes.
#
# This page is generated from `docs/src/examples/transient_inverse_problem.jl` by
# Literate.jl, and the same file runs under `test/examples/transient_inverse_problem.jl` with
# the assertions that the page renders but does not check. It needs `SciMLSensitivity` and
# runs behind the `ext`/`full` test groups, unlike the every-push example pages -- see that
# test file for why.
#
# ## Problem
#
# ```math
# u_t - u_{xx} = 0 \text{ in } (0,1), \qquad u(0,t) = u(1,t) = \beta t, \qquad
# u(x,0) = \kappa \sin(\pi x),
# ```
#
# with no source term: every parameter reaches the trajectory through the initial condition
# or the boundary values, not through a source. `κ` and `β` -- the initial condition's
# amplitude and the boundary ramp's rate -- are both unknown; five noisy point observations
# of `u`, taken at five different times, are all that is given.
#
# ## Solving it

using Bramble

κ_true, β_true = 0.6, 0.9

Ωₕ = mesh(domain(interval(0.0, 1.0)), 41, true)
Wₕ = gridspace(Ωₕ)
I = interval(0.0, 0.5)
fₕ = Rₕ(Wₕ, x -> 0.0)
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
l = form(Wₕ, v -> innerₕ(fₕ, v))
bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t, β) -> β[1] * t)
sd = semidiscretize(a, l; dirichlet = bcs)

# `sd` is built once, fixed for the rest of the page: `κ` reaches the *initial condition*
# `ode_problem` is given, `β` reaches the *residual*'s own `p` through the `(x, t, β)`
# boundary condition above -- neither one touches `sd`/`a`/`l` themselves, so nothing here
# needs rebuilding per parameter the way [the steady page's](inverse_diffusion.md) `a`
# does for a coefficient scaling the operator.

pts = points(Ωₕ)
u₀(κ) = begin
    v = [κ * sinpi(x[1]) for x in pts]
    v[1] = 0.0
    v[end] = 0.0
    v
end

# ## Solving forward
#
# `specialize = SciMLBase.FullSpecialize` is required, not optional, for
# [`Bramble.adjoint_sensitivities`](@ref) to reach this trajectory afterwards: without it, the
# adjoint's own `β`-vjp calls the residual with a differently-typed `β` than the forward
# solve used, and the default specialization's function wrapper raises rather than
# differentiates -- [`ode_problem`](@ref)'s own docstring has the reasoning.

using SciMLBase, OrdinaryDiffEqBDF

ts = [0.1, 0.2, 0.3, 0.4, 0.5]

function forward(κ, β)
    prob = ode_problem(sd, u₀(κ), I; p = [β], specialize = SciMLBase.FullSpecialize)
    return solve(prob, FBDF(); abstol = 1e-9, reltol = 1e-9, saveat = ts)
end

# ## Synthetic observations
#
# Three interior points, the true trajectory there at each of the five times above plus a
# small fixed perturbation standing in for measurement noise -- fixed rather than
# `Random`-drawn, so the page renders the same numbers every build. Fifteen data points in
# all (three points × five times), the "trajectory" the steady page's single-time-point
# recovery has no equivalent of.

obs_idx = (11, 21, 31)
noise = (0.003, -0.004, 0.002)
sol_true = forward(κ_true, β_true)
obs = [[sol_true.u[k][i] + noise[j] for (j, i) in enumerate(obs_idx)] for k in eachindex(ts)]

loss(κ, β) = begin
    sol = forward(κ, β)
    sum(
        sum(abs2, sol.u[k][obs_idx[j]] - obs[k][j] for j in eachindex(obs_idx))
    for k in eachindex(ts)
    )
end

# ## Gradient descent, driven by the adjoint
#
# `using SciMLSensitivity` is what makes `BrambleSciMLSensitivityExt` define
# [`Bramble.adjoint_sensitivities`](@ref), which reaches SciMLSensitivity's own adjoint
# machinery with two corrections a `Semidiscretization`'s index-1 DAE needs and
# `SciMLSensitivity`'s own defaults do not supply -- see that function's docstring for both.
#
# One call returns everything needed for *both* parameters: `du0`, already corrected for the
# mass matrix, gives `∂J/∂κ` by the chain rule through `u₀(κ) = κ * sinpi(x)`; `dp[1]` is
# already `∂J/∂β`, since `β` reached the residual through its own `p` rather than through a
# rebuilt initial condition.

using SciMLSensitivity, LinearAlgebra

function ∂J(κ, β)
    prob = ode_problem(sd, u₀(κ), I; p = [β], specialize = SciMLBase.FullSpecialize)
    sol = solve(prob, FBDF(); abstol = 1e-9, reltol = 1e-9, saveat = ts)
    function dgdu!(out, u, p, t, k)
        fill!(out, 0.0)
        for (j, i) in enumerate(obs_idx)
            out[i] = 2 * (u[i] - obs[k][j])
        end
        return nothing
    end
    du0, dp = Bramble.adjoint_sensitivities(
        sol, FBDF(); t = ts, dgdu_discrete = dgdu!, abstol = 1e-9, reltol = 1e-9
    )
    ∂J_∂κ = dot(du0, sinpi.(getindex.(pts, 1)))
    return ∂J_∂κ, dp[1]
end

κ, β = 0.2, 0.3   # deliberately far from (κ_true, β_true) = (0.6, 0.9)
history = [(κ, β)]
step_κ, step_β = 0.05, 0.02
for _ in 1:150
    g_κ, g_β = ∂J(κ, β)
    global κ -= step_κ * g_κ
    global β -= step_β * g_β
    push!(history, (κ, β))
end

println("(κ, β) = ", (κ, β))

# Both parameters recovered to within the noise level the synthetic observations carry, from
# a starting guess a factor of three away on `κ` and a factor of three on `β`.
@test isapprox(κ, κ_true; atol = 0.01)                                                #src
@test isapprox(β, β_true; atol = 0.01)                                                #src
@test loss(κ, β) < loss(0.2, 0.3)                                                     #src

# ## Cost: one backward solve, for every parameter and every saved step at once
#
# `∂J` above costs one forward solve plus one backward (adjoint) solve, regardless of how
# many parameters `κ`/`β` were together, and regardless of how many points `ts` saves the
# trajectory at -- the same O(1)-in-parameter-count trade the steady page's adjoint rule
# makes, extended to O(1) in the number of saved steps too. A coefficient *field* with one
# value per grid point, or a trajectory saved at a hundred times instead of five, would cost
# this same pair of solves; forward-mode differentiation through the same model would cost
# one solve per parameter, and checkpointing costs would grow with the number of saved steps
# for a naive discrete adjoint.
#
# ## See also
#
#   - [`Bramble.adjoint_sensitivities`](@ref) and [`ode_problem`](@ref) in the
#     [API reference](../api_sciml.md).
#   - [Recovering a diffusion coefficient](inverse_diffusion.md) for the steady-state adjoint
#     this extends.
#   - [Heat equation](heat_equation.md) for the forward transient problem this inverts.
