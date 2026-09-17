# # Acoustic wave equation in two dimensions
#
# Every time-dependent page so far has been first order in time: the
# [heat equation](heat_equation.md) hands `M u' = F - A u` to a stepper. A wave is second
# order, and Bramble semidiscretises it as such rather than asking the caller to rewrite it
# as a first-order system by hand.
#
# This page is generated from `docs/src/examples/wave_equation_2d.jl` by Literate.jl, and the
# same file runs under `test/examples/wave_equation_2d.jl` with the assertions the page
# renders but does not check. It needs `SciMLBase` and a stepper, so it runs behind the
# `ext`/`full` test groups rather than on every push.
#
# ## Problem
#
# ```math
# \partial_{tt} u - c^2 \Delta u = 0 \text{ in } \Omega \times (0, T], \qquad
# u = 0 \text{ on } \partial\Omega, \qquad \Omega = (0,1)^2
# ```
#
# with ``c = 1`` and the standing wave
#
# ```math
# u_{\text{exact}}(x, y, t) = \sin(\pi x)\sin(\pi y)\cos(\sqrt{2}\,\pi c\, t),
# ```
#
# an exact solution of the continuous problem, so the error below is measurable rather than
# merely plausible. It starts at rest in the shape of the first mode and oscillates in place
# with period ``\sqrt{2}``.
#
# ## Semidiscretising in space
#
# [`semidiscretize_second_order`](@ref) takes the stiffness form and a source and produces
#
# ```math
# M \ddot{u}_h + C \dot{u}_h + K u_h = F(t),
# ```
#
# with no damping form here, so `C` is absent. Dirichlet conditions constrain the
# displacement the same way [`semidiscretize`](@ref) constrains its own state, and the
# consistent velocity follows from differentiating that constraint rather than being
# prescribed separately.

using Bramble
using SciMLBase
using OrdinaryDiffEqRosenbrock
using LinearAlgebra: dot

const c = 1.0
const ω = c * sqrt(2) * π

uexact(x, t) = sinpi(x[1]) * sinpi(x[2]) * cos(ω * t)
duexact(x, t) = -ω * sinpi(x[1]) * sinpi(x[2]) * sin(ω * t)

function wave_system(n)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    Ωₕ = mesh(Ω, (n, n), (true, true))
    Wₕ = gridspace(Ωₕ)
    I = interval(0.0, 1.0)                       # the time domain

    fₕ = element(Wₕ, 0.0)                        # no forcing
    K = form(Wₕ, Wₕ, (u, v) -> c^2 * inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)

    return Ωₕ, Wₕ, I, semidiscretize_second_order(K, l; dirichlet = bcs)
end

Ωₕ, Wₕ, I, sd = wave_system(25)

size(stiffness_matrix(sd)), damping_matrix(sd), size(block_mass_matrix(sd))

# Three matrices are worth naming. [`stiffness_matrix`](@ref) is `K`, with a constrained row
# replaced by `eₖ`; [`mass_matrix`](@ref) is `M`, whose constrained rows are zeroed, which is
# what makes the system differential-algebraic; and [`block_mass_matrix`](@ref) is the
# `2n × 2n` matrix the first-order form needs, `M` above the identity.
#
# ## Stepping it
#
# [`second_order_ode_problem`](@ref) hands the whole thing to `OrdinaryDiffEq` as a
# `SecondOrderODEProblem`, with the state ordered velocity first, displacement second.
#
# A symplectic integrator would be the natural choice for a conservative problem, and it
# cannot be used here: `OrdinaryDiffEqCore` refuses an explicit or symplectic method unless
# the mass matrix is exactly `I`, and a discrete `innerₕ` mass matrix never is, since its
# boundary rows carry a half weight. `Rodas5P` is mass-matrix aware and is what
# [`SecondOrderSemidiscretization`](@ref)'s own docstring points to.

u₀ = Rₕ(Wₕ, x -> uexact(x, 0.0))
du₀ = Rₕ(Wₕ, x -> duexact(x, 0.0))

prob = second_order_ode_problem(sd, du₀, u₀, I)
sol = solve(prob, Rodas5P(); reltol = 1.0e-8, abstol = 1.0e-10)

SciMLBase.successful_retcode(sol), length(sol.t)

# The solution at `t = 1` against the exact standing wave. `sol.u[end].x[2]` is the
# displacement half of the state, `x[1]` the velocity:

uT = element(Wₕ, sol.u[end].x[2])
err = normₕ(uT - Rₕ(Wₕ, x -> uexact(x, 1.0)))

# Bracketed rather than one-sided: an exactly zero error would mean the exact solution had #src
# been reproduced by construction instead of stepped for.                                  #src
@test 1.0e-4 < err < 1.0e-2                                                                #src

err

#-

include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide
surface_plot(uT; title = "u at t = 1, on a 25 × 25 mesh") # hide

# ## Energy
#
# The continuous problem conserves
#
# ```math
# E(t) = \tfrac{1}{2}\|\dot u\|_M^2 + \tfrac{1}{2} u^{\mathsf T} K u,
# ```
#
# and the semidiscrete system conserves its discrete counterpart exactly, up to what the time
# stepper does to it. `Rodas5P` is not symplectic and has no conservation property of its own,
# so the right claim is about the drift, measured rather than assumed:

M, Kmat = mass_matrix(sd), stiffness_matrix(sd)
energy(state) = 0.5 * dot(state.x[1], M * state.x[1]) + 0.5 * dot(state.x[2], Kmat * state.x[2])

E = energy.(sol.u)
drift = maximum(abs, E .- E[1]) / E[1]

@test 0.0 < drift < 1.0e-8                                                                 #src

E[1], drift

# Eleven digits of the initial energy survive a hundred-odd adaptive steps. That is the
# integrator's tolerance showing rather than a conservation theorem: tightening `reltol` to
# `1e-10` takes the drift to around `1e-14`, and loosening it gives the digits back.
#
# ## Order of convergence
#
# Refining space, with the time tolerance held fixed and tight enough that the spatial error
# is what is being measured:

function wave_error(n)
    Ωₕ, Wₕ, I, sd = wave_system(n)
    prob = second_order_ode_problem(
        sd, Rₕ(Wₕ, x -> duexact(x, 0.0)), Rₕ(Wₕ, x -> uexact(x, 0.0)), I)
    sol = solve(prob, Rodas5P(); reltol = 1.0e-8, abstol = 1.0e-10)
    uT = element(Wₕ, sol.u[end].x[2])
    return hₘₐₓ(Ωₕ), normₕ(uT - Rₕ(Wₕ, x -> uexact(x, 1.0)))
end

hs, errs = zip(wave_error.((13, 25, 41))...)
orders = [log(errs[i] / errs[i + 1]) / log(hs[i] / hs[i + 1]) for i in 1:(length(hs) - 1)]

@test all(1.9 .< orders .< 2.1)                                                            #src

collect(orders)

#-

include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot([(collect(hs), collect(errs), "2D wave", "#5B5FC7")]; # hide
    title = "Wave equation at t = 1, ‖·‖ₕ") # hide

# Second order, the same rate the steady problems reach, which is the point: the second-order
# semidiscretisation does not cost the scheme its spatial accuracy.
#
# ## See also
#
# - [Heat equation](heat_equation.md), the first-order counterpart, and where
#   [`semidiscretize`](@ref) is introduced.
# - [`SecondOrderSemidiscretization`](@ref) for the full accessor list and the solver
#   restriction.
