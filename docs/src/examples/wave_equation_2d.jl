# # Acoustic wave equation in two dimensions
#
# Every time-dependent page so far has been first order in time: the
# [heat equation](heat_equation.md) hands `M u' = F - A u` to a stepper. A wave is second
# order, and Bramble has no second-order semidiscretisation of its own: the caller writes the
# first-order system for the pair `(u, u̇)` and hands that to [`semidiscretize`](@ref), which
# is what this page does.
#
# This script runs under `test/examples/ext_pages.jl`, in the `examples` test group, with the
# assertions the page renders but does not check; it is not part of the published manual. It
# needs `SciMLBase` and a stepper, so it does not run on every push.
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
# The boundary stays Dirichlet-only here because the state is a composite space and
# `semidiscretize` and `ode_problem` refuse a composite of ghost spaces for now.
#
# ## The first-order system
#
# With the velocity ``v = \dot u`` as a second unknown, the wave equation is
#
# ```math
# \dot u = v, \qquad \dot v = c^2 \Delta u,
# ```
#
# a state ``(u, v)`` on two copies of the scalar space, `Wₕ × Wₕ`. [`semidiscretize`](@ref)
# expects ``M y' = F(t) - A y``, so each equation is multiplied by the mass form and moved
# into that shape:
#
# ```math
# M \dot u = M v \;\Rightarrow\; A_{12} = -M, \qquad
# M \dot v = -c^2 K u \;\Rightarrow\; A_{21} = c^2 K,
# ```
#
# with ``K`` the stiffness form, so one form on `Wₕ × Wₕ` carries both blocks. The mass is
# the default, `innerₕ` on both leaves, and there is no forcing. Homogeneous Dirichlet
# conditions go on both leaves, `u = 0` and, differentiating the constraint, `v = 0` too;
# `dirichlet_components` names the leaves the label binds to, and both are listed. As in the
# heat equation, a constrained row has a zero mass row and reads as the boundary condition
# itself.

using Bramble
using Bramble: mass_matrix, operator_matrix
using SciMLBase
using OrdinaryDiffEqRosenbrock
using LinearAlgebra: dot

const c = 1.0
const ω = c * sqrt(2) * π

uexact(x, t) = sinpi(x[1]) * sinpi(x[2]) * cos(ω * t)
duexact(x, t) = -ω * sinpi(x[1]) * sinpi(x[2]) * sin(ω * t)

function wave_system(Ωₕ)
    Wₕ = gridspace(Ωₕ)
    Vₕ = Wₕ × Wₕ                                 # the state (u, v)
    I = interval(0.0, 1.0)                       # the time domain

    fₕ = element(Wₕ, 0.0)                        # no forcing
    a = form(Vₕ, Vₕ,
        (y, w) -> -innerₕ(y(2), w(1)) + c^2 * inner₊(∇ₕ(y(1)), ∇ₕ(w(2))))
    l = form(Vₕ, w -> innerₕ(fₕ, w(1)) + innerₕ(fₕ, w(2)))
    bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)

    return Wₕ, I, semidiscretize(a, l; dirichlet = bcs, dirichlet_components = (1, 2))
end

Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ = mesh(Ω, (25, 25))
Wₕ, I, sd = wave_system(Ωₕ)
n = ndofs(Wₕ)

size(mass_matrix(sd)), size(operator_matrix(sd))

# [`mass_matrix`](@ref) is the stacked `2n × 2n` mass matrix, one diagonal block per leaf, and
# [`operator_matrix`](@ref) is ``A``, with a constrained row replaced by `eₖ`; its lower left
# block is ``c^2 K``, which the energy below reuses.
#
# The mesh above is uniform, so the weights are known: an interior row of `M` carries the cell
# measure ``h_x h_y = (1/24)^2`` on each leaf, and a Dirichlet row is zero on each.

M = mass_matrix(sd)
mid = 13 + 25 * 12                              # an interior point
@test iszero(M[1, 1]) && iszero(M[n + 1, n + 1]) #src
@test M[mid, mid] ≈ (1 / 24)^2 && M[n + mid, n + mid] ≈ (1 / 24)^2 #src
#
# ## Stepping it
#
# [`ode_problem`](@ref) hands the whole thing to `OrdinaryDiffEq`, with the state ordered
# displacement first, velocity second.
#
# A symplectic integrator would be the natural choice for a conservative problem, and it
# cannot be used here: `OrdinaryDiffEqCore` refuses an explicit or symplectic method unless
# the mass matrix is exactly `I`, and the mass matrix here never is: its interior rows carry
# the cell measure ``h_x h_y``, and its Dirichlet rows are zero, which is what makes the
# system differential-algebraic. `Rodas5P` is mass-matrix aware, and is what the
# [heat equation](heat_equation.md) page names beside `FBDF`.

u₀ = Rₕ(Wₕ, x -> uexact(x, 0.0))
v₀ = Rₕ(Wₕ, x -> duexact(x, 0.0))

prob = ode_problem(sd, vcat(parent(u₀), parent(v₀)), I)
sol = solve(prob, Rodas5P(); reltol = 1.0e-8, abstol = 1.0e-10)

SciMLBase.successful_retcode(sol), length(sol.t)

# The solution at `t = 1` against the exact standing wave. The first `n` entries of the state
# are the displacement, the last `n` the velocity:

uT = element(Wₕ, sol.u[end][1:n])
err = normₕ(uT - Rₕ(Wₕ, x -> uexact(x, 1.0)))

# Bracketed rather than one-sided: an exactly zero error would mean the exact solution had #src
# been reproduced by construction instead of stepped for.                                  #src
@test 1.0e-4 < err < 1.0e-2                                                                #src

err

#-

include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide
surface_plot(uT; title = "u at t = 1, on a 25 × 25 mesh") # hide

# ## Energy drift
#
# The continuous problem conserves
#
# ```math
# E(t) = \tfrac{1}{2}\|\dot u\|_M^2 + \tfrac{c^2}{2} u^{\mathsf T} K u,
# ```
#
# and the semidiscrete system conserves its discrete counterpart exactly, up to what the time
# stepper does to it. `Rodas5P` is not symplectic and has no conservation property of its own,
# so the right claim is about the drift, measured rather than assumed:

Mᵥ = M[(n + 1):(2n), (n + 1):(2n)]
Kc = operator_matrix(sd)[(n + 1):(2n), 1:n]
energy(y) = 0.5 * dot(y[(n + 1):(2n)], Mᵥ * y[(n + 1):(2n)]) + 0.5 * dot(y[1:n], Kc * y[1:n])

E = energy.(sol.u)
drift = maximum(abs, E .- E[1]) / E[1]

@test 0.0 < drift < 1.0e-8                                                                 #src
@test length(sol.t) < 100                                                                  #src

E[1], drift

# Eleven digits of the initial energy survive some fifty adaptive steps. That is the
# integrator's tolerance showing rather than a conservation theorem: tightening `reltol` to
# `1e-10` takes the drift to around `1e-14`, and loosening it gives the digits back.
#
# ## Spatial order of the wave solution
#
# Refining space, with the time tolerance held fixed and tight enough that the spatial error
# is what is being measured. The uniform mesh above would flatter the rate, because on it
# ``\sin(\pi x)\sin(\pi y)`` is an exact discrete eigenmode of the Laplacian. That special
# case says little about the scheme. The series therefore runs on one seeded random coarse
# mesh, refined in place with [`iterative_refinement!`](@ref) so every finer level is the same
# random mesh, dyadically split. The uniform mesh stays above, where the check on the mass
# matrix needs a known grid weight:

using Random

function wave_error(Ωₕ)
    Wₕ, I, sd = wave_system(Ωₕ)
    y₀ = vcat(parent(Rₕ(Wₕ, x -> uexact(x, 0.0))), parent(Rₕ(Wₕ, x -> duexact(x, 0.0))))
    prob = ode_problem(sd, y₀, I)
    sol = solve(prob, Rodas5P(); reltol = 1.0e-8, abstol = 1.0e-10)
    uT = element(Wₕ, sol.u[end][1:ndofs(Wₕ)])
    return hₘₐₓ(Ωₕ), normₕ(uT - Rₕ(Wₕ, x -> uexact(x, 1.0)))
end

function wave_series(levels; n0 = 7)
    Random.seed!(20260903)
    Ωc = mesh(Ω, (n0, n0), (false, false))
    hs, errs = Float64[], Float64[]
    for level in 1:levels
        h, e = wave_error(Ωc)
        push!(hs, h)
        push!(errs, e)
        level < levels && iterative_refinement!(Ωc)
    end
    return hs, errs
end

hs, errs = wave_series(3)   # 7² up to 25² points
orders = [log(errs[i] / errs[i + 1]) / log(hs[i] / hs[i + 1]) for i in 1:(length(hs) - 1)]

@test all(1.9 .< orders .< 2.1)                                                            #src

orders

#-

include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot([(hs, errs, "2D wave", "#5B5FC7")]; # hide
    title = "Wave equation at t = 1, ‖·‖ₕ") # hide

# Second order, the same rate the steady problems reach, which is the point: writing the wave
# as a first-order system does not cost the scheme its spatial accuracy.
#
# ## Where to go next
#
# - [Heat equation](heat_equation.md), the first-order counterpart, and where
#   [`semidiscretize`](@ref) is introduced.
# - [Linear and bilinear forms](@ref tutorial_form) for the stiffness form and the
#   [Dirichlet conditions](@ref form_dirichlet) used here.
# - [`ode_problem`](@ref) for the problem the stepper receives, and the solver restriction on
#   a singular mass matrix.
