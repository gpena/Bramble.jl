# # Acoustic wave equation in two dimensions
#
# The [heat equation](heat_equation.md) hands `M u' = F - A u` to a stepper, first order in
# time. A wave is second order, and [`semidiscretize`](@ref) takes the first-order form, so
# the first-order system for the pair `(u, u̇)` is written out below.
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
# an exact solution of the continuous problem, so the error below can be measured. It starts
# at rest in the shape of the first mode and oscillates in place with period ``\sqrt{2}``.
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
# block is ``c^2 K`` on the interior rows, which the energy below reuses.
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
# The Dirichlet rows of the mass matrix are zero, which makes the system
# differential-algebraic, so [`ode_problem`](@ref) asks for a method that admits a singular
# mass matrix. `Rodas5P` is one, and the [heat equation](heat_equation.md) page names it
# beside `FBDF`.

u₀ = Rₕ(Wₕ, x -> uexact(x, 0.0))
v₀ = Rₕ(Wₕ, x -> duexact(x, 0.0))

prob = ode_problem(sd, vcat(parent(u₀), parent(v₀)), I)
sol = solve(prob, Rodas5P(); reltol = 1.0e-8, abstol = 1.0e-10)

SciMLBase.successful_retcode(sol), length(sol.t)

# The solution at `t = 1` against the exact standing wave. The first `n` entries of the state
# are the displacement, the last `n` the velocity:

uT = element(Wₕ, sol.u[end][1:n])
err = normₕ(uT - Rₕ(Wₕ, x -> uexact(x, 1.0)))

# Bracketed on both sides: an exactly zero error would mean the exact solution had         #src
# been reproduced by construction instead of stepped for.                                  #src
@test 1.0e-4 < err < 1.0e-2                                                                #src
@test abs(sol.u[end][1]) < 1.0e-8 && abs(sol.u[end][n + 1]) < 1.0e-8                       #src

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
# and so does the semidiscrete system, with the discrete counterpart
# ``\tfrac{1}{2} v^{\mathsf T} M v + \tfrac{c^2}{2} u^{\mathsf T} K u``. `Rodas5P` is not
# symplectic, so the stepper can still make the energy drift, and the drift is measured:

Mᵥ = M[(n + 1):(2n), (n + 1):(2n)]
Kc = operator_matrix(sd)[(n + 1):(2n), 1:n]
energy(y) = 0.5 * dot(y[(n + 1):(2n)], Mᵥ * y[(n + 1):(2n)]) + 0.5 * dot(y[1:n], Kc * y[1:n])

E = energy.(sol.u)
drift = maximum(abs, E .- E[1]) / E[1]

@test 0.0 < drift < 1.0e-10                                                                #src
@test 50 <= length(sol.t) <= 60                                                            #src
@test E[1] ≈ π^2 / 4 rtol = 5.0e-3                                                         #src

E[1], drift

# The initial energy is close to the continuous ``\pi^2/4`` and the relative drift over the
# 56 stored time points is about `1e-11`. It follows the integrator's tolerance, and
# tightening `reltol` to `1e-10` lowers it:

sol_tight = solve(prob, Rodas5P(); reltol = 1.0e-10, abstol = 1.0e-10)
drift_tight = maximum(abs, energy.(sol_tight.u) .- E[1]) / E[1]

@test 30 < drift / drift_tight < 300                                                       #src

drift_tight
#
# ## Spatial order of the wave solution
#
# Refining space with the time tolerance held fixed. On the uniform mesh above,
# ``\sin(\pi x)\sin(\pi y)`` is an exact discrete eigenmode of the Laplacian, a special case
# that says little about the scheme. The series therefore runs on one seeded random coarse
# mesh, refined in place with [`iterative_refinement!`](@ref), so every finer level is the
# same random mesh with each cell halved:

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

# The observed orders are 2.04 and 2.01: writing the wave as a first-order system keeps the
# second-order accuracy in space.
#
# ## Where to go next
#
# - [Heat equation](heat_equation.md), the first-order counterpart, and where
#   [`semidiscretize`](@ref) is introduced.
# - [Linear and bilinear forms](@ref tutorial_form) for the stiffness form and the
#   [Dirichlet conditions](@ref form_dirichlet) used here.
# - [`ode_problem`](@ref) for the problem the stepper receives, and the solver restriction on
#   a singular mass matrix.
