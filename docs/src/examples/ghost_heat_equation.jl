# # Heat equation with time-dependent Neumann and Robin data
#
# The heat equation on a non-uniform grid with a flux prescribed on two sides of the square
# and a Robin condition on the other two, both with data that change in time. Space goes
# through the ghost layer (`gridspace` with ghosts, `constrained`), time through the method
# of lines ([`semidiscretize`](@ref), [`ode_problem`](@ref), `FBDF`), and the error at the
# final time is checked for second order in space. Every number and every plot below was
# produced by the code shown.
#
# ## Problem
#
# ```math
# \partial_t u = \Delta u \text{ in } \Omega \times (0, T], \qquad \Omega = (0,1)^2,
# ```
#
# with the initial condition ``u(\cdot, 0) = u_0`` and, with ``\partial_n`` the outward
# normal derivative,
#
# ```math
# \partial_n u = h(x, t) \text{ on } \texttt{xmax} \text{ and } \texttt{ymax}, \qquad
# 2\, u + \partial_n u = h(x, t) \text{ on } \texttt{xmin} \text{ and } \texttt{ymin}.
# ```
#
# The manufactured solution is ``u(x, y, t) = e^{-3t} e^{x} \cos 2y``: it solves the equation
# with no source, since ``\Delta (e^{x} \cos 2y) = -3\, e^{x} \cos 2y``. The data ``h`` come
# from its gradient, so they decay in time, and the initial condition is its value at
# ``t = 0``. The reference shares no code with the solve: it is this formula, restricted to
# the grid.

using Bramble
using OrdinaryDiffEqBDF
using Random

sol(p, t) = exp(-3t) * exp(p[1]) * cos(2p[2])
∇sol(p, t) = (sol(p, t), -2 * exp(-3t) * exp(p[1]) * sin(2p[2]))

# ## Boundary relations with data in time
#
# `constrained` takes a function of the trial function `u` and returns one
# `side => relation => data` entry per side, exactly as in the
# [Neumann and Robin Poisson example](neumann_robin_poisson.md). The one change is that the
# data may now take the time as a second argument, `(x, t) -> ...`; `semidiscretize` finds
# that by the arity and refreshes the boundary values at every step. A side with ``\alpha =
# 0`` carries the bare flux `∂ₙ(u)`:

const SIDES = (:xmin, :xmax, :ymin, :ymax)
const NORMALS = (-1.0, 1.0, -1.0, 1.0)
const AXES = (1, 1, 2, 2)
const ALPHA = (2.0, 0.0, 2.0, 0.0)

function side_relation(u, j)
    data = (p, t) -> ALPHA[j] * sol(p, t) + NORMALS[j] * ∇sol(p, t)[AXES[j]]
    relation = iszero(ALPHA[j]) ? ∂ₙ(u) : ALPHA[j] * u + ∂ₙ(u)
    return SIDES[j] => relation => data
end

# ## Semidiscretisation and time stepping
#
# The spatial operator is written ``\langle -\Delta_h u, v \rangle_h``, which reads the ghost
# values the relations fix, and the source is zero. The mass matrix is that of the discrete
# inner product, diagonal on the physical nodes, and the relations leave no algebraic
# row, so the system is an ordinary differential equation:
#
# ```math
# M \frac{\mathrm{d} u_h}{\mathrm{d} t} = F(t) - A u_h,
# ```
#
# where ``F(t)`` carries the boundary data at time ``t``. `FBDF` steps it with tolerances
# tight enough that the error left at the final time is the one in space. The final state is
# compared with the exact solution on the grid in the discrete ``L^2`` norm `normₕ`:

function solve_case(Ωₕ; T = 0.5)
    Cₕ = constrained(gridspace(Ωₕ; ghosts = :boundary),
        u -> ntuple(j -> side_relation(u, j), 4))
    a = form(Cₕ, Cₕ, (u, v) -> innerₕ(-1.0 * Δₕ(u), v))
    l = form(Cₕ, v -> innerₕ(x -> 0.0, v))

    Wₕ = gridspace(Ωₕ)
    prob = ode_problem(a, l, copy(parent(Rₕ(Wₕ, x -> sol(x, 0.0)))), (0.0, T))
    sl = solve(prob, FBDF(); reltol = 1e-10, abstol = 1e-12)

    uₕ = element(Wₕ)
    parent(uₕ) .= sl.u[end]
    return uₕ, Rₕ(Wₕ, x -> sol(x, T))
end

# The grid is a random non-uniform one refined in place by [`iterative_refinement!`](@ref),
# so each level is the same random mesh split dyadically, not a fresh draw:

function convergence(; levels = 4)
    Random.seed!(20261016)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    Ωₕ = mesh(Ω, (9, 9), (false, false))
    hs, errs = Float64[], Float64[]
    for level in 1:levels
        uₕ, uₑ = solve_case(Ωₕ)
        push!(hs, hₘₐₓ(Ωₕ))
        push!(errs, normₕ(uₕ .- uₑ))
        level < levels && iterative_refinement!(Ωₕ)
    end
    return hs, errs
end

order(hs, errs) = log(errs[end - 1] / errs[end]) / log(hs[end - 1] / hs[end])

hs, errs = convergence()
measured_order = order(hs, errs)

#-

measured_order > 1.8

# The line above renders its verdict; these assert it. Bracketed above as well as below:    #src
# an order far past the promise means the error has reached the time integrator's           #src
# tolerance and the ratio no longer measures the scheme.                                    #src
@test 1.8 < measured_order < 3.0                                                            #src
# An exactly zero error would be the manufactured solution reproduced by construction.      #src
@test all(e -> 1.0e-7 < e < 1.0e-2, errs)                                                   #src
# The solution decays by a factor e^-1.5 over the run, and the data in time drive it: a     #src
# solve whose boundary data stayed at t = 0 would leave an error of the size of the change, #src
# so the error must be a small fraction of the distance from the initial state.             #src
@test let Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 9), (true, true))  #src
    uₕ, uₑ = solve_case(Ωₕ)                                                                 #src
    u0 = copy(uₑ)                                                                           #src
    Rₕ!(u0, x -> sol(x, 0.0))                                                               #src
    normₕ(uₕ .- uₑ) < 0.05 * normₕ(u0 .- uₑ)                                                #src
end                                                                                         #src

# ## Second order
#
# The error at the final time against ``h_{\max}`` on log-log axes. The dashed line has slope
# 2 and is anchored at the finest point.

include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot([(hs, errs, "time-dependent Neumann and Robin", "#5B5FC7")];
    title = "Ghost heat equation, ‖·‖ₕ at t = 0.5") # hide

# The curve runs parallel to the slope-2 line on a grid built not to flatter the scheme: the
# points are random, so no symmetry cancels the one-sided error of the ghost layer. The
# boundary data enter the right-hand side through the same relations as in the steady case,
# and refreshing them in time costs no order.
#
# ## Where to go next
#
#   - [Neumann and Robin Poisson](neumann_robin_poisson.md) for the steady problem and the
#     pure Neumann case.
#   - [Heat equation](heat_equation.md) for the same method of lines with a Dirichlet
#     condition.
