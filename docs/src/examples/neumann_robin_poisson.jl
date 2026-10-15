# # Neumann, Robin and pure Neumann boundaries
#
# Poisson's equation on a non-uniform grid with a flux or a mixed condition on the whole
# boundary, solved through the public path (`gridspace` with ghost points, `constrained`,
# `form`, `assemble`, `pde_solve`) and checked for second order. Every number and every
# plot below was produced by the code shown.
#
# ## Problem
#
# ```math
# -\Delta u = g \text{ in } \Omega = (0,1)^2, \qquad
# \alpha\, u + \partial_n u = h \text{ on } \partial\Omega,
# ```
#
# with ``\partial_n`` the outward normal derivative and a coefficient ``\alpha \ge 0`` that
# may change from one side of the square to the next. Three choices of ``\alpha``:
#
#   - **Robin and Neumann**: ``\alpha = 2`` on `xmin` and `ymin`, ``\alpha = 0`` (Neumann) on
#     `xmax` and `ymax`;
#   - **Robin**: ``\alpha = 2`` on `xmin` and `ymin`, ``\alpha = 1`` on `xmax` and `ymax`;
#   - **pure Neumann**: ``\alpha = 0`` on every side.
#
# The manufactured solution is ``u(x, y) = e^{x} \cos 2y``, so ``g = 3u`` and the data
# ``h = \alpha u + \partial_n u`` come from its gradient.
#
# A pure Neumann problem fixes ``u`` only up to a constant, and the data must satisfy a
# compatibility condition, which the manufactured data do. The solve below selects the
# solution of discrete mean zero, and the error is measured against the exact solution
# shifted to that same mean.
#
# ## One boundary relation per side
#
# Ghost points come first: `gridspace(Ωₕ; ghosts = :boundary)` adds one layer outside every
# side. `constrained` then takes a function of the trial function `u` and returns one
# `side => relation => data` entry per side. A Neumann side is `∂ₙ(u)`, a Robin side is
# `α * u + ∂ₙ(u)`, and the data is a function of the boundary point. Each relation fixes a
# ghost value from the interior ones, which leaves the physical nodes as the only unknowns.

using Bramble
using LinearAlgebra: dot
using Random

sol(p) = exp(p[1]) * cos(2p[2])
∇sol(p) = (sol(p), -2 * exp(p[1]) * sin(2p[2]))
rhs(p) = 3 * sol(p)

# The sides, their outward normals and the coordinate each normal points along:

const SIDES = (:xmin, :xmax, :ymin, :ymax)
const NORMALS = (-1.0, 1.0, -1.0, 1.0)
const AXES = (1, 1, 2, 2)

# The data of the relation on side `j` is ``\alpha u + \partial_n u`` evaluated on the
# exact solution. A side with ``\alpha = 0`` carries the bare flux `∂ₙ(u)`:

function side_relation(u, α, j)
    data = p -> α[j] * sol(p) + NORMALS[j] * ∇sol(p)[AXES[j]]
    relation = iszero(α[j]) ? ∂ₙ(u) : α[j] * u + ∂ₙ(u)
    return SIDES[j] => relation => data
end

# ## Assembling and solving
#
# `assemble(a, l)` takes the bilinear and the linear form together on the constrained
# space, so the relations' data reach the right-hand side. When ``\alpha`` vanishes
# on every side the matrix is singular, its null space the constants. `pde_solve` then
# takes `mean_zero`, the vector of cell measures that is the discrete integral, and solves
# the system bordered by that constraint.

function solve_case(Ωₕ, α)
    Cₕ = constrained(gridspace(Ωₕ; ghosts = :boundary),
        u -> ntuple(j -> side_relation(u, α, j), 4))
    A, F = assemble(form(Cₕ, Cₕ, (u, v) -> innerₕ(-1.0 * Δₕ(u), v)),
        form(Cₕ, v -> innerₕ(rhs, v)))

    Wₕ = gridspace(Ωₕ)
    uₕ = element(Wₕ)
    uₑ = Rₕ(Wₕ, sol)
    if all(iszero, α)
        w = assemble(form(Wₕ, v -> innerₕ(x -> 1.0, v)))
        uₕ .= pde_solve(A, F; mean_zero = w)
        uₑ .-= dot(w, uₑ) / sum(w)
    else
        uₕ .= pde_solve(A, F)
    end
    return uₕ, uₑ
end

# The error is the discrete ``H^1`` norm `norm₁ₕ`, and the grid is a random non-uniform
# one refined in place by [`iterative_refinement!`](@ref), so each level is the same
# random mesh split dyadically, not a fresh draw:

function convergence(α; levels = 5)
    Random.seed!(20260903)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    Ωₕ = mesh(Ω, (9, 9), (false, false))
    hs, errs = Float64[], Float64[]
    for level in 1:levels
        uₕ, uₑ = solve_case(Ωₕ, α)
        push!(hs, hₘₐₓ(Ωₕ))
        push!(errs, norm₁ₕ(uₕ .- uₑ))
        level < levels && iterative_refinement!(Ωₕ)
    end
    return hs, errs
end

order(hs, errs) = log(errs[end - 1] / errs[end]) / log(hs[end - 1] / hs[end])

hs_mixed, errs_mixed = convergence((2.0, 0.0, 2.0, 0.0))
hs_robin, errs_robin = convergence((2.0, 1.0, 2.0, 1.0))
hs_neumann, errs_neumann = convergence((0.0, 0.0, 0.0, 0.0))

order_mixed = order(hs_mixed, errs_mixed)
order_robin = order(hs_robin, errs_robin)
order_neumann = order(hs_neumann, errs_neumann)
(order_mixed, order_robin, order_neumann)

#-

order_mixed > 1.8 && order_robin > 1.8 && order_neumann > 1.8

# The line above renders its verdict; these assert it. Bracketed above as well as below:    #src
# an order far past the promise means the finest error has reached roundoff and the ratio   #src
# no longer measures the scheme, so that should fail rather than read as a better result.   #src
@test 1.8 < order_mixed < 3.0                                                               #src
@test 1.8 < order_robin < 3.0                                                               #src
@test 1.8 < order_neumann < 3.0                                                             #src
# An exactly zero error would be the manufactured solution reproduced by construction.      #src
@test all(e -> 1.0e-6 < e < 1.0, errs_mixed)                                                #src
@test all(e -> 1.0e-6 < e < 1.0, errs_robin)                                                #src
@test all(e -> 1.0e-6 < e < 1.0, errs_neumann)                                              #src
# The pure Neumann solution has discrete mean zero, the constraint `pde_solve` was given.   #src
@test let Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 9), (true, true))  #src
    uₕ, _ = solve_case(Ωₕ, (0.0, 0.0, 0.0, 0.0))                                            #src
    w = assemble(form(gridspace(Ωₕ), v -> innerₕ(x -> 1.0, v)))                             #src
    abs(dot(w, uₕ)) < 1.0e-10                                                               #src
end                                                                                         #src

# ## Second order
#
# The error against ``h_{\max}`` on log-log axes, one curve per choice of ``\alpha``. The
# dashed line has slope 2 and is anchored at the finest point of the first curve.

include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot(
    [(hs_mixed, errs_mixed, "Robin and Neumann", "#5B5FC7"),
        (hs_robin, errs_robin, "Robin", "#0E7C86"),
        (hs_neumann, errs_neumann, "pure Neumann", "#B26A00")];
    title = "Neumann and Robin Poisson, ‖·‖₁ₕ") # hide

# All three curves run parallel to the slope-2 line on a grid built not to flatter the
# scheme: the points are random, so no symmetry cancels the one-sided error of the ghost
# layer. The Neumann and Robin relations are as accurate as the interior operator.
#
# ## Where to go next
#
#   - [Linear Poisson](poisson_linear.md) for the same problem with a Dirichlet condition.
#   - The [forms tutorial](@ref tutorial_form) for the pieces this page assembles.
