# # Graded meshes for a boundary layer
#
# Bramble's schemes are built for non-uniform grids, and this is the problem that makes that
# worth having: a solution with a thin layer, where the points a uniform mesh spends in the
# smooth region are wasted and the ones it spends in the layer are too few.
#
# This script runs under `test/examples/pages.jl`, where the `#src` assertions execute; it is
# not part of the published manual.
#
# ## Problem
#
# ```math
# -\varepsilon u'' + u' = 1 \text{ in } (0,1), \qquad
# u(0) - u'(0) = h, \quad u(1) = 0,
# ```
#
# with ``\varepsilon = 10^{-3}``: a Robin condition at `x = 0`, written with the outward
# normal derivative ``\partial_n u = -u'`` as ``u + \partial_n u = h``, and a Dirichlet
# condition at the layer end `x = 1`. The exact solution
#
# ```math
# u_{\text{exact}}(x) = x - \frac{e^{(x-1)/\varepsilon} - e^{-1/\varepsilon}}
#                                {1 - e^{-1/\varepsilon}}
# ```
#
# rises like `x` across the domain and then drops to zero inside a layer of width
# ``\mathcal{O}(\varepsilon)`` at `x = 1`. Written this way rather than with
# ``e^{1/\varepsilon}`` in a denominator, which overflows at this ``\varepsilon``. It
# vanishes at `x = 0` and has ``u'(0) = 1 - e^{-1/\varepsilon} / (\varepsilon (1 -
# e^{-1/\varepsilon}))``, which is `1` to machine precision, so the Robin datum is
# ``h = u_{\text{exact}}(0) - u_{\text{exact}}'(0)``, computed from the exact solution.

using Bramble
using Bramble: change_points!

const ε = 1.0e-3

uexact(x) = x[1] - (exp((x[1] - 1) / ε) - exp(-1 / ε)) / (1 - exp(-1 / ε))
duexact(x) = 1 - exp((x[1] - 1) / ε) / (ε * (1 - exp(-1 / ε)))
robin_datum(x) = uexact(x) - duexact(x)

Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)

# The Robin end goes through ghost points: `gridspace(Ωₕ; ghosts = (:xmin,))` adds one ghost
# node outside `x = 0` only, and `constrained` fixes it from the relation `u + ∂ₙ(u) = h`,
# with `h` a function of the boundary point. The Dirichlet end needs no ghost and is handed
# to `assemble` as `side => data`. The convective term is the strong operator `∇̽ₕ`, the
# second-order centred derivative on a non-uniform grid, because a ghost value enters a
# difference stencil and not the staggered average of the weak form. The form is
# ``\langle -\varepsilon \Delta_h u + \breve\nabla_h u, v \rangle_h``, with the source `1`
# tested against `v` in the same inner product.

function system_on(pts)
    Ωₕ = mesh(Ω, length(pts))
    change_points!(Ωₕ, collect(pts))
    Cₕ = constrained(gridspace(Ωₕ; ghosts = (:xmin,)),
        u -> (:xmin => u + ∂ₙ(u) => robin_datum,))
    a = form(Cₕ, Cₕ, (u, v) -> innerₕ(-ε * Δₕ(u) + ∇̽ₕ(u), v))
    l = form(Cₕ, v -> innerₕ(x -> 1.0, v))
    A, F = assemble(a, l; dirichlet = :xmax => uexact)
    return Ωₕ, Cₕ, A, F
end

function solve_on(pts)
    Ωₕ, Cₕ, A, F = system_on(pts)
    Wₕ = gridspace(Ωₕ)
    uₕ = element(Wₕ)
    uₕ .= A \ F

    exact = Rₕ(Wₕ, uexact)
    overshoot = maximum(parent(uₕ)) - maximum(parent(exact))

    return normₕ(uₕ - exact), overshoot
end

# ## Building a graded mesh
#
# Bramble ships no stretching helper, and the `uniform = false` flag is not one: it draws
# interior points at random and sorts them, which is what the convergence studies elsewhere
# use to expose order reduction, not a grading. [`iterative_refinement!`](@ref) is not one
# either: it halves every cell, everywhere.
#
# A graded mesh is a point vector, written by hand and installed with
# [`change_points!`](@ref). A `tanh` map clusters points towards `x = 1`, with `σ` setting how
# hard:

uniform_points(n) = collect(range(0.0, 1.0, length = n))
graded_points(n; σ = 4.0) = [tanh(σ * t) / tanh(σ) for t in range(0.0, 1.0, length = n)]

g = graded_points(41)
extrema(diff(g)), extrema(diff(uniform_points(41)))

# The finest cell is about 170 times smaller than the uniform one and the coarsest about four
# times larger, on the same 41 points. The layer is `ε = 10⁻³` wide, so the fine end is where
# it has to be.
#
# ## The comparison
#
# At equal degrees of freedom:

err_uniform, over_uniform = solve_on(uniform_points(41))
err_graded, over_graded = solve_on(graded_points(41))

@test 0.1 < err_uniform < 1.0                                                              #src
@test 1.0e-6 < err_graded < 1.0e-2                                                          #src
@test err_graded < err_uniform / 100                                                       #src
@test 0.8 < over_uniform < 0.95                                                            #src
@test 0.0 < over_graded < 2.0e-3                                                           #src

(err_uniform, over_uniform), (err_graded, over_graded)

# More than two orders of magnitude in the error, and the overshoot tells the same story more
# physically: the uniform solution rises to nearly 1.9 where the exact solution peaks just
# below 1, because a cell wider than the layer cannot represent the drop and oscillates
# instead. The graded solution overshoots by about a thousandth.
#
# The oscillation is not a solver problem and no amount of tightening a tolerance removes it.
# The cell Péclet number `h / (2ε)` is what decides it, and on a uniform mesh with
# `h = 1/40` at `ε = 10⁻³` that number is twelve.
#
# ## How far refinement gets you
#
# Refining the uniform mesh does work, eventually:

ns = (41, 81, 161, 321, 641)
uniform_errors = [solve_on(uniform_points(n))[1] for n in ns]
graded_errors = [solve_on(graded_points(n))[1] for n in ns]

@test all(graded_errors .< uniform_errors)                                                 #src
@test graded_errors[1] < uniform_errors[end]                                               #src

nothing # hide

# The graded mesh at 41 points is already better than the uniform one at 641. That is the
# whole argument for a non-uniform Cartesian grid: the points go where the solution varies,
# and the scheme keeps its order on them: halving the spacing divides the graded error by
# four, which is second order.

observed_orders = log2.(graded_errors[1:(end - 1)] ./ graded_errors[2:end])

@test all(1.9 .< observed_orders .< 2.1)                                                   #src

observed_orders

# ## The Robin relation on the lifted solution
#
# `Bramble.lift(Cₕ, u)` returns the solution on the padded grid, with the ghost value the
# relation fixes filled in. The relation holds at the ghost by construction, so the check
# that means something is one-sided and independent of it: at `x = 0`, the three-point
# forward difference of the lifted solution, with the non-uniform spacings and negated for
# the outward normal, must satisfy ``u + \partial_n u = h`` up to the truncation error, of
# order ``h^2``. `robin_residual` returns the residual at that node:

function robin_residual(pts)
    Ωₕ, Cₕ, A, F = system_on(pts)
    U = parent(Bramble.lift(Cₕ, A \ F))
    G = Bramble.mesh(Cₕ)
    xs = collect(pts)
    h₁, h₂ = xs[2] - xs[1], xs[3] - xs[2]
    at(i) = U[Bramble.padded_index(G, CartesianIndex(i))]
    dₓ = -(2h₁ + h₂) / (h₁ * (h₁ + h₂)) * at(1) + (h₁ + h₂) / (h₁ * h₂) * at(2) -
         h₁ / (h₂ * (h₁ + h₂)) * at(3)
    return abs(at(1) - dₓ - robin_datum(xs[1]))
end

robin_residuals = [robin_residual(graded_points(n)) for n in ns]

@test all(robin_residuals .< 1.0e-5)                                                       #src
@test all(robin_residuals[2:end] .< 1.0e-12)                                               #src
@test robin_residuals[end] < robin_residuals[1] / 4                                        #src

robin_residuals

# The exact solution is linear near `x = 0`, so the one-sided difference has no truncation
# error to speak of and the residual reaches round-off after the first mesh: the relation
# holds to the accuracy of the solve, not only to the order of the scheme.

#-

include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot( # hide
    [(1.0 ./ collect(ns), uniform_errors, "uniform", "#B26A00"), # hide
        (1.0 ./ collect(ns), graded_errors, "graded (σ = 4)", "#0E7C86")]; # hide
    title = "Boundary layer at ε = 10⁻³, ‖·‖ₕ against 1/N") # hide

# ## Choosing σ
#
# Grading is a parameter, and both extremes are bad: too little leaves the layer unresolved,
# too much starves the smooth region and the coarse cells there start to oscillate on their
# own account.

σs = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
errors_by_σ = [solve_on(graded_points(41; σ = σ))[1] for σ in σs]

best_σ = σs[argmin(errors_by_σ)]

@test best_σ in (3.0, 4.0, 6.0)                                                            #src
@test minimum(errors_by_σ) < 1.0e-3                                                        #src

collect(zip(σs, errors_by_σ))

best_σ

# The best value here is around 4, where the finest cells are comparable with `ε`, and the
# curve is flat enough on either side that the choice is not delicate.
# There is nothing sacred about `tanh`: a geometric or piecewise-linear grading does the same
# job, and the mesh only ever sees the point vector.
#
# ## Where to go next
#
# - [Convection-diffusion](convection_diffusion_linear.md), the same operator on a smooth
#   solution, where a uniform mesh is fine.
# - The [mesh tutorial](@ref tutorial_mesh) covers [`change_points!`](@ref) and
#   `Bramble.set_points!`, which also accepts a different point count.
