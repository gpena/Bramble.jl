# # Graded meshes for a boundary layer
#
# Bramble's schemes are built for non-uniform grids, and this is the problem that makes that
# worth having: a solution with a thin layer, where the points a uniform mesh spends in the
# smooth region are wasted and the ones it spends in the layer are too few.
#
# This page is generated from `docs/src/examples/boundary_layer_graded.jl` by Literate.jl,
# and the same file runs under `test/examples/pages.jl`, where the `#src` assertions execute.
#
# ## Problem
#
# ```math
# -\varepsilon u'' + u' = 1 \text{ in } (0,1), \qquad u(0) = u(1) = 0,
# ```
#
# with ``\varepsilon = 10^{-3}``. The exact solution
#
# ```math
# u_{\text{exact}}(x) = x - \frac{e^{(x-1)/\varepsilon} - e^{-1/\varepsilon}}
#                                {1 - e^{-1/\varepsilon}}
# ```
#
# rises like `x` across the domain and then drops to zero inside a layer of width
# ``\mathcal{O}(\varepsilon)`` at `x = 1`. Written this way rather than with
# ``e^{1/\varepsilon}`` in a denominator, which overflows at this ``\varepsilon``.

using Bramble

const ε = 1.0e-3

uexact(x) = x[1] - (exp((x[1] - 1) / ε) - exp(-1 / ε)) / (1 - exp(-1 / ε))

Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)

# The convective term is `inner₊(Mₕ(u), ∇ₕ(v))`, the same staggered discretisation the
# [convection-diffusion example](convection_diffusion_linear.md) uses. It is the discrete
# counterpart of ``\int u v'``, which integration by parts turns into ``-\int u' v``, so a
# convective term `+u'` in the equation enters the form with a minus sign.

function solve_on(pts)
    Ωₕ = mesh(Ω, length(pts))
    change_points!(Ωₕ, collect(pts))
    Wₕ = gridspace(Ωₕ)

    a = form(Wₕ, Wₕ, (u, v) -> ε * inner₊(∇ₕ(u), ∇ₕ(v)) - inner₊(Mₕ(u), ∇ₕ(v)))
    l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, x -> 1.0), v))

    A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0)
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

(err_uniform, over_uniform), (err_graded, over_graded)

# Three orders of magnitude in the error, and the overshoot tells the same story more
# physically: the uniform solution rises to nearly 1.9 where the exact solution peaks just
# below 1, because a cell wider than the layer cannot represent the drop and oscillates
# instead. The graded solution overshoots by less than a thousandth.
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

collect(zip(ns, uniform_errors, graded_errors))

# The graded mesh at 41 points is already better than the uniform one at 641. That is the
# whole argument for a non-uniform Cartesian grid: the points go where the solution varies,
# and the scheme keeps its order on them.

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

@test argmin(errors_by_σ) > 1                                                              #src
@test minimum(errors_by_σ) < 1.0e-3                                                        #src

collect(zip(σs, errors_by_σ))

# The best value here is around 4, where the finest cells are comparable with `ε`, and the
# curve is flat enough on either side that the choice is not delicate.
# There is nothing sacred about `tanh`: a geometric or piecewise-linear grading does the same
# job, and the mesh only ever sees the point vector.
#
# ## See also
#
# - [Convection-diffusion](convection_diffusion_linear.md), the same operator on a smooth
#   solution, where a uniform mesh is fine.
# - The [mesh tutorial](../tutorials/mesh.md) covers [`change_points!`](@ref) and
#   `Bramble.set_points!`, which also accepts a different point count.
