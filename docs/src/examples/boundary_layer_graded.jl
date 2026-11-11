# # Graded meshes for a boundary layer
#
# Bramble's schemes are built for non-uniform grids, and this is the problem that makes that
# worth having: a solution with a thin layer, where the points a uniform mesh spends in the
# smooth region are wasted and the ones it spends in the layer are too few.
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
# ``\mathcal{O}(\varepsilon)`` at `x = 1`. The formula keeps every exponential negative,
# since ``e^{1/\varepsilon}`` overflows at this ``\varepsilon``. The solution vanishes at
# `x = 0` and has ``u'(0) = 1 - e^{-1/\varepsilon} / (\varepsilon (1 - e^{-1/\varepsilon}))``,
# which is `1` to machine precision, so the Robin datum is
# ``h = u_{\text{exact}}(0) - u_{\text{exact}}'(0)``, computed from the exact solution.

using Bramble
using Bramble: change_points!

const ε = 1.0e-3

uexact(x) = x[1] - (exp((x[1] - 1) / ε) - exp(-1 / ε)) / (1 - exp(-1 / ε))
duexact(x) = 1 - exp((x[1] - 1) / ε) / (ε * (1 - exp(-1 / ε)))
robin_datum(x) = uexact(x) - duexact(x)

@test isinf(exp(1 / ε))                                                                    #src
@test uexact(0.0) == 0.0                                                                   #src
@test duexact(0.0) == 1.0                                                                  #src
@test robin_datum(0.0) == -1.0                                                             #src

Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)

# The Robin end goes through ghost points: `gridspace(Ωₕ; ghosts = (:xmin,))` adds one ghost
# node outside `x = 0` only, and `constrained` fixes it from the relation `u + ∂ₙ(u) = h`,
# with `h` a function of the boundary point. The Dirichlet end needs no ghost and is handed
# to `assemble` as `side => data`. The diffusion term is `Δₕ` and the convective term the
# cross-weighted centred gradient `∇̽ₕ`, so the form is
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
# Bramble ships no stretching helper. The `uniform = false` flag draws interior points at
# random and sorts them, which is no grading, and [`iterative_refinement!`](@ref) halves
# every cell, everywhere.
#
# A graded mesh is a point vector, written by hand and installed with
# [`change_points!`](@ref). A `tanh` map clusters points towards `x = 1`, with `σ` setting how
# hard:

uniform_points(n) = collect(range(0.0, 1.0, length = n))
graded_points(n; σ = 4.0) = [tanh(σ * t) / tanh(σ) for t in range(0.0, 1.0, length = n)]

g = graded_points(41)
extrema(diff(g)), extrema(diff(uniform_points(41)))

@test argmin(diff(g)) == length(g) - 1                                                     #src
@test 160 < 0.025 / minimum(diff(g)) < 175                                                 #src
@test 3.9 < maximum(diff(g)) / 0.025 < 4.1                                                 #src

# On the same 41 points the finest cell is about 170 times smaller than the uniform one and
# the coarsest about four times larger. The finest cell, `1.5e-4`, is below the layer width
# `ε = 10⁻³`; the uniform cell, `0.025`, is 25 layer widths.
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
@test 0.99 < maximum(uexact.(range(0.0, 1.0, length = 100_001))) < 1.0                     #src

(err_uniform, over_uniform), (err_graded, over_graded)

# The graded error is more than two orders of magnitude smaller. The overshoot, the amount by
# which the discrete maximum exceeds the exact peak (just below 1), shows it physically: the
# uniform solution overshoots by 0.87, the graded one by about a thousandth.
#
# The overshoot is the oscillation of a centred scheme on cells wider than the layer. With
# `h = 1/40` and `ε = 10⁻³` the cell Péclet number `h / (2ε)` is 12.5 on the uniform mesh,
# well above the value 1 at which centred differences start to oscillate.

@test 12 < (1 / 40) / (2ε) < 13                                                            #src

#
# ## How far refinement gets you
#
# Refining the uniform mesh does reduce the error:

ns = (41, 81, 161, 321, 641)
uniform_errors = [solve_on(uniform_points(n))[1] for n in ns]
graded_errors = [solve_on(graded_points(n))[1] for n in ns]

@test all(graded_errors .< uniform_errors)                                                 #src
@test graded_errors[1] < uniform_errors[end]                                               #src

nothing # hide

# The graded mesh at 41 points is already better than the uniform one at 641. The points go
# where the solution varies, and the scheme keeps its order on them: doubling the number of
# intervals divides the graded error by four, which is second order.

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
# order ``h^2``. `robin_residual` returns the residual at that node. It indexes the padded
# grid with `Bramble.padded_index`, an internal function:

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
# error to speak of and the residual is at round-off from 81 points on.

#-

include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot( # hide
    [(1.0 ./ collect(ns), uniform_errors, "uniform", "#B26A00"), # hide
        (1.0 ./ collect(ns), graded_errors, "graded (σ = 4)", "#0E7C86")]; # hide
    title = "Boundary layer at ε = 10⁻³, ‖·‖ₕ against 1/N") # hide

# ## Choosing σ
#
# Grading is a parameter, and both extremes are bad:

σs = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
errors_by_σ = [solve_on(graded_points(41; σ = σ))[1] for σ in σs]

best_σ = σs[argmin(errors_by_σ)]

@test best_σ in (3.0, 4.0, 6.0)                                                            #src
@test minimum(errors_by_σ) < 1.0e-3                                                        #src
@test maximum(errors_by_σ[3:5]) < 3 * minimum(errors_by_σ)                                 #src
@test errors_by_σ[1] > 100 * minimum(errors_by_σ)                                         #src
@test errors_by_σ[end] > 100 * minimum(errors_by_σ)                                       #src

collect(zip(σs, errors_by_σ))

best_σ

# The error is `0.19` at `σ = 1` and `0.12` at `σ = 8`, against `9.2e-4` at `σ = 4`, where the
# finest cell is below `ε`. Between `σ = 3` and `σ = 6` it stays within a factor of three of
# the best, so the choice is not delicate. Any point vector can be installed the same way;
# the mesh only sees the vector.
#
# ## Where to go next
#
# - [Convection-diffusion](convection_diffusion_linear.md), diffusion plus a convective term
#   on a smooth manufactured solution, in 1D, 2D and 3D.
# - The [mesh tutorial](@ref tutorial_mesh) covers [`change_points!`](@ref) and
#   `Bramble.set_points!`, which also accepts a different point count.
