# # Linear Poisson equation
#
# A worked problem, built from pieces the [forms tutorial](@ref tutorial_form) introduces one
# at a time. Every number and every plot below was produced by the code shown.
#
# ## Problem
#
# ```math
# -\Delta u = g \text{ in } \Omega = (0,1)^2, \qquad
# u = u_{\text{exact}} \text{ on } \texttt{xmin} \text{ and } \texttt{ymin}, \qquad
# u + \partial_n u = h \text{ on } \texttt{xmax} \text{ and } \texttt{ymax},
# ```
#
# with ``\partial_n`` the outward normal derivative: a Dirichlet condition on the two sides
# at the origin and a Robin condition on the two others. The manufactured solution is
# ``u_{\text{exact}}(x, y) = e^{x+y}``, so ``g = -2 u_{\text{exact}}``, and the Robin data
# ``h = u_{\text{exact}} + \partial_n u_{\text{exact}} = 2 u_{\text{exact}}`` there, since
# the outward normal derivative of ``e^{x+y}`` on `xmax` and `ymax` is ``u_{\text{exact}}``
# itself. A manufactured solution is what makes the error checkable at all. Without one
# there is nothing to compare the discrete solution against.
#
# ## Ghost points on the Robin sides
#
# The Robin sides go through ghost points: `gridspace(Ωₕ; ghosts = (:xmax, :ymax))` adds one
# layer outside those two sides only. `constrained` takes a function of the trial function
# `u` and returns one `side => relation => data` entry per side, here `u + ∂ₙ(u)` with the
# data ``h`` as a function of the boundary point. Each relation fixes a ghost value from the
# interior ones, which leaves the physical nodes as the only unknowns.
#
# The Dirichlet sides need no ghosts. There are two ways to impose them. The `dirichlet`
# keyword of `assemble` replaces the rows of those nodes, and `symmetrize = true` moves the
# known values to the right-hand side so the matrix stays symmetric. The alternative is
# the relation `u => g` in `constrained`, which removes the nodes from the unknowns. The
# page solves with the keyword and checks below that the relation gives the same solution.
#
# ## Assembling and solving on a random grid

using Bramble
using Random

sol(x) = exp(x[1] + x[2])
rhs(x) = -2 * sol(x)

Random.seed!(20260903)
Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ = mesh(Ω, (24, 24), (false, false))
Wₕ = gridspace(Ωₕ)

# `(false, false)` builds a non-uniform, randomly-perturbed grid in both directions,
# deliberately, not the default: a uniform grid happens to make this particular manufactured
# solution nearly exact for this scheme regardless of mesh size, which would make the
# convergence check below pass on a broken implementation just as readily as a correct one.
#
# The seed is what makes the numbers below reproducible: `(false, false)` draws the interior
# points from the global RNG, so without it the mesh -- and every figure on this page --
# would differ from build to build, and the suite could not assert what the page prints.
#
# The bilinear form is the discrete Laplacian, written ``\langle -\Delta_h u, v \rangle_h``
# because it reads the ghost values the relations fix; the
# [forms tutorial](@ref form_dirichlet) builds the same operator in one dimension as
# `inner₊(∇ₕ(u), ∇ₕ(v))`. The source is the right-hand side sampled at the grid points and
# tested against `v`, `innerₕ(rhs, v)`:

robin_data(x) = 2 * sol(x)

function poisson_system(Ωₕ)
    Cₕ = constrained(gridspace(Ωₕ; ghosts = (:xmax, :ymax)),
        u -> (:xmax => u + ∂ₙ(u) => robin_data, :ymax => u + ∂ₙ(u) => robin_data))
    a = form(Cₕ, Cₕ, (u, v) -> innerₕ(-1.0 * Δₕ(u), v))
    l = form(Cₕ, v -> innerₕ(rhs, v))
    A, F = assemble(a, l; dirichlet = (:xmin => sol, :ymin => sol), symmetrize = true)
    return Cₕ, A, F
end

Cₕ, A, F = poisson_system(Ωₕ)

uₕ = element(Wₕ)
uₕ .= pde_solve(A, F)
nothing # hide

# The error on this one mesh, before the rates below say anything about how it shrinks.    #src
# Bracketed away from zero as well: an exactly-zero error would mean the manufactured       #src
# solution had been reproduced by construction rather than solved for.                      #src
@test 1.0e-4 < norm₁ₕ(uₕ .- Rₕ(Wₕ, sol)) < 1.0e-1                                           #src
# The Dirichlet nodes hold the data: row replacement leaves them exactly at `sol`.          #src
@test let X = Rₕ(Wₕ, sol), (nx, ny) = Bramble.npoints(Ωₕ, Tuple)                           #src
    U = reshape(parent(uₕ), nx, ny)                                                         #src
    V = reshape(parent(X), nx, ny)                                                          #src
    U[1, :] == V[1, :] && U[:, 1] == V[:, 1]                                                #src
end                                                                                         #src

# ## The Robin relation on the lifted solution
#
# `Bramble.lift(Cₕ, u)` returns the solution on the padded grid, with the ghost values the
# relations fix filled in. The relation is satisfied by the ghosts, so the check that means
# something is one-sided and independent of them: on the physical nodes of `xmax`, the
# three-point backward difference of the lifted solution for the normal derivative, with the
# non-uniform spacings, must satisfy ``u + \partial_n u = h`` up to the truncation error, of
# order ``h^2``. `robin_residual` returns the largest residual over the nodes of the side:

function robin_residual(Ωₕ, Cₕ, u)
    U = parent(Bramble.lift(Cₕ, u))
    G = Bramble.mesh(Cₕ)
    nx, ny = Bramble.npoints(Ωₕ, Tuple)
    xs, ys = collect(points(Ωₕ(1))), collect(points(Ωₕ(2)))
    h₁, h₂ = xs[nx] - xs[nx - 1], xs[nx - 1] - xs[nx - 2]
    lin = LinearIndices(Bramble.npoints(G, Tuple))
    at(i, j) = U[lin[Bramble.padded_index(G, CartesianIndex(i, j))]]
    return maximum(1:ny) do j
        dₙ = (2h₁ + h₂) / (h₁ * (h₁ + h₂)) * at(nx, j) -
             (h₁ + h₂) / (h₁ * h₂) * at(nx - 1, j) + h₁ / (h₂ * (h₁ + h₂)) * at(nx - 2, j)
        abs(at(nx, j) + dₙ - robin_data((xs[nx], ys[j])))
    end
end

residual = robin_residual(Ωₕ, Cₕ, pde_solve(A, F))

# This is the truncation error of the one-sided difference, plus the error of the solution
# itself, so it is small but not round-off. It falls as ``h^2``: the same random mesh at 12
# points per side, then split once, gives

residuals = let
    Random.seed!(20260903)
    Ωc = mesh(Ω, (12, 12), (false, false))
    out = Float64[]
    for level in 1:2
        Cc, Ac, Fc = poisson_system(Ωc)
        push!(out, robin_residual(Ωc, Cc, pde_solve(Ac, Fc)))
        level < 2 && iterative_refinement!(Ωc)
    end
    out
end

#-

residuals[2] < residuals[1]

# The residual is the one-sided truncation error, not a round-off level: it is bracketed     #src
# away from zero, and it shrinks under refinement.                                           #src
@test 1.0e-7 < residual < 1.0e-1                                                             #src
@test 1.0e-7 < residuals[2] && residuals[1] / residuals[2] > 3   # about 4 for ``h^2``      #src
# The two ways of imposing the Dirichlet sides solve the same problem: `u => g` in           #src
# `constrained` and the `dirichlet` keyword give the same values at the free nodes.          #src
@test let (nx, ny) = Bramble.npoints(Ωₕ, Tuple)                                              #src
    Cr = constrained(gridspace(Ωₕ; ghosts = (:xmax, :ymax)),                                 #src
        u -> (:xmin => u => sol, :ymin => u => sol,                                          #src
            :xmax => u + ∂ₙ(u) => robin_data, :ymax => u + ∂ₙ(u) => robin_data))            #src
    ar = form(Cr, Cr, (u, v) -> innerₕ(-1.0 * Δₕ(u), v))                                     #src
    ur = pde_solve(assemble(ar, form(Cr, v -> innerₕ(rhs, v)))...)                          #src
    free = trues(nx, ny)                                                                     #src
    free[1, :] .= false                                                                      #src
    free[:, 1] .= false                                                                      #src
    length(ur) == count(free) && isapprox(parent(uₕ)[vec(free)], ur; rtol = 1.0e-8)         #src
end                                                                                          #src

# ## Browsing the solution
#
# An interactive panel, seeded from the resolution of `uₕ` above but solving its own problem
# in the browser: drag the resolution slider, switch between a uniform and a fully random
# mesh, or pick a different manufactured solution, and every plot and diagnostic below
# updates immediately.

include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide
poisson_interactive_widget(uₕ; title = "Linear Poisson, 2D") # hide

# ## Convergence in one, two and three dimensions
#
# The check is not that it ran but that it converges at the rate the scheme promises, on genuinely random
# grids, in 1D, 2D and 3D, with the same Dirichlet and Robin sides in each: one random coarse
# mesh per dimension, refined in place with
# [`iterative_refinement!`](@ref) so every finer level is the *same* random mesh, dyadically
# split, rather than a fresh independent draw that would add its own noise to the measured rate.

function poisson_series(D::Int; n0::Int = 5, levels::Int)
    sol_d(x) = exp(sum(x))
    rhs_d(x) = -D * sol_d(x)
    robin_d(x) = 2 * sol_d(x)
    lo = (:xmin, :ymin, :zmin)[1:D]    # Dirichlet sides
    hi = (:xmax, :ymax, :zmax)[1:D]    # Robin sides, through ghost points

    Ωd = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D)))
    Ωc = mesh(Ωd, ntuple(_ -> n0, D), ntuple(_ -> false, D))

    hs, errs = Float64[], Float64[]
    for level in 1:levels
        Cc = constrained(gridspace(Ωc; ghosts = hi),
            u -> ntuple(j -> hi[j] => u + ∂ₙ(u) => robin_d, D))
        a_c = form(Cc, Cc, (u, v) -> innerₕ(-1.0 * Δₕ(u), v))
        l_c = form(Cc, v -> innerₕ(rhs_d, v))
        A_c, F_c = assemble(a_c, l_c;
            dirichlet = ntuple(j -> lo[j] => sol_d, D), symmetrize = true)

        Wc = gridspace(Ωc)
        u_c = element(Wc)
        u_c .= pde_solve(A_c, F_c)
        uexact_c = Rₕ(Wc, sol_d)

        push!(hs, hₘₐₓ(Ωc))
        push!(errs, norm₁ₕ(u_c .- uexact_c))
        level < levels && iterative_refinement!(Ωc)
    end
    return hs, errs
end

Random.seed!(20260903)
hs1, errs1 = poisson_series(1; n0 = 6, levels = 7)  # 6 up to 321 points
Random.seed!(20260903)
hs2, errs2 = poisson_series(2; levels = 6)   # 5² up to 129² points
Random.seed!(20260903)
hs3, errs3 = poisson_series(3; levels = 4)   # 5³ up to 33³ points; 3D is expensive per level

order1 = log(errs1[end - 1] / errs1[end]) / log(hs1[end - 1] / hs1[end])
order2 = log(errs2[end - 1] / errs2[end]) / log(hs2[end - 1] / hs2[end])
order3 = log(errs3[end - 1] / errs3[end]) / log(hs3[end - 1] / hs3[end])
(order1, order2, order3)

#-

order1 > 1.9 && order2 > 1.9 && order3 > 1.8   # comfortably above 1st order; 2nd is the promise

# The line above renders its verdict; these assert it. Bracketed above as well as below:    #src
# an order far above the promise means the finest error has reached roundoff and the ratio  #src
# no longer measures the scheme, which should fail rather than read as an excellent result. #src
@test 1.9 < order1 < 3.0                                                                    #src
@test 1.9 < order2 < 3.0                                                                    #src
@test 1.8 < order3 < 3.0   # 3D runs fewer levels                                           #src

#-

include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot([(hs1, errs1, "1D", "#5B5FC7"), (hs2, errs2, "2D", "#0E7C86"), (hs3, errs3, "3D", "#B26A00")];
    title = "Linear Poisson, ‖·‖₁ₕ") # hide

# Second order in every dimension, with a Dirichlet condition on the sides at the origin and a
# Robin condition on the others, on grids chosen specifically not to make that trivially
# true: every curve's markers sit along the same reference slope. 3D uses fewer refinement
# levels than 2D, and 2D fewer than 1D: the same dyadic split costs ``8\times`` the points per
# level in 3D against ``4\times`` in 2D and ``2\times`` in 1D, so matching level counts across
# all three would make the higher dimensions by far the most expensive part of this page for no
# extra information.
#
# ## Where to go next
#
#   - [Convection-diffusion](convection_diffusion_linear.md) for a non-symmetric operator on
#     the same grids.
#   - The [forms tutorial](@ref tutorial_form) for the pieces this page assembles.
