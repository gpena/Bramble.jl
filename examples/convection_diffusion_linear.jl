# # Convection-diffusion equation
#
# One more linear operator, built the same way as the [Poisson example](poisson_linear.md):
# diffusion plus a constant convective term, in 1D, 2D and 3D. Every number and every plot
# below was produced by the code shown.
#
# ## Problem
#
# ```math
# -\nabla \cdot (\epsilon \nabla u) - b \, \mathbf{1} \cdot \nabla u = g \text{ in } \Omega,
# \qquad u = u_{\text{exact}} \text{ on the far sides}, \qquad
# \partial_n u = h \text{ on the outflow sides}, \qquad \Omega = (0,1)^D
# ```
#
# with a constant diffusion coefficient ``\epsilon`` and a constant convective speed ``b`` along
# every coordinate direction at once: the vector field ``-b(1,\dots,1)``. That field points
# towards the sides ``x_i = 0``, so the fluid leaves through `xmin`, `ymin` and `zmin`, where
# the natural condition is a given flux ``\partial_n u = h``. The other sides, where the fluid
# enters, keep the Dirichlet condition ``u = u_{\text{exact}}``. The manufactured solution is
# the same as the other two examples, ``u_{\text{exact}}(x) = e^{\sum_i x_i}``, so
# ``g = -D\,(b+\epsilon)\,u_{\text{exact}}``, and on an outflow side the outward normal is
# ``-e_i``, so ``h = -u_{\text{exact}}`` is exact.
#
# ## Adding the convective term

using Bramble
using Random

ϵ, b = 1.0, 0.1
sol(x) = exp(sum(x))
rhs(x) = -length(x) * sol(x) * (b + ϵ)

Random.seed!(20260903)
Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ = mesh(Ω, (24, 24), (false, false))
Wₕ = gridspace(Ωₕ)

# The seed is what makes the numbers below reproducible: `(false, false)` draws the interior
# points from the global RNG, so without it the mesh -- and every figure on this page --
# would differ from build to build, and the suite could not assert what the page prints.
#
# The outflow sides carry the Neumann relation through ghost points:
# `gridspace(Ωₕ; ghosts = (:xmin, :ymin))` adds one layer of ghosts outside those two sides
# and `constrained` fixes each ghost from the relation `∂ₙ(u) = h`, where `h` is a function of
# the boundary point. The far sides stay Dirichlet and are handed to `assemble` as
# `side => data`. The convective term is now the strong operator `∇̽ₕ`, the second-order
# centred gradient on a non-uniform grid, summed over the directions, because a ghost
# value enters a difference stencil and not the staggered average of the weak form. In 1D
# `∇̽ₕ` is a bare node rather than a one-element tuple, so the sum is only taken for `D > 1`:

outflow_flux(x) = -sol(x)

convection(u, ::Val{1}) = ∇̽ₕ(u)
convection(u, ::Val) = sum(∇̽ₕ(u))

function convdiff_system(Ωₕ, D::Int)
    inflow = (:xmax, :ymax, :zmax)[1:D]
    outflow = (:xmin, :ymin, :zmin)[1:D]
    Cₕ = constrained(gridspace(Ωₕ; ghosts = outflow),
        u -> map(side -> side => ∂ₙ(u) => outflow_flux, outflow))
    a = form(Cₕ, Cₕ,
        (u, v) -> innerₕ(-ϵ * Δₕ(u) - b * convection(u, Val(D)), v))
    l = form(Cₕ, v -> innerₕ(rhs, v))
    A, F = assemble(a, l; dirichlet = map(side -> side => sol, inflow))
    return Cₕ, A, F
end

Cₕ, A, F = convdiff_system(Ωₕ, 2)

uₕ = element(Wₕ)
uₕ .= A \ F
nothing # hide

@test 1.0e-4 < norm₁ₕ(uₕ .- Rₕ(Wₕ, sol)) < 1.0e-1                                           #src

# The Dirichlet nodes hold the data: row replacement leaves them exactly at `sol`.          #src
@test let X = Rₕ(Wₕ, sol), (nx, ny) = Bramble.npoints(Ωₕ, Tuple)                           #src
    U = reshape(parent(uₕ), nx, ny)                                                         #src
    V = reshape(parent(X), nx, ny)                                                          #src
    U[end, :] == V[end, :] && U[:, end] == V[:, end]                                        #src
end                                                                                         #src

# ## The Neumann relation on the lifted solution
#
# `Bramble.lift(Cₕ, u)` returns the solution on the padded grid, with the ghost values the
# relations fix filled in. The relation holds at the ghosts by construction, so the check
# that means something is one-sided and independent of them: on the physical nodes of an
# outflow side, the three-point forward difference of the lifted solution along the normal
# axis, with the non-uniform spacings, negated for the outward normal, must equal ``h`` up
# to the truncation error, of order ``h^2``. `flux_residual` returns the largest residual
# over the nodes of one side, `axis` being 1 for `xmin` and 2 for `ymin`:

function flux_residual(Ωₕ, Cₕ, u, axis)
    U = parent(Bramble.lift(Cₕ, u))
    G = Bramble.mesh(Cₕ)
    n = Bramble.npoints(Ωₕ, Tuple)
    xs = ntuple(d -> collect(points(Ωₕ(d))), 2)
    h₁, h₂ = xs[axis][2] - xs[axis][1], xs[axis][3] - xs[axis][2]
    lin = LinearIndices(Bramble.npoints(G, Tuple))
    at(i, j) = U[lin[Bramble.padded_index(G, CartesianIndex(i, j))]]
    along(k, t) = axis == 1 ? at(k, t) : at(t, k)
    return maximum(1:n[3 - axis]) do t
        dₓ = -(2h₁ + h₂) / (h₁ * (h₁ + h₂)) * along(1, t) +
             (h₁ + h₂) / (h₁ * h₂) * along(2, t) - h₁ / (h₂ * (h₁ + h₂)) * along(3, t)
        point = axis == 1 ? (xs[1][1], xs[2][t]) : (xs[1][t], xs[2][1])
        abs(-dₓ - outflow_flux(point))
    end
end

residuals = [flux_residual(Ωₕ, Cₕ, A \ F, axis) for axis in (1, 2)]

# This is the truncation error of the one-sided difference plus the error of the solution
# itself, so it is small but not round-off. It falls as ``h^2``: the same random mesh at 12
# points per side, then split once, gives

coarse_to_fine = let
    Random.seed!(20260903)
    Ωc = mesh(Ω, (12, 12), (false, false))
    out = Float64[]
    for level in 1:2
        Cc, Ac, Fc = convdiff_system(Ωc, 2)
        push!(out, flux_residual(Ωc, Cc, Ac \ Fc, 1))
        level < 2 && iterative_refinement!(Ωc)
    end
    out
end

#-

coarse_to_fine[2] < coarse_to_fine[1]

# The residual is the one-sided truncation error, not a round-off level: it is bracketed     #src
# away from zero, and it shrinks under refinement.                                           #src
@test all(r -> 1.0e-7 < r < 1.0e-1, residuals)                                               #src
@test 1.0e-7 < coarse_to_fine[2] && coarse_to_fine[1] / coarse_to_fine[2] > 3               #src

# ## Péclet number and flow direction
#
# A different problem from the one just solved, not a plot of `uₕ` above: constant-coefficient
# convection-diffusion again, but on a domain-independent, boundary-driven flow rather than a
# manufactured solution, so the field itself moves as the parameters change instead of always
# resolving to the same prescribed answer.
#
# ```math
# -\nabla \cdot (\epsilon \nabla u) + \mathbf{\beta} \cdot \nabla u = 0 \text{ in } \Omega = (0,1)^2,
# \qquad u = 1 \text{ on } \{x = 0\}, \qquad u = 0 \text{ on the rest of } \partial\Omega
# ```
#
# with ``\epsilon = 1`` fixed and ``\mathbf{\beta} = \text{Pe}``, so the
# Péclet slider sets the advection speed directly. An interactive panel, seeded only from the
# resolution of `uₕ` above but solving this boundary-driven problem itself in the browser: drag
# the Péclet number slider past the point where the cell Péclet number `Pe_h` crosses `1` with
# the centered stencil selected, and the classic grid-scale oscillation appears. Switch to
# upwind to see it damp back out, at the cost of accuracy.

include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide
convection_diffusion_interactive_widget(uₕ; title = "Convection-diffusion, 2D") # hide

# ## Convergence with a convective term
#
# The same pattern as the other two examples: one random coarse mesh per dimension, refined
# in place with [`iterative_refinement!`](@ref). Every dimension has the same mixed problem,
# Neumann on the sides `xmin`, `ymin`, `zmin` that exist and Dirichlet on the opposite ones:

function convdiff_series(D::Int; n0::Int = 5, levels::Int)
    Ωd = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D)))
    Ωc = mesh(Ωd, ntuple(_ -> n0, D), ntuple(_ -> false, D))

    hs, errs = Float64[], Float64[]
    for level in 1:levels
        Wc = gridspace(Ωc)
        _, A_c, F_c = convdiff_system(Ωc, D)
        u_c = element(Wc)
        u_c .= A_c \ F_c

        push!(hs, hₘₐₓ(Ωc))
        push!(errs, norm₁ₕ(u_c .- Rₕ(Wc, sol)))
        level < levels && iterative_refinement!(Ωc)
    end
    return hs, errs
end

Random.seed!(20260903) # hide
hs1, errs1 = convdiff_series(1; n0 = 6, levels = 7) # hide
Random.seed!(20260903) # hide
hs2, errs2 = convdiff_series(2; levels = 6) # hide
Random.seed!(20260903) # hide
hs3, errs3 = convdiff_series(3; levels = 4) # hide

order1 = log(errs1[end - 1] / errs1[end]) / log(hs1[end - 1] / hs1[end]) # hide
order2 = log(errs2[end - 1] / errs2[end]) / log(hs2[end - 1] / hs2[end]) # hide
order3 = log(errs3[end - 1] / errs3[end]) / log(hs3[end - 1] / hs3[end]) # hide
nothing # hide

# Bracketed above as well as below, for the reason poisson_linear.jl gives.                 #src
@test 1.9 < order1 < 3.0                                                                    #src
@test 1.9 < order2 < 3.0                                                                    #src
@test 1.8 < order3 < 3.0                                                                    #src

#-

include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot([(hs1, errs1, "1D", "#5B5FC7"), (hs2, errs2, "2D", "#0E7C86"), (hs3, errs3, "3D", "#B26A00")]; # hide
    title = "Convection-diffusion, ‖·‖₁ₕ") # hide

# Second order in every dimension. Neither the Neumann relation nor the convective term
# changes the rate. The convective term makes the matrix non-symmetric (the first
# difference `∇̽ₕ` is antisymmetric to leading order, not self-adjoint), which is why this
# example does not also check `issymmetric`, unlike the
# [forms tutorial](@ref tutorial_form)'s pure-diffusion Poisson problem.
#
# ## Where to go next
#
#   - [Linear Poisson](poisson_linear.md) for the pure-diffusion case this one extends.
