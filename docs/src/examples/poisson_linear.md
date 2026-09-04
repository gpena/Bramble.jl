# Linear Poisson equation

A worked problem, built from pieces the [forms tutorial](../tutorials/form.md) introduces one
at a time. Every number and every plot below was produced by the code shown.

## Problem

```math
-\Delta u = g \text{ in } \Omega, \qquad u = u_{\text{exact}} \text{ on } \partial\Omega,
\qquad \Omega = (0,1)^2
```

with the manufactured solution ``u_{\text{exact}}(x, y) = e^{x+y}``, so ``g = -2 u_{\text{exact}}``.
A manufactured solution is what makes the error checkable at all — without one there is
nothing to compare the discrete solution against.

## Solving it

```@example poisson_linear
using Bramble

sol(x) = exp(x[1] + x[2])
rhs(x) = -2 * sol(x)

Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ = mesh(Ω, (24, 24), (false, false))
Wₕ = gridspace(Ωₕ)
```

`(false, false)` builds a non-uniform, randomly-perturbed grid in both directions —
deliberately, not the default: a uniform grid happens to make this particular manufactured
solution nearly exact for this scheme regardless of mesh size, which would make the
convergence check below pass on a broken implementation just as readily as a correct one.

The bilinear form is the discrete Laplacian, the same `inner₊(∇₋ₕ(u), ∇₋ₕ(v))` the
[forms tutorial](../tutorials/form.md#5.-Dirichlet-conditions,-and-a-Poisson-problem) builds
in one dimension:

```@example poisson_linear
bcs = dirichlet_constraints(Bramble.set(Ω), :boundary => sol)

a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
A = assemble(a; dirichlet_labels = :boundary)

gₕ = element(Wₕ)
avgₕ!(gₕ, rhs)
l = form(Wₕ, v -> innerₕ(gₕ, v))
F = assemble(l; dirichlet_conditions = bcs, dirichlet_labels = :boundary)

uₕ = element(Wₕ)
uₕ .= A \ F
nothing # hide
```

## Visualizing the solution

A heatmap of the field just solved for, `uₕ` from the block above — no new solve needed:

```@example poisson_linear
include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide
heatmap_plot(uₕ; title = "Linear Poisson, 2D")
```

## Checking the answer

Not just that it ran — that it converges at the rate the scheme promises, on genuinely random
grids, in 1D, 2D and 3D: one random coarse mesh per dimension, refined in place with
[`iterative_refinement!`](@ref) so every finer level is the *same* random mesh, dyadically
split, rather than a fresh independent draw that would add its own noise to the measured rate.

```@example poisson_linear
using Random

function poisson_series(D::Int; n0::Int = 5, levels::Int)
    sol_d(x) = exp(sum(x))
    rhs_d(x) = -D * sol_d(x)

    Ωd = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D)))
    Ωc = mesh(Ωd, ntuple(_ -> n0, D), ntuple(_ -> false, D))

    hs, errs = Float64[], Float64[]
    for level in 1:levels
        Wc = gridspace(Ωc)
        bcs_c = dirichlet_constraints(Bramble.set(Ωd), :boundary => sol_d)

        a_c = form(Wc, Wc, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        A_c = assemble(a_c; dirichlet_labels = :boundary)
        g_c = element(Wc)
        avgₕ!(g_c, rhs_d)
        l_c = form(Wc, v -> innerₕ(g_c, v))
        F_c = assemble(l_c; dirichlet_conditions = bcs_c, dirichlet_labels = :boundary)

        u_c = element(Wc)
        u_c .= A_c \ F_c
        uexact_c = Rₕ(Wc, sol_d)

        push!(hs, hₘₐₓ(Ωc))
        push!(errs, norm₁ₕ(u_c .- uexact_c))
        level < levels && iterative_refinement!(Ωc)
    end
    return hs, errs
end

Random.seed!(20260903)
hs1, errs1 = poisson_series(1; n0 = 6, levels = 7)  # 6 up to 385 points
Random.seed!(20260903)
hs2, errs2 = poisson_series(2; levels = 6)   # 5² up to 129² points
Random.seed!(20260903)
hs3, errs3 = poisson_series(3; levels = 4)   # 5³ up to 33³ points — 3D is expensive per level

order1 = log(errs1[end - 1] / errs1[end]) / log(hs1[end - 1] / hs1[end])
order2 = log(errs2[end - 1] / errs2[end]) / log(hs2[end - 1] / hs2[end])
order3 = log(errs3[end - 1] / errs3[end]) / log(hs3[end - 1] / hs3[end])
(order1, order2, order3)
```

```@example poisson_linear
order1 > 1.9 && order2 > 1.9 && order3 > 1.8   # comfortably above 1st order; 2nd is the promise
```

```@example poisson_linear
include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot([
    (hs1, errs1, "1D", "#5B5FC7"),
    (hs2, errs2, "2D", "#0E7C86"),
    (hs3, errs3, "3D", "#B26A00"),
]; title = "Linear Poisson, ‖·‖₁ₕ")
```

Second order in every dimension, on grids chosen specifically not to make that trivially
true — every curve's markers sit along the same reference slope. 3D uses fewer refinement
levels than 2D, and 2D fewer than 1D: the same dyadic split costs ``8\times`` the points per
level in 3D against ``4\times`` in 2D and ``2\times`` in 1D, so matching level counts across
all three would make the higher dimensions by far the most expensive part of this page for no
extra information.
