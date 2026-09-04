# Convection-diffusion equation

One more linear operator, built the same way as the [Poisson example](poisson_linear.md) —
diffusion plus a constant convective term, in 1D, 2D and 3D. Every number and every plot
below was produced by the code shown.

## Problem

```math
-\nabla \cdot (\epsilon \nabla u) + b \, \mathbf{1} \cdot \nabla u = g \text{ in } \Omega,
\qquad u = u_{\text{exact}} \text{ on } \partial\Omega, \qquad \Omega = (0,1)^D
```

with a constant diffusion coefficient ``\epsilon`` and a constant convective speed ``b`` along
every coordinate direction at once — the vector field ``b(1,\dots,1)``. The manufactured
solution is the same as the other two examples, ``u_{\text{exact}}(x) = e^{\sum_i x_i}``, so
``g = -D\,(b+\epsilon)\,u_{\text{exact}}``.

```@example convdiff
using Bramble

ϵ, b = 1.0, 0.1
sol(x) = exp(x[1] + x[2])
rhs(x) = -2 * sol(x) * (b + ϵ)

Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ = mesh(Ω, (24, 24), (false, false))
Wₕ = gridspace(Ωₕ)
```

`inner₊(M₋ₕ(u), ∇₋ₕ(v))` is the convective term: `M₋ₕ` averages the trial function onto the
same staggered points `∇₋ₕ` differences on, one pair per direction, and `inner₊`'s own
gradient-tuple overload sums them — the identical spelling whether `D` is 1 or 3, since
`M₋ₕ`/`∇₋ₕ` collapse to a bare node instead of a one-element tuple in 1D and `inner₊` has a
method for both:

```@example convdiff
bcs = dirichlet_constraints(Bramble.set(Ω), :boundary => sol)

a = form(Wₕ, Wₕ, (u, v) -> ϵ * inner₊(∇₋ₕ(u), ∇₋ₕ(v)) + b * inner₊(M₋ₕ(u), ∇₋ₕ(v)))
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

```@example convdiff
include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide
heatmap_plot(uₕ; title = "Convection-diffusion, 2D")
```

## Checking the answer

The same pattern as the other two examples — one random coarse mesh per dimension, refined
in place with [`iterative_refinement!`](@ref):

```@example convdiff
using Random

function convdiff_series(D::Int; n0::Int = 5, levels::Int)
    sol_d(x) = exp(sum(x))
    rhs_d(x) = -D * sol_d(x) * (b + ϵ)

    Ωd = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D)))
    Ωc = mesh(Ωd, ntuple(_ -> n0, D), ntuple(_ -> false, D))

    hs, errs = Float64[], Float64[]
    for level in 1:levels
        Wc = gridspace(Ωc)
        bcs_c = dirichlet_constraints(Bramble.set(Ωd), :boundary => sol_d)

        a_c = form(Wc, Wc,
            (u, v) -> ϵ * inner₊(∇₋ₕ(u), ∇₋ₕ(v)) + b * inner₊(M₋ₕ(u), ∇₋ₕ(v)))
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
hs1, errs1 = convdiff_series(1; n0 = 6, levels = 7)
Random.seed!(20260903)
hs2, errs2 = convdiff_series(2; levels = 6)
Random.seed!(20260903)
hs3, errs3 = convdiff_series(3; levels = 4)

order1 = log(errs1[end - 1] / errs1[end]) / log(hs1[end - 1] / hs1[end])
order2 = log(errs2[end - 1] / errs2[end]) / log(hs2[end - 1] / hs2[end])
order3 = log(errs3[end - 1] / errs3[end]) / log(hs3[end - 1] / hs3[end])
(order1, order2, order3)
```

```@example convdiff
order1 > 1.9 && order2 > 1.9 && order3 > 1.8
```

```@example convdiff
include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot([
    (hs1, errs1, "1D", "#5B5FC7"),
    (hs2, errs2, "2D", "#0E7C86"),
    (hs3, errs3, "3D", "#B26A00"),
]; title = "Convection-diffusion, ‖·‖₁ₕ")
```

Second order in every dimension. The convective term does not change the rate — it changes
the matrix from symmetric to non-symmetric (`inner₊(M₋ₕ(u), ∇₋ₕ(v)) ≠ inner₊(M₋ₕ(v), ∇₋ₕ(u))`
in general), which is why this example does not also check `issymmetric`, unlike the
[forms tutorial](../tutorials/form.md)'s pure-diffusion Poisson problem.
