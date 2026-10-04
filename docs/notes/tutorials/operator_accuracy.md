```@meta
CurrentModule = Bramble
```

# [How accurate are the difference operators](@id tutorial_operator_accuracy)

**What you will learn.** How to measure the order of a difference operator, why a truncated boundary point halves the observed order, and when to choose the centered, the second-order or the summation-by-parts difference.

**What you need first.** The [operators tutorial](@ref tutorial_operators), for applying an operator and for the boundary truncation, and the [mesh tutorial](@ref tutorial_mesh) for non-uniform meshes.

**Where next.** The [interpolation tutorial](interpolation.md) moves a grid function between meshes.

Every block below runs when this page is built. The random meshes are drawn from a seeded generator, so every build produces the same numbers.

```@setup operator_accuracy
using Bramble, Random
import Bramble: D₊ₓ, Dcₓ, D̽ₓ, D̃ₓ, change_points!
Random.seed!(20260830)
```

---

## [A convergence study, and the boundary](@id operator_accuracy_convergence)

The backward difference ``D_{-x}`` is first order, so the error against a known derivative should fall by a factor of ten each time the grid is refined by ten. Difference ``\sin`` and compare with ``\cos``, in the discrete norm [`normₕ`](@ref), over every point and then over every point but the first:

```@example operator_accuracy
function study(ns)
    rows = Tuple{Int, Float64, Float64}[]
    for n in ns
        Ω = mesh(domain(interval(0.0, 1.0)), n)
        W = gridspace(Ω)
        e = ∇ₕ(Rₕ(W, sin)) - Rₕ(W, cos)
        every_point = normₕ(e)
        parent(e)[1] = 0.0            # drop the truncated point
        push!(rows, (n, every_point, normₕ(e)))
    end
    return rows
end

rows = study((11, 101, 1001, 10001))
for i in eachindex(rows)
    n, all_pts, interior = rows[i]
    order(k) = i == 1 ? "" : string(round(log10(rows[i - 1][k] / rows[i][k]); digits = 2))
    println(rpad(n, 7), rpad(round(all_pts; sigdigits = 3), 10), rpad(order(2), 6),
        rpad(round(interior; sigdigits = 3), 11), order(3))
end
```

The columns are the number of points, the error over every point and its observed order, then the same two with the first point dropped. Over every point the order is one half, not one.

The cause is the boundary. `D₋ₓ(uₕ)[1]` is `0.0` while ``\cos(0) = 1``, so that one point contributes an error of ``1`` however fine the grid is. It carries a weight of about ``h/2`` in the discrete norm, so it alone contributes about ``\sqrt{h/2}``, which is the half order observed. Dropping it recovers first order.

When you measure convergence or assemble a scheme, treat the truncated slice explicitly. That is where the boundary condition belongs, and leaving the operator's truncated value in place silently halves the observed order. The [form tutorial](form.md) shows where Dirichlet conditions enter.

!!! tip "Try this"
    Replace `sin` and `cos` by `x -> x^2` and `x -> 2x`. The derivative at the left end is now ``0``, so the truncated value happens to be correct there, and the order over every point recovers to one.

---

## The centered difference

Both one-sided differences reach one point. The centered difference reaches both ways and divides by the whole span its stencil covers:

```math
\textrm{Dc}_x(u_h)(i) = \frac{u_{i+1} - u_{i-1}}{h_i + h_{i+1}}
    = \frac{u_{i+1} - u_{i-1}}{x_{i+1} - x_{i-1}}
```

It is the only operator on these pages that truncates on two slices, since neither the first nor the last point has a neighbour on both sides. Compare it with the one-sided differences on a small non-uniform mesh, built by moving the points of a uniform one with [`change_points!`](@ref):

```@example operator_accuracy
Ωₙ = mesh(domain(interval(0.0, 1.0)), 5)
change_points!(Ωₙ, [0.0, 0.1, 0.3, 0.7, 1.0])
uₙ = Rₕ(gridspace(Ωₙ), x -> x^2)

parent(∇ₕ(uₙ)), parent(D₊ₓ(uₙ)), parent(Dcₓ(uₙ))
```

The exact derivative ``2x`` at the interior points is ``0.2, 0.6, 1.4``. The backward difference is low, the forward one is high, and the centered one sits between them. Both ends of the centered result are zero.

Writing the denominator as ``x_{i+1} - x_{i-1}`` rather than as a pair of spacings gives two properties that hold on any grid. First, it reproduces the derivative of an affine function exactly, since numerator and denominator are then the same quantity:

```@example operator_accuracy
parent(Dcₓ(Rₕ(gridspace(Ωₙ), x -> 3x + 1)))
```

Second, it is skew-symmetric in [`innerₕ`](@ref) for grid functions that vanish on the boundary. Check it on a random mesh:

```@example operator_accuracy
Ωₛ = mesh(domain(interval(0.0, 1.0)), 41, false)    # a random, non-uniform grid
Wₛ = gridspace(Ωₛ)
pₕ = Rₕ(Wₛ, x -> sin(pi * x))
qₕ = Rₕ(Wₛ, x -> sin(2pi * x) * x * (1 - x))        # both zero at both ends

innerₕ(Dcₓ(pₕ), qₕ), -innerₕ(pₕ, Dcₓ(qₕ))
```

The two numbers agree to machine precision. The weight of `innerₕ` at point ``i`` is exactly half the centered denominator, so the weights drop out and shifting the index by one turns what is left into minus the right side.

The centered difference approximates the derivative at the midpoint of its stencil, which is ``x_i`` only when the two spacings match. So it is second order on a uniform grid and first order otherwise, where the one-sided differences are first order on both. Like every other family, `Dcₓ` accepts a mesh or a grid space for the matrix and a grid function to apply it, and `∇cₕ` gives every coordinate at once. Both end rows of the matrix are empty, which is the truncation.

---

## Second order on a non-uniform grid

`Dcₓ` is second order only on a uniform grid. The fix is to take the same two one-sided differences and weight them by the opposite spacings:

```math
\overset{\times}{\textrm{D}}_{x}(u_h)(i) = \frac{h_i}{h_i + h_{i+1}}\, D_{-x} u_h(x_{i+1})
                        + \frac{h_{i+1}}{h_i + h_{i+1}}\, D_{-x} u_h(x_i)
```

This is `D̽ₓ`. Compare it with `Dcₓ`, which is the same combination with the weights the other way round. When ``h_i = h_{i+1}`` the two agree, and both reduce to the mean of ``D_{-x}`` and ``D_{+x}``. They part company only where the spacing varies.

The swap makes `D̽ₓ` exact on quadratics, not only on affine functions, on any grid. With ``u = x^2`` the weighted sum telescopes to ``2 x_i (h_i + h_{i+1})`` and the denominator cancels:

```@example operator_accuracy
parent(Dcₓ(uₙ)), parent(D̽ₓ(uₙ)), 2 .* points(Ωₙ)
```

The middle vector matches the last in the interior. One order of extra exactness is one order of extra accuracy. Refine a random grid by halving every interval, so the grids stay nested, and difference ``\sin`` against ``\cos`` at the interior points:

```@example operator_accuracy
function refinement_study(levels)
    Ω = mesh(domain(interval(0.0, 1.0)), 21, false)
    errors = Tuple{Int, Float64, Float64}[]
    for level in 1:levels
        u = Rₕ(gridspace(Ω), sin)
        exact = cos.(points(Ω))
        interior(D) = maximum(abs, (parent(D(u)) .- exact)[2:(end - 1)])
        push!(errors, (npoints(Ω), interior(Dcₓ), interior(D̽ₓ)))
        x = points(Ω)
        Ω = mesh(domain(interval(0.0, 1.0)), 2 * length(x) - 1)
        change_points!(Ω, sort!(vcat(x, (x[1:(end - 1)] .+ x[2:end]) ./ 2)))
    end
    return errors
end

errors = refinement_study(4)
for i in eachindex(errors)
    n, ec, ex = errors[i]
    order(k) = i == 1 ? "" : string(round(log2(errors[i - 1][k] / errors[i][k]); digits = 2))
    println(rpad(n, 6), rpad(round(ec; sigdigits = 3), 11), rpad(order(2), 6),
        rpad(round(ex; sigdigits = 3), 11), order(3))
end
```

The columns are the number of points, then the error and observed order of `Dcₓ` and of `D̽ₓ`. On the random grid `Dcₓ` stays near order one and `D̽ₓ` approaches order two.

`D̽ₓ` is not skew-symmetric, so `Dcₓ` remains the choice when the scheme needs that structure, and `D̽ₓ` the choice when it needs the order. Both accept a mesh or a grid space for the matrix and a grid function to apply it, and `∇̽ₕ` gives every coordinate at once, the second-order, non-uniform-grid counterpart of `∇ₕ` and `∇₊ₕ`. They differ at the boundary: `Dcₓ` truncates both end rows to zero, while `D̽ₓ` has no truncated-boundary convention of its own and falls back to `D₊ₓ`/`D₋ₓ` there.

!!! tip "Try this"
    Pass `true` instead of `false` as the third argument of `mesh` inside `refinement_study` and run the study again. On a uniform grid both operators reach order two, because the spacings match and the two differences coincide.

---

## Summation by parts

Continuous integration by parts, ``\int u' v = -\int u v'`` for ``v`` vanishing on the boundary, has a discrete counterpart, and the forward difference that satisfies it is not the obvious one. The operator is `D̃ₓ`, also reached as `∇̃ₕ[1]` or by destructuring `∇̃ₕ`: the forward difference divided by the averaged spacing rather than by the forward spacing,

```math
\tilde{\textrm{D}}_{+x}(u_h)(i) = \frac{u_{i+1} - u_i}{(h_i + h_{i+1})/2}
```

with the last point truncated to zero, as `D₊ₓ` is. On a uniform grid ``h_i = h_{i+1}`` and it coincides with `D₊ₓ`. The two differ only where the spacing varies:

```@example operator_accuracy
parent(D₊ₓ(uₙ)), parent(D̃ₓ(uₙ))
```

The identity is

```math
(\tilde{\textrm{D}}_{+x} u_h,\, v_h)_h = -(u_h,\, D_{-x} v_h)_{+x}
```

for any `vₕ` that vanishes on the boundary. The left product is `innerₕ`, weighted by the cell measures, and the right is `inner₊(·, ·, :x)`, weighted by the staggered ones. Only `vₕ` has to vanish; `uₕ` is unconstrained, since the boundary term the identity discards is a product of the two. Test it on a random mesh, with a function `aₕ` that does not vanish at the boundary:

```@example operator_accuracy
Ωₚ = mesh(domain(interval(0.0, 1.0)), 21, false)
Wₚ = gridspace(Ωₚ)
aₕ = Rₕ(Wₚ, x -> cos(x) + 0.7)          # not zero at the boundary
bₕ = Rₕ(Wₚ, x -> sin(pi * x))           # zero at both ends

innerₕ(D̃ₓ(aₕ), bₕ), -inner₊(aₕ, ∇ₕ(bₕ), :x), innerₕ(D₊ₓ(aₕ), bₕ)
```

The first two numbers agree to machine precision. The third, with `D₊ₓ`, does not.

This is why the operator exists. Energy estimates for these schemes are derived by moving a difference from one factor to the other, and that step is exact only with this pairing. With `D₊ₓ` it leaves a residual that does not vanish under refinement, since it is a difference of quadrature weights and not a truncation error. The identity holds per coordinate in two and three dimensions with `D̃ᵧ`, `D̃₂` and their inner products, and `∇̃ₕ` returns all coordinates at once, as `∇₊ₕ` does.

Like the other difference families, `D̃` can also be had as a sparse matrix: `D̃ₓ(Wₕ)` is `diag(2/(hᵢ + hᵢ₊₁))` times the undivided forward difference ``u_{i+1} - u_i``, with an empty last row.
