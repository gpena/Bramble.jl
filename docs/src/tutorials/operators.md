# Difference, jump and average operators

Bramble provides the finite difference building blocks that discrete schemes are written
in: differences, jumps, averages, and their algebraic structures.
This tutorial covers:

1. The four operator families and how their names are built.
2. Applying an operator to a grid function, and what happens at the boundary.
3. The same operator as a sparse matrix.
4. Gradients and the other vectorial forms.
5. Summation by parts and skew-symmetry.
6. A convergence study, and the boundary effect that will otherwise spoil it.

Every number below was produced by the code shown.

## 1. The operator families

There are four families. Two of them differ only by a division:

| Family | Meaning | Backward form |
|:--|:--|:--|
| unscaled difference | a plain difference | ``u_i - u_{i-1}`` |
| finite difference | divided by the spacing, so it approximates ``\partial u / \partial x`` | ``\dfrac{u_i - u_{i-1}}{h_i}`` |
| jump | the same arithmetic as the unscaled forward difference, used where the intent is a discontinuity across an interface | ``u_{i+1} - u_i`` |
| average | the mean of a point and its neighbour | ``\dfrac{u_{i-1} + u_i}{2}`` |

The jump and the unscaled forward difference compute the same numbers. They are separate
names because they play different roles in a scheme, and reading `jumpₓ` in a penalty term
says something that `diff₊ₓ` does not.

The jump is also the one family with no backward form. A jump belongs to the interface
between two cells rather than to a direction of travel across it, so
``\llbracket u \rrbracket = u_{i+1} - u_i`` at the interface between ``x_i`` and
``x_{i+1}`` is a single quantity; a backward jump would name that same interface from the
other side and give the same numbers shifted by one index.

### 1.1 How the names are built

A name is a stem, a direction, and a coordinate:

| Piece | Meaning |
|:--|:--|
| `diff` | unscaled difference |
| `D` | finite difference |
| `jump` | jump |
| `M` | average |
| `₋` | backward: the stencil reaches to ``i-1`` |
| `₊` | forward: the stencil reaches to ``i+1`` |
| `ₓ`, `ᵧ`, `₂` | along the first, second or third coordinate |
| `ₕ` | every coordinate at once, returning a tuple |

So `D₋ₓ` is the backward finite difference along ``x``, `M₊ᵧ` the forward average along
``y``, and `∇₋ₕ` the backward finite difference in every coordinate, which is the discrete
gradient and has that extra name for it.

`jump` takes no direction, for the reason given above: it is `jumpₓ`, `jumpᵧ`, `jump₂` and
`jumpₕ`.

## 2. Applying an operator

An operator takes a [`VectorElement`](@ref) and returns a new one on the same space.

```@setup operators
using Bramble, Random
using Bramble: values
# The non-uniform meshes further down are drawn at random, so the page is seeded to make
# every build produce the same numbers.
Random.seed!(20260830)
```

```@repl operators
using Bramble
Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true);
Wₕ = gridspace(Ωₕ);
points(Ωₕ)
spacings(Ωₕ)
uₕ = Rₕ(Wₕ, x -> x^2);
values(uₕ)
```

The four backward operators on that grid function:

```@repl operators
values(diff₋ₓ(uₕ))
values(D₋ₓ(uₕ))
values(M₋ₓ(uₕ))
```

Reading the second entry of each: `diff₋ₓ` gives ``u_2 - u_1 = 0.0625``, `D₋ₓ` divides
that by ``h_2 = 0.25`` to get ``0.25``, and `M₋ₓ` averages ``(u_1 + u_2)/2 = 0.03125``.

The jump has no backward form, so it matches the *forward* unscaled difference instead:

```@repl operators
values(diff₊ₓ(uₕ))
values(jumpₓ(uₕ))
```

entry for entry, as section 1 said it would.

## 3. What happens at the boundary

Every operator has one slice where its stencil runs off the grid: the first point for a
backward operator, the last for a forward one. There is no neighbour there, so the
stencil is truncated, and the two difference families truncate differently.

```@raw html
<figure>
<svg viewBox="0 0 720 250" width="100%" style="max-width: 720px; height: auto;"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="A five point grid showing the backward stencil reaching from a point to its left neighbour, the forward stencil reaching right, and the boundary points where each stencil is truncated.">
  <!-- axis -->
  <line x1="60" y1="120" x2="660" y2="120" stroke="currentColor" stroke-width="1.5"/>
  <!-- points -->
  <circle cx="60"  cy="120" r="5" fill="currentColor"/>
  <circle cx="210" cy="120" r="5" fill="currentColor"/>
  <circle cx="360" cy="120" r="5" fill="currentColor"/>
  <circle cx="510" cy="120" r="5" fill="currentColor"/>
  <circle cx="660" cy="120" r="5" fill="currentColor"/>
  <text x="60"  y="145" font-size="13" fill="currentColor" text-anchor="middle">x₁</text>
  <text x="210" y="145" font-size="13" fill="currentColor" text-anchor="middle">x₂</text>
  <text x="360" y="145" font-size="13" fill="currentColor" text-anchor="middle">x₃</text>
  <text x="510" y="145" font-size="13" fill="currentColor" text-anchor="middle">x₄</text>
  <text x="660" y="145" font-size="13" fill="currentColor" text-anchor="middle">x₅</text>

  <!-- backward stencil at x3 -->
  <path d="M 360 105 L 210 105" stroke="#ef4444" stroke-width="2" fill="none"
        marker-end="url(#arrowR)"/>
  <text x="285" y="95" font-size="13" fill="#ef4444" text-anchor="middle">backward: uses x₂ and x₃</text>

  <!-- forward stencil at x3 -->
  <path d="M 360 135 L 510 135" stroke="#10b981" stroke-width="2" fill="none"
        marker-end="url(#arrowG)"/>
  <text x="435" y="158" font-size="13" fill="#10b981" text-anchor="middle">forward: uses x₃ and x₄</text>

  <!-- truncated ends -->
  <text x="60"  y="192" font-size="12" fill="#8b5cf6" text-anchor="middle">no x₀</text>
  <text x="60"  y="209" font-size="12" fill="#8b5cf6" text-anchor="middle">backward truncated here</text>
  <text x="660" y="192" font-size="12" fill="#8b5cf6" text-anchor="middle">no x₆</text>
  <text x="660" y="209" font-size="12" fill="#8b5cf6" text-anchor="middle">forward truncated here</text>
  <line x1="60"  y1="120" x2="60"  y2="180" stroke="#8b5cf6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="660" y1="120" x2="660" y2="180" stroke="#8b5cf6" stroke-width="1" stroke-dasharray="3,3"/>

  <defs>
    <marker id="arrowR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6"
            markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#ef4444"/>
    </marker>
    <marker id="arrowG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6"
            markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981"/>
    </marker>
  </defs>
</svg>
</figure>
```

The finite difference is **zero** on its truncated slice, because there is no one-sided
stencil to divide by a spacing. The unscaled difference and the jump instead behave as
if the missing neighbour were zero, which is what makes them agree with their matrices:

The backward finite difference is truncated at ``x_1`` and the forward one at ``x_5``,
while the unscaled differences act as though the missing neighbour were zero:

```@repl operators
values(D₋ₓ(uₕ))[1]
values(D₊ₓ(uₕ))[end]
values(diff₊ₓ(uₕ))[end]   # -u₅, not 0
values(diff₋ₓ(uₕ))[1]     # u₁, and u₁ happens to be 0 here
```

Section 9 shows why this matters in practice.

In two or more dimensions, directional operators apply along the coordinate lines of the tensor grid, and each directional family truncates along its corresponding boundary slice:

```@raw html
<figure>
<svg viewBox="0 0 740 310" width="100%" style="max-width:740px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="A 4 by 4 two-dimensional tensor-product grid showing the directional backward stencils D_-x (horizontal) and D_-y (vertical), and the boundary slices where each directional difference is truncated to zero.">
  <defs>
    <marker id="arrowRed" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#ef4444"/>
    </marker>
    <marker id="arrowBlue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#3b82f6"/>
    </marker>
  </defs>

  <!-- Left boundary slice shaded (D_-x truncated) -->
  <rect x="50" y="40" width="40" height="210" rx="6" fill="#ef4444" fill-opacity="0.12" stroke="#ef4444" stroke-dasharray="3,3" stroke-width="1"/>
  <text x="70" y="30" font-size="11" font-weight="bold" fill="#ef4444" text-anchor="middle">D₋ₓ = 0</text>
  <text x="70" y="265" font-size="10" fill="#ef4444" text-anchor="middle">i = 1 slice</text>

  <!-- Bottom boundary slice shaded (D_-y truncated) -->
  <rect x="50" y="210" width="250" height="40" rx="6" fill="#3b82f6" fill-opacity="0.12" stroke="#3b82f6" stroke-dasharray="3,3" stroke-width="1"/>
  <text x="315" y="234" font-size="11" font-weight="bold" fill="#3b82f6">D₋ᵧ = 0 (j = 1 slice)</text>

  <!-- 4x4 Grid lines -->
  <!-- Horizontal lines (y = const) -->
  <line x1="70" y1="70"  x2="280" y2="70"  stroke="currentColor" stroke-opacity="0.3" stroke-width="1.2"/>
  <line x1="70" y1="120" x2="280" y2="120" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.2"/>
  <line x1="70" y1="170" x2="280" y2="170" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.2"/>
  <line x1="70" y1="230" x2="280" y2="230" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.2"/>

  <!-- Vertical lines (x = const) -->
  <line x1="70"  y1="70" x2="70"  y2="230" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.2"/>
  <line x1="140" y1="70" x2="140" y2="230" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.2"/>
  <line x1="210" y1="70" x2="210" y2="230" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.2"/>
  <line x1="280" y1="70" x2="280" y2="230" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.2"/>

  <!-- Grid vertices -->
  <!-- row 4 (j=4) -->
  <circle cx="70"  cy="70" r="3.5" fill="currentColor"/>
  <circle cx="140" cy="70" r="3.5" fill="currentColor"/>
  <circle cx="210" cy="70" r="3.5" fill="currentColor"/>
  <circle cx="280" cy="70" r="3.5" fill="currentColor"/>
  <!-- row 3 (j=3) -->
  <circle cx="70"  cy="120" r="3.5" fill="currentColor"/>
  <circle cx="140" cy="120" r="3.5" fill="currentColor"/>
  <circle cx="210" cy="120" r="3.5" fill="currentColor"/>
  <circle cx="280" cy="120" r="3.5" fill="currentColor"/>
  <!-- row 2 (j=2) -->
  <circle cx="70"  cy="170" r="3.5" fill="currentColor"/>
  <circle cx="140" cy="170" r="3.5" fill="currentColor"/>
  <circle cx="210" cy="170" r="3.5" fill="currentColor"/>
  <circle cx="280" cy="170" r="3.5" fill="currentColor"/>
  <!-- row 1 (j=1) -->
  <circle cx="70"  cy="230" r="3.5" fill="currentColor"/>
  <circle cx="140" cy="230" r="3.5" fill="currentColor"/>
  <circle cx="210" cy="230" r="3.5" fill="currentColor"/>
  <circle cx="280" cy="230" r="3.5" fill="currentColor"/>

  <!-- Stencils at interior point (i=3, j=3), located at (210, 120) -->
  <circle cx="210" cy="120" r="6" fill="#10b981" stroke="currentColor" stroke-width="1.5"/>
  <text x="210" y="110" font-size="11" font-weight="bold" fill="#10b981" text-anchor="middle">(i, j)</text>

  <!-- Horizontal backward stencil D_-x: reaching from (210, 120) to (140, 120) -->
  <path d="M 204 120 L 148 120" stroke="#ef4444" stroke-width="2.2" fill="none" marker-end="url(#arrowRed)"/>
  <text x="175" y="135" font-size="11" font-weight="bold" fill="#ef4444" text-anchor="middle">D₋ₓ</text>

  <!-- Vertical backward stencil D_-y: reaching from (210, 120) down to (210, 170) -->
  <path d="M 210 126 L 210 162" stroke="#3b82f6" stroke-width="2.2" fill="none" marker-end="url(#arrowBlue)"/>
  <text x="225" y="150" font-size="11" font-weight="bold" fill="#3b82f6">D₋ᵧ</text>

  <!-- Legend & Explanation on Right Side -->
  <g transform="translate(440, 50)">
    <rect x="0" y="0" width="280" height="190" rx="6" fill="none" stroke="currentColor" stroke-opacity="0.2" stroke-width="1"/>
    <text x="140" y="25" font-size="13" font-weight="bold" fill="currentColor" text-anchor="middle">2D Directional Stencils</text>

    <!-- Entry 1: D_-x -->
    <line x1="20" y1="55" x2="50" y2="55" stroke="#ef4444" stroke-width="2.5"/>
    <text x="60" y="58" font-size="12" font-weight="bold" fill="#ef4444">D₋ₓ(uₕ)[i, j]</text>
    <text x="60" y="73" font-size="11" fill="currentColor" opacity="0.8">= (u[i, j] - u[i-1, j]) / hₓ,ᵢ</text>
    <text x="60" y="88" font-size="11" fill="#ef4444">Zero on left boundary (i = 1)</text>

    <!-- Entry 2: D_-y -->
    <line x1="20" y1="120" x2="50" y2="120" stroke="#3b82f6" stroke-width="2.5"/>
    <text x="60" y="123" font-size="12" font-weight="bold" fill="#3b82f6">D₋ᵧ(uₕ)[i, j]</text>
    <text x="60" y="138" font-size="11" fill="currentColor" opacity="0.8">= (u[i, j] - u[i, j-1]) / hᵧ,ⱼ</text>
    <text x="60" y="153" font-size="11" fill="#3b82f6">Zero on bottom boundary (j = 1)</text>

    <text x="140" y="178" font-size="11" fill="#10b981" font-weight="bold" text-anchor="middle">∇₋ₕ(uₕ) = (D₋ₓ(uₕ), D₋ᵧ(uₕ))</text>
  </g>
</svg>
</figure>
```

## 4. Operators as matrices

Passing a mesh or a grid space, rather than a grid function, returns the operator itself
as a sparse matrix:

```@repl operators
A = D₋ₓ(Wₕ);
typeof(A)
A * values(uₕ) ≈ values(D₋ₓ(uₕ))
```

Both routes give the same answer. Applying the operator directly to `uₕ` is the fast
path and is what a time-stepping loop should use; the matrix is what to reach for when
assembling a linear system, and it is also how the test suite checks the fast path.

## 5. Gradients and the other vectorial forms

The `ₕ` suffix applies the operator along every coordinate and returns a tuple with one
entry per dimension. On a one-dimensional mesh it returns the single element itself
rather than a one-tuple.

```@repl operators
Ω₂ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (true, true));
W₂ = gridspace(Ω₂);
vₕ = Rₕ(W₂, x -> x[1] + 2x[2]);
g = ∇₋ₕ(vₕ);
length(g)
```

Away from the truncated slices, `g[1]` is `1.0` and `g[2]` is `2.0`, the two partial
derivatives of ``x + 2y``. The same suffix works for the other families as `diff₋ₕ`,
`jumpₕ` and `M₋ₕ`, and all of them accept a mesh, a grid space or a grid function.

## 6. Summation by parts, and `Dstar₊ₓ`

Continuous integration by parts, ``\int u' v = -\int u v'`` for ``v`` vanishing on the
boundary, has a discrete counterpart, and which forward difference it holds for is not
the obvious one. The operator that satisfies it is `Dstar₊ₓ`: the forward difference
divided by the **averaged** spacing rather than by the forward spacing,

```math
\textrm{Dstar}_{+x}(u_h)(i) = \frac{u_{i+1} - u_i}{(h_i + h_{i+1})/2}
```

with the last point truncated to zero, as `D₊ₓ` is. On a uniform grid ``h_i = h_{i+1}``
and it coincides with `D₊ₓ`; the two differ only where the spacing varies:

```@repl operators
Ωₙ = mesh(domain(interval(0.0, 1.0)), 5, true);
set_points!(Ωₙ, [0.0, 0.1, 0.3, 0.7, 1.0])
uₙ = Rₕ(gridspace(Ωₙ), x -> x^2);
values(D₊ₓ(uₙ))
values(Dstar₊ₓ(uₙ))
```

The identity is

```math
(\textrm{Dstar}_{+x} u_h,\, v_h)_h = -(u_h,\, D_{-x} v_h)_{+x}
```

for any `vₕ` that vanishes on the boundary. Note which product sits on each side: the
left is `innerₕ`, weighted by the cell measures, and the right is `inner₊ₓ`, weighted by
the staggered ones. Only `vₕ` has to vanish; `uₕ` is unconstrained, since the boundary
term the identity discards is a product of the two.

```@repl operators
Ωᵣ = mesh(domain(interval(0.0, 1.0)), 21, false);   # a random, non-uniform grid
Wᵣ = gridspace(Ωᵣ);
aₕ = Rₕ(Wᵣ, x -> cos(x) + 0.7);                     # not zero at the boundary
bₕ = Rₕ(Wᵣ, x -> sin(pi * x));                      # zero at both ends
innerₕ(Dstar₊ₓ(aₕ), bₕ)
-inner₊ₓ(aₕ, D₋ₓ(bₕ))                              # equal to machine precision
innerₕ(D₊ₓ(aₕ), bₕ)                                # D₊ₓ does not agree
```

It holds per coordinate in two and three dimensions as well, with `Dstar₊ᵧ`, `Dstar₊₂`
and their inner products. `Dstar₊ₕ` returns all coordinates at once, as `∇₊ₕ` does.

This is why the operator exists. Energy estimates for these schemes are derived by
moving a difference from one factor to the other, and that step is exact only with this
pairing: with `D₊ₓ` it leaves a residual that does not vanish under refinement, since it
is a difference of quadrature weights and not a truncation error. Like the other
difference families, `Dstar₊` can also be had as a sparse matrix: `Dstar₊ₓ(Wₕ)` is
`diag(2/(hᵢ + hᵢ₊₁))` times the unscaled forward difference, with an empty last row.

## 7. The centered difference, `Dcₓ`

Both one-sided differences reach one point; the centered one reaches both ways, and
divides by the whole span its stencil covers:

```math
\textrm{Dc}_x(u_h)(i) = \frac{u_{i+1} - u_{i-1}}{h_i + h_{i+1}}
    = \frac{u_{i+1} - u_{i-1}}{x_{i+1} - x_{i-1}}
```

It is the only operator here that truncates on **two** slices, since neither the first
nor the last point has a neighbour on both sides.

```@repl operators
values(D₋ₓ(uₙ))
values(D₊ₓ(uₙ))
values(Dcₓ(uₙ))
```

Writing the denominator as ``x_{i+1} - x_{i-1}`` rather than as a pair of spacings buys
two properties that hold on **any** grid, not only a uniform one.

First, it reproduces an affine function's derivative exactly, since numerator and
denominator are then the same quantity:

```@repl operators
values(Dcₓ(Rₕ(gridspace(Ωₙ), x -> 3x + 1)))
```

Second, it is skew-symmetric in `innerₕ` for grid functions vanishing on the boundary:

```@repl operators
Ωₛ = mesh(domain(interval(0.0, 1.0)), 41, false);   # a random, non-uniform grid
Wₛ = gridspace(Ωₛ);
pₕ = Rₕ(Wₛ, x -> sin(pi * x));
qₕ = Rₕ(Wₛ, x -> sin(2pi * x) * x * (1 - x));       # both zero at both ends
innerₕ(Dcₓ(pₕ), qₕ)
-innerₕ(pₕ, Dcₓ(qₕ))                               # equal to machine precision
```

The reason is the same cancellation that gives `Dstar₊ₓ` its identity in section 6:
`innerₕ` weights point ``i`` by the cell measure ``(h_i + h_{i+1})/2``, which is exactly
half the centered denominator. The weights cancel, and the left side collapses to

```math
\tfrac{1}{2} \sum_i (u_{i+1} - u_{i-1})\, v_i
```

which shifting the index by one turns into minus the right side. Unlike the `Dstar₊ₓ`
identity, which needs only `vₕ` to vanish, this one needs both: the discarded boundary
term is symmetric in the two.

Accuracy follows the usual rule: the centered difference approximates the derivative at
the midpoint of its stencil, which is ``x_i`` only when the two spacings match. So it is
second order on a uniform grid and first order otherwise, where the one-sided differences
are first order on both. Like every other family, `Dcₓ` accepts a mesh or a grid space for
the matrix and a grid function to apply it; `Dcₕ` gives every coordinate at once. Both end
rows of the matrix are empty, which is the truncation.

## 8. Second order on a non-uniform grid, `Dₕₓ`

`Dcₓ` is second order only on a uniform grid. The fix is to take the same two one-sided
differences and weight them by the **opposite** spacings:

```math
\textrm{D}_{hx}(u_h)(i) = \frac{h_i}{h_i + h_{i+1}}\, D_{-x} u_h(x_{i+1})
                        + \frac{h_{i+1}}{h_i + h_{i+1}}\, D_{-x} u_h(x_i)
```

Compare that with `Dcₓ`, which is the same combination with the weights the other way
round. When ``h_i = h_{i+1}`` the two agree, and both reduce to the mean of ``D_{-x}`` and
``D_{+x}``; they part company only where the spacing varies.

The swap buys exactness on quadratics rather than only on affine functions, on any grid.
With ``u = x^2`` the weighted sum telescopes to ``2 x_i (h_i + h_{i+1})``, and the
denominator cancels:

```@repl operators
values(Dcₓ(uₙ))
values(Dₕₓ(uₙ))
2 .* points(Ωₙ)   # Dₕₓ hits this exactly, away from the two truncated points
```

That one order of extra exactness is one order of extra accuracy. Differencing ``\sin``
against ``\cos`` on a random grid, refined by halving every interval so the grids stay
nested:

| ``n`` | `Dcₓ` error | order | `Dₕₓ` error | order |
|--:|--:|--:|--:|--:|
| 21 | 3.52e-02 | | 2.95e-03 | |
| 41 | 1.78e-02 | 0.99 | 1.30e-03 | 1.18 |
| 81 | 8.94e-03 | 0.99 | 3.32e-04 | 1.97 |
| 161 | 4.48e-03 | 1.00 | 8.36e-05 | 1.99 |

`Dₕₓ` is not skew-symmetric, so `Dcₓ` remains the one to reach for when the scheme needs
that structure and `Dₕₓ` the one to reach for when it needs the order. Both truncate on
two slices, both take a grid function only, and `∇ₕ` gives every coordinate at once, the
centered counterpart of `∇₋ₕ` and `∇₊ₕ`.

## 9. A convergence study, and the boundary

`D₋ₓ` is first order, so the error against a known derivative should fall by a factor of
ten each time the grid is refined by ten. Measuring it naively does not show that:

```@example operators
for n in (11, 101, 1001, 10001)
    Ω = mesh(domain(interval(0.0, 1.0)), n, true)
    W = gridspace(Ω)
    u = Rₕ(W, sin)
    e = D₋ₓ(u) - Rₕ(W, cos)
    println(n, "  ", normₕ(e))
end
```

| ``n`` | every point | order | interior only | order |
|--:|--:|--:|--:|--:|
| 11 | 0.317349 | | 0.026650 | |
| 101 | 0.100034 | 0.50 | 0.002617 | 1.01 |
| 1001 | 0.031624 | 0.50 | 0.000261 | 1.00 |
| 10001 | 0.010000 | 0.50 | 0.000026 | 1.00 |

Over every point the observed order is one half, not one. The cause is section 3:
`D₋ₓ(uₙ)[1]` is `0.0` while ``\cos(0) = 1``, so that one point contributes an error of
``1`` no matter how fine the grid is. It carries a weight of about ``h/2`` in the
discrete norm, so it alone contributes about ``\sqrt{h/2}``, which is exactly the half
order observed.

Excluding that single truncated point recovers the expected first order. So when
measuring convergence, or assembling a scheme, treat the truncated slice explicitly:
that is where the boundary condition belongs, and leaving the operator's truncated value
in place silently halves the observed order.
