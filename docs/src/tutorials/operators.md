```@meta
CurrentModule = Bramble
```

# [Difference, jump and average operators](@id tutorial_operators)

You will apply a difference, a jump or an average to a discrete function, see what each does at the edge of the grid, and get the operator as a matrix. The [mesh tutorial](@ref tutorial_mesh) and the [space tutorial](@ref tutorial_space) come first, for meshes, grid spaces and discrete functions. The [form tutorial](form.md) assembles these operators into linear systems.

---

## Apply an operator

Take a grid of five points and the function ``u(x) = x^2`` restricted to it:

```@setup operators
using Bramble
import Bramble: D₊ₓ, D₋ᵧ, change_points!
```

```@example operators
Ωₕ = mesh(domain(interval(0.0, 1.0)), 5)
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, x -> x^2)

@assert parent(uₕ) == [0.0, 0.0625, 0.25, 0.5625, 1.0]   # x_i = (i-1)/4, squared

points(Ωₕ), parent(uₕ)
```

An operator takes a [`VectorElement`](@ref) such as `uₕ` and returns a new one on the same space. The backward difference [`∇ₕ`](@ref) divides ``u_i - u_{i-1}`` by the spacing, and the backward average [`Mₕ`](@ref) takes ``(u_{i-1} + u_i)/2``:

```@example operators
@assert parent(∇ₕ(uₕ)) ≈ [0.0, 0.25, 0.75, 1.25, 1.75]
@assert parent(Mₕ(uₕ)) ≈ [0.0, 0.03125, 0.15625, 0.40625, 0.78125]

parent(∇ₕ(uₕ)), parent(Mₕ(uₕ))
```

At the second entry the plain difference is ``u_2 - u_1 = 0.0625`` and the spacing is ``h_2 = 0.25``, so `∇ₕ` gives ``0.25``. The average is ``(u_1 + u_2)/2 = 0.03125``. The first entry of the difference is ``0``, which the next section explains.

The jump is the plain difference across an interface, with no division:

```@example operators
@assert parent(jumpₕ(uₕ)) ≈ [0.0625, 0.1875, 0.3125, 0.4375, -1.0]

parent(jumpₕ(uₕ))
```

Here `jumpₕ` is the forward difference ``u_{i+1} - u_i``, so its first entry is ``0.0625``. The last entry, ``-1``, is explained below.

A uniform grid hides a wrong spacing, so check the same operators on a graded one, with points ``0, 0.1, 0.3, 0.6, 1`` and spacings ``0.1, 0.2, 0.3, 0.4``. For ``u = x^2`` the difference quotient is ``(x_i^2 - x_{i-1}^2)/h_i = x_i + x_{i-1}``, whatever the spacing, and the average is ``(x_{i-1}^2 + x_i^2)/2``:

```@example operators
Ωₙ = mesh(domain(interval(0.0, 1.0)), 5)
change_points!(Ωₙ, [0.0, 0.1, 0.3, 0.6, 1.0])
uₙ = Rₕ(gridspace(Ωₙ), x -> x^2)

@assert parent(∇ₕ(uₙ)) ≈ [0.0, 0.1, 0.4, 0.9, 1.6]
@assert parent(Mₕ(uₙ)) ≈ [0.0, 0.005, 0.05, 0.225, 0.68]
@assert parent(jumpₕ(uₙ)) ≈ [0.01, 0.08, 0.27, 0.64, -1.0]

parent(∇ₕ(uₙ))
```

!!! tip "Try this"
    Replace `x^2` by `x`. The derivative is ``1``, so `∇ₕ` returns ones everywhere except at the first point, and `jumpₕ` returns the constant ``0.25``, the spacing, except at the last point.

Names are built from a stem and a direction ([naming scheme](@ref operators_names)).

| Family | Meaning | Backward form |
|:--|:--|:--|
| finite difference | a difference divided by the spacing, so it approximates ``\partial u / \partial x`` | ``\dfrac{u_i - u_{i-1}}{h_i}`` |
| jump | the plain difference across an interface, undivided | ``u_{i+1} - u_i`` |
| average | the mean of a point and its neighbour | ``\dfrac{u_{i-1} + u_i}{2}`` |
| index shift | the neighbour's value, moved onto the point | ``u_{i-1}`` |

The jump has no backward form: it belongs to the interface between ``x_i`` and ``x_{i+1}``, so ``\llbracket u \rrbracket = u_{i+1} - u_i`` is one quantity, and a backward jump would name the same interface from the other side.

---

## Four differences compared

Four families combine the same one-sided differences in different ways:

| Operator | Stencil | Order, non-uniform | Order, uniform | Boundary |
|:--|:--|:--|:--|:--|
| `D₋` | ``\{i-1, i\}`` | 1 | 1 | truncated to `0` at the first point |
| `Dc` | ``\{i-1, i+1\}``, divided by the full span | 1 | 2 | truncated to `0` at both ends; three points at least |
| `D̽` | ``\{i-1, i, i+1\}``, a weighted mean of two backward differences | 2 | 2 | no convention of its own: falls back to `D₊`/`D₋`; three points at least |
| `D̃` | ``\{i, i+1\}``, divided by the averaged spacing | 1 on smoothly graded grids, not consistent where neighbouring spacings differ by an O(1) ratio (random grids); summation by parts holds exactly | 1 | truncated to `0` at the last point, like `D₊` |

On a uniform grid ``h_i = h_{i+1}``: `D̽` and `Dc` collapse to the mean of `D₋` and `D₊`, and `D̃` collapses to `D₊`. They separate only where the spacing varies, which is why a uniform-grid benchmark cannot tell them apart. Use `D̃` with `innerₕ`, for the summation by parts.

---

## What happens at the boundary

Every operator has one slice where its stencil runs off the grid: the first point for a backward operator, the last for a forward one. There is no neighbour there, so the stencil is truncated.

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

The finite difference is zero on its truncated slice, because there is no one-sided stencil to divide by a spacing. The jump instead behaves as if the missing neighbour were zero.

```@example operators
@assert parent(∇ₕ(uₕ))[1] == 0 && parent(∇ₕ(uₕ))[2] ≈ 0.25     # truncated, then not
@assert parent(D₊ₓ(uₕ))[end] == 0 && parent(D₊ₓ(uₕ))[1] ≈ 0.25
@assert parent(jumpₕ(uₕ))[end] ≈ -1                            # -u_5

parent(∇ₕ(uₕ))[1], parent(D₊ₓ(uₕ))[end], parent(jumpₕ(uₕ))[end]
```

The last entry is ``-u_5 = -1``, which makes the jump agree with its matrix.

The index shifts follow the jump: an off-grid neighbour reads as zero, so `S₊ₕ` is zero at the last point and `S₋ₕ` at the first.

```@example operators
@assert parent(S₊ₕ(uₕ)) ≈ [0.0625, 0.25, 0.5625, 1.0, 0.0]
@assert parent(S₋ₕ(uₕ)) ≈ [0.0, 0.0, 0.0625, 0.25, 0.5625]
@assert parent(S₊ₕ(uₕ)) - parent(uₕ) ≈ [0.0625, 0.1875, 0.3125, 0.4375, -1.0]   # the jump

parent(S₊ₕ(uₕ)), parent(S₋ₕ(uₕ))
```

That convention gives three identities that hold at every point, the boundary included. The matrix of `S₊` is the transpose of the matrix of `S₋`, `S₊(u) - u` is `jump(u)`, and `u - S₋(u)` is the undivided backward difference. The exception is `D₊`: it is zero at the last point, so `S₊(u) - u` is not ``h`` times `D₊(u)` there. Inside a form the rule is the same, and a shift of a composed operand, such as `S₊ₓ(D₋ₓ(u))`, reads zero wherever the shifted stencil leaves the grid.

In two or more dimensions the directional operators apply along the coordinate lines of the tensor grid, and each directional family truncates along its own boundary slice:

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

    <text x="140" y="178" font-size="11" fill="#10b981" font-weight="bold" text-anchor="middle">∇ₕ(uₕ) = (D₋ₓ(uₕ), D₋ᵧ(uₕ))</text>
  </g>
</svg>
</figure>
```

---

## Operators as matrices

Pass a mesh or a grid space instead of a grid function and the operator comes back as a sparse matrix:

```@example operators
A = ∇ₕ(Wₕ)

@assert size(A) == (5, 5) && all(iszero, A[1, :])   # the truncated first row
@assert A[3, 2] ≈ -4 && A[3, 3] ≈ 4                 # -1/h and 1/h, with h = 1/4
@assert A * parent(uₕ) ≈ [0.0, 0.25, 0.75, 1.25, 1.75]
@assert ∇ₕ(gridspace(Ωₙ)) * parent(uₙ) ≈ [0.0, 0.1, 0.4, 0.9, 1.6]   # graded grid

typeof(A)
```

Both routes give the same answer. Applying the operator to `uₕ` is the fast path for time loops; the matrix is for assembling a linear system.

---

## Gradients in two dimensions

The `ₕ` suffix applies an operator along every coordinate and returns a tuple with one entry per dimension. On a one-dimensional mesh it returns the single element, which is why `∇ₕ(uₕ)` above was not wrapped in a tuple.

```@example operators
Ω₂ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4))
W₂ = gridspace(Ω₂)
vₕ = Rₕ(W₂, x -> x[1] + 2x[2])
g = ∇ₕ(vₕ)
g₁, g₂ = reshape(parent(g[1]), 4, 4), reshape(parent(g[2]), 4, 4)

@assert length(g) == 2
@assert all(≈(1), g₁[2:end, :]) && all(iszero, g₁[1, :])
@assert all(≈(2), g₂[:, 2:end]) && all(iszero, g₂[:, 1])

length(g), round.(reshape(parent(g[1]), 4, 4); digits = 8), round.(reshape(parent(g[2]), 4, 4); digits = 8)
```

The rows run along ``x`` and the columns along ``y``. Away from the truncated slices `g[1]` is ``1`` and `g[2]` is ``2``, the two partial derivatives of ``x + 2y``; `g[1]` is zero on the first row and `g[2]` on the first column. The suffix works for the other families as `jumpₕ` and `Mₕ`, and all of them accept a mesh, a grid space or a grid function.

!!! tip "Try this"
    Change the function to `x -> x[1]^2` and print both matrices again. `g[1]` now grows down the rows, while `g[2]` is zero everywhere.

The vectorial operators of a [composite space](@ref space_composite) contract with `LinearAlgebra`'s `⋅` and `×` to give the divergence and the curl:

```@example operators
using LinearAlgebra: ⋅, ×
uv = Rₕ(vector_gridspace(Ω₂), x -> (x[1] + 2x[2], 3x[1] + x[2]))
divuv = reshape(parent(∇ₕ ⋅ uv), 4, 4)
curluv = reshape(parent(∇ₕ × uv), 4, 4)

@assert all(≈(2), divuv[2:end, 2:end])    # d/dx (x + 2y) + d/dy (3x + y) = 1 + 1
@assert all(≈(1), curluv[2:end, 2:end])   # d/dx (3x + y) - d/dy (x + 2y) = 3 - 2
@assert parent(∇ₕ ⋅ uv) == parent(divₕ(uv)) && parent(∇ₕ × uv) == parent(curlₕ(uv))

divuv, curluv
```

On a graded rectangle, where the two spacings differ, the same field gives the same values:

```@example operators
Ω₃ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (4, 4))
change_points!(Ω₃, [[0.0, 0.1, 0.4, 1.0], [0.0, 0.5, 1.2, 2.0]])
uv₃ = Rₕ(vector_gridspace(Ω₃), x -> (x[1] + 2x[2], 3x[1] + x[2]))

@assert all(≈(2), reshape(parent(∇ₕ ⋅ uv₃), 4, 4)[2:end, 2:end])
@assert all(≈(1), reshape(parent(∇ₕ × uv₃), 4, 4)[2:end, 2:end])
```

The first two assertions state the divergence and the curl by hand, away from the truncated first row and column. The last checks that `∇ₕ ⋅` and `∇ₕ ×` are the same operators as `divₕ` and `curlₕ`.

---

## Ghost points

At a ghosted boundary the stencils read the ghost as a neighbour. The facet operators `∂ₙ`, `∂ₙₙ` and `jumpₙ`, the corner rules and the Laplacian inside a form are in the [ghost point tutorial](@ref operators_ghost).
