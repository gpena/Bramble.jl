```@meta
CurrentModule = Bramble
```

# [Difference, jump and average operators](@id tutorial_operators)

**What you will learn.** How to apply a difference, a jump or an average to a discrete function, what each does at the edge of the grid, and how to get the same operator as a matrix, and how the normal derivative and the Laplacian read ghost points.

**What you need first.** The [mesh tutorial](@ref tutorial_mesh) and the [space tutorial](@ref tutorial_space), for meshes, grid spaces and discrete functions.

**Where next.** The [form tutorial](form.md) assembles these operators into linear systems.

Every block below runs when this page is built.

---

## Apply an operator

The operators are the building blocks that discrete schemes are written in. Start with a small grid of five points and the function ``u(x) = x^2`` restricted to it:

```@setup operators
using Bramble
import Bramble: D₊ₓ
```

```@example operators
Ωₕ = mesh(domain(interval(0.0, 1.0)), 5)
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, x -> x^2)

points(Ωₕ), parent(uₕ)
```

An operator takes a [`VectorElement`](@ref) such as `uₕ` and returns a new one on the same space. The backward difference [`∇ₕ`](@ref) divides ``u_i - u_{i-1}`` by the spacing, and the backward average [`Mₕ`](@ref) takes ``(u_{i-1} + u_i)/2``:

```@example operators
parent(∇ₕ(uₕ)), parent(Mₕ(uₕ))
```

Read the second entry of each. The plain difference is ``u_2 - u_1 = 0.0625`` and the spacing is ``h_2 = 0.25``, so `∇ₕ` gives ``0.25``. The average is ``(u_1 + u_2)/2 = 0.03125``. The first entry of the difference is ``0``, which the next section explains.

The jump is the plain difference across an interface, with no division:

```@example operators
parent(jumpₕ(uₕ))
```

Here `jumpₕ` is the forward difference ``u_{i+1} - u_i``, so its first entry is ``0.0625``. The last entry, ``-1``, is explained below.

!!! tip "Try this"
    Replace `x^2` by `x`. The derivative is ``1``, so `∇ₕ` returns ones everywhere except at the first point, and `jumpₕ` returns the constant ``0.25``, the spacing, except at the last point.

The four families are summarised below. Their names are built from a stem and a direction, and the [reference](@ref operators_names) at the end of this page spells out the scheme.

| Family | Meaning | Backward form |
|:--|:--|:--|
| finite difference | a difference divided by the spacing, so it approximates ``\partial u / \partial x`` | ``\dfrac{u_i - u_{i-1}}{h_i}`` |
| jump | the plain difference across an interface, undivided | ``u_{i+1} - u_i`` |
| average | the mean of a point and its neighbour | ``\dfrac{u_{i-1} + u_i}{2}`` |
| index shift | the neighbour's value, moved onto the point | ``u_{i-1}`` |

The jump is the one family with no backward form. A jump belongs to the interface between two cells rather than to a direction of travel across it, so ``\llbracket u \rrbracket = u_{i+1} - u_i`` at the interface between ``x_i`` and ``x_{i+1}`` is a single quantity. A backward jump would name the same interface from the other side.

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

The finite difference is zero on its truncated slice, because there is no one-sided stencil to divide by a spacing. The jump instead behaves as if the missing neighbour were zero. Check both ends:

```@example operators
parent(∇ₕ(uₕ))[1], parent(D₊ₓ(uₕ))[end], parent(jumpₕ(uₕ))[end]
```

The last entry is ``-u_5 = -1``, not ``0``. That is what makes the jump agree with its matrix.

The index shifts follow the jump: an off-grid neighbour reads as zero, so `S₊ₕ` is zero at the last point and `S₋ₕ` at the first.

```@example operators
parent(S₊ₕ(uₕ)), parent(S₋ₕ(uₕ)), parent(S₊ₕ(uₕ)) - parent(uₕ) == parent(jumpₕ(uₕ))
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

typeof(A), A * parent(uₕ) ≈ parent(∇ₕ(uₕ))
```

Both routes give the same answer. Applying the operator to `uₕ` is the fast path and what a time-stepping loop should use. The matrix is what to reach for when assembling a linear system, and it is how the test suite checks the fast path.

---

## Gradients in two dimensions

The `ₕ` suffix applies an operator along every coordinate and returns a tuple with one entry per dimension. On a one-dimensional mesh it returns the single element itself rather than a one-tuple, which is why `∇ₕ(uₕ)` above was not wrapped.

```@example operators
Ω₂ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4))
W₂ = gridspace(Ω₂)
vₕ = Rₕ(W₂, x -> x[1] + 2x[2])
g = ∇ₕ(vₕ)

length(g), round.(reshape(parent(g[1]), 4, 4); digits = 8), round.(reshape(parent(g[2]), 4, 4); digits = 8)
```

The rows run along ``x`` and the columns along ``y``. Away from the truncated slices `g[1]` is ``1`` and `g[2]` is ``2``, the two partial derivatives of ``x + 2y``; `g[1]` is zero on the first row and `g[2]` on the first column. The suffix works for the other families as `jumpₕ` and `Mₕ`, and all of them accept a mesh, a grid space or a grid function.

!!! tip "Try this"
    Change the function to `x -> x[1]^2` and print both matrices again. `g[1]` now grows down the rows, while `g[2]` is zero everywhere.

The vectorial operators of a [composite space](@ref space_composite) contract with `LinearAlgebra`'s `⋅` and `×` to give the divergence and the curl:

```@example operators
using LinearAlgebra: ⋅, ×
uv = Rₕ(vector_gridspace(Ω₂), x -> (x[1] + 2x[2], 3x[1] - x[2]))

parent(∇ₕ ⋅ uv) == parent(divₕ(uv)), parent(∇ₕ × uv) == parent(curlₕ(uv))
```

The next page, the [form tutorial](form.md), assembles these operators into linear systems.

---

## [Operators at a ghost boundary](@id operators_ghost)

A difference at the first point has no neighbour, so it is truncated. A second-order Neumann or Robin condition needs that neighbour, and a [ghost space](@ref space_ghost) supplies it. This section reads the operators on a ghost space through the matrices of forms, which the [form tutorial](form.md) introduces: `form(Wₕ, Wₕ, (u, v) -> ...)` takes the unknown `u` and a test function `v` symbolically, and `assemble` returns the matrix. The running example is a graded interval with six points and ghosts on both ends. Its degrees of freedom are the six nodes, then the `xmin` ghost (7), then the `xmax` ghost (8):

```@example operators
import Bramble: change_points!, Dcₓ, Dcᵧ, D₋ₓ, D̃ₓ, D̃ᵧ, D₋ᵧ
using LinearAlgebra: norm

Ω₁ = mesh(domain(interval(0.0, 1.0)), 6)
change_points!(Ω₁, collect(range(0.0, 1.0, length = 6) .^ 2))
W₁ = gridspace(Ω₁; ghosts = :boundary)

x₁ = collect(points(Ω₁))
h₂, hₙ = x₁[2] - x₁[1], x₁[6] - x₁[5]       # the spacings the ghosts copy
row(f, i) = Matrix(assemble(form(W₁, W₁, f)))[i, :]

ndofs(W₁), h₂, hₙ
```

### How differences read the ghost

At a ghosted facet the stencil of every operator reads the ghost as if it were a neighbour, with the boundary spacing repeated beyond it: ``h_1 := h_2`` at the first point and ``h_{n+1} := h_n`` at the last. In 1D an `inner_Γ` term has weight ``1`` at the facet node, so its row is the stencil itself. The backward difference at the first node is ``(u_1 - u_0)/h_2``:

```@example operators
r = row((u, v) -> inner_Γ(D₋ₓ(u), v; markers = (:xmin,)), 1)

@assert r[[1, 7]] ≈ [1 / h₂, -1 / h₂] && count(!iszero, r) == 2

r
```

Entry 1 is the node, entry 7 the ghost, and no other point is read. A facet without ghosts is truncated as in the first sections of this page. With a ghost on `xmin` only, the centred difference of a volume term is zero at the last node and reads the ghost at the first:

```@example operators
W_left = gridspace(Ω₁; ghosts = (:xmin,))
A_dc = Matrix(assemble(form(W_left, W_left, (u, v) -> innerₕ(Dcₓ(u), v))))

@assert all(iszero, A_dc[6, :])             # truncated at xmax, as without ghosts
@assert !iszero(A_dc[1, 7])                 # reads the xmin ghost (degree of freedom 7)

A_dc[1, :], A_dc[6, :]
```

A boundary term that needs a point beyond a facet without ghosts is refused rather than truncated: `inner_Γ(Dcₓ(u), v; markers = (:xmax,))` on this space throws an `ArgumentError` naming the read of a padded point the space lacks.

### The normal derivative and the facet operators

Three operators serve the boundary conditions that name a normal direction. They act at the nodes of a facet with outward normal ``\eta``, and they are only meaningful inside [`inner_Γ`](@ref) or a [`constrained`](@ref) relation: they are not grid functions.

| Operator | Meaning at a facet node, axis ``d`` |
|:--|:--|
| `∂ₙ(u)` | ``\eta\, D_{c,d} u``: the centred difference reading the ghost, times the outward sign |
| `∂ₙₙ(u)` | ``\tilde D_d D_{-d}\, u``: the second difference along the normal, reading the ghost |
| `jumpₙ(w)` | ``w_{I - \eta e_d} - w_I``: the inward jump, which reads no ghost |

On the six-point mesh the stencils are explicit, at both ends:

```@example operators
r∂ₙ_min = row((u, v) -> inner_Γ(∂ₙ(u), v; markers = (:xmin,)), 1)
r∂ₙ_max = row((u, v) -> inner_Γ(∂ₙ(u), v; markers = (:xmax,)), 6)

@assert r∂ₙ_min[[2, 7]] ≈ [-1 / (2h₂), 1 / (2h₂)]     # -(u₂ - u₀) / (2h₂)
@assert r∂ₙ_max[[5, 8]] ≈ [-1 / (2hₙ), 1 / (2hₙ)]     # (u₇ - u₅) / (2hₙ)

r∂ₙ_min, r∂ₙ_max
```

```@example operators
r∂ₙₙ_min = row((u, v) -> inner_Γ(∂ₙₙ(u), v; markers = (:xmin,)), 1)
r∂ₙₙ_max = row((u, v) -> inner_Γ(∂ₙₙ(u), v; markers = (:xmax,)), 6)
rjump_min = row((u, v) -> inner_Γ(jumpₙ(u), v; markers = (:xmin,)), 1)
rjump_max = row((u, v) -> inner_Γ(jumpₙ(u), v; markers = (:xmax,)), 6)

@assert r∂ₙₙ_min[[1, 2, 7]] ≈ [-2, 1, 1] ./ h₂^2      # (u₂ - 2u₁ + u₀) / h₂²
@assert r∂ₙₙ_max[[5, 6, 8]] ≈ [1, -2, 1] ./ hₙ^2      # (u₅ - 2u₆ + u₇) / hₙ²
@assert rjump_min[[1, 2]] == [-1.0, 1.0] && rjump_min[[7, 8]] == [0.0, 0.0]
@assert rjump_max[[5, 6]] == [1.0, -1.0] && rjump_max[[7, 8]] == [0.0, 0.0]

r∂ₙₙ_min, rjump_max
```

The jump is the inward one at both ends, ``u_2 - u_1`` at the left and ``u_{n-1} - u_n`` at the right, which is why it needs no ghost of its own. The plain forward `jumpₓ` would read across the ghost cell at the right end. A facet operator takes scalars and sums, so `α * u + β * ∂ₙ(u)` is the Robin combination. It is not a grid function: `jumpₙ(∂ₙ(u))` throws an `ArgumentError` that begins "jumpₙ takes a field expression", a function coefficient times a facet operator is a `MethodError` (no method matching `*`), and the function belongs in the data of `inner_Γ` or of a relation.

A facet whose ghosts are missing is refused when the form is built, never read as zero:

```@example operators
try
    form(W_left, W_left, (u, v) -> inner_Γ(∂ₙ(u), v; markers = (:xmax,)))
catch e
    @assert e isa ArgumentError
    first(split(sprint(showerror, e), ". "))
end
```

### One value per incident facet

A corner lies on two facets. A node there has two normal derivatives, one per facet. Inside `inner_Γ`, a facet operator expands into one product per facet, each with its own facet weight and its own ghost; `markers` chooses the facets. Take the corner node of the non-uniform grid of the [space tutorial](@ref space_ghost), degree of freedom 1. The `xmin` ghost of its row is degree of freedom 13 and the `ymin` ghost of its column is 19:

```@example operators
Ω_c = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ_c = mesh(Ω_c, (4, 3), (true, true))
change_points!(Ωₕ_c, markers(Ω_c), ([0.0, 0.2, 0.6, 1.0], [0.0, 0.5, 1.0]))
W_c = gridspace(Ωₕ_c; ghosts = :boundary)

row_c(f) = Matrix(assemble(form(W_c, W_c, f)))[1, :]
r_x = row_c((u, v) -> inner_Γ(∂ₙ(u), v; markers = (:xmin,)))
r_y = row_c((u, v) -> inner_Γ(∂ₙ(u), v; markers = (:ymin,)))
r_xy = row_c((u, v) -> inner_Γ(∂ₙ(u), v; markers = (:xmin, :ymin)))

@assert r_x[13] != 0 && r_x[19] == 0         # the xmin term reads only the xmin ghost
@assert r_y[19] != 0 && r_y[13] == 0
@assert r_xy ≈ r_x + r_y

r_x[13], r_y[19]
```

The two entries differ, ``0.625`` and ``0.1``, because each facet term carries its facet's weight, and at the corner that weight is a half cell. A coefficient per facet is a sum over `markers`, one `inner_Γ` per facet, and the corner row follows:

```@example operators
r_c = row_c((u, v) -> inner_Γ(2.0 * ∂ₙ(u), v; markers = (:xmin,)) +
                      inner_Γ(5.0 * ∂ₙ(u), v; markers = (:ymin,)))

@assert r_c ≈ 2.0 * r_x + 5.0 * r_y

r_c[13], r_c[19]
```

The same holds for `∂ₙₙ` and `jumpₙ`. The conormal derivative of a constant tensor ``K``, `∂ₙ(u; K = K)`, is ``\eta_F\,(K\, D_c u)`` per facet. With a mixed entry ``K_{12}`` the `xmin` term also differences along ``y``, so at the corner it reads the `ymin` ghost too:

```@example operators
K = [1.0 0.5; 0.5 2.0]
r_K = row_c((u, v) -> inner_Γ(∂ₙ(u; K = K), v; markers = (:xmin,)))

@assert r_K[19] != 0 && r_K[13] != 0

r_K[13], r_K[19]
```

Two relations that couple two ghosts at a corner are what [`constrained`](@ref) solves as a ``2 \times 2`` block. A conormal term is refused where the ghosts of the neighbouring facet at the corner are missing, and ``K`` must be a constant matrix of numbers.

### The Laplacian inside a form

On trial and test functions, [`Δₕ`](@ref) is one operator: the sum over the axes of the second difference ``\tilde D_d D_{-d}``. It assembles to the matrix of the spelled-out sum, on a space without ghosts and on a ghost space:

```@example operators
W_p = gridspace(Ωₕ_c)
lap(W) = assemble(form(W, W, (u, v) -> innerₕ(Δₕ(u), v)))
spelled(W) = assemble(form(W, W, (u, v) -> innerₕ(D̃ₓ(D₋ₓ(u)) + D̃ᵧ(D₋ᵧ(u)), v)))

@assert norm(lap(W_p) - spelled(W_p)) <= 1e-13 * norm(lap(W_p))
@assert norm(lap(W_c) - spelled(W_c)) <= 1e-13 * norm(lap(W_c))
@assert norm(lap(W_c)) > 0

norm(lap(W_c) - spelled(W_c)) / norm(lap(W_c))
```

At a facet node on a ghost space it is the ordinary three-point stencil with the ghost as a neighbour. On `W₁`, the row of the first node is the cell weight times ``(u_2 - 2u_1 + u_0)/h_2^2``:

```@example operators
r_lap = row((u, v) -> innerₕ(Δₕ(u), v), 1)
ĥ₁ = h₂ / 2

@assert r_lap[[1, 2, 7]] ≈ ĥ₁ .* [-2, 1, 1] ./ h₂^2

r_lap
```

It nests with the other operators, `jumpₓ(Δₕ(u))` or `Dcₓ(Δₕ(u))`, and with itself.

### Mixed differences at a corner

The mixed difference ``D_{c,x} D_{c,y}`` at a corner node would read the padded corner point, which is absent. On a ghost space the stencil there is the inward one-sided difference along both axes, which is second order and reads only nodes. At the corner node of the grid above, with ``h_x = 0.2`` and ``h_y = 0.5``, the row is the cell weight ``h_x h_y / 4`` over ``h_x h_y`` times ``+1, -1, -1, +1`` on the nodes ``(1,1), (2,1), (1,2), (2,2)``:

```@example operators
r_mixed = row_c((u, v) -> innerₕ(Dcₓ(Dcᵧ(u)), v))

@assert r_mixed[[1, 2, 5, 6]] ≈ 0.25 .* [1, -1, -1, 1]
@assert count(!iszero, r_mixed) == 4

r_mixed[[1, 2, 5, 6]]
```

Every other mixed difference that would read an absent point is refused with an `ArgumentError`. Nested mixed differences on a ghost space are assembled as forms, as here. The runtime operators of the sections above are eager: `Dcᵧ(uₕ)` returns a finished grid function before `Dcₓ` sees it, so on a lifted grid function the composition at the ghosts next to a corner is not claimed to equal the form.

---

## Reference

### [How the names are built](@id operators_names)

A name is a stem, a direction, and a coordinate:

| Piece | Meaning |
|:--|:--|
| `D` | finite difference |
| `jump` | jump |
| `M` | average |
| `S` | index shift |
| `₋` | backward: the stencil reaches to ``i-1`` |
| `₊` | forward: the stencil reaches to ``i+1`` |
| `ₓ`, `ᵧ`, `₂` | along the first, second or third coordinate |
| `ₕ` | every coordinate at once, returning a tuple |

So `D₋ₓ` is the backward finite difference along ``x``, `M₊ᵧ` the forward average along ``y``, and `∇ₕ` the backward finite difference in every coordinate, the discrete gradient, which has that extra name. `jump` takes no direction, for the reason given above: it is `jumpₓ`, `jumpᵧ`, `jump₂` and `jumpₕ`. The index shift has a forward and a backward stem: `S₊ₓ` reads the next point along ``x``, `S₋ₓ` the previous one, and `S₊ₕ`/`S₋ₕ` shift along every coordinate.

The forward difference and the forward average are `public` but not exported. Bramble discretises with the backward operator paired with `inner₊`, so the forward ones are the duals to check against, not the ones to write a form with. This page imports them by name.

### Four differences compared

Four families combine the same one-sided differences in different ways:

| Operator | Stencil | Order, non-uniform | Order, uniform | Boundary |
|:--|:--|:--|:--|:--|
| `D₋` | ``\{i-1, i\}`` | 1 | 1 | truncated to `0` at the first point |
| `Dc` | ``\{i-1, i+1\}``, divided by the full span | 1 | 2 | truncated to `0` at both ends |
| `D̽` | ``\{i-1, i, i+1\}``, a weighted mean of two backward differences | 2 | 2 | no convention of its own: falls back to `D₊`/`D₋` |
| `D̃` | ``\{i, i+1\}``, divided by the averaged spacing | 1 on smoothly graded grids, not consistent where neighbouring spacings differ by an O(1) ratio (random grids); summation by parts holds exactly | 1 | truncated to `0` at the last point, like `D₊` |

On a uniform grid ``h_i = h_{i+1}``: `D̽` and `Dc` collapse to the mean of `D₋` and `D₊`, and `D̃` collapses to `D₊`. They separate only where the spacing varies, which is why a uniform-grid benchmark cannot tell them apart. `D̃` is meant to be used paired with `innerₕ`, for the summation by parts, and not as a pointwise approximation of the derivative on a random grid.

### [The direction as an argument](@id operators_direction_argument)

The direction can also come from a variable rather than from the name. Index the vectorial operator with it: `∇ₕ[d]` is the coordinate operator itself, so `∇ₕ[1]`, `∇ₕ[:x]` and `D₋ₓ` are the same function object, and `dx, dy = ∇ₕ` destructures it. The same holds for `∇̃ₕ`, `∇cₕ`, `∇̽ₕ`, `jumpₕ`, `Mₕ` and `Mcₕ`.

That makes a loop over directions writable, which the subscript names cannot express on their own. The discrete ``H^1`` seminorm squared, in any dimension, with the `inner₊` weights of the backward gradient, next to `snorm₁ₕ`:

```@example operators
sum(inner₊(∇ₕ[d](vₕ), ∇ₕ[d](vₕ), d) for d in 1:dim(mesh(space(vₕ)))), snorm₁ₕ(vₕ)^2
```

Underneath, every family has a stem that takes the direction as a second argument, and the coordinate suffix is spelling it:

| Written | Same as |
|:--|:--|
| `Bramble.D₋(uₕ, 1)` | `D₋ₓ(uₕ)` |
| `Bramble.D₋(uₕ, :y)` | `D₋ᵧ(uₕ)` |
| `Bramble.D₋(uₕ, Val(3))` | `D₋₂(uₕ)` |

The stems `D₋`, `D₊`, `Dc`, `D̃` and `jump` are `public` but not exported, so they are written `Bramble.D₋` or imported by name. The `Val` form is the one to use when the direction must be a compile-time constant, as it must inside a form. The averages use `Mₕ`/`M₊ₕ` rather than a bare `M`, because `M` is what most finite-element code calls its mass matrix. So `Mₕ(uₕ)` is the tuple over every coordinate and `Mₕ(uₕ, 2)` is the average along ``y``: the same name, told apart by how many arguments it is given. `∇̽ₕ` works the same way:

```@example operators
∇̽ₕ(vₕ) == (∇̽ₕ(vₕ, 1), ∇̽ₕ(vₕ, 2)), Mₕ(vₕ) == (Mₕ(vₕ, :x), Mₕ(vₕ, :y))
```

An `Int` or a `Symbol` selects between literal `Val`s, one branch per direction the mesh has. An out-of-range direction throws an `ArgumentError`, and the bound is the mesh's own dimension: `Bramble.D₋(uₕ, :y)` on a one-dimensional grid is an error, not a silent zero.

The dimensional entry points are separate names from the vectorial ones, on purpose. `∇ₕ(uₕ)` returns a tuple where `Bramble.D₋(uₕ, d)` returns a grid function, so folding them together would make the return type depend on whether an argument was passed. `∇̽ₕ` and `Mₕ`/`M₊ₕ` can share a name because the two meanings differ by arity, not by the value of an argument.

### Destructuring and composing the operator itself

`∇ₕ` and every other `ₕ`-suffixed family is a tuple-like object of the per-coordinate operators, so it can be destructured or indexed before it is applied. Here `dx` is exactly `D₋ₓ`, reached without importing that name:

```@example operators
dx, dy = ∇ₕ

parent(dx(vₕ)) == parent(∇ₕ(vₕ)[1]), parent(∇ₕ[:x](vₕ)) == parent(∇ₕ(vₕ)[1])
```

`∇cₕ`, `∇̽ₕ`, `∇̃ₕ` and `∇₊ₕ` answer to the same three spellings (destructuring, `[d]` indexing, `⋅`/`×`), each contracting to its own family's `div` and `curl`: `divcₕ`/`curlcₕ`, `div̽ₕ`/`curl̽ₕ`, `diṽₕ`/`curl̃ₕ` and `div₊ₕ`/`curl₊ₕ`.
