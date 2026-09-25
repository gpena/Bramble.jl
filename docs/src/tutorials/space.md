```@meta
CurrentModule = Bramble
```

# Grid spaces and discrete functions

A grid space is the discrete function space over a mesh: it fixes how many degrees of
freedom a field has and which quadrature weight each one carries. A [`VectorElement`](@ref)
is a function in that space. Every block below runs when this page is built.

---

## 1. Scalar grid spaces

[`gridspace`](@ref) builds a `ScalarGridSpace` over a mesh:

```@example space
using Bramble
import Bramble: component_range, component_ranges, weights

Ω = domain(box((0.0, 0.0), (1.0, 1.0)))
Ωₕ = mesh(Ω, (5, 5), (true, true))    # uniform spacing along both axes
Wₕ = gridspace(Ωₕ)

ndofs(Wₕ), ndofs(Wₕ, Tuple)
```

`points(Wₕ)` (equal to `points(Ωₕ)`) answers to the same destructuring as the vectorial
operators: `x, y = points(Wₕ)` gives the two coordinate vectors directly, one array per
axis:

```@example space
x, y = points(Wₕ);
length(x), length(y)
```

`ndofs` counts the grid points; with `Tuple` it gives the grid's shape.

Each degree of freedom carries a quadrature weight, the cell measure around its point.
`weights(Wₕ)` returns the whole `SpaceWeights` bundle, and a weight vector comes from naming
the inner product it belongs to. `Bramble.Innerh` and `Bramble.Innerplus` are public but not
exported, hence the prefix:

```@example space
w = weights(Wₕ, Bramble.Innerh())

length(w), sum(w)      # one weight per point; they tile the unit square
```

---

## 2. Composite spaces

A `CompositeGridSpace` stacks copies of a space for a vector field or a coupled system:

```@example space
Vₕ = Wₕ^Val(2)

ncomponents(Vₕ), ndofs(Vₕ)
```

`Wₕ^Val(N)` is the spelling used throughout this manual: it is type stable whatever `N` is,
while `Wₕ^2` relies on the literal being constant-folded and only reaches `N ≤ 3`.
[`vector_gridspace`](@ref)`(Ωₕ, 2)` builds the same space straight from a mesh, and
`CompositeGridSpace((Wₕ, Wₕ))` builds one from spaces that need not be identical.

```@example space
ndofs(Vₕ, Tuple), spaces(Vₕ) == (Wₕ, Wₕ)
```

`ndofs(Vₕ, Tuple)` is one entry per component, not a grid shape. `weights` is not defined
for a composite space at all, since its leaves may live on different meshes: call it on a
[`components`](@ref) leaf instead.

---

## 3. Vector elements

[`element`](@ref) allocates a field, optionally filled with a constant:

```@example space
uₕ = element(Wₕ)         # uninitialized
u_zero = element(Wₕ, 0.0)

length(u_zero), u_zero[1]
```

`VectorElement <: AbstractVector`, so indexing, `length` and broadcasting work as usual, and
broadcasting keeps the parent space:

```@example space
vₕ = element(Wₕ, 2.0)
wₕ = 3.0 .* u_zero .+ vₕ

space(wₕ) === Wₕ, wₕ[1]
```

A plain `Function` is not a grid function, so multiplying by one restricts it first
(`Rₕ(space(uₕ), f)`) and scales pointwise. This is what lets a spatial condition multiply a
field directly, including inside a form:

```@example space
oneₕ = Rₕ(Wₕ, x -> 1.0)
below_half = (x -> x[1] < 0.5) * oneₕ

sum(below_half)
```

---

## 4. Components

Calling an element returns the `i`-th component as a view:

```@example space
uvec = element(Vₕ)
uₓ = uvec(1)
uᵧ = uvec(2)

uₓ .= 1.5
uᵧ .= -2.0

parent(uvec)[[1, 26]]
```

The component is a zero-copy view, so writing to it writes through to the parent. For a
scalar space `uₕ(1)` is `uₕ` itself. [`component_range`](@ref) and `component_ranges` give
the degrees of freedom each component owns, and [`components`](@ref) destructures the lot:

```@example space
component_range(Vₕ, 1), component_ranges(Vₕ)
```

```@example space
c₁, c₂ = components(uvec)

c₁ === uₓ, c₂ === uᵧ
```

---

## 5. Grid layout

Degrees of freedom are stored flat, but a scalar element also indexes by grid coordinate,
with a tuple or a `CartesianIndex`, without reshaping:

```@example space
u_scal = element(Wₕ, 0.0)
u_scal[2, 3] = 10.0
u_scal[CartesianIndex(4, 1)] = 20.0

u_scal[2, 3], u_scal[4, 1]
```

For matrix operations or plotting, `reshape(uₕ)` returns a `Base.ReshapedArray` view of the
same memory, so writing through it writes through to the element. On a composite element it
returns one view per component:

```@example space
u_grid = reshape(u_scal)
u_grid[2, 3] = -1.0

size(u_grid), u_scal[2, 3], size.(reshape(uvec))
```
```@raw html
<figure>
<svg viewBox="0 0 780 270" width="100%" style="max-width:780px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Diagram of CompositeGridSpace storage showing the contiguous flat degree of freedom buffer, zero-copy component views u_x and u_y, and 2D reshaped matrix views.">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="currentColor"/>
    </marker>
    <marker id="arrowBlue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#3b82f6"/>
    </marker>
    <marker id="arrowPurple" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#8b5cf6"/>
    </marker>
  </defs>

  <!-- 1. Flat 1D Buffer -->
  <text x="30" y="25" font-size="13" font-weight="bold" fill="currentColor">Flat 1D degree of freedom storage: uₕ.data (length 50)</text>

  <!-- Component 1 block -->
  <rect x="30" y="35" width="350" height="36" rx="4" fill="#3b82f6" fill-opacity="0.15" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="205" y="58" font-size="12" font-weight="bold" fill="#3b82f6" text-anchor="middle">Component 1 (uₓ): DOFs 1:25</text>

  <!-- Component 2 block -->
  <rect x="390" y="35" width="350" height="36" rx="4" fill="#8b5cf6" fill-opacity="0.15" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="565" y="58" font-size="12" font-weight="bold" fill="#8b5cf6" text-anchor="middle">Component 2 (uᵧ): DOFs 26:50</text>

  <!-- Connectors from Flat to Component Views -->
  <path d="M 205 71 L 205 105" stroke="#3b82f6" stroke-width="1.5" fill="none" marker-end="url(#arrowBlue)"/>
  <text x="215" y="93" font-size="11" fill="#3b82f6">uₕ(1)</text>

  <path d="M 565 71 L 565 105" stroke="#8b5cf6" stroke-width="1.5" fill="none" marker-end="url(#arrowPurple)"/>
  <text x="575" y="93" font-size="11" fill="#8b5cf6">uₕ(2)</text>

  <!-- 2. Zero-copy component views -->
  <rect x="30" y="110" width="350" height="34" rx="4" fill="#3b82f6" fill-opacity="0.08" stroke="#3b82f6" stroke-dasharray="4,3" stroke-width="1.2"/>
  <text x="205" y="132" font-size="12" fill="currentColor" text-anchor="middle">VectorElement view (SubArray of length 25)</text>

  <rect x="390" y="110" width="350" height="34" rx="4" fill="#8b5cf6" fill-opacity="0.08" stroke="#8b5cf6" stroke-dasharray="4,3" stroke-width="1.2"/>
  <text x="565" y="132" font-size="12" fill="currentColor" text-anchor="middle">VectorElement view (SubArray of length 25)</text>

  <!-- Connectors from Component Views to 2D Grids -->
  <path d="M 205 144 L 205 178" stroke="currentColor" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
  <text x="215" y="166" font-size="11" fill="currentColor">reshape(uₓ)</text>

  <path d="M 565 144 L 565 178" stroke="currentColor" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
  <text x="575" y="166" font-size="11" fill="currentColor">reshape(uᵧ)</text>

  <!-- 3. Reshaped 2D Matrix Views -->
  <g transform="translate(145, 185)">
    <rect x="0" y="0" width="120" height="65" rx="4" fill="none" stroke="#3b82f6" stroke-width="1.5"/>
    <line x1="24" y1="0" x2="24" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="48" y1="0" x2="48" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="72" y1="0" x2="72" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="96" y1="0" x2="96" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="13" x2="120" y2="13" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="26" x2="120" y2="26" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="39" x2="120" y2="39" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="52" x2="120" y2="52" stroke="currentColor" stroke-opacity="0.2"/>
    <text x="60" y="80" font-size="11" fill="currentColor" text-anchor="middle">5 × 5 ReshapedArray (uₓ[i, j])</text>
  </g>

  <g transform="translate(505, 185)">
    <rect x="0" y="0" width="120" height="65" rx="4" fill="none" stroke="#8b5cf6" stroke-width="1.5"/>
    <line x1="24" y1="0" x2="24" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="48" y1="0" x2="48" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="72" y1="0" x2="72" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="96" y1="0" x2="96" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="13" x2="120" y2="13" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="26" x2="120" y2="26" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="39" x2="120" y2="39" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="52" x2="120" y2="52" stroke="currentColor" stroke-opacity="0.2"/>
    <text x="60" y="80" font-size="11" fill="currentColor" text-anchor="middle">5 × 5 ReshapedArray (uᵧ[i, j])</text>
  </g>
</svg>
</figure>
```

---

## 6. Restriction and cell averaging

Two ways to turn a continuous function into a grid function. Nodal restriction $R_h$
evaluates it at the grid points; cell averaging $\mathrm{avg}_h$ integrates it over the cell
around each point and divides by the cell measure:

```math
\mathrm{avg}_h f(x_i) = \frac{1}{|\square_i|} \int_{\square_i} f(x) \, dx
```
```@raw html
<figure>
<svg viewBox="0 0 780 275" width="100%" style="max-width:780px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Comparison of nodal restriction R_h which samples f at point x_i versus cell averaging avg_h which integrates f over the dual cell using a 6-point Gauss-Legendre quadrature rule.">
  <!-- Panel 1: Nodal restriction R_h -->
  <g transform="translate(20, 10)">
    <rect x="0" y="0" width="355" height="255" rx="6" fill="none" stroke="currentColor" stroke-opacity="0.2" stroke-width="1"/>
    <text x="177" y="28" font-size="14" font-weight="bold" fill="currentColor" text-anchor="middle">Nodal restriction: Rₕ(Wₕ, f)</text>
    <text x="177" y="48" font-size="12" fill="currentColor" opacity="0.85" text-anchor="middle">Pointwise evaluation: Rₕ f(xᵢ) = f(xᵢ)</text>

    <!-- Function curve -->
    <path d="M 40 145 Q 177 75 315 115" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="300" y="105" font-size="11" fill="currentColor" font-style="italic">f(x)</text>

    <!-- Grid line & cell -->
    <line x1="30" y1="190" x2="325" y2="190" stroke="currentColor" stroke-width="1.5"/>
    <!-- Cell bounds -->
    <line x1="90" y1="175" x2="90" y2="205" stroke="#8b5cf6" stroke-width="1.5" stroke-dasharray="4,3"/>
    <line x1="265" y1="175" x2="265" y2="205" stroke="#8b5cf6" stroke-width="1.5" stroke-dasharray="4,3"/>
    <text x="90" y="222" font-size="11" fill="#8b5cf6" text-anchor="middle">xᵢ₋½</text>
    <text x="265" y="222" font-size="11" fill="#8b5cf6" text-anchor="middle">xᵢ₊½</text>

    <!-- Center point x_i -->
    <circle cx="177" cy="190" r="5" fill="#3b82f6"/>
    <text x="177" y="222" font-size="12" font-weight="bold" fill="#3b82f6" text-anchor="middle">xᵢ</text>

    <!-- Pointwise sample -->
    <line x1="177" y1="185" x2="177" y2="100" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="2,2"/>
    <circle cx="177" cy="96" r="5" fill="#3b82f6" stroke="currentColor" stroke-width="1"/>
    <text x="187" y="92" font-size="11" font-weight="bold" fill="#3b82f6">f(xᵢ)</text>

    <text x="177" y="246" font-size="11" fill="currentColor" opacity="0.8" text-anchor="middle">Single evaluation at node xᵢ</text>
  </g>

  <!-- Panel 2: Cell averaging avg_h -->
  <g transform="translate(405, 10)">
    <rect x="0" y="0" width="355" height="255" rx="6" fill="none" stroke="currentColor" stroke-opacity="0.2" stroke-width="1"/>
    <text x="177" y="28" font-size="14" font-weight="bold" fill="currentColor" text-anchor="middle">Cell averaging: avgₕ(Wₕ, f)</text>
    <text x="177" y="48" font-size="12" fill="currentColor" opacity="0.85" text-anchor="middle">Integral mean: avgₕ f(xᵢ) = |□ᵢ|⁻¹ ∫_{□ᵢ} f(x) dx</text>

    <!-- Shaded area under curve across the cell -->
    <path d="M 90 190 L 90 134 Q 177 75 265 106 L 265 190 Z" fill="#10b981" fill-opacity="0.15"/>

    <!-- Function curve -->
    <path d="M 40 145 Q 177 75 315 115" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="300" y="105" font-size="11" fill="currentColor" font-style="italic">f(x)</text>

    <!-- Grid line & cell -->
    <line x1="30" y1="190" x2="325" y2="190" stroke="currentColor" stroke-width="1.5"/>
    <!-- Cell bounds -->
    <line x1="90" y1="175" x2="90" y2="205" stroke="#8b5cf6" stroke-width="1.5" stroke-dasharray="4,3"/>
    <line x1="265" y1="175" x2="265" y2="205" stroke="#8b5cf6" stroke-width="1.5" stroke-dasharray="4,3"/>
    <text x="90" y="222" font-size="11" fill="#8b5cf6" text-anchor="middle">xᵢ₋½</text>
    <text x="265" y="222" font-size="11" fill="#8b5cf6" text-anchor="middle">xᵢ₊½</text>

    <!-- Center point x_i -->
    <circle cx="177" cy="190" r="4" fill="currentColor"/>
    <text x="177" y="222" font-size="12" fill="currentColor" text-anchor="middle">xᵢ</text>

    <!-- Gauss quadrature points inside cell (6 points) -->
    <circle cx="96"  cy="190" r="3" fill="#10b981"/>
    <circle cx="120" cy="190" r="3" fill="#10b981"/>
    <circle cx="156" cy="190" r="3" fill="#10b981"/>
    <circle cx="198" cy="190" r="3" fill="#10b981"/>
    <circle cx="235" cy="190" r="3" fill="#10b981"/>
    <circle cx="259" cy="190" r="3" fill="#10b981"/>

    <path d="M 120 185 L 120 128" stroke="#10b981" stroke-width="1" stroke-dasharray="2,2"/>
    <path d="M 198 185 L 198 87"  stroke="#10b981" stroke-width="1" stroke-dasharray="2,2"/>
    <path d="M 235 185 L 235 98"  stroke="#10b981" stroke-width="1" stroke-dasharray="2,2"/>

    <text x="177" y="246" font-size="11" fill="#10b981" font-weight="bold" text-anchor="middle">6-point Gauss-Legendre quadrature (N_q = 6)</text>
  </g>
</svg>
</figure>
```

[`Rₕ`](@ref) allocates, [`Rₕ!`](@ref) writes into an element that already exists, which is
what a time loop wants:

```@example space
f(x) = sin(2π * x[1]) * cos(2π * x[2])

u_proj = Rₕ(Wₕ, f)
Rₕ!(u_proj, f)

u_proj[1]
```

On a composite space both take a tuple of scalar functions, or one function returning a
tuple:

```@example space
Rₕ!(uvec, (x -> x[1], x -> 2 * x[2]))
Rₕ!(uvec, x -> (sin(x[1]), cos(x[2])))

uvec(1)[1], uvec(2)[1]
```

[`avgₕ`](@ref) and [`avgₕ!`](@ref) mirror them, using a tensor-product Gauss-Legendre rule
(`AVG_QUAD_POINTS = 6`, exact through degree eleven):

```@example space
u_avg = avgₕ(Wₕ, x -> exp(-x[1] - x[2]))
avgₕ!(u_avg, x -> exp(-x[1] - x[2]))

u_avg[1]
```

The difference matters for a source term: `Rₕ` of a rough function inherits its roughness,
while `avgₕ` integrates it away.

---

## 7. Inner products and norms

`innerₕ` weights each point by its cell measure, and `normₕ` is the norm it induces:

```math
(u_h, v_h)_h = \sum_i |\square_i| \, u_h(x_i) v_h(x_i), \qquad
\|u_h\|_h = \sqrt{(u_h, u_h)_h}.
```

```@example space
Ω₁ = domain(interval(0.0, 1.0))
W₁ = gridspace(mesh(Ω₁, 100, true))
s = Rₕ(W₁, sin)
c = Rₕ(W₁, cos)

innerₕ(s, c), normₕ(s), normₕ(s)^2 ≈ innerₕ(s, s)
```

The discrete Sobolev norms are built on the backward gradient $\nabla_{-h}$:

```math
|u_h|_{1,h}^2 = \sum_{d=1}^D \|D_{-x_d} u_h\|_h^2, \qquad
\|u_h\|_{1,h}^2 = \|u_h\|_h^2 + |u_h|_{1,h}^2.
```

```@example space
snorm₁ₕ(s), norm₁ₕ(s), norm₁ₕ(s)^2 ≈ normₕ(s)^2 + snorm₁ₕ(s)^2
```

On a composite space `innerₕ` sums over components:

```@example space
V₁ = W₁^Val(2)
u_vec = Rₕ(V₁, (x -> sin(x[1]), x -> cos(x[1])))

normₕ(u_vec)^2 ≈ normₕ(u_vec(1))^2 + normₕ(u_vec(2))^2
```

`inner₊` is the staggered counterpart: it weights by the half-spacings, the interface
quantities a difference or a gradient lands on, and a trailing `:x`/`:y`/`:z` (or `1`/`2`/`3`)
argument picks a single direction. That pairing is what makes summation by parts exact; the
[operators tutorial](operators.md) derives it.

```@example space
D = ∇ₕ(s)

inner₊(D, D, :x) ≈ snorm₁ₕ(s)^2
```

Next: [difference operators](operators.md), which act on the elements built here.
