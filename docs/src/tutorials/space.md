# Grid spaces and discrete functions

Discrete function spaces in Bramble connect continuous mathematical functions to finite-dimensional discrete degrees of freedom defined over computational meshes. 

This tutorial introduces:
- **Scalar grid spaces** defined on discrete meshes
- **Multi-component composite spaces** for vector and tensor fields
- **Vector elements** representing discrete grid functions
- **Component indexing** and zero-copy field views
- **Logical grid layouts** via reshaped matrix views
- **Nodal projection and restriction** ($R_h$)
- **Cell averaging operators** ($\mathrm{avg}_h$)
- **Discrete inner products and norms** (`innerₕ`, `normₕ`, `norm₁ₕ`, `snorm₁ₕ`)

---

## 1. Constructing scalar grid spaces

A scalar grid space represents discrete scalar fields over a mesh. To create a scalar space, call `gridspace` on a mesh $\Omega_h$:

```julia
using Bramble

# 1. Define a 2D domain: [0, 1] × [0, 1]
Ω = domain(box((0.0, 0.0), (1.0, 1.0)))

# 2. Discretize into a uniform 5 × 5 mesh with periodic boundary conditions
Ωₕ = mesh(Ω, (5, 5), (true, true))

# 3. Construct a scalar grid space on the mesh
Wₕ = gridspace(Ωₕ)
```

The resulting space `Wₕ` is an instance of `ScalarGridSpace`.

### Degrees of freedom and quadrature weights

The number of degrees of freedom in `Wₕ` corresponds to the total number of grid points in the mesh:

```julia
ndofs(Wₕ)  # returns 25 (5 * 5)
```

To perform numerical integration and compute inner products, each degree of freedom has an associated quadrature weight given by the cell measure around that point:

```julia
w = weights(Wₕ)
length(w)  # 25
```

---

## 2. Multi-component composite spaces

Many physical problems involve vector-valued quantities such as velocities $\mathbf{u} = (u_x, u_y)$, displacement fields, or coupled state variables. In Bramble, multi-component spaces are represented by `CompositeGridSpace`.

### Constructing vector spaces with power notation

The simplest way to create a vector grid space of dimension $D$ is using exponentiation:

```julia
# A 2-component vector space (e.g., 2D velocity space)
Vₕ = Wₕ^2
```

Alternatively, `vector_gridspace` can be constructed directly from a mesh:

```julia
Vₕ = vector_gridspace(Ωₕ, 2)
```

### Inspecting composite spaces

```julia
ncomponents(Vₕ)  # 2
ndofs(Vₕ)        # 50 (2 * 25)
spaces(Vₕ)       # (Wₕ, Wₕ)
```

`ndofs(Vₕ, Tuple)` also works, but means something different here than it did for `Wₕ`
above: on a `ScalarGridSpace` it is the grid's shape, one entry per spatial dimension
(`Nₓ`, `Nᵧ`); on a `CompositeGridSpace` it is one entry per component instead — `(25, 25)`
for `Vₕ`, not a shape. `weights`, by contrast, is not defined at all for a
`CompositeGridSpace`: its components can sit on different meshes, so there is no single
weight vector to hand back for the whole space — call `weights` on a `components(Vₕ)`
leaf instead.

Composite spaces can also be constructed from distinct constituent spaces:

```julia
V_custom = CompositeGridSpace((Wₕ, Wₕ))
```

---

## 3. Vector elements and grid functions

A `VectorElement` represents a discrete field in a given grid space. It wraps a coefficient vector together with a reference to its parent function space.

### Instantiating elements

You can instantiate uninitialized elements, elements filled with a constant, or wrap existing coefficients:

```julia
# Uninitialized vector element
uₕ = element(Wₕ)

# Element initialized to a constant value
u_zero = element(Wₕ, 0.0)
u_ones = element(Wₕ, 1.0)
```

### Array operations and broadcasting

Because `VectorElement <: AbstractVector`, it supports standard vector indexing, length queries, and arithmetic:

```julia
uₕ[1] = 42.0
length(uₕ)  # 25

# Broadcasting preserves the parent space without unnecessary allocations
vₕ = element(Wₕ, 2.0)
wₕ = 3.0 .* uₕ .+ vₕ
```

---

## 4. Component indexing and field extraction

When working with vector fields in a `CompositeGridSpace`, you often need to inspect or manipulate individual physical components (such as velocity in the $x$ or $y$ direction).

### Functor call syntax and component views

Calling a vector element as a function `uₕ(i)` or using `component(uₕ, i)` returns a `VectorElement` representing the $i$-th component:

```julia
uₕ = element(Vₕ)

# Extract component views using coordinate subscripts ("ₓ", "ᵧ", "₂")
uₓ = uₕ(1)
uᵧ = uₕ(2)

# Alternative named accessor
uₓ = component(uₕ, 1)
```

### Degree-of-freedom ranges

To retrieve the degree-of-freedom index ranges occupied by components in the underlying flat vector, use `component_range` or `component_ranges`:

```julia
# Range of component 1: 1:25
rng1 = component_range(Vₕ, 1)

# All component ranges as a tuple: (1:25, 26:50)
rngs = component_ranges(Vₕ)
```

### Zero-copy view semantics

Component extraction uses zero-copy array views into the parent degree-of-freedom vector. Modifying a component modifies the parent element in-place:

```julia
# Assign values directly to components
uₓ .= 1.5
uᵧ .= -2.0

# The parent vector reflects the updates immediately
values(uₕ)
```

For scalar spaces, `uₕ(1)` or `component(uₕ, 1)` cleanly returns `uₕ` itself.

### Tuple destructuring

All components can be extracted simultaneously as a tuple using `components`:

```julia
uₓ, uᵧ = components(uₕ)
# or using numeric index subscripts:
u₁, u₂ = components(uₕ)
```

---

## 5. Logical grid layouts and reshaped matrix views

While degrees of freedom are stored internally as flat 1D vectors for linear algebra operations, finite difference stencils and visualization require indexing points in physical grid dimensions.

The `to_matrix` function reshapes the flat coefficient vector into a multidimensional array matching the mesh geometry:

### Scalar elements

```julia
u_scal = element(Wₕ, 0.0)
u_grid = to_matrix(u_scal)
size(u_grid)  # (5, 5)

# Access value at grid point (i, j)
u_grid[2, 3] = 10.0
```

Because `to_matrix` returns a `Base.ReshapedArray` view of the underlying vector, mutating `u_grid` modifies `u_scal` in-place with zero memory allocation.

### Multi-component elements

For multi-component vector elements, `to_matrix` returns a tuple of reshaped arrays, one for each component:

```julia
mats = to_matrix(uₕ)
# mats is a Tuple containing (to_matrix(uₓ), to_matrix(uᵧ))

size(mats[1])  # (5, 5)
size(mats[2])  # (5, 5)
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
  <text x="215" y="93" font-size="11" fill="#3b82f6">uₕ(1) / component(uₕ, 1)</text>

  <path d="M 565 71 L 565 105" stroke="#8b5cf6" stroke-width="1.5" fill="none" marker-end="url(#arrowPurple)"/>
  <text x="575" y="93" font-size="11" fill="#8b5cf6">uₕ(2) / component(uₕ, 2)</text>

  <!-- 2. Zero-copy component views -->
  <rect x="30" y="110" width="350" height="34" rx="4" fill="#3b82f6" fill-opacity="0.08" stroke="#3b82f6" stroke-dasharray="4,3" stroke-width="1.2"/>
  <text x="205" y="132" font-size="12" fill="currentColor" text-anchor="middle">VectorElement view (SubArray of length 25)</text>

  <rect x="390" y="110" width="350" height="34" rx="4" fill="#8b5cf6" fill-opacity="0.08" stroke="#8b5cf6" stroke-dasharray="4,3" stroke-width="1.2"/>
  <text x="565" y="132" font-size="12" fill="currentColor" text-anchor="middle">VectorElement view (SubArray of length 25)</text>

  <!-- Connectors from Component Views to 2D Grids -->
  <path d="M 205 144 L 205 178" stroke="currentColor" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
  <text x="215" y="166" font-size="11" fill="currentColor">to_matrix(uₓ)</text>

  <path d="M 565 144 L 565 178" stroke="currentColor" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
  <text x="575" y="166" font-size="11" fill="currentColor">to_matrix(uᵧ)</text>

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

## 6. Nodal restriction and projection

The nodal restriction operator $R_h$ evaluates a continuous function $f(x)$ at the discrete grid points of a mesh and stores the resulting values in a `VectorElement`.

### Projecting scalar functions

```julia
# Define a continuous function of spatial coordinates x = (x₁, x₂)
f(x) = sin(2π * x[1]) * cos(2π * x[2])

# Allocate and project
u_proj = Rₕ(Wₕ, f)

# In-place projection into an existing element
Rₕ!(u_proj, f)
```

### Projecting vector-valued functions

For multi-component spaces, $R_h$ accepts either a tuple of scalar functions or a vector function:

```julia
# Tuple of coordinate functions
fx(x) = x[1]
fy(x) = 2 * x[2]

Rₕ!(uₕ, (fx, fy))

# Or a function returning a tuple/vector
f_vel(x) = (sin(x[1]), cos(x[2]))
Rₕ!(uₕ, f_vel)
```

---

## 7. Numerical cell averaging

When discretizing conservation laws or finite volume formulations, quantities often represent cell averages rather than pointwise values. 

The cell averaging operator $\mathrm{avg}_h$ integrates a function $f$ over each computational cell $\square_i$ around grid point $x_i$, normalized by the cell volume $|\square_i|$:

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

In Bramble, $\mathrm{avg}_h$ uses a tensor-product Gauss-Legendre rule (by default `AVG_QUAD_POINTS = 6`, exact for polynomials up to degree eleven):

```julia
# Compute cell-averaged element
u_avg = avgₕ(Wₕ, x -> exp(-x[1] - x[2]))

# In-place version
avgₕ!(u_avg, x -> exp(-x[1] - x[2]))
```

For multi-component spaces, averages can likewise be computed across components:

```julia
vₕ = avgₕ(Vₕ, (x -> 1.0, x -> 2.0 * x[1]))
```

---

## 7. Discrete inner products and norms

In continuous analysis, function spaces like ``L^2(\Omega)`` and ``H^1(\Omega)`` are equipped with inner products and norms:
```math
(u, v)_{L^2} = \int_\Omega u(x) v(x) \, dx, \quad \|u\|_{L^2} = \sqrt{(u, u)_{L^2}}, \quad |u|_{H^1}^2 = \int_\Omega |\nabla u|^2 \, dx.
```

In Bramble, discrete functions in a `ScalarGridSpace` or `CompositeGridSpace` have direct discrete counterparts that weight grid values by cell measures and quadrature weights.

### The discrete ``L^2`` inner product and norm

The primary discrete inner product is `innerₕ(uₕ, vₕ)`. It weights each point by its cell measure ``w_i = |\square_i|``:
```math
(u_h, v_h)_h = \sum_i w_i \, u_h(x_i) v_h(x_i).
```

The discrete ``L^2`` norm `normₕ(uₕ)` is induced by `innerₕ`:
```math
\|u_h\|_h = \sqrt{(u_h, u_h)_h}.
```

```julia
using Bramble

Ωₕ = mesh(domain(interval(0.0, 1.0)), 100, true)
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, sin)
vₕ = Rₕ(Wₕ, cos)

# Discrete L2 inner product and norm
innerₕ(uₕ, vₕ)
normₕ(uₕ)

# Exact norm identity: ‖uₕ‖ₕ² == (uₕ, uₕ)ₕ
normₕ(uₕ)^2 ≈ innerₕ(uₕ, uₕ)
```

### Discrete Sobolev norms: ``H^1`` seminorm and full ``H^1`` norm

Bramble provides discrete ``H^1`` Sobolev norms based on the forward discrete gradient ``\nabla_{+h}``:
* `snorm₁ₕ(uₕ)`: the discrete ``H^1`` seminorm ``|u_h|_{1,h}``, defined as:
  ```math
  |u_h|_{1,h}^2 = \|\nabla_{+h} u_h\|_h^2 = \sum_{d=1}^D \|D_{+x_d} u_h\|_h^2.
  ```
* `norm₁ₕ(uₕ)`: the full discrete ``H^1`` norm, satisfying the Pythagorean identity:
  ```math
  \|u_h\|_{1,h}^2 = \|u_h\|_h^2 + |u_h|_{1,h}^2.
  ```

```julia
# Discrete H¹ seminorm and full H¹ norm
snorm₁ₕ(uₕ)
norm₁ₕ(uₕ)

# Verification of identity
norm₁ₕ(uₕ)^2 ≈ normₕ(uₕ)^2 + snorm₁ₕ(uₕ)^2
```

### Inner products on composite (vector) spaces

For vector-valued grid functions in a `CompositeGridSpace` (such as velocities or gradients), `innerₕ` sums the discrete inner products across all components:
```math
(\mathbf{u}_h, \mathbf{v}_h)_h = \sum_{c=1}^{\mathrm{NC}} (u_{h,c}, v_{h,c})_h.
```

```julia
Vₕ = Wₕ^2
u_vec = Rₕ(Vₕ, (x -> sin(x), x -> cos(x)))

normₕ(u_vec)^2 ≈ normₕ(u_vec[1])^2 + normₕ(u_vec[2])^2
```

### Staggered weights and directional inner products

Energy estimates in finite difference schemes often balance flux differences against intermediate values at cell faces. Bramble provides staggered inner products:
* `inner₊(uₕ, vₕ)`: inner product using staggered forward weights.
* Coordinate-specific forms: `inner₊ₓ`, `inner₊ᵧ`, `inner₊₂`.

These staggered inner products form the exact algebraic pairing needed for summation by parts (discussed in detail in the [Difference, jump and average operators](operators.md) tutorial).

