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

# Extract component views
u_x = uₕ(1)
u_y = uₕ(2)

# Alternative named accessor
u_x = component(uₕ, 1)
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
u_x .= 1.5
u_y .= -2.0

# The parent vector reflects the updates immediately
values(uₕ)
```

For scalar spaces, `uₕ(1)` or `component(uₕ, 1)` cleanly returns `uₕ` itself.

### Tuple destructuring

All components can be extracted simultaneously as a tuple using `components`:

```julia
u_x, u_y = components(uₕ)
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
# mats is a Tuple containing (to_matrix(u_x), to_matrix(u_y))

size(mats[1])  # (5, 5)
size(mats[2])  # (5, 5)
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
