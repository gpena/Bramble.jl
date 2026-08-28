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

A scalar grid space represents discrete scalar fields over a mesh. To create a scalar space, call `gridspace` on a mesh:

```julia
using Bramble

# 1. Define a 2D domain: [0, 1] × [0, 1]
Ω = domain(box((0.0, 0.0), (1.0, 1.0)))

# 2. Discretize into a uniform 5 × 5 mesh with periodic boundary conditions
m = mesh(Ω, (5, 5), (true, true))

# 3. Construct a scalar grid space on the mesh
W = gridspace(m)
```

The resulting space `W` is an instance of `ScalarGridSpace`.

### Degrees of freedom and quadrature weights

The number of degrees of freedom in `W` corresponds to the total number of grid points in the mesh:

```julia
ndofs(W)  # returns 25 (5 * 5)
```

To perform numerical integration and compute inner products, each degree of freedom has an associated quadrature weight given by the cell measure around that point:

```julia
w = weights(W)
length(w)  # 25
```

---

## 2. Multi-component composite spaces

Many physical problems involve vector-valued quantities such as velocities $\mathbf{u} = (u_x, u_y)$, displacement fields, or coupled state variables. In Bramble, multi-component spaces are represented by `CompositeGridSpace`.

### Constructing vector spaces with power notation

The simplest way to create a vector grid space of dimension $D$ is using exponentiation:

```julia
# A 2-component vector space (e.g., 2D velocity space)
V = W^2
```

Alternatively, `vector_gridspace` can be constructed directly from a mesh:

```julia
V = vector_gridspace(m, 2)
```

### Inspecting composite spaces

```julia
ncomponents(V)  # 2
ndofs(V)        # 50 (2 * 25)
spaces(V)       # (W, W)
```

Composite spaces can also be constructed from distinct constituent spaces:

```julia
V_custom = CompositeGridSpace((W, W))
```

---

## 3. Vector elements and grid functions

A `VectorElement` represents a discrete field in a given grid space. It wraps a coefficient vector together with a reference to its parent function space.

### Instantiating elements

You can instantiate uninitialized elements, elements filled with a constant, or wrap existing coefficients:

```julia
# Uninitialized vector element
u = element(W)

# Element initialized to a constant value
u_zero = element(W, 0.0)
u_ones = element(W, 1.0)
```

### Array operations and broadcasting

Because `VectorElement <: AbstractVector`, it supports standard vector indexing, length queries, and arithmetic:

```julia
u[1] = 42.0
length(u)  # 25

# Broadcasting preserves the parent space without unnecessary allocations
v = element(W, 2.0)
w = 3.0 .* u .+ v
```

---

## 4. Component indexing and field extraction

When working with vector fields in a `CompositeGridSpace`, you often need to inspect or manipulate individual physical components (such as velocity in the $x$ or $y$ direction).

### Functor call syntax and component views

Calling a vector element as a function `u(i)` or using `component(u, i)` returns a `VectorElement` representing the $i$-th component:

```julia
u_vec = element(V)

# Extract component views
u_x = u_vec(1)
u_y = u_vec(2)

# Alternative named accessor
u_x = component(u_vec, 1)
```

### Zero-copy view semantics

Component extraction uses zero-copy array views into the parent degree-of-freedom vector. Modifying a component modifies the parent element in-place:

```julia
# Assign values directly to components
u_x .= 1.5
u_y .= -2.0

# The parent vector reflects the updates immediately
values(u_vec)
```

For scalar spaces, `u(1)` or `component(u, 1)` cleanly returns `u` itself.

### Tuple destructuring

All components can be extracted simultaneously as a tuple using `components`:

```julia
u_x, u_y = components(u_vec)
```

---

## 5. Logical grid layouts and reshaped matrix views

While degrees of freedom are stored internally as flat 1D vectors for linear algebra operations, finite difference stencils and visualization require indexing points in physical grid dimensions.

The `to_matrix` function reshapes the flat coefficient vector into a multidimensional array matching the mesh geometry:

### Scalar elements

```julia
u_grid = to_matrix(u)
size(u_grid)  # (5, 5)

# Access value at grid point (i, j)
u_grid[2, 3] = 10.0
```

Because `to_matrix` returns a `Base.ReshapedArray` view of the underlying vector, mutating `u_grid` modifies `u` in-place with zero memory allocation.

### Multi-component elements

For multi-component vector elements, `to_matrix` returns a tuple of reshaped arrays, one for each component:

```julia
mats = to_matrix(u_vec)
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
u_proj = Rₕ(W, f)

# In-place projection into an existing element
Rₕ!(u, f)
```

### Projecting vector-valued functions

For multi-component spaces, $R_h$ accepts either a tuple of scalar functions or a vector function:

```julia
# Tuple of coordinate functions
fx(x) = x[1]
fy(x) = 2 * x[2]

Rₕ!(u_vec, (fx, fy))

# Or a function returning a tuple/vector
f_vel(x) = (sin(x[1]), cos(x[2]))
Rₕ!(u_vec, f_vel)
```

---

## 7. Numerical cell averaging

When discretizing conservation laws or finite volume formulations, quantities often represent cell averages rather than pointwise values. 

The cell averaging operator $\mathrm{avg}_h$ integrates a function $f$ over each computational cell $\square_i$ around grid point $x_i$, normalized by the cell volume $|\square_i|$:

```math
\mathrm{avg}_h f(x_i) = \frac{1}{|\square_i|} \int_{\square_i} f(x) \, dx
```

In Bramble, $\mathrm{avg}_h$ uses adaptive multidimensional quadrature via `Integrals.jl`:

```julia
# Compute cell-averaged element
u_avg = avgₕ(W, x -> exp(-x[1] - x[2]))

# In-place version
avgₕ!(u, x -> exp(-x[1] - x[2]))
```

For multi-component spaces, averages can likewise be computed across components:

```julia
u_vec_avg = avgₕ(V, (x -> 1.0, x -> 2.0 * x[1]))
```
