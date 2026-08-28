```@meta
CurrentModule = Bramble
```

# [Mesh Tutorial](@id tutorial_mesh)

`Bramble.jl` provides structured, zero-allocation Cartesian and tensor-product mesh representations optimized for finite difference, finite volume, and mimetic discretization schemes.

In this tutorial, you will learn how to:
1. Construct 1D meshes ([`Mesh1D`](@ref)) and multi-dimensional tensor-product meshes ([`MeshnD`](@ref)).
2. Configure uniform and non-uniform coordinate distributions.
3. Query mesh geometric properties: coordinates, half-points (cell centers), spacings, cell measures, and $h_{\max}$.
4. Query mesh boundaries and interiors using `CartesianIndices`.
5. Access and evaluate boundary and region markers on meshes.
6. Perform in-place mesh refinement ([`iterative_refinement!`](@ref)) and coordinate relocation ([`change_points!`](@ref)).

---

## 1. Constructing Meshes

Meshes in `Bramble.jl` are built on top of computational [`Domain`](@ref)s. The primary entry point is the [`mesh`](@ref) function.

### 1.1 One-dimensional meshes

To construct a 1D mesh with $N$ points over an interval $[a, b]$:

```julia
using Bramble

# 1. Define a domain
Ω = domain(interval(0.0, 1.0))

# 2. Build a uniform mesh with 11 grid points (step h = 0.1)
Ωₕ = mesh(Ω, 11)
```

By default, `mesh` generates a **uniform** grid. You can also specify non-uniform point distributions:

```julia
# Explicit non-uniform 1D mesh
Ωₕ_nonunif = mesh(Ω, 11, false)
```

For 1D meshes, [`is_uniform`](@ref) checks whether all cell widths are identical:

```julia
is_uniform(Ωₕ)          # true
is_uniform(Ωₕ_nonunif)   # false
```

### 1.2 Multi-dimensional tensor-product meshes

For 2D and 3D domains, `Bramble.jl` constructs a [`MeshnD`](@ref) as a Cartesian product of 1D submeshes. This allows $O(N_x + N_y + N_z)$ coordinate storage while providing full $O(N_x \times N_y \times N_z)$ grid traversal:

```julia
# 2D Unit square: [0, 1] × [0, 2]
Ω_2d = domain(interval(0.0, 1.0) × interval(0.0, 2.0))

# Create a 2D mesh with 10 × 20 grid points (uniform in both directions)
Ωₕ_2d = mesh(Ω_2d, (10, 20))

# Create a 2D mesh with mixed uniformity (uniform in x, non-uniform in y)
Ωₕ_mixed = mesh(Ω_2d, (10, 20), (true, false))
```

You can retrieve the underlying 1D submesh along any coordinate axis using functor call syntax:

```julia
x_mesh = Ωₕ_2d(1)  # 1D submesh in x-direction
y_mesh = Ωₕ_2d(2)  # 1D submesh in y-direction
```

---

## 2. Accessing Grid Coordinates and Metric Properties

### 2.1 Points and Coordinates

- **`points(Ωₕ)`**: Returns the coordinate vector (1D) or tuple of coordinate vectors (nD).
- **`point(Ωₕ, idx)`** or direct indexing **`Ωₕ[idx]`**: Evaluates the coordinate at linear index `i`, coordinate tuple `(i, j)`, or `CartesianIndex(i, j)`.

```julia
# 1D mesh point access
p3 = Ωₕ[3]          # Coordinate x₃
p3_alt = point(Ωₕ, 3)

# 2D mesh point access
p_ij = Ωₕ[2, 5]     # Coordinate tuple (x₂, y₅)
```

### 2.2 Half-Points (Cell Centers)

Finite volume and staggered-grid methods frequently require cell midpoints $x_{i+1/2}$:

```julia
# Pre-computed cell centers
hp = half_points(Ωₕ)
hp_i = half_point(Ωₕ, 3)  # x_{3+1/2}
```

### 2.3 Spacings and Cell Measures

- **`spacing(Ωₕ, i)`**: Backward spacing $h_i = x_i - x_{i-1}$ (for $i=1$, returns $x_2 - x_1$).
- **`forward_spacing(Ωₕ, i)`**: Forward spacing $h_{i+1} = x_{i+1} - x_i$.
- **`half_spacing(Ωₕ, i)`**: Cell width $h_{i+1/2} = \frac{h_i + h_{i+1}}{2}$.
- **`cell_measure(Ωₕ, idx)`**: Volume/area of the control volume centered at `idx`.
  - In 1D: cell measure is $h_{i+1/2}$.
  - In 2D: cell measure is $h_{x, i+1/2} \times h_{y, j+1/2}$.
  - In 3D: cell measure is $h_{x, i+1/2} \times h_{y, j+1/2} \times h_{z, l+1/2}$.
- **`hₘₐₓ(Ωₕ)`**: Maximum diagonal cell measure across the entire mesh.

```julia
# Maximum grid stepsize
h = hₘₐₓ(Ωₕ_2d)

# Control volume measure at cell (3, 4)
vol = cell_measure(Ωₕ_2d, (3, 4))
```

---

## 3. Boundary and Interior Indexing

`Bramble.jl` leverages Julia's native `CartesianIndices` for zero-overhead, multi-dimensional grid navigation:

```julia
# Complete Cartesian grid indices
idxs = indices(Ωₕ_2d)  # CartesianIndices((1:10, 1:20))

# Interior indices (excluding all boundaries)
interior = interior_indices(Ωₕ_2d)  # CartesianIndices((2:9, 2:19))

# Boundary facets as a tuple of CartesianIndices
facets = boundary_indices(Ωₕ_2d)

# Test whether an index lies on the domain boundary
is_boundary = is_boundary_index(Ωₕ_2d, CartesianIndex(1, 5))  # true
```

---

## 4. Markers on Meshes

When creating a mesh from a labeled [`Domain`](@ref), markers are projected onto the grid points as highly efficient `BitVector`s:

```julia
# Domain with boundary and obstacle markers
I = interval(0.0, 1.0)
X = domain(I × I,
           :left_inlet => :left,
           :right_outlet => :right,
           :walls => (:top, :bottom),
           :obstacle => x -> (x[1]-0.5)^2 + (x[2]-0.5)^2 < 0.15^2)

# Generate mesh
M = mesh(X, (20, 20))

# Query markers
m_dict = markers(M)

# Retrieve bit-vector for a specific label
is_wall = index_in_marker(M, :walls)
is_obs  = index_in_marker(M, :obstacle)
```

---

## 5. Mesh Adaptation and Modification

Meshes in `Bramble.jl` are mutable structures designed for adaptive algorithms:

### 5.1 In-Place Mesh Refinement (`iterative_refinement!`)

Halves every cell by inserting new points at each cell midpoint, simultaneously updating indices and reapplying domain markers:

```julia
# Refine mesh in-place
iterative_refinement!(M)

# Point count increases: (2N_x - 1) × (2N_y - 1)
npoints(M, Tuple)  # (39, 39)
```

### 5.2 Relocating Mesh Coordinates (`change_points!`)

For moving-boundary problems or non-uniform smoothing:

```julia
# Supply new coordinates for a 1D mesh (matching the 11 points of Ωₕ)
new_pts = range(0.0, 1.0, length=npoints(Ωₕ)) |> collect
change_points!(Ωₕ, new_pts)

# Or update points and re-evaluate markers for a multi-dimensional mesh
nx, ny = npoints(M, Tuple)
new_x_pts = range(0.0, 1.0, length=nx) |> collect
new_y_pts = range(0.0, 1.0, length=ny) |> collect
change_points!(M, markers(X), (new_x_pts, new_y_pts))
```
