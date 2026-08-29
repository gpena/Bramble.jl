```@meta
CurrentModule = Bramble
```

# [Mesh tutorial](@id tutorial_mesh)

`Bramble.jl` provides structured, zero-allocation Cartesian and tensor-product mesh representations optimized for finite difference, finite volume, and mimetic discretization schemes.

In this tutorial, you will learn how to:
1. Construct 1D meshes ([`Mesh1D`](@ref)) and multi-dimensional tensor-product meshes ([`MeshnD`](@ref)).
2. Configure uniform and non-uniform coordinate distributions.
3. Query mesh geometric properties: coordinates, half-points (cell centers), spacings, cell measures, and $h_{\max}$.
4. Query mesh boundaries and interiors using `CartesianIndices`.
5. Access and evaluate boundary and region markers on meshes.
6. Perform in-place mesh refinement ([`iterative_refinement!`](@ref)) and coordinate relocation ([`change_points!`](@ref)).

---

## 1. Constructing meshes

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

## 2. Accessing grid coordinates and metric properties

```@raw html
<figure style="margin:1.5em 0;text-align:center">
<svg viewBox="0 0 780 360" width="100%" style="max-width:780px;font-family:system-ui,-apple-system,'Segoe UI',sans-serif" role="img"
     aria-label="One-dimensional staggered mesh showing grid points, half points, cells, spacing and forward spacing.">
  <defs>
    <marker id="bl" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 z" fill="#8b5cf6"/></marker>
    <marker id="blr" markerWidth="9" markerHeight="9" refX="1" refY="3" orient="auto"><path d="M8,0 L0,3 L8,6 z" fill="#8b5cf6"/></marker>
    <marker id="rd" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 z" fill="#ef4444"/></marker>
    <marker id="rdr" markerWidth="9" markerHeight="9" refX="1" refY="3" orient="auto"><path d="M8,0 L0,3 L8,6 z" fill="#ef4444"/></marker>
    <marker id="gn" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 z" fill="#10b981"/></marker>
    <marker id="gnr" markerWidth="9" markerHeight="9" refX="1" refY="3" orient="auto"><path d="M8,0 L0,3 L8,6 z" fill="#10b981"/></marker>
  </defs>

  <!-- cells: [half_points[i], half_points[i+1]] -->
  <g stroke="#3b82f6" stroke-opacity="0.55" fill="#3b82f6">
    <rect x="90"  y="165" width="60"  height="52" fill-opacity="0.20"/>
    <rect x="150" y="165" width="180" height="52" fill-opacity="0.10"/>
    <rect x="330" y="165" width="240" height="52" fill-opacity="0.20"/>
    <rect x="570" y="165" width="120" height="52" fill-opacity="0.10"/>
  </g>

  <!-- axis -->
  <line x1="90" y1="191" x2="690" y2="191" stroke="currentColor" stroke-width="1.6"/>

  <!-- half points: open squares, N+1 of them -->
  <g fill="#fff" stroke="#3b82f6" stroke-width="2">
    <rect x="84"  y="185" width="12" height="12"/><rect x="144" y="185" width="12" height="12"/>
    <rect x="324" y="185" width="12" height="12"/><rect x="564" y="185" width="12" height="12"/>
    <rect x="684" y="185" width="12" height="12"/>
  </g>
  <g fill="#3b82f6" font-size="12" text-anchor="middle">
    <text x="90" y="157">1</text><text x="150" y="157">2</text><text x="330" y="157">3</text>
    <text x="570" y="157">4</text><text x="690" y="157">5</text>
  </g>
  <text x="90" y="137" font-size="13" fill="#3b82f6" text-anchor="start">half_points(Ωₕ) — N+1 cell interfaces</text>

  <!-- grid points -->
  <g fill="currentColor">
    <circle cx="90" cy="191" r="5"/><circle cx="210" cy="191" r="5"/>
    <circle cx="450" cy="191" r="5"/><circle cx="690" cy="191" r="5"/>
  </g>
  <g font-size="15" text-anchor="middle" fill="currentColor">
    <text x="90" y="241">x₁</text><text x="210" y="241">x₂</text>
    <text x="450" y="241">x₃</text><text x="690" y="241">x₄</text>
  </g>
  <g font-size="12" text-anchor="middle" fill="currentColor" opacity="0.65">
    <text x="90" y="258">0.0</text><text x="210" y="258">0.2</text>
    <text x="450" y="258">0.6</text><text x="690" y="258">1.0</text>
  </g>

  <!-- cell_measure / half_spacing of the cell around x₃ -->
  <line x1="330" y1="112" x2="570" y2="112" stroke="#8b5cf6" stroke-width="1.6" marker-start="url(#blr)" marker-end="url(#bl)"/>
  <line x1="330" y1="112" x2="330" y2="165" stroke="#8b5cf6" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="570" y1="112" x2="570" y2="165" stroke="#8b5cf6" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="450" y="103" font-size="13" fill="#8b5cf6" text-anchor="middle">cell_measure(Ωₕ, 3) = half_spacing(Ωₕ, 3) = 0.4</text>

  <!-- spacing (backward) -->
  <line x1="90" y1="292" x2="210" y2="292" stroke="#ef4444" stroke-width="1.6" marker-start="url(#rdr)" marker-end="url(#rd)"/>
  <text x="150" y="311" font-size="13" fill="#ef4444" text-anchor="middle">spacing(Ωₕ, 2) = x₂ − x₁ = 0.2</text>

  <!-- forward_spacing -->
  <line x1="210" y1="336" x2="450" y2="336" stroke="#10b981" stroke-width="1.6" marker-start="url(#gnr)" marker-end="url(#gn)"/>
  <text x="330" y="355" font-size="13" fill="#10b981" text-anchor="middle">forward_spacing(Ωₕ, 2) = x₃ − x₂ = 0.4</text>
</svg>
</figure>
```

The mesh above is `[0.0, 0.2, 0.6, 1.0]`, deliberately non-uniform. Four conventions are
worth reading off it, because they are the ones that most often surprise:

  - **`half_points` has `N + 1` entries, not `N`.** They are the cell interfaces, and the
    first and last *coincide with* `x₁` and `x_N` rather than being extrapolated outside
    the domain. Here they are `[0.0, 0.1, 0.4, 0.8, 1.0]`.
  - **The cell around `xᵢ` spans `half_points[i] .. half_points[i+1]`**, and its width is
    exactly `half_spacing(Ωₕ, i)`, which is what `cell_measure(Ωₕ, i)` returns. The four
    cells here measure `[0.1, 0.3, 0.4, 0.2]` and sum to the domain length.
  - **Boundary cells are half-width.** `x₁` and `x_N` sit on the edge of their own cell,
    not at its centre, which is why the first and last measures are the smallest.
  - **`spacing` looks backward and `forward_spacing` looks forward**, so
    `spacing(Ωₕ, i) = xᵢ − xᵢ₋₁` and `forward_spacing(Ωₕ, i) = xᵢ₊₁ − xᵢ`. Each has one
    special case at the boundary where the neighbour is missing: `spacing(Ωₕ, 1)` returns
    `x₂ − x₁` and `forward_spacing(Ωₕ, N)` returns `x_N − x_{N−1}`.

### 2.1 Points and coordinates

- **`points(Ωₕ)`**: Returns the coordinate vector (1D) or tuple of coordinate vectors (nD).
- **`point(Ωₕ, idx)`** or direct indexing **`Ωₕ[idx]`**: Evaluates the coordinate at linear index `i`, coordinate tuple `(i, j)`, or `CartesianIndex(i, j)`.

```julia
# 1D mesh point access
p3 = Ωₕ[3]          # Coordinate x₃
p3_alt = point(Ωₕ, 3)

# 2D mesh point access
p_ij = Ωₕ[2, 5]     # Coordinate tuple (x₂, y₅)
```

### 2.2 Half-points and cell centers

Finite volume and staggered-grid methods frequently require cell midpoints $x_{i+1/2}$:

```julia
# Pre-computed cell centers
hp = half_points(Ωₕ)
hp_i = half_point(Ωₕ, 3)  # x_{3+1/2}
```

### 2.3 Spacings and cell measures

- **`spacing(Ωₕ, i)`**: Backward spacing $h_i = x_i - x_{i-1}$ (for $i=1$, returns $x_2 - x_1$).
- **`forward_spacing(Ωₕ, i)`**: Forward spacing $h_{i+1} = x_{i+1} - x_i$.
- **`half_spacing(Ωₕ, i)`**: Cell width $h_{i+1/2} = \frac{h_i + h_{i+1}}{2}$.
- **`cell_measure(Ωₕ, idx)`**: Volume/area of the control volume centered at `idx`.
  - In 1D: cell measure is $h_{i+1/2}$.
  - In 2D: cell measure is $h_{x, i+1/2} \times h_{y, j+1/2}$.
  - In 3D: cell measure is $h_{x, i+1/2} \times h_{y, j+1/2} \times h_{z, l+1/2}$.
- **`hₘₐₓ(Ωₕ)`**: Maximum diagonal cell measure across the entire mesh.

A 1D mesh stores its backward spacings rather than recomputing them, so
`spacings(Ωₕ)` hands back the whole vector and `spacing(Ωₕ, i)` is a single array read.
`forward_spacing(Ωₕ, i)` reads the same vector one entry along, since
$x_{i+1} - x_i$ is the backward spacing at $i+1$. The cache is rebuilt by
`set_points!`, and so by `iterative_refinement!` and `change_points!` as well, meaning
it always matches the current points.

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, false)

spacings(Ωₕ)                       # every hᵢ at once
spacings(Ωₕ)[3] == spacing(Ωₕ, 3)  # true, the accessor just indexes it
```

This matters for the difference operators, which need one spacing per grid point: they
index the cached vector instead of calling `spacing` once per point, which measured
about 3.6x faster on a 100 000-point grid.

```julia
# Maximum grid stepsize
h = hₘₐₓ(Ωₕ_2d)

# Control volume measure at cell (3, 4)
vol = cell_measure(Ωₕ_2d, (3, 4))
```

---

```@raw html
<figure style="margin:1.5em 0;text-align:center">
<svg viewBox="0 0 700 380" width="100%" style="max-width:700px;font-family:system-ui,-apple-system,'Segoe UI',sans-serif" role="img"
     aria-label="Two-dimensional tensor-product mesh: the cell around a grid point is the product of its per-axis cell widths.">
  <defs>
    <marker id="p2" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 z" fill="#8b5cf6"/></marker>
    <marker id="p2r" markerWidth="9" markerHeight="9" refX="1" refY="3" orient="auto"><path d="M8,0 L0,3 L8,6 z" fill="#8b5cf6"/></marker>
  </defs>

  <!-- every cell, tiling the domain -->
  <g stroke="#3b82f6" stroke-opacity="0.35" fill="none">
    <g fill="#3b82f6" fill-opacity="0.07">
      <rect x="80"  y="245" width="48"  height="55"/><rect x="128" y="245" width="144" height="55"/>
      <rect x="272" y="245" width="192" height="55"/><rect x="464" y="245" width="96"  height="55"/>
      <rect x="80"  y="135" width="48"  height="110"/><rect x="128" y="135" width="144" height="110"/>
      <rect x="464" y="135" width="96"  height="110"/>
      <rect x="80"  y="80"  width="48"  height="55"/><rect x="128" y="80"  width="144" height="55"/>
      <rect x="272" y="80"  width="192" height="55"/><rect x="464" y="80"  width="96"  height="55"/>
    </g>
    <!-- the highlighted cell around (x₃, y₂) -->
    <rect x="272" y="135" width="192" height="110" fill="#8b5cf6" fill-opacity="0.20" stroke="#8b5cf6" stroke-opacity="0.9" stroke-width="1.6"/>
  </g>

  <!-- grid lines through the points -->
  <g stroke="currentColor" stroke-opacity="0.30" stroke-width="1">
    <line x1="80" y1="80" x2="80" y2="300"/><line x1="176" y1="80" x2="176" y2="300"/>
    <line x1="368" y1="80" x2="368" y2="300"/><line x1="560" y1="80" x2="560" y2="300"/>
    <line x1="80" y1="300" x2="560" y2="300"/><line x1="80" y1="190" x2="560" y2="190"/>
    <line x1="80" y1="80"  x2="560" y2="80"/>
  </g>

  <!-- grid points -->
  <g fill="currentColor">
    <circle cx="80" cy="300" r="4"/><circle cx="176" cy="300" r="4"/><circle cx="368" cy="300" r="4"/><circle cx="560" cy="300" r="4"/>
    <circle cx="80" cy="190" r="4"/><circle cx="176" cy="190" r="4"/><circle cx="560" cy="190" r="4"/>
    <circle cx="80" cy="80"  r="4"/><circle cx="176" cy="80"  r="4"/><circle cx="368" cy="80"  r="4"/><circle cx="560" cy="80"  r="4"/>
  </g>
  <circle cx="368" cy="190" r="5.5" fill="#8b5cf6"/>
  <text x="380" y="182" font-size="14" fill="#8b5cf6">(x₃, y₂)</text>

  <!-- axis labels -->
  <g font-size="14" text-anchor="middle" fill="currentColor">
    <text x="80" y="323">x₁</text><text x="176" y="323">x₂</text><text x="368" y="323">x₃</text><text x="560" y="323">x₄</text>
    <text x="62" y="305">y₁</text><text x="62" y="195">y₂</text><text x="62" y="85">y₃</text>
  </g>

  <!-- per-axis widths of the highlighted cell -->
  <line x1="272" y1="352" x2="464" y2="352" stroke="#8b5cf6" stroke-width="1.6" marker-start="url(#p2r)" marker-end="url(#p2)"/>
  <text x="368" y="371" font-size="13" fill="#8b5cf6" text-anchor="middle">half_spacing(Ωₕ(1), 3) = 0.4</text>
  <line x1="612" y1="135" x2="612" y2="245" stroke="#8b5cf6" stroke-width="1.6" marker-start="url(#p2r)" marker-end="url(#p2)"/>
  <text x="622" y="194" font-size="13" fill="#8b5cf6" text-anchor="start">half_spacing(Ωₕ(2), 2)</text>
  <text x="622" y="211" font-size="13" fill="#8b5cf6" text-anchor="start">= 0.5</text>
</svg>
</figure>
```

An `n`-dimensional mesh is a tensor product of 1D meshes, and every quantity above is
built the same way. The cell around `(xᵢ, yⱼ)` is the rectangle spanned by the two
per-axis intervals, so its measure is the product of the per-axis widths:

```julia
cell_measure(Ωₕ, CartesianIndex(3, 2))          # 0.2
half_spacing(Ωₕ(1), 3) * half_spacing(Ωₕ(2), 2)  # 0.2 — the same number
```

`Ωₕ(k)` is the 1D submesh along axis `k`, so anything documented for a 1D mesh applies to
it directly. As in one dimension the cells tile the domain exactly — here the twelve cell
measures sum to the area `1.0` — and the cells touching a boundary are correspondingly
thinner along that axis.


## 3. Boundary and interior indexing

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

## 4. Markers on meshes

When creating a mesh from a labeled [`Domain`](@ref), markers are projected onto the grid points as highly efficient `BitVector`s:

```julia
# Domain with boundary and obstacle markers
I = interval(0.0, 1.0)
Ω = domain(I × I,
           :left_inlet => :left,
           :right_outlet => :right,
           :walls => (:top, :bottom),
           :obstacle => x -> (x[1]-0.5)^2 + (x[2]-0.5)^2 < 0.15^2)

# Generate mesh
Ωₕ = mesh(Ω, (20, 20))

# Query markers
m_dict = markers(Ωₕ)

# Retrieve bit-vector for a specific label
is_wall = index_in_marker(Ωₕ, :walls)
is_obs  = index_in_marker(Ωₕ, :obstacle)
```

---

## 5. Mesh adaptation and modification

Meshes in `Bramble.jl` are mutable structures designed for adaptive algorithms:

### 5.1 In-place mesh refinement

Halves every cell by inserting new points at each cell midpoint, simultaneously updating indices and reapplying domain markers:

```julia
# Refine mesh in-place
iterative_refinement!(Ωₕ)

# Point count increases: (2N_x - 1) × (2N_y - 1)
npoints(Ωₕ, Tuple)  # (39, 39)
```

### 5.2 Relocating mesh coordinates

For moving-boundary problems or non-uniform smoothing:

```julia
# Supply new coordinates for a 1D mesh (matching the 11 points of Ωₕ)
new_pts = range(0.0, 1.0, length=npoints(Ωₕ)) |> collect
change_points!(Ωₕ, new_pts)

# Or update points and re-evaluate markers for a multi-dimensional mesh
nx, ny = npoints(Ωₕ, Tuple)
new_x_pts = range(0.0, 1.0, length=nx) |> collect
new_y_pts = range(0.0, 1.0, length=ny) |> collect
change_points!(Ωₕ, markers(Ω), (new_x_pts, new_y_pts))
```
