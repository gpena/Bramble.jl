```@meta
CurrentModule = Bramble
```

# [Mesh tutorial](@id tutorial_mesh)

**What you will learn.** How a domain becomes a mesh, how to make the points non-uniform, and what the spacings, half points and cell measures that difference operators read mean.

**What you need first.** The [geometry tutorial](@ref tutorial_geometry), for sets, markers and domains.

**Where next.** The [space tutorial](space.md) puts discrete functions on a mesh.

Every block below runs when this page is built.

---

## A first mesh

Take the unit interval and ask for eleven points. [`mesh`](@ref) takes a domain and a point count:

```@example mesh
using Bramble
import Bramble: cell_measure, change_points!, half_point, half_points, half_spacing,
    index_in_marker, indices, interior_indices, is_boundary_index, is_uniform, point,
    spacings

Ω = domain(interval(0.0, 1.0))
Ωₕ = mesh(Ω, 11)          # 11 uniformly spaced points

points(Ωₕ)
```

The points are equally spaced, ``h = 0.1``, and the largest cell measure confirms it:

```@example mesh
hₘₐₓ(Ωₕ), is_uniform(Ωₕ)
```

The mesh carries more than coordinates. Spacings, half points and cell measures, which the difference operators read, are computed from the points and cached.

!!! tip "Try this"
    Change `11` to `21` and print `hₘₐₓ(Ωₕ)` again. Doubling the number of intervals halves the spacing.

---

## Non-uniform meshes

Real problems rarely want equal spacing: boundary layers and corners need points packed where the solution changes fast. The third positional argument of `mesh`, or the `uniform` keyword, chooses the distribution per axis. `false` draws interior points at random and sorts them, which the convergence studies in this manual use to expose order reduction on non-uniform grids:

```@example mesh
Ωₕ_rand = mesh(Ω, 11, false)

is_uniform(Ωₕ), is_uniform(Ωₕ_rand)
```

To place the points yourself, build any mesh and replace its coordinates. [`change_points!`](@ref) keeps the point count, so it can grade a mesh toward the left end:

```@example mesh
Ωₕ_graded = mesh(Ω, 11)
change_points!(Ωₕ_graded, collect(range(0.0, 1.0, length = npoints(Ωₕ_graded)) .^ 2))

points(Ωₕ_graded), hₘₐₓ(Ωₕ_graded)
```

Squaring the uniform points packs them near ``x = 0`` and stretches the cells near ``x = 1``, so `hₘₐₓ` is no longer ``0.1``. `Bramble.set_points!` is the variant that also accepts a different point count. The [graded boundary layer example](../examples/boundary_layer_graded.md) builds such a mesh for a real problem.

!!! tip "Try this"
    Replace the exponent `2` by `3`. The points crowd closer to ``x = 0`` and `hₘₐₓ` grows.

---

## Half points, spacings and cell measures

One small non-uniform mesh exposes every convention at once. Four points at ``0, 0.2, 0.6, 1``:

```@example mesh
Ωₕ_fig = mesh(domain(interval(0.0, 1.0)), 4, true)
Bramble.set_points!(Ωₕ_fig, [0.0, 0.2, 0.6, 1.0])

points(Ωₕ_fig)
```
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
  <text x="90" y="137" font-size="13" fill="#3b82f6" text-anchor="start">half_points(Ωₕ): N+1 cell interfaces</text>

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

Four conventions are worth reading off that picture, because they are the ones that surprise:

- **`half_points` has `N + 1` entries.** They are the cell interfaces, and the first and last coincide with `x₁` and `x_N` rather than sitting outside the domain.
- **The cell around `xᵢ` spans `half_points[i]` to `half_points[i+1]`**, with width `half_spacing(Ωₕ, i)`, which is what `cell_measure(Ωₕ, i)` returns.
- **Boundary cells are half-width**, since `x₁` and `x_N` sit on the edge of their own cell rather than at its centre.
- **`spacing` looks backward, `forward_spacing` forward**: `spacing(Ωₕ, i) = xᵢ - xᵢ₋₁` and `forward_spacing(Ωₕ, i) = xᵢ₊₁ - xᵢ`, each falling back to its neighbour at the end where the stencil runs out.

```@example mesh
half_points(Ωₕ_fig), [cell_measure(Ωₕ_fig, i) for i in 1:4]
```

The four cells sum to the domain length, and the two boundary cells are the smallest. The [reference](@ref mesh_reference) below lists the accessor for each quantity.

!!! tip "Try this"
    Move the third point with `Bramble.set_points!(Ωₕ_fig, [0.0, 0.2, 0.8, 1.0])` and recompute the cell measures. The cells around `x₂` and `x₄` become 0.4 and 0.1, the cell around `x₃` stays 0.4 because it spans `(x₄ - x₂)/2`, and the sum is still 1.

---

## Meshes in two dimensions

Above one dimension a mesh is a tensor product of 1D submeshes, stored per axis, so coordinate storage is ``O(N_x + N_y + N_z)`` while the grid it addresses is the full product. The point count is one integer per axis, or a single integer for the same resolution along every axis. The distribution flag is per axis too: `(true, false)` would be uniform in ``x`` and random in ``y``.

```@example mesh
Ωₕ_2d = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (10, 20), (true, true))

npoints(Ωₕ_2d, Tuple), size(Ωₕ_2d)
```

`Ωₕ_2d(k)` is the submesh along axis `k`, and everything said above about a 1D mesh applies to it:

```@example mesh
Ωₕ_2d(1)
```

The cell around ``(x_i, y_j)`` is the rectangle spanned by the two per-axis intervals, so its measure is the product of the per-axis widths. The picture shows a ``4 \times 3`` mesh with non-uniform points on both axes:

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

The cell around `(x₃, y₂)` is highlighted. Its measure is the product of the two widths:

```@example mesh
Ω_prod = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ_prod = mesh(Ω_prod, (4, 3), (true, true))
change_points!(Ωₕ_prod, markers(Ω_prod), ([0.0, 0.2, 0.6, 1.0], [0.0, 0.5, 1.0]))

cell_measure(Ωₕ_prod, CartesianIndex(3, 2)),
half_spacing(Ωₕ_prod(1), 3) * half_spacing(Ωₕ_prod(2), 2)
```

The cells tile the domain exactly: the twelve measures sum to the area.

```@example mesh
sum(cell_measure(Ωₕ_prod, I) for I in indices(Ωₕ_prod))
```

Above 1D, `change_points!` takes the markers alongside the per-axis coordinate vectors, so the labels are re-evaluated at the new positions.

!!! tip "Try this"
    Rebuild the product mesh with `(3, 3)` points and `[0.0, 0.5, 1.0]` on both axes. The single interior cell has measure ``0.25``, and the corner cells have ``0.0625``.

---

## Boundary, interior and markers

Indices are Julia's own `CartesianIndices`:

```@example mesh
indices(Ωₕ_prod), interior_indices(Ωₕ_prod)
```

```@example mesh
is_boundary_index(Ωₕ_prod, CartesianIndex(1, 2)),
is_boundary_index(Ωₕ_prod, CartesianIndex(2, 2))
```

Building a mesh over a labelled domain projects each marker onto the grid points as a `BitVector`, so membership is one lookup. Here the domain has an inlet, an outlet, walls, and a disc-shaped obstacle given by a predicate:

```@example mesh
I = interval(0.0, 1.0)
Ω_marked = domain(I × I,
    :left_inlet => :xmin,
    :right_outlet => :xmax,
    :walls => (:ymin, :ymax),
    :obstacle => x -> (x[1] - 0.5)^2 + (x[2] - 0.5)^2 < 0.15^2)

Ωₕ_marked = mesh(Ω_marked, (20, 20))

sum(index_in_marker(Ωₕ_marked, :walls)), sum(index_in_marker(Ωₕ_marked, :obstacle))
```

!!! tip "Try this"
    Change the radius `0.15` to `0.3` and rebuild. The obstacle count grows roughly fourfold, with the disc area.

---

## Refinement

[`iterative_refinement!`](@ref) inserts a point at every cell midpoint, updating indices and reapplying the domain's markers. Refinement is uniform and dyadic: `N` points become `2N - 1` along every axis. Above 1D, refine the whole mesh rather than one submesh: `iterative_refinement!(Ωₕ(1))` leaves the mesh's index set and markers sized for the old grid, and `gridspace` refuses it.

```@raw html
<figure>
<svg viewBox="0 0 740 210" width="100%" style="max-width:740px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Diagram showing dyadic iterative_refinement! on a 1D mesh: coarse mesh with N=5 points splits into 2N-1=9 points by inserting midpoints.">
  <defs>
    <marker id="refineArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#10b981"/>
    </marker>
  </defs>

  <!-- Level 0: Coarse Mesh (N = 5) -->
  <text x="40" y="28" font-size="13" font-weight="bold" fill="currentColor">Coarse mesh: N = 5 points</text>
  <line x1="40" y1="55" x2="680" y2="55" stroke="currentColor" stroke-width="1.5"/>

  <!-- 5 Coarse points -->
  <circle cx="40"  cy="55" r="5" fill="#3b82f6" stroke="currentColor" stroke-width="1"/>
  <circle cx="200" cy="55" r="5" fill="#3b82f6" stroke="currentColor" stroke-width="1"/>
  <circle cx="360" cy="55" r="5" fill="#3b82f6" stroke="currentColor" stroke-width="1"/>
  <circle cx="520" cy="55" r="5" fill="#3b82f6" stroke="currentColor" stroke-width="1"/>
  <circle cx="680" cy="55" r="5" fill="#3b82f6" stroke="currentColor" stroke-width="1"/>

  <text x="40"  y="78" font-size="11" fill="#3b82f6" font-weight="bold" text-anchor="middle">x₁</text>
  <text x="200" y="78" font-size="11" fill="#3b82f6" font-weight="bold" text-anchor="middle">x₂</text>
  <text x="360" y="78" font-size="11" fill="#3b82f6" font-weight="bold" text-anchor="middle">x₃</text>
  <text x="520" y="78" font-size="11" fill="#3b82f6" font-weight="bold" text-anchor="middle">x₄</text>
  <text x="680" y="78" font-size="11" fill="#3b82f6" font-weight="bold" text-anchor="middle">x₅</text>

  <!-- Midpoint insertion indicators -->
  <path d="M 120 70 L 120 120" stroke="#10b981" stroke-width="1.5" stroke-dasharray="3,3" marker-end="url(#refineArrow)"/>
  <path d="M 280 70 L 280 120" stroke="#10b981" stroke-width="1.5" stroke-dasharray="3,3" marker-end="url(#refineArrow)"/>
  <path d="M 440 70 L 440 120" stroke="#10b981" stroke-width="1.5" stroke-dasharray="3,3" marker-end="url(#refineArrow)"/>
  <path d="M 600 70 L 600 120" stroke="#10b981" stroke-width="1.5" stroke-dasharray="3,3" marker-end="url(#refineArrow)"/>

  <text x="360" y="102" font-size="12" font-weight="bold" fill="#10b981" text-anchor="middle">iterative_refinement!(Ωₕ)</text>

  <!-- Level 1: Refined Mesh (2N - 1 = 9) -->
  <text x="40" y="135" font-size="13" font-weight="bold" fill="currentColor">Refined mesh: 2N - 1 = 9 points</text>
  <line x1="40" y1="160" x2="680" y2="160" stroke="currentColor" stroke-width="1.5"/>

  <!-- Original nodes (Blue) -->
  <circle cx="40"  cy="160" r="4.5" fill="#3b82f6" stroke="currentColor" stroke-width="1"/>
  <circle cx="200" cy="160" r="4.5" fill="#3b82f6" stroke="currentColor" stroke-width="1"/>
  <circle cx="360" cy="160" r="4.5" fill="#3b82f6" stroke="currentColor" stroke-width="1"/>
  <circle cx="520" cy="160" r="4.5" fill="#3b82f6" stroke="currentColor" stroke-width="1"/>
  <circle cx="680" cy="160" r="4.5" fill="#3b82f6" stroke="currentColor" stroke-width="1"/>

  <!-- Inserted midpoints (Green) -->
  <circle cx="120" cy="160" r="4.5" fill="#10b981" stroke="currentColor" stroke-width="1"/>
  <circle cx="280" cy="160" r="4.5" fill="#10b981" stroke="currentColor" stroke-width="1"/>
  <circle cx="440" cy="160" r="4.5" fill="#10b981" stroke="currentColor" stroke-width="1"/>
  <circle cx="600" cy="160" r="4.5" fill="#10b981" stroke="currentColor" stroke-width="1"/>

  <text x="40"  y="185" font-size="10" fill="#3b82f6" text-anchor="middle">x'₁</text>
  <text x="120" y="185" font-size="10" fill="#10b981" font-weight="bold" text-anchor="middle">x'₂</text>
  <text x="200" y="185" font-size="10" fill="#3b82f6" text-anchor="middle">x'₃</text>
  <text x="280" y="185" font-size="10" fill="#10b981" font-weight="bold" text-anchor="middle">x'₄</text>
  <text x="360" y="185" font-size="10" fill="#3b82f6" text-anchor="middle">x'₅</text>
  <text x="440" y="185" font-size="10" fill="#10b981" font-weight="bold" text-anchor="middle">x'₆</text>
  <text x="520" y="185" font-size="10" fill="#3b82f6" text-anchor="middle">x'₇</text>
  <text x="600" y="185" font-size="10" fill="#10b981" font-weight="bold" text-anchor="middle">x'₈</text>
  <text x="680" y="185" font-size="10" fill="#3b82f6" text-anchor="middle">x'₉</text>

  <!-- Legend -->
  <circle cx="520" cy="28" r="4" fill="#3b82f6"/>
  <text x="530" y="32" font-size="11" fill="currentColor">original vertices</text>
  <circle cx="620" cy="28" r="4" fill="#10b981"/>
  <text x="630" y="32" font-size="11" fill="currentColor">new midpoints</text>
</svg>
</figure>
```

A mesh carrying custom markers needs the two-argument form: the labels are re-evaluated on the new points, and refining without a domain to re-derive them from is an error rather than a silent loss.

```@example mesh
Ωₕ_ref = mesh(Ω_marked, (20, 20))
iterative_refinement!(Ωₕ_ref, markers(Ω_marked))

npoints(Ωₕ_ref, Tuple), sum(index_in_marker(Ωₕ_ref, :obstacle))
```

---

## [Reference](@id mesh_reference)

### Constructing from a set

A geometric set can be passed directly when no custom labels are needed. `:boundary` and `:interior` are provisioned either way:

```@example mesh
X = interval(0.0, 1.0) × interval(0.0, 2.0)
Ωₕ_direct = mesh(X, 20)      # isotropic: 20 × 20

:boundary in keys(markers(Ωₕ_direct)), :interior in keys(markers(Ωₕ_direct))
```

### Points and half points

`points` returns the coordinate vector, or a tuple of them above 1D. A single coordinate comes from `point(Ωₕ, idx)` or from indexing, with a linear index, a tuple or a `CartesianIndex`:

```@example mesh
Ωₕ_fig[3], point(Ωₕ_fig, 3), Ωₕ_2d[2, 5], point(Ωₕ_2d, CartesianIndex(2, 5))
```

```@example mesh
half_points(Ωₕ_fig), half_point(Ωₕ_fig, 3)
```

### Spacings and cell measures

| Function | Meaning |
| --- | --- |
| `spacing(Ωₕ, i)` | backward spacing ``h_i = x_i - x_{i-1}`` (``x_2 - x_1`` at ``i = 1``) |
| `forward_spacing(Ωₕ, i)` | forward spacing ``h_{i+1} = x_{i+1} - x_i`` |
| `half_spacing(Ωₕ, i)` | cell width ``h_{i+1/2} = (h_i + h_{i+1})/2`` |
| `cell_measure(Ωₕ, idx)` | measure of the control volume at `idx`: the product of the per-axis cell widths |
| `hₘₐₓ(Ωₕ)` | largest cell measure in the mesh |

A 1D mesh stores its backward spacings, so `spacings` hands back the whole vector and the accessors index it. The cache is rebuilt by `set_points!`, and so by [`iterative_refinement!`](@ref) and [`change_points!`](@ref) as well:

```@example mesh
spacings(Ωₕ_fig), spacings(Ωₕ_fig)[3] == spacing(Ωₕ_fig, 3)
```

```@example mesh
hₘₐₓ(Ωₕ_fig), cell_measure(Ωₕ_2d, (3, 4))
```

`boundary_indices(Ωₕ)` gives the boundary facets separately, as a tuple of index sets.
