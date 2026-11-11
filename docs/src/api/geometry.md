```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Geometry

## Sets and intervals

`is_collapsed`, `point_type` and `set` are private; see the
[geometry internals page](../internals/geometry.md).

```@docs
interval
point
box
×
dim
topo_dim
Base.extrema(::CartesianProduct, ::Integer)
center
projection
boundary_symbols
```

## [Boundary symbols](@id geometry_reference)

```@setup geometry_ref
using Bramble
import Bramble: point, topo_dim
```

The canonical names mean the same face in every dimension; the viewpoint aliases do not (`:left` is `:xmin` in 1D and 2D, `:ymin` in 3D):

- **1D**: `:xmin` (`:left`), `:xmax` (`:right`)
- **2D**: `:xmin` (`:left`), `:xmax` (`:right`), `:ymin` (`:bottom`), `:ymax` (`:top`)
- **3D**: `:xmin` (`:back`), `:xmax` (`:front`), `:ymin` (`:left`), `:ymax` (`:right`), `:zmin` (`:bottom`), `:zmax` (`:top`)

[`boundary_symbols`](@ref) lists the canonical names for a given dimension:

```@example geometry_ref
boundary_symbols(2)
```
```@raw html
<figure>
<svg viewBox="0 0 780 280" width="100%" style="max-width:780px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Diagram of standard boundary symbols in 2D (:xmin/:left, :xmax/:right, :ymin/:bottom, :ymax/:top) and 3D (:xmin/:back, :xmax/:front, :ymin/:left, :ymax/:right, :zmin/:bottom, :zmax/:top).">
  <!-- Panel 1: 2D Domain -->
  <g transform="translate(30, 20)">
    <rect x="0" y="0" width="320" height="240" rx="6" fill="none" stroke="currentColor" stroke-opacity="0.2" stroke-width="1"/>
    <text x="160" y="28" font-size="14" font-weight="bold" fill="currentColor" text-anchor="middle">2D boundary facets</text>

    <!-- 2D Box -->
    <rect x="80" y="70" width="160" height="120" fill="currentColor" fill-opacity="0.05" stroke="currentColor" stroke-width="2"/>

    <!-- Labels -->
    <!-- :ymax / :top -->
    <text x="160" y="58" font-size="12" font-weight="bold" fill="#ef4444" text-anchor="middle">:ymax (:top)</text>
    <line x1="80" y1="70" x2="240" y2="70" stroke="#ef4444" stroke-width="3"/>

    <!-- :ymin / :bottom -->
    <text x="160" y="210" font-size="12" font-weight="bold" fill="#ef4444" text-anchor="middle">:ymin (:bottom)</text>
    <line x1="80" y1="190" x2="240" y2="190" stroke="#ef4444" stroke-width="3"/>

    <!-- :xmin / :left -->
    <text x="35" y="134" font-size="12" font-weight="bold" fill="#3b82f6" text-anchor="middle">:xmin</text>
    <text x="35" y="148" font-size="10" fill="#3b82f6" text-anchor="middle">(:left)</text>
    <line x1="80" y1="70" x2="80" y2="190" stroke="#3b82f6" stroke-width="3"/>

    <!-- :xmax / :right -->
    <text x="285" y="134" font-size="12" font-weight="bold" fill="#3b82f6" text-anchor="middle">:xmax</text>
    <text x="285" y="148" font-size="10" fill="#3b82f6" text-anchor="middle">(:right)</text>
    <line x1="240" y1="70" x2="240" y2="190" stroke="#3b82f6" stroke-width="3"/>
  </g>

  <!-- Panel 2: 3D Isometric Domain -->
  <g transform="translate(410, 20)">
    <rect x="0" y="0" width="340" height="240" rx="6" fill="none" stroke="currentColor" stroke-opacity="0.2" stroke-width="1"/>
    <text x="170" y="28" font-size="14" font-weight="bold" fill="currentColor" text-anchor="middle">3D boundary facets</text>

    <!-- Isometric Box Coordinates -->
    <!-- Back/interior edges dashed -->
    <line x1="130" y1="70"  x2="130" y2="150" stroke="currentColor" stroke-dasharray="3,3" stroke-width="1" stroke-opacity="0.4"/>
    <line x1="70"  y1="190" x2="130" y2="150" stroke="currentColor" stroke-dasharray="3,3" stroke-width="1" stroke-opacity="0.4"/>
    <line x1="130" y1="150" x2="250" y2="150" stroke="currentColor" stroke-dasharray="3,3" stroke-width="1" stroke-opacity="0.4"/>

    <!-- Top face (:zmax) -->
    <polygon points="70,110 130,70 250,70 190,110" fill="#ef4444" fill-opacity="0.1" stroke="#ef4444" stroke-width="1.5"/>
    <text x="160" y="94" font-size="11" font-weight="bold" fill="#ef4444" text-anchor="middle">:zmax (:top)</text>

    <!-- Right face (:ymax) -->
    <polygon points="190,110 250,70 250,150 190,190" fill="#3b82f6" fill-opacity="0.1" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="225" y="135" font-size="11" font-weight="bold" fill="#3b82f6" text-anchor="middle">:ymax (:right)</text>

    <!-- Front face (:xmax) -->
    <polygon points="70,110 190,110 190,190 70,190" fill="#10b981" fill-opacity="0.1" stroke="#10b981" stroke-width="1.5"/>
    <text x="130" y="155" font-size="11" font-weight="bold" fill="#10b981" text-anchor="middle">:xmax (:front)</text>

    <!-- Left callout (:ymin) -->
    <text x="35" y="150" font-size="11" font-weight="bold" fill="#3b82f6" text-anchor="middle">:ymin (:left)</text>
    <!-- Back callout (:xmin) -->
    <text x="190" y="60" font-size="11" font-weight="bold" fill="#10b981" text-anchor="middle">:xmin (:back)</text>
    <!-- Bottom callout (:zmin) -->
    <text x="130" y="215" font-size="11" font-weight="bold" fill="#ef4444" text-anchor="middle">:zmin (:bottom)</text>
  </g>
</svg>
</figure>
```

## Collapsed sets

A dimension is **collapsed** when its interval is degenerate (``a = b``), which is how a surface or interface embedded in a higher-dimensional space is written. `dim` is the embedding dimension and `topo_dim` the topological one; they differ only for a collapsed set. [`Bramble.is_collapsed`](@ref) answers per axis; it is private, hence the prefix:

```@example geometry_ref
line_in_2d = interval(0.0, 1.0) × point(0.0)

dim(line_in_2d), topo_dim(line_in_2d),
Bramble.is_collapsed(line_in_2d, 1), Bramble.is_collapsed(line_in_2d, 2)
```

Related constructors: `interval(0, 2)` converts integer bounds to `Float64`, `point(0.5)` is the degenerate interval ``[0.5, 0.5]``, and `box(1.5, 0.2)` sorts its bounds.

## Markers and domains

```@docs
markers
domain
labels
```

## Polygons

`polygon(V)` is exported. `signed_distance`, `polygon_spaces` and `lift` are `public` and not
exported; `using Bramble: polygon_spaces, lift` brings the last two in by name. A polygon is a
marker value: `:P => polygon(V)` in a domain labels the nodes of a 2D mesh `:P` (interior),
`:P_boundary` and `:P_exterior`. The vertices are validated with exact predicates for
orientation, folding back and crossing: at least three, no zero-length edge,
an area above rounding level, a simple boundary. A vertex in the middle of a
straight edge is kept. See [Meshes](meshes.md) for the grid fitted to a polygon and
[Grid spaces](spaces.md) for `polygon_spaces`.

```@docs
polygon
Bramble.signed_distance
```

## Level sets

`levelset(φ)` is exported. It wraps a function `φ(x)` as a 2D marker value: `:D => levelset(φ)` in a domain labels the nodes of a 2D mesh `:D` (where
`φ < 0`), `:D_boundary` (on the curve `φ = 0`) and `:D_exterior` (where `φ > 0`). `φ` must
return a real number, not a `Bool`, and need not be a signed distance. The labels follow the
rules of a polygon's: a 2D set, no reserved label, no clash with another marker's labels.
With a `time_set`, `φ` still takes `x` alone.

```@docs
levelset
```
