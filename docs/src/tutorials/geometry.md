```@meta
CurrentModule = Bramble
```

# [Geometry tutorial](@id tutorial_geometry)

Every Bramble problem starts here: a set, the labels naming its boundary pieces, and the
domain that carries both into [`mesh`](@ref). Every block below runs when this page is
built, so the printed values are the ones the code produces.

---

## 1. Sets and intervals

`CartesianProduct{D, T}` is a product of $D$ closed intervals in $\mathbb{R}^D$ with
coordinate type `T`.

### 1.1 Creating 1D intervals

[`interval`](@ref) builds a closed interval $[a, b] \subset \mathbb{R}$:

```@example geometry
using Bramble

I = interval(0.0, 1.0)
I_int = interval(0, 2)   # integer bounds are converted to Float64
P = point(0.5)           # the degenerate interval [0.5, 0.5]
B = box(1.5, 0.2)        # bounds are sorted
```

### 1.2 Multi-dimensional sets with `×`

The tensor product operator `×` (`\times<tab>`) builds hyper-rectangles:

```@example geometry
Ω_2d = interval(0.0, 1.0) × interval(0.0, 1.0)
Ω_3d = interval(-1.0, 1.0) × interval(0.0, 2.0) × interval(0.0, 0.5)
```

---

## 2. Querying geometric properties

```@example geometry
X = interval(0.0, 2.0) × interval(-1.0, 1.0)

dim(X), topo_dim(X)
```

`dim` is the embedding dimension and `topo_dim` the topological one; they differ only for a
collapsed set (§3). Bounds, center and per-axis projection:

```@example geometry
extrema(X), extrema(X, 1), center(X)
```

```@example geometry
projection(X, 1)
```

### Point containment

`∈` accepts a number in 1D, and a tuple, `SVector` or `Vector` above it:

```@example geometry
0.5 ∈ I, 1.5 ∈ I, (0.5, 0.5) ∈ Ω_2d, (1.2, 0.3) ∈ Ω_2d
```

---

## 3. Collapsed and lower-dimensional geometries

A dimension is **collapsed** when its interval is degenerate ($a = b$), which is how a
surface or interface embedded in a higher-dimensional space is written:

```@example geometry
line_in_2d = interval(0.0, 1.0) × point(0.0)

dim(line_in_2d), topo_dim(line_in_2d)
```

[`Bramble.is_collapsed`](@ref) answers per axis. It is public but not exported, hence the
`Bramble.` prefix:

```@example geometry
Bramble.is_collapsed(line_in_2d, 1), Bramble.is_collapsed(line_in_2d, 2)
```

---

## 4. Boundary markers

A marker names a piece of the boundary so a condition can later be attached to that name.

### 4.1 Boundary symbol conventions

Boundary symbols are coordinate-aligned, so the same name means the same face in every
dimension:

- **1D**: `:xmin` (`:left`), `:xmax` (`:right`)
- **2D**: `:xmin` (`:left`), `:xmax` (`:right`), `:ymin` (`:bottom`), `:ymax` (`:top`)
- **3D**: `:xmin` (`:back`), `:xmax` (`:front`), `:ymin` (`:left`), `:ymax` (`:right`), `:zmin` (`:bottom`), `:zmax` (`:top`)

The viewpoint-dependent names in parentheses are aliases and keep working.
[`boundary_symbols`](@ref) lists the canonical set for a given dimension:

```@example geometry
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

### 4.2 Creating markers

A marker is a `:label => identifier` pair, where the identifier is one boundary symbol, a
tuple of them, or a predicate:

```@example geometry
geom = interval(0.0, 5.0) × interval(0.0, 1.0)

m1 = markers(
    geom,
    :inflow => :left,
    :outflow => :right,
    :wall => (:top, :bottom)
)

collect(labels(m1))
```

`labels` flattens the three marker kinds through `Iterators.flatten`. Inside a loop that
must not allocate, iterate `label_symbols`, `label_tuples` or `label_conditions` instead.

### 4.3 Predicate and time-dependent markers

A predicate marks any subset, not only a face, and a marker set built over a time interval
takes `(p, t)` predicates:

```@example geometry
using LinearAlgebra

m_cond = markers(
    geom,
    :inflow => :left,
    :hot_spot => (p -> p[1] > 2.5 && p[2] ≈ 0.0)
)

m_time = markers(
    geom,
    interval(0.0, 10.0),
    :moving_source => ((p, t) -> norm(p .- [t, 0.5]) < 0.2)
)

m_time(1.5)      # the marker set frozen at t = 1.5
```

---

## 5. Computational domains

A [`Domain`](@ref) is a set together with its markers, and it is what [`mesh`](@ref) takes:

```@example geometry
Ω = domain(
    interval(0.0, 1.0) × interval(0.0, 1.0),
    :dirichlet => (:left, :right),
    :neumann => (:top, :bottom)
)

collect(labels(Ω))
```

`domain(geom)` alone marks the whole external boundary `:boundary`.

### 5.1 Domain traits

A `Domain` forwards the geometric queries of §2 to its set:

```@example geometry
dim(Ω), center(Ω), (0.5, 0.5) ∈ Ω, Bramble.is_collapsed(Ω)
```

```@example geometry
Bramble.set(Ω)
```

---

## 6. Three domains, three dimensions

The same two lines in 1D, 2D and 3D. A rod held at one end and insulated at the other:

![1D Rod Domain](../assets/geometry_example1_1d_rod.svg)

```@example geometry
rod = domain(interval(0.0, 10.0), :dirichlet => :left, :neumann => :right)

dim(rod), center(rod), collect(labels(rod))
```

A channel with inflow, outflow and no-slip walls:

![2D Channel Flow Domain](../assets/geometry_example2_2d_channel.svg)

```@example geometry
channel = domain(
    interval(0.0, 5.0) × interval(0.0, 1.0),
    :inflow => :left,
    :outflow => :right,
    :wall => (:top, :bottom)
)

dim(channel), center(channel)
```

A heat sink, heated below, cooled above, insulated on its four sides:

![3D Heat Sink Domain](../assets/geometry_example3_3d_heatsink.svg)

```@example geometry
sink = domain(
    interval(0.0, 2.0) × interval(0.0, 2.0) × interval(0.0, 1.0),
    :heat_source => :zmin,
    :convection => :zmax,
    :insulated => (:xmin, :xmax, :ymin, :ymax)
)

center(sink), collect(labels(sink))
```

Next: [meshes](mesh.md), which discretize a domain into points.
