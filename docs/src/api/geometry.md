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

## Markers and domains

```@docs
markers
domain
labels
```

## Polygons

`polygon(V)` is exported; `signed_distance` is `public` and not exported. A polygon is a
marker value: `:P => polygon(V)` in a domain labels the nodes of a 2D mesh `:P` (interior),
`:P_boundary` and `:P_exterior`. The vertices are validated with exact predicates: at least
three, no zero-length edge, nonzero area, a simple boundary. A vertex in the middle of a
straight edge is kept. See [Meshes](meshes.md) for the grid fitted to a polygon and
[Grid spaces](spaces.md) for `polygon_spaces`.

```@docs
polygon
Bramble.signed_distance
```

## Level sets

`levelset(φ)` is exported. It wraps a function `φ(x)` as a 2D marker value, distinct from a
predicate: `:D => levelset(φ)` in a domain labels the nodes of a 2D mesh `:D` (where
`φ < 0`), `:D_boundary` (on the curve `φ = 0`) and `:D_exterior` (where `φ > 0`). `φ` must
return a real number, not a `Bool`, and need not be a signed distance. The labels follow the
rules of a polygon's: a 2D set, no reserved label, no clash with another marker's labels.
With a `time_set`, `φ` still takes `x` alone.

```@docs
levelset
```
