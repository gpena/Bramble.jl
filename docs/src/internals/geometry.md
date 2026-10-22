```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Geometry

## Sets

```@autodocs
Modules = [Bramble]
Public = false
Filter = x -> x !== Base.extrema
Pages = ["geometry/set.jl"]
```

## Markers

```@autodocs
Modules = [Bramble]
Public = false
Pages = ["geometry/marker.jl"]
```

## Domains

```@autodocs
Modules = [Bramble]
Public = false
Pages = ["geometry/domain.jl"]
```

## Polygons

```@autodocs
Modules = [Bramble]
Public = false
Pages = ["geometry/polygon.jl"]
```

A polygon is validated at construction with exact predicates: at least three vertices, no
zero-length edge, nonzero area and a simple boundary. A vertex in the middle of a straight
edge is kept. As a marker value, `:P => polygon(V)` defines three labels, `:P`,
`:P_boundary` and `:P_exterior`, whose nodes are classified on a mesh with the tolerance
given in the [mesh internals page](mesh.md).

## Pretty printing

```@autodocs
Modules = [Bramble]
Public = false
Pages = ["geometry/pretty_print.jl"]
```
