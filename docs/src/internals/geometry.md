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

The validation and the three labels of a polygon marker are in the
[Geometry API page](../api/geometry.md#Polygons); the tolerance that classifies the nodes
on a mesh is in the [mesh internals page](mesh.md).

## Pretty printing

```@autodocs
Modules = [Bramble]
Public = false
Pages = ["geometry/pretty_print.jl"]
```
