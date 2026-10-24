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
[Geometry API page](../api/geometry.md#Polygons). Orientation, folding back and crossing
are exact: a bounding-box test rejects most pairs of edges, a floating-point orientation
decides when it exceeds its error bound, and `Rational{BigInt}` decides the nearly
degenerate rest. The area is judged with a tolerance, `|2A| > 8nεL²` for `n` vertices and
extent `L`; a sliver below it is rejected as zero area. The tolerance that classifies the
nodes on a mesh is in the [mesh internals page](mesh.md).

## Level sets

```@autodocs
Modules = [Bramble]
Public = false
Pages = ["geometry/levelset.jl"]
```

The validation and the three labels of a level-set marker are in the
[Geometry API page](../api/geometry.md#Level-sets).

## Pretty printing

```@autodocs
Modules = [Bramble]
Public = false
Pages = ["geometry/pretty_print.jl"]
```
