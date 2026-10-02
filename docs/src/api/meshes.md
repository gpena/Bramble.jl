```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Meshes

## Mesh types and constructors

`AbstractMeshType`, `MeshMarkers` and `submeshes` are private; see the
[mesh internals page](../internals/mesh.md).

```@docs
Mesh1D
MeshnD
mesh
```

## Points and spacings

`host_points`, `host_spacings`, `forward_spacings`, `half_spacings`,
`host_half_spacings`, `stepsize`, `locate_cell` and `cell_measures` are private; see the
[mesh internals page](../internals/mesh.md).

```@docs
npoints
points
half_points
half_point
spacing
forward_spacing
half_spacing
spacings
hₘₐₓ
hₘᵢₙ
normal_vector
cell_measure
is_uniform
```

## Mesh indexing and boundaries

```@docs
indices
boundary_indices
interior_indices
is_boundary_index
index_in_marker
```

## Mesh adaptation and mutation

```@docs
iterative_refinement!
change_points!
set_points!
```
