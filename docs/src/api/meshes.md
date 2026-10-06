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

## Immutable mesh state

A mesh keeps its geometry in an immutable state. A [`Mesh1D`](@ref) holds one
`Mesh1DState`, and a [`MeshnD`](@ref) builds a `MeshnDState` on demand from its submeshes'
states. The state holds plain arrays and isbits values: the points and spacings, the indices,
the backend, a version, whether the points are uniform, the marker bits as one word matrix
and an identity that survives every mutation. The mesh itself keeps the public marker
dictionary returned by `markers`. Every mutation below replaces the state or updates its arrays
in place, and changes the version, so a state taken before the call is stale. See the
[mesh internals page](../internals/mesh.md) for the state types.

The dictionary that `markers(Ωₕ)` returns is read-only. Editing it in place is unsupported:
the word matrix assembly reads is not rebuilt, and a label added that way has no column. To
change the labels use `markers!` or `set_markers!` (both private; see the same internals
page), or `change_points!` and `iterative_refinement!`, which re-evaluate them.

## Mesh adaptation and mutation

```@docs
iterative_refinement!
change_points!
set_points!
```
