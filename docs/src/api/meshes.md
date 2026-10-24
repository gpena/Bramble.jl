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

A domain with polygon markers (`:P => polygon(V)`, see [Geometry](geometry.md)) gets a
grid fitted to them: every vertex coordinate is a grid line, and every crossing of a grid
line with an oblique edge adds a line, until each crossing is a node. The number of points
along an axis may therefore exceed `npts`. See [`mesh`](@ref) for the tolerances and the
errors, and the [mesh internals page](../internals/mesh.md) for the closure.

A domain with a level-set marker (`:D => levelset(φ)`, see [Geometry](geometry.md); one per
domain, none beside a polygon marker) gets a grid fitted to the curve `φ = 0` instead: a
polygon inscribed in the curve has its vertices on nodes and each edge a cell side or a
cell diagonal, so the labels `D`, `D_boundary` and `D_exterior` and
[`polygon_spaces`](@ref Bramble.polygon_spaces) work as for a polygon. The curve must lie
strictly inside the box, be orthogonally convex and be star-shaped about the minimum of `φ`.
[`mesh`](@ref) lists the errors; the [mesh internals page](../internals/mesh.md) says how
the labels are set.

On a fitted mesh, `hₘₐₓ` is no guide to the `h` of an error plot: the lines the closure adds
can sit very close to the seed lines and to each other. Take instead the largest gap between
neighbouring lines that hold nodes of the closed domain Ω̄, the interior and the boundary
nodes: for the disk, `maximum(diff(filter(t -> -1 <= t <= 1, z)))` over the points `z` of
each axis, and the larger of the two axes. This is the `spacing` of the
[disk example](../examples/poisson_circle.md), and the level-set convergence study measures it
the same way.

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
page). The three-argument forms, `change_points!(Ωₕ, markers(Ω), pts)` and
`iterative_refinement!(Ωₕ, markers(Ω))`, re-evaluate them at the new points. The
two-argument `change_points!(Ωₕ, pts)` keeps the face labels, which are index-based. It and
`set_points!` refuse a mesh carrying a label defined by a predicate of position, whose set
would go stale, and a move of a submesh of one; `change_points!(Ωₕ, markers(Ω), pts)` on the
whole mesh is the way to move it.

## Mesh adaptation and mutation

```@docs
iterative_refinement!
change_points!
set_points!
```

## Ghost meshes

`GhostMesh` is `public`, not exported: write `Bramble.GhostMesh`. A space built with
`ghosts` holds one; see [Grid spaces](spaces.md).

```@docs
GhostMesh
```
