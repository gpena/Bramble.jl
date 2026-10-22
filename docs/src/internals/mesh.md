```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Mesh

## Grid closure of polygons

`mesh` of a domain with polygon markers fits the grid to the polygons with `_grid_closure`
(internal), the closure described on the [Meshes API page](../api/meshes.md). Lines within
`max(1e-10, 64ε) L` of each other merge (`L` the polygons' extent), so crossings and
vertices sit on nodes up to that tolerance.

The oblique edges must share one slope magnitude; that is necessary, not
sufficient. A closure that grows past a number of lines proportional to the input throws
an `ArgumentError`, "unbounded or too fine for this grid", and a polygon small relative to
its offset from the origin may hit it, as the rounding of the closure then exceeds the
merge tolerance.

The nodes are then marked by their signed distance `s` to the boundary: on it when
`|s| <= τ`, inside when `s < -τ`, outside when `s > τ`, with
`τ = max(√ε hₘᵢₙ, 10³ ε L) + 4 ε M` and `M` the largest vertex coordinate magnitude.

## Ghost meshes

A [`GhostMesh`](@ref) is a physical mesh together with one ghost point beyond each node of
the facets it names. It wraps two meshes: the physical one, and a padded one of the same
kind (`Mesh1D` or `MeshnD`) with one extra point at each named facet end, at ``x_1 - h_2``
below and ``x_n + h_n`` above, the values `spacing(Ωₕ, 2)` and `forward_spacing(Ωₕ, n)`
already hold. The padded mesh is a real mesh with submeshes of its own, so every operator
family and the assembly walk read the ghosts with no change.

The wrapper forwards the accessors to the padded mesh: points, spacings and indices are the
padded ones, `npoints(G, Tuple)` is the padded size and `G(i)` the padded axis submesh. It
overrides the queries that depend on where the faces are, so that a facet means the physical
facet nodes: `boundary_indices`, `index_in_marker`, the surface weights and masks, and the
geometric markers. A facet label (`:xmin`, `:boundary`, an alias, or a label of the physical
mesh) is an index-derived stored marker on those nodes, never a predicate of position, so
the labels survive `change_points!` and the mesh type carries no function.

A ghost has a number among `1:nghosts(G)`: the named facets in the order `xmin, xmax, ymin,
ymax, zmin, zmax`, each facet's ghosts in the column-major order of its nodes. `ghost_index`
and `boundary_node` convert between a facet node and its ghost number in `O(1)`. A padded
point outside along two or more axes (a 2D corner, a 3D edge or corner) is absent: it is not
a node, has no ghost number and is never a degree of freedom, and `is_absent` tells it.

The padded points follow `change_points!` of the physical mesh through the version stamp of
its submeshes, and are recomputed on the access that finds them stale.
`iterative_refinement!` of the physical mesh changes the point count, which a ghost mesh
cannot follow: every later access throws an `ArgumentError`, and a new space is needed.
`gridspace` of a `GhostMesh` is refused, because it would weigh the ghost and absent points
as nodes: a space with ghosts is built from the physical mesh with
`gridspace(Ωₕ; ghosts)` (see [Spaces](space.md)).

```@autodocs
Modules = [Bramble]
Public = false
Pages = ["mesh/interface.jl", "mesh/indices.jl", "mesh/constructors.jl", "mesh/queries.jl", "mesh/marker.jl", "mesh/mesh1d.jl", "mesh/meshnd.jl", "mesh/pretty_print.jl"]
```
