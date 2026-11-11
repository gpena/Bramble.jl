```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Grid spaces

## Function spaces

```@docs
ScalarGridSpace
CompositeGridSpace
VectorGridSpace
gridspace
vector_gridspace
```

`Wₕ^Val(N)` is type stable for any `N`. `Wₕ^N` accepts `1 ≤ N ≤ 3` and stays inferable for
a runtime `N`; a larger `N` throws an `ArgumentError`. `vector_gridspace(Ωₕ, 2)` builds the
same space straight from a mesh, and `Bramble.CompositeGridSpace((Wₕ, Wₕ))` builds one from
spaces that need not be identical:

```@example spaces
using Bramble
import Bramble: component_range, component_ranges, weights

Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 11))
Vₕ = Wₕ^Val(2)

spaces(Vₕ) == (Wₕ, Wₕ)
```

## Space properties and degrees of freedom

`host_weights` is private; see the [space internals page](../internals/space.md).

```@docs
ndofs
weights
spaces
space
ncomponents
```

`weights(Wₕ)` returns the whole `SpaceWeights` bundle, and a weight vector comes from naming
the inner product it belongs to. `Bramble.Innerh` and `Bramble.Innerplus` are internal, hence
the prefix. `weights` is not defined for a composite space, since its leaves may live on
different meshes: call it on a [`components`](@ref) leaf instead.

```@example spaces
w = weights(Wₕ, Bramble.Innerh())

length(w), sum(w)      # one weight per point; they tile the unit interval
```

## Vector elements and grid functions

`ldiv!(::VectorElement, ::Factorization, ::AbstractVector)` is private; see the
[CSR solvers internals page](../internals/csr_solvers.md).

```@docs
VectorElement
element
Base.parent(::VectorElement)
Base.reshape(::VectorElement{<:ScalarGridSpace})
components
component_range
component_ranges
Base.:*(::Function, ::VectorElement)
```

`component_range` and `component_ranges` give the degrees of freedom each component owns:

```@example spaces
component_range(Vₕ, 1), component_ranges(Vₕ)
```

Degrees of freedom are stored flat, but a scalar element also indexes by grid coordinate
with a `CartesianIndex`, in any dimension. An integer multi-index `u[i, j]` is not a grid
coordinate. Here a two-dimensional space:

```@example spaces
W₂ = gridspace(mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (5, 5), (true, true)))
V₂ = W₂^Val(2)
u_scal = element(W₂, 0.0)
u_scal[CartesianIndex(2, 3)] = 10.0
u_scal[CartesianIndex(4, 1)] = 20.0

u_scal[CartesianIndex(2, 3)], u_scal[CartesianIndex(4, 1)]
```

For matrix operations or plotting, `reshape(uₕ)` returns a `Base.ReshapedArray` view of the
same memory, so writing through it writes through to the element. On a composite element it
returns one view per component:

```@example spaces
u_grid = reshape(u_scal)
u_grid[2, 3] = -1.0

size(u_grid), u_scal[CartesianIndex(2, 3)], size.(reshape(element(V₂)))
```

## Restriction and averaging operators

```@docs
Rₕ
Rₕ!
avgₕ
avgₕ!
```

## Interpolation between grid spaces

Moving a grid function from one mesh to another, the piecewise (multi)linear interpolant,
named after [`Rₕ`](@ref)/[`Rₕ!`](@ref)'s own `Xₕ`/`Xₕ!` convention. One name, `πₕ`, with
methods selected by their arguments:

- `πₕ(Wₕ, uₕ)` and [`πₕ!`](@ref)`(dest, src)` is the **numeric** operator, interpolating a grid
  function's values onto another space's mesh. [`interpolate_at`](@ref) is the single-point
  building block both are written in terms of, and [`interpolation_matrix`](@ref) is the same
  interpolant as a sparse matrix.
- `πₕ(uₕ)` is the **symbolic source**, wrapping a grid function's interpolant as an AST leaf,
  composable with [`D₋ₓ`](@ref)/[`Mₓ`](@ref)/... inside [`innerₕ`](@ref). For the *known*
  side of a linear form.
- `πₕ(u)` over a **trial function** is the **bilinear operator**, contributing matrix columns.
  For the *unknown* side. It names no source space: that is the trial
  function's own, and assembly supplies it once the leaf is known.

This exists for the pattern of a heterogeneous composite space whose leaves live on different
meshes.

```@docs
interpolate_at
πₕ!
πₕ
interpolation_matrix
```

## Ghost spaces

A grid space with ghost nodes beyond named facets, built by
[`gridspace`](@ref)`(Ωₕ; ghosts = ...)`. `constrained` turns boundary relations on it into a
space of unknowns. Exported: `constrained`, `GhostVector` (the grid function of a time loop)
and `update_ghosts!`, which fills its ghosts. `public` and not exported: `lift`, which is
also a Makie name, so write `Bramble.lift`, and `lift!`, `physical`, `ghost_values` and
`GhostGridSpace`. `assemble!` also refills a pair of ghost forms.

```@docs
GhostGridSpace
constrained
lift
Bramble.lift!
GhostVector
update_ghosts!
Bramble.physical
Bramble.ghost_values
assemble!(::AbstractMatrix, ::AbstractVector, ::Bramble.GhostBilinearForm, ::Bramble.GhostLinearForm)
```

## Polygon spaces

`polygon_spaces(Ωₕ, :P, g)` returns the trial and test spaces `(U, W)` of the scheme of
Ferreira and Grigorieff on a polygon or level-set region marked `:P` of a 2D mesh. `public` and not exported:
write `Bramble.polygon_spaces`. See the
[space internals page](../internals/space.md) for the support table.

```@docs
Bramble.polygon_spaces
```
