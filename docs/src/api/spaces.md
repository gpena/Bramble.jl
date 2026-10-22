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

## Space properties and degrees of freedom

`host_weights` is private; see the [mesh internals page](../internals/mesh.md).

```@docs
ndofs
weights
spaces
space
ncomponents
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
methods that dispatch tells apart by what they are given rather than by different names:

- `πₕ(Wₕ, uₕ)` and [`πₕ!`](@ref)`(dest, src)` is the **numeric** operator, interpolating a grid
  function's values onto another space's mesh. [`interpolate_at`](@ref) is the single-point
  building block both are written in terms of, and [`interpolation_matrix`](@ref) is the same
  interpolant as a sparse matrix rather than applied pointwise.
- `πₕ(uₕ)` is the **symbolic source**, wrapping a grid function's interpolant as an AST leaf,
  composable with [`D₋ₓ`](@ref)/[`Mₓ`](@ref)/... inside [`innerₕ`](@ref). For the *known*
  side of a linear form.
- `πₕ(u)` over a **trial function** is the **bilinear operator**, contributing matrix columns
  rather than values. For the *unknown* side. It names no source space: that is the trial
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
Ferreira and Grigorieff on a polygon marked `:P` of a 2D mesh. `public` and not exported:
write `Bramble.polygon_spaces`. The support of each operation is in the
[space internals page](../internals/space.md).

```@docs
Bramble.polygon_spaces
```

## Level-set spaces

`levelset_spaces(Ωₕ, Ω, :D, g)` returns the trial and test spaces `(U, W)` of the
Shortley-Weller scheme on a level set marked `:D => levelset(φ)` in the domain `Ω` of a 2D
mesh. `public` and not exported: write `Bramble.levelset_spaces`. The support of each
operation is in the [space internals page](../internals/space.md).

```@docs
Bramble.levelset_spaces
```
