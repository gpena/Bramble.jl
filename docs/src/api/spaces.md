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

Moving a grid function from one mesh to another — the piecewise (multi)linear interpolant,
named after [`Rₕ`](@ref)/[`Rₕ!`](@ref)'s own `Xₕ`/`Xₕ!` convention. One name, `πₕ`, with
methods that dispatch tells apart by what they are given rather than by different names:

- `πₕ(Wₕ, uₕ)` and [`πₕ!`](@ref)`(dest, src)` — the **numeric** operator, interpolating a grid
  function's values onto another space's mesh. [`interpolate_at`](@ref) is the single-point
  building block both are written in terms of, and [`interpolation_matrix`](@ref) is the same
  interpolant as a sparse matrix rather than applied pointwise.
- `πₕ(uₕ)` — the **symbolic source**, wrapping a grid function's interpolant as an AST leaf,
  composable with [`D₋ₓ`](@ref)/[`Mₓ`](@ref)/... inside [`innerₕ`](@ref). For the *known*
  side of a linear form.
- `πₕ(u)` over a **trial function** — the **bilinear operator**, contributing matrix columns
  rather than values. For the *unknown* side. It names no source space: that is the trial
  function's own, and assembly supplies it once the leaf is known
  ([#10](https://github.com/gpena/Bramble.jl/issues/10)).

See the [operators tutorial](../tutorials/operators.md) for the numeric side and the pattern
this exists for: a heterogeneous composite space whose leaves live on different meshes.

```@docs
interpolate_at
πₕ!
πₕ
interpolation_matrix
```
