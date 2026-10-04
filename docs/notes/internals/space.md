```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Spaces

## Weight storage for the staggered inner product family

Two issues decide this together: `SpaceWeights`'s dense storage (gpena/Bramble.jl#115)
and the full `2^D`-member staggered inner product family (gpena/Bramble.jl#234).

`SpaceWeights` weights every discrete inner product a [`ScalarGridSpace`](@ref) offers.
Before this section's change it stored two dense, full-grid vector families: `innerh`
(the ``L^2`` cell measures) and `innerplus`, one vector per spatial direction. Both are
separable tensor products of `D` one-dimensional vectors, computed from exactly those
per-axis pieces and then thrown away -- an `O(n^D)` cost the underlying data never
needed. On a `100³` mesh the four dense vectors already cost 32,000,000 bytes; a naive
extension to the full `2^D` staggered family this milestone adds (pairs, the full-`D`
combination -- needed for a mimetic elasticity discretisation and the discrete Sobolev
embedding, #234) would have meant eight dense vectors instead of four, roughly doubling
that figure, and worse in higher `D`.

### What is stored, what is computed on demand

`SpaceWeights` carries two `NTuple{D}` fields of *per-axis* vectors, each of length
`npoints(Ωₕ, d)` rather than the full grid:

  - `aligned[d]`: the factor ``h_d(i)`` used on the axis a difference is taken along
    (today's `innerplus[d]`'s own construction).
  - `cellfactor[d]`: the factor ``h_d(i+1/2)`` used on every transverse axis, which is
    also `innerh`'s own per-axis cell measure. The two coincide for any axis with more
    than one point -- both read the submesh's cached half-spacings vector -- so
    `cellfactor` is a zero-copy reference to that vector, not a second fill. (They can
    differ on a topologically collapsed, single-point axis, where `cell_measures`
    coerces the degenerate zero to one and the old transverse fill did not; nothing in
    the suite exercises `inner₊` on such an axis, so this was a documented judgement
    call, not a measured regression -- see `_innerplus_mean_weights!`'s own docstring
    for the boundary-weight history this interacts with.)

`weights(Wₕ, Val(S))` answers any staggered set `S ⊆ 1:D` from these two tuples: entry
`I` is ``\prod_{d \in S} h_d(I_d) \cdot \prod_{d \notin S} h_d(I_d + 1/2)``. `S = ()`
(`weights(Wₕ, Innerh())`) and every singleton `S = (d,)` (`weights(Wₕ, Innerplus(), d)`)
return `innerh`/`innerplus[d]` directly -- the identical `SeparableWeights` object built
once, at `gridspace` construction time, from `aligned`/`cellfactor`, and returned
unchanged on every call rather than recomputed. Every other `S` (a pair, or the full-`D`
combination) builds a fresh `SeparableWeights` on every call instead, from the same two
tuples; it is not cached, since nothing yet asks for the same such set twice in a hot
loop. Either way, nothing `weights` can return for any `S` is ever a plain `Vector` over
the whole grid.

### The four pre-existing families stop being eagerly dense

That last sentence was not always true. The change that added the per-axis factors, and
first wrote this section, kept `innerh` and `innerplus` themselves as dense, full-grid
vectors on purpose: the two places that read a weight in a hot loop -- `_dot`/`_dot_masked`
(`src/space/inner_product.jl`) for the numeric `innerₕ`/`inner₊`, and `compute_weight`
(`src/operators/inner.jl`) for the symbolic ones inside a form -- had no way yet to read a
`SeparableWeights` without paying a division per axis on every point. Keeping
`innerh`/`innerplus` densely materialised was the only way to guarantee those two hot paths
were unaffected by the per-axis-factor change at the time.

`_dot`/`_dot_masked` then gained a `SeparableWeights` specialization that walks
`CartesianIndices(w.dims)` directly instead of converting a flat index, and
`compute_weight` gained a matching `CartesianIndex` read, for the `InnerPlusSet` node
built for `|S| ≥ 2`. Once both existed, keeping `innerh`/`innerplus` dense had nothing
left to protect, and gpena/Bramble.jl#115 removed it: `space_weights` now builds them
the same way as every other `S`, from `aligned`/`cellfactor` alone, with no
full-grid vector filled anywhere in the function. `SpaceWeights` has no dense branch left
-- every weight it returns, for every `S`, is a `SeparableWeights`.

That change left one asymmetry, which gpena/Bramble.jl#428 has since removed. At the time,
`InnerH` and `InnerPlus{Dim}`'s own `compute_weight` read their weight by *linear* index,
not by the `CartesianIndex` the assembly loop already had in hand, so a symbolic `innerₕ(u,
v)` or `inner₊ₓ(u, v)` term paid the division-per-axis of a linear `SeparableWeights` access
once per point. Today all three
nodes, `InnerH`, `InnerPlus{Dim}` and `InnerPlusSet`, read a `SeparableWeights` by
`CartesianIndex` (`_weight_at` in `src/operators/inner.jl`); a dense weight vector, which no
current `weights` method returns, would still be read by linear index. Removing the division
speeds up every symbolic inner-product term; the measured figures are in gpena/Bramble.jl#428.

### The one-dimensional case

`space_weights(Ωₕ::AbstractMeshType{1})` (`src/space/scalar_gridspace.jl`) wraps its
single per-axis vector in a `SeparableWeights{1}` the same way the `D ≥ 2` method wraps
`D` of them -- there is no separate, dense-vector code path kept for one dimension, even
though a one-factor product has nothing left to separate. The comment above that method
calls this measured rather than assumed, and the measurement is narrower than it might
sound: a `SeparableWeights{1}`'s linear `getindex` converts index `i` to
`CartesianIndices((n,))[i]`, which for a one-dimensional shape is the identity, and then
reads `factors[1][i]` -- the same single array access a plain `Vector`'s own `getindex`
already is, with nothing left for the wrapper to add. A `SeparableWeights{1}` is therefore
no slower to read element by element than a plain `Vector`.

### Allocation and construction

One `assemble!` refill of `innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))` still allocates zero
bytes, matching the documented `assemble!` contract, and refill time did not regress when
the dense families went. `gridspace` construction fills no `O(n^D)` vector: it allocates
the `D` per-axis factor vectors and the space structs, while `cellfactor` is a zero-copy
reference to the mesh's own cached half-spacings.

### Why the trade is worth taking

Eliminating the last `O(n^D)` storage costs nothing extra: since the 2026-09-25 line-walk
rewrite, `_dot` walks axis-1 lines with the product of the other axes' factors hoisted out
once per line and reduces each line with `@inbounds @simd` (`src/space/inner_product.jl`),
so `innerₕ`/`inner₊` over `SeparableWeights` are faster than a dense whole-vector
reduction (one fewer full-length vector to read), against weights whose storage is
`O(nD)` rather than `O(n^D)`. The one path that matters for a typical solve, `assemble!`,
is unaffected.

There is no whole-vector-reduction penalty left to pay for calling `innerₕ`/`inner₊` (or
`normₕ`/`norm₊`, built on them) in a per-iteration hot loop. The one place a penalty
remains is a grid whose axis 1 has only a few points, where the per-line overhead leaves
the reduction slower than a dense one.

### Adding a new weight consumer

For a full-grid sweep, walk axis-1 lines and hoist the other axes' factor product once
per line, the way the unmasked `_dot` and `_seminorm_sq_along` do, rather than reading a
weight by `CartesianIndex` (or a linear index converted to one) at every point. For scattered
indices with no line structure to hoist over, read by the `CartesianIndex` you already
hold instead, the way `_dot_masked` and `compute_weight`'s `InnerPlusSet` branch both do,
rather than by a linear index built from it. Either way, expect `weights(Wₕ, ...)` to hand
back a `SeparableWeights`, never a `Vector`, whichever inner product it names.

## Operator matrices: stencil_matrix versus the Kronecker construction

gpena/Bramble.jl#185 asked for a decision between three ways to build a discrete
operator's matrix (`D₋ₓ(Ωₕ)`, `Mᵧ(Ωₕ)`, `jump₂(Ωₕ)`, and the rest of the family) and,
where possible, for the separate implementations each operator carried -- pointwise
grid-function traversal, symbolic form-AST evaluation, and Kronecker matrix construction
-- to collapse into fewer.

### The three candidates

1. **Form-based assembly**, `H \ assemble(...)`: an operator matrix `L` is the bilinear
   form `a(u, v) = innerₕ(L(u), v)` assembled under the unweighted inner product and then
   left-divided by the diagonal metric `H = diag(mesh volumes)`. It reuses the existing
   assembly engine entirely, at the cost of building and discarding a form and a sparse
   solve purely to undo the weighting.
2. **Direct stencil-to-CSC traversal**, `stencil_matrix(Ωₕ, op)`: one sweep over
   `CartesianIndices(Ωₕ)` that writes `colptr`/`rowval`/`nzval` directly from each point's
   fixed neighbour offsets and weights, with no intermediate matrix at any stage.
3. **Lazy matrix-free operator**: represent an operator as a `LinearOperator` whose
   `mul!` evaluates its stencil in place, materialising a `SparseMatrixCSC` only on
   demand.

This milestone implements option 2 as `stencil_matrix`
(`src/operators/stencil_matrix.jl`), routing every family's public per-axis alias (`D₋`,
`D₊`, `D̃`, `Dc`, `D̽ₕ`, `jump`, `M`, `M₊`) through it. Option 3 is delivered separately, as
the Kronecker operator of gpena/Bramble.jl#162, which is built for a whole separable
bilinear *form* and not as a lazy mode of `D₋ₓ`/`Mᵧ`/etc. themselves. Option 1 was reasoned
through rather than prototyped: it shares `stencil_matrix`'s single sweep over the grid
in spirit, but adds a form, an assembly, and a sparse solve where `stencil_matrix` needs
none of the three, so it had nothing left to win on once option 2 existed.

### Why the Kronecker construction is kept

`src/operators/shift.jl` keeps its Kronecker-product construction, now named
`kronecker_operator_matrix`, rather than being deleted once `stencil_matrix` took
over every family's public alias. gpena/Bramble.jl#185's acceptance criterion is exact
agreement between the old and new matrices, and proving that needs two independent
constructions to compare -- checking `stencil_matrix`'s output against itself would prove
nothing. `test/space/operators.jl`'s "stencil_matrix agrees with the Kronecker oracle
(#185)" testset builds both for `D₋`, `D₊`, `D̃`, `Dc`, `D̽ₕ`, `jump`, `M` and `M₊`, along
every axis, in 1D/2D/3D, on non-uniform meshes, and asserts entrywise equality (`==`) and
matching `nnz`.

The routed alias allocates a fixed 9 times regardless of family, axis or mesh size, while
the Kronecker construction's count grows with how many Kronecker shifts and subtractions
the family composes (more for `Dc`'s two-sided stencil than for `D₋`'s one-sided one). It
is faster and uses less memory in every case `benchmark/operator_matrices.jl` measures, with
no regression.

### What unification was and was not achieved

`stencil_matrix`'s own `_stencil_taps`/`_stencil_weights` methods
(`stencil_matrix.jl`) are a second, reduced implementation of the same offsets and
coefficients the form layer already computes under the same names in the AST half of
`src/operators/{difference,average,jump}.jl`, for the AST node types
(`BackwardDifference`, `JumpNode`, and the rest) that back `local_stencil`. They are not
shared code: `stencil_matrix.jl` is included before the AST core and cannot depend on
form-layer AST nodes without inverting the package's own layering (forms are built on top
of the space layer's operators, not the other way around), so calling the form layer's
methods from here was never an option.
The two are held equal only by the equality test above, mesh point by mesh point, not by
a function either implementation calls.

This answers half of gpena/Bramble.jl#185's "eliminate logic triplication": the
Kronecker construction, the third implementation the issue named, is retired from every
production call site and demoted to a test oracle, but the matrix construction and the
pointwise construction (form-layer `local_stencil`) still disagree in code even though
they now agree in shape (both are keyed on a fixed set of neighbour offsets and matching
weights). Finishing the unification would need the form layer's AST nodes to expose
their offsets and weights through an interface `src/space/` can depend on -- a trait, or
a plain function taking the offsets and weights rather than a `LazyOp` tree -- which is
out of scope here and was not attempted.

### Adding a new operator family

A new `StencilOp` subtype implements `_stencil_taps` and
`_stencil_weights` for it and gets `stencil_matrix` for free; keeping
`kronecker_operator_matrix`'s equality check honest means also adding an oracle method
for the new family in `shift.jl` alongside it.

```@autodocs
Modules = [Bramble]
Public = false
Filter = x -> x ∉ (Base.parent, Base.:*, Bramble.ldiv!)
Pages = [
    "space/gridspace.jl",
    "space/scalar_gridspace.jl",
    "space/vector_gridspace.jl",
    "space/vectorelement.jl",
    "operators/projection.jl",
    "operators/restriction.jl",
    "operators/cell_average.jl",
    "operators/shift.jl",
    "operators/stencil.jl",
    "operators/stencil_matrix.jl",
    "operators/difference.jl",
    "operators/jump.jl",
    "operators/average.jl",
    "operators/interpolation.jl",
    "space/inner_product.jl",
]
```
