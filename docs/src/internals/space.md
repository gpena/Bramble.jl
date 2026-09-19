```@meta
CollapsedDocStrings = false
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

That last sentence was not always true. S6.2, which first wrote this section, kept
`innerh` and `innerplus` themselves as dense, full-grid vectors on purpose: the two
places that read a weight in a hot loop -- `_dot`/`_dot_masked`
(`src/space/inner_product.jl`) for the numeric `innerₕ`/`inner₊`, and `compute_weight`
(`src/form/operators/inner.jl`) for the symbolic ones inside a form -- belonged to
subplans S6.3 and S6.4, outside S6.2's own file ownership, and neither yet had a way to
read a `SeparableWeights` without paying a division per axis on every point. Keeping
`innerh`/`innerplus` densely materialised was the only way to guarantee those two hot
paths were unaffected by the per-axis-factor change at the time.

S6.3 then gave `_dot`/`_dot_masked` a `SeparableWeights` specialization that walks
`CartesianIndices(w.dims)` directly instead of converting a flat index; S6.4 gave
`compute_weight` a matching `CartesianIndex` read, for the `InnerPlusSet` node built for
`|S| ≥ 2`. Once both existed, keeping `innerh`/`innerplus` dense had nothing left to
protect, and this subplan (S6.8, gpena/Bramble.jl#115) removes it: `space_weights` now
builds them the same way as every other `S`, from `aligned`/`cellfactor` alone, with no
full-grid vector filled anywhere in the function. `SpaceWeights` has no dense branch left
-- every weight it returns, for every `S`, is a `SeparableWeights`.

One asymmetry survives this change and is worth naming. `InnerH` and `InnerPlus{Dim}`'s
own `compute_weight` still reads `weights(space, Innerh())[lin_idx]` by *linear* index,
not by the `CartesianIndex` the assembly loop already has in hand -- only the
`InnerPlusSet` node S6.4 added reads by `CartesianIndex`. So a symbolic `innerₕ(u, v)` or
`inner₊ₓ(u, v)` term inside a form now pays the same division-per-axis a linear
`SeparableWeights` access always costs, once per assembled point, where before it read a
dense vector directly. That is not a regression in the numbers below: a `CartesianIndex`
conversion is a small fraction of what `local_stencil` already spends on each point
(several operators' own stencils, Dirichlet handling, the sparse write itself), and
nothing measured here moves outside the range already on record for the same form.

### The one-dimensional case

`space_weights(Ωₕ::AbstractMeshType{1})` (`src/space/scalar_gridspace.jl`) wraps its
single per-axis vector in a `SeparableWeights{1}` the same way the `D ≥ 2` method wraps
`D` of them -- there is no separate, dense-vector code path kept for one dimension, even
though a one-factor product has nothing left to separate. The comment above that method
calls this measured rather than assumed, and the measurement is narrower than it might
sound: a `SeparableWeights{1}`'s linear `getindex` converts index `i` to
`CartesianIndices((n,))[i]`, which for a one-dimensional shape is the identity, and then
reads `factors[1][i]` -- the same single array access a plain `Vector`'s own `getindex`
already is, with nothing left for the wrapper to add. Measured directly (a `10^6`-point
1D space, summing every entry by linear index, minimum of 15 back-to-back calls): 0.109
ms for the `SeparableWeights{1}` against 0.109 ms for the plain `Vector` underneath it, a
ratio of 0.997 -- no detectable difference, confirming the comment.

That claim is about single-element access, not about the whole-vector reduction
`_dot`/`_dot_masked` runs. Measured the same way on the same 1D space, `_dot` against the
`SeparableWeights{1}` costs 0.857 ms to the 0.230 ms dense reduction it replaces, a ratio
of 3.73 -- essentially the same penalty measured on the `100³` space below, not a smaller
one for having only one axis. Dimension is not what the `_dot` specialization's cost
tracks: its loop walks `CartesianIndices(w.dims)` with no `@simd` annotation, where the
dense reduction and the generic `_dot` both carry one, and that gap is present whether
`w.dims` has one entry or three. The fast path a single factor buys is real, but it
belongs to `getindex`, not to reduction: uniformity buys nothing extra for `_dot` at
`D = 1`, and costs nothing extra either -- the same fixed penalty this section measures
at `D = 3` below.

### Measured

Machine state at measurement time was mixed, not clean: the session started on battery
power (97%, discharging, load average 2.29 against 8 cores) and had moved to AC power
partway through (charging, load average 4.6-6.1), because a `julia` language-server
process for this editor was already running, not because of a competing test or
benchmark run. Per `bramble-verification` §2/§9, absolute milliseconds under those
conditions are not trustworthy alone; every figure below is a same-process,
back-to-back, minimum-of-`N` measurement (`N ≥ 5`, warmed up first), and the ratios
between paired measurements are the load-bearing numbers, not the raw times.

`Base.summarysize(weights(Wₕ))` on the `100³` mesh this subplan's `CHECK` targets:

| stage | summarysize |
|---|---|
| before #115 (4 dense vectors) | 32,000,000 B |
| after S6.2 (per-axis factors added, 4 families still dense) | 32,005,288 B |
| after S6.8 (no family left dense) | 5,288 B |

The `CHECK` script reproduces the last row directly (`summarysize(weights) on 100^3:
5288 B`). That is a ~6,053x reduction from the S6.2 figure, and a ~6,053x reduction from
the original dense one as well, since the twelve small per-axis vectors S6.2 added cost
the same fixed handful of kilobytes regardless.

`innerₕ(u, u)` and `inner₊ₓ(u, u)` on the same `100³` space, against a dense-vector
reduction over the identical values (the same `muladd`/`@simd` shape `_dot` uses for a
plain `AbstractVector`), minimum of 7 back-to-back calls each, warmed up first:

| call | time | dense equivalent | ratio |
|---|---|---|---|
| `innerₕ(u, u)` | 0.868 ms | 0.231-0.240 ms | 3.6-3.8x |
| `inner₊ₓ(u, u)` | 0.868 ms | 0.227-0.234 ms | 3.7-3.8x |

Both agree with the dense reduction to `rtol = 1e-12`.

`_dot` itself, called directly rather than through `innerₕ`, minimum of 15 back-to-back
calls: `SeparableWeights` 0.868-0.892 ms against the dense vector's 0.180-0.182 ms, a
ratio of 4.81-4.90. `@which` confirms dispatch reaches the `CartesianIndex`-walking
specialization (`src/space/inner_product.jl:333`), not the generic `AbstractVector`
method (`src/utils/linear_algebra.jl:408`) -- so this is the intended path, not the
linear-`getindex` fallback. The ratio measured here sits near that fallback's own ≈4.9x
figure rather than nearer the ≈2.475x S6.3 recorded for this same specialization; the
likely reason, from reading the loop (previous section): it carries no `@simd`
annotation. The load and power state moved during this session (above), so a second
contributor cannot be ruled out; both figures are reported rather than one silently
preferred, per `bramble-verification`.

One `assemble!` refill of `innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))` on the uniform `60³` mesh
S6.2 used (`ndofs = 216,000`, `nnz(A) = 1,490,400`), warmed up, minimum of 5, at 4
threads (this repository's standard, `bramble-benchmarks` §1):

| stage | min time | range | allocation |
|---|---|---|---|
| S6.2 (4 dense families) | 4.04 ms | 4.04-4.97 ms | 0 B, all 5 runs |
| S6.8 (no dense families) | 3.16-3.19 ms | 3.16-4.58 ms | 0 B, all 5 runs |

No regression: the refill sits at or below S6.2's own range, and allocation is still zero
on every run, matching the documented zero-allocation `assemble!` contract. The
division-per-axis `InnerH`/`InnerPlus{Dim}`'s `compute_weight` now pays (previous
section) does not show up here -- it is a small fraction of everything else one
assembled point costs.

`gridspace` construction on the same `100³` mesh, warmed up, minimum of 7: 0.00008-0.00013
ms (83-125 ns), allocating 2,784 B. That allocation is the `D = 3` per-axis `aligned`
vectors plus the `SpaceWeights`/`ScalarGridSpace` structs themselves; `cellfactor` is a
zero-copy reference to the mesh's own cached half-spacings and adds nothing. Construction
was not timed before this change and cannot safely be now, since reproducing the pre-#115
code would mean checking out a different tree; what is measured is that today's
construction fills no `O(n^D)` vector, allocates in the hundreds of bytes rather than
tens of megabytes, and completes in well under a microsecond on a `10^6`-point mesh.

### Why the trade is worth taking

Eliminating the last `O(n^D)` storage costs roughly a 3.7-4.9x slower `innerₕ`/`inner₊`
whole-vector reduction, against weights that no longer scale with the grid at all: 5,288
B instead of 32,005,288 B on `100³`, a figure that would only have grown had the `2^D`
staggered family S6.3/S6.4 added stayed dense alongside it. The one path that matters for
a typical solve, `assemble!`, is unaffected -- it does not move outside the range already
on record.

The case that would make this the wrong trade: an algorithm that calls
`innerₕ`/`inner₊` (or `normₕ`/`norm₊`, built on them) as its own per-iteration hot loop,
at a frequency comparable to `assemble!` itself, on a grid small enough that the old
dense storage was still affordable. A Krylov solver checking a residual norm once per
outer iteration does not qualify -- that is `O(1)` calls per solve, not one per assembled
point -- but a method recomputing `normₕ` on every inner-loop pass over a grid well under
`100³` would pay the 3.7-4.9x penalty often enough to matter, with nothing to show for it
in memory saved.

### Adding a new weight consumer

Read a weight by the `CartesianIndex` you already hold, the way `compute_weight`'s
`InnerPlusSet` branch and `_dot`/`_dot_masked` both do, rather than by a linear index
built from it -- and expect `weights(Wₕ, ...)` to hand back a `SeparableWeights`, never a
`Vector`, whichever inner product it names.

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
(`src/space/operators/stencil.jl`), routing every family's public per-axis alias (`D₋`,
`D₊`, `D̽`, `Dc`, `Dₕ`, `jump`, `M`, `M₊`) through it. Option 3 is delivered separately, as
`KroneckerLinearOperator` (gpena/Bramble.jl#162) -- a matrix-free operator built
for a whole separable bilinear *form*, not a lazy wrapper around one operator's matrix
call -- rather than as a lazy mode of `D₋ₓ`/`Mᵧ`/etc. themselves. Option 1 was reasoned
through rather than prototyped: it shares `stencil_matrix`'s single sweep over the grid
in spirit, but adds a form, an assembly, and a sparse solve where `stencil_matrix` needs
none of the three, so it had nothing left to win on once option 2 existed.

### Why the Kronecker construction is kept

`src/space/operators/shift.jl` keeps its Kronecker-product construction, now named
`kronecker_operator_matrix`, rather than being deleted once `stencil_matrix` took
over every family's public alias. gpena/Bramble.jl#185's acceptance criterion is exact
agreement between the old and new matrices, and proving that needs two independent
constructions to compare -- checking `stencil_matrix`'s output against itself would prove
nothing. `test/space/operators.jl`'s "stencil_matrix agrees with the Kronecker oracle
(#185)" testset builds both for `D₋`, `D₊`, `D̽`, `Dc`, `Dₕ`, `jump`, `M` and `M₊`, along
every axis, in 1D/2D/3D, on non-uniform meshes, and asserts entrywise equality (`==`) and
matching `nnz`.

### Measured

`benchmark/operator_matrices.jl`, run directly (`JULIA_DEPOT_PATH="$TMPDIR/depot-s103:
$HOME/.julia" julia --startup-file=no --threads=4 --project=benchmark
benchmark/operator_matrices.jl`) on 2026-09-19, `kronecker_operator_matrix` ("old")
against the routed public alias ("new"):

| operator | mesh | old time | new time | speedup | old allocs | new allocs | old bytes | new bytes |
|---|---|---|---|---|---|---|---|---|
| `D₋ₓ` | 1000 (1D) | 17.96 μs | 3.62 μs | 5.0x | 54 | 9 | 161.1 KiB | 40.2 KiB |
| `Dcₓ` | 1000 (1D) | 24.96 μs | 4.30 μs | 5.8x | 78 | 9 | 225.6 KiB | 40.2 KiB |
| `Mₓ` | 1000 (1D) | 19.92 μs | 3.39 μs | 5.9x | 63 | 9 | 201.3 KiB | 40.2 KiB |
| `D₋ₓ` | 300² (2D) | 1.32 ms | 317.8 μs | 4.2x | 72 | 9 | 8.29 MiB | 3.44 MiB |
| `Dcₓ` | 300² (2D) | 1.71 ms | 351.8 μs | 4.9x | 114 | 9 | 8.32 MiB | 3.44 MiB |
| `Mₓ` | 300² (2D) | 1.63 ms | 278.2 μs | 5.9x | 81 | 9 | 11.72 MiB | 3.44 MiB |
| `D₋ₓ` | 60³ (3D) | 3.29 ms | 840.9 μs | 3.9x | 73 | 9 | 19.85 MiB | 8.16 MiB |
| `Dcₓ` | 60³ (3D) | 4.36 ms | 844.3 μs | 5.2x | 116 | 9 | 19.82 MiB | 8.03 MiB |
| `Mₓ` | 60³ (3D) | 4.16 ms | 716.7 μs | 5.8x | 82 | 9 | 28.06 MiB | 8.16 MiB |

Allocation count is flat at 9 regardless of family, axis or mesh size, while the old
construction's count grows with how many Kronecker shifts and subtractions the family
composes (more for `Dc`'s two-sided stencil than for `D₋`'s one-sided one). Time improves
3.9-5.9x and memory drops to roughly a fifth to a half across every case; neither
reduction is uniform across mesh size or family, and no case regresses.

Machine state, per `bramble-verification`: on battery (72%, discharging), load average
2.56/4.92/10.02 at the time of the run, with at least one other Julia process from a
concurrent session in this same worktree active during the measurement (an unrelated
`inner₊` check). The absolute times above should be read with that in mind; the ratios
are what this record relies on, and they land in the same range (4-6x, 9 flat
allocations against 54-116) that S10.2's own evidence already reported from a separate
run.

### What unification was and was not achieved

`stencil_matrix`'s own `_stencil_taps`/`_stencil_weights` methods
(`stencil.jl`) are a second, reduced implementation of the same offsets and coefficients
the form layer already computes under the same names in
`src/form/operators/{difference,average,jump}.jl`, for the AST node types
(`BackwardDifference`, `JumpNode`, and the rest) that back `local_stencil`. They are not
shared code: `src/space/` cannot depend on form-layer AST nodes without inverting the
package's own layering (forms are built on top of the space layer's operators, not the
other way around), so calling the form layer's methods from here was never an option.
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
Pages = [
    "space/gridspace.jl",
    "space/scalar_gridspace.jl",
    "space/vector_gridspace.jl",
    "space/vectorelement.jl",
    "space/operators/projection.jl",
    "space/operators/restriction.jl",
    "space/operators/cell_average.jl",
    "space/operators/shift.jl",
    "space/operators/stencil.jl",
    "space/operators/difference.jl",
    "space/operators/jump.jl",
    "space/operators/average.jl",
    "space/operators/interpolation.jl",
    "space/inner_product.jl",
]
```
