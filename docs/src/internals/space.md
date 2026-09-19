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

`SpaceWeights` now also carries two `NTuple{D}` fields of *per-axis* vectors, each of
length `npoints(Ωₕ, d)` rather than the full grid:

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

`weights(Wₕ, Val(S))` answers any staggered set `S ⊆ 1:D` from these two tuples:
entry `I` is ``\prod_{d \in S} h_d(I_d) \cdot \prod_{d \notin S} h_d(I_d + 1/2)``.
`S = ()` and every singleton `S = (d,)` return `innerh`/`innerplus[d]` directly -- the
same dense vectors the four pre-existing families always have, unchanged, so nothing
reading them by linear index (`_dot`, `src/form/operators/inner.jl`'s `compute_weight`)
slows down. Every other `S` (the pairs and the full-`D` set new to this milestone)
returns a `SeparableWeights`: a lazy `AbstractVector` computed from `aligned`
and `cellfactor` at access time and never materialised over the whole grid. It is not
cached, since nothing yet asks for the same new set twice in a hot loop; a future
subplan that does (`inner₊(u, v, Val(S))`, #234) can add that once it exists.

This is a deliberate compromise rather than the fully lazy design the exploration notes
(`.agents/plans/v3-3-0-memory-scaling-notes.md` §2) measured as their headline number:
a lazy vector's linear `getindex` costs a division per axis, measured there at 2.4-4.9x
a dense read. `src/form/operators/inner.jl` and `src/space/inner_product.jl` -- the
consumers that read weights by linear index in assembly's inner loop and in `_dot` --
belong to other subplans of this milestone and are out of this one's file ownership, so
keeping the four pre-existing families dense was the only way to guarantee their
performance is unaffected here, verified rather than assumed (below).

### Measured

`Base.summarysize(weights(Wₕ))`, before this change and after, same mesh:

| mesh | before (4 dense vectors) | after (4 dense + 6 small per-axis) |
|---|---|---|
| `(6, 5, 4)`, 120 dofs | ≈ 3,840 B | 4,568 B |
| `(100, 100, 100)`, ``10^6`` dofs | 32,000,000 B | 32,005,288 B |

The absolute overhead (≈ 700-5,300 B, from the twelve small per-axis vectors) is fixed
by `D` and the per-axis point counts, not by the grid's total size, so it is a
measurable fraction of a tiny space and negligible (0.017% here) on a realistic one. The
counterfactual this buys: the pairs and the full-`(1,2,3)` set added by this milestone
cost this same fixed overhead combined, not another 32,000,000 B apiece.

One `assemble!` refill of `innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))` on a uniform `60³` mesh
(`ndofs = 216,000`, `nnz(A) = 1,490,400`), timed after a warm-up call, minimum of 5 runs
-- the same harness and form the exploration notes' baseline used (§2.4: 2.96 ms, 0 B,
range 2.96-4.24 ms across 5 runs on that same machine and session):

| threads | min time | range | allocation |
|---|---|---|---|
| 4 (this repository's standard, `bramble-benchmarks` §1) | 4.04 ms | 4.04-4.97 ms | 0 B, all 5 runs |
| 1 (matches the notes' own methodology exactly) | 5.60 ms | 5.60-15.3 ms | 0 B, all 5 runs |

Allocation is unchanged: zero bytes, matching the documented zero-allocation `assemble!`
contract. Time sits above the notes' own baseline, but not clearly outside the noise
that baseline already reports for identical code across five consecutive runs in one
process (a 43% spread, 2.96 to 4.24 ms) -- a different session, a different moment's
background load on the same machine cannot be ruled out as the explanation, and nothing
on the read path assembly exercises (`weights(Wₕ, Innerh())`, `weights(Wₕ, Innerplus(),
d)`) changed which object or algorithm answers it. Recorded here rather than asserted
either way, per `bramble-verification`.

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
