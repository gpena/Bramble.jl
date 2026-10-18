# Warm assembly of the 3D elasticity form (#765, steps 1 and 2)

Question: where does a warm `assemble` of the 3D elasticity form spend its time, beyond what
the vector Laplacian and the scalar Laplacian cost per unknown? The answer decides what W2
changes in `bilinear_execution.jl`, `bilinear_traversal.jl` and `bilinear_pattern.jl`.

## Method

Three forms on one seeded non-uniform, non-cubic mesh of the unit cube
(`Random.seed!(20260903)`, `(n + 1, n, n - 1)` points, `n` = 30 or 45, so about 30³ or 45³
points):

- elasticity: `2.0 inner₊(εₕ(u), εₕ(v)) + 2.0 inner₊(divₘₕ(u), divₘₕ(v))` on
  `gridspace(Ωc)^Val(3)`;
- vector Laplacian: `inner₊(∇ₕ(u), ∇ₕ(v))` on the same composite space;
- scalar Laplacian: `inner₊(∇ₕ(u), ∇ₕ(v))` on `gridspace(Ωc)`.

`benchmark/warm_composite.jl` runs each form in a fresh child process (one Julia thread,
default `-O`). The child warms every path, then takes the min of 5 of warm `assemble!`
(the refill), warm `assemble`, and of each stage alone: the pattern copy
(`_allocate_assembled`), the zeroing (`_zero_stored!`) and the replay
(`_replay_bilinear_core!`). The timing table is the median over 3 launches at 30³ and one
launch at 45³. `--profile` samples warm `assemble` every 0.5 ms for about 6 s (30³) or 8 s
(45³) and counts, over the samples inside `assemble`, each function's inclusive share
(inlined frames included).

Figures: Apple M2 (4 performance cores), 16 GB, AC power, Julia 1.13.2, commit 07d1d645
(branch v4.0.0/rest), one Julia thread, no other Julia process. The 30³ timing run started
under load (1-min load 3.8, desktop app) that fell to 2.0 during it; the 45³ run and the
profiles ran at 1-min load 1.9 to 2.0. Per-launch spread at 30³: elasticity 134.8 to
136.3 ms, vector Laplacian 1.699 to 1.708 ms, scalar Laplacian 0.3138 to 0.3150 ms.

## Per unknown

Warm `assemble!`, 30³ (median of 3 launches):

| form | refill ms | ns per unknown | unknowns | stored per row | units | adds per stored | ns per add | `assemble` ms | bytes per `assemble` |
|:--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| elasticity | 136.1 | 1683 | 80910 | 75.7 | 26 | 4.01 | 5.51 | 139.7 | 98.7 MB |
| vector Laplacian | 1.706 | 21.1 | 80910 | 6.80 | 9 | 1.72 | 1.79 | 1.994 | 9.47 MB |
| scalar Laplacian | 0.314 | 11.6 | 26970 | 6.80 | 3 | 1.72 | 0.97 | 0.533 | 3.18 MB |

45³ (one launch): elasticity 478.8 ms, 1752 ns per unknown, 5.61 ns per add; vector
Laplacian 5.795 ms, 21.2 ns per unknown, 1.76 ns per add; scalar Laplacian 0.996 ms,
10.9 ns per unknown, 0.90 ns per add. Every warm `assemble!` allocates 0 B.

A unit is one (term, block) walk of the grid: the 35 summands of the elasticity form are
26 units, 9 of them a transposed pair walked once. An add is one value added to `nzval`
by the replay.

The elasticity form costs 80 times the vector Laplacian per unknown at 30³ (83 at 45³).
That factor is the product of three:

- 11.1 times the stored entries per row (75.7 against 6.80): the operator's own coupling,
  which no kernel change removes;
- 2.33 times the adds per stored entry (4.01 against 1.72): 26 units each add their own
  stencil, so one entry receives the values of up to 9 units at up to 4 points;
- 3.08 times the time per add (5.51 ns against 1.79 ns).

Only the last two are in W2's reach.

## Stages

Warm `assemble`, share of the samples inside it (30³ / 45³):

| stage | 30³ | 45³ |
|:--- | ---: | ---: |
| pattern copy (`_allocate_assembled`) | 7.5 % | 4.2 % |
| garbage collection (overlaps the copy) | 7.5 % | 4.2 % |
| zeroing, `_keep_pattern!` | 0.0 % | 0.0 % |
| replay (`_replay_bilinear_core!`) | 92.5 % | 95.8 % |
| per-block setup: `_bind_interp_spaces`, `_check_block_meshes`, `host_weights` | 0.0 % | 0.0 % |
| interior core (`_visit_interior!`) | 48.1 % | 61.9 % |
| guarded boundary shell (`_visit_guarded_region!`) | 42.6 % | 31.6 % |
| stencil evaluation (`local_stencil`) | 56.9 % | 63.1 % |
| of which the product of the two factors (`multiply_stencils_bilinear*`) | 27.1 % | 30.9 % |
| bounds guards (`_test_inbounds`, `_trial_inbounds`) | 7.7 % | 4.8 % |
| value addition (`_replay_add!`) | 8.8 % | 10.0 % |
| rest of the replay: entry walk, point slices, recorded positions | 19.1 % | 17.9 % |

Timed alone at 30³, the replay is 135.3 ms of the 136.1 ms refill (99.4 %), the zeroing
0.38 ms, the pattern copy 1.43 ms (min of 5, before the collections it triggers).
`allocate_system_matrix` is not on the warm path. It runs on the first `assemble` only; a
repeated one copies the kept pattern. The vector Laplacian at 30³, for contrast, spends
72.8 % in the replay and 27.2 % in the pattern copy (its fill is cheap next to its copy),
46.2 % in stencil evaluation and 12.4 % in value addition.

## The four questions

**Which stages dominate.** The replay, and inside it the stencil evaluation: 57 % to 63 % of
warm `assemble`. Per evaluated entry it costs about 4.3 ns against 0.95 ns for the vector
Laplacian at 30³ (684 evaluated entries per point against 36). Half of it forms the products
of the two factor stencils (`multiply_stencils_bilinear_symmetric` alone holds 18 % to 22 % self
time); the other half evaluates the factors (`_shifted_inner_stencil`,
`_reevaluated_shift`, `scale_stencil`, the `ntuple`s building them).

**Work repeated per term.** Yes. Each of the 26 units walks the whole grid and evaluates its
own two factors at every point: 52 factor evaluations per point. The distinct factors among
them were not counted. At an interior point the units add 972 values; one stencil per block,
the sum of that block's units, would add 576 (the distinct entries over the 9 blocks), 41 %
fewer.

**Guarded evaluation beyond the boundary layer.** Yes. `_stencil_margin` gives 2 for the
`εₕ` units and 3 for the `divₘₕ` units, but walking every point shows every unit needs a
margin of 1 (its reach is 1, and no point one cell in has an entry outside the grid). So the
guarded shell holds 43.0 % of the adds at 30³ (30.5 % at 45³) where 18.7 % (12.8 %) would
do. A guarded add costs about 1.18 times an interior one: each region's profile share over
its share of the adds is 0.99 against 0.84 at 30³, 1.04 against 0.89 at 45³.

**Sparse insertion in the replay.** No. Positions are recorded, so nothing is searched;
`_replay_add!` is 8.8 % to 10 % of the time, about 0.5 ns per add.

## Conclusion for W2

There is something to gain. The elasticity form pays 3.1 times the vector Laplacian per add
and 2.3 times its adds per stored entry. The 11 times more stored entries per row are its
own, so fused per block and as cheap per add as the vector Laplacian it would still cost
about 15 times as much per unknown. Candidates, ranked by the saving this profile supports
(shares of warm `assemble`):

1. Share factor evaluation across the units of a point. Ceiling: the factor evaluation
   itself, 30 % (30³) to 32 % (45³). Plausible 10 % to 15 %, if the 52 evaluations per
   point repeat each distinct factor at least twice (not measured). The factor and product
   kernels live in `src/operators/inner.jl` and `src/assembly/symmetry.jl`, outside W2's
   files, so W2 can only reorganise which units evaluate what.
2. One accumulated stencil per block, written once per point (972 to 576 adds per point).
   Ceiling: 41 % of the value addition and entry walk (27.9 % at 30³), about 11 %; the
   merge of the units' entries into the block's slots costs part of it back. Plausible
   5 % to 10 %. It changes the summation order of each entry, so the matrix agrees to
   rounding, not bit for bit.
3. Peel at the margin the stencil reaches, not `_stencil_margin`'s bound. Saving from the
   shell shares and the 1.18 cost ratio above: 3.6 % at 30³, 2.7 % at 45³. Cheap and
   local to the traversal (`_stencil_margin` itself is in `src/ast/stencil_pattern.jl`).
4. Not worth W2's time: the per-block setup (0.0 %), zeroing (0.3 %), sparse insertion
   (9 % to 10 %, already a recorded position per add), the pattern copy (1 % to 7.5 % with
   its collections, the price of `assemble` returning a new matrix).

Together, 1 to 3 plausibly save 20 % to 30 % of warm `assemble` at 30³ and 45³: from
1683 to about 1200 to 1350 ns per unknown at 30³, still 55 to 65 times the vector
Laplacian. The largest single share, the products of the factor stencils (27 % to 31 %),
stays under all three: every term still forms its product. A values cache for forms without
live data would cut far more, but it gives up re-reading the mesh weights each fill and is
not a kernel change.

## Reproduce

```bash
julia --startup-file=no benchmark/warm_composite.jl --size 30 --launches 3
julia --startup-file=no benchmark/warm_composite.jl --size 45 --launches 1
julia --startup-file=no benchmark/warm_composite.jl --profile --size 30 --seconds 6
julia --startup-file=no benchmark/warm_composite.jl --profile --row vector_laplacian --size 30
```
