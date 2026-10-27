# What each replay kernel argument is used for (v4.1.4, S5)

Answers #916 step 2: for each replay kernel of the bilinear assembly, how many method
instances it has after one `form` and `assemble` of the 3D elasticity form and of the 3D
vector Laplacian, how many distinct types each argument takes, and whether the kernel's
body uses that argument's type in a grid-point loop or only hands it to a callee. It
changes no code. The candidates for #916 step 3 are at the end.

## Method and conditions

`benchmark/kernel_specializations.jl` starts a fresh `julia` per row. The child loads
Bramble, builds a graded 5x5x5 mesh and `gridspace(Ωₕ)^Val(3)`, then calls `form` and
`assemble` once. For each kernel and each of its methods it walks
`Base.specializations(m)` and records the instances and, per argument, the distinct
concrete types in that slot. Rows are the two of `benchmark/trace_compile.jl`:

```julia
# elasticity
(u, v) -> 2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₘₕ(u), divₘₕ(v))
# vector_laplacian
(u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))
```

Command, one `julia` with 1 thread, through `julia_slot.sh 1`, nothing else of mine
running (counts only, no timing):

```
julia --startup-file=no benchmark/kernel_specializations.jl \
    --out benchmark/results/kernel_specializations.toml
```

The figures are in `benchmark/results/kernel_specializations.toml`. Commit `af40fc12` on
`perf/v4.1.4-measure`, Julia 1.13.2, default `-O`.

What the counts are and are not:

- A count is every instance in the process after the form, the ones loaded from the package
  image included. The image holds the precompile workload's instances (860 in these ten
  kernels before the form is built). The TOML's `baseline_instances` is that count; the
  form adds the total minus the baseline.
- An instance is not always compiled code. In the elasticity row 204 of the 967 instances
  have no compiled code, only inference: 139 of the 144 `_replay_unit!` and all 9
  `_replay_pair_unit!` (re-counted in this process through `mi.cache` and `invoke`). They
  are the `@inline` serial methods at `src/assembly/bilinear_execution.jl:427` and `:429`,
  which inline into their callers. An inlined method's instances cost inference, not code.
- An instance with an abstract signature, from a method compiled without full
  specialisation (mostly image entries), counts in `abstract_types`, not in
  `distinct_types`.
- The closures of each walk (`unit` at lines 300, 373, 467 and 517) are separate instances
  and are not in the counts. Nor is `_replay_segment!` (line 138), the callee that holds
  the serial walk's code.
- `_replay_pair!` has no instance in either row, and neither have the scalar-summand
  kernels `_foreach_summand_unit` and `_replay_summands!` any form-added instance. Both
  rows have a composite trial space, which takes `_is_block_pair` (`bilinear_execution.jl:204`) and so the
  block kernels. The summand kernels' instances come from the image, where scalar forms
  were precompiled.

## Instances per kernel

The positive control holds: elasticity has more kernel instances than the vector Laplacian
(967 against 877; the form adds 107 against 17).

| kernel | elasticity | form adds | inferred only | vector Laplacian | form adds |
|---|---:|---:|---:|---:|---:|
| `_foreach_summand_unit` | 159 | 0 | 6 | 159 | 0 |
| `_foreach_block_unit` | 29 | 13 | 8 | 20 | 4 |
| `_replay_summands!` | 171 | 0 | 8 | 171 | 0 |
| `_replay_blocks!` | 27 | 13 | 6 | 18 | 4 |
| `_replay_pair!` | 0 | 0 | 0 | 0 | 0 |
| `_foreach_pair_block_unit` | 9 | 9 | 0 | 0 | 0 |
| `_replay_pair_blocks!` | 9 | 9 | 0 | 0 | 0 |
| `_replay_unit!` | 144 | 12 | 139 | 135 | 3 |
| `_replay_pair_unit!` | 9 | 9 | 9 | 0 | 0 |
| `_visit_guarded_region!` | 410 | 42 | 28 | 374 | 6 |
| total | 967 | 107 | 204 | 877 | 17 |

`inferred only` is the elasticity row, counted in this process.

Of the 42 instances `_visit_guarded_region!` gains in the elasticity row, 21 take a
`_CoordSink` (the recording walk), 12 a `ReplaySink` and 9 a `_PairReplaySink`.

## Reading the arguments

The verdict column uses three words.

- `hot loop`: the body has a per-grid-point loop that reads the argument's type. Only
  `_visit_guarded_region!` has one (`src/assembly/bilinear_traversal.jl:959` and `:969`).
- `unit code`: the body reads the type once per unit, to unroll over a sum's summands,
  route a term to its blocks or choose a branch. Its loops run over blocks (at most 9 here),
  not grid points.
- `passed on`: the body only forwards the argument to a callee.

Counts are `distinct_types` for elasticity, then for the vector Laplacian (`VL`) where they differ.
The `A` argument is `SparseMatrixCSC` (1 or 2 element types) and `α` is `Bool` or
`Float64`; both reach the sink's fields in the callee, so a type that one kernel passes on
is read by a later kernel's hot loop. A `passed on` verdict says only that this body does
not read it.

### `_foreach_summand_unit` (setup walk over a scalar sum)

| method | argument | types | verdict | decided by |
|---|---|---:|---|---|
| 266 | `f` | 1 | passed on | `bilinear_execution.jl:268` and `:269` call or forward it |
| 266 | `op` | 33 | unit code | `_summands(op)` at line 271, unrolled by `_foldl_pairs` (`symmetry.jl:307`) |
| 266 | `sp` | 5 | passed on | forwarded at lines 268 and 269 |
| 276 | `f` | 1 | passed on | the single call at line 277 |
| 276 | `term` | 109 | passed on | only handed to `f` at line 277 |
| 276 | `sp` | 7 | passed on | only handed to `f` at line 277 |

The body of method 276 is one call: its optimized code is 2 statements.

### `_foreach_block_unit` (setup walk over a composite sum)

| method | argument | types | verdict | decided by |
|---|---|---:|---|---|
| 281 | `f` | 1 | passed on | forwarded at lines 285 and 286 |
| 281 | `op` | 5 | unit code | `_summands(op)` at line 288 |
| 281 | `trial_leaves`, `test_leaves` | 3, 3 | passed on | forwarded at lines 285 and 286 |
| 297 | `f` | 1 | passed on | the call at line 304 |
| 297 | `term` | 16, VL 7 | unit code | `_bind_interp_spaces(term, …)` line 301 rebuilds the term; `trial_component_or_nothing(term)` line 308 routes it |
| 297 | `trial_leaves`, `test_leaves` | 3, 3 | unit code | `_component_blocks` (line 307, `block_extract.jl:215`) builds the `Block` tuple type; `_block_run` (line 311, `block_extract.jl:225`) picks the loop at line 312 |

### `_foreach_pair_block_unit` (setup walk of a transposed pair)

| method | argument | types | verdict | decided by |
|---|---|---:|---|---|
| 453 | `f` | 1 | passed on | the calls at lines 475, 477 and 478 |
| 453 | `t1`, `t2` | 9, 9 | unit code | `_bare_product` (lines 456, 457), `_component_blocks` (458 to 465), `_pair_blocks_ok` (line 466) |
| 453 | `trial_leaves`, `test_leaves` | 1, 1 | unit code | `_component_blocks` at lines 458 to 465 |

### `_replay_summands!` (replay over a scalar sum)

| method | argument | types | verdict | decided by |
|---|---|---:|---|---|
| 322 | `mode` | 2 | passed on | forwarded at lines 327 and 328 |
| 322 | `A` | 1 (+3 abstract) | passed on | forwarded at lines 327 and 328 |
| 322 | `op` | 33 | unit code | `_summands(op)` at line 330 |
| 322 | `sp`, `segments` | 5, 3 | passed on | forwarded at lines 327 and 328 |
| 322 | `next`, `α` | 1, 2 | passed on | forwarded at lines 327 and 328 |
| 334 | `mode` | 2 | passed on | picks the `_replay_unit!` method (427 or 807) at line 339 |
| 334 | `A` | 2 | passed on | line 339 |
| 334 | `term` | 109 | passed on | line 339 |
| 334 | `sp` | 7 | passed on | line 339 |
| 334 | `segments` | 3 | passed on | `segments[next]` at line 339 reads a `Segment{D}` element, `D` only |
| 334 | `next`, `α` | 1, 2 | passed on | line 339 |

### `_replay_blocks!` (replay over a composite sum)

| method | argument | types | verdict | decided by |
|---|---|---:|---|---|
| 343 | `mode`, `A`, `segments`, `next`, `α` | 2, 1 (+3 abstract), 3, 1, 1 | passed on | forwarded at lines 354 and 355 |
| 343 | `op` | 5 | unit code | `_summands(op)` at line 359 |
| 343 | `trial_leaves`, `test_leaves` | 3, 3 | passed on | forwarded at lines 354 and 355 |
| 363 | `mode`, `A`, `next`, `α` | 1, 1, 1, 1 | passed on | the call at lines 379 to 381 |
| 363 | `segments` | 3 | passed on | `segments[n]` at line 380 |
| 363 | `term` | 16, VL 7 | unit code | `_bind_interp_spaces` line 374, `_component_blocks` line 384 |
| 363 | `trial_leaves`, `test_leaves` | 3, 3 | unit code | `_component_blocks` line 384, `_block_run` line 388, loop at line 389 |

### `_replay_pair!`

No instance in either row, so no argument to read. Its body is the call at line 439.

### `_replay_pair_blocks!` (replay of a transposed pair, composite)

| method | argument | types | verdict | decided by |
|---|---|---:|---|---|
| 493 | `mode`, `A`, `next` | 1, 1, 1 | passed on | the `_replay_pair_unit!` calls at lines 527 to 538 |
| 493 | `t1`, `t2` | 9, 9 | unit code | `_bare_product` (504, 505), `_term_scale` (515, 516), `_pair_blocks_ok` (514) |
| 493 | `trial_leaves`, `test_leaves` | 1, 1 | unit code | `_component_blocks` at lines 506 to 513 |
| 493 | `segments` | 1 | passed on | `segments[n]`, `segments[n + 1]`, `segments[n + 2]` at lines 528 to 538 |
| 493 | `α` | 1 | passed on | scaled at lines 515 and 516, then forwarded as `α1`, `α2` |

### `_replay_unit!` (one unit's walk)

| method | argument | types | verdict | decided by |
|---|---|---:|---|---|
| 427 (serial, `@inline`) | first | 1 | passed on | selects this method; unused in the body |
| 427 | `A` | 2 | passed on | `_replay_segment!` call at lines 427 and 428 |
| 427 | `term` | 125, VL 116 | passed on | same call; the walk is `_replay_segment!` (line 138) |
| 427 | `sp` | 6 | passed on | same call |
| 427 | `row_offset`, `col_offset` | 1, 1 | passed on | same call |
| 427 | `segment` | 3 | passed on | same call |
| 427 | `α` | 2 | passed on | same call |
| 807 (threaded) | first | 1 | passed on | selects this method |
| 807 | `A` | 1 | passed on | `_DiagonalReplayTarget(A, …)` line 812, `ReplaySink(A, …)` line 815 |
| 807 | `term` | 4 | unit code | `_has_test_interp(term)` line 826 and `_term_colour_strides(term)` line 829 read the type at compile time |
| 807 | `sp` | 3 | passed on | forwarded to `_sweep_unit!` at lines 812 and 814 |
| 807 | `segment` | 2 | unit code | `_diagonal_replay(Val(D)) && segment.is_diagonal` at line 811 |
| 807 | `α` | 1 | passed on | stored in the sink at lines 812 and 815 |

### `_replay_pair_unit!` (one pair unit's walk)

| method | argument | types | verdict | decided by |
|---|---|---:|---|---|
| 429 (serial, `@inline`) | `A`, `sp`, `segment`, `α1`, `α2`, `half` | 1 each | passed on | `_replay_pair_segment!` call at lines 429 to 432 |
| 429 | `p1` | 9 | passed on | same call |
| 429 | `_p2`, `_offsets2` | 9, 1 | passed on | unused in the serial body, as the comment above line 427 says |
| 839 (threaded) | all | 0 | none | no instance in either row |

### `_visit_guarded_region!` (the guarded walk, `bilinear_traversal.jl:948`)

| argument | types (abstract) | verdict | decided by |
|---|---:|---|---|
| `sink` | 23 (10), VL 22 | hot loop | `_sink_point!` lines 961 and 972; `_visit_entries` line 973; `_factored_walk(typeof(sink), …)` line 958 |
| `term` | 135 (1), VL 117 | hot loop | `local_stencil(term, sp, …)` line 971; `_factored_point!` line 962 |
| `sp` | 7 (1) | hot loop | `local_stencil(term, sp, …)` line 971; `_factored_point!` line 962 |
| `mesh_markers` | 2 (1) | hot loop | `local_stencil(…, mesh_markers, …)` line 971 |
| `lin_indices` | 3 (1) | hot loop | `lin_indices[I]` lines 960 and 970 |
| `region` | 3 (1) | hot loop | `for I in region` lines 959 and 969 |
| `row_offset`, `col_offset` | 1, 1 | passed on | `_visit_entries` line 973; one type, so nothing to specialise |

`_visit_interior!` (line 979) is the unguarded twin and is `@inline`, so it has no instance
of its own.

## Unstable calls

For one instance of each kernel in the elasticity row, `Base.code_typed_by_type` (the
inference that `@code_warntype` prints) over its `specTypes`, once unoptimized and once
optimized. Where the form added instances, the instance is one of them;
`_foreach_summand_unit` and `_replay_summands!` have none, so theirs comes from the image
(a 2D `Float32` form, not elasticity). The instances of `_replay_unit!` and
`_replay_pair_unit!` are inferred-only ones. The script is not kept: it is a short loop
around `code_typed_by_type`.

| kernel | form-added | optimized statements | unoptimized calls typed `Any` | optimized non-concrete results |
|---|---|---:|---:|---|
| `_foreach_summand_unit` (276) | no | 2 | 1 (the `f` call) | 1 `invoke`, callee returns `Nothing` |
| `_foreach_block_unit` (297) | yes | 163 | 1 (the `unit` closure) | 1 `invoke` of `_CoordPass`, returns `Nothing` |
| `_replay_summands!` (334) | no | 21 | 1 (`_replay_unit!`) | none |
| `_replay_blocks!` (363) | yes | 182 | 0 | none |
| `_foreach_pair_block_unit` (453) | yes | 311 | 1 (the `unit` closure) | 3 `invoke` of `_CoordPass`, return `Nothing` |
| `_replay_pair_blocks!` (493) | yes | 120 | 0 | none |
| `_replay_unit!` (427) | yes, inferred only | 2 | 0 | none |
| `_replay_pair_unit!` (429) | yes, inferred only | 1038 | 0 | 2 `invoke` of `_visit_guarded_region!`, return `Nothing` |
| `_visit_guarded_region!` (948) | yes | 1743 | 1 (`_factored_point!`) | none |

None of the nine instances examined has a dynamic call or a non-concrete return type. The `Any` results are
calls to a known callee (`_CoordPass`, a closure, `_replay_unit!`, the generated
`_factored_point!`); in the optimized code each is an `invoke` of a fixed `CodeInstance`
whose own return type is `Nothing` (or `Int64`), so nothing is dispatched at run time.
The non-concrete slots are `iterate` results, `Union{Nothing, Tuple{Block, Int64}}`, which
the optimizer splits. In `_visit_guarded_region!` the slots typed `Union{}` are the
unguarded branch of the factored term, dead where `_factored_walk` folds to `true`
(`bilinear_traversal.jl:958`). JET `report_opt` was not run.

## Candidates

Argument by argument, and only where the verdict above is `passed on`. The count is what
would collapse, from the `specTypes` of the concrete instances in the elasticity process:
instances equal in every other argument merge into one. It is not a measured saving in
compile time, and it counts instances in the image as well as the form's own.

| kernel and argument | verdict | instances | form-added | lines |
|---|---|---:|---:|---|
| `_replay_summands!` (334): `term` | passed on | 128 to 11 | 0 | `bilinear_execution.jl:339` |
| `_replay_summands!` (334): `term`, `A`, `mode`, `α`, `sp` | passed on | 128 to 3 | 0 | `bilinear_execution.jl:339` |
| `_foreach_summand_unit` (276): `term`, `sp` | passed on | 119 to 1 | 0 | `bilinear_execution.jl:277` |
| `_replay_summands!` (334): `A`, `mode`, `α` only | passed on | 128 to 119 | 0 | `bilinear_execution.jl:339` |

What the table says:

1. The two worth trying are `@nospecialize term` on `_replay_summands!` (method 334) and on
   `_foreach_summand_unit` (method 276). Each body is one call, so the trampoline per term
   type buys nothing. `_foreach_summand_unit` 276 would drop 119 to 1 concrete instance;
   its callee, `(::_CoordPass)(term, …)` (`bilinear_pattern.jl:175`), is specialised on
   `TERM` anyway, so the instances that matter stay.
2. For `_replay_summands!` the same change makes the call at line 339 dynamic, and its
   callee is the `@inline` serial `_replay_unit!`: the 139 inferred-only instances of it
   would then be compiled as code of their own. The step 3 try has to count those, and a
   function barrier that calls `_replay_segment!` (line 138) directly may serve better than
   `@nospecialize`.
3. Both gains are in the image, not in the form: none of the 107 instances the elasticity
   form adds is in a kernel with a candidate. Neither row exercises these kernels, because
   a composite space takes the block path. The change shows in a first assemble of a new
   scalar form, such as a multi-term 2D form, and must be measured there (#914).
4. Not candidates, with the reason:
   - `_foreach_block_unit`, `_replay_blocks!` and the two pair kernels: every argument whose
     verdict is `unit code` steers block routing or unrolling. Dropping `trial_leaves` and
     `test_leaves`, or `A`, `mode`, `α` and `segments`, merges 0 of the 16 concrete
     instances of method 297 or 363, so the passed-on ones add no multiplicity.
   - `_replay_unit!` (427) and `_replay_pair_unit!` (429): their instances are inference
     only, inlined into the caller. They cost inference, not code, and are not a
     `@nospecialize` target on instance count alone.
   - `_visit_guarded_region!`: every argument but `row_offset` and `col_offset` is read in
     the hot loop. Dropping `mesh_markers`, `lin_indices` or `region` merges 0 of its 382
     concrete instances, and `sp` merges 23. The 42 form-added instances follow the (sink,
     term) pairs the loop is specialised on.

## Step 3 measured

Answers #916 "each tried change measured". There are two trees, detached checkouts of the
branch. The base is `f674c310` (before S5.1) and change 1 is `5a23aea1` (S5.1 alone, the
per-term-type layers of the scalar-sum replay kernels deleted and the summands handed
straight to the unit).
`benchmark/snoop_instances.jl` ran three launches per tree, one launch per call,
interleaved base, change 1, base, ... with `--threads 4`, default `-O`, the machine alone
on AC with OneDrive paused (1-minute load 1.7 to 2.5), rows `scalar_multiterm`,
`elasticity` and `sum40`. Medians of the three launches; the spread is max minus min over
them. Raw launches, loads and counts: `benchmark/results/v4.1.4_replay_step3.toml`.

| row | tree | instances | first `assemble` (s) | spread | warm `assemble` | spread |
|---|---|---:|---:|---:|---:|---:|
| `scalar_multiterm` | base | 192 | 0.3501 | 0.0190 | 1.375 µs | 0.17 µs |
| `scalar_multiterm` | change 1 | 188 | 0.3527 | 0.0078 | 1.209 µs | 0.001 µs |
| `elasticity` | base | 3719 | 16.457 | 0.064 | 167.9 µs | 2.1 µs |
| `elasticity` | change 1 | 3719 | 16.552 | 0.240 | 168.0 µs | 4.7 µs |
| `sum40` | base | 448 | 0.6770 | 0.0093 | 47.81 µs | 0.06 µs |
| `sum40` | change 1 | 448 | 0.7134 | 0.0098 | 42.29 µs | 0.15 µs |

`instances` are the Bramble method instances the first `form` and `assemble` infer in the
process, the same count in all three launches. Only `elasticity` and `scalar_multiterm`
gate the first-call rule; only `elasticity` (base warm time 168 µs, at or above 50 µs)
gates the warm clause. The warm times of `scalar_multiterm` (1.2 µs) and `sum40` (48 µs)
are reported and never gated: rows this short move 8 to 19% with code placement alone.

Per kernel, one fresh process per tree and row, after one `form` and `assemble`, every
method of the kernel through `Base.specializations`. Each cell is all instances, then the
ones with native code (`all/compiled`).

| kernel | elasticity base | elasticity change 1 | scalar base | scalar change 1 |
|---|---:|---:|---:|---:|
| `_foreach_summand_unit` | 162/154 | 38/34 | 165/157 | 39/35 |
| `_foreach_block_unit` | 53/21 | 53/21 | 40/8 | 40/8 |
| `_replay_summands!` | 172/164 | 51/35 | 175/167 | 52/36 |
| `_replay_blocks!` | 27/21 | 27/21 | 14/8 | 14/8 |
| `_replay_pair!` | 0/0 | 0/0 | 0/0 | 0/0 |
| `_foreach_pair_block_unit` | 9/9 | 9/9 | 0/0 | 0/0 |
| `_replay_pair_blocks!` | 9/9 | 9/9 | 0/0 | 0/0 |
| `_replay_unit!` | 145/5 | 161/5 | 135/5 | 151/5 |
| `_replay_pair_unit!` | 9/0 | 9/0 | 0/0 | 0/0 |
| `_visit_guarded_region!` | 360/336 | 366/336 | 322/298 | 328/298 |

Over all methods, `_foreach_summand_unit` falls from 162 to 38 instances (elasticity) and
165 to 39 (scalar), `_replay_summands!` from 172 to 51 and 175 to 52; compiled, from 154 to
34 and 157 to 35, and 164 to 35 and 167 to 36. The `_replay_unit!` compiled count stays at 5
in both trees; its inferred-only instances rise by 16 (145 to 161, 135 to 151), the
callee now inferred from the unit's caller. The kernels the change does not touch keep
their counts. These counts are in the process, the image included: the first `assemble`
infers 4 fewer Bramble instances on `scalar_multiterm` (192 to 188) and none fewer on
`elasticity` (3719 both), whose composite space takes the block path.

Verdict. The first-call `assemble` median of change 1 is not below the base's on
`scalar_multiterm` (0.3527 s against 0.3501 s, +2.7 ms, spreads 0.0078 and 0.0190 s) nor
on `elasticity` (16.552 s against 16.457 s, +95 ms, spreads 0.240 and 0.064 s): the head is
higher on both, inside the spreads, so no gain is shown and the criterion cannot be decided
beyond the spread. Under the undecided clause the change is kept only if the instances fall
and the rest of the rule holds; the instances fall on `scalar_multiterm` (192 to 188) and
do not on `elasticity` (3719), so the clause does not keep it. The warm clause holds on
`elasticity` (167.9 against 168.0 µs, inside the spread). Change 1 is **reverted**:
`bilinear_execution.jl` is back to `f674c310`. `sum40` (not gated) shows a first `assemble`
of 0.7134 s against 0.6770 s, +5.4%, beyond both spreads; its warm time falls from 47.81 to
42.29 µs, reported only.

Rejected before timing, recorded for #916: S5.1's first attempt (`@nospecialize` on the
term of `_foreach_summand_unit`, `c3d64b2f`, reverted at `c34e7d05`) failed the quality
group's zero-dispatch pins (`jet.jl:57`, `alloccheck.jl:261`), and three variants of the
`_replay_summands!` term method (all `@nospecialize(term)`; the body calling `_replay_unit!`,
the serial branch calling `_replay_segment!` directly, an extra barrier passing `segments`
and `next`) broke the 0 B warm `assemble!` pins (432 to 624 B), so none was committed.
