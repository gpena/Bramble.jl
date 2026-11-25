# The out-of-line helpers on long stencils, warm and first call (four threads, 1a97bebf)

The last open line of #853 says a `@noinline` gate on stencils past 32 entries cut the first
`assemble` of `Δₕ³` in 3D from about 18 s to about 8 s at a warm cost of 2 to 5%, measured before
the concatenation fix, and asks for a re-measurement on today's code. The gate itself could not be
found, so this note measures the out-of-line helpers the code has instead.

## What this measures

The gate of the issue could not be identified. The issue gives no code, its comments are empty,
the commit messages (bb7716bd, 0a84b1e5) do not name it, and no other worktree holds a stale
diff of it. The issue's own number argues it is something else: the first `assemble` of `Δₕ³` in
3D takes 18.2 s on this tree, the figure the issue gives after the fix, and the gate was meant to
take that state to about 8 s.

What 1a97bebf does have is an out-of-line path past `_INLINED_ENTRIES` (32) in
`src/ast/common.jl`: three `@noinline` helpers, `_each_entry_outlined` (#759, 2026-10-14),
`_each_pair_outlined` (#759) and `_concatenate_outlined` (the concatenation fix, 2026-10-23). The
note compares them with building the same tuples inline. It does not test the issue's gate, so
it closes nothing in #853.

- **With the helpers**: the tree as it is.
- **Without**: the same tree with the three `@noinline` replaced by `@inline`, so the unrolled
  tuple is built inside every caller. The dispatch on length is unchanged. The changed lines of
  `src/ast/common.jl` are

  ```julia
  @inline _each_entry_outlined(f::F, s::Tuple) where {F} = _each_entry_inlined(f, s)
  @inline function _concatenate_outlined(left::Tuple, right::Tuple)
  @inline _each_pair_outlined(f::F, left::Tuple, right::Tuple) where {F} = _each_pair_inlined(
  ```

## Method

Two copies of the worktree under `/tmp/s312/with` and `/tmp/s312/without` (no `.git`), each
precompiled once before timing. One fresh `julia` per measurement, `--threads=4`,
`--startup-file=no`, default `-O`, alone on an M2 (1-min load 1.7 at the start), through
`julia_slot.sh 4`. Runs alternate between the trees back to back, and the order flips each round
(with, without; without, with; with, without), 3 rounds. Each process builds the form, times
`assemble(a)` (first call, load and precompile excluded), calls `assemble!` once more, then times
20 (`Δₕ³`) or 50 calls of `assemble!` and keeps their median. Every warm call allocated 0 B in
both trees. Meshes are graded (`mesh` with seed 774), the sizes are those of
`test/form/long_stencil_allocations.jl`:

| case | form | mesh | stencil entries |
|---|---|---|---|
| `ctl32` | `innerₕ(D₋ₓ(D₋ᵧ(D₋ₓ(D₋ᵧ(D₋ₓ(u))))), v)` | 2D, 11 × 10 | 32 |
| `Δₕ² 3D` | `innerₕ(Δₕ(Δₕ(u)), v)` | 3D, 7 × 6 × 5 | 81 |
| `Δₕ³ 2D` | `innerₕ(Δₕ(Δₕ(Δₕ(u))), v)` | 2D, 9 × 8 | 216 |
| `Δₕ³ 3D` | `innerₕ(Δₕ(Δₕ(Δₕ(u))), v)` | 3D, 7 × 6 × 5 | 729 |

The control sits below the threshold (32 is not past 32), so it must not move; the entry counts
come from `local_stencil` at an interior point.

The scripts are kept in `.claude/plans/alloc-guarantees-probes/` (`s312_measure.jl`,
`s312_driver.sh`, `s312_mf.jl`; gitignored). The two trees are `tar --exclude=.git -cf - . | tar
-xf - -C /tmp/s312/<tree>` of the worktree, the `without` one with the three lines above. One
measurement, with `TREE` one of the copies, `CASE` one of `ctl32`, `d2_3d`, `d3_2d`, `d3_3d`, and
`<main>` the main checkout:

```bash
cd /tmp/s312/$TREE && BRAMBLE_JULIA_SLOTS=4 <main>/.claude/scripts/julia_slot.sh 4 \
    julia --threads=4 --startup-file=no --project=. \
    <worktree>/.claude/plans/alloc-guarantees-probes/s312_measure.jl $CASE
```

The script (`s312_measure.jl`, shortened; `dims`, `n`, `f` and `reps` come from the table):

```julia
using Bramble, Random
using Bramble: D₋ₓ, D₋ᵧ
function graded_mesh(dims, n)
    Random.seed!(774)
    sets = (interval(0.0, 1.0), interval(0.0, 1.0) × interval(0.0, 1.0),
        box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    return mesh(domain(sets[dims]), n, ntuple(_ -> false, dims))
end
W = gridspace(graded_mesh(dims, n))          # (dims, n, f) from the table above
a = form(W, W, f)
t_first = @elapsed A = assemble(a)
assemble!(A, a)
ts = [(@elapsed Bramble.assemble!(A, a)) for _ in 1:reps]
warm = sort(ts)[div(reps, 2)]                # the warm figure; @allocated gave 0 B
```

## Result

Warm `assemble!`, median of the calls in each run, in ms. Spread is the largest minus the
smallest of the three runs; ratio is with the helpers over without, on the median of the three.

| case | with the helpers, 3 runs | spread | without, 3 runs | spread | ratio |
|---|---|---|---|---|---|
| `ctl32` | 0.0104, 0.0107, 0.0107 | 0.0003 | 0.0106, 0.0107, 0.0107 | 0.0001 | 1.00 |
| `Δₕ² 3D` | 0.0620, 0.0635, 0.0638 | 0.0018 | 0.0360, 0.0361, 0.0362 | 0.0002 | 1.76 |
| `Δₕ³ 2D` | 0.0937, 0.0954, 0.0959 | 0.0022 | 0.0390, 0.0399, 0.0406 | 0.0016 | 2.39 |
| `Δₕ³ 3D` | 1.5972, 1.5987, 1.6354 | 0.0382 | 0.5183, 0.5212, 0.5213 | 0.0030 | 3.07 |

First `assemble`, in seconds, fresh process.

| case | with the helpers, 3 runs | spread | without, 3 runs | spread | ratio |
|---|---|---|---|---|---|
| `ctl32` | 0.339, 0.349, 0.352 | 0.013 | 0.347, 0.349, 0.350 | 0.003 | 1.00 |
| `Δₕ² 3D` | 1.344, 1.346, 1.353 | 0.009 | 1.267, 1.271, 1.280 | 0.013 | 1.06 |
| `Δₕ³ 2D` | 2.672, 2.696, 2.696 | 0.024 | 2.284, 2.293, 2.321 | 0.037 | 1.18 |
| `Δₕ³ 3D` | 18.091, 18.221, 18.432 | 0.341 | 19.405, 19.474, 19.506 | 0.101 | 0.94 |

The control is identical in both trees, so the harness sees no difference where none exists. On
`Δₕ³` in 3D the helpers save 1.3 s of 19.5 s (6%) on the first call. On the two smaller long
stencils the first call is slower with them (6% and 18%).

## One more check, outside the plan: the matrix-free product

#759 added the out-of-line path because a long product's first matrix-free `mul!` took minutes,
so the same two trees ran one fresh process each (four threads, one run, no repeat) on
`matrix_free_operator(a)` over a 9 × 8 graded 2D mesh. 0 B per warm `mul!` in both. Not a
three-run figure: read it as a check that dropping the helpers does not blow the compile up, not as
a ratio.

| form | first `mul!`, with / without (s) | warm `mul!`, with / without (ms) |
|---|---|---|
| `innerₕ(Δₕ(Δₕ(Δₕ(u))), v)` | 3.11 / 2.13 | 0.146 / 0.069 |
| `innerₕ(Δₕ(Δₕ(u)), Δₕ(Δₕ(v)))` | 7.40 / 7.73 | 0.334 / 0.265 |

The 616 s of #759 was the unused gather's rows compiled through a `Bool` flag (`_mf_gathers`),
which the `Val` dispatch removed; neither tree shows it. The case #759 quotes
(`innerₕ(D̽ᵧD̽ₓD̽ₓu, D̽ᵧD̽ₓD̽ₓv)`, 729 entries) was not run in either tree.

## Reading

Warm `assemble!` is 1.8 to 3.1 times slower with the helpers on the three long stencils, far
outside the 0.6 to 4% spread of the runs without them, and the cost grows with the stencil.
Likely cause, not profiled: the call out of line returns the whole unrolled tuple (729 entries of
an offset triple and a weight) through memory at every grid point, where the inlined build can
keep it in registers. The first call gains 6% on `Δₕ³` in 3D and loses on the two smaller cases.

Dropping the three `@noinline` is a source change this subplan does not make. Its rule leaves
`src` untouched unless the measured gate is the faster tree, the issue's gate was not found, and
the matrix-free case #759 quotes was not run. A follow-up needs that 729-entry product and the
CpuPolyester `assemble!` run both ways, three alternating runs each, and a search for what the
issue's 8 s gate was.

Verdict: the issue's gate is unidentified; the three existing out-of-line helpers cost warm `assemble!` 1.8 to 3.1 times (spread under 4%) for no first-call gain worth the cost, `src/ast/common.jl` is untouched, and dropping them is a follow-up.

## Follow-up (S3.15): the matrix-free case of #759 and `CpuPolyester` (8725b63f)

S3.12 left two things unmeasured: the matrix-free case #759 added the helpers for, and
`assemble!` under `CpuPolyester`. This section runs both, plus Serial again, on the same two
trees: the helpers as on HEAD (`with`), and the three `@noinline` replaced by `@inline`
(`without`, the diff shown above).

### The bound #759 set

0a84b1e5 adds one numeric bound, in `test/form/common.jl` ("Past 32 entries (#759)"): the first
matrix-free `mul!` of `innerₕ(D̽ᵧ(D̽ₓ(D̃ₓ(u))), D̽ᵧ(D̽ₓ(D̃ₓ(v))))` on a 7 × 9 grid takes under
20 s (`@test t < 20`). That product has 324 entries. The 729-entry
`innerₕ(D̽ᵧD̽ₓD̽ₓu, D̽ᵧD̽ₓD̽ₓv)` on 9 × 8 comes from the commit message (0.2 s, then 616 s) and
has no bound. Both ran here (`mf324`, `mf729`); `local_stencil` at an interior point gives 324
and 729 entries.

The rule was fixed before the timed rounds. Dropping the helpers needs both of these:

- No warm row is slower without them, beyond the larger of the two spreads.
- No first call goes past 20 s without them, judged on the slowest of the three runs. A first
  call already past 20 s with the helpers must not get longer without them.

### Method

Same as S3.12. Two copies of the worktree at 8725b63f (`tar --exclude=.git`) under
`/tmp/s315/with` and `/tmp/s315/without`, each precompiled once before timing. One fresh
`julia --threads=4 --startup-file=no --project=.` per measurement, default `-O`, Julia 1.13.4,
through `BRAMBLE_JULIA_SLOTS=4 julia_slot.sh 4`, alone on the M2. Three rounds; within a round
each row runs with and without back to back, and the order flips each round (with first in
rounds 1 and 3, without first in round 2). `CpuPolyester` rows build the graded mesh with
`backend(policy = CpuPolyester())` and load Polyester from a temporary project pushed onto
`LOAD_PATH`. Each process prints the mesh's `execution_policy`, so a run that fell back to
serial would show. Matrix-free rows time the first `mul!` of `matrix_free_operator(a)`, then
the median of 50 warm calls, and check the product against `assemble(a) * x`. Assembled rows
time `assemble(a)`, then the median of 20 (`Δₕ³`) or 50 `assemble!` calls. Scripts:
`.claude/plans/alloc-guarantees-probes/s315_measure.jl` and `s315_driver.sh` (gitignored).

Preflight (`gate_preflight.sh --timing`) never passed. Before round 1 it failed for 30 min, and
before rounds 2 and 3 for 5 min each, on the `cpu` check (`suggestd` at 97 to 100% the whole
time, and `CGPDFService` at times) and on the 1-min load (2.0 to 3.3). Every round started
after a failed preflight: 1-min load 3.30, 2.31 and 2.27 at the start of rounds 1, 2 and 3. The
`ctl32` control matches S3.12 (0.0104 against 0.0104 to 0.0107 ms) and is flat across the trees,
and spreads stay under 1.4% warm. The background load seems to have shifted both trees alike;
the ratios are the figure to read.

### Result

Warm calls, median of each run, in ms. Spread is the largest minus the smallest of the three
runs; ratio is with over without, on the median of the three.

| case | policy | with, 3 runs | spread | without, 3 runs | spread | ratio |
|---|---|---|---|---|---|---|
| `ctl32` | Serial | 0.0104, 0.0104, 0.0105 | 0.0001 | 0.0104, 0.0104, 0.0105 | 0.0001 | 1.00 |
| `Δₕ² 3D` | Serial | 0.0616, 0.0618, 0.0620 | 0.0004 | 0.0350, 0.0350, 0.0351 | 0.0001 | 1.77 |
| `Δₕ² 3D` | Polyester | 0.0264, 0.0264, 0.0268 | 0.0004 | 0.0152, 0.0152, 0.0154 | 0.0002 | 1.74 |
| `Δₕ³ 2D` | Serial | 0.0918, 0.0925, 0.0928 | 0.0010 | 0.0386, 0.0387, 0.0388 | 0.0002 | 2.39 |
| `Δₕ³ 2D` | Polyester | 0.0233, 0.0234, 0.0235 | 0.0002 | 0.0095, 0.0096, 0.0098 | 0.0003 | 2.44 |
| `Δₕ³ 3D` | Serial | 1.5741, 1.5803, 1.5812 | 0.0071 | 0.5108, 0.5132, 0.5144 | 0.0036 | 3.08 |
| `Δₕ³ 3D` | Polyester | 0.6360, 0.6388, 0.6395 | 0.0035 | 0.1844, 0.1917, 0.1919 | 0.0075 | 3.33 |
| `mf324`, `mul!` | Serial | 0.0640, 0.0642, 0.0659 | 0.0019 | 0.0611, 0.0635, 0.0636 | 0.0025 | 1.01 |
| `mf729`, `mul!` | Serial | 0.1778, 0.1779, 0.1784 | 0.0006 | 0.1525, 0.1526, 0.1527 | 0.0002 | 1.17 |

First call (`assemble`, or the first `mul!` for `mf`), in s, fresh process.

| case | policy | with, 3 runs | spread | without, 3 runs | spread | ratio |
|---|---|---|---|---|---|---|
| `ctl32` | Serial | 0.322, 0.323, 0.331 | 0.009 | 0.322, 0.327, 0.328 | 0.006 | 0.99 |
| `Δₕ² 3D` | Serial | 1.296, 1.299, 1.438 | 0.142 | 1.223, 1.239, 1.263 | 0.040 | 1.05 |
| `Δₕ² 3D` | Polyester | 3.026, 3.079, 3.085 | 0.059 | 2.992, 3.017, 3.033 | 0.041 | 1.02 |
| `Δₕ³ 2D` | Serial | 2.627, 2.629, 2.632 | 0.005 | 2.243, 2.248, 2.280 | 0.037 | 1.17 |
| `Δₕ³ 2D` | Polyester | 5.432, 5.442, 5.593 | 0.161 | 5.400, 5.435, 5.512 | 0.112 | 1.00 |
| `Δₕ³ 3D` | Serial | 17.808, 17.819, 17.897 | 0.089 | 18.922, 19.103, 19.173 | 0.251 | 0.93 |
| `Δₕ³ 3D` | Polyester | 29.532, 29.906, 30.167 | 0.635 | 37.037, 37.335, 37.369 | 0.332 | 0.80 |
| `mf324` (bound 20 s) | Serial | 1.561, 1.566, 1.574 | 0.013 | 1.816, 1.837, 1.841 | 0.025 | 0.85 |
| `mf729` | Serial | 4.178, 4.206, 4.215 | 0.037 | 4.564, 4.600, 4.616 | 0.052 | 0.91 |

Warm bytes per call are equal in both trees: 0 B for every Serial `assemble!` and every `mul!`,
240 B for every `CpuPolyester` `assemble!`. The 240 B is there with the helpers too, so they do
not cause it; this subplan did not chase it.

### Reading

On the warm side, removing the helpers slows nothing: every long row is faster without them,
1.17 to 3.33 times, far outside the spreads, and the control does not move. `CpuPolyester` pays
the same price as Serial, slightly more on `Δₕ³` in 3D (3.33 against 3.08). For the
matrix-free product the gap is small (1.01 on `mf324`, within the spread; 1.17 on `mf729`).

The first call is what blocks the change. #759's own case stays far inside its bound either way
(`mf324` 1.84 s at worst without the helpers, against 20 s), as does `mf729` (4.62 s). But
`assemble` of `Δₕ³` in 3D under `CpuPolyester` goes past 20 s in both trees, and without the
helpers it takes 7.4 s longer (37.0 to 37.4 s against 29.5 to 30.2 s, spreads 0.3 and 0.6 s).
That breaks the second condition. Serial `Δₕ³` 3D also grows (19.2 s at worst, against 17.9 s)
and comes within 1 s of the bound.

Narrowing the threshold rather than dropping the path would keep the 729-entry tuples out of
line and inline the 81- and 216-entry ones, whose first calls do not grow. That is a different
tree from the one measured, with its own constant: `_INLINED_ENTRIES` cannot move, since
`_map_entries` and the `_LongStencil` dispatch need it for 0 B. It would need its own three
alternating rounds, which this subplan did not run, so `src/ast/common.jl` is unchanged.

Verdict (S3.15): `src/ast/common.jl` unchanged; dropping the three out-of-line helpers would make every warm long-stencil row 1.17 to 3.33 times faster, Serial and CpuPolyester alike, and keeps #759's 324-entry `mul!` far under its 20 s bound (1.84 s), but the first CpuPolyester `assemble` of `Δₕ³` in 3D, already past 20 s with them (30.2 s), grows to 37.4 s without, so the rule fails; a narrower out-of-line threshold, set by its own constant, is the follow-up to measure.
