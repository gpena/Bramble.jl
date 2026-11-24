# The out-of-line gate on long stencils, warm and first call (four threads, 1a97bebf)

Answers the last open line of #853: a `@noinline` gate on stencils past 32 entries was said to cut
the first `assemble` of `Δₕ³` in 3D from about 18 s to about 8 s at a warm cost of 2 to 5%, measured
before the concatenation fix. This note measures it again on today's code.

## What the gate is

The issue does not say which functions it marked. The code at 1a97bebf already has an out-of-line
path past `_INLINED_ENTRIES` (32) in `src/ast/common.jl`, three `@noinline` helpers:
`_each_entry_outlined` (#759, 2026-10-14), `_each_pair_outlined` (#759) and
`_concatenate_outlined` (the concatenation fix, 2026-10-23). The gate was therefore measured on
top of one of them, and the only gate this tree can show is the one in the file.

- **With the gate**: the tree as it is.
- **Without**: the same tree with the three `@noinline` replaced by `@inline`, so the unrolled
  tuple is built inside every caller. The dispatch on length is unchanged.

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

The control sits on the inlined side of the gate (32 is not past 32), so it must not move; the
entry counts come from `local_stencil` at an interior point.

Rerun with `TREE` one of the two copies and `CASE` one of `ctl32`, `d2_3d`,
`d3_2d`, `d3_3d` (the script below is `/tmp/s312/measure.jl`, the copies are
`tar --exclude=.git -cf - . | tar -xf - -C /tmp/s312/<tree>` plus the `sed` that swaps the three
annotations):

```bash
cd /tmp/s312/$TREE && BRAMBLE_JULIA_SLOTS=4 .claude/scripts/julia_slot.sh 4 \
    julia --threads=4 --startup-file=no --project=. /tmp/s312/measure.jl $CASE
```

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
smallest of the three runs; ratio is gate over no gate on the median of the three.

| case | with the gate, 3 runs | spread | without, 3 runs | spread | ratio |
|---|---|---|---|---|---|
| `ctl32` | 0.0104, 0.0107, 0.0107 | 0.0003 | 0.0106, 0.0107, 0.0107 | 0.0001 | 1.00 |
| `Δₕ² 3D` | 0.0620, 0.0635, 0.0638 | 0.0018 | 0.0360, 0.0361, 0.0362 | 0.0002 | 1.76 |
| `Δₕ³ 2D` | 0.0937, 0.0954, 0.0959 | 0.0022 | 0.0390, 0.0399, 0.0406 | 0.0016 | 2.39 |
| `Δₕ³ 3D` | 1.5972, 1.5987, 1.6354 | 0.0382 | 0.5183, 0.5212, 0.5213 | 0.0030 | 3.07 |

First `assemble`, in seconds, fresh process.

| case | with the gate, 3 runs | spread | without, 3 runs | spread | ratio |
|---|---|---|---|---|---|
| `ctl32` | 0.339, 0.349, 0.352 | 0.013 | 0.347, 0.349, 0.350 | 0.003 | 1.00 |
| `Δₕ² 3D` | 1.344, 1.346, 1.353 | 0.009 | 1.267, 1.271, 1.280 | 0.013 | 1.06 |
| `Δₕ³ 2D` | 2.672, 2.696, 2.696 | 0.024 | 2.284, 2.293, 2.321 | 0.037 | 1.18 |
| `Δₕ³ 3D` | 18.091, 18.221, 18.432 | 0.341 | 19.405, 19.474, 19.506 | 0.101 | 0.94 |

The control is identical in both trees, so the harness sees no difference where none exists. The
gate does not reproduce the issue's first-call figure either: on `Δₕ³` in 3D it saves 1.3 s
of 19.5 (6%), against the 10 s of the issue, and on the two smaller long stencils the first call is
slower with the gate (6% and 18%).

## One more check, outside the plan: the matrix-free product

#759 added the out-of-line path because a long product's first matrix-free `mul!` took minutes,
so the same two trees ran one fresh process each (four threads, one run, no repeat) on
`matrix_free_operator(a)` over a 9 × 8 graded 2D mesh. 0 B per warm `mul!` in both. Not a
three-run figure: read it as a check that dropping the gate does not blow the compile up, not as
a ratio.

| form | first `mul!`, with / without (s) | warm `mul!`, with / without (ms) |
|---|---|---|
| `innerₕ(Δₕ(Δₕ(Δₕ(u))), v)` | 3.11 / 2.13 | 0.146 / 0.069 |
| `innerₕ(Δₕ(Δₕ(u)), Δₕ(Δₕ(v)))` | 7.40 / 7.73 | 0.334 / 0.265 |

The 616 s of #759 was the unused gather's rows compiled through a `Bool` flag (`_mf_gathers`),
which the `Val` dispatch removed; neither tree shows it. The case #759 quotes
(`innerₕ(D̽ᵧD̽ₓD̽ₓu, D̽ᵧD̽ₓD̽ₓv)`, 729 entries) was not run in either tree.

## Reading

Warm `assemble!` is 1.8 to 3.1 times slower with the gate on the three long stencils, far outside
the 0.6 to 4% spread of the runs without it, and the cost grows with the stencil. Likely cause, not
profiled: the call out of line returns the whole unrolled tuple (729 entries of an offset triple
and a weight) through memory at every grid point, where the inlined build can keep it in registers. The first-call saving the
gate was added for is 6% on one case and negative on two.

Dropping the three `@noinline` is a source change this subplan does not make: its rule leaves `src`
untouched unless the gate is the faster tree, and the matrix-free case #759 quotes is unmeasured.
That follow-up needs the 729-entry matrix-free product and the CpuPolyester `assemble!` run
both ways, three alternating runs each.

Verdict: the gate fails the rule (warm `assemble!` 1.8 to 3.1 times slower than without it, spread under 4%), `src/ast/common.jl` is left untouched, and dropping the gate waits on a follow-up that measures the 729-entry matrix-free product.
