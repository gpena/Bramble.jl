# What the divₘₕ unroll does to the elasticity first call (v4.1.4, S3.1)

Answers the measuring half of #921: S1 rewrote `divₘₕ` so that the elasticity form infers
concretely. This note records the figures before and after, on three trees run together.
It records figures and says which differences lie outside the launch spread. It does not
decide whether to keep the change when the first-call figures do not move: that is the
user's, in the issue.

## Method and conditions

Three trees, one session:

- `base`: a detached worktree at `995412a2`, `git merge-base HEAD origin/main`.
- `div`: a detached worktree at `995412a2` with `e824a1d3` (S1, `divₘₕ` generated unroll)
  cherry-picked with `--no-commit`. Its `commit` field in the trace TOML reads `995412a2`
  because the change was never committed there. It is base plus S1, and nothing else.
- `head`: `perf/v4.1.4-scalars-div` at `ab551efe`: S1, then the S3.0 script change, then
  S2 (#912, `Float64` normalisation). Shown for the record.

The scripts ran from the head worktree, each with `--tree`, except `bool_audit.jl`, which
has no `--tree`: it ran under `--project=<tree>/test`. Its `commit` field is the script's
own tree, so it reads `ab551efe` in all three files and says nothing about the tree
audited. The base and div trees got the head `Manifest.toml` and were instantiated and
precompiled (root and test projects) before any timing started.

Commands, one at a time, 4 threads, each under `julia_slot.sh 4` with
`BRAMBLE_JULIA_SLOTS=4`, default optimisation level, for T in base, div, head:

```
julia --startup-file=no benchmark/trace_compile.jl --tree T --rows elasticity,vector_laplacian \
    --launches 3 --threads 4 --out benchmark/results/v4.1.4_div_trace_compile_T.toml
julia --startup-file=no --threads=4 --project=T/test benchmark/bool_audit.jl --rows elasticity \
    --out benchmark/results/v4.1.4_div_bool_audit_T.toml
julia --startup-file=no benchmark/multiterm_ttfx.jl --tree T --launches 5 --threads 4 \
    --rows elasticity,vector_laplacian,scalar_2d,scale_float64,scale_float32,scale_pi,scale_rational \
    --section T --out benchmark/results/v4.1.4_multiterm.toml
```

That is run 1. `trace_compile.jl` and `multiterm_ttfx.jl` write medians only, so the
spread, the min to max over launches, is not in the TOMLs. A second run (run 2) took it:
`multiterm_ttfx.jl --child ROW` run directly, 5 rounds, each round the rows elasticity,
vector_laplacian and scalar_2d in turn, each row base, div, head in turn, so the load
drifts through the trees together. The medians of run 2 sit next to its spread below. For
the trace, the spread is that of the per-launch compile time of the root calls (`form`,
`_bilinear_form`, `assemble`), read from the three raw traces of run 1.

A first attempt at run 2 was thrown away. The machine slept during it: one launch
printed a first `assemble` of 591 s, and the load log has two gaps, of 10 and 17 minutes. Run 2 was
repeated under `caffeinate -dimsu`. Run 1 has no gap in its load log.

| quantity | value |
|---|---|
| Julia | 1.13.2, default `-O` |
| machine | Apple M2, on AC power; OneDrive running, not paused (the preflight's ASK line) |
| threads | 4 per child, one child at a time |
| preflight | `gate_preflight.sh --timing` failed on load for about 12 minutes (2.4 to 2.7, from desktop apps, no julia, nothing above 50% CPU), then passed at 18:50 (load 2.00); passed again at 19:44 for run 2 (1.98) |
| load, run 1 | 1-minute load 2.29 to 4.64, 18:50 to 19:06 (it includes the run's own 4 threads) |
| load, run 2 | 1-minute load 1.98 to 6.74, 19:44 to 19:52 |

The preflight passed before both runs, so the times are not labelled "under load".

## Figures

All times are for the elasticity row (5^3 graded mesh, `gridspace(Ωₕ)^Val(3)`,
`2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₘₕ(u), divₘₕ(v))`) unless said. `form` and
`assemble` are the first call in a fresh process. "warm" is the min of 5 repeats inside
the child. Spread is min to max over the 5 launches of run 2.

First call, run 2, seconds, median (spread):

| | first `form` | first `assemble` |
|---|---|---|
| base | 4.669 (4.637 to 4.972) | 16.55 (16.21 to 16.72) |
| div | 4.741 (4.643 to 4.794) | 16.34 (16.30 to 17.67) |
| head | 4.764 (4.680 to 5.894) | 16.44 (16.29 to 18.33) |

The same figures in run 1 (medians of 5, no spread kept): `form` 5.026 base, 4.928 div,
5.153 head; `assemble` 17.83 base, 16.79 div, 18.03 head. The direction of the `form`
change flips between the two runs (div is 0.10 s below base in run 1 and 0.07 s above in
run 2), and `assemble` moves by 1.0 s in run 1 and 0.2 s in run 2.

Warm, run 2, microseconds, median (spread):

| | warm `form` | warm `assemble!` | warm `assemble` |
|---|---|---|---|
| base | 20.79 (19.29 to 23.37) | 167.8 (167.3 to 175.2) | 182.0 (180.0 to 190.0) |
| div | 13.67 (12.08 to 14.21) | 168.5 (167.7 to 175.0) | 182.0 (180.9 to 221.5) |
| head | 13.04 (12.21 to 14.67) | 168.3 (167.5 to 173.8) | 181.4 (180.5 to 202.1) |

Run 1 gives warm `form` 20.21, 12.54 and 13.58 µs, and warm `assemble!` 168.96, 172.96
and 172.63 µs (base, div, head).

Inference and trace, run 1, one process each (the trace is 3 launches of `trace_compile.jl`,
compile times in the trace as printed):

| | base | div | head |
|---|---|---|---|
| `jet_form_reports` | 38 | 32 | 39 |
| `jet_assemble_reports` | 0 | 0 | 0 |
| summands of `form.ast` | 35 | 35 | 35 |
| non-`Bool` predicate sites | 26 | 19 | 19 |
| trace lines, elasticity | 292 | 344 | 347 |
| functions in the trace, elasticity | 61 | 52 | not counted |
| trace sum, elasticity (s) | 22.54 | 23.37 | 23.63 |
| form wall in the trace run (s) | 5.515 | 6.039 | 6.125 |
| assemble wall in the trace run (s) | 16.735 | 16.768 | 16.927 |

The seven sites that the unroll removes are `_has_test_interp` (1 instance),
`_has_trial_interp` (2), `_is_source_only` (2) and `_kron_is_fresh` (2). The 19 that
remain, in div and in head alike, are `_ghost_kron_reads_absent` 1, `_marks_only_face` 1,
`_mf_gathers` 2, `_near_edge` 1, `_pairable_type` 3, `_peelable` 1, `_same_mesh` 2,
`_shares_one_mesh` 1, `_storage_collapsed` 1 and `_type_holds` 6.

The trace roots of the `form` phase, per launch in milliseconds (3 launches):

- base: `form` 1332.0, 1398.1, 1328.9, and `_bilinear_form` 3680.1, 3674.3, 3678.1, so
  5007 to 5072 together.
- div: `form` 5616.1, 5536.2, 5784.5. There is no `_bilinear_form` root any more: its work
  is under `form`.
- head: `form` 5954.0, 5666.9, 5765.5.

The `assemble` root: base 16719.0, 16846.7, 16669.5; div 16749.2, 16654.2, 18735.0; head
17148.3, 16910.3, 16517.9. In vector_laplacian the `form` root is 75.9 to 78.4 (base),
75.3 to 105.7 (div), 79.6 to 83.2 (head) and the `assemble` root 2788 to 2817, 2782 to
2995, 2805 to 2884.

Other rows of run 2 (vector_laplacian, then scalar_2d), base against div, each as median
(min to max).

- First `form`: vector_laplacian 0.07135 s (0.07063 to 0.07756) against 0.07372 (0.06981 to
  0.07857); scalar_2d 6.017 ms (5.938 to 7.787) against 6.110 (5.995 to 7.089).
- First `assemble`: vector_laplacian 2.730 s (2.712 to 2.964) against 2.753 (2.706 to
  2.953); scalar_2d 3.443 ms (3.303 to 4.248) against 3.453 (3.288 to 5.065).
- Warm `assemble!`: vector_laplacian 11.17 µs (10.96 to 11.38) against 11.12 (10.88 to
  11.25); scalar_2d 0.750 µs (0.750 to 0.792) against 0.625 (0.625 to 0.667).

The scale rows, run 1, first `assemble` in seconds, base, div, head: `scale_float64`
0.01029, 0.01157, 0.01156; `scale_float32` 0.2123, 0.2293, 0.01054; `scale_pi` 0.212,
0.2159, 0.0137; `scale_rational` 0.2196, 0.2236, 0.01069. That is S2's effect (head), and
S1 does not touch it.

## Verdict

Base against div is #921's before and after.

- First `form`: inside the spread. Run 2: 4.669 s (4.637 to 4.972) against 4.741 s
  (4.643 to 4.794), the ranges overlap. Run 1 has the opposite sign. No change is
  measured in either direction.
- First `assemble`: inside the spread. Run 2: 16.55 s (16.21 to 16.72) against 16.34 s
  (16.30 to 17.67), overlapping.
- Warm `assemble!`: inside the spread, 167.8 µs (167.3 to 175.2) against 168.5 µs (167.7 to
  175.0). Warm `assemble` also.
- Warm `form`: outside the spread, and faster. 20.79 µs (19.29 to 23.37) against 13.67 µs
  (12.08 to 14.21); the ranges do not overlap, and run 1 agrees (20.21 against 12.54).
  It is 34% less in run 2 and 38% in run 1, about 7 µs per call. Warm `form` of the other two rows is
  inside the spread.
- JET and the sites do move: `jet_form_reports` 38 to 32, non-`Bool` sites 26 to 19,
  trace functions 61 to 52. These counts do not depend on load.

So the first-call criterion of #921 did not hold: neither the first `form` nor the first
`assemble` changed by more than the spread. The inference counts fell and the first-call
cost stayed where it was. What the trace shows is the compile cost moving, not
shrinking: base spends 3.68 s in `_bilinear_form` and 1.33 s in `form`; div has the whole
5.5 to 5.8 s under `form`, and 52 functions instead of 61 with 344 trace lines instead of
292 (115 instances of `OperatorAdd` against 69, 31 of `Base.getproperty` against 5). In the
trace run itself the `form` phase comes out 0.5 to 0.7 s longer for div than for base,
outside the 3-launch spread of those roots, while the untraced first `form` shows no
difference. The trace adds a cost per compiled instance, so div's extra instances weigh
more there. The untraced figures are the first-call figures.

## Head against div

Head differs from div, in the counts. `jet_form_reports` is 39 against 32 (the TOML keeps
the count, not the reports, so the seven extra are not named here); the non-`Bool` sites
are the same 19; the elasticity trace has 347 lines against 344, with `Base.MappingRF`
(2 instances) and one more `Base.mapfoldl_impl` than div; and the vector_laplacian trace
gains `Bramble._normalise_scalars` (1 instance). The first-call and warm times of head
lie inside the spread of div in run 2 (first `form` 4.764 s against 4.741, first
`assemble` 16.44 against 16.34, warm `form` 13.04 µs against 13.67). Head's slowest first
`form` launch took 5.894 s, a single launch.

## Notes on the run

- The multiterm TOML holds the medians of run 1 only. The spread figures live in this note.
- Timer resolution is about 0.04 µs, so the 0.125 µs difference of scalar_2d warm
  `assemble!` is three timer ticks.
