# What a scalar coefficient costs the first call, by spelling (v4.1.4, S3)

Answers #912 step 1: how many method instances of Bramble the first `form` and `assemble`
infer, and how long they take, when the coefficient of a bilinear form is spelled
`2.0`, `2.0f0`, `π` or `1 // 2`. This note records figures. It makes no decision on
normalising the coefficient type, which is the user's, in the issue.

## Method and conditions

`benchmark/snoop_instances.jl` starts a fresh `julia` per launch. The child loads Bramble
and SnoopCompile, builds a graded 2D mesh and its `gridspace`, then runs `form` of
`c * inner₊(∇ₕ(u), ∇ₕ(v))` and `assemble` inside `@snoop_inference`. The count is
the number of distinct `MethodInstance`s of Bramble or its submodules in the flattened
inference tree, root excluded, summed over `form` and `assemble`. Two further runs per
row: a sequence child that runs the four spellings in the order above in one process, and
a repeat of the row's last call, which must infer 0 instances (it did, for every row).
Times are medians over 3 launches.

Command, one `julia` with 1 thread per child, serial:

```
julia --startup-file=no benchmark/snoop_instances.jl \
    --rows scale_float64,scale_float32,scale_pi,scale_rational --launches 3 \
    --out benchmark/results/scalar_spellings.toml
```

| quantity | value |
|---|---|
| commit | `e34e11c2` on `perf/v4.1.4-measure` (adds only the benchmark scripts) |
| Julia | 1.13.2, default `-O` |
| machine | Apple M2, 16 GiB, on AC power, battery 100% |
| threads | 1 per child |
| load | 1-minute load 4.18 at the start (21:11) and 4.10 at the end (21:12) |

## Run under load

`gate_preflight.sh --timing` FAILED at 21:11. Load 4.37 on 8 cores (limit 2), other
julia processes were running (the concurrent v4.1.3-perf plan), and one `julia` was at
99.3% CPU. Those sessions were not paused or stopped. The run went ahead anyway, so every
time below is "under load", with a 1-minute load of 4.1 to 4.4 during the run
(`uptime` 4.37, 4.43, 4.13 before; 4.10, 4.34, 4.11 after). The instance counts do not
depend on load. The times are not measurements to compare across runs. They are one
under-load figure each, enough for an order of magnitude and for the ratio of the
Float64 row to the others.

## Figures

| spelling | row | fresh | added | form, s | assemble, s |
|---|---|---:|---:|---:|---:|
| `2.0` | `scale_float64` | 6 | 6 | 0.01517 | 0.01411 |
| `2.0f0` | `scale_float32` | 89 | 88 | 0.01605 | 0.2637 |
| `π` | `scale_pi` | 90 | 89 | 0.01769 | 0.2549 |
| `1 // 2` | `scale_rational` | 89 | 88 | 0.01587 | 0.2599 |

`fresh`: instances the row infers as the only spelling in its process. `added`: what it
infers in the sequence child after the rows above it. `form` and `assemble` are the wall
times of the first call of each in the fresh child, medians over 3 launches. Inference
time, kept as `fresh_inference_s` in the TOML, is 0.028, 0.296, 0.293 and 0.293 s in the
same order, so inference is nearly all of the `assemble` time.

## What the 2nd to 4th spellings add

The sequence child inferred 6 instances for `2.0`. Up to and including `2.0f0`, `π` and
`1 // 2` it had inferred 94, 183 and 271 distinct instances. The three other spellings
therefore add 88, 89 and 88, 265 in total, on top of the 6 of the first.
Each new spelling adds about as many instances as it would infer alone (89, 90, 89 fresh
against 88, 89, 88 added): none of the three shares its inference with another. The
sequence child did not time each call, so the cost in seconds of the 2nd to 4th spellings
in one process is not recorded here. The fresh-child times above are the nearest figure:
about 0.26 s of `assemble` each, nearly all of it inference.

## Why the Float64 spelling is cheap

The row for `2.0` infers 6 instances and about 0.03 s because the package image already
covers the Float64 path of `form` and `assemble`, so the call finds most of its method
instances compiled. The Float64 row therefore infers little and says nothing about what
the generic path costs. The other three spellings reach the same code with a different
coefficient type, find no image entry and infer about 88 instances each. A user who
writes `2.0` pays about 0.03 s, and one who writes `2.0f0`, `π` or `1 // 2` pays about
0.3 s once per spelling per session.

## To retake

The times are to be retaken on main after v4.1.3-perf merges, on a machine that passes
`gate_preflight.sh --timing`: #895 and #906 change the first-call cost of `form` and
`assemble`, and this run was under load. The instance counts should be retaken with them,
since those changes may alter them too.

## What #912's step 2 decision needs

- Normalising the coefficient to Float64 would make the three 88-instance spellings share
  the 6-instance path of `2.0`. The most it can save in a session using all four
  spellings is the 265 added instances, about 0.8 s of first-call time under this load
  (3 x 0.26 s of `assemble` plus the small `form` differences).
- A session using one non-Float64 spelling would save about 0.3 s (88 to 90 instances).
- The Float64 row sets the floor: 6 instances, 0.03 s.
- Not measured here: the cost of the conversion itself, and the result for a coefficient
  that does not round-trip through Float64.

## After normalising

The figures above stay as they were: one run under load, on `e34e11c2`, before #912 was
changed. This section retakes the four spellings on two trees in one run, with #912 step 2
(S2) on one of them.

### What S2 changes

S2 normalises the scalar coefficient to `Float64` only where that is bit-neutral: every
leaf space is `Float64`, no product meets two constant scales, all data is `Float64`, and
there is no `SourceFunction` or `DiracSource`. Float32 spaces, `εₕ`'s nested `1//2`,
two-sided scales and linear forms with a function source are kept as built. The four
rows here are plain `Float64`-space bilinear forms, so all four are normalised.

### Method and conditions

Two trees: `base`, a detached worktree at `995412a2` (`git merge-base HEAD origin/main`),
and `head`, `perf/v4.1.4-scalars-div` at `00e9d784` (S1, then S2 with its format commit).
Both ran from the head worktree's scripts with `--tree`. The base tree got head's
`Manifest.toml`, was instantiated and precompiled (test project, which covers the root
workspace) before any timing, and no other process ran during the timing. 4 threads
throughout, each `julia` under `julia_slot.sh 4`, default optimisation level, the whole
run under `caffeinate -dims`.

```
julia --startup-file=no benchmark/snoop_instances.jl --tree T \
    --rows scale_float64,scale_float32,scale_pi,scale_rational --launches 3 --threads 4 \
    --out benchmark/results/v4.1.4_scalar_spellings_T.toml
julia --startup-file=no benchmark/multiterm_ttfx.jl --tree T \
    --rows scale_float64,scale_float32,scale_pi,scale_rational --launches 5 --threads 4 \
    --section spellings_T --out benchmark/results/v4.1.4_multiterm.toml
```

for T in base, head. Both scripts write medians only, so the spread comes from a third
step: `multiterm_ttfx.jl --child ROW` run directly, 5 rounds, in each round the four rows
in turn and each row base then head, so the load drifts through both trees together. Every
"spread" below is min to max over those 5 children. Its medians are close to the
multiterm TOML medians, which come from a run of their own.

| quantity | value |
|---|---|
| Julia | 1.13.2, default `-O` |
| machine | Apple M2, 16 GiB, on AC power (battery 100%, charging) |
| preflight | `gate_preflight.sh --timing` OK at 20:52 (load 1.63; no other julia, nothing above 50% CPU); OneDrive was running, not paused (its ASK line) |
| load, 1-minute | 1.95 to 2.90 over the run (20:52 to 20:56); 2.19 to 2.69 over the control below |

These times are measurements, not "under load" figures: the preflight passed. The
benchmark TOMLs are `benchmark/results/v4.1.4_scalar_spellings_base.toml` and
`..._head.toml`, and the sections `spellings_base` and `spellings_head` of
`benchmark/results/v4.1.4_multiterm.toml`. Sections `base`, `div` and `head` of that file
are S3.1's and were not touched.

### First call: instances

| row | fresh base | fresh head | added base | added head |
|---|---:|---:|---:|---:|
| `scale_float64` | 6 | 6 | 6 | 6 |
| `scale_float32` | 91 | 29 | 90 | 24 |
| `scale_pi` | 92 | 30 | 91 | 20 |
| `scale_rational` | 91 | 29 | 90 | 19 |

`fresh` is the instance count of the row alone in its process. `added` is what it infers
after the rows above it in one process, in the order listed. The sequence child inferred 277
distinct instances on base (6 + 90 + 91 + 90) and 69 on head (6 + 24 + 20 + 19). The
repeat of the last call inferred 0 on both trees, for every row. The three
non-`Float64` rows do not reach the 6 of `2.0`: each still infers 19 to 30 instances. The
likely source is the normalisation itself (`_normalise_scalars`, run by `form`), though
this run does not itemise them. The rows share only part of it with each other (24, then
20, then 19 added).

### First call: times

Snoop figures are medians over 3 launches. The spread step gives the first-call time of
`form` and `assemble` as median (min to max) over 5 children, in ms.

| row | form base | form head | assemble base | assemble head |
|---|---:|---:|---:|---:|
| `scale_float64` | 6.87 (6.52 to 7.33) | 6.69 (6.55 to 7.24) | 10.77 (10.29 to 13.29) | 10.90 (10.75 to 12.04) |
| `scale_float32` | 8.51 (8.34 to 8.81) | 17.84 (17.45 to 18.81) | 214.8 (209.4 to 229.2) | 10.44 (10.42 to 10.62) |
| `scale_pi` | 10.71 (10.43 to 12.27) | 22.19 (20.29 to 23.07) | 212.7 (207.8 to 214.7) | 10.46 (10.23 to 12.77) |
| `scale_rational` | 9.06 (8.76 to 9.92) | 17.62 (16.72 to 18.32) | 217.9 (217.2 to 230.6) | 10.44 (9.90 to 10.75) |

The `fresh_form_s` and `fresh_assemble_s` of the snoop TOMLs, in ms, agree: base form
11.7, 14.9, 15.3, 14.1 and assemble 10.6, 240.9, 231.7, 235.7; head form 11.9, 23.3, 26.3,
23.4 and assemble 10.9, 11.3, 11.2, 10.6 (rows in the table's order). The snoop child's `form` times differ
from the multiterm child's because it runs `form` inside `@snoop_inference`.

So on the three non-`Float64` spellings the first `assemble` falls from about 0.21 s to
the 0.01 s of `Float64` (the spreads do not overlap), and the first `form` rises by about
9 to 11 ms, the conversion paying its own inference at `form` time. The first `form` plus
`assemble` of one such spelling goes from about 223 to 228 ms down to about 28 to 33 ms.
The `Float64` row did not move.

### Warm

Warm times are the min of 5 calls inside the child, so each figure is a min taken
once per child. Median (min to max) over 5 children, in µs. The clock step is 0.042 µs.

| row | warm `assemble` base | warm `assemble` head | warm `assemble!` base | warm `assemble!` head |
|---|---:|---:|---:|---:|
| `scale_float64` | 1.042 (1.041 to 1.291) | 1.708 (1.208 to 1.750) | 0.792 (0.791 to 0.792) | 0.917 (0.875 to 0.959) |
| `scale_float32` | 1.083 (1.000 to 1.209) | 1.209 (1.167 to 1.250) | 0.667 (0.666 to 0.709) | 0.916 (0.916 to 0.958) |
| `scale_pi` | 1.083 (1.041 to 1.167) | 1.208 (1.166 to 1.292) | 0.708 (0.708 to 0.750) | 0.958 (0.916 to 0.958) |
| `scale_rational` | 1.083 (1.000 to 1.166) | 1.292 (1.208 to 1.292) | 0.667 (0.666 to 0.750) | 0.916 (0.916 to 0.958) |

Warm `form` is 0.04 to 0.08 µs in all eight cells, which is the clock step, and does not
separate the trees.

A control, same conditions, same interleaving (5 rounds, base and head alternating):
`scalar_2d`, a form with no coefficient that S2 does not touch.

| row | warm `assemble` base | warm `assemble` head | warm `assemble!` base | warm `assemble!` head |
|---|---:|---:|---:|---:|
| `scalar_2d` | 1.167 (1.000 to 1.208) | 0.792 (0.750 to 0.959) | 0.750 (0.708 to 0.792) | 0.500 (0.500 to 0.709) |

### Verdict on warm

Warm `assemble!` did move outside the spread, on all four spellings: head is 0.12 to 0.29
µs higher (+16% for `scale_float64`, +34 to +37% for the other three), and the min to max
ranges do not overlap. Warm `assemble` moved by +0.13 to +0.67 µs in the median; its ranges overlap for
`scale_float32` and `scale_pi`, and separate for `scale_rational` (1.208 against at most
1.166) and, in the median, for `scale_float64`. By the criterion as written, that the warm
times stay inside the spread, it is not met.

Two things bound what that means. First, the control `scalar_2d`, which S2 cannot affect,
moved outside its spread too, and the other way (warm `assemble!` 0.750 to 0.500 µs,
-33%), so these sub-microsecond rows shift by tens of percent between two trees for
reasons of code placement, in either direction. Second, on head the four spellings now
read the same (`assemble!` 0.92, 0.92, 0.96, 0.92 µs): all four reach the one `Float64`
path. On base `Float64` runs the precompiled instance (0.79 µs) and the other three a
freshly inferred one that is faster (0.67 to 0.71 µs), so the three non-`Float64` rows lose the
0.1 to 0.2 µs they had on base and `scale_float64` gains 0.13 µs. An absolute
gain or loss of 0.1 to 0.3 µs on a 9x11 mesh is not a measurement of warm cost per
degree of freedom: the warm times here say that no spelling got slower than the others
by more than a quarter of a microsecond, and nothing more. A warm A/B on a mesh large
enough to take tens of µs would settle it.

### JET

On `bool_audit`'s elasticity row (`μ` and `λ` captured in the closure, a form that is
already not concrete on base), `jet_form_reports` read 38 on base, 32 on the `div` tree
(S1) and 39 on head, from S3.1's files `v4.1.4_div_bool_audit_{base,div,head}.toml`.
The 7 reports head has over `div` (39 against 32) are runtime dispatches
inside `_normalise_scalars`, `_normalised` and `_normalised_in`. None is in the gate. With
literal coefficients `report_opt(form)` is 0 on head. `jet_assemble_reports` is 0 on all
three.
