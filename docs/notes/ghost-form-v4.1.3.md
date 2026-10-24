# Ghost form slowdown since v4.0.0 (v4.1.3, S3.1, #907)

The ghost `form` slowdown is commit `8d81c11d` ("feat: polygon_spaces(Ωₕ, :L, g) assembles the
Ferreira-Grigorieff scheme (#202, #204)", in merge #832, v4.1.0), function
`_ghost_reduction_map` in `src/assembly/ghost_assembly.jl`. See `## Cause`.

## Method

Julia 1.13.2, Apple Silicon (8 logical cores), 4 threads, default `-O`, AC power. One fresh
process per run (`julia_slot.sh 4`, `BRAMBLE_JULIA_SLOTS=4`), the v4.0.0 baseline
`benchmark/baselines/baseline_1784e815.json` supplying `evals` and `samples` of each row (as
`benchmarks.jl --compare` does). Rows measured: serial 2D, 1D and 3D `ghost form`, and serial
3D `plain assemble` as the canary (the `plain form` row, a 17 ns lazy constructor, is too
short to read and is left out of the tables). Interleaved, 3 rounds per step, the order
alternating (before side first in rounds 1 and 3, after side first in round 2).

Each commit is built in a detached worktree `~/worktrees/Bramble/bisect-907-<sha>` with the
branch worktree's `Manifest.toml` copied in, so the dependencies are identical on every side
(the v4.0.0 and main Manifests differ in dependency versions). The v4.0.0 side is
`bisect-907-0aa87e45`, not the older `~/worktrees/Bramble/v4.0.0`, for the same reason.

The harness is `~/bramble-scratch/v413perf/s31/` (`run.jl` is the measured script, `go.sh` one
run, `pair.sh` one bisect step, `summ.py` the ratios, `runs/` the raw output). It runs only the
rows above, on the main checkout's `ghost_vs_plain.jl` (identical to v4.0.0's). It holds a
500 MB ballast array so that the GC heap target is the suite's: without it the 2D `ghost form`
median is bimodal on GC (56 ms against 27 ms) and only the minimum is stable. With the ballast
the v4.0.0 medians are 27.4 ms (2D), 16.9 ms (1D) and 17.4 ms (3D), the figures of a full
`_ghost_vs_plain_group()` setup in the same process (27.4, 17.5, 17.4 ms) and of the issue
(27.5, 17.5, 17.7 ms). The first attempt without ballast (`runs_noballast/`) gave the same ratios
of minima (1.06 to 1.07 against `10739d67`).

Bisect space: only merges whose first-parent diff touches `src/`, `ext/` or `Project.toml` can
change the timing: #826 (`2da88d6d`), #832 (`3b0ee390`), #835 (`35c4c66e`), #850 (`10739d67`),
#856 (`bb293acf`), #858 (`7dbaa523`), #910 (`49f5fb39`), #911 (`9c34251c`). The other 11 of the
19 first-parent merges leave `src/` as their parent had it, so a source state is tested at the
first merge that has it. Inside #832, 17 of its 23 commits touch `src/`; the bisect went to
the commits that matter.

Conditions: not alone on a quiet machine (Spotlight `mds` and other daemons; no other Julia).
`gate_preflight.sh --timing` failed on load throughout (1-minute load 2.0 to 4.1), while
`check_power_load.sh` (threshold 4.0) passed, so every figure is "under load" and the
conclusion rests on the per-round ratios against the canary. The 1-minute load of each run is
listed under each table. Ratios are after / before; "ratio of minima" is the minimum over the
3 rounds of each side.

## Steps

Steps 1 and 2 show the slowdown is already in `10739d67` and in `3b0ee390`; step 3 (run twice)
shows it is not in #826; steps 4 to 6 narrow #832 to one commit; steps 7 and 8 check main against v4.0.0 and against v4.1.1. A first run of step 3 had a
disturbed round 3 (load 4.13, every row including the canary at 1.4 to 1.5; rounds 1 and 2 at
0.95 to 1.05) and was repeated in full; the table is the repeat.

### Step 1: `0aa87e45` against 10739d67 (merge #850, v4.1.1)

| row | median before, ms | median after, ms | ratio per round (median) | ratio per round (min) | ratio of minima |
|---|---|---|---|---|---|
| serial 2D, ghost form | 26.924 | 29.684 | 1.081 1.158 1.067 | 1.071 1.088 1.072 | 1.074 |
| serial 1D, ghost form | 16.749 | 18.308 | 1.077 1.158 1.076 | 1.076 1.082 1.079 | 1.078 |
| serial 3D, ghost form | 17.313 | 18.513 | 1.075 1.043 1.066 | 1.052 1.045 1.055 | 1.048 |
| canary: serial 3D, plain assemble | 0.513 | 0.523 | 0.975 1.041 1.093 | 1.014 1.017 1.002 | 1.014 |

1-minute load over the runs of each round, round 1 1.55 to 2.38, round 2 2.38 to 2.84, round 3 2.54 to 2.87.

### Step 2: `0aa87e45` against 3b0ee390 (merge #832, v4.1.0 polygons)

| row | median before, ms | median after, ms | ratio per round (median) | ratio per round (min) | ratio of minima |
|---|---|---|---|---|---|
| serial 2D, ghost form | 27.285 | 30.674 | 1.147 1.139 1.076 | 1.076 1.102 1.072 | 1.076 |
| serial 1D, ghost form | 17.092 | 18.860 | 1.135 1.101 1.083 | 1.106 1.076 1.083 | 1.089 |
| serial 3D, ghost form | 17.463 | 18.516 | 1.108 1.037 1.060 | 1.080 1.039 1.066 | 1.056 |
| canary: serial 3D, plain assemble | 0.550 | 0.520 | 1.070 0.945 0.820 | 1.049 1.002 0.865 | 0.996 |

1-minute load over the runs of each round, round 1 2.88 to 3.49, round 2 2.77 to 3.03, round 3 2.63 to 2.77.

### Step 3: `0aa87e45` against 2da88d6d (merge #826, v4.0.1 follow-ups)

| row | median before, ms | median after, ms | ratio per round (median) | ratio per round (min) | ratio of minima |
|---|---|---|---|---|---|
| serial 2D, ghost form | 27.792 | 27.188 | 0.965 1.003 0.976 | 0.997 1.004 1.003 | 1.003 |
| serial 1D, ghost form | 16.835 | 17.312 | 0.973 0.999 1.030 | 1.005 1.003 1.004 | 1.003 |
| serial 3D, ghost form | 17.339 | 17.227 | 0.979 0.994 0.999 | 0.986 0.988 1.001 | 1.001 |
| canary: serial 3D, plain assemble | 0.504 | 0.501 | 0.991 0.995 0.992 | 0.983 0.989 0.986 | 0.988 |

1-minute load over the runs of each round, round 1 3.39 to 4.05, round 2 3.07 to 4.05, round 3 2.58 to 3.07.

### Step 4: `0aa87e45` against 7811c79b (#832, last mesh-only commit before the polygon spaces)

| row | median before, ms | median after, ms | ratio per round (median) | ratio per round (min) | ratio of minima |
|---|---|---|---|---|---|
| serial 2D, ghost form | 27.631 | 26.979 | 1.003 0.963 0.976 | 1.004 0.989 0.994 | 1.000 |
| serial 1D, ghost form | 17.884 | 17.020 | 1.008 0.914 0.952 | 1.004 0.994 1.001 | 1.004 |
| serial 3D, ghost form | 17.374 | 17.228 | 0.989 0.945 1.018 | 0.996 0.985 1.002 | 1.001 |
| canary: serial 3D, plain assemble | 0.508 | 0.503 | 0.989 0.931 0.995 | 1.007 0.959 1.003 | 1.002 |

1-minute load over the runs of each round, round 1 2.16 to 2.58, round 2 2.16 to 3.67, round 3 2.64 to 3.67.

### Step 5: `7811c79b` against 8d81c11d (#832, `polygon_spaces` assembles the Ferreira-Grigorieff scheme)

| row | median before, ms | median after, ms | ratio per round (median) | ratio per round (min) | ratio of minima |
|---|---|---|---|---|---|
| serial 2D, ghost form | 27.118 | 29.061 | 1.070 1.075 1.075 | 1.072 1.081 1.072 | 1.072 |
| serial 1D, ghost form | 16.924 | 18.096 | 1.055 1.072 1.069 | 1.072 1.070 1.069 | 1.069 |
| serial 3D, ghost form | 17.227 | 18.264 | 1.054 1.067 1.059 | 1.050 1.055 1.056 | 1.053 |
| canary: serial 3D, plain assemble | 0.522 | 0.503 | 0.996 0.990 0.927 | 1.000 1.016 0.951 | 1.004 |

1-minute load over the runs of each round, round 1 2.25 to 2.69, round 2 2.22 to 2.31, round 3 2.00 to 2.44.

### Step 6: `8d81c11d` against cfae0857 (#832, next commit that touches `ghost_assembly.jl`)

| row | median before, ms | median after, ms | ratio per round (median) | ratio per round (min) | ratio of minima |
|---|---|---|---|---|---|
| serial 2D, ghost form | 29.288 | 29.596 | 1.048 0.990 1.008 | 1.006 1.000 1.007 | 1.000 |
| serial 1D, ghost form | 18.129 | 18.297 | 1.019 1.000 0.996 | 0.999 0.998 0.997 | 0.998 |
| serial 3D, ghost form | 18.504 | 18.388 | 1.014 0.994 0.989 | 1.000 0.991 0.998 | 0.997 |
| canary: serial 3D, plain assemble | 0.501 | 0.504 | 1.011 0.987 1.020 | 1.007 0.987 1.005 | 0.999 |

1-minute load over the runs of each round, round 1 2.03 to 2.44, round 2 2.14 to 2.69, round 3 2.35 to 2.65.

### Step 7: `0aa87e45` against 9c34251c (main)

| row | median before, ms | median after, ms | ratio per round (median) | ratio per round (min) | ratio of minima |
|---|---|---|---|---|---|
| serial 2D, ghost form | 27.045 | 28.958 | 1.054 1.060 1.097 | 1.064 1.068 1.091 | 1.075 |
| serial 1D, ghost form | 16.975 | 18.157 | 1.060 1.082 1.097 | 1.074 1.082 1.081 | 1.082 |
| serial 3D, ghost form | 17.622 | 18.548 | 1.040 1.094 1.047 | 1.038 1.065 1.049 | 1.059 |
| canary: serial 3D, plain assemble | 0.510 | 0.523 | 0.960 1.006 1.108 | 0.969 1.009 1.046 | 1.009 |

1-minute load over the runs of each round, round 1 1.85 to 3.22, round 2 2.28 to 3.22, round 3 2.27 to 2.88.

### Step 8: `10739d67` against 9c34251c (main)

| row | median before, ms | median after, ms | ratio per round (median) | ratio per round (min) | ratio of minima |
|---|---|---|---|---|---|
| serial 2D, ghost form | 29.190 | 29.350 | 1.004 1.007 1.007 | 1.000 0.990 0.998 | 0.998 |
| serial 1D, ghost form | 18.519 | 18.470 | 1.007 0.992 0.997 | 0.998 0.995 1.008 | 1.005 |
| serial 3D, ghost form | 18.406 | 18.662 | 1.014 1.016 1.004 | 1.001 0.996 0.993 | 0.993 |
| canary: serial 3D, plain assemble | 0.507 | 0.506 | 0.995 1.002 0.997 | 0.994 1.007 0.997 | 0.998 |

1-minute load over the runs of each round, round 1 2.72 to 2.88, round 2 2.72 to 3.67, round 3 2.80 to 3.67.

## Profile of `form` in `8d81c11d` against its parent `7811c79b`

2D serial, `_gvp_a(Cg)` (the benchmark row), same ballast, one fresh process per commit
(`prof.jl`, `goprof.sh`; output in `prof_*.txt`), 300 samples of each piece, minimum:

| piece | `7811c79b` (parent), ms | `8d81c11d`, ms |
|---|---|---|
| `form` (the row) | 25.76 | 27.61 |
| `allocate_system_matrix` | 6.21 | 6.14 |
| `_ghost_reduction_map` | 19.25 | 21.20 |
| `_ghost_lift_map` | 0.153 | 0.150 |

`_ghost_reduction_map` takes +1.95 ms of the +1.85 ms of `form`, with the allocation
(91.67 MB) unchanged. The sampling profile (150 forms, 0.5 ms delay) agrees: `_ghost_bilinear_form`
7197 against 7549 samples; inside `_ghost_reduction_map` the final `sparse(rs, cs, ...)` (2556
against 2528) and `sortperm(entry)` (2561 against 2534) do not move, and the accumulation
loop over the stored entries of `A_pad` goes from about 760 samples (lines of the inline
loop) to about 1450, with 1051 of them as self time of the new local function
`contribute!`.

The cause is in the code of that commit. It turned the accumulation loop of
`_ghost_reduction_map` into a call of the local function `contribute!(k, i, j, s)`, shared by
the loop over every stored entry (`s = true`) and by a second loop over the polygon `moves`
(an empty tuple of vectors off a polygon space, so no work on the benchmark). The call is not
inlined (a closure with a double loop and four `push!`es), and the loop pays the call and
the `Bool` multiplier `s * ev[p] * eu[q]` on each stored entry of the 2D row. The
experiment below shows that inlining alone removes the cost; which of the two costs more
was not separated.

Experiment (no repo file edited; `exp.jl`, output `exp.txt`, one process, interleaved 3
rounds, minimum of 60 samples of `_ghost_reduction_map` alone on the 2D benchmark matrix,
results identical to the committed function): the parent's loop 19.34 ms; the committed
`contribute!` 20.88 ms (1.079); the same function marked `@inline` 19.02 ms (0.983); the
parent's loop kept inline and the moves loop separate 18.97 ms (0.981). Both variants give
back the whole slowdown.

## Cause

Commit `8d81c11d8ceb743c9fba1a3620b65cddf5053838` (`8d81c11d`), function
`_ghost_reduction_map` in `src/assembly/ghost_assembly.jl` (the local `contribute!`, line 479 on
main). The polygon work needed the reduction map to take extra contributions (`moves`) and
did it by sharing one non-inlined local function between the main loop and the new one.
Nothing else moves the row: `7811c79b` (step 4) is at 1.00 of v4.0.0, `cfae0857` (step 6)
at 1.00 of `8d81c11d`, and main (`9c34251c`) at 0.99 to 1.01 of `10739d67` (step 8), so
#856, #857, #858 and the later merges add nothing. Main against v4.0.0 (step 7) gives the
issue's 1.06 to 1.08 on the minimum, the same size as the +1.95 ms of this function in the
profile, so this one commit accounts for it. Every policy shows the same slowdown because the map is built serially on the host.

## Proposal

Fix, in a later subplan: `src/assembly/ghost_assembly.jl`, `_ghost_reduction_map`. Mark the
local function `@inline function contribute!(k, i, j, s)` (a one-word change measured at
0.983 of the parent), or keep the parent's inline accumulation loop and write the `moves`
loop separately with the multiplier (0.981). Prefer `@inline`: it keeps one copy of the
body. The check is the serial 2D, 1D and 3D `ghost form` rows against v4.0.0 (interleaved, 3
rounds, ratio of minima within the canary's spread, about 1.00) and `Bramble._ghost_reduction_map`
alone against `7811c79b` (this note's `exp.jl`); the tests that pin the reduction map's
output stay green because the variants give identical vectors. Nothing is accepted as cost.

## Left on disk

- Harness, raw runs and profiles: `~/bramble-scratch/v413perf/s31/`.
- The bisect worktrees `~/worktrees/Bramble/bisect-907-*` are removed at the end of S3.1.
