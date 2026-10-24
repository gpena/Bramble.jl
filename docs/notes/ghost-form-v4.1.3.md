# Ghost form slowdown since v4.0.0 (v4.1.3, S3.1, #907)

## Partial

Paused by the user mid-bisect. The final `## Method`, `## Cause` and proposal sections are
not written yet. Done so far, and the range still suspect, below.

State at the pause: the suspect is merge `3b0ee390` (#832, v4.1.0 polygons). Its first-parent
diff touches `src/assembly/ghost_assembly.jl` (+182 lines) and
`src/assembly/ghost_contract.jl`, both on the ghost `form` path, so the commits of #832 that
edit those files (`8d81c11d`, `cfae0857`) are the first to test. Not yet done: the bisect
inside the 23 commits of #832 (`3b0ee390^1..3b0ee390^2`), the profile of the guilty commit's
`form` against its parent, and a repeat of step 3 (its round 3 ran at 1-minute load 4.13).

## Method so far

Julia 1.13.2, Apple Silicon, 4 threads, default `-O`, AC power. One fresh process per run
(`julia_slot.sh 4`, `BRAMBLE_JULIA_SLOTS=4`), the v4.0.0 baseline
`benchmark/baselines/baseline_1784e815.json` supplying `evals`/`samples` of each row (as
`benchmarks.jl --compare` does). Rows measured: serial 2D, 1D and 3D `ghost form`, serial 3D
`plain assemble` (canary), serial 2D `plain form` (a 17 ns lazy constructor, too short to
read; ignore it). Interleaved, 3 rounds, order alternating (before first in rounds 1 and 3).

Each commit is built in a detached worktree `~/worktrees/Bramble/bisect-907-<sha>` with the
branch worktree's `Manifest.toml` copied in, so dependencies are identical on every side
(v4.0.0 and main have different Manifests). The v4.0.0 side is `bisect-907-0aa87e45`, not the
older `~/worktrees/Bramble/v4.0.0`, for the same reason.

The harness is `~/bramble-scratch/v413perf/s31/` (`run.jl` is the measured script,
`go.sh` one run, `pair.sh` one bisect step, `summ.py` the ratios, `runs/` the raw output).
It runs only the rows above, on the main checkout's `ghost_vs_plain.jl` (identical to
v4.0.0's) copied to `gvp.jl`. It holds a 500 MB ballast array so the GC heap target is the
suite's: without it the 2D `ghost form` median is bimodal on GC (56 ms against 27 ms) and
the minimum is the only stable figure. With the ballast the v4.0.0 medians are 27.4 ms (2D),
16.9 ms (1D), 17.4 ms (3D), the figures of a full `_ghost_vs_plain_group()` setup in the
same process (27.4, 17.5, 17.4 ms) and of the issue (27.5, 17.5, 17.7 ms). The unballasted
first attempt is in `runs_noballast/` and gave the same ratios of minima (1.06 to 1.07).

Only merges whose first-parent diff touches `src/`, `ext/` or `Project.toml` can change the
timing: #826 (`2da88d6d`), #832 (`3b0ee390`), #835 (`35c4c66e`), #850 (`10739d67`), #856
(`bb293acf`), #858 (`7dbaa523`), #910 (`49f5fb39`), #911 (`9c34251c`). The other 11 of the 19
merges leave `src/` as their parent had it, so a source state is tested at the first merge
that gives it. Machine: not alone on a quiet machine (mds and daemons); 1-minute load in the
tables, 2.5 to 3.5, `check_power_load.sh` (threshold 4.0) passing, so every figure is
"under load".

## Steps

Ratios are after / before, per round (rounds 1, 2, 3), on the median and on the minimum of
each run's samples; ratio of minima is the minimum over the 3 rounds of each side.

### Step 1: v4.0.0 (`0aa87e45`) against v4.1.1 (`10739d67`, merge #850)

| row | median before, ms | median after, ms | ratio per round (median) | ratio per round (min) | ratio of minima |
|---|---|---|---|---|---|
| serial 2D, ghost form | 26.924 | 29.684 | 1.081 1.158 1.067 | 1.071 1.088 1.072 | 1.074 |
| serial 1D, ghost form | 16.749 | 18.308 | 1.077 1.158 1.076 | 1.076 1.082 1.079 | 1.078 |
| serial 3D, ghost form | 17.313 | 18.513 | 1.075 1.043 1.066 | 1.052 1.045 1.055 | 1.048 |
| canary: serial 3D, plain assemble | 0.513 | 0.523 | 0.975 1.041 1.093 | 1.014 1.017 1.002 | 1.014 |

Load at start and end of each run: 1.55 to 2.84. The slowdown is already in `10739d67`, so it
is not #856, #857, #858 (the issue's guess) or later; it entered between v4.0.0 and v4.1.1.

### Step 2: v4.0.0 against `3b0ee390` (merge #832, v4.1.0 polygons)

| row | median before, ms | median after, ms | ratio per round (median) | ratio per round (min) | ratio of minima |
|---|---|---|---|---|---|
| serial 2D, ghost form | 27.285 | 30.674 | 1.147 1.139 1.076 | 1.076 1.102 1.072 | 1.076 |
| serial 1D, ghost form | 17.092 | 18.860 | 1.135 1.101 1.083 | 1.106 1.076 1.083 | 1.089 |
| serial 3D, ghost form | 17.463 | 18.516 | 1.108 1.037 1.060 | 1.080 1.039 1.066 | 1.056 |
| canary: serial 3D, plain assemble | 0.550 | 0.520 | 1.070 0.945 0.820 | 1.049 1.002 0.865 | 0.996 |

Load 2.6 to 3.5. The canary's median spread here is 0.82 to 1.07 (a noisy round 3 of the
before side), wider than in step 1, but its minima (ratio of minima 0.996) are flat while
every ghost form row is 1.04 to 1.10 in all three rounds.

### Step 3: v4.0.0 against `2da88d6d` (merge #826, v4.0.1 follow-ups)

| row | median before, ms | median after, ms | ratio per round (median) | ratio per round (min) | ratio of minima |
|---|---|---|---|---|---|
| serial 2D, ghost form | 28.023 | 28.456 | 1.048 0.992 1.013 | 1.030 0.999 1.002 | 1.022 |
| serial 1D, ghost form | 17.341 | 18.221 | 1.053 1.005 1.515 | 1.018 1.023 1.442 | 1.023 |
| serial 3D, ghost form | 18.295 | 17.736 | 1.007 0.949 1.448 | 1.006 0.979 1.417 | 1.003 |
| canary: serial 3D, plain assemble | 0.541 | 0.593 | 1.048 1.099 1.507 | 1.050 1.059 1.499 | 1.050 |

Round 3 is disturbed (every row including the canary at 1.4 to 1.5, load 3.2 to 4.13) and to
be repeated. Rounds 1 and 2: 0.95 to 1.05 on the ghost form rows, no slowdown. #826 is not the
cause, so the slowdown entered with #832 (`3b0ee390`), the only merge with `src/` changes
between `2da88d6d` and `3b0ee390` (`12a58a1a` changes none).

## Left on disk

- Bisect worktrees, with the branch Manifest, precompiled, to be reused:
  `~/worktrees/Bramble/bisect-907-{0aa87e45,2da88d6d,3b0ee390,35c4c66e,10739d67,bb293acf,7dbaa523,49f5fb39,9c34251c}`.
  Remove with `git worktree remove --force <path>` at the end.
- Harness and raw runs: `~/bramble-scratch/v413perf/s31/` (`env/` holds BenchmarkTools and
  Polyester for the stacked load path; `mk.sh <sha>` builds a bisect worktree).
