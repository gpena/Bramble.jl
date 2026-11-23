# First `jacobian!` and precompile, v4.8.0 part 2 (S1.4, #1265)

Measures the first `jacobian!` call before and after the ForwardDiff workload that S1.1 added
to `ext/BrambleForwardDiffExt.jl` (commit 03d06a27). It also records the core precompile time
and the precompile cost of `BrambleForwardDiffExt`.

Two sides: main `7766200b` ("before") and the branch `v4-8-0-part2` at `21b57773` ("after"),
the branch head when the run was made.

Summary:

- The first `jacobian!(J, jf, u)` launch falls from a median of 182.9 ms to 49.0 ms, a reduction
  of 133.9 ms. The reduction is about 20 times the larger spread (6.7 ms). Every launch after is
  faster than every launch before.
- The core precompile is unchanged (+0.013 s). The `BrambleForwardDiffExt` precompile costs
  about +0.4 s, one run per tree, which is inside what one run can resolve.

## What changed

S1.1 added a `@setup_workload`/`@compile_workload` block to `ext/BrambleForwardDiffExt.jl`. On a
2D 8x8 non-uniform mesh it builds the quasilinear form `inner₊(c * ∇ₕ(U)[1], ∇ₕ(V)[1]) +
innerₕ(U, V)`, then calls `jacobian_form` with and without `dirichlet = :boundary`,
`jacobian_prototype` and `jacobian!`.

#1265 proposed a core session. A core call of `jacobian!` throws without the extension, and its
cached instance is invalidated on `using ForwardDiff` (part 1). So the workload lives in the
extension.

## Method

```bash
julia --startup-file=no benchmark/ttfx.jl --before ~/bramble-scratch/v480p2-base \
  --after ~/worktrees/Bramble/v4-8-0-part2 --runs 3 --threads 4
```

Run on 2026-11-23 in the user's terminal, with the whole Julia pool
(`BRAMBLE_JULIA_SLOTS=4 julia_slot.sh 4`), on AC power. Three runs per tree. The load was not
quiet: the 1-minute load average was 2.02 at the start, just above the preflight threshold of 2,
so every figure here is labelled as taken at that load. The full log was kept with the plan.

## First `jacobian!`

| case | before median (ms) | after median (ms) | reduction (ms) | spread before / after (ms) |
|---|---|---|---|---|
| `jacobian!(J, jf, u)` | 182.9 | 49.0 | 133.9 | 6.7 / 1.6 |

The three launches, in seconds: before 0.188161208, 0.18292675, 0.181486834; after 0.050106709,
0.049004542, 0.048532625. The other rows of the ttfx table are unchanged within their spreads.
Some carry FAIL labels, which come from the fixed thresholds of older cases, not from this
change, so they are not tabulated here.

## Precompile

| figure | before | after | difference |
|---|---|---|---|
| core precompile, s, median of 3 | 62.367 | 62.38 | +0.013 |
| `BrambleForwardDiffExt` precompile, s, one fresh-depot run per tree | 7.4 | 7.8 | +0.4 |

The core is unchanged, so its +0.8 s ceiling is vacuous here: the harness's precompile lines do
not load ForwardDiff. The harness's extension line, which also skips ForwardDiff, reads 86.503 s
before and 86.221 s after (info only).

The `BrambleForwardDiffExt` figure is a separate measurement. Each tree's test project gets a
fresh depot stacked over `~/.julia`, with `using Bramble` run first untimed, then
`using Bramble, ForwardDiff` timed once, at 4 threads. One run per tree cannot separate +0.4 s
from noise, so the figure is recorded for information only.

## Verdict

The first `jacobian!` is faster by more than its spread, and the core precompile is unchanged.
The figures are under load 2.02, not on a quiet machine.
