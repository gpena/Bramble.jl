# Ghost form dispatch: precompile time, pkgimage size and JET counts (v4.1.2, S4)

Answers #824 AC5 and #811 AC3: core precompile time and pkgimage size before and after the
ghost-dispatch work, plus the JET counts of `_pc_form_nested` and the inferred type of
ghost `gmg_solve` (#824 (3)).

Before is main `10739d67` (the merge base: origin/main was merged into the branch as
`a36d6c6b`). After is branch `perf/ghost-form-dispatch` at `7e02a214`. Both are built from
a worktree of their own, with the same `Manifest.toml` (gitignored, copied from the branch
worktree into the detached base worktree), so the dependencies are identical and only the
branch's commits differ.

## Method and conditions

Apple Silicon, Julia 1.13.2, default `-O`. Same method as `precompile-v4.0.1.md`: a fresh
temporary depot per build (`JULIA_DEPOT_PATH=<tmp>:~/.julia`, `JULIA_PKG_OFFLINE=true`),
`using Bramble` timed by `/usr/bin/time -p` inside
`BRAMBLE_JULIA_SLOTS=4 julia_slot.sh 4 julia --threads=4 --startup-file=no --project=.`
(GC left at the script's 1 thread, as in v4.0.1), median of 3, the Bramble pkgimage size
read from `<depot>/compiled/v1.13/Bramble` with `du -sk`. Three rounds, each one build of
each side, the order alternating (before first in rounds 1 and 3, after first in round 2),
so drift hits both sides.

Measured under load, not alone: `gate_preflight.sh --timing` fails on load (about 3.3 at
the start, another session idle but Spotlight and other daemons running), so the absolute
seconds are inflated and the conclusion rests on the per-round before/after ratio. The
1-minute load from `uptime` at the start and end of each build is in the table. An earlier
attempt at 1 thread was discarded in full, so every figure below uses the one setting.

## Precompile time and pkgimage size

Raw runs, in seconds (`real` of `/usr/bin/time -p`), and the 1-minute load at the start
and end of each build:

| round | order | before, s | load before | after, s | load after | ratio after/before |
|---|---|---|---|---|---|---|
| 1 | before, after | 105.56 | 2.93 to 4.26 | 106.06 | 4.26 to 5.28 | 1.005 |
| 2 | after, before | 107.43 | 4.26 to 3.95 | 107.39 | 5.28 to 4.26 | 1.000 |
| 3 | before, after | 107.04 | 3.95 to 4.08 | 106.35 | 4.08 to 4.69 | 0.994 |

The median of 3: before 107.04 s, after 106.35 s. The median of the three per-round ratios
is 0.9996. The run to run spread of one side is 1.9 s (before) and 1.3 s (after), larger
than the 0.7 s between the medians.

Sizes read from the depots (KiB, `du -sk`), by round:

| quantity | before (rounds 1, 2, 3) | median before | after (rounds 1, 2, 3) | median after |
|---|---|---|---|---|
| depot `compiled/` | 276136, 276156, 276148 | 276148 | 274388, 274444, 274428 | 274428 |
| Bramble pkgimage | 264244, 264264, 264256 | 264256 | 262496, 262552, 262536 | 262536 |

The Bramble pkgimage is 1720 KiB (1.7 MiB, 0.65%) smaller after. Its `.dylib` is
228451488 to 228501360 bytes before and 226910928 to 226993664 bytes after.

Conclusion: the ghost-dispatch work does not change the core precompile time beyond the
run to run noise (ratios 1.005, 1.000, 0.994) and trims the pkgimage by 1.7 MiB. Nothing is
gated: the issues ask for figures.

## JET counts of `_pc_form_nested`

`JET.report_opt(Bramble._pc_form_nested, (Val{D},))` with JET 0.12.3 from a temporary
environment stacked on the load path, at 4 threads, each side from its own worktree
(1-minute load 4.00 before, 3.93 after). A report counts as under `constrained` when any
frame of `rep.vst` has `f.linfo isa Core.MethodInstance && f.linfo.def isa Method &&
f.linfo.def.name === :constrained`. The filter `target_modules = (Bramble,)` is the one
`test/quality/jet.jl` uses.

| `_pc_form_nested` | filter | before total | before under `constrained` | after total | after under `constrained` |
|---|---|---|---|---|---|
| `Val(1)` | all modules | 291 | 235 | 95 | 0 |
| `Val(1)` | `(Bramble,)` | 133 | 103 | 56 | 0 |
| `Val(2)` | all modules | 612 | 239 | 457 | 0 |
| `Val(2)` | `(Bramble,)` | 239 | 99 | 161 | 0 |

The before counts match the planning measurement on main `578aa546` (291 / 235 and
133 / 103 for 1D; 609 / 239 and 239 / 99 for 2D; the all-modules 2D total is 612 here against
609 there, a difference not investigated). No report is under
`constrained` after.

Base and stdlib share of the all-modules reports (reported, not gated), counted as the
reports whose last frame is outside the Bramble source tree: before, 158 of 291 for
`Val(1)` (54%, 132 of them under `constrained`) and 373 of 612 for `Val(2)` (61%, 140 under
`constrained`); after, 39 of 95 (41%, none under `constrained`) and 296 of 457 (65%, none
under `constrained`). The reports that remain after are the stdlib calls outside
`constrained` (`lu`, `Dict`, `sparse`, broadcasting) that #824 names; they are not
dispatch from `constrained`.

## Ghost `gmg_solve`'s inferred type

The case is the one pinned in `test/quality/type_stability.jl`: a `build` closure over
locals that constrains a ghost `:boundary` space with a Robin relation on four facets and
returns `form(C, C, innerₕ(-1.0 * Δₕ(u), v) + innerₕ(u, v))`, on the 2D 9x9 periodic mesh,
with the right side assembled from the form. `Base.return_types(gmg_solve, (typeof(build),
typeof(Ω), typeof(F)))` is one concrete type before and after, identical on both sides:
`Bramble.VectorElement{Bramble.GhostGridSpace{2, Float64, ...}, Float64, Vector{Float64}}`.
So #824 (3) already held on main when the build captures locals, and the work keeps it.
