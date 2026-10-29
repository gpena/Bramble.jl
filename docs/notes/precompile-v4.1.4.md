# Precompile time, image size and first calls, v4.1.4 against main (S16)

Measures the finished `v4.1.4/compile` branch against main for #987, #1021, #1022, #1023 and
#1031: the precompile and pkgimage of Bramble and `BrambleSciMLExt`, the method instances
inferred by three first calls, a user composite form, a 3D shift form, `c1031.jl`, and two test
files.

Two sides: main `c080379d` ("main") and the branch at `ce327f4a` ("branch"), which includes
origin/main `c080379d`. Every figure below is of `ce327f4a`, before the S14 retry; the retry
is measured in its own section at the end (it adds about +1.1 MiB of pkgimage and nothing
measurable to the precompile time). The plan names `255ed7b2` as the reference; the branch has merged
`c080379d` since, so the comparison is against `c080379d`, and only the branch's own changes
differ.

Summary:

- #987 passes. The first `prob.f` and `prob.f.jac` of an `ode_problem` infers no Bramble method
  on the branch (5 on main). The first `assemble` of `innerₕ(D₋ₓ(u), D₋ₓ(v))` infers 52 against
  55 and the first `Rₕ!` on a `Parallel()` space 18 against 18, so neither grows.
- #1021 passes. The first `assemble` of a user composite form is the same on both sides inside
  the spread. The first `form` of a source that names no component costs 13 ms more on the
  branch (134 ms against 120 ms).
- #1022 passes. The Bramble precompile is 9.0 s shorter (112.9 s to 103.9 s) and the pkgimage
  22.5 MiB smaller. `BrambleSciMLExt` is inside its spread on time and 0.4 MiB larger.
- #1023 failed on the branch as measured and passes after S14's retry. At `ce327f4a` the first 3D
  shift `assemble` is 184 to 237 ms, against the 78 to 114 ms the 3D exclusion precedent
  covers (3.5 to 3.9 ms on main). The retry restores a forward-only 3D shift pass in the
  workload; the first 3D shift `assemble` is then 3.5 to 4.2 ms (see `## The S14 retry`).
- #1031 reported only: the first contraction at 9 summands is 0.171 s on the branch against
  0.165 s on main, and equal at 3 and 6 summands.

## Method

Julia 1.13.3, Apple M2 (8 logical cores, 16 GB), AC power, default `-O` (never
`--optimize=1`), `--threads=4`, `BRAMBLE_JULIA_SLOTS=4`, every Julia through `julia_slot.sh 4`.
Both sides run `--project=<tree>/test`, which resolves against the one root `Manifest.toml`
the two trees share (checked with `diff`), so only `src/` and `ext/` differ. Main is the
detached worktree `~/bramble-scratch/v414c-855/main`, untouched. Every figure is a
back-to-back pair, three runs, the order alternating per run (main first, branch first, main
first), so drift hits both sides. A difference inside the spread of the runs is reported as
"no change".

Conditions. Not a quiet machine: `gate_preflight.sh --timing` failed on load for every group
(1-minute load 2.1 to 2.6, with no other Julia running; the Claude app's own renderer
accounted for it, and an earlier preflight named `corespotlightd`), so every figure is under
that load. The preflight lines are in `~/bramble-scratch/v414c-s16/logs/preflight.txt`. The
load at the start and end of each build is in `logs/build.log`: 2.1 to 3.2 for the first pair,
rising to 4.2 and 5.2 over the last pair (the builds' own threads). The scripts, the
per-run logs and the depots are under `~/bramble-scratch/v414c-s16/`. Each depot ends in
`.noindex` so Spotlight does not index the freshly written cache.

## Precompile time and pkgimage size

The brief says `Pkg.precompile`. A bare `Pkg.precompile()` in the test environment would
rebuild every Bramble extension (Makie, Meshes and the rest) that goes stale with a new
Bramble cache, which would swamp the figure. So each build is a fresh depot
(`JULIA_DEPOT_PATH=<tmp>:~/.julia`, `JULIA_PKG_OFFLINE=true`) and one process that times
`using Bramble` (builds the Bramble pkgimage) and then `using SciMLBase` (loads SciMLBase from
its existing cache and builds `BrambleSciMLExt`). The SciMLBase load is the same on both sides.
Each build was checked clean: `<depot>/compiled/v1.13/Bramble` and `.../BrambleSciMLExt` exist
in the fresh depot after it, and the time is the 100 s of a build, not a cache hit. The
pkgimage size is `du -sk` of those two directories.

| figure | main (3 runs) | branch (3 runs) | main median | branch median | difference |
|---|---|---|---|---|---|
| Bramble precompile, s | 111.48, 112.86, 113.41 | 103.28, 103.85, 103.86 | 112.86 | 103.85 | -9.0 s (-8%) |
| `BrambleSciMLExt` build with the SciMLBase load, s | 22.89, 23.54, 23.41 | 23.23, 24.01, 22.59 | 23.41 | 23.23 | no change |
| Bramble pkgimage, KiB | 274544, 273520, 273560 | 250468, 250512, 250584 | 273560 (267.2 MiB) | 250512 (244.6 MiB) | -23048 KiB (-22.5 MiB) |
| `BrambleSciMLExt` pkgimage, KiB | 1540, 1540, 1540 | 1968, 1968, 1972 | 1540 | 1968 | +428 KiB (+0.4 MiB) |

The spread of one side's three runs is 1.9 s (main) and 0.6 s (branch) for Bramble, and 0.7 s
and 1.4 s for the extension. The Bramble difference is 4.7 times the larger spread, and each of the three
pairs shows it (-8.2, -9.0, -9.5 s). The extension's time difference (0.2 s) is inside the
spread: no change. Its image grew by 0.4 MiB, consistent with the new `prob.f` and `prob.f.jac`
calls of the extension workload (S11); that attribution was not isolated.

Not measured: which of the branch's workload edits account for the 9.0 s. The in-order and
drop-one tools of the v4.1.3 note were not rerun, since the figure asked for here is the
whole branch.

## First-call method instances (`@snoop_inference`, #987)

Each scenario is a fresh process on the pkgimage of its tree. `SnoopCompileCore` is loaded,
then Bramble and SciMLBase (the first scenario needs `BrambleSciMLExt`). Everything before the
call is built outside the snoop. `SnoopCompileCore.@snoop_inference` wraps only the call, and
the count is the number of method instances in the inference tree whose method belongs to a
module named `Bramble*` (the extension included). The total counts every instance, SciMLBase
and Base ones too. The functions and closures are the script's own, as a user's are. The
scripts are `fig/c_snoop.jl` and `fig/c_names.jl`.

| first call | main Bramble instances | branch Bramble instances | main total | branch total | inference time, main | inference time, branch |
|---|---|---|---|---|---|---|
| 1. `prob.f` then `prob.f.jac` of `ode_problem(semidiscretize(a, l; dirichlet = :boundary), u0, I)`, 1D | 5, 5, 5 | 0, 0, 0 | 11 | 5 | 59.5, 58.8, 59.1 ms | 48.2, 48.2, 48.1 ms |
| 2. `assemble` of `innerₕ(D₋ₓ(u), D₋ₓ(v))`, 1D uniform | 55, 55, 55 | 52, 52, 52 | 71 | 65 | 173.2, 169.5, 169.3 ms | 157.8, 158.1, 156.8 ms |
| 2. the same on a non-uniform mesh | 55, 55, 55 | 52, 52, 52 | 71 | 65 | 169.9, 168.0, 168.5 ms | 159.4, 158.4, 157.5 ms |
| 3. `Rₕ!(uₕ, x -> 2x)` on a `Parallel()` space | 18, 18, 18 | 18, 18, 18 | 30 | 30 | 40.0, 41.7, 40.4 ms | 42.6, 38.8, 39.0 ms |

The counts have no spread. Scenario 1: the five main instances are the `Semidiscretization`
functor, the Jacobian closure of the extension, `_assemble_source!`, `jacobian!`, and
`_update_coefficients!(::Nothing, ::SciMLBase.NullParameters, ::Float64)`. A solver reaches
these with `p = NullParameters()`, and all five are cached on the branch. In scenario 2 the
three instances missing from the branch's list are `local_stencil`, `_walk_space` and `_check_product_reads`; the other 52 are the same
on both sides, so the first `assemble` infers less on the branch and nothing new. The
inference time falls by about 10 to 11 ms (medians), more than the spread of the runs
(3.9 ms on main, 1.3 ms on the branch for the uniform mesh). Which edit removed the three was
not traced. Scenario 3 is unchanged: the same count, and the difference of the medians
(1.4 ms) is inside the spread (3.8 ms).

Verdict #987: (1) falls (5 to 0), (2) does not grow (55 to 52), (3) does not grow (18 to 18).
Pass.

## A user composite form (#1021)

`fig/d_comp.jl`: a 1D non-uniform mesh of 12 points, `V = W × W`, user closures `g` and `g2`.
Three forms, built and assembled in one fresh process in this order, each through
`Base.invokelatest` so the compile happens at the call and not when the script is compiled.
Wall time from `@elapsed`; the process runs with `--trace-compile=stderr
--trace-compile-timing`, with marker lines between the steps, and the table gives the methods it
traced and their compile time.

| form | step | main (3 runs), ms | branch (3 runs), ms | reading |
|---|---|---|---|---|
| `form(V, v -> innerₕ(g, v))`, a source naming no component (#1021) | `form` | 120.3, 120.2, 120.3 | 134.1, 133.8, 133.2 | +13.5 ms on the branch |
| | `assemble` | 65.6, 65.5, 65.8 | 65.8, 64.9, 65.2 | no change |
| `form(V, v -> innerₕ(g, v(1)) + innerₕ(g2, v(2)))`, named components | `form` | 90.7, 90.3, 90.2 | 91.4, 91.6, 91.4 | +1.0 ms, at the edge of the spread |
| | `assemble` | 17.1, 17.0, 16.9 | 17.2, 17.0, 17.1 | no change |
| `form(V, V, (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(u(2), v(2)))` | `form` | 13.7, 13.8, 14.2 | 13.8, 13.8, 13.9 | no change |
| | `assemble` | 0.0, 0.0, 0.0 | 0.1, 0.0, 0.0 | no change |

Traced compile: the no-component form traces 1 method (1.8 ms) on main and 2 (6.1 ms) on the
branch in `form`, and 1 method (1.4 against 2.0 ms) in `assemble`; the other forms trace the
same number of methods on both sides. `--trace-compile-timing` prints only the specializations
that become precompile statements, so the wall times above are the figure and the trace is the
cross-check.

The first `form` plus `assemble` of the no-component source is 186 ms on main and 199 ms on the
branch: dropping the workload call that compiled `SourceFunction{typeof(f)}` cores (#1021) moves
13 ms from the precompile to a user's first `form` of that shape, and none to the first
`assemble`.

Verdict #1021: the first composite `assemble` is no slower than main beyond the spread, in all
three shapes. Pass. The `form` of the no-component source is 13.5 ms slower; the acceptance
criterion names `assemble`, so it is a note and not a fail.

## The first 3D shift assemble (#1023)

`fig/e_shift.jl`: a non-uniform (4, 3, 3) mesh, `form(W, W, (u, v) -> innerₕ(S(u), v))` and
`assemble` for `S₊ₓ`, `S₊ᵧ` and `S₊₂` in that order, one fresh process per run, only the
`assemble` timed (the form is built before the timer). The branch's 3D shift pass is gated off,
so the first call is cold by design.

| shape | main (3 runs), ms | branch (3 runs), ms |
|---|---|---|
| `S₊ₓ`, first in the process | 3.7, 3.6, 3.6 | 237.1, 232.0, 231.2 |
| `S₊ᵧ`, second | 3.6, 3.6, 3.6 | 198.0, 196.3, 196.5 |
| `S₊₂`, third | 3.9, 3.6, 3.5 | 187.2, 186.8, 184.4 |

The 3D exclusion precedent in `src/precompile/form_sessions.jl` is 78 to 114 ms per shape. The
first 3D shift `assemble` on the branch is not within it: 231 to 237 ms, and even the third
shape, after the 3D machinery is warm, is 184 to 187 ms. Main's 3.5 to 3.9 ms is the warmed
3D shift pass. So a user's first 3D shift form pays about 190 to 230 ms on the branch that main
had cached. The precompile time saved by the gate was not measured on its own; the whole-branch
figure is 9.0 s.

Verdict #1023 at `ce327f4a`: fail. The first 3D shift `assemble` is outside the 78 to 114 ms
precedent. The retry below changes it.

## The S14 retry (#1023)

The fail above went back to S14 through §8. The retry restores a 3D shift pass in the precompile
workload, forward shifts only (`S₊ₓ`, `S₊ᵧ`, `S₊₂`); the 3D neighbour sums stay cold. The figures
below are the retry against its own parent, `ce327f4a` ("HEAD", the 3D pass gated), not against
main: fresh depots, 4 threads, default `-O`, 3 runs each, taken by the coordinator in two
batches. The tables of the sections above were measured before the retry, so the retry adds
about +1.1 MiB to the branch's Bramble pkgimage and nothing measurable to its precompile time.

Conditions. Batch 1: the preflight CPU check failed (the Claude renderer at 236%), and the
load during and after the first-call runs was 3.16 to 3.39. Batch 2: every preflight line was
OK. The batch 2 builds ran about 4.5 s slower on both sides, so a difference is read within one
batch.

| figure | HEAD `ce327f4a` (3 runs) | retry (3 runs) | reading |
|---|---|---|---|
| Bramble precompile, s | 97.34, 97.22, 98.11 | 97.74, 98.07, 98.25 | +0.46 s of the means, inside HEAD's 0.9 s spread: no change |
| Bramble pkgimage, KiB | 250564, 250492, 250644 | 251760, 251732, 251588 | +1.10 MiB |

First 3D assemble in a fresh process (the shapes of `fig/e_shift.jl`), pooled over 6 runs per
side, the range of the runs:

| first assemble | HEAD, ms | retry, ms |
|---|---|---|
| `S₊ₓ` | 230.6 to 245.0 | 3.7 to 4.1 |
| `S₊ᵧ` | 199.3 to 206.3 | 3.6 to 4.2 |
| `S₊₂` | 183.4 to 228.7 | 3.5 to 3.9 |
| 3D neighbour sum, d = 1, 2, 3 | 97.3 to 103.6 | 101.4 to 110.7 |

The forward shifts fall from 183 to 245 ms to 3.5 to 4.2 ms, the level main had. The 3D
neighbour sums stay cold on both sides and are 97 to 111 ms, inside the 78 to 114 ms precedent,
so leaving them out costs a user nothing the precedent does not already accept.

A rejected variant, forward shifts plus the 3D sums ("full"): +1.85 MiB of pkgimage (spread 1.1
MiB), precompile +0.53 s inside the spread, and the sums fall to 3.6 to 4.0 ms. It was not kept,
since the sums already sit inside the precedent.

Verdict #1023 after the retry: pass. The first 3D shift `assemble` is 3.5 to 4.2 ms, within the
78 to 114 ms precedent (and below it), with no measurable precompile cost and +1.1 MiB of image.

## `c1031.jl` (#1031, reported only)

`~/bramble-scratch/v414d/c1031.jl`, copied to `fig/f_c1031.jl`: a composite `l = form(V, v ->
…)` of n summands on a 1D non-uniform mesh of 9 points, and the first `l(uv)` in a fresh
process, one process per n and run.

| summands | main (3 runs), s | branch (3 runs), s | reading |
|---|---|---|---|
| 3 | 0.048, 0.046, 0.045 | 0.184, 0.046, 0.046 | no change (see below) |
| 6 | 0.102, 0.104, 0.101 | 0.100, 0.103, 0.099 | no change |
| 9 | 0.164, 0.165, 0.167 | 0.171, 0.172, 0.171 | +0.006 s (+4%), branch slower |

The 0.184 s at 3 summands is the first branch run of the group and did not repeat: three more
pairs gave 0.045, 0.046, 0.045 s on the branch and 0.045, 0.045, 0.045 s on main
(`logs/f3_rerun.txt`). The printed values of `l(uv)` differ in the last bit between runs of
the same tree (3.0, 3.0000000000000004, 2.9999999999999996 on main), so the `==` of the issue's
first criterion cannot be read off these runs; every value agrees to 2e-15.

## Test files' first-call cost

`fig/g_test.jl`: a fresh process, `BRAMBLE_TEST_GROUP=unit`, `test/TestUtils.jl` included before
the timer, then the wall time of one `include` of the test file. Default `-O` (the suite runs
`unit` at `-O0`, so these are not the suite's own times). Both files are identical on the two
trees, every test passes.

| file | main (3 runs), s | branch (3 runs), s | reading |
|---|---|---|---|
| `test/form/operators.jl` | 24.79, 24.88, 24.74 | 24.81, 24.57, 24.93 | no change |
| `test/form/stencil_pattern.jl` | 1.57, 1.72, 1.55 | 1.64, 1.63, 1.62 | no change |

## Verdicts

| issue | criterion | verdict |
|---|---|---|
| #987 | (1) falls, (2) and (3) do not grow | pass: 5 to 0, 55 to 52, 18 to 18 |
| #1021 | first composite `assemble` no slower than main beyond the spread | pass: 65.6 against 65.2 ms (medians) and the other two shapes the same; the no-component `form` is 13.5 ms slower |
| #1022 | precompile time not worse | pass: Bramble -9.0 s, `BrambleSciMLExt` no change |
| #1023 | first 3D shift `assemble` within 78 to 114 ms | fail at `ce327f4a`: 184 to 237 ms (main 3.5 to 3.9 ms); pass after the S14 retry: 3.5 to 4.2 ms, precompile no change, +1.1 MiB |
| #1031 | reported only | 9 summands +0.006 s, 3 and 6 no change |

Not done: a precompile figure per subplan (the plan rejects it), the in-order and drop-one
attribution of the 9.0 s, and a run on a quiet machine (the load figures above).
