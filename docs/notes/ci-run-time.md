# CI run time of the unit group

Where a warm pull-request run of `CI.yml` spends its time, why the `Forms` testset dominates
it, and what the v3.35.0 plan (gpena/Bramble.jl#480) changes. Every figure below was read from
a run's log (`gh run view <id> --log`) or its job and step timestamps. They are single GitHub
runner measurements on macOS, not tracked baselines: re-measure before relying on them.

The "before" run is 37341936090, a pull-request run of `CI.yml` on the `v3.27.0/correctness`
branch (the `macos-latest` leg, Julia 1.13) whose package cache was restored from the
nightly run 37318618310. The trace lines come from `CI=true`, which turns on
`TestUtils.TRACE_TESTS`: each file prints `✓   <file>  <seconds>  <memory>` when it ends.

## The warm run, by job part (run 37341936090)

The job took 32m43s (16:35:17 to 17:08:00 UTC, 5 October 2026).

| Part | Time |
|:--|--:|
| Set-up before the test step (checkout, `setup-julia`, cache restore 25 s, registry update, `julia-buildpkg`) | 51 s |
| Precompiling Bramble and its extensions (`20 dependencies successfully precompiled in 316 seconds`) | 316 s (5m16s) |
| `Forms` testset | 17m25.9s |
| Every other testset of `Core library` | 8m48.5s |
| Tear-down after the test step | 9 s |

`Core library` took 26m14.4s in all, so the rest is 26m14.4s less 17m25.9s. Within that
rest: `Operators` 3m27.0s, `Solvers` 2m32.3s, `Grid spaces` 1m03.8s, `Meshes` 31.1s,
`Utilities` 31.0s, `Static allocations` 17.7s, `Exporters` 9.2s, `Sets and Domains` 8.8s and
`Drivers` 7.4s. The test step itself ran 31m43s (16:36:08 to 17:07:51).

`Forms` alone is 66% of `Core library` (1045.9 s of 1574.4 s) and 53% of the job.

## Every Forms file, ranked

The columns after the first run are the same files in two other warm runs, to show the noise
(next section). Shares are of the 1045.88 s that `form/runtests.jl` reports in run
37341936090. The shard column is where S1 puts the file (see "The shard boundary").

| Rank | File | Shard | 37341936090 (s) | Share | 37335400202 (s) | 37592758673 (s) |
|--:|:--|:--|--:|--:|--:|--:|
| 1 | `matrix_free.jl` | forms-2 | 173.31 | 16.6% | 192.17 | 173.41 |
| 2 | `kronecker_edge.jl` | forms-2 | 114.74 | 11.0% | 115.28 | 100.44 |
| 3 | `linear.jl` | forms-1 | 107.55 | 10.3% | 139.41 | 85.34 |
| 4 | `kronecker.jl` | forms-2 | 96.35 | 9.2% | 96.12 | 106.98 |
| 5 | `bilinear.jl` | forms-1 | 94.41 | 9.0% | 90.60 | 69.28 |
| 6 | `kronecker_projection.jl` | forms-2 | 49.06 | 4.7% | 46.25 | 42.38 |
| 7 | `operators.jl` | forms-1 | 46.31 | 4.4% | 46.74 | 41.04 |
| 8 | `normal.jl` | forms-1 | 38.79 | 3.7% | 42.00 | 34.18 |
| 9 | `centered_vector_calculus.jl` | forms-1 | 37.80 | 3.6% | 40.39 | 34.24 |
| 10 | `nested_operators.jl` | forms-1 | 33.91 | 3.2% | 37.46 | 34.16 |
| 11 | `symmetry.jl` | forms-1 | 25.84 | 2.5% | 24.39 | 18.70 |
| 12 | `kronecker_block.jl` | forms-2 | 24.96 | 2.4% | 25.96 | 30.75 |
| 13 | `threaded_replay.jl` | forms-1 | 21.77 | 2.1% | 25.35 | 18.77 |
| 14 | `simplifier.jl` | forms-1 | 19.12 | 1.8% | 21.67 | 23.55 |
| 15 | `reaction_flux.jl` | forms-1 | 14.88 | 1.4% | 13.57 | 10.15 |
| 16 | `interpolation_operator.jl` | forms-1 | 13.93 | 1.3% | 16.04 | 12.95 |
| 17 | `coordinate_walk.jl` | forms-1 | 13.73 | 1.3% | 17.91 | 11.48 |
| 18 | `jacobian_pattern_blocks.jl` | forms-2 | 12.67 | 1.2% | 19.44 | 12.58 |
| 19 | `semidiscrete.jl` | forms-2 | 11.25 | 1.1% | 12.06 | 11.09 |
| 20 | `assemble_add.jl` | forms-1 | 10.89 | 1.0% | 14.91 | 8.85 |
| 21 | `dirac.jl` | forms-1 | 9.50 | 0.9% | 7.44 | 6.74 |
| 22 | `zero_form.jl` | forms-1 | 7.91 | 0.8% | 8.96 | 7.64 |
| 23 | `source_operators.jl` | forms-1 | 7.12 | 0.7% | 8.93 | 6.82 |
| 24 | `bandwidth.jl` | forms-2 | 6.47 | 0.6% | 7.25 | 5.89 |
| 25 | `inner_products.jl` | forms-1 | 5.91 | 0.6% | 5.61 | 4.45 |
| 26 | `skew.jl` | forms-1 | 5.12 | 0.5% | 5.46 | 4.33 |
| 27 | `dirichlet_constraints.jl` | forms-1 | 4.74 | 0.5% | 4.35 | 3.59 |
| 28 | `centered_average.jl` | forms-1 | 4.60 | 0.4% | 4.67 | 5.77 |
| 29 | `forwarddiff_smoke.jl` | forms-1 | 4.30 | 0.4% | 5.31 | 4.02 |
| 30 | `common.jl` | forms-1 | 3.85 | 0.4% | 3.94 | 2.85 |
| 31 | `difference_ast.jl` | forms-1 | 3.68 | 0.4% | 3.49 | 2.58 |
| 32 | `component.jl` | forms-1 | 3.21 | 0.3% | 3.23 | 4.77 |
| 33 | `block_extract.jl` | forms-1 | 3.04 | 0.3% | 3.24 | 4.07 |
| 34 | `interpolation.jl` | forms-1 | 2.79 | 0.3% | 3.59 | 2.43 |
| 35 | `stencil_pattern.jl` | forms-2 | 2.72 | 0.3% | 4.26 | 2.40 |
| 36 | `markers.jl` | forms-1 | 2.48 | 0.2% | 2.35 | 2.29 |
| 37 | `extended_operators.jl` | forms-1 | 2.44 | 0.2% | 2.86 | 2.30 |
| 38 | `type_cached_assemble.jl` | forms-2 | 1.53 | 0.1% | 1.64 | 7.61 |
| 39 | `symmetrize.jl` | forms-1 | 1.08 | 0.1% | 1.70 | 1.23 |
| 40 | `cross_mesh_blocks.jl` | forms-1 | 0.88 | 0.1% | 1.25 | 0.85 |
| 41 | `expression.jl` | forms-2 | 0.70 | 0.1% | 0.72 | 0.64 |

The files in the table sum to 1045.3 s; the other 0.5 s of the testset's 1045.88 s is not
attributed to a file by the trace. `coefficient_shift.jl`, `vector_calculus.jl`,
`jacobian_pattern.jl` and `compile_scaling.jl` are behind the `slow` group and do not appear.
In run 37592758673 two more files ran, `replay_sinks.jl` (2.76 s) and `marker_ids.jl`
(1.30 s), because that branch had added them.

Five files carry the profile: `matrix_free.jl` (173.3 s), `kronecker_edge.jl` (114.7 s),
`linear.jl` (107.6 s), `kronecker.jl` (96.4 s) and `bilinear.jl` (94.4 s) are 586.4 s, 56% of
the testset. With `kronecker_projection.jl` (49.1 s) the six are 635.4 s, 60.8%. The other
35 files together take 409.9 s. These six are exactly the files S3 changes.

## Noise between warm runs

Three warm macOS runs of the same job, each restoring a nightly cache:

| Run | Branch | Job | Precompile | `Core library` | `Forms` | `Forms` tests |
|:--|:--|--:|--:|--:|--:|--:|
| 37341936090 | `v3.27.0/correctness` | 32m43s | 316 s | 26m14.4s | 17m25.9s | 7982 |
| 37335400202 | `claude-plugin` | 29m36s | 8 s | 28m08.7s | 18m48.4s | 7982 |
| 37592758673 | `v3.25.0/polyester-forms` | 29m06s | about 5 s | 24m23.9s | 16m08.0s | 8069 |

The first two ran the same 7982 `Forms` tests, so the 82.5 s between their `Forms` times
(1045.9 s against 1128.4 s, 7.9%) is runner noise. The third ran 87 more tests on a
different branch, so its 968.0 s is not a like-for-like figure. Per file the spread is wider:
`linear.jl` took 107.6, 85.3 and 139.4 s, `kronecker_edge.jl` 114.7, 100.4 and 115.3 s,
and `type_cached_assemble.jl` 1.5, 7.6 and 1.6 s. The ranking at the top is steady: the
same five files lead in all three runs. So a before and after comparison of one file is
only as good as that spread (up to 1.6x on `linear.jl`); the testset total moves by about
8% between identical runs, and a change must be judged against that.

The precompile column explains why the job times differ less than the `Core library` times
do: the first run recompiled 20 dependencies (316 s), the other two almost none.

## Cause: Union-typed case loops

`Forms` is compile-bound, not run-bound. The heavy files loop over heterogeneous
collections of forms, operators or meshes, written as a tuple or a literal array whose
element type is a `Union` of the concrete types. Inside the loop `form`, `assemble` and the
operator builders are inferred on that `Union`, so inference walks the whole assembly
pipeline for every combination of the union's members before the first case runs. The
files hold small meshes, so the numerical work is negligible against that compile.

`test/form/runtests.jl` already records the same cost on a file that was moved behind
`slow`: in `coefficient_shift.jl`, `for (nm, op) in (("D₋ₓ", D₋ₓ), ...)` binds `op` to a
`Union` of seven operator types, and each case costs 417 ms against a median of 65 to
95 ms per test elsewhere in the subsystem (11.7 s in all). `vector_calculus.jl` is the
second precedent there: about 159 s of its 160 s were compile, because helper
functions return a `Union` of subtree types. The
read-only analysis behind S3 (2026-10-07) found the same pattern in the six files that lead
the table above; it is the cause this page records, and the figures of the "After"
section test it.

## What S3 changes

In `matrix_free.jl`, `kronecker_edge.jl`, `kronecker.jl`, `kronecker_projection.jl`,
`linear.jl` and `bilinear.jl`, the case loops iterate an `Any[...]` vector instead of a
`Union`-typed tuple, so the loop body is no longer inferred over the union; each concrete
case still compiles the methods it uses once, as it runs. Every case is kept: the number of
passing tests in each file must not change, and nothing moves behind `slow`. Rejected
as not worth the coverage risk: running fewer of `matrix_free.jl`'s `VectorElement` checks,
and merging the Kronecker test tables.

## The shard boundary

S1 splits the unit group over three jobs selected by `BRAMBLE_TEST_SHARD`: `forms-1` runs
`dirichlet_constraints.jl` through `centered_vector_calculus.jl`, `forms-2` runs
`stencil_pattern.jl` through `compile_scaling.jl` (the `slow` files in that range stay
behind `slow`), and `rest` runs everything else in `Core library`. With the times of run
37341936090:

| Shard | Content | Time |
|:--|:--|--:|
| `forms-1` | 30 files | 551.6 s (9m12s) |
| `forms-2` | 11 files | 493.8 s (8m14s) |
| `rest` | all other testsets | 528.5 s (8m48s) |
| Sum | | 1573.9 s |

The two Forms halves add to 1045.3 s of the 1045.88 s the testset reports, and `Core
library` was 1574.4 s, so the 0.5 s that no file owns is the whole difference. Run
serially the three parts took 26m14s. Run in parallel the longest is `forms-1`, 9m12s, and
the Forms halves are 57.8 s apart and `rest` lies between them. Each shard job still pays the set-up
and, on a pull request that touches `src/`, the precompile of the first table (316 s) in
each shard. The measured figure for the sharded run belongs to "After".

## Nightly `slow` runs, ubuntu against macOS

The nightly `unit-and-slow` job runs on both platforms on the same commit. These are the
test-step times (`julia ... Pkg.test`) and they are the `slow` group, not `unit`:

| Run | Date | macOS | ubuntu |
|:--|:--|--:|--:|
| 37318618310 | 5 October 2026 | 40m35s | 36m29s |
| 37465825322 | 6 October 2026 | 40m55s | 41m47s |

Ubuntu was faster by 4m06s in the first run and slower by 52 s in the second, so these two
runs do not separate the platforms. The unit group on ubuntu is measured on this plan's
pull request (S2's temporary ubuntu leg, S5's decision).

## OpenEXR is version drift, not a cache bug

A warm cache did not stop `OpenEXR_jll` from being installed and precompiled again in two
of the three runs, and in the third other packages were. The logs:

| Run | Cache restored from | Install lines | Precompile |
|:--|:--|:--|:--|
| 37341936090 | nightly 37318618310 | `Installed OpenEXR_jll ─ v3.4.16+0`, `Installed artifact OpenEXR 891.9 KiB` | `20 dependencies successfully precompiled in 316 seconds. 564 already precompiled.` |
| 37335400202 | nightly 37318618310 | the same two lines | `2 dependencies successfully precompiled in 8 seconds. 582 already precompiled.` |
| 37592758673 | nightly 37465825322 | `Installed OpenSSH_jll ──────── v10.6.1+0`, `Installed CpuId ────────────── v0.3.2`, `Installed libpng_jll ───────── v1.6.59+1`, `Installed Adapt ────────────── v4.7.3`, `Installed KernelAbstractions ─ v0.9.44`, `Installed MaterialDocs ─────── v0.2.1`, `Installed artifact OpenSSH 770.2 KiB` | no summary line; the phase ran from 08:18:26 to 08:18:31 |

No `Manifest.toml` is committed (`.gitignore`, line 50), and `julia-actions/julia-buildpkg`
runs its own registry update ("Updating registry" in its step, besides `CI.yml`'s "Update any
cached registries" step). A pull request therefore resolves to whatever was released after
the nightly run saved the cache, and those packages and their dependents are installed and
precompiled again: 8 s for OpenEXR alone in run 37335400202. Pinning the versions would mean
committing a Manifest or resolving against a stale registry, which breaks the first pull
request after a compat bump, so the reason is recorded and not fixed.

## After

The "after" run is 37626129647, the first pull-request run of `CI.yml` on PR #485 (branch
`v3.35.0/ci-shards`, commit e5a1a939), on 7 October 2026. It ran the three shards on
`macos-latest` and, as S2's temporary measurement leg, on `ubuntu-latest`; all jobs passed.
The run took 14m45s from 13:08:03 to 13:22:48 UTC. Figures come from the job and step
timestamps (`gh run view 37626129647 --json jobs`) and from the `✓` trace lines of each
shard's log, as for the before run.

### Job times

| Job | Job time | Test step | Testset total |
|:--|--:|--:|--:|
| macOS `forms-1` | 14m03s (843 s) | 13m10s | `Forms` 740.66 s |
| macOS `forms-2` | 11m54s (714 s) | 10m57s | `Forms` 609.26 s |
| macOS `rest` | 14m18s (858 s) | 13m23s | `Core library` 12m20.1s |
| ubuntu `forms-1` | 13m49s (829 s) | 12m55s | `Forms` 762.78 s |
| ubuntu `forms-2` | 7m25s (445 s) | 6m42s | `Forms` 392.38 s |
| ubuntu `rest` | 12m59s (779 s) | 12m17s | not read |

The longest macOS job is `rest` at 14m18s, then `forms-1` at 14m03s. The before job took
32m43s, so the run's wall-clock fell by 17m58s (to 14m45s). That is not only the
effect of the split: the before run recompiled 20 dependencies (316 s), while these three
jobs had a warm cache. The three jobs also differ in how long the test step spent before the
first test: `Precompiling for configuration` (the test environment) printed at 13:09:16
and `Testing Running tests...` at 13:09:51 on `forms-1` (35 s), 13:09:17 to 13:09:49 on
`forms-2` (32 s) and 13:09:24 to 13:10:08 on `rest` (44 s). No `N dependencies successfully
precompiled` summary line was printed in the three macOS logs, so that figure is the
timestamp difference.

The Forms testset is 740.66 s plus 609.26 s on macOS, 1349.92 s, against 1045.88 s serial
before. The halves ran in parallel, so the longest is `forms-1` with 740.66 s (12m21s).

### Every Forms file, before and after

The before column is the table of "Every Forms file, ranked" (run 37341936090, macOS, serial
`Forms`). The after columns are runs of 37626129647: macOS from the `forms-1` and `forms-2`
trace lines, and ubuntu from its own two shards, for reference. The ratio is macOS after
over before.

| Rank | File | Shard | 37341936090 (s) | 37626129647 macOS (s) | Ratio | 37626129647 ubuntu (s) |
|--:|:--|:--|--:|--:|--:|--:|
| 1 | `matrix_free.jl` | forms-2 | 173.31 | 187.43 | 1.08 | 145.07 |
| 2 | `kronecker_edge.jl` | forms-2 | 114.74 | 131.81 | 1.15 | 87.56 |
| 3 | `linear.jl` | forms-1 | 107.55 | 196.37 | 1.83 | 221.53 |
| 4 | `kronecker.jl` | forms-2 | 96.35 | 124.68 | 1.29 | 74.21 |
| 5 | `bilinear.jl` | forms-1 | 94.41 | 94.41 | 1.00 | 133.28 |
| 6 | `kronecker_projection.jl` | forms-2 | 49.06 | 55.00 | 1.12 | 33.30 |
| 7 | `operators.jl` | forms-1 | 46.31 | 72.71 | 1.57 | 52.51 |
| 8 | `normal.jl` | forms-1 | 38.79 | 43.66 | 1.13 | 43.15 |
| 9 | `centered_vector_calculus.jl` | forms-1 | 37.80 | 47.01 | 1.24 | 35.45 |
| 10 | `nested_operators.jl` | forms-1 | 33.91 | 44.06 | 1.30 | 37.37 |
| 11 | `symmetry.jl` | forms-1 | 25.84 | 22.28 | 0.86 | 27.28 |
| 12 | `kronecker_block.jl` | forms-2 | 24.96 | 52.92 | 2.12 | 16.95 |
| 13 | `threaded_replay.jl` | forms-1 | 21.77 | 25.67 | 1.18 | 27.68 |
| 14 | `simplifier.jl` | forms-1 | 19.12 | 20.70 | 1.08 | 20.05 |
| 15 | `reaction_flux.jl` | forms-1 | 14.88 | 23.11 | 1.55 | 25.78 |
| 16 | `interpolation_operator.jl` | forms-1 | 13.93 | 16.70 | 1.20 | 17.61 |
| 17 | `coordinate_walk.jl` | forms-1 | 13.73 | 14.85 | 1.08 | 14.49 |
| 18 | `jacobian_pattern_blocks.jl` | forms-2 | 12.67 | 18.17 | 1.43 | 11.96 |
| 19 | `semidiscrete.jl` | forms-2 | 11.25 | 19.11 | 1.70 | 11.79 |
| 20 | `assemble_add.jl` | forms-1 | 10.89 | 12.18 | 1.12 | 12.92 |
| 21 | `dirac.jl` | forms-1 | 9.50 | 9.94 | 1.05 | 13.63 |
| 22 | `zero_form.jl` | forms-1 | 7.91 | 8.13 | 1.03 | 6.41 |
| 23 | `source_operators.jl` | forms-1 | 7.12 | 12.46 | 1.75 | 9.02 |
| 24 | `bandwidth.jl` | forms-2 | 6.47 | 11.60 | 1.79 | 6.01 |
| 25 | `inner_products.jl` | forms-1 | 5.91 | 9.97 | 1.69 | 7.32 |
| 26 | `skew.jl` | forms-1 | 5.12 | 5.69 | 1.11 | 4.89 |
| 27 | `dirichlet_constraints.jl` | forms-1 | 4.74 | 11.01 | 2.32 | 8.51 |
| 28 | `centered_average.jl` | forms-1 | 4.60 | 4.87 | 1.06 | 5.48 |
| 29 | `forwarddiff_smoke.jl` | forms-1 | 4.30 | 7.30 | 1.70 | 5.30 |
| 30 | `common.jl` | forms-1 | 3.85 | 4.66 | 1.21 | 4.16 |
| 31 | `difference_ast.jl` | forms-1 | 3.68 | 5.66 | 1.54 | 4.31 |
| 32 | `component.jl` | forms-1 | 3.21 | 3.81 | 1.19 | 3.48 |
| 33 | `block_extract.jl` | forms-1 | 3.04 | 3.83 | 1.26 | 3.65 |
| 34 | `interpolation.jl` | forms-1 | 2.79 | 3.61 | 1.29 | 3.49 |
| 35 | `stencil_pattern.jl` | forms-2 | 2.72 | 5.56 | 2.04 | 3.48 |
| 36 | `markers.jl` | forms-1 | 2.48 | 2.89 | 1.17 | 2.41 |
| 37 | `extended_operators.jl` | forms-1 | 2.44 | 4.35 | 1.78 | 3.46 |
| 38 | `type_cached_assemble.jl` | forms-2 | 1.53 | 1.73 | 1.13 | 1.22 |
| 39 | `symmetrize.jl` | forms-1 | 1.08 | 1.91 | 1.77 | 1.39 |
| 40 | `cross_mesh_blocks.jl` | forms-1 | 0.88 | 0.99 | 1.12 | 1.08 |
| 41 | `expression.jl` | forms-2 | 0.70 | 1.14 | 1.63 | 0.77 |

The `bilinear.jl` times are equal on macOS, 94.41 s in both runs. The two trace lines are
different lines (the memory column reads 2.27 GB before and 1.73 GB after), so this is a
coincidence of the printed figure.

The branch also added two files that run in `forms-1` and have no before figure:
`replay_sinks.jl` (3.05 s on macOS, 3.57 s on ubuntu) and `marker_ids.jl` (2.49 s, 1.89 s).
They are left out of the like-for-like totals below.

| Total over the 41 files | Before (s) | macOS after (s) | ubuntu after (s) |
|:--|--:|--:|--:|
| All 41 files | 1045.34 | 1343.94 (1.29x) | 1149.41 (1.10x) |
| `forms-1` files (30) | 551.58 | 734.79 (1.33x) | 757.09 (1.37x) |
| `forms-2` files (11) | 493.76 | 609.15 (1.23x) | 392.32 (0.79x) |
| The six S3 files | 635.42 | 789.70 (1.24x) | 694.95 (1.09x) |
| The other 35 files | 409.92 | 554.24 (1.35x) | 454.46 (1.11x) |

With the two new files the macOS sum is 1349.48 s and the ubuntu sum 1154.87 s. The Forms
total on macOS is higher after than before, by 298.6 s (28.6%) over the same 41 files. It
is not lower.

The first file of each shard is `dirichlet_constraints.jl` in `forms-1` and
`stencil_pattern.jl` in `forms-2`, the files that pay the compile of the shared code the
serial run had paid in earlier testsets. They took 11.01 s and 5.56 s against 4.74 s and
2.72 s before, so the extra is 6.3 s and 2.8 s. That is a small share of the 298.6 s: the
cold start of a shard does not explain the rise.

### Reading

The macOS Forms total rose. So did the parts this branch does not change: the 35 files S3
does not touch took 554.2 s against 409.9 s (1.35x), and the `rest` shard's `Core library`
took 12m20.1s (740.1 s) against 528.5 s for the same testsets before (1.40x). The noise
section records about 8% between two identical runs of the testset total, so a rise of 29%
is well outside it. The runs were on different branches and on different shared runners,
and the logs do not say which of these moved the times.

The six S3 files rose by 1.24x as a group, less than the 1.35x of the others. The only
normalisation the logs allow is the ratio of the six to the other 35 within one run: 1.55
before (635.4 against 409.9 s), 1.42 on macOS after and 1.53 on ubuntu after. By file on
macOS: `matrix_free.jl` 1.08x, `kronecker_projection.jl` 1.12x, `kronecker_edge.jl` 1.15x,
`kronecker.jl` 1.29x, `bilinear.jl` 1.00x and `linear.jl` 1.83x. `linear.jl` also is the
slowest file of `forms-1` on ubuntu (221.5 s), so its rise holds on both platforms, and
196.4 s is above the 85 to 139 s range of the three warm before runs.

So this run does not show that the `Any` loops cut the per-file time of the six files.
Four of them (`matrix_free.jl`, `kronecker_projection.jl`, `kronecker_edge.jl` and
`bilinear.jl`) rose by less than the 1.35x of the unchanged files, which could be a small
relative gain. A single pair of runs on different runners cannot support that, and
`linear.jl` went the other way. A claim needs a before and an after from the same branch
point and runner class, with the six files timed in both. This section does not make one.

What the run does show is the shape of the split. `forms-1` has 30 files and is the longer
Forms half on both platforms. `forms-2` took 609.2 s on macOS and 392.3 s on ubuntu, so the
macOS job times are set by `rest` and `forms-1`, 14m18s and 14m03s.
