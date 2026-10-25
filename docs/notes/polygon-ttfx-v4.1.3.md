# First polygon `assemble(a, l)`: compile cost and the workload (v4.1.3, S3.3, #883)

Answers #883: the first `mesh` + `polygon_spaces` + `form` + `assemble(a, l)` on Figure 1
`(40, 36)` compiles 2.3 times what a rectangle of similar size does. `_pc_polygon` ran
the polygon path but not the form users write. Extending it brings the first call to 1.3 times
the rectangle's.

All times in this note were taken on battery power and under load (see `## Conditions`).
The ratios are back to back in one session; do not compare absolute times with other notes.

## Result

First call of the whole pipeline, serial (default policy), Figure 1 `(40, 36)` against a
rectangle with 6561 DOFs (the polygon has 6617). Median of 3 fresh processes, each case first
in its own process, `@timed` summed over the five phases, seconds. Three builds of Bramble
(same tree, same Manifest), one fresh depot each:

| build | polygon | rectangle | polygon / rectangle |
|---|---:|---:|---:|
| before: `_pc_polygon(Float64)` and `(Float32)` as at `e4300c13` | 1.314 | 0.581 | 2.26 |
| after: the same, plus the Laplacian-plus-convection pair assembled in `_pc_polygon` | 0.645 | 0.496 | 1.30 |
| without the Float64 call (the S3.2 proposal) | 2.000 | 0.575 | 3.48 |

A second batch of the same builds, run before this one with the threaded rows between
(`## Secondary rows`), gives 1.305 and 0.648 for polygon, 0.579 and 0.470 for the rectangle:
2.25 before, 1.38 after. Both are within the 1.5 bound; against the rectangle of the unchanged
build (0.581) the after ratio is 1.11.

Cost and gain of the workload, `using Bramble` in a fresh depot (4 threads), median of three
builds for the first two rows and of two for the third:

| build | core precompile, s | Bramble pkgimage, KiB | first polygon call, s |
|---|---:|---:|---:|
| without the Float64 call | 101.15 | 262158 | 2.000 |
| before (Float64 and Float32 calls, mixed form only) | 102.68 | 267036 | 1.314 |
| after (adds the convection pair, both precisions) | 104.53 | 269000 | 0.645 |

- The Float64 call that exists today costs about 1.5 s and 4.8 MiB of pkgimage and saves
  0.69 s on this pipeline (2.000 to 1.314). S3.2 measured 0.18 s saved, on the workload's own
  form at `(10, 9)`; the user's pipeline shares the mesh and part of the space code with it,
  so the call is worth more than S3.2 found. Dropping it is not supported by this
  measurement.
- Extending it costs a further 1.9 s and 1.9 MiB and saves another 0.67 s (1.314 to 0.645).
  Float32 compiles the same pair, so the cost covers both precisions.

Decision: keep `_pc_polygon(Float64)` and extend `_pc_polygon` (it is generic in `T`) with the
form `innerₕ(-Δₕ(w), v) + innerₕ(0.5 Dcₓ(w), v)` and its `assemble(b, l)`. The pkgimage cost
of the extension, 1.9 MiB, stands beside its 0.67 s gain.

## Method instances

The method instances are counted the way `benchmark/snoop_instances.jl` does: a fresh process per case, each phase inside
`@snoop_inference`, the Bramble `MethodInstance`s of `flatten(tinf)` (root excluded) counted
per phase, `ALL` the distinct union. The scratch child is `first.jl` under
`~/bramble-scratch/v413perf/s33/`. Counts agree between the two launches of the before and after builds (one launch of the third).

| build, case | mesh | spaces | form `a` | form `l` | `assemble` | all |
|---|---:|---:|---:|---:|---:|---:|
| before, polygon | 0 | 398 | 185 | 354 | 57 | 994 |
| after, polygon | 0 | 398 | 51 | 354 | 0 | 803 |
| without Float64, polygon | 76 | 402 | 185 | 354 | 57 | 1074 |
| before, rectangle | 0 | 79 | 65 | 350 | 110 | 604 |
| after, rectangle | 0 | 79 | 20 | 350 | 94 | 543 |
| without Float64, rectangle | 0 | 79 | 65 | 350 | 110 | 604 |

The first polygon `assemble` itself infers 57 instances before and 0 after (the rectangle's:
110 and 94). Phase times, same cases, seconds (from the timing batch, not the snoop one, which
inflates them):

| phase | polygon before | polygon after | rectangle before | rectangle after |
|---|---:|---:|---:|---:|
| `mesh` | 0.016 | 0.016 | 0.017 | 0.017 |
| `polygon_spaces` or `gridspace` | 0.499 | 0.498 | 0.027 | 0.027 |
| `form(a)` | 0.383 | 0.030 | 0.081 | 0.012 |
| `form(l)` | 0.132 | 0.094 | 0.120 | 0.114 |
| `assemble(a, l)` | 0.279 | 0.008 | 0.339 | 0.323 |

What remains of the 0.645 s after the extension is `polygon_spaces` (0.498 s, 398 instances)
and `form(l)`. Both are called with a fresh anonymous function (`x -> 0.0`, `v -> innerₕ(x ->
1.0, v)`), whose type no workload can name. An emulation bounds what a perfect workload could
reach: the same pipeline run once at `(10, 9)` in the process, then timed at `(40, 36)` with
other closures of the same bodies, takes 0.229 s (spaces 0.160, `form(l)` 0.055). So about 0.4 s
of `polygon_spaces` is code the workload already runs, which a fresh process still compiles. The
cause is not found; it is the next lead for #883 if the 1.3 ratio is to fall.

## Secondary rows

The same pipeline with `mesh(...; backend)` under `CpuThreaded` and `CpuPolyester` (4 threads),
and the rectangle under the same policy. Median of 3 fresh processes per row, seconds, one
batch with the serial rows of the first batch above:

| policy | polygon before | polygon after | rectangle before | rectangle after | after ratio |
|---|---:|---:|---:|---:|---:|
| `CpuThreaded` | 3.344 | 3.250 | 1.310 | 1.223 | 2.66 |
| `CpuPolyester` | 5.632 | 5.468 | 2.108 | 2.004 | 2.73 |

Neither policy is touched by the workload: the precompile runs the default policy only, and
the cut walks added in S2.3 and S2.3.1 compile per form type under each policy (the `assemble`
phase is 1.09 s and 3.21 s of those rows after the change). The first call under a threaded
policy stays about 2.7 times the rectangle's. `CpuPolyester` needs the Polyester extension, so
a core workload cannot cover it; that cost is the one S2.3.1 accepted, recorded for S5.1.

## What the workload does not cover

It covers the pipeline in this note, a form with a Laplacian and a convection term in the
default policy, in Float64 and Float32. A form with other operators compiles its own `form`
and `assemble` (the 0.38 s and 0.28 s of the before column are the order of that cost), and a
user closure is always new. The 1.3 ratio holds for this form spelling only.

## Conditions

- Julia 1.13.2, Apple Silicon (8 logical cores), default `-O`, 4 threads (every `julia` through
  `julia_slot.sh 4`), tree `e4300c13` of `v4.1.3/perf`, nothing else of mine running.
- On battery power throughout (`pmset`: "Battery Power", 100%). The user said no AC was
  available. `gate_preflight.sh --timing` failed at 11:12 on power and on load (1-minute load
  4.89, the end of a build). During the timing batches the load average read 2.1 to 3.5,
  above the preflight's limit of 2, so every time here is "on battery, under load". The
  before and after runs alternated within each batch, so the ratios share the conditions.
- Builds: a fresh depot per build (`JULIA_DEPOT_PATH=<depot>:~/.julia`, `JULIA_PKG_OFFLINE=true`),
  `using Bramble` under `time -p`, the pkgimage read with `du -sk` from
  `<depot>/compiled/v1.13/Bramble`. All builds, seconds and KiB, in the order run: before
  110.66 (267088), 102.53 (266860), 102.68 (267036); after 114.58 (269048), 104.53 (269000),
  104.12 (268940); without Float64 101.21 (262168), 101.08 (262148). The first round of before
  and after ran at load 3 to 4.5 and is the slow outlier of each; the medians above do not
  use it.
- Seconds of a build are about 1 s noisy between builds, so the 1.9 s of the extension and the
  1.5 s of the Float64 call are two to three times the noise at best. The pkgimage figures
  agree to 0.2 MiB between builds.
- The "without" tree removes only the `_pc_polygon(Float64)` line of `src/precompile.jl`; the
  "after" tree is the committed change (`src/precompile/space_sessions.jl`, plus a comment in
  `src/precompile.jl`).
- Form: `innerₕ(-1.0 * Δₕ(u), v) + innerₕ(0.5 * Dcₓ(u), v)`, source `innerₕ(x -> 1.0, v)`,
  Figure 1 on `interval(0, 6) × interval(0, 5)` with `Random.seed!(2)` and the non-uniform
  `mesh(Ω, (40, 36), (false, false))`; the rectangle is `(81, 81)` on the unit square, same
  form and seed.
