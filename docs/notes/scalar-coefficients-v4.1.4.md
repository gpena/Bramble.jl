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
