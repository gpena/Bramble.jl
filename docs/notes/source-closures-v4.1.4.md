# What a fresh source closure costs the second build, against a reused one (v4.1.4, S4)

Answers #917 step 1: how many method instances of Bramble a second build of a 3-term
linear form adds when its three source coefficients are new anonymous closures, against
the same three named functions again. The gap between the two is what the prototype of
#917 (an erased source function in the AST) has to close. This note records figures. It
makes no decision on the prototype, which is the user's, in the issue.

## Method and conditions

`benchmark/snoop_instances.jl` starts a fresh `julia` per launch. The child loads Bramble
and SnoopCompile, builds a graded 2D mesh and its `gridspace`, then builds the form

```julia
form(W, v -> innerₕ(a, v) + innerₕ(b, v) + innerₕ(c, v))
```

and `assemble`s it, each inside `@snoop_inference`, twice with the same `form` call. It times
both builds. The
three sources are those of `test/form/linear.jl`: `x -> x[1]`, `x -> 100 * x[1]` and
`x -> x[2]`. The count is the number of distinct `MethodInstance`s of Bramble or its
submodules in the flattened inference tree, root excluded, summed over `form` and
`assemble`.

- `closures_fresh`: the first build takes the closures of `anonymous_a()`, the second
  those of `anonymous_b()`. Both write the same three lambdas, so each call returns three
  new anonymous types.
- `closures_named`: both builds take `named_1`, `named_2`, `named_3`, three functions
  defined once, so the second build has the same types as the first.

A third build of the second set, in the same process, must infer 0 instances (it did, for
both rows). Counts agreed across the 3 launches. Times are medians over 3 launches. The
`julia` in the command below ran through `julia_slot.sh 1`, one child at a time.

Command, one `julia` with 1 thread per child, serial:

```
julia --startup-file=no benchmark/snoop_instances.jl \
    --rows closures_fresh,closures_named --launches 3 \
    --out benchmark/results/source_closures.toml
```

| quantity | value |
|---|---|
| commit | `82882e01` on `perf/v4.1.4-measure` (the benchmark scripts, which now time the second build) |
| Julia | 1.13.2, default `-O` |
| machine | Apple M2, 8 cores, 16 GiB, on AC power, battery 100% |
| threads | 1 per child |
| load | 1-minute load 2.55 at the start and 2.43 at the end (21:26, under a minute) |

## Run under load

`gate_preflight.sh --timing` FAILED at 21:26 on one check: load 2.60 on 8 cores, limit 2.
Power, OneDrive, other julia processes (none) and CPU (nothing above 50%) passed. Other
sessions were not paused or stopped. The run went ahead anyway, so every time below is
"under load", with a 1-minute load of 2.4 to 2.6 around the run (`uptime` 2.60, 2.87, 3.43
at the preflight; 2.55, 2.85, 3.42 before the first child; 2.43, 2.81, 3.40 after the
last). The instance counts do not depend on load. The times are one under-load figure each
and say nothing about a difference of a few percent.

## Figures

First build, in a fresh child:

| row | fresh | form, s | assemble, s | inference, s |
|---|---:|---:|---:|---:|
| `closures_fresh` | 219 | 0.128 | 0.105 | 0.344 |
| `closures_named` | 219 | 0.126 | 0.100 | 0.337 |

Second build, same child, same `form` call with the second set of sources:

| row | added | sequence | form, s | assemble, s | inference, s |
|---|---:|---:|---:|---:|---:|
| `closures_fresh` | 178 | 397 | 0.110 | 0.000 | 0.158 |
| `closures_named` | 0 | 219 | 0.005 | 0.000 | 0.004 |
| gap (fresh minus named) | 178 | 178 | 0.105 | 0.000 | 0.153 |

Both rows run the same first build (three closures of the same shape), so the 219 agree,
and the differences of 0.002 s in `form` and 0.005 s in `assemble` are load noise. `added`
counts the instances the second build infers over the first. `sequence` counts the
distinct instances of both builds together. `assemble` of the second build is 2 us for
`closures_fresh` and 1 us for `closures_named`, which print as 0.000 at 3 decimals.

## The gap the prototype must close

The gap is 178 instances, 81% of the 219 of the first build, and 0.105 s of `form` (0.110 s
against 0.005 s), 0.153 s of inference (0.158 s against 0.004 s) and no `assemble` time.
A reused function costs the second build nothing, since `form`, the AST walks and
`assemble` find every instance compiled for the first. The 0.005 s that `closures_named`
still spends in `form` is the call itself, with 0 instances inferred. Three new closures
make new types. The second build then infers 178 instances again, and a user who writes
the closures anew at each call pays that each time.

The `added` count is `form` and `assemble` summed: it does not say how many of the 178
are `form` and how many are `assemble`. The times do: the second build's `assemble` takes
2 us in both rows, so the time of the gap sits in `form`.

The `inference` times exceed the wall times of `form` plus `assemble` in the first build
(0.233 s against 0.344 s for `closures_fresh`, 0.226 s against 0.337 s for
`closures_named`) and, for `closures_fresh`, in the second (0.110 s against 0.158 s). They
are not a share of those times, and this run does not explain why. The integrator checked
that the exclusive time of the inference root is about 2e-6 s, so `inference_s` is the
inference time of the tree and not the whole snoop window. The excess is left unexplained.

## Against #888's figure

#888 measured, on main 7dbaa523 under load, the compile time of a 3-term linear form:

| #888 row | form, s | assemble, s |
|---|---:|---:|
| first closures | 0.13 | 0.15 |
| same structure, new closures | 0.17 | 0.00 |
| same structure, reused named functions | 0.00 | 0.00 |

Its headline is 0.17 s of `form` for new closures against 0.00 s for reused functions.
This run gives the second build in seconds as well, `form` and `assemble` apart:

| second build | #888 form, s | here form, s | #888 assemble, s | here assemble, s |
|---|---:|---:|---:|---:|
| new closures | 0.17 | 0.110 | 0.00 | 0.000 |
| reused named functions | 0.00 | 0.005 | 0.00 | 0.000 |
| gap | 0.17 | 0.105 | 0.00 | 0.000 |

- Where the cost sits: in `form`, as #888 found. The second build's `assemble` takes
  2 us for new closures and 1 us for reused functions, 0.00 s at #888's two decimals. The
  earlier version of this note could not tell, since it had no second-build time.
- Size: here 0.105 s of gap in `form` against #888's 0.17 s. The runs differ in machine
  load, in the main commit, and in what #888 does not state (its mesh and process), so
  this note does not say whether 0.105 s and 0.17 s disagree.
- Reused functions agree: 0 instances and 0.005 s of `form` here, 0.00 s there.
- Instances: 178 for new closures here. #888 counts seconds only.
- The first build is a cross-check of order: #888's first closures cost `form` 0.13 s and
  `assemble` 0.15 s, here 0.128 s and 0.105 s (0.126 s and 0.100 s with named functions,
  the same first build).
- Mesh, process and load: here a fresh process per launch and a graded 2D mesh, as #917
  step 1 asks, at a 1-minute load of 2.4 to 2.6. #888's load is not recorded.

For #917, the criterion for the prototype is the second build's count, 178 against 0, and
its `form` time, 0.110 s against 0.005 s, taken in the same script and run as the
prototype's. The prototype closes the gap when `closures_fresh` adds close to 0, as
`closures_named` does.

## To retake

The times are to be retaken on main after v4.1.3-perf merges, on a machine that passes
`gate_preflight.sh --timing`: #895 and #906 change the first-call cost of `form` and
`assemble`, and this run was under load (preflight failed on load 2.60). The instance
counts should be retaken with them, since those changes may alter them too.

## What #917's step 2 needs

- The target: the second build of `closures_fresh` adds 178 instances and spends 0.110 s
  in `form` today, and `closures_named` adds 0 and spends 0.005 s (under load).
- Not recorded: the split of the 178 instances between `form` and `assemble` (the times
  put the gap in `form`), warm `assemble` on a non-uniform mesh, and any `Dual`-valued
  source.
