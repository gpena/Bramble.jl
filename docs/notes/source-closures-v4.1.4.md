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

and `assemble`s it, each inside `@snoop_inference`, twice with the same `form` call. The
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
both rows). Counts agreed across the 3 launches. Times are medians over 3 launches.

Command, one `julia` with 1 thread per child, serial:

```
julia --startup-file=no benchmark/snoop_instances.jl \
    --rows closures_fresh,closures_named --launches 3 \
    --out benchmark/results/source_closures.toml
```

| quantity | value |
|---|---|
| commit | `77f7258c` on `perf/v4.1.4-measure` (the benchmark scripts and the S3 note) |
| Julia | 1.13.2, default `-O` |
| machine | Apple M2, 8 cores, 16 GiB, on AC power, battery 100% |
| threads | 1 per child |
| load | 1-minute load 3.55 at the start and 3.58 at the end (21:17 to 21:20) |

## Run under load

`gate_preflight.sh --timing` FAILED at 21:17. Load 4.33 on 8 cores (limit 2), and other
julia processes were running, two of them above 98% CPU (the concurrent v4.1.3-perf
plan). Those sessions were not paused or stopped. The run went ahead anyway, so every
time below is "under load", with a 1-minute load of 3.6 to 4.3 around the run (`uptime`
4.33, 4.25, 4.11 before; 3.55 at the first child and 3.58 after the last). The instance
counts do not depend on load. The times are one under-load figure each and say nothing
about a difference of a few percent.

## Figures

| row | fresh | added | sequence | form, s | assemble, s | inference, s |
|---|---:|---:|---:|---:|---:|---:|
| `closures_fresh` | 219 | 178 | 397 | 0.1283 | 0.1023 | 0.3421 |
| `closures_named` | 219 | 0 | 219 | 0.1422 | 0.1098 | 0.3751 |

The `fresh`, `form`, `assemble` and `inference` columns come from the FIRST build of the
fresh child, medians over 3 launches. Both rows run the same first build (three closures
of the same shape), so the 219 agree, and the time differences of 0.014 s in `form` and
0.007 s in `assemble` are load noise. `added` counts the instances the SECOND build infers
over the first. `sequence` counts the distinct instances of both builds together.

## The gap the prototype must close

| | second build, added instances |
|---|---:|
| three new closures (`closures_fresh`) | 178 |
| the same three named functions (`closures_named`) | 0 |

The gap is 178 instances, 81% of the 219 of the first build. A reused function costs the
second build nothing, since `form`, the AST walks and `assemble` find every instance
compiled for the first. Three new closures make new types. The second build then infers
178 instances again, and a user who writes the closures anew at each call pays that each
time.

The script does not time the second build. `form_s` and `assemble_s` are the first
build's, and the second build's `form` time, the quantity #888 reports, is not in the
TOML. The `added` count is `form` and `assemble` summed: it does not say how many of the
178 are `form` and how many are `assemble`. The nearest recorded figures are the first
build's, for the same code on new types: `form` 0.128 s and 0.142 s, `assemble` 0.102 s
and 0.110 s, `inference` 0.342 s and 0.375 s. The `inference` times exceed the wall times
of `form` plus `assemble` (0.231 s and 0.252 s), so they are not a share of them, and this
run does not explain why. The first build is also the one that finds part of the image
cold, so its time cannot be scaled to the second build's 178 instances without guessing.

## Against #888's figure

#888 measured, on main 7dbaa523 under load, the compile time of a 3-term linear form:

| #888 row | form, s | assemble, s |
|---|---:|---:|
| first closures | 0.13 | 0.15 |
| same structure, new closures | 0.17 | 0.00 |
| same structure, reused named functions | 0.00 | 0.00 |

Its headline is 0.17 s of `form` for new closures against 0.00 s for reused functions.
The two measurements differ in method:

- Unit: #888 gives compile seconds of the second build, `form` and `assemble` apart. This
  run counts the method instances the second build infers, `form` and `assemble` summed,
  and does not time the second build, so no second-build seconds can be set beside 0.17 s.
- Reused functions agree: 0 instances here, 0.00 s there.
- New closures: 178 instances here. #888 puts the whole cost in `form` (0.17 s) and none
  in `assemble` (0.00 s), as #197 intends. This run cannot say whether the 178 are all
  `form`: it sums the two.
- The first build is a cross-check of order only: #888's first closures cost `form` 0.13 s
  and `assemble` 0.15 s, here 0.128 s and 0.102 s (0.142 s and 0.110 s with named
  functions, the same first build).
- Mesh, process and load: here a fresh process per launch and a graded 2D mesh, as #917
  step 1 asks. #888 does not state its mesh or process, and its load is not recorded.

For #917, the criterion for the prototype is the second build's count, 178 against 0,
taken in the same script and run as the prototype's. The prototype closes the gap when
`closures_fresh` adds close to 0, as `closures_named` does.

## To retake

The times are to be retaken on main after v4.1.3-perf merges, on a machine that passes
`gate_preflight.sh --timing`: #895 and #906 change the first-call cost of `form` and
`assemble`, and this run was under load. The instance counts should be retaken with them,
since those changes may alter them too.

## What #917's step 2 needs

- The target: the second build of `closures_fresh` adds 178 instances today and
  `closures_named` adds 0.
- Not measured here: the second build's `form` time on its own (the script times the
  first build only), the split of the 178 between `form` and `assemble`, warm `assemble`
  on a non-uniform mesh, and any `Dual`-valued source.
