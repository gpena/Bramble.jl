# What a fresh source closure costs the second build, against a reused one (v4.1.4, S4)

Answers #917 step 1: how many method instances of Bramble a second build of a 3-term
linear form adds when its three source coefficients are new anonymous closures, against
the same three named functions again. The gap between the two is what the prototype of
#917 (an erased source function in the AST) has to close. The prototype's decision, by
S6.2's fixed rule, is in "Prototype measured" at the end.

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

## Prototype measured

Question: does S6.1, the erased source function of #917, cut what a second build of
`closures_fresh` costs enough to keep? The rule was fixed before the run (plan S6.2, with
the user's ruling of 2026-10-27 on warm times): ship if head's second-build instances are
at most half of base's, its second-build `form` median is below base's by more than both
trees' launch spreads, and warm `assemble` is within base's spread or +3%, whichever is
larger, on rows whose base warm time is 50 us or more. A shorter row's warm time is
reported, not gated.

### Method

- Head: `fa760198` (S6.1 shipped; it also holds S3 and S5.1). Base: the same commit with
  S6.1's five commits (`fda62e7f`, `ef26801f`, `2f035064`, `9b80cec9`, `fa760198`)
  reverted in a detached tree, since S3's and S6.1's commits interleave. The script
  records `fa760198` for both trees, the base's revert being uncommitted.
- `benchmark/snoop_instances.jl` (the same file in both trees), rows `closures_fresh` and
  `closures_named`, 4 threads per child, default `-O`, three launches interleaved: base,
  head, base, head, base, head, each one `--launches 1` run through `julia_slot.sh 4`.
- Julia 1.13.2, Apple M2, 8 cores, on AC, OneDrive not running, no other `julia`.
  `check_power_load.sh` passed, and `gate_preflight.sh --timing` passed at a 1-minute
  load of 1.72. The load was 1.58 to 1.91 over the run, which took 36 s.
- Raw launches, medians and spreads (max minus min over the three launches) are in
  `benchmark/results/v4.1.4_source_closures_ab.toml`.

### Erased-class check

S6.1 erases a source only when it returns its point's precision, its test side is one leaf
and it has no `markers`. Before timing, a one-thread probe built `closures_fresh`'s AST,
`simplify_ast(resolve_ast(innerₕ(a, t) + innerₕ(b, t) + innerₕ(c, t)))` on the graded 9x11
space, from `anonymous_a()` and from `anonymous_b()`. On head all three leaves are
`SourceFunction{2, Function}` and the two ASTs have one type. On base no leaf is erased,
each names its closure, and the two types differ. So the row measures the change.

### Figures

Second build (the new closures of `anonymous_b()`, or the same named functions again):

| row | tree | instances | `form`, s (spread) | inference, s (spread) |
|---|---|---:|---:|---:|
| `closures_fresh` | base | 177 | 0.1669 (0.0005) | 0.2169 (0.0009) |
| `closures_fresh` | head | 70 | 0.0528 (0.0021) | 0.0584 (0.0025) |
| `closures_named` | base | 0 | 0.0047 (0.0002) | 0.0044 (0.0002) |
| `closures_named` | head | 0 | 0.0050 (0.0004) | 0.0045 (0.0003) |

The instance counts were the same in all three launches on each tree. The first build
also falls, 216 instances to 140 on both rows. Its `form` time falls from 0.115 s to
0.075 s on `closures_fresh`, and from 0.112 s to 0.072 s on `closures_named`.

Warm `assemble!`, median of 20 calls, median over launches:

| row | base, s (spread) | head, s (spread) |
|---|---:|---:|
| `closures_fresh` | 1.88e-7 (4.1e-8) | 1.88e-7 (2.1e-8) |
| `closures_named` | 1.67e-7 (2.1e-8) | 1.67e-7 (4.1e-8) |

Both are far below 50 us, so warm time is reported and not gated. It does not move.

The rule, clause by clause:

- Instances: 70 against 177, at most 88.5. Holds.
- Second-build `form`: 0.0528 s against 0.1669 s, a gap of 0.114 s, larger than both
  spreads (0.0005 s and 0.0021 s). Holds.
- Warm `assemble`: no row is gated (base 0.19 us and 0.17 us). Holds.

**Verdict: S6.1 shipped.**

The 70 instances left (69 in a direct count of the second build's flattened tree on head)
are per closure, three of each: `innerₕ`, `_as_source`, `_erased_source`,
`_keeps_precision`, `_returns_own` (Float64 and Float32), `SourceFunction`, and the
sampling of each closure in `_lower_sources` (`Rₕ!`, `_RₕKernel`, `project!`,
`_sweep_for!`, `_device_project!`, `PointValue`). What is gone is the instances of every
pass over the AST and of `assemble`, which no longer see the closure's type. The rest
grows with the number of closures, not with the form, and the closure must still be
compiled to be sampled. The gap to `closures_named` is not closed: 70 instances and
0.048 s of `form` remain.

### Inference of inline-closure forms

`isconcretetype(only(Base.return_types(f, (typeof(W),))))` with the closures built inside
`f`, on the graded 9x11 2D scalar space:

| form | base | head |
|---|---|---|
| `captured_linear_form`: `a = x -> x[1]^2 + x[2]; form(W, v -> innerₕ(a, v))` | concrete | concrete |
| `restrict_outer_linear_form`: `form(W, v -> restrict_to(:interior, innerₕ(x -> x[1]^2, v)))` | concrete | **not concrete** |
| `form(W, v -> innerₕ(x -> x[1]^2, v))` | concrete | concrete |

The plan wrote these with `x -> x^2`. On a 2D point, which is a tuple, that closure has no
`Float64` result, so it is never erased and is no test of the change. As written, the outer
`restrict_to` form is concrete on both trees, and the plain form throws a `MethodError` on
both and so is not concrete. The table uses `x -> x[1]^2`, which head erases and base
does not (checked, as is the captured `a`). The
outer-`restrict_to` loss is the one S6.1's critic found and the user accepted if the rule
says ship. The assembled values and the 0 B warm assembly do not change (S6.1's critic).

### Scope

The erasure is conditional, so the benefit covers only a source closure that returns its
point's precision (`Float64` at a `Float64` point, `Float32` at a `Float32` point), with
its test side on one leaf and no `markers`. A closure that returns a constant such as
`x -> 1.0`, a `Dual`, a complex, or a `Union`, or one under `markers`, keeps its type and
still costs a second build what base does. Under an outer `restrict_to`, a restricted sum
or a composite term, an erased source got its type back at `form`, and that `form` no
longer inferred a concrete type; the next section samples it instead, on a host space.

## S6.3: an outer restriction keeps the form concrete

The `restrict_outer_linear_form` row of the inference table above now reads concrete on
head as well. The loss came from the order of two decisions: `innerₕ` erases the source when
it builds the term, and an outer `restrict_to` wraps that term afterwards. `_lower_sources`
does not sample a source under a restriction, because sampling it everywhere would call it
at points no assembly reads (#643, #524), so `form` gave it its type back
(`_specialised_sources`) and inferred a `Union`.

`form` now samples such a source where the walk reads it. `_typed_restrictions` already
carries `reach`, the points at which the walk evaluates each node, narrowed by every
enclosing restriction and moved by every enclosing shift. On a host space it fills the
source's values at those points, behind the same `invokelatest` barrier as an unrestricted
erased source, and stores them as a `SourceVector` over a `_ReachSamples` array, whose type
names no closure. The walk reads that array, so the form infers a concrete type and a warm
`assemble!` stays at 0 B. A point outside `reach` (a region that `markers!` moved after
`form`) calls the source when it is read, through the barrier, and allocates there. A
shift, a difference, an average or a jump over the restricted term (directly, or through
scales and sums) reads that term at each of its points and at every tap's clamped
neighbour, inside the region or not, as it reads a typed source, so `reach` there takes all
of those points and the restriction narrows none (`_typed_taps`). The same holds with no
restriction between the operator and the term (`restrict_to(:A, D₊ₓ(innerₕ(f, v)))`): a
tap's neighbour can lie outside `:A`, and a shift at the mesh's edge reads the clamped
point, so over an erased source `reach` takes them too. Form-time memory is one value vector
per sampled source, its `reach` mask shared, with no index vector.

The one-leaf composite case lowered through `_unrouted_term`: a term naming no component on
a composite space with one leaf lowered against that leaf, as a scalar space's term does,
so a typed source there was sampled at `form` too, where main read it at each assembly
(the next section gives main's handling back).

Shapes checked against a typed `source_function` of the same closure (main's path), bit
for bit, on the graded 9x11 mesh in `Float64` and `Float32`: an outer restriction, a
restricted sum, nested restrictions, a shift inside and outside the restriction, an
average over a scaled restriction, an empty region, a one-leaf composite term naming no component (red on head before this change,
with a closure capturing an `Int`), and a restricted `v(2)` on a two-leaf composite space.
`markers` and a `Dual` source are not erased, as before.

What changed for the user at S6.3: a restricted plain function was read at `form` only,
as an unrestricted one already was, so a closure that reads mutable state, or a mesh moved
by `change_points!`, was not seen again at its sampled points. The user rejected that, and
the next section samples it again at every assembly. A device space, and a term
naming no component on a composite space with several leaves, still give the source its
type back at `form` and do not infer a concrete type. The instance figures are the
integrator's to take.

## S6.4: a sampled source is read live at every assembly

The user rejected S6.3's sampling once at `form`. Main calls a restricted source, and a
typed source in a one-leaf composite term naming no component, at every assembly, so a
closure that reads a `Ref` or a moved mesh sees the new state. This change keeps S6.3's
`_ReachSamples` and the concrete `form`, and samples the source again at the `reach` points
at the start of every host assembly, in one serial pass before the sweep
(`_refresh_reach_samples!`, assembly/linear.jl). The pass goes through the barrier S6.3
built (`_sample_reach!`, `invokelatest` into `_sample_reach_typed!`), so it compiles no new
method instance per closure: that instance is the one `form` already compiled. Whether a
form holds a `_ReachSamples` is read off its type (`_holds_reach_samples`, generated, as
`_holds_erased`), so a form without one compiles no walk, and the pass is a no-op there.
A warm `assemble!`, `assemble_add!` and `assemble_add!` with `α` stay at 0 B on the
restricted forms (`restricted_bytes` in test/form/linear_restriction.jl), as on fd7f13aa.

The pass runs once per call, before any threaded sweep, at the four host entry points that
sweep a linear form's AST: `_assemble_linear!` (behind `assemble`, `assemble!` and
`evaluate!`), `_assemble_linear_parallel!` (behind `assemble_parallel!`),
`_assemble_add_linear!` (behind `assemble_add!` with and without `α`), and
`_contract_linear` (behind `l(vₕ)`). A device space reaches the host walk through
`_assemble_linear_device!`, but its AST holds no `_ReachSamples`: device spaces keep
`_specialised_sources`. Nothing in `src/problems/` or `ext/` sweeps a `LinearForm`'s AST
directly; the ghost forms call `assemble!` on their padded form.

A point outside `reach` (a region `markers!` widened after `form`) is still called when the
walk reads it, through `_late_sample`, so every value a walk reads is the closure's at that
assembly: the sampled and the late values no longer mix states.

`_unrouted_term` gives main's handling back on a one-leaf composite: a term naming no
component keeps a typed source (`SourceFunction{D, F}`, `F` concrete) as main left it, and
the walk calls it. Only an erased source there goes through `_ReachSamples`, over every
point of the one leaf, narrowed and widened by any restriction, shift or tap in the term as
on a scalar space. Unrestricted erased sources that `_lower_sources` samples into a plain
`SourceVector` are left alone: main sampled those once at `form` too.

Tests: the `Ref` liveness shapes (an outer restriction, a restricted sum, a `D₊ₓ` and a
shift under a restriction, a one-leaf composite term with an erased and with a typed
source, a restricted one-leaf composite term) through `assemble`, `assemble!`,
`assemble_parallel!`, `assemble_add!` with and without `α` and `l(vₕ)`, on `Serial` and
`Parallel` backends, and a `markers!` case (test/form/source_operators.jl,
`outer restriction: live per assembly`); the S6.3 call-set test now expects the
`reach` points called at every assembly.

Hazards:

- Two tasks assembling one form at once race on the `_ReachSamples` values: each pass
  writes the same vectors the other task's sweep reads. There are precedents: a
  `GhostLinearForm`'s `F_pad` (assembly/ghost_assembly.jl), the padded load vector every
  assembly of that linear form fills, and the bilinear replay cache (`a.cache`, read and
  re-recorded by `_assemble_bilinear_core_cached!` behind `assemble!` and `assemble_add!`,
  assembly/assemble_add.jl). Both are per-form scratch an assembly mutates, so one such form
  is not safe to assemble from two tasks at once either. A threaded sweep inside one call is
  safe: the pass ends before it starts.
- The closure is still called once at `form` at the `reach` points, where main did not call
  a restricted source at `form`, only at assembly.
- Open, for the integrator: after `markers!` shrinks a region, the pass still calls the
  source at the points the region left, because `reach` is fixed at `form`. On a 9-point 1D
  mesh with `:half` shrunk from `x > 0.5` to `x > 0.7` after `form`, a warm `assemble!` calls
  a typed source (main's path) at 0.75, 0.875 and 1.0, and the erased one at 0.625 as well;
  the vectors agree bit for bit, since the walk does not read the value at 0.625. A source
  undefined at the dropped points (#643) throws at assembly where main would not call it.
  S6.3 did not call it there: it only read the stored value, and the walk skipped it.
  Without a `markers!` shrink after `form`, an assembly calls the source only at the points
  its walk reads (`reach`).

The instance figures are the integrator's to take.
