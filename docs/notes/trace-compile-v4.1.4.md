# Where the first call of a form compiles, by function (v4.1.4, S2)

Answers #914: how the compilation time of the first `form` and `assemble` call splits
over functions, for the three rows of `benchmark/multiterm_ttfx.jl`. This note records the
ranking only. It draws no conclusion and proposes no change.

## Method and conditions

`benchmark/trace_compile.jl` (added in S1) starts a fresh `julia` per launch with
`--trace-compile=stderr --trace-compile-timing`. The child loads Bramble, builds a graded
mesh and space, then calls `form` once and `assemble` once, with `#PHASE` markers on
stderr between the calls. The parent ranks the trace lines of the `form` and `assemble`
phases only and groups them by the function of the first argument of each `Tuple{...}`
signature. The rows are `elasticity` and `vector_laplacian` on a graded 5x5x5 mesh with
three components, and `scalar_2d` on a graded 9x9 mesh. Times are medians over launches.
The script stops when a fresh launch changes the number of instances or recompilations of
a function, and it did not stop.

Command, run alone on the machine:

```
julia --startup-file=no benchmark/trace_compile.jl --launches 3 --out benchmark/results/trace_compile.toml
```

| quantity | value |
|---|---|
| branch | `perf/914-trace-compile` at `7c015691` (only the script and `.gitignore` added) |
| main | `9c34251c` (origin/main, includes #911); `src/`, `ext/` and `Project.toml` of the branch equal main's |
| Julia | 1.13.2, default `-O` (the child refuses any other level) |
| machine | Apple M2, 16 GiB (`hw.memsize` 17179869184), on AC power, battery 100% |
| threads | 1 per child (`--threads 1`), 4 slots held for the run |
| launches | 3 per row, interleaved by round |
| load | 1-minute load 1.92 at the start (20:07) and 2.49 at the end (20:08); 5-minute 2.50 then 2.55, 15-minute 3.04 then 3.00 |

`gate_preflight.sh --timing` passed at 20:06 (load 1.92) and the other Julia session was
paused for the run, so this is a run alone on the machine, though the load of about 2 is
not an idle machine. The whole run took about one minute. Figures are therefore
measurements, with the run to run noise of a session that was not completely quiet. The
raw traces of every child are in `/var/folders/jk/p59shldn311gyml_mx2pcry40000gn/T/jl_UBPJIb`
(a temporary directory made with `cleanup = false`, so it stays until the OS clears it). The ranked figures are in
`benchmark/results/trace_compile.toml`.

`trace_s` is the sum of the per-root compile times in the trace. Each root includes the
callees compiled with it, so `trace_s` need not equal Julia's compile clock
(`compile_s`, from `Base.cumulative_compile_time_ns` between the start of `form` and the
end of `assemble`). The two agree to within a few percent on the two large rows and
differ by 19% on `scalar_2d`, where both are about 10 ms.

## Headline figures

| row | form wall, s | assemble wall, s | trace lines | `trace_s`, s | `compile_s`, s | `trace_s` minus `compile_s`, s | functions above 0.5 s | recompiled signatures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| elasticity | 5.600 | 16.745 | 292 | 22.649 | 22.340 | +0.308 (1.4%) | 3 | 0 |
| vector_laplacian | 0.07942 | 2.839 | 20 | 2.989 | 2.918 | +0.071 (2.4%) | 1 | 0 |
| scalar_2d | 0.007155 | 0.004628 | 2 | 0.0096 | 0.0119 | -0.0023 (-19%) | 0 | 0 |

## elasticity

Trace sum 22.65 s over 61 functions and 292 lines, compile clock 22.34 s, form 5.6 s and
assemble 16.75 s (wall).

### by total time

| function | total_s | form_s | assemble_s | instances | recompiled |
|---|---:|---:|---:|---:|---:|
| `Bramble.assemble` | 16.73 | 0.0 | 16.73 | 1 | 0 |
| `Bramble._bilinear_form` | 3.739 | 3.739 | 0.0 | 1 | 0 |
| `Bramble.form` | 1.331 | 1.331 | 0.0 | 1 | 0 |
| `Bramble.OperatorAdd` | 0.1253 | 0.1253 | 0.0 | 69 | 0 |
| `Base.BottomRF` | 0.0544 | 0.0544 | 0.0 | 27 | 0 |
| `Base.foldl` | 0.0497 | 0.0497 | 0.0 | 5 | 0 |
| `Bramble._flatten_tuples` | 0.0471 | 0.0471 | 0.0 | 10 | 0 |
| `Bramble._lower_signs` | 0.0459 | 0.0459 | 0.0 | 1 | 0 |
| `Bramble.inner₊` | 0.04 | 0.04 | 0.0 | 2 | 0 |
| `Base.:(+)` | 0.0348 | 0.0348 | 0.0 | 21 | 0 |
| `Bramble._cross_terms` | 0.0284 | 0.0284 | 0.0 | 9 | 0 |
| `Base.FilteringRF` | 0.023 | 0.023 | 0.0 | 10 | 0 |
| `Base.mapfoldl_impl` | 0.0216 | 0.0216 | 0.0 | 5 | 0 |
| `Base.ntuple` | 0.0212 | 0.0212 | 0.0 | 1 | 0 |
| `Base.indexed_iterate` | 0.021 | 0.0187 | 0.0023 | 7 | 0 |
| `Bramble._pair_plan` | 0.0199 | 0.0 | 0.0199 | 1 | 0 |
| `Bramble._check_form_meshes` | 0.0194 | 0.0194 | 0.0 | 1 | 0 |
| `Bramble.var"#_div_term##0#_div_term##1"` | 0.0194 | 0.0194 | 0.0 | 6 | 0 |
| `Bramble._vc_avg` | 0.0192 | 0.0192 | 0.0 | 6 | 0 |
| `Bramble.BilinearForm` | 0.0184 | 0.0184 | 0.0 | 1 | 0 |

### by instance count

| function | total_s | form_s | assemble_s | instances | recompiled |
|---|---:|---:|---:|---:|---:|
| `Bramble.OperatorAdd` | 0.1253 | 0.1253 | 0.0 | 69 | 0 |
| `Base.BottomRF` | 0.0544 | 0.0544 | 0.0 | 27 | 0 |
| `Base.:(+)` | 0.0348 | 0.0348 | 0.0 | 21 | 0 |
| `Base.:(*)` | 0.0153 | 0.0153 | 0.0 | 12 | 0 |
| `Base.FilteringRF` | 0.023 | 0.023 | 0.0 | 10 | 0 |
| `Bramble._flatten_tuples` | 0.0471 | 0.0471 | 0.0 | 10 | 0 |
| `Bramble.BilinearProduct` | 0.0114 | 0.0114 | 0.0 | 9 | 0 |
| `Bramble._cross_terms` | 0.0284 | 0.0284 | 0.0 | 9 | 0 |
| `Base.first` | 0.0137 | 0.0137 | 0.0 | 7 | 0 |
| `Base.indexed_iterate` | 0.021 | 0.0187 | 0.0023 | 7 | 0 |
| `Base.merge` | 0.0068 | 0.0068 | 0.0 | 6 | 0 |
| `Base.pairs` | 0.0072 | 0.0072 | 0.0 | 6 | 0 |
| `Base.tail` | 0.0154 | 0.0154 | 0.0 | 6 | 0 |
| `Bramble._vc_avg` | 0.0192 | 0.0192 | 0.0 | 6 | 0 |
| `Bramble.var"#_div_term##0#_div_term##1"` | 0.0194 | 0.0194 | 0.0 | 6 | 0 |
| `NamedTuple` | 0.0075 | 0.0075 | 0.0 | 6 | 0 |
| `Base.foldl` | 0.0497 | 0.0497 | 0.0 | 5 | 0 |
| `Base.getproperty` | 0.0127 | 0.0127 | 0.0 | 5 | 0 |
| `Base.mapfoldl_impl` | 0.0216 | 0.0216 | 0.0 | 5 | 0 |
| `Base.argtail` | 0.0038 | 0.0038 | 0.0 | 3 | 0 |

### Functions above 0.5 s

| function | total_s | instances | recompiled |
|---|---:|---:|---:|
| `Bramble.assemble` | 16.73 | 1 | 0 |
| `Bramble._bilinear_form` | 3.739 | 1 | 0 |
| `Bramble.form` | 1.331 | 1 | 0 |

### Recompiled signatures

none

## vector_laplacian

Trace sum 2.989 s over 13 functions and 20 lines, compile clock 2.918 s, form 0.07942 s
and assemble 2.839 s (wall).

### by total time

| function | total_s | form_s | assemble_s | instances | recompiled |
|---|---:|---:|---:|---:|---:|
| `Bramble.assemble` | 2.835 | 0.0 | 2.835 | 1 | 0 |
| `Bramble.form` | 0.0764 | 0.0764 | 0.0 | 1 | 0 |
| `Bramble._pair_plan` | 0.0355 | 0.0 | 0.0355 | 1 | 0 |
| `Bramble.shift_stencil` | 0.0106 | 0.0 | 0.0106 | 3 | 0 |
| `Bramble.D₋` | 0.0071 | 0.0071 | 0.0 | 6 | 0 |
| `Bramble.inner₊` | 0.0038 | 0.0038 | 0.0 | 1 | 0 |
| `Bramble.resolve_ast` | 0.0037 | 0.0037 | 0.0 | 1 | 0 |
| `Bramble.simplify_ast` | 0.0035 | 0.0035 | 0.0 | 1 | 0 |
| `Bramble._lower_signs` | 0.0032 | 0.0032 | 0.0 | 1 | 0 |
| `Bramble._split_restricted_products` | 0.0025 | 0.0025 | 0.0 | 1 | 0 |
| `Base.indexed_iterate` | 0.0022 | 0.0 | 0.0022 | 1 | 0 |
| `Bramble.IndexedTrialFunction` | 0.0013 | 0.0013 | 0.0 | 1 | 0 |
| `Bramble.IndexedTestFunction` | 0.0011 | 0.0011 | 0.0 | 1 | 0 |

### by instance count

| function | total_s | form_s | assemble_s | instances | recompiled |
|---|---:|---:|---:|---:|---:|
| `Bramble.D₋` | 0.0071 | 0.0071 | 0.0 | 6 | 0 |
| `Bramble.shift_stencil` | 0.0106 | 0.0 | 0.0106 | 3 | 0 |
| `Base.indexed_iterate` | 0.0022 | 0.0 | 0.0022 | 1 | 0 |
| `Bramble.IndexedTestFunction` | 0.0011 | 0.0011 | 0.0 | 1 | 0 |
| `Bramble.IndexedTrialFunction` | 0.0013 | 0.0013 | 0.0 | 1 | 0 |
| `Bramble._lower_signs` | 0.0032 | 0.0032 | 0.0 | 1 | 0 |
| `Bramble._pair_plan` | 0.0355 | 0.0 | 0.0355 | 1 | 0 |
| `Bramble._split_restricted_products` | 0.0025 | 0.0025 | 0.0 | 1 | 0 |
| `Bramble.assemble` | 2.835 | 0.0 | 2.835 | 1 | 0 |
| `Bramble.form` | 0.0764 | 0.0764 | 0.0 | 1 | 0 |
| `Bramble.inner₊` | 0.0038 | 0.0038 | 0.0 | 1 | 0 |
| `Bramble.resolve_ast` | 0.0037 | 0.0037 | 0.0 | 1 | 0 |
| `Bramble.simplify_ast` | 0.0035 | 0.0035 | 0.0 | 1 | 0 |

### Functions above 0.5 s

| function | total_s | instances | recompiled |
|---|---:|---:|---:|
| `Bramble.assemble` | 2.835 | 1 | 0 |

### Recompiled signatures

none

## scalar_2d

Trace sum 0.0096 s over 2 functions and 2 lines, compile clock 0.0119 s, form 0.007155 s
and assemble 0.004628 s (wall).

### by total time

| function | total_s | form_s | assemble_s | instances | recompiled |
|---|---:|---:|---:|---:|---:|
| `Bramble.form` | 0.0061 | 0.0061 | 0.0 | 1 | 0 |
| `Bramble.assemble` | 0.0034 | 0.0 | 0.0034 | 1 | 0 |

### by instance count

| function | total_s | form_s | assemble_s | instances | recompiled |
|---|---:|---:|---:|---:|---:|
| `Bramble.assemble` | 0.0034 | 0.0 | 0.0034 | 1 | 0 |
| `Bramble.form` | 0.0061 | 0.0061 | 0.0 | 1 | 0 |

### Functions above 0.5 s

none

### Recompiled signatures

none

## Follow-ups

The section below gives one verdict per function above 0.5 s, read off the source in
`src/assembly/` and `src/ast/`. The tables give cost, not cause, so the causes come from the
code and from four one-off probes (not part of `benchmark/trace_compile.jl`, not kept) on the
same graded 5x5x5 mesh and `gridspace(Ωₕ)^Val(3)`: a count of the summands of `form.ast`, a
count of its distinct summand types, `Base.return_types` of the form closure and of `form`,
and the same for `divₘₕ`.

### What the trace can and cannot say

The trace lists compiled roots only: the entry points reached by a dynamic dispatch. Every
callee dispatched statically is compiled inside its root and is not listed. So the instance
counts of the three roots above 0.5 s (1 each) say nothing about how many specialisations a
root holds, and the trace cannot split `Bramble.assemble` into its kernels or `form` into its
passes. A finer split needs another tool. #915 (Tracy zones per phase of the first `form` and
`assemble`) covers the phase split for these three forms, and #916 (Cthulhu over the replay
kernels) covers the split inside `assemble`. Neither needs a new issue.

### What multiplies the work in `assemble`

`assemble` compiles one walk per summand type, not per form. `_foldl_pairs` is `@generated`
and unrolls `_pair_plan` (`src/assembly/symmetry.jl`), and the unit walks
`_foreach_block_unit`, `_foreach_pair_block_unit`, `_replay_blocks!` and `_replay_pair_blocks!`
(`src/assembly/bilinear_execution.jl`) are `@noinline` with the term type as a parameter, so
each term type keeps its own instance. Each is visited by the coordinate pass
(`_form_coordinates`, sink `_CoordPass`) and by the replay (`ReplaySink`, and
`_PairReplaySink` for a transposed pair). The summand type, with the sink type, is therefore
the parameter that multiplies instances. The probe gives, for the two 3-component rows:

| row | summands in `form.ast` | distinct summand types | `_pair_plan` groups | printed length of `typeof(ast)` | first `assemble`, s |
|---|---:|---:|---:|---:|---:|
| vector_laplacian | 9 | 3 | 9 (0 pairs, 9 single) | 1837 | 2.835 |
| elasticity | 35 | 24 | 26 (9 pairs, 17 single) | 13610 | 16.73 |

The type count rises 8x and the time 5.9x. That is the same order, as a per-type kernel
predicts. Elasticity expands `εₕ` into off-diagonal pieces (`_strain_pieces`) that each give a
new type, where the vector Laplacian's nine summands fall into three types. The 69
`Bramble.OperatorAdd` instances in the elasticity `form` (0.125 s, below the line) show the
same thing on the form side: the AST is built as many distinct type trees. Both rows hold only
`Float64` scalars and no source closures, so #912 (scalar types) and #917 (closure types) do
not apply to them.

### Why `form` is a separate root only for elasticity

The vector Laplacian closure infers to a concrete type and `form` infers to a concrete
`BilinearForm`. The elasticity closure does not: `Base.return_types` gives a non-concrete
`OperatorAdd{3, OperatorScale{3, Float64, ...}, ...}`, and `form(W, W, f)` infers as `Any`. So
inside `form` the calls to `_validate_form_expression`, `_holds_rank_one` and `_bilinear_form`
are dynamic dispatches. That is why `Bramble._bilinear_form` shows up as its own root (3.739 s,
`simplify_ast` and the mesh checks) beside `Bramble.form` (1.331 s).

The loss is in `divₘₕ`. `inner₊(εₕ(u), εₕ(v))` and `2μ * inner₊(εₕ(u), εₕ(v))` infer
concretely, while `divₘₕ(u)` infers as `_DivergenceTerms{3, <:Tuple{Union{...}}}`. `_div_term`
(`src/operators/difference.jl`) stays non-concrete even for a constant component index, and
the same average chain written out by hand, `_vc_avg(_vc_avg(_vc_bwd(u(1), 1), 2), 3)`, infers
concretely. So the cause sits in the `foldl` over `Iterators.filter(!=(i), 1:D)`. The node
type grows at each step of a runtime-filtered range, which gives inference nothing to unroll.

### Verdicts

One line per function above 0.5 s.

- elasticity, `Bramble.assemble`, 16.73 s: #916. The root holds the per-summand-type walks above
  for both sinks, 24 summand types and 9 transposed pairs. #916 asks which kernel arguments the
  hot loop needs and which only multiply instances, which is the question here.
- elasticity, `Bramble._bilinear_form`, 3.739 s: #918. The root is `simplify_ast`, the mesh
  checks and `_split_restricted_products`, all of which only walk a type-encoded tree; each is
  re-specialised on the 13610 character AST type. #918 erases that type in the form-time layer.
- elasticity, `Bramble.form`, 1.331 s: new: make `divₘₕ` infer concretely, so the elasticity form
  infers (paragraph below).
- vector_laplacian, `Bramble.assemble`, 2.835 s: #916. The same kernels with 3 summand types and no
  pairs, so it is the cheap end of the same question, and a useful control for #916's
  before and after.

### new: make `divₘₕ` infer concretely, so the elasticity form infers

Problem. `_div_term` builds each averaged divergence term with a `foldl` over a runtime
filtered range, so `divₘₕ(u)` returns a non-concrete `_DivergenceTerms`, the elasticity
closure is non-concrete, and `form` infers as `Any`. Every pass inside `form` is then reached
by dynamic dispatch, and the 3-component vector Laplacian, whose closure is concrete, does not
pay that. The comment above `_div_term` justifies the `foldl` as builder-time code, which is
true of its run time and says nothing of the inference.

Measured cost. From the tables: `Bramble.form` 1.331 s and `Bramble._bilinear_form` 3.739 s,
5.07 s of the 5.6 s first `form`. The trace cannot say how much of that the non-concrete
inference causes: the same passes compile either way, and a statically dispatched
`_bilinear_form` merges into `form`'s root. The saving may be small or zero, so this is a
measure-first change, not a promised one.

Proposed change. Write `_div_term` so the type is known from `D`: for example a recursion over
`ntuple`-built axes with `Val(D)`, or a `@generated` unrolling, that applies `_vc_avg` once per
axis other than `i`. The result must be the same node as today (same type, so the same
assembled matrices bit for bit and the same `_DivergenceTerms`). Check `εₕ` and `_vc_avg`
for the same pattern.

How to measure. Before and after, in fresh processes: `Base.return_types` of the elasticity
closure and of `form` (the target is `BilinearForm{...}`, not `Any`), then
`julia --startup-file=no benchmark/trace_compile.jl --launches 3 --out ...` on a run alone, and
compare `Bramble.form` plus `Bramble._bilinear_form` (before) with `Bramble.form` (after) in the
elasticity row, and the `compile_s` of the row. A change inside the spread of the
launches is recorded as rejected. The warm `form` and `assemble!` figures must stay within
their spread, and the unit tests of `test/form/` must pass.
