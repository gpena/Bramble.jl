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
(a temporary directory, not kept). The ranked figures are in
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
