# Per-phase first call and warm time of form and assemble (v4.1.4, S7)

Answers #915: how the first call of `form` and `assemble` splits over the phases the issue
names, for the three rows of `benchmark/trace_compile.jl`, read off Tracy zones. It records
figures only and proposes no change. The times are taken while other work used the
machine (load below), and they are retaken on main after the v4.1.3-perf plan merges.

## Method and conditions

`benchmark/tracy_zones.patch` wraps twelve phases in `Tracy.@tracepoint` zones in a
throwaway worktree of HEAD, and `benchmark/tracy_phases.jl` is the recipe: it applies the
patch there, builds a temporary environment with that Bramble plus Tracy and
TracyProfiler_jll, runs one fresh Julia child per row under `tracy-capture`, exports the
zones with `tracy-csvexport` and sums the zone durations per phase per call. Nothing in
`src/`, `ext/`, `Project.toml` or `Manifest.toml` of the branch changes. Command:

```
julia --startup-file=no benchmark/tracy_phases.jl --workload off --out benchmark/results/tracy_phases.toml
```

The measured run is workload off: the temporary environment sets Bramble's
`precompile_workload` preference to false, so the first call pays for everything the
precompile workload would otherwise compile ahead of time. With the workload on, a first
call of the forms it covers has almost nothing left to compile (the last section), and the
#763 figures are compile costs. The zones nest, so each phase has an inclusive time and a
self time (the zone minus the zones inside it). `allocate_system_matrix` contains
`coordinate_pass`, `pattern_build` and `coordinates_to_positions`. Inclusive times of
different phases do not add up, self times do. The warm figure is the median of three
calls after the first.

`lower_sources` sits on the linear-form path (`form(Wₕ, f)`). The three rows are bilinear,
so it reads 0 in all of them and never runs.

| quantity | value |
|---|---|
| branch | `perf/v4.1.4-measure` at `ebaa231d`; `src/`, `ext/`, `Project.toml` and `Manifest.toml` equal main's `9c34251c` |
| Julia | 1.13.2, default `-O`, `--threads 1` per child |
| machine | Apple M2 (`Mac14,2`), 8 cores, 16 GiB, on AC power, battery 100% |
| Tracy | Tracy.jl 0.1.6, TracyProfiler_jll 0.13.1+1, LibTracyClient_jll 0.13.1+0 |
| preflight | `gate_preflight.sh --timing` at the start: FAIL on load (1-minute load 2.63 on 8 cores, limit 2); power OK (AC), OneDrive not running, no other Julia process, no CPU above 50% |
| load | 1-minute, 5-minute, 15-minute averages at the start of each child: 3.66, 3.04, 2.72 (elasticity), 3.52, 3.02, 2.71 (vector_laplacian), 3.64, 3.05, 2.73 (scalar_2d); 2.83, 2.96, 2.74 when both runs had ended |

The preflight failed, so these are not clean measurements: the load came from other
sessions on the machine, and a quiet machine would likely give
smaller first-call figures. They are labelled under load. Ratios between phases are more
trustworthy than absolute times. Raw figures: `benchmark/results/tracy_phases.toml`.

The inference-barrier caveat. Every zone calls through `Base.inferencebarrier`. Without
it, inference of `form` and `assemble` compiles every callee before any zone runs and the
whole compile lands outside the zones (S7.1 saw 0.003 ms of zones in a 295 ms `form`).
With it, a callee compiles where its zone runs, so each phase is charged its own compile.
The price is that the callers see `Any`, so the totals can differ from an un-instrumented
run, and a warm figure carries one dynamic dispatch per zone. The next table compares the
sum of the zone self times with the call totals:

| row | call | first call, s | zone self sum, s | outside any zone, s | outside, % |
|---|---|---:|---:|---:|---:|
| elasticity | `form` | 5.075 | 3.892 | 1.183 | 23 |
| elasticity | `assemble` | 18.680 | 16.175 | 2.505 | 13 |
| vector_laplacian | `form` | 0.362 | 0.160 | 0.202 | 56 |
| vector_laplacian | `assemble` | 3.443 | 2.602 | 0.841 | 24 |
| scalar_2d | `form` | 0.295 | 0.172 | 0.122 | 42 |
| scalar_2d | `assemble` | 0.822 | 0.657 | 0.165 | 20 |

Between 13% and 56% of a call lies outside every zone: the code of `form` and `assemble`
that no zone wraps, such as building the `BilinearForm`, the closure call, the
`inner₊`/`∇ₕ` constructors, and inference and compilation of what runs between the zones.
The share is large for `form` of the two small rows because their zones are small. For
elasticity the zones cover 77% of `form` and 87% of `assemble`.

Per-call totals of the same run:

| row | first `form`, ms | warm `form`, ms | first `assemble`, ms | warm `assemble`, ms |
|---|---:|---:|---:|---:|
| elasticity | 5075.0 | 0.127 | 18680.2 | 2.04 |
| vector_laplacian | 361.7 | 0.0042 | 3443.1 | 0.097 |
| scalar_2d | 294.7 | 0.0032 | 821.5 | 0.017 |

## elasticity

3D elasticity, `2μ·inner₊(εₕ u, εₕ v) + λ·inner₊(divₘₕ u, divₘₕ v)`, graded 5x5x5 mesh, three components. Times in ms, workload off, one thread. Sorted by inclusive first-call time.

| phase | first call, inclusive | first call, self | warm, inclusive | warm, self | zone runs (first/warm) |
|---|---:|---:|---:|---:|---:|
| `allocate_system_matrix` | 9294 | 2256 | 1.72 | 0.083 | 1/1 |
| `coordinate_pass` | 6962 | 6805 | 0.612 | 0.579 | 1/1 |
| `replay` | 6881 | 6881 | 0.290 | 0.272 | 1/1 |
| `simplify_ast` | 3725 | 3725 | 0.055 | 0.055 | 1/1 |
| `check_block_meshes` | 157 | 157 | 0.048 | 0.048 | 96/96 |
| `split_restricted_products` | 102 | 102 | 0.0027 | 0.0027 | 1/1 |
| `pattern_build` | 63.0 | 63.0 | 0.323 | 0.323 | 1/1 |
| `check_form_meshes` | 52.5 | 52.5 | 0.0018 | 0.0018 | 1/1 |
| `resolve_ast` | 12.8 | 12.8 | 0.0021 | 0.0021 | 1/1 |
| `coordinates_to_positions` | 12.7 | 12.7 | 0.675 | 0.675 | 1/1 |
| `validate_form_expression` | 0.021 | 0.021 | 0.0010 | 0.0010 | 1/1 |
| `lower_sources` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/0 |

The top phase by first-call time is `allocate_system_matrix`, 9.294 s inclusive (39% of `form` plus `assemble`), warm 1.72 ms.

The top phase by self time is `replay`, 6.881 s, warm 0.272 ms.

## vector_laplacian

3D vector Laplacian, `inner₊(∇ₕ u, ∇ₕ v)`, graded 5x5x5 mesh, three components. Times in ms, workload off, one thread. Sorted by inclusive first-call time.

| phase | first call, inclusive | first call, self | warm, inclusive | warm, self | zone runs (first/warm) |
|---|---:|---:|---:|---:|---:|
| `allocate_system_matrix` | 1918 | 635 | 0.080 | 0.0078 | 1/1 |
| `coordinate_pass` | 1204 | 1178 | 0.041 | 0.039 | 1/1 |
| `replay` | 684 | 684 | 0.014 | 0.012 | 1/1 |
| `split_restricted_products` | 83.3 | 83.3 | 0.0005 | 0.0005 | 1/1 |
| `pattern_build` | 65.3 | 65.3 | 0.022 | 0.022 | 1/1 |
| `check_form_meshes` | 33.5 | 33.5 | 0.0003 | 0.0003 | 1/1 |
| `simplify_ast` | 29.9 | 29.9 | 0.0002 | 0.0002 | 1/1 |
| `check_block_meshes` | 26.1 | 26.1 | 0.0037 | 0.0037 | 27/27 |
| `coordinates_to_positions` | 13.9 | 13.9 | 0.011 | 0.011 | 1/1 |
| `resolve_ast` | 12.9 | 12.9 | 0.0004 | 0.0004 | 1/1 |
| `validate_form_expression` | 0.053 | 0.053 | 0.0002 | 0.0002 | 1/1 |
| `lower_sources` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/0 |

The top phase by first-call time is `allocate_system_matrix`, 1.918 s inclusive (50% of `form` plus `assemble`), warm 0.080 ms.

The top phase by self time is `coordinate_pass`, 1.178 s, warm 0.039 ms.

## scalar_2d

2D scalar Laplacian, `inner₊(∇ₕ u, ∇ₕ v)`, graded 9x9 mesh. Times in ms, workload off, one thread. Sorted by inclusive first-call time.

| phase | first call, inclusive | first call, self | warm, inclusive | warm, self | zone runs (first/warm) |
|---|---:|---:|---:|---:|---:|
| `allocate_system_matrix` | 547 | 76.7 | 0.015 | 0.0020 | 1/1 |
| `coordinate_pass` | 390 | 377 | 0.0058 | 0.0055 | 1/1 |
| `replay` | 110 | 110 | 0.0012 | 0.0010 | 1/1 |
| `split_restricted_products` | 78.1 | 78.1 | 0.0004 | 0.0004 | 1/1 |
| `pattern_build` | 68.0 | 68.0 | 0.0059 | 0.0059 | 1/1 |
| `simplify_ast` | 53.0 | 53.0 | 0.0001 | 0.0001 | 1/1 |
| `check_form_meshes` | 32.5 | 32.5 | 0.0002 | 0.0002 | 1/1 |
| `check_block_meshes` | 13.4 | 13.4 | 0.0005 | 0.0005 | 3/3 |
| `coordinates_to_positions` | 12.4 | 12.4 | 0.0023 | 0.0023 | 1/1 |
| `resolve_ast` | 8.76 | 8.76 | 0.0002 | 0.0002 | 1/1 |
| `validate_form_expression` | 0.034 | 0.034 | 0.0001 | 0.0001 | 1/1 |
| `lower_sources` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/0 |

The top phase by first-call time is `allocate_system_matrix`, 0.547 s inclusive (49% of `form` plus `assemble`), warm 0.015 ms.

The top phase by self time is `coordinate_pass`, 0.377 s, warm 0.0055 ms.

## Comparison with #763 and #914

#763 (SnoopCompile, under snoop, 3D elasticity, 5x5x5): `simplify_ast` 3.15 s over 248
instances, `_check_walked_meshes` 1.73 s over 122 instances, first `assemble` about 16.5 s
after #797 (17.78 s before it, `allocate_system_matrix` 2.10 s in one instance, the replay
kernels about 15 s together). Against it:

- `simplify_ast`: 3.725 s here (compile plus run, zone time) against 3.15 s of exclusive
  inference there. The same order, and it is the phase of `form` that takes the time: 96%
  of the zones of `form`.
- `_check_walked_meshes`: the mesh check of `form` (`check_form_meshes`) takes 0.052 s.
  The 1.73 s of #763 is gone from `form`, as #797 reduced the instances of `form` from
  5803 to 2427. `check_block_meshes` in `assemble` takes 0.157 s over 96 calls.
- first `assemble`: 18.68 s here (workload off) against 16.5 s after #797 and 17.78 s
  before it, under a load of 3.3 to 3.7 (#763's runs were under a load of 3 to 4.8). Its
  `allocate_system_matrix` self time is 2.256 s, the 2.10 s of #763. The coordinate pass
  (6.8 s self) and the replay (6.9 s self) hold the rest, the 15 s of #763 as a pair of
  phases. First `form`: 5.07 s here against 4.62 s after #797.
- warm `assemble` of elasticity is 2.0 ms here (zones included, 5x5x5, median of three),
  not the 47 ms of #763. `benchmark/results/multiterm_ttfx.toml` reads 0.55 ms for the same
  row; this run did not look into the difference.

#914 ranked compile time by function, with `--trace-compile-timing`
(`docs/notes/trace-compile-v4.1.4.md`). For elasticity `assemble` 16.73 s, `_bilinear_form`
3.739 s and `form` 1.331 s lead, and nothing inside `assemble` appears because the trace
lists the root call. The zones split that root: of the 16.7 s about 9.3 s is in
`allocate_system_matrix` (7.0 s of it in the coordinate pass) and 6.9 s in the replay,
so the two share it about equally, which function-level ranking could not show. The
`_bilinear_form` figure of 3.739 s matches `simplify_ast` (3.725 s). The same ordering
holds for the other rows: `assemble` takes about ten times the `form` time for
vector_laplacian and about three times for scalar_2d (workload off).

## Workload on

One more run, `--workload on` (the library as shipped), written to a file outside the
repository. Only the first-call totals; the zones are not recorded here because with the
workload on a covered form has almost nothing left to attribute. Load at the start 3.28,
3.04, 2.77. The elasticity form is not in the default workload, so it compiles as before.

| row | `form` first call, ms | `assemble` first call, ms | `form` first call workload off, ms | `assemble` first call workload off, ms |
|---|---:|---:|---:|---:|
| elasticity | 4915.4 | 18538.4 | 5075.0 | 18680.2 |
| vector_laplacian | 208.4 | 3138.6 | 361.7 | 3443.1 |
| scalar_2d | 10.6 | 0.5 | 294.7 | 821.5 |

Elasticity is unchanged within the spread (the precompile workload does not cover it, per
`docs/notes/internals/precompile_elasticity.md`). The vector Laplacian loses about 150 ms
of `form` and 300 ms of `assemble`. The scalar 2D form compiles almost nothing at its
first call: 10.6 ms for `form` (all of it outside the zones) and 0.46 ms for `assemble`.

## Retake

These times are not final. Retake them on main after the v4.1.3-perf plan merges, with a
passing preflight, the same command and the same patch.
