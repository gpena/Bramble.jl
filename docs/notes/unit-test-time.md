# Unit test time: baseline on cfff528d

Measured by `test/timing.jl` on the unit group, before any v4.0.1 follow-up test changes
(S5.2 of `.claude/plans/v4-0-1.md`, issue gpena/Bramble.jl#802). Raw output is outside the
repo in `timing-v4.0.1/`: `before.tsv`, `before.tsv.testsets`, `before.log`. S5.11 compares
against `before.tsv`.

## Header

| Field | Value |
|---|---|
| Commit | `cfff528d` (branch `v4.0.1/followups`) |
| Julia | 1.13.2 |
| Threads | 4 (`--threads=4 --optimize=1 --check-bounds=yes`) |
| Group | unit (`BRAMBLE_TEST_GROUP=unit`) |
| Files | 157 trace entries, paths relative to `test/` |
| Top-level testset rows | 459 in `before.tsv.testsets` |
| Testsets | 3355 nested testsets |
| Tests | 27521 |
| Suite wall | 1520.9 s |
| Process wall (`/usr/bin/time`) | 1529.8 s, user 1658 s, sys 187 s |
| Compile total | 1431.8 s, summed over threads (so it can exceed wall) |
| Peak RSS | 4.58 GB (`/usr/bin/time -l`), under the 10 GB slot-watchdog cap |
| Load at start | 1.98 (1-min, preflight, 17:00) |
| Load during and at end | 1-min load 2.0 to 2.6 through the run (2.41 at 17:26) |

**Under load.** The 1-min load stayed a little above the 2.0 idle threshold during the run,
so the figures are "under load". Re-measure on an idle machine before a speedup claim
(`bramble-verification`).

How to read the columns:

- `compile_s` is JIT time summed over threads. It is inclusive of the files a file includes,
  the same as `wall_s`.
- Files run in suite order, so the first file to reach a code path pays its compilation.
- Testset rows: `compile_s` is the compile counter since the previous top-level testset of
  the trace ended. That gap also covers the file's own code between testsets, and code in the
  file before its first testset. So a testset's compile can include earlier code. The per-file
  wall minus the sum of its testset walls is file-level time that no testset covers. The trace
  does not split it further.
- `runtests.jl` rows are containers. Their wall includes the files they include.

## Per-file table, ranked by wall time

Sorted by `wall_s`, longest first. `depth` 0 is a runtests container or a top-level file, 1 is
a file that a runtests container includes. `self_s` is wall minus the files it includes.

| file | depth | wall_s | self_s | compile_s |
|---|---|---|---|---|
| form/runtests.jl | 0 | 993.82 | 0.06 | 935.37 |
| ghost/runtests.jl | 0 | 257.93 | 0.01 | 257.14 |
| form/bilinear.jl | 1 | 216.59 | 216.59 | 199.78 |
| ghost/backends.jl | 1 | 65.53 | 65.53 | 68.27 |
| form/linear_restriction.jl | 1 | 62.33 | 62.33 | 61.33 |
| form/matrix_free.jl | 1 | 61.51 | 61.51 | 61.42 |
| form/kronecker_edge.jl | 1 | 55.98 | 55.98 | 56.25 |
| form/linear.jl | 1 | 54.45 | 54.45 | 45.69 |
| form/nested_operators.jl | 1 | 53.27 | 53.27 | 53.18 |
| solvers/runtests.jl | 0 | 51.65 | 0.01 | 43.71 |
| ghost/cross_component.jl | 1 | 51.22 | 51.22 | 49.17 |
| ghost/allocations.jl | 1 | 35.41 | 35.41 | 38.26 |
| space/ghost_vector.jl | 0 | 34.78 | 34.78 | 38.88 |
| form/kronecker.jl | 1 | 31.36 | 31.36 | 29.00 |
| form/source_operators.jl | 1 | 31.02 | 31.02 | 27.18 |
| utils/runtests.jl | 0 | 28.91 | 0.00 | 25.22 |
| form/ghost_corners.jl | 1 | 26.39 | 26.39 | 25.70 |
| form/ghost_facet_operators.jl | 1 | 25.73 | 25.73 | 24.77 |
| ghost/composite.jl | 1 | 22.64 | 22.64 | 21.84 |
| mesh/runtests.jl | 0 | 22.50 | 0.01 | 17.05 |
| form/ghost_heat_convergence.jl | 1 | 20.96 | 20.96 | 17.13 |
| solvers/multigrid.jl | 1 | 19.41 | 19.41 | 18.68 |
| form/ghost_semidiscrete.jl | 1 | 18.40 | 18.40 | 18.59 |
| form/ghost_time_data.jl | 1 | 18.17 | 18.17 | 18.08 |
| ghost/preconditioners.jl | 1 | 17.77 | 17.77 | 16.88 |
| form/operators.jl | 1 | 16.65 | 16.65 | 16.17 |
| ghost/multigrid.jl | 1 | 16.58 | 16.58 | 14.88 |
| form/simplifier.jl | 1 | 15.52 | 15.52 | 14.96 |
| form/symmetry.jl | 1 | 15.02 | 15.02 | 13.84 |
| form/ghost_rank_one.jl | 1 | 14.90 | 14.90 | 14.67 |
| form/normal.jl | 1 | 14.64 | 14.64 | 13.27 |
| form/kronecker_projection.jl | 1 | 14.62 | 14.62 | 14.21 |
| space/vector_elements.jl | 0 | 14.59 | 14.59 | 11.80 |
| form/threaded_replay.jl | 1 | 14.09 | 14.09 | 13.68 |
| form/ghost_assembly.jl | 1 | 13.97 | 13.97 | 13.74 |
| form/ghost_constrained.jl | 1 | 13.75 | 13.75 | 13.58 |
| form/interpolation_operator.jl | 1 | 13.21 | 13.21 | 11.56 |
| space/inner_product.jl | 0 | 12.66 | 12.66 | 11.69 |
| form/difference_ast.jl | 1 | 12.63 | 12.63 | 10.19 |
| form/reaction_flux.jl | 1 | 12.51 | 12.51 | 11.33 |
| space/weights_staleness.jl | 0 | 12.37 | 12.37 | 4.20 |
| utils/batch_split.jl | 1 | 11.91 | 11.91 | 11.73 |
| form/common.jl | 1 | 11.61 | 11.61 | 10.72 |
| form/rank_one.jl | 1 | 11.07 | 11.07 | 9.14 |
| ghost/matrix_free.jl | 1 | 10.65 | 10.65 | 11.25 |
| ghost/traction.jl | 1 | 9.13 | 9.13 | 8.33 |
| form/two_spaces.jl | 1 | 9.10 | 9.10 | 8.99 |
| form/coordinate_walk.jl | 1 | 8.53 | 8.53 | 9.26 |
| space/ghost_vector_operators.jl | 0 | 8.48 | 8.48 | 9.83 |
| form/jacobian_pattern_blocks.jl | 1 | 8.44 | 8.44 | 8.20 |
| solvers/rank_one_solve.jl | 1 | 8.41 | 8.41 | 6.30 |
| utils/linear_algebra.jl | 1 | 8.34 | 8.34 | 6.90 |
| ghost/kronecker.jl | 1 | 8.27 | 8.27 | 8.12 |
| mesh/meshnd.jl | 1 | 8.18 | 8.18 | 6.02 |
| form/kronecker_block.jl | 1 | 8.09 | 8.09 | 8.12 |
| form/inner_products.jl | 1 | 7.77 | 7.77 | 6.90 |
| quality/alloccheck.jl | 0 | 7.72 | 7.72 | 5.70 |
| solvers/preconditioners.jl | 1 | 7.68 | 7.68 | 7.00 |
| form/semidiscrete.jl | 1 | 7.58 | 7.58 | 5.70 |
| form/dirac.jl | 1 | 7.21 | 7.21 | 6.44 |
| ghost/policies.jl | 1 | 6.78 | 6.78 | 7.15 |
| solvers/neumann_compatibility.jl | 1 | 6.73 | 6.73 | 4.64 |
| space/inplace_operators.jl | 0 | 6.37 | 6.37 | 5.96 |
| form/laplacian_form.jl | 1 | 6.20 | 6.20 | 6.15 |
| space/threaded_vector_calculus.jl | 0 | 6.03 | 6.03 | 8.38 |
| form/ghost_convergence.jl | 1 | 4.97 | 4.97 | 4.87 |
| form/dirichlet_constraints.jl | 1 | 4.61 | 4.61 | 3.42 |
| space/element_type.jl | 0 | 4.43 | 4.43 | 4.34 |
| space/threaded_stencils.jl | 0 | 4.41 | 4.41 | 6.13 |
| ghost/contract.jl | 1 | 4.09 | 4.09 | 3.94 |
| utils/nested_threading.jl | 1 | 4.03 | 4.03 | 3.78 |
| form/warm_composite.jl | 1 | 3.96 | 3.96 | 3.91 |
| geometry/runtests.jl | 0 | 3.79 | 0.00 | 2.46 |
| space/difference.jl | 0 | 3.76 | 3.76 | 3.47 |
| form/forwarddiff_smoke.jl | 1 | 3.66 | 3.66 | 3.60 |
| space/operators.jl | 0 | 3.61 | 3.61 | 3.32 |
| form/replay_markers.jl | 1 | 3.49 | 3.49 | 3.68 |
| space/dimensional_dispatch.jl | 0 | 3.31 | 3.31 | 3.14 |
| form/centered_vector_calculus.jl | 1 | 3.30 | 3.30 | 3.26 |
| space/shift.jl | 0 | 3.28 | 3.28 | 3.79 |
| ghost/dirac.jl | 1 | 3.17 | 3.17 | 3.11 |
| form/ghost_mode_a.jl | 1 | 3.17 | 3.17 | 3.00 |
| space/composite_operators.jl | 0 | 3.16 | 3.16 | 2.84 |
| space/ghost_gridspace.jl | 0 | 3.12 | 3.12 | 2.89 |
| form/ghost_dirichlet.jl | 1 | 3.03 | 3.03 | 2.86 |
| mesh/mesh1d.jl | 1 | 3.01 | 3.01 | 2.12 |
| mesh/meshes.jl | 1 | 2.95 | 2.95 | 2.17 |
| exporters/runtests.jl | 0 | 2.95 | 0.01 | 2.37 |
| ghost/grid_functions.jl | 1 | 2.81 | 2.81 | 2.59 |
| form/skew.jl | 1 | 2.81 | 2.81 | 2.67 |
| mesh/markers.jl | 1 | 2.80 | 2.80 | 2.46 |
| utils/backends.jl | 1 | 2.76 | 2.76 | 0.90 |
| form/assemble_add.jl | 1 | 2.75 | 2.75 | 2.61 |
| solvers/sparse_solvers.jl | 1 | 2.68 | 2.68 | 0.65 |
| geometry/domains.jl | 1 | 2.68 | 2.68 | 1.72 |
| form/component.jl | 1 | 2.68 | 2.68 | 2.17 |
| space/cross_weighted_difference.jl | 0 | 2.54 | 2.54 | 2.26 |
| convergence/runtests.jl | 0 | 2.51 | 0.00 | 1.32 |
| convergence/operators.jl | 1 | 2.51 | 2.51 | 1.32 |
| solvers/gauge_solve.jl | 1 | 2.47 | 2.47 | 2.35 |
| space/star_vector_calculus.jl | 0 | 2.37 | 2.37 | 2.28 |
| space/inference_allocation.jl | 0 | 2.27 | 2.27 | 2.05 |
| solvers/ghost_kernel.jl | 1 | 2.27 | 2.27 | 2.17 |
| space/threaded_broadcast.jl | 0 | 2.22 | 2.22 | 2.70 |
| form/vector_sums.jl | 1 | 2.16 | 2.16 | 2.10 |
| mesh/ghost_mesh.jl | 1 | 2.02 | 2.02 | 1.38 |
| solvers/mean_zero.jl | 1 | 1.99 | 1.99 | 1.92 |
| space/average.jl | 0 | 1.93 | 1.93 | 1.83 |
| space/centered_vector_calculus.jl | 0 | 1.92 | 1.92 | 1.83 |
| form/symmetrize.jl | 1 | 1.88 | 1.88 | 1.61 |
| space/interpolation.jl | 0 | 1.77 | 1.77 | 1.31 |
| form/int32_scatter.jl | 1 | 1.76 | 1.76 | 1.72 |
| ghost/corners.jl | 1 | 1.72 | 1.72 | 1.43 |
| form/replay_sinks.jl | 1 | 1.64 | 1.64 | 1.88 |
| form/markers.jl | 1 | 1.64 | 1.64 | 1.43 |
| form/zero_form.jl | 1 | 1.62 | 1.62 | 1.61 |
| space/gridspaces.jl | 0 | 1.54 | 1.54 | 0.98 |
| form/interpolation.jl | 1 | 1.44 | 1.44 | 1.53 |
| mesh/mesh_state.jl | 1 | 1.42 | 1.42 | 1.22 |
| ghost/oracle_nd.jl | 1 | 1.41 | 1.41 | 1.26 |
| exporters/pgfplots_export.jl | 1 | 1.32 | 1.32 | 1.12 |
| form/repeated_assemble.jl | 1 | 1.29 | 1.29 | 1.23 |
| form/centered_average.jl | 1 | 1.27 | 1.27 | 1.24 |
| form/ghost_lift.jl | 1 | 1.22 | 1.22 | 1.11 |
| form/extended_operators.jl | 1 | 1.21 | 1.21 | 0.99 |
| mesh/constructors.jl | 1 | 1.18 | 1.18 | 0.95 |
| form/block_extract.jl | 1 | 1.16 | 1.16 | 1.06 |
| space/star_difference.jl | 0 | 1.13 | 1.13 | 0.61 |
| space/interpolation_bounds.jl | 0 | 1.12 | 1.12 | 0.64 |
| geometry/sets.jl | 1 | 1.11 | 1.11 | 0.73 |
| form/stencil_pattern.jl | 1 | 1.03 | 1.03 | 0.93 |
| exporters/vtk_export.jl | 1 | 1.01 | 1.01 | 0.70 |
| space/centered_difference.jl | 0 | 0.95 | 0.95 | 0.53 |
| utils/macros.jl | 1 | 0.94 | 0.94 | 0.87 |
| utils/threaded_dot.jl | 1 | 0.93 | 0.93 | 1.04 |
| form/type_cached_assemble.jl | 1 | 0.82 | 0.82 | 0.31 |
| ghost/oracle_1d.jl | 1 | 0.74 | 0.74 | 0.65 |
| form/marker_ids.jl | 1 | 0.74 | 0.74 | 0.65 |
| form/cross_mesh_blocks.jl | 1 | 0.68 | 0.68 | 0.61 |
| drivers/variable_coefficient_poisson.jl | 1 | 0.62 | 0.62 | 0.57 |
| drivers/runtests.jl | 0 | 0.62 | 0.00 | 0.57 |
| exporters/vtk_collection.jl | 1 | 0.61 | 0.61 | 0.55 |
| mesh/inference_allocation.jl | 1 | 0.58 | 0.58 | 0.46 |
| space/commutation.jl | 0 | 0.52 | 0.52 | 0.44 |
| space/sbp_identities.jl | 0 | 0.51 | 0.51 | 0.27 |
| form/expression.jl | 1 | 0.50 | 0.50 | 0.37 |
| space/centered_average.jl | 0 | 0.47 | 0.47 | 0.45 |
| space/inner_plus_boundary.jl | 0 | 0.46 | 0.46 | 0.37 |
| space/discrete_calculus_identities.jl | 0 | 0.46 | 0.46 | 0.27 |
| space/sobolev_inequalities.jl | 0 | 0.37 | 0.37 | 0.21 |
| mesh/marker_diagnostics.jl | 1 | 0.35 | 0.35 | 0.27 |
| space/operator_docstrings.jl | 0 | 0.33 | 0.33 | 0.30 |
| space/backend_profile.jl | 0 | 0.25 | 0.25 | 0.23 |
| space/jump.jl | 0 | 0.21 | 0.21 | 0.14 |
| space/interpolation_outside.jl | 0 | 0.10 | 0.10 | 0.06 |
| space/conservation.jl | 0 | 0.09 | 0.09 | 0.07 |
| form/padding_zero.jl | 1 | 0.00 | 0.00 | 0.00 |

## Files over 30 s

Twelve unit test files take more than 30 s of wall, and `space/ghost_vector.jl` takes 34.8 s
at top level. The three containers `form/runtests.jl` (993.8 s), `ghost/runtests.jl`
(257.9 s) and `solvers/runtests.jl` (51.7 s) are over 30 s too, but they hold the files
above, so they are not analysed.

| file | wall_s | compile_s | testsets sum, wall_s | file-level, not in a testset, wall_s |
|---|---|---|---|---|
| form/bilinear.jl | 216.59 | 199.78 | 142.54 | 74.05 |
| ghost/backends.jl | 65.53 | 68.27 | 64.91 | 0.62 |
| form/linear_restriction.jl | 62.33 | 61.33 | 60.58 | 1.75 |
| form/matrix_free.jl | 61.51 | 61.42 | 53.57 | 7.94 |
| form/kronecker_edge.jl | 55.98 | 56.25 | 46.48 | 9.50 |
| form/linear.jl | 54.45 | 45.69 | 13.42 | 41.03 |
| form/nested_operators.jl | 53.27 | 53.18 | 53.10 | 0.17 |
| ghost/cross_component.jl | 51.22 | 49.17 | 49.45 | 1.77 |
| ghost/allocations.jl | 35.41 | 38.26 | 35.21 | 0.20 |
| space/ghost_vector.jl | 34.78 | 38.88 | 34.53 | 0.25 |
| form/kronecker.jl | 31.36 | 29.00 | 18.27 | 13.09 |
| form/source_operators.jl | 31.02 | 27.18 | 10.84 | 20.18 |

The file-level column is the file's wall minus the sum of its top-level testset walls. It
is time in the file's own code outside every testset: the module header and any setup, and
top-level code between testsets. `form/linear.jl` (41 s), `form/bilinear.jl` (74 s) and
`form/source_operators.jl` (20 s) have large file-level shares. Their levers below are
unresolved until that time is split.

## Per-testset table for each file over 30 s

Every top-level testset of the twelve files and of `space/ghost_vector.jl`, in suite order.
`compile_s` follows the gap rule above.

| file | testset | wall_s | compile_s |
|---|---|---|---|
| form/bilinear.jl | Bilinear forms | 138.19 | 195.44 |
| form/bilinear.jl | bilinear: restricted sum replay width | 0.42 | 0.42 |
| form/bilinear.jl | bilinear: matrix-free form call | 3.11 | 3.10 |
| form/bilinear.jl | pair walk, unequal block counts | 0.42 | 0.42 |
| form/bilinear.jl | zeroing a non-CSC sparse matrix | 0.03 | 0.03 |
| form/bilinear.jl | bilinear: traversal and policy traits | 0.02 | 0.02 |
| form/bilinear.jl | symbolic and source-only leaves | 0.06 | 0.06 |
| form/bilinear.jl | source stencils, non-uniform mesh | 0.29 | 0.28 |
| ghost/backends.jl | Ghost spaces keep the backend | 3.38 | 3.54 |
| ghost/backends.jl | Ghost forms store the backend's types | 9.86 | 10.24 |
| ghost/backends.jl | constrained keeps the backend | 46.81 | 49.21 |
| ghost/backends.jl | Mesh edits keep the backend | 0.29 | 0.49 |
| ghost/backends.jl | A ghost mesh has an immutable walk state | 0.99 | 1.12 |
| ghost/backends.jl | Ghost constructors are inferable | 3.58 | 3.66 |
| form/linear_restriction.jl | Linear forms: restrictions | 60.58 | 61.33 |
| form/matrix_free.jl | matrix-free operator (#326) | 53.57 | 61.42 |
| form/kronecker_edge.jl | Kronecker edge cases | 46.48 | 56.25 |
| form/linear.jl | Linear forms | 13.42 | 45.69 |
| form/nested_operators.jl | Nested stencil operators in forms (#287) | 25.09 | 25.12 |
| form/nested_operators.jl | Δₕ nested in forms (#774) | 13.23 | 13.29 |
| form/nested_operators.jl | jumpₙ nested in inner_Γ (#205) | 14.78 | 14.76 |
| ghost/cross_component.jl | Cross-component ghost relations | 49.45 | 49.17 |
| ghost/allocations.jl | Warm ghost calls allocate (#785) | 10.88 | 11.03 |
| ghost/allocations.jl | Warm refill follows a coefficient | 0.60 | 0.68 |
| ghost/allocations.jl | Warm refill, padded corner rule | 1.16 | 1.16 |
| ghost/allocations.jl | Warm refill, facet rule (#813) | 22.57 | 25.39 |
| space/ghost_vector.jl | GhostVector holds the caller's vector | 0.67 | 0.70 |
| space/ghost_vector.jl | update_ghosts! equals lift (#776) | 17.52 | 17.47 |
| space/ghost_vector.jl | GhostVector refusals (#776) | 0.07 | 0.06 |
| space/ghost_vector.jl | update_ghosts! under CpuThreaded (#776) | 16.27 | 20.64 |
| form/kronecker.jl | Kronecker | 18.27 | 29.00 |
| form/source_operators.jl | Source operators | 10.84 | 27.18 |

## Levers for the files over 30 s

The lever names one of: move a compile path to slow, share types, precompile, or library. A
lever comes from the compile share (compile_s / wall_s), the testset's wall, and whether the
testset is one block or several small ones. Where the file-level share is large, the lever
is unresolved, and the table says so.

| File | Lever | Testset(s) that pay it |
|---|---|---|
| form/bilinear.jl | unresolved for 74 s file-level; share types for the block | "Bilinear forms" (138.2 s wall, 195.4 s compile) |
| ghost/backends.jl | precompile | "constrained keeps the backend" (46.8 s), "Ghost forms store the backend's types" (9.9 s) |
| form/linear_restriction.jl | share types | "Linear forms: restrictions" (60.6 s, all compile) |
| form/matrix_free.jl | precompile | "matrix-free operator (#326)" (53.6 s) |
| form/kronecker_edge.jl | move compile path to slow | "Kronecker edge cases" (46.5 s) |
| form/linear.jl | unresolved: 41 s file-level, not in its one testset | "Linear forms" (13.4 s wall, 45.7 s compile, gap-attributed) |
| form/nested_operators.jl | share types | "Nested stencil operators in forms (#287)" (25.1 s), "jumpₙ nested in inner_Γ (#205)" (14.8 s), "Δₕ nested in forms (#774)" (13.2 s) |
| ghost/cross_component.jl | share types | "Cross-component ghost relations" (49.5 s, all compile) |
| ghost/allocations.jl | precompile | "Warm refill, facet rule (#813)" (22.6 s), "Warm ghost calls allocate (#785)" (10.9 s) |
| space/ghost_vector.jl | library | "update_ghosts! equals lift (#776)" (17.5 s), "update_ghosts! under CpuThreaded (#776)" (16.3 s) |
| form/kronecker.jl | share types | "Kronecker" (18.3 s wall, 13.1 s file-level) |
| form/source_operators.jl | unresolved: 20 s file-level | "Source operators" (10.8 s wall, 27.2 s compile, gap-attributed) |

Notes:

- The compile of the "Linear forms" testset of `form/linear.jl` is 45.7 s against 13.4 s of
  wall. The gap rule explains the excess: the row also holds compile from the gap before
  the testset ends. The trace does not say where that compile happened. The testset is not
  shown to be the lever. Split the file-level time before choosing one.
- `form/bilinear.jl` has 74 s of file-level time that no testset covers. Its "Bilinear
  forms" testset is the largest single block. The lever is "share types" for that block,
  with the file-level time still to be split.
- Compile above wall for a single testset comes from summing over threads, and from the gap
  rule, which can carry compile from earlier code.
- The `ghost/matrix_free.jl` entry in the plan's heavy list is below 30 s and not in this
  table. The 62 s matrix-free testset is `form/matrix_free.jl`, which is in this table.

## After: cdfb0fbe, same harness, run alone

S5.11 ran the S5.2 harness on HEAD `cdfb0fbe`: unit group, 4 threads, `--optimize=1
--check-bounds=yes`, one run on this laptop with nothing else of ours running. Raw output is
outside the repo in `timing-v4.0.1/`: `after.tsv`, `after.tsv.testsets`, `after.log` and
`compare.txt` (`test/timing.jl --compare` of `before.tsv` against `after.tsv`). The run was
started at 14:06 after the timing preflight passed; the load and CPU checks had failed for
about 20 minutes first (macOS system daemons held the CPU).

**Under load.** The 1-min load was 2.9 to 4.1 for most of the run, against 2.0 to 2.6 at
baseline. Compare the figures below with that in mind.

### Header

| Field | Before (cfff528d) | After (cdfb0fbe) |
|---|---|---|
| Commit | `cfff528d` | `cdfb0fbe` |
| Files traced | 157 | 162 (5 new, see below) |
| Testsets | 3355 | 3265 |
| Tests | 27521 | 27947 |
| Suite wall | 1520.9 s | 1456.2 s |
| Compile total | 1431.8 s (summed over threads) | 1327.0 s (summed over threads) |
| `precompile` line | 1.9 s | 2.7 s |
| Peak memory | 4.58 GB RSS (`/usr/bin/time -l`) | 4.66 GB footprint (per-file log column) |
| 1-min load at start | 1.98 | 1.92 at preflight (14:06); 2.09 at the power check |
| 1-min load during and at end | 2.0 to 2.6 | 2.9 to 4.1 during; 3.63 at the end (14:33) |

The run stayed under the 10 GB slot cap. It was not wrapped in `/usr/bin/time`, so the RSS
figure is the per-file footprint column.

### Wall time, with and without the new polygon files

| | Suite wall (s) | Compile (s) |
|---|---|---|
| Baseline cfff528d | 1520.9 | 1431.8 |
| After, as measured | 1456.2 | 1327.0 |
| New polygon files (table below) | 49.2 | 20.7 |
| After, new files taken out | 1407.0 | 1306.3 |

With the new files out, the wall is 113.9 s (7.5%) below baseline. The eight trimmed files
alone fell by 166.5 s of self time (585.4 s to 418.9 s). The rest of the suite, by
difference, rose by 52.6 s. Some of that is drift on untouched files (see below).

### New unit test files since cfff528d

Only the first five are traced by `timing.jl`. `poisson_polygon.jl` is loaded with
`Base.include` inside `convergence/runtests.jl`, so its time is that container's self time,
less `convergence/operators.jl` (33.58 s wall minus 2.89 s). `FGOracle.jl` is loaded from
inside `ghost/polygon_oracle.jl`, so its time is in that file's self time.

| File | Self (s) after | Compile (s) after | How timed |
|---|---|---|---|
| `convergence/poisson_polygon.jl` | 30.69 | 3.39 | container self less `operators.jl` |
| `ghost/polygon_spaces.jl` | 11.21 | 10.66 | traced |
| `ghost/polygon_oracle.jl` (with `FGOracle.jl`) | 4.73 | 4.63 | traced |
| `mesh/polygon_markers.jl` | 1.59 | 1.39 | traced |
| `mesh/polygon_closure.jl` | 0.62 | 0.38 | traced |
| `geometry/polygon.jl` | 0.34 | 0.26 | traced |
| Total | 49.18 | 20.71 | |

### Changed files: before and after

Self time is the file's wall minus the files it includes, as `--compare` reports it. The
trim list is the files the plan's trims changed.

| File | Self before (s) | Self after (s) | Delta (s) | Compile before (s) | Compile after (s) | Status |
|---|---|---|---|---|---|---|
| `form/bilinear.jl` | 216.59 | 164.74 | -51.85 | 199.78 | 147.75 | trimmed |
| `ghost/backends.jl` | 65.53 | 18.35 | -47.18 | 68.27 | 18.56 | trimmed |
| `form/linear_restriction.jl` | 62.33 | 33.71 | -28.62 | 61.33 | 32.74 | trimmed |
| `form/nested_operators.jl` | 53.27 | 42.61 | -10.66 | 53.18 | 42.51 | trimmed |
| `ghost/allocations.jl` | 35.41 | 25.90 | -9.51 | 38.26 | 26.18 | trimmed (#813) |
| `form/kronecker_edge.jl` | 55.98 | 48.28 | -7.70 | 56.25 | 47.05 | trimmed |
| `form/matrix_free.jl` | 61.51 | 56.34 | -5.17 | 61.42 | 55.93 | trimmed |
| `space/ghost_vector.jl` | 34.78 | 28.93 | -5.85 | 38.88 | 31.51 | trimmed |
| `ghost/traction.jl` | 9.13 | 10.02 | +0.89 | 8.33 | 9.20 | slow-only trim, in HEAD before the baseline (`a19e19c1`) |
| `space/ghost_vector_operators.jl` | 8.48 | 12.63 | +4.15 | 9.83 | 14.98 | not on trim list; main commit |
| `form/linear.jl` | 54.45 | 55.51 | +1.06 | 45.69 | 46.77 | left alone (S5.9 abandoned) |
| `ghost/cross_component.jl` | 51.22 | 54.60 | +3.38 | 49.17 | 52.29 | left alone (S5.12 abandoned) |

Notes on the rows:

- `ghost/traction.jl`: the trim is in HEAD. S6 committed it as `a19e19c1`, which comes before
  the baseline commit `cfff528d`, so the baseline and the after run both have it. That is why
  `git diff cfff528d..cdfb0fbe` is empty for this file. Its effect is on the slow group only,
  the 33-point 3D level: S6 measured 365.6 s down to 73.2 s locally. The unit levels read
  `WITH_SLOW_TESTS` (0:3 slow, 0:2 unit), so the unit run is unchanged. Its +0.9 s (+10%) is
  inside the drift range.
- `space/ghost_vector_operators.jl`: not on the plan's trim list. Its only change since the
  baseline is `34687dd1` (main, "GhostVector CpuThreaded bounds allow per-pass closure
  re-boxing"). The +3.7 s is in its "GhostVector operators, CpuThreaded" testset.
- `form/linear.jl` and `ghost/cross_component.jl` were not changed by this plan. S5.9 and
  S5.12 were abandoned, so both files keep their baseline tests. Their +1 to +3 s is drift.

### Per-testset, top-level testsets of the changed files

Wall and compile in seconds, from `after.tsv.testsets` against `before.tsv.testsets`.

| File | Testset | Wall before | Wall after | Delta | Compile before | Compile after |
|---|---|---|---|---|---|---|
| form/bilinear.jl | Bilinear forms | 138.19 | 84.15 | -54.04 | 195.44 | 143.00 |
| form/bilinear.jl | bilinear: restricted sum replay width | 0.42 | 0.45 | +0.03 | 0.42 | 0.45 |
| form/bilinear.jl | bilinear: matrix-free form call | 3.11 | 3.48 | +0.37 | 3.10 | 3.47 |
| form/bilinear.jl | pair walk, unequal block counts | 0.42 | 0.43 | +0.01 | 0.42 | 0.43 |
| form/bilinear.jl | zeroing a non-CSC sparse matrix | 0.03 | 0.03 | 0.00 | 0.03 | 0.03 |
| form/bilinear.jl | bilinear: traversal and policy traits | 0.02 | 0.02 | 0.00 | 0.02 | 0.02 |
| form/bilinear.jl | symbolic and source-only leaves | 0.06 | 0.06 | 0.00 | 0.06 | 0.06 |
| form/bilinear.jl | source stencils, non-uniform mesh | 0.29 | 0.29 | 0.00 | 0.28 | 0.29 |
| ghost/backends.jl | Ghost spaces keep the backend | 3.38 | 3.44 | +0.06 | 3.54 | 3.59 |
| ghost/backends.jl | Ghost forms store the backend's types | 9.86 | 2.56 | -7.30 | 10.24 | 2.66 |
| ghost/backends.jl | constrained keeps the backend | 46.81 | 5.26 | -41.55 | 49.21 | 5.41 |
| ghost/backends.jl | Mesh edits keep the backend | 0.29 | 0.35 | +0.06 | 0.49 | 0.43 |
| ghost/backends.jl | A ghost mesh has an immutable walk state | 0.99 | 1.08 | +0.09 | 1.12 | 1.22 |
| ghost/backends.jl | Ghost constructors are inferable | 3.58 | 5.16 | +1.58 | 3.66 | 5.24 |
| form/linear_restriction.jl | Linear forms: restrictions | 60.58 | 31.94 | -28.64 | 61.33 | 32.74 |
| form/matrix_free.jl | matrix-free operator (#326) | 53.57 | 47.75 | -5.82 | 61.42 | 55.93 |
| form/kronecker_edge.jl | Kronecker edge cases | 46.48 | 36.45 | -10.03 | 56.25 | 47.05 |
| form/nested_operators.jl | Nested stencil operators in forms (#287) | 25.09 | 27.48 | +2.39 | 25.12 | 27.50 |
| form/nested_operators.jl | Δₕ nested in forms (#774) | 13.23 | 7.98 | -5.25 | 13.29 | 8.05 |
| form/nested_operators.jl | jumpₙ nested in inner_Γ (#205) | 14.78 | 6.97 | -7.81 | 14.76 | 6.96 |
| ghost/allocations.jl | Warm ghost calls allocate (#785) | 10.88 | 12.42 | +1.54 | 11.03 | 12.85 |
| ghost/allocations.jl | Warm refill follows a coefficient | 0.60 | 0.61 | +0.01 | 0.68 | 0.69 |
| ghost/allocations.jl | Warm refill, padded corner rule | 1.16 | 1.27 | +0.11 | 1.16 | 1.27 |
| ghost/allocations.jl | Warm refill, facet rule (#813) | 22.57 | 11.07 | -11.50 | 25.39 | 11.06 |
| ghost/allocations.jl | Warm refill on polygon spaces (new) | - | 0.31 | new | - | 0.31 |
| space/ghost_vector.jl | GhostVector holds the caller's vector | 0.67 | 0.38 | -0.29 | 0.70 | 0.41 |
| space/ghost_vector.jl | update_ghosts! equals lift (#776) | 17.52 | 16.92 | -0.60 | 17.47 | 16.91 |
| space/ghost_vector.jl | GhostVector refusals (#776) | 0.07 | 0.07 | 0.00 | 0.06 | 0.07 |
| space/ghost_vector.jl | update_ghosts! under CpuThreaded (#776) | 16.27 | 11.25 | -5.02 | 20.64 | 14.13 |
| ghost/traction.jl | Traction conditions through ghost points | 9.02 | 9.90 | +0.88 | 8.33 | 9.20 |
| space/ghost_vector_operators.jl | operators on a GhostVector (#776) | 5.48 | 5.89 | +0.41 | 5.46 | 5.87 |
| space/ghost_vector_operators.jl | GhostVector operator refusals (#776) | 0.11 | 0.12 | +0.01 | 0.09 | 0.12 |
| space/ghost_vector_operators.jl | GhostVector operators, CpuThreaded | 2.74 | 6.46 | +3.72 | 4.28 | 8.99 |
| form/linear.jl (left alone) | Linear forms | 13.42 | 13.71 | +0.29 | 45.69 | 46.77 |
| ghost/cross_component.jl (left alone) | Cross-component ghost relations | 49.45 | 52.80 | +3.35 | 49.17 | 52.29 |

Two things the testset rows show that the file rows hide:

- `form/bilinear.jl`: the "Bilinear forms" testset fell 54 s, but the file-level time outside
  every testset stayed at about 75 s (74.1 s before, 75.8 s after). That part is still
  unresolved.
- `form/nested_operators.jl`: the "#287" testset rose 2.4 s while the "#774" and "#205"
  testsets fell 13 s. The file's total fell 10.7 s.

### Drift on untouched files, and compile moves

Across the 44 files of at least 5 s that no test diff touches, the median wall ratio is 1.040
(range 0.96 to 1.28). That is the noise floor for this run under load. In the examples
below, the rise is mostly compile. Files with no test diff since the baseline include
`ghost/preconditioners.jl` (self +4.10 s, compile +3.68 s), `ghost/cross_component.jl`
(compile +3.12 s), `ghost/policies.jl` (compile +2.45 s) and `ghost/multigrid.jl`
(compile +1.95 s).

The precompile workload grew since the baseline, from S3 and from the #832 polygon work. The
`precompile` line in the log rose 0.8 s, from 1.9 s to 2.7 s; the +3.6 s figure from S3 does
not show in that line, so it is not counted here. Compile can move between files as the
workload changes, and that may explain part of the drift above. Only test diffs are
attributed to the trims.

### Not yet measured

The gate's own unit wall time comes from `/bramble-gate unit` in §9. The figures above are
S5.11's measurement under load, and are not the speedup claim. Re-measure on an idle machine
before quoting a speedup (`bramble-verification`).
