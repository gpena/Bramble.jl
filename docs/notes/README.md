# Unpublished documentation notes

Documenter builds only `docs/src`, so nothing here is published. The published docs are
home, getting started, eight tutorials (geometry, mesh, space, operators, forms, ghost
points, coupled systems, automatic differentiation), five how-to guides (pure Neumann,
time-dependent data, solvers, backends, plotting and export), fifteen worked examples, the
API reference with its solver and SciML pages, and the internals pages reduced to their
docstring blocks. Everything else lives here:

- **Pages moved whole**, at the path they had under `docs/src`: `tutorials/solvers.md`,
  `solvers_by_problem.md`, `time_stepping.md`, `backend.md`, `plotting.md`,
  `vtk_export.md`, `pgfplots_export.md`, `autodiff.md`, `operator_accuracy.md`, `interpolation.md`; `internals/autodiff.md`,
  `internals/gpu.md`; and `benchmarks.md` (still written by `docs/generate_benchmarks.jl`,
  which `docs/make.jl` no longer calls).
- **Internals prose**: `internals/<page>.md` holds each internals page as it stood before
  being reduced to its `@autodocs`/`@docs` blocks.
- **Sections moved out of pages**: `api_sciml.md` (adjoint, AMG, ILU, matrix-free and
  geometric-multigrid prose; the `@docs` blocks stay published) and `*.sections.md` (the
  GPU, matrix-free and preconditioner sections taken out of a page before the whole page
  moved).
- **Machine-dependent figures**: `*.measurements.md` hold the timings, speedup ratios,
  crossover sizes and memory sizes taken out of the page with the same name, and
  `docstrings.measurements.md` those taken out of docstrings.
- **Level-set domains** (gpena/Bramble.jl#834): `internals/levelset.md` is the design
  record (the class of curves a fitted grid exists for, the closure and orbit paths, the
  named errors, the impossibility result, the accepted limits), and
  `examples/poisson_levelset_convergence.md` the measured convergence as log-log plots. The
  worked page for it, `poisson_circle`, is in `docs/src/examples/`.
- **CI run time**: `ci-run-time.md` profiles the `Forms` testset per file in a warm CI run,
  gives the cause, the shard boundary and the OpenEXR cache finding (gpena/Bramble.jl#480).

Four worked examples stay in `docs/src/examples/` as `.jl` scripts that `docs/make.jl` does
not turn into pages: `amg_preconditioning`, `matrix_free_operator`, `memory_scaling` and
`transient_inplace`. The `examples` test group runs them.

The figures are single-machine measurements, not tracked baselines: re-measure before
relying on them. Tracked baselines live in `benchmark/baselines/`.
