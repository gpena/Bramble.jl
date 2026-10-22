# Unpublished documentation notes

Documenter builds only `docs/src`, so nothing here is published. The published docs are
kept to a small set a maintainer can review: home, getting started, the discrete-foundations
and forms tutorials, three worked examples (`poisson_linear`, `convection_diffusion_linear`,
`heat_equation`), the API reference, and the internals pages reduced to their docstring
blocks. Everything else lives here:

- **Pages moved whole**, at the path they had under `docs/src`: `tutorials/solvers.md`,
  `solvers_by_problem.md`, `time_stepping.md`, `backend.md`, `plotting.md`,
  `vtk_export.md`, `pgfplots_export.md`, `autodiff.md`, `operator_accuracy.md`, `interpolation.md`; `internals/autodiff.md`,
  `internals/gpu.md`; and `benchmarks.md` (still written by `docs/generate_benchmarks.jl`,
  which `docs/make.jl` no longer calls).
- **Internals prose**: `internals/<page>.md` holds each internals page as it stood before
  being reduced to its `@autodocs`/`@docs` blocks.
- **Level-set design**: `internals/levelset.md` records the design of level-set domains
  (#834) on its five points, the thinness limit and the measured orders;
  `examples/poisson_levelset_convergence.md` plots those orders, written by the generator
  `examples/poisson_levelset_convergence.jl` beside it (as `poisson_polygon_convergence`
  does for polygons).
- **Sections moved out of pages**: `api_sciml.md` (adjoint, AMG, ILU, matrix-free and
  geometric-multigrid prose; the `@docs` blocks stay published) and `*.sections.md` (the
  GPU, matrix-free and preconditioner sections taken out of a page before the whole page
  moved).
- **Machine-dependent figures**: `*.measurements.md` hold the timings, speedup ratios,
  crossover sizes and memory sizes taken out of the page with the same name, and
  `docstrings.measurements.md` those taken out of docstrings.
- **CI run time**: `ci-run-time.md` profiles the `Forms` testset per file in a warm CI run,
  gives the cause, the shard boundary and the OpenEXR cache finding (gpena/Bramble.jl#480).

The other worked examples stay in `docs/src/examples/` as `.jl` scripts, where the
`examples` test group runs them; `docs/make.jl` no longer turns them into pages.

The figures are single-machine measurements, not tracked baselines: re-measure before
relying on them. Tracked baselines live in `benchmark/baselines/`.
