# Measurements moved out of `docs/src/tutorials/backend.md`

## Intro sentence

The next section puts numbers on that.

## Measured crossovers

### Measured crossovers

A **crossover** is the smallest problem size at which a parallel policy beats `CpuSerial`
twice running. Below it, starting threads costs more than the work saved. Each crossover
below was measured per operation, not assumed, on an Apple M2, four threads (`--threads=4`),
on AC power, from `benchmark/polyester_crossover.jl` (commit `4b76d62b`). The sizes are
points per axis on a two-dimensional grid, or elements for the vector reductions.

| Operation | `CpuThreaded` | `CpuPolyester` |
|---|---|---|
| `Rₕ!` unmasked | 64-96 points per axis | 8-24 points per axis |
| `Rₕ!` masked | 256 points per axis | 16 points per axis |
| `avgₕ!` (nq=3) | 24-32 points per axis | 8 points per axis |
| `innerₕ`/`_dot` | 100,000-300,000 elements | 1,000 elements |

`_dot`/`_dot_masked(::CpuThreaded, ...)` (`src/utils/linear_algebra.jl`) are a real threaded
reduction (static chunks, dependency-free; masked reductions walk set bits per word;
`SeparableWeights` banded along the last axis), and their crossover against `CpuSerial` was
measured the same way as the rows above: 100,000-300,000 elements across four runs (commit
`560e8394`) -- the exact crossing point moved within that range from one run to the next, the
same jitter the other workloads show near their own crossing point.

The crossover differs by an order of magnitude between workloads, so a figure measured
for one does not transfer to another. That is why the table has four rows rather than a
single number. Where both policies have an entry, `CpuPolyester` crosses over earlier than
`CpuThreaded` on every operation measured: at about 3 to 16 times fewer points per axis on
the grid operations, and 100 to 300 times fewer elements for `innerₕ`/`_dot`.

These are one machine's numbers, taken under one power state, not a portable constant:
re-measure before leaning on them for a different host. The [benchmarks page](../benchmarks.md) carries the measurements across sizes.

## Measure your own crossovers: comparison with the removed table

Those lines count total degrees of freedom per dimension, so they are not directly comparable with the table above, which counts points per axis on a 2D grid.

## CSR backend

Measured on one machine with `benchmark/backends.jl` (commit `7c901266`, AC power, 1-minute load 2.13, 4 threads; the run's output is not saved in the repository):

- **Assembly is a wash**: first assembly and `assemble!` refill land within about 0.94-1.05x of each other across 1D, 2D and 3D, and neither allocates in `assemble!`.
- **CSR stores the same matrix in roughly half the memory**: 5.3 MiB against 10.7 MiB in 1D, 24.4 against 59.1 MiB in 3D.
- **CSR loses the direct solve**, 2.4x to 4.2x slower across the sizes measured, because `SparseMatricesCSR.jl` has no native CSR solve: `\` goes through a transposed factorization of a reinterpreted LU.

Reach for CSR when memory is the constraint and assembly dominates. Keep CSC when `A \ F` is on the critical path. Re-measure before relying on these figures for a specific size.

## GpuOffload

Measured against plain `CpuThreaded()` (`benchmark/gpu_offload.jl`, recorded in `benchmark/results/gpu_offload.toml`, commit `af288c31`, Apple M2, AC power, `--threads=4`): `avgₕ!` wins clearly (3.50x at 1D n=10,000,000, 11.12x at 2D 3000x3000), `Rₕ!` wins at 2D 3000x3000 (1.80x), and `Rₕ!` at 1D n=10,000,000 is even (1.03x; an earlier run at commit `2c7217e6` measured it losing, at 0.86x). Measure the specific call and grid size before choosing it over the inner policy alone. The internals page has the full table.

## Precompilation workload

It makes the package build slower and cuts the time to first result by roughly a factor of three (the estimate in `src/precompile.jl`; no recorded measurement).

