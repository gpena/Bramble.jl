# Measurements moved out of `docs/src/internals/csr_solvers.md`

## Performance: CSC vs. CSR for this fallback path

The commit `7c901266` figure measured a bare `A \ F` on a raw, unconverted `SparseMatrixCSR`
(falling through to a generic, non-CSC-optimized `\` implementation) -- a different code
path from the one built here, so it is not reused below. This is a fresh measurement of the
actual conversion-then-delegate fallback.

**Method**: 2D unit-square Poisson (`-Δu = f`, homogeneous Dirichlet, symmetrized),
assembled once via `Bramble.backend()` (CSC) and once via `csr_backend()` (CSR), both
solved through `pde_solve`/`sparse_factorize` with `:default` (SuiteSparse). Median of 11
samples per call, both paths warmed (JIT-compiled) before timing, load-gated
(`bramble-verification` §9: 1-minute load average confirmed under half `Sys.CPU_THREADS`
== 4, both immediately before each run) and run under `caffeinate -i` to prevent idle sleep
mid-measurement.

**Date**: 2026-09-22. **Power**: battery (user-approved for this run; the figure would
otherwise be re-measured on AC).

| ndofs | nnz (CSC) | `pde_solve` CSC | `pde_solve` CSR (fallback) | ratio | bare conversion |
|---|---|---|---|---|---|
| 3,600 | 17,760 | 2.270 ms | 2.383 ms | **1.05x** | 0.119 ms |
| 22,500 | 111,900 | 13.503 ms | 13.964 ms | **1.03x** | 1.053 ms |

`sparse_factorize` alone (no solve) shows the same shape: 1.06x and 1.08x respectively.
The O(nnz) conversion itself is a small, roughly constant fraction of the total (about
5-8% of the solve time at both sizes measured) -- the fallback is not a "convert to dense
and hope" tax, it costs close to what the CSC solve costs plus one cheap triplet re-layout.
This is consistent with there being no algorithmic reason for CSR to be slower once the
matrix reaches an actual solver: SuiteSparse only ever sees a `SparseMatrixCSC` either way.

