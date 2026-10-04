# Passages moved out of `docs/src/internals/csr_solvers.md`

- **AlgebraicMultigrid.jl, ILUZero.jl**: both already take a generic `AbstractMatrix` (AMG)
  or `SparseMatrixCSC` with no CSR-specific fast path to add, and neither is a *direct*
  solver (this issue's scope) -- they precondition an iterative solve, which is a separate
  concern from `sparse_factorize`/`pde_solve`. No CSR-specific work applies.

