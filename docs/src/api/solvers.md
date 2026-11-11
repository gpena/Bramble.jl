```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Solvers and preconditioners

The linear solve `pde_solve`, the sparse direct solvers, and the preconditioners and
geometric multigrid that serve iterative solves. Time integration and the SciML interface
are in [Scientific computing](../api_sciml.md).

Each backend loads with its package: `AlgebraicMultigrid` for `amg_preconditioner`,
`ILUZero` for `ilu_preconditioner`, and `SuiteSparse`, `AppleAccelerate`, `MUMPS` or
`Sparspak` for the matching direct solver.

## Linear solve

```@docs
pde_solve
Bramble.ghost_kernel
Bramble.neumann_compatibility
```

`ghost_kernel` and `neumann_compatibility` serve the pure-Neumann problems of a ghost space.
Both are `public` and not exported: write `Bramble.ghost_kernel`.

## Algebraic multigrid preconditioning

```@docs
amg_preconditioner
```

## Zero-fill ILU preconditioning

```@docs
ilu_preconditioner
```

## Matrix-free preconditioners

```@docs
Bramble.AbstractMatrixFreePreconditioner
jacobi_preconditioner
Bramble.JacobiPreconditioner
chebyshev_preconditioner
Bramble.ChebyshevPreconditioner
fdm_preconditioner
Bramble.FDMPreconditioner
Bramble.max_eigenvalue_estimate
```

## Geometric multigrid

### Mesh hierarchy and transfers

```@docs
GeometricMeshHierarchy
prolongate!
coarsen!
```

### Smoothers

```@docs
Bramble.AbstractSmoother
Bramble.JacobiSmoother
Bramble.ChebyshevSmoother
Bramble.RedBlackGaussSeidel
jacobi_smoother
chebyshev_smoother
red_black_gauss_seidel
smooth!
```

### Cycles and solve

```@docs
gmg_preconditioner
Bramble.GMGPreconditioner
gmg_solve
Bramble.v_cycle!
Bramble.w_cycle!
Bramble.fmg!
```

## Sparse direct solvers and factorization reuse

Four extensions provide sparse direct solvers:

- SuiteSparse gives CHOLMOD Cholesky for symmetric positive-definite systems and UMFPACK LU
  for unsymmetric systems, plus SPQR sparse QR (below), which needs only `SparseArrays`.
- Apple Accelerate gives the native macOS `libSparse` Cholesky, ``\mathrm{LDL}^T``, LUTPP
  and sparse QR, on Apple Silicon.
- MUMPS is a parallel multifrontal direct solver for large 2D and 3D systems.
- Sparspak is a pure-Julia sparse direct LU (George and Liu's Waterloo package). It factors
  matrices whose entries are `BigFloat` or a `ForwardDiff.Dual` directly. The other three
  take only BLAS floating-point types (MUMPS and Accelerate accept `Float32` as well as
  `Float64`).

With `ForwardDiff` loaded, [`pde_solve`](@ref) solves a `Dual` matrix or load on every
backend: it factors the `Float64` values once, by the route the `Float64` solve takes, and
solves each partial against that factor. A `Dual` matrix passed to a backend's own
factorization still fails.

All four support symbolic reuse through the [`refactor!`](@ref) driver, for transient time
loops and Newton iterations.

Of the names documented below, the four factorization types, `suitesparse_factorize`,
`sparse_factorize` and `refactor!` are `public`. The other `suitesparse_*`, `accelerate_*`,
`mumps_*` and `sparspak_*` functions are neither exported nor `public`: call them as
`Bramble.suitesparse_solve` and so on.

```@docs
sparse_factorize
refactor!
```

### SuiteSparse solver

`suitesparse_factorize` and `suitesparse_solve` accept the same ordering and pivoting
parameters as Julia's own `cholesky` and `lu` on a `SparseMatrixCSC`: a fill-reducing `perm`
for CHOLMOD, or a column ordering `q` and a `control` vector (as
`SparseArrays.UMFPACK.get_umfpack_control` returns) for UMFPACK. They are forwarded
unchanged, so `suitesparse_factorize(A; sym = :spd, perm = my_ordering)` replaces CHOLMOD's
default ordering.

```@docs
SuiteSparseFactorization
suitesparse_factorize
suitesparse_solve
suitesparse_refactor!
```

### SPQR sparse QR

`suitesparse_qr_factorize` and `suitesparse_qr_solve` wrap `SparseArrays.SPQR.qr` for
overdetermined least-squares systems and the rectangular blocks of a constrained
saddle-point form, so `A` need not be square. They need only `SparseArrays`, already a
dependency of Bramble, so no `using SuiteSparse` is required.

```@docs
suitesparse_qr_factorize
suitesparse_qr_solve
```

### Apple Accelerate solver

Accelerate serves the symmetric factorizations (Cholesky and LDLᵀ); the
[`pde_solve`](@ref) docstring states the dispatch. `:default` never routes an unsymmetric
system to it (gpena/Bramble.jl#246). The extension is used only when `Sys.isapple()` and
`Base.get_extension(Bramble, :BrambleAppleAccelerateExt) !== nothing`, so `using
AppleAccelerate` on Linux or Windows still resolves `:default` to `A \ F`.

vecLib reorders floating-point reductions, so two calls into the same factorization are not
always bit-identical: compare with `isapprox`.

Under `:default`, an unrecognised `sym` (anything other than `:auto`, `:spd`, `:definite`,
`:symmetric`, `:unsymmetric` and their integer aliases) falls back to `A \ F` without
raising. `solver = :accelerate` validates `sym` and throws on an unrecognised value.

```@docs
AccelerateFactorization
accelerate_factorize
accelerate_solve
accelerate_refactor!
```

### MUMPS sparse direct solver

```@docs
MUMPSFactorization
mumps_factorize
mumps_solve
mumps_refactor!
```

### Sparspak sparse direct solver

```@docs
SparspakFactorization
sparspak_factorize
sparspak_solve
sparspak_refactor!
```
