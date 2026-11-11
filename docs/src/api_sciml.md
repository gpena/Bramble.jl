```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Scientific computing: SciML, AD and solvers

Time integration, differentiable linear solves, and the sparse and iterative solver
backends layered on top of the [form API](api.md).

## Time-dependent problems and the SciML stack

`semidiscretize` applies the method of lines to a spatial [`BilinearForm`](@ref) and a
source [`LinearForm`](@ref), producing the system `M uₕ' = F(t) - A uₕ` as a callable with
the `(du, u, p, t)` signature a time stepper expects. Dirichlet conditions become algebraic
rows of a singular mass matrix, so the result is an index-1 differential-algebraic system
needing only the boundary data `g`, never its time derivative (see
[the heat equation example](examples/heat_equation.md)).

Nothing in this group needs a weak dependency except the last four and `solve(a, l)`
below, which all require [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl).
`ode_function` and `ode_problem` build a SciMLBase `ODEFunction` and `ODEProblem` and need
nothing else. `linear_problem` builds a `LinearProblem` for `LinearSolve`, and
`nonlinear_problem` a `NonlinearProblem` for `NonlinearSolve`.

Bramble defines no time stepper and exports no `solve` for an ODE problem. The `solve` that
steps an `ode_problem` comes from the package that provides the stepper: load
`OrdinaryDiffEqBDF` for `FBDF` and `QNDF`, or `OrdinaryDiffEqRosenbrock` for the Rosenbrock
methods, and call `solve(ode_problem(...), FBDF())`, as
[the heat equation example](examples/heat_equation.md) does. With Dirichlet rows ``M`` is
singular and the stepper must accept that: `FBDF`, `QNDF` and the Rosenbrock methods do, an
explicit method does not. Without Dirichlet constraints, `semidiscretize_rhs` gives a
right-hand side for explicit steppers.

`solve(a::BilinearForm, l::LinearForm; dirichlet, dirichlet_components, symmetrize, solver,
preconditioner)` extends `SciMLBase.solve`: it assembles, solves with `LinearSolve` and
returns a [`VectorElement`](@ref). `preconditioner` is `:amg`, `:ilu0` or an object with
`ldiv!`. `element(Wₕ, sol)` and `VectorElement(sol, Wₕ)` do the unwrapping step alone, for a
`LinearSolution` already in hand.

```@docs
semidiscretize
Semidiscretization
semidiscretize_rhs
SemidiscretizeRHS
mass_matrix
operator_matrix
Bramble.jacobian!
jacobian_prototype
ode_function
ode_problem
linear_problem
nonlinear_problem
```

## Differentiable linear solve (adjoint gradients)

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

## Zero-fill ILU preconditioning for convection-dominated systems

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

Bramble provides dedicated, first-class extensions for high-performance sparse linear solvers.
- **SuiteSparse** gives CHOLMOD Cholesky for symmetric positive-definite systems and UMFPACK LU for unsymmetric systems via `SuiteSparse.jl`, plus SPQR sparse QR (below) which needs only `SparseArrays`.
- **Apple Accelerate** gives native macOS `libSparse` Cholesky, $\mathrm{LDL}^T$, LUTPP and sparse QR via `AppleAccelerate.jl` (on Apple Silicon / darwin).
- **MUMPS** is a parallel multifrontal direct solver for large 2D/3D systems via `MUMPS.jl`.
- **Sparspak** is a pure-Julia sparse direct LU (George & Liu's Waterloo package) via `Sparspak.jl` with no binary dependency, so it factors matrices whose entries are `BigFloat` or a `ForwardDiff.Dual` directly. The other three take only BLAS floating-point types (MUMPS and Accelerate accept `Float32` as well as `Float64`). With `ForwardDiff` loaded, the `BrambleForwardDiffExt` extension lets [`pde_solve`](@ref) solve a `Dual` matrix or load on every backend: it factors the `Float64` values once, by the route the `Float64` solve takes, and solves each partial against that factor. A `Dual` matrix passed to a backend's own factorisation still fails.

All four solvers support symbolic reuse via the unified [`refactor!`](@ref) driver for transient PDE time loops and Newton iterations.

Of the names documented below, the four factorization types, `suitesparse_factorize`,
`sparse_factorize` and `refactor!` are `public`. The other `suitesparse_*`, `accelerate_*`,
`mumps_*` and `sparspak_*` functions, and `semidiscretize_rhs`, are neither exported nor
`public`: call them as `Bramble.suitesparse_solve` and so on.

```@docs
sparse_factorize
refactor!
```

### SuiteSparse solver

`suitesparse_factorize`/`suitesparse_solve` accept the same ordering and pivoting
parameters as Julia's own `cholesky`/`lu` on a `SparseMatrixCSC`: a fill-reducing `perm`
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

### SPQR sparse QR (least-squares and rectangular systems)

`suitesparse_qr_factorize`/`suitesparse_qr_solve` wrap `SparseArrays.SPQR.qr` for
overdetermined least-squares systems and the rectangular blocks of a constrained
saddle-point form, so `A` need not be square. Unlike the rest of this section they need only
`SparseArrays`, already a dependency of Bramble, so no `using SuiteSparse` is required.

```@docs
suitesparse_qr_factorize
suitesparse_qr_solve
```

### Apple Accelerate solver on macOS

Accelerate serves the symmetric factorisations (Cholesky and LDLᵀ); [`pde_solve`](@ref)'s
docstring states the dispatch. `:default` never routes an unsymmetric system to it
(gpena/Bramble.jl#246). The extension is used only when `Sys.isapple()` and
`Base.get_extension(Bramble, :BrambleAppleAccelerateExt) !== nothing`, so `using
AppleAccelerate` on Linux or Windows still resolves `:default` to `A \ F`.

vecLib has its own thread pool, controlled by `AppleAccelerate.jl`'s
`BLAS_THREADING_SINGLE_THREADED` and `BLAS_THREADING_MULTI_THREADED`. Bramble does not set
it; on a heavily loaded machine you can force the single-threaded mode yourself.

Two calls into the same factorisation are not always bit-identical, because vecLib reorders
floating-point reductions: compare with `isapprox`, never `==`.

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

### Sparspak sparse direct solver (pure Julia)

```@docs
SparspakFactorization
sparspak_factorize
sparspak_solve
sparspak_refactor!
```

## Caching a coefficient-dependent assembly by element type

A Newton residual generic over `T` (`Float64` on a plain call, `ForwardDiff.Dual` while an
AD backend's sparse Jacobian sweep is probing it) cannot preallocate one matrix the way a
Picard loop can. `type_cached_assemble!` gives the sparsity pattern a place to live per
element type it is ever reached at instead, so only the very first call at a given type
pays for it.

```@docs
type_cached_assemble!
```

## JuliaSparse ecosystem evaluation

[gpena/Bramble.jl#244](https://github.com/gpena/Bramble.jl/issues/244) asked whether other
packages in the [JuliaSparse](https://github.com/JuliaSparse) organization and its
neighbours are worth adopting for assembly, direct solves, or iterative preconditioning.
Each candidate was installed and measured directly against Bramble's own functions. The
verdicts are in the summary below.

`Metis.jl`'s nested-dissection ordering gave less fill and a faster CHOLMOD factorization
than CHOLMOD's default AMD on a 3D Poisson system; `SymRCM.jl` (Cuthill-McKee, which
minimises bandwidth, not fill) was worse on both. Both orderings reach
`suitesparse_factorize`/`sparse_factorize` with no new extension:
[gpena/Bramble.jl#248](https://github.com/gpena/Bramble.jl/issues/248) forwards a `perm`
keyword straight to CHOLMOD:

```julia
using Metis
import Bramble: suitesparse_factorize
perm, _ = Metis.permutation(A)
fact = suitesparse_factorize(A; sym = :spd, perm = Int.(perm))
```

### Summary

| Package | Verdict |
|:--- |:--- |
| `SparseMatricesCOO.jl` | Not adopted: wrong tool for this use, and Bramble's own path is already faster |
| `Sparspak.jl` | Done: [#247](https://github.com/gpena/Bramble.jl/issues/247) |
| `Pardiso.jl` | Not adopted: no usable backend without a separate license, same shape as [#245](https://github.com/gpena/Bramble.jl/issues/245) |
| `Krylov.jl` | Already available via `solve`'s `solver` keyword |
| `IncompleteLU.jl` | Works, but `ILUZero.jl` dominates it here |
| `ILUZero.jl` | Done: [`ilu_preconditioner`](@ref), [#255](https://github.com/gpena/Bramble.jl/issues/255) |
| `Metis.jl` | **Recommended**: genuine fill/time win on 3D systems, usable today via existing `perm` forwarding |
| `SymRCM.jl` | Not adopted: worse fill than the CHOLMOD default on the 3D Poisson system measured |
