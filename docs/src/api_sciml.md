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

Nothing in this group needs a weak dependency except the last four, which name their results
the way SciMLBase does: `ode_function` and `ode_problem` hand the semidiscretisation to
`OrdinaryDiffEq`, `linear_problem` hands a steady linear system to `LinearSolve` with its
factorisations and preconditioners, and `nonlinear_problem` hands a steady nonlinear residual
to `NonlinearSolve`. All four require [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl).

`solve(a::BilinearForm, l::LinearForm; ...)` is a further convenience defined alongside
`linear_problem`: assemble, solve and unwrap the `LinearSolve` solution into a
[`VectorElement`](@ref) in one call, for a caller who wants the solved grid function
directly rather than the raw `LinearProblem`. `element(Wₕ, sol)`/`VectorElement(sol, Wₕ)`
do the unwrapping step alone, for a `LinearSolution` already in hand.

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

## Adjoint sensitivities for a transient solve

```@docs
Bramble.adjoint_sensitivities
```

## Second-order wave problems

`semidiscretize_second_order` is the second-order-in-time counterpart of `semidiscretize`:
from a stiffness [`BilinearForm`](@ref) and a source [`LinearForm`](@ref), it produces
`M üₕ + C u̇ₕ + K uₕ = F(t)`, and `second_order_ode_problem`/`second_order_ode_function`
hand that to `OrdinaryDiffEq` as a `SecondOrderODEProblem` -- state `(v, u)`, velocity then
displacement. Dirichlet conditions constrain the displacement `u` the same way `semidiscretize`
constrains its own state, and the consistent velocity follows from differentiating that
constraint in time rather than being prescribed separately. Explicit/symplectic solvers
(`VelocityVerlet` and similar) cannot be used at all -- see
[`SecondOrderSemidiscretization`](@ref)'s docstring for why.

```@docs
semidiscretize_second_order
SecondOrderSemidiscretization
damping_matrix
stiffness_matrix
block_mass_matrix
second_order_ode_function
second_order_ode_problem
```

## Differentiable linear solve (adjoint gradients)

```@docs
pde_solve
```

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
- **Apple Accelerate** gives native macOS `libSparse` Cholesky, $\mathrm{LDL}^T$, and LUTPP via `AppleAccelerate.jl` (on Apple Silicon / darwin).
- **MUMPS** is a parallel multifrontal direct solver for large 2D/3D systems via `MUMPS.jl`.
- **Sparspak** is a pure-Julia sparse direct LU (George & Liu's Waterloo package) via `Sparspak.jl` -- zero binary dependency, so it factors matrices whose entries are `BigFloat` or a `ForwardDiff.Dual`, which the other three backends cannot: they take only BLAS floating-point types (MUMPS and Accelerate accept `Float32` as well as `Float64`).

All four solvers support symbolic reuse via the unified [`refactor!`](@ref) driver for transient PDE time loops and Newton iterations.

```@docs
sparse_factorize
refactor!
```

### SuiteSparse solver

`suitesparse_factorize`/`suitesparse_solve` accept the same ordering and pivoting
parameters as Julia's own `cholesky`/`lu` on a `SparseMatrixCSC` -- a fill-reducing `perm`
for CHOLMOD, or a column ordering `q` and an 8-element `control` vector for UMFPACK -- and
forward them unchanged, so `suitesparse_factorize(A; sym = :spd, perm = my_ordering)`
reaches CHOLMOD's own ordering routine rather than Bramble's default.

```@docs
SuiteSparseFactorization
suitesparse_factorize
suitesparse_solve
suitesparse_refactor!
```

### SPQR sparse QR (least-squares and rectangular systems)

`suitesparse_qr_factorize`/`suitesparse_qr_solve` wrap `SparseArrays.SPQR.qr` for
overdetermined least-squares systems and the rectangular blocks of a constrained
saddle-point form -- `A` need not be square. Unlike the rest of this section these need
only `SparseArrays`, already a dependency of Bramble, so no `using SuiteSparse` is required.

```@docs
suitesparse_qr_factorize
suitesparse_qr_solve
```

### Apple Accelerate solver on macOS

[gpena/Bramble.jl#142](https://github.com/gpena/Bramble.jl/issues/142) asked whether
`AppleAccelerate.jl` is worth wiring in on macOS. It is, but only for the symmetric
factorisations it actually wins on -- [`pde_solve`](@ref)'s own docstring states the
narrowed dispatch; this section answers the issue's four questions from measurement.

**Speedup.** Against `A \ F`, what `:default` did before Accelerate existed, Accelerate wins
on symmetric systems (Cholesky and LDLᵀ). Its unsymmetric LUTPP is slower than SuiteSparse,
which is why `:default` never routes an unsymmetric system to Accelerate
(gpena/Bramble.jl#246). `benchmark/accelerate_solvers.jl` measures both on your machine.

**Extension scoping.** Settled: the guard is `Sys.isapple() &&
Base.get_extension(Bramble, :BrambleAppleAccelerateExt) !== nothing`, so `using
AppleAccelerate` on Linux or Windows still resolves `:default` to `A \ F` and Accelerate
never becomes a hard dependency of Bramble. The test environment installs it on every
platform but loads it only under `Sys.isapple()`.

**Threading.** `AppleAccelerate.jl` exports `BLAS_THREADING_MULTI_THREADED` and
`BLAS_THREADING_SINGLE_THREADED`, the knob for vecLib's own internal thread pool, alongside
a setter that reads vecLib's threading API directly. No dedicated measurement isolated
vecLib threading against Bramble's own `Threads.@threads`/`@batch` assembly sweeps running
concurrently: the benchmarks behind the speedup claim above ran at `--threads=4`
(factorisation comparison) and `--threads=2` (dispatch-narrowing check) without symptoms
attributable to thread contention, but that is not the same as a study built to detect it.
A caller who suspects contention on a heavily loaded machine can force vecLib to
`BLAS_THREADING_SINGLE_THREADED` explicitly; Bramble does not set this itself.

**Accuracy vs. OpenBLAS.** Audited against `LinearAlgebra`'s own factorisations across
sparse SPD/LDLᵀ/QR/LUTPP and the dense `accelerate_factorize` kinds: the worst observed
relative residual was `3.637...e-14` (sparse SPD Cholesky via Accelerate), and a dense QR
least-squares case matched to `0.0`. Every `atol` in the test suite guarding a solve is
`1.0e-12`, two orders of magnitude looser than that residual, so no tolerance needed
tightening or loosening. Two calls into the same factorisation are not always bit-identical
-- vecLib reorders floating-point reductions across calls -- so compare with `isapprox`,
never `==`.

**A `sym` caveat.** Under `:default`, an unrecognised `sym` (anything other than
`:auto`/`:spd`/`:definite`/`:symmetric`/`:unsymmetric` and their integer aliases) silently
falls back to `A \ F` rather than raising, matching `:default`'s pre-Accelerate behaviour of
ignoring `sym` entirely. This is looser than `solver = :accelerate`, which validates `sym`
and throws on an unrecognised value.

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
| `SparseMatricesCOO.jl` | Not adopted -- wrong tool for this use, and Bramble's own path is already faster |
| `Sparspak.jl` | Done -- [#247](https://github.com/gpena/Bramble.jl/issues/247) |
| `Pardiso.jl` | Not adopted -- no usable backend without a separate license, same shape as [#245](https://github.com/gpena/Bramble.jl/issues/245) |
| `Krylov.jl` | Already available via `solve`'s `solver` keyword |
| `IncompleteLU.jl` | Works, but `ILUZero.jl` dominates it here |
| `ILUZero.jl` | Done -- [`ilu_preconditioner`](@ref), [#255](https://github.com/gpena/Bramble.jl/issues/255) |
| `Metis.jl` | **Recommended** -- genuine fill/time win on 3D systems, usable today via existing `perm` forwarding |
| `SymRCM.jl` | Not adopted -- worse fill than the CHOLMOD default on the 3D Poisson system measured |
