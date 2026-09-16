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

## Second-order (wave) problems

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

`pde_solve(A, F)` is `A \ F` under a name `ChainRulesCore.rrule` can attach an adjoint rule
to -- no source-level AD tool, forward or reverse, can differentiate through `\`
itself, since it dispatches into compiled BLAS/SuiteSparse code. `assemble`/`dirichlet_bc!`
are already reverse-mode-differentiable on their own (see the
[automatic differentiation tutorial](tutorials/autodiff.md)), so wrapping only this one
function is enough to differentiate an entire `θ -> assemble(a(θ), l(θ); dirichlet = θ) ->
pde_solve -> J(u)` chain end to end, including a gradient with respect to a Dirichlet
boundary value -- the adjoint solves `Aᵀ λ = ∂J/∂u` once, reusing the forward solve's own LU
factorisation, and returns `∂J/∂A = -λ uᵀ` restricted to `A`'s sparsity (never densified) and
`∂J/∂F = λ`.

Requires [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl), and serves every
`ChainRulesCore` consumer. `Enzyme` instead reaches the same adjoint through `BrambleEnzymeExt`'s
own native `EnzymeRules` rule, which `using Enzyme` is enough to load -- do not call
`Enzyme.@import_rrule`, whose bridge returns a wrong gradient here (see [`pde_solve`](@ref)'s own
docstring). `Mooncake` is not currently supported (a gap in `Mooncake.jl`'s own sparse-array
tangent support, also documented there). See the
[inverse problem worked example](examples/inverse_diffusion.md).

```@docs
pde_solve
```

## Algebraic multigrid preconditioning

`amg_preconditioner` builds an algebraic multigrid hierarchy for a symmetric
positive-definite matrix -- typically an assembled elliptic `BilinearForm`, whose condition
number scales as `O(h^-2)` under refinement -- so that an iterative `LinearSolve` solve gets
grid-independent, `O(1)` iteration counts instead of the `O(h^-1)` an unpreconditioned Krylov
method needs. It returns the bare `MultiLevel` hierarchy; `AlgebraicMultigrid.aspreconditioner`
turns that into the object with `ldiv!` that `Pl`/`Pr` expect. `solve(a::BilinearForm,
l::LinearForm; ...)` (previous section) takes `preconditioner = :amg` directly, building and
applying that preconditioner in one call.

Requires [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl).

```@docs
amg_preconditioner
```

## ILU(0) preconditioning for convection-dominated systems

[gpena/Bramble.jl#244](https://github.com/gpena/Bramble.jl/issues/244) measured classical
algebraic multigrid failing to converge on an unsymmetric, convection-dominated system
(diffusion `1e-2` against unit advection, `ruge_stuben` capped at 2000 GMRES iterations
without converging) -- AMG assumes something close to an M-matrix, which strong advection
breaks. `ILUZero.jl`'s zero-fill incomplete LU (ILU(0)) does not share that assumption: on
the same system, GMRES took 18 iterations against 179 unpreconditioned, at a fraction of
AMG's setup cost, since ILU(0) reuses `A`'s own sparsity pattern with no fill-in parameter to
tune. `ilu_preconditioner` mirrors [`amg_preconditioner`](@ref)'s shape, but returns an
object with `ldiv!` directly -- `ILUZero.ilu0` needs no `aspreconditioner`-style wrapping the
way an AMG hierarchy does. `solve(a::BilinearForm, l::LinearForm; ...)` takes
`preconditioner = :ilu0` the same way it takes `:amg`.

**When to prefer which**: AMG's grid-independent, `O(1)` iteration count wins at scale on
elliptic, symmetric positive-definite forms (Poisson, diffusion-dominated), where its
M-matrix-like assumption holds. ILU(0) is the better default for unsymmetric,
convection-dominated forms, where AMG is this issue's own worked counter-example for why it
should not be the only option offered -- see [`amg_preconditioner`](@ref) for the elliptic
case.

Requires [ILUZero.jl](https://github.com/mohamed82008/ILUZero.jl).

```@docs
ilu_preconditioner
```

## Sparse direct solvers and factorization reuse

Bramble provides dedicated, first-class extensions for high-performance sparse linear solvers:
- **SuiteSparse**: CHOLMOD Cholesky for symmetric positive-definite systems and UMFPACK LU for unsymmetric systems via `SuiteSparse.jl`, plus SPQR sparse QR (below) which needs only `SparseArrays`.
- **Apple Accelerate**: Native macOS `libSparse` Cholesky, $\mathrm{LDL}^T$, and LUTPP via `AppleAccelerate.jl` (on Apple Silicon / darwin).
- **MUMPS**: Parallel multifrontal direct solver for large 2D/3D systems via `MUMPS.jl`.
- **Sparspak**: Pure-Julia sparse direct LU (George & Liu's Waterloo package) via `Sparspak.jl` -- zero binary dependency, so it factors matrices whose entries are `Float32`, `BigFloat`, or a `ForwardDiff.Dual`, where the other three backends require `Float64`/`ComplexF64`.

All four solvers support non-allocating symbolic reuse via the unified [`refactor!`](@ref) driver for transient PDE time loops and Newton iterations.

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

### Apple Accelerate solver (macOS)

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

## JuliaSparse ecosystem evaluation

[gpena/Bramble.jl#244](https://github.com/gpena/Bramble.jl/issues/244) asked whether other
packages in the [JuliaSparse](https://github.com/JuliaSparse) organization and its
neighbours are worth adopting for assembly, direct solves, or iterative preconditioning.
Each candidate below was installed on Julia 1.12 and measured directly against Bramble's
own functions -- never a synthetic microbenchmark standing in for them (see
`bramble-verification`) -- so a "no" here is a measured "no", not a guess. Numbers are a
single run on one machine, not a tracked baseline; treat them as directional.

### Assembly & storage formats

**`SparseMatricesCOO.jl`: not adopted.** Bramble's own assembly already skips the triplet
stage entirely: `assemble` determines the sparsity pattern once (`PatternSink`, the
lock-free colouring sweep documented in [Forms](internals/form.md)) and every subsequent call writes
straight into `nzval` via `add_to_sparse!`, never building `(I, J, V)` at all. Measured on
a 2D 60×60 Poisson system (`n = 3600`, `nnz = 17760`):

| Path | Time |
|:--- |:--- |
| Bramble `assemble` (first call, builds the pattern) | 0.37 ms |
| Bramble `assemble` (repeat call, pattern cached) | 0.37 ms |
| `Base.sparse(I, J, V)` on the identical triplets | 0.06 ms |
| `SparseMatricesCOO.jl` COO→CSC on the identical triplets | 203 **seconds** |

`SparseMatricesCOO.jl` defines no specialised `SparseMatrixCSC(::SparseMatrixCOO)`
constructor, so the conversion falls through to Julia's generic dense-iteration
`AbstractMatrix` fallback -- an `O(m \cdot n \cdot \mathrm{nnz})` scan through every
`getindex`, itself an `O(\mathrm{nnz})` linear search of the triplet arrays (confirmed by
reading `SparseMatricesCOO.jl`'s source, not assumed from the number alone). The package
is designed by [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) as an
NLP-solver interop format (handing Jacobian/Hessian triplets to IPOPT-style solvers that
want COO directly), not as a fast intermediate for building a `SparseMatrixCSC` -- the
wrong tool for what this issue asked it to do here. Bramble's first assembly is already
about as fast as its thousandth, which is the actual bar a triplet library would need to
clear.

**`SymRCM.jl`**: evaluated under reordering, below -- not for assembly.

### Direct sparse solvers

**Sparspak.jl: done, not re-evaluated here.** Built in
[gpena/Bramble.jl#247](https://github.com/gpena/Bramble.jl/issues/247); see
[Sparspak sparse direct solver (pure Julia)](@ref) above.

**`Pardiso.jl`: not adopted, for a licensing reason rather than a technical one.**
`Pardiso.jl` bridges to one of two backends, and neither is available without something
Bramble cannot bundle:
- Intel MKL PARDISO needs a separately installed MKL; `Pardiso.mkl_is_available()` is
  `false` on a plain Julia 1.12 environment, and constructing an `MKLPardisoSolver` throws
  `"MKL is not available"`.
- Panua (formerly the free academic) PARDISO needs a separately downloaded, licensed
  shared library; constructing a `PardisoSolver` throws `"Panua pardiso library was not
  loaded"`.

Both were reproduced directly (not assumed) on a fresh Julia 1.12 environment. This is the
same shape of blocker that closed
[gpena/Bramble.jl#245](https://github.com/gpena/Bramble.jl/issues/245) (`ThreadedSparseCSR.jl`)
as won't-fix: a real, verified dependency the package cannot satisfy on behalf of a user,
rather than missing integration work. A user who already holds an MKL or Panua license and
wants to use it can still call `Pardiso.jl` directly against `A`/`F` from
[`assemble`](@ref) -- nothing in Bramble stands in the way of that -- it is just not
something this package can wire up as a first-class `solver` option for everyone.

### Iterative solvers & preconditioners

Krylov methods are already reachable through `solve` with `solver =
KrylovJL_GMRES()` etc. (`BrambleSciMLExt`), and [`amg_preconditioner`](@ref) already covers
algebraic multigrid preconditioning. What #244 asked to evaluate is whether `ILUZero.jl` /
`IncompleteLU.jl` add anything beyond that. Measured on an unsymmetric 2D convection-diffusion
system (90×90 grid, `n = 8100`, diffusion `1\mathrm{e}{-2}` against unit advection in both
directions -- the convection-dominated regime the issue named), unrestarted GMRES to
`atol = rtol = 1\mathrm{e}{-10}`:

| Preconditioner | Time | Iterations | Converged |
|:--- |:--- |:--- |:--- |
| none | 51.4 ms | 179 | yes |
| AMG (`ruge_stuben`) | 6475.9 ms | 2000 (capped) | **no** |
| `IncompleteLU.jl` (τ = 0.01) | 19.6 ms | 95 | yes |
| `ILUZero.jl` (ILU(0)) | 4.5 ms | 18 | yes |

Classical algebraic multigrid assumes something close to an M-matrix and does not fail
gracefully once advection dominates diffusion this strongly -- it neither converges nor
finishes quickly here, which is a known limitation of `ruge_stuben`-style coarsening on
non-symmetric, convection-dominated operators, not a bug in `AlgebraicMultigrid.jl`.
`ILUZero.jl`'s zero-fill ILU(0), reusing `A`'s own sparsity pattern, is the clear winner:
about 11× fewer iterations and 11× less wall time than no preconditioner, and 4× less than
`IncompleteLU.jl`'s drop-tolerance variant, at a fraction of the setup cost either of the
others carries. Built as [`ilu_preconditioner`](@ref) in
[gpena/Bramble.jl#255](https://github.com/gpena/Bramble.jl/issues/255), mirroring
[`amg_preconditioner`](@ref)'s shape -- see "ILU(0) preconditioning for convection-dominated
systems" above.

`Metis.jl`'s graph partitioning was evaluated under reordering, not as a preconditioner,
below.

### Fill-reducing reordering

Measured on a 3D 24×24×24 Poisson system (`n = 13824`, `nnz = 93312`), CHOLMOD Cholesky
factorization with three orderings:

| Ordering | Factor time | `nnz(L)` |
|:--- |:--- |:--- |
| CHOLMOD default (built-in AMD) | 25.0 ms | 2,147,132 |
| `Metis.jl` (nested dissection) | 20.1 ms | 1,654,868 |
| `SymRCM.jl` (Cuthill-McKee) | 53.3 ms | 4,768,508 |

`Metis.jl`'s nested-dissection ordering measurably beats CHOLMOD's own default AMD here --
about 20% less factorization time and 23% less fill -- a genuine, reproducible win on a 3D
system. `SymRCM.jl` is worse on both counts: Cuthill-McKee minimises bandwidth, not fill,
and 3D discretizations are exactly where that distinction costs the most. Both orderings
reach `suitesparse_factorize`/`sparse_factorize` **today, with no new extension needed** --
[gpena/Bramble.jl#248](https://github.com/gpena/Bramble.jl/issues/248) already forwards a
`perm` keyword straight to CHOLMOD:

```julia
using Metis
perm, _ = Metis.permutation(A)
fact = suitesparse_factorize(A; sym = :spd, perm = Int.(perm))
```

`Metis.jl` is worth naming explicitly in the ordering documentation rather than building
anything further for it.

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
| `SymRCM.jl` | Not adopted -- worse fill than the CHOLMOD default on the systems Bramble assembles |

## Caching a coefficient-dependent assembly by element type

A Newton residual generic over `T` (`Float64` on a plain call, `ForwardDiff.Dual` while an
AD backend's sparse Jacobian sweep is probing it) cannot preallocate one matrix the way a
Picard loop can. `type_cached_assemble!` gives the sparsity pattern a place to live per
element type it is ever reached at instead, so only the very first call at a given type
pays for it.

```@docs
type_cached_assemble!
```
