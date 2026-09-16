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

## Caching a coefficient-dependent assembly by element type

A Newton residual generic over `T` (`Float64` on a plain call, `ForwardDiff.Dual` while an
AD backend's sparse Jacobian sweep is probing it) cannot preallocate one matrix the way a
Picard loop can. `type_cached_assemble!` gives the sparsity pattern a place to live per
element type it is ever reached at instead, so only the very first call at a given type
pays for it.

```@docs
type_cached_assemble!
```
