```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Forms

Linear and bilinear forms, their assembly, and the boundary conditions applied to an
assembled system. See the [forms tutorial](../tutorials/form.md).

## Building a form

```@docs
form
expression
```

## Point (Dirac) sources

`DiracSource` is private; see the [forms internals page](../internals/form.md).

```@docs
dirac
```


## Assembling

`assemble` allocates its result; the mutating forms refill one that already exists, which is
what a time loop wants. `allocate_system_matrix` builds a matrix's sparsity pattern once so
that `assemble!` can refill it without allocating.

```@docs
assemble
assemble!
assemble_parallel!
allocate_system_matrix
evaluate!
```

## Matrix-free Kronecker operators

For a separable `BilinearForm` (one whose assembled matrix is an exact sum of Kronecker
products of one-dimensional factors over a `MeshnD`), `kronecker_operator` builds a
`KroneckerLinearOperator` that applies in one fused pass over the grid instead of ever
assembling the `D`-dimensional matrix: a `200^3` mesh stores `O(200)` numbers per axis
rather than the assembled matrix's `O(200^3)` stored entries. `is_separable` checks the
condition beforehand and lists what factors.

What factors: the difference, average, jump and shift families along any axis and chains of
them, so mixed derivatives and advection terms (whose factors need not be symmetric);
`innerₕ`, `inner₊` and `inner_Γ` weights; a restriction to `:interior`; a grid-function
coefficient that varies along one axis; and scalar or `Ref` coefficients. A composite space
whose leaves share one mesh gives a block operator, one Kronecker operator per block. What
does not: a coefficient varying along several axes, other region restrictions (Dirichlet
rows included), an interpolation, a 1D mesh, trial and test spaces on different meshes, and
the star and cross-weighted differences. A grid-function coefficient is read once, when the
operator is built, and warns so; use a `Ref` for a coefficient that changes, or
`matrix_free_operator`, which reads it live. Under `CpuThreaded` the product runs threaded,
and an operator whose mesh was mutated in place afterwards throws instead of applying.

`fdm_solve` requires `using Kronecker` (the `BrambleKroneckerExt` extension) and solves a
separable, constant-coefficient system by fast diagonalisation instead of a general sparse
factorisation.

```@docs
is_separable
kronecker_operator
KroneckerLinearOperator
fdm_solve
```

## Matrix-free operators

`matrix_free_operator` applies any `BilinearForm` that `assemble` accepts, separable or not,
without building its matrix: `mul!(y, op, x)` walks the form's stencil and agrees with
`assemble(a) * x`, Dirichlet rows included, on vectors and `VectorElement`s. Its execution
policy comes from the trial space, as for assembly. Preconditioners and multigrid built on it
are in [Scientific computing](../api_sciml.md), and the
[solvers tutorial](../tutorials/solvers.md) has measured time and memory against sparse
matrix-vector products.

```@docs
matrix_free_operator
MatrixFreeOperator
```

## Additive accumulation

`assemble_add!` adds a form's contribution to a matrix or vector that already holds
something, without the `fill!` `assemble!` does first -- for `M/Δt + θK`-style operators
built from several independently assembled pieces. See its own docstring for why this
needed no new traversal, and for the Dirichlet-ordering note.

```@docs
assemble_add!
```

## Jacobian sparsity

For a Newton residual built from a [`BilinearForm`](@ref) with a live nonlinear
coefficient (see [the nonlinear Poisson example](../examples/poisson_nonlinear.md)),
`jacobian_pattern` reads the Jacobian's sparsity pattern directly off the form's AST,
without AD tracing. `ast_sparsity_detector` wraps it as an
`ADTypes.AbstractSparsityDetector`, ready to hand `AutoSparse` directly (requires
[ADTypes.jl](https://github.com/SciML/ADTypes.jl)).

```@docs
jacobian_pattern
ast_sparsity_detector
```

Time integration, differentiable linear solves, and the sparse/iterative solver backends
move to their own [scientific computing reference](../api_sciml.md): the SciML stack
(`semidiscretize`, `ode_problem`, `linear_problem`, `nonlinear_problem`, ...), second-order
wave problems, `pde_solve`'s adjoint rule, AMG preconditioning, the SuiteSparse/Apple
Accelerate/MUMPS direct solvers, and `type_cached_assemble!`.

## Bandwidth analysis

`bandwidths`/`blockbandwidths` are private; see the
[forms internals page](../internals/form.md).

## Dirichlet conditions

```@docs
DirichletConstraint
dirichlet_constraints
dirichlet_bc!
symmetrize!
```

## Boundary flux / reaction extraction

`reaction` recovers the flux a Dirichlet constraint had to supply, from the
**unconstrained** operator and load (`assemble` with no `dirichlet` keyword) and an
already-solved `uₕ` -- the constrained assembly has already overwritten those rows by the
time a solution exists. `reaction_density` is its pointwise counterpart, a grid function
suitable for `export_vtk`.

```@docs
reaction
reaction!
reaction_density
reaction_density!
```

## Structural properties

`issymmetric(::BilinearForm)`/`isposdef(::BilinearForm)` are private; see the
[forms internals page](../internals/form.md).
