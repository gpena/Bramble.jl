# Sections moved out of `docs/src/api_sciml.md`

As they stood. The headings and `@docs` blocks stay on the published page.

## Adjoint sensitivities for a transient solve

[`Bramble.adjoint_sensitivities`](@ref) is the transient counterpart of [`pde_solve`](@ref)'s
steady-state adjoint rule (see [the API reference](api.md)): the gradient of a scalar
functional of a [`Semidiscretization`](@ref)'s solved trajectory with respect to its initial
condition and its `p`, from one backward solve regardless of how many parameters or how many
saved steps. It wraps `SciMLSensitivity.adjoint_sensitivities` and requires
[SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl) -- see [the worked
example](examples/transient_inverse_problem.md).

```@docs
Bramble.adjoint_sensitivities
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

Requires [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl). Not every
`ChainRulesCore` consumer can use it, as the next sentences explain. `Enzyme` instead reaches the same adjoint through `BrambleEnzymeExt`'s
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

## Zero-fill ILU preconditioning for convection-dominated systems

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

## Matrix-free preconditioners

AMG and ILU(0) need the assembled matrix. The preconditioners here need only a
[`matrix_free_operator`](@ref), which the [matrix-free operator page](examples/matrix_free_operator.md) explains. Each is a subtype of `Bramble.AbstractMatrixFreePreconditioner`
with `ldiv!`, so it goes straight to `Pl` in `LinearSolve`. `jacobi_preconditioner` reads the
diagonal off one walk of the form's stencil. `chebyshev_preconditioner` is a fixed polynomial
in `D⁻¹A`, Jacobi-scaled, with `D = diag(A)`: unscaled, the top of `A`'s spectrum on a
non-uniform mesh is a few small-cell outliers, and the polynomial wastes its degree on them.
Its upper bound comes from `Bramble.max_eigenvalue_estimate`, power iteration on the
operator. Both need a symmetric positive-definite `A` for conjugate gradients; with
`dirichlet`, CG needs a right-hand side that vanishes on the Dirichlet rows.

```@docs
Bramble.AbstractMatrixFreePreconditioner
jacobi_preconditioner
Bramble.JacobiPreconditioner
chebyshev_preconditioner
Bramble.ChebyshevPreconditioner
Bramble.max_eigenvalue_estimate
```

## Geometric multigrid

`gmg_preconditioner(W -> form(...), Ωₕ)` rediscretises the form on every level of a
`GeometricMeshHierarchy`, which coarsens a non-uniform mesh by 2 through every other point,
so the levels nest exactly. Multilinear `prolongate!` and its transpose `coarsen!` join the
levels, point smoothers smooth them, and the coarsest level is solved directly. Every
grid function in the form must be built from `W` inside the builder. On meshes whose cells
have bounded aspect ratio, CG preconditioned by a V-cycle took 6 iterations from 2D 33² to
513² and 7 from 3D 17³ to 129³. Point smoothers stall on stretched cells, which
[`gmg_preconditioner`](@ref) quotes in counts. Line and plane smoothers are planned in
[gpena/Bramble.jl#394](https://github.com/gpena/Bramble.jl/issues/394).

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

