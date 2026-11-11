```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Scientific computing: SciML and time integration

Time-dependent problems, the SciML interface and Jacobians, layered on the
[form API](api.md). Linear solvers and preconditioners are in
[Solvers and preconditioners](api/solvers.md).

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
`ldiv!`; see [Solvers and preconditioners](api/solvers.md). `element(Wₕ, sol)` and
`VectorElement(sol, Wₕ)` do the unwrapping step alone, for a `LinearSolution` already in
hand.

`semidiscretize_rhs` is neither exported nor `public`: call it as
`Bramble.semidiscretize_rhs`.

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

## Caching a coefficient-dependent assembly by element type

A Newton residual generic over `T` (`Float64` on a plain call, `ForwardDiff.Dual` while an
AD backend's sparse Jacobian sweep is probing it) cannot preallocate one matrix the way a
Picard loop can. `type_cached_assemble!` gives the sparsity pattern a place to live per
element type it is ever reached at instead, so only the very first call at a given type
pays for it.

```@docs
type_cached_assemble!
```
