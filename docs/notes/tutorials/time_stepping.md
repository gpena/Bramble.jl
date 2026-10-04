```@meta
CurrentModule = Bramble
```

# [Solving at every time step](@id tutorial_time_stepping)

**What you will learn.** How to solve the same sparse system at every step of an implicit scheme, by refactoring once or by warm-starting an iterative method.

**What you need first.** The [solver choices](solvers.md), and the [form tutorial](@ref tutorial_form) for assembling the mass and stiffness matrices.

**Where next.** [Solvers by problem](solvers_by_problem.md) for the system classes, or the [in-place transient example](../examples/transient_inplace.md) for a full time loop.

A steady problem needs one linear solve. A time-dependent problem solved by an implicit
scheme needs one per step, on the same sparsity pattern (the mesh does not change) with
different values (a time-varying coefficient, or the previous step's solution in a
semi-implicit term). The repetition makes reuse worth considering.

## The problem

Backward Euler for `∂u/∂t = Δu - c(t) u` has a reaction coefficient `c(t)` that varies in
time, so the matrix changes value at every step and keeps its pattern:

```@example timestep
using Bramble
using SuiteSparse
using SciMLBase, LinearSolve, LinearAlgebra, SparseArrays
import Bramble: sparse_factorize, refactor!

Ωd = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
n = 40
Ωₕ = mesh(Ωd, (n, n), (true, true))
Wₕ = gridspace(Ωₕ)
M = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v)))
K = assemble(form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))))
dt = 0.01
nsteps = 15
c(t) = 1.0 + 0.5 * sin(4t)   # always positive, so A(t) stays SPD
u0vec = fill(1.0, size(M, 1))
nothing # hide
```

Step `k` solves `(M/dt + K + c(k dt) M) uᵏ = (M/dt) uᵏ⁻¹`.

## Strategy A: factorize once, then refactor each step

```@example timestep
function direct_reuse(M, K, c, dt, nsteps, u0vec)
    A1 = M ./ dt .+ K .+ c(dt) .* M
    fact = sparse_factorize(A1; sym = :spd)
    u = copy(u0vec)
    for step in 2:nsteps
        An = M ./ dt .+ K .+ c(step * dt) .* M
        refactor!(fact, An)             # reuses the symbolic factorization
        u = fact \ ((M ./ dt) * u)
    end
    return u
end

uA = direct_reuse(M, K, c, dt, nsteps, u0vec)
nothing # hide
```

[`refactor!`](@ref) recomputes only the numeric values against the elimination tree of the
first factorization. Each step costs a numeric factorization and a back-substitution, with
no fresh symbolic analysis and no iteration count to track.

## Strategies B and C: iterative, cold-started and warm-started

An iterative solve has no factorization to reuse, but it has an initial guess. Consecutive
steps are close, so the previous solution is a better start than the zero vector a cold
solve uses.

`LinearSolve` honours a warm start only for GMRES and FGMRES (`warm_start` is ignored by
every other method, CG included, per the docstring of `KrylovJL_GMRES`), and only when the
same cache is reused across solves rather than a fresh `LinearProblem` built each step:

```@example timestep
function iterative_cold(M, K, c, dt, nsteps, u0vec)
    u = copy(u0vec)
    total = 0
    for step in 2:nsteps
        An = M ./ dt .+ K .+ c(step * dt) .* M
        rhs = (M ./ dt) * u
        sol = solve(LinearProblem(An, rhs), KrylovJL_GMRES(); reltol = 1e-10, abstol = 1e-10)
        total += sol.iters
        u = sol.u
    end
    return u, total
end

function iterative_warm(M, K, c, dt, nsteps, u0vec)
    u = copy(u0vec)
    A0 = M ./ dt .+ K .+ c(2dt) .* M
    cache = init(
        LinearProblem(A0, (M ./ dt) * u),
        KrylovJL_GMRES(warm_start = LinearSolve.WarmStart.Previous);
        reltol = 1e-10, abstol = 1e-10
    )
    sol = solve!(cache)
    total = sol.iters
    u = sol.u
    for step in 3:nsteps
        cache.A = M ./ dt .+ K .+ c(step * dt) .* M
        cache.b = (M ./ dt) * u
        sol = solve!(cache)              # seeds from cache.u, the previous step's solution
        total += sol.iters
        u = sol.u
    end
    return u, total
end

uB, iters_cold = iterative_cold(M, K, c, dt, nsteps, u0vec)
uC, iters_warm = iterative_warm(M, K, c, dt, nsteps, u0vec)
iters_cold, iters_warm
```

The three strategies agree to about `1e-11`, within the solver tolerance:

```@example timestep
norm(uA - uB) / norm(uA), norm(uA - uC) / norm(uA)
```

Warm-starting lowers the total iteration count, by less than a single-step comparison would
suggest. The operator changes as well as the right-hand side, because `c(t)` moves, so the
previous solution is a good but imperfect guess.

!!! tip "Try this"
    Change `dt = 0.01` to `dt = 0.1` and rerun the blocks. Larger steps move the solution
    further between solves, so the warm start saves fewer iterations: about 60 across the run instead of about 140.

## Which to use

For a fixed-pattern time loop, factorize once and call `refactor!`. It is the simplest
correct choice, with no iteration count and no preconditioner to manage.

Warm-started iteration is for matrices too large to factorize at all. That is the memory
argument of [solvers by problem](solvers_by_problem.md), applied at every step, and the warm
start comes on top of whatever the preconditioner already saves.

## Reference

| Strategy | Setup | Per step | Needs |
|---|---|---|---|
| A, refactor | `sparse_factorize` once | `refactor!`, then `\` | A fixed sparsity pattern |
| B, cold iterative | none | `solve` on a new `LinearProblem` | Nothing; starts from zero |
| C, warm iterative | `init` a cache once | update `cache.A`, `cache.b`, `solve!` | `KrylovJL_GMRES` with `warm_start` |
