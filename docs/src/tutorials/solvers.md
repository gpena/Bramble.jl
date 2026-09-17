# Choosing a linear solver

Every worked example so far has used `\` or a single, unremarked `solve(a, l; ...)` call.
Bramble offers more than that: four direct factorization backends (SuiteSparse, Apple
Accelerate, MUMPS, Sparspak -- [`sparse_factorize`](@ref)'s own docstring lists all four)
behind one [`sparse_factorize`](@ref)/[`refactor!`](@ref) interface, iterative Krylov methods
through `LinearSolve`, and two preconditioners, [`amg_preconditioner`](@ref) and
[`ilu_preconditioner`](@ref). This tutorial is about *choosing* among them, not about any one
of their APIs in isolation. Every number below was produced by the code shown.

## 1. Symmetric positive-definite systems

A Poisson problem's stiffness matrix is the model case: symmetric, positive definite, and
increasingly ill-conditioned under refinement (`O(h^-2)`).

```@example solvers
using Bramble
using SuiteSparse
using SciMLBase, LinearSolve, LinearAlgebra, SparseArrays
using AlgebraicMultigrid: aspreconditioner
using ILUZero

Ωd = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
function spd_system(n)
    Ωₕ = mesh(Ωd, (n, n), (true, true))
    Wₕ = gridspace(Ωₕ)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    fₕ = element(Wₕ)
    avgₕ!(fₕ, x -> exp(x[1] + x[2]))
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    bcs = dirichlet_constraints(Ωd, :boundary => (x -> exp(x[1] + x[2])))
    return assemble(a, l; dirichlet = bcs, symmetrize = true)
end

A, F = spd_system(60)
issymmetric(A), size(A)
```

A direct solve needs one line, and no tolerance to pick:

```@example solvers
fact = sparse_factorize(A; sym = :spd)   # SuiteSparse CHOLMOD
u_direct = fact \ F
nothing # hide
```

An iterative solve needs a method and, for anything past a token mesh, a preconditioner to
keep the iteration count from growing with `h`:

```@example solvers
prob = LinearProblem(A, F)
sol_plain = solve(prob, KrylovJL_CG(); reltol = 1e-10, abstol = 1e-10)
P = aspreconditioner(amg_preconditioner(A))
sol_amg = solve(prob, KrylovJL_CG(); Pl = P, reltol = 1e-10, abstol = 1e-10)
sol_plain.iters, sol_amg.iters
```

Twenty times fewer iterations, and -- since AMG's hierarchy assumes the matrix stays close to
what elliptic forms actually assemble -- this ratio only widens as `n` grows, `O(h^-1)`
against AMG's `O(1)`.

**When each wins**: for a single solve on a mesh a direct factorization can afford, `\` is
simpler and exact to round-off, with no tolerance to choose. AMG pays off once the same
operator is solved repeatedly, or once fill-in makes a direct factorization's memory the
binding constraint -- 3D problems past a few hundred thousand degrees of freedom, where a
sparse direct solve's `O(N)`-to-`O(N^{4/3})` fill (mesh-dependent) outgrows what a machine
holds, while AMG's memory stays `O(N)`.

## 2. Unsymmetric, convection-dominated systems

Adding advection to the same operator breaks the symmetry AMG's hierarchy relies on. The
convective term below, `inner₊(Mₕ(u), ∇ₕ(v))`, is the same SBP-staggered discretization the
[convection-diffusion tutorial](../examples/convection_diffusion_linear.md) uses -- not a bare
forward difference, which is unstable for a positive advection direction and produces a
matrix so far from diagonally dominant that even ILU(0) breaks down on it.

```@example solvers
function convection_diffusion_system(n; eps = 1.0e-2)
    Ωₕ = mesh(Ωd, (n, n), (true, true))
    Wₕ = gridspace(Ωₕ)
    a = form(Wₕ, Wₕ, (u, v) -> eps * inner₊(∇ₕ(u), ∇ₕ(v)) + inner₊(Mₕ(u), ∇ₕ(v)))
    fₕ = element(Wₕ)
    avgₕ!(fₕ, x -> exp(x[1] + x[2]))
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    bcs = dirichlet_constraints(Ωd, :boundary => (x -> exp(x[1] + x[2])))
    return assemble(a, l; dirichlet = bcs, symmetrize = false)
end

Acd, Fcd = convection_diffusion_system(60)
issymmetric(Acd)
```

Direct factorization needs only the symmetry hint changed:

```@example solvers
fact_cd = sparse_factorize(Acd; sym = :unsymmetric)   # SuiteSparse UMFPACK
u_direct_cd = fact_cd \ Fcd
nothing # hide
```

Iteratively, GMRES replaces CG (the matrix is no longer symmetric), and ILU(0) replaces AMG:

```@example solvers
prob_cd = LinearProblem(Acd, Fcd)
sol_cd_plain = solve(prob_cd, KrylovJL_GMRES(); reltol = 1e-10, abstol = 1e-10)
P_ilu = ilu_preconditioner(Acd)
sol_cd_ilu = solve(prob_cd, KrylovJL_GMRES(); Pl = P_ilu, reltol = 1e-10, abstol = 1e-10)
sol_cd_plain.iters, sol_cd_ilu.iters
```

And AMG, tried anyway, on the same system:

```@example solvers
P_amg_cd = aspreconditioner(amg_preconditioner(Acd; method = :ruge_stuben))
sol_cd_amg = solve(
    prob_cd, KrylovJL_GMRES(); Pl = P_amg_cd, reltol = 1e-10, abstol = 1e-10, maxiters = 300
)
sol_cd_amg.iters, sol_cd_amg.retcode
```

`MaxIters`: AMG does not merely underperform here, it fails to converge in 300 iterations
where ILU(0) needed a fraction of that -- the same failure
[gpena/Bramble.jl#244](https://github.com/gpena/Bramble.jl/issues/244) measured (`ruge_stuben`
capped at 2000 iterations without converging on a similar system). Classical algebraic
multigrid assumes something close to an M-matrix; strong advection breaks that assumption,
where ILU(0)'s local, pattern-preserving factorization has no such requirement.

**When each wins**: ILU(0) is the right default for unsymmetric, convection-dominated forms.
Reach for AMG only on the elliptic, symmetric systems §1 covers -- not as a general-purpose
default, which is exactly the mistake this section's own numbers argue against.

## 3. Steady vs. unsteady problems

Every comparison above solved one linear system. A steady problem only ever needs one; a
time-dependent problem solved by an implicit scheme needs one *per step*, against the same
sparsity pattern (the mesh does not change) but generally different numerical values (a
time-varying coefficient, or the previous step's solution feeding a semi-implicit term). That
repetition is what makes "factorize once, reuse" worth considering as an alternative to
preconditioning from scratch every step.

Take backward Euler for `∂u/∂t = Δu - c(t) u`, with a reaction coefficient `c(t)` that varies
in time -- the same sparsity pattern every step, different values, so a symbolic
factorization computed once stays valid for all of them:

```@example solvers
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

### Strategy A: factorize once, `refactor!` each step

```@example solvers
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

`refactor!` recomputes only the numeric values against the fixed elimination tree from step
1's factorization -- a pure back-substitution cost per step, no fresh symbolic analysis and no
iteration count to track at all.

### Strategy B/C: iterative, cold-started vs. warm-started

An iterative solve has no factorization to reuse, but has something a direct solve does not:
an initial guess. Consecutive time steps are close together, so the *previous* step's solution
is normally a much better starting point than the zero vector every cold solve implicitly
uses.

`LinearSolve`'s Krylov wrappers only honour a warm start for GMRES/FGMRES (`warm_start`
is ignored by every other method, CG included, per `KrylovJL_GMRES`'s own docstring) --
and only when the *same cache* is reused across solves, not a fresh `LinearProblem` built
each step:

```@example solvers
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

All three strategies agree with each other to near round-off:

```@example solvers
norm(uA - uB) / norm(uA), norm(uA - uC) / norm(uA)
```

Warm-starting cuts the total iteration count across the whole run, though by less than a
single-step comparison in isolation would suggest -- each step's operator has actually
changed (`c(t)` moved), not only its right-hand side, so the previous solution is a good but
imperfect guess, not the exact answer to a nearby problem.

**When each wins**: for a fixed-pattern time loop, factorizing once and calling `refactor!`
is the simplest correct choice -- no iteration count to track, no preconditioner to build, and
the [forms tutorial](form.md)'s own solver note already measured direct reuse running an
order of magnitude faster per step than iterative AMG-CG or GMRES on repeated solves. Reach
for a warm-started iterative solve instead when the matrix is too large to factorize at all
(§1's memory argument, now applied per step rather than once), where the reduction above is
free on top of whatever iteration count the preconditioner alone already bought.

## Where to go next

[`sparse_factorize`](@ref)'s own docstring lists every direct backend and when each is
preferred by problem size and platform. [`amg_preconditioner`](@ref) and
[`ilu_preconditioner`](@ref) cover the preconditioners themselves in more depth, including
[gpena/Bramble.jl#244](https://github.com/gpena/Bramble.jl/issues/244)'s full evaluation of
the wider JuliaSparse ecosystem.
