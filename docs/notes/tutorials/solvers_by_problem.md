```@meta
CurrentModule = Bramble
```

# [Solvers by problem](@id tutorial_solvers_by_problem)

**What you will learn.** Which solver suits a symmetric positive-definite, an unsymmetric, a coupled and a hyperbolic problem, shown by running each.

**What you need first.** The [form tutorial](@ref form_dirichlet) for assembling a system with Dirichlet conditions, and the [solver choices](solvers.md) these examples put to work. The elasticity example reads [composite spaces](@ref space_composite).

**Where next.** [Time stepping](time_stepping.md), where one system is solved at every step.

Each section assembles one system, solves it two ways and compares. The comparisons count
iterations and measure agreement between solvers. They are not wall-clock timings.

## A symmetric positive-definite system

A Poisson problem's stiffness matrix is the model case: symmetric, positive definite, and
increasingly ill-conditioned under refinement (`O(h^-2)`).

```@example byproblem
using Bramble
using SuiteSparse
using SciMLBase, LinearSolve, LinearAlgebra, SparseArrays
using AlgebraicMultigrid: aspreconditioner
using ILUZero
using Random
import Bramble: sparse_factorize

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

A direct solve takes one line and has no tolerance to pick:

```@example byproblem
fact = sparse_factorize(A; sym = :spd)   # SuiteSparse CHOLMOD
u_direct = fact \ F
nothing # hide
```

An iterative solve needs a method and, past a token mesh, a preconditioner to keep the
iteration count from growing with `h`:

```@example byproblem
prob = LinearProblem(A, F)
sol_plain = solve(prob, KrylovJL_CG(); reltol = 1e-10, abstol = 1e-10)
P = aspreconditioner(amg_preconditioner(A))
sol_amg = solve(prob, KrylovJL_CG(); Pl = P, reltol = 1e-10, abstol = 1e-10)
sol_plain.iters, sol_amg.iters
```

AMG cuts the iterations twentyfold. The ratio widens with `n`, as `O(h^-1)` against AMG's
`O(1)`.

!!! tip "Try this"
    Change `spd_system(60)` to `spd_system(30)` and run the blocks again. The plain CG count
    falls roughly in proportion to the mesh size, while the AMG count barely moves.

For a single solve on a mesh a direct factorization can afford, `\` is simpler and exact to
round-off. AMG pays off once the same operator is solved repeatedly, or once fill-in makes
memory bind: in 3D past a few hundred thousand unknowns, a sparse direct solve's fill grows
from `O(N)` towards `O(N^{4/3})` (mesh-dependent), while AMG's memory stays `O(N)`.

## An unsymmetric, convection-dominated system

Adding advection breaks the symmetry that AMG's hierarchy relies on. The convective term
`inner₊(Mₕ(u), ∇ₕ(v))` is the staggered discretization of the
[convection-diffusion example](../examples/convection_diffusion_linear.md). A bare forward
difference is unstable for a positive advection direction, and ILU(0) breaks down on the
matrix it produces.

```@example byproblem
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

A direct solve changes only the symmetry hint:

```@example byproblem
fact_cd = sparse_factorize(Acd; sym = :unsymmetric)   # SuiteSparse UMFPACK
u_direct_cd = fact_cd \ Fcd
nothing # hide
```

Iteratively, GMRES replaces CG because the matrix is not symmetric, and ILU(0) replaces AMG:

```@example byproblem
prob_cd = LinearProblem(Acd, Fcd)
sol_cd_plain = solve(prob_cd, KrylovJL_GMRES(); reltol = 1e-10, abstol = 1e-10)
P_ilu = ilu_preconditioner(Acd)
sol_cd_ilu = solve(prob_cd, KrylovJL_GMRES(); Pl = P_ilu, reltol = 1e-10, abstol = 1e-10)
sol_cd_plain.iters, sol_cd_ilu.iters
```

AMG, tried on the same system:

```@example byproblem
P_amg_cd = aspreconditioner(amg_preconditioner(Acd; method = :ruge_stuben))
sol_cd_amg = solve(
    prob_cd, KrylovJL_GMRES(); Pl = P_amg_cd, reltol = 1e-10, abstol = 1e-10, maxiters = 300
)
sol_cd_amg.iters, sol_cd_amg.retcode
```

`MaxIters`: AMG does not converge in 300 iterations where ILU(0) needed a fraction of
that. [#244](https://github.com/gpena/Bramble.jl/issues/244) measured the same failure
(`ruge_stuben` capped at 2000 iterations on a similar system). Classical algebraic
multigrid assumes something close to an M-matrix, which strong advection breaks. ILU(0)
factorizes locally and keeps the sparsity pattern, so it assumes nothing of the kind.

Use ILU(0) as the default for unsymmetric, convection-dominated forms, and keep AMG for the
elliptic, symmetric systems of the previous section.

## A coupled system: elasticity

The previous systems come from one scalar grid space. Elasticity couples the three
components of a vector unknown through [`εₕ`](@ref) and [`divₕ`](@ref); the
[elasticity example](../examples/elasticity_3d.md) derives the form, and the
[coupled systems tutorial](@ref tutorial_coupled) shows how one form addresses several
unknowns. The solver question is whether the coupling changes the advice. The matrix is an
ordinary `SparseMatrixCSC`. With positive Lamé parameters and enough clamped boundary to
remove rigid-body motion, the symmetrized matrix is SPD, so the same direct or AMG choice
applies. The check below confirms it on a clamped beam:

```@example byproblem
lame(E, ν) = (E / (2 * (1 + ν)), E * ν / ((1 + ν) * (1 - 2ν)))
Emod, νmod = 1.0, 0.3
μe, λe = lame(Emod, νmod)

elasticity_form_el(Vₕ) = form(
    Vₕ, Vₕ, (u, v) -> 2μe * inner₊(εₕ(u), εₕ(v)) + λe * inner₊(divₕ(u), divₕ(v)))

function body_force_el(Vₕ, f)
    fₕ = element(Vₕ)
    avgₕ!(fₕ, f)
    return form(Vₕ, q -> innerₕ(fₕ(1), q(1)) + innerₕ(fₕ(2), q(2)) + innerₕ(fₕ(3), q(3)))
end

Ω_beam = domain(box((0.0, 0.0, 0.0), (2.0, 0.4, 0.4)), :clamped => :back)
function elasticity_system(n)
    Ωₕ = mesh(Ω_beam, n, (true, true, true))
    Vₕ = gridspace(Ωₕ)^Val(3)
    return assemble(elasticity_form_el(Vₕ), body_force_el(Vₕ, x -> (0.0, 0.0, -1.0e-4));
        dirichlet = dirichlet_constraints(Ω_beam, :clamped => x -> 0.0), symmetrize = true)
end

Ael, Fel = elasticity_system((17, 6, 6))
issymmetric(Ael), size(Ael)
```

The direct and iterative solutions are compared with each other:

```@example byproblem
fact_el = sparse_factorize(Ael; sym = :spd)
u_direct_el = fact_el \ Fel

prob_el = LinearProblem(Ael, Fel)
sol_el = solve(prob_el, KrylovJL_CG(); reltol = 1e-10, abstol = 1e-10)
P_el = aspreconditioner(amg_preconditioner(Ael))
sol_el_amg = solve(prob_el, KrylovJL_CG(); Pl = P_el, reltol = 1e-10, abstol = 1e-10)

norm(u_direct_el - sol_el.u) / norm(u_direct_el),
norm(u_direct_el - sol_el_amg.u) / norm(u_direct_el),
sol_el.iters, sol_el_amg.iters
```

The three solutions agree to about `1e-8`, within the solver tolerances, and AMG still cuts the iteration count sharply,
although its hierarchy is built from the graph of the coupled `1836 x 1836` matrix with no
knowledge of the three components. The SPD argument transfers unchanged; only the assembled
operator differs. This is a correctness check on one modest mesh. Whether AMG keeps its
`O(1)` advantage at the scale of the elasticity example's cantilever meshes has not been
measured.

## A hyperbolic system: the acoustic wave equation

The [wave equation example](../examples/wave_equation_2d.md) semidiscretizes
`∂ₜₜu - c²Δu = 0` into `M ü_h + K u_h = F(t)` and hands `K` to `OrdinaryDiffEq` through
[`semidiscretize_second_order`](@ref). On a tensor-product mesh `K` comes from
`inner₊(∇ₕ(u), ∇ₕ(v))`, a separable form, so it can also be built matrix-free with
[`kronecker_operator`](@ref):

```@example byproblem
Ω_wave = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωw = mesh(Ω_wave, (40, 40), (true, true))
Ww = gridspace(Ωw)
Kform_w = form(Ww, Ww, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
is_separable(Kform_w)
```

```@example byproblem
Kw_assembled = assemble(Kform_w)      # SparseMatrixCSC, action by SpMV
Kw_kron = kronecker_operator(Kform_w) # KroneckerLinearOperator, action as one fused pass
Base.summarysize(Kw_assembled), Base.summarysize(Kw_kron)
```

On this `40 x 40` mesh the assembled matrix already holds more bytes than the Kronecker
factors, and the gap grows with the mesh: `O(n^2)` stored entries against `O(n)` numbers per
axis, with `n` points per axis. The two compute the same linear map:

```@example byproblem
Random.seed!(20260922)
xrand = rand(size(Kw_assembled, 1))
norm(Kw_assembled * xrand - Kw_kron * xrand) / norm(Kw_assembled * xrand)
```

!!! tip "Try this"
    Change `(40, 40)` to `(80, 80)` and compare the two byte counts. The matrix grows by a
    factor near four, the Kronecker factors by a factor near two.

The Kronecker operator has no boundary handling, so the wave example's own
`semidiscretize_second_order` call, which needs a boundary-constrained `K`, stays on the
assembled path. The comparison above is about the interior operator only. The
[memory scaling example](../examples/memory_scaling.md) goes further with the operator and
states its limits. No timing comparison is made here, and no mesh size at which one
evaluation becomes faster than the other has been measured.

## Reference

| Problem class | Matrix | Direct | Iterative |
|---|---|---|---|
| Poisson, elasticity | SPD | `sparse_factorize(A; sym = :spd)` | `KrylovJL_CG` + [`amg_preconditioner`](@ref) |
| Convection-diffusion | Unsymmetric | `sparse_factorize(A; sym = :unsymmetric)` | `KrylovJL_GMRES` + [`ilu_preconditioner`](@ref) |
| Wave stiffness | SPD, separable | as SPD | `KrylovJL_CG` on [`kronecker_operator`](@ref) (interior operator only) |
