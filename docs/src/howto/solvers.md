```@meta
CurrentModule = Bramble
```

# [How to choose a linear solver](@id howto_solvers)

[`assemble`](@ref) returns a sparse matrix `A` and a load `F`, and [`pde_solve`](@ref)`(A, F)` solves `A u = F`. Its keywords pick a direct backend, and an iterative method from LinearSolve.jl takes a preconditioner from Bramble. The kind of matrix decides which.

| Problem | Solve with |
|---|---|
| Symmetric positive definite, one solve | `pde_solve(A, F; solver = :suitesparse, sym = :spd)` |
| Symmetric positive definite, many solves with one sparsity pattern | `sparse_factorize` once, then `refactor!` |
| Symmetric positive definite, 3D with many unknowns | CG with [`amg_preconditioner`](@ref) |
| Non-symmetric, for example convection-dominated | `pde_solve(A, F)`, or GMRES with [`ilu_preconditioner`](@ref) |
| Pure Neumann | `pde_solve(A, F; mean_zero = w)` |
| Element type other than `Float64`, or no binary dependency | `pde_solve(A, F; solver = :sparspak)` |

The blocks below solve one system per row of the table and assert its residual. The first builds a Poisson system on a random non-uniform grid, with one row per node, ``40^2 = 1600``.

```@example howto_solvers
using Bramble, LinearAlgebra, Random, SparseArrays, SuiteSparse
using Bramble: sparse_factorize, refactor!

relres(A, u, F) = norm(A * u - F) / norm(F)

function poisson_system(Ω, n; uniform = false)
    Ωₕ = mesh(Ω, n, ntuple(_ -> uniform, length(n)))
    Wₕ = gridspace(Ωₕ)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    fₕ = element(Wₕ)
    avgₕ!(fₕ, x -> exp(sum(x)))
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    bcs = dirichlet_constraints(Ω, :boundary => (x -> exp(sum(x))))
    return assemble(a, l; dirichlet = bcs, symmetrize = true)
end

Random.seed!(20261111)
Ω₂ = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
A, F = poisson_system(Ω₂, (40, 40))

@assert size(A) == (40^2, 40^2) && length(F) == 40^2
# five-point stencil on 40 × 40 nodes, stored zeros of the eliminated couplings included
@assert nnz(A) == 5 * 40^2 - 4 * 40
# without them: the 38 × 38 interior block, plus one identity row per boundary node
@assert count(!iszero, A) == 5 * 38^2 - 4 * 38 + (40^2 - 38^2)
@assert norm(F) > 0                  # control: the load is nonzero
@assert issymmetric(A)               # `symmetrize = true`

size(A)
```

## Symmetric positive definite: one solve

`solver = :suitesparse` with `sym = :spd` factors with CHOLMOD and skips the symmetry test. The default `A \ F` tries Cholesky only on an exactly symmetric matrix, so one that is symmetric up to rounding goes to UMFPACK.

```@example howto_solvers
u = pde_solve(A, F; solver = :suitesparse, sym = :spd)
@assert relres(A, u, F) < 1e-10
relres(A, u, F)
```

## Many solves with one pattern

A time loop or a Picard iteration solves matrices with one sparsity pattern. [`sparse_factorize`](@ref) analyzes the pattern once, and [`refactor!`](@ref) recomputes the numbers in place. Here `A + I` stands for the matrix the next step assembles:

```@example howto_solvers
fact = sparse_factorize(A; sym = :spd)
@assert relres(A, fact \ F, F) < 1e-10

A_next = A + 0.5I
refactor!(fact, A_next)
@assert relres(A_next, fact \ F, F) < 1e-10
relres(A_next, fact \ F, F)
```

## Non-symmetric systems

Advection makes the matrix non-symmetric. The default `pde_solve` factors it with UMFPACK, and `sym = :unsymmetric` selects UMFPACK without a symmetry test. For an iterative solve, GMRES takes the place of CG, and ILU(0) preconditions it. Convection-dominated matrices are far from an M-matrix, and algebraic multigrid degrades on them ([#244](https://github.com/gpena/Bramble.jl/issues/244)), so ILU(0) is the preconditioner here.

```@example howto_solvers
using LinearSolve, ILUZero

function convection_diffusion_system(Ω, n; ε = 1.0e-2)
    Ωₕ = mesh(Ω, n, (false, false))
    Wₕ = gridspace(Ωₕ)
    a = form(Wₕ, Wₕ, (u, v) -> ε * inner₊(∇ₕ(u), ∇ₕ(v)) + inner₊(Mₕ(u), ∇ₕ(v)))
    fₕ = element(Wₕ)
    avgₕ!(fₕ, x -> exp(sum(x)))
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    bcs = dirichlet_constraints(Ω, :boundary => (x -> exp(sum(x))))
    return assemble(a, l; dirichlet = bcs)
end

Acd, Fcd = convection_diffusion_system(Ω₂, (40, 40))
@assert !issymmetric(Acd)

u_direct = pde_solve(Acd, Fcd; solver = :suitesparse, sym = :unsymmetric)
@assert relres(Acd, u_direct, Fcd) < 1e-10

problem = LinearProblem(Acd, Fcd)
plain = solve(problem, KrylovJL_GMRES(); reltol = 1e-10, abstol = 1e-10)
with_ilu = solve(problem, KrylovJL_GMRES(); Pl = ilu_preconditioner(Acd),
    reltol = 1e-10, abstol = 1e-10)
@assert relres(Acd, with_ilu.u, Fcd) < 1e-8
@assert with_ilu.iters < plain.iters
plain.iters, with_ilu.iters
```

## Pure Neumann

With the normal derivative given on the whole boundary, the matrix annihilates constants and `A \ F` is singular, so `pde_solve` logs a warning that names `mean_zero`. Pass `mean_zero = w`, the weights of the discrete average, and it solves the bordered system ``[A\ w; w^T\ 0]``.

The [pure Neumann how-to](@ref howto_pure_neumann) builds `w`, checks the compatibility condition and shows `gauge`, the keyword for a kernel in the ghost values of a ghost space.

## Large systems in 3D

On a 3D grid a sparse direct factorization fills in heavily past a few hundred thousand unknowns. CG with an algebraic multigrid preconditioner stores the hierarchy only. [`amg_preconditioner`](@ref) returns the hierarchy, and `aspreconditioner` from AlgebraicMultigrid.jl turns it into the `Pl` that LinearSolve takes. The system below is small enough to compare with plain CG:

```@example howto_solvers
using AlgebraicMultigrid: aspreconditioner

Ω₃ = domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0))
Random.seed!(20261111)
A₃, F₃ = poisson_system(Ω₃, (14, 14, 14))
problem₃ = LinearProblem(A₃, F₃)
cg = solve(problem₃, KrylovJL_CG(); reltol = 1e-10, abstol = 1e-10)
cg_amg = solve(problem₃, KrylovJL_CG(); Pl = aspreconditioner(amg_preconditioner(A₃)),
    reltol = 1e-10, abstol = 1e-10)
@assert relres(A₃, cg_amg.u, F₃) < 1e-8
@assert 5 * cg_amg.iters < cg.iters
size(A₃, 1), cg.iters, cg_amg.iters
```

`solve(a, l; solver = KrylovJL_CG(), preconditioner = :amg, symmetrize = true, dirichlet = bcs)` assembles the form and builds the hierarchy in one call (`:ilu0` is the other value). It needs `symmetrize = true` for CG, and it refuses a `preconditioner` without a Krylov `solver`.

## Other direct solvers

Each backend loads with its package and takes the same `pde_solve` and [`sparse_factorize`](@ref) keywords:

| `solver` | Use it for | Needs |
|---|---|---|
| `:default` | `A \ F`; on macOS the Accelerate route, when it is loaded and `A` is symmetric | |
| `:suitesparse` | CHOLMOD for a symmetric `A`, UMFPACK otherwise, chosen by `sym` | `using SuiteSparse` |
| `:spqr` | least squares, or a rectangular `A` | |
| `:accelerate` | Apple's `libSparse`: Cholesky, LDLᵀ, LU and QR | macOS, `using AppleAccelerate` |
| `:mumps` | MUMPS multifrontal factorization | `using MUMPS` |
| `:sparspak` | a pure Julia LU that factors `Float32` and `BigFloat` entries | `using Sparspak` |

```julia
using Sparspak, SparseArrays

u32 = pde_solve(SparseMatrixCSC{Float32}(A), Float32.(F); solver = :sparspak)
```
