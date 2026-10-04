# Sections moved out of `docs/src/tutorials/solvers.md`

As they stood. The headings and `@docs` blocks stay on the published page.

## Matrix-free operators

[`matrix_free_operator`](@ref) applies any form [`assemble`](@ref) accepts without storing
its matrix. It agrees with the assembled product, Dirichlet rows and composite spaces
included. The [matrix-free operator page](../examples/matrix_free_operator.md) defines it and
states where it stops. The [memory scaling page](../examples/memory_scaling.md) builds the
cheaper Kronecker operator for separable forms and measures its storage.

## Time and memory against a sparse product

Measured on an Apple M2 on AC power, `--threads=4`, each case alone, Julia 1.13.1, on
2026-09-28 (commit `40b0516d`). The form is mass plus variable diffusion on non-uniform
meshes, the sparse product is serial CSR, and times are the minimum of repeats. A time ratio
above 1 means the matrix-free product is faster. Bytes are `Base.summarysize`.

| Mesh | Unknowns | SpMV / serial matrix-free | SpMV / threaded matrix-free | CSR bytes / matrix-free bytes |
|---|---|---|---|---|
| 1D | 10⁴ | 0.57 | 0.92 | 1.0 |
| 1D | 10⁵ | 0.63 | 3.15 | 1.0 |
| 1D | 10⁷ | 0.57 | 2.20 | 1.0 |
| 2D 32² | 1024 | 0.44 | 0.45 | 6.1 |
| 2D 64² | 4096 | 0.47 | 1.23 | 8.4 |
| 2D 2048² | 4.2 × 10⁶ | 0.49 | 1.61 | 10.6 |
| 3D 16³ | 4096 | 0.30 | 0.69 | 11.4 |
| 3D 32³ | 32768 | 0.36 | 1.13 | 13.7 |
| 3D 128³ | 2.1 × 10⁶ | 0.36 | 1.16 | 14.4 |

Serial, the matrix-free product is 1.6 to 3.4 times slower than serial SpMV at every size,
because it recomputes each entry from the mesh and the coefficient. On 4 threads it beats
serial SpMV from 1D 10⁵ unknowns, 2D 64² and 3D 32³, and stays ahead above them (2.2 times
at 1D 10⁷, 1.6 at 2D 2048², 1.2 at 3D 128³). In 1D the two take the same memory; in 2D and
3D the CSR matrix takes 10.6 and 14.4 times the memory at the largest sizes. Choose the
operator when memory binds or threads are available. Assembled SpMV stays faster on one
thread.

## Jacobi and Chebyshev preconditioning

The matrix-free preconditioners need no assembled matrix. The problem below is mass plus
variable diffusion on a smoothly graded, non-uniform mesh. A graded mesh is where
a matrix-free product earns its keep, because every stencil entry depends on the local
spacing.

```@example solvers
using Bramble
using SciMLBase, LinearSolve, LinearAlgebra, Random

Ω_mf = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
graded(n) = [t + 0.1 * sinpi(2t) for t in range(0.0, 1.0; length = n)]
function graded_mesh(n)
    Ωₕ = mesh(Ω_mf, (n, n), (true, true))
    Bramble.change_points!(Ωₕ, (graded(n), graded(n)))
    return Ωₕ
end
spd_form(W) = form(W, W,
    (u, v) -> innerₕ(u, v) + inner₊(Rₕ(W, x -> 1 + x[1] * x[2]) * ∇ₕ(u), ∇ₕ(v)))

Ωmf = graded_mesh(65)
Wmf = gridspace(Ωmf)
a_mf = spd_form(Wmf)
A_mf = assemble(a_mf)
op = matrix_free_operator(a_mf)
size(op)
```

The operator is a linear map, applied with `mul!` or `*`. It matches the assembled product:

```@example solvers
Random.seed!(20260928)
x_mf = rand(size(op, 2))
y_mf = similar(x_mf)
mul!(y_mf, op, x_mf)
norm(y_mf - A_mf * x_mf) / norm(A_mf * x_mf) < 1e-14
```

A `MatrixFreeOperator` goes directly into a `LinearProblem`, and the preconditioners go into
`Pl`. [`jacobi_preconditioner`](@ref) reads `diag(A)` off one stencil walk.
[`chebyshev_preconditioner`](@ref) is a fixed degree-4 polynomial in `D⁻¹A`, scaled by
Jacobi, on a spectrum bound from `Bramble.max_eigenvalue_estimate`:

```@example solvers
b_mf = op * rand(size(op, 1))
prob_mf = LinearProblem(op, b_mf)
cg_iters(; kw...) =
    solve(prob_mf, KrylovJL_CG(); reltol = 1e-8, abstol = 0.0, maxiters = 5000, kw...).iters
P_jac = jacobi_preconditioner(op)
P_cheb = chebyshev_preconditioner(op)
cg_iters(), cg_iters(Pl = P_jac), cg_iters(Pl = P_cheb)
```

On this 65² mesh CG took 499 iterations unpreconditioned, 329 with Jacobi and 93 with
Chebyshev. Neither builds `A`.

!!! tip "Try this"
    Change `graded_mesh(65)` to `graded_mesh(33)` and rerun the blocks. The three counts fall
    to 246, 179 and 52, and Chebyshev's advantage over plain CG narrows from 5.4 times to 4.7.

## Geometric multigrid

[`gmg_preconditioner`](@ref) takes a builder `W -> form(...)` rather than a form, because a
form is tied to its space: each level of the [`GeometricMeshHierarchy`](@ref) is
rediscretised by calling the builder on that level's space. Every grid function in the form,
here the coefficient `Rₕ(W, κ)`, must be built from `W` inside the builder. One captured from
the finest space gives a wrong coarse operator.

```@example solvers
P_gmg = gmg_preconditioner(W -> spd_form(W), Ωmf)
sol_gmg = solve(prob_mf, KrylovJL_CG(); Pl = P_gmg, reltol = 1e-8, abstol = 0.0)
P_gmg, sol_gmg.iters, norm(A_mf * sol_gmg.u - b_mf) / norm(b_mf) < 1e-7
```

Six levels down to 3², and 10 CG iterations against Chebyshev's 93. [`gmg_solve`](@ref) runs
the cycles as a stationary iteration instead, and [`v_cycle!`](@ref), [`w_cycle!`](@ref) and
[`fmg!`](@ref) are the cycles themselves.

The smoothers are point smoothers, and they stall on stretched cells. On the random meshes of
`mesh(…, false)`, whose largest aspect ratio grows with `n` (96 at 33², 52600 at 513²), CG
with the V-cycle took 14 to 25, 26 to 37 and 32 to 116 iterations at 2D 33², 65² and 129²
over four draws, and 87 at 513². A random base refined with [`iterative_refinement!`](@ref)
keeps its aspect ratio, but the counts still grow per level: 11 to 26 from 17² to 257² on a
2D base of 9² (aspect ratio 10.9), and 14 to 29 from 9³ to 65³ on a 3D base of 5³ (aspect
ratio 15). The mesh-independent counts, 6 iterations from 2D 33² to 513² and 7 from 3D 17³
to 129³, were measured on meshes with bounded aspect ratio (uniform points jittered by up to
`±0.3h`). Line and plane smoothers for stretched meshes are planned in
[#394](https://github.com/gpena/Bramble.jl/issues/394). With Dirichlet rows, CG needs a
right-hand side that is zero on those rows. Device execution of the operator, the
preconditioners and the cycles is tracked on milestone
[v4.4.0](https://github.com/gpena/Bramble.jl/milestone/38).


# Passages removed from the remaining sections of `docs/src/tutorials/solvers.md`

## Intro

a linear solver, and where the matrix-free operator and its preconditioners fit.

## Backends: CSR row (timings)

| 3D problems where matrix memory binds: the same 3D Poisson system takes 24.4 MiB against CSC's 59.1 MiB (commit `7c901266`). Assembly cost is about the same. | A direct solve, measured 2.4x to 4.2x **slower** than CSC in the same benchmark. `SparseMatricesCSR.jl` has no native CSR solve,

## Backends: Kronecker row

| [`KroneckerLinearOperator`](@ref) ([`kronecker_operator`](@ref), for separable forms, see [`is_separable`](@ref)) | A separable operator on a tensor-product mesh: `O(n)` storage per axis instead of `O(n^D)`. | A form with a coefficient varying along several axes, a region restriction other than `:interior`, an interpolation, a 1D mesh, or a star or cross-weighted difference: `kronecker_operator` throws. A grid-function coefficient is read once, at construction; use a `Ref`, or a matrix-free operator, for one that changes. No memory or time crossover against CSC has been measured for it. |

## Execution policies: crossover link

[`CpuThreaded`](@ref) win above a per-operation crossover, and Polyester crosses first. All
three give the same answer. The [backend tutorial](backend.md#Measured-crossovers) holds the
crossover table and the script that measures your own.

## Solvers: Accelerate row (timings)

| Symmetric systems on macOS, measured 1.2x to 1.3x faster than a plain direct solve ([#246](https://github.com/gpena/Bramble.jl/issues/246)). | Unsymmetric systems, measured 2.3x to 3.6x **slower** before `:default` was narrowed to symmetric matrices. |

## Solvers: AMG and ILU rows

| `KrylovJL_CG` + [`amg_preconditioner`](@ref) | SPD systems solved repeatedly or too large to factorize: 20 times fewer iterations than plain CG in [solvers by problem](solvers_by_problem.md), a ratio that widens as `O(h^-1)` against AMG's `O(1)`. | Unsymmetric, convection-dominated systems: AMG did not converge in 300 iterations where ILU(0) needed 18 ([#244](https://github.com/gpena/Bramble.jl/issues/244)). |
| `KrylovJL_GMRES` + [`ilu_preconditioner`](@ref) | Unsymmetric, convection-dominated systems; cheap to build, no fill-in parameter. | Elliptic, symmetric systems: use AMG there. |

## Heading

### Direct and iterative solvers

## Decision tree

Start(["What does the problem look like?"]) --> Sep{{"Is the form separable?"}}
    Sep -->|"Yes"| Kron(["Kronecker operator with CG"])
    Sep -->|"No"| Spd{{"Is it symmetric positive definite?"}}
    Spd -->|"Yes"| How{{"How many solves are needed?"}}
    How -->|"One, or a few"| Chol(["Cholesky factorization"])
    How -->|"Many, same pattern"| Refac(["Factorize once, refactor each step"])
    How -->|"Too large to factorize"| Amg(["CG with an AMG preconditioner"])
    Spd -->|"No"| Conv{{"Is convection dominant?"}}
    Conv -->|"Yes"| Ilu(["GMRES with an ILU preconditioner"])
    Conv -->|"No"| Lu(["LU factorization"])

## Decision tree: Kronecker leaf

| Kronecker operator with CG | [`KroneckerLinearOperator`](@ref) from [`kronecker_operator`](@ref), solved with `KrylovJL_CG` when `issymmetric` holds for it, and with `KrylovJL_GMRES` when it does not (a form with advection, say). | A separable form needs `O(n)` storage per axis instead of `O(n^D)`. |

## Decision tree: AMG and ILU leaves

| CG with an AMG preconditioner | `SparseMatrixCSC` with `KrylovJL_CG` and [`amg_preconditioner`](@ref). | In 3D memory is the limit, and AMG needs `O(1)` iterations instead of `O(h^-1)`. |
| GMRES with an ILU preconditioner | `SparseMatrixCSC` with `KrylovJL_GMRES` and [`ilu_preconditioner`](@ref). | AMG did not converge on convection-dominated systems. |

## Policy tree: crossover link

[backend tutorial](backend.md#Measured-crossovers): the smallest

## Accelerate: timings

Against `A \ F` on Bramble-shaped systems the symmetric path is a 1.2x to 1.3x win (`n = 80`:
0.83 of the runtime; `n = 120`: 0.78), while an unsymmetric convection-diffusion system was
2.3x to 3.6x **slower** through Accelerate before the dispatch tested `issymmetric(A)`.

## Where to go next

[`sparse_factorize`](@ref), [`amg_preconditioner`](@ref) and [`ilu_preconditioner`](@ref)
document the solvers and preconditioners in full, and
[`CpuSerial`](@ref), [`CpuThreaded`](@ref) and [`CpuPolyester`](@ref) document their own
crossovers.

