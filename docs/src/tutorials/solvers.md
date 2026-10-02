```@meta
CurrentModule = Bramble
```

# [Choosing a solver, backend and execution policy](@id tutorial_solvers)

**What you will learn.** How to choose a matrix backend, an execution policy and a linear solver, and where the matrix-free operator and its preconditioners fit.

**What you need first.** The [form tutorial](@ref tutorial_form) for assembling a system, and the [backend tutorial](@ref tutorial_backend) for backends and execution policies.

**Where next.** [Solvers by problem](solvers_by_problem.md), which runs these choices on four problem classes.

Three choices are largely independent. A **backend** fixes how the system matrix is stored.
An **execution policy** fixes how grid operations and assembly are threaded. A **solver**
solves the resulting linear system. This page helps you choose among them. Every number is
produced by code on the page or cited from a named, committed benchmark.

## Rules of thumb

### Backends

| Backend | Recommended for | Avoid when |
|---|---|---|
| `SparseMatrixCSC` (default) | Everything below. Direct solves go straight into SuiteSparse, Accelerate or MUMPS with no conversion. | Never wrong as a default. |
| `SparseMatrixCSR` ([`csr_backend`](@ref), needs `using SparseMatricesCSR`) | 3D problems where matrix memory binds: the same 3D Poisson system takes 24.4 MiB against CSC's 59.1 MiB (commit `7c901266`). Assembly cost is about the same. | A direct solve, measured 2.4x to 4.2x **slower** than CSC in the same benchmark. `SparseMatricesCSR.jl` has no native CSR solve, so `\` goes through a transposed factorization of a reinterpreted LU. |
| [`KroneckerLinearOperator`](@ref) ([`kronecker_operator`](@ref), for separable forms, see [`is_separable`](@ref)) | A separable operator on a tensor-product mesh: `O(n)` storage per axis instead of `O(n^D)`. | Any form with a grid-function coefficient, a region restriction, an interpolation, or a mixed, forward, centered, averaged or jump operator: `kronecker_operator` throws. No memory or time crossover against CSC has been measured for it. |

### Execution policies

[`CpuSerial`](@ref) is the default and the right choice for small grids and for steps
repeated inside a time loop. [`CpuPolyester`](@ref) (after `using Polyester`) and
[`CpuThreaded`](@ref) win above a per-operation crossover, and Polyester crosses first. All
three give the same answer. The [backend tutorial](backend.md#Measured-crossovers) holds the
crossover table and the script that measures your own.

### Direct and iterative solvers

| Solver | Recommended for | Avoid when |
|---|---|---|
| SuiteSparse (`:default`, `:suitesparse`; CHOLMOD for SPD, UMFPACK for unsymmetric) | The general default: exact to round-off, no tolerance to pick. | Fill-in makes memory bind: 3D problems past a few hundred thousand unknowns. |
| Apple Accelerate (`:accelerate`, macOS, `using AppleAccelerate`) | Symmetric systems on macOS, measured 1.2x to 1.3x faster than a plain direct solve ([#246](https://github.com/gpena/Bramble.jl/issues/246)). | Unsymmetric systems, measured 2.3x to 3.6x **slower** before `:default` was narrowed to symmetric matrices. |
| MUMPS (`:mumps`) | Parallel multifrontal factorization. | Not benchmarked here against SuiteSparse, so there is no crossover to report. |
| Sparspak (`:sparspak`) | No binary dependency, and generic over the element type (`Float32`, `BigFloat`, `ForwardDiff.Dual`), where the others cannot factor at all. | Not benchmarked for speed: it is the portability choice. |
| `KrylovJL_CG` + [`amg_preconditioner`](@ref) | SPD systems solved repeatedly or too large to factorize: 20 times fewer iterations than plain CG in [solvers by problem](solvers_by_problem.md), a ratio that widens as `O(h^-1)` against AMG's `O(1)`. | Unsymmetric, convection-dominated systems: AMG did not converge in 300 iterations where ILU(0) needed 18 ([#244](https://github.com/gpena/Bramble.jl/issues/244)). |
| `KrylovJL_GMRES` + [`ilu_preconditioner`](@ref) | Unsymmetric, convection-dominated systems; cheap to build, no fill-in parameter. | Elliptic, symmetric systems: use AMG there. |

[`sparse_factorize`](@ref) lists all four direct backends behind one interface.

## Decision tree

Choose a backend and a solver first, from what the problem looks like. The table below the
diagram explains each leaf.

```@raw html
<pre class="mermaid" data-src="">
flowchart TD
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
</pre>
```


| Leaf | Backend and solver | Why |
|---|---|---|
| Kronecker operator with CG | [`KroneckerLinearOperator`](@ref) from [`kronecker_operator`](@ref), solved with `KrylovJL_CG`. | A separable form needs `O(n)` storage per axis instead of `O(n^D)`. |
| Cholesky factorization | `SparseMatrixCSC` with `sparse_factorize` and `sym = :spd` (SuiteSparse CHOLMOD). | Exact to round-off, with no tolerance to pick. |
| Factorize once, refactor each step | `SparseMatrixCSC`: factorize once, then `refactor!` at each step. | The sparsity pattern is fixed, so the symbolic analysis is reused. See [time stepping](time_stepping.md). |
| CG with an AMG preconditioner | `SparseMatrixCSC` with `KrylovJL_CG` and [`amg_preconditioner`](@ref). | In 3D memory is the limit, and AMG needs `O(1)` iterations instead of `O(h^-1)`. |
| GMRES with an ILU preconditioner | `SparseMatrixCSC` with `KrylovJL_GMRES` and [`ilu_preconditioner`](@ref). | AMG did not converge on convection-dominated systems. |
| LU factorization | `SparseMatrixCSC` with `sparse_factorize` and `sym = :unsymmetric` (SuiteSparse UMFPACK). | The general direct solver for unsymmetric systems. |

The second tree picks an execution policy. Its first question uses the crossover from the
[backend tutorial](backend.md#Measured-crossovers): the smallest size at which a parallel
policy beats `CpuSerial`.

```@raw html
<pre class="mermaid" data-src="">
flowchart TD
    Grid(["Which policy for this operation?"]) --> Small{{"Is the grid small for this operation?"}}
    Small -->|"Yes"| Serial(["CpuSerial"])
    Small -->|"No"| Poly{{"Is Polyester loaded?"}}
    Poly -->|"Yes"| Polyester(["CpuPolyester"])
    Poly -->|"No"| Above{{"Is it above the CpuThreaded crossover?"}}
    Above -->|"Yes"| Threaded(["CpuThreaded"])
    Above -->|"No"| Serial
</pre>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
const root = document.documentElement;
const blocks = [...document.querySelectorAll("pre.mermaid")];
blocks.forEach((el) => { el.dataset.src = el.textContent; el.style.maxWidth = "46rem"; });

function token(name) {
  return getComputedStyle(root).getPropertyValue(name).trim();
}

async function draw() {
  mermaid.initialize({
    startOnLoad: false,
    theme: "base",
    fontFamily: getComputedStyle(document.body).fontFamily,
    themeVariables: {
      primaryColor: token("--md-sys-color-primary-container"),
      primaryTextColor: token("--md-sys-color-on-primary-container"),
      primaryBorderColor: token("--md-sys-color-outline"),
      secondaryColor: token("--md-sys-color-surface-variant"),
      tertiaryColor: token("--md-sys-color-surface-variant"),
      lineColor: token("--md-sys-color-outline"),
      textColor: token("--md-sys-color-on-surface"),
      edgeLabelBackground: token("--md-sys-color-surface"),
      background: token("--md-sys-color-surface"),
      nodeBorder: token("--md-sys-color-outline"),
      fontSize: "15px",
    },
    flowchart: { curve: "basis", useMaxWidth: true, nodeSpacing: 28, rankSpacing: 36, padding: 10 },
  });
  for (const el of blocks) {
    el.removeAttribute("data-processed");
    el.textContent = el.dataset.src;
  }
  await mermaid.run({ nodes: blocks });
  // Rounded leaves take the surface-variant colour; hexagon decisions keep the primary container.
  for (const el of blocks) {
    el.querySelectorAll("g.node rect").forEach((r) => {
      r.style.fill = token("--md-sys-color-surface-variant");
    });
  }
}

await draw();
new MutationObserver(() => requestAnimationFrame(draw)).observe(root, {
  attributes: true,
  attributeFilter: ["data-theme"],
});
// With no saved theme MaterialDocs follows the system scheme in CSS alone and leaves
// `data-theme` unset, so a system change must redraw too.
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (!root.hasAttribute("data-theme")) requestAnimationFrame(draw);
});
</script>
```

| Leaf | When it applies |
|---|---|
| `CpuSerial` | The grid is below the crossover, or Polyester is missing and the grid is below `CpuThreaded`'s higher crossover. |
| `CpuPolyester` | The grid is above the crossover and `using Polyester` has been run. |
| `CpuThreaded` | Polyester is not loaded and the grid is above `CpuThreaded`'s own crossover. |

The trees compose: pick a backend and solver from the first, then a policy from the second.
The policy governs the grid operations and assembly that feed the solve, not the solver.

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

## macOS: when the default solve uses Apple Accelerate

[`pde_solve`](@ref)'s `:default` solver, which every `\` above sits on, uses Apple's
Accelerate framework instead of a bare `A \ F` on macOS, but only when `AppleAccelerate.jl`
is loaded and the system is symmetric:

```julia
using Bramble, AppleAccelerate  # loading it is enough to opt in

# A_sym, F_sym: the SPD system of the solvers-by-problem page; A_uns, F_uns: its
# convection-diffusion system. Both at n = 80.
u_sym = pde_solve(A_sym, F_sym)   # == pde_solve(A_sym, F_sym; solver = :accelerate)
u_uns = pde_solve(A_uns, F_uns)   # == A_uns \ F_uns, Accelerate never runs
```

The rule is narrow on purpose ([#246](https://github.com/gpena/Bramble.jl/issues/246)).
Against `A \ F` on Bramble-shaped systems the symmetric path is a 1.2x to 1.3x win (`n = 80`:
0.83 of the runtime; `n = 120`: 0.78), while an unsymmetric convection-diffusion system was
2.3x to 3.6x **slower** through Accelerate before the dispatch tested `issymmetric(A)`.
`solver = :accelerate` still honours an explicit request on an unsymmetric system.

Symmetry can be asserted instead of detected: `sym = :spd`, `:definite` or `:symmetric` takes
the Accelerate path without testing `issymmetric(A)`, and `sym = :unsymmetric` goes straight
to `A \ F`. Under `:default` an unrecognised `sym` falls back to `A \ F`; an explicit
`solver = :accelerate` validates `sym` and throws. Without `using AppleAccelerate`, or off
macOS, `:default` is exactly `A \ F`. The docstrings of [`accelerate_factorize`](@ref) and
[`accelerate_solve`](@ref) cover threading and accuracy
([#142](https://github.com/gpena/Bramble.jl/issues/142)).

## Where to go next

[Solvers by problem](solvers_by_problem.md) runs the choices above on a Poisson problem, a
convection-diffusion problem, an elasticity problem and a wave equation, and
[time stepping](time_stepping.md) reuses one factorization across steps.
[`sparse_factorize`](@ref), [`amg_preconditioner`](@ref) and [`ilu_preconditioner`](@ref)
document the solvers and preconditioners in full, and
[`CpuSerial`](@ref), [`CpuThreaded`](@ref) and [`CpuPolyester`](@ref) document their own
crossovers.
