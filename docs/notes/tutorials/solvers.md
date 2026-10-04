```@meta
CurrentModule = Bramble
```

# [Choosing a solver, backend and execution policy](@id tutorial_solvers)

**What you will learn.** How to choose a matrix backend, an execution policy and and a linear solver.

**What you need first.** The [form tutorial](@ref tutorial_form) for assembling a system, and the [backend tutorial](@ref tutorial_backend) for backends and execution policies.

**Where next.** [Solvers by problem](solvers_by_problem.md), which runs these choices on three problem classes.

Three choices are largely independent. A **backend** fixes how the system matrix is stored.
An **execution policy** fixes how grid operations and assembly are threaded. A **solver**
solves the resulting linear system. This page helps you choose among them. Every number is
produced by code on the page or cited from a named, committed benchmark.

## Rules of thumb

### Backends

| Backend | Recommended for | Avoid when |
|---|---|---|
| `SparseMatrixCSC` (default) | Everything below. Direct solves go straight into SuiteSparse, Accelerate or MUMPS with no conversion. | Never wrong as a default. |
| `SparseMatrixCSR` ([`csr_backend`](@ref), needs `using SparseMatricesCSR`) | 3D problems where matrix memory binds: the same matrix takes less memory than CSC. Assembly cost is about the same. | A direct solve, which is slower than CSC: `SparseMatricesCSR.jl` has no native CSR solve, so `\` goes through a transposed factorization of a reinterpreted LU. |

### Execution policies

[`CpuSerial`](@ref) is the default and the right choice for small grids and for steps
repeated inside a time loop. [`CpuPolyester`](@ref) (after `using Polyester`) and
[`CpuThreaded`](@ref) win above a per-operation crossover, and Polyester crosses first. All
three give the same answer. The [backend tutorial](backend.md#Crossovers) explains the
crossover and the script that measures your own.

### Direct solvers

| Solver | Recommended for | Avoid when |
|---|---|---|
| SuiteSparse (`:default`, `:suitesparse`; CHOLMOD for SPD, UMFPACK for unsymmetric) | The general default: exact to round-off, no tolerance to pick. | Fill-in makes memory bind: 3D problems past a few hundred thousand unknowns. |
| Apple Accelerate (`:accelerate`, macOS, `using AppleAccelerate`) | Symmetric systems on macOS, where it is faster than a plain direct solve ([#246](https://github.com/gpena/Bramble.jl/issues/246)). | Unsymmetric systems, where it is slower; `:default` is narrowed to symmetric matrices for that reason. |
| MUMPS (`:mumps`) | Parallel multifrontal factorization. | Not benchmarked here against SuiteSparse, so there is no crossover to report. |
| Sparspak (`:sparspak`) | No binary dependency, and generic over the element type (`Float32`, `BigFloat`, `ForwardDiff.Dual`), where the others cannot factor at all. | Not benchmarked for speed: it is the portability choice. |

[`sparse_factorize`](@ref) lists all four direct backends behind one interface.

## Decision tree

Choose a backend and a solver first, from what the problem looks like. The table below the
diagram explains each leaf.

```@raw html
<pre class="mermaid" data-src="">
flowchart TD
    Start(["What does the problem look like?"]) --> Spd{{"Is it symmetric positive definite?"}}
    Spd -->|"Yes"| How{{"How many solves are needed?"}}
    How -->|"One, or a few"| Chol(["Cholesky factorization"])
    How -->|"Many, same pattern"| Refac(["Factorize once, refactor each step"])
    Spd -->|"No"| Lu(["LU factorization"])
</pre>
```


| Leaf | Backend and solver | Why |
|---|---|---|
| Cholesky factorization | `SparseMatrixCSC` with `sparse_factorize` and `sym = :spd` (SuiteSparse CHOLMOD). | Exact to round-off, with no tolerance to pick. |
| Factorize once, refactor each step | `SparseMatrixCSC`: factorize once, then `refactor!` at each step. | The sparsity pattern is fixed, so the symbolic analysis is reused. See [time stepping](time_stepping.md). |
| LU factorization | `SparseMatrixCSC` with `sparse_factorize` and `sym = :unsymmetric` (SuiteSparse UMFPACK). | The general direct solver for unsymmetric systems. |

The second tree picks an execution policy. Its first question uses the crossover from the
[backend tutorial](backend.md#Crossovers): the smallest size at which a parallel
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
Against `A \ F` on Bramble-shaped systems the symmetric path is faster, while an
unsymmetric convection-diffusion system was slower through Accelerate before the dispatch
tested `issymmetric(A)`.
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
convection-diffusion problem and an elasticity problem, and
[time stepping](time_stepping.md) reuses one factorization across steps.
[`sparse_factorize`](@ref) documents the direct solvers in full, and
[`CpuSerial`](@ref), [`CpuThreaded`](@ref) and [`CpuPolyester`](@ref) document the
execution policies.
