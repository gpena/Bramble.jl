# Choosing a solver, backend, and execution policy

Every worked example so far has used `\` or a single, unremarked `solve(a, l; ...)` call,
running under whatever `Backend` and execution policy happened to be the default. Bramble
actually exposes three largely independent choices: a **backend** for how the system matrix is
stored (`SparseMatrixCSC` by default, `SparseMatrixCSR` via [`csr_backend`](@ref), or a
matrix-free [`KroneckerLinearOperator`](@ref) for separable forms), an **execution policy** for
how grid operations and form assembly are threaded ([`CpuSerial`](@ref)/`Serial()`,
[`CpuThreaded`](@ref)/`Parallel()`, [`CpuPolyester`](@ref)), and a **solver** for the resulting
linear system -- four direct factorization backends behind one
[`sparse_factorize`](@ref)/[`refactor!`](@ref) interface (SuiteSparse, Apple Accelerate, MUMPS,
Sparspak -- [`sparse_factorize`](@ref)'s own docstring lists all four), iterative Krylov methods
through `LinearSolve`, and two preconditioners, [`amg_preconditioner`](@ref) and
[`ilu_preconditioner`](@ref). This tutorial is about *choosing* among all three axes, not about
any one of their APIs in isolation. Every number below either comes from the code shown on this
page or is cited from a specific, already-committed benchmark -- nothing here was measured
freshly for this page.

## Rules of thumb

### Backends

| Backend | Recommended for | Avoid when |
|---|---|---|
| `SparseMatrixCSC` (default) | The default for everything below -- direct solves route straight into SuiteSparse/Accelerate/MUMPS with no conversion. | -- |
| `SparseMatrixCSR` ([`csr_backend`](@ref), requires `using SparseMatricesCSR`) | 3D problems where matrix memory is the binding constraint: the same 3D Poisson system stores 24.4 MiB against CSC's 59.1 MiB (commit `7c901266`). Assembly cost itself is roughly a wash between the two backends. | A direct solve -- measured 2.4x-4.2x **slower** than CSC in the same benchmark. `SparseMatricesCSR.jl` has no native CSR solve path: `\` routes through a `TransposeFactorization` wrapped around a reinterpreted LU rather than calling SuiteSparse directly. |
| [`KroneckerLinearOperator`](@ref) ([`kronecker_operator`](@ref), for separable forms -- see [`is_separable`](@ref)) | Matrix-free evaluation of a separable operator (`innerₕ`/`∇ₕ`-only terms on a tensor-product mesh): `O(n)` storage per axis instead of the assembled matrix's `O(n^D)`. | Any form with a grid-function coefficient, a region restriction (Dirichlet included), an interpolation, or a mixed/forward/centered/averaged/jump operator -- `kronecker_operator` throws on these. No CSC/CSR-style memory-or-time crossover has been measured for this repository's own separable forms yet; don't infer one from the CSR figures above -- that comparison has not been run. |

### Execution policies

Measured on an Apple M2, AC power, `--threads=4`, three runs each, as the smallest grid size
where the parallel arm beat serial twice running (commit `4b76d62b`, closing
[gpena/Bramble.jl#190](https://github.com/gpena/Bramble.jl/issues/190)):

| Policy | Crossover vs. `CpuSerial` | Recommended for | Avoid when |
|---|---|---|---|
| [`CpuSerial`](@ref) / `Serial()` (default) | -- | Anything below the crossovers to the right; the safe default, no threading overhead. | Never wrong as a default -- only ever worth leaving once a workload is provably above a measured crossover. |
| [`CpuThreaded`](@ref) / `Parallel()` (`Base.Threads.@threads`, unconditional) | `Rₕ!` unmasked: 64-96 pts/axis. `Rₕ!` masked: 256. `avgₕ!` (`nq = 3`): 24-32. `innerₕ`/`_dot`: no real parallel arm. | Grids at or above these sizes, when Polyester isn't an option. | `innerₕ`/`normₕ` -- `_dot(::CpuThreaded, ...)` forwards to the identical serial reduction ([gpena/Bramble.jl#112](https://github.com/gpena/Bramble.jl/issues/112), closed, superseded by #190), so this policy buys nothing there; a speed ratio for that column would just be the same code timed twice. |
| [`CpuPolyester`](@ref) (Polyester `@batch`, requires `using Polyester`) | `Rₕ!` unmasked: 8-24. `Rₕ!` masked: 16. `avgₕ!` (`nq = 3`): 8. `innerₕ`/`_dot`: 1,000 elements. | Beats `CpuThreaded` at every crossover measured, by 4x-16x in grid size -- the default choice once Polyester is loaded. | Below its own crossover, where `CpuSerial` still wins; requires the `BramblePolyesterExt` extension (`using Polyester`) loaded, or the call errors naming the package. |

### Direct and iterative solvers

| Solver | Recommended for | Avoid when |
|---|---|---|
| SuiteSparse (`:default`/`:suitesparse` -- CHOLMOD for SPD/symmetric, UMFPACK for unsymmetric) | The general default: exact to round-off, no tolerance to pick. | Meshes where fill-in makes memory the binding constraint -- 3D problems past a few hundred thousand DOF (see below). |
| Apple Accelerate (`:accelerate`, macOS, `using AppleAccelerate`) | Symmetric systems on macOS -- measured 1.2-1.3x faster than a plain direct solve (`n = 80`: 0.83x the runtime; `n = 120`: 0.78x; [gpena/Bramble.jl#246](https://github.com/gpena/Bramble.jl/issues/246)). | Unsymmetric systems -- measured 2.3-3.6x **slower** than a plain direct solve before `:default`'s dispatch was narrowed to `issymmetric(A)`. An explicit `solver = :accelerate` still honours a direct request on an unsymmetric system. |
| MUMPS (`:mumps`) | Parallel multifrontal factorization, where that architecture fits the deployment. | Not yet benchmarked in this repository against SuiteSparse -- no measured crossover to report. |
| Sparspak (`:sparspak`) | Zero binary dependency; generic over element type (`Float32`, `BigFloat`, `ForwardDiff.Dual`) where SuiteSparse/MUMPS/Accelerate cannot factor at all. | Not yet benchmarked for speed against SuiteSparse -- treat it as the portability/correctness choice, not (yet, measurably) the fast one. |
| `KrylovJL_CG` + [`amg_preconditioner`](@ref) | Symmetric positive-definite systems, especially solved repeatedly or too large to factorize directly: measured 20x fewer iterations than plain CG on a Poisson system below, a ratio that widens as `O(h^-1)` against AMG's `O(1)`. | Unsymmetric, convection-dominated systems -- `ruge_stuben` AMG failed to converge within 300 iterations on the system below, the same failure [gpena/Bramble.jl#244](https://github.com/gpena/Bramble.jl/issues/244) measured (capped at 2000 iterations without converging), where ILU(0) needed 18. |
| `KrylovJL_GMRES` + [`ilu_preconditioner`](@ref) | Unsymmetric, convection-dominated systems -- 18 iterations against AMG's non-convergence ([gpena/Bramble.jl#244](https://github.com/gpena/Bramble.jl/issues/244)); no fill-in parameter to tune, cheap to build. | Elliptic, symmetric systems -- reach for AMG there instead, which this section's own numbers argue for. |

Kronecker-vs-CSC/CSR memory and time, and any Pardiso comparison, are deliberately absent from
this page: neither has been measured for this repository, and inventing a number would be worse
than leaving the row blank.

## Decision tree

Backend and solver first, as a function of what the problem itself looks like:

```@raw html
<pre class="mermaid">
flowchart TD
    Start["Characterize the problem"] --> Q1{"Is the bilinear form separable?<br/>tensor-product mesh, only innerₕ/∇ₕ terms"}
    Q1 -->|"Yes"| L1["Backend: KroneckerLinearOperator (kronecker_operator)<br/>Solver: KrylovJL_CG, matrix-free mul!<br/>O(n) storage per axis instead of O(n^D)"]
    Q1 -->|"No"| Q2{"Symmetric positive definite?"}
    Q2 -->|"Yes"| Q3{"Solved once, or repeatedly<br/>with a fixed sparsity pattern?"}
    Q3 -->|"Once or a few solves"| L2["Backend: SparseMatrixCSC<br/>Solver: sparse_factorize, sym = spd<br/>SuiteSparse CHOLMOD"]
    Q3 -->|"Repeated, fixed pattern<br/>e.g. implicit time stepping"| L3["Backend: SparseMatrixCSC<br/>Solver: factorize once, then refactor! each step"]
    Q3 -->|"Memory bound: 3D,<br/>hundreds of thousands of DOF"| L4["Backend: SparseMatrixCSC<br/>Solver: KrylovJL_CG plus amg_preconditioner<br/>O(1) iterations instead of O(h^-1)"]
    Q2 -->|"No, unsymmetric"| Q4{"Convection dominated?"}
    Q4 -->|"Yes"| L5["Backend: SparseMatrixCSC<br/>Solver: KrylovJL_GMRES plus ilu_preconditioner<br/>not amg_preconditioner, see issue 244"]
    Q4 -->|"No"| L6["Backend: SparseMatrixCSC<br/>Solver: sparse_factorize, sym = unsymmetric<br/>SuiteSparse UMFPACK"]
</pre>
<pre class="mermaid">
flowchart TD
    G["Grid size for this workload, points per axis or elements"] --> H{"Below the CpuPolyester crossover?<br/>8 to 24 unmasked Rₕ!, 16 masked,<br/>8 avgₕ!, 1000 elements innerₕ or _dot"}
    H -->|"Yes"| H1["CpuSerial (default)"]
    H -->|"No"| I{"Is Polyester.jl loaded?<br/>using Polyester"}
    I -->|"Yes"| I1["CpuPolyester<br/>beats CpuThreaded at every<br/>measured crossover, 4x to 16x smaller grid"]
    I -->|"No"| J{"Above the CpuThreaded crossover?<br/>64 to 96 unmasked Rₕ!, 256 masked,<br/>24 to 32 avgₕ!"}
    J -->|"Yes"| J1["CpuThreaded<br/>note: innerₕ and _dot gain nothing here,<br/>see issue 112"]
    J -->|"No"| H1
</pre>
<script>
(function () {
    // Documenter's own page runs a RequireJS/AMD loader (for highlight.js, KaTeX, ...),
    // whose global `define`/`require` hijack mermaid's UMD bundle into loading as an AMD
    // module instead of a plain global -- the failure mode is a silent-looking
    // "Se.default.extend is not a function" deep inside mermaid's own dependency chain,
    // with every `.mermaid` block left as unrendered text. Hiding `define`/`require` while
    // the script loads, the standard workaround for embedding a UMD library on a page that
    // already runs RequireJS, is what makes this the global (non-module) `mermaid.min.js`
    // build resolves against `window.mermaid`, not the ESM build.
    var savedDefine = window.define, savedRequire = window.require;
    window.define = undefined;
    window.require = undefined;
    var s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js";
    s.onload = function () {
        window.define = savedDefine;
        window.require = savedRequire;
        window.mermaid.initialize({ startOnLoad: false, theme: "neutral" });
        window.mermaid.run({ querySelector: "pre.mermaid" });
    };
    document.currentScript.parentNode.appendChild(s);
})();
</script>
```

The two trees compose: pick a (backend, solver) leaf from the first tree, then an execution
policy from the second -- the policy governs how the grid operations and assembly *feeding*
that solve are threaded, not which solver is chosen. The macOS Apple Accelerate dispatch is a
further, orthogonal narrowing of the direct-solve leaves above; it is covered on its own below
since it only ever applies automatically to `:default` and only on one platform.

## Symmetric positive-definite systems

A Poisson problem's stiffness matrix is the model case: symmetric, positive definite, and
increasingly ill-conditioned under refinement (`O(h^-2)`).

```@example solvers
using Bramble
using SuiteSparse
using SciMLBase, LinearSolve, LinearAlgebra, SparseArrays
using AlgebraicMultigrid: aspreconditioner
using ILUZero
using Random
import Bramble: sparse_factorize, refactor!

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

## Unsymmetric, convection-dominated systems

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
Reach for AMG only on the elliptic, symmetric systems the previous section covers -- not as a
general-purpose default, which is exactly the mistake this section's own numbers argue against.

## Coupled multi-component systems: elasticity

A scalar Poisson system and a convection-diffusion system both come from a single grid space.
Elasticity's bilinear form couples three components of a vector-valued unknown through
[`εₕ`](@ref) and [`divₕ`](@ref) (the [elasticity tutorial](../examples/elasticity_3d.md) derives
the form in full); the solver question is whether that coupling changes anything about the
recommendations above. It assembles into an ordinary `SparseMatrixCSC`, and when the Lamé
parameters are positive and enough of the boundary is clamped to remove rigid-body motion, the
symmetrized matrix is exactly as SPD as the scalar Poisson case -- so the same direct/AMG
choice applies, and this is what is worth checking rather than assuming.

A modest clamped beam, small enough to factor directly and still worth cross-checking against
an iterative solve:

```@example solvers
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

Direct and iterative solves against each other, rather than against a wall clock:

```@example solvers
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

All three agree to near round-off, and AMG still cuts the iteration count sharply even though
its hierarchy is built from the graph of the coupled `1836 x 1836` matrix with no awareness
that three displacement components sit underneath it -- the SPD argument from the Poisson
section transfers unchanged; what changes is only the assembled operator, not which solver
theory applies to it.

**When each wins**: exactly the SPD-systems guidance above, with one caveat -- these are
correctness cross-checks on one modest mesh, not a new timing comparison. Whether AMG's
graph-only coarsening keeps its `O(1)` iteration-count advantage at the scale the
[elasticity tutorial](../examples/elasticity_3d.md)'s own cantilever meshes reach has not been
measured, and this page does not claim it either way.

## Hyperbolic, transient systems: the acoustic wave equation

The [wave equation tutorial](../examples/wave_equation_2d.md) semidiscretizes
`∂ₜₜu - c²Δu = 0` into `M ü_h + K u_h = F(t)` and hands the assembled stiffness matrix `K` to
`OrdinaryDiffEq` through [`semidiscretize_second_order`](@ref). That `K` comes from
`inner₊(∇ₕ(u), ∇ₕ(v))` on a tensor-product mesh -- exactly the separable shape
[`is_separable`](@ref) recognises -- so the same operator can also be built matrix-free with
[`kronecker_operator`](@ref) instead of assembled into a `SparseMatrixCSC`:

```@example solvers
Ω_wave = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωw = mesh(Ω_wave, (40, 40), (true, true))
Ww = gridspace(Ωw)
Kform_w = form(Ww, Ww, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
is_separable(Kform_w)
```

```@example solvers
Kw_assembled = assemble(Kform_w)      # SparseMatrixCSC, action by SpMV
Kw_kron = kronecker_operator(Kform_w) # KroneckerLinearOperator, action as one fused pass
Base.summarysize(Kw_assembled), Base.summarysize(Kw_kron)
```

The structural difference this page can make honestly: on this `40 x 40` mesh the assembled
matrix already holds noticeably more bytes than the Kronecker factors, and that gap only grows
with mesh size (`O(n^2)` stored entries against `O(n)` numbers per axis for a 2D operator, `n`
grid points per axis). What matters more than the byte count is that the two give the same
answer -- the matrix-free `mul!` is not an approximation of the assembled SpMV, it is the same
linear map computed a different way:

```@example solvers
Random.seed!(20260922)
xrand = rand(size(Kw_assembled, 1))
norm(Kw_assembled * xrand - Kw_kron * xrand) / norm(Kw_assembled * xrand)
```

**When each applies, and where the comparison stops**: [`KroneckerLinearOperator`](@ref)
carries no boundary handling of its own -- Dirichlet rows are explicitly out of scope for it,
so the wave tutorial's own [`semidiscretize_second_order`](@ref) call, which needs a
boundary-constrained `K` (a constrained row replaced by `eₖ`), stays on the assembled path.
The comparison above is therefore about the *interior* spatial operator's action, not a
drop-in replacement for the boundary-constrained system the wave example actually steps. A
boundary-aware, matrix-free solve for a separable operator is `bramble-plan`'s v3.3.0 subplan
S5.2, layered through a `Kronecker.jl` extension and its fast-diagonalisation solve -- not
something this page measures or assumes exists yet. No wall-clock comparison is made here
between the two evaluation strategies; the byte counts above are a live structural comparison,
not a timing claim, and no crossover mesh size at which one becomes faster than the other has
been measured for this repository.

## Steady vs. unsteady problems

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
(the memory argument from the SPD section above, now applied per step rather than once), where
the reduction above is free on top of whatever iteration count the preconditioner alone already
bought.

## macOS: when the default solve reaches for Apple Accelerate

[`pde_solve`](@ref)'s `:default` solver -- the one every `\` in the sections above sits on
top of -- reaches for Apple's Accelerate framework instead of a bare `A \ F` on macOS, but
only when `AppleAccelerate.jl` is loaded (`using AppleAccelerate`) **and** the system is
symmetric:

```julia
using Bramble, AppleAccelerate  # loading it is enough to opt in; no other setup

A_sym, F_sym = spd_system(80)                      # symmetric: Accelerate's Cholesky/LDLᵀ path
A_uns, F_uns = convection_diffusion_system(80)     # unsymmetric: stays on `A \ F`

u_sym = pde_solve(A_sym, F_sym)   # == pde_solve(A_sym, F_sym; solver = :accelerate)
u_uns = pde_solve(A_uns, F_uns)   # == A_uns \ F_uns, Accelerate never runs
```

This is narrower than Accelerate's own availability suggests, and deliberately so
([gpena/Bramble.jl#246](https://github.com/gpena/Bramble.jl/issues/246)): measured against
`A \ F` on Bramble-shaped systems, the symmetric path is a 1.2-1.3x win (`n = 80`: 0.83x
the runtime; `n = 120`: 0.78x), while an unsymmetric convection-diffusion system was
2.3-3.6x **slower** through Accelerate before the dispatch was narrowed to
`issymmetric(A)`. `solver = :accelerate` still honours an explicit request on an
unsymmetric system -- only the automatic `:default` choice avoids it.

Symmetry can be asserted rather than detected: `sym = :spd`/`:definite`/`:symmetric` takes
the Accelerate path without testing `issymmetric(A)` again, and `sym = :unsymmetric` skips
straight to `A \ F`. An unrecognised `sym` under `:default` does not error -- it silently
falls back to `A \ F`, the same as `:default` ignoring `sym` entirely before this dispatch
existed. `solver = :accelerate` spelled out explicitly does validate `sym` and throws on a
value it does not recognise, so the two are not symmetric in strictness.

The choice is macOS-only and opt-in: without `using AppleAccelerate`, or on Linux/Windows,
`:default` is exactly `A \ F`, unchanged. [`accelerate_factorize`](@ref) and
[`accelerate_solve`](@ref)'s own documentation covers the extension-scoping, threading and
accuracy questions [gpena/Bramble.jl#142](https://github.com/gpena/Bramble.jl/issues/142)
asked about Accelerate in full, including the measured worst-case residual against
`LinearAlgebra` and the `BLAS_THREADING_MULTI_THREADED`/`BLAS_THREADING_SINGLE_THREADED`
knob for vecLib's own internal threading.

## Where to go next

[`sparse_factorize`](@ref)'s own docstring lists every direct backend and when each is
preferred by problem size and platform. [`amg_preconditioner`](@ref) and
[`ilu_preconditioner`](@ref) cover the preconditioners themselves in more depth, including
[gpena/Bramble.jl#244](https://github.com/gpena/Bramble.jl/issues/244)'s full evaluation of
the wider JuliaSparse ecosystem. [`KroneckerLinearOperator`](@ref) and [`is_separable`](@ref)
cover the matrix-free path in more depth than the wave-equation section above needs, and
[`CpuSerial`](@ref), [`CpuThreaded`](@ref), and [`CpuPolyester`](@ref) each document their own
measured crossover in full, workload by workload, rather than the summary table above. The
[elasticity](../examples/elasticity_3d.md) and [wave equation](../examples/wave_equation_2d.md)
tutorials are where the two new archetypes above are actually derived and solved end to end;
this page only adds the solver comparison on top of them.
