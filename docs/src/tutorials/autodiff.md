```@meta
CollapsedDocStrings = false
```

# Automatic differentiation tutorial

`Bramble.jl` is generic over its scalar types, allowing discrete operators, grid functions, and weak form assemblies to be differentiated with Julia's automatic differentiation (AD) ecosystem.

This tutorial covers:
1. The 5 AD backends supported by `Bramble.jl` through [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl).
2. Configuration details and annotations required by each backend.
3. Four practical use cases: scalar parameter sensitivity, multi-parameter gradients for inverse problems, differentiating through Dirichlet boundary solves, and sparse Jacobian assembly for nonlinear PDEs.
4. Guidelines for choosing the best backend for a given problem.

---

## 1. Supported differentiation backends

Because `Bramble.jl` implements difference stencils and discrete operators via mutating loops (`setindex!`) over preallocated arrays, AD backends that support array mutation work out of the box.

Standardizing calls through `DifferentiationInterface.jl` allows switching between forward and reverse backends without changing user objective functions:

| Backend | Mode | Primary strength | Required packages |
| :--- | :--- | :--- | :--- |
| **ForwardDiff** | Forward | Zero setup, fastest for $\le 20$ parameters and Jacobian columns | `ForwardDiff` |
| **PolyesterForwardDiff** | Forward | Multi-threaded forward chunks across CPU threads | `PolyesterForwardDiff` |
| **ReverseDiff** | Reverse | Tape-based reverse mode, sub-second compile time for scalar losses | `ReverseDiff` |
| **Mooncake** | Reverse | Modern source-to-source reverse mode supporting mutation | `Mooncake` |
| **Enzyme** | Reverse | LLVM-level AD with peak performance for large parameter counts | `Enzyme` |

> [!NOTE]
> Backends that forbid array mutation (such as `Zygote.jl`) cannot differentiate Bramble's stencil kernels.

---

## 2. Setting up the backends

### ForwardDiff

[`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) propagates `Dual` numbers through arithmetic and array operations. It requires no configuration:

```julia
using ForwardDiff, DifferentiationInterface

backend = AutoForwardDiff()
```

### PolyesterForwardDiff

[`PolyesterForwardDiff.jl`](https://github.com/JuliaSIMD/PolyesterForwardDiff.jl) parallelises `ForwardDiff` chunk evaluations over threads with low scheduling overhead:

```julia
using PolyesterForwardDiff, DifferentiationInterface

backend = AutoPolyesterForwardDiff()
```

### ReverseDiff

[`ReverseDiff.jl`](https://github.com/JuliaDiff/ReverseDiff.jl) records operations to a tape and evaluates gradients in reverse mode. It compiles quickly (typically $< 1$ s) and handles mutating operations:

```julia
using ReverseDiff, DifferentiationInterface

backend = AutoReverseDiff()
```

### Mooncake

[`Mooncake.jl`](https://github.com/compintell/Mooncake.jl) is a source-to-source reverse-mode AD tool that handles Julia language features and mutating routines:

```julia
using Mooncake, DifferentiationInterface

backend = AutoMooncake(; config = nothing)
```

### Enzyme

[`Enzyme.jl`](https://github.com/EnzymeAD/Enzyme.jl) performs differentiation at the LLVM compiler level. When differentiating closures that capture grid spaces or mesh objects, two settings are required:

1. `Enzyme.set_runtime_activity(Enzyme.Reverse)`: allows Enzyme to track activity through captured geometry.
2. `function_annotation = Enzyme.Const`: declares that the function object itself (which holds references to the mesh and grid space) is constant and not differentiated:

```julia
using Enzyme, DifferentiationInterface

mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
backend = AutoEnzyme(; mode = mode, function_annotation = Enzyme.Const)
```

---

## 3. Use case: scalar parameter sensitivity

In this use case, we compute the sensitivity of a discrete energy norm with respect to a scalar scaling factor $a$:

```math
\mathcal{J}(a) = \| R_h(a \sin(\pi x)) \|_h^2
```

Analytically, $\int_0^1 \sin^2(\pi x)\,dx = \frac{1}{2}$, so $\mathcal{J}(a) \approx \frac{1}{2} a^2$ and $\frac{d\mathcal{J}}{da} \approx a$.

```@example autodiff_tutorial
using Bramble, ForwardDiff, DifferentiationInterface

Ω = domain(interval(0.0, 1.0))
Ωₕ = mesh(Ω, 32, true)
Wₕ = gridspace(Ωₕ)

# Scalar objective function
function energy(a)
    uₕ = Rₕ(Wₕ, x -> a * sin(π * x[1]))
    return normₕ(uₕ)^2
end

backend = AutoForwardDiff()
a_val = 2.0
dJ = DifferentiationInterface.derivative(energy, backend, a_val)
println("dJ/da at a = $a_val: ", round(dJ, digits = 4))
```

---

## 4. Use case: multi-parameter gradient for inverse problems

When estimating parameters (such as source coefficients or material properties), we compute the gradient of an objective functional with respect to a parameter vector $\mathbf{p}$:

```math
\mathcal{J}(\mathbf{p}) = \| u_h(\mathbf{p}) \|_h^2 + | u_h(\mathbf{p}) |_{1,h}^2
```

where $|\cdot|_{1,h}$ is the discrete $H^1$ semi-norm computed via [`snorm₁ₕ`](@ref).

```@example autodiff_tutorial
function loss(p)
    uₕ = Rₕ(Wₕ, x -> p[1] * sin(π * x[1]) + p[2] * x[1] * (1.0 - x[1]))
    return normₕ(uₕ)^2 + snorm₁ₕ(uₕ)^2
end

p0 = [1.0, 2.0]
g_forward = DifferentiationInterface.gradient(loss, AutoForwardDiff(), p0)
println("ForwardDiff gradient: ", round.(g_forward, digits = 4))
```

For problems with many parameters, swapping `AutoForwardDiff()` for `AutoReverseDiff()` or `AutoEnzyme(...)` uses reverse mode without changing the definition of `loss`.

---

## 5. Use case: differentiating through Dirichlet constraints and linear solves

Parameter sensitivities can also enter boundary conditions and linear form sources.

When solving $A u = F(p)$:
1. Boundary conditions are constructed with `dirichlet_constraints` using a closure that captures the parameter.
2. The vector element for the source term must be allocated with the parameter's scalar type: `element(Wₕ, eltype(p))`.
3. Standard sparse solvers (UMFPACK) expect `Float64` entries. For automatic differentiation through the solve with dual numbers, convert the assembled matrix to dense format `Matrix(A) \ F` or use an iterative solver:

```@example autodiff_tutorial
# Discrete Laplacian on nonuniform grid
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
A = assemble(a; dirichlet_labels = :boundary)
Adense = Matrix(A)

function solve_objective(p)
    # 1. Parameter-dependent Dirichlet boundary data
    bcs = dirichlet_constraints(Bramble.set(Ω), :boundary => (x -> p[1]))

    # 2. Source term allocated with the dual/tracked scalar type
    gₕ = element(Wₕ, eltype(p))
    avgₕ!(gₕ, x -> p[2] * sin(π * x[1]))

    # 3. Assemble right-hand side with constraints
    l = form(Wₕ, v -> innerₕ(gₕ, v))
    F = assemble(l; dirichlet_conditions = bcs, dirichlet_labels = :boundary)

    # 4. Linear solve and scalar objective
    u = Adense \ F
    return sum(abs2, u)
end

p_init = [1.0, 2.0]
grad_solve = DifferentiationInterface.gradient(solve_objective, AutoForwardDiff(), p_init)
println("Sensitivity w.r.t. [boundary_val, source_scale]: ", round.(grad_solve, digits = 4))
```

---

## 6. Use case: sparse Jacobian for nonlinear PDE residuals

For nonlinear PDEs, residuals $R(u) = A(u)u - F$ have a localized stencil structure. Computing the Jacobian $\partial R / \partial u$ entry-by-entry with dense AD is wasteful.

Using sparse forward-mode AD with `SparseConnectivityTracer.jl` and `SparseMatrixColorings.jl`, the sparsity pattern is detected and columns are colored once. The Jacobian is then evaluated in a compressed sweep:

```@example autodiff_tutorial
using SparseArrays
import SparseConnectivityTracer, SparseMatrixColorings

const sparse_backend = AutoSparse(AutoForwardDiff();
    sparsity_detector = SparseConnectivityTracer.TracerSparsityDetector(),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

# Nonlinear residual with state-dependent diffusion
function pde_residual(u_vec)
    T = eltype(u_vec)
    uₕ = element(Wₕ, T)
    uₕ .= u_vec

    # Local diffusion coefficient depending on the state
    αvals = 1.0 .+ uₕ .^ 2
    a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇₋ₕ(U), ∇₋ₕ(V)))
    A_sparse = assemble(a; dirichlet_labels = :boundary)

    F = ones(T, ndofs(Wₕ))
    return A_sparse * u_vec .- F
end

u0 = zeros(ndofs(Wₕ))
prep = DifferentiationInterface.prepare_jacobian(pde_residual, sparse_backend, u0)
J = DifferentiationInterface.jacobian(pde_residual, prep, sparse_backend, u0)

println("Jacobian dimensions: ", size(J))
println("Non-zero entries: ", nnz(J))
```

---

## 7. Choosing the right backend

- **Use ForwardDiff** for low-dimensional parameter sets ($N \le 20$), directional derivatives, and column-colored sparse Jacobians.
- **Use PolyesterForwardDiff** when evaluating forward sweeps across multi-threaded CPU environments.
- **Use ReverseDiff** when differentiating a scalar objective with respect to many parameters ($N > 50$), especially when low compilation latency is desired.
- **Use Mooncake** for reverse-mode sensitivity without tape pre-recording.
- **Use Enzyme** when peak reverse-mode performance and minimal memory footprint are required, keeping in mind the need for `set_runtime_activity` and `function_annotation = Enzyme.Const`.

### Large sparse nonlinear PDEs

For large systems stemming from nonlinear PDEs ($N \sim 10^5 - 10^7+$ degrees of freedom), both Jacobian construction and linear solves require scaling strategies:

1. **Compressed sparse forward AD ($N \le 10^6$)**:
   Because Bramble's Cartesian difference stencils have local footprints, the column-coloring count $C$ is independent of grid size $N$ (typically $4$–$9$ colors in 2D, $8$–$27$ in 3D). `AutoSparse(AutoForwardDiff())` evaluates the full sparse Jacobian in only $C$ directional sweeps. Always reuse `prepare_jacobian` and evaluate into preallocated storage using `DifferentiationInterface.jacobian!`.
2. **Jacobian-free Newton-Krylov ($N > 10^6$ or 3D)**:
   When storing or factoring the sparse Jacobian exceeds available memory, switch to matrix-free Krylov iterations (e.g. GMRES or BiCGStab via `LinearSolve.jl` or `Krylov.jl`). Krylov solvers require only Jacobian-vector products $J(u) \cdot v$, which forward-mode AD computes directly via pushforwards (`DifferentiationInterface.pushforward`) at the cost of one extra residual evaluation per Krylov step, requiring zero Jacobian storage.
3. **Preconditioning and linear solvers**:
   Direct factorization (`\`, UMFPACK) scales poorly in 3D due to fill-in. Pair iterative Krylov solvers with incomplete factorizations (`IncompleteLU.jl`) or algebraic multigrid (`AlgebraicMultigrid.jl`) built on a lagged or Picard linearization $A(u^k)$.
4. **Allocation-free residual evaluations**:
   Newton loops repeatedly evaluate residuals under `Float64` during line searches and `ForwardDiff.Dual` during Jacobian sweeps. Use `PreallocationTools.jl` (`DiffCache`) or pre-allocated scratch buffers to eliminate intermediate allocations in `residual(u)`.

| Problem scale | Jacobian strategy | Linear solve | Recommended tooling |
| :--- | :--- | :--- | :--- |
| **$N \le 10^5$ (2D)** | Compressed sparse AD | Direct sparse LU (`A \ F`) | `DifferentiationInterface` + `SparseMatrixColorings` |
| **$10^5 < N \le 10^6$** | Compressed sparse AD (`jacobian!`) | Preconditioned GMRES / BiCGStab | `LinearSolve.jl` + `IncompleteLU.jl` or `AlgebraicMultigrid.jl` |
| **$N > 10^6$ (3D)** | Matrix-free JFNK (`pushforward`) | GMRES + Lagged preconditioner | `NonlinearSolve.jl` + `Krylov.jl` + `AlgebraicMultigrid.jl` |

