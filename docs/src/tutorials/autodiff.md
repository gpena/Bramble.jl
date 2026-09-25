```@meta
CollapsedDocStrings = false
```

# Automatic differentiation tutorial

Bramble's operators, grid functions and form assembly are generic over the scalar type, so
a discrete quantity can be differentiated with respect to whatever parameter produced it.
Backends reach the code through
[DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl),
so switching between forward and reverse mode leaves the objective untouched.

---

## 1. Backends

The stencil kernels write into preallocated arrays, so a backend has to support array
mutation. That rules out `Zygote.jl`; the five below work.

| Backend | Mode | Reach for it when | Construction |
| :--- | :--- | :--- | :--- |
| ForwardDiff | forward | $\le 20$ parameters or Jacobian colours | `AutoForwardDiff()` |
| PolyesterForwardDiff | forward | the same, with chunks spread over threads | `AutoPolyesterForwardDiff()` |
| ReverseDiff | reverse | a scalar loss over many parameters, compiled in under a second | `AutoReverseDiff()` |
| Mooncake | reverse | the same, source-to-source, no tape recorded up front | `AutoMooncake(; config = nothing)` |
| Enzyme | reverse | many parameters and the tightest reverse-mode cost | see below |

Enzyme needs two annotations whenever the differentiated closure captures a mesh or a grid
space, which in this package it almost always does:

```@example autodiff_tutorial
using Bramble, Enzyme, DifferentiationInterface
import Bramble: ast_sparsity_detector

enzyme_backend = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const)
```

`set_runtime_activity` lets Enzyme track activity through the captured geometry, and
`Enzyme.Const` declares the function object itself, which holds those references, as not
differentiated. [`pde_solve`](@ref) additionally carries its own `EnzymeRules` adjoint, so
differentiating through a linear solve needs nothing beyond `using Enzyme`.

---

## 2. Use case: scalar parameter sensitivity

In this use case, we compute the sensitivity of a discrete energy norm with respect to a scalar scaling factor $a$:

```math
\mathcal{J}(a) = \| R_h(a \sin(\pi x)) \|_h^2
```

Analytically, $\int_0^1 \sin^2(\pi x)\,dx = \frac{1}{2}$, so $\mathcal{J}(a) \approx \frac{1}{2} a^2$ and $\frac{d\mathcal{J}}{da} \approx a$.

```@example autodiff_tutorial
using ForwardDiff

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

## 3. Use case: multi-parameter gradient for inverse problems

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

## 4. Use case: differentiating through Dirichlet constraints and linear solves

Parameter sensitivities can also enter boundary conditions and linear form sources.

When solving $A u = F(p)$:
1. Boundary conditions are constructed with `dirichlet_constraints` using a closure that captures the parameter.
2. The vector element for the source term must be allocated with the parameter's scalar type: `element(Wₕ, eltype(p))`.
3. Standard sparse solvers (UMFPACK) expect `Float64` entries. For automatic differentiation through the solve with dual numbers, convert the assembled matrix to dense format `Matrix(A) \ F` or use an iterative solver:

```@example autodiff_tutorial
# Discrete Laplacian on nonuniform grid
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
A = assemble(a; dirichlet = :boundary)
Adense = Matrix(A)

function solve_objective(p)
    # 1. Parameter-dependent Dirichlet boundary data
    bcs = dirichlet_constraints(Ω, :boundary => (x -> p[1]))

    # 2. Source term allocated with the dual/tracked scalar type
    gₕ = element(Wₕ, eltype(p))
    avgₕ!(gₕ, x -> p[2] * sin(π * x[1]))

    # 3. Assemble right-hand side with constraints
    l = form(Wₕ, v -> innerₕ(gₕ, v))
    F = assemble(l; dirichlet = bcs)

    # 4. Linear solve and scalar objective
    u = Adense \ F
    return sum(abs2, u)
end

p_init = [1.0, 2.0]
grad_solve = DifferentiationInterface.gradient(solve_objective, AutoForwardDiff(), p_init)
println("Sensitivity w.r.t. [boundary_val, source_scale]: ", round.(grad_solve, digits = 4))
```

---

## 5. Use case: sparse Jacobian for nonlinear PDE residuals

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
    a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇ₕ(U), ∇ₕ(V)))
    A_sparse = assemble(a; dirichlet = :boundary)

    F = ones(T, ndofs(Wₕ))
    return A_sparse * u_vec .- F
end

u0 = zeros(ndofs(Wₕ))
prep = DifferentiationInterface.prepare_jacobian(pde_residual, sparse_backend, u0)
J = DifferentiationInterface.jacobian(pde_residual, prep, sparse_backend, u0)

println("Jacobian dimensions: ", size(J))
println("Non-zero entries: ", nnz(J))
```

`TracerSparsityDetector` works this way for *any* Julia function, which is exactly why it has to
run `pde_residual` once to find out. `pde_residual` here is not arbitrary, though: its matrix
`A_sparse` comes from a `BilinearForm`, whose own sparsity is already known directly
from its AST: no tracing needed for that part. [`jacobian_pattern`](@ref) reads that
pattern off the form, widened by the reach of each coefficient's own dependence on the
unknown, named the same way a form term names an operator: a function of the trial
placeholder. `αvals` here is `1.0 .+ uₕ.^2`, a plain pointwise function of `uₕ` at the *same*
grid point (no averaging, unlike the staggered `Mₕ(u)` coefficient in
[the nonlinear Poisson example](../examples/poisson_nonlinear.md)), so its dependency is
just the trial placeholder itself, `U -> U`. [`ast_sparsity_detector`](@ref) hands the
result straight to `AutoSparse` in place of the tracer, once
[ADTypes.jl](https://github.com/SciML/ADTypes.jl) is loaded:

```@example autodiff_tutorial
using ADTypes

αvals0 = 1.0 .+ parent(element(Wₕ, 0.0)) .^ 2
a_for_pattern = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals0 * ∇ₕ(U), ∇ₕ(V)))

const native_backend = AutoSparse(AutoForwardDiff();
    sparsity_detector = ast_sparsity_detector(a_for_pattern, U -> U),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

prep_native = DifferentiationInterface.prepare_jacobian(pde_residual, native_backend, u0)
J_native = DifferentiationInterface.jacobian(pde_residual, prep_native, native_backend, u0)

J == J_native
```

`a_for_pattern` only needs *some* concrete coefficient to build a `BilinearForm` from: the
pattern is a property of the AST, not of `αvals0`'s values, so evaluating it at `u = 0` is as
good as evaluating it at any other point. Same Jacobian either way, but no tracing pass paid
for it: `jacobian_pattern` only ever walks the grid once, touching neither `ForwardDiff` nor
the coefficient's actual values, so `prepare_jacobian` gets cheaper as the mesh grows rather
than scaling with however long one residual call takes to trace. The trade is scope, not
correctness: it only applies when the residual's matrix is assembled from a `BilinearForm`
in the first place (as here) -- composite trial/test spaces are supported too (see
[`jacobian_pattern`](@ref)'s own docstring, and
[the coupled reaction-diffusion example](../examples/coupled_reaction_diffusion.md#Skipping-the-tracer-here-too)).
`TracerSparsityDetector` above keeps working regardless of how `A_sparse` was built,
which is the case to reach for it.

---

## 6. Choosing a backend

ForwardDiff for a handful of parameters, a directional derivative, or a colour-compressed
sparse Jacobian. ReverseDiff, Mooncake or Enzyme once the parameter count grows past a few
dozen and the objective is scalar: one reverse sweep then replaces one forward sweep per
parameter. Enzyme is the fastest of the three here and the one with an adjoint rule for
[`pde_solve`](@ref); ReverseDiff compiles fastest.

Sparse forward mode stays competitive far longer than the parameter count suggests, because
the colour count is set by the stencil rather than by the mesh: a Cartesian difference
stencil needs the same handful of colours at every resolution. Past the point where storing
or factorizing the Jacobian is the binding constraint, the
[solvers tutorial](solvers.md) covers what to do with the linear system itself.
