```@meta
CurrentModule = Bramble
CollapsedDocStrings = false
```

# Automatic differentiation

**What you will learn.** How to differentiate a discrete quantity with respect to a parameter, from a single derivative to the Jacobian of a nonlinear residual, and how to choose an automatic differentiation backend.

**What you need first.** The [space tutorial](@ref tutorial_space), for grid spaces and discrete functions, the [operators tutorial](@ref tutorial_operators), and the [form tutorial](@ref tutorial_form), for assembling a system with Dirichlet conditions.

**Where next.** The [solvers tutorial](solvers.md) covers what to do with the linear system once the Jacobian is large.

Bramble's operators, grid functions and form assembly are generic over the scalar type, so a discrete quantity can be differentiated with respect to whatever parameter produced it. The differentiation backends are reached through [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl), so switching between forward and reverse mode leaves your objective untouched. Every block below runs when this page is built.

---

## Differentiate with respect to a scalar

Take the discrete energy of a scaled sine, as a function of the scale ``a``:

```math
\mathcal{J}(a) = \| R_h(a \sin(\pi x)) \|_h^2 .
```

Analytically ``\int_0^1 \sin^2(\pi x)\,dx = \tfrac12``, so ``\mathcal{J}(a) \approx \tfrac12 a^2`` and ``d\mathcal{J}/da \approx a``. Build a small grid space and write the objective as an ordinary function of ``a``:

```@example autodiff_tutorial
using Bramble, DifferentiationInterface, ForwardDiff

Ω = domain(interval(0.0, 1.0))
Ωₕ = mesh(Ω, 32, true)
Wₕ = gridspace(Ωₕ)

function energy(a)
    uₕ = Rₕ(Wₕ, x -> a * sin(π * x[1]))
    return normₕ(uₕ)^2
end
nothing # hide
```

`energy` restricts a function to the grid with [`Rₕ`](@ref) and takes its discrete norm. Nothing in it mentions differentiation. Hand it to DifferentiationInterface together with a backend, here forward-mode `AutoForwardDiff()`, and a point:

```@example autodiff_tutorial
ad = AutoForwardDiff()
a_val = 2.0
dJ = DifferentiationInterface.derivative(energy, ad, a_val)
round(dJ, digits = 4)
```

The derivative is close to the analytic value ``a = 2``. It works because `Rₕ`, `normₕ` and the grid space accept the dual numbers ForwardDiff passes through them.

!!! tip "Try this"
    Set `a_val = 3.0`. The derivative follows the analytic ``d\mathcal{J}/da \approx a`` and returns about ``3``.

## Differentiate with respect to several parameters

When fitting parameters, such as source coefficients, you want the gradient of an objective with respect to a vector ``\mathbf{p}``. Use the discrete norm plus the ``H^1`` semi-norm [`snorm₁ₕ`](@ref):

```math
\mathcal{J}(\mathbf{p}) = \| u_h(\mathbf{p}) \|_h^2 + | u_h(\mathbf{p}) |_{1,h}^2 .
```

```@example autodiff_tutorial
function loss(p)
    uₕ = Rₕ(Wₕ, x -> p[1] * sin(π * x[1]) + p[2] * x[1] * (1.0 - x[1]))
    return normₕ(uₕ)^2 + snorm₁ₕ(uₕ)^2
end

p0 = [1.0, 2.0]
g_forward = DifferentiationInterface.gradient(loss, AutoForwardDiff(), p0)
round.(g_forward, digits = 4)
```

!!! tip "Try this"
    With many parameters, swap `AutoForwardDiff()` for a reverse-mode backend such as `AutoReverseDiff()` (after `using ReverseDiff`). The definition of `loss` does not change and the gradient agrees. The [backend table](@ref autodiff_backends) below says when each is worth it.

## Differentiate through a linear solve

A parameter can also enter the boundary data and the source of a linear system ``A u = F(p)``. Three things matter:

1. Build the Dirichlet conditions with `dirichlet_constraints`, using a closure that captures the parameter (see [Dirichlet conditions](@ref form_dirichlet)).
2. Allocate the source with the parameter's scalar type, `element(Wₕ, eltype(p))`.
3. Sparse direct solvers such as UMFPACK expect `Float64` entries. To differentiate through the solve with dual numbers, convert the matrix to dense with `Matrix(A) \ F`, or use an iterative solver.

```@example autodiff_tutorial
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
A = assemble(a; dirichlet = :boundary)
Adense = Matrix(A)

function solve_objective(p)
    bcs = dirichlet_constraints(Ω, :boundary => (x -> p[1]))   # boundary data depends on p

    gₕ = element(Wₕ, eltype(p))                                 # source in the tracked scalar type
    avgₕ!(gₕ, x -> p[2] * sin(π * x[1]))

    l = form(Wₕ, v -> innerₕ(gₕ, v))
    F = assemble(l; dirichlet = bcs)

    u = Adense \ F
    return sum(abs2, u)
end

p_init = [1.0, 2.0]
grad_solve = DifferentiationInterface.gradient(solve_objective, AutoForwardDiff(), p_init)
round.(grad_solve, digits = 4)
```

The two entries are the sensitivities with respect to the boundary value and the source scale. The matrix `A` does not depend on ``p``, so it is assembled once outside the function.

## Differentiate a nonlinear residual

For a nonlinear problem, the residual ``R(u) = A(u)u - F`` has a localized stencil, so computing its Jacobian ``\partial R / \partial u`` entry by entry with dense differentiation is wasteful. Sparse forward mode detects the sparsity pattern with `SparseConnectivityTracer.jl`, colours the columns once with `SparseMatrixColorings.jl`, and then evaluates the Jacobian in a compressed sweep. Here the diffusion coefficient depends on the state:

```@example autodiff_tutorial
using SparseArrays
import SparseConnectivityTracer, SparseMatrixColorings

const sparse_backend = AutoSparse(AutoForwardDiff();
    sparsity_detector = SparseConnectivityTracer.TracerSparsityDetector(),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

function pde_residual(u_vec)
    T = eltype(u_vec)
    uₕ = element(Wₕ, T)
    uₕ .= u_vec

    αvals = 1.0 .+ uₕ .^ 2                      # diffusion coefficient depends on the state
    a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇ₕ(U), ∇ₕ(V)))
    A_sparse = assemble(a; dirichlet = :boundary)

    F = ones(T, ndofs(Wₕ))
    return A_sparse * u_vec .- F
end

u0 = zeros(ndofs(Wₕ))
prep = DifferentiationInterface.prepare_jacobian(pde_residual, sparse_backend, u0)
J = DifferentiationInterface.jacobian(pde_residual, prep, sparse_backend, u0)

size(J), nnz(J)
```

The Jacobian is sparse: only the stencil's entries are stored, not all of ``n^2``. `prepare_jacobian` does the detection and colouring once, and `jacobian` reuses it at every point.

### Skip the tracer

`TracerSparsityDetector` works for any Julia function, so it has to run `pde_residual` once to find the pattern. Here that is wasted effort. The matrix comes from a `BilinearForm`, and a form's sparsity is already known from its AST. [`jacobian_pattern`](@ref) reads the pattern off the form and widens it by how each coefficient depends on the unknown.

In this residual `αvals` is a pointwise function of `uₕ` at the same grid point, with no averaging, so its dependency is just the trial placeholder `U -> U`. [`ast_sparsity_detector`](@ref) hands the result to `AutoSparse` in place of the tracer, once [ADTypes.jl](https://github.com/SciML/ADTypes.jl) is loaded:

```@example autodiff_tutorial
using ADTypes
import Bramble: ast_sparsity_detector

αvals0 = 1.0 .+ parent(element(Wₕ, 0.0)) .^ 2
a_for_pattern = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals0 * ∇ₕ(U), ∇ₕ(V)))

const native_backend = AutoSparse(AutoForwardDiff();
    sparsity_detector = ast_sparsity_detector(a_for_pattern, U -> U),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

prep_native = DifferentiationInterface.prepare_jacobian(pde_residual, native_backend, u0)
J_native = DifferentiationInterface.jacobian(pde_residual, prep_native, native_backend, u0)

J == J_native
```

`a_for_pattern` only needs some concrete coefficient to build a `BilinearForm`. The pattern belongs to the AST, not to the values of `αvals0`, so evaluating it at ``u = 0`` is as good as anywhere else. The Jacobian is the same, but no tracing pass paid for it, so `prepare_jacobian` gets cheaper relative to tracing as the mesh grows. The trade is scope: it applies only when the residual's matrix is assembled from a `BilinearForm`, as here. Composite trial and test spaces are supported too (see the [`jacobian_pattern`](@ref) docstring and [the coupled reaction-diffusion example](../examples/coupled_reaction_diffusion.md#Skipping-the-tracer-here-too)). Keep `TracerSparsityDetector` for the case where the matrix was built some other way.

---

## [Reference: choosing a backend](@id autodiff_backends)

The stencil kernels write into preallocated arrays, so a backend has to support array mutation. That rules out `Zygote.jl`. These five work:

| Backend | Mode | Reach for it when | Construction |
| :--- | :--- | :--- | :--- |
| ForwardDiff | forward | $\le 20$ parameters or Jacobian colours | `AutoForwardDiff()` |
| PolyesterForwardDiff | forward | the same, with chunks spread over threads | `AutoPolyesterForwardDiff()` |
| ReverseDiff | reverse | a scalar loss over many parameters, compiled in under a second | `AutoReverseDiff()` |
| Mooncake | reverse | the same, source-to-source, no tape recorded up front | `AutoMooncake(; config = nothing)` |
| Enzyme | reverse | many parameters and the tightest reverse-mode cost | see below |

Enzyme needs two annotations whenever the differentiated closure captures a mesh or a grid space, which in this package it almost always does:

```@example autodiff_tutorial
using Enzyme

enzyme_backend = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const)
```

`set_runtime_activity` lets Enzyme track activity through the captured geometry, and `Enzyme.Const` declares the function object itself, which holds those references, as not differentiated. [`pde_solve`](@ref) carries its own `EnzymeRules` adjoint, so differentiating through a linear solve needs nothing beyond `using Enzyme`.

Use ForwardDiff for a handful of parameters, a directional derivative, or a colour-compressed sparse Jacobian. Use ReverseDiff, Mooncake or Enzyme once the parameter count passes a few dozen and the objective is scalar, because one reverse sweep then replaces one forward sweep per parameter. Enzyme is the fastest of the three and the one with an adjoint rule for `pde_solve`. ReverseDiff compiles fastest.

Sparse forward mode stays competitive far longer than the parameter count suggests, because the colour count is set by the stencil rather than the mesh: a Cartesian difference stencil needs the same handful of colours at every resolution. Once storing or factorizing the Jacobian is the binding constraint, the [solvers tutorial](solvers.md) covers the linear system itself.
