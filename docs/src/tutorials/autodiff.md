```@meta
CurrentModule = Bramble
```

# [Automatic differentiation](@id tutorial_autodiff)

**What you will learn.** How to differentiate a discrete quantity with respect to a parameter: a scalar, several parameters, a linear solve, a matrix, a nonlinear residual, the relation data of a ghost space and a semidiscrete right-hand side, and how to choose between the two backends.

**What you need first.** The [space tutorial](@ref tutorial_space), for grid spaces and discrete functions, the [operators tutorial](@ref tutorial_operators), and the [form tutorial](@ref tutorial_form), for assembling a system with Dirichlet conditions.

Bramble's operators, grid functions and forms are generic over the scalar type, so a discrete quantity can be differentiated with respect to the parameter that produced it. Derivatives go through [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) (DI), so the objective stays the same function whichever backend differentiates it. Two backends are offered: `AutoForwardDiff()` and `AutoEnzyme()`, which runs in reverse mode below and also in forward mode.

Each differentiating block ends in an assertion against a non-AD oracle: central differences, exact up to rounding for the quadratic objectives here, or a closed form.

## Differentiate with respect to a scalar

Take the discrete energy of a scaled sine, as a function of the scale ``a``:

```math
\mathcal{J}(a) = \| R_h(a \sin(\pi x)) \|_h^2 .
```

[`Rₕ`](@ref) is linear and the discrete norm is quadratic, so ``\mathcal{J}(a) = a^2 \mathcal{J}(1)`` exactly and ``\mathcal{J}'(a) = 2a\,\mathcal{J}(1)``. That closed form is the oracle. Build a grid space and write the objective as an ordinary function of ``a``:

```@example autodiff_tutorial
using Bramble, DifferentiationInterface, ForwardDiff

Ω = domain(interval(0.0, 1.0))
Ωₕ = mesh(Ω, 32; uniform = false)
Wₕ = gridspace(Ωₕ)

function energy(a)
    uₕ = Rₕ(Wₕ, x -> a * sin(π * x[1]))
    return normₕ(uₕ)^2
end

a_val = 2.0
dJ = DifferentiationInterface.derivative(energy, AutoForwardDiff(), a_val)
round(dJ, digits = 4)
@assert isapprox(dJ, 2 * a_val * energy(1.0); rtol = 1e-12)
```

`energy` restricts a function to the grid and takes its discrete norm. Nothing in it mentions differentiation.

## Differentiate with respect to several parameters

Fitting parameters, such as source coefficients, needs the gradient of an objective with respect to a vector ``\mathbf{p}``. This example uses the discrete norm plus the ``H^1`` semi-norm [`snorm₁ₕ`](@ref):

```math
\mathcal{J}(\mathbf{p}) = \| u_h(\mathbf{p}) \|_h^2 + | u_h(\mathbf{p}) |_{1,h}^2 .
```

The oracle is a central difference in each parameter, written out here:

```@example autodiff_tutorial
function fd_gradient(f, p; h = 1e-4)
    return [(f(p .+ h .* (eachindex(p) .== i)) - f(p .- h .* (eachindex(p) .== i))) / (2h)
            for i in eachindex(p)]
end

function loss(p)
    uₕ = Rₕ(Wₕ, x -> p[1] * sin(π * x[1]) + p[2] * x[1] * (1.0 - x[1]))
    return normₕ(uₕ)^2 + snorm₁ₕ(uₕ)^2
end

p0 = [1.0, 2.0]
g_forward = DifferentiationInterface.gradient(loss, AutoForwardDiff(), p0)
round.(g_forward, digits = 4)
@assert isapprox(g_forward, fd_gradient(loss, p0); rtol = 1e-8, atol = 1e-8)
```

Reverse mode with Enzyme differentiates the same function. Enzyme needs two annotations whenever the closure captures a mesh or a grid space, which in this package it almost always does. `set_runtime_activity` lets Enzyme track activity through the captured geometry, and `Enzyme.Const` declares the function object, which holds those references, as not differentiated.

```@example autodiff_tutorial
using Enzyme

enzyme_backend = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const)

g_enzyme = DifferentiationInterface.gradient(loss, enzyme_backend, p0)
round.(g_enzyme, digits = 4)
@assert isapprox(g_enzyme, fd_gradient(loss, p0); rtol = 1e-8, atol = 1e-8)
```

## Differentiate through a linear solve

A parameter can enter the boundary data and the source of a linear system ``A u = F(\mathbf{p})`` while ``A`` stays fixed. Three things matter:

1. Build the Dirichlet conditions with `dirichlet_constraints`, using a closure that captures the parameter (see [Dirichlet conditions](@ref form_dirichlet)).
2. Allocate the source with the parameter's scalar type, `element(Wₕ, eltype(p))`.
3. Call [`pde_solve`](@ref) on the `Dual` load as it is. With ForwardDiff loaded, the `Float64` values of `A` are factored once, and every partial is a solve against that factor. No dense conversion is needed.

```@example autodiff_tutorial
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
A = assemble(a; dirichlet = :boundary)

function solve_objective(p)
    bcs = dirichlet_constraints(Ω, :boundary => (x -> p[1]))   # boundary data depends on p

    gₕ = element(Wₕ, eltype(p))                                 # source in the tracked scalar type
    avgₕ!(gₕ, x -> p[2] * sin(π * x[1]))

    l = form(Wₕ, v -> innerₕ(gₕ, v))
    F = assemble(l; dirichlet = bcs)
    u = pde_solve(A, F)
    return sum(abs2, u)
end

p_init = [1.0, 2.0]
grad_solve = DifferentiationInterface.gradient(solve_objective, AutoForwardDiff(), p_init)
round.(grad_solve, digits = 4)
@assert isapprox(grad_solve, fd_gradient(solve_objective, p_init); rtol = 1e-8, atol = 1e-8)
```

The two entries are the sensitivities with respect to the boundary value and the source scale.

The same objective differentiates in reverse mode, because [`pde_solve`](@ref) carries its own `EnzymeRules` adjoint:

```@example autodiff_tutorial
g_solve_enzyme = DifferentiationInterface.gradient(solve_objective, enzyme_backend, p_init)
round.(g_solve_enzyme, digits = 4)
@assert isapprox(g_solve_enzyme, fd_gradient(solve_objective, p_init); rtol = 1e-8, atol = 1e-8)
```

## Differentiate through a matrix

A parameter in a coefficient enters the matrix itself. Here the diffusion scale is ``p_1``, and the objective is the squared norm of the solution:

```@example autodiff_tutorial
l_fixed = form(Wₕ, v -> innerₕ(x -> sin(π * x[1]), v))

function matrix_objective(p)
    a_p = form(Wₕ, Wₕ, (u, v) -> inner₊(p[1] * ∇ₕ(u), ∇ₕ(v)))
    A_p, F_p = assemble(a_p, l_fixed; dirichlet = :boundary => x -> 0.0)
    return sum(abs2, pde_solve(A_p, F_p))
end

g_matrix = DifferentiationInterface.gradient(matrix_objective, AutoForwardDiff(), [1.5])
round.(g_matrix, digits = 4)
@assert isapprox(g_matrix, fd_gradient(matrix_objective, [1.5]; h = 1e-5); rtol = 1e-6)
```

A `Dual` matrix goes to `pde_solve` directly, as the load does above. A non-square `Dual` matrix throws an `ArgumentError`: the derivative of a least-squares solve has a residual term that this path does not compute.

## Differentiate a nonlinear residual

For a nonlinear problem, the residual ``R(u) = A(u)u - F`` has a localized stencil, so computing its Jacobian ``\partial R / \partial u`` entry by entry with dense differentiation is wasteful. Sparse forward mode detects the sparsity pattern with `SparseConnectivityTracer.jl`, colours the columns once with `SparseMatrixColorings.jl`, and then evaluates the Jacobian in a compressed sweep. Here the diffusion coefficient depends on the state:

```@example autodiff_tutorial
using ADTypes, SparseArrays
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

u0 = parent(Rₕ(Wₕ, x -> 0.5 * sin(π * x[1])))   # nonzero state, so the coefficient's derivative is non-zero
prep = DifferentiationInterface.prepare_jacobian(pde_residual, sparse_backend, u0)
J = DifferentiationInterface.jacobian(pde_residual, prep, sparse_backend, u0)

size(J), nnz(J)

function fd_jacobian(f, u; h = 1e-6)
    n = length(u)
    cols = [(f(u .+ h .* (1:n .== j)) - f(u .- h .* (1:n .== j))) / (2h) for j in 1:n]
    return reduce(hcat, cols)
end

J_fd = fd_jacobian(pde_residual, u0)
@assert maximum(abs, Matrix(J) - J_fd) <= 1e-7 * maximum(abs, J_fd)
```

The Jacobian is sparse: the Jacobian stores only the stencil's entries. `prepare_jacobian` does the detection and colouring once, and `jacobian` reuses it at every point. The last line compares every entry with a central difference of the residual, one column at a time.

### Skip the tracer

`TracerSparsityDetector` works for any Julia function, so it has to run `pde_residual` once to find the pattern. Here that is wasted effort. The matrix comes from a `BilinearForm`, and a form's sparsity is already known from its AST. [`jacobian_pattern`](@ref) reads the pattern off the form and widens it by how each coefficient depends on the unknown.

In this residual `αvals` is a pointwise function of `uₕ` at the same grid point, with no averaging, so its dependency is just the trial placeholder `U -> U`. [`ast_sparsity_detector`](@ref) hands the result to `AutoSparse` in place of the tracer:

```@example autodiff_tutorial
import Bramble: ast_sparsity_detector

αvals0 = 1.0 .+ parent(element(Wₕ, 0.0)) .^ 2
a_for_pattern = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals0 * ∇ₕ(U), ∇ₕ(V)))

const native_backend = AutoSparse(AutoForwardDiff();
    sparsity_detector = ast_sparsity_detector(a_for_pattern, U -> U),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

prep_native = DifferentiationInterface.prepare_jacobian(pde_residual, native_backend, u0)
J_native = DifferentiationInterface.jacobian(pde_residual, prep_native, native_backend, u0)

@assert J == J_native
@assert maximum(abs, Matrix(J_native) - J_fd) <= 1e-7 * maximum(abs, J_fd)
```

`a_for_pattern` only needs some concrete coefficient to build a `BilinearForm`. Any concrete coefficient builds it, because the pattern is a property of the AST. The Jacobian is the same, and no tracing pass runs to find it. The detector applies when the residual's matrix is assembled from a `BilinearForm`. The [`jacobian_pattern`](@ref) docstring gives the rules, and the [coupled reaction-diffusion example](../examples/coupled_reaction_diffusion.md) uses the same detector on a coupled system. Keep `TracerSparsityDetector` for a matrix built some other way.

## Differentiate ghost relation data

A constrained ghost space carries its boundary conditions as relations (see [Neumann and Robin conditions with ghost points](@ref form_ghost)). A relation's data may depend on the parameter, and the derivative reaches the solution through the lift of that data into the load:

```@example autodiff_tutorial
function ghost_objective(p)
    C = constrained(gridspace(Ωₕ; ghosts = :boundary), u -> (
        :xmin => ∂ₙ(u) => 0.0, :xmax => u + ∂ₙ(u) => p[1]))
    A_gh, F_gh = assemble(form(C, C, (u, v) -> innerₕ(-Δₕ(u), v)),
        form(C, v -> innerₕ(x -> sin(π * x[1]), v)))
    return sum(abs2, pde_solve(A_gh, F_gh))
end

g_ghost = DifferentiationInterface.gradient(ghost_objective, AutoForwardDiff(), [1.0])
round.(g_ghost, digits = 4)
@assert isapprox(g_ghost, fd_gradient(ghost_objective, [1.0]); rtol = 1e-8, atol = 1e-8)
```

The solution is affine in the boundary datum, so the objective is quadratic in ``p_1``.

## Differentiate a semidiscrete right-hand side

[`semidiscretize_rhs`](@ref) returns the residual `(du, u, p, t)` of the method of lines. Time integrators differentiate it with respect to the state, so its Jacobian is a useful check:

```@example autodiff_tutorial
a_rhs = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
l_rhs = form(Wₕ, v -> innerₕ(x -> sin(π * x[1]), v))
sd = semidiscretize(a_rhs, l_rhs)
rhs = Bramble.semidiscretize_rhs(sd)

function rhs_at(u)
    du = similar(u)
    rhs(du, u, nothing, 0.0)
    return du
end

u_rhs = parent(Rₕ(Wₕ, x -> 0.5 * sin(π * x[1])))
J_rhs = DifferentiationInterface.jacobian(rhs_at, AutoForwardDiff(), u_rhs)

# The right-hand side is affine in the state, so differences of Float64 evaluations are exact.
unit(n, j) = Float64.(1:n .== j)
cols = [rhs_at(u_rhs .+ unit(length(u_rhs), j)) - rhs_at(u_rhs) for j in eachindex(u_rhs)]
@assert isapprox(J_rhs, reduce(hcat, cols); rtol = 1e-10)
```

[`semidiscretize`](@ref) takes a `hook_reads_state` keyword next to `update_coefficients!`. Give `hook_reads_state = false` only when the hook reads no state, such as a source that depends on ``t`` alone: then the `Dual` Jacobian is exact. A hook that does read the state keeps the default, and the docstring gives the details.

## [Reference: choosing a backend](@id autodiff_backends)

The stencil kernels write into preallocated arrays, so a backend has to support array mutation. Two backends do, both reached through DI:

| Backend | Mode | Reach for it when | Construction |
| :--- | :--- | :--- | :--- |
| ForwardDiff | forward | fewer than about 10 parameters, a directional derivative, or Jacobian colours | `AutoForwardDiff()` |
| Enzyme | reverse; forward everywhere except through `pde_solve` | about 16 parameters or more and a scalar objective | `AutoEnzyme` with the annotations shown above |

One reverse sweep replaces one forward sweep per parameter. Enzyme and ChainRulesCore (reverse) have an adjoint rule for `pde_solve`; DI uses the Enzyme one. Forward-mode Enzyme through `pde_solve` throws an `EnzymeNoDerivativeError`.

Sparse forward mode stays competitive on large meshes: a Cartesian difference stencil needs the same handful of colours at every resolution.

Bramble offers no other AD backend. The measurements behind the choice are in the repository's notes.
