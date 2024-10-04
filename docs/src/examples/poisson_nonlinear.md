```@meta
CurrentModule = Bramble
```

# Nonlinear Poisson equation

In this section, we'll demonstrate how to utilize `Bramble.jl` to solve a nonlinear Poisson equation with Dirichlet boundary conditions.

---

## Problem description

Let's consider the following nonlinear Poisson equation on a `1`-dimensional square domain ``\Omega = [0,1]``,

```math
\begin{align*}
- \frac{\partial}{\partial x} \left( \alpha(u) \frac{\partial u}{\partial x} (x) \right) &= f(x), \, x \in \Omega \\
u(x) &= g(x), \, x \in \partial \Omega.
\end{align*}
```

We define

```math
\alpha (u) = 3 + \frac{1} { 1 + u^2}
```

and

```math
u(x,y) = e^{x + y}, \, (x,y) \in [0,1]^2.
```

Function `f` and `g` are calculated such that `u` is the exact solution of the problem.

## Discretization

We refer to [Linear Poisson equation](@ref) for most of the notations used. To discretize the problem above, we just need to introduce an averaging operator on grid functions

```math
M_h (u_h)(x_i) = \frac{u_h(x_i) + u_h(x_{i-1})}{2}.
```

This allows to discretize the differential problem as the following variational problem

> find ``u_h \in W_h(\overline{\Omega}_h)``, with ``u_h(x_i) = u(x_i)`` on ``\partial \overline{\Omega}_h,`` such that
>
>```math
>(\alpha(u_h) D_{-x} u_h, D_{-x} v_h)_+ = ((g)_h, v_h)_h, \, \forall v_h  \in W_{h,0}(\overline{\Omega}_h)
>```

## Implementation

To solve this nonlinear problem, we can use a standard fixed point iteration.

We start by loading the packages needed

```julia
using Bramble
using LinearSolve
using ILUZero      # for reusable sparsity pattern
```

and define the domain and relevant functions to the problem

```julia
I = interval(0, 1)
Ω = domain(I)

sol = @embed(Ω, x -> exp(x[1]))
coeff = @embed(I, u -> 3 + 1 / (1 + u[1]^2))
Ap = @embed(I, u -> -2 * u[1] / (1 + u[1]^2)^2)
g = @embed(Ω, x -> -d * Ap(sol(x)) * sol(x)^2 - d * A(sol(x)) * sol(x))
```

Next, we define a mesh and reinterpret the solution and right hand side functions as being defined on the mesh.

```julia
Mh = mesh(Ω, (10, 20), (true, false))
sol = @embed(Mh, sol)
rhs = @embed(Mh, rhs)
bc = dirichletbcs(sol)
```

Now we define the space of grid functions and do some calculations for the right hand side

```julia
Wh = gridspace(Mh)
u = element(Wh, 0)

uold = similar(u)
avgₕ!(uold, rhs)
```

Next, we introduce the linear and bilinear forms associated with the problem. Here we use a `uold` vector which is due the linearization of the nonlinear function we had before.

```julia
l(V) = innerₕ(uold, V)
lform = form(l, Wh)
F = assemble(lform, bc)

A(u) = D == 1 ? coeff.(M₋ₕ(u)) : sum(ntuple(i -> coeff.(M₋ₕ(u)[i]), D)) ./ D
a(U, V) = inner₊(A(u) * ∇₋ₕ(U), ∇₋ₕ(V))
bform = form(a, Wh, Wh)
mat = assemble(bform, bc)
```

We are aiming at calculating the fixed point of

```math
(\alpha(u_h) D_{-x} u_h, D_{-x} v_h)_+ = (u_h, v_h)_h
```

by using the iterative scheme: given `u_{h,0}`, solve for `n=1,\dots`

```math
\left(\alpha(u_{h}^{(n)}) D_{-x} u_h, D_{-x} v_h \right)_+ = (u_h, v_h)_h
```

This is basically implemented in the following function

```julia
function fixed_point!(A, F, bform, bc, uold, u, α)
  prec = ilu0(A)
  prob = LinearProblem(A, F, KrylovJL_GMRES(), Pl = prec)
  linsolve = init(prob)

  for i in 1:2000
    uold.values .= α(u)
    assemble!(A, bform, bc)

    uold .= u

    linsolve.A = A
    sol = solve!(linsolve)

    u.values .= sol.u
    uold.values .-= u.values

    if norm₁ₕ(uold) < 1e-10
      break
    end
  end
end
```

and we implemented a simple stopping criteria based on the [norm₁ₕ](@ref) of the the difference between two consecutive iterations. Finally, we just need to call the fixed point function and we are done

```julia
fixed_point!(mat, F, bform, bc, uold, u, A)
u .-= uold
```

Vector `u` has the approximate solution to `u_h`.
