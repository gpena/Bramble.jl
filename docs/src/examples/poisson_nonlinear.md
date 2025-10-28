```@meta
CurrentModule = Bramble
```

# Nonlinear Poisson equation

In this section, we'll demonstrate how to utilize `Bramble.jl` to solve a nonlinear Poisson equation with Dirichlet boundary conditions.

---

## Problem description

Let's consider the following nonlinear Poisson equation on a $1$-dimensional square domain $\Omega = [0,1]$,

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

Function $f$ and $g$ are calculated such that $u$ is the exact solution of the problem.

## Discretization

We refer to [Linear Poisson equation](@ref) for most of the notations used. To discretize the problem above, we just need to introduce an averaging operator on grid functions

```math
M_h (u_h)(x_i) = \frac{u_h(x_i) + u_h(x_{i-1})}{2}.
```

This allows to discretize the differential problem as the following variational problem

> find $u_h \in W_h(\overline{\Omega}_h)$, with $u_h(x_i) = u(x_i)$ on $\partial \overline{\Omega}_h$, such that
>
> ```math
> (\alpha(u_h) D_{-x} u_h, D_{-x} v_h)_+ = ((g)_h, v_h)_h, \, \forall v_h  \in W_{h,0}(\overline{\Omega}_h)
> ```

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

sol(x) = exp(x[1])
α(u) = 3 + 1 / (1 + u[1]^2)
dαdu(u) = -2 * u[1] / (1 + u[1]^2)^2
g(x) = -d * dαdu(sol(x)) * sol(x)^2 - d * α(sol(x)) * sol(x)
```

Next, we define the mesh, the gridspace associated and dirichlet constraints objects

```julia
Ωₕ = mesh(Ω, (10, 20), (true, false))
Wₕ = gridspace(Ωₕ)
bcs = dirichlet_constraints(X, :boundary => x -> sol(x))
```

Now we define an auxiliar element to store the approximate solution `uₙ` and calculate the right hand side `u₀` using the average interpolator for our future linear system.

```julia
uₙ = element(Wₕ, 0)

u₀ = similar(uₙ)
avgₕ!(u₀, rhs)
```

Next, we introduce the linear and bilinear forms associated with the problem. Here we use a `u₀` vector which is due to the linearization of the nonlinear function we had before.

```julia
lₕ = form(Wₕ, vₕ -> innerₕ(u₀, vₕ))
F = assemble(lₕ, bcs, dirichlet_labels = :boundary)

αₕ(u) = (α.(M₋ₓ(u)) + α.(M₋ᵧ(u))) ./ 2
aₕ = form(Wₕ, Wₕ, (U, V) -> inner₊(αₕ(uₙ) * ∇₋ₕ(U), ∇₋ₕ(V)))
A = assemble(aₕ, dirichlet_labels = :boundary)
```

We are aiming at calculating the fixed point of the solution of

```math
(\alpha_h(u_h) D_{-x} u_h, D_{-x} v_h)_+ = (u_h, v_h)_h, \, \forall v_h  \in W_{h,0}(\overline{\Omega}_h)
```

by using the iterative scheme: given $u_h^{(0)}$, solve for $n=1,\dots$

```math
\left(\alpha(u_{h}^{(n)}) D_{-x} u_h, D_{-x} v_h \right)_+ = (u_h, v_h)_h
```

This is basically implemented in the following function

```julia
function fixed_point!(matrix, rhs, aₕ, uₚᵣₑᵥ, uₙₑₓₜ, coeff)
	prec = ilu0(matrix)
	prob = LinearProblem(matrix, rhs, KrylovJL_GMRES(), Pl = prec)
	linsolve = init(prob)

	for i in 1:2000
		uₚᵣₑᵥ .= coeff(uₙₑₓₜ)
		assemble!(matrix, aₕ, dirichlet_labels = :boundary)

		uₚᵣₑᵥ .= uₙₑₓₜ
		if i % 10 == 0
			ilu0!(prec, matrix)
		end

		linsolve.A = matrix
		sol = solve!(linsolve)

		uₙₑₓₜ .= sol
		uₚᵣₑᵥ .-= uₙₑₓₜ

		if norm₁ₕ(uₚᵣₑᵥ) < 1e-3
			break
		end
	end
end
```

and we implemented a simple stopping criteria based on the [norm₁ₕ](@ref) of the difference between two consecutive iterations. Finally, we just need to call the fixed point function and we are done

```julia
fixed_point!(A, F, aₕ, u₀, uₙ, αₕ)
```

Vector `uₙ` has the approximate solution to $u_h$.
