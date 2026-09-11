# Heat equation

The first time-dependent problem in this manual. Space is discretised exactly as in the
[linear Poisson example](poisson_linear.md); time is left continuous and handed to a stepper
from the SciML stack. Every number below was produced by the code shown.

## Problem

```math
\partial_t u = \Delta u + f \text{ in } \Omega \times (0, T], \qquad u = g \text{ on }
\partial\Omega, \qquad u(\cdot, 0) = u_0,
```

on ``\Omega = (0,1)`` with the manufactured solution ``u_{\text{exact}}(x, t) = e^{-t}
\sin(\pi x)``, which vanishes on the boundary and forces

```math
f(x, t) = (\pi^2 - 1)\, e^{-t} \sin(\pi x).
```

## The method of lines

Discretising space alone leaves one ordinary differential equation per degree of freedom,

```math
M \frac{\mathrm{d} u_h}{\mathrm{d} t} = F(t) - A u_h,
```

where ``A`` is the same discrete Laplacian the steady problem assembles, ``F(t)`` is the
source at time ``t``, and ``M`` is the mass matrix of the discrete inner product
``\langle \cdot, \cdot \rangle_h`` — diagonal, since that inner product is a weighted sum
over grid points.

[`semidiscretize`](@ref) builds exactly this. The spatial form is written the way the steady
problem writes it, so the steady state of the system below solves ``A u_h = F``:

```@example heat
using Bramble

uexact(x, t) = exp(-t) * sinpi(x[1])
source(x, t) = (pi^2 - 1) * exp(-t) * sinpi(x[1])

Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)
Ωₕ = mesh(Ω, 101)
Wₕ = gridspace(Ωₕ)
I = interval(0.0, 1.0)          # the time domain

fₕ = element(Wₕ, 0.0)
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
l = form(Wₕ, v -> innerₕ(fₕ, v))

bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)

sd = semidiscretize(a, l;
    dirichlet = bcs,
    update_coefficients! = t -> Rₕ!(fₕ, x -> source(x, t)))
```

Two pieces are worth naming. The time domain passed to [`dirichlet_constraints`](@ref) is
what lets the boundary values be written as `g(x, t)` rather than `g(x)`; `semidiscretize`
detects that by the same arity test and re-evaluates them at every step.
`update_coefficients!` is called with the current time just before each assembly, and is
where a time-dependent source belongs — `Rₕ!` writes into the `fₕ` the form already holds a
reference to, so nothing is rebuilt and nothing is allocated.

## Dirichlet conditions as algebraic constraints

A constrained row of `A` is ``e_k`` and the matching entry of `F(t)` is ``g(x_i, t)`` —
which is what `assemble` produces for the steady problem too. [`mass_matrix`](@ref) zeroes
those same rows, so each one reads

```math
0 = g(x_i, t) - u_h[i],
```

the boundary condition itself. The system is therefore a differential-algebraic one, and
needs only ``g`` — never ``\partial_t g``, which prescribing ``u_h'`` on the boundary would
have required:

```@example heat
M = mass_matrix(sd)
(M[1, 1], M[51, 51], M[101, 101])   # boundary, interior, boundary
```

## Solving it

[`ode_problem`](@ref) wraps the semidiscretisation, its mass matrix, and its exact Jacobian
``-A`` into a problem `OrdinaryDiffEq` can step. Because the mass matrix is singular, the
method has to be one that admits that — `FBDF` and `QNDF` here, or a Rosenbrock method such
as `Rodas5P`:

```@example heat
using OrdinaryDiffEqBDF

prob = ode_problem(sd, Rₕ(Wₕ, x -> uexact(x, 0.0)), I)
sol = solve(prob, FBDF(); reltol = 1e-10, abstol = 1e-12)

uₕ = element(Wₕ)
parent(uₕ) .= sol.u[end]
normₕ(Rₕ(Wₕ, x -> uexact(x, 1.0)) - uₕ)
```

The initial condition handed in is copied, never mutated, and the copy is made consistent
with the algebraic rows at ``t_0`` before stepping starts — an index-1 system whose initial
condition disagrees with its own constraints is otherwise rejected by the solver or absorbed
into the first step.

!!! note "Rosenbrock methods need `∂f/∂t`"
    `Rodas5P` and friends also want the time derivative of the right-hand side, which they
    build by differentiating through `t`. An `update_coefficients!` hook writing into a
    `Float64` grid function — the one above does — cannot accept a `ForwardDiff.Dual` time,
    so pass `Rodas5P(autodiff = AutoFiniteDiff())` from `ADTypes`, or use a BDF method,
    which needs no `∂f/∂t` at all.

## Checking the answer

Second order in space is the promise. Refining the mesh while holding the time tolerance far
below the spatial error isolates it:

```@example heat
function heat_series(ns)
    hs, errs = Float64[], Float64[]
    for n in ns
        Ωc = mesh(Ω, n)
        Wc = gridspace(Ωc)
        fc = element(Wc, 0.0)
        ac = form(Wc, Wc, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        lc = form(Wc, v -> innerₕ(fc, v))
        bc = dirichlet_constraints(Ωc, I, :boundary => (x, t) -> 0.0)

        sdc = semidiscretize(ac, lc;
            dirichlet = bc,
            update_coefficients! = t -> Rₕ!(fc, x -> source(x, t)))

        solc = solve(ode_problem(sdc, Rₕ(Wc, x -> uexact(x, 0.0)), I),
                     FBDF(); reltol = 1e-11, abstol = 1e-13)

        uc = element(Wc)
        parent(uc) .= solc.u[end]
        push!(hs, hₘₐₓ(Ωc))
        push!(errs, normₕ(Rₕ(Wc, x -> uexact(x, 1.0)) - uc))
    end
    return hs, errs
end

hs, errs = heat_series((11, 21, 41, 81, 161))
orders = [log(errs[i] / errs[i + 1]) / log(hs[i] / hs[i + 1]) for i in 1:(length(errs) - 1)]
```

```@example heat
all(>(1.95), orders)    # second order is the promise
```

## Time-dependent boundary data

Nothing above exercised the `(x, t)` in the boundary condition, since the manufactured
solution vanishes on ``\partial\Omega``. Driving the problem entirely from the boundary
does: start at zero and raise the left end linearly, with the right end held fixed. The two
ends are named on the domain above — a bare `domain(interval(...))` registers only
`:boundary` and `:interior`, so there would be no `:left` to constrain.

```@example heat
gₕ = element(Wₕ, 0.0)
l_drive = form(Wₕ, v -> innerₕ(gₕ, v))
bcs_drive = dirichlet_constraints(Ωₕ, I,
    :left  => (x, t) -> t,
    :right => (x, t) -> 0.0)

sd_drive = semidiscretize(a, l_drive; dirichlet = bcs_drive)
sol_drive = solve(ode_problem(sd_drive, element(Wₕ, 0.0), I), FBDF();
                  reltol = 1e-10, abstol = 1e-12)

u_end = sol_drive.u[end]
(u_end[1], u_end[51], u_end[101])
```

At ``t = 1`` the ends hold exactly the prescribed ``g``: one and zero. The interior is
climbing towards the straight line ``1 - x`` that the source-free steady problem gives, but
has not arrived — ``0.44`` at the midpoint against the steady ``0.5``. It should not have:
the left end was still moving over the whole interval, so diffusion is chasing a boundary
value that never settled. Holding ``g`` fixed and stepping further is what reaches the line,
and the [steady solve](#Steady-problems-and-LinearSolve) below is its limit.

## Steady problems and LinearSolve

The same forms describe the steady problem, and [`linear_problem`](@ref) hands it straight to
`LinearSolve` — with its factorisations, Krylov methods and preconditioners — instead of
assembling by hand first:

```julia
using LinearSolve, IncompleteLU

prob = linear_problem(a, l; dirichlet = :boundary => x -> 0.0)
sol = solve(prob, KrylovJL_GMRES())
```

`linear_problem` takes the `dirichlet`, `dirichlet_components` and `symmetrize` keywords
[`assemble`](@ref) takes, and returns exactly `LinearProblem(A, F)` for the `A` and `F` that
[`assemble`](@ref)`(a, l; ...)` produces.

## See also

  - [`semidiscretize`](@ref), [`ode_problem`](@ref), [`ode_function`](@ref) in the
    [API reference](../api.md).
  - [Linear Poisson](poisson_linear.md) for the steady version of the same spatial operator.
  - [Coupled reaction-diffusion](coupled_reaction_diffusion.md) for systems on a composite
    space, which `semidiscretize` accepts unchanged.
