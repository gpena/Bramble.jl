```@meta
CurrentModule = Bramble
```

# [How to use boundary data that depend on time](@id howto_time_dependent_data)

**Goal.** Solve a heat equation whose Neumann and Robin data change with time, on a constrained ghost space, and check that the solution follows the data.

**You need.** The relations of the [ghost point tutorial](@ref form_ghost), and OrdinaryDiffEqBDF for the time integration. The [heat example](../examples/ghost_heat_equation.md) runs the same idea on a longer problem.

## The model problem

The data of a relation may take the time as a second argument, `(x, t) -> …`. [`constrained`](@ref) tells it from a datum in `x` alone by the methods the function has. The operators in a relation do not change with time, so the matrix of a form on the space is the same at every `t`. Only the data ``d(t)`` that the relations put into the ghosts move.

Take the heat equation ``\partial_t u = \partial_{xx} u`` on a graded interval, with the exact solution ``u = e^{-t} (\cos x + \sin x)`` and the data it gives: a flux at the left end and a Robin condition at the right end.

```@example howto_time
using Bramble
import Bramble: change_points!, Dcₓ
using LinearAlgebra: norm, diag

function graded_interval(n)
    Ωₕ = mesh(domain(interval(0.0, 1.0)), n)
    change_points!(Ωₕ, collect(range(0.0, 1.0, length = n) .^ 2))
    return Ωₕ
end

robin_α = 2.0
Ω_gh = graded_interval(33)
x_gh = collect(points(Ω_gh))

heat_sol(x, t) = exp(-t) * (cos(x[1]) + sin(x[1]))
heat_dx(x, t) = exp(-t) * (cos(x[1]) - sin(x[1]))
g_left_t(x, t) = -heat_dx(x, t)
g_right_t(x, t) = robin_α * heat_sol(x, t) + heat_dx(x, t)

C_t = constrained(gridspace(Ω_gh; ghosts = :boundary), u -> (
    :xmin => ∂ₙ(u) => g_left_t,
    :xmax => robin_α * u + ∂ₙ(u) => g_right_t))
a_t = form(C_t, C_t, (u, v) -> innerₕ(-Δₕ(u), v))
l_t = form(C_t, v -> innerₕ(x -> 0.0, v))

ndofs(C_t)
```

`Bramble.lift(Cₕ, u, t)` is the grid function ``E u + d(t)`` at time `t`. The exact solution at the nodes lifts to a padded function whose centred difference at the first node gives the flux datum at that time, and the data at two times give different ghosts:

```@example howto_time
u_heat(t) = [heat_sol(x, t) for x in x_gh]
U_a = Bramble.lift(C_t, u_heat(0.3), 0.3)
U_b = Bramble.lift(C_t, u_heat(1.0), 1.0)

@assert -Dcₓ(U_a)[2] ≈ g_left_t(x_gh[1], 0.3)
@assert -Dcₓ(U_b)[2] ≈ g_left_t(x_gh[1], 1.0)
@assert abs(U_a[1] - U_b[1]) > 0.1

U_a[1], U_b[1]
```

## Semidiscretize the problem

[`semidiscretize`](@ref) takes the form and the load on this space and returns the `Semidiscretization` of ``M u' = F(t) - A u``, with ``A = E^T A_{pad} E``, the default mass ``M`` diagonal and positive, and ``F(t) = E^T (F_{pad}(t) - A_{pad}\, d(t))``. The residual at a time is the one a space with the data frozen at that time assembles. The block compares them at three times:

```@example howto_time
sd_t = semidiscretize(a_t, l_t)
u_probe = collect(range(-1, 2; length = ndofs(C_t)) .^ 2)

residual_gaps = map((0.0, 0.5, 1.0)) do t
    C_frozen = constrained(gridspace(Ω_gh; ghosts = :boundary), u -> (
        :xmin => ∂ₙ(u) => (x -> g_left_t(x, t)),
        :xmax => robin_α * u + ∂ₙ(u) => (x -> g_right_t(x, t))))
    A_fr, F_fr = assemble(form(C_frozen, C_frozen, (u, v) -> innerₕ(-Δₕ(u), v)),
        form(C_frozen, v -> innerₕ(x -> 0.0, v)))
    du = similar(u_probe)
    sd_t(du, u_probe, nothing, t)
    norm(du - (F_fr - A_fr * u_probe)) / norm(du)
end

@assert all(<(1e-12), residual_gaps)
@assert Bramble.operator_matrix(sd_t) ≈ assemble(a_t)
@assert all(>(0), diag(Bramble.mass_matrix(sd_t)))

maximum(residual_gaps)
```

## Integrate in time

[`ode_problem`](@ref) wraps the semidiscretization for an ODE solver. The initial condition is the exact solution at ``t = 0``, and the solver is `FBDF` from `OrdinaryDiffEqBDF` with tolerances tight enough that the error left at the final time is the one in space. The solution changes by a factor ``e^{-1}`` over the run and the data change with it, so an error far below that change shows the data in time are followed. The error falls by a factor of four when the number of points doubles:

```@example howto_time
using OrdinaryDiffEqBDF

function heat_error(n; T = 1.0)
    Ωₕ = graded_interval(n)
    C = constrained(gridspace(Ωₕ; ghosts = :boundary), u -> (
        :xmin => ∂ₙ(u) => g_left_t, :xmax => robin_α * u + ∂ₙ(u) => g_right_t))
    a = form(C, C, (u, v) -> innerₕ(-Δₕ(u), v))
    l = form(C, v -> innerₕ(x -> 0.0, v))
    xs = collect(points(Ωₕ))
    prob = ode_problem(a, l, [heat_sol(x, 0.0) for x in xs], (0.0, T))
    sol = solve(prob, FBDF(); reltol = 1e-10, abstol = 1e-12)
    return maximum(abs, sol.u[end] - [heat_sol(x, T) for x in xs])
end

err_heat_33, err_heat_65 = heat_error(33), heat_error(65)
@assert err_heat_33 < 1e-3
@assert 3.5 < err_heat_33 / err_heat_65 < 4.5

err_heat_33, err_heat_33 / err_heat_65
```

A constrained space whose data are all static takes `semidiscretize` and `ode_problem` too.

## What is refused

Each refusal below is an `ArgumentError` that names the fix. On a constrained space `semidiscretize` takes no `dirichlet` keyword (the third refusal), so a Dirichlet condition on a facet is the relation `:F => u => g(x, t)`.

```@example howto_time
W_unknown = gridspace(Ω_gh; ghosts = :boundary)
C_unknown = constrained(W_unknown, u -> (:xmin => ∂ₙ(u) => g_left_t,))

timed_refusals = String[]
for thunk in (
    () -> Bramble.lift(C_t, u_heat(0.0)),
    () -> assemble(a_t, l_t),
    () -> semidiscretize(a_t, l_t; dirichlet = :xmin => 0.0),
    () -> semidiscretize(form(C_unknown, C_unknown, (u, v) -> innerₕ(-Δₕ(u), v)),
        form(C_unknown, v -> innerₕ(x -> 0.0, v))),
    () -> semidiscretize(a_t, l_t;
        mass = form(C_t, C_t, (u, v) -> innerₕ(u, v) + 0.01 * innerₕ(Δₕ(u), v))),
)
    try
        thunk()
    catch e
        @assert e isa ArgumentError
        push!(timed_refusals, first(split(sprint(showerror, e), ". ")))
    end
end

@assert length(timed_refusals) == 5
@assert size(assemble(a_t), 1) == ndofs(C_t)

timed_refusals
```
