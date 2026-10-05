# Heat equation u_t - u_xx = f(x, t) in 1D: semidiscretize in space, hand the ODE to
# OrdinaryDiffEq. The time-dependent source is a grid function refilled by
# `update_coefficients!`, so the form is built once.
using Bramble
using Bramble: mass_matrix
using OrdinaryDiffEqBDF

uexact(x, t) = exp(-t) * sinpi(x[1])
source(x, t) = (pi^2 - 1) * exp(-t) * sinpi(x[1])

Ω = domain(interval(0.0, 1.0))
Ωₕ = mesh(Ω, 101)
Wₕ = gridspace(Ωₕ)
I = interval(0.0, 1.0)                               # the time interval

fₕ = element(Wₕ, 0.0)
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
l = form(Wₕ, v -> innerₕ(fₕ, v))
bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)
sd = semidiscretize(a, l; dirichlet = bcs,
    update_coefficients! = t -> Rₕ!(fₕ, x -> source(x, t)))

M = mass_matrix(sd)                                  # constrained rows carry a zero
prob = ode_problem(sd, Rₕ(Wₕ, x -> uexact(x, 0.0)), I)
sol = solve(prob, FBDF(); reltol = 1e-10, abstol = 1e-12)

uₕ = element(Wₕ)
parent(uₕ) .= sol.u[end]
err = normₕ(Rₕ(Wₕ, x -> uexact(x, 1.0)) - uₕ)

@assert M[1, 1] == 0.0 && M[end, end] == 0.0
@assert 1.0e-5 < err < 5.0e-5
