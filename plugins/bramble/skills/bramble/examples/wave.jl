# Wave equation u_tt = c²Δu on (0,1)², a standing wave, through the second-order
# semidiscretisation M ü + C u̇ + K u = F. Checks second-order convergence in space and that
# the discrete energy is conserved.
using Bramble
using Bramble: mass_matrix, stiffness_matrix
using OrdinaryDiffEqRosenbrock
using LinearAlgebra: dot

const c = 1.0
const ω = c * sqrt(2) * π
uexact(x, t) = sinpi(x[1]) * sinpi(x[2]) * cos(ω * t)
duexact(x, t) = -ω * sinpi(x[1]) * sinpi(x[2]) * sin(ω * t)

function wave_solve(n)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    Ωₕ = mesh(Ω, (n, n), (true, true))
    Wₕ = gridspace(Ωₕ)
    I = interval(0.0, 1.0)
    fₕ = element(Wₕ, 0.0)                            # no forcing
    K = form(Wₕ, Wₕ, (u, v) -> c^2 * inner₊(∇ₕ(u), ∇ₕ(v)))
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)
    sd = semidiscretize_second_order(K, l; dirichlet = bcs)
    prob = second_order_ode_problem(sd, Rₕ(Wₕ, x -> duexact(x, 0.0)),
        Rₕ(Wₕ, x -> uexact(x, 0.0)), I)              # velocity first, then displacement
    sol = solve(prob, Rodas5P(); reltol = 1.0e-8, abstol = 1.0e-10)
    uT = element(Wₕ, sol.u[end].x[2])
    return Ωₕ, sd, sol, normₕ(uT - Rₕ(Wₕ, x -> uexact(x, 1.0)))
end

Ωₕ₁, _, _, e₁ = wave_solve(13)
Ωₕ₂, sd, sol, e₂ = wave_solve(25)
order = log(e₁ / e₂) / log(hₘₐₓ(Ωₕ₁) / hₘₐₓ(Ωₕ₂))

M, K = mass_matrix(sd), stiffness_matrix(sd)
energy(s) = 0.5 * dot(s.x[1], M * s.x[1]) + 0.5 * dot(s.x[2], K * s.x[2])
E = energy.(sol.u)
drift = maximum(abs, E .- E[1]) / E[1]

@assert 1.8 < order < 2.2
@assert drift < 1.0e-6
