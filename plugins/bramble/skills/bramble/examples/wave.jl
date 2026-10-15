# Wave equation u_tt = c²Δu on (0,1)², a standing wave, as the first-order system
# u' = v, v' = c²Δu on a two-leaf space and `M y' = F - A y` through `semidiscretize`.
# Checks second-order convergence in space and that the discrete energy is conserved.
using Bramble
using Bramble: mass_matrix, operator_matrix
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
    Vₕ = Wₕ × Wₕ                                     # the state (u, v)
    I = interval(0.0, 1.0)
    fₕ = element(Wₕ, 0.0)                            # no forcing
    # A₁₂ = -M puts v into the first equation, A₂₁ = c²K puts c²Δu into the second
    a = form(Vₕ, Vₕ, (y, w) -> -innerₕ(y(2), w(1)) + c^2 * inner₊(∇ₕ(y(1)), ∇ₕ(w(2))))
    l = form(Vₕ, w -> innerₕ(fₕ, w(1)) + innerₕ(fₕ, w(2)))
    bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)
    sd = semidiscretize(a, l; dirichlet = bcs, dirichlet_components = (1, 2))
    y₀ = vcat(parent(Rₕ(Wₕ, x -> uexact(x, 0.0))), parent(Rₕ(Wₕ, x -> duexact(x, 0.0))))
    sol = solve(ode_problem(sd, y₀, I), Rodas5P(); reltol = 1.0e-8, abstol = 1.0e-10)
    uT = element(Wₕ, sol.u[end][1:ndofs(Wₕ)])
    return Ωₕ, sd, sol, normₕ(uT - Rₕ(Wₕ, x -> uexact(x, 1.0)))
end

Ωₕ₁, _, _, e₁ = wave_solve(13)
Ωₕ₂, sd, sol, e₂ = wave_solve(25)
order = log(e₁ / e₂) / log(hₘₐₓ(Ωₕ₁) / hₘₐₓ(Ωₕ₂))

n = size(mass_matrix(sd), 1) ÷ 2
M = mass_matrix(sd)[(n + 1):(2n), (n + 1):(2n)]
Kc = operator_matrix(sd)[(n + 1):(2n), 1:n]          # c²K
energy(y) = 0.5 * dot(y[(n + 1):(2n)], M * y[(n + 1):(2n)]) + 0.5 * dot(y[1:n], Kc * y[1:n])
E = energy.(sol.u)
drift = maximum(abs, E .- E[1]) / E[1]

@assert 1.8 < order < 2.2
@assert drift < 1.0e-6
