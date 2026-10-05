# Steady convection-diffusion, -ϵΔu + b·∇u = g on a random non-uniform 2D mesh. The
# convection term pairs the backward average Mₕ(u) with ∇ₕ(v), which keeps the scheme second
# order in norm₁ₕ.
using Bramble
using Random

const ϵ, b = 1.0, 0.1
sol(x) = exp(x[1] + x[2])
rhs(x) = -2 * sol(x) * (b + ϵ)

Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))

function convdiff_error(Ωₕ)
    Wₕ = gridspace(Ωₕ)
    a = form(Wₕ, Wₕ, (u, v) -> ϵ * inner₊(∇ₕ(u), ∇ₕ(v)) + b * inner₊(Mₕ(u), ∇ₕ(v)))
    gₕ = element(Wₕ)
    avgₕ!(gₕ, rhs)
    l = form(Wₕ, v -> innerₕ(gₕ, v))
    A, F = assemble(a, l; dirichlet = dirichlet_constraints(Ω, :boundary => sol))
    uₕ = element(Wₕ)
    uₕ .= A \ F
    return norm₁ₕ(uₕ .- Rₕ(Wₕ, sol))
end

Random.seed!(20260903)
Ωₕ = mesh(Ω, (5, 5), (false, false))
hs, errs = Float64[], Float64[]
for level in 1:5
    push!(hs, hₘₐₓ(Ωₕ))
    push!(errs, convdiff_error(Ωₕ))
    level < 5 && iterative_refinement!(Ωₕ)
end
order = log(errs[end - 1] / errs[end]) / log(hs[end - 1] / hs[end])

@assert 1.8 < order < 3.0
