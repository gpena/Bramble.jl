# Linear Poisson problem on a random non-uniform 2D mesh, with a convergence check.
#
#   -Δu = g in (0,1)²,   u = u_exact on the boundary,   u_exact(x, y) = exp(x + y)
#
# The scheme is second order in the discrete H¹ norm `norm₁ₕ`, also on non-uniform meshes.
using Bramble
using Random

sol(x) = exp(x[1] + x[2])
rhs(x) = -2 * sol(x)

Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))

function poisson_error(Ωₕ)
    Wₕ = gridspace(Ωₕ)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    gₕ = element(Wₕ)
    avgₕ!(gₕ, rhs)                                   # cell averages of the source
    l = form(Wₕ, v -> innerₕ(gₕ, v))
    A, F = assemble(a, l; dirichlet = dirichlet_constraints(Ω, :boundary => sol))
    uₕ = element(Wₕ)
    uₕ .= A \ F
    return norm₁ₕ(uₕ .- Rₕ(Wₕ, sol))
end

Random.seed!(20260903)                               # before the mesh: its points are random
Ωₕ = mesh(Ω, (5, 5), (false, false))
hs, errs = Float64[], Float64[]
for level in 1:5
    push!(hs, hₘₐₓ(Ωₕ))
    push!(errs, poisson_error(Ωₕ))
    level < 5 && iterative_refinement!(Ωₕ)           # halves every cell, keeps the grading
end
order = log(errs[end - 1] / errs[end]) / log(hs[end - 1] / hs[end])

@assert 1.8 < order < 3.0
@assert errs[end] < errs[1]
