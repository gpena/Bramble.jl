# Linear elasticity in 2D, -div σ(u) = f with σ = 2μ ε(u) + λ (div u) I, on a random
# non-uniform mesh. The displacement lives in the composite space Wₕ^Val(2); εₕ and divₕ act
# on its trial and test functions directly. The manufactured forcing comes from ForwardDiff.
using Bramble
using ForwardDiff
using Random

lame(E, ν) = (E / (2 * (1 + ν)), E * ν / ((1 + ν) * (1 - 2ν)))
const μ, λ = lame(1.0, 0.3)

elasticity_form(Vₕ) = form(Vₕ, Vₕ,
    (u, v) -> 2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₕ(u), divₕ(v)))

u₁(x) = sin(π * x[1]) * sin(π * x[2])
u₂(x) = sin(2π * x[1]) * sin(π * x[2])               # differs from u₁: a mixed-up component shows
const u_exact = (u₁, u₂)

function f_exact(x)
    p = [x[1], x[2]]
    H = ntuple(c -> ForwardDiff.hessian(u_exact[c], p), 2)
    return ntuple(i -> -μ * (H[i][1, 1] + H[i][2, 2]) - (λ + μ) * sum(H[j][i, j] for j in 1:2), 2)
end

function body_force(Vₕ)
    fₕ = element(Vₕ)
    avgₕ!(fₕ, f_exact)                               # one call fills both components
    return form(Vₕ, q -> innerₕ(fₕ(1), q(1)) + innerₕ(fₕ(2), q(2)))
end

Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Random.seed!(20260903)
Ωₕ = mesh(Ω, (6, 6), (false, false))
hs, errs = Float64[], Float64[]
for level in 1:4
    Vₕ = gridspace(Ωₕ)^Val(2)
    A, F = assemble(elasticity_form(Vₕ), body_force(Vₕ);
        dirichlet = dirichlet_constraints(Ω, :boundary => x -> 0.0))
    uₕ = element(Vₕ)
    uₕ .= A \ F
    exact = element(Vₕ)
    Rₕ!(exact, x -> (u₁(x), u₂(x)))
    push!(hs, hₘₐₓ(Ωₕ))
    push!(errs, sqrt(sum(norm₁ₕ(e)^2 for e in components(uₕ .- exact))))
    level < 4 && iterative_refinement!(Ωₕ)
end
order = log(errs[end - 1] / errs[end]) / log(hs[end - 1] / hs[end])

@assert 1.8 < order < 3.0
