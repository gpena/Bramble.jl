# Poisson with Robin and Neumann relations on a random non-uniform 2D mesh.
#
#   -Δu = 3u_exact in (0,1)²,   α u + ∂ₙu = h on the boundary,   u_exact = exp(x) cos(2y)
#
# α = 2 on xmin and ymin (Robin), α = 0 on xmax and ymax (Neumann). The relations act on a
# space with ghost points: `constrained` turns each `side => relation => data` into the
# closure that eliminates the ghosts. Pass `assemble` both forms, `assemble(a, l)`.
using Bramble
using Random

sol(p) = exp(p[1]) * cos(2p[2])
∇sol(p) = (sol(p), -2 * exp(p[1]) * sin(2p[2]))
rhs(p) = 3 * sol(p)

sides = (:xmin, :xmax, :ymin, :ymax)
normals = (-1.0, 1.0, -1.0, 1.0)
axes_of_normal = (1, 1, 2, 2)
α = (2.0, 0.0, 2.0, 0.0)

function relation(u, j)
    data = p -> α[j] * sol(p) + normals[j] * ∇sol(p)[axes_of_normal[j]]
    rel = iszero(α[j]) ? ∂ₙ(u) : α[j] * u + ∂ₙ(u)
    return sides[j] => rel => data
end

function neumann_robin_error(Ωₕ)
    Cₕ = constrained(gridspace(Ωₕ; ghosts = :boundary), u -> ntuple(j -> relation(u, j), 4))
    A, F = assemble(form(Cₕ, Cₕ, (u, v) -> innerₕ(-1.0 * Δₕ(u), v)),
        form(Cₕ, v -> innerₕ(rhs, v)))
    Wₕ = gridspace(Ωₕ)
    uₕ = element(Wₕ)
    uₕ .= pde_solve(A, F)
    return norm₁ₕ(uₕ .- Rₕ(Wₕ, sol))
end

Random.seed!(20260903)                               # before the mesh: its points are random
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 9), (false, false))
hs, errs = Float64[], Float64[]
for level in 1:5
    push!(hs, hₘₐₓ(Ωₕ))
    push!(errs, neumann_robin_error(Ωₕ))
    level < 5 && iterative_refinement!(Ωₕ)           # halves every cell, keeps the grading
end
orders = [log(errs[i] / errs[i + 1]) / log(hs[i] / hs[i + 1]) for i in 1:(length(errs) - 1)]

@assert minimum(orders[(end - 1):end]) > 1.8
@assert errs[end] < errs[1]
