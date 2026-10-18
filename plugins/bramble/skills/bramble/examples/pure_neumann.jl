# Pure Neumann Poisson on a random non-uniform 2D mesh, fixed by a mean-zero constraint.
#
#   -Δu = 3u_exact in (0,1)²,   ∂ₙu = ∂ₙu_exact on the boundary,   u_exact = exp(x) cos(2y)
#
# The solution is defined up to a constant and the matrix annihilates constants. `pde_solve`
# takes `mean_zero = w`, the cell measures, and returns the solution with wᵀu = 0. The exact
# solution is shifted to the same discrete mean before the comparison.
using Bramble
using LinearAlgebra: dot
using Random

sol(p) = exp(p[1]) * cos(2p[2])
∇sol(p) = (sol(p), -2 * exp(p[1]) * sin(2p[2]))
rhs(p) = 3 * sol(p)

sides = (:xmin, :xmax, :ymin, :ymax)
normals = (-1.0, 1.0, -1.0, 1.0)
axes_of_normal = (1, 1, 2, 2)

function relation(u, j)
    return sides[j] => ∂ₙ(u) => (p -> normals[j] * ∇sol(p)[axes_of_normal[j]])
end

function pure_neumann_error(Ωₕ)
    Cₕ = constrained(gridspace(Ωₕ; ghosts = :boundary), u -> ntuple(j -> relation(u, j), 4))
    A, F = assemble(form(Cₕ, Cₕ, (u, v) -> innerₕ(-1.0 * Δₕ(u), v)),
        form(Cₕ, v -> innerₕ(rhs, v)))
    Wₕ = gridspace(Ωₕ)
    w = assemble(form(Wₕ, v -> innerₕ(x -> 1.0, v)))
    uₕ = element(Wₕ)
    uₕ .= pde_solve(A, F; mean_zero = w)
    uₑ = Rₕ(Wₕ, sol)
    uₑ .-= dot(w, uₑ) / sum(w)
    return norm₁ₕ(uₕ .- uₑ), abs(dot(w, uₕ))
end

Random.seed!(20260903)                               # before the mesh: its points are random
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 9), (false, false))
hs, errs, means = Float64[], Float64[], Float64[]
for level in 1:5
    err, mean = pure_neumann_error(Ωₕ)
    push!(hs, hₘₐₓ(Ωₕ))
    push!(errs, err)
    push!(means, mean)
    level < 5 && iterative_refinement!(Ωₕ)           # halves every cell, keeps the grading
end
orders = [log(errs[i] / errs[i + 1]) / log(hs[i] / hs[i + 1]) for i in 1:(length(errs) - 1)]

@assert minimum(orders[(end - 1):end]) > 1.8
@assert maximum(means) < 1.0e-10
