# A coupled two-component problem on a composite of ghost spaces (#781), with its exact
# solution: test/ghost/composite.jl checks `pde_solve` against a block reference on it, and
# test/convergence/ghost_composite.jl its second order. Loaded into `Main` by both, never by
# the package.
module GhostCoupled

using Bramble

# u₁ = e^{x + y/2}, u₂ = cos 2x sin(y + 1/2); −Δu₁ + u₁ + κu₂ = f₁, −Δu₂ + κu₁ = f₂.
const κ = 0.5
u1(p) = exp(p[1] + p[2] / 2)
u2(p) = cos(2p[1]) * sin(p[2] + 0.5)
dx1(p) = u1(p)
dx2(p) = -2sin(2p[1]) * sin(p[2] + 0.5)
f1(p) = -1.25u1(p) + u1(p) + κ * u2(p)
f2(p) = 5u2(p) + κ * u1(p)

const fa_c = (u, v) -> innerₕ(-1.0 * Δₕ(u(1)), v(1)) + innerₕ(u(1), v(1)) +
                       κ * innerₕ(u(2), v(1)) + innerₕ(-1.0 * Δₕ(u(2)), v(2)) +
                       κ * innerₕ(u(1), v(2))
const fl_c = v -> innerₕ(f1, v(1)) + innerₕ(f2, v(2))

# Component 1: Neumann on xmin, Robin on xmax, `u => g` on ymin and ymax; component 2:
# Robin on xmin, Neumann on xmax, Dirichlet rows on ymin and ymax (the keyword).
function coupled_spaces(Ωₕ)
    C1 = constrained(gridspace(Ωₕ; ghosts = (:xmin, :xmax)),
        u -> (:xmin => ∂ₙ(u) => p -> -dx1(p), :xmax => u + ∂ₙ(u) => p -> u1(p) + dx1(p),
            :ymin => u => u1, :ymax => u => u1))
    C2 = constrained(gridspace(Ωₕ; ghosts = (:xmin, :xmax)),
        u -> (:xmin => 2.0 * u + ∂ₙ(u) => p -> 2u2(p) - dx2(p),
            :xmax => ∂ₙ(u) => dx2))
    return C1 × C2
end

const BC2 = (:ymin => u2, :ymax => u2)

function coupled_solve(Ωₕ)
    V = coupled_spaces(Ωₕ)
    A, F = assemble(form(V, V, fa_c), form(V, fl_c); dirichlet = BC2,
        dirichlet_components = 2)
    return V, pde_solve(A, F)
end

end # module GhostCoupled
