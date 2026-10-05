# Poisson problem with a point source of strength Q, -Δu = Q δ(x - x₀) with u = 0 on the
# boundary. `dirac` builds the source; `reaction` recovers the flux the boundary condition had
# to supply, which must balance the source exactly.
using Bramble
using Bramble: reaction

const Q = 3.0
const x₀ = (0.35, 0.65)

Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0),
    :left => :xmin, :right => :xmax, :bottom => :ymin, :top => :ymax)
Ωₕ = mesh(Ω, (61, 61), (true, true))
Wₕ = gridspace(Ωₕ)

a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
l = form(Wₕ, v -> innerₕ(dirac(x₀, Q), v))
A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0)
uₕ = element(Wₕ)
uₕ .= A \ F

flux = reaction(a, l, uₕ; marker = :boundary)                 # total boundary flux
per_side = [reaction(a, l, uₕ; marker = m) for m in (:left, :right, :bottom, :top)]

# A source and a sink of equal strength: no net flux leaves the domain.
l₂ = form(Wₕ, v -> innerₕ(dirac([(0.3, 0.5), (0.7, 0.5)], [Q, -Q]), v))
A₂, F₂ = assemble(a, l₂; dirichlet = :boundary => x -> 0.0)
u₂ = element(Wₕ)
u₂ .= A₂ \ F₂

@assert abs(sum(assemble(l)) - Q) < 1.0e-12
@assert abs(flux - Q) < 1.0e-10
@assert sum(per_side) ≈ flux && all(>(0.0), per_side)
@assert abs(reaction(a, l₂, u₂; marker = :boundary)) < 1.0e-10
