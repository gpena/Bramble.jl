# # Point sources and boundary flux recovery
#
# A well injecting into an aquifer, or a thermal probe in a slab, is a source concentrated at
# a point rather than spread over a volume. [`dirac`](@ref) writes one directly in a linear
# form, and [`reaction`](@ref) reads back the flux the boundary condition had to supply,
# which is how the discrete solution can be checked against conservation rather than against
# a norm alone.
#
# This script runs under `test/examples/pages.jl`, where the lines marked `#src` (the
# assertions the page does not show) execute; it is not part of the published manual.
#
# ## Problem
#
# ```math
# -\nabla \cdot (k \nabla u) = \sum_k Q_k \, \delta(x - x_k) \text{ in } \Omega = (0,1)^2,
# \qquad u = 0 \text{ on } \partial\Omega
# ```
#
# with unit conductivity `k = 1`. Read `u` as hydraulic head and `Q_k` as pumping rates.
#
# ## One well
#
# A single well of strength ``Q`` at ``x_0``. The Dirichlet condition holds the head at zero
# on the whole boundary, so all of the injected water has to leave through it.

using Bramble
using Bramble: reaction, reaction_density
using Random

Q = 3.0
x₀ = (0.35, 0.65)

Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0),
    :left => :xmin, :right => :xmax, :bottom => :ymin, :top => :ymax)
Ωₕ = mesh(Ω, (61, 61), (true, true))   # uniform, so the Green's-function probes below hit nodes
Wₕ = gridspace(Ωₕ)

a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
l = form(Wₕ, v -> innerₕ(dirac(x₀, Q), v))

A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0)

uₕ = element(Wₕ)
uₕ .= A \ F

sum(assemble(l)), maximum(parent(uₕ))

# The assembled load sums to `Q` exactly. A point evaluation is not weighted by a cell
# measure the way a density is: `dirac` puts the strength on the node when the point is one,
# and splits it across the ``2^D`` corners of the containing cell by multilinear weights when
# it is not, which is what preserves the total.
#
# ## Checking it against the Green's function
#
# On the unit square with homogeneous Dirichlet data the exact solution is the Green's
# function, and separating variables gives a series that converges exponentially:
#
# ```math
# G(x, y; x_0, y_0) = \sum_{n \ge 1}
#   \frac{2 \sin(n\pi x)\sin(n \pi x_0) \sinh(n\pi y_<)\sinh(n\pi(1 - y_>))}{n\pi \sinh(n\pi)}
# ```
#
# with ``y_< = \min(y, y_0)`` and ``y_> = \max(y, y_0)``. Away from the singularity, where
# the discrete solution cannot be expected to resolve a logarithm, the two agree:

function green(x, y, x₀, y₀; nterms = 60)
    total = 0.0
    ylo, yhi = minmax(y, y₀)
    for n in 1:nterms
        k = n * π
        total += 2 * sin(k * x) * sin(k * x₀) * sinh(k * ylo) * sinh(k * (1 - yhi)) /
                 (k * sinh(k))
    end
    return total
end

xs, ys = points(Ωₕ)
probes = ((0.15, 0.15), (0.8, 0.2), (0.5, 0.9))

errors = map(probes) do p
    i = argmin(abs.(xs .- p[1]))
    j = argmin(abs.(ys .- p[2]))
    return abs(uₕ[CartesianIndex(i, j)] - Q * green(xs[i], ys[j], x₀...))
end

# Small, but bracketed away from zero as well: an exactly zero difference would mean the   #src
# series and the solve were not independent of each other.                                 #src
@test all(1.0e-12 .< errors .< 1.0e-3)                                                     #src

errors

#-

include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide
surface_plot(uₕ; title = "head from one well at (0.35, 0.65)") # hide

# ## The same well beside a leaky side
#
# One side can instead take a flux or a mixed condition, through ghost points, as on the
# [Neumann and Robin Poisson example](neumann_robin_poisson.md). Here the right side lets
# water seep out at a rate proportional to the head: the Robin relation
# ``u + \partial_n u = 0``. The other three sides keep ``u = 0``.
#
# `gridspace(Ωₕ; ghosts = (:xmax,))` adds one layer outside the right side and `constrained`
# attaches the relation. The point source is written in the load exactly as before, since
# `dirac` works on a constrained space too. The relation reaches the system through
# ``\langle -\Delta_h u, v \rangle_h``, which reads the ghost values, so that form replaces
# `inner₊(∇ₕ(u), ∇ₕ(v))` here.

Cᵍ = constrained(gridspace(Ωₕ; ghosts = (:xmax,)),
    u -> (:xmax => u + ∂ₙ(u) => x -> 0.0,))
aᵍ = form(Cᵍ, Cᵍ, (u, v) -> innerₕ(-1.0 * Δₕ(u), v))
lᵍ = form(Cᵍ, v -> innerₕ(dirac(x₀, Q), v))

Aᵍ, Fᵍ = assemble(aᵍ, lᵍ; dirichlet = (:left => x -> 0.0, :bottom => x -> 0.0,
    :top => x -> 0.0))
uᵍ = element(Wₕ)
uᵍ .= pde_solve(Aᵍ, Fᵍ)

maximum(parent(uᵍ)), maximum(parent(uₕ))

# The head peaks higher than with ``u = 0`` on all four sides, since a leaky side holds the
# head up where a Dirichlet side pins it to zero. The well's strength is untouched by the ghost layer: it is far from `xmax`,
# so the load assembled on the constrained space is the plain one, node for node, and still
# sums to `Q`. The Dirichlet rows of `Fᵍ` hold the boundary data, here zero, so they add
# nothing to the sum.

Fₚ = assemble(l)

@test abs(sum(assemble(lᵍ)) - Q) < 1.0e-10                                                 #src
@test abs(sum(Fᵍ) - Q) < 1.0e-10                                                           #src
@test maximum(abs, Fᵍ .- Fₚ) < 1.0e-12                                                     #src
@test maximum(parent(uᵍ)) > maximum(parent(uₕ))                                            #src

sum(Fᵍ) - Q, maximum(abs, Fᵍ .- Fₚ)

# The ghost values are fixed by the relation, so the check that means something is one-sided
# and independent of them. `Bramble.lift(Cᵍ, u)` returns the solution on the padded grid;
# on the physical nodes of `xmax` the three-point backward difference of that lifted solution
# is the normal derivative, and ``u + \partial_n u`` must vanish up to the truncation error,
# of order ``h^2``. The mesh is uniform, so the spacing is one number:

function leaky_residual(Ωₕ, C, u)
    U = parent(Bramble.lift(C, u))
    G = Bramble.mesh(C)
    nx, ny = Bramble.npoints(Ωₕ, Tuple)
    xs = collect(points(Ωₕ(1)))
    h = xs[nx] - xs[nx - 1]
    lin = LinearIndices(Bramble.npoints(G, Tuple))
    at(i, j) = U[lin[Bramble.padded_index(G, CartesianIndex(i, j))]]
    return maximum(1:ny) do j
        dₙ = (3 * at(nx, j) - 4 * at(nx - 1, j) + at(nx - 2, j)) / (2h)
        abs(at(nx, j) + dₙ)
    end
end

leaky = leaky_residual(Ωₕ, Cᵍ, uᵍ)

# It is small compared with the head itself, which is of order one, but not round-off: the
# one-sided difference carries its own truncation error. A solution that ignored the
# relation would leave a residual of the size of ``\partial_n u`` itself, orders larger.

@test 1.0e-8 < leaky < 1.0e-2 * maximum(parent(uᵍ))                                        #src

leaky

#-

surface_plot(uᵍ; title = "head from one well, leaky right side") # hide

# The flux sections below stay Dirichlet on the plain space, because [`reaction`](@ref)
# refuses a ghost form with an `ArgumentError`: it reads rows of the unconstrained plain
# system, and the fix it names is to apply it on the plain space to the physical values of
# `Bramble.lift(Cᵍ, u)`.

# ## What the boundary had to supply
#
# [`reaction`](@ref) recovers that flux from the *unconstrained* form pair and the solved
# `uₕ`: the constrained rows of `A` were overwritten by `dirichlet_bc!`, so the flux they
# carried is gone from the assembled system, but reassembling without the condition and
# reading the residual `A uₕ - F` gives it back. The sign convention is positive for flux
# leaving the domain.

flux_total = reaction(a, l, uₕ; marker = :boundary)

# Every drop injected leaves through the boundary, to round-off, on any mesh: this is a
# discrete conservation statement, not a discretization error that shrinks under refinement.
# The section on a random mesh below checks it off the uniform grid.

@test abs(flux_total - Q) < 1.0e-10                                                        #src

flux_total, flux_total - Q

# Per side, the split follows the geometry. The well sits on the diagonal ``x + y = 1``,
# above centre and left of it, so the left and top sides take equal, largest shares and the
# right and bottom equal, smallest ones:

sides = (:left, :right, :bottom, :top)
per_side = [reaction(a, l, uₕ; marker = m) for m in sides]

@test sum(per_side) ≈ flux_total                                                           #src
@test all(>(0.0), per_side)                                                                #src
@test per_side[1] ≈ per_side[4] && per_side[2] ≈ per_side[3] && per_side[1] > 2 * per_side[2] #src

collect(zip(sides, per_side))

# [`reaction_density`](@ref) is the pointwise counterpart: a grid function, zero away from the
# marker, carrying the flux *density* at each marked point rather than its sum. Integrating
# it over the boundary with [`inner_Γ`](@ref) recovers the scalar above, which is the check
# that the two agree, and it is the form to hand an exporter:

rd = reaction_density(a, l, uₕ; marker = :boundary)

@test inner_Γ(element(Wₕ, 1.0), rd, :boundary) ≈ flux_total                               #src

inner_Γ(element(Wₕ, 1.0), rd, :boundary), maximum(parent(rd))

# ## The same balance on a random mesh
#
# The uniform grid above makes the Green's-function probes land on nodes. Conservation does not
# need it. On a seeded random mesh the well is no longer on a node, so `dirac` splits its
# strength across the corners of the containing cell, and the boundary still returns all of it:

Random.seed!(20260903)
Ωᵣ = mesh(Ω, (61, 61), (false, false))
Wᵣ = gridspace(Ωᵣ)

aᵣ = form(Wᵣ, Wᵣ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
lᵣ = form(Wᵣ, v -> innerₕ(dirac(x₀, Q), v))

Aᵣ, Fᵣ = assemble(aᵣ, lᵣ; dirichlet = :boundary => x -> 0.0)
uᵣ = element(Wᵣ)
uᵣ .= Aᵣ \ Fᵣ

flux_random = reaction(aᵣ, lᵣ, uᵣ; marker = :boundary)

@test abs(sum(assemble(lᵣ)) - Q) < 1.0e-10                                                 #src
@test abs(flux_random - Q) < 1.0e-10                                                       #src

sum(assemble(lᵣ)) - Q, flux_random - Q

# ## Two wells
#
# An injector and a producer of equal strength. `dirac` takes a vector of points and a vector
# of strengths, which assembles as one term rather than two:

l₂ = form(Wₕ, v -> innerₕ(dirac([(0.3, 0.5), (0.7, 0.5)], [Q, -Q]), v))

A₂, F₂ = assemble(a, l₂; dirichlet = :boundary => x -> 0.0)
u₂ = element(Wₕ)
u₂ .= A₂ \ F₂

net = reaction(a, l₂, u₂; marker = :boundary)

# Nothing accumulates and nothing leaves: what one well injects, the other takes back, and
# the net boundary flux is zero to round-off. The sides do carry flux, in opposite
# directions, since the two wells are not at the same place.

@test abs(net) < 1.0e-10                                                                   #src
@test reaction(a, l₂, u₂; marker = :left) > 0.1                                            #src
@test reaction(a, l₂, u₂; marker = :right) < -0.1                                           #src

net, reaction(a, l₂, u₂; marker = :left), reaction(a, l₂, u₂; marker = :right)

#-

surface_plot(u₂; title = "injector at (0.3, 0.5), producer at (0.7, 0.5)") # hide

# ## Where to go next
#
# - The [forms tutorial](@ref tutorial_form) introduces [`dirac`](@ref) on its own, including
#   off-node placement and a strength that changes inside a time loop.
# - [`reaction`](@ref)'s docstring has the sign convention in full, and
#   [`reaction_density`](@ref) is what to hand [`export_vtk`](@ref).
