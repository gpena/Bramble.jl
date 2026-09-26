# # Point sources and boundary flux recovery
#
# A well injecting into an aquifer, or a thermal probe in a slab, is a source concentrated at
# a point rather than spread over a volume. [`dirac`](@ref) writes one directly in a linear
# form, and [`reaction`](@ref) reads back the flux the boundary condition had to supply,
# which is how the discrete solution can be checked against conservation rather than against
# a norm alone.
#
# This page is generated from `docs/src/examples/point_sources_flux.jl` by Literate.jl. The
# same file runs under `test/examples/pages.jl`, where the lines marked `#src` -- the
# assertions the page does not show -- execute.
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
using Bramble: reaction, reaction_density, weights

Q = 3.0
x₀ = (0.35, 0.65)

Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0),
    :left => :xmin, :right => :xmax, :bottom => :ymin, :top => :ymax)
Ωₕ = mesh(Ω, (61, 61), (true, true))
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
    return abs(uₕ[i, j] - Q * green(xs[i], ys[j], x₀...))
end

# Small, but bracketed away from zero as well: an exactly zero difference would mean the   #src
# series and the solve were not independent of each other.                                 #src
@test all(1.0e-12 .< errors .< 1.0e-3)                                                     #src

errors

#-

include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide
surface_plot(uₕ; title = "head from one well at (0.35, 0.65)") # hide

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

@test abs(flux_total - Q) < 1.0e-10                                                        #src

flux_total, flux_total - Q

# Per side, the split follows the geometry. The well sits above centre and left of it, so the
# top boundary takes the largest share and the bottom the smallest:

sides = (:left, :right, :bottom, :top)
per_side = [reaction(a, l, uₕ; marker = m) for m in sides]

@test sum(per_side) ≈ flux_total                                                           #src
@test all(>(0.0), per_side)                                                                #src

collect(zip(sides, per_side))

# [`reaction_density`](@ref) is the pointwise counterpart: a grid function, zero away from the
# marker, carrying the flux *density* at each marked point rather than its sum. Weighting it
# by the cell measures recovers the scalar above, which is the check that the two agree, and
# it is the form to hand an exporter:

rd = reaction_density(a, l, uₕ; marker = :boundary)
w = weights(Wₕ, Bramble.Innerh())

@test sum(parent(rd) .* w) ≈ flux_total                                                    #src

sum(parent(rd) .* w), maximum(parent(rd))

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

# ## See also
#
# - The [forms tutorial](../tutorials/form.md) introduces [`dirac`](@ref) on its own, including
#   off-node placement and a strength that changes inside a time loop.
# - [`reaction`](@ref)'s docstring has the sign convention in full, and
#   [`reaction_density`](@ref) is what to hand [`export_vtk`](@ref).
