# # Poisson problem on a disk
#
# A Dirichlet problem on the unit disk, which is not a polygon and has no grid line along its
# boundary, solved with the Ferreira and Grigorieff
# ([2006](https://doi.org/10.1007/s00211-006-0006-7)) scheme on a tensor-product grid. The
# disk is given as a level set ``\{\varphi < 0\}``; `mesh` adds grid lines until every line
# that meets the circle has a node on it, and the nodes on the circle are the boundary of the
# discrete domain. Every number below was produced by the code shown.
#
# ## Problem
#
# ```math
# -\Delta u = f \text{ in } D = \{x^2 + y^2 < 1\}, \qquad u = u_{\text{exact}} \text{ on } \partial D,
# ```
#
# with the manufactured solution ``u_{\text{exact}} = \sin(2x + \tfrac12)\cos(\tfrac32 y -
# \tfrac15) + e^{0.6x - 0.4y}``. Its Laplacian worked out by hand gives
# ``f = \tfrac{25}{4} \sin(2x + \tfrac12)\cos(\tfrac32 y - \tfrac15) - 0.52\, e^{0.6x - 0.4y}``,
# so the error is checkable.
#
# ## The disk and its grid
#
# `levelset(φ)` wraps a function of the point `(x, y)` that is negative inside, zero on the
# curve and positive outside. Giving it to the domain as the marker `:D` gives the mesh three
# labels, `:D` (interior), `:D_boundary` and `:D_exterior`, as for a polygon. The box around
# the disk must contain the circle strictly.

using Bramble
using Bramble: lift, polygon_spaces
using Random

φ(p) = hypot(p[1], p[2]) - 1
Ω = domain(interval(-1.25, 1.25) × interval(-1.25, 1.25), :D => levelset(φ))

# `(n, n)` is the number of points requested per axis, and they are seeds. A disk is
# symmetric about both axes, so `mesh` closes the grid from the seeds. The seed lines stay,
# and the crossings of each line with the circle are added as lines of the other direction
# until no line is new. The grid has more lines than requested, about ``1.8n`` for the
# uniform seeds that `(true, true)` asks for:

Ωₕ = mesh(Ω, (20, 20), (true, true))
xs, ys = collect(points(Ωₕ(1))), collect(points(Ωₕ(2)))
length(xs), length(ys)

#-

# The boundary nodes lie on the circle to round-off, which is what the scheme asks of a
# domain (the boundary is made of grid nodes), and every node has one of the three labels:

m = markers(Ωₕ)
on_circle = [abs(φ((xs[i], ys[j]))) for i in eachindex(xs), j in eachindex(ys)]
max_off = maximum(on_circle[reshape(BitVector(m[:D_boundary]), length(xs), length(ys))])
sum(count.(BitVector.((m[:D], m[:D_boundary], m[:D_exterior])))) == length(xs) * length(ys)

#-

max_off < 1.0e-12

# The labels partition the grid, and the boundary nodes lie on the circle.             #src
@test sum(count.(BitVector.((m[:D], m[:D_boundary], m[:D_exterior])))) == length(xs) * length(ys) #src
@test max_off < 1.0e-12                                                              #src
@test count(m[:D_boundary]) > 0 && count(m[:D]) > count(m[:D_boundary])             #src

# ## Solving
#
# `polygon_spaces(Ωₕ, :D, g)` returns the trial space `U` and the test space `W`. Their
# unknowns are the interior nodes, and the Dirichlet data `g` sit at the boundary nodes. The
# load is `avgₕ`, the mean of ``f`` over each dual cell, and `lift(U, u)` puts a solution back
# on the whole grid.

u_exact(p) = sin(2p[1] + 0.5) * cos(1.5p[2] - 0.2) + exp(0.6p[1] - 0.4p[2])
f(p) = 6.25 * sin(2p[1] + 0.5) * cos(1.5p[2] - 0.2) - 0.52 * exp(0.6p[1] - 0.4p[2])

function solve(Ωₕ)
    U, W = polygon_spaces(Ωₕ, :D, u_exact)
    fₕ = element(gridspace(Ωₕ))
    avgₕ!(fₕ, f)
    A, F = assemble(form(U, W, (u, v) -> innerₕ(-Δₕ(u), v)), form(W, v -> innerₕ(fₕ, v)))
    return lift(U, A \ F).data
end
nothing # hide

# ## Error
#
# The max error is over the interior nodes. The ``H^1_h`` error sums, over the grid edges with
# both ends in the closed disk, the dual length times the squared difference of the error
# over the edge length: the ``P_1`` seminorm of the error. `h` is the largest spacing between
# lines inside the disk's extent.

function errors(Ωₕ, uh)
    xs, ys = collect(points(Ωₕ(1))), collect(points(Ωₕ(2)))
    nx, ny = length(xs), length(ys)
    m = markers(Ωₕ)
    inside = reshape(BitVector(m[:D]), nx, ny)
    closed = inside .| reshape(BitVector(m[:D_boundary]), nx, ny)
    E = reshape(uh, nx, ny) .- [u_exact((x, y)) for x in xs, y in ys]
    half(z, j) = ((j < length(z) ? z[j + 1] : z[j]) - (j > 1 ? z[j - 1] : z[j])) / 2
    s = 0.0
    for l in 1:ny, j in 1:(nx - 1)

        closed[j, l] && closed[j + 1, l] || continue
        s += half(ys, l) * (E[j + 1, l] - E[j, l])^2 / (xs[j + 1] - xs[j])
    end
    for l in 1:(ny - 1), j in 1:nx

        closed[j, l] && closed[j, l + 1] || continue
        s += half(xs, j) * (E[j, l + 1] - E[j, l])^2 / (ys[l + 1] - ys[l])
    end
    spacing(z) = maximum(diff(filter(t -> -1 <= t <= 1, z)))
    return (h = max(spacing(xs), spacing(ys)), max = maximum(abs, E[inside]), h1 = sqrt(s),
        E = E, boundary = reshape(BitVector(m[:D_boundary]), nx, ny))
end
nothing # hide

# ## Convergence
#
# Four grids, requested at 20, 40, 80 and 160 points per axis. The order is the slope of the
# least-squares line through ``\log e`` against ``\log h``:

levels = (20, 40, 80, 160)
runs = map(levels) do n
    Ωₙ = mesh(Ω, (n, n), (true, true))
    return errors(Ωₙ, solve(Ωₙ))
end
h = [r.h for r in runs]
emax = [r.max for r in runs]
eh1 = [r.h1 for r in runs]
slope(e) = ([ones(length(h)) log.(h)] \ log.(e))[2]
for (n, r) in zip(levels, runs)
    println("n = ", n, ": h = ", round(r.h; sigdigits = 3), ", max error ",
        round(r.max; sigdigits = 3), ", H¹ₕ error ", round(r.h1; sigdigits = 3))
end
println("order: max ", round(slope(emax); digits = 2), ", H¹ₕ ", round(slope(eh1); digits = 2))

# Both orders are about two, as for a polygon: the errors fall by about a factor of four
# per halving of `h`, with the max error scattering more from level to level. The error is
# zero on the circle, where the data sit.
@test slope(emax) > 1.8 && slope(eh1) > 1.8                                           #src
@test all(diff(emax) .< 0) && all(diff(eh1) .< 0)                                     #src
@test 1.0e-6 < emax[end] < 5.0e-5 && 1.0e-5 < eh1[end] < 3.0e-4                      #src
@test all(maximum(abs, r.E[r.boundary]) < 1.0e-12 for r in runs)                      #src

# ## Seeds drawn at random
#
# The seeds need not be uniform. With `(false, false)` the interior points are drawn at
# random, and `mesh` closes the grid on them just the same, so the lines stay at the draws
# and the circle's crossings are added. The error of one draw scatters with it, so the page
# reads no order from a single one:

Random.seed!(1)
Ωr = mesh(Ω, (80, 80), (false, false))
rr = errors(Ωr, solve(Ωr))
println("random seeds, n = 80: max error ", round(rr.max; sigdigits = 3), ", H¹ₕ error ",
    round(rr.h1; sigdigits = 3))

# The random draw stays within a few times the uniform one at the same request.             #src
@test rr.max < 5 * emax[3] && rr.h1 < 5 * eh1[3]                                        #src
@test maximum(abs, rr.E[rr.boundary]) < 1.0e-12                                         #src

# ## Where the method stops
#
# A disk is the simplest curve the grid fits. Curves with no symmetry about a grid line need
# lines that are not drawn from the seeds, and some curves cannot be fitted at all: a curve
# that a horizontal or vertical line meets in more than two points (a star) throws an
# `ArgumentError` naming the line. The design record in `docs/notes/internals/levelset.md`
# gives the class of curves, the errors and the reasons.
