# # Poisson problem on a disk
#
# A Dirichlet problem on the unit disk, which is not a polygon and has no grid line along its
# boundary, solved with the Ferreira and Grigorieff
# ([2006](https://doi.org/10.1080/01630560600796485)) scheme on a tensor-product grid. The
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
using Bramble: polygon_spaces, lift # public, not exported
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

# The count follows the claim of about ``1.8n`` lines.                                  #src
@test 1.5 * 20 < length(xs) < 2.1 * 20 && 1.5 * 20 < length(ys) < 2.1 * 20            #src

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

# A coarse grid of the same disk, eight seeds per axis, keeps the lines readable. Each line
# that meets the circle has a node there, and those nodes are the vertices of the polygon
# inscribed in the circle:

Ωc = mesh(Ω, (8, 8), (true, true))
xc, yc = collect(points(Ωc(1))), collect(points(Ωc(2)))
length(xc), length(yc)

#-

# ```@raw html
# <figure style="margin:1.5em 0;text-align:center">
# <svg viewBox="0 0 540 540" width="100%" style="max-width:540px;font-family:system-ui,sans-serif" role="img" aria-label="Grid closed on the unit disk from eight uniform seeds per axis, with the inscribed polygon whose vertices are the boundary nodes.">
#   <g stroke="#3b82f6" stroke-opacity="0.6" stroke-width="1">
#     <line class="gx" x1="20.0" y1="20.0" x2="20.0" y2="520.0"/>
#     <line class="gx" x1="73.2" y1="20.0" x2="73.2" y2="520.0"/>
#     <line class="gx" x1="91.4" y1="20.0" x2="91.4" y2="520.0"/>
#     <line class="gx" x1="101.1" y1="20.0" x2="101.1" y2="520.0"/>
#     <line class="gx" x1="162.9" y1="20.0" x2="162.9" y2="520.0"/>
#     <line class="gx" x1="179.9" y1="20.0" x2="179.9" y2="520.0"/>
#     <line class="gx" x1="234.3" y1="20.0" x2="234.3" y2="520.0"/>
#     <line class="gx" x1="305.7" y1="20.0" x2="305.7" y2="520.0"/>
#     <line class="gx" x1="360.1" y1="20.0" x2="360.1" y2="520.0"/>
#     <line class="gx" x1="377.1" y1="20.0" x2="377.1" y2="520.0"/>
#     <line class="gx" x1="438.9" y1="20.0" x2="438.9" y2="520.0"/>
#     <line class="gx" x1="448.6" y1="20.0" x2="448.6" y2="520.0"/>
#     <line class="gx" x1="466.8" y1="20.0" x2="466.8" y2="520.0"/>
#     <line class="gx" x1="520.0" y1="20.0" x2="520.0" y2="520.0"/>
#     <line class="gy" y1="520.0" x1="20.0" y2="520.0" x2="520.0"/>
#     <line class="gy" y1="466.8" x1="20.0" y2="466.8" x2="520.0"/>
#     <line class="gy" y1="448.6" x1="20.0" y2="448.6" x2="520.0"/>
#     <line class="gy" y1="438.9" x1="20.0" y2="438.9" x2="520.0"/>
#     <line class="gy" y1="377.1" x1="20.0" y2="377.1" x2="520.0"/>
#     <line class="gy" y1="360.1" x1="20.0" y2="360.1" x2="520.0"/>
#     <line class="gy" y1="305.7" x1="20.0" y2="305.7" x2="520.0"/>
#     <line class="gy" y1="234.3" x1="20.0" y2="234.3" x2="520.0"/>
#     <line class="gy" y1="179.9" x1="20.0" y2="179.9" x2="520.0"/>
#     <line class="gy" y1="162.9" x1="20.0" y2="162.9" x2="520.0"/>
#     <line class="gy" y1="101.1" x1="20.0" y2="101.1" x2="520.0"/>
#     <line class="gy" y1="91.4" x1="20.0" y2="91.4" x2="520.0"/>
#     <line class="gy" y1="73.2" x1="20.0" y2="73.2" x2="520.0"/>
#     <line class="gy" y1="20.0" x1="20.0" y2="20.0" x2="520.0"/>
#   </g>
#   <circle cx="270.0" cy="270.0" r="200.0" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="5,4"/>
#   <polygon class="poly" points="73.2,305.7 91.4,360.1 101.1,377.1 162.9,438.9 179.9,448.6 234.3,466.8 305.7,466.8 360.1,448.6 377.1,438.9 438.9,377.1 448.6,360.1 466.8,305.7 466.8,234.3 448.6,179.9 438.9,162.9 377.1,101.1 360.1,91.4 305.7,73.2 234.3,73.2 179.9,91.4 162.9,101.1 101.1,162.9 91.4,179.9 73.2,234.3" fill="#ef4444" fill-opacity="0.08" stroke="#ef4444" stroke-width="2"/>
#   <g fill="#ef4444">
#     <circle cx="73.2" cy="305.7" r="3.5"/>
#     <circle cx="91.4" cy="360.1" r="3.5"/>
#     <circle cx="101.1" cy="377.1" r="3.5"/>
#     <circle cx="162.9" cy="438.9" r="3.5"/>
#     <circle cx="179.9" cy="448.6" r="3.5"/>
#     <circle cx="234.3" cy="466.8" r="3.5"/>
#     <circle cx="305.7" cy="466.8" r="3.5"/>
#     <circle cx="360.1" cy="448.6" r="3.5"/>
#     <circle cx="377.1" cy="438.9" r="3.5"/>
#     <circle cx="438.9" cy="377.1" r="3.5"/>
#     <circle cx="448.6" cy="360.1" r="3.5"/>
#     <circle cx="466.8" cy="305.7" r="3.5"/>
#     <circle cx="466.8" cy="234.3" r="3.5"/>
#     <circle cx="448.6" cy="179.9" r="3.5"/>
#     <circle cx="438.9" cy="162.9" r="3.5"/>
#     <circle cx="377.1" cy="101.1" r="3.5"/>
#     <circle cx="360.1" cy="91.4" r="3.5"/>
#     <circle cx="305.7" cy="73.2" r="3.5"/>
#     <circle cx="234.3" cy="73.2" r="3.5"/>
#     <circle cx="179.9" cy="91.4" r="3.5"/>
#     <circle cx="162.9" cy="101.1" r="3.5"/>
#     <circle cx="101.1" cy="162.9" r="3.5"/>
#     <circle cx="91.4" cy="179.9" r="3.5"/>
#     <circle cx="73.2" cy="234.3" r="3.5"/>
#   </g>
# </svg>
# <figcaption style="font-size:0.9em;opacity:0.8">The grid after the closure, the circle (dashed) and the inscribed polygon (red), whose vertices are the boundary nodes.</figcaption>
# </figure>
# ```

# The figure's lines sit at the coordinates of the live coarse mesh, and its polygon's          #src
# vertices are the coarse mesh's boundary nodes in angular order.                             #src
let s = read(@__FILE__, String), px(x) = 20 + 200 * (x + 1.25), py(y) = 20 + 200 * (1.25 - y) #src
    gx = [parse(Float64, m[1]) for m in eachmatch(r"<line class=\"gx\" x1=\"([\d.]+)\"", s)] #src
    gy = [parse(Float64, m[1]) for m in eachmatch(r"<line class=\"gy\" y1=\"([\d.]+)\"", s)] #src
    @test length(gx) == length(xc) && length(gy) == length(yc)                               #src
    @test all(abs.(gx .- px.(xc)) .< 0.06) && all(abs.(gy .- py.(yc)) .< 0.06)               #src
    bc = reshape(BitVector(markers(Ωc)[:D_boundary]), length(xc), length(yc))                #src
    V = sort([(xc[i], yc[j]) for i in eachindex(xc), j in eachindex(yc) if bc[i, j]];        #src
        by = v -> atan(v[2], v[1]))                                                          #src
    pts = match(r"<polygon class=\"poly\" points=\"([^\"]*)\"", s)[1]                     #src
    fig = [parse.(Float64, split(q, ",")) for q in split(pts)]                              #src
    @test length(fig) == length(V)                                                           #src
    @test all(maximum(abs, f .- [px(v[1]), py(v[2])]) < 0.06 for (f, v) in zip(fig, V))      #src
end                                                                                          #src

# ## Solving
#
# `polygon_spaces(Ωₕ, :D, g)` returns the trial space `U` and the test space `W`. Their
# unknowns are the interior nodes, and the Dirichlet data `g` sit at the boundary nodes. The
# load is `avgₕ`, the mean of ``f`` over each dual cell, and `lift(U, u)` puts a solution back
# on the whole grid. The function is called `solve_on`, since `solve` already names the
# SciML function that the steppers and `LinearSolve` extend.

u_exact(p) = sin(2p[1] + 0.5) * cos(1.5p[2] - 0.2) + exp(0.6p[1] - 0.4p[2])
f(p) = 6.25 * sin(2p[1] + 0.5) * cos(1.5p[2] - 0.2) - 0.52 * exp(0.6p[1] - 0.4p[2])

function solve_on(Ωₕ)
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
    return errors(Ωₙ, solve_on(Ωₙ))
end
h = [r.h for r in runs]
emax = [r.max for r in runs]
eh1 = [r.h1 for r in runs]
slope(e) = ([ones(length(h)) log.(h)] \ log.(e))[2]
println("order: max ", round(slope(emax); digits = 2), ", H¹ₕ ", round(slope(eh1); digits = 2))

# Both orders are about two, as for a polygon: the errors fall by about a factor of four
# per halving of `h`, with the max error scattering more from level to level. The error is
# zero on the circle, where the data sit.
@test slope(emax) > 1.8 && slope(eh1) > 1.8                                           #src
@test all(diff(emax) .< 0) && all(diff(eh1) .< 0)                                     #src
@test 1.0e-6 < emax[end] < 5.0e-5 && 1.0e-5 < eh1[end] < 3.0e-4                      #src
# The boundary nodes hold the data by construction; the bracket above is the check that  #src
# discriminates.                                                                         #src
@test all(maximum(abs, r.E[r.boundary]) < 1.0e-12 for r in runs)                      #src

# The errors against `h` on a log-log scale, with a slope-two reference line:

include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot([(h, emax, "max", "#5B5FC7"), (h, eh1, "H¹ₕ", "#0E7C86")];       # hide
    title = "Poisson on the unit disk")                                                # hide

# The solution on the grid requested at 80 points per axis. Nodes outside the disk carry no
# value and are left blank; drag the plot to tilt it.

include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide
Ω₈₀ = mesh(Ω, (80, 80), (true, true))
uₕ = element(gridspace(Ω₈₀))
uₕ.data .= solve_on(Ω₈₀)
uₕ.data[BitVector(markers(Ω₈₀)[:D_exterior])] .= NaN
surface_plot(uₕ; title = "u on the unit disk, n = 80") # hide

# The plotted values are the solution at the interior and boundary nodes only.          #src
@test count(isnan, uₕ.data) == count(markers(Ω₈₀)[:D_exterior])                      #src
let ue = vec([u_exact((x, y)) for x in points(Ω₈₀(1)), y in points(Ω₈₀(2))]),          #src
    keep = .!isnan.(uₕ.data)                                                           #src

    @test maximum(abs, uₕ.data[keep] .- ue[keep]) < 1.0e-3                             #src
end                                                                                    #src

# ## Seeds drawn at random
#
# The seeds need not be uniform. With `(false, false)` the interior points are drawn at
# random, and `mesh` closes the grid on them just the same, so the lines stay at the draws
# and the circle's crossings are added. The error of one draw scatters with it, so the page
# reads no order from a single one:

Random.seed!(1)
Ωr = mesh(Ω, (80, 80), (false, false))
rr = errors(Ωr, solve_on(Ωr))
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
# `ArgumentError` naming the line. The [mesh internals page](../internals/mesh.md) and the
# [`mesh`](@ref) docstring give the class of curves, the errors and the reasons.
