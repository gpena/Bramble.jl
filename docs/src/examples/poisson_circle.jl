# # Poisson problem on a disk
#
# A Dirichlet problem on the unit disk, discretised on a tensor-product grid whose lines are
# not fitted to it by hand. The disk is a level set: the curve ``\varphi = 0`` with
# ``\varphi(p) = \lVert p \rVert - 1``, and the scheme is the Shortley-Weller scheme, which
# shortens each grid arm the curve cuts to its crossing. The page follows
# [the polygon example](poisson_polygon.md), and only what changes is explained again. Every
# number and every plot below was produced by the code shown.
#
# ## Problem
#
# ```math
# -\Delta u = f \text{ in } D, \qquad u = u_{\text{exact}} \text{ on } \partial D,
# ```
#
# with ``D`` the unit disk. The manufactured solution is
# ``u_{\text{exact}} = \sin 2x \cos 2y + xy``. The term ``xy`` is harmonic, so
# ``f = 8 \sin 2x \cos 2y``.
#
# ## The disk as a level set
#
# `levelset(φ)` wraps a function that changes sign across the curve. Giving it to the domain
# as the marker `:D` labels the nodes of the mesh `:D` (``\varphi < 0``), `:D_boundary`
# (``\varphi = 0``) and `:D_exterior` (``\varphi > 0``). Unlike a polygon, the curve does
# not close the grid: no line is added, and the mesh has the requested number of points. A
# node lands on the curve only by chance, so `:D_boundary` is empty for a generic grid, and
# the boundary enters through the crossings of the curve with the grid lines. The lines are
# random, and the seed makes them the same on every build:

using Bramble
using Bramble: D₋ₓ, D₋ᵧ, D̃ₓ, D̃ᵧ, lift, levelset, levelset_spaces
using Random

Random.seed!(1)
Ω = domain(interval(-1.0, 1.0) × interval(-1.0, 1.0), :D => levelset(p -> hypot(p...) - 1))
Ωₕ = mesh(Ω, (80, 80), (false, false))
xs, ys = collect(points(Ωₕ(1))), collect(points(Ωₕ(2)))
nx, ny = length(xs), length(ys)
nx, ny

#-

# The figure draws a coarse grid of about 12 by 12 points, so its lines stay readable; the
# problem is solved on the fine one. The interior nodes are green and the exterior ones grey,
# and the curve (red) meets each grid line between an interior and an exterior node:

Random.seed!(1)
Ωc = mesh(Ω, (12, 12), (false, false))
xc, yc = collect(points(Ωc(1))), collect(points(Ωc(2)))
length(xc), length(yc)

#-

# ```@raw html
# <figure style="margin:1.5em 0;text-align:center">
# <svg viewBox="0 0 540 540" width="100%" style="max-width:540px;font-family:system-ui,sans-serif" role="img" aria-label="A coarse grid with the unit circle over it; nodes inside the circle are green and nodes outside are grey.">
#   <g stroke="#3b82f6" stroke-opacity="0.6" stroke-width="1">
#     <line class="gx" x1="20.0" y1="20.0" x2="20.0" y2="520.0"/>
#     <line class="gx" x1="32.0" y1="20.0" x2="32.0" y2="520.0"/>
#     <line class="gx" x1="44.6" y1="20.0" x2="44.6" y2="520.0"/>
#     <line class="gx" x1="63.6" y1="20.0" x2="63.6" y2="520.0"/>
#     <line class="gx" x1="79.5" y1="20.0" x2="79.5" y2="520.0"/>
#     <line class="gx" x1="216.6" y1="20.0" x2="216.6" y2="520.0"/>
#     <line class="gx" x1="350.7" y1="20.0" x2="350.7" y2="520.0"/>
#     <line class="gx" x1="365.9" y1="20.0" x2="365.9" y2="520.0"/>
#     <line class="gx" x1="403.8" y1="20.0" x2="403.8" y2="520.0"/>
#     <line class="gx" x1="421.3" y1="20.0" x2="421.3" y2="520.0"/>
#     <line class="gx" x1="447.9" y1="20.0" x2="447.9" y2="520.0"/>
#     <line class="gx" x1="520.0" y1="20.0" x2="520.0" y2="520.0"/>
#     <line class="gy" y1="520.0" x1="20.0" x2="520.0" y2="520.0"/>
#     <line class="gy" y1="462.5" x1="20.0" x2="520.0" y2="462.5"/>
#     <line class="gy" y1="431.7" x1="20.0" x2="520.0" y2="431.7"/>
#     <line class="gy" y1="416.4" x1="20.0" x2="520.0" y2="416.4"/>
#     <line class="gy" y1="392.8" x1="20.0" x2="520.0" y2="392.8"/>
#     <line class="gy" y1="289.5" x1="20.0" x2="520.0" y2="289.5"/>
#     <line class="gy" y1="233.4" x1="20.0" x2="520.0" y2="233.4"/>
#     <line class="gy" y1="126.8" x1="20.0" x2="520.0" y2="126.8"/>
#     <line class="gy" y1="104.8" x1="20.0" x2="520.0" y2="104.8"/>
#     <line class="gy" y1="73.7" x1="20.0" x2="520.0" y2="73.7"/>
#     <line class="gy" y1="47.8" x1="20.0" x2="520.0" y2="47.8"/>
#     <line class="gy" y1="20.0" x1="20.0" x2="520.0" y2="20.0"/>
#   </g>
#   <circle class="curve" cx="270.0" cy="270.0" r="250.0" fill="#ef4444" fill-opacity="0.06" stroke="#ef4444" stroke-width="2.5"/>
#   <g>
#     <circle class="nD" cx="216.6" cy="462.5" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="350.7" cy="462.5" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="365.9" cy="462.5" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="403.8" cy="462.5" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="421.3" cy="462.5" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="79.5" cy="431.7" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="216.6" cy="431.7" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="350.7" cy="431.7" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="365.9" cy="431.7" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="403.8" cy="431.7" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="421.3" cy="431.7" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="447.9" cy="431.7" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="79.5" cy="416.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="216.6" cy="416.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="350.7" cy="416.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="365.9" cy="416.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="403.8" cy="416.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="421.3" cy="416.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="447.9" cy="416.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="63.6" cy="392.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="79.5" cy="392.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="216.6" cy="392.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="350.7" cy="392.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="365.9" cy="392.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="403.8" cy="392.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="421.3" cy="392.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="447.9" cy="392.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="32.0" cy="289.5" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="44.6" cy="289.5" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="63.6" cy="289.5" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="79.5" cy="289.5" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="216.6" cy="289.5" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="350.7" cy="289.5" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="365.9" cy="289.5" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="403.8" cy="289.5" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="421.3" cy="289.5" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="447.9" cy="289.5" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="32.0" cy="233.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="44.6" cy="233.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="63.6" cy="233.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="79.5" cy="233.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="216.6" cy="233.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="350.7" cy="233.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="365.9" cy="233.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="403.8" cy="233.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="421.3" cy="233.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="447.9" cy="233.4" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="79.5" cy="126.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="216.6" cy="126.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="350.7" cy="126.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="365.9" cy="126.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="403.8" cy="126.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="421.3" cy="126.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="447.9" cy="126.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="216.6" cy="104.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="350.7" cy="104.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="365.9" cy="104.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="403.8" cy="104.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="421.3" cy="104.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="447.9" cy="104.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="216.6" cy="73.7" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="350.7" cy="73.7" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="365.9" cy="73.7" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="403.8" cy="73.7" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="421.3" cy="73.7" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="216.6" cy="47.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="350.7" cy="47.8" r="3" fill="#16a34a"/>
#     <circle class="nD" cx="365.9" cy="47.8" r="3" fill="#16a34a"/>
#     <circle class="nD_exterior" cx="20.0" cy="520.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="32.0" cy="520.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="44.6" cy="520.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="63.6" cy="520.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="79.5" cy="520.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="216.6" cy="520.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="350.7" cy="520.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="365.9" cy="520.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="403.8" cy="520.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="421.3" cy="520.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="447.9" cy="520.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="520.0" cy="520.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="20.0" cy="462.5" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="32.0" cy="462.5" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="44.6" cy="462.5" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="63.6" cy="462.5" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="79.5" cy="462.5" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="447.9" cy="462.5" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="520.0" cy="462.5" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="20.0" cy="431.7" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="32.0" cy="431.7" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="44.6" cy="431.7" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="63.6" cy="431.7" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="520.0" cy="431.7" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="20.0" cy="416.4" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="32.0" cy="416.4" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="44.6" cy="416.4" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="63.6" cy="416.4" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="520.0" cy="416.4" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="20.0" cy="392.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="32.0" cy="392.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="44.6" cy="392.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="520.0" cy="392.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="20.0" cy="289.5" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="520.0" cy="289.5" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="20.0" cy="233.4" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="520.0" cy="233.4" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="20.0" cy="126.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="32.0" cy="126.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="44.6" cy="126.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="63.6" cy="126.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="520.0" cy="126.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="20.0" cy="104.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="32.0" cy="104.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="44.6" cy="104.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="63.6" cy="104.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="79.5" cy="104.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="520.0" cy="104.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="20.0" cy="73.7" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="32.0" cy="73.7" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="44.6" cy="73.7" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="63.6" cy="73.7" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="79.5" cy="73.7" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="447.9" cy="73.7" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="520.0" cy="73.7" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="20.0" cy="47.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="32.0" cy="47.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="44.6" cy="47.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="63.6" cy="47.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="79.5" cy="47.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="403.8" cy="47.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="421.3" cy="47.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="447.9" cy="47.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="520.0" cy="47.8" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="20.0" cy="20.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="32.0" cy="20.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="44.6" cy="20.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="63.6" cy="20.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="79.5" cy="20.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="216.6" cy="20.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="350.7" cy="20.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="365.9" cy="20.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="403.8" cy="20.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="421.3" cy="20.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="447.9" cy="20.0" r="3" fill="#9ca3af"/>
#     <circle class="nD_exterior" cx="520.0" cy="20.0" r="3" fill="#9ca3af"/>#   </g>
# </svg>
# <figcaption style="font-size:0.9em;opacity:0.8">A coarse grid with the circle (red) over it. Interior nodes are green, exterior nodes grey.</figcaption>
# </figure>
# ```

# The grid lines and nodes in the figure sit at the coordinates of the live coarse mesh,
# and the curve is the unit circle. The page's run compares them, so the figure cannot go stale.   #src
let s = read(@__FILE__, String), px(x) = 20 + 250 * (x + 1), py(y) = 20 + 250 * (1 - y)   #src
    gx = [parse(Float64, m[1]) for m in eachmatch(r"<line class=\"gx\" x1=\"([\d.]+)\"", s)]  #src
    gy = [parse(Float64, m[1]) for m in eachmatch(r"<line class=\"gy\" y1=\"([\d.]+)\"", s)]  #src
    @test length(gx) == length(xc) && length(gy) == length(yc)                           #src
    @test all(abs.(gx .- px.(xc)) .< 0.06) && all(abs.(gy .- py.(yc)) .< 0.06)           #src
    mc = markers(Ωc)                                                                     #src
    inn = reshape(BitVector(mc[:D]), length(xc), length(yc))                             #src
    dots = [(parse(Float64, m[2]), parse(Float64, m[3]), m[1] == "nD") for                #src
            m in eachmatch(r"<circle class=\"(nD|nD_exterior)\" cx=\"([\d.]+)\" cy=\"([\d.]+)\"", s)]  #src
    @test length(dots) == length(xc) * length(yc)                                        #src
    want = Set((round(px(xc[i]); digits = 1), round(py(yc[j]); digits = 1), inn[i, j])  #src
               for i in eachindex(xc), j in eachindex(yc))                               #src
    @test Set(dots) == want                                                              #src
    @test occursin("cx=\"270.0\" cy=\"270.0\" r=\"250.0\"", s)                             #src
end                                                                                      #src

# ## Spaces
#
# `levelset_spaces(Ωₕ, Ω, :D, g)` returns the trial space `U` and the test space `W`. Their
# unknowns are the interior nodes. The Dirichlet datum `g` is imposed at the crossings, where
# an exterior node read by an interior neighbour takes the value of the quadratic through the
# neighbour, the node behind it and the crossing. `lift(U, u)` puts a solution back on the
# whole grid. The three labels partition the nodes:

sol(p) = sin(2p[1]) * cos(2p[2]) + p[1] * p[2]
load(p) = 8sin(2p[1]) * cos(2p[2])

U, W = levelset_spaces(Ωₕ, Ω, :D, sol)
m = markers(Ωₕ)
n_int, n_bnd, n_ext = count.(BitVector.((m[:D], m[:D_boundary], m[:D_exterior])))

#-

# ## Assembling and solving
#
# The operator is ``-\tilde D_x D_{-x} - \tilde D_y D_{-y}``, the load is `avgₕ`, the mean of
# ``f`` over each dual cell:

S = gridspace(Ωₕ)
fₕ = element(S)
avgₕ!(fₕ, load)

operator(u) = -D̃ₓ(D₋ₓ(u)) - D̃ᵧ(D₋ᵧ(u))
A, F = assemble(form(U, W, (u, v) -> innerₕ(operator(u), v)), form(W, v -> innerₕ(fₕ, v)))
uh = lift(U, A \ F).data
nothing # hide

# ## Error
#
# The max error is over the interior nodes, against the formula restricted to the grid:

inside = reshape(BitVector(m[:D]), nx, ny)
E = reshape(uh, nx, ny) .- [sol((x, y)) for x in xs, y in ys]
err_max = maximum(abs, E[inside])
println("max error ", round(err_max; sigdigits = 3))

# The error of a single random grid scatters with the draw, so the page reads no order from
# it.

# The point counts are the requested ones, the labels partition the nodes, and the error is   #src
# bracketed away from zero.                                                                  #src
@test (nx, ny) == (80, 80)                                                                   #src
@test n_bnd == 0                                                                             #src
@test n_int + n_bnd + n_ext == nx * ny                                                       #src
@test 3.0e-4 < err_max < 2.0e-3                                                              #src
