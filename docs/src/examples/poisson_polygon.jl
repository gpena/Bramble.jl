# # Poisson-type problem on a polygon
#
# A Dirichlet problem on the polygon of Figure 1 in Ferreira and Grigorieff
# ([2006](https://doi.org/10.1080/01630560600796485)): a non-convex nine-sided shape with
# oblique edges, discretised on a tensor-product grid whose lines are not fitted to it by hand.
# The coefficients vary, and the operator has a mixed term. Every number and every plot below
# was produced by the code shown.
#
# ## Problem
#
# ```math
# -\partial_x(a\, \partial_x u + b\, \partial_y u) - \partial_y(b\, \partial_x u + c\, \partial_y u)
# = f \text{ in } P, \qquad u = u_{\text{exact}} \text{ on } \partial P,
# ```
#
# with ``P`` the polygon, ``a = 2 + \tfrac12 \sin(x+y)``, ``c = 2 + \tfrac3{10} \cos x`` and the
# mixed coefficient ``b = \tfrac12 + \tfrac15 \sin(xy)``. The manufactured solution is
# ``u_{\text{exact}} = \sin x \cos(0.7\, y) + 0.1\, xy``, and ``f`` is the left-hand side
# worked out by hand from it, so the error is checkable.
#
# ## The polygon and its grid
#
# `polygon(V)` takes the vertices in order, either orientation. Giving the polygon to the
# domain as the marker `:P` makes `mesh` close the grid lines: every line that meets an edge
# of the polygon gets a node on it, so the boundary of ``P`` is made of grid nodes, and the
# nodes fall into three labels, `:P` (interior), `:P_boundary` and `:P_exterior`. The closure
# may add lines, so the grid has more points than the requested count.

using Bramble
using Bramble: polygon_spaces, lift # public, not exported
using Bramble: D₋ₓ, D₋ᵧ, D̃ₓ, D̃ᵧ, Dcₓ, Dcᵧ
using Random

V = [(0.0, 2.0), (2.0, 0.0), (4.0, 0.0), (6.0, 2.0), (6.0, 3.0), (4.0, 3.0), (4.0, 5.0),
    (2.0, 5.0), (0.0, 3.0)]

# The figure draws a coarse grid of the same polygon, so its lines stay readable; the
# problem below is solved on a much finer one. Both are random, and the seed makes them the
# same on every build:

Random.seed!(1)
Ω = domain(interval(0.0, 6.0) × interval(0.0, 5.0), :P => polygon(V))
Ωc = mesh(Ω, (8, 7), (false, false))
xc, yc = collect(points(Ωc(1))), collect(points(Ωc(2)))
length(xc), length(yc)

#-

# `(false, false)` draws the interior lines at random, as on the other pages. The coarse
# grid after the closure, with the polygon over it:

# ```@raw html
# <figure style="margin:1.5em 0;text-align:center">
# <svg viewBox="0 0 580 490" width="100%" style="max-width:580px;font-family:system-ui,sans-serif" role="img" aria-label="Grid closed on the nine-sided polygon of Figure 1, with the polygon drawn over it.">
#   <g stroke="#3b82f6" stroke-opacity="0.6" stroke-width="1">
#     <line class="gx" x1="20.0" y1="20.0" x2="20.0" y2="470.0"/>
#     <line class="gx" x1="51.6" y1="20.0" x2="51.6" y2="470.0"/>
#     <line class="gx" x1="59.6" y1="20.0" x2="59.6" y2="470.0"/>
#     <line class="gx" x1="65.9" y1="20.0" x2="65.9" y2="470.0"/>
#     <line class="gx" x1="96.6" y1="20.0" x2="96.6" y2="470.0"/>
#     <line class="gx" x1="101.2" y1="20.0" x2="101.2" y2="470.0"/>
#     <line class="gx" x1="124.1" y1="20.0" x2="124.1" y2="470.0"/>
#     <line class="gx" x1="124.5" y1="20.0" x2="124.5" y2="470.0"/>
#     <line class="gx" x1="182.6" y1="20.0" x2="182.6" y2="470.0"/>
#     <line class="gx" x1="200.0" y1="20.0" x2="200.0" y2="470.0"/>
#     <line class="gx" x1="208.6" y1="20.0" x2="208.6" y2="470.0"/>
#     <line class="gx" x1="359.3" y1="20.0" x2="359.3" y2="470.0"/>
#     <line class="gx" x1="380.0" y1="20.0" x2="380.0" y2="470.0"/>
#     <line class="gx" x1="397.4" y1="20.0" x2="397.4" y2="470.0"/>
#     <line class="gx" x1="455.5" y1="20.0" x2="455.5" y2="470.0"/>
#     <line class="gx" x1="455.9" y1="20.0" x2="455.9" y2="470.0"/>
#     <line class="gx" x1="478.8" y1="20.0" x2="478.8" y2="470.0"/>
#     <line class="gx" x1="483.4" y1="20.0" x2="483.4" y2="470.0"/>
#     <line class="gx" x1="514.1" y1="20.0" x2="514.1" y2="470.0"/>
#     <line class="gx" x1="520.4" y1="20.0" x2="520.4" y2="470.0"/>
#     <line class="gx" x1="528.4" y1="20.0" x2="528.4" y2="470.0"/>
#     <line class="gx" x1="560.0" y1="20.0" x2="560.0" y2="470.0"/>
#     <line class="gy" y1="470.0" x1="20.0" x2="560.0" y2="470.0"/>
#     <line class="gy" y1="452.6" x1="20.0" x2="560.0" y2="452.6"/>
#     <line class="gy" y1="394.5" x1="20.0" x2="560.0" y2="394.5"/>
#     <line class="gy" y1="394.1" x1="20.0" x2="560.0" y2="394.1"/>
#     <line class="gy" y1="371.2" x1="20.0" x2="560.0" y2="371.2"/>
#     <line class="gy" y1="366.6" x1="20.0" x2="560.0" y2="366.6"/>
#     <line class="gy" y1="335.9" x1="20.0" x2="560.0" y2="335.9"/>
#     <line class="gy" y1="329.6" x1="20.0" x2="560.0" y2="329.6"/>
#     <line class="gy" y1="321.6" x1="20.0" x2="560.0" y2="321.6"/>
#     <line class="gy" y1="290.0" x1="20.0" x2="560.0" y2="290.0"/>
#     <line class="gy" y1="213.0" x1="20.0" x2="560.0" y2="213.0"/>
#     <line class="gy" y1="200.0" x1="20.0" x2="560.0" y2="200.0"/>
#     <line class="gy" y1="168.4" x1="20.0" x2="560.0" y2="168.4"/>
#     <line class="gy" y1="160.4" x1="20.0" x2="560.0" y2="160.4"/>
#     <line class="gy" y1="154.1" x1="20.0" x2="560.0" y2="154.1"/>
#     <line class="gy" y1="123.4" x1="20.0" x2="560.0" y2="123.4"/>
#     <line class="gy" y1="118.8" x1="20.0" x2="560.0" y2="118.8"/>
#     <line class="gy" y1="95.9" x1="20.0" x2="560.0" y2="95.9"/>
#     <line class="gy" y1="95.5" x1="20.0" x2="560.0" y2="95.5"/>
#     <line class="gy" y1="37.4" x1="20.0" x2="560.0" y2="37.4"/>
#     <line class="gy" y1="20.0" x1="20.0" x2="560.0" y2="20.0"/>
#   </g>
#   <polygon class="poly" points="20.0,290.0 200.0,470.0 380.0,470.0 560.0,290.0 560.0,200.0 380.0,200.0 380.0,20.0 200.0,20.0 20.0,200.0" fill="#ef4444" fill-opacity="0.08" stroke="#ef4444" stroke-width="2.5"/>
# </svg>
# <figcaption style="font-size:0.9em;opacity:0.8">The grid after the closure, with the polygon (red) over it.</figcaption>
# </figure>
# ```

# The grid lines in the figure sit at the coordinates of the live coarse mesh, and its     #src
# polygon has the vertices of `V`. The page's run compares them, so the figure cannot go stale.   #src
let s = read(@__FILE__, String), px(x) = 20 + 90x, py(y) = 20 + 90 * (5 - y)             #src
    gx = [parse(Float64, m[1]) for m in eachmatch(r"<line class=\"gx\" x1=\"([\d.]+)\"", s)]  #src
    gy = [parse(Float64, m[1]) for m in eachmatch(r"<line class=\"gy\" y1=\"([\d.]+)\"", s)]  #src
    @test length(gx) == length(xc) && length(gy) == length(yc)                           #src
    @test all(abs.(gx .- px.(xc)) .< 0.06) && all(abs.(gy .- py.(yc)) .< 0.06)           #src
    pts = match(r"<polygon class=\"poly\" points=\"([^\"]*)\"", s)[1]                    #src
    @test [parse.(Float64, split(p, ",")) for p in split(pts)] ≈                         #src
          [[px(v[1]), py(v[2])] for v in V] atol = 0.06                                  #src
end                                                                                      #src

# ## The fine grid, labels and spaces
#
# The problem is solved on a grid requested at 72 by 60 points, which the closure enlarges.
# The errors of a random grid scatter with the draw: at this size the max error stays
# below 1e-2 and the ``H^1_h`` error below 1e-1.
#
# `polygon_spaces(Ωₕ, :P, g)` returns the trial space `U` and the test space `W`. Their
# unknowns are the interior nodes. The nodes on the polygon carry the Dirichlet data `g`
# exactly, and `lift(U, u)` puts a solution back on the whole grid.

Random.seed!(1)
Ωₕ = mesh(Ω, (72, 60), (false, false))
xs, ys = collect(points(Ωₕ(1))), collect(points(Ωₕ(2)))

sol(p) = sin(p[1]) * cos(0.7p[2]) + 0.1 * p[1] * p[2]
U, W = polygon_spaces(Ωₕ, :P, sol)

# The labels the closure assigned count the nodes of each kind:

m = markers(Ωₕ)
n_int, n_bnd, n_ext = count.(BitVector.((m[:P], m[:P_boundary], m[:P_exterior])))

#-

n_int + n_bnd + n_ext == length(xs) * length(ys)

# ## Coefficients, load and operator
#
# The diffusion coefficients ``a`` and ``c`` are sampled at the staggered half points, ``a``
# between neighbours in ``x`` and ``c`` between neighbours in ``y``, which is where the
# conservative form ``-\tilde D_x(a\, D_{-x} u)`` reads them. ``b`` is sampled at the grid
# points, and the mixed term is `Dcₓ(bₕ * Dcᵧ(u)) + Dcᵧ(bₕ * Dcₓ(u))` with the central
# differences `Dcₓ` and `Dcᵧ`. The load is `avgₕ`, the mean of ``f`` over each dual cell.

a(p) = 2 + 0.5sin(p[1] + p[2])
b(p) = 0.5 + 0.2sin(p[1] * p[2])
c(p) = 2 + 0.3cos(p[1])

function load(p)
    x, y = p
    ux, uy = cos(x) * cos(0.7y) + 0.1y, -0.7sin(x) * sin(0.7y) + 0.1x
    uxx, uyy = -sin(x) * cos(0.7y), -0.49sin(x) * cos(0.7y)
    uxy = -0.7cos(x) * sin(0.7y) + 0.1
    ax, bx, by = 0.5cos(x + y), 0.2y * cos(x * y), 0.2x * cos(x * y)
    return -(ax * ux + a(p) * uxx) - c(p) * uyy - (bx * uy + by * ux + 2b(p) * uxy)
end

S = gridspace(Ωₕ)
aₕ, cₕ, fₕ = element(S), element(S), element(S)
nx, ny = length(xs), length(ys)
A3, C3 = reshape(aₕ.data, nx, ny), reshape(cₕ.data, nx, ny)
for l in 1:ny, j in 1:nx

    A3[j, l] = j > 1 ? a(((xs[j - 1] + xs[j]) / 2, ys[l])) : 0.0
    C3[j, l] = l > 1 ? c((xs[j], (ys[l - 1] + ys[l]) / 2)) : 0.0
end
bₕ = Rₕ(S, b)
avgₕ!(fₕ, load)

diffusion(u) = -D̃ₓ(aₕ * D₋ₓ(u)) - D̃ᵧ(cₕ * D₋ᵧ(u))
operator(u) = diffusion(u) - Dcₓ(bₕ * Dcᵧ(u)) - Dcᵧ(bₕ * Dcₓ(u))
nothing # hide

# ## Assembling and solving
#
# The forms take `U` as trial and `W` as test space, and the Dirichlet data are already in
# `U`, so no `dirichlet` keyword goes to `assemble`:

A, F = assemble(form(U, W, (u, v) -> innerₕ(operator(u), v)), form(W, v -> innerₕ(fₕ, v)))
uh = lift(U, A \ F).data
nothing # hide

# ## Error
#
# The max error is over the interior nodes. The ``H^1_h`` error sums, over the grid edges with
# both ends in the closed polygon, the dual length times the squared difference of the error
# over the edge length: the ``P_1`` seminorm of the error.

inside = reshape(BitVector(m[:P]), nx, ny)
closed = inside .| reshape(BitVector(m[:P_boundary]), nx, ny)
E = reshape(uh, nx, ny) .- [sol((x, y)) for x in xs, y in ys]

function h1_error(E, xs, ys, closed)
    half(z, j) = ((j < length(z) ? z[j + 1] : z[j]) - (j > 1 ? z[j - 1] : z[j])) / 2
    s = 0.0
    for l in axes(E, 2), j in 1:(size(E, 1) - 1)

        closed[j, l] && closed[j + 1, l] || continue
        s += half(ys, l) * (E[j + 1, l] - E[j, l])^2 / (xs[j + 1] - xs[j])
    end
    for l in 1:(size(E, 2) - 1), j in axes(E, 1)

        closed[j, l] && closed[j, l + 1] || continue
        s += half(xs, j) * (E[j, l + 1] - E[j, l])^2 / (ys[l + 1] - ys[l])
    end
    return sqrt(s)
end

err_max = maximum(abs, E[inside])
err_h1 = h1_error(E, xs, ys, closed)
println("max error ", round(err_max; sigdigits = 3), ", H¹ₕ error ", round(err_h1; sigdigits = 3))

# Both are small on this one grid. The errors of a single random grid scatter by a factor
# of two or three between draws, so the page reads no order from them: the orders, measured
# over six levels, are in the convergence test of the suite. The error vanishes on the
# polygon, where the Dirichlet data sit.

# The errors are bracketed away from zero. The polygon nodes hold the data by construction,  #src
# so the last check does not discriminate and follows the bracket.                           #src
@test 1.0e-6 < err_max < 1.0e-2                                                            #src
@test 1.0e-6 < err_h1 < 1.0e-1                                                             #src
@test maximum(abs, E[BitVector(vec(m[:P_boundary]))]) == 0                                 #src
