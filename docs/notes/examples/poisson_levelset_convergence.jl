# Generator for `poisson_levelset_convergence.md`: the observed orders of the scheme on
# level-set domains (#834), drawn as a hand-built SVG log-log plot, no plotting dependency.
# Run `julia --project=. docs/notes/examples/poisson_levelset_convergence.jl` to rewrite the
# page. The setup is copied from `test/convergence/levelset.jl` (the gate), not included from
# it; the levels here go finer and take two more curves. Closure-path curves pool three
# jittered seeds, orbit-path curves take one (the orbit path fits its own lines, the seeds
# only place the lines outside the curve's extent).

using Bramble
import Bramble
using Bramble: lift, polygon_spaces
using Random

uex(p) = sin(2p[1] + 0.5) * cos(1.5p[2] - 0.2) + exp(0.6p[1] - 0.4p[2])
fex(p) = 6.25 * sin(2p[1] + 0.5) * cos(1.5p[2] - 0.2) - 0.52 * exp(0.6p[1] - 0.4p[2])

disk(p) = p[1]^2 + p[2]^2 - 1

# u² + v²(1 + 0.4u)/0.36 = 1 in (u, v) = the point about (0.1, -0.05) turned by θ.
function egg(θ)
    c, s = cos(θ), sin(θ)
    return function (p)
        x, y = p[1] - 0.1, p[2] + 0.05
        u, v = c * x + s * y, -s * x + c * y
        return u^2 + v^2 * (1 + 0.4u) / 0.36 - 1
    end
end

function ellipse(p)
    c, s = cos(π / 6), sin(π / 6)
    return (c * p[1] + s * p[2])^2 + ((-s * p[1] + c * p[2]) / 0.6)^2 - 1
end

# r = 1 + 0.12 cos θ + 0.05 cos 2θ − 0.06 exp(−(Δθ/0.25)²), Δθ the angle to the diagonal
# direction 3π/4 (wrapped), the whole curve turned by 10°.
function dented_egg(p)
    θ = atan(p[2], p[1]) - deg2rad(10)
    Δ = mod(θ - 3π / 4 + π, 2π) - π
    return hypot(p[1], p[2]) - (1 + 0.12 * cos(θ) + 0.05 * cos(2θ) - 0.06 * exp(-(Δ / 0.25)^2))
end

# (name, φ, box of the seeds, levels, seeds).
const CURVES = (
    ("disk", disk, interval(-1.25, 1.25) × interval(-1.25, 1.25), (20, 40, 80, 160, 320), 1:3),
    ("egg 0°", egg(0.0), interval(-1.2, 1.3) × interval(-1.25, 1.15),
        (20, 40, 80, 160, 320), 1:3),
    ("ellipse 30°", ellipse, interval(-1.2, 1.2) × interval(-1.2, 1.2),
        (40, 80, 160, 320), 1:1),
    ("egg 35°", egg(deg2rad(35)), interval(-1.2, 1.3) × interval(-1.25, 1.15),
        (40, 80, 160, 320), 1:1),
    ("dented egg", dented_egg, interval(-1.35, 1.35) × interval(-1.35, 1.35),
        (40, 80, 160, 320), 1:1))

# H¹ₕ error over the edges with both ends in Ω̄, the max error over the interior nodes, and
# the largest gap between the lines that hold a node of Ω̄.
function errors(xs, ys, inn, Γ, uh)
    nx, ny = length(xs), length(ys)
    E = reshape(uh, nx, ny) .- [uex((x, y)) for x in xs, y in ys]
    In = reshape(inn, nx, ny)
    Ω̄ = In .| reshape(Γ, nx, ny)
    half(z, j) = ((j < length(z) ? z[j + 1] : z[j]) - (j > 1 ? z[j - 1] : z[j])) / 2
    s = 0.0
    for l in 1:ny, j in 1:(nx - 1)

        Ω̄[j, l] && Ω̄[j + 1, l] || continue
        s += half(ys, l) * (E[j + 1, l] - E[j, l])^2 / (xs[j + 1] - xs[j])
    end
    for l in 1:(ny - 1), j in 1:nx

        Ω̄[j, l] && Ω̄[j, l + 1] || continue
        s += half(xs, j) * (E[j, l + 1] - E[j, l])^2 / (ys[l + 1] - ys[l])
    end
    jx, jy = findall(vec(any(Ω̄; dims = 2))), findall(vec(any(Ω̄; dims = 1)))
    h = max(maximum(diff(xs[first(jx):last(jx)])), maximum(diff(ys[first(jy):last(jy)])))
    return h, sqrt(s), maximum(abs, E[In])
end

# One solve with g = u + φ/2 on the vertices, (hₘₐₓ, H¹ₕ error, max error).
function solve_level(φ, box, n, seed)
    Random.seed!(1000seed + n)
    Ωₕ = mesh(domain(box, :D => levelset(φ)), (n, n), (false, false))
    xs, ys = collect(points(Ωₕ(1))), collect(points(Ωₕ(2)))
    U, W = polygon_spaces(Ωₕ, :D, p -> uex(p) + φ(p) / 2)
    fₕ = element(gridspace(Ωₕ))
    avgₕ!(fₕ, fex)
    A, F = assemble(form(U, W, (u, v) -> innerₕ(-1.0 * Δₕ(u), v)),
        form(W, v -> innerₕ(fₕ, v)))
    uh = lift(U, A \ F).data
    m = markers(Ωₕ)
    return errors(xs, ys, BitVector(m[:D]), BitVector(m[:D_boundary]), uh)
end

geomean(v) = exp(sum(log, v) / length(v))

"""
    convergence_errors() -> Vector of (name, h, e1, emax)

Per curve of `CURVES`, over its levels, the geometric mean over its seeds of the largest
step `h`, the H¹ₕ error `e1` and the max error `emax`.
"""
function convergence_errors()
    return map(CURVES) do (name, φ, box, levels, seeds)
        R = [[solve_level(φ, box, n, s) for s in seeds] for n in levels]
        @info "solved $name" levels
        (name = name, h = [geomean(first.(x)) for x in R],
            e1 = [geomean(getindex.(x, 2)) for x in R],
            emax = [geomean(getindex.(x, 3)) for x in R])
    end
end

# Least-squares slope of log e against log h.
function slope(h, e)
    x, y = log.(h), log.(e)
    n = length(x)
    return (n * sum(x .* y) - sum(x) * sum(y)) / (n * sum(x .^ 2) - sum(x)^2)
end

# --- the SVG plot ---------------------------------------------------------------------

const PX0, PX1, PY0, PY1 = 70.0, 430.0, 30.0, 300.0   # plot box in SVG units
const COLORS = ("#2563eb", "#ea580c", "#16a34a", "#9333ea", "#dc2626")

function svg_plot(o)
    hs = reduce(vcat, c.h for c in o)
    es = reduce(vcat, [c.e1; c.emax] for c in o)
    lx0, lx1 = floor(log10(minimum(hs)) * 2) / 2, ceil(log10(maximum(hs)) * 2) / 2
    ly0, ly1 = floor(log10(minimum(es))), ceil(log10(maximum(es)))
    X(h) = PX0 + (log10(h) - lx0) / (lx1 - lx0) * (PX1 - PX0)
    Y(e) = PY1 - (log10(e) - ly0) / (ly1 - ly0) * (PY1 - PY0)
    f(x) = string(round(x; digits = 1))
    pts(h, e) = join(("$(f(X(a))),$(f(Y(b)))" for (a, b) in zip(h, e)), " ")
    io = IOBuffer()
    println(io, "<svg viewBox=\"0 0 600 360\" width=\"100%\" ",
        "style=\"max-width:600px;height:auto;font-family:system-ui,-apple-system,",
        "'Segoe UI',sans-serif\" xmlns=\"http://www.w3.org/2000/svg\" role=\"img\" ",
        "aria-label=\"Log-log plot of the H1 and maximum errors against the grid step for ",
        "the disk, the egg at 0 degrees, the ellipse at 30 degrees, the egg at 35 degrees ",
        "and the dented egg, with reference lines of slope 2.\">")
    println(io, "  <rect x=\"$PX0\" y=\"$PY0\" width=\"$(PX1 - PX0)\" height=\"$(PY1 - PY0)\" ",
        "fill=\"none\" stroke=\"currentColor\" stroke-opacity=\"0.5\"/>")
    for d in ceil(Int, lx0):floor(Int, lx1)
        x = f(X(10.0^d))
        println(io, "  <line x1=\"$x\" y1=\"$PY0\" x2=\"$x\" y2=\"$PY1\" stroke=\"currentColor\" ",
            "stroke-opacity=\"0.15\"/>")
        println(io, "  <text x=\"$x\" y=\"$(PY1 + 16)\" font-size=\"11\" fill=\"currentColor\" ",
            "text-anchor=\"middle\">10^$d</text>")
    end
    for d in Int(ly0):Int(ly1)
        y = f(Y(10.0^d))
        println(io, "  <line x1=\"$PX0\" y1=\"$y\" x2=\"$PX1\" y2=\"$y\" stroke=\"currentColor\" ",
            "stroke-opacity=\"0.15\"/>")
        println(io, "  <text x=\"$(PX0 - 6)\" y=\"$(parse(Float64, y) + 4)\" font-size=\"11\" ",
            "fill=\"currentColor\" text-anchor=\"end\">10^$d</text>")
    end
    println(io, "  <text x=\"$((PX0 + PX1) / 2)\" y=\"$(PY1 + 38)\" font-size=\"12\" ",
        "fill=\"currentColor\" text-anchor=\"middle\">grid step h</text>")
    println(io, "  <text x=\"18\" y=\"$((PY0 + PY1) / 2)\" font-size=\"12\" fill=\"currentColor\" ",
        "text-anchor=\"middle\" transform=\"rotate(-90 18 $((PY0 + PY1) / 2))\">error</text>")
    # Reference lines e = c h², drawn below the data.
    ha, hb = minimum(hs), maximum(hs)
    for c in (0.03, 1.0)
        println(io, "  <polyline points=\"$(pts([ha, hb], [c * ha^2, c * hb^2]))\" fill=\"none\" ",
            "stroke=\"currentColor\" stroke-opacity=\"0.6\" stroke-width=\"1.2\" ",
            "stroke-dasharray=\"2,3\"/>")
        println(io, "  <text x=\"$(f(X(hb)))\" y=\"$(f(Y(c * hb^2)))\" font-size=\"11\" ",
            "fill=\"currentColor\" text-anchor=\"end\" dy=\"-4\">slope 2</text>")
    end
    for (c, col) in zip(o, COLORS), (e, dashed) in ((c.e1, false), (c.emax, true))

        println(io, "  <polyline points=\"$(pts(c.h, e))\" fill=\"none\" stroke=\"$col\" ",
            "stroke-width=\"2\"", dashed ? " stroke-dasharray=\"6,3\"" : "", "/>")
        for (a, b) in zip(c.h, e)
            println(io, "  <circle cx=\"$(f(X(a)))\" cy=\"$(f(Y(b)))\" r=\"3\" fill=\"$col\"/>")
        end
    end
    for (i, (c, col)) in enumerate(zip(o, COLORS))
        y = PY0 + 14 + 16 * (i - 1)
        println(io, "  <line x1=\"$(PX1 + 14)\" y1=\"$(y - 4)\" x2=\"$(PX1 + 40)\" y2=\"$(y - 4)\" ",
            "stroke=\"$col\" stroke-width=\"2\"/>")
        println(io, "  <text x=\"$(PX1 + 46)\" y=\"$y\" font-size=\"11\" fill=\"currentColor\">",
            "$(c.name)</text>")
    end
    println(io, "</svg>")
    return String(take!(io))
end

function page(o)
    r(x) = round(x; digits = 2)
    rows = join(("$(c.name) $(r(slope(c.h, c.e1))) and $(r(slope(c.h, c.emax)))" for c in o),
        "; ")
    return """
# Observed orders of the scheme on level-set domains

The scheme of `polygon_spaces` on the mesh of a level-set domain, solved at four or five
refinement levels (20 or 40 to 320 seeds per axis), against a smooth exact solution with
the datum ``g = u + \\varphi/2`` on the vertices, which equals ``u`` on the curve and not at
a vertex off it. The curves are the unit disk and the egg at 0° (the closure path: the lines
grow to close the grid, the seeds are kept), and the ellipse at 30°, the egg at 35° and a
dented egg (the orbit path: the lines are fitted anew). The plot shows the ``H^1_h`` error
(solid) and the maximum error (dashed) against the largest step between the lines that hold
a node of the domain. The dotted lines are ``0.03h^2`` and ``h^2``.

```@raw html
<figure>
$(svg_plot(o))</figure>
```

The fitted slopes, in the order ``H^1_h``, maximum:
$rows.
The closure path is second order. On the orbit path the vertices lie within ``h^2/4`` of the curve, so the error stays
below a multiple of ``h^2`` at every level, but the number of lines follows a cost rule and
``h`` does not fall by a fixed factor from one level to the next: the fitted slope depends
on the curve (the egg at 35° has the largest jumps), and the gate asserts the bound on
``e/h^2`` and a floor on the slope of the ellipse. The page is written by
`poisson_levelset_convergence.jl` next to it; the test gating these orders
is `test/convergence/levelset.jl`.
"""
end

if abspath(PROGRAM_FILE) == @__FILE__
    o = convergence_errors()
    write(joinpath(@__DIR__, "poisson_levelset_convergence.md"), page(o))
    @info "wrote poisson_levelset_convergence.md" slopes=[(c.name, slope(c.h, c.e1),
                                                              slope(c.h, c.emax)) for c in o]
end
