# Generator for `poisson_polygon_convergence.md`: the observed orders of the polygon scheme
# (#204) on Figure 1's domain, drawn as a hand-built SVG log-log plot, no plotting
# dependency. Run `julia --project=. docs/notes/examples/poisson_polygon_convergence.jl` to
# rewrite the page. The setup is copied from `test/convergence/poisson_polygon.jl` (the gate),
# not included from it. One jittered seed is enough here; the gate pools several.

using Bramble
using Random
import Bramble
using Bramble: D̃ₓ, D̃ᵧ, D₋ₓ, D₋ᵧ, Dcₓ, Dcᵧ

const FIG1 = [(0.0, 2.0), (2.0, 0.0), (4.0, 0.0), (6.0, 2.0), (6.0, 3.0), (4.0, 3.0),
    (4.0, 5.0), (2.0, 5.0), (0.0, 3.0)]
const LX, LY = 6.0, 5.0

uex(p) = sin(p[1]) * cos(0.7p[2]) + 0.1 * p[1] * p[2]
ux(p) = cos(p[1]) * cos(0.7p[2]) + 0.1p[2]
uy(p) = -0.7sin(p[1]) * sin(0.7p[2]) + 0.1p[1]
uxx(p) = -sin(p[1]) * cos(0.7p[2])
uyy(p) = -0.49sin(p[1]) * cos(0.7p[2])
uxy(p) = -0.7cos(p[1]) * sin(0.7p[2]) + 0.1

# -(a uₓ)ₓ - (b u_y)ₓ - (b uₓ)_y - (c u_y)_y + the reaction-free part of the mixed case.
const COEF = (
    poisson = (a = p -> 1.0, ax = p -> 0.0, b = p -> 0.0, bx = p -> 0.0, by = p -> 0.0,
        c = p -> 1.0, cy = p -> 0.0),
    mixed = (a = p -> 2 + 0.5sin(p[1] + p[2]), ax = p -> 0.5cos(p[1] + p[2]),
        b = p -> 0.5 + 0.2sin(p[1] * p[2]), bx = p -> 0.2p[2] * cos(p[1] * p[2]),
        by = p -> 0.2p[1] * cos(p[1] * p[2]), c = p -> 2 + 0.3cos(p[1]), cy = p -> 0.0))

function rhs(k)
    p -> -(k.ax(p) * ux(p) + k.a(p) * uxx(p)) - (k.bx(p) * uy(p) + k.b(p) * uxy(p)) -
         (k.by(p) * ux(p) + k.b(p) * uxy(p)) - (k.cy(p) * uy(p) + k.c(p) * uyy(p))
end

# Seed i of n at (i + 0.7(U - 1/2)) L / (n + 1), U uniform on (0, 1).
jitter(rng, L, n) = [(i + 0.7 * (rand(rng) - 0.5)) * L / (n + 1) for i in 1:n]

# Dual length of node `j` along the grid line `xs`.
function dual(xs, j)
    lo = j > 1 ? (xs[j] - xs[j - 1]) / 2 : 0.0
    hi = j < length(xs) ? (xs[j + 1] - xs[j]) / 2 : 0.0
    return lo + hi
end

# H¹ₕ error over the edges with both ends in the closed polygon, and the max error over
# the interior nodes.
function errors(xs, ys, inn, Γ, uh)
    nx, ny = length(xs), length(ys)
    E = reshape(uh, nx, ny) .- [uex((x, y)) for x in xs, y in ys]
    cl = reshape(inn .| Γ, nx, ny)
    I = reshape(inn, nx, ny)
    s = 0.0
    for l in 1:ny, j in 1:(nx - 1)

        (cl[j, l] && cl[j + 1, l]) || continue
        s += dual(ys, l) * (E[j + 1, l] - E[j, l])^2 / (xs[j + 1] - xs[j])
    end
    for l in 1:(ny - 1), j in 1:nx

        (cl[j, l] && cl[j, l + 1]) || continue
        s += dual(xs, j) * (E[j, l + 1] - E[j, l])^2 / (ys[l + 1] - ys[l])
    end
    return sqrt(s), maximum(abs(E[k]) for k in eachindex(E) if I[k])
end

# One solve: fresh closure from the jittered seeds, the scheme through the library.
function solve_level(case, n, seed)
    k = COEF[case]
    P = Bramble.polygon(FIG1)
    Ω = domain(interval(0.0, LX) × interval(0.0, LY), :P => P)
    rng = Xoshiro(1000seed + n)
    sx, sy = [0.0; jitter(rng, LX, n); LX], [0.0; jitter(rng, LY, n); LY]
    xs, ys = Bramble._grid_closure(P, sx, sy)
    Ωₕ = mesh(domain(interval(0.0, LX) × interval(0.0, LY)), (length(xs), length(ys)),
        (false, false))
    Bramble.change_points!(Ωₕ, markers(Ω), (xs, ys))
    U, W = Bramble.polygon_spaces(Ωₕ, :P, uex)
    G = gridspace(Ωₕ)
    nx, ny = length(xs), length(ys)
    aₓ, c_y, bₕ, fₕ = element(G), element(G), Rₕ(G, k.b), element(G)
    A3, C3 = reshape(aₓ.data, nx, ny), reshape(c_y.data, nx, ny)
    for l in 1:ny, j in 1:nx

        A3[j, l] = j > 1 ? k.a(((xs[j - 1] + xs[j]) / 2, ys[l])) : 0.0
        C3[j, l] = l > 1 ? k.c((xs[j], (ys[l - 1] + ys[l]) / 2)) : 0.0
    end
    avgₕ!(fₕ, rhs(k))
    mixed(u) = Dcₓ(bₕ * Dcᵧ(u)) + Dcᵧ(bₕ * Dcₓ(u))
    a = form(U, W,
        (u, v) -> innerₕ(-D̃ₓ(aₓ * D₋ₓ(u)) - D̃ᵧ(c_y * D₋ᵧ(u)) - mixed(u), v))
    A, F = assemble(a, form(W, v -> innerₕ(fₕ, v)))
    uh = Bramble.lift(U, A \ F).data
    inn = collect(Bramble.index_in_marker(Ωₕ, :P))
    Γ = collect(Bramble.index_in_marker(Ωₕ, :P_boundary))
    e1, emax = errors(xs, ys, inn, Γ, uh)
    return max(maximum(diff(xs)), maximum(diff(ys))), e1, emax
end

geomean(v) = exp(sum(log, v) / length(v))

"""
    convergence_errors(; levels, seeds) -> (poisson = (h, e1, emax), mixed = (h, e1, emax))

Per level `n` of `levels`, the geometric mean over `seeds` of the largest grid step `h`, the
H¹ₕ error `e1` and the max error `emax`, for the Poisson case and the mixed-term case.
"""
function convergence_errors(; levels = (6, 12, 24, 48, 96, 192), seeds = 1:3)
    run(case) = begin
        r = [[solve_level(case, n, s) for s in seeds] for n in levels]
        (h = [geomean(first.(x)) for x in r], e1 = [geomean(getindex.(x, 2)) for x in r],
            emax = [geomean(getindex.(x, 3)) for x in r])
    end
    return (poisson = run(:poisson), mixed = run(:mixed))
end

# Least-squares slope of log e against log h.
function slope(h, e)
    x, y = log.(h), log.(e)
    n = length(x)
    return (n * sum(x .* y) - sum(x) * sum(y)) / (n * sum(x .^ 2) - sum(x)^2)
end

# --- the SVG plot ---------------------------------------------------------------------

const PX0, PX1, PY0, PY1 = 70.0, 430.0, 30.0, 300.0   # plot box in SVG units

function svg_plot(o)
    hs = vcat(o.poisson.h, o.mixed.h)
    es = vcat(o.poisson.e1, o.poisson.emax, o.mixed.e1, o.mixed.emax)
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
        "the Poisson and mixed-term problems, with reference lines of slope 2 and 1.5.\">")
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
    series = ((o.poisson.h, o.poisson.e1, "#2563eb", "Poisson, H¹ₕ", false),
        (o.poisson.h, o.poisson.emax, "#2563eb", "Poisson, max", true),
        (o.mixed.h, o.mixed.e1, "#ea580c", "mixed, H¹ₕ", false),
        (o.mixed.h, o.mixed.emax, "#ea580c", "mixed, max", true))
    for (h, e, c, _, dashed) in series
        println(io, "  <polyline points=\"$(pts(h, e))\" fill=\"none\" stroke=\"$c\" ",
            "stroke-width=\"2\"", dashed ? " stroke-dasharray=\"6,3\"" : "", "/>")
        for (a, b) in zip(h, e)
            println(io, "  <circle cx=\"$(f(X(a)))\" cy=\"$(f(Y(b)))\" r=\"3\" fill=\"$c\"/>")
        end
    end
    # Reference lines through the finest Poisson H¹ₕ point, drawn below the data.
    h0, e0 = o.poisson.h[end], o.poisson.e1[end] / 4
    for (m, c) in ((2, "currentColor"), (1.5, "#16a34a"))
        ha, hb = minimum(hs), maximum(hs)
        ea, eb = e0 * (ha / h0)^m, e0 * (hb / h0)^m
        println(io, "  <polyline points=\"$(pts([ha, hb], [ea, eb]))\" fill=\"none\" ",
            "stroke=\"$c\" stroke-opacity=\"0.6\" stroke-width=\"1.2\" ",
            "stroke-dasharray=\"2,3\"/>")
        println(io, "  <text x=\"$(f(X(hb)) )\" y=\"$(f(Y(eb)) )\" font-size=\"11\" ",
            "fill=\"$c\" text-anchor=\"end\" dy=\"-4\">slope $m</text>")
    end
    for (i, (_, _, c, name, dashed)) in enumerate(series)
        y = PY0 + 14 + 16 * (i - 1)
        println(io, "  <line x1=\"$(PX1 + 14)\" y1=\"$(y - 4)\" x2=\"$(PX1 + 40)\" y2=\"$(y - 4)\" ",
            "stroke=\"$c\" stroke-width=\"2\"", dashed ? " stroke-dasharray=\"6,3\"" : "", "/>")
        println(io, "  <text x=\"$(PX1 + 46)\" y=\"$y\" font-size=\"11\" fill=\"currentColor\">",
            "$name</text>")
    end
    println(io, "</svg>")
    return String(take!(io))
end

function page(o)
    r(x) = round(x; digits = 2)
    sl(c) = (r(slope(c.h, c.e1)), r(slope(c.h, c.emax)))
    sp, sm = sl(o.poisson), sl(o.mixed)
    return """
# Observed orders of the polygon scheme

The scheme of `polygon_spaces` on Figure 1's polygon, solved on a fresh grid closure
at each of six refinement levels (6 to 192 jittered grid lines per axis), against a smooth
exact solution. The plot shows the ``H^1_h`` error (solid) and the maximum error (dashed)
against the largest grid step, for the Poisson problem and for a problem with variable
coefficients and a mixed term. The dotted lines have slopes 2 and 1.5.

```@raw html
<figure>
$(svg_plot(o))</figure>
```

The fitted slopes, in the order ``H^1_h``, maximum: Poisson $(sp[1]) and $(sp[2]), mixed term
$(sm[1]) and $(sm[2]). The page is written by `poisson_polygon_convergence.jl` next to it;
the test gating these orders is `test/convergence/poisson_polygon.jl`.
"""
end

if abspath(PROGRAM_FILE) == @__FILE__
    o = convergence_errors(; seeds = 1:3)
    write(joinpath(@__DIR__, "poisson_polygon_convergence.md"), page(o))
    @info "wrote poisson_polygon_convergence.md" poisson=(slope(o.poisson.h, o.poisson.e1),
        slope(o.poisson.h, o.poisson.emax)) mixed=(slope(o.mixed.h, o.mixed.e1),
        slope(o.mixed.h, o.mixed.emax))
end
