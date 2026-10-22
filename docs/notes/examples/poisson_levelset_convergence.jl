# Generator for `poisson_levelset_convergence.md`: the observed orders of the Shortley-Weller
# scheme on level-set domains (#834), the disk and the five-petal star, drawn as a hand-built
# SVG log-log plot, no plotting dependency. Run
# `julia --project=. docs/notes/examples/poisson_levelset_convergence.jl` to rewrite the
# page. The setup is copied from `test/convergence/poisson_levelset.jl` (the gate), not
# included from it. One jittered seed is enough here; the gate pools three.

using Bramble
using Random
import Bramble
using Bramble: levelset, levelset_spaces, D₋ₓ, D₋ᵧ, D̃ₓ, D̃ᵧ, Dcₓ, Dcᵧ

const LS_BOX = interval(-1.25, 1.25) × interval(-1.25, 1.25)
const LS_SHAPES = (disk = p -> hypot(p[1], p[2]) - 1,
    star = p -> hypot(p[1], p[2]) - (0.6 + 0.2 * sin(5 * atan(p[2], p[1]))))

# Exact solution and its derivatives.
ls_u(p) = sin(2p[1]) * cos(2p[2]) + p[1] * p[2]
ls_ux(p) = 2cos(2p[1]) * cos(2p[2]) + p[2]
ls_uy(p) = -2sin(2p[1]) * sin(2p[2]) + p[1]
ls_uxx(p) = -4sin(2p[1]) * cos(2p[2])
ls_uyy(p) = -4sin(2p[1]) * cos(2p[2])

# −∂ₓ(a ∂ₓu) − ∂ᵧ(c ∂ᵧu) + β·∇u + r u, as `_PP_COEF` in poisson_polygon.jl.
const LS_COEF = (
    poisson = (a = p -> 1.0, ax = p -> 0.0, c = p -> 1.0, cy = p -> 0.0, β = (0.0, 0.0),
        r = 0.0),
    variable = (a = p -> 2 + 0.5sin(p[1] + p[2]), ax = p -> 0.5cos(p[1] + p[2]),
        c = p -> 2 + 0.3cos(p[1]), cy = p -> 0.0, β = (0.4, -0.3), r = 1.5))

function ls_rhs(k)
    return p -> -(k.ax(p) * ls_ux(p) + k.a(p) * ls_uxx(p)) -
                (k.cy(p) * ls_uy(p) + k.c(p) * ls_uyy(p)) +
                k.β[1] * ls_ux(p) + k.β[2] * ls_uy(p) + k.r * ls_u(p)
end

# n points in [lo, hi], jittered about the uniform ones, ends pinned.
function ls_lines(rng, n)
    lo, hi = -1.25, 1.25
    h = (hi - lo) / (n - 1)
    return [lo; [lo + (i + 0.7 * (rand(rng) - 0.5)) * h for i in 1:(n - 2)]; hi]
end

# The H¹ₕ error over the edges with both ends interior or boundary nodes and the max error
# over the interior nodes, of the grid function `uh` against the exact solution.
function ls_errors(xs, ys, inn, Γ, uh)
    nx, ny = length(xs), length(ys)
    E = reshape(uh, nx, ny) .- [ls_u((x, y)) for x in xs, y in ys]
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
    return sqrt(s), maximum(abs, E[In])
end

# One level: (hₘₐₓ, H¹ₕ error, max error).
function ls_level(φ, k, n, seed)
    rng = Xoshiro(1000seed + n)
    xs, ys = ls_lines(rng, n), ls_lines(rng, n)
    Ω = domain(LS_BOX, :D => levelset(φ))
    Ωₕ = mesh(Ω, (n, n), (false, false))
    Bramble.change_points!(Ωₕ, markers(Ω), (xs, ys))
    U, W = levelset_spaces(Ωₕ, Ω, :D, ls_u)
    S = Bramble.padded_space(U)
    aₓ, cᵧ = element(S), element(S)
    A3, C3 = reshape(aₓ.data, n, n), reshape(cᵧ.data, n, n)
    for l in 1:n, j in 1:n

        A3[j, l] = j > 1 ? k.a(((xs[j - 1] + xs[j]) / 2, ys[l])) : 0.0
        C3[j, l] = l > 1 ? k.c((xs[j], (ys[l - 1] + ys[l]) / 2)) : 0.0
    end
    fₕ = avgₕ(S, ls_rhs(k))
    op(u) = -D̃ₓ(aₓ * D₋ₓ(u)) - D̃ᵧ(cᵧ * D₋ᵧ(u)) + k.β[1] * Dcₓ(u) + k.β[2] * Dcᵧ(u) +
            k.r * u
    A, F = assemble(form(U, W, (u, v) -> innerₕ(op(u), v)), form(W, v -> innerₕ(fₕ, v)))
    uh = Bramble.lift(U, A \ F).data
    m = markers(Ωₕ)
    e1, emax = ls_errors(xs, ys, BitVector(m[:D]), BitVector(m[:D_boundary]), uh)
    return max(maximum(diff(xs)), maximum(diff(ys))), e1, emax
end


geomean(v) = exp(sum(log, v) / length(v))

"""
    convergence_errors(; levels, seeds) -> NamedTuple of (h, e1, emax)

Per level `n` of `levels`, the geometric mean over `seeds` of the largest grid step `h`, the
H¹ₕ error `e1` and the max error `emax`, for the disk and the star with `-Δₕ` and for the
disk with variable coefficients.
"""
function convergence_errors(; levels = (20, 40, 80, 160, 320), seeds = 1:1)
    run(shape, case) = begin
        r = [[ls_level(LS_SHAPES[shape], LS_COEF[case], n, s) for s in seeds]
             for n in levels]
        (h = [geomean(first.(x)) for x in r], e1 = [geomean(getindex.(x, 2)) for x in r],
            emax = [geomean(getindex.(x, 3)) for x in r])
    end
    return (disk = run(:disk, :poisson), star = run(:star, :poisson),
        variable = run(:disk, :variable))
end

# Least-squares slope of log e against log h.
function slope(h, e)
    x, y = log.(h), log.(e)
    n = length(x)
    return (n * sum(x .* y) - sum(x) * sum(y)) / (n * sum(x .^ 2) - sum(x)^2)
end

# --- the SVG plot ---------------------------------------------------------------------

const PX0, PX1, PY0, PY1 = 70.0, 430.0, 30.0, 300.0   # plot box in SVG units
const COLORS = (disk = "#2563eb", star = "#ea580c", variable = "#9333ea")
const NAMES = (disk = "disk, -Δₕ", star = "star, -Δₕ", variable = "disk, variable")

function svg_plot(o)
    cs = values(o)
    hs = reduce(vcat, [c.h for c in cs])
    es = reduce(vcat, [[c.e1; c.emax] for c in cs])
    lx0, lx1 = floor(log10(minimum(hs)) * 2) / 2, ceil(log10(maximum(hs)) * 2) / 2
    ly0, ly1 = floor(log10(minimum(es))), ceil(log10(maximum(es)))
    X(h) = PX0 + (log10(h) - lx0) / (lx1 - lx0) * (PX1 - PX0)
    Y(e) = PY1 - (log10(e) - ly0) / (ly1 - ly0) * (PY1 - PY0)
    f(x) = string(round(x; digits = 1))
    pts(h, e) = join(("$(f(X(a))),$(f(Y(b)))" for (a, b) in zip(h, e)), " ")
    io = IOBuffer()
    println(io, "<svg viewBox=\"0 0 640 360\" width=\"100%\" ",
        "style=\"max-width:640px;height:auto;font-family:system-ui,-apple-system,",
        "'Segoe UI',sans-serif\" xmlns=\"http://www.w3.org/2000/svg\" role=\"img\" ",
        "aria-label=\"Log-log plot of the H1 and maximum errors against the grid step for ",
        "the disk, the star and the disk with variable coefficients, with reference lines ",
        "of slope 2 and 3.\">")
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
    series = [(o[k].h, getproperty(o[k], q), COLORS[k], "$(NAMES[k]), $lab", dashed)
              for k in keys(o) for (q, lab, dashed) in ((:e1, "H¹ₕ", false),
        (:emax, "max", true))]
    for (h, e, c, _, dashed) in series
        println(io, "  <polyline points=\"$(pts(h, e))\" fill=\"none\" stroke=\"$c\" ",
            "stroke-width=\"2\"", dashed ? " stroke-dasharray=\"6,3\"" : "", "/>")
        for (a, b) in zip(h, e)
            println(io, "  <circle cx=\"$(f(X(a)))\" cy=\"$(f(Y(b)))\" r=\"3\" fill=\"$c\"/>")
        end
    end
    # Reference lines through the finest disk H¹ₕ point, drawn below the data.
    h0, e0 = o.disk.h[end], o.disk.e1[end] / 4
    for (m, c) in ((2, "currentColor"), (3, "#16a34a"))
        ha, hb = minimum(hs), maximum(hs)
        ea, eb = e0 * (ha / h0)^m, e0 * (hb / h0)^m
        println(io, "  <polyline points=\"$(pts([ha, hb], [ea, eb]))\" fill=\"none\" ",
            "stroke=\"$c\" stroke-opacity=\"0.6\" stroke-width=\"1.2\" ",
            "stroke-dasharray=\"2,3\"/>")
        println(io, "  <text x=\"$(f(X(hb)))\" y=\"$(f(Y(eb)))\" font-size=\"11\" ",
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
    sd, ss, sv = sl(o.disk), sl(o.star), sl(o.variable)
    return """
# Observed orders of the level-set scheme

The scheme of `levelset_spaces` on the disk `r - 1` and the five-petal star
`r - (0.6 + 0.2 sin 5ϑ)` in the box ``[-1.25, 1.25]^2``, solved on a fresh jittered grid at
each of five refinement levels (20 to 320 points per axis, one seed), against the smooth
exact solution `sin 2x cos 2y + xy`. The plot shows the ``H^1_h`` error (solid) and the
maximum error (dashed) against the largest grid step, for `-Δₕ` on the disk and on the star
and for the disk with variable coefficients, a convection term and a reaction term. The
dotted lines have slopes 2 and 3.

```@raw html
<figure>
$(svg_plot(o))</figure>
```

The fitted slopes, in the order ``H^1_h``, maximum: disk $(sd[1]) and $(sd[2]), star
$(ss[1]) and $(ss[2]), disk with variable coefficients $(sv[1]) and $(sv[2]). The page is
written by `poisson_levelset_convergence.jl` next to it; the test gating these orders, with
three seeds pooled, is `test/convergence/poisson_levelset.jl`.
"""
end

if abspath(PROGRAM_FILE) == @__FILE__
    o = convergence_errors()
    write(joinpath(@__DIR__, "poisson_levelset_convergence.md"), page(o))
    @info "wrote poisson_levelset_convergence.md" disk=(slope(o.disk.h, o.disk.e1),
        slope(o.disk.h, o.disk.emax)) star=(slope(o.star.h, o.star.e1),
        slope(o.star.h, o.star.emax)) variable=(slope(o.variable.h, o.variable.e1),
        slope(o.variable.h, o.variable.emax))
end
