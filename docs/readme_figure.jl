# Writes docs/src/assets/readme_convergence.svg, the figure in README.md: the maximum error of
# the README's quick-start Poisson problem against hₘₐₓ, on uniform and on random meshes.
#
#     julia --project=. docs/readme_figure.jl [output.svg]
#
# The SVG is written by hand, so the figure needs no plotting package. Run it again after a
# change to the scheme and commit the new file.

using Bramble, Random, Printf

uexact(x) = sinpi(x[1]) * sinpi(x[2])
g(x) = 2π^2 * uexact(x)

# The quick start of README.md on an n × n mesh: (hₘₐₓ, max error at the grid points).
function poisson_error(n, uniform)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    Ωₕ = mesh(Ω, (n, n), (uniform, uniform))
    Wₕ = gridspace(Ωₕ)
    gₕ = element(Wₕ)
    Rₕ!(gₕ, g)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    l = form(Wₕ, v -> innerₕ(gₕ, v))
    A, F = assemble(a, l; dirichlet = dirichlet_constraints(Ω, :boundary => uexact))
    uₕ = element(Wₕ)
    uₕ .= A \ F
    return hₘₐₓ(Ωₕ), maximum(abs, uₕ .- Rₕ(Wₕ, uexact))
end

# Least-squares slope of log(err) against log(h).
function observed_order(hs, errs)
    x = log.(hs) .- sum(log, hs) / length(hs)
    y = log.(errs)
    return sum(x .* y) / sum(abs2, x)
end

const SIZES = (17, 33, 65, 129, 257)
const SEEDS = 1:5

uniform = [poisson_error(n, true) for n in SIZES]
# A random mesh draws its points from the global generator: seed before each construction.
random = [(Random.seed!(seed); poisson_error(n, false)) for n in SIZES for seed in SEEDS]

for (label, data) in (("uniform", uniform), ("random", random))
    @printf("%-8s order %.2f over %d meshes\n", label,
        observed_order(first.(data), last.(data)), length(data))
    foreach(d -> @printf("  h = %.4e  err = %.4e\n", d...), data)
end

# --- the figure ---------------------------------------------------------------------------

const WIDTH, HEIGHT = 560, 340
const LEFT, RIGHT, TOP, BOTTOM = 62, 18, 16, 46

all_h = vcat(first.(uniform), first.(random))
all_e = vcat(last.(uniform), last.(random))
const XLIM = (minimum(all_h) / 1.4, maximum(all_h) * 1.4)
const YLIM = (minimum(all_e) / 2.5, maximum(all_e) * 2.5)

px(h) = LEFT + (WIDTH - LEFT - RIGHT) * log(h / XLIM[1]) / log(XLIM[2] / XLIM[1])
py(e) = HEIGHT - BOTTOM - (HEIGHT - TOP - BOTTOM) * log(e / YLIM[1]) / log(YLIM[2] / YLIM[1])

const SUPERSCRIPTS = collect("⁰¹²³⁴⁵⁶⁷⁸⁹")
superscript(k) = join(c == '-' ? '⁻' : SUPERSCRIPTS[c - '0' + 1] for c in string(k))
decades(lim) = ceil(Int, log10(lim[1])):floor(Int, log10(lim[2]))

io = IOBuffer()
println(io, """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $WIDTH $HEIGHT" width="$WIDTH" height="$HEIGHT" role="img">
<title>Maximum error against the largest spacing, on uniform and on random meshes</title>
<style>
text { font: 13px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #24292f; }
.frame { fill: none; stroke: #8c959f; }
.grid { stroke: #8c959f; stroke-opacity: 0.3; }
.slope { stroke: #8c959f; stroke-width: 1.5; stroke-dasharray: 6 4; }
.uniform { fill: #4063d8; }
.random { fill: #cb3c33; fill-opacity: 0.75; }
@media (prefers-color-scheme: dark) { text { fill: #e6edf3; } }
</style>""")

for k in decades(XLIM)
    x = px(10.0^k)
    @printf(io, "<line class=\"grid\" x1=\"%.1f\" y1=\"%d\" x2=\"%.1f\" y2=\"%d\"/>\n",
        x, TOP, x, HEIGHT - BOTTOM)
    @printf(io, "<text x=\"%.1f\" y=\"%d\" text-anchor=\"middle\">10%s</text>\n",
        x, HEIGHT - BOTTOM + 18, superscript(k))
end
for k in decades(YLIM)
    y = py(10.0^k)
    @printf(io, "<line class=\"grid\" x1=\"%d\" y1=\"%.1f\" x2=\"%d\" y2=\"%.1f\"/>\n",
        LEFT, y, WIDTH - RIGHT, y)
    @printf(io, "<text x=\"%d\" y=\"%.1f\" text-anchor=\"end\">10%s</text>\n",
        LEFT - 8, y + 4.5, superscript(k))
end
@printf(io, "<rect class=\"frame\" x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\"/>\n",
    LEFT, TOP, WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM)

# The slope-2 line through the finest uniform mesh, clipped to the frame at its low end.
h0, e0 = uniform[end]
slope2(h) = e0 * (h / h0)^2
hlo = max(XLIM[1], h0 * sqrt(YLIM[1] / e0))
hhi = min(XLIM[2], h0 * sqrt(YLIM[2] / e0))
@printf(io, "<line class=\"slope\" x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\"/>\n",
    px(hlo), py(slope2(hlo)), px(hhi), py(slope2(hhi)))

diamond(x, y, r) = @sprintf("M%.1f %.1fl%.1f %.1fl%.1f %.1fl%.1f %.1fz", x, y - r, r, r, -r, r, -r, -r)
for (h, e) in random
    println(io, "<path class=\"random\" d=\"", diamond(px(h), py(e), 4.5), "\"/>")
end
for (h, e) in uniform
    @printf(io, "<circle class=\"uniform\" cx=\"%.1f\" cy=\"%.1f\" r=\"4.5\"/>\n", px(h), py(e))
end

# Legend, top left.
lx, ly = LEFT + 16, TOP + 20
@printf(io, "<circle class=\"uniform\" cx=\"%d\" cy=\"%d\" r=\"4.5\"/>\n", lx, ly - 4)
@printf(io, "<text x=\"%d\" y=\"%d\">uniform mesh</text>\n", lx + 14, ly)
println(io, "<path class=\"random\" d=\"", diamond(lx, ly + 16, 4.5), "\"/>")
@printf(io, "<text x=\"%d\" y=\"%d\">random mesh, %d draws per size</text>\n",
    lx + 14, ly + 20, length(SEEDS))
@printf(io, "<line class=\"slope\" x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\"/>\n",
    lx - 8, ly + 36, lx + 8, ly + 36)
@printf(io, "<text x=\"%d\" y=\"%d\">slope 2</text>\n", lx + 14, ly + 40)

@printf(io, "<text x=\"%.1f\" y=\"%d\" text-anchor=\"middle\">hₘₐₓ</text>\n",
    (LEFT + WIDTH - RIGHT) / 2, HEIGHT - 8)
@printf(io, "<text transform=\"translate(14 %.1f) rotate(-90)\" text-anchor=\"middle\">maximum error</text>\n",
    (TOP + HEIGHT - BOTTOM) / 2)
println(io, "</svg>")

output = isempty(ARGS) ? joinpath(@__DIR__, "src", "assets", "readme_convergence.svg") : ARGS[1]
write(output, take!(io))
println("wrote ", output)
