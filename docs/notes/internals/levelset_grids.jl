# Draws the closure-path and orbit-path grids of `levelset.md` ("Two paths") as one inline
# SVG, printed to stdout. Run from the repository root:
#
#     julia --project=test docs/notes/internals/levelset_grids.jl > /tmp/grids.svg
#
# and paste the output between the figure markers in `levelset.md`.

using Bramble, Random

function seeds(rng, lo, hi, n)
    return sort([lo; [lo + (i + 0.6 * (rand(rng) - 0.5)) * (hi - lo) / (n + 1) for i in 1:n]; hi])
end

egg(θ, k) = p -> begin
    x, y = p[1] - 0.1, p[2] + 0.05
    u, v = cos(θ) * x + sin(θ) * y, -sin(θ) * x + cos(θ) * y
    u^2 + v^2 * (1 + k * u) / 0.36 - 1
end

# The curve φ = 0 by bisection along rays from a point inside (every curve here is
# star-shaped about it).
function curve(φ, c; m = 720)
    return map(range(0, 2π; length = m)) do t
        a, b = 0.0, 3.0
        for _ in 1:60
            r = (a + b) / 2
            φ((c[1] + r * cos(t), c[2] + r * sin(t))) < 0 ? (a = r) : (b = r)
        end
        (c[1] + a * cos(t), c[2] + a * sin(t))
    end
end

fmt(z) = string(round(z; digits = 1))

function panel(io, φ, c, box, n, title, ox; W = 400, pad = 16)
    rng = MersenneTwister(1)
    (x0, x1), (y0, y1) = box
    X, Y, P = Bramble._levelset_grid(φ, seeds(rng, x0, x1, n), seeds(rng, y0, y1, n))
    s = (W - 2pad) / max(x1 - x0, y1 - y0)
    tx(x) = ox + pad + (x - x0) * s
    ty(y) = 34 + pad + (y1 - y) * s
    pts(V) = join(["$(fmt(tx(v[1]))),$(fmt(ty(v[2])))" for v in V], " ")
    V = P.vertices
    println(io, """  <g stroke="#3b82f6" stroke-opacity="0.55" stroke-width="0.6">""")
    foreach(x -> println(io, """    <line x1="$(fmt(tx(x)))" y1="$(fmt(ty(y0)))" x2="$(fmt(tx(x)))" y2="$(fmt(ty(y1)))"/>"""), X)
    foreach(y -> println(io, """    <line x1="$(fmt(tx(x0)))" y1="$(fmt(ty(y)))" x2="$(fmt(tx(x1)))" y2="$(fmt(ty(y)))"/>"""), Y)
    println(io, "  </g>")
    println(io, """  <polyline points="$(pts(curve(φ, c)))" fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="4,3"/>""")
    println(io, """  <polygon points="$(pts(V))" fill="#ef4444" fill-opacity="0.08" stroke="#ef4444" stroke-width="1.4"/>""")
    println(io, """  <g fill="#ef4444">""")
    foreach(v -> println(io, """    <circle cx="$(fmt(tx(v[1])))" cy="$(fmt(ty(v[2])))" r="1.8"/>"""), V)
    println(io, "  </g>")
    println(io, """  <text x="$(ox + W / 2)" y="20" text-anchor="middle" font-size="14" fill="currentColor">$title</text>""")
    println(io,
        """  <text x="$(ox + W / 2)" y="$(W + 52)" text-anchor="middle" font-size="12" fill="currentColor" opacity="0.8">$(length(X)) × $(length(Y)) lines, $(length(V)) vertices</text>""")
end

W = 400
box_d = ((-1.25, 1.25), (-1.25, 1.25))
box_e = ((-1.2, 1.3), (-1.25, 1.15))
io = stdout
println(io,
    """<svg viewBox="0 0 $(3W) $(W + 64)" width="100%" style="max-width:$(3W)px;font-family:system-ui,sans-serif" role="img" aria-label="Fitted grids at 20 jittered seeds per axis: the disk and the egg at 0 degrees on the closure path, the egg at 35 degrees on the orbit path.">""")
for (k, (φ, c, box, t)) in enumerate([
    (p -> hypot(p[1], p[2]) - 1, (0.0, 0.0), box_d, "disk, closure path"),
    (egg(0.0, 0.4), (0.1, -0.05), box_e, "egg at 0°, closure path"),
    (egg(deg2rad(35), 0.4), (0.1, -0.05), box_e, "egg at 35°, orbit path")])
    panel(io, φ, c, box, 20, t, (k - 1) * W)
end
println(io, "</svg>")
