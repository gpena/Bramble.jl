# Writes the six figures of the home-page gallery to docs/src/assets/gallery_*.svg:
#
#     julia --project=docs docs/gallery_figures.jl [output_dir]
#
# Each figure solves a small problem of the worked examples with Bramble, on a seeded or
# deterministic grid, and draws the result by hand, so the figures need no plotting package.
# Run it again after a change to a scheme and commit the new files.

module GalleryFigures

using Bramble, Random, Printf
using Bramble: polygon_spaces, lift, change_points!
using ForwardDiff
using OrdinaryDiffEqBDF

export gallery_figures

# --- drawing -------------------------------------------------------------------------------

const WIDTH, HEIGHT = 340, 270

const VIRIDIS = ((68, 1, 84), (71, 45, 123), (59, 82, 139), (44, 114, 142), (33, 144, 140),
    (39, 173, 129), (93, 200, 99), (170, 220, 50), (253, 231, 37))

# Hex colour of the viridis map at t in [0, 1], quantised to 48 levels so that cells of one
# colour share a path.
function colour(t)
    s = round(clamp(t, 0, 1) * 47) / 47 * (length(VIRIDIS) - 1)
    k = min(floor(Int, s), length(VIRIDIS) - 2)
    f = s - k
    c = ntuple(i -> round(Int, (1 - f) * VIRIDIS[k + 1][i] + f * VIRIDIS[k + 2][i]), 3)
    return @sprintf("#%02x%02x%02x", c...)
end

label(x) = (s = @sprintf("%.1f", x); s == "-0.0" ? "0.0" : s)

function open_svg(title)
    io = IOBuffer()
    println(io, """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $WIDTH $HEIGHT" width="$WIDTH" height="$HEIGHT" role="img">
    <title>$title</title>
    <style>
    text { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #24292f; }
    .frame { fill: none; stroke: #57606a; }
    .bar { stroke: #57606a; }
    .mesh { fill: none; stroke: #fff; stroke-opacity: 0.55; stroke-width: 0.6; }
    .ref { fill: none; stroke: #8c959f; stroke-opacity: 0.5; stroke-width: 0.6; }
    .edge { fill: none; stroke: #24292f; stroke-width: 1; }
    .exact { fill: none; stroke: #8c959f; stroke-width: 1.5; }
    .node { fill: #cb3c33; }
    .rug { stroke: #cb3c33; stroke-width: 0.8; }
    </style>
    <rect width="$WIDTH" height="$HEIGHT" rx="8" fill="#fff"/>""")
    return io
end

function close_svg(io)
    println(io, "</svg>")
    return String(take!(io))
end

# A rectangle of the picture: data limits to pixels, y upwards.
struct Panel
    x0::Float64
    y0::Float64
    w::Float64
    h::Float64
    xlim::NTuple{2, Float64}
    ylim::NTuple{2, Float64}
end
px(p::Panel, x) = p.x0 + p.w * (x - p.xlim[1]) / (p.xlim[2] - p.xlim[1])
py(p::Panel, y) = p.y0 + p.h * (1 - (y - p.ylim[1]) / (p.ylim[2] - p.ylim[1]))

function frame!(io, p::Panel)
    @printf(io, "<rect class=\"frame\" x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\"/>\n",
        p.x0, p.y0, p.w, p.h)
end

function text!(io, x, y, s; anchor = "middle")
    @printf(io, "<text x=\"%.1f\" y=\"%.1f\" text-anchor=\"%s\">%s</text>\n", x, y, anchor, s)
end

# One path per colour: the polygons `polys` (pixel corners) coloured by `vals` over [lo, hi].
# A stroke of the fill colour closes the seams between neighbours.
function cells!(io, polys, vals, lo, hi)
    groups = Dict{String, Vector{Int}}()
    for (k, v) in enumerate(vals)
        push!(get!(groups, colour((v - lo) / (hi - lo)), Int[]), k)
    end
    for c in sort!(collect(keys(groups)))
        @printf(io, "<path fill=\"%s\" stroke=\"%s\" stroke-width=\"0.6\" d=\"", c, c)
        for k in groups[c]
            print(io, "M")
            for (x, y) in polys[k]
                @printf(io, "%.1f %.1f ", x, y)
            end
            print(io, "z")
        end
        println(io, "\"/>")
    end
end

# The mean of the four corner values of every cell of a grid function on a tensor grid.
function corner_means(Z)
    [(Z[i, j] + Z[i + 1, j] + Z[i, j + 1] + Z[i + 1, j + 1]) / 4
     for i in 1:(size(Z, 1) - 1), j in 1:(size(Z, 2) - 1)]
end

function rect_cells!(io, p::Panel, xs, ys, Z, lo, hi)
    polys = [((px(p, xs[i]), py(p, ys[j])), (px(p, xs[i + 1]), py(p, ys[j])),
                 (px(p, xs[i + 1]), py(p, ys[j + 1])), (px(p, xs[i]), py(p, ys[j + 1])))
             for i in 1:(length(xs) - 1), j in 1:(length(ys) - 1)]
    cells!(io, vec(polys), vec(corner_means(Z)), lo, hi)
end

# The grid lines of a tensor grid, as one path.
function grid_lines!(io, p::Panel, xs, ys; class = "mesh")
    @printf(io, "<path class=\"%s\" d=\"", class)
    for x in xs
        @printf(io, "M%.1f %.1fV%.1f", px(p, x), py(p, ys[1]), py(p, ys[end]))
    end
    for y in ys
        @printf(io, "M%.1f %.1fH%.1f", px(p, xs[1]), py(p, y), px(p, xs[end]))
    end
    println(io, "\"/>")
end

function stops(io, id, vertical)
    @printf(io, "<defs><linearGradient id=\"%s\" x1=\"0\" y1=\"%d\" x2=\"%d\" y2=\"0\">", id,
        vertical, 1 - vertical)
    for k in 0:8
        @printf(io, "<stop offset=\"%.3f\" stop-color=\"%s\"/>", k / 8, colour(k / 8))
    end
    println(io, "</linearGradient></defs>")
end

# A colour bar to the right of a panel, low at the bottom.
function colorbar!(io, x, y, h, lo, hi)
    stops(io, "bar", true)
    @printf(io, "<rect class=\"bar\" x=\"%.1f\" y=\"%.1f\" width=\"12\" height=\"%.1f\" fill=\"url(#bar)\"/>\n", x, y,
        h)
    text!(io, x + 18, y + 9, label(hi); anchor = "start")
    text!(io, x + 18, y + h, label(lo); anchor = "start")
end

# A colour bar under a panel, low on the left.
function colorbar_below!(io, p::Panel, id, lo, hi)
    stops(io, id, false)
    @printf(io, "<rect class=\"bar\" x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"8\" fill=\"url(#%s)\"/>\n",
        p.x0, p.y0 + p.h + 14, p.w, id)
    text!(io, p.x0, p.y0 + p.h + 38, label(lo); anchor = "start")
    text!(io, p.x0 + p.w, p.y0 + p.h + 38, label(hi); anchor = "end")
end

const SQUARE = Panel(20, 15, 240, 240, (0, 1), (0, 1))

# --- 1. Poisson on a random grid -----------------------------------------------------------

function poisson_figure()
    uexact(x) = sinpi(x[1]) * sinpi(x[2])
    Random.seed!(7)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    Ωₕ = mesh(Ω, (17, 17), (false, false))
    Wₕ = gridspace(Ωₕ)
    gₕ = element(Wₕ)
    Rₕ!(gₕ, x -> 2π^2 * uexact(x))
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    l = form(Wₕ, v -> innerₕ(gₕ, v))
    A, F = assemble(a, l; dirichlet = dirichlet_constraints(Ω, :boundary => uexact))
    uₕ = element(Wₕ)
    uₕ .= A \ F
    maximum(abs, uₕ .- Rₕ(Wₕ, uexact)) < 0.02 || error("the Poisson solution is off")
    xs, ys = collect(points(Ωₕ(1))), collect(points(Ωₕ(2)))
    U = reshape(collect(parent(uₕ)), length(xs), length(ys))

    io = open_svg("Poisson solution on a random 17 by 17 grid, with its grid lines")
    rect_cells!(io, SQUARE, xs, ys, U, 0.0, 1.0)
    grid_lines!(io, SQUARE, xs, ys)
    frame!(io, SQUARE)
    colorbar!(io, 276, SQUARE.y0, SQUARE.h, 0.0, 1.0)
    return close_svg(io)
end

# --- 2. the disk and its fitted grid -------------------------------------------------------

function disk_figure()
    φ(p) = hypot(p[1], p[2]) - 1
    u_exact(p) = sin(2p[1] + 0.5) * cos(1.5p[2] - 0.2) + exp(0.6p[1] - 0.4p[2])
    f(p) = 6.25 * sin(2p[1] + 0.5) * cos(1.5p[2] - 0.2) - 0.52 * exp(0.6p[1] - 0.4p[2])
    Ω = domain(interval(-1.25, 1.25) × interval(-1.25, 1.25), :D => levelset(φ))
    Ωₕ = mesh(Ω, (10, 10), (true, true))
    xs, ys = collect(points(Ωₕ(1))), collect(points(Ωₕ(2)))
    U, W = polygon_spaces(Ωₕ, :D, u_exact)
    fₕ = element(gridspace(Ωₕ))
    avgₕ!(fₕ, f)
    A, F = assemble(form(U, W, (u, v) -> innerₕ(-Δₕ(u), v)), form(W, v -> innerₕ(fₕ, v)))
    Z = reshape(lift(U, A \ F).data, length(xs), length(ys))
    m = markers(Ωₕ)
    inside = reshape(BitVector(m[:D]), length(xs), length(ys))
    on_circle = reshape(BitVector(m[:D_boundary]), length(xs), length(ys))
    nodes = inside .| on_circle
    err = maximum(abs(Z[i, j] - u_exact((xs[i], ys[j]))) for i in eachindex(xs), j in eachindex(ys) if nodes[i, j])
    err < 0.01 || error("the disk solution is off")
    lo, hi = extrema(Z[nodes])

    p = Panel(20, 15, 240, 240, (-1.1, 1.1), (-1.1, 1.1))
    io = open_svg("Poisson solution on the unit disk, on a grid fitted to the circle")
    # The inscribed polygon whose vertices are the nodes on the circle.
    ring = sort([(xs[i], ys[j]) for i in eachindex(xs), j in eachindex(ys) if on_circle[i, j]];
        by = v -> atan(v[2], v[1]))
    @printf(io, "<polygon class=\"edge\" fill=\"#f6f8fa\" points=\"%s\"/>\n",
        join((@sprintf("%.1f,%.1f", px(p, x), py(p, y)) for (x, y) in ring), " "))
    # Grid lines run between the outermost nodes of their row or column.
    print(io, "<path class=\"ref\" d=\"")
    for i in eachindex(xs)
        js = findall(nodes[i, :])
        isempty(js) || @printf(io, "M%.1f %.1fV%.1f", px(p, xs[i]), py(p, ys[js[1]]), py(p, ys[js[end]]))
    end
    for j in eachindex(ys)
        is = findall(nodes[:, j])
        isempty(is) || @printf(io, "M%.1f %.1fH%.1f", px(p, xs[is[1]]), py(p, ys[j]), px(p, xs[is[end]]))
    end
    println(io, "\"/>")
    for i in eachindex(xs), j in eachindex(ys)

        nodes[i, j] || continue
        @printf(io, "<circle cx=\"%.1f\" cy=\"%.1f\" r=\"%.1f\" fill=\"%s\"%s/>\n", px(p, xs[i]),
            py(p, ys[j]), on_circle[i, j] ? 3.2 : 2.6,
            colour((Z[i, j] - lo) / (hi - lo)),
            on_circle[i, j] ? " stroke=\"#24292f\" stroke-width=\"0.8\"" : "")
    end
    colorbar!(io, 276, p.y0, p.h, lo, hi)
    return close_svg(io)
end

# --- 3. the boundary layer on a graded grid ------------------------------------------------

function layer_figure()
    ε = 1.0e-3
    uexact(x) = x - (exp((x - 1) / ε) - exp(-1 / ε)) / (1 - exp(-1 / ε))
    duexact(x) = 1 - exp((x - 1) / ε) / (ε * (1 - exp(-1 / ε)))
    pts = [tanh(4t) / tanh(4) for t in range(0.0, 1.0; length = 41)]
    Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)
    Ωₕ = mesh(Ω, length(pts))
    change_points!(Ωₕ, pts)
    Cₕ = constrained(gridspace(Ωₕ; ghosts = (:xmin,)),
        u -> (:xmin => u + ∂ₙ(u) => (x -> uexact(x[1]) - duexact(x[1])),))
    a = form(Cₕ, Cₕ, (u, v) -> innerₕ(-ε * Δₕ(u) + ∇̽ₕ(u), v))
    l = form(Cₕ, v -> innerₕ(x -> 1.0, v))
    A, F = assemble(a, l; dirichlet = :xmax => (x -> uexact(x[1])))
    uₕ = element(gridspace(Ωₕ))
    uₕ .= A \ F
    xs, us = collect(points(Ωₕ)), collect(parent(uₕ))
    maximum(abs, us .- uexact.(xs)) < 0.05 || error("the layer solution is off")

    io = open_svg("Boundary layer of width 1e-3 on a graded grid of 41 points")
    for (p, title, tick) in ((Panel(30, 50, 130, 150, (0, 1), (0, 1.05)), "whole interval", "0"),
        (Panel(190, 50, 130, 150, (0.996, 1), (0, 1.05)), "last 0.4 %", "0.996"))
        lo = p.xlim[1]
        grid = sort!(unique!(vcat(range(lo, 1; length = 120), 1 .- 10 .^ range(-7, log10(1 - lo); length = 120))))
        grid = filter(x -> lo <= x <= 1, grid)
        @printf(io, "<polyline class=\"exact\" points=\"%s\"/>\n",
            join((@sprintf("%.1f,%.1f", px(p, x), py(p, uexact(x))) for x in grid), " "))
        for (x, u) in zip(xs, us)
            lo <= x || continue
            @printf(io, "<circle class=\"node\" cx=\"%.1f\" cy=\"%.1f\" r=\"2.3\"/>\n", px(p, x), py(p, u))
        end
        frame!(io, p)
        text!(io, p.x0 + p.w / 2, p.y0 - 12, title)
        text!(io, p.x0, p.y0 + p.h + 16, tick)
        text!(io, p.x0 + p.w, p.y0 + p.h + 16, "1"; anchor = "end")
        text!(io, p.x0 - 6, p.y0 + 10, "1"; anchor = "end")
        text!(io, p.x0 - 6, p.y0 + p.h, "0"; anchor = "end")
        p.xlim[1] == 0 && for x in xs
            @printf(io, "<line class=\"rug\" x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\"/>\n",
                px(p, x), p.y0 + p.h, px(p, x), p.y0 + p.h + 5)
        end
    end
    text!(io, WIDTH / 2, 244, "x"; anchor = "middle")
    text!(io, WIDTH / 2, 262, "ε = 10⁻³: 41 points, $(count(>=(1 - ε), xs)) in x ≥ 1 - ε")
    return close_svg(io)
end

# --- 4. the heat equation, x against t -----------------------------------------------------

function heat_figure()
    T, nt = 0.25, 32
    Random.seed!(3)
    Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)
    Ωₕ = mesh(Ω, 41, false)
    Wₕ = gridspace(Ωₕ)
    I = interval(0.0, T)
    fₕ = element(Wₕ, 0.0)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    sd = semidiscretize(a, l; dirichlet = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0))
    u0(x) = sinpi(x[1]) + 0.5 * sinpi(3x[1])
    sol = solve(ode_problem(sd, Rₕ(Wₕ, u0), I), FBDF(); reltol = 1.0e-9, abstol = 1.0e-12)
    exact(x, t) = exp(-π^2 * t) * sinpi(x) + 0.5 * exp(-9π^2 * t) * sinpi(3x)
    xs = collect(points(Ωₕ))
    ts = collect(range(0.0, T; length = nt + 1))
    Z = [sol(t)[i] for t in ts, i in eachindex(xs)]'    # x down the rows, t along the columns
    maximum(abs, Z .- [exact(x, t) for x in xs, t in ts]) < 0.02 || error("the heat solution is off")
    lo, hi = 0.0, maximum(Z)

    p = Panel(44, 15, 230, 230, (0, 1), (0, T))
    io = open_svg("Heat equation on a random grid of 41 points: the temperature against x and t")
    rect_cells!(io, p, xs, ts, Z, lo, hi)
    frame!(io, p)
    text!(io, p.x0, p.y0 + p.h + 16, "0")
    text!(io, p.x0 + p.w, p.y0 + p.h + 16, "1")
    text!(io, p.x0 + p.w / 2, p.y0 + p.h + 16, "x")
    text!(io, p.x0 - 7, p.y0 + p.h, "0"; anchor = "end")
    text!(io, p.x0 - 7, p.y0 + 10, "0.25"; anchor = "end")
    text!(io, p.x0 - 7, p.y0 + p.h / 2 + 4, "t"; anchor = "end")
    colorbar!(io, 286, p.y0, p.h, lo, hi)
    return close_svg(io)
end

# --- 5. the two species of the reaction-diffusion system -----------------------------------

function coupled_figure()
    u_ex(x) = sin(π * x[1]) * sin(π * x[2])
    v_ex(x) = sin(2π * x[1]) * sin(2π * x[2])
    f1(x) = 2π^2 * u_ex(x) + u_ex(x) + u_ex(x) * v_ex(x)
    f2(x) = 8π^2 * v_ex(x) + v_ex(x) - u_ex(x) * v_ex(x)
    Random.seed!(20260903)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    Ωₕ = mesh(Ω, (17, 17), (false, false))
    Wₕ = gridspace(Ωₕ)
    Vₕ = Wₕ^Val(2)
    f1ₕ, f2ₕ = element(Wₕ), element(Wₕ)
    avgₕ!(f1ₕ, f1)
    avgₕ!(f2ₕ, f2)
    F = assemble(form(Vₕ, q -> innerₕ(f1ₕ, q(1)) + innerₕ(f2ₕ, q(2)));
        dirichlet = dirichlet_constraints(Ω, :boundary => x -> 0.0))
    function matrix(wₕ)
        u_c, v_c = components(wₕ)
        a = form(Vₕ, Vₕ,
            (p, q) -> inner₊(∇ₕ(p(1)), ∇ₕ(q(1))) + innerₕ(p(1), q(1)) + innerₕ(v_c * p(1), q(1)) +
                      inner₊(∇ₕ(p(2)), ∇ₕ(q(2))) + innerₕ(p(2), q(2)) - innerₕ(u_c * p(2), q(2)))
        return assemble(a; dirichlet = :boundary)
    end
    # The coupling terms are frozen at the previous iterate: the matrix is linear in each species.
    w = zeros(ndofs(Vₕ))
    resid = Inf
    for _ in 1:30
        wₕ = element(Vₕ)
        wₕ .= w
        wnew = matrix(wₕ) \ F
        resid = maximum(abs, wnew .- w)
        w = wnew
        resid < 1.0e-12 && break
    end
    xs, ys = collect(points(Ωₕ(1))), collect(points(Ωₕ(2)))
    n = length(xs) * length(ys)
    U = reshape(w[1:n], length(xs), length(ys))
    V = reshape(w[(n + 1):(2n)], length(xs), length(ys))
    eu = maximum(abs(U[i, j] - u_ex((xs[i], ys[j]))) for i in eachindex(xs), j in eachindex(ys))
    ev = maximum(abs(V[i, j] - v_ex((xs[i], ys[j]))) for i in eachindex(xs), j in eachindex(ys))
    (eu < 0.02 && ev < 0.1) || error("the coupled solution is off: $eu, $ev, $resid")

    io = open_svg("The two species u and v of a reaction-diffusion system on a random 17 by 17 grid")
    for (p, Z, name, id) in ((Panel(10, 50, 150, 150, (0, 1), (0, 1)), U, "u", "bar_u"),
        (Panel(180, 50, 150, 150, (0, 1), (0, 1)), V, "v", "bar_v"))
        lo, hi = extrema(Z)
        rect_cells!(io, p, xs, ys, Z, lo, hi)
        grid_lines!(io, p, xs, ys)
        frame!(io, p)
        text!(io, p.x0 + p.w / 2, p.y0 - 12, name)
        colorbar_below!(io, p, id, lo, hi)
    end
    return close_svg(io)
end

# --- 6. a displaced elastic grid -----------------------------------------------------------

function elasticity_figure()
    λ, μ = 2.0, 0.5
    c = λ / (λ + 2μ)
    uᵉ = (x -> sin(x[1] + 0.3) * cos(x[2]) + exp(x[1]) * x[2]^2 / 2,
        x -> -c * cos(x[1] + 0.3) * sin(x[2]) + cos(2x[1]) * x[2]^2)
    hess(f, x) = ForwardDiff.hessian(f, collect(Float64, x))
    grad(f, x) = ForwardDiff.gradient(f, collect(Float64, x))
    second(u, a, b) = a == b ? (a == 1 ? Bramble.D̃ₓ(Bramble.D₋ₓ(u)) : Bramble.D̃ᵧ(Bramble.D₋ᵧ(u))) :
                      Bramble.Dcₓ(Bramble.Dcᵧ(u))
    function elasticity(u, v)
        row(i) = innerₕ(-μ * Δₕ(u(i)), v(i)) +
                 sum(innerₕ(-(λ + μ) * second(u(j), i, j), v(i)) for j in 1:2)
        return row(1) + row(2)
    end
    load(i) = x -> -μ * sum(hess(uᵉ[i], x)[a, a] for a in 1:2) -
                   (λ + μ) * sum(hess(uᵉ[j], x)[i, j] for j in 1:2)
    loadform(v) = innerₕ(load(1), v(1)) + innerₕ(load(2), v(2))
    K(i, j) = [λ * (i == d) * (j == e) + μ * ((i == j) * (d == e) + (i == e) * (j == d))
               for d in 1:2, e in 1:2]
    traction(u, i) = ∂ₙ(u(1); K = K(i, 1)) + ∂ₙ(u(2); K = K(i, 2))
    axis = (xmin = 1, xmax = 1, ymin = 2, ymax = 2)
    outward = (xmin = -1.0, xmax = 1.0, ymin = -1.0, ymax = 1.0)
    normal(F) = ntuple(d -> d == axis[F] ? outward[F] : 0.0, 2)
    tractiondata(F, i) = x -> sum(normal(F)[d] * K(i, j)[d, e] * grad(uᵉ[j], x)[e]
    for j in 1:2, d in 1:2, e in 1:2)
    relations(u) = (
        :xmin => traction(u, 1) => tractiondata(:xmin, 1),
        :xmin => traction(u, 2) => tractiondata(:xmin, 2),
        :ymin => traction(u, 1) => x -> 0.0,
        :ymin => traction(u, 2) => x -> 0.0,
        (F => u(i) => uᵉ[i] for F in (:xmax, :ymax) for i in 1:2)...,
        (F => ∂ₙₙ(u(i)) => (x -> hess(uᵉ[i], x)[axis[F], axis[F]]) for F in (:xmax, :ymax) for i in 1:2)...)

    grade(ξ) = expm1(2ξ) / expm1(2.0)
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (17, 17), (true, true))
    change_points!(Ωₕ, (grade.(points(Ωₕ)[1]), 1 .- grade.(1 .- points(Ωₕ)[2])))
    Cₕ = constrained(gridspace(Ωₕ; ghosts = :boundary)^Val(2), relations)
    A, F = assemble(form(Cₕ, Cₕ, elasticity), form(Cₕ, loadform))
    nx, ny = size(Ωₕ)
    P = reshape(parent(Bramble.lift(Cₕ, pde_solve(A, F))), nx + 2, ny + 2, 2)
    D = P[2:(nx + 1), 2:(ny + 1), :]                    # drop the ghost layer
    xs, ys = collect(points(Ωₕ(1))), collect(points(Ωₕ(2)))
    all(maximum(abs, D[:, :, i] .- [uᵉ[i]((x, y)) for x in xs, y in ys]) < 0.1 for i in 1:2) ||
        error("the elasticity solution is off")

    # Every node drawn at x + s u(x), s = 0.2 (the displacement reaches about 2).
    s = 0.2
    X = [xs[i] + s * D[i, j, 1] for i in 1:nx, j in 1:ny]
    Y = [ys[j] + s * D[i, j, 2] for i in 1:nx, j in 1:ny]
    mag = [hypot(D[i, j, 1], D[i, j, 2]) for i in 1:nx, j in 1:ny]
    xlim = (min(0, minimum(X)), max(1, maximum(X)))
    ylim = (min(0, minimum(Y)), max(1, maximum(Y)))
    side = max(xlim[2] - xlim[1], ylim[2] - ylim[1])
    xlim = (xlim[1], xlim[1] + side)
    ylim = (ylim[1], ylim[1] + side)
    p = Panel(20, 15, 240, 240, xlim, ylim)
    io = open_svg("Displaced 17 by 17 graded grid of a two-dimensional elastic body, scaled by 0.2")
    grid_lines!(io, p, xs, ys; class = "ref")
    lo, hi = extrema(mag)
    polys = [((px(p, X[i, j]), py(p, Y[i, j])), (px(p, X[i + 1, j]), py(p, Y[i + 1, j])),
                 (px(p, X[i + 1, j + 1]), py(p, Y[i + 1, j + 1])), (px(p, X[i, j + 1]), py(p, Y[i, j + 1])))
             for i in 1:(nx - 1), j in 1:(ny - 1)]
    cells!(io, vec(polys), vec(corner_means(mag)), lo, hi)
    print(io, "<path class=\"mesh\" d=\"")
    for j in 1:ny
        print(io, "M")
        for i in 1:nx
            @printf(io, "%.1f %.1f ", px(p, X[i, j]), py(p, Y[i, j]))
            i < nx && print(io, "L")
        end
    end
    for i in 1:nx
        print(io, "M")
        for j in 1:ny
            @printf(io, "%.1f %.1f ", px(p, X[i, j]), py(p, Y[i, j]))
            j < ny && print(io, "L")
        end
    end
    println(io, "\"/>")
    colorbar!(io, 276, p.y0, p.h, lo, hi)
    return close_svg(io)
end

# --- all six -------------------------------------------------------------------------------

"""
    gallery_figures(dir)

Write `gallery_poisson.svg`, `gallery_disk.svg`, `gallery_layer.svg`, `gallery_heat.svg`,
`gallery_coupled.svg` and `gallery_elasticity.svg` into `dir` and return their paths.
"""
function gallery_figures(dir)
    figures = (poisson = poisson_figure, disk = disk_figure, layer = layer_figure,
        heat = heat_figure, coupled = coupled_figure, elasticity = elasticity_figure)
    return map(keys(figures)) do name
        path = joinpath(dir, "gallery_$name.svg")
        write(path, figures[name]())
        path
    end
end

end # module

using .GalleryFigures

if abspath(PROGRAM_FILE) == @__FILE__
    for path in gallery_figures(isempty(ARGS) ? joinpath(@__DIR__, "src", "assets") : ARGS[1])
        println("wrote ", path)
    end
end
