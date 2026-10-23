module MeshLevelSetGeomTests

using Test
using Random
using Bramble

# (Geom) on the mesh of a level-set domain (#834), checked node by node against φ by
# code that calls nothing in the package but `domain`, `mesh`, `markers` and the mesh's
# points: every node off the boundary has the sign of φ its label says; every boundary
# node lies within h²/4 of φ = 0 (|φ|/|∇φ|, the gradient by central differences), h the
# largest line spacing inside the boundary's extent; a cell holding interior and exterior
# nodes holds them at opposite corners, the other two on the boundary. The package's own
# post-check is not reused, so a fault in it cannot cancel here.
#
# Positive control: the same labels on the lines with one interior line moved by h/3
# fail.

const BOX_DISK = interval(-1.25, 1.25) × interval(-1.25, 1.25)
const BOX_EGG = interval(-1.2, 1.3) × interval(-1.25, 1.15)
const BOX_DENTED = interval(-1.35, 1.35) × interval(-1.35, 1.35)

disk(p) = hypot(p[1], p[2]) - 1

function ellipse(θ)
    c, s = cos(θ), sin(θ)
    return p -> (c * p[1] + s * p[2])^2 + ((-s * p[1] + c * p[2]) / 0.6)^2 - 1
end

function egg(θ)
    c, s = cos(θ), sin(θ)
    return function (p)
        x, y = p[1] - 0.1, p[2] + 0.05
        u, v = c * x + s * y, -s * x + c * y
        return u^2 + v^2 * (1 + 0.4u) / 0.36 - 1
    end
end

# r = 1 + 0.12 cos θ + 0.05 cos 2θ − 0.06 exp(−(Δθ/0.25)²), Δθ the angle to the
# diagonal direction 3π/4 (wrapped), the whole curve rotated by `tilt`.
function dented_egg(tilt)
    return function (p)
        θ = atan(p[2], p[1]) - tilt
        Δ = mod(θ - 3π / 4 + π, 2π) - π
        return hypot(p[1], p[2]) - (1 + 0.12 * cos(θ) + 0.05 * cos(2θ) - 0.06 * exp(-(Δ / 0.25)^2))
    end
end

# (name, φ, box): the disk and the egg at 0° take the closure path, the rest the orbit
# path.
const CURVES = (("disk", disk, BOX_DISK), ("egg 0°", egg(0.0), BOX_EGG),
    ("ellipse 30°", ellipse(π / 6), BOX_DISK), ("egg 35°", egg(deg2rad(35)), BOX_EGG),
    ("dented egg 10°", dented_egg(deg2rad(10)), BOX_DENTED))

# |φ| / |∇φ| at p, the gradient by central differences.
function distance(φ, p)
    x, y, δ = p[1], p[2], 1e-6
    gx = (φ((x + δ, y)) - φ((x - δ, y))) / 2δ
    gy = (φ((x, y + δ)) - φ((x, y - δ))) / 2δ
    return abs(φ((x, y))) / hypot(gx, gy)
end

# The largest spacing of the lines `L` within [lo, hi].
spacing(L, lo, hi) = maximum(diff(filter(t -> lo <= t <= hi, L)))

# The (Geom) failures of labels `S` (1 interior, 0 boundary, −1 exterior) on lines xs, ys,
# counted by kind; `h` the largest line spacing inside the boundary's extent.
function geom_failures(φ, xs, ys, S)
    B = findall(==(0), S)
    xl, xh = extrema(k -> xs[k[1]], B)
    yl, yh = extrema(k -> ys[k[2]], B)
    h = max(spacing(xs, xl, xh), spacing(ys, yl, yh))
    sign_bad = dist_bad = cell_bad = 0
    for j in eachindex(ys), i in eachindex(xs)

        p = (xs[i], ys[j])
        if S[i, j] == 0
            dist_bad += distance(φ, p) > 0.25h^2
        else
            sign_bad += S[i, j] == 1 ? !(φ(p) < 0) : !(φ(p) > 0)
        end
    end
    for j in 1:(length(ys) - 1), i in 1:(length(xs) - 1)

        a, b, c, d = S[i, j], S[i + 1, j], S[i + 1, j + 1], S[i, j + 1]
        corners = (a, b, c, d)
        (1 in corners && -1 in corners) || continue
        cell_bad += !((a * c == -1 && b == 0 && d == 0) || (b * d == -1 && a == 0 && c == 0))
    end
    return (; sign_bad, dist_bad, cell_bad, h)
end

# The mesh's lines and its labels on them.
function grid_labels(Ωₕ)
    xs, ys = collect(points(Ωₕ(1))), collect(points(Ωₕ(2)))
    m = markers(Ωₕ)
    shape = (length(xs), length(ys))
    I, Γ, E = reshape(m[:D], shape), reshape(m[:D_boundary], shape), reshape(m[:D_exterior], shape)
    @test all(I .+ Γ .+ E .== 1)
    return xs, ys, Int.(I) .- Int.(E)
end

# A line strictly inside the boundary's x-extent, 30% of the way across, holding two
# boundary nodes where the curve is far from parallel to the line.
function moved_lines(xs, S, h)
    cols = findall(i -> any(==(0), view(S, i, :)), eachindex(xs))
    i = cols[1] + round(Int, 0.3 * (cols[end] - cols[1]))
    moved = copy(xs)
    moved[i] += h / 3
    return moved
end

@testset "(Geom) against φ: $name" for (name, φ, box) in CURVES
    for n in (40, 80), seed in 1:3

        Random.seed!(seed)
        Ωₕ = mesh(domain(box, :D => levelset(φ)), (n, n), (false, false))
        xs, ys, S = grid_labels(Ωₕ)
        @test count(==(1), S) > 0 && count(==(0), S) > 0 && count(==(-1), S) > 0
        r = geom_failures(φ, xs, ys, S)
        @test r.sign_bad == 0
        @test r.dist_bad == 0
        @test r.cell_bad == 0
        # Positive control: the checker sees a line moved off the fitted position.
        c = geom_failures(φ, moved_lines(xs, S, r.h), ys, S)
        @test c.dist_bad > 0
    end
end

# Refinement and `change_points!` refuse a level-set mesh before touching it (#866, #891):
# the refusal used to come from the marking, after the points had changed and the marker
# vectors no longer matched them.
@testset "(Geom) refusals keep the mesh" begin
    Ω = domain(BOX_DISK, :D => levelset(disk))
    mk() = mesh(Ω, (20, 20), (false, false))
    marker_lengths(Ωₕ) = Dict(k => length(v) for (k, v) in pairs(markers(Ωₕ)))

    @testset "iterative_refinement!" begin
        Ωₕ = mk()
        n0 = npoints(Ωₕ, Tuple)
        p0 = map(copy, Tuple(Bramble.points(Ωₕ(i)) for i in 1:2))
        L0 = marker_lengths(Ωₕ)
        @test_throws ArgumentError iterative_refinement!(Ωₕ, markers(Ω))
        @test_throws "level-set marker :D" iterative_refinement!(Ωₕ, markers(Ω))
        @test npoints(Ωₕ, Tuple) == n0
        @test Tuple(Bramble.points(Ωₕ(i)) for i in 1:2) == p0
        @test marker_lengths(Ωₕ) == L0
        @test length(markers(Ωₕ)[:D]) == npoints(Ωₕ)
    end

    @testset "change_points!" begin
        Ωₕ = mk()
        n0 = npoints(Ωₕ, Tuple)
        p0 = map(copy, Tuple(Bramble.points(Ωₕ(i)) for i in 1:2))
        L0 = marker_lengths(Ωₕ)
        pts = Tuple(collect(range(-1.25, 1.25; length = n)) for n in n0)
        @test pts != p0
        @test_throws ArgumentError Bramble.change_points!(Ωₕ, markers(Ω), pts)
        @test_throws "level-set marker :D" Bramble.change_points!(Ωₕ, markers(Ω), pts)
        @test npoints(Ωₕ, Tuple) == n0
        @test Tuple(Bramble.points(Ωₕ(i)) for i in 1:2) == p0
        @test marker_lengths(Ωₕ) == L0
    end
end

end
