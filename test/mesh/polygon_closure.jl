module MeshPolygonClosureTests

using Test
using Random
using Bramble

# `mesh(Ω, npts, unif)` on a domain with polygon markers closes its grid lines: every
# crossing of a grid line with an oblique edge is a grid node (the Geom condition of
# Ferreira & Grigorieff). Each check reads the grid lines and recomputes the crossings here,
# independently of the library's closure.

# Figure 1 of Ferreira & Grigorieff (2006): every oblique edge has slope ±1.
const V = [(0.0, 2.0), (2.0, 0.0), (4.0, 0.0), (6.0, 2.0), (6.0, 3.0), (4.0, 3.0), (4.0, 5.0),
    (2.0, 5.0), (0.0, 3.0)]
const X = interval(0.0, 6.0) × interval(0.0, 5.0)

function edges(W)
    n = length(W)
    return [(W[k], W[mod1(k + 1, n)]) for k in 1:n]
end

# The number of crossings of the lines `xs`, `ys` with the oblique edges of `W` that lie
# farther than `tol` from every line of the other axis.
function off_node_crossings(W, xs, ys, tol)
    off = 0
    for (a, b) in edges(W)
        (a[1] == b[1] || a[2] == b[2]) && continue
        for x in xs
            min(a[1], b[1]) < x < max(a[1], b[1]) || continue
            y = a[2] + (b[2] - a[2]) * (x - a[1]) / (b[1] - a[1])
            any(t -> abs(t - y) <= tol, ys) || (off += 1)
        end
        for y in ys
            min(a[2], b[2]) < y < max(a[2], b[2]) || continue
            x = a[1] + (b[1] - a[1]) * (y - a[2]) / (b[2] - a[2])
            any(t -> abs(t - x) <= tol, xs) || (off += 1)
        end
    end
    return off
end

vertices_on_lines(W, xs, ys) = all(v -> v[1] in xs && v[2] in ys, W)

lines(Ωₕ) = (collect(points(Ωₕ(1))), collect(points(Ωₕ(2))))

function segment_distance(p, a, b)
    v, w = b .- a, p .- a
    t = clamp((w[1] * v[1] + w[2] * v[2]) / (v[1]^2 + v[2]^2), 0, 1)
    return hypot(w[1] - t * v[1], w[2] - t * v[2])
end

# A mesh over the set `Y` of `Ω` on the lines `xs`, `ys`, labelled by the markers of `Ω`.
function mesh_on(Y, Ω, xs, ys)
    Ωₕ = mesh(domain(Y), (length(xs), length(ys)), (true, true))
    Bramble.change_points!(Ωₕ, markers(Ω), (xs, ys))
    return Ωₕ
end

@testset "Polygon grid closure" begin
    P = Bramble.polygon(V)
    Ω = domain(X, :P => P)

    @testset "Random seeds close, 60 draws" begin
        for s in 1:60
            npts = (5 + s % 7, 4 + s % 5)
            Random.seed!(s)
            xs, ys = lines(mesh(Ω, npts, (false, false)))
            Random.seed!(s)
            sx, sy = lines(mesh(X, npts, (false, false)))
            @test vertices_on_lines(V, xs, ys)
            @test off_node_crossings(V, xs, ys, 1e-12) == 0
            @test issorted(xs) && issorted(ys)
            @test (xs[1], xs[end], ys[1], ys[end]) == (0.0, 6.0, 0.0, 5.0)
            # A seed is kept unless it lies within the merge tolerance of another line.
            @test all(x -> any(t -> abs(t - x) <= 6e-10, xs), sx)
            @test all(y -> any(t -> abs(t - y) <= 6e-10, ys), sy)
            @test length(xs) > npts[1] && length(ys) > npts[2]
        end
    end

    @testset "Uniform seeds" begin
        # Integer lines meet the slope ±1 edges at integers: nothing to add.
        xs, ys = lines(mesh(Ω, (7, 6), (true, true)))
        @test (xs, ys) == (collect(0.0:6.0), collect(0.0:5.0))
        # x = 1.5 and 4.5 meet the oblique edges at y = 0.5 and 4.5; y = 1 meets them at
        # x = 1 and 5: four new lines, and the vertex lines x = 2 and 4.
        xs, ys = lines(mesh(Ω, (5, 6), (true, true)))
        @test xs == [0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0]
        @test ys == [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0]
        @test vertices_on_lines(V, xs, ys)
        @test off_node_crossings(V, xs, ys, 1e-12) == 0
    end

    @testset "Mixed slopes throw" begin
        kite = Bramble.polygon([(0.0, 0.0), (3.0, 1.0), (4.0, 4.0), (1.0, 3.0)])
        Ωk = domain(interval(0.0, 4.0) × interval(0.0, 4.0), :K => kite)
        @test_throws ArgumentError mesh(Ωk, (6, 6), (false, false))
        err = try
            Bramble._grid_closure(kite, [0.0, 1.5, 4.0], [0.0, 4.0])
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("slopes", err.msg) && occursin("3.0", err.msg)
    end

    @testset "Sub-tolerance detail throws" begin
        # The closure line y = 0.5 (from the vertex line x = 0.5 on the edge y = x) lies δ
        # from the vertex line y = 0.5 + δ, inside the merge tolerance 4e-10.
        for δ in (1e-10, 3e-10)
            W = [(0.0, 0.0), (4.0, 0.0), (4.0, 0.5 + δ), (3.0, 0.5 + δ), (3.0, 3.0),
                (0.5, 3.0), (0.5, 2.0), (1.0, 2.0), (1.0, 1.0)]
            Ωw = domain(interval(0.0, 4.0) × interval(0.0, 3.0), :P => Bramble.polygon(W))
            @test_throws "finer than the grid closure" mesh(Ωw, (5, 4), (true, true))
            @test_throws ArgumentError Bramble._grid_closure(Bramble.polygon(W),
                [0.0, 4.0], [0.0, 3.0])
        end
    end

    @testset "Rounded lines are one line" begin
        # Scaled and shifted copies of a shape whose vertex lines cross its slope-1 edge:
        # each crossing is computed with rounding, never a reason to throw.
        W = [(0.0, 0.0), (2.0, 0.0), (2.0, 0.9), (1.5, 0.9), (1.5, 1.4), (1.0, 1.4),
            (0.0, 0.4)]
        rng = Xoshiro(1)
        for T in (Float64, Float32), _ in 1:100

            c, o = 10 * rand(rng) + 0.01, 7 * rand(rng)
            Wc = [(T(c * a + o), T(c * b + o)) for (a, b) in W]
            xs, ys = Bramble._grid_closure(Bramble.polygon(Wc), T[o - 1, 2c + o + 1],
                T[o - 1, 1.4c + o + 1])
            @test vertices_on_lines(Wc, xs, ys)
            @test off_node_crossings(Wc, xs, ys, 64 * eps(T) * (2c + o)) == 0
        end
    end

    @testset "Unbounded closure hits the cap" begin
        # Every oblique edge has slope ±1 exactly, yet the closure needs 4097 × 3219 lines.
        Z = [(0.0, 1.0), (0.2939453125, 0.7060546875), (1.396484375, 1.80859375),
            (2.794921875, 0.41015625), (4.0, 1.615234375), (4.0, 8.837890625),
            (3.66015625, 9.177734375), (2.5126953125, 8.0302734375),
            (0.771484375, 9.771484375), (0.0, 9.0)]
        Ωz = domain(interval(0.0, 4.0) × interval(0.0, 20.0), :Z => Bramble.polygon(Z))
        @test_throws "unbounded or too fine" mesh(Ωz, (9, 41), (true, true))
        @test_throws ArgumentError Bramble._grid_closure(Bramble.polygon(Z), [0.0, 4.0],
            [0.0, 20.0])
    end

    @testset "Vertices outrank seeds" begin
        cx, cy = Bramble._grid_closure(P, [0.0, 2.0 + 1e-13, 6.0], [0.0, 5.0])
        @test 2.0 in cx && !(2.0 + 1e-13 in cx)
        @test vertices_on_lines(V, cx, cy)
    end

    @testset "Closure lines outrank seeds" begin
        # The seed 0.3 makes 5.7 a closure line; the seed 5.7 + 5e-11 gives way to it.
        cx, cy = Bramble._grid_closure(P, [0.0, 0.3, 5.7 + 5e-11, 6.0], [0.0, 5.0])
        @test !(5.7 + 5e-11 in cx)
        @test count(x -> abs(x - 5.7) < 1e-9, cx) == 1
        @test off_node_crossings(V, cx, cy, 1e-14) == 0
        @test vertices_on_lines(V, cx, cy)
    end

    @testset "Two polygons close together" begin
        D = [(3.0, 0.5), (3.5, 1.0), (3.0, 1.5), (2.5, 1.0)]
        Ω2 = domain(X, :P => P, :D => Bramble.polygon(D))
        Random.seed!(3)
        xs, ys = lines(mesh(Ω2, (9, 8), (false, false)))
        @test vertices_on_lines(V, xs, ys) && vertices_on_lines(D, xs, ys)
        @test off_node_crossings(V, xs, ys, 1e-12) == 0
        @test off_node_crossings(D, xs, ys, 1e-12) == 0
    end

    @testset "Float32 labels match Float64" begin
        # Seeds whose closure lines round in Float32: the closure merges the rounding.
        seeds = ([0.0f0, 0.3f0, 1.1f0, 2.5f0, 6.0f0], [0.0f0, 0.45f0, 2.6f0, 5.0f0])
        x32, y32 = Bramble._grid_closure(P, seeds...)
        x64, y64 = Bramble._grid_closure(P, map(s -> Float64.(s), seeds)...)
        @test eltype(x32) === Float32
        @test (length(x32), length(y32)) == (length(x64), length(y64))
        @test length(x32) > 7
        @test x32 ≈ x64 && y32 ≈ y64
        W32 = [Float32.(v) for v in V]
        X32 = interval(0.0f0, 6.0f0) × interval(0.0f0, 5.0f0)
        Ω32 = domain(X32, :P => Bramble.polygon(W32))
        M32, M64 = mesh_on(X32, Ω32, x32, y32), mesh_on(X, Ω, x64, y64)
        for L in (:P, :P_boundary, :P_exterior)
            @test Bramble.markers(M32)[L] == Bramble.markers(M64)[L]
        end
        Random.seed!(5)
        xr, yr = lines(mesh(Ω32, (9, 8), (false, false)))
        @test eltype(xr) === Float32 && minimum(diff(xr)) > 6.0f-5
        @test off_node_crossings(W32, xr, yr, 1.0f-5) == 0
    end

    @testset "Closure-made sliver keeps Γ nodes" begin
        # The seed 5.7 + 1e-9 sits outside the merge tolerance of the closure line 5.7, so
        # the closure draws the y-lines 3.3 and 3.3 + 1e-9: a sliver 1e-9 wide.
        xs, ys = Bramble._grid_closure(P, [0.0, 0.3, 1.1, 2.5, 4.6, 5.7 + 1e-9, 6.0],
            [0.0, 0.45, 1.2, 2.6, 3.9, 4.4, 5.0])
        @test minimum([diff(xs); diff(ys)]) < 2e-9
        @test off_node_crossings(V, xs, ys, 1e-12) == 0
        Ωₕ = mesh_on(X, Ω, xs, ys)
        on_edge = [minimum(e -> segment_distance((x, y), e...), edges(V)) <= 1e-12
                   for x in xs, y in ys]
        Γ = Bramble.markers(Ωₕ)[:P_boundary]
        @test count(on_edge) == 48
        @test Γ == vec(on_edge)
    end

    @testset "Refinement keeps the grid fitted" begin
        Random.seed!(7)
        Ωₕ = mesh(Ω, (9, 8), (false, false))
        Bramble.iterative_refinement!(Ωₕ, markers(Ω))
        xs, ys = lines(Ωₕ)
        @test off_node_crossings(V, xs, ys, 1e-12) == 0
        cx, cy = Bramble._grid_closure(P, xs, ys)
        @test (length(cx), length(cy)) == (length(xs), length(ys))
    end
end

end # module
