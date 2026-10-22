module PolygonMarkersTests

using Test
using Bramble
using ..TestUtils: @test_allocs

# A marker `:L => polygon(V)` splits the nodes of a 2D mesh into `L` (inside the polygon),
# `L_boundary` and `L_exterior`. Every classification here is checked node by node against
# an exact oracle in rational arithmetic, on points that are exact binary fractions, so a
# node on an edge is on it exactly and no tolerance enters the oracle.

const V = [(0.0, 2.0), (2.0, 0.0), (4.0, 0.0), (6.0, 2.0), (6.0, 3.0), (4.0, 3.0), (4.0, 5.0),
    (2.0, 5.0), (0.0, 3.0)]
const X = interval(0.0, 6.0) × interval(0.0, 5.0)

# Exact class of point `p` for the closed vertex chain `W`: -1 inside, 0 on, 1 outside.
function exact_class(W, p)
    Q = [(Rational{BigInt}(v[1]), Rational{BigInt}(v[2])) for v in W]
    x, y = Rational{BigInt}(p[1]), Rational{BigInt}(p[2])
    n = length(Q)
    inside = false
    for k in 1:n
        a, b = Q[k], Q[mod1(k + 1, n)]
        cross = (b[1] - a[1]) * (y - a[2]) - (b[2] - a[2]) * (x - a[1])
        if iszero(cross) && min(a[1], b[1]) <= x <= max(a[1], b[1]) &&
           min(a[2], b[2]) <= y <= max(a[2], b[2])
            return 0
        end
        if (a[2] > y) != (b[2] > y)
            x < a[1] + (y - a[2]) * (b[1] - a[1]) / (b[2] - a[2]) && (inside = !inside)
        end
    end
    return inside ? -1 : 1
end

# The three BitVectors the oracle expects on the points of `Ωₕ`, in linear node order.
function exact_labels(W, Ωₕ)
    xs, ys = collect(points(Ωₕ(1))), collect(points(Ωₕ(2)))
    cls = [exact_class(W, (x, y)) for x in xs, y in ys]
    return BitVector(vec(cls .== -1)), BitVector(vec(cls .== 0)), BitVector(vec(cls .== 1))
end

marker_triple(Ωₕ, L) = map(l -> Bramble.markers(Ωₕ)[l], Bramble._polygon_labels(L))

@testset "Polygon node classification" begin
    @testset "Uniform grid matches exact oracle" begin
        Ω = domain(X, :P => Bramble.polygon(V))
        Ωₕ = mesh(Ω, (7, 6), (true, true))
        Pi, Pb, Pe = marker_triple(Ωₕ, :P)
        @test (Pi, Pb, Pe) == exact_labels(V, Ωₕ)
        @test map(count, (Pi, Pb, Pe)) == (13, 16, 13)
        @test all(Pi .+ Pb .+ Pe .== 1)
        @test count(Bramble.index_in_marker(Ωₕ, :P_boundary)) == 16
        @test typeof(Ωₕ) === typeof(mesh(domain(X), (7, 6), (true, true)))
    end

    @testset "Non-uniform grid matches oracle" begin
        # The mesh is built without the marker, whose grid closure would add nodes; the
        # marker then labels the non-uniform points given here.
        Ω = domain(X, :P => Bramble.polygon(V))
        Ωₕ = mesh(domain(X), (7, 6), (false, false))
        xs = [0.0, 0.5, 1.75, 2.0, 3.25, 4.0, 6.0]
        ys = [0.0, 1.0, 2.5, 3.0, 4.75, 5.0]
        Bramble.change_points!(Ωₕ, markers(Ω), (xs, ys))
        @test marker_triple(Ωₕ, :P) == exact_labels(V, Ωₕ)
        @test count(Bramble.markers(Ωₕ)[:P_boundary]) > 0
        @test count(Bramble.markers(Ωₕ)[:P]) > 0
    end

    @testset "Float32 grid matches oracle" begin
        W = [Float32.(v) for v in V]
        Ω = domain(interval(0.0f0, 6.0f0) × interval(0.0f0, 5.0f0), :P => Bramble.polygon(W))
        Ωₕ = mesh(Ω, (13, 11), (true, true))
        @test eltype(points(Ωₕ(1))) === Float32
        @test marker_triple(Ωₕ, :P) == exact_labels(W, Ωₕ)
    end

    @testset "Mixed Float32 and Float64" begin
        # Vertices at 0.1 and 0.9 are not binary fractions, so a node of the other type
        # meant to sit on an edge is a Float32 rounding (about 1.5e-9) off it: the
        # tolerance uses the coarser of the two epsilons and keeps those nodes.
        on_edge(i, j) = (i in (11, 91) && 11 <= j <= 91) || (j in (11, 91) && 11 <= i <= 91)
        expected = BitVector(vec([on_edge(i, j) for i in 1:101, j in 1:101]))
        sq = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
        for (S, W) in ((Float32, sq), (Float64, [Float32.(v) for v in sq]))
            Y = interval(zero(S), one(S)) × interval(zero(S), one(S))
            Ωₕ = mesh(domain(Y, :Q => Bramble.polygon(W)), (101, 101), (true, true))
            @test eltype(points(Ωₕ(1))) === S
            Qi, Qb, Qe = marker_triple(Ωₕ, :Q)
            @test Qb == expected
            @test count(Qb) == 320
            @test count(Qi) == 79^2
        end
    end

    @testset "Offset Float32 grid on Float64 edges" begin
        # On [1000, 1001]² a Float32 node is up to 3e-5 off the Float64 edge it is meant
        # to sit on, more than 10³ ε L for L = 0.2: the 4 ε M term keeps every such node,
        # and the 19² interior nodes show τ stays well below the spacing 0.01.
        sq = [(1000.1, 1000.1), (1000.3, 1000.1), (1000.3, 1000.3), (1000.1, 1000.3)]
        Y = interval(1000.0f0, 1001.0f0) × interval(1000.0f0, 1001.0f0)
        Ωₕ = mesh(domain(Y, :Q => Bramble.polygon(sq)), (101, 101), (true, true))
        xs = collect(points(Ωₕ(1)))
        @test eltype(xs) === Float32
        @test findall(==(Float32(1000.1)), xs) == [11]
        @test findall(==(Float32(1000.3)), xs) == [31]
        on_edge(i, j) = (i in (11, 31) && 11 <= j <= 31) || (j in (11, 31) && 11 <= i <= 31)
        inner(i, j) = 11 < i < 31 && 11 < j < 31
        Qi, Qb, Qe = marker_triple(Ωₕ, :Q)
        @test Qb == BitVector(vec([on_edge(i, j) for i in 1:101, j in 1:101]))
        @test Qi == BitVector(vec([inner(i, j) for i in 1:101, j in 1:101]))
        @test map(count, (Qi, Qb, Qe)) == (19^2, 80, 101^2 - 80 - 19^2)
    end

    @testset "Far grid corner keeps τ local" begin
        # The unit square [1, 2]² on a grid reaching R: the node (1.5, 1 + δ) is δ inside
        # the edge y = 1, thousands of ulps away. A τ scaled by the grid's corner R, not by
        # the vertices, would swallow it into the boundary.
        for (T, R, δ) in ((Float64, 1e8, 5e-8), (Float32, 1.0f4, 1.0f-3))
            sq = [(T(1), T(1)), (T(2), T(1)), (T(2), T(2)), (T(1), T(2))]
            Ω = domain(interval(T(0), R) × interval(T(0), R), :Q => Bramble.polygon(sq))
            xs, ys = T[0, 1, 1.5, 2, R], T[0, 1, 1 + δ, 1.5, 2, R]
            Ωₕ = mesh(domain(interval(T(0), R) × interval(T(0), R)), (5, 6), (true, true))
            Bramble.change_points!(Ωₕ, markers(Ω), (xs, ys))
            Qi, Qb, Qe = marker_triple(Ωₕ, :Q)
            k = LinearIndices((5, 6))[3, 3]
            @test Qi[k]
            @test !Qb[k]
            @test (Qi, Qb, Qe) == exact_labels(sq, Ωₕ)
        end
    end

    @testset "Refinement re-derives labels" begin
        Ω = domain(X, :P => Bramble.polygon(V))
        Ωₕ = mesh(Ω, (7, 6), (true, true))
        Bramble.iterative_refinement!(Ωₕ, markers(Ω))
        @test npoints(Ωₕ, Tuple) == (13, 11)
        @test marker_triple(Ωₕ, :P) == exact_labels(V, Ωₕ)
        @test map(count, marker_triple(Ωₕ, :P)) == (65, 32, 46)
    end

    @testset "Moving points without markers" begin
        Ω = domain(X, :P => Bramble.polygon(V))
        Ωₕ = mesh(Ω, (7, 6), (true, true))
        pts = (collect(points(Ωₕ(1))), collect(points(Ωₕ(2))))
        @test :P_exterior in Bramble._predicate_labels(Ωₕ)
        @test_throws ArgumentError Bramble.change_points!(Ωₕ, pts)
        @test_throws ArgumentError Bramble.iterative_refinement!(Ωₕ)
    end

    @testset "Tolerance follows spacing and scale" begin
        for s in (1.0, 1e-9, 1e3)
            W = [s .* v for v in V]
            Ω = domain(interval(0.0, 6s) × interval(0.0, 5s), :P => Bramble.polygon(W))
            Ωₕ = mesh(Ω, (7, 6), (true, true))
            @test map(count, marker_triple(Ωₕ, :P)) == (13, 16, 13)
            ys = s .* collect(0.0:5.0)
            xs = s .* collect(0.0:6.0)
            # The node (s, s) is on the edge from (0, 2s) to (2s, 0): a rounding-sized move
            # keeps it there, a move of 1e-4 of the spacing takes it off.
            xs[2] = s * (1.0 + 1e-13)
            Bramble.change_points!(Ωₕ, markers(Ω), (xs, ys))
            @test count(Bramble.markers(Ωₕ)[:P_boundary]) == 16
            xs[2] = s * (1.0 + 1e-4)
            Bramble.change_points!(Ωₕ, markers(Ω), (xs, ys))
            @test map(count, marker_triple(Ωₕ, :P)) == (15, 14, 13)
        end
    end

    @testset "Sliver cell keeps boundary nodes" begin
        # A spacing of 1e-12 makes √ε hₘᵢₙ ≈ 1.5e-20; the nodes (1 + 1e-12, 1) and
        # (1 + 1e-12, 4), 7e-13 off the edges through (1, 1) and (1, 4), stay boundary nodes
        # through the scale term: the 16 of the uniform grid and these two.
        Ω = domain(X, :P => Bramble.polygon(V))
        Ωₕ = mesh(domain(X), (8, 6), (false, false))
        xs = [0.0, 1.0, 1.0 + 1e-12, 2.0, 3.0, 4.0, 5.0, 6.0]
        Bramble.change_points!(Ωₕ, markers(Ω), (xs, collect(0.0:5.0)))
        Pb = Bramble.markers(Ωₕ)[:P_boundary]
        lin = LinearIndices(npoints(Ωₕ, Tuple))
        @test Pb[lin[3, 2]]
        @test Pb[lin[3, 5]]
        @test Pb[lin[2, 2]]
        @test !Pb[lin[3, 3]]
        @test count(Pb) == 18
    end

    @testset "Two polygons and a predicate" begin
        tri = [(1.0, 1.0), (4.0, 1.0), (1.0, 4.0)]
        Ω = domain(X, :P => Bramble.polygon(V), :left => :left,
            :Q => Bramble.polygon(tri), :high => x -> x[2] > 3.5)
        Ωₕ = mesh(Ω, (7, 6), (true, true))
        @test marker_triple(Ωₕ, :P) == exact_labels(V, Ωₕ)
        @test marker_triple(Ωₕ, :Q) == exact_labels(tri, Ωₕ)
        @test count(Bramble.markers(Ωₕ)[:high]) == 14
        @test labels(markers(Ω)) ==
              (:left, :high, :P, :P_boundary, :P_exterior, :Q, :Q_boundary, :Q_exterior)
        @test length(markers(Ω)) == 4
        @test Set(Bramble._predicate_labels(Ωₕ)) ==
              Set([:high, :P, :P_boundary, :P_exterior, :Q, :Q_boundary, :Q_exterior])
    end

    @testset "Time-dependent domain" begin
        Ω = domain(X, interval(0.0, 1.0), :P => Bramble.polygon(V),
            :moving => (x, t) -> x[1] < 6t)
        Ωₕ = mesh(Ω(0.5), (7, 6), (true, true))
        @test marker_triple(Ωₕ, :P) == exact_labels(V, Ωₕ)
        @test count(Bramble.markers(Ωₕ)[:moving]) == 18
    end

    @testset "Domain accessors" begin
        P = Bramble.polygon(V)
        Ω = domain(X, :a => :left, :P => P)
        ids = Bramble.marker_identifiers(Ω)
        @test length(ids) == length(markers(Ω)) == 2
        @test ids[1] === :left && ids[2] === P
        @test labels(Ω) == (:a, :P, :P_boundary, :P_exterior)
        @test_allocs labels(Ω)
        text = sprint(show, MIME"text/plain"(), Ω)
        @test occursin("2 markers (1 symbol, 1 polygon)", text)
        @test occursin(":P_exterior", text)
        only_polygon = sprint(show, MIME"text/plain"(), domain(X, :P => P))
        @test occursin("1 marker (1 polygon)", only_polygon)
        @test !occursin("(none)", only_polygon)
    end

    @testset "Display" begin
        dm = markers(X, :P => Bramble.polygon(V))
        @test sprint(show, Bramble.regions(dm)[1]) == "Marker(:P => <polygon>)"
        text = sprint(show, MIME"text/plain"(), dm)
        @test occursin("Polygon markers (1)", text)
        @test occursin("9 vertices", text)
        @test !isempty(dm)
    end
end

@testset "Polygon marker validation" begin
    P = Bramble.polygon(V)
    # Each refusal names its own reason, so a throw for another cause does not pass.
    function refuses(f, reason)
        err = try
            f()
            nothing
        catch e
            e
        end
        return err isa ArgumentError && occursin(reason, err.msg)
    end
    @test refuses(() -> domain(interval(0.0, 6.0), :P => P), "set is 1D")
    Y = interval(0.0, 6.0) × interval(0.0, 5.0) × interval(0.0, 1.0)
    @test refuses(() -> domain(Y, :P => P), "set is 3D")
    S = interval(0.0, 1.0) × interval(0.0, 1.0)
    big = Bramble.polygon([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)])
    @test refuses(() -> domain(S, :P => big), "vertex 2 at (2.0, 0.0) lies outside")
    @test refuses(() -> domain(X, :boundary => P), ":boundary is reserved")
    @test refuses(() -> domain(X, :interior => P), ":interior is reserved")
    clash = "is also the label of another marker"
    @test refuses(() -> domain(X, :P => P, :P_boundary => :left), ":P_boundary " * clash)
    @test refuses(() -> domain(X, :P_exterior => x -> true, :P => P), ":P_exterior " * clash)
    @test refuses(() -> domain(X, :P => P, :P => P), ":P " * clash)
    @test refuses(() -> domain(X, :P => P, :P_exterior => P), ":P_exterior " * clash)
    @test refuses(() -> domain(X, interval(0.0, 1.0), :P => P, :P => :left), ":P " * clash)
    @test_throws ArgumentError domain(X, :P => P, :P_boundary => :left)
    # On the set's edge is inside the set.
    edge = Bramble.polygon([(0.0, 0.0), (6.0, 0.0), (0.0, 5.0)])
    @test labels(domain(X, :E => edge)) == (:E, :E_boundary, :E_exterior)
end

end
