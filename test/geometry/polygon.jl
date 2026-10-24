module GeometryPolygonTests

using Test
using Bramble
using Bramble: Polygon, polygon, signed_distance
using ..TestUtils: @test_allocs

# The non-convex polygon of Ferreira & Grigorieff (2006), Figure 1: two oblique corners at the
# bottom, a reflex vertex at (4, 3).
const FIGURE1 = [(0.0, 2.0), (2.0, 0.0), (4.0, 0.0), (6.0, 2.0), (6.0, 3.0), (4.0, 3.0), (4.0, 5.0), (2.0, 5.0),
    (0.0, 3.0)]

function twice_area(V)
    sum(V[k][1] * V[mod1(k + 1, length(V))][2] - V[mod1(k + 1, length(V))][1] * V[k][2] for k in eachindex(V))
end

# Twice the area computed exactly on the stored floats, so that rounding cannot fake the sign.
exact_twice_area(V) = twice_area([(Rational{BigInt}(v[1]), Rational{BigInt}(v[2])) for v in V])

# The text of the `ArgumentError` that `f` throws; another exception type is rethrown.
function polygon_message(f)
    try
        f()
    catch e
        e isa ArgumentError || rethrow()
        return sprint(showerror, e)
    end
    return "no error"
end

# Signed distance to the unit square, written from its definition and sharing no code with
# the library: inside, minus the distance to the nearest side; outside, the norm of the excess.
function square_distance(x)
    ex, ey = max(-x[1], x[1] - 1, 0.0), max(-x[2], x[2] - 1, 0.0)
    (ex > 0 || ey > 0) && return hypot(ex, ey)
    return -min(x[1], 1 - x[1], x[2], 1 - x[2])
end

@testset "Polygon" begin
    # Invariants tested:
    # 1. `polygon` stores the vertices counter-clockwise, whatever the input orientation, and
    #    converts them to the promoted floating-point type; a small polygon far from the
    #    origin is accepted and stored counter-clockwise (exact area, not the raw shoelace).
    # 2. `signed_distance` is negative inside, zero on the boundary, positive outside, the
    #    same for either input orientation, exact against a hand-derived oracle on a
    #    non-convex polygon (reflex vertex, ray through vertices) and the unit square; a
    #    point exactly on an edge gives exactly zero, not a rounding residue.
    # 3. Invalid input throws ArgumentError: fewer than 3 vertices, non-finite coordinates,
    #    a zero-length edge, zero area, a bowtie, a touching vertex, an edge folding back
    #    (also when only exact arithmetic sees the vertices as collinear), a vertex on or
    #    across a non-adjacent edge that only the exact fallback of the filter detects.
    #    A collinear vertex in the middle of an edge is kept.
    # 4. `polygon` and `signed_distance` are inferred; `signed_distance` allocates 0 bytes.
    @testset "construction and orientation" begin
        P = polygon(FIGURE1)
        Q = polygon(reverse(FIGURE1))
        @test P isa Polygon{Float64}
        @test P.vertices == FIGURE1
        @test twice_area(FIGURE1) > 0
        @test Q.vertices == FIGURE1                     # reversed back to counter-clockwise

        Pi = polygon([(0, 0), (1, 0), (0, 1)])
        @test Pi isa Polygon{Float64}
        @test Pi.vertices == [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]

        P32 = polygon([(0.0f0, 0.0f0), (1.0f0, 0.0f0), (0.0f0, 1.0f0)])
        @test P32 isa Polygon{Float32}
        @test signed_distance(P32, (0.25f0, 0.25f0)) isa Float32
        @test signed_distance(P32, (0.25f0, 0.25f0)) ≈ -0.25f0

        # A collinear vertex mid-edge is a vertex, not an error.
        @test length(polygon([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]).vertices) == 5
    end

    @testset "translated far from the origin" begin
        # The raw shoelace sum cancels here: it rejected the Float32 square at 1e4 and stored
        # the one at 1e5 clockwise (likewise Float64 at 1e8 and 1e10).
        for (T, c) in ((Float32, 1.0f4), (Float32, 1.0f5), (Float64, 1e8), (Float64, 1e10))
            V = [(T(c), T(c)), (T(c) + 1, T(c)), (T(c) + 1, T(c) + 1), (T(c), T(c) + 1)]
            for W in (V, reverse(V))
                P = polygon(W)
                @test exact_twice_area(P.vertices) == 2
                @test signed_distance(P, (c + T(0.5), c + T(0.5))) == -T(0.5)
            end
        end
    end

    @testset "signed distance, non-convex polygon" begin
        for P in (polygon(FIGURE1), polygon(reverse(FIGURE1)))
            sd = x -> signed_distance(P, x)
            @test sd((3.0, 2.0)) ≈ -sqrt(2.0)          # ray along y = 2 passes the vertex (6, 2)
            @test sd((3.5, 2.5)) ≈ -sqrt(0.5)          # nearest point the reflex vertex (4, 3)
            @test sd((4.5, 3.5)) ≈ 0.5                 # outside, in the notch
            @test sd((5.0, 4.0)) ≈ 1.0
            @test sd((0.0, 0.0)) ≈ sqrt(2.0)
            @test sd((-1.0, 2.0)) ≈ 1.0                # ray along y = 2 through (0, 2) and (6, 2)
            @test sd((7.0, 2.0)) ≈ 1.0
            @test sd((3.0, 6.0)) ≈ 1.0
            for k in eachindex(FIGURE1)
                a, b = FIGURE1[k], FIGURE1[mod1(k + 1, length(FIGURE1))]
                @test abs(sd(a)) <= 1e-15
                @test abs(sd(((a[1] + b[1]) / 2, (a[2] + b[2]) / 2))) <= 1e-15
            end
        end
    end

    @testset "signed distance zero on an edge" begin
        @test signed_distance(polygon([(1, 8), (12, 8), (12, 9)]), (8.5, 8.0)) === 0.0
        Q = polygon([(0.0, 0.0), (3.0, 0.0), (0.0, 3.0)])
        @test all(x -> signed_distance(Q, x) === 0.0, ((1.0, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 1.0)))
        S = polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        @test all(x -> signed_distance(S, (x, 0.0)) === 0.0, 0.0:0.1:10.0)
        @test all(y -> signed_distance(S, (10.0, y)) === 0.0, 0.0:0.1:10.0)
    end

    @testset "signed distance, unit square" begin
        for V in ([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], [(1.0, 1.0), (1.0, 0.0), (0.0, 0.0), (0.0, 1.0)])
            P = polygon(V)
            for x in range(-0.5, 1.5; length = 17), y in range(-0.5, 1.5; length = 13)

                @test signed_distance(P, (x, y)) ≈ square_distance((x, y)) atol = 1e-15
            end
        end
        # A tiny copy is still a valid polygon, and its distances scale.
        @test signed_distance(polygon([(0.0, 0.0), (1e-9, 0.0), (1e-9, 1e-9), (0.0, 1e-9)]), (5e-10, 2e-10)) ≈ -2e-10
    end

    @testset "invalid polygons" begin
        @test_throws ArgumentError polygon(NTuple{2, Float64}[])
        @test_throws ArgumentError polygon([(0.0, 0.0), (1.0, 1.0)])
        @test_throws ArgumentError polygon([(0.0, 0.0), (NaN, 0.0), (0.0, 1.0)])
        @test_throws ArgumentError polygon([(0.0, 0.0), (Inf, 0.0), (0.0, 1.0)])
        # zero-length edges: a duplicate vertex, the first vertex repeated at the end
        @test_throws ArgumentError polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
        @test_throws ArgumentError polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)])
        # zero area: all collinear, exactly and with rounding
        @test_throws ArgumentError polygon([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
        @test_throws ArgumentError polygon([(0.0, 0.0), (0.1, 0.1), (0.3, 0.3)])
        # not simple: a bowtie, a vertex touching another edge, an edge folding back
        @test_throws ArgumentError polygon([(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)])
        @test_throws ArgumentError polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 0.0), (0.0, 4.0)])
        @test_throws ArgumentError polygon([(0.0, 0.0), (2.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
        @test_throws ArgumentError polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 4.0), (2.0, -1.0), (0.0, 4.0)])
        # Vertex 3 lies 1.7e-17 (exact orientation) to the right of edge 1, so edge 3 crosses
        # edge 1 just past it; the floating-point orientation, 1.1e-16, has the wrong sign.
        @test_throws ArgumentError polygon([(0.3 * x, 0.3 * y)
                                            for (x, y) in ((3, 3), (-1, 1), (1, 2), (2, -4), (-3, 0), (2, 3))])
        # Vertex 4 against edge 1 from (0.1, 0.1) to (0.7, 0.4), the inside below it. The
        # floating-point orientation lies within the filter's error bound, so only the exact
        # fallback decides. The three points, a few ulps apart, lie on the edge, above it and
        # below it; floats put them below, below and on the edge, and alone would accept the
        # first two.
        a, b = (0.1, 0.1), (0.7, 0.4)
        notch(c) = [a, b, (0.7, -0.5), c, (0.1, -0.5)]
        float_orient(c) = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])
        within_bound(c) = abs(float_orient(c)) <=
                          3 * eps() * (abs((b[1] - a[1]) * (c[2] - a[2])) +
                                       abs((b[2] - a[2]) * (c[1] - a[1])))
        on = (0.3, 0.2)
        above = (0.2999999999999998, 0.19999999999999993)
        below = (0.30000000000000004, 0.20000000000000004)
        @test all(within_bound, (on, above, below))
        @test float_orient(on) < 0 && float_orient(above) < 0 && iszero(float_orient(below))
        for c in (on, above)
            @test polygon_message(() -> polygon(notch(c))) ==
                  "ArgumentError: Invalid polygon: edges 1 and 3 cross or touch (the polygon is not simple)"
        end
        @test polygon(notch(below)).vertices == reverse(notch(below))
        # The messages name the vertex or edge and the value that failed.
        @test polygon_message(() -> polygon([(0.0, 0.0), (1.0, 0.0), (NaN, 1.0)])) ==
              "ArgumentError: Invalid polygon: vertex 3 at (NaN, 1.0) has a coordinate that is not finite"
        @test polygon_message(() -> polygon([(0.0, 0.0), (Inf, 0.0), (0.0, 1.0)])) ==
              "ArgumentError: Invalid polygon: vertex 2 at (Inf, 0.0) has a coordinate that is not finite"
        # The coincidence names the input values, not their Rational{BigInt} copies.
        @test polygon_message(() -> polygon([(0.0, 0.0), (0.1, 0.0), (0.1, 0.0), (0.0, 1.0)])) ==
              "ArgumentError: Invalid polygon: vertices 2 and 3 coincide at (0.1, 0.0) (zero-length edge)"
        @test polygon_message(() -> polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)])) ==
              "ArgumentError: Invalid polygon: vertices 4 and 1 coincide at (0.0, 0.0) (zero-length edge)"
        # Zero area gives |2A| and the tolerance 8nεL², here 8 · 3 · eps(Float64) · 2².
        @test polygon_message(() -> polygon([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])) ==
              "ArgumentError: Invalid polygon: the vertices enclose (nearly) zero area: " *
              "|2A| = 0.0, at most 8nεL² = 2.1316282072803006e-14"
        @test polygon_message(() -> polygon([(0.0, 0.0), (1.0, 0.0), (2.0, 1.0e-17)])) ==
              "ArgumentError: Invalid polygon: the vertices enclose (nearly) zero area: " *
              "|2A| = 1.0e-17, at most 8nεL² = 2.1316282072803006e-14"
    end

    @testset "inference and allocations" begin
        P = polygon(FIGURE1)
        @test (@inferred signed_distance(P, (3.0, 2.0))) ≈ -sqrt(2.0)
        @test_allocs signed_distance(P, (3.0, 2.0))
        @test (@inferred polygon([(0.0f0, 0.0f0), (1.0f0, 0.0f0), (0.0f0, 1.0f0)])) isa Polygon{Float32}
        @test (@inferred polygon([(0, 0), (1, 0), (0, 1)])) isa Polygon{Float64}
        @test (@inferred polygon(FIGURE1)) isa Polygon{Float64}
    end
end

end # module
