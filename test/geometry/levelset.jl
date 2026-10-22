module GeometryLevelSetTests

using Test
using Bramble
using Bramble: LevelSet, _crossing, conditions, regions

circle(p) = hypot(p[1], p[2]) - 1
cubic(p) = p[1]^3 + p[2] - 0.3
const X2 = interval(-1.25, 1.25) × interval(-1.25, 1.25)

@testset "LevelSet" begin
    # Invariants tested:
    # 1. `levelset` is a marker value, not a predicate, and joins `regions` with three labels.
    # 2. `_crossing` returns the root on the segment to rounding, in either node order, for a
    #    signed distance and for a function that is not one, in Float32 too.
    # 3. Invalid use throws an ArgumentError: same signs, reserved or clashing label, not 2D,
    #    a Bool predicate.
    L = levelset(circle)

    @testset "marker value" begin
        @test L isa LevelSet
        @test !(L isa Function)
        Ω = domain(X2, :D => L)
        @test length(regions(markers(Ω))) == 1
        @test length(conditions(Ω)) == 0
        @test labels(Ω) == (:D, :D_boundary, :D_exterior)
        @test L((0.0, 0.5)) == -0.5
        @test occursin("levelset", sprint(show, MIME"text/plain"(), Ω))
        @test occursin("<levelset>", sprint(show, MIME"text/plain"(), markers(Ω)))
        # A time set leaves φ taking x alone.
        T = interval(0.0, 1.0)
        @test labels(domain(X2, T, :D => L)) == (:D, :D_boundary, :D_exterior)
    end

    @testset "crossing" begin
        @test _crossing(L, (0.0, 0.0), (0.8, 0.9)) ≈ 1 / hypot(0.8, 0.9) atol = 1e-14
        @test _crossing(L, (2.0, 0.0), (0.0, 0.0)) ≈ 0.5 atol = 1e-14
        @test _crossing(levelset(cubic), (0.0, 0.0), (1.0, 0.0)) ≈ cbrt(0.3) atol = 1e-14
        t = _crossing(L, (0.0, 0.0), (0.8, 0.9))
        @test 1 - _crossing(L, (0.8, 0.9), (0.0, 0.0)) ≈ t atol = 1e-14
        # A root at an endpoint.
        @test _crossing(L, (1.0, 0.0), (2.0, 0.0)) == 0.0
        @test _crossing(L, (0.0, 0.0), (1.0, 0.0)) == 1.0
        t32 = _crossing(L, (0.0f0, 0.0f0), (2.0f0, 0.0f0))
        @test t32 isa Float32
        @test t32 ≈ 0.5f0 atol = 2.0f-7
        @test_throws ArgumentError _crossing(L, (0.0, 0.0), (0.5, 0.0))
        @test_throws ArgumentError _crossing(L, (2.0, 0.0), (0.0, 2.0))
    end

    @testset "validation" begin
        @test_throws ArgumentError domain(X2, :boundary => L)
        @test_throws ArgumentError domain(X2, :interior => L)
        @test_throws ArgumentError domain(interval(0.0, 1.0), :D => L)
        @test_throws ArgumentError domain(X2, :D => L, :D_boundary => :left)
        @test_throws ArgumentError domain(X2, :D => L, :D_exterior => levelset(circle))
        @test_throws ArgumentError domain(X2, :D => levelset(p -> p[1] > 0))
        @test_throws ArgumentError domain(X2, :D => levelset(p -> "a"))
        @test_throws ArgumentError domain(X2, :D => levelset(p -> p[3]))
    end
end

end
