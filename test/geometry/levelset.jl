module GeometryLevelSetTests

using Test
using Bramble
using Bramble: LevelSet, _crossings, conditions, regions

circle(p) = hypot(p[1], p[2]) - 1
const X2 = interval(-1.25, 1.25) × interval(-1.25, 1.25)

@testset "LevelSet" begin
    # Invariants tested:
    # 1. `levelset` is a marker value, not a predicate, and joins `regions` with three labels.
    # 2. `_crossings` returns every root on a grid line, sorted. It finds two on a circle in
    #    either axis and a near-tangent pair, none off the curve, and four on a wavy curve.
    # 3. Invalid use throws an ArgumentError (reserved or clashing label, not 2D, a Bool
    #    predicate).
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

    @testset "crossings" begin
        r = _crossings(circle, 1, 0.6, -2.0, 2.0)
        @test length(r) == 2
        @test maximum(abs.(r .- [-0.8, 0.8])) < 1e-14
        q = _crossings(circle, 2, -0.6, -2.0, 2.0)
        @test length(q) == 2
        @test maximum(abs.(q .- [-0.8, 0.8])) < 1e-14
        # Near tangency: the dip below zero is narrower than the sample spacing.
        near = _crossings(circle, 1, 1 - 1e-12, -2.0, 2.0; M = 8)
        @test length(near) == 2
        @test all(abs.(near) .< 1e-5) && near[1] < 0 < near[2]
        @test circle((1 - 1e-12, near[1])) ≈ 0 atol = 1e-15
        @test isempty(_crossings(circle, 1, 1.5, -2.0, 2.0))
        @test isempty(_crossings(circle, 1, 1 + 1e-9, -2.0, 2.0; M = 8))
        # A line entirely inside has no root.
        @test isempty(_crossings(p -> hypot(p[1], p[2]) - 5, 1, 0.0, -2.0, 2.0))
        # Four roots, sorted, on a φ that is not a signed distance.
        wave(p) = (p[2] - 1)^2 * (p[2] + 1)^2 * 5 - 1 + 0 * p[1]
        w = _crossings(wave, 1, 0.3, -2.0, 2.0)
        @test length(w) == 4
        @test issorted(w)
        @test all(abs.(wave.(tuple.(0.3, w))) .< 1e-12)
        # Float32 input gives Float32 roots.
        r32 = _crossings(circle, 1, 0.6f0, -2.0f0, 2.0f0)
        @test eltype(r32) == Float32
        @test r32 ≈ Float32[-0.8, 0.8] atol = 2.0f-7
        @test_throws ArgumentError _crossings(circle, 3, 0.0, -2.0, 2.0)
        @test_throws ArgumentError _crossings(circle, 1, 0.0, -2.0, 2.0; M = 2)
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
