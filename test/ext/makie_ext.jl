using Test
using Bramble
using Makie

# BrambleMakieExt's conversion/dispatch, tested directly against `Makie.convert_arguments`/
# `Makie.expand_dimensions` rather than through a real `lines`/`heatmap` call: no renderer
# (GLMakie/CairoMakie) is needed to exercise the data-conversion boundary this extension
# actually owns, and none is loaded here.
#
# `Bramble.mesh` is qualified throughout: `using Bramble, Makie` together makes bare `mesh`
# ambiguous (both packages export a function by that name).

@testset "BrambleMakieExt" begin
    @testset "1D: PointBased convert_arguments and expand_dimensions" begin
        Ωₕ = Bramble.mesh(domain(interval(0.0, 1.0)), 9, true)
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, sin)

        (pts,) = Makie.convert_arguments(Makie.PointBased(), uₕ)
        @test pts isa Vector{<:Makie.Point2}
        @test [p[1] for p in pts] ≈ points(Ωₕ)
        @test [p[2] for p in pts] ≈ values(uₕ)

        # `expand_dimensions` is the one Makie actually reaches first for a `VectorElement`
        # (see the extension's own comment on why `convert_arguments` alone is not enough):
        # it must hand back the two coordinate vectors `convert_arguments` expects, not a
        # single vector of points.
        x, y = Makie.expand_dimensions(Makie.PointBased(), uₕ)
        @test x == points(Ωₕ)
        @test y == values(uₕ)
    end

    @testset "2D: CellGrid and VertexGrid convert_arguments" begin
        Ωₕ = Bramble.mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7),
            (true, true))
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> x[1] + 10x[2])
        px, py = points(Ωₕ)

        for trait in (Makie.CellGrid(), Makie.VertexGrid())
            x, y, z = Makie.convert_arguments(trait, uₕ)
            @test x == px
            @test y == py
            @test z == to_matrix(uₕ)
        end
    end

    @testset "3D: not implemented" begin
        Ωₕ = Bramble.mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (3, 3, 3),
            (true, true, true))
        uₕ = Rₕ(gridspace(Ωₕ), x -> 1.0)
        @test_throws ArgumentError Makie.convert_arguments(Makie.PointBased(), uₕ)
        @test_throws ArgumentError Makie.convert_arguments(Makie.CellGrid(), uₕ)
    end

    @testset "Composite: refused, points to components(...)" begin
        Ωₕ = Bramble.mesh(domain(interval(0.0, 1.0)), 5, true)
        Vₕ = gridspace(Ωₕ, Val(2))
        uv = Rₕ(Vₕ, (x -> x, x -> x^2))
        @test_throws ArgumentError Makie.convert_arguments(Makie.PointBased(), uv)
        @test_throws ArgumentError Makie.convert_arguments(Makie.CellGrid(), uv)

        for c in components(uv)
            (pts,) = Makie.convert_arguments(Makie.PointBased(), c)
            @test pts isa Vector{<:Makie.Point2}
        end
    end
end
