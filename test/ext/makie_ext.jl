module ExtMakieExtTests

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

# The refusals' exact messages, so a drifting or swapped message fails the test.
const COMPOSITE_MSG = "plotting a composite element directly has no single reading — " *
                      "plot each of its components(...) separately."
const DIM3_MSG = "plotting a 3D grid function directly is not implemented — only 1D " *
                 "and 2D. See export_vtk for a full 3D field."

@testset "BrambleMakieExt" begin
    @testset "1D: PointBased conversion" begin
        Ωₕ = Bramble.mesh(domain(interval(0.0, 1.0)), 9, true)
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, sin)

        (pts,) = Makie.convert_arguments(Makie.PointBased(), uₕ)
        @test pts isa Vector{<:Makie.Point2}
        @test [p[1] for p in pts] ≈ points(Ωₕ)
        @test [p[2] for p in pts] ≈ parent(uₕ)

        # `expand_dimensions` is the one Makie actually reaches first for a `VectorElement`
        # (see the extension's own comment on why `convert_arguments` alone is not enough):
        # it must hand back the two coordinate vectors `convert_arguments` expects, not a
        # single vector of points.
        x, y = Makie.expand_dimensions(Makie.PointBased(), uₕ)
        @test x == points(Ωₕ)
        @test y == parent(uₕ)
    end

    @testset "2D: CellGrid and VertexGrid" begin
        Ωₕ = Bramble.mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (true, true)
        )
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> x[1] + 10x[2])
        px, py = points(Ωₕ)

        for trait in (Makie.CellGrid(), Makie.VertexGrid())
            x, y, z = Makie.convert_arguments(trait, uₕ)
            @test x == px
            @test y == py
            @test z == reshape(uₕ)
        end
    end

    @testset "3D: not implemented" begin
        Ωₕ = Bramble.mesh(
            domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (3, 3, 3), (true, true, true)
        )
        uₕ = Rₕ(gridspace(Ωₕ), x -> 1.0)
        @test_throws ArgumentError(DIM3_MSG) Makie.convert_arguments(Makie.PointBased(), uₕ)
        @test_throws ArgumentError(DIM3_MSG) Makie.convert_arguments(Makie.CellGrid(), uₕ)
        # `lines(uₕ)` reaches `expand_dimensions` first
        @test_throws ArgumentError(DIM3_MSG) Makie.expand_dimensions(Makie.PointBased(), uₕ)
    end

    @testset "ghost space: physical values" begin
        Ω1 = Bramble.mesh(domain(interval(0.0, 1.0)), 9, false)
        u1 = element(gridspace(Ω1; ghosts = :boundary), collect(1.0:11.0))
        x, y = Makie.expand_dimensions(Makie.PointBased(), u1)
        @test x == points(Ω1)
        @test y == collect(1.0:9.0)
        (pts,) = Makie.convert_arguments(Makie.PointBased(), u1)
        @test length(pts) == 9
        @test pts[9] == Makie.Point2f(points(Ω1)[9], 9.0)

        Ω2 = Bramble.mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (6, 5), (false, false)
        )
        G2 = gridspace(Ω2; ghosts = :boundary)
        u2 = element(G2, collect(1.0:ndofs(G2)))
        for trait in (Makie.CellGrid(), Makie.VertexGrid())
            xs, ys, z = Makie.convert_arguments(trait, u2)
            @test xs == points(Ω2)[1]
            @test ys == points(Ω2)[2]
            @test size(z) == (6, 5)
            @test vec(z) == collect(1.0:30.0)
        end

        Ω3 = Bramble.mesh(
            domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (4, 4, 4), (false, false, false)
        )
        u3 = element(gridspace(Ω3; ghosts = :boundary), 1.0)
        @test_throws ArgumentError(DIM3_MSG) Makie.convert_arguments(Makie.PointBased(), u3)
        @test_throws ArgumentError(DIM3_MSG) Makie.convert_arguments(Makie.CellGrid(), u3)
        @test_throws ArgumentError(DIM3_MSG) Makie.expand_dimensions(Makie.PointBased(), u3)
    end

    @testset "ghost space: polygon, fixed, data" begin
        V = [(0.0, 2.0), (2.0, 0.0), (4.0, 0.0), (6.0, 2.0), (6.0, 3.0), (4.0, 3.0),
            (4.0, 5.0), (2.0, 5.0), (0.0, 3.0)]
        Ωp = Bramble.mesh(
            domain(interval(0.0, 6.0) × interval(0.0, 5.0), :P => polygon(V)),
            (10, 9), (false, false)
        )
        U, _ = Bramble.polygon_spaces(Ωp, :P, x -> 1.0)
        up = element(U, 1.0)
        xs, ys, z = Makie.convert_arguments(Makie.CellGrid(), up)
        np = Bramble.npoints(Ωp, Tuple)
        @test size(z) == np
        @test vec(z) == parent(Bramble._polygon_element(up))
        @test count(!iszero, z) == length(parent(up))

        Ω2 = Bramble.mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (7, 6), (false, false)
        )
        Cf = constrained(
            gridspace(Ω2; ghosts = (:xmin,)),
            u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => u => 0.0)
        )
        uc = element(Cf)
        uc .= 1.0
        @test_throws ArgumentError Makie.convert_arguments(Makie.CellGrid(), uc)
        Cd = constrained(
            gridspace(Ω2; ghosts = (:xmin,)), u -> (:xmin => ∂ₙ(u) => 1.0,)
        )
        ud = element(Cd)
        ud .= 1.0
        @test_throws ArgumentError Makie.convert_arguments(Makie.CellGrid(), ud)
    end

    @testset "composite: refused" begin
        Ωₕ = Bramble.mesh(domain(interval(0.0, 1.0)), 5, true)
        Vₕ = gridspace(Ωₕ, Val(2))
        uv = Rₕ(Vₕ, (x -> x, x -> x^2))
        @test_throws ArgumentError(COMPOSITE_MSG) Makie.convert_arguments(
            Makie.PointBased(), uv
        )
        @test_throws ArgumentError(COMPOSITE_MSG) Makie.convert_arguments(
            Makie.CellGrid(), uv
        )
        @test_throws ArgumentError(COMPOSITE_MSG) Makie.expand_dimensions(
            Makie.PointBased(), uv
        )

        for c in components(uv)
            (pts,) = Makie.convert_arguments(Makie.PointBased(), c)
            @test pts isa Vector{<:Makie.Point2}
        end
    end
end

end # module ExtMakieExtTests
