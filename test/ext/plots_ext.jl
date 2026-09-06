using Test
using Bramble
using RecipesBase

# BramplePlotsExt's recipe dispatch, tested directly against `RecipesBase.apply_recipe`
# rather than through a real `Plots.plot(...)` call: the extension's own trigger dependency
# is `RecipesBase` alone (Project.toml's `[extensions]` table), not the much heavier full
# `Plots.jl`, so calling into `RecipesBase` is what actually exercises the same boundary a
# user crossing it does, without paying for a renderer this extension never touches.

@testset "BramplePlotsExt" begin
    @testset "1D: line recipe" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, true)
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, sin)

        rd = only(RecipesBase.apply_recipe(Dict{Symbol, Any}(), uₕ))
        @test rd.plotattributes[:seriestype] == :line

        x, y = rd.args
        @test x == points(Ωₕ)
        @test y == values(uₕ)
    end

    @testset "2D: heatmap recipe, transposed to Plots' (row, col) = (y, x) convention" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (true, true))
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> x[1] + 10x[2])

        rd = only(RecipesBase.apply_recipe(Dict{Symbol, Any}(), uₕ))
        @test rd.plotattributes[:seriestype] == :heatmap

        x, y, z = rd.args
        px, py = points(Ωₕ)
        @test x == px
        @test y == py
        @test size(z) == (length(py), length(px))
        @test z == permutedims(to_matrix(uₕ))
    end

    @testset "3D: not implemented" begin
        Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (3, 3, 3),
            (true, true, true))
        uₕ = Rₕ(gridspace(Ωₕ), x -> 1.0)
        @test_throws ArgumentError RecipesBase.apply_recipe(Dict{Symbol, Any}(), uₕ)
    end

    @testset "Composite: refused, points to components(...)" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true)
        Vₕ = gridspace(Ωₕ, Val(2))
        uv = Rₕ(Vₕ, (x -> x, x -> x^2))
        @test_throws ArgumentError RecipesBase.apply_recipe(Dict{Symbol, Any}(), uv)

        # each component, on its own, is an ordinary 1D grid function
        for c in components(uv)
            rd = only(RecipesBase.apply_recipe(Dict{Symbol, Any}(), c))
            @test rd.plotattributes[:seriestype] == :line
        end
    end
end
