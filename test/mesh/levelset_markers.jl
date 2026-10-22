module MeshLevelSetMarkersTests

using Test
using Random
using Bramble

# `:D => levelset(φ)` keeps the grid and labels the nodes `D`, `D_boundary`, `D_exterior`
# from the crossings of φ = 0 with the grid edges. The expectations are recomputed here from
# φ alone, node by node.

φ(p) = hypot(p[1], p[2]) - 1
const X = interval(-1.25, 1.25) × interval(-1.25, 1.25)
const Ω = domain(X, :D => levelset(φ))

@testset "level-set markers" begin
    @testset "uniform grid: size kept, labels by φ" begin
        Ωₕ = mesh(Ω, (41, 41), (true, true))
        @test Bramble.npoints(Ωₕ, Tuple) == (41, 41)
        @test typeof(Ωₕ) == typeof(mesh(domain(X), (41, 41), (true, true)))
        I = Bramble.index_in_marker(Ωₕ, :D)
        Γ = Bramble.index_in_marker(Ωₕ, :D_boundary)
        E = Bramble.index_in_marker(Ωₕ, :D_exterior)
        @test count(Γ) == 4
        @test count(I) + count(Γ) + count(E) == 41^2
        @test !any(I .& E) && !any(I .& Γ) && !any(Γ .& E)
        p = Bramble.host_points(Ωₕ)
        @test all(I[i + 41 * (j - 1)] == (φ((p[1][i], p[2][j])) < 0)
        for i in 1:41, j in 1:41)
        @test all(Γ[i + 41 * (j - 1)] == (φ((p[1][i], p[2][j])) == 0)
        for i in 1:41, j in 1:41)
    end

    @testset "a node near the curve is a boundary node" begin
        Y = interval(0.0, 2.0) × interval(0.0, 1.0)
        for δ in (1e-12, -1e-12)
            # the curve x = 1 + δ crosses the edge between x = 1 and x = 1.5 or 0.5 and 1
            Ψ = domain(Y, :S => levelset(p -> p[1] - 1 - δ))
            Ψₕ = mesh(Ψ, (5, 3), (true, true))
            Γ = Bramble.index_in_marker(Ψₕ, :S_boundary)
            xs = Bramble.host_points(Ψₕ)[1]
            @test all(Γ[i + 5 * (j - 1)] == (xs[i] == 1.0) for i in 1:5, j in 1:3)
        end
        Ψ = domain(Y, :S => levelset(p -> p[1] - 1.1))
        Ψₕ = mesh(Ψ, (5, 3), (true, true))
        @test count(Bramble.index_in_marker(Ψₕ, :S_boundary)) == 0
    end

    @testset "labels re-derive on refine and move" begin
        Ωₕ = mesh(Ω, (41, 41), (true, true))
        Bramble.iterative_refinement!(Ωₕ, markers(Ω))
        @test Bramble.npoints(Ωₕ, Tuple) == (81, 81)
        @test count(Bramble.index_in_marker(Ωₕ, :D_boundary)) == 4

        Random.seed!(1)
        Mr = mesh(Ω, (37, 23), (false, false))
        @test Bramble.npoints(Mr, Tuple) == (37, 23)
        xs = collect(range(-1.25, 1.25, 37))
        xs[2:(end - 1)] .+= 0.01
        ys = collect(range(-1.25, 1.25, 23))
        Bramble.change_points!(Mr, markers(Ω), (xs, ys))
        Ir = Bramble.index_in_marker(Mr, :D)
        Γr = Bramble.index_in_marker(Mr, :D_boundary)
        @test count(Ir) + count(Γr) == count(φ((x, y)) < 0 for x in xs, y in ys)
    end

    @testset "polygons still close their grid" begin
        V = [(0.0, 2.0), (2.0, 0.0), (4.0, 0.0), (6.0, 2.0), (6.0, 3.0), (4.0, 3.0),
            (4.0, 5.0), (2.0, 5.0), (0.0, 3.0)]
        Random.seed!(2)
        Mp = mesh(domain(interval(0.0, 6.0) × interval(0.0, 5.0), :P => polygon(V)),
            (10, 9), (false, false))
        @test Bramble.npoints(Mp, Tuple) != (10, 9)
    end
end

end # module
