using Test
using Bramble

# `:boundary`/`:interior` are reserved markers every mesh now carries automatically,
# computed from the mesh's own shape (see `_ensure_geometric_markers!`
# in src/mesh/marker.jl). Every case here is checked against a real mesh's marker
# BitVectors, not just that construction did or didn't throw.

@testset "Reserved geometric markers" begin
    S = interval(0.0, 1.0) × interval(0.0, 1.0)

    @testset "Default markers" begin
        Ωₕ = mesh(domain(S), (4, 4), (true, true))
        @test Set(keys(Bramble.markers(Ωₕ))) == Set([:boundary, :interior])
        @test sum(Bramble.markers(Ωₕ)[:boundary]) == 12   # 16 points, 4 strictly interior
        @test sum(Bramble.markers(Ωₕ)[:interior]) == 4
        @test Bramble.markers(Ωₕ)[:interior] == .!Bramble.markers(Ωₕ)[:boundary]
    end

    @testset "Custom label agreement" begin
        Ωₕ = mesh(domain(S, :bottom => :bottom), (4, 4), (true, true))
        @test Set(keys(Bramble.markers(Ωₕ))) == Set([:bottom, :boundary, :interior])

        Ωₕ_default = mesh(domain(S), (4, 4), (true, true))
        @test Bramble.markers(Ωₕ)[:boundary] == Bramble.markers(Ωₕ_default)[:boundary]
        @test Bramble.markers(Ωₕ)[:interior] == Bramble.markers(Ωₕ_default)[:interior]
    end

    @testset "Single-point mesh" begin
        Ωₕ = mesh(domain(interval(1.0, 1.0)), 1, true)
        @test Bramble.markers(Ωₕ)[:boundary] == [true]
        @test Bramble.markers(Ωₕ)[:interior] == [false]
    end

    @testset "Geometric interior" begin
        # The bug this closes: :interior used to mean "not :boundary", and a mesh with no
        # :boundary key silently made that "true everywhere".
        Ωₕ = mesh(domain(S, :bottom => :bottom), (4, 4), (true, true))
        Wₕ = gridspace(Ωₕ)
        u = Rₕ(Wₕ, x -> 1.0)
        v = Rₕ(Wₕ, x -> 1.0)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(Bramble.restrict_to(:interior, u), v))
        interior_sum = Bramble.dot(v.data, assemble(a) * u.data)
        full_sum = innerₕ(u, v)
        @test interior_sum < full_sum
        @test interior_sum ≈
              sum(Bramble.weights(Wₕ, Bramble.Innerh())[Bramble.markers(Ωₕ)[:interior]])
    end

    @testset "Reserved symbol match" begin
        Ωₕ = mesh(domain(S, :boundary => (:left, :right, :top, :bottom)), (4, 4), (
            true, true))
        Ωₕ_default = mesh(domain(S), (4, 4), (true, true))
        @test Bramble.markers(Ωₕ)[:boundary] == Bramble.markers(Ωₕ_default)[:boundary]
    end

    @testset "Reserved symbol override" begin
        # `:boundary`/`:interior` were already usable as ordinary custom labels before this
        # existed, so a mismatch warns rather than errors: the custom definition wins, not
        # the geometric one, since erroring would break that pre-existing freedom.
        Ωₕ = @test_logs (:warn, r"boundary.*something other than") mesh(
            domain(S, :boundary => :left), (4, 4), (true, true))
        @test Bramble.markers(Ωₕ)[:boundary] !=
              Bramble.markers(mesh(domain(S), (4, 4), (true, true)))[:boundary]
        @test sum(Bramble.markers(Ωₕ)[:boundary]) == 4   # just the :left face on a 4x4 grid
    end

    @testset "Condition markers" begin
        is_geom_boundary(x) = x[1] == 0.0 || x[1] == 1.0 || x[2] == 0.0 || x[2] == 1.0

        # matches geometry exactly -- no warning, no divergence
        Ωₕ = mesh(domain(S, :boundary => is_geom_boundary), (4, 4), (true, true))
        Ωₕ_default = mesh(domain(S), (4, 4), (true, true))
        @test Bramble.markers(Ωₕ)[:boundary] == Bramble.markers(Ωₕ_default)[:boundary]

        Ωₕ2 = mesh(domain(S, :interior => (x -> !is_geom_boundary(x))), (4, 4), (
            true, true))
        @test Bramble.markers(Ωₕ2)[:interior] == Bramble.markers(Ωₕ_default)[:interior]

        # a custom, non-reserved condition marker is untouched by any of this
        Ωₕ3 = mesh(domain(S, :left_half => (x -> x[1] < 0.5)), (4, 4), (true, true))
        @test haskey(Bramble.markers(Ωₕ3), :left_half)
        @test haskey(Bramble.markers(Ωₕ3), :boundary)
        @test haskey(Bramble.markers(Ωₕ3), :interior)

        # a condition meaning something else under a reserved name warns, keeps its own value
        Ωₕ4 = @test_logs (:warn, r"boundary") mesh(
            domain(S, :boundary => (x -> x[1] < 0.5)), (4, 4), (true, true))
        @test sum(Bramble.markers(Ωₕ4)[:boundary]) == 8   # x[1] < 0.5 on a 4x4 grid
    end
end
