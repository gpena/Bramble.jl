using Test
using Bramble
using Meshes, MakieCore

# BrambleMeshesExt's `Meshes.viz` dispatch. This is a visualization entry point, not a
# conversion accessor: it builds a `Meshes.Box`/`Meshes.RectilinearGrid` from a Bramble
# `CartesianProduct`/`MeshnD` and hands it to `Meshes.viz`, which needs a real Makie backend
# (GLMakie/CairoMakie) to actually render pixels. None is loaded here, so every 2D/3D call
# below falls into the extension's own `catch` and prints its install hint rather than
# throwing -- what is under test is the conversion logic (does building the geometry from
# Bramble's own types succeed at all), not the rendering.
#
# `Bramble.×`/`Bramble.interval`/`Bramble.mesh`/`Bramble.domain` are qualified throughout:
# `using Bramble, Meshes` together makes bare `×`/`mesh`/`domain` ambiguous (both packages
# export names by these spellings).

@testset "BrambleMeshesExt" begin
    @testset "2D CartesianProduct" begin
        S2 = Bramble.box((0.0, 0.0), (1.0, 2.0))
        @test (Meshes.viz(S2); true)   # runs the Box conversion without throwing
    end

    @testset "3D CartesianProduct" begin
        S3 = Bramble.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        @test (Meshes.viz(S3); true)
    end

    @testset "1D CartesianProduct: no override, asserts" begin
        # Only `Mesh1D` gets its own (non-throwing) `@error` method below; a bare 1D
        # `CartesianProduct` falls through to the D >= 2 assertion every other `viz` method
        # shares.
        @test_throws AssertionError Meshes.viz(Bramble.interval(0.0, 1.0))
    end

    @testset "2D MeshnD" begin
        Ω2 = Bramble.mesh(Bramble.domain(Bramble.box((0.0, 0.0), (1.0, 2.0))), (4, 5),
            (true, true))
        @test (Meshes.viz(Ω2); true)   # RectilinearGrid conversion from the mesh's points
    end

    @testset "3D MeshnD" begin
        Ω3 = Bramble.mesh(
            Bramble.domain(Bramble.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (3, 3, 3),
            (true, true, true))
        @test (Meshes.viz(Ω3); true)
    end

    @testset "1D Mesh1D: logs, does not throw" begin
        Ω1 = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 5, true)
        @test_logs (:error, r"1D meshes") Meshes.viz(Ω1)
    end
end
