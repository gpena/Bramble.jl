module ExtMeshesExtTests

using Test
using Bramble
using Meshes, MakieCore

# BrambleMeshesExt's `Meshes.viz` dispatch. This is a visualization entry point, not a
# conversion accessor: it builds a `Meshes.Box`/`Meshes.RectilinearGrid` from a Bramble
# `CartesianProduct`/`MeshnD` and hands it to `Meshes.viz`, which needs a real Makie backend
# (GLMakie/CairoMakie) to actually render pixels. None is loaded here, so every 2D/3D call
# below falls into the extension's no-backend path and warns rather than throwing -- what
# is under test is the conversion logic (does building the geometry from
# Bramble's own types succeed at all), not the rendering.
#
# `Bramble.×`/`Bramble.interval`/`Bramble.mesh`/`Bramble.domain` are qualified throughout:
# `using Bramble, Meshes` together makes bare `×`/`mesh`/`domain` ambiguous (both packages
# export names by these spellings).

_viz_silent(x; kwargs...) = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
    return Meshes.viz(x; kwargs...)
end

@testset "BrambleMeshesExt" begin
    @testset "2D CartesianProduct" begin
        S2 = Bramble.box((0.0, 0.0), (1.0, 2.0))
        @test (_viz_silent(S2); true)   # runs the Box conversion without throwing
        @test (_viz_silent(S2; color = :red); true)   # keywords are accepted and forwarded
    end

    @testset "3D CartesianProduct" begin
        S3 = Bramble.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        @test (_viz_silent(S3); true)
    end

    @testset "no backend: warns only" begin
        S2 = Bramble.box((0.0, 0.0), (1.0, 2.0))
        Ω2 = Bramble.mesh(Bramble.domain(S2), (4, 5), (true, true))
        for x in (S2, Ω2)
            out = mktemp() do path, io
                redirect_stdout(io) do
                    @test_logs (:warn, r"Makie backend") match_mode = :any Meshes.viz(x)
                end
                flush(io)
                close(io)
                return read(path, String)
            end
            @test isempty(out)
            @test isnothing(_viz_silent(x))
        end
    end

    @testset "1D CartesianProduct: throws" begin
        @test_throws ArgumentError Meshes.viz(Bramble.interval(0.0, 1.0))
    end

    @testset "2D MeshnD" begin
        Ω2 = Bramble.mesh(
            Bramble.domain(Bramble.box((0.0, 0.0), (1.0, 2.0))), (4, 5), (true, true)
        )
        @test (_viz_silent(Ω2); true)   # RectilinearGrid conversion from the mesh's points
        @test (_viz_silent(Ω2; color = :red); true)
    end

    @testset "3D MeshnD" begin
        Ω3 = Bramble.mesh(
            Bramble.domain(Bramble.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
            (3, 3, 3),
            (true, true, true)
        )
        @test (_viz_silent(Ω3); true)
    end

    @testset "1D Mesh1D: throws" begin
        Ω1 = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 5, true)
        @test_throws ArgumentError Meshes.viz(Ω1)
        err = try
            Meshes.viz(Ω1)
        catch e
            e
        end
        @test occursin("dimension >= 2, got 1", err.msg)
    end
end

end # module ExtMeshesExtTests
