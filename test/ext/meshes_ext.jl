module ExtMeshesExtTests

using Test
using Bramble
using Meshes, MakieCore

# BrambleMeshesExt's `Meshes.viz` dispatch. This is a visualization entry point, not a
# conversion accessor: it builds a `Meshes.Box`/`Meshes.RectilinearGrid` from a Bramble
# `CartesianProduct`/`MeshnD` and hands it to `Meshes.viz`, which needs a real Makie backend
# (GLMakie/CairoMakie) to actually render pixels. Only MakieCore is loaded here, so run
# alone every 2D/3D call below falls into the extension's no-backend path and warns rather
# than throwing; what is under test is the conversion logic (does building the geometry
# from Bramble's own types succeed at all), not the rendering.
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

    @testset "viz without/with Makie" begin
        S2 = Bramble.box((0.0, 0.0), (1.0, 2.0))
        Ω2 = Bramble.mesh(Bramble.domain(S2), (4, 5), (true, true))
        # Which path `viz` takes depends on whether Meshes' Makie extension is loaded: an
        # earlier file of the `ext` group may have loaded Makie by the time this one runs.
        makie_loaded = Base.get_extension(Meshes, :MeshesMakieExt) !== nothing
        for x in (S2, Ω2)
            local result
            out = mktemp() do path, io
                redirect_stdout(io) do
                    if makie_loaded
                        # No pattern and `min_level = Warn`: any warning or error fails.
                        @test_logs min_level = Base.CoreLogging.Warn (result = Meshes.viz(x))
                    else
                        warning = (:warn, r"Makie backend")
                        @test_logs warning match_mode = :any (result = Meshes.viz(x))
                    end
                end
                flush(io)
                close(io)
                return read(path, String)
            end
            @test isempty(out)
            if makie_loaded
                @test !isnothing(result)
            else
                @test isnothing(result)
            end
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
