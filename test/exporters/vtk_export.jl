using Test
using Bramble
using WriteVTK

# `export_vtk` is a thin wrapper: it reshapes what is already there (`to_matrix`, whose
# correctness is pinned in test/space/vector_elements.jl) and hands it to WriteVTK, which
# writes its own well-formed XML. So what is worth checking here is the wiring — that a
# 1D mesh does not hit the "not implemented" wall the exporter this replaces had, that a
# composite element becomes a multi-component vector field rather than one field per
# component, and that a call with no WriteVTK loaded fails with a message that says why
# rather than a bare MethodError — not the XML byte-for-byte, which is WriteVTK's job to
# get right, not this package's to re-verify.
#
# `Base.read` on the produced file, rather than opening it in ParaView, is enough: the
# structural facts under test — extents, field names, component counts — are all in the
# header, in plain text ahead of the compressed appended data.

@testset "VTK export" begin
    @testset "a 1D mesh, which the exporter this replaces refused" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 6, true)
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, sin)

        mktempdir() do dir
            files = export_vtk(joinpath(dir, "u"), Ωₕ, "u" => uₕ)
            @test length(files) == 1
            @test isfile(only(files))

            xml = read(only(files), String)
            @test occursin("RectilinearGrid", xml)
            # 6 points along x, one along the degenerate y and z axes: extent is
            # point-count minus one in each direction.
            @test occursin("WholeExtent=\"0 5 0 0 0 0\"", xml)
            @test occursin("Name=\"u\" NumberOfComponents=\"1\"", xml)
        end
    end

    @testset "a 2D mesh, mesh-and-fields and the single-element shorthand" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 4), (true, true))
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> sin(x[1]) * x[2])

        mktempdir() do dir
            f1 = only(export_vtk(joinpath(dir, "a"), Ωₕ, "u" => uₕ))
            f2 = only(export_vtk(joinpath(dir, "b"), uₕ))   # shorthand, named "u"
            @test read(f1, String) == read(f2, String)

            xml = read(f1, String)
            @test occursin("WholeExtent=\"0 4 0 3 0 0\"", xml)
        end
    end

    @testset "a composite element becomes one multi-component field" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 4), (true, true))
        Wₕ = gridspace(Ωₕ)
        Vₕ = Wₕ^Val(2)
        vₕ = Rₕ(Vₕ, x -> (x[1], x[2]))

        mktempdir() do dir
            f = only(export_vtk(joinpath(dir, "v"), Ωₕ, "velocity" => vₕ))
            xml = read(f, String)
            @test occursin("Name=\"velocity\" NumberOfComponents=\"2\"", xml)
        end
    end

    @testset "3D, and a raw array alongside an element in one call" begin
        Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (3, 4, 3),
            (true, true, true))
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> x[1] + x[2] + x[3])
        raw = collect(1.0:ndofs(Wₕ))

        mktempdir() do dir
            f = only(export_vtk(joinpath(dir, "m"), Ωₕ, "u" => uₕ, "raw" => raw))
            xml = read(f, String)
            @test occursin("WholeExtent=\"0 2 0 3 0 2\"", xml)
            @test occursin("Name=\"u\" NumberOfComponents=\"1\"", xml)
            @test occursin("Name=\"raw\" NumberOfComponents=\"1\"", xml)
        end
    end

    # Not tested here: what `export_vtk` does when WriteVTK has not been loaded. Once this
    # file's `using WriteVTK` above runs, the extension is active for the rest of this
    # process — multiple dispatch has already resolved `_export_vtk`'s specialization over
    # its `::Any` fallback for any `AbstractMeshType` argument, permanently, since Julia does
    # not un-load a method. Reproducing the unloaded case honestly needs a subprocess that
    # never touches WriteVTK, which nothing else in test/ does (only benchmark/benchmarks.jl
    # spawns Julia, for timing) — disproportionate machinery for one error string. Checked
    # by hand instead: `export_vtk("x", Ωₕ)` without `using WriteVTK` gives "export_vtk
    # requires WriteVTK.jl. Add `using WriteVTK` before calling this function.", which is
    # also in the docstring's own words.
end
