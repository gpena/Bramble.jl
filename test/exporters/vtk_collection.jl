module ExportersVtkCollectionTests

using Test
using Bramble
using WriteVTK
using SciMLBase
using LightXML

# `.pvd` XML: real parsing this time (see test/exporters/vtk_export.jl for the plain-text
# `occursin` checks that suffice for a single `.vtr`). A collection's own correctness lives
# in the `<DataSet>` entries -- their count, their `timestep` values, and that the files they
# name actually exist -- so this reads the XML with `LightXML` (the library WriteVTK itself
# writes with) rather than grepping the text.
function _dataset_entries(pvd_path::AbstractString)
    xdoc = parse_file(pvd_path)
    xroot = root(xdoc)
    collection = find_element(xroot, "Collection")
    entries = [(
                   timestep = parse(Float64, attribute(c, "timestep")),
                   file = attribute(c, "file")
               ) for c in child_elements(collection)]
    free(xdoc)
    return entries
end

@testset "VTK time-series collection" begin
    @testset "do-block: non-uniform times, exact values" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 4), (true, true))
        Wₕ = gridspace(Ωₕ)
        times = (0.0, 0.1, 0.35, 1.0)  # deliberately non-uniform

        mktempdir() do dir
            pvd_path = joinpath(dir, "series")
            files = export_vtk(pvd_path) do pvd
                for (i, t) in enumerate(times)
                    uₕ = Rₕ(Wₕ, x -> t * (x[1] + x[2]))
                    pvd[t] = (joinpath(dir, "step_$i"), Ωₕ, "u" => uₕ)
                end
            end
            @test length(files) == length(times) + 1  # + the .pvd itself
            @test all(isfile, files)

            entries = _dataset_entries(pvd_path * ".pvd")
            @test length(entries) == length(times)
            @test [e.timestep for e in entries] == collect(times)
            @test all(e -> isfile(joinpath(dir, e.file)), entries)
        end
    end

    @testset "do-block: composite field and 1D degenerate axis" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (true, true))
        Vₕ = gridspace(Ωₕ)^Val(2)

        Ω1 = mesh(domain(interval(0.0, 1.0)), 6, true)
        W1 = gridspace(Ω1)

        mktempdir() do dir
            pvd_path = joinpath(dir, "mixed")
            export_vtk(pvd_path) do pvd
                vₕ = Rₕ(Vₕ, x -> (x[1], x[2]))
                pvd[0.0] = (joinpath(dir, "vec"), Ωₕ, "velocity" => vₕ)

                u1 = Rₕ(W1, sin)
                pvd[1.0] = (joinpath(dir, "curve"), Ω1, "u" => u1)
            end

            xml_vec = read(joinpath(dir, "vec.vtr"), String)
            @test occursin("Name=\"velocity\" NumberOfComponents=\"2\"", xml_vec)

            xml_curve = read(joinpath(dir, "curve.vtr"), String)
            @test occursin("WholeExtent=\"0 5 0 0 0 0\"", xml_curve)

            entries = _dataset_entries(pvd_path * ".pvd")
            @test length(entries) == 2
        end
    end

    @testset "do-block: exception leaves a valid partial file" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true)
        Wₕ = gridspace(Ωₕ)

        mktempdir() do dir
            pvd_path = joinpath(dir, "partial")
            @test_throws ErrorException export_vtk(pvd_path) do pvd
                uₕ = Rₕ(Wₕ, x -> x[1])
                pvd[0.0] = (joinpath(dir, "ok"), Ωₕ, "u" => uₕ)
                error("simulated failure mid-series")
            end

            @test isfile(pvd_path * ".pvd")
            entries = _dataset_entries(pvd_path * ".pvd")
            @test length(entries) == 1
            @test only(entries).timestep == 0.0
        end
    end

    @testset "do-block: append" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true)
        Wₕ = gridspace(Ωₕ)

        mktempdir() do dir
            pvd_path = joinpath(dir, "grown")
            export_vtk(pvd_path) do pvd
                pvd[0.0] = (joinpath(dir, "a"), Ωₕ, "u" => Rₕ(Wₕ, x -> x[1]))
            end
            export_vtk(pvd_path; append = true) do pvd
                pvd[1.0] = (joinpath(dir, "b"), Ωₕ, "u" => Rₕ(Wₕ, x -> 2x[1]))
            end

            entries = _dataset_entries(pvd_path * ".pvd")
            @test length(entries) == 2
            @test [e.timestep for e in entries] == [0.0, 1.0]
        end
    end

    # A `SciMLBase.AbstractODESolution` without actually integrating anything: the same
    # `build_solution` construction `BrambleVTKSciMLExt`'s own precompile workload uses,
    # since no solver package (`OrdinaryDiffEq` and the rest) is a dependency of that
    # extension either.
    @testset "One-call SciMLBase solution export" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (true, true))
        Wₕ = gridspace(Ωₕ)

        u0 = collect(parent(Rₕ(Wₕ, x -> x[1])))
        u1 = collect(parent(Rₕ(Wₕ, x -> x[1] + x[2])))
        u2 = collect(parent(Rₕ(Wₕ, x -> 2 * (x[1] + x[2]))))
        ts = [0.0, 0.4, 1.0]  # non-uniform, on purpose

        prob = ODEProblem((du, u, p, t) -> nothing, u0, (0.0, 1.0))
        sol = SciMLBase.build_solution(prob, nothing, ts, [u0, u1, u2])

        @testset "every saved step" begin
            mktempdir() do dir
                pvd_path = joinpath(dir, "sol")
                files = export_vtk(pvd_path, Wₕ, sol)
                @test length(files) == length(ts) + 1

                entries = _dataset_entries(pvd_path * ".pvd")
                @test [e.timestep for e in entries] == ts
                @test all(e -> isfile(joinpath(dir, e.file)), entries)

                xml = read(joinpath(dir, entries[1].file), String)
                @test occursin("Name=\"u\" NumberOfComponents=\"1\"", xml)
            end
        end

        @testset "interpolated times, custom field name" begin
            mktempdir() do dir
                pvd_path = joinpath(dir, "sol_interp")
                interp_times = range(0.0, 1.0; length = 5)
                export_vtk(pvd_path, Wₕ, sol; name = "temperature", times = interp_times)

                entries = _dataset_entries(pvd_path * ".pvd")
                @test length(entries) == length(interp_times)
                @test [e.timestep for e in entries] ≈ collect(interp_times)

                xml = read(joinpath(dir, entries[1].file), String)
                @test occursin("Name=\"temperature\" NumberOfComponents=\"1\"", xml)
            end
        end
    end

    @testset "Fallback stubs, called directly" begin
        @test_throws "export_vtk requires WriteVTK.jl" Bramble._export_vtk_collection(identity, 42)
        @test_throws("export_vtk for a solution object requires WriteVTK.jl and SciMLBase.jl",
            Bramble._export_vtk_solution("x", gridspace(mesh(domain(interval(0.0, 1.0)), 3, true)), 42))
    end
end

end # module ExportersVtkCollectionTests
