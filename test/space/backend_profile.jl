module SpaceBackendProfileTests

using Test
using Bramble
using Bramble: BackendProfile, profile_backends, _crossover

@testset "profile_backends shape" begin
    p = profile_backends()
    has_polyester = Base.get_extension(Bramble, :BramblePolyesterExt) !== nothing

    @test p.sizes == [2^10, 2^12, 2^14, 2^16, 2^18, 2^20, 2^22]
    @test all(>(0), p.times)
    @test size(p.times) == (length(p.sizes), length(p.labels))
    @test length(p.spellings) == length(p.labels)
    @test p.labels[1] == "Serial()"
    @test "Parallel()" in p.labels
    @test ("CpuPolyester()" in p.labels) == has_polyester
    @test length(p.labels) == (has_polyester ? 3 : 2)

    txt = sprint(show, MIME"text/plain"(), p)
    @test occursin("csr_backend()", txt)
    @test occursin("Matrix storage is not profiled", txt)
    @test occursin("Parallel()", txt)
    @test occursin("BackendProfile(", sprint(show, p))
end

@testset "_crossover on a hand-built profile" begin
    sizes = [10, 20, 30, 40]
    #                 serial  always  late  never  middle
    times = [4.0 1.0 4.0 4.0 4.0;
             4.0 1.0 4.0 4.0 1.0;
             4.0 1.0 1.0 4.0 4.0;
             4.0 1.0 1.0 4.0 4.0]
    p = BackendProfile(sizes, ["S", "A", "L", "N", "M"], fill("", 5), times)
    @test _crossover(p, 2) == 10
    @test _crossover(p, 3) == 30
    @test _crossover(p, 4) === nothing
    @test _crossover(p, 5) === nothing
end

end # module SpaceBackendProfileTests
