# test/ext/zero_allocations_polyester.jl: the warm in-place calls of #854 allocate 0 B under
# `CpuPolyester()` too. It runs the table of test/zero_allocation_cases.jl, the one the
# `Serial()` test in test/form/zero_allocations.jl runs, plus `THREADED_CASES`, so the two
# policies cannot drift apart. The long-stencil `assemble!` cases of #853 (item 3) are in
# that table and are pinned at 0 B here.
#
# Gated like every other ext/*.jl file (test/runtests.jl reaches this group under
# BRAMBLE_TEST_GROUP=ext or full). Standalone:
#
#   julia --project=test -t 2 -e 'using Bramble, Test; include("test/TestUtils.jl");
#     include("test/ext/zero_allocations_polyester.jl")'
#
# A case that allocates under `CpuPolyester()` by a limit in the header of
# ext/BramblePolyesterExt.jl goes in `POLYESTER_LIMITS` with that line as its reason. A
# case that fails without one is a defect, never an entry there.
module ZeroAllocationsPolyesterTests

using Test
using Bramble: Bramble, CpuPolyester
using Polyester: Polyester
using ..TestUtils: alloc_test, WITH_3D_TESTS

isdefined(Main, :ZeroAllocationCases) ||
    Base.include(Main, joinpath(@__DIR__, "..", "zero_allocation_cases.jl"))
using Main.ZeroAllocationCases: CASES, THREADED_CASES

# Bytes of the second call: `setup` builds the arguments, `alloc_test` warms and measures
# behind a function barrier.
bytes_of(setup, policy) = alloc_test(setup(policy)...)

wanted(c) = c.dim < 3 || WITH_3D_TESTS

# Case name => the limit of the extension header that explains its bytes.
const POLYESTER_LIMITS = Dict{String, String}()

@testset "Polyester is live: more than one thread" begin
    # With one thread `@batch` runs one chunk inline and the argument box is never built, so
    # 0 B would prove nothing. Run with `--threads 2` or more.
    @test Threads.nthreads() > 1
    @test Base.get_extension(Bramble, :BramblePolyesterExt) !== nothing
end

@testset "CpuPolyester: 0 B once warm" begin
    for c in vcat(CASES, THREADED_CASES)
        wanted(c) || continue
        @testset "$(c.name)" begin
            if haskey(POLYESTER_LIMITS, c.name)
                @test bytes_of(c.setup, CpuPolyester()) > 0
            else
                @test bytes_of(c.setup, CpuPolyester()) == 0
            end
        end
    end
end

@testset "Every limit names a case" begin
    names = Set(c.name for c in vcat(CASES, THREADED_CASES))
    @test all(k -> k in names, keys(POLYESTER_LIMITS))
    @test all(r -> length(r) > 20, values(POLYESTER_LIMITS))
end

end # module
