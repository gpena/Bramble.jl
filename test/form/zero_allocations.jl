module FormZeroAllocationsTests

# The runtime in-place functions, warm, under `Serial()`: 0 B each (#854). The cases are
# in test/zero_allocation_cases.jl, shared with the `CpuPolyester()` file in test/ext/,
# which runs the same table so the two policies cannot drift apart. What `!` promises, and
# what the table leaves out, is in STANDARDS.md ("What `!` promises"; #1091). A mesh in 3D
# runs nightly (`WITH_3D_TESTS`, #650).

using Test
using Bramble: Serial
using ..TestUtils: alloc_test, WITH_3D_TESTS

isdefined(Main, :ZeroAllocationCases) ||
    Base.include(Main, joinpath(@__DIR__, "..", "zero_allocation_cases.jl"))
using Main.ZeroAllocationCases: CASES, EXCLUDED, FAMILIES, KNOWN, SHIFTS, SUFFIXES,
                                THREADED_CASES

# Bytes of the second call: `setup` builds the arguments, `alloc_test` warms and measures
# behind a function barrier.
bytes_of(setup, policy) = alloc_test(setup(policy)...)

wanted(c) = c.dim < 3 || WITH_3D_TESTS

@testset "Serial: 0 B once warm" begin
    for c in CASES
        wanted(c) || continue
        @testset "$(c.name)" begin
            @test bytes_of(c.setup, Serial()) == 0
        end
    end
end

# Each is a call that allocates today, under an issue. A fix turns its entry red.
@testset "Serial: allocation under an issue" begin
    for k in KNOWN
        wanted(k.case) || continue
        @testset "$(k.case.name)" begin
            @test bytes_of(k.case.setup, Serial()) > 0
        end
    end
end

# The table names every function of #854, so a deleted case cannot leave one untested.
@testset "The table covers the listed functions" begin
    names = [c.name for c in CASES]
    covers(fn) = any(n -> startswith(n, fn * " "), names)
    for fam in (FAMILIES..., SHIFTS...), suffix in SUFFIXES[1:2]

        @test covers(string(fam, suffix, "!"))
    end
    for fn in ("D₋₂!", "S₊₂!", "divₕ!", "div₊ₕ!", "divcₕ!", "diṽₕ!", "div̽ₕ!", "Δₕ!",
        "curlₕ!", "curl₊ₕ!", "curlcₕ!", "curl̃ₕ!", "curl̽ₕ!", "∇cₕ!", "∇̃ₕ!", "∇̽ₕ!",
        "εₕ!", "εcₕ!", "ε₊ₕ!", "ε̽ₕ!", "Rₕ!", "avgₕ!", "evaluate!", "assemble!",
        "assemble_add!", "type_cached_assemble!", "dirichlet_bc!", "reaction!",
        "reaction_density!", "lift!", "update_ghosts!", "symmetrize!", "smooth!",
        "v_cycle!", "w_cycle!", "fdm_solve!", "fmg!", "prolongate!", "πₕ!", "project!",
        "jacobian!", "mul!", "ldiv!")
        @test covers(fn)
    end
    # The long cases of #853: 216, 81 and past-32-entry stencils, and 40 products.
    for fn in ("assemble! Δₕ³", "assemble! Δₕ² both", "assemble! 40 products")
        @test covers(fn)
    end
    @test "assemble! Δₕ² 3D" in names
end

@testset "Threaded cases kept for Polyester" begin
    @test !isempty(THREADED_CASES)
    @test all(c -> startswith(c.name, "assemble_parallel!"), THREADED_CASES)
    @test isdisjoint(Set(c.name for c in CASES), Set(c.name for c in THREADED_CASES))
end

@testset "The exclusion list is named and reasoned" begin
    @test allunique(first.(EXCLUDED))
    @test all(e -> length(last(e)) > 20, EXCLUDED)
    @test all(k -> length(k.reason) > 20, KNOWN)
    @test allunique([c.name for c in CASES])
end

end # module
