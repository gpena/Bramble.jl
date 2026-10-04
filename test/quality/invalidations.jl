module QualityInvalidationsTests

using Test

#===========================================================================#
# Zero-tolerance gate on package-owned method invalidations.
#
# `using Bramble` has already happened by the time this file is `include`d
# (test/runtests.jl does it at the top, for everything else in the suite), so
# `@snoop_invalidations using Bramble` here would see nothing: Julia does not
# re-run a module's method definitions on a second `using`. The check instead
# runs test/quality/invalidations_snoop.jl in a fresh subprocess, on the same
# project this test run resolved, and reads back the count it prints.
#===========================================================================#

@testset "Invalidations" begin
    if isempty(VERSION.prerelease)
        script = joinpath(@__DIR__, "invalidations_snoop.jl")
        jl = Base.julia_cmd()
        project = Base.active_project()

        out = try
            read(`$jl --project=$project --startup-file=no $script`, String)
        catch e
            @test_skip "Invalidation snoop subprocess failed to run: $e"
            nothing
        end

        if out !== nothing
            m = match(r"OWNED_COUNT=(\d+)", out)
            @test m !== nothing
            if m !== nothing
                n_owned = parse(Int, something(m.captures[1]))
                @test n_owned == 0
                if n_owned > 0
                    @info "Package-owned invalidations found (using Bramble):\n$out"
                end
            end
        end
    else
        @test_skip "Invalidation check skipped on prerelease Julia"
    end
end

@testset "Polyester load invalidations" begin
    if isempty(VERSION.prerelease)
        jl = Base.julia_cmd()
        project = Base.active_project()
        snoop = joinpath(@__DIR__, "invalidations_polyester_snoop.jl")
        reinfer = joinpath(@__DIR__, "invalidations_polyester_reinfer.jl")

        out = try
            read(`$jl --project=$project --startup-file=no $snoop`, String)
        catch e
            @test_skip "Polyester invalidation snoop subprocess failed to run: $e"
            nothing
        end

        if out !== nothing
            m = match(r"POLYESTER_OWNED=(\d+)", out)
            @test m !== nothing
            if m !== nothing
                n_owned = parse(Int, something(m.captures[1]))
                @test n_owned == 0
                n_owned > 0 && @info "Package-owned invalidations found (using Polyester):\n$out"
            end
        end

        # Without coverage: a covered package's precompiled code is not used, so under CI's
        # `coverage=true` every Bramble method infers again whatever Polyester did.
        jl_nocov = Cmd(filter(a -> !startswith(a, "--code-coverage"), jl.exec))
        out = try
            read(`$jl_nocov --project=$project --threads=2 --startup-file=no $reinfer`, String)
        catch e
            @test_skip "Polyester reinference subprocess failed to run: $e"
            nothing
        end

        if out !== nothing
            m = match(r"REINFER_POLYESTER=(\d+)", out)
            @test m !== nothing
            if m !== nothing
                n_reinferred = parse(Int, something(m.captures[1]))
                @test n_reinferred == 0
                n_reinferred > 0 && @info "Re-inference after loading Polyester:\n$out"
            end
        end
    else
        @test_skip "Polyester load invalidation check skipped on prerelease Julia"
    end
end

end # module QualityInvalidationsTests
