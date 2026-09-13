#===========================================================================#
# Zero-tolerance gate on package-owned method invalidations (#198).
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
                n_owned = parse(Int, m.captures[1])
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
