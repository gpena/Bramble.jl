module QualityInvalidationsTests

using Test
using Logging: NullLogger, with_logger

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

# A subprocess that fails is a failed gate, not a skip: an error inside the script (a renamed
# name it calls, an extension that fails to load) would otherwise pass the zero-invalidation
# check as one more broken.
function _subprocess_output(cmd::Base.AbstractCmd, what::AbstractString)
    return try
        read(cmd, String)
    catch e
        @error "$what subprocess failed" exception = e
        @test false
        nothing
    end
end

# Keeps what is recorded in it, printing nothing and passing nothing up: the control below
# needs a failure that stays inside it.
struct _Recorded <: Test.AbstractTestSet
    description::String
    results::Vector{Any}
end
_Recorded(description; kwargs...) = _Recorded(description, Any[])
Test.record(ts::_Recorded, r) = (push!(ts.results, r); r)
Test.finish(ts::_Recorded) = ts

@testset "a failing subprocess fails" begin
    script = joinpath(@__DIR__, "no_such_script.jl")
    ts = with_logger(NullLogger()) do
        @testset _Recorded "control" begin
            cmd = `$(Base.julia_cmd()) --startup-file=no $script`
            _subprocess_output(pipeline(cmd; stderr = devnull), "Control")
        end
    end
    @test count(r -> r isa Test.Fail, ts.results) == 1
    @test !any(r -> r isa Test.Broken, ts.results)
end

@testset "Invalidations" begin
    if isempty(VERSION.prerelease)
        script = joinpath(@__DIR__, "invalidations_snoop.jl")
        jl = Base.julia_cmd()
        project = Base.active_project()

        out = _subprocess_output(
            `$jl --project=$project --startup-file=no $script`, "Invalidation snoop"
        )

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

# SciMLBase loaded first, so `using Bramble` also loads BrambleSciMLExt. Before #755 the
# extension's `element(::AbstractSpaceType, ::LinearSolution)` alone invalidated 238
# MethodInstances, 249 in all. The bound is the 11 left after it (the two `eltype` trees
# #755 also names) plus 10.
const _SCIML_INVALIDATION_BOUND = 21

@testset "SciMLBase-loaded invalidations" begin
    if isempty(VERSION.prerelease)
        script = joinpath(@__DIR__, "invalidations_sciml_snoop.jl")
        jl = Base.julia_cmd()
        project = Base.active_project()

        out = _subprocess_output(
            `$jl --project=$project --startup-file=no $script`, "SciMLBase invalidation snoop"
        )

        if out !== nothing
            # The extension must have loaded inside the snoop, or nothing of it is counted.
            @test occursin("SCIML_EXT_LOADED=true", out)
            m = match(r"SCIML_OWNED_INVALIDATED=(\d+)", out)
            @test m !== nothing
            if m !== nothing
                n_invalidated = parse(Int, something(m.captures[1]))
                @test n_invalidated <= _SCIML_INVALIDATION_BOUND
                n_invalidated > _SCIML_INVALIDATION_BOUND &&
                    @info "Package-owned invalidations found (SciMLBase loaded first):\n$out"
            end
        end
    else
        @test_skip "SciMLBase-loaded invalidation check skipped on prerelease Julia"
    end
end

@testset "Polyester load invalidations" begin
    if isempty(VERSION.prerelease)
        jl = Base.julia_cmd()
        project = Base.active_project()
        snoop = joinpath(@__DIR__, "invalidations_polyester_snoop.jl")
        reinfer = joinpath(@__DIR__, "invalidations_polyester_reinfer.jl")

        out = _subprocess_output(
            `$jl --project=$project --startup-file=no $snoop`, "Polyester invalidation snoop"
        )

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
        out = _subprocess_output(
            `$jl_nocov --project=$project --threads=2 --startup-file=no $reinfer`,
            "Polyester reinference"
        )

        if out !== nothing
            m = match(r"REINFER_POLYESTER=(\d+)", out)
            @test m !== nothing
            if m !== nothing
                n_reinferred = parse(Int, something(m.captures[1]))
                @test n_reinferred == 0
                n_reinferred > 0 && @info "Re-inference after loading Polyester:\n$out"
            end
            # New user functions in the split sweeps infer nothing of Polyester's family;
            # the extension's own instances show the sweeps reached it.
            m_ext = match(r"REINFER_NEWF_EXT=(\d+)", out)
            m_newf = match(r"REINFER_NEWF_POLYESTER=(\d+)", out)
            @test m_ext !== nothing
            @test m_newf !== nothing
            if m_ext !== nothing && m_newf !== nothing
                @test parse(Int, something(m_ext.captures[1])) > 0
                n_newf = parse(Int, something(m_newf.captures[1]))
                @test n_newf == 0
                n_newf > 0 && @info "Polyester inferred for new user functions:\n$out"
            end
        end
    else
        @test_skip "Polyester load invalidation check skipped on prerelease Julia"
    end
end

end # module QualityInvalidationsTests
