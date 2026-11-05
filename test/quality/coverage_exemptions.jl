module QualityCoverageExemptionsTests

using Test

# `.github/coverage_exemptions.toml` names files, functions and method prefixes in `src/`. A
# source move that renames or deletes one leaves a stale entry, which `Weekly.yml`'s
# `coverage-floor` job would only report after a week. `coverage_floor.jl --check-exempt`
# resolves every entry against the source with no lcov. The script ends in
# `exit(main(ARGS))`, so it runs in a child process, from the repository root: it resolves
# each `path` against the working directory, and `Pkg.test` does not start there.
#
# Deliberately Test-only: no `using Bramble`, so this file runs standalone.

const _ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const _SCRIPT = joinpath(_ROOT, ".github", "scripts", "coverage_floor.jl")
const _TOML = joinpath(_ROOT, ".github", "coverage_exemptions.toml")

# Whether the check passes, and what it printed.
function _check_exempt(toml)
    out = IOBuffer()
    cmd = Cmd(`$(Base.julia_cmd()) --startup-file=no $_SCRIPT --check-exempt $toml`;
        dir = _ROOT, ignorestatus = true)
    p = run(pipeline(cmd; stdout = out, stderr = out))
    return success(p), String(take!(out))
end

@testset "Coverage exemptions" begin
    @testset "Every entry resolves" begin
        ok, output = _check_exempt(_TOML)
        ok || @info "coverage exemptions are stale" output
        @test ok
    end

    @testset "A stale entry fails" begin
        mktempdir() do dir
            stale = joinpath(dir, "stale.toml")
            write(stale, """
                [[exempt_function]]
                path = "src/Bramble.jl"
                methods = ["no_such_method_for_the_positive_control("]
                reason = "uncounted: a method that does not exist"
                """)
            ok, output = _check_exempt(stale)
            @test !ok
            @test occursin("matches 0 definitions", output)
        end
    end

    @testset "A moved file is reported with the rest" begin
        mktempdir() do dir
            stale = joinpath(dir, "moved.toml")
            write(stale, """
                [[exempt_function]]
                path = "src/no_such_file.jl"
                lines = [{ method = "f(", text = "x" }]
                reason = "uncounted: a file that does not exist"

                [[exempt_function]]
                path = "src/Bramble.jl"
                methods = ["no_such_method_for_the_positive_control("]
                reason = "uncounted: a method that does not exist"
                """)
            ok, output = _check_exempt(stale)
            @test !ok
            @test occursin("source file not found: src/no_such_file.jl", output)
            @test occursin("no_such_method_for_the_positive_control(", output)
        end
    end
end

end # module
