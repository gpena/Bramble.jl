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

# What the report prints for lcov files `infos` in `dir`, run from `dir` as Weekly does from the
# root: each `SF` path is read relative to the working directory.
function _floor_report(dir, infos...)
    out = IOBuffer()
    cmd = Cmd(`$(Base.julia_cmd()) --startup-file=no $_SCRIPT --floor 90 $infos`;
        dir = dir, ignorestatus = true)
    run(pipeline(cmd; stdout = out, stderr = out))
    return String(take!(out))
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

@testset "Uncounted function lines" begin
    mktempdir() do dir
        mkpath(joinpath(dir, "src"))
        write(joinpath(dir, "src", "x.jl"), """
            @inline function f(x)
                return x + 1
            end
            function g(x)
                return x
            end
            h(x) = x
            function k(a,
                       b)
                return a
            end
            function m end
            """)
        # f has a 0 header over a hit body, g a 0 header over a 0 body, h is a 0 one-liner, k has
        # a 0 header over two lines and a hit body, and `function m end` is no block.
        write(joinpath(dir, "x.info"), "SF:src/x.jl\nDA:1,0\nDA:2,5\nDA:4,0\nDA:5,0\n" *
                                      "DA:7,0\nDA:8,0\nDA:10,3\nDA:12,0\nend_of_record\n")
        @testset "A header over a hit body is dropped" begin
            # f and k leave their headers; g, h and `function m` stay: 2 of 6 lines, 33.3.
            @test occursin(r"^33\.3 src/x\.jl BELOW$"m, _floor_report(dir, "x.info"))
        end

        @testset "A body hit in another input counts" begin
            write(joinpath(dir, "y.info"), "SF:src/x.jl\nDA:5,2\nend_of_record\n")
            # g now has a hit body, so its header goes too: 3 of 5 lines hit.
            @test occursin(r"^60\.0 src/x\.jl BELOW$"m, _floor_report(dir, "x.info", "y.info"))
        end

        @testset "A file that cannot be read is untouched" begin
            write(joinpath(dir, "z.info"), "SF:src/gone.jl\nDA:1,0\nDA:2,5\nend_of_record\n")
            @test occursin(r"^50\.0 src/gone\.jl BELOW$"m, _floor_report(dir, "z.info"))
        end

        @testset "The summary line keeps its format" begin
            @test occursin(r"^files=1 below=1 exempt=0 dropped=0$"m, _floor_report(dir, "x.info"))
        end
    end

    @testset "A headers entry is an error" begin
        mktempdir() do dir
            old = joinpath(dir, "headers.toml")
            write(old, """
                [[exempt_function]]
                path = "src/Bramble.jl"
                headers = ["f(x)"]
                reason = "uncounted: the per-method form"
                """)
            ok, output = _check_exempt(old)
            @test !ok
            @test occursin("unknown key `headers`", output)
            @test occursin("dropped by the rule itself", output)
        end
    end
end

end # module
