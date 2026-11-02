module ExtEnzymePolyesterHooksTests

# The Enzyme rules of the `CpuPolyester` hooks that the census does not run (gpena/Bramble.jl#369):
# the `CAPTURES` of test/ext/enzyme_policy_cases.jl, each called through the closure route that
# `_sweep_for!` and the matrix-free replay take. Each capture runs in a child Julia process of
# its own, because one capture's Enzyme compile alone peaks near 6 GB and compiled code stays
# resident, so the captures cannot share a process under julia_slot's 10 GB cap. A child prints
# one `CAPTURE_RESULT` line per mode, and this file asserts each. Enzyme is not in
# test/Project.toml: the children load it through a temporary environment
# (test/ext/enzyme_policy_cases.jl says how), and the parent needs at least 2 threads.

using Test
using Bramble
using ..TestUtils: _have

const RUN_ENZYME = _have(:Enzyme) && Threads.nthreads() >= 2
RUN_ENZYME && @eval using Enzyme
RUN_ENZYME && include(joinpath(@__DIR__, "enzyme_policy_cases.jl"))

const HARNESS = joinpath(@__DIR__, "enzyme_policy_cases.jl")
const ENZYME_TOML = "[deps]\nEnzyme = \"7da242da-08ed-463a-9acd-ee780be4f1d9\"\n"

# The source of one child: its Enzyme environment, the harness, then one line per mode.
function _capture_source(hook::AbstractString, modes)
    return """
        _e = mktempdir()
        write(joinpath(_e, "Project.toml"), $(repr(ENZYME_TOML)))
        push!(LOAD_PATH, _e)
        using Bramble, Polyester, Enzyme, ForwardDiff
        Base.include(Main, $(repr(HARNESS)))
        for m in $(repr(modes))
            r = EnzymePolicyCases.evaluate(EnzymePolicyCases.case($(repr(hook)));
                policy = Bramble.CpuPolyester(), mode = m)
            println("CAPTURE_RESULT ", $(repr(hook)), " ", m, " ", r.status, " ", r.relerr)
        end
        """
end

# Runs the child for `hook` and its `modes` with the parent's project, thread count and
# optimisation level. Returns the child's exit code and its results by mode: (status, relerr).
function _run_capture(hook::AbstractString, modes)
    opt = max(Base.JLOptions().opt_level, 0)
    julia = Base.julia_cmd()
    src = _capture_source(hook, modes)
    cmd = `$julia --startup-file=no --project=$(Base.active_project()) --threads=$(Threads.nthreads()) --optimize=$opt -e $src`
    out = IOBuffer()
    p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
    results = Dict{Symbol, Tuple{Symbol, Float64}}()
    for line in split(String(take!(out)), '\n')
        startswith(line, "CAPTURE_RESULT ") || continue
        _, _, mode, status, relerr = split(line)
        results[Symbol(mode)] = (Symbol(status), parse(Float64, relerr))
    end
    return p.exitcode, results
end

# The captures whose modes, run in one child, peak above 7 GB: `_batch_mf_bands_coef` (reverse
# and forward, 5.96 GB for reverse alone) and `_batch_mf_overwrite_coef` (all five modes, 7.6 GB).
# Each of their modes runs in a child of its own.
const SEPARATE_MODES = ("_batch_mf_bands_coef", "_batch_mf_overwrite_coef")

# The groups of modes that share a child: all of them, or one mode per group for those above.
_mode_groups(hook::AbstractString, modes) = hook in SEPARATE_MODES ? [(m,) for m in modes] :
    [modes]

@testset "Captured closures under CpuPolyester" begin
    if RUN_ENZYME
        # One testset per `CAPTURES` hook: reverse and forward, and all five modes for
        # `_batch_mf_overwrite_coef`, the fused rule's own case. Each mode is one line of the
        # child's output, and a missing line fails the mode.
        @testset "Hook rules" begin
            for c in EnzymePolicyCases.CAPTURES
                @testset "$(c.hook)" begin
                    modes = (:reverse, :forward)
                    c.hook == "_batch_mf_overwrite_coef" &&
                        (modes = (EnzymePolicyCases.MODES..., EnzymePolicyCases.BATCH_MODES...))
                    results = Dict{Symbol, Tuple{Symbol, Float64}}()
                    for group in _mode_groups(c.hook, modes)
                        code, got = _run_capture(c.hook, group)
                        @test code == 0
                        merge!(results, got)
                    end
                    for m in modes
                        @test haskey(results, m)
                        haskey(results, m) && @test first(results[m]) == :ok
                    end
                end
            end
        end
    else
        @test_skip RUN_ENZYME
    end
end

end
