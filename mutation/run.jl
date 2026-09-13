#===========================================================================#
# Mutation-testing driver for issue #121.
#
# Scoped to the four core stencil kernels named in the issue, where a
# regression is hardest to diagnose from a failing test alone:
#
#   src/form/operators/difference.jl
#   src/space/operators/difference.jl
#   src/space/operators/cell_average.jl
#   src/form/dirichlet_constraints.jl
#
# Uses Gremlins.jl (github.com/SuperSeriousLab/Gremlins.jl), not the
# MutationTesting.jl named in the issue — that package does not exist.
# Vimes.jl, the only other Julia mutation tester, has been unmaintained
# since 2019 and fails to resolve against current General (stale
# StatsBase upper bound). Gremlins is JuliaSyntax-based, targets Julia
# 1.10+, and resolves cleanly on 1.13.
#
# mutation/Project.toml points Gremlins at a private patched fork, not
# upstream — see the "Gremlins upstream fix tracker" memory for why
# (two real bugs found and reported upstream: SuperSeriousLab/Gremlins.jl#8
# and #9).
#
# Deliberately uses the COLD path (`mutate`), not `mutate_warm`: the warm
# path's own I4 self-check failed 2/6 on this package's first smoke test
# (warm reported `killed` where a cold re-run reported `survived` — a
# false-kill, the dangerous direction). Until #9 is resolved, cold is the
# only path whose results this driver treats as trustworthy. Slower, but
# correct — see #9 for the tracking issue.
#
# Usage (always with 4 threads, matching this repo's test/benchmark
# convention — see CLAUDE.md):
#
#   JULIA_NUM_THREADS=4 julia --project=mutation mutation/run.jl [--max N] [--json path]
#
# `--max N` caps the run to N mutants (deterministic round-robin sample)
# for a smoke run; omit it for the full scoped run.
#===========================================================================#

using Gremlins

const PKGDIR = abspath(joinpath(@__DIR__, ".."))
const TARGET_FILES = [
    "form/operators/difference.jl",
    "space/operators/difference.jl",
    "space/operators/cell_average.jl",
    "form/dirichlet_constraints.jl",
]

function _flag_value(argv, name)
    i = findfirst(==(name), argv)
    i === nothing && return nothing
    return argv[i + 1]
end

max_mutants_arg = _flag_value(ARGS, "--max")
max_mutants = max_mutants_arg === nothing ? nothing : parse(Int, max_mutants_arg)
json_out = something(_flag_value(ARGS, "--json"), joinpath(@__DIR__, "report.json"))
md_out = joinpath(dirname(json_out), splitext(basename(json_out))[1] * ".md")

result = mutate(
    PKGDIR;
    test_dir = "mutation",
    test_file = "scoped_tests.jl",
    files = TARGET_FILES,
    max_mutants = max_mutants,
    verbose = true,
)

print_summary(result)

open(json_out, "w") do io
    write(io, report_json(result))
end
open(md_out, "w") do io
    write(io, report_markdown(result))
    write(io, "\n\n## Surviving mutant diffs\n\n")
    write(io, render_survivor_diffs(result, PKGDIR))
end
println("JSON report: ", json_out)
println("Markdown report: ", md_out)
