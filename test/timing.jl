# Per-file timing of one test group: setup, precompile, load and suite time, then every
# test file ranked by self time (its wall time minus the files it includes).
#
#     BRAMBLE_TEST_GROUP=unit julia --startup-file=no --threads=2 --check-bounds=yes test/timing.jl
#
# `--check-bounds=yes` matches `Pkg.test`, so the run reuses its package images. Run once
# right after editing `src/` to include the recompile of Bramble and its extensions, and once
# more with nothing changed for the suite alone. Measure on an otherwise idle machine.
#
# The per-file times come from the suite's own trace (`BRAMBLE_TEST_TRACE`, see
# `test/TestUtils.jl`), which prints each path as it was passed to `include`: files that share
# a name in different directories are told apart by their order in the output. Files run in
# suite order, so the first file to reach a code path pays its compilation. Compile time is
# summed over threads, so it can exceed wall time.

using Pkg, Printf

const TEST_DIR = @__DIR__
const GROUP = get!(ENV, "BRAMBLE_TEST_GROUP", "unit")
const TSV = joinpath(tempdir(), "bramble-test-timings-$(GROUP).tsv")
ENV["BRAMBLE_TEST_TRACE"] = "1"

Base.JLOptions().check_bounds == 1 ||
    @warn "Run with --check-bounds=yes to reuse the package images Pkg.test builds."

Pkg.activate(TEST_DIR; io = devnull)
t_inst = @elapsed Pkg.instantiate(; io = devnull)
t_pre = @elapsed Pkg.precompile()
Base.cumulative_compile_timing(true)
t_load = @elapsed @eval using Bramble

# Echo the suite's stdout while keeping every line, so the trace can be parsed afterwards.
const ORIG_STDOUT = stdout
const LINES = String[]
rd, wr = redirect_stdout()
reader = @async for line in eachline(rd)
    println(ORIG_STDOUT, line)
    push!(LINES, line)
end

c_start = Base.cumulative_compile_time_ns()[1] / 1e9
t_run = @elapsed try
    Base.include(Main, joinpath(TEST_DIR, "runtests.jl"))
catch err
    # A failing testset throws at the end of the suite; the timings are still valid.
    @warn "Suite finished with errors" exception = err
end
total_compile = Base.cumulative_compile_time_ns()[1] / 1e9 - c_start

redirect_stdout(ORIG_STDOUT)
close(wr)
wait(reader)

# Nested `[trace] start/end` pairs give inclusive times; self time subtracts the children.
rows = Tuple{String, Int, Float64, Float64}[]
stack = Vector{Any}[]
for line in LINES
    if startswith(line, "[trace] start ")
        push!(stack, Any[line[(length("[trace] start ") + 1):end], 0.0])
    elseif (m = match(r"^\[trace\] end (.*) elapsed=([\d.]+)s", line)) !== nothing
        wall = parse(Float64, m[2])
        child = pop!(stack)[2]
        isempty(stack) || (stack[end][2] += wall)
        push!(rows, (m[1], length(stack), wall, wall - child))
    end
end
sort!(rows; by = r -> r[4], rev = true)

open(TSV, "w") do io
    println(io, "file\tdepth\twall_s\tself_s")
    for (path, depth, wall, self) in rows
        @printf(io, "%s\t%d\t%.2f\t%.2f\n", path, depth, wall, self)
    end
end

println()
println("group $GROUP, $(Threads.nthreads()) threads")
@printf("  instantiate  %7.1f s\n", t_inst)
@printf("  precompile   %7.1f s   (Bramble, extensions and test deps)\n", t_pre)
@printf("  using        %7.1f s\n", t_load)
@printf("  suite        %7.1f s   (JIT compile %.1f s, summed over threads)\n", t_run,
    total_compile)
println("\nTop 25 files by self time:")
@printf("  %8s %8s  %s\n", "self_s", "wall_s", "file")
for (path, _, wall, self) in first(rows, 25)
    @printf("  %8.1f %8.1f  %s\n", self, wall, path)
end
println("\nFull table: ", TSV)
