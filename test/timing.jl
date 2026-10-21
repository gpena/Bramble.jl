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
# `test/TestUtils.jl`), which prints each path relative to `test/` (resolved against the
# including file's directory), so `form/matrix_free.jl` and `ghost/matrix_free.jl` differ. Files run in
# suite order, so the first file to reach a code path pays its compilation. Compile time is
# summed over threads, so it can exceed wall time. Each file's `compile_s` is its JIT time
# inclusive of the files it includes, like `wall_s`.
#
# Top-level testsets go to `<table>.testsets`: the top-level testsets of each test file,
# whatever wrapper testsets enclose its include, with the file, wall time, and the compile time
# since the previous top-level testset ended. That gap also holds the file's own code between
# two testsets, which compiles little, so the figure is the testset's cost to within that.
#
# The table's first line is a header that makes the file self-describing (commit, Julia,
# threads, group, file, testset and test counts, suite wall and compile time), because runs
# are compared across days:
#
#     julia test/timing.jl --compare before.tsv after.tsv
#
# prints both headers, then one row per file with before, after and after-minus-before self
# time, sorted by the size of the delta. A table from before the paths carried directories
# (base names only) is still read: if either table has such rows, both are compared by base
# name, in order of appearance, so `form/matrix_free.jl` and `ghost/matrix_free.jl` count as
# two rows of `matrix_free.jl`, as they do in the old table.
#
# Environment overrides, for checking the harness itself: `BRAMBLE_TIMING_SKIP_SETUP=1` skips
# `Pkg.activate/instantiate/precompile`, `BRAMBLE_TIMING_RUNTESTS` runs another file in place
# of `runtests.jl` (which must include `test/TestUtils.jl`, as the suite does, to be traced),
# and `BRAMBLE_TIMING_TSV` sets the output path.

using Pkg, Printf, Test

const TEST_DIR = @__DIR__
const HEADER_PREFIX = "# "

# Reads a table written below: the header line, and file => (wall, self) rows. A path that
# occurs twice (same name in different directories) is keyed by its order of appearance.
function read_table(path; basename_keys = false)
    header = ""
    rows = Dict{Tuple{String, Int}, Tuple{Float64, Float64}}()
    seen = Dict{String, Int}()
    for line in eachline(path)
        if startswith(line, HEADER_PREFIX)
            header = line
        elseif !startswith(line, "file\t") && !isempty(line)
            cols = split(line, '\t')
            cols[1] = basename_keys ? basename(cols[1]) : cols[1]
            k = seen[cols[1]] = get(seen, cols[1], 0) + 1
            rows[(String(cols[1]), k)] = (parse(Float64, cols[3]), parse(Float64, cols[4]))
        end
    end
    return header, rows
end

function compare_tables(before_path, after_path)
    legacy(p) = any(
        l -> !startswith(l, HEADER_PREFIX) && !startswith(l, "file\t") &&
             !isempty(l) && !occursin('/', first(split(l, '\t'))),
        eachline(p))
    bn = legacy(before_path) || legacy(after_path)
    hb, before = read_table(before_path; basename_keys = bn)
    ha, after = read_table(after_path; basename_keys = bn)
    println("before: ", hb)
    println("after:  ", ha)
    keys_all = union(keys(before), keys(after))
    selfs(d, k) = haskey(d, k) ? d[k][2] : NaN
    table = [(k, selfs(before, k), selfs(after, k)) for k in keys_all]
    delta(r) = (isnan(r[2]) ? 0.0 : -r[2]) + (isnan(r[3]) ? 0.0 : r[3])
    sort!(table; by = r -> abs(delta(r)), rev = true)
    println()
    @printf("  %9s %9s %9s  %s\n", "before_s", "after_s", "delta_s", "file")
    for r in table
        fmt(x) = isnan(x) ? @sprintf("%9s", "-") : @sprintf("%9.2f", x)
        @printf("  %s %s %+9.2f  %s\n", fmt(r[2]), fmt(r[3]), delta(r), r[1][1])
    end
    return nothing
end

if "--compare" in ARGS
    length(ARGS) == 3 || error("usage: julia test/timing.jl --compare before.tsv after.tsv")
    compare_tables(ARGS[2], ARGS[3])
    exit(0)
end

const GROUP = get!(ENV, "BRAMBLE_TEST_GROUP", "unit")
const TSV = get(ENV, "BRAMBLE_TIMING_TSV",
    joinpath(tempdir(), "bramble-test-timings-$(GROUP).tsv"))
const RUNTESTS = get(ENV, "BRAMBLE_TIMING_RUNTESTS", joinpath(TEST_DIR, "runtests.jl"))
const SKIP_SETUP = get(ENV, "BRAMBLE_TIMING_SKIP_SETUP", "0") == "1"
ENV["BRAMBLE_TEST_TRACE"] = "1"

Base.JLOptions().check_bounds == 1 ||
    @warn "Run with --check-bounds=yes to reuse the package images Pkg.test builds."

t_inst = t_pre = 0.0
if !SKIP_SETUP
    Pkg.activate(TEST_DIR; io = devnull)
    t_inst = @elapsed Pkg.instantiate(; io = devnull)
    t_pre = @elapsed Pkg.precompile()
end
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

# Each top-level testset of a file is recorded into its parent when it finishes (`Test.finish`
# calls `record(parent, child)`), which is the one point where its compile share can be read: the
# counter now, minus the counter at the previous top-level testset. The method below is
# `Test`'s own `record` for a parent and child that are both `DefaultTestSet`s, with the root
# case added, so the hook is a copy of the original body plus the read.
const ROOT_TS = Ref{Any}(nothing)
const LAST_COMPILE = Ref(c_start)
const TESTSET_ROWS = Tuple{String, String, Float64, Float64}[]

# `TestUtils` (loaded by the suite, so looked up late) keeps the stack of files being included
# with the testset current at each include: a child recorded into that testset is a top-level
# testset of that file. Without it, only the root's direct children count, with no file.
function include_stack()
    return isdefined(Main, :TestUtils) ? Main.TestUtils.INCLUDE_STACK :
           Tuple{String, Any}[]
end
INCLUDE_FILE() = isempty(include_stack()) ? "" : include_stack()[end][1]
function is_file_top_level(ts)
    stack = include_stack()
    return isempty(stack) ? ts === ROOT_TS[] : ts === stack[end][2]
end

function Test.record(ts::Test.DefaultTestSet, t::Test.AbstractTestSet)
    @lock ts.results_lock push!(ts.results, t)
    if is_file_top_level(ts)
        now = Base.cumulative_compile_time_ns()[1] / 1e9
        wall = hasfield(typeof(t), :time_end) ? t.time_end - t.time_start : NaN
        name = hasfield(typeof(t), :description) ? string(t.description) : string(typeof(t))
        push!(TESTSET_ROWS,
            (INCLUDE_FILE(), replace(name, r"[\t\n]" => " "), wall, now - LAST_COMPILE[]))
        LAST_COMPILE[] = now
    end
    return t
end

# One root testset around the suite, so its result tree can be walked afterwards.
root = Test.DefaultTestSet("timing root")
ROOT_TS[] = root
# Julia 1.13 keeps the current testset in scoped values; 1.12 has a stack.
function with_root(f, root)
    if isdefined(Test, :CURRENT_TESTSET)
        Base.ScopedValues.@with Test.CURRENT_TESTSET => root Test.TESTSET_DEPTH => 1 f()
    else
        Test.push_testset(root)
        try
            f()
        finally
            Test.pop_testset()
        end
    end
end

t_run = @elapsed try
    with_root(() -> Base.include(Main, RUNTESTS), root)
catch err
    @warn "Suite finished with errors" exception = err
end
total_compile = Base.cumulative_compile_time_ns()[1] / 1e9 - c_start

redirect_stdout(ORIG_STDOUT)
close(wr)
wait(reader)

# Nested `▸`/`✓` trace pairs give inclusive times; self time subtracts the children.
rows = Tuple{String, Int, Float64, Float64, Float64}[]
stack = Vector{Any}[]
for line in LINES
    if (m = match(r"^▸ \s*(\S+)$", line)) !== nothing
        push!(stack, Any[m[1], 0.0])
    elseif (m = match(r"^✓ \s*(\S+)\s+([\d.]+) s", line)) !== nothing
        wall = parse(Float64, m[2])
        compile = parse(Float64, match(r"([\d.]+) s compile", line)[1])
        child = pop!(stack)[2]
        isempty(stack) || (stack[end][2] += wall)
        push!(rows, (m[1], length(stack), wall, wall - child, compile))
    end
end
sort!(rows; by = r -> r[4], rev = true)

# Every nested testset counts, the tops included. Passes are only a counter on their set;
# failures, errors and broken tests are stored as results.
# Other testset types (Supposition's report) may lack either field: they hold no nested
# sets, and their tests are what `Test.get_test_counts` reports, else their results, else one.
function foreign_tests(ts)
    try
        c = Test.get_test_counts(ts)
        n = c.passes + c.fails + c.errors + c.broken
        n > 0 && return n
    catch
    end
    hasproperty(ts, :results) && ts.results isa AbstractVector && !isempty(ts.results) &&
        return length(ts.results)
    return 1
end

function count_tree(ts)
    (hasproperty(ts, :n_passed) && hasproperty(ts, :results)) || return 0, foreign_tests(ts)
    nsets, ntests = 0, ts.n_passed
    for r in ts.results
        if r isa Test.AbstractTestSet
            n, t = count_tree(r)
            nsets += 1 + n
            ntests += t
        else
            ntests += 1
        end
    end
    return nsets, ntests
end
n_testsets, n_tests = count_tree(root)

# The root never finishes, so a failing testset does not throw at the end of the suite.
function has_failures(ts)
    if !(hasproperty(ts, :n_passed) && hasproperty(ts, :results))
        return try
            c = Test.get_test_counts(ts)
            c.fails + c.errors > 0
        catch
            false
        end
    end
    return any(r -> r isa Test.AbstractTestSet ? has_failures(r) :
                    r isa Union{Test.Fail, Test.Error}, ts.results)
end
has_failures(root) && @warn "Suite finished with errors"

commit = try
    readchomp(pipeline(`git -C $TEST_DIR rev-parse --short HEAD`; stderr = devnull))
catch
    "unknown"
end

open(TSV, "w") do io
    @printf(io, "# commit=%s julia=%s threads=%d group=%s files=%d testsets=%d tests=%d ",
        commit, VERSION, Threads.nthreads(), GROUP, length(rows), n_testsets, n_tests)
    @printf(io, "wall_s=%.1f compile_s=%.1f (summed over threads)\n", t_run, total_compile)
    println(io, "file\tdepth\twall_s\tself_s\tcompile_s")
    for (path, depth, wall, self, compile) in rows
        @printf(io, "%s\t%d\t%.2f\t%.2f\t%.2f\n", path, depth, wall, self, compile)
    end
end

open(TSV * ".testsets", "w") do io
    println(io, "# top-level testsets; compile_s is summed over threads, and is the compile",
        " since the previous top-level testset ended")
    println(io, "file\ttestset\twall_s\tcompile_s")
    for (file, name, wall, compile) in TESTSET_ROWS
        @printf(io, "%s\t%s\t%.2f\t%.2f\n", file, name, wall, compile)
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
@printf("  %8s %8s %10s  %s\n", "self_s", "wall_s", "compile_s", "file")
for (path, _, wall, self, compile) in first(rows, 25)
    @printf("  %8.1f %8.1f %10.1f  %s\n", self, wall, compile, path)
end
println("\nFull table: ", TSV, " (testsets: ", TSV * ".testsets", ")")
