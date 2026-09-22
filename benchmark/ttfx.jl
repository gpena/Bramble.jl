#===========================================================================#
# TTFX (time-to-first-X) harness -- gpena/Bramble.jl#283, #284:
#
#     "TTFX drops >= 150 ms pde_solve, >= 120 ms sd2, >= 80 ms assemble_add!;
#      precompile overhead < +0.8 s"                                  (#283)
#     "TTFX drops >= 250 ms SuiteSparse, >= 200 ms Sparspak, >= 500 ms
#      Kronecker"                                                     (#284)
#
# Plain Julia, Base and stdlib only. This file is launched with no
# `--project` of its own (`julia --startup-file=no benchmark/ttfx.jl ...`)
# because it orchestrates *child* `julia` processes against two other trees
# (`--before`/`--after`), and must not itself compete for either tree's own
# package environment or precompile cache.
#
# ## What "first call" means here, and why a fresh depot
#
# A first call is a first call because nothing about it has been compiled
# yet *in this process's depot* -- the pkgimage cache Julia writes to
# `~/.julia/compiled/v1.<minor>/Bramble/…` after the first `using Bramble`
# in a depot. Measuring TTFX against the developer's own `~/.julia` would
# measure whatever state that depot happens to be in, not the thing the
# issue asks about ("wiped ~/.julia/compiled/v1.12/Bramble"). This harness
# gets the same effect without deleting anything real: every timed process
# runs under a throwaway `mktempdir()` depot layered in *front* of
# `~/.julia` (`JULIA_DEPOT_PATH="<fresh>:<~/.julia>"`), so Bramble's own
# pkgimage is always missing while every dependency (already installed under
# the real depot) is not -- `JULIA_PKG_OFFLINE=true` makes that assumption
# load-bearing rather than silently falling back to a registry hit.
#
# ## Medians, not means (bramble-verification §2)
#
# Every timed quantity here is the median of `--runs` (default 3) *separate*
# process launches, never a mean and never a single sample -- "separate
# process launches are their own noise, distinct from separate commits" per
# bramble-verification §2, which measured 10-30%+ swings between launches of
# the very same command. A single before/after pair would not distinguish a
# real reduction from that noise; medians of repeated launches do.
#
# ## The load gate
#
# A one-minute load average above `Sys.CPU_THREADS/2` at start is polled
# every 30 s for up to 10 minutes (bramble-verification §9: this tree's own
# worktrees routinely run concurrent `Pkg.test`/benchmark/doc-build sessions,
# and TTFX numbers taken under that contention are exactly the kind of
# same-run-not-across-launches noise §2 warns about). If load never settles,
# the run proceeds labelled "MEASURED UNDER LOAD" rather than refusing
# outright -- a TTFX measurement under load is degraded evidence, not
# meaningless evidence, and the integrator (plan gate M) re-runs alone on a
# quiet machine regardless.
#
# ## Invocation
#
#     julia --startup-file=no benchmark/ttfx.jl \
#         --before <tree> --after <tree> [--runs N] [--threads T] [--dry-run]
#
# `<tree>` is a Bramble.jl checkout -- a git worktree of the pre-change
# commit for `--before`, this repository for `--after`. `--dry-run` parses
# and validates arguments, prints the case list and the exact child
# commands, then exits without launching a single Julia child process.
#===========================================================================#

using Dates

# --- CLI ------------------------------------------------------------------- #

function _fail(msg::AbstractString)
    println(stderr, "ERROR: $msg")
    exit(1)
end

function _parse_args(args::Vector{String})
    before = nothing
    after = nothing
    runs = 3
    threads = 4
    dry_run = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--before"
            i += 1
            i <= length(args) || _fail("--before needs a value")
            before = args[i]
        elseif a == "--after"
            i += 1
            i <= length(args) || _fail("--after needs a value")
            after = args[i]
        elseif a == "--runs"
            i += 1
            i <= length(args) || _fail("--runs needs a value")
            runs = parse(Int, args[i])
        elseif a == "--threads"
            i += 1
            i <= length(args) || _fail("--threads needs a value")
            threads = parse(Int, args[i])
        elseif a == "--dry-run"
            dry_run = true
        else
            _fail("unknown argument: $a")
        end
        i += 1
    end
    return (; before, after, runs, threads, dry_run)
end

function _validate_tree(tree::AbstractString, label::AbstractString)
    isfile(joinpath(tree, "Project.toml")) || _fail("$label: $tree has no Project.toml")
    isfile(joinpath(tree, "test", "Project.toml")) ||
        _fail("$label: $tree has no test/Project.toml")
    return nothing
end

# --- Cases (thresholds are before-minus-after reductions, gpena/Bramble.jl#283/#284) --- #

struct TTFXCase
    name::String            # display name, matches the issue's own call shape
    project::Symbol         # :core (--project=<tree>) or :test (--project=<tree>/test)
    using_pkgs::Vector{String}   # extra `using X, Y` beyond `using Bramble`
    threshold_ms::Float64   # minimum before-after reduction required
    setup::String           # untimed Julia source that builds the fixture beyond FIXTURE
    call::String            # the expression timed with @elapsed
end

const CASES = TTFXCase[
    TTFXCase(
        "pde_solve(A, F)",
        :core,
        String[],
        150.0,
        "A = assemble(a_spd)\nF = assemble(l)",
        "pde_solve(A, F)"
    ),
    TTFXCase(
        "sd2(dv, v, u, nothing, 0.0)",
        :core,
        String[],
        120.0,
        "sd2 = semidiscretize_second_order(a, l; dirichlet = :boundary)\n" *
        "n = ndofs(Wₕ)\ndv = zeros(n)\nv = zeros(n)\nu = zeros(n)",
        "sd2(dv, v, u, nothing, 0.0)"
    ),
    TTFXCase(
        "assemble_add!(A, a, 0.5)",
        :core,
        String[],
        80.0,
        "A = assemble(a)",
        "assemble_add!(A, a, 0.5)"
    ),
    TTFXCase(
        "suitesparse_solve(A, F)",
        :test,
        ["SuiteSparse"],
        250.0,
        "A = assemble(a_spd)\nF = assemble(l)",
        "suitesparse_solve(A, F)"
    ),
    TTFXCase(
        "sparspak_solve(A, F)",
        :test,
        ["Sparspak"],
        200.0,
        "A = assemble(a_spd)\nF = assemble(l)",
        "sparspak_solve(A, F)"
    ),
    TTFXCase(
        "fdm_solve(a, F)",
        :test,
        ["Kronecker"],
        500.0,
        "F = assemble(l)",
        "fdm_solve(a_spd, F)"
    )
]

# Every case builds the same 2D fixture (issue-specified): a unit square, a Neumann-friendly
# form `a` and an SPD form `a_spd` (needed by the direct/factorized solvers and fdm_solve),
# and a constant right-hand side.
const FIXTURE = """
S = interval(0.0, 1.0) × interval(0.0, 1.0)
Ω = domain(S, :boundary => boundary_symbols(S))
Ωₕ = mesh(Ω, (8, 8), (false, false))
Wₕ = gridspace(Ωₕ)
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
a_spd = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
fₕ = Rₕ(Wₕ, x -> 1.0)
l = form(Wₕ, v -> innerₕ(fₕ, v))
"""

_project_for(tree::AbstractString, case::TTFXCase) = case.project === :core ? tree : joinpath(tree, "test")

function _case_using_line(case::TTFXCase)
    return isempty(case.using_pkgs) ? "using Bramble" :
           "using Bramble, " * join(case.using_pkgs, ", ")
end

# Generated into a temp dir (never the repo): a script that loads Bramble (plus the case's
# own extension package), builds the fixture and the case's own untimed setup, times only the
# named first call, and prints a single machine-readable marker line.
function _write_case_script(case::TTFXCase)
    dir = mktempdir()
    path = joinpath(dir, "case.jl")
    body = """
    $(_case_using_line(case))

    $FIXTURE
    $(case.setup)

    t = @elapsed $(case.call)
    println("TTFX_RESULT: ", t)
    """
    write(path, body)
    return path
end

# --- Depot / process plumbing ------------------------------------------------ #

function _depot_env(fresh_depot::AbstractString)
    home_depot = joinpath(homedir(), ".julia")
    sep = Sys.iswindows() ? ';' : ':'
    env = copy(ENV)
    env["JULIA_DEPOT_PATH"] = string(fresh_depot, sep, home_depot)
    env["JULIA_PKG_OFFLINE"] = "true"
    return env
end

_core_precompile_cmd(
    tree,
    threads
) = `julia --startup-file=no --threads=$threads --project=$tree -e "using Bramble"`
function _ext_precompile_cmd(tree, threads)
    test_project = joinpath(tree, "test")
    `julia --startup-file=no --threads=$threads --project=$test_project -e "using Bramble, SuiteSparse, Sparspak, Kronecker, SparseMatricesCSR"`
end
function _case_cmd(tree, case::TTFXCase, threads, script)
    project = _project_for(tree, case)
    `julia --startup-file=no --threads=$threads --project=$project $script`
end

function _median(xs::Vector{Float64})
    s = sort(xs)
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

# N fresh-depot launches of `using Bramble`, wall-timed from the orchestrator. Returns the
# median (seconds), every raw sample, and the depots used (the last one is reused downstream:
# it already has Bramble's own pkgimage compiled, so the extension-precompile measurement and
# every per-case run against this tree pay for that compile only once, not again).
function _time_core_precompile(tree::AbstractString, runs::Int, threads::Int)
    times = Float64[]
    depots = String[]
    for _ in 1:runs
        depot = mktempdir()
        push!(depots, depot)
        env = _depot_env(depot)
        cmd = setenv(_core_precompile_cmd(tree, threads), env)
        t = @elapsed run(pipeline(cmd; stdout = devnull, stderr = devnull))
        push!(times, t)
    end
    return _median(times), times, depots
end

# Once, informational only: the extra wall time of `using Bramble, SuiteSparse, Sparspak,
# Kronecker, SparseMatricesCSR` in a depot where core is already compiled (from
# `_time_core_precompile`'s last run). This call also leaves all four extensions precompiled
# in that depot, so the same depot doubles as the "warm depot" §2 of the contract asks for.
function _time_ext_precompile(tree::AbstractString, depot::AbstractString, threads::Int)
    env = _depot_env(depot)
    cmd = setenv(_ext_precompile_cmd(tree, threads), env)
    return @elapsed run(pipeline(cmd; stdout = devnull, stderr = devnull))
end

# N fresh processes against the tree's one warm depot. Each process builds the fixture
# untimed and times only the named first call with `@elapsed`, printing it as `TTFX_RESULT:
# <seconds>`; the orchestrator greps that line out of the child's stdout.
function _time_case(
        tree::AbstractString,
        case::TTFXCase,
        depot::AbstractString,
        threads::Int,
        runs::Int
)
    env = _depot_env(depot)
    times = Float64[]
    for i in 1:runs
        script = _write_case_script(case)
        cmd = setenv(_case_cmd(tree, case, threads, script), env)
        out = read(pipeline(cmd; stderr = devnull), String)
        m = match(r"TTFX_RESULT:\s*([0-9.eE+-]+)", out)
        m === nothing &&
            _fail("case \"$(case.name)\" on $tree (run $i) produced no TTFX_RESULT:\n$out")
        push!(times, parse(Float64, m.captures[1]))
    end
    return _median(times), times
end

# --- Load gate --------------------------------------------------------------- #

function _wait_for_load(
        threshold::Float64;
        max_wait_s::Float64 = 600.0,
        poll_s::Float64 = 30.0
)
    load1 = Sys.loadavg()[1]
    waited = 0.0
    while load1 >= threshold && waited < max_wait_s
        sleep(poll_s)
        waited += poll_s
        load1 = Sys.loadavg()[1]
    end
    return load1, load1 < threshold
end

function _batt_status()
    try
        return strip(read(`pmset -g batt`, String))
    catch
        return "pmset unavailable (non-macOS, or not found)"
    end
end

# --- Dry run ------------------------------------------------------------------ #

function _print_case_list()
    println("Cases (threshold = minimum before-after reduction):")
    for case in CASES
        project = case.project === :core ? "core" : "test"
        pkgs = isempty(case.using_pkgs) ? "" : " (+ using $(join(case.using_pkgs, ", ")))"
        println("  - $(case.name)  [$project$pkgs]  >= $(case.threshold_ms) ms")
    end
    println("  - precompile time (core), median of --runs  ->  after - before < +0.8 s")
    println("  - extension precompile time (info only, not gated)")
    return nothing
end

function _print_dry_run(opts)
    before = something(opts.before, "<before>")
    after = something(opts.after, "<after>")
    println("TTFX harness -- dry run")
    println()
    println("--before  : $(opts.before === nothing ? "(not given)" : opts.before)")
    println("--after   : $(opts.after === nothing ? "(not given)" : opts.after)")
    println("--runs    : $(opts.runs)")
    println("--threads : $(opts.threads)")
    println()
    _print_case_list()
    println()
    println(
        "Exact child commands (depot/script paths are placeholders -- built fresh per run):",
    )
    for (label, tree) in (("before", before), ("after", after))
        println(
            "  [$label] core precompile   : $(_core_precompile_cmd(tree, opts.threads))",
        )
        println(
            "           (env: JULIA_DEPOT_PATH=\"<fresh-depot>:$(joinpath(homedir(), ".julia"))\", JULIA_PKG_OFFLINE=true)",
        )
        println("  [$label] ext precompile    : $(_ext_precompile_cmd(tree, opts.threads))")
        for case in CASES
            println(
                "  [$label] $(case.name) : $(_case_cmd(tree, case, opts.threads, "<tmpdir>/case.jl"))",
            )
        end
    end
    return nothing
end

# --- Table printing ------------------------------------------------------------ #

_fmt_ms(t::Real) = string(round(t * 1000; digits = 1))
_fmt_s(t::Real) = string(round(t; digits = 3))

function _print_table_row(cols::Vector{String}, widths::Vector{Int})
    println(join((rpad(c, w) for (c, w) in zip(cols, widths)), " | "))
    return nothing
end

# --- Main ---------------------------------------------------------------------- #

function main()
    opts = _parse_args(collect(ARGS))

    if opts.dry_run
        opts.before !== nothing && _validate_tree(opts.before, "--before")
        opts.after !== nothing && _validate_tree(opts.after, "--after")
        _print_dry_run(opts)
        println("TTFX-HARNESS-OK")
        return
    end

    opts.before === nothing && _fail("--before is required (unless --dry-run)")
    opts.after === nothing && _fail("--after is required (unless --dry-run)")
    _validate_tree(opts.before, "--before")
    _validate_tree(opts.after, "--after")

    load_threshold = Sys.CPU_THREADS / 2
    load1, load_settled = _wait_for_load(load_threshold)
    under_load = !load_settled

    println("TTFX harness -- gpena/Bramble.jl#283, #284")
    under_load && println(
        "*** MEASURED UNDER LOAD (one-minute load average never settled below Sys.CPU_THREADS/2) ***",
    )
    println()
    println("Date          : $(Dates.now())")
    println(
        "Load average  : $(Sys.loadavg()) (threshold < $(round(load_threshold; digits = 2)) = Sys.CPU_THREADS/2 = $(Sys.CPU_THREADS)/2)",
    )
    println("Power         : $(_batt_status())")
    println("Julia version : $(VERSION)")
    println("Threads       : $(opts.threads)")
    println("Runs          : $(opts.runs)")
    println("--before      : $(opts.before)")
    println("--after       : $(opts.after)")
    println()

    println("Timing core precompile (--before)...")
    before_pre_med, before_pre_times, before_depots = _time_core_precompile(opts.before, opts.runs, opts.threads)
    println("Timing core precompile (--after)...")
    after_pre_med, after_pre_times, after_depots = _time_core_precompile(opts.after, opts.runs, opts.threads)

    println("Timing extension precompile, info only (--before)...")
    before_ext_extra = _time_ext_precompile(opts.before, before_depots[end], opts.threads)
    println("Timing extension precompile, info only (--after)...")
    after_ext_extra = _time_ext_precompile(opts.after, after_depots[end], opts.threads)

    before_warm_depot = before_depots[end]
    after_warm_depot = after_depots[end]

    case_results = Vector{NamedTuple}(undef, length(CASES))
    for (idx, case) in enumerate(CASES)
        println("Timing case: $(case.name)")
        b_med, b_times = _time_case(opts.before, case, before_warm_depot, opts.threads, opts.runs)
        a_med, a_times = _time_case(opts.after, case, after_warm_depot, opts.threads, opts.runs)
        case_results[idx] = (
            case = case,
            before = b_med,
            after = a_med,
            before_times = b_times,
            after_times = a_times
        )
    end

    println()
    println("Raw samples (seconds):")
    println("  precompile core  before: $before_pre_times")
    println("  precompile core  after : $after_pre_times")
    println("  precompile ext   before (single run, info only): $before_ext_extra")
    println("  precompile ext   after  (single run, info only): $after_ext_extra")
    for r in case_results
        println("  $(r.case.name)  before: $(r.before_times)")
        println("  $(r.case.name)  after : $(r.after_times)")
    end
    println()

    widths = [40, 12, 12, 14, 12, 8]
    println("Results:")
    _print_table_row(
        ["case", "before", "after", "reduction", "threshold", "PASS/FAIL"],
        widths
    )

    failed = String[]

    pre_delta_s = after_pre_med - before_pre_med
    pre_pass = pre_delta_s < 0.8
    pre_pass || push!(failed, "precompile time")
    _print_table_row(
        [
            "precompile (core, s)",
            _fmt_s(before_pre_med),
            _fmt_s(after_pre_med),
            _fmt_s(pre_delta_s),
            "< +0.8 s",
            pre_pass ? "PASS" : "FAIL"
        ],
        widths
    )

    _print_table_row(
        [
            "precompile (ext, s, info only)",
            _fmt_s(before_ext_extra),
            _fmt_s(after_ext_extra),
            _fmt_s(after_ext_extra - before_ext_extra),
            "info only",
            "-"
        ],
        widths
    )

    for r in case_results
        reduction_ms = (r.before - r.after) * 1000
        pass = reduction_ms >= r.case.threshold_ms
        pass || push!(failed, r.case.name)
        _print_table_row(
            [
                r.case.name,
                _fmt_ms(r.before),
                _fmt_ms(r.after),
                string(round(reduction_ms; digits = 1)),
                ">= $(r.case.threshold_ms) ms",
                pass ? "PASS" : "FAIL"
            ],
            widths
        )
    end

    println()
    if isempty(failed)
        println("TTFX-ALL-PASS")
    else
        println("TTFX-FAILS: " * join(failed, ", "))
    end
    return nothing
end

main()
