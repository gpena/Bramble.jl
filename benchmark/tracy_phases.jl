#===========================================================================#
# Per-phase first-call and warm time of `form` and `assemble`, read off Tracy zones
# (gpena/Bramble.jl#915). Tracy.jl must not become a dependency of Bramble, so the zones
# live in a patch, benchmark/tracy_zones.patch, that this script applies to a throwaway
# worktree of HEAD; nothing in the repository's Project.toml, Manifest.toml or src/ changes.
#
# One run, in order:
#   1. `git worktree add --detach <scratch> HEAD`, then `git apply` of the patch there (it
#      adds Tracy to the worktree's Project.toml and wraps the phases in `@tracepoint`);
#   2. a temp environment with that worktree `dev`ed plus Tracy and TracyProfiler_jll, put
#      on `LOAD_PATH` of this process and used with `--project` by every child;
#   3. per row, `tracy-capture` listening on a random port, then one child process (a fresh
#      Julia) building the row, calling `form` and `assemble` once (the first call: it
#      includes compilation) and then `--warm` more times;
#   4. `tracy-csvexport -u` of each capture, twice (inclusive and `-e` self times), and the
#      zone durations summed per phase per call into the TOML table of the row;
#   5. the scratch worktree and the temp environment are removed.
#
# Rows are those of benchmark/trace_compile.jl: `elasticity` and `vector_laplacian` on a
# graded 5x5x5 mesh, `scalar_2d` on a graded 9x9 mesh. Plain Julia, Base and TOML only:
# launch it with no `--project` of its own.
#
#     julia --startup-file=no benchmark/tracy_phases.jl [--rows a,b] [--warm 3]
#         [--threads 1] [--workload on|off] [--out FILE] [--scratch DIR] [--patch FILE]
#         [--keep]
#
# `--workload off` sets Bramble's `precompile_workload` preference to false in the temp
# environment (`Bramble.PRECOMPILE_WORKLOAD`), so the first call also pays for what the
# workload compiles ahead of time. The default `on` is the library as shipped, where a form
# the workload covers has almost nothing left to compile at its first call.
#
# `--out FILE` merges a table per row into FILE (other rows already there are kept).
# `--keep` leaves the scratch worktree, the temp environment and the raw captures in place
# and prints their paths. The first run needs the depot to hold, or the network to fetch,
# Tracy, TracyProfiler_jll and the packages Bramble depends on.
#
# Every zone wraps its call as `Base.inferencebarrier(f)(args...)`. Without the barrier the
# inference of `form` and `assemble` reaches every callee before the body runs, so the
# whole first-call compile lands outside the zones (scalar_2d: 295 ms in `form`, 0.003 ms
# in its zones). With it a callee is compiled when its zone runs, so a zone's self time is
# the compile and run time of what only it reaches. The cost is that the callers see `Any`,
# and a warm figure carries one dynamic dispatch per zone.
#
# Tracy.jl 0.1.6's `Tracy.capture` calls `TracyProfiler_jll.capture`, which TracyProfiler_jll
# 0.13 does not define (its executables are `tracy_capture`, `tracy_csvexport`), so this
# script runs `tracy_capture` itself. The capture and the client library
# (LibTracyClient_jll) must share a major.minor version, or the capture never connects: the
# script stops before measuring when the temp environment resolves them apart.
#
# Per row `R` the TOML holds the call durations `R.calls.{form,assemble}_{first,warm}_s`
# (warm is the median), and per zone `R.phases.<zone>` its inclusive time `first_s` and
# `warm_s` (summed over the `form` and `assemble` call of that kind), the same without the
# zones nested in it as `first_self_s` and `warm_self_s`, and how often it ran in a call as
# `first_events` and `warm_events`. Zones nest (`allocate_system_matrix` holds
# `coordinate_pass`), so inclusive times of different phases do not add up, self times do.
# A zone no row reaches (`lower_sources` sits on the linear-form path only) reads 0.0.
# Timings of a run while other jobs use the machine are not measurements, so
# `R.loadavg_start` records the load.
#===========================================================================#

# --- Child: one fresh process, one row ------------------------------------- #

if "--child" in ARGS
    # --child <row> <warm>
    row = ARGS[2]
    nwarm = parse(Int, ARGS[3])
    import Tracy
    # The client listens from the moment Tracy loads. The capture does not retry a refused
    # connection, so the parent starts it only after this line.
    println("READY")
    flush(stdout)
    using Bramble
    import Bramble: change_points!

    # The points of an axis: the exponent differs per axis, so no two axes agree.
    graded(n, axis) = collect(range(0.0, 1.0; length = n)) .^ (1.2 + 0.2 * axis)

    function build(row)
        if row == "scalar_2d"
            n = (9, 9)
            Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), n)
            change_points!(Ωₕ, ntuple(d -> graded(n[d], d), 2))
            return gridspace(Ωₕ), (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))
        end
        n = (5, 5, 5)
        Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), n)
        change_points!(Ωₕ, ntuple(d -> graded(n[d], d), 3))
        W = gridspace(Ωₕ)^Val(3)
        row == "vector_laplacian" && return W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))
        row == "elasticity" || error("unknown row $row")
        μ, λ = 1.0, 2.0
        return W, (u, v) -> 2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₘₕ(u), divₘₕ(v))
    end

    # The `call:` zones bound each call, so the parent can tell which call a phase zone ran
    # in. Zones before the capture connects are not recorded, hence the wait. Defined by
    # `Core.eval` because `@tracepoint` needs `Tracy` loaded when its definition is lowered.
    Core.eval(Main, quote
        function measure(row, nwarm)
            Base.JLOptions().opt_level == 2 || error(
                "timing needs the default -O (2), got $(Base.JLOptions().opt_level)")
            W, fun = build(row)
            Tracy.wait_for_tracy(; timeout = 120.0)
            a = Tracy.@tracepoint "call:form:first" form(W, W, fun)
            A = Tracy.@tracepoint "call:assemble:first" assemble(a)
            for _ in 1:nwarm
                a = Tracy.@tracepoint "call:form:warm" form(W, W, fun)
                A = Tracy.@tracepoint "call:assemble:warm" assemble(a)
            end
            sleep(1)  # the client sends its queue to the capture before the process ends
            return size(A, 1)
        end
    end)

    println("ROWS ", Core.eval(Main, :(measure($row, $nwarm))))
    exit(0)
end

using TOML

const ALL_ROWS = ["elasticity", "vector_laplacian", "scalar_2d"]
const ZONE_NAME = r"@tracepoint \"([^\"]+)\""
const CAPTURE_TIMEOUT_S = 180.0

function arg(name, default)
    i = findfirst(==(name), ARGS)
    return i === nothing ? default : ARGS[i + 1]
end

function median(v)
    s = sort(v)
    isempty(s) && return 0.0
    return isodd(length(s)) ? s[(end + 1) ÷ 2] : (s[end ÷ 2] + s[end ÷ 2 + 1]) / 2
end

# --- The scratch worktree and the temp environment ------------------------------------ #

function make_worktree(root, scratch, patch)
    ispath(scratch) && error("$scratch exists; remove it (`git worktree remove --force`) or pass --scratch")
    run(`git -C $root worktree add --detach $scratch HEAD`)
    run(`git -C $scratch apply $patch`)
    return nothing
end

function remove_worktree(root, scratch)
    ispath(scratch) && run(ignorestatus(`git -C $root worktree remove --force $scratch`))
    run(ignorestatus(`git -C $root worktree prune`))
    return nothing
end

# A temp environment holding the patched Bramble (`dev`ed), Tracy and TracyProfiler_jll.
function make_environment(env, scratch, workload)
    workload == "off" && write(joinpath(env, "LocalPreferences.toml"),
        "[Bramble]\nprecompile_workload = false\n")
    setup = """
        using Pkg
        Pkg.develop(path = ARGS[1])
        Pkg.add(["Tracy", "TracyProfiler_jll"])
        Pkg.precompile()
        """
    run(`$(Base.julia_cmd()) --startup-file=no --project=$env -e $setup $scratch`)
    return nothing
end

# `major.minor` of `name` in the environment's Manifest.
function manifest_version(env, name)
    manifest = TOML.parsefile(joinpath(env, "Manifest.toml"))
    entry = manifest["deps"][name][1]
    return entry["version"]
end

majmin(v) = join(split(v, '.')[1:2], '.')

function check_versions(env)
    client = manifest_version(env, "LibTracyClient_jll")
    profiler = manifest_version(env, "TracyProfiler_jll")
    majmin(client) == majmin(profiler) || error(
        "LibTracyClient_jll $client and TracyProfiler_jll $profiler differ in major.minor: " *
        "the capture would not connect")
    return (; client, profiler)
end

# --- One row: a capture, a child, two csv exports --------------------------------------- #

function capture_row(env, row, threads, nwarm, tracefile)
    port = rand(20000:60000)
    capture = Base.invokelatest(() -> TracyProfiler_jll.tracy_capture())
    start_capture() = run(
        pipeline(`$capture -a 127.0.0.1 -p $port -o $tracefile -f`; stdout = devnull),
        wait = false)
    cap = proc = nothing
    try
        child = `$(Base.julia_cmd()) --project=$env --threads=$threads --startup-file=no
            $(abspath(@__FILE__)) --child $row $nwarm`
        proc = open(addenv(child, "TRACY_PORT" => string(port)), "r")
        readline(proc) == "READY" || error("the child for $row never reported READY")
        sleep(0.5)
        # The capture ends when the client disconnects. One that finds the client not yet
        # listening exits at once, so it is started again while the child runs.
        for attempt in 1:5
            cap = start_capture()
            sleep(1.0)
            (process_running(cap) || !process_running(proc)) && break
        end
        rest = read(proc, String)
        success(proc) || error("the child for $row failed:\n$rest")
        deadline = time() + CAPTURE_TIMEOUT_S
        while process_running(cap) && time() < deadline
            sleep(0.2)
        end
        process_running(cap) && error("tracy-capture still running $CAPTURE_TIMEOUT_S s after the child ended")
        success(cap) || error("tracy-capture failed for $row")
    finally
        cap !== nothing && process_running(cap) && kill(cap)
        proc !== nothing && process_running(proc) && kill(proc)
    end
    isfile(tracefile) || error("no capture file $tracefile")
    return nothing
end

# One zone event per line: name, source file, source line, start (ns), duration (ns), thread,
# value (empty).
# `self` asks for the time outside the zones nested in it.
function export_events(tracefile; self)
    exe = Base.invokelatest(() -> TracyProfiler_jll.tracy_csvexport())
    cmd = self ? `$exe -u -e $tracefile` : `$exe -u $tracefile`
    events = NamedTuple{(:name, :start, :dur), Tuple{String, Int, Int}}[]
    for (i, line) in enumerate(eachline(cmd))
        i == 1 && continue  # header
        f = split(line, ',')
        length(f) >= 7 || continue
        push!(events, (name = String(f[1]), start = parse(Int, f[end - 3]),
            dur = parse(Int, f[end - 2])))
    end
    return events
end

# --- Aggregation ---------------------------------------------------------------------- #

# The calls, in time order: (kind, which, start, stop).
function calls_of(events)
    calls = [(; kind = split(e.name, ':')[2], which = split(e.name, ':')[3],
                 start = e.start, stop = e.start + e.dur)
             for e in events if startswith(e.name, "call:")]
    sort!(calls; by = c -> c.start)
    return calls
end

# Per call, the summed duration and the event count of each zone that started inside it.
function sums_per_call(events, calls, zones)
    sums = [Dict(z => 0.0 for z in zones) for _ in calls]
    counts = [Dict(z => 0 for z in zones) for _ in calls]
    for e in events
        e.name in zones || continue
        k = findfirst(c -> c.start <= e.start <= c.stop, calls)
        k === nothing && continue
        sums[k][e.name] += e.dur / 1.0e9
        counts[k][e.name] += 1
    end
    return sums, counts
end

# Phase `z` over the calls of one kind of `which`: summed over `form` and `assemble` for
# `first`; for `warm`, the median over the warm rounds (each round a form and an assemble).
function phase_figure(calls, per_call, which, z)
    idx = [k for k in eachindex(calls) if calls[k].which == which]
    which == "first" && return sum(per_call[k][z] for k in idx)
    forms = [k for k in idx if calls[k].kind == "form"]
    assems = [k for k in idx if calls[k].kind == "assemble"]
    return median([per_call[f][z] + per_call[a][z] for (f, a) in zip(forms, assems)])
end

function call_figure(calls, kind, which)
    d = [(c.stop - c.start) / 1.0e9 for c in calls if c.kind == kind && c.which == which]
    return which == "first" ? sum(d) : median(d)
end

function summarize(row, tracefile, zones, threads, nwarm)
    incl = export_events(tracefile; self = false)
    self = export_events(tracefile; self = true)
    calls = calls_of(incl)
    length(calls) == 2 + 2 * nwarm || error(
        "$row: found $(length(calls)) `call:` zones, expected $(2 + 2 * nwarm): the capture " *
        "lost zones (is the patch applied?)")
    s_incl, n_incl = sums_per_call(incl, calls, zones)
    s_self, _ = sums_per_call(self, calls, zones)
    phases = Dict{String, Any}()
    for z in zones
        phases[z] = Dict{String, Any}(
            "first_s" => phase_figure(calls, s_incl, "first", z),
            "warm_s" => phase_figure(calls, s_incl, "warm", z),
            "first_self_s" => phase_figure(calls, s_self, "first", z),
            "warm_self_s" => phase_figure(calls, s_self, "warm", z),
            "first_events" => sum(n_incl[k][z] for k in eachindex(calls)
            if calls[k].which == "first"),
            "warm_events" => sum(n_incl[k][z] for k in eachindex(calls)
            if calls[k].which == "warm") ÷ nwarm)
    end
    return Dict{String, Any}(
        "threads" => threads, "warm_calls" => nwarm,
        "loadavg_start" => collect(Sys.loadavg()),
        "calls" => Dict{String, Any}(
            "form_first_s" => call_figure(calls, "form", "first"),
            "assemble_first_s" => call_figure(calls, "assemble", "first"),
            "form_warm_s" => call_figure(calls, "form", "warm"),
            "assemble_warm_s" => call_figure(calls, "assemble", "warm")),
        "phases" => phases)
end

function print_row(row, r)
    println("\n== $row (threads $(r["threads"]), $(r["warm_calls"]) warm calls)")
    c = r["calls"]
    ms(x) = round(1000x; digits = 3)
    println("form     first ", ms(c["form_first_s"]), " ms   warm ", ms(c["form_warm_s"]), " ms")
    println("assemble first ", ms(c["assemble_first_s"]), " ms   warm ",
        ms(c["assemble_warm_s"]), " ms")
    println(rpad("phase (ms)", 28), lpad("first", 12), lpad("first_self", 12), lpad("warm", 12),
        lpad("warm_self", 12))
    for (z, p) in sort!(collect(r["phases"]); by = kv -> -kv[2]["first_s"])
        println(rpad(z, 28), lpad(ms(p["first_s"]), 12), lpad(ms(p["first_self_s"]), 12),
            lpad(ms(p["warm_s"]), 12), lpad(ms(p["warm_self_s"]), 12))
    end
    return nothing
end

# --- Main ------------------------------------------------------------------------------- #

function main()
    rows = split(arg("--rows", join(ALL_ROWS, ",")), ',')
    all(in(ALL_ROWS), rows) || error("unknown row in $rows; known: $ALL_ROWS")
    nwarm = parse(Int, arg("--warm", "3"))
    nwarm >= 1 || error("--warm needs at least 1")
    threads = parse(Int, arg("--threads", "1"))
    out = arg("--out", nothing)
    workload = arg("--workload", "on")
    workload in ("on", "off") || error("--workload takes on or off, got $workload")
    keep = "--keep" in ARGS
    root = readchomp(`git -C $(@__DIR__) rev-parse --show-toplevel`)
    scratch = abspath(arg("--scratch", joinpath(dirname(root), "tracy-scratch")))
    patch = abspath(arg("--patch", joinpath(@__DIR__, "tracy_zones.patch")))
    zones = sort!(unique([m[1] for m in eachmatch(ZONE_NAME, read(patch, String))]))
    commit = readchomp(`git -C $root rev-parse HEAD`)
    env = mktempdir(; cleanup = false)
    work = mktempdir(; cleanup = false)
    result = Dict{String, Any}()
    try
        make_worktree(root, scratch, patch)
        make_environment(env, scratch, workload)
        versions = check_versions(env)
        push!(LOAD_PATH, env)
        Core.eval(Main, :(using TracyProfiler_jll))
        for row in rows
            println("\n-- $row: capture and child")
            tracefile = joinpath(work, "$row.tracy")
            capture_row(env, row, threads, nwarm, tracefile)
            result[row] = summarize(row, tracefile, zones, threads, nwarm)
            print_row(row, result[row])
        end
        result["meta"] = Dict{String, Any}(
            "commit" => commit, "precompile_workload" => workload, "julia" => string(VERSION), "zones" => zones,
            "libtracyclient_jll" => versions.client,
            "tracyprofiler_jll" => versions.profiler,
            "machine" => string(Sys.MACHINE, ", ", Sys.CPU_THREADS, " threads"))
        if out !== nothing
            merged = isfile(out) ? TOML.parsefile(out) : Dict{String, Any}()
            merge!(merged, result)
            open(io -> TOML.print(io, merged), out, "w")
            println("\nwrote $out")
        end
    finally
        if keep
            println("\nkept: worktree $scratch, environment $env, captures $work")
        else
            remove_worktree(root, scratch)
            rm(env; recursive = true, force = true)
            rm(work; recursive = true, force = true)
        end
    end
    return nothing
end

main()
