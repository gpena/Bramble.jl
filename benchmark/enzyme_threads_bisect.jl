# Bisects the plain `Threads.@threads` Enzyme failure behind the threaded AD path
# (#369, plan v4-2-0-followup, S2.1). The reproducer has no Bramble in it: a bare loop
# `y[i] = x[i]^2` differentiated in reverse mode by `Enzyme.autodiff`, under three
# schedules: `serial` (the control), `dynamic` (`Threads.@threads`, the planning failure)
# and `static` (`Threads.@threads :static`, the schedule of Bramble's `_static_*` helpers).
#
# Each Enzyme version gets a fresh environment (`setup`: `Pkg.add(; version)`) and then one
# child process per schedule (`run`), so a crash or a load failure is recorded and never
# inherited. The endpoints are 0.13.204, the version #369 ran (Enzyme in the session
# transcripts of 2026-09-14 to 09-25, before the issue was filed on 2026-09-26), and
# 0.13.216, the current one. The driver bisects between them on the `dynamic` status. If
# both fail it samples older versions, newest first, for one that works, then bisects
# between that and 0.13.204.
#
#     .claude/scripts/julia_slot.sh 2 julia --startup-file=no --threads=2 \
#         benchmark/enzyme_threads_bisect.jl
#
# The outer `julia_slot.sh` holds the two threads and the children run inside them, not
# through the slot wrapper again: a second wait for two slots never ends on a 3-slot pool.
# Julia 1.13.0 is not installed (`juliaup status`), so the run covers 1.13.3 only and the
# meta says so.
#
# Statuses: `ok` (the gradient is `2x`), `wrong` (it ran, the value differs), `throws`
# (the `autodiff` call threw; the first line of the message is kept), `unresolvable` (Pkg
# could not add or precompile the version, or Enzyme did not load), `crash` (the child died
# by signal, exited with an error and no result, or ran past the timeout).
# Writes benchmark/results/enzyme_threads_bisect.toml after every probe.

using Pkg
using TOML
using Dates

include(joinpath(@__DIR__, "results_io.jl"))

const SCRIPT = @__FILE__
const RESULTS = joinpath(@__DIR__, "results", "enzyme_threads_bisect.toml")
const FORMS = ("serial", "dynamic", "static")
const TIMEOUT_S = 1800.0
const KEEP = ("Enzyme", "Enzyme_jll", "GPUCompiler", "LLVM", "EnzymeCore")
const ENDPOINTS = (204, 216)
const ANCHORS = (198, 190, 180, 150, 100, 0)
const JULIA_1_12 = joinpath(
    homedir(), ".julia", "juliaup", "julia-1.12.7+0.aarch64.apple.darwin14",
    "Julia-1.12.app", "Contents", "Resources", "julia", "bin", "julia"
)

function sq_serial!(y, x)
    for i in eachindex(y)
        y[i] = x[i]^2
    end
    return nothing
end

function sq_dynamic!(y, x)
    Threads.@threads for i in eachindex(y)
        y[i] = x[i]^2
    end
    return nothing
end

function sq_static!(y, x)
    Threads.@threads :static for i in eachindex(y)
        y[i] = x[i]^2
    end
    return nothing
end

const KERNELS = Dict(
    "serial" => sq_serial!, "dynamic" => sq_dynamic!, "static" => sq_static!
)

# Child side ------------------------------------------------------------------------------

oneline(s) = first(replace(s, r"[\t\r\n]+" => " "), 200)

function report(status::AbstractString, message::AbstractString)
    println("RESULT\t", status, "\t", oneline(message))
    return nothing
end

# Adds Enzyme at `version` to the active environment and records what Pkg resolved for it.
function child_setup(version::AbstractString)
    env = dirname(Base.active_project())
    try
        Pkg.add(Pkg.PackageSpec(name = "Enzyme", version = version))
        deps = Dict{String, String}()
        for info in values(Pkg.dependencies())
            info.name in KEEP && (deps[info.name] = string(info.version))
        end
        open(io -> TOML.print(io, deps), joinpath(env, "resolved.toml"), "w")
        report("ok", "resolved")
    catch e
        report("unresolvable", sprint(showerror, e))
    end
    return nothing
end

# Runs one schedule of the reproducer in the active environment.
function child_run(form::AbstractString)
    try
        Core.eval(Main, :(using Enzyme))
    catch e
        report("unresolvable", sprint(showerror, e))
        return nothing
    end
    # The binding `Main.Enzyme` was created after this function was compiled: call the
    # reproducer through `invokelatest` so it sees the newest world.
    return Base.invokelatest(reproduce, form)
end

function reproduce(form::AbstractString)
    E = Main.Enzyme
    Threads.nthreads() == 2 || error("expected 2 threads, got $(Threads.nthreads())")
    kernel = KERNELS[form]
    x = collect(1.0:100.0)
    dx = zeros(100)
    y = zeros(100)
    dy = ones(100)
    try
        E.autodiff(E.Reverse, kernel, E.Const, E.Duplicated(y, dy), E.Duplicated(x, dx))
    catch e
        report("throws", sprint(showerror, e))
        return nothing
    end
    gap = maximum(abs, dx .- 2 .* x)
    report(isapprox(dx, 2 .* x; rtol = 1.0e-12) ? "ok" : "wrong", "max |dx - 2x| = $gap")
    return nothing
end

# Driver side -----------------------------------------------------------------------------

# Kills `pid` and its descendants (Pkg's precompile workers), by pid, so none outlives a
# timeout.
function kill_tree(pid::Integer)
    kids = split(read(ignorestatus(`pgrep -P $pid`), String))
    foreach(k -> kill_tree(parse(Int, k)), kids)
    run(ignorestatus(`kill -KILL $pid`))
    return nothing
end

# Runs `cmd` with its output in `log`; returns `(:timeout, 0)`, `(:signal, sig)` or
# `(:exit, code)`.
function run_limited(cmd::Cmd, log::AbstractString, timeout::Real)
    timed_out = Ref(false)
    proc = run(pipeline(cmd; stdout = log, stderr = log); wait = false)
    timer = Timer(timeout) do _
        timed_out[] = true
        kill_tree(getpid(proc))
    end
    wait(proc)
    close(timer)
    timed_out[] && return (:timeout, 0)
    proc.termsignal != 0 && return (:signal, proc.termsignal)
    return (:exit, proc.exitcode)
end

# One child julia on `args`, in the environment `env`; its status is the RESULT line it
# printed if it exited cleanly, else `crash`.
function child(
        args::Vector{String}, env::AbstractString, log::AbstractString, exe::Cmd
)
    cmd = addenv(
        `$exe --startup-file=no --threads=2 $SCRIPT $args`,
        "JULIA_PROJECT" => env, "JULIA_LOAD_PATH" => "@:@stdlib",
        "JULIA_NUM_PRECOMPILE_TASKS" => "2"
    )
    t0 = time()
    state = run_limited(cmd, log, TIMEOUT_S)
    seconds = time() - t0
    text = isfile(log) ? read(log, String) : ""
    results = collect(eachmatch(r"^RESULT\t(\S+)\t(.*)$"m, text))
    if state == (:exit, 0) && !isempty(results)
        return (String(results[end][1]), String(results[end][2]), seconds)
    end
    why = state[1] == :timeout ? "timeout after $TIMEOUT_S s" :
          state[1] == :signal ? "killed by signal $(state[2])" : "exit code $(state[2])"
    tail = isempty(text) ? "" : last(split(strip(text), '\n'))
    return ("crash", "$why; last line: $tail", seconds)
end

# The Enzyme version's environment, then one run row per schedule.
const WORK = Ref("")
const ROWS = Dict{String, Any}[]
const PROBED = Dict{Int, Dict{String, String}}()

# Runs the setup and the three schedules of Enzyme `version` on the Julia `exe` (version
# `jver`), appends one row per schedule and returns the status per schedule.
function probe_on(version::AbstractString, exe::Cmd, jver::AbstractString)
    env = joinpath(WORK[], "env-$version-$jver")
    mkpath(env)
    println("probe Enzyme $version on Julia $jver")
    setup_status, setup_msg, _ = child(
        ["setup", version], env, joinpath(WORK[], "setup-$version-$jver.log"), exe
    )
    resolved = setup_status == "ok" ? TOML.parsefile(joinpath(env, "resolved.toml")) :
               Dict{String, Any}()
    statuses = Dict{String, String}()
    for form in FORMS
        if setup_status == "ok"
            status, message, seconds = child(
                ["run", form], env, joinpath(WORK[], "run-$version-$jver-$form.log"), exe
            )
        else
            status, message, seconds = setup_status, setup_msg, 0.0
        end
        statuses[form] = status
        push!(
            ROWS, Dict{String, Any}(
                "enzyme" => version, "julia" => jver, "schedule" => form,
                "status" => status, "message" => message, "seconds" => seconds,
                "enzyme_jll" => get(resolved, "Enzyme_jll", ""),
                "gpucompiler" => get(resolved, "GPUCompiler", ""),
                "llvm" => get(resolved, "LLVM", ""),
                "enzymecore" => get(resolved, "EnzymeCore", "")
            )
        )
        println("  ", form, ": ", status, " ", oneline(message))
    end
    return statuses
end

probe(k::Int) = get!(PROBED, k) do
    probe_on("0.13.$k", Base.julia_cmd(), string(VERSION))
end

# The reproducer's verdict on one version: the `dynamic` schedule gives the ok/bad split.
good(k::Int) = probe(k)["dynamic"] == "ok"

# Given good(a) != good(b), returns an adjacent pair (a, b) with good(a) != good(b).
function bisect(a::Int, b::Int)
    ga = good(a)
    while b - a > 1
        m = (a + b) ÷ 2
        good(m) == ga ? (a = m) : (b = m)
    end
    return a, b
end

function juliaup_versions()
    out = read(ignorestatus(`juliaup status`), String)
    return sort(unique(m.match for m in eachmatch(r"\d+\.\d+\.\d+", out)))
end

function write_results(bisect_info::Dict{String, Any})
    meta = results_meta(basename(SCRIPT), false)
    sha = Cmd(`git hash-object $SCRIPT`; dir = dirname(SCRIPT))
    meta["script_sha"] = strip(read(sha, String))
    meta["juliaup_versions"] = juliaup_versions()
    meta["julia_1_13_0"] = "1.13.0" in meta["juliaup_versions"] ? "installed" :
                           "not installed (juliaup status); not run"
    data = Dict{String, Any}("meta" => meta, "bisect" => bisect_info, "run" => ROWS)
    mkpath(dirname(RESULTS))
    open(io -> TOML.print(io, data; sorted = true), RESULTS, "w")
    return RESULTS
end

function main()
    WORK[] = mktempdir()
    info = Dict{String, Any}("endpoints" => ["0.13.$(k)" for k in ENDPOINTS])
    write_results(info)
    lo, hi = ENDPOINTS
    if good(lo) != good(hi)
        a, b = bisect(lo, hi)
        info["reason"] = "endpoints differ on dynamic: bisected"
    elseif good(lo)
        a, b = lo, hi
        info["reason"] = "both endpoints ok on dynamic: no boundary in the window"
    else
        info["reason"] = "both endpoints fail on dynamic: sampling older versions"
        found = nothing
        for k in ANCHORS
            if good(k)
                found = k
                break
            end
        end
        if found === nothing
            a, b = lo, hi
            info["reason"] = "no sampled 0.13.x version works on dynamic"
        else
            a, b = bisect(found, lo)
            info["reason"] = "bisected between the newest working sampled version and $lo"
        end
    end
    info["pair"] = ["0.13.$a", "0.13.$b"]
    info["pair_good"] = [good(a), good(b)]
    # #369's version fails on 1.13.3 too: run it on 1.12.7 to see whether 1.13 matters.
    if !good(ENDPOINTS[1]) && isfile(JULIA_1_12)
        probe_on("0.13.$(ENDPOINTS[1])", `$JULIA_1_12`, "1.12.7")
        info["julia_1_12_7"] = "0.13.$(ENDPOINTS[1]) run on 1.12.7 as an extra data point"
    end
    write_results(info)
    println("done: ", RESULTS)
    return nothing
end

if abspath(PROGRAM_FILE) == SCRIPT
    if length(ARGS) == 2 && ARGS[1] == "setup"
        child_setup(ARGS[2])
    elseif length(ARGS) == 2 && ARGS[1] == "run"
        child_run(ARGS[2])
    elseif isempty(ARGS)
        main()
    else
        error("usage: enzyme_threads_bisect.jl [setup VERSION | run FORM]")
    end
end
