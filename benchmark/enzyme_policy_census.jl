# Which `_batch_*` hooks does each Enzyme case reach, and how does Enzyme do on each? (#369)
#
# A one-off census, run by hand, that writes benchmark/results/enzyme_policy_census.toml from
# the cases in test/ext/enzyme_policy_cases.jl. Every measurement runs in a child process, so
# a crash is recorded, not fatal:
#
#   1. a reach pass per hook: counting stub Enzyme rules on all 17 `_batch_*` hooks, then one
#      direct reverse gradient of the case under CpuPolyester, so `hits` names every hook the
#      builder's call enters (`reaches` is true when the case's own hook is among them).
#      Enzyme 0.13 cannot compile a custom rule on a hook whose argument is a by-value struct,
#      tuple or closure that mixes an array with isbits fields (`CallingConventionMismatchError`
#      before the stub runs; docs/notes/internals/enzyme_policies.md). When that error is the
#      stub's whole outcome, the pass is repeated as a primal call (no Enzyme) that counts the
#      hook entries through `Bramble._late`, and `reach_by` says "primal";
#   2. one status pass per mode (`reverse`, `di_reverse`, `forward`) under CpuPolyester, with
#      the real hook rules;
#   3. a serial control: the direct reverse gradient under CpuSerial.
#
# The upstream part runs two plain-Julia reproducers (`Polyester.@batch` and `Threads.@threads`,
# no Bramble) and records each with the Enzyme and Polyester versions. The notes page drafts
# them as issues for Enzyme.jl; nothing is filed from here.
#
# Enzyme is not in test/Project.toml. Each child loads it from a temporary environment whose
# Manifest.toml is a copy of the root one, so Enzyme resolves to the version the root project
# pins. The outer process only schedules, with one slot of two threads at a time:
#
#     .claude/scripts/julia_slot.sh 2 julia --startup-file=no --threads=1 \
#         benchmark/enzyme_policy_census.jl
#
# `--only <hook>` runs the children of one hook and prints them without writing the results
# file. The results file records `script_sha`, the git blob hash of this file: rerun the census
# after the last edit to it or to the harness, or the stored file is stale.

using TOML
using Dates

const ROOT = dirname(@__DIR__)
const SCRIPT = @__FILE__
const HARNESS = joinpath(ROOT, "test", "ext", "enzyme_policy_cases.jl")
const RESULTS = joinpath(ROOT, "benchmark", "results", "enzyme_policy_census.toml")
const THREADS = 2
const CELL_TIMEOUT = 1200
const ERROR_CHARS = 400
const ENZYME_UUID = "7da242da-08ed-463a-9acd-ee780be4f1d9"
const VERSIONED = ("Julia", "Enzyme", "Polyester", "ForwardDiff", "DifferentiationInterface",
    "Kronecker", "Bramble")

const HOOKS = (
    "_batch_dot", "_batch_dot_masked", "_batch_for!", "_batch_axis_for!",
    "_batch_scatter_for!", "_batch_bilinear_band_replay!", "_batch_bilinear_colour_replay!",
    "_batch_linear_colour_sweep!", "_batch_linear_band_sweep!", "_batch_difference_engine!",
    "_batch_average_engine!", "_batch_centered_average_engine!", "_batch_run_bands!",
    "_batch_mf_bands!", "_batch_broadcast!", "_batch_kron_lines!", "_batch_csr_spmv!")

# The planning reproducers. Each is a plain-Julia program: the temporary environment holds
# Enzyme only, and the macro is the one construct the entry is about.
const UPSTREAM_TEMPLATE = raw"""
using @IMPORTS
function sq!(y, x)
    @MACRO for i in eachindex(y)
        y[i] = x[i]^2
    end
    return nothing
end
x = collect(1.0:100.0); dx = zeros(100); y = zeros(100); dy = ones(100)
Enzyme.autodiff(Reverse, sq!, Const, Duplicated(y, dy), Duplicated(x, dx))
dx ≈ 2 .* x
"""

const UPSTREAM = (
    (construct = "Polyester.@batch", macro_text = "Polyester.@batch", imports = "Polyester, Enzyme"),
    (construct = "Threads.@threads", macro_text = "Threads.@threads", imports = "Enzyme"))

function upstream_snippet(u)
    return replace(UPSTREAM_TEMPLATE, "@MACRO" => u.macro_text, "@IMPORTS" => u.imports)
end

# ---------------------------------------------------------------------------------------
# A child: one measurement, run in its own process. The child's code is a quoted block
# evaluated into a module of its own, after the temporary environment is on the load path.
# ---------------------------------------------------------------------------------------

function enzyme_env()
    dir = mktempdir()
    write(joinpath(dir, "Project.toml"), "[deps]\nEnzyme = \"$ENZYME_UUID\"\n")
    cp(joinpath(ROOT, "Manifest.toml"), joinpath(dir, "Manifest.toml"))
    return dir
end

# Stub rules on every `_batch_*` hook: each counts its entries and then runs the hook
# unchanged, with a reverse rule that gives no gradient. They are defined only in the reach
# child, because a stub replaces the hook's derivative for the rest of its process.
function define_stubs!(mod)
    Core.eval(mod, :(const HITS = Dict{String, Int}()))
    for h in HOOKS
        T = typeof(getfield(Base.invokelatest(getfield, mod, :Bramble), Symbol(h)))
        Core.eval(mod, :(HITS[$h] = 0))
        Core.eval(mod, quote
            function EnzymeRules.augmented_primal(
                    config::RevConfig, func::Enzyme.Const{$T}, ::Type{RT}, args...
            ) where {RT}
                HITS[$h] += 1
                r = func.val(map(a -> a.val, args)...)
                primal = EnzymeRules.needs_primal(config) ? r : nothing
                shadow = EnzymeRules.needs_shadow(config) && r isa Number ? zero(r) : nothing
                return AugmentedReturn(primal, shadow, nothing)
            end
            function EnzymeRules.reverse(
                    config::RevConfig, func::Enzyme.Const{$T}, dret, tape, args...
            )
                return map(_ -> nothing, args)
            end
        end)
    end
    return nothing
end

clean(text) = first(filter(isprint, text), ERROR_CHARS)

# The first line of an error, then the method it names after "Failure within method:" and
# the first frame of its stack (the same summary as the harness's `_summary`).
function error_text(e)
    text = try
        sprint(showerror, e)
    catch
        return "$(typeof(e)) (its message could not be printed)"
    end
    lines = filter(!isempty, strip.(split(text, '\n')))
    parts = [first(lines)]
    i = findfirst(==("Failure within method:"), lines)
    i === nothing || push!(parts, "in " * first(split(lines[i + 1], "(")))
    i = findfirst(==("Stacktrace:"), lines)
    i === nothing || i + 2 > length(lines) ||
        push!(parts, "at " * join(lines[(i + 1):(i + 2)], " "))
    return clean(join(parts, " | "))
end

# The versions in this process, for the meta table and each upstream entry.
function versions_now()
    out = Dict{String, Any}()
    out["julia"] = string(VERSION)
    for (id, m) in Base.loaded_modules
        id.name in VERSIONED && (out[id.name] = string(pkgversion(m)))
    end
    return out
end

function write_cell(path, cell)
    open(path, "w") do io
        TOML.print(io, cell)
    end
end

# One measurement in the child's latest world age: the Polyester extension defines the
# policy constructors and the harness defines its cases only after the child started.
function measure!(cell, mod, kind, hook, policy_name, mode_name)
    ec = getfield(mod, :EnzymePolicyCases)
    mode = Symbol(mode_name)
    c = ec.case(hook)
    policy = getfield(mod, Symbol(policy_name))()
    r = ec.evaluate(c; policy, mode)
    cell["hook"] = hook
    cell["case_kind"] = c.kind
    cell["policy"] = policy_name
    cell["mode"] = string(mode)
    cell["status"] = r.status
    cell["relerr"] = r.relerr
    cell["error"] = r.error
    cell["versions"] = versions_now()
    if kind == "reach"
        hits = getfield(mod, :HITS)
        cell["hits"] = Dict{String, Any}(h => hits[h] for h in HOOKS)
    end
    return cell
end

# Counters on `Bramble._late`, the route of every `src/` call into a hook: a method per hook
# type, more specific than the generic one it then calls.
function define_late_counters!(mod)
    Core.eval(mod, :(const HITS = Dict{String, Int}()))
    for h in HOOKS
        T = typeof(getfield(Base.invokelatest(getfield, mod, :Bramble), Symbol(h)))
        Core.eval(mod, :(HITS[$h] = 0))
        Core.eval(mod, quote
            function Bramble._late(f::$T, args...)
                HITS[$h] += 1
                return invoke(Bramble._late, Tuple{Any, Vararg{Any}}, f, args...)
            end
        end)
    end
    return nothing
end

# The primal pass: the case's function once, under CpuPolyester, without Enzyme.
function measure_primal!(cell, mod, hook)
    ec = getfield(mod, :EnzymePolicyCases)
    c = ec.case(hook)
    f, x = c.build(getfield(mod, :CpuPolyester)())
    f(x)
    cell["hook"] = hook
    cell["case_kind"] = c.kind
    cell["status"] = "ok"
    cell["hits"] = Dict{String, Any}(h => getfield(mod, :HITS)[h] for h in HOOKS)
    cell["versions"] = versions_now()
    return cell
end

function child_main(args)
    kind, out = args[1], args[end]
    kind == "upstream" && return upstream_child(parse(Int, args[2]), out)
    push!(LOAD_PATH, enzyme_env())
    mod = Module(:EnzymeCensusCell)
    Core.eval(mod, :(import Bramble))
    Core.eval(mod, :(using Bramble, Polyester, LinearAlgebra, SparseArrays, Random))
    Core.eval(mod, :(using Enzyme, ForwardDiff, Kronecker))
    Core.eval(mod, :(using Enzyme.EnzymeRules: EnzymeRules, RevConfig, AugmentedReturn))
    Core.eval(mod, :(import DifferentiationInterface as DI))
    Core.eval(mod, :(using Bramble: CpuPolyester, CpuSerial))
    cell = Dict{String, Any}("pass" => kind, "status" => "running", "error" => "")
    write_cell(out, cell)
    if kind == "versions"
        cell["versions"] = versions_now()
        cell["status"] = "ok"
        write_cell(out, cell)
        return nothing
    end
    Base.include(mod, HARNESS)
    kind == "reach" && Base.invokelatest(define_stubs!, mod)
    if kind == "primal"
        Base.invokelatest(define_late_counters!, mod)
        Base.invokelatest(measure_primal!, cell, mod, args[2])
    else
        Base.invokelatest(measure!, cell, mod, kind, args[2:4]...)
    end
    write_cell(out, cell)
    return nothing
end

# An upstream reproducer runs in a process with no Bramble: the temporary environment
# supplies Enzyme and the snippet loads what it needs.
function upstream_child(i, out)
    u = UPSTREAM[i]
    push!(LOAD_PATH, enzyme_env())
    cell = Dict{String, Any}("pass" => "upstream", "construct" => u.construct,
        "snippet" => upstream_snippet(u), "status" => "running", "error" => "")
    write_cell(out, cell)
    try
        ok = Base.include_string(Main, cell["snippet"], "upstream.jl")
        cell["status"] = ok === true ? "ok" : "wrong"
    catch err
        err isa InterruptException && rethrow()
        cell["status"] = "throws"
        cell["error"] = error_text(err isa LoadError ? err.error : err)
    end
    cell["versions"] = versions_now()
    write_cell(out, cell)
    return nothing
end

# ---------------------------------------------------------------------------------------
# The scheduler: runs every child, collects the cells and writes the results file.
# ---------------------------------------------------------------------------------------

slot_wrapper() = joinpath(ROOT, ".claude", "scripts", "julia_slot.sh")

# `julia` through the slot wrapper with a private pool of `threads` slots: the outer process
# already holds its slots, so the children must not draw from the machine-wide pool again.
function julia_cmd(threads, args...)
    julia = `julia --startup-file=no --threads=$threads --project=$(joinpath(ROOT, "test")) $args`
    isfile(slot_wrapper()) || return julia
    pool = Dict("BRAMBLE_SLOT_DIR" => joinpath(tempdir(), "enzyme_policy_census_slots"),
        "BRAMBLE_JULIA_SLOTS" => string(threads))
    return addenv(`$(slot_wrapper()) $threads $(julia.exec)`, pool)
end

function descendants(pid)
    kids = try
        parse.(Int, split(read(ignorestatus(`pgrep -P $pid`), String)))
    catch
        Int[]
    end
    return vcat(kids, (descendants(k) for k in kids)...)
end

function kill_tree(proc)
    pid = getpid(proc)
    for p in reverse(descendants(pid))
        run(ignorestatus(`kill -KILL $p`))
    end
    kill(proc, Base.SIGKILL)
end

# Runs `cmd`, its output going to `log`; kills the process tree after `timeout` seconds.
# Returns the exit state: `(:ok, 0)`, `(:exit, code)`, `(:signal, sig)` or `(:timeout, 0)`.
function run_limited(cmd, log, timeout)
    open(log, "w") do io
        proc = run(pipeline(cmd; stdout = io, stderr = io); wait = false)
        timed_out = Ref(false)
        timer = Timer(timeout) do _
            timed_out[] = true
            kill_tree(proc)
        end
        wait(proc)
        close(timer)
        timed_out[] && return (:timeout, 0)
        proc.termsignal != 0 && return (:signal, proc.termsignal)
        return proc.exitcode == 0 ? (:ok, 0) : (:exit, proc.exitcode)
    end
end

# The last lines of a child's log: Enzyme can print megabytes of LLVM IR on a failure.
function log_tail(log)
    isfile(log) || return ""
    text = open(log) do io
        seekend(io)
        seek(io, max(0, position(io) - 4096))
        return read(io, String)
    end
    lines = filter(!isempty, strip.(split(text, '\n')))
    return isempty(lines) ? "" : clean(join(last(lines, 3), " | "))
end

# Runs one child and returns its cell. A child that never finished its cell is a crash.
function run_child(dir, index, args...)
    out = joinpath(dir, "cell$index.toml")
    log = joinpath(dir, "cell$index.log")
    state = run_limited(julia_cmd(THREADS, SCRIPT, "--child", args..., out), log, CELL_TIMEOUT)
    cell = isfile(out) ? TOML.parsefile(out) : Dict{String, Any}("status" => "running")
    if cell["status"] == "running"
        why = state[1] == :timeout ? "timeout after $CELL_TIMEOUT s" :
              state[1] == :signal ? "killed by signal $(state[2])" :
              "exit code $(state[2])"
        cell["status"] = "crash"
        cell["error"] = clean("$why: $(log_tail(log))")
    end
    get!(cell, "relerr", NaN)
    get!(cell, "error", "")
    return cell
end

git(args...) = strip(read(`git -C $ROOT $args`, String))

function enzyme_manifest_version()
    deps = TOML.parsefile(joinpath(ROOT, "Manifest.toml"))["deps"]
    return deps["Enzyme"][1]["version"]
end

# The row of one hook: the reach pass, the three status passes and the serial control.
function case_row(dir, index, hook)
    reach = run_child(dir, index, "reach", hook, "CpuPolyester", "reverse")
    hits = get(reach, "hits", nothing)
    reach_by = "rule"
    reach_error = ""
    if hits !== nothing && hits[hook] == 0 &&
       occursin("CallingConventionMismatchError", get(reach, "error", ""))
        reach_error = reach["error"]
        reach = run_child(dir, index * 10 + 5, "primal", hook)
        hits = get(reach, "hits", nothing)
        reach_by = "primal"
    end
    reaches = hits === nothing ? "crash" : hits[hook] > 0
    also = hits === nothing ? String[] : [h for h in HOOKS if h != hook && hits[h] > 0]
    modes = Dict{String, Any}()
    for (j, mode) in enumerate(("reverse", "di_reverse", "forward"))
        modes[mode] = run_child(dir, index * 10 + j, "status", hook, "CpuPolyester", mode)
    end
    serial = run_child(dir, index * 10 + 4, "status", hook, "CpuSerial", "reverse")
    return Dict{String, Any}(
        "hook" => hook, "policy" => "CpuPolyester", "kind" => get(reach, "case_kind", "unknown"),
        "reaches" => reaches, "reach_by" => reach_by, "reach_error" => reach_error,
        "hits" => hits === nothing ? -1 : hits[hook],
        "also_reaches" => also,
        "reverse" => modes["reverse"]["status"],
        "di_reverse" => modes["di_reverse"]["status"],
        "forward" => modes["forward"]["status"],
        "serial_reverse" => serial["status"],
        "relerr_reverse" => modes["reverse"]["relerr"],
        "relerr_di_reverse" => modes["di_reverse"]["relerr"],
        "relerr_forward" => modes["forward"]["relerr"],
        "error_reverse" => modes["reverse"]["error"],
        "error_di_reverse" => modes["di_reverse"]["error"],
        "error_forward" => modes["forward"]["error"],
        "serial_error" => serial["error"]
    )
end

function main(args)
    if !isempty(args) && args[1] == "--child"
        return child_main(args[2:end])
    end
    only = findfirst(==("--only"), args)
    only_hook = only === nothing ? nothing : args[only + 1]
    dir = mktempdir()
    try
        versions = run_child(dir, 0, "versions")
        versions["status"] == "ok" || error("version pass failed: $(versions["error"])")
        cases = Dict{String, Any}[]
        for (i, hook) in enumerate(HOOKS)
            only_hook !== nothing && hook != only_hook && continue
            t = @elapsed row = case_row(dir, i, hook)
            push!(cases, row)
            println(rpad("[$i] $hook", 44), "reaches=", row["reaches"], "  reverse=",
                row["reverse"], "  di_reverse=", row["di_reverse"], "  forward=",
                row["forward"], "  serial=", row["serial_reverse"], "  ",
                round(t; digits = 1), " s")
            flush(stdout)
        end
        upstream = Dict{String, Any}[]
        for i in eachindex(UPSTREAM)
            only_hook === nothing || break
            cell = run_child(dir, 1000 + i, "upstream", string(i))
            push!(upstream,
                Dict{String, Any}(
                    "construct" => UPSTREAM[i].construct, "snippet" => cell["snippet"],
                    "status" => cell["status"], "error" => cell["error"],
                    "enzyme" => get(get(cell, "versions", Dict()), "Enzyme", "unknown"),
                    "polyester" => get(get(cell, "versions", Dict()), "Polyester", "unknown")))
            println(rpad("[upstream] $(UPSTREAM[i].construct)", 44), cell["status"])
        end
        if only_hook === nothing
            meta = Dict{String, Any}(
                "script" => basename(SCRIPT), "script_sha" => git("hash-object", SCRIPT),
                "commit" => git("rev-parse", "HEAD"),
                "date" => Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
                "threads" => THREADS, "cell_timeout_s" => CELL_TIMEOUT,
                "manifest_enzyme" => enzyme_manifest_version(),
                "versions" => versions["versions"],
                "cpu" => string(Sys.cpu_info()[1].model),
                "os" => string(Sys.KERNEL), "arch" => string(Sys.ARCH))
            mkpath(dirname(RESULTS))
            open(RESULTS, "w") do io
                TOML.print(
                    io, Dict{String, Any}("meta" => meta, "case" => cases,
                        "upstream" => upstream); sorted = true)
            end
            println("wrote ", RESULTS)
        end
    finally
        rm(dir; recursive = true, force = true)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == SCRIPT
    main(ARGS)
end
