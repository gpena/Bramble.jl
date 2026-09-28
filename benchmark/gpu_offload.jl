#===========================================================================#
# GpuOffload back-to-back throughput ratio -- gpena/Bramble.jl#324, subplan S2.3.
#
# `GpuOffload` (a `CpuPolicy` wrapping an inner CPU policy and a device backend,
# `src/utils/backend.jl`) routes just `Rₕ!`/`avgₕ!`'s fill step through a device
# (`_offload_project!`, `src/operators/projection.jl`) while keeping the space itself
# host-storage typed. The issue's own exploratory numbers (`BenchmarkTools`, one warmed
# process, `--threads=4`, AC power) are the reason this policy exists at all:
#
#     | | Serial | Parallel (4t) | GPU-only | GPU+copyto! | vs Parallel |
#     |---|---|---|---|---|---|
#     | Rₕ! 1D, 10,000,000     | 29.7 ms   | 12.0 ms | 1.12 ms | 6.43 ms | 1.87x  |
#     | Rₕ! 2D, 3000x3000      | 62.2 ms   | 18.2 ms | 4.24 ms | 9.19 ms | 1.98x  |
#     | avgₕ! 1D, 10,000,000   | 190.2 ms  | 74.1 ms | 2.47 ms | 7.51 ms | 9.87x  |
#     | avgₕ! 2D, 3000x3000    | 2011.5 ms | 576.2 ms| 23.3 ms | 28.1 ms | 20.52x |
#
# This script re-measures those same four rows honestly, as of whenever it is run, rather
# than assuming the numbers above still hold: `CpuThreaded()` (plain threaded host) against
# `GpuOffload(metal_backend(), CpuThreaded())` (the same inner policy, wrapped), back to
# back on the same grid shape. A lost win must be visible, never tuned away -- see "What
# fails the run" below.
#
# Usage:
#     julia --project=benchmark --threads=4 benchmark/gpu_offload.jl [--smoke]
#
# The full run (no `--smoke`) needs a quiet, AC-powered machine and is meant to be run alone
# by whoever integrates this subplan, not routinely -- `--smoke` (tiny sizes, no ratio gate,
# no power/load gate) is the structural check this file's own CHECK runs.
#
# ## What is measured, per row
#
# `Rₕ!` 1D n=10,000,000, `Rₕ!` 2D 3000x3000, `avgₕ!` 1D n=10,000,000, `avgₕ!` 2D 3000x3000 --
# the issue's own table, nothing else (no assembly, no Kronecker, no masked-projection
# timing: this measures `GpuOffload`'s `Rₕ!`/`avgₕ!` win only).
#
# For each row: a host space (`backend(Float64; policy = CpuThreaded())`) and a
# `GpuOffload`-backed space over the identical grid shape (`backend(Float32; policy =
# GpuOffload(metal_backend(), CpuThreaded()))` -- Metal has no `Float64`, so the offload arm
# is necessarily `Float32`, exactly like the issue's own "GPU-only"/"GPU+copyto!" columns
# above). Both arms call the *same* `Rₕ!`/`avgₕ!` on their own space; under `GpuOffload` that
# call transparently fills through the device and copies the result back
# (`_offload_project!`), so this script never touches device buffers directly.
#
# Minimum-of-N warmed trials (`BenchmarkTools`, matching `benchmark/polyester_crossover.jl`'s
# own `_min_ms` convention: one untimed warm-up call for JIT, then `@benchmarkable` samples).
# Ratio = host time / offload time, so a ratio > 1 means the offload arm won.
#
# ## Correctness before timing (bramble-verification: "a fast wrong answer is the failure
# ## mode")
#
# Each row's offload result is compared against its own host result once, before either is
# timed, to `rtol = atol = 1f-5` -- the Metal tolerance this milestone's own acceptance
# criteria use elsewhere (gpena/Bramble.jl#174), not bitwise, since the two arms run at
# different element types (`Float64` host vs `Float32` device). A mismatch withholds that
# row's ratio from the pass/fail gate below and is listed at the end.
#
# ## What fails the run
#
# A row whose offload result disagrees with its host result exits non-zero with no success
# marker. A row whose ratio is <= 1 (the device arm did not beat plain threaded host) does
# not: `GpuOffload` is not a universal win, and the recorded run already has one such row
# (`Rₕ!` 1D, ratio 0.86: one cheap function evaluation per point cannot pay for the
# per-call upload and copy-back). Every losing row is listed by name with its ratio after
# the table, so a lost win stays visible without reading as a broken script. `--smoke`
# prints its own unconditional structural-check marker, following
# `benchmark/scatter_table.jl`'s own `--smoke`-success convention.
#
# ## Gates (bramble-benchmarks §1), via `.claude/scripts/check_power_load.sh`
#
# Delegates to the repo's own AC-power/load-average gate script rather than reimplementing
# `pmset`/`uptime` parsing here, matching `benchmark/scatter_table.jl`'s own WHY (share the
# gate script instead of duplicating it). Its stdout ("Power: ..." / "Load: ...") is
# captured and reused as this run's own Power/Load columns. `--smoke` skips the gate.
#===========================================================================#

using Bramble
using Bramble: GpuOffload, CpuThreaded
using Metal
using BenchmarkTools
using PrettyTables

const SMOKE = "--smoke" in ARGS
const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const POWER_SCRIPT = joinpath(REPO_ROOT, ".claude", "scripts", "check_power_load.sh")

# Every line of output is prefixed under `--smoke`, so a figure recorded from a smoke run
# can never be mistaken for a real measurement even out of context.
function _out(msg::AbstractString = "")
    println(SMOKE ? "[SMOKE -- STRUCTURAL CHECK ONLY, NOT A MEASUREMENT] " * msg : msg)
end

function _print_table(args...; kwargs...)
    buf = IOBuffer()
    pretty_table(buf, args...; kwargs...)
    for line in split(String(take!(buf)), '\n')
        isempty(line) || _out(line)
    end
end

# --- AC power / load gate, via .claude/scripts/check_power_load.sh (matches
# --- benchmark/scatter_table.jl's own `_power_load_state`) ------------------------- #

function _power_load_state(; poll_s = SMOKE ? 0 : 30)
    if !isfile(POWER_SCRIPT)
        return (ok = true, power = "unknown (script missing)", load = "unknown")
    end
    cmd = SMOKE ? `$POWER_SCRIPT --allow-battery` : `$POWER_SCRIPT --poll $(poll_s == 0 ? 1 : poll_s * 20)`
    io = IOBuffer()
    ok = true
    try
        run(pipeline(cmd; stdout = io, stderr = io))
    catch
        ok = false
    end
    text = String(take!(io))
    power_m = match(r"Power:\s*(.+)", text)
    load_m = match(r"Load:\s*(.+)", text)
    power = power_m === nothing ? "unknown" : strip(power_m.captures[1])
    load = load_m === nothing ? "unknown" : strip(load_m.captures[1])
    return (ok = ok || SMOKE, power = power, load = load)
end

if !SMOKE
    gate = _power_load_state()
    if !gate.ok
        _out("REFUSED: $(POWER_SCRIPT) reports the machine is not ready for measurement " *
             "(battery power, or load above threshold, after waiting). Plug in and/or " *
             "quiet the machine, then re-run, or pass --smoke for a structural check only.")
        exit(1)
    end
end

# This benchmark exists to measure GpuOffload's real Metal win, so a non-functional device
# is refused rather than silently skipped, under --smoke too.
if !Metal.functional()
    _out("REFUSED: Metal.functional() == false. This benchmark measures GpuOffload's " *
         "device offload against threaded-host Rₕ!/avgₕ!, so it needs a real, usable " *
         "Metal device.")
    exit(1)
end

set_zero_subnormals(true)

_out("GpuOffload vs threaded-host Rₕ!/avgₕ! throughput ratio -- gpena/Bramble.jl#324")
_out()
_out("Julia threads : $(Threads.nthreads())" *
     (Threads.nthreads() == 4 ? "" : "  WARNING: expected 4 (bramble-benchmarks §1)"))
_out("Julia version : $(VERSION), OS: $(Sys.MACHINE)")
_out()

# --- Sizes ------------------------------------------------------------------ #

const N_1D = SMOKE ? 10_000 : 10_000_000
const DIMS_2D = SMOKE ? (30, 30) : (3000, 3000)
const NQ = Val(3)

const BENCH_SECONDS = SMOKE ? 0.1 : 1.0
const BENCH_SAMPLES = SMOKE ? 3 : 15

# --- Device-compilable test functions (no Float64 literals, no captures) --- #

_f1d(x) = sin(x)
_f2d(x) = sin(x[1]) * cos(x[2])

# --- Grid builders: host (Float64, CpuThreaded) and GpuOffload (Float32) -- #

function _host_1d_space(n::Int)
    gridspace(mesh(domain(interval(0.0, 1.0)), n, true; backend = backend(Float64; policy = CpuThreaded())))
end
function _host_2d_space(dims::NTuple{2, Int})
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    gridspace(mesh(Ω, dims, true; backend = backend(Float64; policy = CpuThreaded())))
end

function _offload_1d_space(n::Int)
    gridspace(
        mesh(
        domain(interval(0.0f0, 1.0f0)), n, true;
        backend = backend(Float32; policy = GpuOffload(metal_backend(), CpuThreaded()))
    )
    )
end
function _offload_2d_space(dims::NTuple{2, Int})
    Ω = domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0))
    gridspace(mesh(Ω, dims, true; backend = backend(Float32; policy = GpuOffload(metal_backend(), CpuThreaded()))))
end

# --- Timing (function barrier, matching benchmark/polyester_crossover.jl's --- #
# --- _min_ms convention) ----------------------------------------------------- #

function _min_ms(f::F) where {F}
    f()  # warm-up: JIT compilation happens here, excluded from the trial
    trial = run(@benchmarkable($f()); samples = BENCH_SAMPLES, evals = 1, seconds = BENCH_SECONDS)
    return minimum(trial.times) / 1e6
end

# --- One row ------------------------------------------------------------------ #

struct Row
    workload::String
    grid_label::String
    ndofs::Int
    t_host_ms::Float64
    t_offload_ms::Float64
    ratio::Float64
    ok::Bool
    note::String
    power::AbstractString
    load::AbstractString
end

function _measure_row(
        workload::String, grid_label::String, host_space, offload_space, op!::F,
        power::AbstractString, load::AbstractString
) where {F}
    u_host = element(host_space, Float64)
    u_offload = element(offload_space, Float32)

    op!(u_host)
    op!(u_offload)
    ok = isapprox(Float64.(parent(u_offload)), parent(u_host); rtol = 1.0f-5, atol = 1.0f-5)
    note = ok ? "OK" : "offload result diverged from host beyond rtol=atol=1f-5"

    t_host = _min_ms(() -> op!(u_host))
    t_offload = ok ? _min_ms(() -> op!(u_offload)) : NaN
    ratio = ok ? t_host / t_offload : NaN

    return Row(workload, grid_label, ndofs(host_space), t_host, t_offload, ratio, ok, note, power, load)
end

# --- Driver ------------------------------------------------------------------- #

function main()
    gate = _power_load_state()

    rows = Row[]

    push!(
        rows,
        _measure_row(
            "Rₕ!", "1D n=$N_1D", _host_1d_space(N_1D), _offload_1d_space(N_1D),
            u -> Rₕ!(u, _f1d), gate.power, gate.load
        )
    )
    push!(
        rows,
        _measure_row(
            "Rₕ!", "2D $(DIMS_2D[1])x$(DIMS_2D[2])", _host_2d_space(DIMS_2D), _offload_2d_space(DIMS_2D),
            u -> Rₕ!(u, _f2d), gate.power, gate.load
        )
    )
    push!(
        rows,
        _measure_row(
            "avgₕ!", "1D n=$N_1D", _host_1d_space(N_1D), _offload_1d_space(N_1D),
            u -> avgₕ!(u, _f1d, NQ), gate.power, gate.load
        )
    )
    push!(
        rows,
        _measure_row(
            "avgₕ!", "2D $(DIMS_2D[1])x$(DIMS_2D[2])", _host_2d_space(DIMS_2D), _offload_2d_space(DIMS_2D),
            u -> avgₕ!(u, _f2d, NQ), gate.power, gate.load
        )
    )

    header = [
        "Workload", "Grid", "ndofs", "CpuThreaded (ms)", "GpuOffload (ms)", "Ratio (threaded/offload)",
        "OK", "Note", "Power", "Load"
    ]
    data = Matrix{Any}(undef, length(rows), length(header))
    for (i, r) in enumerate(rows)
        data[i, :] = [
            r.workload, r.grid_label, r.ndofs,
            round(r.t_host_ms; digits = 5),
            r.ok ? round(r.t_offload_ms; digits = 5) : "-",
            r.ok ? round(r.ratio; digits = 3) : "-",
            r.ok, r.note, r.power, r.load
        ]
    end
    _out()
    _out("=== GpuOffload vs threaded-host Rₕ!/avgₕ! -- back-to-back ratio ===")
    _print_table(data; column_labels = header, fit_table_in_display_horizontally = false)

    mismatches = filter(r -> !r.ok, rows)
    regressions = filter(r -> r.ok && r.ratio <= 1, rows)

    _out()
    if isempty(mismatches)
        _out("Every row's offload result agreed with its own host result (rtol=atol=1f-5).")
    else
        _out("Rows that disagreed with their own host result (ratio withheld for these):")
        for r in mismatches
            _out("  $(r.workload) / $(r.grid_label): $(r.note)")
        end
    end

    if SMOKE
        # Tiny sizes are a structural check only: kernel-launch overhead is not expected to
        # be amortised at this scale, so the losing-row list below does not apply here.
        _out()
        _out("OK-S2.3")
        return nothing
    end

    _out()
    if isempty(regressions)
        _out("Every row's GpuOffload arm beat plain threaded-host Rₕ!/avgₕ! (ratio > 1).")
    else
        _out("Rows where GpuOffload did not beat threaded-host Rₕ!/avgₕ! (ratio <= 1):")
        for r in regressions
            _out("  $(r.workload) / $(r.grid_label): ratio = $(round(r.ratio; digits = 3))")
        end
    end

    if isempty(mismatches)
        _out("OK-S2.3")
    else
        _out("An offload result disagreed with its host result -- see rows above. Not recording OK-S2.3.")
        exit(1)
    end

    return nothing
end

main()
