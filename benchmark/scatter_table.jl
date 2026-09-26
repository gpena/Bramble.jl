#===========================================================================#
# Scatter-position record/replay table -- gpena/Bramble.jl#318.
#
# `_AssemblyCache` (`BilinearForm.cache`, `src/assembly/bilinear.jl`) records where each
# term's contribution lands in the assembled matrix once, then replays those positions on
# every `assemble!` refill instead of re-searching (`src/assembly/bilinear_pattern.jl`,
# `src/assembly/bilinear_execution.jl`). That record/replay path already covers `CpuSerial`
# and `CpuThreaded` (shipped under gpena/Bramble.jl#338, before this plan started). Whether
# `CpuPolyester` and a Metal-device refill also replay, or still search every position on
# every fill, is exactly what this script measures -- honestly, as of whenever it is run,
# not assumed from the issue. Two other subplans in this plan (S5.2, S5.3, not this one)
# may change `CpuPolyester`/Metal's answer later; this script's job is to report whatever
# is true on the day it runs, in a format stable enough to diff a "before" run against an
# "after" one.
#
# Usage:
#     julia --project=benchmark --threads=4 benchmark/scatter_table.jl [--smoke]
#
# ## What is measured, per (execution policy, grid size)
#
# - First `assemble` (ms): building the sparsity pattern and recording scatter positions
#   for the first time, on a freshly built form -- never reused, since a second call would
#   already be warm.
# - Refill (ms): minimum of 40 warmed `assemble!` calls into the same pre-allocated matrix
#   (one untimed warm-up call first, matching this repo's other benchmarks' convention of
#   excluding JIT compilation from the timed trials).
# - Cache bytes: `Base.summarysize(form.cache)` -- the recorded scatter-position table
#   itself (`_AssemblyCache.segments`, plus the small fixed fields alongside it).
# - Matrix bytes: `Base.summarysize(A)` -- the assembled matrix's own footprint, for
#   comparison against the cache's.
# - Cache/Matrix: the ratio of the two bytes columns above.
#
# A policy where the cache replays should show a refill time roughly independent of the
# grid's stencil-search cost and much smaller than the first-assemble time; a policy that
# still searches every fill should show a refill time much closer to (or slower than) first
# assemble. This script does not classify a row either way -- it prints the numbers and
# leaves the reading to whoever compares this run against a later one.
#
# ## Rows
#
# `CpuSerial()`, `CpuThreaded()`, `CpuPolyester()` (`Bramble.backend(Float64; policy =
# Bramble.CpuPolyester())`), and Metal (`Bramble.metal_backend()`, `Float32`, only if
# `Metal.functional()`) -- skipped with a note otherwise, never silently omitted.
#
# ## Grid sizes
#
# 1D, uniform, n in {513, 1025, 2049} DOFs, plus one non-uniform 2D grid
# `mesh(domain(I x I), (129, 97), (false, false))` (129*97 = 12513 DOFs) -- this repo's own
# convention (`bramble-performance`: "never justify a design with a uniform-only benchmark")
# requires at least one non-uniform case in the mix. The Metal row rebuilds the same grid
# shapes over `Float32` (`Float64` is not supported on Apple Silicon GPUs, `metal_backend`'s
# own docstring) -- a different element type, same DOF counts, so its bytes/timings are not
# read against the CPU rows byte-for-byte, only trend-for-trend.
#
# ## Correctness before timing (bramble-verification: "a fast wrong answer is the failure
# ## mode")
#
# Each arm's own 40-times-refilled `A` is compared, after the last refill, against a fresh
# `assemble(a)` on the same form -- same algorithm, same arm, so this only catches a refill
# that silently diverges from a from-scratch assembly (e.g. a stale scatter position after
# the pattern changes), not a cross-policy discrepancy (that comparison is
# `benchmark/polyester_crossover.jl` and `benchmark/storage_policy_grid.jl`'s job, not this
# one -- this script measures the recording table's footprint and the search-vs-replay
# timing gap, nothing about cross-arm numerical agreement). A mismatch withholds that row's
# timing columns and is listed at the end; `OK-S5.1` only prints under `--smoke` success (see
# below) and the full run prints its own summary regardless, since a slow-but-correct row is
# still useful data for the "before"/"after" comparison this script exists to support.
#
# ## Gates (bramble-benchmarks §1), via `.claude/scripts/check_power_load.sh`
#
# Delegates to the repo's own AC-power/load-average gate script rather than reimplementing
# `pmset`/`uptime` parsing here (this script's own WHY: match `polyester_crossover.jl`'s
# shape but reuse the shared gate script). Its stdout ("Power: ..." / "Load: ...") is
# captured and reused as this run's own Power/Load columns, refreshed once per grid size so
# a long full run's later rows do not silently reuse a stale reading from its first minute.
# `--smoke` skips the gate (tiny sizes, structural check only, never a measurement) and
# prints a banner on every line saying so, matching every other script in this directory.
#===========================================================================#

using Bramble
using Bramble: CpuPolyester, allocate_system_matrix
using Polyester
using BenchmarkTools
using PrettyTables
using SparseArrays

const SMOKE = "--smoke" in ARGS
const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const POWER_SCRIPT = joinpath(REPO_ROOT, ".claude", "scripts", "check_power_load.sh")

metal_functional = false
try
    @eval using Metal
    global metal_functional = Metal.functional()
catch
    global metal_functional = false
end

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

# --- AC power / load gate, via .claude/scripts/check_power_load.sh -------- #

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

set_zero_subnormals(true)

_out("Scatter-position record/replay table -- gpena/Bramble.jl#318")
_out()
_out("Julia threads : $(Threads.nthreads())" *
     (Threads.nthreads() == 4 ? "" : "  WARNING: expected 4 (bramble-benchmarks §1)"))
_out("Julia version : $(VERSION), OS: $(Sys.MACHINE)")
_out("Metal usable  : $metal_functional")
_out()

# --- Sizes ------------------------------------------------------------------ #

const SIZES_1D = SMOKE ? (17, 33) : (513, 1025, 2049)
const SIZE_2D = SMOKE ? (13, 9) : (129, 97)

const BENCH_TRIALS = SMOKE ? 3 : 40

_poisson(u, v) = inner₊(∇ₕ(u), ∇ₕ(v))

# --- Grid builders: CPU (Float64) and Metal (Float32) --------------------- #

function _cpu_1d_space(n::Int, policy)
    gridspace(mesh(domain(interval(0.0, 1.0)), n, true; backend = backend(Float64; policy = policy)))
end
function _cpu_2d_space(dims::NTuple{2, Int}, policy)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    gridspace(mesh(Ω, dims, (false, false); backend = backend(Float64; policy = policy)))
end

function _metal_1d_space(n::Int)
    gridspace(mesh(domain(interval(0.0f0, 1.0f0)), n, true; backend = metal_backend()))
end
function _metal_2d_space(dims::NTuple{2, Int})
    Ω = domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0))
    gridspace(mesh(Ω, dims, (false, false); backend = metal_backend()))
end

# --- One row -------------------------------------------------------------- #

struct Row
    policy::String
    grid_label::String
    ndofs::Int
    nnz::Int
    first_ms::Float64
    refill_ms::Float64
    cache_bytes::Int
    matrix_bytes::Int
    ok::Bool
    note::String
    power::AbstractString
    load::AbstractString
end

function _min_ms(f::F; trials::Int = BENCH_TRIALS) where {F}
    f()  # warm-up, excluded
    best = Inf
    for _ in 1:trials
        t = (@elapsed f()) * 1000
        best = min(best, t)
    end
    return best
end

function _measure_row(policy_name::String, grid_label::String, Wₕ, power::AbstractString, load::AbstractString)
    a = form(Wₕ, Wₕ, _poisson)
    ndofs_n = ndofs(Wₕ)

    t_first = (@elapsed (A = assemble(a))) * 1000

    Awarm = allocate_system_matrix(a)
    t_refill = _min_ms(() -> assemble!(Awarm, a))

    Afresh = assemble(a)
    ok = size(Awarm) == size(Afresh) && nnz(Awarm) == nnz(Afresh) &&
         isapprox(collect(nonzeros_of(Awarm)), collect(nonzeros_of(Afresh)); rtol = 1e-11, atol = 1e-11)
    note = ok ? "OK" : "refilled matrix disagrees with a fresh assemble on the same form"

    cache_bytes = Base.summarysize(a.cache)
    matrix_bytes = Base.summarysize(Awarm)

    return Row(
        policy_name, grid_label, ndofs_n, nnz(Awarm), t_first, ok ? t_refill : NaN,
        cache_bytes, matrix_bytes, ok, note, power, load
    )
end

# `nnz`/`nonzeros` are defined for SparseMatrixCSC; a device-resident matrix (Metal) may not
# support `nonzeros` directly, so fall back to `Array` for the correctness comparison only
# (never for the byte-size columns, which use the real object throughout).
nonzeros_of(A::SparseMatrixCSC) = nonzeros(A)
nonzeros_of(A) = vec(Array(A))

# --- Driver ----------------------------------------------------------------- #

function _run_grid(grid_label::String, cpu_space_builder::Function, metal_space_builder)
    gate = _power_load_state(; poll_s = 5)
    rows = Row[]

    push!(rows, _measure_row("CpuSerial", grid_label, cpu_space_builder(Serial()), gate.power, gate.load))
    push!(rows, _measure_row("CpuThreaded", grid_label, cpu_space_builder(Parallel()), gate.power, gate.load))
    push!(rows, _measure_row("CpuPolyester", grid_label, cpu_space_builder(CpuPolyester()), gate.power, gate.load))

    if metal_functional && metal_space_builder !== nothing
        try
            push!(rows, _measure_row("Metal", grid_label, metal_space_builder(), gate.power, gate.load))
        catch e
            _out("Metal row failed for $grid_label: $(sprint(showerror, e))")
        end
    else
        _out("Metal row skipped for $grid_label: Metal.functional() == false or Metal.jl not usable")
    end

    return rows
end

function main()
    all_rows = Row[]

    for n in SIZES_1D
        label = "1D uniform n=$n"
        _out("=== $label ===")
        rows = _run_grid(label, policy -> _cpu_1d_space(n, policy), () -> _metal_1d_space(n))
        append!(all_rows, rows)
    end

    label2d = "2D non-uniform $(SIZE_2D[1])x$(SIZE_2D[2])"
    _out("=== $label2d ===")
    rows2d = _run_grid(label2d, policy -> _cpu_2d_space(SIZE_2D, policy), () -> _metal_2d_space(SIZE_2D))
    append!(all_rows, rows2d)

    header = [
        "Policy", "Grid", "ndofs", "nnz", "First assemble (ms)", "Refill min-of-$(BENCH_TRIALS) (ms)",
        "Cache bytes", "Matrix bytes", "Cache/Matrix", "OK", "Note", "Power", "Load"
    ]
    data = Matrix{Any}(undef, length(all_rows), length(header))
    for (i, r) in enumerate(all_rows)
        data[i, :] = [
            r.policy, r.grid_label, r.ndofs, r.nnz,
            r.ok ? round(r.first_ms; digits = 4) : "-",
            r.ok ? round(r.refill_ms; digits = 5) : "-",
            r.cache_bytes, r.matrix_bytes,
            round(r.cache_bytes / r.matrix_bytes; digits = 5),
            r.ok, r.note, r.power, r.load
        ]
    end
    _out()
    _out("=== Scatter-position cache vs assembled-matrix footprint, first vs refill timing ===")
    _print_table(data; column_labels = header, fit_table_in_display_horizontally = false)

    all_ok = !isempty(all_rows) && all(r.ok for r in all_rows)
    _out()
    if all_ok
        _out("All rows agreed with a fresh assemble on their own form.")
    else
        _out("Rows that disagreed with a fresh assemble (timing withheld for these):")
        for r in all_rows
            r.ok || _out("  $(r.policy) / $(r.grid_label): $(r.note)")
        end
    end

    if SMOKE
        _out()
        _out("OK-S5.1")
    end
end

main()
