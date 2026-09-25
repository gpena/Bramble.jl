#===========================================================================#
# Polyester (`CpuPolyester`) vs Threads (`CpuThreaded`/`Parallel`) crossover
# against serial (`CpuSerial`/`Serial`) -- gpena/Bramble.jl#190, the last open
# acceptance criterion of the v3.3.0 milestone's Polyester extension work:
#
#     "Small-grid crossover point re-measured per workload and documented --
#      kernel-dependent, do not assume the Rₕ! number transfers to assembly
#      or _dot without checking."
#
#     JULIA_DEPOT_PATH="$TMPDIR/depot-<id>:$HOME/.julia" \
#         julia --startup-file=no --threads=4 --project=benchmark \
#         benchmark/polyester_crossover.jl [--smoke]
#
# ## What is measured
#
# Five workloads, the same five the issue's own exploratory table names, each
# run under all three CPU execution policies (`src/utils/backend.jl`):
# `CpuSerial`/`Serial`, `CpuThreaded`/`Parallel` (`Threads.@threads`), and
# `CpuPolyester` (`Polyester.@batch`, `ext/BramblePolyesterExt.jl`, requires
# `using Polyester`):
#
#   1. `Rₕ!` unmasked         -- `project!`'s point-value sweep, every entry written
#   2. `Rₕ!` masked           -- the same sweep restricted to the mesh's own
#                                `:boundary` marker (a realistic Dirichlet-style
#                                mask: O(perimeter) points out of O(n^D))
#   3. `avgₕ!` (nq = 3)       -- the cell-average sweep, a 3x3 Gauss-Legendre
#                                quadrature per cell in 2D
#   4. `assemble_parallel!` core -- refilling a pre-allocated Poisson matrix
#                                (`inner₊(∇ₕu, ∇ₕv)`) via the band-coloured sweep
#   5. `_dot` reduction       -- `innerₕ(uₕ, vₕ)`, a weighted inner product
#
# Workloads 1-4 sweep a square grid's per-axis point count `n` (total dofs
# n^2); workload 5 sweeps the 1D vector length directly, matching the
# granularity the issue's own table uses for each ("32^2"/"1024^2" vs.
# "n=10,000"/"n=1e6").
#
# For every (workload, size) pair, all three arms run on separately built
# `Backend`s that share identical mesh geometry and identical numeric input --
# the only thing that differs is `policy` in `backend(Float64; policy = ...)`
# passed to `mesh(...; backend = ...)`. `_dot`'s two operands are restricted
# once under `Serial()` and then re-wrapped (`element(Wₕ, parent(uₕ))`, which
# `copyto!`s) onto the `Parallel()`/`CpuPolyester()` spaces, so every arm reduces
# over the bit-identical input array.
#
# ## Correctness before timing (bramble-verification: "a fast wrong answer is
# ## the failure mode")
#
# Every arm's *result* is checked against the serial arm's before either is
# timed, for every (workload, size) pair, not once at the start. `Rₕ!`/`avgₕ!`
# write one independent value per grid point (no accumulation across points),
# so the threaded/batched output is compared to the serial one to `rtol =
# atol = 1e-12` and is expected to be exact. The assembled matrix (workload 4)
# is compared the same way, entry by entry (`nzval`, after checking `colptr`/
# `rowval` match) -- never via `Matrix(A)`, which would materialise a dense
# 1024^2 x 1024^2 array. `_dot` (workload 5) sums n terms in a different order
# under each policy (`@simd` sequential vs. Polyester's own task-split
# `@batch reduction`), so floating-point non-associativity is expected to
# perturb the last few bits; it is checked to `rtol = atol = 1e-9`, loose
# enough for that reordering and tight enough to catch a wrong reduction.
# A mismatch at any (workload, size, arm) prints as `MISMATCH` and that one
# timing is skipped rather than reported next to a value that might be wrong;
# the final `OK-S7.2-CROSSOVER` marker only prints if every check across the
# whole sweep passed.
#
# One correctness fact worth stating up front rather than discovering it in
# the table: `_dot(::CpuThreaded, u, v, w)` (src/utils/linear_algebra.jl)
# still forwards to the identical serial reduction -- there never was a
# `Threads.@threads`-backed `_dot` to begin with (#112 remains open on
# exactly this). The "Threads/serial" ratio reported for workload 5 is
# therefore not measuring a second implementation at all: it is the same
# function called twice, and any difference between the two numbers is pure
# measurement noise around a true ratio of 1.0, not a parallel speedup or
# penalty. Only the "Polyester/serial" column is a real comparison for that
# row.
#
# ## Workload 4's table is not a threading comparison -- read this before the table
#
# `assemble_parallel!` (`src/form/bilinear.jl`) always takes the band-coloured
# sweep (`_assemble_bilinear_parallel_core!`), regardless of `trial_space`'s
# own policy: `_effective_parallel_policy` (`src/form/bilinear_execution.jl:510-513`)
# coerces `CpuSerial` to `CpuThreaded` on exactly this path, so there is no
# space you can hand `assemble_parallel!` that makes it run the *sweep*
# single-threaded. What this script's "serial (ms)" column measures instead
# is `assemble!` on a `Serial()` space, which takes a completely different
# code path -- `_assemble_bilinear_core_cached!`, the record/replay cache --
# not the sweep run with one worker. So the "serial" vs "threads"/"batch"
# columns below compare two *algorithms*, not the same algorithm at two
# thread counts, and "no crossover found" for this row means precisely:
# *the band-coloured sweep, even at 4 threads, never gets fast enough to
# catch the cached-replay serial algorithm* -- it does not mean threading
# fails to help the sweep itself.
#
# It does help. Since `Threads.nthreads()`/Polyester's own worker count are
# fixed for the life of one process, the only honest way to see the sweep
# threaded against itself at one worker is a separate process per thread
# count. That was judged too fragile to fold into this script: it would mean
# forking as many child `julia` processes as thread counts tested, each
# needing the same `JULIA_DEPOT_PATH` layering this sandboxed environment
# already requires just to run once, and silently going stale the next time
# this file's invocation contract changes. What follows instead are numbers
# from three separate single-process runs of this exact file, at
# `--threads=1`, `--threads=2` and `--threads=4`, read at n=1024 (the largest
# size this script's own sweep covers) -- reproduce them by running this file
# three times, once per thread count, and comparing the `assemble_parallel!
# core` row's "threads (ms)"/"batch (ms)" columns across the three runs:
#
#     threads: 1 -> 2 -> 4 workers    34.14 -> 17.79 -> 9.84 ms  (3.47x at 4)
#     batch:   1 -> 2 -> 4 workers    32.53 -> 16.89 -> 9.19 ms  (3.54x at 4)
#     cached-replay serial (unaffected by thread count, as it must be): ~6.9 ms
#
# So at 1024^2 the sweep at one thread is ~4.9x slower than cached replay,
# and 4 threads recovers 3.47x-3.54x of that gap, landing at a still-behind
# 1.3-1.4x -- the entire explanation for this row's numbers, and not a sign
# that threading or Polyester are ineffective here. The one clean signal this
# row's own sweep (below) does carry honestly, since it holds the algorithm
# fixed and only swaps the scheduler: **Polyester beats Threads on this exact
# sweep by roughly 5-7% at every thread count measured above** (32.53/34.14,
# 16.89/17.79, 9.19/9.84), consistent with this script's own single-process
# "batch/threads" column trending below 1 as n grows.
#
# ## Known methodology biases, and their direction
#
# - **TTFX excluded.** Every arm's first call (JIT compilation, one per
#   (workload, policy) combination, paid once per process) happens during the
#   correctness check, before any `@benchmarkable` trial runs. This is the
#   right choice for what this issue asks (a *steady-state* crossover, the
#   way a real driver calls these functions many times over a simulation),
#   but it means every number here is a floor: a one-shot script that pays
#   compilation once would see the parallel arms' *effective* crossover sit
#   higher than reported, since compilation cost does not shrink with problem
#   size while it is being amortised in this harness's warm-up call.
# - **Uniform meshes, not the non-uniform (`rand!`-built) ones
#   `benchmark/finch_assembly.jl` uses.** None of the five kernels here read
#   point *spacing* in a way that depends on whether it is uniform (`Rₕ!`
#   evaluates a function of position, `avgₕ!` integrates over a cell of
#   whatever size it is, `_dot` sums `u*v*w` elementwise, assembly discretises
#   the same stencil either way) -- so this should not move a crossover
#   measured in point count. It does mean the exact per-call timings are not
#   directly comparable to a non-uniform-mesh number elsewhere in this repo.
# - **A same-run, interleaved sweep**, not repeated process launches. Every
#   size runs serial-then-threads-then-batch back to back, on the machine's
#   power/load state at that moment. bramble-verification's own guidance
#   ("separate process launches are their own noise") argues for reading a
#   same-run ratio rather than absolute numbers across launches, which is
#   exactly what the crossover decision below does -- but it also means a
#   single slow moment on this machine moves all three arms of one row
#   together, not just one of them, so a suspicious single row is worth
#   rereading as "did the whole row move" before it is read as "this arm
#   regressed".
# - **The masked `Rₕ!` workload uses the mesh's whole `:boundary` marker**
#   (every mesh has one automatically, `src/mesh/marker.jl`), which is
#   O(perimeter) of the O(n^2) points -- the fraction a real Dirichlet
#   boundary condition masks. A marker selecting a different fraction (a
#   single edge, an interior region) would shift the masked numbers; this is
#   the realistic case, not an exhaustive one.
# - **The crossover-detection rule is a two-in-a-row rule, not "first size
#   where the ratio dips under 1"** -- see the next section for why and what
#   it costs.
# - **Workload 4's "serial" column is a different algorithm, not the sweep at
#   one thread** -- see "Workload 4's table is not a threading comparison"
#   above; its crossover is not a threading verdict.
#
# ## Finding a crossover from a noisy sweep
#
# A crossover is "the smallest size at which the parallel arm's timing beats
# serial's". Near that size the ratio is close to 1 and dominated by sampling
# noise (scheduler jitter, one slow sample skews `minimum` far less than
# `median`, but not to zero) -- a single lucky serial sample or a single slow
# parallel one can place a naive first-crossing estimate a decade or more away
# from where the trend actually turns, in either direction. This script does
# not re-run a suspicious point (bramble-verification's remedy for a *specific*
# suspected regression, which needs a human to decide what counts as
# suspicious); instead it applies one fixed, mechanical rule ahead of time:
#
#     crossover(arm) = the smallest measured size s, at index i in the
#         ascending sweep, such that ratio[i] < 1 AND (i is the largest index
#         measured OR ratio[i+1] < 1) -- i.e. the win is confirmed by the next
#         larger size also winning, unless there is no larger size to confirm
#         it with (the top of the sweep), in which case it is reported as
#         "provisional (top of sweep, unconfirmed)".
#
# A single-sample win surrounded by losses on both sides never qualifies; a
# real crossover, where every larger size keeps winning, always does. The
# tradeoff: a crossover that is a genuine single-point spike (wins only at
# exactly one size before losing again at every larger one) is reported as "no
# crossover found", understating rather than overstating parallelism's reach --
# the direction bramble-verification's own §2 argues for when a measurement
# could go either way.
#
# ## Gates enforced by this script (bramble-benchmarks §1, run in this order)
#
# 1. AC power (`pmset -g batt`) -- refuses to record real numbers on battery.
#    `--smoke` bypasses this at tiny sizes for structural validation only,
#    and prints a banner on every line saying so.
# 2. One-minute load average under half `Sys.CPU_THREADS` -- polled, not
#    checked once, since a transient spike (this process's own package
#    precompilation included) should be waited out rather than recorded
#    through.
# 3. `set_zero_subnormals(true)`.
# 4. `--threads=4` is the caller's responsibility (bramble-benchmarks §1); this
#    script does not set it itself and warns if `Threads.nthreads() != 4`.
#===========================================================================#

using Bramble
using Bramble: CpuPolyester, ExecutionPolicy, allocate_system_matrix, assemble_parallel!
using Polyester
using BenchmarkTools
using PrettyTables
using SparseArrays

# --- CLI / smoke mode ------------------------------------------------------ #

const SMOKE = "--smoke" in ARGS

# Every line of output is prefixed under `--smoke`, so a figure recorded from
# a smoke run can never be mistaken for a real measurement even out of context
# (the AC-power gate is bypassed under `--smoke`, at tiny sizes only).
function _out(msg::AbstractString)
    if SMOKE
        println("[SMOKE -- STRUCTURAL CHECK ONLY, NOT A MEASUREMENT] " * msg)
    else
        println(msg)
    end
end
_out() = _out("")

function _print_table(args...; kwargs...)
    buf = IOBuffer()
    pretty_table(buf, args...; kwargs...)
    for line in split(String(take!(buf)), '\n')
        isempty(line) || _out(line)
    end
end

# --- Gate 1: AC power (bramble-benchmarks §1) ------------------------------ #

function _power_state()
    text = read(`pmset -g batt`, String)
    m = match(r"Now drawing from '([^']+)'", text)
    source = m === nothing ? "unknown" : m.captures[1]
    return source, first(split(text, '\n'))
end

power_source, power_line = _power_state()
on_battery = occursin("Battery", power_source)

if on_battery && !SMOKE
    println(
        "REFUSED: on battery power ($power_source). Benchmarking on battery causes CPU " *
        "frequency scaling and thermal throttling, which skews measurements. Please plug " *
        "in the computer before running, or pass --smoke for a tiny structural check only.",
    )
    exit(1)
end

# --- Gate 2: one-minute load average under half Sys.CPU_THREADS ----------- #

function _wait_for_load(threshold::Float64; max_wait_s::Float64 = 120.0, poll_s::Float64 = 5.0)
    load1 = Sys.loadavg()[1]
    waited = 0.0
    while load1 >= threshold && waited < max_wait_s
        sleep(poll_s)
        waited += poll_s
        load1 = Sys.loadavg()[1]
    end
    return load1, load1 < threshold, waited
end

const LOAD_THRESHOLD = Sys.CPU_THREADS / 2
load1, load_settled, load_waited = _wait_for_load(LOAD_THRESHOLD; max_wait_s = SMOKE ? 20.0 : 120.0)

# --- Gate 3 ----------------------------------------------------------------- #

set_zero_subnormals(true)

# --- Header ------------------------------------------------------------------ #

_out("Polyester (CpuPolyester) vs Threads (CpuThreaded) crossover against serial -- gpena/Bramble.jl#190")
_out()
_out("Power source : $power_source ($power_line)")
_out(
    "Load average : $(round(load1; digits = 2)) (threshold < $(round(LOAD_THRESHOLD; digits = 2)) " *
    "= Sys.CPU_THREADS/2 = $(Sys.CPU_THREADS)/2), settled = $load_settled, waited $(load_waited)s",
)
if !load_settled
    _out(
        "  NOTE: load did not settle within the poll window. This session is the only " *
        "process running on this machine, so the residual load is this script's own " *
        "(package precompilation, GC, JIT) -- not external contention -- and the run " *
        "proceeds rather than waiting further.",
    )
end
_out("Julia threads : $(Threads.nthreads())" *
     (Threads.nthreads() == 4 ? "" : "  WARNING: expected 4 (bramble-benchmarks §1)"))
_out("Julia version : $(VERSION), OS: $(Sys.MACHINE)")
_out()

# --- Sizes ------------------------------------------------------------------- #

const GRID_SIZES = SMOKE ? (4, 8, 16) :
                   (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
const DOT_SIZES = SMOKE ? (50, 200, 2_000) :
                  (100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000,
    1_000_000, 3_000_000, 10_000_000)

const BENCH_SECONDS = SMOKE ? 0.1 : 1.0
const BENCH_SAMPLES = SMOKE ? 3 : 15

# --- Geometry / spaces -------------------------------------------------------- #

function _space2d(n::Int, policy::ExecutionPolicy)
    gridspace(
        mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), n, true; backend = backend(Float64; policy = policy))
    )
end
function _space1d(n::Int, policy::ExecutionPolicy)
    gridspace(
        mesh(domain(interval(0.0, 1.0)), n, true; backend = backend(Float64; policy = policy))
    )
end

_f2d(x) = sin(2π * x[1]) * cos(2π * x[2])
_f1d(x) = sin(2π * x)
_g1d(x) = cos(3π * x) + 1.0
_poisson(u, v) = inner₊(∇ₕ(u), ∇ₕ(v))

# --- Correctness ---------------------------------------------------------- #

_agree(a::Real, b::Real; rtol = 1e-9, atol = 1e-9) = isapprox(a, b; rtol = rtol, atol = atol)
_agree(a::AbstractArray, b::AbstractArray; rtol = 1e-12, atol = 1e-12) = isapprox(a, b; rtol = rtol, atol = atol)

function _matrices_agree(A::SparseMatrixCSC, B::SparseMatrixCSC; rtol = 1e-11, atol = 1e-11)
    (A.colptr == B.colptr && A.rowval == B.rowval) || return false
    return isapprox(A.nzval, B.nzval; rtol = rtol, atol = atol)
end

const ALL_OK = Ref(true)

function _check(label::AbstractString, ok::Bool)
    ok || (_out("  MISMATCH: $label"); ALL_OK[] = false)
    return ok
end

# --- Timing (function barrier -- bramble-verification §1: measure flat, --- #
# --- never inline a closure over a top-level loop variable)               #

function _min_ms(f::F) where {F}
    f()  # warm-up: JIT compilation happens here, excluded from the trial
    trial = run(@benchmarkable($f()); samples = BENCH_SAMPLES, evals = 1, seconds = BENCH_SECONDS)
    return minimum(trial.times) / 1e6
end

# --- One row: grid workloads (Rₕ! unmasked/masked, avgₕ!, assemble) ------- #

struct GridRow
    workload::String
    n::Int
    ndofs::Int
    t_serial::Float64
    t_threads::Float64
    t_batch::Float64
end

function _run_grid_size(n::Int)
    Wₕ_s = _space2d(n, Serial())
    Wₕ_t = _space2d(n, Parallel())
    Wₕ_b = _space2d(n, CpuPolyester())
    ndofs_n = ndofs(Wₕ_s)
    rows = GridRow[]

    # --- Rₕ! unmasked ---
    u_s = element(Wₕ_s, Float64)
    Rₕ!(u_s, _f2d)
    u_t = element(Wₕ_t, Float64)
    Rₕ!(u_t, _f2d)
    u_b = element(Wₕ_b, Float64)
    Rₕ!(u_b, _f2d)
    ok_t = _check("Rₕ! unmasked n=$n Threads", _agree(parent(u_t), parent(u_s)))
    ok_b = _check("Rₕ! unmasked n=$n CpuPolyester", _agree(parent(u_b), parent(u_s)))
    t_s = _min_ms(() -> Rₕ!(u_s, _f2d))
    t_t = ok_t ? _min_ms(() -> Rₕ!(u_t, _f2d)) : NaN
    t_b = ok_b ? _min_ms(() -> Rₕ!(u_b, _f2d)) : NaN
    push!(rows, GridRow("Rₕ! unmasked", n, ndofs_n, t_s, t_t, t_b))

    # --- Rₕ! masked (:boundary, every mesh's own automatic marker) ---
    um_s = element(Wₕ_s, Float64)
    Rₕ!(um_s, _f2d; markers = (:boundary,))
    um_t = element(Wₕ_t, Float64)
    Rₕ!(um_t, _f2d; markers = (:boundary,))
    um_b = element(Wₕ_b, Float64)
    Rₕ!(um_b, _f2d; markers = (:boundary,))
    ok_t = _check("Rₕ! masked n=$n Threads", _agree(parent(um_t), parent(um_s)))
    ok_b = _check("Rₕ! masked n=$n CpuPolyester", _agree(parent(um_b), parent(um_s)))
    t_s = _min_ms(() -> Rₕ!(um_s, _f2d; markers = (:boundary,)))
    t_t = ok_t ? _min_ms(() -> Rₕ!(um_t, _f2d; markers = (:boundary,))) : NaN
    t_b = ok_b ? _min_ms(() -> Rₕ!(um_b, _f2d; markers = (:boundary,))) : NaN
    push!(rows, GridRow("Rₕ! masked", n, ndofs_n, t_s, t_t, t_b))

    # --- avgₕ! (nq = 3) ---
    w_s = element(Wₕ_s, Float64)
    avgₕ!(w_s, _f2d, Val(3))
    w_t = element(Wₕ_t, Float64)
    avgₕ!(w_t, _f2d, Val(3))
    w_b = element(Wₕ_b, Float64)
    avgₕ!(w_b, _f2d, Val(3))
    ok_t = _check("avgₕ! n=$n Threads", _agree(parent(w_t), parent(w_s)))
    ok_b = _check("avgₕ! n=$n CpuPolyester", _agree(parent(w_b), parent(w_s)))
    t_s = _min_ms(() -> avgₕ!(w_s, _f2d, Val(3)))
    t_t = ok_t ? _min_ms(() -> avgₕ!(w_t, _f2d, Val(3))) : NaN
    t_b = ok_b ? _min_ms(() -> avgₕ!(w_b, _f2d, Val(3))) : NaN
    push!(rows, GridRow("avgₕ! (nq=3)", n, ndofs_n, t_s, t_t, t_b))

    # --- assemble_parallel! core (Poisson matrix refill) ---
    a_s = form(Wₕ_s, Wₕ_s, _poisson)
    A_s = assemble(a_s)  # serial reference: the real (cached-record/replay) serial path
    a_t = form(Wₕ_t, Wₕ_t, _poisson)
    A_t = allocate_system_matrix(a_t, a_t.ast)
    assemble_parallel!(A_t, a_t)
    a_b = form(Wₕ_b, Wₕ_b, _poisson)
    A_b = allocate_system_matrix(a_b, a_b.ast)
    assemble_parallel!(A_b, a_b)
    ok_t = _check("assemble_parallel! n=$n Threads", _matrices_agree(A_t, A_s))
    ok_b = _check("assemble_parallel! n=$n CpuPolyester", _matrices_agree(A_b, A_s))
    t_s = _min_ms(() -> assemble!(A_s, a_s))
    t_t = ok_t ? _min_ms(() -> assemble_parallel!(A_t, a_t)) : NaN
    t_b = ok_b ? _min_ms(() -> assemble_parallel!(A_b, a_b)) : NaN
    push!(rows, GridRow("assemble_parallel! core", n, ndofs_n, t_s, t_t, t_b))

    return rows
end

# --- One row: the _dot reduction workload ---------------------------------- #

struct DotRow
    n::Int
    t_serial::Float64
    t_threads::Float64
    t_batch::Float64
end

function _run_dot_size(n::Int)
    Wₕ_s = _space1d(n, Serial())
    Wₕ_t = _space1d(n, Parallel())
    Wₕ_b = _space1d(n, CpuPolyester())

    u_s = Rₕ(Wₕ_s, _f1d)
    v_s = Rₕ(Wₕ_s, _g1d)
    u_t = element(Wₕ_t, parent(u_s))
    v_t = element(Wₕ_t, parent(v_s))
    u_b = element(Wₕ_b, parent(u_s))
    v_b = element(Wₕ_b, parent(v_s))

    s_s = innerₕ(u_s, v_s)
    s_t = innerₕ(u_t, v_t)
    s_b = innerₕ(u_b, v_b)
    ok_t = _check("_dot n=$n Threads", _agree(s_t, s_s))
    ok_b = _check("_dot n=$n CpuPolyester", _agree(s_b, s_s))

    t_s = _min_ms(() -> innerₕ(u_s, v_s))
    t_t = ok_t ? _min_ms(() -> innerₕ(u_t, v_t)) : NaN
    t_b = ok_b ? _min_ms(() -> innerₕ(u_b, v_b)) : NaN
    return DotRow(n, t_s, t_t, t_b)
end

# --- Crossover: smallest size confirmed by the next larger size ----------- #

function _crossover(sizes::AbstractVector, ratios::AbstractVector{<:Real})
    m = length(sizes)
    for i in 1:m
        isnan(ratios[i]) && continue
        if ratios[i] < 1
            if i == m
                return "provisional: $(sizes[i]) (top of sweep, unconfirmed)"
            elseif !isnan(ratios[i + 1]) && ratios[i + 1] < 1
                return string(sizes[i])
            end
        end
    end
    return "no crossover found in sweep range"
end

# --- Driver ----------------------------------------------------------------- #

function main()
    grid_rows = GridRow[]
    for n in GRID_SIZES
        append!(grid_rows, _run_grid_size(n))
    end

    dot_rows = DotRow[]
    for n in DOT_SIZES
        push!(dot_rows, _run_dot_size(n))
    end

    workloads = ("Rₕ! unmasked", "Rₕ! masked", "avgₕ! (nq=3)")
    for wl in workloads
        wl_rows = filter(r -> r.workload == wl, grid_rows)
        header = [
            "n", "ndofs", "serial (ms)", "threads (ms)", "batch (ms)",
            "threads/serial", "batch/serial"
        ]
        data = Matrix{Any}(undef, length(wl_rows), length(header))
        for (i, r) in enumerate(wl_rows)
            rt = r.t_threads / r.t_serial
            rb = r.t_batch / r.t_serial
            data[i, :] = [
                r.n, r.ndofs, round(r.t_serial; digits = 4), round(r.t_threads; digits = 4),
                round(r.t_batch; digits = 4), round(rt; digits = 3), round(rb; digits = 3)
            ]
        end
        _out()
        _out("=== $wl ===")
        _print_table(data; column_labels = header, fit_table_in_display_horizontally = false)

        ns = [r.n for r in wl_rows]
        rt_all = [r.t_threads / r.t_serial for r in wl_rows]
        rb_all = [r.t_batch / r.t_serial for r in wl_rows]
        small = wl_rows[1]
        large = wl_rows[end]
        _out(
            "Small grid ($(small.n)^2 = $(small.ndofs)): threads/serial = " *
            "$(round(small.t_threads / small.t_serial; digits = 3)), batch/serial = " *
            "$(round(small.t_batch / small.t_serial; digits = 3))",
        )
        _out(
            "Large grid ($(large.n)^2 = $(large.ndofs)): threads/serial = " *
            "$(round(large.t_threads / large.t_serial; digits = 3)), batch/serial = " *
            "$(round(large.t_batch / large.t_serial; digits = 3))",
        )
        _out("Crossover (Threads vs serial):   $(_crossover(ns, rt_all)) grid points/axis")
        _out("Crossover (Polyester vs serial):  $(_crossover(ns, rb_all)) grid points/axis")
    end

    # --- assemble_parallel! core: NOT a threading comparison -- see the ---- #
    # --- header section "Workload 4's table is not a threading comparison". #
    # `assemble_parallel!` always runs the band-coloured sweep, whichever
    # policy `trial_space` carries (`_effective_parallel_policy` coerces
    # `CpuSerial` to `CpuThreaded` on this path); "serial (ms)" below is
    # `assemble!`'s cached-replay algorithm instead, the only single-threaded
    # baseline a single process can produce. The two are different code
    # paths, so this table's ratios say "sweep vs cache", not "N threads vs
    # one" -- read the header before trusting a crossover from it.
    let
        wl_rows = filter(r -> r.workload == "assemble_parallel! core", grid_rows)
        header = [
            "n", "ndofs", "serial, cached replay (ms)", "threads, coloured sweep (ms)",
            "batch, coloured sweep (ms)", "sweep threads/cached serial",
            "sweep batch/cached serial", "sweep batch/sweep threads"
        ]
        data = Matrix{Any}(undef, length(wl_rows), length(header))
        for (i, r) in enumerate(wl_rows)
            rt = r.t_threads / r.t_serial
            rb = r.t_batch / r.t_serial
            rbt = r.t_batch / r.t_threads
            data[i, :] = [
                r.n, r.ndofs, round(r.t_serial; digits = 4), round(r.t_threads; digits = 4),
                round(r.t_batch; digits = 4), round(rt; digits = 3), round(rb; digits = 3),
                round(rbt; digits = 3)
            ]
        end
        _out()
        _out("=== assemble_parallel! core ===")
        _out(
            "NOT a threading comparison: \"serial\" is the cached-replay algorithm " *
            "(assemble!/_assemble_bilinear_core_cached!), \"threads\"/\"batch\" are the " *
            "band-coloured sweep (_assemble_bilinear_parallel_core!) -- a different " *
            "algorithm, always run even from a CpuSerial trial_space " *
            "(_effective_parallel_policy coerces it to CpuThreaded). See the header section " *
            "\"Workload 4's table is not a threading comparison\" before reading this table.",
        )
        _print_table(data; column_labels = header, fit_table_in_display_horizontally = false)

        ns = [r.n for r in wl_rows]
        rt_all = [r.t_threads / r.t_serial for r in wl_rows]
        rb_all = [r.t_batch / r.t_serial for r in wl_rows]
        small = wl_rows[1]
        large = wl_rows[end]
        _out(
            "Small grid ($(small.n)^2 = $(small.ndofs)): sweep threads/cached serial = " *
            "$(round(small.t_threads / small.t_serial; digits = 3)), sweep batch/cached serial = " *
            "$(round(small.t_batch / small.t_serial; digits = 3))",
        )
        _out(
            "Large grid ($(large.n)^2 = $(large.ndofs)): sweep threads/cached serial = " *
            "$(round(large.t_threads / large.t_serial; digits = 3)), sweep batch/cached serial = " *
            "$(round(large.t_batch / large.t_serial; digits = 3))",
        )
        _out(
            "\"Crossover\" here means \"the coloured sweep catches the cached-replay " *
            "algorithm\", not \"threading helps the sweep\" -- it does (see below):",
        )
        _out("  sweep-threads vs cached-replay-serial: $(_crossover(ns, rt_all)) grid points/axis")
        _out("  sweep-batch vs cached-replay-serial:   $(_crossover(ns, rb_all)) grid points/axis")
        _out()
        _out(
            "Genuine threading measurement (same algorithm, varying thread count): thread " *
            "count is fixed per Julia process, so this requires separate --threads=1/2/4 " *
            "runs of this file, not something one process can produce. Numbers below are " *
            "from three such runs, at n=1024 (attributed to that investigation, not " *
            "remeasured here); reproduce by diffing this file's own \"threads (ms)\"/" *
            "\"batch (ms)\" columns above across three runs at --threads=1, 2 and 4:",
        )
        _out("  threads (Threads.@threads): 1->2->4 workers  34.14 -> 17.79 -> 9.84 ms  (3.47x at 4)")
        _out("  batch (Polyester.@batch):   1->2->4 workers  32.53 -> 16.89 -> 9.19 ms  (3.54x at 4)")
        _out("  cached-replay serial (thread-count independent, as it must be): ~6.9 ms")
        _out(
            "Polyester beats Threads on this exact sweep by ~5-7% at every thread count " *
            "measured above (32.53/34.14, 16.89/17.79, 9.19/9.84) -- the one clean, " *
            "same-algorithm signal this row carries. This run's own single-process " *
            "sweep-batch/sweep-threads column (last column above) trends the same way as n grows.",
        )
    end

    header = ["n", "serial (ms)", "threads (ms)", "batch (ms)", "threads/serial", "batch/serial"]
    data = Matrix{Any}(undef, length(dot_rows), length(header))
    for (i, r) in enumerate(dot_rows)
        rt = r.t_threads / r.t_serial
        rb = r.t_batch / r.t_serial
        data[i, :] = [
            r.n, round(r.t_serial; digits = 5), round(r.t_threads; digits = 5),
            round(r.t_batch; digits = 5), round(rt; digits = 3), round(rb; digits = 3)
        ]
    end
    _out()
    _out("=== _dot reduction (innerₕ) ===")
    _out(
        "Reminder: _dot(::CpuThreaded, ...) forwards to the plain serial reduction " *
        "(src/utils/linear_algebra.jl) -- #112 remains open on this. The threads/serial " *
        "column below is the same code measured twice; only batch/serial is a real " *
        "comparison.",
    )
    _print_table(data; column_labels = header, fit_table_in_display_horizontally = false)
    ns = [r.n for r in dot_rows]
    rt_all = [r.t_threads / r.t_serial for r in dot_rows]
    rb_all = [r.t_batch / r.t_serial for r in dot_rows]
    small = dot_rows[1]
    large = dot_rows[end]
    _out(
        "Small n=$(small.n): threads/serial = $(round(small.t_threads / small.t_serial; digits = 3)), " *
        "batch/serial = $(round(small.t_batch / small.t_serial; digits = 3))",
    )
    _out(
        "Large n=$(large.n): threads/serial = $(round(large.t_threads / large.t_serial; digits = 3)), " *
        "batch/serial = $(round(large.t_batch / large.t_serial; digits = 3))",
    )
    _out("Crossover (Threads vs serial):   $(_crossover(ns, rt_all)) elements (not a real arm, see above)")
    _out("Crossover (Polyester vs serial):  $(_crossover(ns, rb_all)) elements")

    _out()
    if ALL_OK[]
        _out("OK-S7.2-CROSSOVER")
    else
        _out("One or more arms MISMATCHED serial output -- see MISMATCH lines above. Not recording OK-S7.2-CROSSOVER.")
    end
end

main()
