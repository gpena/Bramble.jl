#===========================================================================#
# Storage x execution-policy grid: CSC vs SparseMatrixCSR, each under
# Serial/CpuThreaded/CpuPolyester, across 1D/2D/3D, at a small and a big grid
# size per dimension.
#
# Neither `benchmark/backends.jl` (CSC vs CSR, one policy -- Serial -- per
# storage) nor `benchmark/polyester_crossover.jl` (Serial vs Threads vs
# Polyester, CSC only) crosses both axes together. This script fills that
# gap: it measures `assemble`/`assemble!` (first-fill time+bytes, refill
# time+bytes) for every (storage, policy, dim, size) combination.
#
# Usage:
#     JULIA_DEPOT_PATH="$TMPDIR/depot-<id>:$HOME/.julia" \
#         julia --startup-file=no --threads=4 --project=benchmark \
#         benchmark/storage_policy_grid.jl [--smoke]
#
# ## Why `assemble!`, not `assemble_parallel!`
#
# `assemble!` (src/assembly/bilinear.jl) already dispatches on
# `execution_policy(form.trial_space)`: `CpuSerial` takes the cached
# record/replay path, anything else takes the band-coloured parallel sweep.
# That is exactly the three-way split this script wants, and it is the same
# entry point `benchmark/backends.jl` already times -- no need for
# `assemble_parallel!`'s always-threaded, policy-blind override (see that
# function's docstring, and `polyester_crossover.jl`'s header on why its own
# "assemble_parallel! core" table is *not* a clean threading comparison).
#
# ## Correctness before timing (bramble-verification)
#
# Reference arm: CSC + Serial, one per (dim, size). Every other arm's
# assembled matrix is checked against it before its timing is trusted:
#   - Same storage (CSC, Threads/Batch): exact structural + near-exact value
#     match (`colptr`/`rowval` identical, `nzval` to `rtol=atol=1e-11`) --
#     `polyester_crossover.jl`'s `_matrices_agree`. A same-storage, same-mesh
#     assembly should reproduce the reference bit-for-bit modulo the
#     nonassociative reduction order threading introduces.
#   - CSR arms (any policy): the non-densifying oracle from `backends.jl` --
#     stored-entry agreement after an O(nnz) CSR->CSC triplet re-sort (gated
#     1e-13 absolute) and action agreement `A*x` vs `Acsc*x` over several
#     random `x` (gated 1e-6 relative; see backends.jl's header for why an
#     absolute bound is wrong for this floating-point recomputation).
# A mismatch withholds that arm's timing and is reported at the end; the
# final `OK-STORAGE-POLICY-GRID` marker only prints if every arm agreed.
#
# ## Sizes
#
# Two sizes per dimension: "small" (grid small enough that thread/batch
# overhead can plausibly dominate the sweep) and "big" (`backends.jl`'s own
# DIMS, chosen there to keep the 3D case's memory and runtime reasonable).
# `--smoke` shrinks both for a fast structural check only, not a measurement.
#===========================================================================#

using Bramble
using Bramble: CpuPolyester, allocate_system_matrix
using SparseMatricesCSR
using Polyester
using PrettyTables
using SparseArrays
using Random

set_zero_subnormals(true)

const SMOKE = "--smoke" in ARGS
const ALLOW_BATTERY = "--allow-battery" in ARGS
const ZERO_BC = :dir => (x -> 0.0)
const RNG = Random.Xoshiro(20260922)

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

# --- preflight (bramble-benchmarks §1) ------------------------------------ #

function _power_source()
    Sys.isapple() || return (on_battery = false, raw = "unknown (not macOS)")
    try
        out = strip(read(`pmset -g batt`, String))
        return (on_battery = occursin("Battery Power", out), raw = out)
    catch
        return (on_battery = false, raw = "unknown")
    end
end

function _preflight(; poll_s = 30, max_wait_s = SMOKE ? 20.0 : 20 * 60.0)
    threads = Sys.CPU_THREADS
    threshold = threads / 2
    waited = 0.0
    while true
        load1 = Sys.loadavg()[1]
        load1 < threshold && return (settled = true, load1 = load1, threads = threads)
        _out(
            "Preflight: 1-min load $(round(load1; digits = 2)) >= half of $threads cores " *
            "($(round(threshold; digits = 2))); waiting $(poll_s)s (waited $(waited)s of $(max_wait_s)s max)...",
        )
        waited >= max_wait_s && return (settled = false, load1 = load1, threads = threads)
        sleep(poll_s)
        waited += poll_s
    end
end

power = _power_source()
if power.on_battery && !SMOKE && !ALLOW_BATTERY
    println(
        "REFUSED: on battery power. Benchmarking on battery causes CPU frequency scaling " *
        "and thermal throttling, which skews measurements. Plug in the computer before " *
        "running, pass --allow-battery to proceed anyway (read ratios, not absolute ms), " *
        "or pass --smoke for a tiny structural check only.",
    )
    exit(1)
end
preflight = _preflight()

_out("Storage x execution-policy grid: CSC vs SparseMatrixCSR under Serial/Threads/Polyester")
_out()
_out("Threads       : $(Threads.nthreads()) ($(preflight.threads) CPU cores)")
_out("1-min load    : $(round(preflight.load1; digits = 2))")
_out("Power (pmset) : $(power.raw)")
power.on_battery &&
    _out("ON BATTERY (--allow-battery): CPU frequency scaling/thermal throttling can skew absolute timings. Read the ratio columns, not the absolute ms columns.")
preflight.settled ||
    _out("PREFLIGHT DID NOT SETTLE: figures below were taken under load; prefer ratios over absolute ms.")
Threads.nthreads() == 4 ||
    _out("WARNING: expected 4 threads (bramble-benchmarks §1), got $(Threads.nthreads())")
_out()

# --- geometry, mesh and form, dimension-generic (backends.jl) ------------- #

_unit_cube(::Val{1}) = interval(0.0, 1.0)
_unit_cube(::Val{D}) where {D} = reduce(×, ntuple(_ -> interval(0.0, 1.0), Val(D)))

_source(::Val{1}) = x -> sin(π * x)
_source(::Val{D}) where {D} = x -> prod(sin(π * xᵢ) for xᵢ in x)

_grid(::Val{1}, Ωd, n; backend) = mesh(Ωd, n, true; backend = backend)
function _grid(::Val{D}, Ωd, n; backend) where {D}
    mesh(Ωd, ntuple(_ -> n, Val(D)), ntuple(_ -> true, Val(D)); backend = backend)
end

_poisson_mass(u, v) = inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v)

# Sizes per dimension: from small (overhead-dominated) up to 1M+ DOFs.
const SIZES = SMOKE ?
              ((1, (("small", 200), ("big", 20_000))),
    (2, (("small", 8), ("big", 24))),
    (3, (("small", 4), ("big", 8)))) :
              (
    (1, (("small (1k)", 1_000), ("medium (100k)", 100_000), ("1M dofs", 1_000_000), ("3M dofs", 3_000_000))),
    (2, (("small (256)", 16), ("medium (90k)", 300), ("1M dofs", 1_000), ("2.25M dofs", 1_500))),
    (3, (("small (512)", 8), ("medium (216k)", 60), ("1M dofs", 100), ("2.2M dofs", 130))))

# --- backends x policies compared ------------------------------------------ #

struct ArmSpec
    storage::String
    policy_name::String
    ctor::Function          # () -> Backend
end

const ARMS = (
    ArmSpec("CSC", "Serial", () -> Bramble.backend(Float64; policy = Serial())),
    ArmSpec("CSC", "Threads", () -> Bramble.backend(Float64; policy = Parallel())),
    ArmSpec("CSC", "Polyester", () -> Bramble.backend(Float64; policy = CpuPolyester())),
    ArmSpec("CSR", "Serial", () -> csr_backend(Float64; policy = Serial())),
    ArmSpec("CSR", "Threads", () -> csr_backend(Float64; policy = Parallel())),
    ArmSpec("CSR", "Polyester", () -> csr_backend(Float64; policy = CpuPolyester()))
)

# --- non-densifying correctness oracle (backends.jl) ----------------------- #

_to_csc(A::SparseMatrixCSC) = A
function _to_csc(A::SparseMatrixCSR{1})
    m, n = size(A)
    rowptr, colval, nzval = A.rowptr, A.colval, A.nzval
    I = Vector{Int}(undef, length(nzval))
    @inbounds for i in 1:m, k in rowptr[i]:(rowptr[i + 1] - 1)

        I[k] = i
    end
    return sparse(I, colval, nzval, m, n)
end

function _matrices_agree(A::SparseMatrixCSC, B::SparseMatrixCSC; rtol = 1e-11, atol = 1e-11)
    (A.colptr == B.colptr && A.rowval == B.rowval) || return false
    return isapprox(A.nzval, B.nzval; rtol = rtol, atol = atol)
end

function _stored_entry_discrepancy(A, Acsc_ref)
    Δ = _to_csc(A) - Acsc_ref
    return isempty(nonzeros(Δ)) ? 0.0 : maximum(abs, nonzeros(Δ))
end

function _action_discrepancy(A, Acsc_ref; trials = 4)
    n = size(Acsc_ref, 2)
    worst_rel = 0.0
    for _ in 1:trials
        x = randn(RNG, n)
        yref = Acsc_ref * x
        d = maximum(abs, A * x - yref)
        s = maximum(abs, yref)
        worst_rel = max(worst_rel, s == 0 ? d : d / s)
    end
    return worst_rel
end

# --- per-arm measurement ---------------------------------------------------- #

function _refill_allocs(A, a)
    assemble!(A, a)
    return @allocated assemble!(A, a)
end

function _refill_time_ms(A, a; trials = 5)
    assemble!(A, a)  # warm-up
    return minimum((@elapsed assemble!(A, a)) for _ in 1:trials) * 1000
end

function _measure(dim::Int, n::Int, be, source)
    Iᴰ = _unit_cube(Val(dim))
    Ωd = domain(Iᴰ, :dir => boundary_symbols(Iᴰ))
    Ωₕ = _grid(Val(dim), Ωd, n; backend = be)
    Wₕ = gridspace(Ωₕ)
    a = form(Wₕ, Wₕ, _poisson_mass)

    t_first_ms = (@elapsed (A = assemble(a))) * 1000
    first_bytes = @allocated assemble(a)

    Awarm = allocate_system_matrix(a)
    refill_bytes = _refill_allocs(Awarm, a)
    refill_ms = _refill_time_ms(Awarm, a)
    Awarm = nothing

    return (
        matrix = A, n = size(A, 1), nnz = nnz(A), first_ms = t_first_ms,
        first_bytes = first_bytes, refill_ms = refill_ms, refill_bytes = refill_bytes
    )
end

struct Row
    dim::Int
    n::Int
    storage::String
    policy::String
    ndofs::Int
    nnz::Int
    first_ms::Float64
    first_bytes::Int
    refill_ms::Float64
    refill_bytes::Int
    ok::Bool
    note::String
end

function _run_size(dim::Int, n::Int)
    rows = Row[]
    source = _source(Val(dim))

    ref_spec = ARMS[1]  # CSC + Serial
    ref = _measure(dim, n, ref_spec.ctor(), source)
    push!(
        rows,
        Row(
            dim, n, ref_spec.storage, ref_spec.policy_name, ref.n, ref.nnz, ref.first_ms,
            ref.first_bytes, ref.refill_ms, ref.refill_bytes, true, "reference"
        )
    )

    for spec in ARMS[2:end]
        local r
        try
            r = _measure(dim, n, spec.ctor(), source)
        catch e
            _out("$(spec.storage)/$(spec.policy_name) failed for dim=$dim n=$n: $(sprint(showerror, e))")
            push!(
                rows,
                Row(
                    dim, n, spec.storage, spec.policy_name, ref.n, 0, NaN, 0, NaN, 0, false,
                    "failed: " * sprint(showerror, e)
                )
            )
            continue
        end

        if spec.storage == "CSC"
            ok = _matrices_agree(r.matrix, ref.matrix)
            note = ok ? "OK" : "value/structure mismatch vs CSC/Serial"
        else
            stored = _stored_entry_discrepancy(r.matrix, ref.matrix)
            action = _action_discrepancy(r.matrix, ref.matrix)
            ok = stored <= 1e-9 && action <= 1e-6
            note = ok ? "OK" : "stored max|Δ|=$stored, action rel|Δ|=$action"
        end

        push!(
            rows,
            Row(
                dim, n, spec.storage, spec.policy_name, r.n, r.nnz, r.first_ms, r.first_bytes,
                r.refill_ms, r.refill_bytes, ok, note
            )
        )
    end

    return rows
end

# --- threaded/serial replay summary (gpena/Bramble.jl#338, #342) ----------- #
#
# One line per dimension at the "1M dofs" size (>=100k DOFs in every
# dimension), reporting the warmed `assemble!` (refill) time ratio of
# CSC/Threads and CSC/Polyester against the CSC/Serial reference, plus bytes
# allocated per warmed refill. Calls nothing beyond what `_run_size` already
# computed, so this runs unchanged against `main`: there, `assemble!` for
# Parallel()/CpuPolyester() takes the position-*search* path instead of this
# branch's recorded-position *replay*, i.e. the same line then reads as
# "threaded search / serial replay" -- the exact comparison #338 wants,
# without any version-guarded code.

function _fmt_ratio(row, ref)
    row === nothing && return "unavailable (arm not run)"
    row.ok || return "unavailable ($(row.note))"
    return "$(round(row.refill_ms / ref.refill_ms; digits = 3))x ($(row.refill_bytes) B/refill)"
end

function _print_replay_summary(dim, label, n, rows)
    ref = rows[1]
    if !ref.ok
        _out("dim=$dim, $label (n=$n): threaded replay / serial replay -- unavailable (serial reference failed: $(ref.note))")
        return
    end
    threads_row = rows[findfirst(r -> r.storage == "CSC" && r.policy == "Threads", rows)]
    poly_row = rows[findfirst(r -> r.storage == "CSC" && r.policy == "Polyester", rows)]
    _out(
        "dim=$dim, $label (n=$n, ndofs=$(ref.ndofs)): threaded replay / serial replay -- " *
        "CSC/Threads=$(_fmt_ratio(threads_row, ref)), " *
        "CSC/Polyester=$(_fmt_ratio(poly_row, ref)); " *
        "serial refill=$(round(ref.refill_ms; digits = 5)) ms ($(ref.refill_bytes) B/refill)",
    )
end

# --- driver ----------------------------------------------------------------- #

function main()
    all_rows = Row[]
    for (dim, size_list) in SIZES
        for (label, n) in size_list
            _out("=== dim=$dim, $label grid (n=$n) ===")
            rows = _run_size(dim, n)
            append!(all_rows, rows)

            ref = rows[1]
            header = [
                "Storage", "Policy", "ndofs", "nnz", "First (ms)", "First×", "First (B)",
                "Refill (ms)", "Refill×", "Refill (B)", "OK", "Note"
            ]
            data = Matrix{Any}(undef, length(rows), length(header))
            for (i, r) in enumerate(rows)
                data[i, :] = [
                    r.storage, r.policy, r.ndofs, r.nnz,
                    r.ok ? round(r.first_ms; digits = 3) : "-",
                    r.ok ? round(r.first_ms / ref.first_ms; digits = 3) : "-",
                    r.ok ? r.first_bytes : "-",
                    r.ok ? round(r.refill_ms; digits = 5) : "-",
                    r.ok ? round(r.refill_ms / ref.refill_ms; digits = 3) : "-",
                    r.ok ? r.refill_bytes : "-",
                    r.ok, r.note
                ]
            end
            _print_table(data; column_labels = header, fit_table_in_display_horizontally = false)
            _out()
            SMOKE || label != "1M dofs" || _print_replay_summary(dim, label, n, rows)
            _out()
            GC.gc()
        end
    end

    all_ok = !isempty(all_rows) && all(r.ok for r in all_rows)
    if all_ok
        _out("OK-STORAGE-POLICY-GRID")
    else
        _out("Disagreements or failures (withholding OK-STORAGE-POLICY-GRID):")
        for r in all_rows
            r.ok || _out("  dim=$(r.dim) n=$(r.n) $(r.storage)/$(r.policy): $(r.note)")
        end
    end
end

main()
