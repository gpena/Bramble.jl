#===========================================================================#
# Backend benchmark: CSC vs CSR -- assembly and direct solve (gpena/Bramble.jl
# #214, subplan S8 of .agents/plans/v3-3-0-memory-scaling.md).
#
# Usage:
#     JULIA_DEPOT_PATH="$TMPDIR/depot-<id>:$HOME/.julia" \
#         julia --startup-file=no --threads=4 --project=benchmark benchmark/backends.jl
#
# RESCOPED (see the plan's S8 entry): the banded and block-banded backends this
# script originally also measured were removed from the codebase on
# 2026-09-19 (`banded_backend`/`block_banded_backend` no longer exist, neither
# package is a dependency). What remains is the milestone's one non-CSC
# storage: `SparseMatrixCSR` (S3.1, `ext/BrambleSparseMatricesCSRExt.jl`),
# compared against the `backend()` (CSC) baseline.
#
# FIXED DEFECT: the script this replaces built `matrix = Matrix(A)` for its
# correctness check. At 300x300 that densifies a 90,000x90,000 matrix (about
# 64 GB); at 60^3 the 216,000^2 matrix is about 373 GB. The kernel killed the
# process with signal 9 before it printed a line. This version never
# densifies. The correctness oracle is two independent, nnz/O(1)-sized checks,
# neither of which builds an n^2 object, and each is held to the tolerance
# that matches what it actually measures:
#   (a) STORED-ENTRY AGREEMENT ("Matrix |Δ|", gated at the requested 1e-13,
#       absolute). A `SparseMatrixCSR`'s raw `(rowptr, colval, nzval)`
#       triplet is converted to a `SparseMatrixCSC` by `_to_csc` below -- an
#       O(nnz) triplet re-sort via `SparseArrays.sparse`, not an O(n^2)
#       densification. The two CSC objects are then subtracted
#       (`SparseArrays` sparse-sparse subtraction, itself O(nnz)) and the
#       largest stored magnitude in the difference is the discrepancy. This
#       can fail: a wrong entry, a missing entry, or a transposed index would
#       all show up as a nonzero difference. Verified separately (ad hoc,
#       outside this file) to be exactly 0.0 for 1D at n up to 1e5: the two
#       backends' assemblers reach bit-identical stored values here, so a
#       tight absolute bound is the right instrument for this check.
#   (b) ACTION AGREEMENT ("Action rel|Δ|", gated at 1e-6, relative). `A * x`
#       for several random `x` is compared against `Acsc * x`. This exercises
#       the backend's own `mul!`, not `_to_csc`'s conversion, so a bug in (a)
#       cannot hide behind a matching (b) and vice versa. Unlike (a) this is
#       a floating-point *recomputation*, not a stored value: CSC and CSR
#       walk the same nonzeros in a different order (column-major scatter vs.
#       row-major dot product), and floating-point addition is not
#       associative, so even two exactly-agreeing matrices produce `A*x`
#       values that differ at the units-in-the-last-place level scaled by
#       the entry magnitude -- at 1D's 1e5 points the stiffness entries
#       themselves reach ~2e5 (h = 1e-5), so an *absolute* 1e-13 bound on
#       this check is unsatisfiable by construction, not evidence of a wrong
#       matrix (bramble-verification §6: pair an absolute floor with a
#       relative one for comparisons whose natural scale moves). 1e-6
#       relative is still four to five orders tighter than a real defect
#       (wrong sign, dropped term, transposed index) would produce.
# `matrix_ok` requires both; a genuinely wrong CSR matrix fails (a), and a
# genuinely wrong `mul!` fails (b) even though (a) cannot see it.
#
# The direct solve is judged by its **residual**, not by agreement with CSC's
# solution vector. Measured directly on the 1D case at n = 1e5: the two
# assembled matrices and right-hand sides are bit-identical (max|Δ| = 0.0),
# yet the solutions differ by 1.1e-7 relative -- because each solve's own
# relative residual is about 2e-7. At h = 1e-5 this system is conditioned
# badly enough that UMFPACK itself does no better, so the two backends land on
# different points of the same tiny residual set and neither is more correct.
# A gate on solution agreement at 1e-9 sits below the accuracy either solver
# reaches, so it can only ever report the conditioning. The residual gate can
# still fail: a genuinely broken solve returns a residual orders larger, and
# the bound here is ten times CSC's own residual, floored at 1e-12.
#
# Correctness before timing, no exception (bramble-verification): a mismatch
# prints the discrepancy and withholds `OK-S8`. These are measurements, not
# assertions -- a backend slower than CSC is a result to print, not a case to
# leave out or tune away.
#===========================================================================#

using Bramble
using Bramble: allocate_system_matrix
using SparseMatricesCSR
using PrettyTables
using SparseArrays
using LinearAlgebra: lu, norm
using Random

# Flush subnormals to zero (bramble-benchmarks house rule): residual tails
# near 1e-308 trigger microcode execution and skew small-N timings.
set_zero_subnormals(true)

const ZERO_BC = :dir => (x -> 0.0)
const RNG = Random.Xoshiro(20260919)

# --- preflight: wait for a quiet machine (bramble-verification §9) -------- #

function _power_source()
    Sys.isapple() || return (on_battery = false, raw = "unknown (not macOS)")
    try
        out = strip(read(`pmset -g batt`, String))
        return (on_battery = occursin("Battery Power", out), raw = out)
    catch
        return (on_battery = false, raw = "unknown")
    end
end

# Polls `Sys.loadavg()[1]` (1-minute load) every `poll_s` seconds, up to
# `max_wait_s`, refusing to start until it drops under half the core count.
# Prints what it is waiting for on every poll so a watcher can see progress.
function _preflight(; poll_s = 30, max_wait_s = 20 * 60)
    threads = Sys.CPU_THREADS
    threshold = threads / 2
    waited = 0
    while true
        load1 = Sys.loadavg()[1]
        if load1 < threshold
            return (settled = true, load1 = load1, threads = threads)
        end
        println(
            "Preflight: 1-min load $(round(load1; digits = 2)) >= half of ",
            "$threads cores ($(round(threshold; digits = 2))); waiting ",
            "$(poll_s)s (waited $(waited)s of $(max_wait_s)s max)..."
        )
        if waited >= max_wait_s
            return (settled = false, load1 = load1, threads = threads)
        end
        sleep(poll_s)
        waited += poll_s
    end
end

function _print_header(preflight, power)
    println("Backend benchmark -- gpena/Bramble.jl#214 (S8): CSC vs SparseMatrixCSR")
    println("Threads       : ", Threads.nthreads(), " (", preflight.threads, " CPU cores)")
    println("1-min load    : ", round(preflight.load1; digits = 2))
    println("Power (pmset) : ", power.raw)
    if !preflight.settled
        println(
            "PREFLIGHT DID NOT SETTLE: the machine never quieted down within the poll ",
            "budget. Every figure below was taken under load; read them as noisier than ",
            "usual, and prefer the ratio columns to the absolute millisecond columns."
        )
    end
    if power.on_battery
        println(
            "ON BATTERY: CPU frequency scaling/thermal throttling can skew absolute ",
            "timings. Read the ratio columns (CSR/CSC), not the absolute ms columns."
        )
    end
    println()
end

# --- geometry, mesh and form, dimension-generic -------------------------- #

_unit_cube(::Val{1}) = interval(0.0, 1.0)
_unit_cube(::Val{D}) where {D} = reduce(×, ntuple(_ -> interval(0.0, 1.0), Val(D)))

_source(::Val{1}) = x -> sin(π * x)
_source(::Val{D}) where {D} = x -> prod(sin(π * xᵢ) for xᵢ in x)

_grid(::Val{1}, Ωd, n; backend) = mesh(Ωd, n, true; backend = backend)
function _grid(::Val{D}, Ωd, n; backend) where {D}
    mesh(
        Ωd, ntuple(_ -> n, Val(D)), ntuple(_ -> true, Val(D)); backend = backend
    )
end

# Poisson plus a mass term (test/form/bilinear.jl's own "mass" + "stiffness"
# combination, `innerₕ(u, v)` for the mass half).
_poisson_mass(u, v) = inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v)

const DIMS = ((1, 100_000), (2, 300), (3, 60))

# --- backends compared ------------------------------------------------------ #

struct BackendSpec
    name::String
    ctor::Function          # () -> Backend
end

const BACKENDS = (
    BackendSpec("CSC", () -> Bramble.backend()),
    BackendSpec("CSR", () -> csr_backend())
)

# --- non-densifying correctness oracle ------------------------------------- #

_to_csc(A::SparseMatrixCSC) = A

# O(nnz): re-lay the CSR triplet out as a CSC triplet through `sparse`, never
# an O(n^2) `Matrix(...)`. `rowptr`/`colval`/`nzval` are the same fields the
# extension itself scatters into (`ext/BrambleSparseMatricesCSRExt.jl`).
function _to_csc(A::SparseMatrixCSR{1})
    m, n = size(A)
    rowptr, colval, nzval = A.rowptr, A.colval, A.nzval
    I = Vector{Int}(undef, length(nzval))
    @inbounds for i in 1:m, k in rowptr[i]:(rowptr[i + 1] - 1)

        I[k] = i
    end
    return sparse(I, colval, nzval, m, n)
end

# (a) stored-entry agreement, O(nnz): sparse-sparse subtraction, never dense.
# Gated at the requested absolute 1e-13 -- see the header for why this check
# (not the action check below) is the one held to that bound.
function _stored_entry_discrepancy(A, Acsc_ref)
    Δ = _to_csc(A) - Acsc_ref
    return isempty(nonzeros(Δ)) ? 0.0 : maximum(abs, nonzeros(Δ))
end

# (b) action agreement, O(nnz) per multiply: several random vectors, not one.
# Returns the worst-case *relative* discrepancy (relative to the reference
# action's own magnitude), gated at 1e-6 -- see the header for why an
# absolute bound is the wrong instrument for this particular check.
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

# --- solve-path introspection: did `\` reach a sparse or a dense factor? --- #

function _factorization_kind(A)
    fact = lu(A)
    tname = string(typeof(fact))
    is_dense = occursin("LU{", tname) && occursin(", Matrix", tname)
    return is_dense ? "DENSE conversion ($tname)" : "sparse ($(nameof(typeof(fact))))"
end

# --- per-(backend, dimension) measurement --------------------------------- #

struct Row
    dim::Int
    backend::String
    n::Int
    nnz::Int
    first_ms::Float64
    refill_ms::Float64
    refill_bytes::Int
    size_mib::Float64
    solve_ms::Float64
    solve_kind::String
    stored_agree::Float64
    action_agree::Float64
    solve_agree::Float64
    matrix_ok::Bool
    solve_ok::Bool
    ran::Bool
    assemble_ratio::Float64
    refill_ratio::Float64
    solve_ratio::Float64
    note::String
end

# Function barrier (bramble-verification §1): the warm-up call and the
# measured call both happen inside one function, over its own arguments, not
# a testset/loop-local binding, so the reported allocation is real.
function _refill_allocs(A, a)
    assemble!(A, a)
    return @allocated assemble!(A, a)
end

function _refill_time_ms(A, a; trials = 5)
    assemble!(A, a)  # warm-up
    times = [(@elapsed assemble!(A, a)) for _ in 1:trials]
    return minimum(times) * 1000
end

function _solve(A, F)
    return A \ F
end

function _solve_time_ms(A, F; trials = 3)
    _solve(A, F)  # warm-up
    times = [(@elapsed _solve(A, F)) for _ in 1:trials]
    return minimum(times) * 1000
end

# Everything about one backend at one dimension: assemble, refill, solve.
# Returns the raw structured matrix and solution vector too, so the caller
# can run the non-densifying oracle against the CSC reference.
function _measure(dim::Int, n::Int, be, source)
    Iᴰ = _unit_cube(Val(dim))
    Ωd = domain(Iᴰ, :dir => boundary_symbols(Iᴰ))
    Ωₕ = _grid(Val(dim), Ωd, n; backend = be)
    Wₕ = gridspace(Ωₕ)
    fₕ = Rₕ(Wₕ, source)

    a = form(Wₕ, Wₕ, _poisson_mass)
    l = form(Wₕ, v -> innerₕ(fₕ, v))

    t_first = (@elapsed (A = assemble(a))) * 1000

    Awarm = allocate_system_matrix(a)
    refill_bytes = _refill_allocs(Awarm, a)
    refill_ms = _refill_time_ms(Awarm, a)
    size_bytes = Base.summarysize(Awarm)
    Awarm = nothing  # guard memory: not needed past here (§ instructions)

    Acd, F = assemble(a, l; dirichlet = ZERO_BC, symmetrize = true)
    solve_kind = _factorization_kind(Acd)
    t_solve = _solve_time_ms(Acd, F)
    x = _solve(Acd, F)

    # The residual, not the solution, is what the solve is judged on: see the header.
    residual = norm(Acd * x - F) / norm(F)

    return (
        matrix = A, n = size(A, 1), nnz = nnz(A), first_ms = t_first, refill_ms = refill_ms,
        refill_bytes = refill_bytes, size_mib = size_bytes / 1024^2, solve_ms = t_solve,
        solve_kind = solve_kind, solution = x, residual = residual
    )
end

function _run_dimension(dim::Int, n::Int)
    rows = Row[]
    source = _source(Val(dim))

    csc_spec = BACKENDS[1]
    csc = _measure(dim, n, csc_spec.ctor(), source)
    csc_ref = csc.matrix  # already SparseMatrixCSC; used as-is, no conversion
    push!(
        rows,
        Row(
            dim, csc_spec.name, csc.n, csc.nnz, csc.first_ms, csc.refill_ms,
            csc.refill_bytes, csc.size_mib, csc.solve_ms, csc.solve_kind, 0.0, 0.0, csc.residual,
            true, true, true, 1.0, 1.0, 1.0, "baseline"
        )
    )

    for spec in BACKENDS[2:end]
        local r
        try
            r = _measure(dim, n, spec.ctor(), source)
        catch e
            println("$(spec.name) failed for dim=$dim: $(sprint(showerror, e))")
            push!(
                rows,
                Row(
                    dim, spec.name, csc.n, 0, NaN, NaN, 0, NaN, NaN, "-", NaN, NaN, NaN,
                    false, false, false, NaN, NaN, NaN,
                    "failed: " * sprint(showerror, e)
                )
            )
            continue
        end

        stored_agree = _stored_entry_discrepancy(r.matrix, csc_ref)
        action_agree = _action_discrepancy(r.matrix, csc_ref)
        # Compare residuals, not solutions. Two backends solving the same
        # ill-conditioned system land on different points of the same tiny
        # residual set, and the gap between them says nothing about either.
        solve_agree = r.residual

        stored_ok = stored_agree <= 1e-13
        action_ok = action_agree <= 1e-6
        solve_ok = r.residual <= max(1e-12, 10 * csc.residual)
        matrix_ok = stored_ok && action_ok

        note = matrix_ok && solve_ok ? "OK" : ""
        stored_ok || (note *= "stored-entry mismatch max|Δ|=$stored_agree; ")
        action_ok || (note *= "action mismatch max rel|Δ|=$action_agree; ")
        solve_ok || (note *= "solve residual $(r.residual) against CSC's $(csc.residual)")
        if r.solve_kind != csc.solve_kind
            note *= "; solve path: $(r.solve_kind) (CSC: $(csc.solve_kind))"
        end

        push!(
            rows,
            Row(
                dim, spec.name, r.n, r.nnz, r.first_ms, r.refill_ms, r.refill_bytes,
                r.size_mib, r.solve_ms, r.solve_kind, stored_agree, action_agree, solve_agree,
                matrix_ok, solve_ok, true, r.first_ms / csc.first_ms,
                r.refill_ms / csc.refill_ms, r.solve_ms / csc.solve_ms, note
            )
        )
    end

    return rows
end

# --- driver ----------------------------------------------------------------- #

function main()
    preflight = _preflight()
    power = _power_source()
    _print_header(preflight, power)

    all_rows = Row[]
    for (dim, n) in DIMS
        append!(all_rows, _run_dimension(dim, n))
        GC.gc()  # guard memory before the next (larger) dimension
    end

    header = [
        "Dim", "Backend", "n", "nnz", "First (ms)", "First×", "Refill (ms)", "Refill×",
        "Refill (B)", "Matrix (MiB)", "Solve (ms)", "Solve×", "Solve path",
        "Matrix |Δ| (stored)", "Action rel|Δ|", "Solve residual", "OK", "Note"
    ]
    data = Matrix{Any}(undef, length(all_rows), length(header))
    for (row, r) in enumerate(all_rows)
        data[row, :] = [
            r.dim, r.backend, r.n, r.nnz,
            r.ran ? round(r.first_ms; digits = 3) : "-",
            r.ran ? round(r.assemble_ratio; digits = 3) : "-",
            r.ran ? round(r.refill_ms; digits = 5) : "-",
            r.ran ? round(r.refill_ratio; digits = 3) : "-",
            r.ran ? r.refill_bytes : "-",
            r.ran ? round(r.size_mib; digits = 3) : "-",
            r.ran ? round(r.solve_ms; digits = 3) : "-",
            r.ran ? round(r.solve_ratio; digits = 3) : "-",
            r.ran ? r.solve_kind : "-",
            r.ran ? r.stored_agree : "-",
            r.ran ? r.action_agree : "-",
            r.ran ? r.solve_agree : "-",
            r.ran ? (r.matrix_ok && r.solve_ok) : false,
            r.note
        ]
    end
    pretty_table(data; column_labels = header, fit_table_in_display_horizontally = false)
    println()

    ran_rows = filter(r -> r.ran, all_rows)
    all_agreed = !isempty(ran_rows) && all(r.matrix_ok && r.solve_ok for r in ran_rows)

    if all_agreed
        println("OK-S8")
    else
        println("Disagreements or failures (withholding OK-S8):")
        for r in all_rows
            (r.ran && r.matrix_ok && r.solve_ok) ||
                println("  dim=$(r.dim) backend=$(r.backend): $(r.note)")
        end
    end
end

main()
