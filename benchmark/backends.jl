#===========================================================================#
# Backend benchmark: CSC vs CSR vs Banded/BlockBanded -- assembly and direct
# solve (gpena/Bramble.jl#175 #216 #214, subplan S8 of
# .agents/plans/v3-3-0-memory-scaling.md).
#
# Usage:
#     JULIA_DEPOT_PATH="$TMPDIR/depot-<id>:$HOME/.julia" \
#         julia --startup-file=no --threads=4 --project=benchmark benchmark/backends.jl
#
# Compares the four matrix backends this milestone added against the plain
# `SparseMatrixCSC` baseline: `backend()` (CSC), `csr_backend()`,
# `banded_backend()` (1D only) and `block_banded_backend()` (2D/3D only). A
# backend that does not apply to a dimension is skipped, printed with its
# reason, not silently omitted. #215's tridiagonal backend was dropped
# 2026-09-19 (ABANDONED, see the plan's S2.1) and is not benchmarked here.
#
# Form measured: 1D Poisson plus a mass term, `inner₊(∇ₕ(u), ∇ₕ(v)) +
# innerₕ(u, v)`, and its dimension-generic 2D/3D equivalent -- `∇ₕ`/`innerₕ`
# are already dimension-generic, the same way `benchmark/finch_assembly.jl`'s
# `_poisson_expr` reuses one closure across 1D/2D/3D. Sizes: 1e5 points (1D),
# 300x300 (2D), 60x60x60 (3D).
#
# Correctness before timing, no exception (bramble-verification): every
# backend's assembled matrix must equal CSC's to 1e-13 and every direct solve
# must agree with CSC's solve to 1e-9. A timing of a wrong matrix is not a
# measurement -- a mismatch prints the discrepancy and withholds `OK-S8`.
# These are measurements, not assertions: a backend slower than CSC is a
# result to print, not a case to leave out or tune away.
#===========================================================================#

using Bramble
using BandedMatrices
using BlockBandedMatrices
using SparseMatricesCSR
using BenchmarkTools
using PrettyTables
using SparseArrays

# Flush subnormals to zero (bramble-benchmarks house rule): residual tails
# near 1e-308 trigger microcode execution and skew small-N timings.
set_zero_subnormals(true)

const ZERO_BC = :dir => (x -> 0.0)

# --- machine state (bramble-verification/bramble-benchmarks: a printed ---
# --- timing means nothing without power source, thread count and load) --- #

function _power_source()
    Sys.isapple() || return "unknown (not macOS)"
    try
        return occursin("Battery Power", read(`pmset -g batt`, String)) ? "BATTERY" : "AC"
    catch
        return "unknown"
    end
end

function _load_average()
    try
        return strip(read(`uptime`, String))
    catch
        return "unknown"
    end
end

function _print_header()
    println("Backend benchmark -- gpena/Bramble.jl#175 #216 #214 (S8)")
    println("Power source : ", _power_source())
    println("Threads      : ", Threads.nthreads())
    println("Load         : ", _load_average())
    println()
end

# --- geometry, mesh and form, dimension-generic -------------------------- #

_unit_cube(::Val{1}) = interval(0.0, 1.0)
_unit_cube(::Val{D}) where {D} = reduce(×, ntuple(_ -> interval(0.0, 1.0), Val(D)))

_source(::Val{1}) = x -> sin(π * x)
_source(::Val{D}) where {D} = x -> prod(sin(π * xᵢ) for xᵢ in x)

_grid(::Val{1}, Ωd, n; backend) = mesh(Ωd, n, true; backend = backend)
_grid(::Val{D}, Ωd, n; backend) where {D} = mesh(
    Ωd, ntuple(_ -> n, Val(D)), ntuple(_ -> true, Val(D)); backend = backend
)

# Poisson plus a mass term (test/form/bilinear.jl's own "mass" + "stiffness"
# combination, `innerₕ(u, v)` for the mass half).
_poisson_mass(u, v) = inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v)

const DIMS = ((1, 100_000), (2, 300), (3, 60))

# --- backends compared ----------------------------------------------------- #

struct BackendSpec
    name::String
    ctor::Function          # () -> Backend
    applicable::Function    # dim::Int -> Bool
    reason::String          # printed when `applicable(dim)` is false
end

const BACKENDS = (
    BackendSpec("CSC", () -> Bramble.backend(), _ -> true, ""),
    BackendSpec("CSR", () -> csr_backend(), _ -> true, ""),
    BackendSpec(
        "Banded", () -> banded_backend(), d -> d == 1,
        "banded_backend is 1D-only; block_banded_backend covers 2D/3D"
    ),
    BackendSpec(
        "BlockBanded", () -> block_banded_backend(), d -> d in (2, 3),
        "block_banded_backend needs the block structure 2D/3D lexicographic ordering gives; banded_backend covers 1D"
    )
)

# --- per-(backend, dimension) measurement --------------------------------- #

struct Row
    dim::Int
    backend::String
    n::Int
    first_ms::Float64
    refill_ms::Float64
    refill_bytes::Int
    size_mib::Float64
    solve_ms::Float64
    matrix_ok::Bool
    solve_ok::Bool
    ran::Bool
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

    Acd, F = assemble(a, l; dirichlet = ZERO_BC, symmetrize = true)
    t_solve = (@elapsed (x = Acd \ F)) * 1000

    return (matrix = Matrix(A), n = size(A, 1), first_ms = t_first, refill_ms = refill_ms,
        refill_bytes = refill_bytes, size_mib = size_bytes / 1024^2, solve_ms = t_solve,
        solution = x)
end

function _run_dimension(dim::Int, n::Int)
    rows = Row[]
    source = _source(Val(dim))

    csc_spec = BACKENDS[1]
    csc = _measure(dim, n, csc_spec.ctor(), source)
    push!(
        rows,
        Row(
            dim, csc_spec.name, csc.n, csc.first_ms, csc.refill_ms, csc.refill_bytes,
            csc.size_mib, csc.solve_ms, true, true, true, "baseline"
        )
    )

    for spec in BACKENDS[2:end]
        if !spec.applicable(dim)
            println("Skipping $(spec.name) for dim=$dim: $(spec.reason)")
            push!(
                rows,
                Row(
                    dim, spec.name, csc.n, NaN, NaN, 0, NaN, NaN, false, false, false,
                    "skipped: " * spec.reason
                )
            )
            continue
        end

        r = _measure(dim, n, spec.ctor(), source)

        matrix_ok = isapprox(r.matrix, csc.matrix; atol = 1e-13)
        solve_ok = isapprox(r.solution, csc.solution; atol = 1e-9)

        note = ""
        if !matrix_ok
            Δ = maximum(abs, r.matrix .- csc.matrix)
            note *= "matrix mismatch max|Δ|=$Δ; "
        end
        if !solve_ok
            Δs = maximum(abs, r.solution .- csc.solution)
            note *= "solve mismatch max|Δ|=$Δs"
        end
        if matrix_ok && solve_ok
            note = "OK"
        end

        push!(
            rows,
            Row(
                dim, spec.name, r.n, r.first_ms, r.refill_ms, r.refill_bytes, r.size_mib,
                r.solve_ms, matrix_ok, solve_ok, true, note
            )
        )
    end

    return rows
end

# --- driver ----------------------------------------------------------------- #

function main()
    _print_header()

    all_rows = Row[]
    for (dim, n) in DIMS
        append!(all_rows, _run_dimension(dim, n))
    end

    header = [
        "Dim", "Backend", "n", "First assemble (ms)", "Refill (ms)", "Refill (B)",
        "Matrix (MiB)", "Solve A\\F (ms)", "Matrix OK", "Solve OK", "Note"
    ]
    data = Matrix{Any}(undef, length(all_rows), length(header))
    for (row, r) in enumerate(all_rows)
        data[row, :] = [
            r.dim, r.backend, r.n,
            r.ran ? round(r.first_ms; digits = 3) : "-",
            r.ran ? round(r.refill_ms; digits = 5) : "-",
            r.ran ? r.refill_bytes : "-",
            r.ran ? round(r.size_mib; digits = 3) : "-",
            r.ran ? round(r.solve_ms; digits = 3) : "-",
            r.ran ? r.matrix_ok : "-",
            r.ran ? r.solve_ok : "-",
            r.note
        ]
    end
    pretty_table(data; column_labels = header, fit_table_in_display_horizontally = false)
    println()

    ran_rows = filter(r -> r.ran, all_rows)
    all_agreed = all(r.matrix_ok && r.solve_ok for r in ran_rows)

    if all_agreed
        println("OK-S8")
    else
        println("Disagreements (a backend's matrix or solve did not verify against CSC):")
        for r in ran_rows
            (r.matrix_ok && r.solve_ok) ||
                println("  dim=$(r.dim) backend=$(r.backend): $(r.note)")
        end
    end
end

main()
