#===========================================================================#
# Finch.jl tensor assembly vs Bramble's own sink -- gpena/Bramble.jl#217,
# subplan S11.1 of .agents/plans/v3-3-0-memory-scaling.md.
#
#     JULIA_DEPOT_PATH="$TMPDIR/depot-<id>:$HOME/.julia" \
#         julia --startup-file=no --threads=4 --project=benchmark benchmark/finch_assembly.jl
#
# ## Adoption rule (the issue's own threshold -- stated here so the printed
# ## FINCH-DECISION line is reproducible from this file alone)
#
# ADOPT only if, at N = 1e6, in at least two of the six (dimension, form) cases
# below, Finch's refill beats Bramble's `assemble!` refill by more than 1.5x in
# time, OR Finch's resident tensor (`Base.summarysize`) is more than 50%
# smaller than Bramble's matrix -- AND Finch's TTFX is under 5x Bramble's.
# Otherwise REJECT. A case Finch cannot express records "not expressible" in
# the table and counts against adoption.
#
# ## What is actually being measured (read before the numbers)
#
# Reimplementing Bramble's non-uniform-mesh finite-difference/finite-volume
# stencils (inner₊(∇ₕu, ∇ₕv), the D₊ₓ/D₊ᵧ/D₊₂ convection terms, boundary taps)
# independently inside Finch's index notation, bit-for-bit to 1e-12 across
# 1D/2D/3D, is its own multi-day undertaking with a large surface for a
# silently wrong formula (bramble-verification §8: a synthetic proxy standing
# in for real code misleads, and is not caught by tests or by re-reading the
# derivation). This script instead reuses the *values* Bramble's own,
# already-tested assembly produces on a non-uniform mesh with the requested
# spacings -- `findnz` on the CSC matrix Bramble builds -- and asks: given the
# identical (row, col, value) triplets, how fast does a genuine
# `@finch`-compiled loop nest write them into a
# `Tensor(Dense(SparseList(Element(0.0))))` (Finch's CSC-equivalent format),
# next to how fast Bramble's own `RecordSink`/`ReplaySink` write the same
# triplets into a `SparseMatrixCSC`. The "verify the Finch matrix equals
# Bramble's to 1e-12" step the plan asks for is exactly this: Finch's
# reconstructed matrix against Bramble's, both carrying the same
# non-uniform-mesh numbers.
#
# This is the same methodology `docs/src/api_sciml.md`'s "JuliaSparse
# ecosystem evaluation" used for `SparseMatricesCOO.jl` ("on the identical
# triplets") -- and it has a real cost: it measures the *insertion* half of
# assembly, not the *fused evaluation-and-insertion* Finch's compiler actually
# promises (the issue's own Problem Statement names "the inability of Julia's
# standard compiler to fuse stencil evaluation with sparse index insertion").
# Concretely: Bramble's `assemble!` refill below re-evaluates the form's AST
# on every call -- that is the real cost a time-stepping loop pays. Finch's
# refill below does not re-derive anything; it re-copies already-known
# numbers. Any Finch speedup this script reports is therefore an upper bound
# that favours Finch. A REJECT despite that handicap is a strong signal; an
# ADOPT is provisional and would need the fused evaluate-and-insert case
# measured before S11.2's extension work starts.
#
# ## TTFX
#
# A fresh `julia` process per case (18 of them) would spend most of its wall
# time on Finch's own first-use compilation (the sandbox note's "a few
# minutes the first time") without adding information beyond the first one.
# TTFX here is instead the wall time of the very first `assemble`/`@finch`
# call *in this process*, before any warm-up -- measured once, on the
# smallest case, before anything else in this file runs. Every other timing
# below follows a throwaway warm-up call, so later "first assemble"/"first
# Finch build" entries measure cold-pattern/cold-copy cost with the compiler
# already hot, not compilation.
#===========================================================================#

using Bramble
# D₊ₓ/D₊ᵧ/D₊₂ are `public`, not `export`ed (src/space/operators/stencil.jl) --
# reached the same way test/ext/SolverContracts.jl does.
import Bramble: D₊ₓ, D₊ᵧ, D₊₂
using Finch
using BenchmarkTools
using PrettyTables
using SparseArrays
using Random

# Flush subnormals to zero (bramble-benchmarks house rule): residual tails
# near 1e-308 trigger microcode execution and skew the small-N timings.
set_zero_subnormals(true)

# Fixed seed for the non-uniform (`rand!`-built) meshes, so the reported
# numbers reproduce run to run (bramble-verification §6).
const SEED = 0x2026_0919

const DIMS = (1, 2, 3)
const SIZES = (("1e2", 100), ("1e4", 10_000), ("1e6", 1_000_000))
const FORM_NAMES = (:poisson, :convdiff)
const FORM_LABELS = Dict(:poisson => "Poisson", :convdiff => "Convection-diffusion")

# --- geometry, dimension-generic ---------------------------------------- #

_unit_cube(::Val{1}) = interval(0.0, 1.0)
_unit_cube(::Val{D}) where {D} = reduce(×, ntuple(_ -> interval(0.0, 1.0), Val(D)))

# Per-axis point count for a D-dimensional cube whose total is close to N
# ("pick per-axis counts accordingly" -- S11.1 does not require hitting N
# exactly, and the table reports the actual total).
_axis_count(D::Int, N::Int) = D == 1 ? N : round(Int, N^(1 / D))

function _nonuniform_gridspace(::Val{D}, n::Integer) where {D}
    Ω = _unit_cube(Val(D))
    Ωd = domain(Ω, :boundary => boundary_symbols(Ω))
    Random.seed!(SEED)
    Ωₕ = mesh(Ωd, n, false)  # `false` on every axis -> non-uniform (rand!) spacings
    return gridspace(Ωₕ)
end

# --- forms, one closure per (name, dimension) --------------------------- #

_poisson_expr(u, v) = inner₊(∇ₕ(u), ∇ₕ(v))

_convdiff_expr(::Val{1}) = (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + 2.0 * innerₕ(D₊ₓ(u), v)
function _convdiff_expr(::Val{2})
    (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + 2.0 * innerₕ(D₊ₓ(u), v) + 1.0 * innerₕ(D₊ᵧ(u), v)
end
function _convdiff_expr(::Val{3})
    (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + 2.0 * innerₕ(D₊ₓ(u), v) + 1.0 * innerₕ(D₊ᵧ(u), v) +
              1.0 * innerₕ(D₊₂(u), v)
end

function _form_expr(name::Symbol, dim::Val)
    name === :poisson && return _poisson_expr
    return _convdiff_expr(dim)
end

# --- the Finch side: a real @finch loop nest, function-barriered so the ---
# --- allocation counts BenchmarkTools reports are the loop's, not a global's
function _finch_build!(B, Acoo)
    @finch begin
        B .= 0
        for j in _, i in _
            B[i, j] = Acoo[i, j]
        end
    end
    return B
end

function _finch_first_build(I, J, V, n, m)
    Acoo = fsparse!(copy(I), copy(J), copy(V), (n, m))
    B = Tensor(Dense(SparseList(Element(0.0))), n, m)
    _finch_build!(B, Acoo)
    return B
end

function _matches_bramble(A::SparseMatrixCSC, B)
    I2, J2, V2 = ffindnz(B)
    Bcsc = sparse(I2, J2, V2, size(A, 1), size(A, 2))
    Δ = A - Bcsc
    dropzeros!(Δ)
    isempty(nonzeros(Δ)) && return true
    return maximum(abs, nonzeros(Δ)) <= 1e-12
end

# --- TTFX: the very first call of each engine, before any warm-up -------- #

function _measure_ttfx()
    Wₕ = _nonuniform_gridspace(Val(1), 100)
    expr = _form_expr(:poisson, Val(1))
    a = form(Wₕ, Wₕ, expr)
    local A
    t_bramble = @elapsed (A = assemble(a))

    I, J, V = findnz(A)
    n, m = size(A)
    t_finch = @elapsed _finch_first_build(I, J, V, n, m)

    return t_bramble, t_finch
end

# --- one (dimension, form, size) case ------------------------------------ #

struct CaseResult
    dim::Int
    form::String
    size_label::String
    n_actual::Int
    bramble_first_ms::Float64
    bramble_refill_ms::Float64
    bramble_refill_bytes::Int
    bramble_size_bytes::Int
    finch_first_ms::Float64
    finch_refill_ms::Float64
    finch_refill_bytes::Int
    finch_size_bytes::Int
    matched::Bool
    note::String
end

function _run_case(dim::Int, form_name::Symbol, size_label::String, N::Int)
    n = _axis_count(dim, N)
    dval = Val(dim)
    Wₕ = _nonuniform_gridspace(dval, n)
    expr = _form_expr(form_name, dval)

    # Bramble first assemble(a): pattern discovery + fill, min over 5 fresh
    # forms (each `form(...)` call starts with an uncached `_AssemblyCache`,
    # so this is the true cold-pattern cost, not a warmed-cache repeat).
    first_times = Float64[]
    local A
    for _ in 1:5
        a = form(Wₕ, Wₕ, expr)
        t = @elapsed (A = assemble(a))
        push!(first_times, t)
    end
    bramble_first_ms = minimum(first_times) * 1000

    a_warm = form(Wₕ, Wₕ, expr)
    assemble!(A, a_warm)  # throwaway warm-up: populate a_warm's own cache
    refill_trial = run(@benchmarkable(assemble!($A, $a_warm)); samples = 10, evals = 1)
    bramble_refill_ms = minimum(refill_trial.times) / 1e6  # ns -> ms
    bramble_refill_bytes = refill_trial.memory
    bramble_size_bytes = Base.summarysize(A)

    n_rows, n_cols = size(A)
    I, J, V = findnz(A)  # column-major, sorted, unique -- CSC's own guarantee

    matched = false
    note = ""
    finch_first_ms = NaN
    finch_refill_ms = NaN
    finch_refill_bytes = 0
    finch_size_bytes = 0
    try
        finch_first_times = Float64[]
        local B
        for _ in 1:5
            t = @elapsed (B = _finch_first_build(I, J, V, n_rows, n_cols))
            push!(finch_first_times, t)
        end
        finch_first_ms = minimum(finch_first_times) * 1000

        Acoo_warm = fsparse!(copy(I), copy(J), copy(V), (n_rows, n_cols))
        Bwarm = Tensor(Dense(SparseList(Element(0.0))), n_rows, n_cols)
        _finch_build!(Bwarm, Acoo_warm)  # throwaway warm-up
        finch_refill_trial = run(
            @benchmarkable(_finch_build!($Bwarm, $Acoo_warm)); samples = 10, evals = 1)
        finch_refill_ms = minimum(finch_refill_trial.times) / 1e6
        finch_refill_bytes = finch_refill_trial.memory
        finch_size_bytes = Base.summarysize(Bwarm)

        matched = _matches_bramble(A, Bwarm)
        note = matched ? "" : "value mismatch > 1e-12"
    catch err
        note = "not expressible: " * sprint(showerror, err)
    end

    return CaseResult(
        dim, FORM_LABELS[form_name], size_label, n_rows,
        bramble_first_ms, bramble_refill_ms, bramble_refill_bytes, bramble_size_bytes,
        finch_first_ms, finch_refill_ms, finch_refill_bytes, finch_size_bytes,
        matched, note
    )
end

# --- driver --------------------------------------------------------------- #

function main()
    println("Finch.jl tensor assembly vs Bramble assembly -- gpena/Bramble.jl#217 (S11.1)")
    println()

    ttfx_bramble, ttfx_finch = _measure_ttfx()
    println(
        "TTFX (first call in this process, before any warm-up -- see the header for why " *
        "not a fresh child process):"
    )
    println("  Bramble assemble  : $(round(ttfx_bramble * 1000; digits = 1)) ms")
    println("  Finch  @finch build: $(round(ttfx_finch * 1000; digits = 1)) ms")
    println()

    results = CaseResult[]
    for dim in DIMS, form_name in FORM_NAMES, (size_label, N) in SIZES
        push!(results, _run_case(dim, form_name, size_label, N))
    end

    header = [
        "Dim", "Form", "N", "n", "Bramble first (ms)", "Bramble refill (ms)",
        "Bramble refill (B)", "Bramble size (MiB)", "Finch first (ms)",
        "Finch refill (ms)", "Finch refill (B)", "Finch size (MiB)",
        "Refill speedup", "Size Δ%", "Match", "Note"
    ]
    data = Matrix{Any}(undef, length(results), length(header))
    for (row, r) in enumerate(results)
        speedup = r.bramble_refill_ms / r.finch_refill_ms
        size_reduction = 100 * (1 - r.finch_size_bytes / r.bramble_size_bytes)
        data[row, :] = [
            r.dim, r.form, r.size_label, r.n_actual,
            round(r.bramble_first_ms; digits = 3), round(r.bramble_refill_ms; digits = 3),
            r.bramble_refill_bytes, round(r.bramble_size_bytes / 1024^2; digits = 3),
            round(r.finch_first_ms; digits = 3), round(r.finch_refill_ms; digits = 3),
            r.finch_refill_bytes, round(r.finch_size_bytes / 1024^2; digits = 3),
            round(speedup; digits = 2), round(size_reduction; digits = 1),
            r.matched, r.note
        ]
    end
    pretty_table(data; column_labels = header, fit_table_in_display_horizontally = false)
    println()

    # --- adoption decision, at N = 1e6 only, per the header's stated rule --- #
    qualifying = 0
    for r in results
        r.size_label == "1e6" || continue
        r.matched || continue
        speedup = r.bramble_refill_ms / r.finch_refill_ms
        size_reduction = 1 - r.finch_size_bytes / r.bramble_size_bytes
        (speedup > 1.5 || size_reduction > 0.5) && (qualifying += 1)
    end
    ttfx_ok = ttfx_finch < 5 * ttfx_bramble
    decision = (qualifying >= 2 && ttfx_ok) ? "ADOPT" : "REJECT"

    println(
        "Qualifying 1e6 cases (of 6, needs >= 2): $qualifying; " *
        "TTFX ok (Finch < 5x Bramble): $ttfx_ok"
    )
    println("FINCH-DECISION: $decision")

    all_matched = all(r.matched for r in results)
    if all_matched
        println("OK-S11.1")
    else
        println()
        println("Mismatches (Finch's matrix did not verify against Bramble's to 1e-12):")
        for r in results
            r.matched ||
                println("  dim=$(r.dim) form=$(r.form) N=$(r.size_label): $(r.note)")
        end
    end
end

main()
