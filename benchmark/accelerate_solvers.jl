#===========================================================================#
# S5.4 (gpena/Bramble.jl#142, #246): Apple Accelerate against the existing
# sparse and dense factorisations, measured on this host.
#
# Usage:
#     julia --project=benchmark --threads=4 benchmark/accelerate_solvers.jl
#
# Internal -- the command above spawns these two as separate processes for
# the dense arm below; they are not meant to be run directly by a person:
#     julia --project=benchmark --threads=4 benchmark/accelerate_solvers.jl --dense-arm=plain
#     julia --project=benchmark --threads=4 benchmark/accelerate_solvers.jl --dense-arm=accelerate
#
# ## The trap: dense "before" and "after" in one process both measure Accelerate
#
# AppleAccelerate.jl v0.7.0 (checked directly: `methods(AppleAccelerate.lu)` etc. all
# empty) defines no dense `lu`, `cholesky`, `getrf`, `potrf`, `LAPACK` or `gemm` bindings.
# Its `__init__` instead calls `BLAS.lbt_forward(libacc; clear = false)`, registering
# Accelerate as a libblastrampoline provider. The moment `using AppleAccelerate` has been
# evaluated anywhere in a process, ordinary `LinearAlgebra.lu`/`cholesky` are *already*
# running on Accelerate's BLAS/LAPACK, with no Bramble dispatch involved
# (`src/solvers/accelerate_solver.jl`'s dense `accelerate_factorize` method docstring says
# the same thing). A dense "before" and "after" timed in the same process would therefore
# both measure Accelerate, and the ratio would be 1.0 by construction -- reproducible and
# meaningless. The only way to get a genuine dense baseline is a separate process that
# never loads `AppleAccelerate` at all, which is what `--dense-arm=plain` below is for.
#
# The sparse side does not have this problem: `accelerate_solve` goes through
# `AAFactorization`/`factor!` (Accelerate's `libSparse`), a distinct code path from
# SuiteSparse's UMFPACK/CHOLMOD, so both arms are real within a single process and the
# sparse comparisons below run in the main (no `--dense-arm`) invocation directly.
#
# ## Correctness note carried over from S5.3
#
# Two separately dispatched calls into the same Accelerate factorisation are not
# bit-identical -- vecLib reorders floating-point reductions between calls. This file
# times solves; it does not assert `==` against a reference solution anywhere.
#===========================================================================#

using Bramble
# `D₊ₓ`/`D₊ᵧ` are `public`, not exported (bramble-naming): the summation-by-parts pairing
# the convection-diffusion fixture below needs them explicitly, the same way
# test/ext/SolverContracts.jl does for the accuracy audit these numbers follow up on.
import Bramble: D₊ₓ, D₊ᵧ
using LinearAlgebra
using SparseArrays
using Printf
using PrettyTables
using BenchmarkTools

const DENSE_ARM = let i = findfirst(a -> startswith(a, "--dense-arm="), ARGS)
    i === nothing ? nothing : Symbol(split(ARGS[i], "=")[2])
end

# Every arm except the plain dense one needs AppleAccelerate loaded: the sparse
# comparisons in the main invocation call `accelerate_solve` directly, and the
# `--dense-arm=accelerate` subprocess needs the LBT forwarding `using AppleAccelerate`
# triggers. `--dense-arm=plain` is the one exception -- see the trap note above.
Sys.isapple() || error("accelerate_solvers.jl (S5.4, #142/#246) only measures the macOS-only AppleAccelerate path.")
if DENSE_ARM !== :plain
    using AppleAccelerate
end

set_zero_subnormals(true) # bramble-benchmarks: subnormal residual tails cost 10-100x

# --------------------------------------------------------------------- machine state
#
# bramble-verification §9 / bramble-benchmarks §1: gate on load and power, and print
# both so the figure below carries the machine state it was taken under, not just a
# number. Polls rather than refusing outright, since the dispatching agent found the
# machine briefly busy and settling.
function _on_ac_power()
    try
        return !occursin("Battery Power", read(`pmset -g batt`, String))
    catch
        return true
    end
end

function _wait_for_quiet_load(threshold::Real; max_wait_s::Real = 300, poll_s::Real = 15)
    elapsed = 0.0
    load1 = Sys.loadavg()[1]
    while load1 >= threshold && elapsed < max_wait_s
        @printf("  load average (1 min) = %.2f >= %.2f threshold, waiting %gs...\n",
            load1, threshold, poll_s)
        sleep(poll_s)
        elapsed += poll_s
        load1 = Sys.loadavg()[1]
    end
    return load1
end

function _print_header()
    threshold = Sys.CPU_THREADS / 2
    load1 = _wait_for_quiet_load(threshold)
    settled = load1 < threshold
    batt = read(`pmset -g batt`, String)
    println("="^100)
    println(" S5.4 -- Apple Accelerate vs. existing factorisations (gpena/Bramble.jl #142, #246)")
    println("="^100)
    @printf(" Julia %s, Bramble %s, %d threads, %s/%s\n", VERSION, pkgversion(Bramble),
        Threads.nthreads(), Sys.MACHINE, Sys.ARCH)
    @printf(" load average (1 min) = %.2f (threshold Sys.CPU_THREADS/2 = %.1f, settled = %s)\n",
        load1, threshold, settled)
    println(" power state (pmset -g batt):")
    for line in split(strip(batt), '\n')
        println("   ", line)
    end
    settled ||
        @warn "load average never settled below Sys.CPU_THREADS/2; figures below are taken under load -- read the ratios, not the absolute times"
    _on_ac_power() ||
        @warn "running on battery power: frequency scaling and thermal throttling make these timings unreliable"
    println("="^100)
    return nothing
end

# --------------------------------------------------------------------- fixtures
#
# Bramble-shaped matrices, not `sprand`: the SBP-discretised 2D Poisson operator (SPD once
# symmetrized -- the `:spd`/`:symmetric` dispatch paths) and the unsymmetric
# convection-diffusion operator (the `:unsymmetric` LUTPP path), the same two fixtures
# `test/ext/SolverContracts.jl` uses for the S5.3 accuracy audit these numbers follow up
# on -- so the conditioning here matches what that audit already checked for accuracy.

_unit_square() = interval(0.0, 1.0) × interval(0.0, 1.0)

function _poisson_system(n::Integer)
    Ωd = domain(_unit_square(), :boundary => boundary_symbols(_unit_square()))
    Wₕ = gridspace(mesh(Ωd, (n, n), (true, true)))
    fₕ = Rₕ(Wₕ, x -> sin(π * x[1]) * sin(π * x[2]))
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    A, F = assemble(a, l; dirichlet = :boundary => (x -> 0.0), symmetrize = true)
    return A, F
end

function _convection_diffusion_system(n::Integer; βx = 2.0, βy = 1.0)
    Ωd = domain(_unit_square(), :boundary => boundary_symbols(_unit_square()))
    Wₕ = gridspace(mesh(Ωd, (n, n), (true, true)))
    fₕ = Rₕ(Wₕ, x -> 1.0)
    a = form(
        Wₕ, Wₕ,
        (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + βx * innerₕ(D₊ₓ(u), v) + βy * innerₕ(D₊ᵧ(u), v)
    )
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    A, F = assemble(a, l; dirichlet = :boundary => (x -> 0.0), symmetrize = false)
    return A, F
end

# Several sizes, not one (bramble-benchmarks): separates a fixed dispatch/factorisation
# overhead from how the two solvers actually scale.
const SPARSE_SIZES = (40, 80, 160)  # n x n grid -> (n-1)^2 interior dofs
const DENSE_SIZES = (10, 20, 40)    # densified via Matrix(A); dense LU/Cholesky is O(n^3)

const ROWS = NamedTuple[]  # collected for the summary table at the end

function _record!(case, n, dofs, t_accel, t_baseline, baseline_name)
    push!(
        ROWS, (
            case = case, n = n, dofs = dofs, t_accel_ms = t_accel * 1e3,
            t_baseline_ms = t_baseline * 1e3, baseline = baseline_name,
            ratio = t_accel / t_baseline
        )
    )
    @printf("  n=%-4d dofs=%-7d accelerate=%9.3f ms  %s=%9.3f ms  accelerate/%s=%.3fx%s\n",
        n, dofs, t_accel * 1e3, baseline_name, t_baseline * 1e3, baseline_name,
        t_accel / t_baseline, t_accel > t_baseline ? "  (Accelerate SLOWER)" : "")
    return nothing
end

# --------------------------------------------------------------------- sparse comparisons
#
# The three symmetry paths `accelerate_factorize`/`accelerate_solve` dispatch on
# (`src/solvers/accelerate_solver.jl`, `ext/BrambleAppleAccelerateExt.jl`), each against
# the SuiteSparse routine that same symmetry maps to, plus the literal
# `pde_solve(:default)` comparison: the generic `A \ F` a caller got before S5.1, against
# the Accelerate auto-dispatch a caller gets now.

function _bench_sparse_spd_cholesky()
    println("\n-- sparse SPD Cholesky (accelerate vs. SuiteSparse CHOLMOD) --")
    for n in SPARSE_SIZES
        A, F = _poisson_system(n)
        dofs = length(F)
        t_accel = @belapsed accelerate_solve($A, $F; sym = :spd) samples=10 evals=1 seconds=15
        t_ss = @belapsed cholesky(Symmetric($A))\$F samples=10 evals=1 seconds=15
        _record!("sparse SPD Cholesky", n, dofs, t_accel, t_ss, "suitesparse")
    end
end

function _bench_sparse_symmetric_ldlt()
    println("\n-- sparse symmetric LDLᵀ (accelerate vs. SuiteSparse CHOLMOD) --")
    for n in SPARSE_SIZES
        # Same SPD fixture as the Cholesky case: `kind = :ldlt` forces the LDLᵀ dispatch
        # path regardless of the matrix's actual definiteness, exactly as
        # test/ext/appleaccelerate_ext.jl does for the S5.3 accuracy audit -- what is
        # being timed is the factorisation kind, not whether this system needed it.
        A, F = _poisson_system(n)
        dofs = length(F)
        t_accel = @belapsed accelerate_solve($A, $F; kind = :ldlt) samples=10 evals=1 seconds=15
        t_ss = @belapsed ldlt(Symmetric($A))\$F samples=10 evals=1 seconds=15
        _record!("sparse symmetric LDLᵀ", n, dofs, t_accel, t_ss, "suitesparse")
    end
end

function _bench_sparse_unsymmetric_lutpp()
    println("\n-- sparse unsymmetric LUTPP (accelerate vs. SuiteSparse UMFPACK) --")
    for n in SPARSE_SIZES
        A, F = _convection_diffusion_system(n)
        dofs = length(F)
        t_accel = @belapsed accelerate_solve($A, $F; sym = :unsymmetric) samples=10 evals=1 seconds=15
        t_ss = @belapsed lu($A)\$F samples=10 evals=1 seconds=15
        _record!("sparse unsymmetric LUTPP", n, dofs, t_accel, t_ss, "suitesparse")
    end
end

function _bench_pde_solve_default()
    println("\n-- pde_solve(:default): generic `\\` (pre-S5.1) vs. Accelerate auto-dispatch (post-S5.1) --")
    for n in SPARSE_SIZES
        A, F = _poisson_system(n)
        dofs = length(F)
        t_accel = @belapsed accelerate_solve($A, $F; sym = :auto) samples=10 evals=1 seconds=15
        # The generic `A \ F` SuiteSparse route a caller got before S5.1's default-routing
        # change: UMFPACK LU, ignoring the matrix's own symmetry, since `A` here is a bare
        # `SparseMatrixCSC` and not wrapped `Symmetric`.
        t_ss = @belapsed $A\$F samples=10 evals=1 seconds=15
        _record!("pde_solve(:default) equivalent", n, dofs, t_accel, t_ss, "suitesparse")
    end
end

# --------------------------------------------------------------------- dense comparison
#
# Runs only inside a `--dense-arm=...` subprocess (see the trap note at the top of this
# file). Prints one machine-parseable `DENSE_RESULT` line per (size, kind) so the parent
# process below can read it back without any inter-process state beyond stdout.
function _run_dense_arm(arm::Symbol)
    for n in DENSE_SIZES
        A, _ = _poisson_system(n)
        Ad = Matrix(A)
        dofs = size(Ad, 1)
        lu(Ad)                  # warm-up: pay JIT/compilation cost before timing
        cholesky(Symmetric(Ad))
        t_lu = @belapsed lu($Ad) samples=10 evals=1 seconds=15
        t_chol = @belapsed cholesky(Symmetric($Ad)) samples=10 evals=1 seconds=15
        println("DENSE_RESULT arm=$arm kind=lu n=$n dofs=$dofs time_s=$t_lu")
        println("DENSE_RESULT arm=$arm kind=cholesky n=$n dofs=$dofs time_s=$t_chol")
    end
    return nothing
end

if DENSE_ARM !== nothing
    _run_dense_arm(DENSE_ARM)
    exit(0)
end

# --------------------------------------------------------------------- main (default arm)

function _parse_dense_results(output::AbstractString)
    results = NamedTuple[]
    for line in split(output, '\n')
        startswith(line, "DENSE_RESULT") || continue
        fields = Dict{String, String}()
        for tok in split(line)[2:end]
            k, v = split(tok, '='; limit = 2)
            fields[k] = v
        end
        push!(
            results, (
                arm = Symbol(fields["arm"]), kind = Symbol(fields["kind"]),
                n = parse(Int, fields["n"]), dofs = parse(Int, fields["dofs"]),
                time_s = parse(Float64, fields["time_s"])
            )
        )
    end
    return results
end

function _bench_dense()
    println(
        "\n-- dense LU/Cholesky: two separate processes, since the moment this (main) ",
        "process's own `using AppleAccelerate` above runs, plain `LinearAlgebra.lu`/",
        "`cholesky` in THIS process are already Accelerate-forwarded -- see the trap note"
    )
    println(
        "   arm \"plain\"      = a fresh process that never loads AppleAccelerate (system BLAS/LAPACK, i.e. OpenBLAS)"
    )
    println(
        "   arm \"accelerate\" = a fresh process that loads AppleAccelerate before any LU/Cholesky call (LBT-forwarded)"
    )

    jl = Base.julia_cmd()
    proj = Base.active_project()
    this_file = @__FILE__
    nthreads = Threads.nthreads()

    cmd_plain = `$jl --project=$proj --threads=$nthreads --startup-file=no $this_file --dense-arm=plain`
    cmd_accel = `$jl --project=$proj --threads=$nthreads --startup-file=no $this_file --dense-arm=accelerate`

    out_plain = read(cmd_plain, String)
    out_accel = read(cmd_accel, String)

    plain = _parse_dense_results(out_plain)
    accel = _parse_dense_results(out_accel)

    for kind in (:lu, :cholesky)
        println("\n  dense $kind:")
        for n in DENSE_SIZES
            rp = only(filter(r -> r.n == n && r.kind == kind, plain))
            ra = only(filter(r -> r.n == n && r.kind == kind, accel))
            ratio = ra.time_s / rp.time_s
            push!(
                ROWS, (
                    case = "dense $kind", n = n, dofs = rp.dofs,
                    t_accel_ms = ra.time_s * 1e3, t_baseline_ms = rp.time_s * 1e3,
                    baseline = "plain(openblas)", ratio = ratio
                )
            )
            @printf("    n=%-4d dofs=%-5d accelerate-forwarded=%9.3f ms  plain(openblas)=%9.3f ms  accelerate/plain=%.3fx%s\n",
                n, rp.dofs, ra.time_s * 1e3, rp.time_s * 1e3, ratio,
                ratio > 1 ? "  (Accelerate SLOWER)" : "")
        end
    end
    return nothing
end

function main()
    _print_header()

    _bench_sparse_spd_cholesky()
    _bench_sparse_symmetric_ldlt()
    _bench_sparse_unsymmetric_lutpp()
    _bench_pde_solve_default()
    _bench_dense()

    println("\n" * "="^100)
    println(" summary")
    println("="^100)
    pretty_table(
        ROWS; column_labels = ["case", "n", "dofs", "accelerate (ms)", "baseline (ms)",
            "baseline", "ratio (accelerate/baseline)"],
        formatters = [fmt__printf("%.3f", [4, 5, 7])], display_size = (-1, -1)
    )

    losses = filter(r -> r.ratio > 1, ROWS)
    if isempty(losses)
        println("\nAccelerate is at or below the incumbent's time in every case measured.")
    else
        println("\nAccelerate is SLOWER than the incumbent in ", length(losses), " case(s):")
        for r in losses
            @printf("  %-32s n=%-4d ratio=%.3fx (baseline: %s)\n", r.case, r.n, r.ratio,
                r.baseline)
        end
    end

    return nothing
end

main()
