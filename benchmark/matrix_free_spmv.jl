# Matrix-free apply against assembled SpMV on the host.
#
# For the SPD form innerₕ(u, v) + inner₊(κ∇ₕu, ∇ₕv) with κ = 1 + |x|² on non-uniform meshes
# of the unit interval, square and cube, each row times one `mul!(y, matrix_free_operator(a), x)`
# (`mf`) against one `mul!(y, assemble(a), x)` (`spmv`), both the minimum over BenchmarkTools
# samples after warm-up, and reports bytes by `Base.summarysize`: `csr_bytes` for the matrix,
# `mf_bytes` for everything the matrix-free operator keeps alive (the form it captures, with its
# mesh, κ and whatever the form caches after the timed products), and `form_bytes` for that
# form alone, so the operator's own share is `mf_bytes - form_bytes`. Once assembled, the CSR
# route can drop the form; the matrix-free route cannot. The matrix-free
# operator is timed twice, with the serial default and with `CpuThreaded()`, as
# separate rows with a `policy=` field; the SpMV is always the serial `SparseMatrixCSC` product
# of SparseArrays, the same matrix in both rows. After the rows, one crossover line per
# dimension and policy names the smallest ndofs from which matrix-free is faster than SpMV at
# every larger size tested (or "none in range"), and the CSR/matrix-free memory ratio there.
#
# Sizes that would make the CSR matrix exceed MEMORY_BUDGET are skipped (and printed as
# skipped); the estimate is (2D + 1) nonzeros per row at 16 bytes each (Float64 value, Int
# row index) plus the column pointers. Setting BRAMBLE_MFSPMV_QUICK=1 caps every ladder at its
# two smallest sizes, for debugging only.
#
# `--smoke` runs one small size per dimension and `--save PATH` writes the rows and the crossover
# lines to a results file (results_io.jl); stdout is the same either way.
#
# Run alone, on a quiet machine, on AC power, with four threads:
#     julia --project=benchmark --startup-file=no --threads=4 benchmark/matrix_free_spmv.jl

using Bramble: CpuSerial, CpuThreaded
using Bramble, BenchmarkTools, LinearAlgebra, Random, SparseArrays

include(joinpath(@__DIR__, "results_io.jl"))

set_zero_subnormals(true)

const SMOKE = "--smoke" in ARGS
const SAVE_PATH = let i = findfirst(==("--save"), ARGS)
    if i === nothing
        nothing
    elseif i == length(ARGS)
        error("--save requires a file path argument")
    else
        ARGS[i + 1]
    end
end

const QUICK = get(ENV, "BRAMBLE_MFSPMV_QUICK", "0") == "1"
const MEMORY_BUDGET = 2 * 2^30 # bytes of CSR matrix per size
const LADDERS = (
    1 => (10^3, 10^4, 10^5, 10^6, 10^7),
    2 => (32, 64, 128, 256, 512, 1024, 2048),
    3 => (16, 32, 64, 96, 128)
)

box(D) = reduce(×, ntuple(_ -> interval(0.0, 1.0), D))
spd(u, v, κ) = innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v))
csr_estimate(D, N) = (2D + 1) * N * 16 + (N + 1) * 8

function measure(D, n)
    Random.seed!(326) # the non-uniform nodes are random draws
    W = gridspace(mesh(domain(box(D)), ntuple(_ -> n, D), ntuple(_ -> false, D)))
    κ = Rₕ(W, x -> 1 + sum(abs2, x))
    a = form(W, W, (u, v) -> spd(u, v, κ))
    A = assemble(a)
    # The matrix-free operators get a form of their own: `assemble` fills the form's cache with
    # the scatter tables of the CSR route, which the matrix-free route never needs.
    amf = form(W, W, (u, v) -> spd(u, v, κ))
    N = ndofs(W)
    x = rand(N)
    y = similar(x)
    t_spmv = @belapsed mul!($y, $A, $x) samples=20 evals=1
    ref = A * x
    rows = map(("serial" => CpuSerial(), "threaded" => CpuThreaded())) do (name, p)
        op = matrix_free_operator(amf; policy = p)
        mul!(y, op, x)
        y ≈ ref || error("matrix-free and assembled products disagree at $(D)D n=$N")
        t_mf = @belapsed mul!($y, $op, $x) samples=20 evals=1
        (; D, N, policy = name, t_mf, t_spmv, csr = Base.summarysize(A),
            mfb = Base.summarysize(op), formb = Base.summarysize(amf))
    end
    return rows
end

function crossover(rows)
    # The smallest N from which matrix-free wins at every larger tested size.
    k = findlast(r -> r.t_mf >= r.t_spmv, rows)
    k == length(rows) && return nothing
    return rows[k === nothing ? 1 : k + 1]
end

function main()
    table = Dict{String, Any}[]
    crossovers = Dict{String, Any}[]
    println("Julia $(VERSION), $(Threads.nthreads()) threads, CSR budget ",
        MEMORY_BUDGET ÷ 2^20, " MiB", QUICK ? ", QUICK ladders" : "", SMOKE ? ", SMOKE" : "")
    for (D, ladder) in LADDERS
        results = Dict("serial" => [], "threaded" => [])
        for n in (SMOKE ? ladder[1:1] : QUICK ? ladder[1:2] : ladder)
            N = D == 1 ? n : n^D
            if csr_estimate(D, N) > MEMORY_BUDGET
                println("$(D)D n=$N skipped: CSR estimate exceeds the memory budget")
                continue
            end
            for r in measure(D, n)
                push!(results[r.policy], r)
                push!(table,
                    Dict{String, Any}("dim" => D, "ndofs" => r.N, "policy" => r.policy,
                        "mf_s" => r.t_mf, "spmv_s" => r.t_spmv, "ratio" => r.t_spmv / r.t_mf,
                        "csr_bytes" => r.csr, "mf_bytes" => r.mfb, "form_bytes" => r.formb))
                println("$(D)D n=$(r.N) policy=$(r.policy) mf=$(r.t_mf) s spmv=$(r.t_spmv) s ",
                    "ratio=$(round(r.t_spmv / r.t_mf, digits = 3)) csr_bytes=$(r.csr) ",
                    "mf_bytes=$(r.mfb) form_bytes=$(r.formb)")
            end
            GC.gc()
        end
        parts = map(("serial", "threaded")) do p
            c = crossover(results[p])
            row = Dict{String, Any}("dim" => D, "policy" => p)
            if c !== nothing
                row["from_ndofs"] = c.N
                row["memory_ratio"] = c.csr / c.mfb
            end
            push!(crossovers, row)
            c === nothing ? "$p none in range" :
            "$p from n=$(c.N) (memory ratio csr/mf=$(round(c.csr / c.mfb, digits = 1)))"
        end
        last_r = results["serial"][end]
        println("$(D)D crossover: ", join(parts, "; "),
            "; memory ratio at the largest size csr/mf=",
            round(last_r.csr / last_r.mfb, digits = 1))
    end
    if SAVE_PATH !== nothing
        save_results(SAVE_PATH, "matrix_free_spmv.jl",
            Dict{String, Any}("spmv" => table, "crossover" => crossovers); smoke = SMOKE)
        println("Results written to $SAVE_PATH")
    end
end

main()
