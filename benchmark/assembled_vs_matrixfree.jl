#===========================================================================#
# Assembled CSR SpMV vs matrix-free Kronecker apply on Metal -- gpena/Bramble.jl#317, S6.1
# of .claude/plans/v3-14-0-and-v3-15-0-gpu-policy-and-platform.md.
#
# #317 has not yet decided how much of the device-resident assembled-matrix architecture a
# matrix-free operator apply would make unnecessary (docs/src/internals/gpu.md, "Matrix-free
# Kronecker operators on a device" and the paragraph just above it). This script is the
# measurement that decision needs, not the decision itself: a separate subplan (S6.2) reads
# this script's numbers and writes the decision doc. Scope is deliberately narrow to what
# `KroneckerLinearOperator` actually covers (`src/assembly/kronecker.jl`): a separable
# bilinear form -- `innerₕ(u, v)` (mass) and `inner₊` of a backward difference on one axis on
# both sides (what `∇ₕ(u)`/`∇ₕ(v)` expand into), scalar/`Ref` coefficients, `D >= 2`
# (`is_separable`, kronecker.jl:130). No GpuOffload, no masked-projection timing, no
# scatter-table timing -- those are other subplans' territory.
#
# Two backends compared at each grid size, both on Metal, both Float32 (Apple Silicon has no
# Float64 GPU path -- `metal_backend`'s own docstring):
#
#   - "CSR SpMV": `assemble` a host `SparseMatrixCSC` from the form, upload it once with
#     `Bramble.metal_sparse_csr` (`ext/BrambleMetalExt.jl`), then `mul!` on the device.
#   - "Kronecker": build a `KroneckerLinearOperator` directly from a Metal-backed
#     `gridspace` (`kronecker_operator`, kronecker.jl:270) -- 1D mass/difference factors are
#     always assembled on the host mirror mesh then moved to device storage once
#     (`_kron_to_storage`); `mul!` then runs the whole operator as one fused
#     `KernelAbstractions` kernel with no per-axis host round-trip
#     (`_launch_kron_fused!`, `BrambleKernelAbstractionsExt`).
#
# Per size, three things are measured (the WHY's own (a)/(b)/(c)):
#
#   (a) Memory: CSR bytes (`nnz*(sizeof(Tv)+sizeof(Ti)) + (n+1)*sizeof(Ti)`, the assembled
#       matrix's own rowptr/colval/nzval) vs Kronecker bytes (the sum of every term's
#       per-axis factor arrays, host or device storage as actually built) vs this Mac's
#       `Metal.current_device().recommendedMaxWorkingSetSize`.
#   (b) Throughput: one device `mul!` apply per backend. This package is 2nd-order finite
#       differences, not high-order FEM -- there is no O(p^6) -> O(p^4) sum-factorisation win
#       to expect here (docs/src/internals/gpu.md's own framing); this measurement exists
#       only to confirm matrix-free doesn't *cost* throughput at this order, not to claim a
#       speedup.
#   (c) Regime: "assemble once, then 100 applies" vs "re-assemble every step, then 1 apply
#       each", for both arms. The re-assemble cost is a full rebuild each time (host
#       `assemble` + `metal_sparse_csr` upload for CSR; `kronecker_operator` for Kronecker),
#       not an incremental refill -- device refill optimisation (gpena/Bramble.jl#318, #338,
#       this plan's own S5.2) is explicitly out of this subplan's scope.
#
# Host-vs-device agreement is checked before any timing (bramble-verification: "a fast wrong
# answer is the failure mode"), for both arms independently, plus a host CSR-vs-Kronecker
# cross-check that the two backends are answering the same problem.
#
# Ratios are reported back-to-back, not bare absolutes (this repo's own convention).
# Load average and AC power state are printed per row, reusing
# `.claude/scripts/check_power_load.sh`'s stdout the same way `benchmark/scatter_table.jl`
# and `benchmark/gpu_offload.jl` already do (`_power_load_state` below is copied from
# there, not imported -- these are standalone scripts with no shared module).
#
# Reuses pieces from `benchmark/kronecker_device.jl` (device Kronecker construction, the
# host-mirror-mesh agreement pattern) and `benchmark/spmv_bench.jl` (CSR assembly + Metal
# upload, the CSC/CSR/Kronecker correctness-before-timing style) by copying, not importing.
#
# ## Usage
#
#     julia --project=benchmark --threads=4 benchmark/assembled_vs_matrixfree.jl
#     julia --project=benchmark --threads=4 benchmark/assembled_vs_matrixfree.jl --full
#
# Bare (this subplan's own CHECK): tiny sizes, few trials, no AC-power/load refusal -- fast
# enough to run unattended as a structural-plus-real-measurement check, and it must not fail
# just because the machine running it happens to be on battery. `--full` scales up to the
# sizes gpena/Bramble.jl#323 already established as fitting comfortably on this machine
# (`benchmark/kronecker_device.jl`, `docs/src/internals/gpu.md`'s own measured table: 2D up
# to 3000x3000, 3D up to 200x200x200 -- both a small fraction of this Mac's
# `recommendedMaxWorkingSetSize`, see the printed header), refuses to proceed on battery or
# under load like every other full sweep in this directory, and is meant to be run alone, on
# a quiet machine, by whoever picks up S6.2 -- not by this subplan.
#===========================================================================#

using Bramble
using Bramble: is_separable, kronecker_operator
using SparseArrays
using Metal
using KernelAbstractions
using BenchmarkTools
using LinearAlgebra: mul!, Diagonal, norm
using PrettyTables
using Random

set_zero_subnormals(true)

const FULL = "--full" in ARGS
const STEPS = 100
const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const POWER_SCRIPT = joinpath(REPO_ROOT, ".claude", "scripts", "check_power_load.sh")
const RNG = Random.Xoshiro(20260926)

function _out(msg::AbstractString = "")
    println(msg)
end

function _print_table(args...; kwargs...)
    buf = IOBuffer()
    pretty_table(buf, args...; fit_table_in_display_horizontally = false, kwargs...)
    for line in split(String(take!(buf)), '\n')
        isempty(line) || _out(line)
    end
end

# --- AC power / load gate, via .claude/scripts/check_power_load.sh (matches
# --- benchmark/scatter_table.jl's and benchmark/gpu_offload.jl's own `_power_load_state`,
# --- copied not imported -- FULL plays the role their SMOKE flag plays there, inverted: the
# --- bare run never refuses, --full always gates) ---------------------------------------- #

function _power_load_state(; poll_s = FULL ? 30 : 0)
    if !isfile(POWER_SCRIPT)
        return (ok = true, power = "unknown (script missing)", load = "unknown")
    end
    cmd = FULL ? `$POWER_SCRIPT --poll $(poll_s == 0 ? 1 : poll_s * 20)` : `$POWER_SCRIPT --allow-battery`
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
    return (ok = ok || !FULL, power = power, load = load)
end

if FULL
    gate = _power_load_state()
    if !gate.ok
        _out(
            "REFUSED: $(POWER_SCRIPT) reports the machine is not ready for measurement " *
            "(battery power, or load above threshold, after waiting). Plug in and/or " *
            "quiet the machine, then re-run.",
        )
        exit(1)
    end
end

_out("Assembled CSR SpMV vs matrix-free Kronecker apply on Metal -- gpena/Bramble.jl#317 (S6.1)")
_out(FULL ? "Mode          : --full (large sweep, gated on AC power/load)" :
     "Mode          : default (fast CHECK, ungated)")
_out("Julia threads : $(Threads.nthreads())" *
     (Threads.nthreads() == 4 ? "" : "  WARNING: expected 4 (bramble-benchmarks §1)"))
_out("Metal usable  : $(Metal.functional())")

if !Metal.functional()
    _out(
        "REFUSED: Metal.functional() == false on this machine. This script measures the " *
        "assembled-vs-matrix-free tradeoff specifically on Metal and has no host-only path.",
    )
    exit(1)
end

const DEVICE = Metal.current_device()
const WORKING_SET_BYTES = Int(DEVICE.recommendedMaxWorkingSetSize)
_out("Device        : $(DEVICE.name)")
_out("Recommended working set: $(round(WORKING_SET_BYTES / 1.0e9, digits = 2)) GB")
_out()

# --- Geometry / form: unit cube, Float32, uniform (matches kronecker_device.jl) ------- #

_unit_cube(::Val{D}) where {D} = reduce(×, ntuple(_ -> interval(0.0f0, 1.0f0), Val(D)))
_poisson_mass(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))

# --- Byte accounting: read off the real objects built below, never estimated ---------- #
#
# CSR: the three arrays actually uploaded by `Bramble.metal_sparse_csr` (rowptr length
# n+1, colval/nzval length nnz), counted on the host `SparseMatrixCSC` about to be
# uploaded -- CSR/CSC storage is the same three array *sizes*, transposed layout only.
#
# Kronecker: the sum of every term's per-axis factor arrays, as `kronecker_operator`
# actually built them for the space passed in -- a `Diagonal`'s `diag` vector on a
# host-backed operator, or a `_KronDeviceDiagonal`/`_KronDeviceSparse`'s device arrays on a
# Metal-backed one (kronecker.jl's own device storage, moved via `_kron_to_storage`).
# `Base.summarysize` cannot be used here: it undercounts a device array's buffer (it only
# sees the small Julia-side wrapper), so every array below is sized as
# `length(x) * sizeof(eltype(x))` instead.

_arr_bytes(x::AbstractArray) = length(x) * sizeof(eltype(x))

function _csr_bytes(A::SparseMatrixCSC)
    Ti = eltype(rowvals(A))
    return nnz(A) * (sizeof(eltype(A)) + sizeof(Ti)) + (size(A, 2) + 1) * sizeof(Ti)
end

_kron_factor_bytes(F::Diagonal) = _arr_bytes(F.diag)
_kron_factor_bytes(F::SparseMatrixCSC) = _csr_bytes(F)
_kron_factor_bytes(F::Bramble._KronDeviceDiagonal) = _arr_bytes(F.diag)
function _kron_factor_bytes(F::Bramble._KronDeviceSparse)
    return _arr_bytes(F.colptr) + _arr_bytes(F.rowval) + _arr_bytes(F.nzval)
end

function _kron_bytes(K)
    total = 0
    for term in K.terms
        for F in term.factors
            total += _kron_factor_bytes(F)
        end
    end
    return total
end

# --- Per-size measurement -------------------------------------------------------------- #

struct Row
    dim::String
    n::Int
    N::Int
    csr_bytes::Int
    kron_bytes::Int
    mem_ratio::Float64
    working_set_pct::Float64
    t_apply_csr::Float64
    t_apply_kron::Float64
    throughput_ratio::Float64
    t_build_csr::Float64
    t_build_kron::Float64
    once100_csr::Float64
    once100_kron::Float64
    once100_ratio::Float64
    reassemble100_csr::Float64
    reassemble100_kron::Float64
    reassemble100_ratio::Float64
    power::String
    load::String
end

function _run_size(D::Int, n::Int, dimlabel::String, rows::Vector{Row})
    poll_s = FULL ? 5 : 0
    gate = _power_load_state(; poll_s = poll_s)

    Ω = domain(_unit_cube(Val(D)))
    dims = ntuple(_ -> n, Val(D))
    unif = ntuple(_ -> true, Val(D))

    # Host Float32 CSC form, the CSR arm's assembled reference and the thing uploaded to
    # the device.
    Ωh = mesh(Ω, dims, unif; backend = Bramble.backend(Float32; policy = Serial()))
    Wh = gridspace(Ωh)
    a_csc = form(Wh, Wh, _poisson_mass)
    is_separable(a_csc) ||
        error("refuse: 2D/3D Poisson-mass form at $dimlabel n=$n is not separable (is_separable == false) -- KroneckerLinearOperator cannot cover it, so this script cannot build the comparison.")

    A_csc = assemble(a_csc)
    A_gpu = Bramble.metal_sparse_csr(A_csc)

    # Metal-backed mesh, the Kronecker arm's own operator (kronecker_device.jl's pattern).
    Ωd = mesh(Ω, dims, unif; backend = metal_backend())
    Wd = gridspace(Ωd)
    ad = form(Wd, Wd, _poisson_mass)
    is_separable(ad) ||
        error("refuse: device-backed form at $dimlabel n=$n is not separable (is_separable == false).")
    Kd = kronecker_operator(ad)

    # Host mirror Kronecker operator: an independent host answer to check the device
    # Kronecker apply against, matching kronecker_device.jl's own agreement pattern.
    Ωh_mirror = Bramble._host_mirror_mesh(mesh(Wd))
    Wh_mirror = gridspace(Ωh_mirror)
    ah = form(Wh_mirror, Wh_mirror, _poisson_mass)
    Kh = kronecker_operator(ah)

    N = ndofs(Wh)
    x = rand(RNG, Float32, N)
    x_gpu = MtlArray(x)

    # --- Correctness before timing (bramble-verification) -------------------------- #
    y_csr_host = A_csc * x
    y_kron_host = Kh * x
    @assert norm(y_kron_host - y_csr_host) / norm(y_csr_host) < 1.0f-3 "host CSR vs host Kronecker mismatch at $dimlabel n=$n"

    y_gpu_csr = similar(x_gpu)
    mul!(y_gpu_csr, A_gpu, x_gpu)
    Metal.synchronize()
    @assert norm(Array(y_gpu_csr) - y_csr_host) / norm(y_csr_host) < 1.0f-3 "device CSR vs host CSR mismatch at $dimlabel n=$n"

    y_gpu_kron = similar(x_gpu)
    mul!(y_gpu_kron, Kd, x_gpu)
    Metal.synchronize()
    @assert norm(Array(y_gpu_kron) - y_kron_host) / norm(y_kron_host) < 1.0f-3 "device Kronecker vs host Kronecker mismatch at $dimlabel n=$n"

    # --- (a) Memory ------------------------------------------------------------------ #
    csr_bytes = _csr_bytes(A_csc)
    kron_bytes = _kron_bytes(Kd)
    mem_ratio = csr_bytes / kron_bytes
    working_set_pct = 100 * csr_bytes / WORKING_SET_BYTES

    # --- (b) Throughput: one device mul! apply per backend --------------------------- #
    apply_samples = FULL ? 15 : 3
    t_apply_csr = minimum((@benchmark(begin
            mul!($y_gpu_csr, $A_gpu, $x_gpu)
            Metal.synchronize()
        end; samples = apply_samples, evals = 1)).times) / 1.0e6
    t_apply_kron = minimum((@benchmark(begin
            mul!($y_gpu_kron, $Kd, $x_gpu)
            Metal.synchronize()
        end; samples = apply_samples, evals = 1)).times) / 1.0e6
    throughput_ratio = t_apply_csr / t_apply_kron

    # --- (c) Regime: assemble-once-then-100-applies vs re-assemble-per-step-then-1 --- #
    build_samples = FULL ? 10 : 3
    t_build_csr = minimum((@benchmark(begin
            Ac = assemble($a_csc)
            Bramble.metal_sparse_csr(Ac)
        end; samples = build_samples, evals = 1)).times) / 1.0e6
    t_build_kron = minimum((@benchmark(kronecker_operator($ad); samples = build_samples, evals = 1)).times) / 1.0e6

    once100_csr = t_build_csr + STEPS * t_apply_csr
    once100_kron = t_build_kron + STEPS * t_apply_kron
    once100_ratio = once100_csr / once100_kron

    reassemble100_csr = STEPS * (t_build_csr + t_apply_csr)
    reassemble100_kron = STEPS * (t_build_kron + t_apply_kron)
    reassemble100_ratio = reassemble100_csr / reassemble100_kron

    push!(
        rows, Row(
            dimlabel, n, N, csr_bytes, kron_bytes, mem_ratio, working_set_pct,
            t_apply_csr, t_apply_kron, throughput_ratio, t_build_csr, t_build_kron,
            once100_csr, once100_kron, once100_ratio, reassemble100_csr, reassemble100_kron,
            reassemble100_ratio, gate.power, gate.load
        )
    )
    return nothing
end

# --- Sizes: fast default (this subplan's own CHECK) vs --full (the integrator's later,
# --- solo, quiet-machine sweep) -------------------------------------------------------- #

const SIZES = FULL ?
              ((2, (500, 1500, 3000)), (3, (60, 120, 200))) :
              ((2, (9, 17)), (3, (6, 9)))

function main()
    rows = Row[]
    t0 = time()
    for (D, ns) in SIZES
        dimlabel = D == 2 ? "2D" : "3D"
        for n in ns
            _run_size(D, n, dimlabel, rows)
            GC.gc()
        end
    end
    elapsed_min = (time() - t0) / 60

    _out("[Memory (a) and single-apply throughput (b)]")
    header1 = [
        "Dim", "n", "N (dofs)", "CSR bytes", "Kron bytes", "Mem CSR/Kron",
        "CSR/working-set %", "SpMV (ms)", "Kron apply (ms)", "SpMV/Kron", "Power", "Load"
    ]
    data1 = permutedims(hcat([[
                                  r.dim, r.n, r.N, r.csr_bytes, r.kron_bytes, round(r.mem_ratio, digits = 2),
                                  round(r.working_set_pct, digits = 4), round(r.t_apply_csr, digits = 4),
                                  round(r.t_apply_kron, digits = 4), round(r.throughput_ratio, digits = 3),
                                  r.power, r.load
                              ] for r in rows]...))
    _print_table(data1; column_labels = header1)
    _out()

    _out("[Regime (c): assemble-once-then-$(STEPS)-applies vs re-assemble-per-step-then-1-apply]")
    header2 = [
        "Dim", "n", "Build CSR (ms)", "Build Kron (ms)", "Once+$(STEPS) CSR (ms)",
        "Once+$(STEPS) Kron (ms)", "Once ratio", "Reassemble x$(STEPS) CSR (ms)",
        "Reassemble x$(STEPS) Kron (ms)", "Reassemble ratio"
    ]
    data2 = permutedims(hcat([[
                                  r.dim, r.n, round(r.t_build_csr, digits = 4), round(r.t_build_kron, digits = 4),
                                  round(r.once100_csr, digits = 2), round(r.once100_kron, digits = 2),
                                  round(r.once100_ratio, digits = 3), round(r.reassemble100_csr, digits = 2),
                                  round(r.reassemble100_kron, digits = 2), round(r.reassemble100_ratio, digits = 3)
                              ] for r in rows]...))
    _print_table(data2; column_labels = header2)
    _out()

    _out("Wall-clock time for this run: $(round(elapsed_min, digits = 2)) min.")
    _out()

    # --- Crossover line: does not classify a row as "matrix-free wins" or "assembled
    # --- wins" (bramble-benchmarks' own restraint, matching scatter_table.jl: "this script
    # --- does not classify a row either way -- it prints the numbers and leaves the
    # --- reading to whoever [...]"). It only flags the one thing this script is positioned
    # --- to flag cheaply: whether the assembled CSR matrix, at any tested size, starts
    # --- drawing a non-trivial share of the device's own recommended working set while the
    # --- Kronecker factors never approach it -- the memory pressure #317 needs to weigh.
    _out("Memory crossover (CSR share of the device's recommended working set):")
    threshold_pct = 50.0
    flagged = false
    for r in rows
        if r.working_set_pct >= threshold_pct
            _out(
                "  $(r.dim) n=$(r.n) (N=$(r.N)): CSR bytes are $(round(r.working_set_pct, digits = 2))% " *
                "of the recommended working set (>= $(threshold_pct)% threshold); Kronecker " *
                "factors are $(round(r.mem_ratio, digits = 1))x smaller at this size.",
            )
            flagged = true
        end
    end
    if !flagged
        worst = argmax(r -> r.working_set_pct, rows)
        _out(
            "  No tested size reached $(threshold_pct)% of the recommended working set " *
            "($(round(WORKING_SET_BYTES / 1.0e9, digits = 2)) GB); the largest share seen was " *
            "$(round(worst.working_set_pct, digits = 4))% at $(worst.dim) n=$(worst.n) " *
            "(N=$(worst.N)), where Kronecker factors were $(round(worst.mem_ratio, digits = 1))x " *
            "smaller than the assembled CSR matrix. Run --full for the larger, decision-scale sweep.",
        )
    end
    _out()
    println("OK-S6.1")
    return nothing
end

main()
