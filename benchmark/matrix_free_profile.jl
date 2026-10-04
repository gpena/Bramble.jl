# Where the matrix-free product's time goes (gpena/Bramble.jl#428).
#
# For the SPD form innerₕ(u, v) + inner₊(∇ₕu, ∇ₕv) on graded non-uniform meshes (2D 512²,
# 3D 64³, built as `operator_routes.jl` builds them), attributes the serial and the threaded
# `MatrixFreeOperator` product to the issue's five causes by substitution: each cause is
# isolated by a controlled variant timed back to back with the real product.
#
#   1. one fused sweep against D + 1 term sweeps: Σ per-term walks minus the full product;
#   2. entries emitted per point against the merged row stencil's: the same scatter loop fed
#      the emitted entries minus fed the merged ones (both cached, in stencil-slot order);
#   3. weights evaluated per call: the real product minus the same walk fed a constant weight
#      (the evaluation, now unused, is dropped by the compiler); the same walk reading its
#      weights from a cached vector is printed beside it;
#   4. the interior walk's code: `@code_llvm` of `_visit_interior!` grepped for vector
#      arithmetic, and the merged scatter minus a hand-written contiguous loop over the
#      stencil's diagonals with the same arithmetic;
#   5. threading: the `CpuThreaded()` product minus serial / threads, split into the
#      launch of empty bands and the band work.
#
# Every figure is the minimum over repeated products, in ms. A `CAUSE` line's `ms` is the
# time that cause accounts for (negative: the variant is slower than the real product).
# Output rows, per dimension D:
#   PROFILE dim=D serial_ms=.. threaded_ms=.. assembled_ms=.. kronecker_ms=..
#   CAUSE dim=D cause=c ms=.. <the variants' times>
#   ENTRIES dim=D emitted=.. merged=..
#   VECTORISED dim=D interior=<yes|no>
#
# Run alone, on a quiet machine, on AC power, with four threads:
#     julia --project=benchmark --startup-file=no --threads=4 benchmark/matrix_free_profile.jl

using Bramble
using Bramble: CpuSerial, CpuThreaded
using InteractiveUtils: code_llvm
using Kronecker
using LinearAlgebra
using Random
using SparseArrays

const REPEATS = 60
const POWER_SCRIPT = joinpath(@__DIR__, "..", ".claude", "scripts", "check_power_load.sh")

box(D) = reduce(×, ntuple(_ -> interval(0.0, 1.0), D))

# The mesh of `operator_routes.jl`: uniform, then cosine-clustered along axis 1 and a power
# grading along the others, so no axis is uniform and no two share their nodes.
function graded_points(n, d)
    t = range(0.0, 1.0; length = n)
    return d == 1 ? @.(0.5 * (1 - cos(π * t))) : t .^ (1 + 0.25 * d)
end

function graded_space(D, n)
    Ωₕ = mesh(domain(box(D)), ntuple(_ -> n, D), ntuple(_ -> true, D))
    Bramble.change_points!(Ωₕ, ntuple(d -> graded_points(n, d), D))
    return gridspace(Ωₕ)
end

# Minimum time of `f()` in ms over `REPEATS` runs after one warm-up.
function best_ms(f)
    f()
    return minimum(@elapsed(f()) for _ in 1:REPEATS) * 1e3
end

# --- Capturing the walk's units ------------------------------------------------------- #
# A policy whose `_mf_visit!` records each (term, leaf) the operator's unit walk reaches,
# so the per-term sweeps and the `@code_llvm` use the very terms the product walks.
struct _Capture
    units::Vector{Any}
end
function Bramble._mf_visit!(c::_Capture, s, term, sp, ro::Int, co::Int)
    push!(c.units, (term, sp, ro, co))
    return nothing
end

# The real walk with each entry's weight read from `ws` instead of the one it is handed, in
# the order a collecting walk of the same form stored them.
mutable struct _CachedWeightSink
    const y::Vector{Float64}
    const x::Vector{Float64}
    const α::Bool
    const ws::Vector{Float64}
    k::Int
end
@inline function Bramble._sink_entry!(s::_CachedWeightSink, row::Int, col::Int, _, ::Int)
    s.k += 1
    @inbounds s.y[row] += s.α * s.ws[s.k] * s.x[col]
    return nothing
end

# The real walk with every weight replaced by one: the stencil's weight evaluation is unused.
struct _UnitWeightSink
    y::Vector{Float64}
    x::Vector{Float64}
    α::Bool
end
@inline function Bramble._sink_entry!(s::_UnitWeightSink, row::Int, col::Int, _, ::Int)
    @inbounds s.y[row] += s.α * s.x[col]
    return nothing
end

# --- Entry collection -------------------------------------------------------------------#
struct _CollectSink
    rows::Vector{Int}
    cols::Vector{Int}
    ws::Vector{Float64}
end
@inline function Bramble._sink_entry!(s::_CollectSink, row::Int, col::Int, w, ::Int)
    push!(s.rows, row)
    push!(s.cols, col)
    push!(s.ws, w)
    return nothing
end

function scatter!(y, x, α, rows, cols, ws)
    @inbounds for k in eachindex(ws)
        y[rows[k]] += α * ws[k] * x[cols[k]]
    end
    return y
end

# The stencil's diagonals, read off the assembled matrix: `W[k][i] = A[i, i + offs[k]]`.
function diagonals(A, offs)
    n = size(A, 1)
    W = [zeros(n) for _ in offs]
    rows = rowvals(A)
    vals = nonzeros(A)
    for j in 1:n, p in nzrange(A, j)

        k = findfirst(==(j - rows[p]), offs)
        k === nothing || (W[k][rows[p]] = vals[p])
    end
    return W
end

# The product of the diagonals' flat loop: contiguous, unit-stride over axis 1.
function diagonal_product!(y, x, α, W, offs, lo, hi)
    m = length(offs)
    @inbounds for k in 1:m
        Wk = W[k]
        o = offs[k]
        @simd for i in lo:hi
            y[i] += α * Wk[i] * x[i + o]
        end
    end
    return y
end

# Whether `_visit_interior!` with an `ActionSink` carries vector arithmetic in its LLVM.
function interior_vectorised(sink, unit)
    term, sp, ro, co = unit
    Ωₕ = mesh(sp)
    ax = axes(Bramble.indices(Ωₕ))
    margin = Bramble._stencil_margin(term)
    region = CartesianIndices(map(r -> Bramble._interior_range(r, margin), ax))
    args = (sink, term, sp, Bramble.markers(Ωₕ), LinearIndices(Bramble.indices(Ωₕ)), region, ro, co)
    io = IOBuffer()
    code_llvm(io, Bramble._visit_interior!, Tuple{map(typeof, args)...};
        optimize = true, debuginfo = :none, dump_module = false)
    ir = String(take!(io))
    return occursin(r"(fmul|fadd|fma|fsub)[^\n]*<\d+ x double>", ir)
end

function profile(D, n)
    Random.seed!(428)
    W = graded_space(D, n)
    a = form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
    N = ndofs(W)
    x = rand(N)
    y = zeros(N)
    α = true
    nthr = Threads.nthreads()
    op = matrix_free_operator(a)
    opT = matrix_free_operator(a; policy = CpuThreaded())
    A = assemble(a)
    K = kronecker_operator(a)

    t_serial = best_ms(() -> mul!(y, op, x, α, false))
    t_thr = best_ms(() -> mul!(y, opT, x, α, false))
    t_asm = best_ms(() -> mul!(y, A, x, α, false))
    t_kron = best_ms(() -> mul!(y, K, x, α, false))
    println("PROFILE dim=$D serial_ms=$(round(t_serial; digits = 4)) ",
        "threaded_ms=$(round(t_thr; digits = 4)) assembled_ms=$(round(t_asm; digits = 4)) ",
        "kronecker_ms=$(round(t_kron; digits = 4))")

    # Cause 1: the walk's units, each swept alone, against the one product.
    cap = _Capture(Any[])
    Bramble._mf_apply!(cap, nothing, a)
    units = cap.units
    sink = Bramble.ActionSink(y, x, α, nothing)
    sweep(u) = (Bramble.visit_bilinear_stencil(sink, u[1], u[2], u[3], u[4]); nothing)
    t_terms = [best_ms(() -> (fill!(y, 0); sweep(u))) for u in units]
    t_fill = best_ms(() -> fill!(y, 0))
    t_sum = sum(t_terms) - (length(units) - 1) * t_fill
    c1 = t_sum - t_serial
    terms_txt = join(round.(t_terms .- t_fill; digits = 3), "+")
    println("CAUSE dim=$D cause=1 ms=$(round(c1; digits = 4)) $(length(units)) term sweeps ",
        "sum_ms=$(round(t_sum; digits = 4)) (terms $terms_txt) vs fused serial_ms=$(round(t_serial; digits = 4))")

    # Cause 2 and 3: the emitted entries cached, then the merged ones.
    coll = _CollectSink(Int[], Int[], Float64[])
    Bramble._mf_apply!(CpuSerial(), coll, a)
    mr, mc, mw = findnz(A)
    order = sortperm(mc .- mr; alg = MergeSort) # stencil-slot major, rows increasing
    mr, mc, mw = mr[order], mc[order], mw[order]
    emitted, merged = length(coll.ws), length(mw)
    println("ENTRIES dim=$D emitted=$emitted merged=$merged")
    t_emit = best_ms(() -> (fill!(y, 0); scatter!(y, x, α, coll.rows, coll.cols, coll.ws))) - t_fill
    t_merge = best_ms(() -> (fill!(y, 0); scatter!(y, x, α, mr, mc, mw))) - t_fill
    c2 = t_emit - t_merge
    println("CAUSE dim=$D cause=2 ms=$(round(c2; digits = 4)) emitted_scatter_ms=$(round(t_emit; digits = 4)) ",
        "merged_scatter_ms=$(round(t_merge; digits = 4)) emitted/merged=$(round(emitted / merged; digits = 2))")
    cached = _CachedWeightSink(y, x, α, coll.ws, 0)
    run_cached() = (fill!(y, 0); cached.k = 0; Bramble._mf_apply!(CpuSerial(), cached, a); nothing)
    run_cached()
    mul!(y, op, x, α, false)
    y_real = copy(y)
    run_cached()
    y ≈ y_real || error("cached-weight walk disagrees with the product")
    t_cached = best_ms(run_cached) - t_fill
    unit = _UnitWeightSink(y, x, α)
    t_unit = best_ms(() -> (fill!(y, 0); Bramble._mf_apply!(CpuSerial(), unit, a); nothing)) - t_fill
    c3 = t_serial - t_fill - t_unit
    println("CAUSE dim=$D cause=3 ms=$(round(c3; digits = 4)) real_ms=$(round(t_serial - t_fill; digits = 4)) ",
        "constant_weight_walk_ms=$(round(t_unit; digits = 4)) ",
        "cached_weight_walk_ms=$(round(t_cached; digits = 4))")

    # Cause 4: vector code in the walk, and the contiguous loop with the same arithmetic.
    vec = interior_vectorised(sink, first(units))
    dims = ntuple(_ -> n, D)
    strides = cumprod((1, Base.front(dims)...))
    offs = sort!(unique(vcat(0, collect(strides), -collect(strides))))
    Wd = diagonals(A, offs)
    lo, hi = 1 + last(strides), N - last(strides)
    t_diag = best_ms(() -> (fill!(y, 0); diagonal_product!(y, x, α, Wd, offs, lo, hi))) - t_fill
    c4 = t_merge - t_diag
    println("CAUSE dim=$D cause=4 ms=$(round(c4; digits = 4)) merged_scatter_ms=$(round(t_merge; digits = 4)) ",
        "contiguous_axis1_loop_ms=$(round(t_diag; digits = 4)) llvm_vector_arithmetic=$(vec ? "yes" : "no")")
    println("VECTORISED dim=$D interior=$(vec ? "yes" : "no")")

    # Cause 5: the threaded product against serial / threads, launch against band work.
    launch() = @sync begin
        for _ in 2:nthr
            Threads.@spawn nothing
        end
    end
    t_launch = best_ms(launch)
    c5 = t_thr - t_serial / nthr
    println("CAUSE dim=$D cause=5 ms=$(round(c5; digits = 4)) threaded_ms=$(round(t_thr; digits = 4)) ",
        "serial/$nthr=$(round(t_serial / nthr; digits = 4)) launch_ms=$(round(t_launch; digits = 4)) ",
        "band_work_ms=$(round(t_thr - t_launch; digits = 4)) speedup=$(round(t_serial / t_thr; digits = 2))")
    return nothing
end

println("# threads=$(Threads.nthreads()); ", strip(read(`bash $POWER_SCRIPT`, String)))
for (D, n) in ((2, 512), (3, 64))
    profile(D, n)
end
