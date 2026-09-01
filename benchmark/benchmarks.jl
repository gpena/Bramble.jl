#===========================================================================#
# Performance regression suite.
#
# Ten groups, chosen to cover the paths where every slowdown found so far
# actually appeared, rather than to cover the API. Run it before tagging:
#
#     julia --project=benchmark benchmark/benchmarks.jl
#
# and compare against a saved baseline:
#
#     julia --project=benchmark benchmark/benchmarks.jl --save baseline.json
#     julia --project=benchmark benchmark/benchmarks.jl --compare baseline.json
#
# ## What is measured, and what is gated
#
# Allocations, not time, are the thing to gate on. Every regression chased in
# this package showed up first as bytes: the difference engine boxing its
# spacing callable (13,768 us and 6.4 MB against 185 us and none), the
# seminorm rebuilding a closure per point, the derivative weights boxing in 2D
# (12,047 us and 6.5 MB against 208 us and none). Allocation counts are also
# reproducible across machines and load, where wall-clock is not, so they can
# be asserted; timings are recorded to be read by a person.
#
# ## Why these six
#
#   - Rₕ! and avgₕ! above PARALLEL_FOR_MIN, which is the threaded branch the
#     test suite deliberately stays below so its allocation tests stay exact.
#   - D₋ₓ and D₋ᵧ on a large 2D grid: the stencil engine along the contiguous
#     direction and across it. The 2D case is the one that hid the derivative
#     weights regression, which 1D did not show.
#   - innerₕ and norm₁ₕ: the reduction path, including the seminorm's sum over
#     directions.
#   - ∇₋ₕ in 3D, where the boxing regressions were worst.
#   - one composite operator, which dispatches per component and so calls the
#     engine N times with a view rather than once with a vector.
#   - gridspace construction, which builds the quadrature weights, the path
#     that rebuilt a vector per axis on every call until recently.
#===========================================================================#

using BenchmarkTools
using Bramble
using DoubleFloats: Double64

# Nothing is imported beyond `Bramble` itself. PkgBenchmark and AirspeedVelocity `include`
# this file, so whatever it brings in lands in the including module, and `using Bramble`
# already makes `values` ambiguous there against `Base.values` — Bramble exports its own.
# Nothing below calls `values`, so the benchmarks do not depend on that resolving.

# Battery power means CPU frequency scaling and thermal throttling, so a timing
# taken on battery is not comparable with one taken on AC, and a baseline saved
# from one silently poisons every later `--compare`. Checked here rather than left
# to whoever runs it, because it was missed once: baseline_5f9b8af.json was
# recorded on battery and had to be withdrawn.
#
# Only macOS is checked. CI runners are mains-powered and there is no portable
# way to ask, so anything else is assumed fine rather than blocked.
function _on_ac_power()
    Sys.isapple() || return true
    try
        return !occursin("Battery Power", read(`pmset -g batt`, String))
    catch
        return true
    end
end

const SUITE = BenchmarkGroup()

# Sizes are chosen to sit above the threading threshold where that is the point
# of the benchmark, and to stay large enough elsewhere that per-call overhead
# does not dominate.
const N1 = 1_000_000                      # 1D, comfortably above PARALLEL_FOR_MIN
const N2 = (1000, 1000)                   # 2D, 1e6 points
const N3 = (100, 100, 100)                # 3D, 1e6 points

_mesh1() = mesh(domain(interval(0.0, 1.0)), N1, true)
_mesh2() = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), N2, (true, true))
_mesh3() = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), N3, (true, true, true))

# --- 1. restriction & cell-averaging across 1D, 2D, 3D -------------------- #
let W1 = gridspace(_mesh1()), u1 = element(W1), W2 = gridspace(_mesh2()), u2 = element(W2),
    W3 = gridspace(_mesh3()), u3 = element(W3)

    g = SUITE["restriction"] = BenchmarkGroup()
    g["Rₕ! 1D"] = @benchmarkable Rₕ!($u1, sin)
    g["avgₕ! 1D"] = @benchmarkable avgₕ!($u1, sin)
    g["Rₕ! 2D"] = @benchmarkable Rₕ!($u2, x -> sin(x[1]) * x[2])
    g["avgₕ! 2D"] = @benchmarkable avgₕ!($u2, x->sin(x[1])*x[2]) samples=5 evals=1
    g["Rₕ! 3D"] = @benchmarkable Rₕ!($u3, x -> sin(x[1]) + x[3])
    g["avgₕ! 3D"] = @benchmarkable avgₕ!($u3, x->sin(x[1])+x[3]) samples=3 evals=1
    g["Rₕ 1D (allocates its output)"] = @benchmarkable Rₕ($W1, sin)
end

# --- 2. the stencil engine, both directions ------------------------------- #
let Wₕ = gridspace(_mesh2()), uₕ = Rₕ(Wₕ, x -> sin(x[1]) * x[2])
    g = SUITE["operators 2D"] = BenchmarkGroup()
    g["D₋ₓ"] = @benchmarkable D₋ₓ($uₕ)            # along the contiguous direction
    g["D₋ᵧ"] = @benchmarkable D₋ᵧ($uₕ)            # across it
    g["M₋ₓ"] = @benchmarkable M₋ₓ($uₕ)
    g["Dcₓ"] = @benchmarkable Dcₓ($uₕ)
end

# --- 3. reductions -------------------------------------------------------- #
let uₕ = Rₕ(gridspace(_mesh2()), x -> sin(x[1]) * x[2])
    g = SUITE["inner products 2D"] = BenchmarkGroup()
    g["innerₕ"] = @benchmarkable innerₕ($uₕ, $uₕ)
    g["normₕ"] = @benchmarkable normₕ($uₕ)
    g["snorm₁ₕ"] = @benchmarkable snorm₁ₕ($uₕ)   # sums over directions
    g["norm₁ₕ"] = @benchmarkable norm₁ₕ($uₕ)
end

# --- 4. 3D, where the boxing regressions were worst ----------------------- #
let uₕ = Rₕ(gridspace(_mesh3()), x -> sin(x[1]) + x[3])
    g = SUITE["operators 3D"] = BenchmarkGroup()
    g["∇₋ₕ"] = @benchmarkable ∇₋ₕ($uₕ)
    g["D₋₂"] = @benchmarkable D₋₂($uₕ)
    g["innerₕ"] = @benchmarkable innerₕ($uₕ, $uₕ)
end

# --- 5. the composite dispatch -------------------------------------------- #
let Vₕ = gridspace(_mesh2(), Val(3))
    cₕ = Rₕ(Vₕ, (x -> x[1] * x[2], x -> sin(x[1]), x -> x[2]^2))
    g = SUITE["composite"] = BenchmarkGroup()
    g["D₋ₓ (3 components)"] = @benchmarkable D₋ₓ($cₕ)
    g["∇₋ₕ (3 components)"] = @benchmarkable ∇₋ₕ($cₕ)
end

# --- 6. construction ------------------------------------------------------ #
let Ωₕ2 = _mesh2(), Ωₕ3 = _mesh3()
    g = SUITE["construction"] = BenchmarkGroup()
    g["gridspace 2D"] = @benchmarkable gridspace($Ωₕ2)   # builds the weights
    g["gridspace 3D"] = @benchmarkable gridspace($Ωₕ3)
    g["hₘₐₓ 3D"] = @benchmarkable hₘₐₓ($Ωₕ3)
end

# --- 7. jumps and averages ------------------------------------------------ #
let uₕ2 = Rₕ(gridspace(_mesh2()), x -> sin(x[1]) * x[2]),
    uₕ3 = Rₕ(gridspace(_mesh3()), x -> sin(x[1]) + x[3])

    g = SUITE["jumps & averages"] = BenchmarkGroup()
    g["jumpₓ 2D"] = @benchmarkable jumpₓ($uₕ2)
    g["jumpᵧ 2D"] = @benchmarkable jumpᵧ($uₕ2)
    g["M₊ₓ 2D"] = @benchmarkable M₊ₓ($uₕ2)
    g["M₊ᵧ 2D"] = @benchmarkable M₊ᵧ($uₕ2)
    g["jump₂ 3D"] = @benchmarkable jump₂($uₕ3)
    g["M₊₂ 3D"] = @benchmarkable M₊₂($uₕ3)
end

# --- 8. startup latency & TTFX -------------------------------------------- #
let jl = Base.julia_cmd(),
    cmd_load = `$jl --project=. --startup-file=no -e "using Bramble"`,
    cmd_ttfx = `$jl --project=. --startup-file=no -e "using Bramble; m = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (10, 10), (true, true)); W = gridspace(m); u = element(W); D₋ₓ(u)"`

    g = SUITE["startup & latency"] = BenchmarkGroup()
    g["using Bramble"] = @benchmarkable run($cmd_load) samples=3 evals=1
    g["TTFX (load + first operator)"] = @benchmarkable run($cmd_ttfx) samples=3 evals=1
end

# --- 9. form assembly ----------------------------------------------------- #
#
# `form`, `assemble` and friends are not exported, so they are reached through
# `Bramble.` rather than imported: the note at the top of this file keeps the
# including module free of everything but `Bramble` itself.
#
# The matrix path uses a smaller grid than the vector path. Matrix assembly costs
# an order more than a vector fill at the same size, and what is being watched
# here is the allocation count and the shape of the cost, neither of which needs
# a million degrees of freedom to show a regression.
let W1 = gridspace(_mesh1()), f1 = Rₕ(W1, sin), v1 = Rₕ(W1, cos),
    l1 = Bramble.form(W1, v -> innerₕ(f1, v)),
    b1 = Bramble.assemble(l1), ast1 = Bramble.resolve_form_ast(l1),
    W2 = gridspace(_mesh2()), f2 = Rₕ(W2, x -> sin(x[1]) * x[2]),
    l2 = Bramble.form(W2, v -> innerₕ(f2, v)),
    b2 = Bramble.assemble(l2), ast2 = Bramble.resolve_form_ast(l2),
    Wm = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)),
        (300, 300), (true, true))),
    am = Bramble.form(Wm, Wm, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v))),
    Am = Bramble.assemble(am), astm = Bramble.resolve_form_ast(am)

    g = SUITE["forms"] = BenchmarkGroup()

    # construction, which built the whole AST eagerly until recently: a linear
    # form cost 48 MB and a bilinear one 9.3 MB, where both are now under 400 KB.
    #
    # These two measure 0.001 ns, which is BenchmarkTools reporting that the call
    # was optimised away: a form is now three stored fields and nothing else, so
    # there is no work left to elide. Kept as a regression guard rather than a
    # measurement — reintroduce eager work and the number stops being zero.
    g["form (linear, 2D)"] = @benchmarkable Bramble.form($W2, v -> innerₕ($f2, v))
    g["form (bilinear, 2D)"] = @benchmarkable Bramble.form(
        $Wm, $Wm, (u, v) -> innerₕ(u, v))

    # filling a vector that already exists, which is the time-loop call
    g["assemble! 1D"] = @benchmarkable Bramble.assemble!($b1, $l1; ast = $ast1)
    # `assemble_parallel!` takes the AST positionally where `assemble!` takes it as a
    # keyword — an inconsistency between the two entry points, not a typo here.
    g["assemble_parallel! 1D"] = @benchmarkable Bramble.assemble_parallel!(
        $b1, $l1, $ast1)
    g["assemble! 2D"] = @benchmarkable Bramble.assemble!($b2, $l2; ast = $ast2)
    g["assemble_parallel! 2D"] = @benchmarkable Bramble.assemble_parallel!(
        $b2, $l2, $ast2)

    # the fused contraction builds no vector at all, so it must allocate nothing
    g["l(vₕ) 1D"] = @benchmarkable $l1($v1)
    g["evaluate! 1D"] = @benchmarkable Bramble.evaluate!($b1, $l1, $v1; ast = $ast1)

    # the matrix: the pattern is built once, then refilled in place
    g["allocate_system_matrix 2D"] = @benchmarkable Bramble.allocate_system_matrix($am) samples=5 evals=1
    g["assemble! (matrix) 2D"] = @benchmarkable Bramble.assemble!(
        $Am, $am; ast = $astm) samples=5 evals=1
end

# --- 10. the same work in three precisions -------------------------------- #
#
# Until the stencil weights were typed, a Float32 form assembled a Float64
# vector, so timing single precision measured double precision and this
# comparison could not have meant anything. It can now.
#
# 1D and 100,000 points: above PARALLEL_FOR_MIN, and small enough that Double64,
# which is software arithmetic and an order slower, does not dominate the suite.
# `avgₕ!` is the expensive one either way — six quadrature nodes per point — and
# it is the path where `_gauss_rule` is built per call for an extended type.
let N = 100_000
    g = SUITE["precision 1D"] = BenchmarkGroup()

    for (lbl, T) in (("Float32", Float32), ("Float64", Float64),
        ("Double64", Double64))
        Ω = mesh(domain(interval(T(0), T(1))), N, true)
        W = gridspace(Ω)
        u = element(W)
        fₕ = Rₕ(W, sin)
        l = Bramble.form(W, v -> innerₕ(fₕ, v))
        b = Bramble.assemble(l)
        ast = Bramble.resolve_form_ast(l)

        g["Rₕ! $lbl"] = @benchmarkable Rₕ!($u, sin)
        g["avgₕ! $lbl"] = @benchmarkable avgₕ!($u, sin) samples=20
        g["assemble! $lbl"] = @benchmarkable Bramble.assemble!($b, $l; ast = $ast)
        g["innerₕ $lbl"] = @benchmarkable innerₕ($fₕ, $fₕ)
    end
end

# --- allocation bounds, which are the part worth gating ------------------- #
#
# The number an operator is allowed to allocate is one output vector: the
# `similar` inside it, and nothing else. These are the exact counts as of
# writing, not upper bounds with slack, because the failure mode being guarded
# against is a type instability that silently spills memory on every cell.
const ALLOCATION_BOUNDS = Dict(
    # contiguous-direction difference: 3 allocs for similar(::VectorElement)
    ("operators 2D", "D₋ₓ") => 3,
    ("operators 2D", "D₋ᵧ") => 3,
    ("operators 2D", "M₋ₓ") => 3,
    ("operators 2D", "Dcₓ") => 3,
    ("operators 3D", "D₋₂") => 3,
    # one per spatial direction
    ("operators 3D", "∇₋ₕ") => 15,
    # reductions allocate nothing at all
    ("inner products 2D", "innerₕ") => 0,
    ("inner products 2D", "normₕ") => 0,
    ("inner products 2D", "snorm₁ₕ") => 0,
    ("inner products 2D", "norm₁ₕ") => 0,
    ("operators 3D", "innerₕ") => 0,
    ("construction", "hₘₐₓ 3D") => 0,
    # jumps and averages
    ("jumps & averages", "jumpₓ 2D") => 3,
    ("jumps & averages", "jumpᵧ 2D") => 3,
    ("jumps & averages", "M₊ₓ 2D") => 3,
    ("jumps & averages", "M₊ᵧ 2D") => 3,
    ("jumps & averages", "jump₂ 3D") => 3,
    ("jumps & averages", "M₊₂ 3D") => 3,
    # form assembly. Only the zeros are gated, deliberately: `assemble_parallel!`
    # and `Rₕ!`/`avgₕ!` allocate one task set per call, so their counts move with
    # the thread count, and `allocate_system_matrix` builds three coordinate
    # vectors whose count can move with the Julia version. Those are printed
    # every run and read by a person, which is how the `restriction` group is
    # already treated.
    #
    # A vector fill into a buffer that exists allocates nothing, and neither does
    # refilling a sparse matrix whose pattern is fixed. The contraction is the
    # sharpest of them: `l(vₕ)` builds no vector at all, and was 721,168 B before
    # it was fused.
    ("forms", "assemble! 1D") => 0,
    ("forms", "assemble! 2D") => 0,
    ("forms", "assemble! (matrix) 2D") => 0,
    ("forms", "evaluate! 1D") => 0,
    ("forms", "l(vₕ) 1D") => 0,
    ("forms", "form (linear, 2D)") => 0,
    ("forms", "form (bilinear, 2D)") => 0,
    # the same assembly in three precisions: none of them may allocate, and the
    # reduction is the path where a widened element type would show first
    ("precision 1D", "assemble! Float32") => 0,
    ("precision 1D", "assemble! Float64") => 0,
    ("precision 1D", "assemble! Double64") => 0,
    ("precision 1D", "innerₕ Float32") => 0,
    ("precision 1D", "innerₕ Float64") => 0,
    ("precision 1D", "innerₕ Double64") => 0)

"""
	check_allocations(results)

Compares the allocation counts in `results` against [`ALLOCATION_BOUNDS`](@ref) and prints
one line per entry. Returns `true` when every bound holds.

This is the part of the suite that can be gated in CI: allocation counts are reproducible
across machines and load, and every performance regression this package has had showed up
here before it showed up in a timing.
"""
function check_allocations(results)
    ok = true
    println("\nallocation bounds")
    for ((group, name), bound) in sort(collect(ALLOCATION_BOUNDS), by = first)
        got = allocs(results[group][name])
        pass = got <= bound
        ok &= pass
        println("  ", pass ? "ok   " : "FAIL ", rpad("$group / $name", 40),
            "allocs = ", got, " (bound ", bound, ")")
    end
    return ok
end

function main(args = ARGS)
    set_zero_subnormals(true)
    ac = _on_ac_power()
    ac || @warn string("running on battery power: frequency scaling and thermal ",
        "throttling make these timings unreliable, and --save is refused. ",
        "Allocation counts are unaffected and still gated.")
    println("tuning...")
    tune!(SUITE)
    results = run(SUITE; verbose = true)
    # The thread count belongs in the tags as much as the Julia version does.
    # Without it, two baselines are indistinguishable while measuring different
    # code: `_parallel_for!` takes its serial branch on one thread and allocates
    # nothing, and allocates one task set — 22 allocations — on more. Comparing a
    # single-threaded baseline against a four-threaded run therefore reports a
    # memory regression on every threaded path, which is what
    # baseline_15f5e3b.json against baseline_5f9b8af.json did: `Rₕ! 1D` moved
    # from 0 allocations to 22, and nothing had changed but the thread count.
    #
    # Baselines saved before this tag existed carry no thread count. The ones in
    # `baselines/` read 0 allocations for `Rₕ!`, so they were single-threaded.
    append!(results.tags,
        ["julia:$(VERSION)", "os:$(Sys.KERNEL)", "arch:$(Sys.ARCH)",
            "threads:$(Threads.nthreads())", "power:$(ac ? "ac" : "battery")"])

    println("\ntimings (median)")
    for (gname, group) in sort(collect(results), by = first)
        println("  ", gname)
        for (bname, trial) in sort(collect(group), by = first)
            m = median(trial)
            println("    ", rpad(bname, 34), lpad(BenchmarkTools.prettytime(time(m)), 12),
                lpad(BenchmarkTools.prettymemory(memory(m)), 12),
                lpad(string(allocs(m), " allocs"), 14))
        end
    end

    passed = check_allocations(results)

    i = findfirst(==("--save"), args)
    if i !== nothing && i < length(args)
        ac || error("refusing to save a baseline recorded on battery power. Plug in " *
                    "and re-run; the allocation gate above is still valid.")
        BenchmarkTools.save(args[i + 1], results)
        println("\nsaved baseline to ", args[i + 1], " (Julia $VERSION)")
    end

    j = findfirst(==("--compare"), args)
    if j !== nothing && j < length(args)
        baseline = BenchmarkTools.load(args[j + 1])[1]
        base_jl = "unknown"
        for t in baseline.tags
            startswith(string(t), "julia:") &&
                (base_jl = replace(string(t), "julia:" => ""))
        end
        base_th = "unknown"
        for t in baseline.tags
            startswith(string(t), "threads:") &&
                (base_th = replace(string(t), "threads:" => ""))
        end
        println("\nagainst ", args[j + 1], " (baseline: Julia $base_jl on $base_th ",
            "thread(s), current: Julia $VERSION on $(Threads.nthreads()))")
        base_th == string(Threads.nthreads()) ||
            @warn "thread counts differ, so memory on every threaded path is not comparable" baseline=base_th current=Threads.nthreads()
        jdg = judge(minimum(results), minimum(baseline))
        for (gname, group) in BenchmarkTools.leaves(jdg)
            group.time != :invariant && println("  ", join(gname, " / "), "  time ",
                group.time, "  memory ", group.memory)
        end
    end

    passed || error("allocation bounds exceeded")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
