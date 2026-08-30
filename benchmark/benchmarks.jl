#===========================================================================#
# Performance regression suite.
#
# Six benchmarks, chosen to cover the paths where every slowdown found so far
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

# Nothing is imported beyond `Bramble` itself. PkgBenchmark and AirspeedVelocity `include`
# this file, so whatever it brings in lands in the including module, and `using Bramble`
# already makes `values` ambiguous there against `Base.values` — Bramble exports its own.
# Nothing below calls `values`, so the benchmarks do not depend on that resolving.

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

# --- 1. restriction, on the threaded branch ------------------------------- #
let Wₕ = gridspace(_mesh1()), uₕ = element(gridspace(_mesh1()))
    g = SUITE["restriction"] = BenchmarkGroup()
    g["Rₕ! 1D"] = @benchmarkable Rₕ!($uₕ, sin)
    g["avgₕ! 1D"] = @benchmarkable avgₕ!($uₕ, sin)
    g["Rₕ 1D (allocates its output)"] = @benchmarkable Rₕ($Wₕ, sin)
end

# --- 2. the stencil engine, both directions ------------------------------- #
let Wₕ = gridspace(_mesh2()), uₕ = Rₕ(gridspace(_mesh2()), x -> sin(x[1]) * x[2])
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

# --- allocation bounds, which are the part worth gating ------------------- #
#
# The number an operator is allowed to allocate is one output vector: the
# `similar` inside it, and nothing else. These are the exact counts as of
# writing, not upper bounds with slack, because the failure mode being guarded
# is a closure starting to box, which shows up as allocation *per grid point*
# and so blows through any sensible slack immediately.
#
# The counts are the measured ones, so a Julia upgrade that changes how many objects a
# `similar` costs will trip this. That is intended: it forces someone to look, which is
# the only way an exact bound stays exact.
const ALLOCATION_BOUNDS = Dict(
    # one output vector each
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
    ("construction", "hₘₐₓ 3D") => 0)

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
    println("tuning...")
    tune!(SUITE)
    results = run(SUITE; verbose = true)
    append!(results.tags, ["julia:$(VERSION)", "os:$(Sys.KERNEL)", "arch:$(Sys.ARCH)"])

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
        println("\nagainst ", args[j + 1], " (baseline: Julia $base_jl, current: Julia $VERSION)")
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
