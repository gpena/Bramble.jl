#===========================================================================#
# First-call cost of each CpuPolyester path (gpena/Bramble.jl#434).
#
# Plain Julia, Base and stdlib only, launched with no `--project` of its own, as
# benchmark/ttfx.jl is: it orchestrates child `julia` processes against the `test`
# environment, which has Polyester and SnoopCompileCore (the benchmark environment has no
# SnoopCompile).
#
#     julia --startup-file=no benchmark/polyester_first_call.jl [--runs N]
#
# Each path runs in its own fresh child (`--project=test --threads=2`), so one path's
# compilation never pays for another's: the issue's single-process figure (1846 instances)
# cannot say which path costs. The child builds a CpuPolyester backend on the issue's 9x8
# non-uniform 2D grid (interior points jittered with a fixed seed), does
# `using Bramble, Polyester`, sets the path up, then wraps the
# path's first call in `@snoop_inference`. It reports the call's wall time (inference
# included, so the figure is inflated by the snooping, as the instance count needs) and
# the number of inferred instances with `CpuPolyester` in their signature, by the rule of
# test/quality/invalidations_polyester_reinfer.jl. The parent reports the median of `--runs`
# (default 3) processes per path (bramble-verification §2).
#
# Output lines:
#   FIRSTCALL path=<name> ms=<median> inferred_cpupolyester=<median>
#   LOAD ms=<median time of `using Bramble, Polyester`>
#   CACHE bytes=<size of the newest BramblePolyesterExt pkgimage>
#
# Run it alone on a quiet machine; the child count of threads is fixed at 2.
#===========================================================================#

const PATHS = ["avg", "innerh", "broadcast", "assemble", "kronecker", "rhs", "matrix_free"]
const ROOT = dirname(@__DIR__)

# --- child ------------------------------------------------------------------ #

function child(path::AbstractString)
    t_load = @elapsed @eval using Bramble, Polyester
    @eval using LinearAlgebra, Random
    @eval using SnoopCompileCore
    r = Base.invokelatest(child_run, path)
    println("CHILD load_ms=", 1000t_load, " ms=", r.ms, " n=", r.n)
    return nothing
end

function child_run(path)
    CpuPolyester = Bramble.CpuPolyester
    mul! = LinearAlgebra.mul!
    X = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ω = mesh(domain(X), (9, 8), (false, false); backend = backend(policy = CpuPolyester()))
    # Jitter the interior points (fixed seed, endpoints kept, sorted) so the grid is
    # genuinely non-uniform, as the issue's is.
    rng = Main.Random.Xoshiro(4321)
    function pts(n)
        x = collect(range(0.0, 1.0; length = n)) .+ 0.3 / (n - 1) .* (2 .* rand(rng, n) .- 1)
        x[1], x[end] = 0.0, 1.0
        return sort!(x)
    end
    Bramble.change_points!(Ω, (pts(9), pts(8)))
    W = gridspace(Ω)
    g(x) = sin(3x[1] + 2x[2]) + x[1] * x[2]
    poisson(W) = form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
    # Each path's setup is a `let` block, so its variables are fresh locals of the closure it
    # returns and are captured with their concrete types, not in a `Core.Box` (a name
    # assigned in several branches of one function is boxed, which widens every call).
    call = if path == "avg"
        let u = element(W)
            () -> avgₕ!(u, g)
        end
    elseif path == "innerh"
        let u = Rₕ(W, g), w = Rₕ(W, x -> x[1])
            () -> innerₕ(u, w)
        end
    elseif path == "broadcast"
        let u = Rₕ(W, g), w = Rₕ(W, x -> x[1])
            () -> (w .= 2.0 .* u .+ w)
        end
    elseif path == "assemble"
        let a = poisson(W)
            () -> begin
                A = Bramble.allocate_system_matrix(a)
                assemble!(A, a)
                assemble!(A, a)
            end
        end
    elseif path == "kronecker"
        let a = poisson(W), x = ones(ndofs(W)), y = similar(x)
            () -> mul!(y, kronecker_operator(a), x)
        end
    elseif path == "rhs"
        let f = Rₕ(W, g), a = poisson(W), l = form(W, v -> innerₕ(f, v)),
            u = ones(ndofs(W)), du = similar(u)

            () -> Bramble.semidiscretize_rhs(semidiscretize(a, l))(du, u, nothing, 0.0)
        end
    elseif path == "matrix_free"
        let a = poisson(W), x = ones(ndofs(W)), y = similar(x)
            () -> mul!(y, matrix_free_operator(a), x)
        end
    else
        error("unknown path $path")
    end
    # `@snoop_inference` is a macro of a package loaded at run time, so it is expanded by
    # `eval` rather than written in this function, which is lowered before the load.
    tinf, ms = Core.eval(Main, :(let t0 = time_ns()
        t = SnoopCompileCore.@snoop_inference $call()
        (t, (time_ns() - t0) / 1.0e6)
    end))
    @eval using SnoopCompile
    return Base.invokelatest(count_cpupolyester, tinf, ms)
end

# A type defined in Main (the user's closures) is new to every session, so its inference
# is never a cache miss; only instances built entirely from package types count.
_from_main(T) = false
_from_main(T::UnionAll) = _from_main(Base.unwrap_unionall(T))
_from_main(T::Union) = _from_main(T.a) || _from_main(T.b)
_from_main(T::DataType) = parentmodule(T) === Main || any(_from_main, T.parameters)

function count_cpupolyester(tinf, ms)
    SC = Main.SnoopCompile
    mi_of(t) = (d = t.ci.def; d isa Core.MethodInstance ? d : d.def)
    mis = [m for m in map(mi_of, SC.flatten(tinf)) if m isa Core.MethodInstance && m.def isa Method]
    hit(mi) = !_from_main(mi.specTypes) &&
              (occursin("CpuPolyester", string(mi.specTypes)) ||
               startswith(string(parentmodule(mi.def)), "BramblePolyesterExt"))
    return (; ms, n = count(hit, mis))
end

# --- orchestrator ----------------------------------------------------------- #

median(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

function run_child(path)
    cmd = `$(Base.julia_cmd()) --project=$(joinpath(ROOT, "test")) --threads=2 --startup-file=no $(@__FILE__) --child $path`
    out = read(pipeline(cmd; stderr = stderr), String)
    m = match(r"^CHILD load_ms=([\d.eE+-]+) ms=([\d.eE+-]+) n=(\d+)"m, out)
    m === nothing && error("child $path printed no result:\n$out")
    return (load = parse(Float64, m[1]), ms = parse(Float64, m[2]), n = parse(Int, m[3]))
end

# The size of the newest BramblePolyesterExt pkgimage in the first depot that has one.
function cache_bytes()
    best = nothing
    for depot in DEPOT_PATH
        dir = joinpath(depot, "compiled", "v$(VERSION.major).$(VERSION.minor)", "BramblePolyesterExt")
        isdir(dir) || continue
        for f in readdir(dir; join = true)
            (endswith(f, ".dylib") || endswith(f, ".so") || endswith(f, ".dll")) || continue
            (best === nothing || mtime(f) > mtime(best)) && (best = f)
        end
        best === nothing || break
    end
    return best === nothing ? 0 : filesize(best)
end

function main(args)
    i = findfirst(==("--runs"), args)
    runs = i === nothing ? 3 : parse(Int, args[i + 1])
    # Warm-up child: builds any missing pkgimage so no timed child pays for precompilation.
    run_child("avg")
    loads = Float64[]
    for p in PATHS
        rs = [run_child(p) for _ in 1:runs]
        append!(loads, (r.load for r in rs))
        println("FIRSTCALL path=", p, " ms=", round(median([r.ms for r in rs]); digits = 1),
            " inferred_cpupolyester=", round(Int, median([r.n for r in rs])))
        flush(stdout)
    end
    println("LOAD ms=", round(median(loads); digits = 1))
    println("CACHE bytes=", cache_bytes())
end

if "--child" in ARGS
    child(ARGS[findfirst(==("--child"), ARGS) + 1])
else
    main(ARGS)
end
