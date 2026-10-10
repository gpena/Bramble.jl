#===========================================================================#
# First-call cost of each CpuPolyester path (gpena/Bramble.jl#434).
#
# Plain Julia, Base and stdlib only, launched with no `--project` of its own, as
# benchmark/ttfx.jl is: it orchestrates child `julia` processes against the `test`
# environment, which has Polyester and SnoopCompileCore (the benchmark environment has no
# SnoopCompile).
#
#     julia --startup-file=no benchmark/polyester_first_call.jl [--runs N] [--threads N]
#                                                               [--project <dir>]
#
# `--project` names the environment the children run in (default `<tree>/test`). A tree
# with no Manifest cannot use its own `test`: build a shadow environment for it with
# .claude/plans/v3-25-0-checks/proj.sh and pass that. `--threads` is the children's thread
# count (default 2, the count test/quality/invalidations_polyester_reinfer.jl uses).
#
# Each path runs in its own fresh child (`--project=<dir> --threads=<N>`), so one path's
# compilation never pays for another's: the issue's single-process figure (1846 instances)
# cannot say which path costs. The child builds a CpuPolyester backend on the issue's 9x8
# non-uniform 2D grid (interior points jittered with a fixed seed), does
# `using Bramble, Polyester`, sets the path up, then wraps the
# path's first call in `@snoop_inference`. It reports the call's wall time (inference
# included, so the figure is inflated by the snooping, as the instance count needs) and
# the number of inferred instances with `CpuPolyester` in their signature, by the rule of
# test/quality/invalidations_polyester_reinfer.jl. It also reports the snooped window's
# total exclusive inference time and its number of inferred instances. The parent reports
# the median of `--runs` (default 3) processes per path (bramble-verification §2).
#
# Paths: `avg`, `Rh`, `innerh`, `broadcast`, `assemble`, `kronecker`, `rhs` and
# `matrix_free` call the operator cold. The `_newf` paths first call the operator warm with
# another user function, then snoop the call with `g`. They isolate what every new user
# function pays, which no workload can precompile: `avg` and `Rh` also pay the operator's
# own cold compile when it is not cached. `avg_newf` and `Rh_newf` are scalar-valued
# functions, `Rhk_newf` a function that captures a `Float64` (a closure payload one word
# longer than the warm-up's) and `Rhc_newf` a composite one, on `W × W`.
#
# Each path's setup is one `setup_path` method, called through `Base.invokelatest`. A single
# function with a branch per path would infer every branch when `child_run` is compiled, and
# the branches that build their inputs with `Rₕ(W, g)` (`innerh`, `broadcast`, `rhs`) would
# compile `Rₕ!(u, g)` before the snoop starts: the window of `Rh_newf` then held one
# instance. With one method per path, the only path inferred is the one the child runs.
#
# Output lines:
#   BRAMBLE path=<directory of the src/Bramble.jl the children loaded>
#   FIRSTCALL path=<name> ms=<median> inferred_cpupolyester=<median> tinf_ms=<median> nmi=<median>
#             (`tinf_ms` is the exclusive inference time of the window and `nmi` the
#             number of instances inferred in it, of any signature)
#   LOAD ms=<median time of `using Bramble` then `using Polyester`, together>
#   LOADEXT ms=<median time of the `using Polyester` that loads the extension>
#   CACHE bytes=<size of the BramblePolyesterExt image the children loaded>
#
# Run it alone on a quiet machine.
#===========================================================================#

const PATHS = ["avg", "avg_newf", "Rh", "Rh_newf", "Rhk_newf", "Rhc_newf", "innerh",
    "broadcast", "assemble", "kronecker", "rhs", "matrix_free"]
const ROOT = dirname(@__DIR__)

# --- child ------------------------------------------------------------------ #

function child(path::AbstractString)
    t_bramble = @elapsed @eval using Bramble
    t_polyester = @elapsed @eval using Polyester
    @eval using LinearAlgebra, Random, Libdl
    @eval using SnoopCompileCore
    r = Base.invokelatest(child_run, path)
    println("CHILD bramble=", Base.invokelatest(() -> pathof(Main.Bramble)))
    println("CHILD cache_bytes=", Base.invokelatest(loaded_cache_bytes))
    println("CHILD load_ms=", 1000(t_bramble + t_polyester), " loadext_ms=", 1000t_polyester,
        " ms=", r.ms, " n=", r.n, " tinf_ms=", r.tinf_ms, " nmi=", r.nmi)
    return nothing
end

# The size of the extension image this process loaded: the `.ji` it came from, with the
# library suffix in place of `.ji`. No depot scan, which reports the newest image any tree
# built.
function loaded_cache_bytes()
    ext = Base.get_extension(Main.Bramble, :BramblePolyesterExt)
    ext === nothing && error("BramblePolyesterExt is not loaded")
    ji = Base.pkgorigins[Base.PkgId(ext)].cachepath
    ji === nothing && error("BramblePolyesterExt was not loaded from a cache file")
    lib = string(first(splitext(ji)), ".", Main.Libdl.dlext)
    return isfile(lib) ? filesize(lib) : 0
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
    # The user function of the `_newf` paths' warm-up call, a different type from `g`.
    h(x) = cos(2x[1] - x[2])
    poisson(W) = form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
    # One method per path behind `invokelatest`, so compiling `child_run` infers no other
    # path's calls (see the header).
    call = Base.invokelatest(setup_path, Val(Symbol(path)), W, g, h, poisson)
    # `@snoop_inference` is a macro of a package loaded at run time, so it is expanded by
    # `eval` rather than written in this function, which is lowered before the load.
    tinf, ms = Core.eval(Main, :(let t0 = time_ns()
        t = SnoopCompileCore.@snoop_inference $call()
        (t, (time_ns() - t0) / 1.0e6)
    end))
    @eval using SnoopCompile
    return Base.invokelatest(count_cpupolyester, tinf, ms)
end

setup_path(::Val{:avg}, W, g, h, poisson) = (u = element(W); () -> avgₕ!(u, g))
function setup_path(::Val{:avg_newf}, W, g, h, poisson)
    u = element(W)
    avgₕ!(u, h)
    return () -> avgₕ!(u, g)
end
setup_path(::Val{:Rh}, W, g, h, poisson) = (u = element(W); () -> Rₕ!(u, g))
function setup_path(::Val{:Rh_newf}, W, g, h, poisson)
    u = element(W)
    Rₕ!(u, h)
    return () -> Rₕ!(u, g)
end
# A user function capturing a Float64: a payload one word longer than the warm-up's.
function setup_path(::Val{:Rhk_newf}, W, g, h, poisson)
    u = element(W)
    Rₕ!(u, h)
    gk = let k = 3.0
        x -> sin(k * x[1]) + x[2]
    end
    return () -> Rₕ!(u, gk)
end
function setup_path(::Val{:Rhc_newf}, W, g, h, poisson)
    u = element(W × W)
    Rₕ!(u, x -> (h(x), x[2]))
    gc = x -> (g(x), x[1] * x[2])
    return () -> Rₕ!(u, gc)
end
function setup_path(::Val{:innerh}, W, g, h, poisson)
    u, w = Rₕ(W, g), Rₕ(W, x -> x[1])
    return () -> innerₕ(u, w)
end
function setup_path(::Val{:broadcast}, W, g, h, poisson)
    u, w = Rₕ(W, g), Rₕ(W, x -> x[1])
    return () -> (w .= 2.0 .* u .+ w)
end
function setup_path(::Val{:assemble}, W, g, h, poisson)
    a = poisson(W)
    return () -> begin
        A = Bramble.allocate_system_matrix(a)
        assemble!(A, a)
        assemble!(A, a)
    end
end
function setup_path(::Val{:kronecker}, W, g, h, poisson)
    a, x = poisson(W), ones(ndofs(W))
    y = similar(x)
    return () -> LinearAlgebra.mul!(y, kronecker_operator(a), x)
end
function setup_path(::Val{:rhs}, W, g, h, poisson)
    f, a = Rₕ(W, g), poisson(W)
    l = form(W, v -> innerₕ(f, v))
    u = ones(ndofs(W))
    du = similar(u)
    return () -> Bramble.semidiscretize_rhs(semidiscretize(a, l))(du, u, nothing, 0.0)
end
function setup_path(::Val{:matrix_free}, W, g, h, poisson)
    a, x = poisson(W), ones(ndofs(W))
    y = similar(x)
    return () -> LinearAlgebra.mul!(y, matrix_free_operator(a), x)
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
    fl = SC.flatten(tinf)
    tinf_ms = 1000 * sum(SC.exclusive(t) for t in fl[2:end]; init = 0.0)
    return (; ms, n = count(hit, mis), tinf_ms, nmi = length(fl) - 1)
end

# --- orchestrator ----------------------------------------------------------- #

median(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

function run_child(path, project, threads)
    cmd = `$(Base.julia_cmd()) --project=$project --threads=$threads --startup-file=no $(@__FILE__) --child $path`
    out = read(pipeline(cmd; stderr = stderr), String)
    m = match(r"^CHILD load_ms=([\d.eE+-]+) loadext_ms=([\d.eE+-]+) ms=([\d.eE+-]+) n=(\d+) tinf_ms=([\d.eE+-]+) nmi=(\d+)"m, out)
    c = match(r"^CHILD cache_bytes=(\d+)"m, out)
    b = match(r"^CHILD bramble=(.+)$"m, out)
    (m === nothing || c === nothing || b === nothing) && error("child $path printed no result:\n$out")
    return (load = parse(Float64, m[1]), loadext = parse(Float64, m[2]), ms = parse(Float64, m[3]),
        n = parse(Int, m[4]), tinf = parse(Float64, m[5]), nmi = parse(Int, m[6]),
        cache = parse(Int, c[1]), bramble = String(b[1]))
end

function main(args)
    option(name, default) = (i = findfirst(==(name), args); i === nothing ? default : args[i + 1])
    runs = parse(Int, option("--runs", "3"))
    threads = parse(Int, option("--threads", "2"))
    project = abspath(option("--project", joinpath(ROOT, "test")))
    # Warm-up child: builds any missing pkgimage so no timed child pays for precompilation.
    warm = run_child("avg", project, threads)
    println("BRAMBLE path=", dirname(warm.bramble))
    loads = Float64[]
    loadexts = Float64[]
    caches = Int[]
    for p in PATHS
        rs = [run_child(p, project, threads) for _ in 1:runs]
        for r in rs
            r.bramble == warm.bramble ||
                error("child of path $p loaded $(r.bramble), the warm-up child $(warm.bramble)")
        end
        append!(loads, (r.load for r in rs))
        append!(loadexts, (r.loadext for r in rs))
        append!(caches, (r.cache for r in rs))
        println("FIRSTCALL path=", p, " ms=", round(median([r.ms for r in rs]); digits = 1),
            " inferred_cpupolyester=", round(Int, median([r.n for r in rs])),
            " tinf_ms=", round(median([r.tinf for r in rs]); digits = 1),
            " nmi=", round(Int, median([r.nmi for r in rs])))
        flush(stdout)
    end
    println("LOAD ms=", round(median(loads); digits = 1))
    println("LOADEXT ms=", round(median(loadexts); digits = 1))
    println("CACHE bytes=", round(Int, median(caches)))
end

if "--child" in ARGS
    child(ARGS[findfirst(==("--child"), ARGS) + 1])
else
    main(ARGS)
end
