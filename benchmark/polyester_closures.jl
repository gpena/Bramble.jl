#===========================================================================#
# Before-and-after timing of a user closure over an array under CpuPolyester, and of
# several callers at once (gpena/Bramble.jl#476).
#
# One side of one ab.sh round, with ab.jl's CLI: run from a worktree root with --threads=4
# --optimize=1 and that worktree's test environment (ab.sh passes this file through AB_JL).
#
#     julia ... benchmark/polyester_closures.jl <policies> <case>...
#     julia ... benchmark/polyester_closures.jl --list
#     julia ... benchmark/polyester_closures.jl --first-call
#
# `<policies>` is a comma list of serial, threaded, polyester. For each case, policy and
# size it prints `AB<TAB>case<TAB>policy<TAB>n<TAB>ns`. The six closure cases call `Rₕ!` or
# `avgₕ!` (plain, masked, composite) with a closure over a `Vector` on a jittered
# non-uniform 2D mesh (n = 65 and 1025 points per axis): the paths whose Polyester argument
# box this plan removes. `contention` has 8 `Threads.@spawn` tasks each call `avgₕ!` 10
# times with a closure of one shared type (n = 65 and 257), so it prices the per-call slot
# claim when callers meet; its figure is the wall time of all 80 calls.
#
# Each case is timed at AB_LAYOUTS (default 5) array placements: every one is built after a
# pad of 16 KiB pages from a fixed seed, so the arrays land at several address offsets
# (one fixed placement made D₋ₓ! look 2x slower on either tree, #475), and the figure is
# the median of the per-placement minima. The seed is the same on every side, so a base and
# a head run draw the same pads. Per placement: the minimum over repeated calls (at least 5,
# up to ~0.3 s or 20000 calls), after two warm-up calls.
#
# `--list` prints one case name per line. `--first-call` runs each closure case's first call
# in a fresh child process (Polyester, 65 points per axis; FC_RUNS children, default 3) and
# prints `FIRSTCALL<TAB>case<TAB>ms`, the median: the slot table adds a registration on
# a closure type's first use, and the figure includes compilation.
#===========================================================================#
const CLOSURE_CASES = ["Rₕ! closure", "Rₕ! masked closure", "Rₕ! composite closure",
    "avgₕ! closure", "avgₕ! masked closure", "avgₕ! composite closure"]
const CASE_NAMES = [CLOSURE_CASES; "contention"]

if !isempty(ARGS) && ARGS[1] == "--list"
    foreach(println, CASE_NAMES)
    exit(0)
end

# `--first-call`: the parent starts one child per run and case; the child is the same file
# with `--first-call-child <case>`.
if !isempty(ARGS) && ARGS[1] == "--first-call"
    project = dirname(Base.active_project())
    runs = parse(Int, get(ENV, "FC_RUNS", "3"))
    for name in CLOSURE_CASES
        ms = Float64[]
        for _ in 1:runs
            cmd = `$(Base.julia_exename()) --project=$project --threads=4 --optimize=1
                --startup-file=no $(@__FILE__) --first-call-child $name`
            out = readchomp(cmd)
            m = match(r"^CHILD\t(\S+)$"m, out)
            m === nothing && error("no CHILD line from the first-call child of \"$name\":\n$out")
            push!(ms, parse(Float64, m[1]))
        end
        println("FIRSTCALL\t", name, "\t", sort(ms)[(length(ms) + 1) ÷ 2])
    end
    exit(0)
end

using Bramble, Polyester
using Bramble: CpuPolyester, CpuSerial, CpuThreaded, change_points!
using Random: Xoshiro
Threads.nthreads() >= 4 || error("run with --threads=4")

const POLICIES = Dict("serial" => CpuSerial(), "threaded" => CpuThreaded(),
    "polyester" => CpuPolyester())

# The jittered non-uniform mesh of test/ext/polyester_ext.jl's `_pa_jitter` (fixed seed).
function jittered_space(n, policy; seed = 4321)
    rng = Xoshiro(seed)
    X = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ω = mesh(domain(X, :dir => boundary_symbols(X)), (n, n), (true, true);
        backend = backend(policy = policy))
    h = 1 / (n - 1)
    function pts()
        x = collect(range(0.0, 1.0; length = n)) .+ 0.3h .* (2 .* rand(rng, n) .- 1)
        x[1], x[end] = 0.0, 1.0
        return sort!(x)
    end
    change_points!(Ω, (pts(), pts()))
    return gridspace(Ω)
end

# A closure over a `Vector`. Every call returns the same closure type, so a kernel of this
# type claims a slot in the same table whichever case or task builds it.
user_function(c) = x -> c[1] * sin(3x[1] + 2x[2]) + c[2] * x[1] * x[2]
# Composite: the closure returns a tuple, one component per space.
user_pair(f) = x -> (f(x), x[2])

# Each case builds its destination and closure in its own function, so every captured name
# is assigned once and is not boxed (a name assigned in two branches would be).
function case_R(n, policy)
    u = element(jittered_space(n, policy))
    f = user_function([0.3, 0.7])
    return (; call = () -> Rₕ!(u, f))
end
function case_R_masked(n, policy)
    u = element(jittered_space(n, policy))
    f = user_function([0.3, 0.7])
    return (; call = () -> Rₕ!(u, f; markers = (:dir,)))
end
function case_R_composite(n, policy)
    W = jittered_space(n, policy)
    u = element(W × W)
    f = user_pair(user_function([0.3, 0.7]))
    return (; call = () -> Rₕ!(u, f))
end
function case_avg(n, policy)
    u = element(jittered_space(n, policy))
    f = user_function([0.3, 0.7])
    return (; call = () -> avgₕ!(u, f))
end
function case_avg_masked(n, policy)
    u = element(jittered_space(n, policy))
    f = user_function([0.3, 0.7])
    return (; call = () -> avgₕ!(u, f; markers = (:dir,)))
end
function case_avg_composite(n, policy)
    W = jittered_space(n, policy)
    u = element(W × W)
    f = user_pair(user_function([0.3, 0.7]))
    return (; call = () -> avgₕ!(u, f))
end

const CONTENDERS = 8
const CONTENTION_CALLS = 10
# One task's share: a function barrier, so the spawned closure is a plain call.
function contend(u, f, k)
    for _ in 1:k
        avgₕ!(u, f)
    end
    return nothing
end
# 8 tasks, each with its own destination and its own closure (own `Vector`) of one type.
function case_contention(n, policy)
    W = jittered_space(n, policy)
    us = [element(W) for _ in 1:CONTENDERS]
    fs = [user_function([0.3 + i / 100, 0.7]) for i in 1:CONTENDERS]
    return (; call = () -> begin
        tasks = [Threads.@spawn(contend(us[i], fs[i], CONTENTION_CALLS)) for i in 1:CONTENDERS]
        foreach(wait, tasks)
    end)
end

const SETUP = Dict("Rₕ! closure" => case_R, "Rₕ! masked closure" => case_R_masked,
    "Rₕ! composite closure" => case_R_composite, "avgₕ! closure" => case_avg,
    "avgₕ! masked closure" => case_avg_masked,
    "avgₕ! composite closure" => case_avg_composite, "contention" => case_contention)
const SIZES = Dict(name => (65, 1025) for name in CLOSURE_CASES)
SIZES["contention"] = (65, 257)

@noinline function once(call::F) where {F}
    t = time_ns()
    call()
    return time_ns() - t
end
@noinline function best(call::F) where {F}
    call()
    call()
    t1 = once(call)
    reps = clamp(ceil(Int, 3e8 / max(t1, 1)), 5, 20_000)
    return minimum(once(call) for _ in 1:reps)
end

# The first call of one case in this fresh process, after its setup (which makes no call
# to the closure's kernels).
if !isempty(ARGS) && ARGS[1] == "--first-call-child"
    call = SETUP[ARGS[2]](65, CpuPolyester()).call
    t = once(call)
    println("CHILD\t", t / 1e6)
    exit(0)
end

const POL = split(ARGS[1], ',')
const NAMES = ARGS[2:end]
all(in(keys(POLICIES)), POL) || error("unknown policy in $(ARGS[1])")
const RNG = Xoshiro(20261007)
const PADS = Vector{UInt8}[]
missing_any = false
for name in NAMES
    haskey(SETUP, name) || (println("AB-MISSING\t", name); global missing_any = true)
end
missing_any && exit(1)
for name in NAMES, pol in POL, n in SIZES[name]
    ts = UInt64[]
    for _ in 1:parse(Int, get(ENV, "AB_LAYOUTS", "5"))
        push!(PADS, Vector{UInt8}(undef, 16384 * rand(RNG, 1:512)))
        push!(ts, best(SETUP[name](n, POLICIES[pol]).call))
    end
    t = sort(ts)[(length(ts) + 1) ÷ 2]
    println("AB\t", name, "\t", pol, "\t", n, "\t", t)
end
