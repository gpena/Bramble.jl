#===========================================================================#
# One side of a base-vs-head A/B of every `_pa` path under the three CPU
# policies (gpena/Bramble.jl#437). Run from a worktree root, with that
# worktree's test environment, 4 threads and -O1:
#
#     julia --project=<test env> --threads=4 --optimize=1 --startup-file=no \
#         <head>/benchmark/polyester_forms_ab.jl <policies> <case>...
#     julia ... benchmark/polyester_forms_ab.jl --list
#
# `<policies>` is a comma list of `serial`, `threaded`, `polyester`. For each
# case, policy and n in (65, 1025) the script prints
# `AB<TAB>case<TAB>policy<TAB>n<TAB>ns`, the minimum over repeated warm calls
# (at least 5, up to ~0.3 s or 20000 calls). An unknown case prints
# `AB-MISSING<TAB>case` and exits 1; `--list` prints every case name.
#
# The cases are the paths between `# BEGIN _pa paths` and `# END _pa paths`
# in the test file next to this script, not the caller's: the base side runs
# the head's script and block against the base's Bramble, so every path the
# head adds is timed on both sides. Two more cases, "restricted bilinear" and
# "restricted linear", assemble a form carrying a `markers = (:dir,)` term on
# the block's jittered non-uniform mesh, warm.
#===========================================================================#
using Bramble, Polyester
using Bramble: CpuPolyester, CpuSerial, CpuThreaded

const SRC = read(joinpath(@__DIR__, "..", "test", "ext", "polyester_ext.jl"), String)
let a = findfirst("# BEGIN _pa paths\n", SRC), b = findfirst("# END _pa paths", SRC)
    (a === nothing || b === nothing) && error("no _pa block in test/ext/polyester_ext.jl")
    include_string(Main, SRC[(last(a) + 1):(first(b) - 1)], "polyester_ext.jl _pa block")
end

function restricted_bilinear(n, p)
    W = _pa_space(n, p)
    a = form(W, W, (u, v) -> innerₕ(u, v; markers = (:dir,)) + inner₊(∇ₕ(u), ∇ₕ(v)))
    A = allocate_system_matrix(a)
    return _pa_case(() -> assemble!(A, a), () -> copy(A))
end

function restricted_linear(n, p)
    W = _pa_space(n, p)
    f = Rₕ(W, _pa_g)
    l = form(W, v -> innerₕ(f, v) + innerₕ(f, v; markers = (:dir,)))
    b = zeros(ndofs(W))
    return _pa_case(() -> assemble!(b, l), () -> copy(b))
end

const CASES = vcat([p[1] => p[4] for p in _pa_paths()],
    ["restricted bilinear" => restricted_bilinear, "restricted linear" => restricted_linear])
const SETUP = Dict(CASES)

if ARGS == ["--list"]
    foreach(c -> println(c[1]), CASES)
    exit(0)
end

Threads.nthreads() >= 4 || error("run with --threads=4")
const POLICIES = Dict("serial" => CpuSerial(), "threaded" => CpuThreaded(),
    "polyester" => CpuPolyester())
const POL = split(ARGS[1], ',')
const NAMES = ARGS[2:end]
all(in(keys(POLICIES)), POL) || error("unknown policy in $(ARGS[1])")

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

missing_any = false
for name in NAMES
    haskey(SETUP, name) || (println("AB-MISSING\t", name); global missing_any = true)
end
missing_any && exit(1)
for name in NAMES, pol in POL, n in (65, 1025)
    t = best(SETUP[name](n, POLICIES[pol]).call)
    println("AB\t", name, "\t", pol, "\t", n, "\t", t)
end
