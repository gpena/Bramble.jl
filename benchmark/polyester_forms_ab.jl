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
#
# Two report modes read files and measure nothing (Base only, no Bramble):
#
#     julia benchmark/polyester_forms_ab.jl --first-call <base.txt> <head.txt>
#     julia benchmark/polyester_forms_ab.jl --vs-serial <dir>
#
# `--first-call` takes the stdout of benchmark/polyester_first_call.jl on each
# tree. For every path in both files it prints
# `FIRSTCALL-AB<TAB>path<TAB>base_ms<TAB>head_ms<TAB>ms_ratio<TAB>base_n<TAB>head_n`
# (ms_ratio = head_ms / base_ms; n = inferred_cpupolyester) and
# `FIRSTCALL-ALARM<TAB>path` when ms_ratio > 2 or head_n > 2 * max(base_n, 1)
# (the #437 "compiles twice" alarm). It exits 0 either way.
#
# `--vs-serial` takes a directory holding the per-round raw outputs of the
# sides of an A/B, `base-<round>` and `head-<round>`, each holding the `AB`
# rows of a run with every policy (ab.sh writes them in its temporary
# directory). Per case and size it prints, from the medians over the rounds,
# `VS-SERIAL<TAB>case<TAB>n<TAB>policy<TAB>base_pol/serial<TAB>head_pol/serial<TAB>flag`
# for the polyester and the threaded policy. A row is flagged `SHRANK` when
# the head's speedup over serial is more than 5% below the base's. No bar.
#===========================================================================#
if !isempty(ARGS) && ARGS[1] == "--first-call"
    length(ARGS) == 3 || error("usage: --first-call <base.txt> <head.txt>")
    function firstcall(file)
        rows = Dict{String,Tuple{Float64,Int}}()
        for m in eachmatch(r"^FIRSTCALL path=(\S+) ms=(\S+) inferred_cpupolyester=(\d+)$"m,
            read(file, String))
            rows[m[1]] = (parse(Float64, m[2]), parse(Int, m[3]))
        end
        return rows
    end
    base, head = firstcall(ARGS[2]), firstcall(ARGS[3])
    for path in sort!(collect(intersect(keys(base), keys(head))))
        (bms, bn), (hms, hn) = base[path], head[path]
        ratio = hms / bms
        println("FIRSTCALL-AB\t", path, "\t", bms, "\t", hms, "\t", round(ratio; digits = 3),
            "\t", bn, "\t", hn)
        (ratio > 2 || hn > 2 * max(bn, 1)) && println("FIRSTCALL-ALARM\t", path)
    end
    exit(0)
end

if !isempty(ARGS) && ARGS[1] == "--vs-serial"
    length(ARGS) == 2 || error("usage: --vs-serial <dir>")
    function median(v)
        w = sort(v)
        n = length(w)
        return isodd(n) ? w[(n + 1) ÷ 2] : (w[n ÷ 2] + w[n ÷ 2 + 1]) / 2
    end
    # (side, case, policy, n) => round => ns
    times = Dict{Tuple{String,String,String,Int},Dict{Int,Int}}()
    for f in readdir(ARGS[2])
        m = match(r"^(base|head)-(\d+)$", f)
        m === nothing && continue
        for line in eachline(joinpath(ARGS[2], f))
            startswith(line, "AB\t") || continue
            _, name, pol, n, ns = split(line, '\t')
            key = (String(m[1]), String(name), String(pol), parse(Int, n))
            get!(Dict{Int,Int}, times, key)[parse(Int, m[2])] = parse(Int, ns)
        end
    end
    function ratio(side, name, pol, n)
        a = get(times, (side, name, pol, n), nothing)
        s = get(times, (side, name, "serial", n), nothing)
        (a === nothing || s === nothing) && return nothing
        rounds = sort!(collect(intersect(keys(a), keys(s))))
        return isempty(rounds) ? nothing : median([a[r] / s[r] for r in rounds])
    end
    for name in sort!(unique(k[2] for k in keys(times))), n in (65, 1025),
        pol in ("polyester", "threaded")

        rb, rh = ratio("base", name, pol, n), ratio("head", name, pol, n)
        (rb === nothing || rh === nothing) && continue
        # Speedup over serial is 1 / ratio: it shrank by more than 5% when 1/rh < 0.95/rb.
        flag = 1 / rh < 0.95 / rb ? "SHRANK" : ""
        println("VS-SERIAL\t", name, "\t", n, "\t", pol, "\t", round(rb; digits = 3), "\t",
            round(rh; digits = 3), "\t", flag)
    end
    exit(0)
end

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
