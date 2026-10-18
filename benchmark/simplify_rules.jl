#===========================================================================#
# Audit of `simplify_ast` rules 1-4 and 7 (gpena/Bramble.jl#771): one rule's fresh-process
# A/B pair. A is a tree at the measured commit (the rule on), B a detached copy of that
# commit with the rule switched off by `benchmark/simplify_rules/rule<N>.patch`.
#
#     julia --startup-file=no benchmark/simplify_rules.jl --rule N --base DIR --patched DIR
#         [--pairs 3] [--max-pairs 5] [--threads 4] [--max-load 6] [--forms a,b,...] [--smoke]
#     julia --startup-file=no benchmark/simplify_rules.jl --candidate NAME --base DIR
#         --patched DIR [...]
#
# `--patched` is applied the patch with `git apply` when its `src/` is still clean; a tree
# that already differs there is used as it is. The rules of the audit are 1 `c*A+c*B`,
# 2 like terms, 3 zeros, 4 nested scale and 7 shared inner-product argument (`_factor`),
# measured at fe3e8659 with A the rule on and B off. `--candidate NAME` measures a rule that
# #771 proposes instead, at 9a7e093a (the rule set that remains after the removals). There
# `--base` is the tree without the candidate and `--patched` the tree with
# `benchmark/simplify_rules/cand_NAME.patch`; the driver swaps them so that A is always the
# rule or candidate on. NAME is one of scalar_lift, same_wrapper, shared_wrapper,
# component_type, rational, interp_fold and cse. The patch of component_type and of cse
# says `not built` and the driver stops there. interp_fold rewrites at block time, so the
# stored AST is identical and the form is timed anyway (`--always` does the same for any
# candidate).
# Every child also reports whether `form` inferred a concrete type (`@inferred`) and whether
# `simplify_ast` is idempotent on the stored AST. For the first pair of a form it dumps the
# assembled matrix, so the parent compares the two bit for bit.
#
# Each child is a fresh `julia` process on one tree and one form: it times the first
# `form`, the first `assemble`, then `assemble!` warm (the min of 15), and reports the
# routed-term count (`BilinearProduct` nodes of the simplified AST), the node count and two
# checksums of the matrix. The parent first builds each form in both trees (`--probe`) and
# skips the forms whose simplified AST has the same type in both: A and B are the same
# program there. The others it times `--pairs` times, the two trees back to back and the
# order swapped on every other pair, and more (up to `--max-pairs`) while the pairs
# disagree on the sign of the difference or the difference is within twice the spread. Each
# sample's 1-minute load and the CPU of any foreign julia process is logged, and an
# unsettled pair is discarded and rerun. It prints per form and tree the median [min-max] of
# the first call (`form` + `assemble`) and of `assemble!`. The checksums of A and B must
# agree (a rule only changes how the form is routed).
#
# Verdict per form and time: the rule (tree A) wins when every A sample is faster than
# every B sample (the min-max ranges are disjoint), loses when every A sample is slower, and
# ties when the ranges overlap, i.e. the difference is within the measured spread. The rule
# is `keep` when it wins on at least one time and loses on neither, over the timed forms,
# else `remove`.
#
# Run the parent plain, not through `julia_slot.sh`: it would hold slots its children wait
# for. Every child runs through `.claude/scripts/julia_slot.sh <threads>` of the tree
# holding this file when `BRAMBLE_JULIA_SLOTS` is set. Time at default optimisation, on AC
# power, with `.claude/scripts/gate_preflight.sh --timing` passing; run nothing alongside.
#===========================================================================#

import Serialization
import SparseArrays
import Test

const FORMS = ["s2d_mix", "s3d_mix", "s3d_27", "v2d_elast", "v3d_elast", "v3d_lap"]

# The forms of the candidate rules (`--candidate`): each is written so that one candidate
# fires on it, and the others are probed too, since the no-op test settles which fire.
const CAND_FORMS = ["s2d_lift", "s3d_lift", "v2d_lift", "s2d_sum", "s3d_sum", "s2d_shared",
    "s3d_shared", "s2d_rat", "s3d_rat", "v2d_rat", "v3d_rat", "s2d_interp", "s3d_interp"]
const CANDIDATES = ["scalar_lift", "same_wrapper", "shared_wrapper", "component_type",
    "rational", "interp_fold", "cse"]
const CAND_DEFAULT = Dict(
    "scalar_lift" => ["s2d_lift", "s3d_lift", "v2d_lift", "s3d_27", "v3d_elast"],
    "same_wrapper" => ["s2d_sum", "s3d_sum", "s3d_27", "v3d_elast"],
    "shared_wrapper" => ["s2d_shared", "s3d_shared", "s3d_27", "v3d_elast"],
    "rational" => ["s2d_rat", "s3d_rat", "v2d_rat", "v3d_rat"],
    "interp_fold" => ["s2d_interp", "s3d_interp"],
    "component_type" => ["v2d_elast", "v3d_elast"],
    "cse" => ["v3d_elast", "s3d_27"])

# Graded points of an axis: the exponent differs per axis, so no two axes agree.
graded(n, axis) = collect(range(0.0, 1.0; length = n)) .^ (1.2 + 0.2 * axis)

const POINTS = Dict(
    "s2d_mix" => (200, 180), "s3d_mix" => (28, 25, 22), "s3d_27" => (22, 20, 18),
    "v2d_elast" => (90, 80), "v3d_elast" => (16, 14, 12), "v3d_lap" => (22, 20, 18),
    "s2d_lift" => (200, 180), "s3d_lift" => (28, 25, 22), "v2d_lift" => (90, 80),
    "s2d_sum" => (200, 180), "s3d_sum" => (28, 25, 22),
    "s2d_shared" => (200, 180), "s3d_shared" => (28, 25, 22),
    "s2d_rat" => (200, 180), "s3d_rat" => (28, 25, 22),
    "v2d_rat" => (90, 80), "v3d_rat" => (16, 14, 12),
    "s2d_interp" => (200, 180), "s3d_interp" => (28, 25, 22))

if "--child" in ARGS
    # --child <form> [--probe]
    name = ARGS[2]
    using Bramble
    import Bramble: D₋ₓ, D₋ᵧ, D₋₂, D₊ₓ, D₊ᵧ, D₊₂, Dcₓ, Dcᵧ, Dc₂, change_points!

    n = POINTS[name]
    d = length(n)
    Ωₕ = mesh(domain(box(ntuple(_ -> 0.0, d), ntuple(_ -> 1.0, d))), n)
    change_points!(Ωₕ, ntuple(k -> graded(n[k], k), d))
    vector = startswith(name, "v")
    W = vector ? gridspace(Ωₕ)^Val(d) : gridspace(Ωₕ)

    # Each scalar form exercises the rules in turn: `2*X + 2*Y` (1), `X + 3*X` (2),
    # `0*X` (3), `2*(3*X)` (4), and a trial operator shared by several products (7).
    fun = if name == "s2d_mix"
        (u, v) -> (2 * innerₕ(D₋ₓ(u), D₋ₓ(v)) + 2 * innerₕ(D₋ᵧ(u), D₋ᵧ(v))) +
                  (innerₕ(u, v) + 3 * innerₕ(u, v)) + 0 * innerₕ(Dcₓ(u), v) +
                  2 * (3 * innerₕ(Dcᵧ(u), v)) + innerₕ(D₊ₓ(u), D₊ₓ(v)) +
                  innerₕ(D₊ₓ(u), D₊ᵧ(v))
    elseif name == "s3d_mix"
        (u, v) -> (2 * innerₕ(D₋ₓ(u), D₋ₓ(v)) + 2 * innerₕ(D₋ᵧ(u), D₋ᵧ(v)) +
                   2 * innerₕ(D₋₂(u), D₋₂(v))) +
                  (innerₕ(u, v) + 3 * innerₕ(u, v)) + 0 * innerₕ(Dcₓ(u), v) +
                  2 * (3 * innerₕ(Dcᵧ(u), v)) + innerₕ(D₊ₓ(u), D₊ₓ(v)) +
                  innerₕ(D₊ₓ(u), D₊ᵧ(v)) + innerₕ(D₊ₓ(u), D₊₂(v))
    elseif name == "s3d_27"
        # 27 distinct products; each of 9 trial operators meets three test operators.
        function f27(u, v)
            tr = (D₋ₓ, D₋ᵧ, D₋₂, D₊ₓ, D₊ᵧ, D₊₂, Dcₓ, Dcᵧ, Dc₂)
            te = (D₋ₓ, D₋ᵧ, D₋₂)
            t(a, b) = innerₕ(tr[a](u), te[b](v))
            return (((t(1, 1) + t(1, 2) + t(1, 3)) + (t(2, 1) + t(2, 2) + t(2, 3))) +
                    ((t(3, 1) + t(3, 2) + t(3, 3)) + (t(4, 1) + t(4, 2) + t(4, 3)))) +
                   (((t(5, 1) + t(5, 2) + t(5, 3)) + (t(6, 1) + t(6, 2) + t(6, 3))) +
                    ((t(7, 1) + t(7, 2) + t(7, 3)) + (t(8, 1) + t(8, 2) + t(8, 3)) +
                     (t(9, 1) + t(9, 2) + t(9, 3))))
        end
        f27
    elseif name == "v3d_lap"
        (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))
    elseif name == "s2d_lift"  # a scalar under a difference or an average
        (u, v) -> innerₕ(D₋ₓ(2 * u), D₋ₓ(v)) + innerₕ(D₋ᵧ(3 * u), D₋ᵧ(v)) +
                  innerₕ(Mₕ(2 * u, Val(1)), v) + innerₕ(D₊ₓ(u), D₊ₓ(v)) +
                  innerₕ(Dcₓ(4 * u), Dcᵧ(v))
    elseif name == "s3d_lift"
        (u, v) -> innerₕ(D₋ₓ(2 * u), D₋ₓ(v)) + innerₕ(D₋ᵧ(3 * u), D₋ᵧ(v)) +
                  innerₕ(D₋₂(5 * u), D₋₂(v)) + innerₕ(Mₕ(2 * u, Val(1)), v) +
                  innerₕ(D₊ₓ(u), D₊ₓ(v)) + innerₕ(Dcₓ(4 * u), Dcᵧ(v))
    elseif name == "v2d_lift"
        (u, v) -> inner₊(εₕ(2 * u), εₕ(v)) + inner₊(divₘₕ(3 * u), divₘₕ(v))
    elseif name == "s2d_sum"  # one wrapper over a same-component sum
        (u, v) -> innerₕ(D₋ₓ(u) + D₋ₓ(D₊ᵧ(u)), D₋ₓ(v)) +
                  innerₕ(D₋ᵧ(u) + D₋ᵧ(D₊ₓ(u)), D₋ᵧ(v)) +
                  innerₕ(Mₕ(u, Val(1)) + Mₕ(D₊ₓ(u), Val(1)), v) + innerₕ(u, v)
    elseif name == "s3d_sum"
        (u, v) -> innerₕ(D₋ₓ(u) + D₋ₓ(D₊ᵧ(u)), D₋ₓ(v)) +
                  innerₕ(D₋ᵧ(u) + D₋ᵧ(D₊₂(u)), D₋ᵧ(v)) +
                  innerₕ(D₋₂(u) + D₋₂(D₊ₓ(u)), D₋₂(v)) +
                  innerₕ(Mₕ(u, Val(1)) + Mₕ(D₊ₓ(u), Val(1)), v) + innerₕ(u, v)
    elseif name == "s2d_shared"  # a shared argument, a shared wrapper on the other
        (u, v) -> innerₕ(D₋ₓ(u), D₊ₓ(v)) + innerₕ(D₋ₓ(u), D₊ₓ(D₊ᵧ(v))) +
                  innerₕ(D₋ᵧ(u), D₊ᵧ(v)) + innerₕ(D₋ᵧ(u), D₊ᵧ(D₊ₓ(v))) + innerₕ(u, v)
    elseif name == "s3d_shared"
        (u, v) -> innerₕ(D₋ₓ(u), D₊ₓ(v)) + innerₕ(D₋ₓ(u), D₊ₓ(D₊ᵧ(v))) +
                  innerₕ(D₋ᵧ(u), D₊ᵧ(v)) + innerₕ(D₋ᵧ(u), D₊ᵧ(D₊₂(v))) +
                  innerₕ(D₋₂(u), D₊₂(v)) + innerₕ(D₋₂(u), D₊₂(D₊ₓ(v))) + innerₕ(u, v)
    elseif name == "s2d_rat"  # Rational coefficients (`op / 2`) nested
        (u, v) -> (innerₕ(D₋ₓ(u), D₋ₓ(v)) / 2) * 3 + innerₕ(D₋ᵧ(u) / 2, D₋ᵧ(v) / 3) +
                  (innerₕ(u, v) / 4) * 2 + innerₕ(D₊ₓ(u), D₊ₓ(v)) / 3
    elseif name == "s3d_rat"
        (u, v) -> (innerₕ(D₋ₓ(u), D₋ₓ(v)) / 2) * 3 + innerₕ(D₋ᵧ(u) / 2, D₋ᵧ(v) / 3) +
                  (innerₕ(D₋₂(u), D₋₂(v)) / 4) * 2 + (innerₕ(u, v) / 4) * 2 +
                  innerₕ(D₊ₓ(u), D₊ₓ(v)) / 3
    elseif name == "v2d_rat"
        (u, v) -> (inner₊(εₕ(u), εₕ(v)) / 2) * 3 + (inner₊(divₘₕ(u), divₘₕ(v)) / 4) * 2
    elseif name == "v3d_rat"
        (u, v) -> (inner₊(εₕ(u), εₕ(v)) / 2) * 3 + (inner₊(divₘₕ(u), divₘₕ(v)) / 4) * 2
    elseif name == "s2d_interp"  # πₕ(u) on the mesh of u
        (u, v) -> innerₕ(πₕ(u), v) + innerₕ(πₕ(u), D₋ₓ(v)) + innerₕ(D₋ₓ(u), D₋ₓ(v)) +
                  innerₕ(D₋ᵧ(u), D₋ᵧ(v))
    elseif name == "s3d_interp"
        (u, v) -> innerₕ(πₕ(u), v) + innerₕ(πₕ(u), D₋ₓ(v)) + innerₕ(D₋ₓ(u), D₋ₓ(v)) +
                  innerₕ(D₋ᵧ(u), D₋ᵧ(v)) + innerₕ(D₋₂(u), D₋₂(v))
    else  # elasticity, Integer Lamé coefficients so that `2μ` and `λ` are equal
        (u, v) -> 2 * inner₊(εₕ(u), εₕ(v)) + 2 * inner₊(divₘₕ(u), divₘₕ(v))
    end

    t_form = @elapsed a = form(W, W, fun)
    function count_nodes(x, T)
        (x isa T ? 1 : 0) +
        (x isa Union{Bramble.LazyOp, Tuple} ?
         sum((count_nodes(getfield(x, i), T) for i in 1:nfields(x)); init = 0) : 0)
    end
    nprod = count_nodes(a.ast, Bramble.BilinearProduct)
    nnodes = count_nodes(a.ast, Bramble.LazyOp)
    if "--probe" in ARGS  # the AST only: which program does this tree build for the form?
        println("RESULT ", hash(repr(typeof(a.ast))), " ", nprod, " ", nnodes)
        exit(0)
    end
    A = nothing
    t_asm = @elapsed A = assemble(a)
    t_warm = minimum(@elapsed(assemble!(A, a)) for _ in 1:15)

    # Outside the timed calls: is `form` concretely inferred, and is `simplify_ast` idempotent
    # on the stored AST (the same type back)?
    inferred = try
        Test.@inferred form(W, W, fun)
        1.0
    catch
        0.0
    end
    idempotent = typeof(Bramble.simplify_ast(a.ast)) === typeof(a.ast) ? 1.0 : 0.0
    di = findfirst(==("--dump"), ARGS)
    if di !== nothing
        Serialization.serialize(ARGS[di + 1], A)
    end
    println("RESULT ", t_form, " ", t_asm, " ", t_warm, " ", nprod, " ", nnodes, " ",
        sum(abs, A), " ", sum(A), " ", inferred, " ", idempotent)
    exit(0)
end

# --- Parent ---------------------------------------------------------------------------- #

function arg(name, default)
    i = findfirst(==(name), ARGS)
    return i === nothing ? default : ARGS[i + 1]
end

const ROOT = abspath(joinpath(@__DIR__, ".."))
const DUMPDIR = mktempdir()
const SLOT = joinpath(ROOT, ".claude", "scripts", "julia_slot.sh")

function child(tree, form, threads, probe = false, dump = "")
    extra = String[]
    probe && push!(extra, "--probe")
    isempty(dump) || push!(extra, "--dump", dump)
    jl = `$(Base.julia_cmd()) --project=$tree --threads=$threads --startup-file=no
        $(abspath(@__FILE__)) --child $form $extra`
    cmd = haskey(ENV, "BRAMBLE_JULIA_SLOTS") && isfile(SLOT) ? `$SLOT $threads $jl` : jl
    out = IOBuffer()
    run(pipeline(ignorestatus(cmd); stdout = out, stderr = stderr))
    for line in eachline(IOBuffer(take!(out)))
        startswith(line, "RESULT ") && return parse.(Float64, split(line)[2:end])
    end
    error("child for $form on $tree printed no RESULT line")
end

median(v) = (s = sort(v); isodd(length(s)) ? s[(end + 1) ÷ 2] :
                          (s[end ÷ 2] + s[end ÷ 2 + 1]) / 2)

function fmt(t)
    t >= 1 ? string(round(t; sigdigits = 3), " s") :
    t >= 1e-3 ? string(round(t * 1e3; sigdigits = 3), " ms") :
    string(round(t * 1e6; sigdigits = 3), " µs")
end
cell(v) = string(fmt(median(v)), " [", fmt(minimum(v)), "-", fmt(maximum(v)), "]")

# win/loss/tie of the rule (tree A) on one time series pair
function verdict(a, b)
    return maximum(a) < minimum(b) ? "win" : minimum(a) > maximum(b) ? "loss" : "tie"
end

# 1-minute load average, and the highest CPU % of a julia process that is not one of this
# driver's children (those run this file with `--child`).
function machine_state()
    load = parse(Float64, split(read(`sysctl -n vm.loadavg`, String))[2])
    foreign = 0.0
    for line in eachline(`ps -Ao pcpu,command`)
        occursin("julia", line) && !occursin("simplify_rules.jl", line) || continue
        foreign = max(foreign, parse(Float64, split(strip(line))[1]))
    end
    return load, foreign
end

function main()
    cand = arg("--candidate", "")
    rule = cand == "" ? parse(Int, arg("--rule", "0")) : 0
    base = abspath(expanduser(arg("--base", "")))
    patched = abspath(expanduser(arg("--patched", "")))
    smoke = "--smoke" in ARGS
    maxload = parse(Float64, arg("--max-load", "6"))
    minpairs = smoke ? 1 : parse(Int, arg("--pairs", "3"))
    maxpairs = smoke ? 1 : parse(Int, arg("--max-pairs", "5"))
    threads = parse(Int, arg("--threads", "4"))
    forms = split(
        arg("--forms", join(cand == "" ? FORMS : get(CAND_DEFAULT, cand, CAND_FORMS), ",")), ",")
    if cand == ""
        rule in (1, 2, 3, 4, 7) || error("--rule must be 1, 2, 3, 4 or 7 (or --candidate NAME)")
    else
        cand in CANDIDATES || error("--candidate must be one of $(join(CANDIDATES, ", "))")
    end
    isdir(base) && isdir(patched) || error("--base and --patched must be directories")

    patch = joinpath(@__DIR__, "simplify_rules", cand == "" ? "rule$rule.patch" : "cand_$cand.patch")
    if cand != "" && startswith(readline(patch), "# not built")
        error("candidate $cand is not built: $(readline(patch))")
    end
    if success(`git -C $patched diff --quiet -- src`)
        run(`git -C $patched apply $patch`)
    end
    sha(t) = strip(read(`git -C $t rev-parse --short HEAD`, String))
    if cand == ""
        println("rule $rule, A (rule on) $base @ $(sha(base)), B (rule off) $patched @ $(sha(patched))")
    else
        # A is the tree with the candidate on (the patched one), B the base without it.
        base, patched = patched, base
        println("candidate $cand, A (candidate on) $base @ $(sha(base)) + patch, B (candidate off) $patched @ $(sha(patched))")
        rule = cand
    end
    println("threads $threads, pairs $minpairs-$maxpairs, forms $(join(forms, ", "))")
    if isfile(joinpath(ROOT, ".claude", "scripts", "check_power_load.sh"))
        run(ignorestatus(`$(joinpath(ROOT, ".claude", "scripts", "check_power_load.sh"))`))
    end

    # Precompile both trees once, outside the timed children.
    for t in (base, patched)
        jl = `$(Base.julia_cmd()) --project=$t --startup-file=no -e "using Bramble"`
        run(haskey(ENV, "BRAMBLE_JULIA_SLOTS") && isfile(SLOT) ? `$SLOT $threads $jl` : jl)
    end

    # The rule is a no-op on a form whose simplified AST has the same type in both trees:
    # A and B are then the same program, so the form is not timed.
    # `interp_fold` rewrites at block time, after the stored AST is built, so the AST probe
    # cannot see it: `--always` times every form (and the matrix comparison shows whether it fired).
    always = "--always" in ARGS || cand == "interp_fold"
    noop = Dict{String, Any}()
    active = String[]
    for f in forms
        pa, pb = child(base, f, threads, true), child(patched, f, threads, true)
        if pa == pb && !always
            noop[f] = (Int(pa[2]), Int(pa[3]))
            println("  $f: no-op (identical AST type hash, $(Int(pa[2])) routed, $(Int(pa[3])) nodes)")
        else
            push!(active, f)
            println("  $f: AST differs (routed $(Int(pa[2]))/$(Int(pb[2])), nodes $(Int(pa[3]))/$(Int(pb[3])))")
        end
    end

    runs = Dict((f, s) => Vector{Vector{Float64}}() for f in active, s in (:A, :B))
    discarded = 0
    loads = Float64[]
    # `minpairs` pairs per form, then up to `maxpairs` while the first ones disagree on the
    # sign of the difference or the median difference is within twice the spread.
    function needs_more(f)
        ra, rb = runs[(f, :A)], runs[(f, :B)]
        for k in (1, 2)  # first call: form + assemble, then warm
            a = [k == 1 ? x[1] + x[2] : x[3] for x in ra]
            b = [k == 1 ? x[1] + x[2] : x[3] for x in rb]
            diffs = b .- a
            (all(>(0), diffs) || all(<(0), diffs)) || return true
            abs(median(diffs)) > 2 * max(maximum(a) - minimum(a), maximum(b) - minimum(b)) ||
                return true
        end
        return false
    end
    for f in active
        r = 0
        while r < minpairs || (r < maxpairs && needs_more(f))
            r += 1
            order = isodd(r) ? ((:A, base), (:B, patched)) : ((:B, patched), (:A, base))
            # A pair is settled when no foreign julia process (not one of this driver's
            # children) ran above 100% CPU and the 1-minute load stayed within `--max-load`
            # (default 6: this run's own 4 threads count in it) at both ends of both
            # samples. An unsettled pair is discarded and rerun, up to 3 times; then the
            # pair with the lowest load is kept.
            best = nothing
            for attempt in 1:4
                pair = Dict{Symbol, Vector{Float64}}()
                lo, hi, foreign = Inf, 0.0, 0.0
                for (s, t) in order
                    l0, f0 = machine_state()
                    dump = r == 1 && attempt == 1 ? joinpath(DUMPDIR, "$(f)_$s.jls") : ""
                    pair[s] = child(t, f, threads, false, dump)
                    l1, f1 = machine_state()
                    lo, hi = min(lo, l0, l1), max(hi, l0, l1)
                    foreign = max(foreign, f0, f1)
                end
                settled = hi <= maxload && foreign <= 100
                if best === nothing || hi < best.hi
                    best = (pair = pair, hi = hi, lo = lo, foreign = foreign, settled = settled)
                end
                settled && break
                discarded += 1
                println("  pair $r $f discarded (load $lo-$hi, foreign julia $foreign% CPU)")
            end
            for s in (:A, :B)
                push!(runs[(f, s)], best.pair[s])
            end
            push!(loads, best.lo, best.hi)
            println("  pair $r $f: A first ", fmt(best.pair[:A][1] + best.pair[:A][2]),
                ", warm ", fmt(best.pair[:A][3]), "; B first ",
                fmt(best.pair[:B][1] + best.pair[:B][2]), ", warm ", fmt(best.pair[:B][3]),
                "; load $(best.lo)-$(best.hi), foreign julia $(best.foreign)% CPU",
                best.settled ? "" : " (UNSETTLED)")
            flush(stdout)
        end
    end
    println("\nload range of the kept pairs: ", isempty(loads) ? "n/a" : "$(minimum(loads))-$(maximum(loads))",
        "; discarded pairs: ", discarded)

    println("\nmedian [min-max]; A = ", cand == "" ? "rule" : "candidate", " on, B = off")
    println("| form | routed A/B | nodes A/B | pairs | first A | first B | assemble! A | assemble! B | first | assemble! |")
    wins = 0
    losses = 0
    for f in forms
        if haskey(noop, f)
            println("| $f | $(noop[f][1])/$(noop[f][1]) | $(noop[f][2])/$(noop[f][2]) | 0 | no-op on this form (identical AST) |")
            continue
        end
        ra, rb = runs[(f, :A)], runs[(f, :B)]
        fa = [x[1] + x[2] for x in ra]
        fb = [x[1] + x[2] for x in rb]
        wa = [x[3] for x in ra]
        wb = [x[3] for x in rb]
        np = (Int(ra[1][4]), Int(rb[1][4]))
        nn = (Int(ra[1][5]), Int(rb[1][5]))
        same = isapprox(ra[1][6], rb[1][6]; rtol = 1e-10) &&
               isapprox(ra[1][7], rb[1][7]; rtol = 1e-10, atol = 1e-10 * ra[1][6])
        v1, v2 = verdict(fa, fb), verdict(wa, wb)
        wins += (v1 == "win") + (v2 == "win")
        losses += (v1 == "loss") + (v2 == "loss")
        println("| $f | $(np[1])/$(np[2]) | $(nn[1])/$(nn[2]) | $(length(ra)) | ",
            cell(fa), " | ", cell(fb), " | ", cell(wa), " | ", cell(wb), " | ", v1, " | ", v2,
            " |", same ? "" : " MATRICES DIFFER")
        # The two matrices bit for bit (the first pair's dumps), and the inference and
        # idempotence flags of both trees' children.
        da = Serialization.deserialize(joinpath(DUMPDIR, "$(f)_A.jls"))
        db = Serialization.deserialize(joinpath(DUMPDIR, "$(f)_B.jls"))
        samepat = da.colptr == db.colptr && da.rowval == db.rowval
        bits = samepat && reinterpret(UInt64, da.nzval) == reinterpret(UInt64, db.nzval)
        ndiff = samepat ? count(reinterpret(UInt64, da.nzval) .!= reinterpret(UInt64, db.nzval)) : -1
        # The difference of the matrices as values: a stored zero held by one tree only does
        # not count, so a differing pattern can still be an equal matrix.
        relmax = maximum(abs, SparseArrays.nonzeros(da - db); init = 0.0) / maximum(abs, db.nzval)
        flags(k) = all(x -> length(x) >= k && x[k] == 1.0, vcat(ra, rb))
        println("  $f matrix: pattern ",
            samepat ? "equal" : "DIFFERS (stored $(length(da.nzval)) / $(length(db.nzval)))",
            ", bit-exact ", bits ? "yes" : "no ($ndiff of $(length(db.nzval)) stored differ)",
            ", max |A-B| / max |B| = $relmax; `form` inferred (on/off) ",
            all(x -> x[8] == 1.0, ra) ? "yes" : "NO", "/", all(x -> x[8] == 1.0, rb) ? "yes" : "NO",
            "; simplify_ast idempotent ", flags(9) ? "yes" : "NO")
    end
    println("\nrule $rule: ", wins, " win(s), ", losses, " loss(es) -> ",
        wins >= 1 && losses == 0 ? "keep" : (cand == "" ? "remove" : "drop"),
        " (before the matrix, inference and idempotence checks above)")
    return nothing
end

main()
