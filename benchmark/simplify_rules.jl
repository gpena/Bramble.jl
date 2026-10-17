#===========================================================================#
# Audit of `simplify_ast` rules 1-4 and 7 (gpena/Bramble.jl#771): one rule's fresh-process
# A/B pair. A is a tree at the measured commit (the rule on), B a detached copy of that
# commit with the rule switched off by `benchmark/simplify_rules/rule<N>.patch`.
#
#     julia --startup-file=no benchmark/simplify_rules.jl --rule N --base DIR --patched DIR
#         [--pairs 3] [--max-pairs 5] [--threads 4] [--max-load 6] [--forms a,b,...] [--smoke]
#
# `--patched` is applied the patch with `git apply` when its `src/ast/simplifier.jl` is
# still clean; a tree that already differs there is used as it is. Rules: 1 `c*A+c*B`,
# 2 like terms, 3 zeros, 4 nested scale, 7 shared inner-product argument (`_factor`).
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

const FORMS = ["s2d_mix", "s3d_mix", "s3d_27", "v2d_elast", "v3d_elast", "v3d_lap"]

# Graded points of an axis: the exponent differs per axis, so no two axes agree.
graded(n, axis) = collect(range(0.0, 1.0; length = n)) .^ (1.2 + 0.2 * axis)

const POINTS = Dict(
    "s2d_mix" => (200, 180), "s3d_mix" => (28, 25, 22), "s3d_27" => (22, 20, 18),
    "v2d_elast" => (90, 80), "v3d_elast" => (16, 14, 12), "v3d_lap" => (22, 20, 18))

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
    else  # elasticity, Integer Lamé coefficients so that `2μ` and `λ` are equal
        (u, v) -> 2 * inner₊(εₕ(u), εₕ(v)) + 2 * inner₊(divₘₕ(u), divₘₕ(v))
    end

    t_form = @elapsed a = form(W, W, fun)
    count_nodes(x, T) = (x isa T ? 1 : 0) +
                        (x isa Union{Bramble.LazyOp, Tuple} ?
                         sum((count_nodes(getfield(x, i), T) for i in 1:nfields(x)); init = 0) : 0)
    nprod = count_nodes(a.ast, Bramble.BilinearProduct)
    nnodes = count_nodes(a.ast, Bramble.LazyOp)
    if "--probe" in ARGS  # the AST only: which program does this tree build for the form?
        println("RESULT ", hash(repr(typeof(a.ast))), " ", nprod, " ", nnodes)
        exit(0)
    end
    A = nothing
    t_asm = @elapsed A = assemble(a)
    t_warm = minimum(@elapsed(assemble!(A, a)) for _ in 1:15)

    println("RESULT ", t_form, " ", t_asm, " ", t_warm, " ", nprod, " ", nnodes, " ",
        sum(abs, A), " ", sum(A))
    exit(0)
end

# --- Parent ---------------------------------------------------------------------------- #

function arg(name, default)
    i = findfirst(==(name), ARGS)
    return i === nothing ? default : ARGS[i + 1]
end

const ROOT = abspath(joinpath(@__DIR__, ".."))
const SLOT = joinpath(ROOT, ".claude", "scripts", "julia_slot.sh")

function child(tree, form, threads, probe = false)
    jl = `$(Base.julia_cmd()) --project=$tree --threads=$threads --startup-file=no
        $(abspath(@__FILE__)) --child $form $(probe ? "--probe" : "")`
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
    rule = parse(Int, arg("--rule", "0"))
    base = abspath(expanduser(arg("--base", "")))
    patched = abspath(expanduser(arg("--patched", "")))
    smoke = "--smoke" in ARGS
    maxload = parse(Float64, arg("--max-load", "6"))
    minpairs = smoke ? 1 : parse(Int, arg("--pairs", "3"))
    maxpairs = smoke ? 1 : parse(Int, arg("--max-pairs", "5"))
    threads = parse(Int, arg("--threads", "4"))
    forms = split(arg("--forms", join(FORMS, ",")), ",")
    rule in (1, 2, 3, 4, 7) || error("--rule must be 1, 2, 3, 4 or 7")
    isdir(base) && isdir(patched) || error("--base and --patched must be directories")

    file = "src/ast/simplifier.jl"
    if success(`git -C $patched diff --quiet -- $file`)
        patch = joinpath(@__DIR__, "simplify_rules", "rule$rule.patch")
        run(`git -C $patched apply $patch`)
    end
    sha(t) = strip(read(`git -C $t rev-parse --short HEAD`, String))
    println("rule $rule, A (rule on) $base @ $(sha(base)), B (rule off) $patched @ $(sha(patched))")
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
    noop = Dict{String, Any}()
    active = String[]
    for f in forms
        pa, pb = child(base, f, threads, true), child(patched, f, threads, true)
        if pa == pb
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
                    pair[s] = child(t, f, threads)
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

    println("\nmedian [min-max]; A = rule on, B = rule off")
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
    end
    println("\nrule $rule: ", wins, " win(s), ", losses, " loss(es) -> ",
        wins >= 1 && losses == 0 ? "keep" : "remove")
    return nothing
end

main()
