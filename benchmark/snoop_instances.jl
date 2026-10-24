#===========================================================================#
# Counts the Bramble method instances inferred by the first `form` and `assemble` of a
# row (gpena/Bramble.jl#912, #917). Each child is a fresh process that loads Bramble and
# SnoopCompile, builds a graded 2D mesh and its `gridspace`, then runs `form` and
# `assemble` inside `@snoop_inference`. The count is the number of distinct
# `MethodInstance`s of Bramble or one of its submodules in `SnoopCompile.flatten`, the
# root excluded, summed over `form` and `assemble`.
#
# Rows, in two groups:
#   scale_float64, scale_float32, scale_pi, scale_rational
#       `form` of `c * inner₊(∇ₕ(u), ∇ₕ(v))` for c = 2.0, 2.0f0, π, 1 // 2, then
#       `assemble`.
#   closures_fresh, closures_named
#       a three-term linear form whose three sources are function coefficients, built twice
#       with the same `form` call. The second build takes three new anonymous closures
#       (`fresh`) or the same three named functions (`named`).
#
# Per row, in the TOML table and the markdown one
#   fresh_instances, fresh_form_s, fresh_assemble_s, fresh_inference_s
#       one fresh child per launch, medians over the launches. The counts must agree across
#       launches or the script stops. For the closure rows this is the first build.
#   sequence_position, sequence_instances, added_instances
#       For the scale rows one sequence child runs the selected scale rows in `--rows`
#       order in one process. `added_instances` is what the row's own call infers over the
#       rows before it (the first row's equals `sequence_instances`), `sequence_instances`
#       the distinct instances inferred up to and including the row. A closure row has
#       position 2, the second build of the fresh child, `added_instances` its count,
#       `sequence_instances` the distinct instances of both builds.
#   second_form_s, second_assemble_s, second_inference_s (closure rows only, TOML only)
#       the wall times of `form` and `assemble` and the inference time of the second build,
#       medians over the launches. Inference carries the positive control: fresh closures
#       infer the second build again (`added_instances` > 0), named functions do not (0).
#   repeat_instances
#       the row's last call repeated in the same process. It must be 0, being the control of
#       the counting oracle, and the script stops when it is not.
#   commit, julia, launches, threads, load_1min (and load_1min_after), the 1-minute load
#       average from `uptime` before the first and after the last child.
#
# Rejected: `--trace-compile` (#914: it lists roots only, one line for all of `assemble`)
# and counting `Base.specializations` (no inference time, and it sees instances compiled
# before the snoop).
#
# Plain Julia, Base and TOML only. The children run against `<tree>/test`, which holds
# SnoopCompile and SnoopCompileCore, with Bramble the tree's own, at the default `-O`.
#
#     julia --startup-file=no benchmark/snoop_instances.jl [--tree DIR] [--rows a,b]
#         [--launches 3] [--threads 1] [--out FILE]
#
# `--out FILE` merges a TOML table per row into FILE. Timings of a run while other jobs use
# the machine are not measurements: the load average is recorded to say so.
#===========================================================================#

# --- Child: one fresh process ---------------------------------------------- #

const CHILD = "--child" in ARGS

if CHILD
    using Bramble
    using SnoopCompileCore
    using SnoopCompile
    import Bramble: change_points!
end

# The macro `@snoop_inference` is expanded when its top-level statement is lowered, so the
# child's code is evaluated from a quote, after the `using` above has run: the parent has no
# SnoopCompileCore.
if CHILD
    Core.eval(Main,
        quote
            # `--child fresh <row>` or `--child sequence <row>...`
            mode = ARGS[2]
            rows = ARGS[3:end]

            const SCALARS = Dict("scale_float64" => 2.0, "scale_float32" => 2.0f0,
                "scale_pi" => π, "scale_rational" => 1 // 2)

            # The points of an axis: the exponent differs per axis, so no two axes agree.
            graded(n, axis) = collect(range(0.0, 1.0; length = n)) .^ (1.2 + 0.2 * axis)

            function setup()
                n = (9, 11)
                Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), n)
                change_points!(Ωₕ, ntuple(d -> graded(n[d], d), 2))
                return gridspace(Ωₕ)
            end

            function in_bramble(m::Module)
                return m === Bramble || (m !== parentmodule(m) && in_bramble(parentmodule(m)))
            end

            # The distinct Bramble instances of one snoop, and its total inference time.
            function bramble_instances(tinf)
                set = Set{Core.MethodInstance}()
                for t in flatten(tinf)
                    mi = Core.MethodInstance(t)
                    mi.def isa Method || continue         # the root's `def` is a module
                    in_bramble(mi.def.module) && push!(set, mi)
                end
                return set
            end

            # One call, snooped. Inference of the caller would infer the thunk's body, and its
            # callees, before the snoop starts, so the count would miss those instances. The
            # thunk is therefore not specialised on and is called through `invokelatest`.
            function phase(@nospecialize(thunk))
                value = Ref{Any}(nothing)
                wall = Ref(0.0)
                tinf = @snoop_inference begin
                    wall[] = @elapsed value[] = Base.invokelatest(thunk)
                end
                return (; set = bramble_instances(tinf), wall = wall[],
                    inference = SnoopCompile.inclusive(tinf), value = value[])
            end

            # `form` then `assemble` of one build, as two snoops.
            function form_assemble(@nospecialize(build))
                f = phase(build)
                a = phase(() -> assemble(f.value))
                return (; set = union(f.set, a.set), form_s = f.wall, assemble_s = a.wall,
                    inference_s = f.inference + a.inference)
            end

            scale_form(W, c) = form(W, W, (u, v) -> c * inner₊(∇ₕ(u), ∇ₕ(v)))

            # Three source coefficients as test/form/linear.jl writes them, one call that builds
            # the form from any three. The form's closure captures them, so its type follows
            # theirs.
            named_1(x) = x[1]
            named_2(x) = 100 * x[1]
            named_3(x) = x[2]
            anonymous_a() = (x -> x[1], x -> 100 * x[1], x -> x[2])
            anonymous_b() = (x -> x[1], x -> 100 * x[1], x -> x[2])
            function sources_form(W, a, b, c)
                return form(W, v -> innerₕ(a, v) + innerₕ(b, v) + innerₕ(c, v))
            end

            function check_default_opt()
                Base.JLOptions().opt_level == 2 ||
                    error("counting needs the default -O (2), got " *
                          "$(Base.JLOptions().opt_level)")
                return nothing
            end

            function run_child()
                check_default_opt()
                W = setup()
                if mode == "sequence"
                    seen = Set{Core.MethodInstance}()
                    for (i, row) in enumerate(rows)
                        c = SCALARS[row]
                        r = form_assemble(() -> scale_form(W, c))
                        union!(seen, r.set)
                        println("SEQUENCE ", row, " ", i, " ", length(r.set), " ", length(seen))
                    end
                    return nothing
                end
                row = only(rows)
                if haskey(SCALARS, row)
                    c = SCALARS[row]
                    r = form_assemble(() -> scale_form(W, c))
                    again = form_assemble(() -> scale_form(W, c))
                    println("FRESH ", length(r.set), " ", r.form_s, " ", r.assemble_s, " ",
                        r.inference_s, " ", length(again.set))
                    return nothing
                end
                named = (named_1, named_2, named_3)
                first_set, second_set = row == "closures_fresh" ?
                                        (anonymous_a(), anonymous_b()) : (named, named)
                r = form_assemble(() -> sources_form(W, first_set...))
                s = form_assemble(() -> sources_form(W, second_set...))
                again = form_assemble(() -> sources_form(W, second_set...))
                println("FRESH ", length(r.set), " ", r.form_s, " ", r.assemble_s, " ",
                    r.inference_s, " ", length(again.set))
                println("SEQUENCE ", row, " 2 ", length(s.set), " ",
                    length(union(r.set, s.set)), " ", s.form_s, " ", s.assemble_s, " ",
                    s.inference_s)
                return nothing
            end

            Base.invokelatest(run_child)
            exit(0)
        end)
end

using TOML

const SCALE_ROWS = ["scale_float64", "scale_float32", "scale_pi", "scale_rational"]
const CLOSURE_ROWS = ["closures_fresh", "closures_named"]
const ROWS = [SCALE_ROWS; CLOSURE_ROWS]

function arg(name, default)
    i = findfirst(==(name), ARGS)
    return i === nothing ? default : ARGS[i + 1]
end

function median(v)
    s = sort(v)
    return isodd(length(s)) ? s[(end + 1) ÷ 2] : (s[end ÷ 2] + s[end ÷ 2 + 1]) / 2
end

# --- Children -------------------------------------------------------------- #

# The lines the child printed that start with `tag`, split into words.
function launch(tree, threads, args)
    cmd = `$(Base.julia_cmd()) --project=$(joinpath(tree, "test")) --threads=$threads
        --startup-file=no $(abspath(@__FILE__)) --child $args`
    out = read(cmd, String)
    return [split(line)
            for line in eachline(IOBuffer(out))
            if startswith(line, "FRESH ") || startswith(line, "SEQUENCE ")]
end

function load_1min()
    m = match(r"load averages?:\s*([0-9]+[.,][0-9]+)", read(`uptime`, String))
    return m === nothing ? NaN : parse(Float64, replace(m[1], ',' => '.'))
end

function git_commit(tree)
    try
        return strip(read(`git -C $tree rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

# --- Aggregation over launches --------------------------------------------- #

function agree(what, row, vals)
    allequal(vals) || error("$what of $row differs between launches: $vals (a count that " *
          "changes between fresh launches is a finding, not noise)")
    return vals[1]
end

# The fresh measurement of a row over its launches, and its closure sequence lines.
function fresh_table(tree, row, n, threads)
    fresh = Vector{Vector{String}}[]
    for _ in 1:n
        push!(fresh, launch(tree, threads, ["fresh", row]))
    end
    freshes = [only(filter(l -> l[1] == "FRESH", lines)) for lines in fresh]
    table = Dict{String, Any}()
    table["fresh_instances"] = agree("fresh_instances", row,
        [parse(Int, f[2]) for f in freshes])
    table["fresh_form_s"] = median([parse(Float64, f[3]) for f in freshes])
    table["fresh_assemble_s"] = median([parse(Float64, f[4]) for f in freshes])
    table["fresh_inference_s"] = median([parse(Float64, f[5]) for f in freshes])
    rep = agree("repeat_instances", row, [parse(Int, f[6]) for f in freshes])
    table["repeat_instances"] = rep
    rep == 0 || error("the repeat of $row inferred $rep Bramble instances in the same " *
          "process; the counting oracle is broken")
    if row in CLOSURE_ROWS
        seqs = [only(filter(l -> l[1] == "SEQUENCE", lines)) for lines in fresh]
        table["sequence_position"] = agree("sequence_position", row,
            [parse(Int, s[3]) for s in seqs])
        table["added_instances"] = agree("added_instances", row,
            [parse(Int, s[4]) for s in seqs])
        table["sequence_instances"] = agree("sequence_instances", row,
            [parse(Int, s[5]) for s in seqs])
        table["second_form_s"] = median([parse(Float64, s[6]) for s in seqs])
        table["second_assemble_s"] = median([parse(Float64, s[7]) for s in seqs])
        table["second_inference_s"] = median([parse(Float64, s[8]) for s in seqs])
    end
    return table
end

# --- Output ---------------------------------------------------------------- #

fmt(x) = string(round(x; sigdigits = 4))

function print_table(title, rows, tables)
    println("\n### ", title, "\n")
    println("| row | fresh | added | sequence | position | repeat | form_s | assemble_s " *
            "| inference_s |")
    println("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows
        t = tables[r]
        println("| `", r, "` | ", t["fresh_instances"], " | ",
            get(t, "added_instances", ""), " | ", get(t, "sequence_instances", ""), " | ",
            get(t, "sequence_position", ""), " | ", t["repeat_instances"], " | ",
            fmt(t["fresh_form_s"]), " | ", fmt(t["fresh_assemble_s"]), " | ",
            fmt(t["fresh_inference_s"]), " |")
    end
    return nothing
end

function write_out(out, tables)
    out === nothing && return nothing
    data = isfile(out) ? TOML.parsefile(out) : Dict{String, Any}()
    merge!(data, tables)
    open(io -> TOML.print(io, data; sorted = true), out, "w")
    println("\nwrote ", out)
    return nothing
end

function main()
    tree = abspath(expanduser(arg("--tree", joinpath(@__DIR__, ".."))))
    rows = String.(split(arg("--rows", join(ROWS, ",")), ","))
    all(in(ROWS), rows) || error("--rows must be among $(join(ROWS, ", "))")
    n = parse(Int, arg("--launches", "3"))
    threads = parse(Int, arg("--threads", "1"))
    out = arg("--out", nothing)
    load_before = load_1min()
    tables = Dict{String, Any}()
    for r in rows
        tables[r] = fresh_table(tree, r, n, threads)
    end
    scale = filter(in(SCALE_ROWS), rows)
    if !isempty(scale)
        for l in launch(tree, threads, ["sequence"; scale])
            l[1] == "SEQUENCE" || continue
            t = tables[l[2]]
            t["sequence_position"] = parse(Int, l[3])
            t["added_instances"] = parse(Int, l[4])
            t["sequence_instances"] = parse(Int, l[5])
        end
        t1 = tables[scale[1]]
        t1["added_instances"] == t1["sequence_instances"] ||
            error("the first scale row's added count differs from its sequence count")
    end
    load_after = load_1min()
    commit, julia = git_commit(tree), string(VERSION)
    for r in rows
        t = tables[r]
        t["commit"], t["julia"], t["launches"], t["threads"] = commit, julia, n, threads
        t["load_1min"], t["load_1min_after"] = load_before, load_after
    end
    println("commit ", commit, ", Julia ", julia, ", ", n, " launch(es), ", threads,
        " thread(s), load ", load_before, " before, ", load_after, " after")
    any(in(SCALE_ROWS), rows) && print_table("scale rows", scale, tables)
    any(in(CLOSURE_ROWS), rows) &&
        print_table("closure rows", filter(in(CLOSURE_ROWS), rows), tables)
    write_out(out, tables)
    return nothing
end

main()
