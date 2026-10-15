#===========================================================================#
# First-call cost of nested forms, with the form-level `Δₕ` and with the spelled-out sum
# `Δ(w) = Σ_d D̃_d(D₋_d(w))` side by side (gpena/Bramble.jl#774, #205).
#
# Plain Julia, Base and stdlib only, launched with no `--project` of its own, as
# benchmark/ttfx.jl is: it orchestrates child `julia` processes against the project named
# by `--project` (default: the tree itself).
#
#     julia --threads=4 --startup-file=no benchmark/nested_form_ttfx.jl [--mode table]
#         [--repeats 3] [--threads 4] [--dims 1,2,3] [--rows id,id] [--variants node,sum]
#         [--sizes literal|padded] [--deep] [--warm-reps 30] [--tree DIR] [--project DIR]
#         [--out FILE] [--dry-run]
#     julia --startup-file=no benchmark/nested_form_ttfx.jl --mode image
#         --tree DIR [--tree DIR ...] [--repeats 3] [--threads 4]
#     julia --startup-file=no benchmark/nested_form_ttfx.jl --mode gate
#         --tree DIR [--tree DIR ...] [--repeats 3] [--threads 4]
#
# ## Table mode
#
# Every first call runs in its own fresh child, so one form's compilation never pays for
# another's. A child builds a non-uniform (graded) mesh, a coefficient `c` and the data
# `fe`, `b`, then times, with nothing about the form compiled yet, `form` and the first
# `assemble` (the `constrained` call too for the ghost row). The warm cost follows in the
# same child: `assemble` repeated `--warm-reps` times, reporting the minimum, the median
# and the least bytes allocated. The parent launches the children `--repeats` times (default
# 3), the repeats interleaved. A round runs every dimension, form and spelling once before
# the next round starts, so a drift of the machine reaches every row alike. Each cell is the
# median [min-max] of the repeats, in seconds, and the ratio of that median to the median of
# the depth-1 baseline `innerₕ(c * D₋ₓ(u), D₋ₓ(v))` on the same mesh. The 1-minute load is
# read before every child and its range printed, with a label when it passes `--load-limit`
# (default 2). The load average counts the children of this very run, so a label is a
# reason to check the machine, not a proof that something else was running.
#
# Meshes. `literal` is #774's three meshes as written (graded; 1D 17, 2D 9x11, 3D 5x6x7
# points). `padded` adds the two padding points per axis the issue's follow-up counts
# (3D 7x8x9, 504 unknowns). The meshes are plain ones with the points moved by
# `change_points!`, with a different grading exponent on each axis.
#
# Spellings. `node` writes `Δₕ(w)`, `sum` the spelled-out sum. Rows that have no `Δ` have
# one spelling. A tree without the form-level `Δₕ` (plan-A main) fails the `node` rows,
# which are then reported as `n/a`: run it with `--variants sum` for the "before".
#
# ## Image mode
#
# The wall time of the first `using Bramble` after the package image is gone, for each
# `--tree`, with the trees interleaved in each round. Every run uses a throwaway depot in
# front of `~/.julia`, as benchmark/ttfx.jl does, so Bramble's own image is rebuilt while
# every dependency is found.
#
# ## Gate mode
#
# The package-owned invalidation count of test/quality/invalidations_snoop.jl (the gate of
# test/quality/invalidations.jl), run for each `--tree` in a temporary project that has the
# tree as `Bramble` and SnoopCompile. The counts are printed, the tree's own script is used.
#===========================================================================#

# --- Shared data ----------------------------------------------------------- #

# `Δ(` stands for `Δₕ(` or the spelled-out sum. `dims` restricts a row; `lap` marks a row
# that has both spellings; `deep` rows run with `--deep` only.
const ROWS = [
    (id = "base", text = "innerₕ(c * D₋ₓ(u), D₋ₓ(v))", depth = 1, side = "both", kind = :bil,
        dims = (1, 2, 3), lap = false, deep = false),
    (id = "dcx_u", text = "innerₕ(c * Dcₓ(u), v)", depth = 1, side = "trial", kind = :bil,
        dims = (1, 2, 3), lap = false, deep = false),
    (id = "dcx_v", text = "innerₕ(c * u, Dcₓ(v))", depth = 1, side = "test", kind = :bil,
        dims = (1, 2, 3), lap = false, deep = false),
    (id = "lap_u", text = "innerₕ(Δ(u), v)", depth = 2, side = "trial", kind = :bil,
        dims = (1, 2, 3), lap = true, deep = false),
    (id = "lap_v", text = "innerₕ(u, Δ(v))", depth = 2, side = "test", kind = :bil,
        dims = (1, 2, 3), lap = true, deep = false),
    (id = "lap_uv", text = "innerₕ(c * Δ(u), Δ(v))", depth = 2, side = "both", kind = :bil,
        dims = (1, 2, 3), lap = true, deep = false),
    (id = "jump", text = "innerₕ(c * jumpₓ(Δ(u)), jumpₓ(Δ(v)))", depth = 3, side = "both",
        kind = :bil, dims = (1, 2, 3), lap = true, deep = false),
    (id = "dm", text = "innerₕ(c * D₋ₓ(Δ(u)), D₋ₓ(Δ(v)))", depth = 3, side = "both",
        kind = :bil, dims = (1, 2, 3), lap = true, deep = false),
    (id = "lap_lap", text = "innerₕ(c * Δ(Δ(u)), v)", depth = 4, side = "trial", kind = :bil,
        dims = (1, 2, 3), lap = true, deep = false),
    (id = "lin0", text = "innerₕ(fe, v)", depth = 0, side = "linear", kind = :lin,
        dims = (1, 2, 3), lap = false, deep = false),
    (id = "lin1", text = "innerₕ(fe, v) + innerₕ(b, Dcₓ(v))", depth = 1, side = "linear",
        kind = :lin, dims = (1, 2, 3), lap = false, deep = false),
    (id = "lin_lap", text = "innerₕ(fe, Δ(v))", depth = 2, side = "linear", kind = :lin,
        dims = (1, 2, 3), lap = true, deep = false),
    (id = "facet", text = "inner_Γ(jumpₙ(Δ(u)), v; markers = (:xmin,))", depth = 3,
        side = "trial, facet", kind = :bil, dims = (1, 2, 3), lap = true, deep = false),
    (id = "mix_c", text = "innerₕ(Dcₓ(Dcᵧ(u)), v)", depth = 2, side = "trial", kind = :bil,
        dims = (2,), lap = false, deep = false),
    (id = "mix_d", text = "innerₕ(D̽ₓ(D̽ᵧ(u)), v)", depth = 2, side = "trial", kind = :bil,
        dims = (2,), lap = false, deep = false),
    (id = "lapk2", text = "innerₕ(Δ(Δ(u)), v)", depth = 4, side = "trial", kind = :bil,
        dims = (1, 2, 3), lap = false, deep = false),
    (id = "lapk3", text = "innerₕ(Δ(Δ(Δ(u))), v)", depth = 6, side = "trial", kind = :bil,
        dims = (1, 2, 3), lap = false, deep = false),
    (id = "lapk4", text = "innerₕ(Δ(Δ(Δ(Δ(u)))), v)", depth = 8, side = "trial",
        kind = :bil, dims = (1, 2), lap = false, deep = true),
    (id = "ghost", text = "constrained Robin space, then -Δₕ(u) and a load", depth = 2,
        side = "trial", kind = :ghost, dims = (1, 2, 3), lap = false, deep = false)
]

# #774's meshes as the issue writes them, and with the two padding points per axis.
const POINTS = Dict(1 => (17,), 2 => (9, 11), 3 => (5, 6, 7))
mesh_points(dim, sizes) = sizes == "padded" ? map(n -> n + 2, POINTS[dim]) : POINTS[dim]

# The graded points of an axis: the exponent differs per axis, so no two axes agree.
graded(n, axis) = collect(range(0.0, 1.0; length = n)) .^ (1.2 + 0.2 * axis)

# The spellings a row runs with: `lap` rows both, the others the one that does not read them.
function spellings(row, wanted)
    return row.lap ? [v for v in ("node", "sum") if v in wanted] : ["-"]
end

# Row text with `Δ(` filled in.
function row_text(row, variant, dim)
    variant == "sum" && return replace(row.text, "Δ(" => "lapsum$(dim)(")
    return replace(row.text, "Δ(" => "Δₕ(")
end

# --- Child: one fresh process, one first call ------------------------------ #

if "--child" in ARGS
    # --child <dim> <row id> <variant> <sizes> <warm reps>
    dim = parse(Int, ARGS[2])
    row = only(r for r in ROWS if r.id == ARGS[3])
    variant = ARGS[4]
    sizes = ARGS[5]
    warm_reps = parse(Int, ARGS[6])

    using Bramble
    import Bramble: D₋ₓ, D₋ᵧ, D₋₂, D̃ₓ, D̃ᵧ, D̃₂, Dcₓ, Dcᵧ, D̽ₓ, D̽ᵧ, jumpₓ, change_points!

    lapsum1(w) = D̃ₓ(D₋ₓ(w))
    lapsum2(w) = lapsum1(w) + D̃ᵧ(D₋ᵧ(w))
    lapsum3(w) = lapsum2(w) + D̃₂(D₋₂(w))

    function build_mesh(dim, sizes)
        n = mesh_points(dim, sizes)
        sets = (interval(0.0, 1.0), interval(0.0, 1.0) × interval(0.0, 1.0),
            box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        Ωₕ = mesh(domain(sets[dim]), dim == 1 ? n[1] : n)
        change_points!(Ωₕ, dim == 1 ? graded(n[1], 1) : ntuple(d -> graded(n[d], d), dim))
        return Ωₕ
    end

    # The first call: `form`, then `assemble`, each timed once. It is called through
    # `invokelatest`, so it runs in the world that has the closure `eval` defined.
    function first_call(W, fun, kind)
        if kind == :bil
            t_form = @elapsed a = form(W, W, fun)
            t_asm = @elapsed assemble(a)
            return a, (; pre = 0.0, form = t_form, asm = t_asm)
        end
        t_form = @elapsed l = form(W, fun)
        t_asm = @elapsed assemble(l)
        return l, (; pre = 0.0, form = t_form, asm = t_asm)
    end

    # The ghost row: the relation rows are built by `constrained`, a first call of their own.
    function first_call_ghost(Ωₕ, dim)
        facets = (:xmin, :xmax, :ymin, :ymax, :zmin, :zmax)[1:(2dim)]
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        t_pre = @elapsed C = constrained(
            Wg, u -> map(F -> F => 2.0 * u + ∂ₙ(u) => 0.5, facets))
        t_form = @elapsed begin
            a = form(C, C, (u, v) -> innerₕ(-1.0 * Δₕ(u), v))
            l = form(C, v -> innerₕ(x -> 1.0, v))
        end
        t_asm = @elapsed assemble(a, l)
        return (a, l), (; pre = t_pre, form = t_form, asm = t_asm)
    end

    # `assemble` repeated on the warm form: the least and the median time, and the least
    # bytes. A call that hits a garbage-collection pause is slow, and the median sees it.
    function warm_stats(f, args, reps, budget)
        ts = Float64[]
        bs = Int[]
        t0 = time()
        for _ in 1:reps
            s = @timed f(args...)
            push!(ts, s.time)
            push!(bs, s.bytes)
            time() - t0 > budget && break
        end
        sort!(ts)
        return minimum(ts), ts[cld(length(ts), 2)], minimum(bs), length(ts)
    end

    function child(dim, row, variant, sizes, warm_reps)
        Ωₕ = build_mesh(dim, sizes)
        W = gridspace(Ωₕ)
        n = ndofs(W)
        c = Rₕ(W, x -> 1.0 + 2.0 * first(x)^2)
        fe = Rₕ(W, x -> 1.0 + first(x))
        b = Rₕ(W, x -> 2.0 - first(x))
        Core.eval(Main, :(const c = $c))
        Core.eval(Main, :(const fe = $fe))
        Core.eval(Main, :(const b = $b))
        if row.kind == :ghost
            obj, t = first_call_ghost(Ωₕ, dim)
            wf, wargs = assemble, obj
        else
            text = row_text(row, variant, dim)
            head = row.kind == :bil ? "(u, v) -> " : "v -> "
            fun = Core.eval(Main, Meta.parse(head * text))
            obj, t = Base.invokelatest(first_call, W, fun, row.kind)
            wf, wargs = assemble, (obj,)
        end
        wmin, wmed, wbytes, wn = Base.invokelatest(warm_stats, wf, wargs, warm_reps, 8.0)
        total = t.pre + t.form + t.asm
        println("NFB\tpre=$(t.pre)\tform=$(t.form)\tasm=$(t.asm)\ttotal=$total",
            "\twarm_min=$wmin\twarm_med=$wmed\twarm_bytes=$wbytes\twarm_n=$wn\tndofs=$n")
    end

    try
        Base.invokelatest(child, dim, row, variant, sizes, warm_reps)
    catch e
        msg = first(split(sprint(showerror, e), '\n'))
        println("NFB\terror=", replace(msg, '\t' => ' '))
    end
    exit(0)
end

# --- Parent --------------------------------------------------------------- #

function fail(msg)
    println(stderr, "ERROR: ", msg)
    exit(1)
end

function parse_args(args)
    o = Dict{Symbol, Any}(:mode => "table", :repeats => 3, :threads => 4, :dims => [1, 2, 3],
        :rows => nothing, :variants => ["node", "sum"], :sizes => "literal", :deep => false,
        :warm_reps => 30, :trees => String[], :project => nothing, :out => nothing,
        :dry_run => false, :load_limit => 2.0)
    i = 1
    value() = (i += 1; i <= length(args) || fail("$(args[i - 1]) needs a value"); args[i])
    while i <= length(args)
        a = args[i]
        if a == "--mode"
            o[:mode] = value()
        elseif a == "--repeats"
            o[:repeats] = parse(Int, value())
        elseif a == "--threads"
            o[:threads] = parse(Int, value())
        elseif a == "--dims"
            o[:dims] = parse.(Int, split(value(), ','))
        elseif a == "--rows"
            o[:rows] = String.(split(value(), ','))
        elseif a == "--variants"
            o[:variants] = String.(split(value(), ','))
        elseif a == "--sizes"
            o[:sizes] = value()
        elseif a == "--deep"
            o[:deep] = true
        elseif a == "--warm-reps"
            o[:warm_reps] = parse(Int, value())
        elseif a == "--tree"
            push!(o[:trees], abspath(value()))
        elseif a == "--project"
            o[:project] = abspath(value())
        elseif a == "--out"
            o[:out] = abspath(value())
        elseif a == "--load-limit"
            o[:load_limit] = parse(Float64, value())
        elseif a == "--dry-run"
            o[:dry_run] = true
        else
            fail("unknown argument: $a")
        end
        i += 1
    end
    o[:mode] in ("table", "image", "gate") || fail("--mode is table, image or gate")
    o[:sizes] in ("literal", "padded") || fail("--sizes is literal or padded")
    return o
end

median_of(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

function sec(x)
    x >= 10 && return string(round(x; digits = 1))
    x >= 1 && return string(round(x; digits = 2))
    return string(round(x; digits = 3))
end

# `median [min-max]` of a vector of seconds.
function cell(v)
    return "$(sec(median_of(v))) [$(sec(minimum(v)))–$(sec(maximum(v)))]"
end

function jobs_of(o)
    jobs = NamedTuple[]
    for row in ROWS
        row.deep && !o[:deep] && continue
        o[:rows] !== nothing && !(row.id in o[:rows]) && row.id != "base" && continue
        for dim in o[:dims], variant in spellings(row, o[:variants])
            dim in row.dims && push!(jobs, (; row, dim, variant))
        end
    end
    return jobs
end

function fresh_depot_env()
    env = copy(ENV)
    env["JULIA_DEPOT_PATH"] = string(mktempdir(), ':', joinpath(homedir(), ".julia"))
    env["JULIA_PKG_OFFLINE"] = "true"
    return env
end

function git_hash(tree)
    try
        return readchomp(`git -C $tree rev-parse --short HEAD`)
    catch
        return "?"
    end
end

function table_mode(o)
    tree = isempty(o[:trees]) ? dirname(@__DIR__) : only(o[:trees])
    project = o[:project] === nothing ? tree : o[:project]
    jobs = jobs_of(o)
    jl = joinpath(Sys.BINDIR, Base.julia_exename())
    script = @__FILE__
    function cmd_of(j)
        return `$jl --project=$project --threads=$(o[:threads]) --startup-file=no $script --child $(j.dim) $(j.row.id) $(j.variant) $(o[:sizes]) $(o[:warm_reps])`
    end
    if o[:dry_run]
        println("tree $tree, project $project, $(length(jobs)) first calls per repeat")
        for j in jobs
            println(cmd_of(j))
        end
        return nothing
    end
    results = Dict{Any, Vector{Any}}()
    loads = Float64[]
    out = o[:out] === nothing ? nothing : open(o[:out], "w")
    for rep in 1:o[:repeats], j in jobs
        push!(loads, Sys.loadavg()[1])
        text = read(pipeline(cmd_of(j); stderr = devnull), String)
        lines = filter(startswith("NFB\t"), split(text, '\n'))
        line = isempty(lines) ? "NFB\terror=child printed no result" : last(lines)
        fields = Dict(split(f, '=')[1] => join(split(f, '=')[2:end], '=')
        for f in split(line, '\t')[2:end])
        key = (j.row.id, j.dim, j.variant)
        push!(get!(results, key, Any[]), fields)
        if out !== nothing
            println(out, rep, '\t', key[1], '\t', key[2], '\t', key[3], '\t', line)
            flush(out)
        end
        print(stderr, "repeat $rep  $(key[1]) $(key[2])D $(key[3])  ",
            haskey(fields, "error") ? "error: $(fields["error"])" :
            "$(sec(parse(Float64, fields["total"]))) s", '\n')
    end
    out === nothing || close(out)
    report_table(o, tree, jobs, results, loads)
    return nothing
end

function report_table(o, tree, jobs, results, loads)
    ok(key) = [f for f in get(results, key, Any[]) if !haskey(f, "error")]
    totals(key) = [parse(Float64, f["total"]) for f in ok(key)]
    base_median(dim) = isempty(totals(("base", dim, "-"))) ? NaN :
                       median_of(totals(("base", dim, "-")))
    println("\nTree $tree ($(git_hash(tree))), Julia $VERSION, $(Sys.CPU_THREADS) cores, ",
        "children at --threads=$(o[:threads]), sizes=$(o[:sizes]), repeats=$(o[:repeats]).")
    lo, hi = extrema(loads)
    label = hi > o[:load_limit] ? "UNDER LOAD (above $(o[:load_limit]))" : "idle"
    println("1-minute load before each child: min $(round(lo; digits = 2)), ",
        "max $(round(hi; digits = 2)), $label.")
    for dim in o[:dims]
        n = [parse(Int, f["ndofs"]) for k in keys(results) if k[2] == dim for f in ok(k)]
        isempty(n) || println("$(dim)D: $(first(n)) unknowns, points $(mesh_points(dim, o[:sizes])).")
    end
    println("\nFirst call of `form` plus `assemble`, seconds: median [min–max] and the ratio ",
        "to the depth-1 baseline on the same mesh. `Δ(w)` is the form-level `Δₕ(w)` or the ",
        "sum Σ_d D̃_d(D₋_d(w)).\n")
    println("| form | spelling | depth | side | ",
        join(["$(d)D" for d in o[:dims]], " | "), " |")
    println("|---|---|---|---|", repeat("---|", length(o[:dims])))
    seen = Set{Tuple{String, String}}()
    for j in jobs
        (j.row.id, j.variant) in seen && continue
        push!(seen, (j.row.id, j.variant))
        spell = j.variant == "node" ? "Δₕ" : j.variant == "sum" ? "sum" : "-"
        cells = String[]
        for dim in o[:dims]
            key = (j.row.id, dim, j.variant)
            if !haskey(results, key)
                push!(cells, "")
            elseif isempty(ok(key))
                push!(cells, "n/a")
            else
                t = totals(key)
                push!(cells, "$(cell(t)), $(round(median_of(t) / base_median(dim); digits = 1))×")
            end
        end
        println("| `$(j.row.text)` | $spell | $(j.row.depth) | $(j.row.side) | ",
            join(cells, " | "), " |")
    end
    println("\nWarm `assemble` after the first call: least and median milliseconds, ",
        "least bytes allocated.\n")
    println("| form | spelling | ", join(["$(d)D" for d in o[:dims]], " | "), " |")
    println("|---|---|", repeat("---|", length(o[:dims])))
    empty!(seen)
    for j in jobs
        (j.row.id, j.variant) in seen && continue
        push!(seen, (j.row.id, j.variant))
        spell = j.variant == "node" ? "Δₕ" : j.variant == "sum" ? "sum" : "-"
        cells = String[]
        for dim in o[:dims]
            key = (j.row.id, dim, j.variant)
            g = ok(key)
            if !haskey(results, key)
                push!(cells, "")
            elseif isempty(g)
                push!(cells, "n/a")
            else
                wmin = [1000 * parse(Float64, f["warm_min"]) for f in g]
                wmed = [1000 * parse(Float64, f["warm_med"]) for f in g]
                bytes = [parse(Int, f["warm_bytes"]) for f in g]
                push!(cells, "$(sec(median_of(wmin))) / $(sec(median_of(wmed))) ms, " *
                             "$(round(minimum(bytes) / 1024; digits = 1)) KiB")
            end
        end
        println("| `$(j.row.text)` | $spell | ", join(cells, " | "), " |")
    end
    println("\nSplit of the ghost row (`constrained`, `form`, `assemble`), median seconds:")
    for dim in o[:dims], f in [ok(("ghost", dim, "-"))]
        isempty(f) && continue
        parts = [median_of([parse(Float64, g[k]) for g in f]) for k in ("pre", "form", "asm")]
        println("  $(dim)D: ", join(sec.(parts), ", "))
    end
    return nothing
end

# --- Image mode ------------------------------------------------------------ #

function image_mode(o)
    trees = o[:trees]
    isempty(trees) && fail("image mode needs at least one --tree")
    jl = joinpath(Sys.BINDIR, Base.julia_exename())
    times = Dict(t => Float64[] for t in trees)
    loads = Float64[]
    for rep in 1:o[:repeats], t in trees
        push!(loads, Sys.loadavg()[1])
        cmd = setenv(
            `$jl --project=$t --threads=$(o[:threads]) --startup-file=no -e "using Bramble"`,
            fresh_depot_env())
        o[:dry_run] && (println(cmd); continue)
        t0 = time()
        run(cmd)
        push!(times[t], time() - t0)
        println(stderr, "repeat $rep  $(basename(t)) ($(git_hash(t)))  ", sec(last(times[t])), " s")
    end
    o[:dry_run] && return nothing
    println("\nFirst `using Bramble` in a fresh depot (the package image rebuilt), seconds, ",
        "children at --threads=$(o[:threads]). 1-minute load before each run: ",
        "max $(round(maximum(loads); digits = 2)).\n")
    println("| tree | commit | median [min–max] | runs |\n|---|---|---|---|")
    for t in trees
        println("| `$t` | $(git_hash(t)) | $(cell(times[t])) | ",
            join(sec.(times[t]), ", "), " |")
    end
    return nothing
end

# --- Gate mode ------------------------------------------------------------- #

function gate_mode(o)
    trees = o[:trees]
    isempty(trees) && fail("gate mode needs at least one --tree")
    jl = joinpath(Sys.BINDIR, Base.julia_exename())
    env = copy(ENV)
    env["JULIA_PKG_OFFLINE"] = "true"
    counts = Dict(t => Tuple{String, String}[] for t in trees)
    projects = Dict{String, String}()
    for t in trees
        dir = mktempdir()
        write(joinpath(dir, "Project.toml"), """
            [deps]
            Bramble = "72287966-2ff3-4442-8755-61ed245be5f7"
            SnoopCompile = "aa65fe97-06da-5843-b5b1-d5d13cad87d2"
            SnoopCompileCore = "e2b509da-e806-4183-be48-004708413034"

            [sources]
            Bramble = {path = "$t"}
            """)
        o[:dry_run] && (println("gate project $dir for $t"); continue)
        run(setenv(`$jl --project=$dir --startup-file=no -e
            "using Pkg; Pkg.resolve(); Pkg.precompile()"`, env))
        projects[t] = dir
    end
    o[:dry_run] && return nothing
    for rep in 1:o[:repeats], t in trees
        script = joinpath(t, "test", "quality", "invalidations_snoop.jl")
        out = read(setenv(`$jl --project=$(projects[t]) --threads=$(o[:threads])
            --startup-file=no $script`, env), String)
        owned = match(r"OWNED_COUNT=(\d+)", out)
        unattributed = match(r"UNATTRIBUTED_COUNT=(\d+)", out)
        push!(counts[t], (owned === nothing ? "?" : owned.captures[1],
            unattributed === nothing ? "?" : unattributed.captures[1]))
        println(stderr, "repeat $rep  $(basename(t)) ($(git_hash(t)))  owned ",
            counts[t][end][1], ", unattributed ", counts[t][end][2])
    end
    println("\nPackage-owned invalidations of `using Bramble` (the gate needs 0).\n")
    println("| tree | commit | owned per run | unattributed per run |\n|---|---|---|---|")
    for t in trees
        println("| `$t` | $(git_hash(t)) | ", join(first.(counts[t]), ", "), " | ",
            join(last.(counts[t]), ", "), " |")
    end
    return nothing
end

function main(args)
    o = parse_args(args)
    o[:mode] == "table" && return table_mode(o)
    o[:mode] == "image" && return image_mode(o)
    return gate_mode(o)
end

main(ARGS)
