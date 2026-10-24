#===========================================================================#
# Ranks the first-call compilation of a form by function (gpena/Bramble.jl#914). Each
# child process is fresh and runs `--trace-compile=stderr --trace-compile-timing`: it
# loads Bramble, builds a graded mesh and space, then calls `form` and `assemble` once,
# printing `#PHASE` markers to stderr between them. The parent keeps the raw trace of every
# child, ranks the `form` and `assemble` phases only, and groups the trace lines by the
# function they compile.
#
# Rows are those of benchmark/multiterm_ttfx.jl: `elasticity` and `vector_laplacian` on a
# graded 5x5x5 mesh, `scalar_2d` on a graded 9x9 mesh. Plain Julia, Base and TOML only,
# launched with no `--project` of its own: the children run against the project at `--tree`.
#
#     julia --startup-file=no benchmark/trace_compile.jl [--tree DIR] [--rows a,b]
#         [--launches 3] [--threads 1] [--top 20] [--out FILE]
#     julia --startup-file=no benchmark/trace_compile.jl --parse TRACE [--top 20]
#         [--out FILE]
#
# The launches are interleaved by round. Times are the median over launches. The number of
# instances and of recompilations of a function must agree across launches: when a fresh
# launch changes the trace, the script stops and names the function. The raw traces go to
# a temp directory, whose path is printed. `--out FILE` merges a TOML table per row into
# FILE. `--parse TRACE` ranks an existing raw trace into the table `parse`.
#
# The grouping key is the function of the first argument of the `Tuple{...}` signature:
# `typeof(f)` gives `f`, `Type{T{...}}` gives `T`, a closure type gives its name, all
# without type parameters, and `Core.kwcall` gives the function after the `NamedTuple`.
#
# Timings of a run while other jobs use the machine are not measurements.
#===========================================================================#

# --- Child: one fresh process, one first call ------------------------------ #

if "--child" in ARGS
    # --child <row>
    row = ARGS[2]
    println(stderr, "#PHASE setup")
    flush(stderr)
    using Bramble
    import Bramble: change_points!

    # The points of an axis: the exponent differs per axis, so no two axes agree.
    graded(n, axis) = collect(range(0.0, 1.0; length = n)) .^ (1.2 + 0.2 * axis)

    function build(row)
        if row == "scalar_2d"
            n = (9, 9)
            Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), n)
            change_points!(Ωₕ, ntuple(d -> graded(n[d], d), 2))
            return gridspace(Ωₕ), (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))
        end
        n = (5, 5, 5)
        Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), n)
        change_points!(Ωₕ, ntuple(d -> graded(n[d], d), 3))
        W = gridspace(Ωₕ)^Val(3)
        row == "vector_laplacian" && return W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))
        row == "elasticity" || error("unknown row $row")
        μ, λ = 1.0, 2.0
        return W, (u, v) -> 2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₘₕ(u), divₘₕ(v))
    end

    function first_call(row)
        W, fun = build(row)
        Base.JLOptions().opt_level == 2 ||
            error("timing needs the default -O (2), got $(Base.JLOptions().opt_level)")
        compile_ns() = (t = Base.cumulative_compile_time_ns(); t isa Tuple ? t[1] : t)
        Base.cumulative_compile_timing(true)
        c0 = compile_ns()
        println(stderr, "#PHASE form")
        flush(stderr)
        t_form = @elapsed a = form(W, W, fun)
        println(stderr, "#PHASE assemble")
        flush(stderr)
        t_asm = @elapsed assemble(a)
        c1 = compile_ns()
        println(stderr, "#PHASE end")
        flush(stderr)
        println("RESULT ", t_form, " ", t_asm, " ", (c1 - c0) / 1.0e9)
        return nothing
    end

    Base.invokelatest(first_call, row)
    exit(0)
end

using TOML

const ROWS = ["elasticity", "vector_laplacian", "scalar_2d"]
const PHASES = ("form", "assemble")
const TRACE_LINE = r"^#=\s*([0-9.]+) ms =# precompile\((Tuple\{.*\})\)( # recompile)?\s*$"

function arg(name, default)
    i = findfirst(==(name), ARGS)
    return i === nothing ? default : ARGS[i + 1]
end

function median(v)
    s = sort(v)
    return isodd(length(s)) ? s[(end + 1) ÷ 2] : (s[end ÷ 2] + s[end ÷ 2 + 1]) / 2
end

# --- Grouping key ---------------------------------------------------------- #

# Splits `s` at the commas that are outside every bracket and every quoted name.
function split_top(s::AbstractString)
    parts = String[]
    depth, quoted, start = 0, false, firstindex(s)
    for (i, c) in pairs(s)
        if quoted
            c == '"' && (quoted = false)
        elseif c == '"'
            quoted = true
        elseif c in "{(["
            depth += 1
        elseif c in "})]"
            depth -= 1
        elseif c == ',' && depth == 0
            push!(parts, strip(s[start:prevind(s, i)]))
            start = nextind(s, i)
        end
    end
    push!(parts, strip(s[start:end]))
    return parts
end

# `x` up to its first type parameter list.
function strip_params(x::AbstractString)
    quoted = false
    for (i, c) in pairs(x)
        if c == '"'
            quoted = !quoted
        elseif c == '{' && !quoted
            return String(x[1:prevind(x, i)])
        end
    end
    return String(x)
end

function key_of(parts::Vector{String})
    f = parts[1]
    f == "typeof(Core.kwcall)" && length(parts) >= 3 && return key_of(parts[3:end])
    if startswith(f, "typeof(") && endswith(f, ")")
        return strip_params(chop(f; head = 7, tail = 1))
    elseif startswith(f, "Type{") && endswith(f, "}")
        return strip_params(chop(f; head = 5, tail = 1))
    end
    return strip_params(f)
end

# The key of a `Tuple{...}` signature.
function group_key(sig::AbstractString)
    inner = chop(sig; head = length("Tuple{"), tail = 1)
    isempty(inner) && return "Tuple{}"
    return key_of(split_top(inner))
end

# --- Parsing a raw trace --------------------------------------------------- #

# One launch: the form and assemble trace lines grouped by function.
function parse_trace(path)
    funcs = Dict{String, Any}()
    recompiles = String[]
    lines, trace_s = 0, 0.0
    phase = ""
    for line in eachline(path)
        if startswith(line, "#PHASE ")
            phase = strip(line[8:end])
            continue
        end
        phase in PHASES || continue
        m = match(TRACE_LINE, line)
        m === nothing && continue
        secs = parse(Float64, m[1]) / 1000
        sig = String(m[2])
        re = m[3] !== nothing
        f = get!(funcs, group_key(sig), Dict{String, Any}("form_s" => 0.0,
            "assemble_s" => 0.0, "instances" => 0, "recompiled" => 0))
        f[phase * "_s"] += secs
        f["instances"] += 1
        re && (f["recompiled"] += 1; push!(recompiles, sig))
        lines += 1
        trace_s += secs
    end
    return (; funcs, recompiles, lines, trace_s)
end

# --- Children -------------------------------------------------------------- #

function launch(tree, row, threads, trace_path)
    cmd = `$(Base.julia_cmd()) --project=$tree --threads=$threads --startup-file=no
        --trace-compile=stderr --trace-compile-timing $(abspath(@__FILE__)) --child $row`
    out = read(pipeline(cmd; stderr = trace_path), String)
    for line in eachline(IOBuffer(out))
        startswith(line, "RESULT ") || continue
        t = parse.(Float64, split(line)[2:4])
        return (; form_wall_s = t[1], assemble_wall_s = t[2], compile_s = t[3])
    end
    error("child for $row printed no RESULT line; its stderr is in $trace_path")
end

# --- Aggregation over launches --------------------------------------------- #

function aggregate(parsed)
    n = length(parsed)
    names = sort!(collect(union((keys(p.funcs) for p in parsed)...)))
    funcs = Dict{String, Any}()
    zero_f = Dict("form_s" => 0.0, "assemble_s" => 0.0, "instances" => 0, "recompiled" => 0)
    for k in names
        fs = [get(p.funcs, k, zero_f) for p in parsed]
        for what in ("instances", "recompiled")
            vals = [f[what] for f in fs]
            allequal(vals) ||
                error("$what of $k differs between launches: $vals (a trace that changes " *
                      "between fresh launches is a finding, not noise)")
        end
        form_s = median([f["form_s"] for f in fs])
        assemble_s = median([f["assemble_s"] for f in fs])
        funcs[k] = Dict{String, Any}("total_s" => form_s + assemble_s, "form_s" => form_s,
            "assemble_s" => assemble_s, "instances" => fs[1]["instances"],
            "recompiled" => fs[1]["recompiled"])
    end
    recompiles = sort(parsed[1].recompiles)
    for p in parsed
        sort(p.recompiles) == recompiles || error(
            "the recompiled signatures differ between launches: ",
            symdiff(recompiles, p.recompiles))
    end
    return Dict{String, Any}("trace_lines" => parsed[1].lines,
        "trace_s" => median([p.trace_s for p in parsed]), "launches" => n,
        "recompiles" => recompiles, "functions" => funcs)
end

# --- Output ---------------------------------------------------------------- #

fmt(x) = string(round(x; sigdigits = 4))

function print_tables(name, table, top)
    funcs = table["functions"]
    println("\n### ", name, "\n")
    println("trace lines ", table["trace_lines"], ", trace sum ", fmt(table["trace_s"]),
        " s, ", length(funcs), " functions, ", length(table["recompiles"]),
        " recompiled signatures")
    if haskey(table, "compile_s")
        println("compile clock ", fmt(table["compile_s"]), " s, form ",
            fmt(table["form_wall_s"]), " s, assemble ", fmt(table["assemble_wall_s"]),
            " s (wall), ", table["launches"], " launch(es), ", table["threads"],
            " thread(s), commit ", table["commit"], ", Julia ", table["julia"])
    end
    for (title, by) in (("Top by total_s", "total_s"), ("Top by instances", "instances"))
        println("\n", title, "\n")
        println("| function | total_s | form_s | assemble_s | instances | recompiled |")
        println("|---|---:|---:|---:|---:|---:|")
        ranked = sort!(collect(funcs); by = kv -> (-kv[2][by], kv[1]))
        for (k, f) in first(ranked, top)
            println("| `", k, "` | ", fmt(f["total_s"]), " | ", fmt(f["form_s"]), " | ",
                fmt(f["assemble_s"]), " | ", f["instances"], " | ", f["recompiled"], " |")
        end
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

function git_commit(tree)
    try
        return strip(read(`git -C $tree rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

function main()
    top = parse(Int, arg("--top", "20"))
    out = arg("--out", nothing)
    parse_file = arg("--parse", nothing)
    if parse_file !== nothing
        table = aggregate([parse_trace(parse_file)])
        print_tables("parse", table, top)
        write_out(out, Dict("parse" => table))
        return nothing
    end
    tree = abspath(expanduser(arg("--tree", joinpath(@__DIR__, ".."))))
    rows = split(arg("--rows", join(ROWS, ",")), ",")
    all(in(ROWS), rows) || error("--rows must be among $(join(ROWS, ", "))")
    n = parse(Int, arg("--launches", "3"))
    threads = parse(Int, arg("--threads", "1"))
    dir = mktempdir(; cleanup = false)
    println("raw traces in ", dir)
    runs = Dict(r => Any[] for r in rows)
    for l in 1:n, r in rows

        path = joinpath(dir, "$(r)_$l.txt")
        t = launch(tree, r, threads, path)
        push!(runs[r], (; t, p = parse_trace(path)))
    end
    commit = git_commit(tree)
    tables = Dict{String, Any}()
    for r in rows
        table = aggregate([x.p for x in runs[r]])
        for k in (:compile_s, :form_wall_s, :assemble_wall_s)
            table[String(k)] = median([getfield(x.t, k) for x in runs[r]])
        end
        table["threads"] = threads
        table["commit"] = commit
        table["julia"] = string(VERSION)
        tables[r] = table
        print_tables(r, table, top)
    end
    write_out(out, tables)
    return nothing
end

main()
