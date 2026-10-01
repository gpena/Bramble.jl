# Per-file coverage report over one or more lcov files (#371).
#
#     julia coverage_floor.jl --floor <pct> [--exempt <toml>] [--compare <before.info>] [--fail] <lcov>...
#
# A line counts as covered if any input covers it, so split uploads (the Weekly `suite` and
# `backends` halves) merge. Prints `<pct> <path> [exempt|BELOW|DROPPED]` per file, sorted by
# coverage, then a summary line.
#
# The `--exempt` TOML holds two kinds of entry. `[[exempt]]` (`path` glob, `reason`) exempts
# whole files. `[[exempt_function]]` (`path`, `functions`, `reason`) drops the lcov lines of the
# named functions from one file's total: each name is found in the source (every method: a
# `function` block through the `end` at its indent, or a one-line `name(args) = expr`) and an
# absent name is an error. `path` is relative to the working directory. Exits 1 only with `--fail` and a BELOW or DROPPED file.
using TOML

function read_lcov!(lines::Dict{String,Dict{Int,Bool}}, file::AbstractString)
    current = ""
    for raw in eachline(file)
        if startswith(raw, "SF:")
            current = strip(raw[4:end])
            get!(lines, current, Dict{Int,Bool}())
        elseif startswith(raw, "DA:") && !isempty(current)
            fields = split(raw[4:end], ',')
            n = parse(Int, fields[1])
            hit = parse(Int, fields[2]) > 0
            d = lines[current]
            d[n] = get(d, n, false) || hit
        elseif raw == "end_of_record"
            current = ""
        end
    end
    return lines
end

function percentages(lines)
    return Dict(p => (isempty(d) ? 100.0 : 100 * count(values(d)) / length(d))
                for (p, d) in lines)
end

glob_regex(pattern) = Regex("^" * replace(pattern, r"[.+?^$()\[\]{}|\\]" => s"\\\0",
                                          "*" => "[^/]*") * "\$")

# Line numbers of every method of `name` in `file`: block definitions run from the `function`
# line to the `end` at the same indent, one-line definitions are a single line.
function function_lines(file::AbstractString, name::AbstractString)
    isfile(file) || error("exempt_function: source file not found: $file")
    src = readlines(file)
    n = replace(name, r"[.+?^$()\[\]{}|\\*]" => s"\\\0")
    head = "(?:@\\w+(?:\\([^)]*\\))?\\s+)*"
    qual = "(?:[A-Za-z_][\\w!]*\\.)*"
    block = Regex("^(\\s*)" * head * "(?:function|macro)\\s+" * qual * n * "(?=[\\s({]|\$)")
    oneline = Regex("^\\s*" * head * qual * n * "\\s*(?:\\{[^}]*\\})?\\(.*\\)(?:\\s+where\\s+.+?)?" *
                    "(?:\\s*::\\s*[^=]+?)?\\s*=(?!=)")
    found = Int[]
    k = 1
    while k <= length(src)
        line = src[k]
        m = match(block, line)
        if m !== nothing
            last = k
            if !occursin(r"\bend\s*$", line)  # else `function f end` or a one-liner
                stop = Regex("^" * m.captures[1] * "end\\b")
                last = findnext(l -> occursin(stop, l), src, k + 1)
                last === nothing && error("exempt_function: no closing `end` for $name at $file:$k")
            end
            append!(found, k:last)
            k = last
        elseif occursin(oneline, line)
            push!(found, k)
        end
        k += 1
    end
    isempty(found) && error("exempt_function: `$name` not found in $file")
    return found
end

function main(args)
    floor = nothing
    exempt = Regex[]
    exempt_functions = Tuple{String,Vector{String}}[]
    compare = nothing
    fail = false
    inputs = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--floor"
            floor = parse(Float64, args[i += 1])
        elseif a == "--exempt"
            toml = TOML.parsefile(args[i += 1])
            for e in get(toml, "exempt", [])
                push!(exempt, glob_regex(e["path"]))
            end
            for e in get(toml, "exempt_function", [])
                push!(exempt_functions, (e["path"], String.(e["functions"])))
            end
        elseif a == "--compare"
            compare = args[i += 1]
        elseif a == "--fail"
            fail = true
        else
            push!(inputs, a)
        end
        i += 1
    end
    floor === nothing && error("--floor <pct> is required")
    isempty(inputs) && error("no lcov input given")

    lines = Dict{String,Dict{Int,Bool}}()
    foreach(f -> read_lcov!(lines, f), inputs)
    for (path, names) in exempt_functions
        drop = Set{Int}()
        for name in names
            union!(drop, function_lines(path, name))
        end
        for (p, d) in lines
            (p == path || endswith(p, "/" * path)) && foreach(l -> delete!(d, l), drop)
        end
    end
    pct = percentages(lines)
    before = compare === nothing ? Dict{String,Float64}() : percentages(read_lcov!(Dict{String,Dict{Int,Bool}}(), compare))
    nbelow = nexempt = ndropped = 0
    for path in sort!(collect(keys(pct)); by = p -> (pct[p], p))
        p = pct[path]
        tag = ""
        if any(r -> occursin(r, path), exempt)
            tag = "exempt"
            nexempt += 1
        else
            if p < floor
                tag = "BELOW"
                nbelow += 1
            end
            if haskey(before, path) && p < before[path] - 1e-9
                tag = strip(tag * " DROPPED")
                ndropped += 1
            end
        end
        println(rstrip(string(round(p; digits = 1), " ", path, " ", tag)))
    end
    println("files=$(length(pct)) below=$nbelow exempt=$nexempt dropped=$ndropped")
    return fail && nbelow + ndropped > 0 ? 1 : 0
end

exit(main(ARGS))
