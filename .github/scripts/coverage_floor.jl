# Per-file coverage report over one or more lcov files (#371).
#
#     julia coverage_floor.jl --floor <pct> [--exempt <toml>] [--compare <before.info>] [--prefix <dir/>] [--fail] <lcov>...
#
# A line counts as covered if any input covers it, so split uploads (the Weekly `suite` and
# `backends` halves) merge. Prints `<pct> <path> [exempt|BELOW|DROPPED]` per file, sorted by
# coverage, then a summary line.
#
# The `--exempt` TOML holds two kinds of entry. `[[exempt]]` (`path` glob, `reason`) exempts
# whole files. `[[exempt_function]]` (`path`, `functions`, `reason`) drops the lcov lines of the
# named functions from one file's total: each name is found in the source (every method: a
# `function` block through the `end` at its indent, or a one-line `name(args) = expr`) and an
# absent name is an error. An entry may instead (or also) carry `methods = [...]`, which names
# single methods by the text their definition starts with, after leading macros (`@inline`,
# `Base.@propagate_inbounds`, ...) and `function `, e.g. `_points!(x::AbstractVector, I::CartesianProduct{1}`.
# A `function` head is read whole: its lines are joined until the parentheses balance, and the
# signature and the prefix are compared with whitespace runs collapsed to one space and no space
# after `(` or before `)`, so `wrap(a::Int` names a `function wrap(` whose next line is `a::Int, b`.
# Any other line is a definition only when it is a real one-line method: after its balanced
# parentheses, an optional `where ...` and/or `::T`, comes a `=` (not `==`, not `=>`); docstring
# signatures (`name(args) -> T`), calls and comments are not definitions.
# A method spans a `function` block through the `end` at its indent, or a one-line definition
# through its balanced `([{` (so a `= throw(` whose body continues counts whole; brackets inside
# plain strings are ignored). A prefix matching no method or two or more is an error naming the
# path and prefix. An entry may also carry `headers = [...]`: method prefixes matched exactly as
# `methods` are, but only the FIRST line of the matched method (the `function` line, or the whole
# of a one-line method) leaves the total; the body stays measured. This is for the header line
# Julia never counts for an inlined function. An entry may also carry
# `lines = [{ method = "<prefix>", text = "<stripped line>" }, ...]`: the method is matched
# exactly as for `methods`, and the one line of it whose stripped text equals `text` leaves the
# total, for a line only another OS runs inside a method the others do run. Zero or two such
# lines is an error naming the path, prefix and text. `path` is relative to the working directory.
# `--prefix <dir/>` limits the BELOW and DROPPED flags, the `below=` and `dropped=` counts and
# `--fail` to files whose path starts with it; every file is still listed. Exits 1 only with
# `--fail` and a BELOW or DROPPED file.
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

# Whitespace runs collapsed to one space, none after `(` or before `)`: how a signature and a
# method prefix are compared.
normsig(s) = replace(replace(replace(strip(s), r"\s+" => " "), r"\( " => "("), r" \)" => ")")

# Net bracket depth of `t`, plain strings blanked out first.
function bracket_depth(t::AbstractString, open = "([{", close = ")]}")
    t = replace(t, r"\"[^\"]*\"" => "\"\"")
    return count(c -> c in open, t) - count(c -> c in close, t)
end

# Text from `head` on (the first line, macros stripped) joined with the following lines of `src`
# after `k` until the parentheses balance.
function joined_head(src, k, head)
    text = head
    j = k
    while bracket_depth(text, "(", ")") > 0 && j < length(src)
        j += 1
        text *= " " * strip(src[j])
    end
    return text
end

# Whether `text` (a candidate starting at a name) is a one-line method: after the balanced
# parentheses of its signature, an optional `where ...` and/or `::T`, then `=` (not `==`, `=>`).
function is_oneline_method(text::AbstractString)
    t = replace(text, r"\"[^\"]*\"" => "\"\"")
    i = findfirst('(', t)
    i === nothing && return false
    depth = 0
    for (j, c) in pairs(t)
        j < i && continue
        depth += (c == '(') - (c == ')')
        if depth == 0
            rest = t[nextind(t, j):end]
            return occursin(r"^\s*(?:::\s*[^=]+?)?(?:\s+where\s+[^=]+?)?(?:\s*::\s*[^=]+?)?\s*=(?![=>])", rest)
        end
    end
    return false
end

# Line numbers of the single method of `file` whose definition text starts with `prefix`
# (both compared after `normsig`): a `function` head is read whole (its lines joined until the
# parentheses balance); any other line counts only as a real one-line method, so docstring
# signatures (`name(args) -> T`), calls and comments are not definitions.
function method_lines(file::AbstractString, prefix::AbstractString)
    isfile(file) || error("exempt_function: source file not found: $file")
    src = readlines(file)
    want = normsig(prefix)
    spans = UnitRange{Int}[]
    for (k, line) in enumerate(src)
        head = replace(strip(line), r"^(?:(?:Base\.)?@\S+\s+)*" => "")
        isblock = startswith(head, "function ")
        isblock && (head = head[(length("function ") + 1):end])
        sig = normsig(joined_head(src, k, head))
        startswith(sig, want) || continue
        if isblock
            stop = Regex("^" * match(r"^\s*", line).match * "end\\b")
            last = findnext(l -> occursin(stop, l), src, k + 1)
            last === nothing && error("exempt_function: no closing `end` for `$prefix` at $file:$k")
            push!(spans, k:last)
        else
            depth = 0
            last = k
            while true
                depth += bracket_depth(src[last])
                (depth <= 0 || last >= length(src)) && break
                last += 1
            end
            is_oneline_method(join(strip.(src[k:last]), " ")) || continue
            push!(spans, k:last)
        end
    end
    length(spans) == 1 ||
        error("exempt_function: method prefix `$prefix` matches $(length(spans)) definitions in $file")
    return collect(only(spans))
end

# Line number of the single line of the method matched by `prefix` whose stripped text is `text`.
function method_line(file::AbstractString, prefix::AbstractString, text::AbstractString)
    src = readlines(file)
    hits = filter(l -> strip(src[l]) == text, method_lines(file, prefix))
    length(hits) == 1 ||
        error("exempt_function: method `$prefix` has $(length(hits)) lines with text `$text` in $file")
    return only(hits)
end

function main(args)
    floor = nothing
    exempt = Regex[]
    exempt_functions = Tuple{String,Vector{String},Vector{String},Vector{String},Vector{Tuple{String,String}}}[]
    prefix = ""
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
                push!(exempt_functions, (e["path"], String.(get(e, "functions", String[])),
                                         String.(get(e, "methods", String[])),
                                         String.(get(e, "headers", String[])),
                                         [(String(d["method"]), String(d["text"]))
                                          for d in get(e, "lines", [])]))
            end
        elseif a == "--compare"
            compare = args[i += 1]
        elseif a == "--prefix"
            prefix = args[i += 1]
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
    for (path, names, methods, headers, exlines) in exempt_functions
        drop = Set{Int}()
        for name in names
            union!(drop, function_lines(path, name))
        end
        for prefix in methods
            union!(drop, method_lines(path, prefix))
        end
        for h in headers
            push!(drop, first(method_lines(path, h)))
        end
        for (m, t) in exlines
            push!(drop, method_line(path, m, t))
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
        elseif startswith(path, prefix)
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
