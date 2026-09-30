# Per-file coverage report over one or more lcov files (#371).
#
#     julia coverage_floor.jl --floor <pct> [--exempt <toml>] [--compare <before.info>] [--fail] <lcov>...
#
# A line counts as covered if any input covers it, so split uploads (the Weekly `suite` and
# `backends` halves) merge. Prints `<pct> <path> [exempt|BELOW|DROPPED]` per file, sorted by
# coverage, then a summary line. Exits 1 only with `--fail` and a BELOW or DROPPED file.
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

function percentages(files)
    lines = Dict{String,Dict{Int,Bool}}()
    foreach(f -> read_lcov!(lines, f), files)
    return Dict(p => (isempty(d) ? 100.0 : 100 * count(values(d)) / length(d))
                for (p, d) in lines)
end

glob_regex(pattern) = Regex("^" * replace(pattern, r"[.+?^$()\[\]{}|\\]" => s"\\\0",
                                          "*" => "[^/]*") * "\$")

function main(args)
    floor = nothing
    exempt = Regex[]
    compare = nothing
    fail = false
    inputs = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--floor"
            floor = parse(Float64, args[i += 1])
        elseif a == "--exempt"
            for e in get(TOML.parsefile(args[i += 1]), "exempt", [])
                push!(exempt, glob_regex(e["path"]))
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

    pct = percentages(inputs)
    before = compare === nothing ? Dict{String,Float64}() : percentages([compare])
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
