# Shared saved-results format for the standalone benchmark scripts.
#
# A results file is TOML: a `[meta]` table describing the run and one `[[tables.<name>]]`
# array of rows (plain scalars) per table. Files live at `benchmark/results/<script>.toml`,
# one per script, and a new full run overwrites the old one. They are not settled baselines.

using TOML
using Dates

"""
    save_results(path, script, tables; smoke = false)

Write `tables` (a name => vector of row `Dict`s of plain scalars) and the run metadata to
the TOML file `path`. `script` is the benchmark script's file name.
"""
function save_results(
        path::AbstractString, script::AbstractString,
        tables::AbstractDict{String}; smoke::Bool = false)
    data = Dict{String, Any}(
        "meta" => results_meta(script, smoke),
        "tables" => Dict{String, Any}(name => rows for (name, rows) in tables)
    )
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        TOML.print(io, data; sorted = true)
    end
    return path
end

"""
    load_results(path)

Read a file written by [`save_results`](@ref) back as a `Dict`.
"""
load_results(path::AbstractString) = TOML.parsefile(path)

function results_meta(script::AbstractString, smoke::Bool)
    return Dict{String, Any}(
        "script" => String(script),
        "commit" => results_commit(),
        "date" => Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "julia" => string(VERSION),
        "threads" => Threads.nthreads(),
        "power" => results_power(),
        "load1" => results_load1(),
        "cpu" => string(Sys.cpu_info()[1].model),
        "os" => string(Sys.KERNEL),
        "arch" => string(Sys.ARCH),
        "smoke" => smoke
    )
end

results_command(cmd::Cmd) =
    try
        strip(read(pipeline(cmd; stderr = devnull), String))
    catch
        ""
    end

function results_commit()
    dir = @__DIR__
    sha = results_command(Cmd(`git rev-parse --short HEAD`; dir))
    isempty(sha) && return "unknown"
    dirty = !isempty(results_command(Cmd(`git status --porcelain`; dir)))
    return dirty ? sha * "-dirty" : sha
end

function results_power()
    Sys.isapple() || return "unknown"
    out = results_command(`pmset -g batt`)
    occursin("AC Power", out) && return "ac"
    occursin("Battery Power", out) && return "battery"
    return "unknown"
end

function results_load1()
    out = Sys.isapple() ? results_command(`sysctl -n vm.loadavg`) : ""
    m = match(r"(\d+(?:[.,]\d+)?)", out)
    if m === nothing
        m = match(r"load averages?:\s*(\d+(?:[.,]\d+)?)", results_command(`uptime`))
    end
    m === nothing && return NaN
    return parse(Float64, replace(m.captures[1], ',' => '.'))
end
