# Generator for docs/src/benchmarks.md from saved benchmark JSON files.

using BenchmarkTools
using Dates

include(joinpath(@__DIR__, "plotly_common.jl"))

const _BENCH_CHART_COUNTER = Ref(0)
_next_bench_div_id() = "bench_chart_$(_BENCH_CHART_COUNTER[] += 1)"

function _get_commit_info(commit_hash::AbstractString, path::AbstractString)
    try
        msg = readchomp(pipeline(`git log -1 --format="%s" $commit_hash`; stderr = devnull))
        ct = parse(
            Int,
            readchomp(pipeline(`git log -1 --format="%ct" $commit_hash`; stderr = devnull))
        )
        return (message = msg, time = ct)
    catch
        return (message = "", time = round(Int, mtime(path)))
    end
end

# Baselines saved before benchmarks.jl started tagging `pkgversion:` carry no version in
# their JSON. Retrace it the same way `_get_commit_info` retraces the commit message: read
# Project.toml as it stood at that commit. Falls back to "unknown" outside a git checkout
# or for a commit no longer reachable.
function _get_pkg_version(commit_hash::AbstractString)
    try
        toml = readchomp(pipeline(`git show $(commit_hash):Project.toml`; stderr = devnull))
        m = match(r"^version\s*=\s*\"([^\"]+)\""m, toml)
        return m !== nothing ? m.captures[1] : "unknown"
    catch
        return "unknown"
    end
end

function _format_time(t_ns::Real)
    if t_ns < 1_000
        return string(round(t_ns; digits = 1), " ns")
    elseif t_ns < 1_000_000
        return string(round(t_ns / 1_000; digits = 1), " μs")
    elseif t_ns < 1_000_000_000
        return string(round(t_ns / 1_000_000; digits = 2), " ms")
    else
        return string(round(t_ns / 1_000_000_000; digits = 2), " s")
    end
end

# Benchmarks kept in the suite for their allocation count alone (checked against
# `benchmark/benchmarks.jl`'s `ALLOCATION_BOUNDS`), whose time is not charted. "form
# (linear, 2D)" constructs a `LinearForm` in about 10 ns: at that scale the in-suite median
# follows whatever ran before it, not the code (22.1 ns in the v3.11.0 baseline, 10.2 ns
# measured alone, identical to v3.4.0), and its v2.0.0 reference was a call optimised away
# entirely, so the chart read a flat 10 ns cost as a tenfold regression.
const _BENCH_ALLOCATION_GUARDS = Set([("forms", "form (linear, 2D)")])

# The first and third quartile of one run's samples of one benchmark, or `nothing` when the
# baseline recorded no spread. `benchmarks.jl` saves a trial with more than 5 samples as
# the five values [min, q1, median, q3, max], and a trial with 5 or fewer as its raw
# samples; baselines from before that saved [min, median, max] (3 values) or raw samples.
# A five-value list is therefore either a reduced trial or five raw samples, and the two
# need no telling apart: sorted, both have q1 at index 2 and q3 at index 4, which is also
# what `_sorted_quantile` returns for five values. So any list of 4 or more values gives its
# quartiles by `_sorted_quantile` on the sorted list, and 3 or fewer (a saved
# [min, median, max], or a raw trial too short to have quartiles) gives none.
function _sorted_quantile(sorted, p)
    pos = 1 + (length(sorted) - 1) * p
    lo = floor(Int, pos)
    hi = min(lo + 1, length(sorted))
    return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo])
end

function _quartiles(times)
    length(times) <= 3 && return nothing
    sorted = sort(collect(Float64, times))
    return (_sorted_quantile(sorted, 0.25), _sorted_quantile(sorted, 0.75))
end

_threads_phrase(n) = n == "1" ? "1 thread" : "$n threads"

# `latest` against `previous` for one benchmark: the ratio of medians and whether the two
# interquartile ranges are disjoint. A pair where either run recorded no spread is never
# flagged (`spread = false`).
function _compare_trials(previous, latest)
    q_prev, q_new = _quartiles(previous.times), _quartiles(latest.times)
    ratio = time(median(latest)) / time(median(previous))
    if q_prev === nothing || q_new === nothing
        return (ratio = ratio, spread = false, flagged = false)
    end
    flagged = q_new[2] < q_prev[1] || q_prev[2] < q_new[1]
    return (ratio = ratio, spread = true, flagged = flagged)
end

# One row per benchmark present in both of the last two runs, in group order, then by name.
function _release_comparison(runs, ordered_groups)
    previous, latest = runs[end - 1], runs[end]
    rows = NamedTuple[]
    for gname in ordered_groups
        (haskey(previous.data, gname) && haskey(latest.data, gname)) || continue
        for bname in sort(string.(collect(keys(latest.data[gname]))))
            (gname, bname) in _BENCH_ALLOCATION_GUARDS && continue
            haskey(previous.data[gname], bname) || continue
            old, new = previous.data[gname][bname], latest.data[gname][bname]
            c = _compare_trials(old, new)
            push!(
                rows,
                (
                    group = gname, name = bname, ratio = c.ratio, spread = c.spread,
                    flagged = c.flagged, old_ns = time(median(old)), new_ns = time(median(new))
                )
            )
        end
    end
    return rows
end

# A horizontal bar per benchmark at its latest/previous ratio of medians on a log axis,
# coloured by whether the change is beyond the spread. The `data-bench` and `data-flagged`
# attributes are read by the checks in `.claude/plans/v3-21-0-benchmarks-page-checks/`.
function _render_release_chart(rows)
    div_id = _next_bench_div_id()
    height = 80 + length(rows) * 16
    flagged = join(("$(r.group)/$(r.name)" for r in rows if r.flagged), "|")
    labels = join(("\"$(r.group)/$(r.name)\"" for r in rows), ",")
    values = join((r.ratio for r in rows), ",")
    colors = join(
        (
            "\"" * (
                !r.flagged ? "#9ca3af" : (r.ratio > 1 ? "#ef4444" : "#10b981")
            ) * "\"" for r in rows
        ),
        ","
    )
    tooltips = join(
        (
            "\"$(_format_time(r.old_ns)) → $(_format_time(r.new_ns)), ×$(round(r.ratio; digits = 3))" *
            (r.spread ? (r.flagged ? " (beyond the spread)" : " (within the spread)") :
             " (no spread recorded)") * "\"" for r in rows
        ),
        ","
    )
    return """
    <div id="$div_id" data-bench="regression" data-flagged="$flagged" style="width:100%; height:$(height)px;"></div>
    <script>
    (function () {
      const theme = window.bramblePlotlyTheme();
      const data = [{
        type: 'bar',
        orientation: 'h',
        y: [$labels],
        x: [$values],
        marker: { color: [$colors] },
        hovertext: [$tooltips],
        hoverinfo: 'text',
      }];
      const layout = {
        paper_bgcolor: theme.bg,
        plot_bgcolor: theme.bg,
        font: { color: theme.text },
        showlegend: false,
        shapes: [{
          type: 'line', xref: 'x', yref: 'paper', x0: 1, x1: 1, y0: 0, y1: 1,
          line: { color: theme.text, width: 1, dash: 'dot' },
        }],
        xaxis: {
          title: { text: "latest / previous median", font: { color: theme.text } },
          type: 'log', color: theme.text, gridcolor: theme.grid,
        },
        yaxis: { color: theme.text, autorange: 'reversed', tickfont: { size: 9 } },
        margin: { t: 20, l: 260, r: 20, b: 50 },
      };
      Plotly.newPlot('$div_id', data, layout, { displayModeBar: false, responsive: true });
      window.brambleRegisterPlotlyChart('$div_id', function () {
        const t = window.bramblePlotlyTheme();
        return {
          'font.color': t.text,
          'xaxis.color': t.text, 'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.text,
          'yaxis.color': t.text,
        };
      });
    })();
    </script>
    """
end

# `results_dir` is where the standalone scripts' saved tables live; the release view reads
# baselines only and takes it for the sections that read those tables.
function generate_benchmarks_markdown(
        benchmark_dir = normpath(joinpath(@__DIR__, "..", "benchmark", "baselines")),
        output_path = normpath(joinpath(@__DIR__, "src", "benchmarks.md")),
        results_dir = normpath(joinpath(@__DIR__, "..", "benchmark", "results"))
)
    json_files = String[]
    for dir in (benchmark_dir, normpath(joinpath(@__DIR__, "..", "benchmark")))
        if isdir(dir)
            for f in readdir(dir)
                if endswith(f, ".json") && startswith(f, "baseline_")
                    p = joinpath(dir, f)
                    p in json_files || push!(json_files, p)
                end
            end
        end
    end

    io = IOBuffer()
    println(io, "# Performance and benchmarks")
    println(io)
    println(
        io,
        "Bramble tracks memory allocations and performance regressions with a dedicated regression suite in `benchmark/benchmarks.jl`."
    )
    println(
        io,
        "Most operator and restriction benchmarks run on about one million grid points per setup (\$1000 \\times 1000\$ in 2D, \$100 \\times 100 \\times 100\$ in 3D); assembly and precision benchmarks use smaller grids, set in `benchmark/benchmarks.jl`."
    )
    println(io)

    if isempty(json_files)
        println(io, "!!! note \"No baselines recorded\"")
        println(io, "    No saved benchmark baselines were found in `benchmark/baselines/`.")
        println(io, "    To run and save a baseline locally on AC power:")
        println(io, "    ```bash")
        println(
            io,
            "    julia --project=benchmark benchmark/benchmarks.jl --save benchmark/baselines/baseline_\$(git rev-parse --short HEAD).json"
        )
        println(io, "    ```")
        open(output_path, "w") do f
            return write(f, String(take!(io)))
        end
        return output_path
    end

    # Parse all benchmark baselines
    runs = []
    for path in json_files
        fname = basename(path)
        m = match(r"baseline_([a-zA-Z0-9_-]+)\.json", fname)
        commit = m !== nothing ? m.captures[1] : replace(fname, ".json" => "")
        info = _get_commit_info(commit, path)
        data = BenchmarkTools.load(path)[1]
        julia_ver = "unknown"
        pkg_ver = nothing
        threads = nothing
        for t in data.tags
            if startswith(string(t), "julia:")
                julia_ver = replace(string(t), "julia:" => "")
            elseif startswith(string(t), "pkgversion:")
                pkg_ver = replace(string(t), "pkgversion:" => "")
            elseif startswith(string(t), "threads:")
                threads = replace(string(t), "threads:" => "")
            end
        end
        # Baselines saved before the `pkgversion:` tag existed carry none — retrace it
        # from Project.toml at that commit instead of leaving it blank.
        pkg_ver === nothing && (pkg_ver = _get_pkg_version(commit))
        push!(
            runs,
            (
                commit = commit,
                message = info.message,
                time = info.time,
                julia = julia_ver,
                version = pkg_ver,
                # The thread count a run was recorded at. `nothing` for baselines saved
                # before benchmarks.jl tagged it; those read 0 allocations for `Rₕ!`, so
                # they were single-threaded (see the note in benchmarks.jl's `main`).
                threads = threads === nothing ? "1" : threads,
                data = data,
                path = path
            )
        )
    end
    # Order runs chronologically by commit timestamp
    sort!(runs; by = r -> r.time)

    # Collect all groups dynamically
    group_order = [
        "operators 2D",
        "operators 3D",
        "jumps & averages",
        "inner products 2D",
        "restriction",
        "composite",
        "construction",
        "startup & latency"
    ]
    all_groups = Set{String}()
    for r in runs
        for k in keys(r.data)
            push!(all_groups, string(k))
        end
    end
    ordered_groups = filter(in(all_groups), group_order)
    for g in sort(collect(all_groups))
        g in ordered_groups || push!(ordered_groups, g)
    end

    println(io, "## Regressions since the previous baseline")
    println(io)
    if length(runs) >= 2
        rows = _release_comparison(runs, ordered_groups)
        n_flagged = count(r -> r.flagged, rows)
        n_blind = count(r -> !r.spread, rows)
        println(
            io,
            "Each bar is one benchmark's median in the latest baseline (v$(runs[end].version), `$(runs[end].commit)`) divided by its median in the one before (v$(runs[end - 1].version), `$(runs[end - 1].commit)`); a bar to the left of the dotted line is faster. The spread of a run is its interquartile range, from the first to the third quartile of the samples of that one run. A change is flagged, in red when slower and green when faster, only when the two runs' interquartile ranges do not overlap; a grey bar is within the spread. No fixed percentage band is used, because separate launches of the same code can differ by 10 to 30%, which a fixed band would either hide on a quiet benchmark or flag on a loud one."
        )
        println(io)
        println(
            io,
            "$n_flagged of $(length(rows)) benchmarks are flagged. " *
            (
                n_blind == 0 ? "" :
                "$n_blind have no spread recorded in one of the two runs (baselines saved before the quartiles were recorded keep only the minimum, median and maximum), so they are never flagged. "
            ) * "Hover a bar for the two medians."
        )
        if runs[end].threads != runs[end - 1].threads
            println(io)
            # Documenter admonition (`!!! note` + 4-space body), not GitHub's `> [!NOTE]`
            # alert syntax, which Documenter renders as a blockquote with a literal "[NOTE]".
            println(io, "!!! note \"Thread count differs\"")
            println(
                io,
                "    The previous baseline was recorded with $(_threads_phrase(runs[end - 1].threads)) and the latest with $(_threads_phrase(runs[end].threads)). Entries on the `Parallel()` backend are not comparable across that change: at one thread the threaded code path runs its serial branch."
            )
        end
        println(io)
        println(io, "```@raw html")
        println(io, plotlyjs_head())
        println(io, "```")
        println(io)
        println(io, "```@raw html")
        println(io, "<div style=\"width:100%; margin:1.2rem 0 2.5rem 0;\">")
        println(io, _render_release_chart(rows))
        println(io, "</div>")
        println(io, "```")
    else
        println(
            io,
            "Only one baseline is recorded, so there is no previous run to compare with."
        )
    end
    println(io)

    println(io, "## How to add new benchmark runs")
    println(io)
    println(io, "To record performance on a new commit or after an optimization pass, run:")
    println(io)
    println(io, "```bash")
    println(
        io,
        "julia --project=benchmark benchmark/benchmarks.jl --save benchmark/baselines/baseline_\$(git rev-parse --short HEAD).json"
    )
    println(io, "```")
    println(io)
    println(
        io,
        "Rebuilding the documentation (`julia -e 'using Pkg; Pkg.activate(\"docs\"); include(\"docs/make.jl\")'`) will automatically discover all `baseline_*.json` files and update the comparison with the previous baseline above."
    )

    open(output_path, "w") do f
        return write(f, String(take!(io)))
    end
    return output_path
end
