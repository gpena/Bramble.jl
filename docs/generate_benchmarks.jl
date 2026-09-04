# Generator for docs/src/benchmarks.md from saved benchmark JSON files.

using BenchmarkTools
using Dates

include(joinpath(@__DIR__, "plotly_common.jl"))

const _BENCH_CHART_COUNTER = Ref(0)
_next_bench_div_id() = "bench_chart_$(_BENCH_CHART_COUNTER[] += 1)"

function _get_commit_info(commit_hash::AbstractString, path::AbstractString)
    try
        msg = readchomp(pipeline(`git log -1 --format="%s" $commit_hash`, stderr = devnull))
        ct = parse(Int,
            readchomp(pipeline(`git log -1 --format="%ct" $commit_hash`, stderr = devnull)))
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
        toml = readchomp(pipeline(`git show $(commit_hash):Project.toml`, stderr = devnull))
        m = match(r"^version\s*=\s*\"([^\"]+)\""m, toml)
        return m !== nothing ? m.captures[1] : "unknown"
    catch
        return "unknown"
    end
end

function _format_time(t_ns::Real)
    if t_ns < 1_000
        return string(round(t_ns, digits = 1), " ns")
    elseif t_ns < 1_000_000
        return string(round(t_ns / 1_000, digits = 1), " μs")
    elseif t_ns < 1_000_000_000
        return string(round(t_ns / 1_000_000, digits = 2), " ms")
    else
        return string(round(t_ns / 1_000_000_000, digits = 2), " s")
    end
end

function _format_memory(b::Real)
    if b == 0
        return "0 B"
    elseif b < 1024
        return string(round(Int, b), " B")
    elseif b < 1024^2
        return string(round(b / 1024, digits = 1), " KiB")
    elseif b < 1024^3
        return string(round(b / (1024^2), digits = 2), " MiB")
    else
        return string(round(b / (1024^3), digits = 2), " GiB")
    end
end

function _select_unit(max_ns::Real)
    if max_ns < 1_000
        return ("ns", 1.0)
    elseif max_ns < 1_000_000
        return ("μs", 1_000.0)
    elseif max_ns < 1_000_000_000
        return ("ms", 1_000_000.0)
    else
        return ("s", 1_000_000_000.0)
    end
end

const _BENCH_PALETTE = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4",
    "#f97316"]

# Some groups have a natural axis their series cluster along — a dimension, a numeric
# precision — and read better split along that axis than at an arbitrary midpoint. Listed
# in the order charts should appear.
const _BENCH_GROUP_SPLIT_TAGS = Dict(
    "restriction" => ["1D", "2D", "3D"],
    "forms" => ["1D", "2D", "3D"],
    "precision 1D" => ["Float32", "Float64", "Double64"],
)

_midpoint_clusters(bnames) = (mid = cld(length(bnames), 2); [bnames[1:mid], bnames[(mid + 1):end]])

# Bucket `bnames` by whichever configured tag each one contains, in tag order, dropping
# empty buckets. Falls back to a plain midpoint split for a group with no configured tags,
# or if some name doesn't match any of them (so a future rename can't silently vanish a
# series into nothing rendered at all).
function _bench_group_clusters(gname, bnames)
    tags = get(_BENCH_GROUP_SPLIT_TAGS, gname, nothing)
    tags === nothing && return _midpoint_clusters(bnames)
    clusters = [String[] for _ in tags]
    for bname in bnames
        idx = findfirst(t -> occursin(t, bname), tags)
        idx === nothing && return _midpoint_clusters(bnames)
        push!(clusters[idx], bname)
    end
    return [c for c in clusters if !isempty(c)]
end

# The package version is what a reader tracks progress against release to release; the
# commit is what pins the measurement exactly, since several baselines can share a
# version between releases. Show both, version first — except on the trend chart's own
# x-axis (`_run_xlabel`), which stays version-only: the axis is read as a release
# timeline, not a commit log, and the exact commit is one hover away (`_run_label`
# supplies it to every point's tooltip via `customdata`).
_run_label(r) = "v$(r.version) ($(r.commit))"
_run_xlabel(r) = "v$(r.version)"

# One short sentence per group, mined from this suite's own "## Why these six" rationale
# in benchmark/benchmarks.jl — what a reader is actually looking at, not just how to read
# the chart (that part is explained once, generically, above the whole section). A group
# not listed here (a future addition to the suite) still gets a plain, data-derived line
# instead of silently rendering with no introduction at all.
const _BENCH_GROUP_BLURBS = Dict(
    "operators 2D" => "The finite-difference stencil engine on a 1000×1000 grid: the difference operator along the grid's contiguous storage direction (`D₋ₓ`) versus across it (`D₋ᵧ`), which access memory very differently and so can perform very differently.",
    "operators 3D" => "The same stencil engine in 3D (`D₋₂`), together with the inner product `innerₕ` and the full gradient `∇₋ₕ`.",
    "jumps & averages" => "Jump and average operators across cell interfaces, in 2D and 3D.",
    "inner products 2D" => "The reduction path — inner products and norms — including the seminorm's sum over directions.",
    "restriction" => "Point interpolation (`Rₕ!`) and cell-averaging (`avgₕ!`), compared across the `Serial()` (the allocation-free default) and `Parallel()` backends, split by dimension.",
    "composite" => "A composite (multi-component) operator, which dispatches per component and calls the engine once per component with a view rather than once with a plain vector.",
    "construction" => "Mesh and grid-space construction, including the quadrature weights `gridspace` builds internally.",
    "startup & latency" => "Time to first `using Bramble` and first operator call — compilation latency, not steady-state performance.",
    "forms" => "Linear and bilinear form assembly, across 1D/2D and the `Serial()`/`Parallel()` backends.",
    "precision 1D" => "The same 1D workload — restriction, assembly, inner product — repeated in `Float32`, `Float64`, and `Double64`, split by precision since `Double64` (software arithmetic) is an order of magnitude slower.",
)

_bench_group_blurb(gname, n_series, n_runs) = get(_BENCH_GROUP_BLURBS, gname,
    "$n_series benchmark$(n_series == 1 ? "" : "s") in this group, across $n_runs recorded releases.")

# Single run: a horizontal bar per benchmark. No trend to show, so no head-script emission
# here — the caller (generate_benchmarks_markdown) emits plotlyjs_head() once for the page.
function _render_plotly_barchart_single(
        gname, sorted_bnames, runs, max_time_ns, unit_label, unit_divisor)
    r = runs[1]
    div_id = _next_bench_div_id()
    height = 60 + length(sorted_bnames) * 34

    labels = String[]
    values = Float64[]
    colors = String[]
    tooltips = String[]
    for (idx, bname) in enumerate(sorted_bnames)
        push!(labels, "\"$bname\"")
        push!(colors, "\"$(_BENCH_PALETTE[mod1(idx, length(_BENCH_PALETTE))])\"")
        if haskey(r.data, gname) && haskey(r.data[gname], bname)
            m = median(r.data[gname][bname])
            t_val = time(m) / unit_divisor
            push!(values, t_val)
            push!(tooltips, "\"$(_format_time(time(m)))\"")
        else
            push!(values, 0.0)
            push!(tooltips, "\"—\"")
        end
    end

    return """
    <div id="$div_id" style="width:100%; height:$(height)px;"></div>
    <script>
    (function () {
      const theme = window.bramblePlotlyTheme();
      const data = [{
        type: 'bar',
        orientation: 'h',
        name: "$(_run_label(r)) (Julia $(r.julia))",
        y: [$(join(labels, ","))],
        x: [$(join(values, ","))],
        marker: { color: [$(join(colors, ","))] },
        hovertext: [$(join(tooltips, ","))],
        hoverinfo: 'text',
      }];
      const layout = {
        paper_bgcolor: theme.bg,
        plot_bgcolor: theme.bg,
        font: { color: theme.text },
        showlegend: true,
        legend: { font: { color: theme.text } },
        xaxis: {
          title: { text: "$unit_label", font: { color: theme.text } },
          color: theme.text, gridcolor: theme.grid,
        },
        yaxis: { color: theme.text, autorange: 'reversed' },
        margin: { t: 30, l: 10, r: 20, b: 40 },
      };
      Plotly.newPlot('$div_id', data, layout, { displayModeBar: false, responsive: true });
      window.brambleRegisterPlotlyChart('$div_id', function () {
        const t = window.bramblePlotlyTheme();
        return {
          'font.color': t.text,
          'legend.font.color': t.text,
          'xaxis.color': t.text, 'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.text,
          'yaxis.color': t.text,
        };
      });
    })();
    </script>
    """
end

# One plot's worth of a trend chart, for a fixed subset of a group's benchmark names. Split
# out of `_render_trend_chart` so a group with more series than the palette has colors for
# (see there) can render as two of these side by side, each restarting the palette from its
# own beginning rather than cycling into a color the other chart already used.
function _render_one_trend_plot(
        gname, bnames_subset, runs, use_normalized, unit_label, unit_divisor)
    div_id = _next_bench_div_id()
    all_labels_js = "[" * join(("\"$(_run_xlabel(r))\"" for r in runs), ",") * "]"

    traces = String[]
    for (idx, bname) in enumerate(bnames_subset)
        color = _BENCH_PALETTE[mod1(idx, length(_BENCH_PALETTE))]
        xs, ys, customdata = String[], String[], String[]
        t0 = 0.0
        for r in runs
            if haskey(r.data, gname) && haskey(r.data[gname], bname)
                m = median(r.data[gname][bname])
                t_ns = time(m)
                t0 == 0.0 && (t0 = t_ns)
                y_val = use_normalized ? t_ns / t0 : t_ns / unit_divisor
                delta_str = use_normalized ?
                            "$(_format_time(t_ns)) (" *
                            (t_ns == t0 ? "baseline" :
                             (t_ns < t0 ? "-" : "+") *
                             "$(round(abs(t_ns / t0 - 1) * 100, digits = 1))%") *
                            ")" :
                            _format_time(t_ns)
                push!(xs, "\"$(_run_xlabel(r))\"")
                push!(ys, "$y_val")
                push!(customdata,
                    """["$(r.julia)","$delta_str",$(allocs(m)),"$(_format_memory(memory(m)))"]""")
            end
            # A run missing this benchmark contributes no point at all, rather than a `null`
            # placeholder: each point already carries its own `x`, so a category axis needs
            # no placeholder to stay aligned, and a leading `null` (the commit before a
            # benchmark existed, e.g. the very first run for a group added later) is exactly
            # the kind of gap a category axis with an explicit `categoryarray` handles by
            # skipping straight to the next real point instead of breaking alignment.
        end
        push!(traces, """
            {
              name: "$bname",
              x: [$(join(xs, ","))],
              y: [$(join(ys, ","))],
              customdata: [$(join(customdata, ","))],
              mode: 'lines+markers',
              type: 'scatter',
              line: { color: "$color", width: 2, shape: 'spline', smoothing: 0.3 },
              marker: { color: "$color", size: 7 },
              hovertemplate: '%{x} (Julia %{customdata[0]})<br>$bname: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
            }""")
    end

    # The 1.0x reference line in normalized mode, flat across every commit.
    if use_normalized
        ref_xs = join(("\"$(_run_xlabel(r))\"" for r in runs), ",")
        ref_ys = join(("1" for _ in runs), ",")
        push!(traces, """
            {
              name: "1.0x (ref)",
              x: [$ref_xs],
              y: [$ref_ys],
              mode: 'lines',
              type: 'scatter',
              line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
              hoverinfo: 'skip',
            }""")
    end

    y_title = use_normalized ? "relative to baseline" : unit_label

    return """
    <div id="$div_id" style="width:100%; height:300px;"></div>
    <script>
    (function () {
      const theme = window.bramblePlotlyTheme();
      const data = [$(join(traces, ",\n"))];
      const layout = {
        paper_bgcolor: theme.bg,
        plot_bgcolor: theme.bg,
        font: { color: theme.text },
        legend: {
          orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
          font: { color: theme.text, size: 11 },
        },
        xaxis: {
          type: 'category', categoryorder: 'array', categoryarray: $all_labels_js,
          tickangle: -45, color: theme.text, gridcolor: theme.grid,
          tickfont: { family: 'monospace', size: 10 },
        },
        yaxis: {
          title: { text: "$y_title", font: { color: theme.text } },
          color: theme.text, gridcolor: theme.grid,
        },
        margin: { t: 20, l: 60, r: 160, b: 60 },
      };
      Plotly.newPlot('$div_id', data, layout, { displayModeBar: false, responsive: true });
      window.brambleRegisterPlotlyChart('$div_id', function () {
        const t = window.bramblePlotlyTheme();
        return {
          'font.color': t.text,
          'legend.font.color': t.text,
          'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
          'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
        };
      });
    })();
    </script>
    """
end

function _render_trend_chart(
        gname, sorted_bnames, runs, max_time_ns, min_time_ns, unit_label, unit_divisor)
    num_runs = length(runs)

    if num_runs == 1
        return _render_plotly_barchart_single(
            gname, sorted_bnames, runs, max_time_ns, unit_label, unit_divisor)
    end

    # If the operations in this group differ by more than 20x (e.g. 150ns vs 1.7ms), plot a
    # normalized relative scale (T / T_baseline) instead of absolute time, so small operations
    # are not flattened against a group's largest one.
    use_normalized = (max_time_ns / max(min_time_ns, 1.0)) > 20.0

    # Beyond one palette's worth of series (7), `_BENCH_PALETTE` repeats colors and two
    # unrelated lines become visually indistinguishable — "restriction" (9 benchmarks),
    # "forms" (13) and "precision 1D" (12) all hit this. Split into separate charts instead
    # of cycling, clustered along whichever axis the group's names carry (dimension,
    # precision — see `_BENCH_GROUP_SPLIT_TAGS`) rather than an arbitrary midpoint; each
    # keeps its own distinct palette rather than inheriting where the previous chart left
    # off. Stacked one per row, each at the full page width, rather than side by side —
    # squeezed into a fraction of the row, a legend of 4-5 series plus rotated version
    # labels on the x-axis has no room to lay out cleanly.
    if length(sorted_bnames) > length(_BENCH_PALETTE)
        clusters = _bench_group_clusters(gname, sorted_bnames)
        panels = [_render_one_trend_plot(gname, names, runs, use_normalized, unit_label,
                      unit_divisor)
                  for names in clusters]
        divs = join(("<div style=\"width:100%;\">$p</div>" for p in panels))
        return """
        <div style="display:flex; flex-direction:column; gap:1.5rem; width:100%;">
          $divs
        </div>
        """
    end

    return _render_one_trend_plot(
        gname, sorted_bnames, runs, use_normalized, unit_label, unit_divisor)
end

function generate_benchmarks_markdown(
        benchmark_dir = normpath(joinpath(@__DIR__, "..", "benchmark", "baselines")),
        output_path = normpath(joinpath(@__DIR__, "src", "benchmarks.md"))
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
    println(io,
        "Bramble tracks memory allocations and performance regressions with a dedicated regression suite in `benchmark/benchmarks.jl`.")
    println(io,
        "All measurements below are run on **1,000,000 grid points** per dimension setup (e.g. \$1000 \\times 1000\$ in 2D, \$100 \\times 100 \\times 100\$ in 3D).")
    println(io)

    if isempty(json_files)
        println(io, "> [!NOTE]")
        println(io, "> No saved benchmark baselines were found in `benchmark/baselines/`.")
        println(io, "> To run and save a baseline locally on AC power:")
        println(io, "> ```bash")
        println(io,
            "> julia --project=benchmark benchmark/benchmarks.jl --save benchmark/baselines/baseline_\$(git rev-parse --short HEAD).json")
        println(io, "> ```")
        open(output_path, "w") do f
            write(f, String(take!(io)))
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
        for t in data.tags
            if startswith(string(t), "julia:")
                julia_ver = replace(string(t), "julia:" => "")
            elseif startswith(string(t), "pkgversion:")
                pkg_ver = replace(string(t), "pkgversion:" => "")
            end
        end
        # Baselines saved before the `pkgversion:` tag existed carry none — retrace it
        # from Project.toml at that commit instead of leaving it blank.
        pkg_ver === nothing && (pkg_ver = _get_pkg_version(commit))
        push!(runs,
            (commit = commit, message = info.message, time = info.time,
                julia = julia_ver, version = pkg_ver, data = data, path = path))
    end
    # Order runs chronologically by commit timestamp
    sort!(runs, by = r -> r.time)

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

    println(io, "## Comparative timings and allocations")
    println(io)
    if length(runs) >= 2
        println(io,
            "Each chart below tracks one benchmark group across all **$(length(runs))** recorded baselines, in chronological release order, against the earliest run (v$(runs[1].version)) as the reference. Where a group's operations span more than a 20× range, the y-axis shows time relative to that reference instead of absolute time, so a cheap operation isn't flattened onto the same line as an expensive one. Hover any point for its exact time, Julia version, allocation count, and memory.")
    else
        println(io,
            "Each chart below shows one benchmark group's timings and allocations for the single recorded baseline. Hover a bar for its exact time.")
    end
    println(io)

    # Loaded once for the whole page — every chart below reuses window.Plotly and the shared
    # theme/registration helpers (plotly_common.jl) rather than each re-loading the CDN
    # script.
    println(io, "```@raw html")
    println(io, plotlyjs_head())
    println(io, "```")
    println(io)

    for gname in ordered_groups
        bnames = Set{String}()
        max_time_ns = 0.0
        min_time_ns = Inf
        for r in runs
            if haskey(r.data, gname)
                for (k, trial) in r.data[gname]
                    push!(bnames, string(k))
                    t_ns = time(median(trial))
                    max_time_ns = max(max_time_ns, t_ns)
                    min_time_ns = min(min_time_ns, t_ns)
                end
            end
        end
        sorted_bnames = sort(collect(bnames))
        isempty(sorted_bnames) && continue

        # Display only — "&" reads better as "and" in a heading, but the underlying group
        # key (benchmark/benchmarks.jl's `SUITE["jumps & averages"]`) stays as-is: it is also
        # the key every saved baseline_*.json carries, and renaming it would fragment that
        # group's trend history across old and new baselines instead. Sentence case
        # (`uppercasefirst`, not `titlecase`) so a multi-word group name reads as a heading
        # rather than a title — "Jumps and averages", not "Jumps And Averages".
        println(io, "### $(uppercasefirst(replace(gname, "&" => "and")))")
        println(io)
        println(io, _bench_group_blurb(gname, length(sorted_bnames), length(runs)))
        println(io)

        unit_label, unit_divisor = _select_unit(max_time_ns)
        chart_html = _render_trend_chart(
            gname, sorted_bnames, runs, max_time_ns, min_time_ns, unit_label, unit_divisor)

        println(io, "```@raw html")
        println(io, "<div style=\"width:100%; margin:1.2rem 0 2.5rem 0;\">")
        println(io, chart_html)
        println(io, "</div>")
        println(io, "```")
        println(io)
    end

    println(io, "## How to add new benchmark runs")
    println(io)
    println(io, "To record performance on a new commit or after an optimization pass, run:")
    println(io)
    println(io, "```bash")
    println(io,
        "julia --project=benchmark benchmark/benchmarks.jl --save benchmark/baselines/baseline_\$(git rev-parse --short HEAD).json")
    println(io, "```")
    println(io)
    println(io,
        "Rebuilding the documentation (`julia -e 'using Pkg; Pkg.activate(\"docs\"); include(\"docs/make.jl\")'`) will automatically discover all `baseline_*.json` files and append new comparison columns, delta calculations, and charts.")

    open(output_path, "w") do f
        write(f, String(take!(io)))
    end
    return output_path
end
