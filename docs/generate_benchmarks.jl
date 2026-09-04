# Generator for docs/src/benchmarks.md from saved benchmark JSON files.

using BenchmarkTools
using Dates

include(joinpath(@__DIR__, "chartjs_common.jl"))

const _BENCH_CHART_COUNTER = Ref(0)
_next_bench_chart_id() = "bench_chart_$(_BENCH_CHART_COUNTER[] += 1)"

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

function _badge_delta(t_curr::Real, t_ref::Real)
    t_ref <= 0 && return "—"
    ratio = (t_curr - t_ref) / t_ref
    pct = round(ratio * 100, digits = 1)
    if abs(pct) < 0.5
        return "<span style=\"opacity:0.6;\">(=)</span>"
    elseif pct < 0
        return "<span style=\"color:#10b981; font-weight:bold;\">$(pct)% 🟢</span>"
    else
        return "<span style=\"color:#ef4444; font-weight:bold;\">+$(pct)% 🔴</span>"
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

# Single run: a horizontal bar per benchmark. No trend to show, so no head-script emission
# here — the caller (generate_benchmarks_markdown) emits chartjs_head() once for the page.
function _render_chartjs_barchart_single(
        gname, sorted_bnames, runs, max_time_ns, unit_label, unit_divisor)
    r = runs[1]
    chart_id = _next_bench_chart_id()
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
    <div style="width:100%; max-width:520px;">
      <canvas id="$chart_id" height="$height"></canvas>
    </div>
    <script>
    (function () {
      const theme = window.brambleChartTheme();
      const tooltips = [$(join(tooltips, ","))];
      const chart = new Chart(document.getElementById('$chart_id').getContext('2d'), {
        type: 'bar',
        data: {
          labels: [$(join(labels, ","))],
          datasets: [{
            label: "$(r.commit) (Julia $(r.julia))",
            data: [$(join(values, ","))],
            backgroundColor: [$(join(colors, ","))],
            borderRadius: 4,
          }],
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: theme.text } },
            tooltip: { callbacks: { label: (c) => tooltips[c.dataIndex] } },
          },
          scales: {
            x: {
              title: { display: true, text: "$unit_label", color: theme.text },
              ticks: { color: theme.text }, grid: { color: theme.grid }, border: { color: theme.axis },
            },
            y: { ticks: { color: theme.text }, grid: { display: false }, border: { color: theme.axis } },
          },
        },
      });
      window.brambleRegisterChart(chart, function (c) {
        const t = window.brambleChartTheme();
        c.options.plugins.legend.labels.color = t.text;
        c.options.scales.x.title.color = t.text;
        c.options.scales.x.ticks.color = t.text;
        c.options.scales.x.grid.color = t.grid;
        c.options.scales.x.border.color = t.axis;
        c.options.scales.y.ticks.color = t.text;
        c.options.scales.y.border.color = t.axis;
      });
    })();
    </script>
    """
end

function _render_chartjs_trend_chart(
        gname, sorted_bnames, runs, max_time_ns, min_time_ns, unit_label, unit_divisor)
    num_runs = length(runs)

    if num_runs == 1
        return _render_chartjs_barchart_single(
            gname, sorted_bnames, runs, max_time_ns, unit_label, unit_divisor)
    end

    chart_id = _next_bench_chart_id()

    # If the operations in this group differ by more than 20x (e.g. 150ns vs 1.7ms), plot a
    # normalized relative scale (T / T_baseline) instead of absolute time, so small operations
    # are not flattened against a group's largest one.
    use_normalized = (max_time_ns / max(min_time_ns, 1.0)) > 20.0

    labels_js = "[" * join(("\"$(r.commit)\"" for r in runs), ",") * "]"

    datasets = String[]
    for (idx, bname) in enumerate(sorted_bnames)
        color = _BENCH_PALETTE[mod1(idx, length(_BENCH_PALETTE))]
        pts = String[]
        t0 = 0.0
        for r in runs
            if haskey(r.data, gname) && haskey(r.data[gname], bname)
                m = median(r.data[gname][bname])
                t_ns = time(m)
                t0 == 0.0 && (t0 = t_ns)
                y_val = use_normalized ? t_ns / t0 : t_ns / unit_divisor
                delta_str = use_normalized ?
                            "\"$(_format_time(t_ns)) (" *
                            (t_ns == t0 ? "baseline" :
                             (t_ns < t0 ? "-" : "+") *
                             "$(round(abs(t_ns / t0 - 1) * 100, digits = 1))%" ) *
                            ")\"" :
                            "\"$(_format_time(t_ns))\""
                push!(pts, """{x:"$(r.commit)",y:$(y_val),julia:"$(r.julia)",
                    detail:$(delta_str),allocs:$(allocs(m)),mem:"$(_format_memory(memory(m)))"}""")
            else
                push!(pts, "null")
            end
        end
        push!(datasets, """
            {
              label: "$bname",
              data: [$(join(pts, ","))],
              borderColor: "$color",
              backgroundColor: "$color",
              spanGaps: true,
              pointRadius: 4,
              pointHoverRadius: 6,
              borderWidth: 2,
              tension: 0.15,
            }""")
    end

    # The 1.0x reference line in normalized mode, flat across every commit.
    if use_normalized
        ref_pts = join(("{x:\"$(r.commit)\",y:1}" for r in runs), ",")
        push!(datasets, """
            {
              label: "1.0x (ref)",
              data: [$ref_pts],
              borderColor: "rgba(128,128,128,0.7)",
              borderDash: [5,4],
              borderWidth: 1.5,
              pointRadius: 0,
            }""")
    end

    y_title = use_normalized ? "relative to baseline" : unit_label

    return """
    <div style="width:100%; max-width:560px;">
      <canvas id="$chart_id" height="280"></canvas>
    </div>
    <script>
    (function () {
      const theme = window.brambleChartTheme();
      const chart = new Chart(document.getElementById('$chart_id').getContext('2d'), {
        type: 'line',
        data: { labels: $labels_js, datasets: [$(join(datasets, ",\n"))] },
        options: {
          responsive: true,
          interaction: { mode: 'nearest', axis: 'x', intersect: false },
          plugins: {
            legend: { position: 'top', labels: { color: theme.text, boxWidth: 12, font: { size: 11 } } },
            tooltip: {
              callbacks: {
                title: (items) => items[0].raw.x + " (Julia " + items[0].raw.julia + ")",
                label: (c) => c.raw ? c.dataset.label + ": " + (c.raw.detail || c.raw.y) +
                  (c.raw.allocs !== undefined ? " (" + c.raw.allocs + " allocs, " + c.raw.mem + ")" : "") : c.dataset.label,
              },
            },
          },
          scales: {
            x: {
              ticks: { color: theme.text, maxRotation: 45, minRotation: 0, font: { family: 'monospace', size: 10 } },
              grid: { color: theme.grid }, border: { color: theme.axis },
            },
            y: {
              title: { display: true, text: "$y_title", color: theme.text },
              ticks: { color: theme.text }, grid: { color: theme.grid }, border: { color: theme.axis },
            },
          },
        },
      });
      window.brambleRegisterChart(chart, function (c) {
        const t = window.brambleChartTheme();
        c.options.plugins.legend.labels.color = t.text;
        c.options.scales.x.ticks.color = t.text;
        c.options.scales.x.grid.color = t.grid;
        c.options.scales.x.border.color = t.axis;
        c.options.scales.y.title.color = t.text;
        c.options.scales.y.ticks.color = t.text;
        c.options.scales.y.grid.color = t.grid;
        c.options.scales.y.border.color = t.axis;
      });
    })();
    </script>
    """
end

function _render_table_html(gname, sorted_bnames, runs)
    num_runs = length(runs)
    io = IOBuffer()
    println(io,
        "<table style=\"width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;\">")
    println(io, "<thead>")
    println(io, "<tr style=\"border-bottom:2px solid rgba(128,128,128,0.3);\">")
    println(io, "<th style=\"padding:8px 6px; text-align:left;\">Benchmark</th>")

    if num_runs <= 3
        for (idx, r) in enumerate(runs)
            ref_label = idx == 1 && num_runs >= 2 ? " (ref)" : ""
            println(io,
                "<th style=\"padding:8px 6px; text-align:right;\"><code>$(r.commit)</code>$ref_label</th>")
            println(io, "<th style=\"padding:8px 6px; text-align:center;\">Allocs</th>")
        end
        if num_runs >= 2
            println(io,
                "<th style=\"padding:8px 6px; text-align:center;\">Δ vs Base</th>")
        end
        println(io, "<th style=\"padding:8px 6px; text-align:right;\">Memory</th>")
    else
        # Compact summary for 4+ runs (avoids horizontal blowout with 10-20 runs)
        println(io,
            "<th style=\"padding:8px 6px; text-align:right;\">Base (<code>$(runs[1].commit)</code>)</th>")
        println(io,
            "<th style=\"padding:8px 6px; text-align:right;\">Prev (<code>$(runs[end-1].commit)</code>)</th>")
        println(io,
            "<th style=\"padding:8px 6px; text-align:right;\">Latest (<code>$(runs[end].commit)</code>)</th>")
        println(io,
            "<th style=\"padding:8px 6px; text-align:center;\">Δ vs Base</th>")
        println(io,
            "<th style=\"padding:8px 6px; text-align:center;\">Δ vs Prev</th>")
        println(io, "<th style=\"padding:8px 6px; text-align:center;\">Allocs</th>")
        println(io, "<th style=\"padding:8px 6px; text-align:right;\">Memory</th>")
    end
    println(io, "</tr>")
    println(io, "</thead>")
    println(io, "<tbody>")

    for bname in sorted_bnames
        println(io, "<tr style=\"border-bottom:1px solid rgba(128,128,128,0.15);\">")
        println(io, "<td style=\"padding:7px 6px; font-weight:600;\"><code>$bname</code></td>")

        t_base = 0.0
        if haskey(runs[1].data, gname) && haskey(runs[1].data[gname], bname)
            t_base = time(median(runs[1].data[gname][bname]))
        end

        if num_runs <= 3
            latest_mem = "—"
            for (idx, r) in enumerate(runs)
                if haskey(r.data, gname) && haskey(r.data[gname], bname)
                    m = median(r.data[gname][bname])
                    t_str = _format_time(time(m))
                    a_str = string(allocs(m))
                    latest_mem = _format_memory(memory(m))
                    println(io, "<td style=\"padding:7px 6px; text-align:right;\">$t_str</td>")
                    println(io, "<td style=\"padding:7px 6px; text-align:center;\">$a_str</td>")
                else
                    println(io,
                        "<td style=\"padding:7px 6px; text-align:right; opacity:0.4;\">—</td>")
                    println(io,
                        "<td style=\"padding:7px 6px; text-align:center; opacity:0.4;\">—</td>")
                end
            end
            if num_runs >= 2
                if haskey(runs[end].data, gname) && haskey(runs[end].data[gname], bname) &&
                   t_base > 0
                    t_latest = time(median(runs[end].data[gname][bname]))
                    delta_badge = _badge_delta(t_latest, t_base)
                    println(io,
                        "<td style=\"padding:7px 6px; text-align:center;\">$delta_badge</td>")
                else
                    println(io,
                        "<td style=\"padding:7px 6px; text-align:center; opacity:0.4;\">—</td>")
                end
            end
            println(io, "<td style=\"padding:7px 6px; text-align:right;\">$latest_mem</td>")
        else
            # Base
            if haskey(runs[1].data, gname) && haskey(runs[1].data[gname], bname)
                m1 = median(runs[1].data[gname][bname])
                println(io,
                    "<td style=\"padding:7px 6px; text-align:right;\">$(_format_time(time(m1)))</td>")
            else
                println(io,
                    "<td style=\"padding:7px 6px; text-align:right; opacity:0.4;\">—</td>")
            end
            # Prev
            t_prev = 0.0
            if haskey(runs[end - 1].data, gname) && haskey(runs[end - 1].data[gname], bname)
                mp = median(runs[end - 1].data[gname][bname])
                t_prev = time(mp)
                println(io,
                    "<td style=\"padding:7px 6px; text-align:right;\">$(_format_time(t_prev))</td>")
            else
                println(io,
                    "<td style=\"padding:7px 6px; text-align:right; opacity:0.4;\">—</td>")
            end
            # Latest
            latest_allocs = "—"
            latest_mem = "—"
            t_latest = 0.0
            if haskey(runs[end].data, gname) && haskey(runs[end].data[gname], bname)
                ml = median(runs[end].data[gname][bname])
                t_latest = time(ml)
                latest_allocs = string(allocs(ml))
                latest_mem = _format_memory(memory(ml))
                println(io,
                    "<td style=\"padding:7px 6px; text-align:right; font-weight:600;\">$(_format_time(t_latest))</td>")
            else
                println(io,
                    "<td style=\"padding:7px 6px; text-align:right; opacity:0.4;\">—</td>")
            end
            # Delta vs Base
            if t_latest > 0 && t_base > 0
                println(io,
                    "<td style=\"padding:7px 6px; text-align:center;\">$(_badge_delta(t_latest, t_base))</td>")
            else
                println(io,
                    "<td style=\"padding:7px 6px; text-align:center; opacity:0.4;\">—</td>")
            end
            # Delta vs Prev
            if t_latest > 0 && t_prev > 0
                println(io,
                    "<td style=\"padding:7px 6px; text-align:center;\">$(_badge_delta(t_latest, t_prev))</td>")
            else
                println(io,
                    "<td style=\"padding:7px 6px; text-align:center; opacity:0.4;\">—</td>")
            end
            println(io,
                "<td style=\"padding:7px 6px; text-align:center;\">$latest_allocs</td>")
            println(io, "<td style=\"padding:7px 6px; text-align:right;\">$latest_mem</td>")
        end
        println(io, "</tr>")
    end
    println(io, "</tbody>")
    println(io, "</table>")
    return String(take!(io))
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
        "All measurements below are run on **1,000,000 grid points** per dimension setup (e.g. ``1000 \\times 1000`` in 2D, ``100 \\times 100 \\times 100`` in 3D).")
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
        for t in data.tags
            if startswith(string(t), "julia:")
                julia_ver = replace(string(t), "julia:" => "")
            end
        end
        push!(runs,
            (commit = commit, message = info.message, time = info.time,
                julia = julia_ver, data = data, path = path))
    end
    # Order runs chronologically by commit timestamp
    sort!(runs, by = r -> r.time)

    println(io, "## Recorded baselines")
    println(io)
    if length(runs) >= 2
        println(io,
            "Comparing **$(length(runs))** recorded baselines in chronological order. The earliest run (`$(runs[1].commit)`) is the reference baseline for relative speedup/slowdown calculations.")
        println(io)
    end
    println(io, "| Commit | Julia | Summary | File |")
    println(io, "|---|:---:|---|---|")
    for (idx, r) in enumerate(runs)
        tag = idx == 1 && length(runs) >= 2 ? " *(baseline)*" : ""
        msg = isempty(r.message) ? "Baseline" : r.message
        println(io, "| `$(r.commit)`$tag | `$(r.julia)` | $msg | `$(basename(r.path))` |")
    end
    println(io)

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

    # Loaded once for the whole page — every chart below reuses window.Chart and the shared
    # theme/registration helpers (chartjs_common.jl) rather than each re-loading the CDN
    # script.
    println(io, "```@raw html")
    println(io, chartjs_head())
    println(io, "```")
    println(io)

    for gname in ordered_groups
        println(io, "### $(titlecase(gname))")
        println(io)

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

        unit_label, unit_divisor = _select_unit(max_time_ns)
        table_html = _render_table_html(gname, sorted_bnames, runs)
        chart_html = _render_chartjs_trend_chart(
            gname, sorted_bnames, runs, max_time_ns, min_time_ns, unit_label, unit_divisor)

        # Side-by-side flex layout
        println(io, "```@raw html")
        println(io,
            "<div style=\"display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;\">")
        println(io, "  <div style=\"flex:1 1 430px; min-width:320px; overflow-x:auto;\">")
        println(io, table_html)
        println(io, "  </div>")
        println(io, "  <div style=\"flex:1 1 450px; min-width:340px;\">")
        println(io, chart_html)
        println(io, "  </div>")
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
