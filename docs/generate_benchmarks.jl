# Generator for docs/src/benchmarks.md from saved benchmark JSON files.

using BenchmarkTools
using Dates

function _get_commit_info(commit_hash::AbstractString)
    try
        msg = readchomp(`git log -1 --format="%s" $commit_hash`)
        date = readchomp(`git log -1 --format="%cd" --date=short $commit_hash`)
        return (date = date, message = msg)
    catch
        return (date = "unknown", message = "")
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

function _format_delta(t_curr::Real, t_base::Real)
    t_base <= 0 && return ""
    ratio = (t_curr - t_base) / t_base
    pct = round(ratio * 100, digits = 1)
    if abs(pct) < 0.5
        return " (=)"
    elseif pct < 0
        return " (**$(pct)%** 🟢)"
    else
        return " (**+$(pct)%** 🔴)"
    end
end

function _select_unit(max_ns::Real)
    if max_ns < 1_000
        return ("ns", 1.0)
    elseif max_ns < 1_000_000
        return ("μs", 1_000.0)
    else
        return ("ms", 1_000_000.0)
    end
end

function generate_benchmarks_markdown(
        benchmark_dir = normpath(joinpath(@__DIR__, "..", "benchmark")),
        output_path = normpath(joinpath(@__DIR__, "src", "benchmarks.md"))
)
    json_files = String[]
    if isdir(benchmark_dir)
        for f in readdir(benchmark_dir)
            if endswith(f, ".json") && startswith(f, "baseline_")
                push!(json_files, joinpath(benchmark_dir, f))
            end
        end
    end
    sort!(json_files)

    io = IOBuffer()
    println(io, "# Performance & Benchmarks")
    println(io)
    println(io,
        "Bramble tracks memory allocations and performance regressions with a dedicated regression suite in `benchmark/benchmarks.jl`.")
    println(io,
        "All measurements below are run on **1,000,000 grid points** per dimension setup (e.g. ``1000 \\times 1000`` in 2D, ``100 \\times 100 \\times 100`` in 3D).")
    println(io)

    if isempty(json_files)
        println(io, "> [!NOTE]")
        println(io, "> No saved benchmark baselines were found in `benchmark/`.")
        println(io, "> To run and save a baseline locally on AC power:")
        println(io, "> ```bash")
        println(io,
            "> julia --project=benchmark benchmark/benchmarks.jl --save benchmark/baseline_\$(git rev-parse --short HEAD).json")
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
        info = _get_commit_info(commit)
        data = BenchmarkTools.load(path)[1]
        push!(runs, (commit = commit, date = info.date,
            message = info.message, data = data, path = path))
    end

    println(io, "## Recorded Baselines")
    println(io)
    if length(runs) >= 2
        println(io,
            "Comparing **$(length(runs))** recorded baselines. The earliest run (`$(runs[1].commit)`) serves as reference baseline for relative speedup/slowdown calculations.")
        println(io)
    end
    println(io, "| Commit | Date | Summary | File |")
    println(io, "|---|---|---|---|")
    for (idx, r) in enumerate(runs)
        tag = idx == 1 && length(runs) >= 2 ? " *(baseline)*" : ""
        msg = isempty(r.message) ? "Baseline" : r.message
        println(io, "| `$(r.commit)`$tag | $(r.date) | $msg | `$(basename(r.path))` |")
    end
    println(io)

    # Collect all groups dynamically
    group_order = [
        "operators 2D",
        "operators 3D",
        "inner products 2D",
        "restriction",
        "composite",
        "construction"
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

    chart_id = 0
    palette = [
        ("rgba(54, 162, 235, 0.85)", "rgb(54, 162, 235)"),
        ("rgba(75, 192, 192, 0.85)", "rgb(75, 192, 192)"),
        ("rgba(255, 159, 64, 0.85)", "rgb(255, 159, 64)"),
        ("rgba(153, 102, 255, 0.85)", "rgb(153, 102, 255)"),
        ("rgba(255, 99, 132, 0.85)", "rgb(255, 99, 132)"),
        ("rgba(201, 203, 207, 0.85)", "rgb(201, 203, 207)")
    ]

    println(io, "## Comparative Timings & Allocations")
    println(io)

    for gname in ordered_groups
        println(io, "### $(titlecase(gname))")
        println(io)

        # Collect benchmark names for this group
        bnames = Set{String}()
        max_time_ns = 0.0
        for r in runs
            if haskey(r.data, gname)
                for (k, trial) in r.data[gname]
                    push!(bnames, string(k))
                    max_time_ns = max(max_time_ns, time(median(trial)))
                end
            end
        end
        sorted_bnames = sort(collect(bnames))

        # Markdown comparison table
        header = "| Benchmark |"
        sep = "|---|"
        for (idx, r) in enumerate(runs)
            col_name = idx == 1 && length(runs) >= 2 ? "`$(r.commit)` (ref)" :
                       "`$(r.commit)`"
            header *= " $col_name Time | Allocs | Memory |"
            sep *= ":---:|:---:|:---:|"
        end
        println(io, header)
        println(io, sep)

        for bname in sorted_bnames
            row = "| **`$bname`** |"
            t_baseline = 0.0
            if haskey(runs[1].data, gname) && haskey(runs[1].data[gname], bname)
                t_baseline = time(median(runs[1].data[gname][bname]))
            end

            for (idx, r) in enumerate(runs)
                if haskey(r.data, gname) && haskey(r.data[gname], bname)
                    trial = r.data[gname][bname]
                    m = median(trial)
                    t_ns = time(m)
                    t_str = _format_time(t_ns)
                    if idx > 1 && t_baseline > 0
                        t_str *= _format_delta(t_ns, t_baseline)
                    end
                    a_str = string(allocs(m))
                    mem_str = _format_memory(memory(m))
                    row *= " $t_str | $a_str | $mem_str |"
                else
                    row *= " — | — | — |"
                end
            end
            println(io, row)
        end
        println(io)

        # Interactive Chart.js for all groups
        if !isempty(sorted_bnames)
            chart_id += 1
            cid = "benchmark_chart_$chart_id"
            labels_js = "[" * join(["\"$b\"" for b in sorted_bnames], ", ") * "]"
            unit_label, unit_divisor = _select_unit(max_time_ns)

            datasets_js = []
            for (idx, r) in enumerate(runs)
                bg_color, border_color = palette[mod1(idx, length(palette))]
                vals = []
                for bname in sorted_bnames
                    if haskey(r.data, gname) && haskey(r.data[gname], bname)
                        m = median(r.data[gname][bname])
                        push!(vals, string(round(time(m) / unit_divisor, digits = 2)))
                    else
                        push!(vals, "null")
                    end
                end
                vals_str = "[" * join(vals, ", ") * "]"
                push!(datasets_js, """
                {
                    label: '$(r.commit) ($(r.date))',
                    data: $vals_str,
                    backgroundColor: '$bg_color',
                    borderColor: '$border_color',
                    borderWidth: 1
                }
                """)
            end
            all_datasets = "[" * join(datasets_js, ",\n") * "]"

            println(io, "```@raw html")
            if chart_id == 1
                println(io, "<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>")
            end
            println(io,
                """
<div style="width: 100%; max-width: 820px; margin: 1.5em auto; background: var(--documenter-bg, #fff); padding: 1.2em; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
  <canvas id="$cid"></canvas>
</div>
<script>
document.addEventListener("DOMContentLoaded", function() {
    var ctx = document.getElementById('$cid').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: $labels_js,
            datasets: $all_datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: '$(titlecase(gname)) - Median Execution Time ($unit_label)'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.parsed.y === null) return context.dataset.label + ': (not measured)';
                            return context.dataset.label + ': ' + context.parsed.y + ' $unit_label';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time ($unit_label)'
                    }
                }
            }
        }
    });
});
</script>
""")
            println(io, "```")
            println(io)
        end
    end

    println(io, "## How to Add New Benchmark Runs")
    println(io)
    println(io, "To record performance on a new commit or after an optimization pass, run:")
    println(io)
    println(io, "```bash")
    println(io,
        "julia --project=benchmark benchmark/benchmarks.jl --save benchmark/baseline_\$(git rev-parse --short HEAD).json")
    println(io, "```")
    println(io)
    println(io,
        "Rebuilding the documentation (`julia -e 'using Pkg; Pkg.activate(\"docs\"); include(\"docs/make.jl\")'`) will automatically discover all `baseline_*.json` files and append new comparison columns, delta calculations, and chart series.")

    open(output_path, "w") do f
        write(f, String(take!(io)))
    end
    return output_path
end
