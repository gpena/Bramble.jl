# Generator for docs/src/benchmarks.md from saved benchmark JSON files.

using BenchmarkTools
using Dates

function _get_commit_info(commit_hash::AbstractString)
    try
        msg = readchomp(`git log -1 --format="%s" $commit_hash`)
        return (message = msg,)
    catch
        return (message = "",)
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

function _render_svg_barchart(
        gname, sorted_bnames, runs, max_time_ns, unit_label, unit_divisor)
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"]
    max_val = max_time_ns / unit_divisor
    magnitude = 10.0^floor(log10(max(max_val, 1e-6)))
    norm = max_val / magnitude
    nice_norm = norm <= 1.0 ? 1.0 :
                (norm <= 1.5 ? 1.5 :
                 (norm <= 2.0 ? 2.0 : (norm <= 3.0 ? 3.0 : (norm <= 5.0 ? 5.0 : 10.0))))
    axis_max = nice_norm * magnitude

    num_runs = length(runs)
    bar_h = num_runs == 1 ? 18 : 14
    bar_gap = 3
    run_group_h = num_runs * bar_h + (num_runs - 1) * bar_gap
    row_gap = 16
    total_row_h = run_group_h + row_gap

    label_w = 210
    bar_area_w = 380
    chart_w = label_w + bar_area_w + 120
    top_m = 50
    bottom_m = 35
    total_h = top_m + length(sorted_bnames) * total_row_h + bottom_m

    io = IOBuffer()
    println(io,
        "<div style=\"width:100%; max-width:$(chart_w)px; margin:1.5em auto; overflow-x:auto; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:1em;\">")
    println(io,
        "<svg viewBox=\"0 0 $chart_w $total_h\" width=\"100%\" style=\"font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;\">")

    # Legend
    leg_x = label_w + 10
    leg_y = 25
    for (idx, r) in enumerate(runs)
        c = palette[mod1(idx, length(palette))]
        println(io, "<rect x=\"$leg_x\" y=\"$(leg_y - 10)\" width=\"12\" height=\"12\" rx=\"2\" fill=\"$c\" />")
        println(io,
            "<text x=\"$(leg_x + 16)\" y=\"$leg_y\" font-size=\"12\" fill=\"currentColor\" opacity=\"0.9\">$(r.commit) (Julia $(r.julia))</text>")
        leg_x += 180
    end

    # Grid lines & ticks
    for frac in 0.0:0.25:1.0
        gx = label_w + 10 + frac * bar_area_w
        tick_val = round(frac * axis_max; digits = 1)
        println(io,
            "<line x1=\"$gx\" y1=\"$(top_m - 10)\" x2=\"$gx\" y2=\"$(total_h - bottom_m + 5)\" stroke=\"rgba(128,128,128,0.2)\" stroke-dasharray=\"3,3\" />")
        println(io,
            "<text x=\"$gx\" y=\"$(total_h - bottom_m + 20)\" font-size=\"11\" fill=\"currentColor\" opacity=\"0.6\" text-anchor=\"middle\">$tick_val $unit_label</text>")
    end

    # Rows
    y = top_m + 10
    for bname in sorted_bnames
        mid_y = y + run_group_h / 2 + 4
        println(io,
            "<text x=\"$label_w\" y=\"$mid_y\" font-size=\"12\" font-weight=\"bold\" fill=\"currentColor\" text-anchor=\"end\">$bname</text>")
        for (idx, r) in enumerate(runs)
            by = y + (idx - 1) * (bar_h + bar_gap)
            c = palette[mod1(idx, length(palette))]
            if haskey(r.data, gname) && haskey(r.data[gname], bname)
                m = median(r.data[gname][bname])
                t_val = time(m) / unit_divisor
                bw = max(2.0, (t_val / axis_max) * bar_area_w)
                bx = label_w + 10
                t_str = string(round(t_val; digits = 1), " ", unit_label)
                println(io,
                    "<rect x=\"$bx\" y=\"$by\" width=\"$bw\" height=\"$bar_h\" rx=\"3\" fill=\"$c\" opacity=\"0.9\">")
                println(io, "<title>$(r.commit) (Julia $(r.julia)): $t_str</title></rect>")
                println(io,
                    "<text x=\"$(bx + bw + 6)\" y=\"$(by + bar_h - 3)\" font-size=\"11\" fill=\"currentColor\" opacity=\"0.85\">$t_str</text>")
            else
                bx = label_w + 10
                println(io,
                    "<text x=\"$bx\" y=\"$(by + bar_h - 3)\" font-size=\"11\" fill=\"currentColor\" opacity=\"0.4\">—</text>")
            end
        end
        y += total_row_h
    end
    println(io, "</svg></div>")
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
        info = _get_commit_info(commit)
        data = BenchmarkTools.load(path)[1]
        julia_ver = "unknown"
        for t in data.tags
            if startswith(string(t), "julia:")
                julia_ver = replace(string(t), "julia:" => "")
            end
        end
        push!(runs, (commit = commit, message = info.message,
            julia = julia_ver, data = data, path = path))
    end

    println(io, "## Recorded Baselines")
    println(io)
    if length(runs) >= 2
        println(io,
            "Comparing **$(length(runs))** recorded baselines. The earliest run (`$(runs[1].commit)`) serves as reference baseline for relative speedup/slowdown calculations.")
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

        # Pure inline SVG bar chart
        if !isempty(sorted_bnames)
            unit_label, unit_divisor = _select_unit(max_time_ns)
            svg_html = _render_svg_barchart(
                gname, sorted_bnames, runs, max_time_ns, unit_label, unit_divisor)
            println(io, "```@raw html")
            println(io, svg_html)
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
