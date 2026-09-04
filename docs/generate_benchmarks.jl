# Generator for docs/src/benchmarks.md from saved benchmark JSON files.

using BenchmarkTools
using Dates

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

function _nice_axis_max(max_val::Real)
    magnitude = 10.0^floor(log10(max(max_val, 1e-6)))
    norm = max_val / magnitude
    nice_norm = norm <= 1.0 ? 1.0 :
                (norm <= 1.5 ? 1.5 :
                 (norm <= 2.0 ? 2.0 : (norm <= 3.0 ? 3.0 : (norm <= 5.0 ? 5.0 : 10.0))))
    return nice_norm * magnitude
end

function _render_svg_barchart_single(
        gname, sorted_bnames, runs, max_time_ns, unit_label, unit_divisor)
    r = runs[1]
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#f97316"]
    max_val = max_time_ns / unit_divisor
    axis_max = _nice_axis_max(max_val)

    bar_h = 16
    row_gap = 14
    total_row_h = bar_h + row_gap
    label_w = 170
    bar_area_w = 260
    chart_w = label_w + bar_area_w + 90
    top_m = 40
    bottom_m = 30
    total_h = top_m + length(sorted_bnames) * total_row_h + bottom_m

    io = IOBuffer()
    println(io,
        "<div style=\"width:100%; max-width:$(chart_w)px; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:0.8em; box-sizing:border-box;\">")
    println(io,
        "<svg viewBox=\"0 0 $chart_w $total_h\" width=\"100%\" style=\"font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;\">")

    # Header legend
    println(io,
        "<rect x=\"$(label_w + 10)\" y=\"15\" width=\"12\" height=\"12\" rx=\"2\" fill=\"#3b82f6\" />")
    println(io,
        "<text x=\"$(label_w + 28)\" y=\"25\" font-size=\"12\" fill=\"currentColor\" opacity=\"0.9\">$(r.commit) (Julia $(r.julia))</text>")

    # Grid ticks
    for frac in 0.0:0.25:1.0
        gx = label_w + 10 + frac * bar_area_w
        tick_val = round(frac * axis_max; digits = 1)
        println(io,
            "<line x1=\"$gx\" y1=\"$(top_m - 5)\" x2=\"$gx\" y2=\"$(total_h - bottom_m + 5)\" stroke=\"rgba(128,128,128,0.18)\" stroke-dasharray=\"3,3\" />")
        println(io,
            "<text x=\"$gx\" y=\"$(total_h - bottom_m + 18)\" font-size=\"10\" fill=\"currentColor\" opacity=\"0.65\" text-anchor=\"middle\">$tick_val $unit_label</text>")
    end

    y = top_m + 5
    for (idx, bname) in enumerate(sorted_bnames)
        c = palette[mod1(idx, length(palette))]
        mid_y = y + bar_h / 2 + 4
        println(io,
            "<text x=\"$label_w\" y=\"$mid_y\" font-size=\"11\" font-weight=\"bold\" fill=\"currentColor\" text-anchor=\"end\">$bname</text>")
        bx = label_w + 10
        if haskey(r.data, gname) && haskey(r.data[gname], bname)
            m = median(r.data[gname][bname])
            t_val = time(m) / unit_divisor
            bw = max(2.0, (t_val / axis_max) * bar_area_w)
            t_str = string(round(t_val; digits = 1), " ", unit_label)
            println(io,
                "<rect x=\"$bx\" y=\"$y\" width=\"$bw\" height=\"$bar_h\" rx=\"3\" fill=\"$c\" opacity=\"0.9\">")
            println(io, "<title>$(r.commit): $t_str</title></rect>")
            println(io,
                "<text x=\"$(bx + bw + 5)\" y=\"$(y + bar_h - 3)\" font-size=\"11\" fill=\"currentColor\" opacity=\"0.85\">$t_str</text>")
        else
            println(io,
                "<text x=\"$bx\" y=\"$(y + bar_h - 3)\" font-size=\"11\" fill=\"currentColor\" opacity=\"0.4\">—</text>")
        end
        y += total_row_h
    end

    println(io, "</svg></div>")
    return String(take!(io))
end

function _render_svg_trend_chart(
        gname, sorted_bnames, runs, max_time_ns, min_time_ns, unit_label, unit_divisor)
    num_runs = length(runs)
    num_benchmarks = length(sorted_bnames)

    if num_runs == 1
        return _render_svg_barchart_single(
            gname, sorted_bnames, runs, max_time_ns, unit_label, unit_divisor)
    end

    palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#f97316"]

    # If the operations in this group differ by more than 20x (e.g. 150ns vs 1.7ms),
    # use a normalized relative scale (T / T_baseline) so small operations are not flattened.
    use_normalized = (max_time_ns / max(min_time_ns, 1.0)) > 20.0

    chart_w = 540
    pad_l = 65
    pad_r = 25
    pad_t = num_benchmarks > 4 ? 65 : 45
    pad_b = num_runs > 6 ? 60 : 45
    plot_w = chart_w - pad_l - pad_r
    plot_h = 195
    chart_h = pad_t + plot_h + pad_b

    io = IOBuffer()
    println(io,
        "<div style=\"width:100%; max-width:$(chart_w)px; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:0.8em; box-sizing:border-box;\">")
    println(io,
        "<svg viewBox=\"0 0 $chart_w $chart_h\" width=\"100%\" style=\"font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;\">")

    # Legend at top (wrapping if needed)
    leg_x = pad_l
    leg_y = 16
    for (idx, bname) in enumerate(sorted_bnames)
        c = palette[mod1(idx, length(palette))]
        if leg_x > chart_w - 110
            leg_x = pad_l
            leg_y += 18
        end
        println(io,
            "<line x1=\"$leg_x\" y1=\"$leg_y\" x2=\"$(leg_x + 14)\" y2=\"$leg_y\" stroke=\"$c\" stroke-width=\"2.5\" />")
        println(io, "<circle cx=\"$(leg_x + 7)\" cy=\"$leg_y\" r=\"3.5\" fill=\"$c\" />")
        println(io,
            "<text x=\"$(leg_x + 18)\" y=\"$(leg_y + 4)\" font-size=\"11\" font-weight=\"bold\" fill=\"currentColor\">$bname</text>")
        leg_x += length(bname) * 7 + 38
    end

    x_coords = [pad_l + (i - 1) * (plot_w / max(1, num_runs - 1)) for i in 1:num_runs]

    if use_normalized
        # Relative scale: 1.0 is baseline
        # Compute max relative ratio across all points
        max_ratio = 1.25
        min_ratio = 0.75
        for bname in sorted_bnames
            t0 = 0.0
            for (i, r) in enumerate(runs)
                if haskey(r.data, gname) && haskey(r.data[gname], bname)
                    t_ns = time(median(r.data[gname][bname]))
                    if t0 == 0.0
                        t0 = t_ns
                    end
                    ratio = t_ns / t0
                    max_ratio = max(max_ratio, ratio)
                    min_ratio = min(min_ratio, ratio)
                end
            end
        end
        y_max = ceil(max_ratio * 1.15; digits = 2)
        y_min = max(0.0, floor(min_ratio * 0.85; digits = 2))

        # Y-axis ticks & grid lines
        for frac in 0.0:0.25:1.0
            y = pad_t + plot_h - frac * plot_h
            val = round(y_min + frac * (y_max - y_min); digits = 2)
            dash = val == 1.0 ? "stroke-width=\"1.8\"" : "stroke-dasharray=\"3,3\""
            color = val == 1.0 ? "rgba(59, 130, 246, 0.4)" : "rgba(128,128,128,0.18)"
            println(io,
                "<line x1=\"$pad_l\" y1=\"$y\" x2=\"$(pad_l + plot_w)\" y2=\"$y\" stroke=\"$color\" $dash />")
            label_text = val == 1.0 ? "1.0× (ref)" : "$(val)×"
            println(io,
                "<text x=\"$(pad_l - 8)\" y=\"$(y + 4)\" font-size=\"10\" fill=\"currentColor\" opacity=\"0.7\" text-anchor=\"end\">$label_text</text>")
        end

        # X-axis ticks (commits)
        for (i, r) in enumerate(runs)
            x = x_coords[i]
            println(io,
                "<line x1=\"$x\" y1=\"$pad_t\" x2=\"$x\" y2=\"$(pad_t + plot_h)\" stroke=\"rgba(128,128,128,0.15)\" stroke-dasharray=\"2,2\" />")
            commit_str = "`$(r.commit)`"
            if num_runs > 6
                println(io,
                    "<text x=\"$x\" y=\"$(pad_t + plot_h + 18)\" font-size=\"10\" font-family=\"monospace\" fill=\"currentColor\" opacity=\"0.8\" text-anchor=\"end\" transform=\"rotate(-35, $x, $(pad_t + plot_h + 18))\">$commit_str</text>")
            else
                println(io,
                    "<text x=\"$x\" y=\"$(pad_t + plot_h + 20)\" font-size=\"11\" font-family=\"monospace\" fill=\"currentColor\" opacity=\"0.8\" text-anchor=\"middle\">$commit_str</text>")
            end
        end

        # Plot lines and nodes
        for (idx, bname) in enumerate(sorted_bnames)
            c = palette[mod1(idx, length(palette))]
            points = Tuple{Float64, Float64, Int, Float64, Float64, Int, Int}[]
            t0 = 0.0
            for (i, r) in enumerate(runs)
                if haskey(r.data, gname) && haskey(r.data[gname], bname)
                    m = median(r.data[gname][bname])
                    t_ns = time(m)
                    if t0 == 0.0
                        t0 = t_ns
                    end
                    ratio = t_ns / t0
                    x = x_coords[i]
                    y = pad_t + plot_h - ((ratio - y_min) / (y_max - y_min)) * plot_h
                    push!(points, (x, y, i, ratio, t_ns, allocs(m), memory(m)))
                end
            end

            if length(points) >= 2
                pts_str = join(
                    ["$(round(p[1], digits=1)),$(round(p[2], digits=1))" for p in points],
                    " ")
                println(io,
                    "<polyline points=\"$pts_str\" fill=\"none\" stroke=\"$c\" stroke-width=\"2.5\" stroke-linejoin=\"round\" opacity=\"0.88\" />")
            end

            for (p_idx, p) in enumerate(points)
                x, y, r_idx, ratio, t_ns, a_cnt, mem_cnt = p
                r = runs[r_idx]
                t_str = _format_time(t_ns)
                mem_str = _format_memory(mem_cnt)
                pct_vs_base = round((ratio - 1.0) * 100; digits = 1)
                delta_str = pct_vs_base == 0.0 ? "baseline" :
                            (pct_vs_base > 0 ? "+$pct_vs_base%" : "$pct_vs_base%")
                println(io,
                    "<circle cx=\"$(round(x, digits=1))\" cy=\"$(round(y, digits=1))\" r=\"4.5\" fill=\"$c\" stroke=\"var(--documenter-bg, #fff)\" stroke-width=\"1.5\">")
                println(io,
                    "<title>$(r.commit) (Julia $(r.julia))\n$bname: $t_str ($delta_str, $a_cnt allocs, $mem_str)</title></circle>")

                if num_runs <= 6 || p_idx == 1 || p_idx == length(points)
                    println(io,
                        "<text x=\"$(round(x, digits=1))\" y=\"$(round(y - 7, digits=1))\" font-size=\"10\" font-weight=\"bold\" fill=\"$c\" text-anchor=\"middle\">$(round(ratio, digits=2))×</text>")
                end
            end
        end
    else
        # Absolute linear scale
        max_val = max_time_ns / unit_divisor
        axis_max = _nice_axis_max(max_val)

        # Y-axis ticks & grid lines
        for frac in 0.0:0.25:1.0
            y = pad_t + plot_h - frac * plot_h
            val = round(frac * axis_max; digits = 1)
            println(io,
                "<line x1=\"$pad_l\" y1=\"$y\" x2=\"$(pad_l + plot_w)\" y2=\"$y\" stroke=\"rgba(128,128,128,0.18)\" stroke-dasharray=\"3,3\" />")
            println(io,
                "<text x=\"$(pad_l - 8)\" y=\"$(y + 4)\" font-size=\"10\" fill=\"currentColor\" opacity=\"0.65\" text-anchor=\"end\">$val $unit_label</text>")
        end

        # X-axis ticks (commits)
        for (i, r) in enumerate(runs)
            x = x_coords[i]
            println(io,
                "<line x1=\"$x\" y1=\"$pad_t\" x2=\"$x\" y2=\"$(pad_t + plot_h)\" stroke=\"rgba(128,128,128,0.15)\" stroke-dasharray=\"2,2\" />")
            commit_str = "`$(r.commit)`"
            if num_runs > 6
                println(io,
                    "<text x=\"$x\" y=\"$(pad_t + plot_h + 18)\" font-size=\"10\" font-family=\"monospace\" fill=\"currentColor\" opacity=\"0.8\" text-anchor=\"end\" transform=\"rotate(-35, $x, $(pad_t + plot_h + 18))\">$commit_str</text>")
            else
                println(io,
                    "<text x=\"$x\" y=\"$(pad_t + plot_h + 20)\" font-size=\"11\" font-family=\"monospace\" fill=\"currentColor\" opacity=\"0.8\" text-anchor=\"middle\">$commit_str</text>")
            end
        end

        # Plot lines and nodes
        for (idx, bname) in enumerate(sorted_bnames)
            c = palette[mod1(idx, length(palette))]
            points = Tuple{Float64, Float64, Int, Float64, Int, Int}[]
            for (i, r) in enumerate(runs)
                if haskey(r.data, gname) && haskey(r.data[gname], bname)
                    m = median(r.data[gname][bname])
                    t_val = time(m) / unit_divisor
                    x = x_coords[i]
                    y = pad_t + plot_h - (min(t_val, axis_max) / axis_max) * plot_h
                    push!(points, (x, y, i, t_val, allocs(m), memory(m)))
                end
            end

            if length(points) >= 2
                pts_str = join(
                    ["$(round(p[1], digits=1)),$(round(p[2], digits=1))" for p in points],
                    " ")
                println(io,
                    "<polyline points=\"$pts_str\" fill=\"none\" stroke=\"$c\" stroke-width=\"2.5\" stroke-linejoin=\"round\" opacity=\"0.88\" />")
            end

            for (p_idx, p) in enumerate(points)
                x, y, r_idx, t_val, a_cnt, mem_cnt = p
                r = runs[r_idx]
                t_str = string(round(t_val, digits = 1), " ", unit_label)
                mem_str = _format_memory(mem_cnt)
                println(io,
                    "<circle cx=\"$(round(x, digits=1))\" cy=\"$(round(y, digits=1))\" r=\"4.5\" fill=\"$c\" stroke=\"var(--documenter-bg, #fff)\" stroke-width=\"1.5\">")
                println(io,
                    "<title>$(r.commit) (Julia $(r.julia))\n$bname: $t_str ($a_cnt allocs, $mem_str)</title></circle>")

                if num_runs <= 6 || p_idx == 1 || p_idx == length(points)
                    println(io,
                        "<text x=\"$(round(x, digits=1))\" y=\"$(round(y - 7, digits=1))\" font-size=\"10\" font-weight=\"bold\" fill=\"$c\" text-anchor=\"middle\">$(round(t_val, digits=1))</text>")
                end
            end
        end
    end

    println(io, "</svg></div>")
    return String(take!(io))
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
        chart_html = _render_svg_trend_chart(
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
