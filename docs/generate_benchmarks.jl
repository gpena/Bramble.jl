# Generator for docs/src/benchmarks.md from saved benchmark JSON files.

using BenchmarkTools
using Dates
using TOML

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

# One short sentence per group, saying what the group measures; the comparison sections
# below list the ones their charts draw on. A group not listed (a later addition to the
# suite) gets no blurb rather than a made-up one.
const _BENCH_GROUP_BLURBS = Dict(
    "operators 2D" => "The finite-difference stencil engine on a 1000×1000 grid: the difference operator along the grid's contiguous storage direction (`D₋ₓ`) versus across it (`D₋ᵧ`), which access memory very differently and so can perform very differently.",
    "operators 3D" => "The same stencil engine in 3D (`D₋₂`), together with the inner product `innerₕ` and the full gradient `∇ₕ`.",
    "jumps & averages" => "Jump and average operators across cell interfaces, in 2D and 3D.",
    "inner products 2D" => "The reduction path — inner products and norms — including the seminorm's sum over directions.",
    "restriction" => "Point interpolation (`Rₕ!`) and cell-averaging (`avgₕ!`), compared across the `Serial()` (the allocation-free default), `Parallel()` and `CpuPolyester()` backends, split by dimension.",
    "composite" => "A composite (multi-component) operator, which dispatches per component and calls the engine once per component with a view rather than once with a plain vector.",
    "construction" => "Mesh and grid-space construction, including the quadrature weights `gridspace` builds internally.",
    "startup & latency" => "Time to first `using Bramble` and first operator call — compilation latency, not steady-state performance.",
    "forms" => "Linear and bilinear form assembly, across 1D/2D and the `Serial()`/`Parallel()`/`CpuPolyester()` backends, and the refill and `assemble_add!` paths on a fixed matrix pattern.",
    "precision 1D" => "The same 1D workload — restriction, assembly, inner product — repeated in `Float32`, `Float64`, and `Double64`; `Double64` (software arithmetic) is an order of magnitude slower."
)

# The palette of the comparison charts: mid-saturation hues that read on the light and the
# dark site theme alike.
const _BENCH_PALETTE = [
    "#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16"
]

# A JavaScript string literal.
_js_str(s) = "\"" * replace(string(s), "\\" => "\\\\", "\"" => "\\\"") * "\""
_js_num(x) = x === nothing ? "null" : string(x)

# The median time in ns of every benchmark of `run`, keyed by (group, benchmark), and each
# group's `points:<n>` tag (the number of grid points its benchmarks run on).
function _latest_medians(run)
    times = Dict{Tuple{String, String}, Float64}()
    points = Dict{String, Int}()
    for (g, grp) in run.data
        for t in grp.tags
            startswith(string(t), "points:") &&
                (points[string(g)] = parse(Int, string(t)[(length("points:") + 1):end]))
        end
        for (k, trial) in grp.data
            times[(string(g), string(k))] = time(median(trial))
        end
    end
    return (times = times, points = points)
end

# A comparison chart: grouped bars (`mode = "bar"`) or lines with markers (`"line"`) of
# `series = [(name, values, hover)]` over the categories `labels`; `nothing` is a gap. The
# `data-bench` and `data-run` attributes are read by the checks in
# `.claude/plans/v3-21-0-benchmarks-page-checks/`.
function _render_comparison_chart(
        kind, run, labels, series, ytitle; mode = "bar", ref = nothing, logy = true,
        height = max(380, 120 + 22 * length(labels) * (mode == "bar" ? 1 : 0)),
        horizontal = false
)
    div_id = _next_bench_div_id()
    traces = String[]
    for (i, (name, values, hover)) in enumerate(series)
        color = _BENCH_PALETTE[mod1(i, length(_BENCH_PALETTE))]
        vals = "[" * join((_js_num(v) for v in values), ",") * "]"
        cats = "[" * join((_js_str(l) for l in labels), ",") * "]"
        hov = "[" * join((_js_str(h) for h in hover), ",") * "]"
        xs, ys = horizontal ? (vals, cats) : (cats, vals)
        extra = if mode == "bar"
            "type: 'bar', $(horizontal ? "orientation: 'h', " : "")marker: { color: '$color' }"
        else
            "type: 'scatter', mode: 'lines+markers', connectgaps: true, line: { color: '$color' }, marker: { color: '$color', size: 7 }"
        end
        push!(
            traces,
            "{ name: $(_js_str(name)), x: $xs, y: $ys, $extra, hovertext: $hov, hoverinfo: 'text' }"
        )
    end
    shapes = ref === nothing ? "[]" :
             "[{ type: 'line', $(horizontal ? "xref: 'x', yref: 'paper', x0: $ref, x1: $ref, y0: 0, y1: 1" : "xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: $ref, y1: $ref"), line: { color: theme.text, width: 1, dash: 'dot' } }]"
    valaxis = horizontal ? "xaxis" : "yaxis"
    cataxis = horizontal ? "yaxis" : "xaxis"
    return """
    <div id="$div_id" data-bench="$kind" data-run="$(run.commit)" style="width:100%; height:$(height)px;"></div>
    <script>
    (function () {
      const theme = window.bramblePlotlyTheme();
      const data = [$(join(traces, ",\n"))];
      const layout = {
        paper_bgcolor: theme.bg,
        plot_bgcolor: theme.bg,
        font: { color: theme.text },
        barmode: 'group',
        legend: { orientation: 'h', y: -0.25 },
        shapes: $shapes,
        $valaxis: {
          title: { text: $(_js_str(ytitle)), font: { color: theme.text } },
          $(logy ? "type: 'log', " : "")color: theme.text, gridcolor: theme.grid,
        },
        $cataxis: { color: theme.text$(horizontal ? ", autorange: 'reversed', tickfont: { size: 10 }" : ", tickangle: -30") },
        margin: { t: 20, l: $(horizontal ? 260 : 70), r: 20, b: $(horizontal ? 60 : 130) },
      };
      Plotly.newPlot('$div_id', data, layout, { displayModeBar: false, responsive: true });
      window.brambleRegisterPlotlyChart('$div_id', function () {
        const t = window.bramblePlotlyTheme();
        return {
          'font.color': t.text,
          '$valaxis.color': t.text, '$valaxis.gridcolor': t.grid, '$valaxis.title.font.color': t.text,
          '$cataxis.color': t.text,
        };
      });
    })();
    </script>
    """
end

# Wraps a chart for Documenter.
_raw_chart(chart) = "```@raw html\n<div style=\"width:100%; margin:1.2rem 0 2.5rem 0;\">\n$chart</div>\n```"

# The groups a chart draws on, each with its blurb.
function _group_lines(groups)
    lines = String[]
    for g in groups
        b = get(_BENCH_GROUP_BLURBS, g, "")
        push!(lines, isempty(b) ? "- `$g`" : "- `$g`: $b")
    end
    return join(lines, "\n")
end

_ms(ns) = ns / 1.0e6
_ratio_hover(label, ns, ratio, what) = "$label: $(_format_time(ns)), ×$(round(ratio; digits = 2)) $what"

# Each comparison below finds its pairs by name in the latest run alone, and returns
# `nothing` when the run lacks the keys.

# Speedup of `Parallel()` and `CpuPolyester()` over `Serial()` per operator and dimension.
function _policy_section(run, med, ordered_groups)
    rx = r"^(.*), (Serial|Parallel|CpuPolyester)\(\) backend( \(default\))?( \(non-uniform\))?$"
    labels, rows, groups = String[], Dict{String, Float64}[], String[]
    for g in ordered_groups
        fam = Dict{String, Dict{String, Float64}}()
        order = String[]
        for k in sort(collect(k for (gg, k) in keys(med.times) if gg == g))
            m = match(rx, k)
            m === nothing && continue
            base = m.captures[1] * (m.captures[4] === nothing ? "" : " (non-uniform)")
            haskey(fam, base) || (fam[base] = Dict{String, Float64}(); push!(order, base))
            fam[base][m.captures[2]] = med.times[(g, k)]
        end
        for base in order
            d = fam[base]
            (haskey(d, "Serial") && length(d) > 1) || continue
            push!(labels, base)
            push!(rows, d)
            g in groups || push!(groups, g)
        end
    end
    isempty(rows) && return nothing
    series = Tuple{String, Vector{Any}, Vector{String}}[]
    for pol in ("Parallel", "CpuPolyester")
        any(haskey(d, pol) for d in rows) || continue
        vals = Any[haskey(d, pol) ? d["Serial"] / d[pol] : nothing for d in rows]
        hov = [haskey(d, pol) ?
               _ratio_hover(l, d[pol], d["Serial"] / d[pol], "faster than Serial()") : l
               for (l, d) in zip(labels, rows)]
        push!(series, ("$pol()", vals, hov))
    end
    chart = _render_comparison_chart(
        "policy", run, labels, series, "speedup over Serial()"; ref = 1)
    text = """
    ### Execution policy

    Each bar is the median of the `Serial()` run of a benchmark divided by its median under `Parallel()` or, where the run has it, `CpuPolyester()`; a bar above the dotted line is faster than `Serial()`. The benchmarks are paired by name from the latest run alone.

    $(_group_lines(groups))

    $(_raw_chart(chart))
    """
    return text
end

# Cost of `Float32` and `Double64` relative to `Float64`.
function _precision_section(run, med, ordered_groups)
    rx = r"^(.*) (Float32|Float64|Double64)$"
    labels, rows, groups, types = String[], Dict{String, Float64}[], String[], String[]
    for g in ordered_groups
        fam = Dict{String, Dict{String, Float64}}()
        order = String[]
        for k in sort(collect(k for (gg, k) in keys(med.times) if gg == g))
            m = match(rx, k)
            m === nothing && continue
            base = m.captures[1]
            haskey(fam, base) || (fam[base] = Dict{String, Float64}(); push!(order, base))
            fam[base][m.captures[2]] = med.times[(g, k)]
        end
        for base in order
            d = fam[base]
            (haskey(d, "Float64") && length(d) > 1) || continue
            push!(labels, base)
            push!(rows, d)
            g in groups || push!(groups, g)
        end
    end
    isempty(rows) && return nothing
    series = Tuple{String, Vector{Any}, Vector{String}}[]
    for ty in ("Float32", "Double64")
        any(haskey(d, ty) for d in rows) || continue
        vals = Any[haskey(d, ty) ? d[ty] / d["Float64"] : nothing for d in rows]
        hov = [haskey(d, ty) ?
               _ratio_hover(l, d[ty], d[ty] / d["Float64"], "the Float64 time") : l
               for (l, d) in zip(labels, rows)]
        push!(series, (ty, vals, hov))
    end
    chart = _render_comparison_chart(
        "precision", run, labels, series, "median time relative to Float64"; ref = 1)
    return """
    ### Precision

    Each bar is the median of a benchmark in `Float32` or `Double64` divided by its median in `Float64`; a bar below the dotted line is cheaper than `Float64`.

    $(_group_lines(groups))

    $(_raw_chart(chart))
    """
end

# Time of a pair `(a, b)` of benchmarks of one group, as two bars per pair.
function _pair_section(
        kind, run, med, ordered_groups, pairs_of, title, caption, name_a, name_b)
    labels, ta, tb, groups = String[], Float64[], Float64[], String[]
    for g in ordered_groups
        for (label, ka, kb) in pairs_of(g)
            push!(labels, label)
            push!(ta, med.times[(g, ka)])
            push!(tb, med.times[(g, kb)])
            g in groups || push!(groups, g)
        end
    end
    isempty(labels) && return nothing
    series = [
        (name_a, Any[_ms(t) for t in ta], [string(l, ": ", _format_time(t)) for (l, t) in zip(labels, ta)]),
        (name_b, Any[_ms(t) for t in tb], [string(l, ": ", _format_time(t)) for (l, t) in zip(labels, tb)])
    ]
    chart = _render_comparison_chart(
        kind, run, labels, series, "median time (ms)"; horizontal = true,
        height = 140 + 46 * length(labels))
    return """
    ### $title

    $caption

    $(_group_lines(groups))

    $(_raw_chart(chart))
    """
end

function _direction_section(run, med, ordered_groups)
    pairs_of = g -> [(replace(k, "ₓ" => "ₓ|ᵧ"), k, replace(k, "ₓ" => "ᵧ"))
                     for k in sort(collect(k for (gg, k) in keys(med.times) if gg == g))
                     if occursin("ₓ", k) && haskey(med.times, (g, replace(k, "ₓ" => "ᵧ")))]
    return _pair_section(
        "direction", run, med, ordered_groups, pairs_of, "Direction",
        "The same operator along `x` (the contiguous storage direction) and along `y` (across it), paired by swapping `ₓ` for `ᵧ` in the benchmark name.",
        "along x", "along y")
end

function _uniformity_section(run, med, ordered_groups)
    sfx = " (non-uniform)"
    pairs_of = g -> [(chopsuffix(k, sfx), chopsuffix(k, sfx), k)
                     for k in sort(collect(k for (gg, k) in keys(med.times) if gg == g))
                     if endswith(k, sfx) && haskey(med.times, (g, chopsuffix(k, sfx)))]
    return _pair_section(
        "uniformity", run, med, ordered_groups, pairs_of, "Uniform and non-uniform grids",
        "Each benchmark on a uniform mesh and on the same mesh with graded axes (the benchmark of the same name with a ` (non-uniform)` suffix).",
        "uniform", "non-uniform")
end

# First assembly, refill and `assemble_add!` of one matrix.
function _assembly_section(run, med, ordered_groups)
    g = "forms"
    first_key = "assemble (BilinearForm) 2D, Serial() backend"
    keys_ = [
        (first_key, "first assembly (allocates and fills)"),
        ("assemble! (matrix) 2D", "refill (pattern reused)"),
        ("assemble-then-add (matrix) 2D", "assemble the pieces, then add"),
        ("assemble_add! (matrix) 2D", "assemble_add!")
    ]
    present = [(k, l) for (k, l) in keys_ if haskey(med.times, (g, k))]
    length(present) < 2 && return nothing
    labels = [l for (_, l) in present]
    ts = [med.times[(g, k)] for (k, _) in present]
    series = [(
        "median time", Any[_ms(t) for t in ts],
        [string(l, ": ", _format_time(t)) for (l, t) in zip(labels, ts)]
    )]
    chart = _render_comparison_chart(
        "assembly", run, labels, series, "median time (ms)"; horizontal = true,
        height = 140 + 46 * length(labels))
    return """
    ### Assembly

    A bilinear form assembled for the first time, refilled in place, and combined with a second piece. The last two bars are a pair of their own: both build `M/Δt + θK` from a mass-like and a stiffness-like piece, which is a different form from the first two, so they are compared with each other and not with the bars above.

    $(_group_lines([g]))

    $(_raw_chart(chart))
    """
end

# Nanoseconds per grid point across dimensions, for the benchmarks of a group tagged with
# `points:<n>` whose names differ only in a `1D`, `2D` or `3D` token.
function _dimension_section(run, med, ordered_groups)
    rx = r"(?<= )([123])D(?=[ ,]|$)"
    labels = ["1D", "2D", "3D"]
    series = Tuple{String, Vector{Any}, Vector{String}}[]
    groups = String[]
    for g in ordered_groups
        haskey(med.points, g) || continue
        fam = Dict{String, Dict{String, Float64}}()
        order = String[]
        for k in sort(collect(k for (gg, k) in keys(med.times) if gg == g))
            m = match(rx, k)
            m === nothing && continue
            base = replace(k, rx => "nD")
            haskey(fam, base) || (fam[base] = Dict{String, Float64}(); push!(order, base))
            fam[base][m.captures[1] * "D"] = med.times[(g, k)] / med.points[g]
        end
        for base in order
            d = fam[base]
            length(d) > 1 || continue
            vals = Any[get(d, l, nothing) for l in labels]
            hov = [haskey(d, l) ? "$base, $l: $(round(d[l]; sigdigits = 3)) ns per point" : l
                   for l in labels]
            push!(series, (base, vals, hov))
            g in groups || push!(groups, g)
        end
    end
    isempty(series) && return nothing
    chart = _render_comparison_chart(
        "dimension", run, labels, series, "median time per grid point (ns)";
        mode = "line", height = 460)
    return """
    ### Dimension

    The median time per grid point, in nanoseconds per point, of benchmarks that differ only in the dimension of their name. The time is divided by the `points:` tag of the group, the number of grid points its benchmarks run on.

    $(_group_lines(groups))

    $(_raw_chart(chart))
    """
end

function _comparison_sections(runs, ordered_groups)
    run = runs[end]
    med = _latest_medians(run)
    sections = [f(run, med, ordered_groups)
                for f in (
        _policy_section, _precision_section, _direction_section,
        _assembly_section, _uniformity_section, _dimension_section
    )]
    return filter(!isnothing, sections)
end

# ---- Standalone results ---------------------------------------------------------------
# The standalone scripts save their tables to `benchmark/results/<script>.toml`
# (`benchmark/results_io.jl`): a `[meta]` table of the run and `[[tables.<name>]]` arrays of
# rows. Every figure on the page is read from those files; none is typed here. A row omits a
# field whose value was missing (TOML has no null), so the readers below take `get`.

# Lines with markers of `series = [(name, xs, ys, hover, color, dash)]` on log axes. The
# `data-bench` and `data-run` attributes are read by the checks in
# `.claude/plans/v3-21-0-benchmarks-page-checks/`.
function _render_xy_chart(kind, meta, series, xtitle, ytitle; height = 420)
    div_id = _next_bench_div_id()
    traces = String[]
    arr(v) = "[" * join((_js_num(x) for x in v), ",") * "]"
    for (name, xs, ys, hover, color, dash) in series
        hov = "[" * join((_js_str(h) for h in hover), ",") * "]"
        push!(
            traces,
            "{ type: 'scatter', mode: 'lines+markers', name: $(_js_str(name)), x: $(arr(xs)), y: $(arr(ys)), line: { color: '$color', dash: '$dash' }, marker: { color: '$color', size: 6 }, hovertext: $hov, hoverinfo: 'text' }"
        )
    end
    return """
    <div id="$div_id" data-bench="$kind" data-run="$(meta["commit"])" style="width:100%; height:$(height)px;"></div>
    <script>
    (function () {
      const theme = window.bramblePlotlyTheme();
      const data = [$(join(traces, ",\n"))];
      const layout = {
        paper_bgcolor: theme.bg,
        plot_bgcolor: theme.bg,
        font: { color: theme.text },
        legend: { orientation: 'h', y: -0.25 },
        xaxis: {
          title: { text: $(_js_str(xtitle)), font: { color: theme.text } },
          type: 'log', color: theme.text, gridcolor: theme.grid,
        },
        yaxis: {
          title: { text: $(_js_str(ytitle)), font: { color: theme.text } },
          type: 'log', color: theme.text, gridcolor: theme.grid,
        },
        margin: { t: 20, l: 70, r: 20, b: 110 },
      };
      Plotly.newPlot('$div_id', data, layout, { displayModeBar: false, responsive: true });
      window.brambleRegisterPlotlyChart('$div_id', function () {
        const t = window.bramblePlotlyTheme();
        return {
          'font.color': t.text,
          'xaxis.color': t.text, 'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.text,
          'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
        };
      });
    })();
    </script>
    """
end

# What a run states about itself, from its `[meta]`. `load1` is read when the run ends, so it
# includes the run itself; the scripts refuse to start unless the machine is quiet.
function _run_statement(meta)
    power = meta["power"] == "ac" ? "AC" : string(meta["power"])
    return "Run on $(meta["cpu"]) with $(_threads_phrase(string(meta["threads"]))), power: $power, commit `$(meta["commit"])`, Julia $(meta["julia"]), $(meta["date"]). The script waited for a quiet machine before starting; the load average at the end of the run, which includes the run itself, was $(meta["load1"])."
end

_standalone_fig(meta, caption, chart) = "$caption\n\n$(_run_statement(meta))\n\n$(_raw_chart(chart))"

# The name and colour of a route in the `operator_routes` charts.
const _ROUTE_STYLE = Dict(
    "assembled" => ("assembled", "#3b82f6"),
    "kronecker" => ("Kronecker", "#f59e0b"),
    "matrix_free_serial" => ("matrix-free, serial", "#10b981"),
    "matrix_free_threaded" => ("matrix-free, CpuThreaded()", "#ef4444"),
    "fdm_solve" => ("fdm_solve (reference)", "#8b5cf6"),
    "direct" => ("direct (reference)", "#9ca3af")
)
const _DIM_DASH = ["solid", "dash", "dot"]
const _XYSeries = Tuple{String, Vector{Any}, Vector{Any}, Vector{String}, String, String}

# One series per route and dimension of `rows`, `yfn(row)` against `ndofs`, sorted by
# `ndofs`; `hoverfn(row)` is the text of a point.
function _route_series(rows, routes, yfn, hoverfn)
    series = _XYSeries[]
    dims = sort(unique(r["dim"] for r in rows))
    for route in routes
        label, color = get(_ROUTE_STYLE, route, (route, _BENCH_PALETTE[1]))
        for (i, d) in enumerate(dims)
            pts = sort(
                [r for r in rows if r["route"] == route && r["dim"] == d];
                by = r -> r["ndofs"])
            isempty(pts) && continue
            push!(
                series,
                (
                    "$label, $(d)D", Any[r["ndofs"] for r in pts], Any[yfn(r) for r in pts],
                    [hoverfn(r) for r in pts], color, _DIM_DASH[mod1(i, 3)]
                )
            )
        end
    end
    return series
end

function _route_hover(r, what)
    "$(get(_ROUTE_STYLE, r["route"], (r["route"], ""))[1]), $(r["dim"])D, n = $(r["n"]) ($(r["ndofs"]) degrees of freedom): $what"
end

function _operator_routes_section(meta, tables)
    parts = String[]
    routes = ["assembled", "kronecker", "matrix_free_serial", "matrix_free_threaded"]
    if haskey(tables, "construction")
        rows = tables["construction"]
        series = _route_series(
            rows, routes, r -> r["time_s"] * 1.0e3,
            r -> _route_hover(r, "$(_format_time(r["time_s"] * 1.0e9)), $(Base.format_bytes(r["bytes_alloc"])) allocated"))
        chart = _render_xy_chart(
            "standalone-operator_routes-construction", meta, series,
            "degrees of freedom", "construction time (ms)")
        push!(
            parts,
            "#### Construction\n\n" * _standalone_fig(
                meta,
                "Time to build the operator of the separable form `innerₕ(u,v) + inner₊(∇ₕu,∇ₕv)` on a non-uniform mesh, by route. Matrix-free construction takes microseconds or less because the operator's plan is built at construction and no matrix is formed; the assembled and Kronecker routes build their matrices here.",
                chart))
    end
    if haskey(tables, "product")
        rows = tables["product"]
        series = _route_series(
            rows, routes, r -> r["time_s"] * 1.0e3,
            r -> _route_hover(r, "$(_format_time(r["time_s"] * 1.0e9)) per product, $(Base.format_bytes(r["bytes_held"])) held"))
        chart = _render_xy_chart(
            "standalone-operator_routes-product", meta, series,
            "degrees of freedom", "time of one product (ms)")
        bseries = _route_series(
            rows, routes, r -> r["bytes_held"],
            r -> _route_hover(r, "$(Base.format_bytes(r["bytes_held"])) held"))
        bchart = _render_xy_chart(
            "standalone-operator_routes-product-bytes", meta, bseries,
            "degrees of freedom", "bytes held by the operator")
        push!(
            parts,
            "#### Product\n\n" *
            _standalone_fig(
                meta,
                "Time of one operator-vector product, then the bytes the operator holds. The assembled route's bytes count its matrix only, not the scatter cache the assembled form keeps for refilling it.",
                chart) * "\n\n" * _raw_chart(bchart))
    end
    if haskey(tables, "solve")
        rows = tables["solve"]
        series = _route_series(
            rows, [routes; "fdm_solve"; "direct"], r -> r["time_s"] * 1.0e3,
            r -> _route_hover(r,
                "$(_format_time(r["time_s"] * 1.0e9)), $(r["iterations"]) CG iterations, true relative residual $(round(r["rel_residual"]; sigdigits = 3))"))
        chart = _render_xy_chart(
            "standalone-operator_routes-solve", meta, series,
            "degrees of freedom", "solve time (ms)")
        cg = [r for r in rows if r["route"] in routes]
        spread, residual = 0.0, cg[argmax([r["rel_residual"] for r in cg])]
        for key in unique((r["dim"], r["n"]) for r in cg)
            its = [r["iterations"] for r in cg if (r["dim"], r["n"]) == key]
            spread = max(spread, (maximum(its) - minimum(its)) / minimum(its))
        end
        push!(
            parts,
            "#### Solve\n\n" * _standalone_fig(
                meta,
                "Time of the solve of the same problem by conjugate gradients on each route, with `fdm_solve` and the sparse direct solve as reference curves. The CG iteration counts agree across routes to within $(round(100 * spread; digits = 2))% at every size. CG stops when its recursively updated residual reaches the tolerance, which is not the true residual: the largest true relative residual among the CG rows is $(round(residual["rel_residual"]; sigdigits = 2)), at $(residual["dim"])D with n = $(residual["n"]).",
                chart))
    end
    return join(parts, "\n\n")
end

function _matrix_free_spmv_section(meta, tables)
    haskey(tables, "spmv") || return ""
    rows = tables["spmv"]
    dims = sort(unique(r["dim"] for r in rows))
    pol_style = Dict(
        "serial" => ("matrix-free, serial", "#10b981"),
        "threaded" => ("matrix-free, CpuThreaded()", "#ef4444"))
    tseries, mseries = _XYSeries[], _XYSeries[]
    for (i, d) in enumerate(dims)
        dash = _DIM_DASH[mod1(i, 3)]
        for pol in ("serial", "threaded")
            pts = sort(
                [r for r in rows if r["dim"] == d && r["policy"] == pol]; by = r -> r["ndofs"])
            isempty(pts) && continue
            label, color = pol_style[pol]
            xs = Any[r["ndofs"] for r in pts]
            push!(
                tseries,
                (
                    "$label, $(d)D", xs, Any[r["mf_s"] * 1.0e3 for r in pts],
                    ["$label, $(d)D, $(r["ndofs"]) degrees of freedom: $(_format_time(r["mf_s"] * 1.0e9)), ×$(round(r["ratio"]; digits = 2)) the SpMV speed"
                     for r in pts],
                    color, dash
                ))
            push!(
                mseries,
                (
                    "$label, $(d)D", xs, Any[r["mf_bytes"] for r in pts],
                    ["$label, $(d)D, $(r["ndofs"]) degrees of freedom: $(Base.format_bytes(r["mf_bytes"])) held"
                     for r in pts],
                    color, dash
                ))
            pol == "serial" || continue
            push!(
                tseries,
                (
                    "SpMV, $(d)D", xs, Any[r["spmv_s"] * 1.0e3 for r in pts],
                    ["SpMV, $(d)D, $(r["ndofs"]) degrees of freedom: $(_format_time(r["spmv_s"] * 1.0e9))" for r in pts],
                    "#3b82f6", dash
                ))
            push!(
                mseries,
                (
                    "assembled matrix, $(d)D", xs, Any[r["csr_bytes"] for r in pts],
                    ["assembled matrix, $(d)D, $(r["ndofs"]) degrees of freedom: $(Base.format_bytes(r["csr_bytes"])) held"
                     for r in pts],
                    "#3b82f6", dash
                ))
        end
    end
    tchart = _render_xy_chart(
        "standalone-matrix_free_spmv-time", meta, tseries,
        "degrees of freedom", "time of one product (ms)")
    mchart = _render_xy_chart(
        "standalone-matrix_free_spmv-memory", meta, mseries,
        "degrees of freedom", "bytes held")
    cross = String[]
    for r in get(tables, "crossover", [])
        where_ = haskey(r, "from_ndofs") ?
                 "from $(r["from_ndofs"]) degrees of freedom, where the assembled matrix holds $(round(r["memory_ratio"]; digits = 1)) times the bytes of the matrix-free operator" :
                 "no size in the range"
        push!(cross, "- $(r["dim"])D, $(r["policy"]): $where_")
    end
    note = isempty(cross) ? "" :
           "\n\nThe smallest size from which matrix-free beats the SpMV at every larger size tested:\n\n" *
           join(cross, "\n")
    return "#### Time\n\n" *
           _standalone_fig(
               meta,
               "One product `mul!(y, matrix_free_operator(a), x)` against one serial sparse matrix-vector product with the assembled matrix, on non-uniform meshes in 1D, 2D and 3D, by degrees of freedom.",
               tchart) * "\n\n#### Memory\n\n" *
           _standalone_fig(
               meta,
               "Bytes the assembled matrix holds against the bytes everything the matrix-free operator keeps alive holds, including the form it captures.$note",
               mchart)
end

function _policy_crossover_section(meta, tables)
    haskey(tables, "crossover") || return ""
    rows = tables["crossover"]
    labels = ["$(r["workload"]) $(r["dim"])D" for r in rows]
    keys_ = [
        ("threads", "CpuThreaded() beats serial"),
        ("polyester", "CpuPolyester() beats serial"),
        ("polyester_vs_threads", "CpuPolyester() beats CpuThreaded()")
    ]
    series = [(
                  name, Any[get(r, k, nothing) for r in rows],
                  [haskey(r, k) ? "$l: $name from $(r[k]) degrees of freedom" :
                   "$l: no crossover in the sweep" for (r, l) in zip(rows, labels)]
              ) for (k, name) in keys_ if any(haskey(r, k) for r in rows)]
    chart = _render_comparison_chart(
        "standalone-policy_crossover", (commit = meta["commit"],), labels, series,
        "degrees of freedom of the crossover"; height = 520)
    return _standalone_fig(
        meta,
        "The smallest size, in degrees of freedom, from which each host policy beats the one it is compared with, per workload and dimension, on a non-uniform mesh; a win counts only when the next larger size wins too. A missing bar means the sweep found no crossover, and the crossover depends on the thread count of the run.",
        chart)
end

function _gpu_offload_section(meta, tables)
    haskey(tables, "offload") || return ""
    rows = tables["offload"]
    labels = ["$(r["workload"]), $(r["grid"])" for r in rows]
    series = [(
        "threaded host time / offload time", Any[r["ratio"] for r in rows],
        ["$l: $(_format_time(r["cpu_threaded_ms"] * 1.0e6)) threaded, $(_format_time(r["gpu_offload_ms"] * 1.0e6)) offloaded, ×$(round(r["ratio"]; digits = 2))"
         for (r, l) in zip(rows, labels)]
    )]
    chart = _render_comparison_chart(
        "standalone-gpu_offload", (commit = meta["commit"],), labels, series,
        "threaded host time / offload time"; ref = 1, logy = false, height = 380)
    return _standalone_fig(
        meta,
        "`CpuThreaded()` against `GpuOffload(metal_backend(), CpuThreaded())` on the same grid; a bar above the dotted line means the offload was faster. The host arm runs in `Float64` and the offload arm in `Float32`, which Metal requires.",
        chart)
end

const _STANDALONE_SECTIONS = [
    ("operator_routes", "Operator routes", _operator_routes_section),
    ("matrix_free_spmv", "Matrix-free against SpMV", _matrix_free_spmv_section),
    ("policy_crossover", "Execution-policy crossover", _policy_crossover_section),
    ("gpu_offload", "Host against device offload", _gpu_offload_section)
]

# The text of the `## Standalone benchmarks` sections for the files of `results_dir`; a smoke
# file or a file of a script the page does not draw is skipped with a note.
function _standalone_sections(results_dir)
    files = isdir(results_dir) ? sort(filter(endswith(".toml"), readdir(results_dir))) :
            String[]
    out = String[]
    found = Dict{String, Any}()
    for f in files
        data = TOML.parsefile(joinpath(results_dir, f))
        script = first(splitext(f))
        meta = get(data, "meta", nothing)
        if meta === nothing || !haskey(meta, "commit")
            push!(out, "!!! note \"Skipped\"\n    `$f` has no `[meta]` table of a run, so it is skipped.")
        elseif get(meta, "smoke", false)
            push!(out,
                "!!! note \"Skipped\"\n    `$f` is a smoke run, which checks a script's structure and measures nothing, so it is skipped.")
        elseif !any(s -> s[1] == script, _STANDALONE_SECTIONS)
            push!(out, "!!! note \"Skipped\"\n    `$f` is the result of a script this page does not draw, so it is skipped.")
        else
            found[script] = (meta, get(data, "tables", Dict{String, Any}()))
        end
    end
    for (script, title, f) in _STANDALONE_SECTIONS
        haskey(found, script) || continue
        meta, tables = found[script]
        body = f(meta, tables)
        isempty(body) && continue
        push!(out, "### $title (`$script.jl`)\n\n$body")
    end
    return out
end

# `results_dir` is where the standalone scripts' saved tables live.
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

    println(io, "```@raw html")
    println(io, plotlyjs_head())
    println(io, "```")
    println(io)
    println(io, "## Comparisons in the latest baseline")
    println(io)
    println(
        io,
        "Every chart in this section reads the latest baseline alone (v$(runs[end].version), `$(runs[end].commit)`), recorded with $(_threads_phrase(runs[end].threads)), and finds its pairs by benchmark name; a comparison whose benchmarks the baseline lacks is left out."
    )
    println(io)
    for section in _comparison_sections(runs, ordered_groups)
        println(io, section)
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

    println(io, "## Standalone benchmarks")
    println(io)
    println(
        io,
        "The scripts in `benchmark/` that compare alternatives outside the regression suite save their tables to `benchmark/results/<script>.toml` with `--save`; every chart below is drawn from those files, and a new full run replaces the file. Each states the machine, thread count, power and commit of its run."
    )
    println(io)
    for section in _standalone_sections(results_dir)
        println(io, section)
        println(io)
    end

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
