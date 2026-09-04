# An interactive log-log convergence plot: Plotly.js (CDN), no bundler, no assets wiring in
# make.jl. `@example` blocks across the worked examples `include` this rather than each
# redefining it.
#
# Points only, no connecting line between a series' own markers: with several curves on one
# plot a connecting line adds nothing a reader doesn't already get from the markers being in
# order left to right along a log axis, and it competes visually with the one reference line
# that matters. That one dashed line has slope `reference_slope` (what the scheme promises)
# and is anchored through the finest point of the first series. It renders as a straight line
# on Plotly's log axes for the same reason it did on a logarithmic axis anywhere else: the log
# transform is applied to the axis, not the data, so a true power-law relationship in the
# underlying (h, error) coordinates always plots straight regardless of which library draws it.
#
# Text/grid colours come from `bramblePlotlyTheme()` (`plotly_common.jl`) rather than a fixed
# palette: a chart's colours are plain JS strings, not CSS `currentColor`, so unlike a plain
# SVG they do not track Documenter's dark/light toggle by themselves — every chart registers
# with `brambleRegisterPlotlyChart` so the shared theme-change observer can repaint it after a
# toggle instead of leaving it in the wrong contrast until the next reload.

include(joinpath(@__DIR__, "..", "plotly_common.jl"))

struct ConvergencePlot
    html::String
end
Base.show(io::IO, ::MIME"text/html", p::ConvergencePlot) = print(io, p.html)

const _CONVERGENCE_PLOT_COUNTER = Ref(0)
_next_convergence_plot_id() = "bramble_cp_$(_CONVERGENCE_PLOT_COUNTER[] += 1)"

"""
    convergence_plot(series; title = "", reference_slope = 2)

`series` is a vector of `(hs, errs, label, color)` tuples, one per curve — e.g. one per
spatial dimension.
"""
function convergence_plot(series; title::AbstractString = "", reference_slope::Real = 2,
        width::Int = 480, height::Int = 340)
    div_id = _next_convergence_plot_id()

    traces = String[]
    for (hs, errs, label, color) in series
        xs = "[" * join(hs, ",") * "]"
        ys = "[" * join(errs, ",") * "]"
        push!(traces, """
            {
              name: "$label",
              x: $xs,
              y: $ys,
              mode: 'markers',
              type: 'scatter',
              marker: { color: "$color", size: 8 },
            }""")
    end

    # The reference line, anchored through the first series' finest (last) point — the number
    # printed is the claim; the markers either sit on it or not.
    hs1, errs1 = series[1][1], series[1][2]
    all_hs = reduce(vcat, (s[1] for s in series))
    hmin, hmax = extrema(all_hs)
    h0, e0 = hs1[end], errs1[end]
    e_at(h) = e0 * (h / h0)^reference_slope
    slope_label = "slope $(isinteger(reference_slope) ? Int(reference_slope) : reference_slope)"
    push!(traces, """
        {
          name: "$slope_label",
          x: [$hmin, $hmax],
          y: [$(e_at(hmin)), $(e_at(hmax))],
          mode: 'lines',
          type: 'scatter',
          line: { dash: 'dash', width: 1.5, color: 'rgba(128,128,128,0.7)' },
        }""")

    title_js = isempty(title) ? "''" : "'$title'"

    html = """
    $(plotlyjs_head())
    <div id="$div_id" style="width:100%; max-width:$(width)px; height:$(height)px; margin: 1em 0;"></div>
    <script>
    (function () {
      const theme = window.bramblePlotlyTheme();
      const data = [$(join(traces, ",\n"))];
      const layout = {
        title: { text: $title_js, font: { color: theme.text, size: 13 } },
        paper_bgcolor: theme.bg,
        plot_bgcolor: theme.bg,
        font: { color: theme.text },
        legend: { orientation: 'h', y: -0.25, font: { color: theme.text, size: 11 } },
        xaxis: {
          type: 'log', title: { text: 'h', font: { color: theme.text } },
          color: theme.text, gridcolor: theme.grid,
        },
        yaxis: {
          type: 'log', title: { text: 'error', font: { color: theme.text } },
          color: theme.text, gridcolor: theme.grid,
        },
        margin: { t: 40, l: 60, r: 20, b: 40 },
      };
      Plotly.newPlot('$div_id', data, layout, { displayModeBar: false, responsive: true });
      window.brambleRegisterPlotlyChart('$div_id', function () {
        const t = window.bramblePlotlyTheme();
        return {
          'font.color': t.text,
          'title.font.color': t.text,
          'legend.font.color': t.text,
          'xaxis.color': t.text, 'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.text,
          'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
        };
      });
    })();
    </script>
    """
    return ConvergencePlot(html)
end
