# An interactive log-log convergence plot: Chart.js (CDN), no bundler, no assets wiring in
# make.jl. `@example` blocks across the worked examples `include` this rather than each
# redefining it.
#
# Was a dependency-free inline SVG. Moved to Chart.js on request, once a spike confirmed two
# things worth recording: Chart.js's UMD build needs the AMD workaround `chartjs_common.jl`
# applies (Documenter ships RequireJS for MathJax, which the UMD wrapper detects and reacts to
# by registering an anonymous module instead of attaching `window.Chart`), and its
# `type: 'logarithmic'` scale is native — no plugin — and was checked to produce correctly
# straight, correctly-sloped lines for a slope-2/slope-1 pair before this replaced the SVG.
#
# Points only, no connecting line between a series' own markers: with several curves on one
# plot a connecting line adds nothing a reader doesn't already get from the markers being in
# order left to right along a log axis, and it competes visually with the one reference line
# that matters. That one dashed line has slope `reference_slope` (what the scheme promises)
# and is anchored through the finest point of the first series.
#
# Text/grid colours come from `brambleChartTheme()` (`chartjs_common.jl`) rather than a fixed
# palette: a canvas chart's colours are plain JS strings, not CSS `currentColor`, so unlike the
# SVG this replaced, they do not track Documenter's dark/light toggle by themselves — every
# chart registers with `brambleRegisterChart` so the shared theme-change observer can repaint
# it after a toggle instead of leaving it in the wrong contrast until the next reload.

include(joinpath(@__DIR__, "..", "chartjs_common.jl"))

struct ConvergencePlot
    html::String
end
Base.show(io::IO, ::MIME"text/html", p::ConvergencePlot) = print(io, p.html)

const _CONVERGENCE_PLOT_COUNTER = Ref(0)

"""
    convergence_plot(series; title = "", reference_slope = 2)

`series` is a vector of `(hs, errs, label, color)` tuples, one per curve — e.g. one per
spatial dimension.
"""
function convergence_plot(series; title::AbstractString = "", reference_slope::Real = 2,
        width::Int = 480, height::Int = 340)
    _CONVERGENCE_PLOT_COUNTER[] += 1
    chart_id = "bramble_cp_$(_CONVERGENCE_PLOT_COUNTER[])"

    datasets = String[]
    for (hs, errs, label, color) in series
        pts = join(("{x:$(h),y:$(e)}" for (h, e) in zip(hs, errs)), ",")
        push!(datasets, """
            {
              label: "$label",
              data: [$pts],
              showLine: false,
              pointBackgroundColor: "$color",
              pointBorderColor: "$color",
              pointRadius: 5,
              pointHoverRadius: 7,
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
    push!(datasets, """
        {
          label: "$slope_label",
          data: [{x:$hmin,y:$(e_at(hmin))}, {x:$hmax,y:$(e_at(hmax))}],
          showLine: true,
          borderDash: [5,4],
          borderWidth: 1.5,
          pointRadius: 0,
          borderColor: 'rgba(128,128,128,0.7)',
        }""")

    title_js = isempty(title) ? "display: false" : "display: true, text: \"$title\""

    html = """
    $(chartjs_head())
    <div style="width:100%; max-width:$(width)px; margin: 1em 0;">
      <canvas id="$chart_id" width="$width" height="$height"></canvas>
    </div>
    <script>
    (function () {
      const theme = window.brambleChartTheme();
      const ctx = document.getElementById('$chart_id').getContext('2d');
      const chart = new Chart(ctx, {
        type: 'scatter',
        data: { datasets: [$(join(datasets, ",\n"))] },
        options: {
          responsive: true,
          plugins: {
            title: { $title_js, color: theme.text, font: { size: 13, weight: '600' } },
            legend: { position: 'bottom', labels: { color: theme.text, boxWidth: 12, font: { size: 11 } } },
          },
          scales: {
            x: {
              type: 'logarithmic',
              title: { display: true, text: 'h', color: theme.text },
              ticks: { color: theme.text },
              grid: { color: theme.grid },
              border: { color: theme.axis },
            },
            y: {
              type: 'logarithmic',
              title: { display: true, text: 'error', color: theme.text },
              ticks: { color: theme.text },
              grid: { color: theme.grid },
              border: { color: theme.axis },
            },
          },
        },
      });
      window.brambleRegisterChart(chart, function (c) {
        const t = window.brambleChartTheme();
        c.options.plugins.title.color = t.text;
        c.options.plugins.legend.labels.color = t.text;
        for (const ax of ['x', 'y']) {
          c.options.scales[ax].title.color = t.text;
          c.options.scales[ax].ticks.color = t.text;
          c.options.scales[ax].grid.color = t.grid;
          c.options.scales[ax].border.color = t.axis;
        }
      });
    })();
    </script>
    """
    return ConvergencePlot(html)
end
