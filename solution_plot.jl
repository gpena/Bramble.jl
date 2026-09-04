# Solution-field plots via Plotly.js (CDN), no bundler, no assets wiring in make.jl —
# `@example` blocks across the worked examples `include` this rather than each redefining it.
# Mirrors `convergence_plot.jl`'s structure exactly; see `plotly_common.jl` for the shared
# CDN-loading/theming infrastructure.

include(joinpath(@__DIR__, "..", "plotly_common.jl"))

struct SolutionPlot
    html::String
end
Base.show(io::IO, ::MIME"text/html", p::SolutionPlot) = print(io, p.html)

const _SOLUTION_PLOT_COUNTER = Ref(0)
_next_solution_plot_id() = "bramble_sp_$(_SOLUTION_PLOT_COUNTER[] += 1)"

"""
    heatmap_plot(uₕ; title = "", width = 480, height = 420)

A flat top-down colour map of a 2D scalar grid function `uₕ`, in physical mesh coordinates
(not index space).
"""
function heatmap_plot(uₕ; title::AbstractString = "", width::Int = 480, height::Int = 420)
    Ωₕ = mesh(space(uₕ))
    nx, ny = npoints(Ωₕ, Tuple)
    xs = [point(Ωₕ(1), i) for i in 1:nx]
    ys = [point(Ωₕ(2), j) for j in 1:ny]

    # to_matrix(uₕ) is (nx, ny) — Plotly's z wants z[row][col] with row = y, col = x, so
    # transpose rather than reindex by hand.
    M = permutedims(to_matrix(uₕ))

    div_id = _next_solution_plot_id()
    x_js = "[" * join(xs, ",") * "]"
    y_js = "[" * join(ys, ",") * "]"
    z_js = "[" * join(("[" * join(row, ",") * "]" for row in eachrow(M)), ",") * "]"
    title_js = isempty(title) ? "''" : "'$title'"

    html = """
    $(plotlyjs_head())
    <div id="$div_id" style="width:100%; max-width:$(width)px; height:$(height)px; margin: 1em 0;"></div>
    <script>
    (function () {
      const theme = window.bramblePlotlyTheme();
      const data = [{
        type: 'heatmap',
        x: $x_js,
        y: $y_js,
        z: $z_js,
        colorscale: 'Viridis',
        colorbar: { tickfont: { color: theme.text } },
      }];
      const layout = {
        title: { text: $title_js, font: { color: theme.text, size: 14 } },
        paper_bgcolor: theme.bg,
        plot_bgcolor: theme.bg,
        font: { color: theme.text },
        xaxis: { title: 'x', color: theme.text, gridcolor: theme.grid, zeroline: false },
        yaxis: { title: 'y', color: theme.text, gridcolor: theme.grid, zeroline: false, scaleanchor: 'x' },
        margin: { t: 40, l: 50, r: 20, b: 40 },
      };
      Plotly.newPlot('$div_id', data, layout, { displayModeBar: false, responsive: true });
      window.brambleRegisterPlotlyChart('$div_id', function () {
        const t = window.bramblePlotlyTheme();
        return {
          'font.color': t.text,
          'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
          'yaxis.color': t.text, 'yaxis.gridcolor': t.grid,
          'title.font.color': t.text,
        };
      });
    })();
    </script>
    """
    return SolutionPlot(html)
end
