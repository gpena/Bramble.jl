# Shared Plotly.js loading/theming for every chart on the docs site (benchmark trend/bar
# charts, convergence plots, solution surface plots). Included once per page that needs it:
# docs/generate_benchmarks.jl for the benchmark page, docs/src/convergence_plot.jl for the
# worked examples, docs/src/solution_plot.jl for their solution-field plots.
#
# Two problems every Plotly-on-docs page has, solved once here rather than per call site.
#
# A Plotly chart is drawn with JS-supplied colours (paper/plot background, font colour, grid
# lines), which do not track MaterialDocs' dark/light toggle on their own, so a chart drawn
# once in light colours turns unreadable text-on-background after a toggle unless something
# repaints it.
#
# `Plotly.newPlot` sizes a chart from its container's dimensions *at chart-creation time*,
# which runs synchronously as each `<script>` tag executes while the page is still loading,
# before web fonts finish swapping in and before every chart above it has settled the page's
# final layout. Fixed the same way for every chart at once: resize every registered plot
# once, after `window.load` and `document.fonts.ready` have both resolved.

"""
    plotlyjs_head() -> String

The `<script>` tag that loads Plotly.js from a CDN and makes `Plotly`/`bramblePlotlyTheme`/
`brambleRegisterPlotlyChart` available. `@raw html` this once per page, before any plot's
own `<div>` + `Plotly.newPlot(...)` script.
"""
function plotlyjs_head()
    return """
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
      // Colour tokens read from the page's own theme, not hard-coded — MaterialDocs sets
      // `data-theme="light"|"dark"` on <html>; when the attribute is absent the browser's
      // `prefers-color-scheme` decides. Recomputed on every call so a caller can re-theme
      // after a toggle.
      window.bramblePlotlyTheme = function () {
        const attr = document.documentElement.getAttribute('data-theme');
        const dark = attr
          ? attr === 'dark'
          : window.matchMedia('(prefers-color-scheme: dark)').matches;
        return dark
          ? { bg: 'rgba(0,0,0,0)', text: '#c3c2b7', grid: 'rgba(255,255,255,0.12)' }
          : { bg: 'rgba(0,0,0,0)', text: '#52514e', grid: 'rgba(0,0,0,0.10)' };
      };

      // Every plot this page creates registers itself (div id + the function that
      // reapplies theme-dependent layout colours) so one observer can repaint all of them
      // together when the theme toggles, instead of each plot wiring its own observer.
      window.__bramble_plotly_charts = window.__bramble_plotly_charts || [];
      window.brambleRegisterPlotlyChart = function (divId, restyle) {
        window.__bramble_plotly_charts.push({ divId, restyle });
      };

      if (!window.__bramble_plotly_theme_observer) {
        const repaint = function () {
          for (const { divId, restyle } of window.__bramble_plotly_charts) {
            const layout = restyle();
            Plotly.relayout(divId, layout);
          }
        };
        window.__bramble_plotly_theme_observer = new MutationObserver(repaint);
        window.__bramble_plotly_theme_observer.observe(document.documentElement, {
          attributes: true,
          attributeFilter: ['data-theme'],
        });
        // With no saved theme MaterialDocs follows the system scheme in CSS alone and
        // leaves `data-theme` unset, so a system change must repaint too.
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () {
          if (!document.documentElement.hasAttribute('data-theme')) repaint();
        });
      }

      // The layout-race fix (see the module note above): once per page, after everything
      // (fonts included) has truly finished loading, force every plot created so far to
      // resize against its now-final container.
      if (!window.__bramble_plotly_load_fix_installed) {
        window.__bramble_plotly_load_fix_installed = true;
        const rescue = function () {
          for (const { divId } of window.__bramble_plotly_charts) {
            Plotly.Plots.resize(document.getElementById(divId));
          }
        };
        const loaded = document.readyState === 'complete' ? Promise.resolve() :
          new Promise((r) => window.addEventListener('load', r, { once: true }));
        Promise.all([loaded, document.fonts.ready]).then(rescue);
      }
    </script>
    """
end
