# Shared Plotly.js loading/theming for every chart on the docs site (benchmark trend/bar
# charts, convergence plots, solution heatmaps). Included once per page that needs it —
# docs/generate_benchmarks.jl for the benchmark page, docs/src/convergence_plot.jl for the
# worked examples, docs/src/solution_plot.jl for their solution-field plots.
#
# Three problems every Plotly-on-Documenter page has, solved once here rather than per call
# site.
#
# Documenter ships RequireJS for MathJax, and Plotly's UMD build can detect the global AMD
# `define` and register as an anonymous module instead of attaching `window.Plotly` — a bare
# `<script src>` would then fail silently with "Plotly is not defined". Guarded against below
# (verified live against a built page, not assumed).
#
# A Plotly chart is drawn with JS-supplied colours (paper/plot background, font colour, grid
# lines), which do not track Documenter's dark/light toggle on their own, so a chart drawn
# once in light colours turns unreadable text-on-background after a toggle unless something
# repaints it.
#
# `Plotly.newPlot` sizes a chart from its container's dimensions *at chart-creation time*,
# which runs synchronously as each `<script>` tag executes while the page is still loading —
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
    <script>
      // See the module note above: hide `define` from Plotly's UMD wrapper so it attaches
      // `window.Plotly` instead of registering as an anonymous AMD module.
      window.__bramble_amd_define = window.define;
      window.define = undefined;
    </script>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
      window.define = window.__bramble_amd_define;

      // Colour tokens read from the page's own theme, not hard-coded — Documenter stamps
      // `theme--documenter-dark` on <html> when dark mode is active, light mode has no such
      // class. Recomputed on every call so a caller can re-theme after a toggle.
      window.bramblePlotlyTheme = function () {
        const dark = document.documentElement.className.includes('documenter-dark');
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
        window.__bramble_plotly_theme_observer = new MutationObserver(function () {
          for (const { divId, restyle } of window.__bramble_plotly_charts) {
            const layout = restyle();
            Plotly.relayout(divId, layout);
          }
        });
        window.__bramble_plotly_theme_observer.observe(document.documentElement, {
          attributes: true,
          attributeFilter: ['class'],
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
