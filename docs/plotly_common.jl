# Shared Plotly.js loading/theming for solution-field plots (docs/src/solution_plot.jl).
# Mirrors chartjs_common.jl's role for Chart.js — a separate file, and a separate registry/
# observer, because Plotly's repaint API (`Plotly.relayout`/`Plotly.Plots.resize`) is shaped
# nothing like Chart.js's (`chart.update()`/`chart.resize()`), so there is nothing to share
# beyond the general pattern.
#
# Plotly is used here (not Chart.js) because neither a heatmap nor a 3D isosurface has a
# Chart.js trace type at all — both are native Plotly trace types.
#
# Same three problems as chartjs_common.jl, solved the same way:
#
# Plotly's UMD build can hit the same Documenter-ships-RequireJS AMD clash Chart.js's does
# (a bare `<script src>` registering as an anonymous module instead of attaching
# `window.Plotly`) — guarded the same way, verified live against the built page rather than
# assumed.
#
# A Plotly chart is drawn with JS-supplied colours (paper/plot background, font colour),
# which do not track Documenter's dark/light toggle on their own.
#
# `Plotly.newPlot` sizes a chart from its container at creation time, before web fonts finish
# swapping in and before layout has settled — the same "baked-in wrong size" risk
# chartjs_common.jl documents for Chart.js, fixed the same way: resize every registered plot
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

      // Colour tokens read from the page's own theme, not hard-coded — same check
      // chartjs_common.jl uses: Documenter stamps `theme--documenter-dark` on <html> when
      // dark mode is active, light mode has no such class.
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
