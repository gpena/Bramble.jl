# Shared Chart.js loading/theming for every chart on the docs site (benchmark trend charts,
# convergence plots). Included once per page that needs it — `generate_benchmarks.jl` for the
# benchmark page, `convergence_plot.jl` for the four worked examples.
#
# Two problems every Chart.js-on-Documenter page has, solved once here rather than per call
# site: Documenter ships RequireJS for MathJax, and Chart.js's UMD build detects the global
# AMD `define` and registers as an anonymous module instead of attaching `window.Chart` — a
# bare `<script src>` fails silently with "Chart is not defined". And Chart.js draws on a
# canvas with JS-supplied colours, which (unlike the SVG plots' `currentColor`) do not track
# Documenter's dark/light toggle on their own, so a chart drawn once in light colours turns
# unreadable text-on-background after a toggle unless something repaints it.

"""
    chartjs_head() -> String

The `<script>` tags that load Chart.js from a CDN and make `Chart`/`brambleChartTheme`/
`brambleRegisterChart` available. `@raw html` this once per page, before any chart's own
canvas + `new Chart(...)` script.
"""
function chartjs_head()
    return """
    <script>
      // See the module note above: hide `define` from Chart.js's UMD wrapper so it attaches
      // `window.Chart` instead of registering as an anonymous AMD module.
      window.__bramble_amd_define = window.define;
      window.define = undefined;
    </script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
    <script>
      window.define = window.__bramble_amd_define;

      // Colour tokens read from the page's own theme, not hard-coded — Documenter stamps
      // `theme--documenter-dark` on <html> when dark mode is active, light mode has no such
      // class. Recomputed on every call so a caller can re-theme after a toggle.
      window.brambleChartTheme = function () {
        const dark = document.documentElement.className.includes('documenter-dark');
        return dark
          ? { text: '#c3c2b7', grid: 'rgba(255,255,255,0.12)', axis: 'rgba(255,255,255,0.35)' }
          : { text: '#52514e', grid: 'rgba(0,0,0,0.10)', axis: 'rgba(0,0,0,0.35)' };
      };

      // Every chart this page creates registers itself (chart instance + the function that
      // reapplies theme-dependent option colours) so one observer can repaint all of them
      // together when the theme toggles, instead of each chart wiring its own observer.
      window.__bramble_charts = window.__bramble_charts || [];
      window.brambleRegisterChart = function (chart, restyle) {
        window.__bramble_charts.push({ chart, restyle });
      };

      if (!window.__bramble_theme_observer) {
        window.__bramble_theme_observer = new MutationObserver(function () {
          for (const { chart, restyle } of window.__bramble_charts) {
            restyle(chart);
            chart.update('none');
          }
        });
        window.__bramble_theme_observer.observe(document.documentElement, {
          attributes: true,
          attributeFilter: ['class'],
        });
      }
    </script>
    """
end
