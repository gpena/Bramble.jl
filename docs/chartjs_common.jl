# Shared Chart.js loading/theming for every chart on the docs site (benchmark trend charts,
# convergence plots). Included once per page that needs it — `generate_benchmarks.jl` for the
# benchmark page, `convergence_plot.jl` for the four worked examples.
#
# Three problems every Chart.js-on-Documenter page has, solved once here rather than per call
# site.
#
# Documenter ships RequireJS for MathJax, and Chart.js's UMD build detects the global AMD
# `define` and registers as an anonymous module instead of attaching `window.Chart` — a bare
# `<script src>` fails silently with "Chart is not defined".
#
# Chart.js draws on a canvas with JS-supplied colours, which (unlike the SVG plots'
# `currentColor`) do not track Documenter's dark/light toggle on their own, so a chart drawn
# once in light colours turns unreadable text-on-background after a toggle unless something
# repaints it.
#
# `responsive: true` sizes a chart from its container's dimensions *at chart-creation time*,
# which runs synchronously as each `<script>` tag executes while the page is still loading —
# before web fonts finish swapping in and before every chart above it has settled the page's
# final layout. A chart created against a not-yet-final container size can bake that size in
# permanently: found live, on the deployed site rather than a fast local build (font/layout
# timing is exactly the kind of thing that reproduces differently under real network
# conditions), as every point in one chart landing on the same pixel row regardless of its
# real value — axes drawn, data not. Fixed the same way for every chart at once, on
# `window.load` once the whole page (fonts included) has truly finished, rather than any
# per-chart guess about how long to wait.

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

      // The layout-race fix (see the module note above): once per page, after everything
      // (fonts included) has truly finished loading, force every chart created so far to
      // resize against its now-final container and relayout. `resize()` reads the actual
      // current size; `update()` (not 'none' — this one may need to move real distance, not
      // just recolour) redraws from it. A chart registered *after* window.load (later
      // content on the same page) already has this page's final layout available at its own
      // creation time, so it does not need the same rescue.
      if (!window.__bramble_load_fix_installed) {
        window.__bramble_load_fix_installed = true;
        const rescue = function () {
          for (const { chart } of window.__bramble_charts) {
            chart.resize();
            chart.update();
          }
        };
        // `window.load` does not wait for web fonts — those load asynchronously and can
        // still swap in (reflowing text, and with it every container's width) afterwards.
        // `document.fonts.ready` is the one signal that actually waits for that; checked
        // live and found `load` alone left one chart still stuck at its pre-font-swap size
        // while every other chart on the same page had already settled by the time `load`
        // fired. Both awaited, in whichever order they resolve, before the rescue pass runs.
        const loaded = document.readyState === 'complete' ? Promise.resolve() :
          new Promise((r) => window.addEventListener('load', r, { once: true }));
        const fontsReady = (document.fonts && document.fonts.ready) || Promise.resolve();
        Promise.all([loaded, fontsReady]).then(rescue);
      }
    </script>
    """
end
