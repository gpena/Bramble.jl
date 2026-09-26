# Performance and benchmarks

Bramble tracks memory allocations and performance regressions with a dedicated regression suite in `benchmark/benchmarks.jl`.
All measurements below are run on **1,000,000 grid points** per dimension setup (e.g. $1000 \times 1000$ in 2D, $100 \times 100 \times 100$ in 3D).

## Comparative timings and allocations

Each chart below tracks one benchmark group across all **6** recorded baselines, in chronological release order, against the earliest run (v3.0.0) as the reference. Where a group's operations span more than a 20× range, the y-axis shows time relative to that reference instead of absolute time, so a cheap operation isn't flattened onto the same line as an expensive one. Hover any point for its exact time, Julia version, thread count, allocation count, and memory.

```@raw html
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

```

### Operators 2D

The finite-difference stencil engine on a 1000×1000 grid: the difference operator along the grid's contiguous storage direction (`D₋ₓ`) versus across it (`D₋ᵧ`), which access memory very differently and so can perform very differently.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div style="display:flex; flex-direction:column; gap:1.5rem; width:100%;">
  <div style="width:100%;"><div id="bench_chart_1" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Dcₓ",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [0.33675,0.255,0.25375,0.255333,0.265583,0.397292],
  customdata: [["1.13.0","336.8 μs",3,"7.64 MiB","4"],["1.13.0","255.0 μs",3,"7.64 MiB","4"],["1.13.0","253.8 μs",3,"7.64 MiB","4"],["1.13.0","255.3 μs",3,"7.64 MiB","4"],["1.13.0","265.6 μs",3,"7.64 MiB","4"],["1.13.0","397.3 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Dcₓ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "D₋(uₕ, d) over d",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [0.457958,0.367041,0.365917,0.367416,0.377688,0.525792],
  customdata: [["1.13.0","458.0 μs",6,"15.28 MiB","4"],["1.13.0","367.0 μs",6,"15.28 MiB","4"],["1.13.0","365.9 μs",6,"15.28 MiB","4"],["1.13.0","367.4 μs",6,"15.28 MiB","4"],["1.13.0","377.7 μs",6,"15.28 MiB","4"],["1.13.0","525.8 μs",6,"15.28 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>D₋(uₕ, d) over d: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "D₋ᵧ",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [0.202125,0.162667,0.161875,0.162042,0.16475,0.244375],
  customdata: [["1.13.0","202.1 μs",3,"7.64 MiB","4"],["1.13.0","162.7 μs",3,"7.64 MiB","4"],["1.13.0","161.9 μs",3,"7.64 MiB","4"],["1.13.0","162.0 μs",3,"7.64 MiB","4"],["1.13.0","164.8 μs",3,"7.64 MiB","4"],["1.13.0","244.4 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>D₋ᵧ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "D₋ₓ",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [0.2245,0.204459,0.203208,0.203209,0.209042,0.278709],
  customdata: [["1.13.0","224.5 μs",3,"7.64 MiB","4"],["1.13.0","204.5 μs",3,"7.64 MiB","4"],["1.13.0","203.2 μs",3,"7.64 MiB","4"],["1.13.0","203.2 μs",3,"7.64 MiB","4"],["1.13.0","209.0 μs",3,"7.64 MiB","4"],["1.13.0","278.7 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>D₋ₓ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Mₓ",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [0.213334,0.171833,0.170875,0.171334,0.176333,0.236417],
  customdata: [["1.13.0","213.3 μs",3,"7.64 MiB","4"],["1.13.0","171.8 μs",3,"7.64 MiB","4"],["1.13.0","170.9 μs",3,"7.64 MiB","4"],["1.13.0","171.3 μs",3,"7.64 MiB","4"],["1.13.0","176.3 μs",3,"7.64 MiB","4"],["1.13.0","236.4 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Mₓ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "ms", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_1', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_1', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div><div style="width:100%;"><div id="bench_chart_2" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "curlₕ!",
  x: ["v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [0.673333,0.705916,0.707708,0.731708,0.9972295],
  customdata: [["1.13.0","673.3 μs",0,"0 B","4"],["1.13.0","705.9 μs",0,"0 B","4"],["1.13.0","707.7 μs",0,"0 B","4"],["1.13.0","731.7 μs",0,"0 B","4"],["1.13.0","997.2 μs",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>curlₕ!: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "divₕ!",
  x: ["v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [0.711125,0.709208,0.726417,0.725417,1.01475],
  customdata: [["1.13.0","711.1 μs",0,"0 B","4"],["1.13.0","709.2 μs",0,"0 B","4"],["1.13.0","726.4 μs",0,"0 B","4"],["1.13.0","725.4 μs",0,"0 B","4"],["1.13.0","1.01 ms",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>divₕ!: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Δₕ",
  x: ["v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.1921875,1.1911665,1.1965,1.239271,1.547083],
  customdata: [["1.13.0","1.19 ms",3,"7.64 MiB","4"],["1.13.0","1.19 ms",3,"7.64 MiB","4"],["1.13.0","1.2 ms",3,"7.64 MiB","4"],["1.13.0","1.24 ms",3,"7.64 MiB","4"],["1.13.0","1.55 ms",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Δₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Δₕ!",
  x: ["v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.146083,1.149625,1.149417,1.162,1.517042],
  customdata: [["1.13.0","1.15 ms",0,"0 B","4"],["1.13.0","1.15 ms",0,"0 B","4"],["1.13.0","1.15 ms",0,"0 B","4"],["1.13.0","1.16 ms",0,"0 B","4"],["1.13.0","1.52 ms",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Δₕ!: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "ms", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_2', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_2', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div>
</div>

</div>
```

### Operators 3D

The same stencil engine in 3D (`D₋₂`), together with the inner product `innerₕ` and the full gradient `∇ₕ`.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_3" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "D₋₂",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [286.208,229.417,228.375,228.583,231.958,316.625],
  customdata: [["1.13.0","286.2 μs",3,"7.64 MiB","4"],["1.13.0","229.4 μs",3,"7.64 MiB","4"],["1.13.0","228.4 μs",3,"7.64 MiB","4"],["1.13.0","228.6 μs",3,"7.64 MiB","4"],["1.13.0","232.0 μs",3,"7.64 MiB","4"],["1.13.0","316.6 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>D₋₂: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [290.4165,895.166,892.125,197.375,197.375,262.125],
  customdata: [["1.13.0","290.4 μs",0,"0 B","4"],["1.13.0","895.2 μs",0,"0 B","4"],["1.13.0","892.1 μs",0,"0 B","4"],["1.13.0","197.4 μs",0,"0 B","4"],["1.13.0","197.4 μs",0,"0 B","4"],["1.13.0","262.1 μs",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>innerₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "∇ₕ",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [873.9795,684.8545,687.708,690.667,721.417,951.75],
  customdata: [["1.13.0","874.0 μs",9,"22.92 MiB","4"],["1.13.0","684.9 μs",9,"22.92 MiB","4"],["1.13.0","687.7 μs",9,"22.92 MiB","4"],["1.13.0","690.7 μs",9,"22.92 MiB","4"],["1.13.0","721.4 μs",9,"22.92 MiB","4"],["1.13.0","951.8 μs",9,"22.92 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>∇ₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "μs", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_3', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_3', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>

</div>
```

### Jumps and averages

Jump and average operators across cell interfaces, in 2D and 3D.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div style="display:flex; flex-direction:column; gap:1.5rem; width:100%;">
  <div style="width:100%;"><div id="bench_chart_4" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "M₊ᵧ 2D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [213.416,162.292,161.584,161.584,165.625,226.917],
  customdata: [["1.13.0","213.4 μs",3,"7.64 MiB","4"],["1.13.0","162.3 μs",3,"7.64 MiB","4"],["1.13.0","161.6 μs",3,"7.64 MiB","4"],["1.13.0","161.6 μs",3,"7.64 MiB","4"],["1.13.0","165.6 μs",3,"7.64 MiB","4"],["1.13.0","226.9 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>M₊ᵧ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "M₊₂ 3D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [284.0,227.667,226.417,227.833,229.833,283.375],
  customdata: [["1.13.0","284.0 μs",3,"7.64 MiB","4"],["1.13.0","227.7 μs",3,"7.64 MiB","4"],["1.13.0","226.4 μs",3,"7.64 MiB","4"],["1.13.0","227.8 μs",3,"7.64 MiB","4"],["1.13.0","229.8 μs",3,"7.64 MiB","4"],["1.13.0","283.4 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>M₊₂ 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "M₊ₓ 2D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [203.708,164.042,162.709,162.875,165.5415,213.625],
  customdata: [["1.13.0","203.7 μs",3,"7.64 MiB","4"],["1.13.0","164.0 μs",3,"7.64 MiB","4"],["1.13.0","162.7 μs",3,"7.64 MiB","4"],["1.13.0","162.9 μs",3,"7.64 MiB","4"],["1.13.0","165.5 μs",3,"7.64 MiB","4"],["1.13.0","213.6 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>M₊ₓ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jumpᵧ 2D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [191.875,153.0,161.541,161.5,165.958,200.458],
  customdata: [["1.13.0","191.9 μs",3,"7.64 MiB","4"],["1.13.0","153.0 μs",3,"7.64 MiB","4"],["1.13.0","161.5 μs",3,"7.64 MiB","4"],["1.13.0","161.5 μs",3,"7.64 MiB","4"],["1.13.0","166.0 μs",3,"7.64 MiB","4"],["1.13.0","200.5 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jumpᵧ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "μs", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_4', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_4', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div><div style="width:100%;"><div id="bench_chart_5" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "jump₂ 3D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [283.666,227.334,227.333,227.0,229.792,283.583],
  customdata: [["1.13.0","283.7 μs",3,"7.64 MiB","4"],["1.13.0","227.3 μs",3,"7.64 MiB","4"],["1.13.0","227.3 μs",3,"7.64 MiB","4"],["1.13.0","227.0 μs",3,"7.64 MiB","4"],["1.13.0","229.8 μs",3,"7.64 MiB","4"],["1.13.0","283.6 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jump₂ 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jumpₓ 2D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [196.709,163.791,162.375,162.834,165.584,203.667],
  customdata: [["1.13.0","196.7 μs",3,"7.64 MiB","4"],["1.13.0","163.8 μs",3,"7.64 MiB","4"],["1.13.0","162.4 μs",3,"7.64 MiB","4"],["1.13.0","162.8 μs",3,"7.64 MiB","4"],["1.13.0","165.6 μs",3,"7.64 MiB","4"],["1.13.0","203.7 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jumpₓ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jumpₕ 2D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [410.291,333.625,330.833,331.958,341.25,422.625],
  customdata: [["1.13.0","410.3 μs",6,"15.28 MiB","4"],["1.13.0","333.6 μs",6,"15.28 MiB","4"],["1.13.0","330.8 μs",6,"15.28 MiB","4"],["1.13.0","332.0 μs",6,"15.28 MiB","4"],["1.13.0","341.2 μs",6,"15.28 MiB","4"],["1.13.0","422.6 μs",6,"15.28 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jumpₕ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jumpₕ 3D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [784.917,641.084,641.583,645.625,678.375,811.792],
  customdata: [["1.13.0","784.9 μs",9,"22.92 MiB","4"],["1.13.0","641.1 μs",9,"22.92 MiB","4"],["1.13.0","641.6 μs",9,"22.92 MiB","4"],["1.13.0","645.6 μs",9,"22.92 MiB","4"],["1.13.0","678.4 μs",9,"22.92 MiB","4"],["1.13.0","811.8 μs",9,"22.92 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jumpₕ 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "μs", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_5', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_5', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div>
</div>

</div>
```

### Inner products 2D

The reduction path — inner products and norms — including the seminorm's sum over directions.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_6" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "innerₕ",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,3.1205457047125673,3.1206940708306914,0.6748609509208414,0.674419385093091,0.8029150409684775],
  customdata: [["1.13.0","283.1 μs (baseline)",0,"0 B","4"],["1.13.0","883.4 μs (+212.1%)",0,"0 B","4"],["1.13.0","883.4 μs (+212.1%)",0,"0 B","4"],["1.13.0","191.0 μs (-32.5%)",0,"0 B","4"],["1.13.0","190.9 μs (-32.6%)",0,"0 B","4"],["1.13.0","227.3 μs (-19.7%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>innerₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "norm₁ₕ",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,3.396276505574146,3.405479933383539,0.6332622031719873,0.6329622030692477,0.7894266402146028],
  customdata: [["1.13.0","973.3 μs (baseline)",0,"0 B","4"],["1.13.0","3.31 ms (+239.6%)",0,"0 B","4"],["1.13.0","3.31 ms (+240.5%)",0,"0 B","4"],["1.13.0","616.4 μs (-36.7%)",0,"0 B","4"],["1.13.0","616.1 μs (-36.7%)",0,"0 B","4"],["1.13.0","768.4 μs (-21.1%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>norm₁ₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "normₕ",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,3.8195001102507233,3.8195001102507233,0.6049349088377844,0.6038540193954801,0.7557233095970012],
  customdata: [["1.13.0","231.3 μs (baseline)",0,"0 B","4"],["1.13.0","883.4 μs (+282.0%)",0,"0 B","4"],["1.13.0","883.4 μs (+282.0%)",0,"0 B","4"],["1.13.0","139.9 μs (-39.5%)",0,"0 B","4"],["1.13.0","139.7 μs (-39.6%)",0,"0 B","4"],["1.13.0","174.8 μs (-24.4%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>normₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "snorm₁ₕ",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,3.4856770945702986,3.4911298094210714,0.6817101761956131,0.6854239482200647,0.8117583603020496],
  customdata: [["1.13.0","695.2 μs (baseline)",0,"0 B","4"],["1.13.0","2.42 ms (+248.6%)",0,"0 B","4"],["1.13.0","2.43 ms (+249.1%)",0,"0 B","4"],["1.13.0","474.0 μs (-31.8%)",0,"0 B","4"],["1.13.0","476.5 μs (-31.5%)",0,"0 B","4"],["1.13.0","564.4 μs (-18.8%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>snorm₁ₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_6', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_6', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>

</div>
```

### Restriction

Point interpolation (`Rₕ!`) and cell-averaging (`avgₕ!`), compared across the `Serial()` (the allocation-free default) and `Parallel()` backends, split by dimension.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div style="display:flex; flex-direction:column; gap:1.5rem; width:100%;">
  <div style="width:100%;"><div id="bench_chart_7" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ 1D (allocates its output)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.6651599226884469,0.6480039902737078,0.6435563314421099,0.7782904171082985,1.3830453270154],
  customdata: [["1.13.0","2.0 ms (baseline)",25,"7.64 MiB","4"],["1.13.0","1.33 ms (-33.5%)",25,"7.64 MiB","4"],["1.13.0","1.3 ms (-35.2%)",25,"7.64 MiB","4"],["1.13.0","1.29 ms (-35.6%)",25,"7.64 MiB","4"],["1.13.0","1.56 ms (-22.2%)",25,"7.64 MiB","4"],["1.13.0","2.77 ms (+38.3%)",25,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ 1D (allocates its output): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 1D, Parallel() backend",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.7438769659982564,0.7410636442894507,0.7197210113339145,0.8379893286835223,1.2951814472537053],
  customdata: [["1.13.0","1.79 ms (baseline)",22,"1.6 KiB","4"],["1.13.0","1.33 ms (-25.6%)",22,"1.6 KiB","4"],["1.13.0","1.33 ms (-25.9%)",22,"1.6 KiB","4"],["1.13.0","1.29 ms (-28.0%)",22,"1.6 KiB","4"],["1.13.0","1.5 ms (-16.2%)",22,"1.6 KiB","4"],["1.13.0","2.32 ms (+29.5%)",22,"1.6 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! 1D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 1D, Serial() backend (default)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.8800318428066469,0.8800380321711377,0.8764337589160006,0.8897948284617632,1.1569935250405476],
  customdata: [["1.13.0","3.39 ms (baseline)",0,"0 B","4"],["1.13.0","2.99 ms (-12.0%)",0,"0 B","4"],["1.13.0","2.99 ms (-12.0%)",0,"0 B","4"],["1.13.0","2.97 ms (-12.4%)",0,"0 B","4"],["1.13.0","3.02 ms (-11.0%)",0,"0 B","4"],["1.13.0","3.93 ms (+15.7%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! 1D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 1D, Parallel() backend",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.6419999553601538,0.6317951978645071,0.622936726563295,0.7738748442137855,1.2519872587887115],
  customdata: [["1.13.0","12.41 ms (baseline)",22,"2.0 KiB","4"],["1.13.0","7.97 ms (-35.8%)",22,"2.0 KiB","4"],["1.13.0","7.84 ms (-36.8%)",22,"2.0 KiB","4"],["1.13.0","7.73 ms (-37.7%)",22,"2.0 KiB","4"],["1.13.0","9.6 ms (-22.6%)",22,"2.0 KiB","4"],["1.13.0","15.54 ms (+25.2%)",22,"2.0 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! 1D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 1D, Serial() backend (default)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.8779469002279622,0.8796627987362095,0.8757500985550966,0.8784792005896166,1.1151219055128008],
  customdata: [["1.13.0","21.88 ms (baseline)",0,"0 B","4"],["1.13.0","19.21 ms (-12.2%)",0,"0 B","4"],["1.13.0","19.25 ms (-12.0%)",0,"0 B","4"],["1.13.0","19.16 ms (-12.4%)",0,"0 B","4"],["1.13.0","19.22 ms (-12.2%)",0,"0 B","4"],["1.13.0","24.4 ms (+11.5%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! 1D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_7', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_7', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div><div style="width:100%;"><div id="bench_chart_8" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ! 2D, Parallel() backend",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.6460168548258864,0.6423065034186675,0.6437271426299889,0.7822122435999364,1.2567764668468755],
  customdata: [["1.13.0","1.97 ms (baseline)",22,"1.6 KiB","4"],["1.13.0","1.27 ms (-35.4%)",22,"1.6 KiB","4"],["1.13.0","1.26 ms (-35.8%)",22,"1.6 KiB","4"],["1.13.0","1.27 ms (-35.6%)",22,"1.6 KiB","4"],["1.13.0","1.54 ms (-21.8%)",22,"1.6 KiB","4"],["1.13.0","2.47 ms (+25.7%)",22,"1.6 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! 2D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 2D, Serial() backend (default)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.9675260191846523,0.9294763788968825,0.9177158273381295,0.9227017985611511,1.1383943645083934],
  customdata: [["1.13.0","4.17 ms (baseline)",0,"0 B","4"],["1.13.0","4.03 ms (-3.2%)",0,"0 B","4"],["1.13.0","3.88 ms (-7.1%)",0,"0 B","4"],["1.13.0","3.83 ms (-8.2%)",0,"0 B","4"],["1.13.0","3.85 ms (-7.7%)",0,"0 B","4"],["1.13.0","4.75 ms (+13.8%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! 2D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 2D, Parallel() backend",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.8027422335490488,0.7832578061283799,0.7799937441436127,0.8612430415224777,1.4211164632576714],
  customdata: [["1.13.0","41.28 ms (baseline)",22,"2.0 KiB","4"],["1.13.0","33.14 ms (-19.7%)",22,"2.0 KiB","4"],["1.13.0","32.34 ms (-21.7%)",22,"2.0 KiB","4"],["1.13.0","32.2 ms (-22.0%)",22,"2.0 KiB","4"],["1.13.0","35.55 ms (-13.9%)",22,"2.0 KiB","4"],["1.13.0","58.67 ms (+42.1%)",22,"2.0 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! 2D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 2D, Serial() backend (default)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.8436446654722808,0.8222420788224813,0.8316207541464361,0.8446691427452883,1.3852735660816988],
  customdata: [["1.13.0","133.12 ms (baseline)",0,"0 B","4"],["1.13.0","112.3 ms (-15.6%)",0,"0 B","4"],["1.13.0","109.45 ms (-17.8%)",0,"0 B","4"],["1.13.0","110.7 ms (-16.8%)",0,"0 B","4"],["1.13.0","112.44 ms (-15.5%)",0,"0 B","4"],["1.13.0","184.4 ms (+38.5%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! 2D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_8', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_8', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div><div style="width:100%;"><div id="bench_chart_9" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ! 3D, Parallel() backend",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.6386799367021397,0.6334934639214382,0.6319704244403553,0.7826368711777127,1.4150786348831097],
  customdata: [["1.13.0","2.3 ms (baseline)",22,"1.7 KiB","4"],["1.13.0","1.47 ms (-36.1%)",22,"1.7 KiB","4"],["1.13.0","1.46 ms (-36.7%)",22,"1.7 KiB","4"],["1.13.0","1.45 ms (-36.8%)",22,"1.7 KiB","4"],["1.13.0","1.8 ms (-21.7%)",22,"1.7 KiB","4"],["1.13.0","3.25 ms (+41.5%)",22,"1.7 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! 3D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 3D, Serial() backend (default)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.8615092902759368,0.8579467405711532,0.8486724665622364,0.8957142788287745,1.1928304615013856],
  customdata: [["1.13.0","5.19 ms (baseline)",0,"0 B","4"],["1.13.0","4.47 ms (-13.8%)",0,"0 B","4"],["1.13.0","4.45 ms (-14.2%)",0,"0 B","4"],["1.13.0","4.4 ms (-15.1%)",0,"0 B","4"],["1.13.0","4.65 ms (-10.4%)",0,"0 B","4"],["1.13.0","6.19 ms (+19.3%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! 3D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 3D, Parallel() backend",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.5827310744955686,0.579170703656881,0.566938767713284,0.7515831449304569,1.1115831616864216],
  customdata: [["1.13.0","305.56 ms (baseline)",22,"2.1 KiB","4"],["1.13.0","178.06 ms (-41.7%)",22,"2.1 KiB","4"],["1.13.0","176.97 ms (-42.1%)",22,"2.1 KiB","4"],["1.13.0","173.24 ms (-43.3%)",22,"2.1 KiB","4"],["1.13.0","229.66 ms (-24.8%)",22,"2.1 KiB","4"],["1.13.0","339.66 ms (+11.2%)",22,"2.1 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! 3D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 3D, Serial() backend (default)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.9629543315379739,0.9628949307658564,0.9652902744950631,0.9606714275466405,1.1297898918510318],
  customdata: [["1.13.0","655.16 ms (baseline)",0,"0 B","4"],["1.13.0","630.89 ms (-3.7%)",0,"0 B","4"],["1.13.0","630.85 ms (-3.7%)",0,"0 B","4"],["1.13.0","632.42 ms (-3.5%)",0,"0 B","4"],["1.13.0","629.39 ms (-3.9%)",0,"0 B","4"],["1.13.0","740.19 ms (+13.0%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! 3D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_9', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_9', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div>
</div>

</div>
```

### Composite

A composite (multi-component) operator, which dispatches per component and calls the engine once per component with a view rather than once with a plain vector.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_10" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "D₋ₓ (3 components)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [0.833625,0.671542,0.659584,0.737333,0.717834,0.901333],
  customdata: [["1.13.0","833.6 μs",3,"22.89 MiB","4"],["1.13.0","671.5 μs",3,"22.89 MiB","4"],["1.13.0","659.6 μs",3,"22.89 MiB","4"],["1.13.0","737.3 μs",3,"22.89 MiB","4"],["1.13.0","717.8 μs",3,"22.89 MiB","4"],["1.13.0","901.3 μs",3,"22.89 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>D₋ₓ (3 components): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "∇ₕ (3 components)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.912416,1.465333,1.4417495,1.47525,1.555917,1.9421875],
  customdata: [["1.13.0","1.91 ms",6,"45.78 MiB","4"],["1.13.0","1.47 ms",6,"45.78 MiB","4"],["1.13.0","1.44 ms",6,"45.78 MiB","4"],["1.13.0","1.48 ms",6,"45.78 MiB","4"],["1.13.0","1.56 ms",6,"45.78 MiB","4"],["1.13.0","1.94 ms",6,"45.78 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>∇ₕ (3 components): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "ms", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_10', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_10', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>

</div>
```

### Construction

Mesh and grid-space construction, including the quadrature weights `gridspace` builds internally.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_11" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "gridspace 2D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.0015506945864120012,0.0016060219094169393,0.0016404987116083291,0.0016698060503586806,0.0016712807966349043],
  customdata: [["1.13.0","529.5 μs (baseline)",87,"22.96 MiB","4"],["1.13.0","821.1 ns (-99.8%)",6,"16.1 KiB","4"],["1.13.0","850.4 ns (-99.8%)",6,"16.1 KiB","4"],["1.13.0","868.6 ns (-99.8%)",6,"16.1 KiB","4"],["1.13.0","884.2 ns (-99.8%)",6,"16.1 KiB","4"],["1.13.0","884.9 ns (-99.8%)",6,"16.1 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>gridspace 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "gridspace 3D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.00026794710518185876,0.00027108809564054887,0.0002752508566774744,0.0002736266602878538,0.00027994358819381483],
  customdata: [["1.13.0","689.4 μs (baseline)",112,"30.58 MiB","4"],["1.13.0","184.7 ns (-100.0%)",6,"2.7 KiB","4"],["1.13.0","186.9 ns (-100.0%)",6,"2.7 KiB","4"],["1.13.0","189.8 ns (-100.0%)",6,"2.7 KiB","4"],["1.13.0","188.6 ns (-100.0%)",6,"2.7 KiB","4"],["1.13.0","193.0 ns (-100.0%)",6,"2.7 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>gridspace 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "hₘₐₓ 3D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.7193308217256058,0.7183678928168417,0.7069273803055722,0.7221966815731182,0.820461330201619],
  customdata: [["1.13.0","43.9 ns (baseline)",0,"0 B","4"],["1.13.0","31.6 ns (-28.1%)",0,"0 B","4"],["1.13.0","31.5 ns (-28.2%)",0,"0 B","4"],["1.13.0","31.0 ns (-29.3%)",0,"0 B","4"],["1.13.0","31.7 ns (-27.8%)",0,"0 B","4"],["1.13.0","36.0 ns (-18.0%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>hₘₐₓ 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_11', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_11', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>

</div>
```

### Startup and latency

Time to first `using Bramble` and first operator call — compilation latency, not steady-state performance.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_12" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "TTFX (load + first operator)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [563.247792,522.508792,470.129792,479.044375,529.997291,588.464625],
  customdata: [["1.13.0","563.25 ms",45,"1.3 KiB","4"],["1.13.0","522.51 ms",45,"1.3 KiB","4"],["1.13.0","470.13 ms",45,"1.3 KiB","4"],["1.13.0","479.04 ms",45,"1.3 KiB","4"],["1.13.0","530.0 ms",45,"1.3 KiB","4"],["1.13.0","588.46 ms",45,"1.3 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>TTFX (load + first operator): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "TTFX first-assembly (assemble)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [598.798542,583.592833,507.164708,517.530917,572.93325,628.420958],
  customdata: [["1.13.0","598.8 ms",45,"1.3 KiB","4"],["1.13.0","583.59 ms",45,"1.3 KiB","4"],["1.13.0","507.16 ms",45,"1.3 KiB","4"],["1.13.0","517.53 ms",45,"1.3 KiB","4"],["1.13.0","572.93 ms",45,"1.3 KiB","4"],["1.13.0","628.42 ms",45,"1.3 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>TTFX first-assembly (assemble): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "TTFX first-projection (Rₕ)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [595.986791,560.785,488.720333,506.951625,506.449416,617.339459],
  customdata: [["1.13.0","595.99 ms",45,"1.3 KiB","4"],["1.13.0","560.78 ms",45,"1.3 KiB","4"],["1.13.0","488.72 ms",45,"1.3 KiB","4"],["1.13.0","506.95 ms",45,"1.3 KiB","4"],["1.13.0","506.45 ms",45,"1.3 KiB","4"],["1.13.0","617.34 ms",45,"1.3 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>TTFX first-projection (Rₕ): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "TTFX mesh construction",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [559.741917,506.614291,457.741542,473.628917,507.270083,603.441084],
  customdata: [["1.13.0","559.74 ms",45,"1.3 KiB","4"],["1.13.0","506.61 ms",45,"1.3 KiB","4"],["1.13.0","457.74 ms",45,"1.3 KiB","4"],["1.13.0","473.63 ms",45,"1.3 KiB","4"],["1.13.0","507.27 ms",45,"1.3 KiB","4"],["1.13.0","603.44 ms",45,"1.3 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>TTFX mesh construction: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "using Bramble",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [515.704916,441.599292,447.629167,460.878167,487.654834,600.19125],
  customdata: [["1.13.0","515.7 ms",45,"1.3 KiB","4"],["1.13.0","441.6 ms",45,"1.3 KiB","4"],["1.13.0","447.63 ms",45,"1.3 KiB","4"],["1.13.0","460.88 ms",45,"1.3 KiB","4"],["1.13.0","487.65 ms",45,"1.3 KiB","4"],["1.13.0","600.19 ms",45,"1.3 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>using Bramble: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "ms", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_12', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_12', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>

</div>
```

### Forms

Linear and bilinear form assembly, across 1D/2D and the `Serial()`/`Parallel()` backends.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div style="display:flex; flex-direction:column; gap:1.5rem; width:100%;">
  <div style="width:100%;"><div id="bench_chart_13" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "assemble! 1D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.8650933911751425,0.8436017928065132,0.818170218888536,0.9651238866220908,1.2974889668920797],
  customdata: [["1.13.0","1.11 ms (baseline)",0,"0 B","4"],["1.13.0","956.0 μs (-13.5%)",0,"0 B","4"],["1.13.0","932.2 μs (-15.6%)",0,"0 B","4"],["1.13.0","904.1 μs (-18.2%)",0,"0 B","4"],["1.13.0","1.07 ms (-3.5%)",0,"0 B","4"],["1.13.0","1.43 ms (+29.7%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble! 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! 1D, Parallel() backend",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.4823352849802844,0.49177488349862586,0.4656274345799976,0.5767714183295495,1.1039349982076712],
  customdata: [["1.13.0","1.05 ms (baseline)",22,"1.9 KiB","4"],["1.13.0","504.6 μs (-51.8%)",22,"2.0 KiB","4"],["1.13.0","514.5 μs (-50.8%)",22,"2.0 KiB","4"],["1.13.0","487.1 μs (-53.4%)",22,"2.0 KiB","4"],["1.13.0","603.4 μs (-42.3%)",22,"2.0 KiB","4"],["1.13.0","1.15 ms (+10.4%)",22,"2.0 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble! 1D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble_parallel! 1D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.41723799361113856,0.41453875079690305,0.5160335193626139,0.511115306334655,0.9528744762599591],
  customdata: [["1.13.0","1.17 ms (baseline)",22,"1.9 KiB","4"],["1.13.0","489.5 μs (-58.3%)",22,"2.0 KiB","4"],["1.13.0","486.4 μs (-58.5%)",22,"2.0 KiB","4"],["1.13.0","605.5 μs (-48.4%)",22,"2.0 KiB","4"],["1.13.0","599.7 μs (-48.9%)",22,"2.0 KiB","4"],["1.13.0","1.12 ms (-4.7%)",22,"2.0 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble_parallel! 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "evaluate! 1D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.6473526512192141,0.6314794510398473,0.6103535198550467,0.7898216449010081,1.2994980047915965],
  customdata: [["1.13.0","1.81 ms (baseline)",0,"0 B","4"],["1.13.0","1.17 ms (-35.3%)",0,"0 B","4"],["1.13.0","1.14 ms (-36.9%)",0,"0 B","4"],["1.13.0","1.1 ms (-39.0%)",0,"0 B","4"],["1.13.0","1.43 ms (-21.0%)",0,"0 B","4"],["1.13.0","2.35 ms (+29.9%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>evaluate! 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "l(vₕ) 1D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.8020209869628093,0.8021716849395528,0.8020582075474267,0.8020963359511811,1.0033671011791663],
  customdata: [["1.13.0","1.1 ms (baseline)",0,"0 B","4"],["1.13.0","883.5 μs (-19.8%)",0,"0 B","4"],["1.13.0","883.6 μs (-19.8%)",0,"0 B","4"],["1.13.0","883.5 μs (-19.8%)",0,"0 B","4"],["1.13.0","883.5 μs (-19.8%)",0,"0 B","4"],["1.13.0","1.11 ms (+0.3%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>l(vₕ) 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_13', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_13', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div><div style="width:100%;"><div id="bench_chart_14" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "allocate_system_matrix 2D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.6485608652085277,0.8141045575481152,0.8540937769945751,0.9250312014097647,1.3745092429425383],
  customdata: [["1.13.0","3.62 ms (baseline)",21,"15.13 MiB","4"],["1.13.0","2.35 ms (-35.1%)",21,"15.13 MiB","4"],["1.13.0","2.95 ms (-18.6%)",52,"23.38 MiB","4"],["1.13.0","3.09 ms (-14.6%)",52,"23.38 MiB","4"],["1.13.0","3.35 ms (-7.5%)",52,"23.38 MiB","4"],["1.13.0","4.97 ms (+37.5%)",52,"23.38 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>allocate_system_matrix 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble (BilinearForm) 2D, Parallel() backend",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.6864426341986015,0.6998505100873118,0.7175604308239792,0.7915661729476112,1.6367436912970617],
  customdata: [["1.13.0","4.07 ms (baseline)",67,"15.13 MiB","4"],["1.13.0","2.79 ms (-31.4%)",67,"15.13 MiB","4"],["1.13.0","2.85 ms (-30.0%)",81,"15.82 MiB","4"],["1.13.0","2.92 ms (-28.2%)",81,"15.82 MiB","4"],["1.13.0","3.22 ms (-20.8%)",81,"15.82 MiB","4"],["1.13.0","6.66 ms (+63.7%)",109,"24.76 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble (BilinearForm) 2D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble (BilinearForm) 2D, Serial() backend",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.7930135551238492,0.8012845274466309,0.8432225380541983,0.9318097414603119,1.0995545127061221],
  customdata: [["1.13.0","5.5 ms (baseline)",36,"18.56 MiB","4"],["1.13.0","4.36 ms (-20.7%)",36,"18.56 MiB","4"],["1.13.0","4.41 ms (-19.9%)",52,"23.38 MiB","4"],["1.13.0","4.64 ms (-15.7%)",52,"23.38 MiB","4"],["1.13.0","5.13 ms (-6.8%)",52,"23.38 MiB","4"],["1.13.0","6.05 ms (+10.0%)",52,"23.38 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble (BilinearForm) 2D, Serial() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! (matrix) 2D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.4150283068719677,0.36810284052429193,0.39216967223555554,0.5171310490253078,1.0672100610782718],
  customdata: [["1.13.0","792.9 μs (baseline)",0,"0 B","4"],["1.13.0","329.1 μs (-58.5%)",0,"0 B","4"],["1.13.0","291.9 μs (-63.2%)",0,"0 B","4"],["1.13.0","311.0 μs (-60.8%)",0,"0 B","4"],["1.13.0","410.0 μs (-48.3%)",0,"0 B","4"],["1.13.0","846.2 μs (+6.7%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble! (matrix) 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! 2D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.6603535811951055,0.6575133578905854,0.653632717110932,0.6754354773490251,1.1247284285899184],
  customdata: [["1.13.0","1.8 ms (baseline)",0,"0 B","4"],["1.13.0","1.19 ms (-34.0%)",0,"0 B","4"],["1.13.0","1.19 ms (-34.2%)",0,"0 B","4"],["1.13.0","1.18 ms (-34.6%)",0,"0 B","4"],["1.13.0","1.22 ms (-32.5%)",0,"0 B","4"],["1.13.0","2.03 ms (+12.5%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble! 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble-then-add (matrix) 2D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.5825480731485381,0.5202059561298831,0.5745224702405535,0.5800395719701157,0.9826112220834619],
  customdata: [["1.13.0","11.74 ms (baseline)",85,"32.78 MiB","4"],["1.13.0","6.84 ms (-41.7%)",85,"32.78 MiB","4"],["1.13.0","6.11 ms (-48.0%)",113,"37.82 MiB","4"],["1.13.0","6.75 ms (-42.5%)",113,"37.82 MiB","4"],["1.13.0","6.81 ms (-42.0%)",113,"37.82 MiB","4"],["1.13.0","11.54 ms (-1.7%)",113,"37.82 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#06b6d4", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#06b6d4", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble-then-add (matrix) 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble_add! (matrix) 2D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.8406558385408773,0.8387971687600845,0.8512395589113633,0.8623848917335996,1.2972445331930789],
  customdata: [["1.13.0","448.7 μs (baseline)",0,"0 B","4"],["1.13.0","377.2 μs (-15.9%)",0,"0 B","4"],["1.13.0","376.4 μs (-16.1%)",0,"0 B","4"],["1.13.0","382.0 μs (-14.9%)",0,"0 B","4"],["1.13.0","387.0 μs (-13.8%)",0,"0 B","4"],["1.13.0","582.1 μs (+29.7%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f97316", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f97316", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble_add! (matrix) 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble_parallel! 2D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.5133930654478301,0.5146891852798116,0.5088996772143171,0.6699861159643599,1.1352278112061485],
  customdata: [["1.13.0","964.4 μs (baseline)",22,"2.0 KiB","4"],["1.13.0","495.1 μs (-48.7%)",22,"2.4 KiB","4"],["1.13.0","496.4 μs (-48.5%)",22,"2.4 KiB","4"],["1.13.0","490.8 μs (-49.1%)",22,"2.4 KiB","4"],["1.13.0","646.1 μs (-33.0%)",22,"2.4 KiB","4"],["1.13.0","1.09 ms (+13.5%)",22,"2.4 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble_parallel! 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "form (bilinear, 2D)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,1.160607987967568,1.1652194831770442,1.146851763039863,1.4180652162366214,1.6248537531564198],
  customdata: [["1.13.0","18.2 ns (baseline)",1,"32 B","4"],["1.13.0","21.1 ns (+16.1%)",1,"32 B","4"],["1.13.0","21.2 ns (+16.5%)",1,"32 B","4"],["1.13.0","20.9 ns (+14.7%)",1,"32 B","4"],["1.13.0","25.8 ns (+41.8%)",1,"32 B","4"],["1.13.0","29.5 ns (+62.5%)",1,"32 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>form (bilinear, 2D): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_14', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_14', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div>
</div>

</div>
```

### Jacobian sparsity

8 benchmarks in this group, across 6 recorded releases.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div style="display:flex; flex-direction:column; gap:1.5rem; width:100%;">
  <div style="width:100%;"><div id="bench_chart_15" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "jacobian (native), 1D n=100",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.8571215621972125,0.8726988149362749,0.8695684579265112,0.8850711783558173,1.049638518297682],
  customdata: [["1.13.0","13.4 μs (baseline)",59,"70.6 KiB","4"],["1.13.0","11.5 μs (-14.3%)",59,"70.6 KiB","4"],["1.13.0","11.7 μs (-12.7%)",78,"84.7 KiB","4"],["1.13.0","11.7 μs (-13.0%)",78,"84.7 KiB","4"],["1.13.0","11.9 μs (-11.5%)",78,"84.7 KiB","4"],["1.13.0","14.1 μs (+5.0%)",78,"84.7 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jacobian (native), 1D n=100: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jacobian (native), 1D n=10000",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.8537384209661921,0.7702834845741879,0.7705987401105093,0.8561377610060994,0.9543398605702161],
  customdata: [["1.13.0","1.06 ms (baseline)",64,"6.14 MiB","4"],["1.13.0","904.5 μs (-14.6%)",64,"6.14 MiB","4"],["1.13.0","816.1 μs (-23.0%)",84,"7.38 MiB","4"],["1.13.0","816.4 μs (-22.9%)",84,"7.38 MiB","4"],["1.13.0","907.0 μs (-14.4%)",84,"7.38 MiB","4"],["1.13.0","1.01 ms (-4.6%)",84,"7.38 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jacobian (native), 1D n=10000: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jacobian (traced), 1D n=100",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.9335194787927137,0.8060763196383459,0.8060763196383459,0.8393165802419891,1.1162744315915436],
  customdata: [["1.13.0","15.0 μs (baseline)",59,"70.6 KiB","4"],["1.13.0","14.0 μs (-6.6%)",59,"70.6 KiB","4"],["1.13.0","12.1 μs (-19.4%)",78,"84.7 KiB","4"],["1.13.0","12.1 μs (-19.4%)",78,"84.7 KiB","4"],["1.13.0","12.6 μs (-16.1%)",78,"84.7 KiB","4"],["1.13.0","16.8 μs (+11.6%)",78,"84.7 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jacobian (traced), 1D n=100: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jacobian (traced), 1D n=10000",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.8460619512782671,0.7644753806634363,0.7653339794695528,0.8644571494988305,0.9483486267340111],
  customdata: [["1.13.0","1.07 ms (baseline)",64,"6.14 MiB","4"],["1.13.0","902.6 μs (-15.4%)",64,"6.14 MiB","4"],["1.13.0","815.6 μs (-23.6%)",84,"7.38 MiB","4"],["1.13.0","816.5 μs (-23.5%)",84,"7.38 MiB","4"],["1.13.0","922.2 μs (-13.6%)",84,"7.38 MiB","4"],["1.13.0","1.01 ms (-5.2%)",84,"7.38 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jacobian (traced), 1D n=10000: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_15', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_15', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div><div style="width:100%;"><div id="bench_chart_16" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "prepare_jacobian (native), 1D n=100",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.8778971170152629,0.9507168919266149,0.9785446323038183,1.0556554807544067,1.0556554807544067],
  customdata: [["1.13.0","19.5 μs (baseline)",134,"86.7 KiB","4"],["1.13.0","17.1 μs (-12.2%)",134,"86.9 KiB","4"],["1.13.0","18.5 μs (-4.9%)",153,"93.4 KiB","4"],["1.13.0","19.0 μs (-2.1%)",153,"93.4 KiB","4"],["1.13.0","20.5 μs (+5.6%)",153,"93.4 KiB","4"],["1.13.0","20.5 μs (+5.6%)",153,"93.4 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>prepare_jacobian (native), 1D n=100: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "prepare_jacobian (native), 1D n=10000",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.853153996828654,0.8105273761775954,0.8430803096726052,0.8421169667008674,1.005844977147654],
  customdata: [["1.13.0","1.34 ms (baseline)",163,"7.48 MiB","4"],["1.13.0","1.14 ms (-14.7%)",163,"7.48 MiB","4"],["1.13.0","1.09 ms (-18.9%)",183,"8.03 MiB","4"],["1.13.0","1.13 ms (-15.7%)",183,"8.03 MiB","4"],["1.13.0","1.13 ms (-15.8%)",183,"8.03 MiB","4"],["1.13.0","1.35 ms (+0.6%)",183,"8.03 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>prepare_jacobian (native), 1D n=10000: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "prepare_jacobian (traced), 1D n=100",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.8213988889374918,0.8365775775337911,0.857224093434233,0.8833675983844393,1.0224837058746337],
  customdata: [["1.13.0","68.6 μs (baseline)",2595,"224.0 KiB","4"],["1.13.0","56.3 μs (-17.9%)",2595,"224.2 KiB","4"],["1.13.0","57.4 μs (-16.3%)",2633,"239.7 KiB","4"],["1.13.0","58.8 μs (-14.3%)",2633,"239.7 KiB","4"],["1.13.0","60.6 μs (-11.7%)",2633,"239.7 KiB","4"],["1.13.0","70.1 μs (+2.2%)",2633,"239.7 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>prepare_jacobian (traced), 1D n=100: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "prepare_jacobian (traced), 1D n=10000",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.8614001613993763,0.8569215468167245,0.8836615847673887,0.9751248664093001,1.072433422757312],
  customdata: [["1.13.0","5.73 ms (baseline)",240583,"21.17 MiB","4"],["1.13.0","4.94 ms (-13.9%)",240583,"21.17 MiB","4"],["1.13.0","4.91 ms (-14.3%)",240623,"22.5 MiB","4"],["1.13.0","5.06 ms (-11.6%)",240623,"22.5 MiB","4"],["1.13.0","5.59 ms (-2.5%)",240623,"22.5 MiB","4"],["1.13.0","6.15 ms (+7.2%)",240623,"22.5 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>prepare_jacobian (traced), 1D n=10000: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_16', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_16', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div>
</div>

</div>
```

### Precision 1D

The same 1D workload — restriction, assembly, inner product — repeated in `Float32`, `Float64`, and `Double64`, split by precision since `Double64` (software arithmetic) is an order of magnitude slower.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div style="display:flex; flex-direction:column; gap:1.5rem; width:100%;">
  <div style="width:100%;"><div id="bench_chart_17" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ! Float32",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.9995661164977203,0.9997095779783128,0.9995661164977203,0.9998565385194076,1.1911361799356872],
  customdata: [["1.13.0","285.8 μs (baseline)",0,"0 B","4"],["1.13.0","285.7 μs (-0.0%)",0,"0 B","4"],["1.13.0","285.7 μs (-0.0%)",0,"0 B","4"],["1.13.0","285.7 μs (-0.0%)",0,"0 B","4"],["1.13.0","285.8 μs (-0.0%)",0,"0 B","4"],["1.13.0","340.4 μs (+19.1%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! Float32",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,1.1740854059744703,1.1754507171996316,1.1747762863534676,1.1761415975786287,1.3934070272404264],
  customdata: [["1.13.0","60.8 μs (baseline)",0,"0 B","4"],["1.13.0","71.4 μs (+17.4%)",0,"0 B","4"],["1.13.0","71.5 μs (+17.5%)",0,"0 B","4"],["1.13.0","71.4 μs (+17.5%)",0,"0 B","4"],["1.13.0","71.5 μs (+17.6%)",0,"0 B","4"],["1.13.0","84.7 μs (+39.3%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble! Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! Float32",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,1.0119534335804865,1.004827385489571,1.0018826138174763,1.0402719146282309,1.1036538008454022],
  customdata: [["1.13.0","1.8 ms (baseline)",0,"0 B","4"],["1.13.0","1.83 ms (+1.2%)",0,"0 B","4"],["1.13.0","1.81 ms (+0.5%)",0,"0 B","4"],["1.13.0","1.81 ms (+0.2%)",0,"0 B","4"],["1.13.0","1.88 ms (+4.0%)",0,"0 B","4"],["1.13.0","1.99 ms (+10.4%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ Float32",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,6.882582864290181,6.882582864290181,0.8957160725453408,0.8989993746091307,1.1270325203252032],
  customdata: [["1.13.0","12.8 μs (baseline)",0,"0 B","4"],["1.13.0","88.0 μs (+588.3%)",0,"0 B","4"],["1.13.0","88.0 μs (+588.3%)",0,"0 B","4"],["1.13.0","11.5 μs (-10.4%)",0,"0 B","4"],["1.13.0","11.5 μs (-10.1%)",0,"0 B","4"],["1.13.0","14.4 μs (+12.7%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>innerₕ Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_17', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_17', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div><div style="width:100%;"><div id="bench_chart_18" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ! Float64",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.9998608400509121,1.0,1.0,1.0014153585065761,1.1938905388205345],
  customdata: [["1.13.0","294.6 μs (baseline)",0,"0 B","4"],["1.13.0","294.6 μs (-0.0%)",0,"0 B","4"],["1.13.0","294.6 μs (baseline)",0,"0 B","4"],["1.13.0","294.6 μs (baseline)",0,"0 B","4"],["1.13.0","295.0 μs (+0.1%)",0,"0 B","4"],["1.13.0","351.8 μs (+19.4%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! Float64",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,1.1778527246774328,1.174941923927342,1.1276973886758654,1.1778527246774328,1.4723054101710096],
  customdata: [["1.13.0","71.5 μs (baseline)",0,"0 B","4"],["1.13.0","84.2 μs (+17.8%)",0,"0 B","4"],["1.13.0","84.0 μs (+17.5%)",0,"0 B","4"],["1.13.0","80.6 μs (+12.8%)",0,"0 B","4"],["1.13.0","84.2 μs (+17.8%)",0,"0 B","4"],["1.13.0","105.2 μs (+47.2%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble! Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! Float64",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,0.9413734793631992,0.9407827163017038,0.946449665669381,0.9702256411556976,1.177547920556292],
  customdata: [["1.13.0","2.01 ms (baseline)",0,"0 B","4"],["1.13.0","1.89 ms (-5.9%)",0,"0 B","4"],["1.13.0","1.89 ms (-5.9%)",0,"0 B","4"],["1.13.0","1.9 ms (-5.4%)",0,"0 B","4"],["1.13.0","1.95 ms (-3.0%)",0,"0 B","4"],["1.13.0","2.37 ms (+17.8%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ Float64",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,3.569366739641612,3.569366739641612,0.9273899294575529,0.9290926781804913,1.1622070866780183],
  customdata: [["1.13.0","24.7 μs (baseline)",0,"0 B","4"],["1.13.0","88.0 μs (+256.9%)",0,"0 B","4"],["1.13.0","88.0 μs (+256.9%)",0,"0 B","4"],["1.13.0","22.9 μs (-7.3%)",0,"0 B","4"],["1.13.0","22.9 μs (-7.1%)",0,"0 B","4"],["1.13.0","28.7 μs (+16.2%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>innerₕ Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_18', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_18', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div><div style="width:100%;"><div id="bench_chart_19" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ! Double64",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,1.0013142390447223,1.0067965528894898,1.0097723893207164,1.023633209417596,1.0718101272952574],
  customdata: [["1.13.0","8.88 ms (baseline)",0,"0 B","4"],["1.13.0","8.89 ms (+0.1%)",0,"0 B","4"],["1.13.0","8.94 ms (+0.7%)",0,"0 B","4"],["1.13.0","8.96 ms (+1.0%)",0,"0 B","4"],["1.13.0","9.09 ms (+2.4%)",0,"0 B","4"],["1.13.0","9.51 ms (+7.2%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! Double64",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,1.0123843964226242,0.9980951845656125,1.0145678607765207,1.0226653507451595,1.2253407614307978],
  customdata: [["1.13.0","1.05 ms (baseline)",0,"0 B","4"],["1.13.0","1.06 ms (+1.2%)",0,"0 B","4"],["1.13.0","1.05 ms (-0.2%)",0,"0 B","4"],["1.13.0","1.06 ms (+1.5%)",0,"0 B","4"],["1.13.0","1.07 ms (+2.3%)",0,"0 B","4"],["1.13.0","1.29 ms (+22.5%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble! Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! Double64",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,1.0060791250666952,1.0070613133110076,1.0083224108337097,1.030904634616317,1.1034871194801694],
  customdata: [["1.13.0","83.21 ms (baseline)",32,"2.9 KiB","4"],["1.13.0","83.72 ms (+0.6%)",32,"2.9 KiB","4"],["1.13.0","83.8 ms (+0.7%)",32,"2.9 KiB","4"],["1.13.0","83.9 ms (+0.8%)",32,"2.9 KiB","4"],["1.13.0","85.78 ms (+3.1%)",32,"2.9 KiB","4"],["1.13.0","91.82 ms (+10.3%)",32,"2.9 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ Double64",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1.0,1.0069254223568598,1.0069254223568598,1.00363878123835,1.0125981649274074,1.1037630162726297],
  customdata: [["1.13.0","1.06 ms (baseline)",0,"0 B","4"],["1.13.0","1.07 ms (+0.7%)",0,"0 B","4"],["1.13.0","1.07 ms (+0.7%)",0,"0 B","4"],["1.13.0","1.07 ms (+0.4%)",0,"0 B","4"],["1.13.0","1.08 ms (+1.3%)",0,"0 B","4"],["1.13.0","1.18 ms (+10.4%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>innerₕ Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
  y: [1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [],
    annotations: [],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1","v3.12.0","v3.13.0"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 160, b: 60 },
  };
  Plotly.newPlot('bench_chart_19', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_19', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'legend.font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div>
</div>

</div>
```

## How to add new benchmark runs

To record performance on a new commit or after an optimization pass, run:

```bash
julia --project=benchmark benchmark/benchmarks.jl --save benchmark/baselines/baseline_$(git rev-parse --short HEAD).json
```

Rebuilding the documentation (`julia -e 'using Pkg; Pkg.activate("docs"); include("docs/make.jl")'`) will automatically discover all `baseline_*.json` files and append new comparison columns, delta calculations, and charts.
