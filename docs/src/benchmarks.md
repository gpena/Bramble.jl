# Performance and benchmarks

Bramble tracks memory allocations and performance regressions with a dedicated regression suite in `benchmark/benchmarks.jl`.
All measurements below are run on **1,000,000 grid points** per dimension setup (e.g. $1000 \times 1000$ in 2D, $100 \times 100 \times 100$ in 3D).

## Comparative timings and allocations

Each chart below tracks one benchmark group across all **5** recorded baselines, in chronological release order, against the earliest run (v2.0.0) as the reference. Where a group's operations span more than a 20× range, the y-axis shows time relative to that reference instead of absolute time, so a cheap operation isn't flattened onto the same line as an expensive one. Hover any point for its exact time, Julia version, allocation count, and memory.

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
<div id="bench_chart_1" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Dcₓ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [256.5,268.708,278.5625,254.125,256.833],
  customdata: [["1.12.7","256.5 μs",3,"7.64 MiB"],["1.12.7","268.7 μs",3,"7.64 MiB"],["1.12.7","278.6 μs",3,"7.64 MiB"],["1.12.7","254.1 μs",3,"7.64 MiB"],["1.12.7","256.8 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Dcₓ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "D₋ᵧ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [161.958,162.209,177.6455,162.167,161.958],
  customdata: [["1.12.7","162.0 μs",3,"7.64 MiB"],["1.12.7","162.2 μs",3,"7.64 MiB"],["1.12.7","177.6 μs",3,"7.64 MiB"],["1.12.7","162.2 μs",3,"7.64 MiB"],["1.12.7","162.0 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>D₋ᵧ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "D₋ₓ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [204.042,205.208,210.291,203.958,203.25],
  customdata: [["1.12.7","204.0 μs",3,"7.64 MiB"],["1.12.7","205.2 μs",3,"7.64 MiB"],["1.12.7","210.3 μs",3,"7.64 MiB"],["1.12.7","204.0 μs",3,"7.64 MiB"],["1.12.7","203.2 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>D₋ₓ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "M₋ₓ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [171.959,161.458,180.583,172.209,171.291],
  customdata: [["1.12.7","172.0 μs",3,"7.64 MiB"],["1.12.7","161.5 μs",3,"7.64 MiB"],["1.12.7","180.6 μs",3,"7.64 MiB"],["1.12.7","172.2 μs",3,"7.64 MiB"],["1.12.7","171.3 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>M₋ₓ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "μs", font: { color: theme.text } },
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

</div>
```

### Operators 3D

The same stencil engine in 3D (`D₋₂`), together with the inner product `innerₕ` and the full gradient `∇₋ₕ`.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_2" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "D₋₂",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [229.584,212.792,228.583,227.416,228.25],
  customdata: [["1.12.7","229.6 μs",3,"7.64 MiB"],["1.12.7","212.8 μs",3,"7.64 MiB"],["1.12.7","228.6 μs",3,"7.64 MiB"],["1.12.7","227.4 μs",3,"7.64 MiB"],["1.12.7","228.2 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>D₋₂: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [240.375,240.292,240.666,239.041,239.167],
  customdata: [["1.12.7","240.4 μs",0,"0 B"],["1.12.7","240.3 μs",0,"0 B"],["1.12.7","240.7 μs",0,"0 B"],["1.12.7","239.0 μs",0,"0 B"],["1.12.7","239.2 μs",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>innerₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "∇₋ₕ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [690.75,696.667,685.875,701.3335,687.125],
  customdata: [["1.12.7","690.8 μs",15,"22.92 MiB"],["1.12.7","696.7 μs",15,"22.92 MiB"],["1.12.7","685.9 μs",15,"22.92 MiB"],["1.12.7","701.3 μs",15,"22.92 MiB"],["1.12.7","687.1 μs",15,"22.92 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>∇₋ₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "μs", font: { color: theme.text } },
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
```

### Jumps and averages

Jump and average operators across cell interfaces, in 2D and 3D.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_3" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "M₊ᵧ 2D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [161.292,161.291,161.625,160.959,162.125],
  customdata: [["1.12.7","161.3 μs",3,"7.64 MiB"],["1.12.7","161.3 μs",3,"7.64 MiB"],["1.12.7","161.6 μs",3,"7.64 MiB"],["1.12.7","161.0 μs",3,"7.64 MiB"],["1.12.7","162.1 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>M₊ᵧ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "M₊₂ 3D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [227.708,226.792,228.458,228.291,228.458],
  customdata: [["1.12.7","227.7 μs",3,"7.64 MiB"],["1.12.7","226.8 μs",3,"7.64 MiB"],["1.12.7","228.5 μs",3,"7.64 MiB"],["1.12.7","228.3 μs",3,"7.64 MiB"],["1.12.7","228.5 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>M₊₂ 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "M₊ₓ 2D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [161.625,159.625,162.417,163.666,164.416],
  customdata: [["1.12.7","161.6 μs",3,"7.64 MiB"],["1.12.7","159.6 μs",3,"7.64 MiB"],["1.12.7","162.4 μs",3,"7.64 MiB"],["1.12.7","163.7 μs",3,"7.64 MiB"],["1.12.7","164.4 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>M₊ₓ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jumpᵧ 2D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [160.834,159.917,160.958,161.166,161.583],
  customdata: [["1.12.7","160.8 μs",3,"7.64 MiB"],["1.12.7","159.9 μs",3,"7.64 MiB"],["1.12.7","161.0 μs",3,"7.64 MiB"],["1.12.7","161.2 μs",3,"7.64 MiB"],["1.12.7","161.6 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>jumpᵧ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jump₂ 3D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [227.334,226.667,227.25,227.667,221.75],
  customdata: [["1.12.7","227.3 μs",3,"7.64 MiB"],["1.12.7","226.7 μs",3,"7.64 MiB"],["1.12.7","227.2 μs",3,"7.64 MiB"],["1.12.7","227.7 μs",3,"7.64 MiB"],["1.12.7","221.8 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>jump₂ 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jumpₓ 2D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [161.458,161.834,160.5,163.292,164.125],
  customdata: [["1.12.7","161.5 μs",3,"7.64 MiB"],["1.12.7","161.8 μs",3,"7.64 MiB"],["1.12.7","160.5 μs",3,"7.64 MiB"],["1.12.7","163.3 μs",3,"7.64 MiB"],["1.12.7","164.1 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#06b6d4", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#06b6d4", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>jumpₓ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
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

### Inner products 2D

The reduction path — inner products and norms — including the seminorm's sum over directions.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_4" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "innerₕ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [240.375,238.041,236.959,239.917,238.833],
  customdata: [["1.12.7","240.4 μs",0,"0 B"],["1.12.7","238.0 μs",0,"0 B"],["1.12.7","237.0 μs",0,"0 B"],["1.12.7","239.9 μs",0,"0 B"],["1.12.7","238.8 μs",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>innerₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "norm₁ₕ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [789.791,788.458,785.667,792.708,792.958],
  customdata: [["1.12.7","789.8 μs",0,"0 B"],["1.12.7","788.5 μs",0,"0 B"],["1.12.7","785.7 μs",0,"0 B"],["1.12.7","792.7 μs",0,"0 B"],["1.12.7","793.0 μs",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>norm₁ₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "normₕ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [189.292,186.167,184.417,187.208,189.542],
  customdata: [["1.12.7","189.3 μs",0,"0 B"],["1.12.7","186.2 μs",0,"0 B"],["1.12.7","184.4 μs",0,"0 B"],["1.12.7","187.2 μs",0,"0 B"],["1.12.7","189.5 μs",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>normₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "snorm₁ₕ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [578.5,582.375,578.209,580.375,579.5],
  customdata: [["1.12.7","578.5 μs",0,"0 B"],["1.12.7","582.4 μs",0,"0 B"],["1.12.7","578.2 μs",0,"0 B"],["1.12.7","580.4 μs",0,"0 B"],["1.12.7","579.5 μs",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>snorm₁ₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
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

</div>
```

### Restriction

Point interpolation (`Rₕ!`) and cell-averaging (`avgₕ!`), compared across the `Serial()` (the allocation-free default) and `Parallel()` backends, split by dimension.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div style="display:flex; flex-direction:column; gap:1.5rem; width:100%;">
  <div style="width:100%;"><div id="bench_chart_5" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ 1D (allocates its output)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0149558149254283,1.0576855523560704,1.0014603442143817,1.0023077913658966],
  customdata: [["1.12.7","3.2 ms (baseline)",10,"7.64 MiB"],["1.12.7","3.24 ms (+1.5%)",10,"7.64 MiB"],["1.12.7","3.38 ms (+5.8%)",10,"7.64 MiB"],["1.12.7","3.2 ms (+0.1%)",10,"7.64 MiB"],["1.12.7","3.2 ms (+0.2%)",10,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ 1D (allocates its output): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 1D, Parallel() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.024612615432775,1.0042131789012365,1.0028173423070903,1.0015000782595085],
  customdata: [["1.12.7","3.19 ms (baseline)",7,"448 B"],["1.12.7","3.27 ms (+2.5%)",7,"448 B"],["1.12.7","3.21 ms (+0.4%)",7,"448 B"],["1.12.7","3.2 ms (+0.3%)",7,"448 B"],["1.12.7","3.2 ms (+0.2%)",7,"448 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! 1D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 1D, Serial() backend (default)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0301685765215176,1.0069036584330702,1.0014642220524574,1.0011104320516084],
  customdata: [["1.12.7","2.95 ms (baseline)",0,"0 B"],["1.12.7","3.03 ms (+3.0%)",0,"0 B"],["1.12.7","2.97 ms (+0.7%)",0,"0 B"],["1.12.7","2.95 ms (+0.1%)",0,"0 B"],["1.12.7","2.95 ms (+0.1%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! 1D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 1D, Parallel() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0068065935214991,0.9960489659344681,0.9949062282935864,0.9958261545011166],
  customdata: [["1.12.7","16.74 ms (baseline)",7,"544 B"],["1.12.7","16.85 ms (+0.7%)",7,"544 B"],["1.12.7","16.67 ms (-0.4%)",7,"544 B"],["1.12.7","16.65 ms (-0.5%)",7,"544 B"],["1.12.7","16.67 ms (-0.4%)",7,"544 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! 1D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 1D, Serial() backend (default)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,0.994873716820243,0.9963419560811928,0.9964045252268809,0.997997816198286],
  customdata: [["1.12.7","17.32 ms (baseline)",0,"0 B"],["1.12.7","17.24 ms (-0.5%)",0,"0 B"],["1.12.7","17.26 ms (-0.4%)",0,"0 B"],["1.12.7","17.26 ms (-0.4%)",0,"0 B"],["1.12.7","17.29 ms (-0.2%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! 1D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
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
</div><div style="width:100%;"><div id="bench_chart_6" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ! 2D, Parallel() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0244596787155744,1.0037520529878992,1.0018323096971185,0.9997382227734669],
  customdata: [["1.12.7","3.82 ms (baseline)",7,"448 B"],["1.12.7","3.91 ms (+2.4%)",7,"448 B"],["1.12.7","3.83 ms (+0.4%)",7,"448 B"],["1.12.7","3.83 ms (+0.2%)",7,"448 B"],["1.12.7","3.82 ms (-0.0%)",7,"448 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! 2D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 2D, Serial() backend (default)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,0.9939257266957312,0.995707065803144,0.9941954426632402,0.9934426653979228],
  customdata: [["1.12.7","3.71 ms (baseline)",0,"0 B"],["1.12.7","3.69 ms (-0.6%)",0,"0 B"],["1.12.7","3.69 ms (-0.4%)",0,"0 B"],["1.12.7","3.69 ms (-0.6%)",0,"0 B"],["1.12.7","3.68 ms (-0.7%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! 2D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 2D, Parallel() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,0.996973378521717,1.0030935735212574,0.9988203732425536,0.9973027528709953],
  customdata: [["1.12.7","106.39 ms (baseline)",7,"560 B"],["1.12.7","106.07 ms (-0.3%)",7,"560 B"],["1.12.7","106.72 ms (+0.3%)",7,"560 B"],["1.12.7","106.26 ms (-0.1%)",7,"560 B"],["1.12.7","106.1 ms (-0.3%)",7,"560 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! 2D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 2D, Serial() backend (default)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,0.990859277878407,0.9963455292959916,0.9925425798108675,0.9908653413907442],
  customdata: [["1.12.7","110.0 ms (baseline)",0,"0 B"],["1.12.7","109.0 ms (-0.9%)",0,"0 B"],["1.12.7","109.6 ms (-0.4%)",0,"0 B"],["1.12.7","109.18 ms (-0.7%)",0,"0 B"],["1.12.7","109.0 ms (-0.9%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! 2D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
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
</div><div style="width:100%;"><div id="bench_chart_7" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ! 3D, Parallel() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.031481808717184,1.0251649174146713,1.001388221390585,1.0013317200821632],
  customdata: [["1.12.7","4.44 ms (baseline)",7,"464 B"],["1.12.7","4.58 ms (+3.1%)",7,"464 B"],["1.12.7","4.55 ms (+2.5%)",7,"464 B"],["1.12.7","4.45 ms (+0.1%)",7,"464 B"],["1.12.7","4.45 ms (+0.1%)",7,"464 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! 3D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 3D, Serial() backend (default)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,0.9934655799749492,0.9946625222703774,0.9932147016111117,0.9945561943211609],
  customdata: [["1.12.7","4.32 ms (baseline)",0,"0 B"],["1.12.7","4.29 ms (-0.7%)",0,"0 B"],["1.12.7","4.29 ms (-0.5%)",0,"0 B"],["1.12.7","4.29 ms (-0.7%)",0,"0 B"],["1.12.7","4.29 ms (-0.5%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! 3D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 3D, Parallel() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0626395001054676,1.0031843637550093,0.9998218554882745,1.0021153017605628],
  customdata: [["1.12.7","620.75 ms (baseline)",7,"576 B"],["1.12.7","659.64 ms (+6.3%)",7,"576 B"],["1.12.7","622.73 ms (+0.3%)",7,"576 B"],["1.12.7","620.64 ms (-0.0%)",7,"576 B"],["1.12.7","622.07 ms (+0.2%)",7,"576 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! 3D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 3D, Serial() backend (default)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,0.9953563221063543,0.9961396104082121,0.9993011818337315,0.9935846831953266],
  customdata: [["1.12.7","643.71 ms (baseline)",0,"0 B"],["1.12.7","640.72 ms (-0.5%)",0,"0 B"],["1.12.7","641.22 ms (-0.4%)",0,"0 B"],["1.12.7","643.26 ms (-0.1%)",0,"0 B"],["1.12.7","639.58 ms (-0.6%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! 3D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
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
</div>
</div>

</div>
```

### Composite

A composite (multi-component) operator, which dispatches per component and calls the engine once per component with a view rather than once with a plain vector.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_8" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "D₋ₓ (3 components)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [0.670208,0.693958,0.659917,0.657834,0.65625],
  customdata: [["1.12.7","670.2 μs",3,"22.89 MiB"],["1.12.7","694.0 μs",3,"22.89 MiB"],["1.12.7","659.9 μs",3,"22.89 MiB"],["1.12.7","657.8 μs",3,"22.89 MiB"],["1.12.7","656.2 μs",3,"22.89 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>D₋ₓ (3 components): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "∇₋ₕ (3 components)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.380625,1.407666,1.395312,1.385937,1.397542],
  customdata: [["1.12.7","1.38 ms",10,"45.78 MiB"],["1.12.7","1.41 ms",10,"45.78 MiB"],["1.12.7","1.4 ms",10,"45.78 MiB"],["1.12.7","1.39 ms",10,"45.78 MiB"],["1.12.7","1.4 ms",10,"45.78 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>∇₋ₕ (3 components): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "ms", font: { color: theme.text } },
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

</div>
```

### Construction

Mesh and grid-space construction, including the quadrature weights `gridspace` builds internally.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_9" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "gridspace 2D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0100427623813584,1.0026278231190768,1.0086912332613118,1.0026845873421188],
  customdata: [["1.12.7","2.22 ms (baseline)",42,"22.95 MiB"],["1.12.7","2.24 ms (+1.0%)",42,"22.95 MiB"],["1.12.7","2.23 ms (+0.3%)",42,"22.95 MiB"],["1.12.7","2.24 ms (+0.9%)",42,"22.95 MiB"],["1.12.7","2.23 ms (+0.3%)",42,"22.95 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>gridspace 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "gridspace 3D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0131330029173518,1.000413423237967,1.0038948566148846,1.0024061780884717],
  customdata: [["1.12.7","6.2 ms (baseline)",52,"30.57 MiB"],["1.12.7","6.28 ms (+1.3%)",52,"30.57 MiB"],["1.12.7","6.2 ms (+0.0%)",52,"30.57 MiB"],["1.12.7","6.22 ms (+0.4%)",52,"30.57 MiB"],["1.12.7","6.21 ms (+0.2%)",52,"30.57 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>gridspace 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "hₘₐₓ 3D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0106493855220369,0.9980022868874013,1.000662731514902,1.0013258453409757],
  customdata: [["1.12.7","153.0 ns (baseline)",0,"0 B"],["1.12.7","154.6 ns (+1.1%)",0,"0 B"],["1.12.7","152.7 ns (-0.2%)",0,"0 B"],["1.12.7","153.1 ns (+0.1%)",0,"0 B"],["1.12.7","153.2 ns (+0.1%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>hₘₐₓ 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
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
```

### Startup and latency

Time to first `using Bramble` and first operator call — compilation latency, not steady-state performance.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_10" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "TTFX (load + first operator)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [598.631542,654.503208,457.110625,447.726084,452.362375],
  customdata: [["1.12.7","598.63 ms",45,"1.3 KiB"],["1.12.7","654.5 ms",45,"1.3 KiB"],["1.12.7","457.11 ms",45,"1.3 KiB"],["1.12.7","447.73 ms",45,"1.3 KiB"],["1.12.7","452.36 ms",45,"1.3 KiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>TTFX (load + first operator): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "using Bramble",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [501.70775,487.277792,395.600916,388.855333,391.447667],
  customdata: [["1.12.7","501.71 ms",45,"1.3 KiB"],["1.12.7","487.28 ms",45,"1.3 KiB"],["1.12.7","395.6 ms",45,"1.3 KiB"],["1.12.7","388.86 ms",45,"1.3 KiB"],["1.12.7","391.45 ms",45,"1.3 KiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>using Bramble: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
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

### Forms

Linear and bilinear form assembly, across 1D/2D and the `Serial()`/`Parallel()` backends.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div style="display:flex; flex-direction:column; gap:1.5rem; width:100%;">
  <div style="width:100%;"><div id="bench_chart_11" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "assemble! 1D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0150463394448215,1.000620983710628,0.9987132961879269,1.0018198124693103],
  customdata: [["1.12.7","938.8 μs (baseline)",0,"0 B"],["1.12.7","953.0 μs (+1.5%)",0,"0 B"],["1.12.7","939.4 μs (+0.1%)",0,"0 B"],["1.12.7","937.6 μs (-0.1%)",0,"0 B"],["1.12.7","940.5 μs (+0.2%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble! 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! 1D, Parallel() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0114479998458745,1.0133326855598732,1.0069808758846506,1.0832411233396966],
  customdata: [["1.12.7","1.19 ms (baseline)",7,"480 B"],["1.12.7","1.21 ms (+1.1%)",7,"480 B"],["1.12.7","1.21 ms (+1.3%)",7,"480 B"],["1.12.7","1.2 ms (+0.7%)",7,"480 B"],["1.12.7","1.29 ms (+8.3%)",7,"480 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble! 1D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble_parallel! 1D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,0.9451508406118697,1.041935681318264,1.044453759180331,0.9637784184456564],
  customdata: [["1.12.7","1.29 ms (baseline)",7,"480 B"],["1.12.7","1.22 ms (-5.5%)",7,"480 B"],["1.12.7","1.34 ms (+4.2%)",7,"480 B"],["1.12.7","1.35 ms (+4.4%)",7,"480 B"],["1.12.7","1.24 ms (-3.6%)",7,"480 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble_parallel! 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "evaluate! 1D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0108406550714244,0.984801899960271,0.9784285598765237,0.9813220967067121],
  customdata: [["1.12.7","1.14 ms (baseline)",0,"0 B"],["1.12.7","1.15 ms (+1.1%)",0,"0 B"],["1.12.7","1.12 ms (-1.5%)",0,"0 B"],["1.12.7","1.11 ms (-2.2%)",0,"0 B"],["1.12.7","1.12 ms (-1.9%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>evaluate! 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "l(vₕ) 1D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0075918079096045,0.9970755468636825,0.999951334564682,1.0001403375344053],
  customdata: [["1.12.7","883.6 μs (baseline)",0,"0 B"],["1.12.7","890.3 μs (+0.8%)",0,"0 B"],["1.12.7","881.0 μs (-0.3%)",0,"0 B"],["1.12.7","883.5 μs (-0.0%)",0,"0 B"],["1.12.7","883.7 μs (+0.0%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>l(vₕ) 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
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
</div><div style="width:100%;"><div id="bench_chart_12" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "allocate_system_matrix 2D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.1201502186338541,1.0020717135970576,1.0620293764332527,1.0353425733969965],
  customdata: [["1.12.7","2.84 ms (baseline)",21,"15.13 MiB"],["1.12.7","3.18 ms (+12.0%)",21,"15.13 MiB"],["1.12.7","2.84 ms (+0.2%)",21,"15.13 MiB"],["1.12.7","3.01 ms (+6.2%)",21,"15.13 MiB"],["1.12.7","2.94 ms (+3.5%)",21,"15.13 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>allocate_system_matrix 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble (BilinearForm) 2D, Parallel() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,0.9389424435152599,0.9377114436036916,1.0111355406812994,0.9948887678496944],
  customdata: [["1.12.7","5.11 ms (baseline)",35,"15.13 MiB"],["1.12.7","4.8 ms (-6.1%)",35,"15.13 MiB"],["1.12.7","4.79 ms (-6.2%)",35,"15.13 MiB"],["1.12.7","5.17 ms (+1.1%)",35,"15.13 MiB"],["1.12.7","5.09 ms (-0.5%)",35,"15.13 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble (BilinearForm) 2D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble (BilinearForm) 2D, Serial() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0252965369306104,0.9901446389160652,0.998691792509425,0.9931761928722526],
  customdata: [["1.12.7","4.71 ms (baseline)",21,"15.13 MiB"],["1.12.7","4.83 ms (+2.5%)",21,"15.13 MiB"],["1.12.7","4.67 ms (-1.0%)",21,"15.13 MiB"],["1.12.7","4.71 ms (-0.1%)",21,"15.13 MiB"],["1.12.7","4.68 ms (-0.7%)",21,"15.13 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble (BilinearForm) 2D, Serial() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! (matrix) 2D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,0.9931328213306763,0.9567273159754381,0.9574690049848619,0.9580533660225897],
  customdata: [["1.12.7","1.07 ms (baseline)",0,"0 B"],["1.12.7","1.06 ms (-0.7%)",0,"0 B"],["1.12.7","1.02 ms (-4.3%)",0,"0 B"],["1.12.7","1.02 ms (-4.3%)",0,"0 B"],["1.12.7","1.02 ms (-4.2%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble! (matrix) 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! 2D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0225387640157861,0.39797427504845106,0.4152390146996709,1.002954111895058],
  customdata: [["1.12.7","1.18 ms (baseline)",0,"0 B"],["1.12.7","1.21 ms (+2.3%)",0,"0 B"],["1.12.7","471.6 μs (-60.2%)",0,"0 B"],["1.12.7","492.0 μs (-58.5%)",0,"0 B"],["1.12.7","1.19 ms (+0.3%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble! 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble_parallel! 2D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.302873613909273,1.002922896488245,1.0018074160238535,0.9939722719268906],
  customdata: [["1.12.7","1.72 ms (baseline)",7,"496 B"],["1.12.7","2.24 ms (+30.3%)",7,"496 B"],["1.12.7","1.72 ms (+0.3%)",7,"496 B"],["1.12.7","1.72 ms (+0.2%)",7,"496 B"],["1.12.7","1.71 ms (-0.6%)",7,"496 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#06b6d4", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#06b6d4", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble_parallel! 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "form (bilinear, 2D)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0,1.0,0.9803168506961112,0.9803168506961112],
  customdata: [["1.12.7","2.1 ns (baseline)",0,"0 B"],["1.12.7","2.1 ns (baseline)",0,"0 B"],["1.12.7","2.1 ns (baseline)",0,"0 B"],["1.12.7","2.0 ns (-2.0%)",0,"0 B"],["1.12.7","2.0 ns (-2.0%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f97316", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f97316", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>form (bilinear, 2D): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "form (linear, 2D)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0,1.0,0.9803168506961112,0.9803168506961112],
  customdata: [["1.12.7","2.1 ns (baseline)",0,"0 B"],["1.12.7","2.1 ns (baseline)",0,"0 B"],["1.12.7","2.1 ns (baseline)",0,"0 B"],["1.12.7","2.0 ns (-2.0%)",0,"0 B"],["1.12.7","2.0 ns (-2.0%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>form (linear, 2D): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
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
</div>

</div>
```

### Precision 1D

The same 1D workload — restriction, assembly, inner product — repeated in `Float32`, `Float64`, and `Double64`, split by precision since `Double64` (software arithmetic) is an order of magnitude slower.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div style="display:flex; flex-direction:column; gap:1.5rem; width:100%;">
  <div style="width:100%;"><div id="bench_chart_13" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ! Float32",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,0.9989825174825174,0.998979020979021,1.000437062937063,0.9988321678321679],
  customdata: [["1.12.7","286.0 μs (baseline)",0,"0 B"],["1.12.7","285.7 μs (-0.1%)",0,"0 B"],["1.12.7","285.7 μs (-0.1%)",0,"0 B"],["1.12.7","286.1 μs (+0.0%)",0,"0 B"],["1.12.7","285.7 μs (-0.1%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! Float32",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.119748223122538,1.0023411324351983,1.1179958784854134,1.0029299202332722],
  customdata: [["1.12.7","71.3 μs (baseline)",0,"0 B"],["1.12.7","79.9 μs (+12.0%)",0,"0 B"],["1.12.7","71.5 μs (+0.2%)",0,"0 B"],["1.12.7","79.8 μs (+11.8%)",0,"0 B"],["1.12.7","71.5 μs (+0.3%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble! Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! Float32",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0091040787194028,0.9996498191821893,0.9994029806146125,1.00136119174781],
  customdata: [["1.12.7","1.61 ms (baseline)",0,"0 B"],["1.12.7","1.62 ms (+0.9%)",0,"0 B"],["1.12.7","1.61 ms (-0.0%)",0,"0 B"],["1.12.7","1.61 ms (-0.1%)",0,"0 B"],["1.12.7","1.61 ms (+0.1%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ Float32",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0,1.003539364640884,1.003539364640884,1.003539364640884],
  customdata: [["1.12.7","11.6 μs (baseline)",0,"0 B"],["1.12.7","11.6 μs (baseline)",0,"0 B"],["1.12.7","11.6 μs (+0.4%)",0,"0 B"],["1.12.7","11.6 μs (+0.4%)",0,"0 B"],["1.12.7","11.6 μs (+0.4%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>innerₕ Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
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
  name: "Rₕ! Float64",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0007085868461754,1.0,1.0,1.000143080036247],
  customdata: [["1.12.7","293.5 μs (baseline)",0,"0 B"],["1.12.7","293.8 μs (+0.1%)",0,"0 B"],["1.12.7","293.5 μs (baseline)",0,"0 B"],["1.12.7","293.5 μs (baseline)",0,"0 B"],["1.12.7","293.6 μs (+0.0%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! Float64",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0044665189737727,1.0,1.0024893399080492,0.9990114104671384],
  customdata: [["1.12.7","84.0 μs (baseline)",0,"0 B"],["1.12.7","84.3 μs (+0.4%)",0,"0 B"],["1.12.7","84.0 μs (baseline)",0,"0 B"],["1.12.7","84.2 μs (+0.2%)",0,"0 B"],["1.12.7","83.9 μs (-0.1%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble! Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! Float64",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.039595387843113,1.0002670582355868,0.9985196761052483,1.000861750620612],
  customdata: [["1.12.7","1.72 ms (baseline)",0,"0 B"],["1.12.7","1.78 ms (+4.0%)",0,"0 B"],["1.12.7","1.72 ms (+0.0%)",0,"0 B"],["1.12.7","1.71 ms (-0.1%)",0,"0 B"],["1.12.7","1.72 ms (+0.1%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ Float64",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0018129235550568,1.0,1.0,1.0],
  customdata: [["1.12.7","23.2 μs (baseline)",0,"0 B"],["1.12.7","23.2 μs (+0.2%)",0,"0 B"],["1.12.7","23.2 μs (baseline)",0,"0 B"],["1.12.7","23.2 μs (baseline)",0,"0 B"],["1.12.7","23.2 μs (baseline)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>innerₕ Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
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
</div><div style="width:100%;"><div id="bench_chart_15" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ! Double64",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0108921227371217,0.9980802684000839,1.0165024673236878,1.0041937513105472],
  customdata: [["1.12.7","8.94 ms (baseline)",0,"0 B"],["1.12.7","9.04 ms (+1.1%)",0,"0 B"],["1.12.7","8.92 ms (-0.2%)",0,"0 B"],["1.12.7","9.09 ms (+1.7%)",0,"0 B"],["1.12.7","8.98 ms (+0.4%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! Double64",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,0.9757634899651484,0.9847374113688259,0.976445138805432,0.9631053959860594],
  customdata: [["1.12.7","1.04 ms (baseline)",0,"0 B"],["1.12.7","1.01 ms (-2.4%)",0,"0 B"],["1.12.7","1.02 ms (-1.5%)",0,"0 B"],["1.12.7","1.02 ms (-2.4%)",0,"0 B"],["1.12.7","1.0 ms (-3.7%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble! Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! Double64",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0152270976688864,1.0015953961178983,1.003690557211355,0.9999570082730335],
  customdata: [["1.12.7","72.2 ms (baseline)",33,"2.9 KiB"],["1.12.7","73.3 ms (+1.5%)",33,"2.9 KiB"],["1.12.7","72.32 ms (+0.2%)",33,"2.9 KiB"],["1.12.7","72.47 ms (+0.4%)",33,"2.9 KiB"],["1.12.7","72.2 ms (-0.0%)",33,"2.9 KiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ Double64",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1.0,1.0,1.0001571025399811,0.9986284101599248,1.0009021636876765],
  customdata: [["1.12.7","1.06 ms (baseline)",0,"0 B"],["1.12.7","1.06 ms (baseline)",0,"0 B"],["1.12.7","1.06 ms (+0.0%)",0,"0 B"],["1.12.7","1.06 ms (-0.1%)",0,"0 B"],["1.12.7","1.06 ms (+0.1%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>innerₕ Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
  y: [1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.2.1","v2.2.2"],
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
