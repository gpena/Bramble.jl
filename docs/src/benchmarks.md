# Performance and benchmarks

Bramble tracks memory allocations and performance regressions with a dedicated regression suite in `benchmark/benchmarks.jl`.
All measurements below are run on **1,000,000 grid points** per dimension setup (e.g. ``1000 \times 1000`` in 2D, ``100 \times 100 \times 100`` in 3D).

## Recorded baselines

Comparing **3** recorded baselines in chronological order. The earliest run (v2.0.0, `2dec0c7`) is the reference baseline for relative speedup/slowdown calculations.

| Version | Commit | Julia | Summary | File |
|---|---|:---:|---|---|
| v2.0.0 *(baseline)* | `2dec0c7` | `1.12.7` | chore: bump version to 2.0.0 | `baseline_2dec0c7.json` |
| v2.1.0 | `274ae7d` | `1.12.7` | chore: bump version to 2.1.0 | `baseline_274ae7d.json` |
| v2.2.0 | `d7c2416` | `1.12.7` | Merge remote-tracking branch 'origin/main' | `baseline_d7c2416.json` |

## Comparative timings and allocations

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

```@raw html
<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;">
  <div style="flex:1 1 430px; min-width:320px; overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:8px 6px; text-align:left;">Benchmark</th>
<th style="padding:8px 6px; text-align:right;">v2.0.0 <code>2dec0c7</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.1.0 <code>274ae7d</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.2.0 <code>d7c2416</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Dcₓ</code></td>
<td style="padding:7px 6px; text-align:right;">256.5 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">268.7 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">278.6 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+8.6% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋ᵧ</code></td>
<td style="padding:7px 6px; text-align:right;">162.0 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">162.2 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">177.6 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+9.7% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋ₓ</code></td>
<td style="padding:7px 6px; text-align:right;">204.0 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">205.2 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">210.3 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+3.1% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₋ₓ</code></td>
<td style="padding:7px 6px; text-align:right;">172.0 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">161.5 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">180.6 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+5.0% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div id="bench_chart_1" style="width:100%; max-width:560px; height:280px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Dcₓ",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [256.5,268.708,278.5625],
  customdata: [["1.12.7","256.5 μs",3,"7.64 MiB"],["1.12.7","268.7 μs",3,"7.64 MiB"],["1.12.7","278.6 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Dcₓ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "D₋ᵧ",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [161.958,162.209,177.6455],
  customdata: [["1.12.7","162.0 μs",3,"7.64 MiB"],["1.12.7","162.2 μs",3,"7.64 MiB"],["1.12.7","177.6 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>D₋ᵧ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "D₋ₓ",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [204.042,205.208,210.291],
  customdata: [["1.12.7","204.0 μs",3,"7.64 MiB"],["1.12.7","205.2 μs",3,"7.64 MiB"],["1.12.7","210.3 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>D₋ₓ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "M₋ₓ",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [171.959,161.458,180.583],
  customdata: [["1.12.7","172.0 μs",3,"7.64 MiB"],["1.12.7","161.5 μs",3,"7.64 MiB"],["1.12.7","180.6 μs",3,"7.64 MiB"]],
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
    legend: { orientation: 'h', y: -0.3, font: { color: theme.text, size: 11 } },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "μs", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 20, b: 90 },
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
</div>
```

### Operators 3D

```@raw html
<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;">
  <div style="flex:1 1 430px; min-width:320px; overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:8px 6px; text-align:left;">Benchmark</th>
<th style="padding:8px 6px; text-align:right;">v2.0.0 <code>2dec0c7</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.1.0 <code>274ae7d</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.2.0 <code>d7c2416</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋₂</code></td>
<td style="padding:7px 6px; text-align:right;">229.6 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">212.8 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">228.6 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>innerₕ</code></td>
<td style="padding:7px 6px; text-align:right;">240.4 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">240.3 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">240.7 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>∇₋ₕ</code></td>
<td style="padding:7px 6px; text-align:right;">690.8 μs</td>
<td style="padding:7px 6px; text-align:center;">15</td>
<td style="padding:7px 6px; text-align:right;">696.7 μs</td>
<td style="padding:7px 6px; text-align:center;">15</td>
<td style="padding:7px 6px; text-align:right;">685.9 μs</td>
<td style="padding:7px 6px; text-align:center;">15</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.7% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">22.92 MiB</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div id="bench_chart_2" style="width:100%; max-width:560px; height:280px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "D₋₂",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [229.584,212.792,228.583],
  customdata: [["1.12.7","229.6 μs",3,"7.64 MiB"],["1.12.7","212.8 μs",3,"7.64 MiB"],["1.12.7","228.6 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>D₋₂: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [240.375,240.292,240.666],
  customdata: [["1.12.7","240.4 μs",0,"0 B"],["1.12.7","240.3 μs",0,"0 B"],["1.12.7","240.7 μs",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>innerₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "∇₋ₕ",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [690.75,696.667,685.875],
  customdata: [["1.12.7","690.8 μs",15,"22.92 MiB"],["1.12.7","696.7 μs",15,"22.92 MiB"],["1.12.7","685.9 μs",15,"22.92 MiB"]],
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
    legend: { orientation: 'h', y: -0.3, font: { color: theme.text, size: 11 } },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "μs", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 20, b: 90 },
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
```

### Jumps and Averages

```@raw html
<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;">
  <div style="flex:1 1 430px; min-width:320px; overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:8px 6px; text-align:left;">Benchmark</th>
<th style="padding:8px 6px; text-align:right;">v2.0.0 <code>2dec0c7</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.1.0 <code>274ae7d</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.2.0 <code>d7c2416</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₊ᵧ 2D</code></td>
<td style="padding:7px 6px; text-align:right;">161.3 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">161.3 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">161.6 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₊₂ 3D</code></td>
<td style="padding:7px 6px; text-align:right;">227.7 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">226.8 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">228.5 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₊ₓ 2D</code></td>
<td style="padding:7px 6px; text-align:right;">161.6 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">159.6 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">162.4 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.5% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>jumpᵧ 2D</code></td>
<td style="padding:7px 6px; text-align:right;">160.8 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">159.9 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">161.0 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>jump₂ 3D</code></td>
<td style="padding:7px 6px; text-align:right;">227.3 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">226.7 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">227.2 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>jumpₓ 2D</code></td>
<td style="padding:7px 6px; text-align:right;">161.5 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">161.8 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">160.5 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.6% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div id="bench_chart_3" style="width:100%; max-width:560px; height:280px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "M₊ᵧ 2D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [161.292,161.291,161.625],
  customdata: [["1.12.7","161.3 μs",3,"7.64 MiB"],["1.12.7","161.3 μs",3,"7.64 MiB"],["1.12.7","161.6 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>M₊ᵧ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "M₊₂ 3D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [227.708,226.792,228.458],
  customdata: [["1.12.7","227.7 μs",3,"7.64 MiB"],["1.12.7","226.8 μs",3,"7.64 MiB"],["1.12.7","228.5 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>M₊₂ 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "M₊ₓ 2D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [161.625,159.625,162.417],
  customdata: [["1.12.7","161.6 μs",3,"7.64 MiB"],["1.12.7","159.6 μs",3,"7.64 MiB"],["1.12.7","162.4 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>M₊ₓ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jumpᵧ 2D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [160.834,159.917,160.958],
  customdata: [["1.12.7","160.8 μs",3,"7.64 MiB"],["1.12.7","159.9 μs",3,"7.64 MiB"],["1.12.7","161.0 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>jumpᵧ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jump₂ 3D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [227.334,226.667,227.25],
  customdata: [["1.12.7","227.3 μs",3,"7.64 MiB"],["1.12.7","226.7 μs",3,"7.64 MiB"],["1.12.7","227.2 μs",3,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>jump₂ 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jumpₓ 2D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [161.458,161.834,160.5],
  customdata: [["1.12.7","161.5 μs",3,"7.64 MiB"],["1.12.7","161.8 μs",3,"7.64 MiB"],["1.12.7","160.5 μs",3,"7.64 MiB"]],
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
    legend: { orientation: 'h', y: -0.3, font: { color: theme.text, size: 11 } },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "μs", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 20, b: 90 },
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
</div>
```

### Inner Products 2D

```@raw html
<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;">
  <div style="flex:1 1 430px; min-width:320px; overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:8px 6px; text-align:left;">Benchmark</th>
<th style="padding:8px 6px; text-align:right;">v2.0.0 <code>2dec0c7</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.1.0 <code>274ae7d</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.2.0 <code>d7c2416</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>innerₕ</code></td>
<td style="padding:7px 6px; text-align:right;">240.4 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">238.0 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">237.0 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-1.4% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>norm₁ₕ</code></td>
<td style="padding:7px 6px; text-align:right;">789.8 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">788.5 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">785.7 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.5% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>normₕ</code></td>
<td style="padding:7px 6px; text-align:right;">189.3 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">186.2 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">184.4 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-2.6% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>snorm₁ₕ</code></td>
<td style="padding:7px 6px; text-align:right;">578.5 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">582.4 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">578.2 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div id="bench_chart_4" style="width:100%; max-width:560px; height:280px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "innerₕ",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [240.375,238.041,236.959],
  customdata: [["1.12.7","240.4 μs",0,"0 B"],["1.12.7","238.0 μs",0,"0 B"],["1.12.7","237.0 μs",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>innerₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "norm₁ₕ",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [789.791,788.458,785.667],
  customdata: [["1.12.7","789.8 μs",0,"0 B"],["1.12.7","788.5 μs",0,"0 B"],["1.12.7","785.7 μs",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>norm₁ₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "normₕ",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [189.292,186.167,184.417],
  customdata: [["1.12.7","189.3 μs",0,"0 B"],["1.12.7","186.2 μs",0,"0 B"],["1.12.7","184.4 μs",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>normₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "snorm₁ₕ",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [578.5,582.375,578.209],
  customdata: [["1.12.7","578.5 μs",0,"0 B"],["1.12.7","582.4 μs",0,"0 B"],["1.12.7","578.2 μs",0,"0 B"]],
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
    legend: { orientation: 'h', y: -0.3, font: { color: theme.text, size: 11 } },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "μs", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 20, b: 90 },
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
</div>
```

### Restriction

```@raw html
<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;">
  <div style="flex:1 1 430px; min-width:320px; overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:8px 6px; text-align:left;">Benchmark</th>
<th style="padding:8px 6px; text-align:right;">v2.0.0 <code>2dec0c7</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.1.0 <code>274ae7d</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.2.0 <code>d7c2416</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ 1D (allocates its output)</code></td>
<td style="padding:7px 6px; text-align:right;">3.2 ms</td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:right;">3.24 ms</td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:right;">3.38 ms</td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+5.8% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">3.19 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">3.27 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">448 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 1D, Parallel() backend</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">3.21 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">448 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 1D, Serial() backend (default)</code></td>
<td style="padding:7px 6px; text-align:right;">2.95 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">3.03 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">2.97 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.7% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 2D</code></td>
<td style="padding:7px 6px; text-align:right;">3.82 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">3.91 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">448 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 2D, Parallel() backend</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">3.83 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">448 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 3D</code></td>
<td style="padding:7px 6px; text-align:right;">4.44 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">4.58 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">464 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 3D, Parallel() backend</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">4.55 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">464 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">16.74 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">16.85 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">544 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 1D, Parallel() backend</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">16.67 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">544 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 1D, Serial() backend (default)</code></td>
<td style="padding:7px 6px; text-align:right;">17.32 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">17.24 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">17.26 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 2D</code></td>
<td style="padding:7px 6px; text-align:right;">106.39 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">106.07 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">560 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 2D, Parallel() backend</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">106.72 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">560 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 3D</code></td>
<td style="padding:7px 6px; text-align:right;">620.75 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">659.64 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">576 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 3D, Parallel() backend</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">622.73 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">576 B</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="display:flex; flex-wrap:wrap; gap:1rem;">
  <div style="flex: 0 0 380px;"><div id="bench_chart_5" style="width:100%; max-width:380px; height:280px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ 1D (allocates its output)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0149558149254283,1.0576855523560704],
  customdata: [["1.12.7","3.2 ms (baseline)",10,"7.64 MiB"],["1.12.7","3.24 ms (+1.5%)",10,"7.64 MiB"],["1.12.7","3.38 ms (+5.8%)",10,"7.64 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ 1D (allocates its output): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 1D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)"],
  y: [1.0,1.024612615432775],
  customdata: [["1.12.7","3.19 ms (baseline)",7,"448 B"],["1.12.7","3.27 ms (+2.5%)",7,"448 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 1D, Parallel() backend",
  x: ["v2.2.0 (d7c2416)"],
  y: [1.0],
  customdata: [["1.12.7","3.21 ms (baseline)",7,"448 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! 1D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 1D, Serial() backend (default)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0301685765215176,1.0069036584330702],
  customdata: [["1.12.7","2.95 ms (baseline)",0,"0 B"],["1.12.7","3.03 ms (+3.0%)",0,"0 B"],["1.12.7","2.97 ms (+0.7%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! 1D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 1D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)"],
  y: [1.0,1.0068065935214991],
  customdata: [["1.12.7","16.74 ms (baseline)",7,"544 B"],["1.12.7","16.85 ms (+0.7%)",7,"544 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 1D, Parallel() backend",
  x: ["v2.2.0 (d7c2416)"],
  y: [1.0],
  customdata: [["1.12.7","16.67 ms (baseline)",7,"544 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#06b6d4", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#06b6d4", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! 1D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 1D, Serial() backend (default)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,0.994873716820243,0.9963419560811928],
  customdata: [["1.12.7","17.32 ms (baseline)",0,"0 B"],["1.12.7","17.24 ms (-0.5%)",0,"0 B"],["1.12.7","17.26 ms (-0.4%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f97316", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f97316", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! 1D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: { orientation: 'h', y: -0.3, font: { color: theme.text, size: 11 } },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 20, b: 90 },
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
</div><div style="flex: 0 0 380px;"><div id="bench_chart_6" style="width:100%; max-width:380px; height:280px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ! 2D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)"],
  y: [1.0,1.0244596787155744],
  customdata: [["1.12.7","3.82 ms (baseline)",7,"448 B"],["1.12.7","3.91 ms (+2.4%)",7,"448 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 2D, Parallel() backend",
  x: ["v2.2.0 (d7c2416)"],
  y: [1.0],
  customdata: [["1.12.7","3.83 ms (baseline)",7,"448 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! 2D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 2D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)"],
  y: [1.0,0.996973378521717],
  customdata: [["1.12.7","106.39 ms (baseline)",7,"560 B"],["1.12.7","106.07 ms (-0.3%)",7,"560 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 2D, Parallel() backend",
  x: ["v2.2.0 (d7c2416)"],
  y: [1.0],
  customdata: [["1.12.7","106.72 ms (baseline)",7,"560 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! 2D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: { orientation: 'h', y: -0.3, font: { color: theme.text, size: 11 } },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 20, b: 90 },
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
</div><div style="flex: 0 0 380px;"><div id="bench_chart_7" style="width:100%; max-width:380px; height:280px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ! 3D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)"],
  y: [1.0,1.031481808717184],
  customdata: [["1.12.7","4.44 ms (baseline)",7,"464 B"],["1.12.7","4.58 ms (+3.1%)",7,"464 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 3D, Parallel() backend",
  x: ["v2.2.0 (d7c2416)"],
  y: [1.0],
  customdata: [["1.12.7","4.55 ms (baseline)",7,"464 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! 3D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 3D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)"],
  y: [1.0,1.0626395001054676],
  customdata: [["1.12.7","620.75 ms (baseline)",7,"576 B"],["1.12.7","659.64 ms (+6.3%)",7,"576 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 3D, Parallel() backend",
  x: ["v2.2.0 (d7c2416)"],
  y: [1.0],
  customdata: [["1.12.7","622.73 ms (baseline)",7,"576 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! 3D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: { orientation: 'h', y: -0.3, font: { color: theme.text, size: 11 } },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 20, b: 90 },
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
</div>
```

### Composite

```@raw html
<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;">
  <div style="flex:1 1 430px; min-width:320px; overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:8px 6px; text-align:left;">Benchmark</th>
<th style="padding:8px 6px; text-align:right;">v2.0.0 <code>2dec0c7</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.1.0 <code>274ae7d</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.2.0 <code>d7c2416</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋ₓ (3 components)</code></td>
<td style="padding:7px 6px; text-align:right;">670.2 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">694.0 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">659.9 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-1.5% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">22.89 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>∇₋ₕ (3 components)</code></td>
<td style="padding:7px 6px; text-align:right;">1.38 ms</td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:right;">1.41 ms</td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:right;">1.4 ms</td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+1.1% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">45.78 MiB</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div id="bench_chart_8" style="width:100%; max-width:560px; height:280px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "D₋ₓ (3 components)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [0.670208,0.693958,0.659917],
  customdata: [["1.12.7","670.2 μs",3,"22.89 MiB"],["1.12.7","694.0 μs",3,"22.89 MiB"],["1.12.7","659.9 μs",3,"22.89 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>D₋ₓ (3 components): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "∇₋ₕ (3 components)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.380625,1.407666,1.395312],
  customdata: [["1.12.7","1.38 ms",10,"45.78 MiB"],["1.12.7","1.41 ms",10,"45.78 MiB"],["1.12.7","1.4 ms",10,"45.78 MiB"]],
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
    legend: { orientation: 'h', y: -0.3, font: { color: theme.text, size: 11 } },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "ms", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 20, b: 90 },
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
</div>
```

### Construction

```@raw html
<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;">
  <div style="flex:1 1 430px; min-width:320px; overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:8px 6px; text-align:left;">Benchmark</th>
<th style="padding:8px 6px; text-align:right;">v2.0.0 <code>2dec0c7</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.1.0 <code>274ae7d</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.2.0 <code>d7c2416</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>gridspace 2D</code></td>
<td style="padding:7px 6px; text-align:right;">2.22 ms</td>
<td style="padding:7px 6px; text-align:center;">42</td>
<td style="padding:7px 6px; text-align:right;">2.24 ms</td>
<td style="padding:7px 6px; text-align:center;">42</td>
<td style="padding:7px 6px; text-align:right;">2.23 ms</td>
<td style="padding:7px 6px; text-align:center;">42</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">22.95 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>gridspace 3D</code></td>
<td style="padding:7px 6px; text-align:right;">6.2 ms</td>
<td style="padding:7px 6px; text-align:center;">52</td>
<td style="padding:7px 6px; text-align:right;">6.28 ms</td>
<td style="padding:7px 6px; text-align:center;">52</td>
<td style="padding:7px 6px; text-align:right;">6.2 ms</td>
<td style="padding:7px 6px; text-align:center;">52</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">30.57 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>hₘₐₓ 3D</code></td>
<td style="padding:7px 6px; text-align:right;">153.0 ns</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">154.6 ns</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">152.7 ns</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div id="bench_chart_9" style="width:100%; max-width:560px; height:280px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "gridspace 2D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0100427623813584,1.0026278231190768],
  customdata: [["1.12.7","2.22 ms (baseline)",42,"22.95 MiB"],["1.12.7","2.24 ms (+1.0%)",42,"22.95 MiB"],["1.12.7","2.23 ms (+0.3%)",42,"22.95 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>gridspace 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "gridspace 3D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0131330029173518,1.000413423237967],
  customdata: [["1.12.7","6.2 ms (baseline)",52,"30.57 MiB"],["1.12.7","6.28 ms (+1.3%)",52,"30.57 MiB"],["1.12.7","6.2 ms (+0.0%)",52,"30.57 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>gridspace 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "hₘₐₓ 3D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0106493855220369,0.9980022868874013],
  customdata: [["1.12.7","153.0 ns (baseline)",0,"0 B"],["1.12.7","154.6 ns (+1.1%)",0,"0 B"],["1.12.7","152.7 ns (-0.2%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>hₘₐₓ 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: { orientation: 'h', y: -0.3, font: { color: theme.text, size: 11 } },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 20, b: 90 },
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
```

### Startup and Latency

```@raw html
<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;">
  <div style="flex:1 1 430px; min-width:320px; overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:8px 6px; text-align:left;">Benchmark</th>
<th style="padding:8px 6px; text-align:right;">v2.0.0 <code>2dec0c7</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.1.0 <code>274ae7d</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.2.0 <code>d7c2416</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>TTFX (load + first operator)</code></td>
<td style="padding:7px 6px; text-align:right;">598.63 ms</td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:right;">654.5 ms</td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:right;">457.11 ms</td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-23.6% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">1.3 KiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>using Bramble</code></td>
<td style="padding:7px 6px; text-align:right;">501.71 ms</td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:right;">487.28 ms</td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:right;">395.6 ms</td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-21.1% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">1.3 KiB</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div id="bench_chart_10" style="width:100%; max-width:560px; height:280px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "TTFX (load + first operator)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [598.631542,654.503208,457.110625],
  customdata: [["1.12.7","598.63 ms",45,"1.3 KiB"],["1.12.7","654.5 ms",45,"1.3 KiB"],["1.12.7","457.11 ms",45,"1.3 KiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>TTFX (load + first operator): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "using Bramble",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [501.70775,487.277792,395.600916],
  customdata: [["1.12.7","501.71 ms",45,"1.3 KiB"],["1.12.7","487.28 ms",45,"1.3 KiB"],["1.12.7","395.6 ms",45,"1.3 KiB"]],
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
    legend: { orientation: 'h', y: -0.3, font: { color: theme.text, size: 11 } },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "ms", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 20, b: 90 },
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
</div>
```

### Forms

```@raw html
<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;">
  <div style="flex:1 1 430px; min-width:320px; overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:8px 6px; text-align:left;">Benchmark</th>
<th style="padding:8px 6px; text-align:right;">v2.0.0 <code>2dec0c7</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.1.0 <code>274ae7d</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.2.0 <code>d7c2416</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>allocate_system_matrix 2D</code></td>
<td style="padding:7px 6px; text-align:right;">2.84 ms</td>
<td style="padding:7px 6px; text-align:center;">21</td>
<td style="padding:7px 6px; text-align:right;">3.18 ms</td>
<td style="padding:7px 6px; text-align:center;">21</td>
<td style="padding:7px 6px; text-align:right;">2.84 ms</td>
<td style="padding:7px 6px; text-align:center;">21</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">15.13 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble (BilinearForm) 2D, Parallel() backend</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">4.79 ms</td>
<td style="padding:7px 6px; text-align:center;">35</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">15.13 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble (BilinearForm) 2D, Serial() backend</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">4.67 ms</td>
<td style="padding:7px 6px; text-align:center;">21</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">15.13 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble (BilinearForm), Parallel() backend</code></td>
<td style="padding:7px 6px; text-align:right;">5.11 ms</td>
<td style="padding:7px 6px; text-align:center;">35</td>
<td style="padding:7px 6px; text-align:right;">4.8 ms</td>
<td style="padding:7px 6px; text-align:center;">35</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">15.13 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble (BilinearForm), Serial() backend</code></td>
<td style="padding:7px 6px; text-align:right;">4.71 ms</td>
<td style="padding:7px 6px; text-align:center;">21</td>
<td style="padding:7px 6px; text-align:right;">4.83 ms</td>
<td style="padding:7px 6px; text-align:center;">21</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">15.13 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! (matrix) 2D</code></td>
<td style="padding:7px 6px; text-align:right;">1.07 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.06 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.02 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-4.3% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">938.8 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">953.0 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">939.4 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! 1D, Parallel() backend</code></td>
<td style="padding:7px 6px; text-align:right;">1.19 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">1.21 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">1.21 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+1.3% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">480 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! 2D</code></td>
<td style="padding:7px 6px; text-align:right;">1.18 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.21 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">471.6 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-60.2% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble_parallel! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">1.29 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">1.22 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">1.34 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+4.2% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">480 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble_parallel! 2D</code></td>
<td style="padding:7px 6px; text-align:right;">1.72 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">2.24 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">1.72 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">496 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>evaluate! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">1.14 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.15 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.12 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-1.5% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>form (bilinear, 2D)</code></td>
<td style="padding:7px 6px; text-align:right;">2.1 ns</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">2.1 ns</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">2.1 ns</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>form (linear, 2D)</code></td>
<td style="padding:7px 6px; text-align:right;">2.1 ns</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">2.1 ns</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">2.1 ns</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>l(vₕ) 1D</code></td>
<td style="padding:7px 6px; text-align:right;">883.6 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">890.3 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">881.0 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="display:flex; flex-wrap:wrap; gap:1rem;">
  <div style="flex: 0 0 380px;"><div id="bench_chart_11" style="width:100%; max-width:380px; height:280px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "allocate_system_matrix 2D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.1201502186338541,1.0020717135970576],
  customdata: [["1.12.7","2.84 ms (baseline)",21,"15.13 MiB"],["1.12.7","3.18 ms (+12.0%)",21,"15.13 MiB"],["1.12.7","2.84 ms (+0.2%)",21,"15.13 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>allocate_system_matrix 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble (BilinearForm) 2D, Parallel() backend",
  x: ["v2.2.0 (d7c2416)"],
  y: [1.0],
  customdata: [["1.12.7","4.79 ms (baseline)",35,"15.13 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble (BilinearForm) 2D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble (BilinearForm) 2D, Serial() backend",
  x: ["v2.2.0 (d7c2416)"],
  y: [1.0],
  customdata: [["1.12.7","4.67 ms (baseline)",21,"15.13 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble (BilinearForm) 2D, Serial() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble (BilinearForm), Parallel() backend",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)"],
  y: [1.0,0.9389424435152599],
  customdata: [["1.12.7","5.11 ms (baseline)",35,"15.13 MiB"],["1.12.7","4.8 ms (-6.1%)",35,"15.13 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble (BilinearForm), Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble (BilinearForm), Serial() backend",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)"],
  y: [1.0,1.0252965369306104],
  customdata: [["1.12.7","4.71 ms (baseline)",21,"15.13 MiB"],["1.12.7","4.83 ms (+2.5%)",21,"15.13 MiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble (BilinearForm), Serial() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! (matrix) 2D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,0.9931328213306763,0.9567273159754381],
  customdata: [["1.12.7","1.07 ms (baseline)",0,"0 B"],["1.12.7","1.06 ms (-0.7%)",0,"0 B"],["1.12.7","1.02 ms (-4.3%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#06b6d4", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#06b6d4", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble! (matrix) 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! 1D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0150463394448215,1.000620983710628],
  customdata: [["1.12.7","938.8 μs (baseline)",0,"0 B"],["1.12.7","953.0 μs (+1.5%)",0,"0 B"],["1.12.7","939.4 μs (+0.1%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f97316", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f97316", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble! 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! 1D, Parallel() backend",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0114479998458745,1.0133326855598732],
  customdata: [["1.12.7","1.19 ms (baseline)",7,"480 B"],["1.12.7","1.21 ms (+1.1%)",7,"480 B"],["1.12.7","1.21 ms (+1.3%)",7,"480 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble! 1D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: { orientation: 'h', y: -0.3, font: { color: theme.text, size: 11 } },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 20, b: 90 },
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
</div><div style="flex: 0 0 380px;"><div id="bench_chart_12" style="width:100%; max-width:380px; height:280px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "assemble! 2D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0225387640157861,0.39797427504845106],
  customdata: [["1.12.7","1.18 ms (baseline)",0,"0 B"],["1.12.7","1.21 ms (+2.3%)",0,"0 B"],["1.12.7","471.6 μs (-60.2%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble! 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble_parallel! 1D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,0.9451508406118697,1.041935681318264],
  customdata: [["1.12.7","1.29 ms (baseline)",7,"480 B"],["1.12.7","1.22 ms (-5.5%)",7,"480 B"],["1.12.7","1.34 ms (+4.2%)",7,"480 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble_parallel! 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble_parallel! 2D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.302873613909273,1.002922896488245],
  customdata: [["1.12.7","1.72 ms (baseline)",7,"496 B"],["1.12.7","2.24 ms (+30.3%)",7,"496 B"],["1.12.7","1.72 ms (+0.3%)",7,"496 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble_parallel! 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "evaluate! 1D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0108406550714244,0.984801899960271],
  customdata: [["1.12.7","1.14 ms (baseline)",0,"0 B"],["1.12.7","1.15 ms (+1.1%)",0,"0 B"],["1.12.7","1.12 ms (-1.5%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>evaluate! 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "form (bilinear, 2D)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0,1.0],
  customdata: [["1.12.7","2.1 ns (baseline)",0,"0 B"],["1.12.7","2.1 ns (baseline)",0,"0 B"],["1.12.7","2.1 ns (baseline)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>form (bilinear, 2D): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "form (linear, 2D)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0,1.0],
  customdata: [["1.12.7","2.1 ns (baseline)",0,"0 B"],["1.12.7","2.1 ns (baseline)",0,"0 B"],["1.12.7","2.1 ns (baseline)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#06b6d4", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#06b6d4", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>form (linear, 2D): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "l(vₕ) 1D",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0075918079096045,0.9970755468636825],
  customdata: [["1.12.7","883.6 μs (baseline)",0,"0 B"],["1.12.7","890.3 μs (+0.8%)",0,"0 B"],["1.12.7","881.0 μs (-0.3%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f97316", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f97316", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>l(vₕ) 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: { orientation: 'h', y: -0.3, font: { color: theme.text, size: 11 } },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 20, b: 90 },
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
</div>
```

### Precision 1D

```@raw html
<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;">
  <div style="flex:1 1 430px; min-width:320px; overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:8px 6px; text-align:left;">Benchmark</th>
<th style="padding:8px 6px; text-align:right;">v2.0.0 <code>2dec0c7</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.1.0 <code>274ae7d</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">v2.2.0 <code>d7c2416</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! Double64</code></td>
<td style="padding:7px 6px; text-align:right;">8.94 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">9.04 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">8.92 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! Float32</code></td>
<td style="padding:7px 6px; text-align:right;">286.0 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">285.7 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">285.7 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! Float64</code></td>
<td style="padding:7px 6px; text-align:right;">293.5 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">293.8 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">293.5 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! Double64</code></td>
<td style="padding:7px 6px; text-align:right;">1.04 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.01 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.02 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-1.5% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! Float32</code></td>
<td style="padding:7px 6px; text-align:right;">71.3 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">79.9 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">71.5 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! Float64</code></td>
<td style="padding:7px 6px; text-align:right;">84.0 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">84.3 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">84.0 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! Double64</code></td>
<td style="padding:7px 6px; text-align:right;">72.2 ms</td>
<td style="padding:7px 6px; text-align:center;">33</td>
<td style="padding:7px 6px; text-align:right;">73.3 ms</td>
<td style="padding:7px 6px; text-align:center;">33</td>
<td style="padding:7px 6px; text-align:right;">72.32 ms</td>
<td style="padding:7px 6px; text-align:center;">33</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">2.9 KiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! Float32</code></td>
<td style="padding:7px 6px; text-align:right;">1.61 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.62 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.61 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! Float64</code></td>
<td style="padding:7px 6px; text-align:right;">1.72 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.78 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.72 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>innerₕ Double64</code></td>
<td style="padding:7px 6px; text-align:right;">1.06 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.06 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.06 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>innerₕ Float32</code></td>
<td style="padding:7px 6px; text-align:right;">11.6 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">11.6 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">11.6 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>innerₕ Float64</code></td>
<td style="padding:7px 6px; text-align:right;">23.2 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">23.2 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">23.2 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="display:flex; flex-wrap:wrap; gap:1rem;">
  <div style="flex: 0 0 380px;"><div id="bench_chart_13" style="width:100%; max-width:380px; height:280px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ! Float32",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,0.9989825174825174,0.998979020979021],
  customdata: [["1.12.7","286.0 μs (baseline)",0,"0 B"],["1.12.7","285.7 μs (-0.1%)",0,"0 B"],["1.12.7","285.7 μs (-0.1%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! Float32",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.119748223122538,1.0023411324351983],
  customdata: [["1.12.7","71.3 μs (baseline)",0,"0 B"],["1.12.7","79.9 μs (+12.0%)",0,"0 B"],["1.12.7","71.5 μs (+0.2%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble! Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! Float32",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0091040787194028,0.9996498191821893],
  customdata: [["1.12.7","1.61 ms (baseline)",0,"0 B"],["1.12.7","1.62 ms (+0.9%)",0,"0 B"],["1.12.7","1.61 ms (-0.0%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ Float32",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0,1.003539364640884],
  customdata: [["1.12.7","11.6 μs (baseline)",0,"0 B"],["1.12.7","11.6 μs (baseline)",0,"0 B"],["1.12.7","11.6 μs (+0.4%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>innerₕ Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: { orientation: 'h', y: -0.3, font: { color: theme.text, size: 11 } },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 20, b: 90 },
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
</div><div style="flex: 0 0 380px;"><div id="bench_chart_14" style="width:100%; max-width:380px; height:280px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ! Float64",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0007085868461754,1.0],
  customdata: [["1.12.7","293.5 μs (baseline)",0,"0 B"],["1.12.7","293.8 μs (+0.1%)",0,"0 B"],["1.12.7","293.5 μs (baseline)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! Float64",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0044665189737727,1.0],
  customdata: [["1.12.7","84.0 μs (baseline)",0,"0 B"],["1.12.7","84.3 μs (+0.4%)",0,"0 B"],["1.12.7","84.0 μs (baseline)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble! Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! Float64",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.039595387843113,1.0002670582355868],
  customdata: [["1.12.7","1.72 ms (baseline)",0,"0 B"],["1.12.7","1.78 ms (+4.0%)",0,"0 B"],["1.12.7","1.72 ms (+0.0%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ Float64",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0018129235550568,1.0],
  customdata: [["1.12.7","23.2 μs (baseline)",0,"0 B"],["1.12.7","23.2 μs (+0.2%)",0,"0 B"],["1.12.7","23.2 μs (baseline)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>innerₕ Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: { orientation: 'h', y: -0.3, font: { color: theme.text, size: 11 } },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 20, b: 90 },
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
</div><div style="flex: 0 0 380px;"><div id="bench_chart_15" style="width:100%; max-width:380px; height:280px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "Rₕ! Double64",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0108921227371217,0.9980802684000839],
  customdata: [["1.12.7","8.94 ms (baseline)",0,"0 B"],["1.12.7","9.04 ms (+1.1%)",0,"0 B"],["1.12.7","8.92 ms (-0.2%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>Rₕ! Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! Double64",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,0.9757634899651484,0.9847374113688259],
  customdata: [["1.12.7","1.04 ms (baseline)",0,"0 B"],["1.12.7","1.01 ms (-2.4%)",0,"0 B"],["1.12.7","1.02 ms (-1.5%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>assemble! Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! Double64",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0152270976688864,1.0015953961178983],
  customdata: [["1.12.7","72.2 ms (baseline)",33,"2.9 KiB"],["1.12.7","73.3 ms (+1.5%)",33,"2.9 KiB"],["1.12.7","72.32 ms (+0.2%)",33,"2.9 KiB"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>avgₕ! Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ Double64",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1.0,1.0,1.0001571025399811],
  customdata: [["1.12.7","1.06 ms (baseline)",0,"0 B"],["1.12.7","1.06 ms (baseline)",0,"0 B"],["1.12.7","1.06 ms (+0.0%)",0,"0 B"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]})<br>innerₕ Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
  y: [1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: { orientation: 'h', y: -0.3, font: { color: theme.text, size: 11 } },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"],
      tickangle: -45, color: theme.text, gridcolor: theme.grid,
      tickfont: { family: 'monospace', size: 10 },
    },
    yaxis: {
      title: { text: "relative to baseline", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    margin: { t: 20, l: 60, r: 20, b: 90 },
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
</div>
```

## How to add new benchmark runs

To record performance on a new commit or after an optimization pass, run:

```bash
julia --project=benchmark benchmark/benchmarks.jl --save benchmark/baselines/baseline_$(git rev-parse --short HEAD).json
```

Rebuilding the documentation (`julia -e 'using Pkg; Pkg.activate("docs"); include("docs/make.jl")'`) will automatically discover all `baseline_*.json` files and append new comparison columns, delta calculations, and charts.
