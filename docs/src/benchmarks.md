# Performance and benchmarks

Bramble tracks memory allocations and performance regressions with a dedicated regression suite in `benchmark/benchmarks.jl`.
All measurements below are run on **1,000,000 grid points** per dimension setup (e.g. $1000 \times 1000$ in 2D, $100 \times 100 \times 100$ in 3D).

## Comparative timings and allocations

Each chart below tracks one benchmark group across all **19** recorded baselines, in chronological release order, against the earliest run (v2.0.0) as the reference. Where a group's operations span more than a 20× range, the y-axis shows time relative to that reference instead of absolute time, so a cheap operation isn't flattened onto the same line as an expensive one. Hover any point for its exact time, Julia version, thread count, allocation count, and memory.

!!! note "Thread count changes at v2.9.0"
    Baselines before v2.9.0 were recorded with 1 thread; from v2.9.0 onward, 4 threads. Entries on the `Parallel()` backend are not comparable across that line, and the charts mark it with a dotted rule: at one thread the threaded code path runs its serial branch, so those entries measured task-spawn overhead rather than parallelism. Serial entries are unaffected.

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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [0.2565,0.268708,0.2785625,0.289292,0.26225,0.257417,0.254625,0.2685,0.256083,0.255625,0.253959,0.272209,0.257375,0.254542,0.263917,0.33675,0.255,0.25375,0.255333],
  customdata: [["1.12.7","256.5 μs",3,"7.64 MiB","1"],["1.12.7","268.7 μs",3,"7.64 MiB","1"],["1.12.7","278.6 μs",3,"7.64 MiB","1"],["1.12.7","289.3 μs",3,"7.64 MiB","1"],["1.12.7","262.2 μs",3,"7.64 MiB","1"],["1.12.7","257.4 μs",3,"7.64 MiB","1"],["1.12.7","254.6 μs",3,"7.64 MiB","1"],["1.12.7","268.5 μs",3,"7.64 MiB","1"],["1.12.7","256.1 μs",3,"7.64 MiB","1"],["1.13.0","255.6 μs",3,"7.64 MiB","4"],["1.13.0","254.0 μs",3,"7.64 MiB","4"],["1.13.0","272.2 μs",3,"7.64 MiB","4"],["1.13.0","257.4 μs",3,"7.64 MiB","4"],["1.13.0","254.5 μs",3,"7.64 MiB","4"],["1.13.0","263.9 μs",3,"7.64 MiB","4"],["1.13.0","336.8 μs",3,"7.64 MiB","4"],["1.13.0","255.0 μs",3,"7.64 MiB","4"],["1.13.0","253.8 μs",3,"7.64 MiB","4"],["1.13.0","255.3 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Dcₓ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "D₋(uₕ, d) over d",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [0.457958,0.367041,0.365917,0.367416],
  customdata: [["1.13.0","458.0 μs",6,"15.28 MiB","4"],["1.13.0","367.0 μs",6,"15.28 MiB","4"],["1.13.0","365.9 μs",6,"15.28 MiB","4"],["1.13.0","367.4 μs",6,"15.28 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>D₋(uₕ, d) over d: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "D₋ᵧ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [0.161958,0.162209,0.1776455,0.177709,0.159125,0.161916,0.16175,0.165125,0.161334,0.154083,0.162,0.1638955,0.162375,0.161542,0.160125,0.202125,0.162667,0.161875,0.162042],
  customdata: [["1.12.7","162.0 μs",3,"7.64 MiB","1"],["1.12.7","162.2 μs",3,"7.64 MiB","1"],["1.12.7","177.6 μs",3,"7.64 MiB","1"],["1.12.7","177.7 μs",3,"7.64 MiB","1"],["1.12.7","159.1 μs",3,"7.64 MiB","1"],["1.12.7","161.9 μs",3,"7.64 MiB","1"],["1.12.7","161.8 μs",3,"7.64 MiB","1"],["1.12.7","165.1 μs",3,"7.64 MiB","1"],["1.12.7","161.3 μs",3,"7.64 MiB","1"],["1.13.0","154.1 μs",3,"7.64 MiB","4"],["1.13.0","162.0 μs",3,"7.64 MiB","4"],["1.13.0","163.9 μs",3,"7.64 MiB","4"],["1.13.0","162.4 μs",3,"7.64 MiB","4"],["1.13.0","161.5 μs",3,"7.64 MiB","4"],["1.13.0","160.1 μs",3,"7.64 MiB","4"],["1.13.0","202.1 μs",3,"7.64 MiB","4"],["1.13.0","162.7 μs",3,"7.64 MiB","4"],["1.13.0","161.9 μs",3,"7.64 MiB","4"],["1.13.0","162.0 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>D₋ᵧ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "D₋ₓ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [0.204042,0.205208,0.210291,0.216458,0.203583,0.203709,0.203916,0.204625,0.2032295,0.204167,0.203333,0.210208,0.204917,0.204,0.204458,0.2245,0.204459,0.203208,0.203209],
  customdata: [["1.12.7","204.0 μs",3,"7.64 MiB","1"],["1.12.7","205.2 μs",3,"7.64 MiB","1"],["1.12.7","210.3 μs",3,"7.64 MiB","1"],["1.12.7","216.5 μs",3,"7.64 MiB","1"],["1.12.7","203.6 μs",3,"7.64 MiB","1"],["1.12.7","203.7 μs",3,"7.64 MiB","1"],["1.12.7","203.9 μs",3,"7.64 MiB","1"],["1.12.7","204.6 μs",3,"7.64 MiB","1"],["1.12.7","203.2 μs",3,"7.64 MiB","1"],["1.13.0","204.2 μs",3,"7.64 MiB","4"],["1.13.0","203.3 μs",3,"7.64 MiB","4"],["1.13.0","210.2 μs",3,"7.64 MiB","4"],["1.13.0","204.9 μs",3,"7.64 MiB","4"],["1.13.0","204.0 μs",3,"7.64 MiB","4"],["1.13.0","204.5 μs",3,"7.64 MiB","4"],["1.13.0","224.5 μs",3,"7.64 MiB","4"],["1.13.0","204.5 μs",3,"7.64 MiB","4"],["1.13.0","203.2 μs",3,"7.64 MiB","4"],["1.13.0","203.2 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>D₋ₓ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Mₓ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [0.171959,0.161458,0.180583,0.179917,0.1675,0.171791,0.171458,0.173584,0.1715,0.1604375,0.171583,0.17225,0.173584,0.171291,0.169125,0.213334,0.171833,0.170875,0.171334],
  customdata: [["1.12.7","172.0 μs",3,"7.64 MiB","1"],["1.12.7","161.5 μs",3,"7.64 MiB","1"],["1.12.7","180.6 μs",3,"7.64 MiB","1"],["1.12.7","179.9 μs",3,"7.64 MiB","1"],["1.12.7","167.5 μs",3,"7.64 MiB","1"],["1.12.7","171.8 μs",3,"7.64 MiB","1"],["1.12.7","171.5 μs",3,"7.64 MiB","1"],["1.12.7","173.6 μs",3,"7.64 MiB","1"],["1.12.7","171.5 μs",3,"7.64 MiB","1"],["1.13.0","160.4 μs",3,"7.64 MiB","4"],["1.13.0","171.6 μs",3,"7.64 MiB","4"],["1.13.0","172.2 μs",3,"7.64 MiB","4"],["1.13.0","173.6 μs",3,"7.64 MiB","4"],["1.13.0","171.3 μs",3,"7.64 MiB","4"],["1.13.0","169.1 μs",3,"7.64 MiB","4"],["1.13.0","213.3 μs",3,"7.64 MiB","4"],["1.13.0","171.8 μs",3,"7.64 MiB","4"],["1.13.0","170.9 μs",3,"7.64 MiB","4"],["1.13.0","171.3 μs",3,"7.64 MiB","4"]],
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
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v3.4.0","v3.11.0","v3.11.1"],
  y: [0.673333,0.705916,0.707708],
  customdata: [["1.13.0","673.3 μs",0,"0 B","4"],["1.13.0","705.9 μs",0,"0 B","4"],["1.13.0","707.7 μs",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>curlₕ!: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "divₕ!",
  x: ["v3.4.0","v3.11.0","v3.11.1"],
  y: [0.711125,0.709208,0.726417],
  customdata: [["1.13.0","711.1 μs",0,"0 B","4"],["1.13.0","709.2 μs",0,"0 B","4"],["1.13.0","726.4 μs",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>divₕ!: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Δₕ",
  x: ["v3.4.0","v3.11.0","v3.11.1"],
  y: [1.1921875,1.1911665,1.1965],
  customdata: [["1.13.0","1.19 ms",3,"7.64 MiB","4"],["1.13.0","1.19 ms",3,"7.64 MiB","4"],["1.13.0","1.2 ms",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Δₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Δₕ!",
  x: ["v3.4.0","v3.11.0","v3.11.1"],
  y: [1.146083,1.149625,1.149417],
  customdata: [["1.13.0","1.15 ms",0,"0 B","4"],["1.13.0","1.15 ms",0,"0 B","4"],["1.13.0","1.15 ms",0,"0 B","4"]],
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
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [229.584,212.792,228.583,228.541,225.584,203.041,229.916,229.792,227.791,199.8955,227.083,228.875,222.8125,228.417,223.875,286.208,229.417,228.375,228.583],
  customdata: [["1.12.7","229.6 μs",3,"7.64 MiB","1"],["1.12.7","212.8 μs",3,"7.64 MiB","1"],["1.12.7","228.6 μs",3,"7.64 MiB","1"],["1.12.7","228.5 μs",3,"7.64 MiB","1"],["1.12.7","225.6 μs",3,"7.64 MiB","1"],["1.12.7","203.0 μs",3,"7.64 MiB","1"],["1.12.7","229.9 μs",3,"7.64 MiB","1"],["1.12.7","229.8 μs",3,"7.64 MiB","1"],["1.12.7","227.8 μs",3,"7.64 MiB","1"],["1.13.0","199.9 μs",3,"7.64 MiB","4"],["1.13.0","227.1 μs",3,"7.64 MiB","4"],["1.13.0","228.9 μs",3,"7.64 MiB","4"],["1.13.0","222.8 μs",3,"7.64 MiB","4"],["1.13.0","228.4 μs",3,"7.64 MiB","4"],["1.13.0","223.9 μs",3,"7.64 MiB","4"],["1.13.0","286.2 μs",3,"7.64 MiB","4"],["1.13.0","229.4 μs",3,"7.64 MiB","4"],["1.13.0","228.4 μs",3,"7.64 MiB","4"],["1.13.0","228.6 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>D₋₂: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [240.375,240.292,240.666,244.541,239.542,240.833,238.958,239.583,239.666,241.875,239.125,238.917,239.25,238.917,242.0,290.4165,895.166,892.125,197.375],
  customdata: [["1.12.7","240.4 μs",0,"0 B","1"],["1.12.7","240.3 μs",0,"0 B","1"],["1.12.7","240.7 μs",0,"0 B","1"],["1.12.7","244.5 μs",0,"0 B","1"],["1.12.7","239.5 μs",0,"0 B","1"],["1.12.7","240.8 μs",0,"0 B","1"],["1.12.7","239.0 μs",0,"0 B","1"],["1.12.7","239.6 μs",0,"0 B","1"],["1.12.7","239.7 μs",0,"0 B","1"],["1.13.0","241.9 μs",0,"0 B","4"],["1.13.0","239.1 μs",0,"0 B","4"],["1.13.0","238.9 μs",0,"0 B","4"],["1.13.0","239.2 μs",0,"0 B","4"],["1.13.0","238.9 μs",0,"0 B","4"],["1.13.0","242.0 μs",0,"0 B","4"],["1.13.0","290.4 μs",0,"0 B","4"],["1.13.0","895.2 μs",0,"0 B","4"],["1.13.0","892.1 μs",0,"0 B","4"],["1.13.0","197.4 μs",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>innerₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "∇ₕ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [690.75,696.667,685.875,700.646,694.291,690.75,699.833,691.375,685.5,779.7295,680.125,694.75,694.542,684.292,692.417,873.9795,684.8545,687.708,690.667],
  customdata: [["1.12.7","690.8 μs",15,"22.92 MiB","1"],["1.12.7","696.7 μs",15,"22.92 MiB","1"],["1.12.7","685.9 μs",15,"22.92 MiB","1"],["1.12.7","700.6 μs",15,"22.92 MiB","1"],["1.12.7","694.3 μs",15,"22.92 MiB","1"],["1.12.7","690.8 μs",15,"22.92 MiB","1"],["1.12.7","699.8 μs",15,"22.92 MiB","1"],["1.12.7","691.4 μs",15,"22.92 MiB","1"],["1.12.7","685.5 μs",15,"22.92 MiB","1"],["1.13.0","779.7 μs",15,"22.92 MiB","4"],["1.13.0","680.1 μs",9,"22.92 MiB","4"],["1.13.0","694.8 μs",9,"22.92 MiB","4"],["1.13.0","694.5 μs",9,"22.92 MiB","4"],["1.13.0","684.3 μs",9,"22.92 MiB","4"],["1.13.0","692.4 μs",9,"22.92 MiB","4"],["1.13.0","874.0 μs",9,"22.92 MiB","4"],["1.13.0","684.9 μs",9,"22.92 MiB","4"],["1.13.0","687.7 μs",9,"22.92 MiB","4"],["1.13.0","690.7 μs",9,"22.92 MiB","4"]],
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
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [161.292,161.291,161.625,160.25,163.417,159.75,171.583,175.084,161.375,161.584,162.458,171.334,161.667,162.958,165.667,213.416,162.292,161.584,161.584],
  customdata: [["1.12.7","161.3 μs",3,"7.64 MiB","1"],["1.12.7","161.3 μs",3,"7.64 MiB","1"],["1.12.7","161.6 μs",3,"7.64 MiB","1"],["1.12.7","160.2 μs",3,"7.64 MiB","1"],["1.12.7","163.4 μs",3,"7.64 MiB","1"],["1.12.7","159.8 μs",3,"7.64 MiB","1"],["1.12.7","171.6 μs",3,"7.64 MiB","1"],["1.12.7","175.1 μs",3,"7.64 MiB","1"],["1.12.7","161.4 μs",3,"7.64 MiB","1"],["1.13.0","161.6 μs",3,"7.64 MiB","4"],["1.13.0","162.5 μs",3,"7.64 MiB","4"],["1.13.0","171.3 μs",3,"7.64 MiB","4"],["1.13.0","161.7 μs",3,"7.64 MiB","4"],["1.13.0","163.0 μs",3,"7.64 MiB","4"],["1.13.0","165.7 μs",3,"7.64 MiB","4"],["1.13.0","213.4 μs",3,"7.64 MiB","4"],["1.13.0","162.3 μs",3,"7.64 MiB","4"],["1.13.0","161.6 μs",3,"7.64 MiB","4"],["1.13.0","161.6 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>M₊ᵧ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "M₊₂ 3D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [227.708,226.792,228.458,224.458,226.208,228.583,229.875,230.792,227.0,227.083,227.25,228.0,227.459,228.166,202.0,284.0,227.667,226.417,227.833],
  customdata: [["1.12.7","227.7 μs",3,"7.64 MiB","1"],["1.12.7","226.8 μs",3,"7.64 MiB","1"],["1.12.7","228.5 μs",3,"7.64 MiB","1"],["1.12.7","224.5 μs",3,"7.64 MiB","1"],["1.12.7","226.2 μs",3,"7.64 MiB","1"],["1.12.7","228.6 μs",3,"7.64 MiB","1"],["1.12.7","229.9 μs",3,"7.64 MiB","1"],["1.12.7","230.8 μs",3,"7.64 MiB","1"],["1.12.7","227.0 μs",3,"7.64 MiB","1"],["1.13.0","227.1 μs",3,"7.64 MiB","4"],["1.13.0","227.2 μs",3,"7.64 MiB","4"],["1.13.0","228.0 μs",3,"7.64 MiB","4"],["1.13.0","227.5 μs",3,"7.64 MiB","4"],["1.13.0","228.2 μs",3,"7.64 MiB","4"],["1.13.0","202.0 μs",3,"7.64 MiB","4"],["1.13.0","284.0 μs",3,"7.64 MiB","4"],["1.13.0","227.7 μs",3,"7.64 MiB","4"],["1.13.0","226.4 μs",3,"7.64 MiB","4"],["1.13.0","227.8 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>M₊₂ 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "M₊ₓ 2D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [161.625,159.625,162.417,162.1455,159.25,163.417,172.958,173.083,161.833,162.917,165.959,167.834,162.25,165.166,163.042,203.708,164.042,162.709,162.875],
  customdata: [["1.12.7","161.6 μs",3,"7.64 MiB","1"],["1.12.7","159.6 μs",3,"7.64 MiB","1"],["1.12.7","162.4 μs",3,"7.64 MiB","1"],["1.12.7","162.1 μs",3,"7.64 MiB","1"],["1.12.7","159.2 μs",3,"7.64 MiB","1"],["1.12.7","163.4 μs",3,"7.64 MiB","1"],["1.12.7","173.0 μs",3,"7.64 MiB","1"],["1.12.7","173.1 μs",3,"7.64 MiB","1"],["1.12.7","161.8 μs",3,"7.64 MiB","1"],["1.13.0","162.9 μs",3,"7.64 MiB","4"],["1.13.0","166.0 μs",3,"7.64 MiB","4"],["1.13.0","167.8 μs",3,"7.64 MiB","4"],["1.13.0","162.2 μs",3,"7.64 MiB","4"],["1.13.0","165.2 μs",3,"7.64 MiB","4"],["1.13.0","163.0 μs",3,"7.64 MiB","4"],["1.13.0","203.7 μs",3,"7.64 MiB","4"],["1.13.0","164.0 μs",3,"7.64 MiB","4"],["1.13.0","162.7 μs",3,"7.64 MiB","4"],["1.13.0","162.9 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>M₊ₓ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jumpᵧ 2D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [160.834,159.917,160.958,161.166,152.75,160.875,171.417,167.041,160.75,155.1875,161.791,164.625,161.25,162.875,159.041,191.875,153.0,161.541,161.5],
  customdata: [["1.12.7","160.8 μs",3,"7.64 MiB","1"],["1.12.7","159.9 μs",3,"7.64 MiB","1"],["1.12.7","161.0 μs",3,"7.64 MiB","1"],["1.12.7","161.2 μs",3,"7.64 MiB","1"],["1.12.7","152.8 μs",3,"7.64 MiB","1"],["1.12.7","160.9 μs",3,"7.64 MiB","1"],["1.12.7","171.4 μs",3,"7.64 MiB","1"],["1.12.7","167.0 μs",3,"7.64 MiB","1"],["1.12.7","160.8 μs",3,"7.64 MiB","1"],["1.13.0","155.2 μs",3,"7.64 MiB","4"],["1.13.0","161.8 μs",3,"7.64 MiB","4"],["1.13.0","164.6 μs",3,"7.64 MiB","4"],["1.13.0","161.2 μs",3,"7.64 MiB","4"],["1.13.0","162.9 μs",3,"7.64 MiB","4"],["1.13.0","159.0 μs",3,"7.64 MiB","4"],["1.13.0","191.9 μs",3,"7.64 MiB","4"],["1.13.0","153.0 μs",3,"7.64 MiB","4"],["1.13.0","161.5 μs",3,"7.64 MiB","4"],["1.13.0","161.5 μs",3,"7.64 MiB","4"]],
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
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [227.334,226.667,227.25,225.333,223.2915,203.1045,241.25,230.0,227.875,214.708,225.375,227.875,226.708,227.208,199.75,283.666,227.334,227.333,227.0],
  customdata: [["1.12.7","227.3 μs",3,"7.64 MiB","1"],["1.12.7","226.7 μs",3,"7.64 MiB","1"],["1.12.7","227.2 μs",3,"7.64 MiB","1"],["1.12.7","225.3 μs",3,"7.64 MiB","1"],["1.12.7","223.3 μs",3,"7.64 MiB","1"],["1.12.7","203.1 μs",3,"7.64 MiB","1"],["1.12.7","241.2 μs",3,"7.64 MiB","1"],["1.12.7","230.0 μs",3,"7.64 MiB","1"],["1.12.7","227.9 μs",3,"7.64 MiB","1"],["1.13.0","214.7 μs",3,"7.64 MiB","4"],["1.13.0","225.4 μs",3,"7.64 MiB","4"],["1.13.0","227.9 μs",3,"7.64 MiB","4"],["1.13.0","226.7 μs",3,"7.64 MiB","4"],["1.13.0","227.2 μs",3,"7.64 MiB","4"],["1.13.0","199.8 μs",3,"7.64 MiB","4"],["1.13.0","283.7 μs",3,"7.64 MiB","4"],["1.13.0","227.3 μs",3,"7.64 MiB","4"],["1.13.0","227.3 μs",3,"7.64 MiB","4"],["1.13.0","227.0 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jump₂ 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jumpₓ 2D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [161.458,161.834,160.5,162.416,148.6665,162.917,173.166,169.166,161.333,162.541,165.416,162.459,162.583,164.5,147.792,196.709,163.791,162.375,162.834],
  customdata: [["1.12.7","161.5 μs",3,"7.64 MiB","1"],["1.12.7","161.8 μs",3,"7.64 MiB","1"],["1.12.7","160.5 μs",3,"7.64 MiB","1"],["1.12.7","162.4 μs",3,"7.64 MiB","1"],["1.12.7","148.7 μs",3,"7.64 MiB","1"],["1.12.7","162.9 μs",3,"7.64 MiB","1"],["1.12.7","173.2 μs",3,"7.64 MiB","1"],["1.12.7","169.2 μs",3,"7.64 MiB","1"],["1.12.7","161.3 μs",3,"7.64 MiB","1"],["1.13.0","162.5 μs",3,"7.64 MiB","4"],["1.13.0","165.4 μs",3,"7.64 MiB","4"],["1.13.0","162.5 μs",3,"7.64 MiB","4"],["1.13.0","162.6 μs",3,"7.64 MiB","4"],["1.13.0","164.5 μs",3,"7.64 MiB","4"],["1.13.0","147.8 μs",3,"7.64 MiB","4"],["1.13.0","196.7 μs",3,"7.64 MiB","4"],["1.13.0","163.8 μs",3,"7.64 MiB","4"],["1.13.0","162.4 μs",3,"7.64 MiB","4"],["1.13.0","162.8 μs",3,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jumpₓ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jumpₕ 2D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [410.291,333.625,330.833,331.958],
  customdata: [["1.13.0","410.3 μs",6,"15.28 MiB","4"],["1.13.0","333.6 μs",6,"15.28 MiB","4"],["1.13.0","330.8 μs",6,"15.28 MiB","4"],["1.13.0","332.0 μs",6,"15.28 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jumpₕ 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jumpₕ 3D",
  x: ["v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [784.917,641.084,641.583,645.625],
  customdata: [["1.13.0","784.9 μs",9,"22.92 MiB","4"],["1.13.0","641.1 μs",9,"22.92 MiB","4"],["1.13.0","641.6 μs",9,"22.92 MiB","4"],["1.13.0","645.6 μs",9,"22.92 MiB","4"]],
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
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.9902901716068643,0.9857888715548622,1.0130005200208008,0.998789391575663,0.9968798751950078,0.9960145605824233,1.0022548101924076,0.9982652106084243,1.0130005200208008,0.9991305252210089,1.0185460218408737,1.0038107124284972,0.9921996879875195,1.0180260010400417,1.1776744669786792,3.6749869994799793,3.675161726469059,0.7947665106604264],
  customdata: [["1.12.7","240.4 μs (baseline)",0,"0 B","1"],["1.12.7","238.0 μs (-1.0%)",0,"0 B","1"],["1.12.7","237.0 μs (-1.4%)",0,"0 B","1"],["1.12.7","243.5 μs (+1.3%)",0,"0 B","1"],["1.12.7","240.1 μs (-0.1%)",0,"0 B","1"],["1.12.7","239.6 μs (-0.3%)",0,"0 B","1"],["1.12.7","239.4 μs (-0.4%)",0,"0 B","1"],["1.12.7","240.9 μs (+0.2%)",0,"0 B","1"],["1.12.7","240.0 μs (-0.2%)",0,"0 B","1"],["1.13.0","243.5 μs (+1.3%)",0,"0 B","4"],["1.13.0","240.2 μs (-0.1%)",0,"0 B","4"],["1.13.0","244.8 μs (+1.9%)",0,"0 B","4"],["1.13.0","241.3 μs (+0.4%)",0,"0 B","4"],["1.13.0","238.5 μs (-0.8%)",0,"0 B","4"],["1.13.0","244.7 μs (+1.8%)",0,"0 B","4"],["1.13.0","283.1 μs (+17.8%)",0,"0 B","4"],["1.13.0","883.4 μs (+267.5%)",0,"0 B","4"],["1.13.0","883.4 μs (+267.5%)",0,"0 B","4"],["1.13.0","191.0 μs (-20.5%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>innerₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "norm₁ₕ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.998312211711706,0.9947783654156606,1.0103407103904705,1.0122931256496972,1.0042213699573685,0.9944086473510081,0.9882361282921684,1.062807122390607,0.9964661537039546,0.989765646861005,1.0333961769632725,1.0127679348080694,1.004063100237911,1.0026127165288032,1.2323931267892392,4.185547822145352,4.1968900633205495,0.780427986644568],
  customdata: [["1.12.7","789.8 μs (baseline)",0,"0 B","1"],["1.12.7","788.5 μs (-0.2%)",0,"0 B","1"],["1.12.7","785.7 μs (-0.5%)",0,"0 B","1"],["1.12.7","798.0 μs (+1.0%)",0,"0 B","1"],["1.12.7","799.5 μs (+1.2%)",0,"0 B","1"],["1.12.7","793.1 μs (+0.4%)",0,"0 B","1"],["1.12.7","785.4 μs (-0.6%)",0,"0 B","1"],["1.12.7","780.5 μs (-1.2%)",0,"0 B","1"],["1.12.7","839.4 μs (+6.3%)",0,"0 B","1"],["1.13.0","787.0 μs (-0.4%)",0,"0 B","4"],["1.13.0","781.7 μs (-1.0%)",0,"0 B","4"],["1.13.0","816.2 μs (+3.3%)",0,"0 B","4"],["1.13.0","799.9 μs (+1.3%)",0,"0 B","4"],["1.13.0","793.0 μs (+0.4%)",0,"0 B","4"],["1.13.0","791.9 μs (+0.3%)",0,"0 B","4"],["1.13.0","973.3 μs (+23.2%)",0,"0 B","4"],["1.13.0","3.31 ms (+318.6%)",0,"0 B","4"],["1.13.0","3.31 ms (+319.7%)",0,"0 B","4"],["1.13.0","616.4 μs (-22.0%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>norm₁ₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "normₕ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.9834911142573378,0.9742461382414471,1.0143059400291612,1.0063816748726835,1.0048443674323269,0.9819485239735435,1.0008769520106502,0.988330198846227,1.0118863977347168,0.9929526868541724,1.0427012235065403,0.9969148194324113,0.9865710119814889,1.0118811148912792,1.2218741415379413,4.666948418316675,4.666948418316675,0.7391543224225007],
  customdata: [["1.12.7","189.3 μs (baseline)",0,"0 B","1"],["1.12.7","186.2 μs (-1.7%)",0,"0 B","1"],["1.12.7","184.4 μs (-2.6%)",0,"0 B","1"],["1.12.7","192.0 μs (+1.4%)",0,"0 B","1"],["1.12.7","190.5 μs (+0.6%)",0,"0 B","1"],["1.12.7","190.2 μs (+0.5%)",0,"0 B","1"],["1.12.7","185.9 μs (-1.8%)",0,"0 B","1"],["1.12.7","189.5 μs (+0.1%)",0,"0 B","1"],["1.12.7","187.1 μs (-1.2%)",0,"0 B","1"],["1.13.0","191.5 μs (+1.2%)",0,"0 B","4"],["1.13.0","188.0 μs (-0.7%)",0,"0 B","4"],["1.13.0","197.4 μs (+4.3%)",0,"0 B","4"],["1.13.0","188.7 μs (-0.3%)",0,"0 B","4"],["1.13.0","186.8 μs (-1.3%)",0,"0 B","4"],["1.13.0","191.5 μs (+1.2%)",0,"0 B","4"],["1.13.0","231.3 μs (+22.2%)",0,"0 B","4"],["1.13.0","883.4 μs (+366.7%)",0,"0 B","4"],["1.13.0","883.4 μs (+366.7%)",0,"0 B","4"],["1.13.0","139.9 μs (-26.1%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>normₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "snorm₁ₕ",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0066983578219533,0.9994969749351772,1.0051132238547968,1.0114520311149524,1.000504753673293,1.0014399308556612,1.0025220397579948,0.9985600691443388,1.039829732065687,0.9969749351771824,1.0243439930855662,1.0028798617113224,0.9980553154710458,1.0226153846153847,1.2018150388936906,4.189139152981849,4.195692307692307,0.8192895419187554],
  customdata: [["1.12.7","578.5 μs (baseline)",0,"0 B","1"],["1.12.7","582.4 μs (+0.7%)",0,"0 B","1"],["1.12.7","578.2 μs (-0.1%)",0,"0 B","1"],["1.12.7","581.5 μs (+0.5%)",0,"0 B","1"],["1.12.7","585.1 μs (+1.1%)",0,"0 B","1"],["1.12.7","578.8 μs (+0.1%)",0,"0 B","1"],["1.12.7","579.3 μs (+0.1%)",0,"0 B","1"],["1.12.7","580.0 μs (+0.3%)",0,"0 B","1"],["1.12.7","577.7 μs (-0.1%)",0,"0 B","1"],["1.13.0","601.5 μs (+4.0%)",0,"0 B","4"],["1.13.0","576.8 μs (-0.3%)",0,"0 B","4"],["1.13.0","592.6 μs (+2.4%)",0,"0 B","4"],["1.13.0","580.2 μs (+0.3%)",0,"0 B","4"],["1.13.0","577.4 μs (-0.2%)",0,"0 B","4"],["1.13.0","591.6 μs (+2.3%)",0,"0 B","4"],["1.13.0","695.2 μs (+20.2%)",0,"0 B","4"],["1.13.0","2.42 ms (+318.9%)",0,"0 B","4"],["1.13.0","2.43 ms (+319.6%)",0,"0 B","4"],["1.13.0","474.0 μs (-18.1%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>snorm₁ₕ: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0149558149254283,1.0576855523560704,1.058611235027482,1.0036183927951972,1.0033054507156125,1.0027512302926682,1.0048831482098384,1.0014928901906583,0.40432116682332025,0.4157304091608161,0.4266965255136201,0.4177646891491559,0.4171389614610264,0.4294870300372761,0.6274097518072014,0.4173278220060557,0.4065640227077032,0.40377351818404716],
  customdata: [["1.12.7","3.2 ms (baseline)",10,"7.64 MiB","1"],["1.12.7","3.24 ms (+1.5%)",10,"7.64 MiB","1"],["1.12.7","3.38 ms (+5.8%)",10,"7.64 MiB","1"],["1.12.7","3.38 ms (+5.9%)",10,"7.64 MiB","1"],["1.12.7","3.21 ms (+0.4%)",10,"7.64 MiB","1"],["1.12.7","3.21 ms (+0.3%)",10,"7.64 MiB","1"],["1.12.7","3.2 ms (+0.3%)",10,"7.64 MiB","1"],["1.12.7","3.21 ms (+0.5%)",10,"7.64 MiB","1"],["1.12.7","3.2 ms (+0.1%)",10,"7.64 MiB","1"],["1.13.0","1.29 ms (-59.6%)",25,"7.64 MiB","4"],["1.13.0","1.33 ms (-58.4%)",25,"7.64 MiB","4"],["1.13.0","1.36 ms (-57.3%)",25,"7.64 MiB","4"],["1.13.0","1.33 ms (-58.2%)",25,"7.64 MiB","4"],["1.13.0","1.33 ms (-58.3%)",25,"7.64 MiB","4"],["1.13.0","1.37 ms (-57.1%)",25,"7.64 MiB","4"],["1.13.0","2.0 ms (-37.3%)",25,"7.64 MiB","4"],["1.13.0","1.33 ms (-58.3%)",25,"7.64 MiB","4"],["1.13.0","1.3 ms (-59.3%)",25,"7.64 MiB","4"],["1.13.0","1.29 ms (-59.6%)",25,"7.64 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ 1D (allocates its output): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 1D, Parallel() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.024612615432775,1.0042131789012365,1.0451429018625764,1.0026152762560652,1.003299890436688,1.0026999530442948,1.0075586163718893,1.0001173892627955,0.41691438409766785,0.4201295977461261,0.43479668179683834,0.4167190483643763,0.41626232587259354,0.4333491939270621,0.5610228517764908,0.41733197683518547,0.4157536390671467,0.4037799342620128],
  customdata: [["1.12.7","3.19 ms (baseline)",7,"448 B","1"],["1.12.7","3.27 ms (+2.5%)",7,"448 B","1"],["1.12.7","3.21 ms (+0.4%)",7,"448 B","1"],["1.12.7","3.34 ms (+4.5%)",7,"448 B","1"],["1.12.7","3.2 ms (+0.3%)",7,"448 B","1"],["1.12.7","3.21 ms (+0.3%)",7,"448 B","1"],["1.12.7","3.2 ms (+0.3%)",7,"448 B","1"],["1.12.7","3.22 ms (+0.8%)",7,"448 B","1"],["1.12.7","3.19 ms (+0.0%)",7,"448 B","1"],["1.13.0","1.33 ms (-58.3%)",22,"1.6 KiB","4"],["1.13.0","1.34 ms (-58.0%)",22,"1.6 KiB","4"],["1.13.0","1.39 ms (-56.5%)",22,"1.6 KiB","4"],["1.13.0","1.33 ms (-58.3%)",22,"1.6 KiB","4"],["1.13.0","1.33 ms (-58.4%)",22,"1.6 KiB","4"],["1.13.0","1.38 ms (-56.7%)",22,"1.6 KiB","4"],["1.13.0","1.79 ms (-43.9%)",22,"1.6 KiB","4"],["1.13.0","1.33 ms (-58.3%)",22,"1.6 KiB","4"],["1.13.0","1.33 ms (-58.4%)",22,"1.6 KiB","4"],["1.13.0","1.29 ms (-59.6%)",22,"1.6 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! 1D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 1D, Serial() backend (default)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0301685765215176,1.0069036584330702,1.0091106018164842,1.0059205500381971,1.0022636448518802,1.0003252695017402,1.0038409303115186,0.999801714625244,1.0093795093795095,1.0124070961718021,1.0502220524573467,1.0185753331635685,1.0145220269926152,1.0332029539088363,1.1519962651727358,1.0137933961463372,1.0138005262711145,1.0096484169425346],
  customdata: [["1.12.7","2.95 ms (baseline)",0,"0 B","1"],["1.12.7","3.03 ms (+3.0%)",0,"0 B","1"],["1.12.7","2.97 ms (+0.7%)",0,"0 B","1"],["1.12.7","2.97 ms (+0.9%)",0,"0 B","1"],["1.12.7","2.96 ms (+0.6%)",0,"0 B","1"],["1.12.7","2.95 ms (+0.2%)",0,"0 B","1"],["1.12.7","2.95 ms (+0.0%)",0,"0 B","1"],["1.12.7","2.96 ms (+0.4%)",0,"0 B","1"],["1.12.7","2.94 ms (-0.0%)",0,"0 B","1"],["1.13.0","2.97 ms (+0.9%)",0,"0 B","4"],["1.13.0","2.98 ms (+1.2%)",0,"0 B","4"],["1.13.0","3.09 ms (+5.0%)",0,"0 B","4"],["1.13.0","3.0 ms (+1.9%)",0,"0 B","4"],["1.13.0","2.99 ms (+1.5%)",0,"0 B","4"],["1.13.0","3.04 ms (+3.3%)",0,"0 B","4"],["1.13.0","3.39 ms (+15.2%)",0,"0 B","4"],["1.13.0","2.99 ms (+1.4%)",0,"0 B","4"],["1.13.0","2.99 ms (+1.4%)",0,"0 B","4"],["1.13.0","2.97 ms (+1.0%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! 1D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 1D, Parallel() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0068065935214991,0.9960489659344681,1.0348360507584642,0.9921527235247107,0.9950083128561719,0.9914431805450784,0.9916896533695823,0.9945178468731561,0.4104569307411363,0.41637720798571953,0.42302450537385444,0.4773307142483699,0.48500375684335534,0.48912409533270096,0.7415358692648388,0.4760659949659793,0.4684988012458081,0.4619299270291062],
  customdata: [["1.12.7","16.74 ms (baseline)",7,"544 B","1"],["1.12.7","16.85 ms (+0.7%)",7,"544 B","1"],["1.12.7","16.67 ms (-0.4%)",7,"544 B","1"],["1.12.7","17.32 ms (+3.5%)",7,"544 B","1"],["1.12.7","16.6 ms (-0.8%)",7,"544 B","1"],["1.12.7","16.65 ms (-0.5%)",7,"544 B","1"],["1.12.7","16.59 ms (-0.9%)",7,"544 B","1"],["1.12.7","16.6 ms (-0.8%)",7,"544 B","1"],["1.12.7","16.64 ms (-0.5%)",7,"544 B","1"],["1.13.0","6.87 ms (-59.0%)",22,"2.0 KiB","4"],["1.13.0","6.97 ms (-58.4%)",22,"2.0 KiB","4"],["1.13.0","7.08 ms (-57.7%)",22,"2.0 KiB","4"],["1.13.0","7.99 ms (-52.3%)",22,"2.0 KiB","4"],["1.13.0","8.12 ms (-51.5%)",22,"2.0 KiB","4"],["1.13.0","8.19 ms (-51.1%)",22,"2.0 KiB","4"],["1.13.0","12.41 ms (-25.8%)",22,"2.0 KiB","4"],["1.13.0","7.97 ms (-52.4%)",22,"2.0 KiB","4"],["1.13.0","7.84 ms (-53.2%)",22,"2.0 KiB","4"],["1.13.0","7.73 ms (-53.8%)",22,"2.0 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! 1D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 1D, Serial() backend (default)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.994873716820243,0.9963419560811928,1.0333541454627586,0.9932310747237795,0.9975432663622212,0.9918469632578854,0.9935822180796778,0.9981661872296258,0.9596364651826659,0.9668154954220916,0.9963540196895404,1.110579507461919,1.1109931333825844,1.143313069742144,1.2628476707394525,1.1087131979858045,1.1108801164201698,1.1059389721101496],
  customdata: [["1.12.7","17.32 ms (baseline)",0,"0 B","1"],["1.12.7","17.24 ms (-0.5%)",0,"0 B","1"],["1.12.7","17.26 ms (-0.4%)",0,"0 B","1"],["1.12.7","17.9 ms (+3.3%)",0,"0 B","1"],["1.12.7","17.21 ms (-0.7%)",0,"0 B","1"],["1.12.7","17.28 ms (-0.2%)",0,"0 B","1"],["1.12.7","17.18 ms (-0.8%)",0,"0 B","1"],["1.12.7","17.21 ms (-0.6%)",0,"0 B","1"],["1.12.7","17.29 ms (-0.2%)",0,"0 B","1"],["1.13.0","16.63 ms (-4.0%)",0,"0 B","4"],["1.13.0","16.75 ms (-3.3%)",0,"0 B","4"],["1.13.0","17.26 ms (-0.4%)",0,"0 B","4"],["1.13.0","19.24 ms (+11.1%)",0,"0 B","4"],["1.13.0","19.25 ms (+11.1%)",0,"0 B","4"],["1.13.0","19.81 ms (+14.3%)",0,"0 B","4"],["1.13.0","21.88 ms (+26.3%)",0,"0 B","4"],["1.13.0","19.21 ms (+10.9%)",0,"0 B","4"],["1.13.0","19.25 ms (+11.1%)",0,"0 B","4"],["1.13.0","19.16 ms (+10.6%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! 1D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0244596787155744,1.0037520529878992,1.0590197699397021,1.0040355577242344,1.000785200790986,1.000834284020961,1.0144956521420445,0.9991709515235696,0.5229000099998901,0.3352491935952537,0.3705074446825454,0.33017752160840114,0.33164988761903663,0.3452295812454418,0.5144740555208556,0.3323589112371016,0.3304500317012221,0.3311809137177026],
  customdata: [["1.12.7","3.82 ms (baseline)",7,"448 B","1"],["1.12.7","3.91 ms (+2.4%)",7,"448 B","1"],["1.12.7","3.83 ms (+0.4%)",7,"448 B","1"],["1.12.7","4.05 ms (+5.9%)",7,"448 B","1"],["1.12.7","3.84 ms (+0.4%)",7,"448 B","1"],["1.12.7","3.82 ms (+0.1%)",7,"448 B","1"],["1.12.7","3.82 ms (+0.1%)",7,"448 B","1"],["1.12.7","3.88 ms (+1.4%)",7,"448 B","1"],["1.12.7","3.82 ms (-0.1%)",7,"448 B","1"],["1.13.0","2.0 ms (-47.7%)",22,"1.6 KiB","4"],["1.13.0","1.28 ms (-66.5%)",22,"1.6 KiB","4"],["1.13.0","1.42 ms (-62.9%)",22,"1.6 KiB","4"],["1.13.0","1.26 ms (-67.0%)",22,"1.6 KiB","4"],["1.13.0","1.27 ms (-66.8%)",22,"1.6 KiB","4"],["1.13.0","1.32 ms (-65.5%)",22,"1.6 KiB","4"],["1.13.0","1.97 ms (-48.6%)",22,"1.6 KiB","4"],["1.13.0","1.27 ms (-66.8%)",22,"1.6 KiB","4"],["1.13.0","1.26 ms (-67.0%)",22,"1.6 KiB","4"],["1.13.0","1.27 ms (-66.9%)",22,"1.6 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! 2D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 2D, Serial() backend (default)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.9939257266957312,0.995707065803144,1.0045739782350003,1.0009889135948715,0.9952632481786081,0.9952968278165629,1.0107495298850686,0.9949035819359349,1.0345406359470968,1.0658500746034365,1.1040709849271928,1.0662658417673516,1.0871015351154008,1.088556922476079,1.1247155845122618,1.088191592198088,1.0453965687813478,1.0321692931607582],
  customdata: [["1.12.7","3.71 ms (baseline)",0,"0 B","1"],["1.12.7","3.69 ms (-0.6%)",0,"0 B","1"],["1.12.7","3.69 ms (-0.4%)",0,"0 B","1"],["1.12.7","3.72 ms (+0.5%)",0,"0 B","1"],["1.12.7","3.71 ms (+0.1%)",0,"0 B","1"],["1.12.7","3.69 ms (-0.5%)",0,"0 B","1"],["1.12.7","3.69 ms (-0.5%)",0,"0 B","1"],["1.12.7","3.75 ms (+1.1%)",0,"0 B","1"],["1.12.7","3.69 ms (-0.5%)",0,"0 B","1"],["1.13.0","3.84 ms (+3.5%)",0,"0 B","4"],["1.13.0","3.95 ms (+6.6%)",0,"0 B","4"],["1.13.0","4.09 ms (+10.4%)",0,"0 B","4"],["1.13.0","3.95 ms (+6.6%)",0,"0 B","4"],["1.13.0","4.03 ms (+8.7%)",0,"0 B","4"],["1.13.0","4.04 ms (+8.9%)",0,"0 B","4"],["1.13.0","4.17 ms (+12.5%)",0,"0 B","4"],["1.13.0","4.03 ms (+8.8%)",0,"0 B","4"],["1.13.0","3.88 ms (+4.5%)",0,"0 B","4"],["1.13.0","3.83 ms (+3.2%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! 2D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 2D, Parallel() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.996973378521717,1.0030935735212574,1.0309193802280494,0.9927420808412974,0.9994426898031672,0.9940771833453169,0.9892376588294584,0.9945820072010056,0.3180186793474068,0.2995131835462389,0.36407151474527805,0.3079060822345165,0.3291279999741328,0.3139252335078216,0.3880353268417705,0.3114923449648981,0.303931698802394,0.3026651274433031],
  customdata: [["1.12.7","106.39 ms (baseline)",7,"560 B","1"],["1.12.7","106.07 ms (-0.3%)",7,"560 B","1"],["1.12.7","106.72 ms (+0.3%)",7,"560 B","1"],["1.12.7","109.68 ms (+3.1%)",7,"560 B","1"],["1.12.7","105.62 ms (-0.7%)",7,"560 B","1"],["1.12.7","106.33 ms (-0.1%)",7,"560 B","1"],["1.12.7","105.76 ms (-0.6%)",7,"560 B","1"],["1.12.7","105.24 ms (-1.1%)",7,"560 B","1"],["1.12.7","105.81 ms (-0.5%)",7,"560 B","1"],["1.13.0","33.83 ms (-68.2%)",22,"2.0 KiB","4"],["1.13.0","31.87 ms (-70.0%)",22,"2.0 KiB","4"],["1.13.0","38.73 ms (-63.6%)",22,"2.0 KiB","4"],["1.13.0","32.76 ms (-69.2%)",22,"2.0 KiB","4"],["1.13.0","35.02 ms (-67.1%)",22,"2.0 KiB","4"],["1.13.0","33.4 ms (-68.6%)",22,"2.0 KiB","4"],["1.13.0","41.28 ms (-61.2%)",22,"2.0 KiB","4"],["1.13.0","33.14 ms (-68.9%)",22,"2.0 KiB","4"],["1.13.0","32.34 ms (-69.6%)",22,"2.0 KiB","4"],["1.13.0","32.2 ms (-69.7%)",22,"2.0 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! 2D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 2D, Serial() backend (default)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.990859277878407,0.9963455292959916,0.9953345681565604,0.9938891431766168,0.9957648593551496,0.9891206770770598,0.9936679295196235,1.0025741928006018,0.9877453233911124,1.0191840803256298,1.0728299557509051,0.9995352367792477,0.9995204643541382,1.0512175160053545,1.2101267655888857,1.0209169903342885,0.9950171473765309,1.0063665334118166],
  customdata: [["1.12.7","110.0 ms (baseline)",0,"0 B","1"],["1.12.7","109.0 ms (-0.9%)",0,"0 B","1"],["1.12.7","109.6 ms (-0.4%)",0,"0 B","1"],["1.12.7","109.49 ms (-0.5%)",0,"0 B","1"],["1.12.7","109.33 ms (-0.6%)",0,"0 B","1"],["1.12.7","109.54 ms (-0.4%)",0,"0 B","1"],["1.12.7","108.81 ms (-1.1%)",0,"0 B","1"],["1.12.7","109.31 ms (-0.6%)",0,"0 B","1"],["1.12.7","110.29 ms (+0.3%)",0,"0 B","1"],["1.13.0","108.65 ms (-1.2%)",0,"0 B","4"],["1.13.0","112.11 ms (+1.9%)",0,"0 B","4"],["1.13.0","118.01 ms (+7.3%)",0,"0 B","4"],["1.13.0","109.95 ms (-0.0%)",0,"0 B","4"],["1.13.0","109.95 ms (-0.0%)",0,"0 B","4"],["1.13.0","115.64 ms (+5.1%)",0,"0 B","4"],["1.13.0","133.12 ms (+21.0%)",0,"0 B","4"],["1.13.0","112.3 ms (+2.1%)",0,"0 B","4"],["1.13.0","109.45 ms (-0.5%)",0,"0 B","4"],["1.13.0","110.7 ms (+0.6%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! 2D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.031481808717184,1.0251649174146713,1.0449270941782267,1.0049523059174428,1.0014397703930893,1.0012662145811644,1.009482540307831,1.0003471116238498,0.3274712287909058,0.32554838346605136,0.33519041053490534,0.32882185767748107,0.3299944286558429,0.36738039899828356,0.517225132952531,0.3303413151748783,0.3276587411013253,0.32687098680323023],
  customdata: [["1.12.7","4.44 ms (baseline)",7,"464 B","1"],["1.12.7","4.58 ms (+3.1%)",7,"464 B","1"],["1.12.7","4.55 ms (+2.5%)",7,"464 B","1"],["1.12.7","4.64 ms (+4.5%)",7,"464 B","1"],["1.12.7","4.46 ms (+0.5%)",7,"464 B","1"],["1.12.7","4.45 ms (+0.1%)",7,"464 B","1"],["1.12.7","4.45 ms (+0.1%)",7,"464 B","1"],["1.12.7","4.48 ms (+0.9%)",7,"464 B","1"],["1.12.7","4.44 ms (+0.0%)",7,"464 B","1"],["1.13.0","1.45 ms (-67.3%)",22,"1.7 KiB","4"],["1.13.0","1.45 ms (-67.4%)",22,"1.7 KiB","4"],["1.13.0","1.49 ms (-66.5%)",22,"1.7 KiB","4"],["1.13.0","1.46 ms (-67.1%)",22,"1.7 KiB","4"],["1.13.0","1.47 ms (-67.0%)",22,"1.7 KiB","4"],["1.13.0","1.63 ms (-63.3%)",22,"1.7 KiB","4"],["1.13.0","2.3 ms (-48.3%)",22,"1.7 KiB","4"],["1.13.0","1.47 ms (-67.0%)",22,"1.7 KiB","4"],["1.13.0","1.46 ms (-67.2%)",22,"1.7 KiB","4"],["1.13.0","1.45 ms (-67.3%)",22,"1.7 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! 3D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "Rₕ! 3D, Serial() backend (default)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.9934655799749492,0.9946625222703774,1.0002896799575058,0.9975676612924336,0.9939965247671152,0.9960042466317321,0.994203852685522,0.9943148136608482,1.0225570458713598,1.0333289474019496,1.0799488884559583,1.0344002188641535,1.0369244073143435,1.0678546054480218,1.2015463651246179,1.0351433562521413,1.0308627876037826,1.019719317379199],
  customdata: [["1.12.7","4.32 ms (baseline)",0,"0 B","1"],["1.12.7","4.29 ms (-0.7%)",0,"0 B","1"],["1.12.7","4.29 ms (-0.5%)",0,"0 B","1"],["1.12.7","4.32 ms (+0.0%)",0,"0 B","1"],["1.12.7","4.31 ms (-0.2%)",0,"0 B","1"],["1.12.7","4.29 ms (-0.6%)",0,"0 B","1"],["1.12.7","4.3 ms (-0.4%)",0,"0 B","1"],["1.12.7","4.29 ms (-0.6%)",0,"0 B","1"],["1.12.7","4.29 ms (-0.6%)",0,"0 B","1"],["1.13.0","4.41 ms (+2.3%)",0,"0 B","4"],["1.13.0","4.46 ms (+3.3%)",0,"0 B","4"],["1.13.0","4.66 ms (+8.0%)",0,"0 B","4"],["1.13.0","4.47 ms (+3.4%)",0,"0 B","4"],["1.13.0","4.48 ms (+3.7%)",0,"0 B","4"],["1.13.0","4.61 ms (+6.8%)",0,"0 B","4"],["1.13.0","5.19 ms (+20.2%)",0,"0 B","4"],["1.13.0","4.47 ms (+3.5%)",0,"0 B","4"],["1.13.0","4.45 ms (+3.1%)",0,"0 B","4"],["1.13.0","4.4 ms (+2.0%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! 3D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 3D, Parallel() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0626395001054676,1.0031843637550093,1.067240084053911,1.0099172320242649,1.0017094778533038,1.0000209422579436,1.002750549714537,0.9993585740799327,0.28233620469249016,0.291194915121826,0.29505755350522583,0.3056677600120147,0.2888285736824647,0.2973997291898631,0.49224428205033643,0.2868460393934923,0.2850934672061695,0.27907236667952795],
  customdata: [["1.12.7","620.75 ms (baseline)",7,"576 B","1"],["1.12.7","659.64 ms (+6.3%)",7,"576 B","1"],["1.12.7","622.73 ms (+0.3%)",7,"576 B","1"],["1.12.7","662.49 ms (+6.7%)",7,"576 B","1"],["1.12.7","626.91 ms (+1.0%)",7,"576 B","1"],["1.12.7","621.82 ms (+0.2%)",7,"576 B","1"],["1.12.7","620.77 ms (+0.0%)",7,"576 B","1"],["1.12.7","622.46 ms (+0.3%)",7,"576 B","1"],["1.12.7","620.36 ms (-0.1%)",7,"576 B","1"],["1.13.0","175.26 ms (-71.8%)",22,"2.1 KiB","4"],["1.13.0","180.76 ms (-70.9%)",22,"2.1 KiB","4"],["1.13.0","183.16 ms (-70.5%)",22,"2.1 KiB","4"],["1.13.0","189.74 ms (-69.4%)",22,"2.1 KiB","4"],["1.13.0","179.29 ms (-71.1%)",22,"2.1 KiB","4"],["1.13.0","184.61 ms (-70.3%)",22,"2.1 KiB","4"],["1.13.0","305.56 ms (-50.8%)",22,"2.1 KiB","4"],["1.13.0","178.06 ms (-71.3%)",22,"2.1 KiB","4"],["1.13.0","176.97 ms (-71.5%)",22,"2.1 KiB","4"],["1.13.0","173.24 ms (-72.1%)",22,"2.1 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! 3D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! 3D, Serial() backend (default)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.9953563221063543,0.9961396104082121,1.0118407947449466,1.0025172547814285,0.9973020837723272,0.9957489045726894,0.9989963070018376,0.994810784907797,0.9770465494793164,0.9809596913496603,1.0298435564743893,0.9759521070891544,0.9657529759294509,0.993835378080025,1.01779232741108,0.9800875302866153,0.9800270726365118,0.9824650351056107],
  customdata: [["1.12.7","643.71 ms (baseline)",0,"0 B","1"],["1.12.7","640.72 ms (-0.5%)",0,"0 B","1"],["1.12.7","641.22 ms (-0.4%)",0,"0 B","1"],["1.12.7","651.33 ms (+1.2%)",0,"0 B","1"],["1.12.7","645.33 ms (+0.3%)",0,"0 B","1"],["1.12.7","641.97 ms (-0.3%)",0,"0 B","1"],["1.12.7","640.97 ms (-0.4%)",0,"0 B","1"],["1.12.7","643.06 ms (-0.1%)",0,"0 B","1"],["1.12.7","640.37 ms (-0.5%)",0,"0 B","1"],["1.13.0","628.93 ms (-2.3%)",0,"0 B","4"],["1.13.0","631.45 ms (-1.9%)",0,"0 B","4"],["1.13.0","662.92 ms (+3.0%)",0,"0 B","4"],["1.13.0","628.23 ms (-2.4%)",0,"0 B","4"],["1.13.0","621.66 ms (-3.4%)",0,"0 B","4"],["1.13.0","639.74 ms (-0.6%)",0,"0 B","4"],["1.13.0","655.16 ms (+1.8%)",0,"0 B","4"],["1.13.0","630.89 ms (-2.0%)",0,"0 B","4"],["1.13.0","630.85 ms (-2.0%)",0,"0 B","4"],["1.13.0","632.42 ms (-1.8%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! 3D, Serial() backend (default): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [0.670208,0.693958,0.659917,0.702583,0.694708,0.663333,0.671708,0.679125,0.731542,0.730708,0.958375,0.7645835,0.674167,0.6734795,0.70525,0.833625,0.671542,0.659584,0.737333],
  customdata: [["1.12.7","670.2 μs",3,"22.89 MiB","1"],["1.12.7","694.0 μs",3,"22.89 MiB","1"],["1.12.7","659.9 μs",3,"22.89 MiB","1"],["1.12.7","702.6 μs",3,"22.89 MiB","1"],["1.12.7","694.7 μs",3,"22.89 MiB","1"],["1.12.7","663.3 μs",3,"22.89 MiB","1"],["1.12.7","671.7 μs",3,"22.89 MiB","1"],["1.12.7","679.1 μs",3,"22.89 MiB","1"],["1.12.7","731.5 μs",3,"22.89 MiB","1"],["1.13.0","730.7 μs",3,"22.89 MiB","4"],["1.13.0","958.4 μs",3,"22.89 MiB","4"],["1.13.0","764.6 μs",3,"22.89 MiB","4"],["1.13.0","674.2 μs",3,"22.89 MiB","4"],["1.13.0","673.5 μs",3,"22.89 MiB","4"],["1.13.0","705.2 μs",3,"22.89 MiB","4"],["1.13.0","833.6 μs",3,"22.89 MiB","4"],["1.13.0","671.5 μs",3,"22.89 MiB","4"],["1.13.0","659.6 μs",3,"22.89 MiB","4"],["1.13.0","737.3 μs",3,"22.89 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>D₋ₓ (3 components): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "∇ₕ (3 components)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.380625,1.407666,1.395312,1.4914585,1.42925,1.425667,1.457334,1.510709,1.435042,1.712125,1.5270625,1.532709,1.469875,1.474709,1.651,1.912416,1.465333,1.4417495,1.47525],
  customdata: [["1.12.7","1.38 ms",10,"45.78 MiB","1"],["1.12.7","1.41 ms",10,"45.78 MiB","1"],["1.12.7","1.4 ms",10,"45.78 MiB","1"],["1.12.7","1.49 ms",10,"45.78 MiB","1"],["1.12.7","1.43 ms",10,"45.78 MiB","1"],["1.12.7","1.43 ms",12,"45.78 MiB","1"],["1.12.7","1.46 ms",12,"45.78 MiB","1"],["1.12.7","1.51 ms",12,"45.78 MiB","1"],["1.12.7","1.44 ms",12,"45.78 MiB","1"],["1.13.0","1.71 ms",12,"45.78 MiB","4"],["1.13.0","1.53 ms",6,"45.78 MiB","4"],["1.13.0","1.53 ms",6,"45.78 MiB","4"],["1.13.0","1.47 ms",6,"45.78 MiB","4"],["1.13.0","1.47 ms",6,"45.78 MiB","4"],["1.13.0","1.65 ms",6,"45.78 MiB","4"],["1.13.0","1.91 ms",6,"45.78 MiB","4"],["1.13.0","1.47 ms",6,"45.78 MiB","4"],["1.13.0","1.44 ms",6,"45.78 MiB","4"],["1.13.0","1.48 ms",6,"45.78 MiB","4"]],
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
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0100427623813584,1.0026278231190768,1.005932086562737,1.001989901374415,1.0038764107711464,1.001811499530569,1.0074336354151086,1.002308862246746,0.3158636180975155,0.1906408410475612,0.2221012853942951,0.2069718179147888,0.19172927249890526,0.20340513256698628,0.23854488968819323,0.00036991026905572923,0.00038310831921868524,0.00039133258419423193],
  customdata: [["1.12.7","2.22 ms (baseline)",42,"22.95 MiB","1"],["1.12.7","2.24 ms (+1.0%)",42,"22.95 MiB","1"],["1.12.7","2.23 ms (+0.3%)",42,"22.95 MiB","1"],["1.12.7","2.23 ms (+0.6%)",42,"22.95 MiB","1"],["1.12.7","2.22 ms (+0.2%)",42,"22.95 MiB","1"],["1.12.7","2.23 ms (+0.4%)",42,"22.95 MiB","1"],["1.12.7","2.22 ms (+0.2%)",42,"22.95 MiB","1"],["1.12.7","2.24 ms (+0.7%)",42,"22.95 MiB","1"],["1.12.7","2.22 ms (+0.2%)",42,"22.95 MiB","1"],["1.13.0","701.1 μs (-68.4%)",87,"22.96 MiB","4"],["1.13.0","423.2 μs (-80.9%)",87,"22.96 MiB","4"],["1.13.0","493.0 μs (-77.8%)",87,"22.96 MiB","4"],["1.13.0","459.4 μs (-79.3%)",87,"22.96 MiB","4"],["1.13.0","425.6 μs (-80.8%)",87,"22.96 MiB","4"],["1.13.0","451.5 μs (-79.7%)",87,"22.96 MiB","4"],["1.13.0","529.5 μs (-76.1%)",87,"22.96 MiB","4"],["1.13.0","821.1 ns (-100.0%)",6,"16.1 KiB","4"],["1.13.0","850.4 ns (-100.0%)",6,"16.1 KiB","4"],["1.13.0","868.6 ns (-100.0%)",6,"16.1 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>gridspace 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "gridspace 3D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0131330029173518,1.000413423237967,1.0066000931049133,1.0027623382560218,1.0015761861762753,1.0026446666789257,1.0029371922513226,0.9999261225739411,0.36611402158059625,0.09092859408032121,0.11932494743895354,0.09577450157739596,0.09034733358948475,0.09566360478609581,0.11119923709459763,2.97955136779286e-5,3.014478942065635e-5,3.060768527216958e-5],
  customdata: [["1.12.7","6.2 ms (baseline)",52,"30.57 MiB","1"],["1.12.7","6.28 ms (+1.3%)",52,"30.57 MiB","1"],["1.12.7","6.2 ms (+0.0%)",52,"30.57 MiB","1"],["1.12.7","6.24 ms (+0.7%)",52,"30.57 MiB","1"],["1.12.7","6.22 ms (+0.3%)",52,"30.57 MiB","1"],["1.12.7","6.21 ms (+0.2%)",52,"30.57 MiB","1"],["1.12.7","6.22 ms (+0.3%)",52,"30.57 MiB","1"],["1.12.7","6.22 ms (+0.3%)",52,"30.57 MiB","1"],["1.12.7","6.2 ms (-0.0%)",52,"30.57 MiB","1"],["1.13.0","2.27 ms (-63.4%)",112,"30.57 MiB","4"],["1.13.0","563.7 μs (-90.9%)",112,"30.58 MiB","4"],["1.13.0","739.8 μs (-88.1%)",112,"30.58 MiB","4"],["1.13.0","593.8 μs (-90.4%)",112,"30.58 MiB","4"],["1.13.0","560.1 μs (-91.0%)",112,"30.58 MiB","4"],["1.13.0","593.1 μs (-90.4%)",112,"30.58 MiB","4"],["1.13.0","689.4 μs (-88.9%)",112,"30.58 MiB","4"],["1.13.0","184.7 ns (-100.0%)",6,"2.7 KiB","4"],["1.13.0","186.9 ns (-100.0%)",6,"2.7 KiB","4"],["1.13.0","189.8 ns (-100.0%)",6,"2.7 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>gridspace 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "hₘₐₓ 3D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0106493855220369,0.9980022868874013,0.9989929629649091,0.9971232192340702,1.0013258453409757,0.9996712319979473,0.19943739480859884,0.20141756671222166,0.23039115147632758,0.21121270878308956,0.27148311816564463,0.20491413212895104,0.20299431544522614,0.2262819548073959,0.28676885036779254,0.20628167278037152,0.2060055347642193,0.20272475214374422],
  customdata: [["1.12.7","153.0 ns (baseline)",0,"0 B","1"],["1.12.7","154.6 ns (+1.1%)",0,"0 B","1"],["1.12.7","152.7 ns (-0.2%)",0,"0 B","1"],["1.12.7","152.9 ns (-0.1%)",0,"0 B","1"],["1.12.7","152.6 ns (-0.3%)",0,"0 B","1"],["1.12.7","153.2 ns (+0.1%)",0,"0 B","1"],["1.12.7","153.0 ns (-0.0%)",0,"0 B","1"],["1.12.7","30.5 ns (-80.1%)",0,"0 B","1"],["1.12.7","30.8 ns (-79.9%)",0,"0 B","1"],["1.13.0","35.3 ns (-77.0%)",0,"0 B","4"],["1.13.0","32.3 ns (-78.9%)",0,"0 B","4"],["1.13.0","41.5 ns (-72.9%)",0,"0 B","4"],["1.13.0","31.4 ns (-79.5%)",0,"0 B","4"],["1.13.0","31.1 ns (-79.7%)",0,"0 B","4"],["1.13.0","34.6 ns (-77.4%)",0,"0 B","4"],["1.13.0","43.9 ns (-71.3%)",0,"0 B","4"],["1.13.0","31.6 ns (-79.4%)",0,"0 B","4"],["1.13.0","31.5 ns (-79.4%)",0,"0 B","4"],["1.13.0","31.0 ns (-79.7%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>hₘₐₓ 3D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [598.631542,654.503208,457.110625,455.424,458.909666,456.0315,459.406542,458.50575,461.389708,539.016125,484.567417,812.178792,509.591458,499.982166,532.248125,563.247792,522.508792,470.129792,479.044375],
  customdata: [["1.12.7","598.63 ms",45,"1.3 KiB","1"],["1.12.7","654.5 ms",45,"1.3 KiB","1"],["1.12.7","457.11 ms",45,"1.3 KiB","1"],["1.12.7","455.42 ms",45,"1.3 KiB","1"],["1.12.7","458.91 ms",45,"1.3 KiB","1"],["1.12.7","456.03 ms",45,"1.3 KiB","1"],["1.12.7","459.41 ms",45,"1.3 KiB","1"],["1.12.7","458.51 ms",45,"1.3 KiB","1"],["1.12.7","461.39 ms",45,"1.3 KiB","1"],["1.13.0","539.02 ms",45,"1.3 KiB","4"],["1.13.0","484.57 ms",45,"1.3 KiB","4"],["1.13.0","812.18 ms",45,"1.3 KiB","4"],["1.13.0","509.59 ms",45,"1.3 KiB","4"],["1.13.0","499.98 ms",45,"1.3 KiB","4"],["1.13.0","532.25 ms",45,"1.3 KiB","4"],["1.13.0","563.25 ms",45,"1.3 KiB","4"],["1.13.0","522.51 ms",45,"1.3 KiB","4"],["1.13.0","470.13 ms",45,"1.3 KiB","4"],["1.13.0","479.04 ms",45,"1.3 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>TTFX (load + first operator): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "TTFX first-assembly (assemble)",
  x: ["v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [578.823584,528.666916,538.802334,538.241834,598.798542,583.592833,507.164708,517.530917],
  customdata: [["1.13.0","578.82 ms",45,"1.3 KiB","4"],["1.13.0","528.67 ms",45,"1.3 KiB","4"],["1.13.0","538.8 ms",45,"1.3 KiB","4"],["1.13.0","538.24 ms",45,"1.3 KiB","4"],["1.13.0","598.8 ms",45,"1.3 KiB","4"],["1.13.0","583.59 ms",45,"1.3 KiB","4"],["1.13.0","507.16 ms",45,"1.3 KiB","4"],["1.13.0","517.53 ms",45,"1.3 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>TTFX first-assembly (assemble): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "TTFX first-projection (Rₕ)",
  x: ["v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [569.152,512.33725,515.240125,526.979792,595.986791,560.785,488.720333,506.951625],
  customdata: [["1.13.0","569.15 ms",45,"1.3 KiB","4"],["1.13.0","512.34 ms",45,"1.3 KiB","4"],["1.13.0","515.24 ms",45,"1.3 KiB","4"],["1.13.0","526.98 ms",45,"1.3 KiB","4"],["1.13.0","595.99 ms",45,"1.3 KiB","4"],["1.13.0","560.78 ms",45,"1.3 KiB","4"],["1.13.0","488.72 ms",45,"1.3 KiB","4"],["1.13.0","506.95 ms",45,"1.3 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>TTFX first-projection (Rₕ): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "TTFX mesh construction",
  x: ["v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [536.8775,498.930333,490.774667,494.1685,559.741917,506.614291,457.741542,473.628917],
  customdata: [["1.13.0","536.88 ms",45,"1.3 KiB","4"],["1.13.0","498.93 ms",45,"1.3 KiB","4"],["1.13.0","490.77 ms",45,"1.3 KiB","4"],["1.13.0","494.17 ms",45,"1.3 KiB","4"],["1.13.0","559.74 ms",45,"1.3 KiB","4"],["1.13.0","506.61 ms",45,"1.3 KiB","4"],["1.13.0","457.74 ms",45,"1.3 KiB","4"],["1.13.0","473.63 ms",45,"1.3 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>TTFX mesh construction: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "using Bramble",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [501.70775,487.277792,395.600916,396.802834,396.239292,397.67425,397.573541,398.48925,400.930375,593.014875,398.075875,477.02025,417.810333,414.022791,417.645042,515.704916,441.599292,447.629167,460.878167],
  customdata: [["1.12.7","501.71 ms",45,"1.3 KiB","1"],["1.12.7","487.28 ms",45,"1.3 KiB","1"],["1.12.7","395.6 ms",45,"1.3 KiB","1"],["1.12.7","396.8 ms",45,"1.3 KiB","1"],["1.12.7","396.24 ms",45,"1.3 KiB","1"],["1.12.7","397.67 ms",45,"1.3 KiB","1"],["1.12.7","397.57 ms",45,"1.3 KiB","1"],["1.12.7","398.49 ms",45,"1.3 KiB","1"],["1.12.7","400.93 ms",45,"1.3 KiB","1"],["1.13.0","593.01 ms",45,"1.3 KiB","4"],["1.13.0","398.08 ms",45,"1.3 KiB","4"],["1.13.0","477.02 ms",45,"1.3 KiB","4"],["1.13.0","417.81 ms",45,"1.3 KiB","4"],["1.13.0","414.02 ms",45,"1.3 KiB","4"],["1.13.0","417.65 ms",45,"1.3 KiB","4"],["1.13.0","515.7 ms",45,"1.3 KiB","4"],["1.13.0","441.6 ms",45,"1.3 KiB","4"],["1.13.0","447.63 ms",45,"1.3 KiB","4"],["1.13.0","460.88 ms",45,"1.3 KiB","4"]],
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
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0150463394448215,1.000620983710628,0.9541100493911058,0.9590363781417994,0.9545542178427899,0.9537106173302388,0.9605456987557958,1.00133144020289,0.8992110417933754,0.8714276127916254,0.9451894000317416,0.994830283980218,0.9699760234248264,0.9547757694925508,1.1770815469843945,1.0182854671704127,0.9929881033154991,0.9630530669458786],
  customdata: [["1.12.7","938.8 μs (baseline)",0,"0 B","1"],["1.12.7","953.0 μs (+1.5%)",0,"0 B","1"],["1.12.7","939.4 μs (+0.1%)",0,"0 B","1"],["1.12.7","895.8 μs (-4.6%)",0,"0 B","1"],["1.12.7","900.4 μs (-4.1%)",0,"0 B","1"],["1.12.7","896.2 μs (-4.5%)",0,"0 B","1"],["1.12.7","895.4 μs (-4.6%)",0,"0 B","1"],["1.12.7","901.8 μs (-3.9%)",0,"0 B","1"],["1.12.7","940.1 μs (+0.1%)",0,"0 B","1"],["1.13.0","844.2 μs (-10.1%)",0,"0 B","4"],["1.13.0","818.1 μs (-12.9%)",0,"0 B","4"],["1.13.0","887.4 μs (-5.5%)",0,"0 B","4"],["1.13.0","934.0 μs (-0.5%)",0,"0 B","4"],["1.13.0","910.6 μs (-3.0%)",0,"0 B","4"],["1.13.0","896.4 μs (-4.5%)",0,"0 B","4"],["1.13.0","1.11 ms (+17.7%)",0,"0 B","4"],["1.13.0","956.0 μs (+1.8%)",0,"0 B","4"],["1.13.0","932.2 μs (-0.7%)",0,"0 B","4"],["1.13.0","904.1 μs (-3.7%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble! 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! 1D, Parallel() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0114479998458745,1.0133326855598732,1.1347206853889948,1.1434111806257659,1.1077060191835877,1.1041812380793627,1.0110291807983194,0.9992327235048788,0.406673295176126,0.41792863826012516,0.5053403616753767,0.5081322094463798,0.49486946666744847,0.4397771715139387,0.8762741522474249,0.42265794294511877,0.43092961913433453,0.4080172854997307],
  customdata: [["1.12.7","1.19 ms (baseline)",7,"480 B","1"],["1.12.7","1.21 ms (+1.1%)",7,"480 B","1"],["1.12.7","1.21 ms (+1.3%)",7,"480 B","1"],["1.12.7","1.35 ms (+13.5%)",7,"480 B","1"],["1.12.7","1.37 ms (+14.3%)",7,"480 B","1"],["1.12.7","1.32 ms (+10.8%)",7,"480 B","1"],["1.12.7","1.32 ms (+10.4%)",7,"480 B","1"],["1.12.7","1.21 ms (+1.1%)",7,"480 B","1"],["1.12.7","1.19 ms (-0.1%)",7,"480 B","1"],["1.13.0","485.5 μs (-59.3%)",22,"1.7 KiB","4"],["1.13.0","498.9 μs (-58.2%)",22,"1.8 KiB","4"],["1.13.0","603.3 μs (-49.5%)",22,"1.8 KiB","4"],["1.13.0","606.6 μs (-49.2%)",22,"1.8 KiB","4"],["1.13.0","590.8 μs (-50.5%)",22,"1.9 KiB","4"],["1.13.0","525.0 μs (-56.0%)",22,"1.9 KiB","4"],["1.13.0","1.05 ms (-12.4%)",22,"1.9 KiB","4"],["1.13.0","504.6 μs (-57.7%)",22,"2.0 KiB","4"],["1.13.0","514.5 μs (-56.9%)",22,"2.0 KiB","4"],["1.13.0","487.1 μs (-59.2%)",22,"2.0 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble! 1D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble_parallel! 1D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.9451508406118697,1.041935681318264,0.9527209574584304,0.9217451131856629,1.008264718939897,0.9275886034120342,1.0356404866630975,1.0092649769460287,0.37474112222594985,0.3699637474267181,0.462906388712193,0.47969383272370025,0.38707350540456986,0.3939172536370729,0.9090586495199768,0.37929380700056636,0.37684003697313095,0.46910473421881865],
  customdata: [["1.12.7","1.29 ms (baseline)",7,"480 B","1"],["1.12.7","1.22 ms (-5.5%)",7,"480 B","1"],["1.12.7","1.34 ms (+4.2%)",7,"480 B","1"],["1.12.7","1.23 ms (-4.7%)",7,"480 B","1"],["1.12.7","1.19 ms (-7.8%)",7,"480 B","1"],["1.12.7","1.3 ms (+0.8%)",7,"480 B","1"],["1.12.7","1.2 ms (-7.2%)",7,"480 B","1"],["1.12.7","1.34 ms (+3.6%)",7,"480 B","1"],["1.12.7","1.3 ms (+0.9%)",7,"480 B","1"],["1.13.0","483.7 μs (-62.5%)",22,"1.7 KiB","4"],["1.13.0","477.5 μs (-63.0%)",22,"1.8 KiB","4"],["1.13.0","597.5 μs (-53.7%)",22,"1.8 KiB","4"],["1.13.0","619.1 μs (-52.0%)",22,"1.8 KiB","4"],["1.13.0","499.6 μs (-61.3%)",22,"1.9 KiB","4"],["1.13.0","508.4 μs (-60.6%)",22,"1.9 KiB","4"],["1.13.0","1.17 ms (-9.1%)",22,"1.9 KiB","4"],["1.13.0","489.5 μs (-62.1%)",22,"2.0 KiB","4"],["1.13.0","486.4 μs (-62.3%)",22,"2.0 KiB","4"],["1.13.0","605.5 μs (-53.1%)",22,"2.0 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble_parallel! 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "evaluate! 1D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0108406550714244,0.984801899960271,0.9805899228976152,0.9444057701976254,0.9470795669890693,0.9487645336061625,0.9478666758078523,0.9803701828588707,0.9402307094614787,0.8975646650986018,0.9650068383100058,1.0002206189988996,0.9730088915609277,0.9798568701283633,1.5896723939710364,1.0290786388071456,1.0038454506780299,0.9702621410766207],
  customdata: [["1.12.7","1.14 ms (baseline)",0,"0 B","1"],["1.12.7","1.15 ms (+1.1%)",0,"0 B","1"],["1.12.7","1.12 ms (-1.5%)",0,"0 B","1"],["1.12.7","1.12 ms (-1.9%)",0,"0 B","1"],["1.12.7","1.07 ms (-5.6%)",0,"0 B","1"],["1.12.7","1.08 ms (-5.3%)",0,"0 B","1"],["1.12.7","1.08 ms (-5.1%)",0,"0 B","1"],["1.12.7","1.08 ms (-5.2%)",0,"0 B","1"],["1.12.7","1.12 ms (-2.0%)",0,"0 B","1"],["1.13.0","1.07 ms (-6.0%)",0,"0 B","4"],["1.13.0","1.02 ms (-10.2%)",0,"0 B","4"],["1.13.0","1.1 ms (-3.5%)",0,"0 B","4"],["1.13.0","1.14 ms (+0.0%)",0,"0 B","4"],["1.13.0","1.11 ms (-2.7%)",0,"0 B","4"],["1.13.0","1.11 ms (-2.0%)",0,"0 B","4"],["1.13.0","1.81 ms (+59.0%)",0,"0 B","4"],["1.13.0","1.17 ms (+2.9%)",0,"0 B","4"],["1.13.0","1.14 ms (+0.4%)",0,"0 B","4"],["1.13.0","1.1 ms (-3.0%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>evaluate! 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "l(vₕ) 1D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0075918079096045,0.9970755468636825,1.000234273142112,1.0001403375344053,1.0000464019266986,0.9999988682456903,0.997170614225699,0.9991975861944082,1.0001403375344053,0.9970280131826742,1.000235404896422,1.0000928038533972,1.0000464019266986,0.9971230805446907,1.2466737740837317,0.9998585307112849,1.0000464019266986,0.9999049326379835],
  customdata: [["1.12.7","883.6 μs (baseline)",0,"0 B","1"],["1.12.7","890.3 μs (+0.8%)",0,"0 B","1"],["1.12.7","881.0 μs (-0.3%)",0,"0 B","1"],["1.12.7","883.8 μs (+0.0%)",0,"0 B","1"],["1.12.7","883.7 μs (+0.0%)",0,"0 B","1"],["1.12.7","883.6 μs (+0.0%)",0,"0 B","1"],["1.12.7","883.6 μs (-0.0%)",0,"0 B","1"],["1.12.7","881.1 μs (-0.3%)",0,"0 B","1"],["1.12.7","882.9 μs (-0.1%)",0,"0 B","1"],["1.13.0","883.7 μs (+0.0%)",0,"0 B","4"],["1.13.0","881.0 μs (-0.3%)",0,"0 B","4"],["1.13.0","883.8 μs (+0.0%)",0,"0 B","4"],["1.13.0","883.7 μs (+0.0%)",0,"0 B","4"],["1.13.0","883.6 μs (+0.0%)",0,"0 B","4"],["1.13.0","881.0 μs (-0.3%)",0,"0 B","4"],["1.13.0","1.1 ms (+24.7%)",0,"0 B","4"],["1.13.0","883.5 μs (-0.0%)",0,"0 B","4"],["1.13.0","883.6 μs (+0.0%)",0,"0 B","4"],["1.13.0","883.5 μs (-0.0%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>l(vₕ) 1D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.1201502186338541,1.0020717135970576,1.0212345353316399,1.0126672906966163,0.9977660480211502,1.0182810913987559,1.0987241714732727,1.027936038576055,0.9112683020765807,0.8400096778721661,0.8491355868479692,0.8519130374058224,1.0091403693533108,0.862640874055755,1.275907882457587,0.8275039201730731,1.0387224221202864,1.089744982425351],
  customdata: [["1.12.7","2.84 ms (baseline)",21,"15.13 MiB","1"],["1.12.7","3.18 ms (+12.0%)",21,"15.13 MiB","1"],["1.12.7","2.84 ms (+0.2%)",21,"15.13 MiB","1"],["1.12.7","2.9 ms (+2.1%)",21,"15.13 MiB","1"],["1.12.7","2.87 ms (+1.3%)",21,"15.13 MiB","1"],["1.12.7","2.83 ms (-0.2%)",21,"15.13 MiB","1"],["1.12.7","2.89 ms (+1.8%)",21,"15.13 MiB","1"],["1.12.7","3.12 ms (+9.9%)",21,"15.13 MiB","1"],["1.12.7","2.91 ms (+2.8%)",21,"15.13 MiB","1"],["1.13.0","2.58 ms (-8.9%)",21,"15.13 MiB","4"],["1.13.0","2.38 ms (-16.0%)",21,"15.13 MiB","4"],["1.13.0","2.41 ms (-15.1%)",21,"15.13 MiB","4"],["1.13.0","2.42 ms (-14.8%)",21,"15.13 MiB","4"],["1.13.0","2.86 ms (+0.9%)",21,"15.13 MiB","4"],["1.13.0","2.45 ms (-13.7%)",21,"15.13 MiB","4"],["1.13.0","3.62 ms (+27.6%)",21,"15.13 MiB","4"],["1.13.0","2.35 ms (-17.2%)",21,"15.13 MiB","4"],["1.13.0","2.95 ms (+3.9%)",52,"23.38 MiB","4"],["1.13.0","3.09 ms (+9.0%)",52,"23.38 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>allocate_system_matrix 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble (BilinearForm) 2D, Parallel() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.9389424435152599,0.9377114436036916,0.9799301233425912,0.8903570760582648,0.8822460152931979,1.0151378555558948,0.9325757557971643,1.0100268190508388,0.6446184252435587,0.6399147221485292,0.587253672848274,0.6413171072988982,0.6265376738405867,0.5528527816450323,0.7958522033176739,0.5463068828781451,0.5569775704459851,0.5710720498848432],
  customdata: [["1.12.7","5.11 ms (baseline)",35,"15.13 MiB","1"],["1.12.7","4.8 ms (-6.1%)",35,"15.13 MiB","1"],["1.12.7","4.79 ms (-6.2%)",35,"15.13 MiB","1"],["1.12.7","5.01 ms (-2.0%)",35,"15.13 MiB","1"],["1.12.7","4.55 ms (-11.0%)",35,"15.13 MiB","1"],["1.12.7","4.51 ms (-11.8%)",37,"15.13 MiB","1"],["1.12.7","5.19 ms (+1.5%)",37,"15.13 MiB","1"],["1.12.7","4.77 ms (-6.7%)",37,"15.13 MiB","1"],["1.12.7","5.16 ms (+1.0%)",37,"15.13 MiB","1"],["1.13.0","3.29 ms (-35.5%)",67,"15.13 MiB","4"],["1.13.0","3.27 ms (-36.0%)",67,"15.13 MiB","4"],["1.13.0","3.0 ms (-41.3%)",67,"15.13 MiB","4"],["1.13.0","3.28 ms (-35.9%)",67,"15.13 MiB","4"],["1.13.0","3.2 ms (-37.3%)",67,"15.13 MiB","4"],["1.13.0","2.83 ms (-44.7%)",67,"15.13 MiB","4"],["1.13.0","4.07 ms (-20.4%)",67,"15.13 MiB","4"],["1.13.0","2.79 ms (-45.4%)",67,"15.13 MiB","4"],["1.13.0","2.85 ms (-44.3%)",81,"15.82 MiB","4"],["1.13.0","2.92 ms (-42.9%)",81,"15.82 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble (BilinearForm) 2D, Parallel() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble (BilinearForm) 2D, Serial() backend",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0252965369306104,0.9901446389160652,1.175166373785448,1.1192437385502676,1.1220809811619818,1.1015308594416222,1.1374336986782585,1.1256693771260757,1.0595994895296732,0.9880055170845492,0.9776289518812138,1.0805386581995569,1.1013626401226622,1.0025896865647705,1.1669641440415572,0.9254183845684548,0.9350703127055012,0.984010467356967],
  customdata: [["1.12.7","4.71 ms (baseline)",21,"15.13 MiB","1"],["1.12.7","4.83 ms (+2.5%)",21,"15.13 MiB","1"],["1.12.7","4.67 ms (-1.0%)",21,"15.13 MiB","1"],["1.12.7","5.54 ms (+17.5%)",54,"28.78 MiB","1"],["1.12.7","5.28 ms (+11.9%)",54,"28.78 MiB","1"],["1.12.7","5.29 ms (+12.2%)",54,"28.78 MiB","1"],["1.12.7","5.19 ms (+10.2%)",54,"28.78 MiB","1"],["1.12.7","5.36 ms (+13.7%)",54,"25.17 MiB","1"],["1.12.7","5.31 ms (+12.6%)",54,"28.78 MiB","1"],["1.13.0","5.0 ms (+6.0%)",54,"25.17 MiB","4"],["1.13.0","4.66 ms (-1.2%)",59,"25.17 MiB","4"],["1.13.0","4.61 ms (-2.2%)",59,"25.17 MiB","4"],["1.13.0","5.09 ms (+8.1%)",37,"18.56 MiB","4"],["1.13.0","5.19 ms (+10.1%)",37,"18.56 MiB","4"],["1.13.0","4.73 ms (+0.3%)",36,"18.56 MiB","4"],["1.13.0","5.5 ms (+16.7%)",36,"18.56 MiB","4"],["1.13.0","4.36 ms (-7.5%)",36,"18.56 MiB","4"],["1.13.0","4.41 ms (-6.5%)",52,"23.38 MiB","4"],["1.13.0","4.64 ms (-1.6%)",52,"23.38 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble (BilinearForm) 2D, Serial() backend: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! (matrix) 2D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.9931328213306763,0.9567273159754381,0.6080847847931278,0.6090605928080515,0.6083582357915517,0.6172547580005487,0.6080847847931278,0.6084359633013776,0.6425780061114426,0.2632284261677622,0.2632284261677622,0.27782153201858345,0.2903468988128293,0.2969012944908052,0.742547757935932,0.30817833874772554,0.27333393892116087,0.2912047108489811],
  customdata: [["1.12.7","1.07 ms (baseline)",0,"0 B","1"],["1.12.7","1.06 ms (-0.7%)",0,"0 B","1"],["1.12.7","1.02 ms (-4.3%)",0,"0 B","1"],["1.12.7","649.3 μs (-39.2%)",0,"0 B","1"],["1.12.7","650.4 μs (-39.1%)",0,"0 B","1"],["1.12.7","649.6 μs (-39.2%)",0,"0 B","1"],["1.12.7","659.1 μs (-38.3%)",0,"0 B","1"],["1.12.7","649.3 μs (-39.2%)",0,"0 B","1"],["1.12.7","649.7 μs (-39.2%)",0,"0 B","1"],["1.13.0","686.2 μs (-35.7%)",0,"0 B","4"],["1.13.0","281.1 μs (-73.7%)",0,"0 B","4"],["1.13.0","281.1 μs (-73.7%)",0,"0 B","4"],["1.13.0","296.7 μs (-72.2%)",0,"0 B","4"],["1.13.0","310.0 μs (-71.0%)",0,"0 B","4"],["1.13.0","317.0 μs (-70.3%)",0,"0 B","4"],["1.13.0","792.9 μs (-25.7%)",0,"0 B","4"],["1.13.0","329.1 μs (-69.2%)",0,"0 B","4"],["1.13.0","291.9 μs (-72.7%)",0,"0 B","4"],["1.13.0","311.0 μs (-70.9%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble! (matrix) 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! 2D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0225387640157861,0.39797427504845106,0.7349759506345581,0.7432403750848658,0.7386334626908875,0.8568515268678186,0.7364882398835064,0.854179281384116,0.7484793771258655,0.7645482943073534,0.7749216533743587,0.743890608827229,0.8636378404813333,0.9023868768399906,1.522486652486142,1.0053795132909717,1.00105531121976,0.9951470874296442],
  customdata: [["1.12.7","1.18 ms (baseline)",0,"0 B","1"],["1.12.7","1.21 ms (+2.3%)",0,"0 B","1"],["1.12.7","471.6 μs (-60.2%)",0,"0 B","1"],["1.12.7","870.9 μs (-26.5%)",0,"0 B","1"],["1.12.7","880.7 μs (-25.7%)",0,"0 B","1"],["1.12.7","875.2 μs (-26.1%)",0,"0 B","1"],["1.12.7","1.02 ms (-14.3%)",0,"0 B","1"],["1.12.7","872.7 μs (-26.4%)",0,"0 B","1"],["1.12.7","1.01 ms (-14.6%)",0,"0 B","1"],["1.13.0","886.9 μs (-25.2%)",0,"0 B","4"],["1.13.0","906.0 μs (-23.5%)",0,"0 B","4"],["1.13.0","918.2 μs (-22.5%)",0,"0 B","4"],["1.13.0","881.5 μs (-25.6%)",0,"0 B","4"],["1.13.0","1.02 ms (-13.6%)",0,"0 B","4"],["1.13.0","1.07 ms (-9.8%)",0,"0 B","4"],["1.13.0","1.8 ms (+52.2%)",0,"0 B","4"],["1.13.0","1.19 ms (+0.5%)",0,"0 B","4"],["1.13.0","1.19 ms (+0.1%)",0,"0 B","4"],["1.13.0","1.18 ms (-0.5%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#ec4899", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#ec4899", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble! 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble-then-add (matrix) 2D",
  x: ["v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.8560190139064475,1.5836380615254952,0.9225453013063633,0.8238179519595449,0.9098356510745891],
  customdata: [["1.13.0","7.42 ms (baseline)",87,"32.78 MiB","4"],["1.13.0","6.35 ms (-14.4%)",85,"32.78 MiB","4"],["1.13.0","11.74 ms (+58.4%)",85,"32.78 MiB","4"],["1.13.0","6.84 ms (-7.7%)",85,"32.78 MiB","4"],["1.13.0","6.11 ms (-17.6%)",113,"37.82 MiB","4"],["1.13.0","6.75 ms (-9.0%)",113,"37.82 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#06b6d4", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#06b6d4", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble-then-add (matrix) 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble_add! (matrix) 2D",
  x: ["v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.9629869171418987,1.2041811472660182,1.0123019121100303,1.0100637370010064,1.0250466286481046],
  customdata: [["1.13.0","372.6 μs (baseline)",0,"0 B","4"],["1.13.0","358.8 μs (-3.7%)",0,"0 B","4"],["1.13.0","448.7 μs (+20.4%)",0,"0 B","4"],["1.13.0","377.2 μs (+1.2%)",0,"0 B","4"],["1.13.0","376.4 μs (+1.0%)",0,"0 B","4"],["1.13.0","382.0 μs (+2.5%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f97316", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f97316", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble_add! (matrix) 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble_parallel! 2D",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.302873613909273,1.002922896488245,1.0034448318934261,1.2940924073098903,1.0046217336464363,1.0020501902167822,1.288888772450427,1.2874938396776285,0.34526390922923267,0.2755225103200864,0.2828726882235013,0.28544364946084627,0.2819993997597292,0.2808350151413665,0.5614761602437756,0.28825796708342905,0.28898570746990576,0.28573503671159156],
  customdata: [["1.12.7","1.72 ms (baseline)",7,"496 B","1"],["1.12.7","2.24 ms (+30.3%)",7,"496 B","1"],["1.12.7","1.72 ms (+0.3%)",7,"496 B","1"],["1.12.7","1.72 ms (+0.3%)",7,"496 B","1"],["1.12.7","2.22 ms (+29.4%)",7,"496 B","1"],["1.12.7","1.73 ms (+0.5%)",7,"496 B","1"],["1.12.7","1.72 ms (+0.2%)",7,"496 B","1"],["1.12.7","2.21 ms (+28.9%)",7,"496 B","1"],["1.12.7","2.21 ms (+28.7%)",7,"496 B","1"],["1.13.0","593.0 μs (-65.5%)",22,"1.8 KiB","4"],["1.13.0","473.2 μs (-72.4%)",22,"1.9 KiB","4"],["1.13.0","485.9 μs (-71.7%)",22,"1.9 KiB","4"],["1.13.0","490.3 μs (-71.5%)",22,"1.9 KiB","4"],["1.13.0","484.4 μs (-71.8%)",22,"2.0 KiB","4"],["1.13.0","482.4 μs (-71.9%)",22,"2.0 KiB","4"],["1.13.0","964.4 μs (-43.9%)",22,"2.0 KiB","4"],["1.13.0","495.1 μs (-71.2%)",22,"2.4 KiB","4"],["1.13.0","496.4 μs (-71.1%)",22,"2.4 KiB","4"],["1.13.0","490.8 μs (-71.4%)",22,"2.4 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble_parallel! 2D: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "form (bilinear, 2D)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0,1.0,3.1406625060009596,3.14018242918867,3.1204992798847813,3.1406625060009596,3.2602016322611616,3.14018242918867,3.1603456553048486,3.1406625060009596,3.2602016322611616,3.240518482957273,5.226061395048433,5.760921747479596,8.729805177236765,10.13188162210162,10.172139076856107,10.011792458508506],
  customdata: [["1.12.7","2.1 ns (baseline)",0,"0 B","1"],["1.12.7","2.1 ns (baseline)",0,"0 B","1"],["1.12.7","2.1 ns (baseline)",0,"0 B","1"],["1.12.7","6.5 ns (+214.1%)",1,"32 B","1"],["1.12.7","6.5 ns (+214.0%)",1,"32 B","1"],["1.12.7","6.5 ns (+212.0%)",1,"32 B","1"],["1.12.7","6.5 ns (+214.1%)",1,"32 B","1"],["1.12.7","6.8 ns (+226.0%)",1,"32 B","1"],["1.12.7","6.5 ns (+214.0%)",1,"32 B","1"],["1.13.0","6.6 ns (+216.0%)",1,"32 B","4"],["1.13.0","6.5 ns (+214.1%)",1,"32 B","4"],["1.13.0","6.8 ns (+226.0%)",1,"32 B","4"],["1.13.0","6.8 ns (+224.1%)",1,"32 B","4"],["1.13.0","10.9 ns (+422.6%)",1,"32 B","4"],["1.13.0","12.0 ns (+476.1%)",1,"48 B","4"],["1.13.0","18.2 ns (+773.0%)",1,"32 B","4"],["1.13.0","21.1 ns (+913.2%)",1,"32 B","4"],["1.13.0","21.2 ns (+917.2%)",1,"32 B","4"],["1.13.0","20.9 ns (+901.2%)",1,"32 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>form (bilinear, 2D): %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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

8 benchmarks in this group, across 19 recorded releases.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div style="display:flex; flex-direction:column; gap:1.5rem; width:100%;">
  <div style="width:100%;"><div id="bench_chart_15" style="width:100%; height:300px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
  name: "jacobian (native), 1D n=100",
  x: ["v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0445005829770695,1.0080645161290323,0.9959191605130199,1.0202098717450447,1.1700349786241742,1.0930820054411192,1.1295180722891567,1.113291877186164,1.3036338904003109,1.1173727166731442,1.137679751263117,1.1335989117761367],
  customdata: [["1.12.7","10.3 μs (baseline)",56,"72.2 KiB","1"],["1.12.7","10.8 μs (+4.5%)",56,"72.2 KiB","1"],["1.12.7","10.4 μs (+0.8%)",56,"72.2 KiB","1"],["1.13.0","10.2 μs (-0.4%)",56,"72.2 KiB","4"],["1.13.0","10.5 μs (+2.0%)",65,"73.5 KiB","4"],["1.13.0","12.0 μs (+17.0%)",65,"73.5 KiB","4"],["1.13.0","11.2 μs (+9.3%)",61,"70.6 KiB","4"],["1.13.0","11.6 μs (+13.0%)",61,"70.6 KiB","4"],["1.13.0","11.5 μs (+11.3%)",60,"70.6 KiB","4"],["1.13.0","13.4 μs (+30.4%)",59,"70.6 KiB","4"],["1.13.0","11.5 μs (+11.7%)",59,"70.6 KiB","4"],["1.13.0","11.7 μs (+13.8%)",78,"84.7 KiB","4"],["1.13.0","11.7 μs (+13.4%)",78,"84.7 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jacobian (native), 1D n=100: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jacobian (native), 1D n=10000",
  x: ["v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0056528666347686,0.998562119874315,0.9839181863472912,0.9970712132945952,1.134622451647273,1.095263874433357,1.1561518085720346,1.1225731148479003,1.3064953706672455,1.1154052947530941,1.006371806697611,1.0067836865963922],
  customdata: [["1.12.7","810.9 μs (baseline)",72,"6.63 MiB","1"],["1.12.7","815.5 μs (+0.6%)",72,"6.63 MiB","1"],["1.12.7","809.8 μs (-0.1%)",72,"6.63 MiB","1"],["1.13.0","797.9 μs (-1.6%)",72,"6.63 MiB","4"],["1.13.0","808.5 μs (-0.3%)",82,"6.7 MiB","4"],["1.13.0","920.1 μs (+13.5%)",82,"6.7 MiB","4"],["1.13.0","888.2 μs (+9.5%)",66,"6.14 MiB","4"],["1.13.0","937.5 μs (+15.6%)",66,"6.14 MiB","4"],["1.13.0","910.3 μs (+12.3%)",65,"6.14 MiB","4"],["1.13.0","1.06 ms (+30.6%)",64,"6.14 MiB","4"],["1.13.0","904.5 μs (+11.5%)",64,"6.14 MiB","4"],["1.13.0","816.1 μs (+0.6%)",84,"7.38 MiB","4"],["1.13.0","816.4 μs (+0.7%)",84,"7.38 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jacobian (native), 1D n=10000: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jacobian (traced), 1D n=100",
  x: ["v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.9890042761148442,0.9853390348197923,0.9672746312941792,0.9890042761148442,1.1163277772929574,1.0581202548215376,1.101754079762632,1.0726939523518633,1.312679989527882,1.2254123396456933,1.0581202548215376,1.0581202548215376],
  customdata: [["1.12.7","11.5 μs (baseline)",56,"72.2 KiB","1"],["1.12.7","11.3 μs (-1.1%)",56,"72.2 KiB","1"],["1.12.7","11.3 μs (-1.5%)",56,"72.2 KiB","1"],["1.13.0","11.1 μs (-3.3%)",56,"72.2 KiB","4"],["1.13.0","11.3 μs (-1.1%)",65,"73.5 KiB","4"],["1.13.0","12.8 μs (+11.6%)",65,"73.5 KiB","4"],["1.13.0","12.1 μs (+5.8%)",61,"70.6 KiB","4"],["1.13.0","12.6 μs (+10.2%)",61,"70.6 KiB","4"],["1.13.0","12.3 μs (+7.3%)",60,"70.6 KiB","4"],["1.13.0","15.0 μs (+31.3%)",59,"70.6 KiB","4"],["1.13.0","14.0 μs (+22.5%)",59,"70.6 KiB","4"],["1.13.0","12.1 μs (+5.8%)",78,"84.7 KiB","4"],["1.13.0","12.1 μs (+5.8%)",78,"84.7 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jacobian (traced), 1D n=100: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "jacobian (traced), 1D n=10000",
  x: ["v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0169743950156254,0.9987174737432905,0.983588094906943,0.997229398652732,1.1374420124584748,1.0949266362979548,1.1540077714690833,1.1306724400862074,1.3131179542598457,1.1109791386396113,1.0038463479387882,1.004973789446606],
  customdata: [["1.12.7","812.5 μs (baseline)",72,"6.63 MiB","1"],["1.12.7","826.2 μs (+1.7%)",72,"6.63 MiB","1"],["1.12.7","811.4 μs (-0.1%)",72,"6.63 MiB","1"],["1.13.0","799.1 μs (-1.6%)",72,"6.63 MiB","4"],["1.13.0","810.2 μs (-0.3%)",82,"6.7 MiB","4"],["1.13.0","924.1 μs (+13.7%)",82,"6.7 MiB","4"],["1.13.0","889.6 μs (+9.5%)",66,"6.14 MiB","4"],["1.13.0","937.6 μs (+15.4%)",66,"6.14 MiB","4"],["1.13.0","918.6 μs (+13.1%)",65,"6.14 MiB","4"],["1.13.0","1.07 ms (+31.3%)",64,"6.14 MiB","4"],["1.13.0","902.6 μs (+11.1%)",64,"6.14 MiB","4"],["1.13.0","815.6 μs (+0.4%)",84,"7.38 MiB","4"],["1.13.0","816.5 μs (+0.5%)",84,"7.38 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>jacobian (traced), 1D n=10000: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0070753382011661,1.0117733627667402,1.0070753382011661,0.9457180053206543,1.0754514065772345,0.9457180053206543,0.9740193581253184,0.9598686817229863,1.101432048451916,0.9669440199241524,1.0471500537725704,1.0778004188600214],
  customdata: [["1.12.7","17.7 μs (baseline)",133,"88.4 KiB","1"],["1.12.7","17.8 μs (+0.7%)",133,"88.4 KiB","1"],["1.12.7","17.9 μs (+1.2%)",133,"88.4 KiB","1"],["1.13.0","17.8 μs (+0.7%)",131,"88.3 KiB","4"],["1.13.0","16.7 μs (-5.4%)",140,"89.6 KiB","4"],["1.13.0","19.0 μs (+7.5%)",140,"89.6 KiB","4"],["1.13.0","16.7 μs (-5.4%)",136,"86.7 KiB","4"],["1.13.0","17.2 μs (-2.6%)",136,"86.8 KiB","4"],["1.13.0","17.0 μs (-4.0%)",135,"86.7 KiB","4"],["1.13.0","19.5 μs (+10.1%)",134,"86.7 KiB","4"],["1.13.0","17.1 μs (-3.3%)",134,"86.9 KiB","4"],["1.13.0","18.5 μs (+4.7%)",153,"93.4 KiB","4"],["1.13.0","19.0 μs (+7.8%)",153,"93.4 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>prepare_jacobian (native), 1D n=100: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "prepare_jacobian (native), 1D n=10000",
  x: ["v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.024733071117253,1.0222532156904434,0.9518800892952057,0.9489393005208887,1.1881930477304135,0.9595156798129053,1.0128270436908684,0.9722194110768576,1.1396832146274052,0.9723252896778994,0.923744445625598,0.9608444775167428],
  customdata: [["1.12.7","1.18 ms (baseline)",173,"7.96 MiB","1"],["1.12.7","1.2 ms (+2.5%)",173,"7.96 MiB","1"],["1.12.7","1.2 ms (+2.2%)",173,"7.96 MiB","1"],["1.13.0","1.12 ms (-4.8%)",171,"7.96 MiB","4"],["1.13.0","1.12 ms (-5.1%)",181,"8.04 MiB","4"],["1.13.0","1.4 ms (+18.8%)",181,"8.04 MiB","4"],["1.13.0","1.13 ms (-4.0%)",165,"7.48 MiB","4"],["1.13.0","1.19 ms (+1.3%)",165,"7.48 MiB","4"],["1.13.0","1.14 ms (-2.8%)",164,"7.48 MiB","4"],["1.13.0","1.34 ms (+14.0%)",163,"7.48 MiB","4"],["1.13.0","1.14 ms (-2.8%)",163,"7.48 MiB","4"],["1.13.0","1.09 ms (-7.6%)",183,"8.03 MiB","4"],["1.13.0","1.13 ms (-3.9%)",183,"8.03 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>prepare_jacobian (native), 1D n=10000: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "prepare_jacobian (traced), 1D n=100",
  x: ["v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.932806324110672,1.0085691699604744,0.8985454545454545,0.8656126482213439,1.0553359683794465,0.9532332015810276,0.9795731225296442,0.9407114624505929,1.0843162055335969,0.8906561264822135,0.9071146245059288,0.929501976284585],
  customdata: [["1.12.7","63.2 μs (baseline)",2592,"227.4 KiB","1"],["1.12.7","59.0 μs (-6.7%)",2592,"227.4 KiB","1"],["1.12.7","63.8 μs (+0.9%)",2592,"227.4 KiB","1"],["1.13.0","56.8 μs (-10.1%)",2589,"227.3 KiB","4"],["1.13.0","54.8 μs (-13.4%)",2607,"229.8 KiB","4"],["1.13.0","66.8 μs (+5.5%)",2607,"229.8 KiB","4"],["1.13.0","60.3 μs (-4.7%)",2599,"224.0 KiB","4"],["1.13.0","62.0 μs (-2.0%)",2599,"224.1 KiB","4"],["1.13.0","59.5 μs (-5.9%)",2597,"224.0 KiB","4"],["1.13.0","68.6 μs (+8.4%)",2595,"224.0 KiB","4"],["1.13.0","56.3 μs (-10.9%)",2595,"224.2 KiB","4"],["1.13.0","57.4 μs (-9.3%)",2633,"239.7 KiB","4"],["1.13.0","58.8 μs (-7.0%)",2633,"239.7 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>prepare_jacobian (traced), 1D n=100: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "prepare_jacobian (traced), 1D n=10000",
  x: ["v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0007164820761518,1.116097568648011,0.9696134215872599,0.955798623763025,1.1964522970050462,1.0543722131201259,1.0850514450488236,1.0588898355069138,1.2018874107084343,1.0353060095681237,1.0299232190838195,1.0620617340585883],
  customdata: [["1.12.7","4.77 ms (baseline)",240604,"22.14 MiB","1"],["1.12.7","4.77 ms (+0.1%)",240604,"22.14 MiB","1"],["1.12.7","5.32 ms (+11.6%)",240604,"22.14 MiB","1"],["1.13.0","4.62 ms (-3.0%)",240599,"22.14 MiB","4"],["1.13.0","4.56 ms (-4.4%)",240619,"22.3 MiB","4"],["1.13.0","5.71 ms (+19.6%)",240619,"22.3 MiB","4"],["1.13.0","5.03 ms (+5.4%)",240587,"21.17 MiB","4"],["1.13.0","5.17 ms (+8.5%)",240587,"21.17 MiB","4"],["1.13.0","5.05 ms (+5.9%)",240585,"21.17 MiB","4"],["1.13.0","5.73 ms (+20.2%)",240583,"21.17 MiB","4"],["1.13.0","4.94 ms (+3.5%)",240583,"21.17 MiB","4"],["1.13.0","4.91 ms (+3.0%)",240623,"22.5 MiB","4"],["1.13.0","5.06 ms (+6.2%)",240623,"22.5 MiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>prepare_jacobian (traced), 1D n=10000: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.9989825174825174,0.998979020979021,0.9988356643356643,0.9994195804195805,0.9989825174825174,0.998979020979021,0.9991258741258742,0.9992727272727273,0.9991258741258742,0.9991258741258742,1.0625,0.9989825174825174,0.9992692307692308,1.0001433566433566,0.9992692307692308,0.9988356643356643,0.998979020979021,0.9988356643356643],
  customdata: [["1.12.7","286.0 μs (baseline)",0,"0 B","1"],["1.12.7","285.7 μs (-0.1%)",0,"0 B","1"],["1.12.7","285.7 μs (-0.1%)",0,"0 B","1"],["1.12.7","285.7 μs (-0.1%)",0,"0 B","1"],["1.12.7","285.8 μs (-0.1%)",0,"0 B","1"],["1.12.7","285.7 μs (-0.1%)",0,"0 B","1"],["1.12.7","285.7 μs (-0.1%)",0,"0 B","1"],["1.12.7","285.8 μs (-0.1%)",0,"0 B","1"],["1.12.7","285.8 μs (-0.1%)",0,"0 B","1"],["1.13.0","285.8 μs (-0.1%)",0,"0 B","4"],["1.13.0","285.8 μs (-0.1%)",0,"0 B","4"],["1.13.0","303.9 μs (+6.2%)",0,"0 B","4"],["1.13.0","285.7 μs (-0.1%)",0,"0 B","4"],["1.13.0","285.8 μs (-0.1%)",0,"0 B","4"],["1.13.0","286.0 μs (+0.0%)",0,"0 B","4"],["1.13.0","285.8 μs (-0.1%)",0,"0 B","4"],["1.13.0","285.7 μs (-0.1%)",0,"0 B","4"],["1.13.0","285.7 μs (-0.1%)",0,"0 B","4"],["1.13.0","285.7 μs (-0.1%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! Float32",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.119748223122538,1.0023411324351983,1.1179958784854134,1.003504689274249,1.002915901476175,1.0011775755961476,1.0064346095075212,1.003504689274249,0.8685741522156646,0.8679993831746877,0.8714900536918396,0.8715040724489367,0.8533918382796182,0.8539666073205949,0.8522282814405675,1.000588787798074,1.0017523446371244,1.0011775755961476],
  customdata: [["1.12.7","71.3 μs (baseline)",0,"0 B","1"],["1.12.7","79.9 μs (+12.0%)",0,"0 B","1"],["1.12.7","71.5 μs (+0.2%)",0,"0 B","1"],["1.12.7","79.8 μs (+11.8%)",0,"0 B","1"],["1.12.7","71.6 μs (+0.4%)",0,"0 B","1"],["1.12.7","71.5 μs (+0.3%)",0,"0 B","1"],["1.12.7","71.4 μs (+0.1%)",0,"0 B","1"],["1.12.7","71.8 μs (+0.6%)",0,"0 B","1"],["1.12.7","71.6 μs (+0.4%)",0,"0 B","1"],["1.13.0","62.0 μs (-13.1%)",0,"0 B","4"],["1.13.0","61.9 μs (-13.2%)",0,"0 B","4"],["1.13.0","62.2 μs (-12.9%)",0,"0 B","4"],["1.13.0","62.2 μs (-12.8%)",0,"0 B","4"],["1.13.0","60.9 μs (-14.7%)",0,"0 B","4"],["1.13.0","60.9 μs (-14.6%)",0,"0 B","4"],["1.13.0","60.8 μs (-14.8%)",0,"0 B","4"],["1.13.0","71.4 μs (+0.1%)",0,"0 B","4"],["1.13.0","71.5 μs (+0.2%)",0,"0 B","4"],["1.13.0","71.4 μs (+0.1%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble! Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! Float32",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0091040787194028,0.9996498191821893,1.0080927565175655,1.0032552808823685,1.002191665011738,0.99979207041218,0.9984041092862973,1.0122694020405736,0.9724782778947814,0.9724779666229434,0.9981709666795948,1.1281658680468876,1.1232375010349789,1.1236265908325462,1.12299097373924,1.136414571755318,1.1284120840707883,1.1251051320633028],
  customdata: [["1.12.7","1.61 ms (baseline)",0,"0 B","1"],["1.12.7","1.62 ms (+0.9%)",0,"0 B","1"],["1.12.7","1.61 ms (-0.0%)",0,"0 B","1"],["1.12.7","1.62 ms (+0.8%)",0,"0 B","1"],["1.12.7","1.61 ms (+0.3%)",0,"0 B","1"],["1.12.7","1.61 ms (+0.2%)",0,"0 B","1"],["1.12.7","1.61 ms (-0.0%)",0,"0 B","1"],["1.12.7","1.6 ms (-0.2%)",0,"0 B","1"],["1.12.7","1.63 ms (+1.2%)",0,"0 B","1"],["1.13.0","1.56 ms (-2.8%)",0,"0 B","4"],["1.13.0","1.56 ms (-2.8%)",0,"0 B","4"],["1.13.0","1.6 ms (-0.2%)",0,"0 B","4"],["1.13.0","1.81 ms (+12.8%)",0,"0 B","4"],["1.13.0","1.8 ms (+12.3%)",0,"0 B","4"],["1.13.0","1.8 ms (+12.4%)",0,"0 B","4"],["1.13.0","1.8 ms (+12.3%)",0,"0 B","4"],["1.13.0","1.83 ms (+13.6%)",0,"0 B","4"],["1.13.0","1.81 ms (+12.8%)",0,"0 B","4"],["1.13.0","1.81 ms (+12.5%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ Float32",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0,1.003539364640884,1.0,1.0,1.003539364640884,1.0,1.0,0.9999136740331491,1.0,1.003539364640884,1.028746546961326,1.0071650552486189,1.0,1.007078729281768,1.104281767955801,7.600310773480663,7.600310773480663,0.9891229281767956],
  customdata: [["1.12.7","11.6 μs (baseline)",0,"0 B","1"],["1.12.7","11.6 μs (baseline)",0,"0 B","1"],["1.12.7","11.6 μs (+0.4%)",0,"0 B","1"],["1.12.7","11.6 μs (baseline)",0,"0 B","1"],["1.12.7","11.6 μs (baseline)",0,"0 B","1"],["1.12.7","11.6 μs (+0.4%)",0,"0 B","1"],["1.12.7","11.6 μs (baseline)",0,"0 B","1"],["1.12.7","11.6 μs (baseline)",0,"0 B","1"],["1.12.7","11.6 μs (-0.0%)",0,"0 B","1"],["1.13.0","11.6 μs (baseline)",0,"0 B","4"],["1.13.0","11.6 μs (+0.4%)",0,"0 B","4"],["1.13.0","11.9 μs (+2.9%)",0,"0 B","4"],["1.13.0","11.7 μs (+0.7%)",0,"0 B","4"],["1.13.0","11.6 μs (baseline)",0,"0 B","4"],["1.13.0","11.7 μs (+0.7%)",0,"0 B","4"],["1.13.0","12.8 μs (+10.4%)",0,"0 B","4"],["1.13.0","88.0 μs (+660.0%)",0,"0 B","4"],["1.13.0","88.0 μs (+660.0%)",0,"0 B","4"],["1.13.0","11.5 μs (-1.1%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>innerₕ Float32: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0007085868461754,1.0,1.0004224267736814,1.0001396733687173,1.0001396733687173,1.000143080036247,1.000425833441211,1.000425833441211,1.0701892744479495,1.0039721743396175,1.070119437763591,1.0036894209346534,1.003549747565936,1.008942502265434,1.0036894209346534,1.003549747565936,1.0036894209346534,1.0036894209346534],
  customdata: [["1.12.7","293.5 μs (baseline)",0,"0 B","1"],["1.12.7","293.8 μs (+0.1%)",0,"0 B","1"],["1.12.7","293.5 μs (baseline)",0,"0 B","1"],["1.12.7","293.7 μs (+0.0%)",0,"0 B","1"],["1.12.7","293.6 μs (+0.0%)",0,"0 B","1"],["1.12.7","293.6 μs (+0.0%)",0,"0 B","1"],["1.12.7","293.6 μs (+0.0%)",0,"0 B","1"],["1.12.7","293.7 μs (+0.0%)",0,"0 B","1"],["1.12.7","293.7 μs (+0.0%)",0,"0 B","1"],["1.13.0","314.1 μs (+7.0%)",0,"0 B","4"],["1.13.0","294.7 μs (+0.4%)",0,"0 B","4"],["1.13.0","314.1 μs (+7.0%)",0,"0 B","4"],["1.13.0","294.6 μs (+0.4%)",0,"0 B","4"],["1.13.0","294.6 μs (+0.4%)",0,"0 B","4"],["1.13.0","296.2 μs (+0.9%)",0,"0 B","4"],["1.13.0","294.6 μs (+0.4%)",0,"0 B","4"],["1.13.0","294.6 μs (+0.4%)",0,"0 B","4"],["1.13.0","294.6 μs (+0.4%)",0,"0 B","4"],["1.13.0","294.6 μs (+0.4%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! Float64",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0044665189737727,1.0,0.9657685985850067,0.9608018294861717,0.9627909192691584,0.9642797589270826,0.9662569379928059,1.0128993067962553,0.8878486862478858,0.9493794516305772,0.9151361394983206,0.8863598465899617,0.8784153981752781,0.8774268086424164,0.85111603420758,1.0024893399080492,1.0000119107172634,0.9598013292360466],
  customdata: [["1.12.7","84.0 μs (baseline)",0,"0 B","1"],["1.12.7","84.3 μs (+0.4%)",0,"0 B","1"],["1.12.7","84.0 μs (baseline)",0,"0 B","1"],["1.12.7","81.1 μs (-3.4%)",0,"0 B","1"],["1.12.7","80.7 μs (-3.9%)",0,"0 B","1"],["1.12.7","80.8 μs (-3.7%)",0,"0 B","1"],["1.12.7","81.0 μs (-3.6%)",0,"0 B","1"],["1.12.7","81.1 μs (-3.4%)",0,"0 B","1"],["1.12.7","85.0 μs (+1.3%)",0,"0 B","1"],["1.13.0","74.5 μs (-11.2%)",0,"0 B","4"],["1.13.0","79.7 μs (-5.1%)",0,"0 B","4"],["1.13.0","76.8 μs (-8.5%)",0,"0 B","4"],["1.13.0","74.4 μs (-11.4%)",0,"0 B","4"],["1.13.0","73.8 μs (-12.2%)",0,"0 B","4"],["1.13.0","73.7 μs (-12.3%)",0,"0 B","4"],["1.13.0","71.5 μs (-14.9%)",0,"0 B","4"],["1.13.0","84.2 μs (+0.2%)",0,"0 B","4"],["1.13.0","84.0 μs (+0.0%)",0,"0 B","4"],["1.13.0","80.6 μs (-4.0%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble! Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! Float64",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.039595387843113,1.0002670582355868,1.000036695024737,0.9880595554426876,0.996117025676033,0.996396024356177,0.9955467966408326,1.0100352155745334,0.9552842000542854,0.9523110293595145,0.9571285618928576,1.1002924535225476,1.102137106591475,1.0998069725206687,1.1713043159173697,1.1026348192682662,1.1019428559446522,1.1085805781970977],
  customdata: [["1.12.7","1.72 ms (baseline)",0,"0 B","1"],["1.12.7","1.78 ms (+4.0%)",0,"0 B","1"],["1.12.7","1.72 ms (+0.0%)",0,"0 B","1"],["1.12.7","1.72 ms (+0.0%)",0,"0 B","1"],["1.12.7","1.7 ms (-1.2%)",0,"0 B","1"],["1.12.7","1.71 ms (-0.4%)",0,"0 B","1"],["1.12.7","1.71 ms (-0.4%)",0,"0 B","1"],["1.12.7","1.71 ms (-0.4%)",0,"0 B","1"],["1.12.7","1.73 ms (+1.0%)",0,"0 B","1"],["1.13.0","1.64 ms (-4.5%)",0,"0 B","4"],["1.13.0","1.63 ms (-4.8%)",0,"0 B","4"],["1.13.0","1.64 ms (-4.3%)",0,"0 B","4"],["1.13.0","1.89 ms (+10.0%)",0,"0 B","4"],["1.13.0","1.89 ms (+10.2%)",0,"0 B","4"],["1.13.0","1.89 ms (+10.0%)",0,"0 B","4"],["1.13.0","2.01 ms (+17.1%)",0,"0 B","4"],["1.13.0","1.89 ms (+10.3%)",0,"0 B","4"],["1.13.0","1.89 ms (+10.2%)",0,"0 B","4"],["1.13.0","1.9 ms (+10.9%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ Float64",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0018129235550568,1.0,1.0,1.0,0.9981870764449432,0.999956835153451,1.0,0.999956835153451,1.0,0.999956835153451,1.0,0.999956835153451,1.0,1.0018129235550568,1.0647041049769068,3.8003194198644623,3.8003194198644623,0.9873958648077006],
  customdata: [["1.12.7","23.2 μs (baseline)",0,"0 B","1"],["1.12.7","23.2 μs (+0.2%)",0,"0 B","1"],["1.12.7","23.2 μs (baseline)",0,"0 B","1"],["1.12.7","23.2 μs (baseline)",0,"0 B","1"],["1.12.7","23.2 μs (baseline)",0,"0 B","1"],["1.12.7","23.1 μs (-0.2%)",0,"0 B","1"],["1.12.7","23.2 μs (-0.0%)",0,"0 B","1"],["1.12.7","23.2 μs (baseline)",0,"0 B","1"],["1.12.7","23.2 μs (-0.0%)",0,"0 B","1"],["1.13.0","23.2 μs (baseline)",0,"0 B","4"],["1.13.0","23.2 μs (-0.0%)",0,"0 B","4"],["1.13.0","23.2 μs (baseline)",0,"0 B","4"],["1.13.0","23.2 μs (-0.0%)",0,"0 B","4"],["1.13.0","23.2 μs (baseline)",0,"0 B","4"],["1.13.0","23.2 μs (+0.2%)",0,"0 B","4"],["1.13.0","24.7 μs (+6.5%)",0,"0 B","4"],["1.13.0","88.0 μs (+280.0%)",0,"0 B","4"],["1.13.0","88.0 μs (+280.0%)",0,"0 B","4"],["1.13.0","22.9 μs (-1.3%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>innerₕ Float64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0108921227371217,0.9980802684000839,1.0010880268400084,1.0078096596071853,0.9979869993709373,1.0001305095407842,1.0093846089326903,1.0006290626965821,1.0161343677919898,0.9953914587264975,0.9990750821276299,1.008315314181869,0.9965261759977634,0.9975723212413504,0.9927448102327532,0.9940495142238065,0.9994920528412665,1.0024462990144685],
  customdata: [["1.12.7","8.94 ms (baseline)",0,"0 B","1"],["1.12.7","9.04 ms (+1.1%)",0,"0 B","1"],["1.12.7","8.92 ms (-0.2%)",0,"0 B","1"],["1.12.7","8.95 ms (+0.1%)",0,"0 B","1"],["1.12.7","9.01 ms (+0.8%)",0,"0 B","1"],["1.12.7","8.92 ms (-0.2%)",0,"0 B","1"],["1.12.7","8.94 ms (+0.0%)",0,"0 B","1"],["1.12.7","9.03 ms (+0.9%)",0,"0 B","1"],["1.12.7","8.95 ms (+0.1%)",0,"0 B","1"],["1.13.0","9.09 ms (+1.6%)",0,"0 B","4"],["1.13.0","8.9 ms (-0.5%)",0,"0 B","4"],["1.13.0","8.93 ms (-0.1%)",0,"0 B","4"],["1.13.0","9.02 ms (+0.8%)",0,"0 B","4"],["1.13.0","8.91 ms (-0.3%)",0,"0 B","4"],["1.13.0","8.92 ms (-0.2%)",0,"0 B","4"],["1.13.0","8.88 ms (-0.7%)",0,"0 B","4"],["1.13.0","8.89 ms (-0.6%)",0,"0 B","4"],["1.13.0","8.94 ms (-0.1%)",0,"0 B","4"],["1.13.0","8.96 ms (+0.2%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#3b82f6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#3b82f6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>Rₕ! Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "assemble! Double64",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,0.9757634899651484,0.9847374113688259,0.9597404158154068,0.9626650642951569,0.9780872491287105,0.9621439730801586,0.960261507030405,0.956455233745944,0.9772863838480952,0.95615478908785,0.9697555582261748,0.958218002643913,1.0102554981372431,1.017065256579738,1.0092133157072467,1.0217118134839562,1.007290950606898,1.0239153947842807],
  customdata: [["1.12.7","1.04 ms (baseline)",0,"0 B","1"],["1.12.7","1.01 ms (-2.4%)",0,"0 B","1"],["1.12.7","1.02 ms (-1.5%)",0,"0 B","1"],["1.12.7","998.2 μs (-4.0%)",0,"0 B","1"],["1.12.7","1.0 ms (-3.7%)",0,"0 B","1"],["1.12.7","1.02 ms (-2.2%)",0,"0 B","1"],["1.12.7","1.0 ms (-3.8%)",0,"0 B","1"],["1.12.7","998.8 μs (-4.0%)",0,"0 B","1"],["1.12.7","994.8 μs (-4.4%)",0,"0 B","1"],["1.13.0","1.02 ms (-2.3%)",0,"0 B","4"],["1.13.0","994.5 μs (-4.4%)",0,"0 B","4"],["1.13.0","1.01 ms (-3.0%)",0,"0 B","4"],["1.13.0","996.7 μs (-4.2%)",0,"0 B","4"],["1.13.0","1.05 ms (+1.0%)",0,"0 B","4"],["1.13.0","1.06 ms (+1.7%)",0,"0 B","4"],["1.13.0","1.05 ms (+0.9%)",0,"0 B","4"],["1.13.0","1.06 ms (+2.2%)",0,"0 B","4"],["1.13.0","1.05 ms (+0.7%)",0,"0 B","4"],["1.13.0","1.06 ms (+2.4%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#10b981", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#10b981", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>assemble! Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "avgₕ! Double64",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0152270976688864,1.0015953961178983,1.0064877687740381,1.002254157075967,1.0000291412994644,1.002458450881623,0.9958249617598354,1.004509179803653,1.2196464297493332,1.0063241536739558,1.006864618130729,1.158454950186072,1.1630639707963737,1.1817995618071,1.1525168633283096,1.1595231574819576,1.1606551459964907,1.1621085821577062],
  customdata: [["1.12.7","72.2 ms (baseline)",33,"2.9 KiB","1"],["1.12.7","73.3 ms (+1.5%)",33,"2.9 KiB","1"],["1.12.7","72.32 ms (+0.2%)",33,"2.9 KiB","1"],["1.12.7","72.67 ms (+0.6%)",33,"2.9 KiB","1"],["1.12.7","72.36 ms (+0.2%)",33,"2.9 KiB","1"],["1.12.7","72.2 ms (+0.0%)",33,"2.9 KiB","1"],["1.12.7","72.38 ms (+0.2%)",33,"2.9 KiB","1"],["1.12.7","71.9 ms (-0.4%)",33,"2.9 KiB","1"],["1.12.7","72.53 ms (+0.5%)",33,"2.9 KiB","1"],["1.13.0","88.06 ms (+22.0%)",32,"2.9 KiB","4"],["1.13.0","72.66 ms (+0.6%)",32,"2.9 KiB","4"],["1.13.0","72.7 ms (+0.7%)",32,"2.9 KiB","4"],["1.13.0","83.64 ms (+15.8%)",32,"2.9 KiB","4"],["1.13.0","83.97 ms (+16.3%)",32,"2.9 KiB","4"],["1.13.0","85.33 ms (+18.2%)",32,"2.9 KiB","4"],["1.13.0","83.21 ms (+15.3%)",32,"2.9 KiB","4"],["1.13.0","83.72 ms (+16.0%)",32,"2.9 KiB","4"],["1.13.0","83.8 ms (+16.1%)",32,"2.9 KiB","4"],["1.13.0","83.9 ms (+16.2%)",32,"2.9 KiB","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#f59e0b", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#f59e0b", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>avgₕ! Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "innerₕ Double64",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1.0,1.0,1.0001571025399811,1.0007836312323612,1.000274694261524,0.9997841015992475,1.0009793038570085,0.9986669802445908,0.9999609595484478,1.005722483537159,1.0005098777046095,1.0054873000940734,1.0040376293508937,1.003214487300094,1.0035277516462842,1.0018033866415805,1.0087412982126058,1.0087412982126058,1.0054487300094073],
  customdata: [["1.12.7","1.06 ms (baseline)",0,"0 B","1"],["1.12.7","1.06 ms (baseline)",0,"0 B","1"],["1.12.7","1.06 ms (+0.0%)",0,"0 B","1"],["1.12.7","1.06 ms (+0.1%)",0,"0 B","1"],["1.12.7","1.06 ms (+0.0%)",0,"0 B","1"],["1.12.7","1.06 ms (-0.0%)",0,"0 B","1"],["1.12.7","1.06 ms (+0.1%)",0,"0 B","1"],["1.12.7","1.06 ms (-0.1%)",0,"0 B","1"],["1.12.7","1.06 ms (-0.0%)",0,"0 B","1"],["1.13.0","1.07 ms (+0.6%)",0,"0 B","4"],["1.13.0","1.06 ms (+0.1%)",0,"0 B","4"],["1.13.0","1.07 ms (+0.5%)",0,"0 B","4"],["1.13.0","1.07 ms (+0.4%)",0,"0 B","4"],["1.13.0","1.07 ms (+0.3%)",0,"0 B","4"],["1.13.0","1.07 ms (+0.4%)",0,"0 B","4"],["1.13.0","1.06 ms (+0.2%)",0,"0 B","4"],["1.13.0","1.07 ms (+0.9%)",0,"0 B","4"],["1.13.0","1.07 ms (+0.9%)",0,"0 B","4"],["1.13.0","1.07 ms (+0.5%)",0,"0 B","4"]],
  mode: 'lines+markers',
  type: 'scatter',
  line: { color: "#8b5cf6", width: 2, shape: 'spline', smoothing: 0.3 },
  marker: { color: "#8b5cf6", size: 7 },
  hovertemplate: '%{x} (Julia %{customdata[0]}, %{customdata[4]} thread(s))<br>innerₕ Double64: %{customdata[1]} (%{customdata[2]} allocs, %{customdata[3]})<extra></extra>',
},
{
  name: "1.0x (ref)",
  x: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
  y: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  mode: 'lines',
  type: 'scatter',
  line: { color: 'rgba(128,128,128,0.7)', dash: 'dash', width: 1.5 },
  hoverinfo: 'skip',
}];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    shapes: [{type:'line',xref:'x',yref:'paper',x0:8.5,x1:8.5,y0:0,y1:1,layer:'below',line:{color:theme.grid,width:1.5,dash:'dot'}}],
    annotations: [{xref:'x',yref:'paper',x:8.5,y:1.06,text:'1→4 threads',showarrow:false,font:{color:theme.text,size:9},xanchor:'left'}],
    legend: {
      orientation: 'v', x: 1.02, xanchor: 'left', y: 1, yanchor: 'top',
      font: { color: theme.text, size: 11 },
    },
    xaxis: {
      type: 'category', categoryorder: 'array', categoryarray: ["v2.0.0","v2.1.0","v2.2.0","v2.3.0","v2.4.0","v2.5.0","v2.6.0","v2.7.0","v2.8.0","v2.9.0","v2.10.0","v2.11.0","v2.13.0","v2.16.0","v2.17.0","v3.0.0","v3.4.0","v3.11.0","v3.11.1"],
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
