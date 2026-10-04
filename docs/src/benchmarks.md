# Performance and benchmarks

Bramble tracks memory allocations and performance regressions with a dedicated regression suite in `benchmark/benchmarks.jl`.
Most operator and restriction benchmarks run on about one million grid points per setup ($1000 \times 1000$ in 2D, $100 \times 100 \times 100$ in 3D); assembly and precision benchmarks use smaller grids, set in `benchmark/benchmarks.jl`.

```@raw html
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

```

## Comparisons in the latest baseline

Every chart in this section reads the latest baseline alone (v3.23.0, `54e4457e`), recorded with 4 threads, and finds its pairs by benchmark name; a comparison whose benchmarks the baseline lacks is left out.

### Execution policy

Each bar is the median of the `Serial()` run of a benchmark divided by its median under `Parallel()` or, where the run has it, `CpuPolyester()`; a bar above the dotted line is faster than `Serial()`. The benchmarks are paired by name from the latest run alone.

- `restriction`: Point interpolation (`Rₕ!`) and cell-averaging (`avgₕ!`), compared across the `Serial()` (the allocation-free default), `Parallel()` and `CpuPolyester()` backends, split by dimension.
- `forms`: Linear and bilinear form assembly, across 1D/2D and the `Serial()`/`Parallel()`/`CpuPolyester()` backends, and the refill and `assemble_add!` paths on a fixed matrix pattern.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_1" data-bench="policy" data-run="54e4457e" style="width:100%; height:380px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ name: "Parallel()", x: ["Rₕ! 1D","Rₕ! 2D","Rₕ! 3D","avgₕ! 1D","avgₕ! 2D","avgₕ! 3D","assemble (BilinearForm) 2D"], y: [2.1974865522920344,3.1418034321372854,2.9530822638333682,2.4203645058926253,3.4752270260047506,3.439032638022571,1.1918093913922772], type: 'bar', marker: { color: '#3b82f6' }, hovertext: ["Rₕ! 1D: 1.35 ms, ×2.2 faster than Serial()","Rₕ! 2D: 1.28 ms, ×3.14 faster than Serial()","Rₕ! 3D: 1.49 ms, ×2.95 faster than Serial()","avgₕ! 1D: 7.88 ms, ×2.42 faster than Serial()","avgₕ! 2D: 32.53 ms, ×3.48 faster than Serial()","avgₕ! 3D: 180.96 ms, ×3.44 faster than Serial()","assemble (BilinearForm) 2D: 3.96 ms, ×1.19 faster than Serial()"], hoverinfo: 'text' },
{ name: "CpuPolyester()", x: ["Rₕ! 1D","Rₕ! 2D","Rₕ! 3D","avgₕ! 1D","avgₕ! 2D","avgₕ! 3D","assemble (BilinearForm) 2D"], y: [2.2610390115043595,3.489909671829308,3.253291303541628,2.3462388093978803,3.4708087594825967,3.411627875041029,0.9067580739839194], type: 'bar', marker: { color: '#f59e0b' }, hovertext: ["Rₕ! 1D: 1.31 ms, ×2.26 faster than Serial()","Rₕ! 2D: 1.15 ms, ×3.49 faster than Serial()","Rₕ! 3D: 1.35 ms, ×3.25 faster than Serial()","avgₕ! 1D: 8.13 ms, ×2.35 faster than Serial()","avgₕ! 2D: 32.58 ms, ×3.47 faster than Serial()","avgₕ! 3D: 182.42 ms, ×3.41 faster than Serial()","assemble (BilinearForm) 2D: 5.2 ms, ×0.91 faster than Serial()"], hoverinfo: 'text' }];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    barmode: 'group',
    legend: { orientation: 'h', y: -0.25 },
    shapes: [{ type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: 1, y1: 1, line: { color: theme.text, width: 1, dash: 'dot' } }],
    yaxis: {
      title: { text: "speedup over Serial()", font: { color: theme.text } },
      type: 'log', color: theme.text, gridcolor: theme.grid,
    },
    xaxis: { color: theme.text, automargin: true, tickangle: -30 },
    margin: { t: 20, l: 70, r: 20, b: 130 },
  };
  Plotly.newPlot('bench_chart_1', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_1', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
      'xaxis.color': t.text,
    };
  });
})();
</script>
</div>
```

### Precision

Each bar is the median of a benchmark in `Float32` or `Double64` divided by its median in `Float64`; a bar below the dotted line is cheaper than `Float64`.

- `precision 1D`: The same 1D workload — restriction, assembly, inner product — repeated in `Float32`, `Float64`, and `Double64`; `Double64` (software arithmetic) is an order of magnitude slower.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_2" data-bench="precision" data-run="54e4457e" style="width:100%; height:380px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ name: "Float32", x: ["Rₕ!","assemble!","avgₕ!","innerₕ"], y: [0.9730365859058596,0.8315694754089554,0.9541113065915092,0.5], type: 'bar', marker: { color: '#3b82f6' }, hovertext: ["Rₕ!: 285.7 μs, ×0.97 the Float64 time","assemble!: 61.9 μs, ×0.83 the Float64 time","avgₕ!: 1.81 ms, ×0.95 the Float64 time","innerₕ: 11.5 μs, ×0.5 the Float64 time"], hoverinfo: 'text' },
{ name: "Double64", x: ["Rₕ!","assemble!","avgₕ!","innerₕ"], y: [30.46909391892582,13.86574981868973,44.24879470623943,46.6504625589108], type: 'bar', marker: { color: '#f59e0b' }, hovertext: ["Rₕ!: 8.95 ms, ×30.47 the Float64 time","assemble!: 1.03 ms, ×13.87 the Float64 time","avgₕ!: 83.79 ms, ×44.25 the Float64 time","innerₕ: 1.07 ms, ×46.65 the Float64 time"], hoverinfo: 'text' }];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    barmode: 'group',
    legend: { orientation: 'h', y: -0.25 },
    shapes: [{ type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: 1, y1: 1, line: { color: theme.text, width: 1, dash: 'dot' } }],
    yaxis: {
      title: { text: "median time relative to Float64", font: { color: theme.text } },
      type: 'log', color: theme.text, gridcolor: theme.grid,
    },
    xaxis: { color: theme.text, automargin: true, tickangle: -30 },
    margin: { t: 20, l: 70, r: 20, b: 130 },
  };
  Plotly.newPlot('bench_chart_2', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_2', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
      'xaxis.color': t.text,
    };
  });
})();
</script>
</div>
```

### Direction

The same operator along `x` (the contiguous storage direction) and along `y` (across it), paired by swapping `ₓ` for `ᵧ` in the benchmark name.

- `operators 2D`: The finite-difference stencil engine on a 1000×1000 grid: the difference operator along the grid's contiguous storage direction (`D₋ₓ`) versus across it (`D₋ᵧ`), which access memory very differently and so can perform very differently.
- `jumps & averages`: Jump and average operators across cell interfaces, in 2D and 3D.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_3" data-bench="direction" data-run="54e4457e" style="width:100%; height:324px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ name: "along x", x: [0.204625,0.203958,0.162625,0.162709], y: ["D₋ₓ|ᵧ","D₋ₓ|ᵧ (non-uniform)","M₊ₓ|ᵧ 2D","jumpₓ|ᵧ 2D"], type: 'bar', orientation: 'h', marker: { color: '#3b82f6' }, hovertext: ["D₋ₓ|ᵧ: 204.6 μs","D₋ₓ|ᵧ (non-uniform): 204.0 μs","M₊ₓ|ᵧ 2D: 162.6 μs","jumpₓ|ᵧ 2D: 162.7 μs"], hoverinfo: 'text' },
{ name: "along y", x: [0.16225,0.161916,0.159917,0.1603335], y: ["D₋ₓ|ᵧ","D₋ₓ|ᵧ (non-uniform)","M₊ₓ|ᵧ 2D","jumpₓ|ᵧ 2D"], type: 'bar', orientation: 'h', marker: { color: '#f59e0b' }, hovertext: ["D₋ₓ|ᵧ: 162.2 μs","D₋ₓ|ᵧ (non-uniform): 161.9 μs","M₊ₓ|ᵧ 2D: 159.9 μs","jumpₓ|ᵧ 2D: 160.3 μs"], hoverinfo: 'text' }];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    barmode: 'group',
    legend: { orientation: 'h', y: -0.25 },
    shapes: [],
    xaxis: {
      title: { text: "median time (ms)", font: { color: theme.text } },
      type: 'log', color: theme.text, gridcolor: theme.grid,
    },
    yaxis: { color: theme.text, automargin: true, autorange: 'reversed', tickfont: { size: 10 } },
    margin: { t: 20, l: 260, r: 20, b: 60 },
  };
  Plotly.newPlot('bench_chart_3', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_3', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.text,
      'yaxis.color': t.text,
    };
  });
})();
</script>
</div>
```

### Assembly

A bilinear form assembled for the first time, refilled in place, and combined with a second piece. The last two bars are a pair of their own: both build `M/Δt + θK` from a mass-like and a stiffness-like piece, which is a different form from the first two, so they are compared with each other and not with the bars above.

- `forms`: Linear and bilinear form assembly, across 1D/2D and the `Serial()`/`Parallel()`/`CpuPolyester()` backends, and the refill and `assemble_add!` paths on a fixed matrix pattern.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_4" data-bench="assembly" data-run="54e4457e" style="width:100%; height:324px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ name: "median time", x: [4.7145,0.297583,5.662167,0.354375], y: ["first assembly (allocates and fills)","refill (pattern reused)","assemble the pieces, then add","assemble_add!"], type: 'bar', orientation: 'h', marker: { color: '#3b82f6' }, hovertext: ["first assembly (allocates and fills): 4.71 ms","refill (pattern reused): 297.6 μs","assemble the pieces, then add: 5.66 ms","assemble_add!: 354.4 μs"], hoverinfo: 'text' }];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    barmode: 'group',
    legend: { orientation: 'h', y: -0.25 },
    shapes: [],
    xaxis: {
      title: { text: "median time (ms)", font: { color: theme.text } },
      type: 'log', color: theme.text, gridcolor: theme.grid,
    },
    yaxis: { color: theme.text, automargin: true, autorange: 'reversed', tickfont: { size: 10 } },
    margin: { t: 20, l: 260, r: 20, b: 60 },
  };
  Plotly.newPlot('bench_chart_4', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_4', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.text,
      'yaxis.color': t.text,
    };
  });
})();
</script>
</div>
```

### Uniform and non-uniform grids

Each benchmark on a uniform mesh and on the same mesh with graded axes (the benchmark of the same name with a ` (non-uniform)` suffix).

- `operators 2D`: The finite-difference stencil engine on a 1000×1000 grid: the difference operator along the grid's contiguous storage direction (`D₋ₓ`) versus across it (`D₋ᵧ`), which access memory very differently and so can perform very differently.
- `operators 3D`: The same stencil engine in 3D (`D₋₂`), together with the inner product `innerₕ` and the full gradient `∇ₕ`.
- `forms`: Linear and bilinear form assembly, across 1D/2D and the `Serial()`/`Parallel()`/`CpuPolyester()` backends, and the refill and `assemble_add!` paths on a fixed matrix pattern.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_5" data-bench="uniformity" data-run="54e4457e" style="width:100%; height:784px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ name: "uniform", x: [0.294583,0.369083,0.16225,0.204625,0.170834,0.705917,0.704042,1.2401045,1.14875,0.229292,0.197375,0.691,4.7145,0.297583], y: ["Dcₓ","D₋(uₕ, d) over d","D₋ᵧ","D₋ₓ","Mₓ","curlₕ!","divₕ!","Δₕ","Δₕ!","D₋₂","innerₕ","∇ₕ","assemble (BilinearForm) 2D, Serial() backend","assemble! (matrix) 2D"], type: 'bar', orientation: 'h', marker: { color: '#3b82f6' }, hovertext: ["Dcₓ: 294.6 μs","D₋(uₕ, d) over d: 369.1 μs","D₋ᵧ: 162.2 μs","D₋ₓ: 204.6 μs","Mₓ: 170.8 μs","curlₕ!: 705.9 μs","divₕ!: 704.0 μs","Δₕ: 1.24 ms","Δₕ!: 1.15 ms","D₋₂: 229.3 μs","innerₕ: 197.4 μs","∇ₕ: 691.0 μs","assemble (BilinearForm) 2D, Serial() backend: 4.71 ms","assemble! (matrix) 2D: 297.6 μs"], hoverinfo: 'text' },
{ name: "non-uniform", x: [0.257792,0.365916,0.161916,0.203958,0.15625,0.705834,0.702792,1.191208,1.147792,0.214791,0.197416,0.6868545,3.155125,0.267875], y: ["Dcₓ","D₋(uₕ, d) over d","D₋ᵧ","D₋ₓ","Mₓ","curlₕ!","divₕ!","Δₕ","Δₕ!","D₋₂","innerₕ","∇ₕ","assemble (BilinearForm) 2D, Serial() backend","assemble! (matrix) 2D"], type: 'bar', orientation: 'h', marker: { color: '#f59e0b' }, hovertext: ["Dcₓ: 257.8 μs","D₋(uₕ, d) over d: 365.9 μs","D₋ᵧ: 161.9 μs","D₋ₓ: 204.0 μs","Mₓ: 156.2 μs","curlₕ!: 705.8 μs","divₕ!: 702.8 μs","Δₕ: 1.19 ms","Δₕ!: 1.15 ms","D₋₂: 214.8 μs","innerₕ: 197.4 μs","∇ₕ: 686.9 μs","assemble (BilinearForm) 2D, Serial() backend: 3.16 ms","assemble! (matrix) 2D: 267.9 μs"], hoverinfo: 'text' }];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    barmode: 'group',
    legend: { orientation: 'h', y: -0.25 },
    shapes: [],
    xaxis: {
      title: { text: "median time (ms)", font: { color: theme.text } },
      type: 'log', color: theme.text, gridcolor: theme.grid,
    },
    yaxis: { color: theme.text, automargin: true, autorange: 'reversed', tickfont: { size: 10 } },
    margin: { t: 20, l: 260, r: 20, b: 60 },
  };
  Plotly.newPlot('bench_chart_5', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_5', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.text,
      'yaxis.color': t.text,
    };
  });
})();
</script>
</div>
```

### Dimension

The median time per grid point, in nanoseconds per point, of benchmarks that differ only in the dimension of their name. The time is divided by the `points:` tag of the group, the number of grid points its benchmarks run on.

- `jumps & averages`: Jump and average operators across cell interfaces, in 2D and 3D.
- `restriction`: Point interpolation (`Rₕ!`) and cell-averaging (`avgₕ!`), compared across the `Serial()` (the allocation-free default), `Parallel()` and `CpuPolyester()` backends, split by dimension.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_6" data-bench="dimension" data-run="54e4457e" style="width:100%; height:460px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ name: "jumpₕ nD", x: ["1D","2D","3D"], y: [null,0.331,0.643583], type: 'scatter', mode: 'lines+markers', connectgaps: true, line: { color: '#3b82f6' }, marker: { color: '#3b82f6', size: 7 }, hovertext: ["1D","jumpₕ nD, 2D: 0.331 ns per point","jumpₕ nD, 3D: 0.644 ns per point"], hoverinfo: 'text' },
{ name: "Rₕ! nD, CpuPolyester() backend", x: ["1D","2D","3D"], y: [1.312459,1.154125,1.34825], type: 'scatter', mode: 'lines+markers', connectgaps: true, line: { color: '#f59e0b' }, marker: { color: '#f59e0b', size: 7 }, hovertext: ["Rₕ! nD, CpuPolyester() backend, 1D: 1.31 ns per point","Rₕ! nD, CpuPolyester() backend, 2D: 1.15 ns per point","Rₕ! nD, CpuPolyester() backend, 3D: 1.35 ns per point"], hoverinfo: 'text' },
{ name: "Rₕ! nD, Parallel() backend", x: ["1D","2D","3D"], y: [1.350416,1.282,1.4853125], type: 'scatter', mode: 'lines+markers', connectgaps: true, line: { color: '#10b981' }, marker: { color: '#10b981', size: 7 }, hovertext: ["Rₕ! nD, Parallel() backend, 1D: 1.35 ns per point","Rₕ! nD, Parallel() backend, 2D: 1.28 ns per point","Rₕ! nD, Parallel() backend, 3D: 1.49 ns per point"], hoverinfo: 'text' },
{ name: "Rₕ! nD, Serial() backend (default)", x: ["1D","2D","3D"], y: [2.967521,4.027792,4.38625], type: 'scatter', mode: 'lines+markers', connectgaps: true, line: { color: '#ef4444' }, marker: { color: '#ef4444', size: 7 }, hovertext: ["Rₕ! nD, Serial() backend (default), 1D: 2.97 ns per point","Rₕ! nD, Serial() backend (default), 2D: 4.03 ns per point","Rₕ! nD, Serial() backend (default), 3D: 4.39 ns per point"], hoverinfo: 'text' },
{ name: "avgₕ! nD, CpuPolyester() backend", x: ["1D","2D","3D"], y: [8.133834,32.576125,182.415875], type: 'scatter', mode: 'lines+markers', connectgaps: true, line: { color: '#8b5cf6' }, marker: { color: '#8b5cf6', size: 7 }, hovertext: ["avgₕ! nD, CpuPolyester() backend, 1D: 8.13 ns per point","avgₕ! nD, CpuPolyester() backend, 2D: 32.6 ns per point","avgₕ! nD, CpuPolyester() backend, 3D: 182.0 ns per point"], hoverinfo: 'text' },
{ name: "avgₕ! nD, Parallel() backend", x: ["1D","2D","3D"], y: [7.8847285,32.534709,180.96225], type: 'scatter', mode: 'lines+markers', connectgaps: true, line: { color: '#06b6d4' }, marker: { color: '#06b6d4', size: 7 }, hovertext: ["avgₕ! nD, Parallel() backend, 1D: 7.88 ns per point","avgₕ! nD, Parallel() backend, 2D: 32.5 ns per point","avgₕ! nD, Parallel() backend, 3D: 181.0 ns per point"], hoverinfo: 'text' },
{ name: "avgₕ! nD, Serial() backend (default)", x: ["1D","2D","3D"], y: [19.083917,113.0655,622.335084], type: 'scatter', mode: 'lines+markers', connectgaps: true, line: { color: '#ec4899' }, marker: { color: '#ec4899', size: 7 }, hovertext: ["avgₕ! nD, Serial() backend (default), 1D: 19.1 ns per point","avgₕ! nD, Serial() backend (default), 2D: 113.0 ns per point","avgₕ! nD, Serial() backend (default), 3D: 622.0 ns per point"], hoverinfo: 'text' }];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    barmode: 'group',
    legend: { orientation: 'h', y: -0.25 },
    shapes: [],
    yaxis: {
      title: { text: "median time per grid point (ns)", font: { color: theme.text } },
      type: 'log', color: theme.text, gridcolor: theme.grid,
    },
    xaxis: { color: theme.text, automargin: true, tickangle: -30 },
    margin: { t: 20, l: 70, r: 20, b: 130 },
  };
  Plotly.newPlot('bench_chart_6', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_6', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
      'xaxis.color': t.text,
    };
  });
})();
</script>
</div>
```

## Regressions since the previous baseline

Each bar is one benchmark's median in the latest baseline (v3.23.0, `54e4457e`) divided by its median in the one before (v3.20.0, `ba309208`); a bar to the left of the dotted line is faster. The spread of a run is its interquartile range, from the first to the third quartile of the samples of that one run. A change is flagged, in red when slower and green when faster, only when the two runs' interquartile ranges do not overlap; a grey bar is within the spread. No fixed percentage band is used, because separate launches of the same code can differ by 10 to 30%, which a fixed band would either hide on a quiet benchmark or flag on a loud one.

0 of 81 benchmarks are flagged. 81 have no spread recorded in one of the two runs (baselines saved before the quartiles were recorded keep only the minimum, median and maximum), so they are never flagged. Hover a bar for the two medians.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_7" data-bench="regression" data-flagged="" style="width:100%; height:1376px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
    type: 'bar',
    orientation: 'h',
    y: ["operators 2D/Dcₓ","operators 2D/D₋(uₕ, d) over d","operators 2D/D₋ᵧ","operators 2D/D₋ₓ","operators 2D/Mₓ","operators 2D/curlₕ!","operators 2D/divₕ!","operators 2D/Δₕ","operators 2D/Δₕ!","operators 3D/D₋₂","operators 3D/innerₕ","operators 3D/∇ₕ","jumps & averages/M₊ᵧ 2D","jumps & averages/M₊₂ 3D","jumps & averages/M₊ₓ 2D","jumps & averages/jumpᵧ 2D","jumps & averages/jump₂ 3D","jumps & averages/jumpₓ 2D","jumps & averages/jumpₕ 2D","jumps & averages/jumpₕ 3D","inner products 2D/innerₕ","inner products 2D/norm₁ₕ","inner products 2D/normₕ","inner products 2D/snorm₁ₕ","restriction/Rₕ 1D (allocates its output)","restriction/Rₕ! 1D, Parallel() backend","restriction/Rₕ! 1D, Serial() backend (default)","restriction/Rₕ! 2D, Parallel() backend","restriction/Rₕ! 2D, Serial() backend (default)","restriction/Rₕ! 3D, Parallel() backend","restriction/Rₕ! 3D, Serial() backend (default)","restriction/avgₕ! 1D, Parallel() backend","restriction/avgₕ! 1D, Serial() backend (default)","restriction/avgₕ! 2D, Parallel() backend","restriction/avgₕ! 2D, Serial() backend (default)","restriction/avgₕ! 3D, Parallel() backend","restriction/avgₕ! 3D, Serial() backend (default)","composite/D₋ₓ (3 components)","composite/∇ₕ (3 components)","construction/gridspace 2D","construction/gridspace 3D","construction/hₘₐₓ 3D","startup & latency/TTFX (load + first operator)","startup & latency/TTFX first-assembly (assemble)","startup & latency/TTFX first-projection (Rₕ)","startup & latency/TTFX mesh construction","startup & latency/using Bramble","forms/allocate_system_matrix 2D","forms/assemble (BilinearForm) 2D, Parallel() backend","forms/assemble (BilinearForm) 2D, Serial() backend","forms/assemble! (matrix) 2D","forms/assemble! 1D","forms/assemble! 1D, Parallel() backend","forms/assemble! 2D","forms/assemble-then-add (matrix) 2D","forms/assemble_add! (matrix) 2D","forms/assemble_parallel! 1D","forms/assemble_parallel! 2D","forms/evaluate! 1D","forms/form (bilinear, 2D)","forms/l(vₕ) 1D","jacobian sparsity/jacobian (native), 1D n=100","jacobian sparsity/jacobian (native), 1D n=10000","jacobian sparsity/jacobian (traced), 1D n=100","jacobian sparsity/jacobian (traced), 1D n=10000","jacobian sparsity/prepare_jacobian (native), 1D n=100","jacobian sparsity/prepare_jacobian (native), 1D n=10000","jacobian sparsity/prepare_jacobian (traced), 1D n=100","jacobian sparsity/prepare_jacobian (traced), 1D n=10000","precision 1D/Rₕ! Double64","precision 1D/Rₕ! Float32","precision 1D/Rₕ! Float64","precision 1D/assemble! Double64","precision 1D/assemble! Float32","precision 1D/assemble! Float64","precision 1D/avgₕ! Double64","precision 1D/avgₕ! Float32","precision 1D/avgₕ! Float64","precision 1D/innerₕ Double64","precision 1D/innerₕ Float32","precision 1D/innerₕ Float64"],
    x: [1.1601546962196307,0.9981960784313726,1.0017967510295815,1.0051158369907187,0.9690454986357784,1.0042093553242157,0.9941463242432266,1.0796426161715094,1.037519598846829,1.0034660831509847,1.0002128381973618,1.0091878447632499,0.9894323279195669,0.9627741980916165,0.9976993865030674,0.9932937255290678,0.9994506508688505,1.001033585372306,1.004805459341807,1.0079284846766863,0.9997852931011008,0.9997278836189197,1.0,0.9999134792930625,1.0063709077380953,1.005771349113369,1.0067781900987332,0.9982643282467535,1.0101572696456642,1.0074751134781983,0.9926683422930916,1.0167830430082443,1.0117932159033443,1.029358151880773,1.0649331009469607,1.0368274693159234,1.0158261957107704,1.0487369240287765,0.9991835493797864,1.003102503033299,0.9597724827056111,1.004,1.2402308088113065,1.178656939039888,1.1733758429822185,1.1911379720056885,1.1983183334306042,0.8659792952901352,0.8749802862622731,0.9654349904603269,0.9818531557362175,0.9261156065417051,0.9012810485769078,0.8471719381329973,0.8206336461465995,0.9399867374005305,1.0057719908508465,0.9712399245063288,0.9706988940827728,0.9959151757258472,0.999577841520581,1.1505376344086022,1.0167847946045372,1.1100714285714286,1.0089885092691895,1.0622089552238807,1.0113965095779462,0.9877647335449522,0.9645086519114688,1.0074047312347219,0.9994192413052376,0.9998569604086845,0.96227078965286,0.8679872150727563,0.9216011486285771,1.0036117527384172,1.0020684969495286,1.0040099413460584,0.958173644000043,0.9963478260869565,0.9999563642710652],
    marker: { color: ["#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af"] },
    hovertext: ["253.9 μs → 294.6 μs, ×1.16 (no spread recorded)","369.8 μs → 369.1 μs, ×0.998 (no spread recorded)","162.0 μs → 162.2 μs, ×1.002 (no spread recorded)","203.6 μs → 204.6 μs, ×1.005 (no spread recorded)","176.3 μs → 170.8 μs, ×0.969 (no spread recorded)","703.0 μs → 705.9 μs, ×1.004 (no spread recorded)","708.2 μs → 704.0 μs, ×0.994 (no spread recorded)","1.15 ms → 1.24 ms, ×1.08 (no spread recorded)","1.11 ms → 1.15 ms, ×1.038 (no spread recorded)","228.5 μs → 229.3 μs, ×1.003 (no spread recorded)","197.3 μs → 197.4 μs, ×1.0 (no spread recorded)","684.7 μs → 691.0 μs, ×1.009 (no spread recorded)","161.6 μs → 159.9 μs, ×0.989 (no spread recorded)","227.2 μs → 218.8 μs, ×0.963 (no spread recorded)","163.0 μs → 162.6 μs, ×0.998 (no spread recorded)","161.4 μs → 160.3 μs, ×0.993 (no spread recorded)","227.5 μs → 227.4 μs, ×0.999 (no spread recorded)","162.5 μs → 162.7 μs, ×1.001 (no spread recorded)","329.4 μs → 331.0 μs, ×1.005 (no spread recorded)","638.5 μs → 643.6 μs, ×1.008 (no spread recorded)","191.0 μs → 190.9 μs, ×1.0 (no spread recorded)","613.7 μs → 613.5 μs, ×1.0 (no spread recorded)","139.8 μs → 139.8 μs, ×1.0 (no spread recorded)","473.9 μs → 473.8 μs, ×1.0 (no spread recorded)","1.34 ms → 1.35 ms, ×1.006 (no spread recorded)","1.34 ms → 1.35 ms, ×1.006 (no spread recorded)","2.95 ms → 2.97 ms, ×1.007 (no spread recorded)","1.28 ms → 1.28 ms, ×0.998 (no spread recorded)","3.99 ms → 4.03 ms, ×1.01 (no spread recorded)","1.47 ms → 1.49 ms, ×1.007 (no spread recorded)","4.42 ms → 4.39 ms, ×0.993 (no spread recorded)","7.75 ms → 7.88 ms, ×1.017 (no spread recorded)","18.86 ms → 19.08 ms, ×1.012 (no spread recorded)","31.61 ms → 32.53 ms, ×1.029 (no spread recorded)","106.17 ms → 113.07 ms, ×1.065 (no spread recorded)","174.53 ms → 180.96 ms, ×1.037 (no spread recorded)","612.64 ms → 622.34 ms, ×1.016 (no spread recorded)","664.3 μs → 696.7 μs, ×1.049 (no spread recorded)","1.45 ms → 1.45 ms, ×0.999 (no spread recorded)","915.4 ns → 918.2 ns, ×1.003 (no spread recorded)","172.3 ns → 165.3 ns, ×0.96 (no spread recorded)","31.4 ns → 31.6 ns, ×1.004 (no spread recorded)","472.99 ms → 586.62 ms, ×1.24 (no spread recorded)","507.35 ms → 597.99 ms, ×1.179 (no spread recorded)","489.58 ms → 574.46 ms, ×1.173 (no spread recorded)","463.17 ms → 551.7 ms, ×1.191 (no spread recorded)","454.06 ms → 544.11 ms, ×1.198 (no spread recorded)","3.35 ms → 2.9 ms, ×0.866 (no spread recorded)","4.52 ms → 3.96 ms, ×0.875 (no spread recorded)","4.88 ms → 4.71 ms, ×0.965 (no spread recorded)","303.1 μs → 297.6 μs, ×0.982 (no spread recorded)","908.0 μs → 840.9 μs, ×0.926 (no spread recorded)","546.0 μs → 492.1 μs, ×0.901 (no spread recorded)","1.18 ms → 1.0 ms, ×0.847 (no spread recorded)","6.9 ms → 5.66 ms, ×0.821 (no spread recorded)","377.0 μs → 354.4 μs, ×0.94 (no spread recorded)","483.5 μs → 486.3 μs, ×1.006 (no spread recorded)","491.2 μs → 477.0 μs, ×0.971 (no spread recorded)","1.1 ms → 1.06 ms, ×0.971 (no spread recorded)","16.3 ns → 16.3 ns, ×0.996 (no spread recorded)","888.3 μs → 887.9 μs, ×1.0 (no spread recorded)","11.6 μs → 13.4 μs, ×1.151 (no spread recorded)","815.5 μs → 829.2 μs, ×1.017 (no spread recorded)","14.0 μs → 15.5 μs, ×1.11 (no spread recorded)","815.9 μs → 823.2 μs, ×1.009 (no spread recorded)","16.8 μs → 17.8 μs, ×1.062 (no spread recorded)","1.12 ms → 1.13 ms, ×1.011 (no spread recorded)","59.5 μs → 58.8 μs, ×0.988 (no spread recorded)","5.28 ms → 5.09 ms, ×0.965 (no spread recorded)","8.88 ms → 8.95 ms, ×1.007 (no spread recorded)","285.8 μs → 285.7 μs, ×0.999 (no spread recorded)","293.6 μs → 293.6 μs, ×1.0 (no spread recorded)","1.07 ms → 1.03 ms, ×0.962 (no spread recorded)","71.3 μs → 61.9 μs, ×0.868 (no spread recorded)","80.8 μs → 74.5 μs, ×0.922 (no spread recorded)","83.49 ms → 83.79 ms, ×1.004 (no spread recorded)","1.8 ms → 1.81 ms, ×1.002 (no spread recorded)","1.89 ms → 1.89 ms, ×1.004 (no spread recorded)","1.12 ms → 1.07 ms, ×0.958 (no spread recorded)","11.5 μs → 11.5 μs, ×0.996 (no spread recorded)","22.9 μs → 22.9 μs, ×1.0 (no spread recorded)"],
    hoverinfo: 'text',
  }];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    showlegend: false,
    shapes: [{
      type: 'line', xref: 'x', yref: 'paper', x0: 1, x1: 1, y0: 0, y1: 1,
      line: { color: theme.text, width: 1, dash: 'dot' },
    }],
    xaxis: {
      title: { text: "latest / previous median", font: { color: theme.text } },
      type: 'log', color: theme.text, gridcolor: theme.grid,
    },
    yaxis: { color: theme.text, autorange: 'reversed', tickfont: { size: 9 }, automargin: true },
    margin: { t: 20, l: 260, r: 20, b: 50 },
  };
  Plotly.newPlot('bench_chart_7', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_7', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.text,
      'yaxis.color': t.text,
    };
  });
})();
</script>

</div>
```

## Standalone benchmarks

The scripts in `benchmark/` that compare alternatives outside the regression suite save their tables to `benchmark/results/<script>.toml` with `--save`; every chart below is drawn from those files, and a new full run replaces the file. Each states the machine, thread count, power and commit of its run.

### Operator routes (`operator_routes.jl`)

#### Construction

Time to build the operator of the separable form `innerₕ(u,v) + inner₊(∇ₕu,∇ₕv)` on a non-uniform mesh, by route. Matrix-free construction takes microseconds or less because the operator's plan is built at construction and no matrix is formed; the assembled and Kronecker routes build their matrices here.

Run on Apple M2 with 4 threads, power: AC, commit `86e058eb-dirty`, Julia 1.13.1, 2026-10-03T22:22:32. The script waited for a quiet machine before starting; the load average at the end of the run, which includes the run itself, was 5.85.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_8" data-bench="standalone-operator_routes-construction" data-run="86e058eb-dirty" style="width:100%; height:420px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ type: 'scatter', mode: 'lines+markers', name: "assembled, 2D", x: [1024,4096,16384,65536,262144], y: [0.093959,0.608958,1.8922080000000001,6.1828330000000005,32.1905], line: { color: '#3b82f6', dash: 'solid' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 2D, n = 32 (1024 degrees of freedom): 94.0 μs, 628.953 KiB allocated","assembled, 2D, n = 64 (4096 degrees of freedom): 609.0 μs, 2.269 MiB allocated","assembled, 2D, n = 128 (16384 degrees of freedom): 1.89 ms, 8.831 MiB allocated","assembled, 2D, n = 256 (65536 degrees of freedom): 6.18 ms, 36.081 MiB allocated","assembled, 2D, n = 512 (262144 degrees of freedom): 32.19 ms, 153.987 MiB allocated"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "assembled, 3D", x: [512,4096,32768,110592,262144], y: [0.078084,0.660042,5.746290999999999,26.398999999999997,60.614709000000005], line: { color: '#3b82f6', dash: 'dash' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 3D, n = 8 (512 degrees of freedom): 78.1 μs, 386.969 KiB allocated","assembled, 3D, n = 16 (4096 degrees of freedom): 660.0 μs, 3.114 MiB allocated","assembled, 3D, n = 32 (32768 degrees of freedom): 5.75 ms, 24.677 MiB allocated","assembled, 3D, n = 48 (110592 degrees of freedom): 26.4 ms, 94.036 MiB allocated","assembled, 3D, n = 64 (262144 degrees of freedom): 60.61 ms, 206.427 MiB allocated"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 2D", x: [1024,4096,16384,65536,262144], y: [0.004875,0.009417,0.015792,0.0295,0.072666], line: { color: '#f59e0b', dash: 'solid' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 2D, n = 32 (1024 degrees of freedom): 4.9 μs, 26.078 KiB allocated","Kronecker, 2D, n = 64 (4096 degrees of freedom): 9.4 μs, 44.906 KiB allocated","Kronecker, 2D, n = 128 (16384 degrees of freedom): 15.8 μs, 83.156 KiB allocated","Kronecker, 2D, n = 256 (65536 degrees of freedom): 29.5 μs, 162.656 KiB allocated","Kronecker, 2D, n = 512 (262144 degrees of freedom): 72.7 μs, 318.656 KiB allocated"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 3D", x: [512,4096,32768,110592,262144], y: [0.003875,0.006709,0.008124999999999999,0.010750000000000001,0.015625], line: { color: '#f59e0b', dash: 'dash' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 3D, n = 8 (512 degrees of freedom): 3.9 μs, 16.516 KiB allocated","Kronecker, 3D, n = 16 (4096 degrees of freedom): 6.7 μs, 24.141 KiB allocated","Kronecker, 3D, n = 32 (32768 degrees of freedom): 8.1 μs, 39.016 KiB allocated","Kronecker, 3D, n = 48 (110592 degrees of freedom): 10.8 μs, 54.203 KiB allocated","Kronecker, 3D, n = 64 (262144 degrees of freedom): 15.6 μs, 67.000 KiB allocated"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 2D", x: [1024,4096,16384,65536,262144], y: [0.000292,0.00033299999999999996,0.0005,0.0012079999999999999,0.002042], line: { color: '#10b981', dash: 'solid' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 2D, n = 32 (1024 degrees of freedom): 292.0 ns, 2.078 KiB allocated","matrix-free, serial, 2D, n = 64 (4096 degrees of freedom): 333.0 ns, 2.578 KiB allocated","matrix-free, serial, 2D, n = 128 (16384 degrees of freedom): 500.0 ns, 3.641 KiB allocated","matrix-free, serial, 2D, n = 256 (65536 degrees of freedom): 1.2 μs, 5.578 KiB allocated","matrix-free, serial, 2D, n = 512 (262144 degrees of freedom): 2.0 μs, 9.578 KiB allocated"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 3D", x: [512,4096,32768,110592,262144], y: [0.00020800000000000001,0.00025,0.000375,0.0005,0.000625], line: { color: '#10b981', dash: 'dash' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 3D, n = 8 (512 degrees of freedom): 208.0 ns, 2.797 KiB allocated","matrix-free, serial, 3D, n = 16 (4096 degrees of freedom): 250.0 ns, 2.984 KiB allocated","matrix-free, serial, 3D, n = 32 (32768 degrees of freedom): 375.0 ns, 3.359 KiB allocated","matrix-free, serial, 3D, n = 48 (110592 degrees of freedom): 500.0 ns, 3.828 KiB allocated","matrix-free, serial, 3D, n = 64 (262144 degrees of freedom): 625.0 ns, 4.109 KiB allocated"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 2D", x: [1024,4096,16384,65536,262144], y: [0.001958,0.002125,0.002458,0.00325,0.0043749999999999995], line: { color: '#ef4444', dash: 'solid' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 2D, n = 32 (1024 degrees of freedom): 2.0 μs, 3.828 KiB allocated","matrix-free, CpuThreaded(), 2D, n = 64 (4096 degrees of freedom): 2.1 μs, 4.328 KiB allocated","matrix-free, CpuThreaded(), 2D, n = 128 (16384 degrees of freedom): 2.5 μs, 5.391 KiB allocated","matrix-free, CpuThreaded(), 2D, n = 256 (65536 degrees of freedom): 3.2 μs, 7.328 KiB allocated","matrix-free, CpuThreaded(), 2D, n = 512 (262144 degrees of freedom): 4.4 μs, 11.328 KiB allocated"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 3D", x: [512,4096,32768,110592,262144], y: [0.0025830000000000002,0.002375,0.003,0.003417,0.003375], line: { color: '#ef4444', dash: 'dash' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 3D, n = 8 (512 degrees of freedom): 2.6 μs, 5.188 KiB allocated","matrix-free, CpuThreaded(), 3D, n = 16 (4096 degrees of freedom): 2.4 μs, 5.375 KiB allocated","matrix-free, CpuThreaded(), 3D, n = 32 (32768 degrees of freedom): 3.0 μs, 5.750 KiB allocated","matrix-free, CpuThreaded(), 3D, n = 48 (110592 degrees of freedom): 3.4 μs, 6.219 KiB allocated","matrix-free, CpuThreaded(), 3D, n = 64 (262144 degrees of freedom): 3.4 μs, 6.500 KiB allocated"], hoverinfo: 'text' }];
  // The legend is anchored to the bottom of the chart; its rows depend on the width, so
  // the bottom margin and the chart's height grow with the rows it needs.
  const el = document.getElementById('bench_chart_8');
  const perRow = Math.max(1, Math.floor((el.clientWidth - 40) / 190));
  const legendPx = 22 * Math.ceil(data.length / perRow);
  el.style.height = (420 + legendPx) + 'px';
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: { orientation: 'h', x: 0, xanchor: 'left', yref: 'container', y: 0, yanchor: 'bottom' },
    xaxis: {
      title: { text: "degrees of freedom", font: { color: theme.text }, standoff: 10 },
      type: 'log', color: theme.text, gridcolor: theme.grid, automargin: true,
    },
    yaxis: {
      title: { text: "construction time (ms)", font: { color: theme.text } },
      type: 'log', color: theme.text, gridcolor: theme.grid, automargin: true,
    },
    margin: { t: 20, l: 70, r: 20, b: 70 + legendPx },
  };
  Plotly.newPlot('bench_chart_8', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_8', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.text,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div>
```

#### Product

Time of one operator-vector product, then the bytes the operator holds. The assembled route's bytes count its matrix only, not the scatter cache the assembled form keeps for refilling it.

Run on Apple M2 with 4 threads, power: AC, commit `86e058eb-dirty`, Julia 1.13.1, 2026-10-03T22:22:32. The script waited for a quiet machine before starting; the load average at the end of the run, which includes the run itself, was 5.85.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_9" data-bench="standalone-operator_routes-product" data-run="86e058eb-dirty" style="width:100%; height:420px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ type: 'scatter', mode: 'lines+markers', name: "assembled, 2D", x: [1024,4096,16384,65536,262144], y: [0.003334,0.015333000000000001,0.061166000000000005,0.287375,0.9744579999999999], line: { color: '#3b82f6', dash: 'solid' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 2D, n = 32 (1024 degrees of freedom): 3.3 μs per product, 86.164 KiB held","assembled, 2D, n = 64 (4096 degrees of freedom): 15.3 μs per product, 348.164 KiB held","assembled, 2D, n = 128 (16384 degrees of freedom): 61.2 μs per product, 1.367 MiB held","assembled, 2D, n = 256 (65536 degrees of freedom): 287.4 μs per product, 5.485 MiB held","assembled, 2D, n = 512 (262144 degrees of freedom): 974.5 μs per product, 21.969 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "assembled, 3D", x: [512,4096,32768,110592,262144], y: [0.001958,0.018542,0.1595,0.580792,1.263333], line: { color: '#3b82f6', dash: 'dash' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 3D, n = 8 (512 degrees of freedom): 2.0 μs per product, 54.164 KiB held","assembled, 3D, n = 16 (4096 degrees of freedom): 18.5 μs per product, 456.164 KiB held","assembled, 3D, n = 32 (32768 degrees of freedom): 159.5 μs per product, 3.656 MiB held","assembled, 3D, n = 48 (110592 degrees of freedom): 580.8 μs per product, 12.445 MiB held","assembled, 3D, n = 64 (262144 degrees of freedom): 1.26 ms per product, 29.625 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 2D", x: [1024,4096,16384,65536,262144], y: [0.0014169999999999999,0.005332999999999999,0.021083,0.08820800000000001,0.375958], line: { color: '#f59e0b', dash: 'solid' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 2D, n = 32 (1024 degrees of freedom): 1.4 μs per product, 5.016 KiB held","Kronecker, 2D, n = 64 (4096 degrees of freedom): 5.3 μs per product, 9.516 KiB held","Kronecker, 2D, n = 128 (16384 degrees of freedom): 21.1 μs per product, 18.516 KiB held","Kronecker, 2D, n = 256 (65536 degrees of freedom): 88.2 μs per product, 36.516 KiB held","Kronecker, 2D, n = 512 (262144 degrees of freedom): 376.0 μs per product, 72.516 KiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 3D", x: [512,4096,32768,110592,262144], y: [0.002,0.012416,0.083167,0.261125,0.690834], line: { color: '#f59e0b', dash: 'dash' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 3D, n = 8 (512 degrees of freedom): 2.0 μs per product, 2.398 KiB held","Kronecker, 3D, n = 16 (4096 degrees of freedom): 12.4 μs per product, 4.023 KiB held","Kronecker, 3D, n = 32 (32768 degrees of freedom): 83.2 μs per product, 7.273 KiB held","Kronecker, 3D, n = 48 (110592 degrees of freedom): 261.1 μs per product, 10.523 KiB held","Kronecker, 3D, n = 64 (262144 degrees of freedom): 690.8 μs per product, 13.773 KiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 2D", x: [1024,4096,16384,65536,262144], y: [0.001792,0.006041,0.025,0.0955,0.4105], line: { color: '#10b981', dash: 'solid' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 2D, n = 32 (1024 degrees of freedom): 1.8 μs per product, 712 bytes held","matrix-free, serial, 2D, n = 64 (4096 degrees of freedom): 6.0 μs per product, 1.195 KiB held","matrix-free, serial, 2D, n = 128 (16384 degrees of freedom): 25.0 μs per product, 2.195 KiB held","matrix-free, serial, 2D, n = 256 (65536 degrees of freedom): 95.5 μs per product, 4.195 KiB held","matrix-free, serial, 2D, n = 512 (262144 degrees of freedom): 410.5 μs per product, 8.195 KiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 3D", x: [512,4096,32768,110592,262144], y: [0.003292,0.016125,0.10516700000000001,0.325,0.807875], line: { color: '#10b981', dash: 'dash' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 3D, n = 8 (512 degrees of freedom): 3.3 μs per product, 440 bytes held","matrix-free, serial, 3D, n = 16 (4096 degrees of freedom): 16.1 μs per product, 632 bytes held","matrix-free, serial, 3D, n = 32 (32768 degrees of freedom): 105.2 μs per product, 1016 bytes held","matrix-free, serial, 3D, n = 48 (110592 degrees of freedom): 325.0 μs per product, 1.367 KiB held","matrix-free, serial, 3D, n = 64 (262144 degrees of freedom): 807.9 μs per product, 1.742 KiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 2D", x: [1024,4096,16384,65536,262144], y: [0.005332999999999999,0.009000000000000001,0.02225,0.077,0.30770800000000004], line: { color: '#ef4444', dash: 'solid' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 2D, n = 32 (1024 degrees of freedom): 5.3 μs per product, 1.062 KiB held","matrix-free, CpuThreaded(), 2D, n = 64 (4096 degrees of freedom): 9.0 μs per product, 1.562 KiB held","matrix-free, CpuThreaded(), 2D, n = 128 (16384 degrees of freedom): 22.2 μs per product, 2.562 KiB held","matrix-free, CpuThreaded(), 2D, n = 256 (65536 degrees of freedom): 77.0 μs per product, 4.562 KiB held","matrix-free, CpuThreaded(), 2D, n = 512 (262144 degrees of freedom): 307.7 μs per product, 8.562 KiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 3D", x: [512,4096,32768,110592,262144], y: [0.006709,0.017583,0.080375,0.235625,0.5082920000000001], line: { color: '#ef4444', dash: 'dash' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 3D, n = 8 (512 degrees of freedom): 6.7 μs per product, 1.031 KiB held","matrix-free, CpuThreaded(), 3D, n = 16 (4096 degrees of freedom): 17.6 μs per product, 1.219 KiB held","matrix-free, CpuThreaded(), 3D, n = 32 (32768 degrees of freedom): 80.4 μs per product, 1.594 KiB held","matrix-free, CpuThreaded(), 3D, n = 48 (110592 degrees of freedom): 235.6 μs per product, 1.969 KiB held","matrix-free, CpuThreaded(), 3D, n = 64 (262144 degrees of freedom): 508.3 μs per product, 2.344 KiB held"], hoverinfo: 'text' }];
  // The legend is anchored to the bottom of the chart; its rows depend on the width, so
  // the bottom margin and the chart's height grow with the rows it needs.
  const el = document.getElementById('bench_chart_9');
  const perRow = Math.max(1, Math.floor((el.clientWidth - 40) / 190));
  const legendPx = 22 * Math.ceil(data.length / perRow);
  el.style.height = (420 + legendPx) + 'px';
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: { orientation: 'h', x: 0, xanchor: 'left', yref: 'container', y: 0, yanchor: 'bottom' },
    xaxis: {
      title: { text: "degrees of freedom", font: { color: theme.text }, standoff: 10 },
      type: 'log', color: theme.text, gridcolor: theme.grid, automargin: true,
    },
    yaxis: {
      title: { text: "time of one product (ms)", font: { color: theme.text } },
      type: 'log', color: theme.text, gridcolor: theme.grid, automargin: true,
    },
    margin: { t: 20, l: 70, r: 20, b: 70 + legendPx },
  };
  Plotly.newPlot('bench_chart_9', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_9', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.text,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div>
```

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_10" data-bench="standalone-operator_routes-product-bytes" data-run="86e058eb-dirty" style="width:100%; height:420px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ type: 'scatter', mode: 'lines+markers', name: "assembled, 2D", x: [1024,4096,16384,65536,262144], y: [88232,356520,1433768,5750952,23036072], line: { color: '#3b82f6', dash: 'solid' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 2D, n = 32 (1024 degrees of freedom): 86.164 KiB held","assembled, 2D, n = 64 (4096 degrees of freedom): 348.164 KiB held","assembled, 2D, n = 128 (16384 degrees of freedom): 1.367 MiB held","assembled, 2D, n = 256 (65536 degrees of freedom): 5.485 MiB held","assembled, 2D, n = 512 (262144 degrees of freedom): 21.969 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "assembled, 3D", x: [512,4096,32768,110592,262144], y: [55464,467112,3834024,13050024,31064232], line: { color: '#3b82f6', dash: 'dash' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 3D, n = 8 (512 degrees of freedom): 54.164 KiB held","assembled, 3D, n = 16 (4096 degrees of freedom): 456.164 KiB held","assembled, 3D, n = 32 (32768 degrees of freedom): 3.656 MiB held","assembled, 3D, n = 48 (110592 degrees of freedom): 12.445 MiB held","assembled, 3D, n = 64 (262144 degrees of freedom): 29.625 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 2D", x: [1024,4096,16384,65536,262144], y: [5136,9744,18960,37392,74256], line: { color: '#f59e0b', dash: 'solid' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 2D, n = 32 (1024 degrees of freedom): 5.016 KiB held","Kronecker, 2D, n = 64 (4096 degrees of freedom): 9.516 KiB held","Kronecker, 2D, n = 128 (16384 degrees of freedom): 18.516 KiB held","Kronecker, 2D, n = 256 (65536 degrees of freedom): 36.516 KiB held","Kronecker, 2D, n = 512 (262144 degrees of freedom): 72.516 KiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 3D", x: [512,4096,32768,110592,262144], y: [2456,4120,7448,10776,14104], line: { color: '#f59e0b', dash: 'dash' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 3D, n = 8 (512 degrees of freedom): 2.398 KiB held","Kronecker, 3D, n = 16 (4096 degrees of freedom): 4.023 KiB held","Kronecker, 3D, n = 32 (32768 degrees of freedom): 7.273 KiB held","Kronecker, 3D, n = 48 (110592 degrees of freedom): 10.523 KiB held","Kronecker, 3D, n = 64 (262144 degrees of freedom): 13.773 KiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 2D", x: [1024,4096,16384,65536,262144], y: [712,1224,2248,4296,8392], line: { color: '#10b981', dash: 'solid' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 2D, n = 32 (1024 degrees of freedom): 712 bytes held","matrix-free, serial, 2D, n = 64 (4096 degrees of freedom): 1.195 KiB held","matrix-free, serial, 2D, n = 128 (16384 degrees of freedom): 2.195 KiB held","matrix-free, serial, 2D, n = 256 (65536 degrees of freedom): 4.195 KiB held","matrix-free, serial, 2D, n = 512 (262144 degrees of freedom): 8.195 KiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 3D", x: [512,4096,32768,110592,262144], y: [440,632,1016,1400,1784], line: { color: '#10b981', dash: 'dash' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 3D, n = 8 (512 degrees of freedom): 440 bytes held","matrix-free, serial, 3D, n = 16 (4096 degrees of freedom): 632 bytes held","matrix-free, serial, 3D, n = 32 (32768 degrees of freedom): 1016 bytes held","matrix-free, serial, 3D, n = 48 (110592 degrees of freedom): 1.367 KiB held","matrix-free, serial, 3D, n = 64 (262144 degrees of freedom): 1.742 KiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 2D", x: [1024,4096,16384,65536,262144], y: [1088,1600,2624,4672,8768], line: { color: '#ef4444', dash: 'solid' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 2D, n = 32 (1024 degrees of freedom): 1.062 KiB held","matrix-free, CpuThreaded(), 2D, n = 64 (4096 degrees of freedom): 1.562 KiB held","matrix-free, CpuThreaded(), 2D, n = 128 (16384 degrees of freedom): 2.562 KiB held","matrix-free, CpuThreaded(), 2D, n = 256 (65536 degrees of freedom): 4.562 KiB held","matrix-free, CpuThreaded(), 2D, n = 512 (262144 degrees of freedom): 8.562 KiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 3D", x: [512,4096,32768,110592,262144], y: [1056,1248,1632,2016,2400], line: { color: '#ef4444', dash: 'dash' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 3D, n = 8 (512 degrees of freedom): 1.031 KiB held","matrix-free, CpuThreaded(), 3D, n = 16 (4096 degrees of freedom): 1.219 KiB held","matrix-free, CpuThreaded(), 3D, n = 32 (32768 degrees of freedom): 1.594 KiB held","matrix-free, CpuThreaded(), 3D, n = 48 (110592 degrees of freedom): 1.969 KiB held","matrix-free, CpuThreaded(), 3D, n = 64 (262144 degrees of freedom): 2.344 KiB held"], hoverinfo: 'text' }];
  // The legend is anchored to the bottom of the chart; its rows depend on the width, so
  // the bottom margin and the chart's height grow with the rows it needs.
  const el = document.getElementById('bench_chart_10');
  const perRow = Math.max(1, Math.floor((el.clientWidth - 40) / 190));
  const legendPx = 22 * Math.ceil(data.length / perRow);
  el.style.height = (420 + legendPx) + 'px';
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: { orientation: 'h', x: 0, xanchor: 'left', yref: 'container', y: 0, yanchor: 'bottom' },
    xaxis: {
      title: { text: "degrees of freedom", font: { color: theme.text }, standoff: 10 },
      type: 'log', color: theme.text, gridcolor: theme.grid, automargin: true,
    },
    yaxis: {
      title: { text: "bytes held by the operator", font: { color: theme.text } },
      type: 'log', color: theme.text, gridcolor: theme.grid, automargin: true,
    },
    margin: { t: 20, l: 70, r: 20, b: 70 + legendPx },
  };
  Plotly.newPlot('bench_chart_10', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_10', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.text,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div>
```

#### Solve

Time of the solve of the same problem by conjugate gradients on each route, with `fdm_solve` and the sparse direct solve as reference curves. The CG iteration counts agree across routes to within 0.4% at every size. CG stops when its recursively updated residual reaches the tolerance, which is not the true residual: the largest true relative residual among the CG rows is 7.2e-8, at 2D with n = 512.

Run on Apple M2 with 4 threads, power: AC, commit `86e058eb-dirty`, Julia 1.13.1, 2026-10-03T22:22:32. The script waited for a quiet machine before starting; the load average at the end of the run, which includes the run itself, was 5.85.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_11" data-bench="standalone-operator_routes-solve" data-run="86e058eb-dirty" style="width:100%; height:420px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ type: 'scatter', mode: 'lines+markers', name: "assembled, 2D", x: [1024,4096,16384,65536,262144], y: [1.90275,25.577291,269.58712499999996,3542.7772910000003,39671.728417], line: { color: '#3b82f6', dash: 'solid' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 2D, n = 32 (1024 degrees of freedom): 1.9 ms, 431 CG iterations, true relative residual 8.24e-9","assembled, 2D, n = 64 (4096 degrees of freedom): 25.58 ms, 1251 CG iterations, true relative residual 9.75e-9","assembled, 2D, n = 128 (16384 degrees of freedom): 269.59 ms, 3604 CG iterations, true relative residual 9.95e-9","assembled, 2D, n = 256 (65536 degrees of freedom): 3.54 s, 10646 CG iterations, true relative residual 1.21e-8","assembled, 2D, n = 512 (262144 degrees of freedom): 39.67 s, 30328 CG iterations, true relative residual 7.19e-8"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "assembled, 3D", x: [512,4096,32768,110592,262144], y: [0.366792,8.795542,181.998292,1207.78525,5086.857834], line: { color: '#3b82f6', dash: 'dash' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 3D, n = 8 (512 degrees of freedom): 366.8 μs, 151 CG iterations, true relative residual 9.32e-9","assembled, 3D, n = 16 (4096 degrees of freedom): 8.8 ms, 380 CG iterations, true relative residual 9.51e-9","assembled, 3D, n = 32 (32768 degrees of freedom): 182.0 ms, 993 CG iterations, true relative residual 9.77e-9","assembled, 3D, n = 48 (110592 degrees of freedom): 1.21 s, 1796 CG iterations, true relative residual 9.99e-9","assembled, 3D, n = 64 (262144 degrees of freedom): 5.09 s, 2753 CG iterations, true relative residual 1.0e-8"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 2D", x: [1024,4096,16384,65536,262144], y: [0.901167,11.346541,128.371209,1343.1322910000001,16464.396957999998], line: { color: '#f59e0b', dash: 'solid' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 2D, n = 32 (1024 degrees of freedom): 901.2 μs, 431 CG iterations, true relative residual 8.87e-9","Kronecker, 2D, n = 64 (4096 degrees of freedom): 11.35 ms, 1253 CG iterations, true relative residual 9.31e-9","Kronecker, 2D, n = 128 (16384 degrees of freedom): 128.37 ms, 3614 CG iterations, true relative residual 1.0e-8","Kronecker, 2D, n = 256 (65536 degrees of freedom): 1.34 s, 10634 CG iterations, true relative residual 1.24e-8","Kronecker, 2D, n = 512 (262144 degrees of freedom): 16.46 s, 30335 CG iterations, true relative residual 7.05e-8"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 3D", x: [512,4096,32768,110592,262144], y: [0.36195900000000003,6.125,111.453916,625.174209,2368.935959], line: { color: '#f59e0b', dash: 'dash' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 3D, n = 8 (512 degrees of freedom): 362.0 μs, 151 CG iterations, true relative residual 9.2e-9","Kronecker, 3D, n = 16 (4096 degrees of freedom): 6.12 ms, 380 CG iterations, true relative residual 9.9e-9","Kronecker, 3D, n = 32 (32768 degrees of freedom): 111.45 ms, 993 CG iterations, true relative residual 9.89e-9","Kronecker, 3D, n = 48 (110592 degrees of freedom): 625.17 ms, 1795 CG iterations, true relative residual 9.97e-9","Kronecker, 3D, n = 64 (262144 degrees of freedom): 2.37 s, 2752 CG iterations, true relative residual 1.0e-8"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 2D", x: [1024,4096,16384,65536,262144], y: [1.240042,13.784208,162.040834,1696.4055,17802.258167], line: { color: '#10b981', dash: 'solid' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 2D, n = 32 (1024 degrees of freedom): 1.24 ms, 432 CG iterations, true relative residual 9.11e-9","matrix-free, serial, 2D, n = 64 (4096 degrees of freedom): 13.78 ms, 1256 CG iterations, true relative residual 8.86e-9","matrix-free, serial, 2D, n = 128 (16384 degrees of freedom): 162.04 ms, 3616 CG iterations, true relative residual 9.89e-9","matrix-free, serial, 2D, n = 256 (65536 degrees of freedom): 1.7 s, 10633 CG iterations, true relative residual 1.18e-8","matrix-free, serial, 2D, n = 512 (262144 degrees of freedom): 17.8 s, 30334 CG iterations, true relative residual 6.95e-8"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 3D", x: [512,4096,32768,110592,262144], y: [0.554042,7.30425,127.42062500000002,732.425167,2672.7651250000004], line: { color: '#10b981', dash: 'dash' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 3D, n = 8 (512 degrees of freedom): 554.0 μs, 151 CG iterations, true relative residual 8.76e-9","matrix-free, serial, 3D, n = 16 (4096 degrees of freedom): 7.3 ms, 380 CG iterations, true relative residual 9.73e-9","matrix-free, serial, 3D, n = 32 (32768 degrees of freedom): 127.42 ms, 993 CG iterations, true relative residual 9.6e-9","matrix-free, serial, 3D, n = 48 (110592 degrees of freedom): 732.43 ms, 1794 CG iterations, true relative residual 9.96e-9","matrix-free, serial, 3D, n = 64 (262144 degrees of freedom): 2.67 s, 2752 CG iterations, true relative residual 9.94e-9"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 2D", x: [1024,4096,16384,65536,262144], y: [4.090708,36.585083999999995,215.464875,2102.5867909999997,26464.018375], line: { color: '#ef4444', dash: 'solid' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 2D, n = 32 (1024 degrees of freedom): 4.09 ms, 432 CG iterations, true relative residual 9.11e-9","matrix-free, CpuThreaded(), 2D, n = 64 (4096 degrees of freedom): 36.59 ms, 1256 CG iterations, true relative residual 8.86e-9","matrix-free, CpuThreaded(), 2D, n = 128 (16384 degrees of freedom): 215.46 ms, 3616 CG iterations, true relative residual 9.89e-9","matrix-free, CpuThreaded(), 2D, n = 256 (65536 degrees of freedom): 2.1 s, 10633 CG iterations, true relative residual 1.18e-8","matrix-free, CpuThreaded(), 2D, n = 512 (262144 degrees of freedom): 26.46 s, 30334 CG iterations, true relative residual 6.95e-8"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 3D", x: [512,4096,32768,110592,262144], y: [1.3890829999999998,14.5725,152.10104199999998,922.9957079999999,3034.832125], line: { color: '#ef4444', dash: 'dash' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 3D, n = 8 (512 degrees of freedom): 1.39 ms, 151 CG iterations, true relative residual 8.76e-9","matrix-free, CpuThreaded(), 3D, n = 16 (4096 degrees of freedom): 14.57 ms, 380 CG iterations, true relative residual 9.73e-9","matrix-free, CpuThreaded(), 3D, n = 32 (32768 degrees of freedom): 152.1 ms, 993 CG iterations, true relative residual 9.6e-9","matrix-free, CpuThreaded(), 3D, n = 48 (110592 degrees of freedom): 923.0 ms, 1794 CG iterations, true relative residual 9.96e-9","matrix-free, CpuThreaded(), 3D, n = 64 (262144 degrees of freedom): 3.03 s, 2752 CG iterations, true relative residual 9.94e-9"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "fdm_solve (reference), 2D", x: [1024,4096,16384,65536,262144], y: [0.238042,0.802292,3.026084,15.453167,106.03462499999999], line: { color: '#8b5cf6', dash: 'solid' }, marker: { color: '#8b5cf6', size: 6 }, hovertext: ["fdm_solve (reference), 2D, n = 32 (1024 degrees of freedom): 238.0 μs, 0 CG iterations, true relative residual 3.51e-12","fdm_solve (reference), 2D, n = 64 (4096 degrees of freedom): 802.3 μs, 0 CG iterations, true relative residual 2.79e-11","fdm_solve (reference), 2D, n = 128 (16384 degrees of freedom): 3.03 ms, 0 CG iterations, true relative residual 1.81e-10","fdm_solve (reference), 2D, n = 256 (65536 degrees of freedom): 15.45 ms, 0 CG iterations, true relative residual 1.48e-9","fdm_solve (reference), 2D, n = 512 (262144 degrees of freedom): 106.03 ms, 0 CG iterations, true relative residual 1.57e-8"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "fdm_solve (reference), 3D", x: [512,4096,32768,110592,262144], y: [0.15341700000000003,0.560917,2.6919169999999997,8.886458999999999,20.269416], line: { color: '#8b5cf6', dash: 'dash' }, marker: { color: '#8b5cf6', size: 6 }, hovertext: ["fdm_solve (reference), 3D, n = 8 (512 degrees of freedom): 153.4 μs, 0 CG iterations, true relative residual 2.08e-13","fdm_solve (reference), 3D, n = 16 (4096 degrees of freedom): 560.9 μs, 0 CG iterations, true relative residual 5.13e-12","fdm_solve (reference), 3D, n = 32 (32768 degrees of freedom): 2.69 ms, 0 CG iterations, true relative residual 6.31e-12","fdm_solve (reference), 3D, n = 48 (110592 degrees of freedom): 8.89 ms, 0 CG iterations, true relative residual 2.68e-11","fdm_solve (reference), 3D, n = 64 (262144 degrees of freedom): 20.27 ms, 0 CG iterations, true relative residual 5.45e-11"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "direct (reference), 2D", x: [1024,4096,16384,65536,262144], y: [0.52325,2.60575,9.546208,42.586458,245.98562500000003], line: { color: '#9ca3af', dash: 'solid' }, marker: { color: '#9ca3af', size: 6 }, hovertext: ["direct (reference), 2D, n = 32 (1024 degrees of freedom): 523.2 μs, 0 CG iterations, true relative residual 1.26e-12","direct (reference), 2D, n = 64 (4096 degrees of freedom): 2.61 ms, 0 CG iterations, true relative residual 7.3e-12","direct (reference), 2D, n = 128 (16384 degrees of freedom): 9.55 ms, 0 CG iterations, true relative residual 2.92e-11","direct (reference), 2D, n = 256 (65536 degrees of freedom): 42.59 ms, 0 CG iterations, true relative residual 1.6e-10","direct (reference), 2D, n = 512 (262144 degrees of freedom): 245.99 ms, 0 CG iterations, true relative residual 8.96e-10"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "direct (reference), 3D", x: [512,4096,32768,110592,262144], y: [0.534458,5.759334,145.2255,1469.591834,8080.564], line: { color: '#9ca3af', dash: 'dash' }, marker: { color: '#9ca3af', size: 6 }, hovertext: ["direct (reference), 3D, n = 8 (512 degrees of freedom): 534.5 μs, 0 CG iterations, true relative residual 6.55e-14","direct (reference), 3D, n = 16 (4096 degrees of freedom): 5.76 ms, 0 CG iterations, true relative residual 3.54e-13","direct (reference), 3D, n = 32 (32768 degrees of freedom): 145.23 ms, 0 CG iterations, true relative residual 1.83e-12","direct (reference), 3D, n = 48 (110592 degrees of freedom): 1.47 s, 0 CG iterations, true relative residual 4.87e-12","direct (reference), 3D, n = 64 (262144 degrees of freedom): 8.08 s, 0 CG iterations, true relative residual 1.05e-11"], hoverinfo: 'text' }];
  // The legend is anchored to the bottom of the chart; its rows depend on the width, so
  // the bottom margin and the chart's height grow with the rows it needs.
  const el = document.getElementById('bench_chart_11');
  const perRow = Math.max(1, Math.floor((el.clientWidth - 40) / 190));
  const legendPx = 22 * Math.ceil(data.length / perRow);
  el.style.height = (420 + legendPx) + 'px';
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: { orientation: 'h', x: 0, xanchor: 'left', yref: 'container', y: 0, yanchor: 'bottom' },
    xaxis: {
      title: { text: "degrees of freedom", font: { color: theme.text }, standoff: 10 },
      type: 'log', color: theme.text, gridcolor: theme.grid, automargin: true,
    },
    yaxis: {
      title: { text: "solve time (ms)", font: { color: theme.text } },
      type: 'log', color: theme.text, gridcolor: theme.grid, automargin: true,
    },
    margin: { t: 20, l: 70, r: 20, b: 70 + legendPx },
  };
  Plotly.newPlot('bench_chart_11', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_11', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.text,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div>
```

### Matrix-free against SpMV (`matrix_free_spmv.jl`)

#### Time

One product `mul!(y, matrix_free_operator(a), x)` against one serial sparse matrix-vector product with the assembled matrix, on non-uniform meshes in 1D, 2D and 3D, by degrees of freedom.

Run on Apple M2 with 4 threads, power: AC, commit `86e058eb-dirty`, Julia 1.13.1, 2026-10-03T22:23:29. The script waited for a quiet machine before starting; the load average at the end of the run, which includes the run itself, was 3.58.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_12" data-bench="standalone-matrix_free_spmv-time" data-run="86e058eb-dirty" style="width:100%; height:420px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 1D", x: [1000,10000,100000,1000000,10000000], y: [0.0012920000000000002,0.014292,0.138458,1.566167,17.329416000000002], line: { color: '#10b981', dash: 'solid' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 1D, 1000 degrees of freedom: 1.3 μs, ×1.9 the SpMV speed","matrix-free, serial, 1D, 10000 degrees of freedom: 14.3 μs, ×1.97 the SpMV speed","matrix-free, serial, 1D, 100000 degrees of freedom: 138.5 μs, ×2.21 the SpMV speed","matrix-free, serial, 1D, 1000000 degrees of freedom: 1.57 ms, ×1.82 the SpMV speed","matrix-free, serial, 1D, 10000000 degrees of freedom: 17.33 ms, ×1.72 the SpMV speed"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "SpMV, 1D", x: [1000,10000,100000,1000000,10000000], y: [0.002458,0.028084,0.305417,2.84975,29.734083000000002], line: { color: '#3b82f6', dash: 'solid' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["SpMV, 1D, 1000 degrees of freedom: 2.5 μs","SpMV, 1D, 10000 degrees of freedom: 28.1 μs","SpMV, 1D, 100000 degrees of freedom: 305.4 μs","SpMV, 1D, 1000000 degrees of freedom: 2.85 ms","SpMV, 1D, 10000000 degrees of freedom: 29.73 ms"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 1D", x: [1000,10000,100000,1000000,10000000], y: [0.005083000000000001,0.00925,0.054416,1.286541,12.905167], line: { color: '#ef4444', dash: 'solid' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 1D, 1000 degrees of freedom: 5.1 μs, ×0.48 the SpMV speed","matrix-free, CpuThreaded(), 1D, 10000 degrees of freedom: 9.2 μs, ×3.04 the SpMV speed","matrix-free, CpuThreaded(), 1D, 100000 degrees of freedom: 54.4 μs, ×5.61 the SpMV speed","matrix-free, CpuThreaded(), 1D, 1000000 degrees of freedom: 1.29 ms, ×2.22 the SpMV speed","matrix-free, CpuThreaded(), 1D, 10000000 degrees of freedom: 12.91 ms, ×2.3 the SpMV speed"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 2D", x: [1024,4096,16384,65536,262144,1048576,4194304], y: [0.00625,0.018541000000000002,0.054709,0.189666,0.694917,2.882208,12.922583], line: { color: '#10b981', dash: 'dash' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 2D, 1024 degrees of freedom: 6.2 μs, ×0.57 the SpMV speed","matrix-free, serial, 2D, 4096 degrees of freedom: 18.5 μs, ×0.84 the SpMV speed","matrix-free, serial, 2D, 16384 degrees of freedom: 54.7 μs, ×1.06 the SpMV speed","matrix-free, serial, 2D, 65536 degrees of freedom: 189.7 μs, ×1.42 the SpMV speed","matrix-free, serial, 2D, 262144 degrees of freedom: 694.9 μs, ×1.41 the SpMV speed","matrix-free, serial, 2D, 1048576 degrees of freedom: 2.88 ms, ×1.39 the SpMV speed","matrix-free, serial, 2D, 4194304 degrees of freedom: 12.92 ms, ×1.29 the SpMV speed"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "SpMV, 2D", x: [1024,4096,16384,65536,262144,1048576,4194304], y: [0.003542,0.0155,0.058249999999999996,0.270166,0.9777919999999999,4.006832999999999,16.630333], line: { color: '#3b82f6', dash: 'dash' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["SpMV, 2D, 1024 degrees of freedom: 3.5 μs","SpMV, 2D, 4096 degrees of freedom: 15.5 μs","SpMV, 2D, 16384 degrees of freedom: 58.2 μs","SpMV, 2D, 65536 degrees of freedom: 270.2 μs","SpMV, 2D, 262144 degrees of freedom: 977.8 μs","SpMV, 2D, 1048576 degrees of freedom: 4.01 ms","SpMV, 2D, 4194304 degrees of freedom: 16.63 ms"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 2D", x: [1024,4096,16384,65536,262144,1048576,4194304], y: [0.011125,0.010875,0.02175,0.07054200000000001,0.249917,1.075833,5.686], line: { color: '#ef4444', dash: 'dash' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 2D, 1024 degrees of freedom: 11.1 μs, ×0.32 the SpMV speed","matrix-free, CpuThreaded(), 2D, 4096 degrees of freedom: 10.9 μs, ×1.43 the SpMV speed","matrix-free, CpuThreaded(), 2D, 16384 degrees of freedom: 21.8 μs, ×2.68 the SpMV speed","matrix-free, CpuThreaded(), 2D, 65536 degrees of freedom: 70.5 μs, ×3.83 the SpMV speed","matrix-free, CpuThreaded(), 2D, 262144 degrees of freedom: 249.9 μs, ×3.91 the SpMV speed","matrix-free, CpuThreaded(), 2D, 1048576 degrees of freedom: 1.08 ms, ×3.72 the SpMV speed","matrix-free, CpuThreaded(), 2D, 4194304 degrees of freedom: 5.69 ms, ×2.92 the SpMV speed"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 3D", x: [4096,32768,262144,884736,2097152], y: [0.069375,0.359625,1.955625,5.677874999999999,14.011833], line: { color: '#10b981', dash: 'dot' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 3D, 4096 degrees of freedom: 69.4 μs, ×0.26 the SpMV speed","matrix-free, serial, 3D, 32768 degrees of freedom: 359.6 μs, ×0.42 the SpMV speed","matrix-free, serial, 3D, 262144 degrees of freedom: 1.96 ms, ×0.63 the SpMV speed","matrix-free, serial, 3D, 884736 degrees of freedom: 5.68 ms, ×0.73 the SpMV speed","matrix-free, serial, 3D, 2097152 degrees of freedom: 14.01 ms, ×0.74 the SpMV speed"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "SpMV, 3D", x: [4096,32768,262144,884736,2097152], y: [0.018209000000000003,0.150333,1.234,4.1505,10.407542000000001], line: { color: '#3b82f6', dash: 'dot' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["SpMV, 3D, 4096 degrees of freedom: 18.2 μs","SpMV, 3D, 32768 degrees of freedom: 150.3 μs","SpMV, 3D, 262144 degrees of freedom: 1.23 ms","SpMV, 3D, 884736 degrees of freedom: 4.15 ms","SpMV, 3D, 2097152 degrees of freedom: 10.41 ms"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 3D", x: [4096,32768,262144,884736,2097152], y: [0.026834,0.133833,0.686333,1.895458,5.284291], line: { color: '#ef4444', dash: 'dot' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 3D, 4096 degrees of freedom: 26.8 μs, ×0.68 the SpMV speed","matrix-free, CpuThreaded(), 3D, 32768 degrees of freedom: 133.8 μs, ×1.12 the SpMV speed","matrix-free, CpuThreaded(), 3D, 262144 degrees of freedom: 686.3 μs, ×1.8 the SpMV speed","matrix-free, CpuThreaded(), 3D, 884736 degrees of freedom: 1.9 ms, ×2.19 the SpMV speed","matrix-free, CpuThreaded(), 3D, 2097152 degrees of freedom: 5.28 ms, ×1.97 the SpMV speed"], hoverinfo: 'text' }];
  // The legend is anchored to the bottom of the chart; its rows depend on the width, so
  // the bottom margin and the chart's height grow with the rows it needs.
  const el = document.getElementById('bench_chart_12');
  const perRow = Math.max(1, Math.floor((el.clientWidth - 40) / 190));
  const legendPx = 22 * Math.ceil(data.length / perRow);
  el.style.height = (420 + legendPx) + 'px';
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: { orientation: 'h', x: 0, xanchor: 'left', yref: 'container', y: 0, yanchor: 'bottom' },
    xaxis: {
      title: { text: "degrees of freedom", font: { color: theme.text }, standoff: 10 },
      type: 'log', color: theme.text, gridcolor: theme.grid, automargin: true,
    },
    yaxis: {
      title: { text: "time of one product (ms)", font: { color: theme.text } },
      type: 'log', color: theme.text, gridcolor: theme.grid, automargin: true,
    },
    margin: { t: 20, l: 70, r: 20, b: 70 + legendPx },
  };
  Plotly.newPlot('bench_chart_12', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_12', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.text,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div>
```

#### Memory

Bytes the assembled matrix holds against the bytes everything the matrix-free operator keeps alive holds, including the form it captures.

The smallest size from which matrix-free beats the SpMV at every larger size tested:

- 1D, serial: from 1000 degrees of freedom, where the assembled matrix holds 1.0 times the bytes of the matrix-free operator
- 1D, threaded: from 10000 degrees of freedom, where the assembled matrix holds 1.0 times the bytes of the matrix-free operator
- 2D, serial: from 16384 degrees of freedom, where the assembled matrix holds 9.6 times the bytes of the matrix-free operator
- 2D, threaded: from 4096 degrees of freedom, where the assembled matrix holds 8.3 times the bytes of the matrix-free operator
- 3D, serial: no size in the range
- 3D, threaded: from 32768 degrees of freedom, where the assembled matrix holds 13.6 times the bytes of the matrix-free operator

Run on Apple M2 with 4 threads, power: AC, commit `86e058eb-dirty`, Julia 1.13.1, 2026-10-03T22:23:29. The script waited for a quiet machine before starting; the load average at the end of the run, which includes the run itself, was 3.58.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_13" data-bench="standalone-matrix_free_spmv-memory" data-run="86e058eb-dirty" style="width:100%; height:420px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 1D", x: [1000,10000,100000,1000000,10000000], y: [57592,563848,5626344,56251336,562501336], line: { color: '#10b981', dash: 'solid' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 1D, 1000 degrees of freedom: 56.242 KiB held","matrix-free, serial, 1D, 10000 degrees of freedom: 550.633 KiB held","matrix-free, serial, 1D, 100000 degrees of freedom: 5.366 MiB held","matrix-free, serial, 1D, 1000000 degrees of freedom: 53.645 MiB held","matrix-free, serial, 1D, 10000000 degrees of freedom: 536.443 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "assembled matrix, 1D", x: [1000,10000,100000,1000000,10000000], y: [56136,560136,5600136,56000136,560000136], line: { color: '#3b82f6', dash: 'solid' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled matrix, 1D, 1000 degrees of freedom: 54.820 KiB held","assembled matrix, 1D, 10000 degrees of freedom: 547.008 KiB held","assembled matrix, 1D, 100000 degrees of freedom: 5.341 MiB held","assembled matrix, 1D, 1000000 degrees of freedom: 53.406 MiB held","assembled matrix, 1D, 10000000 degrees of freedom: 534.058 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 1D", x: [1000,10000,100000,1000000,10000000], y: [57840,564096,5626592,56251584,562501584], line: { color: '#ef4444', dash: 'solid' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 1D, 1000 degrees of freedom: 56.484 KiB held","matrix-free, CpuThreaded(), 1D, 10000 degrees of freedom: 550.875 KiB held","matrix-free, CpuThreaded(), 1D, 100000 degrees of freedom: 5.366 MiB held","matrix-free, CpuThreaded(), 1D, 1000000 degrees of freedom: 53.646 MiB held","matrix-free, CpuThreaded(), 1D, 10000000 degrees of freedom: 536.443 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 2D", x: [1024,4096,16384,65536,262144,1048576,4194304], y: [14376,42280,148808,564616,2207240,8736520,34771208], line: { color: '#10b981', dash: 'dash' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 2D, 1024 degrees of freedom: 14.039 KiB held","matrix-free, serial, 2D, 4096 degrees of freedom: 41.289 KiB held","matrix-free, serial, 2D, 16384 degrees of freedom: 145.320 KiB held","matrix-free, serial, 2D, 65536 degrees of freedom: 551.383 KiB held","matrix-free, serial, 2D, 262144 degrees of freedom: 2.105 MiB held","matrix-free, serial, 2D, 1048576 degrees of freedom: 8.332 MiB held","matrix-free, serial, 2D, 4194304 degrees of freedom: 33.160 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "assembled matrix, 2D", x: [1024,4096,16384,65536,262144,1048576,4194304], y: [88232,356520,1433768,5750952,23036072,92209320,368967848], line: { color: '#3b82f6', dash: 'dash' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled matrix, 2D, 1024 degrees of freedom: 86.164 KiB held","assembled matrix, 2D, 4096 degrees of freedom: 348.164 KiB held","assembled matrix, 2D, 16384 degrees of freedom: 1.367 MiB held","assembled matrix, 2D, 65536 degrees of freedom: 5.485 MiB held","assembled matrix, 2D, 262144 degrees of freedom: 21.969 MiB held","assembled matrix, 2D, 1048576 degrees of freedom: 87.938 MiB held","assembled matrix, 2D, 4194304 degrees of freedom: 351.875 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 2D", x: [1024,4096,16384,65536,262144,1048576,4194304], y: [15024,42928,149456,565264,2207888,8737168,34771856], line: { color: '#ef4444', dash: 'dash' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 2D, 1024 degrees of freedom: 14.672 KiB held","matrix-free, CpuThreaded(), 2D, 4096 degrees of freedom: 41.922 KiB held","matrix-free, CpuThreaded(), 2D, 16384 degrees of freedom: 145.953 KiB held","matrix-free, CpuThreaded(), 2D, 65536 degrees of freedom: 552.016 KiB held","matrix-free, CpuThreaded(), 2D, 262144 degrees of freedom: 2.106 MiB held","matrix-free, CpuThreaded(), 2D, 1048576 degrees of freedom: 8.332 MiB held","matrix-free, CpuThreaded(), 2D, 4194304 degrees of freedom: 33.161 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 3D", x: [4096,32768,262144,884736,2097152], y: [41152,279616,2175808,7316080,17322352], line: { color: '#10b981', dash: 'dot' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 3D, 4096 degrees of freedom: 40.188 KiB held","matrix-free, serial, 3D, 32768 degrees of freedom: 273.062 KiB held","matrix-free, serial, 3D, 262144 degrees of freedom: 2.075 MiB held","matrix-free, serial, 3D, 884736 degrees of freedom: 6.977 MiB held","matrix-free, serial, 3D, 2097152 degrees of freedom: 16.520 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "assembled matrix, 3D", x: [4096,32768,262144,884736,2097152], y: [467112,3834024,31064232,105283752,250085544], line: { color: '#3b82f6', dash: 'dot' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled matrix, 3D, 4096 degrees of freedom: 456.164 KiB held","assembled matrix, 3D, 32768 degrees of freedom: 3.656 MiB held","assembled matrix, 3D, 262144 degrees of freedom: 29.625 MiB held","assembled matrix, 3D, 884736 degrees of freedom: 100.406 MiB held","assembled matrix, 3D, 2097152 degrees of freedom: 238.500 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 3D", x: [4096,32768,262144,884736,2097152], y: [42520,280984,2177176,7317448,17323720], line: { color: '#ef4444', dash: 'dot' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 3D, 4096 degrees of freedom: 41.523 KiB held","matrix-free, CpuThreaded(), 3D, 32768 degrees of freedom: 274.398 KiB held","matrix-free, CpuThreaded(), 3D, 262144 degrees of freedom: 2.076 MiB held","matrix-free, CpuThreaded(), 3D, 884736 degrees of freedom: 6.978 MiB held","matrix-free, CpuThreaded(), 3D, 2097152 degrees of freedom: 16.521 MiB held"], hoverinfo: 'text' }];
  // The legend is anchored to the bottom of the chart; its rows depend on the width, so
  // the bottom margin and the chart's height grow with the rows it needs.
  const el = document.getElementById('bench_chart_13');
  const perRow = Math.max(1, Math.floor((el.clientWidth - 40) / 190));
  const legendPx = 22 * Math.ceil(data.length / perRow);
  el.style.height = (420 + legendPx) + 'px';
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    legend: { orientation: 'h', x: 0, xanchor: 'left', yref: 'container', y: 0, yanchor: 'bottom' },
    xaxis: {
      title: { text: "degrees of freedom", font: { color: theme.text }, standoff: 10 },
      type: 'log', color: theme.text, gridcolor: theme.grid, automargin: true,
    },
    yaxis: {
      title: { text: "bytes held", font: { color: theme.text } },
      type: 'log', color: theme.text, gridcolor: theme.grid, automargin: true,
    },
    margin: { t: 20, l: 70, r: 20, b: 70 + legendPx },
  };
  Plotly.newPlot('bench_chart_13', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_13', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'xaxis.color': t.text, 'xaxis.gridcolor': t.grid, 'xaxis.title.font.color': t.text,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
    };
  });
})();
</script>
</div>
```

### Execution-policy crossover (`policy_crossover.jl`)

The smallest size, in degrees of freedom, from which each host policy beats the one it is compared with, per workload and dimension, on a non-uniform mesh; a win counts only when the next larger size wins too. A missing bar means the sweep found no crossover, and the crossover depends on the thread count of the run.

Run on Apple M2 with 4 threads, power: AC, commit `a6e862d4`, Julia 1.13.1, 2026-10-02T17:42:22. The script waited for a quiet machine before starting; the load average at the end of the run, which includes the run itself, was 5.03.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_14" data-bench="standalone-policy_crossover" data-run="a6e862d4" style="width:100%; height:520px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ name: "CpuThreaded() beats serial", x: ["Rₕ! unmasked 1D","Rₕ! unmasked 2D","Rₕ! unmasked 3D","Rₕ! masked 1D","Rₕ! masked 2D","Rₕ! masked 3D","avgₕ! 1D","avgₕ! 2D","avgₕ! 3D","innerₕ 1D","innerₕ 2D","innerₕ 3D","innerₕ masked 1D","innerₕ masked 2D","innerₕ masked 3D","D₋ₓ! 1D","D₋ₓ! 2D","D₋ₓ! 3D","broadcast axpy 1D","broadcast axpy 2D","broadcast axpy 3D","assemble bilinear first 1D","assemble bilinear first 2D","assemble bilinear first 3D","assemble! bilinear 1D","assemble! bilinear 2D","assemble! bilinear 3D","assemble! linear 1D","assemble! linear 2D","assemble! linear 3D"], y: [10000,3025,2744,100000,99856,10648,1000,289,125,null,300304,300763,null,null,300763,300000,300304,300763,300000,300304,300763,1000000,null,1000000,100000,99856,97336,100000,99856,97336], type: 'bar', marker: { color: '#3b82f6' }, hovertext: ["Rₕ! unmasked 1D: CpuThreaded() beats serial from 10000 degrees of freedom","Rₕ! unmasked 2D: CpuThreaded() beats serial from 3025 degrees of freedom","Rₕ! unmasked 3D: CpuThreaded() beats serial from 2744 degrees of freedom","Rₕ! masked 1D: CpuThreaded() beats serial from 100000 degrees of freedom","Rₕ! masked 2D: CpuThreaded() beats serial from 99856 degrees of freedom","Rₕ! masked 3D: CpuThreaded() beats serial from 10648 degrees of freedom","avgₕ! 1D: CpuThreaded() beats serial from 1000 degrees of freedom","avgₕ! 2D: CpuThreaded() beats serial from 289 degrees of freedom","avgₕ! 3D: CpuThreaded() beats serial from 125 degrees of freedom","innerₕ 1D: no crossover in the sweep","innerₕ 2D: CpuThreaded() beats serial from 300304 degrees of freedom","innerₕ 3D: CpuThreaded() beats serial from 300763 degrees of freedom","innerₕ masked 1D: no crossover in the sweep","innerₕ masked 2D: no crossover in the sweep","innerₕ masked 3D: CpuThreaded() beats serial from 300763 degrees of freedom","D₋ₓ! 1D: CpuThreaded() beats serial from 300000 degrees of freedom","D₋ₓ! 2D: CpuThreaded() beats serial from 300304 degrees of freedom","D₋ₓ! 3D: CpuThreaded() beats serial from 300763 degrees of freedom","broadcast axpy 1D: CpuThreaded() beats serial from 300000 degrees of freedom","broadcast axpy 2D: CpuThreaded() beats serial from 300304 degrees of freedom","broadcast axpy 3D: CpuThreaded() beats serial from 300763 degrees of freedom","assemble bilinear first 1D: CpuThreaded() beats serial from 1000000 degrees of freedom","assemble bilinear first 2D: no crossover in the sweep","assemble bilinear first 3D: CpuThreaded() beats serial from 1000000 degrees of freedom","assemble! bilinear 1D: CpuThreaded() beats serial from 100000 degrees of freedom","assemble! bilinear 2D: CpuThreaded() beats serial from 99856 degrees of freedom","assemble! bilinear 3D: CpuThreaded() beats serial from 97336 degrees of freedom","assemble! linear 1D: CpuThreaded() beats serial from 100000 degrees of freedom","assemble! linear 2D: CpuThreaded() beats serial from 99856 degrees of freedom","assemble! linear 3D: CpuThreaded() beats serial from 97336 degrees of freedom"], hoverinfo: 'text' },
{ name: "CpuPolyester() beats serial", x: ["Rₕ! unmasked 1D","Rₕ! unmasked 2D","Rₕ! unmasked 3D","Rₕ! masked 1D","Rₕ! masked 2D","Rₕ! masked 3D","avgₕ! 1D","avgₕ! 2D","avgₕ! 3D","innerₕ 1D","innerₕ 2D","innerₕ 3D","innerₕ masked 1D","innerₕ masked 2D","innerₕ masked 3D","D₋ₓ! 1D","D₋ₓ! 2D","D₋ₓ! 3D","broadcast axpy 1D","broadcast axpy 2D","broadcast axpy 3D","assemble bilinear first 1D","assemble bilinear first 2D","assemble bilinear first 3D","assemble! bilinear 1D","assemble! bilinear 2D","assemble! bilinear 3D","assemble! linear 1D","assemble! linear 2D","assemble! linear 3D"], y: [100,100,125,300,100,125,100,100,125,null,10000,10648,null,null,343,3000,1024,1000,10000,10000,10648,1000000,null,null,1000,1024,125,1000,1024,1000], type: 'bar', marker: { color: '#f59e0b' }, hovertext: ["Rₕ! unmasked 1D: CpuPolyester() beats serial from 100 degrees of freedom","Rₕ! unmasked 2D: CpuPolyester() beats serial from 100 degrees of freedom","Rₕ! unmasked 3D: CpuPolyester() beats serial from 125 degrees of freedom","Rₕ! masked 1D: CpuPolyester() beats serial from 300 degrees of freedom","Rₕ! masked 2D: CpuPolyester() beats serial from 100 degrees of freedom","Rₕ! masked 3D: CpuPolyester() beats serial from 125 degrees of freedom","avgₕ! 1D: CpuPolyester() beats serial from 100 degrees of freedom","avgₕ! 2D: CpuPolyester() beats serial from 100 degrees of freedom","avgₕ! 3D: CpuPolyester() beats serial from 125 degrees of freedom","innerₕ 1D: no crossover in the sweep","innerₕ 2D: CpuPolyester() beats serial from 10000 degrees of freedom","innerₕ 3D: CpuPolyester() beats serial from 10648 degrees of freedom","innerₕ masked 1D: no crossover in the sweep","innerₕ masked 2D: no crossover in the sweep","innerₕ masked 3D: CpuPolyester() beats serial from 343 degrees of freedom","D₋ₓ! 1D: CpuPolyester() beats serial from 3000 degrees of freedom","D₋ₓ! 2D: CpuPolyester() beats serial from 1024 degrees of freedom","D₋ₓ! 3D: CpuPolyester() beats serial from 1000 degrees of freedom","broadcast axpy 1D: CpuPolyester() beats serial from 10000 degrees of freedom","broadcast axpy 2D: CpuPolyester() beats serial from 10000 degrees of freedom","broadcast axpy 3D: CpuPolyester() beats serial from 10648 degrees of freedom","assemble bilinear first 1D: CpuPolyester() beats serial from 1000000 degrees of freedom","assemble bilinear first 2D: no crossover in the sweep","assemble bilinear first 3D: no crossover in the sweep","assemble! bilinear 1D: CpuPolyester() beats serial from 1000 degrees of freedom","assemble! bilinear 2D: CpuPolyester() beats serial from 1024 degrees of freedom","assemble! bilinear 3D: CpuPolyester() beats serial from 125 degrees of freedom","assemble! linear 1D: CpuPolyester() beats serial from 1000 degrees of freedom","assemble! linear 2D: CpuPolyester() beats serial from 1024 degrees of freedom","assemble! linear 3D: CpuPolyester() beats serial from 1000 degrees of freedom"], hoverinfo: 'text' },
{ name: "CpuPolyester() beats CpuThreaded()", x: ["Rₕ! unmasked 1D","Rₕ! unmasked 2D","Rₕ! unmasked 3D","Rₕ! masked 1D","Rₕ! masked 2D","Rₕ! masked 3D","avgₕ! 1D","avgₕ! 2D","avgₕ! 3D","innerₕ 1D","innerₕ 2D","innerₕ 3D","innerₕ masked 1D","innerₕ masked 2D","innerₕ masked 3D","D₋ₓ! 1D","D₋ₓ! 2D","D₋ₓ! 3D","broadcast axpy 1D","broadcast axpy 2D","broadcast axpy 3D","assemble bilinear first 1D","assemble bilinear first 2D","assemble bilinear first 3D","assemble! bilinear 1D","assemble! bilinear 2D","assemble! bilinear 3D","assemble! linear 1D","assemble! linear 2D","assemble! linear 3D"], y: [100,100,125,100,100,125,100,100,125,100,100,125,100,100,125,100,100,125,100,100,125,100,100,125,100,100,125,100,100,125], type: 'bar', marker: { color: '#10b981' }, hovertext: ["Rₕ! unmasked 1D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","Rₕ! unmasked 2D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","Rₕ! unmasked 3D: CpuPolyester() beats CpuThreaded() from 125 degrees of freedom","Rₕ! masked 1D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","Rₕ! masked 2D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","Rₕ! masked 3D: CpuPolyester() beats CpuThreaded() from 125 degrees of freedom","avgₕ! 1D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","avgₕ! 2D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","avgₕ! 3D: CpuPolyester() beats CpuThreaded() from 125 degrees of freedom","innerₕ 1D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","innerₕ 2D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","innerₕ 3D: CpuPolyester() beats CpuThreaded() from 125 degrees of freedom","innerₕ masked 1D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","innerₕ masked 2D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","innerₕ masked 3D: CpuPolyester() beats CpuThreaded() from 125 degrees of freedom","D₋ₓ! 1D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","D₋ₓ! 2D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","D₋ₓ! 3D: CpuPolyester() beats CpuThreaded() from 125 degrees of freedom","broadcast axpy 1D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","broadcast axpy 2D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","broadcast axpy 3D: CpuPolyester() beats CpuThreaded() from 125 degrees of freedom","assemble bilinear first 1D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","assemble bilinear first 2D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","assemble bilinear first 3D: CpuPolyester() beats CpuThreaded() from 125 degrees of freedom","assemble! bilinear 1D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","assemble! bilinear 2D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","assemble! bilinear 3D: CpuPolyester() beats CpuThreaded() from 125 degrees of freedom","assemble! linear 1D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","assemble! linear 2D: CpuPolyester() beats CpuThreaded() from 100 degrees of freedom","assemble! linear 3D: CpuPolyester() beats CpuThreaded() from 125 degrees of freedom"], hoverinfo: 'text' }];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    barmode: 'group',
    legend: { orientation: 'h', y: -0.25 },
    shapes: [],
    yaxis: {
      title: { text: "degrees of freedom of the crossover", font: { color: theme.text } },
      type: 'log', color: theme.text, gridcolor: theme.grid,
    },
    xaxis: { color: theme.text, automargin: true, tickangle: -30 },
    margin: { t: 20, l: 70, r: 20, b: 130 },
  };
  Plotly.newPlot('bench_chart_14', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_14', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
      'xaxis.color': t.text,
    };
  });
})();
</script>
</div>
```

### Host against device offload (`gpu_offload.jl`)

`CpuThreaded()` against `GpuOffload(metal_backend(), CpuThreaded())` on the same grid; a bar above the dotted line means the offload was faster. The host arm runs in `Float64` and the offload arm in `Float32`, which Metal requires.

Run on Apple M2 with 4 threads, power: AC, commit `af288c31`, Julia 1.13.1, 2026-10-02T17:42:59. The script waited for a quiet machine before starting; the load average at the end of the run, which includes the run itself, was 3.74.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_15" data-bench="standalone-gpu_offload" data-run="af288c31" style="width:100%; height:380px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ name: "threaded host time / offload time", x: ["Rₕ!, 1D n=10000000","Rₕ!, 2D 3000x3000","avgₕ!, 1D n=10000000","avgₕ!, 2D 3000x3000"], y: [1.0314036746139335,1.8040043321483634,3.498276182414996,11.123185765093265], type: 'bar', marker: { color: '#3b82f6' }, hovertext: ["Rₕ!, 1D n=10000000: 11.89 ms threaded, 11.52 ms offloaded, ×1.03","Rₕ!, 2D 3000x3000: 17.7 ms threaded, 9.81 ms offloaded, ×1.8","avgₕ!, 1D n=10000000: 42.36 ms threaded, 12.11 ms offloaded, ×3.5","avgₕ!, 2D 3000x3000: 155.85 ms threaded, 14.01 ms offloaded, ×11.12"], hoverinfo: 'text' }];
  const layout = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    font: { color: theme.text },
    barmode: 'group',
    legend: { orientation: 'h', y: -0.25 },
    shapes: [{ type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: 1, y1: 1, line: { color: theme.text, width: 1, dash: 'dot' } }],
    yaxis: {
      title: { text: "threaded host time / offload time", font: { color: theme.text } },
      color: theme.text, gridcolor: theme.grid,
    },
    xaxis: { color: theme.text, automargin: true, tickangle: -30 },
    margin: { t: 20, l: 70, r: 20, b: 130 },
  };
  Plotly.newPlot('bench_chart_15', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_15', function () {
    const t = window.bramblePlotlyTheme();
    return {
      'font.color': t.text,
      'yaxis.color': t.text, 'yaxis.gridcolor': t.grid, 'yaxis.title.font.color': t.text,
      'xaxis.color': t.text,
    };
  });
})();
</script>
</div>
```

## How to add new benchmark runs

To record performance on a new commit or after an optimization pass, run:

```bash
julia --project=benchmark benchmark/benchmarks.jl --save benchmark/baselines/baseline_$(git rev-parse --short HEAD).json
```

Rebuilding the documentation (`julia -e 'using Pkg; Pkg.activate("docs"); include("docs/make.jl")'`) will automatically discover all `baseline_*.json` files and update the comparison with the previous baseline above.
