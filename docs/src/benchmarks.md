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

Every chart in this section reads the latest baseline alone (v3.20.0, `ba309208`), recorded with 4 threads, and finds its pairs by benchmark name; a comparison whose benchmarks the baseline lacks is left out.

### Execution policy

Each bar is the median of the `Serial()` run of a benchmark divided by its median under `Parallel()` or, where the run has it, `CpuPolyester()`; a bar above the dotted line is faster than `Serial()`. The benchmarks are paired by name from the latest run alone.

- `restriction`: Point interpolation (`Rₕ!`) and cell-averaging (`avgₕ!`), compared across the `Serial()` (the allocation-free default), `Parallel()` and `CpuPolyester()` backends, split by dimension.
- `forms`: Linear and bilinear form assembly, across 1D/2D and the `Serial()`/`Parallel()`/`CpuPolyester()` backends, and the refill and `assemble_add!` paths on a fixed matrix pattern.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_1" data-bench="policy" data-run="ba309208" style="width:100%; height:380px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ name: "Parallel()", x: ["Rₕ! 1D","Rₕ! 2D","Rₕ! 3D","avgₕ! 1D","avgₕ! 2D","avgₕ! 3D","assemble (BilinearForm) 2D"], y: [2.1952889286770287,3.1048138610792937,2.9971308261864,2.4323009373940545,3.359134264559339,3.510131479215211,1.0801449426991043], type: 'bar', marker: { color: '#3b82f6' }, hovertext: ["Rₕ! 1D: 1.34 ms, ×2.2 faster than Serial()","Rₕ! 2D: 1.28 ms, ×3.1 faster than Serial()","Rₕ! 3D: 1.47 ms, ×3.0 faster than Serial()","avgₕ! 1D: 7.75 ms, ×2.43 faster than Serial()","avgₕ! 2D: 31.61 ms, ×3.36 faster than Serial()","avgₕ! 3D: 174.53 ms, ×3.51 faster than Serial()","assemble (BilinearForm) 2D: 4.52 ms, ×1.08 faster than Serial()"], hoverinfo: 'text' }];
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
<div id="bench_chart_2" data-bench="precision" data-run="ba309208" style="width:100%; height:380px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ name: "Float32", x: ["Rₕ!","assemble!","avgₕ!","innerₕ"], y: [0.9734627501064282,0.8829339538568175,0.9559598369619247,0.5018108827507963], type: 'bar', marker: { color: '#3b82f6' }, hovertext: ["Rₕ!: 285.8 μs, ×0.97 the Float64 time","assemble!: 71.3 μs, ×0.88 the Float64 time","avgₕ!: 1.8 ms, ×0.96 the Float64 time","innerₕ: 11.5 μs, ×0.5 the Float64 time"], hoverinfo: 'text' },
{ name: "Double64", x: ["Rₕ!","assemble!","avgₕ!","innerₕ"], y: [30.24081055768412,13.279724477671056,44.266350664413295,48.684731858445694], type: 'bar', marker: { color: '#f59e0b' }, hovertext: ["Rₕ!: 8.88 ms, ×30.24 the Float64 time","assemble!: 1.07 ms, ×13.28 the Float64 time","avgₕ!: 83.49 ms, ×44.27 the Float64 time","innerₕ: 1.12 ms, ×48.68 the Float64 time"], hoverinfo: 'text' }];
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
<div id="bench_chart_3" data-bench="direction" data-run="ba309208" style="width:100%; height:278px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ name: "along x", x: [0.2035835,0.163,0.162541], y: ["D₋ₓ|ᵧ","M₊ₓ|ᵧ 2D","jumpₓ|ᵧ 2D"], type: 'bar', orientation: 'h', marker: { color: '#3b82f6' }, hovertext: ["D₋ₓ|ᵧ: 203.6 μs","M₊ₓ|ᵧ 2D: 163.0 μs","jumpₓ|ᵧ 2D: 162.5 μs"], hoverinfo: 'text' },
{ name: "along y", x: [0.161959,0.161625,0.161416], y: ["D₋ₓ|ᵧ","M₊ₓ|ᵧ 2D","jumpₓ|ᵧ 2D"], type: 'bar', orientation: 'h', marker: { color: '#f59e0b' }, hovertext: ["D₋ₓ|ᵧ: 162.0 μs","M₊ₓ|ᵧ 2D: 161.6 μs","jumpₓ|ᵧ 2D: 161.4 μs"], hoverinfo: 'text' }];
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
<div id="bench_chart_4" data-bench="assembly" data-run="ba309208" style="width:100%; height:324px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ name: "median time", x: [4.883291,0.303083,6.89975,0.377], y: ["first assembly (allocates and fills)","refill (pattern reused)","assemble the pieces, then add","assemble_add!"], type: 'bar', orientation: 'h', marker: { color: '#3b82f6' }, hovertext: ["first assembly (allocates and fills): 4.88 ms","refill (pattern reused): 303.1 μs","assemble the pieces, then add: 6.9 ms","assemble_add!: 377.0 μs"], hoverinfo: 'text' }];
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

## Regressions since the previous baseline

Each bar is one benchmark's median in the latest baseline (v3.20.0, `ba309208`) divided by its median in the one before (v3.13.0, `7c39cb79`); a bar to the left of the dotted line is faster. The spread of a run is its interquartile range, from the first to the third quartile of the samples of that one run. A change is flagged, in red when slower and green when faster, only when the two runs' interquartile ranges do not overlap; a grey bar is within the spread. No fixed percentage band is used, because separate launches of the same code can differ by 10 to 30%, which a fixed band would either hide on a quiet benchmark or flag on a loud one.

0 of 81 benchmarks are flagged. 81 have no spread recorded in one of the two runs (baselines saved before the quartiles were recorded keep only the minimum, median and maximum), so they are never flagged. Hover a bar for the two medians.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_5" data-bench="regression" data-flagged="" style="width:100%; height:1376px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{
    type: 'bar',
    orientation: 'h',
    y: ["operators 2D/Dcₓ","operators 2D/D₋(uₕ, d) over d","operators 2D/D₋ᵧ","operators 2D/D₋ₓ","operators 2D/Mₓ","operators 2D/curlₕ!","operators 2D/divₕ!","operators 2D/Δₕ","operators 2D/Δₕ!","operators 3D/D₋₂","operators 3D/innerₕ","operators 3D/∇ₕ","jumps & averages/M₊ᵧ 2D","jumps & averages/M₊₂ 3D","jumps & averages/M₊ₓ 2D","jumps & averages/jumpᵧ 2D","jumps & averages/jump₂ 3D","jumps & averages/jumpₓ 2D","jumps & averages/jumpₕ 2D","jumps & averages/jumpₕ 3D","inner products 2D/innerₕ","inner products 2D/norm₁ₕ","inner products 2D/normₕ","inner products 2D/snorm₁ₕ","restriction/Rₕ 1D (allocates its output)","restriction/Rₕ! 1D, Parallel() backend","restriction/Rₕ! 1D, Serial() backend (default)","restriction/Rₕ! 2D, Parallel() backend","restriction/Rₕ! 2D, Serial() backend (default)","restriction/Rₕ! 3D, Parallel() backend","restriction/Rₕ! 3D, Serial() backend (default)","restriction/avgₕ! 1D, Parallel() backend","restriction/avgₕ! 1D, Serial() backend (default)","restriction/avgₕ! 2D, Parallel() backend","restriction/avgₕ! 2D, Serial() backend (default)","restriction/avgₕ! 3D, Parallel() backend","restriction/avgₕ! 3D, Serial() backend (default)","composite/D₋ₓ (3 components)","composite/∇ₕ (3 components)","construction/gridspace 2D","construction/gridspace 3D","construction/hₘₐₓ 3D","startup & latency/TTFX (load + first operator)","startup & latency/TTFX first-assembly (assemble)","startup & latency/TTFX first-projection (Rₕ)","startup & latency/TTFX mesh construction","startup & latency/using Bramble","forms/allocate_system_matrix 2D","forms/assemble (BilinearForm) 2D, Parallel() backend","forms/assemble (BilinearForm) 2D, Serial() backend","forms/assemble! (matrix) 2D","forms/assemble! 1D","forms/assemble! 1D, Parallel() backend","forms/assemble! 2D","forms/assemble-then-add (matrix) 2D","forms/assemble_add! (matrix) 2D","forms/assemble_parallel! 1D","forms/assemble_parallel! 2D","forms/evaluate! 1D","forms/form (bilinear, 2D)","forms/l(vₕ) 1D","jacobian sparsity/jacobian (native), 1D n=100","jacobian sparsity/jacobian (native), 1D n=10000","jacobian sparsity/jacobian (traced), 1D n=100","jacobian sparsity/jacobian (traced), 1D n=10000","jacobian sparsity/prepare_jacobian (native), 1D n=100","jacobian sparsity/prepare_jacobian (native), 1D n=10000","jacobian sparsity/prepare_jacobian (traced), 1D n=100","jacobian sparsity/prepare_jacobian (traced), 1D n=10000","precision 1D/Rₕ! Double64","precision 1D/Rₕ! Float32","precision 1D/Rₕ! Float64","precision 1D/assemble! Double64","precision 1D/assemble! Float32","precision 1D/assemble! Float64","precision 1D/avgₕ! Double64","precision 1D/avgₕ! Float32","precision 1D/avgₕ! Float64","precision 1D/innerₕ Double64","precision 1D/innerₕ Float32","precision 1D/innerₕ Float64"],
    x: [0.6391193379177028,0.7032248493700931,0.6627478260869565,0.7304518332741318,0.7456781872707969,0.7049109558030524,0.6978935698447893,0.7424456218573923,0.729846635755635,0.7216739044611133,0.7528202193609919,0.7194210664565275,0.7122648369227514,0.8017926775474195,0.7630193095377413,0.8052360095381577,0.8023823712987027,0.7980723435804524,0.7794545992309967,0.7865567780909396,0.8401439557925487,0.7987089637221408,0.7999965673486201,0.8396456256921373,0.48470282920031604,0.5784345909543651,0.7508545864397721,0.5199395050564615,0.8399419056395325,0.4534277531092017,0.7141751032901317,0.4990814926447651,0.7730959251531335,0.5387414258167674,0.5757577588217111,0.5138531289515793,0.8276751722352916,0.7370095181248217,0.7485652131938858,1.0343993085566119,0.8926665487524472,0.8732464718380211,0.8037677286039071,0.8073445729987891,0.7930403894690944,0.7675433099944518,0.756523024952463,0.6738199349657785,0.6790265845599279,0.8073112615276368,0.35816565411145473,0.6332390173751058,0.47275153396013697,0.5819272940174579,0.5979266963838337,0.6476728444691832,0.4325062611806798,0.44862275799140144,0.46649292628443784,0.5533403931462184,0.8037023297896404,0.8254633245757296,0.8065608857037454,0.8337800011911143,0.8063998023227081,0.8154025898159868,0.8273002571296731,0.8490695187165775,0.8591620907057149,0.9332594177587721,0.8396579479225418,0.8347547974413646,0.8341267249757046,0.8421164470888228,0.7679263934301575,0.9092397666601516,0.9056417260235371,0.7964790962837838,0.9492026652691473,0.7976694180481376,0.7994209369658493],
    marker: { color: ["#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af","#9ca3af"] },
    hovertext: ["397.3 μs → 253.9 μs, ×0.639 (no spread recorded)","525.8 μs → 369.8 μs, ×0.703 (no spread recorded)","244.4 μs → 162.0 μs, ×0.663 (no spread recorded)","278.7 μs → 203.6 μs, ×0.73 (no spread recorded)","236.4 μs → 176.3 μs, ×0.746 (no spread recorded)","997.2 μs → 703.0 μs, ×0.705 (no spread recorded)","1.01 ms → 708.2 μs, ×0.698 (no spread recorded)","1.55 ms → 1.15 ms, ×0.742 (no spread recorded)","1.52 ms → 1.11 ms, ×0.73 (no spread recorded)","316.6 μs → 228.5 μs, ×0.722 (no spread recorded)","262.1 μs → 197.3 μs, ×0.753 (no spread recorded)","951.8 μs → 684.7 μs, ×0.719 (no spread recorded)","226.9 μs → 161.6 μs, ×0.712 (no spread recorded)","283.4 μs → 227.2 μs, ×0.802 (no spread recorded)","213.6 μs → 163.0 μs, ×0.763 (no spread recorded)","200.5 μs → 161.4 μs, ×0.805 (no spread recorded)","283.6 μs → 227.5 μs, ×0.802 (no spread recorded)","203.7 μs → 162.5 μs, ×0.798 (no spread recorded)","422.6 μs → 329.4 μs, ×0.779 (no spread recorded)","811.8 μs → 638.5 μs, ×0.787 (no spread recorded)","227.3 μs → 191.0 μs, ×0.84 (no spread recorded)","768.4 μs → 613.7 μs, ×0.799 (no spread recorded)","174.8 μs → 139.8 μs, ×0.8 (no spread recorded)","564.4 μs → 473.9 μs, ×0.84 (no spread recorded)","2.77 ms → 1.34 ms, ×0.485 (no spread recorded)","2.32 ms → 1.34 ms, ×0.578 (no spread recorded)","3.93 ms → 2.95 ms, ×0.751 (no spread recorded)","2.47 ms → 1.28 ms, ×0.52 (no spread recorded)","4.75 ms → 3.99 ms, ×0.84 (no spread recorded)","3.25 ms → 1.47 ms, ×0.453 (no spread recorded)","6.19 ms → 4.42 ms, ×0.714 (no spread recorded)","15.54 ms → 7.75 ms, ×0.499 (no spread recorded)","24.4 ms → 18.86 ms, ×0.773 (no spread recorded)","58.67 ms → 31.61 ms, ×0.539 (no spread recorded)","184.4 ms → 106.17 ms, ×0.576 (no spread recorded)","339.66 ms → 174.53 ms, ×0.514 (no spread recorded)","740.19 ms → 612.64 ms, ×0.828 (no spread recorded)","901.3 μs → 664.3 μs, ×0.737 (no spread recorded)","1.94 ms → 1.45 ms, ×0.749 (no spread recorded)","884.9 ns → 915.4 ns, ×1.034 (no spread recorded)","193.0 ns → 172.3 ns, ×0.893 (no spread recorded)","36.0 ns → 31.4 ns, ×0.873 (no spread recorded)","588.46 ms → 472.99 ms, ×0.804 (no spread recorded)","628.42 ms → 507.35 ms, ×0.807 (no spread recorded)","617.34 ms → 489.58 ms, ×0.793 (no spread recorded)","603.44 ms → 463.17 ms, ×0.768 (no spread recorded)","600.19 ms → 454.06 ms, ×0.757 (no spread recorded)","4.97 ms → 3.35 ms, ×0.674 (no spread recorded)","6.66 ms → 4.52 ms, ×0.679 (no spread recorded)","6.05 ms → 4.88 ms, ×0.807 (no spread recorded)","846.2 μs → 303.1 μs, ×0.358 (no spread recorded)","1.43 ms → 908.0 μs, ×0.633 (no spread recorded)","1.15 ms → 546.0 μs, ×0.473 (no spread recorded)","2.03 ms → 1.18 ms, ×0.582 (no spread recorded)","11.54 ms → 6.9 ms, ×0.598 (no spread recorded)","582.1 μs → 377.0 μs, ×0.648 (no spread recorded)","1.12 ms → 483.5 μs, ×0.433 (no spread recorded)","1.09 ms → 491.2 μs, ×0.449 (no spread recorded)","2.35 ms → 1.1 ms, ×0.466 (no spread recorded)","29.5 ns → 16.3 ns, ×0.553 (no spread recorded)","1.11 ms → 888.3 μs, ×0.804 (no spread recorded)","14.1 μs → 11.6 μs, ×0.825 (no spread recorded)","1.01 ms → 815.5 μs, ×0.807 (no spread recorded)","16.8 μs → 14.0 μs, ×0.834 (no spread recorded)","1.01 ms → 815.9 μs, ×0.806 (no spread recorded)","20.5 μs → 16.8 μs, ×0.815 (no spread recorded)","1.35 ms → 1.12 ms, ×0.827 (no spread recorded)","70.1 μs → 59.5 μs, ×0.849 (no spread recorded)","6.15 ms → 5.28 ms, ×0.859 (no spread recorded)","9.51 ms → 8.88 ms, ×0.933 (no spread recorded)","340.4 μs → 285.8 μs, ×0.84 (no spread recorded)","351.8 μs → 293.6 μs, ×0.835 (no spread recorded)","1.29 ms → 1.07 ms, ×0.834 (no spread recorded)","84.7 μs → 71.3 μs, ×0.842 (no spread recorded)","105.2 μs → 80.8 μs, ×0.768 (no spread recorded)","91.82 ms → 83.49 ms, ×0.909 (no spread recorded)","1.99 ms → 1.8 ms, ×0.906 (no spread recorded)","2.37 ms → 1.89 ms, ×0.796 (no spread recorded)","1.18 ms → 1.12 ms, ×0.949 (no spread recorded)","14.4 μs → 11.5 μs, ×0.798 (no spread recorded)","28.7 μs → 22.9 μs, ×0.799 (no spread recorded)"],
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

## Standalone benchmarks

The scripts in `benchmark/` that compare alternatives outside the regression suite save their tables to `benchmark/results/<script>.toml` with `--save`; every chart below is drawn from those files, and a new full run replaces the file. Each states the machine, thread count, power and commit of its run.

### Operator routes (`operator_routes.jl`)

#### Construction

Time to build the operator of the separable form `innerₕ(u,v) + inner₊(∇ₕu,∇ₕv)` on a non-uniform mesh, by route. Matrix-free construction takes microseconds or less because the operator's plan is built at construction and no matrix is formed; the assembled and Kronecker routes build their matrices here.

Run on Apple M2 with 4 threads, power: AC, commit `dfd9a2c2`, Julia 1.13.1, 2026-10-02T17:38:56. The script waited for a quiet machine before starting; the load average at the end of the run, which includes the run itself, was 8.12.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_6" data-bench="standalone-operator_routes-construction" data-run="dfd9a2c2" style="width:100%; height:420px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ type: 'scatter', mode: 'lines+markers', name: "assembled, 2D", x: [1024,4096,16384,65536,262144], y: [0.100833,0.5195420000000001,2.007833,7.187708,48.486042], line: { color: '#3b82f6', dash: 'solid' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 2D, n = 32 (1024 degrees of freedom): 100.8 μs, 628.953 KiB allocated","assembled, 2D, n = 64 (4096 degrees of freedom): 519.5 μs, 2.269 MiB allocated","assembled, 2D, n = 128 (16384 degrees of freedom): 2.01 ms, 8.831 MiB allocated","assembled, 2D, n = 256 (65536 degrees of freedom): 7.19 ms, 35.081 MiB allocated","assembled, 2D, n = 512 (262144 degrees of freedom): 48.49 ms, 154.472 MiB allocated"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "assembled, 3D", x: [512,4096,32768,110592,262144], y: [0.09191600000000001,0.891792,7.8694999999999995,31.243084,79.88666699999999], line: { color: '#3b82f6', dash: 'dash' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 3D, n = 8 (512 degrees of freedom): 91.9 μs, 386.969 KiB allocated","assembled, 3D, n = 16 (4096 degrees of freedom): 891.8 μs, 3.114 MiB allocated","assembled, 3D, n = 32 (32768 degrees of freedom): 7.87 ms, 24.395 MiB allocated","assembled, 3D, n = 48 (110592 degrees of freedom): 31.24 ms, 82.520 MiB allocated","assembled, 3D, n = 64 (262144 degrees of freedom): 79.89 ms, 202.317 MiB allocated"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 2D", x: [1024,4096,16384,65536,262144], y: [0.004625,0.009125,0.018040999999999998,0.030708,0.096125], line: { color: '#f59e0b', dash: 'solid' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 2D, n = 32 (1024 degrees of freedom): 4.6 μs, 26.078 KiB allocated","Kronecker, 2D, n = 64 (4096 degrees of freedom): 9.1 μs, 44.906 KiB allocated","Kronecker, 2D, n = 128 (16384 degrees of freedom): 18.0 μs, 83.156 KiB allocated","Kronecker, 2D, n = 256 (65536 degrees of freedom): 30.7 μs, 162.656 KiB allocated","Kronecker, 2D, n = 512 (262144 degrees of freedom): 96.1 μs, 318.656 KiB allocated"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 3D", x: [512,4096,32768,110592,262144], y: [0.004167,0.00625,0.012208,0.014334,0.0235], line: { color: '#f59e0b', dash: 'dash' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 3D, n = 8 (512 degrees of freedom): 4.2 μs, 16.516 KiB allocated","Kronecker, 3D, n = 16 (4096 degrees of freedom): 6.2 μs, 24.141 KiB allocated","Kronecker, 3D, n = 32 (32768 degrees of freedom): 12.2 μs, 39.016 KiB allocated","Kronecker, 3D, n = 48 (110592 degrees of freedom): 14.3 μs, 54.203 KiB allocated","Kronecker, 3D, n = 64 (262144 degrees of freedom): 23.5 μs, 67.000 KiB allocated"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 2D", x: [1024,4096,16384,65536,262144], y: [4.1e-5,4.1e-5,4.2e-5,4.1e-5,8.3e-5], line: { color: '#10b981', dash: 'solid' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 2D, n = 32 (1024 degrees of freedom): 41.0 ns, 720 bytes allocated","matrix-free, serial, 2D, n = 64 (4096 degrees of freedom): 41.0 ns, 720 bytes allocated","matrix-free, serial, 2D, n = 128 (16384 degrees of freedom): 42.0 ns, 720 bytes allocated","matrix-free, serial, 2D, n = 256 (65536 degrees of freedom): 41.0 ns, 720 bytes allocated","matrix-free, serial, 2D, n = 512 (262144 degrees of freedom): 83.0 ns, 720 bytes allocated"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 3D", x: [512,4096,32768,110592,262144], y: [8.4e-5,8.3e-5,0.000125,0.000125,0.000167], line: { color: '#10b981', dash: 'dash' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 3D, n = 8 (512 degrees of freedom): 84.0 ns, 1.156 KiB allocated","matrix-free, serial, 3D, n = 16 (4096 degrees of freedom): 83.0 ns, 1.156 KiB allocated","matrix-free, serial, 3D, n = 32 (32768 degrees of freedom): 125.0 ns, 1.156 KiB allocated","matrix-free, serial, 3D, n = 48 (110592 degrees of freedom): 125.0 ns, 1.156 KiB allocated","matrix-free, serial, 3D, n = 64 (262144 degrees of freedom): 167.0 ns, 1.156 KiB allocated"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 2D", x: [1024,4096,16384,65536,262144], y: [0.001708,0.001667,0.00175,0.001916,0.0026249999999999997], line: { color: '#ef4444', dash: 'solid' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 2D, n = 32 (1024 degrees of freedom): 1.7 μs, 2.672 KiB allocated","matrix-free, CpuThreaded(), 2D, n = 64 (4096 degrees of freedom): 1.7 μs, 2.672 KiB allocated","matrix-free, CpuThreaded(), 2D, n = 128 (16384 degrees of freedom): 1.8 μs, 2.672 KiB allocated","matrix-free, CpuThreaded(), 2D, n = 256 (65536 degrees of freedom): 1.9 μs, 2.672 KiB allocated","matrix-free, CpuThreaded(), 2D, n = 512 (262144 degrees of freedom): 2.6 μs, 2.672 KiB allocated"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 3D", x: [512,4096,32768,110592,262144], y: [0.002542,0.002792,0.004,0.003834,0.003541], line: { color: '#ef4444', dash: 'dash' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 3D, n = 8 (512 degrees of freedom): 2.5 μs, 4.156 KiB allocated","matrix-free, CpuThreaded(), 3D, n = 16 (4096 degrees of freedom): 2.8 μs, 4.156 KiB allocated","matrix-free, CpuThreaded(), 3D, n = 32 (32768 degrees of freedom): 4.0 μs, 4.156 KiB allocated","matrix-free, CpuThreaded(), 3D, n = 48 (110592 degrees of freedom): 3.8 μs, 4.156 KiB allocated","matrix-free, CpuThreaded(), 3D, n = 64 (262144 degrees of freedom): 3.5 μs, 4.156 KiB allocated"], hoverinfo: 'text' }];
  // The legend is anchored to the bottom of the chart; its rows depend on the width, so
  // the bottom margin and the chart's height grow with the rows it needs.
  const el = document.getElementById('bench_chart_6');
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
  Plotly.newPlot('bench_chart_6', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_6', function () {
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

Run on Apple M2 with 4 threads, power: AC, commit `dfd9a2c2`, Julia 1.13.1, 2026-10-02T17:38:56. The script waited for a quiet machine before starting; the load average at the end of the run, which includes the run itself, was 8.12.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_7" data-bench="standalone-operator_routes-product" data-run="dfd9a2c2" style="width:100%; height:420px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ type: 'scatter', mode: 'lines+markers', name: "assembled, 2D", x: [1024,4096,16384,65536,262144], y: [0.003542,0.015333000000000001,0.061,0.289375,1.0360829999999999], line: { color: '#3b82f6', dash: 'solid' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 2D, n = 32 (1024 degrees of freedom): 3.5 μs per product, 86.164 KiB held","assembled, 2D, n = 64 (4096 degrees of freedom): 15.3 μs per product, 348.164 KiB held","assembled, 2D, n = 128 (16384 degrees of freedom): 61.0 μs per product, 1.367 MiB held","assembled, 2D, n = 256 (65536 degrees of freedom): 289.4 μs per product, 5.485 MiB held","assembled, 2D, n = 512 (262144 degrees of freedom): 1.04 ms per product, 21.969 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "assembled, 3D", x: [512,4096,32768,110592,262144], y: [0.002208,0.021625000000000002,0.231083,0.786834,1.869791], line: { color: '#3b82f6', dash: 'dash' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 3D, n = 8 (512 degrees of freedom): 2.2 μs per product, 54.164 KiB held","assembled, 3D, n = 16 (4096 degrees of freedom): 21.6 μs per product, 456.164 KiB held","assembled, 3D, n = 32 (32768 degrees of freedom): 231.1 μs per product, 3.656 MiB held","assembled, 3D, n = 48 (110592 degrees of freedom): 786.8 μs per product, 12.445 MiB held","assembled, 3D, n = 64 (262144 degrees of freedom): 1.87 ms per product, 29.625 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 2D", x: [1024,4096,16384,65536,262144], y: [0.001458,0.005332999999999999,0.02125,0.090583,0.65525], line: { color: '#f59e0b', dash: 'solid' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 2D, n = 32 (1024 degrees of freedom): 1.5 μs per product, 5.016 KiB held","Kronecker, 2D, n = 64 (4096 degrees of freedom): 5.3 μs per product, 9.516 KiB held","Kronecker, 2D, n = 128 (16384 degrees of freedom): 21.2 μs per product, 18.516 KiB held","Kronecker, 2D, n = 256 (65536 degrees of freedom): 90.6 μs per product, 36.516 KiB held","Kronecker, 2D, n = 512 (262144 degrees of freedom): 655.2 μs per product, 72.516 KiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 3D", x: [512,4096,32768,110592,262144], y: [0.0022919999999999998,0.01475,0.13558299999999998,0.358166,1.036333], line: { color: '#f59e0b', dash: 'dash' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 3D, n = 8 (512 degrees of freedom): 2.3 μs per product, 2.398 KiB held","Kronecker, 3D, n = 16 (4096 degrees of freedom): 14.8 μs per product, 4.023 KiB held","Kronecker, 3D, n = 32 (32768 degrees of freedom): 135.6 μs per product, 7.273 KiB held","Kronecker, 3D, n = 48 (110592 degrees of freedom): 358.2 μs per product, 10.523 KiB held","Kronecker, 3D, n = 64 (262144 degrees of freedom): 1.04 ms per product, 13.773 KiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 2D", x: [1024,4096,16384,65536,262144], y: [0.006916,0.026042,0.106542,0.441334,2.6311669999999996], line: { color: '#10b981', dash: 'solid' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 2D, n = 32 (1024 degrees of freedom): 6.9 μs per product, 88 bytes held","matrix-free, serial, 2D, n = 64 (4096 degrees of freedom): 26.0 μs per product, 88 bytes held","matrix-free, serial, 2D, n = 128 (16384 degrees of freedom): 106.5 μs per product, 88 bytes held","matrix-free, serial, 2D, n = 256 (65536 degrees of freedom): 441.3 μs per product, 88 bytes held","matrix-free, serial, 2D, n = 512 (262144 degrees of freedom): 2.63 ms per product, 88 bytes held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 3D", x: [512,4096,32768,110592,262144], y: [0.013,0.095291,0.9324169999999999,2.7026250000000003,9.66375], line: { color: '#10b981', dash: 'dash' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 3D, n = 8 (512 degrees of freedom): 13.0 μs per product, 88 bytes held","matrix-free, serial, 3D, n = 16 (4096 degrees of freedom): 95.3 μs per product, 88 bytes held","matrix-free, serial, 3D, n = 32 (32768 degrees of freedom): 932.4 μs per product, 88 bytes held","matrix-free, serial, 3D, n = 48 (110592 degrees of freedom): 2.7 ms per product, 88 bytes held","matrix-free, serial, 3D, n = 64 (262144 degrees of freedom): 9.66 ms per product, 88 bytes held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 2D", x: [1024,4096,16384,65536,262144], y: [0.010208,0.011167,0.078625,0.27999999999999997,1.142292], line: { color: '#ef4444', dash: 'solid' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 2D, n = 32 (1024 degrees of freedom): 10.2 μs per product, 128 bytes held","matrix-free, CpuThreaded(), 2D, n = 64 (4096 degrees of freedom): 11.2 μs per product, 128 bytes held","matrix-free, CpuThreaded(), 2D, n = 128 (16384 degrees of freedom): 78.6 μs per product, 128 bytes held","matrix-free, CpuThreaded(), 2D, n = 256 (65536 degrees of freedom): 280.0 μs per product, 128 bytes held","matrix-free, CpuThreaded(), 2D, n = 512 (262144 degrees of freedom): 1.14 ms per product, 128 bytes held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 3D", x: [512,4096,32768,110592,262144], y: [0.006083,0.029750000000000002,0.398708,1.092625,2.592292], line: { color: '#ef4444', dash: 'dash' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 3D, n = 8 (512 degrees of freedom): 6.1 μs per product, 136 bytes held","matrix-free, CpuThreaded(), 3D, n = 16 (4096 degrees of freedom): 29.8 μs per product, 136 bytes held","matrix-free, CpuThreaded(), 3D, n = 32 (32768 degrees of freedom): 398.7 μs per product, 136 bytes held","matrix-free, CpuThreaded(), 3D, n = 48 (110592 degrees of freedom): 1.09 ms per product, 136 bytes held","matrix-free, CpuThreaded(), 3D, n = 64 (262144 degrees of freedom): 2.59 ms per product, 136 bytes held"], hoverinfo: 'text' }];
  // The legend is anchored to the bottom of the chart; its rows depend on the width, so
  // the bottom margin and the chart's height grow with the rows it needs.
  const el = document.getElementById('bench_chart_7');
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
  Plotly.newPlot('bench_chart_7', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_7', function () {
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
<div id="bench_chart_8" data-bench="standalone-operator_routes-product-bytes" data-run="dfd9a2c2" style="width:100%; height:420px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ type: 'scatter', mode: 'lines+markers', name: "assembled, 2D", x: [1024,4096,16384,65536,262144], y: [88232,356520,1433768,5750952,23036072], line: { color: '#3b82f6', dash: 'solid' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 2D, n = 32 (1024 degrees of freedom): 86.164 KiB held","assembled, 2D, n = 64 (4096 degrees of freedom): 348.164 KiB held","assembled, 2D, n = 128 (16384 degrees of freedom): 1.367 MiB held","assembled, 2D, n = 256 (65536 degrees of freedom): 5.485 MiB held","assembled, 2D, n = 512 (262144 degrees of freedom): 21.969 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "assembled, 3D", x: [512,4096,32768,110592,262144], y: [55464,467112,3834024,13050024,31064232], line: { color: '#3b82f6', dash: 'dash' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 3D, n = 8 (512 degrees of freedom): 54.164 KiB held","assembled, 3D, n = 16 (4096 degrees of freedom): 456.164 KiB held","assembled, 3D, n = 32 (32768 degrees of freedom): 3.656 MiB held","assembled, 3D, n = 48 (110592 degrees of freedom): 12.445 MiB held","assembled, 3D, n = 64 (262144 degrees of freedom): 29.625 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 2D", x: [1024,4096,16384,65536,262144], y: [5136,9744,18960,37392,74256], line: { color: '#f59e0b', dash: 'solid' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 2D, n = 32 (1024 degrees of freedom): 5.016 KiB held","Kronecker, 2D, n = 64 (4096 degrees of freedom): 9.516 KiB held","Kronecker, 2D, n = 128 (16384 degrees of freedom): 18.516 KiB held","Kronecker, 2D, n = 256 (65536 degrees of freedom): 36.516 KiB held","Kronecker, 2D, n = 512 (262144 degrees of freedom): 72.516 KiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 3D", x: [512,4096,32768,110592,262144], y: [2456,4120,7448,10776,14104], line: { color: '#f59e0b', dash: 'dash' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 3D, n = 8 (512 degrees of freedom): 2.398 KiB held","Kronecker, 3D, n = 16 (4096 degrees of freedom): 4.023 KiB held","Kronecker, 3D, n = 32 (32768 degrees of freedom): 7.273 KiB held","Kronecker, 3D, n = 48 (110592 degrees of freedom): 10.523 KiB held","Kronecker, 3D, n = 64 (262144 degrees of freedom): 13.773 KiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 2D", x: [1024,4096,16384,65536,262144], y: [88,88,88,88,88], line: { color: '#10b981', dash: 'solid' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 2D, n = 32 (1024 degrees of freedom): 88 bytes held","matrix-free, serial, 2D, n = 64 (4096 degrees of freedom): 88 bytes held","matrix-free, serial, 2D, n = 128 (16384 degrees of freedom): 88 bytes held","matrix-free, serial, 2D, n = 256 (65536 degrees of freedom): 88 bytes held","matrix-free, serial, 2D, n = 512 (262144 degrees of freedom): 88 bytes held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 3D", x: [512,4096,32768,110592,262144], y: [88,88,88,88,88], line: { color: '#10b981', dash: 'dash' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 3D, n = 8 (512 degrees of freedom): 88 bytes held","matrix-free, serial, 3D, n = 16 (4096 degrees of freedom): 88 bytes held","matrix-free, serial, 3D, n = 32 (32768 degrees of freedom): 88 bytes held","matrix-free, serial, 3D, n = 48 (110592 degrees of freedom): 88 bytes held","matrix-free, serial, 3D, n = 64 (262144 degrees of freedom): 88 bytes held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 2D", x: [1024,4096,16384,65536,262144], y: [128,128,128,128,128], line: { color: '#ef4444', dash: 'solid' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 2D, n = 32 (1024 degrees of freedom): 128 bytes held","matrix-free, CpuThreaded(), 2D, n = 64 (4096 degrees of freedom): 128 bytes held","matrix-free, CpuThreaded(), 2D, n = 128 (16384 degrees of freedom): 128 bytes held","matrix-free, CpuThreaded(), 2D, n = 256 (65536 degrees of freedom): 128 bytes held","matrix-free, CpuThreaded(), 2D, n = 512 (262144 degrees of freedom): 128 bytes held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 3D", x: [512,4096,32768,110592,262144], y: [136,136,136,136,136], line: { color: '#ef4444', dash: 'dash' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 3D, n = 8 (512 degrees of freedom): 136 bytes held","matrix-free, CpuThreaded(), 3D, n = 16 (4096 degrees of freedom): 136 bytes held","matrix-free, CpuThreaded(), 3D, n = 32 (32768 degrees of freedom): 136 bytes held","matrix-free, CpuThreaded(), 3D, n = 48 (110592 degrees of freedom): 136 bytes held","matrix-free, CpuThreaded(), 3D, n = 64 (262144 degrees of freedom): 136 bytes held"], hoverinfo: 'text' }];
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
      title: { text: "bytes held by the operator", font: { color: theme.text } },
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

#### Solve

Time of the solve of the same problem by conjugate gradients on each route, with `fdm_solve` and the sparse direct solve as reference curves. The CG iteration counts agree across routes to within 0.36% at every size. CG stops when its recursively updated residual reaches the tolerance, which is not the true residual: the largest true relative residual among the CG rows is 7.2e-8, at 2D with n = 512.

Run on Apple M2 with 4 threads, power: AC, commit `dfd9a2c2`, Julia 1.13.1, 2026-10-02T17:38:56. The script waited for a quiet machine before starting; the load average at the end of the run, which includes the run itself, was 8.12.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_9" data-bench="standalone-operator_routes-solve" data-run="dfd9a2c2" style="width:100%; height:420px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ type: 'scatter', mode: 'lines+markers', name: "assembled, 2D", x: [1024,4096,16384,65536,262144], y: [1.904667,25.812292,265.56383300000005,3532.9831249999997,69730.788833], line: { color: '#3b82f6', dash: 'solid' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 2D, n = 32 (1024 degrees of freedom): 1.9 ms, 431 CG iterations, true relative residual 8.24e-9","assembled, 2D, n = 64 (4096 degrees of freedom): 25.81 ms, 1251 CG iterations, true relative residual 9.75e-9","assembled, 2D, n = 128 (16384 degrees of freedom): 265.56 ms, 3604 CG iterations, true relative residual 9.95e-9","assembled, 2D, n = 256 (65536 degrees of freedom): 3.53 s, 10646 CG iterations, true relative residual 1.21e-8","assembled, 2D, n = 512 (262144 degrees of freedom): 69.73 s, 30328 CG iterations, true relative residual 7.19e-8"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "assembled, 3D", x: [512,4096,32768,110592,262144], y: [0.400208,10.250208,315.664208,1688.3037080000001,7176.103792], line: { color: '#3b82f6', dash: 'dash' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled, 3D, n = 8 (512 degrees of freedom): 400.2 μs, 151 CG iterations, true relative residual 9.32e-9","assembled, 3D, n = 16 (4096 degrees of freedom): 10.25 ms, 380 CG iterations, true relative residual 9.51e-9","assembled, 3D, n = 32 (32768 degrees of freedom): 315.66 ms, 993 CG iterations, true relative residual 9.77e-9","assembled, 3D, n = 48 (110592 degrees of freedom): 1.69 s, 1796 CG iterations, true relative residual 9.99e-9","assembled, 3D, n = 64 (262144 degrees of freedom): 7.18 s, 2753 CG iterations, true relative residual 1.0e-8"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 2D", x: [1024,4096,16384,65536,262144], y: [0.9276249999999999,11.402334,124.123375,1423.980875,25342.180583], line: { color: '#f59e0b', dash: 'solid' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 2D, n = 32 (1024 degrees of freedom): 927.6 μs, 431 CG iterations, true relative residual 8.87e-9","Kronecker, 2D, n = 64 (4096 degrees of freedom): 11.4 ms, 1253 CG iterations, true relative residual 9.31e-9","Kronecker, 2D, n = 128 (16384 degrees of freedom): 124.12 ms, 3614 CG iterations, true relative residual 1.0e-8","Kronecker, 2D, n = 256 (65536 degrees of freedom): 1.42 s, 10634 CG iterations, true relative residual 1.24e-8","Kronecker, 2D, n = 512 (262144 degrees of freedom): 25.34 s, 30335 CG iterations, true relative residual 7.05e-8"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "Kronecker, 3D", x: [512,4096,32768,110592,262144], y: [0.412875,7.3303329999999995,190.256958,865.873292,3537.018375], line: { color: '#f59e0b', dash: 'dash' }, marker: { color: '#f59e0b', size: 6 }, hovertext: ["Kronecker, 3D, n = 8 (512 degrees of freedom): 412.9 μs, 151 CG iterations, true relative residual 9.2e-9","Kronecker, 3D, n = 16 (4096 degrees of freedom): 7.33 ms, 380 CG iterations, true relative residual 9.9e-9","Kronecker, 3D, n = 32 (32768 degrees of freedom): 190.26 ms, 993 CG iterations, true relative residual 9.89e-9","Kronecker, 3D, n = 48 (110592 degrees of freedom): 865.87 ms, 1795 CG iterations, true relative residual 9.97e-9","Kronecker, 3D, n = 64 (262144 degrees of freedom): 3.54 s, 2752 CG iterations, true relative residual 1.0e-8"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 2D", x: [1024,4096,16384,65536,262144], y: [3.278208,36.475333,430.608792,5209.95875,73385.091917], line: { color: '#10b981', dash: 'solid' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 2D, n = 32 (1024 degrees of freedom): 3.28 ms, 432 CG iterations, true relative residual 9.39e-9","matrix-free, serial, 2D, n = 64 (4096 degrees of freedom): 36.48 ms, 1253 CG iterations, true relative residual 9.09e-9","matrix-free, serial, 2D, n = 128 (16384 degrees of freedom): 430.61 ms, 3601 CG iterations, true relative residual 9.73e-9","matrix-free, serial, 2D, n = 256 (65536 degrees of freedom): 5.21 s, 10655 CG iterations, true relative residual 1.23e-8","matrix-free, serial, 2D, n = 512 (262144 degrees of freedom): 73.39 s, 30330 CG iterations, true relative residual 7.05e-8"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 3D", x: [512,4096,32768,110592,262144], y: [2.044709,37.710125,1002.9945419999999,5179.05775,17975.487166], line: { color: '#10b981', dash: 'dash' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 3D, n = 8 (512 degrees of freedom): 2.04 ms, 151 CG iterations, true relative residual 8.87e-9","matrix-free, serial, 3D, n = 16 (4096 degrees of freedom): 37.71 ms, 380 CG iterations, true relative residual 9.88e-9","matrix-free, serial, 3D, n = 32 (32768 degrees of freedom): 1.0 s, 993 CG iterations, true relative residual 1.0e-8","matrix-free, serial, 3D, n = 48 (110592 degrees of freedom): 5.18 s, 1795 CG iterations, true relative residual 9.94e-9","matrix-free, serial, 3D, n = 64 (262144 degrees of freedom): 17.98 s, 2753 CG iterations, true relative residual 9.9e-9"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 2D", x: [1024,4096,16384,65536,262144], y: [6.614375,49.382417,436.544708,4313.187208,54267.684208], line: { color: '#ef4444', dash: 'solid' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 2D, n = 32 (1024 degrees of freedom): 6.61 ms, 432 CG iterations, true relative residual 9.39e-9","matrix-free, CpuThreaded(), 2D, n = 64 (4096 degrees of freedom): 49.38 ms, 1253 CG iterations, true relative residual 9.09e-9","matrix-free, CpuThreaded(), 2D, n = 128 (16384 degrees of freedom): 436.54 ms, 3601 CG iterations, true relative residual 9.73e-9","matrix-free, CpuThreaded(), 2D, n = 256 (65536 degrees of freedom): 4.31 s, 10655 CG iterations, true relative residual 1.23e-8","matrix-free, CpuThreaded(), 2D, n = 512 (262144 degrees of freedom): 54.27 s, 30330 CG iterations, true relative residual 7.05e-8"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 3D", x: [512,4096,32768,110592,262144], y: [2.509167,28.429666,478.82225,2646.445167,10033.744666], line: { color: '#ef4444', dash: 'dash' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 3D, n = 8 (512 degrees of freedom): 2.51 ms, 151 CG iterations, true relative residual 8.87e-9","matrix-free, CpuThreaded(), 3D, n = 16 (4096 degrees of freedom): 28.43 ms, 380 CG iterations, true relative residual 9.88e-9","matrix-free, CpuThreaded(), 3D, n = 32 (32768 degrees of freedom): 478.82 ms, 993 CG iterations, true relative residual 1.0e-8","matrix-free, CpuThreaded(), 3D, n = 48 (110592 degrees of freedom): 2.65 s, 1795 CG iterations, true relative residual 9.94e-9","matrix-free, CpuThreaded(), 3D, n = 64 (262144 degrees of freedom): 10.03 s, 2753 CG iterations, true relative residual 9.9e-9"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "fdm_solve (reference), 2D", x: [1024,4096,16384,65536,262144], y: [0.187833,0.796083,3.017166,16.691791000000002,146.319875], line: { color: '#8b5cf6', dash: 'solid' }, marker: { color: '#8b5cf6', size: 6 }, hovertext: ["fdm_solve (reference), 2D, n = 32 (1024 degrees of freedom): 187.8 μs, 0 CG iterations, true relative residual 3.51e-12","fdm_solve (reference), 2D, n = 64 (4096 degrees of freedom): 796.1 μs, 0 CG iterations, true relative residual 2.79e-11","fdm_solve (reference), 2D, n = 128 (16384 degrees of freedom): 3.02 ms, 0 CG iterations, true relative residual 1.81e-10","fdm_solve (reference), 2D, n = 256 (65536 degrees of freedom): 16.69 ms, 0 CG iterations, true relative residual 1.48e-9","fdm_solve (reference), 2D, n = 512 (262144 degrees of freedom): 146.32 ms, 0 CG iterations, true relative residual 1.57e-8"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "fdm_solve (reference), 3D", x: [512,4096,32768,110592,262144], y: [0.176708,0.866375,4.3974590000000005,10.509375,25.399541000000003], line: { color: '#8b5cf6', dash: 'dash' }, marker: { color: '#8b5cf6', size: 6 }, hovertext: ["fdm_solve (reference), 3D, n = 8 (512 degrees of freedom): 176.7 μs, 0 CG iterations, true relative residual 2.08e-13","fdm_solve (reference), 3D, n = 16 (4096 degrees of freedom): 866.4 μs, 0 CG iterations, true relative residual 5.13e-12","fdm_solve (reference), 3D, n = 32 (32768 degrees of freedom): 4.4 ms, 0 CG iterations, true relative residual 6.31e-12","fdm_solve (reference), 3D, n = 48 (110592 degrees of freedom): 10.51 ms, 0 CG iterations, true relative residual 2.68e-11","fdm_solve (reference), 3D, n = 64 (262144 degrees of freedom): 25.4 ms, 0 CG iterations, true relative residual 5.45e-11"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "direct (reference), 2D", x: [1024,4096,16384,65536,262144], y: [0.525375,2.594583,9.373083000000001,46.657958,318.38624999999996], line: { color: '#9ca3af', dash: 'solid' }, marker: { color: '#9ca3af', size: 6 }, hovertext: ["direct (reference), 2D, n = 32 (1024 degrees of freedom): 525.4 μs, 0 CG iterations, true relative residual 1.26e-12","direct (reference), 2D, n = 64 (4096 degrees of freedom): 2.59 ms, 0 CG iterations, true relative residual 7.3e-12","direct (reference), 2D, n = 128 (16384 degrees of freedom): 9.37 ms, 0 CG iterations, true relative residual 2.92e-11","direct (reference), 2D, n = 256 (65536 degrees of freedom): 46.66 ms, 0 CG iterations, true relative residual 1.6e-10","direct (reference), 2D, n = 512 (262144 degrees of freedom): 318.39 ms, 0 CG iterations, true relative residual 8.96e-10"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "direct (reference), 3D", x: [512,4096,32768,110592,262144], y: [0.5721670000000001,8.1815,212.60075,2104.413375,13003.549083], line: { color: '#9ca3af', dash: 'dash' }, marker: { color: '#9ca3af', size: 6 }, hovertext: ["direct (reference), 3D, n = 8 (512 degrees of freedom): 572.2 μs, 0 CG iterations, true relative residual 6.55e-14","direct (reference), 3D, n = 16 (4096 degrees of freedom): 8.18 ms, 0 CG iterations, true relative residual 3.54e-13","direct (reference), 3D, n = 32 (32768 degrees of freedom): 212.6 ms, 0 CG iterations, true relative residual 1.83e-12","direct (reference), 3D, n = 48 (110592 degrees of freedom): 2.1 s, 0 CG iterations, true relative residual 4.87e-12","direct (reference), 3D, n = 64 (262144 degrees of freedom): 13.0 s, 0 CG iterations, true relative residual 1.05e-11"], hoverinfo: 'text' }];
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
      title: { text: "solve time (ms)", font: { color: theme.text } },
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

### Matrix-free against SpMV (`matrix_free_spmv.jl`)

#### Time

One product `mul!(y, matrix_free_operator(a), x)` against one serial sparse matrix-vector product with the assembled matrix, on non-uniform meshes in 1D, 2D and 3D, by degrees of freedom.

Run on Apple M2 with 4 threads, power: AC, commit `6e77a7be`, Julia 1.13.1, 2026-10-02T17:40:50. The script waited for a quiet machine before starting; the load average at the end of the run, which includes the run itself, was 4.02.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_10" data-bench="standalone-matrix_free_spmv-time" data-run="6e77a7be" style="width:100%; height:420px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 1D", x: [1000,10000,100000,1000000,10000000], y: [0.004667,0.048374999999999994,0.48258300000000004,4.9050839999999996,51.580875], line: { color: '#10b981', dash: 'solid' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 1D, 1000 degrees of freedom: 4.7 μs, ×0.54 the SpMV speed","matrix-free, serial, 1D, 10000 degrees of freedom: 48.4 μs, ×0.56 the SpMV speed","matrix-free, serial, 1D, 100000 degrees of freedom: 482.6 μs, ×0.63 the SpMV speed","matrix-free, serial, 1D, 1000000 degrees of freedom: 4.91 ms, ×0.58 the SpMV speed","matrix-free, serial, 1D, 10000000 degrees of freedom: 51.58 ms, ×0.56 the SpMV speed"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "SpMV, 1D", x: [1000,10000,100000,1000000,10000000], y: [0.002542,0.027291,0.30574999999999997,2.836583,28.739041999999998], line: { color: '#3b82f6', dash: 'solid' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["SpMV, 1D, 1000 degrees of freedom: 2.5 μs","SpMV, 1D, 10000 degrees of freedom: 27.3 μs","SpMV, 1D, 100000 degrees of freedom: 305.8 μs","SpMV, 1D, 1000000 degrees of freedom: 2.84 ms","SpMV, 1D, 10000000 degrees of freedom: 28.74 ms"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 1D", x: [1000,10000,100000,1000000,10000000], y: [0.006291000000000001,0.011875,0.097459,1.209667,13.677875], line: { color: '#ef4444', dash: 'solid' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 1D, 1000 degrees of freedom: 6.3 μs, ×0.4 the SpMV speed","matrix-free, CpuThreaded(), 1D, 10000 degrees of freedom: 11.9 μs, ×2.3 the SpMV speed","matrix-free, CpuThreaded(), 1D, 100000 degrees of freedom: 97.5 μs, ×3.14 the SpMV speed","matrix-free, CpuThreaded(), 1D, 1000000 degrees of freedom: 1.21 ms, ×2.34 the SpMV speed","matrix-free, CpuThreaded(), 1D, 10000000 degrees of freedom: 13.68 ms, ×2.1 the SpMV speed"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 2D", x: [1024,4096,16384,65536,262144,1048576,4194304], y: [0.007958000000000002,0.030875,0.126542,0.48895799999999995,1.9514999999999998,7.940416000000001,32.710125], line: { color: '#10b981', dash: 'dash' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 2D, 1024 degrees of freedom: 8.0 μs, ×0.47 the SpMV speed","matrix-free, serial, 2D, 4096 degrees of freedom: 30.9 μs, ×0.47 the SpMV speed","matrix-free, serial, 2D, 16384 degrees of freedom: 126.5 μs, ×0.46 the SpMV speed","matrix-free, serial, 2D, 65536 degrees of freedom: 489.0 μs, ×0.55 the SpMV speed","matrix-free, serial, 2D, 262144 degrees of freedom: 1.95 ms, ×0.5 the SpMV speed","matrix-free, serial, 2D, 1048576 degrees of freedom: 7.94 ms, ×0.51 the SpMV speed","matrix-free, serial, 2D, 4194304 degrees of freedom: 32.71 ms, ×0.49 the SpMV speed"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "SpMV, 2D", x: [1024,4096,16384,65536,262144,1048576,4194304], y: [0.003708,0.0145,0.05775,0.2705,0.9767910000000001,4.040417,16.054916], line: { color: '#3b82f6', dash: 'dash' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["SpMV, 2D, 1024 degrees of freedom: 3.7 μs","SpMV, 2D, 4096 degrees of freedom: 14.5 μs","SpMV, 2D, 16384 degrees of freedom: 57.8 μs","SpMV, 2D, 65536 degrees of freedom: 270.5 μs","SpMV, 2D, 262144 degrees of freedom: 976.8 μs","SpMV, 2D, 1048576 degrees of freedom: 4.04 ms","SpMV, 2D, 4194304 degrees of freedom: 16.05 ms"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 2D", x: [1024,4096,16384,65536,262144,1048576,4194304], y: [0.011375,0.011291,0.039,0.150333,0.583125,2.353709,10.059958], line: { color: '#ef4444', dash: 'dash' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 2D, 1024 degrees of freedom: 11.4 μs, ×0.33 the SpMV speed","matrix-free, CpuThreaded(), 2D, 4096 degrees of freedom: 11.3 μs, ×1.28 the SpMV speed","matrix-free, CpuThreaded(), 2D, 16384 degrees of freedom: 39.0 μs, ×1.48 the SpMV speed","matrix-free, CpuThreaded(), 2D, 65536 degrees of freedom: 150.3 μs, ×1.8 the SpMV speed","matrix-free, CpuThreaded(), 2D, 262144 degrees of freedom: 583.1 μs, ×1.68 the SpMV speed","matrix-free, CpuThreaded(), 2D, 1048576 degrees of freedom: 2.35 ms, ×1.72 the SpMV speed","matrix-free, CpuThreaded(), 2D, 4194304 degrees of freedom: 10.06 ms, ×1.6 the SpMV speed"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 3D", x: [4096,32768,262144,884736,2097152], y: [0.06145799999999999,0.417458,3.059291,10.349917,31.731457999999996], line: { color: '#10b981', dash: 'dot' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 3D, 4096 degrees of freedom: 61.5 μs, ×0.3 the SpMV speed","matrix-free, serial, 3D, 32768 degrees of freedom: 417.5 μs, ×0.36 the SpMV speed","matrix-free, serial, 3D, 262144 degrees of freedom: 3.06 ms, ×0.4 the SpMV speed","matrix-free, serial, 3D, 884736 degrees of freedom: 10.35 ms, ×0.4 the SpMV speed","matrix-free, serial, 3D, 2097152 degrees of freedom: 31.73 ms, ×0.33 the SpMV speed"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "SpMV, 3D", x: [4096,32768,262144,884736,2097152], y: [0.018167,0.150084,1.221125,4.089375,10.532209], line: { color: '#3b82f6', dash: 'dot' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["SpMV, 3D, 4096 degrees of freedom: 18.2 μs","SpMV, 3D, 32768 degrees of freedom: 150.1 μs","SpMV, 3D, 262144 degrees of freedom: 1.22 ms","SpMV, 3D, 884736 degrees of freedom: 4.09 ms","SpMV, 3D, 2097152 degrees of freedom: 10.53 ms"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 3D", x: [4096,32768,262144,884736,2097152], y: [0.024916999999999998,0.15437499999999998,1.006708,3.29175,10.083917], line: { color: '#ef4444', dash: 'dot' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 3D, 4096 degrees of freedom: 24.9 μs, ×0.73 the SpMV speed","matrix-free, CpuThreaded(), 3D, 32768 degrees of freedom: 154.4 μs, ×0.97 the SpMV speed","matrix-free, CpuThreaded(), 3D, 262144 degrees of freedom: 1.01 ms, ×1.21 the SpMV speed","matrix-free, CpuThreaded(), 3D, 884736 degrees of freedom: 3.29 ms, ×1.24 the SpMV speed","matrix-free, CpuThreaded(), 3D, 2097152 degrees of freedom: 10.08 ms, ×1.04 the SpMV speed"], hoverinfo: 'text' }];
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
      title: { text: "time of one product (ms)", font: { color: theme.text } },
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

#### Memory

Bytes the assembled matrix holds against the bytes everything the matrix-free operator keeps alive holds, including the form it captures.

The smallest size from which matrix-free beats the SpMV at every larger size tested:

- 1D, serial: no size in the range
- 1D, threaded: from 10000 degrees of freedom, where the assembled matrix holds 1.0 times the bytes of the matrix-free operator
- 2D, serial: no size in the range
- 2D, threaded: from 4096 degrees of freedom, where the assembled matrix holds 8.4 times the bytes of the matrix-free operator
- 3D, serial: no size in the range
- 3D, threaded: from 262144 degrees of freedom, where the assembled matrix holds 14.3 times the bytes of the matrix-free operator

Run on Apple M2 with 4 threads, power: AC, commit `6e77a7be`, Julia 1.13.1, 2026-10-02T17:40:50. The script waited for a quiet machine before starting; the load average at the end of the run, which includes the run itself, was 4.02.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_11" data-bench="standalone-matrix_free_spmv-memory" data-run="6e77a7be" style="width:100%; height:420px;"></div>
<script>
(function () {
  const theme = window.bramblePlotlyTheme();
  const data = [{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 1D", x: [1000,10000,100000,1000000,10000000], y: [57592,563848,5626344,56251336,562501336], line: { color: '#10b981', dash: 'solid' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 1D, 1000 degrees of freedom: 56.242 KiB held","matrix-free, serial, 1D, 10000 degrees of freedom: 550.633 KiB held","matrix-free, serial, 1D, 100000 degrees of freedom: 5.366 MiB held","matrix-free, serial, 1D, 1000000 degrees of freedom: 53.645 MiB held","matrix-free, serial, 1D, 10000000 degrees of freedom: 536.443 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "assembled matrix, 1D", x: [1000,10000,100000,1000000,10000000], y: [56136,560136,5600136,56000136,560000136], line: { color: '#3b82f6', dash: 'solid' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled matrix, 1D, 1000 degrees of freedom: 54.820 KiB held","assembled matrix, 1D, 10000 degrees of freedom: 547.008 KiB held","assembled matrix, 1D, 100000 degrees of freedom: 5.341 MiB held","assembled matrix, 1D, 1000000 degrees of freedom: 53.406 MiB held","assembled matrix, 1D, 10000000 degrees of freedom: 534.058 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 1D", x: [1000,10000,100000,1000000,10000000], y: [57624,563880,5626376,56251368,562501368], line: { color: '#ef4444', dash: 'solid' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 1D, 1000 degrees of freedom: 56.273 KiB held","matrix-free, CpuThreaded(), 1D, 10000 degrees of freedom: 550.664 KiB held","matrix-free, CpuThreaded(), 1D, 100000 degrees of freedom: 5.366 MiB held","matrix-free, CpuThreaded(), 1D, 1000000 degrees of freedom: 53.645 MiB held","matrix-free, CpuThreaded(), 1D, 10000000 degrees of freedom: 536.443 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 2D", x: [1024,4096,16384,65536,262144,1048576,4194304], y: [14376,42280,148808,564616,2207240,8736520,34771208], line: { color: '#10b981', dash: 'dash' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 2D, 1024 degrees of freedom: 14.039 KiB held","matrix-free, serial, 2D, 4096 degrees of freedom: 41.289 KiB held","matrix-free, serial, 2D, 16384 degrees of freedom: 145.320 KiB held","matrix-free, serial, 2D, 65536 degrees of freedom: 551.383 KiB held","matrix-free, serial, 2D, 262144 degrees of freedom: 2.105 MiB held","matrix-free, serial, 2D, 1048576 degrees of freedom: 8.332 MiB held","matrix-free, serial, 2D, 4194304 degrees of freedom: 33.160 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "assembled matrix, 2D", x: [1024,4096,16384,65536,262144,1048576,4194304], y: [88232,356520,1433768,5750952,23036072,92209320,368967848], line: { color: '#3b82f6', dash: 'dash' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled matrix, 2D, 1024 degrees of freedom: 86.164 KiB held","assembled matrix, 2D, 4096 degrees of freedom: 348.164 KiB held","assembled matrix, 2D, 16384 degrees of freedom: 1.367 MiB held","assembled matrix, 2D, 65536 degrees of freedom: 5.485 MiB held","assembled matrix, 2D, 262144 degrees of freedom: 21.969 MiB held","assembled matrix, 2D, 1048576 degrees of freedom: 87.938 MiB held","assembled matrix, 2D, 4194304 degrees of freedom: 351.875 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 2D", x: [1024,4096,16384,65536,262144,1048576,4194304], y: [14416,42320,148848,564656,2207280,8736560,34771248], line: { color: '#ef4444', dash: 'dash' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 2D, 1024 degrees of freedom: 14.078 KiB held","matrix-free, CpuThreaded(), 2D, 4096 degrees of freedom: 41.328 KiB held","matrix-free, CpuThreaded(), 2D, 16384 degrees of freedom: 145.359 KiB held","matrix-free, CpuThreaded(), 2D, 65536 degrees of freedom: 551.422 KiB held","matrix-free, CpuThreaded(), 2D, 262144 degrees of freedom: 2.105 MiB held","matrix-free, CpuThreaded(), 2D, 1048576 degrees of freedom: 8.332 MiB held","matrix-free, CpuThreaded(), 2D, 4194304 degrees of freedom: 33.160 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, serial, 3D", x: [4096,32768,262144,884736,2097152], y: [41152,279616,2175808,7316080,17322352], line: { color: '#10b981', dash: 'dot' }, marker: { color: '#10b981', size: 6 }, hovertext: ["matrix-free, serial, 3D, 4096 degrees of freedom: 40.188 KiB held","matrix-free, serial, 3D, 32768 degrees of freedom: 273.062 KiB held","matrix-free, serial, 3D, 262144 degrees of freedom: 2.075 MiB held","matrix-free, serial, 3D, 884736 degrees of freedom: 6.977 MiB held","matrix-free, serial, 3D, 2097152 degrees of freedom: 16.520 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "assembled matrix, 3D", x: [4096,32768,262144,884736,2097152], y: [467112,3834024,31064232,105283752,250085544], line: { color: '#3b82f6', dash: 'dot' }, marker: { color: '#3b82f6', size: 6 }, hovertext: ["assembled matrix, 3D, 4096 degrees of freedom: 456.164 KiB held","assembled matrix, 3D, 32768 degrees of freedom: 3.656 MiB held","assembled matrix, 3D, 262144 degrees of freedom: 29.625 MiB held","assembled matrix, 3D, 884736 degrees of freedom: 100.406 MiB held","assembled matrix, 3D, 2097152 degrees of freedom: 238.500 MiB held"], hoverinfo: 'text' },
{ type: 'scatter', mode: 'lines+markers', name: "matrix-free, CpuThreaded(), 3D", x: [4096,32768,262144,884736,2097152], y: [41200,279664,2175856,7316128,17322400], line: { color: '#ef4444', dash: 'dot' }, marker: { color: '#ef4444', size: 6 }, hovertext: ["matrix-free, CpuThreaded(), 3D, 4096 degrees of freedom: 40.234 KiB held","matrix-free, CpuThreaded(), 3D, 32768 degrees of freedom: 273.109 KiB held","matrix-free, CpuThreaded(), 3D, 262144 degrees of freedom: 2.075 MiB held","matrix-free, CpuThreaded(), 3D, 884736 degrees of freedom: 6.977 MiB held","matrix-free, CpuThreaded(), 3D, 2097152 degrees of freedom: 16.520 MiB held"], hoverinfo: 'text' }];
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
      title: { text: "bytes held", font: { color: theme.text } },
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

### Execution-policy crossover (`policy_crossover.jl`)

The smallest size, in degrees of freedom, from which each host policy beats the one it is compared with, per workload and dimension, on a non-uniform mesh; a win counts only when the next larger size wins too. A missing bar means the sweep found no crossover, and the crossover depends on the thread count of the run.

Run on Apple M2 with 4 threads, power: AC, commit `a6e862d4`, Julia 1.13.1, 2026-10-02T17:42:22. The script waited for a quiet machine before starting; the load average at the end of the run, which includes the run itself, was 5.03.

```@raw html
<div style="width:100%; margin:1.2rem 0 2.5rem 0;">
<div id="bench_chart_12" data-bench="standalone-policy_crossover" data-run="a6e862d4" style="width:100%; height:520px;"></div>
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
  Plotly.newPlot('bench_chart_12', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_12', function () {
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
<div id="bench_chart_13" data-bench="standalone-gpu_offload" data-run="af288c31" style="width:100%; height:380px;"></div>
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
  Plotly.newPlot('bench_chart_13', data, layout, { displayModeBar: false, responsive: true });
  window.brambleRegisterPlotlyChart('bench_chart_13', function () {
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
