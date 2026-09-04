# Performance and benchmarks

Bramble tracks memory allocations and performance regressions with a dedicated regression suite in `benchmark/benchmarks.jl`.
All measurements below are run on **1,000,000 grid points** per dimension setup (e.g. ``1000 \times 1000`` in 2D, ``100 \times 100 \times 100`` in 3D).

## Recorded baselines

Comparing **6** recorded baselines in chronological order. The earliest run (`0b9a62b`) is the reference baseline for relative speedup/slowdown calculations.

| Commit | Julia | Summary | File |
|---|:---:|---|---|
| `0b9a62b` *(baseline)* | `1.12.7` | test: run the allocation assertions under coverage instead of skipping them | `baseline_0b9a62b.json` |
| `855fbf5` | `1.12.7` | docs(benchmarks): switch to inline SVG charts and streamline baselines table | `baseline_855fbf5.json` |
| `41036bb` | `1.12.7` | fix(space): only fetch the Gauss rule inside the kernel where it truly folds | `baseline_41036bb.json` |
| `15f5e3b` | `1.12.7` | test(form): measure the evaluate! allocation behind a barrier | `baseline_15f5e3b.json` |
| `e6655b1` | `1.12.7` | perf(space): add specialized 2D and 3D tensor-product loops to _cell_average for faster, zero-alloc cell quadrature | `baseline_e6655b1.json` |
| `2dec0c7` | `1.12.7` | chore: bump version to 2.0.0 | `baseline_2dec0c7.json` |

## Comparative timings and allocations

```@raw html
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

```

### Operators 2D

```@raw html
<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;">
  <div style="flex:1 1 430px; min-width:320px; overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:8px 6px; text-align:left;">Benchmark</th>
<th style="padding:8px 6px; text-align:right;">Base (<code>0b9a62b</code>)</th>
<th style="padding:8px 6px; text-align:right;">Prev (<code>e6655b1</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>2dec0c7</code>)</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Prev</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Dcₓ</code></td>
<td style="padding:7px 6px; text-align:right;">257.2 μs</td>
<td style="padding:7px 6px; text-align:right;">261.5 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">256.5 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-1.9% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋ᵧ</code></td>
<td style="padding:7px 6px; text-align:right;">161.4 μs</td>
<td style="padding:7px 6px; text-align:right;">161.2 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">162.0 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋ₓ</code></td>
<td style="padding:7px 6px; text-align:right;">203.7 μs</td>
<td style="padding:7px 6px; text-align:right;">205.2 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">204.0 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.6% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₋ₓ</code></td>
<td style="padding:7px 6px; text-align:right;">171.4 μs</td>
<td style="padding:7px 6px; text-align:right;">155.5 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">172.0 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+10.6% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:560px;">
  <canvas id="bench_chart_1" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_1').getContext('2d'), {
    type: 'line',
    data: { labels: ["0b9a62b","855fbf5","41036bb","15f5e3b","e6655b1","2dec0c7"], datasets: [{
  label: "Dcₓ",
  data: [{x:"0b9a62b",y:257.208,julia:"1.12.7",
detail:"257.2 μs",allocs:3,mem:"7.64 MiB"},{x:"855fbf5",y:256.542,julia:"1.12.7",
detail:"256.5 μs",allocs:3,mem:"7.64 MiB"},{x:"41036bb",y:255.041,julia:"1.12.7",
detail:"255.0 μs",allocs:3,mem:"7.64 MiB"},{x:"15f5e3b",y:260.0835,julia:"1.12.7",
detail:"260.1 μs",allocs:3,mem:"7.64 MiB"},{x:"e6655b1",y:261.458,julia:"1.12.7",
detail:"261.5 μs",allocs:3,mem:"7.64 MiB"},{x:"2dec0c7",y:256.5,julia:"1.12.7",
detail:"256.5 μs",allocs:3,mem:"7.64 MiB"}],
  borderColor: "#3b82f6",
  backgroundColor: "#3b82f6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "D₋ᵧ",
  data: [{x:"0b9a62b",y:161.417,julia:"1.12.7",
detail:"161.4 μs",allocs:3,mem:"7.64 MiB"},{x:"855fbf5",y:162.625,julia:"1.12.7",
detail:"162.6 μs",allocs:3,mem:"7.64 MiB"},{x:"41036bb",y:162.0,julia:"1.12.7",
detail:"162.0 μs",allocs:3,mem:"7.64 MiB"},{x:"15f5e3b",y:161.5,julia:"1.12.7",
detail:"161.5 μs",allocs:3,mem:"7.64 MiB"},{x:"e6655b1",y:161.25,julia:"1.12.7",
detail:"161.2 μs",allocs:3,mem:"7.64 MiB"},{x:"2dec0c7",y:161.958,julia:"1.12.7",
detail:"162.0 μs",allocs:3,mem:"7.64 MiB"}],
  borderColor: "#10b981",
  backgroundColor: "#10b981",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "D₋ₓ",
  data: [{x:"0b9a62b",y:203.709,julia:"1.12.7",
detail:"203.7 μs",allocs:3,mem:"7.64 MiB"},{x:"855fbf5",y:203.2915,julia:"1.12.7",
detail:"203.3 μs",allocs:3,mem:"7.64 MiB"},{x:"41036bb",y:203.666,julia:"1.12.7",
detail:"203.7 μs",allocs:3,mem:"7.64 MiB"},{x:"15f5e3b",y:203.083,julia:"1.12.7",
detail:"203.1 μs",allocs:3,mem:"7.64 MiB"},{x:"e6655b1",y:205.25,julia:"1.12.7",
detail:"205.2 μs",allocs:3,mem:"7.64 MiB"},{x:"2dec0c7",y:204.042,julia:"1.12.7",
detail:"204.0 μs",allocs:3,mem:"7.64 MiB"}],
  borderColor: "#f59e0b",
  backgroundColor: "#f59e0b",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "M₋ₓ",
  data: [{x:"0b9a62b",y:171.416,julia:"1.12.7",
detail:"171.4 μs",allocs:3,mem:"7.64 MiB"},{x:"855fbf5",y:171.041,julia:"1.12.7",
detail:"171.0 μs",allocs:3,mem:"7.64 MiB"},{x:"41036bb",y:168.166,julia:"1.12.7",
detail:"168.2 μs",allocs:3,mem:"7.64 MiB"},{x:"15f5e3b",y:173.0835,julia:"1.12.7",
detail:"173.1 μs",allocs:3,mem:"7.64 MiB"},{x:"e6655b1",y:155.542,julia:"1.12.7",
detail:"155.5 μs",allocs:3,mem:"7.64 MiB"},{x:"2dec0c7",y:171.959,julia:"1.12.7",
detail:"172.0 μs",allocs:3,mem:"7.64 MiB"}],
  borderColor: "#8b5cf6",
  backgroundColor: "#8b5cf6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
}] },
    options: {
      responsive: true,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { color: theme.text, boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: (items) => items[0].raw.x + " (Julia " + items[0].raw.julia + ")",
            label: (c) => c.raw ? c.dataset.label + ": " + (c.raw.detail || c.raw.y) +
              (c.raw.allocs !== undefined ? " (" + c.raw.allocs + " allocs, " + c.raw.mem + ")" : "") : c.dataset.label,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: theme.text, maxRotation: 45, minRotation: 0, font: { family: 'monospace', size: 10 } },
          grid: { color: theme.grid }, border: { color: theme.axis },
        },
        y: {
          title: { display: true, text: "μs", color: theme.text },
          ticks: { color: theme.text }, grid: { color: theme.grid }, border: { color: theme.axis },
        },
      },
    },
  });
  window.brambleRegisterChart(chart, function (c) {
    const t = window.brambleChartTheme();
    c.options.plugins.legend.labels.color = t.text;
    c.options.scales.x.ticks.color = t.text;
    c.options.scales.x.grid.color = t.grid;
    c.options.scales.x.border.color = t.axis;
    c.options.scales.y.title.color = t.text;
    c.options.scales.y.ticks.color = t.text;
    c.options.scales.y.grid.color = t.grid;
    c.options.scales.y.border.color = t.axis;
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
<th style="padding:8px 6px; text-align:right;">Base (<code>0b9a62b</code>)</th>
<th style="padding:8px 6px; text-align:right;">Prev (<code>e6655b1</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>2dec0c7</code>)</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Prev</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋₂</code></td>
<td style="padding:7px 6px; text-align:right;">200.9 μs</td>
<td style="padding:7px 6px; text-align:right;">201.0 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">229.6 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+14.3% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+14.2% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>innerₕ</code></td>
<td style="padding:7px 6px; text-align:right;">240.2 μs</td>
<td style="padding:7px 6px; text-align:right;">238.5 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">240.4 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.8% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>∇₋ₕ</code></td>
<td style="padding:7px 6px; text-align:right;">694.1 μs</td>
<td style="padding:7px 6px; text-align:right;">706.5 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">690.8 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.5% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-2.2% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">15</td>
<td style="padding:7px 6px; text-align:right;">22.92 MiB</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:560px;">
  <canvas id="bench_chart_2" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_2').getContext('2d'), {
    type: 'line',
    data: { labels: ["0b9a62b","855fbf5","41036bb","15f5e3b","e6655b1","2dec0c7"], datasets: [{
  label: "D₋₂",
  data: [{x:"0b9a62b",y:200.917,julia:"1.12.7",
detail:"200.9 μs",allocs:3,mem:"7.64 MiB"},{x:"855fbf5",y:222.791,julia:"1.12.7",
detail:"222.8 μs",allocs:3,mem:"7.64 MiB"},{x:"41036bb",y:228.125,julia:"1.12.7",
detail:"228.1 μs",allocs:3,mem:"7.64 MiB"},{x:"15f5e3b",y:229.667,julia:"1.12.7",
detail:"229.7 μs",allocs:3,mem:"7.64 MiB"},{x:"e6655b1",y:201.0205,julia:"1.12.7",
detail:"201.0 μs",allocs:3,mem:"7.64 MiB"},{x:"2dec0c7",y:229.584,julia:"1.12.7",
detail:"229.6 μs",allocs:3,mem:"7.64 MiB"}],
  borderColor: "#3b82f6",
  backgroundColor: "#3b82f6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "innerₕ",
  data: [{x:"0b9a62b",y:240.25,julia:"1.12.7",
detail:"240.2 μs",allocs:0,mem:"0 B"},{x:"855fbf5",y:240.0,julia:"1.12.7",
detail:"240.0 μs",allocs:0,mem:"0 B"},{x:"41036bb",y:239.458,julia:"1.12.7",
detail:"239.5 μs",allocs:0,mem:"0 B"},{x:"15f5e3b",y:236.958,julia:"1.12.7",
detail:"237.0 μs",allocs:0,mem:"0 B"},{x:"e6655b1",y:238.542,julia:"1.12.7",
detail:"238.5 μs",allocs:0,mem:"0 B"},{x:"2dec0c7",y:240.375,julia:"1.12.7",
detail:"240.4 μs",allocs:0,mem:"0 B"}],
  borderColor: "#10b981",
  backgroundColor: "#10b981",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "∇₋ₕ",
  data: [{x:"0b9a62b",y:694.084,julia:"1.12.7",
detail:"694.1 μs",allocs:15,mem:"22.92 MiB"},{x:"855fbf5",y:686.625,julia:"1.12.7",
detail:"686.6 μs",allocs:15,mem:"22.92 MiB"},{x:"41036bb",y:685.667,julia:"1.12.7",
detail:"685.7 μs",allocs:15,mem:"22.92 MiB"},{x:"15f5e3b",y:688.0,julia:"1.12.7",
detail:"688.0 μs",allocs:15,mem:"22.92 MiB"},{x:"e6655b1",y:706.521,julia:"1.12.7",
detail:"706.5 μs",allocs:15,mem:"22.92 MiB"},{x:"2dec0c7",y:690.75,julia:"1.12.7",
detail:"690.8 μs",allocs:15,mem:"22.92 MiB"}],
  borderColor: "#f59e0b",
  backgroundColor: "#f59e0b",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
}] },
    options: {
      responsive: true,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { color: theme.text, boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: (items) => items[0].raw.x + " (Julia " + items[0].raw.julia + ")",
            label: (c) => c.raw ? c.dataset.label + ": " + (c.raw.detail || c.raw.y) +
              (c.raw.allocs !== undefined ? " (" + c.raw.allocs + " allocs, " + c.raw.mem + ")" : "") : c.dataset.label,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: theme.text, maxRotation: 45, minRotation: 0, font: { family: 'monospace', size: 10 } },
          grid: { color: theme.grid }, border: { color: theme.axis },
        },
        y: {
          title: { display: true, text: "μs", color: theme.text },
          ticks: { color: theme.text }, grid: { color: theme.grid }, border: { color: theme.axis },
        },
      },
    },
  });
  window.brambleRegisterChart(chart, function (c) {
    const t = window.brambleChartTheme();
    c.options.plugins.legend.labels.color = t.text;
    c.options.scales.x.ticks.color = t.text;
    c.options.scales.x.grid.color = t.grid;
    c.options.scales.x.border.color = t.axis;
    c.options.scales.y.title.color = t.text;
    c.options.scales.y.ticks.color = t.text;
    c.options.scales.y.grid.color = t.grid;
    c.options.scales.y.border.color = t.axis;
  });
})();
</script>

  </div>
</div>
```

### Jumps & Averages

```@raw html
<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;">
  <div style="flex:1 1 430px; min-width:320px; overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:8px 6px; text-align:left;">Benchmark</th>
<th style="padding:8px 6px; text-align:right;">Base (<code>0b9a62b</code>)</th>
<th style="padding:8px 6px; text-align:right;">Prev (<code>e6655b1</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>2dec0c7</code>)</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Prev</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₊ᵧ 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">153.5 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">161.3 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+5.1% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₊₂ 3D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">221.0 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">227.7 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+3.0% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₊ₓ 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">152.3 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">161.6 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+6.1% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>jumpᵧ 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">159.8 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">160.8 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.6% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>jump₂ 3D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">210.4 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">227.3 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+8.1% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>jumpₓ 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">163.0 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">161.5 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.9% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:560px;">
  <canvas id="bench_chart_3" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_3').getContext('2d'), {
    type: 'line',
    data: { labels: ["0b9a62b","855fbf5","41036bb","15f5e3b","e6655b1","2dec0c7"], datasets: [{
  label: "M₊ᵧ 2D",
  data: [null,{x:"855fbf5",y:161.625,julia:"1.12.7",
detail:"161.6 μs",allocs:3,mem:"7.64 MiB"},{x:"41036bb",y:165.125,julia:"1.12.7",
detail:"165.1 μs",allocs:3,mem:"7.64 MiB"},{x:"15f5e3b",y:161.709,julia:"1.12.7",
detail:"161.7 μs",allocs:3,mem:"7.64 MiB"},{x:"e6655b1",y:153.5,julia:"1.12.7",
detail:"153.5 μs",allocs:3,mem:"7.64 MiB"},{x:"2dec0c7",y:161.292,julia:"1.12.7",
detail:"161.3 μs",allocs:3,mem:"7.64 MiB"}],
  borderColor: "#3b82f6",
  backgroundColor: "#3b82f6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "M₊₂ 3D",
  data: [null,{x:"855fbf5",y:227.833,julia:"1.12.7",
detail:"227.8 μs",allocs:3,mem:"7.64 MiB"},{x:"41036bb",y:223.8125,julia:"1.12.7",
detail:"223.8 μs",allocs:3,mem:"7.64 MiB"},{x:"15f5e3b",y:227.125,julia:"1.12.7",
detail:"227.1 μs",allocs:3,mem:"7.64 MiB"},{x:"e6655b1",y:221.042,julia:"1.12.7",
detail:"221.0 μs",allocs:3,mem:"7.64 MiB"},{x:"2dec0c7",y:227.708,julia:"1.12.7",
detail:"227.7 μs",allocs:3,mem:"7.64 MiB"}],
  borderColor: "#10b981",
  backgroundColor: "#10b981",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "M₊ₓ 2D",
  data: [null,{x:"855fbf5",y:160.417,julia:"1.12.7",
detail:"160.4 μs",allocs:3,mem:"7.64 MiB"},{x:"41036bb",y:162.5,julia:"1.12.7",
detail:"162.5 μs",allocs:3,mem:"7.64 MiB"},{x:"15f5e3b",y:161.125,julia:"1.12.7",
detail:"161.1 μs",allocs:3,mem:"7.64 MiB"},{x:"e6655b1",y:152.292,julia:"1.12.7",
detail:"152.3 μs",allocs:3,mem:"7.64 MiB"},{x:"2dec0c7",y:161.625,julia:"1.12.7",
detail:"161.6 μs",allocs:3,mem:"7.64 MiB"}],
  borderColor: "#f59e0b",
  backgroundColor: "#f59e0b",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "jumpᵧ 2D",
  data: [null,{x:"855fbf5",y:162.0,julia:"1.12.7",
detail:"162.0 μs",allocs:3,mem:"7.64 MiB"},{x:"41036bb",y:160.833,julia:"1.12.7",
detail:"160.8 μs",allocs:3,mem:"7.64 MiB"},{x:"15f5e3b",y:160.292,julia:"1.12.7",
detail:"160.3 μs",allocs:3,mem:"7.64 MiB"},{x:"e6655b1",y:159.8335,julia:"1.12.7",
detail:"159.8 μs",allocs:3,mem:"7.64 MiB"},{x:"2dec0c7",y:160.834,julia:"1.12.7",
detail:"160.8 μs",allocs:3,mem:"7.64 MiB"}],
  borderColor: "#8b5cf6",
  backgroundColor: "#8b5cf6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "jump₂ 3D",
  data: [null,{x:"855fbf5",y:227.625,julia:"1.12.7",
detail:"227.6 μs",allocs:3,mem:"7.64 MiB"},{x:"41036bb",y:227.625,julia:"1.12.7",
detail:"227.6 μs",allocs:3,mem:"7.64 MiB"},{x:"15f5e3b",y:227.25,julia:"1.12.7",
detail:"227.2 μs",allocs:3,mem:"7.64 MiB"},{x:"e6655b1",y:210.3955,julia:"1.12.7",
detail:"210.4 μs",allocs:3,mem:"7.64 MiB"},{x:"2dec0c7",y:227.334,julia:"1.12.7",
detail:"227.3 μs",allocs:3,mem:"7.64 MiB"}],
  borderColor: "#ec4899",
  backgroundColor: "#ec4899",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "jumpₓ 2D",
  data: [null,{x:"855fbf5",y:164.667,julia:"1.12.7",
detail:"164.7 μs",allocs:3,mem:"7.64 MiB"},{x:"41036bb",y:162.5,julia:"1.12.7",
detail:"162.5 μs",allocs:3,mem:"7.64 MiB"},{x:"15f5e3b",y:162.917,julia:"1.12.7",
detail:"162.9 μs",allocs:3,mem:"7.64 MiB"},{x:"e6655b1",y:162.958,julia:"1.12.7",
detail:"163.0 μs",allocs:3,mem:"7.64 MiB"},{x:"2dec0c7",y:161.458,julia:"1.12.7",
detail:"161.5 μs",allocs:3,mem:"7.64 MiB"}],
  borderColor: "#06b6d4",
  backgroundColor: "#06b6d4",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
}] },
    options: {
      responsive: true,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { color: theme.text, boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: (items) => items[0].raw.x + " (Julia " + items[0].raw.julia + ")",
            label: (c) => c.raw ? c.dataset.label + ": " + (c.raw.detail || c.raw.y) +
              (c.raw.allocs !== undefined ? " (" + c.raw.allocs + " allocs, " + c.raw.mem + ")" : "") : c.dataset.label,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: theme.text, maxRotation: 45, minRotation: 0, font: { family: 'monospace', size: 10 } },
          grid: { color: theme.grid }, border: { color: theme.axis },
        },
        y: {
          title: { display: true, text: "μs", color: theme.text },
          ticks: { color: theme.text }, grid: { color: theme.grid }, border: { color: theme.axis },
        },
      },
    },
  });
  window.brambleRegisterChart(chart, function (c) {
    const t = window.brambleChartTheme();
    c.options.plugins.legend.labels.color = t.text;
    c.options.scales.x.ticks.color = t.text;
    c.options.scales.x.grid.color = t.grid;
    c.options.scales.x.border.color = t.axis;
    c.options.scales.y.title.color = t.text;
    c.options.scales.y.ticks.color = t.text;
    c.options.scales.y.grid.color = t.grid;
    c.options.scales.y.border.color = t.axis;
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
<th style="padding:8px 6px; text-align:right;">Base (<code>0b9a62b</code>)</th>
<th style="padding:8px 6px; text-align:right;">Prev (<code>e6655b1</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>2dec0c7</code>)</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Prev</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>innerₕ</code></td>
<td style="padding:7px 6px; text-align:right;">242.0 μs</td>
<td style="padding:7px 6px; text-align:right;">242.2 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">240.4 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.7% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.8% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>norm₁ₕ</code></td>
<td style="padding:7px 6px; text-align:right;">790.2 μs</td>
<td style="padding:7px 6px; text-align:right;">799.8 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">789.8 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-1.3% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>normₕ</code></td>
<td style="padding:7px 6px; text-align:right;">190.0 μs</td>
<td style="padding:7px 6px; text-align:right;">188.0 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">189.3 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.7% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>snorm₁ₕ</code></td>
<td style="padding:7px 6px; text-align:right;">578.1 μs</td>
<td style="padding:7px 6px; text-align:right;">580.6 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">578.5 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:560px;">
  <canvas id="bench_chart_4" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_4').getContext('2d'), {
    type: 'line',
    data: { labels: ["0b9a62b","855fbf5","41036bb","15f5e3b","e6655b1","2dec0c7"], datasets: [{
  label: "innerₕ",
  data: [{x:"0b9a62b",y:242.041,julia:"1.12.7",
detail:"242.0 μs",allocs:0,mem:"0 B"},{x:"855fbf5",y:238.458,julia:"1.12.7",
detail:"238.5 μs",allocs:0,mem:"0 B"},{x:"41036bb",y:239.958,julia:"1.12.7",
detail:"240.0 μs",allocs:0,mem:"0 B"},{x:"15f5e3b",y:239.792,julia:"1.12.7",
detail:"239.8 μs",allocs:0,mem:"0 B"},{x:"e6655b1",y:242.25,julia:"1.12.7",
detail:"242.2 μs",allocs:0,mem:"0 B"},{x:"2dec0c7",y:240.375,julia:"1.12.7",
detail:"240.4 μs",allocs:0,mem:"0 B"}],
  borderColor: "#3b82f6",
  backgroundColor: "#3b82f6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "norm₁ₕ",
  data: [{x:"0b9a62b",y:790.25,julia:"1.12.7",
detail:"790.2 μs",allocs:0,mem:"0 B"},{x:"855fbf5",y:782.917,julia:"1.12.7",
detail:"782.9 μs",allocs:0,mem:"0 B"},{x:"41036bb",y:796.625,julia:"1.12.7",
detail:"796.6 μs",allocs:0,mem:"0 B"},{x:"15f5e3b",y:783.708,julia:"1.12.7",
detail:"783.7 μs",allocs:0,mem:"0 B"},{x:"e6655b1",y:799.791,julia:"1.12.7",
detail:"799.8 μs",allocs:0,mem:"0 B"},{x:"2dec0c7",y:789.791,julia:"1.12.7",
detail:"789.8 μs",allocs:0,mem:"0 B"}],
  borderColor: "#10b981",
  backgroundColor: "#10b981",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "normₕ",
  data: [{x:"0b9a62b",y:190.0,julia:"1.12.7",
detail:"190.0 μs",allocs:0,mem:"0 B"},{x:"855fbf5",y:186.834,julia:"1.12.7",
detail:"186.8 μs",allocs:0,mem:"0 B"},{x:"41036bb",y:189.416,julia:"1.12.7",
detail:"189.4 μs",allocs:0,mem:"0 B"},{x:"15f5e3b",y:188.333,julia:"1.12.7",
detail:"188.3 μs",allocs:0,mem:"0 B"},{x:"e6655b1",y:188.042,julia:"1.12.7",
detail:"188.0 μs",allocs:0,mem:"0 B"},{x:"2dec0c7",y:189.292,julia:"1.12.7",
detail:"189.3 μs",allocs:0,mem:"0 B"}],
  borderColor: "#f59e0b",
  backgroundColor: "#f59e0b",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "snorm₁ₕ",
  data: [{x:"0b9a62b",y:578.083,julia:"1.12.7",
detail:"578.1 μs",allocs:0,mem:"0 B"},{x:"855fbf5",y:577.084,julia:"1.12.7",
detail:"577.1 μs",allocs:0,mem:"0 B"},{x:"41036bb",y:582.417,julia:"1.12.7",
detail:"582.4 μs",allocs:0,mem:"0 B"},{x:"15f5e3b",y:577.542,julia:"1.12.7",
detail:"577.5 μs",allocs:0,mem:"0 B"},{x:"e6655b1",y:580.625,julia:"1.12.7",
detail:"580.6 μs",allocs:0,mem:"0 B"},{x:"2dec0c7",y:578.5,julia:"1.12.7",
detail:"578.5 μs",allocs:0,mem:"0 B"}],
  borderColor: "#8b5cf6",
  backgroundColor: "#8b5cf6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
}] },
    options: {
      responsive: true,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { color: theme.text, boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: (items) => items[0].raw.x + " (Julia " + items[0].raw.julia + ")",
            label: (c) => c.raw ? c.dataset.label + ": " + (c.raw.detail || c.raw.y) +
              (c.raw.allocs !== undefined ? " (" + c.raw.allocs + " allocs, " + c.raw.mem + ")" : "") : c.dataset.label,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: theme.text, maxRotation: 45, minRotation: 0, font: { family: 'monospace', size: 10 } },
          grid: { color: theme.grid }, border: { color: theme.axis },
        },
        y: {
          title: { display: true, text: "μs", color: theme.text },
          ticks: { color: theme.text }, grid: { color: theme.grid }, border: { color: theme.axis },
        },
      },
    },
  });
  window.brambleRegisterChart(chart, function (c) {
    const t = window.brambleChartTheme();
    c.options.plugins.legend.labels.color = t.text;
    c.options.scales.x.ticks.color = t.text;
    c.options.scales.x.grid.color = t.grid;
    c.options.scales.x.border.color = t.axis;
    c.options.scales.y.title.color = t.text;
    c.options.scales.y.ticks.color = t.text;
    c.options.scales.y.grid.color = t.grid;
    c.options.scales.y.border.color = t.axis;
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
<th style="padding:8px 6px; text-align:right;">Base (<code>0b9a62b</code>)</th>
<th style="padding:8px 6px; text-align:right;">Prev (<code>e6655b1</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>2dec0c7</code>)</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Prev</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ 1D (allocates its output)</code></td>
<td style="padding:7px 6px; text-align:right;">2.87 ms</td>
<td style="padding:7px 6px; text-align:right;">2.92 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">3.2 ms</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+11.2% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+9.5% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">2.87 ms</td>
<td style="padding:7px 6px; text-align:right;">2.89 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">3.19 ms</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+11.5% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+10.4% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">448 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 1D, Serial() backend (default)</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">2.95 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">3.54 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">3.82 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+8.1% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">448 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 3D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">4.34 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">4.44 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+2.2% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">464 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">16.27 ms</td>
<td style="padding:7px 6px; text-align:right;">16.23 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">16.74 ms</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+2.8% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+3.1% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">544 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 1D, Serial() backend (default)</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">17.32 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">105.79 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">106.39 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.6% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">560 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 3D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">655.97 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">620.75 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-5.4% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">576 B</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:560px;">
  <canvas id="bench_chart_5" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_5').getContext('2d'), {
    type: 'line',
    data: { labels: ["0b9a62b","855fbf5","41036bb","15f5e3b","e6655b1","2dec0c7"], datasets: [{
  label: "Rₕ 1D (allocates its output)",
  data: [{x:"0b9a62b",y:1.0,julia:"1.12.7",
detail:"2.87 ms (baseline)",allocs:6,mem:"7.64 MiB"},{x:"855fbf5",y:1.0020014793543053,julia:"1.12.7",
detail:"2.88 ms (+0.2%)",allocs:6,mem:"7.64 MiB"},{x:"41036bb",y:1.0074691728669016,julia:"1.12.7",
detail:"2.89 ms (+0.7%)",allocs:11,mem:"7.64 MiB"},{x:"15f5e3b",y:1.00423513031371,julia:"1.12.7",
detail:"2.89 ms (+0.4%)",allocs:11,mem:"7.64 MiB"},{x:"e6655b1",y:1.0162438323978593,julia:"1.12.7",
detail:"2.92 ms (+1.6%)",allocs:6,mem:"7.64 MiB"},{x:"2dec0c7",y:1.1122932602358264,julia:"1.12.7",
detail:"3.2 ms (+11.2%)",allocs:10,mem:"7.64 MiB"}],
  borderColor: "#3b82f6",
  backgroundColor: "#3b82f6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "Rₕ! 1D",
  data: [{x:"0b9a62b",y:1.0,julia:"1.12.7",
detail:"2.87 ms (baseline)",allocs:3,mem:"64 B"},{x:"855fbf5",y:1.0046082860880943,julia:"1.12.7",
detail:"2.88 ms (+0.5%)",allocs:3,mem:"64 B"},{x:"41036bb",y:1.02808547754034,julia:"1.12.7",
detail:"2.95 ms (+2.8%)",allocs:0,mem:"0 B"},{x:"15f5e3b",y:1.0301569995638902,julia:"1.12.7",
detail:"2.95 ms (+3.0%)",allocs:0,mem:"0 B"},{x:"e6655b1",y:1.0095361535106846,julia:"1.12.7",
detail:"2.89 ms (+1.0%)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:1.1145224596598342,julia:"1.12.7",
detail:"3.19 ms (+11.5%)",allocs:7,mem:"448 B"}],
  borderColor: "#10b981",
  backgroundColor: "#10b981",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "Rₕ! 1D, Serial() backend (default)",
  data: [null,null,null,null,null,{x:"2dec0c7",y:1.0,julia:"1.12.7",
detail:"2.95 ms (baseline)",allocs:0,mem:"0 B"}],
  borderColor: "#f59e0b",
  backgroundColor: "#f59e0b",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "Rₕ! 2D",
  data: [null,null,{x:"41036bb",y:1.0,julia:"1.12.7",
detail:"3.39 ms (baseline)",allocs:0,mem:"0 B"},{x:"15f5e3b",y:1.007540760969849,julia:"1.12.7",
detail:"3.41 ms (+0.8%)",allocs:0,mem:"0 B"},{x:"e6655b1",y:1.043731778425656,julia:"1.12.7",
detail:"3.54 ms (+4.4%)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:1.1278125253718123,julia:"1.12.7",
detail:"3.82 ms (+12.8%)",allocs:7,mem:"448 B"}],
  borderColor: "#8b5cf6",
  backgroundColor: "#8b5cf6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "Rₕ! 3D",
  data: [null,null,{x:"41036bb",y:1.0,julia:"1.12.7",
detail:"3.84 ms (baseline)",allocs:0,mem:"0 B"},{x:"15f5e3b",y:1.0008362910433581,julia:"1.12.7",
detail:"3.84 ms (+0.1%)",allocs:0,mem:"0 B"},{x:"e6655b1",y:1.1324801261956277,julia:"1.12.7",
detail:"4.34 ms (+13.2%)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:1.1578988386280187,julia:"1.12.7",
detail:"4.44 ms (+15.8%)",allocs:7,mem:"464 B"}],
  borderColor: "#ec4899",
  backgroundColor: "#ec4899",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "avgₕ! 1D",
  data: [{x:"0b9a62b",y:1.0,julia:"1.12.7",
detail:"16.27 ms (baseline)",allocs:2,mem:"128 B"},{x:"855fbf5",y:1.005656810495804,julia:"1.12.7",
detail:"16.37 ms (+0.6%)",allocs:2,mem:"128 B"},{x:"41036bb",y:1.0023477158374103,julia:"1.12.7",
detail:"16.31 ms (+0.2%)",allocs:3,mem:"48 B"},{x:"15f5e3b",y:0.982308718733836,julia:"1.12.7",
detail:"15.99 ms (-1.8%)",allocs:3,mem:"48 B"},{x:"e6655b1",y:0.997493508573083,julia:"1.12.7",
detail:"16.23 ms (-0.3%)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:1.0283622728062025,julia:"1.12.7",
detail:"16.74 ms (+2.8%)",allocs:7,mem:"544 B"}],
  borderColor: "#06b6d4",
  backgroundColor: "#06b6d4",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "avgₕ! 1D, Serial() backend (default)",
  data: [null,null,null,null,null,{x:"2dec0c7",y:1.0,julia:"1.12.7",
detail:"17.32 ms (baseline)",allocs:0,mem:"0 B"}],
  borderColor: "#f97316",
  backgroundColor: "#f97316",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "avgₕ! 2D",
  data: [null,null,{x:"41036bb",y:1.0,julia:"1.12.7",
detail:"122.55 ms (baseline)",allocs:4,mem:"128 B"},{x:"15f5e3b",y:0.9950879352926364,julia:"1.12.7",
detail:"121.95 ms (-0.5%)",allocs:4,mem:"128 B"},{x:"e6655b1",y:0.8632950112333189,julia:"1.12.7",
detail:"105.79 ms (-13.7%)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:0.868153015068287,julia:"1.12.7",
detail:"106.39 ms (-13.2%)",allocs:7,mem:"560 B"}],
  borderColor: "#3b82f6",
  backgroundColor: "#3b82f6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "avgₕ! 3D",
  data: [null,null,{x:"41036bb",y:1.0,julia:"1.12.7",
detail:"755.03 ms (baseline)",allocs:4,mem:"144 B"},{x:"15f5e3b",y:0.9942083628646255,julia:"1.12.7",
detail:"750.65 ms (-0.6%)",allocs:4,mem:"144 B"},{x:"e6655b1",y:0.8688007671265353,julia:"1.12.7",
detail:"655.97 ms (-13.1%)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:0.8221637041006734,julia:"1.12.7",
detail:"620.75 ms (-17.8%)",allocs:7,mem:"576 B"}],
  borderColor: "#10b981",
  backgroundColor: "#10b981",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "1.0x (ref)",
  data: [{x:"0b9a62b",y:1},{x:"855fbf5",y:1},{x:"41036bb",y:1},{x:"15f5e3b",y:1},{x:"e6655b1",y:1},{x:"2dec0c7",y:1}],
  borderColor: "rgba(128,128,128,0.7)",
  borderDash: [5,4],
  borderWidth: 1.5,
  pointRadius: 0,
}] },
    options: {
      responsive: true,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { color: theme.text, boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: (items) => items[0].raw.x + " (Julia " + items[0].raw.julia + ")",
            label: (c) => c.raw ? c.dataset.label + ": " + (c.raw.detail || c.raw.y) +
              (c.raw.allocs !== undefined ? " (" + c.raw.allocs + " allocs, " + c.raw.mem + ")" : "") : c.dataset.label,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: theme.text, maxRotation: 45, minRotation: 0, font: { family: 'monospace', size: 10 } },
          grid: { color: theme.grid }, border: { color: theme.axis },
        },
        y: {
          title: { display: true, text: "relative to baseline", color: theme.text },
          ticks: { color: theme.text }, grid: { color: theme.grid }, border: { color: theme.axis },
        },
      },
    },
  });
  window.brambleRegisterChart(chart, function (c) {
    const t = window.brambleChartTheme();
    c.options.plugins.legend.labels.color = t.text;
    c.options.scales.x.ticks.color = t.text;
    c.options.scales.x.grid.color = t.grid;
    c.options.scales.x.border.color = t.axis;
    c.options.scales.y.title.color = t.text;
    c.options.scales.y.ticks.color = t.text;
    c.options.scales.y.grid.color = t.grid;
    c.options.scales.y.border.color = t.axis;
  });
})();
</script>

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
<th style="padding:8px 6px; text-align:right;">Base (<code>0b9a62b</code>)</th>
<th style="padding:8px 6px; text-align:right;">Prev (<code>e6655b1</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>2dec0c7</code>)</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Prev</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋ₓ (3 components)</code></td>
<td style="padding:7px 6px; text-align:right;">711.0 μs</td>
<td style="padding:7px 6px; text-align:right;">712.9 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">670.2 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-5.7% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-6.0% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">22.89 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>∇₋ₕ (3 components)</code></td>
<td style="padding:7px 6px; text-align:right;">1.43 ms</td>
<td style="padding:7px 6px; text-align:right;">1.46 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">1.38 ms</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-3.3% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-5.6% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:right;">45.78 MiB</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:560px;">
  <canvas id="bench_chart_6" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_6').getContext('2d'), {
    type: 'line',
    data: { labels: ["0b9a62b","855fbf5","41036bb","15f5e3b","e6655b1","2dec0c7"], datasets: [{
  label: "D₋ₓ (3 components)",
  data: [{x:"0b9a62b",y:0.711042,julia:"1.12.7",
detail:"711.0 μs",allocs:3,mem:"22.89 MiB"},{x:"855fbf5",y:0.675875,julia:"1.12.7",
detail:"675.9 μs",allocs:3,mem:"22.89 MiB"},{x:"41036bb",y:0.67025,julia:"1.12.7",
detail:"670.2 μs",allocs:3,mem:"22.89 MiB"},{x:"15f5e3b",y:0.672666,julia:"1.12.7",
detail:"672.7 μs",allocs:3,mem:"22.89 MiB"},{x:"e6655b1",y:0.712875,julia:"1.12.7",
detail:"712.9 μs",allocs:3,mem:"22.89 MiB"},{x:"2dec0c7",y:0.670208,julia:"1.12.7",
detail:"670.2 μs",allocs:3,mem:"22.89 MiB"}],
  borderColor: "#3b82f6",
  backgroundColor: "#3b82f6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "∇₋ₕ (3 components)",
  data: [{x:"0b9a62b",y:1.427333,julia:"1.12.7",
detail:"1.43 ms",allocs:10,mem:"45.78 MiB"},{x:"855fbf5",y:1.398333,julia:"1.12.7",
detail:"1.4 ms",allocs:10,mem:"45.78 MiB"},{x:"41036bb",y:1.4105,julia:"1.12.7",
detail:"1.41 ms",allocs:10,mem:"45.78 MiB"},{x:"15f5e3b",y:1.400542,julia:"1.12.7",
detail:"1.4 ms",allocs:10,mem:"45.78 MiB"},{x:"e6655b1",y:1.462125,julia:"1.12.7",
detail:"1.46 ms",allocs:10,mem:"45.78 MiB"},{x:"2dec0c7",y:1.380625,julia:"1.12.7",
detail:"1.38 ms",allocs:10,mem:"45.78 MiB"}],
  borderColor: "#10b981",
  backgroundColor: "#10b981",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
}] },
    options: {
      responsive: true,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { color: theme.text, boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: (items) => items[0].raw.x + " (Julia " + items[0].raw.julia + ")",
            label: (c) => c.raw ? c.dataset.label + ": " + (c.raw.detail || c.raw.y) +
              (c.raw.allocs !== undefined ? " (" + c.raw.allocs + " allocs, " + c.raw.mem + ")" : "") : c.dataset.label,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: theme.text, maxRotation: 45, minRotation: 0, font: { family: 'monospace', size: 10 } },
          grid: { color: theme.grid }, border: { color: theme.axis },
        },
        y: {
          title: { display: true, text: "ms", color: theme.text },
          ticks: { color: theme.text }, grid: { color: theme.grid }, border: { color: theme.axis },
        },
      },
    },
  });
  window.brambleRegisterChart(chart, function (c) {
    const t = window.brambleChartTheme();
    c.options.plugins.legend.labels.color = t.text;
    c.options.scales.x.ticks.color = t.text;
    c.options.scales.x.grid.color = t.grid;
    c.options.scales.x.border.color = t.axis;
    c.options.scales.y.title.color = t.text;
    c.options.scales.y.ticks.color = t.text;
    c.options.scales.y.grid.color = t.grid;
    c.options.scales.y.border.color = t.axis;
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
<th style="padding:8px 6px; text-align:right;">Base (<code>0b9a62b</code>)</th>
<th style="padding:8px 6px; text-align:right;">Prev (<code>e6655b1</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>2dec0c7</code>)</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Prev</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>gridspace 2D</code></td>
<td style="padding:7px 6px; text-align:right;">368.8 μs</td>
<td style="padding:7px 6px; text-align:right;">363.6 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">2.22 ms</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+501.9% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+510.5% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">42</td>
<td style="padding:7px 6px; text-align:right;">22.95 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>gridspace 3D</code></td>
<td style="padding:7px 6px; text-align:right;">1.63 ms</td>
<td style="padding:7px 6px; text-align:right;">1.66 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">6.2 ms</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+279.6% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+273.8% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">52</td>
<td style="padding:7px 6px; text-align:right;">30.57 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>hₘₐₓ 3D</code></td>
<td style="padding:7px 6px; text-align:right;">153.0 ns</td>
<td style="padding:7px 6px; text-align:right;">153.3 ns</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">153.0 ns</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:560px;">
  <canvas id="bench_chart_7" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_7').getContext('2d'), {
    type: 'line',
    data: { labels: ["0b9a62b","855fbf5","41036bb","15f5e3b","e6655b1","2dec0c7"], datasets: [{
  label: "gridspace 2D",
  data: [{x:"0b9a62b",y:1.0,julia:"1.12.7",
detail:"368.8 μs (baseline)",allocs:38,mem:"30.59 MiB"},{x:"855fbf5",y:0.9762209472576814,julia:"1.12.7",
detail:"360.0 μs (-2.4%)",allocs:38,mem:"30.59 MiB"},{x:"41036bb",y:0.9646974610370596,julia:"1.12.7",
detail:"355.8 μs (-3.5%)",allocs:29,mem:"30.59 MiB"},{x:"15f5e3b",y:0.9634529245252323,julia:"1.12.7",
detail:"355.3 μs (-3.7%)",allocs:21,mem:"22.95 MiB"},{x:"e6655b1",y:0.9858234005401125,julia:"1.12.7",
detail:"363.6 μs (-1.4%)",allocs:21,mem:"22.95 MiB"},{x:"2dec0c7",y:6.018535188659805,julia:"1.12.7",
detail:"2.22 ms (+501.9%)",allocs:42,mem:"22.95 MiB"}],
  borderColor: "#3b82f6",
  backgroundColor: "#3b82f6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "gridspace 3D",
  data: [{x:"0b9a62b",y:1.0,julia:"1.12.7",
detail:"1.63 ms (baseline)",allocs:44,mem:"38.21 MiB"},{x:"855fbf5",y:1.0019648446493037,julia:"1.12.7",
detail:"1.64 ms (+0.2%)",allocs:44,mem:"38.21 MiB"},{x:"41036bb",y:0.9994385283442158,julia:"1.12.7",
detail:"1.63 ms (-0.1%)",allocs:32,mem:"38.21 MiB"},{x:"15f5e3b",y:1.0000514325180871,julia:"1.12.7",
detail:"1.63 ms (+0.0%)",allocs:24,mem:"30.57 MiB"},{x:"e6655b1",y:1.0154478180366493,julia:"1.12.7",
detail:"1.66 ms (+1.5%)",allocs:24,mem:"30.57 MiB"},{x:"2dec0c7",y:3.7958778061336953,julia:"1.12.7",
detail:"6.2 ms (+279.6%)",allocs:52,mem:"30.57 MiB"}],
  borderColor: "#10b981",
  backgroundColor: "#10b981",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "hₘₐₓ 3D",
  data: [{x:"0b9a62b",y:1.0,julia:"1.12.7",
detail:"153.0 ns (baseline)",allocs:0,mem:"0 B"},{x:"855fbf5",y:1.0007088961684272,julia:"1.12.7",
detail:"153.1 ns (+0.1%)",allocs:0,mem:"0 B"},{x:"41036bb",y:0.9966936474669364,julia:"1.12.7",
detail:"152.5 ns (-0.3%)",allocs:0,mem:"0 B"},{x:"15f5e3b",y:1.0003247524752477,julia:"1.12.7",
detail:"153.1 ns (+0.0%)",allocs:0,mem:"0 B"},{x:"e6655b1",y:1.001567455034012,julia:"1.12.7",
detail:"153.3 ns (+0.2%)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:0.9999062139342768,julia:"1.12.7",
detail:"153.0 ns (-0.0%)",allocs:0,mem:"0 B"}],
  borderColor: "#f59e0b",
  backgroundColor: "#f59e0b",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "1.0x (ref)",
  data: [{x:"0b9a62b",y:1},{x:"855fbf5",y:1},{x:"41036bb",y:1},{x:"15f5e3b",y:1},{x:"e6655b1",y:1},{x:"2dec0c7",y:1}],
  borderColor: "rgba(128,128,128,0.7)",
  borderDash: [5,4],
  borderWidth: 1.5,
  pointRadius: 0,
}] },
    options: {
      responsive: true,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { color: theme.text, boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: (items) => items[0].raw.x + " (Julia " + items[0].raw.julia + ")",
            label: (c) => c.raw ? c.dataset.label + ": " + (c.raw.detail || c.raw.y) +
              (c.raw.allocs !== undefined ? " (" + c.raw.allocs + " allocs, " + c.raw.mem + ")" : "") : c.dataset.label,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: theme.text, maxRotation: 45, minRotation: 0, font: { family: 'monospace', size: 10 } },
          grid: { color: theme.grid }, border: { color: theme.axis },
        },
        y: {
          title: { display: true, text: "relative to baseline", color: theme.text },
          ticks: { color: theme.text }, grid: { color: theme.grid }, border: { color: theme.axis },
        },
      },
    },
  });
  window.brambleRegisterChart(chart, function (c) {
    const t = window.brambleChartTheme();
    c.options.plugins.legend.labels.color = t.text;
    c.options.scales.x.ticks.color = t.text;
    c.options.scales.x.grid.color = t.grid;
    c.options.scales.x.border.color = t.axis;
    c.options.scales.y.title.color = t.text;
    c.options.scales.y.ticks.color = t.text;
    c.options.scales.y.grid.color = t.grid;
    c.options.scales.y.border.color = t.axis;
  });
})();
</script>

  </div>
</div>
```

### Startup & Latency

```@raw html
<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;">
  <div style="flex:1 1 430px; min-width:320px; overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:8px 6px; text-align:left;">Benchmark</th>
<th style="padding:8px 6px; text-align:right;">Base (<code>0b9a62b</code>)</th>
<th style="padding:8px 6px; text-align:right;">Prev (<code>e6655b1</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>2dec0c7</code>)</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Prev</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>TTFX (load + first operator)</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">651.65 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">598.63 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-8.1% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:right;">1.3 KiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>using Bramble</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">556.36 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">501.71 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-9.8% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:right;">1.3 KiB</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:560px;">
  <canvas id="bench_chart_8" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_8').getContext('2d'), {
    type: 'line',
    data: { labels: ["0b9a62b","855fbf5","41036bb","15f5e3b","e6655b1","2dec0c7"], datasets: [{
  label: "TTFX (load + first operator)",
  data: [null,{x:"855fbf5",y:560.776833,julia:"1.12.7",
detail:"560.78 ms",allocs:45,mem:"1.3 KiB"},{x:"41036bb",y:623.567291,julia:"1.12.7",
detail:"623.57 ms",allocs:45,mem:"1.3 KiB"},{x:"15f5e3b",y:640.640917,julia:"1.12.7",
detail:"640.64 ms",allocs:45,mem:"1.3 KiB"},{x:"e6655b1",y:651.652791,julia:"1.12.7",
detail:"651.65 ms",allocs:45,mem:"1.3 KiB"},{x:"2dec0c7",y:598.631542,julia:"1.12.7",
detail:"598.63 ms",allocs:45,mem:"1.3 KiB"}],
  borderColor: "#3b82f6",
  backgroundColor: "#3b82f6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "using Bramble",
  data: [null,{x:"855fbf5",y:527.500292,julia:"1.12.7",
detail:"527.5 ms",allocs:45,mem:"1.3 KiB"},{x:"41036bb",y:533.709208,julia:"1.12.7",
detail:"533.71 ms",allocs:45,mem:"1.3 KiB"},{x:"15f5e3b",y:546.493334,julia:"1.12.7",
detail:"546.49 ms",allocs:45,mem:"1.3 KiB"},{x:"e6655b1",y:556.356292,julia:"1.12.7",
detail:"556.36 ms",allocs:45,mem:"1.3 KiB"},{x:"2dec0c7",y:501.70775,julia:"1.12.7",
detail:"501.71 ms",allocs:45,mem:"1.3 KiB"}],
  borderColor: "#10b981",
  backgroundColor: "#10b981",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
}] },
    options: {
      responsive: true,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { color: theme.text, boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: (items) => items[0].raw.x + " (Julia " + items[0].raw.julia + ")",
            label: (c) => c.raw ? c.dataset.label + ": " + (c.raw.detail || c.raw.y) +
              (c.raw.allocs !== undefined ? " (" + c.raw.allocs + " allocs, " + c.raw.mem + ")" : "") : c.dataset.label,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: theme.text, maxRotation: 45, minRotation: 0, font: { family: 'monospace', size: 10 } },
          grid: { color: theme.grid }, border: { color: theme.axis },
        },
        y: {
          title: { display: true, text: "ms", color: theme.text },
          ticks: { color: theme.text }, grid: { color: theme.grid }, border: { color: theme.axis },
        },
      },
    },
  });
  window.brambleRegisterChart(chart, function (c) {
    const t = window.brambleChartTheme();
    c.options.plugins.legend.labels.color = t.text;
    c.options.scales.x.ticks.color = t.text;
    c.options.scales.x.grid.color = t.grid;
    c.options.scales.x.border.color = t.axis;
    c.options.scales.y.title.color = t.text;
    c.options.scales.y.ticks.color = t.text;
    c.options.scales.y.grid.color = t.grid;
    c.options.scales.y.border.color = t.axis;
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
<th style="padding:8px 6px; text-align:right;">Base (<code>0b9a62b</code>)</th>
<th style="padding:8px 6px; text-align:right;">Prev (<code>e6655b1</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>2dec0c7</code>)</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Prev</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>allocate_system_matrix 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">3.75 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">2.84 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-24.4% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">21</td>
<td style="padding:7px 6px; text-align:right;">15.13 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble (BilinearForm), Parallel() backend</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">5.11 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;">35</td>
<td style="padding:7px 6px; text-align:right;">15.13 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble (BilinearForm), Serial() backend</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">4.71 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;">21</td>
<td style="padding:7px 6px; text-align:right;">15.13 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! (matrix) 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">1.06 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">1.07 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.6% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! 1D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">910.5 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">938.8 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+3.1% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! 1D, Parallel() backend</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">1.19 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">480 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">483.5 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">1.18 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+145.1% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble_parallel! 1D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">1.2 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">1.29 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+7.5% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">480 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble_parallel! 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">2.23 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">1.72 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-23.1% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">496 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>evaluate! 1D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">1.11 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">1.14 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+2.4% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>form (bilinear, 2D)</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">2.1 ns</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">2.1 ns</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>form (linear, 2D)</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">2.1 ns</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">2.1 ns</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>l(vₕ) 1D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">884.3 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">883.6 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:560px;">
  <canvas id="bench_chart_9" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_9').getContext('2d'), {
    type: 'line',
    data: { labels: ["0b9a62b","855fbf5","41036bb","15f5e3b","e6655b1","2dec0c7"], datasets: [{
  label: "allocate_system_matrix 2D",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"3.75 ms (baseline)",allocs:21,mem:"15.13 MiB"},{x:"2dec0c7",y:0.7555938001171498,julia:"1.12.7",
detail:"2.84 ms (-24.4%)",allocs:21,mem:"15.13 MiB"}],
  borderColor: "#3b82f6",
  backgroundColor: "#3b82f6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "assemble (BilinearForm), Parallel() backend",
  data: [null,null,null,null,null,{x:"2dec0c7",y:1.0,julia:"1.12.7",
detail:"5.11 ms (baseline)",allocs:35,mem:"15.13 MiB"}],
  borderColor: "#10b981",
  backgroundColor: "#10b981",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "assemble (BilinearForm), Serial() backend",
  data: [null,null,null,null,null,{x:"2dec0c7",y:1.0,julia:"1.12.7",
detail:"4.71 ms (baseline)",allocs:21,mem:"15.13 MiB"}],
  borderColor: "#f59e0b",
  backgroundColor: "#f59e0b",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "assemble! (matrix) 2D",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"1.06 ms (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:1.005847639232309,julia:"1.12.7",
detail:"1.07 ms (+0.6%)",allocs:0,mem:"0 B"}],
  borderColor: "#8b5cf6",
  backgroundColor: "#8b5cf6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "assemble! 1D",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"910.5 μs (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:1.0311656331209127,julia:"1.12.7",
detail:"938.8 μs (+3.1%)",allocs:0,mem:"0 B"}],
  borderColor: "#ec4899",
  backgroundColor: "#ec4899",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "assemble! 1D, Parallel() backend",
  data: [null,null,null,null,null,{x:"2dec0c7",y:1.0,julia:"1.12.7",
detail:"1.19 ms (baseline)",allocs:7,mem:"480 B"}],
  borderColor: "#06b6d4",
  backgroundColor: "#06b6d4",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "assemble! 2D",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"483.5 μs (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:2.4507931747673215,julia:"1.12.7",
detail:"1.18 ms (+145.1%)",allocs:0,mem:"0 B"}],
  borderColor: "#f97316",
  backgroundColor: "#f97316",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "assemble_parallel! 1D",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"1.2 ms (baseline)",allocs:7,mem:"480 B"},{x:"2dec0c7",y:1.0746227026198918,julia:"1.12.7",
detail:"1.29 ms (+7.5%)",allocs:7,mem:"480 B"}],
  borderColor: "#3b82f6",
  backgroundColor: "#3b82f6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "assemble_parallel! 2D",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"2.23 ms (baseline)",allocs:7,mem:"496 B"},{x:"2dec0c7",y:0.7690877108851282,julia:"1.12.7",
detail:"1.72 ms (-23.1%)",allocs:7,mem:"496 B"}],
  borderColor: "#10b981",
  backgroundColor: "#10b981",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "evaluate! 1D",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"1.11 ms (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:1.0244619294002388,julia:"1.12.7",
detail:"1.14 ms (+2.4%)",allocs:0,mem:"0 B"}],
  borderColor: "#f59e0b",
  backgroundColor: "#f59e0b",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "form (bilinear, 2D)",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"2.1 ns (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:0.9995201535508638,julia:"1.12.7",
detail:"2.1 ns (-0.0%)",allocs:0,mem:"0 B"}],
  borderColor: "#8b5cf6",
  backgroundColor: "#8b5cf6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "form (linear, 2D)",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"2.1 ns (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:0.9995201535508638,julia:"1.12.7",
detail:"2.1 ns (-0.0%)",allocs:0,mem:"0 B"}],
  borderColor: "#ec4899",
  backgroundColor: "#ec4899",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "l(vₕ) 1D",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"884.3 μs (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:0.999151904144814,julia:"1.12.7",
detail:"883.6 μs (-0.1%)",allocs:0,mem:"0 B"}],
  borderColor: "#06b6d4",
  backgroundColor: "#06b6d4",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "1.0x (ref)",
  data: [{x:"0b9a62b",y:1},{x:"855fbf5",y:1},{x:"41036bb",y:1},{x:"15f5e3b",y:1},{x:"e6655b1",y:1},{x:"2dec0c7",y:1}],
  borderColor: "rgba(128,128,128,0.7)",
  borderDash: [5,4],
  borderWidth: 1.5,
  pointRadius: 0,
}] },
    options: {
      responsive: true,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { color: theme.text, boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: (items) => items[0].raw.x + " (Julia " + items[0].raw.julia + ")",
            label: (c) => c.raw ? c.dataset.label + ": " + (c.raw.detail || c.raw.y) +
              (c.raw.allocs !== undefined ? " (" + c.raw.allocs + " allocs, " + c.raw.mem + ")" : "") : c.dataset.label,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: theme.text, maxRotation: 45, minRotation: 0, font: { family: 'monospace', size: 10 } },
          grid: { color: theme.grid }, border: { color: theme.axis },
        },
        y: {
          title: { display: true, text: "relative to baseline", color: theme.text },
          ticks: { color: theme.text }, grid: { color: theme.grid }, border: { color: theme.axis },
        },
      },
    },
  });
  window.brambleRegisterChart(chart, function (c) {
    const t = window.brambleChartTheme();
    c.options.plugins.legend.labels.color = t.text;
    c.options.scales.x.ticks.color = t.text;
    c.options.scales.x.grid.color = t.grid;
    c.options.scales.x.border.color = t.axis;
    c.options.scales.y.title.color = t.text;
    c.options.scales.y.ticks.color = t.text;
    c.options.scales.y.grid.color = t.grid;
    c.options.scales.y.border.color = t.axis;
  });
})();
</script>

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
<th style="padding:8px 6px; text-align:right;">Base (<code>0b9a62b</code>)</th>
<th style="padding:8px 6px; text-align:right;">Prev (<code>e6655b1</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>2dec0c7</code>)</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Prev</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! Double64</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">9.03 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">8.94 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-1.0% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! Float32</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">278.4 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">286.0 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+2.7% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! Float64</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">285.8 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">293.5 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+2.7% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! Double64</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">1.02 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">1.04 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+2.0% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! Float32</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">71.6 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">71.3 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! Float64</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">80.5 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">84.0 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+4.2% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! Double64</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">72.82 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">72.2 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.9% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">33</td>
<td style="padding:7px 6px; text-align:right;">2.9 KiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! Float32</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">1.6 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">1.61 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.7% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! Float64</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">1.64 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">1.72 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+4.6% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>innerₕ Double64</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">1.07 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">1.06 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.6% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>innerₕ Float32</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">11.6 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">11.6 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>innerₕ Float64</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">23.2 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">23.2 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:560px;">
  <canvas id="bench_chart_10" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_10').getContext('2d'), {
    type: 'line',
    data: { labels: ["0b9a62b","855fbf5","41036bb","15f5e3b","e6655b1","2dec0c7"], datasets: [{
  label: "Rₕ! Double64",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"9.03 ms (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:0.9900855530248217,julia:"1.12.7",
detail:"8.94 ms (-1.0%)",allocs:0,mem:"0 B"}],
  borderColor: "#3b82f6",
  backgroundColor: "#3b82f6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "Rₕ! Float32",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"278.4 μs (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:1.027391109115402,julia:"1.12.7",
detail:"286.0 μs (+2.7%)",allocs:0,mem:"0 B"}],
  borderColor: "#10b981",
  backgroundColor: "#10b981",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "Rₕ! Float64",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"285.8 μs (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:1.0272685914260717,julia:"1.12.7",
detail:"293.5 μs (+2.7%)",allocs:0,mem:"0 B"}],
  borderColor: "#f59e0b",
  backgroundColor: "#f59e0b",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "assemble! Double64",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"1.02 ms (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:1.0196884049872457,julia:"1.12.7",
detail:"1.04 ms (+2.0%)",allocs:0,mem:"0 B"}],
  borderColor: "#8b5cf6",
  backgroundColor: "#8b5cf6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "assemble! Float32",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"71.6 μs (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:0.9965075506754397,julia:"1.12.7",
detail:"71.3 μs (-0.3%)",allocs:0,mem:"0 B"}],
  borderColor: "#ec4899",
  backgroundColor: "#ec4899",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "assemble! Float64",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"80.5 μs (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:1.0424126542673389,julia:"1.12.7",
detail:"84.0 μs (+4.2%)",allocs:0,mem:"0 B"}],
  borderColor: "#06b6d4",
  backgroundColor: "#06b6d4",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "avgₕ! Double64",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"72.82 ms (baseline)",allocs:33,mem:"2.9 KiB"},{x:"2dec0c7",y:0.9914917982718104,julia:"1.12.7",
detail:"72.2 ms (-0.9%)",allocs:33,mem:"2.9 KiB"}],
  borderColor: "#f97316",
  backgroundColor: "#f97316",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "avgₕ! Float32",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"1.6 ms (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:1.0070662716091487,julia:"1.12.7",
detail:"1.61 ms (+0.7%)",allocs:0,mem:"0 B"}],
  borderColor: "#3b82f6",
  backgroundColor: "#3b82f6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "avgₕ! Float64",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"1.64 ms (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:1.0458125564220302,julia:"1.12.7",
detail:"1.72 ms (+4.6%)",allocs:0,mem:"0 B"}],
  borderColor: "#10b981",
  backgroundColor: "#10b981",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "innerₕ Double64",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"1.07 ms (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:0.9940385739333722,julia:"1.12.7",
detail:"1.06 ms (-0.6%)",allocs:0,mem:"0 B"}],
  borderColor: "#f59e0b",
  backgroundColor: "#f59e0b",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "innerₕ Float32",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"11.6 μs (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:0.9964731182795699,julia:"1.12.7",
detail:"11.6 μs (-0.4%)",allocs:0,mem:"0 B"}],
  borderColor: "#8b5cf6",
  backgroundColor: "#8b5cf6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "innerₕ Float64",
  data: [null,null,null,null,{x:"e6655b1",y:1.0,julia:"1.12.7",
detail:"23.2 μs (baseline)",allocs:0,mem:"0 B"},{x:"2dec0c7",y:1.0,julia:"1.12.7",
detail:"23.2 μs (baseline)",allocs:0,mem:"0 B"}],
  borderColor: "#ec4899",
  backgroundColor: "#ec4899",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "1.0x (ref)",
  data: [{x:"0b9a62b",y:1},{x:"855fbf5",y:1},{x:"41036bb",y:1},{x:"15f5e3b",y:1},{x:"e6655b1",y:1},{x:"2dec0c7",y:1}],
  borderColor: "rgba(128,128,128,0.7)",
  borderDash: [5,4],
  borderWidth: 1.5,
  pointRadius: 0,
}] },
    options: {
      responsive: true,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { color: theme.text, boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: (items) => items[0].raw.x + " (Julia " + items[0].raw.julia + ")",
            label: (c) => c.raw ? c.dataset.label + ": " + (c.raw.detail || c.raw.y) +
              (c.raw.allocs !== undefined ? " (" + c.raw.allocs + " allocs, " + c.raw.mem + ")" : "") : c.dataset.label,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: theme.text, maxRotation: 45, minRotation: 0, font: { family: 'monospace', size: 10 } },
          grid: { color: theme.grid }, border: { color: theme.axis },
        },
        y: {
          title: { display: true, text: "relative to baseline", color: theme.text },
          ticks: { color: theme.text }, grid: { color: theme.grid }, border: { color: theme.axis },
        },
      },
    },
  });
  window.brambleRegisterChart(chart, function (c) {
    const t = window.brambleChartTheme();
    c.options.plugins.legend.labels.color = t.text;
    c.options.scales.x.ticks.color = t.text;
    c.options.scales.x.grid.color = t.grid;
    c.options.scales.x.border.color = t.axis;
    c.options.scales.y.title.color = t.text;
    c.options.scales.y.ticks.color = t.text;
    c.options.scales.y.grid.color = t.grid;
    c.options.scales.y.border.color = t.axis;
  });
})();
</script>

  </div>
</div>
```

## How to add new benchmark runs

To record performance on a new commit or after an optimization pass, run:

```bash
julia --project=benchmark benchmark/benchmarks.jl --save benchmark/baselines/baseline_$(git rev-parse --short HEAD).json
```

Rebuilding the documentation (`julia -e 'using Pkg; Pkg.activate("docs"); include("docs/make.jl")'`) will automatically discover all `baseline_*.json` files and append new comparison columns, delta calculations, and charts.
