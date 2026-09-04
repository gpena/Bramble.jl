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
<td style="padding:7px 6px; text-align:right;">256.8 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋ᵧ</code></td>
<td style="padding:7px 6px; text-align:right;">162.0 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">162.2 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">161.1 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.5% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋ₓ</code></td>
<td style="padding:7px 6px; text-align:right;">204.0 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">205.2 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">204.4 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₋ₓ</code></td>
<td style="padding:7px 6px; text-align:right;">172.0 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">161.5 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">170.2 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-1.0% 🟢</span></td>
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
    data: { labels: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"], datasets: [{
  label: "Dcₓ",
  data: [{x:"v2.0.0 (2dec0c7)",y:256.5,julia:"1.12.7",
detail:"256.5 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.1.0 (274ae7d)",y:268.708,julia:"1.12.7",
detail:"268.7 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.2.0 (d7c2416)",y:256.75,julia:"1.12.7",
detail:"256.8 μs",allocs:3,mem:"7.64 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:161.958,julia:"1.12.7",
detail:"162.0 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.1.0 (274ae7d)",y:162.209,julia:"1.12.7",
detail:"162.2 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.2.0 (d7c2416)",y:161.083,julia:"1.12.7",
detail:"161.1 μs",allocs:3,mem:"7.64 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:204.042,julia:"1.12.7",
detail:"204.0 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.1.0 (274ae7d)",y:205.208,julia:"1.12.7",
detail:"205.2 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.2.0 (d7c2416)",y:204.375,julia:"1.12.7",
detail:"204.4 μs",allocs:3,mem:"7.64 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:171.959,julia:"1.12.7",
detail:"172.0 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.1.0 (274ae7d)",y:161.458,julia:"1.12.7",
detail:"161.5 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.2.0 (d7c2416)",y:170.25,julia:"1.12.7",
detail:"170.2 μs",allocs:3,mem:"7.64 MiB"}],
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
<td style="padding:7px 6px; text-align:right;">228.1 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.6% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>innerₕ</code></td>
<td style="padding:7px 6px; text-align:right;">240.4 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">240.3 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">239.8 μs</td>
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
<td style="padding:7px 6px; text-align:right;">695.8 μs</td>
<td style="padding:7px 6px; text-align:center;">15</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.7% 🔴</span></td>
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
    data: { labels: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"], datasets: [{
  label: "D₋₂",
  data: [{x:"v2.0.0 (2dec0c7)",y:229.584,julia:"1.12.7",
detail:"229.6 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.1.0 (274ae7d)",y:212.792,julia:"1.12.7",
detail:"212.8 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.2.0 (d7c2416)",y:228.125,julia:"1.12.7",
detail:"228.1 μs",allocs:3,mem:"7.64 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:240.375,julia:"1.12.7",
detail:"240.4 μs",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:240.292,julia:"1.12.7",
detail:"240.3 μs",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:239.75,julia:"1.12.7",
detail:"239.8 μs",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:690.75,julia:"1.12.7",
detail:"690.8 μs",allocs:15,mem:"22.92 MiB"},{x:"v2.1.0 (274ae7d)",y:696.667,julia:"1.12.7",
detail:"696.7 μs",allocs:15,mem:"22.92 MiB"},{x:"v2.2.0 (d7c2416)",y:695.834,julia:"1.12.7",
detail:"695.8 μs",allocs:15,mem:"22.92 MiB"}],
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
<td style="padding:7px 6px; text-align:right;">165.5 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+2.6% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₊₂ 3D</code></td>
<td style="padding:7px 6px; text-align:right;">227.7 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">226.8 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">225.8 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.9% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₊ₓ 2D</code></td>
<td style="padding:7px 6px; text-align:right;">161.6 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">159.6 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">159.2 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-1.5% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>jumpᵧ 2D</code></td>
<td style="padding:7px 6px; text-align:right;">160.8 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">159.9 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">161.3 μs</td>
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
<td style="padding:7px 6px; text-align:right;">225.8 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.7% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>jumpₓ 2D</code></td>
<td style="padding:7px 6px; text-align:right;">161.5 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">161.8 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">162.0 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
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
    data: { labels: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"], datasets: [{
  label: "M₊ᵧ 2D",
  data: [{x:"v2.0.0 (2dec0c7)",y:161.292,julia:"1.12.7",
detail:"161.3 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.1.0 (274ae7d)",y:161.291,julia:"1.12.7",
detail:"161.3 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.2.0 (d7c2416)",y:165.458,julia:"1.12.7",
detail:"165.5 μs",allocs:3,mem:"7.64 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:227.708,julia:"1.12.7",
detail:"227.7 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.1.0 (274ae7d)",y:226.792,julia:"1.12.7",
detail:"226.8 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.2.0 (d7c2416)",y:225.75,julia:"1.12.7",
detail:"225.8 μs",allocs:3,mem:"7.64 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:161.625,julia:"1.12.7",
detail:"161.6 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.1.0 (274ae7d)",y:159.625,julia:"1.12.7",
detail:"159.6 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.2.0 (d7c2416)",y:159.209,julia:"1.12.7",
detail:"159.2 μs",allocs:3,mem:"7.64 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:160.834,julia:"1.12.7",
detail:"160.8 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.1.0 (274ae7d)",y:159.917,julia:"1.12.7",
detail:"159.9 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.2.0 (d7c2416)",y:161.291,julia:"1.12.7",
detail:"161.3 μs",allocs:3,mem:"7.64 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:227.334,julia:"1.12.7",
detail:"227.3 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.1.0 (274ae7d)",y:226.667,julia:"1.12.7",
detail:"226.7 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.2.0 (d7c2416)",y:225.833,julia:"1.12.7",
detail:"225.8 μs",allocs:3,mem:"7.64 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:161.458,julia:"1.12.7",
detail:"161.5 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.1.0 (274ae7d)",y:161.834,julia:"1.12.7",
detail:"161.8 μs",allocs:3,mem:"7.64 MiB"},{x:"v2.2.0 (d7c2416)",y:162.041,julia:"1.12.7",
detail:"162.0 μs",allocs:3,mem:"7.64 MiB"}],
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
<td style="padding:7px 6px; text-align:right;">241.1 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>norm₁ₕ</code></td>
<td style="padding:7px 6px; text-align:right;">789.8 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">788.5 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">801.5 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+1.5% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>normₕ</code></td>
<td style="padding:7px 6px; text-align:right;">189.3 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">186.2 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">190.1 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>snorm₁ₕ</code></td>
<td style="padding:7px 6px; text-align:right;">578.5 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">582.4 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">581.9 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.6% 🔴</span></td>
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
    data: { labels: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"], datasets: [{
  label: "innerₕ",
  data: [{x:"v2.0.0 (2dec0c7)",y:240.375,julia:"1.12.7",
detail:"240.4 μs",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:238.041,julia:"1.12.7",
detail:"238.0 μs",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:241.084,julia:"1.12.7",
detail:"241.1 μs",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:789.791,julia:"1.12.7",
detail:"789.8 μs",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:788.458,julia:"1.12.7",
detail:"788.5 μs",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:801.458,julia:"1.12.7",
detail:"801.5 μs",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:189.292,julia:"1.12.7",
detail:"189.3 μs",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:186.167,julia:"1.12.7",
detail:"186.2 μs",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:190.084,julia:"1.12.7",
detail:"190.1 μs",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:578.5,julia:"1.12.7",
detail:"578.5 μs",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:582.375,julia:"1.12.7",
detail:"582.4 μs",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:581.917,julia:"1.12.7",
detail:"581.9 μs",allocs:0,mem:"0 B"}],
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
<td style="padding:7px 6px; text-align:right;">3.21 ms</td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">3.19 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">3.27 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">3.21 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.5% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">448 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 1D, Serial() backend (default)</code></td>
<td style="padding:7px 6px; text-align:right;">2.95 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">3.03 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">3.01 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+2.3% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 2D</code></td>
<td style="padding:7px 6px; text-align:right;">3.82 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">3.91 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">3.87 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+1.2% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">448 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 3D</code></td>
<td style="padding:7px 6px; text-align:right;">4.44 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">4.58 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">4.72 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+6.3% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">464 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">16.74 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">16.85 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">16.72 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">544 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 1D, Serial() backend (default)</code></td>
<td style="padding:7px 6px; text-align:right;">17.32 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">17.24 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">17.25 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.5% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 2D</code></td>
<td style="padding:7px 6px; text-align:right;">106.39 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">106.07 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">106.47 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">560 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 3D</code></td>
<td style="padding:7px 6px; text-align:right;">620.75 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">659.64 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">622.8 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">576 B</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="display:flex; flex-wrap:wrap; gap:1rem;">
  <div style="flex:1 1 400px; min-width:320px;"><div style="width:100%; max-width:460px;">
  <canvas id="bench_chart_5" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_5').getContext('2d'), {
    type: 'line',
    data: { labels: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"], datasets: [{
  label: "Rₕ 1D (allocates its output)",
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"3.2 ms (baseline)",allocs:10,mem:"7.64 MiB"},{x:"v2.1.0 (274ae7d)",y:1.0149558149254283,julia:"1.12.7",
detail:"3.24 ms (+1.5%)",allocs:10,mem:"7.64 MiB"},{x:"v2.2.0 (d7c2416)",y:1.0044658399467121,julia:"1.12.7",
detail:"3.21 ms (+0.4%)",allocs:10,mem:"7.64 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"3.19 ms (baseline)",allocs:7,mem:"448 B"},{x:"v2.1.0 (274ae7d)",y:1.024612615432775,julia:"1.12.7",
detail:"3.27 ms (+2.5%)",allocs:7,mem:"448 B"},{x:"v2.2.0 (d7c2416)",y:1.004656440757552,julia:"1.12.7",
detail:"3.21 ms (+0.5%)",allocs:7,mem:"448 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"2.95 ms (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.0301685765215176,julia:"1.12.7",
detail:"3.03 ms (+3.0%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:1.0233426704014938,julia:"1.12.7",
detail:"3.01 ms (+2.3%)",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"3.82 ms (baseline)",allocs:7,mem:"448 B"},{x:"v2.1.0 (274ae7d)",y:1.0244596787155744,julia:"1.12.7",
detail:"3.91 ms (+2.4%)",allocs:7,mem:"448 B"},{x:"v2.2.0 (d7c2416)",y:1.0118616496886683,julia:"1.12.7",
detail:"3.87 ms (+1.2%)",allocs:7,mem:"448 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"4.44 ms (baseline)",allocs:7,mem:"464 B"},{x:"v2.1.0 (274ae7d)",y:1.031481808717184,julia:"1.12.7",
detail:"4.58 ms (+3.1%)",allocs:7,mem:"464 B"},{x:"v2.2.0 (d7c2416)",y:1.0627620360730465,julia:"1.12.7",
detail:"4.72 ms (+6.3%)",allocs:7,mem:"464 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1},{x:"v2.1.0 (274ae7d)",y:1},{x:"v2.2.0 (d7c2416)",y:1}],
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
  <div style="flex:1 1 400px; min-width:320px;"><div style="width:100%; max-width:460px;">
  <canvas id="bench_chart_6" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_6').getContext('2d'), {
    type: 'line',
    data: { labels: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"], datasets: [{
  label: "avgₕ! 1D",
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"16.74 ms (baseline)",allocs:7,mem:"544 B"},{x:"v2.1.0 (274ae7d)",y:1.0068065935214991,julia:"1.12.7",
detail:"16.85 ms (+0.7%)",allocs:7,mem:"544 B"},{x:"v2.2.0 (d7c2416)",y:0.9989244822203467,julia:"1.12.7",
detail:"16.72 ms (-0.1%)",allocs:7,mem:"544 B"}],
  borderColor: "#3b82f6",
  backgroundColor: "#3b82f6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "avgₕ! 1D, Serial() backend (default)",
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"17.32 ms (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:0.994873716820243,julia:"1.12.7",
detail:"17.24 ms (-0.5%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:0.9954365216680587,julia:"1.12.7",
detail:"17.25 ms (-0.5%)",allocs:0,mem:"0 B"}],
  borderColor: "#10b981",
  backgroundColor: "#10b981",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "avgₕ! 2D",
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"106.39 ms (baseline)",allocs:7,mem:"560 B"},{x:"v2.1.0 (274ae7d)",y:0.996973378521717,julia:"1.12.7",
detail:"106.07 ms (-0.3%)",allocs:7,mem:"560 B"},{x:"v2.2.0 (d7c2416)",y:1.0008005106966111,julia:"1.12.7",
detail:"106.47 ms (+0.1%)",allocs:7,mem:"560 B"}],
  borderColor: "#f59e0b",
  backgroundColor: "#f59e0b",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "avgₕ! 3D",
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"620.75 ms (baseline)",allocs:7,mem:"576 B"},{x:"v2.1.0 (274ae7d)",y:1.0626395001054676,julia:"1.12.7",
detail:"659.64 ms (+6.3%)",allocs:7,mem:"576 B"},{x:"v2.2.0 (d7c2416)",y:1.0033005530130232,julia:"1.12.7",
detail:"622.8 ms (+0.3%)",allocs:7,mem:"576 B"}],
  borderColor: "#8b5cf6",
  backgroundColor: "#8b5cf6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "1.0x (ref)",
  data: [{x:"v2.0.0 (2dec0c7)",y:1},{x:"v2.1.0 (274ae7d)",y:1},{x:"v2.2.0 (d7c2416)",y:1}],
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
<td style="padding:7px 6px; text-align:right;">742.8 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+10.8% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">22.89 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>∇₋ₕ (3 components)</code></td>
<td style="padding:7px 6px; text-align:right;">1.38 ms</td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:right;">1.41 ms</td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:right;">1.44 ms</td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+4.1% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">45.78 MiB</td>
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
    data: { labels: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"], datasets: [{
  label: "D₋ₓ (3 components)",
  data: [{x:"v2.0.0 (2dec0c7)",y:0.670208,julia:"1.12.7",
detail:"670.2 μs",allocs:3,mem:"22.89 MiB"},{x:"v2.1.0 (274ae7d)",y:0.693958,julia:"1.12.7",
detail:"694.0 μs",allocs:3,mem:"22.89 MiB"},{x:"v2.2.0 (d7c2416)",y:0.7428125,julia:"1.12.7",
detail:"742.8 μs",allocs:3,mem:"22.89 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.380625,julia:"1.12.7",
detail:"1.38 ms",allocs:10,mem:"45.78 MiB"},{x:"v2.1.0 (274ae7d)",y:1.407666,julia:"1.12.7",
detail:"1.41 ms",allocs:10,mem:"45.78 MiB"},{x:"v2.2.0 (d7c2416)",y:1.436584,julia:"1.12.7",
detail:"1.44 ms",allocs:10,mem:"45.78 MiB"}],
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
<td style="padding:7px 6px; text-align:right;">2.24 ms</td>
<td style="padding:7px 6px; text-align:center;">42</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+1.0% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">22.95 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>gridspace 3D</code></td>
<td style="padding:7px 6px; text-align:right;">6.2 ms</td>
<td style="padding:7px 6px; text-align:center;">52</td>
<td style="padding:7px 6px; text-align:right;">6.28 ms</td>
<td style="padding:7px 6px; text-align:center;">52</td>
<td style="padding:7px 6px; text-align:right;">6.22 ms</td>
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
<td style="padding:7px 6px; text-align:right;">154.4 ns</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.9% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
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
    data: { labels: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"], datasets: [{
  label: "gridspace 2D",
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"2.22 ms (baseline)",allocs:42,mem:"22.95 MiB"},{x:"v2.1.0 (274ae7d)",y:1.0100427623813584,julia:"1.12.7",
detail:"2.24 ms (+1.0%)",allocs:42,mem:"22.95 MiB"},{x:"v2.2.0 (d7c2416)",y:1.0095548603690214,julia:"1.12.7",
detail:"2.24 ms (+1.0%)",allocs:42,mem:"22.95 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"6.2 ms (baseline)",allocs:52,mem:"30.57 MiB"},{x:"v2.1.0 (274ae7d)",y:1.0131330029173518,julia:"1.12.7",
detail:"6.28 ms (+1.3%)",allocs:52,mem:"30.57 MiB"},{x:"v2.2.0 (d7c2416)",y:1.0028900913595995,julia:"1.12.7",
detail:"6.22 ms (+0.3%)",allocs:52,mem:"30.57 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"153.0 ns (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.0106493855220369,julia:"1.12.7",
detail:"154.6 ns (+1.1%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:1.0089876062952494,julia:"1.12.7",
detail:"154.4 ns (+0.9%)",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1},{x:"v2.1.0 (274ae7d)",y:1},{x:"v2.2.0 (d7c2416)",y:1}],
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
<td style="padding:7px 6px; text-align:right;">454.61 ms</td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-24.1% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">1.3 KiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>using Bramble</code></td>
<td style="padding:7px 6px; text-align:right;">501.71 ms</td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:right;">487.28 ms</td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:right;">400.0 ms</td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-20.3% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">1.3 KiB</td>
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
    data: { labels: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"], datasets: [{
  label: "TTFX (load + first operator)",
  data: [{x:"v2.0.0 (2dec0c7)",y:598.631542,julia:"1.12.7",
detail:"598.63 ms",allocs:45,mem:"1.3 KiB"},{x:"v2.1.0 (274ae7d)",y:654.503208,julia:"1.12.7",
detail:"654.5 ms",allocs:45,mem:"1.3 KiB"},{x:"v2.2.0 (d7c2416)",y:454.605959,julia:"1.12.7",
detail:"454.61 ms",allocs:45,mem:"1.3 KiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:501.70775,julia:"1.12.7",
detail:"501.71 ms",allocs:45,mem:"1.3 KiB"},{x:"v2.1.0 (274ae7d)",y:487.277792,julia:"1.12.7",
detail:"487.28 ms",allocs:45,mem:"1.3 KiB"},{x:"v2.2.0 (d7c2416)",y:399.997458,julia:"1.12.7",
detail:"400.0 ms",allocs:45,mem:"1.3 KiB"}],
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
<td style="padding:7px 6px; text-align:right;">2.86 ms</td>
<td style="padding:7px 6px; text-align:center;">21</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+1.0% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">15.13 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble (BilinearForm), Parallel() backend</code></td>
<td style="padding:7px 6px; text-align:right;">5.11 ms</td>
<td style="padding:7px 6px; text-align:center;">35</td>
<td style="padding:7px 6px; text-align:right;">4.8 ms</td>
<td style="padding:7px 6px; text-align:center;">35</td>
<td style="padding:7px 6px; text-align:right;">4.78 ms</td>
<td style="padding:7px 6px; text-align:center;">35</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-6.4% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">15.13 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble (BilinearForm), Serial() backend</code></td>
<td style="padding:7px 6px; text-align:right;">4.71 ms</td>
<td style="padding:7px 6px; text-align:center;">21</td>
<td style="padding:7px 6px; text-align:right;">4.83 ms</td>
<td style="padding:7px 6px; text-align:center;">21</td>
<td style="padding:7px 6px; text-align:right;">4.73 ms</td>
<td style="padding:7px 6px; text-align:center;">21</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">15.13 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! (matrix) 2D</code></td>
<td style="padding:7px 6px; text-align:right;">1.07 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.06 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.04 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-2.5% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">938.8 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">953.0 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">937.2 μs</td>
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
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+1.6% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">480 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble! 2D</code></td>
<td style="padding:7px 6px; text-align:right;">1.18 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.21 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">481.7 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-59.4% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>assemble_parallel! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">1.29 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">1.22 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:right;">1.35 ms</td>
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
<td style="padding:7px 6px; text-align:right;">1.73 ms</td>
<td style="padding:7px 6px; text-align:center;">7</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.7% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">496 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>evaluate! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">1.14 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.15 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">1.13 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-0.5% 🟢</span></td>
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
<td style="padding:7px 6px; text-align:right;">884.3 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="display:flex; flex-wrap:wrap; gap:1rem;">
  <div style="flex:1 1 400px; min-width:320px;"><div style="width:100%; max-width:460px;">
  <canvas id="bench_chart_10" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_10').getContext('2d'), {
    type: 'line',
    data: { labels: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"], datasets: [{
  label: "allocate_system_matrix 2D",
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"2.84 ms (baseline)",allocs:21,mem:"15.13 MiB"},{x:"v2.1.0 (274ae7d)",y:1.1201502186338541,julia:"1.12.7",
detail:"3.18 ms (+12.0%)",allocs:21,mem:"15.13 MiB"},{x:"v2.2.0 (d7c2416)",y:1.0102132588259443,julia:"1.12.7",
detail:"2.86 ms (+1.0%)",allocs:21,mem:"15.13 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"5.11 ms (baseline)",allocs:35,mem:"15.13 MiB"},{x:"v2.1.0 (274ae7d)",y:0.9389424435152599,julia:"1.12.7",
detail:"4.8 ms (-6.1%)",allocs:35,mem:"15.13 MiB"},{x:"v2.2.0 (d7c2416)",y:0.9359668749114706,julia:"1.12.7",
detail:"4.78 ms (-6.4%)",allocs:35,mem:"15.13 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"4.71 ms (baseline)",allocs:21,mem:"15.13 MiB"},{x:"v2.1.0 (274ae7d)",y:1.0252965369306104,julia:"1.12.7",
detail:"4.83 ms (+2.5%)",allocs:21,mem:"15.13 MiB"},{x:"v2.2.0 (d7c2416)",y:1.003261291058878,julia:"1.12.7",
detail:"4.73 ms (+0.3%)",allocs:21,mem:"15.13 MiB"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"1.07 ms (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:0.9931328213306763,julia:"1.12.7",
detail:"1.06 ms (-0.7%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:0.9747938113918562,julia:"1.12.7",
detail:"1.04 ms (-2.5%)",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"938.8 μs (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.0150463394448215,julia:"1.12.7",
detail:"953.0 μs (+1.5%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:0.998225456497588,julia:"1.12.7",
detail:"937.2 μs (-0.2%)",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"1.19 ms (baseline)",allocs:7,mem:"480 B"},{x:"v2.1.0 (274ae7d)",y:1.0114479998458745,julia:"1.12.7",
detail:"1.21 ms (+1.1%)",allocs:7,mem:"480 B"},{x:"v2.2.0 (d7c2416)",y:1.0161245333308762,julia:"1.12.7",
detail:"1.21 ms (+1.6%)",allocs:7,mem:"480 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"1.18 ms (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.0225387640157861,julia:"1.12.7",
detail:"1.21 ms (+2.3%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:0.40648343380801943,julia:"1.12.7",
detail:"481.7 μs (-59.4%)",allocs:0,mem:"0 B"}],
  borderColor: "#f97316",
  backgroundColor: "#f97316",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "1.0x (ref)",
  data: [{x:"v2.0.0 (2dec0c7)",y:1},{x:"v2.1.0 (274ae7d)",y:1},{x:"v2.2.0 (d7c2416)",y:1}],
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
  <div style="flex:1 1 400px; min-width:320px;"><div style="width:100%; max-width:460px;">
  <canvas id="bench_chart_11" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_11').getContext('2d'), {
    type: 'line',
    data: { labels: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"], datasets: [{
  label: "assemble_parallel! 1D",
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"1.29 ms (baseline)",allocs:7,mem:"480 B"},{x:"v2.1.0 (274ae7d)",y:0.9451508406118697,julia:"1.12.7",
detail:"1.22 ms (-5.5%)",allocs:7,mem:"480 B"},{x:"v2.2.0 (d7c2416)",y:1.0422099581069324,julia:"1.12.7",
detail:"1.35 ms (+4.2%)",allocs:7,mem:"480 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"1.72 ms (baseline)",allocs:7,mem:"496 B"},{x:"v2.1.0 (274ae7d)",y:1.302873613909273,julia:"1.12.7",
detail:"2.24 ms (+30.3%)",allocs:7,mem:"496 B"},{x:"v2.2.0 (d7c2416)",y:1.0066105025746,julia:"1.12.7",
detail:"1.73 ms (+0.7%)",allocs:7,mem:"496 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"1.14 ms (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.0108406550714244,julia:"1.12.7",
detail:"1.15 ms (+1.1%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:0.994543415357895,julia:"1.12.7",
detail:"1.13 ms (-0.5%)",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"2.1 ns (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.0,julia:"1.12.7",
detail:"2.1 ns (baseline)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:1.0,julia:"1.12.7",
detail:"2.1 ns (baseline)",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"2.1 ns (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.0,julia:"1.12.7",
detail:"2.1 ns (baseline)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:1.0,julia:"1.12.7",
detail:"2.1 ns (baseline)",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"883.6 μs (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.0075918079096045,julia:"1.12.7",
detail:"890.3 μs (+0.8%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:1.0008476839779805,julia:"1.12.7",
detail:"884.3 μs (+0.1%)",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1},{x:"v2.1.0 (274ae7d)",y:1},{x:"v2.2.0 (d7c2416)",y:1}],
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
<td style="padding:7px 6px; text-align:right;">9.0 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.7% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! Float32</code></td>
<td style="padding:7px 6px; text-align:right;">286.0 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">285.7 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">285.9 μs</td>
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
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-2.3% 🟢</span></td>
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
<td style="padding:7px 6px; text-align:right;">84.2 μs</td>
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
<td style="padding:7px 6px; text-align:right;">72.86 ms</td>
<td style="padding:7px 6px; text-align:center;">33</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.9% 🔴</span></td>
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
<td style="padding:7px 6px; text-align:right;">1.73 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.9% 🔴</span></td>
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
  <div style="flex:1 1 400px; min-width:320px;"><div style="width:100%; max-width:460px;">
  <canvas id="bench_chart_12" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_12').getContext('2d'), {
    type: 'line',
    data: { labels: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"], datasets: [{
  label: "Rₕ! Double64",
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"8.94 ms (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.0108921227371217,julia:"1.12.7",
detail:"9.04 ms (+1.1%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:1.0069476480044732,julia:"1.12.7",
detail:"9.0 ms (+0.7%)",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"286.0 μs (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:0.9989825174825174,julia:"1.12.7",
detail:"285.7 μs (-0.1%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:0.9995629370629371,julia:"1.12.7",
detail:"285.9 μs (-0.0%)",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"293.5 μs (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.0007085868461754,julia:"1.12.7",
detail:"293.8 μs (+0.1%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:1.0,julia:"1.12.7",
detail:"293.5 μs (baseline)",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"1.04 ms (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:0.9757634899651484,julia:"1.12.7",
detail:"1.01 ms (-2.4%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:0.9769258502583824,julia:"1.12.7",
detail:"1.02 ms (-2.3%)",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"71.3 μs (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.119748223122538,julia:"1.12.7",
detail:"79.9 μs (+12.0%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:1.0029299202332722,julia:"1.12.7",
detail:"71.5 μs (+0.3%)",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"84.0 μs (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.0044665189737727,julia:"1.12.7",
detail:"84.3 μs (+0.4%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:1.0034779294409109,julia:"1.12.7",
detail:"84.2 μs (+0.3%)",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1},{x:"v2.1.0 (274ae7d)",y:1},{x:"v2.2.0 (d7c2416)",y:1}],
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
  <div style="flex:1 1 400px; min-width:320px;"><div style="width:100%; max-width:460px;">
  <canvas id="bench_chart_13" height="280"></canvas>
</div>
<script>
(function () {
  const theme = window.brambleChartTheme();
  const chart = new Chart(document.getElementById('bench_chart_13').getContext('2d'), {
    type: 'line',
    data: { labels: ["v2.0.0 (2dec0c7)","v2.1.0 (274ae7d)","v2.2.0 (d7c2416)"], datasets: [{
  label: "avgₕ! Double64",
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"72.2 ms (baseline)",allocs:33,mem:"2.9 KiB"},{x:"v2.1.0 (274ae7d)",y:1.0152270976688864,julia:"1.12.7",
detail:"73.3 ms (+1.5%)",allocs:33,mem:"2.9 KiB"},{x:"v2.2.0 (d7c2416)",y:1.0091008804543633,julia:"1.12.7",
detail:"72.86 ms (+0.9%)",allocs:33,mem:"2.9 KiB"}],
  borderColor: "#3b82f6",
  backgroundColor: "#3b82f6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "avgₕ! Float32",
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"1.61 ms (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.0091040787194028,julia:"1.12.7",
detail:"1.62 ms (+0.9%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:1.0000385977079187,julia:"1.12.7",
detail:"1.61 ms (+0.0%)",allocs:0,mem:"0 B"}],
  borderColor: "#10b981",
  backgroundColor: "#10b981",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "avgₕ! Float64",
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"1.72 ms (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.039595387843113,julia:"1.12.7",
detail:"1.78 ms (+4.0%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:1.009258795447953,julia:"1.12.7",
detail:"1.73 ms (+0.9%)",allocs:0,mem:"0 B"}],
  borderColor: "#f59e0b",
  backgroundColor: "#f59e0b",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "innerₕ Double64",
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"1.06 ms (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.0,julia:"1.12.7",
detail:"1.06 ms (baseline)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:0.999412041392286,julia:"1.12.7",
detail:"1.06 ms (-0.1%)",allocs:0,mem:"0 B"}],
  borderColor: "#8b5cf6",
  backgroundColor: "#8b5cf6",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "innerₕ Float32",
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"11.6 μs (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.0,julia:"1.12.7",
detail:"11.6 μs (baseline)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:1.003539364640884,julia:"1.12.7",
detail:"11.6 μs (+0.4%)",allocs:0,mem:"0 B"}],
  borderColor: "#ec4899",
  backgroundColor: "#ec4899",
  spanGaps: true,
  pointRadius: 4,
  pointHoverRadius: 6,
  borderWidth: 2,
  tension: 0.15,
},
{
  label: "innerₕ Float64",
  data: [{x:"v2.0.0 (2dec0c7)",y:1.0,julia:"1.12.7",
detail:"23.2 μs (baseline)",allocs:0,mem:"0 B"},{x:"v2.1.0 (274ae7d)",y:1.0018129235550568,julia:"1.12.7",
detail:"23.2 μs (+0.2%)",allocs:0,mem:"0 B"},{x:"v2.2.0 (d7c2416)",y:1.0,julia:"1.12.7",
detail:"23.2 μs (baseline)",allocs:0,mem:"0 B"}],
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
  data: [{x:"v2.0.0 (2dec0c7)",y:1},{x:"v2.1.0 (274ae7d)",y:1},{x:"v2.2.0 (d7c2416)",y:1}],
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

  </div>
</div>
```

## How to add new benchmark runs

To record performance on a new commit or after an optimization pass, run:

```bash
julia --project=benchmark benchmark/benchmarks.jl --save benchmark/baselines/baseline_$(git rev-parse --short HEAD).json
```

Rebuilding the documentation (`julia -e 'using Pkg; Pkg.activate("docs"); include("docs/make.jl")'`) will automatically discover all `baseline_*.json` files and append new comparison columns, delta calculations, and charts.
