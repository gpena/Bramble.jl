# Performance & Benchmarks

Bramble tracks memory allocations and performance regressions with a dedicated regression suite in `benchmark/benchmarks.jl`.
All measurements below are run on **1,000,000 grid points** per dimension setup (e.g. ``1000 \times 1000`` in 2D, ``100 \times 100 \times 100`` in 3D).

## Recorded Baselines

Comparing **4** recorded baselines in chronological order. The earliest run (`0b9a62b`) serves as reference baseline for relative speedup/slowdown calculations.

| Commit | Julia | Summary | File |
|---|:---:|---|---|
| `0b9a62b` *(baseline)* | `1.12.7` | test: run the allocation assertions under coverage instead of skipping them | `baseline_0b9a62b.json` |
| `855fbf5` | `1.12.7` | docs(benchmarks): switch to inline SVG charts and streamline baselines table | `baseline_855fbf5.json` |
| `41036bb` | `1.12.7` | fix(space): only fetch the Gauss rule inside the kernel where it truly folds | `baseline_41036bb.json` |
| `ddb8d78` | `1.12.7` | feat(form): index a form's arguments by component | `baseline_ddb8d78.json` |

## Comparative Timings & Allocations

### Operators 2D

```@raw html
<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;">
  <div style="flex:1 1 430px; min-width:320px; overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:8px 6px; text-align:left;">Benchmark</th>
<th style="padding:8px 6px; text-align:right;">Base (<code>0b9a62b</code>)</th>
<th style="padding:8px 6px; text-align:right;">Prev (<code>41036bb</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>ddb8d78</code>)</th>
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
<td style="padding:7px 6px; text-align:right;">255.0 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">315.2 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+22.5% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+23.6% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋ᵧ</code></td>
<td style="padding:7px 6px; text-align:right;">161.4 μs</td>
<td style="padding:7px 6px; text-align:right;">162.0 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">135.5 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-16.1% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-16.4% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋ₓ</code></td>
<td style="padding:7px 6px; text-align:right;">203.7 μs</td>
<td style="padding:7px 6px; text-align:right;">203.7 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">159.5 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-21.7% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-21.7% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₋ₓ</code></td>
<td style="padding:7px 6px; text-align:right;">171.4 μs</td>
<td style="padding:7px 6px; text-align:right;">168.2 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">144.0 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-16.0% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-14.4% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:540px; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:0.8em; box-sizing:border-box;">
<svg viewBox="0 0 540 285" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<line x1="65" y1="16" x2="79" y2="16" stroke="#3b82f6" stroke-width="2.5" />
<circle cx="72" cy="16" r="3.5" fill="#3b82f6" />
<text x="83" y="20" font-size="11" font-weight="bold" fill="currentColor">Dcₓ</text>
<line x1="124" y1="16" x2="138" y2="16" stroke="#10b981" stroke-width="2.5" />
<circle cx="131" cy="16" r="3.5" fill="#10b981" />
<text x="142" y="20" font-size="11" font-weight="bold" fill="currentColor">D₋ᵧ</text>
<line x1="183" y1="16" x2="197" y2="16" stroke="#f59e0b" stroke-width="2.5" />
<circle cx="190" cy="16" r="3.5" fill="#f59e0b" />
<text x="201" y="20" font-size="11" font-weight="bold" fill="currentColor">D₋ₓ</text>
<line x1="242" y1="16" x2="256" y2="16" stroke="#8b5cf6" stroke-width="2.5" />
<circle cx="249" cy="16" r="3.5" fill="#8b5cf6" />
<text x="260" y="20" font-size="11" font-weight="bold" fill="currentColor">M₋ₓ</text>
<line x1="65" y1="240.0" x2="515" y2="240.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="244.0" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">0.0 μs</text>
<line x1="65" y1="191.25" x2="515" y2="191.25" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="195.25" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">125.0 μs</text>
<line x1="65" y1="142.5" x2="515" y2="142.5" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="146.5" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">250.0 μs</text>
<line x1="65" y1="93.75" x2="515" y2="93.75" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="97.75" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">375.0 μs</text>
<line x1="65" y1="45.0" x2="515" y2="45.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="49.0" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">500.0 μs</text>
<line x1="65.0" y1="45" x2="65.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="65.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`0b9a62b`</text>
<line x1="215.0" y1="45" x2="215.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="215.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="365.0" y1="45" x2="365.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="365.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`41036bb`</text>
<line x1="515.0" y1="45" x2="515.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`ddb8d78`</text>
<polyline points="65.0,139.7 215.0,139.9 365.0,140.5 515.0,117.1" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="139.7" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
Dcₓ: 257.2 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="65.0" y="132.7" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">257.2</text>
<circle cx="215.0" cy="139.9" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
Dcₓ: 256.5 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="215.0" y="132.9" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">256.5</text>
<circle cx="365.0" cy="140.5" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
Dcₓ: 255.0 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="365.0" y="133.5" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">255.0</text>
<circle cx="515.0" cy="117.1" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
Dcₓ: 315.2 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="110.1" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">315.2</text>
<polyline points="65.0,177.0 215.0,176.6 365.0,176.8 515.0,187.2" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="177.0" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
D₋ᵧ: 161.4 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="65.0" y="170.0" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">161.4</text>
<circle cx="215.0" cy="176.6" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
D₋ᵧ: 162.6 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="215.0" y="169.6" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">162.6</text>
<circle cx="365.0" cy="176.8" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
D₋ᵧ: 162.0 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="365.0" y="169.8" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">162.0</text>
<circle cx="515.0" cy="187.2" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
D₋ᵧ: 135.5 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="180.2" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">135.5</text>
<polyline points="65.0,160.6 215.0,160.7 365.0,160.6 515.0,177.8" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="160.6" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
D₋ₓ: 203.7 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="65.0" y="153.6" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">203.7</text>
<circle cx="215.0" cy="160.7" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
D₋ₓ: 203.3 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="215.0" y="153.7" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">203.3</text>
<circle cx="365.0" cy="160.6" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
D₋ₓ: 203.7 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="365.0" y="153.6" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">203.7</text>
<circle cx="515.0" cy="177.8" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
D₋ₓ: 159.5 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="170.8" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">159.5</text>
<polyline points="65.0,173.1 215.0,173.3 365.0,174.4 515.0,183.9" fill="none" stroke="#8b5cf6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="173.1" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
M₋ₓ: 171.4 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="65.0" y="166.1" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">171.4</text>
<circle cx="215.0" cy="173.3" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
M₋ₓ: 171.0 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="215.0" y="166.3" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">171.0</text>
<circle cx="365.0" cy="174.4" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
M₋ₓ: 168.2 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="365.0" y="167.4" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">168.2</text>
<circle cx="515.0" cy="183.9" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
M₋ₓ: 144.0 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="176.9" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">144.0</text>
</svg></div>

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
<th style="padding:8px 6px; text-align:right;">Prev (<code>41036bb</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>ddb8d78</code>)</th>
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
<td style="padding:7px 6px; text-align:right;">228.1 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">190.4 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-5.2% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-16.5% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>innerₕ</code></td>
<td style="padding:7px 6px; text-align:right;">240.2 μs</td>
<td style="padding:7px 6px; text-align:right;">239.5 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">170.8 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-28.9% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-28.7% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>∇₋ₕ</code></td>
<td style="padding:7px 6px; text-align:right;">694.1 μs</td>
<td style="padding:7px 6px; text-align:right;">685.7 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">587.1 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-15.4% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-14.4% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">15</td>
<td style="padding:7px 6px; text-align:right;">22.92 MiB</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:540px; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:0.8em; box-sizing:border-box;">
<svg viewBox="0 0 540 285" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<line x1="65" y1="16" x2="79" y2="16" stroke="#3b82f6" stroke-width="2.5" />
<circle cx="72" cy="16" r="3.5" fill="#3b82f6" />
<text x="83" y="20" font-size="11" font-weight="bold" fill="currentColor">D₋₂</text>
<line x1="124" y1="16" x2="138" y2="16" stroke="#10b981" stroke-width="2.5" />
<circle cx="131" cy="16" r="3.5" fill="#10b981" />
<text x="142" y="20" font-size="11" font-weight="bold" fill="currentColor">innerₕ</text>
<line x1="204" y1="16" x2="218" y2="16" stroke="#f59e0b" stroke-width="2.5" />
<circle cx="211" cy="16" r="3.5" fill="#f59e0b" />
<text x="222" y="20" font-size="11" font-weight="bold" fill="currentColor">∇₋ₕ</text>
<line x1="65" y1="240.0" x2="515" y2="240.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="244.0" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">0.0 μs</text>
<line x1="65" y1="191.25" x2="515" y2="191.25" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="195.25" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">250.0 μs</text>
<line x1="65" y1="142.5" x2="515" y2="142.5" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="146.5" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">500.0 μs</text>
<line x1="65" y1="93.75" x2="515" y2="93.75" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="97.75" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">750.0 μs</text>
<line x1="65" y1="45.0" x2="515" y2="45.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="49.0" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">1000.0 μs</text>
<line x1="65.0" y1="45" x2="65.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="65.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`0b9a62b`</text>
<line x1="215.0" y1="45" x2="215.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="215.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="365.0" y1="45" x2="365.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="365.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`41036bb`</text>
<line x1="515.0" y1="45" x2="515.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`ddb8d78`</text>
<polyline points="65.0,200.8 215.0,196.6 365.0,195.5 515.0,202.9" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="200.8" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
D₋₂: 200.9 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="65.0" y="193.8" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">200.9</text>
<circle cx="215.0" cy="196.6" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
D₋₂: 222.8 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="215.0" y="189.6" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">222.8</text>
<circle cx="365.0" cy="195.5" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
D₋₂: 228.1 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="365.0" y="188.5" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">228.1</text>
<circle cx="515.0" cy="202.9" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
D₋₂: 190.4 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="195.9" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">190.4</text>
<polyline points="65.0,193.2 215.0,193.2 365.0,193.3 515.0,206.7" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="193.2" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
innerₕ: 240.2 μs (0 allocs, 0 B)</title></circle>
<text x="65.0" y="186.2" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">240.2</text>
<circle cx="215.0" cy="193.2" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
innerₕ: 240.0 μs (0 allocs, 0 B)</title></circle>
<text x="215.0" y="186.2" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">240.0</text>
<circle cx="365.0" cy="193.3" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
innerₕ: 239.5 μs (0 allocs, 0 B)</title></circle>
<text x="365.0" y="186.3" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">239.5</text>
<circle cx="515.0" cy="206.7" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
innerₕ: 170.8 μs (0 allocs, 0 B)</title></circle>
<text x="515.0" y="199.7" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">170.8</text>
<polyline points="65.0,104.7 215.0,106.1 365.0,106.3 515.0,125.5" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="104.7" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
∇₋ₕ: 694.1 μs (15 allocs, 22.92 MiB)</title></circle>
<text x="65.0" y="97.7" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">694.1</text>
<circle cx="215.0" cy="106.1" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
∇₋ₕ: 686.6 μs (15 allocs, 22.92 MiB)</title></circle>
<text x="215.0" y="99.1" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">686.6</text>
<circle cx="365.0" cy="106.3" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
∇₋ₕ: 685.7 μs (15 allocs, 22.92 MiB)</title></circle>
<text x="365.0" y="99.3" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">685.7</text>
<circle cx="515.0" cy="125.5" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
∇₋ₕ: 587.1 μs (15 allocs, 22.92 MiB)</title></circle>
<text x="515.0" y="118.5" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">587.1</text>
</svg></div>

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
<th style="padding:8px 6px; text-align:right;">Prev (<code>41036bb</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>ddb8d78</code>)</th>
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
<td style="padding:7px 6px; text-align:right;">165.1 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">133.7 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-19.1% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₊₂ 3D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">223.8 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">189.7 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-15.3% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₊ₓ 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">162.5 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">135.5 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-16.6% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>jumpᵧ 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">160.8 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">133.5 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-17.0% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>jump₂ 3D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">227.6 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">192.5 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-15.4% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>jumpₓ 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">162.5 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">135.3 μs</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-16.7% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:540px; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:0.8em; box-sizing:border-box;">
<svg viewBox="0 0 540 305" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<line x1="65" y1="16" x2="79" y2="16" stroke="#3b82f6" stroke-width="2.5" />
<circle cx="72" cy="16" r="3.5" fill="#3b82f6" />
<text x="83" y="20" font-size="11" font-weight="bold" fill="currentColor">M₊ᵧ 2D</text>
<line x1="145" y1="16" x2="159" y2="16" stroke="#10b981" stroke-width="2.5" />
<circle cx="152" cy="16" r="3.5" fill="#10b981" />
<text x="163" y="20" font-size="11" font-weight="bold" fill="currentColor">M₊₂ 3D</text>
<line x1="225" y1="16" x2="239" y2="16" stroke="#f59e0b" stroke-width="2.5" />
<circle cx="232" cy="16" r="3.5" fill="#f59e0b" />
<text x="243" y="20" font-size="11" font-weight="bold" fill="currentColor">M₊ₓ 2D</text>
<line x1="305" y1="16" x2="319" y2="16" stroke="#8b5cf6" stroke-width="2.5" />
<circle cx="312" cy="16" r="3.5" fill="#8b5cf6" />
<text x="323" y="20" font-size="11" font-weight="bold" fill="currentColor">jumpᵧ 2D</text>
<line x1="399" y1="16" x2="413" y2="16" stroke="#ec4899" stroke-width="2.5" />
<circle cx="406" cy="16" r="3.5" fill="#ec4899" />
<text x="417" y="20" font-size="11" font-weight="bold" fill="currentColor">jump₂ 3D</text>
<line x1="65" y1="34" x2="79" y2="34" stroke="#06b6d4" stroke-width="2.5" />
<circle cx="72" cy="34" r="3.5" fill="#06b6d4" />
<text x="83" y="38" font-size="11" font-weight="bold" fill="currentColor">jumpₓ 2D</text>
<line x1="65" y1="260.0" x2="515" y2="260.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="264.0" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">0.0 μs</text>
<line x1="65" y1="211.25" x2="515" y2="211.25" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="215.25" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">75.0 μs</text>
<line x1="65" y1="162.5" x2="515" y2="162.5" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="166.5" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">150.0 μs</text>
<line x1="65" y1="113.75" x2="515" y2="113.75" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="117.75" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">225.0 μs</text>
<line x1="65" y1="65.0" x2="515" y2="65.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="69.0" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">300.0 μs</text>
<line x1="65.0" y1="65" x2="65.0" y2="260" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="65.0" y="280" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`0b9a62b`</text>
<line x1="215.0" y1="65" x2="215.0" y2="260" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="215.0" y="280" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="365.0" y1="65" x2="365.0" y2="260" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="365.0" y="280" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`41036bb`</text>
<line x1="515.0" y1="65" x2="515.0" y2="260" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="280" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`ddb8d78`</text>
<polyline points="215.0,154.9 365.0,152.7 515.0,173.1" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="215.0" cy="154.9" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
M₊ᵧ 2D: 161.6 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="215.0" y="147.9" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">161.6</text>
<circle cx="365.0" cy="152.7" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
M₊ᵧ 2D: 165.1 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="365.0" y="145.7" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">165.1</text>
<circle cx="515.0" cy="173.1" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
M₊ᵧ 2D: 133.7 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="166.1" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">133.7</text>
<polyline points="215.0,111.9 365.0,114.5 515.0,136.7" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="215.0" cy="111.9" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
M₊₂ 3D: 227.8 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="215.0" y="104.9" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">227.8</text>
<circle cx="365.0" cy="114.5" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
M₊₂ 3D: 223.8 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="365.0" y="107.5" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">223.8</text>
<circle cx="515.0" cy="136.7" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
M₊₂ 3D: 189.7 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="129.7" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">189.7</text>
<polyline points="215.0,155.7 365.0,154.4 515.0,171.9" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="215.0" cy="155.7" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
M₊ₓ 2D: 160.4 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="215.0" y="148.7" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">160.4</text>
<circle cx="365.0" cy="154.4" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
M₊ₓ 2D: 162.5 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="365.0" y="147.4" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">162.5</text>
<circle cx="515.0" cy="171.9" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
M₊ₓ 2D: 135.5 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="164.9" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">135.5</text>
<polyline points="215.0,154.7 365.0,155.5 515.0,173.3" fill="none" stroke="#8b5cf6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="215.0" cy="154.7" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
jumpᵧ 2D: 162.0 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="215.0" y="147.7" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">162.0</text>
<circle cx="365.0" cy="155.5" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
jumpᵧ 2D: 160.8 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="365.0" y="148.5" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">160.8</text>
<circle cx="515.0" cy="173.3" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
jumpᵧ 2D: 133.5 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="166.3" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">133.5</text>
<polyline points="215.0,112.0 365.0,112.0 515.0,134.9" fill="none" stroke="#ec4899" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="215.0" cy="112.0" r="4.5" fill="#ec4899" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
jump₂ 3D: 227.6 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="215.0" y="105.0" font-size="10" font-weight="bold" fill="#ec4899" text-anchor="middle">227.6</text>
<circle cx="365.0" cy="112.0" r="4.5" fill="#ec4899" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
jump₂ 3D: 227.6 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="365.0" y="105.0" font-size="10" font-weight="bold" fill="#ec4899" text-anchor="middle">227.6</text>
<circle cx="515.0" cy="134.9" r="4.5" fill="#ec4899" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
jump₂ 3D: 192.5 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="127.9" font-size="10" font-weight="bold" fill="#ec4899" text-anchor="middle">192.5</text>
<polyline points="215.0,153.0 365.0,154.4 515.0,172.0" fill="none" stroke="#06b6d4" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="215.0" cy="153.0" r="4.5" fill="#06b6d4" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
jumpₓ 2D: 164.7 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="215.0" y="146.0" font-size="10" font-weight="bold" fill="#06b6d4" text-anchor="middle">164.7</text>
<circle cx="365.0" cy="154.4" r="4.5" fill="#06b6d4" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
jumpₓ 2D: 162.5 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="365.0" y="147.4" font-size="10" font-weight="bold" fill="#06b6d4" text-anchor="middle">162.5</text>
<circle cx="515.0" cy="172.0" r="4.5" fill="#06b6d4" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
jumpₓ 2D: 135.3 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="165.0" font-size="10" font-weight="bold" fill="#06b6d4" text-anchor="middle">135.3</text>
</svg></div>

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
<th style="padding:8px 6px; text-align:right;">Prev (<code>41036bb</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>ddb8d78</code>)</th>
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
<td style="padding:7px 6px; text-align:right;">240.0 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">170.9 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-29.4% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-28.8% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>norm₁ₕ</code></td>
<td style="padding:7px 6px; text-align:right;">790.2 μs</td>
<td style="padding:7px 6px; text-align:right;">796.6 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">530.3 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-32.9% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-33.4% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>normₕ</code></td>
<td style="padding:7px 6px; text-align:right;">190.0 μs</td>
<td style="padding:7px 6px; text-align:right;">189.4 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">136.7 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-28.1% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-27.8% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>snorm₁ₕ</code></td>
<td style="padding:7px 6px; text-align:right;">578.1 μs</td>
<td style="padding:7px 6px; text-align:right;">582.4 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">394.1 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-31.8% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-32.3% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:540px; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:0.8em; box-sizing:border-box;">
<svg viewBox="0 0 540 285" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<line x1="65" y1="16" x2="79" y2="16" stroke="#3b82f6" stroke-width="2.5" />
<circle cx="72" cy="16" r="3.5" fill="#3b82f6" />
<text x="83" y="20" font-size="11" font-weight="bold" fill="currentColor">innerₕ</text>
<line x1="145" y1="16" x2="159" y2="16" stroke="#10b981" stroke-width="2.5" />
<circle cx="152" cy="16" r="3.5" fill="#10b981" />
<text x="163" y="20" font-size="11" font-weight="bold" fill="currentColor">norm₁ₕ</text>
<line x1="225" y1="16" x2="239" y2="16" stroke="#f59e0b" stroke-width="2.5" />
<circle cx="232" cy="16" r="3.5" fill="#f59e0b" />
<text x="243" y="20" font-size="11" font-weight="bold" fill="currentColor">normₕ</text>
<line x1="298" y1="16" x2="312" y2="16" stroke="#8b5cf6" stroke-width="2.5" />
<circle cx="305" cy="16" r="3.5" fill="#8b5cf6" />
<text x="316" y="20" font-size="11" font-weight="bold" fill="currentColor">snorm₁ₕ</text>
<line x1="65" y1="240.0" x2="515" y2="240.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="244.0" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">0.0 μs</text>
<line x1="65" y1="191.25" x2="515" y2="191.25" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="195.25" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">250.0 μs</text>
<line x1="65" y1="142.5" x2="515" y2="142.5" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="146.5" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">500.0 μs</text>
<line x1="65" y1="93.75" x2="515" y2="93.75" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="97.75" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">750.0 μs</text>
<line x1="65" y1="45.0" x2="515" y2="45.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="49.0" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">1000.0 μs</text>
<line x1="65.0" y1="45" x2="65.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="65.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`0b9a62b`</text>
<line x1="215.0" y1="45" x2="215.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="215.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="365.0" y1="45" x2="365.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="365.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`41036bb`</text>
<line x1="515.0" y1="45" x2="515.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`ddb8d78`</text>
<polyline points="65.0,192.8 215.0,193.5 365.0,193.2 515.0,206.7" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="192.8" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
innerₕ: 242.0 μs (0 allocs, 0 B)</title></circle>
<text x="65.0" y="185.8" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">242.0</text>
<circle cx="215.0" cy="193.5" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
innerₕ: 238.5 μs (0 allocs, 0 B)</title></circle>
<text x="215.0" y="186.5" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">238.5</text>
<circle cx="365.0" cy="193.2" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
innerₕ: 240.0 μs (0 allocs, 0 B)</title></circle>
<text x="365.0" y="186.2" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">240.0</text>
<circle cx="515.0" cy="206.7" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
innerₕ: 170.9 μs (0 allocs, 0 B)</title></circle>
<text x="515.0" y="199.7" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">170.9</text>
<polyline points="65.0,85.9 215.0,87.3 365.0,84.7 515.0,136.6" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="85.9" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
norm₁ₕ: 790.2 μs (0 allocs, 0 B)</title></circle>
<text x="65.0" y="78.9" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">790.2</text>
<circle cx="215.0" cy="87.3" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
norm₁ₕ: 782.9 μs (0 allocs, 0 B)</title></circle>
<text x="215.0" y="80.3" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">782.9</text>
<circle cx="365.0" cy="84.7" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
norm₁ₕ: 796.6 μs (0 allocs, 0 B)</title></circle>
<text x="365.0" y="77.7" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">796.6</text>
<circle cx="515.0" cy="136.6" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
norm₁ₕ: 530.3 μs (0 allocs, 0 B)</title></circle>
<text x="515.0" y="129.6" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">530.3</text>
<polyline points="65.0,203.0 215.0,203.6 365.0,203.1 515.0,213.4" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="203.0" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
normₕ: 190.0 μs (0 allocs, 0 B)</title></circle>
<text x="65.0" y="196.0" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">190.0</text>
<circle cx="215.0" cy="203.6" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
normₕ: 186.8 μs (0 allocs, 0 B)</title></circle>
<text x="215.0" y="196.6" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">186.8</text>
<circle cx="365.0" cy="203.1" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
normₕ: 189.4 μs (0 allocs, 0 B)</title></circle>
<text x="365.0" y="196.1" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">189.4</text>
<circle cx="515.0" cy="213.4" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
normₕ: 136.7 μs (0 allocs, 0 B)</title></circle>
<text x="515.0" y="206.4" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">136.7</text>
<polyline points="65.0,127.3 215.0,127.5 365.0,126.4 515.0,163.2" fill="none" stroke="#8b5cf6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="127.3" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
snorm₁ₕ: 578.1 μs (0 allocs, 0 B)</title></circle>
<text x="65.0" y="120.3" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">578.1</text>
<circle cx="215.0" cy="127.5" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
snorm₁ₕ: 577.1 μs (0 allocs, 0 B)</title></circle>
<text x="215.0" y="120.5" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">577.1</text>
<circle cx="365.0" cy="126.4" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
snorm₁ₕ: 582.4 μs (0 allocs, 0 B)</title></circle>
<text x="365.0" y="119.4" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">582.4</text>
<circle cx="515.0" cy="163.2" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
snorm₁ₕ: 394.1 μs (0 allocs, 0 B)</title></circle>
<text x="515.0" y="156.2" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">394.1</text>
</svg></div>

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
<th style="padding:8px 6px; text-align:right;">Prev (<code>41036bb</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>ddb8d78</code>)</th>
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
<td style="padding:7px 6px; text-align:right;">2.89 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">2.06 ms</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-28.3% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-28.8% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">11</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">2.87 ms</td>
<td style="padding:7px 6px; text-align:right;">2.95 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">2.06 ms</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-28.3% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-30.2% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">3.39 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">2.32 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-31.4% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 3D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">3.84 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">2.67 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-30.4% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">16.27 ms</td>
<td style="padding:7px 6px; text-align:right;">16.31 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">11.1 ms</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-31.8% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-31.9% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">48 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">122.55 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">88.16 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-28.1% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">4</td>
<td style="padding:7px 6px; text-align:right;">128 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 3D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">755.03 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">562.3 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-25.5% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">4</td>
<td style="padding:7px 6px; text-align:right;">144 B</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:540px; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:0.8em; box-sizing:border-box;">
<svg viewBox="0 0 540 305" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<line x1="65" y1="16" x2="79" y2="16" stroke="#3b82f6" stroke-width="2.5" />
<circle cx="72" cy="16" r="3.5" fill="#3b82f6" />
<text x="83" y="20" font-size="11" font-weight="bold" fill="currentColor">Rₕ 1D (allocates its output)</text>
<line x1="299" y1="16" x2="313" y2="16" stroke="#10b981" stroke-width="2.5" />
<circle cx="306" cy="16" r="3.5" fill="#10b981" />
<text x="317" y="20" font-size="11" font-weight="bold" fill="currentColor">Rₕ! 1D</text>
<line x1="379" y1="16" x2="393" y2="16" stroke="#f59e0b" stroke-width="2.5" />
<circle cx="386" cy="16" r="3.5" fill="#f59e0b" />
<text x="397" y="20" font-size="11" font-weight="bold" fill="currentColor">Rₕ! 2D</text>
<line x1="65" y1="34" x2="79" y2="34" stroke="#8b5cf6" stroke-width="2.5" />
<circle cx="72" cy="34" r="3.5" fill="#8b5cf6" />
<text x="83" y="38" font-size="11" font-weight="bold" fill="currentColor">Rₕ! 3D</text>
<line x1="145" y1="34" x2="159" y2="34" stroke="#ec4899" stroke-width="2.5" />
<circle cx="152" cy="34" r="3.5" fill="#ec4899" />
<text x="163" y="38" font-size="11" font-weight="bold" fill="currentColor">avgₕ! 1D</text>
<line x1="239" y1="34" x2="253" y2="34" stroke="#06b6d4" stroke-width="2.5" />
<circle cx="246" cy="34" r="3.5" fill="#06b6d4" />
<text x="257" y="38" font-size="11" font-weight="bold" fill="currentColor">avgₕ! 2D</text>
<line x1="333" y1="34" x2="347" y2="34" stroke="#f97316" stroke-width="2.5" />
<circle cx="340" cy="34" r="3.5" fill="#f97316" />
<text x="351" y="38" font-size="11" font-weight="bold" fill="currentColor">avgₕ! 3D</text>
<line x1="65" y1="260.0" x2="515" y2="260.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="264.0" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">0.57×</text>
<line x1="65" y1="211.25" x2="515" y2="211.25" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="215.25" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">0.79×</text>
<line x1="65" y1="162.5" x2="515" y2="162.5" stroke="rgba(59, 130, 246, 0.4)" stroke-width="1.8" />
<text x="57" y="166.5" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">1.0× (ref)</text>
<line x1="65" y1="113.75" x2="515" y2="113.75" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="117.75" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">1.22×</text>
<line x1="65" y1="65.0" x2="515" y2="65.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="69.0" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">1.44×</text>
<line x1="65.0" y1="65" x2="65.0" y2="260" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="65.0" y="280" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`0b9a62b`</text>
<line x1="215.0" y1="65" x2="215.0" y2="260" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="215.0" y="280" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="365.0" y1="65" x2="365.0" y2="260" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="365.0" y="280" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`41036bb`</text>
<line x1="515.0" y1="65" x2="515.0" y2="260" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="280" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`ddb8d78`</text>
<polyline points="65.0,163.6 215.0,163.2 365.0,161.9 515.0,227.1" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="163.6" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
Rₕ 1D (allocates its output): 2.87 ms (baseline, 6 allocs, 7.64 MiB)</title></circle>
<text x="65.0" y="156.6" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">1.0×</text>
<circle cx="215.0" cy="163.2" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
Rₕ 1D (allocates its output): 2.88 ms (+0.2%, 6 allocs, 7.64 MiB)</title></circle>
<text x="215.0" y="156.2" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">1.0×</text>
<circle cx="365.0" cy="161.9" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
Rₕ 1D (allocates its output): 2.89 ms (+0.7%, 11 allocs, 7.64 MiB)</title></circle>
<text x="365.0" y="154.9" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">1.01×</text>
<circle cx="515.0" cy="227.1" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
Rₕ 1D (allocates its output): 2.06 ms (-28.3%, 11 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="220.1" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">0.72×</text>
<polyline points="65.0,163.6 215.0,162.6 365.0,157.3 515.0,227.0" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="163.6" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
Rₕ! 1D: 2.87 ms (baseline, 3 allocs, 64 B)</title></circle>
<text x="65.0" y="156.6" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.0×</text>
<circle cx="215.0" cy="162.6" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
Rₕ! 1D: 2.88 ms (+0.5%, 3 allocs, 64 B)</title></circle>
<text x="215.0" y="155.6" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.0×</text>
<circle cx="365.0" cy="157.3" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
Rₕ! 1D: 2.95 ms (+2.8%, 0 allocs, 0 B)</title></circle>
<text x="365.0" y="150.3" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.03×</text>
<circle cx="515.0" cy="227.0" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
Rₕ! 1D: 2.06 ms (-28.3%, 0 allocs, 0 B)</title></circle>
<text x="515.0" y="220.0" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">0.72×</text>
<polyline points="365.0,163.6 515.0,234.1" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="365.0" cy="163.6" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
Rₕ! 2D: 3.39 ms (baseline, 0 allocs, 0 B)</title></circle>
<text x="365.0" y="156.6" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">1.0×</text>
<circle cx="515.0" cy="234.1" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
Rₕ! 2D: 2.32 ms (-31.4%, 0 allocs, 0 B)</title></circle>
<text x="515.0" y="227.1" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">0.69×</text>
<polyline points="365.0,163.6 515.0,231.7" fill="none" stroke="#8b5cf6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="365.0" cy="163.6" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
Rₕ! 3D: 3.84 ms (baseline, 0 allocs, 0 B)</title></circle>
<text x="365.0" y="156.6" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">1.0×</text>
<circle cx="515.0" cy="231.7" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
Rₕ! 3D: 2.67 ms (-30.4%, 0 allocs, 0 B)</title></circle>
<text x="515.0" y="224.7" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">0.7×</text>
<polyline points="65.0,163.6 215.0,162.4 365.0,163.1 515.0,234.8" fill="none" stroke="#ec4899" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="163.6" r="4.5" fill="#ec4899" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
avgₕ! 1D: 16.27 ms (baseline, 2 allocs, 128 B)</title></circle>
<text x="65.0" y="156.6" font-size="10" font-weight="bold" fill="#ec4899" text-anchor="middle">1.0×</text>
<circle cx="215.0" cy="162.4" r="4.5" fill="#ec4899" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
avgₕ! 1D: 16.37 ms (+0.6%, 2 allocs, 128 B)</title></circle>
<text x="215.0" y="155.4" font-size="10" font-weight="bold" fill="#ec4899" text-anchor="middle">1.01×</text>
<circle cx="365.0" cy="163.1" r="4.5" fill="#ec4899" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
avgₕ! 1D: 16.31 ms (+0.2%, 3 allocs, 48 B)</title></circle>
<text x="365.0" y="156.1" font-size="10" font-weight="bold" fill="#ec4899" text-anchor="middle">1.0×</text>
<circle cx="515.0" cy="234.8" r="4.5" fill="#ec4899" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
avgₕ! 1D: 11.1 ms (-31.8%, 3 allocs, 48 B)</title></circle>
<text x="515.0" y="227.8" font-size="10" font-weight="bold" fill="#ec4899" text-anchor="middle">0.68×</text>
<polyline points="365.0,163.6 515.0,226.5" fill="none" stroke="#06b6d4" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="365.0" cy="163.6" r="4.5" fill="#06b6d4" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
avgₕ! 2D: 122.55 ms (baseline, 4 allocs, 128 B)</title></circle>
<text x="365.0" y="156.6" font-size="10" font-weight="bold" fill="#06b6d4" text-anchor="middle">1.0×</text>
<circle cx="515.0" cy="226.5" r="4.5" fill="#06b6d4" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
avgₕ! 2D: 88.16 ms (-28.1%, 4 allocs, 128 B)</title></circle>
<text x="515.0" y="219.5" font-size="10" font-weight="bold" fill="#06b6d4" text-anchor="middle">0.72×</text>
<polyline points="365.0,163.6 515.0,220.8" fill="none" stroke="#f97316" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="365.0" cy="163.6" r="4.5" fill="#f97316" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
avgₕ! 3D: 755.03 ms (baseline, 4 allocs, 144 B)</title></circle>
<text x="365.0" y="156.6" font-size="10" font-weight="bold" fill="#f97316" text-anchor="middle">1.0×</text>
<circle cx="515.0" cy="220.8" r="4.5" fill="#f97316" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
avgₕ! 3D: 562.3 ms (-25.5%, 4 allocs, 144 B)</title></circle>
<text x="515.0" y="213.8" font-size="10" font-weight="bold" fill="#f97316" text-anchor="middle">0.74×</text>
</svg></div>

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
<th style="padding:8px 6px; text-align:right;">Prev (<code>41036bb</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>ddb8d78</code>)</th>
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
<td style="padding:7px 6px; text-align:right;">670.2 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">558.5 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-21.5% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-16.7% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">22.89 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>∇₋ₕ (3 components)</code></td>
<td style="padding:7px 6px; text-align:right;">1.43 ms</td>
<td style="padding:7px 6px; text-align:right;">1.41 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">1.06 ms</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-26.0% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-25.1% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:right;">45.78 MiB</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:540px; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:0.8em; box-sizing:border-box;">
<svg viewBox="0 0 540 285" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<line x1="65" y1="16" x2="79" y2="16" stroke="#3b82f6" stroke-width="2.5" />
<circle cx="72" cy="16" r="3.5" fill="#3b82f6" />
<text x="83" y="20" font-size="11" font-weight="bold" fill="currentColor">D₋ₓ (3 components)</text>
<line x1="229" y1="16" x2="243" y2="16" stroke="#10b981" stroke-width="2.5" />
<circle cx="236" cy="16" r="3.5" fill="#10b981" />
<text x="247" y="20" font-size="11" font-weight="bold" fill="currentColor">∇₋ₕ (3 components)</text>
<line x1="65" y1="240.0" x2="515" y2="240.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="244.0" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">0.0 ms</text>
<line x1="65" y1="191.25" x2="515" y2="191.25" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="195.25" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">0.4 ms</text>
<line x1="65" y1="142.5" x2="515" y2="142.5" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="146.5" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">0.8 ms</text>
<line x1="65" y1="93.75" x2="515" y2="93.75" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="97.75" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">1.1 ms</text>
<line x1="65" y1="45.0" x2="515" y2="45.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="49.0" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">1.5 ms</text>
<line x1="65.0" y1="45" x2="65.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="65.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`0b9a62b`</text>
<line x1="215.0" y1="45" x2="215.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="215.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="365.0" y1="45" x2="365.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="365.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`41036bb`</text>
<line x1="515.0" y1="45" x2="515.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`ddb8d78`</text>
<polyline points="65.0,147.6 215.0,152.1 365.0,152.9 515.0,167.4" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="147.6" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
D₋ₓ (3 components): 0.7 ms (3 allocs, 22.89 MiB)</title></circle>
<text x="65.0" y="140.6" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">0.7</text>
<circle cx="215.0" cy="152.1" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
D₋ₓ (3 components): 0.7 ms (3 allocs, 22.89 MiB)</title></circle>
<text x="215.0" y="145.1" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">0.7</text>
<circle cx="365.0" cy="152.9" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
D₋ₓ (3 components): 0.7 ms (3 allocs, 22.89 MiB)</title></circle>
<text x="365.0" y="145.9" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">0.7</text>
<circle cx="515.0" cy="167.4" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
D₋ₓ (3 components): 0.6 ms (3 allocs, 22.89 MiB)</title></circle>
<text x="515.0" y="160.4" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">0.6</text>
<polyline points="65.0,54.4 215.0,58.2 365.0,56.6 515.0,102.7" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="54.4" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
∇₋ₕ (3 components): 1.4 ms (10 allocs, 45.78 MiB)</title></circle>
<text x="65.0" y="47.4" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.4</text>
<circle cx="215.0" cy="58.2" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
∇₋ₕ (3 components): 1.4 ms (10 allocs, 45.78 MiB)</title></circle>
<text x="215.0" y="51.2" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.4</text>
<circle cx="365.0" cy="56.6" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
∇₋ₕ (3 components): 1.4 ms (10 allocs, 45.78 MiB)</title></circle>
<text x="365.0" y="49.6" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.4</text>
<circle cx="515.0" cy="102.7" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
∇₋ₕ (3 components): 1.1 ms (10 allocs, 45.78 MiB)</title></circle>
<text x="515.0" y="95.7" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.1</text>
</svg></div>

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
<th style="padding:8px 6px; text-align:right;">Prev (<code>41036bb</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>ddb8d78</code>)</th>
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
<td style="padding:7px 6px; text-align:right;">355.8 μs</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">329.8 μs</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-10.6% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-7.3% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">29</td>
<td style="padding:7px 6px; text-align:right;">30.59 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>gridspace 3D</code></td>
<td style="padding:7px 6px; text-align:right;">1.63 ms</td>
<td style="padding:7px 6px; text-align:right;">1.63 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">1.11 ms</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-32.2% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-32.2% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">32</td>
<td style="padding:7px 6px; text-align:right;">38.21 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>hₘₐₓ 3D</code></td>
<td style="padding:7px 6px; text-align:right;">153.0 ns</td>
<td style="padding:7px 6px; text-align:right;">152.5 ns</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">119.5 ns</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-21.9% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-21.6% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:540px; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:0.8em; box-sizing:border-box;">
<svg viewBox="0 0 540 285" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<line x1="65" y1="16" x2="79" y2="16" stroke="#3b82f6" stroke-width="2.5" />
<circle cx="72" cy="16" r="3.5" fill="#3b82f6" />
<text x="83" y="20" font-size="11" font-weight="bold" fill="currentColor">gridspace 2D</text>
<line x1="187" y1="16" x2="201" y2="16" stroke="#10b981" stroke-width="2.5" />
<circle cx="194" cy="16" r="3.5" fill="#10b981" />
<text x="205" y="20" font-size="11" font-weight="bold" fill="currentColor">gridspace 3D</text>
<line x1="309" y1="16" x2="323" y2="16" stroke="#f59e0b" stroke-width="2.5" />
<circle cx="316" cy="16" r="3.5" fill="#f59e0b" />
<text x="327" y="20" font-size="11" font-weight="bold" fill="currentColor">hₘₐₓ 3D</text>
<line x1="65" y1="240.0" x2="515" y2="240.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="244.0" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">0.57×</text>
<line x1="65" y1="191.25" x2="515" y2="191.25" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="195.25" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">0.79×</text>
<line x1="65" y1="142.5" x2="515" y2="142.5" stroke="rgba(59, 130, 246, 0.4)" stroke-width="1.8" />
<text x="57" y="146.5" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">1.0× (ref)</text>
<line x1="65" y1="93.75" x2="515" y2="93.75" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="97.75" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">1.22×</text>
<line x1="65" y1="45.0" x2="515" y2="45.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="49.0" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">1.44×</text>
<line x1="65.0" y1="45" x2="65.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="65.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`0b9a62b`</text>
<line x1="215.0" y1="45" x2="215.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="215.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="365.0" y1="45" x2="365.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="365.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`41036bb`</text>
<line x1="515.0" y1="45" x2="515.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`ddb8d78`</text>
<polyline points="65.0,143.6 215.0,149.0 365.0,151.5 515.0,167.3" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="143.6" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
gridspace 2D: 368.8 μs (baseline, 38 allocs, 30.59 MiB)</title></circle>
<text x="65.0" y="136.6" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">1.0×</text>
<circle cx="215.0" cy="149.0" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
gridspace 2D: 360.0 μs (-2.4%, 38 allocs, 30.59 MiB)</title></circle>
<text x="215.0" y="142.0" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">0.98×</text>
<circle cx="365.0" cy="151.5" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
gridspace 2D: 355.8 μs (-3.5%, 29 allocs, 30.59 MiB)</title></circle>
<text x="365.0" y="144.5" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">0.96×</text>
<circle cx="515.0" cy="167.3" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
gridspace 2D: 329.8 μs (-10.6%, 29 allocs, 30.59 MiB)</title></circle>
<text x="515.0" y="160.3" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">0.89×</text>
<polyline points="65.0,143.6 215.0,143.2 365.0,143.7 515.0,215.8" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="143.6" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
gridspace 3D: 1.63 ms (baseline, 44 allocs, 38.21 MiB)</title></circle>
<text x="65.0" y="136.6" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.0×</text>
<circle cx="215.0" cy="143.2" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
gridspace 3D: 1.64 ms (+0.2%, 44 allocs, 38.21 MiB)</title></circle>
<text x="215.0" y="136.2" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.0×</text>
<circle cx="365.0" cy="143.7" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
gridspace 3D: 1.63 ms (-0.1%, 32 allocs, 38.21 MiB)</title></circle>
<text x="365.0" y="136.7" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.0×</text>
<circle cx="515.0" cy="215.8" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
gridspace 3D: 1.11 ms (-32.2%, 32 allocs, 38.21 MiB)</title></circle>
<text x="515.0" y="208.8" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">0.68×</text>
<polyline points="65.0,143.6 215.0,143.5 365.0,144.4 515.0,192.7" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="143.6" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
hₘₐₓ 3D: 153.0 ns (baseline, 0 allocs, 0 B)</title></circle>
<text x="65.0" y="136.6" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">1.0×</text>
<circle cx="215.0" cy="143.5" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
hₘₐₓ 3D: 153.1 ns (+0.1%, 0 allocs, 0 B)</title></circle>
<text x="215.0" y="136.5" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">1.0×</text>
<circle cx="365.0" cy="144.4" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
hₘₐₓ 3D: 152.5 ns (-0.3%, 0 allocs, 0 B)</title></circle>
<text x="365.0" y="137.4" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">1.0×</text>
<circle cx="515.0" cy="192.7" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
hₘₐₓ 3D: 119.5 ns (-21.9%, 0 allocs, 0 B)</title></circle>
<text x="515.0" y="185.7" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">0.78×</text>
</svg></div>

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
<th style="padding:8px 6px; text-align:right;">Prev (<code>41036bb</code>)</th>
<th style="padding:8px 6px; text-align:right;">Latest (<code>ddb8d78</code>)</th>
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
<td style="padding:7px 6px; text-align:right;">623.57 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">17.34 s</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+2680.6% 🔴</span></td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:right;">1.3 KiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>using Bramble</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">533.71 ms</td>
<td style="padding:7px 6px; text-align:right; font-weight:600;">437.94 ms</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-17.9% 🟢</span></td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:right;">1.3 KiB</td>
</tr>
</tbody>
</table>

  </div>
  <div style="flex:1 1 450px; min-width:340px;">
<div style="width:100%; max-width:540px; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:0.8em; box-sizing:border-box;">
<svg viewBox="0 0 540 285" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<line x1="65" y1="16" x2="79" y2="16" stroke="#3b82f6" stroke-width="2.5" />
<circle cx="72" cy="16" r="3.5" fill="#3b82f6" />
<text x="83" y="20" font-size="11" font-weight="bold" fill="currentColor">TTFX (load + first operator)</text>
<line x1="299" y1="16" x2="313" y2="16" stroke="#10b981" stroke-width="2.5" />
<circle cx="306" cy="16" r="3.5" fill="#10b981" />
<text x="317" y="20" font-size="11" font-weight="bold" fill="currentColor">using Bramble</text>
<line x1="65" y1="240.0" x2="515" y2="240.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="244.0" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">0.63×</text>
<line x1="65" y1="191.25" x2="515" y2="191.25" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="195.25" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">9.36×</text>
<line x1="65" y1="142.5" x2="515" y2="142.5" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="146.5" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">18.1×</text>
<line x1="65" y1="93.75" x2="515" y2="93.75" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="97.75" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">26.83×</text>
<line x1="65" y1="45.0" x2="515" y2="45.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="49.0" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">35.56×</text>
<line x1="65.0" y1="45" x2="65.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="65.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`0b9a62b`</text>
<line x1="215.0" y1="45" x2="215.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="215.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="365.0" y1="45" x2="365.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="365.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`41036bb`</text>
<line x1="515.0" y1="45" x2="515.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`ddb8d78`</text>
<polyline points="215.0,237.9 365.0,237.3 515.0,70.9" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="215.0" cy="237.9" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
TTFX (load + first operator): 560.78 ms (baseline, 45 allocs, 1.3 KiB)</title></circle>
<text x="215.0" y="230.9" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">1.0×</text>
<circle cx="365.0" cy="237.3" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
TTFX (load + first operator): 623.57 ms (+11.2%, 45 allocs, 1.3 KiB)</title></circle>
<text x="365.0" y="230.3" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">1.11×</text>
<circle cx="515.0" cy="70.9" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
TTFX (load + first operator): 17.34 s (+2992.0%, 45 allocs, 1.3 KiB)</title></circle>
<text x="515.0" y="63.9" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">30.92×</text>
<polyline points="215.0,237.9 365.0,237.9 515.0,238.9" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="215.0" cy="237.9" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
using Bramble: 527.5 ms (baseline, 45 allocs, 1.3 KiB)</title></circle>
<text x="215.0" y="230.9" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.0×</text>
<circle cx="365.0" cy="237.9" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>41036bb (Julia 1.12.7)
using Bramble: 533.71 ms (+1.2%, 45 allocs, 1.3 KiB)</title></circle>
<text x="365.0" y="230.9" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.01×</text>
<circle cx="515.0" cy="238.9" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>ddb8d78 (Julia 1.12.7)
using Bramble: 437.94 ms (-17.0%, 45 allocs, 1.3 KiB)</title></circle>
<text x="515.0" y="231.9" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">0.83×</text>
</svg></div>

  </div>
</div>
```

## How to Add New Benchmark Runs

To record performance on a new commit or after an optimization pass, run:

```bash
julia --project=benchmark benchmark/benchmarks.jl --save benchmark/baselines/baseline_$(git rev-parse --short HEAD).json
```

Rebuilding the documentation (`julia -e 'using Pkg; Pkg.activate("docs"); include("docs/make.jl")'`) will automatically discover all `baseline_*.json` files and append new comparison columns, delta calculations, and charts.
