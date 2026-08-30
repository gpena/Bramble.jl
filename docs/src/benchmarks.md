# Performance & Benchmarks

Bramble tracks memory allocations and performance regressions with a dedicated regression suite in `benchmark/benchmarks.jl`.
All measurements below are run on **1,000,000 grid points** per dimension setup (e.g. ``1000 \times 1000`` in 2D, ``100 \times 100 \times 100`` in 3D).

## Recorded Baselines

Comparing **3** recorded baselines in chronological order. The earliest run (`0b9a62b`) serves as reference baseline for relative speedup/slowdown calculations.

| Commit | Julia | Summary | File |
|---|:---:|---|---|
| `0b9a62b` *(baseline)* | `1.12.7` | test: run the allocation assertions under coverage instead of skipping them | `baseline_0b9a62b.json` |
| `855fbf5` | `1.12.7` | docs(benchmarks): switch to inline SVG charts and streamline baselines table | `baseline_855fbf5.json` |
| `50c6a46` | `1.12.7` | fix(docs): attach detached docstrings and enable full history in docs CI | `baseline_50c6a46.json` |

## Comparative Timings & Allocations

### Operators 2D

```@raw html
<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:start; margin:1.2rem 0 2.5rem 0;">
  <div style="flex:1 1 430px; min-width:320px; overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4;">
<thead>
<tr style="border-bottom:2px solid rgba(128,128,128,0.3);">
<th style="padding:8px 6px; text-align:left;">Benchmark</th>
<th style="padding:8px 6px; text-align:right;"><code>0b9a62b</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>855fbf5</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>50c6a46</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Dcₓ</code></td>
<td style="padding:7px 6px; text-align:right;">257.2 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">256.5 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">293.8 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+14.2% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋ᵧ</code></td>
<td style="padding:7px 6px; text-align:right;">161.4 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">162.6 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">174.4 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+8.1% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋ₓ</code></td>
<td style="padding:7px 6px; text-align:right;">203.7 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">203.3 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">226.2 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+11.0% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₋ₓ</code></td>
<td style="padding:7px 6px; text-align:right;">171.4 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">171.0 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">191.0 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+11.4% 🔴</span></td>
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
<text x="57" y="195.25" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">75.0 μs</text>
<line x1="65" y1="142.5" x2="515" y2="142.5" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="146.5" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">150.0 μs</text>
<line x1="65" y1="93.75" x2="515" y2="93.75" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="97.75" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">225.0 μs</text>
<line x1="65" y1="45.0" x2="515" y2="45.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="49.0" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">300.0 μs</text>
<line x1="65.0" y1="45" x2="65.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="65.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`0b9a62b`</text>
<line x1="290.0" y1="45" x2="290.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="290.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="515.0" y1="45" x2="515.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`50c6a46`</text>
<polyline points="65.0,72.8 290.0,73.2 515.0,49.0" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="72.8" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
Dcₓ: 257.2 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="65.0" y="65.8" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">257.2</text>
<circle cx="290.0" cy="73.2" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
Dcₓ: 256.5 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="290.0" y="66.2" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">256.5</text>
<circle cx="515.0" cy="49.0" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
Dcₓ: 293.8 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="42.0" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">293.8</text>
<polyline points="65.0,135.1 290.0,134.3 515.0,126.6" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="135.1" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
D₋ᵧ: 161.4 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="65.0" y="128.1" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">161.4</text>
<circle cx="290.0" cy="134.3" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
D₋ᵧ: 162.6 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="290.0" y="127.3" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">162.6</text>
<circle cx="515.0" cy="126.6" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
D₋ᵧ: 174.4 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="119.6" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">174.4</text>
<polyline points="65.0,107.6 290.0,107.9 515.0,93.0" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="107.6" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
D₋ₓ: 203.7 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="65.0" y="100.6" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">203.7</text>
<circle cx="290.0" cy="107.9" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
D₋ₓ: 203.3 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="290.0" y="100.9" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">203.3</text>
<circle cx="515.0" cy="93.0" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
D₋ₓ: 226.2 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="86.0" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">226.2</text>
<polyline points="65.0,128.6 290.0,128.8 515.0,115.9" fill="none" stroke="#8b5cf6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="128.6" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
M₋ₓ: 171.4 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="65.0" y="121.6" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">171.4</text>
<circle cx="290.0" cy="128.8" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
M₋ₓ: 171.0 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="290.0" y="121.8" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">171.0</text>
<circle cx="515.0" cy="115.9" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
M₋ₓ: 191.0 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="108.9" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">191.0</text>
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
<th style="padding:8px 6px; text-align:right;"><code>0b9a62b</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>855fbf5</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>50c6a46</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋₂</code></td>
<td style="padding:7px 6px; text-align:right;">200.9 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">222.8 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">221.0 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+10.0% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>innerₕ</code></td>
<td style="padding:7px 6px; text-align:right;">240.2 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">240.0 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">240.3 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>∇₋ₕ</code></td>
<td style="padding:7px 6px; text-align:right;">694.1 μs</td>
<td style="padding:7px 6px; text-align:center;">15</td>
<td style="padding:7px 6px; text-align:right;">686.6 μs</td>
<td style="padding:7px 6px; text-align:center;">15</td>
<td style="padding:7px 6px; text-align:right;">683.9 μs</td>
<td style="padding:7px 6px; text-align:center;">15</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-1.5% 🟢</span></td>
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
<line x1="290.0" y1="45" x2="290.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="290.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="515.0" y1="45" x2="515.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`50c6a46`</text>
<polyline points="65.0,200.8 290.0,196.6 515.0,196.9" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="200.8" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
D₋₂: 200.9 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="65.0" y="193.8" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">200.9</text>
<circle cx="290.0" cy="196.6" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
D₋₂: 222.8 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="290.0" y="189.6" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">222.8</text>
<circle cx="515.0" cy="196.9" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
D₋₂: 221.0 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="189.9" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">221.0</text>
<polyline points="65.0,193.2 290.0,193.2 515.0,193.1" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="193.2" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
innerₕ: 240.2 μs (0 allocs, 0 B)</title></circle>
<text x="65.0" y="186.2" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">240.2</text>
<circle cx="290.0" cy="193.2" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
innerₕ: 240.0 μs (0 allocs, 0 B)</title></circle>
<text x="290.0" y="186.2" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">240.0</text>
<circle cx="515.0" cy="193.1" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
innerₕ: 240.3 μs (0 allocs, 0 B)</title></circle>
<text x="515.0" y="186.1" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">240.3</text>
<polyline points="65.0,104.7 290.0,106.1 515.0,106.6" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="104.7" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
∇₋ₕ: 694.1 μs (15 allocs, 22.92 MiB)</title></circle>
<text x="65.0" y="97.7" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">694.1</text>
<circle cx="290.0" cy="106.1" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
∇₋ₕ: 686.6 μs (15 allocs, 22.92 MiB)</title></circle>
<text x="290.0" y="99.1" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">686.6</text>
<circle cx="515.0" cy="106.6" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
∇₋ₕ: 683.9 μs (15 allocs, 22.92 MiB)</title></circle>
<text x="515.0" y="99.6" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">683.9</text>
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
<th style="padding:8px 6px; text-align:right;"><code>0b9a62b</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>855fbf5</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>50c6a46</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₊ᵧ 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">161.6 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">172.6 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₊₂ 3D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">227.8 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">227.4 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>M₊ₓ 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">160.4 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">170.0 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>jumpᵧ 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">162.0 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">165.8 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>jump₂ 3D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">227.6 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">222.2 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>jumpₓ 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">164.7 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">169.4 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
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
<line x1="290.0" y1="65" x2="290.0" y2="260" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="290.0" y="280" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="515.0" y1="65" x2="515.0" y2="260" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="280" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`50c6a46`</text>
<polyline points="290.0,154.9 515.0,147.8" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="290.0" cy="154.9" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
M₊ᵧ 2D: 161.6 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="290.0" y="147.9" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">161.6</text>
<circle cx="515.0" cy="147.8" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
M₊ᵧ 2D: 172.6 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="140.8" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">172.6</text>
<polyline points="290.0,111.9 515.0,112.2" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="290.0" cy="111.9" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
M₊₂ 3D: 227.8 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="290.0" y="104.9" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">227.8</text>
<circle cx="515.0" cy="112.2" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
M₊₂ 3D: 227.4 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="105.2" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">227.4</text>
<polyline points="290.0,155.7 515.0,149.5" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="290.0" cy="155.7" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
M₊ₓ 2D: 160.4 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="290.0" y="148.7" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">160.4</text>
<circle cx="515.0" cy="149.5" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
M₊ₓ 2D: 170.0 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="142.5" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">170.0</text>
<polyline points="290.0,154.7 515.0,152.2" fill="none" stroke="#8b5cf6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="290.0" cy="154.7" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
jumpᵧ 2D: 162.0 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="290.0" y="147.7" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">162.0</text>
<circle cx="515.0" cy="152.2" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
jumpᵧ 2D: 165.8 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="145.2" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">165.8</text>
<polyline points="290.0,112.0 515.0,115.6" fill="none" stroke="#ec4899" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="290.0" cy="112.0" r="4.5" fill="#ec4899" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
jump₂ 3D: 227.6 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="290.0" y="105.0" font-size="10" font-weight="bold" fill="#ec4899" text-anchor="middle">227.6</text>
<circle cx="515.0" cy="115.6" r="4.5" fill="#ec4899" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
jump₂ 3D: 222.2 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="108.6" font-size="10" font-weight="bold" fill="#ec4899" text-anchor="middle">222.2</text>
<polyline points="290.0,153.0 515.0,149.9" fill="none" stroke="#06b6d4" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="290.0" cy="153.0" r="4.5" fill="#06b6d4" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
jumpₓ 2D: 164.7 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="290.0" y="146.0" font-size="10" font-weight="bold" fill="#06b6d4" text-anchor="middle">164.7</text>
<circle cx="515.0" cy="149.9" r="4.5" fill="#06b6d4" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
jumpₓ 2D: 169.4 μs (3 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="142.9" font-size="10" font-weight="bold" fill="#06b6d4" text-anchor="middle">169.4</text>
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
<th style="padding:8px 6px; text-align:right;"><code>0b9a62b</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>855fbf5</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>50c6a46</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>innerₕ</code></td>
<td style="padding:7px 6px; text-align:right;">242.0 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">238.5 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">239.7 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#10b981; font-weight:bold;">-1.0% 🟢</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>norm₁ₕ</code></td>
<td style="padding:7px 6px; text-align:right;">790.2 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">782.9 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">795.2 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+0.6% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>normₕ</code></td>
<td style="padding:7px 6px; text-align:right;">190.0 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">186.8 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">190.8 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>snorm₁ₕ</code></td>
<td style="padding:7px 6px; text-align:right;">578.1 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">577.1 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">578.5 μs</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
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
<line x1="290.0" y1="45" x2="290.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="290.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="515.0" y1="45" x2="515.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`50c6a46`</text>
<polyline points="65.0,192.8 290.0,193.5 515.0,193.3" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="192.8" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
innerₕ: 242.0 μs (0 allocs, 0 B)</title></circle>
<text x="65.0" y="185.8" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">242.0</text>
<circle cx="290.0" cy="193.5" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
innerₕ: 238.5 μs (0 allocs, 0 B)</title></circle>
<text x="290.0" y="186.5" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">238.5</text>
<circle cx="515.0" cy="193.3" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
innerₕ: 239.7 μs (0 allocs, 0 B)</title></circle>
<text x="515.0" y="186.3" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">239.7</text>
<polyline points="65.0,85.9 290.0,87.3 515.0,84.9" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="85.9" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
norm₁ₕ: 790.2 μs (0 allocs, 0 B)</title></circle>
<text x="65.0" y="78.9" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">790.2</text>
<circle cx="290.0" cy="87.3" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
norm₁ₕ: 782.9 μs (0 allocs, 0 B)</title></circle>
<text x="290.0" y="80.3" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">782.9</text>
<circle cx="515.0" cy="84.9" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
norm₁ₕ: 795.2 μs (0 allocs, 0 B)</title></circle>
<text x="515.0" y="77.9" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">795.2</text>
<polyline points="65.0,203.0 290.0,203.6 515.0,202.8" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="203.0" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
normₕ: 190.0 μs (0 allocs, 0 B)</title></circle>
<text x="65.0" y="196.0" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">190.0</text>
<circle cx="290.0" cy="203.6" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
normₕ: 186.8 μs (0 allocs, 0 B)</title></circle>
<text x="290.0" y="196.6" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">186.8</text>
<circle cx="515.0" cy="202.8" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
normₕ: 190.8 μs (0 allocs, 0 B)</title></circle>
<text x="515.0" y="195.8" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">190.8</text>
<polyline points="65.0,127.3 290.0,127.5 515.0,127.2" fill="none" stroke="#8b5cf6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="127.3" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
snorm₁ₕ: 578.1 μs (0 allocs, 0 B)</title></circle>
<text x="65.0" y="120.3" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">578.1</text>
<circle cx="290.0" cy="127.5" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
snorm₁ₕ: 577.1 μs (0 allocs, 0 B)</title></circle>
<text x="290.0" y="120.5" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">577.1</text>
<circle cx="515.0" cy="127.2" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
snorm₁ₕ: 578.5 μs (0 allocs, 0 B)</title></circle>
<text x="515.0" y="120.2" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">578.5</text>
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
<th style="padding:8px 6px; text-align:right;"><code>0b9a62b</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>855fbf5</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>50c6a46</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ 1D (allocates its output)</code></td>
<td style="padding:7px 6px; text-align:right;">2.87 ms</td>
<td style="padding:7px 6px; text-align:center;">6</td>
<td style="padding:7px 6px; text-align:right;">2.88 ms</td>
<td style="padding:7px 6px; text-align:center;">6</td>
<td style="padding:7px 6px; text-align:right;">3.23 ms</td>
<td style="padding:7px 6px; text-align:center;">11</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+12.3% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">7.64 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">2.87 ms</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">2.88 ms</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">3.26 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+13.9% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">3.61 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>Rₕ! 3D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">4.06 ms</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">0 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 1D</code></td>
<td style="padding:7px 6px; text-align:right;">16.27 ms</td>
<td style="padding:7px 6px; text-align:center;">2</td>
<td style="padding:7px 6px; text-align:right;">16.37 ms</td>
<td style="padding:7px 6px; text-align:center;">2</td>
<td style="padding:7px 6px; text-align:right;">16.57 ms</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+1.8% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">48 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 2D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">128.98 ms</td>
<td style="padding:7px 6px; text-align:center;">4</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">128 B</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>avgₕ! 3D</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">806.4 ms</td>
<td style="padding:7px 6px; text-align:center;">4</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
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
<text x="57" y="264.0" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">0.63×</text>
<line x1="65" y1="211.25" x2="515" y2="211.25" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="215.25" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">0.83×</text>
<line x1="65" y1="162.5" x2="515" y2="162.5" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="166.5" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">1.03×</text>
<line x1="65" y1="113.75" x2="515" y2="113.75" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="117.75" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">1.24×</text>
<line x1="65" y1="65.0" x2="515" y2="65.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="69.0" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">1.44×</text>
<line x1="65.0" y1="65" x2="65.0" y2="260" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="65.0" y="280" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`0b9a62b`</text>
<line x1="290.0" y1="65" x2="290.0" y2="260" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="290.0" y="280" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="515.0" y1="65" x2="515.0" y2="260" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="280" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`50c6a46`</text>
<polyline points="65.0,170.9 290.0,170.4 515.0,141.3" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="170.9" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
Rₕ 1D (allocates its output): 2.87 ms (baseline, 6 allocs, 7.64 MiB)</title></circle>
<text x="65.0" y="163.9" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">1.0×</text>
<circle cx="290.0" cy="170.4" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
Rₕ 1D (allocates its output): 2.88 ms (+0.2%, 6 allocs, 7.64 MiB)</title></circle>
<text x="290.0" y="163.4" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">1.0×</text>
<circle cx="515.0" cy="141.3" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
Rₕ 1D (allocates its output): 3.23 ms (+12.3%, 11 allocs, 7.64 MiB)</title></circle>
<text x="515.0" y="134.3" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">1.12×</text>
<polyline points="65.0,170.9 290.0,169.8 515.0,137.5" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="170.9" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
Rₕ! 1D: 2.87 ms (baseline, 3 allocs, 64 B)</title></circle>
<text x="65.0" y="163.9" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.0×</text>
<circle cx="290.0" cy="169.8" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
Rₕ! 1D: 2.88 ms (+0.5%, 3 allocs, 64 B)</title></circle>
<text x="290.0" y="162.8" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.0×</text>
<circle cx="515.0" cy="137.5" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
Rₕ! 1D: 3.26 ms (+13.9%, 0 allocs, 0 B)</title></circle>
<text x="515.0" y="130.5" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.14×</text>
<circle cx="515.0" cy="170.9" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
Rₕ! 2D: 3.61 ms (baseline, 0 allocs, 0 B)</title></circle>
<text x="515.0" y="163.9" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">1.0×</text>
<circle cx="515.0" cy="170.9" r="4.5" fill="#8b5cf6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
Rₕ! 3D: 4.06 ms (baseline, 0 allocs, 0 B)</title></circle>
<text x="515.0" y="163.9" font-size="10" font-weight="bold" fill="#8b5cf6" text-anchor="middle">1.0×</text>
<polyline points="65.0,170.9 290.0,169.6 515.0,166.5" fill="none" stroke="#ec4899" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="170.9" r="4.5" fill="#ec4899" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
avgₕ! 1D: 16.27 ms (baseline, 2 allocs, 128 B)</title></circle>
<text x="65.0" y="163.9" font-size="10" font-weight="bold" fill="#ec4899" text-anchor="middle">1.0×</text>
<circle cx="290.0" cy="169.6" r="4.5" fill="#ec4899" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
avgₕ! 1D: 16.37 ms (+0.6%, 2 allocs, 128 B)</title></circle>
<text x="290.0" y="162.6" font-size="10" font-weight="bold" fill="#ec4899" text-anchor="middle">1.01×</text>
<circle cx="515.0" cy="166.5" r="4.5" fill="#ec4899" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
avgₕ! 1D: 16.57 ms (+1.8%, 3 allocs, 48 B)</title></circle>
<text x="515.0" y="159.5" font-size="10" font-weight="bold" fill="#ec4899" text-anchor="middle">1.02×</text>
<circle cx="515.0" cy="170.9" r="4.5" fill="#06b6d4" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
avgₕ! 2D: 128.98 ms (baseline, 4 allocs, 128 B)</title></circle>
<text x="515.0" y="163.9" font-size="10" font-weight="bold" fill="#06b6d4" text-anchor="middle">1.0×</text>
<circle cx="515.0" cy="170.9" r="4.5" fill="#f97316" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
avgₕ! 3D: 806.4 ms (baseline, 4 allocs, 144 B)</title></circle>
<text x="515.0" y="163.9" font-size="10" font-weight="bold" fill="#f97316" text-anchor="middle">1.0×</text>
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
<th style="padding:8px 6px; text-align:right;"><code>0b9a62b</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>855fbf5</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>50c6a46</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>D₋ₓ (3 components)</code></td>
<td style="padding:7px 6px; text-align:right;">711.0 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">675.9 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:right;">784.6 μs</td>
<td style="padding:7px 6px; text-align:center;">3</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+10.3% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">22.89 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>∇₋ₕ (3 components)</code></td>
<td style="padding:7px 6px; text-align:right;">1.43 ms</td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:right;">1.4 ms</td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:right;">1.47 ms</td>
<td style="padding:7px 6px; text-align:center;">10</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+2.8% 🔴</span></td>
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
<line x1="290.0" y1="45" x2="290.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="290.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="515.0" y1="45" x2="515.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`50c6a46`</text>
<polyline points="65.0,147.6 290.0,152.1 515.0,138.0" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="147.6" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
D₋ₓ (3 components): 0.7 ms (3 allocs, 22.89 MiB)</title></circle>
<text x="65.0" y="140.6" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">0.7</text>
<circle cx="290.0" cy="152.1" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
D₋ₓ (3 components): 0.7 ms (3 allocs, 22.89 MiB)</title></circle>
<text x="290.0" y="145.1" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">0.7</text>
<circle cx="515.0" cy="138.0" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
D₋ₓ (3 components): 0.8 ms (3 allocs, 22.89 MiB)</title></circle>
<text x="515.0" y="131.0" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">0.8</text>
<polyline points="65.0,54.4 290.0,58.2 515.0,49.3" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="54.4" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
∇₋ₕ (3 components): 1.4 ms (10 allocs, 45.78 MiB)</title></circle>
<text x="65.0" y="47.4" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.4</text>
<circle cx="290.0" cy="58.2" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
∇₋ₕ (3 components): 1.4 ms (10 allocs, 45.78 MiB)</title></circle>
<text x="290.0" y="51.2" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.4</text>
<circle cx="515.0" cy="49.3" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
∇₋ₕ (3 components): 1.5 ms (10 allocs, 45.78 MiB)</title></circle>
<text x="515.0" y="42.3" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.5</text>
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
<th style="padding:8px 6px; text-align:right;"><code>0b9a62b</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>855fbf5</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>50c6a46</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>gridspace 2D</code></td>
<td style="padding:7px 6px; text-align:right;">368.8 μs</td>
<td style="padding:7px 6px; text-align:center;">38</td>
<td style="padding:7px 6px; text-align:right;">360.0 μs</td>
<td style="padding:7px 6px; text-align:center;">38</td>
<td style="padding:7px 6px; text-align:right;">369.8 μs</td>
<td style="padding:7px 6px; text-align:center;">29</td>
<td style="padding:7px 6px; text-align:center;"><span style="opacity:0.6;">(=)</span></td>
<td style="padding:7px 6px; text-align:right;">30.59 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>gridspace 3D</code></td>
<td style="padding:7px 6px; text-align:right;">1.63 ms</td>
<td style="padding:7px 6px; text-align:center;">44</td>
<td style="padding:7px 6px; text-align:right;">1.64 ms</td>
<td style="padding:7px 6px; text-align:center;">44</td>
<td style="padding:7px 6px; text-align:right;">1.74 ms</td>
<td style="padding:7px 6px; text-align:center;">32</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+6.4% 🔴</span></td>
<td style="padding:7px 6px; text-align:right;">38.21 MiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>hₘₐₓ 3D</code></td>
<td style="padding:7px 6px; text-align:right;">153.0 ns</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">153.1 ns</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:right;">162.8 ns</td>
<td style="padding:7px 6px; text-align:center;">0</td>
<td style="padding:7px 6px; text-align:center;"><span style="color:#ef4444; font-weight:bold;">+6.4% 🔴</span></td>
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
<text x="57" y="244.0" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">0.63×</text>
<line x1="65" y1="191.25" x2="515" y2="191.25" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="195.25" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">0.83×</text>
<line x1="65" y1="142.5" x2="515" y2="142.5" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="146.5" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">1.03×</text>
<line x1="65" y1="93.75" x2="515" y2="93.75" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="97.75" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">1.24×</text>
<line x1="65" y1="45.0" x2="515" y2="45.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="49.0" font-size="10" fill="currentColor" opacity="0.7" text-anchor="end">1.44×</text>
<line x1="65.0" y1="45" x2="65.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="65.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`0b9a62b`</text>
<line x1="290.0" y1="45" x2="290.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="290.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="515.0" y1="45" x2="515.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`50c6a46`</text>
<polyline points="65.0,150.9 290.0,156.7 515.0,150.3" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="150.9" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
gridspace 2D: 368.8 μs (baseline, 38 allocs, 30.59 MiB)</title></circle>
<text x="65.0" y="143.9" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">1.0×</text>
<circle cx="290.0" cy="156.7" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
gridspace 2D: 360.0 μs (-2.4%, 38 allocs, 30.59 MiB)</title></circle>
<text x="290.0" y="149.7" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">0.98×</text>
<circle cx="515.0" cy="150.3" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
gridspace 2D: 369.8 μs (+0.3%, 29 allocs, 30.59 MiB)</title></circle>
<text x="515.0" y="143.3" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">1.0×</text>
<polyline points="65.0,150.9 290.0,150.5 515.0,135.4" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="150.9" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
gridspace 3D: 1.63 ms (baseline, 44 allocs, 38.21 MiB)</title></circle>
<text x="65.0" y="143.9" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.0×</text>
<circle cx="290.0" cy="150.5" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
gridspace 3D: 1.64 ms (+0.2%, 44 allocs, 38.21 MiB)</title></circle>
<text x="290.0" y="143.5" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.0×</text>
<circle cx="515.0" cy="135.4" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
gridspace 3D: 1.74 ms (+6.4%, 32 allocs, 38.21 MiB)</title></circle>
<text x="515.0" y="128.4" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">1.06×</text>
<polyline points="65.0,150.9 290.0,150.8 515.0,135.5" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="65.0" cy="150.9" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>0b9a62b (Julia 1.12.7)
hₘₐₓ 3D: 153.0 ns (baseline, 0 allocs, 0 B)</title></circle>
<text x="65.0" y="143.9" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">1.0×</text>
<circle cx="290.0" cy="150.8" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
hₘₐₓ 3D: 153.1 ns (+0.1%, 0 allocs, 0 B)</title></circle>
<text x="290.0" y="143.8" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">1.0×</text>
<circle cx="515.0" cy="135.5" r="4.5" fill="#f59e0b" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
hₘₐₓ 3D: 162.8 ns (+6.4%, 0 allocs, 0 B)</title></circle>
<text x="515.0" y="128.5" font-size="10" font-weight="bold" fill="#f59e0b" text-anchor="middle">1.06×</text>
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
<th style="padding:8px 6px; text-align:right;"><code>0b9a62b</code> (ref)</th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>855fbf5</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:right;"><code>50c6a46</code></th>
<th style="padding:8px 6px; text-align:center;">Allocs</th>
<th style="padding:8px 6px; text-align:center;">Δ vs Base</th>
<th style="padding:8px 6px; text-align:right;">Memory</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>TTFX (load + first operator)</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">560.78 ms</td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:right;">689.78 ms</td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">1.3 KiB</td>
</tr>
<tr style="border-bottom:1px solid rgba(128,128,128,0.15);">
<td style="padding:7px 6px; font-weight:600;"><code>using Bramble</code></td>
<td style="padding:7px 6px; text-align:right; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
<td style="padding:7px 6px; text-align:right;">527.5 ms</td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:right;">596.04 ms</td>
<td style="padding:7px 6px; text-align:center;">45</td>
<td style="padding:7px 6px; text-align:center; opacity:0.4;">—</td>
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
<text x="57" y="244.0" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">0.0 ms</text>
<line x1="65" y1="191.25" x2="515" y2="191.25" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="195.25" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">250.0 ms</text>
<line x1="65" y1="142.5" x2="515" y2="142.5" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="146.5" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">500.0 ms</text>
<line x1="65" y1="93.75" x2="515" y2="93.75" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="97.75" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">750.0 ms</text>
<line x1="65" y1="45.0" x2="515" y2="45.0" stroke="rgba(128,128,128,0.18)" stroke-dasharray="3,3" />
<text x="57" y="49.0" font-size="10" fill="currentColor" opacity="0.65" text-anchor="end">1000.0 ms</text>
<line x1="65.0" y1="45" x2="65.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="65.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`0b9a62b`</text>
<line x1="290.0" y1="45" x2="290.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="290.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`855fbf5`</text>
<line x1="515.0" y1="45" x2="515.0" y2="240" stroke="rgba(128,128,128,0.15)" stroke-dasharray="2,2" />
<text x="515.0" y="260" font-size="11" font-family="monospace" fill="currentColor" opacity="0.8" text-anchor="middle">`50c6a46`</text>
<polyline points="290.0,130.6 515.0,105.5" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="290.0" cy="130.6" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
TTFX (load + first operator): 560.8 ms (45 allocs, 1.3 KiB)</title></circle>
<text x="290.0" y="123.6" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">560.8</text>
<circle cx="515.0" cy="105.5" r="4.5" fill="#3b82f6" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
TTFX (load + first operator): 689.8 ms (45 allocs, 1.3 KiB)</title></circle>
<text x="515.0" y="98.5" font-size="10" font-weight="bold" fill="#3b82f6" text-anchor="middle">689.8</text>
<polyline points="290.0,137.1 515.0,123.8" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" opacity="0.88" />
<circle cx="290.0" cy="137.1" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>855fbf5 (Julia 1.12.7)
using Bramble: 527.5 ms (45 allocs, 1.3 KiB)</title></circle>
<text x="290.0" y="130.1" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">527.5</text>
<circle cx="515.0" cy="123.8" r="4.5" fill="#10b981" stroke="var(--documenter-bg, #fff)" stroke-width="1.5">
<title>50c6a46 (Julia 1.12.7)
using Bramble: 596.0 ms (45 allocs, 1.3 KiB)</title></circle>
<text x="515.0" y="116.8" font-size="10" font-weight="bold" fill="#10b981" text-anchor="middle">596.0</text>
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
