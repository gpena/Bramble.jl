# Performance & Benchmarks

Bramble tracks memory allocations and performance regressions with a dedicated regression suite in `benchmark/benchmarks.jl`.
All measurements below are run on **1,000,000 grid points** per dimension setup (e.g. ``1000 \times 1000`` in 2D, ``100 \times 100 \times 100`` in 3D).

## Recorded Baselines

Comparing **2** recorded baselines. The earliest run (`0b9a62b`) serves as reference baseline for relative speedup/slowdown calculations.

| Commit | Julia | Summary | File |
|---|:---:|---|---|
| `0b9a62b` *(baseline)* | `1.12.7` | test: run the allocation assertions under coverage instead of skipping them | `baseline_0b9a62b.json` |
| `855fbf5` | `1.12.7` | docs(benchmarks): switch to inline SVG charts and streamline baselines table | `baseline_855fbf5.json` |

## Comparative Timings & Allocations

### Operators 2D

| Benchmark | `0b9a62b` (ref) Time | Allocs | Memory | `855fbf5` Time | Allocs | Memory |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`Dcₓ`** | 257.2 μs | 3 | 7.64 MiB | 256.5 μs (=) | 3 | 7.64 MiB |
| **`D₋ᵧ`** | 161.4 μs | 3 | 7.64 MiB | 162.6 μs (**+0.7%** 🔴) | 3 | 7.64 MiB |
| **`D₋ₓ`** | 203.7 μs | 3 | 7.64 MiB | 203.3 μs (=) | 3 | 7.64 MiB |
| **`M₋ₓ`** | 171.4 μs | 3 | 7.64 MiB | 171.0 μs (=) | 3 | 7.64 MiB |

```@raw html
<div style="width:100%; max-width:710px; margin:1.5em auto; overflow-x:auto; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:1em;">
<svg viewBox="0 0 710 273" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<rect x="220" y="15" width="12" height="12" rx="2" fill="#3b82f6" />
<text x="236" y="25" font-size="12" fill="currentColor" opacity="0.9">0b9a62b (Julia 1.12.7)</text>
<rect x="400" y="15" width="12" height="12" rx="2" fill="#10b981" />
<text x="416" y="25" font-size="12" fill="currentColor" opacity="0.9">855fbf5 (Julia 1.12.7)</text>
<line x1="220.0" y1="40" x2="220.0" y2="243" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="220.0" y="258" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.0 μs</text>
<line x1="315.0" y1="40" x2="315.0" y2="243" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="315.0" y="258" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">75.0 μs</text>
<line x1="410.0" y1="40" x2="410.0" y2="243" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="410.0" y="258" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">150.0 μs</text>
<line x1="505.0" y1="40" x2="505.0" y2="243" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="505.0" y="258" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">225.0 μs</text>
<line x1="600.0" y1="40" x2="600.0" y2="243" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="600.0" y="258" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">300.0 μs</text>
<text x="210" y="79.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">Dcₓ</text>
<rect x="220" y="60" width="325.7968" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 257.2 μs</title></rect>
<text x="551.7968000000001" y="71" font-size="11" fill="currentColor" opacity="0.85">257.2 μs</text>
<rect x="220" y="77" width="324.9532" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 256.5 μs</title></rect>
<text x="550.9531999999999" y="88" font-size="11" fill="currentColor" opacity="0.85">256.5 μs</text>
<text x="210" y="126.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">D₋ᵧ</text>
<rect x="220" y="107" width="204.4615333333333" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 161.4 μs</title></rect>
<text x="430.4615333333333" y="118" font-size="11" fill="currentColor" opacity="0.85">161.4 μs</text>
<rect x="220" y="124" width="205.99166666666667" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 162.6 μs</title></rect>
<text x="431.9916666666667" y="135" font-size="11" fill="currentColor" opacity="0.85">162.6 μs</text>
<text x="210" y="173.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">D₋ₓ</text>
<rect x="220" y="154" width="258.0314" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 203.7 μs</title></rect>
<text x="484.0314" y="165" font-size="11" fill="currentColor" opacity="0.85">203.7 μs</text>
<rect x="220" y="171" width="257.50256666666667" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 203.3 μs</title></rect>
<text x="483.50256666666667" y="182" font-size="11" fill="currentColor" opacity="0.85">203.3 μs</text>
<text x="210" y="220.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">M₋ₓ</text>
<rect x="220" y="201" width="217.12693333333334" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 171.4 μs</title></rect>
<text x="443.12693333333334" y="212" font-size="11" fill="currentColor" opacity="0.85">171.4 μs</text>
<rect x="220" y="218" width="216.65193333333332" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 171.0 μs</title></rect>
<text x="442.6519333333333" y="229" font-size="11" fill="currentColor" opacity="0.85">171.0 μs</text>
</svg></div>

```

### Operators 3D

| Benchmark | `0b9a62b` (ref) Time | Allocs | Memory | `855fbf5` Time | Allocs | Memory |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`D₋₂`** | 200.9 μs | 3 | 7.64 MiB | 222.8 μs (**+10.9%** 🔴) | 3 | 7.64 MiB |
| **`innerₕ`** | 240.2 μs | 0 | 0 B | 240.0 μs (=) | 0 | 0 B |
| **`∇₋ₕ`** | 694.1 μs | 15 | 22.92 MiB | 686.6 μs (**-1.1%** 🟢) | 15 | 22.92 MiB |

```@raw html
<div style="width:100%; max-width:710px; margin:1.5em auto; overflow-x:auto; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:1em;">
<svg viewBox="0 0 710 226" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<rect x="220" y="15" width="12" height="12" rx="2" fill="#3b82f6" />
<text x="236" y="25" font-size="12" fill="currentColor" opacity="0.9">0b9a62b (Julia 1.12.7)</text>
<rect x="400" y="15" width="12" height="12" rx="2" fill="#10b981" />
<text x="416" y="25" font-size="12" fill="currentColor" opacity="0.9">855fbf5 (Julia 1.12.7)</text>
<line x1="220.0" y1="40" x2="220.0" y2="196" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="220.0" y="211" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.0 μs</text>
<line x1="315.0" y1="40" x2="315.0" y2="196" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="315.0" y="211" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">250.0 μs</text>
<line x1="410.0" y1="40" x2="410.0" y2="196" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="410.0" y="211" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">500.0 μs</text>
<line x1="505.0" y1="40" x2="505.0" y2="196" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="505.0" y="211" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">750.0 μs</text>
<line x1="600.0" y1="40" x2="600.0" y2="196" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="600.0" y="211" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">1000.0 μs</text>
<text x="210" y="79.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">D₋₂</text>
<rect x="220" y="60" width="76.34846" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 200.9 μs</title></rect>
<text x="302.34846" y="71" font-size="11" fill="currentColor" opacity="0.85">200.9 μs</text>
<rect x="220" y="77" width="84.66058" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 222.8 μs</title></rect>
<text x="310.66058" y="88" font-size="11" fill="currentColor" opacity="0.85">222.8 μs</text>
<text x="210" y="126.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">innerₕ</text>
<rect x="220" y="107" width="91.295" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 240.2 μs</title></rect>
<text x="317.295" y="118" font-size="11" fill="currentColor" opacity="0.85">240.2 μs</text>
<rect x="220" y="124" width="91.2" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 240.0 μs</title></rect>
<text x="317.2" y="135" font-size="11" fill="currentColor" opacity="0.85">240.0 μs</text>
<text x="210" y="173.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">∇₋ₕ</text>
<rect x="220" y="154" width="263.75192" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 694.1 μs</title></rect>
<text x="489.75192" y="165" font-size="11" fill="currentColor" opacity="0.85">694.1 μs</text>
<rect x="220" y="171" width="260.9175" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 686.6 μs</title></rect>
<text x="486.9175" y="182" font-size="11" fill="currentColor" opacity="0.85">686.6 μs</text>
</svg></div>

```

### Jumps & Averages

| Benchmark | `0b9a62b` (ref) Time | Allocs | Memory | `855fbf5` Time | Allocs | Memory |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`M₊ᵧ 2D`** | — | — | — | 161.6 μs | 3 | 7.64 MiB |
| **`M₊₂ 3D`** | — | — | — | 227.8 μs | 3 | 7.64 MiB |
| **`M₊ₓ 2D`** | — | — | — | 160.4 μs | 3 | 7.64 MiB |
| **`jumpᵧ 2D`** | — | — | — | 162.0 μs | 3 | 7.64 MiB |
| **`jump₂ 3D`** | — | — | — | 227.6 μs | 3 | 7.64 MiB |
| **`jumpₓ 2D`** | — | — | — | 164.7 μs | 3 | 7.64 MiB |

```@raw html
<div style="width:100%; max-width:710px; margin:1.5em auto; overflow-x:auto; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:1em;">
<svg viewBox="0 0 710 367" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<rect x="220" y="15" width="12" height="12" rx="2" fill="#3b82f6" />
<text x="236" y="25" font-size="12" fill="currentColor" opacity="0.9">0b9a62b (Julia 1.12.7)</text>
<rect x="400" y="15" width="12" height="12" rx="2" fill="#10b981" />
<text x="416" y="25" font-size="12" fill="currentColor" opacity="0.9">855fbf5 (Julia 1.12.7)</text>
<line x1="220.0" y1="40" x2="220.0" y2="337" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="220.0" y="352" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.0 μs</text>
<line x1="315.0" y1="40" x2="315.0" y2="337" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="315.0" y="352" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">75.0 μs</text>
<line x1="410.0" y1="40" x2="410.0" y2="337" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="410.0" y="352" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">150.0 μs</text>
<line x1="505.0" y1="40" x2="505.0" y2="337" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="505.0" y="352" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">225.0 μs</text>
<line x1="600.0" y1="40" x2="600.0" y2="337" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="600.0" y="352" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">300.0 μs</text>
<text x="210" y="79.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">M₊ᵧ 2D</text>
<text x="220" y="71" font-size="11" fill="currentColor" opacity="0.4">—</text>
<rect x="220" y="77" width="204.725" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 161.6 μs</title></rect>
<text x="430.725" y="88" font-size="11" fill="currentColor" opacity="0.85">161.6 μs</text>
<text x="210" y="126.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">M₊₂ 3D</text>
<text x="220" y="118" font-size="11" fill="currentColor" opacity="0.4">—</text>
<rect x="220" y="124" width="288.5884666666667" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 227.8 μs</title></rect>
<text x="514.5884666666667" y="135" font-size="11" fill="currentColor" opacity="0.85">227.8 μs</text>
<text x="210" y="173.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">M₊ₓ 2D</text>
<text x="220" y="165" font-size="11" fill="currentColor" opacity="0.4">—</text>
<rect x="220" y="171" width="203.19486666666666" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 160.4 μs</title></rect>
<text x="429.19486666666666" y="182" font-size="11" fill="currentColor" opacity="0.85">160.4 μs</text>
<text x="210" y="220.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">jumpᵧ 2D</text>
<text x="220" y="212" font-size="11" fill="currentColor" opacity="0.4">—</text>
<rect x="220" y="218" width="205.20000000000002" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 162.0 μs</title></rect>
<text x="431.20000000000005" y="229" font-size="11" fill="currentColor" opacity="0.85">162.0 μs</text>
<text x="210" y="267.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">jump₂ 3D</text>
<text x="220" y="259" font-size="11" fill="currentColor" opacity="0.4">—</text>
<rect x="220" y="265" width="288.325" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 227.6 μs</title></rect>
<text x="514.325" y="276" font-size="11" fill="currentColor" opacity="0.85">227.6 μs</text>
<text x="210" y="314.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">jumpₓ 2D</text>
<text x="220" y="306" font-size="11" fill="currentColor" opacity="0.4">—</text>
<rect x="220" y="312" width="208.5782" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 164.7 μs</title></rect>
<text x="434.57820000000004" y="323" font-size="11" fill="currentColor" opacity="0.85">164.7 μs</text>
</svg></div>

```

### Inner Products 2D

| Benchmark | `0b9a62b` (ref) Time | Allocs | Memory | `855fbf5` Time | Allocs | Memory |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`innerₕ`** | 242.0 μs | 0 | 0 B | 238.5 μs (**-1.5%** 🟢) | 0 | 0 B |
| **`norm₁ₕ`** | 790.2 μs | 0 | 0 B | 782.9 μs (**-0.9%** 🟢) | 0 | 0 B |
| **`normₕ`** | 190.0 μs | 0 | 0 B | 186.8 μs (**-1.7%** 🟢) | 0 | 0 B |
| **`snorm₁ₕ`** | 578.1 μs | 0 | 0 B | 577.1 μs (=) | 0 | 0 B |

```@raw html
<div style="width:100%; max-width:710px; margin:1.5em auto; overflow-x:auto; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:1em;">
<svg viewBox="0 0 710 273" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<rect x="220" y="15" width="12" height="12" rx="2" fill="#3b82f6" />
<text x="236" y="25" font-size="12" fill="currentColor" opacity="0.9">0b9a62b (Julia 1.12.7)</text>
<rect x="400" y="15" width="12" height="12" rx="2" fill="#10b981" />
<text x="416" y="25" font-size="12" fill="currentColor" opacity="0.9">855fbf5 (Julia 1.12.7)</text>
<line x1="220.0" y1="40" x2="220.0" y2="243" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="220.0" y="258" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.0 μs</text>
<line x1="315.0" y1="40" x2="315.0" y2="243" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="315.0" y="258" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">250.0 μs</text>
<line x1="410.0" y1="40" x2="410.0" y2="243" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="410.0" y="258" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">500.0 μs</text>
<line x1="505.0" y1="40" x2="505.0" y2="243" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="505.0" y="258" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">750.0 μs</text>
<line x1="600.0" y1="40" x2="600.0" y2="243" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="600.0" y="258" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">1000.0 μs</text>
<text x="210" y="79.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">innerₕ</text>
<rect x="220" y="60" width="91.97558000000001" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 242.0 μs</title></rect>
<text x="317.97558000000004" y="71" font-size="11" fill="currentColor" opacity="0.85">242.0 μs</text>
<rect x="220" y="77" width="90.61404" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 238.5 μs</title></rect>
<text x="316.61404" y="88" font-size="11" fill="currentColor" opacity="0.85">238.5 μs</text>
<text x="210" y="126.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">norm₁ₕ</text>
<rect x="220" y="107" width="300.295" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 790.2 μs</title></rect>
<text x="526.2950000000001" y="118" font-size="11" fill="currentColor" opacity="0.85">790.2 μs</text>
<rect x="220" y="124" width="297.50846" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 782.9 μs</title></rect>
<text x="523.50846" y="135" font-size="11" fill="currentColor" opacity="0.85">782.9 μs</text>
<text x="210" y="173.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">normₕ</text>
<rect x="220" y="154" width="72.2" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 190.0 μs</title></rect>
<text x="298.2" y="165" font-size="11" fill="currentColor" opacity="0.85">190.0 μs</text>
<rect x="220" y="171" width="70.99692" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 186.8 μs</title></rect>
<text x="296.99692" y="182" font-size="11" fill="currentColor" opacity="0.85">186.8 μs</text>
<text x="210" y="220.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">snorm₁ₕ</text>
<rect x="220" y="201" width="219.67154" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 578.1 μs</title></rect>
<text x="445.67154" y="212" font-size="11" fill="currentColor" opacity="0.85">578.1 μs</text>
<rect x="220" y="218" width="219.29191999999998" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 577.1 μs</title></rect>
<text x="445.29192" y="229" font-size="11" fill="currentColor" opacity="0.85">577.1 μs</text>
</svg></div>

```

### Restriction

| Benchmark | `0b9a62b` (ref) Time | Allocs | Memory | `855fbf5` Time | Allocs | Memory |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`Rₕ 1D (allocates its output)`** | 2.87 ms | 6 | 7.64 MiB | 2.88 ms (=) | 6 | 7.64 MiB |
| **`Rₕ! 1D`** | 2.87 ms | 3 | 64 B | 2.88 ms (**+0.5%** 🔴) | 3 | 64 B |
| **`avgₕ! 1D`** | 16.27 ms | 2 | 128 B | 16.37 ms (**+0.6%** 🔴) | 2 | 128 B |

```@raw html
<div style="width:100%; max-width:710px; margin:1.5em auto; overflow-x:auto; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:1em;">
<svg viewBox="0 0 710 226" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<rect x="220" y="15" width="12" height="12" rx="2" fill="#3b82f6" />
<text x="236" y="25" font-size="12" fill="currentColor" opacity="0.9">0b9a62b (Julia 1.12.7)</text>
<rect x="400" y="15" width="12" height="12" rx="2" fill="#10b981" />
<text x="416" y="25" font-size="12" fill="currentColor" opacity="0.9">855fbf5 (Julia 1.12.7)</text>
<line x1="220.0" y1="40" x2="220.0" y2="196" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="220.0" y="211" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.0 ms</text>
<line x1="315.0" y1="40" x2="315.0" y2="196" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="315.0" y="211" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">5.0 ms</text>
<line x1="410.0" y1="40" x2="410.0" y2="196" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="410.0" y="211" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">10.0 ms</text>
<line x1="505.0" y1="40" x2="505.0" y2="196" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="505.0" y="211" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">15.0 ms</text>
<line x1="600.0" y1="40" x2="600.0" y2="196" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="600.0" y="211" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">20.0 ms</text>
<text x="210" y="79.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">Rₕ 1D (allocates its output)</text>
<rect x="220" y="60" width="54.584625" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 2.9 ms</title></rect>
<text x="280.584625" y="71" font-size="11" fill="currentColor" opacity="0.85">2.9 ms</text>
<rect x="220" y="77" width="54.693875000000006" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 2.9 ms</title></rect>
<text x="280.693875" y="88" font-size="11" fill="currentColor" opacity="0.85">2.9 ms</text>
<text x="210" y="126.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">Rₕ! 1D</text>
<rect x="220" y="107" width="54.45875" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 2.9 ms</title></rect>
<text x="280.45875" y="118" font-size="11" fill="currentColor" opacity="0.85">2.9 ms</text>
<rect x="220" y="124" width="54.7097115" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 2.9 ms</title></rect>
<text x="280.7097115" y="135" font-size="11" fill="currentColor" opacity="0.85">2.9 ms</text>
<text x="210" y="173.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">avgₕ! 1D</text>
<rect x="220" y="154" width="309.216298" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 16.3 ms</title></rect>
<text x="535.216298" y="165" font-size="11" fill="currentColor" opacity="0.85">16.3 ms</text>
<rect x="220" y="171" width="310.96547599999997" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 16.4 ms</title></rect>
<text x="536.965476" y="182" font-size="11" fill="currentColor" opacity="0.85">16.4 ms</text>
</svg></div>

```

### Composite

| Benchmark | `0b9a62b` (ref) Time | Allocs | Memory | `855fbf5` Time | Allocs | Memory |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`D₋ₓ (3 components)`** | 711.0 μs | 3 | 22.89 MiB | 675.9 μs (**-4.9%** 🟢) | 3 | 22.89 MiB |
| **`∇₋ₕ (3 components)`** | 1.43 ms | 10 | 45.78 MiB | 1.4 ms (**-2.0%** 🟢) | 10 | 45.78 MiB |

```@raw html
<div style="width:100%; max-width:710px; margin:1.5em auto; overflow-x:auto; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:1em;">
<svg viewBox="0 0 710 179" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<rect x="220" y="15" width="12" height="12" rx="2" fill="#3b82f6" />
<text x="236" y="25" font-size="12" fill="currentColor" opacity="0.9">0b9a62b (Julia 1.12.7)</text>
<rect x="400" y="15" width="12" height="12" rx="2" fill="#10b981" />
<text x="416" y="25" font-size="12" fill="currentColor" opacity="0.9">855fbf5 (Julia 1.12.7)</text>
<line x1="220.0" y1="40" x2="220.0" y2="149" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="220.0" y="164" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.0 ms</text>
<line x1="315.0" y1="40" x2="315.0" y2="149" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="315.0" y="164" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.4 ms</text>
<line x1="410.0" y1="40" x2="410.0" y2="149" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="410.0" y="164" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.8 ms</text>
<line x1="505.0" y1="40" x2="505.0" y2="149" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="505.0" y="164" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">1.1 ms</text>
<line x1="600.0" y1="40" x2="600.0" y2="149" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="600.0" y="164" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">1.5 ms</text>
<text x="210" y="79.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">D₋ₓ (3 components)</text>
<rect x="220" y="60" width="180.13063999999997" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 0.7 ms</title></rect>
<text x="406.13063999999997" y="71" font-size="11" fill="currentColor" opacity="0.85">0.7 ms</text>
<rect x="220" y="77" width="171.22166666666666" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 0.7 ms</title></rect>
<text x="397.2216666666667" y="88" font-size="11" fill="currentColor" opacity="0.85">0.7 ms</text>
<text x="210" y="126.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">∇₋ₕ (3 components)</text>
<rect x="220" y="107" width="361.59102666666666" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 1.4 ms</title></rect>
<text x="587.5910266666667" y="118" font-size="11" fill="currentColor" opacity="0.85">1.4 ms</text>
<rect x="220" y="124" width="354.24436" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 1.4 ms</title></rect>
<text x="580.2443599999999" y="135" font-size="11" fill="currentColor" opacity="0.85">1.4 ms</text>
</svg></div>

```

### Construction

| Benchmark | `0b9a62b` (ref) Time | Allocs | Memory | `855fbf5` Time | Allocs | Memory |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`gridspace 2D`** | 368.8 μs | 38 | 30.59 MiB | 360.0 μs (**-2.4%** 🟢) | 38 | 30.59 MiB |
| **`gridspace 3D`** | 1.63 ms | 44 | 38.21 MiB | 1.64 ms (=) | 44 | 38.21 MiB |
| **`hₘₐₓ 3D`** | 153.0 ns | 0 | 0 B | 153.1 ns (=) | 0 | 0 B |

```@raw html
<div style="width:100%; max-width:710px; margin:1.5em auto; overflow-x:auto; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:1em;">
<svg viewBox="0 0 710 226" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<rect x="220" y="15" width="12" height="12" rx="2" fill="#3b82f6" />
<text x="236" y="25" font-size="12" fill="currentColor" opacity="0.9">0b9a62b (Julia 1.12.7)</text>
<rect x="400" y="15" width="12" height="12" rx="2" fill="#10b981" />
<text x="416" y="25" font-size="12" fill="currentColor" opacity="0.9">855fbf5 (Julia 1.12.7)</text>
<line x1="220.0" y1="40" x2="220.0" y2="196" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="220.0" y="211" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.0 ms</text>
<line x1="315.0" y1="40" x2="315.0" y2="196" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="315.0" y="211" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.5 ms</text>
<line x1="410.0" y1="40" x2="410.0" y2="196" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="410.0" y="211" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">1.0 ms</text>
<line x1="505.0" y1="40" x2="505.0" y2="196" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="505.0" y="211" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">1.5 ms</text>
<line x1="600.0" y1="40" x2="600.0" y2="196" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="600.0" y="211" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">2.0 ms</text>
<text x="210" y="79.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">gridspace 2D</text>
<rect x="220" y="60" width="70.07428" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 0.4 ms</title></rect>
<text x="296.07428" y="71" font-size="11" fill="currentColor" opacity="0.85">0.4 ms</text>
<rect x="220" y="77" width="68.40798" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 0.4 ms</title></rect>
<text x="294.40798" y="88" font-size="11" fill="currentColor" opacity="0.85">0.4 ms</text>
<text x="210" y="126.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">gridspace 3D</text>
<rect x="220" y="107" width="310.30952" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 1.6 ms</title></rect>
<text x="536.30952" y="118" font-size="11" fill="currentColor" opacity="0.85">1.6 ms</text>
<rect x="220" y="124" width="310.91923" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 1.6 ms</title></rect>
<text x="536.91923" y="135" font-size="11" fill="currentColor" opacity="0.85">1.6 ms</text>
<text x="210" y="173.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">hₘₐₓ 3D</text>
<rect x="220" y="154" width="2.0" height="14" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 0.0 ms</title></rect>
<text x="228.0" y="165" font-size="11" fill="currentColor" opacity="0.85">0.0 ms</text>
<rect x="220" y="171" width="2.0" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 0.0 ms</title></rect>
<text x="228.0" y="182" font-size="11" fill="currentColor" opacity="0.85">0.0 ms</text>
</svg></div>

```

### Startup & Latency

| Benchmark | `0b9a62b` (ref) Time | Allocs | Memory | `855fbf5` Time | Allocs | Memory |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`TTFX (load + first operator)`** | — | — | — | 560.78 ms | 45 | 1.3 KiB |
| **`using Bramble`** | — | — | — | 527.5 ms | 45 | 1.3 KiB |

```@raw html
<div style="width:100%; max-width:710px; margin:1.5em auto; overflow-x:auto; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:1em;">
<svg viewBox="0 0 710 179" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<rect x="220" y="15" width="12" height="12" rx="2" fill="#3b82f6" />
<text x="236" y="25" font-size="12" fill="currentColor" opacity="0.9">0b9a62b (Julia 1.12.7)</text>
<rect x="400" y="15" width="12" height="12" rx="2" fill="#10b981" />
<text x="416" y="25" font-size="12" fill="currentColor" opacity="0.9">855fbf5 (Julia 1.12.7)</text>
<line x1="220.0" y1="40" x2="220.0" y2="149" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="220.0" y="164" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.0 ms</text>
<line x1="315.0" y1="40" x2="315.0" y2="149" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="315.0" y="164" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">250.0 ms</text>
<line x1="410.0" y1="40" x2="410.0" y2="149" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="410.0" y="164" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">500.0 ms</text>
<line x1="505.0" y1="40" x2="505.0" y2="149" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="505.0" y="164" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">750.0 ms</text>
<line x1="600.0" y1="40" x2="600.0" y2="149" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="600.0" y="164" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">1000.0 ms</text>
<text x="210" y="79.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">TTFX (load + first operator)</text>
<text x="220" y="71" font-size="11" fill="currentColor" opacity="0.4">—</text>
<rect x="220" y="77" width="213.09519654000002" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 560.8 ms</title></rect>
<text x="439.09519654" y="88" font-size="11" fill="currentColor" opacity="0.85">560.8 ms</text>
<text x="210" y="126.5" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">using Bramble</text>
<text x="220" y="118" font-size="11" fill="currentColor" opacity="0.4">—</text>
<rect x="220" y="124" width="200.45011095999996" height="14" rx="3" fill="#10b981" opacity="0.9">
<title>855fbf5 (Julia 1.12.7): 527.5 ms</title></rect>
<text x="426.45011095999996" y="135" font-size="11" fill="currentColor" opacity="0.85">527.5 ms</text>
</svg></div>

```

## How to Add New Benchmark Runs

To record performance on a new commit or after an optimization pass, run:

```bash
julia --project=benchmark benchmark/benchmarks.jl --save benchmark/baselines/baseline_$(git rev-parse --short HEAD).json
```

Rebuilding the documentation (`julia -e 'using Pkg; Pkg.activate("docs"); include("docs/make.jl")'`) will automatically discover all `baseline_*.json` files and append new comparison columns, delta calculations, and charts.
