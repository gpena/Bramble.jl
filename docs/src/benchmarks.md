# Performance & Benchmarks

Bramble tracks memory allocations and performance regressions with a dedicated regression suite in `benchmark/benchmarks.jl`.
All measurements below are run on **1,000,000 grid points** per dimension setup (e.g. ``1000 \times 1000`` in 2D, ``100 \times 100 \times 100`` in 3D).

## Recorded Baselines

| Commit | Julia | Summary | File |
|---|:---:|---|---|
| `0b9a62b` | `1.12.7` | test: run the allocation assertions under coverage instead of skipping them | `baseline_0b9a62b.json` |

## Comparative Timings & Allocations

### Operators 2D

| Benchmark | `0b9a62b` Time | Allocs | Memory |
|---|:---:|:---:|:---:|
| **`Dcₓ`** | 257.2 μs | 3 | 7.64 MiB |
| **`D₋ᵧ`** | 161.4 μs | 3 | 7.64 MiB |
| **`D₋ₓ`** | 203.7 μs | 3 | 7.64 MiB |
| **`M₋ₓ`** | 171.4 μs | 3 | 7.64 MiB |

```@raw html
<div style="width:100%; max-width:710px; margin:1.5em auto; overflow-x:auto; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:1em;">
<svg viewBox="0 0 710 221" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<rect x="220" y="15" width="12" height="12" rx="2" fill="#3b82f6" />
<text x="236" y="25" font-size="12" fill="currentColor" opacity="0.9">0b9a62b (Julia 1.12.7)</text>
<line x1="220.0" y1="40" x2="220.0" y2="191" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="220.0" y="206" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.0 μs</text>
<line x1="315.0" y1="40" x2="315.0" y2="191" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="315.0" y="206" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">75.0 μs</text>
<line x1="410.0" y1="40" x2="410.0" y2="191" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="410.0" y="206" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">150.0 μs</text>
<line x1="505.0" y1="40" x2="505.0" y2="191" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="505.0" y="206" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">225.0 μs</text>
<line x1="600.0" y1="40" x2="600.0" y2="191" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="600.0" y="206" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">300.0 μs</text>
<text x="210" y="73.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">Dcₓ</text>
<rect x="220" y="60" width="325.7968" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 257.2 μs</title></rect>
<text x="551.7968000000001" y="75" font-size="11" fill="currentColor" opacity="0.85">257.2 μs</text>
<text x="210" y="107.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">D₋ᵧ</text>
<rect x="220" y="94" width="204.4615333333333" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 161.4 μs</title></rect>
<text x="430.4615333333333" y="109" font-size="11" fill="currentColor" opacity="0.85">161.4 μs</text>
<text x="210" y="141.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">D₋ₓ</text>
<rect x="220" y="128" width="258.0314" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 203.7 μs</title></rect>
<text x="484.0314" y="143" font-size="11" fill="currentColor" opacity="0.85">203.7 μs</text>
<text x="210" y="175.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">M₋ₓ</text>
<rect x="220" y="162" width="217.12693333333334" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 171.4 μs</title></rect>
<text x="443.12693333333334" y="177" font-size="11" fill="currentColor" opacity="0.85">171.4 μs</text>
</svg></div>

```

### Operators 3D

| Benchmark | `0b9a62b` Time | Allocs | Memory |
|---|:---:|:---:|:---:|
| **`D₋₂`** | 200.9 μs | 3 | 7.64 MiB |
| **`innerₕ`** | 240.2 μs | 0 | 0 B |
| **`∇₋ₕ`** | 694.1 μs | 15 | 22.92 MiB |

```@raw html
<div style="width:100%; max-width:710px; margin:1.5em auto; overflow-x:auto; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:1em;">
<svg viewBox="0 0 710 187" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<rect x="220" y="15" width="12" height="12" rx="2" fill="#3b82f6" />
<text x="236" y="25" font-size="12" fill="currentColor" opacity="0.9">0b9a62b (Julia 1.12.7)</text>
<line x1="220.0" y1="40" x2="220.0" y2="157" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="220.0" y="172" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.0 μs</text>
<line x1="315.0" y1="40" x2="315.0" y2="157" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="315.0" y="172" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">250.0 μs</text>
<line x1="410.0" y1="40" x2="410.0" y2="157" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="410.0" y="172" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">500.0 μs</text>
<line x1="505.0" y1="40" x2="505.0" y2="157" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="505.0" y="172" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">750.0 μs</text>
<line x1="600.0" y1="40" x2="600.0" y2="157" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="600.0" y="172" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">1000.0 μs</text>
<text x="210" y="73.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">D₋₂</text>
<rect x="220" y="60" width="76.34846" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 200.9 μs</title></rect>
<text x="302.34846" y="75" font-size="11" fill="currentColor" opacity="0.85">200.9 μs</text>
<text x="210" y="107.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">innerₕ</text>
<rect x="220" y="94" width="91.295" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 240.2 μs</title></rect>
<text x="317.295" y="109" font-size="11" fill="currentColor" opacity="0.85">240.2 μs</text>
<text x="210" y="141.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">∇₋ₕ</text>
<rect x="220" y="128" width="263.75192" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 694.1 μs</title></rect>
<text x="489.75192" y="143" font-size="11" fill="currentColor" opacity="0.85">694.1 μs</text>
</svg></div>

```

### Inner Products 2D

| Benchmark | `0b9a62b` Time | Allocs | Memory |
|---|:---:|:---:|:---:|
| **`innerₕ`** | 242.0 μs | 0 | 0 B |
| **`norm₁ₕ`** | 790.2 μs | 0 | 0 B |
| **`normₕ`** | 190.0 μs | 0 | 0 B |
| **`snorm₁ₕ`** | 578.1 μs | 0 | 0 B |

```@raw html
<div style="width:100%; max-width:710px; margin:1.5em auto; overflow-x:auto; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:1em;">
<svg viewBox="0 0 710 221" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<rect x="220" y="15" width="12" height="12" rx="2" fill="#3b82f6" />
<text x="236" y="25" font-size="12" fill="currentColor" opacity="0.9">0b9a62b (Julia 1.12.7)</text>
<line x1="220.0" y1="40" x2="220.0" y2="191" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="220.0" y="206" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.0 μs</text>
<line x1="315.0" y1="40" x2="315.0" y2="191" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="315.0" y="206" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">250.0 μs</text>
<line x1="410.0" y1="40" x2="410.0" y2="191" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="410.0" y="206" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">500.0 μs</text>
<line x1="505.0" y1="40" x2="505.0" y2="191" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="505.0" y="206" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">750.0 μs</text>
<line x1="600.0" y1="40" x2="600.0" y2="191" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="600.0" y="206" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">1000.0 μs</text>
<text x="210" y="73.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">innerₕ</text>
<rect x="220" y="60" width="91.97558000000001" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 242.0 μs</title></rect>
<text x="317.97558000000004" y="75" font-size="11" fill="currentColor" opacity="0.85">242.0 μs</text>
<text x="210" y="107.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">norm₁ₕ</text>
<rect x="220" y="94" width="300.295" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 790.2 μs</title></rect>
<text x="526.2950000000001" y="109" font-size="11" fill="currentColor" opacity="0.85">790.2 μs</text>
<text x="210" y="141.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">normₕ</text>
<rect x="220" y="128" width="72.2" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 190.0 μs</title></rect>
<text x="298.2" y="143" font-size="11" fill="currentColor" opacity="0.85">190.0 μs</text>
<text x="210" y="175.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">snorm₁ₕ</text>
<rect x="220" y="162" width="219.67154" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 578.1 μs</title></rect>
<text x="445.67154" y="177" font-size="11" fill="currentColor" opacity="0.85">578.1 μs</text>
</svg></div>

```

### Restriction

| Benchmark | `0b9a62b` Time | Allocs | Memory |
|---|:---:|:---:|:---:|
| **`Rₕ 1D (allocates its output)`** | 2.87 ms | 6 | 7.64 MiB |
| **`Rₕ! 1D`** | 2.87 ms | 3 | 64 B |
| **`avgₕ! 1D`** | 16.27 ms | 2 | 128 B |

```@raw html
<div style="width:100%; max-width:710px; margin:1.5em auto; overflow-x:auto; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:1em;">
<svg viewBox="0 0 710 187" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<rect x="220" y="15" width="12" height="12" rx="2" fill="#3b82f6" />
<text x="236" y="25" font-size="12" fill="currentColor" opacity="0.9">0b9a62b (Julia 1.12.7)</text>
<line x1="220.0" y1="40" x2="220.0" y2="157" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="220.0" y="172" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.0 ms</text>
<line x1="315.0" y1="40" x2="315.0" y2="157" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="315.0" y="172" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">5.0 ms</text>
<line x1="410.0" y1="40" x2="410.0" y2="157" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="410.0" y="172" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">10.0 ms</text>
<line x1="505.0" y1="40" x2="505.0" y2="157" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="505.0" y="172" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">15.0 ms</text>
<line x1="600.0" y1="40" x2="600.0" y2="157" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="600.0" y="172" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">20.0 ms</text>
<text x="210" y="73.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">Rₕ 1D (allocates its output)</text>
<rect x="220" y="60" width="54.584625" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 2.9 ms</title></rect>
<text x="280.584625" y="75" font-size="11" fill="currentColor" opacity="0.85">2.9 ms</text>
<text x="210" y="107.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">Rₕ! 1D</text>
<rect x="220" y="94" width="54.45875" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 2.9 ms</title></rect>
<text x="280.45875" y="109" font-size="11" fill="currentColor" opacity="0.85">2.9 ms</text>
<text x="210" y="141.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">avgₕ! 1D</text>
<rect x="220" y="128" width="309.216298" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 16.3 ms</title></rect>
<text x="535.216298" y="143" font-size="11" fill="currentColor" opacity="0.85">16.3 ms</text>
</svg></div>

```

### Composite

| Benchmark | `0b9a62b` Time | Allocs | Memory |
|---|:---:|:---:|:---:|
| **`D₋ₓ (3 components)`** | 711.0 μs | 3 | 22.89 MiB |
| **`∇₋ₕ (3 components)`** | 1.43 ms | 10 | 45.78 MiB |

```@raw html
<div style="width:100%; max-width:710px; margin:1.5em auto; overflow-x:auto; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:1em;">
<svg viewBox="0 0 710 153" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<rect x="220" y="15" width="12" height="12" rx="2" fill="#3b82f6" />
<text x="236" y="25" font-size="12" fill="currentColor" opacity="0.9">0b9a62b (Julia 1.12.7)</text>
<line x1="220.0" y1="40" x2="220.0" y2="123" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="220.0" y="138" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.0 ms</text>
<line x1="315.0" y1="40" x2="315.0" y2="123" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="315.0" y="138" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.4 ms</text>
<line x1="410.0" y1="40" x2="410.0" y2="123" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="410.0" y="138" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.8 ms</text>
<line x1="505.0" y1="40" x2="505.0" y2="123" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="505.0" y="138" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">1.1 ms</text>
<line x1="600.0" y1="40" x2="600.0" y2="123" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="600.0" y="138" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">1.5 ms</text>
<text x="210" y="73.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">D₋ₓ (3 components)</text>
<rect x="220" y="60" width="180.13063999999997" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 0.7 ms</title></rect>
<text x="406.13063999999997" y="75" font-size="11" fill="currentColor" opacity="0.85">0.7 ms</text>
<text x="210" y="107.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">∇₋ₕ (3 components)</text>
<rect x="220" y="94" width="361.59102666666666" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 1.4 ms</title></rect>
<text x="587.5910266666667" y="109" font-size="11" fill="currentColor" opacity="0.85">1.4 ms</text>
</svg></div>

```

### Construction

| Benchmark | `0b9a62b` Time | Allocs | Memory |
|---|:---:|:---:|:---:|
| **`gridspace 2D`** | 368.8 μs | 38 | 30.59 MiB |
| **`gridspace 3D`** | 1.63 ms | 44 | 38.21 MiB |
| **`hₘₐₓ 3D`** | 153.0 ns | 0 | 0 B |

```@raw html
<div style="width:100%; max-width:710px; margin:1.5em auto; overflow-x:auto; background:var(--documenter-bg, #fff); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:1em;">
<svg viewBox="0 0 710 187" width="100%" style="font-family:-apple-system, BlinkMacSystemFont, juliamono, monospace; display:block;">
<rect x="220" y="15" width="12" height="12" rx="2" fill="#3b82f6" />
<text x="236" y="25" font-size="12" fill="currentColor" opacity="0.9">0b9a62b (Julia 1.12.7)</text>
<line x1="220.0" y1="40" x2="220.0" y2="157" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="220.0" y="172" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.0 ms</text>
<line x1="315.0" y1="40" x2="315.0" y2="157" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="315.0" y="172" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">0.5 ms</text>
<line x1="410.0" y1="40" x2="410.0" y2="157" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="410.0" y="172" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">1.0 ms</text>
<line x1="505.0" y1="40" x2="505.0" y2="157" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="505.0" y="172" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">1.5 ms</text>
<line x1="600.0" y1="40" x2="600.0" y2="157" stroke="rgba(128,128,128,0.2)" stroke-dasharray="3,3" />
<text x="600.0" y="172" font-size="11" fill="currentColor" opacity="0.6" text-anchor="middle">2.0 ms</text>
<text x="210" y="73.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">gridspace 2D</text>
<rect x="220" y="60" width="70.07428" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 0.4 ms</title></rect>
<text x="296.07428" y="75" font-size="11" fill="currentColor" opacity="0.85">0.4 ms</text>
<text x="210" y="107.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">gridspace 3D</text>
<rect x="220" y="94" width="310.30952" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 1.6 ms</title></rect>
<text x="536.30952" y="109" font-size="11" fill="currentColor" opacity="0.85">1.6 ms</text>
<text x="210" y="141.0" font-size="12" font-weight="bold" fill="currentColor" text-anchor="end">hₘₐₓ 3D</text>
<rect x="220" y="128" width="2.0" height="18" rx="3" fill="#3b82f6" opacity="0.9">
<title>0b9a62b (Julia 1.12.7): 0.0 ms</title></rect>
<text x="228.0" y="143" font-size="11" fill="currentColor" opacity="0.85">0.0 ms</text>
</svg></div>

```

## How to Add New Benchmark Runs

To record performance on a new commit or after an optimization pass, run:

```bash
julia --project=benchmark benchmark/benchmarks.jl --save benchmark/baselines/baseline_$(git rev-parse --short HEAD).json
```

Rebuilding the documentation (`julia -e 'using Pkg; Pkg.activate("docs"); include("docs/make.jl")'`) will automatically discover all `baseline_*.json` files and append new comparison columns, delta calculations, and charts.
