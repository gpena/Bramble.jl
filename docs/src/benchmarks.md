# Performance & Benchmarks

Bramble tracks memory allocations and performance regressions with a dedicated regression suite in `benchmark/benchmarks.jl`.
All measurements below are run on **1,000,000 grid points** per dimension setup (e.g. ``1000 \times 1000`` in 2D, ``100 \times 100 \times 100`` in 3D).

## Recorded Baselines

| Commit | Julia | Date | Summary | File |
|---|:---:|:---:|---|---|
| `0b9a62b` | `1.12.7` | 2026-08-30 | test: run the allocation assertions under coverage instead of skipping them | `baseline_0b9a62b.json` |

## Comparative Timings & Allocations

### Operators 2D

| Benchmark | `0b9a62b` Time | Allocs | Memory |
|---|:---:|:---:|:---:|
| **`Dcₓ`** | 257.2 μs | 3 | 7.64 MiB |
| **`D₋ᵧ`** | 161.4 μs | 3 | 7.64 MiB |
| **`D₋ₓ`** | 203.7 μs | 3 | 7.64 MiB |
| **`M₋ₓ`** | 171.4 μs | 3 | 7.64 MiB |

```@raw html
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<div style="width: 100%; max-width: 820px; margin: 1.5em auto; background: var(--documenter-bg, #fff); padding: 1.2em; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
  <canvas id="benchmark_chart_1"></canvas>
</div>
<script>
document.addEventListener("DOMContentLoaded", function() {
    var ctx = document.getElementById('benchmark_chart_1').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ["Dcₓ", "D₋ᵧ", "D₋ₓ", "M₋ₓ"],
            datasets: [{
    label: '0b9a62b (Julia 1.12.7)',
    data: [257.21, 161.42, 203.71, 171.42],
    backgroundColor: 'rgba(54, 162, 235, 0.85)',
    borderColor: 'rgb(54, 162, 235)',
    borderWidth: 1
}
]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Operators 2D - Median Execution Time (μs)'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.parsed.y === null) return context.dataset.label + ': (not measured)';
                            return context.dataset.label + ': ' + context.parsed.y + ' μs';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (μs)'
                    }
                }
            }
        }
    });
});
</script>

```

### Operators 3D

| Benchmark | `0b9a62b` Time | Allocs | Memory |
|---|:---:|:---:|:---:|
| **`D₋₂`** | 200.9 μs | 3 | 7.64 MiB |
| **`innerₕ`** | 240.2 μs | 0 | 0 B |
| **`∇₋ₕ`** | 694.1 μs | 15 | 22.92 MiB |

```@raw html
<div style="width: 100%; max-width: 820px; margin: 1.5em auto; background: var(--documenter-bg, #fff); padding: 1.2em; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
  <canvas id="benchmark_chart_2"></canvas>
</div>
<script>
document.addEventListener("DOMContentLoaded", function() {
    var ctx = document.getElementById('benchmark_chart_2').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ["D₋₂", "innerₕ", "∇₋ₕ"],
            datasets: [{
    label: '0b9a62b (Julia 1.12.7)',
    data: [200.92, 240.25, 694.08],
    backgroundColor: 'rgba(54, 162, 235, 0.85)',
    borderColor: 'rgb(54, 162, 235)',
    borderWidth: 1
}
]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Operators 3D - Median Execution Time (μs)'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.parsed.y === null) return context.dataset.label + ': (not measured)';
                            return context.dataset.label + ': ' + context.parsed.y + ' μs';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (μs)'
                    }
                }
            }
        }
    });
});
</script>

```

### Inner Products 2D

| Benchmark | `0b9a62b` Time | Allocs | Memory |
|---|:---:|:---:|:---:|
| **`innerₕ`** | 242.0 μs | 0 | 0 B |
| **`norm₁ₕ`** | 790.2 μs | 0 | 0 B |
| **`normₕ`** | 190.0 μs | 0 | 0 B |
| **`snorm₁ₕ`** | 578.1 μs | 0 | 0 B |

```@raw html
<div style="width: 100%; max-width: 820px; margin: 1.5em auto; background: var(--documenter-bg, #fff); padding: 1.2em; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
  <canvas id="benchmark_chart_3"></canvas>
</div>
<script>
document.addEventListener("DOMContentLoaded", function() {
    var ctx = document.getElementById('benchmark_chart_3').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ["innerₕ", "norm₁ₕ", "normₕ", "snorm₁ₕ"],
            datasets: [{
    label: '0b9a62b (Julia 1.12.7)',
    data: [242.04, 790.25, 190.0, 578.08],
    backgroundColor: 'rgba(54, 162, 235, 0.85)',
    borderColor: 'rgb(54, 162, 235)',
    borderWidth: 1
}
]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Inner Products 2D - Median Execution Time (μs)'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.parsed.y === null) return context.dataset.label + ': (not measured)';
                            return context.dataset.label + ': ' + context.parsed.y + ' μs';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (μs)'
                    }
                }
            }
        }
    });
});
</script>

```

### Restriction

| Benchmark | `0b9a62b` Time | Allocs | Memory |
|---|:---:|:---:|:---:|
| **`Rₕ 1D (allocates its output)`** | 2.87 ms | 6 | 7.64 MiB |
| **`Rₕ! 1D`** | 2.87 ms | 3 | 64 B |
| **`avgₕ! 1D`** | 16.27 ms | 2 | 128 B |

```@raw html
<div style="width: 100%; max-width: 820px; margin: 1.5em auto; background: var(--documenter-bg, #fff); padding: 1.2em; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
  <canvas id="benchmark_chart_4"></canvas>
</div>
<script>
document.addEventListener("DOMContentLoaded", function() {
    var ctx = document.getElementById('benchmark_chart_4').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ["Rₕ 1D (allocates its output)", "Rₕ! 1D", "avgₕ! 1D"],
            datasets: [{
    label: '0b9a62b (Julia 1.12.7)',
    data: [2.87, 2.87, 16.27],
    backgroundColor: 'rgba(54, 162, 235, 0.85)',
    borderColor: 'rgb(54, 162, 235)',
    borderWidth: 1
}
]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Restriction - Median Execution Time (ms)'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.parsed.y === null) return context.dataset.label + ': (not measured)';
                            return context.dataset.label + ': ' + context.parsed.y + ' ms';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (ms)'
                    }
                }
            }
        }
    });
});
</script>

```

### Composite

| Benchmark | `0b9a62b` Time | Allocs | Memory |
|---|:---:|:---:|:---:|
| **`D₋ₓ (3 components)`** | 711.0 μs | 3 | 22.89 MiB |
| **`∇₋ₕ (3 components)`** | 1.43 ms | 10 | 45.78 MiB |

```@raw html
<div style="width: 100%; max-width: 820px; margin: 1.5em auto; background: var(--documenter-bg, #fff); padding: 1.2em; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
  <canvas id="benchmark_chart_5"></canvas>
</div>
<script>
document.addEventListener("DOMContentLoaded", function() {
    var ctx = document.getElementById('benchmark_chart_5').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ["D₋ₓ (3 components)", "∇₋ₕ (3 components)"],
            datasets: [{
    label: '0b9a62b (Julia 1.12.7)',
    data: [0.71, 1.43],
    backgroundColor: 'rgba(54, 162, 235, 0.85)',
    borderColor: 'rgb(54, 162, 235)',
    borderWidth: 1
}
]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Composite - Median Execution Time (ms)'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.parsed.y === null) return context.dataset.label + ': (not measured)';
                            return context.dataset.label + ': ' + context.parsed.y + ' ms';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (ms)'
                    }
                }
            }
        }
    });
});
</script>

```

### Construction

| Benchmark | `0b9a62b` Time | Allocs | Memory |
|---|:---:|:---:|:---:|
| **`gridspace 2D`** | 368.8 μs | 38 | 30.59 MiB |
| **`gridspace 3D`** | 1.63 ms | 44 | 38.21 MiB |
| **`hₘₐₓ 3D`** | 153.0 ns | 0 | 0 B |

```@raw html
<div style="width: 100%; max-width: 820px; margin: 1.5em auto; background: var(--documenter-bg, #fff); padding: 1.2em; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
  <canvas id="benchmark_chart_6"></canvas>
</div>
<script>
document.addEventListener("DOMContentLoaded", function() {
    var ctx = document.getElementById('benchmark_chart_6').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ["gridspace 2D", "gridspace 3D", "hₘₐₓ 3D"],
            datasets: [{
    label: '0b9a62b (Julia 1.12.7)',
    data: [0.37, 1.63, 0.0],
    backgroundColor: 'rgba(54, 162, 235, 0.85)',
    borderColor: 'rgb(54, 162, 235)',
    borderWidth: 1
}
]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Construction - Median Execution Time (ms)'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.parsed.y === null) return context.dataset.label + ': (not measured)';
                            return context.dataset.label + ': ' + context.parsed.y + ' ms';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (ms)'
                    }
                }
            }
        }
    });
});
</script>

```

## How to Add New Benchmark Runs

To record performance on a new commit or after an optimization pass, run:

```bash
julia --project=benchmark benchmark/benchmarks.jl --save benchmark/baselines/baseline_$(git rev-parse --short HEAD).json
```

Rebuilding the documentation (`julia -e 'using Pkg; Pkg.activate("docs"); include("docs/make.jl")'`) will automatically discover all `baseline_*.json` files and append new comparison columns, delta calculations, and chart series.
