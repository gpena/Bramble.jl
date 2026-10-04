# Measurements moved out of `docs/src/internals/gpu.md`

## Intro disclaimer

**No general performance claim is made on this page.** Every figure below is a specific
measurement, cited beside the script, sizes and machine state it came from, on one Apple M2
host; none should be read as a speedup beyond those conditions.

## Form assembly on a device: measured cost

Measured on this host (`innerₕ(D₋ₓ(U), D₋ₓ(V))`, `BenchmarkTools`, one warmed
process), device `assemble`/`assemble!` are 2-8x slower than the host path, in 1D, 2D and
3D, both forms, with no offsetting benefit anywhere in the number, because the device
never does anything the host wasn't already going to do. Full detail and the table are in

## Device assembly: per-call allocation

none of which the host path pays. A 2D `n = 17` linear form measured roughly 17 KB per
call; a 1D Metal `assemble!` at `n = 100,001` measured roughly 2.47 MB per call, almost all
of it rebuilding `host_weights(W)`, plus roughly 0.4 MB per grid-function coefficient in the
form.

## The evidence: memory, throughput, assembly frequency

**Memory and throughput: assembled CSR against Kronecker on Metal.**
`benchmark/assembled_vs_matrixfree.jl --full` (v3.14.0, commit `9051ac58`)
compared, for the same separable operator `innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))`, a host
`assemble` uploaded once with `metal_sparse_csr` against a Metal-backed
`kronecker_operator`, both `Float32`, after checking host-against-device agreement for both
at every size. Apple M2, recommended working set 12.71 GB, AC power, load 2.42-3.28 against a
threshold of 4.0, `--threads=4`:

| Case | N (dofs) | CSR bytes | Kronecker bytes | CSR/Kron | CSR share of working set | SpMV (ms) | Kron apply (ms) | SpMV/Kron |
|---|---|---|---|---|---|---|---|---|
| 2D 500² | 250,000 | 16,976,008 | 35,976 | 471.87 | 0.13% | 0.6747 | 0.6507 | 1.037 |
| 2D 1500² | 2,250,000 | 152,928,008 | 107,976 | 1416.31 | 1.20% | 2.0782 | 1.3717 | 1.515 |
| 2D 3000² | 9,000,000 | 611,856,008 | 215,976 | 2832.98 | 4.81% | 7.645 | 4.7157 | 1.621 |
| 3D 60³ | 216,000 | 19,612,808 | 7,164 | 2737.69 | 0.15% | 0.9352 | 0.6802 | 1.375 |
| 3D 120³ | 1,728,000 | 157,939,208 | 14,364 | 10995.5 | 1.24% | 2.1991 | 1.8521 | 1.187 |
| 3D 200³ | 8,000,000 | 733,120,008 | 23,964 | 30592.6 | 5.77% | 9.5458 | 7.7066 | 1.239 |

- **There is no memory crossover in the measured range.** The assembled matrix never
  exceeds 5.77% of the device's recommended working set, at the largest size measured (3D
  200³, 8,000,000 dofs). The device CSR's own bytes are what the table counts; the host
  mirror that the fill path accumulates into holds roughly a second copy in the same unified
  DRAM (#316's own description), which at most doubles the 5.77% figure -- still a small
  fraction of the working set. Kronecker factors are 472x to 30,593x smaller than the matrix.
- **Matrix-free costs no throughput.** The second amendment kept the throughput
  measurement only as the check that matrix-free does not cost throughput at low order. The
  Kronecker apply was faster than the assembled SpMV at every size: 1.04-1.62x in 2D, growing
  with size, and 1.19-1.38x in 3D, where the margin does not grow monotonically.

**Assembly frequency: assemble once against re-assemble every step.** The same run timed
both regimes over 100 steps, computed from one measured build and one measured apply per
arm (`build + 100 * apply`, and `100 * (build + apply)`):

| Case | CSR build (ms) | Kron build (ms) | Once + 100 applies, CSR/Kron | Re-assemble every step, CSR/Kron |
|---|---|---|---|---|
| 2D 500² | 30.85 | 3.05 | 1.443 | 8.528 |
| 2D 1500² | 290.22 | 3.12 | 3.550 | 65.098 |
| 2D 3000² | 1735.69 | 3.60 | 5.262 | 209.681 |
| 3D 60³ | 40.25 | 4.16 | 1.853 | 8.513 |
| 3D 120³ | 362.77 | 4.62 | 3.069 | 56.426 |
| 3D 200³ | 3035.24 | 4.19 | 5.149 | 255.844 |

The CSR build grows with the problem (31 ms to 1736 ms in 2D, 40 ms to 3035 ms in 3D) while
the Kronecker build stays at 3-5 ms, so re-assembling every step is where an assembled
operator hurts most, and for a separable operator the Kronecker path removes that cost
outright. The CSR "build" here is a full `assemble` plus upload, not a refill: an
`assemble!` into an existing matrix replays recorded positions (below) and costs a small
fraction of a first assembly, so the re-assemble column is an upper bound on the CSR side.

## Scatter-position table: measurements

`benchmark/scatter_table.jl` (commit `fb839b94`; AC power, load 3.01,
`--threads=4`, every row agreeing with a fresh assemble) measured the refill at the minimum of
40 `assemble!` calls; at 1D `n = 2049` CpuSerial took 0.00621 ms, CpuThreaded 0.05117 ms,
CpuPolyester 0.01196 ms and Metal 1.29733 ms, and on a non-uniform 2D `129 × 97` grid
0.07325, 0.14017, 0.06304 and 2.53271 ms respectively, against first assemblies of 0.0859 ms
(1D, CpuSerial) and 6.298 ms (2D, CpuSerial). The recorded positions cost 16,984 bytes
against a 114,880-byte matrix at 1D `n = 2049` (ratio 0.148 for CpuSerial; 0.086 for the
threaded policies, whose matrix measures 196,792 bytes) and 996,280 bytes against 1,094,080 on the 2D grid
(ratio 0.91 for CpuSerial, 0.42 threaded). The Metal rows in that run predate commit
`208b23de` and searched on every refill; their 64-byte cache figure is the absence of a
table, not its cost. Commit `208b23de`'s own review measured the Metal replay on a
four-term 2D `257²` form at 2.94-3.0 ms against 5.0-6.7 ms for the search it replaced.

## Shared Metal storage: upload against gap

The step it removes, the one bulk
upload at the end of a fill, is about 33 µs against a 2,040 µs gap between Metal and serial
`assemble!` at 1D `N = 200,001` -- roughly 1.6% of the measured gap, because every
scatter-add already runs on the CPU whatever the storage mode.

## Matrix-free Kronecker on a device: measured throughput

**Measured throughput.** `benchmark/kronecker_device.jl` timed the host and Metal backends
back to back on uniform meshes, `Float32`, 4 threads, AC power, load 2.4:

| Case | Operation | Host | Metal | Ratio (host/Metal) |
|---|---|---|---|---|
| 2D 3000x3000 | `mul!` | 5.97 ms | 4.28 ms | 1.39 |
| 2D 3000x3000 | 30 CG-shaped iterations | 523.9 ms | 304.5 ms | 1.72 |
| 2D 3000x3000 | `fdm_solve` | 5382.2 ms | 2088.6 ms | 2.58 |
| 3D 200x200x200 | `mul!` | 7.86 ms | 7.08 ms | 1.11 |
| 3D 200x200x200 | 30 CG-shaped iterations | 577.2 ms | 369.0 ms | 1.56 |
| 3D 200x200x200 | `fdm_solve` | 486.1 ms | 122.2 ms | 3.98 |

The `fdm_solve` host eigendecomposition alone (the LAPACK call above, unaffected by
backend) took 1821.3 ms in the 2D case and 4.99 ms in the 3D case, out of the totals above.
Before the fused pass replaced the per-axis sweep, the same benchmark measured `mul!` at
41.49 ms host / 41.12 ms Metal in 2D and 43.43 ms host / 77.39 ms Metal in 3D, and the
30-iteration CG loop at 1686.1 ms host / 1416.5 ms Metal in 2D and 1755.7 ms host / 2484.1
ms Metal in 3D. For comparison, #323's hand-rolled matrix-free CG loop -- calling `Δₕ!`
directly rather than going through `mul!(y, K, x, ...)` -- measured 1.55x (2D) and 1.16x
(3D) Serial/Metal, at an absolute Metal time of 867.1 ms (2D) and 1144.0 ms (3D). The CG
loop through `K` now beats that hand-rolled loop on both counts: a higher Serial/Metal
ratio (1.72x and 1.56x) and a lower absolute Metal time (304.5 ms and 369.0 ms).

**The plain conclusion.** The fused pass turns the device `mul!` from roughly break-even
(2D) or slower-on-device (3D) into a device win, and that carries through to an iterative
solve: 30 CG-shaped iterations through `K` now gain 1.72x in 2D and 1.56x in 3D, and
`fdm_solve` gains 2.6-4.0x. No cause for the `fdm_solve` gain is established beyond what
the eigendecomposition split above already shows. For `mul!` itself, the one cause
measured is the switch to `Int32` index arithmetic in the device kernel, which took its
time from 10.8/18.4 ms (2D/3D) to 4.3/7.1 ms; no other cause was measured, so none is
claimed.

## Per-call device routing: measured

**Measured, honestly: three wins and one loss, not a universal speedup.**
`benchmark/gpu_offload.jl` (commit `2c7217e6`) re-measured the issue's own exploratory table
back to back, `CpuThreaded()` against `GpuOffload(metal_backend(), CpuThreaded())`, on the
same four rows, correctness gated before timing (`rtol = atol = 1f-5`, `Float64` host vs
`Float32` device). Apple M2, AC power, load 3.09/4.0, `--threads=4`:

| Case | Host (`CpuThreaded`) | `GpuOffload` | Ratio |
|---|---|---|---|
| `Rₕ!` 1D, n=10,000,000 | faster | slower | **0.86x -- offload loses** |
| `Rₕ!` 2D, 3000x3000 | -- | -- | 1.79x |
| `avgₕ!` 1D, n=10,000,000 | -- | -- | 3.27x |
| `avgₕ!` 2D, 3000x3000 | -- | -- | 10.86x |

Three of the four rows win clearly, close to the issue's own exploratory figures
(1.98x/9.87x/20.52x for the same three rows). `Rₕ!` 1D at ten million points *loses*
at 0.86x, so the host wins. The reason is the no-hidden-cache design named above. `Rₕ!` is
one cheap function evaluation per point and already fast on four CPU threads, so it does not
amortise the per-call mesh-axis upload plus full result copy-back that `GpuOffload` pays on
every call with nothing cached. `avgₕ!`'s per-cell quadrature is expensive enough on the
host that the same round trip is dwarfed. That is why its two rows show the
largest wins in the table. Read this policy as a win for expensive per-point/per-cell work
on a large grid, not as something to reach for by default -- measure the specific call
before choosing it over the wrapped inner policy alone.

