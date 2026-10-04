# Measurements moved out of `docs/src/internals/space.md`

## Linear-index weight read

once per point; the refill timings below were measured with that cost in. Today

## The one-dimensional case: measurements

Measured directly (a `10^6`-point
1D space, summing every entry by linear index, minimum of 15 back-to-back calls): 0.109
ms for the `SeparableWeights{1}` against 0.109 ms for the plain `Vector` underneath it, a
ratio of 0.997 -- no detectable difference, confirming the comment.

That claim is about single-element access, not about the whole-vector reduction
`_dot`/`_dot_masked` runs. Measured the same way on the same 1D space, `_dot` against the
`SeparableWeights{1}` costs 0.857 ms to the 0.230 ms dense reduction it replaces, a ratio
of 3.73 -- essentially the same penalty measured on the `100³` space below, not a smaller
one for having only one axis. Dimension is not what the `_dot` specialization's cost
tracks: its loop walks `CartesianIndices(w.dims)` with no `@simd` annotation, where the
dense reduction and the generic `_dot` both carry one, and that gap is present whether
`w.dims` has one entry or three. The fast path a single factor buys is real, but it
belongs to `getindex`, not to reduction: uniformity buys nothing extra for `_dot` at
`D = 1`, and costs nothing extra either -- the same fixed penalty this section measures
at `D = 3` below.

## Measured (weight storage)

### Measured

Machine state at measurement time was mixed, not clean: the session started on battery
power (97%, discharging, load average 2.29 against 8 cores) and had moved to AC power
partway through (charging, load average 4.6-6.1), because a `julia` language-server
process for this editor was already running, not because of a competing test or
benchmark run. Per `bramble-verification` §2/§9, absolute milliseconds under those
conditions are not trustworthy alone; every figure below is a same-process,
back-to-back, minimum-of-`N` measurement (`N ≥ 5`, warmed up first), and the ratios
between paired measurements are the load-bearing numbers, not the raw times.

`Base.summarysize(weights(Wₕ))` on the `100³` mesh, whose weights were to fit in under 1 MB:

| stage | summarysize |
|---|---|
| before #115 (4 dense vectors) | 32,000,000 B |
| per-axis factors added, 4 families still dense | 32,005,288 B |
| no family left dense | 5,288 B |

Measuring the last row directly prints `summarysize(weights) on 100^3: 5288 B`. That is a
~6,053x reduction from the per-axis-factor figure, and a ~6,053x reduction from
the original dense one as well, since the twelve small per-axis vectors cost
the same fixed handful of kilobytes regardless.

`innerₕ(u, u)` and `inner₊ₓ(u, u)` on the same `100³` space, against a dense-vector
reduction over the identical values (the same `muladd`/`@simd` shape `_dot` uses for a
plain `AbstractVector`), minimum of 7 back-to-back calls each, warmed up first, before the
2026-09-25 line-walk rewrite below:

| call | time | dense equivalent | ratio |
|---|---|---|---|
| `innerₕ(u, u)` | 0.868 ms | 0.231-0.240 ms | 3.6-3.8x |
| `inner₊ₓ(u, u)` | 0.868 ms | 0.227-0.234 ms | 3.7-3.8x |

Both agreed with the dense reduction to `rtol = 1e-12`.

Calling `_dot` directly rather than through `innerₕ`, the minimum of 15 back-to-back
calls was 0.868-0.892 ms for `SeparableWeights` against 0.180-0.182 ms for the dense
vector. The ratio was 4.81-4.90. `@which` confirmed dispatch reached the `CartesianIndex`-walking
specialization (`src/space/inner_product.jl:333`), not the generic `AbstractVector`
method (`src/utils/linear_algebra.jl:408`) -- so this was the intended path, not the
linear-`getindex` fallback. The ratio measured here sat near that fallback's own ≈4.9x
figure rather than nearer the ≈2.475x recorded when this specialization was added; the
likely reason, from reading the loop as it stood then: it carried no `@simd`
annotation. The load and power state moved during this session (above), so a second
contributor could not be ruled out; both figures were reported rather than one silently
preferred, per `bramble-verification`.

**Reconciled for gpena/Bramble.jl#273** (2026-09-22, Julia 1.13.0, this machine,
`--threads=4`, battery power at 76-77%, load average ~2.2 on 8 cores -- not the quiet
machine `bramble-verification` asks for, but the same caveat the paragraph above already
carries). Re-running the exact 100³/`weights(Wₕ, Val((1,2)))` case measured
above, `_dot` against that `SeparableWeights` versus the same weights `collect`ed
to a dense vector, `@belapsed`, four independent process runs gave 2.43-2.49x -- not the
4.81-4.90x above, but squarely the ≈2.4x the source comment then carried and the ≈2.475x
recorded when this specialization was added. At the time the loop was unchanged (`@simd`
had been tried and reverted: on a deterministic-seed mesh it changed the reduction at the
bit level, not only its speed, so it failed the correctness bar this figure was measured
under). Nothing else had moved either -- same specialization, same `@which` dispatch,
same missing `@simd`. The likeliest explanation was the one this section already named
for the earlier run: its own mixed battery/AC/competing-language-server state, not a
property of the code.

**Superseded by the 2026-09-25 line-walk rewrite.** The loop no longer walks
`CartesianIndices(w.dims)` point by point with a serial `muladd` chain and no `@simd`; it
walks axis-1 lines with the product of the other axes' factors hoisted out once per line,
and reduces each line with `@inbounds @simd` (`src/space/inner_product.jl`, comment above
`_dot`). Measured the same way, on non-uniform 1000² and 100³ grids, minimum of 15 runs:
`innerₕ`/`inner₊ₓ` now cost **0.74-0.78x** the dense reduction over the collected weights
-- faster than dense, because there is one fewer full-length vector to read, not slower.
The 3.6-3.8x, 4.81-4.90x and 2.4-2.5x figures above are the pre-rewrite history; none of
them is the current cost.

One `assemble!` refill of `innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))` on the uniform `60³` mesh
the per-axis-factor change was measured on (`ndofs = 216,000`, `nnz(A) = 1,490,400`),
warmed up, minimum of 5, at 4 threads (this repository's standard, `bramble-benchmarks` §1):

| stage | min time | range | allocation |
|---|---|---|---|
| 4 dense families | 4.04 ms | 4.04-4.97 ms | 0 B, all 5 runs |
| no dense families | 3.16-3.19 ms | 3.16-4.58 ms | 0 B, all 5 runs |

No regression: the refill sits at or below the dense-family range, and allocation is still
zero on every run, matching the documented zero-allocation `assemble!` contract. The
division-per-axis `InnerH`/`InnerPlus{Dim}`'s `compute_weight` paid at the time (previous
section, since removed) did not show up here -- it was a small fraction of everything
else one assembled point costs.

`gridspace` construction on the same `100³` mesh, warmed up, minimum of 7: 0.00008-0.00013
ms (83-125 ns), allocating 2,784 B. That allocation is the `D = 3` per-axis `aligned`
vectors plus the `SpaceWeights`/`ScalarGridSpace` structs themselves; `cellfactor` is a
zero-copy reference to the mesh's own cached half-spacings and adds nothing. Construction
was not timed before this change and cannot safely be now, since reproducing the pre-#115
code would mean checking out a different tree; what is measured is that today's
construction fills no `O(n^D)` vector, allocates in the hundreds of bytes rather than
tens of megabytes, and completes in well under a microsecond on a `10^6`-point mesh.

## Why the trade is worth taking (as it stood)

### Why the trade is worth taking

Eliminating the last `O(n^D)` storage now costs nothing extra: after the 2026-09-25
line-walk rewrite, `innerₕ`/`inner₊` cost **0.74-0.78x** a dense whole-vector reduction --
faster than dense, not slower -- against weights that no longer scale with the grid at
all: 5,288 B instead of 32,005,288 B on `100³`, a figure that would only have grown had
the `2^D` staggered family stayed dense alongside it. Before that
rewrite the same reduction cost roughly 3.6-3.8x dense (measured above); that history no
longer applies. The one path that matters for a typical solve, `assemble!`, is unaffected
-- it does not move outside the range already on record.

The case that would make this the wrong trade has mostly closed with the rewrite: there
is no whole-vector-reduction penalty left to pay for calling `innerₕ`/`inner₊` (or
`normₕ`/`norm₊`, built on them) in a per-iteration hot loop. The one place a penalty
remains is a grid whose axis 1 has only 1-4 points, where the per-line overhead leaves
the reduction at 1.4-3.3x dense -- still never slower than the pre-rewrite loop, but not
free either.

## Measured (operator matrices)

### Measured

`benchmark/operator_matrices.jl`, run directly (`JULIA_DEPOT_PATH="$TMPDIR/depot-s103:
$HOME/.julia" julia --startup-file=no --threads=4 --project=benchmark
benchmark/operator_matrices.jl`) on 2026-09-19, `kronecker_operator_matrix` ("old")
against the routed public alias ("new"):

| operator | mesh | old time | new time | speedup | old allocs | new allocs | old bytes | new bytes |
|---|---|---|---|---|---|---|---|---|
| `D₋ₓ` | 1000 (1D) | 17.96 μs | 3.62 μs | 5.0x | 54 | 9 | 161.1 KiB | 40.2 KiB |
| `Dcₓ` | 1000 (1D) | 24.96 μs | 4.30 μs | 5.8x | 78 | 9 | 225.6 KiB | 40.2 KiB |
| `Mₓ` | 1000 (1D) | 19.92 μs | 3.39 μs | 5.9x | 63 | 9 | 201.3 KiB | 40.2 KiB |
| `D₋ₓ` | 300² (2D) | 1.32 ms | 317.8 μs | 4.2x | 72 | 9 | 8.29 MiB | 3.44 MiB |
| `Dcₓ` | 300² (2D) | 1.71 ms | 351.8 μs | 4.9x | 114 | 9 | 8.32 MiB | 3.44 MiB |
| `Mₓ` | 300² (2D) | 1.63 ms | 278.2 μs | 5.9x | 81 | 9 | 11.72 MiB | 3.44 MiB |
| `D₋ₓ` | 60³ (3D) | 3.29 ms | 840.9 μs | 3.9x | 73 | 9 | 19.85 MiB | 8.16 MiB |
| `Dcₓ` | 60³ (3D) | 4.36 ms | 844.3 μs | 5.2x | 116 | 9 | 19.82 MiB | 8.03 MiB |
| `Mₓ` | 60³ (3D) | 4.16 ms | 716.7 μs | 5.8x | 82 | 9 | 28.06 MiB | 8.16 MiB |

Allocation count is flat at 9 regardless of family, axis or mesh size, while the old
construction's count grows with how many Kronecker shifts and subtractions the family
composes (more for `Dc`'s two-sided stencil than for `D₋`'s one-sided one). Time improves
3.9-5.9x and memory drops to roughly a fifth to a half across every case; neither
reduction is uniform across mesh size or family, and no case regresses.

Machine state, per `bramble-verification`: on battery (72%, discharging), load average
2.56/4.92/10.02 at the time of the run, with at least one other Julia process from a
concurrent session in this same worktree active during the measurement (an unrelated
`inner₊` check). The absolute times above should be read with that in mind; the ratios
are what this record relies on, and they land in the same range (4-6x, 9 flat
allocations against 54-116) that an earlier, separate run already reported.

