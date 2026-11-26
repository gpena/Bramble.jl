# Warm timings before and after part 1 of v4.4.0 (S6.2)

Part 1 moved its hot items to part 3 but changed four warm paths: the solve entry (#948),
`_fdm_ldiv!` (#1068), the inlined `_replay_pair_segment!` (#1076) and the shared
`_scale_only!` (#1074, #1076). This note times each before and after the part, on
non-uniform meshes, as the milestone requires. It changes no code.

## Conditions

| | |
|---|---|
| Date | 2026-11-26 |
| Machine | macbook, Apple M2 (4 performance and 4 efficiency cores), 16 GB, on AC power |
| Julia | 1.13.4, default `-O`, `--startup-file=no`, 4 threads |
| Base | `5ee2b6bc`, a detached worktree of the plan's base commit |
| Tip | `54788af2`, the tip of `v4-4-0-part1` after part 1 |
| Load before the first run | 1.89 (`check_power_load.sh`, threshold 4.0) |
| Other work | no other Julia process, OneDrive and other sessions held; macOS `XProtect` was scanning until a few minutes before the first run and was gone by then |
| Packages | one manifest for both trees (the tip's), SuiteSparse, Sparspak, MUMPS, AppleAccelerate, Kronecker and Polyester all loaded, nothing skipped |

Meshes. Every mesh is built as the test suite's `_graded_mesh`: the unit square with `n`
points per axis, then the points of axis `d` replaced by `range(0, 1; length = n)^(1 + d/4)`
through `Bramble.change_points!`, so no two axes share their spacing and no cell is uniform.
`n` is 8 or 32 for the solves and 64 for everything else. The domain carries a `:boundary`
label, and the Dirichlet rows are `:boundary => (x -> 0.0)`.

Method. One script, run once per launch in each tree. Per case it warms the call three
times, picks a batch of calls that lasts about 2 ms, takes 40 to 300 batches (about 1 s) and
keeps the median of the per-call times of that launch. Three launches alternate base, tip,
base, tip, base, tip, in the foreground, nothing else running. A median in a table is the
median of the three launch medians. The spread is `(max - min) / median` over the same
three. A case is "inside" when the three launch medians of the two sides overlap in range
and "outside" when the tip's lowest is above the base's highest (or the reverse). The last
column repeats the ratio over six launches (three more, tip first, same conditions), because
two rows below move by launch.

`pde_solve` takes no form: `pde_solve(A, F; solver)` is the entry, with `A, F = assemble(a, l;
dirichlet, symmetrize = true)` built once outside the loop. The rows named `*_solve(a, l)` call
each backend's form-level entry, which assembles on every call and so adds the assembly to
the solve. Each operator runs under its default execution policy.

## Solve entry (S2.5, S2.6)

| Case | base median (µs) | tip median (µs) | ratio | base spread | tip spread | 3 runs | ratio over 6 runs |
|---|---:|---:|---:|---:|---:|---|---|
| pde_solve(A,F) default 8x8 | 35.7 | 36.7 | 1.029 | 17 % | 8 % | inside | 1.028 (inside) |
| pde_solve(A,F) suitesparse 8x8 | 19.2 | 19.6 | 1.018 | 4 % | 1 % | inside | 1.013 (inside) |
| pde_solve(A,F) spqr 8x8 | 37.9 | 37.2 | 0.983 | 2 % | 5 % | inside | 0.993 (inside) |
| pde_solve(A,F) accelerate 8x8 | 33.9 | 35.2 | 1.038 | 4 % | 9 % | inside | 0.999 (inside) |
| pde_solve(A,F) mumps 8x8 | 12 145 | 12 134 | 0.999 | 43 % | 35 % | inside | 0.998 (inside) |
| pde_solve(A,F) sparspak 8x8 | 38.8 | 39.0 | 1.005 | 1 % | 3 % | inside | 1.000 (inside) |
| suitesparse_solve(a,l) 8x8 | 21.3 | 20.9 | 0.981 | 3 % | 3 % | inside | 1.010 (inside) |
| accelerate_solve(a,l) 8x8 | 31.8 | 35.8 | 1.127 | 21 % | 22 % | inside | 1.073 (inside) |
| mumps_solve(a,l) 8x8 | 12 146 | 12 123 | 0.998 | 27 % | 24 % | inside | 0.998 (inside) |
| sparspak_solve(a,l) 8x8 | 39.6 | 41.2 | 1.041 | 4 % | 5 % | inside | 1.004 (inside) |
| pde_solve(A,F) default 32x32 | 559.6 | 565.1 | 1.010 | 6 % | 3 % | inside | 1.012 (inside) |
| pde_solve(A,F) suitesparse 32x32 | 472.2 | 466.4 | 0.988 | 3 % | 1 % | inside | 0.997 (inside) |
| pde_solve(A,F) spqr 32x32 | 1 644 | 1 639 | 0.997 | 1 % | 1 % | inside | 0.993 (inside) |
| pde_solve(A,F) accelerate 32x32 | 545.1 | 539.2 | 0.989 | 4 % | 3 % | inside | 1.004 (inside) |
| pde_solve(A,F) mumps 32x32 | 12 969 | 13 021 | 1.004 | 27 % | 28 % | inside | 1.004 (inside) |
| pde_solve(A,F) sparspak 32x32 | 860.5 | 864.2 | 1.004 | 2 % | 2 % | inside | 1.006 (inside) |
| suitesparse_solve(a,l) 32x32 | 475.2 | 474.5 | 0.999 | 3 % | 1 % | inside | 0.998 (inside) |
| accelerate_solve(a,l) 32x32 | 543.4 | 546.2 | 1.005 | 1 % | 8 % | inside | 0.999 (inside) |
| mumps_solve(a,l) 32x32 | 12 966 | 13 462 | 1.038 | 24 % | 19 % | inside | 1.016 (inside) |
| sparspak_solve(a,l) 32x32 | 868.8 | 878.6 | 1.011 | 2 % | 2 % | inside | 1.012 (inside) |

Every ratio is inside the spread. The largest is `accelerate_solve(a, l)` at 8 x 8 (1.127
over three runs, 1.073 over six, with 21 % and 22 % spread: the row is 30 to 38 µs and moves
that much between launches in both trees). Next come 1 % to 4 % on `sparspak_solve(a, l)`,
`pde_solve` with `:default` and `mumps_solve(a, l)`, whose six-run ratios sit between 1.004
and 1.028. MUMPS is bimodal by launch in both trees: the first launch of each runs 7 to 9 ms
and the other five 12.1 ms at 8 x 8 and 13.0 ms at 32 x 32. That gives its 19 % to 43 %
spread, and inside either mode the two trees agree to under 1 %. No solve-entry row shows a
slowdown to blame on S2.5 or S2.6.

## `ldiv!` of the fast-diagonalisation factorizations (S2.8)

| Case | base median (µs) | tip median (µs) | ratio | base spread | tip spread | 3 runs | ratio over 6 runs |
|---|---:|---:|---:|---:|---:|---|---|
| fdm symmetric 64x64 | 20.8 | 20.9 | 1.005 | 1 % | 2 % | inside | 1.006 (inside) |
| fdm Schur 64x64 | 664.0 | 656.4 | 0.988 | 7 % | 6 % | inside | 0.991 (inside) |

The symmetric row is `_FDMFactorization` and the Schur row `_SchurFactorization`, built by
`fdm_factorize(a; dirichlet = :boundary)` from `innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))` and from
that plus `1.5 innerₕ(D₋ₓ(u), v) + 2.5 innerₕ(D₋ᵧ(u), v)`. Both ratios are inside the spread, and the 20.8 µs symmetric row
agrees to 0.5 %. The single `_fdm_ldiv!` of S2.8 costs nothing measurable on either route.

## Warm `assemble!` of a two-component form (S3.6, the pair replay)

| Case | base median (µs) | tip median (µs) | ratio | base spread | tip spread | 3 runs | ratio over 6 runs |
|---|---:|---:|---:|---:|---:|---|---|
| elasticity 64x64 | 933.8 | 937.5 | 1.004 | 2 % | 3 % | inside | 1.000 (inside) |
| vector_laplacian 64x64 | 88.7 | 87.9 | 0.991 | 1 % | 1 % | inside | 0.986 (inside) |

`elasticity` is `2.0 inner₊(εₕ(u), εₕ(v)) + 2.0 inner₊(divₘₕ(u), divₘₕ(v))` on
`gridspace(Ω)^Val(2)`; three of its ten recorded segments carry a pair half, so
`_replay_pair_segment!` runs. `vector_laplacian` has none and is the control. Each call fills the matrix the form
recorded, after two warm fills. Both ratios are inside the spread, the pair-replay row at
1.004 over three launches and 1.000 over six.

## `mul!` of the Kronecker block and matrix-free operators (S3.6, `_scale_only!`)

| Case | base median (µs) | tip median (µs) | ratio | base spread | tip spread | 3 runs | ratio over 6 runs |
|---|---:|---:|---:|---:|---:|---|---|
| kron block (3 blocks) β=1.0 64x64 | 11.5 | 11.4 | 0.992 | 2 % | 2 % | inside | 0.992 (inside) |
| kron block (3 blocks) β=0.5 64x64 | 11.8 | 12.0 | 1.021 | 1 % | 8 % | outside | 1.009 (inside) |
| kron block (idle component) β=1.0 64x64 | 11.4 | 11.8 | 1.036 | 2 % | 4 % | inside | 1.010 (inside) |
| kron block (idle component) β=0.5 64x64 | 12.8 | 12.8 | 1.000 | 2 % | 2 % | inside | 0.999 (inside) |
| matrix-free β=1.0 64x64 | 5.7 | 7.1 | 1.251 | 28 % | 23 % | inside | 1.260 (inside) |
| matrix-free β=0.5 64x64 | 6.1 | 7.5 | 1.231 | 28 % | 22 % | inside | 1.239 (inside) |

The Kronecker rows time `kronecker_operator` of `innerₕ(u(1), v(1)) + inner₊ₓ(D₋ₓ(u(2)),
D₋ₓ(v(2))) + innerₕ(u(2), v(2)) + innerₕ(D₋ₓ(u(1)), v(2))`, on two components (three blocks,
no idle rows, so `_scale_only!` is not reached) and on three (the third component idle,
so the shared `_scale_only!` scales its rows). The matrix-free rows time
`inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v)` on one component, where `_scale_only!` runs on the
whole of `y` at every call. Both operators use their default policy.

Kronecker block. The idle-component rows agree to 0.1 % at `β = 0.5` and 1 % at `β = 1`,
inside the spread. The 3-block `β = 0.5` row is outside the spread over three runs (1.021, one
tip launch at 12.9 µs against 11.8 to 11.9 µs on the base) and inside over six (1.009). That
operator does not call `_scale_only!`, so the row does not test S3.6; it reads as noise of
this size on a 12 µs call.

Matrix-free. The medians differ here by more than the launches of one tree do: 1.25 and
1.23 over three launches, 1.26 and 1.24 over six. The per-call cost has not changed. Each
tree has two clusters of launches, a fast one (5.6 µs at
`β = 1`, 5.9 µs at `β = 0.5`) and a slow one (7.2 and 7.5 µs), and a launch lands in one for
its whole run. Over 10 launches of each tree (the six above and four more of the
matrix-free and Kronecker rows alone, run alternately) the fast cluster is 5.67 µs on the base
and 5.55 µs on the tip at `β = 1` (5.94 and 5.93 µs at `β = 0.5`), the slow one 7.16 µs and 7.19 µs
(7.47 and 7.47 µs). The base was slow in 3 of the 10 launches and the tip in 8 of 10 (Fisher
exact test, two-sided, p = 0.07), which is what moves the median. At equal cluster the two
trees run the same time to about 2 %, so `_scale_only!` and the `isone` branch it gained cost
nothing measurable per call. The difference in how often a launch lands in the slow cluster is
not significant and was not traced to a cause (code placement, array alignment and the
scheduler's choice of core are all candidates; macOS gives no way to pin a thread to a
core). Part 3 reworks this path and should re-measure it.

## Reading

No path shows a slowdown outside its spread that a subplan caused. Over six launches the
solve entry, the fast-diagonalisation `ldiv!` and the pair replay are within 3 % of the base
(7 % for `accelerate_solve(a, l)` at 8 x 8, a row with 22 % spread) and inside the
launch-to-launch spread. The Kronecker block `mul!` is inside it too, apart from one row that
clears it over three launches and not over six, on an operator that never reaches
`_scale_only!`. The matrix-free `mul!` has a 25 % higher median, from more launches landing in
the slow cluster. Part 3 should re-measure it.
