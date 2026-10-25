# Before and after timings of the v4.1.3 performance branch (S5.1)

Records the timing of each change on `v4.1.3/perf` (HEAD `7537cd5c`) against main `9c34251c`,
and against v4.0.0 (`0aa87e45`) where the issue asks for it. Issues: #876, #884, #886, #895, #902,
#906, #907 (and #883 for the first polygon call). Earlier notes hold their own figures and are
referenced, not repeated: `ghost-form-v4.1.3.md` (S3.1), `precompile-v4.1.3.md` (S3.2, S3.2.1),
`polygon-ttfx-v4.1.3.md` (S3.3).

## Verdicts

| issue | criterion | verdict | deciding figure |
|---|---|---|---|
| #886 Jacobi | setup time scales with threads on at least 10^5 non-uniform DOFs | met | ghost box 4 threads: 3.04x (`CpuThreaded`) and 3.24x (`CpuPolyester`) over main; plain box 2.02x and 2.35x. Costs: plain box at 1 thread is 1.12-1.32x slower (banded walk), `CpuSerial` ghost at 4 threads is 5-7% slower |
| #876 ghost-box and padded assembly | padded/plain `assemble!` at most 1.2 | met for the polygon padded form and the single-component ghost box (serial 2.5 to 1.0; ghost box 2.25x faster serial, 1.22x `CpuPolyester`), **not met for composite ghost forms**: traction (elasticity) assembly is 1.57x (2D) and 2.0x (3D) slower than main, cause S2.2 | polygon_vs_plain; `## Facet widening` |
| #884 polygon per DOF | at most 2x the rectangle | met serial (1.5 assemble and mul!, was 6.2-7.4) and `CpuPolyester` in the fast mode (1.7-1.8). `CpuThreaded` P `mul!` is 2.2 in the fast mode (D 2.0). Per-process ratios scatter 0.54-4.9 under `CpuPolyester` because the rectangle's own time is bimodal; the mode is not in the polygon walk | 12 fresh processes |
| #902 facet widening | `_face_margins` change measured on a non-uniform mesh | the widening itself is a 5% (2D) to 10% (3D) gain; the traction path as a whole is slower than main because of S2.2 | `## Facet widening` |
| #895 surface weights | no per-entry work added, assembly time unchanged | **not met on plain boundary-only forms**: `inner_Γ(u, v; markers = :boundary)` `assemble!` +53%, `mul!` +58% (27 and 18 us at 250x250); Laplacian plus `inner_Γ` +6% and +17%; pure Laplacians unchanged (0.98-1.00). Cause S4.2 (`9f4e07e5`), not S4.1 | `## Surface weights` |
| #906 TTFX | precompile within +0.8 s, or the cost accepted | **exceeded**: core precompile 91.8 s to 105.6 s (+13.8 s), accepted per piece in the S3.2, S3.2.1 and S3.3 notes; harness first-call cases unchanged (they are cached on both sides); threaded first calls up 0.4 to 3 s | `## TTFX` |
| #907 ghost form | serial 2D, 1D, 3D ghost `form` back to v4.0.0 | met: 1.009 (2D), 1.011 (1D), 0.958 (3D) of v4.0.0, main 1.07 | `## Ghost form` |

Two findings the earlier subplans did not record, both found by this run and bisected to a
commit by a built-tree comparison (no source change): composite ghost forms slow down at
`29119591` (S2.2), and plain surface forms at `9f4e07e5` (S4.2).

## Conditions

Julia 1.13.2, Apple Silicon (8 logical cores, 4 threads except where the table says), default `-O`
(`opt_level` 2; `polygon_vs_plain.jl` and the scratch scripts refuse to time below it), AC power,
alone (no other Julia process). Every `julia` went through `julia_slot.sh 4`. Non-uniform meshes
only: seeded random meshes (seed 2) or graded meshes, with the spacing ratio checked (Jacobi box
spacing 2.4e-5 to 1.7e-2).

`gate_preflight.sh --timing` failed on load before every batch (1-minute load 2.9 to 5.8, from
the system daemons `corespotlightd` and `mobileassetd`; no other Julia). The figures were taken
under that load, so **all of them are "under load"**. The 1-minute load while each batch ran, from
the load line written before and after each process: `polygon_vs_plain` 2.9-5.5, ratio
distribution 2.4-3.4, Jacobi 2.6-3.1, facet and surface 2.7-6.6 (round 3 reached 6.6 for one
process pair), ghost form 2.8-5.2, threaded first calls 3.0-3.9, bisect 1.8-3.0, TTFX 2.0-2.9. Main
and the branch alternated within every round (the order swapped between rounds), so each
comparison is back to back; ratios held across rounds where the load changed (the facet ratio is
1.55-1.61 in rounds with load 2.7 and 6.6). A figure inside its round-to-round spread is
called "no change".

Trees: the branch is `~/worktrees/Bramble/v4.1.3-perf`; main and v4.0.0 are detached worktrees
(`s51-main` at `9c34251c`, `s51-v400` at `0aa87e45`) with the branch `Manifest.toml` copied in, so
only Bramble's source differs (as in S3.1). The TTFX harness uses each tree's own Manifest. The
bisect trees `s51-c1` to `s51-c4` are `29119591`, `bfc2c26c`, `1c57861a`, `9f4e07e5`. All were
removed at the end. Scripts and raw output are in `~/bramble-scratch/v413perf/s51/`
(`runs/`).

## Jacobi

`jacobi_preconditioner(a; policy)` on a 330x330 non-uniform box: plain `gridspace` (108900 DOFs)
and `gridspace(Ω; ghosts = :boundary)` (110220 DOFs), form `-Δₕ + 0.5 Dcₓ`. One fresh process per
thread count, three rounds, main and branch alternating; each cell the minimum of 15 calls after
two warm-ups, then the minimum over the rounds, in ms. Ratios are branch/main per round; speed-up
is main/branch of the minima. On main every policy walks serially (the `policy` argument is
ignored), so main's three policies agree.

| threads | form | policy | main, ms | branch, ms | branch/main per round | speed-up |
|---:|---|---|---:|---:|---|---:|
| 1 | plain | CpuSerial | 0.373 | 0.373 | 1.000 0.996 1.046 | 1.00 |
| 1 | plain | CpuThreaded | 0.374 | 0.483 | 1.315 1.288 1.311 | 0.78 |
| 1 | plain | CpuPolyester | 0.374 | 0.468 | 1.123 1.251 1.247 | 0.80 |
| 1 | ghost | CpuSerial | 3.209 | 3.237 | 1.014 0.981 1.010 | 0.99 |
| 1 | ghost | CpuThreaded | 3.210 | 3.257 | 1.026 0.997 1.017 | 0.99 |
| 1 | ghost | CpuPolyester | 3.217 | 3.246 | 1.004 1.009 1.008 | 0.99 |
| 2 | plain | CpuSerial | 0.372 | 0.372 | 0.996 1.007 1.001 | 1.00 |
| 2 | plain | CpuThreaded | 0.375 | 0.276 | 0.737 0.763 0.736 | 1.36 |
| 2 | plain | CpuPolyester | 0.374 | 0.261 | 0.731 0.713 0.665 | 1.43 |
| 2 | ghost | CpuSerial | 3.288 | 3.331 | 1.022 1.009 1.040 | 0.99 |
| 2 | ghost | CpuThreaded | 3.209 | 1.780 | 0.546 0.549 0.560 | 1.80 |
| 2 | ghost | CpuPolyester | 3.253 | 1.727 | 0.514 0.537 0.529 | 1.88 |
| 4 | plain | CpuSerial | 0.372 | 0.374 | 1.010 1.003 1.002 | 0.99 |
| 4 | plain | CpuThreaded | 0.375 | 0.185 | 0.493 0.591 0.535 | 2.02 |
| 4 | plain | CpuPolyester | 0.375 | 0.160 | 0.425 0.428 0.428 | 2.35 |
| 4 | ghost | CpuSerial | 3.291 | 3.457 | 1.071 1.062 1.050 | 0.95 |
| 4 | ghost | CpuThreaded | 3.209 | 1.055 | 0.334 0.327 0.338 | 3.04 |
| 4 | ghost | CpuPolyester | 3.208 | 0.989 | 0.313 0.319 0.308 | 3.24 |

Scaling inside the branch (its 1, 2, 4 thread minima): plain `CpuThreaded` 0.483, 0.276, 0.185 ms
(2.6x at 4 threads), plain `CpuPolyester` 0.468, 0.261, 0.160 (2.9x), ghost `CpuThreaded` 3.257,
1.780, 1.055 (3.1x), ghost `CpuPolyester` 3.246, 1.727, 0.989 (3.3x). Against main the gain
grows with the thread count in every row (#886 AC2 met).

Costs, both beyond the spread of the rounds (about 1-4% between rounds for the serial rows):

- the plain box at 1 thread, where the threaded policies cost 25-31% (`CpuThreaded`) and 12-25%
  (`CpuPolyester`) over main's serial walk: the walk is cut into bands and has nothing to gain from
  them (0.37 ms against 0.47 ms, an absolute 0.1 ms);
- the `CpuSerial` ghost row at 4 threads, 5-7% over main (3.29 to 3.46 ms; at 1 and 2 threads
  0.98-1.04). S1.1 recorded that serial ghost builds keep one `ndofs` vector per band, so the
  band count follows the thread count; this table does not isolate that as the cause.

## Polygon and ghost assembly

`benchmark/polygon_vs_plain.jl` (S2.1), run unchanged on both sides, 4 threads, three rounds with
main and branch alternating. Cases: Figure 1 polygon `(40, 36)` (6617 DOFs, closing to 14577
padded rows), disk level set `(60, 60)` (10804 DOFs), rectangles of about the same DOFs, the plain
box and the ghost box on the Figure 1 box. Warm `assemble!` and matrix-free `mul!`, ns per DOF,
the minimum of 30 calls, for rounds 1, 2, 3. Allocations of one warm call of each: **0 B in every
row, both sides, under `CpuSerial` and `CpuPolyester`**.

| policy | case | DOFs | `assemble!` main | `assemble!` branch | `mul!` main | `mul!` branch |
|---|---|---:|---|---|---|---|
| CpuSerial | fig1 polygon | 6617 | 43.2 42.5 42.6 | 10.5 10.5 10.6 | 42.9 42.7 43.0 | 9.3 9.3 9.4 |
| CpuSerial | fig1 padded | 14577 | 16.4 16.6 16.6 | 6.7 6.5 6.5 | 18.0 18.1 18.2 | 9.0 8.8 8.8 |
| CpuSerial | disk | 10804 | 49.6 49.7 49.6 | 10.5 10.5 10.5 | 50.3 50.5 50.9 | 9.0 9.0 9.0 |
| CpuSerial | disk padded | 29232 | 16.0 16.0 16.0 | 6.4 6.4 6.4 | 17.4 17.5 17.5 | 8.2 8.2 8.4 |
| CpuSerial | rect ~ fig1 (81x81) | 6561 | 6.8 6.8 6.8 | 6.8 6.8 7.1 | 6.0 6.0 6.0 | 6.0 6.1 6.2 |
| CpuSerial | rect ~ disk (104x104) | 10816 | 6.7 6.8 6.7 | 6.7 6.7 6.9 | 6.0 6.0 6.0 | 6.0 6.0 6.2 |
| CpuSerial | plain box | 14577 | 6.7 6.7 6.7 | 6.7 6.6 7.0 | 5.9 5.9 5.9 | 5.9 5.9 6.1 |
| CpuSerial | ghost box | 15061 | 22.4 22.5 22.5 | 10.0 9.9 10.2 | 19.4 19.5 19.6 | 10.3 10.8 10.5 |
| CpuPolyester | fig1 polygon | 6617 | 9.3 9.2 9.2 | 4.8 4.8 4.8 | 6.9 6.9 6.9 | 3.1 3.0 3.1 |
| CpuPolyester | fig1 padded | 14577 | 3.1 3.0 3.0 | 2.7 2.7 2.7 | 2.7 2.7 2.7 | 2.4 2.4 2.4 |
| CpuPolyester | disk | 10804 | 23.6 10.3 10.3 | 4.7 4.7 4.7 | 7.8 7.8 7.8 | 3.0 3.0 3.0 |
| CpuPolyester | disk padded | 29232 | 2.9 2.9 6.7 | 2.6 2.6 2.6 | 5.9 2.5 2.5 | 2.3 2.5 2.3 |
| CpuPolyester | rect ~ fig1 | 6561 | 6.0 2.6 5.9 | 2.6 2.6 5.8 | 1.7 1.7 1.7 | 1.7 1.7 1.7 |
| CpuPolyester | rect ~ disk | 10816 | 2.6 2.5 2.5 | 2.5 2.5 2.6 | 1.7 1.7 1.7 | 1.7 1.7 5.9 |
| CpuPolyester | plain box | 14577 | 2.6 2.5 2.5 | 2.5 2.5 2.5 | 1.7 1.7 1.7 | 1.7 1.7 4.5 |
| CpuPolyester | ghost box | 15061 | 5.6 5.6 5.7 | 4.6 4.5 4.6 | 3.2 3.2 3.2 | 2.9 2.9 2.9 |

The printed ratios (polygon per DOF over the rectangle's, same policy, same process):

| ratio | main | branch |
|---|---|---|
| serial, fig1 `assemble!` / `mul!` | 6.2-6.4 / 7.1 | 1.5 / 1.5 |
| serial, disk `assemble!` / `mul!` | 7.3-7.4 / 8.4-8.5 | 1.5-1.6 / 1.4-1.5 |
| serial, padded/plain `assemble!` | 2.5 | 0.9-1.0 |
| `CpuPolyester`, padded/plain `assemble!` | 1.2 | 1.1 |
| serial, ghost box `assemble!` / `mul!` per DOF | 22.5 / 19.5 | 10.0 / 10.5 |

- #876: the padded form costs what the plain box costs serially (0.9-1.0, criterion 1.2; main
  2.5). The ghost box gains 2.25x (`assemble!`) and 1.9x (`mul!`) serially and 1.22x and 1.10x
  under `CpuPolyester`.
- #884, serial: 1.5 on both quantities and both cases (criterion 2.0; main 6.2-8.5). The
  `CpuPolyester` polygon rows are 4.8 and 3.0 ns per DOF against main's 9.2 and 6.9-7.8.
- The `CpuPolyester` ratios the script prints are not reliable per process (round 3 above:
  fig1 0.8, disk 0.5), because the rectangle rows have a slow mode of their own: see the next
  subsection.

### The CpuPolyester and CpuThreaded ratio over 12 fresh processes (S2.3.1 MANUAL)

The S2.3.1 PROBE, as `~/bramble-scratch/v413perf/s51/ratio.jl`, run in 12 fresh processes on the
branch and 3 on main (interleaved), each timing `CpuPolyester` then `CpuThreaded` on Figure 1 (P)
and the disk (D) and the 83x83 rectangle (6889 DOFs), the minimum of 30 calls, and printing the
absolute ns per DOF beside the ratio. Stored rows equal `ndofs(V)` in every process (6617, 12960;
main 14577 and 32942). Ratios are polygon over rectangle per DOF, sorted over the 12 processes:

| policy, case | quantity | 12 sorted values | fast mode |
|---|---|---|---|
| CpuPolyester P | `assemble!` | 0.85 1.79 1.80 1.80 1.80 1.80 1.80 1.80 1.81 1.81 3.48 3.50 | 1.80 |
| CpuPolyester P | `mul!` | 0.57 0.57 0.58 0.58 0.58 1.54 1.80 1.81 1.81 1.81 1.82 1.84 | 1.80-1.84 |
| CpuPolyester D | `assemble!` | 0.83 1.73 1.73 1.73 1.74 1.74 1.75 1.75 1.75 3.43 3.43 3.45 | 1.73-1.75 |
| CpuPolyester D | `mul!` | 0.54 0.54 0.55 0.55 0.55 1.54 1.73 1.73 1.73 1.73 1.74 4.87 | 1.73-1.74 |
| CpuThreaded P | `assemble!` | 1.33 1.53 1.55 1.58 1.60 1.62 1.66 1.68 1.75 1.75 1.75 2.07 | 1.53-1.75 |
| CpuThreaded P | `mul!` | 0.68 0.68 2.16 2.17 2.19 2.21 2.21 2.23 2.24 2.70 5.45 5.96 | 2.16-2.24 |
| CpuThreaded D | `assemble!` | 0.84 0.91 0.98 0.99 1.00 1.03 1.04 1.05 1.06 1.36 1.48 1.58 | 0.9-1.06 |
| CpuThreaded D | `mul!` | 0.63 0.64 1.83 1.98 1.98 1.99 2.00 2.03 2.06 4.27 4.52 4.71 | 1.98-2.06 |

On main (3 processes): `CpuPolyester` P `assemble!` 3.5, 3.5, 7.7 and `mul!` 1.3, 1.3, 11.1
(fast-mode absolutes 9.3 and 7.0 ns per DOF, so 3.5 and 4.1 against the fast rectangle);
`CpuThreaded` P `mul!` 12.2, 12.8, 19.3 and `assemble!` 2.3-2.9.

Reading, from the absolutes the script prints:

- The ratios are bimodal because the **denominators and numerators each have a slow mode**. The
  83x83 rectangle's `CpuPolyester` `mul!` is 1.69-1.70 ns per DOF in 6 of 12 processes and
  5.29-5.41 in the other 6 (3.1x); its `assemble!` is 2.66-2.70 in 11 and 5.59 in one. The
  polygon's own `CpuPolyester` `mul!` is 2.92-3.12 in 10 of 12 processes and 8.20-8.28 in 2
  (2.7x; process 4 in both cases, process 11 in the disk only), its `assemble!` 4.64-4.87 in 9
  and 9.15-9.36 in 3 (2.0x). The 0.54-0.58 `mul!` ratios are
  the processes with a slow rectangle and a fast polygon; the 4.87 (D, process 11) is the
  reverse. The mode changes inside a process: in process 11 the P `mul!` is 3.07 and the D
  `mul!` 8.28 with the same rectangle (1.70), and in process 5 only the rectangle's `assemble!`
  is slow.
- The slow mode is therefore not the polygon walk: it appears on the plain rectangle (no polygon
  code), in `CpuThreaded` too (rectangle `mul!` 6.0 against 1.9 in processes 7 and 10, polygon P
  `mul!` 11.2 against 4.2 in processes 1 and 4), and on main (rectangle `mul!` 5.3 in 2 of 3
  processes, the same absolute value). Its factor is 2-3x on every threaded `mul!` and 2x on
  `assemble!`. The cause of the process-level mode was not found: the figures show it is
  per process, per time window, and independent of the code; they do not show whether it is
  thread placement or the system load (1-minute 2.4-3.4 here).
- The 2x criterion, read in the fast mode (the minimum of each column over the processes, which
  is what a quiet machine gives): `CpuPolyester` 1.7-1.8 on P and D, both quantities, met.
  `CpuThreaded` D 1.0 (`assemble!`) and 2.0 (`mul!`); **`CpuThreaded` P `mul!` 2.2 exceeds the
  criterion by 10%** at 4 threads (S2.3.1's worker had 1.8-1.9 at 2 threads, 4.1 ns per DOF
  against the rectangle's 1.9). Read per process, the criterion cannot be decided: the ratio
  of one process is 0.54 to 4.9 on the same code.

## Facet widening

Traction forms (#902, S2.4): linear elasticity `-μ Δₕ(uᵢ) - (λ+μ) ∂ᵢ(div u)` on a composite
`gridspace(Ω; ghosts = :boundary)^Val(D)` constrained by traction relations (zero data; the
walk sees only the structure) on `:xmin` (and `:ymin` in 2D) and displacement on the other
facets, as in `test/ghost/traction.jl`. 2D random (seed 2) 200x200 and graded (`x^2`) 200x200
(79202 DOFs), 3D random 26^3 (43200 DOFs). Warm `assemble(a)` of the prebuilt form, the minimum
of 15 calls (9 for 3D), and the minimum of `form(C, C, f)`. Three rounds, main and branch
alternating; times in ms; "main" and "branch" are the minima over the rounds.

| case | quantity | main | branch | branch/main per round | ratio of minima |
|---|---|---:|---:|---|---:|
| 2D random 200^2 | `assemble` | 7.010 | 11.028 | 1.551 1.573 1.612 | 1.573 |
| 2D graded 200^2 | `assemble` | 6.954 | 10.876 | 1.600 1.608 1.563 | 1.564 |
| 3D random 26^3 | `assemble` | 17.765 | 35.100 | 1.996 1.976 1.998 | 1.976 |
| 2D random 200^2 | `form` | 67.362 | 56.020 | 0.813 0.967 0.807 | 0.832 |
| 2D graded 200^2 | `form` | 59.955 | 55.641 | 0.820 0.811 0.950 | 0.928 |
| 3D random 26^3 | `form` | 126.139 | 108.972 | 0.977 0.864 1.265 | 0.864 |

The traction forms assemble **1.56-1.57x (2D) and 1.98x (3D) slower than on main**, `form` 7-17%
faster (the minima; the medians are GC-bound and vary 0.9-1.3).

Bisect (one process per tree, 4 threads, same script, `FACET_SAMPLES=9`; `assemble` minima, ms;
the trees agree within about 1% in 2D and 3% in 3D):

| tree | 2D random | 2D graded | 3D random |
|---|---:|---:|---:|
| main `9c34251c` | 6.90 | 6.86 | 17.43 |
| `29119591` (S2.2, factored walk on ghost spaces) | 11.42 | 11.40 | 38.41 |
| `bfc2c26c` (S2.4, widening 0) | 10.80 | 10.80 | 34.60 |
| `1c57861a` (S2.3 and S4.1) | 10.86 | 10.82 | 34.79 |
| `9f4e07e5` (S4.2) | 10.91 | 10.81 | 35.52 |
| branch `7537cd5c` | 10.86 | 10.83 | 36.34 |

- The jump is S2.2 (`29119591`: 1.65x in 2D, 2.2x in 3D). **S2.4's widening of 0 is a gain** of 5%
  (2D) and 10% (3D) on that tree, and the later steps are inside the spread. So #902's
  timing question is answered (the widening does not cost anything and gains 5-10%), and the
  composite slowdown belongs to #876's change.
- A second probe splits the 2D form by term (`facet2.jl`, composite, 2D random 200^2, ms): the
  constrained `assemble` of the second differences only 3.26 (main), 5.67 (`29119591`), 5.89
  (branch); of the mixed `Dc Dc` terms only 3.56, 5.72, 5.13; of one Laplacian 0.96, 1.60, 1.61.
  The padded `assemble!` of the same forms on the padded composite: 2.68, 5.12, 5.12; 2.55,
  3.62, 3.62; 0.66, 1.31, 1.31. So any form on the two-component padded space is 1.4-2.0x slower
  after S2.2, while a single-component ghost box with the form `-Δₕ + 0.5 Dcₓ` is 2.3x faster
  (`## Surface weights`, `## Polygon and ghost assembly`). S2.2's thresholds (padded/plain at
  most 1.2) were measured on single-component spaces. The cause inside S2.2's walk was not
  searched.

## Surface weights

Warm `assemble!(A, a)` and matrix-free `mul!(y, op, x)` (minimum of 30 calls), and the minimum
of `form` (9 calls, the timer reads 0 for lazy plain forms and the rows are left out), on a 250x250
non-uniform box (seed 2), plain and `ghosts = :boundary`. Three rounds, main and branch
alternating; times in ms; ratio columns are branch/main per round and of the minima.

| case | quantity | main | branch | per round | ratio of minima |
|---|---|---:|---:|---|---:|
| plain Laplacian `inner₊(∇ₕ(u), ∇ₕ(v))` | `assemble!` | 0.324 | 0.317 | 0.977 0.980 0.972 | 0.980 |
| plain Laplacian | `mul!` | 0.069 | 0.069 | 1.001 0.312 1.015 | 1.001 |
| plain `-Δₕ + 0.5 Dcₓ` | `assemble!` | 0.410 | 0.409 | 1.006 1.003 0.995 | 0.998 |
| plain `-Δₕ + 0.5 Dcₓ` | `mul!` | 0.350 | 0.350 | 1.000 1.000 0.998 | 1.000 |
| plain `inner_Γ(u, v; markers = :boundary)` | `assemble!` | 0.053 | 0.080 | 1.528 1.532 1.524 | 1.529 |
| plain `inner_Γ` | `mul!` | 0.030 | 0.048 | 1.541 1.621 1.551 | 1.579 |
| plain `inner_Γ` | `form` | 0.012 | 0.009 | 0.724 0.769 0.718 | 0.744 |
| plain Laplacian + `inner_Γ` | `assemble!` | 0.380 | 0.402 | 1.064 1.092 1.028 | 1.057 |
| plain Laplacian + `inner_Γ` | `mul!` | 0.097 | 0.114 | 1.123 1.076 1.204 | 1.172 |
| plain Laplacian + `inner_Γ` | `form` | 0.010 | 0.009 | 0.840 0.854 0.878 | 0.857 |
| ghost `-Δₕ + 0.5 Dcₓ` | `assemble!` | 1.443 | 0.635 | 0.435 0.440 0.438 | 0.440 |
| ghost `-Δₕ + 0.5 Dcₓ` | `mul!` | 1.183 | 0.602 | 0.508 0.512 0.525 | 0.509 |
| ghost `-Δₕ + 0.5 Dcₓ` | `form` | 23.638 | 23.349 | 0.996 0.988 0.997 | 0.988 |
| ghost `inner_Γ(∂ₙ(u), v; markers = :boundary)` | `assemble!` | 1.726 | 0.627 | 0.365 0.363 0.365 | 0.363 |
| ghost `inner_Γ(∂ₙ)` | `mul!` | 2.376 | 0.909 | 0.381 0.374 0.383 | 0.382 |
| ghost `inner_Γ(∂ₙ)` | `form` | 19.270 | 17.520 | 0.909 0.811 0.925 | 0.909 |
| ghost `-Δₕ + inner_Γ(∂ₙ)` | `assemble!` | 2.509 | 0.981 | 0.397 0.386 0.394 | 0.391 |
| ghost `-Δₕ + inner_Γ(∂ₙ)` | `mul!` | 2.915 | 1.218 | 0.416 0.428 0.418 | 0.418 |
| ghost `-Δₕ + inner_Γ(∂ₙ)` | `form` | 29.989 | 26.300 | 0.877 0.880 0.861 | 0.877 |

(The `mul!` round-2 ratio 0.312 of the plain Laplacian is one disturbed main sample, load 3.1; the
other two rounds and the minima agree at 1.00.) Warm `mul!` allocates 0 B in every row, both
sides.

- No regression on the Laplacian forms (0.98-1.00, inside the spread of 1-3%), the plain
  `-Δₕ + 0.5 Dcₓ` and every ghost form are unchanged or faster (the ghost gains are S2.2's).
- **Plain boundary-only surface forms are 1.53x (`assemble!`) and 1.58x (`mul!`) slower**, and
  a Laplacian plus `inner_Γ` 1.06x and 1.17x. The O22 criterion (assembly time unchanged) is not
  met for them. In absolute terms the loss is 27 us and 18 us on a boundary-only form of a
  250x250 mesh, 22 us on a 0.38 ms `assemble!` and 17 us on a 0.10 ms `mul!` of the mixed form.
- Bisect (one process per tree, same script; `assemble!` / `mul!` minima in ms, plain `inner_Γ`):
  main 0.0528 / 0.0301, `29119591` 0.0527 / 0.0299, `bfc2c26c` 0.0547 / 0.0300, `1c57861a`
  (S4.1) 0.0528 / 0.0304, `9f4e07e5` (S4.2) 0.0803 / 0.0477, branch 0.0805 / 0.0478. The step
  is S4.2 alone; S4.1's field is free. For the mixed Laplacian + `inner_Γ` form `mul!`: 0.1005,
  0.1030, 0.0972, 0.1002, 0.1185, 0.1170 (the step at S4.2 again). S4.2's evidence names a trade
  off, surface products no longer pairing in `_pair_plan` and being walked separately; this run
  shows where the time goes up, not that this is the reason.
- `form` of the surface forms is 7-26% faster in the minima, a few microseconds (the timer's
  resolution); the concrete return type is S4.2's gain and is not a warm-walk figure.

## TTFX

`benchmark/ttfx.jl --before ~/worktrees/Bramble/v4.0.0 --after ~/worktrees/Bramble/v4.1.3-perf
--runs 3 --threads 4`, run alone at 14:50 on AC, 1-minute load 2.0 to 2.9, the harness's own load
gate passed (threshold 4.0). Each tree builds in a fresh depot with its own Manifest.

| quantity | v4.0.0 | branch | difference |
|---|---:|---:|---:|
| core precompile, median of 3 | 91.78 s (91.78, 91.93, 91.78) | 105.59 s (105.59, 105.83, 105.42) | **+13.80 s**, ceiling +0.8 s |
| extension precompile after the core, info only | 91.64 s | 83.43 s | -8.2 s (info only, not a criterion) |
| `pde_solve(A, F)` first call | 27.4 ms | 27.5 ms | no change |
| `assemble_add!(A, a, 0.5)` | 0.02 ms | 0.02 ms | no change (microseconds) |
| `suitesparse_solve` | 1.9 ms | 1.8 ms | no change |
| `sparspak_solve` | 20.4 ms | 20.8 ms | no change |
| `fdm_solve(a, F)` | 13.2 ms | 9.8 ms | -3.4 ms |
| `matrix_free_apply` / `jacobi` / `chebyshev` / `gmg_vcycle` | 1.6 / 7.3 / 17.7 / 9.7 ms | 1.6 / 7.3 / 17.9 / 10.0 ms | no change |
| `shift_gridfunction` | 53.1 ms | 23.7 ms | -29.5 ms |
| `shift_form` / `polyester_matrix_free` | 11.5 / 5.7 ms | 11.8 / 6.1 ms | no change (inside the 3-run spread) |

The harness reports 12 FAILs and one PASS. Only the first is the AC3 criterion of #906: the
core precompile is +13.8 s against +0.8 s. The case thresholds (150 ms for `pde_solve`, 80 ms for
`assemble_add!`, and so on) are the before-minus-after reductions that #283 and #284 required
against a build without a workload; v4.0.0 already carries those workloads, so a v4.0.0 against
branch comparison cannot show them (the first calls are tens of ms or less on both sides). The
"before" value of the case rows is therefore a baseline, not a regression: nothing slower than the
spread (the 3-run ranges are within 5%).

The +13.8 s is the sum of the pieces the earlier notes priced, each accepted there or by the
user's decision of 2026-10-25:

| piece | precompile | pkgimage | first-call effect | source |
|---|---:|---:|---|---|
| `_pc_polygon` (v4.1.0) | 4.9 s | 11.9 MiB | the polygon workload | `precompile-v4.1.3.md`, `## Per-session cost` |
| ghost operators and ghost `semidiscretize` | 4.3 s | 10.5 MiB | | same |
| S3.3: extended `_pc_polygon` (Laplacian plus convection pair) | +1.9 s | +1.9 MiB | first polygon pipeline 1.31 s to 0.65 s (2.26x to 1.30x the rectangle's) | `polygon-ttfx-v4.1.3.md`, `## Result` |
| S3.2.1: partial-ghost Robin workload | +1.5 s | +3.2 MiB | first call of the tutorials' snippet 2.43 s to 0.98 s | `precompile-v4.1.3.md`, `## Partial-ghost workload` |

These four add to +12.6 s; the remaining about 1.2 s is the change in what the ghost and
polygon source (S2.2, S2.3, S2.3.1, S4.1, S4.2) gives the sessions to compile, plus the spread of
the two-tree comparison (3 runs per side, ranges under 0.4 s). The Float64 `_pc_polygon` call was
kept by S3.3's measurement (it saves 0.69 s on the user's pipeline). S3.2 measured +9.0 s for main
against v4.0.0 on the shared Manifest; this run, with each tree's own Manifest, gives +13.8 s for
the branch.

Threaded first calls (S2.3.1), on the 10x9 Figure 1 polygon (`polygon_spaces`, the form
`-Δₕ + 0.5 Dcₓ`), one fresh process per policy at 4 threads, 5 rounds with main and branch
alternating, median seconds (range) of the first call of each step (the precompile does not cover
either policy: the workload runs the default policy, and Polyester lives in an extension):

| policy | step | main | branch |
|---|---|---:|---:|
| CpuThreaded | `assemble` | 0.512 (0.49-0.53) | 0.890 (0.86-0.90) |
| CpuThreaded | `mul!` | 0.546 (0.52-0.55) | 1.787 (1.77-1.80) |
| CpuThreaded | `form` | 0.607 | 0.751 |
| CpuPolyester | `assemble` | 1.711 (1.69-1.73) | 2.470 (2.44-2.51) |
| CpuPolyester | `mul!` | 2.954 (2.93-3.06) | 5.996 (5.90-6.12) |
| CpuPolyester | `form` | 0.638 | 0.786 |

The threaded first calls cost 0.4 to 1.2 s more under `CpuThreaded` and 0.8 s (`assemble`) and 3.0 s
(`mul!`) more under `CpuPolyester`, each against its own steady state of microseconds (S2.3.1
recorded 0.6 to 1.6 s, 1.7 to 2.3 s and 2.8 to 4.9 s; the `mul!` here is 1.1 s above the 4.9 s
of its worker). Everything else in the first-call chain (mesh, spaces, `matrix_free_operator`)
is the same on both sides within 0.1 s. Serial first calls are unchanged by S2.3.1 and are covered
by S3.3's note. A user who runs a threaded polygon `mul!` once per session pays these seconds once.

## Ghost form

The serial `ghost vs plain` rows of `benchmark/ghost_vs_plain.jl` through the S3.1 harness
(`~/bramble-scratch/v413perf/s31/run.jl` with its 500 MB ballast array, one fresh process per
run, the v4.0.0 baseline's `evals` and `samples`, 4 threads). Three rounds, the order rotating
over v4.0.0, main and branch; rows are serial 2D, 1D and 3D `ghost form` and serial 3D plain
`assemble` as the canary. The "plain form" row (a 17 ns lazy constructor) is left out as in S3.1.
Ratios after/before.

| row | pair | median before, after | per-round median ratio | per-round minimum ratio | ratio of minima |
|---|---|---|---|---|---:|
| serial 2D, ghost form | v4.0.0 to branch | 27.872 to 28.158 ms | 1.010 1.002 1.005 | 1.002 1.009 1.004 | **1.009** |
| serial 1D, ghost form | v4.0.0 to branch | 17.482 to 17.697 ms | 1.007 1.018 1.016 | 1.022 1.019 1.010 | **1.011** |
| serial 3D, ghost form | v4.0.0 to branch | 17.654 to 16.921 ms | 0.958 0.973 0.949 | 0.967 0.965 0.958 | **0.958** |
| canary: serial 3D, plain assemble | v4.0.0 to branch | 0.531 to 0.524 ms | 0.986 0.948 0.998 | 0.984 1.010 0.952 | 0.984 |
| serial 2D, ghost form | v4.0.0 to main | 27.872 to 29.885 ms | 1.072 1.083 1.068 | 1.066 1.073 1.068 | 1.073 |
| serial 1D, ghost form | v4.0.0 to main | 17.482 to 18.874 ms | 1.079 1.076 1.095 | 1.077 1.074 1.071 | 1.072 |
| serial 3D, ghost form | v4.0.0 to main | 17.654 to 19.053 ms | 1.085 1.076 1.075 | 1.070 1.060 1.061 | 1.061 |
| canary, v4.0.0 to main | | 0.531 to 0.529 ms | 1.016 0.919 1.040 | 0.992 0.976 0.995 | 0.991 |
| serial 2D, ghost form | main to branch | 29.885 to 28.158 ms | 0.943 0.925 0.941 | 0.940 0.941 0.940 | 0.941 |
| serial 1D, ghost form | main to branch | 18.874 to 17.697 ms | 0.934 0.946 0.928 | 0.949 0.948 0.943 | 0.943 |
| serial 3D, ghost form | main to branch | 19.053 to 16.921 ms | 0.883 0.904 0.882 | 0.903 0.910 0.903 | 0.903 |
| canary, main to branch | | 0.529 to 0.524 ms | 0.971 1.032 0.960 | 0.991 1.034 0.957 | 0.993 |

1-minute load at the start and end of each process (v4.0.0, main, branch), by round. Round 1
read 2.80 2.87, 2.87 3.19 and 3.19 3.06. Round 2 read 5.20 4.77, 3.49 5.20 and 3.06 3.49.
Round 3 read 3.92 3.30, 4.77 3.92 and 3.30 3.01.

- The slowdown #907 reported reproduces on main (1.06 to 1.07x) and is gone on the branch:
  2D 1.009 and 1D 1.011 (the per-round minima 1.002-1.022), at v4.0.0's time within the 1-2% of the
  runs; the canary's minima are 0.984 and 0.993 on this branch.
- 3D is 4% below v4.0.0 (0.958, per-round 0.949-0.973). The canary's own spread of 0.95-1.03
  per round covers this in the medians; the minima ratio (0.958 against the canary's 0.984) is
  in the right direction and is not claimed as a gain beyond the S3.1.1 fix.
- main to branch: 6% (2D), 6% (1D), 10% (3D) faster, which is S3.1.1's `@inline` (S3.1 measured
  0.983x of the parent) and S2.2's factored walk on the padded ghost leaves.
