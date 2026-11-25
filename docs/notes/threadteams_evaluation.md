# ThreadTeams against `Threads.@threads` and `Polyester.@batch` (S5.2, #1125)

Is ThreadTeams 2.0.0 a better backend for Bramble's threaded CPU policies than what they
use now? This note measures a prototype on Bramble's own paths and ends with a
recommendation. Planning-time figures are in §5 of `.claude/plans/alloc-guarantees-notes.md`.

## The prototype

Branch `eval/threadteams` (commits 573b5418 and 4e9b79aa, cut from `alloc-guarantees-2` at
`20fa87a0`; not for merging) has two pieces under `eval/`:

- `BrambleThreadTeams`, a package of its own standing in for a `BrambleThreadTeamsExt`
  extension, so it precompiles without a change to the root `Project.toml`. It fills the 17
  `_batch_*` hooks that `BramblePolyesterExt` fills, and is loaded in place of Polyester, so
  `CpuPolyester()` runs on the team. In the tables this is **tt-hooks**. It is 328 lines,
  against 1411 in `ext/BramblePolyesterExt.jl` plus the 293 of `src/utils/batch_split.jl`
  that only the Polyester path needs.
- `threaded_on_teams.jl`, which overwrites at run time the seven functions holding every
  `Threads.@threads` loop in `src/` and the two that `@spawn` matrix-free bands, so
  `CpuThreaded()` runs on the team. This is **tt-threaded** (73 lines). Method
  overwriting is refused during precompilation, so it cannot be a package; a real port would
  edit `src/`. Bramble's own Enzyme rules for those functions stay in force.

Every loop goes through `ThreadTeams.run_chunks` or `run_fold` with
`StaticSchedule(busy = :serial)`. The plan named `foreach_range`, but ThreadTeams' EnzymeCore
extension has rules for `run_chunks` and `run_fold` only, so a backend built on
`foreach_range` could not be differentiated. A loop of more than `nworkers()` items is split
into `nworkers()` contiguous blocks first. The chunks capture ordinary closures over forms,
spaces, sinks and user kernels: the team stores each closure in a slot typed on it, so none
of the splitting, raw pointers or argument slots of `BramblePolyesterExt` are needed. The
band bodies are `CpuThreaded`'s own (`_dot_band`, `_difference_band!`, ...), and reductions
fold left to right as `sum` adds fewer than 16 partials.

A coverage run (`--code-coverage=@eval/BrambleThreadTeams`) of the check below entered all
17 hooks. The first version of the prototype wrapped each replay point in a closure and swept
`_batch_for!` index by index. That cost up to 2.9 times `@batch` on the linear and bilinear
assembly rows. Rewritten as callable structs and contiguous slabs (as in
`BramblePolyesterExt`), those rows match `@batch`. The time table holds the rewritten
version for the 15 rows it touches, re-timed against `@batch` in three more alternating runs;
the multigrid rows keep the first version.

## Conditions

Apple M2 (4 performance and 4 efficiency cores), Julia 1.13.4, default `-O`, `--threads=4`,
`--startup-file=no`, every run through `julia_slot.sh 4` with the machine otherwise idle of
Julia. Environment: a scratch copy of the workspace (`~/bramble-scratch/s52-tt`) that resolved
to the #1284 stack: KernelAbstractions 0.10.8, ThreadTeams 2.0.0, Enzyme 0.13.221, Metal
1.12.1, GPUArrays 12.1.1, LLVM 10.7.0, Polyester 0.7.19. All four policies were measured in
it; none of the figures come from S5.1's `benchmark` environment. `gate_preflight.sh --timing`
passed before the timing runs; the load and Metal runs started at a load average of 2.2
(threshold 2).

Polyester and the prototype define the same methods, so they run in separate processes: the
**poly** process has serial, threads (`CpuThreaded` on `Threads.@threads`) and batch
(`CpuPolyester` on `@batch`); the **tt** process has serial, tt-threaded and tt-hooks.

Paths: those of `benchmark/cputhreaded_bytes.jl` (S5.1), plus rows for the hooks its paths
do not reach: `innerₕ`/`normₕ` reductions, `_dot`/`_dot_masked` with vector weights (a graded
tensor mesh's own weights are separable), `Rₕ!` and `avgₕ!` with a closure over an array,
`Rₕ!` into a composite space, a `CartesianIndices` sweep, `mul!` with a
`kronecker_operator`, the explicit right-hand side of `semidiscretize`, and an `n²/3 × 3`
mesh, too thin to band, on which the bilinear and linear sweeps colour points. Meshes are the
unit square at 65 × 65 and 513 × 513 points, graded: axis `d` has nodes `t^(1 + d/4)`.

Scripts, all on `eval/threadteams`: `eval/threadteams_eval.jl` (tasks `check`, `bytes`,
`time`, `wake`, `allocs`), `eval/run_measure.sh`, `eval/load_eval.jl`,
`eval/enzyme_tt_driver.jl` with `eval/enzyme_policy_cases_tt.jl` (a copy of
`test/ext/enzyme_policy_cases.jl` that loads the prototype instead of Polyester), and
`eval/tables.py`, which made the tables below.

## Results agree bitwise

For each of the 42 paths at both sizes, the tt process's results hash equal to the poly
process's: tt-threaded to threads and tt-hooks to batch, 168 pairs, none differ. The serial
results hash equal across the two processes. Against `CpuSerial`, 12 of the 42 paths differ
in the last bits at one size or both (bilinear assembly, `jacobian!`, multigrid, the
reductions; relative error at most 2.4e-15). Eleven differ by the same amount under all four
threaded arms. The explicit right-hand side differs more under batch and tt-hooks (9.3e-16)
than under threads and tt-threaded (9.7e-18), since the two policies take different routes to
the CSR product.

## Warm bytes per call

Median of 11 calls after 3 warm-up calls. Every row had the same bytes on all 11 calls,
and the same at 65 × 65 and 513 × 513 except where two figures are given.

| Path | serial | threads | batch | tt-threaded | tt-hooks |
|---|---:|---:|---:|---:|---:|
| `D₋ₓ!` | 0 | 1584 | 0 | 0 | 0 |
| `D₋ᵧ!` | 0 | 1584 | 0 | 0 | 0 |
| `D₊ₓ!` | 0 | 1616 | 0 | 0 | 0 |
| `Dcₓ!` | 0 | 1584 | 0 | 0 | 0 |
| `D̃ₓ!` | 0 | 1584 | 0 | 0 | 0 |
| `Mₓ!` | 0 | 1568 | 0 | 0 | 0 |
| `M₊ₓ!` | 0 | 1568 | 0 | 0 | 0 |
| `Mcₓ!` | 0 | 1568 | 0 | 0 | 0 |
| `∇cₕ!` | 0 | 3168 | 0 | 0 | 0 |
| `Δₕ!` | 0 | 0 | 0 | 0 | 0 |
| `divₕ!` | 0 | 3232 | 0 | 0 | 0 |
| `divcₕ!` | 0 | 3232 | 0 | 0 | 0 |
| `diṽₕ!` | 0 | 3232 | 0 | 0 | 0 |
| `curlₕ!` | 0 | 3232 | 0 | 0 | 0 |
| `curlcₕ!` | 0 | 3232 | 0 | 0 | 0 |
| `εₕ!` | 0 | 12864 | 0 | 128 | 0 |
| `v .= 2u + u*u` | 0 | 1664 | 0 | 96 | 0 |
| `v .= u + w` | 0 | 1632 | 0 | 64 | 0 |
| `assemble! innerₕ(f, v)` | 0 | 1648 | 0 | 0 | 0 |
| `assemble! innerₕ(f, v; markers)` | 0 | 1664 | 0 | 0 | 0 |
| `assemble! innerₕ(f, restrict_to(v))` | 0 | 1664 | 0 | 0 | 0 |
| `assemble! bilinear` | 0 | 8240 | 0 | 0 | 0 |
| `assemble! bilinear, masked term` | 0 | 9936 | 0 | 0 | 0 |
| `assemble! bilinear, dirichlet` | 0 | 8240 | 0 | 0 | 0 |
| `jacobian!` | 0 | 14064 | 0 | 832 | 0 |
| `mul! plain` | 0 | 1584 | 0 | 0 | 0 |
| `mul! masked term` | 0 | 1616 | 0 | 0 | 0 |
| `v_cycle!` | 0 | 41440 / 103600 | 0 | 0 | 0 |
| `ldiv! multigrid` | 0 | 41440 / 103600 | 0 | 0 | 0 |
| `assemble! bilinear, n²/3 × 3` | 0 | 8240 | 0 | 0 | 0 |
| `assemble! innerₕ(f, D₋ᵧ(v)), n²/3 × 3` | 0 | 3296 | 0 | 0 | 0 |
| `Rₕ!(composite, closure)` | 0 | 1648 | 0 | 0 | 0 |
| `_sweep_for!(CartesianIndices)` | 0 | 1584 | 0 | 0 | 0 |
| `_dot, vector weights` | 0 | 1680 | 0 | 96 | 0 |
| `_dot_masked, vector weights` | 0 | 1696 | 0 | 96 | 0 |
| `innerₕ(u, w)` | 0 | 1712 | 0 | 96 | 0 |
| `innerₕ(u, w; markers)` | 0 | 1712 | 0 | 96 | 0 |
| `normₕ(u)` | 0 | 1712 | 0 | 96 | 0 |
| `Rₕ!(u, closure over array)` | 0 | 1584 | 0 | 0 | 0 |
| `avgₕ!(u, closure over array)` | 0 | 1680 | 0 | 0 | 0 |
| `mul! kronecker_operator` | 0 | 1952 | 0 | 0 | 0 |
| `explicit rhs(du, u)` | 0 | 1648 | 0 | 0 | 0 |

tt-hooks allocates 0 B on every path, as batch does, though its closures capture forms,
spaces and user kernels whole. Moving `CpuThreaded` onto the team takes 33 of the 42 paths to
0 B, from 1568 to 103600 B per call under threads (`Δₕ!` was 0 B already). On the other 9 the
bytes are `src/`'s own,
found with `Profile.Allocs` (`eval/threadteams_eval.jl allocs`):

- 96 B, the five reductions: the `zeros(T, nchunks)` partial sums of `_threaded_dot`
  (`src/utils/linear_algebra.jl:924`, `src/space/inner_product.jl:676`). `run_fold`, as
  tt-hooks uses it, needs none.
- 64 and 96 B, the broadcasts; 128 B, `εₕ!`: the `Ref(bc)` that `_static_broadcast_ref!`
  (`src/space/vectorelement.jl:522`) builds for Bramble's Enzyme rule. With the team behind
  the runner it escapes into the chunk closure.
- 832 B, `jacobian!`: the argument tuples `_run_bands!`'s `CpuThreaded` method
  (`src/operators/vector_calculus.jl:113`) packs for `_static_or_serial`, whose `try` stays
  out of line (#1243), and the closure over them.

## Time per call

BenchmarkTools, 0.4 s per row, samples back to back. Each cell is the median over three
alternating runs (poly, tt, poly, tt, poly, tt) of each run's minimum, in µs, with the range
of the three minima in brackets. The last two columns are ratios of medians.

| Path | Mesh | serial | threads | batch | tt-threaded | tt-hooks | tt-hooks / batch | tt-threaded / threads |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `D₋ₓ!` | 65² | 0.74 (0.731–0.752) | 6.54 (6.5–6.54) | 0.425 (0.4–0.433) | 0.434 (0.424–0.444) | 0.403 (0.398–0.405) | 0.95 | 0.07 |
| `D₋ₓ!` | 513² | 200 (200–200) | 58.5 (57.9–59.8) | 53.8 (53.4–54) | 53.7 (53.7–54.5) | 53.7 (53.6–54.5) | 1.00 | 0.92 |
| `D₋ᵧ!` | 65² | 0.697 (0.691–0.707) | 6.54 (6.46–6.71) | 0.398 (0.395–0.41) | 0.437 (0.432–0.437) | 0.394 (0.392–0.394) | 0.99 | 0.07 |
| `D₋ᵧ!` | 513² | 43.8 (42.6–44.9) | 39.9 (38.7–40.1) | 22.4 (22.4–25.2) | 23.1 (22.8–23.5) | 27.1 (16.3–27.4) | 1.21 | 0.58 |
| `D₊ₓ!` | 65² | 0.778 (0.753–0.836) | 6.88 (6.5–7.33) | 0.376 (0.364–0.378) | 0.425 (0.419–0.433) | 0.465 (0.448–0.475) | 1.24 | 0.06 |
| `D₊ₓ!` | 513² | 53.9 (53.4–55.3) | 41.5 (38.2–41.8) | 20.8 (18.1–24.5) | 22.6 (22.4–23.7) | 17.2 (17–20.5) | 0.83 | 0.55 |
| `Dcₓ!` | 65² | 1.05 (1.04–1.12) | 7.62 (7.58–9.12) | 0.504 (0.502–0.505) | 0.524 (0.521–0.531) | 0.498 (0.496–0.512) | 0.99 | 0.07 |
| `Dcₓ!` | 513² | 712 (265–718) | 194 (172–199) | 190 (143–190) | 189 (189–192) | 189 (131–192) | 0.99 | 0.98 |
| `D̃ₓ!` | 65² | 1.05 (1.04–1.09) | 7.33 (6.75–7.96) | 0.474 (0.473–0.478) | 0.471 (0.46–0.496) | 0.456 (0.452–0.458) | 0.96 | 0.06 |
| `D̃ₓ!` | 513² | 68.2 (68.2–72.9) | 44.3 (43.6–44.6) | 24.1 (22.7–26.2) | 22.4 (22.1–22.5) | 24.5 (22.6–25.9) | 1.02 | 0.51 |
| `Mₓ!` | 65² | 0.736 (0.682–0.759) | 7.21 (6.79–7.25) | 0.36 (0.355–0.381) | 0.394 (0.391–0.411) | 0.36 (0.352–0.367) | 1.00 | 0.05 |
| `Mₓ!` | 513² | 53.2 (50.9–58.8) | 39.5 (38.9–39.5) | 25.5 (23.7–25.8) | 25.2 (21.7–26.6) | 26.2 (26–26.5) | 1.03 | 0.64 |
| `M₊ₓ!` | 65² | 0.779 (0.755–0.787) | 7.5 (7.12–7.62) | 0.344 (0.34–0.352) | 0.39 (0.373–0.402) | 0.33 (0.327–0.343) | 0.96 | 0.05 |
| `M₊ₓ!` | 513² | 43 (41.3–44.8) | 39.8 (36.7–40) | 22 (20.4–24.8) | 25.6 (25–26.2) | 25.2 (22–25.2) | 1.14 | 0.64 |
| `Mcₓ!` | 65² | 0.836 (0.825–0.863) | 7.12 (6.54–7.79) | 0.392 (0.369–0.408) | 0.425 (0.416–0.429) | 0.385 (0.378–0.389) | 0.98 | 0.06 |
| `Mcₓ!` | 513² | 94.9 (93–95.8) | 50.4 (48.6–52.9) | 34.5 (33.8–37.2) | 35.8 (34.3–37.8) | 37 (35.8–37.8) | 1.07 | 0.71 |
| `∇cₕ!` | 65² | 1.81 (1.8–1.85) | 37.5 (33.1–40.5) | 0.992 (0.918–2.5) | 0.977 (0.976–0.988) | 0.941 (0.939–0.944) | 0.95 | 0.03 |
| `∇cₕ!` | 513² | 755 (755–756) | 99.7 (95–101) | 221 (221–222) | 100 (75.6–100) | 219 (214–221) | 0.99 | 1.00 |
| `Δₕ!` | 65² | 4.17 (4.16–17.8) | 4.16 (4.16–4.17) | 4.17 (4.17–4.18) | 4.16 (4.16–4.16) | 4.16 (4.16–4.18) | 1.00 | 1.00 |
| `Δₕ!` | 513² | 758 (287–1146) | 1151 (1150–1152) | 1146 (1146–1146) | 708 (707–708) | 478 (478–481) | 0.42 | 0.61 |
| `divₕ!` | 65² | 2.17 (2.17–2.19) | 42.1 (35.2–48.2) | 1.37 (1.37–1.39) | 2.25 (2.24–2.25) | 2.19 (2.19–2.2) | 1.60 | 0.05 |
| `divₕ!` | 513² | 302 (302–303) | 146 (133–157) | 106 (105–106) | 89.5 (88.6–102) | 262 (262–262) | 2.48 | 0.61 |
| `divcₕ!` | 65² | 2.56 (2.52–2.56) | 41 (40.9–43.1) | 2.2 (2.19–2.24) | 2.37 (2.37–2.37) | 2.33 (2.32–2.34) | 1.06 | 0.06 |
| `divcₕ!` | 513² | 310 (190–519) | 134 (134–319) | 125 (73.2–127) | 82.7 (80.3–83.6) | 171 (123–172) | 1.37 | 0.62 |
| `diṽₕ!` | 65² | 2.52 (2.49–2.53) | 45.3 (40.6–50.6) | 2.21 (2.2–2.23) | 2.44 (2.43–2.44) | 2.43 (2.4–2.44) | 1.10 | 0.05 |
| `diṽₕ!` | 513² | 201 (190–217) | 131 (130–134) | 69.7 (69.4–75.8) | 82 (82–82.7) | 72.1 (70.4–72.7) | 1.04 | 0.63 |
| `curlₕ!` | 65² | 2.12 (2.12–2.14) | 46 (26.5–51.2) | 1.49 (1.48–1.49) | 2.34 (2.34–2.35) | 2.3 (2.3–2.31) | 1.55 | 0.05 |
| `curlₕ!` | 513² | 186 (179–192) | 121 (115–132) | 62 (61.8–62) | 67.3 (66.9–67.7) | 69.2 (68.9–69.2) | 1.12 | 0.55 |
| `curlcₕ!` | 65² | 2.51 (2.51–2.55) | 40.7 (40.3–43.3) | 2.35 (2.34–2.36) | 2.54 (2.53–2.54) | 2.5 (2.49–2.51) | 1.07 | 0.06 |
| `curlcₕ!` | 513² | 190 (188–193) | 125 (124–126) | 74 (73.4–74.1) | 70.7 (69.9–71.2) | 70.5 (70.2–70.6) | 0.95 | 0.57 |
| `εₕ!` | 65² | 11.8 (11.6–11.9) | 256 (227–311) | 5.9 (5.85–5.92) | 7.12 (7.02–7.47) | 6.26 (6.26–6.28) | 1.06 | 0.03 |
| `εₕ!` | 513² | 735 (714–789) | 537 (526–544) | 280 (274–281) | 264 (263–293) | 270 (265–274) | 0.96 | 0.49 |
| `v .= 2u + u*u` | 65² | 0.662 (0.661–0.664) | 7.17 (6.96–7.79) | 0.424 (0.422–0.428) | 0.47 (0.466–0.475) | 0.431 (0.423–0.435) | 1.02 | 0.07 |
| `v .= 2u + u*u` | 513² | 48.9 (48.8–49.2) | 40.9 (40.9–41.6) | 23.7 (20.5–25.3) | 25.6 (20.5–27) | 25 (24.2–25) | 1.05 | 0.63 |
| `v .= u + w` | 65² | 0.49 (0.48–0.504) | 6.54 (6.38–7.12) | 0.38 (0.374–0.383) | 0.436 (0.43–0.445) | 0.385 (0.377–0.389) | 1.01 | 0.07 |
| `v .= u + w` | 513² | 55.9 (47.6–61.2) | 40.5 (39.8–43.1) | 19.8 (18.5–23.3) | 23.2 (21.2–23.6) | 22.8 (22.6–23.3) | 1.15 | 0.57 |
| `assemble! innerₕ(f, v)` | 65² | 2.51 (2.51–2.53) | 12.4 (8.25–17.8) | 2.46 (2.44–2.49) | 2.29 (2.18–2.29) | 2.28 (2.25–2.28) | 0.93 | 0.18 |
| `assemble! innerₕ(f, v)` | 513² | 166 (156–175) | 78.5 (78.2–79) | 61.3 (58.9–61.7) | 61.1 (61–62.3) | 60.8 (60.3–61.2) | 0.99 | 0.78 |
| `assemble! innerₕ(f, v; markers)` | 65² | 3.43 (3.39–3.47) | 19.8 (18.7–20.7) | 1.42 (1.41–1.43) | 1.34 (1.34–1.36) | 1.32 (1.32–1.33) | 0.93 | 0.07 |
| `assemble! innerₕ(f, v; markers)` | 513² | 175 (175–176) | 76.2 (74.3–76.4) | 57 (57–57.2) | 57 (57–57) | 57.1 (57.1–57.2) | 1.00 | 0.75 |
| `assemble! innerₕ(f, restrict_to(v))` | 65² | 3.49 (3.47–3.6) | 21 (18.8–21) | 1.38 (1.38–1.42) | 1.36 (1.35–1.36) | 1.36 (1.36–1.42) | 0.98 | 0.06 |
| `assemble! innerₕ(f, restrict_to(v))` | 513² | 176 (176–176) | 76.7 (75.9–76.9) | 57 (56.9–57.1) | 57.2 (57.1–57.2) | 57.4 (57.3–57.5) | 1.01 | 0.75 |
| `assemble! bilinear` | 65² | 27 (26.9–27.2) | 196 (176–203) | 11.5 (11.5–11.7) | 10.7 (10.7–10.8) | 10.5 (10.5–10.5) | 0.91 | 0.05 |
| `assemble! bilinear` | 513² | 1616 (1604–1627) | 782 (763–783) | 660 (659–666) | 650 (649–689) | 636 (631–648) | 0.96 | 0.83 |
| `assemble! bilinear, masked term` | 65² | 30.5 (30.4–30.6) | 266 (262–296) | 12.8 (12.7–12.9) | 12.3 (12.2–12.3) | 12 (12–12.1) | 0.94 | 0.05 |
| `assemble! bilinear, masked term` | 513² | 1777 (1772–1795) | 853 (852–865) | 721 (713–721) | 709 (707–713) | 710 (703–715) | 0.98 | 0.83 |
| `assemble! bilinear, dirichlet` | 65² | 43.4 (43.4–43.8) | 185 (183–192) | 29.9 (29.9–30) | 28.9 (28.8–29) | 28.7 (28.7–28.8) | 0.96 | 0.16 |
| `assemble! bilinear, dirichlet` | 513² | 2651 (2629–2656) | 1843 (1828–1867) | 1786 (1784–1815) | 1729 (1720–1738) | 1730 (1726–1746) | 0.97 | 0.94 |
| `jacobian!` | 65² | 52.4 (52.4–52.8) | 278 (267–284) | 33.2 (33–33.2) | 33.4 (33.3–33.6) | 32.2 (32.2–32.7) | 0.97 | 0.12 |
| `jacobian!` | 513² | 3298 (3282–3322) | 2380 (2380–2388) | 2094 (2076–2107) | 2063 (2059–2064) | 2067 (2041–2088) | 0.99 | 0.87 |
| `mul! plain` | 65² | 5.99 (5.98–6.05) | 3.75 (3.62–8.79) | 3.15 (3.12–3.17) | 3.12 (3.11–3.13) | 3.11 (3.07–3.11) | 0.99 | 0.83 |
| `mul! plain` | 513² | 521 (348–1012) | 122 (121–123) | 214 (212–295) | 119 (118–215) | 118 (117–119) | 0.55 | 0.97 |
| `mul! masked term` | 65² | 9.27 (9.21–9.54) | 4.46 (4.46–9.79) | 4.2 (4.14–4.22) | 4.16 (4.15–4.17) | 4.16 (4.14–4.16) | 0.99 | 0.93 |
| `mul! masked term` | 513² | 529 (511–536) | 171 (171–172) | 166 (166–167) | 171 (169–171) | 166 (166–169) | 1.00 | 1.00 |
| `v_cycle!` | 65² | 263 (262–266) | 683 (663–768) | 173 (172–175) | 157 (156–157) | 156 (155–157) | 0.90 | 0.23 |
| `v_cycle!` | 513² | 11177 (10848–12597) | 7648 (6795–7750) | 6970 (6943–9782) | 6246 (5769–6636) | 5248 (5183–6086) | 0.75 | 0.82 |
| `ldiv! multigrid` | 65² | 264 (263–265) | 663 (599–669) | 172 (169–173) | 158 (157–164) | 158 (156–158) | 0.92 | 0.24 |
| `ldiv! multigrid` | 513² | 11060 (10884–13445) | 7511 (7097–7766) | 6806 (6577–8140) | 5664 (5223–5788) | 5290 (5232–5309) | 0.78 | 0.75 |
| `assemble! bilinear, n²/3 × 3` | 65² | 45.3 (45.2–45.6) | 180 (179–195) | 59.2 (59.2–59.2) | 27.6 (27.2–27.8) | 27.5 (27.3–27.7) | 0.46 | 0.15 |
| `assemble! bilinear, n²/3 × 3` | 513² | 2847 (2832–2874) | 1541 (1534–1548) | 3692 (3686–3699) | 1631 (1630–1651) | 1640 (1630–1688) | 0.44 | 1.06 |
| `assemble! innerₕ(f, D₋ᵧ(v)), n²/3 × 3` | 65² | 6.78 (6.78–6.8) | 50.2 (47.2–53.3) | 40.5 (40.3–44.4) | 7.8 (7.61–7.83) | 7.8 (7.79–7.81) | 0.19 | 0.16 |
| `assemble! innerₕ(f, D₋ᵧ(v)), n²/3 × 3` | 513² | 418 (417–420) | 257 (257–287) | 2463 (2456–2718) | 444 (439–445) | 445 (443–445) | 0.18 | 1.73 |
| `Rₕ!(composite, closure)` | 65² | 16.5 (16.4–16.5) | 23.4 (21.4–24.2) | 4.68 (4.57–4.72) | 7.73 (7.67–7.98) | 5.35 (5.22–5.54) | 1.14 | 0.33 |
| `Rₕ!(composite, closure)` | 513² | 948 (944–949) | 318 (315–325) | 255 (254–256) | 465 (458–465) | 307 (306–318) | 1.21 | 1.46 |
| `_sweep_for!(CartesianIndices)` | 65² | 25.5 (25.4–25.6) | 27.9 (27.9–28) | 6.83 (6.82–6.83) | 6.81 (6.78–6.81) | 7.06 (6.79–7.06) | 1.03 | 0.24 |
| `_sweep_for!(CartesianIndices)` | 513² | 2335 (2333–2346) | 652 (651–655) | 640 (639–641) | 649 (645–650) | 650 (648–652) | 1.02 | 1.00 |
| `_dot, vector weights` | 65² | 0.7 (0.698–0.706) | 7.33 (7.25–8.54) | 0.423 (0.421–0.435) | 0.459 (0.458–0.462) | 0.44 (0.438–0.442) | 1.04 | 0.06 |
| `_dot, vector weights` | 513² | 76.3 (72.2–87.4) | 42.7 (38.3–44.9) | 24.3 (24.2–26.1) | 27.8 (26.5–27.8) | 26.2 (24.8–28.8) | 1.08 | 0.65 |
| `_dot_masked, vector weights` | 65² | 0.275 (0.275–0.276) | 6.46 (6.38–6.5) | 0.287 (0.275–0.288) | 0.299 (0.292–0.3) | 0.259 (0.259–0.264) | 0.90 | 0.05 |
| `_dot_masked, vector weights` | 513² | 3.77 (3.72–3.92) | 18.3 (17.2–20) | 1.5 (1.49–1.5) | 1.64 (1.63–1.65) | 1.62 (1.6–1.62) | 1.08 | 0.09 |
| `innerₕ(u, w)` | 65² | 0.716 (0.715–0.717) | 8.29 (6.88–16.8) | 0.512 (0.51–0.513) | 0.564 (0.56–0.585) | 0.522 (0.519–0.526) | 1.02 | 0.07 |
| `innerₕ(u, w)` | 513² | 63 (61.8–70.7) | 42.5 (41.6–43) | 21.8 (21.6–22) | 23 (21.7–23.1) | 22.4 (22.2–24.3) | 1.03 | 0.54 |
| `innerₕ(u, w; markers)` | 65² | 0.288 (0.287–0.288) | 6.58 (6.46–6.75) | 0.261 (0.259–0.263) | 0.309 (0.309–0.31) | 0.28 (0.276–0.282) | 1.07 | 0.05 |
| `innerₕ(u, w; markers)` | 513² | 3.62 (3.6–3.64) | 17.5 (17.3–17.6) | 1.38 (1.37–1.39) | 1.54 (1.53–1.55) | 1.52 (1.5–1.52) | 1.10 | 0.09 |
| `normₕ(u)` | 65² | 0.479 (0.478–0.479) | 8.58 (6.5–8.96) | 0.51 (0.498–0.51) | 0.552 (0.548–0.553) | 0.518 (0.517–0.522) | 1.02 | 0.06 |
| `normₕ(u)` | 513² | 36.9 (36.9–37) | 39.7 (38.8–40.1) | 20.6 (20.4–20.7) | 19.9 (19.8–20.5) | 20.5 (19.8–20.6) | 0.99 | 0.50 |
| `Rₕ!(u, closure over array)` | 65² | 4.99 (4.99–4.99) | 19.2 (17.8–19.8) | 1.61 (1.6–1.69) | 3.21 (3.2–3.26) | 1.69 (1.68–1.73) | 1.05 | 0.17 |
| `Rₕ!(u, closure over array)` | 513² | 311 (310–311) | 108 (108–109) | 89.9 (86.7–90) | 192 (192–193) | 97.1 (97.1–97.1) | 1.08 | 1.78 |
| `avgₕ!(u, closure over array)` | 65² | 127 (127–127) | 48.5 (48–49.1) | 28.7 (28.7–28.7) | 31.6 (31.6–31.8) | 28.7 (28.7–28.7) | 1.00 | 0.65 |
| `avgₕ!(u, closure over array)` | 513² | 7883 (7883–7895) | 1774 (1774–1776) | 1749 (1749–1750) | 1955 (1955–1969) | 1754 (1754–1754) | 1.00 | 1.10 |
| `mul! kronecker_operator` | 65² | 5.5 (5.48–5.53) | 17.9 (17.8–18.1) | 1.88 (1.85–7.03) | 2.02 (2.01–2.02) | 2.03 (2.03–2.03) | 1.08 | 0.11 |
| `mul! kronecker_operator` | 513² | 347 (344–515) | 122 (122–124) | 102 (102–102) | 195 (124–195) | 148 (104–149) | 1.46 | 1.60 |
| `explicit rhs(du, u)` | 65² | 18.5 (18.5–32.3) | 37.2 (29.2–37.9) | 8.15 (8.07–8.24) | 19.5 (19.3–19.7) | 7.94 (7.92–8.01) | 0.97 | 0.52 |
| `explicit rhs(du, u)` | 513² | 1506 (1224–1807) | 1346 (1330–1774) | 771 (764–801) | 1452 (1168–1749) | 740 (738–748) | 0.96 | 1.08 |

On the 51 rows under 50 µs under batch (`Δₕ!` excluded: it has no threaded path, and its
513 × 513 row moves with code placement, 287 to 1146 µs serially), tt-hooks / batch has
median 1.02. Five rows are above 1.15: `divₕ!` and `curlₕ!` at 65 × 65 (1.60, 1.55; 2.2
against 1.4 µs, the same under tt-threaded), `D₊ₓ!` at 65 × 65 (1.24), `D₋ᵧ!` and
`v .= u + w` at 513 × 513 (1.21, 1.15). tt-threaded / threads has median 0.07 there: the
team's region costs what `@batch`'s does, and `Threads.@threads` costs 6 to 40 µs more. On
the 31 longer rows the medians are 0.99 (tt-hooks / batch) and 0.83 (tt-threaded /
threads).

Where tt-hooks is slower than batch on long rows: `divₕ!` at 513 × 513 (2.48 and 1.37 for
`divcₕ!`; tt-threaded is not, about 0.6 of threads), `mul! kronecker_operator` at
513 × 513 (1.45, its range 104 to 149 µs), and the two `Rₕ!` closure rows at 513 × 513 (1.08, 1.21).
None is explained. tt-threaded is above batch on `Rₕ!` with a closure (2.1), the Kronecker
product (1.9) and the explicit right-hand side (1.9). On the first two it is above threads as
well (192 against 108 µs, 195 against 122 µs), also unexplained. The right-hand side is slow
under threads too (1346 µs): `_rhs_csr` has a `CpuPolyester` method only
(`src/problems/semidiscrete_rhs.jl:181`), so `CpuThreaded` takes another route.

Where the team is well ahead: the thin meshes, where the sweeps colour points one at a time
(`@batch` 59 µs and 3.7 ms for the bilinear row, the team 27.5 µs and 1.6 ms; 40 µs and
2.5 ms against 7.8 µs and 0.44 ms for the linear row), and the multigrid cycle at
513 × 513 (5.2 against 7.0 ms).

## With idle time between calls

`eval/threadteams_eval.jl wake`: each call follows 1 ms of `Libc.systemsleep`, 200 calls,
median µs per call, then CPU time over wall time across the run (4 would be every thread busy
throughout). The last column sets ThreadTeams' spin time to 2 ms
(`ThreadTeams.set_spin_time!(2e-3)`; the default is 100 µs).

| Path | Mesh | serial | threads | batch | tt-threaded | tt-hooks | tt-hooks, spin 2 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `D₋ₓ!` | 65² | 5.79 / 0.01 | 125 / 0.33 | 1.75 / 2.99 | 17.7 / 0.30 | 6.58 / 0.29 | 0.75 / 2.97 |
| `D₋ₓ!` | 513² | 325 / 0.19 | 184 / 0.55 | 54.6 / 2.97 | 173 / 0.73 | 178 / 0.77 | 55 / 3.01 |
| `Dcₓ!` | 65² | 1.88 / 0.00 | 129 / 0.33 | 0.75 / 2.95 | 20.5 / 0.29 | 8.88 / 0.29 | 0.875 / 2.96 |
| `Dcₓ!` | 513² | 576 / 0.27 | 226 / 0.69 | 192 / 3.07 | 208 / 0.86 | 244 / 0.88 | 193 / 3.08 |
| `divₕ!` | 65² | 13.7 / 0.02 | 222 / 0.54 | 10.9 / 2.94 | 31 / 0.45 | 29.8 / 0.44 | 11.5 / 2.94 |
| `divₕ!` | 513² | 583 / 0.32 | 334 / 0.77 | 87.5 / 3.04 | 411 / 0.94 | 347 / 1.05 | 73.8 / 3.01 |
| `v .= u + w` | 65² | 5.6 / 0.01 | 108 / 0.33 | 1.35 / 2.99 | 18.5 / 0.29 | 18.2 / 0.29 | 0.75 / 2.95 |
| `v .= u + w` | 513² | 407 / 0.20 | 194 / 0.59 | 36 / 2.95 | 255 / 0.83 | 90.8 / 0.67 | 34.4 / 3.00 |
| `innerₕ(u, w)` | 65² | 6.38 / 0.01 | 126 / 0.33 | 0.896 / 2.97 | 20.3 / 0.30 | 20 / 0.30 | 2.69 / 2.96 |
| `innerₕ(u, w)` | 513² | 245 / 0.17 | 308 / 0.60 | 27.6 / 3.02 | 107 / 0.66 | 177 / 0.76 | 28.1 / 3.01 |
| `assemble! bilinear` | 65² | 82.6 / 0.06 | 528 / 1.01 | 11.8 / 2.98 | 29.6 / 0.54 | 69.1 / 0.56 | 11 / 3.00 |
| `assemble! bilinear` | 513² | 2361 / 0.61 | 1231 / 1.85 | 751 / 3.34 | 991 / 1.77 | 968 / 1.75 | 711 / 3.34 |
| `jacobian!` | 65² | 151 / 0.12 | 600 / 1.19 | 34.4 / 2.98 | 100 / 0.69 | 97.6 / 0.68 | 34.9 / 3.01 |
| `jacobian!` | 513² | 3410 / 0.70 | 2773 / 1.88 | 2282 / 3.61 | 2545 / 1.69 | 2605 / 1.68 | 2295 / 3.14 |
| `mul! plain` | 65² | 31.4 / 0.03 | 55.8 / 0.27 | 3.46 / 2.99 | 11.1 / 0.29 | 45.4 / 0.35 | 3.33 / 2.98 |
| `mul! plain` | 513² | 1343 / 0.48 | 427 / 0.80 | 138 / 3.06 | 229 / 0.79 | 512 / 1.35 | 138 / 3.03 |
| `v_cycle!` | 65² | 291 / 1.10 | 1318 / 1.97 | 173 / 3.07 | 352 / 1.10 | 342 / 1.12 | 159 / 3.06 |
| `v_cycle!` | 513² | 11559 / 1.12 | 8286 / 3.30 | 8733 / 4.11 | 5379 / 3.45 | 5234 / 3.47 | 5104 / 4.15 |
| `explicit rhs(du, u)` | 65² | 27.2 / 0.03 | 147 / 0.33 | 8.67 / 2.97 | 74.7 / 0.32 | 21 / 0.54 | 8.35 / 2.97 |
| `explicit rhs(du, u)` | 513² | 2111 / 0.58 | 1843 / 0.75 | 872 / 3.38 | 1409 / 0.81 | 1403 / 1.99 | 1077 / 3.39 |

After a gap longer than its 100 µs spin the team is asleep, and short calls slow down: the
65 × 65 stencil, broadcast and reduction rows take 6.6 to 31 µs where batch takes 0.75 to
11 µs, still a fifth or less of `Threads.@threads`. Polyester does not sleep through a 1 ms
gap: its CPU time is 2.9 to 4.1 times the wall time on every row of the table, three cores spinning for a
caller that runs on one. With a 2 ms spin the team is within 1.17 of batch on 18 of the 20
rows (`innerₕ(u, w)` at 65 × 65 is the exception at 2.7 against 0.9 µs, and the explicit
right-hand side at 513 × 513 at 1077 against 872 µs) and costs the same CPU. In the second
after the last call, batch used 45 ms of CPU time (median over all rows, at most 60 ms), the
team 0.85 ms (at most 7.7 ms), `Threads.@threads` 0.13 ms. Part of the slowdown is the
machine: serial calls after the gap take up to twice their back-to-back time too (`D₋ₓ!`
at 513 × 513, 325 against 200 µs), so idle cores clock down.

## Gradients

`eval/enzyme_tt_driver.jl` runs each case of `test/ext/enzyme_policy_cases.jl` (`CASES`
and `CAPTURES`, 29 cases) in a process of its own, under `CpuPolyester` on the team, with
Polyester not loaded: each child asserts `BrambleEnzymePolyesterExt` is absent, so the only
rules between Enzyme and the team are ThreadTeams' own. Modes: Enzyme reverse called
directly, reverse through DifferentiationInterface, and forward through DI. Each is compared
with ForwardDiff under the same policy (tolerance 1e-10). That oracle is itself within
2.8e-16 of ForwardDiff under `CpuSerial` on every case. Cells give the relative error.

| Case | Enzyme reverse | DI reverse | DI forward |
|---|---|---|---|
| `_batch_dot` | ok (3e-16) | ok (3e-16) | batched reduction |
| `_batch_dot_masked` | ok (4e-16) | ok (4e-16) | batched reduction |
| `_batch_for!` | ok (1e-15) | ok (6e-16) | ok (1e-16) |
| `_batch_axis_for!` | ok (1e-15) | ok (2e-15) | ok (4e-16) |
| `_batch_scatter_for!` | ok (1e-15) | ok (1e-15) | ok (0e+00) |
| `_batch_bilinear_band_replay!` | ok (1e-16) | ok (1e-16) | ok (9e-16) |
| `_batch_bilinear_colour_replay!` | ok (2e-16) | ok (2e-16) | ok (2e-16) |
| `_batch_linear_colour_sweep!` | ok (4e-16) | ok (4e-16) | ok (2e-16) |
| `_batch_linear_band_sweep!` | ok (2e-16) | ok (2e-16) | ok (6e-16) |
| `_batch_difference_engine!` | ok (3e-16) | ok (3e-16) | ok (2e-16) |
| `_batch_average_engine!` | ok (3e-16) | ok (3e-16) | ok (3e-16) |
| `_batch_centered_average_engine!` | ok (6e-17) | ok (6e-17) | ok (2e-16) |
| `_batch_run_bands!` | ok (2e-16) | ok (2e-16) | ok (3e-16) |
| `_batch_mf_bands!` | calling convention | calling convention | calling convention |
| `_batch_broadcast!` | calling convention | calling convention | calling convention |
| `_batch_kron_lines!` | calling convention | calling convention | calling convention |
| `_batch_csr_spmv!` | calling convention | calling convention | calling convention |
| `_batch_for_const_capture` | calling convention | calling convention | calling convention |
| `_batch_for_mixed_capture` | ok (3e-16) | ok (3e-16) | ok (1e-16) |
| `_batch_for_reads_v` | ok (5e-17) | ok (5e-17) | ok (2e-16) |
| `_batch_mf_bands_coef` | calling convention | calling convention | calling convention |
| `_batch_mf_unit_band` | calling convention | calling convention | calling convention |
| `_batch_mf_unit_colour` | calling convention | calling convention | calling convention |
| `_batch_mf_scalar_param` | crash | crash | crash |
| `_batch_mf_overwrite_x` | crash | crash | crash |
| `_batch_mf_unit_overwrite_x` | ok (2e-16) | ok (2e-16) | ok (2e-16) |
| `_batch_kron_scaled` | ok (5e-16) | ok (5e-16) | ok (2e-16) |
| `_batch_csr_values` | calling convention | calling convention | calling convention |
| `_batch_mf_overwrite_coef` | crash | crash | crash |

- 15 of 29 cases pass all three modes, and the two reductions pass both reverse modes: 13
  of the 17 hooks differentiate without one of Bramble's hand-written rules.
- "batched reduction": DI's forward gradient pushes several tangents at once, and
  ThreadTeams' `run_fold` rule throws "batched differentiation of reductions is not
  supported".
- "calling convention": Enzyme's `CallingConventionMismatchError` on the chunk closure
  passed to `run_chunks` by value, when it mixes arrays with plain numbers. This is the shape
  for which `src/` passes Bramble's kernels behind a `Ref` to Bramble's own rules. That the same
  `Ref` would fix it here is a guess, not tested.
- "crash": the child process died of a segmentation fault inside LLVM's dead-code elimination
  (`AggressiveDeadCodeElimination::markLive`) while Enzyme compiled the case.

Under `CpuThreaded` on the team (`S52_AD_POLICY=CpuThreaded`, with `threaded_on_teams.jl`
included and Bramble's `BrambleEnzymeExt` rules in force), the 9 `THREADED` cases pass all
three modes, relative error at most 9.1e-16. Those rules wrap the runners whose bodies the
team now runs, so this shows that keeping them keeps `CpuThreaded` differentiable on the
team. It says nothing about ThreadTeams' own rules.

## Load, invalidations, Metal

`eval/load_eval.jl`, a fresh process per figure, `using Bramble` first.

| Loaded after Bramble | Load s (min of 5) | First `D₋ₓ!` s | Invalidations (trees) |
|---|---:|---:|---:|
| nothing | none | 0.005 | none |
| Polyester (with `BramblePolyesterExt`) | 0.185 | 0.159 | 3836 (63) |
| ThreadTeams | 0.011 | 0.005 | 0 |
| the prototype | 0.015 | 0.362 | 0 |
| the prototype and `threaded_on_teams.jl` | 0.032 | 0.393 | 474 (16) |

The first `D₋ₓ!` runs under the loaded policy (`CpuSerial` for the first and third rows).
The prototype has no precompile workload, which `BramblePolyesterExt` has, so its first call
compiles. The 474 invalidations of the last row are the overwritten methods themselves, which
a change to `src/` would not have. The largest Polyester trees are `ManualMemory`,
`Static` and `Polyester.zip` bindings.

Metal's first operation, three fresh processes each, the package loaded before Metal:

| Loaded before Metal | `MtlArray` broadcast, s | then Bramble `D₋ₓ!` on a Metal mesh, s |
|---|---|---|
| nothing | 0.63, 0.64, 0.63 | 1.73, 1.71, 1.72 |
| Polyester | 5.28, 5.38, 5.28 | 1.68, 1.65, 1.67 |
| ThreadTeams | 0.63, 0.63, 0.63 | 1.70, 1.69, 1.68 |
| the prototype | 0.63, 0.63, 0.63 | 1.71, 1.68, 1.80 |

ThreadTeams costs Metal nothing; Metal 1.12.1 loads it anyway, through KernelAbstractions
0.10.8. Polyester adds 4.7 s to the first launch.

## Not measured

- 3D meshes, ghost spaces, composite bilinear forms, and the matrix-free ghost product (the
  `_ghost_mf_bands!` overwrite runs, but no row times it).
- Two or eight threads, and an x86 machine.
- The `BATCH_MODES` and `NESTED_MODES` of the Enzyme cases, and the `THREADED_CASES` and
  `ASSEMBLY_REVERSE_CASES` lists.
- Whether a `Ref` around the chunk closure clears the calling-convention failures.
- The cause of the tt-hooks slowdowns listed under the time table.
- An exception thrown inside a chunk: ThreadTeams documents that it rethrows the first
  failing chunk's exception, but no row tests that Bramble's messages come through unchanged.

## Recommendation

Recommendation: back `CpuThreaded` with ThreadTeams, as a dependency of Bramble, keeping Bramble's `CpuThreaded` Enzyme rules; leave `CpuPolyester` on Polyester until ThreadTeams' rules carry the hooks that fail here, then move it onto the same team and retire Polyester.

The figures behind it: on the team, `CpuThreaded` takes 0.07 of its present time on the 51
short rows (median) and 0.83 on the long ones, allocates 0 B on 33 of 42 paths with the
other 9 traced to `src/`, returns the present results bitwise, and keeps its gradients with
Bramble's rules in place. ThreadTeams loads in 11 ms with no invalidations, leaves Metal's
first launch alone, and comes with KernelAbstractions 0.10.8 anyway. Replacing Polyester
outright fails on gradients today: 4 of the 17 hooks, and the forward mode of 2 more, do not
differentiate through ThreadTeams' rules, and Enzyme under `CpuPolyester` has to work. A third
policy would leave three bodies where there are now two. Making ThreadTeams a hard dependency,
rather than an extension under which `CpuThreaded` would fall back to `Threads.@threads`, is
what lets one body serve the policy. That changes `Project.toml`, so it is the user's call.

Outline of the follow-up plan:

1. `CpuThreaded` on the team. OWNS `Project.toml` (ThreadTeams 2 in `[deps]` and
   `[compat]`), `src/utils/linear_algebra.jl`, `src/assembly/bilinear_execution.jl`,
   `src/assembly/matrix_free.jl`, `src/assembly/ghost_matrix_free.jl`. The seven
   `Threads.@threads` runners and the two `@spawn` sites call one `run_chunks` helper, as
   `eval/threaded_on_teams.jl` does. CHECK: no `Threads.@threads` or `Threads.@spawn` left in
   `src/` outside comments, EXPECT 0; the `CpuThreaded` test files and
   `test/ext/enzyme_threaded_ext.jl` green. WHY: one body for the policy; rejected keeping
   `@threads` beside the team behind an extension, which keeps both bodies.
2. `CpuThreaded` at 0 B. OWNS the three `src/` sites the bytes section names. `_threaded_dot`
   and its masked and separable forms fold with `run_fold` instead of a partials vector, and
   the `Ref(bc)` and the argument tuple stop escaping. CHECK `benchmark/cputhreaded_bytes.jl`,
   EXPECT every row 0. This changes S5.3: its docstring would state 0 B rather than a bound.
   WHY: the bytes are `src/`'s, not the team's; rejected documenting 96 to 832 B as the bound.
3. Spin time. Either leave ThreadTeams' 100 µs default (the team sleeps through a gap, and a
   short call then takes 6 to 30 µs) or document `ThreadTeams.set_spin_time!` for callers with
   short gaps (it matches `@batch`, at `@batch`'s CPU cost). A paragraph in the `CpuThreaded`
   docstring, no code. WHY: the trade-off is measured and the choice is the user's; rejected a
   longer default, which spends three cores for every caller.
4. Prune `ext/BrambleEnzymeExt/threaded.jl`. Delete its rules one runner at a time and rerun
   the `THREADED` Enzyme cases with only ThreadTeams' rules. CHECK the cases, EXPECT all
   modes ok; keep each rule whose deletion fails a case. WHY: pruning by measurement; rejected
   deleting the 1022 lines at once, since some closure shapes fail on `CpuPolyester`'s hooks.
5. Later, in a plan of its own: `CpuPolyester` on the same team. It waits on the
   calling-convention failures (a `Ref` around the chunk closure is the untested guess),
   ThreadTeams' rule for batched `run_fold` (an upstream issue, filed only on the user's yes),
   and the three Enzyme compile crashes. Then `CpuPolyester` runs the `CpuThreaded` bodies,
   and `ext/BramblePolyesterExt.jl` (1411 lines), `src/utils/batch_split.jl` (293) and
   `ext/BrambleEnzymePolyesterExt/` (2956) can go.
