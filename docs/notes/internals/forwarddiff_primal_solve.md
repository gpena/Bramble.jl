# Primal-once solve against the Dual-factor solve (gpena/Bramble.jl#1123)

`BrambleForwardDiffExt` solves a `pde_solve` system whose matrix or load holds
`ForwardDiff.Dual`s. It factors the `Float64` values once. Every partial is then a solve
against that factor. Before it, a `Dual` matrix went to Sparspak, which factors the `Dual`
matrix itself. This page records what that costs and saves on a non-uniform 2D mesh, at
sizes up to 10⁶ dofs. Every figure comes from one `--full` run of
`benchmark/forwarddiff_primal_solve.jl`, with the script's own output table.

## What is measured

The system is `-∇·(κ ∇u) = g` on the unit square with `u = 0` on the boundary, on an `m × m`
node mesh placed at random (`mesh(Ω, (m, m), (false, false))`, seeded), so the spacings are
non-uniform. `κ` is a `ForwardDiff.Dual` with `B` partials, `g = exp(xy)`. A system has
`m²` dofs, the Dirichlet rows included.

Three paths solve the same system, each in its own fresh julia process:

| path | call | what it is |
|---|---|---|
| `new` | `pde_solve(A, F)` | the extension: the `Float64` primal factored once by `pde_solve`'s default route |
| `new_sparspak` | `pde_solve(A, F; solver = :sparspak)` | the extension with Sparspak as the primal solver, the old path's solver |
| `old` | `sparspak_factorize(A) \ F` | the `Dual` matrix factored by Sparspak (the oracle the extension's tests use) |

`new_sparspak` is not in the plan's rows. It is a control: it keeps the solver of the old path
and so separates the effect of factoring the primal once from the effect of the solver.

Each case runs one warm-up on a 16 × 16 system, checked entrywise against the old path, then
one timed solve of the full system. The timed figure includes factorization and the solves,
and excludes assembly. Peak memory is the process footprint from `top`'s MEM column, the
figure `julia_slot.sh` watches. The driver samples it once a second for the whole process.
Each case also reads its own footprint just before the solve (the `before` column of the log),
so the idle baseline is visible. Three back-to-back runs per row, each a fresh process.
Spread is `(max − min) / median` over the three.

Machine: Apple M2, 4 performance and 4 efficiency cores, 16 GB, AC power. Julia 1.13.3
(aarch64, Darwin). 4 threads (`--threads=4`), default `-O`, alone on the machine, after
`check_power_load.sh` and `gate_preflight.sh --timing` (1-minute load 1.96, AC power, no other
Julia process). Bramble at commit `1036e29d`, ForwardDiff 1.4.7, Sparspak 0.3.15,
AppleAccelerate 0.7.0 loaded. The matrices are nonsymmetric at every size, so `pde_solve`'s
default takes `factorize` (UMFPACK LU) and not Accelerate.

## Results

Time in seconds for one solve (median of 3), spread across the 3, and peak footprint in MB as
median (max) of the three runs.

| dofs | partials | path | time median (s) | spread | peak MB, median (max) |
|---:|---:|---|---:|---:|---:|
| 4096 | 1 | new | 0.01 | 6.9 % | 625 (626) |
| 4096 | 1 | new_sparspak | 0.01 | 9.2 % | 625 (630) |
| 4096 | 1 | old | 0.00 | 3.6 % | 628 (629) |
| 4096 | 4 | new | 0.01 | 2.7 % | 643 (644) |
| 4096 | 4 | new_sparspak | 0.01 | 6.6 % | 634 (635) |
| 4096 | 4 | old | 0.01 | 4.1 % | 639 (641) |
| 16384 | 1 | new | 0.02 | 1.0 % | 629 (630) |
| 16384 | 1 | new_sparspak | 0.02 | 5.4 % | 628 (629) |
| 16384 | 1 | old | 0.03 | 5.2 % | 624 (629) |
| 16384 | 4 | new | 0.03 | 4.6 % | 635 (638) |
| 16384 | 4 | new_sparspak | 0.03 | 4.8 % | 636 (644) |
| 16384 | 4 | old | 0.05 | 25.4 % | 638 (642) |
| 65536 | 1 | new | 0.13 | 3.3 % | 662 (665) |
| 65536 | 1 | new_sparspak | 0.10 | 3.6 % | 659 (661) |
| 65536 | 1 | old | 0.19 | 4.2 % | 665 (667) |
| 65536 | 4 | new | 0.14 | 1.9 % | 638 (646) |
| 65536 | 4 | new_sparspak | 0.13 | 32.4 % | 635 (637) |
| 65536 | 4 | old | 0.38 | 1.7 % | 861 (864) |
| 262144 | 1 | new | 0.72 | 2.5 % | 1218 (1219) |
| 262144 | 1 | new_sparspak | 0.58 | 4.9 % | 806 (1075) |
| 262144 | 1 | old | 1.49 | 0.8 % | 1129 (1135) |
| 262144 | 4 | new | 0.86 | 7.7 % | 1252 (1293) |
| 262144 | 4 | new_sparspak | 0.69 | 1.8 % | 1151 (1157) |
| 262144 | 4 | old | 3.02 | 0.6 % | 1699 (1705) |
| 1000000 | 1 | new | 4.25 | 3.6 % | 3029 (3029) |
| 1000000 | 1 | new_sparspak | 2.53 | 9.0 % | 2251 (2404) |
| 1000000 | 1 | old | 11.56 | 1.4 % | 2753 (2755) |
| 1000000 | 4 | new | 4.43 | 1.7 % | 3497 (3538) |
| 1000000 | 4 | new_sparspak | 2.93 | 3.0 % | 2977 (3014) |
| 1000000 | 4 | old | 24.81 | 2.5 % | 5199 (5205) |

All 90 cases completed. No case was killed by the 10 GB watchdog and none timed out.

## Reading the figures

**The largest size the old path reached is 10⁶ dofs, for both partial counts.** Its peak at
10⁶ was 2.75 GB for `B = 1` and 5.2 GB for `B = 4`, about half the 10 GB cap. The cap was
not reached, so this run records no kill.

**Primal-once at the same solver (Sparspak, 10⁶ dofs).** `new_sparspak` against `old`:
2.53 s against 11.56 s for `B = 1` (4.6×) and 2.93 s against 24.81 s for `B = 4` (8.5×). Peak
footprint 2.3 GB against 2.8 GB (`B = 1`) and 3.0 GB against 5.2 GB (`B = 4`). In this run the gap
widens with `B`, as the `Dual` factor's cost grows with the partials.

**The default route at 10⁶ dofs.** `new` against `old`: 4.25 s against 11.56 s (`B = 1`,
2.7×) and 4.43 s against 24.81 s (`B = 4`, 5.6×). Most of that gain is the primal-once effect
and some is the change of solver. Here the default's UMFPACK LU is slower than the Sparspak
primal (`new` against `new_sparspak`: 4.25 s against 2.53 s, and 4.43 s against 2.93 s), and
its peak is higher (3.0 GB against 2.3 GB; 3.5 GB against 3.0 GB). The default is what the
extension uses, so these are the figures users see. The solver choice for this matrix is a
separate question, and this page does not answer it.

**Below about 65536 dofs the figures are noise.** Times are milliseconds, and every
footprint is inside the 610 to 650 MB idle baseline of a julia process with Bramble loaded.
Some spreads reach 25 to 32 %. Ratios at those sizes should not be read.

**Agreement.** The three paths agree at every size. Taking the norms of the value and of the
partials of each solution, their relative spread across paths is at most 7 × 10⁻¹² at 10⁶
dofs. The 16 × 16 warm-up is checked entrywise against the old path to a relative 10⁻¹⁰.

## Caveats

- The peak is the largest of one-second samples of `top`. A spike shorter than a second can be
  missed, and at the smallest sizes the sampled peak can read lower than the case's own `before`
  reading.
- The timed solve includes factorization and the solves, as the old path does. Assembly is not
  timed; its memory is part of the baseline footprint.
- Each row is three separate processes run back to back, so the spread includes process-level
  variation, not only repeats of one solve.
- One machine (Apple M2). These figures are not portable to other hardware.
- The old path is `sparspak_factorize(A) \ F`. The old default route, which overflowed the stack
  with the extension loaded, is not measured here.
- The matrices are nonsymmetric. A symmetric positive definite system takes Accelerate in the
  default route on macOS. That case is not measured here.

## Reproduce

```bash
.claude/scripts/check_power_load.sh
.claude/scripts/gate_preflight.sh --timing
BRAMBLE_JULIA_SLOTS=4 .claude/scripts/julia_slot.sh 4 julia --startup-file=no \
    --threads=4 --project=test benchmark/forwarddiff_primal_solve.jl --full
```

`--smoke` runs two small sizes once. It checks the plumbing and is not a timing.
