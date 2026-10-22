# Ghost precompile workload: what each warm-up pays (v4.0.1, S2.2)

Measured on the S2.1 base (f01fdecc, `constrained` type-stable), Apple Silicon, AC, 4
threads, default `-O`. Method: a fresh temporary depot per build
(`JULIA_DEPOT_PATH=<tmp>:~/.julia`, `JULIA_PKG_OFFLINE=true`), `using Bramble` timed with
`/usr/bin/time -p` inside `julia_slot.sh 4`, median of 3. One build drops one warm-up
through a temporary env toggle (not committed). The first call is timed in a fresh process
on a non-uniform mesh (2D 9x9, 3D 4x4x4 with a variable coefficient), median of 3, once
per depot.

Load: 1-min load 1.93 when the builds started (under the 2 gate); the load average then
sat at 3 to 8 during the runs, from the Julia processes' own threads plus background
processes, so the decisions rest on the same-run ratios below, not on absolute seconds.

## Precompile time and pkgimage size

No warm-up was dropped, so before and after are the same code. The build time is the whole
`using Bramble` in a fresh depot (Bramble and its workload; the dependencies are cached in
`~/.julia`).

| build | time, s (3 runs) | median, s | depot `compiled/`, KiB | Bramble pkgimage, KiB |
|---|---|---|---|---|
| full workload (before = after) | 97.84, 98.48, 99.17 | 98.48 | 262544 | 250652 |
| no workload at all | 9.80, 9.93, 9.97 | 9.93 | 30820 | n/a |
| no ghost nested session | 86.95, 88.07, 90.36 | 88.07 | 238148 | 226256 |
| no 3D nests | 94.75, 95.44, 96.25 | 95.44 | 250940 | 239048 |
| no conormal relation | 94.77, 94.85, 95.56 | 94.85 | 255808 | 243916 |
| no mixed second differences | 97.66, 98.35, 99.36 | 98.35 | 262544 | 250656 |
| no plain `Δₕ(Δₕ(u))` | 97.97, 98.63, 100.73 | 98.63 | 262560 | 250672 |

The workload is about 88.5 s of the 98.5 s, and 244.8 MiB of the Bramble pkgimage.

## Candidates

Cost is the full-build median minus the median without that warm-up. Saving is the first
call without it minus the first call with it.

| candidate | precompile cost, s | pkgimage cost | first call with, s | first call without, s | saving, s | verdict |
|---|---|---|---|---|---|---|
| 2D conormal `∂ₙ(u; K)` relation | 3.6 | 6.6 MiB | 0.327 | 4.054 | 3.7 | keep |
| 3D nests (`jumpₓ(Δₕ)`, `D₋ₓ(Δₕ)`, `jumpₙ(Δₕ)`) | 3.0 | 11.3 MiB | 0.101 | 3.837 | 3.7 | keep |
| plain and ghost mixed `Dcₓ(Dcᵧ)`, `D̽ₓ(D̽ᵧ)` | 0.1 (inside the noise) | 0.1 MiB | 0.043 | 0.105 | 0.06 | keep |
| plain `Δₕ(Δₕ(u))` | none measurable | none | 0.0103 | 0.0105 | 0 | keep |
| whole ghost nested session (for scale) | 10.4 | 23.8 MiB | 0.276 | 3.363 | 3.1 | keep |

The first-call cases: conormal is `constrained` with `2u + ∂ₙ(u; K) = 1` on four facets,
`Δₕ` form, assemble and `pde_solve`; mixed assembles the four mixed forms (plain and
ghost-constrained); nest2 assembles `innerₕ(Δₕ(Δₕ(u)), v)`; 3D assembles the three nests
of the workload; the whole-session row is the Robin ghost problem of #811.

## What was dropped, and why

Nothing. The rule is to drop a warm-up only when its precompile cost exceeds its first-call
saving, and none does.

- Conormal and the 3D nests each cost about 3 to 3.6 s to precompile and save about 3.7 s
  on the first call, so they pay back; the conormal is the closest (3.6 against 3.7).
- The mixed differences cost nothing measurable (the pkgimage does not change by more than
  its run to run noise) and save 0.06 s, so dropping them would gain nothing.
- `Δₕ(Δₕ(u))` costs nothing measurable and its first call is 10 ms with or without the
  warm-up: the other warm-ups already compile that path, so the line is redundant but free.
  Dropping it changes neither the build time nor the user's first call.
- The issue rejects dropping everything at once, and the table shows why it would be wrong:
  the two warm-ups that cost anything are the two that save seconds.

## The ghost session now, against #811

The whole ghost nested session costs 10.4 s and 23.8 MiB of pkgimage now (#811 measured
22 s and 47 MiB), so S2.1's type-stable `constrained` roughly halved it. The first call of
the Robin ghost problem is 0.28 s with the session and 3.4 s without.

Per-facet compile that S2.1 introduced: each relation compiles one form per facet of the
dimension (four in 2D). The conormal relation's four facet forms cost 3.6 s together, so
about 0.9 s per facet if shared evenly; this is an estimate, not a per-facet measurement,
because the facets are compiled in one workload call and cannot be toggled apart.

## Ghost operators, rank-one and semidiscretize first calls (#812, S3)

Measured on 2edafc95 (S2.2's result) with the S2.2 method above, a fresh temporary depot
per build and `/usr/bin/time -p` inside `julia_slot.sh 4`, 4 threads, default `-O`, median
of 3. Each candidate is dropped by building a copy of the source without it, not by a
runtime toggle. A toggle leaves the skipped call inside a method the workload infers, so
its code is compiled anyway (a first try measured the `semidiscretize` session that way
and saw a 0.06 s saving where the real one is 0.44 s). Each first call is timed in its own
fresh process, median of 3, on the PROBE's case, the non-uniform 2D (11, 14) mesh, the
Robin space `2u + ∂ₙ(u) = 0.5` on four facets with `ghosts = :boundary`, and the
stiffness form `inner₊(∇ₕ(u), ∇ₕ(v))`. The rank-one form is that form plus
`0.5 * innerₕ(1, u) * innerₕ(1, v)`, assembled; the plain rank-one form is the same on
`gridspace(Ωₕ)`; semidiscretize is `semidiscretize(a, l)` with
`l = form(C, v -> innerₕ(x -> 1.0, v))` and one residual call. The control, a plain
stiffness `assemble`, takes 0.010 s before and after.

Load: OneDrive synced twice during the work (1-min load up to 23), and the blocks run then
were discarded. The builds below ran at a 1-min load of 3.3 to 4.9, from the build's own
threads.

### First calls, before and after

| path | before, s | after, s |
|---|---|---|
| ghost `kronecker_operator(a) * x` | 2.186 | 0.0034 |
| ghost `matrix_free_operator(a) * x` | 1.109 | 0.0002 |
| ghost rank-one `form` and `assemble` | 0.548 | 0.0060 |
| plain rank-one `form` and `assemble` | 0.207 | 0.086 |
| ghost `semidiscretize` and residual | 0.993 | 0.179 |

The issue measured 2.42 s and 1.29 s for the two operators on the same case.

### Precompile time and pkgimage size

| build | time, s (3 runs) | median, s | depot `compiled/`, KiB | Bramble pkgimage, KiB |
|---|---|---|---|---|
| before (2edafc95) | 95.37, 95.67, 95.68 | 95.67 | 262056 | 250164 |
| after | 99.11, 99.24, 99.26 | 99.24 | 274148 | 262256 |

The added core precompile time is 3.6 s and the pkgimage grows by 11.8 MiB. The two rows
come from consecutive blocks at the same load; within the second block, the build without
the operators (below) is 96.16 s, so the operators carry 3.1 s of the 3.6 s.

### Candidates

Cost is the full-build median minus the median without that warm-up, all from one block
of interleaved builds. Saving is the first call without it minus the first call with it.

| candidate | precompile cost, s | pkgimage cost | first call without, s | first call with, s | saving, s | verdict |
|---|---|---|---|---|---|---|
| ghost `matrix_free_operator` and `kronecker_operator`, a product each | 3.1 | 10.3 MiB | 2.114 + 1.082 | 0.0034 + 0.0002 | 3.19 | keep |
| ghost rank-one form | none measurable (-0.4) | 0.3 MiB | 0.221 ghost, 0.203 plain | 0.006, 0.086 | 0.33 | keep |
| ghost `semidiscretize` session | 0.07 (inside the noise) | 0.5 MiB | 0.616 | 0.179 | 0.44 | keep |

Nothing was dropped.

- The operators cost about what they save, 3.1 s against 3.2 s, as S2.2's conormal did.
  They are the issue's two figures and the subplan's gate.
- The rank-one form and the `semidiscretize` session cost no build time the runs resolve
  (the full build's three runs span 0.15 s, the others' up to 0.7 s) and half a MiB at
  most, and save 0.33 s and 0.44 s.

The two operators run behind `Base.inferencebarrier`. They compile for the concrete form
all the same (the after column above), but JET then does not follow the calls into unions
their bodies guard at run time (`_ghost_kron_closures`'s per-axis closures, the padded
`matrix_free_operator`), which `report_package` counted as 5 missing methods when the
workload called them directly.
