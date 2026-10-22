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
