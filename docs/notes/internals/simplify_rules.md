# `simplify_ast` rules 1-4 and 7: measured with and without each rule

Audit of gpena/Bramble.jl#771. Each rule is switched off by a patch
(`benchmark/simplify_rules/rule<N>.patch`, applied to a detached scratch worktree at the
measured commit; nothing ships in `src/`) and the same form is timed with the rule on (A)
and off (B). Rules 5, 6, 8 and 9 are correctness rules and are not audited.

Every cell is the median over fresh-process pairs, the min-max range in the spread column.
"First" is the first `form` plus the first `assemble`; "steady" is `assemble!` (the min of
15 calls in the same process). The forms are the scalar `s2d_mix` (2D) and `s3d_mix` (3D),
written so that every audited rule fires on them, and for rule 7 the 27-term `s3d_27`.

| Rule | What it does | Steady `assemble!`, with / without | First `form`+`assemble`, with / without | Spread (min-max, with / without) | Verdict |
|---|---|---|---|---|---|
| 1 | `c*A + c*B -> c*(A+B)`, `Integer` `c`, component-free | `s2d` 456 µs / 454 µs; `s3d` 408 µs / 374 µs | `s2d` 869 ms / 875 ms; `s3d` 1.06 s / 1.07 s | steady `s3d` 406-409 µs / 372-375 µs; first `s2d` 846-902 ms / 850-885 ms | remove |
| 2 | like terms `c1*A + c2*A -> (c1+c2)*A` | `s2d` 456 µs / 494 µs; `s3d` 407 µs / 427 µs | `s2d` 874 ms / 856 ms; `s3d` 1.04 s / 986 ms | steady `s3d` 406-422 µs / 426-436 µs; first `s3d` 1.03-1.07 s / 980-994 ms | remove |
| 3 | zeros: `0*A` collapses, `A + 0` drops | `s2d` 456 µs / 530 µs; `s3d` 406 µs / 440 µs | `s2d` 877 ms / 880 ms; `s3d` 1.04 s / 1.02 s | steady `s2d` 454-458 µs / 527-547 µs; first `s3d` 1.03-1.07 s / 997 ms-1.13 s | keep |
| 4 | nested scale `c1*(c2*A) -> (c1*c2)*A` | `s2d` 456 µs / 482 µs; `s3d` 407 µs / 422 µs | `s2d` 873 ms / 945 ms; `s3d` 1.04 s / 1.07 s | first `s2d` 868-877 ms / 941-974 ms; steady `s3d` 407-408 µs / 421-434 µs | keep |
| 7 | shared inner-product argument `⟨Au,Bv⟩ + ⟨Au,Cv⟩ -> ⟨Au,(B+C)v⟩` (`_absorb`/`_factor`) | 27-term `s3d_27` 4.5 ms / 902 µs; `s2d` 455 µs / 523 µs; `s3d` 407 µs / 417 µs | `s3d_27` 1.55 s / 3.29 s; `s2d` 875 ms / 934 ms; `s3d` 1.03 s / 1.23 s | `s3d_27` steady 4.5 ms / 901-903 µs, first 1.55-1.57 s / 3.29 s | remove |

Verdict by the keep criterion of #771: a rule is kept if it wins one of the two times and
loses neither, a win or loss being a difference larger than the observed spread (every A
sample faster, or slower, than every B sample; overlapping ranges are a tie). Per rule:

- **1** loses steady-state on the 3D form (408 µs against 374 µs, 9% slower with the
  rule, five pairs, disjoint ranges) and wins nothing: ties everywhere else.
- **2** wins steady-state (456 against 494 µs in 2D, 407 against 427 µs in 3D) but loses
  the 3D first call by 5% (1.04 s against 986 ms, nine pairs, disjoint ranges).
  A win on one time and a loss on the other, so `remove` by the criterion.
- **3** wins steady-state in 2D (456 against 530 µs, 14% faster) and ties the rest. A first
  run of three pairs showed a 3% first-call loss in 3D that nine pairs did not reproduce
  (ranges overlap), so it is a tie.
- **4** wins the 2D first call (873 against 945 ms) and the 3D steady state (407 against
  422 µs) and loses nothing.
- **7** wins the first call everywhere it fires (the 27-term form 1.55 s against 3.29 s,
  in line with the 6.6 s against 16.4 s noted in `form.measurements.md`), and loses
  steady-state on that same form by a factor of five: 4.5 ms with the rule, 902 µs
  without. It also wins in the two mixed forms. One loss is enough for `remove` by the
  criterion, but the figure is a surprise worth a look before the rule goes: a one-term
  factored product assembles five times slower than the 27 products it replaces, so
  the fewer sweeps seem to be paid for inside the stencil (not investigated here).

## Forms, meshes, commit, machine

- Commit measured: `fe3e8659` (branch `v4.0.0/rest`), detached scratch worktrees
  `/tmp/M1-base` and `/tmp/M1-rule<N>`; `benchmark/simplify_rules.jl --rule N --base DIR
  --patched DIR` reruns a pair.
- Meshes, all graded (a different exponent per axis, so non-uniform):
  `s2d_mix` 200x180, `s3d_mix` 28x25x22, `s3d_27` 22x20x18, `v2d_elast` 90x80,
  `v3d_elast` 16x14x12, `v3d_lap` 22x20x18 points.
- Forms: `s2d_mix` and `s3d_mix` combine `2X + 2Y`, `X + 3X`, `0*X`, `2*(3*X)` and a trial
  operator shared by several products, with `Integer` coefficients; `s3d_27` is 27 distinct
  `innerₕ` products, nine trial operators each meeting three test operators (the scalar
  form behind `_factor`'s first measurement); `v2d_elast` and `v3d_elast` are
  `2 inner₊(εₕ(u), εₕ(v)) + 2 inner₊(divₘₕ(u), divₘₕ(v))`; `v3d_lap` is
  `inner₊(∇ₕ(u), ∇ₕ(v))`.
- Machine: Apple M2, 8 cores (4 performance, 4 efficiency), 16 GB, macOS 27.0.1.
- Julia 1.13.2, default optimisation, 4 threads (`--threads=4`), every timed process
  through `julia_slot.sh 4` with `BRAMBLE_JULIA_SLOTS=4`.
- Power and load: AC power throughout (`check_power_load.sh`: "AC Power", load 1.9 to 3.9
  before the sessions). `gate_preflight.sh --timing` failed on each session only because
  other agents' one-thread Julia jobs ran in the pool and the 1-minute load from the
  previous session had not decayed; no foreign process above 100% CPU was seen during any
  sample (logged at both ends of every sample, at most 1.1%). 1-minute load across the
  kept pairs, which include this run's own four threads: rule 1 3.4-4.4, rule 2 1.7-5.0,
  rule 3 1.7-2.6, rule 4 1.6-2.8, rule 7 1.8-2.5. No pair was discarded.

## Footer: skipped forms and pairs

- Forms where the rule leaves the simplified AST type identical (so A and B are the same
  program) are not timed: rules 1-4 are a no-op on `s3d_27`, `v2d_elast`, `v3d_elast` and
  `v3d_lap` (the vector forms are component-indexed, and the 27 products carry no
  coefficients); rule 7 is a no-op on `v2d_elast`, `v3d_elast` and `v3d_lap`. Routed-term
  counts with / without the rule on the timed forms: rule 1 `s2d` 4/4, `s3d` 5/5; rule 2
  4/5, 5/6; rule 3 4/4, 5/5; rule 4 4/5, 5/6; rule 7 4/6, 5/8 and `s3d_27` 1/27.
- Pairs per timed form: rule 1 five; rule 2 nine; rule 3 nine; rule 4 five; rule 7 three
  (the order of A and B swapped on every other pair). Rules 2 and 3 were rerun with more
  pairs because three disagreed with an earlier, noisier run on the first-call time.
- The matrix assembled with and without each rule agreed (checksums compared in the
  driver).
