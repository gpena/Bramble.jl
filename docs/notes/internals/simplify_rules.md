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

## Candidate rules

The candidates #771 proposes, each prototyped as a patch
(`benchmark/simplify_rules/cand_<name>.patch`, a `git diff` against the tree at `9a7e093a`,
the rule set that remains after the removals; nothing ships in `src/`) and measured by the
same protocol as above with `benchmark/simplify_rules.jl --candidate NAME --base DIR
--patched DIR`. "With" is the tree carrying the candidate, "without" the base. A candidate is
`keep` only if it wins one of the two times, loses neither beyond the spread, is bit-exact
on every timed form, keeps `form` inferred (`@inferred`) and keeps `simplify_ast` idempotent
on the stored AST. Spread is the min-max over the settled pairs. Rows are written as each
candidate is measured.

| Candidate | What it does | Steady `assemble!`, with / without | First `form`+`assemble`, with / without | Spread (min-max, with / without) | Matrix, inference, idempotence | Verdict |
|---|---|---|---|---|---|---|
| shared_wrapper | `⟨Au, W(x)⟩ + ⟨Au, W(y)⟩ -> ⟨Au, W(x + y)⟩` for one difference or average `W`, shared argument settled by singleton type, component-free only (re-introduces part of removed rule 7, restricted to a shared wrapper; routed terms 3/5 and 4/7) | `s2d_shared` 632 µs / 669 µs; `s3d_shared` 732 µs / 722 µs | `s2d_shared` 576 ms / 701 ms; `s3d_shared` 934 ms / 1.20 s | steady `s2d` 625-646 / 669-671 µs, `s3d` 732-734 / 720-723 µs; first `s2d` 559-624 / 693-777 ms, `s3d` 894-995 ms / 1.16-1.31 s | pattern equal, NOT bit-exact (10225 of 285345 and 5705 of 185497 stored entries differ, max relative 2.2e-16: the merged wrapper adds the two operands before the scatter); inferred, idempotent; no-op on `s3d_27` (different axes) and `v3d_elast` | drop |
| scalar_lift | `D₋ₓ(c * u) -> c * D₋ₓ(u)` under a difference, average, jump or shift, so the scalar meets the nested-scale fold and the product lift | `s2d_lift` 524 µs / 536 µs; `s3d_lift` 309 µs / 331 µs; `v2d_lift` 2.73 ms / 2.77 ms | `s2d_lift` 728 ms / 725 ms; `s3d_lift` 916 ms / 941 ms; `v2d_lift` 3.53 s / 3.22 s | steady `s2d` 523-525 / 536-538 µs, `s3d` 308-310 / 330-351 µs, `v2d` 2.71-2.75 / 2.75-2.78 ms; first `s3d` 914-930 / 934-1050 ms, `v2d` 3.49-3.65 / 3.19-3.29 s | pattern equal; scalar forms NOT bit-exact (39583 of 321724 and 32188 of 161092 stored entries differ, max relative 3.2e-16: `c(a - b)` against `ca - cb` for `c` = 3, 5); `v2d_lift` bit-exact; inferred, idempotent; no-op on `s3d_27`, `v3d_elast` | drop |
| same_wrapper | `D₋ₓ(u) + D₋ₓ(w) -> D₋ₓ(u + w)`: one wrapper over a same-component sum, for differences, averages and jumps of component-free operands (a shift keeps its runtime amount); routed terms 4/4 and 5/5, nodes 29/32 and 38/42 | `s2d_sum` 753 µs / 735 µs; `s3d_sum` 863 µs / 858 µs | `s2d_sum` 719 ms / 618 ms; `s3d_sum` 1.01 s / 951 ms | steady `s2d` 752-755 / 731-738 µs, `s3d` 862-863 / 855-859 µs; first `s2d` 712-756 / 618-622 ms, `s3d` 1.01-1.03 s / 945-976 ms | pattern equal, bit-exact yes (0 entries differ, both forms); inferred, idempotent; no-op on `s3d_27`, `v3d_elast` | drop |
| component_type | same-component factoring decided by a component index in the type, so `c*A + c*B` factors when `A` and `B` name one component | not timed | not timed | not timed | not applicable | not built: the component index is an `Int` field of `IndexedTrialFunction`/`IndexedTestFunction`, and a type parameter reaches every operator and the router (outside `simplifier.jl`) and makes a runtime `u(i)` infer a `Union`; the type-decidable part is the removed rule 1 and the component-free gate used by shared_wrapper and same_wrapper |
| rational | folds two coefficients, one a `Rational` and the other a `Rational` or `Integer` of at most 64 bits, into one `Rational` of the next wider integer type (`(1//2) * ((1//3) * A)` stays exact and cannot overflow, #633 holds); routed terms equal, nodes 25/28, 32/36, 128/149, 358/417 | `s2d_rat` 341 µs / 336 µs; `s3d_rat` 216 µs / 216 µs; `v2d_rat` 2.45 ms / 2.45 ms; `v3d_rat` 11.1 ms / 11.7 ms | `s2d_rat` 676 ms / 578 ms; `s3d_rat` 845 ms / 757 ms; `v2d_rat` 5.67 s / 5.45 s; `v3d_rat` 35.3 s / 32.6 s | steady `s2d` 340-369 / 336-340 µs, `v3d` 11.1-11.2 / 11.7-11.8 ms; first `s2d` 630-773 / 573-590 ms, `s3d` 830-869 / 750-761 ms, `v2d` 5.63-5.80 / 5.43-5.54 s, `v3d` 35.3-35.4 / 32.3-32.7 s | pattern equal, bit-exact yes on all four forms (the coefficients are `/2`, `/3`, `/4` and small integers); inferred and idempotent (`v3d_rat`: `form` is not concretely inferred with or without the candidate, the 3D elasticity form as at the base) | drop |
| interp_fold | `πₕ(u)` is dropped when the leaf it interpolates from is the leaf the sweep walks (identity), decided at block time in `_bind_interp_spaces` by `===` of the two leaves; a patch of `src/operators/interpolation.jl`, not of `simplifier.jl`, since only there is the space known; the stored AST is unchanged, so the AST probe cannot see it and the form is timed with `--always` | `s2d_interp` 296 µs / 3.99 ms; `s3d_interp` 227 µs / 1.86 ms | `s2d_interp` 476 ms / 568 ms; `s3d_interp` 497 ms / 761 ms | steady `s2d` 296 / 3.98-4.00 ms, `s3d` 226-228 µs / 1.86-1.87 ms; first `s2d` 473-497 / 566-569 ms, `s3d` 494-510 / 759-766 ms | values equal (max `|A-B|` 0.0) but the stored pattern differs: 179240 against 286520 and 104068 against 222868 entries, the base storing explicit zeros of the identity interpolation; `form` inferred, idempotent; caveats for M4: the bound type becomes a `Union` when both leaves share a type (the choice is a runtime `===`), the fold fires only on a same-space `πₕ`, an artificial form, and `assemble!` into a matrix built without the fold sees another pattern | keep |
| cse | the same difference of one grid function in several products evaluated once per sweep (3D elasticity first) | not timed | not timed | not timed | not applicable | not built: sharing is an execution change in the router, not a `simplify_ast` rewrite, and one was built, bit-exact, and lost (first assembly 1.133x slower, warm 0.907x; `form.md`, "Shared sub-operators across terms"); the AST-level form of it is removed rule 7 (4.5 ms against 902 µs warm on `s3d_27`) |

Reading the table. Two candidates are bit-exact and lose on time (`same_wrapper` and
`rational`; the first call is 4-17% slower on every timed form, the cause not investigated).
Two win time and fail bit-exactness by reassociation at 1e-16 relative. `shared_wrapper`
has a first call 18-22% faster in 2D and 3D and a steady state 5% faster in 2D and 1% slower
in 3D. `scalar_lift` has a steady state 2-7% faster, a first call 10% slower on the vector
form, and rounding differences for a coefficient that is not a power of two. `interp_fold`
is the one win, and a different kind of rule. It leaves `simplify_ast`, and its gain is the
cost of an identity interpolation, large only because that interpolation is slow in the
base (2D 296 µs against 3.99 ms).

## Candidates: forms, meshes, commit, machine

- Commit measured: `9a7e093a` (branch `v4.0.0/rest`), detached scratch worktrees
  `/tmp/M3-base` and `/tmp/M3-<name>` (the patch applied); `benchmark/simplify_rules.jl
  --candidate NAME --base DIR --patched DIR` reruns a candidate. The base is the rule set
  that remains after the removal of rules 1, 2 and 7.
- Machine: Apple M2, 8 cores, 16 GB, Julia 1.13.2, 4 threads (`--threads=4`), default
  optimisation, AC power, every timed process through `julia_slot.sh 4` with
  `BRAMBLE_JULIA_SLOTS=4`.
- Meshes, graded as above: `s2d_*` 200x180, `s3d_*` 28x25x22, `v2d_lift` and `v2d_rat`
  90x80, `v3d_rat` 16x14x12 points. Forms: `*_lift` scalars `2, 3, 4, 5` under differences
  and averages (`v2d_lift`: `2 * u` under `εₕ`, `3 * u` under `divₘₕ`); `*_sum`
  `innerₕ(D₋ₓ(u) + D₋ₓ(D₊ᵧ(u)), D₋ₓ(v))` and similar; `*_shared` pairs
  `⟨D₋ₓ(u), D₊ₓ(v)⟩ + ⟨D₋ₓ(u), D₊ₓ(D₊ᵧ(v))⟩`; `*_rat` `(A / 2) * 3`, `A(u / 2, v / 3)`,
  `(A / 4) * 2`; `*_interp` `innerₕ(πₕ(u), v) + innerₕ(πₕ(u), D₋ₓ(v))` plus differences.
- Load (1-minute, at both ends of every kept sample, including this run's own four
  threads) across the settled pairs: shared_wrapper 2.34-4.46, scalar_lift 1.75-3.69,
  same_wrapper 1.82-3.46, rational 1.78-3.03, interp_fold 1.88-2.43; foreign julia at most
  1.0% CPU in every sample; no pair discarded. The table is not "measured under load".
- Pairs per form (3 first, up to 5 while the signs or spread disagreed): shared_wrapper 5
  and 3; scalar_lift 5, 5, 5; same_wrapper 3 and 5; rational 5, 5, 5, 3; interp_fold 3, 3.
- Skipped no-op forms (identical simplified AST type, not timed): `s3d_27` and `v3d_elast`
  for scalar_lift, same_wrapper and shared_wrapper (27 routed terms and 36, nodes
  161 and 382, both with and without). `s3d_27`'s trial operators meet three different
  test axes, so no shared wrapper exists; vector forms carry runtime component indices,
  which the `_component_free` gate refuses. component_type and cse are not built and time
  nothing.
- Every timed form also reports `@inferred form`, idempotence of `simplify_ast` and the
  two matrices bit for bit (first pair). The 3D elasticity `v3d_rat` is not concretely
  inferred with or without `rational`.
