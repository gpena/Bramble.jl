# Structural padding for restricted products (gpena/Bramble.jl#800)

#800 asks whether a product's stencils can each have one fixed length, with the padding a
zero known from the type rather than decided at each point. Two things would follow. The
refill would run on stencils of one type and stop boxing without a product being split
into pieces, and a padded slot would add nothing, so an `Inf` beside padding stays `Inf`
(#795). This note weighs two designs. It rejects one from a count of the stencil types it
needs, prototypes the other in two scopes, and measures both against the reference tree.

| Tree | Branch | Commit | Worktree |
| --- | --- | --- | --- |
| reference | `alloc-guarantees-2` | `6bf0ea71` (`main` at `a0ca3a40` plus this branch's fixes) | `~/worktrees/Bramble/alloc-guarantees-2` |
| full prototype | `eval/padding` | `aeff58bc` | detached copy at `~/bramble-scratch/s61-padding/full` |
| scoped prototype | `eval/padding` | `af13c791` (tip; timed at `24b358ca`) | `~/worktrees/Bramble/eval-padding` |

`eval/padding` is local and never pushed. It holds four commits over `6bf0ea71`:
`adbc288d` (padding, products whole), `aeff58bc` (the reference's grouping kept, the groups
summed into one product: the full prototype), `24b358ca` (padding scoped to products
holding two restrictions or more) and `af13c791` (that scope without a `BigFloat` weight
or a grid function scale over a restricted sum: the scoped prototype). `af13c791` builds
every one of the 30 timed forms exactly as `24b358ca` does (0 B, one product, the same
pattern and bit-identical values), so the timings taken at `24b358ca` stand for it; it was
not timed again. The forms #800 cites at `test/form/bilinear.jl:869` and
`:1543` are at lines 909 and 1577 on the reference.

## Two designs

Design (a), liveness in the type. Each class of points, by which restrictions hold there,
gets its own statically resolved stencil: inside a region the operand, outside it nothing,
so no padding is ever multiplied. A shifted tap reads the neighbour's marker, so a class is
set by membership over the stencil's footprint, not at the point alone. The count of
distinct footprints is the number of stencil types, and of compiled walks, one product
needs. Counted with the reference's marker table (`classes_membership.jl`, below), labels
`:l, :r, :b, :t, :c` (`:c` is `x₁ < 1/2`), footprint a box of radius `r` (`r = 1` for one
difference or average a side, `r = 2` for two):

| D | mesh | 1 label | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 41, `r = 1` | 3 | 5 | 8 | | |
| 2 | 7x9, `r = 1` | 7 | 13 | 19 | 25 | 35 |
| 2 | 33x33, `r = 1` | 7 | 13 | 19 | 25 | 40 |
| 2 | 33x33, `r = 2` | 16 | 31 | 40 | 49 | 84 |
| 3 | 9x9x9, `r = 1` | 19 | 37 | 55 | 73 | 149 |
| 3 | 9x9x9, `r = 2` | 76 | 151 | 196 | 241 | 440 |

(1D has three labels.) The count grows with every summand and roughly threefold per
dimension. The reference already caps the split at 16 products because each one is a walk
to compile: a sixteen-piece split took the first `assemble` from 0.51 s to 4.0 s
(`src/assembly/bilinear.jl`, the comment above `_split_restricted_products`). Design (a)
would compile 25 to 73 walks for the four- and five-label forms below. It was not built: this is an estimate
from the class counts, not a timing.

Design (b), structural padding with liveness as a value, is the one prototyped. Every restriction in a
product answers its operand's stencil at every point, so a product has one stencil type
whatever the region holds. Each weight carries a two-bit mask (`_Padded{T}`, two
fields): *absent* (mask 0) where the reference answered `()`, *present and dead* (mask 1)
where it stored a zero (a shifted tap outside the region, a tap zeroed by `_zero_entries`
or by the off-grid clamp of `_fixed_shift`), *live* (mask 3) otherwise. Products and
scales keep the mask (`a * b` is live only where both are). The walk (`_visit_entries`)
skips an absent entry without advancing the slot, as it skips an off-grid one, and adds
nothing for a dead one. So the coordinate walk and the replays drop the same entries, the
stored pattern is the reference's, and a dead slot is never multiplied into the matrix.

The mask is option 1's mechanism from `docs/notes/padding_zero.md` (a weight type carrying
liveness through `*`), plus a presence bit so the pattern can stay the reference's. The
difference is where it is applied. Options 1 and 2 kept `()` and the split, and added a padded
weight only at the sites that zero-scale. A product that already held a union of
stencil shapes then gained one more weight type and boxed more (13 to 14 capped figures
rose). Design (b) replaces the `()` itself, so the union is gone and the product needs no
split. One product needs one stencil type at any number of restricted summands and in
any dimension. The type cost is one `_Padded{T}` weight type per weight type `T` (an
`Int` leaf weight, then the element type after the first product).

## What the prototype changes

Seven files, `git diff 6bf0ea71 af13c791 --shortstat`: +181 −14 (the scope rules are
+55 −7 of that over `aeff58bc`).

- `src/ast/common.jl`: `_Padded`, its `*` methods, `_pad_mask`, `_present`, `_value`,
  `_padded_all` (a stencil's weights all padded where one is, so the walk indexes one
  type) and `_gate_stencil`.
- `src/operators/region_restriction.jl`: `_PadOutside`, a fourth value of
  `RegionRestriction`'s `zero_stencil` parameter: the node answers its operand's stencil
  everywhere, masked live or absent. A shifted tap outside the region is marked dead
  rather than scaled by `zero(T)` (`_region_gate`).
- `src/assembly/linear.jl`: `_zero_entries` marks a padded weight dead and keeps its value.
- `src/assembly/stencil_eval.jl`: `_fixed_shift`'s off-grid zero marks a padded weight
  dead; a padded restriction's `_chain_in` is marked live, the type of its zeroed branch.
- `src/assembly/bilinear.jl`: `_split_product` sums each side's groups into one operand
  and returns one padded product instead of one product per pair (the grouping and the
  distribution through nodes are the reference's, which keeps its stored pattern).
  `_padded_restrictions` marks every unprobed restriction of a source-free operand
  `_PadOutside` (an operand holding a `SourceFunction`, `DiracSource`, source data or a
  `Function` is left unpadded, as it may not be evaluated outside its region). In the
  scoped prototype a product the split leaves whole is padded only when it holds two
  restrictions or more (`_restriction_count`), and no product is padded that carries a
  non-bits weight (`_bits_weights`) or a grid function scale over a restricted sum
  (`_scaled_restricted_sum`): those keep the reference's split.
- `src/assembly/bilinear_traversal.jl`: `_visit_entries` and `_visit_entries_unguarded`
  skip absent entries and add `_value(w)`.
- `src/assembly/matrix_free.jl`: `_mf_point_weights` strips the masks (compiles; values not
  checked, see below).

Linear forms are untouched: `_split_restricted_products(::LinearProduct)` and
`_typed_restrictions` are the reference's. The off-grid clamp of `ShiftNode` and
`_reevaluated_shift`, which `padding_zero.md` left out of #795's scope, is untouched too.

## Values

Every value figure is identical on the full and the scoped prototype unless the row says
otherwise. Scripts are under `~/bramble-scratch/s61-padding/`; `repro.jl`, `genuine.jl`,
`clamp.jl`, `corpus.jl` and `corpus_compare.jl` are copied verbatim from `padding_zero.md`.

| Check | Reference | Full prototype | Scoped prototype |
| --- | --- | --- | --- |
| #795 reproduction, `gi[k] = ±Inf`, k = 2, 9, 16, Float64 and Float32 (12 cases) | 2 `NaN` each, at `(k-1, k)` and `(k, k)`; 0 `Inf` | 12 of 12: 0 `NaN`, 1 `Inf`, `A[k-1, k] = ±Inf` | as the reference (the form holds one restriction, so it is not padded) |
| Genuine `0 * Inf` outside padding (`genuine.jl`) | `A[9, 9] = NaN`, 1 `NaN`; under `D₋ₓ` 1 `NaN` | the same | the same |
| Off-grid clamp sites (`clamp.jl`), `NaN` entries | 6, 4, 2 | 6, 4, 2 | 6, 4, 2 |
| #795 finite corpus (43 forms) | | 43 bit-identical, 43 same stored pattern | 43 bit-identical, 43 same stored pattern |
| Measured forms below (30), stored pattern | | 30 of 30 the reference's | 30 of 30 |
| Measured forms below, values | | 26 bit-identical; 4 the reference splits differ by at most 2.2e-16 relative | the same |
| Every refill pin of `test/form/bilinear.jl` and `test/form/linear_restriction.jl` (`pins.jl` from `padding_zero.md`: 54 pin lines, 476 measurements) | all pass | not run | `af13c791`: 465 equal, 11 lower, 0 higher; no 0 B pin moves |

The four forms that differ (`L3 | R3`, `L5 | L5[1:4]`, `DD(C(u+RL)) | DD(C(v+RB))`,
`MD(RL)+JD(RB) | MD(RB)+JD(RL)`) are those the reference assembles as several products.
One product adds each entry's terms in another order. The corpus has no such form, which is
why it stays bit-identical.

The pin census is the check that rejected options 1 and 2, and it caught `24b358ca` too:
12 measurements rose. Ten are `BigFloat` forms (lines 1397, 1531, 1548, 1563, 1574, from
1.4 to 3.6 times their reference bytes, 557808 to 761280 B at line 1574), as each padded
entry is one more `BigFloat` product; two are the grid-function-scaled sums at line 1119
(65344 to 91728 B against a 72800 B cap, 85344 to 1110816 B against 91376 B). `af13c791`
leaves both kinds to the reference's split, and then every pin sits at or under the
reference's bytes: the 11 lower are line 919 (82848 to 0), line 1657's nine (all to 0) and
line 1236 (49408 to 47056 B). The census run reports two other failures on `af13c791`,
the product-count pins at `bilinear.jl:807` and `:809` (`== 2` and `== 9`; one product
each now).

## Measurements

Each figure is three alternating back-to-back runs per tree (reference, prototype,
reference, ...), at `--threads=4`, the default `-O`, `--startup-file=no`, every process
through `BRAMBLE_JULIA_SLOTS=4 julia_slot.sh 4`. Every table gives the median, the spread
(min to max) and the ratio of each round's pair (prototype over reference): median and
range.

Every timed run is under load. `gate_preflight.sh --timing` failed before every round. In every round `suggestd` (macOS `CoreSuggestions`) held one core at 99 to 100 %.
The 1-minute load read 1.8 to 2.9. Before the scoped rounds 2 and 3 the check also
listed another session's Julia processes (60120 60180, 60997 61052). No timed run
started after a passing preflight. The alternating ratios are the evidence, not the
absolute seconds.

### Warm refill bytes, 7x9 (1D: 11 points; 3D: 5x6x7), seeded non-uniform meshes

`assemble!` twice, then `@allocated assemble!`. The same in every run.

| Form | Reference | Full | Scoped |
| --- | --- | --- | --- |
| `L5 \| L5[1:4]` (#800's :869) | 82848 | 0 | 0 |
| `U5 \| V5` | 11952 | 0 | 0 |
| `L3 \| R3`, `id + R(u) \| id + R(D₋ₓ(v))` | 0, 0 | 0, 0 | 0, 0 |
| :1543 set, 1D: `Mₓ(r(u+L(u))), D₋ₓ(2.0L+3Rr)`; `Mₓ(L+Rr+u+0.5u), C(jumpₓ(…))`; `Mₓ(L+C(Rr)), C(r(L+Rr+v))`; `u, S₊ₓ(C(C(v+L)))`; `u, S₊ₓ(C(C(L+Rr+v)))`; `D₊ₓ(L+Rr+u), S₊ₓ(…)` | 3440, 5568, 2896, 0, 1728, 5184 | all 0 | all 0 |
| :1543 set, 2D: the same six, then `2D₊ₓ(2.0L+3Rr), M₊ᵧ(2.0RR)`, `M₊ᵧ(L+C(Rr)), M₊ᵧ((1//3)(…))` | 30384, 42336, 27216, 0, 13104, 39312, 49536, 20736 | all 0 | all 0 |
| #767 paired average `MD(L+R) \| MD(B+T)`, `MD = Mₓ∘D₊ᵧ` | 0 | 0 | 0 |
| `MD(S3) \| MD(S3)`, `S4`, `S5` (`Sk` the first k of `:l, :r, :b, :t, :c`) | 0, 23776, 55296 | 0, 0, 0 | 0, 0, 0 |
| `DD(R4) \| DD(RB)`, `DD(C(u+RL)) \| DD(C(v+RB))`, `MD(RL) \| MD(RB)+v`, `MD(RL)+u \| MD(RB)`, `MD(RL)+JD(RB) \| MD(RB)+JD(RL)` | 0 each | 0 each | 0 each |
| 1D `Mₓ(D₋ₓ(L+R))` both sides; 3D `MD(L+R) \| MD(B+T)` | 0, 0 | 0, 0 | 0, 0 |

Every measured form reaches 0 B under both prototypes. #767's figures (92208 B with two
summands, 637744 B with five) are not the reference's any more: it refills the
two-summand form at 0 B and allocates only from four summands.

### First call (`form` + `assemble`), seconds, one fresh process per group per tree per round

Forms in one process share compilation, so a group's total is the primary figure.
`MD(S2) | MD(S2)` is left out: its type equals the form before it, so its 0.04 s is a
reuse.

| Group | Reference | Full | ratio | Scoped | ratio |
| --- | --- | --- | --- | --- | --- |
| :869 and its rows (4 forms) | 2.997 (2.986-3.113) | 2.179 (2.172-2.180) | 0.73 (0.70-0.73) | 2.196 (2.186-2.206) | 0.70 (0.69-0.72) |
| :1543 set (14 forms) | 8.649 (8.600-8.735) | 5.081 (5.062-5.208) | 0.59 (0.58-0.60) | 5.112 (5.100-5.116) | 0.58 (0.58-0.59) |
| #767 paired (11 forms) | 20.90 (20.89-21.05) | 17.52 (17.43-17.55) | 0.84 (0.83-0.84) | 17.47 (17.47-17.56) | 0.83 (0.83-0.83) |

(The full prototype's paired total includes `MD(S2)`, 0.04 s on either tree. The scoped
ratios are against their own alternating reference runs, 3.139, 8.727 and 21.03 s.)

Forms that move most, scoped against its reference (the full prototype is within 3 % of
these): the jump forms of the :1543 set ×0.32 and ×0.33, `MD(S4)` ×0.65, `L5 | L5[1:4]`
×0.62. Five forms are slower, every one already at 0 B on the reference:
`DD(R4) | DD(RB)` ×2.44 (0.58 → 1.42 s), `MD(RL) | MD(RB)+v` ×2.02, `MD(RL)+u | MD(RB)`
×1.80, `id + R(u) | id + R(D₋ₓ(v))` ×1.09, 3D `MD(L+R) | MD(B+T)` ×1.07. Each holds two
restrictions or more, so the scope rule pads it. Padding gains nothing there, and one long
masked stencil compiles slower than the reference's pieces.

### Warm `assemble!`, 201x201 seeded non-uniform mesh, microseconds

The minimum over repeated refills (up to 2000, about 1.5 s of them), default policy
(serial).

| Form | Reference | Full | ratio | Scoped | ratio | Reference bytes |
| --- | --- | --- | --- | --- | --- | --- |
| #795 form, finite `gi` | 390.3 (377.7-392.5) | 420.5 (420.4-421.0) | 1.079 (1.071-1.114) | 390.0 (388.1-390.2) | 0.995 (0.990-1.001) | 0 |
| control, `innerₕ(u, v) + inner₊(∇ₕu, ∇ₕv)` | 237.8 (237.4-245.9) | 245.7 (242.6-248.4) | 1.033 (0.987-1.046) | 247.4 | 1.017 (1.005-1.023) | 0 |
| `innerₕ(u, v) + innerₕ(u, R(:b, v))` | 51.33 (51.29-52.29) | 53.04 (52.88-53.08) | 1.030 (1.015-1.034) | 52.75 | 1.013 (1.002-1.028) | 0 |
| `L5 \| L5[1:4]` | 10220 (10160-10600) | 9179 (9148-9194) | 0.895 (0.866-0.905) | 9391 | 0.926 (0.916-0.991) | 51876576 |
| `D₊ₓ(L+Rr+u), S₊ₓ(C(C(L+Rr+v)))` | 5712 (5579-5742) | 988.5 (988.0-988.8) | 0.173 (0.172-0.177) | 1015 | 0.194 (0.193-0.214) | 23222736 |
| `MD(S4) \| MD(S4)` | 57420 (57360-57630) | 67940 (67550-68790) | 1.183 (1.178-1.194) | 69250 (69190-69550) | 1.184 (1.178-1.192) | 14259744 |

Both prototypes refill every one of these at 0 B.

Placement control. A third tree, the reference plus one unused function
(`~/bramble-scratch/s61-padding/noop`), ran beside the reference and the full prototype,
three alternating rounds. Its own ratio to the reference was 1.012 on the control and
1.019 on the `R(:b, v)` form. Full prototype over that tree: control 1.002 (0.999-1.022),
`R(:b, v)` 1.008, #795 form 1.079 (1.078-1.084), `MD(S4)` 1.186 (1.167-1.190). So the
1-3 % rows are code placement. The #795 form's 7.9 % and `MD(S4)`'s 18 % are real.

Two costs, with different causes:

- The #795 fix costs about 8 % warm on the padded form, whatever the design. Option 1
  measured +5.9 %, option 2 +7.7 % and the full prototype +7.9 % on the same form. Any
  mechanism that stops a padded slot from multiplying pays it. A one-restriction product
  never boxes on the reference, and padded it carries its operand's stencil at every point
  where the reference's compiler split `()` from the full tuple.
- `MD(S4)`'s 18 % comes from the fixed length itself. At an interior point every boundary
  region is absent, so the reference's stencils there are empty and the prototype's hold
  every padded entry: four restrictions under a four-tap average make 16 entries a side and
  256 masked products a point, at every one of 40401 points. The reference boxes there, at
  14 MB a refill, and is still faster.

### Compile cost

| Figure | Reference | Full | Scoped |
| --- | --- | --- | --- |
| Precompile cache, `.ji` + `.dylib` (bytes); scoped column at `24b358ca` | 146286617 | 146376092 (+0.06 %) | 146358889 (+0.05 %) |
| `using Bramble`, s, 3 rounds | 0.695 (0.695-0.787) | 0.704 (0.698-0.708), ratio 1.013 | 0.702 (0.701-0.703), ratio 1.009 |
| Invalidations loading Bramble (`uinvalidated`, trees) | 29, 8 | 29, 8 | 29, 8 |
| Specialisations inferred, :869 group (frames, of them Bramble's) | 3830, 1907 | 3285, 1469 | 3332, 1511 |
| the same, :1543 group | 7027, 4990 | 7266, 5042 | 7383, 5150 |
| the same, #767 paired group | 6725, 4243 | 7114, 4464 | 7213, 4557 |
| Inference time the snoop reports, s, :869 / :1543 / paired (one run, not alternated) | 4.57 / 12.09 / 32.41 | 3.51 / 7.52 / 25.36 | 3.23 / 7.21 / 20.07 |
| Products per form, the 30 measured | 1 to 16 (`U5 \| V5` 16, `L5` 12, `L3` 9, `DD(C(…))` 9, `DD(R4)` 8) | 1 each | 1 each |

Counts are from `SnoopCompileCore.@snoop_inference` around forming and assembling every
form of the group once (`snoop.jl`, distinct `MethodInstance`s). They and the
invalidations come from one process per tree and group. They ran beside other probes, so
the inference seconds are indicative only; the counts are deterministic. The prototype
infers fewer instances for the :869 group (its 12- and 16-piece splits become one product
each) and 3 to 7 % more for the other two (the masked stencil types).

## Not measured

- Linear forms (`LinearProduct`): the prototype does not pad them.
- Matrix-free products, GPU backends (Metal, KernelAbstractions), ghost spaces, the
  Jacobian and residual paths, and AD (Enzyme or ForwardDiff through `_Padded`): the
  prototype compiles the matrix-free weight read and nothing more.
- Threaded refill (`assemble!` under `CpuThreaded` or `CpuPolyester`): every warm figure
  above is the default serial policy.
- `af13c791`'s timings and snoop counts (it builds the 30 timed forms as `24b358ca` does),
  `BigFloat` timings (#800 leaves them out; their bytes are in the pin census), Julia 1.14,
  and the full test suite (only the forms above and the two pinned test files ran).
- A finer scope rule (pad only where the reference would box) and block liveness (one
  mask per restricted summand, skipping a whole absent block): either might remove the
  `MD(S4)` and first-call costs, but neither was built. That is a guess.

## Recommendation

Under the rule `padding_zero.md` applied (every value constraint holds, and every
first-call and warm figure sits within the reference's spread), neither prototype passes.
The full prototype fails warm on the #795 form (+7.9 %) and `MD(S4)` (+18 %). The scoped
one fails warm on `MD(S4)` (+18 %) and first call on five forms the reference already
assembles at 0 B (×1.07 to ×2.44). The scoped prototype buys this:
- no refill pin of the two pinned test files rises, and every timed form refills at 0 B (16 of them allocated 1728 to 82848 B on
  the small meshes; 14 MB, 23 MB and 52 MB a refill on 201x201);
- the three groups' first calls fall by 17 to 42 %;
- warm refills of the forms that boxed run up to 5.2 times faster;
- the stored pattern is the reference's, and the corpus stays bit-identical.

The allocation-guarantees milestone puts 0 B refills first, which is why the
recommendation sets the warm rule aside for `MD(S4)` openly rather than quietly. The #795
`Inf` is a separate question: the third design to fix it costs about 8 % warm on the
padded form, so padding is not the way to fix it.

Recommendation: adopt the scoped structural padding (`af13c791`) for bilinear products of two restrictions or more with bits weights, giving up the within-spread rule for warm `MD(S4) | MD(S4)` (+18 % at 201x201, against 14 MB a refill on the reference) and for the first call of five forms already at 0 B (×1.07 to ×2.44, `DD(R4) | DD(RB)` worst), and leaving #795's `Inf` as the reference has it

### Implementation plan (subplans for `bramble-plan`)

S1. Padded weights and the walk.
- OWNS: `src/ast/common.jl`, `src/assembly/bilinear_traversal.jl`,
  `src/operators/region_restriction.jl`, `src/assembly/stencil_eval.jl`,
  `src/assembly/linear.jl` (`_zero_entries` only), and a new `test/form/structural_padding.jl`.
- CHECK: the new test file. It pins each `_Padded` mask rule, absent entries skipped
  without advancing the slot, and a dead weight's value kept, on a 7x9 non-uniform mesh.
- EXPECT: pass, and `test/form/bilinear.jl` unchanged and passing (no product is padded
  yet).
- WORKER: `opus-high` (dense: the coordinate walk and the replays must drop the same
  entries, or `slot` drifts).
- WHY: a two-bit mask on the weight, read by the walk. Rejected: liveness in the type,
  which needs 7 to 149 stencil types a product (the class table) where the split cap is 16.

S2. Bilinear products padded and kept whole.
- OWNS: `src/assembly/bilinear.jl`, `test/form/bilinear.jl`.
- CHECK: `test/form/bilinear.jl`, its restricted-sum byte pins of bits-weight forms
  rewritten to `== 0`, the product-count pins at lines 807 and 809 rewritten to `== 1`,
  and a `nnz` pin and a parts-equal check on each; then `pins.jl` against the reference.
- EXPECT: bits-weight pins 0 B; `BigFloat` pins and the line-1119 pins at or under their
  caps (the reference's bytes); nnz as on the reference (171, 82, 69, the :1577 set's);
  0 pins higher in the census.
- WORKER: `opus-high`.
- WHY: sum the reference's groups into one padded product, which keeps its stored
  pattern bit for bit, and leave non-bits weights and grid-function-scaled restricted sums
  to the split, which the census shows box more padded. Rejected: dropping the grouping and padding the operand as built,
  which stored 45 to 99 fewer zeros on seven corpus forms (`adbc288d`) and changes every
  `nnz` pin.

S3. The other walks.
- OWNS: `src/assembly/matrix_free.jl`, `src/assembly/ghost_assembly.jl`,
  `src/assembly/jacobian_pattern.jl`, the Metal and KernelAbstractions extensions'
  stencil reads.
- CHECK: their restricted-form tests (matrix-free against the assembled matrix, ghost,
  Jacobian pattern) with `Metal`, `KernelAbstractions` and `Polyester` loaded.
- EXPECT: pass, values equal to the assembled matrix.
- WORKER: `opus-medium`.
- WHY: strip the mask where weights are read (`_value`). Rejected: keeping the split for
  these paths, which keeps two AST shapes per form.

S4. AD through padded weights.
- OWNS: the Enzyme and ForwardDiff rules that read stencil weights, and their tests.
- CHECK: the AD test group on one restricted-sum form, DI against direct Enzyme.
- EXPECT: equal gradients; threaded policies differentiate.
- WORKER: `opus-high`.
- WHY: `_Padded` is a two-field bits struct Enzyme must see through. Rejected: an
  `isbits` tuple encoding, which `entry_weights` would split into mixed types.

S5. Timing gate.
- OWNS: `benchmark/` (a restricted-forms case for `MD(S4)` and `L5`).
- CHECK: the three-round alternating warm and first-call runs of this note on a machine
  whose preflight passes.
- EXPECT: the ratios above within their spread.
- WORKER: `sonnet-medium`.
- WHY: every figure here is under load. Rejected: trusting these figures for a baseline.

## Rerunning the figures

Every script is under `~/bramble-scratch/s61-padding/` (scratch, not in the repository):
`groups.jl <plain|padtaps|paired> <out.jls>` (bytes, nnz, first call, matrices),
`groups_compare.jl`, `warm.jl`, `load.jl`, `snoop.jl`, `invalidations.jl` (SnoopCompile
from a scratch environment `snoopenv` stacked on `JULIA_LOAD_PATH`), `products.jl`,
`classes_membership.jl`, the drivers `values.sh`, `timing.sh`, `timing2.sh`,
`placement.sh`, `compile.sh`, and `summary2.jl`, which prints the tables. Each tree needs
`.claude/scripts` and the workspace `Manifest.toml` copied in.
