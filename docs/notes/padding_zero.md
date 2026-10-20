# Padding as a strong zero: options 1 and 2 prototyped (gpena/Bramble.jl#795)

A stencil padded with zeros turns an infinite coefficient into `NaN`: the padding is a plain
`zero(T)` by the time it meets the other factor, and `0 * Inf` is `NaN`. #795 lists three
options and four constraints. This note records the **values** of options 1 and 2, each
built on a local branch from `042ad0f4` (the base of the v4.0.0-final plan), against the
constraints that are not timings. Compile and warm-assembly costs, and the recommendation,
are P1b's (a later section of this note); nothing here was timed.

| Tree | Branch | Commit | Worktree |
| --- | --- | --- | --- |
| base | `main` | `042ad0f4` | `~/worktrees/Bramble/proto-795-base` (detached) |
| option 1 | `proto/795-option-1` | `5f3d7a2a749b351c3e18a401590110c0063ddf87` | `~/worktrees/Bramble/proto-795-option-1` |
| option 2 | `proto/795-option-2` | `c02eb43bf091674454d5f5f4e6364846a6252d1e` | `~/worktrees/Bramble/proto-795-option-2` |

Both branches are local and never pushed. Each is one commit over `042ad0f4`.

## The padding sites both prototypes convert

Both prototypes route every site the issue names through one helper, `_pad(s, live::Bool)`,
which keeps the weight value base gave (`zero` where the slot is dead) and marks each entry
live or dead. The sites, with base's line numbers:

- `src/operators/region_restriction.jl:148` (`_chain_stencil`): `_pad(s, !outside)`.
- `region_restriction.jl:237` (`shifted_inner_stencil` over a restriction): the
  `_in_region(…) ? one(T) : zero(T)` scale becomes `_pad(scale_stencil(s, one(T)), _in_region(…))`.
- `region_restriction.jl:268` (a scale chain shifted): `_pad(s, !(outside || clamped))`.
- `src/assembly/linear.jl:340` (`_zero_entries`, the zero stencil `form` stores for a probed
  restriction): `_pad(s, false)`. So that a probed restriction keeps one stencil type, its
  inside stencil is padded live too (`_inside_stencil` in `local_stencil` and
  `_source_stencil_at`), as are the inner results of `_probed_chain` and `_shifted_chain`.
- `src/assembly/stencil_eval.jl:568` (`_fixed_shift`): `on_grid ? one(T) : zero(T)` becomes
  `_pad(scale_stencil(s, one(T)), on_grid)`.
- `stencil_eval.jl:654` (`_chain_out(…) && return _zero_entries(…)`): `_pad(…, false)`, and
  the `_chain_in` branch `_pad(…, true)`, so both branches have one type.
- `concatenate_stencils` (`src/ast/common.jl`): a stencil holding one padded entry holds only
  padded ones (the plain side is padded live, decided from the types), so a stencil's
  entries keep one type, the homogeneity #759 and #760 established.

Not converted (not named by the issue): the off-grid clamp padding of `_reevaluated_shift`
(`src/ast/common.jl:492`) and of `ShiftNode` (`src/operators/shift.jl:562`, `:568`). On
both prototypes an `Inf` trial coefficient at a boundary node still gives the `NaN` entries
base gives there (`D₊ₓ(D₋ₓ(v))` 6, `Mₓ(D₋ₓ(v))` 4, `D₋ₓ(D₋ₓ(v))` 2 over nodes 1, 7, 8, 14 of
the 7 by 9 mesh; `clamp.jl` below). Whether those zeros are padding or the operator's own
boundary rule is a question for P2's scope.

## Pins the rewrite misses, and the critic's confirmation

`pins.jl` rewrites `@test <bytes call> == / <=`, `@test_allocs` and `@test b == 0` lines.
It misses four allocation pins: `@test alloc_test(Bramble._zero_entries, s40) == 0`
(linear_restriction.jl line 186) and the `contract` pins (bilinear.jl lines 2173, 2182 and
2193). They are not recorded in the byte tables, and they pass on base and on both
prototypes (the test files run whole).

The integrator's critic (one round) ran the real, unmodified test files on both branches.
Each fails with 14 failures. Option 1's are its 14 byte caps. Option 2's are its 13 byte
caps plus the `_zero_entries` layout test (linear_restriction.jl line 185). The failures
and their bytes are identical at 1 and 2 threads, at -O1 and the default -O, and byte for
byte equal to the `pins.jl` figures above. The critic also found that the `NaN` entries
left under both options sit only at the off-grid clamp sites named above, never at a slot
outside the region (its scripts `variants.jl` and `detail.jl`, under `/tmp/p1a-critic/`).

## Option 1

**A strong-zero weight type.** `_Padded{T} <: Number` holds the weight and a live flag.
Arithmetic keeps the flag (`a * b` is live only if both are), and the weight collapses to a
plain `T` only where it is added or converted (`_strong(p) = ifelse(p.live, p.w, zero(p.w))`,
through `+`, `convert` and a `promote_rule`): a dead slot gives zero there whatever it was
multiplied by, as `false * Inf == 0.0`. The products (`local_stencil(::BilinearProduct)`,
`multiply_stencils_bilinear_symmetric`, `multiply_stencils_linear`, `_product_entries`) are
untouched: the sinks' `nzval[k] += α * w` collapses the weight. Four files, +65 −15.

| Constraint | Figure (option 1) | Base |
| --- | --- | --- |
| The issue's reproduction, `Inf` and `-Inf` at nodes 2, 9, 16, Float64 and Float32 | 0 `NaN`, 1 `Inf` each, `A[k-1, k] = ±Inf` (12 of 12 cases) | 2 `NaN` (at `(k-1, k)`, `(k, k)`), 0 `Inf` |
| A genuine 0 * Inf outside padding (`innerₕ(gi * u, z * v)`, `gi[9] = Inf`, `z[9] = 0`) | `A[9, 9] = NaN`, 1 `NaN`; under `D₋ₓ(z * v)` 1 `NaN` (Float64, Float32) | the same |
| Refill bytes of the byte pins in test/form/bilinear.jl and test/form/linear_restriction.jl that `pins.jl` records (57 pin lines, 532 measurements; the four it misses are listed below) | all 500 measurements at 0 B stay 0 B; 512 equal, 16 higher, 4 lower; 14 above their recorded cap | every pin at or under its cap |
| Finite-coefficient corpus (43 forms), values | bit-identical, 43 of 43 | — |
| Finite-coefficient corpus, stored pattern | 43 of 43 unchanged | — |

Bytes that rose, base → option 1, by pin line. In bilinear.jl, line 869 82848 → 93600
(cap 82848); line 1468, the `BigFloat` form, 557808 → 736656 (cap 617808); line 1543,
twelve of the fourteen 1D and 2D restricted-sum forms, by 6 % to 21 % (3440 → 3904, 6000 →
7264, 2896 → 3344, 2256 → 2432, 6768 → 7296, 30384 → 33408, 48096 → 56736, 27216 → 30672,
21168 → 23184, 63504 → 69552, 49536 → 52704, 20736 → 23472). In linear_restriction.jl, line
498 11088 → 15696 twice (under its cap). Four fell, in bilinear.jl line 1153 44704 → 33808,
line 1303 89760 → 88224, line 1442 62160 → 57840 and 118176 → 115872. Every capped form is
one that already boxes at base (a `BigFloat` weight, a restricted sum the walk cannot
split), and a padded weight there is one more boxed type. The test files otherwise pass,
the 14 failures being those pins.

## Option 2

**A per-slot live flag beside the weight.** A padded entry is `(offset, _Live(live),
weight)`. The flag travels through `scale_stencil` and `shift_stencil` (which now keeps
`Base.tail(t)` rather than `t[2]`), and every product reads it: `_live_product(l, r, vol) =
ifelse(live(l) & live(r), l[end] * r[end] * vol, zero)` in `multiply_stencils_bilinear`,
`multiply_stencils_linear` and `multiply_stencils_bilinear_symmetric`, the same `ifelse` in
`_product_entries` (bilinear_traversal.jl, the factor-by-factor walk of #765), and
`sum_stencil_values` sums live weights only. `entry_offsets` drops the flag and
`entry_lives` reads it. Six files, +73 −24. A plain entry's flag is the constant `true`,
which the compiler can fold away; P1b measures whether it does.

| Constraint | Figure (option 2) | Base |
| --- | --- | --- |
| The issue's reproduction, `Inf` and `-Inf` at nodes 2, 9, 16, Float64 and Float32 | 0 `NaN`, 1 `Inf` each, `A[k-1, k] = ±Inf` (12 of 12 cases) | 2 `NaN`, 0 `Inf` |
| A genuine 0 * Inf outside padding | `A[9, 9] = NaN`, 1 `NaN`; under `D₋ₓ(z * v)` 1 `NaN` (Float64, Float32) | the same |
| Refill bytes of the pins `pins.jl` records (57 pin lines, 532 measurements) | all 500 measurements at 0 B stay 0 B; 508 equal, 22 higher, 2 lower; 13 above their recorded cap | every pin at or under its cap |
| Finite-coefficient corpus (43 forms), values | bit-identical, 43 of 43 | — |
| Finite-coefficient corpus, stored pattern | 43 of 43 unchanged | — |

Bytes above their cap, base → option 2, by pin line. In bilinear.jl, line 982 56688 →
62736 (cap 56688); line 1468 (`BigFloat`) 557808 → 749184; line 1543, ten forms, by
1 % to 20 % (6000 → 7200, 2896 → 3248, 2256 → 2432, 6768 → 7296, 48096 → 56016, 27216 →
29520, 21168 → 23184, 63504 → 69552, 49536 → 50112, 20736 → 23040). In
linear_restriction.jl, line 181 (a `BigFloat` linear form) 65760 → 71808. Higher but within
their caps, bilinear.jl line 982's `big(3) * v + 0.5 * D₋ₓ(v)` form 63744 → 69792 (cap
72000), and lines 1303, 1425, 1442 and 1457 (by 3264 to 5472 B). One more
failure is not a pin. linear_restriction.jl line 185 compares `_zero_entries` with base's
entry layout, which the flag changes by design (P2 would rewrite it).

## Option 3

**A value-level select on every product**, `ifelse(iszero(w), zero, w * c)`. Rejected in the
issue: it cannot tell padding from a genuine zero, so it also changes a genuine 0 * Inf, and
it costs a select per entry of every form, padded or not. The counterexample is the genuine
case above: with `gi[9] = Inf` on the trial side and `z[9] = 0` on the test side,
`innerₕ(gi * u, z * v)` has no padding at all, and IEEE gives `A[9, 9] = 0 * Inf = NaN` on
base and on both prototypes. Under option 3 the select sees `iszero(z[9])` and writes `0.0`,
hiding a `NaN` the coefficients really produce (`ifelse(iszero(0.0), 0.0, 0.0 * Inf) ===
0.0`, against `0.0 * Inf === NaN`). Options 1 and 2 decide from the marker or the clamp,
never from the value, so that entry stays `NaN`.

## Summary of the values (for P1b)

| | Reproduction | genuine 0 * Inf | refill bytes | Corpus |
| --- | --- | --- | --- | --- |
| Option 1 | fixed (0 `NaN`, `±Inf` at `(k-1, k)`) | still `NaN` | 0 B pins hold; 14 capped measurements above cap | bit-identical, same pattern |
| Option 2 | fixed | still `NaN` | 0 B pins hold; 13 capped measurements above cap | bit-identical, same pattern |
| Option 3 | (not built) | becomes `0.0` | — | — |

Neither prototype meets the refill constraint as the issue words it ("stays at 0 B, or at
its recorded cap"): every 0 B pin holds, but forms that already box at base box more. P1b
times the two branches; whether a cap rise of this size is acceptable is the
recommendation's call, not this note's.

## Rerunning the figures

The scripts are below; save each under `/tmp/p795/` with the name in its heading. The base
and prototype worktrees need `.claude/scripts` and the workspace `Manifest.toml` copied in
and the test environment instantiated:

```bash
cd ~/worktrees/Bramble/v4.0.0-final
git worktree add --detach ~/worktrees/Bramble/proto-795-base 042ad0f4
git worktree add ~/worktrees/Bramble/proto-795-option-1 proto/795-option-1
git worktree add ~/worktrees/Bramble/proto-795-option-2 proto/795-option-2
for t in base option-1 option-2; do
    w=~/worktrees/Bramble/proto-795-$t
    mkdir -p $w/.claude && cp -R .claude/scripts $w/.claude/scripts && cp Manifest.toml $w/
    (cd $w && .claude/scripts/julia_slot.sh 1 julia --threads=1 --startup-file=no \
        --project=test -e 'using Pkg; Pkg.instantiate()')
done
```

(Skip a `git worktree add` whose worktree exists already.) Then, each tree alone, about ten
minutes each at one thread (the pins run both test files whole):

```bash
chmod +x /tmp/p795/run_tree.sh
/tmp/p795/run_tree.sh ~/worktrees/Bramble/proto-795-base base
/tmp/p795/run_tree.sh ~/worktrees/Bramble/proto-795-option-1 opt1
/tmp/p795/run_tree.sh ~/worktrees/Bramble/proto-795-option-2 opt2
cd ~/worktrees/Bramble/proto-795-base
J() { .claude/scripts/julia_slot.sh 1 julia --threads=1 --startup-file=no --project=test "$@"; }
cat /tmp/p795/repro-*.txt /tmp/p795/genuine-*.txt          # reproduction, genuine 0 * Inf
J /tmp/p795/corpus_compare.jl /tmp/p795/corpus-base.jls /tmp/p795/corpus-opt1.jls
J /tmp/p795/corpus_compare.jl /tmp/p795/corpus-base.jls /tmp/p795/corpus-opt2.jls
J /tmp/p795/pins_compare.jl /tmp/p795/pins-base.tsv /tmp/p795/pins-opt1.tsv
J /tmp/p795/pins_compare.jl /tmp/p795/pins-base.tsv /tmp/p795/pins-opt2.tsv
grep -E "Test Failed at|Error During Test at" /tmp/p795/pins-opt*.txt   # the failing lines
for t in base option-1 option-2; do
    (cd ~/worktrees/Bramble/proto-795-$t && J /tmp/p795/clamp.jl)    # clamp sites
done
```

`corpus_compare.jl` compares stored values byte for byte and the stored pattern (`colptr`,
`rowval`, size). Its positive control, which must report one form's values and another's
pattern as differing:

```bash
J -e 'include("/tmp/p795/control.jl"); empty!(ARGS);
    push!(ARGS, "/tmp/p795/corpus-base.jls", "/tmp/p795/corpus-mutant.jls");
    include("/tmp/p795/corpus_compare.jl")' /tmp/p795/corpus-base.jls /tmp/p795/corpus-mutant.jls
```

It printed `41 bit-identical, 42 same stored pattern` of 43. The pin bytes are
deterministic: every capped pin measured on base equals the figure its test records (82848,
557808, 65760, …).

## Scripts

### `repro.jl`

```julia
# The issue's reproduction (#795), plus the Float32 and -Inf variants of P2's PROBE.
using Bramble, Random
using Bramble: D₋ₓ, D₊ₓ, restrict_to
for T in (Float64, Float32), s in (1, -1)
    Random.seed!(20261014)
    W = gridspace(mesh(domain(interval(zero(T), one(T)) × interval(zero(T), one(T)), :l => :left),
        (7, 9), (false, false); backend = backend(T)))
    for k in (2, 9, 16)
        gi = element(W); gi .= 1 .+ rand(T, length(gi)); gi[k] = s * T(Inf)
        A = Matrix(assemble(form(W, W, (u, v) -> innerₕ(gi * u, D₊ₓ(D₋ₓ(restrict_to(:l, v)))))))
        println("repro T=$T sign=$s k=$k NaN=$(count(isnan, A)) Inf=$(count(isinf, A)) A[k-1,k]=$(A[k-1, k]) nan_at=$(Tuple.(findall(isnan, A)))")
    end
end
```

### `genuine.jl`

```julia
# A genuine 0 * Inf outside any padding: a test-side weight z with z[9] = 0 against a
# trial-side Inf at the same node. IEEE gives NaN; a strong zero must not reach here.
using Bramble
using Bramble: restrict_to, D₋ₓ
for T in (Float64, Float32)
    W = gridspace(mesh(domain(interval(zero(T), one(T)) × interval(zero(T), one(T)), :l => :left),
        (7, 9), (false, false); backend = backend(T)))
    gi = element(W); gi .= 1; gi[9] = T(Inf); z = element(W); z .= 1; z[9] = 0
    A = Matrix(assemble(form(W, W, (u, v) -> innerₕ(gi * u, z * v))))
    B = Matrix(assemble(form(W, W, (u, v) -> innerₕ(gi * u, restrict_to(:l, z * v)))))
    C = Matrix(assemble(form(W, W, (u, v) -> innerₕ(gi * u, D₋ₓ(z * v)))))
    println("genuine T=$T A[9,9]=$(A[9, 9]) NaN(A)=$(count(isnan, A)) | under a restriction B[9,9]=$(B[9, 9]) | under D₋ₓ NaN(C)=$(count(isnan, C))")
end
```

### `corpus.jl`

```julia
# Finite-coefficient corpus for #795: A2's and A3's probe forms (v4-0-0-rest.md) plus the
# 3D elasticity form of benchmark/multiterm_ttfx.jl, assembled once each. Writes, per form,
# the stored values (bits) and the stored pattern to ARGS[1] for corpus_compare.jl.
using Bramble, Random, SparseArrays, Serialization
using Bramble: D₋ₓ, D₊ₓ, D₊ᵧ, D₋ᵧ, Mₓ, jumpₓ, S₊ₓ, restrict_to
out = Dict{String, Any}()
rec(n, A::SparseMatrixCSC) = (out[n] = (:sparse, size(A), copy(A.colptr), copy(A.rowval), copy(A.nzval)))
rec(n, y::AbstractVector) = (out[n] = (:vector, size(y), Int[], Int[], collect(y)))
R = restrict_to
# A2's probe forms (seed and mesh as there).
Random.seed!(20261014)
mk(T) = gridspace(mesh(domain(interval(zero(T), one(T)) × interval(zero(T), one(T)), :l => :left,
    :r => :right, :b => :bottom, :t => :top, :c => x -> x[1] < 0.5), (7, 9), (false, false);
    backend = backend(T)))
W32 = mk(Float32); W64 = mk(Float64)
g32 = element(W32); g32 .= 1 .+ rand(Float32, length(g32)); g = element(W64); g .= 1 .+ rand(length(g))
f = x -> 1 + x[1]; RR = v -> R(:l, v) + R(:r, v)
lin = (("g*v + 0.5D₋ₓ(v), source g", W32, v -> innerₕ(g32, g32 * v + 0.5 * D₋ₓ(v))),
    ("innerₕ(f, 2v + 3D₋ₓ(v)) + innerₕ(1.0, v)", W32, v -> innerₕ(f, 2 * v + 3 * D₋ₓ(v)) + innerₕ(1.0, v)),
    ("innerₕ(f, πv + 3D₋ₓ(v)) + innerₕ(1.0, v)", W32, v -> innerₕ(f, π * v + 3 * D₋ₓ(v)) + innerₕ(1.0, v)),
    ("v + 0.1D₋ₓ(v)", W32, v -> innerₕ(f, v + 0.1 * D₋ₓ(v))),
    ("g*(0.1v + 3D₋ₓ(v))", W32, v -> innerₕ(f, g32 * (0.1 * v + 3 * D₋ₓ(v)))),
    ("g*v + 0.1v + 3D₋ₓ(v)", W32, v -> innerₕ(f, g32 * v + 0.1 * v + 3 * D₋ₓ(v))),
    ("Ref(0.1f0)*v + 0.5D₋ₓ(v)", W32, v -> innerₕ(f, Ref(0.1f0) * v + 0.5 * D₋ₓ(v))),
    ("R(:c, 0.1v + 3D₋ₓ(v)) Float32", W32, v -> innerₕ(f, R(:c, 0.1 * v + 3 * D₋ₓ(v)))),
    ("R(:c, 0.1v + 3D₋ₓ(v)) Float64", W64, v -> innerₕ(f, R(:c, 0.1 * v + 3 * D₋ₓ(v)))),
    ("0.5D₋ₓ(v) + g*v", W32, v -> innerₕ(f, 0.5 * D₋ₓ(v) + g32 * v)),
    ("g*v + 0.5D₊ᵧ(v)", W32, v -> innerₕ(f, g32 * v + 0.5 * D₊ᵧ(v))),
    ("innerₕ(1.0, v) + innerₕ(f, 2v + 3D₋ₓ(v))", W32, v -> innerₕ(1.0, v) + innerₕ(f, 2 * v + 3 * D₋ₓ(v))),
    ("innerₕ(f, v) + innerₕ(1.0, D₋ₓ(v))", W32, v -> innerₕ(f, v) + innerₕ(1.0, D₋ₓ(v))),
    ("innerₕ(g, v) + innerₕ(1.0, v)", W32, v -> innerₕ(g32, v) + innerₕ(1.0, v)),
    ("Ref*v + 0.5D₋ₓ(v) + 0.1D₊ᵧ(v)", W32, v -> innerₕ(f, Ref(0.1f0) * v + 0.5 * D₋ₓ(v) + 0.1 * D₊ᵧ(v))),
    ("R(:c, 0.1v + 3D₊ᵧ(v)) Float32", W32, v -> innerₕ(f, R(:c, 0.1 * v + 3 * D₊ᵧ(v)))),
    ("D₋ₓ(0.1v + 3D₊ᵧ(v)) Float32", W32, v -> innerₕ(f, D₋ₓ(0.1 * v + 3 * D₊ᵧ(v)))))
for (n, W, w) in lin
    rec("A2 lin: " * n, assemble(form(W, w)))
end
bil = (("u, v + 0.1D₋ₓ(v) Float32", W32, (u, v) -> innerₕ(u, v + 0.1 * D₋ₓ(v))),
    ("u, g*v + 0.5D₋ₓ(v) Float32", W32, (u, v) -> innerₕ(u, g32 * v + 0.5 * D₋ₓ(v))),
    ("u, Ref*v + 0.5D₋ₓ(v) Float32", W32, (u, v) -> innerₕ(u, Ref(0.1f0) * v + 0.5 * D₋ₓ(v))),
    ("u, g*(0.1v + 3D₋ₓ(v)) Float32", W32, (u, v) -> innerₕ(u, g32 * (0.1 * v + 3 * D₋ₓ(v)))),
    ("D₋ₓ(g*RR(u)), D₋ₓ(g*RR(v))", W64, (u, v) -> innerₕ(D₋ₓ(g * RR(u)), D₋ₓ(g * RR(v)))),
    ("g*(2.0*RR(u)), g*(2.0*RR(v))", W64, (u, v) -> innerₕ(g * (2.0 * RR(u)), g * (2.0 * RR(v)))),
    ("u, D₋ₓ(2.0*RR(v))", W64, (u, v) -> innerₕ(u, D₋ₓ(2.0 * RR(v)))),
    ("u, D₋ᵧ(D₊ᵧ(2.0*RR(v)))", W64, (u, v) -> innerₕ(u, D₋ᵧ(D₊ᵧ(2.0 * RR(v))))),
    ("Mₓ(D₊ᵧ(2.0*RR))", W64, (u, v) -> innerₕ(u, Mₓ(D₊ᵧ(2.0 * RR(v))))),
    ("Mₓ(D₊ᵧ(2*RR)) Int", W64, (u, v) -> innerₕ(u, Mₓ(D₊ᵧ(2 * RR(v))))),
    ("D₋ₓ(g*(R+R)) pinned", W64, (u, v) -> innerₕ(u, D₋ₓ(g * RR(v)))),
    ("g*(2.0*(R+R)) pinned", W64, (u, v) -> innerₕ(u, g * (2.0 * RR(v)))),
    ("Mₓ(g*D₊ᵧ(R+R)) pinned", W64, (u, v) -> innerₕ(u, Mₓ(g * D₊ᵧ(RR(v))))))
for (n, W, w) in bil
    rec("A2 bil: " * n, assemble(form(W, W, w)))
end
# A3's probe forms (seeds and meshes as there).
Random.seed!(20261014)
W = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :l => :left, :r => :right,
    :b => :bottom, :t => :top, :c => x -> x[1] < 0.5), (7, 9), (false, false)))
V = W × W; DDD = w -> D₊ₓ(D₋ₓ(D₊ᵧ(w)))
for (n, S, w) in (("composite DDD", V, (u, v) -> sum(innerₕ(DDD(u(i)), DDD(v(j))) for i in 1:2, j in 1:2)),
    ("jumpₓ(R(:c, v + R(:l, v) + R(:r, v)))", W, (u, v) -> innerₕ(u, jumpₓ(R(:c, v + R(:l, v) + R(:r, v))))),
    ("S₊ₓ(R(:l, v) + R(:r, v))", W, (u, v) -> innerₕ(u, S₊ₓ(R(:l, v) + R(:r, v)))))
    rec("A3: " * n, assemble(form(S, S, w)))
end
for T in (Float64, Float32)
    Random.seed!(20261017)
    I = interval(zero(T), one(T))
    Wt = gridspace(mesh(domain(I × I, :l => :left, :r => :right, :c => x -> x[1] < 0.5), (7, 9),
        (false, false); backend = backend(T)))
    RRt = v -> restrict_to(:l, v) + restrict_to(:r, v); r = Ref(T(0.1))
    fs = (("Mₓ(D₋ₓ(π RR(u))), D₋ₓ(2 RR(v))", (u, v) -> innerₕ(Mₓ(D₋ₓ(π * RRt(u))), D₋ₓ(2 * RRt(v)))),
        ("u, D₋ₓ(big(2.0) RR(v)) + 1//3 R(:c, v)", (u, v) -> innerₕ(u, D₋ₓ(big(2.0) * RRt(v)) + 1 // 3 * restrict_to(:c, v))),
        ("u, D₋ₓ(π RR(v)) + 2v", (u, v) -> innerₕ(u, D₋ₓ(π * RRt(v)) + 2 * v)),
        ("u, D₋ₓ(r RR(v)) + 0.5v", (u, v) -> innerₕ(u, D₋ₓ(r * RRt(v)) + 0.5 * v)))
    for (n, w) in fs
        rec("A3 $T: " * n, assemble(form(Wt, Wt, w)))
    end
end
# The issue's own form with finite coefficients, and the 3D elasticity form.
Random.seed!(20261014)
Wl = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :l => :left), (7, 9), (false, false)))
gi = element(Wl); gi .= 1 .+ rand(length(gi))
rec("#795 form, finite", assemble(form(Wl, Wl, (u, v) -> innerₕ(gi * u, D₊ₓ(D₋ₓ(restrict_to(:l, v)))))))
Random.seed!(20261014)
Ωc = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 5, 5), (false, false, false))
We = gridspace(Ωc)^Val(3); μ, λ = 1.0, 2.0
rec("3D elasticity", assemble(form(We, We, (u, v) -> 2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₘₕ(u), divₘₕ(v)))))
serialize(ARGS[1], out)
println("corpus: $(length(out)) forms written to $(ARGS[1])")
```

### `corpus_compare.jl`

```julia
# Compares two corpus files: values bit for bit (`isequal` on the stored values, so a NaN
# and a signed zero count) and the stored pattern (colptr, rowval, size).
using Serialization, SparseArrays
a = deserialize(ARGS[1]); b = deserialize(ARGS[2])
@assert keys(a) == keys(b)
bits(v) = reinterpret(UInt8, collect(v))
nbit = 0; npat = 0
for k in sort!(collect(keys(a)))
    x, y = a[k], b[k]
    pat = x[1] == y[1] && x[2] == y[2] && x[3] == y[3] && x[4] == y[4]
    val = pat && eltype(x[5]) == eltype(y[5]) && (isbitstype(eltype(x[5])) ? bits(x[5]) == bits(y[5]) : isequal(x[5], y[5]))
    pat || (global npat += 1; println("PATTERN DIFFERS: $k"))
    val || (global nbit += 1; println("VALUES DIFFER: $k", pat ? " max |Δ| = $(maximum(abs, x[5] .- y[5]))" : ""))
end
println("corpus compare: $(length(a)) forms, $(length(a) - nbit) bit-identical, $(length(a) - npat) same stored pattern")
```

### `pins.jl`

```julia
# Every refill byte pin of test/form/bilinear.jl and test/form/linear_restriction.jl, with
# its bytes: each `@test <bytes call> == / <= ...` and `@test_allocs` line is rewritten to
# record the bytes it measures (file, line, value per evaluation), then both files run as
# the suite runs them. Usage: julia --project=test pins.jl <tree> <out.tsv>
using Test
tree, outfile = ARGS[1], ARGS[2]
const _P795 = Dict{Tuple{String, Int}, Vector{Int}}()
_p795rec(f, l, b) = (push!(get!(_P795, (basename(String(f)), l), Int[]), b); b)
include(joinpath(tree, "test", "TestUtils.jl"))
pin = r"@test (_alloc\(A, a\)|_loop_bytes\([^()]*\)|restricted_bytes\([^()]*\)|bytes\((?:[^()]|\([^()]*\))*\)|\(@allocated [^\n]*?\)) (==|<=)"
xdir = mktempdir()
for name in ("bilinear.jl", "linear_restriction.jl")
    src = read(joinpath(tree, "test", "form", name), String)
    lines = split(src, '\n'; keepempty = true)
    for (i, ln) in enumerate(lines)
        ln = replace(ln, pin => s -> begin
            m = match(pin, s)
            "@test Main._p795rec(\"$name\", $i, $(m.captures[1])) $(m.captures[2])"
        end)
        ln = replace(ln, r"@test_allocs ([A-Za-z_!]+)\(((?:[^()]|\([^()]*\))*)\)" => s -> begin
            m = match(r"@test_allocs ([A-Za-z_!]+)\(((?:[^()]|\([^()]*\))*)\)", s)
            "@test Main._p795rec(\"$name\", $i, Main.TestUtils.alloc_test($(m.captures[1]), $(m.captures[2]))) == 0"
        end)
        ln = replace(ln, r"@test b == 0$" => "@test Main._p795rec(\"$name\", $i, b) == 0")
        lines[i] = ln
    end
    write(joinpath(xdir, name), join(lines, '\n'))
end
try
    @testset "refill pins" begin
        include(joinpath(xdir, "bilinear.jl"))
        include(joinpath(xdir, "linear_restriction.jl"))
    end
catch e
    println("testset ended with ", typeof(e))
end
open(outfile, "w") do io
    for k in sort!(collect(keys(_P795)))
        println(io, k[1], '\t', k[2], '\t', join(_P795[k], ','))
    end
end
println("pins: $(length(_P795)) pin lines, $(sum(length, values(_P795))) measurements -> $outfile")
```

### `pins_compare.jl`

```julia
# Compares two pins files (pins.jl output): per measurement, equal, higher or lower bytes;
# zero pins (base 0 B) that stopped being 0 B are listed apart.
rd(f) = Dict((s = split(l, '\t'); (s[1], parse(Int, s[2])) => parse.(Int, split(s[3], ','))) for l in eachline(f))
a = rd(ARGS[1]); b = rd(ARGS[2])
eq = up = down = zlost = 0
for k in sort!(collect(keys(a))), (x, y) in zip(a[k], b[k])
    x == y ? (global eq += 1) : x < y ? (global up += 1; println("UP   $(k[1]):$(k[2]) $x -> $y B")) : (global down += 1; println("DOWN $(k[1]):$(k[2]) $x -> $y B"))
    x == 0 && y != 0 && (global zlost += 1)
end
nz = count(==(0), reduce(vcat, collect(values(a))))
println("pins compare: $(length(a)) pin lines, $(eq + up + down) measurements: $eq equal, $up higher, $down lower; $nz at 0 B in the first, $zlost of them no longer 0 B")
```

### `control.jl`

```julia
# Positive control for corpus_compare.jl: one value moved by one ulp and one moved row index.
using Serialization
a = deserialize(ARGS[1]); k = sort!(collect(keys(a)))
x = a[k[1]]; v = copy(x[5]); v[1] = nextfloat(v[1]); a[k[1]] = (x[1], x[2], x[3], x[4], v)
y = a[k[2]]; r = copy(y[4]); r[1] += 1; a[k[2]] = (y[1], y[2], y[3], r, y[5])
serialize(ARGS[2], a)
```

### `clamp.jl`

```julia
# Off-grid clamp padding (sites neither prototype converts: ast/common.jl `_reevaluated_shift`,
# operators/shift.jl): an Inf trial coefficient at each boundary node of x, unrestricted test side.
using Bramble, Random
using Bramble: D₋ₓ, D₊ₓ, Mₓ
Random.seed!(20261014)
W = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 9), (false, false)))
for (name, f) in (("D₊ₓ(D₋ₓ(v))", v -> D₊ₓ(D₋ₓ(v))), ("Mₓ(D₋ₓ(v))", v -> Mₓ(D₋ₓ(v))), ("D₋ₓ(D₋ₓ(v))", v -> D₋ₓ(D₋ₓ(v))))
    nan = 0
    for k in (1, 7, 8, 14)
        gi = element(W); gi .= 1 .+ rand(length(gi)); gi[k] = Inf
        A = Matrix(assemble(form(W, W, (u, v) -> innerₕ(gi * u, f(v)))))
        nan += count(isnan, A)
    end
    println("clamp $name: NaN entries over Inf at boundary nodes 1, 7, 8, 14: $nan")
end
```

### `run_tree.sh`

```bash
#!/bin/bash
# run_tree.sh <tree> <tag>: every recorded figure of #795's note for one tree, 1 thread.
tree=$1; tag=$2
cd "$tree" || exit 1
J=".claude/scripts/julia_slot.sh 1 julia --threads=1 --startup-file=no --project=test"
$J /tmp/p795/repro.jl > /tmp/p795/repro-$tag.txt 2>&1
$J /tmp/p795/genuine.jl > /tmp/p795/genuine-$tag.txt 2>&1
$J /tmp/p795/corpus.jl /tmp/p795/corpus-$tag.jls > /tmp/p795/corpus-$tag.txt 2>&1
$J /tmp/p795/pins.jl "$tree" /tmp/p795/pins-$tag.tsv > /tmp/p795/pins-$tag.txt 2>&1
echo "run_tree $tag done"
```

## Measured costs

Method (P1b). Each figure is measured on the three trees of the table above, alone on the
machine (no other agent ran), at `--threads=4`, the default `-O` and `--startup-file=no`,
every process through `julia_slot.sh 4`. The first-call figures come from
`benchmark/multiterm_ttfx.jl` (C0's script, #797), run with `--launches 1` once per tree and
round. There were five rounds, alternating the trees (base, option 1, option 2, base, ...), each launch
a set of fresh child processes, so every first-call figure is a fresh process. Rows:
`elasticity` (the 3D elasticity form on a 5^3 mesh), `vector_laplacian` and `scalar_2d`
(the no-regression row). The warm figures come from `/tmp/p1b/warm.jl`, one process per
tree and round, three rounds alternating the trees, each figure the minimum over repeated
`assemble` calls (2000 for the 7 by 9 reproduction form, 50 for 201 by 201, 200 for the
5^3 elasticity form, 20 for 9^3). The reproduction form is the issue's,
`innerₕ(gi * u, D₊ₓ(D₋ₓ(restrict_to(:l, v))))` with finite `gi`. Every table gives the
median over the back-to-back runs, the spread (min to max), and the ratio to base of each
run pair (option over base of the same round): its median and its range.

Machine state. **Every figure is under load.** The preflight failed on load and the
user ruled "if load stays above 2, run it anyway and label the figures as under load,
relying on back-to-back alternating ratios". The output of `check_power_load.sh` was
`Power: AC Power`, `Load:  3.71 (threshold: < 4.0 on 8 logical cores)`. The output of
`gate_preflight.sh --timing` was `FAIL load 1-min load 3.71 on 8 cores (above 2)`,
`OK power on AC`, `OK onedrive not running`, `FAIL julia other julia processes: 45040`
(that process had exited when checked a minute later), `OK cpu nothing above 50% CPU`.
The 1-minute load average read 2.69 to 4.03 across the run's uptime lines. The ratios of
back-to-back pairs are the evidence; the absolute seconds are not.

### First call, fresh process, seconds (5 rounds)

| Row, figure | Tree | Median | Spread (min-max) | Ratio to base, median | Ratio range |
| --- | --- | --- | --- | --- | --- |
| elasticity, `form` | base | 4.693 | 4.587-4.754 | 1 | 1 |
| | option 1 | 4.830 | 4.696-4.891 | 1.023 | 1.016-1.060 |
| | option 2 | 4.741 | 4.591-4.805 | 1.009 | 0.981-1.047 |
| elasticity, `assemble` | base | 16.99 | 16.44-17.23 | 1 | 1 |
| | option 1 | 16.82 | 16.40-17.28 | 0.994 | 0.952-1.027 |
| | option 2 | 16.41 | 16.36-17.15 | 0.981 | 0.954-1.006 |
| vector_laplacian, `form` | base | 0.0713 | 0.0656-0.0814 | 1 | 1 |
| | option 1 | 0.0658 | 0.0649-0.0697 | 0.931 | 0.825-1.002 |
| | option 2 | 0.0672 | 0.0655-0.0725 | 0.939 | 0.891-1.077 |
| vector_laplacian, `assemble` | base | 2.865 | 2.710-3.030 | 1 | 1 |
| | option 1 | 2.774 | 2.727-2.809 | 0.968 | 0.919-1.020 |
| | option 2 | 2.784 | 2.735-2.866 | 0.968 | 0.905-1.029 |
| scalar_2d, `form` | base | 0.00634 | 0.00610-0.00722 | 1 | 1 |
| | option 1 | 0.00637 | 0.00615-0.00645 | 0.996 | 0.884-1.039 |
| | option 2 | 0.00630 | 0.00619-0.00645 | 0.993 | 0.857-1.041 |
| scalar_2d, `assemble` | base | 0.003661 | 0.003314-0.003859 | 1 | 1 |
| | option 1 | 0.003555 | 0.003425-0.003690 | 0.971 | 0.956-1.081 |
| | option 2 | 0.003493 | 0.003418-0.003578 | 0.977 | 0.905-1.071 |

Every option median lies inside base's spread, and every ratio range straddles 1 except
option 1's `elasticity form` (1.016 to 1.060, medians 1.023: the largest first-call figure
that moves, about 0.14 s of 4.7 s, with its spread overlapping base's). Neither prototype
changes compile cost beyond base's spread; under load this is not a win or a loss.

### Warm `assemble`, microseconds (3 rounds)

| Form | Tree | Median | Spread (min-max) | Ratio to base, median | Ratio range |
| --- | --- | --- | --- | --- | --- |
| reproduction, 7 by 9 (timer resolution about 0.04 us) | base | 1.208 | 1.208-1.208 | 1 | 1 |
| | option 1 | 1.167 | 1.166-1.208 | 0.966 | 0.965-1.000 |
| | option 2 | 1.250 | 1.208-1.250 | 1.035 | 1.000-1.035 |
| reproduction, 201 by 201 | base | 421.3 | 420.5-425.7 | 1 | 1 |
| | option 1 | 445.7 | 445.4-447.3 | 1.059 | 1.047-1.062 |
| | option 2 | 453.7 | 452.5-455.0 | 1.077 | 1.063-1.082 |
| elasticity, 5^3 | base | 178.2 | 176.2-178.6 | 1 | 1 |
| | option 1 | 178.1 | 178.1-178.6 | 1.000 | 0.999-1.011 |
| | option 2 | 178.6 | 178.1-178.8 | 1.002 | 0.997-1.015 |
| elasticity, 9^3 | base | 1046 | 1004-1069 | 1 | 1 |
| | option 1 | 1034 | 1032-1036 | 0.990 | 0.967-1.028 |
| | option 2 | 1015 | 1001-1082 | 0.957 | 0.950-1.077 |

The elasticity form, which has no padded entry, is unchanged (ratios within 1.00 and the
base spread). The 201 by 201 reproduction form, which does carry padding, is slower by 5.9 %
(option 1) and 7.7 % (option 2), and both option spreads lie wholly above base's spread in
all three back-to-back pairs (ratios 1.047 to 1.082). That cost is real within this
measurement, though taken under load. The 7 by 9 figure is at the timer's resolution and
says nothing. The in-process `warm assemble` column of the TTFX script (`1e-5` s rows) is
equally at that resolution and is not tabulated.

### Recommendation

Under the rule of the plan (an option passes if every value constraint of the section
"Summary of the values" holds and its first-call and warm figures sit within base's
spread), neither option passes, whatever the timings: both fail the refill-bytes
constraint, and the 201 by 201 warm figure fails the spread test as well.

Recommendation: do not change the padding

- Option 1 fails the refill constraint ("stays at 0 B, or at its recorded cap"): 14
  measurements of forms that already box at base are above their recorded cap
  (bilinear.jl lines 869, 1468 and twelve forms of 1543; 16 higher in all). Its 201 by 201
  warm `assemble` of the padded reproduction form is 5.9 % slower, outside base's spread.
- Option 2 fails the same constraint with 13 measurements above their cap (bilinear.jl
  lines 982, 1468 and ten forms of 1543, and linear_restriction.jl line 181), plus the
  `_zero_entries` layout test at linear_restriction.jl line 185, which the flag changes by
  design. Its 201 by 201 warm `assemble` is 7.7 % slower, outside base's spread.

Both fix the reproduction, keep a genuine `0 * Inf` as `NaN`, and leave the finite corpus
bit-identical; those are not what rejects them.
