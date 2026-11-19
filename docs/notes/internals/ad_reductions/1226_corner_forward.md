# #1226: the wrong forward gradient through a traction corner

Reduction of gpena/Bramble.jl#1226, where Enzyme's forward mode gives a wrong gradient with no error. This note takes the issue's `β` case, `constrained(S × S, ...)` on a composite of 2D ghost spaces, with `β * u(i) + trac(u, i)` on `:xmin` and `:ymin`, `:xmax` and `:ymax` clamped, and the loss `sum(abs2, nonzeros(E)) + sum(abs2, d)` of `closure`. The reproducer is `1226_corner_forward.jl`, next to this note. It runs one case a process, and its header lists them all. The tree is 12d859e1 on `v4-2-9`, with S2.1's component index in the type and the `@noinline` on `_composite_relation` (0eb08850).

Enzyme: 0.13.220, 2026-11-19, macbook (Apple M2, 16 GB, macOS arm64, Julia 1.13.4, default `-O2`, `--startup-file=no`), 1 thread

The columns are case, mode, batch width, status, and relative error against ForwardDiff (largest absolute difference over largest absolute reference). Every case calls `DifferentiationInterface.gradient` with `AutoEnzyme(; mode = set_runtime_activity(Forward), function_annotation = Const)`, as the issue does. `nora` and `relsnora` drop the runtime activity.

Case: beta | fwd | width 1 | wrong | 0.04586450960566223

Case: beta2 | fwd | width 2 | wrong | 0.045864509605662275

Case: betag | fwd | width 2 | ok | 1.1366034503872306e-16

Case: nora | fwd | width 1 | fails | NaN

Case: inline | fwd | width 1 | ok | 1.1366034503872306e-16

Case: nocorner | fwd | width 1 | ok | 0.0

Case: rels | fwd | width 1 | wrong | 0.07499999999999996

Case: relsinl | fwd | width 1 | ok | 0.0

Case: relsnc | fwd | width 1 | ok | 0.0

Case: relsnora | fwd | width 1 | wrong | 0.07499999999999996

Case: relsfld | fwd | width 1 | ok | 0.0

Verdict: construct: src/assembly/ghost_relations.jl:329, `_composite_relations` returns the parsed relations as a tuple from a call, and Enzyme's forward mode gives the datum of the fourth relation (`:ymin`, `0.3`) a tangent equal to its value once that relation is passed on to another call (`_assemble_relation`, line 846)

## What the runs say

- `beta` reads the issue's number to the last digit: Enzyme gives 2.0431802469135802, ForwardDiff 1.9535802469135803. The difference, 0.0896, is `2 Σ dₖ ḋₖ` over seven entries of `d` whose tangent Enzyme sets to their own value.
- Those seven entries are the `:ymin` ghosts of component 2. A scratch run of `closure` with a vector output (`ForwardWithPrimal`, every entry of `E` and `d` against ForwardDiff) found every entry of `E` right, and these seven entries of `d` with tangent 0.08 where ForwardDiff gives 0. Each tangent equals the entry's value to the last bit. The primal under `autodiff` matched a plain call bit for bit.
- Going back along `d`: the same seven rows of `b` from `_relation_rows` carry the wrong tangent, and so does the load `bF` that `_assemble_relation` returns for that relation (its seven facet nodes, 0.025 to 0.05, tangent equal to value). That stays true with the padded space built outside the differentiated function. The load is `inner_Γ(data, v)`, linear in the datum. Its tangent equals its value when the datum's tangent is the datum, `0.3` where it should be 0.
- `rels` shows this with no space, form or closure built inside the differentiated function: `_composite_relations` parses the four traction relations, and the fourth goes to a `@noinline` function that returns `([β], [datum])`. Enzyme gives 4.3 for the derivative of `β² + datum`, ForwardDiff 4.0. The extra 0.3 is the datum. Read inline in the differentiated function, the datum's tangent comes out 0. Scratch runs pinned down which steps matter. Returning the tuple from a function of its own, as `_composite_relations` does, gives the wrong tangent: a verbatim copy defined in the script gives it, with or without the `check` closure. The same parse, components and `_with_components` written inline in the differentiated function give the right one. So does rebuilding the relation from its fields (`_FacetRelation(r.d, r.s, r.name, r.op, r.data, r.c)`) before the `@noinline` call (`relsfld`).
- `inline` and `relsinl` redefine `_composite_relations` with its body unchanged and `@inline` added. That change alone makes both the issue's case (1.1e-16) and the small case exact.
- `relsnora` is wrong by the same 0.3 without runtime activity, so runtime activity is not the cause. `nora`, the whole case without runtime activity, stops earlier with an `EnzymeRuntimeActivityError` in `_dirichlet_values`, which says nothing about the relations.
- The corner is not the construct. Its code (`_one_sided_corners!`, `_one_sided_weights`, `_corner_entry`) changes rows of `R` and so entries of `E`, which are all right. The tangent is already wrong before any corner code runs. `nocorner` (the `:ymin` relations reading one component each, no one-sided corner) is exact, and so is `relsnc`, `rels` on those four relations. Whether the tangent comes out wrong depends on the types of the relations in the tuple, which the tractions change, not on what the corner does.
- Width: `beta2` adds a second, unused parameter (DifferentiationInterface's `pick_batchsize` gives one batch of 2, so one width-2 call) and is wrong by the same amount in the first entry, with 0.0896 in the second where ForwardDiff gives 0. The issue's width-2 run (`betag`) made the `:ymin` datum the second parameter, so the datum had a tangent of its own. It is exact (1.1e-16), because a datum that is a parameter gets its own tangent and the constant `0.3` does not. The width plays no part.
- Only the fourth relation's datum is affected. The first (`:xmin`, datum 1.0, `β` in its operator too) gets a right tangent. Why the position matters was not found.

## What each lead ruled out

- Primal values under `autodiff`: identical to a plain call (largest difference 0.0). Ruled out a primal miscompile.
- Without `set_runtime_activity`, the whole case fails earlier (`nora`). The small case is wrong by the same amount (`relsnora`). Ruled out runtime activity, and a shadow that is the primal, as the cause.
- The forward rules of `_dense_block_solve` and `_corner_entry` (ext/BrambleEnzymeExt/relations.jl) were not swapped. The wrong tangent is already in `_assemble_relation`'s load, which calls neither, and in `rels`, which reaches none of the relation rules. Ruled out.
- `_one_sided_weights` reading mesh points built inside `f`, and the stores of `_one_sided_corners!`, feed `E`, which is right entry by entry, and the case stays wrong with the space built outside. Ruled out.
- No Bramble-free reproducer was found. Pure-Julia stand-ins gave the right tangent: a `@noinline` builder of a tuple of four or twelve immutable structs `{Int, Int, Symbol, op, Float64, Int}`, with nested operator structs, a singleton scalar and a `Pair` parse as in `_composite_relations`, under the same `DifferentiationInterface` call. Without one the verdict stays `construct`. The wrong tangent is still Enzyme's, since the primal code has no path that turns a datum into its own derivative.

## For the fix

- The site is `_composite_relations`, src/assembly/ghost_relations.jl:329, and the relations it returns to `constrained` (src/assembly/ghost_composite.jl:881). `@inline` on it fixes the issue's case in this script. Rebuilding the relation from its fields before the `@noinline` call fixes the small case (`relsfld`). It was not tried in front of `_assemble_relation` on the whole case.
- Check with `beta`, `beta2` and `rels` of the script, each to 1e-8. The `λ` case of the issue (refused by name since #1180) was not run here.

## Commands run

From the worktree root, one process each, under `julia_slot.sh 1` and `/usr/bin/time -l`:

```text
.claude/scripts/julia_slot.sh 1 /usr/bin/time -l \
    julia --startup-file=no --project=. --threads=1 \
    docs/notes/internals/ad_reductions/1226_corner_forward.jl <case>
```

The whole-case runs (`beta`, `beta2`, `betag`, `nora`, `inline`, `nocorner`) took 260 to 330 s of Enzyme compile and 4.0 to 5.0 GB of peak memory footprint. The `rels` cases took about 9 s and 0.9 GB. The scratch runs behind the second and third points above lived in `/tmp/s31` and are not kept. Their results are the ones quoted. `Project.toml` and `Manifest.toml` were unchanged by the runs.
