# Where a predicate of Bramble infers a type other than `Bool` (v4.1.4, S6)

Answers #913 step 1: which call sites of `form` and `assemble` hold a predicate (a
function with a `Bool` instance and no concrete non-`Bool` one) whose inferred return type is
not `Bool`, for the 3D elasticity form and a 40-term sum. It also records the JET
`report_opt` counts, the baseline for the issue's "do not rise" criterion. This note adds no
`::Bool` (step 2 changes `src/`) and decides nothing on step 2, which is the user's.

## Method and conditions

`benchmark/bool_audit.jl` builds a form once per row, runs `form` and `assemble` through
`Base.invokelatest`, then measures twice.

- JET: `JET.report_opt` on `form(W, W, fun)` and on `assemble(a)`, with
  `target_modules = (Bramble,)`. The counts are the numbers of reports.
- Detector: for every function owned by Bramble or a submodule, walk
  `Base.specializations` of its methods and read the return type of each instance from its
  native `CodeInstance`s (instances inferred but not compiled count). A function is a
  predicate when one instance returns `Bool` and none returns a concrete non-`Bool` type.
  A site is an instance of a predicate whose return type is not `Bool` (`Union{}` instances
  excluded). A site's caller is the method JET's runtime-dispatch reports name as the
  innermost frame; none when JET reports no dispatch to it.
- Rows: `elasticity` (a graded 5x5x5 mesh, `gridspace(Ωₕ)^Val(3)`,
  `2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₘₕ(u), divₘₕ(v))`, 35 summands after
  simplification) and `sum40` (a graded 9x11 scalar space,
  `foldl(+, ntuple(k -> (k + 0.5) * inner₊(∇ₕ(u), ∇ₕ(v)), Val(40)))`, 40 summands).

The detector reads the whole code cache, so two rows in one process mix their instances.
Each row therefore ran in its own `julia` (1 thread, through `julia_slot.sh 1`, one at a
time), and `--out` merged the results into `benchmark/results/bool_audit.toml`:

```
julia --startup-file=no --project=test benchmark/bool_audit.jl --control --rows sum40 \
    --out benchmark/results/bool_audit.toml
julia --startup-file=no --project=test benchmark/bool_audit.jl --rows elasticity \
    --out benchmark/results/bool_audit.toml
```

The file holds `control`, `sum40` and `elasticity`. Julia 1.13.2, default `-O`, commit
`96ff3a4c` on `perf/v4.1.4-measure`. The sites include instances of the package image
(the cache holds those too), so a site is not necessarily reached by the row's form. The
two checks below, made with temporary scripts outside the repository (not kept), separate
the two.

## The positive control

The control module `AuditControl` defines `isp(@nospecialize x) = x > 0`, `isp(x::Int)` and
`isq(x::Int)`, and calls each. The detector must list `isp` and not `isq`. It lists
`AuditControl.isp` (instance `Tuple{typeof(AuditControl.isp), Any}`, inferred `Any`) and
not `isq`: the detector finds a known non-`Bool` predicate and does not flag a plain one.

## JET counts

| row | summands | `form` reports | `assemble` reports |
|---|---|---|---|
| `elasticity` | 35 | 38 | 0 |
| `sum40` | 40 | 5 | 0 |

`assemble` is clean for both forms: every report is in `form`. All 43 reports are runtime
dispatches.

The 38 reports of `elasticity`, by cause:

- `εₕ`/`divₘₕ` return tuples whose element type is not concrete (a `BackwardAverage`
  with free `Dim` and `OpType`, or a `Union` of it and a `BackwardDifference`). Dispatches
  follow on `_DivergenceTerms` (2), `Mₕ(BackwardAverage, Val{1|2|3})` (3), `<=(Int64,
  Integer)` and `_throw_stencil_dim_error(Int64, Integer)` (in `_check_stencil_dim`, called
  from `Mₓ`), and on `inner₊` and `_inner₊_val`, `_product`, `_restrict_by_markers` and
  `_throw_both_sides_interpolated`, with `_throw_invalid_staggered_set` from
  `_check_staggered_set`. Callers: `_inner₊_val`, `_inner`, `_product`,
  `_check_one_interpolated_side`, `_check_stencil_dim`.
- `foldl(+, ...)` in the form's `fun` accumulates an `OperatorAdd` tree whose operand types
  are abstract: 15 dispatches on the `OperatorAdd` constructor, directly or in `+` (frames
  `afoldl > BottomRF > +`), and one on `OperatorScale(::Float64, ::OperatorAdd)`.
- `form` itself: one dispatch on a callee of type `Any`, one on `_bilinear_form`, one on
  `_holds_rank_one` (innermost frame `form`), one on `_holds_functional` (caller
  `_check_bilinear_terms`). The last two are predicates, discussed below.

The 5 reports of `sum40`: the `Any` callee, `_bilinear_form`, `_holds_rank_one`,
`_holds_functional` (caller `_check_bilinear_terms`), and one `OperatorAdd` constructor
dispatch.

## Sites

`elasticity` has 26 sites and `sum40` 21, all 21 also in `elasticity`. The 5 further sites of
`elasticity` are `_is_source_only` (2), `_has_test_interp` (1) and `_has_trial_interp` (2).
Instances are shortened, and the TOML holds the full signatures. A signature is abstract
when an argument is `Any`, `Type`, `DataType`, `Tuple`, a `UnionAll` or an abstract type. A
predicate called on such an argument infers `Any` (or `Union{Missing, Bool}`, from an `all`
or `any` over an `Any` result) whatever the form does.

| function | signature | inferred type | caller | rows |
|---|---|---|---|---|
| `Bramble._is_source_only` | `(BackwardAverage{3, Dim, OpType} where {Dim, OpType<:LazyOp{3}})` | `Any` | none | elasticity |
| `Bramble._is_source_only` | `(BackwardAverage)` | `Any` | none | elasticity |
| `Bramble._has_test_interp` | `(BackwardAverage)` | `Any` | `Bramble._check_one_interpolated_side` | elasticity |
| `Bramble._has_trial_interp` | `(BackwardAverage{3, Dim, OpType} where {Dim, OpType<:LazyOp{3}})` | `Any` | `Bramble._check_one_interpolated_side` | elasticity |
| `Bramble._has_trial_interp` | `(BackwardAverage)` | `Any` | `Bramble._check_one_interpolated_side` | elasticity |
| `Bramble._shares_one_mesh` | `(Tuple)` | `Union{Missing, Bool}` | none | both |
| `Bramble._ghost_kron_reads_absent` | `(Tuple, GhostMesh{2, MeshnD{2, …}, …})` | `Union{Missing, Bool}` | none | both |
| `Bramble._kron_is_fresh` | `(KroneckerLinearOperator)` | `Any` | none | both |
| `Bramble._kron_is_fresh` | `(AbstractMeshType, Int64)` | `Any` | none | both |
| `Bramble._marks_only_face` | `(AbstractMeshType{D}, BitVector, Symbol) where D` | `Any` | none | both |
| `Bramble._mf_gathers` | `(OperatorAdd)` | `Any` | none | both |
| `Bramble._mf_gathers` | `(LazyOp)` | `Any` | none | both |
| `Bramble._near_edge` | `(GhostMesh{1, M, P, CI} where {…}, CartesianIndex{1}, Int64)` | `Any` | none | both |
| `Bramble._pairable_type` | `(Any)` | `Union{Missing, Bool}` | none | both |
| `Bramble._pairable_type` | `(Type)` | `Union{Missing, Bool}` | none | both |
| `Bramble._pairable_type` | `(DataType)` | `Union{Missing, Bool}` | none | both |
| `Bramble._peelable` | `(NTuple{D, AbstractUnitRange{Int64}}, NTuple{D, Tuple{Int64, Int64}}) where D` | `Any` | none | both |
| `Bramble._same_mesh` | `(Any, Mesh1D{…})` | `Any` | none | both |
| `Bramble._same_mesh` | `(Any, MeshnD{2, …})` | `Any` | none | both |
| `Bramble._storage_collapsed` | `(Any, Any, Any)` | `Any` | none | both |
| `Bramble._type_holds` | `(Any, Type{RankOneTerm})` | `Any` | none | both |
| `Bramble._type_holds` | `(Type, Type{RankOneTerm})` | `Any` | none | both |
| `Bramble._type_holds` | `(Any, Type{LinearProduct})` | `Any` | none | both |
| `Bramble._type_holds` | `(Type, Type{LinearProduct})` | `Any` | none | both |
| `Bramble._type_holds` | `(Type, Type{Union{BilinearProduct, LinearProduct}})` | `Any` | none | both |
| `Bramble._type_holds` | `(Any, Type{Union{BilinearProduct, LinearProduct}})` | `Any` | none | both |

All 26 signatures are abstract. That alone does not make a site a leak: the next two
sections say which ones the forms reach.

### The 21 common sites: abstract instances of the image

A detector run right after `using Bramble`, before any mesh or form exists, lists the same
21 sites. Neither row adds one to them, and no JET report of either form names one of
the 21 as a callee. They are abstract instances present in the package image (the
`precompile` statements and workloads, for instance `precompile(_type_holds_restriction,
(Any,))`; which statement produced each was not traced), not call sites of the audited
forms. A predicate called on `Any` legitimately infers `Any`, so none is a non-`Bool`
leak to a caller:

- `_type_holds` and `_pairable_type` take a `Type`, and run when a `@generated` body
  or a pairing plan is built, not when `form` or `assemble` run.
- `_storage_collapsed(Any, Any, Any)` and `_same_mesh(Any, …)` are called with `Any`
  arguments on mesh construction and build-time checks only; `convert(eltype(backend), a)`
  on an `Any` gives `Any`.
- `_shares_one_mesh`, `_ghost_kron_reads_absent`, `_kron_is_fresh`, `_marks_only_face`,
  `_near_edge`, `_peelable` and `_mf_gathers` are instances for an abstract `Tuple`, mesh,
  range or `LazyOp`. The forms reach these functions with concrete arguments and their
  concrete instances return `Bool`.

### The 5 elasticity-only sites: a real leak, at form build time

Elasticity adds `_is_source_only`, `_has_test_interp` and `_has_trial_interp` on a
`BackwardAverage` with free parameters. The cause is upstream: `εₕ` and `divₘₕ` build
tuples of non-concrete element type (the dispatches on `_DivergenceTerms` and `Mₕ` above),
so `inner₊` reaches `_product` through a runtime dispatch (JET reports `_product` and
`_inner₊_val` with `BackwardAverage` arguments). Inside it, `_product` calls `_is_source_only`
on the left operand, and `_check_one_interpolated_side` calls `_has_trial_interp` and
`_has_test_interp`. JET names that caller for the last two; for `_is_source_only` the
caller is `_product`, itself a dispatch target, so JET reports no separate dispatch. These
five instances return `Any` to a caller that then branches on them: a real leak to
the caller, in `form`, once per call. `assemble` shows no report (0) for either row.

### JET-only call sites: two predicates dispatched at run time

`_holds_rank_one` and `_holds_functional` are not detector sites: their instances return
`Bool`. In both rows JET reports a runtime dispatch to each, with a `UnionAll` operand
type (the `OperatorAdd` tree that `fun` returned, widened), from `form` and from
`_check_bilinear_terms`. The result of the call at the call site then infers `Any`. These
are the one place where the issue's `::Bool` assertion at a call site applies as stated.

## #853

None of the sites traces to the 32-entry limit of #853. The dispatches on `_holds_*`
and on `_bilinear_form` are in `sum40` and also in a 30-term sum. A temporary scan of
`form` with `Val(N)`, N = 8, 16, 24, 30, 31, 32, 33, 34, 40, gave the same six dispatch
reports at every N (the same five as `sum40` plus `_throw_form_dimension`, which the
script's `assemble` before JET removes). The set does not change across 32, so the long
tuple is not its cause: the cause is the abstract term type `fun` returns, at 8 terms as
at 40. Whether the 15 `OperatorAdd` constructor dispatches of `elasticity` (35 summands,
past 32) owe anything to the limit was not tested; no scan over the elasticity size was
run.

## Result

The list is not empty: 26 sites for `elasticity` and 21 for `sum40`, so #913's empty-list
criterion does not close the issue on this evidence. Of the 26, 21 are abstract instances
the forms do not reach, and 5 are real, all in `elasticity`'s `form` (`_is_source_only`,
`_has_test_interp`, `_has_trial_interp`), plus the two JET-only predicates `_holds_rank_one`
and `_holds_functional`. Whether those seven call sites warrant step 2 is for the user to
decide. The baseline for "do not rise" is `form` 38 and `assemble` 0 for `elasticity`,
`form` 5 and `assemble` 0 for `sum40`, as the script reports them (one `julia` per row,
`assemble` run before JET).

## After #939

Retaken on `perf/v4.1.4-followups` at `6ee42fb0` (S1, which unwraps a builder's sum under a
restriction), Julia 1.13.2, 1 thread, one `julia` for both rows:

```
julia --startup-file=no --project=test benchmark/bool_audit.jl --control \
    --rows elasticity,sum40 --out benchmark/results/v4.1.4_bool_audit_after939.toml
```

The result is `benchmark/results/v4.1.4_bool_audit_after939.toml`. The earlier table above
stays as measured. A second run of each row alone in its own process (a temporary script
outside the repository, not kept) gave the same counts and the same site list, so sharing
the process did not change them. The positive control finds `AuditControl.isp` and not `isq`,
as before.

### JET counts against the original table

| row | summands | `form` reports, before | `form` reports, after | `assemble` reports, before | `assemble` reports, after |
|---|---|---|---|---|---|
| `elasticity` | 35 | 38 | 0 | 0 | 0 |
| `sum40` | 40 | 5 | 7 | 0 | 0 |

`elasticity` is clean: the 38 reports all traced to the non-concrete tuples of `εₕ` and
`divₘₕ` and to the abstract `OperatorAdd` tree of `foldl`, and none is left.

`sum40` rises by 2, so its count does not fall. The 7 reports are the 5 of the original
table (the `Any` callee, `_bilinear_form`, `_holds_rank_one` from `form`, `_holds_functional`
from `_check_bilinear_terms`, one `OperatorAdd` constructor dispatch in `+`) and two new
ones: runtime dispatches to `_normalisable` and to `_normalised`, both with `Bramble._normalise_scalars`
as innermost frame. `_normalise_scalars` came with #912 (`ab551efe`, 2026-10-25), after the
original table's commit `96ff3a4c` (2026-10-24), so the two are not an effect of #939. The
cause is the one the original table gives for `sum40`: the 40-term `foldl` returns an abstract
`OperatorAdd` type, so the AST that reaches `_normalise_scalars` infers `Any`.

### Sites still leaking

The detector lists 20 sites for each row (original: 26 for `elasticity`, 21 for `sum40`),
with no caller named by any JET report. A detector run right after `using Bramble`, before
any mesh or form exists, lists the same 20. All 20 are abstract-signature instances of the
package image, classed apart as before: `_shares_one_mesh`, `_ghost_kron_reads_absent`,
`_kron_is_fresh` (2), `_marks_only_face`, `_mf_gathers` (2), `_pairable_type` (3), `_peelable`,
`_same_mesh` (2), `_storage_collapsed` and `_type_holds` (6). The original 21 had one more,
`_near_edge(GhostMesh{1, …}, CartesianIndex{1}, Int64)`, which no longer appears at load. Which
change removed it was not traced (97 commits separate `96ff3a4c` from this head).

The 5 elasticity-only sites (`_is_source_only` (2), `_has_test_interp` (1),
`_has_trial_interp` (2)) are gone. They traced to the non-concrete AST, as the original
table said, and the elasticity form is now concrete.

So no detector site leaks from either form: every site left is an image instance.

### Predicate call sites JET still reports

`elasticity` has none. `sum40` has three predicate call sites reached through a runtime
dispatch. None is a detector site, since the instances that return are `Bool` or generated.

| function | file | inferred type at the call | caller |
|---|---|---|---|
| `Bramble._holds_rank_one` | `src/assembly/bilinear.jl:357` | `Any` (dispatch on the abstract `OperatorAdd` union) | `form` |
| `Bramble._holds_functional` | `src/assembly/rank_one.jl:206` | `Any` | `Bramble._check_bilinear_terms` |
| `Bramble._normalisable` | `src/assembly/bilinear.jl:485` | `Any` (argument `ast::Any`) | `Bramble._normalise_scalars` |

The inferred type is that of the call expression, read from the JET report's runtime
dispatch (the callee has an abstract argument); it was not read per call from the
inference result. All three come from the abstract term type that the 40-term `foldl`
returns, not from #939 or from a concrete AST.

### Site list for S4.2

Site list after #939: for `elasticity` empty; for `sum40` the three JET-only predicate call
sites `_holds_rank_one` (`src/assembly/bilinear.jl:357`), `_holds_functional`
(`src/assembly/rank_one.jl:206`) and `_normalisable` (`src/assembly/bilinear.jl:485`), with
no detector site leaking from either row.
