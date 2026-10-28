# Why a captured coefficient made `form` infer a `Union` (v4.1.4, #939)

Answers #939: which call is the first one that infers a non-concrete type when the
coefficient of a vector-calculus form is captured by the form's closure, why a literal
coefficient hides it, what S1 changed, and what the JET counts say before and after. The
note is the source for the comment on the issue.

## Method and conditions

- Base is head 6ee42fb0 with S1's three commits (8e1ff09e, 26ca885d, 6ee42fb0) reverted, in a
  scratch worktree; `src` and `test` differ from head only in S1's files. The base TOML's
  `commit` field names that scratch commit (bf1259bf), which is not in the repository.
  Head is 6ee42fb0. Julia 1.13.2, one thread.
- `benchmark/bool_audit.jl --rows elasticity --out <toml>` ran once in each tree. The row is
  the 3D elasticity form on a graded 5x5x5 mesh, `2μ * inner₊(εₕ(u), εₕ(v)) + λ *
  inner₊(divₘₕ(u), divₘₕ(v))` with `μ, λ = 1.0, 2.0` captured from the building function's
  scope. The results are `benchmark/results/v4.1.4_captured_bool_audit_base.toml` and
  `..._head.toml`.
- The inference figures come from `Base.return_types(form, (typeof(W), typeof(W),
  typeof(f)))`: concrete means one return type and that type concrete. The two TOMLs hold
  them under `[form_inference]` (`elasticity_2d`, `elasticity_3d`, `divmh_3d`), measured by
  a script run through `julia_slot.sh 1` in each tree, on the graded meshes (9, 11) (2D) and
  (5, 5, 5) (3D), with `μ, λ` locals of the measuring function. A coefficient that is a global
  makes the closure read an untyped binding, which is a different question.
- The diagnosis scripts `.claude/plans/v4-1-4-followups-scratch/diag939.jl`, `diag939b.jl`
  and `diag939c.jl` ran on both trees. `diag939b.jl` walks the AST of the captured form and
  `diag939c.jl` names the deepest node whose `simplify_ast` is non-concrete.
- Counts, not timing: the machine's load does not change them.

## The first non-concrete call

`simplify_ast(::OperatorScale)` on the root node of the resolved AST. On base, in
`diag939c.jl`:

    D=2 (3) deepest non-concrete simplify_ast node: ("OperatorScale", 0, "Union of 2")
    D=3 (3) deepest non-concrete simplify_ast node: ("OperatorScale", 0, "Union of 3")

The depth 0 is the root: every node below it infers concretely, so the leak is in this one
method, a `Union` of 2 trees in 2D and of 3 in 3D. Above it, `simplify_ast(resolved)` and
`_bilinear_form(W, W, raw)` inherit the `Union` and `form` returns a `Union` of
`BilinearForm`s (in 3D, a `BilinearForm` whose type parameter is not concrete).
`εₕ(u)`, `inner₊(εₕ u, εₕ v)`, `Float64 * strain sum` and `resolve_ast` are all concrete,
so no builder is at fault.

## Why a literal hides it

`simplify_ast(::OperatorScale)` distributes a scalar over a sum only when the sum mixes
components, and #529 made that decision by value: `_mixes_components` compares the component
indices the two operands of the sum name, and an index is a runtime `Int` in
`IndexedTrialFunction`. The result type depends on the answer (a distributed tree has
another shape than a kept one), so inference can only close the `Union` if it knows the
indices.

With a literal coefficient, `2.0 * inner₊(εₕ(u), εₕ(v))` is a closure whose body has the
component indices as constants from the builders; constant propagation folds the gate and the
method infers one tree. A captured `μ` is a field of the closure object, so the value reaches
`simplify_ast` through a value the compiler does not propagate into the builders' output,
and the indices stay unknown. On base the literal form infers concretely and the captured one
does not, with the same matrix (the planning probe pinned equal bits).

## What S1 changed

The strain and `divₘₕ` builders know each term's (trial, test) component pair. They now wrap
their sum in `_ComponentSum`, whose type carries the pairs. `simplify_ast` reads that
wrapper by dispatch (`simplify_ast(::OperatorScale{D, C, <:_ComponentSum})`) and answers the
`_mixes_components` gate from the pairs and the simplified sum's type with a generated method
(`_scale_labelled_sum`, `_labelled_mixes`), so the tree is the one the literal-index path
builds. A user-written sum carries no pairs and keeps #529's value gate, and
`simplify_ast(::_ComponentSum)` unwraps the sum anywhere else. No public name changed. The
design choices are in the comment above `simplify_ast(::OperatorScale)` in
`src/ast/simplifier.jl`, which also updates the #529 comment there.

## Inference of `form`, base and head

`form(W, W, f)` with the coefficients captured. Same meshes and closures on both trees.

| form | base | head 6ee42fb0 |
|---|---|---|
| elasticity 2D, `2μ * inner₊(εₕ, εₕ) + λ * inner₊(divₘₕ, divₘₕ)` | non-concrete (a `Union` of 2) | concrete |
| elasticity 3D | non-concrete | concrete |
| `μ * inner₊(εₕ(u), εₕ(v))`, 2D and 3D (`fcs` in `diag939.jl`) | non-concrete | concrete |
| `divₘₕ` only, 2D | concrete (see below) | concrete |
| `divₘₕ` only, 3D (`diag939c.jl`) | non-concrete | concrete |
| scalar Laplacian, 2D and 3D | concrete | concrete |
| 2D vector Laplacian | concrete | concrete |
| 3D vector Laplacian | non-concrete | non-concrete here; concrete since #958 (see The 3D gradient (#958)) |
| literal coefficients (`flit`) | concrete | concrete |

The `[form_inference]` tables read, base then head: `elasticity_2d` false then true,
`elasticity_3d` false then true, and `divmh_3d` false then true. The 2D `divₘₕ`-only form
is concrete on both trees, so it has no key (see Surprising).

### The `divₘₕ` and gradient cases

- `divₘₕ`-only in 2D already infers concretely on base: the planning notes saw the 2D
  gradient and `divₘₕ` forms inferred through constant propagation alone. The 3D `divₘₕ`
  form is the one that leaked, and head fixes it.
- The 3D vector Laplacian, `μ * inner₊(∇ₕ(u), ∇ₕ(v))`, stayed a `Union` on head 6ee42fb0. The
  gradient `inner₊` works on plain tuples (`foldl(+, map(inner₊, left, right))`), so nothing
  in the types says which component each tuple entry is. A type-level answer needed a tagged
  container from `∇ₕ(composite)`, a public API change, which the user ruled out of S1. The
  user later took that route (2026-10-28) and #958 built it: see "The 3D gradient (#958)"
  below. At 6ee42fb0 the form was concrete in 2D by constant propagation and a `Union` in 3D.
- What also stays a `Union`: a scale meeting an already distributed sum (`2 * (μ * X)`,
  `-(μ * X)`, `μ * (X + Y)` over two builders), an `Int` coefficient (#240), and a
  user-written sum that mixes components under one shared runtime coefficient. These take
  the value gate as before and `test/form/simplifier.jl` keeps pinning them.

## JET `report_opt`, base and head

`JET.report_opt` on `form(W, W, fun)` and on `assemble(a)`, `target_modules = (Bramble,)`,
for the elasticity row (3D):

| | base | head 6ee42fb0 |
|---|---|---|
| `jet_form_reports` | 0 | 0 |
| `jet_assemble_reports` | 0 | 0 |
| `summands` of `form.ast` | 35 | 35 |
| `sites` (predicates not inferring `Bool`) | 20 | 20 |

The counts do not move, and they cannot show this change. `report_opt` reports run-time
dispatch (and the like) inside the target module. A `Union` return type of `form` is not
such a call, so it adds no report, and base already reads 0.

The change is one of return-type concreteness, and the inference table above is where it
shows. The JET figures are recorded as a regression catch: head does not rise above base.
The 20 sites (`_type_holds`, `_pairable_type`, `_mf_gathers` and the rest) are the #913 list,
which S4.1 retakes on head.

## Surprising

- JET reads 0 and 0 on both sides, so the planned criterion (head below base) was
  unfalsifiable and the plan's PROBE was replaced by the `[form_inference]` table.
- The 2D `divₘₕ`-only form infers concretely on base through constant propagation, so it
  cannot show the change. The third `[form_inference]` key is the 3D `divₘₕ`-only form, which
  is non-concrete on base. The elasticity forms are non-concrete on base in 2D and 3D.

## The 3D gradient (#958)

The 3D vector Laplacian was the one vector-calculus form left a `Union` above. The user ruled
on 2026-10-28 to take the route the paragraph above set aside: `∇ₕ` and `∇₊ₕ` of a composite
trial or test function return a tagged container. This section records that change and
supersedes the "out of scope" row of the table above.

### The container and why

`Bramble._GradientRows{D, N, T}` holds the N-tuple of D-tuples that `∇ₕ(composite)` returned
before. It is internal, but it is the return type of a public operator. `inner₊` of two of
them forms the same sum as before (`foldl(+, map(inner₊, rows, rows))`, same order) and holds
it in a `_ComponentSum` whose pairs are `(c, c)` for row `c`. Row `c` of `∇ₕ(u)` is `∇ₕ(u(c))`,
so every trial leaf in it is `u(c)` and every test leaf is `v(c)`: the labels are true by
construction, and `_scale_labelled_sum` (S1 of #939) gives the tree the literal path gives.
Only a composite whose components are all scalar leaves is wrapped; a nested composite, `D == 1`
and a scalar space return what they did.

The builder is the right place because `∇ₕ(u)` is the one call that knows the rows. The
alternative, a label on the plain-tuple `inner₊`, cannot know them. The planning findings on
main 8ce4b1b0 (scripts `e958.jl`, `e958b.jl`, `e958c.jl`) show it is unsafe for a hand-built
tuple, and these assemble on main:

- A repeated component, `∇ₕ((u(1), u(1), u(2)))`, and a permuted one,
  `∇ₕ((u(2), u(1), u(3)))`, are `Union`s that assemble. A positional label would distribute
  rows that share a component, so the bits would move.
- One row that mixes components within it is concrete on main; two such rows are a `Union`
  that assembles (nnz 1375). A positional label would treat a mixed row as unroutable, and a
  runtime guard against it would throw on tuples that assemble today.

Two routes with no API change were tried by redefining methods in-process and both stayed a
`Union` with unchanged bits: an unrolled row sum in place of `foldl(+, map(...))`, and
`Base.@constprop :aggressive` on `simplify_ast(::OperatorScale)`. A hand-built tuple, a
`∇ₕ(components(u))` and a container meeting a plain tuple keep the plain-tuple path, so they
keep their bits and their inference (a `Union` in 3D for the repeated and mixed cases).

### Inference of `form`, before and after

`form(W, W, f)` with `μ` captured, graded meshes (9, 11) and (5, 5, 5), Julia 1.13.

| form | main 8ce4b1b0 | after S1 of #958 |
|---|---|---|
| `μ * inner₊(∇ₕ(u), ∇ₕ(v))`, 3D | `Union` | concrete |
| `μ * inner₊(∇₊ₕ(u), ∇₊ₕ(v))`, 3D | `Union` | concrete |
| the same two, 2D | concrete | concrete |
| scalar Laplacian, 2D and 3D | concrete | concrete |
| `restrict_to(:interior, μ * inner₊(∇ₕ(u), ∇ₕ(v)))`, outer, 2D | not recorded | concrete |
| the same outer `restrict_to`, 3D | `Union` | still a `Union` |
| `μ * restrict_to(:interior, inner₊(...))`, 3D | concrete | concrete |
| `μ * inner₊(...; markers = (:boundary,))`, 3D | concrete | concrete |

The `restrict_to` rows were not asked for by #958. The outer one, concrete in 2D after the
change, stays a `Union` in 3D; its bits are unchanged. The two rows with the scale outside
the `restrict_to` or with `markers` were concrete on main and must stay so.

### Bits and types are unchanged

Assembled matrices are bit-identical to main in every row above, in 2D and 3D, and captured
equals literal. The check used a non-dyadic `0.7`, so a distributed tree and a kept tree would
hash differently; the captured and literal `form` types are equal (`===`). The hand-built
repeated and mixed-row tuples assemble to main's bits as well. The only change in the
before and after comparison of type, bits, matrix-free and Kronecker assembly is `conc=false`
becoming `conc=true`.

### What no longer holds on a composite gradient

A composite `∇ₕ(u)` or `∇₊ₕ(u)` is no longer `isa Tuple`, a public break in a patch release
that the user accepted. These work as before: `∇ₕ(u)[c][d]`, `length`, iteration, `map` and
broadcast over it, and `Tuple(∇ₕ(u))`. These need `Tuple(∇ₕ(u))`: `==`, `isequal`, `hash`,
`eltype`, `reverse`, range indexing, `eachindex` and `keys`, `show`, `convert(Tuple, .)`,
and a `map` that mixes the container with a plain tuple (it returns a `Vector`). No caller in
`src/`, `ext/`, `test/` or `docs/` used any of them. `c * ∇ₕ(u)` and `∇ₕ(u) + ∇ₕ(w)` on a
composite were a `MethodError` on main and stay one.
