# Two-level form AST: go or no-go (v4.1.4, S7, #918)

Answers #918: whether a type-erased form-time layer, lowered once to the typed `LazyOp`
tree, is worth building. A prototype of it was measured against the branch in the same
runs. Every open question of the issue has a section below, and `## Decision` gives the
verdict.

The base is `perf/v4.1.4-followups` at `63ecf980`, which already holds S1 (#939,
`_ComponentSum`), S3 (#912), S4 (`::Bool` asserts), S5 and S6.1 (#917). S6.1 was reverted afterwards (`docs/notes/source-closures-v4.1.4.md`, `## Outcome: reverted`), so the figures below that depend on the erased source type were measured with it and a head without it infers those forms as base does. The prototype is an
uncommitted change on top of `63ecf980`, run from detached scratch worktrees and kept as a
patch (see the end of `## Decision`; not on this branch). The figure from the time of #914
comes from a third worktree at `9c34251c`, the main commit #914 and #915 measured.

Conditions for every run: Julia 1.13.2 at the default `-O`, Apple M2 (8 cores, 16 GiB) on AC
power, OneDrive not running, and no other Julia process. Each timing `julia` ran through
`julia_slot.sh 4` with `--threads=1` (as #914's runs did), in a fresh process per launch,
base and prototype interleaved, never in parallel. The count, correctness and test runs
used 1 or 2 threads.
`gate_preflight.sh --timing` passed before each timing batch, at a 1-minute load of 1.69 to
1.96. The load rose to 2.6 during the timing A/B and to 3.5 during the precompile A/B. That
is the runs' own load, and both trees took it alike, since the launches were interleaved.
The scratch scripts named below (`measure.jl`, `phases.jl`, `check.jl`, `rt.jl`, `q3.jl`,
`canon.jl`, `precomp.jl`, `enzyme_probe.jl`, `moshi_probe.jl`) live in
`~/bramble-scratch/v414/s7`, outside the repository.

### The prototype

`src/ast/erased.jl` (new) and a 31-line change to `_bilinear_form`
(`src/assembly/bilinear.jl`):

- One concrete node, `_ENode` (`kind::UInt8`, `D::Int`, `kids::Vector{_ENode}`,
  `data::Any`), a plain struct. Its kinds are leaf, zero, sum, scale, grid-function scale,
  bilinear product, shift and wrapper.
- `_erase` turns the resolved typed tree into `_ENode`s. `_esimplify` applies
  `simplify_ast`'s rules (`src/ast/simplifier.jl`) by value: zeros, nested scales, scalar
  lift, component distribution, and wrapper and shift distribution. `_elower` rebuilds the
  typed tree through the runtime's generic constructor (`jl_new_structv`), so no method
  compiles per node type, and it folds `_lower_signs` in. `_echeck_form_meshes` asks
  `_reads_meshes` before any typed mesh check. Every function is
  `Base.@nospecializeinfer` with `@nospecialize` arguments, so each compiles once, into the
  package image through the existing workload.
- `_bilinear_form` erases, simplifies and lowers. It then calls `_bilinear_form_finish`
  through `Base.inferencebarrier`, and that function runs the passes left typed:
  `_normalise_scalars`, `_split_restricted_products`, `_check_space_components` and the
  `BilinearForm` constructor. A tree holding a node the layer does not cover (a
  `LinearProduct`, a Dirac source) takes the old path, `_bilinear_form_typed`, behind a
  barrier. Linear forms (`form(Wₕ, f)`) are not covered.

Two prototype toggles, `_E_VALUE_RULES` (for "simplify_ast trade-offs") and
`_E_NEW_BY_CONSTRUCTOR` (for the Enzyme probe), were added after the timing and precompile
runs, which therefore measured the prototype without them. Both are `Ref`s that are off by
default, and `check.jl` gave 28 of 28 again with them in. The file is 333 lines with the
toggles.

The lowered tree is the base's tree. `check.jl` built 28 bilinear forms on graded 2D and 3D
meshes through both paths in one process. All 28 take the erased path, every AST type is
`===` the base's, and every assembled matrix is equal bit for bit. The 28 cover scalar,
vector-Laplacian and 3D elasticity forms (literal and captured coefficients), `0 * A`,
`1 * A`, `0.0 * A`, `Rational` and `Ref` scales, `-A` and `A - B`, `D₋ₓ` and `S₊ₓ` over a
component-mixing sum, a scale and a grid-function scale over such a sum, and a runtime
component index. Because the tree is the same, `assemble` compiles the same kernels on both
trees.

## Share of first-call time

The issue proposed this redesign when #914 and #915 (`docs/notes/tracy-phases-v4.1.4.md`,
`docs/notes/trace-compile-v4.1.4.md`) put `simplify_ast` at 3.7 s of a 4.9 to 5.1 s first
elasticity `form`. That is no longer the state of the branch. The same harness
(`measure.jl time elasticity`) on `9c34251c` and on `63ecf980` gives:

| commit | first `form`, s | first `assemble`, s | launches |
|---|---|---|---:|
| `9c34251c` (#914's main) | 4.99, 4.64, 4.59 | 17.75, 16.55, 16.49 | 3 |
| `63ecf980` (base) | 1.82, 1.94, 1.83, 1.83, 1.82 | 16.46 to 16.62 | 5 |

The `9c34251c` launches 2 and 3 ran under a load of 3.1 to 4.1, left by that tree's own
precompile. The ratio between the trees is still clear. The commits between them took
about 3 s off the first elasticity `form`. Which commit did it was not measured. The
likely one is #939, which made the captured-coefficient form infer concretely, but that is
a guess.

On the base, `phases.jl` times each pass of `form` in order, on the graded 5x5x5 mesh with
3 components, 2 launches each (ms):

| pass | elasticity | vector Laplacian |
|---|---|---|
| closure (the builders) | 862, 872 | 40.5, 40.7 |
| `resolve_ast` | 49.1, 48.9 | 8.8, 9.2 |
| `simplify_ast` | 986, 853 | 9.4, 8.0 |
| `_lower_signs` | 29.5, 28.9 | 5.0, 4.7 |
| `_check_form_meshes` | 18.9, 18.9 | 1.5, 1.5 |
| `BilinearForm` constructor | 21.0, 19.8 | 21.6, 21.4 |

So what the erased layer can remove at most is `simplify_ast`, the sign lowering and the
mesh check: about 0.9 to 1.0 s for elasticity. The first `form` plus `assemble` takes
18.3 s, so that is 5 to 6% of it. The builders inside the closure (0.86 s) are erased only
if every operator builder emits erased nodes, which this prototype does not do. With them,
the ceiling is the whole first `form`, 1.83 s, or 10%. For the vector Laplacian the passes
are about 25 ms of a 2.86 s first call (under 1%). For the scalar forms the whole `form` is
6 to 26 ms. `assemble` holds the rest: 16.5 s of 18.3 s for elasticity (90%) and 2.77 s of
2.86 s for the vector Laplacian (97%), in the per-summand-type kernels of #916
(`docs/notes/replay-kernels-v4.1.4.md`). The design leaves `assemble` unchanged, and
"Interaction with #912 and #917" below covers the one part of the issue that could change
it, canonical order.

### Base against prototype, one run

`measure.jl time ROW`, 3 launches per tree, interleaved (base, proto, then proto, base, and
so on). Rows: `scalar_2d` (9x9, the precompile workload's form), `scalar_multiterm` (9x11,
four terms, not in the workload), `vector_laplacian` and `elasticity` (5x5x5, 3
components, `μ` and `λ` captured). All meshes are graded. Each cell is the median, then
min to max. Warm figures come from the same child on a larger graded mesh of the same types
(201x201, 201x221 and 17x17x17): the minimum of 20 `assemble!` calls and of 20 `form` calls.
The `form` result is kept with `Base.donotdelete`.

| row | tree | first `form`, ms | first `assemble`, ms | warm `assemble!`, µs | warm `form`, µs |
|---|---|---|---|---|---|
| `scalar_2d` | base | 6.06 (5.97 to 6.16) | 3.34 (3.29 to 3.50) | 201.9 (197.0 to 202.1) | 0.0 |
| `scalar_2d` | proto | 2.78 (2.77 to 2.79) | 0.043 (0.040 to 0.044) | 207.8 (204.3 to 208.1) | 7.96 |
| `scalar_multiterm` | base | 26.2 (26.0 to 26.2) | 303.9 (303.4 to 305.1) | 426.0 (417.4 to 426.6) | 0.0 |
| `scalar_multiterm` | proto | 32.1 (32.1 to 32.4) | 299.3 (298.3 to 301.9) | 426.2 (424.8 to 430.7) | 20.2 |
| `vector_laplacian` | base | 91.1 (90.5 to 94.0) | 2765 (2763 to 2877) | 307.0 (306.9 to 307.2) | 0.0 |
| `vector_laplacian` | proto | 160.4 (160.1 to 162.2) | 2768 (2763 to 2770) | 306.6 (306.4 to 307.6) | 39.4 |
| `elasticity` | base | 1828 (1822 to 1939) | 16492 (16462 to 16505) | 6147 (6145 to 6159) | 0.04 |
| `elasticity` | proto | 1118 (1118 to 1131) | 16397 (16253 to 16655) | 6145 (6122 to 6311) | 1170 |

- Elasticity: the first `form` falls by 0.71 s (39%), outside both spreads. The first
  `assemble` stays inside the spreads. The first call in total goes from 18.32 s to
  17.52 s (-4.4%).
- Vector Laplacian: the first `form` rises by 69 ms (+76%), outside both spreads.
  `scalar_multiterm`'s rises by 6 ms. Its first `assemble` is 4.6 ms (1.5%) lower, also
  outside the spreads.
- `scalar_2d`: both trees take under 10 ms. Its first `assemble` (3.3 ms on base) is too
  small to read anything into.
- Warm `assemble!`, by S6.2's rule (every base row is at or above 50 µs, so it is gated:
  within base's spread or +3%, whichever is larger). `scalar_multiterm`,
  `vector_laplacian` and `elasticity` lie within base's spread. `scalar_2d` does not:
  207.8 µs against at most 202.1 µs, +2.9% on the median, so it passes on the +3% clause
  only. Both trees allocate 0 B on every row. The lowered tree is the same.
- Warm `form`: base's takes 0.04 µs or less, which prints as 0. That `form` folds to a
  constant at compile time is a guess that fits. The prototype pays to build `_ENode`s and
  dispatch at run time: 8 µs to 1.17 ms per call, 1.17 ms for elasticity.

Where the prototype's +69 ms on the vector Laplacian goes. In `phases.jl` the erased passes
themselves take 1.2 ms (vector Laplacian) and 2.3 ms (elasticity) at first call, since they
are in the package image. `--trace-compile-timing` lists `_bilinear_form_finish` at 67 ms
and `_bilinear_form` at 63 ms as roots for the vector Laplacian (162 ms for elasticity),
although every typed pass inside them had been called, and so compiled, the line before.
Removing `@constprop :aggressive` from the prototype's `_bilinear_form` left the 65 ms as it
was. The prototype infers fewer instances and still takes longer, which points to code
generation, but the trace does not separate inference from code generation. The
explanation is a guess: on base the typed passes fold into constants inside `form`, and
across the barrier the passes left typed (`resolve_ast`, the checks,
`_normalise_scalars`, `_split_restricted_products`) compile as real code. If so, removing
the cost means erasing those passes too, `_split_restricted_products` (#727) among them.

### Method instances

`measure.jl snoop ROW`: distinct Bramble `MethodInstance`s in `@snoop_inference` of the
first `form` and of the first `assemble`, one launch per tree. S5 and S6.2 found these
counts equal across launches.

| row | base `form` | base `assemble` | base total | proto `form` | proto `assemble` | proto total |
|---|---:|---:|---:|---:|---:|---:|
| `scalar_2d` | 1 | 0 | 1 | 1 | 0 | 1 |
| `scalar_multiterm` | 84 | 108 | 192 | 60 | 110 | 170 |
| `vector_laplacian` | 193 | 285 | 478 | 109 | 310 | 419 |
| `elasticity` | 1860 | 1745 | 3605 | 841 | 2001 | 2842 |

`scalar_multiterm`'s 192 matches S5.3's count for that row. Elasticity's does not: S5.3
read 3719 instances and a 16.457 s first `assemble` at `f674c310` with 4 threads, where
this run reads 3605 and 16.49 s at `63ecf980` with 1 thread. The difference was not looked
into. The thread count, which selects the threaded kernels, is a guess. On the prototype, `assemble` infers
2 to 256 more instances: the routing queries (`trial_component_or_nothing` and the rest)
that `_check_form_meshes` used to compile now compile in `assemble`. The sum falls by 11 to
21%.

### Precompile time and package image

`precomp.jl` times `Base.compilecache` into a fresh first depot (with `~/.julia` behind it
for the dependencies), 2 runs per tree, interleaved base, proto, proto, base:

| tree | precompile, s | `.ji`, bytes | `.dylib`, bytes |
|---|---|---:|---|
| base | 100.00, 100.13 | 2717379 | 237092928, 237160784 |
| proto | 97.24, 97.66 | 2732213 | 237551184, 237502496 |

Precompiling takes 2.6 s less (-2.6%), since the workload's bilinear forms no longer
specialise `simplify_ast`. The image grows by 0.4 MB (+0.2%). It keeps both paths, so that
size is an upper bound. The `.dylib` moved by 0.07 MB between two runs of the same tree.

## Return type of form

Lowering builds the tree's type from run-time data, so the prototype's `form` infers as
`Any`. `rt.jl` gives `Base.return_types(form, …)` for the four rows: one concrete
`BilinearForm` on base, `Any` on the prototype, and `@inferred form(…)` fails for all four.

Five test files, three that pin `form`'s return type and two of ghost forms, run with
`runtests_subset.jl` at `--optimize=1` (2 threads) on both trees:

| file | base | proto |
|---|---|---|
| `test/form/simplifier.jl` | 320 pass | 299 pass, 16 fail, 1 error |
| `test/quality/type_stability.jl` | 183 pass | 156 pass, 14 fail, 13 error |
| `test/form/bilinear.jl` | 1674 pass | 1673 pass, 1 fail |
| `test/ghost/allocations.jl` | 186 pass | 186 pass |
| `test/ghost/composite.jl` | 198 pass | 198 pass |

The 45 failing assertions, itemised from the logs:

- 37 return-type pins: `isconcretetype(Base.return_types(form, …))`, `@inferred form(…)`
  and `_infers(…)`. There are 17 in `simplifier.jl` (17 lines; the error is an `@inferred`
  in the testset "πₕ forms: the stored tree") and 20 in `type_stability.jl` (lines 85,
  185, 202 to 205, 289, 302 and 315 to 318).
- 7 JET `@test_opt` reports in `type_stability.jl`, at lines 78 (`_diffusion(Wₕ)`, 3 runs),
  86 (`semidiscretize`) and 149 (ForwardDiff assembly, `_fd_bilinear`, 3 runs). They are
  the run-time dispatch the barrier adds inside `form`.
- 1 allocation pin: `bilinear.jl:1975`, `_form_bytes(Wₕ) < 8 * n`, which the `_ENode`s
  exceed.

The 0 B ghost assembly pins hold.

A typed wrapper restores inference but not the 0 B pins. `rt.jl` defines
`EForm{W, V}` (the spaces typed, the lowered form in an `Any` field), an `assemble` that
asserts `::SparseMatrixCSC{promote_type(eltype(W), eltype(V)), Int}`, and an `assemble!`
that makes one dynamic call:

- `@inferred eform(…)` and `@inferred easm(…)` pass for all four rows, on both trees.
- `assemble!` through the barrier allocates 48 B a call, where the direct call allocates
  0 B. 48 B is the size of a boxed `SparseMatrixCSC`: the matrix, an immutable struct, is
  boxed to cross the dynamic call. That `assemble!` takes a pre-boxed argument to avoid it
  is a guess, not tried.
- The barrier costs nothing measurable when warm: 166.25 µs against 166.5 µs for
  elasticity, 0.46 µs against 0.42 µs for `scalar_2d` (minimum of 200).
- Every other reader of `BilinearForm` (Kronecker, matrix-free, ghost, SciML, sparse AD)
  would have to forward through the wrapper.

Enzyme (`enzyme_probe.jl`, Enzyme 0.13.216 stacked from a scratch environment): reverse
mode of `θ -> sum(abs2, assemble(form(W, W, (u, v) -> θ * inner₊(∇ₕ(u), ∇ₕ(v)) +
innerₕ(u, v))).nzval)` on the graded 9x11 mesh.

- Base: plain `Reverse` raises `EnzymeRuntimeActivityError`, the quirk
  `test/space/autodiff_heavy.jl` documents. With `set_runtime_activity` it gives
  7421.507869, against 7421.507868 by central differences.
- Prototype: both modes fail with `AssertionError: Illegal generic_setup, expected all
  arguments to by jlvaluet`, at the `jl_new_structv` call. A second build with that call
  deleted from `_enew`, so that every node is built through its constructor, failed both
  modes with `IllegalTypeAnalysisException`, the error the issue recalls for a `Union`.
  So the prototype's layer cannot be differentiated under Enzyme 0.13.216 with respect to
  a coefficient that enters `form`, in either construction. Whether some other erased
  construction would pass was not established. A wrapper does not help, since the layer
  runs inside the differentiated function.

## simplify_ast trade-offs

The prototype keeps base's rules, so its results equal base's bit for bit, and the
"off" control in `q3.jl` reads 0 ULP. `q3.jl` then turns on `_E_VALUE_RULES`, which
decides by value what `bramble-form` §1 and §6 decide by type: a `Float64` zero or one
collapses, and `c1 * A + c2 * A` becomes `(c1 + c2) * A`. It compares each assembled
matrix against base's, on a graded 41x43 mesh, in one process:

- Like terms, 500 random pairs `c1, c2` in (-5, 5) with `A = inner₊(∇ₕ(u), ∇ₕ(v))`. All 500
  differ from base. For same-sign pairs (256) the maximum entry difference has median
  3 ULP and maximum 4. For opposite-sign pairs (244) the median is 4 ULP and the maximum
  509, the cancellation in `c1 + c2`. Three terms, `0.1 A + 0.2 A + 0.3 A`: 3 ULP.
- `1.0 * A` unwrapped to `A`: up to 2 ULP. Without the scale node the stencil takes another
  evaluation path, although multiplying by `1.0` is exact.
- `0.0 * innerₕ(g * u, v) + innerₕ(u, v)` with one `NaN` in `g`: base's matrix holds 1
  `NaN` entry and the value rule's holds none. The term is dropped, as the module comment
  of `simplifier.jl` says of the `Integer` collapse.
- The tree's type follows the value. For `θ * inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v)`, the
  values `θ = 2.0, 0.0, 1.0` give 3 AST types, and the first `assemble` of each new one
  compiles: 0.096 s for 2.0, 0.092 s for 1.0. The tree for 0.0 equals one built earlier
  in the process. So a `θ` sweep that passes through 0 or 1 compiles new kernels, where
  base compiles once.
- Warm: the two like terms combined into one take 11.1 µs against 21.0 µs as two sweeps.
  Both are under 50 µs, so this is reported only.

So yes: deciding by value changes results. Like terms move by 3 to 4 ULP (up to 509 with
cancellation), an exact `1.0` by 2 ULP, and a `NaN` vanishes. The type of the kernels then
depends on coefficient values. On an erased tree the old reason against these rules, a
`Union` return type (#240), is gone, but `form` is already `Any` there. The bit-for-bit
guarantee in the `simplifier.jl` module comment is what would have to give.

## Interaction with #912 and #917

#912 is not superseded. `_elower` builds `OperatorScale{D, typeof(c), …}`
(`erased.jl`, `_elower`), so the scalar's type reaches the kernels exactly as before, and
the prototype still has to run `_normalise_scalars` after lowering. `snoop_instances.jl
--rows scale_float64,scale_float32 --launches 1` (counts only, load 2.9 to 3.3): base 6
and 31 fresh instances (26 added for `2.0f0` after `2.0`), proto 6 and 27 (22 added).
The 4 fewer are the typed `simplify_ast`'s. A two-level design could normalise on the
erased tree, but the decision is the same `_normalisable` rule and has to be made somewhere.

#917 is not superseded either, and the prototype does not reach it. `closures_fresh` is a
linear form, and the prototype covers bilinear forms only. S6.2
(`docs/notes/source-closures-v4.1.4.md`) already erased the source type in the AST (177 to
70 second-build instances). What it left, 70 instances, is three per closure in `innerₕ`,
`_as_source`, `SourceFunction` and the sampling in `_lower_sources` (`Rₕ!`, `_RₕKernel`,
`project!`). These compile because the closure must be called to be sampled, so erasing the
tree cannot remove them.

Canonical order (part 3 of the proposal). `canon.jl` on base, one process, a 3-term scalar
form on the graded 9x11 mesh:

| build | instances | `form` + `assemble`, s |
|---|---:|---:|
| first order, first time | 252 | 0.494 |
| the same terms reversed | 72 | 0.112 |
| first order, a new closure | 1 | 0.007 |

Sorting the terms would save the 72 instances and 0.1 s of the reversed build, and only in
a session that writes one form twice in two orders. Even then it changes the matrix: the
two orders differ by up to 4.4e-16 (summation order). Within one form it merges nothing.
The elasticity form has 24 distinct summand types among its 35 summands
(`docs/notes/trace-compile-v4.1.4.md`), and the kernels are per summand type.

## Moshi.jl or a plain struct

`moshi_probe.jl` in a scratch environment (`Pkg.add("Moshi")` offline from the depot):

- What the dependency costs: Moshi 0.3.7, which brings ExproniconLite 0.10.14 and Jieko
  0.2.1, three packages Bramble does not have today. They precompile in 5.4 s together
  (3.5 s, 1.0 s, 0.9 s), and `using Moshi` takes 0.024 to 0.025 s (0.22 s the first time
  after install).
- What it buys: a match that leaves out a variant is accepted at definition and throws
  `matching non-exhaustive` when it is reached. On this version, exhaustiveness is a
  run-time error, the same as the plain struct's fall-through branch. A 2047-node walk runs
  in 7.25 µs warm against 13.2 µs for the plain struct with a `Vector` of children. The
  erased passes run once per `form`, so the gap does not matter there.
- The plain struct, `_ENode` in `erased.jl`, was enough for the prototype. No pass needed
  more than a `kind` byte and an `if` chain.

So a plain struct. If the layer were ever built, a `kind` enum with an exhaustive
`@nospecialize` dispatch table does what Moshi's match does here, without a dependency.

## Extensions and device paths

What reads the typed tree before lowering is only `form`'s own front end: the closure (the
operator builders), `_validate_form_expression`, `_holds_rank_one`, `_check_bilinear_terms`,
`resolve_ast` and `_resolve_surfaces_for_space` (`src/assembly/bilinear.jl:349` to `:369`).
The rank-one split also does, through `rank_one.jl:265`, which calls `_bilinear_form` on the
rest of the form. Every other reader takes the stored tree, which is the lowered one:

- Ghost forms build `form(padded_space(Wₕ), padded_space(Vₕ), f)` and then wrap the
  finished `BilinearForm` (`ghost_assembly.jl:256` to `:268`, `ghost_composite.jl:215` to
  `:221`). On the prototype, `test/ghost/allocations.jl` (186) and `test/ghost/composite.jl`
  (198) pass, the 0 B pins included.
- Extensions: a grep for `.ast`, `resolve_form_ast`, `BilinearForm{`, `LazyOp` and
  `simplify_ast` in `ext/` finds one tree reader, `BrambleKroneckerExt`
  (`_kron_leaves(resolve_form_ast(…))`, line 1651), and it reads the stored tree.
  `BrambleSciMLExt` and `BrambleSparseADExt` take a `BilinearForm`. `BrambleMetalExt`,
  `BrambleKernelAbstractionsExt` and `BramblePolyesterExt` name no AST node and call no
  `local_stencil`. The tree reaches their kernels from `src/`, after `form`, as the stored
  typed tree, which device code needs typed.
- In `src/`, the `.ast` field is read by `matrix_free.jl` (14 sites), `linear.jl` (9),
  `ghost_assembly.jl` (8), `assemble_add.jl` (8), `bilinear_pattern.jl` (6),
  `bilinear.jl` (6), `expression.jl` (3), `bilinear_execution.jl` (3),
  `jacobian_pattern.jl` (2), `gmg_smoothers.jl` (1) and `bilinear_traversal.jl` (1). All of
  them run after `form`.

So the layer would touch no extension and no device path. The cost it imposes lands on the
callers of `form`, through its return type, not on the code after it. Metal and
KernelAbstractions were not run, since the `gpu` group is off until v4.11.0. The claim rests
on the grep and on the lowered tree being `===` in type to the base's.

## Decision

**No-go.** At `63ecf980` the first elasticity `form` takes 1.83 s of an 18.3 s first call.
The passes this design erases are about 1.0 s of it (5 to 6%), and 90% of the first call is
in `assemble`'s per-summand-type kernels, which the design leaves alone. The prototype
removes 0.71 s from elasticity (first call 18.32 s to 17.52 s, -4.4%) and 763 of 3605
instances. But it adds 69 ms to the vector Laplacian's first `form` and 6 ms to
`scalar_multiterm`'s, and 8 µs to 1.17 ms to every `form` call once warm. It also makes
`form` infer as `Any`: 37 return-type pins, 7 JET dispatch pins and 1 allocation pin
fail. A typed wrapper
restores inference but costs 48 B per `assemble!`. And Enzyme can no longer differentiate
a coefficient through `form`, where base gives the finite-difference derivative.

What made the issue worth asking has mostly gone. Between `9c34251c` and `63ecf980` the
first elasticity `form` fell from 4.6 to 5.0 s to 1.8 s (3 and 5 launches) without any
erasure. Of the parts of the proposal:

- Part 1 (erased passes) saves 0.7 s on the largest form at the costs above.
- Part 2 (one lowering barrier) is where the return type is lost.
- Part 3 (canonical order) helps only a form written twice in two orders, and it changes
  results by up to 4.4e-16.

The remaining first-call cost is the kernels of #916.

Nothing is proposed for a milestone. For the issue, a closing comment that cites this note
and the figures above. What would reopen it: a form-time share of a first call above about
20%, measured as here, on a form users write. Erasing only `simplify_ast` and keeping `form`
typed is not possible by erasure, since lowering from run-time data is what makes the type
unknown. A type-level route does exist, the one #939 took (`_scale_labelled_sum`, a
generated body that decides from types and keeps `form` concrete). It is not erasure, and
it was not prototyped here.

The prototype is not on this branch. Its worktrees are removed; it is kept as the patch
`~/bramble-scratch/v414/s7/proto_final.patch` (against `63ecf980`, outside the repository),
next to the scripts and raw outputs.
