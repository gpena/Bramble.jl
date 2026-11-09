```@meta
CurrentModule = Bramble
CollapsedDocStrings = false
```

# AD coverage: every public feature under each kept backend and policy

[`ad_backend_evaluation.md`](ad_backend_evaluation.md) chose the AD backends, and
[`ad_di_evaluation.md`](ad_di_evaluation.md) chose to call them through
DifferentiationInterface (DI). This page records what the kept backends differentiate. Each
row is one public feature's smallest differentiable call, and each cell is what that call
did on this branch.

The cells are pinned by `test/space/ad_coverage.jl`. It builds every call on a uniform and
on a seeded non-uniform mesh, runs it through DI under `CpuSerial`, `CpuThreaded` and
`CpuPolyester`, and compares the gradient with ForwardDiff's on the same policy. ForwardDiff
is itself compared with a central difference, and every component of that reference is
clearly non-zero, so a backend that drops a parameter cannot pass. Enzyme runs in child
Julia processes, one per policy and group of cells: an Enzyme abort ends the process that
met it. A cell reads `works`, `open` (an upstream bug with an issue), `excluded` with its reason,
or `not run` with the cost that stopped the run. A cell never reads `works` on a wrong
gradient.

## What was measured

| Item | Value |
|:--|:--|
| Date | 2026-11-03 for the first matrix; the three-summand cells 2026-11-04; the cells changed in v4.2.2 and the compile table 2026-11-07; the ghost load and `Δₕ!` cells and the new `centered_difference!` cell 2026-11-08 and 2026-11-09 |
| Tree | branch `v4-2-2`, with the v4.2.2 AD fixes |
| Machine | Apple M2, Julia 1.13.3, on mains power |
| Enzyme | 0.13.219 for the ghost load, `Δₕ!` and `centered_difference!` cells (2026-11-08); 0.13.218 for the other v4.2.2 cells (2026-11-07), the newest in the depot, which a temporary environment without a manifest loads; 0.13.216 for the 2026-11-03 cells |
| Threads | 4, the test's thread count, so both threaded policies run on 4 threads; the 2026-11-03 figures ran on 2 |
| Problem | 9 x 9 mesh in 2D, 17 nodes in 1D, 5 x 5 x 5 mesh in 3D, three parameters, `Float64` |
| Agreement | within 1e-8 relative to ForwardDiff on the same policy |

The `ghost_data` cell compares with a central difference where ForwardDiff cannot run
(rel. 1e-5).

## Kept backends

ForwardDiff, through DI, and Enzyme, called directly and through DI. PolyesterForwardDiff,
ReverseDiff and Mooncake are dropped by the AD backend evaluation
(`ad_backend_evaluation.md`).

## The matrix

ForwardDiff has one column: every cell is the same under the three policies, because a
`Dual` is a plain value and each thread writes its own entries. Enzyme has one column per
policy. Each Enzyme cell holds both modes, reverse then forward, unless it says otherwise.

| Feature | Call | ForwardDiff | Enzyme `CpuSerial` | Enzyme `CpuThreaded` | Enzyme `CpuPolyester` |
|:--|:--|:--|:--|:--|:--|
| mesh | `D₋ₓ(Rₕ(…))` on `mesh(…, [0, b] × [0, 1])`, gradient in `b` | works | works | works | works |
| mesh | `assemble` of a form on that mesh, gradient in `b` | works | works | not run | not run |
| operator | `D₋ₓ`, `Mₓ` and `Mᵧ`, `Δₕ`, `innerₕ` with `normₕ` | works | works | works | works |
| operator | `εₕ` on a composite space | works | works | works | works |
| operator | `D₋ₓ` and `D₋₂` on a grid function, 3D | works | works | not run | not run |
| operator | `M₂` on a grid function, 3D | works | forward works, reverse open: `EnzymeNoTypeError` (`memset`, gpena/Bramble.jl#1155) | not run | not run |
| interpolation | `Rₕ`, `avgₕ` | works | works | works | works |
| assemble | `assemble` of a linear form | works | works | works | works |
| assemble | `assemble` of `inner₊(κ∇U, ∇V)` | works | works | works | works |
| assemble | the same with `Mₕ(κ)` | works | works | works | works |
| assemble | `assemble(a, l; dirichlet)`, 1D | works | works | works | works |
| assemble | the same in 2D, non-uniform mesh | works | works | works | works |
| assemble | three summands, the third scaled by a constant | works | reverse works, forward open (#1124) | reverse works, forward open (#1124) | reverse works, forward open (#1124) |
| assemble | the linear form `innerₕ(u, v)`, 3D | works | works | not run | not run |
| assemble | `inner₊(κ∇U, ∇V)`, 3D | works | works | not run | not run |
| assemble | the same with `Mₕ(κ)`, 3D | works | works with `Enzyme.API.maxtypeoffset!(1024)`, see the limits | not run | not run |
| pde_solve | `pde_solve(A, F)` with `A` from a 1D `assemble` of `κ` | works (the `Float64` factor of the values, solved per partial) | works | works | works |
| semidiscretize | `semidiscretize_rhs(sd)(du, u, p, t)` | works | works | works | works |
| matrix_free | `mul!` with `matrix_free_operator` | works | works | works | works |
| kronecker | `mul!` with `kronecker_operator` | works | works | works | works |
| ghost | `assemble(a, l)` on a constrained ghost space, parameters scale the form's coefficients, `Δₕ` in the form | works | reverse works, forward not run | not run | not run |
| ghost | the same with `inner₊` and `innerₕ` terms only | works | reverse works, forward not run | not run | not run |
| ghost | the load of `assemble(a, l)` alone, parameters in the linear form | works | works | works | works |
| ghost | relation data that depend on the parameter | works | excluded: fails (measured outside this test, see the limits) | excluded: fails (measured outside this test) | excluded: fails (measured outside this test) |
| ghost | a Robin coefficient that depends on the parameter | works | not run | not run | not run |
| ghost | `Δₕ!` on a `GhostVector` | works | works | works | works |
| ghost | `centered_difference!` on a `GhostVector` | works | works | works | works |
| sparse Jacobian | `AutoSparse` with `ast_sparsity_detector`, 1D diffusion residual | works | works | works | works |

Three more facts about the table:

- ForwardDiff and Enzyme agree on every `works` cell to 1e-8 relative or better. The worst
  cell, the mesh endpoint, agrees to 5e-11. The rest agree to 1e-15 and 1e-16, and the
  cells measured on 2026-11-07 (the 2D Dirichlet, the ghost and three-summand reverse cells)
  agree to 6e-16 or better.
- The 3D cells run on a `5 x 5 x 5` mesh in the `threed` group, under `CpuSerial` only. The
  averaging operator `M₂` on a grid function and `Mₕ(κ)` in a form are in the limits below.
- The ghost cells run on a non-uniform mesh only, since the ghost test helper builds those.
- The ghost load, `Δₕ!` and `centered_difference!` cells differentiate in both modes under
  all three policies. Enzyme reverse of the load under the threaded policies was
  gpena/Bramble.jl#1157, `Δₕ!` reverse under `CpuPolyester` was gpena/Bramble.jl#1127, and
  forward of the load, in the layout the test uses (a closure over the form), crashed until
  gpena/Bramble.jl#1159; all three are fixed in v4.2.2. `centered_difference!` failed under
  `CpuSerial` on 2026-11-03 and passes now; the likely cause is the ghost facet band
  rewrite it predates (not bisected). The
  measurements are in `ad_reductions/1157_ghost_load.md` and `ad_reductions/1158_ghost_stencil.md`
  (gpena/Bramble.jl#1158).

## Enzyme under the threaded policies

Enzyme differentiates `CpuSerial`, `CpuThreaded` and `CpuPolyester`, and the derivative runs
threaded too. It never enters a `Threads.@threads` or `@batch` body: each engine hook has
its own forward and reverse rule in `ext/BrambleEnzymeExt.jl`, and the rule runs the
derivative threaded itself. The evidence is `test/ext/enzyme_threaded_ext.jl` (196
passing in its last full run; its three-summand reverse cases, added after, pass in their
own run under both threaded policies), `test/ext/enzyme_polyester_hooks.jl` (66 passing)
and the table above, where every feature
that ran under `CpuSerial` also ran under both threaded policies, except the cells marked
`not run`.

Inside a user's own threaded region, a `Threads.@threads` loop of the caller's, the
derivative runs its bands in order, as the primal does: a `:static` loop cannot start there.
The nested reverse and forward modes of `test/ext/enzyme_policy_cases.jl` pin this.

## DI settings Enzyme needs

Measured on `difference` (`D₋ₓ`) and on a closure that fills a captured buffer, `CpuSerial`,
non-uniform mesh. The reference of the buffer cell is a central difference, since ForwardDiff
cannot write a `Dual` into a `Float64` buffer.

| `AutoEnzyme` | Mode | Result |
|:--|:--|:--|
| `AutoEnzyme()` | reverse and forward | throws `EnzymeMutabilityException` |
| `mode = set_runtime_activity(…)` | reverse and forward | throws `EnzymeMutabilityException` |
| `mode = set_runtime_activity(…), function_annotation = Enzyme.Const` | reverse and forward | works |

For a closure that writes the parameters into a captured buffer, which the differentiated
function then reads:

| `function_annotation` | Mode | Result |
|:--|:--|:--|
| `Enzyme.Const` | reverse and forward | returns a zero gradient, no error |
| `Enzyme.Duplicated` | reverse and forward | works (1.9e-10 against the central difference) |

So: `AutoEnzyme(; mode = set_runtime_activity(Reverse), function_annotation = Enzyme.Const)`
for an ordinary closure, and `function_annotation = Enzyme.Duplicated` for a closure that
captures a buffer the input is written into. The plain call is an error, never a wrong
value. The `Const` annotation on a buffer-writing closure is a wrong value: Enzyme treats
the captured buffer as constant, so its derivative is dropped. The forward mode is
`set_runtime_activity(Forward)`.

## Sparse Jacobians

`AutoSparse(AutoEnzyme(; mode = set_runtime_activity(Forward), function_annotation =
Enzyme.Const); sparsity_detector = Bramble.ast_sparsity_detector(a, U -> Mₕ(U)),
coloring_algorithm = GreedyColoringAlgorithm())` returns the Jacobian of the 1D diffusion
residual `A(u) u - F` that ForwardDiff's dense Jacobian does, to 1e-15, under the three
policies and on both meshes. Reverse mode works too. The detector never runs the residual,
so a sparse cell's verdict is its inner backend's.

## Accepted limits

- **`idxs` must be unique.** A sweep over repeated indices has no defined reverse.
- **A closure that reaches `v` only through a global gets no gradient.** Per-slab Enzyme
  compilations treat a global as constant. `test/ext/enzyme_threaded_ext.jl` pins it with a
  `@test_broken`.
- **Ghost spaces whose boundary data depends on the parameter fail.** Under `CpuSerial`
  Enzyme throws `Illegal calling convention fixup` in reverse and `EnzymeNoDerivativeError`
  in forward (dgetrf), measured on 2026-11-02 by a review of the `CpuThreaded` rules,
  on a ghost space whose boundary data takes the parameter (gpena/Bramble.jl#1153). This
  page's own `ghost_data` cell was started in reverse and stopped after more than 100 minutes
  of compile with no result, so it is recorded from that measurement and is not in the test.
  ForwardDiff differentiates the data, and a Robin coefficient too since v4.2.2.
- **Ghost assembly in Enzyme reverse under `CpuSerial` works.** The cells with `Δₕ` in the form
  and with the parameters in the coefficients differentiate in reverse, in 366 s and 370 s,
  4.8 and 4.9 GB (measured 2026-11-07, before the rows above were set). Their forward mode and
  the threaded policies were not run.
- **Plan geometry is inactive.** The plans that carry the mesh geometry (the replay plans of
  the threaded assembly and the matrix-free plan `_mf_plan`) are inactive for Enzyme
  (`EnzymeRules.inactive` on `_mf_plan` in `ext/BrambleEnzymeExt/threaded.jl`), so a
  coordinate read only through a plan gets no derivative. The `mesh` row that
  assembles in the endpoint `b` works under `CpuSerial` in both modes (6e-10 and 2e-10
  against ForwardDiff); that route was not measured through a matrix-free operator or under
  the threaded policies.
- **3D `Mₕ(κ)` needs a wider type-analysis bound.** `assemble` of `inner₊(Mₕ(κ)∇U, ∇V)` on a
  `5 x 5 x 5` mesh throws `EnzymeNoTypeError` (`Cannot deduce type of copy`) under `CpuSerial`
  with Enzyme's default bound of 512 bytes, in reverse and forward. `Enzyme.API.maxtypeoffset!(1024)`,
  set before the first gradient, makes it differentiate in both modes. Bramble does not set the
  bound: it would change every Enzyme compile in the user's session. The test sets it in the
  `threed` child only. The threaded 3D cells were not run.
- **3D `M₂` on a grid function, reverse.** `EnzymeNoTypeError` (`Cannot deduce type of memset`,
  in `_average_band!`), gpena/Bramble.jl#1155. The forward mode works.
- **Compile memory.** `_batch_mf_bands_coef`, the matrix-free reverse with a coefficient,
  compiles at 5.96 GB and 747 s under `CpuPolyester`, against 2.46 GB and 215 s under
  `CpuSerial`, measured on 2026-11-02 by the `CpuPolyester` matrix-free rule's own memory
  run, and not re-run here.

## Open failures

- Ghost relation data that depend on the parameter, in Enzyme
  ([gpena/Bramble.jl#1153](https://github.com/gpena/Bramble.jl/issues/1153)).
- 3D `M₂` on a grid function, Enzyme reverse: `EnzymeNoTypeError`
  ([gpena/Bramble.jl#1155](https://github.com/gpena/Bramble.jl/issues/1155)).
- Ghost assembly matrix with `Δₕ` ([gpena/Bramble.jl#1152](https://github.com/gpena/Bramble.jl/issues/1152)):
  its cells differentiate in reverse under `CpuSerial` (see the limits). The issue's own
  reproducer was not re-run on 2026-11-07.
- Enzyme forward on a three-summand form: wrong gradient
  ([gpena/Bramble.jl#1124](https://github.com/gpena/Bramble.jl/issues/1124)), see the paragraph below.

## Compile cost of the cells

Measured 2026-11-07 on mains power, the machine otherwise idle, at default `-O`, 4 threads.
Each child is one policy and one group, run alone as the test runs it: Enzyme compiles every
cell of the group on both meshes in both modes, and the wall time includes the ForwardDiff
reference. Peak is the child's peak memory footprint (`/usr/bin/time -l`), the figure the
local 10 GB cap reads; the resident set is lower. Each unchanged child ran twice and the table
gives both runs, wall time first; the children split on 2026-11-07 ran once.

| Group | Cells | `CpuSerial` | `CpuThreaded` | `CpuPolyester` |
|:--|:--|:--|:--|:--|
| dense | mesh, operators, interpolation | 181, 182 s; 3.2 GB | 206, 197 s; 3.4 GB | 198, 201 s; 3.4 GB |
| assemble | the 1D and 2D assemblies, 1D Dirichlet | see below | 533 s; 6.5 GB | 315 s; 4.9 GB |
| three | the three-summand form | see below | 530 s; 6.3 GB | see below |
| boundary | 2D Dirichlet | 181 s; 3.4 GB | 291 s; 4.2 GB | 197 s; 3.5 GB |
| geometry | the mesh endpoint | 132, 132 s; 2.8 GB | 94, 94 s; 2.4 GB | 94, 96 s; 2.4 GB |
| solve | `pde_solve`, `semidiscretize` | 142, 143 s; 3.0 GB | 178, 178 s; 3.3 GB | 168, 170 s; 3.2 GB |
| operators | matrix-free, Kronecker | 112, 112 s; 2.7 GB | 216, 217 s; 3.7 GB | 205, 207 s; 3.6 GB |
| ghost_fast | the ghost cells | 702, 709 s; 5.4 GB | 114, 114 s; 2.6 GB | 120, 121 s; 2.6 GB |
| threed | the 3D cells | 2514, 2394 s (1235, 1203 s CPU); 8.8 GB | no cell | no cell |
| sparse Jacobian | the residual, four calls | 130 s; 3.1 GB | 166, 167 s; 3.5 GB | 143, 146 s; 3.1 GB |
| DI settings | ten calls | 101, 101 s; 2.5 GB | | |

Before the split, `assemble` held the `three` and `boundary` cells: `CpuSerial` 662 and 664 s,
6.6 GB; `CpuPolyester` 663 and 669 s, 8.5 GB; `CpuThreaded` was stopped by the 10 GB cap at
11.3 GB. The `CpuSerial` and `CpuPolyester` children of the split `assemble` and `three`
groups were not run again. The `threed` wall time is twice its CPU time in both runs; the
second `CpuSerial` sparse run, 2053 s for 123 s of CPU, is left out as a stall.

The `ghost_fast` row predates the load and `centered_difference!` cells' Enzyme runs: each of the
latter's six calls compiled in under 30 s and 1.6 GB (measured alone, 2026-11-08, 2 threads),
and the child was not re-timed. Every child stays under the 10 GB cap; the largest is `CpuSerial` `threed`, at 8.8 GB. A cell that ends in an Enzyme failure still pays its
compile. Before 2026-11-07 the ghost assembly cells cost more than 22 minutes each before they
threw; they now differentiate in 6 minutes.

The 2026-11-03 figures, at 2 threads and as maximum resident set, were: dense 179 s, 3.4 GB (`CpuSerial`); assemble 578 s,
3.9 GB; solve 176 s, 3.9 GB; operators 118 s, 2.5 GB; sparse Jacobian 133 s, 1.8 GB; DI
settings 100 s, 2.8 GB.

## Open bug: Enzyme forward gives a wrong gradient on a three-summand form (gpena/Bramble.jl#1124)

<!-- Marked paragraph: its handling may change. Edit here only. -->

Open. Enzyme forward mode, through `set_runtime_activity(Forward)`, returns a wrong
gradient when a constant-scaled summand is the third of exactly three summands. An example
is `inner₊(κ∇U, ∇V) + innerₕ(κU, V) + innerₕ(κU, 0.5 D₋ᵧV)`. Every entry is off by one
constant, the same for all parameters. The cell `assemble`, three summands measured a
relative error of 0.17 on the uniform mesh and 0.10 on the non-uniform one, under the three
policies, again on 2026-11-07 (Enzyme 0.13.218). Reverse mode is exact under `CpuSerial`
(6e-16). The workaround is reverse mode.
It is an upstream Enzyme bug. `test/space/ad_coverage.jl` pins the wrong
value and holds an `@test_broken` for the fixed one, so a fix fails the test until this
paragraph and the table's `open` cells are updated.

## Reproducing

```bash
# ForwardDiff cells, no Enzyme needed
julia --project=test -t 2 -e 'using Test; include("test/TestUtils.jl"); include("test/space/ad_coverage.jl")'
```

With Enzyme in the environment (the weekly `ad-*` shards install it into the test project),
the same file runs the Enzyme cells in child processes, with `BRAMBLE_TEST_GROUP=ad` and
`BRAMBLE_TEST_FILE=space/ad_coverage.jl`. No cell is marked slow now: `ghost_data` is not
run, so `BRAMBLE_AD_COVERAGE_SLOW=true` adds nothing.
