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
| Date | 2026-11-03; the three-summand cells 2026-11-04 |
| Tree | branch `v4-2-0-followup`; the three-summand cells with the `_allocate_walked` fix |
| Machine | Apple M2, Julia 1.13.3 |
| Enzyme | 0.13.216; the `pde_solve` forward cells 0.13.218 (2026-11-05), where the error changed |
| Threads | 2, so both threaded policies run on 2 threads |
| Problem | 9 x 9 mesh in 2D, 17 nodes in 1D, three parameters, `Float64` |
| Agreement | within 1e-8 relative to ForwardDiff on the same policy |

The `pde_solve` and `ghost_data` cells compare with a central difference where ForwardDiff
cannot run (rel. 1e-5).

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
| interpolation | `Rₕ`, `avgₕ` | works | works | works | works |
| assemble | `assemble` of a linear form | works | works | works | works |
| assemble | `assemble` of `inner₊(κ∇U, ∇V)` | works | works | works | works |
| assemble | the same with `Mₕ(κ)` | works | works | works | works |
| assemble | `assemble(a, l; dirichlet)`, 1D | works | works | works | works |
| assemble | the same in 2D | works | reverse excluded: aborts (`needsCacheWholeAllocation`), forward works | not run | not run |
| assemble | three summands, the third scaled by a constant | works | reverse works, forward open (#1124) | reverse works, forward open (#1124) | reverse works, forward open (#1124) |
| pde_solve | `pde_solve(A, F)` with `A` from a 1D `assemble` of `κ` | excluded: `StackOverflowError` on a `Dual` matrix | reverse works, forward excluded: `EnzymeNoDerivativeError` | reverse works, forward excluded: `EnzymeNoDerivativeError` | reverse works, forward excluded: `EnzymeNoDerivativeError` |
| semidiscretize | `semidiscretize_rhs(sd)(du, u, p, t)` | works | works | works | works |
| matrix_free | `mul!` with `matrix_free_operator` | works | works | works | works |
| kronecker | `mul!` with `kronecker_operator` | works | works | works | works |
| ghost | `assemble(a, l)` on a constrained ghost space, parameters scale the form's coefficients | works | reverse excluded: `EnzymeInternalError`, forward not run | not run | not run |
| ghost | the same with `inner₊` and `innerₕ` terms only | works | reverse excluded: `EnzymeInternalError`, forward not run | not run | not run |
| ghost | relation data that depend on the parameter | works | excluded: fails (measured outside this test, see the limits) | excluded: fails (measured outside this test) | excluded: fails (measured outside this test) |
| ghost | a Robin coefficient that depends on the parameter | excluded: `MethodError` (`Float64(::Dual)`) | not run | not run | not run |
| ghost | `Δₕ!` on a `GhostVector` | works | reverse excluded: `EnzymeNoDerivativeError`, forward excluded: segmentation fault | reverse excluded: `CompositeException`, forward not run | reverse excluded: `EnzymeNoDerivativeError`, forward not run |
| sparse Jacobian | `AutoSparse` with `ast_sparsity_detector`, 1D diffusion residual | works | works | works | works |

Three more facts about the table:

- ForwardDiff and Enzyme agree on every `works` cell to 1e-8 relative or better. The worst
  cell, the mesh endpoint, agrees to 5e-11. The rest agree to 1e-15 and 1e-16.
- The 3D cell (`Mₕ(κ)` on a `5 x 5 x 5` mesh) is in the limits below.
- The ghost cells run on a non-uniform mesh only, since the ghost test helper builds those.

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
- **Ghost `Δₕ!` and `centered_difference!` fail under serial too.** Row `ghost`, `Δₕ!` on a
  `GhostVector`: reverse throws `EnzymeNoDerivativeError`, forward ends the process with a
  segmentation fault.
- **Ghost spaces whose boundary data depends on the parameter fail.** Under `CpuSerial`
  Enzyme throws `Illegal calling convention fixup` in reverse and `EnzymeNoDerivativeError`
  in forward (dgetrf), measured on 2026-11-02 by a review of the `CpuThreaded` rules,
  on a ghost space whose boundary data takes the parameter. This page's own `ghost_data`
  cell was started in reverse and stopped after more than 100 minutes of compile with no
  result, so it is recorded from that measurement and is not in the test. ForwardDiff
  differentiates the data and fails on a Robin coefficient with `MethodError:
  Float64(::Dual)`.
- **Ghost assembly fails in Enzyme reverse under `CpuSerial`.** The parameters scale the
  form's coefficients and nothing else depends on them. Both ghost assembly cells throw
  `EnzymeInternalError` after a compile of 22 and 29 minutes. The forward mode and the
  threaded policies were not run, because the reverse compile alone is that long. The
  test runs these two cells, under `CpuSerial`, only with `BRAMBLE_AD_COVERAGE_SLOW=true`.
- **The Dirichlet path in 2D aborts Enzyme reverse.** `assemble(a, l; dirichlet = …)` on a
  2D mesh ends the process with `needsCacheWholeAllocation` under `CpuSerial`. Forward
  works there. Under `CpuThreaded` the reverse call was still compiling after 22 minutes
  and was stopped, so the threaded policies are not run.
- **Plan geometry is inactive.** The plans that carry the mesh geometry (the replay plans of
  the threaded assembly and the matrix-free plan `_mf_plan`) are inactive for Enzyme
  (`EnzymeRules.inactive` on `_mf_plan` in `ext/BrambleEnzymeExt/threaded.jl`), so a
  coordinate read only through a plan gets no derivative. The `mesh` row that
  assembles in the endpoint `b` works under `CpuSerial` in both modes (6e-10 and 2e-10
  against ForwardDiff); that route was not measured through a matrix-free operator or under
  the threaded policies.
- **3D `Mₕ(κ)`.** `assemble` of `inner₊(Mₕ(κ)∇U, ∇V)` on a `5 x 5 x 5` mesh throws
  `EnzymeNoTypeError` under `CpuSerial`, in reverse and forward, on both meshes. The 2D
  form works. The threaded policies were not run. The test checks this cell only where
  3D meshes are allowed (not in the `unit` group).
- **Compile memory.** `_batch_mf_bands_coef`, the matrix-free reverse with a coefficient,
  compiles at 5.96 GB and 747 s under `CpuPolyester`, against 2.46 GB and 215 s under
  `CpuSerial`, measured on 2026-11-02 by the `CpuPolyester` matrix-free rule's own memory
  run, and not re-run here.

## Open failures

- Enzyme reverse of the 2D and 3D Dirichlet assembly
  ([gpena/Bramble.jl#1126](https://github.com/gpena/Bramble.jl/issues/1126)):
  `needsCacheWholeAllocation` abort.
- Ghost assembly in Enzyme reverse, `Δₕ!` on a `GhostVector`, and the Robin coefficient
  ([gpena/Bramble.jl#1127](https://github.com/gpena/Bramble.jl/issues/1127)).
- 3D `assemble` with `Mₕ(κ)` in Enzyme: `EnzymeNoTypeError`
  ([gpena/Bramble.jl#1128](https://github.com/gpena/Bramble.jl/issues/1128)).
- Enzyme forward through `pde_solve`: `EnzymeNoDerivativeError` (no forward rule for UMFPACK's
  `umfpack_dl_defaults`); reverse has the adjoint rule
  ([gpena/Bramble.jl#1129](https://github.com/gpena/Bramble.jl/issues/1129)).
- ForwardDiff through `pde_solve`: `StackOverflowError` from the sparse `lu` of a
  `Dual`-valued matrix; the tutorial's workaround is `Matrix(A) \ F`
  ([gpena/Bramble.jl#1130](https://github.com/gpena/Bramble.jl/issues/1130)).
- ForwardDiff on a mesh built with `backend(policy = …)`
  ([gpena/Bramble.jl#1131](https://github.com/gpena/Bramble.jl/issues/1131)).

## Compile cost of the cells

One child per policy and group, with Enzyme compiling every cell of the group on both
meshes in both modes. Wall time includes the ForwardDiff reference. Peak is the child's
maximum resident set. The runs shared the machine with another Julia job at times.

| Group | Cells | `CpuSerial` | `CpuThreaded` | `CpuPolyester` |
|:--|:--|:--|:--|:--|
| dense | mesh, operators, interpolation (8) | 179 s, 3.4 GB | 208 s, 2.9 GB | 214 s, 2.4 GB |
| assemble | five assemblies and the three-summand one | 578 s, 3.9 GB | 1148 s, 3.0 GB | 873 s, 2.3 GB |
| solve | `pde_solve`, `semidiscretize` | 176 s, 3.9 GB | 282 s, 3.2 GB | 228 s, 3.6 GB |
| operators | matrix-free, Kronecker | 118 s, 2.5 GB | 229 s, 3.0 GB | 383 s, 4.0 GB |
| sparse Jacobian | the residual, four calls | 133 s, 1.8 GB | 178 s, 3.1 GB | not timed in isolation |
| DI settings | ten calls | 100 s, 2.8 GB | | |

Every child stays under 4.1 GB. A cell that ends in an Enzyme failure still pays its
compile: the ghost assembly cells cost more than 22 minutes each before they throw.

## Open bug: Enzyme forward gives a wrong gradient on a three-summand form (gpena/Bramble.jl#1124)

<!-- Marked paragraph: its handling may change. Edit here only. -->

Open. Enzyme forward mode, through `set_runtime_activity(Forward)`, returns a wrong
gradient when a constant-scaled summand is the third of exactly three summands. An example
is `inner₊(κ∇U, ∇V) + innerₕ(κU, V) + innerₕ(κU, 0.5 D₋ᵧV)`. Every entry is off by one
constant, the same for all parameters. The cell `assemble`, three summands measured a
relative error of 0.17 on the uniform mesh and 0.10 on the non-uniform one, under the three
policies. Reverse mode is exact under `CpuSerial` (6e-16). The workaround is reverse mode.
It is an upstream Enzyme bug (Enzyme 0.13.216). `test/space/ad_coverage.jl` pins the wrong
value and holds an `@test_broken` for the fixed one, so a fix fails the test until this
paragraph and the table's `open` cells are updated.

## Reproducing

```bash
# ForwardDiff cells, no Enzyme needed
julia --project=test -t 2 -e 'using Test; include("test/TestUtils.jl"); include("test/space/ad_coverage.jl")'
```

With Enzyme in the environment (the weekly `ad-2` shard installs it), the same file runs the
Enzyme cells in child processes. `BRAMBLE_AD_COVERAGE_SLOW=true` adds the cells that
compile for tens of minutes before Enzyme fails.
