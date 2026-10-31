```@meta
CurrentModule = Bramble
CollapsedDocStrings = false
```

# DifferentiationInterface against direct AD calls (gpena/Bramble.jl#1117)

Part 1 ([`ad_backend_evaluation.md`](ad_backend_evaluation.md)) measured five AD backends
through DifferentiationInterface (DI). On 2026-10-31 the user decided to drop
PolyesterForwardDiff, ReverseDiff and Mooncake and to keep ForwardDiff and Enzyme. This page
asks one question about the two kept: should Bramble's tests and examples call them through
DI only, or directly where DI cannot express the call? It recommends and does not decide:
the user decides, and the follow-up plan is planned from that decision.

Every timing, status and error below is read by a script from
`benchmark/results/ad_di_vs_direct.toml`, produced by `benchmark/ad_di_vs_direct.jl`. The
sparse Jacobian figures of part 1 come from `benchmark/results/ad_backend_evaluation.toml`.
The census of direct calls is recomputed from the tree by script, with the regular
expression given in its section. Part 1's page and results file are unchanged.

## What was measured

The file holds 26 timing cells and 12 capability cells. Its meta records commit
`f51ae623e771de302700335f32ecc8aaa5add32f`, 2026-10-31, an Apple M2 (aarch64, Darwin), Julia
1.13.3, DifferentiationInterface 0.7.21, ForwardDiff 1.4.6, Enzyme 0.13.216, 4 threads, a
seeded non-uniform 2D mesh (`uniform = false`), 9 rounds per timing cell and a 3600 s
timeout per cell. `meta.script_sha` ties the file to the script and `meta.base_script_sha`
to part 1's script, which the script includes and does not copy.

Load. `meta.load_1min` is 2.53. That is the load at the last write of the file, with the
run's own 4 threads counted in it. It is not the quiet-machine load. The preflight 1-minute
load average right before the run was 1.69, with every preflight check passing (recorded by
the person who ran it, not in the file). The run took 2674 s. The file records no power
state.

The two workloads are part 1's. `dense_gradient` is the gradient of a scalar functional of
`D₋ₓ`, `normₕ` and a linear form's assembly with respect to `np` nodal entries, on a 34 x 34
mesh (n = 1156) under `CpuSerial`, with `np` in 2, 4, 16, 64, 256 and 1024. `adjoint_solve`
is the gradient of a misfit through `pde_solve(assemble(a(κ)), F)`, n = 1156, `np` in 4, 64
and 1024. Each backend runs both a DI call and the equivalent direct call:

- DI: `DI.prepare_gradient`, then `DI.gradient(f, prep, ad, x)`.
- Direct ForwardDiff: a `GradientConfig` with the chunk DI's preparation chose, then
  `ForwardDiff.gradient!`.
- Direct Enzyme: `autodiff` in reverse mode with `set_runtime_activity`, `Const(f)`,
  `Active` and a zeroed `Duplicated` shadow.

Per-call times come from one child process that interleaves the two prepared calls: 9
rounds, each the best of 3 calls per side, the side that goes first alternating between
rounds. The file stores the median and the minimum over rounds. First-call times
(preparation plus the first call, compilation included) come from a fresh child per side, so
one side's compiled code never helps the other. Policy cells (`CpuThreaded`, `CpuPolyester`;
41 x 41 mesh, n = 1681, `np` = 16) record statuses, values and first-call times only: they
check correctness and are not timed. Capability cells run one DI spelling and, where a
direct one exists, the direct one.

Each cell carries an `attribution` from its two statuses: `none` (both ok), `di` (DI fails,
direct ok), `di_cannot_express` (DI `unsupported`, direct ok), `direct` (only DI ok),
`backend` (both fail) and `n/a` (no direct spelling). The timing cells have 22 `none` and 4
`backend`. The capability cells have 7 `none`, 2 `backend`, 1 `di_cannot_express` and 2
`n/a`. No cell is `di` or `direct`.

How the figures are read, per `bramble-verification`. A same-run ratio against a neighbour
cancels clock, power and thermal drift, so the per-call comparison is the ratio of medians
inside one child. Rows near 20 µs are short, so no difference under about 20 % is read on
them. The first-call ratio compares two separate processes, one run each, and separate
launches swing 10 to 30 %, so the same 20 % bar applies there. Each cell is one run of the
script, not a repeat. Spread is read as the median over the minimum of the rounds, per side.

## Cost of a DI call

Per-call times, prepared, `CpuSerial`, 9 interleaved rounds per row. The ratio columns are
DI over direct.

| Workload | Backend | np | DI median | direct median | median ratio | min ratio | median/min (DI, direct) |
|:--|:--|--:|--:|--:|--:|--:|--:|
| dense_gradient | ForwardDiff | 2 | 11.4 µs | 11.6 µs | 0.982 | 0.978 | 1.03, 1.03 |
| dense_gradient | ForwardDiff | 4 | 18.7 µs | 19.1 µs | 0.980 | 1.045 | 1.94, 2.07 |
| dense_gradient | ForwardDiff | 16 | 82.4 µs | 82.1 µs | 1.004 | 1.009 | 1.81, 1.82 |
| dense_gradient | ForwardDiff | 64 | 175.5 µs | 176.5 µs | 0.995 | 1.002 | 1.02, 1.03 |
| dense_gradient | ForwardDiff | 256 | 761.0 µs | 761.5 µs | 0.999 | 1.049 | 1.12, 1.18 |
| dense_gradient | ForwardDiff | 1024 | 3.14 ms | 3.09 ms | 1.018 | 1.003 | 1.04, 1.02 |
| dense_gradient | Enzyme | 2 | 20.5 µs | 20.4 µs | 1.004 | 0.996 | 1.04, 1.03 |
| dense_gradient | Enzyme | 4 | 20.2 µs | 20.2 µs | 0.996 | 0.992 | 1.03, 1.02 |
| dense_gradient | Enzyme | 16 | 21.9 µs | 22.2 µs | 0.987 | 0.962 | 1.05, 1.02 |
| dense_gradient | Enzyme | 64 | 19.9 µs | 20.2 µs | 0.988 | 1.017 | 1.02, 1.05 |
| dense_gradient | Enzyme | 256 | 21.0 µs | 21.2 µs | 0.992 | 1.017 | 1.03, 1.06 |
| dense_gradient | Enzyme | 1024 | 23.0 µs | 22.2 µs | 1.032 | 1.010 | 1.05, 1.03 |
| adjoint_solve | ForwardDiff | 4 | 1.36 ms | 1.38 ms | 0.981 | 0.992 | 1.01, 1.02 |
| adjoint_solve | ForwardDiff | 64 | 11.22 ms | 11.39 ms | 0.985 | 1.006 | 1.01, 1.03 |
| adjoint_solve | ForwardDiff | 1024 | 235.16 ms | 234.86 ms | 1.001 | 1.012 | 1.25, 1.26 |
| adjoint_solve | Enzyme | 4 | 3.47 ms | 3.47 ms | 1.000 | 1.005 | 1.01, 1.01 |
| adjoint_solve | Enzyme | 64 | 3.46 ms | 3.46 ms | 1.001 | 1.002 | 1.01, 1.01 |
| adjoint_solve | Enzyme | 1024 | 3.40 ms | 3.38 ms | 1.007 | 1.000 | 1.01, 1.01 |

First-call times and per-call allocation, same rows.

| Workload | Backend | np | DI first call | direct first call | ratio | DI minus direct allocation per call |
|:--|:--|--:|--:|--:|--:|--:|
| dense_gradient | ForwardDiff | 2 | 0.51 s | 0.50 s | 1.017 | 80 B |
| dense_gradient | ForwardDiff | 4 | 0.51 s | 0.50 s | 1.018 | 96 B |
| dense_gradient | ForwardDiff | 16 | 0.55 s | 0.51 s | 1.084 | 192 B |
| dense_gradient | ForwardDiff | 64 | 0.53 s | 0.54 s | 0.984 | 576 B |
| dense_gradient | ForwardDiff | 256 | 0.52 s | 0.52 s | 1.012 | 2112 B |
| dense_gradient | ForwardDiff | 1024 | 0.52 s | 0.52 s | 0.992 | 8256 B |
| dense_gradient | Enzyme | 2 | 6.42 s | 6.37 s | 1.007 | 80 B |
| dense_gradient | Enzyme | 4 | 6.42 s | 6.35 s | 1.011 | 96 B |
| dense_gradient | Enzyme | 16 | 6.46 s | 6.47 s | 0.998 | 192 B |
| dense_gradient | Enzyme | 64 | 6.44 s | 6.36 s | 1.012 | 576 B |
| dense_gradient | Enzyme | 256 | 6.46 s | 6.36 s | 1.015 | 2112 B |
| dense_gradient | Enzyme | 1024 | 6.54 s | 6.33 s | 1.033 | 8256 B |
| adjoint_solve | ForwardDiff | 4 | 3.42 s | 3.45 s | 0.990 | 96 B |
| adjoint_solve | ForwardDiff | 64 | 3.51 s | 3.50 s | 1.003 | 576 B |
| adjoint_solve | ForwardDiff | 1024 | 3.97 s | 3.81 s | 1.042 | 8256 B |
| adjoint_solve | Enzyme | 4 | 101.4 s | 101.4 s | 1.000 | 96 B |
| adjoint_solve | Enzyme | 64 | 101.4 s | 101.3 s | 1.001 | 752 B |
| adjoint_solve | Enzyme | 1024 | 101.0 s | 101.3 s | 0.997 | 8256 B |

Reading:

- The ratio of medians is 0.980 to 1.032 over the 18 rows, and the ratio of minima 0.962 to
  1.049. The largest departure of a median ratio from 1 is 3.2 %, and of a minimum ratio 4.9
  %. No row reaches the 20 % bar.
- 8 rows sit near 20 µs (ForwardDiff at np = 2 and 4, Enzyme dense at every np). There the
  bar applies, and the ratios are inside it by a wide margin (0.980 to 1.032). On the 10
  rows of 80 µs or more the ratio of medians is 0.981 to 1.018.
- Round spread. The median over the minimum is 1.01 to 2.07 per side. 4 rows exceed 1.1:
  ForwardDiff dense at np = 4 (1.94, 2.07) and np = 16 (1.81, 1.82), np = 256 (1.12, 1.18)
  and adjoint np = 1024 (1.25, 1.26). The two sides spread about alike on every row and the
  ratio of medians stays inside 2.0 % there, because the rounds alternate sides. The
  absolute per-call figure on a wide row is not stable to better than that factor, so it is
  not quoted as a cost of DI.
- First call, which is the cost of an unprepared one-shot call: the ratio is 0.984 to 1.084.
  The largest is ForwardDiff dense_gradient np = 16 (0.55 s against 0.51 s). Its neighbours
  at np = 2, 4 and 64 are 1.017, 1.018 and 0.984, so there is no trend with np, and it is
  inside the cross-process bar. Compilation dominates and DI changes it by no amount this
  run reads: about 0.5 s for ForwardDiff dense, 3.4 to 4.0 s for the ForwardDiff adjoint,
  6.3 to 6.5 s for Enzyme dense and 101 s for the Enzyme adjoint, on both sides.
- Allocation: DI allocates 80 to 8256 B more per call. In 17 of 18 rows the difference is
  exactly 8 np + 64 B, which fits the fresh gradient vector that `DI.gradient` returns where
  the direct call overwrites its buffer. That reading is inferred from the pattern, not from
  the code. The remaining row (Enzyme adjoint_solve np = 64: 752 B) is 176 B above that.

DI adds no per-call or first-call cost that this run can see, for either backend, on either
workload. The limit is the run: one run, 4 threads, one machine, a preflight load of 1.69.

## What DI can express

One line per capability cell. `status_direct = n/a` means no direct spelling exists.

| Cell | DI | direct | attribution | relerr (DI) | what it compares |
|:--|:--|:--|:--|--:|:--|
| `capability/constant_context/ForwardDiff` | ok | ok | none | 1.16e-11 | DI.Constant context against a closure over the constant |
| `capability/constant_context/Enzyme` | ok | ok | none | 1.16e-11 | the same, closure held in Const |
| `capability/chunk_size/ForwardDiff` | ok | ok | none | 1.13e-11 | `AutoForwardDiff(chunksize = 4)` against `Chunk{4}`; the prepared chunk is 4 |
| `capability/enzyme_forward/Enzyme` | ok | ok | none | 1.13e-11 | `AutoEnzyme` in forward mode with runtime activity and `Const` function |
| `capability/enzyme_reverse/Enzyme` | ok | ok | none | 1.13e-11 | the same in reverse mode |
| `capability/enzyme_no_runtime_activity_di/Enzyme` | throws | throws | backend | n/a | reverse mode without `set_runtime_activity`, DI spelling |
| `capability/enzyme_no_runtime_activity_direct/Enzyme` | throws | throws | backend | n/a | the same call spelled directly |
| `capability/hessian/ForwardDiff` | ok | ok | none | 0.00e+00 | `DI.hessian` on a closed-form field, the shape of `elasticity_3d.jl` |
| `capability/derivative/ForwardDiff` | ok | ok | none | 2.57e-16 | `DI.derivative` on a closed-form scalar function |
| `capability/two_active_arguments/Enzyme` | unsupported | ok | di_cannot_express | 0.00e+00 | gradient of `pde_solve` with respect to `A` and `F` together: direct has two `Duplicated`, DI differentiates one argument |
| `capability/sparse_jacobian_cpu_threaded/ForwardDiff` | ok | n/a | n/a | 5.16e-12 | `AutoSparse(AutoForwardDiff())` Jacobian under `CpuThreaded`, no direct route exists |
| `capability/sparse_jacobian_cpu_polyester/ForwardDiff` | ok | n/a | n/a | 5.16e-12 | the same under `CpuPolyester` |

Reading:

- DI expresses a `Constant` context, the ForwardDiff chunk size, Enzyme forward and reverse
  mode, and `hessian` and `derivative` for ForwardDiff. DI and direct agree on all of them
  (`value_diff`, the max-norm relative difference, is 0.0), with `relerr` as in the column.
- `set_runtime_activity` is required with or without DI. Reverse mode without it throws
  `EnzymeRuntimeActivityError` in both spellings (`backend`), so DI neither removes nor adds
  the requirement.
- `two_active_arguments` is the one cell DI cannot express (`di_cannot_express`). Direct
  Enzyme returns the gradient of `J(A, F) = sum(abs2, pde_solve(A, F))` with respect to `A`
  and `F` through two `Duplicated` arguments. DI differentiates one argument. With `F`
  active and `A` a `DI.Constant` it returns the gradient with respect to `F` only (`relerr`
  0.0 against the adjoint), and the gradient with respect to `A` is not available that way.
- The sparse Jacobian is DI-only. No direct ForwardDiff route exists without re-implementing
  compression and colouring, which `AutoSparse` with `ast_sparsity_detector` provides. Part
  1's `sparse_jacobian` cell (`benchmark/results/ad_backend_evaluation.toml`, ForwardDiff,
  `CpuSerial`, n = 10000) is `ok` at 2.34 ms per call and 1.26 s first call, `relerr`
  7.34e-12. The two cells above (`sparse_jacobian_cpu_threaded` and
  `sparse_jacobian_cpu_polyester`, smoke size) show it is also ok under both threaded
  policies.

## Threaded policies

Both threaded policies are checked for correctness, not timed. The cells run `np` = 16 on a
41 x 41 mesh (n = 1681), each side in its own process. One row per cell.

| Cell (`key`) | Backend | Workload | DI | direct | attribution | error type | DI vs direct | relerr (DI) |
|:--|:--|:--|:--|:--|:--|:--|--:|--:|
| `dense_gradient/ForwardDiff/CpuThreaded/16` | ForwardDiff | dense_gradient | ok | ok | none | none | 0.0 | 5.66e-10 |
| `dense_gradient/Enzyme/CpuThreaded/16` | Enzyme | dense_gradient | throws | throws | backend | AssertionError | n/a | n/a |
| `adjoint_solve/ForwardDiff/CpuThreaded/16` | ForwardDiff | adjoint_solve | ok | ok | none | none | 0.0 | 8.22e-08 |
| `adjoint_solve/Enzyme/CpuThreaded/16` | Enzyme | adjoint_solve | throws | throws | backend | AssertionError | n/a | n/a |
| `dense_gradient/ForwardDiff/CpuPolyester/16` | ForwardDiff | dense_gradient | ok | ok | none | none | 0.0 | 5.66e-10 |
| `dense_gradient/Enzyme/CpuPolyester/16` | Enzyme | dense_gradient | throws | throws | backend | IllegalTypeAnalysisException | n/a | n/a |
| `adjoint_solve/ForwardDiff/CpuPolyester/16` | ForwardDiff | adjoint_solve | ok | ok | none | none | 0.0 | 8.22e-08 |
| `adjoint_solve/Enzyme/CpuPolyester/16` | Enzyme | adjoint_solve | throws | throws | backend | IllegalTypeAnalysisException | n/a | n/a |

Reading:

- ForwardDiff. All 4 cells are ok on both sides, `none`, DI and direct agreeing (`DI vs
  direct` is the max-norm relative difference, 0.0). The `relerr` is 5.66e-10 for the dense
  gradient and 8.22e-08 for the adjoint, against a finite-difference reference. Nothing
  needs a guard for ForwardDiff: the follow-up plan pins these four as passing.
- Enzyme. All 4 cells throw on both sides with the same error type, so all are `backend`.
  Under `CpuThreaded` it is `AssertionError: mi2 !== nothing`, under `CpuPolyester`
  `IllegalTypeAnalysisException` on a `BatchClosure` (the closure numbers in the error text
  differ between the two sides, as they do between processes). Direct Enzyme fails exactly
  as DI does, so DI neither causes these failures nor can its migration hide them.
- No cell is `di`. The plan had a subplan to reduce each DI-side failure to a minimal DI
  reproducer. It was abandoned for that reason, so there is no `ad_di_upstream.md`. A row
  would be called a DI-side failure and a DI bug only once such a reproducer in plain Julia
  exists. None does.
- What the rows mean for the policy guards of the follow-up plan. The user's goal is that
  both kept backends differentiate under `CpuThreaded` and `CpuPolyester`, with no silent
  serial fallback and no refusal. A guard is then a test that pins the result, not a runtime
  check that diverts or refuses: the four ForwardDiff cells stay ok, and the four Enzyme
  cells must go from `throws` to ok in both spellings. The Enzyme failures belong to that
  plan to fix, not to DI. `backend` says both spellings fail alike. It does not say whether
  the cause lies in Bramble's threaded code or in Enzyme.

## Direct calls today

Every file under `test/` and `docs/src/examples/` with a line matching
`ForwardDiff.(gradient|jacobian|derivative|hessian)`, `Enzyme.(autodiff|gradient|jacobian)`,
`ReverseDiff.(gradient|jacobian)` or `Mooncake.(value_and_gradient|prepare_)`. The script
that built the table walks the tree with that expression, so a file cannot be missed. A
count is the number of matching lines, so a line with two calls counts once. The classes:

- (a) differentiates Bramble code, so it moves to DI if DI is adopted.
- (b) a closed-form calculus helper (manufactured solutions in `elasticity_traction_2d.jl`
  and `elasticity_3d.jl`), which differentiates no Bramble code.
- (c) tests a backend-specific rule or behaviour (the `pde_solve` rules in
  `test/ext/chainrules_enzyme_ext.jl`, the ReverseDiff `mul!` extension).
- (d) names a call in a comment (a line starting with `#`), with no call, counted apart so
  the migration does not chase prose.

| Class | File | Lines | Backend | Note |
|:--|:--|--:|:--|:--|
| (a) | `docs/src/examples/inverse_diffusion.jl` | 2 | Enzyme | `Enzyme.gradient` of a loss through `pde_solve`; the DI spelling is cell `enzyme_reverse` |
| (a) | `test/TestUtils.jl` | 1 | ForwardDiff | `_matches_fd` helper, used by the form and space tests |
| (a) | `test/ext/ad_backend_verification.jl` | 4 | ForwardDiff | dense `ForwardDiff.jacobian` of a Bramble residual: 2 pattern checks, 2 references for DI results |
| (a) | `test/ext/sciml_ext.jl` | 2 | ForwardDiff |  |
| (a) | `test/ext/sparspak_ext.jl` | 1 | ForwardDiff |  |
| (a) | `test/form/autodiff.jl` | 12 | ForwardDiff |  |
| (a) | `test/form/bilinear.jl` | 5 | ForwardDiff |  |
| (a) | `test/form/dirac.jl` | 4 | ForwardDiff |  |
| (a) | `test/form/forwarddiff_smoke.jl` | 4 | ForwardDiff |  |
| (a) | `test/form/ghost_semidiscrete.jl` | 4 | ForwardDiff |  |
| (a) | `test/form/ghost_time_data.jl` | 2 | ForwardDiff |  |
| (a) | `test/form/interpolation_operator.jl` | 3 | ForwardDiff |  |
| (a) | `test/form/jacobian_pattern.jl` | 1 | ForwardDiff |  |
| (a) | `test/form/kronecker.jl` | 1 | ForwardDiff |  |
| (a) | `test/form/linear.jl` | 17 | ForwardDiff |  |
| (a) | `test/form/rank_one.jl` | 1 | ForwardDiff |  |
| (a) | `test/form/reaction_flux.jl` | 6 | ForwardDiff |  |
| (a) | `test/form/semidiscrete.jl` | 3 | ForwardDiff |  |
| (a) | `test/form/simplifier.jl` | 3 | ForwardDiff |  |
| (a) | `test/form/source_operators.jl` | 2 | ForwardDiff |  |
| (a) | `test/solvers/multigrid.jl` | 2 | ForwardDiff |  |
| (a) | `test/space/autodiff.jl` | 1 | ForwardDiff |  |
| (a) | `test/space/autodiff_backends.jl` | 3 | ForwardDiff | lines 65 and 69 are references for DI results, line 102 an `@inferred` call |
| (a) | `test/space/autodiff_policies.jl` | 4 | ForwardDiff | references for the policy cross-check, themselves checked against central differences |
| (a) | `test/space/weights_staleness.jl` | 1 | ForwardDiff |  |
| (b) | `docs/src/examples/elasticity_3d.jl` | 1 | ForwardDiff | closed-form Hessian of the manufactured solution |
| (b) | `docs/src/examples/elasticity_traction_2d.jl` | 2 | ForwardDiff | `∇` and `hess` helpers over closed-form fields |
| (c) | `test/ext/ad_backend_verification.jl` | 1 | ReverseDiff | `ReverseDiff.gradient` in the `ReverseDiff: Kronecker mul!` testset (dropped backend, testset deleted) |
| (c) | `test/ext/chainrules_enzyme_ext.jl` | 12 | Enzyme | the `pde_solve` rules: 11 `Enzyme.gradient` with a `Const` loss (one inside a `@test_throws` for a missing rule), 1 `Enzyme.autodiff` with `Duplicated` `A` (line 226) |
| (c) | `test/ext/chainrules_enzyme_ext.jl` | 1 | Mooncake | `Mooncake.value_and_gradient!!` in the `Mooncake: pinned unsupported` testset (dropped backend, testset deleted) |
| (d) | `docs/src/examples/coupled_reaction_diffusion.jl` | 1 | ForwardDiff | comment, line 261 |
| (d) | `docs/src/examples/inverse_diffusion.jl` | 2 | Enzyme | comment lines 87 and 122 |
| (d) | `docs/src/examples/poisson_nonlinear.jl` | 1 | ForwardDiff | comment, line 116 |
| (d) | `test/examples/inverse_diffusion.jl` | 1 | Enzyme | comment, line 9 |
| (d) | `test/ext/ad_backend_verification.jl` | 1 | ForwardDiff | comment, line 40 |
| (d) | `test/ext/sparse_ad_ext.jl` | 1 | ForwardDiff | comment, line 24 |

Per class, over 32 files (a file can sit in two classes, so the file counts add to more):

| Class | Lines | Files | Of which |
|:--|--:|--:|:--|
| (a) differentiates Bramble | 89 | 25 | ForwardDiff 87, Enzyme 2 |
| (b) closed-form helper | 3 | 2 | ForwardDiff 3 |
| (c) backend-specific test | 14 | 2 | Enzyme 12, ReverseDiff 1, Mooncake 1 |
| (d) prose only | 7 | 6 | ForwardDiff 4, Enzyme 3 |

Notes on the table:

- The two class (c) lines of dropped backends, `ReverseDiff.gradient` in
  `test/ext/ad_backend_verification.jl` and `Mooncake.value_and_gradient!!` in
  `test/ext/chainrules_enzyme_ext.jl`, sit in testsets that exist for the dropped backend
  (`ReverseDiff: Kronecker mul! (#295)` and `Mooncake: pinned unsupported`). The follow-up
  plan marks them deleted, not moved. No file in the census exists for a dropped backend
  alone, so nothing else here is deleted. The `Auto*` spellings of dropped backends in the
  survey tests (for example `AutoReverseDiff()` in `test/form/autodiff.jl`) are DI calls,
  not direct calls, and are outside this table.
- 8 class (a) lines are references that a DI result is compared against: 2 in
  `test/ext/ad_backend_verification.jl`, 2 in `test/space/autodiff_backends.jl` and 4 in
  `test/space/autodiff_policies.jl`. They move like the rest. DI and direct ForwardDiff
  agree exactly (difference 0.0) on all 18 `CpuSerial` rows and the 4 policy rows, so a DI
  reference gives the same oracle, and the files already check ForwardDiff against central
  differences.
- Not calls to differentiate: 72 code lines in 14 files build or read a `ForwardDiff.Dual`
  (`ForwardDiff.Dual`, `value`, `partials`), for example in `test/form/kronecker_edge.jl`
  and `test/space/element_type.jl`. They test that Bramble's containers widen to `Dual`, so
  they are not in the census, DI does not express them, and `ForwardDiff` stays a test
  dependency under any decision.

## Recommendation

DI with named exceptions. Calls that differentiate Bramble go through DI. The measurements
give no reason for an exception on cost: DI over direct is 0.980 to 1.032 per call and 0.984
to 1.084 on first call, and the only capability cell DI cannot express is
`two_active_arguments`. DI also does one thing no direct call does, the sparse Jacobian. The
user decides.

| Exception | Forced by | What stays direct |
|:--|:--|:--|
| `test/ext/chainrules_enzyme_ext.jl` | cell `capability/two_active_arguments/Enzyme`: `di_cannot_express`, DI differentiates one argument and the file differentiates `pde_solve` with respect to `A` and `F` together, and tests the rule itself | the 12 Enzyme lines of the file, as one direct-Enzyme file. Line 226 needs `A` and `F` both active. The other lines could be spelled `AutoEnzyme`, but the file exists to call Enzyme's own entry points so that the `pde_solve` rule fires |
| `ForwardDiff.Dual`, `value` and `partials` sites | not differentiation calls, DI has no spelling for a hand-built `Dual` | 72 code lines in 14 files, and ForwardDiff as a test dependency |
| class (b), optional | no cell forces it: `hessian` and `derivative` are ok in DI and agree exactly | 3 lines in 2 docs examples. They differentiate no Bramble code and DI shortens none of them, so the page recommends leaving them |

What the follow-up plan would move, by class:

| Class | Lines | Files | Action |
|:--|--:|--:|:--|
| (a) | 89 | 25 | move to DI (87 ForwardDiff, 2 Enzyme in `docs/src/examples/inverse_diffusion.jl`, which also needs `set_runtime_activity` through DI) |
| (b) | 3 | 2 | leave, unless the user moves them |
| (c) | 14 | 2 | 12 Enzyme lines stay direct, 1 ReverseDiff and 1 Mooncake lines are deleted with their testsets |
| (d) | 7 | 6 | no migration; reword a line only where a moved call changes what it names (the two Enzyme lines in `docs/src/examples/inverse_diffusion.jl`) |

The Enzyme threaded failures are outside this choice. They are the same under DI and direct,
so neither answer fixes or worsens them; the follow-up plan fixes them to meet the user's
goal that `CpuThreaded` and `CpuPolyester` differentiate.
