# #1135: LLVM verification abort in `chainrules_enzyme_ext.jl`

Reduction of gpena/Bramble.jl#1135 (Weekly ad-4 shard aborts with `LLVM ERROR: function
failed verification (2)`). The reproducer is `1135_llvm_verify.jl`, next to this note. It
needs nothing from Bramble.

Reproduces: `julia --project=. --threads=3 --optimize=1 --check-bounds=auto docs/notes/internals/ad_reductions/1135_llvm_verify.jl` from the repository root dies with `LLVM ERROR: function failed verification (2)` and exit 134 (SIGABRT), in 15 s; the same command with `--check-bounds=yes` does the same. It runs a 2 by 2 sparse `A \ [θ, 1.0]` under `Enzyme.gradient(set_runtime_activity(Reverse), Const(f), 0.7)`.

check-bounds: both. `yes` and `auto` abort identically, so the flag is not needed and the switch in 50b56e4c did not remove the abort.

Fix site: `Enzyme` upstream, the reverse-mode compile of `SparseArrays.CHOLMOD._extract_args` (SparseArrays `src/solvers/cholmod.jl:1119`), reached by the sparse `\`; no Bramble function is at fault, and the one Bramble-side change is the test `loss_no_rule` in `test/ext/chainrules_enzyme_ext.jl`.

Compile: 15 s and 0.78 GB peak for the reproducer; 257 s and 2.1 GB (auto) for `loss_no_rule` alone, 601 s and 6.0 GB for the cases before `loss_overwritten` under `auto` (sequential, one process).

Enzyme: 0.13.218 (Julia 1.13.3, macOS arm64, 3 threads).

## What aborts

The call at `chainrules_enzyme_ext.jl:89` is the testset `Enzyme: native BrambleEnzymeExt
rule`, not one statement. The cases in it were rerun in one process (Enzyme and
DifferentiationInterface, each loss through both, as the test's `grad` does) under
`--optimize=1`, 3 threads, a coverage tracefile, with the issue's dump naming the function
`preprocess_julia__extract_args_*`:

| Case in the test | `--check-bounds=yes` | `--check-bounds=auto` |
|---|---|---|
| `loss`, `loss_symmetrized`, `loss_kw` | gradient 29.4, direct = DI | same |
| `loss_coeff`, `loss_coeff_sum`, `loss_stiff_3d`, `loss_sum_three` | pass | pass |
| `solve_loss` shadow of a real and of a complex `A` | pass | pass |
| `loss_no_rule`: `sum(abs2, A \ F)` on the assembled system | **abort** | **abort** |
| `loss_overwritten`, `loss_rect` | 0.5, `ArgumentError` | 0.5, `ArgumentError` |

`loss_no_rule` is the only case that aborts. It is the pinned failure: the test expects
`@test_throws Exception Enzyme.gradient(mode, Enzyme.Const(loss_no_rule), θ0)`, because
Enzyme has no rule for `A \ F` and used to raise `IllegalTypeAnalysisException` there. On
0.13.218 the compiler dies in LLVM's verifier instead (`PHI nodes not grouped at top of
basic block!` in `preprocess_julia__extract_args_*`, whose body is CHOLMOD's
`_extract_args`: it wraps `cholmod_l_sparse_struct_typed`), so no exception is thrown for
the test to catch. The three 2D losses that run in child processes are not involved.

The `pde_solve` cases pass, so the native `EnzymeRules` rule in `BrambleEnzymeExt` is not at
fault, and neither are #1126's Dirichlet assembly or e35b59fe's adjoint, both of which the
issue listed as candidates.

## Reduction

Bramble is not needed: `Enzyme.gradient` of `θ -> sum(SPD2 \ [θ, 1.0])` aborts, with `SPD2`
a 2 by 2 sparse symmetric positive definite matrix. Variants, same flags (checked under
both `yes` and `auto`):

- `\` with a 21-point tridiagonal `A`, `θ` in the right-hand side: abort.
- `\` with `θ` in the matrix values (`A + θI`): abort.
- `\` with an upper triangular sparse `A`: abort.
- An explicit `cholesky(A) \ b`: an Enzyme exception (`EnzymeNoDerivativeError` on the 2 by 2
  matrix, `EnzymeRuntimeException` on the 21-point one), no abort.
- An explicit `lu(A) \ b` on the 21-point matrix: `EnzymeNoDerivativeError`.

So the abort needs the generic sparse `\`, not a particular factorisation or Bramble code.

## For the fix subplan

- The cause is in Enzyme, so no Bramble workaround that changes results (plan departure on
  unknown fix sites). The report to file upstream (plan gate item G3) is the reproducer
  above, with the dump's function `_extract_args` from SparseArrays' CHOLMOD wrapper.
- Bramble's side, owned by the fix subplan: `loss_no_rule` cannot stay an in-process
  `@test_throws`, since an abort ends the process. Move it into a child process as the 2D
  losses are (`_child_gradient`), assert the child's exit code and output, and pin it
  `@test_broken` with the upstream link, so a fixed Enzyme says so.
- The abort also needs no Bramble input: any user differentiating a sparse `A \ F` by Enzyme
  gets it. The documented route through `pde_solve` is unaffected.

## Commands run

All through `.claude/scripts/julia_slot.sh 3`, on a `git archive` copy of `0914c024` (so a
concurrent edit to `src/solvers/pde_solve.jl` in the tree could not move the result), with
the tree's `Manifest.toml` (Enzyme 0.13.218) and Enzyme, ChainRulesCore,
DifferentiationInterface and SuiteSparse loaded from a temporary environment:

```text
julia --project=<copy> --threads=3 --optimize=1 --check-bounds=<yes|auto> --code-coverage=<lcov file> <cases script>
```

The `cases script` had the test's losses, one `CASE` line each. The final reproducer was
rerun from the tree itself with `--project=.` under both flags: exit 134 both times, 15 s
(auto) and 16 s (yes), 0.78 and 0.75 GB peak memory footprint (`/usr/bin/time -l`).
