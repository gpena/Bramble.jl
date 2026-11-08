# #1124: Enzyme forward offset in the three-summand `assemble`

Reduction of gpena/Bramble.jl#1124 (Enzyme forward mode through `assemble` of
`inner₊(2.0 * Mₕ(κ) * ∇ₕ(U), ∇ₕ(V)) + innerₕ(κ * U, 0.5 * D₋ᵧ(V))` returns a gradient off by
one constant in every entry). The reproducer is `1124_forward_offset.jl`, next to this note.
It needs Bramble and nothing else from the repository.

Reproduces: `julia --project=. --threads=1 --optimize=1 --check-bounds=auto docs/notes/internals/ad_reductions/1124_forward_offset.jl` from the repository root exits 1 at 2e674be6 in 88 s under `CpuSerial`: the four Enzyme entries sit 249.0 above ForwardDiff's (the same 249.24 in each, max relative difference 2.5e-3) on the issue's 17 by 17 seeded non-uniform mesh. The variant `small_single` does the same on a 5 by 5 uniform mesh with an offset of 19.0 (87 s). The offset still exists, so the page's open-bug paragraph and the two pins stay as they are.

Latest Enzyme: 0.13.218, which is the newest release the registry resolved today and the version the Manifest holds, so no newer release exists to try; the same command under a temporary environment (Bramble developed from this checkout, `Pkg.add` of Enzyme, ForwardDiff 1.4.7, Polyester 0.7.19, every other dependency re-resolved) still exits 1 with offset 249.0 in 84 s. The issue and the test comments say 0.13.216, which was not run.

Fix site: not placed to a line. The offset enters in `_assemble_bilinear_core_cached!` (`src/assembly/bilinear_execution.jl:595`, called from `_assemble_bilinear!`, `src/assembly/bilinear.jl:1307`), the record and replay walk that routes the summands into the matrix. The form's tree reads exactly: the stored `κ` of the third summand and the matrix-free `a(u, v)` over the same tree both match ForwardDiff to 1e-15. No Bramble-free case reproduced it, so the report to Enzyme has no body yet, and whether the fault is Bramble code Enzyme cannot differentiate or an Enzyme miscompile of it is open. When a fix lands, three pins change. The `@test_broken` lines of `test/ext/enzyme_forward_assembly.jl` become `@test`, the `assemble_three` cell's `fwd = :wrong` in `test/space/ad_coverage.jl` goes (see "What does not reproduce"), and the open-bug paragraph and `open` cells of `docs/notes/internals/ad_coverage.md` change.

Compile: 88 s and 1.4 GB peak footprint for the default variant (`/usr/bin/time -l`; the maximum resident set was 1.4 to 1.7 GB in most runs and 4.4 GB in one), 87 s for `small_single`.

Enzyme: 0.13.218 (Julia 1.13.3, macOS arm64, 1 thread, `--optimize=1`, `--check-bounds=auto`).

## What the offset is

On the 5 by 5 uniform mesh the offset is 19.0 in each entry. With `A` the assembled form, `A₃`
the assembled third summand alone, `innerₕ(κ * U, 0.5 * D₋ᵧ(V))`, and `A₁₂` the first two, in
a plain Float64 run at the same `p`:

| quantity | value |
|---|---|
| observed offset, every entry | 19.0 |
| `2 <A, A₃>` | 18.989 |
| `2 <A, A₁₂>` | 999.35 |
| `2 <A, A>` | 1018.34 |

The functional is `sum(abs2, nonzeros(A))`, whose tangent is `2 <A, dA>`. So the wrong tangent
is `dA + A₃`: the primal values of the last, scaled summand are added to its tangent, once,
whatever the seed. That explains the seed independence and why the issue saw different sizes
from run to run (the offset follows the mesh draw, which the issue did not seed). The test
comment's reading, a wrong shadow "before assembly runs", was not reproduced: a function that
returns a sum over the stored `κ` of the third summand, or over its scale times that sum,
read off `_summands(form.ast)`, gives an exact tangent (no `@noinline` reader was tried).

## Reduction

Every row is one `Enzyme.autodiff` of a closure built inside a builder function (as the tests
do), `set_runtime_activity(Forward)`, `CpuSerial`, compared with ForwardDiff on the same
mesh object.

| Differentiated function | Outcome |
|---|---|
| the issue's form, 17 by 17 non-uniform, `single` and `di` | offset 249.0 |
| the same, 5 by 5 uniform (`small`) | offset 19.0 |
| the first summand pair alone (`small_diffonly`) | exact |
| the last summand unscaled (`small_noscale`) | exact |
| the scaled summand first (`small_scalefirst`) | exact |
| the scaled summand written outside, `... + 0.5 * innerₕ(κ * U, D₋ᵧ(V))` | offset 19.0 |
| `inner₊(κ∇U, ∇V) + innerₕ(κU, V)` (`plain3`), no scale | exact |
| the matrix-free `a(u, v)` of the issue's form (`small_contract`) | exact |
| the issue's form as top-level code with non-`const` globals in `f` | exact (the miscompile follows how `f` compiles) |
| `inner₊(κ∇U, ∇V) + innerₕ(κU, V) + innerₕ(κ U, 0.5 D₋ᵧ(V))` (`three_uniform`, `three_nonuniform`) | exact |
| a Bramble-free tree of the same shapes (see below) | exact |

So the offset needs a scaled summand last, after two products whose grid function is the
result of `2.0 * Mₕ(κ)` computed inside the differentiated function. The list `_summands`
returns for it is `(BilinearProduct, BilinearProduct, OperatorScale)`, the first two holding
a fresh `VectorElement`, the last holding `κ` itself. The rows `plain3` and `three` have no
fresh element in a summand before the scaled one. 
A Bramble-free analog was tried: `Elem`, `GridFunctionScale`, `BilinearProduct`,
`OperatorScale` and `OperatorAdd` as one-field structs, `_summands` as the tuple splat, the
accumulation through `map` over the tuple, two summands holding a vector built inside and the
last one scaled and holding the input. It differentiates exactly. It leaves out the
sparse-matrix record and replay (`_keep_pattern!`, `_scatter_add!`), which is where the
reduction stopped, so the table says only that the tree shape and the `map` are not enough.

## What does not reproduce

- `assemble_three` of `test/space/ad_coverage.jl` (the page's three-summand cell, relative
  error 0.17 uniform and 0.10 non-uniform, pinned `:wrong`) is exact here: `three_uniform`
  by `single` and by `di`, and `three_nonuniform` by `single`, all CpuSerial. Its form has no
  `Mₕ` product before the scaled summand. If the suite's cell still says `:wrong` under
  this Manifest, the cell and the issue describe different failures: the issue's form is the
  one `enzyme_forward_assembly.jl` pins, and the cell needs a rerun under the ad group.
- Threads above one, `CpuThreaded` and `CpuPolyester`, reverse mode, and the 0.13.216
  release were not run.

## For the fix subplan

- The offset is real at this commit, so the work is still open: a reduction to the record and
  replay walk (`_assemble_bilinear_core_cached!`, `_keep_pattern!`) is the next step, by
  making the Bramble-free analog write into a `SparseMatrixCSC` through recorded positions.
  Until that case reproduces, an Enzyme report has no standalone body.
- `diffusion_single` is the quick check of a candidate fix (88 s); `small_single` is the
  smaller (87 s, offset 19.0). A fix that works must make both exit 0 and keep
  `small_contract` exact.
- The issue suspects the shadow of a `similar` on the matrix values. No `similar` was traced
  here, so that reading is open.

## Commands run

All through `.claude/scripts/julia_slot.sh 1`, one Enzyme process at a time, in the worktree
`~/worktrees/Bramble/v4-2-2` at 2e674be6 (with others' uncommitted edits to
`src/mesh/constructors.jl`, `benchmark/assembled_jacobian_workload.jl` and
`test/ext/forwarddiff_ext.jl`, none in the assembly), the tree's `Manifest.toml` (Enzyme
0.13.218) and Enzyme, ForwardDiff and DifferentiationInterface from a temporary environment:

```text
julia --project=. --threads=1 --optimize=1 --check-bounds=auto \
    docs/notes/internals/ad_reductions/1124_forward_offset.jl <variant>
```

The variants run in the final file: `diffusion_single` (exit 1), `small_single` (1),
`small_contract` (0), `small_noscale_single` (0), `three_uniform_single` and
`three_uniform_di` (0), `three_nonuniform_single` (0), and `diffusion_di` (1). The latest-Enzyme
run is `diffusion_single` with `--project=<temporary environment>`. The rows `small_diffonly`,
`small_scalefirst`, the scaled-outside form and `plain3`, the reader rows, the formula check and
the Bramble-free analog came from scratch scripts with the same closure shape, run at
`--check-bounds=auto`; the reproducer holds `small_diffonly` and `small_scalefirst` as
variants, but they were not run through it. The issue's call as a script with global `W` and
`B` ran exact at `--check-bounds=auto` and `yes`, with `Duplicated` and `BatchDuplicated`,
and through DI, before the closure builders were used.
