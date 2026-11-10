# #1124: Enzyme forward offset in the three-summand `assemble`

Reduction of gpena/Bramble.jl#1124 (Enzyme forward mode through `assemble` of
`inner₊(2.0 * Mₕ(κ) * ∇ₕ(U), ∇ₕ(V)) + innerₕ(κ * U, 0.5 * D₋ᵧ(V))` returns a gradient off by
one constant in every entry). The reproducer is `1124_forward_offset.jl`, next to this note.
It needs Bramble and nothing else from the repository.

Reproduces: `julia --project=. --threads=1 --optimize=1 --check-bounds=auto docs/notes/internals/ad_reductions/1124_forward_offset.jl` from the repository root exits 1 at 2e674be6 in 88 s under `CpuSerial`: the four Enzyme entries sit 249.0 above ForwardDiff's (the same 249.24 in each, max relative difference 2.5e-3) on the issue's 17 by 17 seeded non-uniform mesh. The variant `small_single` does the same on a 5 by 5 uniform mesh with an offset of 19.0 (87 s). On branch `v4-2-4` it no longer reproduces. Both exit 1 at 501128cb and 0 from 3aefc79f on (bisect below), under one Manifest with Enzyme 0.13.219, and every variant of the file exits 0 at e3fcca1d. The two `@test_broken` pins of `test/ext/enzyme_forward_assembly.jl` are now `@test`.

Fixed by: 3aefc79f perf(space): hold SpaceWeights by reference so the 3D form tree fits Enzyme's type analysis (#1154)

Cause: the tree was too wide for Enzyme's type analysis. Each `ScalarGridSpace` held its `SpaceWeights` inline (144 bytes a space, 152 a `VectorElement`), so in the `BilinearForm` (760 bytes) the `OperatorScale` scalar `0.5` of the third summand sat at byte 592, past Enzyme's `maxtypeoffset` of 512, the bound #1128 met in 3D. The bytes past that bound reach the derivative untyped, and the observed tangent `dA + A₃` is what a shadow of `0.5` equal to `0.5` gives, `d(0.5 A₃') = 0.5 dA₃' + 0.5 A₃' = dA₃ + A₃`. At 501128cb, `Enzyme.API.maxtypeoffset!(1024)` before the call makes `diffusion_single` exact (exit 0, 1.7e-15). After 3aefc79f a space is a 16-byte reference, the form 120 bytes, the scalar at byte 80. Past the bound is needed but not enough. `three` has its scalar at byte 744 at 501128cb and was exact, and `small_scalefirst` (byte 288) and `mass2` (byte 440) sit under it. What else the miscompile needs (the note found a fresh element from `2.0 * Mₕ(κ)` before the scaled summand) is not explained. Lowering the bound to 64 at e3fcca1d (scalar at byte 80) did not bring the offset back (exit 0, 20 s, so the compiled derivative may come from the precompile cache, untested). A next shape of this bug: a tangent off by a summand's own primal, on a form whose `sizeof` passes 512 bytes, which `maxtypeoffset!` removes.

Latest Enzyme: 0.13.218, which is the newest release the registry resolved today and the version the Manifest holds, so no newer release exists to try; the same command under a temporary environment (Bramble developed from this checkout, `Pkg.add` of Enzyme, ForwardDiff 1.4.7, Polyester 0.7.19, every other dependency re-resolved) still exits 1 with offset 249.0 in 84 s. The issue and the test comments say 0.13.216, which was not run.

Fix site: `SpaceWeights` in `src/space/scalar_gridspace.jl`, made a `mutable struct` with `const` fields by 3aefc79f, so a space holds it by reference and the form tree stays under the 512-byte bound; nothing changed in the assembly. Before the bisect the offset was placed in `_assemble_bilinear_core_cached!` (`src/assembly/bilinear_execution.jl`, from `_assemble_bilinear!`, `src/assembly/bilinear.jl`), the record and replay walk; that was the walk that first read the untyped scalar, not a fault in it. Of the three pins, the `@test_broken` lines of `test/ext/enzyme_forward_assembly.jl` are `@test` now. The `assemble_three` cell's `fwd = :wrong` in `test/space/ad_coverage.jl` and the open-bug paragraph and `open` cells of `docs/notes/internals/ad_coverage.md` are not this subplan's files and still need a rerun under the ad group.

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

- Written at 2e674be6, before the fix; the bisect below closes it. The Bramble-free analog
  above was a few bytes wide, so it stayed under the 512-byte bound, which may be why it
  was exact. An Enzyme report would need a struct past 512 bytes with a `Float64` scale beyond
  the bound, which was not built.
- The issue suspects the shadow of a `similar` on the matrix values. No `similar` was traced
  here, and the fix needed none.

## Bisect (v4-2-4)

Each commit in a detached temporary worktree, the branch's `Manifest.toml` (Enzyme 0.13.219)
copied in, the branch tip's reproducer, through `julia_slot.sh 1`, one Enzyme process at a
time, `--threads=1 --optimize=1 --check-bounds=auto`. Times include compile on a loaded
machine.

| commit | `diffusion_single` | `small_single` |
|---|---|---|
| 501128cb (planning base) | exit 1, offset 249.0, 170 s | exit 1, offset 19.0, 377 s |
| 3aefc79f `SpaceWeights` by reference | exit 0, 1.7e-15, 172 s | exit 0, 2.0e-16, 330 s |
| 2cca767f | exit 0, 226 s | exit 0, 162 s |
| 55fa1ea2 | exit 0, 61 s | exit 0, 56 s |
| d2480cd3 | exit 0, 58 s | exit 0, 54 s |

At 501128cb with `Enzyme.API.maxtypeoffset!(1024)` set before the reproducer is included,
`diffusion_single` exits 0 (1.7e-15, 92 s). The byte offsets above are `fieldoffset`s of the
`Float64` leaves of `typeof(form(W, W, ...))` at Float64, read at 501128cb and at the tip.

Every variant at e3fcca1d exits 0: `diffusion_` `single`, `batch`, `batch2`, `di`,
`reverse`, `contract`; `mass2_single`, `react2_single`; `small_` `single`, `di`, `contract`;
`small_diffonly_single`, `small_noscale_single`, `small_scalefirst_single`;
`three_uniform_single`, `three_uniform_di`, `three_nonuniform_single`. Largest relative
difference 2.2e-15.

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
