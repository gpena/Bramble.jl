# #1128: Enzyme `EnzymeNoTypeError` in the 3D `Mₕ(κ)` assembly

Reduction of gpena/Bramble.jl#1128 (Enzyme through `assemble` of
`inner₊(Mₕ(κ) * ∇ₕ(U), ∇ₕ(V))` on a 5 by 5 by 5 mesh raises `EnzymeNoTypeError`, in reverse
and in forward mode). The reproducer is `1128_3d_notype.jl`, next to this note. It needs
Bramble and nothing else from the repository.

Reproduces: `julia --project=. --threads=1 --optimize=1 --check-bounds=auto docs/notes/internals/ad_reductions/1128_3d_notype.jl` from the repository root exits 1 with `EnzymeNoTypeError` at 60cb3acd (after #1126's fix, 54ff4c57), in 168 s under `CpuSerial`; the argument `averaged forward` does the same in 76 s. The #1126 fix did not touch it: this cause is not the run-time tuple index of #1126.

Fix site: `Enzyme` upstream, the type analysis bound `Enzyme.API.maxtypeoffset!` (default 512 bytes) on the `memcpy` of the form's expression tree in the `BilinearForm` constructor (`src/assembly/bilinear.jl:249`, reached from `form`); the Bramble-side route is to raise that bound in `ext/BrambleEnzymeExt.jl`, or to shrink the tree below it. A second, separate failure, `M₂` called on a grid function in 3D, sits in `src/operators/average.jl`, function `_average_band!` (boundary loop, line 178).

Compile: 168 s and 3.0 GB peak (maximum resident set; 2.5 to 3.9 GB over three runs) for the reproducer up to the error; 366 s and 3.2 GB for the same call with `maxtypeoffset!(1024)`, which differentiates, and 235 s and 1.8 GB for its forward mode.

Enzyme: 0.13.218 (Julia 1.13.3, macOS arm64, 1 thread, `--optimize=1`, `--check-bounds=auto`).

## What fails

All runs are under `CpuSerial`, on a 5 by 5 by 5 non-uniform mesh (seeded), through
`Enzyme.autodiff` of a scalar function of three parameters, `set_runtime_activity`. The
coefficient `κ` is a grid function holding `2 + B p`, as the coverage test's `_field` does.
A gradient was checked against a central difference (at most 6e-9 relative); the reproducer
prints that figure. Uniform meshes and the threaded policies were not run.

| Differentiated function, 3D | Reverse | Forward |
|---|---|---|
| `assemble` of `inner₊(Mₕ(κ) * ∇ₕ(U), ∇ₕ(V))` (the issue) | `EnzymeNoTypeError`, `memcpy`, 168 s | `EnzymeNoTypeError`, `memcpy`, 76 s |
| `assemble` of `inner₊(κ * ∇ₕ(U), ∇ₕ(V))` | gradient, 168 s | gradient, 71 s |
| `assemble` of the linear form `innerₕ(κ, v)` | gradient, 11 s | gradient, 10 s |
| `D₋ₓ`, `D₋ᵧ`, `D₋₂` on `κ` | gradient, 9 s | gradient, 9 s |
| `Mₓ`, `Mᵧ`, `M₂` on `κ` | `EnzymeNoTypeError`, `memset`, 9 s | gradient, 9 s |
| `Mₓ` alone, `Mᵧ` alone | gradient | not run |
| `M₂` alone | `EnzymeNoTypeError`, `memset`, 9 s | not run |
| a form with one `Mₓ(κ) * D₋ₓ(U)` term | gradient, 37 s | not run |
| a form with the `x` and `y` terms of that kind | gradient, 92 s | not run |
| a form with those two and a third, `Mₓ(κ) * D₋₂(U)` | `EnzymeNoTypeError`, `memcpy`, 163 s | not run |
| a form with three terms, three separate plain coefficients | gradient, 160 s | not run |

The same issue call in 2D (`averaged2d`, 9 by 9) gives a gradient in 67 s, and `Mᵧ` alone
in 2D gives a gradient.

For the 3D cells S4 pins (CpuSerial only), the states are: the 3D linear form, the 3D
`inner₊(κ∇U, ∇V)` and the difference operators work in both modes. `Mₕ(κ)` in a form fails
in both modes until the first site below is fixed. The averaging operators on a grid
function work in forward and fail in reverse because of `M₂`.

## Failure one: the form with three averaged terms

The error's `Cannot deduce type of copy` line is a 256-byte `memcpy`: the stack copy of a
`ScalarGridSpace` (32 words in 3D) that the `BilinearForm` constructor makes with the
expression tree's `VectorElement`s. The tree in 3D is an `OperatorAdd` of three
`BilinearProduct`s, 792 bytes, and Enzyme's type analysis stops at an offset of 512 bytes by
default. In 2D the sum has two terms. What the table's rows say:

- Two averaged terms pass and three fail (`formxy`, `formxyx`), with no `M₂` in the third.
- Three plain coefficients pass (`formthree`), and so does `κ * ∇ₕ(U)` with the same `κ`
  three times: that tree has the same 792 bytes, and its three elements are one value. The
  failing tree holds three elements that are each a separate `Mₓ(κ)`, `Mᵧ(κ)`, `M₂(κ)`
  result.
- `Enzyme.API.maxtypeoffset!(4096)` makes `formxyx` and `averaged` differentiate, and
  `maxtypeoffset!(1024)` makes `averaged` differentiate in reverse (366 s, 3.2 GB) and in
  forward (235 s, 1.8 GB). The default is 512, so the least passing value is between 513 and
  1024; it was not searched.

The cause is the offset bound: raising it removes the error. This reduction did not read
Enzyme's type analysis to say which aggregate's offset exceeds 512, nor why three distinct
elements and not three equal ones reach it.

## Failure two: `M₂` on a grid function, reverse

The averaging operator `M₂` (the third axis), called on a grid function inside the
differentiated function, raises `Cannot deduce type of memset` in reverse. The instruction is
an `llvm.memset` of zero bytes, which the loop idiom pass makes of the boundary loop of
`_average_band!` (`src/operators/average.jl:178`), where `_compute_average` writes
`zero(cur)`, a constant, to every point of the one boundary slice. `Mₓ` and `Mᵧ` give a
gradient, and so does `Mᵧ` in 2D, although its slice is as contiguous as `M₂`'s. `D₋ₓ`,
`D₋ᵧ` and `D₋₂` give gradients too, though `_difference_band!` has the same boundary loop
with `zero(cur)`. Why only `M₂` fails was not found.

The reproducer's `fixed_avgz` redefines the boundary value as `cur - cur`, which is no
constant, so the loop stays a loop, and `avgz` then differentiates. That is a probe: it gives
`NaN` where the grid value is infinite or `NaN`, so it is not the change Bramble should make.
`--optimize` does not matter (0, 1 and 2 all fail), since the IR is Enzyme's own.

The form of the issue does not raise it: `averaged` with `maxtypeoffset!(1024)` and no probe
differentiates, so the averages the form evaluates did not meet the `memset`. Which route
they take was not traced. Failure two is a second 3D finding for S4 (the `M₂` cell), and not
a cause of #1128's call.

## For the fix subplan

- #1128's acceptance criterion (3D `Mₕ(κ)` matches ForwardDiff in both modes) needs only the
  first site. The options are an upstream Enzyme issue (gate item G3) asking the type analysis
  to type a `memcpy` of an aggregate at an offset past `maxtypeoffset`, or a Bramble change
  that sets `Enzyme.API.maxtypeoffset!` (a process-wide setting of the user's Enzyme) from
  `BrambleEnzymeExt`'s `__init__`, or a change that shrinks the form's tree. Which of these is
  right is the fix subplan's decision; the reproducer's `BRAMBLE_REPRO_TYPEOFFSET` tests a
  value. The slowdown from raising the bound was not isolated: `averaged` takes 366 s with
  it, and its compile at the default stops at the error after 168 s.
- The `M₂` failure has a Bramble-side fix in `_average_band!`: write the boundary zeros so
  that LLVM does not turn them into a `memset`, with the exactness of a literal zero. The
  alternative is an Enzyme report about the `memset` type deduction.
- The test cell `assemble_averaged_3d` in `test/space/ad_coverage.jl` expects `:throws` with
  `EnzymeNoTypeError` for `CpuSerial` in both modes, which is still right. Threaded 3D
  cells are `:skip` there and were not measured here.

## Commands run

All through `.claude/scripts/julia_slot.sh 1`, in the worktree at 60cb3acd with the tree's
`Manifest.toml` (Enzyme 0.13.218) and Enzyme from a temporary environment:

```text
julia --project=. --threads=1 --optimize=1 --check-bounds=auto \
    docs/notes/internals/ad_reductions/1128_3d_notype.jl <variant> <mode>
```

One process per row (one Enzyme compile at a time); `BRAMBLE_REPRO_VERBOSE=1` was set for the
runs whose error text is quoted. Times are wall clock and sizes are maximum resident set from
`/usr/bin/time -l`. The `maxtypeoffset!` runs set `BRAMBLE_REPRO_TYPEOFFSET`.
