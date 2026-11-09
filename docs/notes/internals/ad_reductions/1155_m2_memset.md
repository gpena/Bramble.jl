# #1155: Enzyme `EnzymeNoTypeError` in the serial reverse of 3D `M₂`

Follow-up of "Failure two" in `1128_3d_notype.md`: `M₂` (the backward average along the third
axis) called on a grid function inside the differentiated function raises
`EnzymeNoTypeError`, `Cannot deduce type of memset`, in reverse mode under `CpuSerial`, on
uniform and non-uniform 5 by 5 by 5 meshes. `M₊₂` fails the same way. Forward mode, `Mₓ`,
`Mᵧ`, 2D `Mᵧ` and the centered `Mcₓ`, `Mcᵧ`, `Mc₂` give gradients. The threaded policies
reach the band through Enzyme rules and were never red.

Line: `src/operators/average.jl:178` before the fix, the boundary loop of `_average_band!`,
which stores `zero(cur)` at every point of the one boundary slice.

Why the third axis: LLVM's loop idiom pass turns the boundary loop into one `llvm.memset` per
contiguous row of the slice, for `Mᵧ` as for `M₂` (`code_llvm` of `_average_band!` shows one
memset for `DIM = 2` and `DIM = 3` in 3D and for `DIM = 2` in 2D, none for `DIM = 1`, whose
slice is strided). Enzyme types a memset only from the pointer it writes. Inlined into the
differentiated function, `out` is the `Memory` that `element(W, T)` allocated, a phi with
Julia's empty-memory global, and its data pointer is typed at offsets 0 and 8 alone
(`{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double}`), not at every offset. For `Mᵧ`
the slice is `(1:n₁, 1:1, 1:n₃)`, one outer induction variable, and the memset's offset is
`8n₁n₂ · iv` with `iv` known to start at 0 (intvals `{0}`), so Enzyme reads the type at
offset 0 and the memset is typed. For `M₂` the slice is `(1:n₁, 1:n₂, 1:1)`, whose outer
`CartesianIndices` loop runs over two indices with a carry. The offset is
`8n₁ · ((i₂ + 2⁶¹ − 1) + (i₃ + 2⁶¹ − 1) n₂)`, LLVM's form of `i − 1` before the multiply by
8. Enzyme's integer tracking gives `{2⁶¹}` for each term and nothing for the product, so
the offset is unknown, the base has no type at an unknown offset, and the memset is untyped.
Only the last axis gives a boundary slice with two outer extents above 1, so only `M₂` and
`M₊₂` fail. The difference operators were not traced again.

Fix: the boundary loop moved, unchanged, into `@noinline _average_boundary!(out, in_ref, li,
boundary, dir)`, which `_average_band!` calls. In its own function `out` is an argument, typed
by its Julia type at every offset, and the memset is typed. That is the condition under which
`_average_band!` differentiated on all three axes when called directly with `Duplicated`
arrays. The primal is the same loop (a literal `zero(cur)`, also at an infinite or `NaN` grid
value), with no allocation (`@allocated` 0 in a function) and no `jl_apply_generic` in
`code_llvm`. `_compute_average(…, Val(true), cur)`, which the KernelAbstractions kernels
share, and the `@simd` interior loop are untouched. The serial centered engine
(`_centered_average_engine!`) differentiates on all three axes as it is, and
`_centered_average_band!` runs only under the threaded policies, through rules, so neither
was changed.

Enzyme: 0.13.219 (Julia 1.13.3, macOS arm64, 2 threads, `--startup-file=no`); the IR and the
type analysis were read with `Enzyme.Compiler.VERBOSE_ERRORS[] = true` and, for the `Mᵧ`
call that passes, `Enzyme.API.printall!(true)`.

## Commands run

All through `.claude/scripts/julia_slot.sh 2 julia --startup-file=no --threads=2
--project=.` in the worktree, Enzyme from a temporary environment: `Enzyme.autodiff` of
`p -> sum(abs2, parent(op(u(p))))` for `op` in `M₂`, `Mᵧ`, `Mₓ`, `M₊₂`, `M₊ᵧ`, `Mc₂`, `Mcᵧ`,
`Mcₓ` (reverse, `set_runtime_activity`, `CpuSerial`, non-uniform 5 by 5 by 5 mesh, seed
1234); `Enzyme.autodiff` of `_average_band!` alone on `(5, 5, 5)` along each axis and on
`(9, 9)`; `code_llvm` of `_average_band!` for each axis and direction. The subplan's probe
then compares `M₂` with ForwardDiff on every policy, uniform and non-uniform.
