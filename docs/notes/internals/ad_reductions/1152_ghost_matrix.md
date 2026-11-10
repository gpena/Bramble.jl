# #1152: the ghost `Δₕ` matrix cell under Enzyme

Reduction of gpena/Bramble.jl#1152: the six cells of `ghost_assemble` in test/space/ad_coverage.jl (`build_ghost_assemble`: Neumann and Robin sides, `p[1] * innerₕ(-Δₕ(U), V) + p[2]^2 * innerₕ(U, V)`, `assemble(a, l)`, the 7 by 6 non-uniform ghost mesh), reverse and forward under `CpuSerial`, `CpuThreaded` and `CpuPolyester`. The reproducer is `1152_ghost_matrix.jl`, next to this note. It runs one policy and one mode per process, with the cell's space and loss, against ForwardDiff on the same policy. Its command is in the last section. The tree under test is e3fcca1d, which holds S2.1's `SpaceWeights` by reference (3aefc79f).

Cell: CpuSerial rev: fails EnzymeNoTypeError, "Cannot deduce type of copy" (a 32-byte `llvm.memcpy` of a `ScalarGridSpace` piece, in `_replay_segment!`)

Cell: CpuThreaded rev: ok 8.5e-16

Cell: CpuPolyester rev: ok 8.5e-16

Cell: CpuSerial fwd: fails EnzymeNoTypeError, same message, same frame

Cell: CpuThreaded fwd: ok 8.5e-16

Cell: CpuPolyester fwd: ok 8.5e-16

Fix site: src/assembly/bilinear_execution.jl, line 138 (`_replay_segment!`) is the frame Enzyme prints, variants `CpuSerial rev` and `CpuSerial fwd`. With `Enzyme.Compiler.VERBOSE_ERRORS[] = true` the stack is `_visit_stencil!` (src/assembly/bilinear_traversal.jl, line 1416, the `_each_piece(...) do piece` of the interior box), called from `_replay_segment!` at line 160, the `ReplaySink` branch. The value Enzyme cannot type is a 32-byte piece of a `ScalarGridSpace` that SROA split out of the walk space `wsp = _walk_space(sp, TERM)` (line 1409): `_walk_space(::ScalarGridSpace{D, T, VT, <:GhostMesh})` (line 1119) rebuilds the space around the mesh's walk state with `sp.weights`, which since 3aefc79f is a pointer to a `SpaceWeights` and no longer inline data. This is the candidate; it is not probed.

Fix site: the regression is 3aefc79f (S2.1, `SpaceWeights` becomes a `mutable struct`). The same cell at 501128cb, its parent, is `ok` (variant `CpuSerial rev`, 139.3 s, relative error 5.1e-16, below), and at 3aefc79f it fails at the same frame (116.4 s). The failure is not in `_ghost_maps` or `_ghost_bilinear_form`, so the issue's split of the integer index structure from the values does not apply to it. A reverse that passes under `CpuSerial` in the test file at 2026-11-07 (366 s, 4.8 GB) now fails here, so the `:ok` for `ghost_assemble` under `CpuSerial` in test/space/ad_coverage.jl is stale at this tree; the other ghost cells that walk a `GhostMesh` space (`ghost_inner`, `ghost_load`) were not run here.

Compile: time runs from the reference to the end of the Enzyme call, compile included (reverse mode compiles inside the first call of `measure`, before its `enzyme start` step). Memory is the maximum resident set size of the whole process (`/usr/bin/time -l`), an upper bound of the footprint; `julia_slot.sh` kills at 10 GB of footprint and never did. Figures per cell follow, one to a line, in the order of the cells above.

- Reverse, CpuSerial, 120 s and 2.62 GB, failed.
- Reverse, CpuThreaded, 169 s and 2.70 GB, ok.
- Reverse, CpuPolyester, 153 s and 3.29 GB, ok.
- Forward, CpuSerial, 84 s and 2.27 GB, failed.
- Forward, CpuThreaded, 153 s and 2.31 GB, ok.
- Forward, CpuPolyester, 115 s and 2.77 GB, ok.

No cell reached the 25 minute bound, and no passing cell took over 10 minutes or 10 GB, so the issue's third criterion holds for every cell that differentiates. The control at 501128cb took 139 s; its 5.2 GB includes a precompile of the unpacked tree, so it is not a figure of the compile.

Enzyme: 0.13.219 (Julia 1.13.3, macOS arm64, 2 threads, `--optimize=1`, `--check-bounds=auto`, `--startup-file=no`), loaded from a temporary environment holding DifferentiationInterface, ForwardDiff and Polyester, with `AutoEnzyme(; mode = Enzyme.set_runtime_activity(m), function_annotation = Enzyme.Const)`. Default `maxtypeoffset`, no global Enzyme setting.

## What the cells say

- All four threaded cells pass at 8.5e-16 against ForwardDiff, and reverse and forward agree with each other to 1e-15 (reverse `[5.468279682833906e6, 427.6934679410291, 52280.091065894354]`, reference `[5.468279682833902e6, 427.69346794102904, 52280.09106589434]`, every component nonzero).
- Both `CpuSerial` cells fail with `EnzymeNoTypeError`, forward as reverse, at the same frame: forward after 84 s, reverse after 120 s.
- The cause is the layout change of 3aefc79f: before it (501128cb, the parent) the serial reverse compiles and returns 5.1e-16 against ForwardDiff in 139 s. Why the threaded policies pass is not probed: they replay through their own band loops (`bilinear_execution.jl`, `ext/BramblePolyesterExt.jl`).
- This also means S2.1's goal, the 3D `Mₕ(κ)` form fitting Enzyme's type analysis, costs the 2D ghost serial cell. A fix subplan for `CpuSerial` starts at `_walk_space` (line 1119) and `_visit_stencil!` (line 1416). Neither is probed.

## Commands run

Each cell is one process, run in order through `.claude/scripts/julia_slot.sh 2`, by a driver in `/tmp/bramble-1152` (outside the repository) that times it with `/usr/bin/time -l` and stops it at 25 minutes. Variant `CpuSerial rev` also ran with `BRAMBLE_VERBOSE_ERRORS=1` for the stack above, and at the two commits above from a `git archive` of each in `/tmp/bramble-1152` with the root Manifest copied in. The note's repository commands run from the repository root:

```text
julia --startup-file=no --project=. --threads=2 --optimize=1 --check-bounds=auto \
    docs/notes/internals/ad_reductions/1152_ghost_matrix.jl <policy> <rev|fwd>
```

`<policy>` is CpuSerial, CpuThreaded or CpuPolyester. Prefix the line with `.claude/scripts/julia_slot.sh 2` and with `/usr/bin/time -l` for the figures. The script exits 0 for `ok` and 1 otherwise, and prints one `RESULT` line.
