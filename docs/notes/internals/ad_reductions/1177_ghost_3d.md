# #1177: the 3D ghost `Δₕ` cell aborts Enzyme's reverse in LLVM verification

Reduction of gpena/Bramble.jl#1177 (with #1173). Enzyme reverse of the ghost `Δₕ` assembly on a (4, 4, 3) non-uniform mesh with ghosts `(:xmin, :zmax)`, under `CpuSerial`, ends the process with `Instruction does not dominate all uses!`, `LLVM ERROR: function failed verification (4)` and signal 6. The reproducer is `1177_ghost_3d.jl`, next to this note: one step, one cell and one mode per process, checked against ForwardDiff on the same loss. Its command is in the last section. The tree under test is 8a29f783.

Enzyme: 0.13.219, 2026-11-12, macbook (Apple M2, 16 GB, macOS arm64, Julia 1.13.3, default `-O2`, `--startup-file=no`), 1 thread

Each step ran as one process under `julia_slot.sh 1` and `/usr/bin/time -l`. Seconds are the process's `real` time: loading, the ForwardDiff reference and the compile all count, the wait for a slot does not. Peak is the peak memory footprint from `/usr/bin/time -l`. Other sessions were running Julia at the same time (load average 2 to 29), so treat the seconds as rough. A status of `ok` carries the relative error against ForwardDiff; `abort` names the function the verifier printed. `proto flat` and `proto flat_replay` are the prototype of the fix, set through `BRAMBLE_PROTO` (see "Reduction").

Step: cell, 3d | rev | abort in `diffejulia__replay_segment_` | 101 | 1.89

Step: cell, 3d | fwd | ok 1.5e-16 | 125 | 2.01

Step: cell, 2d (control) | rev | ok 1.7e-16 | 108 | 2.14

Step: record, 3d (pattern pass, then replay) | rev | abort in `diffejulia__replay_segment_` | 30 | 1.13

Step: replay, 3d (`_replay_bilinear_core!`, segments recorded outside) | rev | abort in `diffejulia__replay_segment_` | 27 | 1.09

Step: replay_segment, 3d (`_replay_segment!` alone) | rev | abort in `diffejulia__replay_segment_` | 24 | 1.03

Step: replay, 2d (control) | rev | ok 9.2e-14 | 21 | 1.01

Step: replay_segment, 2d (control) | rev | ok 9.2e-14 | 20 | 0.98

Step: pattern, 3d (pattern pass without replay) | rev | ok 0.0 | 25 | 1.06

Step: coord, 3d (`_form_coordinates`) | rev | ok 0.0 | 24 | 1.08

Step: coord_walk, 3d (`_coord_walk!` alone) | rev | ok 0.0 | 22 | 1.06

Step: replay_segment, 3d, proto flat | rev | ok 5.9e-15 | 27 | 1.09

Step: replay, 3d, proto flat | rev | ok 5.9e-15 | 29 | 1.15

Step: record, 3d, proto flat | rev | ok 5.9e-15 | 34 | 1.21

Step: record, 3d, proto flat_replay | rev | ok 5.9e-15 | 33 | 1.19

Step: cell, 3d, proto flat_replay (replay loop fixed, pattern loop not) | rev | abort in `diffejulia__coord_walk_` | 101 | 1.97

Step: cell, 3d, proto flat | rev | ok 1.5e-16 | 194 | 2.85

Step: control_nest, 3d (no Bramble) | rev | ok 2.4e-16 | 15 | 0.92

Step: control_slabs, 3d (no Bramble) | rev | ok 5.2e-17 | 14 | 0.91

Function: `diffejulia__replay_segment_`, the reverse pass of `_replay_segment!` (src/assembly/bilinear_execution.jl line 138), into which `_visit_stencil!` (the `_RowRestricted` method), `_visit_pieces!` and `_visit_interior!` are inlined. The instruction the verifier names is `%_unwrap811 = sub i64 %1050, %1054`, with `%1050 = load i64, ptr %1049, align 8, !invariant.group !69597` and `%1054 = load i64, ptr %1053, align 8, !invariant.group !69598`; both loads come after the `sub` in the same block (step `cell`, 3d, rev; the value numbers change from run to run, the shape stays the same). Once the replay is fixed, the hidden one is `diffejulia__coord_walk_`, the `_RowRestricted` method of `_coord_walk!` (src/assembly/bilinear_pattern.jl line 253): `%_unwrap429 = sub i64 %356, %360`, the same shape (step `cell, 3d, proto flat_replay`).

Fix site: src/assembly/bilinear_traversal.jl, line 1401 (`_visit_pieces!`, the replay) and line 1384 (`_each_piece`, which runs the `do piece` closures of `_coord_walk!` at bilinear_pattern.jl lines 268, 273 and 279): the loop `for J in CartesianIndices(Base.tail(region.indices))`. On a 3D grid that loop is two nested loops whose trip counts come from the ranges of the region, and the region changes on each pass of the loop over the boundary slabs. Walking the tail as one loop over its linear index, `tail = CartesianIndices(Base.tail(region.indices)); for k in 1:length(tail); J = tail[k]`, makes the whole 3D cell differentiate, 1.5e-16 against ForwardDiff (step `cell, 3d, proto flat`). Both methods need it. With only `_visit_pieces!` changed, the abort moves to `_coord_walk!`. The prototype redefines the two methods from the script (`BRAMBLE_PROTO=flat`), so no file under `src/` changed. The wrong IR is Enzyme's, but no Bramble-free nest reproduced it (`control_nest`, `control_slabs`), so it is not filed upstream.

## What the verifier names

The two loads read Enzyme's cache of two `i64` phis of the slab loop. In the forward pass they are the stop and the start of the region's second range: `phi i64 [ %"sp::_RowRestricted.mesh.indices.sroa.4.0.copyload", … ], [ %.sroa.7125.0.copyload, … ]` and `phi i64 [ 1, … ], [ %.sroa.6124.0.copyload, … ]`. The cache is indexed by the slab loop's induction variable (`iv6`). In the reverse pass Enzyme recomputes `stop - start + 1`, the trip count of the tail's inner loop, to index a cache of the loop nest below it (`iv8 * (stop - start + 1) + iv10`). It emits that subtraction before it emits the two loads, so the `sub` uses values its block has not defined yet. The verifier stops at the first bad function, which is how `_coord_walk!` stayed hidden behind `_replay_segment!` after 9faf6345, as the issue suspected.

In 2D the tail of a region is one range, so no cache inside the slab loop has a second dimension whose extent changes per slab. That is why the 2D cells pass, and why one loop over the tail's linear index removes the abort in 3D.

## Reduction

From the whole cell down. `record`, the first fill of the padded form (the pattern pass, then the replay), aborts in 30 s instead of 101 s. `replay` and `replay_segment` keep the abort with the pattern pass taken out, its segments recorded outside the differentiated function. `pattern`, `coord` and `coord_walk` pass on their own. Their loss is `sum(abs2, p)` times an integer sum, so these steps only show that Enzyme compiles those calls in that setting. They do not clear `_coord_walk!`, which aborts in the cell once the replay is fixed. With the replay fixed, `record` passes as well (`record, 3d, proto flat_replay`), so the `_coord_walk!` abort needs the whole cell. The reverse of `_coord_walk!` that Enzyme builds there differs from the one it builds for these steps.

Forward mode, never run on this cell before, differentiates it (1.5e-16).

The two Bramble-free controls copy the walk's lines as plain vectors and run `_each_piece`'s nest: `control_nest` alone, and `control_slabs` inside a loop over a tuple of the interior box and six boundary slabs, with each point adding to a vector. Both differentiate, so the abort needs more of Bramble's walk than this nest.

## Closure sizes against the 512-byte hypothesis

The planned first hypothesis was 9faf6345's: a `do` block that holds the walk space by value pushes a field past Enzyme's 512-byte type offset. Step `sizes` prints the layouts (`julia_slot.sh 1`, no Enzyme call):

- `_coord_walk!`'s two `do piece` closures are 72 bytes in 3D and 64 bytes in 2D. They hold the space `sp` (16 bytes: the `ScalarGridSpace` stores its weights by reference) and the sink by reference, not the walk space.
- The walk space `wsp = _walk_space(sp, TERM)` that `_visit_stencil!` passes to `_visit_pieces!` is 984 bytes in 3D and 688 bytes in 2D. Both are over 512, and 2D passes. Since 9faf6345 it is an argument, not captured.

Size does not separate the cells. The failure is also of another kind, an LLVM dominance abort where #1152 had `EnzymeNoTypeError` ("Cannot deduce type of copy"). A fix by analogy with 9faf6345, removing the `do` blocks of `_coord_walk!`, would not touch the loop that breaks. The plan rejected that route, and these numbers agree.

## For the fix subplan

- Change the tail loop of both `_each_piece` (line 1382) and `_visit_pieces!` (line 1398) in src/assembly/bilinear_traversal.jl. The prototype is in the script under `BRAMBLE_PROTO == "flat"`; it keeps the visit order (`CartesianIndices` linear order is column-major, the same order as the nested loop), so the coordinates of the pattern pass and the replay still line up.
- `_each_piece` is also used by `_each_walked` (bilinear_traversal.jl line 1367) and by the matrix-free walk (src/assembly/matrix_free.jl lines 828 and 1645). The prototype changes those too, and only the assembly cells were run here.
- Run `cell 3d rev` without `BRAMBLE_PROTO` after the change. With the fix in `src/`, the prototype methods redefine identical code, so `proto flat` adds nothing.
- The 3D cell compiled and differentiated in 194 s at 2.85 GB with the prototype, against 108 s and 2.14 GB for the 2D cell.

## Commands run

Each step is one process, run in order through `.claude/scripts/julia_slot.sh 1` with `/usr/bin/time -l`, by a driver in `/tmp/s51` (outside the repository) that writes one log per step. The verifier's message and the bad function's IR go to stderr. From the repository root:

```text
BRAMBLE_PROTO=<flat|flat_replay|unset> .claude/scripts/julia_slot.sh 1 /usr/bin/time -l \
    julia --startup-file=no --project=. --threads=1 \
    docs/notes/internals/ad_reductions/1177_ghost_3d.jl <step> <3d|2d> <rev|fwd>
```

`<step>` is one of `cell`, `record`, `pattern`, `coord`, `coord_walk`, `replay`, `replay_segment`, `control_nest`, `control_slabs` or `sizes`. The script prints one `RESULT` line and exits 0 for `ok` and 1 otherwise. An abort prints no `RESULT` line and exits 134. To find the function, grep the log for `^define internal fastcc void @diffe`; the instruction is in the lines after `Instruction does not dominate all uses!`.
