# #1157: the ghost load cell under Enzyme

Reduction of gpena/Bramble.jl#1157: the five unmeasured cells of `ghost_load` in test/space/ad_coverage.jl (reverse under `CpuThreaded` and `CpuPolyester`, forward under all three policies), with `CpuSerial` reverse as a control. The reproducer is `1157_ghost_load.jl`, next to this note. It runs one policy and one mode per process, with the cell's space and loss, against ForwardDiff on the same policy. Its command is in the last section.

Cell: CpuSerial rev: ok 1.2e-16

Cell: CpuThreaded rev: fails EnzymeNoDerivativeError, "No create nofree of empty function (ijl_in_threaded_region)"

Cell: CpuPolyester rev: fails EnzymeNoDerivativeError, "No create nofree of empty function (jl_set_precompile_field_replace)"

Cell: CpuSerial fwd: ok 3.5e-16

Cell: CpuThreaded fwd: ok 3.5e-16

Cell: CpuPolyester fwd: ok 3.5e-16

Fix site: src/assembly/ghost_assembly.jl, line 670 (`_assemble_padded!(a)` in `_assemble_ghost!`) is the frame Enzyme prints, variant `CpuThreaded rev`. The unhandled call sits in `_assemble_padded!(::GhostBilinearForm)` (line 684), in its `_ThreadedReplay()` branch (line 692), which the walk takes because it is not whole. Under `CpuThreaded` that branch reaches the `Threads.@threads :static` loops of src/assembly/bilinear_execution.jl (lines 995, 1014 and 1053). Those are candidates for `ijl_in_threaded_region`, not probed.

Fix site: src/assembly/ghost_assembly.jl, line 670 and the same function and branch (lines 684 and 692), variant `CpuPolyester rev`. Under `CpuPolyester` the branch calls `_batch_bilinear_band_replay!` through `_late` (src/utils/linear_algebra.jl, line 196), into the `Ref` method at src/assembly/bilinear_execution.jl (lines 1278 to 1284). BramblePolyesterExt defines the method that runs (ext/BramblePolyesterExt.jl, line 1098), and the Enzyme rules are in ext/BrambleEnzymePolyesterExt/assembly.jl (lines 282 and 293). Two candidate homes. Neither is probed.

Compile: time runs from the reference to the end of the Enzyme call, compile included. Memory is the peak footprint of the process. Figures per cell follow, one to a line, in the order of the cells above.

- Reverse, CpuSerial, 53 s and 1.38 GB, ok.
- Reverse, CpuThreaded, 74 s and 1.55 GB, failed.
- Reverse, CpuPolyester, 73 s and 1.68 GB, failed.
- Forward, CpuSerial, 50 s and 1.33 GB.
- Forward, CpuThreaded, 74 s and 1.70 GB.
- Forward, CpuPolyester, 68 s and 1.70 GB.

No cell reached the 20 minute bound.

Enzyme: 0.13.219 (Julia 1.13.3, macOS arm64, 2 threads, `--optimize=1`, `--check-bounds=auto`, `--startup-file=no`), loaded from a temporary environment holding DifferentiationInterface, ForwardDiff and Polyester. The 1127 note measured 0.13.218, so the version moved.

## What the cells say

- Forward mode passes under all three policies, each at 3.5e-16 against ForwardDiff.
- Reverse passes under `CpuSerial` (1.2e-16) and fails under both threaded policies, at the same frame: `assemble_padded` inside `_assemble_ghost!`. Enzyme names a runtime symbol it has no derivative for, `ijl_in_threaded_region` under `CpuThreaded` and `jl_set_precompile_field_replace` under `CpuPolyester`; the names are as Enzyme prints them.
- The walk is not whole for this form under any of the three policies (`a.walk.whole == false`, one-thread probe), so `_assemble_padded!` takes the replay branch and not `assemble!(a.A_pad, a.padded)`. The serial reverse goes through the same branch with `_SerialReplay()` and passes, so the failure sits on the threaded side of that branch.
- The sites are not localized below the branch. The candidate loops and hooks above are not probed; a fix subplan starts there.

## Commands run

Each cell is one process, run in order through `.claude/scripts/julia_slot.sh 2`, by a driver in `/tmp/bramble-1157` (outside the repository) that times it and stops it at 20 minutes. The walk probe ran through the same wrapper at one thread.

```text
julia --startup-file=no --project=. --threads=2 --optimize=1 --check-bounds=auto \
    docs/notes/internals/ad_reductions/1157_ghost_load.jl <policy> <rev|fwd>
```

Two earlier batches were discarded: a script bug (top-level `try` assignments) printed the wrong status, and the reverse seconds measured from the wrong step. Every row above is from the last batch, run with the final script.
