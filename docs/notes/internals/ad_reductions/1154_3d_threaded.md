# #1154: the 3D `Mₕ(κ)` cell under the threaded policies

Measurement for gpena/Bramble.jl#1154: the `assemble_averaged_3d` cell of test/space/ad_coverage.jl (`assemble` of `inner₊(Mₕ(κ) * ∇ₕ(U), ∇ₕ(V))` on a non-uniform 5 x 5 x 5 mesh, `κ` a grid function holding `2 + B p`), reverse and forward under `CpuThreaded` and `CpuPolyester`, with `CpuSerial` as a control. The test file records the threaded cells as `:skip`, so none had run. The reproducer is `1154_3d_threaded.jl`, next to this note. It runs one policy and one mode per process, with the cell's mesh, basis and loss, against ForwardDiff on the same policy. It never calls `Enzyme.API.maxtypeoffset!`: every figure is at Enzyme's default type-analysis bound. Its command is in the last section.

Cell: CpuSerial rev: ok 1.8e-15

Cell: CpuThreaded rev: ok 1.9e-15

Cell: CpuPolyester rev: ok 1.9e-15

Cell: CpuSerial fwd: ok 1.8e-15

Cell: CpuThreaded fwd: ok 1.8e-15

Cell: CpuPolyester fwd: ok 1.8e-15

No cell fails, so there is no `Fix site:` line and no subplan S2.2.x to add.

Compile: time runs from the reference to the end of the Enzyme call, compile included. Memory is the maximum resident set size of the Julia process, from `/usr/bin/time -l`. Figures per cell follow, one to a line, in the order of the cells above.

- Reverse, CpuSerial, 118.7 s and 2.45 GB.
- Reverse, CpuThreaded, 206.0 s and 2.62 GB.
- Reverse, CpuPolyester, 160.4 s and 2.37 GB.
- Forward, CpuSerial, 86.5 s and 1.85 GB.
- Forward, CpuThreaded, 178.4 s and 2.69 GB.
- Forward, CpuPolyester, 122.2 s and 2.25 GB.

No cell reached the 25 minute bound, and none came near the 10 GB cap of `julia_slot.sh`. Each was a single run, with the other Enzyme worker of the plan active on the machine, so the seconds are indicative, not a timing measurement.

Enzyme: 0.13.219 (Julia 1.13.3, macOS arm64, 2 threads, `--optimize=1`, `--check-bounds=auto`, `--startup-file=no`), default `maxtypeoffset`, loaded from a temporary environment holding DifferentiationInterface, ForwardDiff and Polyester. DifferentiationInterface ran with `set_runtime_activity` and `function_annotation = Enzyme.Const`, the closure built inside a builder function as the tests build it.

## What the cells say

- All six cells match ForwardDiff on the same policy to 2e-15, well inside the 1e-8 bound of the test's `_relerr`.
- The serial control agrees with the figures S2.1 recorded for the default bound: 124.5 s reverse and 81.9 s forward then, 118.7 s and 86.5 s here.
- The threaded policies cost more to compile than serial: about 1.7 times in reverse for `CpuThreaded` (206.0 s against 118.7 s), 1.4 times for `CpuPolyester`, and 2.1 and 1.4 times in forward. Memory stays within 1.5 times of serial, under 2.7 GB.
- The reverse runs print `step enzyme start` only after Enzyme has compiled (119.9 s for serial). Enzyme compiles the reverse thunk while Julia compiles the function that calls `gradient`, before that function prints anything. For a reverse cell stopped by the 25 minute bound the last step would read `reference`, and it would mean "stopped in Enzyme's reverse compile". The forward runs compile inside the call, between `enzyme start` and `enzyme done`.
- Nothing here needs the test's `Enzyme.API.maxtypeoffset!(1024)` in the 3D cells: that is S6's to remove (O9). The threaded `:skip` entries of the `assemble_averaged_3d` cell can become `:ok` there.

## Commands run

Each cell is one process, run in order through `.claude/scripts/julia_slot.sh 2`, by a driver in `/tmp/bramble-1154` (outside the repository) that timed it and would have stopped it at 25 minutes by the PID the script prints. Rerun one cell from the repository root with:

```text
/usr/bin/time -l .claude/scripts/julia_slot.sh 2 julia --startup-file=no --project=. \
    --threads=2 --optimize=1 --check-bounds=auto \
    docs/notes/internals/ad_reductions/1154_3d_threaded.jl <policy> <rev|fwd>
```

`<policy>` is CpuSerial, CpuThreaded or CpuPolyester. The exit status is 0 for `ok`.
