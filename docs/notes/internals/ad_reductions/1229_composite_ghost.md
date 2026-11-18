# #1229: the 2D composite ghost `Δₕ` cell under Enzyme, after the component index moved into the type

Measurement of gpena/Bramble.jl#1229, the follow-up to #1174: Enzyme on the composite space `C^2` of a 2D ghost space, with `Δₕ` on both components and a coupling term, assembled under `CpuSerial` (cell `comp` of `1174_composite_ghost.jl`). #1174 left this cell failing in both modes with an `IllegalTypeAnalysisException` in `simplify_ast`. S2.1 of v4.2.9 put the component index in the type of `IndexedTrialFunction` and `IndexedTestFunction`. This note measures the tree it leaves. The script is the one of the 1174 note, unedited, and its first criterion is the bound used here: both modes under 600 s and 10 GB, within 1e-8 of ForwardDiff.

Enzyme: 0.13.220, 2026-11-18, macbook (Apple M2, 16 GB, macOS arm64, Julia 1.13.4, default `-O2`, `--startup-file=no`), 1 thread

Each run was one process under `julia_slot.sh 1` and `/usr/bin/time -l`, with `S1174_LIMIT=1500`, nothing else of Julia's running on the machine. The columns are cell, commit, mode, status, relative error against ForwardDiff, the seconds of the first Enzyme `gradient` call (the compile included, loading and the reference excluded), and the peak memory footprint from `/usr/bin/time -l` in GB (10⁹ bytes). The load average was about 3 at the start with no Julia of mine running, so treat the seconds as rough. The ForwardDiff reference is `[93531.24063274513, 118171.56203520531, 72404.07333773715]`, the same as in the 1174 note.

Cell: comp | 2dd93d28 | rev | ok | 0.0 | 299 | 2.99

Cell: comp | 2dd93d28 | fwd | ok | 1.2e-16 | 192 | 2.12

Verdict: within the bound

## What the runs say

- Both modes differentiate the cell and match ForwardDiff. The reverse gradient is bit-identical to the reference. The forward differs in the last digit of two entries (`93531.24063274512` against `...13`, `72404.07333773717` against `...15`).
- The compile is 299 s in reverse and 192 s in forward, against 982 s and 592 s before the fix (the tip of the 1174 note, where both failed). Peak memory is 2.99 GB and 2.12 GB, against 5.70 GB and 4.08 GB. The two ends of the comparison ran on different Julia and Enzyme versions (1.13.3 with 0.13.219, now 1.13.4 with 0.13.220) and under the same kind of load, so the size of the drop is rough. The direction is not in doubt, since the failure that cost most of the earlier time is gone.
- Both modes are under 600 s and 10 GB with room to spare, so no `sample <pid>` was taken: there is no compile time to place.
- The scalar control of the 1174 note (`one`, 100 s, 2.13 GB) is still the cheaper cell. The composite costs about three times its compile in reverse and a little under twice in forward.

## Commands run

From the worktree root, one process each, in the order reverse then forward:

```text
S1174_LIMIT=1500 .claude/scripts/julia_slot.sh 1 /usr/bin/time -l \
    julia --startup-file=no --project=. --threads=1 \
    docs/notes/internals/ad_reductions/1174_composite_ghost.jl comp <rev|fwd>
```

The script printed `RESULT comp rev ok 0.0 299.0` and `RESULT comp fwd ok 1.2314227702288973e-16 191.6`. The `/usr/bin/time -l` figures were 305.96 s real and 2987101328 bytes of peak memory footprint for the reverse, 198.52 s and 2122861160 bytes for the forward. `Project.toml` and `Manifest.toml` were unchanged by the runs.
