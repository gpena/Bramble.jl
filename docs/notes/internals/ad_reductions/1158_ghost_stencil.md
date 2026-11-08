# #1158: Enzyme on `centered_difference!` of a `GhostVector`

Measurement for gpena/Bramble.jl#1158: `centered_difference!(out, v, Val(1))` on a
`GhostVector`, the cell `ad_coverage.md` lists among its accepted limits (measured failing
on 2026-11-03). The reproducer is `1158_ghost_stencil.jl`, next to this note. It needs
Bramble and nothing else from the repository.

The cell is `build_ghost_runtime` of `test/space/ad_coverage.jl` with
`centered_difference!(out, v, Val(1))` in place of `Δₕ!`: the same `_ghost_space` (7 by 6
non-uniform mesh, seed 11, Neumann sides `:xmin` and `:ymax`, Robin side `:xmax`, Dirichlet
`:ymin`, ghosts on `:xmin`, `:xmax` and `:ymax`), the same basis (`randn`, seed 1234, 42 by
3) and start point. The gradient is taken as the coverage test takes it, through
DifferentiationInterface with `AutoEnzyme(mode = set_runtime_activity(...),
function_annotation = Const)`, and compared with ForwardDiff on the same policy: ok means
`maximum(abs, g - ref) / maximum(abs, ref) <= 1e-8`. Along `Val(1)` the stencil reads the
ghosts at `:xmin` and `:xmax`, so a gradient that dropped the ghost path would not match.

Every cell passes at e6817256 (S1's rule fix included). The 2026-11-03 failure predates
S3.4's `_ghost_facet_band!` rewrite (`ntuple(Val(F))` in place of `for f in fixes`), which
`_ghost_stencil_band!` calls (`src/operators/ghost_vector_operators.jl:189`); that rewrite
is the likely cause, not bisected here.

Cell: CpuSerial rev: ok 1.1e-16

Cell: CpuSerial fwd: ok 1.1e-16

Cell: CpuThreaded rev: ok 1.1e-16

Cell: CpuThreaded fwd: ok 1.1e-16

Cell: CpuPolyester rev: ok 1.1e-16

Cell: CpuPolyester fwd: ok 1.1e-16

Cell: CpuSerial rev Val(2): ok 1.4e-16

No cell fails, so there is no fix site to place.

Compile: first gradient call (compile included; the second call takes under 1 ms) and peak memory footprint (maximum resident set) per process, from `/usr/bin/time -l`: CpuSerial rev 11.5 s, 1.13 GB (1.28 GB); CpuSerial fwd 12.0 s, 1.12 GB (1.29 GB); CpuThreaded rev 22.9 s, 1.23 GB (1.46 GB); CpuThreaded fwd 22.9 s, 1.26 GB (1.48 GB); CpuPolyester rev 27.2 s, 1.26 GB (1.52 GB); CpuPolyester fwd 21.1 s, 1.27 GB (1.52 GB); CpuSerial rev Val(2) 11.5 s, 1.12 GB (1.30 GB). Whole processes took 15 to 36 s.

Enzyme: 0.13.219 (Julia 1.13.3, macOS arm64, 2 threads, `--optimize=1`, `--check-bounds=auto`, `set_runtime_activity`, `function_annotation = Const`, `BrambleEnzymePolyesterExt` loaded).

## The `Δₕ!` control

`test/space/ad_coverage.jl` records the `ghost_runtime` cell (`Δₕ!` on the same
`GhostVector`) as `:throws` with `ReverseRuleReturnError` for `CpuPolyester` reverse
(measured 2026-11-07). The reproducer's control variant `CpuPolyester_rev_laplacian` runs
that cell here, and at e6817256 it passes: relative error 4.3e-16 against ForwardDiff, 28.8 s
for the first call, 1.30 GB peak footprint (1.53 GB resident). S1's fix to the
`CpuPolyester` bands reverse rule (e6817256) is the likely cause; this note did not bisect it.
S2 should turn that entry into `:ok` (and rerun it in the coverage child).

## For S2

- A `ghost_stencil` cell, `build_ghost_runtime` with `centered_difference!(out, v, Val(1))`,
  can be `:ok` in both modes under all three policies, `kinds = (:nonuniform,)`. The six
  cells compile in under 30 s each and stay under 1.6 GB.
- `ghost_runtime` reverse under `CpuPolyester`: `:ok` now (control above), not `:throws`.
- The `ad_coverage.md` limit for `centered_difference!` on a ghost space can go.

## Commands run

Each cell one process, one after another, through `.claude/scripts/julia_slot.sh 2`, in the
worktree at e6817256 with its `Manifest.toml`, and Enzyme, DifferentiationInterface,
ForwardDiff and Polyester from a temporary environment:

```text
/usr/bin/time -l julia --startup-file=no --project=. --threads=2 --optimize=1 \
    --check-bounds=auto docs/notes/internals/ad_reductions/1158_ghost_stencil.jl <variant>
```

The variants: `CpuSerial_rev`, `CpuSerial_fwd`, `CpuThreaded_rev`, `CpuThreaded_fwd`,
`CpuPolyester_rev`, `CpuPolyester_fwd`, `CpuSerial_rev_2` and `CpuPolyester_rev_laplacian`.
Each exited 0.
