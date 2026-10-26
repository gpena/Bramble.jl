# Polyester 1D reduction parity (#942)

Subplan S2.1 of `.claude/plans/v4-1-3-docs.md`. Julia 1.13.2, aarch64 (Apple M-series).
`test/ext/polyester_parity.jl`, testset "reductions nested = top level (#473)", fails on the
non-uniform `(1001,)` grid for `innerₕ`, `normₕ` and `inner₊ₓ` in some full `ext` runs:
the top-level `CpuPolyester` value `p` is `0.6919146312703492`, `CpuThreaded`'s `t` and
the nested value are `0.691914631270349`.

## Cause

The two values come from two compiled copies of one `@simd` loop, built at different
optimisation levels and loaded from two different package images.

- `CpuThreaded`'s `_threaded_dot`, and a nested `CpuPolyester` call (which `@batch` runs on
  the caller with the arrays themselves), run `_separable_line_band`
  (`src/space/inner_product.jl`) on `Vector`s. A top-level `CpuPolyester` call
  (`_batch_dot`, `ext/BramblePolyesterExt.jl`) runs it on the `PtrArray`s `@batch` makes of
  the arrays. The two are separate specializations. Both are cached in package images: the
  `Vector` one in Bramble's, the `PtrArray` one in `BramblePolyesterExt`'s (in a fresh
  process, `Base.specializations` of the method lists both with a code instance before any
  call).
- The loop is `@inbounds @simd`. At `-O2` LLVM vectorizes it, which reassociates the sum; at
  `-O1` it does not, so the sum is sequential. The two orders differ in the last bit on this
  grid: `-O1` gives `…3492`, `-O2` gives `…349`.
- Julia loads a package image built at a higher optimisation level than the process asks
  for, and of the acceptable images it takes the newest. `jl_match_cache_flags`
  (`src/staticdata_utils.c`, lines 648-670, read from the `v1.13.2` tag of JuliaLang/julia
  with `gh api repos/JuliaLang/julia/contents/src/staticdata_utils.c?ref=v1.13.2`, since
  the binary install ships no C source) requires every flag but the optimisation and debug
  levels to match exactly (lines 663-666), and accepts
  `actual_opt_level >= requested_opt_level` (lines 667-669: "allow for higher optimization
  and debug level flags in cache to minimize required compile option combinations").
  `find_all_in_cache_path` (`base/loading.jl` of the 1.13.2 install, lines 1255-1288) sorts
  the candidates by pkgimage, then by `mtime`, newest first.
- Loading an image touches it (`base/loading.jl`, lines 1946-1947: "update timestamp of
  precompilation file so that it is the first to be tried by code loading", then
  `touch(path_to_try)`; and line 2184: `touch(path_to_try) # update timestamp of
  precompilation file`).
  Images of different levels sit side by side in the depot, under different file names. So
  the image a process at `-O1` takes is the acceptable one some process loaded last. A
  default-level process (a `verify_subplan.sh` PROBE, which launches `julia` without
  `--optimize`, a timing run, another agent's check) loads and touches the `-O2` image. A
  later `-O1` process accepts that image and touches it again, and an `-O0` process (the
  unit group) accepts every level and touches whichever is newest.
- The `ext` group runs at `--optimize=1`. If the newest Bramble image is an `-O2` one, the
  group loads it, and `CpuThreaded` runs vectorized code. The extension's image is built by
  the first process that needs one at its level, often the group itself at `-O1`, and then
  the top-level `CpuPolyester` call runs sequential code. Whether a run fails depends on
  which processes loaded which images before it, not on the code. That is why runs on
  identical code and flags both fail and pass.
- `Pkg.test` runs the tests "with the same `check-bounds` setting as the current Julia
  session" (`Pkg/src/Pkg.jl`, line 411). `test.sh` passes none, so every image involved has
  `check_bounds=0`, as the cache flags below show.

Only the `@simd` bodies are exposed. The masked reductions (`innerₕ :a`, `:a :b`) walk the
mask's words in a `while` loop without `@simd`, which LLVM may not reassociate at any
level, so their sum order is the same in every copy; they never failed. Why the 2D and 3D
grids, whose `_separable_block_band` has an `@simd` line loop too, passed in the failing
runs is not established.

### Evidence

- Gate run 20261026-164125 (16:41:25 to 16:50 UTC) failed the six tests. Read at 16:55
  with `Base.parse_cache_header`, the worktree's Bramble images were `iZJOX_2IS0d` (`-O1`,
  mtime 16:24:19), `iZJOX_pADtf` (`-O2`, 16:26:54) and `iZJOX_gjf3d` (`-O2`, 16:45:00).
  The extension's newest image was
  `GCfMG_2lcci` (`-O1`, 16:45:01). Loading touches an image, so `gjf3d` and `2lcci` were
  loaded together at 16:45, inside the run: that is an `-O2` Bramble image with an `-O1`
  extension image. `-O1` processes are the only ones that accept `2lcci`, which fits a
  process of the gate.
- Run 20261026-053104, which also failed, printed Pkg's precompile step for
  `--optimize=1`. It built `Bramble → BramblePolyesterExt` (54.1 s) but not Bramble,
  which was "already precompiled", so it accepted an existing image of `-O1` or above. The
  other gate logs print no precompile step, and the depot keeps no history of earlier
  loads, so the other failing and passing runs cannot be lined up with image levels.
- Standalone, 4 threads, `--optimize=1`, one process each:

  | Bramble image | extension image | `innerₕ` `t` | `innerₕ` `p` | equal |
  |---|---|---|---|---|
  | `-O2` (`gjf3d`) | `-O1` (`2lcci`) | `0.691914631270349` | `0.6919146312703492` | no (all three) |
  | `-O1` (`2IS0d`, rebuilt into a temporary depot) | `-O1` (`2lcci`) | `0.6919146312703492` | `0.6919146312703492` | yes (all three) |

  Same data in both spaces (`parent(up) == parent(ut)`), 64-byte aligned vectors in both.
- In an earlier standalone process (4 threads, `--optimize=1`), the band body called
  directly on the `Vector`s and on `PtrArray`s made of them, `t` and `p` all give `…3492`,
  with `--check-bounds=yes` too. So the split is not a band count, an alignment or a
  bounds-check effect. (At planning, no band count 1-16 of the `Vector` body gave `…3492`
  in a default-level process: that is the vectorized order.)
- In a fresh `-O1` process, `Base.specializations(only(methods(_separable_line_band)))`
  lists the `Vector{Float64}` and the `PtrArray` specializations, each with a code instance,
  before any call: both come from images.
- `test/ext/polyester_ext.jl` overrides methods only in a child process, so it leaves no
  state in the group's process; the earlier files matter only through the images they
  build.

## Reproduction

`test/ext/polyester_parity_repro.jl`, `Repro942.run(n)`, builds the failing state in a
temporary depot placed first on `JULIA_DEPOT_PATH`, and runs the reductions in a child
process at the `ext` group's flags, whatever the caller's flags:

1. A child at `-O2` loads Bramble: the newest Bramble image at `-O2` or above (built into
   the temporary depot if none exists). The image is copied into the temporary depot,
   which makes it the newest.
2. A child at `-O1` compiles `BramblePolyesterExt`'s image into the temporary depot
   (`Base.compilecache`), against that Bramble image.
3. A child at `--optimize=1` with the caller's thread count builds the `(1001,)` spaces under
   `CpuThreaded` and `CpuPolyester`, takes `CpuThreaded`'s three values, and runs the three
   `CpuPolyester` reductions `n` times. It reports the images it loaded; `run` throws if they
   are not an `-O2` (or higher) Bramble image and an `-O1` extension image.

It returns `(; ran = n, mismatches)`, `mismatches` counting the runs in which any of the
three differs from `CpuThreaded`'s value. It throws an `ArgumentError` below 2 threads:
`@batch` makes `PtrArray`s only when it finds a free worker. The temporary depot is deleted
before it returns. The extension's image is compiled on every call: a call takes about a
minute (59-60 s for `n = 20`, 4 threads).

The children do not inherit the caller's flags (`Base.julia_cmd()` would pass on its bounds
checks, coverage and debug level, and an image must match those exactly). Each starts
`julia` with the flags the tests' process gets under `test.sh`, spelled out at their
defaults (`_FLAGS`): `--startup-file=no --check-bounds=auto --code-coverage=none
--track-allocation=none --debug-info=1 --inline=yes --pkgimages=yes`, then its own
`--optimize` and `--threads`. Before that change, a caller at `--check-bounds=yes
--code-coverage=user` got `(ran = 2, mismatches = 0)`: the children loaded bounds-checked
images, whose loop is not vectorized at either level.

The children load real images from the depots after the temporary one, and loading an image
touches it, which would make the `-O2` Bramble image the one the next `-O1` process takes.
Before the children start, `run` therefore records the access and modification times, to
the nanosecond, of every Bramble and `BramblePolyesterExt` image file in `DEPOT_PATH`. It
sets them back with `utimensat` before returning, also when a child fails. This leaves the
choice of image for the next process as it was before the call. It does not undo anything a
process running at the same time does to those files, and the times of files created during
the call are not touched. A call that is killed outright (no `finally`) leaves the touches
in place.

On base (2026-10-26, 4 threads, `n = 20`):

- from a caller at the default level (the PROBE's flags): `(ran = 20, mismatches = 20)`;
- from a caller at `--optimize=1` (the `ext` group's flags): `(ran = 20, mismatches = 20)`;
- control, a copy of the tree outside the repository with only `@simd` removed from
  `_separable_line_band`, caller at the default level: `(ran = 20, mismatches = 0)`. So the
  count measures the `@simd` body's association under split levels, not the forced image
  state alone. The copy's images were deleted afterwards.
- from a caller at `--check-bounds=yes --code-coverage=user --optimize=2`, `n = 2`:
  `(ran = 2, mismatches = 2)`, and the times of all 65 Bramble and extension image files in
  the depots are the same before and after the call.
- with no `-O2` Bramble image anywhere (a copy of the tree under `/tmp`, whose only image,
  built by the caller at `--optimize=1`, was `-O1`), `n = 2`: the `-O2` child built one
  in the temporary depot, and `run` took 222 s and returned `(ran = 2, mismatches = 2)`
  with the depots' image times unchanged. The copy and its images were deleted afterwards.

## The fix (S2.2)

The reductions' sums must not depend on how the compiler vectorizes the band body, since
the process cannot choose which images it loads. The `@simd` band bodies the reductions
share should fix their association in the source: a fixed number of accumulators combined
in a fixed order, no `@simd`, and the multiply-add written so that it cannot contract at one
level and not another (`muladd` may or may not fuse; `fma` or a separate multiply and add
cannot). Files:

- `src/space/inner_product.jl`: `_separable_line_band` (the failing body) and
  `_separable_block_band` (its inner `@simd` line loop: the same construct, though no run
  has failed on it).
- `src/utils/linear_algebra.jl`: `_dot_band` (`@simd` over `_dot_term`, the dense-weight
  reduction `_batch_dot` also runs).
- `test/ext/polyester_parity.jl`: the new testset, which calls `Repro942.run` (it spawns its
  own children, so it runs at the group's `--optimize=1` and leaves the group's process
  untouched).

`ext/BramblePolyesterExt.jl` needs no change: its masked band copies have no `@simd`. Handing
the `Vector`s to the tasks instead (`_Opaque`), so that one specialization serves both
paths, was considered and is worse: it boxes the argument tuple on every call (the 0 B
allocation tests), and a `CpuSerial` or `CpuThreaded` sum would still depend on the image's
level.
