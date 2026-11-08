# #1127: Enzyme on a constrained ghost space

Reduction of gpena/Bramble.jl#1127, the Enzyme half: `Δₕ!` on a `GhostVector`, and reverse
mode through a ghost-space `assemble(a, l)`. The reproducer is `1127_ghost.jl`, next to this
note. It needs Bramble and nothing else from the repository.

Reproduces (Δₕ!): `julia --project=. --threads=1 --optimize=1 --check-bounds=auto docs/notes/internals/ad_reductions/1127_ghost.jl` from the repository root ends with `EnzymeNoDerivativeError` ("cannot handle (augmented) unknown intrinsic `llvm.smax.i64`", in `_ghost_facet_band!`) and exit 1, in 17 s, at 60cb3acd. The variant `runtime_fwd` ends the process with a segmentation fault (exit 139), also in 17 s. Both still reproduce at HEAD: neither S1.2 (fa0bc1f2, 4bc9f086) nor S3.2 (54ff4c57) touched this path.

Reproduces (assembly): two failures at 60cb3acd. The load half, `.../1127_ghost.jl loadbuffer_rev`, ends in `EnzymeNoDerivativeError` ("No create nofree of empty function `ejlptr$strcmp$0`", at `assemble!` into the load form's `F_pad`) and exit 1, in 13 s; `pairmass_rev` (the mass form, `assemble(a, l)`) ends the same way in 329 s. The matrix half is the issue's own cell, `assemble_rev` with the `Δₕ` form: it was still compiling after 20 minutes (3 GB) when I stopped it, so no run of the unmodified cell here reaches the `EnzymeInternalError` the issue measured at 22 to 29 minutes. With the load half fixed, `inact-walk_padded_rev` ends in `EnzymeInternalError` after 965 s and `inact-red_assemble_rev` after 1322 s. The matrix half does not reproduce in minutes.

Fix site: for Δₕ! it is `src/operators/ghost_vector_operators.jl`, function `_ghost_facet_band!` (line 95), which loops `for f in fixes` over an `NTuple{4, _FacetFix{Float64}}` with a run-time index (variant `unroll_runtime_rev` differentiates it), and `Enzyme` upstream for the forward segfault, which is `emit_error` (`jlrt.jl`, `emit_allocobj!`) crashing while it reports the same intrinsic error. For the assembly it is `src/assembly/ghost_assembly.jl`, function `_assemble_ghost_load!` (line 1151, its `assemble!(l.F_pad, l.padded)`) for the load half, and `_ghost_bilinear_form` (line 276) with `_ghost_maps` (line 361) for the matrix half; with both changed as in the reproducer (`split+loadbuf_assemble_rev`) the issue's cell differentiates.

Compile: for Δₕ! 14 s inside Enzyme before the error and 1.23 GB peak resident set (16.7 s wall with the load), 13.5 s and 1.23 GB with the fix. For the assembly, the load half takes 13 s (`loadbuffer_rev`) and 329 s with the mass form; the issue's cell with both changes compiles and runs in 1088 s at about 4.2 GB resident (sampled every 10 s), and unmodified it is above 20 minutes and 3 GB.

Enzyme: 0.13.218 (Julia 1.13.3, macOS arm64, 1 thread, `--optimize=1`, `--check-bounds=auto`).

## Δₕ! on a GhostVector

The reverse message names `_band_slab` (`src/operators/difference.jl:394`, `max`), but the
call that cannot be handled is an integer `llvm.smax.i64` whose operand Enzyme has no type
for. A probe that redefined `_band_slab` without `max` and `min` moved the error to
`EnzymeNoTypeError`, "Cannot deduce adding type of: `%value_phi14 = select i1 %.not34, i64
%.sroa.641.0, i64 1`", the same value one step earlier. That value is a field of an element
of `fixes`, loaded through `getelementptr {i64, i64, i64, i8, i8, [3 x double]}, ptr %fixes,
i64 %iv` with negative offsets after the run-time index, and Enzyme's type analysis does not
follow it.

| Variant | Outcome |
|---|---|
| `runtime_rev` | `EnzymeNoDerivativeError`, 14 s |
| `runtime_fwd` | segmentation fault, exit 139 |
| `unroll_runtime_rev` | gradient, 13.5 s, 5e-11 relative to a central difference |
| `unroll_runtime_fwd` | gradient, 12.6 s, equal to the reverse one to 1e-15 |

The idiom is the one that caused #1126 (a tuple indexed with a run-time value), so the
remedy has the same shape: build each element with a constant index, here
`ntuple(k -> ..., Val(F))`. The forward segfault is the error path of the same intrinsic, so
the same change removes it. The page also lists `centered_difference!` on a `GhostVector`;
it shares `_ghost_facet_band!`, and I did not run it. `CpuThreaded` and `CpuPolyester` call
the same band function, and I did not run them.

## Ghost assembly

Reverse mode, `CpuSerial`, a 7 by 6 non-uniform mesh, the issue's Neumann and Robin space,
the load form built outside the function. `padded` builds the ghost form inside the
function and assembles its padded form only, with no reduction and no load; `plain` is the
same padded form built on the padded space with no ghost construction. Every row is one
process. Relative differences are against a central difference of step 1e-6.

| Variant | Outcome |
|---|---|
| `loadbuffer_rev` (`assemble!(L.F_pad, L.padded)`) | `EnzymeNoDerivativeError` (`strcmp`), 13 s |
| `load_rev` (the same into `zeros(eltype(p), n)`) | gradient, 10 s, 6e-11 |
| `pairmass_rev` (mass form, `assemble(a, l)`) | `EnzymeNoDerivativeError` (`strcmp`) in `_assemble_ghost_load!`, 329 s |
| `loadbuf_pairmass_rev` (the buffer made inside `_assemble_ghost_load!`) | gradient, 339 s, 7e-8 |
| `assemble_rev`, `padded_rev` | no result after 20 minutes; the log fills with "Could not find use of stored value" naming `augmented_julia_form_*_inner`, an `sret` of a deeply nested struct |
| `plain_rev` (control) | gradient, 419 s |
| `mass_rev` (`innerₕ(U, V)`, no `Δₕ`, `assemble(a)`) | gradient, 361 s |
| `formonly_rev` (the ghost `form`, only `a.coef` read) | gradient, 255 s, 5e-11 |
| `mapsonly_rev` (`_ghost_walk`, `_ghost_maps`, no assembly) | gradient, 79 s, 5e-11 |
| `inact-maps_padded_rev` (`_ghost_maps` inactive) | gradient, 595 s, equal to `plain_rev`'s |
| `inact-maps-walk_padded_rev` (run as `inact_padded_rev`; the same two rules) | gradient, 614 s |
| `inact-walk_padded_rev` | `EnzymeInternalError`, 965 s |
| `inact-red_assemble_rev` (the two index builders inactive) | `EnzymeInternalError`, 1322 s |
| `inact-maps_assemble_rev` | `EnzymeNoDerivativeError` in `_assemble_ghost_load!`, 962 s |
| `inact-maps_mass_rev` | gradient `[0, 0, 0]` against `[0.4396, 0, 0]`, wrong, 324 s |
| `inact-red_mass_rev`, `split_mass_rev` (draft runs) | gradient, 350 s and 342 s, 7e-11 |
| `split_assemble_rev` | `EnzymeNoDerivativeError` (`strcmp`) in `_assemble_ghost_load!`, 1088 s |
| `split+loadbuf_assemble_rev` | gradient, 1088 s, 2.5e-7 |

What the table says:

- The relation replay and the constrained solve are not the cause. The replay is the cut
  walk's `_assemble_bilinear_core_cached!`, which `padded` does not reach and which fails
  there anyway; `constrained` runs once, outside the function.
- The load half is the buffer, as the ghost-buffers candidate has it. `assemble!(l.F_pad,
  l.padded)` writes memory that the differentiated function did not allocate, the load
  form's own `F_pad`, and Enzyme then needs a derivative for the whole of
  `_assemble_linear!`, which contains a `strcmp` it has none for. The same call into a
  buffer allocated in the function (`load_rev`, and `_assemble_ghost_load!` with
  `F_pad = similar(l.F_pad)`, `loadbuf_pairmass_rev`) differentiates in seconds to minutes.
- The matrix half is the structure `_ghost_maps` builds. `form` alone (255 s) and the maps
  alone (79 s) differentiate; consumed by an assembly in the same function, they do not.
  With `_ghost_maps` inactive the combination does (595 s), so the sparse patterns `src`,
  `dst`, `colptr`, `rowval`, `liftptr`, `liftsrc`, `liftcol`, `coef` and the walk's
  `_GridLines`, live in one `GhostBilinearForm`, are what Enzyme chokes on.
- Marking `_ghost_maps` inactive is not a fix. It returns `A_pad`, the matrix that
  `_assemble_padded!` then fills, so `A_pad` becomes constant and the gradient is dropped
  without an error (`inact-maps_mass_rev`). The `split` variant keeps `A_pad` allocated in
  the differentiated code and puts only the index structure behind `EnzymeRules.inactive`;
  it gives the right gradient on the mass form (`split_mass_rev`).
- `coef` is in the inactive part of `split`, so it is only valid where the closure `E` does
  not depend on the parameter. A parameter-dependent Robin coefficient would lose its
  derivative; that is the `ghost_robin` row, and a fix must keep `coef` out of the inactive
  call there.
- Together (`split+loadbuf_assemble_rev`) they differentiate the issue's cell. The 2.5e-7
  relative difference is the central difference's own error at this step; the acceptance
  criterion of 1e-8 against ForwardDiff was not measured. Forward mode, `CpuThreaded` and
  `CpuPolyester` were not run.

The matrix half does not reproduce in minutes. The fastest runs that tell it apart are
`formonly_rev` (255 s) and `mapsonly_rev` (79 s), which pass, and `inact-maps_padded_rev`
(595 s), which passes where `padded_rev` does not finish in 20 minutes. Runs I stopped:
`assemble_rev` and `padded_rev` at 20 minutes (the second by an output cap on the warning
log, not by choice), `robin_rev` at 30, `data_rev` at 47, over the 30 minute bound.

## The `ghost_data` row

The fix above does not cover it. `data_rev` differentiates only `constrained(W, ...)` with
the Neumann data `g = p[1]`, reading the closure's `d`; no form and no assembly. It had not
finished after 47 minutes (killed). `robin_rev`, which differentiates `constrained` with the
Robin coefficient `β = p[1]` and reads the entries of `E`, had not finished after 30
minutes either (killed, 3.1 GB resident). The construction of the constrained space
(`src/assembly/ghost_composite.jl`, `constrained`, line 836) has its own failure, which
this reduction did not localize; the page's "Illegal calling convention fixup" (reverse) and
`dgetrf` (forward) are its earlier measurements. ForwardDiff's `Float64(::Dual)` failure on
a Robin coefficient is a separate matter.

## For the fix subplans

- Δₕ!: unroll the loop in `_ghost_facet_band!` (`ntuple(Val(F))`), then test with
  `unroll_runtime_rev` and `unroll_runtime_fwd` of the reproducer, whose default variant then
  exits 0, and with the `ghost_runtime` cell of `test/space/ad_coverage.jl`. Check the
  `CpuThreaded` and `CpuPolyester` cells and `centered_difference!`, which this reduction
  did not run.
- Upstream Enzyme (gate item G3) for the segfault: forward `Δₕ!` reaches `emit_error` and
  crashes inside `emit_allocobj!` instead of raising the error. A second report for the
  `strcmp`: a call that writes a `Const` buffer fails with "No create nofree of empty
  function" instead of being skipped as inactive.
- Assembly, load half: in `_assemble_ghost_load!`, assemble into a buffer made there (as
  the reproducer's `loadbuf`), and look for the same idiom where a ghost form's own
  `F_pad` or `A_pad` is filled inside a differentiated call (`_assemble_padded!`, the
  composite forms in `ghost_composite.jl`, which this reduction did not run).
- Assembly, matrix half: split `_ghost_maps` as in the reproducer's `split` variant, in
  `src/assembly/ghost_assembly.jl`, keeping `A_pad` and `coef` active, then test with
  `split+loadbuf_assemble_rev`.
- The 10 minute compile bound of the issue is not met: the control `plain_rev`, with no
  ghost construction, already costs 419 s, so the `Δₕ` form's own compile is most of the
  1088 s. S4's cells need a cheaper form (the mass form takes 6 minutes) or a looser bound.
- Not covered: forward mode on the assembly, the threaded policies, the Dirichlet part, and
  a Robin coefficient or relation data that depend on the parameter.

## Commands run

Each through `.claude/scripts/julia_slot.sh 1`, in the worktree at 60cb3acd (three
whitespace-only edits uncommitted, none in a file these runs read), with the tree's
`Manifest.toml` (Enzyme 0.13.218) and Enzyme from a temporary environment:

```text
julia --startup-file=no --project=. --threads=1 --optimize=1 --check-bounds=auto <script> <variant>
```

The `Δₕ!` variants, `robin_rev`, `load_rev`, `loadbuffer_rev`, `pairmass_rev`,
`loadbuf_pairmass_rev`, `split+loadbuf_assemble_rev`, `mapsonly_rev` and `inact-maps_mass_rev`
ran from the final script. The others ran from earlier drafts with the same loss functions
and the same inactive rules, under the names above except `fixed_rev` and `fixed_fwd` (now
`unroll_runtime_rev` and `unroll_runtime_fwd`), `inact_padded_rev` (now
`inact-maps-walk_padded_rev`), and the drafts' `pairmass` runs of `inact-maps`, `inact-red`
and `split`, which a draft's `occursin` chain sent to the mass loss (`pairmass` contains
`mass`), so the table lists them as `mass`. `data_rev` ran from a draft, `formonly_rev` and
`inact-red_mass_rev` once each. Resident sizes are from `/usr/bin/time -l` (the `Δₕ!` runs)
and from `ps` every 10 s (the long ones).
