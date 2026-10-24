# Core precompile +9 s and the first `pde_solve` (v4.1.3, S3.2, #906)

Answers #906: which `src/precompile` sessions account for the core precompile growth since
v4.0.0, and what the first `pde_solve(A, F)` compiles now that v4.0.0 cached.

Two sides: v4.0.0 (`0aa87e45`, "old") and main `9c34251c` ("new"). The branch `v4.1.3/perf`
has the same `src/` as `9c34251c`.

Summary:

- The core precompile is +9.0 s and +22.6 MiB of Bramble pkgimage. Two warm-ups added since
  v4.0.0 are all of it: `_pc_polygon` (4.9 s, 11.9 MiB) and the ghost operators plus the ghost
  `semidiscretize` (4.3 s, 10.5 MiB). Every other session together is -0.2 s, inside the noise.
- The first `pde_solve(A, F)` is 27.2 ms on both sides. The +49% of the review did not
  reproduce, and the first call infers no Bramble method on either side. The cause is not
  found; see `## First pde_solve`.

## Method

Julia 1.13.2, Apple Silicon (8 logical cores), AC power, default `-O`, 4 threads. Same method as
`precompile-v4.0.1.md` and `precompile-v4.1.2.md`: a fresh temporary depot per build
(`JULIA_DEPOT_PATH=<tmp>:~/.julia`, `JULIA_PKG_OFFLINE=true`), `using Bramble` under
`/usr/bin/time -p` inside `BRAMBLE_JULIA_SLOTS=4 julia_slot.sh 4 julia --threads=4
--startup-file=no --project=.`, the Bramble pkgimage read from
`<depot>/compiled/v1.13/Bramble` with `du -sk`. Three rounds, the order alternating, so drift
hits both sides.

Both sides are built in detached worktrees (`~/worktrees/Bramble/pc-906-new` and `pc-906-old`)
with the branch `Manifest.toml` copied in, as in S3.1, so only Bramble's source differs. The
`using Bramble` line is the same on both. (The first pde_solve was also run with each tree's own
Manifest; see below.)

Per-session cost uses three tools, because the drop-one build alone does not add up:

1. In-order timing. A local, uncommitted macro `@pct "name" begin ... end` wraps each top-level
   call of `src/precompile.jl` and writes the wall time of the block, in the precompile process,
   to a file (`PC_TIMES`). The blocks run in the same order on both sides, so the times add up
   exactly to the in-process workload time. Three builds per side.
2. Drop-one builds. The same macro skips a block when its name is in `PC_SKIP`, so a build drops
   one session without touching the others. The cost is the median of the interleaved full
   builds minus the build without it. Three runs where a row says so, one run otherwise.
3. The remainder. What the workload does not take in the process (the native code of the
   pkgimage, written when the precompile ends) is the build time minus the build with
   `precompile_workload = false` (9.4 to 9.8 s) minus the in-order sum. It is 0.174 s per MiB of
   workload pkgimage on both sides (41.3 s for 236.5 MiB new, 37.4 s for 214.6 MiB old), so it is
   attributed by size.

The in-process timing reports the workload's inference and execution, not LLVM time
(`Base.cumulative_compile_time_ns` read 0 there). Inference is hoisted to the outermost call:
inside `_pc_form_session` the inner blocks (`fs/...`, `rel2d/...`) read near zero and the outer
`form_session_2d` carries their inference, so only the top-level blocks are meaningful. The
build lines are in `~/bramble-scratch/v413perf/s32/builds.log`, the in-order files in `times/`.

Conditions: not alone on a quiet machine. `gate_preflight.sh --timing` failed on load for the
whole session (1-minute load 2.3 to 5.3, `duetexpertd` and Spotlight, no other Julia of mine), so
every figure is under load. The load at the start and end of each build is in `builds.log`;
round by round it was 3.9 to 5.3 for the first headline set and 2.3 to 3.8 for the interleaved
sets below. A late stretch ran at load 5 to 9, so the prototype timings of the last section are
not used. Spread of one side's three builds: 0.7 to 1.1 s.

## Total difference

| set | new, s (3 runs) | old, s (3 runs) | new - old, median | load |
|---|---|---|---|---|
| pristine trees | 102.22, 103.89, 103.38 | 94.52, 95.00, 94.80 | +8.6 | 3.9 to 5.3 |
| interleaved with the drop-one builds | 100.02, 100.38, 100.72 | 91.96, 92.05, 90.98, then 91.07, 91.48, 91.34 | +9.0 (old median of six, 91.41) | 2.3 to 3.8 |
| no workload at all | 9.60, 9.86, 9.80 | 9.56, 9.43, 9.44 | +0.4 | 3.0 to 3.8 |

The issue measured +9.44 s. The pkgimage is 261604 KiB (255.5 MiB) new and 238392 KiB
(232.8 MiB) old, +22.7 MiB. The dependency difference is not part of it: both sides use one
Manifest, and the build time of the later `newO` and `oldO` trees (own Manifests, 103 s and
91 s, one build each) is the same.

## Per-session cost

In-order block times, median of three builds per side, seconds, with the file that owns the
block. A block of `src/precompile.jl` that calls a session function is named after it; "inline"
is code written in `src/precompile.jl` itself.

| block | file | old | new | new - old | new without polygon and ghost operators |
|---|---|---|---|---|---|
| geometry | geometry_sessions | 0.06 | 0.06 | +0.00 | 0.06 |
| linalg64 | utils_sessions | 0.07 | 0.07 | 0.00 | 0.07 |
| mesh_sessions | mesh_sessions | 1.00 | 1.02 | +0.03 | 1.03 |
| mesh_mutation | mesh_sessions | 0.27 | 0.29 | +0.02 | 0.27 |
| mesh_queries | inline | 0.04 | 0.04 | 0.00 | 0.04 |
| space_sessions | space_sessions | 2.02 | 2.02 | -0.01 | 1.98 |
| operator_sessions | operator_sessions | 6.35 | 6.46 | +0.11 | 6.37 |
| float32_block | mesh, space, operator | 3.55 | 3.56 | +0.01 | 3.51 |
| form_session_1d | form_sessions | 5.79 | 5.68 | -0.11 | 5.67 |
| form_session_2d | form_sessions | 9.73 | 11.16 | +1.43 | 8.82 |
| form3d_laplacian | inline | 0.58 | 0.63 | +0.05 | 0.54 |
| form_varcoef | inline | 0.51 | 0.52 | +0.01 | 0.57 |
| form3d_nests | inline | 2.00 | 1.79 | -0.21 | 1.89 |
| jacobian_typecached | form_sessions | 0.26 | 0.27 | +0.01 | 0.27 |
| semidiscretize | semidiscretize_sessions | 0.29 | 0.45 | +0.16 | 0.28 |
| solver | solver_sessions | 10.29 | 10.55 | +0.26 | 10.83 |
| parallel | parallel_sessions | 0.38 | 0.38 | +0.01 | 0.38 |
| interpolation | space_sessions | 0.86 | 0.90 | +0.04 | 0.87 |
| exporters | exporters_sessions | 0.12 | 0.12 | 0.00 | 0.12 |
| domains and markers (two blocks) | inline | 0.21 | 0.23 | +0.02 | 0.23 |
| polygon64 (`_pc_polygon(Float64)`) | space_sessions | 0 | 1.07 | +1.07 | 0 |
| polygon32 (`_pc_polygon(Float32)`) | space_sessions | 0 | 1.72 | +1.72 | 0 |
| mesh_positional | inline | 0.01 | 0.01 | 0.00 | 0.01 |
| sum of the three builds, median | | 44.56 | 49.32 | +4.76 | 43.95 |

Builds that drop one session, difference to the full build of the same side (seconds, Bramble
pkgimage in MiB). The runs are three where the "runs" column says 3, one otherwise; a single
run is good to about 1.5 s (the same `parallel` block reads 2.2 s on old and 0.6 s on new).

| dropped | runs | old, s | old, MiB | new, s | new, MiB |
|---|---|---|---|---|---|
| polygon64 and polygon32 | 3 | none | none | 5.6 | 11.9 |
| ghost operators and ghost `semidiscretize` | 3 | none | none | 4.2 | 10.5 |
| both of the above | 3 | none | none | 9.25 | 22.2 |
| form_session_2d | 3 | 19.2 | 43.1 | 16.6 | 41.3 |
| form_session_1d | 1 | 8.1 | 19.4 | 8.3 | 20.0 |
| float32_block | 3 | 7.3 | 18.2 | 6.6 | 16.6 |
| space_sessions and operator_sessions | 1 | 14.3 | 34.4 | 14.5 | 34.5 |
| operator_sessions | 1 | 5.0 | 28.5 | 11.7 | 28.5 |
| the three inline 3D forms | 1 | 5.9 | 16.0 | 5.8 | 15.8 |
| solver | 1 | 18.9 | 46.8 | 19.4 | 48.5 |
| semidiscretize (plain and ghost) | 1 | 1.9 | 1.2 | 1.9 | 1.5 |
| interpolation | 1 | 2.2 | 3.9 | 1.2 | 3.9 |
| parallel | 1 | 2.2 | 1.7 | 0.6 | 1.8 |
| exporters | 1 | 0.7 | 0.7 | -0.3 | 0.6 |

Dropping `jacobian_typecached` does not build (its `Wₕ_pc` feeds `semidiscretize`), so that
row has no figure. The drop-one rows overlap: a method inferred by an early session is not
inferred again by a later one, so the sum of the rows is not the total and a session can cost
more when dropped on one side than on the other (`operator_sessions`, 5.0 against 11.7 s). The
in-order table does not have that problem, so the reconciliation uses it.

### Reconciliation with the total

| part | old | new | new - old |
|---|---|---|---|
| build without the workload | 9.44 | 9.80 | +0.36 |
| workload in the process (in-order sum) | 44.56 | 49.32 | +4.76 |
| pkgimage written after the workload (remainder, 0.174 s per MiB) | 37.41 | 41.26 | +3.85 |
| full build | 91.41 | 100.38 | +8.97 |

The three parts add up to the measured difference (0.36 + 4.76 + 3.85 = 8.97). By session,
with the remainder attributed by pkgimage size:

| session | in the process, s | pkgimage, MiB | remainder, s | total, s | drop-one build, s |
|---|---|---|---|---|---|
| `_pc_polygon` (both precisions) | 2.79 | 11.9 | 2.07 | 4.9 | 5.6 |
| ghost operators and ghost `semidiscretize` | 2.51 | 10.5 | 1.83 | 4.3 | 4.2 |
| every other session and the source growth (+0.36 outside the workload) | -0.54 | +0.3 | -0.05 | -0.2 | |
| total | 4.76 | 22.7 | 3.85 | 9.0 | |

The last row is also measured directly: new with polygon and ghost operators skipped builds in
91.13, 90.65, 92.17 s with a 233.3 MiB pkgimage; old builds in 91.07, 91.48, 91.34 s with 232.8
MiB. The two additions are the whole difference. Under them the work that remains went the other
way: `nested/relations` (the ghost relations session, 5.3 s old, 1.8 s new in-order) became
cheaper once `constrained` was type-stable (v4.1.2), which pays for the rest of the growth.

The two additions, from the source diff of the workload files: `_pc_polygon` (in
`space_sessions.jl`, called twice from `precompile.jl`) arrived with the polygon work of v4.1.0;
`_pc_ghost_operators` and `_pc_ghost_semidiscretize_session` (in `form_sessions.jl` and
`semidiscretize_sessions.jl`) are the #812 warm-ups. The ghost `form` slowdown of S3.1
(`8d81c11d`, #832) does not show in precompile: every other block is within the noise.

### What each addition saves on the first call

A fresh process, `using Bramble`, one call timed with `@elapsed`, in a depot built with the
session skipped and in one built with it (the same tree and Manifest). Three runs, seconds.

| first call | without the session | with it | saving |
|---|---|---|---|
| ghost `matrix_free_operator` and `kronecker_operator`, a product each (2D Robin, (11, 14) mesh) | 3.91, 3.94, 3.91 | 0.061 to 0.064 | 3.85 |
| ghost `semidiscretize` and one residual call | 0.66, 0.68, 0.68 | 0.10 | 0.57 |
| polygon, Float64 (the workload's form, with the mixed term) | 0.37, 0.37, 0.34 | 0.18 to 0.19 | 0.18 |
| polygon, Float32 | 4.12, 4.12, 4.06, 4.13 | 0.53 | 3.6 |

`_pc_polygon(Float64)` and `_pc_polygon(Float32)` are independent: a depot built without only
the Float64 call has the same Float32 first call (0.52 s) and one without only the Float32 call
has the same Float64 one (0.19 s). Pkgimage by call: Float64 3.7 MiB, Float32 7.2 MiB (depots
without one of them, 257712 and 254064 KiB against 261452 KiB). With the in-order times
(1.07 s and 1.72 s) and 0.174 s per MiB, the Float64 call costs about 1.7 s of precompile and
the Float32 one about 3.0 s.

## First pde_solve

Not reproduced. The first `pde_solve(A, F)` of `benchmark/ttfx.jl` (non-uniform 8 by 8 mesh,
`a_spd = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))`, `A = assemble(a_spd)`), in a fresh process
after `using Bramble` in a warm depot, five runs per tree, interleaved, milliseconds:

| tree | Manifest | runs | median |
|---|---|---|---|
| v4.0.0 `0aa87e45` | branch | 27.36, 27.14, 27.37, 27.16, 27.12 | 27.16 |
| main `9c34251c` | branch | 30.30, 27.16, 27.23, 27.95, 27.44 | 27.44 |
| v4.0.0 | its own | 27.29, 27.36, 27.06, 27.15, 27.17 | 27.17 |
| main (this worktree) | its own | 27.45, 27.44, 27.18, 27.28, 27.26 | 27.28 |
| `3e57de02`, the commit of the review | branch | 27.37, 27.18, 27.13, 27.24, 27.27 | 27.24 |

The issue reported 28.0 ms against 41.8 ms. The 30.30 ms run is the first run after a build;
the others are inside 0.3 ms of v4.0.0, with either Manifest and also at `3e57de02`. The
`pde_solve` source (`src/solvers/pde_solve.jl`, `sparse_solvers.jl`) is unchanged between the two
trees.

`SnoopCompileCore.@snoop_inference` on the same call, in a fresh process with a warm depot, ran
three times each on main and v4.0.0 with the branch Manifest and twice on `3e57de02`. Every
tree infers 8 instances in 23.4 to 24.0 ms. The instances are the same by method on both sides,
the root and seven SuiteSparse methods. They are `cholmod_l_allocate_dense` (2.0 ms),
`cholmod_l_analyze` (3.2 ms), `cholmod_l_solve` (3.9 ms), `cholmod_l_factorize_p` (3.9 to
4.0 ms), `getproperty` (3.8 ms) and two `setproperty!` (3.3 and 3.4 ms) on
`cholmod_common_struct`. No Bramble method is among them. `A \ F` of an SPD matrix goes
through CHOLMOD, and the saved image does not cover those wrappers on either side. Per method
the two trees differ by at most 0.3 ms, so there is nothing to diff. A snoop with the v4.0.0
tree's own Manifest did not load SnoopCompile, and the cause was not looked at. The v4.0.0
snoop above is the one with the branch Manifest.

Not measured: the `benchmark/ttfx.jl` harness itself was not rerun, so whether the review's
41.8 ms came from the harness (the case script is the same code) or from the machine at that
moment is not known. The figure of the review is one median of three launches, under a load not
recorded here. AC3 of #906 (a rerun against v4.0.0) is still to do on a quiet machine.

## Candidates of the #906 comment

The tutorials infer 6 s more on this version, mostly in `_at_facet`, `_relation_rows`,
`_resolved_inner` and `_visit_guarded_region!`. Checked against the sessions, in a fresh
process after `using Bramble`, `SnoopCompileCore.@snoop_inference`, `poisson_linear.jl`'s
`poisson_system` (Robin on `(:xmax, :ymax)`, `Δₕ` form, `dirichlet = ...`, `symmetrize`) with
named data functions:

| snippet | inference | Bramble-owned | largest methods (ms, exclusive) |
|---|---|---|---|
| control: `_pc_form_nested(Val(2))`, the workload's own call | 0 s, 1 instance | 0 | none |
| tutorial snippet, `ghosts = (:xmax, :ymax)`, first function types | 6.0 s, 5174 inst. | 5.26 s | `_at_facet` 1071, `_visit_guarded_region!` 691, `_replay_segment!` 335, `_throw_rows_dirichlet` 222 |
| the same, other data function types | 0.82 s, 825 inst. | 0.74 s | `_at_facet` 130, `_form_rows!` 81 |
| same form, no `dirichlet` keyword, `(:xmax, :ymax)` | 4.8 s | 4.28 s | `_at_facet` 902, `_visit_guarded_region!` 727 |
| `neumann_robin_poisson.jl` shape, `ghosts = :boundary`, four relations | 1.4 s | 1.10 s | `_at_facet` 283, `_scatter_term!` 142 |

So the sessions do cover what they call (the control infers nothing), and the part that is
left falls in two causes:

1. The facet set. Every session builds `gridspace(Ωₕ; ghosts = :boundary)`; a space with
   `ghosts = (:xmax, :ymax)` (also `(:xmin,)`, `(:xmax,)`, in four more examples) is another
   type, and the guarded-region walk, `_at_facet` and the relation code compile for it. That is
   3.4 to 3.6 s of the 4.8 to 6.0 s.
2. The data's function type. A named user function or a closure is a type no workload has
   (`_at_facet` re-infers 130 to 300 ms per set of function types, 1.4 s for a four-relation
   `:boundary` space). A workload cannot cover it.

The review's 4.7 s for `_at_facet` over 152 blocks is this: one instance per facet set and per
data type, each unrolled over the facets.

## Proposals

### Finding 1: the core precompile is +9.0 s over v4.0.0

Each addition saves about what it costs on the first call, so the ceiling of +0.8 s cannot be
met without paying the first call of the users that need them.

- Ghost operators and ghost `semidiscretize` (4.3 s, 10.5 MiB; saves 3.85 s + 0.57 s): keep, as
  S2.2 kept the conormal. No change.
- `_pc_polygon(Float32)` (about 3.0 s, 7.2 MiB; saves 3.6 s on a Float32 polygon first call):
  keep. Marginal; dropping it moves 3.6 s to the few Float32 polygon users.
- `_pc_polygon(Float64)` (about 1.7 s, 3.7 MiB; saves 0.18 s): drop. File
  `src/precompile/space_sessions.jl` and `src/precompile.jl`: delete the `_pc_polygon(Float64)`
  call (the Float32 one stays, and the function stays for it). It brings the core precompile to
  about +7.3 s. Measured: the saving of the call (three runs), its size (one depot) and its
  in-order time (three builds). Not measured: a full build without the call alone, so the
  1.7 s is an estimate; that build is the check of this change.
- The rest is `## Accepted cost`.

Accepted cost of finding 1: about +7.3 s of core precompile after the Float64 polygon call is dropped (+9.0 s if it is not):
the ghost operators, the ghost `semidiscretize` and the Float32 polygon path, each paying back its
first call, and 22 MiB of pkgimage with them. The existing sessions together did not grow (-0.2 s
inside the noise); `constrained`'s type stability gave back about 3.5 s of in-order time.

### Finding 2: first `pde_solve`

No fix. 27.2 ms on both sides with either Manifest, the same seven CHOLMOD methods inferred.
Nothing to accept. The cause of the review's 41.8 ms is not found; rerun
`benchmark/ttfx.jl` against v4.0.0 on a quiet machine to close AC3.

### Finding 3: tutorials infer 4.8 to 6.0 s on the first partial-ghost snippet

A prototype session (not committed, a temporary worktree): `_pc_ghost_partial()` in
`src/precompile/form_sessions.jl`, called from `src/precompile.jl`, builds the 2D Robin space on
`ghosts = (:xmax, :ymax)` with constant data, assembles `-Δₕ` with `assemble(a, l)` and with the
`dirichlet` keyword and `symmetrize = true`. Measured on it (depot built with it, same snippets as
above):

| snippet | without | with |
|---|---|---|
| `(:xmax, :ymax)`, `dirichlet` keyword | 5.25 s | 1.66 s |
| `(:xmax, :ymax)`, no keyword | 4.81 s | 1.40 s |
| `ghosts = :boundary` (control) | 1.40 s | 1.44 s |

In-order time of the block 1.51, 1.54, 1.61 s; pkgimage 267.4 to 268.0 MiB against 261.4 to 261.7
MiB, +6.1 MiB, so about 2.6 s of precompile. The build times of those runs (109 to 113 s against
107 to 111 s) are at load 4 to 9 and are not used. Saving 3.4 to 3.6 s against a cost of about
2.6 s, on a shape of four tutorial pages. It pays, by the S2.2 rule, but it adds to the +7.3 s,
so it goes in only if the integrator prefers the tutorials' first call to the precompile
budget; the Float64 polygon drop above (1.7 s) offsets most of it. Not covered by it: the
data-type part (1.4 s), which no workload can reach. A type-erased data argument for `_at_facet`
and `_relation_rows` (`@nospecialize` on the data, in `src/assembly/ghost_relations.jl`) would
remove that part; it was not prototyped, and the unrolled recursion exists to keep the forms
type-stable, so it is a design change and not part of this subplan.

## Accepted cost

Finding 1 after the Float64 polygon drop (+7.3 s, 22 MiB less 3.7) and the data-type part of
finding 3 (1.4 s per set of function types). Details under the findings above.

## Not done

- The `benchmark/ttfx.jl` rerun (AC3), and a full build without `_pc_polygon(Float64)` alone.
- The drop-one rows with one run have the spread given above.
- The prototype's timing under load.
