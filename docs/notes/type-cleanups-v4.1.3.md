# Type cleanups v4.1.3: base against branch (#928, #930)

Measured by subplan S8.2 of `.claude/plans/v4-1-3-types.md`. Base is the detached tree at
`d64ab24a` (the merge base with `origin/main`), head is the branch `v4.1.3/type-cleanups` at
`9f3ed728`. Both trees were unmodified. Raw figures are in
`benchmark/results/v4.1.3_type_cleanups.toml` (timing: tables `base` and `head`, each row
with subtables `set1`, `set2` and `all`, every one holding the per-launch arrays, `_min`,
`_max`, `_median` and the load before each launch; plus `swapped_order` and `summary`) and in
`benchmark/results/v4.1.3_type_cleanups_instances_base.toml` and
`benchmark/results/v4.1.3_type_cleanups_instances_head.toml` (instance counts).

## Machine state

All timing figures were taken under background load, so all of them are labelled "under
load". The load never fell below about 1.6 and the machine's own background (the Claude app,
WindowServer, `cloudd`) kept it near 2.

- Power: AC. Julia 1.13.2, 4 threads, default `-O` (the script refuses below `-O2`), the
  whole slot pool (`BRAMBLE_JULIA_SLOTS=4`), no other job of this plan running.
- Set 1. `check_power_load.sh` at the start read a 1-minute load of 1.58 on 8 logical cores.
  Repeated `gate_preflight.sh --timing` calls then failed the load check (2.14 to 3.85,
  threshold 2) and once the cpu check (OneDrive and its File Provider at 84% and 78% for a
  moment). Waiting a few minutes did not bring the load under 2 for long, so the run went
  ahead. The onedrive line stayed `ASK` (the user said sync is paused).
- Set 2 and the swapped runs. After about 3.5 minutes with nothing of this plan running,
  `gate_preflight.sh --timing` read load 1.70, power OK, no other julia, nothing above 50%
  CPU (onedrive again `ASK`). A second preflight call, made as the set 2 command began,
  read load 1.64 and listed one other julia process (pid 83692, gone when checked right
  after; not identified). The 1-minute load then rose to about 3.3 within the run, a rise
  caused by the 4-thread launches themselves.
- Load before each launch (1-minute average), in launch order:
  - set 1 base 2.09, 2.28, 2.48, 2.90, 2.82; head 1.99, 2.31, 2.51, 2.76, 2.78;
  - set 2 base 1.64, 2.83, 3.29, 3.19, 2.82; head 2.14, 2.79, 3.13, 3.06, 2.63;
  - swapped order (ghost_2d only) head 2.27, 2.28, 2.48, 2.38, 2.28; base 2.32, 2.26, 2.32,
    2.35, 2.18.
- Base and head launches alternated (base first, then head, in each of five rounds, in
  single-launch invocations) twice, set 1 and set 2: ten pairs. Each launch is one child
  process per row; a figure is the minimum of 30 warm calls in that child.

## Timing

Seeded non-uniform meshes (seed 5), serial policy, `benchmark/type_cleanups_ab.jl`. The
figures are ns per DOF, the range is the minimum to the maximum over all ten launches
(set 1 and set 2 together). The ratio is head minimum over base minimum over the ten
launches. The last column counts the pairs (launch k of base against launch k of head,
five in each set) in which head is above base.

| row | DOFs | figure | base min to max | head min to max | ratio of minima | head above base |
|---|---:|---|---|---|---:|---:|
| plain_2d | 90000 | assemble! | 6.558 to 7.014 | 6.404 to 6.610 | 0.9765 | 6 of 10 |
| plain_2d | 90000 | mul! | 5.449 to 5.726 | 5.566 to 5.792 | 1.0215 | 6 of 10 |
| plain_3d | 64000 | assemble! | 10.327 to 10.643 | 10.600 to 10.670 | 1.0265 | 7 of 10 |
| plain_3d | 64000 | mul! | 22.698 to 23.429 | 22.879 to 23.512 | 1.0080 | 5 of 10 |
| ghost_2d | 91200 | assemble! | 10.160 to 11.059 | 10.188 to 11.046 | 1.0027 | 6 of 10 |
| ghost_2d | 91200 | mul! | 9.150 to 9.376 | 9.454 to 9.504 | 1.0333 | 10 of 10 |
| kron_2d | 90000 | assemble! | 5.997 to 6.275 | 6.088 to 6.173 | 1.0153 | 6 of 10 |
| kron_2d | 90000 | mul! | 1.206 to 1.277 | 1.232 to 1.238 | 1.0219 | 4 of 10 |
| kron_2d_mf | 90000 | assemble! | 5.937 to 6.201 | 6.075 to 6.561 | 1.0232 | 8 of 10 |
| kron_2d_mf | 90000 | mul! | 1.690 to 1.739 | 1.397 to 1.740 | 0.8264 | 5 of 10 |

Each set alone, in the TOML under `set1` and `set2`, gives the same picture; the ratio of
minima per set for each row is under `summary`. The `kron_2d_mf` `mul!` head launches 4 and
5 of set 2 read 1.397 and 1.451 against about 1.72 to 1.74 in every other launch of either
tree. That is a jump of about 20% in two consecutive launches, not seen in set 1 or in
base, and is not explained; the 0.8264 ratio comes from it and is not a speedup to quote.

Allocations of one warm call, identical in all ten launches and on both trees: 0 bytes for
every `assemble!` and `mul!`, except `assemble!` on `ghost_2d`, 48 bytes on both trees
(48 / 91200 DOFs, about 0.0005 bytes per DOF). Allocation per DOF is therefore unchanged on
every row.

### ghost_2d mul!, three runs

ns per DOF, the five launches of each run in launch order, then minimum, median and maximum.

| run | tree | launches | min | median | max |
|---|---|---|---:|---:|---:|
| set 1, base first | base | 9.150, 9.357, 9.376, 9.364, 9.363 | 9.150 | 9.363 | 9.376 |
| set 1, base first | head | 9.454, 9.464, 9.479, 9.470, 9.454 | 9.454 | 9.464 | 9.479 |
| set 2, base first | base | 9.158, 9.372, 9.375, 9.370, 9.353 | 9.158 | 9.370 | 9.375 |
| set 2, base first | head | 9.504, 9.480, 9.489, 9.491, 9.468 | 9.468 | 9.489 | 9.504 |
| swapped, head first | head | 9.724, 9.709, 9.708, 9.692, 9.746 | 9.692 | 9.709 | 9.746 |
| swapped, head first | base | 9.387, 9.614, 9.594, 9.601, 9.624 | 9.387 | 9.601 | 9.624 |

Over the ten pairs of sets 1 and 2 head is above base in 10 of 10. The ratio of minima is
1.0333 (set 1 1.0333, set 2 1.0338) and the ratio of medians 1.0119. The head minimum
(9.454) is 0.078 ns per DOF above the base maximum (9.376), while base's own spread over
its ten launches is 0.226 (9.150 to 9.376) and head's is 0.050 (9.454 to 9.504). The gap
between the two ranges is therefore a third of the spread inside base, and base's own
minimum comes from its first launch of each set (9.150 and 9.158, the others 9.35 to 9.38).
In the swapped runs (head first, then base, each launch a single-row invocation) head is
above base in 5 of 5 pairs and the ratio of minima is 1.0325. The order of the two trees
in the sequence does not move the sign: head is the slower tree in both orders. The
swapped runs are all about 2.5% slower than the runs of sets 1 and 2 for both trees, with
a single-row process in place of a five-row one; that shift hits both trees alike.

## Instance counts

`benchmark/snoop_instances.jl`, three launches per tree, each a fresh process. The counts
agreed across the three launches within each tree.

| row | base fresh | head fresh | base added | head added | base sequence | head sequence |
|---|---:|---:|---:|---:|---:|---:|
| scale_float64 | 6 | 6 | 6 | 6 | 6 | 6 |
| closures_fresh | 221 | 216 | 180 | 177 | 401 | 393 |

The repeat control was 0 on both trees for both rows. The head count on `closures_fresh`
is lower by 5 than base (3 lower on `added`, 8 on `sequence`), and the criterion is "not
higher" (#930 requires that the count does not grow), so the lower count is not a
regression.

## Verdict

Under load, per row, whether the branch minimum lies inside base's minimum to maximum over
the ten launches (a branch minimum below base's minimum counts as not slower):

- plain_2d assemble!: below base's minimum (6.404 against 6.558), faster.
- plain_2d mul!: inside (5.566 in 5.449 to 5.726).
- plain_3d assemble!: inside (10.600 in 10.327 to 10.643).
- plain_3d mul!: inside (22.879 in 22.698 to 23.429).
- ghost_2d assemble!: inside (10.188 in 10.160 to 11.059).
- ghost_2d mul!: slowdown. The head minimum 9.454 lies above base's maximum 9.376 and
  the head range does not overlap base's; head is above base in 10 of 10 pairs, and in 5 of
  5 pairs with the order of the trees swapped, so it is a property of the tree and not of its
  position in the sequence. Ratio of minima 1.0333, ratio of medians 1.0119 (about 1.2%),
  gap between the ranges 0.078 ns per DOF. Small, but repeated and ordered the same way
  in every pair; it was not measured on a quiet machine.
- kron_2d assemble!: inside (6.088 in 5.997 to 6.275).
- kron_2d mul!: inside (1.232 in 1.206 to 1.277).
- kron_2d_mf assemble!: inside (6.075 in 5.937 to 6.201).
- kron_2d_mf mul!: head minimum 1.397 is below base's minimum 1.690, but the two head
  launches at 1.397 and 1.451 are an unexplained jump of the head figure in set 2, not a
  stable result; the other eight head launches read 1.715 to 1.740, inside base's range.

The only row where head is slower than base's whole range is `ghost_2d` `mul!`. The
`assemble!` ratios of minima run 0.9765 to 1.0265 and the other `mul!` ratios 1.0080
to 1.0219 (leaving out `kron_2d_mf`), the same size as the spread over launches inside a
tree on those rows. Allocations did not change. The instance count of `closures_fresh`
is lower on head (216 against 221), which meets the criterion.

The `ghost_2d` `mul!` slowdown above was addressed afterwards by restoring the stored `indices` field (S11). The three-tree measurement and its outcome are in `## S11`: medians, measured under load, base 9.619, branch without the fix 9.711 and with it 9.604 ns per DOF.

## Bisect

Question: which part of the branch carries the `ghost_2d` `mul!` shift. Single row
(`--rows ghost_2d`), `type_cleanups_ab.jl --launches 1`, 4 threads, default `-O`, the whole
slot pool, five rounds over the trees with the starting tree rotated each round so that
position cancels. All figures are under load: the 1-minute load before each launch is in
the TOML under `bisect` (`load_1min_before`, with `round_position`) and ran from 1.25 to
2.30. Before each of the two runs the machine sat idle for about 3.5 minutes; the
preflight then read load 1.66 and 1.30, power OK, no other julia, cpu OK, onedrive `ASK`
(the user said sync is paused).

### The four commits

Trees: base `d64ab24a`, wave 1 `669fbbbb`, wave 2 `ca3ebec0`, wave 3 `2b49c763`, head
`9f3ed728`. `mul!` in ns per DOF over the five rounds; the assemble median is for
reference. The count column is the rounds in which the tree is above base's launch of the
same round.

| tree | mul! min | mul! median | mul! max | above base | assemble! median |
|---|---:|---:|---:|---:|---:|
| d64ab24a (base) | 9.356 | 9.606 | 9.629 | n/a | 10.312 |
| 669fbbbb (wave 1) | 9.688 | 9.729 | 11.756 | 5 of 5 | 10.503 |
| ca3ebec0 (wave 2) | 9.709 | 9.734 | 9.763 | 5 of 5 | 10.312 |
| 2b49c763 (wave 3) | 9.460 | 9.731 | 9.748 | 5 of 5 | 10.442 |
| 9f3ed728 (head) | 9.702 | 9.734 | 9.746 | 5 of 5 | 10.314 |

The shift first appears at wave 1 (`669fbbbb`), whose median is 1.0127 times base's. Waves 2
and 3 and the head sit at the same level (medians 9.731 to 9.734, within 0.003 of wave 1), so
no later wave adds to it.

One wave 1 launch reads 11.756 (round 3). Its other four launches do not repeat it. Base's
first launch (9.356) and wave 3's first (9.460) read low in the first round, which is why
the medians are used. The `assemble!` medians follow no pattern: wave 1 (10.503) and wave 3
(10.442) are above base's 10.312, wave 2 and the head equal to it.

### Splitting wave 1

Two trees built from base by copying files from `669fbbbb` over a fresh detached worktree
of `d64ab24a`, `test/` untouched, then precompiled:

- `s2only`: `src/mesh/ghost_mesh.jl` and `src/utils/batch_split.jl` (S2);
- `s41only`: `ext/BramblePolyesterExt.jl`, `src/assembly/jacobian_pattern.jl`,
  `src/assembly/kronecker.jl`, `src/assembly/matrix_free.jl`, `src/mesh/mesh1d.jl`,
  `src/operators/cell_average.jl`, `src/operators/normal.jl`,
  `src/operators/stencil_matrix.jl`, `src/solvers/matrix_free_preconditioners.jl`,
  `src/utils/linear_algebra.jl` (S4.1).

Wave 1 changes other files too (`src/ast`, `src/geometry/domain.jl`,
`src/operators/average.jl`, `difference.jl`, `shift.jl`, `src/problems`, `src/space`
and `ext/BrambleKroneckerExt.jl`); they are in neither tree, and were not measured
alone. The four trees (base, wave 1, `s2only`, `s41only`) ran together for five rounds,
rotated.

| tree | mul! min | mul! median | mul! max | above base | assemble! median |
|---|---:|---:|---:|---:|---:|
| d64ab24a (base) | 9.363 | 9.618 | 11.576 | n/a | 10.346 |
| 669fbbbb (wave 1) | 9.693 | 9.722 | 9.753 | 4 of 5 | 10.387 |
| s2only | 9.466 | 9.696 | 9.722 | 4 of 5 | 10.307 |
| s41only | 9.364 | 9.617 | 9.631 | 2 of 5 | 10.430 |

Base has one high launch here too (11.576, round 4). Median ratios against base:
`669fbbbb` 1.0108, `s2only` 1.0081, `s41only` 0.9999. The S4.1 files alone leave the
figure where base has it; the S2 files alone move it up by most of the wave 1 shift
(0.8% against 1.1%, a difference of 0.026 ns per DOF, which the spread between launches
covers). The shift follows S2 (`src/mesh/ghost_mesh.jl`, `src/utils/batch_split.jl`), not
S4.1. That wave 1 sits a little above `s2only` leaves room for a small share from the
files in neither tree; it was not tested.

## S11

Question: does restoring the stored `indices` field remove the `ghost_2d` `mul!` shift
that the bisect put on S2? Candidate A puts the stored `indices` field back into
`GhostMesh` and `GhostMeshState`, without bringing back the `CI` parameter. The public type
stays `GhostMesh{D, M, P}`. The field is typed
`CartesianIndices{D, NTuple{D, Base.OneTo{Int}}}`, the type `generate_indices` gives every
padded mesh. The constructor stores the padded indices, `_walk_mesh` copies them into the
state, `indices(G::_GhostMeshLike) = G.indices` again, and `npoints`, `_padded_markers` and
`_ensure_geometric_markers!` read `size(G.indices)`. Only `src/mesh/ghost_mesh.jl`
changes, and `src/utils/batch_split.jl` stays as S2 left it.

Trees: base `d64ab24a`, head the branch at `31f5354e` (its `src/` differs from `9f3ed728`
in whitespace only), and A, a detached tree at `7a74c2ff` with the change above. `diff -rq`
of `src/` against head lists only `src/mesh/ghost_mesh.jl`, plus the two files `31f5354e`
reflows. Method of the bisect: `type_cleanups_ab.jl --rows ghost_2d --launches 1`, 4
threads, default `-O`, `BRAMBLE_JULIA_SLOTS=4`, nothing else running, a quiet minute
between rounds. Ten rounds ran in the cycle base, head, A, with the starting tree rotated
each round. Five more ran in the reversed cycle A, head, base, also rotated. A pair is the
two launches of the same round. Figures are in ns per DOF and are in the TOML under
`s11.rotated` and `s11.reversed`, each tree with its per-launch arrays, min, median, max,
`round_position` and `load_1min_before`.

Machine state: AC power, no other julia, cpu check OK, onedrive `ASK` (the user said sync
is paused). The preflight passed after about five minutes of waiting, when macOS indexing
daemons had settled. The 1-minute load before each launch ran from 1.41 to 2.96, mostly
1.6 to 2.3, so these figures are under load like the earlier ones.

| run | tree | mul! launches | min | median | max | above base |
|---|---|---|---:|---:|---:|---:|
| rotated | base | 9.622, 9.619, 9.618, 9.348, 9.593, 9.612, 9.621, 11.596, 9.622, 9.619 | 9.348 | 9.619 | 11.596 | n/a |
| rotated | head | 9.702, 9.715, 9.729, 9.467, 9.707, 9.704, 9.736, 9.722, 9.697, 9.766 | 9.467 | 9.711 | 9.766 | 9 of 10 |
| rotated | A | 9.631, 9.610, 9.623, 9.597, 9.608, 9.592, 9.600, 9.590, 9.597, 9.614 | 9.590 | 9.604 | 9.631 | 4 of 10 |
| reversed | base | 9.608, 9.614, 9.351, 9.623, 9.617 | 9.351 | 9.614 | 9.623 | n/a |
| reversed | head | 9.715, 9.721, 9.709, 9.698, 9.713 | 9.698 | 9.713 | 9.721 | 5 of 5 |
| reversed | A | 9.622, 9.616, 9.608, 9.597, 9.595 | 9.595 | 9.608 | 9.622 | 3 of 5 |

Ratios against base, rotated rounds: head minima 1.0128, medians 1.0096; A minima 1.0259,
medians 0.9985. Reversed rounds: head minima 1.0371, medians 1.0102; A minima 1.0261,
medians 0.9994. The ratio of minima is not a good measure here: base's minima are its two
low launches (9.348 and 9.351, both base starting its round, as in S8.2), and A never had
a low launch. Base's 11.596 (round 7) is a single high launch like the ones in the bisect.
A is below head in 9 of 10 rotated pairs and 5 of 5 reversed pairs.

`assemble!` (for reference; the change does not touch it): medians base 10.158, head
10.320, A 10.323 in the rotated rounds, and 10.267, 10.344, 10.339 in the reversed ones.
Head and A are alike and both have their minimum inside base's range (10.079 to 10.687).
Allocations are those of S8.2 on all three trees (`mul!` 0 bytes, `assemble!` 48 bytes).

The control worked: head without A showed the shift again (9 of 10 and 5 of 5 pairs above
base, median ratio about 1.010, as in S8.2 and the bisect), so this run could detect the
shift.

Verdict against the MANUAL criterion: met. A's median (9.604) lies inside base's min to max
(9.348 to 11.596), and also inside base's eight launches other than the two low ones and
the high one (9.593 to 9.622). A is above base in 4 of 10 pairs, which is at most 6. In
the reversed order it is 3 of 5. The stored field brings `ghost_2d` `mul!` back to base's
level. The mechanism is not identified: the indices are read once per product, not per
point, so the extra indirection of `indices(G.padded)` alone does not explain about 1%.
The struct layout or code placement that the stored field changes may. Not pinned further.
