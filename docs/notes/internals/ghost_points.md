# Ghost points for Neumann and Robin conditions: structures, API and evidence

Design history for milestone v4.0.0 (gpena/Bramble.jl#218, #205; #774 for the form-level
`Δₕ`). Nothing here is published, and no issue has been edited: the acceptance-criteria
edits below are proposals for the maintainer to read first.

The evidence is a set of throwaway prototypes. The three plain-matrix oracles and their
tests are merged and run in the test tree under `test/ghost/`. The others stay on the local
branch `proto/v4-ghost-points`, under `test/prototypes/ghost/`. Nothing under `src/`,
`ext/` or `docs/src/` changed. Every claim below cites a prototype test file and testset,
or a figure printed by one. Files and what they pin:

| file | where | pins |
|---|---|---|
| `oracle_1d.jl` (`GhostOracle1D.jl`) | `test/ghost/` | 1D Neumann and Robin as explicit sparse matrices on the extended index range, independent of the form layer |
| `oracle_nd.jl` (`GhostOracleND.jl`) | `test/ghost/` | the same in 2D and 3D with facet-only ghosts, built from the 1D factors |
| `corners.jl` (`GhostCorners.jl`) | `test/ghost/` | a mixed-derivative problem with conormal Robin data, three corner closures |
| `mode_a_existing_forms.jl` | local branch | today's forms against the oracles' ghost-eliminated systems |
| `pure_neumann.jl` | local branch | four mean-zero treatments of pure Neumann |
| `ghost_layout.jl` (`GhostLayoutProto.jl`) | local branch | facet-only and padded ghost layouts, `GhostVector`, ghost updates |
| `ghost_layout_bench.jl` | local branch | stencil and copy cost on both layouts |
| `embed_forms.jl` (`GhostEmbedProto.jl`) | local branch | Bramble forms on a ghost-padded mesh against the oracles |

Files marked "local branch" are under `test/prototypes/ghost/` on `proto/v4-ghost-points`.

Notation: `u_G` is the ghost beyond facet node `I` of facet `F` (normal axis `d`, outward
normal `η_F`), at `x_1 − h_2` or `x_n + h_n` along `d`, which is what `spacing(Ωₕ, 1)` and
`forward_spacing(Ωₕ, n)` already hold. Orders are the least-squares slope of the discrete
`H¹` error against `hₘₐₓ` over four meshes (2D from 9 points per axis, 3D from 5, 1D and the
corner problem from 17). "Padded mesh" is a Bramble mesh with one extra point at each named
facet end, built from the physical mesh with `change_points!`.

## Decisions

1. Ghost-eliminated Neumann and Robin for Poisson and Helmholtz need no ghost code: today's
   forms already assemble them (Mode A). #205 keeps tests, the mean-zero constraint and
   the `∂ₙ` forms; it loses the new boundary form terms and Mode B.
2. A ghost space is a mesh padded by one point at each named facet end. Every difference,
   average and nested composition is ghost-aware on it with no change to an operator family.
   What is missing is the layer above: physical-only weights, markers and `inner_Γ` on the
   physical facets, `∂ₙ`, and a constrained space. These are the seven gaps of S5.
3. Corners and 3D edges carry no degrees of freedom the user sees. The mixed difference at
   the four 2D corners is the inward one-sided one.
4. Assembly uses the padded mesh. A grid function that carries ghosts (#218's `GhostVector`)
   is facet-only: the solver keeps its own `Vector` and no copy is made.
5. The discrete zero mean is a bordered system, not a stored rank-one term, which is dense.
6. A form-level `Δₕ` is one operator node on trial and test functions.
7. A boundary relation is written as an operator and its data, `α * u + β * ∂ₙ(u) => g`,
   and `constrained` collects the relations of the ghosted facets. Dirichlet is offered two
   ways (user, 2026-10-15): today's row replacement by `assemble(...; dirichlet)` and
   `symmetrize!`, unchanged and keeping every existing program working, with the unknowns the
   physical nodes of the space without ghosts; or a node relation `u => g` in `constrained`,
   which eliminates the Dirichlet nodes through `E` and is symmetric by construction.
8. The user-facing interface stays today's: one new keyword on `gridspace`, one on
   `pde_solve`, three new names (`constrained`, `∂ₙ`, `∂ₙₙ`), the `=>` relation form
   (syntax, not a name) and a method of the existing `Δₕ`.

## Requirements by scheme

The hypotheses of the planning notes, confirmed or killed. `Mesh`, `Space` and `Form` name
the layer that has to provide the item.

### Neumann and Robin in 1D

Neumann `β η u' = g` and Robin `α u + β η u' = g` at the end nodes, `−u'' = f`, with the
ghost eliminated: `u_0 = u_2 + (2 h_2/β)(g_1 − α u_1)`, and the mirror at the right end.

| layer | requirement | verdict and evidence |
|---|---|---|
| Mesh | ghost coordinates and spacings from the mesh's own arrays | confirmed: `oracle_1d.jl` "Mesh facts", `ghost_layout.jl` "ghost coordinates" |
| Mesh | ghosts follow `change_points!` | confirmed: `ghost_layout.jl` "layouts follow change_points!" |
| Space | quadrature weights on physical points only | confirmed needed: unweighted padded assembly differs from the physical one by 0.76 (S5 critic) |
| Space | ghost closure `u_ext = E u_phys + d` | confirmed: `oracle_1d.jl` "Ghost closure" for Neumann, Robin and Dirichlet relations |
| Form | boundary rows of every operator read the ghost | confirmed entry by entry on non-uniform meshes: `oracle_1d.jl` "Boundary rows"; a mutant (one-sided `Dc` relation) fails 27 of 71 tests |
| Form | eliminated row times its cell measure is the variational row | confirmed: `oracle_1d.jl` "Eliminated rows are variational"; the summation-by-parts identity holds numerically ("Summation by parts") |
| all | second order | measured: Neumann 1.999 uniform, 2.035 graded, 2.002 random; Robin 2.000, 2.046, 2.001. Augmented (`_b`) and eliminated (`_a`) errors agree to 1e-13 |

### Neumann and Robin in 2D and 3D

| layer | requirement | verdict and evidence |
|---|---|---|
| Mesh | facet nodes, ghost count `Σ_d 2 Π_{e≠d} n_e`, boundary to ghost maps in O(1) | confirmed: `ghost_layout.jl` "counts and named facets", "ghost maps round-trip" (1D, 2D, 3D, named facets, mixed uniform axes) |
| Mesh | padded corner and edge points are not ghosts | confirmed: `oracle_nd.jl` "Δ_H never reads absent points", with a positive control (`Dcₓ(Dcᵧ(u))` does read them) |
| Space | elimination per (facet, node) along the normal; a corner node carries one datum per incident facet | confirmed: `oracle_nd.jl` "Closure per facet and node"; the data enter as `ĥ f + Σ_F ω_F g_F/β_F`, one term per incident facet |
| Space | facet weight `ω_F(I)` equals Bramble's `_transverse_measure` | confirmed exactly, half cell at corners, on 2-point and skewed (ratio 2000) meshes (S2.1 critic) |
| Form | scalar `g` on `:boundary` is ambiguous at a corner | confirmed: `mode_a_existing_forms.jl` "Scalar g on :boundary at a corner": the lumped scalar differs from the per-facet flux by −0.330 at corner (1,1) |
| Form | nested evaluation at a ghost needs a second layer, absent | confirmed in the sense that no tested row needs it: `Δₕ` rows at physical nodes never read a padded corner or edge point |
| all | second order | measured, uniform / graded / random: 2D Neumann 1.998 / 2.077 / 1.999, 2D Robin 1.997 / 2.067 / 1.984, 3D Neumann 1.999 / 2.039 / 1.978, 3D Robin 1.983 / 1.996 / 1.943. `_a` and `_b` errors agree to about 1e-12 |

### Conormal Robin with a mixed derivative (2D)

`−∇·(K∇u) = f`, `K = [1 .5; .5 2]`, `α u + β (K∇u)·η_F = g_F` on every facet.

| layer | requirement | verdict and evidence |
|---|---|---|
| Space | at a corner the two facet relations couple the two facet ghosts: a 2×2 solve per corner, not one row of `E` per ghost | confirmed: `corners.jl` "Corner relations are 2×2". Block determinant `β_x β_y det(K)/(4 h_x h_y)`, solvable exactly when `K` is non-singular |
| Form | the mixed difference reads the corner point at the four corner nodes only | confirmed: `corners.jl` "Which rows read a corner" |
| Form | an `O(1)` truncation at four points may break second order | killed: orders 2.0 (uniform), 1.885 (random), 2.03 (graded) for facet-only, see "Corner policy" |

### What the three need that forms do not yet say

Everything above was built through Bramble forms on a padded mesh in `embed_forms.jl`, and
each form-level matrix equals the oracle's: Neumann and Robin, `_a` and `_b`, 1D (random
and graded meshes) 1.2e-16 and 2.7e-16, 2D 1.0e-16 and 2.2e-16, 3D 1.4e-16 (S5 critic,
outside its goal), the corner problem 1.3e-16 and 1.7e-16. Every operator family is
ghost-aware on the padded mesh: `D₋ₓ`, `D₊ₓ`, `Dcₓ`, `D̃ₓ`, `D̽ₓ`, `M₊ₓ`, `Mₕ`, `jumpₓ`, `D̃ₓ(D₋ₓ)`
and a depth-4 composition (interior rows) agree with the explicit 1D rows to 2.3e-16, mixed
`Dcₓ(Dcᵧ)` and `D̽ₓ(D̽ᵧ)` in 2D to 4.5e-17 and 8.6e-17 (`embed_forms.jl` "Operator rows of
notes §6", kept as a regression).

The prototype needed seven `# WORKAROUND:` lines. Each was checked a real gap by trying
the obvious supported spelling (S5, second critic round). They are the core of the
proposal and each maps to a structure below:

| # | gap | trial and result | proposed home |
|---|---|---|---|
| 1 | physical-only weights on a padded mesh | unweighted padded assembly differs by 0.76 | ghost space weights |
| 2 | markers lifted to padded indices | `:xmin` names the ghost layer, not the facet nodes | ghost mesh markers |
| 3 | `inner_Γ` faces on the physical facet nodes | refused (not whole boundary faces); the padded face differs by 1.85 | `inner_Γ` on a ghost space |
| 4 | `∂ₙ` | absent; `normal_vector` refuses off whole faces | `∂ₙ` |
| 5 | corner DOFs to drop or keep | no `gridspace(Ω; ghosts)` | ghost mesh `absent` set |
| 6 | boundary-condition rows on ghost test functions | no form names them | ghost relations |
| 7 | `constrained` | absent | constrained space |

The grid-function coefficient that rescales a data term is not a gap: a coefficient on the
test function, `innerₕ(f, c * v)`, gives the identical vector (S5 critic, round 1).

## Mode A with existing forms

Mode A is #205's direct elimination: no ghost degrees of freedom, an `N_phys × N_phys`
matrix. `mode_a_existing_forms.jl` pins that today's forms already assemble it:

```julia
# Robin, one inner_Γ per face; α, β, g per facet
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) +
    inner_Γ((α₁ / β₁) * u, v; markers = (:xmin,)) + inner_Γ((α₂ / β₂) * u, v; markers = (:xmax,)))
l = form(Wₕ, v -> innerₕ(f̂ₕ, v) + inner_Γ(x -> g₁(x) / β₁, v; markers = (:xmin,)) +
    inner_Γ(x -> g₂(x) / β₂, v; markers = (:xmax,)))
# Neumann
l = form(Wₕ, v -> innerₕ(f̂ₕ, v) + inner_Γ(dot(F, η), v; markers = (:boundary,)))
```

Against the oracles' ghost-eliminated `:robin_a` and `:neumann_a` systems, on random
non-uniform meshes, matrices and right-hand sides agree to 1.3e-16 (1D), 1.8e-16 (2D) and
2.2e-16 (3D), every matrix exactly symmetric, with and without a Helmholtz term (κ = 3).
The test would fail if the data were wrong: swapping `α` between two faces fails at 7.5e-3,
and lumping the data at one 3D corner or edge node fails at 1.5e-2 and 0.10 (S3.1 critic).
Solves through Bramble on graded 2D meshes give orders 2.067 (Robin) and 2.077 (Neumann,
bordered mean-zero), with errors equal to the oracle's to 1e-8 and a ratio between 3.5 and
4.5 at every halving.

What Mode A cannot say today, the two `# GAP:` lines of the file:

- **The mean-zero constraint.** No form term adds it; the test borders the matrix by hand.
  See "Mean-zero constraint".
- **Per-facet data in one term.** A scalar `inner_Γ(g, v; markers = (:boundary,))` lumps
  both facets' weights onto one value at a corner. One `inner_Γ` per face, or a vector
  field through `dot(F, η)`, gives per-facet data. This is a spelling, not a change: no
  new name is proposed for it. The corner case is tested in 2D only; the 3D edge lumping is
  stated by the weight definition (`_surface_weight`) and was seen to matter in the critic's
  mutant, not pinned by a test.

Consequences for #205:

- The form terms `neumann(marker, g, v)` and `robin(marker, α, β, u, v)` of #205 are sugar
  for `inner_Γ(...)` spellings that already give the oracle's matrices. Proposal: do not add
  them, as form terms or as condition constructors; boundary relations on a ghost space are
  written `operator => data` (see "Proposed API").
- Mode A is not "the default mode of a new feature". It is today's behaviour and its
  deliverable is the test file.
- Mode B (the augmented system, ghosts as unknowns with boundary-condition rows) has no
  customer once Mode A is today's forms: `:neumann_b` and `:robin_b` give the physical
  values of `_a` to 1e-13 in 1D and about 1e-12 in 2D/3D (`oracle_1d.jl`, `oracle_nd.jl`
  "Augmented equals eliminated"), and the prototype assembled them through forms (`embed_forms.jl`
  "Neumann and Robin systems"). The unconstrained ghost space needed to build `E` already
  holds the augmented matrix. Proposal: no public Mode B.

## Corner policy

Measured on the problem of "Requirements by scheme" (`corners.jl` "Convergence", H¹, four
levels from 17 points per axis), orders on uniform / random / graded meshes, mixed term as
`Dcₓ(Dcᵧ)`:

| closure | order |
|---|---|
| facet-only, one-sided inward mixed difference at the four corner nodes | 2.003 / 1.885 / 2.03 |
| padded, corner ghost by bilinear extrapolation `u_c = u_a + u_b − u_n` | 1.992 / 1.886 / 2.024 |
| padded, corner mixed difference extrapolated from the three adjacent cells | 2.0 / 1.885 / 2.03 |

The cross-weighted `D̽ₓ(D̽ᵧ)` equals `Dcₓ(Dcᵧ)` on uniform meshes and is slightly better on
graded ones (2.4e-4 against 3.3e-4); on random meshes its errors are 6 to 11 times larger,
still at order 2.13. Only the mixed term at the four corner nodes reads the corner point.

The S2.2 critic separated the closures with a more anisotropic tensor:

- `padded_extrapolate` loses order: `K = [1 .9; .9 1]` uniform gives 1.598 (error ratios
  about 3), `k₁₂ = .99` gives 1.332 (graded 1.481).
- Facet-only and the three-cell closure stay at 2.0.
- For `k₁₂ = .99` on a graded mesh the three-cell closure's errors are non-monotone and the
  matrix is worse conditioned (7.5e5 against 1.1e4); facet-only was monotone in every case
  tried.

Policy:

1. **No corner or edge degrees of freedom in any space the user sees.** Facet-only ghosts
   are the whole ghost set. Corner storage buys nothing here and costs conditioning or order.
2. **At the four 2D corner nodes the mixed term is the inward one-sided difference.** This
   is the facet-only closure; it is exact for bilinear `u` and second order in the corner
   problem.
3. **The relations at a corner are a 2×2 block.** The constrained space solves it per
   corner; the matrix `E` is then not one row per ghost with physical-only entries. The
   block is solvable exactly when `K` is non-singular.
4. **The padded route has no one-sided closure.** S5 ran `padded_extrapolate` and
   `padded_mixed` through forms ("Anisotropic corner problem", both at 1.3e-16 and
   1.7e-16 from the oracle) but not the facet-only closure. Whether the library implements
   the one-sided rule as a tap rule in the stencil engine or as an `E` row that reproduces it
   is open (see "Open questions"). The rule, not the storage, is the decision.

`ghost_closure` in the prototypes uses dense solves and is not meant for reuse at scale.

3D edges are counted, not solved: on a 4×5×6 mesh the padded layout has 216 extra points,
148 of them facet ghosts and 68 edge and corner points (60 on the 12 edge lines, 8 corners).

## Layout decision

Two layouts were built on equal terms (`ghost_layout.jl`, 1D to 3D, `@inferred`, 0 bytes
after warm-up for `update_ghosts!` with Neumann, Robin and an affine closure, and for the
Laplacian applied with ghosts, `ghost_layout.jl` "updates allocate nothing"):

- **Facet-only** (#218 as filed): the physical values are the very `Vector` a solver sees,
  the ghosts a facet buffer.
- **Padded**: one array padded at the named facets, the physical part a view.

DOF order for both: physical nodes in mesh linear order, then facet ghosts, facets
`xmin, xmax, ymin, …`, each facet's ghosts in the mesh's own (column-major) order; the
padded layout then appends corner and edge points. The padded unconstrained DOFs are
therefore the facet-only DOFs followed by a suffix that is dropped.

Cost (`ghost_layout_bench.jl`, 2D 256² and 1024², 3D 64³, random meshes, `--threads=4`,
Apple M2, Julia 1.13.1; all measurements were taken under load, one-minute load average
2.4 to 3.6, so absolute times are biased up and ratios less so). The first measurement
showed facet-only at 2× a contiguous array. The critic traced it to the prototype's kernel
(per-read box tests in `GhostVector`'s `getindex`, `@inbounds` unreachable), not to the
layout. With kernels written row-wise (SIMD inner loop, boundary handled once per row; the
critic's probes, one thread, load about 3):

- Δ-stencil time against a contiguous row-wise kernel: padded 0.95 (3D 64³) and 0.97 (2D
  256²); facet-only 1.004 (3D) and about 1.0 to 1.1 (2D).
- Neither layout pays a SIMD or latency penalty, provided the kernel avoids per-read box
  tests and keeps `@inbounds` reachable.
- The padded layout's costs: a copy in and out of about 2× (2D) to 2.4× (3D) a stencil
  application when the solver keeps its own contiguous vector (`copyto!` through the
  D-dimensional view, 2D 256² 2.0×, 3D 1.06× against the first fair kernel); a product on
  the view with a hand-written CSC loop costs at most 1.07× a contiguous product; the stock
  `mul!(y, A, physical(v))` misses SparseArrays' CSC kernel (the view is a `ReshapedArray` of
  a non-contiguous `SubArray`) and falls back to the generic O(N²) method, 40,700× slower at
  256².
- Facet-only keeps the solver's `Vector` as is: no copy, stock `mul!` works.

Decision:

- **Assembly route: the padded mesh.** S5 showed it needs no change to any operator family
  (the seven gaps above are all above the operator level), spacings already follow the
  ghost convention, and the separable assembly path is reused.
- **Grid functions that carry ghosts: facet-only `GhostVector`.** The padded layout's
  extra cost (a copy, a stock-`mul!` trap) buys nothing the assembly route does not already
  have, and corner storage is not needed (S2.2). A runtime operator and an assembled stencil
  are already separate code paths in Bramble, so two representations add no new split.
- **The two are connected by the DOF order**: dropping the suffix of padded points (corner
  and edge) leaves the facet-only DOFs.

The "no SIMD or latency penalty" criterion of #218 is met by either layout for row-wise
kernels, and is a kernel-writing constraint, not a layout choice. The criterion should be
rewritten to name the kernel shape (see the proposed edits).

## Mean-zero constraint

A pure Neumann Poisson problem on 2D graded meshes assembled with today's forms, error in
the discrete `H¹` norm against the exact solution shifted to discrete mean zero
(`pure_neumann.jl`, n = 9 to 65 per axis):

| treatment | order | notes |
|---|---|---|
| dense penalty `A + σ w wᵀ`, `w` the cell measures, `σ = γ_m/hₘₐₓ` | 2.038 | stored, `nnz = N²` (asserted); at N = 4225 `lu` takes 1.415 s |
| bordered `[A w; wᵀ 0]` | 2.037 | `lu` 7.7 ms at N = 4225 |
| one DOF pinned, shifted to mean zero afterwards | 2.032 | `lu` 4.2 ms; error about 9% larger, same order |
| penalty as a rank-one operator, Jacobi CG | 2.038 | 448 iterations at the finest level, doubling per halving of `h` |

Timings are minima of three runs of one process, printed and not gated. The order gate does
not discriminate `σ` (any `σ > 0` gives the same mean-zero solution).

#205 asked for "a rank-one modification, not a new space type". Stored in a sparse matrix a
rank-one term fills the matrix, so that representation is dense and does not work with the
sparse direct solvers. Proposal:

- The constraint is a bordered system, applied by `pde_solve` from the weight vector (see
  "Proposed API"). The assembled `A` stays sparse and singular and is not modified; the
  caller never sees the `N + 1` system.
- The rank-one operator is kept for iterative solvers, which need a preconditioner that
  accounts for the penalty.
- Pinning stays available as a symmetric positive definite alternative at about 9% more
  error.

Diagnostics (`pure_neumann.jl` "flags a form that kills constants", "flags incompatible
Neumann data"):

- **Singular form.** `‖A 1‖∞ ≤ 1e-10 ‖A‖∞` flags a bilinear form that annihilates constants.
  This is exact and cheap (one product).
- **Compatibility of the data.** `innerₕ(f, 1) + inner_Γ(g, 1) = 0` holds only up to the
  `O(hₘₐₓ²)` quadrature defect. The prototype's relative threshold `hₘₐₓ²/10` passed its own
  data and was shown (two critic rounds) to accept an `O(1)` defect when set to `hₘₐₓ`, and to
  reject exactly compatible harmonic data when set to `hₘₐₓ²/10` (defect over tolerance
  about 3 at every n from 9 to 129). Ruling: a single-mesh threshold on `|ΣF|/Σ|F|`
  cannot separate an `O(h²)` quadrature defect from a small incompatibility. The check
  needs a data-scaled reference (the defect on a refined mesh, or the quadrature error of the
  data), or it should be a warning only. Proposal: a warning that reports the relative defect,
  never an error.

## Proposed structures

New types and functions, each with the `src/` file it would live in and the evidence behind
it. Names follow `bramble-conventions`; none besides those in "Proposed API" is exported.

| structure | what it is | `src/` file | evidence |
|---|---|---|---|
| `GhostMesh` | the physical mesh, its padded mesh (one point at each named facet end, from the mesh's own spacings), the padded indices of physical nodes, facet ghosts (with `(facet, node)` per ghost) and absent corner/edge points; follows `change_points!` | `src/mesh/ghost_mesh.jl` (new) | `embed_forms.jl` "Padded mesh and markers"; `ghost_layout.jl` "ghost coordinates", "layouts follow change_points!" |
| ghost markers | physical box and each facet's physical nodes as markers on the padded mesh, re-evaluated when points change (gaps 2, 5) | `src/mesh/marker.jl`, `src/mesh/ghost_mesh.jl` | `embed_forms.jl` "Padded mesh and markers" |
| ghost space weights | per-axis cell measures, 0 at the ghost end points of each axis: the tensor product is the physical weight at physical nodes and 0 at every ghost and absent point (gap 1) | `src/space/inner_product.jl` (`SpaceWeights`), `src/space/ghost_gridspace.jl` (new) | derived from the per-axis storage of `SpaceWeights`; the physical sums are `embed_forms.jl`'s `physical_weight` |
| `inner_Γ` on the physical facets | `_surface_weight` and `_transverse_measure` resolved on the physical facet nodes of the ghost mesh (gap 3) | `src/mesh/queries.jl`, `src/operators/normal.jl` | `oracle_nd.jl` facet weight equals `_transverse_measure`; `embed_forms.jl` "Neumann and Robin systems" |
| `∂ₙ(u)`, `∂ₙ(u; K)` | `dot(Dcₕ(u), η)` per facet, and the conormal `dot(K * Dcₕ(u), η)` for a constant tensor `K`; meaningful inside `inner_Γ` and in a `constrained` relation, one value per facet and two at a 2D corner; refused on a space without ghosts on the named facets (gap 4) | `src/operators/normal.jl` (replaces the note at line 175), `src/operators/difference.jl` | `embed_forms.jl` `normal_derivative` and `ConormalRobin`, "Refusals"; `corners.jl` |
| `∂ₙₙ(u)` | the normal second difference `D̃_d D₋_d` at a facet node, reading the ghost | `src/operators/normal.jl`, `src/operators/difference.jl` (the per-axis part of the `Δₕ` node) | the per-axis row of `oracle_1d.jl` "Boundary rows"; not run through forms |
| form-level `Δₕ` | one `LazyOp` node, see "Proposed API" | `src/operators/difference.jl`, `src/operators/vector_calculus.jl` | `embed_forms.jl` `laplacian(u, D)`; #774 |
| ghost relations | the relation `operator => data` of each ghost's facet, a `LazyOp` of the trial placeholder, assembled by forms on the node's test function and moved to the ghost's row (gap 6) | `src/assembly/ghost_relations.jl` (new) | `embed_forms.jl` "Ghost closures": `E`, `d` from form-assembled relation rows match all three oracles' closures |
| `ConstrainedGridSpace` | a ghost space with the closure `u_ghost = E u_phys + d`; `E` is the identity on every physical node (Dirichlet nodes included: they stay unknowns) and has non-trivial rows for ghosts only; trial and test functions go through `E`; the 2×2 corner block of conormal data is solved in its construction (gap 7) | `src/space/constrained_gridspace.jl` (new), `src/assembly/` (substitution at each tap) | `embed_forms.jl` "Constrained matrix is EᵀAE": equals `Eᵀ A_full E`, cross-checked through runtime operators |
| `GhostVector` and `update_ghosts!` | facet-only: the solver's contiguous `Vector` plus a facet buffer; Neumann, Robin and affine closures; 0 bytes, `@inferred`, 1D to 3D | `src/space/ghost_vector.jl` (new), `src/space/vectorelement.jl` | `ghost_layout.jl` "updates allocate nothing", "ghost maps round-trip", "copy in and out in DOF order" |
| ghost-aware runtime operators | `D₋`, `D₊`, `M₋`, `M₊`, `Dc`, `∇ₕ`, `D̽` read the ghost neighbour on ghosted facets; row-wise kernels | `src/operators/difference.jl`, `src/operators/average.jl` | `ghost_layout.jl` "Neumann and Robin are exact": `Δ_H` exact on quadratics at boundary nodes of a random mesh |
| `pde_solve` mean-zero | bordered solve from the weight vector; singularity and compatibility diagnostics | `src/solvers/pde_solve.jl` | `pure_neumann.jl` |
| precompile workload | the nested forms of "Revised acceptance criteria" | `src/precompile/form_sessions.jl` | #774 |

The constrained space has a requirement: its matrix equals `Eᵀ A_full E`, `A_full` the same
form assembled on the unconstrained ghost space. The prototype computes it that way; the
library would substitute `E` at each stencil tap so `assemble!` builds the constrained
pattern directly, with `Eᵀ A_full E` as its test oracle. On a ghost space whose weights are
physical-only, `Eᵀ A_full E` equals `R A_full E` for a collocation form (ghost rows carry
weight 0), so one reduction serves both.

Prototype refusals to keep, all `ArgumentError`: `constrained` rejects conditions that do
not match the ghosted facets and a corner closure on a space that is not padded on every
facet (`embed_forms.jl` "Refusals"); the ghost mesh rejects a facet the dimension does not
have ("Padded mesh and markers").

## Proposed API

Today's interface stays: `mesh`, `gridspace`, `form`, `innerₕ`, `inner₊`, `inner_Γ`, the
operators, `assemble`, `pde_solve`. The additions below are the fewest that cover the seven
gaps and the mean-zero constraint; each lists the user code before and after. Names that
are not new and signatures that change are listed in the last subsection.

### Ghosts on a gridspace: `gridspace(Ωₕ; ghosts)`

`ghosts` is `:boundary` or a tuple of facet names (`:xmin`, `:xmax`, `:ymin`, …); only the
named facets are padded. Default `ghosts = ()`: every existing call is unchanged.

Before (S5, the seven workarounds, abridged): the user pads the mesh by hand with
`change_points!`, re-declares the face markers on the padded domain, multiplies every term
by a coefficient `ĥ_phys/ĥ_pad`, resolves facets with `restrict_to`, and writes boundary
rows at matrix level.

After:

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (65, 65), (false, false))
Wₕ = gridspace(Ωₕ; ghosts = :boundary)      # facet ghosts on all four facets
Wₓ = gridspace(Ωₕ; ghosts = (:xmin,))        # a ghost layer on one facet only
```

`innerₕ`, `inner₊` and `inner_Γ` on `Wₕ` use physical-only weights, and `markers` name the
physical facet nodes. Every operator reads the ghost neighbour at a ghosted facet and
truncates, as today, at an unghosted one.

### Constrained spaces: `constrained` and relations written `operator => data`

A relation is a pair of a facet name and an `operator => data` pair, in the pair idiom of
`dirichlet_constraints(Ωₕ, :left => x -> 0.0)`. `constrained(Wₕ, relations)` takes the
relations of the ghosted facets and returns the space whose unknowns are exactly the
physical nodes of the space without ghosts.

```julia
Cₕ = constrained(Wₕ, u -> (
    :xmin => ∂ₙ(u) => g₁,                  # Neumann
    :xmax => α * u + β * ∂ₙ(u) => g₂,      # Robin, β ≠ 0
    :ymin => ∂ₙₙ(u) => g₃))                # a second-order normal condition
```

`=>` is right-associative and binds looser than `+` and `*`, so
`:xmax => α * u + β * ∂ₙ(u) => g₂` is `:xmax => ((α * u + β * ∂ₙ(u)) => g₂)`. The data `g` is
a number or a function of the point tuple, as in `dirichlet_constraints`. `∂ₙ` and `∂ₙₙ` are
the only operators a relation may use besides the placeholder itself (`u`, `α * u`, sums);
each relation must contain the ghost of its facet with a non-zero coefficient, which is what
`β ≠ 0` says for Robin.

- **Where the placeholder comes from.** `constrained` calls the given function with a trial
  placeholder, the way `form(Wₕ, Wₕ, (u, v) -> ...)` calls its closure. S5 built every
  relation as a `LazyOp` of a trial function (`relation(bc, u, D)` in `GhostEmbedProto.jl`)
  and assembled it by forms on the facet node's test function, so a function of `u` reuses
  that path unchanged. The alternatives were a symbol `u` bound by `constrained` (a macro or
  a reserved name, with no precedent in the API) and the placeholder passed per relation
  (`:xmin => u -> ∂ₙ(u) => g` parses as `(:xmin => u) -> ...`, so it needs parentheses on
  every line). One lambda over all facets states the placeholder once.
- **Conormal data.** `∂ₙ(u; K)` with `K` a constant matrix is `dot(K * Dcₕ(u), η)`: the
  conormal flux `(K∇u)·η_F` with the centred `Dc` along every axis, which is the relation
  `ConormalRobin` of S5 and the conormal Robin problem of S2.2
  (`α u + β (K∇u)·η_F = g`, the 2×2 corner block of "Corner policy"). A keyword on `∂ₙ` adds no
  name, whereas `dot(K * ∇ₕ(u), η)` would use the backward gradient, which is truncated on
  the min facets and is not the centred difference these relations need. `K` is constant in
  S2.2 and S5; a grid function `K` is an open question.
- **`∂ₙₙ(u)`** is introduced generically, as a second-order normal condition: the normal
  second difference `D̃_d D₋_d` at the facet node, `(u_2 − 2u_1 + u_0)/h_2²` on a min facet
  (with `h_1 := h_2`) and the mirror on a max facet. At a corner each facet's relation reads
  only its own ghost, so no 2×2 coupling arises. It is one axis of the form-level `Δₕ` node
  ("Form-level `Δₕ`"), so its stencil is the 1D row of `oracle_1d.jl` "Boundary rows".
  S5 did not build a relation with it through forms.
- **Dirichlet, two ways (user, 2026-10-15: "i like both options, for now").** A relation
  list may also hold `u => g`, and the facet name may repeat to give a node relation and a
  ghost relation on the same facet. Both ways reach the same discrete solution; they differ
  in the unknowns and in the matrix.

  *(a) Row replacement, today's route, unchanged.* `constrained` is given ghost relations
  only. It eliminates ghosts and never physical nodes, so the unknowns are exactly the
  physical nodes of the space without ghosts, Dirichlet nodes included. The ghost closures
  are written in terms of the boundary node's own value, which is an unknown; the row
  replacement then fixes that value afterwards, as on a space without ghosts:

  ```julia
  # before: no ghosts, Dirichlet on :ymax by row replacement
  Wₕ = gridspace(Ωₕ)
  A, F = assemble(a, l; dirichlet = :ymax => h, symmetrize = true)
  # after: Neumann/Robin by ghosts on the other facets, Dirichlet unchanged
  Wₕ = gridspace(Ωₕ; ghosts = (:xmin, :xmax, :ymin))
  Cₕ = constrained(Wₕ, u -> (:xmin => ∂ₙ(u) => g₁, :xmax => ∂ₙ(u) => g₂,
      :ymin => ∂ₙ(u) => g₃))
  a = form(Cₕ, Cₕ, (u, v) -> innerₕ(-Δₕ(u), v))
  l = form(Cₕ, v -> innerₕ(f̂ₕ, v))
  A, F = assemble(a, l; dirichlet = :ymax => h, symmetrize = true)
  ```

  Data come from the markers or `dirichlet_constraints`, and `symmetrize!` acts on the
  assembled system, on regular and ghost spaces alike. The Dirichlet facets need no ghosts,
  since their rows are replaced; `ghosts` names only the facets with a ghost relation. Every
  existing program keeps working, since route (a) changes no existing call. The matrix
  `Eᵀ A_full E` of a self-adjoint form is exactly symmetric, but the row replacement makes it
  non-symmetric until `symmetrize!`, as today; the vectors keep the length `N_phys`.

  *(b) A node relation `u => g` in `constrained`.* The Dirichlet nodes are eliminated through
  `E` (their rows of `E` are empty and the lift `d` carries `g`), so the unknowns are the
  physical nodes that carry no `u => g`:

  ```julia
  Wₕ = gridspace(Ωₕ; ghosts = :boundary)
  Cₕ = constrained(Wₕ, u -> (
      :xmin => ∂ₙ(u) => g₁, :xmax => ∂ₙ(u) => g₂, :ymin => ∂ₙ(u) => g₃,
      :ymax => u => h,                         # Dirichlet nodes eliminated
      :ymax => ∂ₙₙ(u) => g₄))                  # the ghost relation of that facet
  a = form(Cₕ, Cₕ, (u, v) -> innerₕ(-Δₕ(u), v))
  l = form(Cₕ, v -> innerₕ(f̂ₕ, v))
  A, F = assemble(a, l)                         # symmetric by construction, N_phys − N_Dirichlet
  ```

  A facet that carries ghosts needs one ghost relation (`∂ₙ`, `∂ₙₙ` or a combination with
  `u`) and may carry one `u => g`; a Dirichlet facet that names no ghosts needs only
  `u => g`. Where two `u => g` meet at a corner node they must agree there, else an
  `ArgumentError`. The evidence for the closure with Dirichlet relations is the plain-matrix
  builder of `oracle_1d.jl` "Ghost closure" (Neumann, Robin and Dirichlet relations); S5's
  `constrained` took Neumann and Robin relations on every facet only, so (b) through forms is
  not prototyped.

  *Trade-off.*

  | | (a) row replacement | (b) `u => g` in `constrained` |
  |---|---|---|
  | unknowns | all `N_phys` physical nodes | `N_phys` minus the Dirichlet nodes |
  | solution vector | one entry per node, a grid function as today | one entry per kept node; the grid function is `E u + d` on the physical nodes |
  | symmetry of a self-adjoint form | after `symmetrize!`, as today | by construction (`Eᵀ A_full E`) |
  | data | markers, `dirichlet_constraints`, time-dependent forms as today | the relation's `g`, carried by the lift `d` |
  | existing programs | unchanged | new route, nothing existing depends on it |

  A label given both ways, `u => g` in `constrained` and the same label in
  `assemble(...; dirichlet)`, throws an `ArgumentError` at `assemble` (the data would be
  imposed twice, possibly with different values). Which route is the recommended default is
  open (see "Open questions").

Before (a Poisson problem, Mode A, no ghosts): see "Mode A with existing forms". For a form
that differentiates the unknown at the boundary no spelling exists: `Dcₕ(u)` is truncated
to zero on both end slices, which is the reason #218 exists.

After (relations on every facet, no Dirichlet):

```julia
Wₕ = gridspace(Ωₕ; ghosts = :boundary)
Cₕ = constrained(Wₕ, u -> (
    :xmin => ∂ₙ(u) => g₁, :xmax => α * u + β * ∂ₙ(u) => g₂,
    :ymin => ∂ₙ(u) => g₃, :ymax => ∂ₙ(u) => g₄))
a = form(Cₕ, Cₕ, (u, v) -> innerₕ(-Δₕ(u), v))
l = form(Cₕ, v -> innerₕ(f̂ₕ, v))
A, F = assemble(a, l)                  # Eᵀ A_full E, Eᵀ (F_full − A_full d)
u = pde_solve(A, F)                    # N_phys × N_phys here, the solver sees no ghost
```

The pair `assemble(a, l)` is needed because `d ≠ 0` when the data are non-zero: the load
is `Eᵀ(F_full − A_full d)`, the same shape as the lifting `assemble(a, l; dirichlet)`
already does. `ghosts` must name every facet that carries a ghost relation (S5 "Refusals").

Names this adds: `constrained`, `∂ₙ`, `∂ₙₙ` and the `=>` relation form (syntax). It removes
`neumann` and `robin` of the earlier draft of this note.

### Normal derivative: `∂ₙ(u)`

New exported name, defined inside `inner_Γ` and in a `constrained` relation (user decision,
2026-10-13): `∂ₙ(u) := dot(Dcₕ(u), η)`, one value per facet and two at a 2D corner;
`∂ₙ(u; K)` is the conormal form. Refused with an `ArgumentError` on a space without ghosts
on the named facets, where `Dcₕ` truncates. `∂ₙₙ(u)` (the normal second difference at a
facet node) follows the same rules and is exported with it.

Before: not expressible on an unknown, since `Dcₕ(u)` is zero on the end slices; known
data go through `inner_Γ(dot(F, η), v)`.

After:

```julia
Wₕ = gridspace(Ωₕ; ghosts = :boundary)
Cₕ = constrained(Wₕ, u -> (:xmin => ∂ₙ(u) => g₁, :xmax => ∂ₙ(u) => g₂,
    :ymin => ∂ₙ(u) => g₃, :ymax => ∂ₙ(u) => g₄))
a = form(Cₕ, Cₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + inner_Γ(κ * ∂ₙ(u), v; markers = (:xmax,)))
```

The boundary term on an unknown is a matrix entry only once the ghost is eliminated, so
`∂ₙ(u)` on a trial function needs the constrained space.

### Form-level `Δₕ`

User, 2026-10-14: "we will need to create a Delta_H operator that works in forms". Today
`Δₕ` exists only for `VectorElement`; S5 spells the Laplacian on trial and test functions
as `Σ_d D̃_d(D₋_d(u))` (`laplacian(u, D)` in `GhostEmbedProto.jl`), a `d`-term sum of
nested pairs in every Poisson-type form of the prototype.

- **Signature.** A method `Δₕ(u::LazyOp{D}) -> LazyOp{D}` of the existing exported
  `Δₕ`. No new name.
- **Before / after.**

  ```julia
  # before
  a = form(Wₕ, Wₕ, (u, v) -> innerₕ(-1.0 * D̃ₓ(D₋ₓ(u)) - 1.0 * D̃ᵧ(D₋ᵧ(u)), v))
  # after
  a = form(Wₕ, Wₕ, (u, v) -> innerₕ(-Δₕ(u), v))
  ```

- **Stencil.** One node whose axis `d` contribution has the three taps `(+1, 0, −1)` and
  weights `2/((h_i + h_{i+1}) h_{i+1})`, `−2/(h_i + h_{i+1}) (1/h_i + 1/h_{i+1})`,
  `2/((h_i + h_{i+1}) h_i)`; the form-level `Δₕ` is the sum over axes. It is `D̃_d D₋_d`
  exactly, so it must assemble to the same matrix as the spelled-out sum on non-uniform
  meshes (the prototype's matrices equal the oracles to 1.0e-16 in 2D).
- **At a facet node.** On a ghost space the node is an interior node of the padded mesh and
  the ordinary interior stencil reads the ghost: with `h_1 := h_2` the first row is
  `(u_2 − 2u_1 + u_0)/h_2²`, matching `oracle_1d.jl` "Boundary rows". At a corner it reads
  one ghost per incident facet and never a padded corner or edge point (`oracle_nd.jl` "Δ_H
  never reads absent points"). On a space without ghosts it truncates, as every operator does
  today.
- **Identity.** `innerₕ(Δₕ(u), v) = −inner₊(∇ₕ(u), ∇ₕ(v))` plus boundary terms: the
  summation by parts of `oracle_1d.jl` "Summation by parts",
  `(D̃u, v)_{H,*} = −(u, D₋v)_+ + Σ_{i∈{1,n}} η_i (M₋u)_{i+1} v_i`. This is the same
  identity the `VectorElement` method documents.
- **Nesting.** The node is a `LazyOp` and nests like any other: a difference, an average or
  `Δₕ` itself of `Δₕ(u)` re-evaluates the inner stencil at each tap, as `_reevaluated_shift`
  does today. An inner `Δₕ` evaluated at a ghost point would read a second ghost layer; that
  tap is absent and zeroed, the rule every nested operator already applies off the grid. What
  S5 verified: `D̃ₓ(D₋ₓ)` and a depth-4 composition on interior rows, and `jumpₓ(Δ_H)` on every
  row including the boundary (all at most 2.3e-16 from the oracle). Rows at a facet node of
  `Δₕ(Δₕ(u))` on a ghost space were not checked.
- **Why one node.** #774 measured the first-call cost of the spelled-out nesting: at depth 3
  in 3D, 3.78 s against 0.18 s for a depth-1 form (20.7×), the cost growing with depth and
  dimension (those figures were taken at a one-minute load of 3.0 to 3.2). A form-level node
  compiles one Laplacian stencil per side instead of a `d`-term sum.
- **Home.** The node family and its `_stencil_taps`/`_stencil_weights` in
  `src/operators/difference.jl`; the `Δₕ(::LazyOp)` method next to the `VectorElement` one in
  `src/operators/vector_calculus.jl`; a workload in `src/precompile/form_sessions.jl`.
  Acceptance criteria are in #774.

### The mean-zero constraint: `pde_solve(A, F; mean_zero = w)`

`w` is the vector of cell measures, assembled today as
`assemble(form(Wₕ, v -> innerₕ(x -> 1.0, v)))` (what `mode_a_existing_forms.jl` does). The
solve borders the system and returns the `N` solution values.

Before (`mode_a_existing_forms.jl` "Graded 2D solves through Bramble"):

```julia
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
l = form(Wₕ, v -> innerₕ(f̂ₕ, v) + inner_Γ(dot(F, η), v; markers = (:boundary,)))
A, F = assemble(a, l)                  # singular, and nothing says so
w = assemble(form(Wₕ, v -> innerₕ(x -> 1.0, v)))
u = (lu([A w; w' 0.0]) \ [F; 0.0])[1:end-1]
```

After:

```julia
A, F = assemble(a, l)
u = pde_solve(A, F; mean_zero = w)
```

On a ghost space `w` is the physical-only weight vector, so the same call serves.

### Changes to existing signatures or behaviour

Nothing here removes or renames an existing name. Six additions change what an existing call
does or accepts; each has its justification.

| change | justification |
|---|---|
| `gridspace(Ωₕ)` gains the keyword `ghosts` (default `()`) | the only way to ask for ghosts without a new constructor; user decision, 2026-10-13 |
| `form`, `assemble` accept a constrained space | the constrained space is the object of S5's `constrained`; it substitutes `E` at the taps. `assemble(...; dirichlet, symmetrize)` is unchanged and acts on the unknowns of the constrained space |
| `assemble(...; dirichlet = label)` throws an `ArgumentError` when the constrained space already holds `u => g` on that label | the data would be imposed twice |
| `assemble(l)` on a constrained space with `d ≠ 0` throws an `ArgumentError` directing to `assemble(a, l)` | `Eᵀ F_full` alone would silently drop the lift `−Eᵀ A_full d` (a wrong right-hand side that solves) |
| `pde_solve(A, F)` warns when `‖A 1‖∞ ≤ 1e-10 ‖A‖∞` and no `mean_zero` is given | today a pure Neumann matrix is solved silently and returns whichever particular solution the factorisation lands on; cost is one product. A warning, not an error, because a singular but consistent system is a legitimate input for some solvers |
| `pde_solve` gains the keyword `mean_zero` | the constraint of #205; applied as a bordered system |

Operators on a ghost space change behaviour (a ghost neighbour instead of truncation) only
on a space that asked for ghosts; no existing space is affected.

## Revised acceptance criteria

Proposed edits, for the maintainer to read before anything is posted. Each item names the
prototype that makes it testable. File paths follow the current tree (`src/operators/`,
`src/space/`), which the issue texts predate.

### #218: mesh/space: ghost meshes, ghost grid spaces and the centred normal derivative ∂ₙ

Replace the Design and Acceptance sections as follows.

```markdown
### Acceptance criteria

- [ ] `gridspace(Ωₕ; ghosts = :boundary | (:xmin, …))` returns a ghost space on a mesh padded
      by one point at each named facet end; only the named facets are padded. Counts: facet
      ghosts `Σ_d 2 Π_{e≠d} n_e`, no padded corner or edge point is a ghost, in 1D, 2D and
      3D on uniform and non-uniform meshes. Ghost coordinates are `x_1 − h_2` and
      `x_n + h_n` and follow `change_points!`.
- [ ] Boundary-to-ghost and ghost-to-boundary maps in O(1), in both directions, round-trip
      for every ghost.
- [ ] Quadrature weights are physical-only: 0 at every ghost and padded point; `markers` and
      `inner_Γ` name and weigh the physical facet nodes (the facet weight equals
      `_transverse_measure`, half cell at corners).
- [ ] No corner or edge degrees of freedom in a space the user sees. At the four 2D corner
      nodes the mixed difference is the inward one-sided difference; the anisotropic Robin
      problem (`K = [1 .5; .5 2]`) converges at second order on uniform, graded and random
      meshes.
- [ ] `D₋`, `D₊`, `M₋`, `M₊`, `Dc`, `∇ₕ`, `D̃`, `D̽` and nested compositions read the ghost
      neighbour at a ghosted facet node, and truncate as today at an unghosted one.
- [ ] `∂ₙ(u) = dot(Dcₕ(u), η)` inside `inner_Γ`: one value per facet, two at a 2D corner;
      `ArgumentError` on a space without ghosts on the named facet; second order on a smooth
      field on uniform and non-uniform meshes.
- [ ] A facet-only `GhostVector` keeps the solver's contiguous `Vector{T}` as the physical
      part; `update_ghosts!(v, closure)` with Neumann, Robin and affine closures allocates
      0 bytes and is `@inferred`, in 1D, 2D and 3D.
- [ ] Stencil evaluation over `GhostVector` against a contiguous array, with row-wise
      kernels (the SIMD loop over a row, the boundary handled once per row, no per-read
      bounds test, `@inbounds` reachable): within 5% of the contiguous kernel in 2D (256²,
      1024²) and 3D (64³), measured on a quiet machine with its spread.
- [ ] Decision recorded in `docs/src/internals/space.md`: padded mesh for assembly,
      facet-only `GhostVector` for grid functions, and why (copy of about 2× to 2.4× of a
      stencil application and an O(N²) stock `mul!` on the padded view).
- [ ] Operator application allocates 0 bytes (`@test_allocs`) and is `@inferred`.
- [ ] Unit tests in `test/space/ghost_gridspace.jl` (new) validate O(1) indexing and boundary
      facet access.
- [ ] `.claude/scripts/format.sh` and the quality group pass.
```

Files:

- `src/mesh/ghost_mesh.jl` (new), `src/mesh/marker.jl`, `src/mesh/queries.jl`.
- `src/space/ghost_gridspace.jl` (new), `src/space/ghost_vector.jl` (new),
  `src/space/inner_product.jl`, `src/space/vectorelement.jl`.
- `src/operators/difference.jl`, `src/operators/average.jl`, `src/operators/normal.jl`
  (`∂ₙ`, replacing the note at line 175).
- `src/api.jl`: `∂ₙ`.
- `test/mesh/`, `test/space/ghost_gridspace.jl` (new), `test/form/normal.jl`.
- `docs/src/internals/space.md`.

Removed from the issue: the evaluation of `AppendedArrays`, `PosNegReindex` and
`VectorsWithEntryInserted` (replaced by the layout decision and its measurement), and corner
ghosts as a possible extension (decided: none).

### #205: form: Neumann and Robin boundary terms, ghost elimination and the pure-Neumann constraint

```markdown
### Acceptance criteria

Forms and assembly:

- [ ] Mode A (ghost-eliminated Neumann and Robin for Poisson and Helmholtz) is today's
      forms: `inner_Γ(g, v; markers = …)`, `inner_Γ((α/β) * u, v; markers = …)` and
      `inner_Γ(dot(F, η), v)`. A test per dimension (1D, 2D, 3D, random non-uniform meshes)
      compares matrix and right-hand side with a plain-matrix ghost-eliminated reference to
      1e-12, exactly symmetric, with and without a mass term.
- [ ] `constrained(Wₕ, u -> (:facet => operator => data, …))` with `∂ₙ(u) => g` (Neumann),
      `α * u + β * ∂ₙ(u) => g` (Robin, `β ≠ 0`), `∂ₙ(u; K)` (conormal, constant `K`) and
      `∂ₙₙ(u) => g` builds a constrained ghost space; with ghost relations only the unknowns
      are the physical nodes of the space without ghosts (ghosts eliminated, no physical node
      removed), and with `u => g` relations (Dirichlet, route (b)) the Dirichlet nodes are
      eliminated too, the lift `d` carrying `g`. `assemble(a, l)` on it returns
      `Eᵀ A_full E` and `Eᵀ (F_full − A_full d)`, where `A_full` is the same form on the
      unconstrained ghost space; `assemble(l)` alone with `d ≠ 0` throws an `ArgumentError`.
- [ ] `inner_Γ(∂ₙ(u), v)` assembles on an unknown through the constrained space, zero runtime
      allocation during assembly (was #213).
- [ ] The 2×2 corner block of conormal data is solved at construction; the anisotropic
      problem of #218 converges at second order.
- [ ] The constrained matrix `Eᵀ A_full E` of a self-adjoint form is exactly symmetric, by
      construction with `u => g` relations (route (b)). Route (a), `assemble(...; dirichlet =
      labels)`, is unchanged: it replaces rows on regular and ghost spaces alike and the
      system is symmetric after `symmetrize!` (or `symmetrize = true`), as today. Tests with
      Dirichlet on some facets and Neumann or Robin relations on the others, by (a) and by (b),
      compare with the plain-matrix reference and agree with each other on the shared nodes.
- [ ] A label given both as `u => g` in `constrained` and in `assemble(...; dirichlet)` throws
      an `ArgumentError`; two `u => g` meeting at a corner with different values throw an
      `ArgumentError`.
- [ ] 1D, 2D and 3D Poisson with pure Neumann and mixed Robin, and a Helmholtz case, converge
      at O(h²) on an MMS, uniform and non-uniform meshes (2D Robin 2.067 and Neumann 2.077
      on graded meshes in the prototypes).
- [ ] Zero allocations in ghost buffer updates and operator application along the solve path.
- [ ] No public Mode B: the augmented matrix exists only as the unconstrained ghost
      assembly the constrained space is built from.

Pure Neumann (was #272):

- [ ] `pde_solve(A, F; mean_zero = w)` borders the system with the cell measures and returns
      the solution of mean zero; second order on an MMS against the exact solution shifted to
      mean zero. The constraint is not stored as a rank-one term in the matrix (it is dense:
      `nnz = N²`, `lu` 1.415 s against 7.7 ms bordered at N = 4225).
- [ ] A matrix that annihilates constants (`‖A 1‖∞ ≤ 1e-10 ‖A‖∞`) and carries no `mean_zero`
      is reported by a warning from `pde_solve`.
- [ ] Data compatibility `innerₕ(f, 1) + inner_Γ(g, 1) ≈ 0` is checked against a data-scaled
      reference (the defect on a refined mesh, or the quadrature error of the data), and a
      violation is a warning that reports the defect; there is no fixed threshold on a
      single mesh.

Nested boundary forms (was #345):

- [ ] The list of nested forms, dimensions and sides this issue needs: relation rows
      `Dc(u)` along one axis and, for conormal data, along every axis (depth 1, trial
      side); `Δₕ(u)` and `Δₕ(v)` (#774); the 2D mixed `Dcₓ(Dcᵧ(u))` and `D̽ₓ(D̽ᵧ(u))` (depth 2,
      trial side, 2D); `∂ₙ(u)` inside `inner_Γ` (depth 1, trial side).
- [ ] Fresh-process first-call time for each listed form, before and after, measured on a
      quiet machine (`bramble-verification`); the first-call table of #774 for the Laplacian.
- [ ] Package-image build time and invalidation-gate result, before and after.
- [ ] If the inlining option is taken: `assemble!` runtime for the listed forms, before and
      after, and still 0 bytes.
- [ ] `test/form/nested_operators.jl` (pairwise composition against the runtime operators)
      and `test/form/compile_scaling.jl` (first-call time flat in depth) still pass, and the
      listed forms are added to them.

Quality:

- [ ] `.claude/scripts/format.sh` and the quality group (Aqua, JET) pass.
```

Files:

- `src/operators/normal.jl` (`∂ₙ`, `∂ₙₙ`), `src/space/constrained_gridspace.jl` (new;
  with the relation parsing), `src/assembly/ghost_relations.jl` (new),
  `src/assembly/` (substitution of `E` at the taps), `src/problems/`.
- `src/solvers/pde_solve.jl` (`mean_zero`, the singularity warning).
- `src/precompile/form_sessions.jl`.
- `src/api.jl`: `constrained`, `∂ₙ`, `∂ₙₙ`.
- `test/form/` (boundary forms, `nested_operators.jl`), MMS tests, a Mode A test per
  dimension.
- `docs/src/tutorials/` or `docs/src/examples/`: a Neumann and a Robin example.

Removed from the issue: the form terms `neumann(marker, g_N, v)` and
`robin(marker, α, β, u, v)` (the same matrices come from `inner_Γ`; relations on a ghost
space are written `operator => data`), Mode B, and "the constraint is a rank-one
modification".

## Open questions

1. **3D edges.** The edge lines carry padded points (60 on a 4×5×6 mesh, plus 8 corners).
   Mixed differences at edge nodes read them, as the 2D corner does. The one-sided rule
   generalises, but only the 2D corner problem was built; a 3D conormal problem with a full
   tensor was not.
2. **Elasticity tractions.** A natural condition for a symmetric gradient or a Stokes
   stress needs the mixed differences `D_e u_d` with `e ≠ d` at facet nodes and vector-valued
   ghosts per component. The corner problem here is scalar. Not prototyped.
3. **Corner rule as a tap rule or as `E` rows.** S5 ran only the padded closures through
   forms; the facet-only one-sided closure was run only in the plain-matrix oracle. A tap
   rule in the stencil engine ("a mixed operator that would read an absent point uses the
   inward difference at that node") keeps `E` purely a facet matter; the alternative stores
   no corner but writes corner rows of `E` that reproduce the one-sided stencil (the
   `padded_mixed` closure is the prototype of this, but its conditioning was worse, 7.5e5
   against 1.1e4). Needs a prototype in the stencil engine before the plan fixes it.
4. **Which Dirichlet route is the default.** Both are offered (user, 2026-10-15). Route (a)
   keeps every existing program and the vector sizes, and needs `symmetrize!` for symmetry;
   route (b) gives fewer unknowns and a symmetric matrix by construction but a solution vector
   shorter than the grid, whose expansion to a grid function (`E u + d` on the physical nodes)
   needs a named helper. Which one the tutorials recommend, and whether the other is kept,
   is for the maintainer. Neither was run on a ghost space: S5 tested relations on every
   facet, and Dirichlet relations only in the plain-matrix closure.
5. **Variable tensor and `∂ₙₙ` through forms.** `∂ₙ(u; K)` takes a constant matrix, as in S2.2
   and S5; a grid-function `K` is untested. `∂ₙₙ` as a relation was not run through forms.
6. **Warning or error for a singular matrix without `mean_zero`.** A warning is proposed;
   the maintainer may prefer an error with an escape keyword.
7. **A helper for the weights.** `w` is assembled with a one-line form today; a named
   helper (a `weights` method on a space) would spare the spelling but adds a name.
8. **Iterative solvers and the bordered system.** The bordered matrix is symmetric
   indefinite, which a Cholesky-based or conjugate-gradient `pde_solve` method does not
   accept. S3.2 compared the representations with `lu` and a hand-written CG; the bordered
   path with `pde_solve`'s other solvers is untested. Pinning is the positive definite
   fallback.
9. **Time-dependent conditions.** `semidiscretize` with time-dependent `g` on a constrained
   space: `d` depends on `t`. Not prototyped.
10. **Device support.** Out of scope here (v4.4.0): device storage for `GhostVector`, the
    ghost-aware stencils and `update_ghosts!` kernels belong to the GPU issue.
11. **Nested `Δₕ` rows at facet nodes.** `Δₕ(Δₕ(u))` rows at a ghosted facet node read a second
    ghost layer that does not exist; zeroing it is the existing off-grid rule, but its
    effect on those rows was not checked (see "Proposed API").
