# Ghost points for Neumann and Robin conditions: structures, API and evidence

Design history for milestone v4.0.0 (gpena/Bramble.jl#218, #205; #774 for the form-level
`Δₕ`). Nothing here is published, and no issue has been edited: the acceptance-criteria
edits below are proposals for the maintainer to read first.

## Status: what the implementation decided differently

The design below is kept as written; the implementation (`src/mesh/ghost_mesh.jl`,
`src/space/ghost_gridspace.jl`, `src/assembly/ghost_assembly.jl`,
`src/assembly/ghost_relations.jl`) follows it except in the places below. The published account
of what was built is `docs/src/internals/mesh.md` and `docs/src/internals/space.md`.

- **Forms use padded `VectorElement`s, time loops use the facet-only `GhostVector`.**
  Decision 4 and the layout section chose the facet-only layout. The implementation assembles
  on the padded mesh anyway, so `lift(Cₕ, u)` returns the padded grid function for forms, and
  `GhostVector` with `update_ghosts!` and the ghost-aware runtime operators serves time loops.
  Runtime operators on a `GhostVector` return the physical values as a plain `Vector`, so
  mixed differences at ghost corners stay form-only: nested mixed differences on a ghost
  space are assembled as forms. Time-dependent relation data are supported through
  `d(t) = L g(t)`, `update_ghosts!(v, Cₕ, t)` and `semidiscretize`. Still refused with an
  `ArgumentError`: a `semidiscretize` call with the `dirichlet` keyword, on a space keeping
  ghosts as unknowns, or with a mass that reads a ghost whose data depend on time,
  `semidiscretize(build, l)` and `semidiscretize_rhs`.
- **The reduction replaces the per-tap substitution, and one space type replaces two.** The
  proposal substituted the closure at each stencil tap and kept a `ConstrainedGridSpace`.
  The implementation assembles on the padded space and reduces the result once per form,
  `Eᵀ A_pad E`, with the load `Eᵀ(F_pad − A_pad d)`, which leaves the guarded walk, the
  pattern recorder and the replay untouched. `GhostGridSpace` carries the closure `(E, d)`
  for both `gridspace(Ωₕ; ghosts)` and `constrained`.
- **The compatibility diagnostic is a function of a problem builder.** No single-mesh
  threshold separates a quadrature defect of order `h²` from an incompatibility, so
  `Bramble.neumann_compatibility(problem, Ωₕ)` builds the problem on `Ωₕ` and on copies
  refined once and twice, and logs a warning for incompatible data; `pde_solve` itself, which
  sees one mesh, does not check. `Bramble.ghost_kernel` and `pde_solve(A, F; gauge)` fix the
  ghost-only kernel that a form reading a corner's ghosts only through `Δₕ` leaves.
- **`lift` is `public`, not exported.** Makie exports a `lift` of its own, so an export would
  make both unusable after `using Bramble, GLMakie`: write `Bramble.lift`.
- **Out of scope, refused with an `ArgumentError`:** device backends, automatic
  differentiation through ghost spaces, a grid-function `K` in `∂ₙ(u; K)`, and elasticity
  tractions. Matrix-free and Kronecker operators (where separable) and composite spaces of
  ghost spaces now work within limits; the published table `Support on ghost spaces` in
  `docs/src/internals/space.md` has each operation's status.

The evidence is a set of throwaway prototypes. The three plain-matrix oracles and their
tests are merged and run in the test tree under `test/ghost/`. The others stay on the local
branch `proto/v4-ghost-points`, under `test/prototypes/ghost/`. At the time of writing
nothing under `src/`, `ext/` or `docs/src/` had changed. Every claim below cites a prototype test file and testset,
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

## Ghost-only kernels

gpena/Bramble.jl#777 item 1 asks first whether a ghost extension without the ghost-only
kernel exists. Answer: none exists that keeps the form, keeps every ghost of a ghosted facet
an unknown and leaves the physical solution unchanged. The kernel belongs to the form, not
to the ghost storage. A relation tying the ghosts of each multi-facet node removes it
exactly, but that relation is a gauge: it selects one representative of a set of ghost
values that no physical value sees. Removing it through the form instead changes the
physical solution. S7.2 and S7.3 therefore compute the kernel locally and fix it by
bordering.

### Which forms produce it

Unknowns are the physical nodes, then the facet ghosts. At a node on `m ≥ 2` ghosted facets,
`Δₕ` reads one ghost per facet, with weight `c_d = 2/((h_1 + h_2) h_2)` on axis `d`
(`1/h_2²` with the mirrored spacing). A form whose terms read those ghosts only through
`Δₕ` at that node sees them only through `Σ_d c_d z_d`. The `m` ghost columns of the
assembled matrix are then parallel (`A e_{g_d} = c_d · (column of Δₕ at the node)`), and
`m − 1` combinations of those ghosts lie in the kernel. That gives 4 in 2D (one per corner)
and `4 Σ_e (n_e − 2) + 16` in 3D: `m = 2` at the `n_e − 2` interior nodes of the four edge
lines along each axis `e`, `m = 3` at the 8 corners. 1D has no such node.

Script `/tmp/s71/ghost_kernels.jl` (listed at the end of this section). The run used one
thread, `julia --project=. --startup-file=no`, Julia 1.13.2, on the branch
`v4.0.0/ghost-dynamics` at 9fc39556. The local kernel of a node is `m` minus the rank of its
`m` ghost columns, normalised, at a 1e-12 relative singular-value threshold. Each node's
ghosts come from `inner_Γ(∂ₙ(u), v)`, whose ghost columns have one entry each, in the row of
the facet node.

```
== A. Forms on gridspace(Ωₕ; ghosts = :boundary), 2D 6×7 random (seed 11)
  42 physical nodes, 26 ghosts, max h ratio 50; ghosts of the multi-facet nodes [[43, 57], [49, 63], [50, 62], [56, 68]]
  innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)
      local kernel 4 (σ_min/σ_max: largest counted 1e-16, smallest not counted none); zero ghost columns 0, rows 0
  innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v) + inner_Γ(∂ₙ(u), ∂ₙ(v))
      local kernel 0 (σ_min/σ_max: largest counted 0e+00, smallest not counted 4e-03); zero ghost columns 0, rows 0
  innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v) + inner_Γ(∂ₙ(u), v)
      local kernel 4 (σ_min/σ_max: largest counted 2e-16, smallest not counted none); zero ghost columns 0, rows 0
  innerₕ(Δₕ(u), v) + innerₕ(u, v)
      local kernel 4 (σ_min/σ_max: largest counted 0e+00, smallest not counted none); zero ghost columns 0, rows 26
  innerₕ(u, Δₕ(v)) + innerₕ(u, v)
      local kernel 8 (σ_min/σ_max: largest counted 0e+00, smallest not counted none); zero ghost columns 26, rows 0
```

- **`innerₕ(Δₕ(u), Δₕ(v))` (+ mass) is the form with this kernel.** `Δₕ` on both sides gives
  every ghost a row and a column, and the corner columns are parallel to rounding (1e-16).
  The system is square, symmetric and singular only by the ghost-only kernel.
- **A trial-side boundary flux keeps the kernel.** With `inner_Γ(∂ₙ(u), v)` added, the
  columns stay parallel (2e-16). At a facet node, the ghost tap of `∂ₙ` times the facet
  weight is proportional to the ghost tap of `Δₕ` times the cell weight, with the same factor
  on every facet: this is the summation by parts of "Form-level `Δₕ`". So the combination
  the forms see is the node's discrete outward flux, summed over its facets.
- **A test-side normal derivative per facet removes it.** `inner_Γ(∂ₙ(u), ∂ₙ(v))` gives each
  ghost its own row (its facet's `∂ₙ(v)`), and the columns separate (`σ_min/σ_max ≥ 4e-3`).
- **`Δₕ` on one side only does not give a ghost-only kernel; it gives a singular system that
  needs a relation per ghost.** With `Δₕ(u)` against a plain `v`, every ghost row is zero
  (26 of 26). With `u` against `Δₕ(v)`, every ghost column is zero, so all 26 ghosts are in
  the kernel; the "local kernel 8" counts those zero columns at the four corners. These are
  the Poisson-type forms. They are meant for `constrained`, whose relations remove every
  ghost.

A facet relation eliminates its own facet's ghost, `z_F = a · (the node's other ghosts) +
(physical values)`. An axis-aligned relation (`α u + ∂ₙ(u)`, `u`, `∂ₙₙ`) reads one ghost, so
`a = 0`. A conormal `∂ₙ(u; K)` also reads the ghost of the other facet at a corner (the
off-diagonal `K` term), so `a ≠ 0`. In both cases, substituting into `Σ_d c_d z_d` leaves the
node's remaining ghosts inside one combination again. A relation therefore never enlarges
the count: a node keeps `r − 1` kernel vectors when `r ≥ 1` of its ghosts stay unknowns, and
none when `r ≤ 1`. The exception is below. Part B (nullity is the count of singular values
below `1e-14 σ₁`):

```
== B. Relations or fewer ghosted facets (bilap, same mesh)
  constrained, ∂ₙ(u) => 0 on (:xmin,)                     (61, 61), nullity 2
  constrained, ∂ₙ(u) => 0 on (:xmin, :ymin)               (55, 55), nullity 1
  constrained, ∂ₙ(u) => 0 on (:xmin, :xmax)               (54, 54), nullity 0
  constrained, ∂ₙ(u) => 0 on (:xmin, :xmax, :ymin, :ymax) (42, 42), nullity 0
  gridspace(Ωₕ; ghosts = (:xmin,))       (49, 49), nullity 0
  gridspace(Ωₕ; ghosts = (:xmin, :ymin)) (55, 55), nullity 1
```

The count follows the corners whose two facets both keep ghosts as unknowns: two corners
for `(:xmin,)` related (`:xmax` meets `:ymin` and `:ymax`), one for `(:xmin, :ymin)`
related, none for two opposite facets. A space ghosting only adjacent facets has the kernel
too, with no relation at all.

The exception: a conormal relation can cancel the remaining ghost's weight. At an `:xmin`
corner, after the `x`-ghost is eliminated, the `y`-ghost enters with weight `c_y + a c_x`,
and `a` is proportional to `k₁₂/k₁₁`. The weight vanishes at `k₁₂/k₁₁ = −η_y h_x/h_y`
(`h_x` the `x` spacing at `:xmin`, `h_y` the `y` spacing at the corner's `y` facet, `η_y`
that facet's outward sign). No row reads that ghost then. Its column is zero, and the
node's count rises from 0 to 1.

Script `/tmp/s71/conormal.jl`, same mesh, with `:xmin => ∂ₙ(u; K) => 0.0` as the only
relation (nullity 2 at `t = 0`, as for `(:xmin,)` in part B). It finds the root from the
reduced matrix, then checks it with a symmetric positive definite `K` of the same ratio:

```
reduced ghost column 50: t* = 22.302130 (from 2 affine rows, spread 3.2e-14); ‖column(t*)‖/‖column(0)‖ 4.6e-16, local kernel 1, nullity 3; t = 0: nullity 2
reduced ghost column 56: t* = -3.620066 (from 2 affine rows, spread 2.2e-15); ‖column(t*)‖/‖column(0)‖ 9.8e-18, local kernel 1, nullity 3; t = 0: nullity 2
h_x/h_y at (xmin, ymin) 22.302130, at (xmin, ymax) 3.620066
SPD K = [1 22.3021; 22.3021 498.3850] (eigvals [0.002, 499.0]): nullity 3
SPD K = [1 -3.6201; -3.6201 14.1049] (eigvals [0.0665, 15.0]): nullity 3
```

Away from those ratios a conormal relation gives the same count as an axis-aligned one.
The S7.1 critic's sweep of `K = [1 t; t 1]` (`/tmp/s71-critic/conormal.jl`) gave nullity 2
for `t` in ±0.5, ±1, ±2, ±5, ±20, as at `t = 0`. The kernel S7.2 has to find is therefore
the ghost columns of a node that are parallel or zero, and a local test sees both cases.

### The dimension in 3D, and why a global spectrum miscounts it

```
== C. 3D 4×5×6 random: local count against 4 Σ_e (n_e − 2) + 16 = 52, and the global spectrum
  seed 7 (h ratio     14): 36 nodes with m=2, 8 with m=3, local kernel 52; count(s < 1e-14 σ₁)  52, next s/σ₁ 9.6e-11, rank() nullity  52
  seed 1 (h ratio     46): 36 nodes with m=2, 8 with m=3, local kernel 52; count(s < 1e-14 σ₁)  52, next s/σ₁ 4.5e-14, rank() nullity  53
  seed 2 (h ratio    301): 36 nodes with m=2, 8 with m=3, local kernel 52; count(s < 1e-14 σ₁)  70, next s/σ₁ 3.1e-18, rank() nullity  70
  seed 3 (h ratio  29560): 36 nodes with m=2, 8 with m=3, local kernel 52; count(s < 1e-14 σ₁) 148, next s/σ₁ 1.8e-26, rank() nullity 148
  seed 4 (h ratio     13): 36 nodes with m=2, 8 with m=3, local kernel 52; count(s < 1e-14 σ₁)  52, next s/σ₁ 1.2e-11, rank() nullity  52
  seed 5 (h ratio     42): 36 nodes with m=2, 8 with m=3, local kernel 52; count(s < 1e-14 σ₁)  53, next s/σ₁ 2.8e-15, rank() nullity  57
```

The local count is 52 = `4 Σ_e (n_e − 2) + 16` on every mesh (36 edge nodes with `m = 2`,
8 corners with `m = 3`). A count of the global singular values is not. The `Δₕ`-squared
form scales as `h⁻⁴`, so on random meshes its nonzero singular values reach far below
`σ₁`. The planning measurement (notes §4, 4×5×6, seed 7) found the 53rd at
`3.4e-11 σ₁`, and `rank`'s default tolerance `n eps σ₁` gave nullities from 53 to 147 over
the meshes tried, against 52. This rerun, on `[0, 1] × [0, 2] × [0, 1.5]`, has the 53rd at
`9.6e-11 σ₁` for seed 7. What decides a count is where the smallest nonzero singular value
falls against the threshold. `rank`'s default is `n eps σ₁ ≈ 6e-14 σ₁` for `n = 268`:

- Seeds 7 and 4 have it at `9.6e-11` and `1.2e-11 σ₁`. Both counts are right.
- Seed 1 has it at `4.5e-14 σ₁`, between the two thresholds. `1e-14 σ₁` counts 52, `rank` 53.
- Seed 5 has it at `2.8e-15 σ₁`, under both: 53 and 57.
- Seeds 2 and 3 have it at `3.1e-18` and `1.8e-26 σ₁`: 70 and 148 by both.

The mesh's spacing ratio does not order these: seeds 1 and 5 (ratios 46 and 42) miscount,
and seeds 4 and 7 (13 and 14) do not. Only the two largest ratios (301, 29560) go with the
worst counts. No fixed relative threshold is safe. Counting per node, from that node's ghost
columns only, reads the parallelism directly and needs no global threshold: S7.2's
`ghost_kernel`.

The same spread bounds a gauged solve. On the 2D 6×7 mesh the nonzero part has
`κ = σ₁/σ_{N−4} = 1.9e12` (part D below), so the physical part of a solve against a
manufactured `x` is accurate only to about 1e-9 (1.1e-9 here, 9.3e-10 in the planning run),
and `pinv(rtol = 1e-14)` only to 1.2e-7 (4.9e-8 in planning). #777's flat 1e-10 is below
rounding on these forms (plan departure 4).

### Extensions tried, and what each costs

```
== D. Extensions on 2D 6×7 (seed 11), F = A x for a random x
  κ of the nonzero part σ₁/σ_{N−4} = 1.9e+12
  pinv(rtol = 1e-14): physical vs x 1.2e-07
  D1 keep the x-ghost at every corner: nullity 1, κ 1.0e+14, physical vs x 1.1e-09, ‖A u − F‖/‖F‖ 4.7e-16
  D1' keep the y-ghost instead: nullity 0, κ 1.9e+12, physical vs x 1.1e-09
  D2 relation z_x = z_y per corner: nullity 0, κ 1.9e+12, physical vs x 1.1e-09, ‖A u − F‖/‖F‖ 3.7e-16
  D1 against D2: ghosts differ by 9.7e-01, physical parts by 1.1e-10
  D3 + inner_Γ(∂ₙ(u), ∂ₙ(v)): nullity 0, κ 4.0e+07, physical vs x 3.5e+00
```

1. **More storage (padded corner and edge points).** No row reads them: `Δₕ` reads one ghost
   per incident facet and never a corner or edge point (`oracle_nd.jl` "Δ_H never reads
   absent points"). Each added unknown is a zero column, so padding enlarges the kernel.
2. **Less storage: one ghost per multi-facet node (D1).** D1 keeps the `x`-ghost at every
   corner and drops the `y`-ghost, the relation `z_y = 0`; D1' keeps the `y`-ghost instead.
   Both are structurally nonsingular, and the physical part matches `x` (1.1e-9). Conditioning
   depends on which axis is kept: D1 has `κ` 1.0e14, enough that the global count reads
   nullity 1, while D1' keeps the nonzero part's 1.9e12. A rule that keeps a good ghost needs
   the mesh's weights at each node. The ghost values have no meaning (D1's are 0.97 from
   D2's), and the choice breaks the symmetry between axes.
3. **A relation tying the ghosts of each multi-facet node (D2).** `z_x = z_y` at each 2D
   corner, `z_x = z_y = z_z` at a 3D corner, and pairs along edges. It is transversal to the
   kernel whenever the weights `c_d` are positive (a kernel vector has `Σ_d c_d z_d = 0`, so
   it cannot have all its ghosts equal), so it removes the kernel exactly. The reduced matrix
   keeps the nonzero part's `κ` (1.9e12), and the physical part equals D1's to 1.1e-10.
   Cost: a new relation kind in `constrained`, one that reads two or three facets' ghosts
   (the corner-block problem of "Corner policy" item 3, extended to the 3D edge lines). It
   applies to a mode no physical value sees. Its ghost values are a choice, as D1's are, and
   are still not extrapolations of `u`. S7.3's bordered gauge `Zᵀu = 0` gives the same
   physical part without a new relation kind, and its ghost values are the minimum-norm
   ones.
4. **A change of form (D3).** Adding `inner_Γ(∂ₙ(u), ∂ₙ(v))` removes the kernel and lowers
   `κ` to 4.0e7. It is a different problem: the physical part moves by 3.5 (relative) from
   the original form's. It is admissible only where the boundary term is part of the PDE's
   weak form, for example a least-squares Neumann penalty. The trial-side
   `inner_Γ(∂ₙ(u), v)` does not remove the kernel (part A).

So the kernel is removable only by a gauge (items 2, 3, and S7.3's border) or by changing
the problem (item 4). Of the gauges, the border needs no new relation kind and does not
depend on an ordering of the ghosts.

The main script:

```julia
# S7.1 evidence: ghost-only kernels of forms on ghost spaces (gpena/Bramble.jl#777).
# Run: julia --project=. --startup-file=no /tmp/s71/ghost_kernels.jl
using Bramble, LinearAlgebra, SparseArrays, Random, Printf

function rmesh(dims, seed)
    Bramble._seed_mesh1d_rng!(seed)
    I, J, K = interval(0.0, 1.0), interval(0.0, 2.0), interval(0.0, 1.5)
    Ω = length(dims) == 2 ? mesh(domain(I × J), dims, (false, false)) :
        mesh(domain(I × J × K), dims, (false, false, false))
    Bramble._unseed_mesh1d_rng!()
    return Ω
end
hratio(Ω) = maximum(p -> maximum(diff(p)) / minimum(diff(p)), Bramble.points(Ω))
gnullity(A; r = 1e-14) = (s = svdvals(Matrix(A)); count(<(r * s[1]), s))
bilap(u, v) = innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)
asm(W, f) = sparse(assemble(form(W, W, f)))

# Each ghost column of inner_Γ(∂ₙ(u), v) has one entry, in the row of the ghost's facet node:
# that maps every ghost to its node. Nodes with m ≥ 2 ghosts are the multi-facet nodes.
function multifacet(W, np)
    B = asm(W, (u, v) -> inner_Γ(∂ₙ(u), v))
    node = Dict{Int, Vector{Int}}()
    for j in (np + 1):size(B, 2)
        r = findall(!iszero, B[:, j]); length(r) == 1 || error("ghost $j reads $(r)")
        push!(get!(node, r[1], Int[]), j)
    end
    return sort([g for g in values(node) if length(g) ≥ 2]; by = first)
end
# Local kernel count: m − rank of the node's m ghost columns, normalised, at 1e-12.
function local_kernel(A, groups)
    k, worst, best = 0, 0.0, Inf
    for g in groups
        C = Matrix(A[:, g]); nz = vec(sum(abs2, C; dims = 1)) .> 0
        k += count(!, nz); count(nz) == 0 && continue
        C = C[:, nz] ./ sqrt.(sum(abs2, C[:, nz]; dims = 1))
        s = svdvals(C); d = count(<(1e-12 * s[1]), s); k += d
        d > 0 ? (worst = max(worst, s[end] / s[1])) : (best = min(best, s[end] / s[1]))
    end
    return k, worst, best
end

println("== A. Forms on gridspace(Ωₕ; ghosts = :boundary), 2D 6×7 random (seed 11)")
Ω = rmesh((6, 7), 11); W = gridspace(Ω; ghosts = :boundary); np = npoints(Ω)
G = multifacet(W, np)
@printf("  %d physical nodes, %d ghosts, max h ratio %.0f; ghosts of the multi-facet nodes %s\n",
        np, size(asm(W, bilap), 1) - np, hratio(Ω), string(G))
forms = [
    "innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)" => bilap,
    "innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v) + inner_Γ(∂ₙ(u), ∂ₙ(v))" =>
        (u, v) -> bilap(u, v) + inner_Γ(∂ₙ(u), ∂ₙ(v)),
    "innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v) + inner_Γ(∂ₙ(u), v)" =>
        (u, v) -> bilap(u, v) + inner_Γ(∂ₙ(u), v),
    "innerₕ(Δₕ(u), v) + innerₕ(u, v)" => (u, v) -> innerₕ(Δₕ(u), v) + innerₕ(u, v),
    "innerₕ(u, Δₕ(v)) + innerₕ(u, v)" => (u, v) -> innerₕ(u, Δₕ(v)) + innerₕ(u, v),
]
for (name, f) in forms
    A = try asm(W, f) catch e; println("  $name: ", typeof(e)); continue end
    k, worst, best = local_kernel(A, G)
    zc = count(j -> iszero(A[:, j]), (np + 1):size(A, 2))
    zr = count(i -> iszero(A[i, :]), (np + 1):size(A, 1))
    @printf("  %-60s\n      local kernel %d (σ_min/σ_max: largest counted %.0e, smallest not counted %s); zero ghost columns %d, rows %d\n",
            name, k, worst, isinf(best) ? "none" : @sprintf("%.0e", best), zc, zr)
end

println("== B. Relations or fewer ghosted facets (bilap, same mesh)")
for fs in ((:xmin,), (:xmin, :ymin), (:xmin, :xmax), (:xmin, :xmax, :ymin, :ymax))
    C = constrained(W, u -> map(F -> F => ∂ₙ(u) => 0.0, fs))
    @printf("  constrained, ∂ₙ(u) => 0 on %-28s %s, nullity %d\n", string(fs),
            size(asm(C, bilap)), gnullity(asm(C, bilap)))
end
for gs in ((:xmin,), (:xmin, :ymin))
    Wg = gridspace(Ω; ghosts = gs)
    @printf("  gridspace(Ωₕ; ghosts = %-15s %s, nullity %d\n", string(gs) * ")",
            size(asm(Wg, bilap)), gnullity(asm(Wg, bilap)))
end

println("== C. 3D 4×5×6 random: local count against 4 Σ_e (n_e − 2) + 16 = ",
        4 * sum((4, 5, 6) .- 2) + 16, ", and the global spectrum")
for seed in (7, 1, 2, 3, 4, 5)
    Ω3 = rmesh((4, 5, 6), seed); W3 = gridspace(Ω3; ghosts = :boundary)
    A3 = asm(W3, bilap); G3 = multifacet(W3, npoints(Ω3)); M3 = Matrix(A3)
    s = svdvals(M3); k = local_kernel(A3, G3)[1]
    @printf("  seed %d (h ratio %6.0f): %d nodes with m=2, %d with m=3, local kernel %d; count(s < 1e-14 σ₁) %3d, next s/σ₁ %.1e, rank() nullity %3d\n",
            seed, hratio(Ω3), count(g -> length(g) == 2, G3), count(g -> length(g) == 3, G3),
            k, count(<(1e-14 * s[1]), s), s[end - 52] / s[1], size(M3, 1) - rank(M3))
end

println("== D. Extensions on 2D 6×7 (seed 11), F = A x for a random x")
A = asm(W, bilap); M = Matrix(A); N = size(M, 1)
Random.seed!(777); x = rand(N); Fx = M * x
relerr(u) = norm(u[1:np] - x[1:np], Inf) / norm(x[1:np], Inf)
s = svdvals(M); @printf("  κ of the nonzero part σ₁/σ_{N−4} = %.1e\n", s[1] / s[end - 4])
@printf("  pinv(rtol = 1e-14): physical vs x %.1e\n", relerr(pinv(M; rtol = 1e-14) * Fx))
drop = [g[2] for g in G]; keep = setdiff(1:N, drop)
# D1: keep the x-ghost at every corner, drop the y-ghost (the relation z_y = 0)
u1 = zeros(N); A1 = M[keep, keep]; u1[keep] = A1 \ Fx[keep]
@printf("  D1 keep the x-ghost at every corner: nullity %d, κ %.1e, physical vs x %.1e, ‖A u − F‖/‖F‖ %.1e\n",
        gnullity(A1), cond(A1), relerr(u1), norm(M * u1 - Fx) / norm(Fx))
# D1': the same with the other ghost kept at every corner (no x-ghost at the corners)
keepy = setdiff(1:N, [g[1] for g in G]); A1y = M[keepy, keepy]
u1y = zeros(N); u1y[keepy] = A1y \ Fx[keepy]
@printf("  D1' keep the y-ghost instead: nullity %d, κ %.1e, physical vs x %.1e\n",
        gnullity(A1y), cond(A1y), relerr(u1y))
# D2: a relation tying the corner ghosts, z_x − z_y = 0
E = Matrix(1.0I, N, N)[:, keep]
for g in G
    E[g[2], findfirst(==(g[1]), keep)] = 1.0
end
A2 = E' * M * E; u2 = E * (A2 \ (E' * Fx))
@printf("  D2 relation z_x = z_y per corner: nullity %d, κ %.1e, physical vs x %.1e, ‖A u − F‖/‖F‖ %.1e\n",
        gnullity(A2), cond(A2), relerr(u2), norm(M * u2 - Fx) / norm(Fx))
@printf("  D1 against D2: ghosts differ by %.1e, physical parts by %.1e\n",
        norm(u1[drop] - u2[drop], Inf), norm(u1[1:np] - u2[1:np], Inf) / norm(u2[1:np], Inf))
# D3: the boundary term inner_Γ(∂ₙ(u), ∂ₙ(v)) added: a different problem
A3 = Matrix(asm(W, (u, v) -> bilap(u, v) + inner_Γ(∂ₙ(u), ∂ₙ(v)))); u3 = A3 \ Fx
@printf("  D3 + inner_Γ(∂ₙ(u), ∂ₙ(v)): nullity %d, κ %.1e, physical vs x %.1e\n",
        gnullity(A3), cond(A3), relerr(u3))
```

The conormal script:

```julia
# Conormal relation on :xmin only, K = [1 t; t 1]: when does the free y-ghost of an
# xmin corner leave the reduced form? Its reduced column is (c_y + a t c_x) times a vector.
using Statistics; include_string(Main, replace(read("/tmp/s71/ghost_kernels.jl", String), r"println\(\"== A\..*"s => ""))
Ω = rmesh((6, 7), 11); W = gridspace(Ω; ghosts = :boundary); np = npoints(Ω)
red(t) = Matrix(asm(constrained(W, u -> (:xmin => ∂ₙ(u; K = [1.0 t; t 1.0]) => 0.0,)), bilap))
R = [red(t) for t in (0.0, 1.0, 2.0)]; n = size(R[1], 2)
for j in (np + 1):n
    R[2][:, j] ≈ R[1][:, j] && continue
    roots = Float64[]
    for i in 1:n
        v0, v1, v2 = R[1][i, j], R[2][i, j], R[3][i, j]
        s = v1 - v0
        abs(s) > 1e-8 * abs(v0) && abs(v2 - 2v1 + v0) < 1e-10 * max(abs(v0), abs(v2)) &&
            push!(roots, -v0 / s)
    end
    ts = median(sort(roots)); A = red(ts)
    @printf("reduced ghost column %d: t* = %.6f (from %d affine rows, spread %.1e); ‖column(t*)‖/‖column(0)‖ %.1e, local kernel %d, nullity %d; t = 0: nullity %d\n",
            j, ts, length(roots), isempty(roots) ? NaN : maximum(roots) - minimum(roots),
            norm(A[:, j]) / norm(R[1][:, j]), count(iszero, eachcol(A[:, (np + 1):end])) +
            (norm(A[:, j]) < 1e-12 * norm(R[1][:, j])), gnullity(A), gnullity(R[1]))
end
x, y = Bramble.points(Ω); hx, hy1, hyn = x[2] - x[1], y[2] - y[1], y[end] - y[end - 1]
@printf("h_x/h_y at (xmin, ymin) %.6f, at (xmin, ymax) %.6f\n", hx / hy1, hx / hyn)
for t in (hx / hy1, -hx / hyn)
    K = [1.0 t; t t^2 + 1.0]
    A = Matrix(asm(constrained(W, u -> (:xmin => ∂ₙ(u; K) => 0.0,)), bilap))
    @printf("SPD K = [1 %.4f; %.4f %.4f] (eigvals %s): nullity %d\n", t, t, t^2 + 1,
            string(round.(eigvals(K); sigdigits = 3)), gnullity(A))
end
```

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
