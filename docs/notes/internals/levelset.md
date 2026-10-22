# Level-set domains on a fitted grid: design record

Design record for milestone v4.1.1 (gpena/Bramble.jl#834): `:D => levelset(φ)` as a domain
marker, a grid fitted to the curve `φ = 0`, and the polygon inscribed in it, so that
`polygon_spaces` solves the Dirichlet problem with the Ferreira and Grigorieff
([2006](https://doi.org/10.1007/s00211-006-0006-7)) scheme on a curved domain. Nothing here
is published. The worked example is `docs/src/examples/poisson_circle.jl`; the measured
convergence, as log-log plots, is
[poisson_levelset_convergence.md](../examples/poisson_levelset_convergence.md).

## What the scheme asks of the grid

The scheme needs (Geom): no grid edge joins an interior and an exterior node, so that every
cell cut by the boundary is cut along a diagonal whose other two corners are boundary
nodes, and the boundary nodes are the vertices of a polygon whose edges are cell diagonals
or cell sides. On labels from `φ`, (Geom) holds exactly when no grid edge joins a node with
`φ < 0` to one with `φ > 0`. For a polygon the grid closure adds lines until it holds. For
a curve there is a second constraint, and it is what shapes the whole design.

Suppose every grid line meets the curve in at most two points. Then each grid line inside
the curve's extent carries exactly two vertices, and the vertex set is closed under
``\sigma_v`` (the other crossing of a vertical line) and ``\sigma_h`` (of a horizontal one).
The crossing map ``T = \sigma_h \circ \sigma_v`` is an orientation-preserving homeomorphism
of the circle without fixed points, with a rotation number ``\rho``. A grid with ``N`` lines
per axis is a union of closed pseudo-orbits of ``T``, all of rotation number ``p/q``.

## The class of curves

A curve gets a fitted grid when both hold:

- **Orthogonally convex**: every horizontal and vertical line meets `φ = 0` in at most two
  points. Convexity is not needed: a dented egg is in the class.
- **The crossing map is tame**: ``T`` has finite order (``T^q = \mathrm{id}``), or it is
  conjugate to an irrational rotation. The disk, every ellipse in any orientation, eggs at
  generic tilts and the dented egg are in. One grid-aligned symmetry axis suffices for
  finite order: if the curve is symmetric about a horizontal line, ``\sigma_v`` is the
  reflection, it commutes with ``\sigma_h``, and ``T^2 = \sigma_h^2 \sigma_v^2 = \mathrm{id}``.
  An affine image of a circle (every ellipse) has ``T`` exactly conjugate to the rotation by
  twice the angle between the images of the axes.

The class is checkable from `φ` before any line is built (`_levelset_class`).

## Two paths

- **Closure path** (``T`` of finite order). The polygon closure `_grid_closure` takes a
  crossing rule as a callback: polygon edges for polygons, which stay bit for bit as in
  v4.1.0, and the root-finding crossings of `φ` for a level set. Every seed line adds the
  other coordinate of each of its crossings, until no line is new. It terminates because
  every orbit is finite. The vertices are the crossings, on the curve to round-off, and the
  user's seeds are all kept, so `npts` are seeds. Lines per axis: about 3.4n on the disk from
  jittered seeds, 2.8n on an egg with its axis on a grid line, 1.8n on the disk from uniform
  seeds (`poisson_circle`). The settled design had guessed about 2n. The grid has slivers
  (`hmin/h²` down to 0.002), which a sparse direct solve takes without
  trouble and an iterative solver would not. A locked finite-order curve with `q` up to 60
  takes this path too (integrator ruling at S2: `T^q = id` with the rotation locked at
  `p/q`, `q ≤ 60`; `q = 61` is the orbit path).
- **Orbit path** (``T`` conjugate to an irrational rotation, no finite order). Periodic
  orbits of ``T`` are started at interleaved phases of the conjugacy to the rotation and
  closed by Levenberg-Marquardt on the vertical lines alone, with ``\sigma_v`` exact by
  construction and an anchor of weight `0.5 (x-extent/ℓ)/N` to the starting positions (`ℓ` half
  the curve's larger extent). The number `N` of lines per axis inside the extent minimises
  ``N\,\|N\rho\|``, which predicts the vertices' distance to the curve over ``h^2``, within
  [`nin/1.25`, `1.25 nin`], or is the least `N` up to `4 nin` with cost at most `cmax = 2`.
  `nin` is the requested number of lines inside the extent, from the larger axis. The lines
  inside the extent are replaced, the seeds farther than a quarter of their spacing outside
  it are kept. The rotation number is the weighted Birkhoff average (Das, Sander and Yorke):
  exact to 1e-14 in 8000 steps, where the plain average is 2.4e-7 off after 40 000 steps and
  shifts the cost by up to 0.8; and its non-convergence is the near-locking test. Requested
  lines come back at 1.0 to 1.24 times for the ellipse and up to 4.2 times for a curve whose
  rotation number has a large partial quotient (an egg at 20°, ``\rho = 0.39938888``,
  partial quotient 64 after 2/5).

Both paths return the lines and the inscribed polygon `(X, Y, P)`. The labels `:D`,
`:D_boundary` and `:D_exterior` come from that polygon, not from `φ` (snapping each crossing
to the nearest perpendicular line disagrees with the construction at 2 of 52 vertices on the
ellipse at n = 40), so the polygon is passed from `_close_grid!` to `_set_markers!` and is
never stored in the mesh type, which carries no function. Both paths end in the same
post-check, `_levelset_fit`: consecutive vertices are corners of one cell, every other
node's sign agrees with its label, no grid edge joins an interior and an exterior node, and
every vertex is within ``h^2/4`` of `φ = 0` after scaling by `ℓ` (departure at S4: the bound
is scale invariant, ``\mathrm{dist}\cdot\ell \le h^2/4``, so a curve scaled by 0.01 with
scaled seeds still builds; with `ℓ` near 0.9 on the unit-scale curves it is about 9% looser
than ``0.25 h^2`` absolute, which the independent test and the probe still assert). A grid
that fails is rebuilt once with the next-best `N`, then refused.

## Boundary nodes: on the curve, or within ``0.25 h^2``

The issue asks for boundary nodes lying on the curve. On the closure path they do, to
round-off. On the orbit path they lie within ``0.25 h^2`` of `φ = 0` (measured
`dmax/h² ≤ 0.19`), the price of O(n) lines: exact vertices need finite orbits of ``T``
(next section). The inscribed polygon is then within `d + κh²/8` of the curve, with `d` the
vertex offset and `κ` the curvature.

`g` is read **at the vertex** (user ruling, 2026-10-22), so `polygon_spaces` is unchanged.
The architect's report recommended the foot of the vertex on `φ = 0` (gradient projection).
At a vertex `v` off the curve by `d ≤ 0.25 h²` the datum is `g(v)`, which differs from the
value at the foot by `|∇g| d = O(h²)`: the same order as the polygon's distance to the
curve, so second order stays. The convergence test therefore uses a `g` that equals `u` on
the curve but not off it (`g = u + 0.5 φ`), so the vertex offset is visible in the error.
Reading `g` at the foot is a follow-up (#849, with the mixed term).

With `b = 0` the five-point operator is an M-matrix, and the discrete maximum principle
bounds the max-norm error that the vertex offset induces by `|∇u| dmax`. Measured on orbit
grids `emax ≈ |∇u| dmax` (egg at 20°, n = 80: `dmax = 9.7e-4`, `emax = 1.1e-3`), so this
term dominates the scheme's own error, about ``0.02 h^2`` on the disk. Since
`dmax/h² ≈ (0.01 to 0.1)·N‖Nρ‖` and the cost is at most `cmax` by construction,
`emax/h² ≤ 0.2` and `eH1/h² ≤ 0.4` held in every run: second order with a constant set by
`cmax`. The per-level orders plateau and jump (egg at 20° at n = 20 to 80 sits at
`emax ≈ 1e-3` while `h` falls, until the cost rule jumps to q = 318), which is that constant
moving inside its bound, not a lower order. The egg at 35° shows the same mechanism. In the
architect's run (seed 1), ``\rho = 0.35287719`` lies within 6.4e-5 of `6/17`, so a 17-step
orbit misses closing by only ``\|17\rho\| \approx 0.0011`` and `N = 17m` costs about
`0.019 m²`. The rule takes `m = 1, 2, 3, 6` for n = 20 to 160; the 17-step mismatch is
spread over the lines but does not shrink with `h`, so `dmax` stays near 1e-4,
`dmax/h²` climbs from 0.004 to 0.127, and `emax` stays at 1.3e-4 to 1.7e-4 from n = 40 to
160. At n = 320 the multiples of 17 near the requested count cost more than `cmax`, the rule
moves to `N = 887` (cost 1.84), and `emax` falls to 2.3e-6. The fitted slope over these
levels is about 1.6 (1.4 on the convergence page, other seeds) with `emax/h² ≤ 0.19`
throughout. Hence the convergence test asserts a bounded
`emax/h²` and a fitted slope, never consecutive orders. For `b ≠ 0` the M-matrix argument is
gone; the paper's stability estimate covers boundary data perturbations of size O(h²). Not
measured (#849).

## Where the grid cannot exist: the impossibility result

Exact vertices first. If every crossing of a grid line with the curve is a node, the vertex
set is closed under ``\sigma_v`` and ``\sigma_h``, so it is a union of whole orbits of
``T``, and a finite grid needs finite orbits. When ``\rho`` is irrational every orbit of
``T`` is infinite (and dense on a smooth curve, by Denjoy), so no grid with finitely many
lines has every vertex on the curve, at any resolution. This is why the egg at 35°, whose
weighted Birkhoff average converges to ``\rho = 0.35287719`` with no sign of locking,
cannot take the closure path. Its symmetric twin at 0° can: the reflection across the axis
gives ``T^2 = \mathrm{id}``, so every orbit has two points. When ``\rho = p/q`` is rational
but ``T^q \ne \mathrm{id}``, only a few periodic orbits exist and the rest are attracted
to them, so the lines cannot be placed where they are needed (the phase-locked error).

With vertices allowed off the curve, they form closed pseudo-orbits of ``T`` instead, for
polygons whose edges are cell diagonals as (Geom) asks, and two facts follow.

1. A closed ``\varepsilon``-pseudo-orbit of rotation ``p/q`` exists only for
   ``\varepsilon \gtrsim |q\rho - p|/q``, so `dmax/h² ≳ c N‖Nρ‖` (measured constant 0.01 to
   0.1). A rotation number that sits near a rational forces either many lines or a large
   offset.
2. A hyperbolic periodic cycle of ``T`` with transversal crossings (an inscribed
   axis-parallel rectangle is the shortest) forbids vertices in an O(1) arc around it:
   counting vertices in an arc `A` and its image, the annulus between them holds O(1) lines
   for every `A`, so an O(1) neighbourhood would hold O(log 1/h) lines and the maximum
   spacing cannot go to zero there. No grid refines. The argument needs every polygon edge
   to be a cell diagonal (one vertex per crossing).

The star is outside the class for a simpler reason: a line meets it 4 or 6 times, ``T`` is
multivalued, and the vertex set is no longer closed under the two maps. The architect found
no O(n) construction for it and no proof that none exists.

There is an escape hatch the plan declined: a polygon may have axis-aligned micro-edges of
normal size at most ``h^2``, which break the one-vertex-per-crossing count. A greedy closure
with merge tolerance ``h^2`` then closes the star, a locked clover and the ellipse, with
`n^1.4` to `n^1.7` lines per axis and an H¹ₕ order of 1.3 to 1.5. The user chose a named
error over that fallback; a construction for the star and its kin is #847.

## Named errors

Every failure is an `ArgumentError` from `mesh`, with the prefix "level-set grid:".

| Cause | Detected by | Text (abridged) |
|---|---|---|
| not orthogonally convex (the star) | a sampled grid line meets `φ = 0` more than twice | "the grid line x = 0.31 meets φ = 0 in 4 points; the Ferreira and Grigorieff grid needs every horizontal and vertical line to meet the curve at most twice (the domain must be orthogonally convex)" |
| not star-shaped about its centre | a ray from the minimum of `φ` changes sign more than once | "the curve must be star-shaped about that point" |
| phase-locked | rotation number exactly `p/q`, `q ≤ 60`, with ``T^q \ne \mathrm{id}`` | "the curve's crossing map is phase-locked at rotation number 1/2 without being periodic ...; no Cartesian grid fits this curve at every refinement. Rotate or symmetrise the domain" |
| nearly phase-locked | the weighted Birkhoff average at 8000 and 32000 steps differs by more than 1e-10 | "the curve's crossing map is nearly phase-locked (rotation number ≈ 0.49999984, not converged); a fitted grid would need far more lines than requested" |
| too many lines | no `N ≤ 4 nin` has cost at most `cmax` | "no grid with at most 1040 lines per axis inside the curve's extent keeps its vertices within 0.2 h² of φ = 0 for the 260 lines asked there" |
| too coarse | fewer than 6 lines inside the extent | "too coarse: 5 × 5 points are too few to resolve the curve; use more" |
| check failed | the post-check refuses both grids | "the fitted grid does not fit the level set (...), so the curve would cross a grid line off the nodes. Use more points per axis" |
| curve touches the box, region empty | a side of the bounding box meets `φ ≤ 0`, or `φ ≥ 0` everywhere sampled | "φ = 0 reaches the domain's bounding box, or φ < 0 on it; the curve must lie strictly inside the box" |

The measured examples: the star `0.6 + 0.2 sin 5θ` (4 or 6 crossings on about 20% of the
lines); the clover ``r = 1 + 0.15\cos 4(\theta - 10°)``, orthogonally convex but locked at
``\rho = 1/2`` with ``\max|T^2 p - p| = 0.65`` and one inscribed square of multiplier 0.024;
an egg a fraction of a degree off symmetry (0.2°, ρ ≈ 0.49999984), where the Birkhoff
average does not converge.

## The grid's orientation matters

The class is stated for the grid's own axes. A domain rotated by a few degrees can leave it:
the clover is a rotation of a curve with grid-aligned symmetry, and an egg that has
``T^2 = \mathrm{id}`` on the axes becomes nearly locked a fraction of a degree off. A user
whose curve is refused with "phase-locked" or "nearly phase-locked" can rotate or symmetrise
the domain, which is what the first error text says.

## Limits accepted along the way

- Crossings are found by 129 samples per line and bisection of every sign change. Two roots
  between two same-sign samples are missed unless the line has no other crossing (S1).
  A tangent line is caught by a golden-section search at each sampled local minimum.
- `φ` that is NaN along a whole side of the box except a window of 1e-6 around a tangency
  misses the touch, as do windows no sample hits (S2.1).
- A rotated ellipse centred at x = 1e7 throws "nearly phase-locked": the rotation number
  carries 3e-11 of noise from round-off in `φ`'s own argument at that magnitude. Curves
  within 1e5 of the origin are exact to 1e-12 (S2).
- The egg at 35° with coarse x seeds (12 to 40 against 320 in y) is refused with "check
  failed" at 0.266 h²/ℓ, a borderline caused by the 1/ℓ anchor; scale invariance and the
  unit-scale anchor cannot both hold (S4). The 45° ellipse at n = 12 is refused the same
  way, and "too many lines" for an aspect-0.2 ellipse is not monotone in n.
- The disk built through the orbit path at ρ = 1/2 gives 37 lines at n = 40 and 136 at
  n = 160, against 143 and 547 on the closure path, with `hmin/h²` near 1 and no slivers
  (`dmax/h²` 9e-15 and 1.2e-13). At ρ = 1/2 every even `N` costs 0, so the tie-break takes
  the lowest even `N`, fewer lines than asked. That route for finite-order curves is #848.
- A level-set mesh keeps no polygon: `set_markers!`, `change_points!` and
  `iterative_refinement!` throw an `ArgumentError` on it, since the labels cannot be
  recomputed from `φ` by snapping. A domain takes one level-set marker, and none beside a
  polygon marker.
- A level-set mesh has no conflict nodes (an exterior node read by two interior nodes needs
  a line that meets the curve four times, which orthogonal convexity excludes), so the
  conflict paths of the polygon spaces never run on it.
- Neumann and Robin conditions on a level-set boundary are not covered; they come with
  v4.2.0 (#827). Level-set boundaries are Dirichlet only.
- Float32: the construction runs in Float64 and the lines are converted to the seeds' type
  with the post-check run in that type, so a merged line throws and does not mislabel.

## The issue's five points, as settled

1. **Construction for a general `φ`.** There is none for every `φ`, and the impossibility
   result above says why. There is one for a class checkable from `φ`, two paths, and a
   named error outside it.
2. **Non-convex domains.** Orthogonal convexity is the condition, not convexity. No grid
   node lies strictly between a chord of the inscribed polygon and the curve (a chord is a
   cell diagonal); `g` is evaluated only at the vertices, which are the boundary nodes; `f`
   must be defined on the dual cells of interior nodes, within `h/2` of the closed domain.
3. **Control over the grid** (`npts`). Seeds, as for polygons: the closure path keeps every
   seed and adds lines (3.4n per axis on the disk from jittered seeds), the orbit path
   replaces the lines inside the curve's extent at 1.0 to 4.2 times the requested count.
4. **Closure.** One generalised `_grid_closure` with a crossings callback: polygon edges for
   polygons (bit for bit unchanged) and level-set root finding for the closure path. The
   orbit path is not a closure (no queue, no merge tolerance, a global least squares) and
   has its own entry point `_levelset_orbit_grid`.
5. **Neumann and Robin.** Deferred to v4.2.0 (#827).

## Where it lives

| What | File |
|---|---|
| `LevelSet`, `levelset`, `_crossings` | `src/geometry/levelset.jl` |
| classification and the named errors of the class | `src/mesh/levelset_class.jl` |
| `_levelset_grid`, `_levelset_orbit_grid`, `_levelset_fit` | `src/mesh/levelset_grid.jl` |
| `_grid_closure` with the crossings callback, `_close_grid!` | `src/mesh/meshnd.jl` |
| labels by the fitted polygon, `_set_markers!` | `src/mesh/marker.jl` |
| `polygon_spaces` on a level-set mesh | `src/space/polygon_space.jl` |

Tests: `test/geometry/levelset.jl` (marker, crossings), `test/mesh/levelset_class.jl`,
`test/mesh/levelset_orbit.jl`, `test/mesh/levelset_geom.jl` (an independent node-by-node
(Geom) check against `φ`, calling no package code but `mesh`), `test/ghost/levelset_oracle.jl`
(`A` and `F` against the oracle of the scheme), `test/ghost/levelset_spaces.jl` (the polygon
spaces' capabilities on a level-set mesh), `test/convergence/levelset.jl`.

Follow-ups: #847 (curves outside the class), #848 (orbit-path grids for finite-order curves:
about n lines, no slivers), #849 (the mixed term and Dirichlet data at the foot point);
polygon-space gaps #840 to #846 apply to level-set meshes as to polygon ones.
