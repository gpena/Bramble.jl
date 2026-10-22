# Level-set domains: the design record

Gpena/Bramble.jl#834 (milestone v4.1.1) asked for domains given by a level-set function `φ`
(`φ < 0` inside), solved on the user's grid with the Shortley-Weller scheme, and left five
points open. This page records how each was settled, what was rejected and why, the limit on
thin features, and the orders measured. Decisions are numbered D1 to D19 as in the plan.

## The scheme in one paragraph

A marker `:D => levelset(φ)` classifies every node of the mesh as interior (`φ < 0`),
boundary (a node fixed to the Dirichlet datum `g`) or exterior. Each grid edge whose end
nodes differ in sign carries a crossing, the root of `φ` on it. In the equation of an
interior node `k`, an exterior neighbour `p` read along an axis takes the value at `x_p` of
the quadratic through the node behind `k`, `k` itself and the crossing between `k` and `p`,
where it is `g`. On `-Δₕ` that is the Shortley-Weller row with the arm shortened to
`θh`. The nodes stay the user's grid; refinement is a finer grid, and `npts` is the only
mesh parameter.

## The five points of #834

**1. Data type (D1).** A marker kind `:D => levelset(φ)` beside `polygon`, giving the labels
`:D`, `:D_boundary` and `:D_exterior`. `levelset(φ)` returns a `LevelSet` wrapper, not a
`Function`, because a bare function in a marker is already a predicate. `φ` lives on the
domain, never on the mesh: mesh types carry no function types. `levelset` is exported like
`polygon`. Rejected: generalising the polygon marker, which would make one type answer two
different questions (vertices against a function). A polygon can still be posed as
`levelset(x -> signed_distance(P, x))`.

**2. Crossings (D2).** Root-finding on any `φ`. Each cut edge gets a bisection on the
segment parameter, stopped when the bracket can no longer be halved in the floating-point
type. Rejected: secant and Newton, which can leave the bracket on a `φ` that is not a signed
distance, and requiring a signed distance, which excludes the natural way to write a star or
a kidney.
The price is stated under the thinness limit below: two crossings on one edge are invisible.

**3. Where the shortened arms live (D3, D4).** In the ghost-space reduction, per reading
row, not in a new stencil family. The reduction of v4.1.0 already moves reads of exterior
nodes onto other columns with an integer sign (`_exterior_moves`); it now moves them with
real weights and a load term, so the crossing value `g` enters as a constant. The rule is
per reader, so an exterior node read by two interior nodes along the two axes has two values,
one in each equation. This reproduces the Shortley-Weller row exactly for `-Δₕ` on
non-uniform grids, since any 3-point second difference is exact on quadratics, and it serves
every form that reads the 5-point stencil, such as `-Δₕ`, `-D̃ₓ(a D₋ₓ u)`, first or central
differences, reaction terms. Rejected: a new stencil family for cut edges, which would be
operator-specific code for each operator.

The per-reader rules live on the trial space (`_ArmRules` in `_RowsDirichlet`); `_Moves`
carries real weights, and the lift map gained constant terms (`liftcoef`). A form reading a
diagonal neighbour or a node two away throws the reach error. The matrix-free path and the
contraction `a(u, v)` refuse forms reading exterior nodes on a level-set space.

**4. Neumann and Robin (D7, D19).** Deferred; Dirichlet only. A Neumann or Robin condition
on the curve needs the unit normal at each crossing, which is `∇φ / |∇φ|` there, and a row
that couples the crossing value to its neighbours instead of fixing it. Neither exists yet,
and mixed derivatives (diagonal ghosts) are the other follow-up. Both are follow-ups for the
user to file with `/create-issue`, not planned here.

**5. Closure kept for polygons, level sets never close (D5).** The grid closure of v4.1.0
adds the grid lines through every crossing, so the nodes sit on the boundary. It terminates
only when the oblique edges share one slope magnitude, so it cannot reach a curve. It stays
for polygon markers. A level-set marker never closes the grid.

| Use | When |
|---|---|
| `polygon` | the edges are axis-aligned or share one slope magnitude; the boundary nodes sit exactly on the edges |
| `levelset` | the boundary is curved, or a polygon with slopes the closure rejects; the grid is yours, sized by `npts`, and the condition is Dirichlet |

## The constructor (D8)

`levelset_spaces(Ωₕ, Ω, :D, g) -> (U, W)` is `public`, not exported, like `polygon_spaces`.
It reads `φ` from the `:D` marker of `Ω` and checks the labels of the mesh against a fresh
classification, so a mesh not built from `Ω`, or from another `φ`, throws an `ArgumentError`.

## Short arms: the boundary-node rule (D9)

No snapping. A node with a crossing within `√ε · h` of it on an incident edge (`ε` the
machine epsilon, `h` that edge's length), or with `φ = 0`, is a boundary node fixed to `g`.
Every other arm `θ > 0` keeps its exact Shortley-Weller row, however small. Rejected:
snapping a near node onto the curve, which changes the geometry the scheme solves. The
boundary-node rule only removes the arms so short that the crossing is at rounding distance
from the node.

## The thinness limit (D10)

The grid resolves a feature only if it is thicker than about one cell. Two failures are
possible and they are not alike.

- **Saddle cells throw.** A cell whose four edges carry four crossings (a saddle sign
  pattern, the diagonal corners of one sign) has a curve that enters and leaves it twice.
  The grid cannot say which crossings join, so `levelset_spaces` throws an `ArgumentError`
  naming the cell and asking for a finer grid.
- **Two crossings on one edge are invisible.** A curve that leaves an edge and returns
  within it leaves both end nodes the same sign. The nodes do not show it and no solve is
  tried; the domain is read as if the curve missed the edge. This is a limit of any
  node-sign classification; it is stated in the docs, not detected.

This departs from the issue's guess, "a node with no interior neighbour on either side along
an axis". Shortley-Weller handles a node cut on both sides, through both crossings and `u_k`,
so that case is no failure.

`levelset_spaces` also refuses an interior node on the grid's outer boundary: it has no
neighbour beyond it for the quadratic to use. Enlarge the box so the domain lies inside it.

## The inscribed polygon (D11)

#834 asked for the polygon inscribed in the curve from the grid: its vertices are the
crossings, and consecutive crossings bound a piece of the curve in one cell, so each chord
stays in its cell. There is no such object in the code. The scheme needs only the crossings
and the arms they cut, never the chords. What the polygon contributes is a well-definedness
statement, that every cell contains at most one piece of the curve, and the per-cell check of
D10 is exactly that statement. The inscribed polygon is therefore a concept recorded here,
and a departure from the first bullet of #834, which read as an object.

## Measured orders

The prototype (plain Julia, Shortley-Weller rows built directly, `u = sin 2x cos 2y + xy` on
the box ``[-1.25, 1.25]^2``) gave, as the slope of log error against log `hₘₐₓ`, over
n = 20 to 320 on jittered grids with three seeds pooled, max-norm orders 2.51 to 2.64 and
H¹ₕ orders 1.92 to 2.03. Single i.i.d. random grids scatter far more, from 1.46 to 2.61 in
the max norm, which is why the gate pools seeds and jitters rather than draws freely; on
uniform grids the H¹ₕ order is 2.00 for the disk and 2.04 for the star.

The library, through `levelset_spaces` (`test/convergence/poisson_levelset.jl`, 3 seeds
pooled, n = 20 to 320), measured:

| Case | Max order | H¹ₕ order |
|---|---|---|
| disk, `-Δₕ` | 2.607 | 1.997 |
| star, `-Δₕ` | 2.598 | 1.957 |
| disk, variable coefficients | 2.311 | 1.975 |

The plot of the same three cases, one seed, is written by
`../examples/poisson_levelset_convergence.jl` into
[`../examples/poisson_levelset_convergence.md`](../examples/poisson_levelset_convergence.md); its
fitted slopes differ slightly from the pooled ones. The gate requires at least 1.8 in both
norms.

The polygon path is bit-for-bit unchanged by this work, as the adversarial review of the
branch confirmed.
