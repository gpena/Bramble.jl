```@meta
CurrentModule = Bramble
```

# [Ghost points](@id tutorial_ghost)

A second-order Neumann or Robin condition needs the solution one point past the boundary, and every stencil stops at the last point. A ghost point is that extra point. You will pad a mesh with ghosts, build a grid space on it, see how difference operators read a ghost, tie the ghosts to boundary data with relations, solve, and step a heat equation in time. The [mesh](mesh.md), [space](space.md) and [operators](operators.md) tutorials come first, and the [form tutorial](form.md) for `form` and `assemble`.

```@setup ghost
using Bramble
import Bramble: change_points!, indices, point
using LinearAlgebra: norm, dot, diag
```

---

## [Ghost points beyond the boundary](@id mesh_ghost)

A [`Bramble.GhostMesh`](@ref) wraps a mesh and pads it with one ghost point beyond each node of the facets you name, either `:boundary` or a tuple such as `(:xmin, :ymax)`. `GhostMesh` is `public` and not exported, and you rarely build one yourself: [`gridspace`](@ref)`(Ωₕ; ghosts = ...)` builds one with a space on top. The running 1D mesh has six points graded by ``x \mapsto x^2``:

```@example ghost
Ω₁ = mesh(domain(interval(0.0, 1.0)), 6)
change_points!(Ω₁, collect(range(0.0, 1.0, length = 6) .^ 2))
x₁ = collect(points(Ω₁))
h₂, hₙ = x₁[2] - x₁[1], x₁[6] - x₁[5]

G₁ = Bramble.GhostMesh(Ω₁; ghosts = :boundary)
xg = collect(points(G₁))

@assert xg[1] ≈ x₁[1] - h₂ && xg[end] ≈ x₁[end] + hₙ
@assert xg[2:(end - 1)] == x₁      # the physical points are untouched
@assert (try; Bramble.GhostMesh(Ω₁; ghosts = (:ymin,)); false; catch e; e isa ArgumentError; end)

npoints(G₁), Bramble.nghosts(G₁), (xg[1], xg[end])
```

A ghost sits one boundary spacing outside its facet node: ``x_1 - h_2`` below and ``x_n + h_n`` above, with ``h_2 = 0.04`` and ``h_n = 0.36`` here, so on a graded mesh the two ghosts do not mirror each other. A facet the dimension does not have, such as `:ymin` on an interval, is an `ArgumentError`.

In two dimensions only the facet nodes get ghosts. The padded mesh is a tensor product, so it also has points outside along two axes at once, the four corners. Those points are *absent*: they are not nodes and carry no degree of freedom. The running 2D mesh has ``4 \times 3`` nodes:

```@example ghost
Ω_g = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ_g = mesh(Ω_g, (4, 3), (true, true))
change_points!(Ωₕ_g, markers(Ω_g), ([0.0, 0.2, 0.6, 1.0], [0.0, 0.5, 1.0]))

G₂ = Bramble.GhostMesh(Ωₕ_g; ghosts = :boundary)
absent = [Bramble.is_absent(G₂, I) for I in CartesianIndices(npoints(G₂, Tuple))]

@assert sum(absent) == 4 && all(absent[[1, end], [1, end]])
@assert prod(npoints(G₂, Tuple)) - sum(absent) == prod(npoints(Ωₕ_g, Tuple)) + Bramble.nghosts(G₂)

npoints(G₂, Tuple), Bramble.nghosts(G₂), sum(absent)
```

The padded size is ``(4 + 2) \times (3 + 2)``. The ghosts are ``2 \cdot 3 + 2 \cdot 4 = 14``, one per facet node, and the padded grid minus its four absent corners leaves the nodes and exactly those.

A ghost mesh follows its physical mesh. After `change_points!` the ghosts move with the facet spacings. After `iterative_refinement!` the point count changes, which a ghost mesh cannot follow: build a new one.

---

## [Ghost spaces](@id space_ghost)

The keyword `ghosts` of [`gridspace`](@ref) names the facets that get ghosts. The default `ghosts = ()` returns the plain space.

```@example ghost
W_plain = gridspace(Ωₕ_g)
W_ghost = gridspace(Ωₕ_g; ghosts = :boundary)
W_part = gridspace(Ωₕ_g; ghosts = (:xmin, :ymax))

@assert typeof(gridspace(Ωₕ_g; ghosts = ())) == typeof(W_plain)

ndofs(W_plain), ndofs(W_ghost), ndofs(W_part)
```

The degrees of freedom of a ghost space are the physical nodes plus the ghosts. Here ``12`` nodes plus ``2 \cdot 3 + 2 \cdot 4 = 14`` ghosts give 26, and the two named facets give ``12 + 3 + 4 = 19``. The absent corners are not degrees of freedom.

### The order of the degrees of freedom

The physical nodes come first, in the mesh's linear order. The ghosts follow, facet by facet in the order `xmin, xmax, ymin, ymax, zmin, zmax` (only the named ones), each facet's ghosts in the column-major order of its nodes. A system assembled on a ghost space is indexed this way. On a space without Dirichlet relations or relation data, its first rows and columns are the physical nodes.

Each degree of freedom is one point of the padded mesh. `Bramble.padded_space` is the internal space on that mesh, and `Bramble.closure` returns the sparse matrix `E` that sends the degrees of freedom to padded values, one entry ``1`` per column.

```@example ghost
E, _ = Bramble.closure(W_ghost)
Pₕ = Bramble.padded_space(W_ghost)
padded = CartesianIndices(npoints(mesh(Pₕ), Tuple))

dof_point(j) = point(mesh(Pₕ), padded[findfirst(!iszero, E[:, j])])

@assert [dof_point(j) for j in 1:12] == vec([point(Ωₕ_g, I) for I in indices(Ωₕ_g)])
@assert all(dof_point(j)[1] < 0.0 for j in 13:15)        # xmin ghosts, at x = -h₂
@assert all(dof_point(j)[1] > 1.0 for j in 16:18)        # xmax ghosts, at x = 1 + hₙ
@assert all(dof_point(j)[2] < 0.0 for j in 19:22)        # ymin ghosts
@assert all(dof_point(j)[2] > 1.0 for j in 23:26)        # ymax ghosts

[dof_point(j) for j in (1, 12, 13, 15, 16, 18, 19, 22, 23, 26)]
```

The first two entries are the first and last physical node. The next four are the `xmin` and `xmax` ghosts at ``x = -0.2`` and ``x = 1.4``, the distances ``h_2 = 0.2`` and ``h_n = 0.4`` from the facet. The last four are the `ymin` and `ymax` ghosts at ``y = -0.5`` and ``y = 1.5``.

```@raw html
<figure style="margin:1.5em 0;text-align:center">
<svg viewBox="0 0 560 340" width="100%" style="max-width:560px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif" role="img"
     aria-label="Padded 6 by 5 grid of the 4 by 3 non-uniform mesh: 12 physical nodes numbered 1 to 12, 14 ghost points numbered 13 to 26, and four absent corners.">
  <!-- padded grid lines -->
  <g stroke="currentColor" stroke-opacity="0.25" stroke-dasharray="4 3">
    <line x1="40" y1="300" x2="40" y2="20"/>
    <line x1="100" y1="300" x2="100" y2="20"/>
    <line x1="160" y1="300" x2="160" y2="20"/>
    <line x1="280" y1="300" x2="280" y2="20"/>
    <line x1="400" y1="300" x2="400" y2="20"/>
    <line x1="520" y1="300" x2="520" y2="20"/>
    <line x1="40" y1="300" x2="520" y2="300"/>
    <line x1="40" y1="230" x2="520" y2="230"/>
    <line x1="40" y1="160" x2="520" y2="160"/>
    <line x1="40" y1="90" x2="520" y2="90"/>
    <line x1="40" y1="20" x2="520" y2="20"/>
  </g>
  <!-- physical mesh -->
  <rect x="100" y="90" width="300" height="140" fill="#3b82f6" fill-opacity="0.08" stroke="#3b82f6" stroke-width="1.6"/>
  <!-- absent corners -->
  <g stroke="#ef4444" stroke-width="2">
    <line x1="34" y1="294" x2="46" y2="306"/><line x1="34" y1="306" x2="46" y2="294"/>
    <line x1="34" y1="14" x2="46" y2="26"/><line x1="34" y1="26" x2="46" y2="14"/>
    <line x1="514" y1="294" x2="526" y2="306"/><line x1="514" y1="306" x2="526" y2="294"/>
    <line x1="514" y1="14" x2="526" y2="26"/><line x1="514" y1="26" x2="526" y2="14"/>
  </g>
  <!-- physical nodes: 1 to 12 -->
  <g fill="currentColor">
    <circle cx="100" cy="230" r="5"/>
    <circle cx="160" cy="230" r="5"/>
    <circle cx="280" cy="230" r="5"/>
    <circle cx="400" cy="230" r="5"/>
    <circle cx="100" cy="160" r="5"/>
    <circle cx="160" cy="160" r="5"/>
    <circle cx="280" cy="160" r="5"/>
    <circle cx="400" cy="160" r="5"/>
    <circle cx="100" cy="90" r="5"/>
    <circle cx="160" cy="90" r="5"/>
    <circle cx="280" cy="90" r="5"/>
    <circle cx="400" cy="90" r="5"/>
  </g>
  <g font-size="12" text-anchor="start" fill="currentColor">
    <text x="108" y="222">1</text>
    <text x="168" y="222">2</text>
    <text x="288" y="222">3</text>
    <text x="408" y="222">4</text>
    <text x="108" y="152">5</text>
    <text x="168" y="152">6</text>
    <text x="288" y="152">7</text>
    <text x="408" y="152">8</text>
    <text x="108" y="82">9</text>
    <text x="168" y="82">10</text>
    <text x="288" y="82">11</text>
    <text x="408" y="82">12</text>
  </g>
  <!-- ghost points: 13 to 26 -->
  <g fill="#fff" stroke="#10b981" stroke-width="2">
    <circle cx="40" cy="230" r="5"/>
    <circle cx="40" cy="160" r="5"/>
    <circle cx="40" cy="90" r="5"/>
    <circle cx="520" cy="230" r="5"/>
    <circle cx="520" cy="160" r="5"/>
    <circle cx="520" cy="90" r="5"/>
    <circle cx="100" cy="300" r="5"/>
    <circle cx="160" cy="300" r="5"/>
    <circle cx="280" cy="300" r="5"/>
    <circle cx="400" cy="300" r="5"/>
    <circle cx="100" cy="20" r="5"/>
    <circle cx="160" cy="20" r="5"/>
    <circle cx="280" cy="20" r="5"/>
    <circle cx="400" cy="20" r="5"/>
  </g>
  <g font-size="12" text-anchor="start" fill="#10b981">
    <text x="48" y="222">13</text>
    <text x="48" y="152">14</text>
    <text x="48" y="82">15</text>
    <text x="528" y="222">16</text>
    <text x="528" y="152">17</text>
    <text x="528" y="82">18</text>
    <text x="108" y="292">19</text>
    <text x="168" y="292">20</text>
    <text x="288" y="292">21</text>
    <text x="408" y="292">22</text>
    <text x="108" y="12">23</text>
    <text x="168" y="12">24</text>
    <text x="288" y="12">25</text>
    <text x="408" y="12">26</text>
  </g>
  <!-- legend -->
  <g font-size="12" fill="currentColor">
    <circle cx="40" cy="328" r="5" fill="currentColor"/><text x="50" y="332">node 1 to 12</text>
    <circle cx="150" cy="328" r="5" fill="#fff" stroke="#10b981" stroke-width="2"/><text x="160" y="332">ghost 13 to 26</text>
    <g stroke="#ef4444" stroke-width="2"><line x1="274" y1="322" x2="286" y2="334"/><line x1="274" y1="334" x2="286" y2="322"/></g><text x="292" y="332">absent corner</text>
  </g>
</svg>
<figcaption style="font-size:0.9em;opacity:0.8">The padded grid of the 4 &times; 3 mesh. Filled dots are nodes, open dots are ghosts, crosses are the four absent corners.</figcaption>
</figure>
```

In the figure, ghost 13 sits one spacing left of node 1 and ghost 19 one spacing below it. The four padded corners are absent.

```@example ghost
xs_fig, ys_fig = [0.0, 0.2, 0.6, 1.0], [0.0, 0.5, 1.0] # hide
drawn = vcat([(x, y) for y in ys_fig for x in xs_fig],             # 1 to 12 # hide
             [(-0.2, y) for y in ys_fig], [(1.4, y) for y in ys_fig],   # 13 to 18 # hide
             [(x, -0.5) for x in xs_fig], [(x, 1.5) for x in xs_fig])   # 19 to 26 # hide

@assert length(drawn) == ndofs(W_ghost) == 26 # hide
@assert all(collect(dof_point(j)) ≈ [drawn[j]...] for j in 1:26) # hide
@assert [Bramble.is_absent(G₂, I) for I in CartesianIndices((6, 5))] == # hide
        [i in (1, 6) && j in (1, 5) for i in 1:6, j in 1:5] # hide
drawn[[1, 13, 19]]
```

### Physical-only weights

A ghost carries no quadrature weight. The inner products `innerₕ`, `inner₊` and `inner_Γ` weigh the physical nodes, cells and facets exactly as on the plain space, and give zero weight to every ghost. A mass matrix assembled on the ghost space shows it:

```@example ghost
M_plain = assemble(form(W_plain, W_plain, (u, v) -> innerₕ(u, v)))
M_ghost = assemble(form(W_ghost, W_ghost, (u, v) -> innerₕ(u, v)))

@assert diag(M_ghost)[1:12] == diag(M_plain)
@assert all(iszero, diag(M_ghost)[13:end])
@assert sum(diag(M_ghost)) ≈ 1.0       # the physical cells tile the unit square

sum(diag(M_ghost)[1:12]), sum(diag(M_ghost)[13:end])
```

The weights of `inner_Γ` are facet weights: the transverse measure at each facet node, a half cell at the facet's ends. The markers `:xmin` to `:zmax` name the physical facet nodes. `gridspace` of a bare `GhostMesh` is refused, since its weights would count the ghosts: build the space from the physical mesh with `ghosts`.

---

## [Operators at a ghost boundary](@id operators_ghost)

A difference at the first point has no neighbour, so it is truncated. A ghost space supplies the neighbour. The rows below are read off the matrices of forms: `form(Wₕ, Wₕ, (u, v) -> ...)` takes the unknown `u` and a test function `v` symbolically, and `assemble` returns the matrix. The 1D mesh `Ω₁` with ghosts on both ends has the six nodes as degrees of freedom, then the `xmin` ghost (7), then the `xmax` ghost (8):

```@example ghost
import Bramble: Dcₓ, Dcᵧ, D₋ₓ, D₋ᵧ, D̃ₓ, D̃ᵧ

W₁ = gridspace(Ω₁; ghosts = :boundary)
row(f, i) = Matrix(assemble(form(W₁, W₁, f)))[i, :]

ndofs(W₁)
```

### How differences read the ghost

At a ghosted facet the stencil of every operator reads the ghost as if it were a neighbour, with the boundary spacing repeated beyond it: ``h_1 := h_2`` at the first point and ``h_{n+1} := h_n`` at the last. In 1D an `inner_Γ` term has weight ``1`` at the facet node, so its row is the stencil itself. The backward difference at the first node is ``(u_1 - u_0)/h_2``:

```@example ghost
r = row((u, v) -> inner_Γ(D₋ₓ(u), v; markers = (:xmin,)), 1)

@assert r[[1, 7]] ≈ [1 / h₂, -1 / h₂] && count(!iszero, r) == 2

r
```

Entry 1 is the node, entry 7 the ghost, and no other point is read. A facet without ghosts is truncated as in the [operators tutorial](@ref tutorial_operators): with a ghost on `xmin` only, the centred difference of a volume term is zero at the last node and reads the ghost at the first. A boundary term that needs a point beyond a facet without ghosts, `inner_Γ(Dcₓ(u), v; markers = (:xmax,))` or the same with `∂ₙ` on a space with `ghosts = (:xmin,)`, throws an `ArgumentError` naming the padded point the space lacks.

### The normal derivative and the facet operators

Three operators express the boundary conditions that name a normal direction. They act at the nodes of a facet with outward normal ``\eta`` and are only meaningful inside [`inner_Γ`](@ref) or a [`constrained`](@ref) relation.

| Operator | Meaning at a facet node, axis ``d`` |
|:--|:--|
| `∂ₙ(u)` | ``\eta\, D_{c,d} u``: the centred difference reading the ghost, times the outward sign |
| `∂ₙₙ(u)` | ``\tilde D_d D_{-d}\, u``: the second difference along the normal, reading the ghost |
| `jumpₙ(w)` | ``w_{I - \eta e_d} - w_I``: the inward jump, which reads no ghost |

On `Ω₁` the stencils are explicit, at both ends:

```@example ghost
r∂ₙ_min = row((u, v) -> inner_Γ(∂ₙ(u), v; markers = (:xmin,)), 1)
r∂ₙ_max = row((u, v) -> inner_Γ(∂ₙ(u), v; markers = (:xmax,)), 6)
r∂ₙₙ_min = row((u, v) -> inner_Γ(∂ₙₙ(u), v; markers = (:xmin,)), 1)
r∂ₙₙ_max = row((u, v) -> inner_Γ(∂ₙₙ(u), v; markers = (:xmax,)), 6)
rjump_min = row((u, v) -> inner_Γ(jumpₙ(u), v; markers = (:xmin,)), 1)
rjump_max = row((u, v) -> inner_Γ(jumpₙ(u), v; markers = (:xmax,)), 6)

@assert r∂ₙ_min[[2, 7]] ≈ [-1 / (2h₂), 1 / (2h₂)]     # -(u₂ - u₀) / (2h₂)
@assert r∂ₙ_max[[5, 8]] ≈ [-1 / (2hₙ), 1 / (2hₙ)]     # (u₇ - u₅) / (2hₙ)
@assert r∂ₙₙ_min[[1, 2, 7]] ≈ [-2, 1, 1] ./ h₂^2      # (u₂ - 2u₁ + u₀) / h₂²
@assert r∂ₙₙ_max[[5, 6, 8]] ≈ [1, -2, 1] ./ hₙ^2      # (u₅ - 2u₆ + u₇) / hₙ²
@assert rjump_min[[1, 2]] == [-1.0, 1.0] && rjump_min[[7, 8]] == [0.0, 0.0]
@assert rjump_max[[5, 6]] == [1.0, -1.0] && rjump_max[[7, 8]] == [0.0, 0.0]

r∂ₙ_min, r∂ₙ_max, r∂ₙₙ_min, rjump_max
```

The jump is the inward one at both ends, ``u_2 - u_1`` at the left and ``u_{n-1} - u_n`` at the right, which is why it needs no ghost of its own. The plain forward `jumpₓ` would read across the ghost cell at the right end. A facet operator takes scalars and sums, so `α * u + β * ∂ₙ(u)` is the Robin combination. `jumpₙ(∂ₙ(u))` throws an `ArgumentError` that begins "jumpₙ takes a field expression", a function coefficient times a facet operator is a `MethodError` (no method matching `*`), and the function belongs in the data of `inner_Γ` or of a relation.

### One value per incident facet

A corner lies on two facets, so a node there has two normal derivatives. Inside `inner_Γ`, a facet operator expands into one product per facet, each with its own facet weight and its own ghost; `markers` chooses the facets. Take the corner node of the 2D grid, degree of freedom 1. The `xmin` ghost of its row is degree of freedom 13 and the `ymin` ghost of its column is 19:

```@example ghost
row_c(f) = Matrix(assemble(form(W_ghost, W_ghost, f)))[1, :]
r_x = row_c((u, v) -> inner_Γ(∂ₙ(u), v; markers = (:xmin,)))
r_y = row_c((u, v) -> inner_Γ(∂ₙ(u), v; markers = (:ymin,)))
r_xy = row_c((u, v) -> inner_Γ(∂ₙ(u), v; markers = (:xmin, :ymin)))

@assert r_x[13] != 0 && r_x[19] == 0         # the xmin term reads only the xmin ghost
@assert r_y[19] != 0 && r_y[13] == 0
@assert r_xy ≈ r_x + r_y

r_x[13], r_y[19]
```

The two entries differ, ``0.625`` and ``0.1``, because each facet term carries its facet's weight, and at the corner that weight is a half cell. A coefficient per facet is a sum over `markers`, one `inner_Γ` per facet: `inner_Γ(2.0 * ∂ₙ(u), v; markers = (:xmin,)) + inner_Γ(5.0 * ∂ₙ(u), v; markers = (:ymin,))` gives the row `2.0 * r_x + 5.0 * r_y`.

The same holds for `∂ₙₙ` and `jumpₙ`. The conormal derivative of a constant tensor ``K``, `∂ₙ(u; K = K)`, is ``\eta_F\,(K\, D_c u)`` per facet. With a mixed entry ``K_{12}`` the `xmin` term also differences along ``y``, so at the corner it reads the `ymin` ghost too:

```@example ghost
K_mixed = [1.0 0.5; 0.5 2.0]
r_K = row_c((u, v) -> inner_Γ(∂ₙ(u; K = K_mixed), v; markers = (:xmin,)))

@assert r_K[19] != 0 && r_K[13] != 0

r_K[13], r_K[19]
```

Two relations that couple two ghosts at a corner are what [`constrained`](@ref) solves as a ``2 \times 2`` block. A conormal term with a nonzero off-diagonal entry ``K_{de}`` on a facet of axis ``d`` is refused where its corner lacks the ghosts of the ``e`` facets, and ``K`` must be a constant matrix of numbers.

### The Laplacian inside a form

On trial and test functions, [`Δₕ`](@ref) is one operator: the sum over the axes of the second difference ``\tilde D_d D_{-d}``. It assembles to the matrix of the spelled-out sum, on a space without ghosts and on a ghost space:

```@example ghost
lap(W) = assemble(form(W, W, (u, v) -> innerₕ(Δₕ(u), v)))
spelled(W) = assemble(form(W, W, (u, v) -> innerₕ(D̃ₓ(D₋ₓ(u)) + D̃ᵧ(D₋ᵧ(u)), v)))

@assert norm(lap(W_plain) - spelled(W_plain)) <= 1e-13 * norm(lap(W_plain))
@assert norm(lap(W_ghost) - spelled(W_ghost)) <= 1e-13 * norm(lap(W_ghost))
@assert norm(lap(W_ghost)) > 0

norm(lap(W_ghost) - spelled(W_ghost)) / norm(lap(W_ghost))
```

At a facet node on a ghost space it is the ordinary three-point stencil with the ghost as a neighbour. On `W₁`, the row of the first node is the cell weight ``h_2/2`` times ``(u_2 - 2u_1 + u_0)/h_2^2``, as `row((u, v) -> innerₕ(Δₕ(u), v), 1)` shows. It nests with the other operators, `jumpₓ(Δₕ(u))` or `Dcₓ(Δₕ(u))`, and with itself.

### Mixed differences at a corner

The mixed difference ``D_{c,x} D_{c,y}`` at a corner node would read the padded corner point, which is absent. On a ghost space the stencil there is the inward one-sided difference along both axes, which is second order and reads only nodes. At the corner node of the grid above, with ``h_x = 0.2`` and ``h_y = 0.5``, the row is the cell weight ``h_x h_y / 4`` over ``h_x h_y`` times ``+1, -1, -1, +1`` on the nodes ``(1,1), (2,1), (1,2), (2,2)``:

```@example ghost
r_mixed = row_c((u, v) -> innerₕ(Dcₓ(Dcᵧ(u)), v))

@assert r_mixed[[1, 2, 5, 6]] ≈ 0.25 .* [1, -1, -1, 1]
@assert count(!iszero, r_mixed) == 4

r_mixed[[1, 2, 5, 6]]
```

Every other mixed difference that would read an absent point is refused with an `ArgumentError`. Nested mixed differences on a ghost space are assembled as forms, as here. The runtime operators are eager: `Dcᵧ(uₕ)` returns a finished grid function before `Dcₓ` sees it, so on a lifted grid function the composition at the ghosts next to a corner is not claimed to equal the form.

---

## [Neumann and Robin conditions](@id form_ghost)

A Dirichlet condition fixes a value, and the rows of the [form tutorial](@ref form_dirichlet) replace it. A Neumann or Robin condition fixes a normal derivative, and a second-order scheme needs the ghost to write it. The model problem on the interval is

```math
-u'' = f \text{ on } (0, 1), \qquad \partial_\eta u = g_0 \text{ at } 0, \qquad
\alpha u + \partial_\eta u = g_1 \text{ at } 1,
```

where ``\partial_\eta`` is the derivative along the outward normal, ``-u'`` at the left end and ``u'`` at the right. The exact solution is ``u = \cos \pi x + x``, and the data come from it:

```@example ghost
function graded_interval(n)
    Ωₕ = mesh(domain(interval(0.0, 1.0)), n)
    change_points!(Ωₕ, collect(range(0.0, 1.0, length = n) .^ 2))
    return Ωₕ
end

u_exact(x) = cos(π * x[1]) + x[1]
du_exact(x) = -π * sin(π * x[1]) + 1
f_src(x) = π^2 * cos(π * x[1])
robin_α = 2.0
g_left(x) = -du_exact(x)                         # ∂ₙu = -u' at x = 0
g_right(x) = robin_α * u_exact(x) + du_exact(x)  # αu + u' at x = 1

Ω_gh = graded_interval(33)
x_gh = collect(points(Ω_gh))
extrema(diff(x_gh))
```

The spacings run from ``10^{-3}`` to ``0.06``.

### Without ghosts: the forms you already have

The weak form of the problem is ``(D_{-}u, D_{-}v)_+`` plus a boundary term per facet, and [`inner_Γ`](@ref) writes both. The Robin coefficient multiplies `u` inside the matrix and the data go on the right:

```@example ghost
Wₕ_gh = gridspace(Ω_gh)
a_modeA = form(Wₕ_gh, Wₕ_gh, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) +
    inner_Γ(robin_α * u, v; markers = (:xmax,)))
l_modeA = form(Wₕ_gh, v -> innerₕ(f_src, v) +
    inner_Γ(g_left, v; markers = (:xmin,)) + inner_Γ(g_right, v; markers = (:xmax,)))

A_modeA, F_modeA = assemble(a_modeA, l_modeA)
u_modeA = pde_solve(A_modeA, F_modeA)

err_modeA = maximum(abs, u_modeA - u_exact.(x_gh))
@assert err_modeA < 2e-2

err_modeA
```

Nothing here needs a ghost: a ghost-eliminated Neumann or Robin condition is this system. Ghosts matter when the scheme differentiates the unknown at the boundary, for instance the collocation form ``(-\Delta_h u, v)_h``, a conormal condition with a mixed derivative, or a condition on a second normal derivative.

### Boundary relations

Give the space ghosts on the facets that carry such a condition, and tie each facet's ghosts to a *relation*, written `operator => data`. [`constrained`](@ref) takes a function of the unknown `u` that returns one `facet => relation` pair per facet:

```@example ghost
W_ghosts = gridspace(Ω_gh; ghosts = :boundary)
C_gh = constrained(W_ghosts, u -> (
    :xmin => ∂ₙ(u) => g_left,
    :xmax => robin_α * u + ∂ₙ(u) => g_right))

ndofs(W_ghosts), ndofs(C_gh)
```

`=>` binds looser than `+` and `*` and is right-associative, so the second line reads `:xmax => ((robin_α * u + ∂ₙ(u)) => g_right)`. The data is a number or a function of the point tuple, as for [`dirichlet_constraints`](@ref). `constrained` solves each ghost's relation for the ghost, and its unknowns are the 33 physical nodes again: the 2 ghosts were eliminated.

The form below is the collocation Laplacian at every node, the boundary ones included, with the cell weight as the test weight. Assemble both forms together with `assemble(a, l)`: the data move the ghosts, so the load is ``E^T (F - A d)`` and needs the matrix.

```@example ghost
a_gh = form(C_gh, C_gh, (u, v) -> innerₕ(-Δₕ(u), v))
l_gh = form(C_gh, v -> innerₕ(f_src, v))
A_gh, F_gh = assemble(a_gh, l_gh)
u_gh = pde_solve(A_gh, F_gh)

@assert norm(A_gh - A_modeA) <= 1e-12 * norm(A_modeA)
@assert norm(F_gh - F_modeA) <= 1e-12 * norm(F_modeA)
@assert maximum(abs, u_gh - u_modeA) <= 1e-10

norm(A_gh - A_modeA) / norm(A_modeA), maximum(abs, u_gh - u_exact.(x_gh))
```

The ghost-eliminated matrix and load equal those of the forms of the previous section to rounding. The error decreases at second order on this graded mesh: it falls by a factor of four when the number of points doubles.

```@example ghost
function ghost_error(n)
    Ωₕ = graded_interval(n)
    C = constrained(gridspace(Ωₕ; ghosts = :boundary), u -> (
        :xmin => ∂ₙ(u) => g_left, :xmax => robin_α * u + ∂ₙ(u) => g_right))
    A, F = assemble(form(C, C, (u, v) -> innerₕ(-Δₕ(u), v)),
        form(C, v -> innerₕ(f_src, v)))
    return maximum(abs, pde_solve(A, F) - u_exact.(points(Ωₕ)))
end

ratio = ghost_error(33) / ghost_error(65)
@assert 3.5 < ratio < 4.5

ratio
```

The relations a facet takes:

| Relation | Condition |
|:--|:--|
| `∂ₙ(u) => g` | Neumann |
| `α * u + β * ∂ₙ(u) => g` | Robin, with ``\beta \neq 0`` |
| `∂ₙ(u; K = K) => g` | conormal, ``\eta \cdot (K \nabla u)``, with a constant matrix `K` |
| `∂ₙₙ(u) => g` | second normal difference, ``(\tilde D D_{-} u)`` at the facet node |
| `u => g` | Dirichlet, see below |

Each relation of a ghosted facet must read that facet's ghost: `β = 0` or a lone `jumpₙ` is an `ArgumentError`. The stencils behind each relation are the rows of the previous section and the [operators page](../api/operators.md).

On a composite space `S × S`, `constrained` also takes relations that read several components of `u`, such as the traction of elasticity. Component ``i`` of the traction is ``\sum_j \eta \cdot (K_{ij} \nabla u_j)``, a sum of conormal derivatives with off-diagonal `K`. A facet whose relation reads several components needs one ghost relation per component, written unkeyed as `:xmin => t₁ => g₁, :xmin => t₂ => g₂`. The ghosts of a node are solved together, so component 1 may read the ghost of component 2. A displacement facet has ghosts as well. They are closed by `∂ₙₙ(u(i)) => 0` beside the Dirichlet relation `u(i) => 0` of each component.

A component of the lifted displacement is `Bramble.lift(Cₕ, u)(i)`, since a component view of the constrained space itself is refused because the closure couples the components. The [elasticity example](../examples/elasticity_traction_2d.md) solves a traction problem this way. The refusals and the rule for mixed differences on a traction facet are in the [space internals](../internals/space.md).

### Dirichlet conditions beside ghost relations

Keep the Robin condition at ``x = 1`` and impose ``u(0) = 1`` instead, so only `xmax` needs a ghost. There are two routes, and they give the same solution.

Route (a), row replacement. `constrained` receives only the ghost relation. The unknowns are the physical nodes, as on a space without ghosts, so the `dirichlet` keyword of `assemble` replaces the rows of `:xmin` and `symmetrize` restores symmetry, as in the [Dirichlet section](@ref form_dirichlet):

```@example ghost
W_right = gridspace(Ω_gh; ghosts = (:xmax,))
C_a = constrained(W_right, u -> (:xmax => robin_α * u + ∂ₙ(u) => g_right,))
a_a = form(C_a, C_a, (u, v) -> innerₕ(-Δₕ(u), v))
l_a = form(C_a, v -> innerₕ(f_src, v))
A_a, F_a = assemble(a_a, l_a; dirichlet = :xmin => u_exact, symmetrize = true)
u_a = pde_solve(A_a, F_a)

@assert length(u_a) == 33 && u_a[1] ≈ u_exact(x_gh[1])
@assert norm(A_a - A_a') <= 1e-12 * norm(A_a)

ndofs(C_a), maximum(abs, u_a - u_exact.(x_gh))
```

Route (b), `u => g` in the relations. The Dirichlet node leaves the unknowns: its column is removed and its value goes into the data. The matrix is ``E^T A E``, symmetric by construction, and it has one row less. The solution vector has one entry per kept node, so the Dirichlet value is prepended below:

```@example ghost
C_b = constrained(W_right, u -> (
    :xmin => u => u_exact,
    :xmax => robin_α * u + ∂ₙ(u) => g_right))
a_b = form(C_b, C_b, (u, v) -> innerₕ(-Δₕ(u), v))
l_b = form(C_b, v -> innerₕ(f_src, v))
A_b, F_b = assemble(a_b, l_b)
u_b = pde_solve(A_b, F_b)

@assert ndofs(C_b) == 32 && size(A_b) == (32, 32)
@assert norm(A_b - A_b') <= 1e-12 * norm(A_b)
@assert maximum(abs, [u_exact(x_gh[1]); u_b] - u_a) <= 1e-10

ndofs(C_b), maximum(abs, [u_exact(x_gh[1]); u_b] - u_exact.(x_gh))
```

Give each facet's condition once. The same label given both ways, `u => g` in `constrained` and `dirichlet` in `assemble`, is an `ArgumentError`, since the value would be imposed twice. A ghosted facet that carries only `u => g` is refused too, since its ghosts would be unknowns no equation reads: give it a ghost relation as well, or drop it from `ghosts`.


### Data per facet

A scalar datum on `:boundary` gives one value per node, but a corner node lies on two facets. When the data differ from facet to facet, give each facet its own `inner_Γ` term or its own relation. A facet's term carries that facet's weight, a half cell at the corner.

With the same function on every facet, one term on `:boundary` and one term per facet assemble to the same vector. With different data the corner entry is the sum of the facet terms, each at its own weight:

```@example ghost
q(p) = 1 + p[1] + 2p[2]
l_all = assemble(form(W_plain, v -> inner_Γ(q, v; markers = (:boundary,))))
l_each = assemble(form(W_plain, v -> sum(inner_Γ(q, v; markers = (m,)) for m in (:xmin, :xmax, :ymin, :ymax))))

data_x(p) = 10.0 + p[2]
data_y(p) = -3.0 + p[1]
l_corner = assemble(form(W_plain, v -> inner_Γ(data_x, v; markers = (:xmin,)) +
    inner_Γ(data_y, v; markers = (:ymin,))))

hx, hy = 0.2, 0.5
expected = (hy / 2) * data_x((0.0, 0.0)) + (hx / 2) * data_y((0.0, 0.0))
@assert l_all ≈ l_each
@assert l_corner[1] ≈ expected

l_corner[1], expected
```

---

## Solutions and time loops

### Solutions with their ghost values

A solution of `assemble(a, l)` is a vector of unknowns. `Bramble.lift(Cₕ, u)` is the grid function it describes, with the eliminated ghosts filled in by the relations. It lives on the padded space, one point longer at each ghosted end, so the runtime operators read its ghosts. `lift` is public and not exported, because it is also a Makie name: write `Bramble.lift`.

```@example ghost
U = Bramble.lift(C_gh, u_gh)

@assert length(U) == 35

points(space(U))[[1, 2, end - 1, end]]
```

The solution of route (b) lifts the same way, and the lifted function holds the Dirichlet value at its first node, `Bramble.lift(C_b, u_b)[1] ≈ u_exact(x_gh[1])`.

The relations hold on the lifted function. The runtime centred difference at the first and last nodes gives the data, and the collocation equation ``-\Delta_h u = f`` holds at every node, the two boundary nodes included:

```@example ghost
@assert -Dcₓ(U)[2] ≈ g_left(x_gh[1])
@assert robin_α * U[end - 1] + Dcₓ(U)[end - 1] ≈ g_right(x_gh[end])
@assert maximum(abs, [(-Δₕ(U))[i + 1] - f_src(x_gh[i]) for i in 1:33]) < 1e-6

(-Dcₓ(U)[2], g_left(x_gh[1])), (robin_α * U[end - 1] + Dcₓ(U)[end - 1], g_right(x_gh[end]))
```

`Bramble.lift(Cₕ)` alone is the data ``d``: the ghost values of the zero function. For the Neumann relation at the left, ``-(u_2 - u_0)/(2 h_2) = g_0`` with ``u_2 = 0`` gives ``u_0 = 2 h_2 g_0``, the first entry of `Bramble.lift(C_gh)`, and the other entries are zero. A relation on the second normal difference puts its data into the same ghost: ``(u_2 - 2u_1 + u_0)/h_2^2 = g`` gives ``u_0 = g h_2^2``.

`Bramble.lift(Cₕ)` and `Bramble.lift(Cₕ, u)` are both data in forms on `Cₕ`, so a linear form can name the full `Δₕ` of the lifted function at a facet node. A system whose ghost data are nonzero needs the matrix to build its load. Assembling the linear form alone is an `ArgumentError`, and the message names the fix:

```@example ghost
try
    assemble(l_gh)
catch e
    @assert e isa ArgumentError
    sprint(showerror, e)
end
```

### Ghost values in a time loop

An explicit time step assembles no system. It holds the solver's vector, refreshes the ghosts from it and applies an operator. [`GhostVector`](@ref) is that pair of the physical values and a buffer of ghosts. `GhostVector(Cₕ, u)` wraps the vector `u` of the physical values, in the mesh's linear order, without a copy. `Bramble.physical(v)` is `u` itself, and `Bramble.ghost_values(v)` is the buffer, which starts at zero. The space below is a [`constrained`](@ref) ghost space on the ``4 \times 3`` grid, with a Neumann datum on every facet.

```@example ghost
import Bramble: Δₕ!

C_g = constrained(W_ghost, u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => ∂ₙ(u) => 0.0,
                                 :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0))

u_g = [sinpi(p[1]) + p[2]^2 for p in vec([point(Ωₕ_g, I) for I in indices(Ωₕ_g)])]
v_g = GhostVector(C_g, u_g)

@assert Bramble.physical(v_g) === u_g
@assert length(v_g) == ndofs(W_ghost) == 26
@assert all(iszero, Bramble.ghost_values(v_g))

length(v_g), length(Bramble.ghost_values(v_g))
```

`update_ghosts!(v, Cₕ)` writes into the buffer the ghosts that the relations give, from the values of `v`. It reads the rows of the closure that belong to the ghosts and nothing else, so its cost follows the facets and it allocates nothing on `CpuSerial` and `CpuPolyester`. With a time, `update_ghosts!(v, Cₕ, t)` takes the data of the relations at `t`. The data here do not depend on time, so both calls fill the same ghosts.

A runtime operator on a `GhostVector` reads the ghosts from the buffer and returns the values at the physical nodes, a plain vector as long as `u`. It equals the same operator on `Bramble.lift(Cₕ, u)` at those nodes, with no padded function built:

```@example ghost
update_ghosts!(v_g, C_g)
ghosts_static = copy(Bramble.ghost_values(v_g))

fill!(Bramble.ghost_values(v_g), 0.0)
update_ghosts!(v_g, C_g, 0.7)

U_g = Bramble.lift(C_g, u_g)
phys = Bramble.physical_indices(mesh(C_g))
dims = npoints(mesh(Bramble.padded_space(C_g)), Tuple)
at_nodes(F) = reshape(collect(F), dims)[phys]

out_g = similar(u_g)
Δₕ!(out_g, v_g)

@assert Bramble.ghost_values(v_g) == ghosts_static && any(!iszero, ghosts_static)
@assert Bramble.physical(v_g) === u_g                  # the update left the physical part alone
@assert out_g ≈ vec(at_nodes(Δₕ(U_g)))
@assert Δₕ(v_g) ≈ out_g
@assert D₋ₓ(v_g) ≈ vec(at_nodes(D₋ₓ(U_g)))

extrema(out_g)
```

#### An explicit heat step

This is the explicit heat equation ``u_t = \Delta_h u`` with a Neumann relation, on a non-uniform grid. Each step refreshes the ghosts, applies `Δₕ!` and advances the physical vector in place. The unit square has ``11 \times 9`` nodes graded by ``x \mapsto x^{3/2}``, and the datum is zero on all four facets. The function ``e^{-2\pi^2 t}\cos \pi x \cos \pi y`` solves the continuous problem, so it is the reference. The step is ``0.2 / (h_x^{-2} + h_y^{-2})``, rounded down to divide the final time, with ``h_x`` and ``h_y`` the smallest spacings of each axis: inside the explicit stability limit ``1 / (2 (h_x^{-2} + h_y^{-2}))`` of the uniform grid with those spacings.

```@example ghost
Ωₕ_h = mesh(Ω_g, (11, 9), (true, true))
xs = [(i / 10)^1.5 for i in 0:10]
ys = [(j / 8)^1.5 for j in 0:8]
change_points!(Ωₕ_h, markers(Ω_g), (xs, ys))

Cₕ_h = constrained(gridspace(Ωₕ_h; ghosts = :boundary),
                   u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.0,
                         :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0))

u₀ = [cospi(p[1]) * cospi(p[2]) for p in vec([point(Ωₕ_h, I) for I in indices(Ωₕ_h)])]
u_h = copy(u₀)
v_h = GhostVector(Cₕ_h, u_h)
Lu = similar(u_h)

T = 0.05
Δt₀ = 0.2 / (1 / minimum(diff(xs))^2 + 1 / minimum(diff(ys))^2)
nsteps = ceil(Int, T / Δt₀)
Δt = T / nsteps

for _ in 1:nsteps
    update_ghosts!(v_h, Cₕ_h)
    Δₕ!(Lu, v_h)
    u_h .+= Δt .* Lu
end

u_exact_h = exp(-2π^2 * T) .* u₀
err = maximum(abs, u_h - u_exact_h)

@assert Δt <= Δt₀
@assert err < 2e-2 * maximum(abs, u₀)
@assert maximum(abs, u_h) < maximum(abs, u₀)            # the mode decayed
@assert Bramble.physical(v_h) === u_h                   # the loop wrote into the solver's vector

nsteps, err
```

The error is the spatial error of the scheme on this grid, about half a percent of the amplitude.
