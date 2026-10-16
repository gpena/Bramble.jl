```@meta
CurrentModule = Bramble
```

# [Grid spaces and discrete functions](@id tutorial_space)

**What you will learn.** How to put a function on a mesh, measure it, stack copies into a composite space with vector elements, and add ghost points beyond named facets.

**What you need first.** The [mesh tutorial](mesh.md), for building a mesh; the [geometry tutorial](geometry.md), for the domain it sits on.

**Where next.** [Difference operators](operators.md), which act on the elements built here.

A grid space is the discrete function space over a mesh. It fixes how many degrees of
freedom a field has and which quadrature weight each one carries. A
[`VectorElement`](@ref) is a function in that space. Every block below runs when this page
is built.

The running problem is small: sample $f(x) = \sin(\pi x)$ on the unit interval and measure
it, first as one field, then as a pair of fields.

## A grid space over a mesh

[`gridspace`](@ref) builds a grid space over a mesh. `ndofs` counts its degrees of freedom,
and with `Tuple` it gives the grid shape:

```@example space
using Bramble
import Bramble: component_range, component_ranges, weights

Ω = domain(interval(0.0, 1.0))
Ωₕ = mesh(Ω, 11)
Wₕ = gridspace(Ωₕ)

ndofs(Wₕ), ndofs(Wₕ, Tuple)
```

There is one degree of freedom per mesh point. `points(Wₕ)` is `points(Ωₕ)`, so the space
and its mesh agree on where the degrees of freedom sit.

## Discrete functions

[`Rₕ`](@ref) restricts a function to the grid: it evaluates it at every point and returns a
[`VectorElement`](@ref), the discrete function.

```@example space
uₕ = Rₕ(Wₕ, x -> sin(π * x))

length(uₕ), uₕ[6]
```

`VectorElement <: AbstractVector`, so indexing, `length` and broadcasting work as usual, and
broadcasting keeps the parent space:

```@example space
wₕ = 3.0 .* uₕ .+ element(Wₕ, 2.0)

space(wₕ) === Wₕ, wₕ[6]
```

[`element`](@ref) allocates a field, optionally filled with a constant. [`Rₕ!`](@ref)
writes into an element that already exists, which is what a time loop wants.

## Inner products and norms

Each degree of freedom carries a quadrature weight, the cell measure around its point.
`innerₕ` weights each point by it, and `normₕ` is the norm it induces:

```math
(u_h, v_h)_h = \sum_i |\square_i| \, u_h(x_i) v_h(x_i), \qquad
\|u_h\|_h = \sqrt{(u_h, u_h)_h}.
```

```@example space
vₕ = Rₕ(Wₕ, x -> cos(π * x))

innerₕ(uₕ, vₕ), normₕ(uₕ), normₕ(uₕ)^2 ≈ innerₕ(uₕ, uₕ)
```

The norm of $\sin(\pi x)$ on $[0, 1]$ is $1/\sqrt{2} \approx 0.707$, which `normₕ` reproduces here. The discrete Sobolev norms are built on the backward gradient
$\nabla_{-h}$:

```math
|u_h|_{1,h}^2 = \sum_{d=1}^D \|D_{-x_d} u_h\|_+^2, \qquad
\|u_h\|_{1,h}^2 = \|u_h\|_h^2 + |u_h|_{1,h}^2.
```

```@example space
snorm₁ₕ(uₕ), norm₁ₕ(uₕ), norm₁ₕ(uₕ)^2 ≈ normₕ(uₕ)^2 + snorm₁ₕ(uₕ)^2
```

!!! tip "Try this"
    Build the mesh as `mesh(Ω, 11, false)`, a non-uniform one with the same number of
    points, and rerun the blocks above. `ndofs` is unchanged, but the weights are no longer
    all equal, so `normₕ` and `innerₕ` change while still tiling the interval. The
    [mesh tutorial](mesh.md) builds such meshes on purpose.

## [Composite spaces and vector elements](@id space_composite)

A **composite space** stacks copies of a space for a vector field or a coupled system. Its
elements are `VectorElement`s too: one flat vector holding every component, one after
another. `Wₕ^Val(N)` stacks `N` copies:

```@example space
Vₕ = Wₕ^Val(2)

ncomponents(Vₕ), ndofs(Vₕ), ndofs(Vₕ, Tuple)
```

`ndofs(Vₕ, Tuple)` is one entry per component, not a grid shape. A **component** is one of
the stacked fields, and calling a composite element with `i` returns the `i`-th component
as a view:

```@example space
uvec = element(Vₕ)
uₓ = uvec(1)
uᵧ = uvec(2)

uₓ .= 1.5
uᵧ .= -2.0

parent(uvec)[[1, 12]]
```

The component is a zero-copy view, so writing to it writes through to the parent. For a
scalar space `uₕ(1)` is `uₕ` itself. [`components`](@ref) destructures the lot:

```@example space
c₁, c₂ = components(uvec)

c₁ === uₓ, c₂ === uᵧ
```

On a composite space `Rₕ!` takes a tuple of scalar functions, or one function returning a
tuple, and `innerₕ` sums over components:

```@example space
Rₕ!(uvec, (x -> sin(π * x), x -> cos(π * x)))

normₕ(uvec)^2 ≈ normₕ(uvec(1))^2 + normₕ(uvec(2))^2
```

!!! tip "Try this"
    Replace `Val(2)` with `Val(3)` and add a third function to the tuple. `ncomponents` and
    `ndofs` follow, and every other line of this section stays the same.

## [Ghost spaces](@id space_ghost)

A grid space can carry the ghost points of the [mesh tutorial](@ref mesh_ghost). The keyword `ghosts` of [`gridspace`](@ref) names the facets that get them: `:boundary` for every facet, or a tuple such as `(:xmin, :ymax)`. The default `ghosts = ()` returns the space of the sections above, unchanged. The running example is a small non-uniform grid in the unit square:

```@example space
import Bramble: change_points!, indices, point
using LinearAlgebra: diag

Ω_g = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ_g = mesh(Ω_g, (4, 3), (true, true))
change_points!(Ωₕ_g, markers(Ω_g), ([0.0, 0.2, 0.6, 1.0], [0.0, 0.5, 1.0]))

W_plain = gridspace(Ωₕ_g)
W_ghost = gridspace(Ωₕ_g; ghosts = :boundary)
W_part = gridspace(Ωₕ_g; ghosts = (:xmin, :ymax))

@assert typeof(gridspace(Ωₕ_g; ghosts = ())) == typeof(W_plain)

ndofs(W_plain), ndofs(W_ghost), ndofs(W_part)
```

The degrees of freedom of a ghost space are the physical nodes plus the ghosts. Here ``12`` nodes plus ``2 \cdot 3 + 2 \cdot 4 = 14`` ghosts give 26, and the two named facets give ``12 + 3 + 4 = 19``. The padded corners are not degrees of freedom.

### The order of the degrees of freedom

The order is fixed. The physical nodes come first, in the mesh's linear order. The ghosts follow, facet by facet in the order `xmin, xmax, ymin, ymax, zmin, zmax` (only the named ones), each facet's ghosts in the column-major order of its nodes. A system assembled on a ghost space is indexed this way, so its first rows and columns are the physical nodes.

Each degree of freedom is one point of the padded mesh. `Bramble.padded_space` is the internal space on that mesh, and `Bramble.closure` returns the sparse matrix `E` that sends the degrees of freedom to padded values, one entry ``1`` per column. This reads off where each one sits:

```@example space
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

The first two entries are the first and last physical node. The next four are the `xmin` and `xmax` ghosts: ``x = -0.2`` and ``x = 1.4``, the distances ``h_2 = 0.2`` and ``h_n = 0.4`` from the facet. The last four are the `ymin` and `ymax` ghosts at ``y = -0.5`` and ``y = 1.5``. The spacings differ at the two ends of each axis, so a ghost is not a mirror image: its coordinate is read from the mesh.

### Physical-only weights

A ghost carries no quadrature weight. The inner products `innerₕ`, `inner₊` and `inner_Γ` weigh the physical nodes, cells and facets exactly as on the space without ghosts, and give zero weight to every ghost. A form is assembled on the ghost space, so its mass matrix shows this:

```@example space
M_plain = assemble(form(W_plain, W_plain, (u, v) -> innerₕ(u, v)))
M_ghost = assemble(form(W_ghost, W_ghost, (u, v) -> innerₕ(u, v)))

@assert diag(M_ghost)[1:12] == diag(M_plain)
@assert all(iszero, diag(M_ghost)[13:end])
@assert sum(diag(M_ghost)) ≈ 1.0       # the physical cells tile the unit square

sum(diag(M_ghost)[1:12]), sum(diag(M_ghost)[13:end])
```

The weights of `inner_Γ` are facet weights: the transverse measure at each facet node, a half cell at the facet's ends. The markers `:xmin` to `:zmax` name the physical facet nodes, never the ghosts. `gridspace` of a bare `GhostMesh` is refused, since its weights would count the ghosts: build the space from the physical mesh with `ghosts`.

### What to do with a ghost space

A ghost space is a space of unknowns for forms. [`form`](@ref) builds on it as on any space, and [`constrained`](@ref) ties its ghosts to boundary relations, which the [form tutorial](@ref form_ghost) shows. The solution vector comes back in the order above, and `Bramble.lift` turns it into a grid function with its ghosts filled in. That grid function lives on the padded space, one point longer at each ghosted end.

### Ghost values in a time loop

An explicit time step assembles no system. It holds the solver's vector, refreshes the ghosts from it and applies an operator. [`GhostVector`](@ref) is that pair of the physical values and a buffer of ghosts. `GhostVector(Cₕ, u)` wraps the vector `u` of the physical values, in the mesh's linear order, without a copy. `Bramble.physical(v)` is `u` itself, and `Bramble.ghost_values(v)` is the buffer, which starts at zero. The space below is a [`constrained`](@ref) ghost space on the grid of this section, with a Neumann datum on every facet.

```@example space
import Bramble: Δₕ!, D₋ₓ

Cₕ_g = constrained(W_ghost, u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => ∂ₙ(u) => 0.0,
                                  :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0))

u_g = [sinpi(p[1]) + p[2]^2 for p in vec([point(Ωₕ_g, I) for I in indices(Ωₕ_g)])]
v_g = GhostVector(Cₕ_g, u_g)

@assert Bramble.physical(v_g) === u_g
@assert length(v_g) == ndofs(W_ghost) == 26
@assert all(iszero, Bramble.ghost_values(v_g))

length(v_g), length(Bramble.ghost_values(v_g))
```

`update_ghosts!(v, Cₕ)` writes into the buffer the ghosts that the relations give, from the values of `v`. It reads the rows of the closure that belong to the ghosts and nothing else, so its cost follows the facets, not the mesh, and it allocates nothing. With a time, `update_ghosts!(v, Cₕ, t)` takes the data of the relations at `t`. The data here do not depend on time, so the time is ignored and both calls fill the same ghosts. They are the ghosts of `Bramble.lift`, which the next block compares through an operator:

```@example space
update_ghosts!(v_g, Cₕ_g)
ghosts_static = copy(Bramble.ghost_values(v_g))

fill!(Bramble.ghost_values(v_g), 0.0)
update_ghosts!(v_g, Cₕ_g, 0.7)

@assert Bramble.ghost_values(v_g) == ghosts_static
@assert any(!iszero, ghosts_static)
@assert Bramble.physical(v_g) === u_g                  # the update left the physical part alone
@assert Δₕ!(similar(u_g), v_g) ≈ Δₕ!(similar(u_g), update_ghosts!(GhostVector(Cₕ_g, u_g), Cₕ_g))

ghosts_static[1:3]
```

A runtime operator on a `GhostVector` reads the ghosts from the buffer and returns the values at the physical nodes, a plain vector. It equals the same operator on `Bramble.lift(Cₕ, u)` at those nodes, with no padded function built:

```@example space
U_g = Bramble.lift(Cₕ_g, u_g)
phys = Bramble.physical_indices(mesh(Cₕ_g))
dims = npoints(mesh(Bramble.padded_space(Cₕ_g)), Tuple)
at_nodes(F) = reshape(collect(F), dims)[phys]

out_g = similar(u_g)
Δₕ!(out_g, v_g)

@assert out_g ≈ vec(at_nodes(Δₕ(U_g)))
@assert Δₕ(v_g) ≈ out_g
@assert D₋ₓ(v_g) ≈ vec(at_nodes(D₋ₓ(U_g)))

extrema(out_g)
```

Two things follow from the physical-only result. The result carries no ghosts: it is as long as `u`, not as `v`. And the nested mixed differences of the [operators tutorial](operators.md), which a `lift` function composes, stay forms: a `GhostVector` takes one operator at a time.

#### An explicit heat step

The use this serves is the explicit heat equation ``u_t = \Delta_h u`` with a Neumann relation, on a non-uniform grid. Each step refreshes the ghosts, applies `Δₕ!` and advances the physical vector in place. The unit square has ``11 \times 9`` nodes graded by ``x \mapsto x^{3/2}``, and the datum is zero on all four facets. The function ``e^{-2\pi^2 t}\cos \pi x \cos \pi y`` solves the continuous problem, so it is the reference. The step is ``0.2 / (h_x^{-2} + h_y^{-2})``, rounded down to divide the final time, with ``h_x`` and ``h_y`` the smallest spacings of each axis: inside the explicit stability limit ``1 / (2 (h_x^{-2} + h_y^{-2}))`` of the uniform grid with those spacings.

```@example space
Ω_h = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ_h = mesh(Ω_h, (11, 9), (true, true))
xs = [(i / 10)^1.5 for i in 0:10]
ys = [(j / 8)^1.5 for j in 0:8]
change_points!(Ωₕ_h, markers(Ω_h), (xs, ys))

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

u_exact = exp(-2π^2 * T) .* u₀
err = maximum(abs, u_h - u_exact)

@assert Δt <= Δt₀
@assert err < 2e-2 * maximum(abs, u₀)
@assert maximum(abs, u_h) < maximum(abs, u₀)            # the mode decayed
@assert Bramble.physical(v_h) === u_h                   # the loop wrote into the solver's vector

nsteps, err
```

The error is the spatial error of the scheme on this grid, about half a percent of the amplitude.

## Reference

### Constructors

`Wₕ^Val(N)` is type stable whatever `N` is, while `Wₕ^2` relies on the literal being
constant-folded and only reaches `N ≤ 3`. [`vector_gridspace`](@ref)`(Ωₕ, 2)` builds the same
space straight from a mesh, and `Bramble.CompositeGridSpace((Wₕ, Wₕ))` builds one from spaces that
need not be identical:

```@example space
spaces(Vₕ) == (Wₕ, Wₕ)
```

### Weights

`weights(Wₕ)` returns the whole `SpaceWeights` bundle, and a weight vector comes from naming
the inner product it belongs to. `Bramble.Innerh` and `Bramble.Innerplus` are internal, hence the
prefix. `weights` is not defined for a composite space, since its leaves
may live on different meshes: call it on a [`components`](@ref) leaf instead.

```@example space
w = weights(Wₕ, Bramble.Innerh())

length(w), sum(w)      # one weight per point; they tile the unit interval
```

### Component ranges

[`component_range`](@ref) and `component_ranges` give the degrees of freedom each component
owns:

```@example space
component_range(Vₕ, 1), component_ranges(Vₕ)
```

### Functions as factors

A plain `Function` is not a grid function, so multiplying by one restricts it first
(`Rₕ(space(uₕ), f)`) and scales pointwise. This is what lets a spatial condition multiply a
field directly, including inside a form:

```@example space
oneₕ = Rₕ(Wₕ, x -> 1.0)
below_half = (x -> x[1] < 0.5) * oneₕ

sum(below_half)
```

### Grid layout

Degrees of freedom are stored flat, but on a two-dimensional space a scalar element also
indexes by grid coordinate, with a `CartesianIndex`, without reshaping:

```@example space
W₂ = gridspace(mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (5, 5), (true, true)))
V₂ = W₂^Val(2)
u_scal = element(W₂, 0.0)
u_scal[CartesianIndex(2, 3)] = 10.0
u_scal[CartesianIndex(4, 1)] = 20.0

u_scal[CartesianIndex(2, 3)], u_scal[CartesianIndex(4, 1)]
```

For matrix operations or plotting, `reshape(uₕ)` returns a `Base.ReshapedArray` view of the
same memory, so writing through it writes through to the element. On a composite element it
returns one view per component:

```@example space
u_grid = reshape(u_scal)
u_grid[2, 3] = -1.0

size(u_grid), u_scal[CartesianIndex(2, 3)], size.(reshape(element(V₂)))
```
```@raw html
<figure>
<svg viewBox="0 0 780 270" width="100%" style="max-width:780px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Diagram of CompositeGridSpace storage showing the contiguous flat degree of freedom buffer, zero-copy component views u_x and u_y, and 2D reshaped matrix views.">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="currentColor"/>
    </marker>
    <marker id="arrowBlue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#3b82f6"/>
    </marker>
    <marker id="arrowPurple" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#8b5cf6"/>
    </marker>
  </defs>

  <!-- 1. Flat 1D Buffer -->
  <text x="30" y="25" font-size="13" font-weight="bold" fill="currentColor">Flat 1D degree of freedom storage: uₕ.data (length 50)</text>

  <!-- Component 1 block -->
  <rect x="30" y="35" width="350" height="36" rx="4" fill="#3b82f6" fill-opacity="0.15" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="205" y="58" font-size="12" font-weight="bold" fill="#3b82f6" text-anchor="middle">Component 1 (uₓ): DOFs 1:25</text>

  <!-- Component 2 block -->
  <rect x="390" y="35" width="350" height="36" rx="4" fill="#8b5cf6" fill-opacity="0.15" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="565" y="58" font-size="12" font-weight="bold" fill="#8b5cf6" text-anchor="middle">Component 2 (uᵧ): DOFs 26:50</text>

  <!-- Connectors from Flat to Component Views -->
  <path d="M 205 71 L 205 105" stroke="#3b82f6" stroke-width="1.5" fill="none" marker-end="url(#arrowBlue)"/>
  <text x="215" y="93" font-size="11" fill="#3b82f6">uₕ(1)</text>

  <path d="M 565 71 L 565 105" stroke="#8b5cf6" stroke-width="1.5" fill="none" marker-end="url(#arrowPurple)"/>
  <text x="575" y="93" font-size="11" fill="#8b5cf6">uₕ(2)</text>

  <!-- 2. Zero-copy component views -->
  <rect x="30" y="110" width="350" height="34" rx="4" fill="#3b82f6" fill-opacity="0.08" stroke="#3b82f6" stroke-dasharray="4,3" stroke-width="1.2"/>
  <text x="205" y="132" font-size="12" fill="currentColor" text-anchor="middle">VectorElement view (SubArray of length 25)</text>

  <rect x="390" y="110" width="350" height="34" rx="4" fill="#8b5cf6" fill-opacity="0.08" stroke="#8b5cf6" stroke-dasharray="4,3" stroke-width="1.2"/>
  <text x="565" y="132" font-size="12" fill="currentColor" text-anchor="middle">VectorElement view (SubArray of length 25)</text>

  <!-- Connectors from Component Views to 2D Grids -->
  <path d="M 205 144 L 205 178" stroke="currentColor" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
  <text x="215" y="166" font-size="11" fill="currentColor">reshape(uₓ)</text>

  <path d="M 565 144 L 565 178" stroke="currentColor" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>
  <text x="575" y="166" font-size="11" fill="currentColor">reshape(uᵧ)</text>

  <!-- 3. Reshaped 2D Matrix Views -->
  <g transform="translate(145, 185)">
    <rect x="0" y="0" width="120" height="65" rx="4" fill="none" stroke="#3b82f6" stroke-width="1.5"/>
    <line x1="24" y1="0" x2="24" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="48" y1="0" x2="48" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="72" y1="0" x2="72" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="96" y1="0" x2="96" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="13" x2="120" y2="13" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="26" x2="120" y2="26" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="39" x2="120" y2="39" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="52" x2="120" y2="52" stroke="currentColor" stroke-opacity="0.2"/>
    <text x="60" y="80" font-size="11" fill="currentColor" text-anchor="middle">5 × 5 ReshapedArray (uₓ[i, j])</text>
  </g>

  <g transform="translate(505, 185)">
    <rect x="0" y="0" width="120" height="65" rx="4" fill="none" stroke="#8b5cf6" stroke-width="1.5"/>
    <line x1="24" y1="0" x2="24" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="48" y1="0" x2="48" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="72" y1="0" x2="72" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="96" y1="0" x2="96" y2="65" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="13" x2="120" y2="13" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="26" x2="120" y2="26" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="39" x2="120" y2="39" stroke="currentColor" stroke-opacity="0.2"/>
    <line x1="0" y1="52" x2="120" y2="52" stroke="currentColor" stroke-opacity="0.2"/>
    <text x="60" y="80" font-size="11" fill="currentColor" text-anchor="middle">5 × 5 ReshapedArray (uᵧ[i, j])</text>
  </g>
</svg>
</figure>
```

### Restriction and cell averaging

Two ways to turn a continuous function into a grid function. Nodal restriction $R_h$
evaluates it at the grid points; cell averaging $\mathrm{avg}_h$ integrates it over the cell
around each point and divides by the cell measure:

```math
\mathrm{avg}_h f(x_i) = \frac{1}{|\square_i|} \int_{\square_i} f(x) \, dx
```
```@raw html
<figure>
<svg viewBox="0 0 780 275" width="100%" style="max-width:780px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Comparison of nodal restriction R_h which samples f at point x_i versus cell averaging avg_h which integrates f over the dual cell using a 6-point Gauss-Legendre quadrature rule.">
  <!-- Panel 1: Nodal restriction R_h -->
  <g transform="translate(20, 10)">
    <rect x="0" y="0" width="355" height="255" rx="6" fill="none" stroke="currentColor" stroke-opacity="0.2" stroke-width="1"/>
    <text x="177" y="28" font-size="14" font-weight="bold" fill="currentColor" text-anchor="middle">Nodal restriction: Rₕ(Wₕ, f)</text>
    <text x="177" y="48" font-size="12" fill="currentColor" opacity="0.85" text-anchor="middle">Pointwise evaluation: Rₕ f(xᵢ) = f(xᵢ)</text>

    <!-- Function curve -->
    <path d="M 40 145 Q 177 75 315 115" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="300" y="105" font-size="11" fill="currentColor" font-style="italic">f(x)</text>

    <!-- Grid line & cell -->
    <line x1="30" y1="190" x2="325" y2="190" stroke="currentColor" stroke-width="1.5"/>
    <!-- Cell bounds -->
    <line x1="90" y1="175" x2="90" y2="205" stroke="#8b5cf6" stroke-width="1.5" stroke-dasharray="4,3"/>
    <line x1="265" y1="175" x2="265" y2="205" stroke="#8b5cf6" stroke-width="1.5" stroke-dasharray="4,3"/>
    <text x="90" y="222" font-size="11" fill="#8b5cf6" text-anchor="middle">xᵢ₋½</text>
    <text x="265" y="222" font-size="11" fill="#8b5cf6" text-anchor="middle">xᵢ₊½</text>

    <!-- Center point x_i -->
    <circle cx="177" cy="190" r="5" fill="#3b82f6"/>
    <text x="177" y="222" font-size="12" font-weight="bold" fill="#3b82f6" text-anchor="middle">xᵢ</text>

    <!-- Pointwise sample -->
    <line x1="177" y1="185" x2="177" y2="100" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="2,2"/>
    <circle cx="177" cy="96" r="5" fill="#3b82f6" stroke="currentColor" stroke-width="1"/>
    <text x="187" y="92" font-size="11" font-weight="bold" fill="#3b82f6">f(xᵢ)</text>

    <text x="177" y="246" font-size="11" fill="currentColor" opacity="0.8" text-anchor="middle">Single evaluation at node xᵢ</text>
  </g>

  <!-- Panel 2: Cell averaging avg_h -->
  <g transform="translate(405, 10)">
    <rect x="0" y="0" width="355" height="255" rx="6" fill="none" stroke="currentColor" stroke-opacity="0.2" stroke-width="1"/>
    <text x="177" y="28" font-size="14" font-weight="bold" fill="currentColor" text-anchor="middle">Cell averaging: avgₕ(Wₕ, f)</text>
    <text x="177" y="48" font-size="12" fill="currentColor" opacity="0.85" text-anchor="middle">Integral mean: avgₕ f(xᵢ) = |□ᵢ|⁻¹ ∫_{□ᵢ} f(x) dx</text>

    <!-- Shaded area under curve across the cell -->
    <path d="M 90 190 L 90 134 Q 177 75 265 106 L 265 190 Z" fill="#10b981" fill-opacity="0.15"/>

    <!-- Function curve -->
    <path d="M 40 145 Q 177 75 315 115" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="3,3"/>
    <text x="300" y="105" font-size="11" fill="currentColor" font-style="italic">f(x)</text>

    <!-- Grid line & cell -->
    <line x1="30" y1="190" x2="325" y2="190" stroke="currentColor" stroke-width="1.5"/>
    <!-- Cell bounds -->
    <line x1="90" y1="175" x2="90" y2="205" stroke="#8b5cf6" stroke-width="1.5" stroke-dasharray="4,3"/>
    <line x1="265" y1="175" x2="265" y2="205" stroke="#8b5cf6" stroke-width="1.5" stroke-dasharray="4,3"/>
    <text x="90" y="222" font-size="11" fill="#8b5cf6" text-anchor="middle">xᵢ₋½</text>
    <text x="265" y="222" font-size="11" fill="#8b5cf6" text-anchor="middle">xᵢ₊½</text>

    <!-- Center point x_i -->
    <circle cx="177" cy="190" r="4" fill="currentColor"/>
    <text x="177" y="222" font-size="12" fill="currentColor" text-anchor="middle">xᵢ</text>

    <!-- Gauss quadrature points inside cell (6 points) -->
    <circle cx="96"  cy="190" r="3" fill="#10b981"/>
    <circle cx="120" cy="190" r="3" fill="#10b981"/>
    <circle cx="156" cy="190" r="3" fill="#10b981"/>
    <circle cx="198" cy="190" r="3" fill="#10b981"/>
    <circle cx="235" cy="190" r="3" fill="#10b981"/>
    <circle cx="259" cy="190" r="3" fill="#10b981"/>

    <path d="M 120 185 L 120 128" stroke="#10b981" stroke-width="1" stroke-dasharray="2,2"/>
    <path d="M 198 185 L 198 87"  stroke="#10b981" stroke-width="1" stroke-dasharray="2,2"/>
    <path d="M 235 185 L 235 98"  stroke="#10b981" stroke-width="1" stroke-dasharray="2,2"/>

    <text x="177" y="246" font-size="11" fill="#10b981" font-weight="bold" text-anchor="middle">6-point Gauss-Legendre quadrature (N_q = 6)</text>
  </g>
</svg>
</figure>
```

[`Rₕ`](@ref) allocates and [`Rₕ!`](@ref) writes in place. [`avgₕ`](@ref) and
[`avgₕ!`](@ref) mirror them, using a tensor-product Gauss-Legendre rule
(`AVG_QUAD_POINTS = 6`, exact through degree eleven):

```@example space
u_avg = avgₕ(W₂, x -> exp(-x[1] - x[2]))
avgₕ!(u_avg, x -> exp(-x[1] - x[2]))

u_avg[1]
```

The difference matters for a source term: `Rₕ` of a rough function inherits its roughness,
while `avgₕ` integrates it away.

### Staggered inner product

`inner₊` is the staggered counterpart of `innerₕ`: it weights by the half-spacings, the
interface quantities a difference or a gradient lands on, and a trailing `:x`/`:y`/`:z` (or
`1`/`2`/`3`) argument picks a single direction. That pairing is what makes summation by parts
exact; the [operators tutorial](operators.md) derives it.

```@example space
D = ∇ₕ(uₕ)

inner₊(D, D, :x) ≈ snorm₁ₕ(uₕ)^2
```
