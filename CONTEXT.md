# Bramble

Finite-difference discretisation of PDEs on Cartesian meshes: build a mesh over a domain, put
a grid space on it, write a form with difference operators, assemble it into a matrix. Below
is the code's vocabulary; terms under _Avoid_ mean something else here, or nothing.

## Language

### Geometry

**Set**:
A geometric region, a `CartesianProduct` of intervals. Pure geometry: no discretisation, no
names.
_Avoid_: region, area, box (`box` is the 3D constructor, not the concept)

**Domain**:
A set with its markers; what a mesh is built from.
_Avoid_: geometry, region

**Marker**:
A named subset of a domain, `label => face` or `label => predicate`: the pairing of a `Symbol`
name with the rule selecting points.
_Avoid_: tag, region, boundary condition (a marker names *where*, never *what value*)

**Label**:
The `Symbol` half of a marker (`:left`, `:boundary`). Label is the name, marker the
name-plus-rule.

**Level set**:
A domain given by a function `φ` with `φ < 0` inside, marked `label => levelset(φ)`
(`levelset(φ)` is a `LevelSet`, never a bare `Function`, which would be a predicate). The
nodes stay the user's grid; the labels are `:label`, `:label_boundary` and `:label_exterior`.
Owned by [docs/notes/internals/levelset.md](docs/notes/internals/levelset.md).
_Avoid_: implicit surface, signed distance (`φ` need not be one)

**Crossing**:
The point where `φ = 0` meets a grid edge whose end nodes differ in sign, found by a
bracketed root solve on that edge. The crossing carries the Dirichlet datum.
_Avoid_: intersection, cut point

**Shortened arm**:
The arm of a node's stencil along an axis that a crossing cuts, of length `θh` with
`0 < θ ≤ 1` instead of `h`. The reduction gives each reading row its own value at the
exterior node (the quadratic through the crossing), which on `-Δₕ` is the Shortley-Weller row.
_Avoid_: cut stencil, short stencil

### Meshes

**Mesh**:
A domain discretised into points, uniform or randomly perturbed: `Mesh1D`, or `MeshnD` above
one dimension.
_Avoid_: grid alone (only as an adjective: *grid space*, *grid function*), triangulation,
cells

**Submesh**:
The 1D mesh along one axis of an `nD` mesh; every `MeshnD` is a tuple of them.
_Avoid_: slice, axis mesh

**Reserved markers**:
`:boundary` and `:interior`, which every mesh computes from its geometry whether or not the
domain names them. A domain's own definition wins.

**Refinement**:
Dyadic halving in place (`iterative_refinement!`): the refined mesh is the same mesh split,
not a new draw, which makes a convergence order measurable on a random mesh.

**Ghost mesh**:
A physical mesh `Ωₕ` plus one ghost point beyond each node of the facets it names,
built by `Bramble.GhostMesh(Ωₕ; ghosts = :boundary)` or `ghosts = (:xmin, :ymax)` for named facets.
Its points, spacings and indices are those of the padded mesh; its labels and facet queries
name the physical nodes. After `iterative_refinement!` of `Ωₕ` it is stale and refuses every
access. Owned by [docs/src/internals/mesh.md](docs/src/internals/mesh.md).
_Avoid_: halo, boundary layer

**Padded mesh**:
The mesh of a ghost mesh's points: `Ωₕ` extended by one point on each ghosted side of each
axis, with sizes `npoints(G, Tuple)`. Forms on a ghost space are assembled on the space over
it, then reduced (see **Closure**).

**Facet ghost**:
The padded point beyond a node of a ghosted facet, at `x₁ - h₂` beyond the first node and
`xₙ + hₙ` beyond the last. Facet ghosts are numbered `1:nghosts(G)`: facets in the order
`xmin, xmax, ymin, ymax, zmin, zmax` (named ones only), each facet's ghosts in the
column-major order of its nodes; `Bramble.ghost_index` and `Bramble.boundary_node` map between
a facet node and its ghost.

**Absent point**:
A padded point outside the physical mesh along two or more axes (a 2D corner, a 3D edge or
corner of the padded grid): no node and no ghost, never a degree of freedom, with value 0 in
a lifted grid function. `Bramble.is_absent(G, J)` tests it.

### Spaces and grid functions

**Grid space**:
The discrete function space over a mesh, with its quadrature weights: `ScalarGridSpace` for
one field, `CompositeGridSpace` for several coupled.
_Avoid_: function space, discrete space, FE space

**Vector element**:
A function in a grid space, the discrete unknown `uₕ`. A flat vector is its storage, not the
concept.
_Avoid_: solution vector, DOF vector, array

**Grid function**:
Accepted synonym for vector element; the docstrings' word in mathematical prose.

**Leaf space**:
One scalar component of a composite space, numbered depth-first (velocity and pressure in
Stokes are two leaves). Assembly and the Dirichlet path address leaves through
`leaf_spaces_offsets`.
_Avoid_: component (the `components` keyword *selects* leaves), field, block

`u(i)`, `components(u)` and `component_range` number leaves the same way, so on a nested
`CompositeGridSpace((W × W, W))` `u(2)` is the second leaf, inside the first child, not the
second child. (`×` flattens: `W × W × W` is one level of three leaves.)

**Backend**:
Where a space's arrays live and how they are iterated, with the execution policy
(`CpuSerial()`/`Serial()`, `CpuThreaded()`/`Parallel()`, `CpuPolyester()`, `GpuKernel()`, ...)
as a trait.
_Avoid_: device, mode

**DOF order**:
The order of the degrees of freedom of a ghost space: the physical nodes in the mesh's linear
order (those no Dirichlet relation fixes), then the ghosts that remain unknowns, in the
numbering of the ghost mesh. The first entries are therefore the values at the physical
nodes, as on `gridspace(Ωₕ)`. Owned by
[docs/src/internals/space.md](docs/src/internals/space.md).

**Physical-only weights**:
The quadrature weights of the padded space `Wₕ`: per axis, the factor of a ghost node and of
a ghost cell is zero and a facet node keeps its physical half cell. `innerₕ`, `inner₊`,
`inner_Γ` and the facet markers then have the weights of `gridspace(Ωₕ)` at the physical
nodes and exactly zero at every ghost and absent point, so a form on the padded space has no
weight in its ghost rows.

**Closure**:
The pair `(E, d)` of a ghost space, `Bramble.closure(Wₕ)`: its unknowns `u` stand for the
padded values `E u + d`, `E` sparse of size `npoints(mesh) × ndofs`, `d` the data at the
eliminated ghosts. `Bramble.padded_space(Wₕ)` is the space those values live on. Assembly
fills `A_pad` on the padded space and returns `Eᵀ A_pad E`; a load is `Eᵀ (F_pad - A_pad d)`,
so a space with `d ≠ 0` needs the pair `assemble(a, l)`. On `gridspace(Ωₕ; ghosts)` `E`
selects the unknowns and `d = 0`. A composite of ghost spaces whose components ghost the same
facets has the composite of the padded spaces as its padded space and
`(blockdiag(E₁, E₂, …), vcat(d₁, d₂, …))` as its closure. If the data depend on time, `E`
stays and `d(t)` varies.

**Relation**:
A boundary condition of a ghost space written `facet => operator => data`, as in
`:xmin => α * u + β * ∂ₙ(u) => g`, with `operator` either `u` (a Dirichlet relation, whose
facet nodes leave the unknowns and take `g`) or a facet operator reading the facet's ghost. At each node
of the facet it is solved for that node's ghost; where two conormal relations read each
other's ghost at a 2D corner, the 2×2 block is solved together. Its rows are assembled by
`inner_Γ(relation, v; markers = (F,))`, so the facet's weight cancels. The `data` is a
number, `x -> …` or `(x, t) -> …`.
_Avoid_: boundary condition (a relation pairs an operator with data; a marker names *where*)

**Constrained space**:
A ghost space whose ghosts are tied to relations by `constrained(Wₕ, u -> (:xmin => ∂ₙ(u) =>
g, …))` on `gridspace(Ωₕ; ghosts)`. Its closure `(E, d)` comes from the relations, `E` and
`d` stay sparse, and its unknowns are the physical nodes no Dirichlet relation fixes plus
the ghosts left unrelated. Refused with an `ArgumentError`: a space without ghosts, an
already constrained one, a facet with two relations of one kind, a ghosted facet with only
`u => g`. See [docs/src/tutorials/form.md](docs/src/tutorials/form.md) (the ghost sections).

**Lift**:
`Bramble.lift(Wₕ, u)` is `E u + d`: the grid function on the padded space of the unknowns
`u`, with the eliminated ghosts filled and absent points 0. `lift(Wₕ)` is `d`;
`lift(Wₕ, u, t)` is `E u + d(t)` where the data depend on time, the only form such a space
accepts. `lift!` fills one in place. `public`, not exported.

### Operators

**Difference operator**:
A discrete derivative: `D₋ₓ` backward, `Dcₓ` centred, the subscript naming the direction. A
direction in a variable indexes the vectorial operator: `∇ₕ[1] === ∇ₕ[:x] === D₋ₓ`. The
`public`, unexported stem takes it as an argument: `Bramble.D₋(uₕ, 1)`, `(uₕ, :x)` and
`(uₕ, Val(1))` all equal `D₋ₓ(uₕ)`. Averages pass it to `Mₕ`/`M₊ₕ`; there is no bare `M`.
_Avoid_: derivative (the continuous object), gradient (that is `∇ₕ`)

**Restriction (`Rₕ`)**:
Projection of a continuous function onto grid functions by evaluating it at mesh points.
_Avoid_: interpolation (that is `πₕ`), sampling

**Cell average (`avgₕ`)**:
The same projection by quadrature over each cell instead: same target space, different rule,
and the expensive one (six quadrature nodes per point).
_Avoid_: restriction (name the rule, since both project)

**Stencil**:
The neighbouring points one operator application reads, with weights; `local_stencil` returns
it for an index.
_Avoid_: footprint, pattern (a *pattern* is a matrix's sparsity)

**`innerₕ` and `inner₊`**:
`innerₕ` is the discrete ``L^2`` inner product, each point weighted by its cell measure;
`inner₊` the *modified* ``L^2_+`` product used with backward differences. Distinct objects,
not spellings.

**Facet operator**:
`∂ₙ` (conormal `∂ₙ(u; K)` included), `∂ₙₙ` and `jumpₙ`, operators on the facets `F`
of outward unit normal `η_F` (`η`) that live only inside `inner_Γ`. Each
holds one operator per (axis, side) and expands into one product per facet its `markers`
select, so a corner node gets one value per facet. `∂ₙ` and `∂ₙₙ` read the ghost beyond the
facet and need a space with ghosts there; `jumpₙ` reads the facet node and its interior
neighbour and needs none.

**Corner rule**:
The stencil of a mixed difference (a first difference of first differences across two axes)
at a 2D corner node, or a 3D edge node, of a ghost mesh, where it would read an absent
point. It uses the inward one-sided difference along both axes, `D₊` at a first node and `D₋` at a
last, reading physical nodes only, second order. It depends on the form, not on
the values of the coefficients. Any other read of an absent point is refused with an
`ArgumentError`. See [docs/src/internals/space.md](docs/src/internals/space.md).

### Forms and assembly

**Form**:
A symbolic expression in trial and test functions (`LinearForm` in one, `BilinearForm` in
two). Structure, not values.
_Avoid_: weak form, variational form, integrand

**Trial and test function**:
The two arguments of a bilinear form. The test function indexes matrix **rows**, the trial
function columns: load-bearing and easy to write backwards.

**Source**:
A term carrying known data, not an unknown, so it can appear in a linear form.

**AST**:
The resolved operator expression a form compiles to (`resolve_form_ast`); what assembly
walks.
_Avoid_: expression tree, symbolic form, IR

**Pattern**:
A system matrix's sparsity: which entries can be non-zero, fixed by the stencil and invariant
while mesh and expression are unchanged.
_Avoid_: stencil, structure

**Separable form**:
A bilinear form on a `MeshnD` whose matrix is a sum of Kronecker products of one-dimensional
factors, one per axis (`is_separable`).
_Avoid_: tensor-product form, factorisable

**Laplacian-like form**:
A separable form with one mass per axis, every term differing from those masses on at most
one axis: what `fdm_solve` solves directly. A mixed derivative is separable but not
Laplacian-like.
_Avoid_: Poisson-like, elliptic

**Assembly**:
Filling a matrix or vector from a form. `assemble` allocates and fills; `assemble!` refills an
existing one, the time-loop call, and must not allocate.

**Block**:
One leaf-space pair's rectangle in a composite system matrix: row offset from the test leaf,
column offset from the trial leaf.

**Mean-zero**:
`pde_solve(A, F; mean_zero = w)`: the bordered solve `[A w; wᵀ 0] [u; λ] = [F; 0]`, which
returns the `u` with `wᵀ u = 0` for a pure-Neumann `A` annihilating constants, `w` typically
the assembled `innerₕ(1, v)`. `A` is not modified; without `mean_zero` such an `A` only logs a
warning.

**Rank-one**:
A term `σ innerₕ(g, u) innerₕ(h, v)` (`g`, `h` data) of a bilinear form, not stored in the
sparse matrix (`nnz = N²`): `assemble` returns `RankOneUpdate(A, σ, ws, wt)`, the matrix
`A + σ ws wtᵀ` kept as its parts, `ws` assembled from the test side and `wt` from the trial
side. `pde_solve` solves it exactly through the sparse bordered system, as for **Mean-zero**.
`public`, not exported (`Bramble.RankOneUpdate`); supported on plain and scalar ghost spaces,
refused on a composite of ghost spaces. See [docs/src/tutorials/form.md](docs/src/tutorials/form.md)
and [docs/src/api/forms.md](docs/src/api/forms.md).
