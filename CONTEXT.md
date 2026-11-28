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
A named subset of a domain, `label => face` or `label => predicate`: a `Symbol` name paired
with the rule selecting points.
_Avoid_: tag, region, boundary condition (a marker names *where*, never *what value*)

**Label**:
The `Symbol` half of a marker (`:left`, `:boundary`).

**Level-set marker**:
A 2D marker `label => levelset(φ)`: the region where `φ(x) < 0`, with the labels `label`,
`label_boundary` and `label_exterior`, as for a polygon. Its mesh is a **Fitted grid**.
_Avoid_: implicit surface, signed-distance function (`φ` need not be a distance)

**Orthogonally convex**:
Of a curve: every horizontal and vertical line meets it in at most two points. Weaker than
convex (a dented egg qualifies; a star fails), and required for a level-set fitted grid.

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
`:boundary` and `:interior`, which every mesh computes from its geometry unless the domain
defines them.

**Refinement**:
Dyadic halving in place (`iterative_refinement!`): the same mesh split, not a new draw, so a
convergence order is measurable on a random mesh.

**Fitted grid**:
The grid `mesh` builds for a polygon or level-set marker: every line meeting the boundary has
a node on it (or within a quarter of `h²`), and no edge joins an interior and an exterior
node. `polygon_spaces` solves on the polygon these nodes form. Design record:
`docs/notes/internals/levelset.md`.
_Avoid_: conforming mesh, body-fitted mesh

**Closure path**:
The fitted-grid construction for a level-set curve whose crossing map `T` has finite order
(`_grid_closure`); vertices lie on the curve to round-off. Not the **Closure** of a ghost
space.

**Orbit path**:
The fitted-grid construction when `T` is conjugate to an irrational rotation (a rotated
ellipse, an egg): periodic orbits of `T` replace the grid lines. A curve on neither path
makes `mesh` throw.

**Ghost mesh**:
A physical mesh `Ωₕ` plus one ghost point beyond each node of its ghosted facets
(`Bramble.GhostMesh(Ωₕ; ghosts = :boundary)` or named facets). Points and indices are the
padded mesh's; labels name physical nodes. Stale after `iterative_refinement!` of `Ωₕ`.
Owner: [docs/src/internals/mesh.md](docs/src/internals/mesh.md).
_Avoid_: halo, boundary layer

**Padded mesh**:
The mesh of a ghost mesh's points: `Ωₕ` extended by one point on each ghosted side
(`npoints(G, Tuple)`).

**Facet ghost**:
The padded point beyond a node of a ghosted facet, numbered `1:nghosts(G)`;
`Bramble.ghost_index` and `Bramble.boundary_node` map between a facet node and its ghost.

**Absent point**:
A padded point outside the physical mesh along two or more axes (a 2D corner): never a degree
of freedom, 0 in a lifted grid function (`Bramble.is_absent(G, J)`).

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
Stokes are two leaves), addressed through `leaf_spaces_offsets`.
_Avoid_: component (the `components` keyword *selects* leaves), field, block

`u(i)`, `components(u)` and `component_range` number leaves the same way, so on
`CompositeGridSpace((W × W, W))` `u(2)` is the second leaf, inside the first child. (`×`
flattens: `W × W × W` is one level of three leaves.)

**Backend**:
Where a space's arrays live and how they are iterated, with the execution policy
(`CpuSerial()`/`Serial()`, `CpuThreaded()`/`Parallel()`, `CpuPolyester()`, `GpuKernel()`, ...)
as a trait.
_Avoid_: device, mode

**DOF order**:
A ghost space's unknowns: the physical nodes no Dirichlet relation fixes, in mesh order, then
the remaining ghosts. Owner: [docs/src/internals/space.md](docs/src/internals/space.md).

**Physical-only weights**:
The quadrature weights of a padded space: those of `gridspace(Ωₕ)` at physical nodes, zero at
every ghost and absent point.

**Closure**:
The pair `(E, d)` of a ghost space (`Bramble.closure(Wₕ)`): unknowns `u` stand for the padded
values `E u + d`, with `d` the data at eliminated ghosts. Assembly returns `Eᵀ A_pad E`, so a
space with `d ≠ 0` needs the pair `assemble(a, l)` for its load. `Bramble.padded_space(Wₕ)` is
the space of the padded values.

**Relation**:
A boundary condition of a ghost space, `facet => operator => data`
(`:xmin => α * u + β * ∂ₙ(u) => g`): a Dirichlet relation when `operator` is `u`, otherwise
solved at each facet node for its ghost.
_Avoid_: boundary condition (a relation pairs an operator with data; a marker names *where*)

**Constrained space**:
A ghost space whose ghosts are tied to relations, `constrained(Wₕ, u -> (:xmin => ∂ₙ(u) =>
g, …))`; its closure comes from the relations. See
[docs/src/tutorials/form.md](docs/src/tutorials/form.md) (the ghost sections).

**Lift**:
`Bramble.lift(Wₕ, u)`, the padded grid function `E u + d` (`E u + d(t)` with `lift(Wₕ, u, t)`
for time-dependent data); `lift!` fills one in place.

### Operators

**Difference operator**:
A discrete derivative: `D₋ₓ` backward, `Dcₓ` centred, the subscript naming the direction. As
a variable, a direction indexes the vectorial operator (`∇ₕ[1] === ∇ₕ[:x] === D₋ₓ`) or is the
stem's argument (`Bramble.D₋(uₕ, 1)`). Averages take it through `Mₕ`/`M₊ₕ`; there is no bare
`M`.
_Avoid_: derivative (the continuous object), gradient (that is `∇ₕ`)

**Restriction (`Rₕ`)**:
Projection of a continuous function onto grid functions by evaluating it at mesh points.
_Avoid_: interpolation (that is `πₕ`), sampling

**Cell average (`avgₕ`)**:
The same projection by quadrature over each cell: same target space, different and costlier
rule.
_Avoid_: restriction (name the rule, since both project)

**Stencil**:
The neighbouring points one operator application reads, with weights (`local_stencil`).
_Avoid_: footprint, pattern (a *pattern* is a matrix's sparsity)

**`innerₕ` and `inner₊`**:
`innerₕ` is the discrete ``L^2`` inner product, each point weighted by its cell measure;
`inner₊` the modified ``L^2_+`` product used with backward differences. Distinct objects,
not spellings.

**Facet operator**:
`∂ₙ` (conormal `∂ₙ(u; K)` included), `∂ₙₙ` and `jumpₙ`: operators on facets that live only
inside `inner_Γ`, one value per facet a `markers` entry selects. `∂ₙ` and `∂ₙₙ` read the ghost
beyond the facet; `jumpₙ` needs no ghost.

**Corner rule**:
The stencil of a mixed difference at a 2D corner (or 3D edge) node of a ghost mesh, where it
would read an absent point: inward one-sided differences along both axes, second order. See
[docs/src/internals/space.md](docs/src/internals/space.md).

### Forms and assembly

**Form**:
A symbolic expression in trial and test functions (`LinearForm` in one, `BilinearForm` in
two). Structure, not values.
_Avoid_: weak form, variational form, integrand

**Trial and test function**:
The two arguments of a bilinear form. The test function indexes matrix **rows**, the trial
function columns: easy to write backwards.

**Source**:
A term carrying known data, not an unknown, so it can appear in a linear form.

**AST**:
The resolved operator expression a form compiles to (`resolve_form_ast`); what assembly
walks.
_Avoid_: expression tree, symbolic form, IR

**Pattern**:
A system matrix's sparsity, fixed by the stencil while mesh and expression are unchanged.
_Avoid_: stencil, structure

**Separable form**:
A bilinear form on a `MeshnD` whose matrix is a sum of Kronecker products of 1D factors, one
per axis (`is_separable`).
_Avoid_: tensor-product form, factorisable

**Laplacian-like form**:
A separable form with one mass per axis, each term differing from them on at most one axis:
what `fdm_solve` solves directly. A mixed derivative is separable but not Laplacian-like.
_Avoid_: Poisson-like, elliptic

**Assembly**:
Filling a matrix or vector from a form. `assemble` allocates; `assemble!` refills, the
time-loop call, and must not allocate.

**Block**:
One leaf-space pair's rectangle in a composite system matrix: rows from the test leaf,
columns from the trial leaf.

**Mean-zero**:
`pde_solve(A, F; mean_zero = w)`: the bordered solve returning the `u` with `wᵀ u = 0`, for a
pure-Neumann `A` that annihilates constants.

**Rank-one**:
A term `σ innerₕ(g, u) innerₕ(h, v)` (`g`, `h` data) kept out of the sparse matrix:
`assemble` returns `Bramble.RankOneUpdate(A, σ, ws, wt)` and `pde_solve` solves it exactly.
See [docs/src/api/forms.md](docs/src/api/forms.md).
