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

**Assembly**:
Filling a matrix or vector from a form. `assemble` allocates and fills; `assemble!` refills an
existing one, the time-loop call, and must not allocate.

**Block**:
One leaf-space pair's rectangle in a composite system matrix: row offset from the test leaf,
column offset from the trial leaf.
