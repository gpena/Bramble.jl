# Bramble

Finite-difference discretisation of PDEs on Cartesian meshes: build a mesh over a domain,
put a discrete function space on it, write a form in terms of difference operators, assemble
it into a matrix. The vocabulary below is the one the code uses; the terms under _Avoid_ are
the ones people reach for instead, which mean something else here or nothing at all.

## Language

### Geometry

**Set**:
A geometric region, built as a `CartesianProduct` of intervals. Pure geometry — it knows
nothing about discretisation or naming.
_Avoid_: region, area, box (`box` is the 3D constructor, not the concept)

**Domain**:
A set together with its markers. This is what a mesh is built from.
_Avoid_: geometry, region

**Marker**:
A named subset of a domain, given as `label => face` or `label => predicate`. The name is a
`Symbol`; the marker is the pairing of that name with the rule that selects points.
_Avoid_: tag, region, boundary condition (a marker names *where*, never *what value*)

**Label**:
The `Symbol` half of a marker — `:left`, `:boundary`. Say label for the name, marker for the
name-plus-rule.

### Meshes

**Mesh**:
The discretisation of a domain into points, uniform or randomly perturbed. `Mesh1D` in one
dimension, `MeshnD` above it.
_Avoid_: grid on its own (it survives only as an adjective, in *grid space* and *grid
function*), triangulation, cells

**Submesh**:
The one-dimensional mesh along a single axis of an `nD` mesh. Every `MeshnD` is a tuple of
these.
_Avoid_: slice, axis mesh

**Reserved markers**:
`:boundary` and `:interior`, which every mesh computes from its own geometry whether or not
the domain named them. A domain may redefine them; the custom definition wins.

**Refinement**:
Dyadic halving in place via `iterative_refinement!`, so a refined mesh is the *same* mesh
split, not an independent draw. This is what makes an order of convergence measurable on a
random mesh.

### Spaces and grid functions

**Grid space**:
The discrete function space over a mesh, carrying the quadrature weights. `ScalarGridSpace`
for one field, `CompositeGridSpace` for several coupled.
_Avoid_: function space, discrete space, FE space

**Vector element**:
A function *in* a grid space — the discrete unknown, `uₕ`. Backed by a flat vector, but the
element is the concept and the vector is storage.
_Avoid_: solution vector, DOF vector, array

**Grid function**:
Acceptable synonym for vector element, and the one the docstrings use in mathematical prose.

**Leaf space**:
One scalar component of a composite space, numbered by its depth-first position — the same
order `u(1)`, `u(2)` address. Velocity and pressure in a Stokes system are two leaves.
_Avoid_: component (used for the `components` keyword, which *selects* leaves), field, block

**Backend**:
Where a space's arrays live and how they are iterated, carrying the execution policy
(`Serial()` or `Parallel()`) as a trait.
_Avoid_: device, mode

### Operators

**Difference operator**:
A discrete derivative — `D₋ₓ` backward, `Dcₓ` centred, subscript naming the direction.
_Avoid_: derivative (reserve for the continuous object), gradient (that is `∇₋ₕ`)

**Restriction (`Rₕ`)**:
The projection of a continuous function onto the space of grid functions, taken by
evaluating it at the mesh points.
_Avoid_: interpolation (that is `πₕ`, a different operator), sampling

**Cell average (`avgₕ`)**:
The same projection taken by averaging over each cell instead, by quadrature. Both land in
the grid-function space; they differ in the rule, and this is the expensive one — six
quadrature nodes per point.
_Avoid_: restriction (name the rule, since both project)

**Stencil**:
The set of neighbouring points one operator application reads, with their weights.
`local_stencil` returns it for a given index.
_Avoid_: footprint, pattern (a *pattern* is the sparsity structure of a matrix)

**`innerₕ` and `inner₊`**:
`innerₕ` is the discrete ``L^2`` inner product, weighting each point by its cell measure.
`inner₊` is the *modified* ``L^2_+`` product used with backward differences — a distinct
object, not a spelling variant.

### Forms and assembly

**Form**:
A symbolic expression in trial and test functions — `LinearForm` in one argument,
`BilinearForm` in two. It stores structure, not values.
_Avoid_: weak form, variational form, integrand

**Trial and test function**:
The two arguments of a bilinear form. Matrix **rows** are indexed by the test function,
columns by the trial function — the asymmetry is load-bearing and easy to write backwards.

**Source**:
A term carrying known data rather than an unknown, so it can appear in a linear form.

**AST**:
The resolved operator expression a form is compiled to, via `resolve_form_ast`. What
assembly actually walks.
_Avoid_: expression tree, symbolic form, IR

**Pattern**:
The sparsity structure of a system matrix: which entries can be non-zero, fixed by the
stencil and invariant while mesh and expression are unchanged.
_Avoid_: stencil, structure

**Assembly**:
Filling a matrix or vector from a form. `assemble` allocates and fills; `assemble!` refills
one that exists, which is the time-loop call and must not allocate.

**Block**:
One leaf-space pair's rectangle within a composite system matrix, located by a row offset
from the test leaf and a column offset from the trial leaf.
