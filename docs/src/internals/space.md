```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Spaces

## Ghost spaces

`gridspace(Ωₕ; ghosts = :boundary)` (or `ghosts = (:xmin, :ymax)` for named facets) returns a
[`GhostGridSpace`](@ref): a grid space whose degrees of freedom are the physical nodes of
`Ωₕ` and the ghost values beyond the named facets. It holds a `ScalarGridSpace` on the
padded mesh (a [`GhostMesh`](@ref), see [Mesh](mesh.md)), reached by `padded_space`, and a
closure `(E, d)`, reached by `closure`, with `u_pad = E u + d`. Every form on a ghost space
is assembled on the padded space, with the engines, patterns and replays of a plain space,
and reduced afterwards.

### DOF order

The degrees of freedom are the physical nodes in the mesh's linear order, then the facet
ghosts: the facets `xmin, xmax, ymin, ymax, zmin, zmax` (named ones only), each facet's
ghosts in the column-major order of its nodes. Absent padded points (2D corners, 3D edges
and corners) are never degrees of freedom. The first entries of a vector of the space are
therefore the values at the physical nodes, as on `gridspace(Ωₕ)`.

### Weights

The padded space has physical-only weights: per axis, the factor of a ghost node and of a
ghost cell is zero, and a facet node keeps its physical half cell. The tensor products then
give `innerₕ`, `inner₊`, `inner_Γ` and the facet markers the weights of `gridspace(Ωₕ)` at
the physical nodes, cells and facets, and exactly zero at every ghost and absent point. A
form on the padded space has no weight in its ghost rows, so only the reduction below gives
the ghosts a role.

### The closure and the reduction

On `gridspace(Ωₕ; ghosts)` the closure is the selection of the degrees of freedom among the
padded points. `E` is `N_pad × ndofs` with one entry 1 per column, and `d = 0`. On the
result of [`constrained`](@ref) the closure comes from the relations. `E` expresses each
eliminated ghost through the physical nodes, and `d` carries the relations' data at the
eliminated ghosts. Assembly fills `A_pad` on the padded space and returns
``E_v^T A_{\mathrm{pad}} E_u``; the load of a linear form is
``E_v^T (F_{\mathrm{pad}} - A_{\mathrm{pad}}\, d_u)``, so a space with ``d \neq 0`` needs the
bilinear form, and `assemble(l)` alone throws an `ArgumentError`. The reduction is a sparse
map from the stored values of `A_pad` to those of the reduced matrix, computed once with the
form, so a warm `assemble!` allocates nothing. It is symmetric whenever `A_pad` and the two
closures are. Substituting `E` at each stencil tap was not done: it would rewrite the guarded
walk, the pattern recorder and the replay of the assembly, and the reduction gives the same
matrix.

`dirichlet` and `symmetrize` replace rows of the reduced system when its unknowns are the
physical nodes in mesh order (every ghost eliminated, no node fixed by a Dirichlet
relation). A Dirichlet condition is also available as the relation `:F => u => g` of
`constrained`, which removes the facet's nodes from the unknowns and moves their data into
`d`. Both routes agree on the solution.

### The corner rule

A mixed difference at a 2D corner node, or at a 3D edge node, of a ghost mesh would read an
absent padded point: the tap to the facet ghost along one axis re-evaluates the inner
difference there, whose tap along the other axis lands outside along both. At such a node
the stencil engine uses the inward one-sided difference along both axes, `D₊` at a first
node and `D₋` at a last, which reads physical nodes only and is second order. The rule
applies to a first difference of first differences across two axes, with constant or
grid-function scales, and it distributes over sums and differences of such terms.

The rule is structural: which stencil a corner node uses depends on the form, not on the
values of its coefficients, so the discretisation is continuous in the data. A coefficient
that is zero at the ghost tap still gets the one-sided row. Off a ghost mesh the test is a
constant `false` and plain meshes assemble bitwise as before.

Reading an absent point is otherwise refused. A product on a ghost space asks, at every
node where its weight is not zero, whether either operand reads a point the padded grid
lacks, and throws an `ArgumentError` naming the term when one does: an average, jump or
shift of a cross difference, a difference around a second difference, and deeper nestings.
Reads a zero weight or a zero scale multiplies are not reads. Under a facet product, a read
that leaves the padded grid (past the one ghost layer, or across a facet without ghosts) is
refused too. A volume product's operand at the edge of the padded grid truncates, as on a
plain mesh, and that is allowed.

Nested mixed differences on a ghost space are assembled as forms, where the rule applies.
The runtime operators are eager: `Dcᵧ(uₕ)` returns a finished grid function, which cannot
apply the rule element by element, so composing mixed differences at runtime on a grid
function with ghosts reads the inner result's truncated ghost values and is not claimed to
equal the form at the ghosts next to a corner. Likewise `avgₕ` has no method on a ghost
space and must not be used on a padded space: it averages over cells past the physical
facet.

### Relations

`constrained(Wₕ, u -> (:F => operator => data, ...))` calls the lambda once with the trial
placeholder of the padded space. The facet operators `∂ₙ`, `∂ₙₙ` and `jumpₙ` are builder-only:
each holds one operator per (axis, side) and `inner_Γ` expands it into one product per
incident facet with that facet's weight, so a corner node gets one value per facet and no
assembly node is added. A relation takes its facet's component. Its rows on the facet nodes
are assembled by forms, `inner_Γ(relation, v; markers = (F,))`, and solved for the ghosts:
one entry per ghost, or a 2×2 block per 2D corner where two conormal relations read each
other's ghost. Each block is solved by its own LU, and `E` and `d` stay sparse; no dense
factorisation of the whole closure is formed. Ghosted facets without a relation keep their
ghosts as unknowns.

### Grid functions with ghosts

A grid function on a ghost space is a padded `VectorElement`: `lift(Cₕ, u)` returns
`E u + d` on the padded space, ghosts filled and absent points 0, and the existing runtime
operators read the ghosts there. This layout was chosen because assembly already needs the
padded mesh, so the padded layout adds no second code path. It is not the only candidate.
A facet-only layout keeps the solver's own `Vector` as the physical part and stores the
ghosts in a facet buffer. Measured on 2D and 3D grids, neither layout pays a SIMD or
latency penalty for a row-wise stencil kernel, but the padded layout costs a copy in and out
of about two stencil applications when the solver keeps its own contiguous vector, and a
stock `mul!` on its physical view falls back to a generic quadratic method, while the
facet-only layout needs neither. The layout of grid functions with ghosts is revisited when
runtime ghost-aware operators are added; the DOF order above is the same for both, the
padded layout appending the absent points as a suffix that is dropped.

## Operator matrices: stencil_matrix versus the Kronecker construction

### Adding a new operator family

```@autodocs
Modules = [Bramble]
Public = false
Filter = x -> x ∉ (Base.parent, Base.:*, Bramble.ldiv!)
Pages = [
    "space/gridspace.jl",
    "space/scalar_gridspace.jl",
    "space/vector_gridspace.jl",
    "space/vectorelement.jl",
    "operators/projection.jl",
    "operators/restriction.jl",
    "operators/cell_average.jl",
    "operators/shift.jl",
    "operators/stencil.jl",
    "operators/stencil_matrix.jl",
    "operators/difference.jl",
    "operators/jump.jl",
    "operators/average.jl",
    "operators/interpolation.jl",
    "space/inner_product.jl",
]
```
