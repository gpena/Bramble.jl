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

The degrees of freedom are the physical nodes in the mesh's linear order, then the ghosts in
the numbering of the [ghost mesh](mesh.md), whose absent points are never degrees of
freedom. The first entries of a vector of the space are therefore the values at the physical
nodes, as on `gridspace(Ωₕ)`.

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
closures are. Substituting `E` at each stencil tap instead would rewrite the guarded walk,
the pattern recorder and the replay of the assembly, for the same matrix.

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

Nested mixed differences on a ghost space are assembled as forms, where the rule applies;
the runtime operators are eager and cannot apply it (see the
[operators tutorial](@ref operators_ghost)). `Rₕ` and `avgₕ` refuse on a ghost space with an `ArgumentError`
(`avgₕ` would average over cells past the physical facet); the
[support table](@ref "Support on ghost spaces") lists what else does.

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

A grid function on a ghost space has two layouts, one for time loops and one for forms.

A [`GhostVector`](@ref) is the layout of a time loop: the physical values stay the solver's
own contiguous `Vector`, and the ghosts sit in a facet buffer in the order of the
[ghost mesh](mesh.md). It carries the mesh with ghosts, so the runtime operators know the
padded spacings and the ghosted facets. A padded `VectorElement` is the layout of forms:
`lift(Cₕ, u)` returns `E u + d` on the padded space, ghosts filled and absent points 0, and
assembly already works on the padded mesh. Both are kept because the padded layout costs a
copy in and out of about two stencil applications when the solver keeps its own contiguous
vector, and a stock `mul!` on its physical view falls back to a generic quadratic method;
the facet-only layout needs neither, and forms need the padded mesh whatever the layout. The
DOF order above is the same for both, the padded layout appending the absent points as a
suffix that is dropped.

[`update_ghosts!`](@ref)`(v, Cₕ)` fills the ghost buffer from the closure. The ghost update
is precomputed when the space is built: the closure's rows for the ghosts that `d` and `E`
determine are stored once, so the update is one sparse product plus the data, with no
allocation on the serial and Polyester policies. With time-dependent relation data the data
are affine in the unknown, `d(t) = L g(t)` with `L` fixed at construction, so
`update_ghosts!(v, Cₕ, t)` and `lift(Cₕ, u, t)` evaluate `g(t)` at the ghosts' relations and
apply `L`, and `semidiscretize` differentiates `d(t)` exactly for the time derivative of the
load. A space with time-dependent data refuses the calls without `t`.

A runtime operator on a `GhostVector` returns the physical values as a plain `Vector`, with
no ghosts: it reads the ghosts of its argument and truncates, as the operators do on a
plain space. Composing operators on the result is that truncation, so the corner rule, which
lives in the stencil engine, is never applied to a runtime composition. Mixed differences
at a ghost corner stay form-only.

The reduction `Eᵀ A_pad E`, the load `Eᵀ(F_pad − A_pad d)` and `lift` run in bands. The
reduction and the load are grouped by destination entry: each band owns a set of output
entries and sums their contributions in the serial order, so `CpuSerial`, `CpuThreaded` and
`CpuPolyester` all give the sums of the serial run, bitwise. The `!` calls, `lift!`, the
pair `assemble!(A, F, a, l)` and `update_ghosts!` allocate nothing on `CpuSerial` and
`CpuPolyester`; on `CpuThreaded` each band pass costs the task spawn of the plain space's own
call and nothing that grows with the grid.

### Support on ghost spaces

Each row is an operation of the standard space, its status on a ghost space (a scalar one
unless the row says composite) and the test file that pins it. "Refused" is an
`ArgumentError` that names the fix. `Cₕ` is the constrained ghost space, `a` and `l` its
forms.

| Operation | Status | Test |
|:----------|:-------|:-----|
| `assemble`, `assemble!`, `lift`, `pde_solve` | supported; `pde_solve` runs on the reduced matrix | `test/ghost/policies.jl` |
| `matrix_free_operator` | supported; one product at a time per operator (a concurrent entry throws), and 0 B per warm serial apply in 1D to 3D | `test/ghost/matrix_free.jl` |
| `kronecker_operator` | supported where the closure is separable, trial and test on one mesh object and a square reduced system; refused otherwise, naming `assemble(a)` | `test/ghost/kronecker.jl` |
| `gmg_solve`, `gmg_preconditioner` | supported for Neumann, Robin, mixed and pure-Neumann ghost problems, with level-independent cycle counts on meshes of bounded aspect ratio; on graded meshes as slow as the plain path | `test/ghost/multigrid.jl` |
| `evaluate!`, `l(v)`, `a(u, v)` | supported; `l(v)` and `evaluate!` refuse when the affine part `d` is not zero, naming `assemble(a, l)` | `test/ghost/contract.jl` |
| `normₕ`, `interpolate_at` | supported on physical values only; refused on a DOF vector that is not a grid function, naming `lift` | `test/ghost/grid_functions.jl` |
| `jacobi_preconditioner`, `chebyshev_preconditioner`, `ilu_preconditioner`, `amg_preconditioner` | supported | `test/ghost/preconditioners.jl` |
| `fdm_solve`, `fdm_factorize`, `fdm_solve!`, `fdm_preconditioner` | supported where the form is separable; refused with the `dirichlet` keyword, on a 1D form and on a non-separable form | `test/ghost/preconditioners.jl` |
| `solve(a, l; preconditioner = :amg)` and `:ilu0` (BrambleSciMLExt) | not tested on ghost forms | none |
| `assemble_parallel!` | refused, naming `assemble!` (it runs in parallel under the policy of the mesh) | `test/ghost/policies.jl` |
| `Rₕ`, `avgₕ` | refused, naming `gridspace(Ωₕ)` and `lift` | `test/ghost/grid_functions.jl` |
| `CpuSerial`, `CpuThreaded`, `CpuPolyester` | supported; the reduction gives the serial sums bitwise | `test/ghost/policies.jl`, `test/ext/ghost_polyester.jl` |
| device backends | refused; no test runs on a device | none |
| `semidiscretize`, `ode_problem` | supported on a constrained space with `(x, t)` data; refused when the ghosts are unknowns and on composite ghost spaces | `test/form/ghost_semidiscrete.jl` |
| `dirac` | supported, including ghosted facets that meet | `test/ghost/dirac.jl` |
| `reaction` | refused, naming the plain-space route (`gridspace` and `lift`) | `test/ghost/dirac.jl` |
| rank-one terms | supported on a scalar ghost space, with the pair load and `dirichlet`; refused on composite ghost spaces | `test/form/ghost_rank_one.jl` |
| composite spaces `C₁ × C₂`, `C^Val(n)`, `C^N` with `N ≤ 3` | supported for ghost components with the same ghosted facets and mesh size: forms, `assemble`, loads, `lift`, `dirichlet` per component by keyword or relation, contractions; refused beside a plain component, with different facets per component, and for `semidiscretize`, matrix-free, Kronecker, `fdm_*` and the preconditioners | `test/ghost/composite.jl` |

A mixed difference that reaches an absent corner is refused on a composite, where the scalar
form takes the corner rule; with ghosts on three facets it assembles. It refuses and never
computes wrong numbers.

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
    "operators/ghost_vector_operators.jl",
    "space/inner_product.jl",
]
```
