```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Spaces

## Ghost spaces

`gridspace(Ωₕ; ghosts = :boundary)` (or `ghosts = (:xmin, :ymax)` for named facets)
returns a [`GhostGridSpace`](@ref): a grid space whose degrees of freedom are the physical
nodes of `Ωₕ` and the ghost values beyond the named facets. It holds a `ScalarGridSpace` on
the padded mesh (a [`GhostMesh`](@ref), see [Mesh](mesh.md)), reached by `padded_space`, and a
closure `(E, d)`, reached by `closure`, with `u_pad = E u + d`. Every form on a ghost space
is assembled on the padded space, with the engines, patterns and replays of a plain space,
and reduced afterwards.

### DOF order

The degrees of freedom are the physical nodes in the mesh's linear order, then the ghosts in
the numbering of the [ghost mesh](mesh.md), whose absent points are never degrees of
freedom. The first entries of a vector of the space are therefore the values at the physical
nodes, as on `gridspace(Ωₕ)`.

### Weights

Per axis the weight of a ghost node and of a ghost cell is 0, so a form has no weight in
its ghost rows and only the reduction below gives the ghosts a role. The
[space tutorial](@ref space_ghost) gives the weights and why they are physical-only.

### The closure and the reduction

On `gridspace(Ωₕ; ghosts)` the closure is the selection of the degrees of freedom among the
padded points. `E` is `N_pad × ndofs` with one entry 1 per column, and `d = 0`. On the
result of [`constrained`](@ref) the closure comes from the relations. `E` expresses each
eliminated ghost through the physical nodes, and `d` carries the relations' data at the
eliminated ghosts. Assembly fills `A_pad` on the padded space (on a traction facet of a
composite, by the facet rule below) and returns ``E_v^T A_{\mathrm{pad}} E_u``; the load of
a linear form is ``E_v^T (F_{\mathrm{pad}} - A_{\mathrm{pad}}\, d_u)``, so a space with ``d
\neq 0`` needs the bilinear form, and `assemble(l)` alone throws an `ArgumentError`. The
reduction is a sparse map from the stored values of `A_pad` to those of the reduced matrix,
computed once with the form, so a warm `assemble!` allocates nothing. It is symmetric
whenever `A_pad` and the two closures are.

When the test closure skips rows, as on polygon and level-set spaces, which read the
interior nodes only, `A_pad` stores the rows the closure reads and assembly walks only the
points those rows need, under `CpuSerial`; the matrix-free product walks the same points.
A threaded policy, or a walk that would keep more than 7/8 of the points, walks the whole
padded grid. The whole-grid padded matrix is `assemble(a.padded)`.

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
that is zero at the ghost tap still gets the one-sided row.

Reading an absent point is otherwise refused. A product on a ghost space asks, at every
node where its weight is not zero, whether either operand reads a point the padded grid
lacks, and throws an `ArgumentError` naming the term when one does: an average, jump or
shift of a cross difference, a difference around a second difference, and deeper nestings.
Reads a zero weight or a zero scale multiplies are not reads. Under a facet product, a read
that leaves the padded grid (past the one ghost layer, or across a facet without ghosts) is
refused too. A volume product's operand at the edge of the padded grid truncates, as on a
plain mesh, and that is allowed.

The rule applies in forms only; the runtime operators are eager (see the
[operators tutorial](@ref operators_ghost)). `Rₕ` and `avgₕ` refuse on a ghost space with
an `ArgumentError` (`avgₕ` would average over cells past the physical facet), except that
`Rₕ(U, f)` works on the trial space of [`polygon_spaces`](@ref Bramble.polygon_spaces); the
[support table](@ref "Support on ghost spaces") lists what else refuses.

### Relations

`constrained(Wₕ, u -> (:F => operator => data, ...))` calls the lambda once with the trial
placeholder of the padded space. The facet operators `∂ₙ`, `∂ₙₙ` and `jumpₙ` are
builder-only: each holds one operator per (axis, side) and `inner_Γ` expands it into one
product per incident facet with that facet's weight, so a corner node gets one value per
facet and no assembly node is added. A relation of a scalar space takes its facet's
component. Its rows on the facet nodes are assembled by forms,
`inner_Γ(relation, v; markers = (F,))`, and solved for the ghosts: one entry per ghost, or a
2×2 block per 2D corner where two conormal relations read each other's ghost. Each block is
solved by its own LU, and `E` and `d` stay sparse; no dense factorisation of the whole
closure is formed. Ghosted facets without a relation keep their ghosts as unknowns.

On a composite `C₁ × C₂`, `constrained(V, u -> (...))` constrains the composite as a whole,
and a relation may read several components, as the traction of elasticity does. Component
`i` of it is `∂ₙ(u(1); K = Kᵢ₁) + ∂ₙ(u(2); K = Kᵢ₂)`, with `Kᵢⱼ` a constant matrix. The
relation is split by the component it reads, each piece assembled by its own `inner_Γ` and
tested against `v(i)` of the ghost it is attached to. A facet whose relation reads several
components carries one ghost relation per component, written unkeyed. The rows of all of
them at a node, or at a corner, are solved together as one block, so component 1's relation
may read component 2's ghost. A Dirichlet relation names its component, `u(i) => g`, and a
displacement facet's ghosts are closed by `∂ₙₙ(u(i)) => 0`. The rows are solved once on the
padded composite: the padded space is `padded(C₁) × padded(C₂)`, and the unknowns are
component 1's then component 2's, each in the scalar order. The closure `(E, d)` is
therefore general, not `blockdiag`, wherever a relation couples components; with relations
of one component each it equals `constrained(C₁, r₁) × constrained(C₂, r₂)`. The result
stores its own `(E, d)`, which `closure`, `lift` and its forms read. A component view
`uₕ(i)` of it is refused, since `E` couples the components, so take
`lift(Vₕ, parent(uₕ))(i)` instead. `×` and `^` of it, data depending on time, the
`dirichlet` rows and `GhostVector` are refused too. A corner of two traction facets uses a
one-sided tangential difference along the facets, since the two shear relations there are
one equation.

On a facet whose ghost relation reads several components, a traction, the mixed differences
of the form read no ghost of that facet. The mixed differences in the left argument of a
product, the argument whose stencil gives the matrix columns, extrapolate the function to
the ghost from its values at the three nodes inward along the normal, to second order on
non-uniform spacing. Second differences and the relations still read the ghosts of the
closure, and `E` and `d` are unchanged, so the relations hold on lifted values exactly as
before. The shear-closed ghosts make the reduced matrix of the collocated form acquire an
eigenvalue with negative real part, measured for a traction on one facet of a uniform grid
(`docs/notes/internals/ghost_points.md`), once the cells are wider across the facet than
along it. A composite whose relations each read one component is not affected.

### Grid functions with ghosts

A grid function on a ghost space has two layouts, one for time loops and one for forms.

A [`GhostVector`](@ref) is the layout of a time loop: the physical values stay the solver's
own contiguous `Vector`, and the ghosts sit in a facet buffer in the order of the
[ghost mesh](mesh.md). It carries the mesh with ghosts. The runtime operators then know the
padded spacings and the ghosted facets. A padded `VectorElement` is the layout of forms:
`lift(Cₕ, u)` returns `E u + d` on the padded space, ghosts filled and absent points 0, and
assembly already works on the padded mesh. The padded layout costs a copy in and out of
about two stencil applications, which is why a time loop keeps the facet buffer. The DOF
order above is the same for both, the padded layout appending the absent points as a suffix
that is dropped.

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
`CpuPolyester`; on `CpuThreaded` each band pass costs the task spawn of the plain space's
own call and nothing that grows with the grid.

### Support on ghost spaces

Each row is an operation of the standard space, its status on a ghost space (a scalar one
unless the row says composite) and the test file that pins it. "Refused" is an
`ArgumentError` that names the fix. `Cₕ` is the constrained ghost space, `a` and `l` its
forms.

| Operation | Status | Test |
|:----------|:-------|:-----|
| `assemble`, `assemble!`, `lift`, `pde_solve` | supported; `pde_solve` runs on the reduced matrix | `test/ghost/policies.jl` |
| `matrix_free_operator` | supported; one product at a time per operator (a concurrent entry throws), and 0 B per warm serial apply in 1D and 2D (3D on the nightly) | `test/ghost/matrix_free.jl` |
| `kronecker_operator` | supported where the closure is separable, trial and test on one mesh object and a square reduced system; refused otherwise, naming `assemble(a)` | `test/ghost/kronecker.jl` |
| `gmg_solve`, `gmg_preconditioner` | supported for Neumann, Robin, mixed and pure-Neumann ghost problems, with level-independent cycle counts on 1D graded and on jittered meshes; on 2D meshes graded as `t^(1 + d/4)`, where point smoothers slow down on every space, within 3 cycles of the plain path (3D on the nightly) | `test/ghost/multigrid.jl` |
| `evaluate!`, `l(v)`, `a(u, v)` | supported; `l(v)` and `evaluate!` refuse when the affine part `d` is not zero, naming `assemble(a, l)` | `test/ghost/contract.jl` |
| `normₕ`, `interpolate_at` | supported on physical values only; refused on a DOF vector that is not a grid function, naming `lift` | `test/ghost/grid_functions.jl` |
| `jacobi_preconditioner`, `chebyshev_preconditioner`, `ilu_preconditioner`, `amg_preconditioner` | supported | `test/ghost/preconditioners.jl` |
| `fdm_solve`, `fdm_factorize`, `fdm_solve!`, `fdm_preconditioner` | supported where the form is separable; refused with the `dirichlet` keyword, on a 1D form and on a non-separable form | `test/ghost/preconditioners.jl` |
| `solve(a, l; preconditioner = :amg)` and `:ilu0` (BrambleSciMLExt) | not tested on ghost forms | none |
| `assemble_parallel!` | refused, naming `assemble!` (it runs in parallel under the policy of the mesh) | `test/ghost/policies.jl` |
| `Rₕ`, `avgₕ` | refused, naming `gridspace(Ωₕ)` and `lift`; on the trial space of `polygon_spaces`, `Rₕ` is supported (see the polygon table) | `test/ghost/grid_functions.jl` |
| `CpuSerial`, `CpuThreaded`, `CpuPolyester` | supported; the reduction gives the serial sums bitwise | `test/ghost/policies.jl`, `test/ext/ghost_polyester.jl` |
| device backends | refused; no test runs on a device | none |
| `semidiscretize`, `ode_problem` | supported on a constrained space with `(x, t)` data; refused when the ghosts are unknowns and on composite ghost spaces | `test/form/ghost_semidiscrete.jl` |
| `dirac` | supported, including ghosted facets that meet | `test/ghost/dirac.jl` |
| `reaction` | refused, naming the plain-space route (`gridspace` and `lift`) | `test/ghost/dirac.jl` |
| rank-one terms | supported on a scalar ghost space, with the pair load and `dirichlet`; refused on composite ghost spaces | `test/form/ghost_rank_one.jl` |
| composite spaces `C₁ × C₂`, `C^Val(n)`, `C^N` with `N ≤ 3` | supported for ghost components with the same ghosted facets and mesh size: forms, `assemble`, loads, `lift`, `dirichlet` per component by keyword or relation, relations reading several components (`constrained` of the whole composite), contractions; refused beside a plain component, with different facets per component, and for `semidiscretize`, matrix-free, Kronecker, `fdm_*` and the preconditioners | `test/ghost/composite.jl`, `test/ghost/cross_component.jl` |

A mixed difference at a corner takes the corner rule on a composite as on a scalar space.
On a facet of a composite constrained as a whole whose relations read several components,
the facet rule applies as well, and the assembled rows differ from the plain padded form's
there (see the traction rule above).

## Polygon spaces

[`polygon_spaces`](@ref Bramble.polygon_spaces) builds `(U, W)` as two `GhostGridSpace`s
over the mesh itself. Each exterior diagonal neighbour `p = k + (sx, sy)` of an interior
node `k` takes the relation (3.2) of the paper, `u_p = u_{k+(sx,0)} + u_{k+(0,sy)} - u_k`.
At a V-notch, an exterior node read by two interior nodes has two different values and so no
row of the extension; the reduction substitutes the value of each reading equation, as the
paper does.

The spaces of a level-set marker (`:D => levelset(φ)`, [Meshes](../api/meshes.md)) are the
same call on the polygon inscribed in the curve: `g` is read at the polygon's vertices, the
boundary nodes. The curve is orthogonally convex, so no line meets it four times near a
notch and no exterior node has two readers; the V-notch rows below do not arise there.

### Support on polygon spaces

Each row is an operation and its status on `(U, W)`, with the test that pins it: the file,
and the testset where the file has several. A row with no test says "untested", and
nothing pins its status. "Throws" is an `ArgumentError`. A
level-set mesh (`:D => levelset(φ)`) has the same behaviour; its own tests are named below
the table. Files are in `test/ghost/` unless a path says otherwise.

| Operation | Status | Test |
|:----------|:-------|:-----|
| `assemble(a, l)`, `assemble(a)` | supported; each row equals the plain matrix's with every exterior column replaced by its own (3.2) row | `polygon_spaces.jl` ("rows match the plain-matrix oracle"), `polygon_oracle.jl` |
| warm `assemble!` | 0 B on `CpuSerial` and on `CpuPolyester`; under `CpuThreaded` the values are the serial ones and no allocation is asserted | `polygon_spaces.jl` ("warm assemble! allocates nothing"), `allocations.jl` ("Warm refill on polygon spaces"), `ext/ghost_polyester.jl` ("polygon_spaces under CpuPolyester") |
| diffusion, mixed, convection (central) and reaction (the zeroth-order term `q u`) | supported; the rows equal the oracle's, with variable `a`, `b`, `c`, `d` and `q` | `polygon_oracle.jl` ("Variable coefficients (seed $seed)") |
| upwind terms (`D₋ₓ(u) + D₋ᵧ(u)`) | supported; exact for a linear `u` | `polygon_support.jl` ("upwind D₋ₓ(w) + D₋ᵧ(w)") |
| `CpuSerial`, `CpuThreaded`, `CpuPolyester` | supported; matrix, load, refilled matrix, solution and `lift` equal the serial ones bit for bit, the matrix-free product to 1e-13 | `policies.jl` ("polygon_spaces under CpuThreaded"), `ext/ghost_polyester.jl` ("polygon_spaces under CpuPolyester") |
| `pde_solve`, `linear_problem`, `solve(a, l)` | supported, with the default solver and with `KrylovJL_GMRES` | `policies.jl` (`pde_solve`), `polygon_spaces.jl` ("linear_problem and solve (#880)") |
| `ilu_preconditioner`, `amg_preconditioner` | supported; each takes under half the iterations of unpreconditioned GMRES | `polygon_support.jl` ("ILU and AMG preconditioned GMRES") |
| `jacobi_preconditioner`, `chebyshev_preconditioner`, `solve(a, l; preconditioner = :amg)` | untested | none |
| `kronecker_operator`, `fdm_solve`, `fdm_factorize`, `fdm_factorize!`, `fdm_preconditioner`, `gmg_preconditioner` | throw, naming polygon spaces; the multigrid message names #846 | `polygon_spaces.jl` ("Kronecker, fdm, GMG refusals (#880)") |
| `matrix_free_operator`, contracting `a(u, v)` | supported; a form that reads a V-notch conflict node throws, decided when `form` is called, before and after a warm `assemble!`; 0 B per warm `mul!` under `CpuPolyester` | `polygon_spaces.jl` ("matrix-free, contract, semidiscretize"), `polygon_oracle.jl` ("V-notch: conflict read set by form"), `ext/ghost_polyester.jl` ("Warm calls allocate nothing") |
| `Rₕ(U, f)`, `Rₕ!`, `normₕ`, `snorm₁ₕ`, `norm₁ₕ`, `norminf` on `U` | supported; the norms read the element at the interior nodes and 0 at the boundary nodes; `Rₕ(U, f; markers = ...)`, and all of them on `W`, throw | `polygon_oracle.jl` ("Rₕ and norms on the trial space"), `convergence/poisson_polygon.jl` ("Norm errors match the hand sums", and the convergence orders) |
| `interpolate_at(lift(U, u), x)` | supported; the bilinear blend on the cell holding `x` | `polygon_support.jl` ("interpolate_at of lift(U, u)") |
| `export_vtk` of `lift(U, u)` | supported; the whole box, the extension included | `polygon_support.jl` ("export_vtk of lift(U, u)") |
| `Float32` | supported; matrix, load, solution and lift keep the element type | `polygon_support.jl` ("Float32 keeps its eltype") |
| `semidiscretize` on `(W, W)`, as `semidiscretize(form(W, W, f), form(W, g))` | supported, with the boundary nodes read as 0; the mass matrix is the dual-cell areas | `polygon_support.jl` ("semidiscretize on (W, W)") |
| `semidiscretize` on `(U, W)` | throws | `polygon_spaces.jl` ("matrix-free, contract, semidiscretize") |
| `W` as trial space | supported only for forms that read no exterior node, with the boundary nodes read as 0; otherwise throws | `polygon_spaces.jl` ("test space as trial side") |
| `U` as test space, `form(U, f)` | throw | `polygon_spaces.jl` ("Figure 1: exact for quadratics") |
| `dirichlet =` | throws; `g` fixes the boundary | `polygon_spaces.jl` ("refusals") |
| rank-one terms | throw | `polygon_spaces.jl` ("refusals") |
| composites (`U × U`, beside a plain ghost space) | forms throw | `polygon_spaces.jl` ("composites refuse polygon spaces") |
| a form that reads an exterior node past the diagonal neighbours of the interior node | throws, naming the node and the equation; `Dcₓ(Dcₓ(u))` is the tested case | `polygon_spaces.jl` ("refusals") |
| a form that reaches two nodes along an axis and lands on an interior or boundary node | assembles; the V-notch with `Dcᵧ(Dcᵧ(u))` is the probed case | untested |

The last two rows come from one probe on the jittered Figure 1, the integer Figure 1 and the
integer V-notch. `Dcₓ(Dcₓ(u))`, `Dcᵧ(Dcᵧ(u))`, `Dcₓ(Dcᵧ(Dcₓ(u)))` and `Δₕ(Δₕ(u))` throw on
the two Figure 1 meshes, each naming an exterior node two nodes from the equation's interior
node. On the V-notch only `Dcᵧ(Dcᵧ(u))` assembled, as every second node it reads along `y`
is an interior or boundary node, and the others threw. The guard looks at what each
equation reads, not at the width of the stencil: it refuses an exterior node that (3.2)
gives no value, and a wide stencil that lands on interior or boundary nodes passes.

On a level-set mesh the same rows are pinned by `levelset_spaces.jl` (the quadratic solve,
(3.2) at every extended node, the rows against the plain matrix, warm `assemble!`,
matrix-free and contraction, the refusals), by `levelset_oracle.jl` (every term of the
third row, `Rₕ` and the norms), and by the level-set disk of `policies.jl` and
`ext/ghost_polyester.jl`. The test space as trial side has its own level-set testset in
`polygon_support.jl` ("test space as trial side"). The rows about a conflict node do not
arise there, and the remaining rows are untested on a level-set mesh.

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
