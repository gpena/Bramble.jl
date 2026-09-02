```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# API reference

Documentation for `Bramble.jl`'s public API.

---

## Utilities

### Linear algebra backends

```@docs
backend
ExecutionPolicy
Serial
Parallel
execution_policy
vector
matrix
vector_type
matrix_type
backend_types
backend_eye
backend_zeros
metal_backend
```

### Function embedding

```@docs
BrambleFunction
embed_function
has_time
```

---

## Geometry

### Sets and intervals

```@docs
interval
point
box
cartesian_product
×
dim
topo_dim
tails
center
projection
is_collapsed
point_type
get_boundary_symbols
set
```

### Markers and domains

```@docs
markers
domain
labels
```

---

## Meshes

### Mesh types and constructors

```@docs
AbstractMeshType
Mesh1D
MeshnD
MeshMarkers
mesh
submeshes
```

### Points and spacings

```@docs
npoints
points
half_points
half_point
spacing
forward_spacing
half_spacing
spacings
half_spacings
hₘₐₓ
hₘᵢₙ
stepsize
locate_cell
normal_vector
cell_measure
cell_measures
is_uniform
```

### Mesh iterators

```@docs
points_iterator
half_points_iterator
spacings_iterator
forward_spacings_iterator
half_spacings_iterator
cell_measures_iterator
```

### Mesh indexing and boundaries

```@docs
indices
boundary_indices
interior_indices
is_boundary_index
index_in_marker
```

### Mesh adaptation and mutation

```@docs
iterative_refinement!
change_points!
set_points!
```

---

## Grid spaces

### Function spaces

```@docs
ScalarGridSpace
CompositeGridSpace
gridspace
vector_gridspace
```

### Space properties and degrees of freedom

```@docs
ndofs
weights
spaces
```

### Vector elements and grid functions

```@docs
VectorElement
element
to_matrix
component
components
component_range
component_ranges
```

### Restriction and averaging operators

```@docs
Rₕ
Rₕ!
avgₕ
avgₕ!
```


## Difference, jump and average operators

The unscaled difference and its finite difference counterpart, per coordinate and over
every coordinate at once. See the [operators tutorial](tutorials/operators.md).

```@docs
diff₋ₓ
diff₋ᵧ
diff₋₂
diff₋ₕ
diff₊ₓ
diff₊ᵧ
diff₊₂
diff₊ₕ
D₋ₓ
D₋ᵧ
D₋₂
∇₋ₕ
D₊ₓ
D₊ᵧ
D₊₂
∇₊ₕ
```

The forward difference over the averaged spacing, which is the one that satisfies
the discrete summation-by-parts identity
``(\textrm{Dstar}_{+x} u_h, v_h)_h = -(u_h, D_{-x} v_h)_{+x}`` for grid functions
`vₕ` vanishing on the boundary.

```@docs
Dstar₊ₓ
Dstar₊ᵧ
Dstar₊₂
Dstar₊ₕ
```

The centered difference, over the span its stencil covers. It reproduces the derivative
of an affine function exactly on any grid, and is skew-symmetric in `innerₕ` for grid
functions vanishing on the boundary.

```@docs
Dcₓ
Dcᵧ
Dc₂
Dcₕ
```

The cross-weighted centered difference, the same two one-sided differences weighted by
the opposite spacings. It reproduces the derivative of a quadratic exactly on any
grid, and so is second order on a non-uniform one where `Dcₓ` is first.

```@docs
Dₕₓ
Dₕᵧ
Dₕ₂
∇ₕ
```

Jumps across an interface, ``\llbracket u \rrbracket = u_{i+1} - u_i``. There is one
of these rather than a forward and a backward pair: a jump belongs to the interface
between two cells, not to a direction of travel across it.

```@docs
jumpₓ
jumpᵧ
jump₂
jumpₕ
```

Averages of a point with its neighbour.

```@docs
M₋ₓ
M₋ᵧ
M₋₂
M₋ₕ
M₊ₓ
M₊ᵧ
M₊₂
M₊ₕ
```

## Inner products and norms

```@docs
innerₕ
inner₊
inner₊ₓ
inner₊ᵧ
inner₊₂
normₕ
norm₁ₕ
snorm₁ₕ
norm₊
```

---

## Forms

Linear and bilinear forms, their assembly, and the boundary conditions applied to an
assembled system. See the [forms tutorial](tutorials/form.md).

### Building a form

```@docs
form
```

### Assembling

`assemble` allocates its result; the mutating forms refill one that already exists, which is
what a time loop wants. `allocate_system_matrix` builds a matrix's sparsity pattern once so
that `assemble!` can refill it without allocating.

```@docs
assemble
assemble!
assemble_parallel!
allocate_system_matrix
evaluate!
```

### Dirichlet conditions

```@docs
DirichletConstraint
dirichlet_constraints
dirichlet_bc!
symmetrize!
```

### Structural properties

Whether a `BilinearForm` is symmetric, or symmetric positive semi-definite, by construction
— a cheap, symbolic check on its expression, answered before any matrix is assembled.

```@docs
issymmetric(::BilinearForm)
isposdef(::BilinearForm)
```

---

## Exporters

Writing a mesh and its grid functions to a file a viewer can open. See the
[VTK export tutorial](tutorials/vtk_export.md) and the
[PGFPlots export tutorial](tutorials/pgfplots_export.md).

```@docs
export_vtk
export_pgfplots
```
