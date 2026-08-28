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
```

### Restriction and averaging operators

```@docs
Rₕ
Rₕ!
avgₕ
avgₕ!
```

