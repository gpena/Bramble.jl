```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# API Reference

Documentation for `Bramble.jl`'s public API.

---

## Utilities

### Linear Algebra Backends

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

### Function Embedding

```@docs
BrambleFunction
embed_function
has_time
```

---

## Geometry

### Sets and Intervals (`CartesianProduct`)

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

### Markers and Domains

```@docs
markers
domain
labels
```

---

## Meshes

### Mesh Types and Constructors

```@docs
AbstractMeshType
Mesh1D
MeshnD
MeshMarkers
mesh
submeshes
```

### Points and Spacings

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

### Mesh Iterators

```@docs
points_iterator
half_points_iterator
spacings_iterator
forward_spacings_iterator
half_spacings_iterator
cell_measures_iterator
```

### Mesh Indexing and Boundaries

```@docs
indices
boundary_indices
interior_indices
is_boundary_index
index_in_marker
```

### Mesh Adaptation and Mutation

```@docs
iterative_refinement!
change_points!
set_points!
```
