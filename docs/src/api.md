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
