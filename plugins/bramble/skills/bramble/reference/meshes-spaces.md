# Geometry, meshes and spaces

Part of the `bramble` skill. Names written `Bramble.name` are `public` but not exported.

## Geometry & domains

```julia
using Bramble

I = interval(0.0, 1.0)                      # 1D interval
R2 = interval(0.0, 1.0) × interval(0.0, 2.0) # product of intervals (× or \times)
B = box((0.0, 0.0), (1.0, 1.0))              # nD box

Ω = domain(R2,
    :inlet  => :left,
    :outlet => :right,
    :walls  => (:top, :bottom),
    :source => x -> norm(x) < 0.2
)
```

`center`, `projection`, `is_collapsed`, `dim`, `topo_dim`, `point_type`, `boundary_symbols`,
`markers`, `labels`, `set` query one of these.

## Meshes

```julia
Ωₕ_1d = mesh(domain(interval(0.0, 1.0)), 11)
Ωₕ    = mesh(Ω, (10, 20))                         # 10 × 20 uniform
Ωₕ_mx = mesh(Ω, (10, 20), (true, false))          # mixed uniformity

# Queries & metrics
points(Ωₕ), npoints(Ωₕ)          # coordinates, point count
Bramble.half_points(Ωₕ)                  # cell midpoints x_{i+1/2}
Bramble.spacings(Ωₕ)                     # backward nodal spacing h_i = x_i - x_{i-1}
half_spacings(Ωₕ)                # cell width h_{i+1/2}
Bramble.cell_measure(Ωₕ, idx), cell_measures(Ωₕ)   # area/volume of one cell, all cells
hₘₐₓ(Ωₕ), Bramble.hₘᵢₙ(Ωₕ)       # max/min cell diagonal (hₘᵢₙ public only)
Bramble.is_uniform(Ωₕ)                   # true iff spacing is constant along every axis
locate_cell(Ωₕ, x)               # cell index containing a point
Bramble.normal_vector(Ωₕ, idx)   # outward normal at a boundary index (public only; η is the form-side normal)

# Indices & markers
Bramble.indices(Ωₕ), Bramble.interior_indices(Ωₕ), Bramble.boundary_indices(Ωₕ), Bramble.is_boundary_index(Ωₕ, idx)
Bramble.index_in_marker(Ωₕ, :walls)      # BitVector mask for marker

# Mutation, in-place (mesh identity kept, only its points change)
Bramble.set_points!(Ωₕ, new_points), Bramble.change_points!(Ωₕ, new_points), iterative_refinement!(Ωₕ, ...)
```

## Discrete spaces & vector elements

```julia
Wₕ = gridspace(Ωₕ)                     # scalar space
Vₕ = Wₕ^2                              # composite vector space (or vector_gridspace(Ωₕ, 2))
ndofs(Wₕ), ncomponents(Vₕ), Bramble.weights(Wₕ)# DoFs, component count, quadrature weights

uₕ = element(Wₕ)                       # uninitialized
u_zero = element(Wₕ, 0.0)              # initialized to constant
parent(uₕ)                             # raw coefficient vector, write with copyto!

# Component indexing & slicing (zero-copy VectorElement views, mutate in-place)
vₕ = element(Vₕ)
uₓ, uᵧ = components(vₕ)              # N-tuple of scalar VectorElement views
uₓ = vₕ(1)                            # component 1 as a scalar VectorElement
uₓ .= 2.5                             # mutates parent vₕ in-place, zero allocations
Bramble.component_range(Vₕ, 1), Bramble.component_ranges(Vₕ)   # DoF slice(s)

reshape(uₕ), reshape(vₕ)               # D-dimensional array view(s) of flat coefficients
```
