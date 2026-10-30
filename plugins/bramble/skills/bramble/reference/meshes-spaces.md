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

`change_points!(Ωₕ, pts)` and `set_points!` throw an `ArgumentError` on a mesh with a label
`set_markers!` built from a predicate of position (`:c => x -> x[1] < 0.5`): the set would stay
stale. Use `change_points!(Ωₕ, domain_markers, pts)`, which sets the markers again; face labels are fine.

## Polygon and level-set domains (2D)

`polygon(V)` and `levelset(φ)` are exported; `Bramble.signed_distance`, `Bramble.polygon_spaces`
and `Bramble.lift` are `public`. Either is a marker value: `:P => polygon(V)` makes `mesh` close
the grid lines (every line meeting an edge or the curve gets a node on it, so the count can
exceed the request) and label the nodes `:P` (interior), `:P_boundary`, `:P_exterior`.
`polygon_spaces(Ωₕ, :P, g)` returns trial and test spaces whose unknowns are the interior nodes;
the nodes on the boundary carry the Dirichlet data `g`, so `assemble` takes no `dirichlet`.
`lift(U, x)` puts a solution back on the whole grid.

```julia
using Bramble
using Bramble: polygon_spaces, lift, signed_distance, D₋ₓ, D₋ᵧ, D̃ₓ, D̃ᵧ

V = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]  # L-shape
Ω = domain(interval(0.0, 2.0) × interval(0.0, 2.0), :P => polygon(V))
Ωₕ = mesh(Ω, (21, 21), (true, true))
U, W = polygon_spaces(Ωₕ, :P, p -> p[1]^2 + p[2]^2)
fₕ = element(gridspace(Ωₕ), -4.0)
A, F = assemble(form(U, W, (u, v) -> innerₕ(-D̃ₓ(D₋ₓ(u)) - D̃ᵧ(D₋ᵧ(u)), v)),
                form(W, v -> innerₕ(fₕ, v)))
uₕ = lift(U, A \ F)                       # whole-grid element
signed_distance(polygon(V), (0.5, 0.5))   # -0.5: negative inside, positive outside
```

`levelset(φ)`: `φ(x)` returns a real (not a `Bool`), negative inside, zero on the curve, and
need not be a signed distance; the labels are `:D`, `:D_boundary`, `:D_exterior` for
`:D => levelset(φ)`. `polygon_spaces(Ωₕ, :D, g)` works the same. Traps: polygon vertices are
validated (at least three, no zero-length edge, simple boundary); `polygon_spaces` refuses
composite spaces; a polygon marker needs a 2D mesh.

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

## Ghost points and constrained spaces

`gridspace(Ωₕ; ghosts = :boundary)` (or a tuple of facet names, `ghosts = (:xmax,)`) adds one
layer of ghosts outside those facets. `constrained(Wₕ, u -> (:F => operator => data, ...))`
closes it: each relation (`∂ₙ`, `∂ₙₙ`, `jumpₙ`, or `u` for Dirichlet) determines the ghosts of
its facet, and the constrained space `C` carries `E` and `d` with `u_pad = E u + d`.

```julia
Wₕ = gridspace(Ωₕ; ghosts = :boundary)
C = constrained(Wₕ, u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => u => g))
vₕ = Bramble.lift(C, uₕ)                     # E u + d on the padded space; lift is public
```

- `lift` is `public`, not exported: write `Bramble.lift` (`lift!`, `update_ghosts!` likewise
  for the in-place and time-loop forms; `GhostVector` and `update_ghosts!` are exported).
- A ghosted facet with only `u => g` is refused: its ghosts would be unknowns no equation
  reads. Give it a ghost relation too, or drop it from `ghosts`.
- The same label given as `u => g` in `constrained` and as `dirichlet` in `assemble` is an
  `ArgumentError`.
- `Rₕ` and `avgₕ` refuse on a ghost space (an `ArgumentError` naming `gridspace(Ωₕ)` and
  `lift`): restrict on the plain space, then lift.
- Data in time, `:F => u => g(x, t)`: `lift(C, u, t)` and `semidiscretize`; the calls
  without `t` refuse.
