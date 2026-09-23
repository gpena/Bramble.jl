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
Locality
HostLocality
DeviceLocality
locality
ExecutionPolicy
CpuPolicy
CpuSerial
CpuThreaded
CpuBatch
GpuPolicy
GpuAsync
Serial
Parallel
execution_policy
vector
matrix
supports_undef_construction
vector_type
matrix_type
backend_types
backend_eye
backend_zeros
metal_sparse_csr
ka_device
gpu_backend
metal_backend
csr_backend
```

---

## Geometry

### Sets and intervals

```@docs
interval
point
box
×
dim
topo_dim
Base.extrema(::CartesianProduct, ::Integer)
center
projection
is_collapsed
point_type
boundary_symbols
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
host_points
half_points
half_point
spacing
forward_spacing
half_spacing
spacings
host_spacings
forward_spacings
half_spacings
host_half_spacings
hₘₐₓ
hₘᵢₙ
stepsize
locate_cell
normal_vector
cell_measure
cell_measures
is_uniform
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
host_weights
spaces
space
ncomponents
```

### Vector elements and grid functions

```@docs
VectorElement
element
Base.parent(::VectorElement)
Base.reshape(::VectorElement{<:ScalarGridSpace})
components
component_range
component_ranges
Base.:*(::Function, ::VectorElement)
ldiv!(::VectorElement, ::Factorization, ::AbstractVector)
```

### Restriction and averaging operators

```@docs
Rₕ
Rₕ!
avgₕ
avgₕ!
```

### Interpolation between grid spaces

Moving a grid function from one mesh to another — the piecewise (multi)linear interpolant,
named after [`Rₕ`](@ref)/[`Rₕ!`](@ref)'s own `Xₕ`/`Xₕ!` convention. One name, `πₕ`, with
methods that dispatch tells apart by what they are given rather than by different names:

- `πₕ(Wₕ, uₕ)` and [`πₕ!`](@ref)`(dest, src)` — the **numeric** operator, interpolating a grid
  function's values onto another space's mesh. [`interpolate_at`](@ref) is the single-point
  building block both are written in terms of, and [`interpolation_matrix`](@ref) is the same
  interpolant as a sparse matrix rather than applied pointwise.
- `πₕ(uₕ)` — the **symbolic source**, wrapping a grid function's interpolant as an AST leaf,
  composable with [`D₋ₓ`](@ref)/[`Mₓ`](@ref)/... inside [`innerₕ`](@ref). For the *known*
  side of a linear form.
- `πₕ(u)` over a **trial function** — the **bilinear operator**, contributing matrix columns
  rather than values. For the *unknown* side. It names no source space: that is the trial
  function's own, and assembly supplies it once the leaf is known
  ([#10](https://github.com/gpena/Bramble.jl/issues/10)).

See the [operators tutorial](tutorials/operators.md) for the numeric side and the pattern
this exists for: a heterogeneous composite space whose leaves live on different meshes.

```@docs
interpolate_at
πₕ!
πₕ
interpolation_matrix
```


## Difference, jump and average operators

The finite difference, the jump and the average, per coordinate and over every coordinate
at once. See the [operators tutorial](tutorials/operators.md).

Every family also takes the direction as an argument rather than as part of the name:
`D₋(uₕ, 2)`, `D₋(uₕ, :y)` and `D₋(uₕ, Val(2))` are all `D₋ᵧ(uₕ)`. That is what makes a
dimension-agnostic expression writable — `sum(innerₕ(D₋(uₕ, d), D₋(uₕ, d)) for d in 1:D)`
reads the same in 1D, 2D and 3D — and it costs nothing: the `Int` and `Symbol` forms branch
over literal `Val`s, so the direction still reaches the stencil engine as a compile-time
constant. The averages put this on `Mₕ`/`M₊ₕ` rather than on a bare `M`, which would take
the most common local name in finite-element code away from anyone writing `using Bramble`;
`Mₕ(uₕ)` is still the tuple over every coordinate and `Mₕ(uₕ, 2)` is the `y` average.

The same names carry the symbolic form: `D₋(uₕ, Val(1))` differences a grid function now,
`D₋(U, Val(1))` builds the AST node that will difference it during assembly. Inside a form
the direction must be a `Val`, since it is a type parameter of the node.

Three families are documented here but not exported, so `using Bramble` does not bring them
into scope and they are written `Bramble.D₊ₓ` or imported by name: the unscaled differences
`diff₋*`/`diff₊*`, the forward differences `D₊*`/`∇₊ₕ`, and the forward averages `M₊*`.
Bramble discretises with the backward operator paired with [`inner₊`](@ref), so the forward
ones are what the backward ones are built and checked against rather than what a form is
written with.

The unscaled differences (`diff₋ₓ` and its siblings) are the plain, undivided differences
these are built from, and are the one family of the three that is not even declared
`public`: they have no form-layer node, so they cannot appear inside a bilinear form, and in
a form the undivided forward difference is spelled [`jumpₓ`](@ref), which says which of the
two is meant.

```@docs
diff₋ₓ
diff₋ₓ!
diff₋ᵧ
diff₋ᵧ!
diff₋₂
diff₋₂!
diff₋ₕ
diff₊ₓ
diff₊ₓ!
diff₊ᵧ
diff₊ᵧ!
diff₊₂
diff₊₂!
diff₊ₕ
diff₋
diff₊
D₋ₓ
D₋ₓ!
D₋ᵧ
D₋ᵧ!
D₋₂
D₋₂!
∇ₕ
D₊ₓ
D₊ₓ!
D₊ᵧ
D₊ᵧ!
D₊₂
D₊₂!
∇₊ₕ
D₋
D₊
```

The forward difference over the averaged spacing, which is the one that satisfies
the discrete summation-by-parts identity
``(\overset{\times}{\textrm{D}}_{+x} u_h, v_h)_h = -(u_h, D_{-x} v_h)_{+x}`` for grid functions
`vₕ` vanishing on the boundary.

```@docs
D̽ₓ
D̽ₓ!
D̽ᵧ
D̽ᵧ!
D̽₂
D̽₂!
D̽ₕ
D̽
```

The centered difference, over the span its stencil covers. It reproduces the derivative
of an affine function exactly on any grid, and is skew-symmetric in `innerₕ` for grid
functions vanishing on the boundary.

```@docs
Dcₓ
Dcₓ!
Dcᵧ
Dcᵧ!
Dc₂
Dc₂!
Dcₕ
Dc
```

The cross-weighted centered difference, the same two one-sided differences weighted by
the opposite spacings. It reproduces the derivative of a quadratic exactly on any
grid, and so is second order on a non-uniform one where `Dcₓ` is first.

```@docs
Dₕₓ
Dₕₓ!
Dₕᵧ
Dₕᵧ!
Dₕ₂
Dₕ₂!
Dₕ
```

The vector calculus operators built on those differences: the divergence and the curl of a
vector field, and the conservative discrete Laplacian of a grid function. The unsubscripted
spellings use the backward differences, as [`∇ₕ`](@ref) does; `div₊ₕ` and `curl₊ₕ` are their
forward twins. [`εₕ`](@ref)/[`εₕ!`](@ref) are the discrete symmetric small-strain tensor,
over a composite `VectorElement` at runtime or, inside a [`form`](@ref), over a composite
trial or test function -- the same name spans both, dispatching on what it is given.

```@docs
divₕ
divₕ!
div₊ₕ
divcₕ
divcₕ!
curlₕ
curlₕ!
curl₊ₕ
curlcₕ
curlcₕ!
Δₕ
Δₕ!
εₕ
εₕ!
εcₕ
εcₕ!
∇cₕ
∇cₕ!
∇̽ₕ
∇̽ₕ!
div̽ₕ
div̽ₕ!
curl̽ₕ
curl̽ₕ!
ε₊ₕ
ε₊ₕ!
```

Jumps across an interface, ``\llbracket u \rrbracket = u_{i+1} - u_i``. There is one
of these rather than a forward and a backward pair: a jump belongs to the interface
between two cells, not to a direction of travel across it.

```@docs
jumpₓ
jumpₓ!
jumpᵧ
jumpᵧ!
jump₂
jump₂!
jumpₕ
jump
```

Averages of a point with its neighbour.

```@docs
Mₓ
Mₓ!
Mᵧ
Mᵧ!
M₂
M₂!
Mₕ
Mcₓ
Mcₓ!
Mcᵧ
Mcᵧ!
Mc₂
Mc₂!
Mcₕ
M₊ₓ
M₊ₓ!
M₊ᵧ
M₊ᵧ!
M₊₂
M₊₂!
M₊ₕ
```

## Inner products and norms

```@docs
innerₕ
inner_Γ
n
skew_symmetric
inner₊
inner₊ₓ
inner₊ᵧ
inner₊₂
normₕ
norm₁ₕ
snorm₁ₕ
norm₊
norminf_h
norm∞ₕ
```

---

## Forms

Linear and bilinear forms, their assembly, and the boundary conditions applied to an
assembled system. See the [forms tutorial](tutorials/form.md).

### Building a form

```@docs
form
expression
```

### Point (Dirac) sources

```@docs
dirac
DiracSource
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

### Matrix-free Kronecker operators

For a separable `BilinearForm` -- one whose assembled matrix is an exact sum of Kronecker
products of one-dimensional factors over a `MeshnD`, such as `innerₕ(u, v) +
inner₊(∇ₕ(u), ∇ₕ(v))` -- `kronecker_operator` builds a `KroneckerLinearOperator` that
applies by sum factorisation instead of ever assembling the `D`-dimensional matrix: a
`200^3` mesh stores `O(200)` numbers per axis rather than the assembled matrix's `O(200^3)`
stored entries ([#162](https://github.com/gpena/Bramble.jl/issues/162)). `is_separable`
checks the condition beforehand. `fdm_solve` requires `using Kronecker` (the
`BrambleKroneckerExt` extension, [#259](https://github.com/gpena/Bramble.jl/issues/259))
and solves a separable, constant-coefficient system by fast diagonalisation instead of a
general sparse factorisation.

```@docs
is_separable
kronecker_operator
KroneckerLinearOperator
fdm_solve
```

### Additive accumulation

`assemble_add!` adds a form's contribution to a matrix or vector that already holds
something, without the `fill!` `assemble!` does first -- for `M/Δt + θK`-style operators
built from several independently assembled pieces. See its own docstring for why this
needed no new traversal, and for the Dirichlet-ordering note.

```@docs
assemble_add!
```

### Jacobian sparsity

For a Newton residual built from a [`BilinearForm`](@ref) with a live nonlinear
coefficient (see [the nonlinear Poisson example](examples/poisson_nonlinear.md)),
`jacobian_pattern` reads the Jacobian's sparsity pattern directly off the form's AST,
without AD tracing. `ast_sparsity_detector` wraps it as an
`ADTypes.AbstractSparsityDetector`, ready to hand `AutoSparse` directly (requires
[ADTypes.jl](https://github.com/SciML/ADTypes.jl)).

```@docs
jacobian_pattern
ast_sparsity_detector
```

Time integration, differentiable linear solves, and the sparse/iterative solver backends
move to their own [scientific computing reference](api_sciml.md): the SciML stack
(`semidiscretize`, `ode_problem`, `linear_problem`, `nonlinear_problem`, ...), second-order
wave problems, `pde_solve`'s adjoint rule, AMG preconditioning, the SuiteSparse/Apple
Accelerate/MUMPS direct solvers, and `type_cached_assemble!`.

### Bandwidth analysis

`bandwidths`/`blockbandwidths` read a `BilinearForm`'s resolved AST alone, without
assembling anything, and answer what storage the assembled matrix would need: the plain
bandwidth in 1D, or the block/sub-block bandwidth pair a `D >= 2` mesh's blocked
lexicographic layout has ([#175](https://github.com/gpena/Bramble.jl/issues/175)).

```@docs
bandwidths
blockbandwidths
```

### Dirichlet conditions

```@docs
DirichletConstraint
dirichlet_constraints
dirichlet_bc!
symmetrize!
```

### Boundary flux / reaction extraction

`reaction` recovers the flux a Dirichlet constraint had to supply, from the
**unconstrained** operator and load (`assemble` with no `dirichlet` keyword) and an
already-solved `uₕ` -- the constrained assembly has already overwritten those rows by the
time a solution exists. `reaction_density` is its pointwise counterpart, a grid function
suitable for `export_vtk`.

```@docs
reaction
reaction_density
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
