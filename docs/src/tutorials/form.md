# Linear and bilinear forms

The operators in the previous tutorial act on grid functions. A form is the other half: an
expression written in the *test* function, which Bramble assembles into the vector or the
matrix a solver wants. Every number below was produced by the code shown.

## 1. A form is an expression in the test function

A linear form is a function of one argument, and that argument stands for the test function
rather than for any particular grid function:

```@example forms
using Bramble
using SparseArrays
import Bramble: CompositeGridSpace, allocate_system_matrix, evaluate!, assemble_parallel!, reaction

Ωₕ = mesh(domain(interval(0.0, 1.0)), 33, true)
Wₕ = gridspace(Ωₕ)
fₕ = Rₕ(Wₕ, x -> sin(π * x))

l = form(Wₕ, v -> innerₕ(fₕ, v))
```

Nothing has been computed. `v` is symbolic, so `innerₕ(fₕ, v)` builds a description of

```math
\ell(v) = (f_h, v)_h
```

and the form stores the expression, not a vector. Building one is free, which is what makes
it reasonable to write a form inside a function that is called repeatedly.

The asymmetry is worth naming early, because it decides what an expression costs. The
*source* side is eager and the *test* side is symbolic: `D₋ₓ(fₕ)` computes a grid function,
while `D₋ₓ(v)` adds a node to an expression. So a term is as cheap as its test side is
symbolic, however elaborate the coefficient in front of it.

## 2. Assembling, and refilling

`assemble` allocates the vector and fills it:

```@example forms
b = assemble(l)
length(b), sum(b)
```

In a time loop the allocation is the part worth avoiding. `assemble!` refills a vector that
already exists with zero allocations:

```@example forms
assemble!(b, l)
sum(b)
```

Forms store their resolved abstract syntax tree (`ast`) directly upon creation, retaining direct references to the underlying coefficient arrays.

### Live grid coefficients and dynamic scalars

- **Grid functions**: Overwrite a coefficient element in-place with `Rₕ!(fₕ, ...)` or `parent(fₕ) .= ...` between steps, and the next `assemble!(b, l)` evaluates the new values live with **0 bytes allocated**, without needing to reconstruct the form.
- **Scalar coefficients**: Constant scalar factors can be written directly as plain numbers (e.g. `2.5 * innerₕ(fₕ, v)`). A `Ref(val)` is only needed when you want a **dynamic scalar coefficient** that changes across loop iterations:

```@example forms
α = Ref(1.0)
l_dyn = form(Wₕ, v -> α * innerₕ(fₕ, v))
b_dyn = assemble(l_dyn)
α[] = 2.0
assemble!(b_dyn, l_dyn) # 0 bytes allocated, live 2x scaling
sum(b_dyn) ≈ 2 * sum(b)
```

### Point (Dirac) sources

A continuous source function ``f(x)`` is a density, integrated over cell volumes as
``\ell(v) = (f, v)_h \approx \sum_i |\square_i| \, f_i \, v_i``. A point source is a singular
functional:

```math
\ell(v) = S \, v(x_0)
```

representing a Dirac distribution ``S \, \delta(x - x_0)``.

Because a point evaluation acts directly on the test function rather than through cell volume
quadrature, writing [`dirac`](@ref)`(x0, strength)` inside `innerₕ` (or `inner₊`) evaluates this
functional without cell-measure weighting:

```@example forms
l_pt = form(Wₕ, v -> innerₕ(dirac(0.5, 2.0), v))
b_pt = assemble(l_pt)
sum(b_pt)
```

- **On-grid points**: When ``x_0`` coincides with a grid node, the exact strength ``S`` is
  placed directly in that degree of freedom.
- **Off-grid points**: When ``x_0`` falls inside a cell, ``S`` is distributed across the
  ``2^D`` surrounding cell vertices via multilinear interpolation weights
  ``w_c = \prod_d (1 - t_d) \text{ or } t_d``:

```@example forms
l_off = form(Wₕ, v -> innerₕ(dirac(0.35, 1.0), v))
b_off = assemble(l_off)
sum(b_off) ≈ 1.0 # exact conservation of total source strength
```

The sum of assembled entries ``\sum b_i = S`` is preserved to machine precision, and
evaluating the form against smooth grid functions contracts with ``\mathcal{O}(h^2)``
accuracy.

- **Superposition**: A vector of coordinates and intensities describes multiple point
  sources. Each coordinate is wrapped (a 1-tuple here, in 1D) rather than passed as a bare
  number, since a bare `AbstractVector{<:Real}` is read as the coordinates of one point in
  `length(x0)` dimensions, not a list of scalar points:

```@example forms
l_multi = form(Wₕ, v -> innerₕ(dirac([(0.2,), (0.7,)], [1.0, -1.0]), v))
b_multi = assemble(l_multi)
sum(b_multi) ≈ 0.0 # balanced dipole
```

- **Dynamic strengths in time loops**: Wrapping a point strength in a `Ref` allows live
  source updates without reallocating or rebuilding the form:

```@example forms
S_live = Ref(1.0)
l_dyn_pt = form(Wₕ, v -> innerₕ(dirac(0.5, S_live), v))
b_dyn_pt = assemble(l_dyn_pt)
S_live[] = 3.5
assemble!(b_dyn_pt, l_dyn_pt) # 0 bytes allocated
sum(b_dyn_pt) ≈ 3.5
```


## 3. Contracting without a vector

Often the vector is not wanted, only the number ``\ell(v_h)``. A form is callable, and takes
that shortcut:

```@example forms
oneₕ = Rₕ(Wₕ, x -> 1.0)
l(oneₕ), sum(b)
```

Against the all-ones grid function a linear form is the sum of its assembled vector, which is
the check above. The difference is that `l(oneₕ)` builds no vector: it contracts as it walks
the grid, and allocates nothing at all. Where the result is a scalar, prefer it.

`evaluate!` is the middle case: it wants the assembled vector *and* the number, so it takes
a scratch vector, fills it, and returns the contraction:

```@example forms
scratch = zeros(length(b))
evaluate!(scratch, l, oneₕ)
```

A form contracts against an element of its test space, never against a bare vector. The
length of a vector says nothing about whether its blocks line up with the components a form
routes to, so accepting one would make a coupled mismatch silent rather than loud.

## 4. Bilinear forms and the system matrix

A bilinear form takes two symbolic arguments, trial first and test second:

```@example forms
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v), :x))
A = assemble(a)
size(A), nnz(A)
```

which is the stiffness matrix of

```math
a(u, v) = (D_{-x} u, D_{-x} v)_{+x}.
```

Ninety-seven nonzeros in a 33-by-33 matrix is the tridiagonal band, and the band is the point:
the sparsity pattern follows from the stencil, so it is known before any value is computed.

That is what makes the two-step idiom worth using. `allocate_system_matrix` builds the pattern
and nothing else; `assemble!` then fills a matrix whose structure already exists:

```@example forms
A2 = allocate_system_matrix(a)
assemble!(A2, a)
A2 ≈ A
```

Inside a time loop, build the pattern once outside it and call `assemble!` within. Refilling
a matrix whose pattern is fixed allocates nothing, where `assemble` allocates a new matrix
every step.

`a` above is `innerₕ(L(u), L(v))` with the same `D₋ₓ` on both sides, which is symmetric,
and, since the quadrature weight `inner₊ₓ` carries is positive, positive semi-definite,
purely by that construction. `issymmetric`/`isposdef` answer this from the expression alone,
without assembling anything:

```@example forms
using LinearAlgebra: issymmetric, isposdef
issymmetric(a), isposdef(a)
```

```@example forms
c = form(Wₕ, Wₕ, (u, v) -> inner₊(u, ∇ₕ(v)))
issymmetric(c)  # different operators either side, not this pattern
```

Knowing this before assembling is what makes a positive answer worth something: it says
`cholesky` is worth trying on the result rather than a general factorization, at a cost of
a few nanoseconds, against tens of microseconds to assemble even this small a matrix, close
enough to free that there is no reason not to check.

## 5. Dirichlet conditions, and a Poisson problem

Boundary conditions come in two pieces, because a matrix and a right-hand side need different
things done to them. Labels on the domain say where:

```@example forms
Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)
Ωd = mesh(Ω, 33, true)
Wd = gridspace(Ωd)

fd = Rₕ(Wd, x -> π^2 * sin(π * x))
ad = form(Wd, Wd, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v), :x))
ld = form(Wd, v -> innerₕ(fd, v))

Ad = assemble(ad)
bd = assemble(ld)
nothing # hide
```

`dirichlet_constraints` records the values, `dirichlet_bc!` applies them: to the matrix by
replacing the constrained rows, and to the vector by writing the boundary values in.
`dirichlet_constraints` takes the mesh (or a `Domain`/grid space) directly; no need to
extract the underlying `CartesianProduct` first:

```@example forms
bcs = dirichlet_constraints(Ωd, :left => (x -> 0.0), :right => (x -> 0.0))
dirichlet_bc!(Ad, Ωd, :left, :right)
dirichlet_bc!(bd, Ωd, bcs, :left, :right)
nothing # hide
```

Solving ``-u'' = \pi^2 \sin(\pi x)`` with ``u(0) = u(1) = 0`` gives ``u = \sin(\pi x)``:

```@example forms
uh = Ad \ bd
exact = Rₕ(Wd, x -> sin(π * x))
maximum(abs, uh .- parent(exact))
```

Eight parts in ten thousand on 33 points, which is second order behaving itself.

Imposing conditions by replacing rows destroys symmetry, and a symmetric solver will want it
back. `symmetrize!` moves the constrained columns onto the right-hand side, restoring
symmetry and leaving the solution unchanged:

```@example forms
issymmetric(ad)          # true: the form is symmetric by construction, before any boundary condition
```

```@example forms
issymmetric(Matrix(Ad))  # false: dirichlet_bc! zeroed rows, not columns
```

```@example forms
symmetrize!(Ad, bd, Ωd, :left, :right)
issymmetric(Matrix(Ad))  # true again, and the solution above is unchanged
```

`issymmetric(ad)` is a claim about the expression `ad`, not about any one matrix that gets
assembled from it: it says nothing about what `dirichlet_bc!` alone leaves behind, which is
exactly why the middle line above answers `false` even though the first one answers `true`.

### Boundary fluxes: `reaction`

By the time `uh` exists, `dirichlet_bc!` has already overwritten `Ad`'s constrained rows;
the flux information they carried is gone. [`reaction`](@ref) recovers it by reassembling
the *unconstrained* `ad`/`ld` (`assemble` with no `dirichlet` keyword) and reading the flux
straight off the residual `A*uh - F` there, which is `≈ 0` on every unconstrained row and,
on a constrained one, exactly the flux the condition had to supply:

```@example forms
uh_elt = element(Wd, uh)
reaction(ad, ld, uh_elt; marker = :left), reaction(ad, ld, uh_elt; marker = :right)
```

Both come out `≈ π`: for `u = sin(πx)`, the flux `-u'` leaving the domain is `π` at each end,
and the two together recover the net source `∫₀¹ π² sin(πx) dx = 2π` to round-off, regardless
of mesh resolution. See [`reaction`](@ref)'s own docstring for the sign convention and
[`reaction_density`](@ref) for the pointwise quantity, suitable for [`export_vtk`](@ref).

## 6. Coupled systems

A composite space stacks copies of a space, and a form over one addresses its blocks by
component: `u[1]` (or `u(1)`) is the trial function of the first block, `v[2]` (or `v(2)`) the test function of the
second. Both functor indexing `u(i)` and standard bracket indexing `u[i]` are supported on trial and test functions,
as well as on compound operators (`(D₋ₓ(u))[i]`).

In addition, trial and test functions support tuple destructuring via [`components`](@ref) or direct iteration:

```@example forms
Vₕ = Wₕ^Val(2)
ac = form(Vₕ, Vₕ, (u, v) -> begin
    u₁, u₂ = components(u)
    v₁, v₂ = components(v)
    innerₕ(u₁, v₁) + inner₊(∇ₕ(u₂), ∇ₕ(v₂), :x)
end)
Ac = assemble(ac)
size(Ac)
```

Sixty-six by sixty-six: two blocks of 33, assembled into one matrix. A term naming `u[i]` and
`v[j]` lands in block ``(j, i)``, so off-diagonal coupling is written the same way:
`innerₕ(u[1], v[2])` fills the block that couples the first unknown to the second equation.

Component indices are checked against the number of blocks at form construction time: accessing `u[3]`
or `u(3)` on a 2-component space raises an immediate `ArgumentError`. A term must name both
components or neither:

```@example forms
try
    form(Vₕ, Vₕ, (u, v) -> innerₕ(u[1], v))
catch e
    println(e)
end
```

Naming one and leaving the other open has no reading as mathematics: the term would belong
to every equation at once, so it is refused rather than guessed at. Naming neither is fine
and means the diagonal, applied to every block.

### Constraining one block, leaving another free

`dirichlet` on its own binds to every leaf sharing the named marker. That is fine when every
block wants the same treatment, not when they don't. A Stokes-style system prescribing
velocity while leaving pressure unconstrained needs `dirichlet_components` too: 1-based leaf
positions, the same order `u(1)`/`u(2)` addressing already uses.

```@example forms
Ωc = domain(interval(0.0, 1.0), :left => :left, :right => :right)
Ωdc = mesh(Ωc, 21, true)
Vc = gridspace(Ωdc)^Val(2)           # 1: velocity-like, 2: pressure-like
ac2 = form(Vc, Vc, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
Ac2 = assemble(ac2; dirichlet = (:left, :right), dirichlet_components = 1)
nothing # hide
```

Block 1 (rows `1:21`) has its boundary rows pinned; block 2 is untouched, still the plain
assembled operator, no rows replaced at all. Leaving `dirichlet_components` at its default
(`nothing`) applies the labels to every leaf, exactly as before this keyword existed; call
`assemble!`/`dirichlet_bc!` again with a different `dirichlet`/`dirichlet_components`
pair to constrain another block differently.

### Interpolating between the leaves of a heterogeneous composite space

The composite spaces above stack copies of *one* space: every leaf shares a mesh. A
composite space can also be built directly from a tuple of leaves over different meshes,
and then a term coupling two leaves needs a way to move a value from one leaf's grid to
the other's: [`πₕ`](@ref), applied to the trial function alone (see the
[operators tutorial](operators.md) for the numeric side and a diagram of the interpolant
itself); the same name as the numeric `πₕ`/[`πₕ!`](@ref) pair, told apart by dispatch
rather than a different one. It names no source space: the space it interpolates from is
the trial function's own, and assembly supplies it once the leaf is known.

```@raw html
<figure>
<svg viewBox="0 0 720 170" width="100%" style="max-width:640px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="A grid function on the small leaf is wrapped by pi-h into a source, which composes with D-x the same way any other source does, and is assembled by inner-h into the big leaf's block.">
  <defs>
    <marker id="arrowFlow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="currentColor"/>
    </marker>
  </defs>

  <rect x="10"  y="55" width="160" height="60" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="90" y="80" font-size="12" font-weight="bold" fill="currentColor" text-anchor="middle">uₕ on Wsmall</text>
  <text x="90" y="98" font-size="11" fill="currentColor" opacity="0.75" text-anchor="middle">u(2), the small leaf</text>

  <path d="M 175 85 L 225 85" stroke="currentColor" stroke-width="2" marker-end="url(#arrowFlow)"/>

  <rect x="230" y="45" width="190" height="80" rx="8" fill="none" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="325" y="70" font-size="12" font-weight="bold" fill="#8b5cf6" text-anchor="middle">πₕ(u(2))</text>
  <text x="325" y="88" font-size="11" fill="currentColor" opacity="0.75" text-anchor="middle">a SourceFunction:</text>
  <text x="325" y="103" font-size="11" fill="currentColor" opacity="0.75" text-anchor="middle">composes with D₋ₓ, Mₓ, ...</text>

  <path d="M 425 85 L 475 85" stroke="currentColor" stroke-width="2" marker-end="url(#arrowFlow)"/>

  <rect x="480" y="45" width="230" height="80" rx="8" fill="none" stroke="#10b981" stroke-width="1.5"/>
  <text x="595" y="68" font-size="12" font-weight="bold" fill="#10b981" text-anchor="middle">innerₕ(πₕ(u(2)), v(1))</text>
  <text x="595" y="86" font-size="11" fill="currentColor" opacity="0.75" text-anchor="middle">a LinearProduct: assembled</text>
  <text x="595" y="101" font-size="11" fill="currentColor" opacity="0.75" text-anchor="middle">into Wbig's block, leaf 1</text>
</svg>
</figure>
```

`πₕ(uₕ)` reads exactly like any other source: it is one, an AST leaf wrapping
`x -> interpolate_at(uₕ, x)`, so it composes with `D₋ₓ`, `Mₓ`, and the rest the same way
`sin`, a `VectorElement`, or any other source does, and can sit on the left of `innerₕ`
inside a coupled form:

```@example forms
Ωbig = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (8, 8), (true, true))
Ωsmall = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (4, 4), (true, true))
Wbig, Wsmall = gridspace(Ωbig), gridspace(Ωsmall)
Vh = CompositeGridSpace((Wbig, Wsmall))
uv = Rₕ(Vh, (x -> 0.0, x -> x[1] + x[2]))   # only the small leaf (2) carries data

lh = form(Vh, v -> innerₕ(πₕ(uv(2)), v(1)) + innerₕ(∇ₕ[:x](πₕ(uv(2))), ∇ₕ[:x](v(1))))
b = assemble(lh)

# the differenced term is not a no-op: dropping it changes the answer
b_plain = assemble(form(Vh, v -> innerₕ(πₕ(uv(2)), v(1))))
maximum(abs, b .- b_plain)
```

The two terms land in the same block (leaf 1, `Wbig`) even though the source they read
from lives on leaf 2's own, coarser mesh: `πₕ` is what makes that a well-posed
expression rather than a size mismatch. This is exactly what makes a heterogeneous
composite space useful for more than indexing: leaf 2 can represent one field at a
resolution the problem calls for, and a term over leaf 1 can still read it.

That last line is the check worth keeping, not `length(b) == ndofs(Vh)`: a differenced
source whose offsets are discarded assembles to exactly zero, and a zero vector has the
right length and is perfectly finite.

An operated source is worth a word on what it means. `innerₕ(D₋ₓ(f), v)` is
``\sum_i |\square_i| \, (D_{-x}f)_i \, v_i``: the operator acts on the *source*, producing
another grid function, which is then integrated against the test function. It agrees entry
for entry with applying the numeric operator first:
`assemble(form(Wₕ, v -> innerₕ(D₋ₓ(fₕ), v)))` equals
`parent(D₋ₓ(fₕ)) .* weights(Wₕ, Innerh())`. That equivalence is what
`test/form/source_operators.jl` pins, for every operator, against the numeric layer.

A *bilinear* term coupling two leaves over different meshes is a different matter, and it is
refused:

```@example forms
try
    assemble(form(Vh, Vh, (u, v) -> innerₕ(u(2), v(1))))
catch e
    println(e)
end
```

(The refusal is raised when the matrix is built, not when the form is written: `form` resolves
the expression, and which leaves a term couples is a question about the spaces it is assembled
against. `allocate_system_matrix` refuses it too, so neither entry point can be reached
around.)

A coupled block is assembled by walking the test leaf's grid and reading the trial column out
of that same index space, so it needs the two leaves to agree on what an index means. Two
leaves over meshes of different sizes do not: index `(3, 3)` on an 8×8 grid and on a 4×4 grid
name different points, and nothing in the term says how to get from one to the other. So there
is no assembly to give, and the error says so rather than guessing: in one direction it used
to overrun the trial block and throw from deep inside `sparse!`, and in the other it quietly
filled in-range but wrong columns.

Coupling leaves that *share* a mesh is unaffected, which is every composite space built by
repeating one space (`Wₕ^Val(2)`), including off-diagonal blocks.

## 7. Threading, chosen once on the backend

Assembly reads the execution policy off the space it is given, so `assemble!`/`assemble`
thread when the backend says `Parallel()` and nothing about the call changes. The
[backend tutorial](backend.md) covers building a backend and choosing the policy.

```@example forms
Wₕ_par = gridspace(mesh(domain(interval(0.0, 1.0)), 33, true;
    backend = backend(policy = Parallel())))

l_par = form(Wₕ_par, v -> innerₕ(Rₕ(Wₕ_par, x -> sin(π * x)), v))
b_par = assemble(l_par)      # threads, because Wₕ_par's backend says Parallel()

execution_policy(Wₕ_par)
```

`assemble_parallel!` is the lower-level entry point that threads whatever the backend says,
for a forced comparison or a benchmark rather than everyday use:

```@example forms
bp = similar(b)
assemble_parallel!(bp, l)

bp ≈ b
```

No locks are involved. Assembly partitions the grid by stride: two points further apart than
the stencil's own footprint cannot write to the same entry, so points sharing a stride are
written concurrently. A term whose test argument carries no difference has stride 1 in every
direction and sweeps the grid in one pass; a 2D gradient term needs four colours. The
[internals page](../internals/form.md) has the colouring itself, and the
[benchmarks](../benchmarks.md) measure when threading pays, which is what should decide the
policy.

## 8. Restricting a term to part of the mesh

`innerₕ`, `inner₊` and the directional products all take a `markers` keyword, restricting the
sum to the union of the regions the labels name; the same idea as `restrict_to`, spelled at
the call site rather than wrapping an argument:

```@example forms
a_left = form(Wd, Wd, (u, v) -> innerₕ(u, v; markers = (:left,)))
size(assemble(a_left))
```

Every mesh also carries `:boundary` and `:interior` automatically, computed from its own
shape rather than needing any label set up in `domain(...)`:

```@example forms
a_boundary = form(Wd, Wd, (u, v) -> innerₕ(u, v; markers = (:boundary,)))
size(assemble(a_boundary))
```

This is a masked *sum* of the existing cell measures, not a surface integral, and the two
are not interchangeable; a masked `innerₕ` scales like `h` and vanishes under refinement,
where a true boundary integral does not. `markers` is for the former; a Neumann or Robin
term needs the latter, which is [`inner_Γ`](@ref):

```@example forms
# a Robin boundary mass and a Neumann flux, on the right end of this 1D mesh
β, g = 1.7, x -> 2.0 + x[1]
a_robin = form(Wd, Wd, (u, v) -> inner_Γ(β * u, v; markers = (:xmax,)))
l_neumann = form(Wd, v -> inner_Γ(g, v; markers = (:xmax,)))
sum(assemble(l_neumann))
```

That is `g(x_N)` exactly, and it stays what it is under refinement, which is what makes it a
surface integral rather than a masked sum. In 1D a ``(D-1)``-face is a *point*, of measure 1,
so the term is the plain endpoint pairing `g(x_N) v(x_N)` with no spacing in it; in 2D the
same expression gives an edge integral, weighing each point of the edge by its transverse
half-spacing, and in 3D a face integral.

`inner_Γ` integrates over whole coordinate faces — `:boundary`, `:xmin`…`:zmax`, or a
viewpoint alias — and needs no marker declared in `domain(...)`, since it reads the face from
the point's index rather than from a marker table.

### The outward normal, `η`

A flux term ``\int_\Gamma \mathbf{F} \cdot \boldsymbol{\eta}\, v`` needs the outward normal at
each boundary point. `η` is exported for this; like `∇ₕ`, it destructures and indexes into
its per-coordinate components:

```@example forms
Ω2 = mesh(domain(box((0.0, 0.0), (1.0, 1.0)),
    :xmin => :xmin, :xmax => :xmax, :ymin => :ymin, :ymax => :ymax), (6, 6), (true, true))
W2 = gridspace(Ω2)
ηₓ, ηᵧ = η
F1 = Rₕ(W2, x -> 1.0 + x[1])
F2 = Rₕ(W2, x -> 2.0 + x[2])
l_flux = form(W2, v -> inner_Γ(F1 * ηₓ + F2 * ηᵧ, v; markers = (:xmax,)))
b_flux = assemble(l_flux)
sum(b_flux)
```

`F1 * ηₓ + F2 * ηᵧ` is the same quantity `dot((F1, F2), η)` computes with `LinearAlgebra.dot`,
component by component; the destructured form is what reads naturally inside a form.

A marker that does not exist anywhere the term reaches is a loud error rather than a silent
all-zero contribution: `RegionRestriction`'s own per-point check cannot tell "nothing here
is marked" from "no such marker", so this is caught once, before assembling anything:

```@example forms
try
    assemble(form(Wd, Wd, (u, v) -> innerₕ(u, v; markers = (:nope,))))
catch e
    println(e)
end
```

On a composite space, a marker used without naming a component reaches every diagonal block,
and has to exist on every leaf that reaches: write the term per component, each with its own
markers, if it does not.

## 9. What `form(...)` simplifies automatically

`form` resolves the expression once and runs an algebraic simplification pass before storing
it. This decides how a form is worth writing: the assembler routes summands one at a time, so
every top-level `+` is a separate sweep over the mesh, and fewer summands means the same
matrix assembled in fewer sweeps.

Identical terms merge, so a form accumulated one physical effect at a time costs nothing for
the repetition:

```@example forms
a_dup = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + innerₕ(u, v))
Bramble.resolve_form_ast(a_dup)     # 2 * innerₕ(u, v): one term, not two
```

```@example forms
Matrix(assemble(a_dup)) ≈ 2 .* Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))))
```

A zero-scaled term leaves nothing behind, which is what lets a coefficient switch a term off
without a branch around the form:

```@example forms
a_off = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + 0 * inner₊(∇ₕ(u), ∇ₕ(v)))

Matrix(assemble(a_off)) ≈ Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))))
```

Two limits are worth knowing while writing a form. A rule whose two outcomes have different
node types only fires when the compiler can settle it from the types alone, so a scalar
coefficient has to be an `Integer` (or the same `Ref` on both terms) for the like-term and
factoring rules to apply; a `Float64` coefficient known at run time leaves the terms apart,
with the same numbers and one extra sweep. And the pass stops at an inner product's own
arguments: a scalar buried inside a difference, as in `innerₕ(D₋ₓ(2 * u), v)`, is invisible
to it. Write it where the pass can see it, `2 * innerₕ(D₋ₓ(u), v)`.

## Where to go next

The [internals page on forms](../internals/form.md) documents the colouring and the stencil
algebra underneath all of this, including how the matrix path colours on the test-side span
alone, and the exact rewrite rules behind §9's automatic simplification.
