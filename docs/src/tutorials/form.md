```@meta
CurrentModule = Bramble
```

# [Linear and bilinear forms](@id tutorial_form)

**What you will learn.** How to write a linear or bilinear form, assemble it into a vector or a matrix, impose Dirichlet conditions and solve a Poisson problem.

**What you need first.** The [mesh tutorial](mesh.md), the [space tutorial](space.md) for grid spaces and elements, and the [operators tutorial](operators.md) for `D₋ₓ`, `innerₕ` and `inner₊`.

**Where next.** [Coupled systems](coupled_systems.md), where one form addresses several unknowns.

The operators in the previous tutorial act on grid functions. A form is the other half: an
expression written in the *test* function, which Bramble assembles into the vector or the
matrix a solver wants. Every number below was produced by the code shown. The running
problem is the one-dimensional Poisson equation on the unit interval.

## A form is an expression in the test function

A linear form is a function of one argument, and that argument stands for the test function
rather than for any particular grid function:

```@example forms
using Bramble
using SparseArrays
import Bramble: allocate_system_matrix, evaluate!, reaction

Ωₕ = mesh(domain(interval(0.0, 1.0)), 33, true)
Wₕ = gridspace(Ωₕ)
fₕ = Rₕ(Wₕ, x -> sin(π * x))

l = form(Wₕ, v -> innerₕ(fₕ, v))
```

Nothing has been computed. `v` is symbolic, so `innerₕ(fₕ, v)` builds a description of

```math
\ell(v) = (f_h, v)_h
```

and the form stores the expression, not a vector. Building one is free, so it is reasonable
to write a form inside a function that is called repeatedly.

The *source* side is eager and the *test* side is symbolic: `D₋ₓ(fₕ)` computes a grid
function, while `D₋ₓ(v)` adds a node to an expression. A term is as cheap as its test side
is symbolic, however elaborate the coefficient in front of it.

## Assembling and refilling

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

### Live grid coefficients and dynamic scalars

A form keeps direct references to its coefficient arrays. Overwrite a grid function in place
with `Rₕ!(fₕ, ...)` or `parent(fₕ) .= ...` between steps, and the next `assemble!(b, l)`
reads the new values with no allocation and no rebuilt form.

A constant factor is written as a plain number, as in `2.5 * innerₕ(fₕ, v)`. A `Ref` is for
a scalar that changes across iterations:

```@example forms
α = Ref(1.0)
l_dyn = form(Wₕ, v -> α * innerₕ(fₕ, v))
b_dyn = assemble(l_dyn)
α[] = 2.0
assemble!(b_dyn, l_dyn)
sum(b_dyn) ≈ 2 * sum(b)
```

## Contracting without a vector

Often the vector is not wanted, only the number ``\ell(v_h)``. A form is callable:

```@example forms
oneₕ = Rₕ(Wₕ, x -> 1.0)
l(oneₕ), sum(b)
```

Against the all-ones grid function a linear form is the sum of its assembled vector. The
difference is that `l(oneₕ)` builds no vector and allocates nothing, so prefer it when the
result is a scalar.

`evaluate!` is the middle case. It takes a scratch vector, fills it with the assembled
vector and returns the contraction:

```@example forms
scratch = zeros(length(b))
evaluate!(scratch, l, oneₕ)
```

A form contracts against an element of its test space, never against a bare vector. The
length of a vector says nothing about whether its blocks match the components a form routes
to, so accepting one would make a coupled mismatch silent.

## Bilinear forms and the system matrix

A bilinear form takes two symbolic arguments, trial first and test second:

```@example forms
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v), :x))
A = assemble(a)
size(A), nnz(A)
```

This is the stiffness matrix of

```math
a(u, v) = (D_{-x} u, D_{-x} v)_{+x}.
```

Ninety-seven nonzeros in a 33-by-33 matrix is the tridiagonal band. The sparsity pattern
follows from the stencil, so it is known before any value is computed. That makes a two-step
idiom possible: `allocate_system_matrix` builds the pattern and nothing else, and `assemble!`
fills a matrix whose structure already exists:

```@example forms
A2 = allocate_system_matrix(a)
assemble!(A2, a)
A2 ≈ A
```

Inside a time loop, build the pattern once outside it and call `assemble!` within. Refilling
a matrix of fixed pattern allocates nothing, where `assemble` allocates a new matrix every
step.

`a` pairs the same `D₋ₓ` on both sides, so it is symmetric. The weight `inner₊ₓ` carries is
positive, so it is also positive semi-definite. `issymmetric` and `isposdef` answer from the
expression alone, without assembling anything:

```@example forms
using LinearAlgebra: issymmetric, isposdef
issymmetric(a), isposdef(a)
```

```@example forms
c = form(Wₕ, Wₕ, (u, v) -> inner₊(u, ∇ₕ(v)))
issymmetric(c)  # different operators either side
```

A positive answer says `cholesky` is worth trying on the result rather than a general
factorization. The check costs a few nanoseconds, against tens of microseconds to assemble
even this small a matrix.

## [Dirichlet conditions and a Poisson problem](@id form_dirichlet)

Boundary conditions come in two pieces, because a matrix and a right-hand side need
different things done to them. Labels on the domain say where:

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

[`dirichlet_constraints`](@ref) records the values and [`dirichlet_bc!`](@ref) applies them:
to the matrix by replacing the constrained rows, and to the vector by writing the boundary
values in. `dirichlet_constraints` takes the mesh, a `Domain` or a grid space directly.

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

!!! tip "Try this"
    Change the `33` in `mesh(Ω, 33, true)` to `65` and run the block again. The error falls
    by about a factor of four, as second order predicts.

Imposing conditions by replacing rows destroys symmetry, and a symmetric solver wants it
back. [`symmetrize!`](@ref) moves the constrained columns onto the right-hand side, which
restores symmetry and leaves the solution unchanged:

```@example forms
issymmetric(ad)          # true: the form is symmetric before any boundary condition
```

```@example forms
issymmetric(Matrix(Ad))  # false: dirichlet_bc! zeroed rows, not columns
```

```@example forms
symmetrize!(Ad, bd, Ωd, :left, :right)
issymmetric(Matrix(Ad))  # true again, and the solution above is unchanged
```

`issymmetric(ad)` is a claim about the expression `ad`, not about any matrix assembled from
it. That is why the first line answers `true` while the second answers `false`.

To constrain one block of a coupled system and leave another free, see
[Coupled systems](@ref coupled_one_block).

### Boundary fluxes

By the time `uh` exists, `dirichlet_bc!` has overwritten the constrained rows of `Ad`, and
the flux they carried is gone. [`reaction`](@ref) recovers it. It reassembles the
unconstrained `ad` and `ld` and reads the flux off the residual `A*uh - F`, which is
approximately zero on every free row and, on a constrained row, exactly the flux the
condition had to supply:

```@example forms
uh_elt = element(Wd, uh)
reaction(ad, ld, uh_elt; marker = :left), reaction(ad, ld, uh_elt; marker = :right)
```

Both come out near ``\pi``. For ``u = \sin(\pi x)`` the flux ``-u'`` leaving the domain is
``\pi`` at each end, and the two together recover the net source
``\int_0^1 \pi^2 \sin(\pi x)\,dx = 2\pi`` to round-off, at any resolution. The sign
convention is in the docstring of [`reaction`](@ref); [`reaction_density`](@ref) gives the
pointwise quantity, suitable for [`export_vtk`](@ref).

## Threading

Assembly reads the execution policy off the space it is given. A space whose backend says
`Parallel()` assembles on several threads, and nothing about the call changes. The
[backend tutorial](backend.md) covers choosing the policy and the measured crossovers where
threading starts to pay. The [internals page](../internals/form.md) describes the colouring
that keeps the threads from writing to the same entry.

## Reference

### Point sources

A continuous source ``f(x)`` is a density, integrated over cell volumes as
``\ell(v) = (f, v)_h``. A point source is a singular functional, the Dirac distribution
``S\,\delta(x - x_0)``:

```math
\ell(v) = S \, v(x_0).
```

Writing [`dirac`](@ref)`(x0, strength)` inside `innerₕ` or `inner₊` evaluates it without
cell-measure weighting:

```@example forms
l_pt = form(Wₕ, v -> innerₕ(dirac(0.5, 2.0), v))
sum(assemble(l_pt))
```

When ``x_0`` is a grid node, the strength ``S`` lands in that degree of freedom. When it
falls inside a cell, ``S`` is spread over the ``2^D`` surrounding vertices with multilinear
weights, and the total is conserved:

```@example forms
l_off = form(Wₕ, v -> innerₕ(dirac(0.35, 1.0), v))
sum(assemble(l_off)) ≈ 1.0
```

Several sources take a vector of coordinates and a vector of strengths. Each coordinate is a
tuple (a 1-tuple in 1D), because a bare vector of reals is read as the coordinates of one
point:

```@example forms
l_multi = form(Wₕ, v -> innerₕ(dirac([(0.2,), (0.7,)], [1.0, -1.0]), v))
sum(assemble(l_multi)) ≈ 0.0
```

A strength wrapped in a `Ref` updates live, without rebuilding the form:

```@example forms
S_live = Ref(1.0)
l_dyn_pt = form(Wₕ, v -> innerₕ(dirac(0.5, S_live), v))
b_dyn_pt = assemble(l_dyn_pt)
S_live[] = 3.5
assemble!(b_dyn_pt, l_dyn_pt)
sum(b_dyn_pt) ≈ 3.5
```

### Restricting a term to part of the mesh

`innerₕ`, `inner₊` and the directional products take a `markers` keyword, which restricts
the sum to the union of the regions the labels name:

```@example forms
a_left = form(Wd, Wd, (u, v) -> innerₕ(u, v; markers = (:left,)))
size(assemble(a_left))
```

Every mesh also carries `:boundary` and `:interior`, computed from its shape and needing no
label in `domain(...)`:

```@example forms
a_boundary = form(Wd, Wd, (u, v) -> innerₕ(u, v; markers = (:boundary,)))
size(assemble(a_boundary))
```

This is a masked *sum* of cell measures, not a surface integral. A masked `innerₕ` scales
like ``h`` and vanishes under refinement, where a true boundary integral does not. A Neumann
or Robin term needs the integral, which is [`inner_Γ`](@ref):

```@example forms
β, g = 1.7, x -> 2.0 + x[1]
a_robin = form(Wd, Wd, (u, v) -> inner_Γ(β * u, v; markers = (:xmax,)))
l_neumann = form(Wd, v -> inner_Γ(g, v; markers = (:xmax,)))
sum(assemble(l_neumann))
```

That is ``g(x_N)`` exactly, whatever the refinement. In 1D a face is a point of measure 1,
so the term is the endpoint pairing ``g(x_N)\,v(x_N)``. In 2D the same expression is an edge
integral weighted by the transverse half-spacing, and in 3D a face integral. `inner_Γ`
integrates over whole coordinate faces (`:boundary`, `:xmin` to `:zmax`, or a viewpoint
alias) and needs no marker in `domain(...)`.

A marker that exists nowhere the term reaches is an error, not a silent zero:

```@example forms
try
    assemble(form(Wd, Wd, (u, v) -> innerₕ(u, v; markers = (:nope,))))
catch e
    showerror(stdout, e)
end
```

On a composite space, a marker used without naming a component reaches every diagonal block
and must exist on every leaf. Write the term per component otherwise.

### The outward normal

A flux term ``\int_\Gamma \mathbf{F} \cdot \boldsymbol{\eta}\, v`` needs the outward normal
at each boundary point. [`η`](@ref) destructures and indexes into its per-coordinate
components like `∇ₕ`:

```@example forms
Ω2 = mesh(domain(box((0.0, 0.0), (1.0, 1.0)),
    :xmin => :xmin, :xmax => :xmax, :ymin => :ymin, :ymax => :ymax), (6, 6), (true, true))
W2 = gridspace(Ω2)
ηₓ, ηᵧ = η
F1 = Rₕ(W2, x -> 1.0 + x[1])
F2 = Rₕ(W2, x -> 2.0 + x[2])
l_flux = form(W2, v -> inner_Γ(F1 * ηₓ + F2 * ηᵧ, v; markers = (:xmax,)))
sum(assemble(l_flux))
```

### Automatic simplification

`form` resolves the expression once and simplifies it before storing it. The assembler
routes summands one at a time, so every top-level `+` is a separate sweep over the mesh, and
fewer summands means fewer sweeps.

Identical terms merge, so a form accumulated one physical effect at a time pays nothing for
repetition:

```@example forms
a_dup = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + innerₕ(u, v))
Bramble.resolve_form_ast(a_dup)     # 2 * innerₕ(u, v): one term, not two
```

```@example forms
Matrix(assemble(a_dup)) ≈ 2 .* Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))))
```

A zero-scaled term leaves nothing behind, so a coefficient can switch a term off without a
branch around the form:

```@example forms
a_off = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + 0 * inner₊(∇ₕ(u), ∇ₕ(v)))
Matrix(assemble(a_off)) ≈ Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))))
```

!!! tip "Try this"
    Replace `0 *` by `0.0 *` in `a_off` and compare `Bramble.resolve_form_ast(a_off)`.
    A `Float64` coefficient is only known at run time, so the rule does not fire.

Two limits apply. A rule whose outcomes have different node types fires only when the
compiler settles it from types, so a scalar coefficient must be an `Integer` (or the same
`Ref` on both terms) for the like-term and factoring rules to apply. A `Float64` known at run
time leaves the terms apart: same numbers, one extra sweep. And the pass stops at an inner
product's own arguments, so a scalar buried in a difference such as `innerₕ(D₋ₓ(2 * u), v)`
is invisible to it. Write `2 * innerₕ(D₋ₓ(u), v)` instead.

The [internals page on forms](../internals/form.md) documents the stencil algebra and the
exact rewrite rules.
