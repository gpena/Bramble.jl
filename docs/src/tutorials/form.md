# Linear and bilinear forms

The operators in the previous tutorial act on grid functions. A form is the other half: an
expression written in the *test* function, which Bramble assembles into the vector or the
matrix a solver wants.
This tutorial covers:

1. Writing a linear form and assembling its vector.
2. Refilling that vector inside a time loop, and why the pattern is built once.
3. Contracting a form against a grid function without building a vector at all.
4. Bilinear forms and the system matrix.
5. Imposing Dirichlet conditions, and solving a Poisson problem.
6. Coupled systems, where a term names which block it belongs to.
7. Threaded assembly, and why it needs no locks.

Every number below was produced by the code shown.

## 1. A form is an expression in the test function

A linear form is a function of one argument, and that argument stands for the test function
rather than for any particular grid function:

```@example forms
using Bramble
using SparseArrays

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

- **Grid functions**: Overwrite a coefficient element in-place with `Rₕ!(fₕ, ...)` or `values(fₕ) .= ...` between steps, and the next `assemble!(b, l)` evaluates the new values live with **0 bytes allocated**, without needing to reconstruct the form.
- **Scalar coefficients**: Constant scalar factors can be written directly as plain numbers (e.g. `2.5 * innerₕ(fₕ, v)`). A `Ref(val)` is only needed when you want a **dynamic scalar coefficient** that changes across loop iterations:

```@example forms
α = Ref(1.0)
l_dyn = form(Wₕ, v -> α * innerₕ(fₕ, v))
b_dyn = assemble(l_dyn)
α[] = 2.0
assemble!(b_dyn, l_dyn) # 0 bytes allocated, live 2x scaling
sum(b_dyn) ≈ 2 * sum(b)
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

`evaluate!` is the middle case — it wants the assembled vector *and* the number, so it takes
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
a = form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
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

`a` above is `innerₕ(L(u), L(v))` with the same `D₋ₓ` on both sides, which is symmetric —
and, since the quadrature weight `inner₊ₓ` carries is positive, positive semi-definite —
purely by that construction. `issymmetric`/`isposdef` answer this from the expression alone,
without assembling anything:

```@example forms
using LinearAlgebra: issymmetric, isposdef
issymmetric(a), isposdef(a)
```

```@example forms
c = form(Wₕ, Wₕ, (u, v) -> inner₊(u, D₋ₓ(v)))
issymmetric(c)  # different operators either side — not this pattern
```

Knowing this before assembling is what makes a positive answer worth something: it says
`cholesky` is worth trying on the result rather than a general factorization, at a cost —
a few nanoseconds, against tens of microseconds to assemble even this small a matrix — close
enough to free that there is no reason not to check.

## 5. Dirichlet conditions, and a Poisson problem

Boundary conditions come in two pieces, because a matrix and a right-hand side need different
things done to them. Labels on the domain say where:

```@example forms
Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)
Ωd = mesh(Ω, 33, true)
Wd = gridspace(Ωd)

fd = Rₕ(Wd, x -> π^2 * sin(π * x))
ad = form(Wd, Wd, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
ld = form(Wd, v -> innerₕ(fd, v))

Ad = assemble(ad)
bd = assemble(ld)
nothing # hide
```

`dirichlet_constraints` records the values, `dirichlet_bc!` applies them — to the matrix by
replacing the constrained rows, and to the vector by writing the boundary values in:

```@example forms
bcs = dirichlet_constraints(set(Ωd), :left => (x -> 0.0), :right => (x -> 0.0))
dirichlet_bc!(Ad, Ωd, :left, :right)
dirichlet_bc!(bd, Ωd, bcs, :left, :right)
nothing # hide
```

Solving ``-u'' = \pi^2 \sin(\pi x)`` with ``u(0) = u(1) = 0`` gives ``u = \sin(\pi x)``:

```@example forms
uh = Ad \ bd
exact = Rₕ(Wd, x -> sin(π * x))
maximum(abs, uh .- values(exact))
```

Eight parts in ten thousand on 33 points, which is second order behaving itself.

Imposing conditions by replacing rows destroys symmetry, and a symmetric solver will want it
back. `symmetrize!` moves the constrained columns onto the right-hand side, restoring
symmetry and leaving the solution unchanged:

```@example forms
issymmetric(ad)          # true — the form is symmetric by construction, before any boundary condition
```

```@example forms
issymmetric(Matrix(Ad))  # false — dirichlet_bc! zeroed rows, not columns
```

```@example forms
symmetrize!(Ad, bd, Ωd, :left, :right)
issymmetric(Matrix(Ad))  # true again, and the solution above is unchanged
```

`issymmetric(ad)` is a claim about the expression `ad`, not about any one matrix that gets
assembled from it — it says nothing about what `dirichlet_bc!` alone leaves behind, which is
exactly why the middle line above answers `false` even though the first one answers `true`.

## 6. Coupled systems

A composite space stacks copies of a space, and a form over one addresses its blocks by
index: `u(1)` is the trial function of the first block, `v(2)` the test function of the
second.

```@example forms
Vₕ = Wₕ^Val(2)
ac = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(1)) + inner₊ₓ(D₋ₓ(u(2)), D₋ₓ(v(2))))
Ac = assemble(ac)
size(Ac)
```

Sixty-six by sixty-six: two blocks of 33, assembled into one matrix. A term naming `u(i)` and
`v(j)` lands in block ``(j, i)``, so off-diagonal coupling is written the same way —
`innerₕ(u(1), v(2))` fills the block that couples the first unknown to the second equation.

A term must name both components or neither:

```julia
form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v))   # ArgumentError
```

Naming one and leaving the other open has no reading as mathematics — the term would belong
to every equation at once — so it is refused rather than guessed at. Naming neither is fine
and means the diagonal, applied to every block.

## 7. Threaded assembly

`assemble_parallel!` fills the same vector using every available thread:

```@example forms
bp = similar(b)
assemble_parallel!(bp, l)
bp ≈ b
```

It takes no locks, and it needs none. Assembly partitions the grid by *stride*: the offsets a
stencil reaches give the width of the footprint one point writes, and two points separated by
at least that width cannot overlap. Points sharing a stride are therefore written
concurrently with nothing to coordinate.

The common case is one colour. A form whose test argument carries no difference — `innerₕ(fₕ, v)`
above — reaches only its own point, so the stride is 1 in every direction and the whole grid
is swept in a single flat parallel pass. A gradient term in two dimensions reaches one point
back along each axis, giving four colours swept in turn.

Whether threading pays depends on the size, and not always in the obvious direction:
assembly is memory-bound, so the gain flattens well before the thread count does. The
[benchmarks](../benchmarks.md) page carries the measurements.

## 8. Restricting a term to part of the mesh

`innerₕ`, `inner₊` and the directional products all take a `markers` keyword, restricting the
sum to the union of the regions the labels name — the same idea as `restrict_to`, spelled at
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

This is a masked *sum* of the existing cell measures — not a surface integral, and the two
are not interchangeable; a masked `innerₕ` scales like `h` and vanishes under refinement,
where a true boundary integral does not. `markers` is for the former; a Neumann or Robin
term needing the latter is a separate, not-yet-built piece (`inner_Γ`).

A marker that does not exist anywhere the term reaches is a loud error rather than a silent
all-zero contribution — `RegionRestriction`'s own per-point check cannot tell "nothing here
is marked" from "no such marker", so this is caught once, before assembling anything:

```@example forms
try
    assemble(form(Wd, Wd, (u, v) -> innerₕ(u, v; markers = (:nope,))))
catch e
    println(e)
end
```

On a composite space, a marker used without naming a component reaches every diagonal block,
and has to exist on every leaf that reaches — write the term per component, each with its own
markers, if it does not.

## Where to go next

The [internals page on forms](../internals/form.md) documents the colouring and the stencil
algebra underneath all of this, including how the matrix path colours on the test-side span
alone.
