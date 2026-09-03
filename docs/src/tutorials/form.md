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
6. Coupled systems, where a term names which block it belongs to, including a term that
   reads from a leaf built over a different mesh.
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

### Constraining one block, leaving another free

`dirichlet_labels` on its own binds to every leaf sharing the named marker — fine when every
block wants the same treatment, not when they don't. A Stokes-style system prescribing
velocity while leaving pressure unconstrained needs `dirichlet_components` too: 1-based leaf
positions, the same order `u(1)`/`u(2)` addressing already uses.

```@example forms
Ωc = domain(interval(0.0, 1.0), :left => :left, :right => :right)
Ωdc = mesh(Ωc, 21, true)
Vc = gridspace(Ωdc)^Val(2)           # 1: velocity-like, 2: pressure-like
ac2 = form(Vc, Vc, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
Ac2 = assemble(ac2; dirichlet_labels = (:left, :right), dirichlet_components = 1)
nothing # hide
```

Block 1 (rows `1:21`) has its boundary rows pinned; block 2 is untouched — still the plain
assembled operator, no rows replaced at all. Leaving `dirichlet_components` at its default
(`nothing`) applies the labels to every leaf, exactly as before this keyword existed; call
`assemble!`/`dirichlet_bc!` again with a different `dirichlet_labels`/`dirichlet_components`
pair to constrain another block differently.

### Interpolating between the leaves of a heterogeneous composite space

The composite spaces above stack copies of *one* space — every leaf shares a mesh. A
composite space can also be built directly from a tuple of leaves over different meshes,
and then a term coupling two leaves needs a way to move a value from one leaf's grid to
the other's: [`πₕ`](@ref), one argument fewer than the numeric `πₕ`/[`πₕ!`](@ref) pair (see
the [operators tutorial](operators.md) for the numeric side and a diagram of the
interpolant itself) — the same name, told apart by dispatch rather than a different one.

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
  <text x="325" y="88" font-size="11" fill="currentColor" opacity="0.75" text-anchor="middle">a SourceFunction —</text>
  <text x="325" y="103" font-size="11" fill="currentColor" opacity="0.75" text-anchor="middle">composes with D₋ₓ, M₋ₓ, ...</text>

  <path d="M 425 85 L 475 85" stroke="currentColor" stroke-width="2" marker-end="url(#arrowFlow)"/>

  <rect x="480" y="45" width="230" height="80" rx="8" fill="none" stroke="#10b981" stroke-width="1.5"/>
  <text x="595" y="68" font-size="12" font-weight="bold" fill="#10b981" text-anchor="middle">innerₕ(πₕ(u(2)), v(1))</text>
  <text x="595" y="86" font-size="11" fill="currentColor" opacity="0.75" text-anchor="middle">a LinearProduct: assembled</text>
  <text x="595" y="101" font-size="11" fill="currentColor" opacity="0.75" text-anchor="middle">into Wbig's block, leaf 1</text>
</svg>
</figure>
```

`πₕ(uₕ)` reads exactly like any other source — it is one, an AST leaf wrapping
`x -> interpolate_at(uₕ, x)` — so it composes with `D₋ₓ`, `M₋ₓ`, and the rest the same way
`sin`, a `VectorElement`, or any other source does, and can sit on the left of `innerₕ`
inside a coupled form:

```@example forms
Ωbig = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (8, 8), (true, true))
Ωsmall = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (4, 4), (true, true))
Wbig, Wsmall = gridspace(Ωbig), gridspace(Ωsmall)
Vh = CompositeGridSpace((Wbig, Wsmall))
uv = Rₕ(Vh, (x -> 0.0, x -> x[1] + x[2]))   # only the small leaf (2) carries data

lh = form(Vh, v -> innerₕ(πₕ(uv(2)), v(1)) + innerₕ(D₋ₓ(πₕ(uv(2))), D₋ₓ(v(1))))
b = assemble(lh)

# the differenced term is not a no-op: dropping it changes the answer
b_plain = assemble(form(Vh, v -> innerₕ(πₕ(uv(2)), v(1))))
maximum(abs, b .- b_plain)
```

The two terms land in the same block (leaf 1, `Wbig`) even though the source they read
from lives on leaf 2's own, coarser mesh — `πₕ` is what makes that a well-posed
expression rather than a size mismatch. This is exactly what makes a heterogeneous
composite space useful for more than indexing: leaf 2 can represent one field at a
resolution the problem calls for, and a term over leaf 1 can still read it.

That last line is the check worth keeping, not `length(b) == ndofs(Vh)`. An earlier draft of
this page showed the shape instead, and the differenced term contributed *exactly zero*: an
operator wrapped around a source had its offsets discarded, so `D₋ₓ`'s `+s/h` and `−s/h`
cancelled. The page was green either way, because a zero vector has the right length and is
perfectly finite. A worked example should show what it computes.

An operated source is worth a word on what it means. `innerₕ(D₋ₓ(f), v)` is
``\sum_i |\square_i| \, (D_{-x}f)_i \, v_i`` — the operator acts on the *source*, producing
another grid function, which is then integrated against the test function. It agrees entry
for entry with applying the numeric operator first:
`assemble(form(Wₕ, v -> innerₕ(D₋ₓ(fₕ), v)))` equals
`values(D₋ₓ(fₕ)) .* weights(Wₕ, Innerh())`. That equivalence is what
`test/form/source_operators.jl` pins, for every operator, against the numeric layer.

A *bilinear* term coupling two leaves over different meshes is a different matter, and it is
refused:

```julia
assemble(form(Vh, Vh, (u, v) -> innerₕ(u(2), v(1))))   # ArgumentError: ... over different meshes ...
```

(The refusal is raised when the matrix is built, not when the form is written: `form` resolves
the expression, and which leaves a term couples is a question about the spaces it is assembled
against. `allocate_system_matrix` refuses it too, so neither entry point can be reached
around.)

A coupled block is assembled by walking the test leaf's grid and reading the trial column out
of that same index space, so it needs the two leaves to agree on what an index means. Two
leaves over meshes of different sizes do not: index `(3, 3)` on an 8×8 grid and on a 4×4 grid
name different points, and nothing in the term says how to get from one to the other. So there
is no assembly to give, and the error says so rather than guessing — in one direction it used
to overrun the trial block and throw from deep inside `sparse!`, and in the other it quietly
filled in-range but wrong columns.

Coupling leaves that *share* a mesh is unaffected, which is every composite space built by
repeating one space (`Wₕ^Val(2)`), including off-diagonal blocks.

What makes the linear case above legitimate is exactly what the bilinear case is missing:
something that states the mapping between the two meshes. On the trial side that is the
interpolation *operator*, `πₕ(Wsrc, u)`, and supplying it is what turns the refusal above into
an assembly.

### 6.1 The interpolation operator, `πₕ(Wsrc, u)`

One-argument `πₕ` takes a grid function, never a trial or test function:

```julia
form(Vh, Vh, (u, v) -> innerₕ(πₕ(u), πₕ(v)))   # MethodError: no method matching πₕ(::TrialFunction{2})
```

`u`/`v` there are symbolic placeholders with no data of their own — `interpolate_at`, what
that `πₕ` is built from, needs concrete nodal values to blend, and a `TrialFunction`/
`TestFunction` carries none. `πₕ(uₕ)` only ever wraps an already-evaluated
[`VectorElement`](@ref) (`uv(2)` above is exactly that: a component pulled out of a concrete
`uv`, not the symbolic `u`).

The *unknown* is interpolated by a different node, which needs one more argument: the space
the unknown lives on, since a trial function does not carry it.

```@example forms
a_coupled = form(Vh, Vh, (u, v) -> innerₕ(πₕ(Wsmall, u(2)), v(1)))
A = assemble(a_coupled)
size(A)
```

That off-diagonal block is `M · P`: the test leaf's own mass matrix times exactly the matrix
[`interpolation_matrix`](@ref) builds. Both factors are things the tutorial can build for
itself, so this is checkable rather than assertable:

```@example forms
nb, ns = ndofs(Wbig), ndofs(Wsmall)
M = assemble(form(Wbig, Wbig, (u, v) -> innerₕ(u, v)))
P = interpolation_matrix(Wbig, Wsmall)
maximum(abs, A[1:nb, (nb + 1):(nb + ns)] - M * P)
```

Operators wrap it from the outside, and act on the mesh being integrated over — so the same
identity holds with the leaf's stiffness matrix in place of its mass matrix:

```@example forms
K = assemble(form(Wbig, Wbig, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v))))
A2 = assemble(form(Vh, Vh, (u, v) -> inner₊ₓ(D₋ₓ(πₕ(Wsmall, u(2))), D₋ₓ(v(1)))))
maximum(abs, A2[1:nb, (nb + 1):(nb + ns)] - K * P)
```

That is the whole content of the operator: whatever the form would have assembled on the test
leaf alone, applied to the interpolant of the unknown instead of to the unknown. It is
computed a row at a time during the sweep rather than as a matrix product, which is what lets
[`assemble!`](@ref) refill such a block allocating nothing.

Two things about the operator are worth knowing before reaching for it.

The first is that the space argument has to be the trial function's own. The columns `πₕ`
names are numbered in `Wsrc`, and the block writes them into the trial leaf's column range,
so pairing it with the wrong space would write into the wrong part of the matrix — the same
silent-wrong answer the cross-mesh refusal exists to prevent. It is checked, not assumed:

```julia
assemble(form(Vh, Vh, (u, v) -> innerₕ(πₕ(Wbig, u(2)), v(1))))   # ArgumentError: ... not the trial function's ...
```

The second is that the operator goes *outside*, never inside. `D₋ₓ(πₕ(Wsmall, u(2)))`
differences on the mesh being integrated over, which is what the assembled identity above
says. `πₕ(Wsmall, D₋ₓ(u(2)))` would mean something else — difference on the source mesh
first, then interpolate — and rather than quietly assemble one when the other was written,
it is refused:

```julia
form(Vh, Vh, (u, v) -> innerₕ(πₕ(Wsmall, D₋ₓ(u(2))), v(1)))   # ArgumentError: ... is a different operator ...
```

### 6.2 What the operator does not do

**Only the trial side.** `πₕ(Wsrc, v)` on a test function is refused, so interpolating
*both* arguments — `innerₕ(πₕ(u), πₕ(v))`, the ``P^\top H P`` that would give a coarse
space's mass matrix computed by a finer mesh's quadrature — is not available:

```julia
form(Wsmall, Wsmall, (u, v) -> innerₕ(πₕ(Wsmall, u), πₕ(Wsmall, v)))   # ArgumentError: ...
```

(This one *is* raised as the form is written — `πₕ` refuses the test function on the spot,
before there is a matrix to assemble.)

This is a limitation of `form`, not of the interpolation. A form has exactly two spaces, and
the test space doubles as the mesh integrated over: rows come from it and so does the
quadrature weight. ``P^\top H P`` needs *three* — trial and test both on `Wsrc`, quadrature on
a mesh that is neither — and there is nowhere in `form(trial, test, f)` to say which the third
one is. Adding a separate integration space is a larger change than this operator, so the
one-sided case is what exists; the refusal names the restriction rather than assembling
something else.

**A mix of interpolated and plain trial factors in one term is refused.** Both kinds of entry
can live in one stencil, but only the absolute ones are free of the two meshes agreeing on
what an index means:

```julia
assemble(form(Wbig, Wsmall, (u, v) -> innerₕ(πₕ(Wbig, u) + u, v)))   # ArgumentError: ... over different meshes ...
```

The `πₕ(Wbig, u)` summand names columns of `Wbig` outright; the bare `u` beside it still reads
its column out of the index space being walked, which is `Wsmall`'s. On leaves of one size
that is fine and the sum assembles (`P` is then the identity). Across sizes it is the cross-mesh
failure of the previous section exactly, and in one direction it used to be silent — the bare summand's column landed in range and wrong. Every interpolation in a term
is checked against the leaf it writes into, too, so a sum carrying one from the right space and
one from the wrong space is caught rather than half-assembled.

## 7. Threading, chosen once on the backend

`Rₕ!`, `avgₕ!`, gridspace construction and form assembly all thread the same way: `Serial()`
or `Parallel()`, chosen once when the backend is built, rather than decided per call. See the
[backend tutorial](backend.md) for backend construction in general — vector/matrix types
included — and for how to choose between the two policies; this section only covers what the
choice means for assembly specifically.

```@example forms
Wₕ_par = gridspace(mesh(domain(interval(0.0, 1.0)), 33, true;
    backend = backend(policy = Parallel())))
execution_policy(Wₕ_par)
```

There is no automatic size threshold. A `Parallel()` backend threads every eligible call,
however small, however often a time loop repeats it — asking for `Parallel()` and getting it
is the point, rather than a heuristic guessing on the caller's behalf whether a given call is
big enough to be worth it. Pick `Serial()` (the default `backend()` already is) for small,
frequently repeated calls instead.

`assemble!`/`assemble` follow `test_space(form)`'s (or, for a `BilinearForm`, `trial_space`'s)
policy directly, so the ordinary entry points already thread when the backend says to:

```@example forms
l_par = form(Wₕ_par, v -> innerₕ(Rₕ(Wₕ_par, x -> sin(π * x)), v))
b_par = assemble(l_par)      # threads, because Wₕ_par's backend says Parallel()
nothing # hide
```

`assemble_parallel!` still exists underneath, as a lower-level entry point that always
threads regardless of the backend's policy — useful for a one-off forced comparison or a
benchmark, not the everyday call:

```@example forms
bp = similar(b)
assemble_parallel!(bp, l)    # always threads, whatever l's own backend says
bp ≈ b
```

Threading takes no locks, and needs none. Assembly partitions the grid by *stride*: the
offsets a stencil reaches give the width of the footprint one point writes, and two points
separated by at least that width cannot overlap. Points sharing a stride are therefore
written concurrently with nothing to coordinate.

The common case is one colour. A form whose test argument carries no difference — `innerₕ(fₕ, v)`
above — reaches only its own point, so the stride is 1 in every direction and the whole grid
is swept in a single flat parallel pass. A gradient term in two dimensions reaches one point
back along each axis, giving four colours swept in turn.

Whether threading pays depends on the size, and not always in the obvious direction:
assembly is memory-bound, so the gain flattens well before the thread count does. The
[benchmarks](../benchmarks.md) page carries the measurements — that is what should decide
which policy a backend is built with, not a guess.

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
