```@meta
CurrentModule = Bramble
```

# [Coupled systems](@id tutorial_coupled)

**What you will learn.** How to write a form over several unknowns, constrain one block and not another, and read a value across leaves that live on different meshes.

**What you need first.** The [form tutorial](form.md) for assembling and for [Dirichlet conditions](@ref form_dirichlet), and the [space tutorial](space.md) for [composite spaces and vector elements](@ref space_composite).

**Where next.** The [backend tutorial](backend.md), to choose how a system is stored and threaded.

A composite space stacks grid spaces, and a form over one addresses its blocks by component.
This page shows how, using small examples that run when the page is built.

## Addressing blocks by component

`u[1]` (or `u(1)`) is the trial function of the first block, and `v[2]` (or `v(2)`) the test
function of the second. Both spellings work on trial and test functions and on compound
operators such as `(D₋ₓ(u))[i]`. [`components`](@ref) destructures a function into its
blocks:

```@example coupled
using Bramble
import Bramble: CompositeGridSpace

Ωₕ = mesh(domain(interval(0.0, 1.0)), 33, true)
Wₕ = gridspace(Ωₕ)
Vₕ = Wₕ^Val(2)
ac = form(Vₕ, Vₕ, (u, v) -> begin
    u₁, u₂ = components(u)
    v₁, v₂ = components(v)
    innerₕ(u₁, v₁) + inner₊(∇ₕ(u₂), ∇ₕ(v₂), :x)
end)
Ac = assemble(ac)
size(Ac)
```

Sixty-six by sixty-six: two blocks of 33 in one matrix. A term naming `u[i]` and `v[j]`
lands in block ``(j, i)``, so off-diagonal coupling is written the same way.
`innerₕ(u[1], v[2])` fills the block that couples the first unknown to the second equation.

Indices are checked against the number of blocks when the form is built: `u[3]` on a
two-component space raises an `ArgumentError`. A term must name both components or neither:

```@example coupled
try
    form(Vₕ, Vₕ, (u, v) -> innerₕ(u[1], v))
catch e
    showerror(stdout, e)
end
```

Naming one and leaving the other open has no reading as mathematics, because the term would
belong to every equation at once. Naming neither is fine and means the diagonal, applied to
every block.

!!! tip "Try this"
    Add `+ innerₕ(u₁, v₂)` to the body of `ac` and compare `nnz(assemble(ac))` before and
    after (`using SparseArrays`). The new term fills the off-diagonal block that couples the
    first unknown to the second equation.

## [Constraining one block, leaving another free](@id coupled_one_block)

`dirichlet` on its own binds to every leaf that shares the named marker. That is right when
every block wants the same treatment. A Stokes-style system prescribes velocity and leaves
pressure free, so it also needs `dirichlet_components`: 1-based leaf positions, in the order
`u(1)`, `u(2)` already use.

```@example coupled
Ωc = domain(interval(0.0, 1.0), :left => :left, :right => :right)
Ωdc = mesh(Ωc, 21, true)
Vc = gridspace(Ωdc)^Val(2)           # 1: velocity-like, 2: pressure-like
ac2 = form(Vc, Vc, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
Ac2 = assemble(ac2; dirichlet = (:left, :right), dirichlet_components = 1)
nothing # hide
```

Block 1 (rows `1:21`) has its boundary rows pinned. Block 2 is the plain assembled operator,
with no row replaced. Leaving `dirichlet_components` at its default, `nothing`, applies the
labels to every leaf. Call `assemble!` or `dirichlet_bc!` again with another
`dirichlet`/`dirichlet_components` pair to constrain a different block.

!!! tip "Try this"
    Set `dirichlet_components = 2` and compare `Ac2[1, 1]` and `Ac2[22, 22]`. The pinned
    diagonal entry `1.0` moves from block 1 to block 2, and the other keeps its weight. The Dirichlet conditions themselves are explained in the
    [form tutorial](@ref form_dirichlet).

## Interpolating between the leaves of a heterogeneous space

The composite spaces above stack copies of *one* space, so every leaf shares a mesh. A
composite space can also be built from a tuple of leaves over different meshes. A term that
couples two such leaves must move a value from one grid to the other. That is the job of
[`πₕ`](@ref), applied to the trial function alone (the [operators tutorial](operators.md)
has the numeric side and a diagram of the interpolant). It names no source space. The space
it interpolates from is the trial function's own, and assembly supplies it once the leaf is
known.

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

`πₕ(uₕ)` is a source like any other: an AST leaf wrapping `x -> interpolate_at(uₕ, x)`. It
composes with `D₋ₓ`, `Mₓ` and the rest, and sits on the left of `innerₕ` inside a coupled
form:

```@example coupled
Ωbig = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (8, 8), (true, true))
Ωsmall = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (4, 4), (true, true))
Wbig, Wsmall = gridspace(Ωbig), gridspace(Ωsmall)
Vh = CompositeGridSpace((Wbig, Wsmall))
uv = Rₕ(Vh, (x -> 0.0, x -> x[1] + x[2]))   # only the small leaf (2) carries data

lh = form(Vh, v -> innerₕ(πₕ(uv(2)), v(1)) + innerₕ(∇ₕ[:x](πₕ(uv(2))), ∇ₕ[:x](v(1))))
bh = assemble(lh)

# the differenced term is not a no-op: dropping it changes the answer
b_plain = assemble(form(Vh, v -> innerₕ(πₕ(uv(2)), v(1))))
maximum(abs, bh .- b_plain)
```

Both terms land in leaf 1, the big mesh, although the source lives on leaf 2's coarser mesh.
`πₕ` makes that a well-posed expression instead of a size mismatch. Leaf 2 can hold one
field at the resolution the problem calls for, and a term over leaf 1 can still read it.

The last line is the check worth keeping, not `length(bh) == ndofs(Vh)`. A differenced
source whose offsets were discarded assembles to exactly zero, and a zero vector has the
right length and is finite.

An operated source means the following. `innerₕ(D₋ₓ(f), v)` is
``\sum_i |\square_i| \, (D_{-x}f)_i \, v_i``: the operator acts on the *source*, producing
another grid function, which is then integrated against the test function. It agrees entry
for entry with applying the numeric operator first, which is what
`test/form/source_operators.jl` pins for every operator.

### Bilinear coupling across meshes is refused

A *bilinear* term coupling two leaves over different meshes has no assembly to give:

```@example coupled
try
    assemble(form(Vh, Vh, (u, v) -> innerₕ(u(2), v(1))))
catch e
    showerror(stdout, e)
end
```

A coupled block is assembled by walking the test leaf's grid and reading the trial column out
of the same index space, so the two leaves must agree on what an index means. Index
``(3, 3)`` on an 8-by-8 grid and on a 4-by-4 grid name different points, and nothing in the
term says how to get from one to the other. The error is raised when the matrix is built, not
when the form is written, because which leaves a term couples is a question about the spaces
it is assembled against. `allocate_system_matrix` refuses it too.

Leaves that *share* a mesh are unaffected. That covers every space built by repeating one
space (`Wₕ^Val(2)`), off-diagonal blocks included.
