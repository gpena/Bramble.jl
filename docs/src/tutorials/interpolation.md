```@meta
CurrentModule = Bramble
```

# [Interpolation between meshes](@id tutorial_interpolation)

**What you will learn.** How to move a grid function from one mesh to another, what error that costs, and how to get the interpolation as a reusable matrix.

**What you need first.** The [space tutorial](@ref tutorial_space), for grid spaces and discrete functions, and the [operators tutorial](@ref tutorial_operators), since an interpolated function is an ordinary element those operators apply to.

**Where next.** The [form tutorial](form.md) uses the same interpolation inside a form over a heterogeneous composite space.

Every block below runs when this page is built.

```@setup interpolation
using Bramble
import Bramble: interpolation_matrix
```

---

## Move a function to a finer mesh

Every operator in the [operators tutorial](@ref tutorial_operators) maps a grid space to itself. [`πₕ`](@ref) is the one that does not: it moves a grid function from one mesh to a different one. That is what makes a heterogeneous [composite space](@ref space_composite), one whose leaves are built over different meshes, useful for more than indexing.

Take a coarse mesh of five points and a function on it, then ask for its values on a mesh of seventeen points:

```@example interpolation
Wcoarse = gridspace(mesh(domain(interval(0.0, 1.0)), 5))
Wfine = gridspace(mesh(domain(interval(0.0, 1.0)), 17))

coarse = Rₕ(Wcoarse, x -> x^2)
fine = πₕ(Wfine, coarse)

maximum(abs, parent(fine) .- parent(Rₕ(Wfine, x -> x^2)))
```

The result is the largest difference between the interpolated values and the exact ones. It is not zero, because the interpolant is piecewise linear and ``x^2`` is not. The error is at most ``h^2/8`` times the largest second derivative, which for ``h = 0.25`` and ``u'' = 2`` is ``0.0156``.

!!! tip "Try this"
    Replace `x^2` by `x`. A piecewise linear interpolant reproduces an affine function exactly, so the maximum difference drops to rounding error. Then go back to `x^2` and raise `5` to `9`: halving ``h`` divides the error by four.

The method is the standard piecewise (multi)linear interpolant. To read a value at a physical point ``x``, find the cell of the source mesh that contains it ([`locate_cell`](@ref)) and blend the values at that cell's corners, weighted by how close ``x`` is to each one. In two dimensions that is a bilinear blend of four corners:

```@raw html
<figure>
<svg viewBox="0 0 720 330" width="100%" style="max-width:640px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="A single cell of the source mesh with its four corner values. A destination point inside the cell is blended from the four corners, weighted by its relative position within the cell along each coordinate.">
  <defs>
    <marker id="arrowPurple" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#8b5cf6"/>
    </marker>
  </defs>

  <text x="300" y="24" font-size="12" font-weight="bold" fill="currentColor" text-anchor="middle">one cell of Wsrc's mesh</text>

  <!-- the source cell -->
  <rect x="150" y="50" width="300" height="200" fill="none" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.5"/>

  <!-- corner points -->
  <circle cx="150" cy="50"  r="5" fill="currentColor"/>
  <circle cx="450" cy="50"  r="5" fill="currentColor"/>
  <circle cx="150" cy="250" r="5" fill="currentColor"/>
  <circle cx="450" cy="250" r="5" fill="currentColor"/>
  <text x="150" y="35" font-size="12" fill="currentColor" text-anchor="middle">u[i, j+1]</text>
  <text x="450" y="35" font-size="12" fill="currentColor" text-anchor="middle">u[i+1, j+1]</text>
  <text x="150" y="272" font-size="12" fill="currentColor" text-anchor="middle">u[i, j]</text>
  <text x="450" y="272" font-size="12" fill="currentColor" text-anchor="middle">u[i+1, j]</text>

  <!-- weight labels, each beside its own corner -->
  <text x="150" y="90" font-size="11" fill="currentColor" opacity="0.75" text-anchor="middle">(1-tₓ)(1-t_y)</text>
  <text x="450" y="90" font-size="11" fill="currentColor" opacity="0.75" text-anchor="middle">tₓ(1-t_y)</text>
  <text x="150" y="235" font-size="11" fill="currentColor" opacity="0.75" text-anchor="middle">(1-tₓ)t_y</text>
  <text x="450" y="235" font-size="11" fill="currentColor" opacity="0.75" text-anchor="middle">tₓ t_y</text>

  <!-- destination point -->
  <circle cx="330" cy="105" r="6" fill="#10b981" stroke="currentColor" stroke-width="1.2"/>
  <text x="330" y="90" font-size="12" font-weight="bold" fill="#10b981" text-anchor="middle">x  (a point of Wdest)</text>

  <!-- guide lines to the two edges, showing tx and ty -->
  <line x1="330" y1="105" x2="330" y2="250" stroke="#8b5cf6" stroke-width="1.5" stroke-dasharray="4,3"/>
  <line x1="330" y1="105" x2="150" y2="105" stroke="#8b5cf6" stroke-width="1.5" stroke-dasharray="4,3"/>

  <path d="M 150 285 L 330 285" stroke="#8b5cf6" stroke-width="2" fill="none" marker-end="url(#arrowPurple)"/>
  <text x="240" y="303" font-size="12" fill="#8b5cf6" text-anchor="middle">tₓ = (x₁ - u[i]) / (u[i+1] - u[i])</text>

  <path d="M 480 250 L 480 105" stroke="#8b5cf6" stroke-width="2" fill="none" marker-end="url(#arrowPurple)"/>
  <text x="600" y="180" font-size="12" fill="#8b5cf6" text-anchor="middle">t_y, the same</text>
  <text x="600" y="196" font-size="12" fill="#8b5cf6" text-anchor="middle">idea along y</text>
</svg>
</figure>
```

The weights always sum to ``1`` (a partition of unity), so the interpolant never overshoots the range of the corner values. [`interpolate_at`](@ref) computes the blend at one point. [`πₕ`](@ref) and [`πₕ!`](@ref) apply it at every point of a destination space, and are exactly [`Rₕ`](@ref) and [`Rₕ!`](@ref) applied to the interpolant as an ordinary function of position. Restricting a continuous function and interpolating a discrete one are the same mechanism; `πₕ` is the case where that function is another grid function's own interpolant.

The names follow `Rₕ`: `πₕ` and `πₕ!` are the numeric pair here. The same name `πₕ`, with one argument fewer, is the symbolic wrapper that the form tutorial uses, told apart by argument count.

---

## Interpolate in two dimensions

The same call works on a box. An affine function has an exact interpolant, which makes the result easy to check:

```@example interpolation
Ωbig = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (10, 10))
Ωsmall = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (4, 4))
Wbig, Wsmall = gridspace(Ωbig), gridspace(Ωsmall)

src = Rₕ(Wsmall, x -> x[1] + x[2])        # affine, so the interpolant is exact
dest = πₕ(Wbig, src)
exact = Rₕ(Wbig, x -> x[1] + x[2])

maximum(abs, parent(dest) .- parent(exact))
```

Once `πₕ` returns an ordinary [`VectorElement`](@ref), every operator applies to it as normal: `D₋ₓ(dest)`, `Mₓ(dest)`, a bilinear form, anything. The gradient of the interpolated plane is ``1`` in the first coordinate, away from the truncated first row:

```@example interpolation
gx = reshape(parent(∇ₕ(dest)[1]), 10, 10)

all(≈(1.0), gx[2:end, :])
```

---

## Interpolation as a matrix

Like the operators, the interpolant is available as a matrix. It goes between the two spaces rather than from one to itself, and it is rectangular, since the destination and source generally carry a different number of degrees of freedom:

```@example interpolation
P = interpolation_matrix(Wbig, Wsmall)

size(P), P * parent(src) ≈ parent(dest)
```

Each row of `P` has at most ``2^D`` nonzero entries, one destination point's corner weights, so it is sparse. It is always a `SparseMatrixCSC`, whatever matrix type either space's [backend](@ref tutorial_backend) uses. A destination point's source cell has no regular diagonal structure to exploit, so `P` is assembled directly from `locate_cell` rather than composed from shifts, unlike the operator matrices.

`πₕ!(dest, src)` locates every destination point's cell on every call. That is wasted work when the same two meshes are interpolated between repeatedly, for example in a time loop that moves a coefficient across two composite leaves. Build `P` once and pass it to `πₕ!`: no cell search, no allocation, just a matrix-vector product. It is the same "build the pattern once" split that [`allocate_system_matrix`](@ref) and [`assemble!`](@ref) use.

```@example interpolation
dest2 = similar(dest)
πₕ!(dest2, P, src)

parent(dest2) ≈ parent(dest)
```

---

## Interpolation inside a form

`πₕ(uₕ)`, with one argument, is the symbolic counterpart. It wraps the interpolant as an AST source, so it composes with the other operators and can sit inside a [`form`](@ref). The [form tutorial](form.md) works it through against a heterogeneous composite space.
