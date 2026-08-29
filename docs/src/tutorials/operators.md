# Difference, jump and average operators

Bramble provides the finite difference building blocks that discrete schemes are written
in: differences, jumps, averages, and the inner products and norms that go with them.
This tutorial covers:

1. The four operator families and how their names are built.
2. Applying an operator to a grid function, and what happens at the boundary.
3. The same operator as a sparse matrix.
4. Gradients and the other vectorial forms.
5. Inner products and norms.
6. A convergence study, and the boundary effect that will otherwise spoil it.

Every number below was produced by the code shown.

## 1. The operator families

There are four families. Two of them differ only by a division:

| Family | Meaning | Backward form |
|:--|:--|:--|
| unscaled difference | a plain difference | ``u_i - u_{i-1}`` |
| finite difference | divided by the spacing, so it approximates ``\partial u / \partial x`` | ``\dfrac{u_i - u_{i-1}}{h_i}`` |
| jump | the same arithmetic as the unscaled difference, used where the intent is a discontinuity across an interface | ``u_i - u_{i-1}`` |
| average | the mean of a point and its neighbour | ``\dfrac{u_{i-1} + u_i}{2}`` |

The jump and the unscaled difference compute the same numbers. They are separate names
because they play different roles in a scheme, and reading `jump₋ₓ` in a penalty term
says something that `diff₋ₓ` does not.

### 1.1 How the names are built

A name is a stem, a direction, and a coordinate:

| Piece | Meaning |
|:--|:--|
| `diff` | unscaled difference |
| `D` | finite difference |
| `jump` | jump |
| `M` | average |
| `₋` | backward: the stencil reaches to ``i-1`` |
| `₊` | forward: the stencil reaches to ``i+1`` |
| `ₓ`, `ᵧ`, `₂` | along the first, second or third coordinate |
| `ₕ` | every coordinate at once, returning a tuple |

So `D₋ₓ` is the backward finite difference along ``x``, `M₊ᵧ` the forward average along
``y``, and `jump₋ₕ` the backward jump along every coordinate. The backward finite
difference in every coordinate is the discrete gradient, and has the extra name `∇₋ₕ`.

## 2. Applying an operator

An operator takes a [`VectorElement`](@ref) and returns a new one on the same space.

```julia
using Bramble

Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true)
Wₕ = gridspace(Ωₕ)

points(Ωₕ)       # [0.0, 0.25, 0.5, 0.75, 1.0]
spacings(Ωₕ)     # [0.25, 0.25, 0.25, 0.25, 0.25]

uₕ = Rₕ(Wₕ, x -> x^2)
values(uₕ)       # [0.0, 0.0625, 0.25, 0.5625, 1.0]
```

The four backward operators on that grid function:

```julia
values(diff₋ₓ(uₕ))   # [0.0, 0.0625, 0.1875, 0.3125, 0.4375]
values(D₋ₓ(uₕ))      # [0.0, 0.25,   0.75,   1.25,   1.75  ]
values(M₋ₓ(uₕ))      # [0.0, 0.03125, 0.15625, 0.40625, 0.78125]
values(jump₋ₓ(uₕ))   # [0.0, 0.0625, 0.1875, 0.3125, 0.4375]
```

Reading the second entry of each: `diff₋ₓ` gives ``u_2 - u_1 = 0.0625``, `D₋ₓ` divides
that by ``h_2 = 0.25`` to get ``0.25``, and `M₋ₓ` averages ``(u_1 + u_2)/2 = 0.03125``.
`jump₋ₓ` matches `diff₋ₓ` entry for entry, as section 1 said it would.

## 3. What happens at the boundary

Every operator has one slice where its stencil runs off the grid: the first point for a
backward operator, the last for a forward one. There is no neighbour there, so the
stencil is truncated, and the two difference families truncate differently.

```@raw html
<figure>
<svg viewBox="0 0 720 250" width="100%" style="max-width: 720px; height: auto;"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="A five point grid showing the backward stencil reaching from a point to its left neighbour, the forward stencil reaching right, and the boundary points where each stencil is truncated.">
  <!-- axis -->
  <line x1="60" y1="120" x2="660" y2="120" stroke="currentColor" stroke-width="1.5"/>
  <!-- points -->
  <circle cx="60"  cy="120" r="5" fill="currentColor"/>
  <circle cx="210" cy="120" r="5" fill="currentColor"/>
  <circle cx="360" cy="120" r="5" fill="currentColor"/>
  <circle cx="510" cy="120" r="5" fill="currentColor"/>
  <circle cx="660" cy="120" r="5" fill="currentColor"/>
  <text x="60"  y="145" font-size="13" fill="currentColor" text-anchor="middle">x₁</text>
  <text x="210" y="145" font-size="13" fill="currentColor" text-anchor="middle">x₂</text>
  <text x="360" y="145" font-size="13" fill="currentColor" text-anchor="middle">x₃</text>
  <text x="510" y="145" font-size="13" fill="currentColor" text-anchor="middle">x₄</text>
  <text x="660" y="145" font-size="13" fill="currentColor" text-anchor="middle">x₅</text>

  <!-- backward stencil at x3 -->
  <path d="M 360 105 L 210 105" stroke="#ef4444" stroke-width="2" fill="none"
        marker-end="url(#arrowR)"/>
  <text x="285" y="95" font-size="13" fill="#ef4444" text-anchor="middle">backward: uses x₂ and x₃</text>

  <!-- forward stencil at x3 -->
  <path d="M 360 135 L 510 135" stroke="#10b981" stroke-width="2" fill="none"
        marker-end="url(#arrowG)"/>
  <text x="435" y="158" font-size="13" fill="#10b981" text-anchor="middle">forward: uses x₃ and x₄</text>

  <!-- truncated ends -->
  <text x="60"  y="192" font-size="12" fill="#8b5cf6" text-anchor="middle">no x₀</text>
  <text x="60"  y="209" font-size="12" fill="#8b5cf6" text-anchor="middle">backward truncated here</text>
  <text x="660" y="192" font-size="12" fill="#8b5cf6" text-anchor="middle">no x₆</text>
  <text x="660" y="209" font-size="12" fill="#8b5cf6" text-anchor="middle">forward truncated here</text>
  <line x1="60"  y1="120" x2="60"  y2="180" stroke="#8b5cf6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="660" y1="120" x2="660" y2="180" stroke="#8b5cf6" stroke-width="1" stroke-dasharray="3,3"/>

  <defs>
    <marker id="arrowR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6"
            markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#ef4444"/>
    </marker>
    <marker id="arrowG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6"
            markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981"/>
    </marker>
  </defs>
</svg>
</figure>
```

The finite difference is **zero** on its truncated slice, because there is no one-sided
stencil to divide by a spacing. The unscaled difference and the jump instead behave as
if the missing neighbour were zero, which is what makes them agree with their matrices:

```julia
values(D₋ₓ(uₕ))[1]      # 0.0,   the backward finite difference is truncated at x₁
values(D₊ₓ(uₕ))[end]    # 0.0,   and the forward one at x₅

values(diff₊ₓ(uₕ))[end] # -1.0,  which is -u₅, not 0
values(diff₋ₓ(uₕ))[1]   # 0.0,   which is u₁, and u₁ happens to be 0 here
```

Section 6 shows why this matters in practice.

## 4. Operators as matrices

Passing a mesh or a grid space, rather than a grid function, returns the operator itself
as a sparse matrix:

```julia
A = D₋ₓ(Wₕ)                       # SparseMatrixCSC{Float64, Int64}
A * values(uₕ) ≈ values(D₋ₓ(uₕ))  # true
```

Both routes give the same answer. Applying the operator directly to `uₕ` is the fast
path and is what a time-stepping loop should use; the matrix is what to reach for when
assembling a linear system, and it is also how the test suite checks the fast path.

## 5. Gradients and the other vectorial forms

The `ₕ` suffix applies the operator along every coordinate and returns a tuple with one
entry per dimension. On a one-dimensional mesh it returns the single element itself
rather than a one-tuple.

```julia
Ω₂ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (true, true))
W₂ = gridspace(Ω₂)
vₕ = Rₕ(W₂, x -> x[1] + 2x[2])

g = ∇₋ₕ(vₕ)      # a 2-tuple
length(g)        # 2
```

Away from the truncated slices, `g[1]` is `1.0` and `g[2]` is `2.0`, the two partial
derivatives of ``x + 2y``. The same suffix works for the other families as `diff₋ₕ`,
`jump₋ₕ` and `M₋ₕ`, and all of them accept a mesh, a grid space or a grid function.

## 6. Inner products and norms

The discrete inner product weights each point by its cell measure, and the norms are
built from it:

```julia
innerₕ(uₕ, uₕ)                                    # 0.22070312
normₕ(uₕ)                                         # 0.46979051
normₕ(uₕ)^2 ≈ innerₕ(uₕ, uₕ)                      # true
norm₁ₕ(uₕ)^2 ≈ normₕ(uₕ)^2 + snorm₁ₕ(uₕ)^2        # true
```

`normₕ` is the discrete ``L^2`` norm, `snorm₁ₕ` the ``H^1`` seminorm, and `norm₁ₕ` the
full ``H^1`` norm, which is why the last identity holds. `inner₊` and its per-coordinate
forms use the staggered weights instead, which is what the energy estimates for these
schemes are written in.

## 7. A convergence study, and the boundary

`D₋ₓ` is first order, so the error against a known derivative should fall by a factor of
ten each time the grid is refined by ten. Measuring it naively does not show that:

```julia
for n in (11, 101, 1001, 10001)
    Ωₙ = mesh(domain(interval(0.0, 1.0)), n, true)
    Wₙ = gridspace(Ωₙ)
    uₙ = Rₕ(Wₙ, sin)
    eₙ = D₋ₓ(uₙ) - Rₕ(Wₙ, cos)
    @show n, normₕ(eₙ)
end
```

| ``n`` | every point | order | interior only | order |
|--:|--:|--:|--:|--:|
| 11 | 0.317349 | | 0.026650 | |
| 101 | 0.100034 | 0.50 | 0.002617 | 1.01 |
| 1001 | 0.031624 | 0.50 | 0.000261 | 1.00 |
| 10001 | 0.010000 | 0.50 | 0.000026 | 1.00 |

Over every point the observed order is one half, not one. The cause is section 3:
`D₋ₓ(uₙ)[1]` is `0.0` while ``\cos(0) = 1``, so that one point contributes an error of
``1`` no matter how fine the grid is. It carries a weight of about ``h/2`` in the
discrete norm, so it alone contributes about ``\sqrt{h/2}``, which is exactly the half
order observed.

Excluding that single truncated point recovers the expected first order. So when
measuring convergence, or assembling a scheme, treat the truncated slice explicitly:
that is where the boundary condition belongs, and leaving the operator's truncated value
in place silently halves the observed order.

## Summary

- Four families: `diff` unscaled, `D` divided by the spacing, `jump` for interface terms,
  `M` for averages.
- `₋` and `₊` choose the direction, `ₓ`/`ᵧ`/`₂` the coordinate, `ₕ` every coordinate at
  once. `∇₋ₕ` is the discrete gradient.
- A grid function in gives a grid function out; a mesh or grid space in gives the sparse
  matrix.
- Each operator truncates on one slice. The finite differences are zero there, the
  unscaled differences and jumps act as though the missing neighbour were zero.
- The truncated slice is where the boundary condition goes. Ignoring it costs half an
  order of convergence.
