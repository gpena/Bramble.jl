# Getting started

A Poisson problem, end to end, in twenty lines. Six steps take a continuous problem to a
solved grid function, and every tutorial afterwards is one of them in detail.

```@raw html
<figure style="margin:1.5em 0;text-align:center">
<svg viewBox="0 0 760 150" width="100%" style="max-width:760px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="The six steps of a Bramble solve: domain, mesh, grid space, form, assemble, solve, each labelled with the function that performs it.">
  <defs>
    <marker id="gsArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="currentColor"/>
    </marker>
  </defs>

  <g font-size="12" text-anchor="middle">
    <rect x="10"  y="40" width="104" height="44" rx="6" fill="none" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="62"  y="60" fill="currentColor" font-weight="bold">geometry</text>
    <text x="62"  y="76" fill="#3b82f6">domain</text>

    <rect x="140" y="40" width="104" height="44" rx="6" fill="none" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="192" y="60" fill="currentColor" font-weight="bold">points</text>
    <text x="192" y="76" fill="#3b82f6">mesh</text>

    <rect x="270" y="40" width="104" height="44" rx="6" fill="none" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="322" y="60" fill="currentColor" font-weight="bold">unknowns</text>
    <text x="322" y="76" fill="#3b82f6">gridspace</text>

    <rect x="400" y="40" width="104" height="44" rx="6" fill="none" stroke="#8b5cf6" stroke-width="1.5"/>
    <text x="452" y="60" fill="currentColor" font-weight="bold">equation</text>
    <text x="452" y="76" fill="#8b5cf6">form</text>

    <rect x="530" y="40" width="104" height="44" rx="6" fill="none" stroke="#8b5cf6" stroke-width="1.5"/>
    <text x="582" y="60" fill="currentColor" font-weight="bold">matrix</text>
    <text x="582" y="76" fill="#8b5cf6">assemble</text>

    <rect x="660" y="40" width="90" height="44" rx="6" fill="none" stroke="#10b981" stroke-width="1.5"/>
    <text x="705" y="60" fill="currentColor" font-weight="bold">solution</text>
    <text x="705" y="76" fill="#10b981">A \ F</text>
  </g>

  <g stroke="currentColor" stroke-width="1.2" marker-end="url(#gsArrow)">
    <path d="M 118 62 L 134 62"/>
    <path d="M 248 62 L 264 62"/>
    <path d="M 378 62 L 394 62"/>
    <path d="M 508 62 L 524 62"/>
    <path d="M 638 62 L 654 62"/>
  </g>

  <text x="380" y="118" font-size="11" fill="currentColor" opacity="0.75" text-anchor="middle">
    the same six steps in 1D, 2D and 3D, linear or nonlinear, steady or transient
  </text>
</svg>
</figure>
```

## The problem

```math
-\Delta u = g \text{ in } \Omega = (0,1)^2, \qquad u = u_{\text{exact}} \text{ on } \partial\Omega
```

with the manufactured solution $u_{\text{exact}}(x, y) = \sin(\pi x)\sin(\pi y)$, so
$g = 2\pi^2 u_{\text{exact}}$. Knowing the answer in advance is what makes the error below
checkable.

## The code

```@example gs
using Bramble

uexact(x) = sinpi(x[1]) * sinpi(x[2])
g(x) = 2π^2 * uexact(x)

Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))   # geometry plus boundary labels
Ωₕ = mesh(Ω, (33, 33), (true, true))                  # 33 × 33 points, uniformly spaced
Wₕ = gridspace(Ωₕ)                                    # one unknown per point

gₕ = element(Wₕ)
Rₕ!(gₕ, g)                                            # the source, sampled at the points

a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))    # the discrete Laplacian
l = form(Wₕ, v -> innerₕ(gₕ, v))                      # the load
bcs = dirichlet_constraints(Ω, :boundary => uexact)

A, F = assemble(a, l; dirichlet = bcs)

uₕ = element(Wₕ)
uₕ .= A \ F

normₕ(uₕ .- Rₕ(Wₕ, uexact))
```

Four parts in ten thousand. Halving the spacing should cut that by four, since the scheme
is second order:

```@example gs
function poisson_error(n)
    Ωₕ = mesh(Ω, (n, n), (true, true))
    Wₕ = gridspace(Ωₕ)
    gₕ = element(Wₕ)
    Rₕ!(gₕ, g)
    A, F = assemble(
        form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))),
        form(Wₕ, v -> innerₕ(gₕ, v));
        dirichlet = dirichlet_constraints(Ω, :boundary => uexact))
    uₕ = element(Wₕ)
    uₕ .= A \ F
    return normₕ(uₕ .- Rₕ(Wₕ, uexact))
end

e₁, e₂ = poisson_error(33), poisson_error(65)

e₁, e₂, log2(e₁ / e₂)
```

## What each step gives you

The form is the step worth dwelling on. `a` and `l` are expressions in the trial and test
functions, not matrices: `form` stores the expression, and `assemble` is what walks the mesh.
That separation is what lets the same `a` be refilled every step of a time loop, handed to a
nonlinear solver, or differentiated.

Where to go from here:

- [Discrete foundations](tutorials/geometry.md) covers the first three steps: domains,
  meshes and their metric, grid spaces and the operators that act on them.
- [Linear and bilinear forms](tutorials/form.md) covers the last three: writing a form,
  assembling it, imposing conditions, and coupled systems.
- The [worked examples](examples/poisson_linear.md) run the whole chain on real problems,
  from nonlinear Poisson to 3D elasticity, a heat equation and an inverse problem.
