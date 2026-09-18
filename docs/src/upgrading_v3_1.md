# Upgrading to v3.1

Bramble 3.1 changes where a grid-function coefficient is read when it sits inside a shifting
operator. [Upgrading to v3.0](@ref) opens by saying everything on it is a compile-time or
load-time error, never a silent behaviour change. This page is the opposite: nothing here
stops your code from compiling or running, and if every coefficient you use is exactly
constant across the mesh, nothing here changes what it computes either. What changes is the
*value* an assembled matrix holds once a coefficient varies from one grid point to the next.
Read this page as a warning to check your forms, not as a checklist a compiler will walk you
through.

## 1. A coefficient inside a difference now moves with the tap

`D₋ₓ(cₕ * u)` reads as "difference the product `cₕ * u`", so at node `i` it ought to compute

```math
D_{-x}(c_h u)_i = \frac{c_i u_i - c_{i-1} u_{i-1}}{h}
```

sampling the coefficient at both ends of the stencil, the same way `u` is sampled. Before this
release it instead computed

```math
c_i \cdot D_{-x}(u)_i = c_i \cdot \frac{u_i - u_{i-1}}{h}
```

reading the coefficient once, at the point being visited, and scaling the whole difference by
it. The two agree when the coefficient is constant. They do not agree on a mesh where
`cᵢ ≠ cᵢ₋₁`, and `D₋ₓ(cₕ * u)` now assembles to a genuinely different matrix than it did before
v3.1.

Both spellings remain available, and both compute what they read as:

- `D₋ₓ(cₕ * u)` — the coefficient moves with the difference's tap. This is what changed.
- `cₕ * D₋ₓ(u)` — the coefficient is read once, at the base point, and scales the whole
  difference afterwards. If this is the operator you meant, write it this way. Before v3.1,
  `D₋ₓ(cₕ * u)` silently computed this instead of what it reads as.

These are not two notations for the same operator. They are different discrete operators
whenever the coefficient is not constant, and only one of them matches what `D₋ₓ(cₕ * u)`
reads as.

The change is the same whether the coefficient's term stands alone or is one summand of a
sum: `D₋ₓ(cₕ * u + u)` also assembles to a different matrix, for the same reason -- the
coefficient-carrying summand moves its coefficient with the tap, and the other summand is
untouched.

## 2. Which nodes are affected

A difference, an average or a jump builds its stencil by folding together taps at `{-1, 0,
+1}` along its direction, and any of them reaches the code this release changes:

- The one-sided differences, `D₋ₓ` and `D₊ₓ`.
- The centred difference `Dcₓ`, the star difference `D̽ₓ`, and the cross-weighted difference
  `Dₕₓ`.
- The two averages, `Mₓ` and `M₊ₓ`.
- The jump, `jumpₓ`.

Each also has its `ᵧ`/`₂` twin in higher dimension. `Bramble.shift_op` is affected too, but it
is not built from any of the nodes above — it is deliberately excluded from that family, since
it relabels its operand's whole stencil rather than folding taps, and reaches the same
coefficient-reading code by its own route.

A coefficient multiplying a **source** — a plain function or array on the right-hand side of a
linear form, not a trial or test function — was already read at the shifted point, and that
case is unchanged: `test/form/source_operators.jl`'s coefficient assertions still pass as they
did before.

## Checklist

1. Grep your forms for a grid-function coefficient multiplying a trial or test function inside
   any of the nodes in §2 — for example `cₕ * u` inside `D₋ₓ`, `D₊ₓ`, `Dcₓ`, `D̽ₓ`, `Dₕₓ`,
   `Mₓ`, `M₊ₓ`, `jumpₓ`, or an argument to `Bramble.shift_op`.
2. For each occurrence, decide which operator you meant:
   - The coefficient sampled at the shifted point, matching the difference's tap — no change
     needed, this is what `D₋ₓ(cₕ * u)` now computes.
   - The coefficient sampled once at the base point — rewrite `D₋ₓ(cₕ * u)` as
     `cₕ * D₋ₓ(u)` (and correspondingly for the other nodes in §2).
3. Where the coefficient is not constant across the mesh, re-check any numbers downstream of a
   form like this one; they may have depended on the old, unintended reading.
