# Operators, differences, inner products and norms

Part of the `bramble` skill. Names written `Bramble.name` are `public` but not exported.

## Discrete operators

Continuous `f` receives scalar (1D) or `NTuple{D}` (nD). A trailing `!` writes into and
returns its first argument.

```julia
# Nodal restriction
uₕ = Rₕ(Wₕ, x -> sin(x[1]) * cos(x[2]))
Rₕ!(uₕ, x -> sin(x[1]) * cos(x[2]))
Rₕ!(vₕ, (x -> x[1], x -> x[2]))            # multi-component: tuple of functions
Rₕ!(vₕ, x -> (sin(x[1]), cos(x[2])))       # or a single vector function, evaluated once
Rₕ!(uₕ, x -> 1.0; markers = (:inlet,))     # restrict to a marked region only

# Cell averaging: tensor-product Gauss-Legendre, exact for polynomials up to
# degree 2*quad_points - 1
u_avg = avgₕ(Wₕ, x -> exp(-x[1]))
avgₕ!(uₕ, x -> exp(-x[1]); quad_points = 6, markers = ())
avgₕ!(vₕ, x -> (sin(x[1]), cos(x[2])))

# Interpolation between grid spaces (piecewise (multi)linear)
πₕ(Wₕ, uₕ)                             # numeric: uₕ's values onto Wₕ's mesh
πₕ!(dest, src)                         # in-place numeric interpolation
interpolate_at(uₕ, x)                  # single-point building block
Bramble.interpolation_matrix(Wₕ_dest, Wₕ_src)  # the same interpolant as a sparse matrix
```

A masked call (non-empty `markers`) has no device kernel yet (see `gpu.md`).

## Finite differences, jumps and averages

A trailing subscript (`ₓ`, `ᵧ`, `₂`, or `ₕ` for the `D`-tuple) picks the direction; `!` is
in-place (`SKILL.md`, notation).

```julia
∇ₕ(uₕ)                                        # backward differences, one per direction (a tuple)
Bramble.D₋ₓ(uₕ), Bramble.D₋ᵧ(uₕ)              # one direction (public, not exported)
dx, dy = ∇ₕ                                   # or destructure: dx === Bramble.D₋ₓ
jumpₕ(uₕ), Bramble.jumpₓ(uₕ)                  # interface jumps (unscaled forward difference)
Mₕ(uₕ), Bramble.Mₓ(uₕ)                        # backward-neighbour averages

Bramble.D₋(uₕ, 1), Bramble.D₋(uₕ, :x), Bramble.D₋(uₕ, Val(1))   # direction as an argument
Bramble.Dc(uₕ, d), Bramble.D̽ₕ(uₕ, d), Bramble.jump(uₕ, d)
Mₕ(uₕ, d), Bramble.M₊ₕ(uₕ, d)                # the averages: no bare `M` exists
∇̽ₕ(uₕ), Mₕ(uₕ)                               # one argument gives the tuple over directions
sum(innerₕ(∇ₕ[d](uₕ), ∇ₕ[d](uₕ)) for d in 1:D)   # dimension-agnostic
```

Visibility:

- Exported: `∇̽ₕ`, `Mₕ`, `∇̃ₕ`, `∇cₕ`.
- `public` only (write `Bramble.D₋` or import by name): the stems `D₋`, `Dc`, `D̃`, `jump`;
  `D̃ₕ`, `Dcₕ`, `D̽ₕ` (spellings of `∇̃ₕ`, `∇cₕ`, `∇̽ₕ`); every in-place vector-calculus form
  (`divₕ!`, `Δₕ!`, ...); `D₊` and its family (forward differences, `∇₊ₕ`, forward averages
  `M₊ₕ`, `Bramble.M₊ₓ`).
- Internal: `diff₋`/`diff₊` and their subscripted forms.

Inside a `form` the direction is a `Val`: `Bramble.D₋(U, Val(1))` builds the node, whose `Dim` is a
type parameter.

Vector calculus built on those differences: `divₕ`/`divₕ!`/`div₊ₕ`, `curlₕ`/`curlₕ!`/`curl₊ₕ`,
`Δₕ`/`Δₕ!` (conservative discrete Laplacian), `εₕ`/`εₕ!` (symmetric small-strain tensor, over
a composite `VectorElement` or a trial/test function inside a `form`).

## Inner products and norms

```julia
innerₕ(uₕ, vₕ)                         # standard discrete L² inner product
inner₊(uₕ, vₕ)                         # modified staggered inner product
Bramble.inner₊ₓ(uₕ, vₕ), Bramble.inner₊ᵧ(uₕ, vₕ)       # directional components
inner_Γ(uₕ, vₕ)                        # boundary inner product

normₕ(uₕ), norminf(uₕ)                # discrete L² and max norms; Bramble.norm₊ is public only
norm(uₕ, "h"), norm(uₕ, "1h"), norm(uₕ, "∞")   # same norms by name; norm(uₕ) is still Euclidean
norm₁ₕ(uₕ), snorm₁ₕ(uₕ)                # discrete H¹ norm and seminorm
```
