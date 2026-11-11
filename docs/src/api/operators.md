```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Difference, jump and average operators

The finite difference, the jump and the average, per coordinate and over every coordinate
at once. See the [operators tutorial](../tutorials/operators.md).

A direction held in a variable indexes the vectorial operator: `∇ₕ[2]`, `∇ₕ[:y]` and
`D₋ᵧ` are the same function, so `sum(innerₕ(∇ₕ[d](uₕ), ∇ₕ[d](uₕ)) for d in 1:D)` reads the
same in 1D, 2D and 3D, and `dx, dy = ∇ₕ` destructures it. Indexing and destructuring work
for `∇ₕ`, `∇₊ₕ`, `∇̃ₕ`, `∇cₕ`, `∇̽ₕ`, `jumpₕ`, `Mₕ`, `M₊ₕ`, `Mcₕ`, `S₊ₕ` and `S₋ₕ`.
With `⋅` and `×` from `LinearAlgebra`, the five gradients `∇ₕ`, `∇₊ₕ`, `∇̃ₕ`, `∇cₕ` and
`∇̽ₕ` contract with a vector field to the divergence and the curl of their own family:
`∇ₕ ⋅ uₕ` is `divₕ(uₕ)` and `∇ₕ × uₕ` is `curlₕ(uₕ)`. Inside a form only `∇ₕ`, `∇cₕ` and
`∇̽ₕ` contract.

The same stems carry the symbolic form: `D₋(uₕ, Val(1))` differences a grid function now,
`D₋(U, Val(1))` builds the AST node that will difference it during assembly. Inside a form
the direction must be a `Val`, since it is a type parameter of the node.

## [How the names are built](@id operators_names)

A name is a stem, a direction, and a coordinate:

| Piece | Meaning |
|:--|:--|
| `D` | finite difference |
| `jump` | jump |
| `M` | average |
| `S` | index shift |
| `₋` | backward: the stencil reaches to ``i-1`` |
| `₊` | forward: the stencil reaches to ``i+1`` |
| `ₓ`, `ᵧ`, `₂` | along the first, second or third coordinate |
| `ₕ` | every coordinate at once, returning a tuple |

So `D₋ₓ` is the backward finite difference along ``x``, `M₊ᵧ` the forward average along ``y``, and `∇ₕ` the backward finite difference in every coordinate, the discrete gradient, which has that extra name. `jump` has no `₋`/`₊`: it is `jumpₓ`, `jumpᵧ`, `jump₂` and `jumpₕ`. The index shift has a forward and a backward stem: `S₊ₓ` reads the next point along ``x``, `S₋ₓ` the previous one, and `S₊ₕ`/`S₋ₕ` shift along every coordinate.

The forward difference and the forward average are not exported (see below); the examples import them by name.

## [The direction as an argument](@id operators_direction_argument)

```@setup operators_ref
using Bramble
using Bramble: D₋ᵧ
W₂ = gridspace(mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (5, 5)))
vₕ = Rₕ(W₂, x -> x[1] + 2x[2])
```


The direction can come from a variable: index the vectorial operator with it. `∇ₕ[d]` is the coordinate operator itself, so `∇ₕ[1]`, `∇ₕ[:x]` and `D₋ₓ` are the same function object. This allows a loop over directions. The discrete ``H^1`` seminorm squared, in any dimension, with the `inner₊` weights of the backward gradient, next to `snorm₁ₕ`:

```@example operators_ref
sum(inner₊(∇ₕ[d](vₕ), ∇ₕ[d](vₕ), d) for d in 1:dim(mesh(space(vₕ)))), snorm₁ₕ(vₕ)^2
```

Every family also has a stem that takes the direction as a second argument, and the coordinate suffix is spelling it. An `Int`, a `Symbol` and a `Val` give the same operator:

```@example operators_ref
Bramble.D₋(vₕ, 2) == Bramble.D₋(vₕ, :y) == Bramble.D₋(vₕ, Val(2)) == D₋ᵧ(vₕ)
```

The averages are `Mₕ`/`M₊ₕ`, with no bare `M`, which is the usual name of a mass matrix. The `Val` form is the one to use when the direction must be a compile-time constant, as it must inside a form. `Mₕ(uₕ)` is the tuple over every coordinate and `Mₕ(uₕ, 2)` the average along ``y``, and `∇̽ₕ` works the same way:

```@example operators_ref
∇̽ₕ(vₕ) == (∇̽ₕ(vₕ, 1), ∇̽ₕ(vₕ, 2)), Mₕ(vₕ) == (Mₕ(vₕ, :x), Mₕ(vₕ, :y))
```

An `Int` or a `Symbol` selects between literal `Val`s, one branch per direction the mesh has. An out-of-range direction throws an `ArgumentError`, and the bound is the mesh's own dimension: `Bramble.D₋(uₕ, :y)` on a one-dimensional grid is an error. `∇ₕ(uₕ)` returns a tuple and `Bramble.D₋(uₕ, d)` a grid function.

### Destructuring


A vectorial operator is a tuple-like object of the per-coordinate operators, so it can be destructured or indexed before it is applied. Here `dx` is exactly `D₋ₓ`, reached without importing that name:

```@example operators_ref
dx, dy = ∇ₕ

parent(dx(vₕ)) == parent(∇ₕ(vₕ)[1]), parent(∇ₕ[:x](vₕ)) == parent(∇ₕ(vₕ)[1])
```

The [operator reference](../api/operators.md) lists which operators accept `[d]`, destructuring and `⋅`/`×`.

## Operators

The vectorial operators are exported: `∇ₕ`, `∇̃ₕ`, `∇cₕ`, `∇̽ₕ`, `jumpₕ`, `Mₕ`, `Mcₕ`, `S₊ₕ`,
`S₋ₕ`, and the divergence, curl, Laplacian and strain operators of the backward, tilde,
centred and cross-weighted families. The per-coordinate aliases (`D₋ₓ`, `jumpₓ`, `Mₓ`, ...),
the stems `D₋`, `D₊`, `Dc`, `D̃` and `jump`, the in-place `!` forms and the forward operators
(`D₊*`, `∇₊ₕ`, `div₊ₕ`, `M₊*`, ...) are `public` and not exported: write `Bramble.D₊ₓ` or
import by name. Forms are written with the backward operator paired with
[`inner₊`](@ref); the forward ones are what it is built and checked against.

The unscaled differences `diff₋*`/`diff₊*` are the plain, undivided differences the others
are built from. They are private and have no form-layer node, so they cannot appear inside
a form; there the undivided forward difference is [`jumpₓ`](@ref). They are documented on
the [space internals page](../internals/space.md).

```@docs
D₋ₓ
D₋ₓ!
D₋ᵧ
D₋ᵧ!
D₋₂
D₋₂!
∇ₕ
D₊ₓ
D₊ₓ!
D₊ᵧ
D₊ᵧ!
D₊₂
D₊₂!
∇₊ₕ
D₋
D₊
```

The forward difference over the averaged spacing, which is the one that satisfies
the discrete summation-by-parts identity
``(\tilde{\textrm{D}}_{+x} u_h, v_h)_h = -(u_h, D_{-x} v_h)_{+x}`` for grid functions
`vₕ` vanishing on the boundary.

```@docs
D̃ₓ
D̃ₓ!
D̃ᵧ
D̃ᵧ!
D̃₂
D̃₂!
D̃ₕ
D̃
```

The centered difference, over the span its stencil covers. It reproduces the derivative
of an affine function exactly on any grid, and is skew-symmetric in `innerₕ` for grid
functions vanishing on the boundary.

```@docs
Dcₓ
Dcₓ!
Dcᵧ
Dcᵧ!
Dc₂
Dc₂!
Dcₕ
Dc
```

The cross-weighted centered difference, the same two one-sided differences weighted by
the opposite spacings. It reproduces the derivative of a quadratic exactly on any
grid, and so is second order on a non-uniform one where `Dcₓ` is first.

```@docs
D̽ₓ
D̽ₓ!
D̽ᵧ
D̽ᵧ!
D̽₂
D̽₂!
D̽ₕ
```

The vector calculus operators built on those differences: the divergence and the curl of a
vector field, and the conservative discrete Laplacian of a grid function. The unsubscripted
spellings use the backward differences, as [`∇ₕ`](@ref) does; `div₊ₕ` and `curl₊ₕ` are their
forward twins. [`εₕ`](@ref)/[`εₕ!`](@ref) are the discrete symmetric small-strain tensor,
over a composite `VectorElement` at runtime or, inside a [`form`](@ref), over a composite
trial or test function; the same name spans both. So does [`divₕ`](@ref): inside a form it is the runtime divergence, term by term. The
staggered divergence [`divₘₕ`](@ref) exists only inside a form. The in-place `!` forms write
into a preallocated result. `D̃ₕ`, `Dcₕ` and `D̽ₕ` are `public` and are the same functions as
the exported `∇̃ₕ`, `∇cₕ` and `∇̽ₕ`.

```@docs
divₕ
divₕ!
divₘₕ
div₊ₕ
div₊ₕ!
divcₕ
divcₕ!
div̽ₕ
div̽ₕ!
curlₕ
curlₕ!
curl₊ₕ
curl₊ₕ!
curlcₕ
curlcₕ!
curl̽ₕ
curl̽ₕ!
Δₕ
Δₕ!
εₕ
εₕ!
εcₕ
εcₕ!
ε̽ₕ
ε̽ₕ!
∇cₕ
∇cₕ!
∇̽ₕ
∇̽ₕ!
∇̃ₕ
∇̃ₕ!
diṽₕ
diṽₕ!
curl̃ₕ
curl̃ₕ!
ε₊ₕ
ε₊ₕ!
```

Jumps across an interface, ``\llbracket u \rrbracket = u_{i+1} - u_i``. There is no
forward and backward pair.

```@docs
jumpₓ
jumpₓ!
jumpᵧ
jumpᵧ!
jump₂
jump₂!
jumpₕ
jump
```

Averages of a point with its neighbour.

```@docs
Mₓ
Mₓ!
Mᵧ
Mᵧ!
M₂
M₂!
Mₕ
Mcₓ
Mcₓ!
Mcᵧ
Mcᵧ!
Mc₂
Mc₂!
Mcₕ
M₊ₓ
M₊ₓ!
M₊ᵧ
M₊ᵧ!
M₊₂
M₊₂!
M₊ₕ
```

Index shifts, ``(S_+ u)_i = u_{i+1}`` and ``(S_- u)_i = u_{i-1}``. A neighbour off the grid
reads as zero, so the matrix of `S₊` is the transpose of that of `S₋` and `S₊ₓ(uₕ) - uₕ` is
`jumpₓ(uₕ)` at every point. `S₊ₕ`/`S₋ₕ` are exported; the rest are `public`.

```@docs
S₊ₓ
S₊ₓ!
S₊ᵧ
S₊ᵧ!
S₊₂
S₊₂!
S₊ₕ
S₊
forward_shift
S₋ₓ
S₋ₓ!
S₋ᵧ
S₋ᵧ!
S₋₂
S₋₂!
S₋ₕ
S₋
backward_shift
```

## Facet operators

Operators for the facets of a space with ghost nodes (see
[`gridspace`](@ref)`(Ωₕ; ghosts = ...)`): the outward normal derivative, the second normal
derivative and the normal jump, read in the integrals over a facet
([`inner_Γ`](@ref)) and in the relations of [`constrained`](@ref).

```@docs
∂ₙ
∂ₙₙ
jumpₙ
```
