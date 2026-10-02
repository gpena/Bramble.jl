```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Difference, jump and average operators

The finite difference, the jump and the average, per coordinate and over every coordinate
at once. See the [operators tutorial](../tutorials/operators.md).

A direction held in a variable indexes the vectorial operator: `∇ₕ[2]`, `∇ₕ[:y]` and
`D₋ᵧ` are the same function, so `sum(innerₕ(∇ₕ[d](uₕ), ∇ₕ[d](uₕ)) for d in 1:D)` reads the
same in 1D, 2D and 3D. Underneath, every family has a stem that takes the direction as an
argument: `Bramble.D₋(uₕ, 2)`, `Bramble.D₋(uₕ, :y)` and `Bramble.D₋(uₕ, Val(2))` are all
`D₋ᵧ(uₕ)`. The stems `D₋`, `D₊`, `Dc`, `D̃` and `jump` are `public` but not exported. The
averages put this on `Mₕ`/`M₊ₕ` rather than on a bare `M`, which would take the most common
local name in finite-element code away from anyone writing `using Bramble`; `Mₕ(uₕ)` is
still the tuple over every coordinate and `Mₕ(uₕ, 2)` is the `y` average.

The same stems carry the symbolic form: `D₋(uₕ, Val(1))` differences a grid function now,
`D₋(U, Val(1))` builds the AST node that will difference it during assembly. Inside a form
the direction must be a `Val`, since it is a type parameter of the node.

Three families are not exported, so `using Bramble` does not bring them into scope and they
are written `Bramble.D₊ₓ` or imported by name: the unscaled differences `diff₋*`/`diff₊*`,
the forward differences `D₊*`/`∇₊ₕ`, and the forward averages `M₊*`. Bramble discretises with
the backward operator paired with [`inner₊`](@ref), so the forward ones are what the backward
ones are built and checked against rather than what a form is written with.

The unscaled differences (`diff₋ₓ` and its siblings) are the plain, undivided differences
these are built from, and are the one family of the three that is not even declared
`public`: they have no form-layer node, so they cannot appear inside a bilinear form, and in
a form the undivided forward difference is spelled [`jumpₓ`](@ref), which says which of the
two is meant. `diff₋*`/`diff₊*` are private; see the [forms internals page](../internals/form.md).

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
trial or test function -- the same name spans both, dispatching on what it is given. The
in-place `!` forms write into a preallocated result and are `public` but not exported, as
are `D̃ₕ`, `Dcₕ` and `D̽ₕ`, which are the same functions as the exported `∇̃ₕ`, `∇cₕ` and
`∇̽ₕ`.

```@docs
divₕ
divₕ!
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

Jumps across an interface, ``\llbracket u \rrbracket = u_{i+1} - u_i``. There is one
of these rather than a forward and a backward pair: a jump belongs to the interface
between two cells, not to a direction of travel across it.

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
