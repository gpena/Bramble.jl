```@meta
CurrentModule = Bramble
```

# [Linear and bilinear forms](@id tutorial_form)

**What you will learn.** How to write a linear or bilinear form, assemble it into a vector or a matrix, impose Dirichlet conditions and solve a Poisson problem, and impose Neumann and Robin conditions through ghost points, with data that depend on time, a gauge for the ghosts and a check on the compatibility of pure-Neumann data.

**What you need first.** The [mesh tutorial](mesh.md), the [space tutorial](space.md) for grid spaces and elements, and the [operators tutorial](operators.md) for `D₋ₓ`, `innerₕ` and `inner₊`.

**Where next.** [Coupled systems](coupled_systems.md), where one form addresses several unknowns.

The operators in the previous tutorial act on grid functions. A form is the other half: an
expression written in the *test* function, which Bramble assembles into the vector or the
matrix a solver wants. Every number below was produced by the code shown. The running
problem is the one-dimensional Poisson equation on the unit interval.

## A form is an expression in the test function

A linear form is a function of one argument, and that argument stands for the test function
rather than for any particular grid function:

```@example forms
using Bramble
using SparseArrays
import Bramble: allocate_system_matrix, evaluate!, reaction

Ωₕ = mesh(domain(interval(0.0, 1.0)), 33, true)
Wₕ = gridspace(Ωₕ)
fₕ = Rₕ(Wₕ, x -> sin(π * x))

l = form(Wₕ, v -> innerₕ(fₕ, v))
```

Nothing has been computed. `v` is symbolic, so `innerₕ(fₕ, v)` builds a description of

```math
\ell(v) = (f_h, v)_h
```

and the form stores the expression, not a vector. Building one is free, so it is reasonable
to write a form inside a function that is called repeatedly.

The *source* side is eager and the *test* side is symbolic: `D₋ₓ(fₕ)` computes a grid
function, while `D₋ₓ(v)` adds a node to an expression. A term is as cheap as its test side
is symbolic, however elaborate the coefficient in front of it.

## Assembling and refilling

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

### Live grid coefficients and dynamic scalars

A form keeps direct references to its coefficient arrays. Overwrite a grid function in place
with `Rₕ!(fₕ, ...)` or `parent(fₕ) .= ...` between steps, and the next `assemble!(b, l)`
reads the new values with no allocation and no rebuilt form.

A constant factor is written as a plain number, as in `2.5 * innerₕ(fₕ, v)`. A `Ref` is for
a scalar that changes across iterations:

```@example forms
α = Ref(1.0)
l_dyn = form(Wₕ, v -> α * innerₕ(fₕ, v))
b_dyn = assemble(l_dyn)
α[] = 2.0
assemble!(b_dyn, l_dyn)
sum(b_dyn) ≈ 2 * sum(b)
```

## Contracting without a vector

Often the vector is not wanted, only the number ``\ell(v_h)``. A form is callable:

```@example forms
oneₕ = Rₕ(Wₕ, x -> 1.0)
l(oneₕ), sum(b)
```

Against the all-ones grid function a linear form is the sum of its assembled vector. The
difference is that `l(oneₕ)` builds no vector and allocates nothing, so prefer it when the
result is a scalar.

`evaluate!` is the middle case. It takes a scratch vector, fills it with the assembled
vector and returns the contraction:

```@example forms
scratch = zeros(length(b))
evaluate!(scratch, l, oneₕ)
```

A form contracts against an element of its test space, never against a bare vector. The
length of a vector says nothing about whether its blocks match the components a form routes
to, so accepting one would make a coupled mismatch silent.

## Bilinear forms and the system matrix

A bilinear form takes two symbolic arguments, trial first and test second:

```@example forms
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v), :x))
A = assemble(a)
size(A), nnz(A)
```

This is the stiffness matrix of

```math
a(u, v) = (D_{-x} u, D_{-x} v)_{+x}.
```

Ninety-seven nonzeros in a 33-by-33 matrix is the tridiagonal band. The sparsity pattern
follows from the stencil, so it is known before any value is computed. That makes a two-step
idiom possible: `allocate_system_matrix` builds the pattern and nothing else, and `assemble!`
fills a matrix whose structure already exists:

```@example forms
A2 = allocate_system_matrix(a)
assemble!(A2, a)
A2 ≈ A
```

Inside a time loop, build the pattern once outside it and call `assemble!` within. Refilling
a matrix of fixed pattern allocates nothing, where `assemble` allocates a new matrix every
step.

`a` pairs the same `D₋ₓ` on both sides, so it is symmetric. The weight `inner₊ₓ` carries is
positive, so it is also positive semi-definite. `issymmetric` and `isposdef` answer from the
expression alone, without assembling anything:

```@example forms
using LinearAlgebra: issymmetric, isposdef
issymmetric(a), isposdef(a)
```

```@example forms
c = form(Wₕ, Wₕ, (u, v) -> inner₊(u, ∇ₕ(v)))
issymmetric(c)  # different operators either side
```

A positive answer says `cholesky` is worth trying on the result rather than a general
factorization. `isposdef` answers for semi-definiteness: this matrix has the constants in its
kernel, so it is singular until the Dirichlet conditions below are applied.

## [Dirichlet conditions and a Poisson problem](@id form_dirichlet)

Boundary conditions come in two pieces, because a matrix and a right-hand side need
different things done to them. Labels on the domain say where:

```@example forms
Ω = domain(interval(0.0, 1.0), :left => :left, :right => :right)
Ωd = mesh(Ω, 33, true)
Wd = gridspace(Ωd)

fd = Rₕ(Wd, x -> π^2 * sin(π * x))
ad = form(Wd, Wd, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v), :x))
ld = form(Wd, v -> innerₕ(fd, v))

Ad = assemble(ad)
bd = assemble(ld)
nothing # hide
```

[`dirichlet_constraints`](@ref) records the values and [`dirichlet_bc!`](@ref) applies them:
to the matrix by replacing the constrained rows, and to the vector by writing the boundary
values in. `dirichlet_constraints` takes the mesh, a `Domain` or a grid space directly.

```@example forms
bcs = dirichlet_constraints(Ωd, :left => (x -> 0.0), :right => (x -> 0.0))
dirichlet_bc!(Ad, Ωd, :left, :right)
dirichlet_bc!(bd, Ωd, bcs, :left, :right)
nothing # hide
```

Solving ``-u'' = \pi^2 \sin(\pi x)`` with ``u(0) = u(1) = 0`` gives ``u = \sin(\pi x)``:

```@example forms
uh = Ad \ bd
exact = Rₕ(Wd, x -> sin(π * x))
maximum(abs, uh .- parent(exact))
```

Eight parts in ten thousand on 33 points, which is second order behaving itself.

!!! tip "Try this"
    Change the `33` in `mesh(Ω, 33, true)` to `65` and run the block again. The error falls
    by about a factor of four, as second order predicts.

Imposing conditions by replacing rows destroys symmetry, and a symmetric solver wants it
back. [`symmetrize!`](@ref) moves the constrained columns onto the right-hand side, which
restores symmetry and leaves the solution unchanged:

```@example forms
issymmetric(ad)          # true: the form is symmetric before any boundary condition
```

```@example forms
issymmetric(Matrix(Ad))  # false: dirichlet_bc! zeroed rows, not columns
```

```@example forms
symmetrize!(Ad, bd, Ωd, :left, :right)
issymmetric(Matrix(Ad))  # true again, and the solution above is unchanged
```

`issymmetric(ad)` is a claim about the expression `ad`, not about any matrix assembled from
it. That is why the first line answers `true` while the second answers `false`.

To constrain one block of a coupled system and leave another free, see
[Coupled systems](@ref coupled_one_block).

### Boundary fluxes

By the time `uh` exists, `dirichlet_bc!` has overwritten the constrained rows of `Ad`, and
the flux they carried is gone. [`reaction`](@ref) recovers it. It reassembles the
unconstrained `ad` and `ld` and reads the flux off the residual `A*uh - F`, which is
approximately zero on every free row and, on a constrained row, exactly the flux the
condition had to supply:

```@example forms
uh_elt = element(Wd, uh)
reaction(ad, ld, uh_elt; marker = :left), reaction(ad, ld, uh_elt; marker = :right)
```

Both come out near ``\pi``. For ``u = \sin(\pi x)`` the flux ``-u'`` leaving the domain is
``\pi`` at each end, and the two together recover the discrete net
source ``(f_h, 1)_h`` to round-off at any resolution, which approaches ``2\pi`` at second order
(``6.2781`` here). The sign
convention is in the docstring of [`reaction`](@ref); [`reaction_density`](@ref) gives the
pointwise quantity, suitable for [`export_vtk`](@ref).

## [Neumann and Robin conditions with ghost points](@id form_ghost)

A Dirichlet condition fixes a value, and the rows above replace it. A Neumann or Robin condition fixes a normal derivative, and a second-order scheme needs the solution one point past the boundary to write it. This section solves three problems: Neumann and Robin on an interval, Dirichlet beside Robin by two routes, and Neumann on a square. All use non-uniform meshes. The model problem on the interval is

```math
-u'' = f \text{ on } (0, 1), \qquad \partial_\eta u = g_0 \text{ at } 0, \qquad
\alpha u + \partial_\eta u = g_1 \text{ at } 1,
```

where ``\partial_\eta`` is the derivative along the outward normal, ``-u'`` at the left end and ``u'`` at the right. The exact solution is ``u = \cos \pi x + x``, and the data come from it:

```@example forms
import Bramble: change_points!
using LinearAlgebra: norm, dot

function graded_interval(n)
    Ωₕ = mesh(domain(interval(0.0, 1.0)), n)
    change_points!(Ωₕ, collect(range(0.0, 1.0, length = n) .^ 2))
    return Ωₕ
end

u_exact(x) = cos(π * x[1]) + x[1]
du_exact(x) = -π * sin(π * x[1]) + 1
f_src(x) = π^2 * cos(π * x[1])
robin_α = 2.0
g_left(x) = -du_exact(x)                         # ∂ₙu = -u' at x = 0
g_right(x) = robin_α * u_exact(x) + du_exact(x)  # αu + u' at x = 1

Ω_gh = graded_interval(33)
x_gh = collect(points(Ω_gh))
extrema(diff(x_gh))
```

The spacings run from ``10^{-3}`` to ``0.06``.

### Without ghosts: the forms you already have

The weak form of the problem is ``(D_{-}u, D_{-}v)_+`` plus a boundary term per facet, and [`inner_Γ`](@ref) writes both. The Robin coefficient multiplies `u` inside the matrix and the data go on the right:

```@example forms
Wₕ_gh = gridspace(Ω_gh)
a_modeA = form(Wₕ_gh, Wₕ_gh, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) +
    inner_Γ(robin_α * u, v; markers = (:xmax,)))
l_modeA = form(Wₕ_gh, v -> innerₕ(f_src, v) +
    inner_Γ(g_left, v; markers = (:xmin,)) + inner_Γ(g_right, v; markers = (:xmax,)))

A_modeA, F_modeA = assemble(a_modeA, l_modeA)
u_modeA = pde_solve(A_modeA, F_modeA)

err_modeA = maximum(abs, u_modeA - u_exact.(x_gh))
@assert err_modeA < 2e-2

err_modeA
```

Nothing here needs a ghost: a ghost-eliminated Neumann or Robin condition is this system. Ghosts matter when the scheme differentiates the unknown at the boundary, for instance the collocation form ``(-\Delta_h u, v)_h``, a conormal condition with a mixed derivative, or a condition on a second normal derivative. No spelling of those exists on a space without ghosts, because every operator is truncated on the end slices.

### Relations and `constrained`

Give the space ghosts on the facets that carry such a condition, and tie each facet's ghosts to a *relation*, written `operator => data`. [`constrained`](@ref) takes a function of the unknown `u` that returns one `facet => relation` pair per facet:

```@example forms
W_ghosts = gridspace(Ω_gh; ghosts = :boundary)
C_gh = constrained(W_ghosts, u -> (
    :xmin => ∂ₙ(u) => g_left,
    :xmax => robin_α * u + ∂ₙ(u) => g_right))

ndofs(W_ghosts), ndofs(C_gh)
```

`=>` binds looser than `+` and `*` and is right-associative, so the second line reads `:xmax => ((robin_α * u + ∂ₙ(u)) => g_right)`. The data is a number or a function of the point tuple, as for [`dirichlet_constraints`](@ref). `constrained` solves each ghost's relation for the ghost, and its unknowns are the 33 physical nodes again: the 2 ghosts were eliminated.

The form below is the collocation Laplacian at every node, the boundary ones included, with the cell weight as the test weight. Assemble both forms together with `assemble(a, l)`: the data move the ghosts, so the load is ``E^T (F - A d)`` and needs the matrix.

```@example forms
a_gh = form(C_gh, C_gh, (u, v) -> innerₕ(-Δₕ(u), v))
l_gh = form(C_gh, v -> innerₕ(f_src, v))
A_gh, F_gh = assemble(a_gh, l_gh)
u_gh = pde_solve(A_gh, F_gh)

@assert norm(A_gh - A_modeA) <= 1e-12 * norm(A_modeA)
@assert norm(F_gh - F_modeA) <= 1e-12 * norm(F_modeA)
@assert maximum(abs, u_gh - u_modeA) <= 1e-10

norm(A_gh - A_modeA) / norm(A_modeA), maximum(abs, u_gh - u_exact.(x_gh))
```

The ghost-eliminated matrix and load equal those of the forms of the previous section to rounding. The two spellings are the same scheme, and the ghost one extends to what the first cannot say. The error decreases at second order on this graded mesh: it falls by a factor of four when the number of points doubles.

```@example forms
function ghost_error(n)
    Ωₕ = graded_interval(n)
    C = constrained(gridspace(Ωₕ; ghosts = :boundary), u -> (
        :xmin => ∂ₙ(u) => g_left, :xmax => robin_α * u + ∂ₙ(u) => g_right))
    A, F = assemble(form(C, C, (u, v) -> innerₕ(-Δₕ(u), v)),
        form(C, v -> innerₕ(f_src, v)))
    return maximum(abs, pde_solve(A, F) - u_exact.(points(Ωₕ)))
end

ratio = ghost_error(33) / ghost_error(65)
@assert 3.5 < ratio < 4.5

ratio
```

The relations a facet takes:

| Relation | Condition |
|:--|:--|
| `∂ₙ(u) => g` | Neumann |
| `α * u + β * ∂ₙ(u) => g` | Robin, with ``\beta \neq 0`` |
| `∂ₙ(u; K = K) => g` | conormal, ``\eta \cdot (K \nabla u)``, with a constant matrix `K` |
| `∂ₙₙ(u) => g` | second normal difference, ``(\tilde D D_{-} u)`` at the facet node |
| `u => g` | Dirichlet, see below |

Each relation of a ghosted facet must read that facet's ghost: `β = 0` or a lone `jumpₙ` is an `ArgumentError`. At a 2D corner two conormal relations that read each other's ghost are solved together as a ``2 \times 2`` block, and a singular `K` is refused. A ghosted facet with no relation keeps its ghosts as unknowns.

```@example forms
try
    constrained(W_ghosts, u -> (:xmin => jumpₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.0))
catch e
    @assert e isa ArgumentError
    first(split(sprint(showerror, e), ". "))
end
```

### Relations that read several components

On a composite space `S × S`, `constrained` takes relations that read several components of `u`. The case that needs it is the traction of elasticity, whose component ``i`` is ``\sum_j \eta \cdot (K_{ij} \nabla u_j)`` with ``(K_{ij})_{de} = \lambda\,\delta_{id}\delta_{je} + \mu\,(\delta_{ij}\delta_{de} + \delta_{ie}\delta_{jd})``. The tangential part is the conormal derivative with an off-diagonal `K`, so `∂ₙ(u(j); K = Kᵢⱼ)` summed over `j` is the traction on every facet, and no new operator is needed.

A facet whose relation reads several components needs one ghost relation per component, written unkeyed: `:xmin => t₁ => g₁, :xmin => t₂ => g₂`. The ghosts of a node are solved together, so the relation of component 1 may read the ghost of component 2. A displacement facet carries ghosts too, closed by `∂ₙₙ(u(i)) => 0` beside the Dirichlet relation `u(i) => 0` of each component. The block below loads the facet ``x = 0`` with a traction, clamps the other three, solves the collocated form, and checks that both traction relations hold on the lifted values:

```@example forms_traction
using Bramble, LinearAlgebra
import Bramble: Dcₓ, Dcᵧ, lift

Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 11), (false, false))
S = gridspace(Ωₕ; ghosts = :boundary)
V = S × S
λ, μ = 2.0, 0.5

# Kᵢⱼ[d, e] = λ δᵢd δⱼe + μ (δᵢⱼ δde + δᵢe δⱼd), so that Σⱼ ∂ₙ(u(j); K = Kᵢⱼ) is the
# traction of component i on every facet.
K(i, j) = [λ * (i == d) * (j == e) + μ * ((i == j) * (d == e) + (i == e) * (j == d))
           for d in 1:2, e in 1:2]
traction(u, i) = ∂ₙ(u(1); K = K(i, 1)) + ∂ₙ(u(2); K = K(i, 2))
displacement(F, u) = (F => u(1) => 0.0, F => u(2) => 0.0,
    F => ∂ₙₙ(u(1)) => 0.0, F => ∂ₙₙ(u(2)) => 0.0)

Cₕ = constrained(V, u -> (
    :xmin => traction(u, 1) => 1.0, :xmin => traction(u, 2) => 0.0,
    displacement(:ymin, u)..., displacement(:xmax, u)..., displacement(:ymax, u)...))

Dc = (Dcₓ, Dcᵧ)
a = form(Cₕ, Cₕ, (u, v) -> sum(
    innerₕ(-μ * Δₕ(u(i)), v(i)) +
    sum(innerₕ(-(λ + μ) * Dc[i](Dc[j](u(j))), v(i)) for j in 1:2) for i in 1:2))
l = form(Cₕ, v -> innerₕ(1.0, v(1)) + innerₕ(0.5, v(2)))
A, F = assemble(a, l)
U = lift(Cₕ, pde_solve(A, F))

# Check on the lifted padded values that each traction row equals its data on :xmin.
PV = Bramble.padded_space(V)
for (i, g) in ((1, 1.0), (2, 0.0))
    R = assemble(form(PV, PV, (w, v) -> sum(
        inner_Γ(∂ₙ(w(j); K = K(i, j)), v(i); markers = (:xmin,)) for j in 1:2)))
    W = assemble(form(PV, PV, (w, v) -> inner_Γ(w(i), v(i); markers = (:xmin,))))
    r = R * collect(U) - g * (W * ones(length(U)))
    @assert norm(r, Inf) <= 1e-9 * max(1.0, norm(R * collect(U), Inf))
end
ndofs(Cₕ), ndofs(PV)
```

`Bramble.lift(Cₕ, u)` returns the padded displacement with its ghosts filled; take a component of it as `Bramble.lift(Cₕ, u)(i)`. A component view `uₕ(i)` of the constrained space itself is refused, because the closure couples the components, and so are `×` and `^` of it, relations with data in time, the `dirichlet` keyword and `GhostVector`. A corner of two traction facets closes with a one-sided tangential difference along the facets. On a facet with a traction, the mixed differences of the form (those with `∂ₓ` and `∂ᵧ` of different axes, in the left argument of the product) do not read the ghosts of the facet: they extrapolate the function to the ghost from the three nodes inward, to second order, which keeps the collocated form stable on cells wider across the facet than along it. The relations still hold on the lifted values, as the check above asserts.

### Dirichlet conditions beside ghost relations

Keep the Robin condition at ``x = 1`` and impose ``u(0) = 1`` instead, so only `xmax` needs a ghost. There are two routes, and they give the same solution.

**Route (a), row replacement, as above.** `constrained` receives only the ghost relation. The unknowns are the physical nodes, as on a space without ghosts, so the `dirichlet` keyword of `assemble` replaces the rows of `:xmin` and `symmetrize` restores symmetry, exactly as in the [Dirichlet section](@ref form_dirichlet). Every existing program keeps working:

```@example forms
W_right = gridspace(Ω_gh; ghosts = (:xmax,))
C_a = constrained(W_right, u -> (:xmax => robin_α * u + ∂ₙ(u) => g_right,))
a_a = form(C_a, C_a, (u, v) -> innerₕ(-Δₕ(u), v))
l_a = form(C_a, v -> innerₕ(f_src, v))
A_a, F_a = assemble(a_a, l_a; dirichlet = :xmin => u_exact, symmetrize = true)
u_a = pde_solve(A_a, F_a)

@assert length(u_a) == 33 && u_a[1] ≈ u_exact(x_gh[1])
@assert norm(A_a - A_a') <= 1e-12 * norm(A_a)

ndofs(C_a), maximum(abs, u_a - u_exact.(x_gh))
```

**Route (b), `u => g` in the relations.** The Dirichlet node leaves the unknowns: its column is removed and its value goes into the data. The matrix is ``E^T A E``, symmetric by construction, and it has one row less:

```@example forms
C_b = constrained(W_right, u -> (
    :xmin => u => u_exact,
    :xmax => robin_α * u + ∂ₙ(u) => g_right))
a_b = form(C_b, C_b, (u, v) -> innerₕ(-Δₕ(u), v))
l_b = form(C_b, v -> innerₕ(f_src, v))
A_b, F_b = assemble(a_b, l_b)
u_b = pde_solve(A_b, F_b)

@assert ndofs(C_b) == 32 && size(A_b) == (32, 32)
@assert norm(A_b - A_b') <= 1e-12 * norm(A_b)
@assert maximum(abs, [u_exact(x_gh[1]); u_b] - u_a) <= 1e-10

ndofs(C_b), maximum(abs, [u_exact(x_gh[1]); u_b] - u_exact.(x_gh))
```

The solution vector of route (b) has one entry per kept node, so the Dirichlet value is prepended above; `Bramble.lift(C_b, u_b)` below gives the grid function. The two routes compare as follows:

| | (a) row replacement | (b) `u => g` in `constrained` |
|:--|:--|:--|
| unknowns | all physical nodes | physical nodes minus the Dirichlet nodes |
| solution vector | one entry per node, a grid function as today | one entry per kept node; `Bramble.lift(Cₕ, u)` is the grid function |
| symmetry of a self-adjoint form | after `symmetrize`, as today | by construction |
| data | `dirichlet`, `dirichlet_constraints`, time-dependent as today | the relation's `g`, a number or a function |
| existing programs | unchanged | new route, nothing depends on it |

Route (a) is the one every earlier program uses; route (b) is the symmetric alternative when the vector should carry only unknowns. Give each facet's condition once. The same label given both ways, `u => g` in `constrained` and `dirichlet` in `assemble`, is an `ArgumentError`, since the value would be imposed twice. A ghosted facet that carries only `u => g` is refused too, since its ghosts would be unknowns no equation reads: give it a ghost relation as well, or drop it from `ghosts`.

```@example forms
refusals = String[]
for thunk in (
    () -> assemble(a_b, l_b; dirichlet = :xmin => u_exact),
    () -> constrained(W_ghosts, u -> (:xmin => u => u_exact, :xmax => ∂ₙ(u) => g_right)),
)
    try
        thunk()
    catch e
        @assert e isa ArgumentError
        push!(refusals, first(split(sprint(showerror, e), ". ")))
    end
end

@assert length(refusals) == 2

refusals
```

### Solutions as grid functions: `lift`

A solution of `assemble(a, l)` is a vector of unknowns. `Bramble.lift(Cₕ, u)` is the grid function it describes, with the eliminated ghosts filled in by the relations. It lives on the padded space, one point longer at each ghosted end, so the runtime operators read its ghosts. `lift` is `public`, not exported, because it is also a Makie name: write `Bramble.lift`.

```@example forms
U = Bramble.lift(C_gh, u_gh)

@assert length(U) == 35

points(space(U))[[1, 2, end - 1, end]]
```

The solution of route (b) above lifts the same way, and the lifted function holds the Dirichlet value at its first node:

```@example forms
U_b = Bramble.lift(C_b, u_b)

@assert length(U_b) == 34 && U_b[1] ≈ u_exact(x_gh[1])
@assert U_b[1:33] ≈ [u_exact(x_gh[1]); u_b]

U_b[1], u_exact(x_gh[1])
```

The relations hold on the lifted function. The runtime centred difference at the first and last nodes gives the data, and the collocation equation ``-\Delta_h u = f`` holds at every node, the two boundary nodes included:

```@example forms
import Bramble: Dcₓ

@assert -Dcₓ(U)[2] ≈ g_left(x_gh[1])
@assert robin_α * U[end - 1] + Dcₓ(U)[end - 1] ≈ g_right(x_gh[end])
@assert maximum(abs, [(-Δₕ(U))[i + 1] - f_src(x_gh[i]) for i in 1:33]) < 1e-6

(-Dcₓ(U)[2], g_left(x_gh[1])), (robin_α * U[end - 1] + Dcₓ(U)[end - 1], g_right(x_gh[end]))
```

`Bramble.lift(Cₕ)` alone is the data ``d``: the ghost values of the zero function. For the Neumann relation at the left, ``-(u_2 - u_0)/(2 h_2) = g_0`` with ``u_2 = 0`` gives ``u_0 = 2 h_2 g_0``:

```@example forms
d_gh = Bramble.lift(C_gh)
h₂ = x_gh[2] - x_gh[1]

@assert d_gh[1] ≈ 2h₂ * g_left(x_gh[1])
@assert iszero(d_gh[2:(end - 1)])

d_gh[1], 2h₂ * g_left(x_gh[1])
```

Both are data in forms on `Cₕ`, so a linear form can name the full `Δₕ` of the lifted function at a facet node. A relation on the second normal difference puts its data into the same ghost:

```@example forms
C_nn = constrained(W_ghosts, u -> (:xmin => ∂ₙₙ(u) => 3.0, :xmax => u + ∂ₙ(u) => 1.0))
d_nn = Bramble.lift(C_nn)

@assert d_nn[1] ≈ 3.0 * h₂^2          # (u₂ - 2u₁ + u₀) / h₂² = 3 with u = 0

d_nn[1]
```

A system whose ghost data are nonzero needs the matrix to build its load. Assembling the linear form alone is an `ArgumentError`, and the message names the fix:

```@example forms
try
    assemble(l_gh)
catch e
    @assert e isa ArgumentError
    sprint(showerror, e)
end
```

### Data that depend on time

The data of a relation may take the time as a second argument, `(x, t) -> …`; `constrained` tells it from a datum in `x` alone by the methods the function has. The operators in a relation do not change with time, so the matrix of a form on the space is the same at every `t`, and only the data ``d(t)`` that the relations put into the ghosts move. Take the heat equation ``\partial_t u = \partial_{xx} u`` on the graded interval, with the exact solution ``u = e^{-t} (\cos x + \sin x)`` and the data it gives, a flux at the left end and a Robin condition at the right end:

```@example forms
using LinearAlgebra: diag

heat_sol(x, t) = exp(-t) * (cos(x[1]) + sin(x[1]))
heat_dx(x, t) = exp(-t) * (cos(x[1]) - sin(x[1]))
g_left_t(x, t) = -heat_dx(x, t)
g_right_t(x, t) = robin_α * heat_sol(x, t) + heat_dx(x, t)

C_t = constrained(gridspace(Ω_gh; ghosts = :boundary), u -> (
    :xmin => ∂ₙ(u) => g_left_t,
    :xmax => robin_α * u + ∂ₙ(u) => g_right_t))
a_t = form(C_t, C_t, (u, v) -> innerₕ(-Δₕ(u), v))
l_t = form(C_t, v -> innerₕ(x -> 0.0, v))

ndofs(C_t)
```

`Bramble.lift(Cₕ, u, t)` is the grid function ``E u + d(t)`` at time `t`. The exact solution at the nodes lifts to a padded function whose centred difference at the first node gives the flux datum at that time, and the data at two times give different ghosts:

```@example forms
u_heat(t) = [heat_sol(x, t) for x in x_gh]
U_a = Bramble.lift(C_t, u_heat(0.3), 0.3)
U_b = Bramble.lift(C_t, u_heat(1.0), 1.0)

@assert -Dcₓ(U_a)[2] ≈ g_left_t(x_gh[1], 0.3)
@assert -Dcₓ(U_b)[2] ≈ g_left_t(x_gh[1], 1.0)
@assert abs(U_a[1] - U_b[1]) > 0.1

U_a[1], U_b[1]
```

[`semidiscretize`](@ref) takes the form and the load on this space and returns the `Semidiscretization` of ``M u' = F(t) - A u``, with ``A = E^T A_{pad} E``, the default mass ``M`` diagonal and positive, and ``F(t) = E^T (F_{pad}(t) - A_{pad}\, d(t))``. The residual at a time is the one a space with the data frozen at that time assembles. Here is the comparison at three times:

```@example forms
sd_t = semidiscretize(a_t, l_t)
u_probe = collect(range(-1, 2; length = ndofs(C_t)) .^ 2)

residual_gaps = map((0.0, 0.5, 1.0)) do t
    C_frozen = constrained(gridspace(Ω_gh; ghosts = :boundary), u -> (
        :xmin => ∂ₙ(u) => (x -> g_left_t(x, t)),
        :xmax => robin_α * u + ∂ₙ(u) => (x -> g_right_t(x, t))))
    A_fr, F_fr = assemble(form(C_frozen, C_frozen, (u, v) -> innerₕ(-Δₕ(u), v)),
        form(C_frozen, v -> innerₕ(x -> 0.0, v)))
    du = similar(u_probe)
    sd_t(du, u_probe, nothing, t)
    norm(du - (F_fr - A_fr * u_probe)) / norm(du)
end

@assert all(<(1e-12), residual_gaps)
@assert Bramble.operator_matrix(sd_t) ≈ assemble(a_t)
@assert all(>(0), diag(Bramble.mass_matrix(sd_t)))

maximum(residual_gaps)
```

[`ode_problem`](@ref) wraps it for an ODE solver. The initial condition is the exact solution at ``t = 0`` and the solver is `FBDF` from `OrdinaryDiffEqBDF`, with tolerances tight enough that the error left at the final time is the one in space. The solution changes by a factor ``e^{-1}`` over the run and the data change with it, so an error far below that change shows the data in time are followed. The error falls by a factor of four when the number of points doubles:

```@example forms
using OrdinaryDiffEqBDF

function heat_error(n; T = 1.0)
    Ωₕ = graded_interval(n)
    C = constrained(gridspace(Ωₕ; ghosts = :boundary), u -> (
        :xmin => ∂ₙ(u) => g_left_t, :xmax => robin_α * u + ∂ₙ(u) => g_right_t))
    a = form(C, C, (u, v) -> innerₕ(-Δₕ(u), v))
    l = form(C, v -> innerₕ(x -> 0.0, v))
    xs = collect(points(Ωₕ))
    prob = ode_problem(a, l, [heat_sol(x, 0.0) for x in xs], (0.0, T))
    sol = solve(prob, FBDF(); reltol = 1e-10, abstol = 1e-12)
    return maximum(abs, sol.u[end] - [heat_sol(x, T) for x in xs])
end

err_heat_33, err_heat_65 = heat_error(33), heat_error(65)
@assert err_heat_33 < 1e-3
@assert 3.5 < err_heat_33 / err_heat_65 < 4.5

err_heat_33, err_heat_33 / err_heat_65
```

The refusals are the same everywhere on a space with data in time, and each is an `ArgumentError` that names what to do. `lift(Cₕ, u)`, `lift(Cₕ)` and `assemble(a, l)` have no single ``d``: give the time through `lift(Cₕ, u, t)`, or pose the problem in time. `form` and `assemble(a)` read `E` only and work. `semidiscretize` takes no `dirichlet` keyword, since a Dirichlet condition on a facet is the relation `:F => u => g(x, t)`. It refuses a space that keeps ghosts as unknowns, since the mass has no row for them. A mass that reads a point where the data depend on time is refused too, because its term ``E^T M_{pad}\, d'(t)`` needs ``\partial_t g``, which no relation carries. The default mass reads the physical nodes only. `semidiscretize_rhs` and `semidiscretize(build, l)` with a ghost load are refused as well.

```@example forms
W_unknown = gridspace(Ω_gh; ghosts = :boundary)
C_unknown = constrained(W_unknown, u -> (:xmin => ∂ₙ(u) => g_left_t,))

timed_refusals = String[]
for thunk in (
    () -> Bramble.lift(C_t, u_heat(0.0)),
    () -> assemble(a_t, l_t),
    () -> semidiscretize(a_t, l_t; dirichlet = :xmin => 0.0),
    () -> semidiscretize(form(C_unknown, C_unknown, (u, v) -> innerₕ(-Δₕ(u), v)),
        form(C_unknown, v -> innerₕ(x -> 0.0, v))),
    () -> semidiscretize(a_t, l_t;
        mass = form(C_t, C_t, (u, v) -> innerₕ(u, v) + 0.01 * innerₕ(Δₕ(u), v))),
)
    try
        thunk()
    catch e
        @assert e isa ArgumentError
        push!(timed_refusals, first(split(sprint(showerror, e), ". ")))
    end
end

@assert length(timed_refusals) == 5
@assert size(assemble(a_t), 1) == ndofs(C_t)

timed_refusals
```

A constrained space whose data are all static takes `semidiscretize` and `ode_problem` too.

### Pure Neumann and `mean_zero`

With Neumann conditions on every facet the solution is defined up to a constant. The matrix annihilates constants and is singular. Fix the constant by a discrete mean-zero condition: [`pde_solve`](@ref) takes `mean_zero = w`, a weight vector, and returns the solution with ``w^T u = 0``. Internally it solves the bordered system

```math
\begin{pmatrix} A & w \\ w^T & 0 \end{pmatrix}
\begin{pmatrix} u \\ \lambda \end{pmatrix} =
\begin{pmatrix} F \\ 0 \end{pmatrix}
```

with a symmetric or LU factorization, never Cholesky, since the border makes it indefinite. `A` is not modified and stays sparse. A matrix that annihilates constants, passed without `mean_zero`, produces a warning that names the keyword.

The problem is ``-\Delta u = f`` on a graded square with the normal derivative given on each facet. Each facet carries its own datum, so a corner node takes two values, one per facet:

```@example forms
Ω_sq = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ_sq = mesh(Ω_sq, (17, 17), (true, true))
grid = collect(range(0.0, 1.0, length = 17) .^ 2)
change_points!(Ωₕ_sq, markers(Ω_sq), (grid, grid))

u_sq(p) = cos(π * p[1]) * cos(π * p[2]) + p[1] * p[2] / 2
grad_sq(p) = (-π * sin(π * p[1]) * cos(π * p[2]) + p[2] / 2,
    -π * cos(π * p[1]) * sin(π * p[2]) + p[1] / 2)
f_sq(p) = 2π^2 * cos(π * p[1]) * cos(π * p[2])

W_sq = gridspace(Ωₕ_sq; ghosts = :boundary)
C_sq = constrained(W_sq, u -> (
    :xmin => ∂ₙ(u) => (p -> -grad_sq(p)[1]), :xmax => ∂ₙ(u) => (p -> grad_sq(p)[1]),
    :ymin => ∂ₙ(u) => (p -> -grad_sq(p)[2]), :ymax => ∂ₙ(u) => (p -> grad_sq(p)[2])))
A_sq, F_sq = assemble(form(C_sq, C_sq, (u, v) -> innerₕ(-Δₕ(u), v)),
    form(C_sq, v -> innerₕ(f_sq, v)))

@assert norm(A_sq * ones(size(A_sq, 1)), Inf) <= 1e-10 * norm(A_sq, Inf)

ndofs(C_sq), norm(A_sq * ones(size(A_sq, 1)), Inf)
```

The matrix has 289 unknowns, one per node, and ``A \mathbf{1} = 0`` to rounding. The weight vector is the cell measure, the linear form of the constant function ``1`` on the space without ghosts:

```@example forms
W_plain_sq = gridspace(Ωₕ_sq)
w_sq = assemble(form(W_plain_sq, v -> innerₕ(x -> 1.0, v)))
u_sol = pde_solve(A_sq, F_sq; mean_zero = w_sq)

# the defining equations of the bordered system
λ = sum(F_sq) / sum(w_sq)
@assert abs(dot(w_sq, u_sol)) <= 1e-12
@assert norm(A_sq * u_sol - (F_sq - λ * w_sq)) <= 1e-9 * norm(F_sq)

# against the exact solution shifted to discrete mean zero
u_ex = parent(Rₕ(W_plain_sq, u_sq))
u_ex .-= dot(w_sq, u_ex) / sum(w_sq)
err_sq = maximum(abs, u_sol - u_ex)
@assert err_sq < 2e-2

abs(dot(w_sq, u_sol)), err_sq
```

The residual of the system is ``F - \lambda w`` with ``\lambda = \sum F / \sum w``, because ``\mathbf{1}^T A = 0``. The compatibility of the data holds only to the quadrature error, so ``\lambda`` is small, not zero. The error is the discrete difference to the exact solution shifted to the same mean.

#### A kernel in the ghosts

`mean_zero` fixes the constant of a pure-Neumann problem. A space that keeps its ghosts as unknowns has a second kind of kernel, in the ghosts themselves. The form `innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)` reads the ghost of a corner node only through `Δₕ`, where the two ghosts of a 2D corner enter in one combination, so the matrix assembled on `gridspace(Ωₕ; ghosts = :boundary)` is singular: a vector that is zero at every physical node and nonzero on the ghosts of a corner solves ``A z = 0``. The physical solution is unique, and the ghost values are not. `Bramble.ghost_kernel(A, Wₕ)` returns a sparse basis of that kernel, one column per corner here, found node by node and never as a dense null space. `pde_solve(A, F; gauge = Z)` solves the bordered system ``[A\ Z; Z^T\ 0]`` and returns the solution with ``Z^T u = 0``:

```@example forms
Ω_rect = domain(interval(0.0, 1.0) × interval(0.0, 2.0))
Ωₕ_rect = mesh(Ω_rect, (9, 11), (true, true))
change_points!(Ωₕ_rect, markers(Ω_rect), (collect(range(0.0, 1.0, length = 9) .^ 1.5),
    2 .* collect(range(0.0, 1.0, length = 11) .^ 1.5)))

W_rect = gridspace(Ωₕ_rect; ghosts = :boundary)
A_rect = assemble(form(W_rect, W_rect, (u, v) -> innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)))
Z_rect = Bramble.ghost_kernel(A_rect, W_rect)

n_phys = npoints(Ωₕ_rect)
@assert size(Z_rect) == (ndofs(W_rect), 4)
@assert all(>(n_phys), rowvals(Z_rect))
@assert norm(A_rect * Z_rect, Inf) <= 1e-10 * norm(A_rect, Inf)

size(Z_rect), norm(A_rect * Z_rect, Inf)
```

The kernel has 4 columns, each supported on the two ghosts of a corner, and `A * Z` vanishes to rounding. The physical values come first in the vector of unknowns, then the ghosts. To check the solve, choose a physical vector, set the ghosts to zero, and form the load `F = A x`. The gauged solve returns the physical values of `x`, and the load of a vector shifted by any combination of the kernel columns gives the same answer:

```@example forms
using Random

Random.seed!(7)
x_phys = rand(n_phys)
x_ghost0 = [x_phys; zeros(ndofs(W_rect) - n_phys)]

u_gauged = pde_solve(A_rect, A_rect * x_ghost0; gauge = Z_rect)
u_shifted = pde_solve(A_rect, A_rect * (x_ghost0 + Z_rect * [1.0, -2.0, 3.0, 0.5]);
    gauge = Z_rect)

@assert maximum(abs, u_gauged[1:n_phys] - x_phys) <= 1e-7
@assert norm(Z_rect' * u_gauged, Inf) <= 1e-10 * norm(u_gauged, Inf)
@assert maximum(abs, u_shifted - u_gauged) <= 1e-6
@assert norm(A_rect * u_gauged - A_rect * x_ghost0, Inf) <= 1e-8 * norm(A_rect, Inf)

maximum(abs, u_gauged[1:n_phys] - x_phys), norm(Z_rect' * u_gauged, Inf)
```

The physical error is about ``10^{-10}`` here and is only as small as the conditioning of `A` away from its kernel allows: these forms are ill conditioned, and the accuracy of the physical part is near ``10^{-8}`` relative in general. A form that never reads a ghost leaves each ghost in the kernel as one column. With a single ghosted facet no node carries two ghosts and a form that reads them through `Δₕ` alone has no ghost-only kernel. A kernel vector spread over the ghosts of several nodes, which a form that differences the ghosts along their own facet produces, is not found: `ghost_kernel` returns no column for it, and a gauged solve with such a `Z` still gives the right physical part, while the ghost components it leaves out are arbitrary. With `mean_zero = w` as well, the border is `[w Z]`.

#### Checking that the data are compatible

The Neumann data of ``-\Delta u = f`` must satisfy ``\int f = -\oint \partial_\eta u``. A source off by a constant leaves a system with no solution: the bordered solve returns a ``\lambda`` that measures the part of `F` no `u` reaches, and the answer looks plausible. `Bramble.neumann_compatibility(problem, Ωₕ)` checks the data. `problem` is a function of a mesh that returns `(A, F, w)`, the matrix, the load and the weights of the discrete average, because the check builds the problem on `Ωₕ` and on copies of it refined once and twice. For compatible data the removed load is a quadrature error and falls by about four per refinement. For an incompatible datum it tends to a constant, the integral of the discrepancy, and the check logs a warning that reports the relative defects, ``\lambda`` and the ratios. Nothing is thrown, and the result is a `NamedTuple` with `compatible`. The problem below is the square from above on a random non-uniform mesh, once as it is and once with the source shifted by ``0.2``:

```@example forms
using Logging

struct WarningLog <: Logging.AbstractLogger
    messages::Vector{String}
end
Logging.min_enabled_level(::WarningLog) = Logging.Warn
Logging.shouldlog(::WarningLog, args...) = true
function Logging.handle_message(log::WarningLog, level, message, args...; kwargs...)
    push!(log.messages, string(message))
    return nothing
end

function neumann_problem(shift)
    return function (Ωₕ)
        C = constrained(gridspace(Ωₕ; ghosts = :boundary), u -> (
            :xmin => ∂ₙ(u) => (p -> -grad_sq(p)[1]), :xmax => ∂ₙ(u) => (p -> grad_sq(p)[1]),
            :ymin => ∂ₙ(u) => (p -> -grad_sq(p)[2]), :ymax => ∂ₙ(u) => (p -> grad_sq(p)[2])))
        A, F = assemble(form(C, C, (u, v) -> innerₕ(-Δₕ(u), v)),
            form(C, v -> innerₕ(p -> f_sq(p) + shift, v)))
        w = assemble(form(gridspace(Ωₕ), v -> innerₕ(p -> 1.0, v)))
        return A, F, w
    end
end

Random.seed!(20261016)
Ωₕ_rand = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (17, 17), (false, false))

function check_compatibility(shift)
    log = WarningLog(String[])
    result = Logging.with_logger(log) do
        Bramble.neumann_compatibility(neumann_problem(shift), Ωₕ_rand)
    end
    return result, log.messages
end

ok, ok_warnings = check_compatibility(0.0)
bad, bad_warnings = check_compatibility(0.2)

@assert ok.compatible && isempty(ok_warnings)
@assert ok.defect < 1e-4
@assert !bad.compatible && length(bad_warnings) == 1
@assert occursin("incompatible", only(bad_warnings))
@assert isapprox(bad.λ, 0.2; atol = 1e-3) && isapprox(bad.second_ratio, 1.0; atol = 0.1)

(ok.compatible, ok.defect), (bad.compatible, bad.λ, bad.second_ratio)
```

The compatible problem raises no warning and its relative defect is below ``10^{-4}``. The shifted one warns once. Its ``\lambda`` is ``0.2``, the shift, on the mesh and on both refinements, so the ratio of the removed loads is ``1`` where a quadrature error would give about ``4``. The mesh `Ωₕ` is not modified, and a mesh that carries markers of its own needs the domain's `markers(Ω)` as a third argument so that the refinement can evaluate them again.

The check has limits. A defect on the refined mesh below `5e-3` counts as rounding, so a small incompatibility is not detected. Compatible data with a singular flux at a boundary point, whose quadrature error falls slower than the factor of four, can be reported incompatible. When the defect is above the floor the problem is built on a mesh with 16 times the points of `Ωₕ`.

### Pure Neumann as one penalised form

The bordered solve above keeps `A` and the weights apart. A form can also carry the constraint itself, as a rank-one penalty ``\sigma\,(1, u)_h (1, v)_h``, and then it assembles to a single matrix ``A + \sigma\, w w^T``, where ``w`` is the weight vector of the previous section. The matrix is nonsingular because the penalty fixes the constant. Take the Neumann problem with zero flux on the graded square: `inner₊(∇ₕ(u), ∇ₕ(v))` on the space without ghosts is its stiffness matrix. The penalty is `σ * innerₕ(1, u) * innerₕ(1, v)` with ``\sigma = \gamma / h_{max}``:

```@example forms
using LinearAlgebra: norm

σ_pen = 1.0 / hₘₐₓ(Ωₕ_sq)   # γ = 1
a_pen = form(W_plain_sq, W_plain_sq,
    (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + σ_pen * innerₕ(1, u) * innerₕ(1, v))
K_pen = assemble(a_pen)

# the load with the data made compatible: 1ᵀF = 0
F_raw = assemble(form(W_plain_sq, v -> innerₕ(f_sq, v)))
F_pen = F_raw - (sum(F_raw) / sum(w_sq)) * w_sq

typeof(K_pen).name.name, size(K_pen), Base.summarysize(K_pen) < 10 * Base.summarysize(K_pen.A)
```

`assemble` returns a `Bramble.RankOneUpdate`: the sparse matrix `K_pen.A` of the other terms, `σ` and the two weight vectors, with no dense block. Its products cost one product by `A` and two passes over the vectors. `K_pen'` and `transpose(K_pen)` are lazy wrappers of `K_pen`, and read its parts when they are applied. `pde_solve(K_pen, F_pen)` does not form the dense matrix either. It solves the sparse bordered system

```math
\begin{pmatrix} A & w \\ w^T & -1/\sigma \end{pmatrix}
\begin{pmatrix} u \\ \mu \end{pmatrix} =
\begin{pmatrix} F \\ 0 \end{pmatrix},
```

whose second row gives ``\mu = \sigma w^T u``, so its first row is ``A u + \sigma w (w^T u) = F``. For compatible data ``w^T u = 0``, so the solution is the `mean_zero` one for every ``\sigma > 0``. The next block checks that against the `mean_zero` solve and the dense solve, and shows the relative differences:

```@example forms
A_pen = assemble(form(W_plain_sq, W_plain_sq, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))))

u_pen = pde_solve(K_pen, F_pen)
u_mz = pde_solve(A_pen, F_pen; mean_zero = w_sq)
u_dense = Matrix(K_pen) \ F_pen

rel_mz = norm(u_pen - u_mz) / norm(u_mz)
rel_dense = norm(u_pen - u_dense) / norm(u_dense)

@assert Matrix(K_pen) ≈ Matrix(A_pen) + σ_pen * w_sq * w_sq'
@assert abs(sum(F_pen)) <= 1e-12 * norm(F_pen)
@assert abs(dot(w_sq, u_pen)) <= 1e-10 * norm(w_sq) * norm(u_pen)
@assert rel_mz < 1e-10 && rel_dense < 1e-10
@assert !all(iszero, u_pen)

rel_mz, rel_dense
```

The difference from the `mean_zero` solve is about ``10^{-14}/\gamma``: a small ``\sigma`` makes ``-1/\sigma`` large against the entries of `A` and costs digits, so take ``\gamma \ge 10^{-2}``. A large ``\sigma`` costs nothing. [`pde_solve`](@ref) documents the figures for the `RankOneUpdate` method.

Four rules apply to the term. The `σ` is read when the form is assembled, so with a `Ref` factor call `assemble!(K_pen, a_pen)` after changing it, before the next solve. With `dirichlet`, the Dirichlet rows of `K_pen` are identity rows and the penalty's row vector is zeroed there. One rank-one term per form: the sum of two is refused with an `ArgumentError`, so add the scalars when they share the functionals. The term also builds on a ghost space, where `assemble(a, l)` reduces it with the closure. A composite ghost space is covered by the support table in [the space internals](../internals/space.md).

### Data per facet

A scalar datum on `:boundary` gives one value per node. A corner node lies on two facets, though. When the data differ from facet to facet, give each facet its own `inner_Γ` term or, as above, its own relation. A facet's term carries that facet's weight, which is a half cell at the corner. With the same function on every facet, the two spellings agree:

```@example forms
q(p) = 1 + p[1] + 2p[2]
l_all = assemble(form(W_plain_sq, v -> inner_Γ(q, v; markers = (:boundary,))))
l_each = assemble(form(W_plain_sq, v -> inner_Γ(q, v; markers = (:xmin,)) +
    inner_Γ(q, v; markers = (:xmax,)) + inner_Γ(q, v; markers = (:ymin,)) +
    inner_Γ(q, v; markers = (:ymax,))))

@assert l_all ≈ l_each

norm(l_all - l_each)
```

With different data the corner entry is the sum of the facet terms, each at its own weight:

```@example forms
data_x(p) = 10.0 + p[2]
data_y(p) = -3.0 + p[1]
l_corner = assemble(form(W_plain_sq, v -> inner_Γ(data_x, v; markers = (:xmin,)) +
    inner_Γ(data_y, v; markers = (:ymin,))))

hx, hy = grid[2] - grid[1], grid[2] - grid[1]
expected = (hy / 2) * data_x((0.0, 0.0)) + (hx / 2) * data_y((0.0, 0.0))
@assert l_corner[1] ≈ expected

l_corner[1], expected
```

Ghost spaces are for `assemble`, and for `semidiscretize` and `ode_problem` once `constrained` has eliminated every ghost. `matrix_free_operator(a)` applies a ghost-space form, one product at a time per operator. `kronecker_operator(a)` works where the closure and the form are separable and both spaces share one mesh object. The device backends refuse ghost spaces with an `ArgumentError`. Which entry points accept a composite space with a ghost leaf is in the support table of [the space internals](../internals/space.md).

## Threading

Assembly reads the execution policy off the space it is given. A space whose backend says
`Parallel()` assembles on several threads, and nothing about the call changes.

## Reference

### Point sources

A continuous source ``f(x)`` is a density, integrated over cell volumes as
``\ell(v) = (f, v)_h``. A point source is a singular functional, the Dirac distribution
``S\,\delta(x - x_0)``:

```math
\ell(v) = S \, v(x_0).
```

Writing [`dirac`](@ref)`(x0, strength)` inside `innerₕ` or `inner₊` evaluates it without
cell-measure weighting:

```@example forms
l_pt = form(Wₕ, v -> innerₕ(dirac(0.5, 2.0), v))
sum(assemble(l_pt))
```

When ``x_0`` is a grid node, the strength ``S`` lands in that degree of freedom. When it
falls inside a cell, ``S`` is spread over the ``2^D`` surrounding vertices with multilinear
weights, and the total is conserved:

```@example forms
l_off = form(Wₕ, v -> innerₕ(dirac(0.35, 1.0), v))
sum(assemble(l_off)) ≈ 1.0
```

Several sources take a vector of coordinates and a vector of strengths. Each coordinate is a
tuple (a 1-tuple in 1D), because a bare vector of reals is read as the coordinates of one
point:

```@example forms
l_multi = form(Wₕ, v -> innerₕ(dirac([(0.2,), (0.7,)], [1.0, -1.0]), v))
sum(assemble(l_multi)) ≈ 0.0
```

A strength wrapped in a `Ref` updates live, without rebuilding the form:

```@example forms
S_live = Ref(1.0)
l_dyn_pt = form(Wₕ, v -> innerₕ(dirac(0.5, S_live), v))
b_dyn_pt = assemble(l_dyn_pt)
S_live[] = 3.5
assemble!(b_dyn_pt, l_dyn_pt)
sum(b_dyn_pt) ≈ 3.5
```

### Restricting a term to part of the mesh

`innerₕ`, `inner₊` and the directional products take a `markers` keyword, which restricts
the sum to the union of the regions the labels name:

```@example forms
a_left = form(Wd, Wd, (u, v) -> innerₕ(u, v; markers = (:left,)))
size(assemble(a_left))
```

Every mesh also carries `:boundary` and `:interior`, computed from its shape and needing no
label in `domain(...)`:

```@example forms
a_boundary = form(Wd, Wd, (u, v) -> innerₕ(u, v; markers = (:boundary,)))
size(assemble(a_boundary))
```

This is a masked *sum* of cell measures, not a surface integral. A masked `innerₕ` scales
like ``h`` and vanishes under refinement, where a true boundary integral does not. A Neumann
or Robin term needs the integral, which is [`inner_Γ`](@ref):

```@example forms
β, g = 1.7, x -> 2.0 + x[1]
a_robin = form(Wd, Wd, (u, v) -> inner_Γ(β * u, v; markers = (:xmax,)))
l_neumann = form(Wd, v -> inner_Γ(g, v; markers = (:xmax,)))
sum(assemble(l_neumann))
```

That is ``g(x_N)`` exactly, whatever the refinement. In 1D a face is a point of measure 1,
so the term is the endpoint pairing ``g(x_N)\,v(x_N)``. In 2D the same expression is an edge
integral weighted by the transverse half-spacing, and in 3D a face integral. `inner_Γ`
integrates over whole coordinate faces. A label is read from its definition: a marker in
`domain(...)` selects the faces its points cover, so under `:left => :ymin` the label
`:left` is the bottom edge. A label that is not a marker is a face name (`:xmin` to
`:zmax`, or a viewpoint alias) and needs no marker; `:boundary` is a marker every mesh holds,
the union of its faces.

A marker that exists nowhere the term reaches is an error, not a silent zero:

```@example forms
try
    assemble(form(Wd, Wd, (u, v) -> innerₕ(u, v; markers = (:nope,))))
catch e
    showerror(stdout, e)
end
```

On a composite space, a marker used without naming a component reaches every diagonal block
and must exist on every leaf. Write the term per component otherwise.

### The outward normal

A flux term ``\int_\Gamma \mathbf{F} \cdot \boldsymbol{\eta}\, v`` needs the outward normal
at each boundary point. [`η`](@ref) destructures and indexes into its per-coordinate
components like `∇ₕ`:

```@example forms
Ω2 = mesh(domain(box((0.0, 0.0), (1.0, 1.0)),
    :xmin => :xmin, :xmax => :xmax, :ymin => :ymin, :ymax => :ymax), (6, 6), (true, true))
W2 = gridspace(Ω2)
ηₓ, ηᵧ = η
F1 = Rₕ(W2, x -> 1.0 + x[1])
F2 = Rₕ(W2, x -> 2.0 + x[2])
l_flux = form(W2, v -> inner_Γ(F1 * ηₓ + F2 * ηᵧ, v; markers = (:xmax,)))
sum(assemble(l_flux))
```

### Automatic simplification

`form` resolves the expression once and simplifies it before storing it. The assembler
routes summands one at a time, so every top-level `+` is a separate sweep over the mesh.
The pass is small and keeps terms as written: it does not combine like terms or factor a
shared argument. Two identical terms stay two terms and are summed at assembly:

```@example forms
a_dup = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + innerₕ(u, v))
expression(a_dup)     # two terms, as written
```

```@example forms
Matrix(assemble(a_dup)) ≈ 2 .* Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))))
```

Write `2 * innerₕ(u, v)` when one sweep is wanted. Nested scales fold into one:

```@example forms
a_fold = form(Wₕ, Wₕ, (u, v) -> 2 * (3 * innerₕ(u, v)))
expression(a_fold)    # 6 * innerₕ(u, v)
```

A zero-scaled term leaves nothing behind, so a coefficient can switch a term off without a
branch around the form:

```@example forms
a_off = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + 0 * inner₊(∇ₕ(u), ∇ₕ(v)))
expression(a_off)     # innerₕ(u, v): the zero term is gone
```

```@example forms
Matrix(assemble(a_off)) ≈ Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))))
```

!!! tip "Try this"
    Replace `0 *` by `0.0 *` in `a_off` and compare `expression(a_off)`.
    A `Float64` coefficient is only known at run time, so the term stays.

Two limits apply. A rule whose outcomes have different node types fires only when the
compiler settles it from types, so the scalar of a dropped zero term must be an `Integer`.
A `Float64` known at run time, or a `Ref`, leaves the term in: same numbers, one extra
sweep. And the pass stops at an inner product's own arguments, so a scalar buried in a
difference such as `innerₕ(D₋ₓ(2 * u), v)` is invisible to it. Write
`2 * innerₕ(D₋ₓ(u), v)` instead.

The [internals page on forms](../internals/form.md) documents the stencil algebra and the
exact rewrite rules.
