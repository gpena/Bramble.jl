# # Memory scaling with a matrix-free Kronecker operator
#
# Every bilinear form assembled so far in this manual becomes one `SparseMatrixCSC`: exact,
# but its storage grows with the number of nonzeros, which on a Cartesian mesh grows with
# the number of unknowns. A separable form -- one whose assembled matrix is an exact sum of
# Kronecker products of one-dimensional factors -- never needs that matrix at all: applying
# it is sum factorisation over the per-axis factors, so the storage is `O(D \cdot n)` instead
# of `O(n^D)` stored nonzeros. This page builds that operator, measures what it actually
# costs against the matrix it replaces, and checks that it still computes the right answer.
# Every number below was produced by the code shown.
#
# ## Problem
#
# Two related problems on the unit cube, chosen so both the matrix-free operator and the
# fast-diagonalisation solve introduced below get to run:
#
# ```math
# u - \Delta u = g \text{ in } \Omega = (0,1)^3, \qquad \partial_n u = 0 \text{ on } \partial\Omega,
# ```
#
# with manufactured solution ``u_{\text{exact}}(x, y, z) = \cos(\pi x)\cos(\pi y)\cos(\pi z)``,
# whose normal derivative vanishes on every face of the cube -- exactly the boundary
# condition an *unconstrained* assembly imposes, so no `dirichlet` handling is needed here.
# The mass term keeps the discrete operator well-posed without one.
#
# ## Building the mesh and the form

using Bramble
using Kronecker
using LinearSolve: LinearProblem, solve, KrylovJL_CG

sol(x) = cospi(x[1]) * cospi(x[2]) * cospi(x[3])
rhs(x) = (1 + 3 * pi^2) * sol(x)

n = 41
Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ = mesh(Ω, (n, n, n), true)
Wₕ = gridspace(Ωₕ)

a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
ndofs(Wₕ)

# `41^3 = 68921` unknowns -- large enough for the memory gap below to be worth looking at,
# small enough that this page, including both direct solves further down, still runs in
# seconds.
#
# ## Separability
#
# `is_separable` walks `a`'s resolved AST and answers whether every term is one of the two
# shapes a Kronecker product can represent: `innerₕ(u, v)` (the identity on every axis) or
# `inner₊` of a backward difference along one axis (what `∇ₕ(u)` expands into, one term per
# axis). `a` above is exactly that sum:

is_separable(a)

@test is_separable(a) #src

# A grid-function coefficient breaks the shape match -- it has no tensor structure to factor
# out, so `is_separable` refuses it rather than guessing:

fₕ = Rₕ(Wₕ, x -> 1.0 + x[1])
is_separable(form(Wₕ, Wₕ, (u, v) -> innerₕ(fₕ * u, v)))

@test !is_separable(form(Wₕ, Wₕ, (u, v) -> innerₕ(fₕ * u, v))) #src

# The same refusal covers a region restriction, an interpolation node, a surface (`InnerGamma`)
# weight, a composite space, a 1D mesh (nothing to factor), and any difference family other
# than the plain backward one `∇ₕ` builds -- forward, centered, star, cross-weighted,
# averages, jumps. Every one of these is a false negative rather than a wrong answer: `a`
# would still assemble and solve the ordinary way, just without the fast path below.
#
# ## The operator, and what it costs against the matrix it replaces
#
# `kronecker_operator` builds the matrix-free operator without ever forming the
# `68921 x 68921` matrix; `assemble` builds that matrix, for comparison:

K = kronecker_operator(a)
A = assemble(a)

bytes_kronecker = Base.summarysize(K)
bytes_csc = Base.summarysize(A)
(dofs = ndofs(Wₕ), bytes_kronecker = bytes_kronecker, bytes_csc = bytes_csc,
    kronecker_over_csc_percent = round(100 * bytes_kronecker / bytes_csc; digits = 4))

# Bracketed rather than pinned to one figure: allocator bookkeeping moves the byte counts a #src
# little between Julia versions, the ratio itself is what this page is making a claim about. #src
@test ndofs(Wₕ) == 68921                             #src
@test bytes_kronecker < 0.001 * bytes_csc             #src
@test bytes_csc > 15_000_000                          #src

# The operator holds three `41`-length one-dimensional factors per term instead of the
# assembled matrix's stored nonzeros, so it costs a fraction of a percent of `A` here -- and
# the gap only widens with `n`, since `bytes_csc` grows like `n^3` while `bytes_kronecker`
# grows like `n`. Measured separately (not by this page, to keep this one fast): on a
# uniform `60x60x60` mesh with the same mass-plus-stiffness form, `test/form/kronecker.jl`
# (gpena/Bramble.jl#162) records `19,432` bytes for the operator against `61,948,960` bytes
# for the equivalent `SparseMatrixCSC` -- about `0.03%`, for a 216,000-unknown problem this
# page does not build directly.
#
# ## Solving it: iteratively, through the operator itself
#
# `K` subtypes `AbstractMatrix`, so `LinearSolve`'s `KrylovJL_CG` runs against it exactly as
# it would against `A`, applying `K` by `mul!` (sum factorisation) rather than a sparse
# matrix-vector product:

gₕ = element(Wₕ)
avgₕ!(gₕ, rhs)
l = form(Wₕ, v -> innerₕ(gₕ, v))
F = assemble(l)

xref = A \ F   # the ordinary sparse solve, kept only as the reference the two solves below agree with

cg_kwargs = (; reltol = 1e-10, abstol = 1e-10, maxiters = 2000)
sol_cg = solve(LinearProblem(K, F), KrylovJL_CG(); cg_kwargs...)
maximum(abs.(sol_cg.u .- xref))

@test maximum(abs.(sol_cg.u .- xref)) < 1.0e-6 #src

# ## Solving it: directly, by fast diagonalisation
#
# A separable, constant-coefficient system like this one also admits a direct solve that
# never factorises a matrix at all: `fdm_solve` diagonalises each axis's own generalised
# eigenproblem once and combines the `D` small eigendecompositions instead. It lives in the
# `Kronecker.jl` extension rather than in `Bramble` itself, since the fast-diagonalisation
# solve needs that package's types. `Bramble` declares the name and exports it, and the
# extension attaches its methods to it once `using Kronecker` has loaded, so it is spelled
# plainly here.

x_fdm = fdm_solve(a, F)
maximum(abs.(x_fdm .- xref))

@test maximum(abs.(x_fdm .- xref)) < 1.0e-9 #src

# ## Checking the answer
#
# Both solves above only proved they agree with the ordinary sparse solve -- not that any of
# the three actually solved the problem posed at the top of the page. That needs the
# manufactured solution:

uₕ = element(Wₕ)
uₕ .= x_fdm
normₕ(uₕ .- Rₕ(Wₕ, sol))

# Bracketed away from zero as well as above: an exactly-zero error would mean the         #src
# manufactured solution had been reproduced by construction rather than solved for.       #src
@test 1.0e-5 < normₕ(uₕ .- Rₕ(Wₕ, sol)) < 1.0e-3                                          #src
#
# ## The operator's real limit: no boundary constraint of its own
#
# `K` and `kronecker_operator` carry no Dirichlet handling: a `KroneckerLinearOperator` is
# built for the whole grid, boundary rows included, because a boundary-restricted term has
# no tensor structure of its own to factor. The problem above sidesteps this by using an
# unconstrained (natural) boundary condition instead. `fdm_solve` alone reaches further,
# through `dirichlet = :boundary`: since a point is interior in the domain iff it is
# interior along *every* axis, restricting each axis's own factors to its interior
# (`2:end-1`) and running the same derivation on the restriction gives homogeneous Dirichlet
# on the whole boundary, still without ever assembling a matrix:

sol_d(x) = sinpi(x[1]) * sinpi(x[2]) * sinpi(x[3])   # vanishes on every face
rhs_d(x) = 3 * pi^2 * sol_d(x)

a_d = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
is_separable(a_d)

@test is_separable(a_d) #src

g_d = element(Wₕ)
avgₕ!(g_d, rhs_d)
l_d = form(Wₕ, v -> innerₕ(g_d, v))
bcs_d = dirichlet_constraints(Ω, :boundary => sol_d)
F_d = assemble(l_d; dirichlet = bcs_d)

A_d = assemble(a_d; dirichlet = :boundary)
xref_d = A_d \ F_d
x_fdm_d = fdm_solve(a_d, F_d; dirichlet = :boundary)
maximum(abs.(x_fdm_d .- xref_d))

@test maximum(abs.(x_fdm_d .- xref_d)) < 1.0e-8 #src

u_d = element(Wₕ)
u_d .= x_fdm_d
normₕ(u_d .- Rₕ(Wₕ, sol_d))

@test 1.0e-5 < normₕ(u_d .- Rₕ(Wₕ, sol_d)) < 1.0e-3 #src
#
# There is no matching Dirichlet path for `K` itself -- `LinearSolve`'s `KrylovJL_CG` above
# only ever ran against the unconstrained problem. A reader reaching for the matrix-free
# operator on a Dirichlet problem meets this limit directly, not as a caveat in a docstring.
#
# ## See also
#
#   - `kronecker_operator` is written as a plain code span throughout this page rather than
#     a link, along with `is_separable`, `KroneckerLinearOperator` and `fdm_solve`: none of
#     the four have an `@docs` entry on the [API reference](../api.md) yet.
#   - [Linear Poisson](poisson_linear.md) assembles the same discrete Laplacian into a
#     `SparseMatrixCSC` directly, with Dirichlet conditions from the start.
#   - The [forms tutorial](../tutorials/form.md) introduces `innerₕ`, `inner₊` and `∇ₕ` one
#     at a time.
