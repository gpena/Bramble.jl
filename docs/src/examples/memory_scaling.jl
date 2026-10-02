# # Memory scaling with a matrix-free Kronecker operator
#
# Every bilinear form assembled so far in this manual becomes one `SparseMatrixCSC`: exact,
# but its storage grows with the number of nonzeros, which on a Cartesian mesh grows with
# the number of unknowns. A separable form, one whose assembled matrix is an exact sum of
# Kronecker products of one-dimensional factors, never needs that matrix at all. Applying
# it is one fused pass over the per-axis factors, so the storage is `O(D \cdot n)` instead
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
# whose normal derivative vanishes on every face of the cube. That is exactly the boundary
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

# `41^3 = 68921` unknowns, large enough for the memory gap below to be worth looking at,
# small enough that this page, including both direct solves further down, still runs in
# seconds.
#
# ## Mathematical background
#
# The operator this page builds is a sum of Kronecker products, and `fdm_solve` exploits one
# property of such sums. Both are shown below on matrices small enough to print, using plain
# dense matrices and `kron` from `LinearAlgebra`, so that the definition is visible. The lazy
# type loaded above enters only once the operator itself is built.
#
# ### The Kronecker product
#
# For ``A \in \mathbb{R}^{m \times n}`` and ``B \in \mathbb{R}^{p \times q}``, the Kronecker
# product ``A \otimes B`` is the block matrix whose ``(i, j)`` block is the scalar
# ``a_{ij}`` times the whole of ``B``:
#
# ```math
# A \otimes B =
# \begin{pmatrix}
# a_{11} B & \cdots & a_{1n} B \\
# \vdots & & \vdots \\
# a_{m1} B & \cdots & a_{mn} B
# \end{pmatrix}
# \in \mathbb{R}^{mp \times nq}.
# ```

using LinearAlgebra

A₀ = [1 2; 3 4]
B₀ = [0 5; 6 7]
kron(A₀, B₀)

# The top-left ``2 \times 2`` block is ``1 \cdot B_0``, the top-right block is ``2 \cdot B_0``,
# and so on. Two ``2 \times 2`` factors give a ``4 \times 4`` matrix, as the size ``mp \times
# nq`` predicts.

@test kron(A₀, B₀)[1:2, 3:4] == 2 * B₀ #src
@test size(kron(rand(2, 3), rand(4, 5))) == (8, 15) #src

# Products of Kronecker products factor block by block: ``(A \otimes B)(C \otimes D) = AC
# \otimes BD`` whenever the products ``AC`` and ``BD`` are defined. The two sides agree
# entry for entry:

C₀ = [2 0; 1 3]
D₀ = [1 1; 0 2]
kron(A₀, B₀) * kron(C₀, D₀) == kron(A₀ * C₀, B₀ * D₀)

@test kron(A₀, B₀) * kron(C₀, D₀) == kron(A₀ * C₀, B₀ * D₀) #src

# ### Applying a Kronecker product without forming it
#
# Stacking the columns of a matrix ``X \in \mathbb{R}^{q \times n}`` into one vector gives
# ``\operatorname{vec}(X)``, which is what `vec(X)` returns. The two factors then act on
# ``X`` separately:
#
# ```math
# (A \otimes B)\,\operatorname{vec}(X) = \operatorname{vec}(B X A^{\mathsf T}).
# ```
#
# With ``A \in \mathbb{R}^{m \times n}`` and ``B \in \mathbb{R}^{p \times q}``, applying
# ``A \otimes B`` to a vector therefore takes two small products on a reshaped vector, and
# the ``mp \times nq`` matrix is never built:

X₀ = [1 2; 3 4]
y_big = kron(A₀, B₀) * vec(X₀)
y_small = vec(B₀ * X₀ * A₀')
(y_big, y_small)

@test y_big == y_small #src
@test kron(A₀, B₀) * vec(X₀) == vec(B₀ * X₀ * A₀') #src

# For square factors of sizes ``m`` and ``n`` the saving is large. Storing ``A`` and ``B``
# takes ``m^2 + n^2`` numbers against ``m^2 n^2`` for the product, and applying them costs
# about ``2mn(m + n)`` floating-point operations (one ``n \times n`` times ``n \times m``
# product, one ``n \times m`` times ``m \times m`` product) against ``2m^2 n^2``:

function counts(m, n)
    (stored_factors = m^2 + n^2, stored_matrix = m^2 * n^2,
        flops_factors = 2m * n * (m + n), flops_matrix = 2m^2 * n^2)
end
counts(41, 41)

@test counts(41, 41).stored_matrix == 41^4 #src
@test counts(41, 41).flops_factors < counts(41, 41).flops_matrix ÷ 20 #src

# ### The Kronecker sum
#
# The Kronecker sum of square matrices ``A \in \mathbb{R}^{m \times m}`` and
# ``B \in \mathbb{R}^{n \times n}`` is
#
# ```math
# A \oplus B = A \otimes I_n + I_m \otimes B,
# ```
#
# where ``I_n`` is the identity of size ``n``. For eigenpairs ``A v = \lambda v`` and
# ``B w = \mu w``,
#
# ```math
# (A \oplus B)(v \otimes w) = (\lambda + \mu)(v \otimes w),
# ```
#
# so the eigenvalues of ``A \oplus B`` are all sums ``\lambda_i(A) + \mu_j(B)``, with the
# products ``v_i \otimes w_j`` of the factors' eigenvectors as eigenvectors. A symmetric
# example:

Aₛ = [2.0 -1.0; -1.0 2.0]
Bₛ = [1.0 0.0 0.0; 0.0 3.0 1.0; 0.0 1.0 3.0]
Sₖ = kron(Aₛ, Matrix(1.0I, 3, 3)) + kron(Matrix(1.0I, 2, 2), Bₛ)
λ = eigvals(Aₛ)
μ = eigvals(Bₛ)
(sort(eigvals(Sₖ)), sort(vec(λ .+ μ')))

@test sort(eigvals(Sₖ)) ≈ sort(vec(λ .+ μ')) #src

# The eigenvector claim is checked the same way, on one pair:

Vₐ = eigvecs(Aₛ)
Wᵦ = eigvecs(Bₛ)
v₂ = kron(Vₐ[:, 2], Wᵦ[:, 3])
norm(Sₖ * v₂ - (λ[2] + μ[3]) * v₂)

@test Sₖ * v₂ ≈ (λ[2] + μ[3]) * v₂ #src
@test Sₖ ≈ kron(Aₛ, Matrix(1.0I, 3, 3)) + kron(Matrix(1.0I, 2, 2), Bₛ) #src
#
# The factors of the operator below are not abstract matrices: each one is a one-dimensional
# form that Bramble assembles, and the next subsections build the page's operator from them.
#
# ### The one-dimensional factors
#
# Take the one-dimensional mass form `innerₕ(u, v)` and stiffness form
# `inner₊(∇ₕ(u), ∇ₕ(v))` on a small non-uniform mesh, four points at ``0, 0.2, 0.6, 1``:

Ωx = mesh(domain(interval(0.0, 1.0)), 4, true)
Bramble.set_points!(Ωx, [0.0, 0.2, 0.6, 1.0])
Wx = gridspace(Ωx)

Mx = Matrix(assemble(form(Wx, Wx, (u, v) -> innerₕ(u, v))))

# The mass matrix is diagonal, and its entries are the cell widths ``0.1, 0.3, 0.4, 0.2`` of
# the mesh, which are not equal. The stiffness matrix is tridiagonal, with the inverse
# spacings of the same mesh as its entries:

Kx = Matrix(assemble(form(Wx, Wx, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))))

@test Mx ≈ Diagonal([0.1, 0.3, 0.4, 0.2]) #src
@test issymmetric(Kx) && all(iszero, Kx * ones(4)) #src

# Every row of ``K_x`` sums to zero: constants are in its kernel, which is the unconstrained
# (natural) boundary condition again. Two more meshes, with their own points, give the
# matching factors on the other axes:

Ωy = mesh(domain(interval(0.0, 1.0)), 3, true)
Bramble.set_points!(Ωy, [0.0, 0.5, 1.0])
Ωz = mesh(domain(interval(0.0, 1.0)), 3, true)
Bramble.set_points!(Ωz, [0.0, 0.3, 1.0])

function assemble_factors(Ω)
    (W = gridspace(Ω);
        (Matrix(assemble(form(W, W, (u, v) -> innerₕ(u, v)))),
            Matrix(assemble(form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))))))
end
My, Ky = assemble_factors(Ωy)
Mz, Kz = assemble_factors(Ωz)
(My, Ky)

# ### Two and three dimensions
#
# On the tensor mesh of those axes, a grid function is stored with the first axis varying
# fastest, so reshaping the vector of values to ``n_x \times n_y`` gives a matrix whose
# column index is ``y``. That is the `vec` convention used above, and it fixes which factor
# sits where in a Kronecker product: the slowest axis comes first, as the left factor. The
# form ``u v + \nabla u \cdot \nabla v`` then assembles into
#
# ```math
# A_{2} = M_y \otimes M_x + K_y \otimes M_x + M_y \otimes K_x,
# ```
#
# one term per summand of the form, with the stiffness matrix of an axis in the term that
# differentiates along it and the mass matrix of the other axes everywhere else. In three
# dimensions the same rule gives four terms, with ``M_z \otimes M_y \otimes M_x`` for the
# mass term. Assembling the forms on the tensor meshes and comparing:

Ω₂ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 3), true)
Bramble.change_points!(Ω₂, [points(Ωx), points(Ωy)])
W₂ = gridspace(Ω₂)
a₂ = form(W₂, W₂, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
A₂ = assemble(a₂)

Ω₃ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 3, 3), true)
Bramble.change_points!(Ω₃, [points(Ωx), points(Ωy), points(Ωz)])
W₃ = gridspace(Ω₃)
a₃ = form(W₃, W₃, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
A₃ = assemble(a₃)

kron_2d = kron(My, Mx) + kron(Ky, Mx) + kron(My, Kx)
kron_3d = kron(Mz, My, Mx) + kron(Kz, My, Mx) + kron(Mz, Ky, Mx) + kron(Mz, My, Kx)
(norm(Matrix(A₂) - kron_2d), norm(Matrix(A₃) - kron_3d))

@test Matrix(A₂) ≈ kron(My, Mx) + kron(Ky, Mx) + kron(My, Kx) #src
@test Matrix(A₃) ≈ kron(Mz, My, Mx) + kron(Kz, My, Mx) + kron(Mz, Ky, Mx) + kron(Mz, My, Kx) #src

# The assembled matrices equal the Kronecker sums of the one-dimensional factors up to
# rounding, on meshes whose points are unevenly spaced along every axis. This is the
# structure `is_separable` looks for term by term: each term of the form must split into one
# factor per axis. A grid-function coefficient does not split, since its values are not a
# product of one function of ``x`` and one of ``y``, and that is why it is refused below.

is_separable(a₂) && is_separable(a₃)

@test is_separable(a₂) && is_separable(a₃) #src

# ### Why the storage is linear in the points per axis
#
# Each axis contributes only its own two matrices, ``M`` and ``K``, whatever the dimension: a
# diagonal and a tridiagonal one, so a few times ``n`` stored numbers per axis, ``O(D \cdot
# n)`` in all. The assembled matrix has a nonzero in every row for each neighbour along each
# axis, so its nonzeros grow like ``n^D``. On the small meshes above:

nnz_factors(Ms, Ks) = sum(count.(!iszero, Ms)) + sum(count.(!iszero, Ks))
(factor_nonzeros = nnz_factors((Mx, My, Mz), (Kx, Ky, Kz)), matrix_nonzeros = count(!iszero, A₃))

@test nnz_factors((Mx, My, Mz), (Kx, Ky, Kz)) < count(!iszero, A₃) #src

# Already here the factors hold a fraction of the matrix's nonzeros, and the gap widens at
# once as the axes refine: doubling the points per axis doubles `factor_nonzeros` but
# multiplies `matrix_nonzeros` by ``2^D``. The page measures that gap in bytes on the
# `41^3` mesh below, as `bytes_kronecker` against `bytes_csc`.
#
# ### Fast diagonalisation
#
# For a system of the shape above, the Kronecker structure also gives a direct solve. On
# each axis solve the generalised eigenproblem ``K v = \lambda M v``. Both matrices are
# symmetric and ``M`` is positive definite, so the eigenvectors can be scaled to satisfy
# ``V^{\mathsf T} M V = I`` and ``V^{\mathsf T} K V = \Lambda``, with ``\Lambda`` the
# diagonal matrix of eigenvalues. The two-argument `eigen` returns exactly that:

Fx = eigen(Kx, Mx)
Fy = eigen(Ky, My)
(round.(Fx.values; digits = 6), norm(Fx.vectors' * Mx * Fx.vectors - I))

@test Fx.vectors' * Mx * Fx.vectors ≈ I #src
@test Mx * Fx.vectors * Diagonal(Fx.values) ≈ Kx * Fx.vectors #src

# The smallest eigenvalue is zero up to rounding, the constants again. Write ``V = V_y
# \otimes V_x``. By the product rule from above, ``V^{\mathsf T} (M_y \otimes M_x) V = I``,
# ``V^{\mathsf T} (K_y \otimes M_x) V = \Lambda_y \otimes I`` and ``V^{\mathsf T} (M_y
# \otimes K_x) V = I \otimes \Lambda_x``, so
#
# ```math
# V^{\mathsf T} A_2 V = I + \Lambda_y \otimes I + I \otimes \Lambda_x,
# ```
#
# which is diagonal, with entries ``1 + \lambda^y_j + \lambda^x_i``. Hence
# ``A_2^{-1} = V (I + \Lambda_y \otimes I + I \otimes \Lambda_x)^{-1} V^{\mathsf T}``, and
# solving ``A_2 x = b`` is three steps: transform ``b`` by ``V^{\mathsf T}``, divide pointwise
# by the diagonal, transform back by ``V``. With ``B`` the vector ``b`` reshaped to
# ``n_x \times n_y``, applying ``V_y \otimes V_x`` is two small products as in the vec
# identity:

b₂ = collect(1.0:12.0)
B₂ = reshape(b₂, 4, 3)
Λ₂ = 1 .+ Fx.values .+ Fy.values'
X₂ = Fx.vectors * ((Fx.vectors' * B₂ * Fy.vectors) ./ Λ₂) * Fy.vectors'
norm(vec(X₂) - Matrix(A₂) \ b₂)

@test vec(X₂) ≈ Matrix(A₂) \ b₂ #src

# This agrees with the dense solve to rounding, though only the two small eigenproblems and
# two small products were used, and no ``12 \times 12`` matrix was factorised. On a real
# grid the saving is the same in kind: `fdm_solve`, used further down, does exactly this
# with one eigendecomposition per axis and one reshaped product per axis, in any dimension,
# and without ever assembling the matrix. The derivation needs the form to be a sum of
# Kronecker products of per-axis factors that share their eigenvectors. That holds for the
# mass and stiffness terms with constant coefficients, and fails for the grid-function
# coefficient shown under Separability. A Dirichlet condition is not part of this
# derivation either: it is handled by restricting each axis to its interior, as described
# in the section on the operator's real limit.

# ## Separability
#
# `is_separable` walks `a`'s resolved AST and answers whether every term is one of the two
# shapes a Kronecker product can represent: `innerₕ(u, v)` (the identity on every axis) or
# `inner₊` of a backward difference along one axis (what `∇ₕ(u)` expands into, one term per
# axis). `a` above is exactly that sum:

is_separable(a)

@test is_separable(a) #src

# A grid-function coefficient breaks the shape match, because it has no tensor structure to factor
# out, so `is_separable` refuses it rather than guessing:

fₕ = Rₕ(Wₕ, x -> 1.0 + x[1])
is_separable(form(Wₕ, Wₕ, (u, v) -> innerₕ(fₕ * u, v)))

@test !is_separable(form(Wₕ, Wₕ, (u, v) -> innerₕ(fₕ * u, v))) #src

# The same refusal covers a region restriction, an interpolation node, a surface (`InnerGamma`)
# weight, a composite space, a 1D mesh (nothing to factor), and any difference family other
# than the plain backward one `∇ₕ` builds: forward, centered, star, cross-weighted,
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
# `A` stores 472,361 nonzeros in arrays sized exactly to them, 8,109,312 bytes (Julia 1.13). #src
@test ndofs(Wₕ) == 68921                             #src
@test bytes_kronecker < 0.002 * bytes_csc             #src
@test bytes_csc > 7_500_000                           #src

# The operator holds three `41`-length one-dimensional factors per term instead of the
# assembled matrix's stored nonzeros, so it costs a fraction of a percent of `A` here, and
# the gap only widens with `n`, since `bytes_csc` grows like `n^3` while `bytes_kronecker`
# grows like `n`. Measured separately (not by this page, to keep this one fast): on a
# uniform `60x60x60` mesh with the same mass-plus-stiffness form, `test/form/kronecker.jl`
# (gpena/Bramble.jl#162) recorded `19,432` bytes for the operator against `61,948,960` bytes
# for the equivalent `SparseMatrixCSC`, about `0.03%`, for a 216,000-unknown problem this
# page does not build directly. That matrix figure predates assembly sizing the matrix's
# arrays exactly to their nonzeros, which on this page's `41x41x41` mesh took `A` from
# `18,625,640` to `8,109,312` bytes with the same 472,361 nonzeros.
#
# ## Solving it: iteratively, through the operator itself
#
# `K` subtypes `AbstractMatrix`, so `LinearSolve`'s `KrylovJL_CG` runs against it exactly as
# it would against `A`, applying `K` by `mul!` (one fused pass over the grid) rather than a
# sparse matrix-vector product:

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

# ## Error against the manufactured solution
#
# Both solves above only proved they agree with the ordinary sparse solve. They did not
# show that any of the three solved the problem posed at the top of the page. That needs the
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
# `K` and `kronecker_operator` carry no Dirichlet handling. A `KroneckerLinearOperator` is
# built for the whole grid, boundary rows included, because a boundary-restricted term has
# no tensor structure of its own to factor. The problem above sidesteps this by using an
# unconstrained (natural) boundary condition instead. Only `fdm_solve` reaches further, through
# the keyword `dirichlet = :boundary`. A point is interior in the domain iff it is
# interior along *every* axis, so restricting each axis's own factors to its interior
# (`2:end-1`) and running the same derivation on the restriction gives homogeneous Dirichlet
# on the whole boundary, still without ever assembling a matrix.

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
# There is no matching Dirichlet path for `K` itself, because `LinearSolve`'s `KrylovJL_CG` above
# only ever ran against the unconstrained problem. A reader reaching for the matrix-free
# operator on a Dirichlet problem meets this limit directly, not as a caveat in a docstring.
#
# ## Where to go next
#
#   - `kronecker_operator` is written as a plain code span throughout this page rather than
#     a link, along with `is_separable`, `KroneckerLinearOperator` and `fdm_solve`: none of
#     the four have an `@docs` entry on the [API reference](../api.md) yet.
#   - [Linear Poisson](poisson_linear.md) assembles the same discrete Laplacian into a
#     `SparseMatrixCSC` directly, with Dirichlet conditions from the start.
#   - The [matrix-free operator](matrix_free_operator.md) page applies a form that is not
#     separable, and compares it with this Kronecker route.
#   - The [forms tutorial](../tutorials/form.md) introduces `innerₕ`, `inner₊` and `∇ₕ` one
#     at a time.
