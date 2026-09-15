# # Algebraic multigrid preconditioning
#
# A worked comparison of three ways to solve the same discrete Poisson system: a direct
# sparse factorization, plain (unpreconditioned) conjugate gradient, and conjugate gradient
# preconditioned by algebraic multigrid (AMG). Every number below was produced by the code
# shown.
#
# ## Problem
#
# ```math
# -\Delta u = g \text{ in } \Omega, \qquad u = u_{\text{exact}} \text{ on } \partial\Omega,
# \qquad \Omega = (0,1)^2
# ```
#
# with the manufactured solution ``u_{\text{exact}}(x, y) = e^{x+y}``, so ``g = -2
# u_{\text{exact}}``. This is deliberately *not* a trigonometric solution such as
# ``\sin(\pi x)\sin(\pi y)``: on a uniform grid that happens to be (very nearly) a single
# eigenmode of the discrete Laplacian, which both plain and preconditioned CG then solve in a
# handful of iterations regardless of mesh size -- a measurement made once already, that
# looked like a working comparison and said nothing about the preconditioner at all.
# ``e^{x+y}`` excites the discrete spectrum broadly, so the iteration counts below reflect
# the operator's actual conditioning.
#
# ## Why this needs a preconditioner at all
#
# The condition number of the assembled Laplacian scales as ``\mathcal{O}(h^{-2})``, so an
# unpreconditioned Krylov method needs ``\mathcal{O}(h^{-1})`` iterations -- doubling, very
# roughly, every time the mesh is refined by a factor of two. Algebraic multigrid builds a
# coarse-grid hierarchy directly from the graph of the assembled matrix and gives a Krylov
# method grid-independent, ``\mathcal{O}(1)`` iterations instead. [`amg_preconditioner`](@ref)
# wraps [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl)'s
# two constructions, `:smoothed_aggregation` (the default) and `:ruge_stuben`.
#
# ## Assembling a symmetric system
#
# AMG is built for symmetric positive-definite matrices. `assemble(a::BilinearForm;
# dirichlet = ...)` alone does **not** produce one: a Dirichlet row becomes the identity, but
# the matching *column* is left alone, so the assembled matrix is not exactly symmetric even
# though the underlying operator is. Passing `symmetrize = true` to the two-form
# [`assemble`](@ref) (or to [`linear_problem`](@ref)/`solve` below) restores that symmetry by
# folding the removed columns into the right-hand side -- do this before handing the matrix to
# AMG, not after.

using Bramble
using LinearSolve
using AlgebraicMultigrid

uex(x) = exp(x[1] + x[2])
rhs(x) = -2 * uex(x)

function poisson_system(n)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    Ωₕ = mesh(Ω, (n, n), (true, true))
    Wₕ = gridspace(Ωₕ)
    bcs = dirichlet_constraints(Ω, :boundary => uex)

    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
    gₕ = element(Wₕ)
    avgₕ!(gₕ, rhs)
    l = form(Wₕ, v -> innerₕ(gₕ, v))

    A, F = assemble(a, l; dirichlet = bcs, symmetrize = true)
    return a, l, bcs, A, F
end

a, l, bcs, A, F = poisson_system(65)
issymmetric(A)

#-

@test issymmetric(A) && isposdef(A)                                                        #src

# ## Three ways to solve it
#
# **Direct**: `A \ F`, a sparse `LU`/`Cholesky` factorization -- exact up to round-off, and
# perfectly fine at this size.

u_direct = A \ F

# **Plain CG**: no preconditioner.

sol_cg = solve(LinearProblem(A, F), KrylovJL_CG(); reltol = 1e-8, abstol = 1e-10)
sol_cg.iters

# **AMG-preconditioned CG**: build the hierarchy once with [`amg_preconditioner`](@ref), wrap
# it with `AlgebraicMultigrid.aspreconditioner`, and pass it as `Pl`.

P = aspreconditioner(amg_preconditioner(A))
sol_amg = solve(LinearProblem(A, F), KrylovJL_CG(); Pl = P, reltol = 1e-8, abstol = 1e-10)
sol_amg.iters

#-

# All three agree with each other and with the manufactured solution, restricted to the         #src
# interior where the SBP scheme is second order (the boundary rows are exact by             #src
# construction).                                                                             #src
@test u_direct≈sol_cg.u atol=1e-6 rtol=1e-6                                               #src
@test u_direct≈sol_amg.u atol=1e-6 rtol=1e-6                                              #src

# The same three lines collapse into one call through [`linear_problem`](@ref)'s companion
# `solve(a::BilinearForm, l::LinearForm; ...)`, which assembles, solves and unwraps the result
# to a [`VectorElement`](@ref) directly -- `preconditioner = :amg` reaches
# [`amg_preconditioner`](@ref) internally, so there is nothing to build by hand:

uₕ = solve(a, l; dirichlet = bcs, symmetrize = true, solver = KrylovJL_CG(), preconditioner = :amg)
normₕ(uₕ .- Rₕ(space(uₕ), uex))

#-

@test normₕ(uₕ .- Rₕ(space(uₕ), uex)) < 1.0e-2                                              #src

# ## Iteration counts under refinement
#
# The point of the comparison: plain CG's iteration count against AMG's, on the same
# manufactured problem, as the mesh is refined.

function iters(n)
    _, _, _, A_n, F_n = poisson_system(n)
    prob = LinearProblem(A_n, F_n)

    plain = solve(prob, KrylovJL_CG(); reltol = 1e-8, abstol = 1e-10).iters

    P_n = aspreconditioner(amg_preconditioner(A_n))
    amg = solve(prob, KrylovJL_CG(); Pl = P_n, reltol = 1e-8, abstol = 1e-10).iters

    return plain, amg
end

ns = (16, 32, 64, 128)
results = iters.(ns)
plain_iters = first.(results)
amg_iters = last.(results)

for (n, p, m) in zip(ns, plain_iters, amg_iters)
    println("n = $(lpad(n, 3))   plain CG = $(lpad(p, 4))   AMG-CG = $(lpad(m, 3))")
end

# `plain CG` roughly doubles at each refinement -- the `O(h^-1)` growth the condition number
# predicts -- while `AMG-CG` stays within a handful of iterations across a 64-fold increase in
# degrees of freedom.

#-

# Growth, not a single snapshot: the finest mesh needs several times the coarsest mesh's      #src
# plain-CG count, while AMG's count barely moves.                                             #src
@test plain_iters[end] > 4 * plain_iters[1]                                                  #src
@test maximum(amg_iters) - minimum(amg_iters) <= 6                                           #src
@test all(<=(15), amg_iters)                                                                 #src
@test plain_iters[end] > 8 * amg_iters[end]                                                  #src
