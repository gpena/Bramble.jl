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
# -\Delta u = g \text{ in } \Omega = (0,1)^2, \qquad
# u = u_{\text{exact}} \text{ on } \texttt{xmin} \text{ and } \texttt{ymin}, \qquad
# u + \partial_n u = h \text{ on } \texttt{xmax} \text{ and } \texttt{ymax},
# ```
#
# with ``\partial_n`` the outward normal derivative, the manufactured solution
# ``u_{\text{exact}}(x, y) = e^{x+y}``, so ``g = -2 u_{\text{exact}}``, and the Robin data
# ``h = 2 u_{\text{exact}}``, since the outward normal derivative of ``e^{x+y}`` on `xmax`
# and `ymax` is ``u_{\text{exact}}`` itself. This is deliberately *not* a trigonometric
# solution such as ``\sin(\pi x)\sin(\pi y)``: on a uniform grid that happens to be (very
# nearly) a single eigenmode of the discrete Laplacian, which both plain and preconditioned
# CG then solve in a handful of iterations regardless of mesh size. A measurement made once
# already did exactly that: it looked like a working comparison and said nothing about the
# preconditioner at all. ``e^{x+y}`` excites the discrete spectrum broadly, so the iteration
# counts below reflect the operator's actual conditioning.
#
# The grid is graded, not uniform: the points on each axis are ``x_i = s_i^{3/2}`` for
# uniform ``s_i`` in ``[0, 1]``, so the cells shrink towards `xmin` and `ymin`. A grid of
# random points would not do here: the last section shows AMG's iteration count growing
# with the mesh when a random grid is refined, which is the grid and not the preconditioner.
#
# ## Why this needs a preconditioner at all
#
# The condition number of the assembled Laplacian scales as ``\mathcal{O}(h_{\min}^{-2})``,
# so an unpreconditioned Krylov method needs ``\mathcal{O}(h_{\min}^{-1})`` iterations, at
# least doubling every time the mesh is refined by a factor of two. Algebraic multigrid builds
# a coarse-grid hierarchy directly from the graph of the assembled matrix and gives a Krylov
# method grid-independent, ``\mathcal{O}(1)`` iterations instead. [`amg_preconditioner`](@ref)
# wraps [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl)'s
# two constructions, `:smoothed_aggregation` (the default) and `:ruge_stuben`.
#
# ## Assembling a symmetric system
#
# AMG and CG are built for symmetric positive-definite matrices. The Robin sides go through
# ghost points: `gridspace(Ωₕ; ghosts = (:xmax, :ymax))` adds one layer outside those two
# sides only, and `constrained` fixes each ghost value from the relation `u + ∂ₙ(u)` and its
# data ``h``, so the physical nodes are the only unknowns. The bilinear form is the discrete
# Laplacian ``\langle -\Delta_h u, v \rangle_h``, which reads those ghost values. Eliminating
# the ghosts keeps the matrix symmetric, and the Robin coefficient, positive, keeps it
# positive definite.
#
# The Dirichlet sides need no ghosts. The `dirichlet` keyword of `assemble` alone does
# **not** keep the matrix symmetric: a Dirichlet row becomes the identity, but the matching
# *column* is left alone. Passing `symmetrize = true` restores the symmetry by folding the
# removed columns into the right-hand side. The relation `u => g` in `constrained` is the
# other symmetric way, which removes the Dirichlet nodes from the unknowns altogether.
#
# One step remains. The two triangles of the assembled matrix agree only to round-off, a
# unit in the last place where the grid spacings divide differently, and `issymmetric`
# compares bit for bit. Averaging `A` with its transpose removes that last unit, and
# changes nothing else.

using Bramble
using LinearAlgebra: isposdef, issymmetric, norm
using LinearSolve
using AlgebraicMultigrid
using Random

uex(x) = exp(x[1] + x[2])
rhs(x) = -2 * uex(x)
robin_data(x) = 2 * uex(x)

graded(n) = range(0.0, 1.0; length = n) .^ 1.5

Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))

function poisson_system(n)
    Ωₕ = mesh(Ω, (n, n), (true, true))
    Bramble.change_points!(Ωₕ, (graded(n), graded(n)))
    return Ωₕ, system_on(Ωₕ)...
end

function system_on(Ωₕ)
    Cₕ = constrained(gridspace(Ωₕ; ghosts = (:xmax, :ymax)),
        u -> (:xmax => u + ∂ₙ(u) => robin_data, :ymax => u + ∂ₙ(u) => robin_data))
    a = form(Cₕ, Cₕ, (u, v) -> innerₕ(-1.0 * Δₕ(u), v))
    l = form(Cₕ, v -> innerₕ(rhs, v))

    A₀, F = assemble(a, l; dirichlet = (:xmin => uex, :ymin => uex), symmetrize = true)
    return A₀, F
end

Ωₕ, A₀, F = poisson_system(65)
A = (A₀ + A₀') / 2
issymmetric(A) && isposdef(A)

#-

# Averaging is only honest if the triangles already agreed: the raw matrix is symmetric to   #src
# round-off (5e-15 measured, entries near 10), so an elimination that broke it would fail.   #src
@test norm(A₀ - A₀', Inf) <= 1.0e-12 * norm(A₀, Inf)                                         #src
@test issymmetric(A) && isposdef(A)                                                          #src

# ## Three ways to solve it
#
# **Direct**: `A \ F`, a sparse `LU`/`Cholesky` factorization. It is exact up to round-off, and
# perfectly fine at this size.

u_direct = A \ F

# **Plain CG**: no preconditioner.

sol_cg = solve(LinearProblem(A, F), KrylovJL_CG(); reltol = 1e-8, abstol = 1e-10)
sol_cg.iters

# **AMG-preconditioned CG**: build the hierarchy once from the assembled matrix with
# [`amg_preconditioner`](@ref), wrap it with `AlgebraicMultigrid.aspreconditioner`, and pass
# it as `Pl`.

P = aspreconditioner(amg_preconditioner(A))
sol_amg = solve(LinearProblem(A, F), KrylovJL_CG(); Pl = P, reltol = 1e-8, abstol = 1e-10)
sol_amg.iters

#-

# All three agree with each other.                                                           #src
@test u_direct≈sol_cg.u atol=1e-6 rtol=1e-6                                                  #src
@test u_direct≈sol_amg.u atol=1e-6 rtol=1e-6                                                 #src

# All three solve for the same discrete solution, and its error against the exact one is the
# scheme's, not the solver's:

Wₕ = gridspace(Ωₕ)
uₕ = element(Wₕ)
uₕ .= sol_amg.u
norm₁ₕ(uₕ .- Rₕ(Wₕ, uex))

#-

# The discretisation error of the graded grid at n = 65, 2.2e-4 measured; bracketed away from #src
# zero, since an exact match would mean the solution was reproduced by construction.          #src
@test 1.0e-4 < norm₁ₕ(uₕ .- Rₕ(Wₕ, uex)) < 1.0e-3                                            #src
# Only a preconditioned solve gets there in 20 iterations: plain CG needs 325 at n = 65.      #src
@test solve(LinearProblem(A, F), KrylovJL_CG(); Pl = P, reltol = 1e-8, abstol = 1e-10,        #src
    maxiters = 20).u≈u_direct atol=1e-6 rtol=1e-6                                            #src
@test sol_cg.iters > 8 * sol_amg.iters                                                       #src

# ## Iteration counts under refinement
#
# The point of the comparison: plain CG's iteration count against AMG's, on the same
# manufactured problem, as the graded mesh is refined.

iters(n) = iters(poisson_system(n)[2:3]...)

function iters(A_n₀, F_n)
    A_n = (A_n₀ + A_n₀') / 2
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

# `plain CG` more than doubles at each refinement: the smallest cell, at the corner where
# the grading is finest, shrinks faster than ``h``, and the condition number follows it. `AMG-CG`
# stays within a handful of iterations across a 64-fold increase in degrees of freedom.

#-

# Growth, not a single snapshot: the finest mesh needs several times the coarsest mesh's      #src
# plain-CG count (60 to 755 measured), while AMG's count barely moves (6 to 10).              #src
@test plain_iters[end] > 4 * plain_iters[1]                                                  #src
@test maximum(amg_iters) - minimum(amg_iters) <= 6                                           #src
@test all(<=(15), amg_iters)                                                                 #src
@test plain_iters[end] > 8 * amg_iters[end]                                                  #src

# ## Why not a random grid
#
# The same comparison on a grid of random points, seeded so the page prints the same numbers
# every build: a ``9 \times 9`` random grid refined in place three times by
# [`iterative_refinement!`](@ref), up to ``65 \times 65`` points.

random_amg_iters = let
    Random.seed!(20260903)
    Ωᵣ = mesh(Ω, (9, 9), (false, false))
    out = Int[]
    for level in 1:4
        push!(out, last(iters(system_on(Ωᵣ)...)))
        level < 4 && iterative_refinement!(Ωᵣ)
    end
    out
end

# AMG's count now climbs with every refinement instead of staying flat, so on this grid the
# comparison above would measure the grid rather than the preconditioner.

#-

# Measured 11, 16, 21, 27: the growth the graded grid avoids. Fails if a random grid stops     #src
# hurting AMG, in which case the page should move to one.                                      #src
@test random_amg_iters[end] > 2 * random_amg_iters[1]                                         #src
@test issorted(random_amg_iters)                                                              #src

# ## Where to go next
#
#   - [Choosing a solver](@ref tutorial_solvers) compares the direct and Krylov solvers
#     and the preconditioners this page uses, and [Solvers by problem](@ref
#     tutorial_solvers_by_problem) says which to reach for on which kind of problem.
#   - [Linear Poisson](poisson_linear.md) solves the same Dirichlet and Robin problem with
#     the default direct solve, and checks its order on random grids.
#   - The [matrix-free operator](matrix_free_operator.md) page applies a form without
#     assembling the matrix, which is the other way to save memory on a large system.
