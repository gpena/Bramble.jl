module ExtAlgebraicMultigridExtTests

using Test
using Bramble
using AlgebraicMultigrid: AlgebraicMultigrid, aspreconditioner
using SciMLBase: LinearProblem, solve
using LinearSolve: KrylovJL_CG
using LinearAlgebra: norm, issymmetric

# BrambleAlgebraicMultigridExt: `amg_preconditioner` builds the AMG hierarchy
# (solvers/amg_preconditioner.jl explains the underscored-fallback idiom), and
# `BrambleSciMLExt`'s `preconditioner = :amg` reaches the same hierarchy, wrapped by
# `AlgebraicMultigrid.aspreconditioner`, through `solve`. What is checked here: the
# hierarchy from a plain matrix and from a `BilinearForm` agree, `aspreconditioner` of the
# result is a genuine (approximately linear) operator with `ldiv!`, an unknown `method`
# errors, `preconditioner = :amg` in `solve` reaches the right answer, and -- the point of
# #173 -- CG iteration counts stay essentially flat under mesh refinement for 2D and 3D
# Poisson, unlike the unpreconditioned O(h^-1) growth.
#
# `Bramble.domain`/`Bramble.mesh`/`Bramble.element` are qualified throughout for the reason
# meshes_ext.jl gives: every ext file is included into the same `Main`, and other ext test
# files already bind unqualified `domain`/`mesh`/`element` to Meshes.jl's versions.

# `exp(sum(x))`, not a trigonometric manufactured solution: `sin(pi x₁) sin(pi x₂) ...` is
# (very nearly) a single eigenmode of the discrete Laplacian on this uniform grid, so both an
# unpreconditioned and an AMG-preconditioned CG converge to it in a handful of iterations
# regardless of mesh size -- a real measurement that was made here first, and that would have
# passed a "CG stays fast" test whether or not AMG was doing anything at all. `exp(sum(x))`
# has broadband content across the discrete spectrum instead, so the O(h^-1) growth an
# unpreconditioned Krylov method shows on a genuinely elliptic right-hand side is real, and
# AMG's O(1) count against it is a real contrast, not an artifact of the test problem.
function _poisson_2d(n)
    Ωd = Bramble.domain(Bramble.interval(0.0, 1.0) × Bramble.interval(0.0, 1.0))
    Ωₕ = Bramble.mesh(Ωd, (n, n), (true, true))
    Wₕ = gridspace(Ωₕ)
    uex(x) = exp(x[1] + x[2])
    rhs(x) = -2 * uex(x)
    bcs = dirichlet_constraints(Ωd, :boundary => uex)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
    fₕ = Bramble.element(Wₕ)
    avgₕ!(fₕ, rhs)
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    return a, l, bcs
end

function _poisson_3d(n)
    Ωd = Bramble.domain(
        Bramble.interval(0.0, 1.0) × Bramble.interval(0.0, 1.0) × Bramble.interval(0.0, 1.0)
    )
    Ωₕ = Bramble.mesh(Ωd, (n, n, n), (true, true, true))
    Wₕ = gridspace(Ωₕ)
    uex(x) = exp(sum(x))
    rhs(x) = -3 * uex(x)
    bcs = dirichlet_constraints(Ωd, :boundary => uex)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
    fₕ = Bramble.element(Wₕ)
    avgₕ!(fₕ, rhs)
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    return a, l, bcs
end

# CG iteration count for one manufactured Poisson system, with and without AMG, solved with
# the low-level LinearProblem/Pl path so `sol.iters` is visible (the high-level `solve(a, l;
# ...)` convenience unwraps straight to a VectorElement, further down).
function _cg_iters(a, l, bcs; preconditioned::Bool)
    A, F = assemble(a, l; dirichlet = bcs, symmetrize = true)
    prob = LinearProblem(A, F)
    sol = if preconditioned
        P = aspreconditioner(amg_preconditioner(A))
        solve(prob, KrylovJL_CG(); Pl = P, reltol = 1e-8, abstol = 1e-10)
    else
        solve(prob, KrylovJL_CG(); reltol = 1e-8, abstol = 1e-10)
    end
    return sol.iters
end

@testset "BrambleAlgebraicMultigridExt" begin
    a, l, bcs = _poisson_2d(33)
    A, F = assemble(a, l; dirichlet = bcs, symmetrize = true)
    @test issymmetric(A)

    @testset "amg_preconditioner: matrix and BilinearForm methods agree" begin
        ml_matrix = amg_preconditioner(A)
        @test ml_matrix isa AlgebraicMultigrid.MultiLevel

        # `assemble(a; dirichlet = ...)` alone does not symmetrize (solvers/amg_preconditioner.jl
        # documents why), so the `BilinearForm` method's own matrix differs from the
        # symmetrized `A` above by the boundary columns only -- both are still valid AMG
        # input, and this checks the convenience path actually reaches the same routine.
        ml_form = amg_preconditioner(a; dirichlet = :boundary)
        A_unsym = assemble(a; dirichlet = :boundary)
        @test ml_form.levels[1].A == A_unsym

        # :ruge_stuben is the other constructor AlgebraicMultigrid offers.
        ml_rs = amg_preconditioner(A; method = :ruge_stuben)
        @test ml_rs isa AlgebraicMultigrid.MultiLevel

        @test_throws ArgumentError amg_preconditioner(A; method = :not_a_method)
    end

    @testset "aspreconditioner(amg_preconditioner(A)) is a valid linear operator with ldiv!" begin
        P = aspreconditioner(amg_preconditioner(A))
        n = size(A, 1)

        x = fill(1.0, n)
        y = similar(x)
        ldiv!(y, P, x)
        @test all(isfinite, y)
        @test length(y) == n

        # Linearity: P(αx₁ + βx₂) ≈ αP(x₁) + βP(x₂). A V-cycle built from linear smoothers
        # (the Gauss-Seidel default here) is itself linear, so this must hold to near
        # round-off, not merely approximately.
        x1 = collect(1.0:n)
        x2 = collect(n:-1.0:1)
        α, β = 2.0, -3.5
        y1, y2, y12 = similar(x1), similar(x1), similar(x1)
        ldiv!(y1, P, x1)
        ldiv!(y2, P, x2)
        ldiv!(y12, P, α * x1 .+ β * x2)
        @test y12≈α*y1 .+ β*y2 atol=1e-9 rtol=1e-9

        # A genuine preconditioner for this SPD system: applying it does not blow up the
        # residual direction relative to the identity, i.e. it is not degenerate.
        @test norm(y) > 0
    end

    @testset "preconditioner = :amg in solve reaches the direct answer" begin
        expected = A \ F

        uₕ = solve(a, l; dirichlet = bcs, symmetrize = true, solver = KrylovJL_CG(), preconditioner = :amg)
        @test uₕ isa Bramble.VectorElement
        @test parent(uₕ)≈expected atol=1e-6 rtol=1e-6

        # An already-built operator is accepted as-is, not only the `:amg` symbol.
        P = aspreconditioner(amg_preconditioner(A))
        uₕ_P = solve(a, l; dirichlet = bcs, symmetrize = true, solver = KrylovJL_CG(), preconditioner = P)
        @test parent(uₕ_P)≈expected atol=1e-6 rtol=1e-6

        # `nothing` (default) leaves the solve unpreconditioned, still correct.
        uₕ_none = solve(a, l; dirichlet = bcs, symmetrize = true, solver = KrylovJL_CG())
        @test parent(uₕ_none)≈expected atol=1e-6 rtol=1e-6

        @test_throws ArgumentError solve(
            a, l; dirichlet = bcs, symmetrize = true, preconditioner = :not_amg
        )
    end

    @testset "CG iteration count stays essentially flat under refinement (2D)" begin
        # #173's acceptance criterion: bounded iterations (issue asks <= 15) as h -> 0. Both
        # counts are measured on the same three meshes so the contrast is direct: this is not
        # "AMG is fast", it is "AMG stays flat where plain CG does not", on identical systems.
        meshes = [_poisson_2d(n) for n in (16, 32, 64)]
        amg_iters = [_cg_iters(m...; preconditioned = true) for m in meshes]
        plain_iters = [_cg_iters(m...; preconditioned = false) for m in meshes]

        @test all(<=(15), amg_iters)
        # "Essentially flat": the spread across a 16x growth in dof count is a handful of
        # iterations, not proportional to it.
        @test maximum(amg_iters) - minimum(amg_iters) <= 6

        # The O(h^-1) growth AMG replaces: unpreconditioned CG roughly doubles its iteration
        # count at each refinement here (measured: 45, 94, 188), so the finest mesh needs
        # well over 3x the coarsest mesh's count -- the opposite of AMG's flat row above.
        @test plain_iters[end] > 3 * plain_iters[1]
        @test plain_iters[end] > 10 * amg_iters[end]
    end

    @testset "CG iteration count stays essentially flat under refinement (3D)" begin
        meshes = [_poisson_3d(n) for n in (8, 16, 24)]
        amg_iters = [_cg_iters(m...; preconditioned = true) for m in meshes]
        plain_iters = [_cg_iters(m...; preconditioned = false) for m in meshes]

        @test all(<=(15), amg_iters)
        @test maximum(amg_iters) - minimum(amg_iters) <= 6

        # Measured: unpreconditioned 22, 50, 76 against AMG's 5, 6, 6.
        @test plain_iters[end] > 2.5 * plain_iters[1]
        @test plain_iters[end] > 8 * amg_iters[end]
    end
end

end # module
