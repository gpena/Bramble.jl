module ExtILUZeroExtTests

using Test
using Bramble
using ILUZero: ILUZero, ILU0Precon
using SciMLBase: LinearProblem, solve
using LinearSolve: KrylovJL_GMRES
using LinearAlgebra: ldiv!, norm, issymmetric

# BrambleILUZeroExt: `ilu_preconditioner` builds the ILU(0) factorization
# (solvers/ilu_preconditioner.jl explains the underscored-fallback idiom), and
# `BrambleSciMLExt`'s `preconditioner = :ilu0` reaches the same object through `solve`. What
# is checked here: the factorization from a plain matrix and from a `BilinearForm` agree, the
# result has a working `ldiv!`, `preconditioner = :ilu0` in `solve` reaches the right answer,
# and -- the point of #244/#255 -- GMRES needs far fewer iterations on a convection-dominated
# system with ILU(0) than without, unlike AMG, which does not converge there at all.
#
# `Bramble.domain`/`Bramble.mesh`/`Bramble.element` are qualified throughout for the reason
# meshes_ext.jl gives: every ext file is included into the same `Main`, and other ext test
# files already bind unqualified `domain`/`mesh`/`element` to Meshes.jl's versions.

# Diffusion `1e-2` against unit advection in both directions -- the convection-dominated
# regime gpena/Bramble.jl#244 measured (90x90 grid, diffusion 1e-2, unit advection): far from
# an M-matrix, exactly where classical AMG stops converging and ILU(0) does not. The
# convective term is `inner₊(Mₕ(u), ∇ₕ(v))`, the same SBP-staggered discretization
# `docs/src/examples/convection_diffusion_linear.jl` uses, not a bare forward difference: an
# unstable (downwind) discretization of the advection term leaves the assembled matrix so far
# from diagonally dominant that ILU(0) itself breaks down (a one-shot application inflated a
# unit-norm residual by 3e5x, measured while building this test), which is a discretization
# bug, not a fact about ILU(0) -- `#244`'s own measurement used a stable discretization.
function _convection_diffusion_2d(n; eps = 1.0e-2)
    Ωd = Bramble.domain(Bramble.interval(0.0, 1.0) × Bramble.interval(0.0, 1.0))
    Ωₕ = Bramble.mesh(Ωd, (n, n), (true, true))
    Wₕ = gridspace(Ωₕ)
    a = form(Wₕ, Wₕ, (u, v) -> eps * inner₊(∇ₕ(u), ∇ₕ(v)) + inner₊(Mₕ(u), ∇ₕ(v)))
    fₕ = Bramble.element(Wₕ)
    # `exp(x + y)` as the source density: broadband spectral content, the same reasoning
    # test/ext/algebraicmultigrid_ext.jl gives for its own right-hand side -- not a
    # manufactured solution (nothing here is compared against one), only smooth data.
    avgₕ!(fₕ, x -> exp(x[1] + x[2]))
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    bcs = dirichlet_constraints(Ωd, :boundary => (x -> exp(x[1] + x[2])))
    return a, l, bcs
end

# GMRES iteration count for one convection-dominated system, with and without ILU(0), solved
# with the low-level LinearProblem/Pl path so `sol.iters` is visible (the high-level `solve(a,
# l; ...)` convenience unwraps straight to a VectorElement, further down).
function _gmres_iters(a, l, bcs; preconditioned::Bool)
    A, F = assemble(a, l; dirichlet = bcs, symmetrize = false)
    prob = LinearProblem(A, F)
    sol = if preconditioned
        P = ilu_preconditioner(A)
        solve(prob, KrylovJL_GMRES(); Pl = P, reltol = 1e-10, abstol = 1e-10)
    else
        solve(prob, KrylovJL_GMRES(); reltol = 1e-10, abstol = 1e-10)
    end
    return sol.iters
end

@testset "BrambleILUZeroExt" begin
    a, l, bcs = _convection_diffusion_2d(40)
    A, F = assemble(a, l; dirichlet = bcs, symmetrize = false)
    @test !issymmetric(A)

    @testset "ilu_preconditioner: matrix and BilinearForm methods agree" begin
        P_matrix = ilu_preconditioner(A)
        @test P_matrix isa ILU0Precon

        # `assemble(a; dirichlet = ...)` alone does not symmetrize (solvers/amg_preconditioner.jl
        # documents why `BilinearForm` methods here match that default) -- for an already
        # unsymmetric convection-dominated form the two matrices coincide exactly.
        P_form = ilu_preconditioner(a; dirichlet = :boundary)
        A_unsym = assemble(a; dirichlet = :boundary)
        @test P_form.l_nzval == ilu_preconditioner(A_unsym).l_nzval
    end

    @testset "ilu_preconditioner(A) is a valid preconditioner with ldiv!" begin
        P = ilu_preconditioner(A)
        n = size(A, 1)

        x = fill(1.0, n)
        y = similar(x)
        ldiv!(y, P, x)
        @test all(isfinite, y)
        @test length(y) == n
        @test norm(y) > 0

        # In-place single-argument ldiv! form.
        b = copy(x)
        ldiv!(P, b)
        @test b ≈ y

        # Regression: a `VectorElement` destination used to hit a method ambiguity between
        # `ILUZero`'s own `ldiv!(::AbstractVector, ::ILU0Precon, ::AbstractVector)` and
        # Bramble's `ldiv!(::VectorElement, ::Factorization, ::AbstractVector)`, since
        # `ILU0Precon <: Factorization`.
        uₕ_pc = element(Bramble.trial_space(a))
        @test ldiv!(uₕ_pc, P, x) === uₕ_pc
        @test parent(uₕ_pc) ≈ y
    end

    @testset "preconditioner = :ilu0 in solve reaches the direct answer" begin
        expected = A \ F

        uₕ = solve(a, l; dirichlet = bcs, solver = KrylovJL_GMRES(), preconditioner = :ilu0)
        @test uₕ isa Bramble.VectorElement
        @test parent(uₕ)≈expected atol=1e-6 rtol=1e-6

        # An already-built operator is accepted as-is, not only the `:ilu0` symbol.
        P = ilu_preconditioner(A)
        uₕ_P = solve(a, l; dirichlet = bcs, solver = KrylovJL_GMRES(), preconditioner = P)
        @test parent(uₕ_P)≈expected atol=1e-6 rtol=1e-6

        # `nothing` (default) leaves the solve unpreconditioned, still correct.
        uₕ_none = solve(a, l; dirichlet = bcs, solver = KrylovJL_GMRES())
        @test parent(uₕ_none)≈expected atol=1e-6 rtol=1e-6

        @test_throws ArgumentError solve(
            a, l; dirichlet = bcs, preconditioner = :not_ilu0
        )
    end

    @testset "GMRES needs far fewer iterations with ILU(0) on a convection-dominated system" begin
        # #255's acceptance criterion: fewer iterations than unpreconditioned GMRES -- an
        # iteration-count assertion, not a timing claim, per bramble-verification.
        a2, l2, bcs2 = _convection_diffusion_2d(60)
        ilu_iters = _gmres_iters(a2, l2, bcs2; preconditioned = true)
        plain_iters = _gmres_iters(a2, l2, bcs2; preconditioned = false)

        @test ilu_iters < plain_iters
        # #244 measured an ~11x reduction (179 -> 18 iterations) on a similar system; measured
        # here (60x60 grid): 14 vs 131, a ~9.4x reduction -- require a substantial margin, not
        # just "fewer by one", set safely below the measured ratio rather than at it.
        @test plain_iters > 5 * ilu_iters
    end
end

end # module
