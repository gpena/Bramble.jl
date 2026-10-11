# Normally test/runtests.jl includes test/ext/SolverContracts.jl immediately before this
# file (SolverContracts.jl's own header comment documents that ordering contract), so
# `Main.ExtSolverContracts` already exists by the time the `using ..ExtSolverContracts`
# below runs. Included here too, guarded, so `include("test/ext/suitesparse_ext.jl")`
# also works standalone without double-defining the module when
# runtests.jl already loaded it.
if !isdefined(Main, :ExtSolverContracts)
    include(joinpath(@__DIR__, "SolverContracts.jl"))
end

module TestSuiteSparseExt

using Test
using Bramble
using Bramble: SuiteSparseFactorization, suitesparse_factorize, suitesparse_qr_factorize,
               suitesparse_qr_solve, suitesparse_refactor!, suitesparse_solve
using Bramble: change_points!, sparse_factorize
using LinearAlgebra
using SparseArrays
using SuiteSparse

# The direct-solver contract this backend shares with MUMPS, Sparspak and AppleAccelerate
# lives in test/ext/SolverContracts.jl, which test/runtests.jl includes just before this
# file. Everything below the contract calls is SuiteSparse's own.
using ..ExtSolverContracts: ZERO_BC, poisson_system, convection_diffusion_system,
                            poisson_solve_contract, refactor_contract,
                            unsymmetric_refactor_contract, validation_contract

@testset "SuiteSparse extension" begin
    @testset "Poisson 1D/2D/3D (sym = :spd)" begin
        poisson_solve_contract(;
            atol = 1.0e-12,
            solver = :suitesparse,
            facttype = SuiteSparseFactorization,
            unified_kwargs = (; sym = :spd),
            solve = (A, F) -> suitesparse_solve(A, F; sym = :spd),
            solve_form = (a, l) -> suitesparse_solve(
                a, l; dirichlet = ZERO_BC, symmetrize = true, sym = :spd
            ),
            factorize_form = a -> suitesparse_factorize(a; dirichlet = ZERO_BC)
        )
    end

    @testset "convection-diffusion (:unsymmetric)" begin
        cd = convection_diffusion_system(10)
        @test !issymmetric(cd.A)
        @test isapprox(
            suitesparse_solve(cd.A, cd.F; sym = :unsymmetric), cd.u_ref; atol = 1.0e-12
        )

        # Auto-detection correctly selects unsymmetric LU
        u_auto = pde_solve(cd.A, cd.F; solver = :suitesparse, sym = :auto)
        @test isapprox(u_auto, cd.u_ref; atol = 1.0e-12)
    end

    # Exercises factorization reuse and suitesparse_refactor!.
    @testset "factorization reuse, refactor" begin
        p = poisson_system(Val(2), 8; source = x -> 1.0)

        # CHOLMOD, on the symmetrised matrix
        refactor_contract(
            p;
            atol = 1.0e-12,
            solver = :suitesparse,
            facttype = SuiteSparseFactorization,
            unified_kwargs = (; sym = :spd),
            factorize = A -> suitesparse_factorize(A; sym = :spd),
            backend_refactor! = suitesparse_refactor!
        )

        # UMFPACK: the BilinearForm refactoring routes assemble without symmetrising, so
        # they need the factorization that can legally take that matrix.
        unsymmetric_refactor_contract(
            p; atol = 1.0e-12, factorize = A -> suitesparse_factorize(A; sym = :unsymmetric)
        )
    end

    @testset "ordering/pivot parameters forwarded" begin
        p = poisson_system(Val(2), 8; source = x -> 1.0)
        A, F, a = p.A, p.F, p.a
        n = size(A, 1)

        # CHOLMOD's `perm`: a user-supplied fill-reducing permutation. A permutation of the
        # right length still solves correctly; a wrong-length one must error, which is only
        # true if the keyword actually reaches CHOLMOD instead of being silently dropped.
        fact_perm = suitesparse_factorize(A; sym = :spd, perm = collect(n:-1:1))
        @test isapprox(fact_perm \ F, p.u_ref; atol = 1.0e-12)
        @test_throws BoundsError suitesparse_factorize(A; sym = :spd, perm = [1])

        # UMFPACK's `q`: same check on the unsymmetric path.
        A_unsym, F_unsym = assemble(a, p.l; dirichlet = ZERO_BC, symmetrize = false)
        fact_q = suitesparse_factorize(A_unsym; sym = :unsymmetric, q = collect(n:-1:1))
        @test isapprox(fact_q \ F_unsym, A_unsym \ F_unsym; atol = 1.0e-12)
        @test_throws ArgumentError suitesparse_factorize(A_unsym; sym = :unsymmetric, q = [1])
    end

    # SPQR solves least-squares problems on rectangular systems.
    @testset "SPQR: least squares" begin
        p = poisson_system(Val(2), 8; source = x -> 1.0)
        A, F, u_ref = p.A, p.F, p.u_ref

        # Square system: same answer as the default sparse solve.
        u_qr = pde_solve(A, F; solver = :spqr)
        @test isapprox(u_qr, u_ref; atol = 1.0e-10)

        fact = suitesparse_qr_factorize(A)
        @test isapprox(fact \ F, u_ref; atol = 1.0e-10)

        u_direct = suitesparse_qr_solve(A, F)
        @test isapprox(u_direct, u_ref; atol = 1.0e-10)

        u_elem = suitesparse_qr_solve(p.a, p.l; dirichlet = ZERO_BC)
        @test u_elem isa Bramble.VectorElement
        @test isapprox(parent(u_elem), u_ref; atol = 1.0e-10)

        # Overdetermined least-squares: stack A on top of itself plus noise so the
        # normal-equations solution differs from an exact solve of the top block.
        A_ls = [A; A]
        F_ls = [F; F .+ 1.0e-3]
        u_ls = suitesparse_qr_solve(A_ls, F_ls)
        @test length(u_ls) == size(A, 2)
        # a genuine least-squares residual is orthogonal to A_ls's column space
        residual = A_ls * u_ls - F_ls
        @test isapprox(A_ls' * residual, zeros(size(A_ls, 2)); atol = 1.0e-8)
    end

    @testset "factorize(a): symmetrize refused" begin
        Ωₕ = mesh(
            domain(
                interval(0.0, 1.0) × interval(0.0, 1.0),
                :boundary => boundary_symbols(interval(0.0, 1.0) × interval(0.0, 1.0))
            ), (9, 7), (true, true)
        )
        change_points!(Ωₕ, (range(0.0, 1.0; length = 9) .^ 1.25, range(0.0, 1.0; length = 7) .^ 1.5))
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, x -> 1.0), v))
        bc = :boundary => x -> 1.0 + x[1]
        for symmetrize in (true, false)
            @test_throws ArgumentError suitesparse_factorize(a; dirichlet = bc, symmetrize = symmetrize)
            @test_throws ArgumentError sparse_factorize(
                a; solver = :suitesparse, dirichlet = bc, symmetrize = symmetrize
            )
        end

        # The plain call still solves, with the Dirichlet data lifted into F by assemble
        A, F = assemble(a, l; dirichlet = bc)
        @test suitesparse_factorize(a; dirichlet = bc) \ F ≈ Matrix(A) \ F
        @test sparse_factorize(a; solver = :suitesparse, dirichlet = bc) \ F ≈ Matrix(A) \ F
    end

    @testset "Error handling & validation" begin
        validation_contract(
            poisson_system(Val(1), 5; source = x -> 1.0);
            factorize = suitesparse_factorize,
            backend_refactor! = suitesparse_refactor!,
            invalid_sym_solve = (A, F) -> suitesparse_solve(A, F; sym = :invalid_sym)
        )
    end
end

end # module
