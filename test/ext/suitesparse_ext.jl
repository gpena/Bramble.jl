module TestSuiteSparseExt

using Test
using Bramble
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
    @testset "1D/2D/3D Poisson (SPD, sym = :spd)" begin
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

    @testset "Unsymmetric convection-diffusion (sym = :unsymmetric)" begin
        cd = convection_diffusion_system(10)
        @test !issymmetric(cd.A)
        @test isapprox(
            suitesparse_solve(cd.A, cd.F; sym = :unsymmetric), cd.u_ref; atol = 1.0e-12
        )

        # Auto-detection correctly selects unsymmetric LU
        u_auto = pde_solve(cd.A, cd.F; solver = :suitesparse, sym = :auto)
        @test isapprox(u_auto, cd.u_ref; atol = 1.0e-12)
    end

    @testset "Factorization reuse and refactoring (suitesparse_refactor!)" begin
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

    @testset "Ordering/pivot control parameters are forwarded" begin
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

    @testset "SPQR (least-squares / rectangular systems)" begin
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
