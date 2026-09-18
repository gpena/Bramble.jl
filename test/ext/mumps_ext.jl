module TestMUMPSExt

using Test
using Bramble
using LinearAlgebra
using SparseArrays
using MUMPS

# The direct-solver contract this backend shares with SuiteSparse, Sparspak and
# AppleAccelerate lives in test/ext/SolverContracts.jl, which test/runtests.jl includes
# just before this file. Everything below the contract calls is MUMPS's own: the symmetric
# indefinite path no other backend here has, factorization reuse across right-hand sides,
# and the icntl/cntl control options.
using ..ExtSolverContracts: ZERO_BC, poisson_system, convection_diffusion_system,
                            poisson_solve_contract, refactor_contract,
                            unsymmetric_refactor_contract, validation_contract

@testset "MUMPS extension" begin
    @testset "1D/2D/3D Poisson (SPD, sym = :spd)" begin
        poisson_solve_contract(;
            atol = 1.0e-12,
            solver = :mumps,
            facttype = MUMPSFactorization,
            unified_kwargs = (; sym = :spd),
            solve = (A, F) -> mumps_solve(A, F; sym = :spd),
            solve_form = (a, l) -> mumps_solve(
                a, l; dirichlet = ZERO_BC, symmetrize = true, sym = :spd
            ),
            # src/solvers/mumps_solver.jl:57 has had this BilinearForm method all along;
            # this file simply never called it.
            factorize_form = a -> mumps_factorize(a; dirichlet = ZERO_BC)
        )
    end

    @testset "Symmetric indefinite (Helmholtz / saddle point, sym = :symmetric)" begin
        I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ω2 = mesh(domain(I2, :boundary => boundary_symbols(I2)), (10, 10), (true, true))
        W2 = gridspace(Ω2)
        # Shifted Helmholtz: -Δu - k²u = f (indefinite symmetric matrix)
        k2 = 50.0
        a_helm = form(W2, W2, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) - k2 * innerₕ(u, v))
        f_helm = Rₕ(W2, x -> 1.0)
        l_helm = form(W2, v -> innerₕ(f_helm, v))
        A_h, F_h = assemble(a_helm, l_helm; dirichlet = ZERO_BC, symmetrize = true)

        u_ss = A_h \ F_h
        fact_h = mumps_factorize(A_h; sym = :symmetric)
        u_mumps = fact_h \ F_h
        @test isapprox(u_mumps, u_ss; atol = 1.0e-12)
    end

    @testset "Unsymmetric (Convection-diffusion, sym = :unsymmetric)" begin
        cd = convection_diffusion_system(12; βx = 5.0, βy = 2.0)
        @test !issymmetric(cd.A)
        @test isapprox(
            pde_solve(cd.A, cd.F; solver = :mumps, sym = :unsymmetric), cd.u_ref;
            atol = 1.0e-12
        )
    end

    @testset "Factorization reuse for time stepping" begin
        p = poisson_system(Val(2), 8; source = x -> 1.0)
        A, F = p.A, p.F

        fact = factorize(A, MUMPSFactorization; sym = :spd)
        @test fact isa MUMPSFactorization

        u1 = similar(F)
        ldiv!(u1, fact, F)
        @test isapprox(u1, p.u_ref; atol = 1.0e-12)

        # Repeated solves with different RHS vectors
        F2 = rand(length(F))
        u2 = similar(F2)
        ldiv!(u2, fact, F2)
        @test isapprox(u2, A \ F2; atol = 1.0e-12)

        # In-place mutating vector solve
        F3 = copy(F2)
        ldiv!(fact, F3)
        @test isapprox(F3, u2; atol = 1.0e-12)
    end

    @testset "Factorization reuse and refactoring (mumps_refactor!)" begin
        p = poisson_system(Val(1), 15; source = x -> 1.0)

        refactor_contract(
            p;
            atol = 1.0e-12,
            solver = :mumps,
            facttype = MUMPSFactorization,
            unified_kwargs = (; sym = :spd),
            factorize = A -> mumps_factorize(A; sym = :spd),
            backend_refactor! = mumps_refactor!
        )

        # The BilinearForm refactoring routes assemble without symmetrising, so they need
        # the factorization that can legally take that matrix.
        unsymmetric_refactor_contract(
            p; atol = 1.0e-12, factorize = A -> mumps_factorize(A; sym = :unsymmetric)
        )
    end

    @testset "Control options (icntl, cntl)" begin
        p = poisson_system(Val(1), 15; source = x -> 1.0)

        # Custom memory relaxation and METIS ordering (7 => 5)
        fact = mumps_factorize(p.A; sym = :spd, icntl = (7 => 5, 14 => 35))
        u = fact \ p.F
        @test isapprox(u, p.u_ref; atol = 1.0e-12)

        # pde_solve with MUMPSFactorization directly
        @test isapprox(pde_solve(fact, p.F), u; atol = 1.0e-12)

        # a numeric update reusing that symbolic METIS analysis
        A_modified = copy(p.A)
        A_modified[1, 1] += 5.0
        refactor!(fact, A_modified)
        @test isapprox(fact \ p.F, A_modified \ p.F; atol = 1.0e-12)
    end

    @testset "Error handling & validation" begin
        p = poisson_system(Val(1), 5; source = x -> 1.0)
        validation_contract(
            p;
            factorize = mumps_factorize,
            backend_refactor! = mumps_refactor!,
            invalid_sym_solve = (A, F) -> mumps_solve(A, F; sym = :invalid_sym)
        )

        # Not a MUMPS assertion: the dispatcher's unknown-symbol path, which happens to be
        # checked only here.
        @test_throws ArgumentError pde_solve(p.A, p.F; solver = :invalid_solver)
    end
end

end # module
