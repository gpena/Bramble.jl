module TestAppleAccelerateExt

using Test
using Bramble
using LinearAlgebra
using SparseArrays

if Sys.isapple()
    using AppleAccelerate
end

# The direct-solver contract this backend shares with SuiteSparse, MUMPS and Sparspak
# lives in test/ext/SolverContracts.jl, which test/runtests.jl includes just before this
# file. The contract calls stay inside the `Sys.isapple()` branch below: building the
# closures is safe anywhere, since `accelerate_factorize` exists unconditionally with a
# throwing fallback, but calling them is not.
using ..ExtSolverContracts: ZERO_BC, poisson_system, convection_diffusion_system,
                            poisson_solve_contract, refactor_contract,
                            unsymmetric_refactor_contract, validation_contract

@testset "AppleAccelerate extension" begin
    if !Sys.isapple()
        @testset "Non-macOS guard" begin
            A = spdiagm(0 => [2.0, 2.0], 1 => [-1.0], -1 => [-1.0])
            F = [1.0, 1.0]
            @test_throws ArgumentError accelerate_factorize(A)
            @test_throws ArgumentError accelerate_solve(A, F)
            @test_throws ArgumentError pde_solve(A, F; solver = :accelerate)
        end
    else
        @testset "1D/2D/3D Poisson (SPD, sym = :spd)" begin
            poisson_solve_contract(;
                atol = 1.0e-12,
                solver = :accelerate,
                facttype = AccelerateFactorization,
                unified_kwargs = (; sym = :spd),
                solve = (A, F) -> accelerate_solve(A, F; sym = :spd),
                solve_form = (a, l) -> accelerate_solve(
                    a, l; dirichlet = ZERO_BC, symmetrize = true, sym = :spd
                ),
                factorize_form = a -> accelerate_factorize(a; dirichlet = ZERO_BC)
            )
        end

        @testset "Symmetric LDLᵀ and QR" begin
            p = poisson_system(Val(2), 8; source = x -> 1.0)

            u_ldlt = accelerate_solve(p.A, p.F; kind = :ldlt)
            @test isapprox(u_ldlt, p.u_ref; atol = 1.0e-12)

            u_qr = accelerate_solve(p.A, p.F; kind = :qr)
            @test isapprox(u_qr, p.u_ref; atol = 1.0e-12)
        end

        @testset "Unsymmetric convection-diffusion (sym = :unsymmetric)" begin
            cd = convection_diffusion_system(10)
            @test !issymmetric(cd.A)
            @test isapprox(
                accelerate_solve(cd.A, cd.F; sym = :unsymmetric), cd.u_ref; atol = 1.0e-12
            )

            # Auto-detection correctly selects unsymmetric LUTPP
            u_auto = pde_solve(cd.A, cd.F; solver = :accelerate, sym = :auto)
            @test isapprox(u_auto, cd.u_ref; atol = 1.0e-12)
        end

        @testset "Factorization reuse and refactoring (accelerate_refactor!)" begin
            p = poisson_system(Val(2), 8; source = x -> 1.0)

            refactor_contract(
                p;
                atol = 1.0e-12,
                solver = :accelerate,
                facttype = AccelerateFactorization,
                unified_kwargs = (; sym = :spd),
                factorize = A -> accelerate_factorize(A; sym = :spd),
                backend_refactor! = accelerate_refactor!
            )

            # The BilinearForm refactoring routes assemble without symmetrising, so they
            # need the factorization that can legally take that matrix.
            unsymmetric_refactor_contract(
                p; atol = 1.0e-12,
                factorize = A -> accelerate_factorize(A; sym = :unsymmetric)
            )
        end

        @testset "Error handling & validation" begin
            validation_contract(
                poisson_system(Val(1), 5; source = x -> 1.0);
                factorize = accelerate_factorize,
                backend_refactor! = accelerate_refactor!,
                invalid_sym_solve = (A, F) -> accelerate_solve(A, F; sym = :invalid_sym)
            )
        end
    end
end

end # module
