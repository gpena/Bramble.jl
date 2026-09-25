module TestSparspakExt

using Test
using Bramble
using Bramble: SparspakFactorization, sparspak_factorize, sparspak_refactor!, sparspak_solve
using LinearAlgebra
using SparseArrays
using Sparspak
using ForwardDiff

# The direct-solver contract this backend shares with SuiteSparse, MUMPS and
# AppleAccelerate lives in test/ext/SolverContracts.jl, which test/runtests.jl includes
# just before this file. Sparspak is the backend whose entry points take no keywords at
# all (src/solvers/sparspak_solver.jl:46,81), so its closures below pass none, and it is
# the one backend with no invalid symmetry option to reject.
using ..ExtSolverContracts: ZERO_BC, poisson_system, convection_diffusion_system,
                            poisson_solve_contract, refactor_contract,
                            unsymmetric_refactor_contract, validation_contract
using ..TestUtils: _fd

@testset "Sparspak extension" begin
    @testset "1D/2D/3D Poisson" begin
        poisson_solve_contract(;
            atol = 1.0e-10,
            solver = :sparspak,
            facttype = SparspakFactorization,
            solve = sparspak_solve,
            solve_form = (a, l) -> sparspak_solve(
                a, l; dirichlet = ZERO_BC, symmetrize = true
            ),
            factorize_form = a -> sparspak_factorize(a; dirichlet = ZERO_BC)
        )
    end

    @testset "Unsymmetric convection-diffusion" begin
        cd = convection_diffusion_system(10)
        @test !issymmetric(cd.A)
        @test isapprox(
            pde_solve(cd.A, cd.F; solver = :sparspak), cd.u_ref; atol = 1.0e-10
        )
    end

    @testset "Factorization reuse and refactoring (sparspak_refactor!)" begin
        p = poisson_system(Val(2), 8; source = x -> 1.0)

        refactor_contract(
            p;
            atol = 1.0e-10,
            solver = :sparspak,
            facttype = SparspakFactorization,
            factorize = sparspak_factorize,
            backend_refactor! = sparspak_refactor!
        )

        # Sparspak has no symmetry flag, so the same factorization takes the unsymmetrised
        # matrix the BilinearForm routes assemble.
        unsymmetric_refactor_contract(p; atol = 1.0e-10, factorize = sparspak_factorize)
    end

    @testset "Non-Float64 element types (no binary dependency)" begin
        p = poisson_system(Val(2), 8; source = x -> 1.0)
        Wₕ, l, A, F = p.Wₕ, p.l, p.A, p.F

        # Float32
        A32 = SparseMatrixCSC{Float32}(A)
        F32 = Float32.(F)
        u_ref32 = Float32.(p.u_ref)
        u_sp32 = sparspak_solve(A32, F32)
        @test isapprox(u_sp32, u_ref32; atol = 1.0f-3)

        # ForwardDiff.Dual: Sparspak's factorization is generic over the matrix element
        # type (`sparspaklu(m::SparseMatrixCSC{FT,IT}; ...) where {FT,IT}`), so a matrix
        # assembled from a `Dual`-valued coefficient factors and solves directly, which
        # SuiteSparse/MUMPS cannot do (both require `Float64`/`ComplexF64` matrix entries).
        # Sparspak's triangular solve additionally requires the right-hand side to share
        # the matrix's own element type exactly, hence the explicit `eltype(Aθ).(...)`.
        function g(θ)
            aθ = form(Wₕ, Wₕ, (u, v) -> (1.0 + θ) * inner₊(∇ₕ(u), ∇ₕ(v)))
            Aθ, Fθ0 = assemble(aθ, l; dirichlet = ZERO_BC)
            Fθ = eltype(Aθ).(Fθ0)
            uθ = sparspak_solve(Aθ, Fθ)
            return sum(uθ)
        end
        d = ForwardDiff.derivative(g, 1.0)
        @test isapprox(d, _fd(g, 1.0); atol = 1.0e-6, rtol = 1.0e-6)
    end

    @testset "Error handling & validation" begin
        validation_contract(
            poisson_system(Val(1), 5; source = x -> 1.0);
            factorize = sparspak_factorize,
            backend_refactor! = sparspak_refactor!
        )
    end
end

end # module
