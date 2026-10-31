# Normally test/runtests.jl includes test/ext/SolverContracts.jl immediately before this
# file (SolverContracts.jl's own header comment documents that ordering contract), so
# `Main.ExtSolverContracts` already exists by the time the `using ..ExtSolverContracts`
# below runs. Included here too, guarded, so `include("test/ext/sparspak_ext.jl")`
# also works standalone without double-defining the module when runtests.jl already
# loaded it.
if !isdefined(Main, :ExtSolverContracts)
    include(joinpath(@__DIR__, "SolverContracts.jl"))
end

module TestSparspakExt

using Test
using Bramble
using Bramble: SparspakFactorization, sparspak_factorize, sparspak_refactor!, sparspak_solve
using Bramble: sparse_factorize
using LinearAlgebra
using SparseArrays
using Sparspak
using DifferentiationInterface: DifferentiationInterface, AutoForwardDiff
using ForwardDiff: ForwardDiff

# The direct-solver contract this backend shares with SuiteSparse, MUMPS and
# AppleAccelerate lives in test/ext/SolverContracts.jl, which test/runtests.jl includes
# just before this file. Sparspak is the backend whose entry points take no keywords at
# all (src/solvers/sparspak_solver.jl:46,81), so its closures below pass none, and it is
# the one backend with no invalid symmetry option to reject.
using ..ExtSolverContracts: ZERO_BC, poisson_system, convection_diffusion_system,
                            poisson_solve_contract, refactor_contract,
                            unsymmetric_refactor_contract, validation_contract
using ..TestUtils: _fd, alloc_test

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

    # Exercises factorization reuse and sparspak_refactor!.
    @testset "factorization reuse, refactor" begin
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

    # Sparspak has no binary dependency, so non-Float64 element types work.
    @testset "non-Float64 element types" begin
        p = poisson_system(Val(2), 8; source = x -> 1.0)
        Wₕ, A, F = p.Wₕ, p.A, p.F

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
        # The load stays real: the extension promotes it to the matrix's element type
        # before Sparspak's triangular solve, which takes no other.
        function g(θ)
            aθ = form(Wₕ, Wₕ, (u, v) -> (1.0 + θ) * inner₊(∇ₕ(u), ∇ₕ(v)))
            Aθ = assemble(aθ; dirichlet = ZERO_BC)
            return sum(pde_solve(Aθ, F; solver = :sparspak))
        end
        @test eltype(F) == Float64
        d = DifferentiationInterface.derivative(g, AutoForwardDiff(), 1.0)
        @test isapprox(d, _fd(g, 1.0); atol = 1.0e-6, rtol = 1.0e-6)
    end

    # A real load against a `Dual` matrix: `pde_solve`, `sparspak_solve` and `\` on the
    # factorization all promote the load, and the derivative matches the dense solve's.
    @testset "Dual matrix, real load" begin
        b = [1.0, 1.0]
        Aθ(θ) = sparse([2.0+θ 0.5; 0.5 1.0])
        dense(θ) = sum(Matrix(Aθ(θ)) \ b)
        for solve in (θ -> pde_solve(Aθ(θ), b; solver = :sparspak),
                      θ -> sparspak_solve(Aθ(θ), b),
                      θ -> sparspak_factorize(Aθ(θ)) \ b)
            d = DifferentiationInterface.derivative(θ -> sum(solve(θ)), AutoForwardDiff(), 1.0)
            @test isapprox(d, _fd(dense, 1.0); rtol = 1.0e-6)
        end
    end

    # The promotion copies nothing when the load already has the matrix's element type: a
    # Float64 solve allocates what an in-place solve of a copy does, short of one more
    # load-sized vector (32 KiB here, far above any `-O0` boxing).
    @testset "Float64 solve allocates no extra copy" begin
        n = 4096
        A = sparse(SymTridiagonal(fill(4.0, n), fill(-1.0, n - 1)))
        b = ones(n)
        fact = sparspak_factorize(A)
        in_place = alloc_test((f, v) -> ldiv!(f, copy(v)), fact, b)
        @test alloc_test(\, fact, b) - in_place < sizeof(b)
    end

    # Sparspak factors BigFloat matrices, which SuiteSparse, MUMPS and Accelerate cannot
    # (docs/src/api_sciml.md). The solve must keep the element type and the residual must
    # hold at BigFloat precision, not merely at Float64.
    @testset "BigFloat element type" begin
        n = 8
        A = sparse(SymTridiagonal(fill(BigFloat(4), n), fill(BigFloat(1), n - 1)))
        b = ones(BigFloat, n)
        x = pde_solve(A, b; solver = :sparspak)
        @test eltype(x) == BigFloat
        @test maximum(abs, A * x .- b) < 1.0e-30
    end

    # Sparspak reports no singularity itself: a stored zero pivot or an empty row/column
    # leaves `Inf` in the solution, a rank-deficient matrix throws `ErrorException("An
    # empty problem. No matrix.")`. The extension turns both into `SingularException`,
    # as SuiteSparse, MUMPS and Accelerate throw it (#722).
    @testset "singular matrix throws (#722)" begin
        stored_zero = sparse([1, 2, 3, 4], [1, 2, 3, 4], [2.0, 2.0, 0.0, 1.0])
        rank_deficient = sparse([1.0 1 0; 1 1 0; 0 0 1])
        empty_row = sparse([1.0 0 0; 0 1 1; 0 0 0])
        for S in (stored_zero, rank_deficient, empty_row)
            n = size(S, 1)
            b = ones(n)
            @test_throws SingularException pde_solve(S, b; solver = :sparspak)
            @test_throws SingularException sparspak_solve(S, b)
            @test_throws SingularException sparse_factorize(S; solver = :sparspak) \ b
            @test_throws SingularException sparse_factorize(S; solver = :sparspak) \
                                           ComplexF64.(b)
            @test_throws SingularException factorize(S, SparspakFactorization) \ b
            @test_throws SingularException ldiv!(
                zeros(n), sparse_factorize(S; solver = :sparspak), b
            )
            @test_throws SingularException ldiv!(
                sparse_factorize(S; solver = :sparspak), copy(b)
            )
        end

        # refactor onto a singular matrix of the same pattern
        A = sparse([4.0 1 0; 1 4 1; 0 1 4])
        b = ones(3)
        singular = sparse([1.0 1 0; 1 1 0; 0 0 4])
        fact = sparspak_factorize(sparse([1.0 0.5 0; 0.5 1 0; 0 0 4]))
        @test_throws SingularException sparspak_refactor!(fact, singular)

        # A nonfinite right-hand side is propagated, not blamed on the matrix
        @test isnan(sum(pde_solve(A, [1.0, NaN, 1.0]; solver = :sparspak)))
        # nonsingular systems are unaffected
        @test pde_solve(A, b; solver = :sparspak) ≈ Matrix(A) \ b
    end

    @testset "unknown keyword throws (#657)" begin
        A = sparse(SymTridiagonal(fill(4.0, 6), fill(-1.0, 5)))
        b = collect(1.0:6)
        @test_throws ArgumentError pde_solve(A, b; solver = :sparspak, bogus = 1)
        @test_throws ArgumentError sparse_factorize(A; solver = :sparspak, bogus = 1)
        # `sym` is a named keyword and stays ignored here
        @test pde_solve(A, b; solver = :sparspak, sym = :spd) ≈ Matrix(A) \ b
        @test sparse_factorize(A; solver = :sparspak, sym = :spd) \ b ≈ Matrix(A) \ b
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
