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
using Bramble: change_points!, refactor!, sparse_factorize
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

        # UMFPACK's `control`, as `SparseArrays.UMFPACK.get_umfpack_control` returns it: an
        # edited vector must reach the factorization, which keeps its own copy of it.
        control = SparseArrays.UMFPACK.get_umfpack_control(Float64, Int)
        control[SparseArrays.UMFPACK.JL_UMFPACK_IRSTEP] = 3.0
        fact_ctl = suitesparse_factorize(A_unsym; sym = :unsymmetric, control = control)
        @test fact_ctl.fact.control == control
        @test isapprox(fact_ctl \ F_unsym, A_unsym \ F_unsym; atol = 1.0e-12)
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

    @testset "sym = :auto: indefinite and singular" begin
        # Shifted Helmholtz, -Δu - 50u = f: symmetric, positive diagonal, indefinite. A uniform
        # mesh keeps the diagonal positive (a random one gives a non-positive entry about once
        # in 200 runs), so `:auto` meets a Cholesky candidate that is not positive definite.
        S2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ω2 = mesh(domain(S2, :boundary => boundary_symbols(S2)), (10, 10), (true, true))
        W2 = gridspace(Ω2)
        a_helm = form(W2, W2, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) - 50.0 * innerₕ(u, v))
        l_helm = form(W2, v -> innerₕ(Rₕ(W2, x -> 1.0), v))
        A_h, F_h = assemble(a_helm, l_helm; dirichlet = ZERO_BC, symmetrize = true)
        @test issymmetric(A_h)
        @test all(>(0), diag(A_h))
        @test minimum(eigvals(Symmetric(Matrix(A_h)))) < 0
        u_ref = Matrix(A_h) \ F_h

        @test isapprox(suitesparse_factorize(A_h) \ F_h, u_ref; atol = 1.0e-10)
        @test isapprox(sparse_factorize(A_h; solver = :suitesparse) \ F_h, u_ref; atol = 1.0e-10)
        @test isapprox(pde_solve(A_h, F_h; solver = :suitesparse), u_ref; atol = 1.0e-10)
        # `symmetrize = true` no longer forces Cholesky
        u_form = suitesparse_solve(a_helm, l_helm; dirichlet = ZERO_BC, symmetrize = true)
        @test isapprox(parent(u_form), u_ref; atol = 1.0e-10)

        # Symmetric with a positive diagonal, a zero pivot in the LDLᵀ order: solvable
        M = sparse([1.0 1 0; 1 1 1; 0 1 1])
        b = [1.0, 2.0, 3.0]
        @test isapprox(suitesparse_factorize(M) \ b, Matrix(M) \ b; atol = 1.0e-12)

        # Singular: the fallback reports it rather than returning a wrong answer
        Sg = sparse([1.0 1; 1 1])
        @test_throws SingularException suitesparse_factorize(Sg) \ [1.0, 2.0]
        @test_throws SingularException pde_solve(Sg, [1.0, 2.0]; solver = :suitesparse)

        # The Cholesky route stays for a positive definite matrix, on a non-uniform mesh
        Ωn = mesh(domain(S2, :boundary => boundary_symbols(S2)), (9, 7), (true, true))
        change_points!(Ωn, (range(0.0, 1.0; length = 9) .^ 1.25, range(0.0, 1.0; length = 7) .^ 1.5))
        Wn = gridspace(Ωn)
        a_spd = form(Wn, Wn, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v))
        l_spd = form(Wn, v -> innerₕ(Rₕ(Wn, x -> 1.0), v))
        A_s, F_s = assemble(a_spd, l_spd; dirichlet = ZERO_BC, symmetrize = true)
        @test suitesparse_factorize(A_s).sym === :spd
        @test isapprox(suitesparse_factorize(A_s) \ F_s, Matrix(A_s) \ F_s; atol = 1.0e-12)
    end

    # A dense-backed form assembles into a `Matrix`; `refactor!` converts it as the factorize
    # and solve form methods do. Oracle: the refilled factorization solves to the dense `\`
    # answer of the matrix assembled with the new coefficient, which differs from the first
    # factorization's.
    @testset "refactor!, dense-backed form" begin
        I1 = interval(0.0, 1.0)
        Ωd = mesh(
            domain(I1, :boundary => boundary_symbols(I1)), 12, false;
            backend = backend(matrix_type = Matrix{Float64})
        )
        Wd = gridspace(Ωd)
        bc = :boundary => x -> 0.0
        a1 = form(Wd, Wd, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        a2 = form(Wd, Wd, (u, v) -> 3 * inner₊(∇ₕ(u), ∇ₕ(v)))
        rhs = ones(size(assemble(a1; dirichlet = bc), 1))
        A2 = assemble(a2; dirichlet = bc)
        @test A2 isa Matrix{Float64}
        u2 = A2 \ rhs    # dense LU: not the sparse code under test
        @test !(assemble(a1; dirichlet = bc) \ rhs ≈ u2)

        fact = sparse_factorize(a1; solver = :suitesparse, dirichlet = bc)
        @test refactor!(fact, a2; dirichlet = bc) === fact
        @test fact \ rhs ≈ u2 rtol = 1e-10
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
