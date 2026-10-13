# Normally test/runtests.jl includes test/ext/SolverContracts.jl immediately before this
# file (SolverContracts.jl's own header comment documents that ordering contract), so
# `Main.ExtSolverContracts` already exists by the time the `using ..ExtSolverContracts`
# below runs. Included here too, guarded, so `include("test/ext/mumps_ext.jl")`
# also works standalone without double-defining the module when
# runtests.jl already loaded it.
if !isdefined(Main, :ExtSolverContracts)
    include(joinpath(@__DIR__, "SolverContracts.jl"))
end

module TestMUMPSExt

using Test
using Bramble
using Bramble: MUMPSFactorization, mumps_factorize, mumps_refactor!, mumps_solve, refactor!
using Bramble: change_points!, sparse_factorize
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

# A `MethodError` that names the offending keyword, not any other failure.
function throws_unknown_kw(f, kw::Symbol)
    try
        f()
    catch e
        return e isa MethodError && occursin(string(kw), sprint(showerror, e))
    end
    return false
end

# The MUMPS instance of a factorization no caller can reach once this returns; built in a
# function of its own so no frame of the testset roots it.
@noinline dropped_instance(A) = mumps_factorize(A).mumps

@testset "MUMPS extension" begin
    @testset "Poisson 1D/2D/3D (sym = :spd)" begin
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

    # Helmholtz and saddle-point systems are symmetric indefinite.
    @testset "symmetric indefinite (:symmetric)" begin
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

    @testset "convection-diffusion (:unsymmetric)" begin
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
        @test @inferred(ldiv!(u1, fact, F)) === u1
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

    # Exercises factorization reuse and mumps_refactor!.
    @testset "factorization reuse, refactor" begin
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

        # Custom memory relaxation and METIS ordering (7 => 5). The solution is the same
        # under any ordering, so the controls are read back: MUMPS's defaults are 7 and 5.
        fact = mumps_factorize(p.A; sym = :spd, icntl = (7 => 5, 14 => 35))
        @test fact.mumps.icntl[7] == 5
        @test fact.mumps.icntl[14] == 35
        # INFOG(7) is the ordering MUMPS actually used in the analysis (the default picks 2)
        @test fact.mumps.infog[7] == 5
        u = fact \ p.F
        @test isapprox(u, p.u_ref; atol = 1.0e-12)

        # pde_solve with MUMPSFactorization directly
        @test isapprox(pde_solve(fact, p.F), u; atol = 1.0e-12)

        # a numeric update reusing that symbolic METIS analysis, on a matrix whose solution
        # differs: every entry doubled halves it, Dirichlet rows included
        A_modified = copy(p.A)
        nonzeros(A_modified) .*= 2.0
        @test !isapprox(A_modified \ p.F, p.u_ref; atol = 1.0e-12)
        refactor!(fact, A_modified)
        @test isapprox(fact \ p.F, A_modified \ p.F; atol = 1.0e-12)

        # a real control (relative pivoting threshold; MUMPS's default is -1.0), read back
        # only: under sym = :spd MUMPS does not use CNTL(1)
        fact_c = mumps_factorize(p.A; sym = :spd, cntl = (1 => 0.5,))
        @test fact_c.mumps.cntl[1] == 0.5
        @test isapprox(fact_c \ p.F, p.u_ref; atol = 1.0e-12)
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
            @test_throws ArgumentError mumps_factorize(a; dirichlet = bc, symmetrize = symmetrize)
            @test_throws ArgumentError sparse_factorize(
                a; solver = :mumps, dirichlet = bc, symmetrize = symmetrize
            )
        end

        # The plain call still solves, with the Dirichlet data lifted into F by assemble
        A, F = assemble(a, l; dirichlet = bc)
        @test mumps_factorize(a; dirichlet = bc) \ F ≈ Matrix(A) \ F
        @test sparse_factorize(a; solver = :mumps, dirichlet = bc) \ F ≈ Matrix(A) \ F
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

        @test isapprox(mumps_factorize(A_h) \ F_h, u_ref; atol = 1.0e-10)
        @test isapprox(sparse_factorize(A_h; solver = :mumps) \ F_h, u_ref; atol = 1.0e-10)
        @test isapprox(pde_solve(A_h, F_h; solver = :mumps), u_ref; atol = 1.0e-10)
        # `symmetrize = true` no longer forces Cholesky
        u_form = mumps_solve(a_helm, l_helm; dirichlet = ZERO_BC, symmetrize = true)
        @test isapprox(parent(u_form), u_ref; atol = 1.0e-10)

        # Symmetric with a positive diagonal, a zero pivot in the LDLᵀ order: solvable
        M = sparse([1.0 1 0; 1 1 1; 0 1 1])
        b = [1.0, 2.0, 3.0]
        @test isapprox(mumps_factorize(M) \ b, Matrix(M) \ b; atol = 1.0e-12)

        # Singular: the fallback reports it rather than returning a wrong answer
        Sg = sparse([1.0 1; 1 1])
        @test_throws SingularException mumps_factorize(Sg) \ [1.0, 2.0]
        @test_throws SingularException pde_solve(Sg, [1.0, 2.0]; solver = :mumps)

        # The Cholesky route stays for a positive definite matrix, on a non-uniform mesh
        Ωn = mesh(domain(S2, :boundary => boundary_symbols(S2)), (9, 7), (true, true))
        change_points!(Ωn, (range(0.0, 1.0; length = 9) .^ 1.25, range(0.0, 1.0; length = 7) .^ 1.5))
        Wn = gridspace(Ωn)
        a_spd = form(Wn, Wn, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v))
        l_spd = form(Wn, v -> innerₕ(Rₕ(Wn, x -> 1.0), v))
        A_s, F_s = assemble(a_spd, l_spd; dirichlet = ZERO_BC, symmetrize = true)
        @test mumps_factorize(A_s).sym == 1
        @test isapprox(mumps_factorize(A_s) \ F_s, Matrix(A_s) \ F_s; atol = 1.0e-12)
    end

    # #716: a singular matrix of the same pattern is refused, and the factorization it
    # leaves behind refuses to solve until a refactorization succeeds.
    @testset "refactor onto a singular matrix (#716)" begin
        p = poisson_system(Val(1), 15; source = x -> 1.0)
        Z = copy(p.A)
        nonzeros(Z) .= 0
        B = copy(p.A)
        nonzeros(B) .*= 2
        for sym in (:unsymmetric, :spd, :auto)
            fact = mumps_factorize(p.A; sym = sym)
            @test_throws SingularException mumps_refactor!(fact, Z)
            @test_throws SingularException refactor!(fact, Z)
            # as SuiteSparse does after its own refused refactorization
            @test_throws SingularException fact \ p.F
            @test_throws SingularException ldiv!(similar(p.F), fact, p.F)
            b = copy(p.F)
            @test_throws SingularException ldiv!(fact, b)
            @test b == p.F
            @test mumps_refactor!(fact, B) === fact
            @test isapprox(fact \ p.F, p.u_ref / 2; atol = 1.0e-12)
        end

        # `:auto` guessed SYM = 1 on an SPD matrix; a refactorization onto a nonsingular
        # indefinite one of the same pattern repeats the factorization's SYM = 2 fallback,
        # with the caller's controls
        n = 10
        A = sparse(SymTridiagonal(fill(4.0, n), fill(-1.0, n - 1)))
        P = sparse(SymTridiagonal(fill(0.0, n), fill(1.0, n - 1)))
        b = collect(1.0:n)
        fact = mumps_factorize(A; icntl = (14 => 35,))
        @test fact.sym == 1
        @test mumps_refactor!(fact, P) === fact
        @test fact.sym == 2
        @test fact.mumps.icntl[14] == 35
        @test norm(fact \ b - Matrix(P) \ b) < 1.0e-10
        @test mumps_refactor!(fact, A) === fact
        @test norm(fact \ b - Matrix(A) \ b) < 1.0e-10
        # without `:auto` the SYM = 1 request stands, and the indefinite matrix is refused
        fact_spd = mumps_factorize(A; sym = :spd)
        @test_throws SingularException mumps_refactor!(fact_spd, P)
    end

    @testset "unknown keyword throws (#657)" begin
        A = sparse(SymTridiagonal(fill(4.0, 6), fill(-1.0, 5)))
        b = collect(1.0:6)
        ref = Matrix(A) \ b
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ω = mesh(domain(S, :boundary => boundary_symbols(S)), (6, 6), (true, true))
        W = gridspace(Ω)
        a_form = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v))
        bc = ZERO_BC

        @test throws_unknown_kw(() -> mumps_factorize(A; bogus = 1), :bogus)
        @test throws_unknown_kw(() -> mumps_solve(A, b; bogus = 1), :bogus)
        @test throws_unknown_kw(() -> mumps_factorize(a_form; dirichlet = bc, bogus = 1), :bogus)
        @test throws_unknown_kw(() -> pde_solve(A, b; solver = :mumps, bogus = 1), :bogus)
        @test throws_unknown_kw(() -> sparse_factorize(A; solver = :mumps, bogus = 1), :bogus)
        @test throws_unknown_kw(() -> factorize(A, MUMPSFactorization; bogus = 1), :bogus)
        # The `symmetrize` refusal still comes first on the form method
        @test_throws ArgumentError mumps_factorize(a_form; dirichlet = bc, symmetrize = true, bogus = 1)

        # The accepted keywords still solve
        @test mumps_factorize(A; sym = :spd, icntl = [14 => 30], cntl = [1 => 0.01]) \ b ≈ ref
        @test mumps_solve(A, b; sym = :spd, icntl = [14 => 30]) ≈ ref
        @test pde_solve(A, b; solver = :mumps, sym = :spd) ≈ ref

        # `:auto` on the indefinite 3×3 factors twice, the second time through `sym = 2`
        M = sparse([1.0 1 0; 1 1 1; 0 1 1])
        bM = [1.0, 2.0, 3.0]
        @test mumps_factorize(M).sym == 2
        @test mumps_factorize(M; icntl = [14 => 30], cntl = [1 => 0.01]) \ bM ≈ Matrix(M) \ bM

        # `:default` takes only `kind`, on every platform
        for Aᵢ in (A, sparse([4.0 1; 0 3]))
            bᵢ = ones(size(Aᵢ, 1))
            @test_throws ArgumentError pde_solve(Aᵢ, bᵢ; bogus = 1)
        end
        @test pde_solve(A, b; kind = :ldlt) ≈ ref
    end

    # #715: a factorization frees its MUMPS instance on `finalize`, and after an `:auto`
    # refactorization that instance is the SYM = 2 one, not the SYM = 1 one it replaced.
    @testset "finalize frees the held instance (#715)" begin
        n = 10
        A = sparse(SymTridiagonal(fill(4.0, n), fill(-1.0, n - 1)))
        P = sparse(SymTridiagonal(fill(0.0, n), fill(1.0, n - 1)))
        b = collect(1.0:n)

        fact = mumps_factorize(A)
        m1 = fact.mumps
        @test !MUMPS.is_finalized(m1)
        finalize(fact)
        @test MUMPS.is_finalized(m1)

        fact = mumps_factorize(A)
        m1 = fact.mumps
        @test mumps_refactor!(fact, P) === fact
        m2 = fact.mumps
        @test m2 !== m1
        @test MUMPS.is_finalized(m1)
        # the replaced instance's former owner is gone; the new one stays live
        GC.gc(true)
        @test !MUMPS.is_finalized(m2)
        @test norm(fact \ b - Matrix(P) \ b) < 1.0e-10
        finalize(fact)
        @test MUMPS.is_finalized(m2)
        # a second `finalize` is harmless
        finalize(fact)
        @test MUMPS.is_finalized(m2)
        # the freed factors are refused, not read as garbage
        @test_throws ArgumentError fact \ b
        @test_throws ArgumentError ldiv!(similar(b), fact, b)
        bb = copy(b)
        @test_throws ArgumentError ldiv!(fact, bb)
        @test bb == b
        @test_throws ArgumentError mumps_refactor!(fact, A)
        @test_throws ArgumentError refactor!(fact, A)

        # unreachable factorizations are freed by the collector
        mref = dropped_instance(A)
        GC.gc(true)
        GC.gc(true)
        yield()
        @test MUMPS.is_finalized(mref)
    end

    @testset "Error handling & validation" begin
        p = poisson_system(Val(1), 5; source = x -> 1.0)
        validation_contract(
            p;
            factorize = mumps_factorize,
            backend_refactor! = mumps_refactor!,
            invalid_sym_solve = (A, F) -> mumps_solve(A, F; sym = :invalid_sym)
        )
    end
end

end # module
