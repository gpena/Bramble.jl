# Normally test/runtests.jl includes test/ext/SolverContracts.jl immediately before this
# file (SolverContracts.jl's own header comment documents that ordering contract), so
# `Main.ExtSolverContracts` already exists by the time the `using ..ExtSolverContracts`
# below runs. Included here too, guarded, so `include("test/ext/appleaccelerate_ext.jl")`
# also works standalone (the S5.3 CHECK) without double-defining the module when
# runtests.jl already loaded it.
if !isdefined(Main, :ExtSolverContracts)
    include(joinpath(@__DIR__, "SolverContracts.jl"))
end

module TestAppleAccelerateExt

using Test
using Bramble
using Bramble: AccelerateFactorization, accelerate_factorize, accelerate_refactor!,
               accelerate_solve
using Bramble: change_points!, sparse_factorize
using LinearAlgebra
using SparseArrays
using Random

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

# A `MethodError` that names the offending keyword, not any other failure.
function throws_unknown_kw(f, kw::Symbol)
    try
        f()
    catch e
        return e isa MethodError && occursin(string(kw), sprint(showerror, e))
    end
    return false
end

@testset "AppleAccelerate extension" begin
    if !Sys.isapple()
        @testset "Non-macOS guard" begin
            # Issue #84's failure mode: a platform-gated testset that silently reports
            # nothing run, rather than an explicit signal that it was skipped. This one
            # still exercises the macOS-only fallback below, but the #142 accuracy audit
            # further down (`accelerate_factorize`/`accelerate_solve` vs `LinearAlgebra`)
            # only runs inside the `else` (macOS) branch, so a non-macOS run must say so.
            @warn "AppleAccelerate accuracy audit (#142) skipped: not running on macOS (Sys.isapple() == false)."
            A = spdiagm(0 => [2.0, 2.0], 1 => [-1.0], -1 => [-1.0])
            F = [1.0, 1.0]
            @test_throws ArgumentError accelerate_factorize(A)
            @test_throws ArgumentError accelerate_solve(A, F)
            @test_throws ArgumentError pde_solve(A, F; solver = :accelerate)
        end
    else
        @testset "Poisson 1D/2D/3D (sym = :spd)" begin
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

        @testset "convection-diffusion (:unsymmetric)" begin
            cd = convection_diffusion_system(10)
            @test !issymmetric(cd.A)
            @test isapprox(
                accelerate_solve(cd.A, cd.F; sym = :unsymmetric), cd.u_ref; atol = 1.0e-12
            )

            # Auto-detection correctly selects unsymmetric LUTPP
            u_auto = pde_solve(cd.A, cd.F; solver = :accelerate, sym = :auto)
            @test isapprox(u_auto, cd.u_ref; atol = 1.0e-12)
        end

        # gpena/Bramble.jl#246 (R2): `:default` used to dispatch every macOS solve to
        # Accelerate whenever the extension was loaded, but Accelerate only wins the
        # factorisations it reaches on the symmetric side (SPD/Cholesky, LDLᵀ) -- measured at
        # 1.2-1.3x -- and loses 2.3-3.6x to plain `A \ F` on unsymmetric systems, of which
        # convection-diffusion is an ordinary example, not an edge case. `:default` now takes
        # Accelerate only when `issymmetric(A)` (`sym = :auto`, the default) or an explicit
        # symmetric `sym` hint says so; an explicit `sym = :unsymmetric` is trusted the same
        # way and skips straight to `A \ F` without paying for the check. This testset pins
        # that narrowing so it cannot regress silently.
        # pde_solve(:default) picks Accelerate only for the symmetric case, where it wins.
        @testset "pde_solve(:default): symmetric (#246)" begin
            p = poisson_system(Val(2), 8; source = x -> 1.0)
            @test issymmetric(p.A)
            @test isapprox(pde_solve(p.A, p.F), p.u_ref; atol = 1.0e-12)
            # Routed, not only accurate: `A \ F` gives the same values. `:default` hands a
            # symmetric `A` to `accelerate_solve` with every keyword, so a `kind` it rejects
            # throws, where `A \ F` would ignore it.
            @test_throws ArgumentError pde_solve(p.A, p.F; kind = :nonsense)
            # The plain call, with no keyword at all, reaches Accelerate too. On this 2D
            # Poisson system Accelerate's result differs from `A \ F` in the last bits and
            # is repeatable, so equality tells the two routes apart.
            @test pde_solve(p.A, p.F) == accelerate_solve(p.A, p.F)
            @test pde_solve(p.A, p.F) != p.A \ p.F

            cd = convection_diffusion_system(20)
            @test !issymmetric(cd.A)
            # `:default` on an unsymmetric matrix must be exactly `A \ F`, not merely close to
            # it: unlike a genuine Accelerate result (not bit-identical across separate
            # dispatches, per the #142 audit above), `A \ F` here is deterministic, so exact
            # equality is what shows it never touched Accelerate at all.
            @test pde_solve(cd.A, cd.F) == cd.A \ cd.F

            # An explicit `solver = :accelerate` is honoured unconditionally, symmetric or
            # not -- this narrows only the automatic `:default` choice.
            @test isapprox(
                pde_solve(cd.A, cd.F; solver = :accelerate, sym = :unsymmetric), cd.u_ref;
                atol = 1.0e-12
            )
        end

        # Exercises factorization reuse and accelerate_refactor!.
        @testset "factorization reuse, refactor" begin
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

        @testset "factorize(a): symmetrize refused" begin
            Ωₕ = mesh(
                domain(
                    interval(0.0, 1.0) × interval(0.0, 1.0),
                    :boundary => boundary_symbols(interval(0.0, 1.0) × interval(0.0, 1.0))
                ), (9, 7), (true, true)
            )
            change_points!(
                Ωₕ, (range(0.0, 1.0; length = 9) .^ 1.25, range(0.0, 1.0; length = 7) .^ 1.5)
            )
            Wₕ = gridspace(Ωₕ)
            a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
            l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, x -> 1.0), v))
            bc = :boundary => x -> 1.0 + x[1]
            for symmetrize in (true, false)
                @test_throws ArgumentError accelerate_factorize(
                    a; dirichlet = bc, symmetrize = symmetrize
                )
                @test_throws ArgumentError sparse_factorize(
                    a; solver = :accelerate, dirichlet = bc, symmetrize = symmetrize
                )
            end

            # The plain call still solves, with the Dirichlet data lifted into F by assemble
            A, F = assemble(a, l; dirichlet = bc)
            @test accelerate_factorize(a; dirichlet = bc) \ F ≈ Matrix(A) \ F
            @test sparse_factorize(a; solver = :accelerate, dirichlet = bc) \ F ≈ Matrix(A) \ F
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

            @test isapprox(accelerate_factorize(A_h) \ F_h, u_ref; atol = 1.0e-10)
            @test isapprox(sparse_factorize(A_h; solver = :accelerate) \ F_h, u_ref; atol = 1.0e-10)
            @test isapprox(pde_solve(A_h, F_h; solver = :accelerate), u_ref; atol = 1.0e-10)
            # `symmetrize = true` no longer forces Cholesky
            u_form = accelerate_solve(a_helm, l_helm; dirichlet = ZERO_BC, symmetrize = true)
            @test isapprox(parent(u_form), u_ref; atol = 1.0e-10)

            # Symmetric with a positive diagonal, a zero pivot in the LDLᵀ order: solvable
            M = sparse([1.0 1 0; 1 1 1; 0 1 1])
            b = [1.0, 2.0, 3.0]
            @test isapprox(accelerate_factorize(M) \ b, Matrix(M) \ b; atol = 1.0e-12)

            # Singular: the fallback reports it rather than returning a wrong answer
            Sg = sparse([1.0 1; 1 1])
            @test_throws SingularException accelerate_factorize(Sg) \ [1.0, 2.0]
            @test_throws SingularException pde_solve(Sg, [1.0, 2.0]; solver = :accelerate)

            # The Cholesky route stays for a positive definite matrix, on a non-uniform mesh
            Ωn = mesh(domain(S2, :boundary => boundary_symbols(S2)), (9, 7), (true, true))
            change_points!(Ωn, (range(0.0, 1.0; length = 9) .^ 1.25, range(0.0, 1.0; length = 7) .^ 1.5))
            Wn = gridspace(Ωn)
            a_spd = form(Wn, Wn, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v))
            l_spd = form(Wn, v -> innerₕ(Rₕ(Wn, x -> 1.0), v))
            A_s, F_s = assemble(a_spd, l_spd; dirichlet = ZERO_BC, symmetrize = true)
            @test accelerate_factorize(A_s).sym === :spd
            @test isapprox(accelerate_factorize(A_s) \ F_s, Matrix(A_s) \ F_s; atol = 1.0e-12)
        end

        @testset "LU and QR on a Hermitian matrix (#653)" begin
            # AppleAccelerate stores a Hermitian matrix as its lower triangle; LU and QR must
            # factor the full matrix. Helmholtz -Δu - 50u (uniform) and SPD (non-uniform).
            S3 = interval(0.0, 1.0) × interval(0.0, 1.0)
            for (k, uni) in ((-50.0, true), (1.0, false))
                Ω3 = mesh(domain(S3, :boundary => boundary_symbols(S3)), (10, 9), (uni, uni))
                W3 = gridspace(Ω3)
                a3 = form(W3, W3, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + k * innerₕ(u, v))
                l3 = form(W3, v -> innerₕ(Rₕ(W3, x -> 1.0 + x[1]), v))
                A, F = assemble(a3, l3; dirichlet = ZERO_BC, symmetrize = true)
                @test issymmetric(A)
                ref = Matrix(A) \ F
                A2 = copy(A)
                foreach(i -> A2[i, i] += 2, 1:size(A, 1))
                ref2 = Matrix(A2) \ F
                for o in ((sym = :unsymmetric,), (kind = :lu,), (kind = :lutpp,), (kind = :qr,))
                    f = accelerate_factorize(A; o...)
                    @test f.sym === (get(o, :kind, nothing) === :qr ? :qr : :unsymmetric)
                    @test isapprox(f \ F, ref; rtol = 1.0e-10)
                    @test isapprox(accelerate_refactor!(f, A2) \ F, ref2; rtol = 1.0e-10)
                end
            end

            # Complex Hermitian, not symmetric: explicit LU factors the full matrix
            du = fill(1.0 + 0.5im, 5)
            H = sparse(Tridiagonal(conj.(du), fill(4.0 + 0im, 6), du))
            bh = ComplexF64.(1:6)
            @test !issymmetric(H) && ishermitian(H)
            fH = accelerate_factorize(H; kind = :lu)
            @test isapprox(fH \ bh, Matrix(H) \ bh; rtol = 1.0e-12)

            # Unsymmetric 3×3: LU on a matrix that was never stored as a triangle
            Au = sparse([4.0 1 0; 2 5 1; 0 3 6])
            bu = [1.0, 2, 3]
            u_u = accelerate_factorize(Au; sym = :unsymmetric) \ bu
            @test isapprox(u_u, Matrix(Au) \ bu; rtol = 1.0e-12)
        end

        # The pivoted LDLᵀ and LUTPP return a solution for a singular matrix, so `:auto`
        # checks each factorization once against a fixed vector: every route throws, for a
        # consistent right-hand side too, as SuiteSparse does (gpena/Bramble.jl#668).
        @testset ":auto: singular throws on every route" begin
            cases = (
                ([0.0 0; 0 1], [1.0, 1]),     # direct LDLᵀ, non-positive diagonal
                ([-1.0 1; 1 -1], [1.0, 2]),   # direct LDLᵀ
                (zeros(2, 2), [1.0, 1]),      # direct LDLᵀ, the zero matrix
                ([1.0 1; 2 2], [1.0, 2]),     # LU, consistent: libSparse's residual is 0
                ([1.0 1; 2 2], [1.0, 3]),     # LU, inconsistent
                ([1.0 1; 1 1], [1.0, 1])      # Cholesky fails, LDLᵀ fallback, consistent
            )
            for (M, b) in cases
                S = sparse(M)
                @test_throws SingularException accelerate_factorize(S) \ b
                @test_throws SingularException sparse_factorize(S; solver = :accelerate) \ b
                @test_throws SingularException pde_solve(S, b; solver = :accelerate)
                @test_throws SingularException accelerate_solve(S, b)
                # `nothing` spells `:auto` for `sym` and for `kind`
                for o in ((sym = nothing,), (kind = nothing,), (sym = nothing, kind = nothing))
                    @test_throws SingularException accelerate_factorize(S; o...)
                end
            end
            @test_throws SingularException accelerate_factorize(sparse(zeros(Float32, 3, 3)))

            # Nonsingular indefinite, unsymmetric and Float32 matrices still solve
            for M in ([0.0 1 0; 1 0 1; 0 1 -2], [4.0 1 0; 2 5 1; 0 3 6], [-2.0 1; 1 -3],
                    Float32[4 1 0; 2 5 1; 0 3 6])
                b = collect(one(eltype(M)):size(M, 1))
                tol = sqrt(eps(eltype(M)))
                @test isapprox(accelerate_factorize(sparse(M)) \ b, M \ b; rtol = tol)
            end
            # A positive diagonal and indefinite under `sym = nothing`: the LDLᵀ fallback
            # runs as under `:auto`, instead of libSparse's Cholesky error escaping
            Mi = [1.0 2; 2 1]
            for o in ((sym = nothing,), (kind = nothing,))
                @test isapprox(accelerate_factorize(sparse(Mi); o...) \ [1.0, 2], Mi \ [1.0, 2])
            end

            # A refused `accelerate_refactor!` leaves every later solve refusing too, until
            # a refactor onto a nonsingular matrix of the same pattern
            A1 = sparse([1.0 1; 2 3])
            f = accelerate_factorize(A1)
            @test_throws SingularException accelerate_refactor!(f, sparse([1.0 1; 2 2]))
            b = [1.0, 3.0]
            x = zeros(2)
            @test_throws SingularException f \ b
            @test_throws SingularException f \ ComplexF64.(b)
            @test_throws SingularException ldiv!(x, f, b)
            @test_throws SingularException ldiv!(f, copy(b))
            @test accelerate_refactor!(f, A1) === f
            @test isapprox(f \ b, Matrix(A1) \ b; rtol = 1.0e-12)

            # On the `:auto` Cholesky route, libSparse's refactor fails on a matrix that is
            # not positive definite; the refactor falls back to LDLᵀ as the factorization
            # does, so a singular matrix throws and an indefinite one solves
            fc = accelerate_factorize(sparse([2.0 1; 1 2]))
            @test fc.sym === :spd
            @test_throws SingularException accelerate_refactor!(fc, sparse([1.0 1; 1 1]))
            @test_throws SingularException fc \ b
            @test accelerate_refactor!(fc, sparse([1.0 2; 2 1])) === fc
            @test fc.sym === :symmetric
            @test isapprox(fc \ b, [1.0 2; 2 1] \ b; rtol = 1.0e-12)
            fc = accelerate_factorize(sparse([2.0 1; 1 2]))
            @test accelerate_refactor!(fc, sparse([1.0 2; 2 1])) === fc
            @test isapprox(fc \ b, [1.0 2; 2 1] \ b; rtol = 1.0e-12)

            # What the docstrings concede: the ordering rounds a pure-Neumann Laplacian's
            # zero pivot to about eps, so it solves without an error and a huge solution
            nN = 100
            N = spdiagm(-1 => fill(-1.0, nN - 1), 0 => [1.0; fill(2.0, nN - 2); 1.0],
                1 => fill(-1.0, nN - 1))
            @test norm(accelerate_factorize(N) \ ones(nN), Inf) > 1.0e12

            # An explicit route stays unchecked, as #668 scopes the check to `:auto`
            @test accelerate_factorize(sparse([1.0 1; 2 2]); sym = :unsymmetric) isa
                  AccelerateFactorization

            # The check runs per factorization, not per solve: `ldiv!` allocates nothing,
            # after a factorization and after a refactor, on the LU and LDLᵀ routes
            ldiv_bytes(x, f, b) = (ldiv!(x, f, b); @allocated ldiv!(x, f, b))
            ldiv_bytes!(f, b) = (ldiv!(f, b); @allocated ldiv!(f, b))
            for M in ([4.0 1 0; 2 5 1; 0 3 6], [0.0 1 0; 1 0 1; 0 1 -2])
                S = sparse(M)
                fM = accelerate_factorize(S)
                b3 = [1.0, 2, 3]
                x3 = zeros(3)
                @test ldiv_bytes(x3, fM, b3) == 0
                @test ldiv_bytes!(fM, copy(b3)) == 0
                accelerate_refactor!(fM, 2 * S)
                @test ldiv_bytes(x3, fM, b3) == 0
                @test isapprox(x3, (2 * M) \ b3; rtol = 1.0e-12)
            end
        end

        # libSparse's Cholesky and LDLᵀ factor `A = LDLᴴ`: `:auto` routes on `ishermitian`,
        # so a complex symmetric matrix goes to LU, and an explicit Cholesky or LDLᵀ refuses
        # any matrix that is not Hermitian, real or complex, with an `ArgumentError`.
        @testset "Hermitian routing (#665)" begin
            Ac = sparse(SymTridiagonal(fill(4.0 + 1.0im, 6), fill(1.0 - 0.5im, 5)))
            b = ComplexF64.(1:6) .+ 0.5im
            @test issymmetric(Ac) && !ishermitian(Ac)
            fc = accelerate_factorize(Ac)
            @test fc.sym === :unsymmetric
            @test isapprox(fc \ b, Matrix(Ac) \ b; rtol = 1.0e-12)
            @test isapprox(accelerate_refactor!(fc, 2 * Ac) \ b, Matrix(2 * Ac) \ b;
                rtol = 1.0e-12)
            explicit = ((sym = :spd,), (sym = :definite,), (kind = :cholesky,),
                (sym = :symmetric,), (kind = :ldlt,))
            Ar = sparse([4.0 1; 2 5])
            for o in explicit
                @test_throws ArgumentError accelerate_factorize(Ac; o...)
                @test_throws ArgumentError accelerate_factorize(Ar; o...)
                @test_throws ArgumentError accelerate_solve(Ar, [1.0, 2]; o...)
            end

            # Complex Hermitian on the three `:auto` routes: positive definite (Cholesky),
            # a non-positive diagonal (LDLᵀ), and a positive diagonal but indefinite
            # (Cholesky fails, LDLᵀ fallback); the explicit kinds agree
            hh(d, e) = sparse(Tridiagonal(conj.(fill(e, 5)), fill(d, 6), fill(e, 5)))
            Hp = hh(4.0 + 0im, 1.0 + 0.5im)
            Hn = hh(-1.0 + 0im, 1.0 + 0.5im)
            Hi = hh(1.0 + 0im, 1.0 + 0.5im)
            @test minimum(eigvals(Hermitian(Matrix(Hi)))) < 0
            for (H, route) in ((Hp, :spd), (Hn, :symmetric), (Hi, :symmetric))
                @test ishermitian(H) && !issymmetric(H)
                f = accelerate_factorize(H)
                @test f.sym === route
                @test isapprox(f \ b, Matrix(H) \ b; rtol = 1.0e-12)
            end
            @test isapprox(accelerate_factorize(Hp; sym = :spd) \ b, Matrix(Hp) \ b;
                rtol = 1.0e-12)
            @test isapprox(accelerate_factorize(Hi; kind = :ldlt) \ b, Matrix(Hi) \ b;
                rtol = 1.0e-12)

            # `refactor!` on the complex Hermitian Cholesky route: an indefinite matrix
            # falls back to LDLᵀ, a singular one throws
            fh = accelerate_factorize(Hp)
            @test accelerate_refactor!(fh, Hi) === fh
            @test fh.sym === :symmetric
            @test isapprox(fh \ b, Matrix(Hi) \ b; rtol = 1.0e-12)
            Hs = sparse(ComplexF64[1 1im; -1im 1])
            @test_throws SingularException accelerate_refactor!(
                accelerate_factorize(sparse(ComplexF64[2 1im; -1im 2])), Hs)

            # A Cholesky or LDLᵀ factorization refuses a non-Hermitian matrix at
            # `refactor!` too, and keeps solving with its old factors
            As = sparse([4.0 1; 1 5])
            for (A0, A1, o) in ((As, Ar, (sym = :spd,)), (As, Ar, (kind = :ldlt,)),
                (As, Ar, (sym = :auto,)), (Hp, Ac, (sym = :auto,)),
                (Hn, Ac, (sym = :auto,)))
                f = accelerate_factorize(A0; o...)
                bb = b[1:size(A0, 1)]
                @test_throws ArgumentError accelerate_refactor!(f, A1)
                @test isapprox(f \ bb, Matrix(A0) \ bb; rtol = 1.0e-12)
            end
        end

        # A matrix right-hand side is one multi-column libSparse solve on every kind;
        # `\` and the three-argument `ldiv!` reach it through `LinearAlgebra`
        @testset "matrix right-hand side (#665)" begin
            A = sparse(SymTridiagonal(fill(4.0, 7), fill(-1.0, 6)))
            Au = sparse([4.0 1 0; 2 5 1; 0 3 6])
            Ai = sparse([0.0 1 0; 1 0 1; 0 1 -2])
            for (M, kinds) in ((A, (:auto, :cholesky, :ldlt, :lu, :qr)),
                (Au, (:auto, :lu, :qr)), (Ai, (:auto, :ldlt, :lu)))
                n = size(M, 1)
                B = reshape(collect(1.0:(3 * n)), n, 3)
                B0 = copy(B)
                ref = Matrix(M) \ B
                for k in kinds
                    f = accelerate_factorize(M; kind = k)
                    X = f \ B
                    @test X !== B && B == B0
                    @test isapprox(X, ref; rtol = 1.0e-10)
                    Y = similar(B)
                    @test ldiv!(Y, f, B) === Y
                    @test isapprox(Y, ref; rtol = 1.0e-10)
                    @test B == B0
                    C = copy(B)
                    @test ldiv!(f, C) === C
                    @test isapprox(C, ref; rtol = 1.0e-10)
                    @test isapprox(reduce(hcat, (f \ B[:, j] for j in 1:3)), X;
                        rtol = 1.0e-12)
                    # No columns: nothing to solve, where libSparse would throw
                    @test size(f \ zeros(n, 0)) == (n, 0)
                    Z = zeros(n, 0)
                    @test ldiv!(Z, f, zeros(n, 0)) === Z
                    @test ldiv!(f, Z) === Z
                end
            end
            fA = accelerate_factorize(A)
            @test_throws DimensionMismatch fA \ ones(3, 2)
            @test_throws DimensionMismatch ldiv!(fA, ones(3, 0))
            B7 = reshape(collect(1.0:21), 7, 3)
            for X in (fill(NaN, 7, 4), zeros(7, 2), zeros(3, 7), zeros(6, 3))
                @test_throws DimensionMismatch ldiv!(X, fA, B7)
            end
            @test_throws DimensionMismatch ldiv!(zeros(7, 1), fA, zeros(7, 0))

            # A matching `Matrix` goes straight to libSparse: no copy
            ldivm_bytes(X, f, B) = (ldiv!(X, f, B); @allocated ldiv!(X, f, B))
            ldivm_bytes!(f, B) = (ldiv!(f, B); @allocated ldiv!(f, B))
            @test ldivm_bytes(similar(B7), fA, B7) == 0
            @test ldivm_bytes!(fA, copy(B7)) == 0

            # Views, adjoints, other precisions and complex right-hand sides on a real
            # factorization go through a copy, by `\` and by both `ldiv!`s
            ref7 = Matrix(A) \ B7
            Bc7 = complex.(B7, 2B7)
            refc = Matrix(A) \ Bc7
            @test isapprox(fA \ view(Bc7, :, 1:2), refc[:, 1:2]; rtol = 1.0e-12)
            @test isapprox(fA \ copy(Bc7')', refc; rtol = 1.0e-12)
            @test isapprox(fA \ copy(B7')', ref7; rtol = 1.0e-12)
            @test isapprox(fA \ Float32.(B7), ref7; rtol = 1.0e-6)
            Xc = similar(Bc7)
            @test ldiv!(Xc, fA, Bc7) === Xc
            @test isapprox(Xc, refc; rtol = 1.0e-12)
            Ct = copy(Bc7')'
            @test ldiv!(fA, Ct) === Ct
            @test isapprox(Ct, refc; rtol = 1.0e-12)
            xc = similar(Bc7, 7)
            @test ldiv!(xc, fA, Bc7[:, 2]) === xc
            @test isapprox(xc, refc[:, 2]; rtol = 1.0e-12)
            # A view with a row stride other than 1 goes through a copy too
            C14 = repeat(B7; inner = (2, 1))
            Cv = @view C14[1:2:end, :]
            @test ldiv!(fA, Cv) === Cv
            @test isapprox(Cv, ref7; rtol = 1.0e-12)
            Y14 = zeros(14, 3)
            Yv = @view Y14[1:2:end, :]
            @test ldiv!(Yv, fA, B7) === Yv
            @test isapprox(Y14[1:2:end, :], ref7; rtol = 1.0e-12)
            @test all(iszero, Y14[2:2:end, :])
            c14 = repeat(B7[:, 1]; inner = 2)
            cv = @view c14[1:2:end]
            @test ldiv!(fA, cv) === cv
            @test isapprox(cv, ref7[:, 1]; rtol = 1.0e-12)
            f32 = accelerate_factorize(SparseMatrixCSC{Float32, Int}(A))
            @test isapprox(f32 \ B7, ref7; rtol = 1.0e-5)
            @test eltype(f32 \ B7) == Float64
            @test isapprox(f32 \ Bc7, refc; rtol = 1.0e-5)
            @test isapprox(f32 \ B7[:, 1], ref7[:, 1]; rtol = 1.0e-5)
            X64 = similar(B7)
            @test ldiv!(X64, f32, B7) === X64
            @test isapprox(X64, ref7; rtol = 1.0e-5)

            # A complex B on a complex factorization, and on a real one
            Ac = sparse(SymTridiagonal(fill(4.0 + 1.0im, 6), fill(1.0 - 0.5im, 5)))
            Bc = reshape(ComplexF64.(1:12) .+ 0.5im, 6, 2)
            Hp = sparse(Tridiagonal(fill(1.0 - 0.5im, 5), fill(4.0 + 0im, 6),
                fill(1.0 + 0.5im, 5)))
            for H in (Ac, Hp)
                @test isapprox(accelerate_factorize(H) \ Bc, Matrix(H) \ Bc; rtol = 1.0e-12)
            end
            Ar = sparse(SymTridiagonal(fill(4.0, 6), fill(-1.0, 5)))
            @test isapprox(accelerate_factorize(Ar) \ Bc, Matrix(Ar) \ Bc; rtol = 1.0e-12)

            # A refused `refactor!` leaves matrix solves refusing too
            A1 = sparse([1.0 1; 2 3])
            f = accelerate_factorize(A1)
            @test_throws SingularException accelerate_refactor!(f, sparse([1.0 1; 2 2]))
            B2 = [1.0 2; 3 4]
            @test_throws SingularException f \ B2
            @test_throws SingularException f \ ComplexF64.(B2)
            @test_throws SingularException ldiv!(similar(B2), f, B2)
            @test_throws SingularException ldiv!(f, copy(B2))
            @test_throws SingularException ldiv!(f, zeros(2, 0))
            @test accelerate_refactor!(f, A1) === f
            @test isapprox(f \ B2, Matrix(A1) \ B2; rtol = 1.0e-12)
        end

        # `AASparseMatrix(A)` takes `Int64` indices: the triangle kinds widen an `Int32`
        # matrix, in `refactor!` and in the `:auto` LDLᵀ fallback too
        @testset "Int32 indices, every kind (#665)" begin
            A = sparse(SymTridiagonal(fill(4.0, 7), fill(-1.0, 6)))
            Ai = sparse([0.0 1 0; 1 0 1; 0 1 -2])
            Ax = sparse([1.0 2; 2 1])
            b7 = collect(1.0:7)
            to32(M) = SparseMatrixCSC{Float64, Int32}(M)
            @test to32(A) isa SparseMatrixCSC{Float64, Int32}
            for opts in (
                (;), (sym = :spd,), (kind = :cholesky,), (kind = :ldlt,),
                (sym = :symmetric,), (kind = :lu,), (kind = :qr,),
            )
                f = accelerate_factorize(to32(A); opts...)
                @test isapprox(f \ b7, Matrix(A) \ b7; rtol = 1.0e-12)
            end
            for opts in ((;), (kind = :ldlt,))
                f = accelerate_factorize(to32(Ai); opts...)
                @test isapprox(f \ [1.0, 2, 3], Matrix(Ai) \ [1.0, 2, 3]; rtol = 1.0e-12)
            end
            # indefinite with a positive diagonal: Cholesky fails, :auto falls to LDLᵀ
            f = accelerate_factorize(to32(Ax))
            @test isapprox(f \ [1.0, 2], Matrix(Ax) \ [1.0, 2]; rtol = 1.0e-12)
            for opts in ((;), (kind = :ldlt,))
                f = accelerate_factorize(to32(A); opts...)
                @test accelerate_refactor!(f, to32(2 * A)) === f
                @test isapprox(f \ b7, Matrix(2 * A) \ b7; rtol = 1.0e-12)
            end
            # :auto Cholesky refactored to an indefinite matrix: the LDLᵀ fallback
            f = accelerate_factorize(to32(sparse([2.0 1; 1 2])))
            @test accelerate_refactor!(f, to32(Ax)) === f
            @test isapprox(f \ [1.0, 2], Matrix(Ax) \ [1.0, 2]; rtol = 1.0e-12)
        end

        @testset "unknown keyword throws (#657)" begin
            A = sparse(SymTridiagonal(fill(4.0, 6), fill(-1.0, 5)))
            b = collect(1.0:6)
            ref = Matrix(A) \ b
            S = interval(0.0, 1.0) × interval(0.0, 1.0)
            Ω = mesh(domain(S, :boundary => boundary_symbols(S)), (6, 6), (true, true))
            W = gridspace(Ω)
            a_form = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v))

            @test throws_unknown_kw(() -> accelerate_factorize(A; bogus = 1), :bogus)
            @test throws_unknown_kw(() -> accelerate_solve(A, b; bogus = 1), :bogus)
            @test throws_unknown_kw(
                () -> accelerate_factorize(a_form; dirichlet = ZERO_BC, bogus = 1), :bogus
            )
            @test throws_unknown_kw(() -> accelerate_factorize(Matrix(A); bogus = 1), :bogus)
            @test throws_unknown_kw(() -> pde_solve(A, b; solver = :accelerate, bogus = 1), :bogus)
            @test throws_unknown_kw(
                () -> sparse_factorize(A; solver = :accelerate, bogus = 1), :bogus
            )
            @test throws_unknown_kw(() -> factorize(A, AccelerateFactorization; bogus = 1), :bogus)
            # The `symmetrize` refusal still comes first on the form method
            @test_throws ArgumentError accelerate_factorize(
                a_form; dirichlet = ZERO_BC, symmetrize = true, bogus = 1
            )

            # The accepted keywords still solve
            @test accelerate_factorize(A; sym = :symmetric, kind = :ldlt) \ b ≈ ref
            @test accelerate_solve(A, b; sym = :spd, kind = :cholesky) ≈ ref
            @test pde_solve(A, b; solver = :accelerate, kind = :ldlt) ≈ ref
            @test accelerate_factorize(Matrix(A); sym = :spd, kind = :cholesky) \ b ≈ ref

            # `:default` takes only `kind`, on every platform
            for Aᵢ in (A, sparse([4.0 1; 0 3]))
                @test_throws ArgumentError pde_solve(Aᵢ, ones(size(Aᵢ, 1)); bogus = 1)
            end
            @test pde_solve(A, b; kind = :ldlt) ≈ ref
        end

        @testset "Error handling & validation" begin
            validation_contract(
                poisson_system(Val(1), 5; source = x -> 1.0);
                factorize = accelerate_factorize,
                backend_refactor! = accelerate_refactor!,
                invalid_sym_solve = (A, F) -> accelerate_solve(A, F; sym = :invalid_sym)
            )
        end

        # gpena/Bramble.jl#142 asks whether Accelerate's libSparse/LBT-forwarded LAPACK differ
        # numerically from the plain `LinearAlgebra` path enough to need new suite
        # tolerances. Answered here by relative residual, ‖A*x - b‖ / ‖b‖, computed for
        # every symmetry branch `accelerate_factorize` dispatches on (sparse SPD Cholesky,
        # symmetric LDLᵀ, unsymmetric LUTPP, sparse QR) plus the dense `accelerate_factorize`
        # methods (cholesky, lu, qr and auto), against the
        # same right-hand sides solved by plain `LinearAlgebra.lu`/`\\`. `worst[]` is the
        # maximum observed across all of that -- the figure S5.5 quotes when answering
        # #142 -- printed at the end so it shows up in a plain `include` of this file, not
        # only inside `@testset` output.
        #
        # Run on this host (Apple M2, arm64) 2026-09-20: worst = 3.6e-14, from the sparse
        # SPD Cholesky path on the 1D Poisson system (n=41). Every existing tolerance in
        # this suite that guards a solve (`atol = 1.0e-12` throughout this file and
        # `test/form/sparse_solvers.jl:28`) sits two orders of magnitude above that --
        # **no tolerance change needed**. The dense rectangular QR (least-squares) case is
        # not a residual-to-zero problem by construction, so it is checked separately by
        # the *difference between solutions*, not folded into `worst[]`.
        @testset "Accuracy audit vs LinearAlgebra (#142)" begin
            relres(A, x, b) = norm(A * x - b) / norm(b)

            worst = Ref(0.0)
            worst_case = Ref("")
            function record!(name::String, r::Real)
                @test r < 1.0e-9
                if r > worst[]
                    worst[] = r
                    worst_case[] = name
                end
                return r
            end

            # sparse SPD Cholesky, 1D/2D/3D, accelerate vs dense lu(...) vs backslash
            for (D, n) in ((1, 41), (2, 16), (3, 8))
                p = poisson_system(Val(D), n)
                x_acc = accelerate_solve(p.A, p.F; sym = :spd)
                x_lu_dense = lu(Matrix(p.A)) \ p.F
                @test isapprox(x_acc, p.u_ref; atol = 1.0e-12)
                @test isapprox(x_lu_dense, p.u_ref; atol = 1.0e-12)
                record!("sparse SPD Cholesky, accelerate, D=$D", relres(p.A, x_acc, p.F))
                record!("sparse SPD Cholesky, dense lu ref, D=$D", relres(p.A, x_lu_dense, p.F))
            end

            # symmetric LDLᵀ and square sparse QR
            p2 = poisson_system(Val(2), 8; source = x -> 1.0)
            x_ldlt = accelerate_solve(p2.A, p2.F; kind = :ldlt)
            x_qr_sq = accelerate_solve(p2.A, p2.F; kind = :qr)
            record!("sparse symmetric LDLᵀ, accelerate", relres(p2.A, x_ldlt, p2.F))
            record!("sparse QR (square), accelerate", relres(p2.A, x_qr_sq, p2.F))

            # unsymmetric LUTPP. `pde_solve(:default)` on this same system is no longer part
            # of this audit: since #246/S5.6, `:default` routes an unsymmetric matrix to
            # plain `A \ F` rather than Accelerate, so it has nothing to do with Accelerate's
            # accuracy -- that routing is pinned in the "narrowed to the symmetric win"
            # testset above instead.
            cd = convection_diffusion_system(20)
            x_lutpp = accelerate_solve(cd.A, cd.F; sym = :unsymmetric)
            x_lu_dense_un = lu(Matrix(cd.A)) \ cd.F
            record!("sparse unsymmetric LUTPP, accelerate", relres(cd.A, x_lutpp, cd.F))
            record!("sparse unsymmetric, dense lu ref", relres(cd.A, x_lu_dense_un, cd.F))

            # dense accelerate_factorize vs plain LinearAlgebra, fixed seed for reproducibility
            rng = Random.Xoshiro(20260920)
            n = 60
            M = randn(rng, n, n)
            A_spd = M' * M + n * I
            b = randn(rng, n)
            x_dense_chol = accelerate_factorize(A_spd; sym = :spd) \ b
            x_dense_chol_ref = cholesky(A_spd) \ b
            record!("dense SPD, accelerate_factorize(:spd)", relres(A_spd, x_dense_chol, b))
            record!("dense SPD, cholesky ref", relres(A_spd, x_dense_chol_ref, b))

            A_un = randn(rng, n, n) + n * I
            b_un = randn(rng, n)
            x_dense_lu = accelerate_factorize(A_un; sym = :unsymmetric) \ b_un
            record!("dense unsymmetric, accelerate_factorize(:lu)", relres(A_un, x_dense_lu, b_un))

            x_dense_auto = accelerate_factorize(A_spd) \ b
            record!("dense :auto on SPD input", relres(A_spd, x_dense_auto, b))

            println(
                "S5.3 accuracy audit (#142): worst relative residual = ", worst[],
                ", case = \"", worst_case[], "\""
            )
        end
    end
end

end # module
