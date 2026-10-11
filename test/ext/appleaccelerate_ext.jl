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
