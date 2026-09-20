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

        # gpena/Bramble.jl#142: does Accelerate's libSparse/LBT-forwarded LAPACK differ
        # numerically from the plain `LinearAlgebra` path enough to need new suite
        # tolerances? Answered here by relative residual, ‖A*x - b‖ / ‖b‖, computed for
        # every symmetry branch `accelerate_factorize` dispatches on (sparse SPD Cholesky,
        # symmetric LDLᵀ, unsymmetric LUTPP, sparse QR, `pde_solve(:default)`) plus the
        # dense `accelerate_factorize` methods (:cholesky, :lu, :qr, :auto), against the
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

            # unsymmetric LUTPP, and pde_solve(:default) routing through accelerate_solve
            cd = convection_diffusion_system(20)
            x_lutpp = accelerate_solve(cd.A, cd.F; sym = :unsymmetric)
            x_lu_dense_un = lu(Matrix(cd.A)) \ cd.F
            x_default = pde_solve(cd.A, cd.F)
            record!("sparse unsymmetric LUTPP, accelerate", relres(cd.A, x_lutpp, cd.F))
            record!("sparse unsymmetric, dense lu ref", relres(cd.A, x_lu_dense_un, cd.F))
            record!("pde_solve(:default) on macOS w/ AppleAccelerate", relres(cd.A, x_default, cd.F))
            # Not `==`: vecLib's internal threading can reorder floating-point reductions
            # between two separately-dispatched calls to the same LUTPP factorization, so
            # bit-identity is not guaranteed even though both pick the same factorization
            # kind (`:auto` on an unsymmetric matrix and explicit `:unsymmetric` agree).
            @test isapprox(x_default, x_lutpp; atol = 1.0e-12)

            # dense accelerate_factorize vs plain LinearAlgebra, fixed seed for reproducibility
            rng = Random.Xoshiro(20260920)
            n = 60
            M = randn(rng, n, n)
            A_spd = M' * M + n * I
            b = randn(rng, n)
            x_dense_chol = accelerate_factorize(A_spd; sym = :spd) \ b
            x_dense_chol_ref = cholesky(A_spd) \ b
            @test isapprox(x_dense_chol, x_dense_chol_ref; atol = 1.0e-12)
            record!("dense SPD, accelerate_factorize(:spd)", relres(A_spd, x_dense_chol, b))
            record!("dense SPD, cholesky ref", relres(A_spd, x_dense_chol_ref, b))

            A_un = randn(rng, n, n) + n * I
            b_un = randn(rng, n)
            x_dense_lu = accelerate_factorize(A_un; sym = :unsymmetric) \ b_un
            x_dense_lu_ref = lu(A_un) \ b_un
            @test isapprox(x_dense_lu, x_dense_lu_ref; atol = 1.0e-12)
            record!("dense unsymmetric, accelerate_factorize(:lu)", relres(A_un, x_dense_lu, b_un))

            x_dense_auto = accelerate_factorize(A_spd) \ b
            record!("dense :auto on SPD input", relres(A_spd, x_dense_auto, b))

            # dense rectangular QR (overdetermined least squares): the fitted residual is
            # not near zero by construction, so accuracy is judged by how closely the two
            # solvers agree with each other, not by relres against `worst[]`.
            m, k = 100, 40
            A_rect = randn(rng, m, k)
            b_rect = randn(rng, m)
            x_acc_qr = accelerate_factorize(A_rect; kind = :qr) \ b_rect
            x_ref_qr = qr(A_rect) \ b_rect
            soldiff = norm(x_acc_qr - x_ref_qr) / norm(x_ref_qr)
            @test soldiff < 1.0e-9

            # dense symmetric indefinite has no dense counterpart; :ldlt/:symmetric must
            # point callers at `bunchkaufman` rather than silently running `:lu`.
            @test_throws ArgumentError accelerate_factorize(A_spd; kind = :ldlt)
            @test_throws ArgumentError accelerate_factorize(A_spd; sym = :symmetric)

            println(
                "S5.3 accuracy audit (#142): worst relative residual = ", worst[],
                ", case = \"", worst_case[], "\"; dense QR least-squares solution diff = ",
                soldiff
            )
        end
    end
end

end # module
