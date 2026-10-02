module SolversSparseSolversTests

using Test
using Bramble
using LinearAlgebra
using SparseArrays
using Random
using ..TestUtils: WITH_SLOW_TESTS
using Bramble: refactor!, sparse_factorize, sparse_refactor!, sparspak_factorize, sparspak_solve,
               sparspak_refactor!, mumps_factorize, mumps_solve, mumps_refactor!, suitesparse_factorize,
               suitesparse_solve, suitesparse_refactor!, suitesparse_qr_factorize, suitesparse_qr_solve,
               SuiteSparseFactorization, AccelerateFactorization, MUMPSFactorization,
               SparspakFactorization, _default_wants_accelerate

# Core fallback and validation.
@testset "Sparse solver interface" begin
    I1 = interval(0.0, 1.0)
    Ω1 = mesh(domain(I1, :boundary => boundary_symbols(I1)), 10, true)
    W1 = gridspace(Ω1)
    a1 = form(W1, W1, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    l1 = form(W1, v -> innerₕ(Rₕ(W1, x -> sin(π * x)), v))
    A, F = assemble(a1, l1; dirichlet = :boundary => x -> 0.0, symmetrize = true)

    # 1. Alias equivalence
    @test sparse_refactor! === refactor!

    # 2. Default pde_solve
    u_default = pde_solve(A, F)
    @test isapprox(u_default, A \ F; atol = 1e-12)

    u_default_sym = pde_solve(A, F; solver = :default)
    @test isapprox(u_default_sym, A \ F; atol = 1e-12)

    # 3. pde_solve on Factorization object
    fact_lu = lu(A)
    @test isapprox(pde_solve(fact_lu, F), u_default; atol = 1e-12)

    # 4. Unknown solver error
    @test_throws ArgumentError sparse_factorize(A; solver = :nonexistent_solver)
    @test_throws ArgumentError sparse_factorize(a1; solver = :nonexistent_solver)
    @test_throws ArgumentError pde_solve(A, F; solver = :nonexistent_solver)

    # 5. Type safety: refactor! only accepts SparseMatrixCSC (or BilinearForm)
    @test_throws ArgumentError refactor!(fact_lu, Matrix(A))
    @test_throws ArgumentError sparse_refactor!(fact_lu, Matrix(A))
    @test_throws ArgumentError refactor!(fact_lu, [1.0, 2.0])

    # 6. Type safety: sparse_factorize only accepts SparseMatrixCSC
    @test_throws MethodError sparse_factorize(Matrix(A))
end

# The solver wrappers forward to extension methods that narrow on the same signature; when
# the weak dependency is not loaded the `::Any` fallback must raise its message. The suite
# process may have loaded the extensions, so every fallback runs in a child process on the
# root project (which loads none) built from `Base.julia_cmd()`, so that it keeps the parent's
# coverage flag. The oracle is the exact error message the fallback owes the user.
const FALLBACK_CHILD = raw"""
using Bramble, SparseArrays, Random
using Bramble: refactor!, sparse_factorize, sparspak_factorize, sparspak_solve, sparspak_refactor!,
               mumps_factorize, mumps_solve, mumps_refactor!, suitesparse_factorize,
               suitesparse_solve, suitesparse_refactor!, SuiteSparseFactorization,
               AccelerateFactorization, MUMPSFactorization, SparspakFactorization
Random.seed!(7)
I1 = interval(0.0, 1.0)
W = gridspace(mesh(domain(I1, :boundary => boundary_symbols(I1)), 10, true))
a = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
l = form(W, v -> innerₕ(Rₕ(W, x -> sin(π * x)), v))
bc = :boundary => x -> 0.0
A, F = assemble(a, l; dirichlet = bc, symmetrize = true)
struct FakeSS <: SuiteSparseFactorization{Float64} end
struct FakeAcc <: AccelerateFactorization{Float64} end
struct FakeMUMPS <: MUMPSFactorization{Float64} end
struct FakeSparspak <: SparspakFactorization{Float64} end
function probe(label, f)
    msg = try
        f()
        "NOERROR"
    catch e
        e isa ErrorException ? e.msg : "OTHER " * sprint(showerror, e)
    end
    println(label, "\t", replace(msg, '\n' => ' '))
end
probe("amg_matrix", () -> amg_preconditioner(A))
probe("amg_form", () -> amg_preconditioner(a; dirichlet = bc))
probe("amg_operator", () -> Bramble._amg_operator(A))
probe("ilu_matrix", () -> ilu_preconditioner(A))
probe("ilu_form", () -> ilu_preconditioner(a; dirichlet = bc))
probe("ilu_operator", () -> Bramble._ilu_operator(A))
probe("sparspak_factorize", () -> sparspak_factorize(A))
probe("sparspak_factorize_form", () -> sparspak_factorize(a; dirichlet = bc))
probe("sparspak_solve", () -> sparspak_solve(A, F))
probe("sparspak_solve_form", () -> sparspak_solve(a, l; dirichlet = bc))
probe("sparspak_refactor", () -> sparspak_refactor!(FakeSparspak(), A))
probe("refactor_sparspak", () -> refactor!(FakeSparspak(), A))
probe("mumps_factorize", () -> mumps_factorize(A))
probe("mumps_factorize_form", () -> mumps_factorize(a; dirichlet = bc))
probe("mumps_solve", () -> mumps_solve(A, F))
probe("mumps_solve_form", () -> mumps_solve(a, l; dirichlet = bc))
probe("mumps_refactor", () -> mumps_refactor!(FakeMUMPS(), A))
probe("refactor_mumps", () -> refactor!(FakeMUMPS(), A))
probe("suitesparse_factorize", () -> suitesparse_factorize(A))
probe("suitesparse_factorize_form", () -> suitesparse_factorize(a; dirichlet = bc))
probe("suitesparse_solve", () -> suitesparse_solve(A, F))
probe("suitesparse_solve_form", () -> suitesparse_solve(a, l; dirichlet = bc, symmetrize = true))
probe("suitesparse_refactor", () -> suitesparse_refactor!(FakeSS(), A))
probe("refactor_suitesparse", () -> refactor!(FakeSS(), A))
probe("accelerate_factorize", () -> sparse_factorize(A; solver = :accelerate))
probe("accelerate_solve", () -> pde_solve(A, F; solver = :accelerate))
probe("refactor_accelerate", () -> refactor!(FakeAcc(), A))
probe("acc_factorize_direct", () -> Bramble.accelerate_factorize(A))
probe("acc_factorize_form", () -> Bramble.accelerate_factorize(a; dirichlet = bc))
probe("acc_solve_direct", () -> Bramble.accelerate_solve(A, F))
probe("acc_solve_form", () -> Bramble.accelerate_solve(a, l; dirichlet = bc, symmetrize = true))
probe("acc_refactor_direct", () -> Bramble.accelerate_refactor!(FakeAcc(), A))
probe("pde_suitesparse", () -> pde_solve(A, F; solver = :suitesparse))
probe("pde_mumps", () -> pde_solve(A, F; solver = :mumps))
probe("pde_sparspak", () -> pde_solve(A, F; solver = :sparspak))
probe("sparse_factorize_mumps", () -> sparse_factorize(A; solver = :mumps))
probe("sparse_factorize_sparspak", () -> sparse_factorize(A; solver = :sparspak))
probe("sparse_factorize_suitesparse", () -> sparse_factorize(A; solver = :suitesparse))
"""

function _fallback_messages()
    root = pkgdir(Bramble)
    cmd = `$(Base.julia_cmd()) --project=$root --startup-file=no --threads=1 -e $FALLBACK_CHILD`
    out = read(setenv(cmd, "JULIA_PKG_PRECOMPILE_AUTO" => "0"), String)
    return Dict(Pair(split(l, '\t'; limit = 2)...) for l in split(out, '\n'; keepempty = false))
end

@testset "Fallbacks without the weak dependency" begin
    msg = _fallback_messages()
    needs(pkg, name) = "$name requires $pkg.jl. Add `using $pkg` before calling this function."

    @test msg["amg_matrix"] == needs("AlgebraicMultigrid", "amg_preconditioner")
    @test msg["amg_form"] == needs("AlgebraicMultigrid", "amg_preconditioner")
    @test msg["amg_operator"] == needs("AlgebraicMultigrid", "preconditioner = :amg")
    @test msg["ilu_matrix"] == needs("ILUZero", "ilu_preconditioner")
    @test msg["ilu_form"] == needs("ILUZero", "ilu_preconditioner")
    @test msg["ilu_operator"] == needs("ILUZero", "preconditioner = :ilu0")

    for k in ("sparspak_factorize", "sparspak_factorize_form", "sparse_factorize_sparspak")
        @test msg[k] == needs("Sparspak", "sparspak_factorize")
    end
    for k in ("sparspak_solve", "sparspak_solve_form", "pde_sparspak")
        @test msg[k] == needs("Sparspak", "sparspak_solve")
    end
    for k in ("sparspak_refactor", "refactor_sparspak")
        @test msg[k] == needs("Sparspak", "sparspak_refactor!")
    end

    for k in ("mumps_factorize", "mumps_factorize_form", "sparse_factorize_mumps")
        @test msg[k] == needs("MUMPS", "mumps_factorize")
    end
    for k in ("mumps_solve", "mumps_solve_form", "pde_mumps")
        @test msg[k] == needs("MUMPS", "mumps_solve")
    end
    for k in ("mumps_refactor", "refactor_mumps")
        @test msg[k] == needs("MUMPS", "mumps_refactor!")
    end

    for k in ("suitesparse_factorize", "suitesparse_factorize_form", "sparse_factorize_suitesparse")
        @test msg[k] == needs("SuiteSparse", "suitesparse_factorize")
    end
    for k in ("suitesparse_solve", "suitesparse_solve_form", "pde_suitesparse")
        @test msg[k] == needs("SuiteSparse", "suitesparse_solve")
    end
    for k in ("suitesparse_refactor", "refactor_suitesparse")
        @test msg[k] == needs("SuiteSparse", "suitesparse_refactor!")
    end

    # Off macOS the wrapper rejects the platform first; on macOS the fallback speaks.
    for (k, name) in (
        ("accelerate_factorize", "accelerate_factorize"), ("accelerate_solve", "accelerate_solve"),
        ("refactor_accelerate", "accelerate_refactor!"),
        ("acc_factorize_direct", "accelerate_factorize"),
        ("acc_factorize_form", "accelerate_factorize"),
        ("acc_solve_direct", "accelerate_solve"), ("acc_solve_form", "accelerate_solve"),
        ("acc_refactor_direct", "accelerate_refactor!"))
        if Sys.isapple()
            @test msg[k] == needs("AppleAccelerate", name)
        else
            @test msg[k] == "OTHER ArgumentError: AppleAccelerate is only supported on macOS (darwin)."
        end
    end
end

@testset "SPQR and pde_solve, non-uniform mesh" begin
    Random.seed!(11)
    I1 = interval(0.0, 1.0)
    W = gridspace(mesh(domain(I1, :boundary => boundary_symbols(I1)), 12, true))
    a = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    l = form(W, v -> innerₕ(Rₕ(W, x -> sin(π * x)), v))
    bc = :boundary => x -> 0.0
    A, F = assemble(a, l; dirichlet = bc)
    u_ref = Matrix(A) \ F    # dense LU: not the sparse code under test

    @test suitesparse_qr_factorize(A) \ F ≈ u_ref rtol = 1e-10
    @test suitesparse_qr_factorize(a; dirichlet = bc) \ F ≈ u_ref rtol = 1e-10
    @test suitesparse_qr_solve(A, F) ≈ u_ref rtol = 1e-10
    @test collect(suitesparse_qr_solve(a, l; dirichlet = bc)) ≈ u_ref rtol = 1e-10
    @test pde_solve(A, F; solver = :spqr) ≈ u_ref rtol = 1e-10

    # A dense matrix is not a sparse system: no method.
    @test_throws MethodError pde_solve(Matrix(A), F)
end

@testset "pde_solve uses Accelerate when loaded" begin
    if Sys.isapple()
        # Slow group only: the child loads the whole test environment (about 90 s).
        if WITH_SLOW_TESTS
            # A child on the test project loads AppleAccelerate, which this process may not have.
            code = raw"""
            using Bramble, AppleAccelerate, SparseArrays, Random
            Random.seed!(5)
            I1 = interval(0.0, 1.0)
            W = gridspace(mesh(domain(I1, :boundary => boundary_symbols(I1)), 12, true))
            a = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
            l = form(W, v -> innerₕ(Rₕ(W, x -> sin(π * x)), v))
            A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0, symmetrize = true)
            println(join(repr.(pde_solve(A, F)), ' '))
            println(join(repr.(Matrix(A) \ F), ' '))
            """
            cmd = `$(Base.julia_cmd()) --project=$(dirname(something(Base.active_project()))) --startup-file=no --threads=1 -e $code`
            lines = split(read(cmd, String), '\n'; keepempty = false)
            u_acc = parse.(Float64, split(lines[1]))
            u_ref = parse.(Float64, split(lines[2]))    # dense LU: not the Accelerate path
            @test u_acc ≈ u_ref rtol = 1e-10
        end
    else
        @test !Sys.isapple()    # the routing is macOS only; Linux and Windows fall through to `\`
    end
end

# The dense method is plain LinearAlgebra under the sparse methods' vocabulary, on every OS.
# Oracle: each factorization solves a nonsymmetric or SPD system to the dense `\` answer,
# and the two rejected options raise their exact messages.
@testset "accelerate_factorize, dense" begin
    Random.seed!(3)
    B = rand(5, 5)
    S = B' * B + 5I    # SPD
    N = S + triu(B, 1)    # nonsymmetric
    b = rand(5)
    acc = Bramble.accelerate_factorize
    @test acc(S; sym = :spd) isa Cholesky
    @test acc(S; kind = :cholesky) \ b ≈ S \ b rtol = 1e-12
    @test acc(N; sym = :unsymmetric) isa LU
    @test acc(N; kind = :lu) \ b ≈ N \ b rtol = 1e-12
    @test acc(N; kind = :qr) \ b ≈ N \ b rtol = 1e-12
    @test acc(S) isa Cholesky
    @test acc(N) isa LU
    @test acc(N) \ b ≈ N \ b rtol = 1e-12
    msg_ldlt = "accelerate_factorize does not wrap dense symmetric indefinite factorization; " *
               "call LinearAlgebra.bunchkaufman(A) directly."
    @test_throws ArgumentError(msg_ldlt) acc(S; sym = :symmetric)
    @test_throws ArgumentError(msg_ldlt) acc(S; kind = :ldlt)
    @test_throws ArgumentError(
        "Unknown factorization option for accelerate_factorize: sym=auto, kind=bogus."
    ) acc(S; kind = :bogus)
end

@testset "_default_wants_accelerate" begin
    S = sparse([2.0 -1.0; -1.0 2.0])
    N = sparse([2.0 -1.0; 0.0 2.0])
    for s in (:spd, :definite, 1, :symmetric, 2)
        @test _default_wants_accelerate(N, s)
    end
    @test _default_wants_accelerate(S, :auto)
    @test !_default_wants_accelerate(N, :auto)
    @test !_default_wants_accelerate(S, :unsymmetric)
    @test !_default_wants_accelerate(S, 0)
end

end # module
