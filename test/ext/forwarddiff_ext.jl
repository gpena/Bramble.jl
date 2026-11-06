module TestForwardDiffExt

using Test
using Bramble
using Bramble: sparspak_factorize
using LinearAlgebra
using SparseArrays
using Sparspak
using SuiteSparse
using MUMPS
using DifferentiationInterface: DifferentiationInterface, AutoForwardDiff
using ForwardDiff: ForwardDiff
using ..TestUtils: alloc_test

if Sys.isapple()
    using AppleAccelerate
end

# A stand-in for a native factor: its finalizer counts the releases.
mutable struct ReleaseProbe end
const RELEASED = Ref(0)
function release_probe()
    p = ReleaseProbe()
    finalizer(_ -> RELEASED[] += 1, p)
    return p
end

# The `ForwardDiff.Dual` route of `pde_solve` (`BrambleForwardDiffExt`): the `Float64`
# values of the matrix are factored once, by the route the `Float64` solve takes, and each
# partial is one solve against that factor. The oracles are the direct Sparspak factor of
# the `Dual` matrix (the route `pde_solve` took before), the dense solve, and a central
# difference of a `Float64` derivative, which uses no `Dual` arithmetic at all.
@testset "ForwardDiff extension" begin
    m = 30
    A0 = spdiagm(0 => fill(4.0, m), 1 => fill(-1.0, m - 1), -1 => fill(-1.2, m - 1))
    D = ForwardDiff.Dual{:t}
    Ad = SparseMatrixCSC(A0.m, A0.n, A0.colptr, A0.rowval, D.(A0.nzval, 0.5 .* A0.nzval))
    b = ones(m)
    x0 = A0 \ b
    p1 = A0 \ (-(0.5 .* A0) * x0)

    @testset "every solver, Dual matrix" begin
        solvers = [:default, :suitesparse, :spqr, :sparspak]
        Base.get_extension(Bramble, :BrambleAppleAccelerateExt) !== nothing &&
            push!(solvers, :accelerate)
        Base.get_extension(Bramble, :BrambleMUMPSExt) !== nothing && push!(solvers, :mumps)
        for solver in solvers
            x = pde_solve(Ad, b; solver = solver)
            @test ForwardDiff.value.(x) ≈ x0 rtol = 1.0e-12
            @test ForwardDiff.partials.(x, 1) ≈ p1 rtol = 1.0e-12
        end
    end

    @testset "matches the direct Sparspak factor" begin
        x = pde_solve(Ad, b; solver = :sparspak)
        old = sparspak_factorize(Ad) \ b
        @test ForwardDiff.value.(x) ≈ ForwardDiff.value.(old) rtol = 1.0e-12
        @test ForwardDiff.partials.(x, 1) ≈ ForwardDiff.partials.(old, 1) rtol = 1.0e-10
    end

    # A symmetric positive definite matrix is the case `:default` sends to Accelerate on
    # macOS, and to `factorize`'s Cholesky elsewhere.
    @testset "symmetric matrix under :default" begin
        S0 = spdiagm(0 => fill(4.0, m), 1 => fill(-1.0, m - 1), -1 => fill(-1.0, m - 1))
        Sd = SparseMatrixCSC(S0.m, S0.n, S0.colptr, S0.rowval, D.(S0.nzval, 0.5 .* S0.nzval))
        xs0 = S0 \ b
        x = pde_solve(Sd, b)
        @test ForwardDiff.value.(x) ≈ xs0 rtol = 1.0e-12
        @test ForwardDiff.partials.(x, 1) ≈ S0 \ (-(0.5 .* S0) * xs0) rtol = 1.0e-12
    end

    @testset "real matrix with a Dual load" begin
        bd = D.(b, collect(1.0:m))
        for solver in (:default, :suitesparse, :sparspak)
            xb = pde_solve(A0, bd; solver = solver)
            @test ForwardDiff.value.(xb) ≈ x0 rtol = 1.0e-12
            @test ForwardDiff.partials.(xb, 1) ≈ A0 \ collect(1.0:m) rtol = 1.0e-12
        end
    end

    # The second derivative of a nested tag: the outer tag `s` perturbs the matrix by 0.1 A0,
    # the inner `t` scales it by 1 + t / 2. The mixed partial comes from one factor of the
    # value, and the Sparspak factor of the nested matrix is the oracle.
    @testset "nested tags recurse over one factor" begin
        E = ForwardDiff.Dual{:s}
        An = SparseMatrixCSC(A0.m, A0.n, A0.colptr, A0.rowval,
            E.(Ad.nzval, D.(0.1 .* A0.nzval, 0.0)))
        xn = pde_solve(An, b)
        xr = sparspak_factorize(An) \ b
        mixed = ForwardDiff.partials.(ForwardDiff.partials.(xn, 1), 1)
        @test mixed ≈ ForwardDiff.partials.(ForwardDiff.partials.(xr, 1), 1) rtol = 1.0e-10
        @test ForwardDiff.value.(ForwardDiff.partials.(xn, 1)) ≈
              ForwardDiff.value.(ForwardDiff.partials.(xr, 1)) rtol = 1.0e-10

        # The first derivative in `s` in closed form, `-A(t)⁻¹ 0.1 A0 x(t)`, with no `Dual`
        # arithmetic: its central difference in `t` is the mixed partial.
        ds(τ) = (A0 .* (1 + 0.5τ)) \ (-(0.1 .* A0) * ((A0 .* (1 + 0.5τ)) \ b))
        h = 1.0e-5
        @test ForwardDiff.value.(mixed) ≈ (ds(h) - ds(-h)) / (2h) rtol = 1.0e-6
    end

    # The factor is shared: a Dual with N partials costs N solves and one factorization, not
    # N factorizations. A factorization of this matrix allocates about 1.8 MB and a partial
    # about 0.5 MB, so the bound of half a factorization per partial fails a refactor on
    # every partial.
    @testset "one factorization for every partial" begin
        n = 3000
        T0 = spdiagm(0 => fill(4.0, n), 1 => fill(-1.0, n - 1), -1 => fill(-1.0, n - 1))
        dual(N) = SparseMatrixCSC(T0.m, T0.n, T0.colptr, T0.rowval,
            [ForwardDiff.Dual{Nothing, Float64, N}(v, ForwardDiff.Partials(ntuple(
                 k -> k * v, N))) for v in T0.nzval])
        rhs = ones(n)
        solve(A) = pde_solve(A, rhs; solver = :sparspak)
        per_partial = (alloc_test(solve, dual(8)) - alloc_test(solve, dual(1))) / 7
        factor = alloc_test(sparspak_factorize, T0)
        @test per_partial < factor / 2
    end

    # A rank-deficient `A` under `:spqr` is a least-squares solve whose value is not smooth in
    # the parameter: for `[1 0; t 0]` and `F = [0, 1]` the solution is `x₁ = t / (1 + t²)`,
    # with derivative 1 at `t = 0`, and the forward formula gives 0. A `Dual` matrix of rank
    # below its size is refused, naming the rank. A real rank-deficient matrix with a `Dual`
    # load is linear in the load, so it stays allowed.
    @testset "rank-deficient spqr refuses" begin
        At = SparseMatrixCSC(2, 2, [1, 3, 3], [1, 2], [D(1.0, 0.0), D(0.0, 1.0)])
        err = try
            pde_solve(At, [0.0, 1.0]; solver = :spqr)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("rank 1", sprint(showerror, err))
        Ar = SparseMatrixCSC(2, 2, [1, 2, 2], [1], [1.0])
        @test pde_solve(Ar, D.([0.0, 1.0], [1.0, 0.0]); solver = :spqr) isa Vector
    end

    # A `Float32` matrix with a `Float64`-partials load: the primal factor is built at the
    # common scalar type, `Float64`, so each partial is a solve against it. A factor left at
    # `Float32` would be refactored at `Float64` for every partial. The oracle is the dense
    # solve of the same matrix in `Float64`.
    @testset "Float32 matrix, Float64 Dual load" begin
        A32 = SparseMatrixCSC{Float32}(A0)
        A64 = Float64.(A32)
        x = pde_solve(A32, D.(b, collect(1.0:m)); solver = :sparspak)
        @test ForwardDiff.value.(x) ≈ A64 \ b rtol = 1.0e-12
        @test ForwardDiff.partials.(x, 1) ≈ A64 \ collect(1.0:m) rtol = 1.0e-12

        n = 3000
        T0 = spdiagm(0 => fill(4.0f0, n), 1 => fill(-1.0f0, n - 1), -1 => fill(-1.0f0, n - 1))
        dual(N) = [ForwardDiff.Dual{Nothing, Float64, N}(1.0, ForwardDiff.Partials(
                       ntuple(k -> float(k), N))) for _ in 1:n]
        solve(rhs) = pde_solve(T0, rhs; solver = :sparspak)
        per_partial = (alloc_test(solve, dual(8)) - alloc_test(solve, dual(1))) / 7
        factor = alloc_test(sparspak_factorize, Float64.(T0))
        @test per_partial < factor / 2
    end

    # The primal factor is released once the solve is done, as `_mumps_solve` releases its
    # native factor, and also when the solve throws: otherwise native MUMPS factors outlive
    # the gradient call. MUMPS is in the test environment, but a finalizer on a probe object
    # tests the release without a MUMPS factor: `_run_and_finalize` is the helper that
    # `BrambleForwardDiffExt` runs its work through.
    @testset "primal factor released after the solve" begin
        ext = Base.get_extension(Bramble, :BrambleForwardDiffExt)
        before = RELEASED[]
        @test ext._run_and_finalize(_ -> 42, release_probe()) == 42
        @test RELEASED[] == before + 1
        @test_throws ErrorException ext._run_and_finalize(
            _ -> error("solve failed"), release_probe())
        @test RELEASED[] == before + 2
    end

    @testset "the Dual route refuses bad input" begin
        rect = SparseMatrixCSC(3, 2, [1, 3, 5], [1, 2, 2, 3],
            D.([1.0, 0.5, 2.0, 1.0], [1.0, 0.0, 0.0, 1.0]))
        @test_throws ArgumentError pde_solve(rect, ones(3); solver = :spqr)
        @test_throws ArgumentError pde_solve(Ad, b; solver = :bogus)
        @test_throws ArgumentError pde_solve(Ad, b; solver = :sparspak, bogus = 1)
        @test_throws ArgumentError pde_solve(Ad, b; bogus = 1)
        # A singular Dual matrix reaches the primal factorization, which throws.
        sing = SparseMatrixCSC(2, 2, [1, 3, 5], [1, 2, 1, 2],
            D.([1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]))
        @test_throws SingularException pde_solve(sing, ones(2); solver = :sparspak)
    end
end

end # module
