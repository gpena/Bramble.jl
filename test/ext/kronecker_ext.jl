module TestKroneckerExt

using Test
using Bramble
using Bramble: is_separable, kronecker_operator, KroneckerLinearOperator
using Kronecker: kronecker
using LinearAlgebra: LinearAlgebra, mul!, issymmetric, ldiv!, lu, norm
using SparseArrays: SparseMatrixCSC
using LinearSolve: LinearProblem, solve, KrylovJL_GMRES
using Random
# `CpuPolyester` meshes below; test/ext/polyester_ext.jl, next in the ext group, loads it too.
using Polyester
using Bramble: CpuPolyester, Serial, execution_policy

# `Kronecker.jl` interop and fast diagonalisation for a separable `BilinearForm`
# (gpena/Bramble.jl#259), layered on `KroneckerLinearOperator`
# (gpena/Bramble.jl#162, test/form/kronecker.jl). `fdm_solve` has no forward stub in
# `src/Bramble.jl` yet (see `ext/BrambleKroneckerExt.jl`'s module docstring), so it is
# reached the same way any other not-yet-exported extension function would be: off the
# loaded extension module itself.
const KronExt = Base.get_extension(Bramble, :BrambleKroneckerExt)
@assert KronExt !== nothing "BrambleKroneckerExt did not load -- is Kronecker.jl a test dependency?"
# `fdm_solve` is Bramble's own binding (`function fdm_solve end` in `src/Bramble.jl`), and
# this extension adds methods to it, so the exported spelling is the one to test: reaching
# into the extension module would pass even if the methods had attached to a function of
# the extension's own instead, which is precisely the failure this asserts against.
@assert !isempty(methods(Bramble.fdm_solve)) "fdm_solve has no methods -- are the extension's definitions dot-qualified as `Bramble.fdm_solve`?"

const KRON_EXT_SEED = 20260919

using Bramble: D₋ₓ, D₋ᵧ, Mₓ, inner₊ₓ, inner₊ᵧ

# Uniform, then moved to `t^(1 + d/4)` along axis `d`, so no two axes share their nodes.
function graded_space(n::NTuple{D, Int}) where {D}
    Ω = mesh(domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D))), n, ntuple(_ -> false, D))
    Bramble.change_points!(Ω, ntuple(d -> range(0.0, 1.0; length = n[d]) .^ (1 + 0.25d), D))
    return gridspace(Ω)
end

# A vector indexed `0:n-1`, for the factorisation's refusal of a non-1-based one.
struct ZeroBasedVector <: AbstractVector{Float64}
    p::Vector{Float64}
end
Base.size(z::ZeroBasedVector) = size(z.p)
Base.axes(z::ZeroBasedVector) = (Base.IdentityUnitRange(0:(length(z.p) - 1)),)
Base.getindex(z::ZeroBasedVector, i::Int) = z.p[i + 1]
Base.setindex!(z::ZeroBasedVector, v, i::Int) = (z.p[i + 1] = v)

@testset "Kronecker extension" begin
    @testset "Kronecker.jl object equals CSC" begin
        Random.seed!(KRON_EXT_SEED)
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (13, 11), (false, false))
        W2 = gridspace(Ω2)

        Random.seed!(KRON_EXT_SEED + 1)
        Ω3 = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (8, 7, 6), (false, false, false)
        )
        W3 = gridspace(Ω3)

        for Wₕ in (W2, W3)
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + 2.5 * inner₊(∇ₕ(u), ∇ₕ(v)))
            K = kronecker_operator(a)
            Aref = SparseMatrixCSC(K)

            Kjl = kronecker(K)
            @test collect(Kjl) ≈ Aref
            @test Matrix(Kjl) ≈ Matrix(Aref)

            # `mul!` through the Kronecker.jl object agrees with the operator's own `mul!`.
            n = ndofs(Wₕ)
            x = rand(n)
            yref = similar(x)
            mul!(yref, K, x)
            y = Kjl * x
            @test isapprox(y, yref; rtol = 1e-10, atol = 1e-10)
        end
    end

    # fdm_solve against sparse backslash on 2D 25x19 and 3D 11x9x8 meshes, homogeneous Dirichlet.
    @testset "fdm_solve vs \\, zero Dirichlet" begin
        Random.seed!(KRON_EXT_SEED + 4)
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (25, 19), (false, false))
        W2 = gridspace(Ω2)

        Random.seed!(KRON_EXT_SEED + 5)
        Ω3 = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (11, 9, 8), (false, false, false)
        )
        W3 = gridspace(Ω3)

        for (Wₕ, tag) in ((W2, "2D"), (W3, "3D"))
            @testset "$tag" begin
                a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
                A = assemble(a; dirichlet = :boundary)
                n = ndofs(Wₕ)
                dims = ndofs(Wₕ, Tuple)

                F = rand(n)
                Farr = reshape(F, dims)
                D = length(dims)
                for d in 1:D
                    idx_first = ntuple(k -> k == d ? 1 : Colon(), D)
                    idx_last = ntuple(k -> k == d ? dims[k] : Colon(), D)
                    Farr[idx_first...] .= 0.0
                    Farr[idx_last...] .= 0.0
                end
                F = vec(Farr)

                xref = A \ F
                x = fdm_solve(a, F; dirichlet = :boundary)
                @test isapprox(x, xref; rtol = 1e-9)
            end
        end
    end

    # Laplacian-like forms beyond the classic one (#427), on graded meshes whose axes carry
    # different nodes: each solves to the sparse direct solve, from the form and from its
    # operator, with and without homogeneous Dirichlet.
    @testset "fdm_solve: Laplacian-like forms" begin
        for n in ((9, 7), (6, 5, 7))
            Wₕ = graded_space(n)
            D = length(n)
            fx = Rₕ(Wₕ, x -> 1 + x[1])
            c = Ref(2.5)
            lap(op) = (u, v) -> innerₕ(u, v) +
                                sum(innerₕ(op(u, Val(d)), op(v, Val(d))) for d in 1:D)
            accepted = [
                "Ref coefficient" => (u, v) -> innerₕ(u, v) + c * inner₊(∇ₕ(u), ∇ₕ(v)),
                "forward" => lap(Bramble.D₊),
                "averaged mass" => (u, v) -> innerₕ(Mₓ(u), Mₓ(v)) +
                                             inner₊(∇ₕ(u), ∇ₕ(v)),
                "x-coefficient" => (u, v) -> innerₕ(fx * u, v) +
                                             inner₊(∇ₕ(u), ∇ₕ(v)),
                "Robin face" => (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) +
                                          inner_Γ(u, v; markers = (:xmin,)),
                # Indefinite but nonsingular: the singularity test must not refuse it.
                "negative mass" => (u, v) -> -1.0 * innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))
            ]
            for (name, f) in accepted, dir in (nothing, :boundary)

                a = form(Wₕ, Wₕ, f)
                c[] = 2.5
                K = kronecker_operator(a)
                c[] = 4.0  # read when fdm_solve is called, not when `a` or `K` was built
                A = assemble(a; dirichlet = dir)
                F = rand(MersenneTwister(length(name)), size(A, 1))
                dir === :boundary && (F[Bramble._combined_mask(mesh(Wₕ), (:boundary,))] .= 0)
                xref = A \ F
                @test isapprox(fdm_solve(a, F; dirichlet = dir), xref; rtol = 1e-9)
                dir === nothing && @test isapprox(fdm_solve(K, F), xref; rtol = 1e-9)
            end
        end
    end

    # `_fdm_factorize` once, then `ldiv!` per right-hand side (the preconditioner's use):
    # each matches the sparse direct solve and allocates nothing after a warm-up, on graded
    # 2D and 3D meshes (every mode-product branch: a first axis, a middle one, a last one).
    @testset "fdm factorisation: ldiv! vs \\, 0 bytes" begin
        ldiv_bytes(x, f, F) = (ldiv!(x, f, F); @allocated ldiv!(x, f, F))
        for n in ((13, 9), (7, 6, 8)), dir in (nothing, :boundary)

            Wₕ = graded_space(n)
            fx = Rₕ(Wₕ, x -> 1 + x[1])
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(fx * u, v) + 2.5 * inner₊(∇ₕ(u), ∇ₕ(v)))
            A = assemble(a; dirichlet = dir)
            f = KronExt._fdm_factorize(kronecker_operator(a), dir)
            x = fill(NaN, size(A, 1))
            for seed in 1:2
                F = rand(MersenneTwister(KRON_EXT_SEED + seed), size(A, 1))
                dir === :boundary && (F[Bramble._combined_mask(mesh(Wₕ), (:boundary,))] .= 0)
                @test ldiv_bytes(x, f, F) == 0
                @test isapprox(x, A \ F; rtol = 1e-10)
                y = copy(F)
                @test ldiv!(y, f, y) == x  # `x` may alias `F`
            end
            @test_throws DimensionMismatch ldiv!(x, f, zeros(length(x) + 1))
            # `f.interior` holds 1-based positions: a 0-based vector is refused, not misread.
            @test_throws ArgumentError ldiv!(x, f, ZeroBasedVector(zeros(length(x))))
            @test_throws ArgumentError ldiv!(ZeroBasedVector(copy(x)), f, zeros(length(x)))
        end
    end

    # Non-symmetric (advection) forms take the generalised Schur route: each matches the
    # sparse direct solve on graded 2D and 3D meshes, with and without homogeneous Dirichlet,
    # from `fdm_solve(a, F)`, `fdm_solve(K, F)` and a reused factorisation whose `ldiv!`
    # allocates nothing. Distinct per-axis scales put a scalar multiple of a mass factor into
    # a term (`kronecker_operator` folds the scale into another axis), which the
    # classification must read as the mass. A symmetric form keeps fast diagonalisation.
    @testset "fdm_solve: advection by Schur form" begin
        ldiv_bytes(x, f, F) = (ldiv!(x, f, F); @allocated ldiv!(x, f, F))
        for n in ((13, 9), (7, 6, 8))
            Wₕ = graded_space(n)
            D = length(n)
            fx = Rₕ(Wₕ, x -> 1 + x[1])
            Dm(u, d) = Bramble.D₋(u, Val(d))
            L(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))
            forms = [
                "axis-1 advection" => (u, v) -> L(u, v) + innerₕ(D₋ₓ(u), v),
                "distinct scales" => (u, v) -> L(u, v) +
                                               sum((d + 0.5) * innerₕ(Dm(u, d), v) for d in 1:D),
                "x-coefficient mass" => (u, v) -> innerₕ(fx * u, v) +
                                                  inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(Dm(u, D), v),
                "dominant advection" => (u, v) -> innerₕ(u, v) +
                                                  1e-3 * inner₊(∇ₕ(u), ∇ₕ(v)) +
                                                  innerₕ(D₋ₓ(u), v)
            ]
            for (name, f) in forms, dir in (nothing, :boundary)

                a = form(Wₕ, Wₕ, f)
                A = assemble(a; dirichlet = dir)
                @test !issymmetric(A)
                K = kronecker_operator(a)
                fact = KronExt._fdm_factorize(K, dir)
                @test fact isa KronExt._SchurFactorization
                x = fill(NaN, size(A, 1))
                for seed in 1:2
                    F = rand(MersenneTwister(KRON_EXT_SEED + seed), size(A, 1))
                    dir === :boundary &&
                        (F[Bramble._combined_mask(mesh(Wₕ), (:boundary,))] .= 0)
                    xref = A \ F
                    @test ldiv_bytes(x, fact, F) == 0
                    @test isapprox(x, xref; rtol = 1e-10)
                    y = copy(F)
                    @test ldiv!(y, fact, y) == x  # `x` may alias `F`
                    @test isapprox(fdm_solve(a, F; dirichlet = dir), xref; rtol = 1e-10)
                    dir === nothing && @test isapprox(fdm_solve(K, F), xref; rtol = 1e-10)
                end
                @test_throws DimensionMismatch ldiv!(x, fact, zeros(length(x) + 1))
                @test_throws ArgumentError ldiv!(x, fact, ZeroBasedVector(zeros(length(x))))
            end
            a = form(Wₕ, Wₕ, L)
            @test KronExt._fdm_factorize(kronecker_operator(a), nothing) isa
                  KronExt._FDMFactorization
        end
    end

    # The Schur route's QZ is OpenBLAS's `zgges3`/`cgges3`, called directly: this file runs
    # after AppleAccelerate is loaded in the ext group, and Accelerate's `zgges3` fails to
    # converge on these strongly non-normal pencils (LAPACKException 63, 127, 15 on the
    # Float64 cases; gpena/Bramble.jl#443). Float64 to 1e-10 of the sparse solve, Float32
    # within 10x of a dense Float32 LU's residual, 0 bytes per `ldiv!`. Without OpenBLAS the
    # active LAPACK's failure is an ArgumentError naming it, never a raw LAPACKException.
    @testset "fdm_solve: Schur route under any LAPACK" begin
        @test all(!=(C_NULL), KronExt._OPENBLAS_GGES3[])
        G(u, v) = inner₊(∇ₕ(u), ∇ₕ(v))
        Dm(u, d) = Bramble.D₋(u, Val(d))
        ldiv_bytes(x, f, F) = (ldiv!(x, f, F); @allocated ldiv!(x, f, F))
        space(T, n) = gridspace(mesh(
            domain(reduce(×, ntuple(_ -> interval(zero(T), one(T)), length(n)))), n,
            ntuple(_ -> true, length(n))))
        advection(W, T, Pe, D) = form(W, W, (u, v) -> T(1 / Pe) * G(u, v) +
                                                      sum(innerₕ(Dm(u, d), v) for d in 1:D))
        function boundary_rhs(W, T, N)
            F = rand(MersenneTwister(KRON_EXT_SEED + 9), T, N)
            F[Bramble._combined_mask(mesh(W), (:boundary,))] .= 0
            return F
        end
        for (T, n, Pe) in ((Float64, (65, 17), 1e6), (Float64, (65, 17), 1e4),
            (Float64, (129, 33), 1e6), (Float64, (17, 17, 17), 1e6),
            (Float32, (65, 17), 1e3), (Float32, (33, 33), 1e3))
            @testset "$T $n Pe = $Pe" begin
                W = space(T, n)
                a = advection(W, T, Pe, length(n))
                A = assemble(a; dirichlet = :boundary)
                F = boundary_rhs(W, T, size(A, 1))
                x = fdm_solve(a, F; dirichlet = :boundary)
                @test eltype(x) == T
                if T == Float64
                    @test norm(x - A \ F) <= 1e-10 * norm(A \ F)
                else
                    r(z) = norm(Float64.(A) * Float64.(z) - F) / norm(F)
                    @test r(x) <= 10 * r(lu(Matrix(A)) \ F) + 10 * eps(T)
                end
                f = KronExt._fdm_factorize(kronecker_operator(a), :boundary)
                @test f isa KronExt._SchurFactorization
                @test ldiv_bytes(zeros(T, length(F)), f, F) == 0
            end
        end
        W = space(Float64, (65, 17))
        a = advection(W, Float64, 1e6, 2)
        F = boundary_rhs(W, Float64, ndofs(W))
        found = KronExt._OPENBLAS_GGES3[]
        # From the libraries' names: inside the suite `string(get_config())` can print only
        # `LBTConfig(...)`, which would send an Accelerate run down the OpenBLAS branch.
        accelerate = any(lib -> occursin("Accelerate", lib.libname),
            LinearAlgebra.BLAS.get_config().loaded_libs)
        try
            KronExt._OPENBLAS_GGES3[] = (C_NULL, C_NULL)
            if accelerate
                @test_throws ArgumentError fdm_solve(a, F; dirichlet = :boundary)
                @test_throws "active LAPACK" fdm_solve(a, F; dirichlet = :boundary)
                @test_throws "Accelerate" fdm_solve(a, F; dirichlet = :boundary)
            else
                A = assemble(a; dirichlet = :boundary)
                @test norm(fdm_solve(a, F; dirichlet = :boundary) - A \ F) <=
                      1e-10 * norm(A \ F)
            end
        finally
            KronExt._OPENBLAS_GGES3[] = found
        end
    end

    # A 3-point axis under `dirichlet = :boundary` has a 1x1 interior, where every factor is
    # a multiple of every other: matching up to a scalar must not merge its stiffness into
    # its mass. A negative multiple of the mass seen first must not become the mass
    # candidate. Both solve to the sparse direct solve.
    @testset "fdm_solve: 3-point axes and signs" begin
        L(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))
        adv(u, v) = L(u, v) + innerₕ(D₋ₓ(u), v)
        scaled(u, v) = adv(u, v) + 0.5 * innerₕ(D₋ᵧ(u), v)
        cases = [
            ((3, 7), :boundary, L), ((3, 7), :boundary, adv), ((3, 7), :boundary, scaled),
            ((5, 3, 4), :boundary, L), ((5, 3, 4), :boundary, adv),
            ((3, 6, 5), :boundary, scaled),
            ((17, 13), :boundary,
                (u, v) -> -2.0 * innerₕ(D₋ₓ(u), v) - 3.0 * innerₕ(D₋ᵧ(u), v) + L(u, v)),
            ((17, 13), nothing,
                (u, v) -> L(u, v) + (-1.0) * innerₕ(D₋ₓ(u), v) + (-0.5) * innerₕ(D₋ᵧ(u), v)),
            ((17, 13), :boundary,
                (u, v) -> -0.1 * innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ₓ(u), v))
        ]
        for (n, dir, f) in cases
            Wₕ = graded_space(n)
            a = form(Wₕ, Wₕ, f)
            A = assemble(a; dirichlet = dir)
            F = rand(MersenneTwister(KRON_EXT_SEED + length(n)), size(A, 1))
            dir === :boundary && (F[Bramble._combined_mask(mesh(Wₕ), (:boundary,))] .= 0)
            @test isapprox(fdm_solve(a, F; dirichlet = dir), A \ F; rtol = 1e-10)
        end
    end

    # Float32 on a graded mesh (points `t^2`, condition about 2e4): the singularity bound
    # (`D eps` times the largest entry, measured 15 eps away here) must not refuse it, on
    # either route, while the pure-Neumann form on the same mesh is still refused.
    @testset "fdm_solve: Float32 graded" begin
        Ω = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (65, 9),
            (false, false))
        Bramble.change_points!(Ω, (Float32.(range(0, 1; length = 65) .^ 2),
            Float32.(range(0, 1; length = 9) .^ 2)))
        Wₕ = gridspace(Ω)
        L(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))
        for f in (L, (u, v) -> L(u, v) + innerₕ(D₋ₓ(u), v))
            a = form(Wₕ, Wₕ, f)
            A = assemble(a; dirichlet = :boundary)
            F = rand(MersenneTwister(KRON_EXT_SEED), Float32, size(A, 1))
            F[Bramble._combined_mask(mesh(Wₕ), (:boundary,))] .= 0
            x = fdm_solve(a, F; dirichlet = :boundary)
            @test eltype(x) === Float32
            @test isapprox(x, Float64.(A) \ Float64.(F); rtol = 1e-3)
        end
        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        G = ones(Float32, ndofs(Wₕ))
        @test_throws ArgumentError fdm_solve(a, G)
        @test_throws "the system is singular" fdm_solve(a, G)
    end

    # A singular form is refused in every precision (gpena/Bramble.jl#443). G1, Neumann plus
    # strong advection, has a zero eigenvalue Float32 arithmetic returns far from zero, so the
    # structural test (a constant kernel on every axis, no mass) refuses it. Uniform-mesh
    # advection without a mass, or pure advection, has no constant kernel once Dirichlet rows
    # go and must still be solved. An ill-conditioned Float32 system the eigenvalue test
    # refuses names Float64 as the way out.
    @testset "fdm_solve: singular in every precision" begin
        # `p === 1` is a uniform mesh; `unif = false` alone would draw random points.
        function space_t(T, n, p)
            Ω = mesh(domain(reduce(×, ntuple(_ -> interval(zero(T), one(T)), length(n)))), n,
                ntuple(_ -> p === 1, length(n)))
            p === 1 && return gridspace(Ω)
            Bramble.change_points!(Ω, ntuple(d -> T.(range(0.0, 1.0; length = n[d]) .^
                                                     (p === :g ? (1 + 0.25d) : p)),
                length(n)))
            return gridspace(Ω)
        end
        G(u, v) = inner₊(∇ₕ(u), ∇ₕ(v))
        for n in ((9, 7), (65, 33), (7, 6, 5))
            Wₕ = space_t(Float32, n, :g)
            adv(u, v) = G(u, v) + sum(1e3 * innerₕ(Bramble.D₊(u, Val(d)), v) for d in 1:length(n))
            F = ones(Float32, ndofs(Wₕ))
            @test_throws ArgumentError fdm_solve(form(Wₕ, Wₕ, adv), F)
            @test_throws "the system is singular" fdm_solve(form(Wₕ, Wₕ, adv), F)
        end
        Dm(u, d) = Bramble.D₋(u, Val(d))
        f1(T, Pe, D) = (u, v) -> T(1 / Pe) * G(u, v) + sum(innerₕ(Dm(u, d), v) for d in 1:D)
        f2(u, v) = innerₕ(D₋ₓ(u), v) + innerₕ(D₋ᵧ(u), v)
        for (T, n, f) in ((Float64, (65, 17), f1(Float64, 1e6, 2)),
            (Float64, (17, 17, 17), f1(Float64, 1e6, 3)), (Float32, (33, 33), f1(Float32, 1e3, 2)),
            (Float64, (33, 17), f2), (Float32, (33, 17), f2))
            Wₕ = space_t(T, n, 1)
            a = form(Wₕ, Wₕ, f)
            A = assemble(a; dirichlet = :boundary)
            F = rand(MersenneTwister(KRON_EXT_SEED), T, size(A, 1))
            F[Bramble._combined_mask(mesh(Wₕ), (:boundary,))] .= 0
            x = fdm_solve(a, F; dirichlet = :boundary)
            @test eltype(x) === T
            if T == Float64
                @test isapprox(x, A \ F; rtol = 1e-10)
            else  # within 10x of a dense Float32 LU's residual
                resid(z) = norm(Float64.(A) * Float64.(z) - F) / norm(F)
                @test resid(x) <= 10 * resid(lu(Matrix(A)) \ F) + 10 * eps(Float32)
            end
        end
        Wₕ = space_t(Float32, (129, 9), 2.0)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + G(u, v) + innerₕ(D₋ₓ(u), v))
        @test_throws r"singular.*assemble in Float64" fdm_solve(a, ones(Float32, ndofs(Wₕ));
            dirichlet = :boundary)
        # No mass, Dirichlet, points `0.5 + 0.5 sign(s)|s|^6`: both boundary rows sum to
        # almost nothing against the fine middle, so in Float32 the structural test fires on
        # a nonsingular system, and the refusal must still name Float64.
        Ω = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (17, 17), (true, true))
        c = Float32.(0.5 .+ 0.5 .* sign.(range(-1, 1; length = 17)) .* abs.(range(-1, 1; length = 17)) .^ 6)
        Bramble.change_points!(Ω, (c, c))
        Wₕ = gridspace(Ω)
        @test_throws r"singular.*assemble in Float64" fdm_solve(form(Wₕ, Wₕ, G),
            zeros(Float32, ndofs(Wₕ)); dirichlet = :boundary)
    end

    # Every other form throws, naming the reason, and is never solved wrongly.
    @testset "fdm_solve: refusals name the reason" begin
        refusal(f) =
            try
                f()
                "no error"
            catch e
                e isa ArgumentError ? sprint(showerror, e) : "wrong error $(typeof(e))"
            end
        Wₕ = graded_space((9, 7))
        fx = Rₕ(Wₕ, x -> 1 + x[1])
        fy = Rₕ(Wₕ, x -> 2 + x[2]^2)
        fxy = Rₕ(Wₕ, x -> (1 + x[1]) * (2 + x[2]^2))
        L(u, v) = inner₊(∇ₕ(u), ∇ₕ(v))
        rint(u) = Bramble.restrict_to(:interior, u)
        refused = [
            "mixed" => ((u, v) -> innerₕ(D₋ₓ(D₋ᵧ(u)), v) + L(u, v),
                r"non-mass factors on two axes"),
            # Advection takes the Schur route, which refuses a singular system the same way.
            "advection, no mass" => ((u, v) -> innerₕ(D₋ₓ(u), v) + L(u, v),
                r"the system is singular"),
            "two-axis coefficient" => ((u, v) -> innerₕ(fx * (fy * u), v) + L(u, v),
                r"non-mass factors on two axes"),
            "one-axis gradient" => ((u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(u, v),
                r"axis 2 has no term of its own"),
            "singular mass" => ((u, v) -> innerₕ(rint(u), v),
                r"mass is not symmetric positive definite"),
            # Pure Neumann: constants are in the kernel, so a solve would return garbage.
            "gradient only" => (L, r"the system is singular"),
            "chain, average, y-stiffness" => ((u, v) -> innerₕ(D₋ₓ(Mₓ(u)), D₋ₓ(Mₓ(v))) +
                       innerₕ(Mₓ(u), Mₓ(v)) +
                       inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(v)),
                r"the system is singular"),
            "1e-14 mass" => ((u, v) -> 1e-14 * innerₕ(u, v) + L(u, v), r"the system is singular")
        ]
        F = rand(MersenneTwister(KRON_EXT_SEED), ndofs(Wₕ))
        for (name, (f, why)) in refused
            a = form(Wₕ, Wₕ, f)
            @test is_separable(a)
            K = kronecker_operator(a)
            for m in (refusal(() -> fdm_solve(a, F)), refusal(() -> fdm_solve(K, F)))
                @test occursin("fdm_solve does not support this form", m)
                @test occursin(why, m)
                @test occursin("kronecker_operator(a)", m)
            end
        end
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(fxy * u, v) + L(u, v))
        m = refusal(() -> fdm_solve(a, F))
        @test occursin(r"does not support this form: it is not separable", m)
        @test !occursin("kronecker_operator(a)", m)  # it would throw too

        Vₕ = gridspace(mesh(Wₕ), Val(2))
        b = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)) +
                                   L(u(1), v(1)) + L(u(2), v(2)))
        G = rand(2 * ndofs(Wₕ))
        @test occursin(r"does not support this form: it is posed on a composite space",
            refusal(() -> fdm_solve(b, G)))
        @test occursin("composite space", refusal(() -> fdm_solve(kronecker_operator(b), G)))
    end

    # `fdm_preconditioner` inverts the Laplacian-like part `K_L` exactly and leaves the
    # two-axis terms (`K_R`) to the Krylov solver: GMRES with `Pl = P` matches the sparse
    # direct solve on mixed-derivative forms, with advection too (the Schur route), in fewer
    # iterations than without, while an application allocates nothing. `P` is the
    # preconditioner of the form without its mixed term, whose `K_R` is empty, and that one
    # inverts its own interior block; on the boundary rows (identity rows in `assemble`
    # under `dirichlet = :boundary`) it is the identity.
    @testset "fdm_preconditioner: mixed derivatives" begin
        gmres(A, F; kw...) = solve(LinearProblem(A, F), KrylovJL_GMRES(); reltol = 1e-12,
            abstol = 0.0, maxiters = 5000, kw...)
        ldiv_bytes(y, P, x) = (ldiv!(y, P, x); @allocated ldiv!(y, P, x))
        lap(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))
        # `c * innerₕ(D₋ᵧ(u), v)` puts `c` into its axis-1 mass factor: still a one-axis term,
        # which `P` must keep (only the up-to-a-scalar matching reads it as one).
        advs = ("none" => lap,
            "x" => (u, v) -> lap(u, v) + innerₕ(D₋ₓ(u), v),
            "0.5 y" => (u, v) -> lap(u, v) + 0.5 * innerₕ(D₋ᵧ(u), v),
            "x + 50.0 y" => (u, v) -> lap(u, v) + innerₕ(D₋ₓ(u), v) +
                                      50.0 * innerₕ(D₋ᵧ(u), v))
        for n in ((33, 25), (9, 8, 7)), dir in (nothing, :boundary), (name, L) in advs
            adv = name != "none"
            @testset "$n, dirichlet = $dir, advection = $name" begin
                Wₕ = graded_space(n)
                a = form(Wₕ, Wₕ, (u, v) -> L(u, v) + 0.25 * innerₕ(D₋ₓ(D₋ᵧ(u)), v))
                b = form(Wₕ, Wₕ, L)
                P = fdm_preconditioner(a; dirichlet = dir)
                @test P isa Bramble.FDMPreconditioner{Float64}
                @test P isa Bramble.AbstractMatrixFreePreconditioner{Float64}
                @test size(P) == (ndofs(Wₕ), ndofs(Wₕ))
                @test (P.factorization isa KronExt._SchurFactorization) == adv

                bd = Bramble._combined_mask(mesh(Wₕ), (:boundary,))
                G = rand(MersenneTwister(KRON_EXT_SEED), ndofs(Wₕ))
                G0 = copy(G)
                dir === :boundary && (G0[bd] .= 0)
                y = fill(NaN, length(G))
                @test ldiv_bytes(y, P, G) == 0
                @test y == fdm_preconditioner(b; dirichlet = dir) \ G
                @test ldiv!(P, copy(G)) == y  # in place: `x` aliases `y`
                if dir === :boundary
                    @test y[bd] == G[bd]
                    @test (P \ G0)[.!bd] == y[.!bd]
                end
                @test isapprox(P \ G0, assemble(b; dirichlet = dir) \ G0; rtol = 1e-10)

                A = assemble(a; dirichlet = dir)
                xref = A \ G0
                s0 = gmres(A, G0)
                s1 = gmres(A, G0; Pl = P)
                @test norm(s1.u - xref) <= 1e-8 * norm(xref)
                @test s1.iters < s0.iters
            end
        end
    end

    # The preconditioner refuses what `fdm_solve` refuses, a two-axis term apart, worded for
    # itself; a form whose terms all differ from the masses on two axes or none has no
    # Laplacian-like part.
    @testset "fdm_preconditioner: refusals" begin
        refusal(f) =
            try
                f()
                "no error"
            catch e
                e isa ArgumentError ? sprint(showerror, e) : "wrong error $(typeof(e))"
            end
        Wₕ = graded_space((9, 7))
        L(u, v) = inner₊(∇ₕ(u), ∇ₕ(v))
        fxy = Rₕ(Wₕ, x -> (1 + x[1]) * (2 + x[2]^2))
        refused = [
            "no Laplacian-like part" => ((u, v) -> innerₕ(u, v) + innerₕ(D₋ₓ(D₋ᵧ(u)), v),
                r"axis 1 has no term of its own.*no Laplacian-like part"),
            "singular mass" => ((u, v) -> innerₕ(Bramble.restrict_to(:interior, u), v),
                r"mass is not symmetric positive definite"),
            "gradient and mixed only" => ((u, v) -> L(u, v) + innerₕ(D₋ₓ(D₋ᵧ(u)), v),
                r"the Laplacian-like part is singular"),
            "not separable" => ((u, v) -> innerₕ(fxy * u, v) + L(u, v), r"it is not separable")
        ]
        for (name, (f, why)) in refused
            m = refusal(() -> fdm_preconditioner(form(Wₕ, Wₕ, f)))
            @test occursin("fdm_preconditioner does not support this form", m)
            @test occursin(why, m)
            @test occursin("jacobi_preconditioner(a)", m)
        end
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + L(u, v))
        @test_throws "fdm_preconditioner only supports" fdm_preconditioner(a; dirichlet = :left)
        Vₕ = gridspace(mesh(Wₕ), Val(2))
        b = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)) +
                                   L(u(1), v(1)) + L(u(2), v(2)))
        @test occursin(r"fdm_preconditioner does not support this form: it is posed on a composite",
            refusal(() -> fdm_preconditioner(b)))
    end

    # Well conditioned in Float32 (eigenvalue ratio about 8e3): the singularity test must
    # not grow with the number of unknowns, which refused this solve.
    @testset "fdm_solve: Float32 Laplacian" begin
        Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
        Wₕ = gridspace(mesh(Ω, (33, 33), (true, true); backend = backend(Float32)))
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        A = assemble(a)
        F = rand(MersenneTwister(KRON_EXT_SEED), Float32, size(A, 1))
        x = fdm_solve(a, F)
        @test eltype(x) === Float32
        @test isapprox(x, Float64.(A) \ Float64.(F); rtol = 1e-3)
    end

    # `fdm_factorize` once, then `fdm_solve!` per right-hand side: it returns `x`, matches the
    # sparse direct solve and `ldiv!`, and allocates nothing after a warm-up, for a symmetric
    # form (fast diagonalisation) and an advection form (Schur), graded 2D and 3D, on Serial
    # and CpuPolyester meshes. The solve is serial whatever the mesh's policy, so the
    # CpuPolyester result is bitwise CpuSerial's; `fdm_preconditioner`'s `ldiv!` allocates
    # nothing on either.
    @testset "fdm_solve!: 0 bytes, CpuPolyester too" begin
        function policy_space(n::NTuple{D, Int}, policy) where {D}
            Ω = mesh(domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D))), n,
                ntuple(_ -> false, D); backend = backend(policy = policy))
            Bramble.change_points!(Ω,
                ntuple(d -> range(0.0, 1.0; length = n[d]) .^ (1 + 0.25d), D))
            return gridspace(Ω)
        end
        solve_bytes(x, f, F) = (fdm_solve!(x, f, F); fdm_solve!(x, f, F);
            @allocated fdm_solve!(x, f, F))
        ldiv_bytes(x, f, F) = (ldiv!(x, f, F); ldiv!(x, f, F); @allocated ldiv!(x, f, F))
        forms = ("symmetric" => (u, v) -> innerₕ(u, v) + 2.5 * inner₊(∇ₕ(u), ∇ₕ(v)),
            "advection" => (u, v) -> innerₕ(u, v) + 0.1 * inner₊(∇ₕ(u), ∇ₕ(v)) +
                                     innerₕ(D₋ₓ(u), v))
        serial = Dict{Any, Vector{Float64}}()
        for policy in (Serial(), CpuPolyester()), n in ((13, 9), (7, 6, 8)),
            dir in (nothing, :boundary)
            Wₕ = policy_space(n, policy)
            @test execution_policy(backend(mesh(Wₕ))) isa typeof(policy)
            bd = Bramble._combined_mask(mesh(Wₕ), (:boundary,))
            for (name, L) in forms
                a = form(Wₕ, Wₕ, L)
                A = assemble(a; dirichlet = dir)
                F = rand(MersenneTwister(KRON_EXT_SEED + 9), size(A, 1))
                dir === :boundary && (F[bd] .= 0)
                f = fdm_factorize(a; dirichlet = dir)
                @test size(f) == size(A) && size(f, 1) == size(f, 2) == size(A, 1)
                @test size(f, 3) == 1
                @test_throws BoundsError size(f, 0)
                @test (f isa KronExt._SchurFactorization) == (name == "advection")
                x = fill(NaN, length(F))
                @test fdm_solve!(x, f, F) === x
                @test solve_bytes(x, f, F) == 0
                @test isapprox(x, A \ F; rtol = 1e-10)
                @test x == fdm_solve(a, F; dirichlet = dir)
                dir === :boundary && @test all(iszero, x[bd])
                y = fill(NaN, length(F))
                @test ldiv_bytes(y, f, F) == 0
                @test y == x
                key = (name, n, dir)
                policy isa Serial ? (serial[key] = copy(x)) : @test(x == serial[key])
                # `K` takes `dirichlet` too, though `fdm_solve(K, F)` does not.
                fK = fdm_factorize(kronecker_operator(a); dirichlet = dir)
                @test fdm_solve!(fill(NaN, length(F)), fK, F) == x
                @test_throws DimensionMismatch fdm_solve!(x, f, zeros(length(x) + 1))
                @test_throws DimensionMismatch fdm_solve!(zeros(length(x) + 1), f, F)
                @test_throws "x has length $(length(x) - 1)" fdm_solve!(zeros(length(x) - 1), f, F)
                @test_throws "F has length $(length(x) - 1)" fdm_solve!(x, f, zeros(length(x) - 1))
            end
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
                                       0.25 * innerₕ(D₋ₓ(D₋ᵧ(u)), v))
            P = fdm_preconditioner(a; dirichlet = dir)
            @test size(P, 1) == size(P, 2) == ndofs(Wₕ) && size(P, 3) == 1
            @test ldiv_bytes(zeros(size(P, 1)), P, rand(size(P, 1))) == 0
            @test_throws "y has length" ldiv!(zeros(size(P, 1) - 1), P, rand(size(P, 1)))
        end
    end

    # `fdm_factorize` refuses what `fdm_solve` refuses, with `fdm_solve`'s message; a K built
    # before `change_points!` is refused with the stale-weights error.
    @testset "fdm_factorize: refusals" begin
        Wₕ = graded_space((9, 7))
        fxy = Rₕ(Wₕ, x -> (1 + x[1]) * (2 + x[2]^2))
        b = form(Wₕ, Wₕ, (u, v) -> innerₕ(fxy * u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        @test_throws ArgumentError fdm_factorize(b)
        @test_throws "fdm_solve does not support this form: it is not separable" fdm_factorize(b)
        m = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + innerₕ(D₋ₓ(D₋ᵧ(u)), v))
        @test_throws "fdm_solve does not support this form" fdm_factorize(m)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        @test_throws "fdm_solve only supports" fdm_factorize(a; dirichlet = :left)
        @test_throws "fdm_solve only supports" fdm_factorize(kronecker_operator(a);
            dirichlet = :left)
        Vₕ = gridspace(mesh(Wₕ), Val(2))
        c = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
        @test_throws "posed on a composite" fdm_factorize(c)
        @test_throws "posed on a composite" fdm_factorize(kronecker_operator(c))

        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), (false, false))
        Uₕ = gridspace(Ωₕ)
        K = kronecker_operator(form(Uₕ, Uₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))))
        @test size(fdm_factorize(K)) == size(K)
        Bramble.change_points!(Ωₕ,
            (range(0.0, 1.0; length = 9) .^ 2, range(0.0, 1.0; length = 7) .^ 2))
        @test_throws "change_points!" fdm_factorize(K)
    end

    # gpena/Bramble.jl#442: a K built before `change_points!` refuses `fdm_solve` and the
    # conversion with the stale-weights `ArgumentError`; both work before the move.
    @testset "stale K after a mesh move" begin
        stale(f) =
            try
                f()
                false
            catch e
                e isa ArgumentError && occursin("change_points!", sprint(showerror, e))
            end
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), (false, false))
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        K = kronecker_operator(a)
        F = rand(MersenneTwister(KRON_EXT_SEED + 8), size(K, 1))
        @test !stale(() -> fdm_solve(K, F)) && !stale(() -> kronecker(K))
        Bramble.change_points!(Ωₕ,
            (range(0.0, 1.0; length = 9) .^ 2, range(0.0, 1.0; length = 7) .^ 2))
        @test stale(() -> fdm_solve(K, F))
        @test stale(() -> kronecker(K))
        # The form's own space predates the move: `fdm_solve(a, F)` refuses it as `assemble`
        # does, with the space's stale-weights error, not a factorisation of stale weights.
        @test_throws "gridspace(mesh(Wₕ)) again" fdm_solve(a, F)
        @test_throws "gridspace(mesh(Wₕ)) again" fdm_solve(a, F; dirichlet = :boundary)
        Bramble.iterative_refinement!(Ωₕ)
        @test_throws "gridspace(mesh(Wₕ)) again" fdm_solve(a, F)
    end
end

end # module
