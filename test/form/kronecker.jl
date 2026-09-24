module TestFormKronecker

using Test
using Bramble
using Bramble: is_separable, kronecker_operator, KroneckerLinearOperator
using LinearAlgebra: issymmetric, mul!
using SparseArrays: SparseMatrixCSC
using Random
using LinearSolve: LinearProblem, solve, KrylovJL_CG
using ForwardDiff

# `is_separable`/`kronecker_operator` (gpena/Bramble.jl#162): a bilinear form whose
# resolved AST is a sum of `innerₕ(u, v)`/`inner₊(∇ₕ(u), ∇ₕ(v))`-shaped terms over a
# `ScalarGridSpace` on a `MeshnD` factors as a sum of Kronecker products of 1D matrices,
# `H_D ⊗ ... ⊗ A_d ⊗ ... ⊗ H_1`. `KroneckerLinearOperator` applies that sum by sum
# factorisation instead of assembling the full matrix; every check here compares it against
# the explicit `assemble(a)` it is meant to agree with.

const KRON_SEED = 20260919

# Inside a function so `@allocated` measures `mul!` alone, not global-scope boxing.
_kron_alloc_with_scratch(y, K, x, s) = @allocated mul!(y, K, x; scratch = s)
_kron_alloc5_with_scratch(y, K, x, α, β, s) = @allocated mul!(y, K, x, α, β; scratch = s)
_kron_alloc_no_scratch(y, K, x) = @allocated mul!(y, K, x)

@testset "Kronecker" begin
    @testset "is_separable" begin
        for (npts2, npts3) in ((true, true), (false, false))
            Random.seed!(KRON_SEED)
            Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), npts2)
            W2 = gridspace(Ω2)

            Random.seed!(KRON_SEED)
            Ω3 = mesh(
                domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
                (6, 5, 4), npts3
            )
            W3 = gridspace(Ω3)

            for (Wₕ, tag) in ((W2, "2D"), (W3, "3D"))
                unif_tag = npts2 === true ? "uniform" : "non-uniform"
                @testset "$tag, $unif_tag" begin
                    @test is_separable(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v)))
                    @test is_separable(form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))))
                    @test is_separable(
                        form(
                        Wₕ, Wₕ,
                        (u, v) -> innerₕ(u, v) + 2.5 * inner₊(∇ₕ(u), ∇ₕ(v))
                    )
                    )
                end
            end
        end

        # A grid-function coefficient has no tensor structure: not separable.
        Random.seed!(KRON_SEED)
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), false)
        W2 = gridspace(Ω2)
        fₕ = Rₕ(W2, x -> 1.0 + x[1])
        @test !is_separable(form(W2, W2, (u, v) -> innerₕ(fₕ * u, v)))

        # A 1D mesh has nothing to factor.
        Ω1 = mesh(domain(interval(0.0, 1.0)), 9, false)
        W1 = gridspace(Ω1)
        @test !is_separable(form(W1, W1, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))))
    end

    @testset "mul! agrees with assemble, 2D (31x17) and 3D (9x8x7)" begin
        Random.seed!(KRON_SEED)
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (31, 17), (false, false))
        W2 = gridspace(Ω2)

        Random.seed!(KRON_SEED + 1)
        Ω3 = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (9, 8, 7), (false, false, false)
        )
        W3 = gridspace(Ω3)

        for Wₕ in (W2, W3)
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + 2.5 * inner₊(∇ₕ(u), ∇ₕ(v)))
            A = assemble(a)
            K = kronecker_operator(a)

            n = ndofs(Wₕ)
            @test size(K) == (n, n)
            @test eltype(K) == eltype(A)

            x = rand(n)
            y = similar(x)
            mul!(y, K, x)
            yref = A * x
            @test isapprox(y, yref; rtol = 1e-12, atol = 1e-12)

            # `Base.:*` and `LinearAlgebra.issymmetric` follow the same contract.
            @test isapprox(K * x, yref; rtol = 1e-12, atol = 1e-12)
            @test issymmetric(K)
            @test issymmetric(Matrix(A))

            # `SparseMatrixCSC(K)`: an explicit `kron` of the factors.
            @test SparseMatrixCSC(K) ≈ A

            # Zero allocations with caller-owned scratch (`K` holds no buffers).
            s = (zeros(n), zeros(n))
            _kron_alloc_with_scratch(y, K, x, s)
            @test _kron_alloc_with_scratch(y, K, x, s) == 0
            @test isapprox(y, yref; rtol = 1e-12, atol = 1e-12)

            # One shared `K`, many threads: no shared state, so no race.
            xs = [rand(n) for _ in 1:32]
            ys = [zeros(n) for _ in 1:32]
            Threads.@threads for i in 1:32
                for _ in 1:10
                    mul!(ys[i], K, xs[i])
                end
            end
            @test all(isapprox(ys[i], A * xs[i]; rtol = 1e-12, atol = 1e-12) for i in 1:32)

            # ForwardDiff Duals flow through the promoted scratch.
            x0, v = rand(n), rand(n)
            dK = ForwardDiff.derivative(t -> K * (x0 .+ t .* v), 0.0)
            @test isapprox(dK, A * v; rtol = 1e-12, atol = 1e-12)

            # Five-argument `mul!`: `y = α * K * x + β * y`, Int/Bool/Float α and β.
            M = Matrix(A)
            for (α, β) in ((1, 0), (2.0, 0.0), (-1, 1), (0.5, -3.0), (0, 2.0), (true, false))
                y0 = rand(n)
                y5 = copy(y0)
                mul!(y5, K, x, α, β)
                @test isapprox(y5, α * M * x + β * y0; rtol = 1e-12, atol = 1e-12)
            end

            # `β = 0` overwrites `y`: a NaN already there must not survive.
            ynan = fill(NaN, n)
            mul!(ynan, K, x, 1.0, 0.0)
            @test isapprox(ynan, yref; rtol = 1e-12, atol = 1e-12)
            ynan = fill(NaN, n)
            mul!(ynan, K, x, 2, false)
            @test isapprox(ynan, 2 * yref; rtol = 1e-12, atol = 1e-12)

            # `semidiscretize_rhs`'s pattern: `copyto!(du, F); mul!(du, K, u, -1, 1)`.
            F = rand(n)
            du = similar(F)
            copyto!(du, F)
            mul!(du, K, x, -1, 1)
            @test isapprox(du, F - A * x; rtol = 1e-12, atol = 1e-12)

            # Zero allocations for the five-argument form with caller-owned scratch.
            _kron_alloc5_with_scratch(du, K, x, -1, 1, s)
            @test _kron_alloc5_with_scratch(du, K, x, -1, 1, s) == 0
            _kron_alloc5_with_scratch(du, K, x, 0.5, 0.0, s)
            @test _kron_alloc5_with_scratch(du, K, x, 0.5, 0.0, s) == 0

            # The fused pass needs no work vectors: 0 bytes without `scratch` too.
            _kron_alloc_no_scratch(y, K, x)
            @test _kron_alloc_no_scratch(y, K, x) == 0

            # A `Ref` coefficient stays live through `mul!`.
            c = Ref(2.5)
            aref = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + c * inner₊(∇ₕ(u), ∇ₕ(v)))
            Kref = kronecker_operator(aref)
            c[] = -0.75
            @test isapprox(Kref * x, assemble(aref) * x; rtol = 1e-12, atol = 1e-12)
        end
    end

    @testset "fused mul!: axis-1 factor that is not tridiagonal" begin
        # `kronecker_operator` always stores the axis-1 difference factor in tridiagonal
        # form (`_KronTridiag`); a factor with any other sparsity keeps its CSC matrix and
        # takes the row-gather path. Force that path on the same factors and compare.
        Random.seed!(KRON_SEED + 4)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (13, 9), (false, false))
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        K = kronecker_operator(a)
        @test any(t -> t.line isa Bramble._KronTridiag, K.terms)
        terms = map(K.terms) do t
            line = t.line isa Bramble._KronTridiag ? t.factors[1] : t.line
            Bramble.KroneckerTerm{2, typeof(t.scales), typeof(t.factors), typeof(line)}(
                t.scales, t.factors, line
            )
        end
        Kcsc = KroneckerLinearOperator{eltype(K), 2, typeof(terms)}(terms, K.dims, K.n)
        x = rand(size(K, 1))
        @test isapprox(Kcsc * x, K * x; rtol = 1e-12, atol = 1e-12)
        @test isapprox(Kcsc * x, assemble(a) * x; rtol = 1e-12, atol = 1e-12)
    end

    @testset "LinearSolve agreement (SPD, no Dirichlet)" begin
        Random.seed!(KRON_SEED + 2)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (17, 13), (false, false))
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        A = assemble(a)
        K = kronecker_operator(a)
        @test issymmetric(Matrix(A))

        n = ndofs(Wₕ)
        Random.seed!(KRON_SEED + 3)
        b = rand(n)

        # Explicit, tight tolerances: `KrylovJL_CG`'s defaults stop CG early enough on this
        # system (relative residual, not solution accuracy) that the two solves can agree
        # with each other while both sitting a few permille off the true solution -- a
        # weaker, still faithful, "reach the same solution" check than intended.
        cg_kwargs = (; reltol = 1e-10, abstol = 1e-10, maxiters = 2000)
        sol_A = solve(LinearProblem(A, b), KrylovJL_CG(); cg_kwargs...)
        sol_K = solve(LinearProblem(K, b), KrylovJL_CG(); cg_kwargs...)

        @test isapprox(sol_A.u, sol_K.u; rtol = 1e-6, atol = 1e-8)
        # Both actually solve the system, not merely agree with each other.
        @test isapprox(A * sol_K.u, b; rtol = 1e-6, atol = 1e-8)
    end

    @testset "Memory: summarysize(K) << summarysize(assemble(a))" begin
        Ωₕ = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (60, 60, 60), true
        )
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        A = assemble(a)
        K = kronecker_operator(a)

        size_A = Base.summarysize(A)
        size_K = Base.summarysize(K)
        @test size_K < 0.01 * size_A
    end
end

end # module
