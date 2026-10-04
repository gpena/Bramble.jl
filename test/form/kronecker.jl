module TestFormKronecker

using Test
using Bramble
using Bramble: is_separable, kronecker_operator, KroneckerLinearOperator
using LinearAlgebra: Diagonal, issymmetric, mul!
using SparseArrays: SparseMatrixCSC, sparse
using Random
using LinearSolve: LinearProblem, solve, KrylovJL_CG
using ForwardDiff
using ..TestUtils: WITH_AD_TESTS

# `is_separable`/`kronecker_operator`: a bilinear form whose
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

_kron_alloc5_no_scratch(y, K, x, α, β) = @allocated mul!(y, K, x, α, β)

# A graded mesh on backend `be`, as benchmark/operator_routes.jl builds one: uniform, then
# moved by `change_points!` to `t^(1 + d/4)` along axis `d`.
function _kron_graded_space(n::NTuple{D, Int}, be) where {D}
    Ωₕ = mesh(domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D))), n, ntuple(_ -> true, D);
        backend = be)
    Bramble.change_points!(Ωₕ, ntuple(d -> range(0.0, 1.0; length = n[d]) .^ (1 + 0.25d), D))
    return gridspace(Ωₕ)
end

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

    # mul! agrees with assemble, 2D (31x17) and 3D (9x8x7).
    @testset "mul!: matches assemble, 2D and 3D" begin
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

            if WITH_AD_TESTS
                # ForwardDiff Duals flow through the promoted scratch.
                x0, v = rand(n), rand(n)
                dK = ForwardDiff.derivative(t -> K * (x0 .+ t .* v), 0.0)
                @test isapprox(dK, A * v; rtol = 1e-12, atol = 1e-12)
            end

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

    # Fused mul! with an axis-1 factor that is not tridiagonal.
    @testset "fused mul!: non-tridiagonal axis 1" begin
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

    # `getindex` reads the Kronecker structure entry by entry, without `mul!` or `kron`.
    @testset "getindex and complex α match assemble" begin
        Random.seed!(KRON_SEED + 5)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 5), (false, false))
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + 2.5 * inner₊(∇ₕ(u), ∇ₕ(v)))
        A = assemble(a)
        K = kronecker_operator(a)
        n = ndofs(Wₕ)
        @test all(
            isapprox(K[i, j], A[i, j]; rtol = 1e-12, atol = 1e-12) for i in 1:n, j in 1:n
        )
        @test count(i -> !iszero(K[i, i + 1]), 1:(n - 1)) > 0  # off-diagonals reached
        @test_throws BoundsError K[n + 1, 1]

        # A complex `α` keeps `α * c_t` complex (`_kron_scalar`'s generic method) rather
        # than converting it to the operator's `Float64`.
        x = rand(n)
        α = 0.5 + 2.0im
        y0 = rand(ComplexF64, n)
        y = copy(y0)
        mul!(y, K, x, α, 1)
        @test isapprox(y, α * (A * x) + y0; rtol = 1e-12, atol = 1e-12)

        @test_throws "cannot multiply a vector of length $(n + 1)" mul!(
            zeros(n), K, rand(n + 1)
        )
        @test_throws DimensionMismatch mul!(zeros(n + 1), K, rand(n), 1.0, 0.0)
    end

    @testset "kronecker_operator refuses non-factors" begin
        Random.seed!(KRON_SEED + 6)
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 5), (false, false))
        W2 = gridspace(Ω2)
        Ω2b = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 5), (false, false))
        W2b = gridspace(Ω2b)
        Ω1 = mesh(domain(interval(0.0, 1.0)), 9, false)
        W1 = gridspace(Ω1)
        fₕ = Rₕ(W2, x -> 1.0 + x[1])

        @test_throws "got a 1D form" kronecker_operator(
            form(W1, W1, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        )
        # Two meshes with the same shape are still two meshes.
        @test_throws "sharing one mesh" kronecker_operator(form(W2, W2b, (u, v) -> innerₕ(u, v)))
        @test_throws "is not one of the recognised separable shapes" kronecker_operator(
            form(W2, W2, (u, v) -> innerₕ(fₕ * u, v))
        )
    end

    # Hand-built terms reach what `kronecker_operator` never builds from a valid form: a
    # one-point axis 1 with a finite difference factor (a collapsed axis has zero spacing,
    # so the factor it assembles is `NaN`), a plain-vector mass diagonal, and a term with
    # two non-diagonal factors.
    @testset "fused mul!: hand-built factors" begin
        Random.seed!(KRON_SEED + 7)
        Ωc = mesh(domain(interval(0.5, 0.5) × interval(0.0, 1.0)), (1, 9), (true, false))
        Wc = gridspace(Ωc)
        am = form(Wc, Wc, (u, v) -> innerₕ(u, v))
        Km = kronecker_operator(am)
        @test Km.dims == (1, 9)
        x = rand(9)
        @test isapprox(Km * x, assemble(am) * x; rtol = 1e-12, atol = 1e-12)

        # Axis 1 holds one point: the tridiagonal line has no neighbours.
        hy = collect(Km.terms[1].factors[2].diag)
        S1 = sparse([3.0;;])
        Dy = Diagonal(hy)
        line = Bramble._kron_line_operator(S1)
        @test line isa Bramble._KronTridiag
        factors = (S1, Dy)
        term = Bramble.KroneckerTerm{2, Tuple{}, typeof(factors), typeof(line)}((), factors, line)
        K1 = KroneckerLinearOperator{Float64, 2, Tuple{typeof(term)}}((term,), (1, 9), 9)
        @test isapprox(K1 * x, 3.0 .* hy .* x; rtol = 1e-12, atol = 1e-12)

        # Two sparse factors in one term: refused when applied.
        S2 = sparse([2.0 -1.0; -1.0 2.0])
        f3 = (Diagonal([1.0, 2.0]), S2, S2)
        t3 = Bramble.KroneckerTerm{3, Tuple{}, typeof(f3), Nothing}((), f3, nothing)
        K3 = KroneckerLinearOperator{Float64, 3, Tuple{typeof(t3)}}((t3,), (2, 2, 2), 8)
        @test_throws "more than one non-diagonal factor" K3 * ones(8)
    end

    # LinearSolve agreement (SPD, no Dirichlet).
    @testset "LinearSolve: SPD, no Dirichlet" begin
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

    # The 0 B above is measured under the default (serial) policy only. `CpuThreaded()` runs
    # the lines serially too and must stay at 0 B; `CpuPolyester`'s are tested in
    # test/ext/polyester_ext.jl. Graded meshes, small and large, 2D and 3D, 3- and 5-argument
    # `mul!`, each measured after a warm call.
    @testset "Kronecker: zero bytes per policy" begin
        for P in (Bramble.CpuSerial(), Bramble.CpuThreaded()),
            n in ((9, 7), (257, 257), (6, 5, 4), (33, 33, 33))

            @testset "$P $(join(n, '×'))" begin
                W = _kron_graded_space(n, backend(; policy = P))
                a = form(W, W, (u, v) -> innerₕ(u, v) + 2.5 * inner₊(∇ₕ(u), ∇ₕ(v)))
                K = kronecker_operator(a)
                @test K.policy === P
                A = assemble(a)
                N = ndofs(W)
                x = rand(N)
                y = similar(x)
                y0 = rand(N)
                s = (zeros(N), zeros(N))
                mul!(y, K, x)
                @test isapprox(y, A * x; rtol = 1e-12, atol = 1e-12)
                y5 = copy(y0)
                mul!(y5, K, x, 0.5, -3.0)
                @test isapprox(y5, 0.5 * (A * x) - 3.0 * y0; rtol = 1e-12, atol = 1e-12)
                _kron_alloc_no_scratch(y, K, x)
                @test _kron_alloc_no_scratch(y, K, x) == 0
                _kron_alloc_with_scratch(y, K, x, s)
                @test _kron_alloc_with_scratch(y, K, x, s) == 0
                _kron_alloc5_no_scratch(y5, K, x, 0.5, -3.0)
                @test _kron_alloc5_no_scratch(y5, K, x, 0.5, -3.0) == 0
                _kron_alloc5_with_scratch(y5, K, x, 0.5, -3.0, s)
                @test _kron_alloc5_with_scratch(y5, K, x, 0.5, -3.0, s) == 0
            end
        end
    end

    # summarysize(K) << summarysize(assemble(a)).
    @testset "memory: K smaller than assembled" begin
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
