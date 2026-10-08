module TestFormKronecker

using Test
using Bramble
using Bramble: is_separable, kronecker_operator, KroneckerLinearOperator
using Bramble: D₊ₓ, D₋ₓ, D₋ᵧ, Mₓ, jumpₓ, restrict_to
using LinearAlgebra: I, Diagonal, issymmetric, kron, mul!
using SparseArrays: SparseMatrixCSC, sparse, spdiagm, nnz, findnz, dropzeros
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

# A user's call site, for `code_typed`: it must invoke `kronecker_operator`, not inline it.
_kron_call!(y, a, x) = (mul!(y, kronecker_operator(a), x); nothing)

# The method instances `ci` invokes, as strings.
_kron_invokes(ci) = [string(s.args[1]) for s in ci.code if s isa Expr && s.head === :invoke]

# Whether `f()` throws an `ArgumentError` whose message contains `needle`: the stale-operator
# error by default (gpena/Bramble.jl#442), or a stale space's with its remedy as `needle`.
function _kron_stale(f, needle = "KroneckerLinearOperator's factors")
    try
        f()
        return false
    catch e
        msg = sprint(showerror, e)
        return e isa ArgumentError && occursin("change_points!", msg) && occursin(needle, msg)
    end
end

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

        # A grid-function coefficient varying along both axes has no tensor structure: not
        # separable. One varying along a single axis is (see "Kronecker: projected forms").
        Random.seed!(KRON_SEED)
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), false)
        W2 = gridspace(Ω2)
        fₕ = Rₕ(W2, x -> 1.0 + x[1] * x[2])
        @test !is_separable(form(W2, W2, (u, v) -> innerₕ(fₕ * u, v)))
        # Two meshes with the same shape are still two meshes: the factors are built on one.
        W2b = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), false))
        @test !is_separable(form(W2, W2b, (u, v) -> innerₕ(u, v)))

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
            Bramble.KroneckerTerm{
                2, typeof(t.scales), typeof(t.factors), typeof(line), typeof(t.rows)
            }(t.scales, t.factors, line, t.rows, t.symmetric)
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
        fₕ = Rₕ(W2, x -> 1.0 + x[1] * x[2])

        @test_throws "got a 1D form" kronecker_operator(
            form(W1, W1, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        )
        # Two meshes with the same shape are still two meshes.
        @test_throws "sharing one mesh" kronecker_operator(form(W2, W2b, (u, v) -> innerₕ(u, v)))
        # The error names the node with no projection.
        @test_throws ArgumentError kronecker_operator(form(W2, W2, (u, v) -> innerₕ(fₕ * u, v)))
        @test_throws "the node GridFunctionScale" kronecker_operator(
            form(W2, W2, (u, v) -> innerₕ(fₕ * u, v))
        )
        @test_throws "the node RegionRestriction(:boundary)" kronecker_operator(
            form(W2, W2, (u, v) -> innerₕ(D₋ₓ(restrict_to(:boundary, u)), v) + innerₕ(u, v))
        )
    end

    # Hand-built terms reach what `kronecker_operator` never builds from a valid form: a
    # one-point axis 1 with a finite difference factor (a collapsed axis has zero spacing,
    # so the factor it assembles is `NaN`), a plain-vector mass diagonal, and a term with
    # two non-diagonal factors on one-point-wide axes.
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
        term = Bramble._kron_term((), (S1, Dy))
        @test term.line isa Bramble._KronTridiag
        K1 = KroneckerLinearOperator{Float64, 2, Tuple{typeof(term)}}((term,), (1, 9), 9)
        @test isapprox(K1 * x, 3.0 .* hy .* x; rtol = 1e-12, atol = 1e-12)

        # Two sparse factors on axes 2 and 3 over a two-point axis 1, and a one-point axis 1
        # under two sparse factors with a tridiagonal axis-1 sweep.
        S2 = sparse([2.0 -1.0; -1.0 2.0])
        f3 = (Diagonal([1.0, 2.0]), S2, S2)
        t3 = Bramble._kron_term((), f3)
        K3 = KroneckerLinearOperator{Float64, 3, Tuple{typeof(t3)}}((t3,), (2, 2, 2), 8)
        x8 = rand(8)
        @test isapprox(K3 * x8, kron(S2, S2, sparse(f3[1])) * x8; rtol = 1e-12, atol = 1e-12)
        f1 = (S1, S2, S2)
        t1 = Bramble._kron_term((Ref(-2.0),), f1)
        @test t1.line isa Bramble._KronTridiag
        K1b = KroneckerLinearOperator{Float64, 3, Tuple{typeof(t1)}}((t1,), (1, 2, 2), 4)
        x4 = rand(4)
        @test isapprox(K1b * x4, -2.0 * (kron(S2, S2, S1) * x4); rtol = 1e-12, atol = 1e-12)
    end

    # The line kernels index under `@inbounds`, so sizes are checked at construction: a
    # non-square factor is refused by `_kron_term`, a factor that does not match the grid
    # (or `n != prod(dims)`) by the operator constructor, with or without a policy.
    @testset "Kronecker: size checks" begin
        I3 = sparse([2.0 -1 0; -1 2 -1; 0 -1 2])
        rect = sparse([1.0 0 0 0; 0 1 0 0; 0 0 0 1])
        @test_throws ArgumentError Bramble._kron_term((), (I3, rect))
        @test_throws "axis-2 factor is 3 × 4" Bramble._kron_term((), (I3, rect))
        t = Bramble._kron_term((), (I3, Diagonal([1.0, 2.0, 3.0])))
        T = Tuple{typeof(t)}
        @test_throws DimensionMismatch KroneckerLinearOperator{Float64, 2, T}((t,), (3, 4), 12)
        @test_throws "axis-1 factor is 3 × 3" KroneckerLinearOperator{Float64, 2, T}(
            (t,), (4, 3), 12)
        @test_throws DimensionMismatch KroneckerLinearOperator{Float64, 2, T}((t,), (3, 3), 10)
        @test_throws DimensionMismatch KroneckerLinearOperator{
            Float64, 2, T, Bramble.CpuThreaded}((t,), (3, 4), 12, Bramble.CpuThreaded())
        @test_throws DimensionMismatch KroneckerLinearOperator{Float64, 3, T}(
            (t,), (3, 3, 1), 9)
        @test KroneckerLinearOperator{Float64, 2, T}((t,), (3, 3), 9) * ones(9) ≈
              kron(Diagonal([1.0, 2.0, 3.0]), I3) * ones(9)
    end

    # An empty axis 1 under a symmetric (so tridiagonal-stored) 0 × 0 factor: every line has
    # `m = 0` points and must write nothing. `x` and `y` are empty views into padded
    # buffers, so a kernel writing or reading past them (under `@inbounds`) changes the
    # padding instead of going unnoticed.
    @testset "Kronecker: empty axis 1" begin
        for f2 in (Diagonal([1.0, 2.0]), sparse([1.0 2; 3 4]))
            t = Bramble._kron_term((), (sparse(zeros(0, 0)), f2))
            @test t.line isa Bramble._KronTridiag
            K = KroneckerLinearOperator{Float64, 2, Tuple{typeof(t)}}((t,), (0, 2), 0)
            xb = fill(NaN, 3)
            yb = fill(7.0, 3)
            y = view(yb, 2:1)
            @test isempty(mul!(y, K, view(xb, 2:1)))
            @test isempty(mul!(y, K, view(xb, 2:1), 2.0, 1.0))
            @test yb == fill(7.0, 3)
            @test size(K * Float64[]) == (0,)
        end
    end

    # `_kron_term` stores a non-symmetric factor's rows as the CSC of its transpose, so the
    # line kernels gather rows without assuming symmetry, and a term may carry any number of
    # non-diagonal factors. Oracle: an explicit `kron` of the same factors, last axis
    # leftmost. Every axis-1 kernel (diagonal, tridiagonal sweep, row gather) meets one and
    # two neighbour axes; serial and threaded agree bit for bit, and serial allocates 0 B.
    @testset "Kronecker: general factors" begin
        rng = Random.MersenneTwister(KRON_SEED + 9)
        band(m, lo, hi) = spdiagm((k => randn(rng, m - abs(k)) for k in (-lo):hi)...)
        symtri(m) = (B = band(m, 1, 0); B + B')
        dg(m) = Diagonal(rand(rng, m) .+ 0.5)
        coeff(s) = prod(c -> c isa Ref ? c[] : c, s; init = 1.0)
        oracle(specs) = sum(coeff(s) * kron(map(sparse, reverse(f))...) for (s, f) in specs)
        cases = Any[  # Any: each case's factor tuple is its own type
            (6, 5) => ((((), (band(6, 1, 1), band(5, 2, 1))), ((), (dg(6), band(5, 0, 2))),
                ((), (band(6, 0, 1), dg(5))))),
            (6, 5) => ((((2.0,), (symtri(6), band(5, 1, 0))),)),
            (5, 4, 6) => ((((), (dg(5), band(4, 1, 1), band(6, 1, 2))),)),
            (5, 4, 6) => ((((), (symtri(5), band(4, 1, 0), band(6, 0, 1))),)),
            (5, 4, 6) => ((((Ref(0.5),), (band(5, 2, 0), band(4, 1, 1), Matrix(band(6, 1, 1)))),
                ((), (dg(5), dg(4), dg(6))))),
            (5, 4, 6) => ((((), (symtri(5), symtri(4), dg(6))),))
        ]
        for (dims, specs) in cases
            terms = map(sp -> Bramble._kron_term(sp[1], sp[2]), specs)
            A = oracle(specs)
            N = prod(dims)
            D = length(dims)
            Ks, Kt = (KroneckerLinearOperator{Float64, D, typeof(terms), typeof(P)}(
                          terms, dims, N, P) for P in (Bramble.CpuSerial(), Bramble.CpuThreaded()))
            x = randn(rng, N)
            y0 = randn(rng, N)
            y = mul!(fill(NaN, N), Ks, x)
            @test isapprox(y, A * x; rtol = 1e-12, atol = 1e-12)
            y5 = mul!(copy(y0), Ks, x, 0.5, -3.0)
            @test isapprox(y5, 0.5 * (A * x) - 3.0 * y0; rtol = 1e-12, atol = 1e-12)
            @test mul!(fill(NaN, N), Kt, x) == y
            @test mul!(copy(y0), Kt, x, 0.5, -3.0) == y5
            @test issymmetric(Ks) == issymmetric(Matrix(A))
            _kron_alloc_no_scratch(y, Ks, x)
            _kron_alloc5_no_scratch(y5, Ks, x, 0.5, -3.0)
            @test _kron_alloc_no_scratch(y, Ks, x) == 0
            @test _kron_alloc5_no_scratch(y5, Ks, x, 0.5, -3.0) == 0
            for (t, (_, f)) in zip(terms, specs), d in 1:D

                f[d] isa Diagonal && continue
                R = sparse(f[d])
                @test sparse(t.rows[d]) == (issymmetric(R) ? R : sparse(transpose(R)))
                @test !(issymmetric(R) && f[d] isa SparseMatrixCSC) || t.rows[d] === f[d]
            end
        end
    end

    # `is_separable` and `kronecker_operator` go through `_kron_project` (#427): every
    # family below factors on graded meshes, and `SparseMatrixCSC(K)` is `assemble(a)`,
    # stored pattern included. A coefficient varying along one axis is a factor too.
    @testset "Kronecker: projected forms" begin
        for n in ((9, 7), (6, 5, 7))
            Wₕ = _kron_graded_space(n, backend())
            fx = Rₕ(Wₕ, x -> 1 + x[1])
            fy = Rₕ(Wₕ, x -> 2 + x[2]^2)
            fams = [
                (u, v) -> innerₕ(D₊ₓ(u), D₊ₓ(v)),
                (u, v) -> innerₕ(Mₓ(u), Mₓ(v)),
                (u, v) -> innerₕ(jumpₓ(u), jumpₓ(v)),
                (u, v) -> innerₕ(D₋ₓ(Mₓ(u)), D₋ₓ(Mₓ(v))),
                (u, v) -> innerₕ(D₋ₓ(D₋ᵧ(u)), v),
                (u, v) -> innerₕ(D₋ₓ(u), D₋ᵧ(v)),
                (u, v) -> innerₕ(D₋ₓ(u), v) + innerₕ(u, v),
                (u, v) -> innerₕ(fx * (fy * u), v) + 0.5 * inner₊(∇ₕ(u), ∇ₕ(v)),
                (u, v) -> innerₕ(restrict_to(:interior, u), v),
                (u, v) -> inner_Γ(u, v; markers = (:xmin,)) + innerₕ(u, v)
            ]
            x = rand(MersenneTwister(KRON_SEED), ndofs(Wₕ))
            for (i, f) in enumerate(fams)
                a = form(Wₕ, Wₕ, f)
                A = assemble(a)
                @test is_separable(a)
                K = @test_logs min_level = Base.CoreLogging.Error kronecker_operator(a)
                B = SparseMatrixCSC(K)
                @test findnz(B)[1:2] == findnz(A)[1:2]
                @test maximum(abs, B - A) <= 1e-13 * maximum(abs, A)
                @test isapprox(K * x, A * x; rtol = 1e-12)
            end
        end
    end

    # Today's forms build the operator they always did: the mass factor is the axis's lazy
    # weights in a `Diagonal`, shared by every term; the stiffness factor is its own rows;
    # axis 1's line is the tridiagonal sweep. Diagonal factors that are not the mass (an
    # `:interior` restriction, a coefficient) become a `Diagonal` too.
    @testset "Kronecker: today's factor types" begin
        Wₕ = _kron_graded_space((9, 7), backend())
        K = kronecker_operator(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))))
        @test length(K.terms) == 3
        mass(t, d) = t.factors[d] isa Diagonal{Float64, <:Bramble.SeparableWeights{1}}
        @test all(d -> mass(K.terms[1], d), 1:2)
        @test K.terms[2].factors[1] isa SparseMatrixCSC && mass(K.terms[2], 2)
        @test mass(K.terms[3], 1) && K.terms[3].factors[2] isa SparseMatrixCSC
        @test all(t -> t.rows === t.factors, K.terms)
        @test K.terms[2].line isa Bramble._KronTridiag
        @test K.terms[1].factors[2] === K.terms[2].factors[2]
        @test K.terms[1].factors[1] === K.terms[3].factors[1]

        Ki = kronecker_operator(form(Wₕ, Wₕ, (u, v) -> innerₕ(restrict_to(:interior, u), v)))
        @test all(F -> F isa Diagonal{Float64, Vector{Float64}}, only(Ki.terms).factors)
    end

    # A grid-function coefficient is read once, at construction, and `kronecker_operator`
    # warns so; a scalar or `Ref` coefficient stays live and warns nothing.
    @testset "Kronecker: coefficient snapshot" begin
        Wₕ = _kron_graded_space((9, 7), backend())
        fx = Rₕ(Wₕ, x -> 1 + x[1])
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(fx * u, v))
        K = @test_logs (:warn, r"read once") kronecker_operator(a)
        x = rand(MersenneTwister(KRON_SEED), ndofs(Wₕ))
        y0 = K * x
        @test isapprox(y0, assemble(a) * x; rtol = 1e-12)
        Rₕ!(fx, x -> 5 + x[1])
        @test K * x == y0
        @test !isapprox(K * x, assemble(a) * x; rtol = 1e-3)

        c = Ref(2.0)
        ac = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + c * inner₊(∇ₕ(u), ∇ₕ(v)))
        Kc = @test_logs min_level = Base.CoreLogging.Warn kronecker_operator(ac)
        c[] = 7.0
        @test isapprox(Kc * x, assemble(ac) * x; rtol = 1e-12)
    end

    # Ruling (S2.5): wherever `assemble(a)` stores explicit zeros (a pure `inner_Γ` form on
    # the interior points of the face rows' diagonal, `inner₊` at its zero-weight points,
    # an all-zero factor on a two-point axis), `SparseMatrixCSC(K)` stores the pattern of
    # `dropzeros(assemble(a))`: the values agree, and the factors are not padded to repeat
    # the zeros.
    @testset "Kronecker: pure inner_Γ" begin
        Wₕ = _kron_graded_space((9, 7), backend())
        a = form(Wₕ, Wₕ, (u, v) -> inner_Γ(u, v; markers = (:boundary,)))
        A = assemble(a)
        B = SparseMatrixCSC(kronecker_operator(a))
        @test B == A
        @test findnz(B)[1:2] == findnz(dropzeros(A))[1:2]
        @test nnz(B) < nnz(A)
    end

    # The device kernel applies at most one non-diagonal factor per term, read as symmetric:
    # `_kron_check_device` refuses any other term on a device backend, naming the shape.
    @testset "Kronecker: device term check" begin
        T3 = sparse([2.0 -1 0; -1 2 -1; 0 -1 2])
        H4 = Diagonal(ones(4))
        @test Bramble._kron_check_device(Bramble._kron_term((), (T3, H4))) === nothing
        @test Bramble._kron_check_device(Bramble._kron_term((), (H4, T3))) === nothing
        two = Bramble._kron_term((), (T3, sparse(ones(4, 4) + 4I)))
        @test_throws ArgumentError Bramble._kron_check_device(two)
        @test_throws "non-diagonal factors on axes (1, 2)" Bramble._kron_check_device(two)
        up = Bramble._kron_term((), (H4, sparse([1.0 0 0; -1 1 0; 0 -1 1])))
        @test_throws ArgumentError Bramble._kron_check_device(up)
        @test_throws "axis-2 factor is not symmetric" Bramble._kron_check_device(up)
    end

    # The simplifier merges like terms into a constant scalar inside a side
    # (`innerₕ(D₋ₓ(u), v) + 0.3 * innerₕ(u, v)` becomes `innerₕ(D₋ₓ(u) + 0.3 * u, v)`), which
    # the projection carries on axis 1. A `Ref` there is carried beside the factors and read
    # at every product.
    @testset "Kronecker: like terms merged" begin
        for n in ((9, 7), (6, 5, 7))
            Wₕ = _kron_graded_space(n, backend())
            fx = Rₕ(Wₕ, x -> 1 + x[1])
            fams = [
                (u, v) -> innerₕ(D₋ₓ(D₋ᵧ(u)), v),
                (u, v) -> innerₕ(D₋ₓ(u), v),
                (u, v) -> innerₕ(fx * u, v) + inner₊(∇ₕ(u), ∇ₕ(v)),
                (u, v) -> innerₕ(restrict_to(:interior, u), v)
            ]
            x = rand(MersenneTwister(KRON_SEED), ndofs(Wₕ))
            for f in fams
                a = form(Wₕ, Wₕ, (u, v) -> f(u, v) + 0.3 * innerₕ(u, v))
                A = assemble(a)
                @test is_separable(a)
                K = @test_logs min_level = Base.CoreLogging.Error kronecker_operator(a)
                @test maximum(abs, SparseMatrixCSC(K) - A) <= 1e-13 * maximum(abs, A)
                @test isapprox(K * x, A * x; rtol = 1e-12)
            end
        end
        # A `Ref` advection coefficient beside the mass and stiffness terms, merged into the
        # trial side of `innerₕ` or the test side of `inner₊ₓ`.
        c = Ref(0.3)
        advection = [
            (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) + c * innerₕ(D₋ₓ(u), v),
            (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) + c * Bramble.inner₊ₓ(D₋ₓ(u), v),
            (u, v) -> innerₕ(D₋ₓ(u), v) + c * innerₕ(u, v)
        ]
        for n in ((17, 13), (6, 5, 7)), f in advection

            Wₕ = _kron_graded_space(n, backend())
            a = form(Wₕ, Wₕ, f)
            @test is_separable(a)
            K = @test_logs min_level = Base.CoreLogging.Error kronecker_operator(a)
            x = rand(MersenneTwister(KRON_SEED), ndofs(Wₕ))
            for value in (0.3, -1.7, 0.0)
                c[] = value
                A = assemble(a)
                @test maximum(abs, SparseMatrixCSC(K) - A) <= 1e-13 * maximum(abs, A)
                @test isapprox(K * x, A * x; rtol = 1e-12)
            end
            c[] = 0.3
        end
    end

    # The 1D factors are assembled on a serial host backend whatever the mesh's policy, so
    # the operator holds the same factors, and the threaded product equals the serial one
    # bit for bit, on a form whose 1D assembly would otherwise round differently.
    @testset "Kronecker: factors across policies" begin
        f = (u, v) -> innerₕ(D₋ₓ(Mₓ(u)), D₋ₓ(Mₓ(v)))
        for n in ((5, 4), (12, 9, 11))
            Ks = [kronecker_operator(form(W, W, f))
                  for W in (_kron_graded_space(n, backend(; policy = P))
            for P in (Bramble.CpuSerial(), Bramble.CpuThreaded(), Bramble.CpuPolyester()))]
            for K in Ks[2:3], (t, ts) in zip(K.terms, Ks[1].terms)

                @test all(map((F, G) -> F == G && typeof(F) === typeof(G), t.factors, ts.factors))
            end
            x = rand(MersenneTwister(KRON_SEED), size(Ks[1], 1))
            @test Ks[2] * x == Ks[1] * x
        end
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

    # The 0 B above is measured under the default (serial) policy only. `CpuSerial()` must
    # stay at 0 B; `CpuThreaded()` runs the lines in one `:static` threaded loop, whose task
    # launches cost a constant number of bytes, the same on every grid (and 0 B would mean
    # the lines did not thread); `CpuPolyester`'s are tested in test/ext/polyester_ext.jl.
    # Graded meshes, small and large, 2D and 3D, 3- and 5-argument `mul!`, each measured
    # after a warm call; the threaded product must equal the serial one bit for bit, also
    # when called from inside a user's `Threads.@threads` loop.
    @testset "Kronecker: warm bytes per policy" begin
        kform(W) = form(W, W, (u, v) -> innerₕ(u, v) + 2.5 * inner₊(∇ₕ(u), ∇ₕ(v)))
        for P in (Bramble.CpuSerial(), Bramble.CpuThreaded())
            bytes = Dict{Tuple, Int}()
            for n in ((9, 7), (257, 257), (6, 5, 4), (33, 33, 33))
                @testset "$P $(join(n, '×'))" begin
                    W = _kron_graded_space(n, backend(; policy = P))
                    a = kform(W)
                    K = kronecker_operator(a)
                    @test K.policy === P
                    Ks = kronecker_operator(kform(_kron_graded_space(n, backend())))
                    @test Ks.policy === Bramble.CpuSerial()
                    A = assemble(a)
                    N = ndofs(W)
                    x = rand(N)
                    y = similar(x)
                    y0 = rand(N)
                    s = (zeros(N), zeros(N))
                    mul!(y, K, x)
                    @test isapprox(y, A * x; rtol = 1e-12, atol = 1e-12)
                    @test y == mul!(similar(x), Ks, x)
                    y5 = copy(y0)
                    mul!(y5, K, x, 0.5, -3.0)
                    @test isapprox(y5, 0.5 * (A * x) - 3.0 * y0; rtol = 1e-12, atol = 1e-12)
                    @test y5 == mul!(copy(y0), Ks, x, 0.5, -3.0)
                    yn = [fill(NaN, N) for _ in 1:4]
                    Threads.@threads :static for k in 1:4
                        mul!(yn[k], K, x)
                    end
                    @test all(==(y), yn)
                    _kron_alloc_no_scratch(y, K, x)
                    b3 = _kron_alloc_no_scratch(y, K, x)
                    _kron_alloc_with_scratch(y, K, x, s)
                    b3s = _kron_alloc_with_scratch(y, K, x, s)
                    _kron_alloc5_no_scratch(y5, K, x, 0.5, -3.0)
                    b5 = _kron_alloc5_no_scratch(y5, K, x, 0.5, -3.0)
                    _kron_alloc5_with_scratch(y5, K, x, 0.5, -3.0, s)
                    b5s = _kron_alloc5_with_scratch(y5, K, x, 0.5, -3.0, s)
                    bytes[n] = b3
                    if P isa Bramble.CpuSerial
                        @test b3 == b3s == b5 == b5s == 0
                    elseif Threads.nthreads() > 1
                        @test b3 > 0 && b3s > 0 && b5 > 0 && b5s > 0
                    end
                end
            end
            # Constant in grid size: a large grid allocates no more than a small one.
            @test bytes[(257, 257)] <= bytes[(9, 7)]
            @test bytes[(33, 33, 33)] <= bytes[(6, 5, 4)]
        end
    end

    # gpena/Bramble.jl#442: an operator built before `change_points!` refuses every entry
    # point with the stale-weights `ArgumentError`, and a stale `mul!` leaves `y` as it was.
    # The same operator works before the mutation; one rebuilt after it matches `assemble`.
    # Hand-made terms have no mesh, so nothing makes them stale.
    @testset "Kronecker: stale after a mesh move" begin
        stale = _kron_stale
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), (false, false))
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        K = kronecker_operator(a)
        V = gridspace(Ωₕ, Val(2))
        b = form(V, V, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(D₋ₓ(u(1)), v(2)))
        KB = kronecker_operator(b)
        x = rand(MersenneTwister(KRON_SEED), size(K, 2))
        xb = rand(MersenneTwister(KRON_SEED + 1), size(KB, 2))
        @test !stale(() -> K * x) && !stale(() -> K[1, 1]) && !stale(() -> SparseMatrixCSC(K))
        @test !stale(() -> KB * xb) && !stale(() -> KB[1, 1])

        Bramble.change_points!(Ωₕ,
            (range(0.0, 1.0; length = 9) .^ 2, range(0.0, 1.0; length = 7) .^ 2))
        y0 = rand(MersenneTwister(KRON_SEED + 2), size(K, 1))
        y = copy(y0)
        @test stale(() -> mul!(y, K, x))
        @test stale(() -> mul!(y, K, x, 0.5, 1.0))
        @test y == y0
        @test stale(() -> K[1, 1])
        @test stale(() -> SparseMatrixCSC(K))
        yb0 = rand(MersenneTwister(KRON_SEED + 3), size(KB, 1))
        yb = copy(yb0)
        @test stale(() -> mul!(yb, KB, xb, 0.5, 1.0))
        @test yb == yb0
        @test stale(() -> KB[1, 1])
        @test stale(() -> SparseMatrixCSC(KB))

        W2 = gridspace(Ωₕ)
        a2 = form(W2, W2, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        @test isapprox(kronecker_operator(a2) * x, assemble(a2) * x; rtol = 1e-12)
        V2 = gridspace(Ωₕ, Val(2))
        b2 = form(V2, V2, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(D₋ₓ(u(1)), v(2)))
        @test isapprox(kronecker_operator(b2) * xb, assemble(b2) * xb; rtol = 1e-12)

        t = K.terms[1]
        Kh = KroneckerLinearOperator{Float64, 2, Tuple{typeof(t)}}((t,), K.dims, K.n)
        @test Kh.mesh === nothing
        @test !stale(() -> Kh * x) && !stale(() -> Kh[1, 1])
    end

    # A refinement changes the sizes too: vectors and indices sized for the refined mesh
    # still get the stale error from `mul!` and `getindex`, not one that hides the cause. Displaying
    # a stale operator prints its summary and says it is stale instead of throwing.
    @testset "Kronecker: stale after refinement" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (false, false))
        Wₕ = gridspace(Ωₕ)
        K = kronecker_operator(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))))
        V = gridspace(Ωₕ, Val(2))
        KB = kronecker_operator(form(V, V,
            (u, v) -> innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2)))))
        txt(A) = sprint(show, MIME"text/plain"(), A)
        # Fresh: the default matrix display, a summary line then rows of entries.
        for A in (K, KB)
            @test startswith(txt(A), summary(A)) && !occursin("stale", txt(A))
            @test countlines(IOBuffer(txt(A))) > 3
        end

        Bramble.iterative_refinement!(Ωₕ)
        n, nb = ndofs(gridspace(Ωₕ)), ndofs(gridspace(Ωₕ, Val(2)))
        @test n > size(K, 1) && nb > size(KB, 1)
        x, y = rand(MersenneTwister(KRON_SEED), n), zeros(n)
        xb, yb = rand(MersenneTwister(KRON_SEED + 1), nb), zeros(nb)
        for (A, u, v, m) in ((K, x, y, n), (KB, xb, yb, nb))
            @test _kron_stale(() -> mul!(v, A, u))
            @test _kron_stale(() -> mul!(v, A, u, 0.5, 1.0))
            # `LinearAlgebra`'s `*` checks sizes before our `mul!` runs: it throws, but
            # names the size mismatch (no `*` method of ours, see `kronecker.jl`).
            @test_throws DimensionMismatch A * u
            @test _kron_stale(() -> A[m, m])
            @test !_kron_stale(() -> txt(A))
            @test startswith(txt(A), summary(A)) && occursin("stale", txt(A))
        end
    end

    # A form whose spaces predate a mesh mutation is refused as `assemble` refuses it: the
    # space's own stale-weights error, from `kronecker_operator` and `is_separable` alike.
    @testset "Kronecker: stale spaces refused" begin
        space_error = "gridspace(mesh(Wₕ)) again"
        for mutate! in (Ω -> Bramble.change_points!(Ω,
            (range(0.0, 1.0; length = 7) .^ 2, range(0.0, 1.0; length = 6) .^ 2)),
            Bramble.iterative_refinement!)
            Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (false, false))
            Wₕ = gridspace(Ωₕ)
            V = gridspace(Ωₕ, Val(2))
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
            b = form(V, V, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(D₋ₓ(u(1)), v(2)))
            @test is_separable(a) && is_separable(b)
            mutate!(Ωₕ)
            for f in (a, b)
                @test _kron_stale(() -> assemble(f), space_error)
                @test _kron_stale(() -> kronecker_operator(f), space_error)
                @test _kron_stale(() -> is_separable(f), space_error)
            end
        end
    end

    # `kronecker_operator` is `@noinline`, so a caller compiles one call to it (cached once by
    # the precompile workload) instead of the whole build; the product is unchanged.
    @testset "kronecker_operator stays a call" begin
        for n in ((9, 8), (6, 5, 7))
            Wₕ = _kron_graded_space(n, backend())
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
            x = rand(MersenneTwister(KRON_SEED), ndofs(Wₕ))
            y = similar(x)
            ci = first(only(code_typed(_kron_call!, (typeof(y), typeof(a), typeof(x)))))
            @test any(c -> occursin("kronecker_operator(", c), _kron_invokes(ci))
            _kron_call!(y, a, x)
            @test isapprox(y, assemble(a) * x; rtol = 1e-12)
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
