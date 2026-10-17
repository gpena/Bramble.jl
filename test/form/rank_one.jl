module FormRankOneTests

using Test
using Bramble
using Bramble: RankOneUpdate, BilinearForm, is_uniform, index_in_marker, D₋ₓ
using LinearAlgebra, SparseArrays, Random
using ..TestUtils: WITH_3D_TESTS, alloc_test

# Rank-one penalty terms `σ * innerₕ(g, u) * innerₕ(h, v)` (#780).
#
# A form holding one assembles to a `RankOneUpdate`: the matrix of its other terms, `σ` and
# the two functionals' vectors. The references share no code with that path: the other
# terms and the two functionals are assembled as forms of their own, and the matrix is
# their dense sum `A + σ wₛ wₜᵀ`. The trial functional `g` and the test functional `h`
# differ, so a swap of the two vectors changes the answer. Every mesh is non-uniform and
# seeded.

const DIMS = WITH_3D_TESTS ? (1, 2, 3) : (1, 2)
const SIZES = ((9,), (7, 9), (5, 6, 4))

function nonuniform_space(D, T; seed = 780)
    Random.seed!(seed)
    I = interval(zero(T), one(T))
    Ω = D == 1 ? domain(I) : D == 2 ? domain(I × I) : domain(I × I × I)
    n = D == 1 ? SIZES[1][1] : SIZES[D]
    Ωₕ = mesh(Ω, n, D == 1 ? false : ntuple(_ -> false, D); backend = backend(T))
    return gridspace(Ωₕ)
end

lap(u, v) = inner₊(∇ₕ(u), ∇ₕ(v))

# The trial and test data `g` and `h` in each of their three kinds.
function data_pairs(W)
    gₕ = element(W)
    parent(gₕ) .= 1 .+ rand(length(gₕ))
    return (("constant", 2, x -> 2 - x[1]),
        ("function", x -> 1 + x[1], x -> 2 - x[1]),
        ("grid function", gₕ, 1))
end

functional(W, g) = assemble(form(W, v -> innerₕ(g, v)))

# A scalar trial space and a two-component test space over it, for a rectangular form.
function rectangular_spaces()
    W = nonuniform_space(2, Float64)
    return W, W × W
end

rectangular(u, v) = innerₕ(u, v(1)) + 2.0 * innerₕ(x -> x[1], u) * innerₕ(1, v(2))

@testset "Rank-one terms" begin
    @testset "values against a dense reference" begin
        for D in DIMS, T in (Float64, Float32)
            W = nonuniform_space(D, T)
            @test !is_uniform(mesh(W))
            Ωₕ = mesh(W)
            σ = 3.0 / hₘₐₓ(Ωₕ)
            A0 = assemble(form(W, W, lap))
            for (name, g, h) in data_pairs(W)
                @testset "$(D)D $T, g a $name" begin
                    a = form(W, W, (u, v) -> lap(u, v) + σ * innerₕ(g, u) * innerₕ(h, v))
                    K = assemble(a)
                    @test K isa RankOneUpdate
                    wt, ws = functional(W, g), functional(W, h)
                    # each part is what its own form assembles, bit for bit
                    @test K.A == A0 && typeof(K.A) == typeof(A0)
                    @test K.ws == ws && K.wt == wt
                    @test K.σ == σ
                    @test eltype(K) == promote_type(eltype(A0), typeof(σ), T)
                    M = Matrix(A0) + σ * ws * wt'
                    @test Matrix(K) ≈ M
                    # the transposed placement differs: the two vectors are not parallel
                    @test !(Matrix(K) ≈ Matrix(A0) + σ * wt * ws')
                    i, j = size(K, 1), 1
                    @test K[i, j] == A0[i, j] + σ * (ws[i] * wt[j])
                    @test Matrix(K) == [K[p, q] for p in axes(K, 1), q in axes(K, 2)]
                    @test Array(K) == Matrix(K)
                    @test size(K) == size(A0)
                end
            end
        end
    end

    @testset "where the scalars sit" begin
        W = nonuniform_space(2, Float64)
        A0 = Matrix(assemble(form(W, W, lap)))
        g = x -> 1 + x[1]
        wt, ws = functional(W, g), functional(W, 1)
        ref(σ) = A0 + σ * ws * wt'
        forms = (("canonical", 2.5, (u, v) -> lap(u, v) + 2.5 * innerₕ(g, u) * innerₕ(1, v)),
            ("test factor first", 2.5,
                (u, v) -> lap(u, v) + innerₕ(1, v) * (2.5 * innerₕ(g, u))),
            ("scale on the right", 2.5,
                (u, v) -> lap(u, v) + innerₕ(g, u) * innerₕ(1, v) * 2.5),
            ("a factor on each side", 6.0,
                (u, v) -> lap(u, v) + (2.0 * innerₕ(g, u)) * (3 * innerₕ(1, v))),
            ("no scale", 1, (u, v) -> lap(u, v) + innerₕ(g, u) * innerₕ(1, v)),
            ("subtracted", -2.5,
                (u, v) -> lap(u, v) - 2.5 * innerₕ(g, u) * innerₕ(1, v)),
            ("negated", -1, (u, v) -> lap(u, v) + -(innerₕ(g, u) * innerₕ(1, v))),
            ("scaled sum", 4.0,
                (u, v) -> 2 * (lap(u, v) + 2.0 * innerₕ(g, u) * innerₕ(1, v))),
            ("term first", 2.5, (u, v) -> 2.5 * innerₕ(g, u) * innerₕ(1, v) + lap(u, v)),
            ("term in the middle", 2.5,
                (u, v) -> 0.5 * lap(u, v) + 2.5 * innerₕ(g, u) * innerₕ(1, v) + 0.5 * lap(u, v)))
        for (name, σ, f) in forms
            @testset "$name" begin
                a = form(W, W, f)
                K = assemble(a)
                @test K.σ == σ
                # a scaled sum scales the other terms too
                A = name == "scaled sum" ? 2 * A0 : A0
                @test Matrix(K) ≈ A + σ * ws * wt'
            end
        end
    end

    @testset "functionals of operands" begin
        W = nonuniform_space(2, Float64)
        A0 = Matrix(assemble(form(W, W, lap)))
        g = x -> 1 + x[2]
        # a functional is any sum of `innerₕ`/`inner₊` source products of one argument
        a = form(W, W,
            (u, v) -> lap(u, v) +
                      2.0 * (innerₕ(g, D₋ₓ(u)) + 0.5 * innerₕ(1, u)) *
                      innerₕ(1, v; markers = (:boundary,)))
        K = assemble(a)
        wt = assemble(form(W, v -> innerₕ(g, D₋ₓ(v)) + 0.5 * innerₕ(1, v)))
        ws = assemble(form(W, v -> innerₕ(1, v; markers = (:boundary,))))
        @test count(!iszero, ws) < length(ws)
        @test Matrix(K) ≈ A0 + 2.0 * ws * wt'
    end

    @testset "a Ref σ is read at each assembly" begin
        W = nonuniform_space(2, Float64)
        A0 = Matrix(assemble(form(W, W, lap)))
        w = functional(W, 1)
        r = Ref(2.0)
        for (name, f, σ) in (("inside", (u, v) -> lap(u, v) + 3.0 * (r * innerₕ(1, u)) * innerₕ(1, v), () -> 3.0 * r[]),
            ("outside", (u, v) -> lap(u, v) + r * (innerₕ(1, u) * innerₕ(1, v)), () -> r[]))
            @testset "$name" begin
                r[] = 2.0
                a = form(W, W, f)
                K = assemble(a)
                @test K.σ == σ()
                r[] = 5.0
                assemble!(K, a)
                @test K.σ == σ() == (name == "inside" ? 15.0 : 5.0)
                @test Matrix(K) ≈ A0 + σ() * w * w'
            end
        end
    end

    @testset "grid function data is read live" begin
        W = nonuniform_space(2, Float64)
        gₕ = element(W)
        parent(gₕ) .= 1 .+ rand(length(gₕ))
        a = form(W, W, (u, v) -> lap(u, v) + 2.0 * innerₕ(gₕ, u) * innerₕ(1, v))
        K = assemble(a)
        before = copy(K.wt)
        parent(gₕ) .*= 3
        assemble!(K, a)
        @test K.wt ≈ 3 * before
        @test K.wt == functional(W, gₕ)
    end

    @testset "warm refills allocate nothing" begin
        for D in DIMS, T in (Float64, Float32)
            W = nonuniform_space(D, T)
            gₕ = element(W)
            parent(gₕ) .= 1 .+ rand(T, length(gₕ))
            r = Ref(T(2))
            for (name, f) in (("constant", (u, v) -> lap(u, v) + 2 * innerₕ(1, u) * innerₕ(1, v)),
                ("function", (u, v) -> lap(u, v) + innerₕ(x -> 1 + x[1], u) * innerₕ(2, v)),
                ("grid function", (u, v) -> lap(u, v) + r * innerₕ(gₕ, u) * innerₕ(gₕ, v)))
                @testset "$(D)D $T, $name" begin
                    a = form(W, W, f)
                    K = assemble(a)
                    @test alloc_test(assemble!, K, a) == 0
                    @test alloc_test(assemble!, K, a; dirichlet = :boundary) == 0
                    x = rand(eltype(K), size(K, 2))
                    y = similar(x)
                    @test alloc_test(mul!, y, K, x) == 0
                    @test alloc_test(mul!, y, K, x, 2.0, 0.5) == 0
                    X = rand(eltype(K), size(K, 2), 3)
                    Y = similar(X)
                    @test alloc_test(mul!, Y, K, X, 2.0, 0.5) == 0
                end
            end
        end
    end

    @testset "products, adjoint and transpose" begin
        W = nonuniform_space(2, Float64)
        a = form(W, W, (u, v) -> innerₕ(D₋ₓ(u), v) + 1.5 * innerₕ(x -> x[1], u) * innerₕ(1, v))
        K = assemble(a)
        M = Matrix(K)
        @test !issymmetric(M)
        x = rand(size(K, 2))
        y0 = rand(size(K, 1))
        @test K * x ≈ M * x
        @test mul!(copy(y0), K, x, 2.0, 0.5) ≈ 2.0 * M * x + 0.5 * y0
        X = rand(size(K, 2), 3)
        Y0 = rand(size(K, 1), 3)
        @test K * X ≈ M * X
        @test mul!(copy(Y0), K, X, 2.0, 0.5) ≈ 2.0 * M * X + 0.5 * Y0
        @test K' isa RankOneUpdate && transpose(K) isa RankOneUpdate
        @test Matrix(K') ≈ M' && Matrix(transpose(K)) ≈ transpose(M)
        @test K' * x ≈ M' * x
        @test transpose(K) * x ≈ transpose(M) * x
        # the adjoint shares the parts: a refill of `K` is seen through it
        Kt = K'
        @test Kt.ws === K.wt && Kt.wt === K.ws
        # `α = 0` reads no `x`, and `β = 0` no `y`, as `mul!` with a matrix does
        xinf = fill(Inf, size(K, 2))
        y1 = rand(size(K, 1))
        @test mul!(copy(y1), K, xinf, 0.0, 1.0) == y1
        @test mul!(copy(y1), K, xinf, 0.0, 2.0) == 2 * y1
        @test iszero(mul!(fill(NaN, size(K, 1)), K, xinf, 0.0, 0.0))
        @test mul!(fill(NaN, size(K, 1)), K, x, 1.0, 0.0) ≈ M * x
        Xinf = fill(Inf, size(K, 2), 2)
        Y1 = rand(size(K, 1), 2)
        @test mul!(copy(Y1), K, Xinf, 0.0, 1.0) == Y1
        @test iszero(mul!(fill(NaN, size(K, 1), 2), K, Xinf, 0.0, 0.0))
        @test mul!(fill(NaN, size(K, 1), 3), K, X, 1.0, 0.0) ≈ M * X
        @test_throws DimensionMismatch mul!(rand(size(K, 1)), K, rand(size(K, 2) + 1), 1, 0)
        @test_throws DimensionMismatch mul!(rand(size(K, 1), 2), K, rand(size(K, 2) + 1, 2), 1, 0)
    end

    @testset "complex data and σ" begin
        W = nonuniform_space(2, Float64)
        g = x -> 1 + im * x[1]
        σ = 1.5 - 2.0im
        a = form(W, W, (u, v) -> lap(u, v) + σ * innerₕ(g, u) * innerₕ(x -> x[2] - im, v))
        K = assemble(a)
        wt = functional(W, g)
        ws = functional(W, x -> x[2] - im)
        @test eltype(wt) <: Complex && eltype(ws) <: Complex
        M = Matrix(assemble(form(W, W, lap))) + σ * ws * transpose(wt)
        @test Matrix(K) ≈ M
        x = rand(ComplexF64, size(K, 2))
        @test K * x ≈ M * x
        @test K' * x ≈ M' * x
        @test Matrix(adjoint(K)) ≈ M'
        @test Matrix(transpose(K)) ≈ transpose(M)
    end

    @testset "trial and test spaces differ" begin
        Wu, Wv = rectangular_spaces()
        K = assemble(form(Wu, Wv, rectangular))
        A0 = Matrix(assemble(form(Wu, Wv, (u, v) -> innerₕ(u, v(1)))))
        wt = functional(Wu, x -> x[1])
        ws = assemble(form(Wv, v -> innerₕ(1, v(2))))
        @test size(K) == (ndofs(Wv), ndofs(Wu)) == (length(ws), length(wt))
        @test size(K, 1) != size(K, 2)
        @test Matrix(K) ≈ A0 + 2.0 * ws * wt'
        @test !issymmetric(K)
        a = form(Wu, Wv, rectangular)
        @test Bramble.trial_space(a) === Wu && Bramble.test_space(a) === Wv
        @test transpose(K) * ones(size(K, 1)) ≈ transpose(Matrix(K)) * ones(size(K, 1))
    end

    @testset "a term alone" begin
        W = nonuniform_space(1, Float64)
        a = form(W, W, (u, v) -> 4.0 * innerₕ(1, u) * innerₕ(1, v))
        K = assemble(a)
        w = functional(W, 1)
        @test nnz(K.A) == 0
        @test Matrix(K) ≈ 4.0 * w * w'
        @test alloc_test(assemble!, K, a) == 0
    end

    @testset "composite spaces" begin
        W = nonuniform_space(2, Float64)
        V = W × W
        a = form(V, V,
            (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)) +
                      2.0 * innerₕ(x -> x[1], u(1)) * innerₕ(1, v(2)))
        K = assemble(a)
        A0 = Matrix(assemble(form(V, V, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))))
        wt = assemble(form(V, v -> innerₕ(x -> x[1], v(1))))
        ws = assemble(form(V, v -> innerₕ(1, v(2))))
        n = ndofs(W)
        @test iszero(wt[(n + 1):end]) && iszero(ws[1:n])
        @test Matrix(K) ≈ A0 + 2.0 * ws * wt'
        # a trial functional of the second component, a test functional of the first
        a2 = form(V, V,
            (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)) +
                      2.0 * innerₕ(x -> x[1], u(2)) * innerₕ(1, v(1)))
        wt2 = assemble(form(V, v -> innerₕ(x -> x[1], v(2))))
        ws2 = assemble(form(V, v -> innerₕ(1, v(1))))
        @test iszero(wt2[1:n]) && iszero(ws2[(n + 1):end])
        @test Matrix(assemble(a2)) ≈ A0 + 2.0 * ws2 * wt2'
        Kd = assemble(a; dirichlet = :boundary, dirichlet_components = 2)
        D2 = n .+ findall(index_in_marker(mesh(W), :boundary))
        @test iszero(Kd.ws[D2]) && !iszero(Kd.ws)
        Md = Matrix(Kd)
        @test all(Md[i, :] == (1:(2n) .== i) for i in D2)
    end

    @testset "Dirichlet rows" begin
        for D in DIMS
            W = nonuniform_space(D, Float64)
            Ωₕ = mesh(W)
            σ = 3.0 / hₘₐₓ(Ωₕ)
            a = form(W, W, (u, v) -> lap(u, v) + σ * innerₕ(x -> 1 + x[1], u) * innerₕ(2, v))
            K = assemble(a; dirichlet = :boundary)
            bd = findall(index_in_marker(Ωₕ, :boundary))
            N = ndofs(W)
            M = Matrix(assemble(form(W, W, lap))) +
                σ * functional(W, 2) * functional(W, x -> 1 + x[1])'
            # the reference: identity rows on Γ_D, every column kept
            Mref = copy(M)
            for i in bd
                Mref[i, :] .= 0
                Mref[i, i] = 1
            end
            @testset "$(D)D" begin
                @test Matrix(K) ≈ Mref
                @test all(Matrix(K)[i, :] == (1:N .== i) for i in bd)
                @test iszero(K.ws[bd])
                inner = setdiff(1:N, bd)
                @test Matrix(K)[inner, bd] ≈ M[inner, bd]
                # a refill keeps the identity rows; one without `dirichlet` restores them
                fill!(K.ws, NaN)
                assemble!(K, a; dirichlet = :boundary)
                @test Matrix(K) ≈ Mref
                assemble!(K, a)
                @test Matrix(K) ≈ M
            end
        end
    end

    @testset "symmetrize!" begin
        W = nonuniform_space(2, Float64)
        Ωₕ = mesh(W)
        σ = 2.0 / hₘₐₓ(Ωₕ)
        gD = x -> 1 + x[1] * x[2]
        a = form(W, W, (u, v) -> lap(u, v) + σ * innerₕ(1, u) * innerₕ(1, v))
        l = form(W, v -> innerₕ(x -> x[1] + 1, v))
        K0, F0 = assemble(a, l; dirichlet = :boundary => gD)
        K, F = assemble(a, l; dirichlet = :boundary => gD, symmetrize = true)
        @test !issymmetric(K0) && issymmetric(K)
        @test issymmetric(Matrix(K))
        bd = findall(index_in_marker(Ωₕ, :boundary))
        @test iszero(K.wt[bd]) && iszero(K.ws[bd])
        # the reference: the dense system with identity Dirichlet rows, solved
        w = functional(W, 1)
        M = Matrix(assemble(form(W, W, lap))) + σ * w * w'
        Fr = assemble(l)
        for i in bd
            M[i, :] .= 0
            M[i, i] = 1
            Fr[i] = gD(Bramble.point(Ωₕ, Bramble.indices(Ωₕ)[i]))
        end
        @test F0 ≈ Fr
        @test Matrix(K) \ F ≈ M \ Fr
        @test Matrix(K0) \ F0 ≈ M \ Fr
        @test_throws ArgumentError assemble(a, l; symmetrize = true)
        # a bad `components` is refused before anything is written
        K1, F1 = assemble(a, l; dirichlet = :boundary => gD)
        wt1, F1c = copy(K1.wt), copy(F1)
        @test_throws ArgumentError symmetrize!(K1, F1, W, :boundary; components = 2)
        @test K1.wt == wt1 && F1 == F1c
        Wu, Wv = rectangular_spaces()
        Kr = assemble(form(Wu, Wv, rectangular))
        @test_throws ArgumentError symmetrize!(Kr, zeros(ndofs(Wv)), Wv, :boundary)
    end

    @testset "symmetrize! on a composite space" begin
        W = nonuniform_space(2, Float64)
        V = W × W
        gD = x -> 1 + x[1] * x[2]
        a = form(V, V,
            (u, v) -> lap(u(1), v(1)) + lap(u(2), v(2)) + innerₕ(u(2), v(2)) +
                      3.0 * innerₕ(1, u(1)) * innerₕ(1, v(1)))
        l = form(V, v -> innerₕ(x -> x[1] + 1, v(1)) + innerₕ(1, v(2)))
        bc = :boundary => gD
        K0, F0 = assemble(a, l; dirichlet = bc, dirichlet_components = 1)
        K, F = assemble(a, l; dirichlet = bc, dirichlet_components = 1, symmetrize = true)
        @test !issymmetric(Matrix(K0)) && issymmetric(Matrix(K)) && issymmetric(K)
        bd = findall(index_in_marker(mesh(W), :boundary))
        @test iszero(K.wt[bd]) && !iszero(K.wt[setdiff(axes(K.wt, 1), bd)])
        # the same solution, the eliminated columns moved into the load
        @test Matrix(K) \ F ≈ Matrix(K0) \ F0
    end

    @testset "issymmetric" begin
        W = nonuniform_space(2, Float64)
        sym(f) = issymmetric(assemble(form(W, W, f)))
        @test sym((u, v) -> lap(u, v) + 2.0 * innerₕ(1, u) * innerₕ(1, v))
        @test !sym((u, v) -> lap(u, v) + 2.0 * innerₕ(x -> x[1], u) * innerₕ(1, v))
        @test sym((u, v) -> lap(u, v) + 0.0 * innerₕ(x -> x[1], u) * innerₕ(1, v))
        @test !sym((u, v) -> innerₕ(D₋ₓ(u), v) + 2.0 * innerₕ(1, u) * innerₕ(1, v))
    end

    @testset "similar and copy" begin
        W = nonuniform_space(2, Float64)
        a = form(W, W, (u, v) -> lap(u, v) + 2.0 * innerₕ(x -> x[1], u) * innerₕ(1, v))
        K = assemble(a)
        S = similar(K)
        @test S isa RankOneUpdate{Float64} && size(S) == size(K)
        @test S.A !== K.A && S.ws !== K.ws && S.wt !== K.wt
        assemble!(S, a)
        @test Matrix(S) == Matrix(K)
        @test similar(K, Float32) isa RankOneUpdate{Float32}
        C = copy(K)
        @test Matrix(C) == Matrix(K)
        C.ws .= 0
        C.wt .= 0
        nonzeros(C.A) .= 0
        @test !iszero(K.ws) && !iszero(K.wt) && !iszero(nonzeros(K.A))
        @test_throws DimensionMismatch RankOneUpdate(K.A, 1.0, K.ws[1:(end - 1)], K.wt)
        @test_throws DimensionMismatch RankOneUpdate(K.A, 1.0, K.ws, K.wt[1:(end - 1)])
        @test_throws BoundsError K[size(K, 1) + 1, 1]
        @test_throws BoundsError K[1, 0]
        @test RankOneUpdate(K.A, 1.0f0, K.ws, K.wt) isa RankOneUpdate{Float64}
    end

    @testset "show" begin
        W = nonuniform_space(2, Float64)
        a = form(W, W, (u, v) -> lap(u, v) + 2.0 * innerₕ(1, u) * innerₕ(1, v))
        K = assemble(a)
        @test sprint(show, K) == "RankOneUpdate{Float64}(63×63 A + σ wₛ wₜᵀ, σ = 2.0)"
        text = sprint(show, MIME"text/plain"(), K)
        @test occursin("63×63 RankOneUpdate{Float64}: A + σ wₛ wₜᵀ", text)
        @test occursin("SparseMatrixCSC", text) && countlines(IOBuffer(text)) == 5
        @test sprint(show, a) == "RankOneForm{2D, 63×63}"
        @test startswith(sprint(show, MIME"text/plain"(), a),
            "RankOneForm: the bilinear form below + σ wₛ wₜᵀ, σ = 2.0\nBilinearForm")
    end

    @testset "forms without the term" begin
        W = nonuniform_space(2, Float64)
        a = @inferred form(W, W, lap)
        @test a isa BilinearForm
        @test assemble(a) isa SparseMatrixCSC
        @test functional(W, 1) ≈ assemble(form(W, v -> innerₕ(x -> 1.0, v)))
    end

    @testset "refusals" begin
        W = nonuniform_space(2, Float64)
        msg(f) = sprint(showerror, (@test_throws ArgumentError f()).value)
        fails(f) = msg(() -> form(W, W, f))
        @test occursin("two functionals of the trial function",
            fails((u, v) -> innerₕ(1, u) * innerₕ(1, u)))
        @test occursin("two functionals of the test function",
            fails((u, v) -> innerₕ(1, v) * innerₕ(1, v)))
        @test occursin("two functionals of the test function",
            msg(() -> form(W, v -> innerₕ(1, v) * innerₕ(2, v))))
        @test occursin("left factor, a BilinearProduct",
            fails((u, v) -> innerₕ(u, v) * innerₕ(1, v)))
        @test occursin("right factor, a BilinearProduct",
            fails((u, v) -> innerₕ(1, u) * innerₕ(u, v)))
        @test occursin("left factor, a TrialFunction", fails((u, v) -> u * v))
        @test occursin("left factor, a OperatorAdd",
            fails((u, v) -> (innerₕ(1, u) + innerₕ(1, v)) * innerₕ(1, v)))
        @test occursin("left factor, a GridFunctionScale",
            fails((u, v) -> (x -> 2.0) * innerₕ(1, u) * innerₕ(1, v)))
        @test occursin("left factor, a LinearProduct",
            fails((u, v) -> innerₕ(1, u + v) * innerₕ(1, v)))
        @test occursin("left factor, a LinearProduct",
            fails((u, v) -> innerₕ(1, innerₕ(u, u)) * innerₕ(1, v)))
        @test occursin("right factor, a RankOneTerm",
            fails((u, v) -> innerₕ(1, u) * (innerₕ(1, u) * innerₕ(1, v))))
        @test occursin("2 rank-one terms",
            fails((u, v) -> innerₕ(1, u) * innerₕ(1, v) + 2.0 * innerₕ(2, u) * innerₕ(1, v)))
        @test occursin("inside a BackwardDifference",
            fails((u, v) -> D₋ₓ(innerₕ(1, u) * innerₕ(1, v))))
        gₕ = element(W)
        @test occursin("inside a GridFunctionScale",
            fails((u, v) -> gₕ * (innerₕ(1, u) * innerₕ(1, v))))
        lone = "functional of one of them alone"
        @test occursin(lone, fails((u, v) -> lap(u, v) + innerₕ(1, u)))
        @test occursin(lone, fails((u, v) -> lap(u, v) + innerₕ(1, v)))
        a = form(W, W, (u, v) -> lap(u, v) + innerₕ(1, u) * innerₕ(1, v))
        K = assemble(a)
        target = "assembles into the RankOneUpdate"
        @test occursin(target, msg(() -> assemble!(copy(K.A), a)))
        @test occursin(target, msg(() -> assemble!(K, form(W, W, lap))))
        unsupported = "does not take a form holding a rank-one term"
        @test occursin(unsupported, msg(() -> assemble_add!(copy(K.A), a)))
        @test occursin(unsupported, msg(() -> assemble_add!(copy(K.A), a, 0.5)))
        @test occursin(unsupported, msg(() -> Bramble.allocate_system_matrix(a)))
        @test occursin(unsupported, msg(() -> matrix_free_operator(a)))
        l = form(W, v -> innerₕ(1, v))
        nosd = "the term has no semidiscretisation"
        @test occursin(nosd, msg(() -> semidiscretize(a, l)))
        @test occursin(nosd, msg(() -> semidiscretize(t -> (a, t -> nothing), l)))
        @test occursin(nosd, msg(() -> semidiscretize(form(W, W, lap), l; mass = a)))
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 9), (false, false))
        C = constrained(gridspace(Ωₕ; ghosts = (:xmin,)),
            u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u => 0.0, :ymin => u => 0.0, :ymax => u => 0.0))
        f = (u, v) -> lap(u, v) + innerₕ(1, u) * innerₕ(1, v)
        ghost = "not supported on ghost spaces"
        @test occursin(ghost, msg(() -> form(C, C, f)))
        P = Bramble.padded_space(C)
        Wp = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)),
            npoints(mesh(P), Tuple), (false, false)))
        @test occursin(ghost, msg(() -> form(P, Wp, f)))
        @test occursin(ghost, msg(() -> form(Wp, P, f)))
    end
end

end
