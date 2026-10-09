module SpaceStarVectorCalculusTests

using Test
using Bramble
using Bramble: ∇̃ₕ!, diṽₕ!, curl̃ₕ!
using Bramble: D̃
using Bramble: D₋ₓ
using Random
using Bramble: D₊, M₊ₕ, ε₊ₕ, ε₊ₕ!
using ..TestUtils: alloc_test, @test_allocs
using ..TestUtils: _bubble, _composite, _field, _fixture

# Built on the averaged-spacing difference D̃.
@testset "Averaged-spacing vector calculus (#287)" begin
    Random.seed!(287)

    @testset "∇̃ₕ ($(D)D)" for D in 1:3
        _, _, u, _ = _fixture(D)
        g = ∇̃ₕ(u[1])
        dest = D == 1 ? similar(u[1]) : ntuple(_ -> similar(u[1]), D)
        ∇̃ₕ!(dest, u[1])
        if D == 1
            @test parent(g) == parent(D̃(u[1], 1))
            @test parent(dest) == parent(g)
        else
            @test all(parent(g[d]) == parent(D̃(u[1], d)) for d in 1:D)
            @test all(parent(dest[d]) == parent(g[d]) for d in 1:D)
        end
        @test_allocs ∇̃ₕ!(dest, u[1])
    end

    @testset "diṽₕ ($(D)D)" for D in 1:3
        Ωₕ, _, u, _ = _fixture(D)
        oracle = sum(parent(D̃(u[d], d)) for d in 1:D)
        F = _field(u, D)
        @test parent(diṽₕ(F)) ≈ oracle
        v = similar(u[1])
        diṽₕ!(v, F)
        @test parent(v) ≈ oracle
        @test_allocs diṽₕ!(v, F)
        if D > 1
            uc = _composite(Ωₕ, u, D)
            @test parent(diṽₕ(uc)) ≈ oracle
            @test_allocs diṽₕ!(v, uc)
        end
        @test_throws DimensionMismatch diṽₕ(ntuple(_ -> u[1], D + 1))
    end

    @testset "curl̃ₕ ($(D)D)" for D in 1:3
        Ωₕ, _, u, _ = _fixture(D)
        ds(k, d) = parent(D̃(u[k], d))
        if D == 1
            @test_throws ArgumentError curl̃ₕ(u)
        elseif D == 2
            oracle = ds(2, 1) .- ds(1, 2)
            @test parent(curl̃ₕ(u)) ≈ oracle
            @test parent(curl̃ₕ(_composite(Ωₕ, u, D))) ≈ oracle
            c = similar(u[1])
            curl̃ₕ!(c, u)
            @test parent(c) ≈ oracle
            @test_allocs curl̃ₕ!(c, u)
        else
            oracle = (ds(3, 2) .- ds(2, 3), ds(1, 3) .- ds(3, 1), ds(2, 1) .- ds(1, 2))
            cc = curl̃ₕ(u)
            @test all(parent(cc[k]) ≈ oracle[k] for k in 1:3)
            c = ntuple(_ -> similar(u[1]), 3)
            curl̃ₕ!(c, _composite(Ωₕ, u, D))
            @test all(parent(c[k]) ≈ oracle[k] for k in 1:3)
            @test_allocs curl̃ₕ!(c, u)
        end
    end

    @testset "ε₊ₕ ($(D)D)" for D in 1:3
        Ωₕ, _, u, _ = _fixture(D)
        F = _field(u, D)
        ε = ε₊ₕ(F)
        for i in 1:D, j in 1:D

            oracle = i == j ? parent(D₊(u[i], i)) :
                     (parent(M₊ₕ(D₊(u[i], j), i)) .+ parent(M₊ₕ(D₊(u[j], i), j))) ./ 2
            @test parent(ε[i][j]) ≈ oracle
            @test parent(ε[i][j]) == parent(ε[j][i])
        end
        dest = ntuple(_ -> ntuple(_ -> similar(u[1]), D), D)
        ε₊ₕ!(dest, F)
        @test all(parent(dest[i][j]) == parent(ε[i][j]) for i in 1:D, j in 1:D)
        @test_allocs ε₊ₕ!(dest, F)
        if D > 1
            εc = ε₊ₕ(_composite(Ωₕ, u, D))
            @test all(parent(εc[i][j]) == parent(ε[i][j]) for i in 1:D, j in 1:D)
        end
    end

    # Summation by parts in 1D: innerₕ(diṽₕ(u), v) = -inner₊(u, D₋ₓ(v)) when the fields
    # vanish on the boundary, on a non-uniform mesh. Without that the boundary term is left
    # over, so the identity must fail.
    @testset "SBP duality (1D)" begin
        _, _, u, dims = _fixture(1)
        w = Rₕ(space(u[1]), x -> cos(5x) + 0.4)
        ub, wb = _bubble(u[1], dims), _bubble(w, dims)
        @test isapprox(innerₕ(diṽₕ(ub), wb), -inner₊(ub, D₋ₓ(wb)); atol = 1e-12)
        @test !isapprox(innerₕ(diṽₕ(u[1]), w), -inner₊(u[1], D₋ₓ(w)); atol = 1e-6)
    end
end

end # module SpaceStarVectorCalculusTests
