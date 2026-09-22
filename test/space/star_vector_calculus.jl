module SpaceStarVectorCalculusTests

using Test
using Bramble
using Random
using Bramble: components, D₊, M₊ₕ, ε₊ₕ, ε₊ₕ!
using ..TestUtils: alloc_test, _zero_boundary!, @test_allocs

# Random non-uniform meshes on the unit cube in 1D, 2D and 3D, and one smooth field per
# spatial dimension on each.
function _fixture(D)
    dom = D == 1 ? domain(interval(0.0, 1.0)) :
          D == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 1.0)) :
          domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    n = (11, 9, 7)
    Ωₕ = D == 1 ? mesh(dom, n[1], false) : mesh(dom, n[1:D], ntuple(_ -> false, D))
    Wₕ = gridspace(Ωₕ)
    fs = (x -> sin(3x[1]) + (D > 1 ? x[2]^2 : 0.0) + (D > 2 ? x[3] : 0.0),
        x -> cos(2x[D]) + x[1]^3,
        x -> x[1] * x[D] + sin(x[1]))
    u = ntuple(k -> Rₕ(Wₕ, D == 1 ? (x -> fs[k]((x,))) : fs[k]), D)
    return Ωₕ, Wₕ, u, npoints(Ωₕ, Tuple)
end

# The same field, spelled as a `D`-leaf composite grid function.
function _composite(Ωₕ, u, D)
    uc = element(gridspace(Ωₕ, Val(D)), 0.0)
    for d in 1:D
        parent(components(uc)[d]) .= parent(u[d])
    end
    return uc
end

function _bubble(u, dims)
    w = copy(u)
    _zero_boundary!(reshape(parent(w), dims))
    return w
end

_field(u, D) = D == 1 ? u[1] : u

@testset "Starred vector calculus (#287)" begin
    Random.seed!(287)

    @testset "∇̽ₕ ($(D)D)" for D in 1:3
        _, _, u, _ = _fixture(D)
        g = ∇̽ₕ(u[1])
        dest = D == 1 ? similar(u[1]) : ntuple(_ -> similar(u[1]), D)
        ∇̽ₕ!(dest, u[1])
        if D == 1
            @test parent(g) == parent(D̽(u[1], 1))
            @test parent(dest) == parent(g)
        else
            @test all(parent(g[d]) == parent(D̽(u[1], d)) for d in 1:D)
            @test all(parent(dest[d]) == parent(g[d]) for d in 1:D)
        end
        @test_allocs ∇̽ₕ!(dest, u[1])
    end

    @testset "div̽ₕ ($(D)D)" for D in 1:3
        Ωₕ, _, u, _ = _fixture(D)
        oracle = sum(parent(D̽(u[d], d)) for d in 1:D)
        F = _field(u, D)
        @test parent(div̽ₕ(F)) ≈ oracle
        v = similar(u[1])
        div̽ₕ!(v, F)
        @test parent(v) ≈ oracle
        @test_allocs div̽ₕ!(v, F)
        if D > 1
            uc = _composite(Ωₕ, u, D)
            @test parent(div̽ₕ(uc)) ≈ oracle
            @test_allocs div̽ₕ!(v, uc)
        end
        @test_throws DimensionMismatch div̽ₕ(ntuple(_ -> u[1], D + 1))
    end

    @testset "curl̽ₕ ($(D)D)" for D in 1:3
        Ωₕ, _, u, _ = _fixture(D)
        ds(k, d) = parent(D̽(u[k], d))
        if D == 1
            @test_throws ArgumentError curl̽ₕ(u)
        elseif D == 2
            oracle = ds(2, 1) .- ds(1, 2)
            @test parent(curl̽ₕ(u)) ≈ oracle
            @test parent(curl̽ₕ(_composite(Ωₕ, u, D))) ≈ oracle
            c = similar(u[1])
            curl̽ₕ!(c, u)
            @test parent(c) ≈ oracle
            @test_allocs curl̽ₕ!(c, u)
        else
            oracle = (ds(3, 2) .- ds(2, 3), ds(1, 3) .- ds(3, 1), ds(2, 1) .- ds(1, 2))
            cc = curl̽ₕ(u)
            @test all(parent(cc[k]) ≈ oracle[k] for k in 1:3)
            c = ntuple(_ -> similar(u[1]), 3)
            curl̽ₕ!(c, _composite(Ωₕ, u, D))
            @test all(parent(c[k]) ≈ oracle[k] for k in 1:3)
            @test_allocs curl̽ₕ!(c, u)
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

    # Summation by parts in 1D: innerₕ(div̽ₕ(u), v) = -inner₊(u, D₋ₓ(v)) when the fields
    # vanish on the boundary, on a non-uniform mesh. Without that the boundary term is left
    # over, so the identity must fail.
    @testset "SBP duality (1D)" begin
        _, _, u, dims = _fixture(1)
        w = Rₕ(space(u[1]), x -> cos(5x) + 0.4)
        ub, wb = _bubble(u[1], dims), _bubble(w, dims)
        @test isapprox(innerₕ(div̽ₕ(ub), wb), -inner₊(ub, D₋ₓ(wb)); atol = 1e-12)
        @test !isapprox(innerₕ(div̽ₕ(u[1]), w), -inner₊(u[1], D₋ₓ(w)); atol = 1e-6)
    end
end

end # module SpaceStarVectorCalculusTests
