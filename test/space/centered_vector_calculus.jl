module SpaceCenteredVectorCalculusTests

using Test
using Bramble
using Random
using Bramble: components
using ..TestUtils: alloc_test, _zero_boundary!

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

@testset "Centered vector calculus (#287)" begin
    Random.seed!(287)

    @testset "∇cₕ ($(D)D)" for D in 1:3
        _, _, u, _ = _fixture(D)
        g = ∇cₕ(u[1])
        dest = D == 1 ? similar(u[1]) : ntuple(_ -> similar(u[1]), D)
        ∇cₕ!(dest, u[1])
        if D == 1
            @test parent(g) == parent(Dc(u[1], 1))
            @test parent(dest) == parent(g)
        else
            @test all(parent(g[d]) == parent(Dc(u[1], d)) for d in 1:D)
            @test all(parent(dest[d]) == parent(g[d]) for d in 1:D)
        end
        @test alloc_test(∇cₕ!, dest, u[1]) == 0
    end

    @testset "divcₕ ($(D)D)" for D in 1:3
        Ωₕ, _, u, _ = _fixture(D)
        oracle = sum(parent(Dc(u[d], d)) for d in 1:D)
        F = _field(u, D)
        @test parent(divcₕ(F)) ≈ oracle
        v = similar(u[1])
        divcₕ!(v, F)
        @test parent(v) ≈ oracle
        @test alloc_test(divcₕ!, v, F) == 0
        if D > 1
            uc = _composite(Ωₕ, u, D)
            @test parent(divcₕ(uc)) ≈ oracle
            @test alloc_test(divcₕ!, v, uc) == 0
        end
        @test_throws DimensionMismatch divcₕ(ntuple(_ -> u[1], D + 1))
    end

    @testset "curlcₕ ($(D)D)" for D in 1:3
        Ωₕ, _, u, _ = _fixture(D)
        dc(k, d) = parent(Dc(u[k], d))
        if D == 1
            @test_throws ArgumentError curlcₕ(u)
        elseif D == 2
            oracle = dc(2, 1) .- dc(1, 2)
            @test parent(curlcₕ(u)) ≈ oracle
            @test parent(curlcₕ(_composite(Ωₕ, u, D))) ≈ oracle
            c = similar(u[1])
            curlcₕ!(c, u)
            @test parent(c) ≈ oracle
            @test alloc_test(curlcₕ!, c, u) == 0
        else
            oracle = (dc(3, 2) .- dc(2, 3), dc(1, 3) .- dc(3, 1), dc(2, 1) .- dc(1, 2))
            cc = curlcₕ(u)
            @test all(parent(cc[k]) ≈ oracle[k] for k in 1:3)
            c = ntuple(_ -> similar(u[1]), 3)
            curlcₕ!(c, _composite(Ωₕ, u, D))
            @test all(parent(c[k]) ≈ oracle[k] for k in 1:3)
            @test alloc_test(curlcₕ!, c, u) == 0
        end
    end

    @testset "εcₕ ($(D)D)" for D in 1:3
        Ωₕ, _, u, _ = _fixture(D)
        F = _field(u, D)
        ε = εcₕ(F)
        for i in 1:D, j in 1:D

            oracle = i == j ? parent(Dc(u[i], i)) :
                     (parent(Dc(u[i], j)) .+ parent(Dc(u[j], i))) ./ 2
            @test parent(ε[i][j]) ≈ oracle
            @test parent(ε[i][j]) == parent(ε[j][i])
        end
        dest = ntuple(_ -> ntuple(_ -> similar(u[1]), D), D)
        εcₕ!(dest, F)
        @test all(parent(dest[i][j]) == parent(ε[i][j]) for i in 1:D, j in 1:D)
        @test alloc_test(εcₕ!, dest, F) == 0
        if D > 1
            εc = εcₕ(_composite(Ωₕ, u, D))
            @test all(parent(εc[i][j]) == parent(ε[i][j]) for i in 1:D, j in 1:D)
        end
    end

    # Summation by parts: for F and u vanishing on the boundary, divcₕ is minus the adjoint
    # of the centered gradient, exactly, on a non-uniform mesh. Non-vanishing fields pick up
    # boundary terms, so the identity must fail for them.
    @testset "SBP duality ($(D)D)" for D in 1:3
        _, _, u, dims = _fixture(D)
        duality(F, w) = (innerₕ(divcₕ(_field(F, D)), w),
            -sum(innerₕ(F[d], Dc(w, d)) for d in 1:D))

        Fb = ntuple(d -> _bubble(u[d], dims), D)
        wb = _bubble(u[D], dims)
        lhs, rhs = duality(Fb, wb)
        @test isapprox(lhs, rhs; atol = 1e-12)

        lhs, rhs = duality(u, u[D])
        @test !isapprox(lhs, rhs; atol = 1e-6)
    end
end

end # module SpaceCenteredVectorCalculusTests
