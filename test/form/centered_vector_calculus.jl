module FormCenteredVectorCalculusTests

using Test
using Bramble
using Random

function _setup(D)
    Random.seed!(287)
    dom = D == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 1.0)) :
          domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    Ωₕ = mesh(dom, (9, 8, 7)[1:D], ntuple(_ -> false, D))
    Wₕ = gridspace(Ωₕ)
    Vₕ = gridspace(Ωₕ, Val(D))
    p = element(Wₕ, 0.0)
    parent(p) .= rand(length(parent(p)))
    rnd() = (w = element(Vₕ, 0.0); parent(w) .= rand(length(parent(w))); w)
    return Wₕ, Vₕ, p, rnd(), rnd()
end

@testset "Centered vector calculus in forms (#287)" begin
    for D in 2:3
        Wₕ, Vₕ, p, uₕ, vₕ = _setup(D)

        A = assemble(form(Wₕ, Vₕ, (q, v) -> innerₕ(q, divcₕ(v))))
        @test size(A) == (length(parent(vₕ)), length(parent(p)))
        @test parent(vₕ)' * (A * parent(p)) ≈ innerₕ(p, divcₕ(vₕ))
        A1 = assemble(form(Wₕ, Vₕ, (q, v) -> innerₕ(q(1), divcₕ(v))))
        @test A1 == A

        B = assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(εcₕ(u), εcₕ(v))))
        εu, εv = εcₕ(uₕ), εcₕ(vₕ)
        @test parent(vₕ)' * (B * parent(uₕ)) ≈
              sum(innerₕ(εu[i][j], εv[i][j]) for i in 1:D, j in 1:D)

        # An unnamed side on a genuine composite is still refused.
        @test_throws ArgumentError assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(u, v(1))))
    end
end

# The cross-weighted family (gpena/Bramble.jl#349): the same builders with D̽ in place of Dc.
# The two differ on a non-uniform mesh and at the ends (D̽ is one-sided there, Dc zero), so
# each form is checked against its own runtime operator and against the centered one.
@testset "Cross-weighted vector calculus in forms (#349)" begin
    for D in 2:3
        Wₕ, Vₕ, p, uₕ, vₕ = _setup(D)

        A = assemble(form(Wₕ, Vₕ, (q, v) -> innerₕ(q, div̽ₕ(v))))
        @test size(A) == (length(parent(vₕ)), length(parent(p)))
        @test parent(vₕ)' * (A * parent(p)) ≈ innerₕ(p, div̽ₕ(vₕ))
        @test !(A ≈ assemble(form(Wₕ, Vₕ, (q, v) -> innerₕ(q, divcₕ(v)))))
        # the trial side: the transpose pairing
        At = assemble(form(Vₕ, Wₕ, (u, q) -> innerₕ(div̽ₕ(u), q)))
        @test parent(p)' * (At * parent(uₕ)) ≈ innerₕ(div̽ₕ(uₕ), p)

        B = assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(ε̽ₕ(u), ε̽ₕ(v))))
        εu, εv = ε̽ₕ(uₕ), ε̽ₕ(vₕ)
        @test parent(vₕ)' * (B * parent(uₕ)) ≈
              sum(innerₕ(εu[i][j], εv[i][j]) for i in 1:D, j in 1:D)
        @test B ≈ B'
        @test !(B ≈ assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(εcₕ(u), εcₕ(v)))))
    end
end

# The strain forms scale their shear pieces by one half: that scale must not promote a
# Float32 form to Float64. The Float64 twin reuses the Float32 mesh's points, so the two
# matrices describe the same grid and differ only by rounding.
@testset "Strain forms keep the element type (#349)" begin
    Random.seed!(349)
    Ω32 = mesh(domain(box((0.0f0, 0.0f0, 0.0f0), (1.0f0, 1.0f0, 1.0f0))), (6, 5, 7),
        (false, false, false))
    Ω64 = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (6, 5, 7), (true, true, true))
    for k in 1:3
        Bramble.set_points!(Ω64(k), Float64.(points(Ω32)[k]))
    end
    V32, V64 = gridspace(Ω32, Val(3)), gridspace(Ω64, Val(3))
    W32, W64 = gridspace(Ω32), gridspace(Ω64)
    for ε in (εcₕ, ε̽ₕ)
        B32 = assemble(form(V32, V32, (u, v) -> innerₕ(ε(u), ε(v))))
        B64 = assemble(form(V64, V64, (u, v) -> innerₕ(ε(u), ε(v))))
        @test eltype(B32) === Float32
        @test eltype(B64) === Float64
        @test isapprox(B32, B64; rtol = 10 * eps(Float32))
    end
    for div in (divcₕ, div̽ₕ)
        A32 = assemble(form(W32, V32, (q, v) -> innerₕ(q, div(v))))
        @test eltype(A32) === Float32
        @test isapprox(A32, assemble(form(W64, V64, (q, v) -> innerₕ(q, div(v)))); rtol = 10 * eps(Float32))
    end
end

end # module FormCenteredVectorCalculusTests
