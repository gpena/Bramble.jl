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

end # module FormCenteredVectorCalculusTests
