module FormCenteredAverageTests

using Test
using Bramble
using Random
import Bramble: Mcₓ, Mcᵧ, Mc₂, Mcₕ, CenteredAverage, D₋ₓ, TrialFunction

# A random non-uniform mesh on the unit cube of dimension `D`.
function _nonuniform_space(D)
    dom = D == 1 ? domain(interval(0.0, 1.0)) :
          D == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 1.0)) :
          domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    n = (9, 8, 7)
    Ωₕ = D == 1 ? mesh(dom, n[1], false) : mesh(dom, n[1:D], ntuple(_ -> false, D))
    return gridspace(Ωₕ)
end

_random_element(Wₕ) = (uₕ = element(Wₕ); parent(uₕ) .= randn(length(parent(uₕ))); uₕ)

@testset "Centered average in forms (#287)" begin
    Random.seed!(287)
    dirs = (Mcₓ, Mcᵧ, Mc₂)

    @testset "node construction" begin
        u = TrialFunction{2}()
        @test Mcₓ(u) isa CenteredAverage{2, 1}
        @test Mcᵧ(u) isa CenteredAverage{2, 2}
        @test Mcₕ(u) === (Mcₕ(u, Val(1)), Mcₕ(u, Val(2)))
        @test Mcₕ(u, 2) === Mcᵧ(u)
        @test !(Mcₕ(TrialFunction{1}()) isa Tuple)
    end

    for D in 1:3
        Wₕ = _nonuniform_space(D)
        uₕ, vₕ = _random_element(Wₕ), _random_element(Wₕ)
        for d in 1:D
            @testset "$(D)D, direction $d" begin
                A = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(dirs[d](u), v)))
                @test parent(vₕ)' * (A * parent(uₕ)) ≈ innerₕ(dirs[d](uₕ), vₕ)
                B = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(Mcₕ(u, d), v)))
                @test B ≈ A
                # The average on the test side.
                C = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, dirs[d](v))))
                @test parent(vₕ)' * (C * parent(uₕ)) ≈ innerₕ(uₕ, dirs[d](vₕ))
            end
        end
        # A composed tap re-evaluates its inner stencil at the neighbour, so the composition
        # matches the runtime on a non-uniform mesh too.
        @testset "$(D)D composition with D₋ₓ" begin
            A = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(Mcₓ(D₋ₓ(u)), v)))
            @test parent(vₕ)' * (A * parent(uₕ)) ≈ innerₕ(Mcₓ(D₋ₓ(uₕ)), vₕ)
        end
    end

    @testset "expression" begin
        Wₕ = _nonuniform_space(2)
        f = form(Wₕ, Wₕ, (u, v) -> innerₕ(Mcᵧ(D₋ₓ(u)), v))
        @test expression(f) == "innerₕ(Mcᵧ(D₋ₓ(u)), v)"
    end
end

end # module FormCenteredAverageTests
