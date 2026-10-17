module GhostCompositeTests

using Test
using Bramble
import Bramble
using ForwardDiff: Dual

isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
const O = Main.GhostOracleND

@testset "Ghost composite refusals" begin
    Ω = O.make_mesh(:random, 0)
    Wg = gridspace(Ω; ghosts = (:xmin, :ymax))
    @test_throws ArgumentError form(Bramble.CompositeGridSpace(Wg, Wg),
        Bramble.CompositeGridSpace(Wg, Wg), (u, v) -> innerₕ(u(1), v(1)))
    @test_throws ArgumentError form(Bramble.CompositeGridSpace(Wg, Wg),
        v -> innerₕ(x -> 1.0, v(1)))
end

end
