module GhostContractTests

using Test
using Bramble
import Bramble
using ForwardDiff: Dual

isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
const O = Main.GhostOracleND

@testset "Ghost contraction refusals" begin
    Ω = O.make_mesh(:random, 0)
    Wg = gridspace(Ω; ghosts = (:xmin, :ymax))
    l = form(Wg, v -> innerₕ(x -> 1.0, v))
    vₕ = element(Bramble.padded_space(Wg), 1.0)
    @test_throws ArgumentError l(vₕ)
    @test_throws ArgumentError Bramble.evaluate!(zeros(length(parent(vₕ))), l, vₕ)
end

end
