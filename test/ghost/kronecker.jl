module GhostKroneckerTests

using Test
using Bramble
import Bramble
using ForwardDiff: Dual

isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
const O = Main.GhostOracleND

@testset "Ghost kronecker_operator refusal" begin
    Ω = O.make_mesh(:random, 0)
    Wg = gridspace(Ω; ghosts = (:xmin, :ymax))
    a = form(Wg, Wg, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    @test_throws ArgumentError kronecker_operator(a)
end

end
