module ConvergenceGhostTractionTests

# H¹ convergence of linear elasticity with traction and displacement facets through ghost
# points (GhostTraction.jl, #793), each component on its own. Moved here from
# test/ghost/traction.jl, which checks the reduced system.
#
# A random mesh is refined, never redrawn, across levels (`O.make_mesh(:random, ℓ)`), and
# carries no order gate: on these meshes the collocated form reaches only 1.77-1.84 over
# levels 0-3 with displacement on every facet, no traction at all (`n0 = 9`, seeds
# 20261014 and 7). It is held instead to the error of that all-displacement problem: the
# traction's within a factor 5 at every level.

using Test
using Bramble

isdefined(Main, :GhostTraction) ||
    Base.include(Main, joinpath(@__DIR__, "..", "ghost", "GhostTraction.jl"))
using Main.GhostTraction: ALL_DISPLACEMENT, CASE2, UEX2, UEX3, errors, lsq_order

# Graded meshes with a loaded traction facet, a traction-free one and two displacement
# facets, the two tractions meeting at a corner.
@testset "Order, graded meshes" begin
    levels = 0:3
    hs, errs = errors(UEX2, CASE2, :graded, levels; n0 = 9)
    @test maximum(hs) / minimum(hs) > 3
    for i in 1:2
        @test errs[end][i] < errs[1][i] / 8
        @test lsq_order(hs, getindex.(errs, i)) > 1.8
    end
end

# Random meshes: no order gate (see the header); the traction's error stays within 5×
# that of displacement on every facet, through ghosts, level by level.
@testset "Stability, random meshes, seed $seed" for seed in (20261014, 7)
    levels = 0:3
    hs, errs = errors(UEX2, CASE2, :random, levels; n0 = 9, seed)
    _, base = errors(UEX2, ALL_DISPLACEMENT, :random, levels; n0 = 9, seed)
    @test maximum(hs) / minimum(hs) > 3
    for ℓ in eachindex(levels), i in 1:2

        @test errs[ℓ][i] <= 5 * base[ℓ][i]
    end
    # the errors at least halve per level: a scheme stalled as the all-displacement
    # one is would pass the ratio alone
    for ℓ in 2:length(levels), i in 1:2

        @test errs[ℓ][i] < errs[ℓ - 1][i] / 2
    end
end

# Traction on `:xmin`, displacement on the other five.
@testset "3D order, graded meshes" begin
    case = (tfacets = (:xmin,), dfacets = (:xmax, :ymin, :ymax, :zmin, :zmax), free = ())
    # the order on graded meshes of 9 and 17 points per axis. The 33-point level costs
    # ~280 s on one thread and is left out: measured, the slope of u₃ is 1.84 from 9 to
    # 17 and 1.96 from 17 to 33 (u₁, u₂ above 1.98), and 1.68 from 5 to 17 points
    hs, errs = errors(UEX3, case, :graded, 0:1; n0 = 9)
    for i in 1:3
        @test lsq_order(hs, getindex.(errs, i)) > 1.8
    end
end

end # module ConvergenceGhostTractionTests
