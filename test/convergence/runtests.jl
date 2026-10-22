# Standalone entry point: `include("test/convergence/runtests.jl")` alone in a fresh session
# runs this subsystem's tests without going through the full test/runtests.jl.
isdefined(Main, :TestUtils) || include(joinpath(@__DIR__, "..", "TestUtils.jl"))

# Every convergence study of the suite: a sequence of refined meshes or time steps with the
# observed order, or the fall of the error, asserted. The `convergence` group, and `all` and
# `full`, run them (test/runtests.jl); no other group does.
@testset verbose=true "Convergence" begin
    include("operators.jl")
    include("space_operators.jl")
    include("variable_coefficient_poisson.jl")
    include("dirac.jl")
    include("reaction_flux.jl")
    include("two_spaces.jl")
    include("heat_1d.jl")
    include("ghost_oracles.jl")
    include("ghost_relations.jl")
    include("ghost_composite.jl")
    include("ghost_traction.jl")
    include("ghost_heat.jl")
    # poisson_polygon.jl defines its names at top level (the probe reads `polygon_errors`
    # from the module it is included into), so it gets a module of its own here.
    let m = Module(:ConvergencePolygonTests)
        Core.eval(m, :(using Test))
        Base.include(m, joinpath(@__DIR__, "poisson_polygon.jl"))
    end
end
