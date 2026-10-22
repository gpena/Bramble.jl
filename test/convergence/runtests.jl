# Standalone entry point: `include("test/convergence/runtests.jl")` alone in a fresh session
# runs this subsystem's tests without going through the full test/runtests.jl.
isdefined(Main, :TestUtils) || include(joinpath(@__DIR__, "..", "TestUtils.jl"))

@testset "Convergence" begin
    include("operators.jl")
    # poisson_polygon.jl defines its names at top level (the probe reads `polygon_errors`
    # from the module it is included into), so it gets a module of its own here.
    let m = Module(:ConvergencePolygonTests)
        Core.eval(m, :(using Test))
        Base.include(m, joinpath(@__DIR__, "poisson_polygon.jl"))
    end
end
