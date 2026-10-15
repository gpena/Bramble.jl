# Standalone entry point: `include("test/ghost/runtests.jl")` alone in a fresh session
# runs this subsystem's tests without going through the full test/runtests.jl.
isdefined(Main, :TestUtils) || include(joinpath(@__DIR__, "..", "TestUtils.jl"))

@testset "Ghost points" begin
    include("oracle_1d.jl")
    include("oracle_nd.jl")
    include("corners.jl")
end
