# Standalone entry point: `include("test/geometry/runtests.jl")` alone in a fresh session
# runs this subsystem's tests without going through the full test/runtests.jl.
isdefined(Main, :TestUtils) || include(joinpath(@__DIR__, "..", "TestUtils.jl"))

@testset "Sets and Domains" begin
    include("sets.jl")
    include("domains.jl")
end
