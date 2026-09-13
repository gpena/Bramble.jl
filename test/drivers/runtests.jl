# Standalone entry point: `include("test/drivers/runtests.jl")` alone in a fresh session
# runs this subsystem's tests without going through the full test/runtests.jl.
isdefined(Main, :TestUtils) || include(joinpath(@__DIR__, "..", "TestUtils.jl"))

@testset "Drivers" begin
    include("variable_coefficient_poisson.jl")
end
