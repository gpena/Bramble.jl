# Standalone entry point: `include("test/utils/runtests.jl")` alone in a fresh session
# runs this subsystem's tests without going through the full test/runtests.jl.
isdefined(Main, :TestUtils) || include(joinpath(@__DIR__, "..", "TestUtils.jl"))

@testset "Utilities" begin
    include("macros.jl")
    include("backends.jl")
    include("linear_algebra.jl")
end
