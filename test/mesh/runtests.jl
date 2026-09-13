# Standalone entry point: `include("test/mesh/runtests.jl")` alone in a fresh session
# runs this subsystem's tests without going through the full test/runtests.jl.
isdefined(Main, :TestUtils) || include(joinpath(@__DIR__, "..", "TestUtils.jl"))

@testset "Meshes" begin
    include("constructors.jl")
    include("mesh1d.jl")
    include("meshnd.jl")
    include("meshes.jl")
    include("markers.jl")
    include("inference_allocation.jl")
end
