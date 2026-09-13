# Standalone entry point: `include("test/exporters/runtests.jl")` alone in a fresh session
# runs this subsystem's tests without going through the full test/runtests.jl.
isdefined(Main, :TestUtils) || include(joinpath(@__DIR__, "..", "TestUtils.jl"))

@testset "Exporters" begin
    include("vtk_export.jl")
    include("pgfplots_export.jl")
end
