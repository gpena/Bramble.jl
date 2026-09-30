# Standalone entry point: `include("test/solvers/runtests.jl")` alone in a fresh session
# runs this subsystem's tests without going through the full test/runtests.jl.
isdefined(Main, :TestUtils) || include(joinpath(@__DIR__, "..", "TestUtils.jl"))

@testset "Solvers" begin
    include("sparse_solvers.jl")
    # Preconditioners from a matrix-free diagonal (gpena/Bramble.jl#327).
    include("preconditioners.jl")
    # Geometric multigrid: nested hierarchies (gpena/Bramble.jl#329).
    include("multigrid.jl")
end
