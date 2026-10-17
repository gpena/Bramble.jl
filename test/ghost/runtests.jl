# Standalone entry point: `include("test/ghost/runtests.jl")` alone in a fresh session
# runs this subsystem's tests without going through the full test/runtests.jl.
isdefined(Main, :TestUtils) || include(joinpath(@__DIR__, "..", "TestUtils.jl"))

@testset "Ghost points" begin
    include("oracle_1d.jl")
    include("oracle_nd.jl")
    include("corners.jl")
    include("backends.jl")
    include("policies.jl")
    include("allocations.jl")
    include("matrix_free.jl")
    include("kronecker.jl")
    include("multigrid.jl")
    include("composite.jl")
    include("contract.jl")
    include("grid_functions.jl")
    include("preconditioners.jl")
end
