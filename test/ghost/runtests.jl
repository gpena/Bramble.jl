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
    include("cross_component.jl")
    include("traction.jl")
    include("contract.jl")
    include("grid_functions.jl")
    include("preconditioners.jl")
    include("dirac.jl")
    include("polygon_spaces.jl")
    include("polygon_oracle.jl")
    include("levelset_spaces.jl")
    include("levelset_oracle.jl")
    include("polygon_support.jl")
end
