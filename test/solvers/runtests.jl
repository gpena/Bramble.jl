# Standalone entry point: `include("test/solvers/runtests.jl")` alone in a fresh session
# runs this subsystem's tests without going through the full test/runtests.jl.
isdefined(Main, :TestUtils) || include(joinpath(@__DIR__, "..", "TestUtils.jl"))

@testset "Solvers" begin
    include("sparse_solvers.jl")
    # Preconditioners from a matrix-free diagonal.
    include("preconditioners.jl")
    # Geometric multigrid: nested hierarchies.
    include("multigrid.jl")
    # The mean-zero constraint on a singular (pure Neumann) solve.
    include("mean_zero.jl")
    # The ghost-only kernel, the gauge solve and the Neumann compatibility check (v4.0.0).
    include("ghost_kernel.jl")
    include("gauge_solve.jl")
    include("neumann_compatibility.jl")
    include("rank_one_solve.jl")
end
