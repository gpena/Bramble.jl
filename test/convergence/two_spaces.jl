module ConvergenceTwoSpacesTests

# The observed order of the problems in test/form/TwoSpacesCases.jl, over the plain space
# with row Dirichlet and over the ghost space of the same problem (gpena/Bramble.jl#786).
# Moved here from test/form/two_spaces.jl, which checks that the two systems agree.

using Test
using Bramble
import ..TestUtils

isdefined(Main, :TwoSpacesCases) ||
    Base.include(Main, joinpath(@__DIR__, "..", "form", "TwoSpacesCases.jl"))
using Main.TwoSpacesCases: PROBLEMS, manufactured, mesh_axes, node_solution, problem_data,
                           random_mesh

@testset "Observed order: $(prob.name)" for prob in PROBLEMS
    ex = manufactured(prob.D)
    levels = prob.D == 1 ? (0:3) : (1:3)
    for which in 1:2
        eoc = TestUtils._check_eoc(levels; order = 1.8) do level
            Ω = random_mesh(prob.D, prob.n0 - (prob.D == 2), level)
            c = TestUtils.plain_and_ghost_spaces(Ω,
                problem_data(prob.D; mixed = prob.mixed))[which]
            axs = mesh_axes(Ω)
            pts = vec([ntuple(d -> axs[d][I[d]], prob.D) for I in CartesianIndices(c.dims)])
            err = maximum(abs, node_solution(c) .- ex.u.(pts))
            return err, maximum(maximum(diff(collect(x))) for x in axs)
        end
        @test length(eoc) == length(levels) - 1
    end
end

end # module ConvergenceTwoSpacesTests
